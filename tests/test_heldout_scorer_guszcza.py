"""guszcza_growth_curve can score cells it was not trained on - checkable.

The newest density member of the milestone-6 board: a lognormal on cumulative
paid loss ratios around a growth-curve share of a per-accident-year ultimate
(Guszcza 2008 structure; likelihood and priors verbatim from Gesmann's brms
post). Layers mirror ``test_heldout_scorer_csr.py``:

1. **Closed form.** ``log_lik_cells`` against a hand-written scalar loop over
   ``model.stan``'s formula - with the growth curves spelled out LITERALLY as
   the post writes them, so the reuse of ``statistical/clark``'s ``growth()``
   has something independent to disagree with.
2. **The agreement gate** (``slow``): score the training cells through the
   held-out code path and reproduce the fit's own ``log_lik`` elementwise, for
   BOTH curves, with a perturbation negative control.
3. **The carries.** ``log_lik_at`` applies the ``- log premium`` loss-ratio
   carry; normalization on the amount scale is checked with a mutant that
   integrates to premium instead of 1. ``predict_at`` is a pass-through
   (cumulative draws on a cumulative triangle) and threads its seed.

Plus the entry-level contracts this branch introduces: backend and per-backend
control validation, curve validation, and ATOMIC ``fit()`` - a failed refit must
not leave the entry torn between an old posterior and a new contract. Backend
validation was Stan-only until the milestone-5 ports landed; cross-backend
parity itself lives in ``tests/test_parity_guszcza.py``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr.gallery.bayesian.guszcza_growth_curve import scorer
from ibnr.gallery.bayesian.guszcza_growth_curve.model import (
    BACKENDS,
    CURVE_CODES,
    GuszczaGrowthCurve,
)
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout
from ibnr.gallery.statistical.clark.model import GROWTH_CURVES, growth
from ibnr.kernels.contract import stan_data
from ibnr.kernels.densities import check_normalization, normal_lpdf
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal, training_index
from ibnr.kernels.rng import heldout_stream
from ibnr.triangle.core import Triangle

N_W = N_D = 6
PREMIUM = 1000.0
#: ground-truth curve parameters for the synthetic cohort (loglogistic,
#: half of ultimate at ~2.5 years, ULR rising slowly in w)
OMEGA_TRUE, THETA_TRUE = 1.8, 2.5


def _triangle(*, through: int, noise: float = 0.0, seed: int = 0) -> Triangle:
    """The fitted cohort ``FIT_CO``: paid loss on a loglogistic emergence
    pattern, with premium. It carries a segment column deliberately - an
    unlabelled triangle is not what the mart looks like.

    ``noise`` multiplies each level by ``exp(N(0, noise))`` - used by the slow
    Stan gate, where noise-free curves would let the sampler chase sigma
    toward zero.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # origins run to `through`, not N_W: the post-cutoff triangle carries a
    # brand-new origin, which is what gives the new_origin exclusion real work
    for w in range(1, through + 1):
        for d in range(1, N_D + 1):
            if w + d - 1 > through:
                continue
            t = float(d)  # annual grain: t in years == d
            ulr_w = 0.60 * (1.0 + 0.04 * w)
            g = t**OMEGA_TRUE / (t**OMEGA_TRUE + THETA_TRUE**OMEGA_TRUE)
            cum = PREMIUM * ulr_w * g
            if noise:
                cum *= float(np.exp(rng.normal(0.0, noise)))
            for field, value in (("paid_loss", cum), ("earned_premium", PREMIUM)):
                rows.append(
                    {
                        "lob": "FIT_CO",
                        "origin_period": dt.date(2010 + w - 1, 1, 1),
                        "dev_lag": 12 * d,
                        "eval_date": dt.date(2010 + w - 1 + d - 1, 12, 31),
                        "field": field,
                        "value": float(value),
                    }
                )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


@pytest.fixture(scope="module")
def contract() -> dict:
    return stan_data(_triangle(through=N_W), loss_field="paid_loss", premium_field="earned_premium")


@pytest.fixture(scope="module")
def heldout(contract):
    """The diagonal after the training cutoff. The source triangle carries a
    7th origin too, so the ``new_origin`` exclusion has real work to do."""
    return next_diagonal(
        _triangle(through=N_W + 1),
        as_of=dt.date(2010 + N_W - 1, 12, 31),
        fields="paid_loss",
        premium_field="earned_premium",
    )


def fake_posterior(contract: dict, n_draws: int = 7, seed: int = 0) -> dict:
    """Draws with DISTINCT, non-symmetric values per index (the CSR-fixture
    discipline): ``ulr`` rises in w while ``t`` rises in d, and omega/theta/
    sigma differ per draw, so a transposed or off-by-one index lands on a
    visibly wrong number rather than a plausible neighbour."""
    rng = np.random.default_rng(seed)
    n_w = contract["n_w"]
    return {
        "ulr": np.tile(np.linspace(0.55, 0.85, n_w), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_w)),
        "omega": rng.uniform(1.5, 2.5, n_draws),
        "theta": rng.uniform(2.0, 4.0, n_draws),
        "sigma": rng.uniform(0.1, 0.3, n_draws),
    }


def _subset(cells: CellIndex, rows, value: np.ndarray | None = None) -> CellIndex:
    """A CellIndex over ``rows`` of ``cells``, optionally with the observed
    values replaced (for the normalization sweeps)."""
    rows = np.asarray(rows, dtype=int)
    return CellIndex(
        w=cells.w[rows],
        d=cells.d[rows],
        value=cells.value[rows] if value is None else np.asarray(value, dtype=float),
        prev_value=cells.prev_value[rows],
        premium=cells.premium[rows],
    )


class _StubGuszcza(GuszczaGrowthCurve):
    """A fitted entry without a sampler: the scorer only needs contract +
    draws + the fitted curve."""

    def __init__(self, contract, post, curve="loglogistic"):
        super().__init__()
        self.contract_ = contract
        self.curve_ = curve
        self._post = post

    def _posterior(self):
        return self._post

    def _log_lik_native(self, cells):
        return scorer.log_lik_cells(self.contract_, self._post, cells, curve=self.curve_)

    def _draws_native(self, cells, *, rng):
        return scorer.draw_cells(self.contract_, self._post, cells, rng=rng, curve=self.curve_)


# -- layer 1: closed form -----------------------------------------------------


@pytest.mark.parametrize("curve", ["loglogistic", "weibull"])
def test_matches_a_scalar_loop_over_the_stan_formula(contract, curve):
    """``mu[i] = log(ulr[w[i]] * G(t[i]))`` and
    ``log_lik[i] = lognormal_lpdf(y[i] | mu[i], sigma)``, written out one cell
    at a time with the growth curve spelled LITERALLY as the post writes it -
    independent of the ``growth()`` the scorer reuses."""
    post = fake_posterior(contract)
    cells = training_index(contract)
    got = scorer.log_lik_cells(contract, post, cells, curve=curve)

    prem = np.asarray(contract["premium"], dtype=float)
    step_years = contract["dev_grain_months"] / 12.0
    expected = np.empty((len(post["sigma"]), cells.n_cells))
    for s in range(expected.shape[0]):
        om, th, sig = post["omega"][s], post["theta"][s], post["sigma"][s]
        for i in range(cells.n_cells):
            w0, d = int(cells.w[i]) - 1, int(cells.d[i])
            t = d * step_years
            if curve == "loglogistic":
                g = t**om / (t**om + th**om)
            else:
                g = 1.0 - np.exp(-((t / th) ** om))
            mu = np.log(post["ulr"][s, w0] * g)
            ratio = cells.value[i] / prem[w0]
            expected[s, i] = normal_lpdf(np.log(ratio), mu, sig) - np.log(ratio)

    assert got.shape == expected.shape
    assert np.allclose(got, expected)


@pytest.mark.parametrize("curve", ["loglogistic", "weibull"])
def test_growth_curve_equals_the_clark_entrys_growth(curve):
    """The entry reuses ``statistical/clark``'s ``growth()`` rather than
    re-deriving the curves; this pins that the shared function IS the post's
    algebra, on a grid wide enough to catch a swapped curve or a transposed
    (omega, theta)."""
    for t in (0.5, 1.0, 2.5, 7.0):
        for om in (0.7, 1.8, 3.0):
            for th in (1.5, 2.5, 4.0):
                if curve == "loglogistic":
                    literal = t**om / (t**om + th**om)
                else:
                    literal = 1.0 - np.exp(-((t / th) ** om))
                assert np.isclose(growth(t, om, th, curve), literal, rtol=1e-12)
    # and the two curves genuinely differ where both are evaluated, or the
    # parametrization above cannot tell them apart
    assert not np.isclose(growth(2.5, 1.8, 2.5, "loglogistic"), growth(2.5, 1.8, 2.5, "weibull"))


def test_indices_are_not_interchangeable(contract):
    """A guard on the guard: if swapping w for d changed nothing, the
    closed-form test would be vacuous. ``ulr`` varies in w and ``t`` in d, so
    it has to matter."""
    post = fake_posterior(contract)
    cells = training_index(contract)
    swapped = CellIndex(
        w=cells.d, d=cells.w, value=cells.value, prev_value=cells.prev_value, premium=cells.premium
    )
    assert (cells.w != cells.d).any(), "fixture cannot distinguish w from d"

    a = scorer.log_lik_cells(contract, post, cells, curve="loglogistic")
    b = scorer.log_lik_cells(contract, post, swapped, curve="loglogistic")
    assert not np.allclose(a, b)


def test_off_by_one_index_arithmetic_is_pinned(contract):
    """One cell at a known position, against hand-picked draws: (w=2, d=3)
    must read ``ulr[:, 1]`` (0-based) and ``t = 3`` years - not the
    neighbouring origin's ulr and not ``t = 2``. The fake posterior is
    asymmetric in w, so each wrong read is visibly wrong."""
    post = fake_posterior(contract)
    prem = np.asarray(contract["premium"], dtype=float)
    one = CellIndex(
        w=np.array([2]),
        d=np.array([3]),
        value=np.array([400.0]),
        prev_value=np.array([300.0]),
        premium=np.array([prem[1]]),
    )
    got = scorer.mu_cells(contract, post, one, curve="loglogistic")

    t = 3.0 * contract["dev_grain_months"] / 12.0

    def mu_from(ulr_col: int, age: float) -> np.ndarray:
        g = growth(age, post["omega"], post["theta"], "loglogistic").reshape(-1, 1)
        return np.log(post["ulr"][:, [ulr_col]] * g)

    assert np.allclose(got, mu_from(1, t))
    assert not np.allclose(got, mu_from(2, t))  # off by one origin
    assert not np.allclose(got, mu_from(1, t - 1.0))  # off by one dev step


def test_missing_posterior_variable_names_what_is_needed(contract):
    post = fake_posterior(contract)
    del post["theta"]
    with pytest.raises(KeyError, match="theta"):
        scorer.log_lik_cells(contract, post, training_index(contract), curve="loglogistic")


def test_unknown_curve_is_refused(contract):
    with pytest.raises(ValueError, match="curve must be one of"):
        scorer.log_lik_cells(
            contract, fake_posterior(contract), training_index(contract), curve="gompertz"
        )


def test_nonpositive_loss_is_refused_not_nan(contract):
    """The lognormal has no density at a zero cell, and the mart has 316k of
    them - it has to be an error, the same rule the contract applies at fit
    time, not a silent ``-inf``."""
    cells = training_index(contract)
    zeroed = CellIndex(
        w=cells.w,
        d=cells.d,
        value=np.where(np.arange(cells.n_cells) == 0, 0.0, cells.value),
        prev_value=cells.prev_value,
        premium=cells.premium,
    )
    with pytest.raises(ValueError, match="non-positive loss"):
        scorer.log_lik_cells(contract, fake_posterior(contract), zeroed, curve="loglogistic")


def test_nonpositive_ulr_draws_are_refused(contract):
    """The sampler rejects any draw whose ulr[w] is non-positive for a trained
    origin, so such a posterior cannot be a fit of these cells - refusing
    beats returning NaN that pandas would then silently drop."""
    post = fake_posterior(contract)
    post["ulr"][2, 1] = -0.05
    with pytest.raises(ValueError, match="non-positive ulr"):
        scorer.mu_cells(contract, post, training_index(contract), curve="loglogistic")


def test_the_ulr_guard_covers_origins_outside_the_scored_cells(contract):
    """The guard checks the WHOLE posterior, not the columns a caller selects.

    The invariant it asserts is that the sampler rejected such draws for EVERY
    trained origin. A version that sliced first (``post["ulr"][:, w0]`` then
    tested) let a bad origin through silently whenever it was not among the
    scored cells - a narrower check than the message claimed, and exactly the
    case a single-cell held-out panel hits.
    """
    post = fake_posterior(contract)
    # break an origin that the scored cell does NOT index
    post["ulr"][1, contract["n_w"] - 1] = -0.02
    one = CellIndex(
        w=np.array([1]),
        d=np.array([2]),
        value=np.array([300.0]),
        prev_value=np.array([200.0]),
        premium=np.array([PREMIUM]),
    )
    with pytest.raises(ValueError, match="non-positive ulr"):
        scorer.mu_cells(contract, post, one, curve="loglogistic")


def test_predict_refuses_a_nonpositive_ulr_rather_than_emitting_nan(contract, monkeypatch):
    """``predict()`` and the scorer must not disagree about what is scorable.

    Before the guard was hoisted into one helper, ``predict()`` took
    ``log(ulr * G)`` unguarded: a single bad draw produced NaN ultimates that
    flowed into ``PredictiveDistribution``, ``evaluate()``'s summary /
    percentiles / CRPS, and any retro CSV, behind nothing louder than an
    "invalid value encountered in log" warning. Unreachable from Stan, which
    rejects such a proposal outright - but the ports reproduce that rejection as
    a ``-inf`` factor/Potential rather than by exception, so this is exactly the
    posterior a port would hand back if its safe substitution were dropped. See
    ``tests/test_parity_guszcza.py`` for the measurement.
    """
    post = fake_posterior(contract, n_draws=16)
    entry = _StubGuszcza(contract, post)
    entry.idata_ = object()  # only pooled() reads it, and that is patched below

    def fake_pooled(_idata, name):
        return post[name]

    monkeypatch.setattr("ibnr.gallery.bayesian.guszcza_growth_curve.model.pooled", fake_pooled)
    # a healthy posterior predicts fine, and every ultimate is finite
    healthy = entry.predict(seed=0)
    assert np.isfinite(healthy.samples).all()

    # break ONE draw at a NOT-fully-developed origin (origin 1 is anchored at
    # its observed value, so a break there would never reach the log)
    post["ulr"][3, 2] = -0.01
    with pytest.raises(ValueError, match="non-positive ulr"):
        entry.predict(seed=0)


def test_a_premium_disagreeing_with_the_contract_is_refused(contract):
    """The observed ratio divides by the CONTRACT's premium while the measure
    carry divides by the CELLS'; if the two differed, the carried density
    would silently stop integrating to 1."""
    cells = training_index(contract)
    doctored = CellIndex(
        w=cells.w,
        d=cells.d,
        value=cells.value,
        prev_value=cells.prev_value,
        premium=cells.premium * 1.1,
    )
    with pytest.raises(ValueError, match="disagrees with"):
        scorer.log_lik_cells(contract, fake_posterior(contract), doctored, curve="loglogistic")


# -- layer 3: the carries ------------------------------------------------------


def test_log_lik_at_carries_the_density_by_minus_log_premium(contract):
    """The entry reports a density on loss RATIOS; the leaderboard sums
    densities on amounts. The carry is exactly ``- log premium``, applied by
    the base class - the entry cannot skip it."""
    post = fake_posterior(contract)
    entry = _StubGuszcza(contract, post)
    cells = training_index(contract)

    native = entry._log_lik_native(cells)
    carried = entry.log_lik_at(cells)

    assert np.allclose(carried, native - np.log(cells.premium)[None, :])
    assert entry.heldout_measure == "loss_ratio"
    assert entry.heldout_draw_scale == "cumulative"
    assert isinstance(entry, ScoresHeldout)
    assert isinstance(entry, PredictsHeldout)


def test_held_out_cells_index_into_the_fitted_contract(contract, heldout):
    """The end-to-end shape: cells from ``next_diagonal`` are indexed against
    the fit and scored, one column per scorable cell, and the 7th origin was
    excluded rather than scored against a ulr the fit never estimated."""
    assert heldout.exclusion_counts()["new_origin"] == 1
    idx = index_into(heldout, contract, field="paid_loss")
    assert idx.n_cells == heldout.n_cells
    assert idx.w.min() >= 1 and idx.d.max() <= contract["n_d"]

    entry = _StubGuszcza(contract, fake_posterior(contract))
    out = entry.log_lik_at(heldout, field="paid_loss")
    assert out.shape == (7, heldout.n_cells)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("curve", ["loglogistic", "weibull"])
def test_density_is_normalized_on_the_amount_scale(contract, curve):
    """One deep-dev cell, one fixed posterior draw: the carried density must
    integrate to 1 over the AMOUNT - the observation space the data lives in.
    The un-carried mutant (the ratio density read as an amount density)
    integrates to premium, not 1: exactly the silent scale mixing the carry
    exists to prevent."""
    post = fake_posterior(contract, n_draws=1, seed=5)
    cells = training_index(contract)
    i = int(np.flatnonzero(cells.d == contract["n_d"])[0])
    entry = _StubGuszcza(contract, post, curve)

    mu = float(scorer.mu_cells(contract, post, _subset(cells, [i]), curve=curve)[0, 0])
    sig = float(post["sigma"][0])
    median = PREMIUM * np.exp(mu)  # the amount's median
    lo, hi = median * np.exp(-12 * sig), median * np.exp(12 * sig)

    def logpdf(xs):
        return entry.log_lik_at(_subset(cells, [i], value=np.asarray(xs, dtype=float)))[0]

    check_normalization(logpdf, lo=lo, hi=hi)

    def uncarried(xs):
        return entry._log_lik_native(_subset(cells, [i], value=np.asarray(xs, dtype=float)))[0]

    with pytest.raises(AssertionError, match="integrates to"):
        check_normalization(uncarried, lo=lo, hi=hi)


def test_draws_are_cumulative_amounts_with_mixture_moments(contract):
    """One draw per posterior draw: ``log(draw / premium)`` is the
    N(mu, sigma) mixture, checked by both moments (a plug-in at the posterior
    mean would miss the Var(mu) term), and the ratio draw is scaled by premium
    - raw ratio draws would be off by log(1000) in the mean, far away, not
    adjacent. Every cell has spread: no zero-variance anchoring leaks in from
    ``predict()``."""
    post = fake_posterior(contract, n_draws=4000)
    cells = training_index(contract)
    draws = scorer.draw_cells(
        contract, post, cells, rng=np.random.default_rng(17), curve="loglogistic"
    )
    assert (draws > 0).all()
    assert (draws.var(axis=0) > 0).all()

    prem = np.asarray(contract["premium"], dtype=float)[cells.w - 1]
    logratio = np.log(draws / prem[None, :])
    mu = scorer.mu_cells(contract, post, cells, curve="loglogistic")
    sig = scorer.sigma_cells(post, cells)

    assert np.allclose(logratio.mean(axis=0), mu.mean(axis=0), atol=0.03)
    want = (sig**2).mean(axis=0) + mu.var(axis=0)
    assert np.allclose(logratio.var(axis=0), want, rtol=0.15)
    assert (mu.var(axis=0) > 0).all()


def test_predict_at_is_a_pass_through_and_threads_its_seed(contract, heldout):
    """Cumulative draws on a cumulative triangle: ``predict_at`` must NOT add
    the training-diagonal anchor (the declared scale matches the triangle's
    basis), and the seed must reach the scorer's generator or two calls cannot
    be compared."""
    post = fake_posterior(contract, n_draws=64)
    entry = _StubGuszcza(contract, post)

    got = entry.predict_at(heldout, field="paid_loss", seed=7)
    np.testing.assert_array_equal(got, entry.predict_at(heldout, field="paid_loss", seed=7))
    assert not np.array_equal(got, entry.predict_at(heldout, field="paid_loss", seed=8))

    idx = index_into(heldout, contract, field="paid_loss")
    # the stream predict_at derives, rebuilt: a study-level seed no longer names
    # a generator directly (kernels.rng)
    raw = scorer.draw_cells(
        contract,
        post,
        idx,
        rng=np.random.default_rng(heldout_stream(7, heldout, field="paid_loss")),
        curve="loglogistic",
    )
    np.testing.assert_array_equal(got, raw)
    # the anchors are far from zero here, so an anchor wrongly added would be
    # a large shift - this equality has teeth
    assert (idx.prev_value > 100.0).all()


# -- entry-level contracts -----------------------------------------------------


def test_entry_is_registered_with_its_declarations():
    import ibnr.gallery as gallery

    assert "guszcza_growth_curve" in gallery.list()
    cls = gallery.get("guszcza_growth_curve")
    assert cls is GuszczaGrowthCurve
    assert cls.family == "bayesian"
    assert cls.heldout_measure == "loss_ratio"
    assert cls.heldout_draw_scale == "cumulative"
    # all three since the milestone-5 ports landed; the parity gate that makes
    # the two ports admissible is tests/test_parity_guszcza.py
    assert BACKENDS == ("stan", "numpyro", "pymc")
    assert cls.card()  # card.md ships with the entry

    # TWO validators police the curve argument - the entry's CURVE_CODES (which
    # maps to the Stan `curve` data value) and the scorer's GROWTH_CURVES
    # (which the shared growth() accepts). They must not drift: a curve in one
    # and not the other is either an entry that accepts a name Stan cannot fit,
    # or a scorer that refuses a curve the entry just fitted.
    assert set(CURVE_CODES) == set(GROWTH_CURVES) == {"loglogistic", "weibull"}


def test_fit_validates_before_touching_a_sampler():
    """Backend, curve and premium validation must fire without cmdstanpy
    installed - and without leaving any state behind.

    ``jags`` rather than ``numpyro`` for the unknown-backend case: numpyro is a
    real backend since the ports landed, so asking for it here would sample
    (and fail on a missing arviz in a core environment) instead of exercising
    the validator. The two port-only control validations are checked here too,
    for the same reason the rest of this test exists - they have to fire before
    anything is imported or compiled.
    """
    tri = _triangle(through=N_W)
    entry = GuszczaGrowthCurve()

    with pytest.raises(ValueError, match="backend must be one of"):
        entry.fit(tri, backend="jags")
    with pytest.raises(ValueError, match="pymc-backend control"):
        entry.fit(tri, backend="numpyro", nuts_sampler="numpyro")
    with pytest.raises(ValueError, match="stan-backend control"):
        entry.fit(tri, backend="numpyro", parallel_chains=4)
    with pytest.raises(ValueError, match="growth_curve must be one of"):
        entry.fit(tri, growth_curve="gompertz")
    with pytest.raises(ValueError, match="premium_field"):
        entry.fit(tri, premium_field=None)
    assert entry.contract_ is None and entry.idata_ is None


def test_fit_is_atomic_when_the_sampler_fails(monkeypatch):
    """A failed FIRST fit leaves no partial state behind."""
    tri = _triangle(through=N_W)
    entry = GuszczaGrowthCurve()

    def boom(*args, **kwargs):
        raise RuntimeError("sampler exploded")

    monkeypatch.setattr(entry, "_sample_stan", boom)
    with pytest.raises(RuntimeError, match="sampler exploded"):
        entry.fit(tri)

    assert entry.contract_ is None
    assert entry.stan_data_ is None
    assert entry.idata_ is None
    assert entry.curve_ is None
    assert entry.backend_ is None


def test_a_failed_REFIT_leaves_the_previous_fit_intact(monkeypatch):
    """The torn-refit-state defect proper, which the first-fit case cannot
    see: after a SUCCESSFUL fit, a failed re-fit on different data must leave
    the old contract AND the old posterior both intact and mutually
    consistent.

    The dangerous version is not "some state is None" - it is an entry holding
    the NEW cohort's contract against the OLD cohort's posterior, which indexes
    cleanly and scores a company the posterior never saw.
    """
    first = _triangle(through=N_W)
    entry = GuszczaGrowthCurve()

    sentinel_idata, sentinel_fit = object(), object()
    monkeypatch.setattr(entry, "_sample_stan", lambda *a, **k: (sentinel_idata, sentinel_fit))
    entry.fit(first, growth_curve="loglogistic")

    good_contract = entry.contract_
    good_stan_data = entry.stan_data_
    assert good_contract is not None and entry.idata_ is sentinel_idata

    # now a re-fit on a DIFFERENT triangle (one origin fewer) that blows up
    def boom(*args, **kwargs):
        raise RuntimeError("sampler exploded")

    monkeypatch.setattr(entry, "_sample_stan", boom)
    with pytest.raises(RuntimeError, match="sampler exploded"):
        entry.fit(_triangle(through=N_W - 1), growth_curve="weibull")

    # every piece of the previous fit survives, and they still agree
    assert entry.contract_ is good_contract
    assert entry.stan_data_ is good_stan_data
    assert entry.idata_ is sentinel_idata
    assert entry.fit_ is sentinel_fit
    assert entry.curve_ == "loglogistic"  # NOT the refit's weibull
    assert entry.backend_ == "stan"
    assert entry.contract_["n_w"] == N_W  # NOT the refit's smaller triangle


def test_a_scorer_needs_a_fit_first():
    with pytest.raises(RuntimeError, match=r"call fit\(\) first"):
        GuszczaGrowthCurve()._log_lik_native(None)
    with pytest.raises(RuntimeError, match=r"call fit\(\) first"):
        GuszczaGrowthCurve()._draws_native(None, rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError, match=r"call fit\(\) first"):
        GuszczaGrowthCurve().predict()


# -- layer 2: the agreement gate ----------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("curve", ["loglogistic", "weibull"])
def test_scorer_reproduces_the_fits_own_log_lik(curve):
    """THE gate, per curve. Score the training cells through the held-out code
    path and the result must equal the fit's ``log_likelihood`` group
    elementwise - the scorer is a second, independent implementation of the
    same density, and the fit's own value is the reference."""
    pytest.importorskip("cmdstanpy")

    tri = _triangle(through=N_W, noise=0.05, seed=42)
    entry = GuszczaGrowthCurve().fit(
        tri,
        loss_field="paid_loss",
        premium_field="earned_premium",
        growth_curve=curve,
        backend="stan",
        chains=1,
        iter_warmup=300,
        iter_sampling=300,
        seed=11,
        # a wiring gate, not the retro: stage-1-style settings keep it quick
        target_accept=0.9,
        max_treedepth=12,
    )

    raw = np.asarray(entry.idata_.log_likelihood["log_lik"].values)
    reference = raw.reshape((raw.shape[0] * raw.shape[1], *raw.shape[2:]))
    got = entry._log_lik_native(entry.training_cells())

    assert got.shape == reference.shape
    agreement = np.abs(got - reference).max()

    # Tolerance set by cmdstan's CSV output precision (sig_figs=6), the CSR
    # precedent: parameters and reference alike come back rounded, so the
    # recomputation cannot agree better than ~1e-6 relative. An index error
    # would be O(0.1) or worse.
    assert agreement < 1e-5

    # the gate must be able to fail, with the mutant far above the floor
    perturbed = dict(entry._posterior())
    perturbed["omega"] = perturbed["omega"] * 1.01
    off = np.abs(
        scorer.log_lik_cells(entry.contract_, perturbed, entry.training_cells(), curve=curve)
        - reference
    ).max()
    assert off > 1e-3
    assert off > 100 * agreement

    # and the fitted entry produces a sane predictive distribution: per-origin
    # ultimates plus the total, positive, with the fully developed first
    # origin anchored at its observed value
    pred = entry.predict(seed=0)
    assert pred.samples.shape[1] == entry.contract_["n_w"] + 1
    assert np.isfinite(pred.samples).all()
    assert (pred.samples > 0).all()
    first = entry.contract_["loss"][
        (entry.contract_["w"] == 1) & (entry.contract_["d"] == entry.contract_["n_d"])
    ][0]
    assert np.allclose(pred.samples[:, 0], first)
