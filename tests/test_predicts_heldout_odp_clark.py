"""Held-out draws for the ODP family: england_verrall_odp, clark_growth_curve,
and the MLE clark.

All three model INCREMENTS on a cumulative triangle and all three draw the
outcome as ``X = phi * Poisson(mu / phi)``; none of them has a normalized
predictive density (``kernels/densities.py``, the odp-not-a-density note), so
these are draws-only entries: ``PredictsHeldout`` and never ``ScoresHeldout``.

The layers, mirroring ``test_predicts_heldout.py``:

1. **Declarations** - the mixin, the incremental scale, and the deliberate
   ABSENCE of the density mixin, asserted so adding it later is a test failure
   rather than a drive-by.
2. **Closed forms** against the model files, with fake posteriors whose values
   are distinct per index so an off-by-one lands on a visibly wrong number.
3. **Moments** - E[X] = mu and Var[X] = phi * mu, the od-Poisson law.
4. **The scale carry** - predict_at must add the training-diagonal anchor
   (the 996-vs-3.4 bug class from CLAUDE.md).
5. **Slow agreement gates** - the scorer's mu at the training cells must
   reproduce a real fit's own ``log_mu`` / ``mu`` elementwise.

Every substantive test names the mutation it was verified to catch.
"""

from __future__ import annotations

import copy
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ibnr.gallery.bayesian.clark_growth_curve import scorer as clark_scorer
from ibnr.gallery.bayesian.clark_growth_curve.model import ClarkGrowthCurve
from ibnr.gallery.bayesian.england_verrall_odp import scorer as odp_scorer
from ibnr.gallery.bayesian.england_verrall_odp.model import EnglandVerrallODP
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout
from ibnr.gallery.statistical.clark import scorer as mle_scorer
from ibnr.gallery.statistical.clark.model import Clark, age_interval, growth
from ibnr.kernels.contract import odp_stan_data
from ibnr.kernels.densities import POISSON_RATE_MAX
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal, training_index
from ibnr.kernels.rng import cohort_stream, heldout_stream
from ibnr.triangle.core import Triangle

N_W = N_D = 6
PHI = 50.0

#: a dispersion small enough that every cell's Poisson rate ``mu / phi`` is
#: past the largest rate numpy can draw. Reachable on real fits: a triangle
#: that develops exactly on the fitted curve fits itself to rounding error and
#: the Pearson scale collapses to about 1e-29 (``tests/test_clark.py``).
TINY_PHI = 1e-30


def _premium(w: int) -> float:
    # Per-origin premium, DELIBERATELY not constant: the ODP contract's
    # "logprem" is per TRAINING ROW, and with equal premiums the wrong lookup
    # returns the right number (see the premium-offset test below).
    return 1000.0 + 200.0 * w


def _triangle(*, through: int) -> Triangle:
    rows = []
    for w in range(1, N_W + 1):
        for d in range(1, N_D + 1):
            if w + d - 1 > through:
                continue
            cum = _premium(w) * 0.65 * (1.0 - np.exp(-0.6 * d))
            for f, v in (("paid_loss", float(cum)), ("earned_premium", _premium(w))):
                rows.append(
                    {
                        "lob": "FIT_CO",
                        "origin_period": dt.date(2010 + w - 1, 1, 1),
                        "dev_lag": 12 * d,
                        "eval_date": dt.date(2010 + w - 1 + d - 1, 12, 31),
                        "field": f,
                        "value": v,
                    }
                )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


@pytest.fixture(scope="module")
def contract() -> dict:
    c = odp_stan_data(
        _triangle(through=N_W), loss_field="paid_loss", premium_field="earned_premium"
    )
    # fit() injects the plug-in Pearson dispersion; the tests pin a known one
    c["phi"] = PHI
    return c


@pytest.fixture(scope="module")
def heldout():
    """The diagonal after the training cutoff: cells (2,6) .. (6,2)."""
    return next_diagonal(
        _triangle(through=N_W + 1),
        as_of=dt.date(2010 + N_W - 1, 12, 31),
        fields="paid_loss",
        premium_field="earned_premium",
    )


@pytest.fixture(scope="module")
def clark_cape() -> Clark:
    return Clark().fit(
        _triangle(through=N_W),
        loss_field="paid_loss",
        premium_field="earned_premium",
        growth_curve="loglogistic",
        method="cape_cod",
    )


@pytest.fixture(scope="module")
def clark_ldf() -> Clark:
    return Clark().fit(
        _triangle(through=N_W),
        loss_field="paid_loss",
        premium_field="earned_premium",
        growth_curve="loglogistic",
        method="ldf",
    )


def odp_posterior(contract: dict, n_draws: int, seed: int = 0, jitter: float = 1e-3) -> dict:
    """Distinct, non-symmetric values per index, so a transposed or off-by-one
    index lands on a visibly wrong number. ``jitter=0`` pins every draw at the
    same known parameters (for the moment checks)."""
    rng = np.random.default_rng(seed)
    n_w, n_d = contract["n_w"], contract["n_d"]
    return {
        "c": np.full(n_draws, -0.55) + rng.normal(0, jitter, n_draws),
        "alpha": np.tile(np.linspace(0.0, 0.5, n_w), (n_draws, 1))
        + rng.normal(0, jitter, (n_draws, n_w)),
        "beta": np.tile(np.linspace(-1.2, -3.0, n_d), (n_draws, 1))
        + rng.normal(0, jitter, (n_draws, n_d)),
    }


def clark_posterior(n_draws: int, seed: int = 0, jitter: float = 1e-3) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "logelr": np.full(n_draws, -0.45) + rng.normal(0, jitter, n_draws),
        "omega": np.full(n_draws, 1.4) + rng.normal(0, jitter, n_draws),
        "theta": np.full(n_draws, 42.0) + rng.normal(0, jitter, n_draws),
    }


def _fake_idata(post: dict) -> SimpleNamespace:
    """Just enough of an ``InferenceData`` for ``pooled()``, which reads
    ``idata.posterior[name].values`` of shape ``(chain, draw, *dims)``. One
    chain, so pooling is a reshape that changes nothing. This is what lets the
    stubs below reach ``predict()``, not only ``_draws_native``."""
    return SimpleNamespace(
        posterior={
            name: SimpleNamespace(values=np.asarray(values, dtype=float)[None, ...])
            for name, values in post.items()
        }
    )


class _StubODP(EnglandVerrallODP):
    """A fitted-shaped entry without a sampler: real contract, fake posterior.
    Only ``_posterior`` and the ``idata_`` it reads are overridden, so
    ``_draws_native``, ``predict`` and the base-class carry are the real code
    paths under test."""

    def __init__(self, contract: dict, post: dict) -> None:
        super().__init__()
        self.contract_ = contract
        self.idata_ = _fake_idata(post)  # non-None: the fit guard passes
        self._loss_field = "paid_loss"
        self._post = post

    def _posterior(self) -> dict:
        return self._post


class _StubClarkGC(ClarkGrowthCurve):
    """Same idea for the Bayesian Clark; the curve is fitted state."""

    def __init__(self, contract: dict, post: dict, curve: str = "loglogistic") -> None:
        super().__init__()
        self.contract_ = contract
        self.idata_ = _fake_idata(post)
        self._loss_field = "paid_loss"
        self._curve = curve
        self._post = post

    def _posterior(self) -> dict:
        return self._post


# -- layer 1: declarations ----------------------------------------------------


@pytest.mark.parametrize("cls", [EnglandVerrallODP, ClarkGrowthCurve, Clark])
def test_entries_declare_draws_and_not_density(cls):
    """The family draws increments and is permanently ELPD-ineligible: the ODP
    quasi-likelihood is not a normalized density on any scale (densities.py,
    odp-not-a-density - the integral is 0.69 at mu/phi=0.5 and VARIES with
    mu/phi). Subclassing ScoresHeldout later must be a deliberate act that
    breaks this test, not a drive-by."""
    assert issubclass(cls, PredictsHeldout)
    assert cls.heldout_draw_scale == "incremental"
    assert not issubclass(cls, ScoresHeldout)
    assert not hasattr(cls, "heldout_measure")


@pytest.mark.parametrize("cls", [EnglandVerrallODP, ClarkGrowthCurve, Clark])
def test_unfitted_entry_refuses_to_draw(cls):
    with pytest.raises(RuntimeError, match="call fit\\(\\) first"):
        cls()._draws_native(None, rng=np.random.default_rng(0))


# -- layer 2: closed forms against the model files ----------------------------


@pytest.mark.parametrize("where", ["training", "heldout"])
def test_odp_mu_cells_matches_the_stan_formula_cellwise(contract, heldout, where):
    """model.stan:51: log_mu[i] = logprem[i] + c + alpha[w[i]] + beta[d[i]],
    re-evaluated with a scalar loop so the vectorized gather is pinned - at
    held-out cells as well as training cells, because the premium trap only
    bites where the cell list stops being the training rows.

    Mutations (verified to fail): alpha read one origin off (the
    index-arithmetic mutation this file exists for); beta read one dev off;
    premium read from contract["logprem"][cells.w - 1] (the per-training-row
    trap - returns origin 1's rows for every cell, wrong wherever premiums
    differ)."""
    cells = training_index(contract) if where == "training" else index_into(heldout, contract)
    post = odp_posterior(contract, n_draws=7)
    got = odp_scorer.mu_cells(contract, post, cells)
    assert got.shape == (7, cells.n_cells)

    prem = np.asarray(contract["premium"], dtype=float)
    for i in range(cells.n_cells):
        w, d = int(cells.w[i]), int(cells.d[i])
        for k in range(7):
            want = np.exp(
                np.log(prem[w - 1])
                + post["c"][k]
                + post["alpha"][k, w - 1]
                + post["beta"][k, d - 1]
            )
            assert np.isclose(got[k, i], want, rtol=1e-12), (i, k)


def test_odp_mu_at_training_cells_reproduces_the_fits_own_premium_offset(contract):
    """contract["logprem"] IS the fit's per-row offset (model.stan:51 reads
    logprem[i], one entry per TRAINING ROW). At the training cells the
    scorer's per-origin premium read must reproduce it row for row - the fast
    twin of the slow agreement gate below. The fixture's premiums differ by
    origin, so a wrong read cannot hide.

    Mutation (verified): mu_cells reading contract["logprem"][cells.w - 1]."""
    assert len(np.unique(np.asarray(contract["premium"]))) == contract["n_w"]
    cells = training_index(contract)
    post = odp_posterior(contract, n_draws=5)
    got = np.log(odp_scorer.mu_cells(contract, post, cells))
    want = (
        np.asarray(contract["logprem"], dtype=float)[None, :]
        + post["c"][:, None]
        + np.take(post["alpha"], cells.w - 1, axis=1)
        + np.take(post["beta"], cells.d - 1, axis=1)
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_odp_draw_cells_is_phi_times_poisson_of_mu_over_phi(contract, heldout):
    """model.stan:59 read forwards: X = phi * Poisson(mu / phi), one draw per
    posterior draw, reproduced exactly by seeding the same generator.

    Compared BYTE for byte against the literal expression, so the shared
    ``odp_draw`` cannot change an ordinary fit's numbers at all: the rate cap
    and the point mass at it must be reachable only past what numpy can draw.

    Mutation (verified): rng.poisson(mu) without the phi scaling - the mean
    survives it, so only an exact or a variance check can see it."""
    post = odp_posterior(contract, n_draws=50)
    idx = index_into(heldout, contract)
    got = odp_scorer.draw_cells(contract, post, idx, rng=np.random.default_rng(3))
    mu = odp_scorer.mu_cells(contract, post, idx)
    want = PHI * np.random.default_rng(3).poisson(mu / PHI)
    assert got.shape == (50, idx.n_cells)
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))
    assert (got >= 0).all()  # non-negative multiples of phi by construction


@pytest.mark.parametrize("curve", ["loglogistic", "weibull"])
def test_clark_mu_cells_matches_the_stan_formula_cellwise(contract, heldout, curve):
    """model.stan:51-53: mu[i] = exp(logprem_w[w[i]] + logelr) *
    (G(age_hi) - G(age_lo)) with mid-period ages, re-evaluated with a scalar
    loop, for both curves.

    Mutations (verified): age_hi computed as step*d + step/2 (mid-period shift
    sign flip); growth called with omega and theta swapped."""
    cells = index_into(heldout, contract)
    post = clark_posterior(n_draws=5)
    got = clark_scorer.mu_cells(contract, post, cells, curve=curve)
    assert got.shape == (5, cells.n_cells)

    step = contract["dev_grain_months"]
    prem = np.asarray(contract["premium"], dtype=float)
    for i in range(cells.n_cells):
        w, d = int(cells.w[i]), int(cells.d[i])
        lo = max(step * (d - 1) - step / 2, 0.0)
        hi = step * d - step / 2
        for k in range(5):
            om, th = post["omega"][k], post["theta"][k]
            ginc = growth(hi, om, th, curve) - growth(lo, om, th, curve)
            want = max(np.exp(post["logelr"][k]) * prem[w - 1] * ginc, 1e-12)
            assert np.isclose(got[k, i], want, rtol=1e-10), (i, k)


def test_clark_age_interval_first_cell_clamps_to_zero_exactly(contract):
    """The mid-period convention at its one sharp edge: dev 1 covers
    (0, step/2], and the lower age is EXACTLY 0.0 - not merely small - because
    model.stan's growth_curve branches on x <= 0 and growth() returns exactly
    0.0 there (the milestone-5 NaN-gradient story hinges on this cell).

    Mutation (verified): dropping the np.maximum clamp makes the first cell's
    lower age -step/2 and growth() silently clips it back - the ages would be
    wrong while every value stayed finite."""
    step = contract["dev_grain_months"]
    lo, hi = age_interval(1, step)
    assert float(lo) == 0.0
    assert float(hi) == step / 2
    assert float(growth(lo, 1.4, 42.0, "loglogistic")) == 0.0
    assert float(growth(lo, 1.4, 42.0, "weibull")) == 0.0

    # the scorer's ages at the training rows ARE the entry's stan_data ages
    cells = training_index(contract)
    slo, shi = age_interval(cells.d, step)
    np.testing.assert_array_equal(slo, np.maximum(step * (cells.d - 1) - step / 2, 0.0))
    np.testing.assert_array_equal(shi, step * cells.d - step / 2)
    assert (cells.d == 1).any()
    assert (slo[cells.d == 1] == 0.0).all()


def test_clark_draw_cells_is_phi_times_poisson_of_mu_over_phi(contract, heldout):
    """model.stan:61 read forwards, reproduced exactly by seeding the same
    generator, byte for byte for the reason above. Mutation (verified):
    dropping the phi scaling."""
    post = clark_posterior(n_draws=50)
    idx = index_into(heldout, contract)
    got = clark_scorer.draw_cells(
        contract, post, idx, curve="loglogistic", rng=np.random.default_rng(3)
    )
    mu = clark_scorer.mu_cells(contract, post, idx, curve="loglogistic")
    want = PHI * np.random.default_rng(3).poisson(mu / PHI)
    assert got.shape == (50, idx.n_cells)
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))


def test_mle_param_draws_reproduce_the_predict_recipe(clark_cape, clark_ldf):
    """model.py's MVN recipe verbatim: exp of a multivariate normal on the log
    parameters, levels recovered the way the fitted method defines them -
    cape_cod scales ONE ELR draw by each origin's premium, ldf exponentiates
    the first n_w entries. Reproduced exactly by seeding the same generator.

    Mutation (verified): cape_cod levels read from draws[:, :n_w] (the ldf
    recovery applied to the wrong parameter vector)."""
    for entry in (clark_cape, clark_ldf):
        c, prm = entry.contract_, entry.params_
        post = mle_scorer.param_draws(c, prm, n_draws=64, rng=np.random.default_rng(9))
        ld = np.random.default_rng(9).multivariate_normal(
            prm["log_params"], prm["log_cov"], size=64
        )
        np.testing.assert_allclose(post["omega"], np.exp(ld[:, -2]))
        np.testing.assert_allclose(post["theta"], np.exp(ld[:, -1]))
        assert post["level"].shape == (64, N_W)
        if prm["method"] == "ldf":
            want_level = np.exp(ld[:, :N_W])
        else:
            want_level = np.exp(ld[:, [0]]) * np.asarray(c["premium"], dtype=float)[None, :]
        np.testing.assert_allclose(post["level"], want_level)


def test_mle_draw_cells_reproduces_the_recipe_exactly(clark_cape, heldout):
    """The whole chain - MVN parameter sample, shared ages, growth increment,
    floor, od-Poisson - reproduced end to end with the same seeded generator,
    including the rng consumption ORDER (parameters first, then one Poisson
    call). Mutations (verified): a plug-in sample with the parameter risk
    zeroed; a hard-coded draw count; cape_cod levels recovered the ldf way."""
    c, prm = clark_cape.contract_, clark_cape.params_
    idx = index_into(heldout, c)
    got = mle_scorer.draw_cells(c, prm, idx, n_draws=64, rng=np.random.default_rng(9))

    r = np.random.default_rng(9)
    ld = r.multivariate_normal(prm["log_params"], prm["log_cov"], size=64)
    om, th = np.exp(ld[:, -2]), np.exp(ld[:, -1])
    level = np.exp(ld[:, [0]]) * np.asarray(c["premium"], dtype=float)[None, :]
    lo, hi = age_interval(idx.d, c["dev_grain_months"])
    ginc = growth(hi[None, :], om[:, None], th[:, None], "loglogistic") - growth(
        lo[None, :], om[:, None], th[:, None], "loglogistic"
    )
    mu = np.maximum(level[:, idx.w - 1] * ginc, 1e-12)
    want = prm["phi"] * r.poisson(mu / prm["phi"])
    assert got.shape == (64, idx.n_cells)
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))


# -- layer 3: moments (the od-Poisson law) ------------------------------------


def test_odp_draw_moments_are_mu_and_phi_mu(contract, heldout):
    """With the posterior pinned at known parameters, the draws' empirical
    mean converges to mu and their variance to phi * mu - the od-Poisson law
    (E&V 3.2) and the reason phi matters at all.

    Mutation (verified): rng.poisson(mu) without phi keeps the mean and
    collapses the variance by a factor of phi = 50."""
    post = odp_posterior(contract, n_draws=20_000, jitter=0.0)
    idx = index_into(heldout, contract)
    mu = odp_scorer.mu_cells(contract, post, idx)[0]  # identical across draws
    draws = odp_scorer.draw_cells(contract, post, idx, rng=np.random.default_rng(11))
    np.testing.assert_allclose(draws.mean(axis=0), mu, rtol=0.03)
    np.testing.assert_allclose(draws.var(axis=0), PHI * mu, rtol=0.08)


def test_clark_draw_moments_are_mu_and_phi_mu(contract, heldout):
    """Same law, same mutation target, for the Bayesian Clark."""
    post = clark_posterior(n_draws=20_000, jitter=0.0)
    idx = index_into(heldout, contract)
    mu = clark_scorer.mu_cells(contract, post, idx, curve="loglogistic")[0]
    draws = clark_scorer.draw_cells(
        contract, post, idx, curve="loglogistic", rng=np.random.default_rng(11)
    )
    np.testing.assert_allclose(draws.mean(axis=0), mu, rtol=0.03)
    np.testing.assert_allclose(draws.var(axis=0), PHI * mu, rtol=0.08)


def test_clark_entry_threads_its_fitted_curve_into_the_draws(contract, heldout):
    """The growth-curve name is FITTED STATE (``_curve``), not a contract key,
    so the entry glue must thread it into the scorer. A hard-coded curve in
    ``_draws_native`` agrees with itself through every other test (the stub
    and the carry compare the same mutated path to itself); only this
    entry-vs-scorer comparison, run for BOTH curves, can see it - the
    inert-parameter bug class: a signature proves the wire exists, not that it
    is connected.

    Mutation (verified): curve="weibull" hard-coded in _draws_native."""
    post = clark_posterior(60, jitter=0.0)
    idx = index_into(heldout, contract)
    for curve in ("loglogistic", "weibull"):
        entry = _StubClarkGC(contract, post, curve=curve)
        native = entry._draws_native(idx, rng=np.random.default_rng(21))
        want = clark_scorer.draw_cells(
            contract, post, idx, curve=curve, rng=np.random.default_rng(21)
        )
        np.testing.assert_allclose(native, want)
    # the two curves genuinely disagree at these cells, or the loop is vacuous
    assert not np.allclose(
        clark_scorer.mu_cells(contract, post, idx, curve="loglogistic"),
        clark_scorer.mu_cells(contract, post, idx, curve="weibull"),
    )


def test_mle_draws_agree_with_predicts_own_machinery(clark_cape, heldout):
    """Origin 2's ONLY unobserved cell inside the training window is the
    held-out cell (2, 6), so predict()'s ultimate for origin 2 minus its
    paid-to-date is that single future increment drawn by the entry's own
    published machinery. Moment agreement there pins _draws_native to
    predict() - same MVN, same ages, same floor, same process law.

    Mutation (verified): recovering cape_cod levels the ldf way inside
    param_draws breaks this agreement. (The plug-in mutation - zeroing the
    parameter risk - is pinned by the exact-reproduction test above; on this
    smooth fixture process noise dominates the variance, so a moment check
    alone cannot see it.)"""
    c = clark_cape.contract_
    assert int(c["latest_d"][1]) == N_D - 1

    pred = clark_cape.predict(n_draws=30_000, seed=101)
    inc_pred = pred.samples[:, 1] - c["paid_to_date"][1]

    idx = index_into(heldout, c)
    assert (int(idx.w[0]), int(idx.d[0])) == (2, 6)
    entry = copy.deepcopy(clark_cape)
    entry.n_heldout_draws = 30_000
    native = entry._draws_native(idx, rng=np.random.default_rng(202))

    assert np.isclose(native[:, 0].mean(), inc_pred.mean(), rtol=0.05)
    assert np.isclose(native[:, 0].var(), inc_pred.var(), rtol=0.20)


def test_mle_zero_and_clark_zero_emergence_cells_draw_all_zeros(contract, heldout, clark_cape):
    """A weibull with a tiny theta is fully emerged before every held-out
    cell's age interval: G(lo) == G(hi) == 1.0 exactly in float, the growth
    increment is 0, and mu hits the shared 1e-12 floor - so every draw is 0.
    Finite, legal, and the model's own statement that the cell has no expected
    emergence; the zero-variance panel bookkeeping is forecast.py's job, per
    the family's rules, not a refusal here.

    The floor VALUE is asserted, in both scorers: rng.poisson refuses a
    negative rate, so a roundoff-negative growth increment without the floor
    is a crash on real data. Mutation (verified): dropping the np.maximum
    floor in clark's mu_cells returns 0.0 here."""
    idx = index_into(heldout, contract)
    assert (idx.d >= 2).all()  # every held-out cell has a positive lower age

    post = {
        "logelr": np.full(50, -0.45),
        "omega": np.full(50, 8.0),
        "theta": np.full(50, 0.5),
    }
    mu = clark_scorer.mu_cells(contract, post, idx, curve="weibull")
    np.testing.assert_array_equal(mu, np.full_like(mu, 1e-12))
    draws = clark_scorer.draw_cells(
        contract, post, idx, curve="weibull", rng=np.random.default_rng(2)
    )
    assert np.isfinite(draws).all()
    assert (draws == 0.0).all()

    mle_post = {
        "level": np.full((50, N_W), 500.0),
        "omega": np.full(50, 8.0),
        "theta": np.full(50, 0.5),
    }
    mle_mu = mle_scorer.mu_cells(clark_cape.contract_, mle_post, idx, curve="weibull")
    np.testing.assert_array_equal(mle_mu, np.full_like(mle_mu, 1e-12))


# -- the other end of the same axis: a dispersion too small to draw -----------


@pytest.mark.parametrize("phi", [TINY_PHI, 0.0])
@pytest.mark.parametrize("which", ["odp", "clark_growth_curve", "clark_mle"])
def test_negligible_dispersion_draws_the_mean_exactly(contract, heldout, clark_cape, which, phi):
    """When the over-dispersion collapses, ``phi * Poisson(mu / phi)`` asks
    numpy for a rate it cannot represent, and the honest answer is the mean:
    at the limit the draw's coefficient of variation ``sqrt(phi / mu)`` is
    below 3.3e-10, so the law is a point mass to any precision that matters.
    Same convention as ``kernels.mack.draw_step``'s zero-variance step.

    All three scorers, because all three write the same draw, and both ends of
    the collapse: a rate past the cap and ``phi`` exactly 0.

    Pre-fix (verified) all six cases raised ``ValueError: lam value too large``
    out of numpy: at ``phi = 0`` the rate is not merely huge but infinite."""
    if which == "clark_mle":
        c = clark_cape.contract_
        prm = dict(clark_cape.params_)
        prm["phi"] = phi
        idx = index_into(heldout, c)
        post = mle_scorer.param_draws(c, prm, n_draws=40, rng=np.random.default_rng(5))
        mu = mle_scorer.mu_cells(c, post, idx, curve=prm["growth_curve"])
        got = mle_scorer.draw_cells(c, prm, idx, n_draws=40, rng=np.random.default_rng(5))
    else:
        c = dict(contract)
        c["phi"] = phi
        idx = index_into(heldout, c)
        if which == "odp":
            post = odp_posterior(contract, n_draws=40)
            mu = odp_scorer.mu_cells(c, post, idx)
            got = odp_scorer.draw_cells(c, post, idx, rng=np.random.default_rng(5))
        else:
            post = clark_posterior(n_draws=40)
            mu = clark_scorer.mu_cells(c, post, idx, curve="loglogistic")
            got = clark_scorer.draw_cells(
                c, post, idx, curve="loglogistic", rng=np.random.default_rng(5)
            )

    # the case has to BE the case: every rate past the cap, or the test is a
    # test of ordinary Poisson draws that happen to be tight
    if phi > 0:
        assert (mu / phi > POISSON_RATE_MAX).all()
    assert got.shape == mu.shape
    assert np.array_equal(got.view(np.uint8), mu.view(np.uint8))


@pytest.mark.parametrize("which", ["odp", "clark_growth_curve", "clark_mle"])
def test_predict_routes_through_the_shared_odp_draw(contract, clark_cape, which):
    """The same collapse through the PUBLIC entry point, for all three
    ``predict()`` methods. Each one adds a process draw per future cell, so
    each one has to reach the shared draw; a site left writing
    ``phi * rng.poisson(mu / phi)`` directly raises here while every scorer
    test above still passes (the inert-parameter bug class, one layer up).

    The expected ultimates are rebuilt cell by cell in the same order
    ``predict()`` accumulates them, so the comparison is byte exact - which
    also pins that a capped cell consumes NO random numbers: the MLE arm
    rebuilds the whole generator stream from its parameter sample alone.

    Pre-fix (verified): ``ValueError: lam value too large`` from all three."""
    if which == "clark_mle":
        entry = copy.deepcopy(clark_cape)
        entry.params_ = dict(entry.params_)
        entry.params_["phi"] = TINY_PHI
        c = entry.contract_
        pred = entry.predict(n_draws=40, seed=5)

        rng = np.random.default_rng(
            cohort_stream(5, label="predict", cohorts=entry.cohorts(), field=entry._loss_field)
        )
        post = mle_scorer.param_draws(c, entry.params_, n_draws=40, rng=rng)
        want = np.tile(np.asarray(c["paid_to_date"], dtype=float), (40, 1))
        for j in range(c["n_w"]):
            for dev in range(int(c["latest_d"][j]) + 1, c["n_d"] + 1):
                lo, hi = age_interval(dev, c["dev_grain_months"])
                ginc = growth(hi, post["omega"], post["theta"], "loglogistic") - growth(
                    lo, post["omega"], post["theta"], "loglogistic"
                )
                mu = np.maximum(post["level"][:, j] * ginc, 1e-12)
                assert (mu / TINY_PHI > POISSON_RATE_MAX).all()
                want[:, j] += mu
    else:
        c = dict(contract)
        c["phi"] = TINY_PHI
        want = np.tile(np.asarray(c["paid_to_date"], dtype=float), (40, 1))
        if which == "odp":
            post = odp_posterior(contract, n_draws=40, jitter=0.0)
            entry = _StubODP(c, post)
            pred = entry.predict(seed=5)
            logprem = np.log(np.asarray(c["premium"], dtype=float))
            for j in range(c["n_w"]):
                for dev in range(int(c["latest_d"][j]) + 1, c["n_d"] + 1):
                    mu = np.exp(
                        logprem[j] + post["c"] + post["alpha"][:, j] + post["beta"][:, dev - 1]
                    )
                    assert (mu / TINY_PHI > POISSON_RATE_MAX).all()
                    want[:, j] += mu
        else:
            post = clark_posterior(n_draws=40, jitter=0.0)
            entry = _StubClarkGC(c, post)
            pred = entry.predict(seed=5)
            elr_prem = np.exp(
                post["logelr"][:, None] + np.log(np.asarray(c["premium"], dtype=float))[None, :]
            )
            for j in range(c["n_w"]):
                for dev in range(int(c["latest_d"][j]) + 1, c["n_d"] + 1):
                    lo, hi = age_interval(dev, c["dev_grain_months"])
                    ginc = growth(hi, post["omega"], post["theta"], "loglogistic") - growth(
                        lo, post["omega"], post["theta"], "loglogistic"
                    )
                    mu = np.maximum(elr_prem[:, j] * ginc, 1e-12)
                    assert (mu / TINY_PHI > POISSON_RATE_MAX).all()
                    want[:, j] += mu

    got = pred.samples[:, :-1]  # the last column is the total with_total() adds
    assert got.shape == want.shape
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))


# -- the index matters (guards on the guards) ---------------------------------


def test_odp_the_index_matters(contract):
    """If swapping w for d changed nothing, the closed-form tests above would
    be vacuous."""
    cells = training_index(contract)
    assert (cells.w != cells.d).any()
    swapped = CellIndex(
        w=cells.d, d=cells.w, value=cells.value, prev_value=cells.prev_value, premium=cells.premium
    )
    post = odp_posterior(contract, n_draws=7)
    assert not np.allclose(
        odp_scorer.mu_cells(contract, post, cells), odp_scorer.mu_cells(contract, post, swapped)
    )


def test_clark_the_index_matters(contract, clark_cape):
    cells = training_index(contract)
    assert (cells.w != cells.d).any()
    swapped = CellIndex(
        w=cells.d, d=cells.w, value=cells.value, prev_value=cells.prev_value, premium=cells.premium
    )
    post = clark_posterior(n_draws=7)
    assert not np.allclose(
        clark_scorer.mu_cells(contract, post, cells, curve="loglogistic"),
        clark_scorer.mu_cells(contract, post, swapped, curve="loglogistic"),
    )
    mle_post = mle_scorer.param_draws(
        clark_cape.contract_, clark_cape.params_, n_draws=7, rng=np.random.default_rng(1)
    )
    assert not np.allclose(
        mle_scorer.mu_cells(clark_cape.contract_, mle_post, cells, curve="loglogistic"),
        mle_scorer.mu_cells(clark_cape.contract_, mle_post, swapped, curve="loglogistic"),
    )


# -- layer 4: the scale carry through predict_at ------------------------------


def test_odp_predict_at_adds_the_training_anchor(contract, heldout):
    """The 996-vs-3.4 bug class: ODP draws INCREMENTS while the triangle is
    cumulative, so predict_at must add each cell's training-diagonal anchor.
    With the same seed the carried draws are native + anchor ELEMENTWISE, and
    the uncarried mean misses the realized outcome by several times the
    carried error on every cell - finite, plausible, and wrong.

    Mutation (verified): heldout_draw_scale = "cumulative" on the entry leaves
    the draws uncarried and both assertions fail."""
    post = odp_posterior(contract, n_draws=400, jitter=0.0)
    entry = _StubODP(contract, post)
    idx = index_into(heldout, contract)

    got = entry.predict_at(heldout, field="paid_loss", seed=7)
    # the stream predict_at derives, rebuilt: a study-level seed no longer names
    # a generator directly (kernels.rng)
    stream = heldout_stream(7, heldout, field="paid_loss")
    native = entry._draws_native(idx, rng=np.random.default_rng(stream))
    np.testing.assert_allclose(got, native + idx.prev_value[None, :])

    carried_err = np.abs(got.mean(axis=0) - heldout.values)
    uncarried_err = np.abs(native.mean(axis=0) - heldout.values)
    assert (uncarried_err > 5 * carried_err).all()


def test_clark_predict_at_adds_the_training_anchor(contract, heldout):
    """Same carry, Bayesian Clark. Mutation (verified): scale declared
    cumulative."""
    entry = _StubClarkGC(contract, clark_posterior(400, jitter=0.0))
    idx = index_into(heldout, contract)
    got = entry.predict_at(heldout, field="paid_loss", seed=7)
    stream = heldout_stream(7, heldout, field="paid_loss")
    native = entry._draws_native(idx, rng=np.random.default_rng(stream))
    np.testing.assert_allclose(got, native + idx.prev_value[None, :])
    assert (idx.prev_value > 0).all()


def test_mle_predict_at_adds_the_training_anchor(clark_cape, heldout):
    """Same carry, MLE Clark - through the real fitted entry end to end."""
    idx = index_into(heldout, clark_cape.contract_)
    got = clark_cape.predict_at(heldout, field="paid_loss", seed=5)
    stream = heldout_stream(5, heldout, field="paid_loss")
    native = clark_cape._draws_native(idx, rng=np.random.default_rng(stream))
    np.testing.assert_allclose(got, native + idx.prev_value[None, :])


# -- plumbing: shapes, draw count, seeds --------------------------------------


def test_draws_native_shapes_and_the_draw_count_is_delivered(contract, heldout, clark_cape):
    """(n_draws, n_cells) for every entry - and the MLE entry's draw count is
    an INSTANCE attribute that must actually reach the draws through the
    public predict_at (the inert-parameter bug class: a wire that exists is
    not a wire that is connected).

    Mutation (verified): draw_cells ignoring n_draws for a hard-coded
    10_000."""
    idx = index_into(heldout, contract)
    assert _StubODP(contract, odp_posterior(contract, n_draws=40))._draws_native(
        idx, rng=np.random.default_rng(0)
    ).shape == (40, idx.n_cells)
    assert _StubClarkGC(contract, clark_posterior(40))._draws_native(
        idx, rng=np.random.default_rng(0)
    ).shape == (40, idx.n_cells)

    entry = copy.deepcopy(clark_cape)
    entry.n_heldout_draws = 123
    assert entry.predict_at(heldout, field="paid_loss", seed=0).shape == (123, idx.n_cells)


def test_predict_at_is_seed_reproducible(contract, heldout, clark_cape):
    entries = [
        _StubODP(contract, odp_posterior(contract, n_draws=60)),
        _StubClarkGC(contract, clark_posterior(60)),
        clark_cape,
    ]
    for entry in entries:
        a = entry.predict_at(heldout, field="paid_loss", seed=11)
        b = entry.predict_at(heldout, field="paid_loss", seed=11)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, entry.predict_at(heldout, field="paid_loss", seed=12))


def test_mle_predict_still_runs_after_the_scorer_refactor(clark_ldf):
    """predict() now routes its parameter sample through scorer.param_draws;
    a smoke check that both methods' predict machinery survived the refactor
    (the cape_cod path is exercised by the agreement test above)."""
    pred = clark_ldf.predict(n_draws=200, seed=3)
    assert pred.samples.shape == (200, N_W + 1)
    assert np.isfinite(pred.samples).all()


# -- layer 5: slow agreement gates against a real fit -------------------------


@pytest.mark.slow
def test_odp_scorer_reproduces_the_fits_own_log_mu():
    """The scorer handed the TRAINING cells must reproduce the fit's own
    log_mu elementwise - without this, the index arithmetic can be wrong and
    every held-out number merely plausible. Tolerance is set by cmdstan's CSV
    precision (sig_figs=6), not by the scorer."""
    pytest.importorskip("cmdstanpy")
    from ibnr.gallery.bayesian.england_verrall_odp.model import pooled

    entry = EnglandVerrallODP().fit(
        _triangle(through=N_W),
        backend="stan",
        chains=1,
        iter_warmup=300,
        iter_sampling=300,
        seed=11,
    )
    cells = training_index(entry.contract_)
    got = odp_scorer.mu_cells(entry.contract_, entry._posterior(), cells)
    want = np.exp(pooled(entry.idata_, "log_mu"))
    agreement = float(np.abs(got / want - 1.0).max())
    assert agreement < 1e-4

    # negative control: the gate must be able to fail, by orders of magnitude
    post = entry._posterior()
    post["beta"] = post["beta"] + 0.01
    off = float(np.abs(odp_scorer.mu_cells(entry.contract_, post, cells) / want - 1.0).max())
    assert off > 1e-3 and off > 100 * agreement


@pytest.mark.slow
def test_clark_scorer_reproduces_the_fits_own_mu():
    """Same gate for the Bayesian Clark: scorer mu at training cells ==
    the fit's transformed-parameter mu, which pins the shared age convention
    against model.stan's age_lo/age_hi data block as sampled."""
    pytest.importorskip("cmdstanpy")
    from ibnr.gallery.bayesian.clark_growth_curve.model import pooled

    entry = ClarkGrowthCurve().fit(
        _triangle(through=N_W),
        backend="stan",
        chains=1,
        iter_warmup=300,
        iter_sampling=300,
        seed=11,
    )
    cells = training_index(entry.contract_)
    got = clark_scorer.mu_cells(entry.contract_, entry._posterior(), cells, curve=entry._curve)
    want = pooled(entry.idata_, "mu")
    agreement = float(np.abs(got / want - 1.0).max())
    assert agreement < 1e-4

    post = entry._posterior()
    post["theta"] = post["theta"] * 1.05
    off = float(
        np.abs(
            clark_scorer.mu_cells(entry.contract_, post, cells, curve=entry._curve) / want - 1.0
        ).max()
    )
    assert off > 1e-3 and off > 100 * agreement
