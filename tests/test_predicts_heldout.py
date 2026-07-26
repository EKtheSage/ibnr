"""Drawing the outcome at a held-out cell, and the scale that has to be declared.

``ScoresHeldout`` gives a density and hence ELPD; ``PredictsHeldout`` gives draws
and hence CRPS. They are separate mixins because the capabilities are genuinely
independent - ``england_verrall_odp`` can draw and has no usable density, and the
reverse is possible too.

The load-bearing part is ``heldout_draw_scale``. Three of the five Bayesian
entries model INCREMENTS while the Schedule P triangles are CUMULATIVE, so draws
handed to a scorer without a declared scale are wrong by the whole
training-diagonal anchor - and wrong in the way this package keeps finding:
finite, smooth, correctly signed. That is the same bug class as an unconverted
Jacobian on a density, so the conversion lives in the base class where an entry
cannot skip it.

Three layers, mirroring ``test_heldout_scorer_csr.py``:

1. **The scale carry**, on a stub entry, with no sampler.
2. **Closed form** for CSR's own ``draw_cells``.
3. **The agreement gate**: the draws and the density must describe the SAME
   distribution. An index slip in one and not the other is otherwise invisible -
   each is internally consistent and only the comparison sees it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ibnr.gallery.bayesian.meyers_csr import scorer
from ibnr.gallery.bayesian.meyers_csr.model import MeyersCSR
from ibnr.gallery.entry import DRAW_SCALES, PredictsHeldout
from ibnr.kernels.contract import stan_data
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal, training_index
from ibnr.kernels.scores import crps
from ibnr.triangle.core import Triangle

N_W = N_D = 6
PREMIUM = 1000.0


def _triangle(*, through: int) -> Triangle:
    rows = []
    for w in range(1, N_W + 1):
        for d in range(1, N_D + 1):
            if w + d - 1 > through:
                continue
            cum = PREMIUM * 0.65 * (1.0 - np.exp(-0.6 * d)) * (1.0 + 0.03 * w)
            for f, v in (("paid_loss", float(cum)), ("earned_premium", PREMIUM)):
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
    return stan_data(_triangle(through=N_W), loss_field="paid_loss", premium_field="earned_premium")


@pytest.fixture(scope="module")
def heldout():
    """The diagonal after the training cutoff, for the fitted cohort."""
    return next_diagonal(
        _triangle(through=N_W + 1),
        as_of=dt.date(2010 + N_W - 1, 12, 31),
        fields="paid_loss",
        premium_field="earned_premium",
    )


def fake_posterior(contract: dict, n_draws: int = 4000, seed: int = 0) -> dict:
    """Distinct, non-symmetric values per index, so a transposed or off-by-one
    index lands on a visibly wrong number rather than a plausible neighbour."""
    rng = np.random.default_rng(seed)
    n_w, n_d = contract["n_w"], contract["n_d"]
    gamma = rng.uniform(0.01, 0.05, size=n_draws)
    return {
        "logelr": rng.normal(-0.4, 0.05, size=n_draws),
        "alpha": np.tile(np.linspace(0.0, 0.5, n_w), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_w)),
        "beta": np.tile(np.linspace(-1.7, 0.0, n_d), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_d)),
        "speedup": np.cumprod(
            np.column_stack([np.ones(n_draws), np.tile(1.0 - gamma, (n_w - 1, 1)).T]), axis=1
        ),
        "sig": np.tile(np.linspace(0.30, 0.05, n_d), (n_draws, 1)),
    }


class _StubEntry(PredictsHeldout):
    """A fitted entry without a sampler: draws a constant per cell, on a declared
    scale, so the base class's carry is the only thing under test."""

    def __init__(self, contract, per_cell, scale):
        self.contract_ = contract
        self._per_cell = np.asarray(per_cell, dtype=float)
        self.heldout_draw_scale = scale

    def _draws_native(self, cells, *, rng):
        # tiled from _per_cell, NOT sized from cells, so a deliberately wrong
        # width reaches the base class's check instead of a numpy broadcast error
        base = np.tile(self._per_cell, (64, 1))
        return base + rng.normal(0.0, 1e-6, size=base.shape)


# -- layer 1: the scale carry -------------------------------------------------


def test_matching_scale_passes_through_unchanged(contract, heldout):
    """CSR draws cumulatives and the triangle is cumulative, so nothing happens.
    That is the common case, and it is why an undeclared scale is so easy to
    ship: on this data the wrong declaration is a no-op until it isn't."""
    idx = index_into(heldout, contract, field="paid_loss")
    entry = _StubEntry(contract, idx.value, "cumulative")
    got = entry.predict_at(heldout, field="paid_loss")
    assert np.allclose(got.mean(axis=0), idx.value, atol=1e-4)


def test_increment_draws_are_carried_to_the_triangle_basis(contract, heldout):
    """THE test of this mixin.

    An entry drawing increments must have the training-diagonal anchor added
    before its draws can be scored against ``HoldoutCells.values``.

    Mutations: ``return draws`` in place of the carry, or short-circuiting the
    scale comparison. The assertions on the mutant are the point - its CRPS is
    finite, positive and smooth, and it is wrong by the anchor, which on this
    fixture is two orders of magnitude.
    """
    idx = index_into(heldout, contract, field="paid_loss")
    anchor = idx.prev_value
    assert np.isfinite(anchor).all() and (anchor > 0).all()

    true_increment = idx.value - anchor
    entry = _StubEntry(contract, true_increment, "incremental")
    got = entry.predict_at(heldout, field="paid_loss")

    # carried to cumulative, so it lands on the observed outcome
    assert np.allclose(got.mean(axis=0), idx.value, atol=1e-4)

    # what the mutant would produce: plausible, and wrong by the anchor
    uncarried = crps(true_increment[None, :] + np.zeros((64, len(anchor))), heldout.values)
    carried = crps(got, heldout.values)
    assert np.isfinite(uncarried).all() and (uncarried > 0).all()
    assert (uncarried > 10 * carried.clip(min=1e-9)).all()


def test_the_scale_must_be_declared_and_known(contract, heldout):
    entry = _StubEntry(contract, np.ones(heldout.n_cells), "amount")  # a densities.MEASURES word
    with pytest.raises(ValueError, match="heldout_draw_scale must be one of"):
        entry.predict_at(heldout, field="paid_loss")
    assert set(DRAW_SCALES) == {"cumulative", "incremental"}


def test_predict_at_refuses_a_bare_cell_index(contract, heldout):
    """The target basis is a property of the TRIANGLE and only ``HoldoutCells``
    records it. A ``target_scale=`` argument would be a knob whose wrong setting
    is exactly the anchor-sized error above."""
    entry = _StubEntry(contract, np.ones(heldout.n_cells), "cumulative")
    with pytest.raises(TypeError, match="needs a HoldoutCells"):
        entry.predict_at(index_into(heldout, contract, field="paid_loss"))


def test_a_wrong_width_from_an_entry_is_refused(contract, heldout):
    entry = _StubEntry(contract, np.ones(heldout.n_cells - 1), "cumulative")
    with pytest.raises(ValueError, match="expected \\(n_draws"):
        entry.predict_at(heldout, field="paid_loss")


# -- layer 2: CSR's own draws, in closed form ---------------------------------


def test_draw_cells_is_exp_of_a_normal_at_mu_and_sig(contract):
    """``model.stan``'s ``logloss[i] ~ normal(mu[i], sig[d[i]])`` read forwards.
    Reproduced exactly by seeding the same generator."""
    post = fake_posterior(contract, n_draws=50)
    cells = training_index(contract)

    got = scorer.draw_cells(contract, post, cells, rng=np.random.default_rng(3))

    mu = scorer.mu_cells(contract, post, cells)
    sig = np.asarray(post["sig"])[:, cells.d - 1]
    want = np.exp(np.random.default_rng(3).normal(mu, sig))
    assert np.allclose(got, want)
    assert got.shape == (50, cells.n_cells)


def test_draws_are_one_per_posterior_draw_not_a_plug_in(contract):
    """The posterior predictive carries parameter uncertainty AND process noise.
    Drawing repeatedly at the posterior mean would be narrower - better CRPS,
    and nothing in the output says so."""
    post = fake_posterior(contract, n_draws=4000)
    cells = training_index(contract)
    got = scorer.draw_cells(contract, post, cells, rng=np.random.default_rng(5))

    mu = scorer.mu_cells(contract, post, cells)
    sig = np.asarray(post["sig"])[:, cells.d - 1]
    # law of total variance on the log scale: Var(log C) = E[sig^2] + Var(mu)
    want = (sig**2).mean(axis=0) + mu.var(axis=0)
    assert np.allclose(np.log(got).var(axis=0), want, rtol=0.15)
    # a plug-in at the posterior mean would miss the Var(mu) term entirely
    assert (mu.var(axis=0) > 0).any()


def test_draws_do_not_inherit_the_density_refusal_of_a_zero_cell(contract):
    """A zero-paid held-out cell has no lognormal DENSITY at the outcome but a
    perfectly well-defined lognormal PREDICTIVE. So an entry can be CRPS-scorable
    where it is not ELPD-scorable - which is why they are separate mixins."""
    post = fake_posterior(contract, n_draws=20)
    cells = training_index(contract)
    zeroed = CellIndex(
        w=cells.w,
        d=cells.d,
        value=np.where(np.arange(cells.n_cells) == 0, 0.0, cells.value),
        prev_value=cells.prev_value,
        premium=cells.premium,
    )
    with pytest.raises(ValueError, match="non-positive loss"):
        scorer.log_lik_cells(contract, post, zeroed)
    got = scorer.draw_cells(contract, post, zeroed, rng=np.random.default_rng(1))
    assert np.isfinite(got).all() and (got > 0).all()


def test_the_index_matters_for_draws_too(contract):
    """A guard on the guard: if swapping w for d changed nothing, the closed-form
    test would be vacuous."""
    post = fake_posterior(contract, n_draws=30)
    cells = training_index(contract)
    swapped = CellIndex(
        w=cells.d,
        d=cells.w,
        value=cells.value,
        prev_value=cells.prev_value,
        premium=cells.premium,
    )
    a = scorer.draw_cells(contract, post, cells, rng=np.random.default_rng(0))
    b = scorer.draw_cells(contract, post, swapped, rng=np.random.default_rng(0))
    assert not np.allclose(a, b)


# -- layer 3: the agreement gate ----------------------------------------------


def test_the_draws_and_the_density_describe_the_same_distribution(contract):
    """The gate with teeth.

    ``log_lik_cells`` and ``draw_cells`` are separate code paths over the same
    posterior. Each is internally consistent, so an index slip in one and not the
    other is invisible from either side - only this comparison sees it.

    The check: the empirical CDF of the draws must equal the analytic mixture CDF
    implied by the density's own ``mu`` and ``sig``, to Monte Carlo error.

    Probed **across the distribution** rather than at the observed value. The
    training cells sit far into the tail (PIT above 0.99), where the CDF is flat
    and a wrong ``sig`` barely moves it - measured, and it made the first version
    of this test unable to fail. Probing at one standard deviation either side of
    the centre is where a scale error actually shows.
    """
    post = fake_posterior(contract, n_draws=20000)
    cells = training_index(contract)

    draws = scorer.draw_cells(contract, post, cells, rng=np.random.default_rng(17))
    mu = scorer.mu_cells(contract, post, cells)
    sig = np.asarray(post["sig"])[:, cells.d - 1]

    def mixture_cdf(points, scale):
        return norm.cdf((np.log(points)[None, :] - mu) / scale).mean(axis=0)

    mc_error = 3.0 / (2.0 * np.sqrt(draws.shape[0]))
    for k in (-1.0, 0.0, 1.0):
        probe = np.exp(mu.mean(axis=0) + k * sig.mean(axis=0))
        empirical = (draws <= probe[None, :]).mean(axis=0)
        assert np.abs(empirical - mixture_cdf(probe, sig)).max() < mc_error, f"at k={k}"

    # The gate must be capable of failing, and the negative control is the REAL
    # bug it guards against: reading sig one development lag off. CSR's sig
    # shrinks with dev, so that is a large relative change - which is exactly why
    # it would be caught here and nowhere else.
    off_by_one = np.asarray(post["sig"])[:, np.clip(cells.d - 2, 0, None)]
    assert not np.allclose(off_by_one, sig)
    worst = max(
        float(
            np.abs(
                (draws <= np.exp(mu.mean(axis=0) + k * sig.mean(axis=0))[None, :]).mean(axis=0)
                - mixture_cdf(np.exp(mu.mean(axis=0) + k * sig.mean(axis=0)), off_by_one)
            ).max()
        )
        for k in (-1.0, 1.0)
    )
    assert worst > mc_error


def test_csr_declares_both_capabilities():
    entry = MeyersCSR()
    assert isinstance(entry, PredictsHeldout)
    assert entry.heldout_draw_scale == "cumulative"
    assert entry.heldout_measure == "log_amount"


def test_csr_needs_a_fit_before_drawing(heldout):
    with pytest.raises(RuntimeError, match="call fit\\(\\) first"):
        MeyersCSR()._draws_native(None, rng=np.random.default_rng(0))
