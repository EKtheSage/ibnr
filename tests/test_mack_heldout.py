"""deterministic/mack joins the CRPS board: held-out one-step-ahead draws.

The entry subclasses ``PredictsHeldout`` (draws, hence CRPS) and deliberately
NOT ``ScoresHeldout``: Mack states two conditional moments and no observation
model, so its ELPD is a permanent, documented N/A - the process law behind the
draws is an assumption of the simulation, and claiming it as a predictive
density is an explicitly reserved decision.

What this file protects, in descending order of subtlety:

1. **The step index.** ``CellIndex.d`` is 1-based while ``f``/``sigma2`` are
   0-based per step, so the factor driving a cell at dev ``d`` is ``f[d - 2]``.
   An off-by-one reads the NEIGHBOURING factor and produces entirely plausible
   draws - finite, positive, correctly ordered - which is the bug class this
   package keeps finding, so the pin is against both candidates.
2. **The closed-form moments**, with and without parameter risk, so the draw is
   Mack's one-step predictive and not merely "something gamma-shaped".
3. **The shared factor draw.** Parameter risk is drawn once per draw and shared
   across cells; that is what correlates them, exactly as it correlates the
   accident years inside the CDR simulation (which shares the same core).
4. **The guards**: ``require_positive_open_diagonals`` reached through
   ``predict_at``, the hand-built-cells refusals, and ``index_into`` against
   the (new) deterministic contract identity.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ibnr import gallery
from ibnr.errors import Refusal
from ibnr.gallery.deterministic.mack.model import Mack
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal
from ibnr.kernels.mack import draw_next_cells, fit_mack
from ibnr.kernels.rng import heldout_stream

from .conftest import make_cohort_triangle

# =============================================================================
# kernel level: draw_next_cells on a hand-built fit and hand-built cells
# =============================================================================


def staircase(n_w: int = 6, seed: int = 3) -> np.ndarray:
    """An upper staircase generated under Mack's own dynamics.

    Factors fall steeply from step to step (1.6, 1.25, ...) on purpose: the
    off-by-one test below is vacuous unless neighbouring factors are far apart.
    """
    rng = np.random.default_rng(seed)
    factors = np.array([1.6, 1.25, 1.12, 1.06, 1.03])
    sigmas = np.array([6.0, 4.0, 3.0, 2.0, 1.5])
    cum = np.empty((n_w, n_w))
    cum[:, 0] = rng.uniform(900.0, 1100.0, size=n_w)
    for j in range(n_w - 1):
        eps = rng.standard_normal(n_w) * sigmas[j]
        cum[:, j + 1] = factors[j] * cum[:, j] + np.sqrt(cum[:, j]) * eps
    cum[np.arange(n_w)[:, None] + np.arange(n_w)[None, :] >= n_w] = np.nan
    return cum


@pytest.fixture(scope="module")
def fit():
    return fit_mack(make_cohort_triangle(None, staircase()), loss_field="paid_loss")


def cell(d: int, prev, w=None) -> CellIndex:
    """Hand-built cells at one dev index: the draw reads only ``d`` and
    ``prev_value``, so ``value`` (a forecast needs no outcome) and ``premium``
    are deliberately NaN and ``w`` is arbitrary."""
    prev = np.atleast_1d(np.asarray(prev, dtype=float))
    n = len(prev)
    return CellIndex(
        w=np.asarray(w if w is not None else np.arange(1, n + 1), dtype=int),
        d=np.full(n, d, dtype=int),
        value=np.full(n, np.nan),
        prev_value=prev,
        premium=np.full(n, np.nan),
    )


def test_one_step_moments_process_only(fit):
    """Closed form with parameter risk off: mean ``f[d-2] * prev``, variance
    ``sigma2[d-2] * prev`` - Mack's two conditional moments, nothing else."""
    j, prev = 1, 1234.5  # 0-based step 1 ends at 1-based dev index 3
    draws = draw_next_cells(
        fit, cell(3, prev), rng=np.random.default_rng(0), n_draws=200_000, parameter_risk=False
    )
    assert draws.shape == (200_000, 1)
    assert draws.mean() == pytest.approx(fit.f[j] * prev, rel=1e-3)
    assert draws.var(ddof=1) == pytest.approx(fit.sigma2[j] * prev, rel=0.03)


def test_parameter_risk_adds_the_estimation_term(fit):
    """With parameter risk on, the variance grows by exactly the estimation
    term ``prev^2 * sigma2 / S`` (Var(f-hat) scaled to the cell), while the
    mean does not move."""
    j, prev = 1, 1234.5
    draws = draw_next_cells(
        fit, cell(3, prev), rng=np.random.default_rng(1), n_draws=200_000, parameter_risk=True
    )
    process = fit.sigma2[j] * prev
    estimation = prev**2 * fit.sigma2[j] / fit.s[j]
    assert estimation > 0.1 * process  # the fixture must make the term visible
    assert draws.mean() == pytest.approx(fit.f[j] * prev, rel=1e-3)
    assert draws.var(ddof=1) == pytest.approx(process + estimation, rel=0.03)


def test_the_driving_factor_is_f_d_minus_2(fit):
    """THE index test. A cell at 1-based dev ``d`` is driven by ``f[d - 2]``;
    the off-by-one reads ``f[d - 1]``, a perfectly plausible neighbouring
    factor, so the pin is against both candidates at once."""
    prev = 1000.0
    assert abs(fit.f[0] - fit.f[1]) * prev > 100.0  # separated, or this is vacuous
    draws = draw_next_cells(
        fit, cell(2, prev), rng=np.random.default_rng(2), n_draws=100_000, parameter_risk=False
    )
    got = float(draws.mean())
    assert got == pytest.approx(fit.f[0] * prev, rel=2e-3)
    assert abs(got - fit.f[1] * prev) > 100.0


def test_shared_factor_draw_correlates_cells_of_a_step(fit):
    """Parameter risk is drawn ONCE per draw and shared across cells: two cells
    on the same development step are positively correlated through it, and
    independent without it. This is the same mechanism that correlates the
    accident years in the CDR simulation - literally the same code."""
    cells = cell(3, [800.0, 1200.0])
    coupled = draw_next_cells(
        fit, cells, rng=np.random.default_rng(5), n_draws=50_000, parameter_risk=True
    )
    independent = draw_next_cells(
        fit, cells, rng=np.random.default_rng(5), n_draws=50_000, parameter_risk=False
    )
    assert np.corrcoef(coupled, rowvar=False)[0, 1] > 0.05
    assert abs(np.corrcoef(independent, rowvar=False)[0, 1]) < 0.02


def test_refuses_dev_index_one(fit):
    """No development step ends at dev 1, so there is no f[-1] to read - and
    numpy would happily serve f[-1] as the LAST factor, silently."""
    with pytest.raises(ValueError, match="dev index 1"):
        draw_next_cells(fit, cell(1, 1000.0), rng=np.random.default_rng(0))


def test_refuses_devs_beyond_the_fitted_steps(fit):
    """d = n_d is the deepest drawable cell (driven by the last fitted factor);
    one past it has no factor and must not wrap or extrapolate."""
    deepest = draw_next_cells(fit, cell(fit.n_d, 1000.0), rng=np.random.default_rng(0), n_draws=8)
    assert deepest.shape == (8, 1)
    with pytest.raises(ValueError, match="beyond the fitted steps"):
        draw_next_cells(fit, cell(fit.n_d + 1, 1000.0), rng=np.random.default_rng(0))


def test_refuses_a_missing_predecessor(fit):
    with pytest.raises(ValueError, match="no training predecessor"):
        draw_next_cells(fit, cell(3, np.nan), rng=np.random.default_rng(0))


def test_zero_sigma_step_is_a_silent_point_mass():
    """A triangle developing by exactly constant factors has sigma2 = 0, and
    the draws collapse to the conditional mean exactly - documented behavior
    (the card calls it out), not an error: zero estimated variance IS the
    model's answer, however implausible the triangle that produced it."""
    factors = [1.5, 1.2, 1.1]
    cum = np.full((4, 4), np.nan)
    for i in range(4):
        cum[i, 0] = 100.0 * (i + 1)
        for j in range(3 - i):
            cum[i, j + 1] = cum[i, j] * factors[j]
    fit0 = fit_mack(make_cohort_triangle(None, cum), loss_field="paid_loss")
    draws = draw_next_cells(fit0, cell(2, 500.0), rng=np.random.default_rng(0), n_draws=50)
    np.testing.assert_array_equal(draws, np.full((50, 1), fit0.f[0] * 500.0))


# =============================================================================
# entry level: predict_at through the gallery, cells from next_diagonal
# =============================================================================

AS_OF = "2015-12-31"


def full_square(n_w: int = 6, seed: int = 7) -> np.ndarray:
    """A COMPLETE run-off square: ``as_of`` cuts the training staircase out of
    it, so the next diagonal is real observed data rather than a simulation."""
    rng = np.random.default_rng(seed)
    factors = np.array([1.5, 1.2, 1.1, 1.05, 1.02])
    cum = np.empty((n_w, n_w))
    cum[:, 0] = rng.uniform(900.0, 1100.0, size=n_w)
    for j in range(n_w - 1):
        cum[:, j + 1] = factors[j] * cum[:, j] + np.sqrt(cum[:, j]) * rng.standard_normal(n_w) * 3.0
    return cum


@pytest.fixture(scope="module")
def square() -> np.ndarray:
    return full_square()


@pytest.fixture(scope="module")
def triangle(square):
    return make_cohort_triangle(None, square, start_year=2010, segment={"lob": "wkcomp"})


@pytest.fixture(scope="module")
def entry(triangle):
    return gallery.fit("mack", triangle, loss_field="paid_loss", as_of=AS_OF)


@pytest.fixture(scope="module")
def heldout(triangle):
    return next_diagonal(triangle, as_of=AS_OF, fields="paid_loss")


def test_entry_declares_draws_and_not_a_density():
    """CRPS member, permanent ELPD N/A: the mixin split is the design."""
    entry = Mack()
    assert isinstance(entry, PredictsHeldout)
    assert not isinstance(entry, ScoresHeldout)
    assert entry.heldout_draw_scale == "cumulative"


def test_contract_carries_the_indexable_identity(entry, heldout):
    """The grid the entry now keeps as ``contract_`` must satisfy everything
    ``index_into`` demands: cohort identity, measure, and the training cells.
    Premium stays optional - absent means NaN, and the draws never read it."""
    for key in ("segment", "fields", "models", "measure", "w", "d"):
        assert key in entry.contract_, key
    assert entry.contract_["segment"] == {"lob": "wkcomp"}
    assert entry.contract_["fields"] == ("paid_loss",) == entry.contract_["models"]
    idx = index_into(heldout, entry.contract_)
    assert idx.n_cells == heldout.n_cells == 5
    assert np.isnan(idx.premium).all()


def test_index_into_refuses_training_cells(entry, heldout):
    """The overlap refusal depends on the contract's new ``w``/``d``: shifting
    every held-out cell one dev step back lands it exactly on the training
    diagonal, and scoring there would report in-sample fit as held-out skill."""
    onto_training = replace(
        heldout, frame=heldout.frame.assign(dev_lag=heldout.frame["dev_lag"] - 12)
    )
    with pytest.raises(ValueError, match="TRAINING data"):
        index_into(onto_training, entry.contract_)


def test_predict_at_is_reproducible_and_passed_through(entry, heldout):
    """Seed reproducibility, plus the cumulative pass-through: mack draws
    cumulatives and the triangle is cumulative, so the base class's anchor
    step must not touch the draws - they equal the kernel's exactly.

    The kernel is handed the stream ``predict_at`` derives from the seed and
    these cells, not ``default_rng(11)``: a study-level seed no longer names a
    generator directly (``kernels.rng``), so rebuilding the stream the same way
    is what keeps this an equality against the kernel rather than against the
    derivation."""
    a = entry.predict_at(heldout, seed=11)
    b = entry.predict_at(heldout, seed=11)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, entry.predict_at(heldout, seed=12))

    idx = index_into(heldout, entry.contract_)
    want = draw_next_cells(
        entry.fit_,
        idx,
        rng=np.random.default_rng(heldout_stream(11, heldout)),
        n_draws=10_000,
    )
    np.testing.assert_array_equal(a, want)
    assert a.shape == (10_000, heldout.n_cells)


def test_heldout_knobs_reach_the_kernel(triangle, heldout):
    """Parameter delivery through the PUBLIC entry point - the inert-parameter
    bug class. Exact stream equality is the teeth: an ignored ``process`` or
    ``parameter_risk`` changes the rng consumption, an ignored ``n_draws``
    changes the shape."""
    entry = gallery.fit(
        "mack",
        triangle,
        loss_field="paid_loss",
        as_of=AS_OF,
        heldout_n_draws=64,
        heldout_process="normal",
        heldout_parameter_risk=False,
    )
    got = entry.predict_at(heldout, seed=5)
    assert got.shape == (64, heldout.n_cells)
    idx = index_into(heldout, entry.contract_)
    want = draw_next_cells(
        entry.fit_,
        idx,
        rng=np.random.default_rng(heldout_stream(5, heldout)),
        n_draws=64,
        process="normal",
        parameter_risk=False,
    )
    np.testing.assert_array_equal(got, want)


def test_draw_means_track_the_one_step_moments(entry, heldout):
    """End to end: every held-out cell's draws centre on ``f[d-2]`` times its
    OWN training predecessor - each cell of the diagonal reads a different
    development step, so a shuffled or broadcast prev would show here."""
    idx = index_into(heldout, entry.contract_)
    draws = entry.predict_at(heldout, seed=3)
    want = entry.fit_.f[idx.d - 2] * idx.prev_value
    np.testing.assert_allclose(draws.mean(axis=0), want, rtol=0.02)


def test_zero_open_diagonal_refused_through_predict_at(square):
    """``require_positive_open_diagonals`` must fire on the draw path: without
    it, var = sigma2 * 0 makes the cell a silent point mass, not an error.
    ``fit`` itself succeeds (the point estimate is fine) - only the draw
    refuses, naming the origin."""
    bad = square.copy()
    bad[5, 0] = 0.0  # the youngest origin's training diagonal IS its dev-1 cell
    tri = make_cohort_triangle(None, bad, start_year=2010, segment={"lob": "wkcomp"})
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of=AS_OF)
    cells = next_diagonal(tri, as_of=AS_OF, fields="paid_loss")
    with pytest.raises(ValueError, match="latest diagonal of open origin"):
        entry.predict_at(cells, seed=0)


def test_failed_refit_leaves_the_previous_fit_intact(triangle, heldout):
    """fit() must be ATOMIC - a confirmed review defect, not a hypothetical.

    ``cohort_grid`` accepts cohorts that ``fit_mack_grid`` then refuses (here:
    every link ratio from 12 to 24 months starts from zero). Assigning
    ``contract_`` before that raise left a TORN entry: the NEW cohort's contract
    over the OLD cohort's factors.
    ``index_into`` checks identity against the contract, so ``predict_at`` on
    the new cohort's cells passed every check and returned 10,000 x n_cells of
    plausible draws of cohort B's cells from cohort A's factors.

    Post-fix: a failed refit changes NOTHING - the entry still predicts A's
    cells bit-identically, and B's cells are refused as the wrong cohort.
    """
    entry = gallery.fit("mack", triangle, loss_field="paid_loss", as_of=AS_OF)
    before = entry.predict_at(heldout, seed=21)

    # cohort B passes the grid contract but fails the estimator: every pair
    # origin at the first dev step sits at zero, so the factor is 0/0
    bad = full_square()
    bad[:, 0] = 0.0
    tri_b = make_cohort_triangle(None, bad, start_year=2010, segment={"lob": "comauto"})
    with pytest.raises(Refusal, match=r"every link ratio from 12 to 24 months starts from zero"):
        entry.fit(tri_b, loss_field="paid_loss", as_of=AS_OF)

    # (a) the previous fitted state survives, fully consistent: same draws
    np.testing.assert_array_equal(entry.predict_at(heldout, seed=21), before)

    # (b) the failed cohort's cells are refused against the SURVIVING contract
    cells_b = next_diagonal(tri_b, as_of=AS_OF, fields="paid_loss")
    with pytest.raises(ValueError, match="belong to cohort"):
        entry.predict_at(cells_b, seed=0)
