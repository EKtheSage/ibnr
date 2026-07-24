"""kernels.mack + kernels.contract.cohort_grid: the native distribution-free
chain ladder.

What this file protects: everything downstream of the chain ladder - the
one-year CDR (``test_cdr.py``), the deterministic gallery entry, the future
speed benchmark - reads its factors and sigmas from here. chainladder-python is
the reference implementation for all of it, so the tie-outs below are exact
rather than approximate; where a convention is a choice (the last step's sigma)
BOTH conventions are pinned against their chainladder counterpart, because the
choice visibly moves the youngest accident year.

The grid guards get their own tests because ``cohort_grid`` deliberately raises
on anything that is not a clean run-off staircase: the factors are estimated
from exactly those cells, so a silently filled hole would change them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.contract import cohort_grid
from ibnr.kernels.mack import fit_mack, simulate_ultimates

from .conftest import make_cohort_triangle

# A 4x4 run-off square with hand-checkable factors: every column doubles-ish,
# and column sums are round numbers so S_j and f_j can be verified by eye.
SMALL = np.array(
    [
        [100.0, 150.0, 165.0, 170.0],
        [120.0, 180.0, 200.0, np.nan],
        [110.0, 165.0, np.nan, np.nan],
        [130.0, np.nan, np.nan, np.nan],
    ]
)


def test_cohort_grid_layout(backend_name):
    """The grid is the dense (n_w, n_d) view of the long triangle: values in
    place, NaN outside, and each origin's diagonal index recorded."""
    grid = cohort_grid(make_cohort_triangle(backend_name, SMALL), loss_field="paid_loss")
    assert (grid["n_w"], grid["n_d"]) == (4, 4)
    np.testing.assert_allclose(np.nan_to_num(grid["cum"]), np.nan_to_num(SMALL))
    np.testing.assert_array_equal(grid["obs_mask"], ~np.isnan(SMALL))
    np.testing.assert_array_equal(grid["latest_dev"], [3, 2, 1, 0])
    assert grid["dev_grain_months"] == 12


def test_cohort_grid_accepts_more_origins_than_devs(backend_name):
    """A triangle whose oldest years have fully run off: the staircase flattens
    at the last dev column instead of continuing to rise."""
    cum = np.array(
        [
            [100.0, 150.0, 165.0],
            [120.0, 180.0, 198.0],
            [110.0, 165.0, 181.0],
            [130.0, 195.0, np.nan],
            [140.0, np.nan, np.nan],
        ]
    )
    grid = cohort_grid(make_cohort_triangle(backend_name, cum), loss_field="paid_loss")
    np.testing.assert_array_equal(grid["latest_dev"], [2, 2, 2, 1, 0])


def test_cohort_grid_rejects_interior_hole(backend_name):
    """A missing cell inside the observed region is an error, not a gap to fill:
    it would silently drop an origin from one development factor."""
    holed = SMALL.copy()
    holed[0, 1] = np.nan
    with pytest.raises(ValueError, match="not a run-off triangle"):
        cohort_grid(make_cohort_triangle(backend_name, holed), loss_field="paid_loss")


def test_cohort_grid_rejects_ragged_diagonal(backend_name):
    """Origins evaluated at different dates (here one origin a year behind the
    rest) break the single-diagonal assumption every recursion below relies on."""
    ragged = SMALL.copy()
    ragged[1, 2] = np.nan  # 2011 stops a year short of the common diagonal
    with pytest.raises(ValueError, match="not a run-off triangle"):
        cohort_grid(make_cohort_triangle(backend_name, ragged), loss_field="paid_loss")


def test_cohort_grid_rejects_multiple_cohorts(backend_name):
    """Two segment combinations in one call would be silently pooled."""
    frames = [
        make_cohort_triangle(None, SMALL, segment={"lob": "auto"}).execute(),
        make_cohort_triangle(None, SMALL * 2, segment={"lob": "home"}).execute(),
    ]
    both = Triangle.from_long(
        pd.concat(frames, ignore_index=True), measure="cumulative", backend=backend_name
    )
    with pytest.raises(ValueError, match="multiple segment combinations"):
        cohort_grid(both, loss_field="paid_loss")


def test_cohort_grid_rejects_incremental(backend_name):
    tri = make_cohort_triangle(backend_name, SMALL).to_incremental()
    with pytest.raises(ValueError, match="cumulative"):
        cohort_grid(tri, loss_field="paid_loss")


def test_factors_and_volumes_by_hand(backend_name):
    """f_j and S_j are the volume-weighted answers, computed here from the
    matrix directly rather than from the fitted object."""
    fit = fit_mack(make_cohort_triangle(backend_name, SMALL), loss_field="paid_loss")
    # step 0 uses origins 0..2 (those observing dev 0 and dev 1)
    s0 = 100.0 + 120.0 + 110.0
    f0 = (150.0 + 180.0 + 165.0) / s0
    s1 = 150.0 + 180.0
    f1 = (165.0 + 200.0) / s1
    np.testing.assert_allclose(fit.s[:2], [s0, s1])
    np.testing.assert_allclose(fit.f[:2], [f0, f1])
    np.testing.assert_array_equal(fit.n_obs, [3, 2, 1])
    np.testing.assert_allclose(fit.latest, [170.0, 200.0, 165.0, 130.0])
    # the last step has one observation, so its sigma is extrapolated, not fitted
    assert np.isfinite(fit.sigma2).all()


def test_last_sigma_rules_differ_only_in_the_last_step(backend_name):
    """The two conventions are identical wherever the data can speak, and only
    the extrapolated final step moves - which is why the choice is exposed."""
    tri = make_cohort_triangle(backend_name, SMALL)
    a = fit_mack(tri, loss_field="paid_loss", sigma_rule="mack")
    b = fit_mack(tri, loss_field="paid_loss", sigma_rule="log_linear")
    np.testing.assert_allclose(a.sigma2[:-1], b.sigma2[:-1])
    np.testing.assert_allclose(a.f, b.f)


def test_rejects_unknown_sigma_rule(backend_name):
    with pytest.raises(ValueError, match="sigma_rule"):
        fit_mack(make_cohort_triangle(backend_name, SMALL), loss_field="paid_loss", sigma_rule="x")


def test_rejects_non_positive_cumulative(backend_name):
    """Mack's variance is proportional to C, so a zero cumulative has no defined
    conditional variance. Named cell, hard error."""
    bad = SMALL.copy()
    bad[2, 0] = 0.0
    with pytest.raises(ValueError, match="non-positive cumulative"):
        fit_mack(make_cohort_triangle(backend_name, bad), loss_field="paid_loss")


def test_ultimate_is_latest_rolled_forward(backend_name):
    """The point estimate is exactly the classical projection; nothing in the
    variance machinery may perturb it."""
    fit = fit_mack(make_cohort_triangle(backend_name, SMALL), loss_field="paid_loss")
    expected = [
        170.0,
        200.0 * fit.f[2],
        165.0 * fit.f[1] * fit.f[2],
        130.0 * fit.f[0] * fit.f[1] * fit.f[2],
    ]
    np.testing.assert_allclose(fit.ultimate, expected)
    np.testing.assert_allclose(fit.reserve, np.array(expected) - fit.latest)
    assert fit.reserve[0] == 0.0  # fully developed origin


def test_simulated_ultimates_recover_the_point_estimate(backend_name):
    """The bootstrap wrapper must be unbiased for the chain-ladder mean: the
    simulation is there for the spread, not to move the answer."""
    fit = fit_mack(make_cohort_triangle(backend_name, SMALL), loss_field="paid_loss")
    pred = simulate_ultimates(fit, n_draws=20_000, seed=4, process="gamma")
    assert pred.n_targets == fit.n_w + 1  # per origin + total
    np.testing.assert_allclose(pred.mean()[:-1], fit.ultimate, rtol=0.02)
    # the total column is the row-sum of the same draws, not an independent sum
    np.testing.assert_allclose(pred.samples[:, -1], pred.samples[:, :-1].sum(axis=1))


def test_simulation_is_reproducible(backend_name):
    fit = fit_mack(make_cohort_triangle(backend_name, SMALL), loss_field="paid_loss")
    a = simulate_ultimates(fit, n_draws=500, seed=99)
    b = simulate_ultimates(fit, n_draws=500, seed=99)
    np.testing.assert_array_equal(a.samples, b.samples)


# -- the gallery entry --------------------------------------------------------


def full_square(n_w: int = 6, seed: int = 7) -> np.ndarray:
    """A COMPLETE run-off square (no NaN): the training triangle is cut out of
    it with ``as_of``, and its last column is the realized ultimate to score
    against - the same backtest shape the rest of the gallery uses."""
    rng = np.random.default_rng(seed)
    factors = np.array([1.5, 1.2, 1.1, 1.05, 1.02])
    cum = np.empty((n_w, n_w))
    cum[:, 0] = rng.uniform(900.0, 1100.0, size=n_w)
    for j in range(n_w - 1):
        cum[:, j + 1] = factors[j] * cum[:, j] + np.sqrt(cum[:, j]) * rng.standard_normal(n_w) * 3.0
    return cum


def test_entry_trains_only_on_the_as_of_slice(backend_name):
    """``as_of`` is the backtest seam: the fit must see the upper triangle only,
    even though it is handed the full square."""
    from ibnr import gallery

    tri = make_cohort_triangle(backend_name, full_square(), start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of="2015-12-31")
    np.testing.assert_array_equal(entry.fit_.latest_dev, [5, 4, 3, 2, 1, 0])
    assert entry.fit_.obs_mask.sum() == 21  # 6+5+4+3+2+1


def test_entry_scores_against_realized_ultimates(backend_name):
    """The full ``fit -> predict -> evaluate`` path on the shared harness, with
    outcomes read off the square's last column. Targets and outcomes must line
    up including the total, or every percentile in the leaderboard is wrong."""
    from ibnr import gallery

    square = full_square()
    tri = make_cohort_triangle(backend_name, square, start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of="2015-12-31")
    outcomes = entry.realized_ultimates(tri)
    np.testing.assert_allclose(outcomes[:-1], square[:, -1])
    assert outcomes[-1] == pytest.approx(square[:, -1].sum())

    pred = entry.predict(n_draws=2_000, seed=8)
    scored = entry.evaluate(outcomes)
    assert len(scored["summary"]) == pred.n_targets
    assert ((scored["percentiles"] >= 0) & (scored["percentiles"] <= 100)).all()


def test_entry_exposes_both_cdr_routes(backend_name):
    """Analytic and simulated one-year CDR are both reachable from the entry and
    answer the same question on the same fit."""
    from ibnr import gallery

    tri = make_cohort_triangle(backend_name, full_square(), start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of="2015-12-31")
    analytic = entry.one_year_cdr()
    simulated = entry.cdr_distribution(n_draws=20_000, seed=9, process="normal")
    assert analytic.method == "merz_wuthrich"
    total = float(np.sqrt((simulated.samples[:, -1] ** 2).mean()))
    assert total == pytest.approx(np.sqrt(analytic.msep_total), rel=0.1)


def test_entry_requires_fit_first():
    from ibnr.gallery.deterministic.mack.model import Mack

    with pytest.raises(RuntimeError, match="call fit"):
        Mack().one_year_cdr()


# -- tie-out to chainladder-python -------------------------------------------

pytest_tieout = pytest.mark.tieout


@pytest.fixture(scope="module")
def raa_triangle():
    """raa as an ibnr Triangle. Module-scoped and duckdb-only: this block is
    about agreeing with chainladder's numbers, and the backend parametrization
    is exercised by every test above."""
    cl = pytest.importorskip("chainladder")
    from ibnr import Triangle

    return Triangle.from_chainladder(cl.load_sample("raa"))


@pytest_tieout
@pytest.mark.parametrize(
    ("rule", "interpolation"), [("mack", "mack"), ("log_linear", "log-linear")]
)
def test_raa_factors_and_sigmas_match_chainladder(raa_triangle, rule, interpolation):
    """Development factors and sigmas, including the extrapolated last one under
    both conventions."""
    cl = pytest.importorskip("chainladder")
    dev = cl.Development(average="volume", sigma_interpolation=interpolation).fit(
        cl.load_sample("raa")
    )
    fit = fit_mack(raa_triangle, loss_field="values", sigma_rule=rule)
    np.testing.assert_allclose(fit.f, np.asarray(dev.ldf_.values).ravel(), rtol=1e-12)
    np.testing.assert_allclose(
        np.sqrt(fit.sigma2), np.asarray(dev.sigma_.values).ravel(), rtol=1e-12
    )


@pytest_tieout
@pytest.mark.parametrize(
    ("rule", "interpolation"), [("mack", "mack"), ("log_linear", "log-linear")]
)
def test_raa_ultimates_and_msep_match_chainladder(raa_triangle, rule, interpolation):
    """Ultimates, per-origin Mack standard error and the aggregate, against
    ``cl.MackChainladder``. The aggregate is the demanding one - it carries the
    estimation-error covariance between accident years."""
    cl = pytest.importorskip("chainladder")
    raa = cl.load_sample("raa")
    dev = cl.Development(average="volume", sigma_interpolation=interpolation).fit(raa)
    ref = cl.MackChainladder().fit(dev.transform(raa)).summary_.to_frame(origin_as_datetime=False)
    fit = fit_mack(raa_triangle, loss_field="values", sigma_rule=rule)
    got = fit.summary()

    np.testing.assert_allclose(got["ultimate"][:-1], ref["Ultimate"].to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(got["latest"][:-1], ref["Latest"].to_numpy(), rtol=1e-12)
    # chainladder reports NaN (not 0) for the fully developed origin's IBNR
    np.testing.assert_allclose(got["ibnr"][1:-1], ref["IBNR"].to_numpy()[1:], rtol=1e-10)
    np.testing.assert_allclose(
        got["runoff_se"][1:-1], ref["Mack Std Err"].to_numpy()[1:], rtol=1e-9
    )
    total = np.asarray(cl.MackChainladder().fit(dev.transform(raa)).total_mack_std_err_).ravel()[0]
    assert got["runoff_se"].iloc[-1] == pytest.approx(float(total), rel=1e-9)


@pytest_tieout
def test_raa_process_parameter_split_matches_chainladder(raa_triangle):
    """The split matters on its own: the one-year CDR damps the two halves
    differently, so a compensating error in either would be invisible in the
    total but wrong in the CDR."""
    cl = pytest.importorskip("chainladder")
    raa = cl.load_sample("raa")
    dev = cl.Development(average="volume", sigma_interpolation="log-linear").fit(raa)
    ref = cl.MackChainladder().fit(dev.transform(raa))
    risk = fit_mack(raa_triangle, loss_field="values", sigma_rule="log_linear").msep_runoff()
    # chainladder reports these as running totals across development; the last
    # entry is the all-origins, all-development figure ours corresponds to.
    assert np.sqrt(risk["process_total"]) == pytest.approx(
        float(np.asarray(ref.total_process_risk_).ravel()[-1]), rel=1e-9
    )
    assert np.sqrt(risk["parameter_total"]) == pytest.approx(
        float(np.asarray(ref.total_parameter_risk_).ravel()[-1]), rel=1e-9
    )
