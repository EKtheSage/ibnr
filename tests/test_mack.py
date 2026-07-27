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
from ibnr.kernels.contract import cohort_grid, cohort_grid_frame
from ibnr.kernels.mack import _tail_sigma2, fit_mack, fit_mack_many, simulate_ultimates

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


def test_cohort_grid_carries_identity_and_cell_indices(backend_name):
    """Milestone 6: the deterministic contract carries the same cohort identity
    as the Stan contracts (so ``kernels.holdout.index_into`` can refuse cells
    that are not this fit's) plus 1-based ``w``/``d`` for every observed cell
    (so it can refuse cells the fit was TRAINED on)."""
    tri = make_cohort_triangle(backend_name, SMALL, segment={"lob": "auto"})
    grid = cohort_grid(tri, loss_field="paid_loss")
    assert grid["segment"] == {"lob": "auto"}
    assert grid["fields"] == ("paid_loss",) == grid["models"]
    assert grid["measure"] == "cumulative"
    w0, d0 = np.nonzero(grid["obs_mask"])
    np.testing.assert_array_equal(grid["w"], w0 + 1)
    np.testing.assert_array_equal(grid["d"], d0 + 1)


def test_cohort_grid_frame_takes_identity_from_the_caller(backend_name):
    """The frame half has no Triangle to read an identity from, so the batch
    caller (fit_mack_many's group loop) supplies segment and measure - and they
    must land on the contract unchanged, or every batch fit is unindexable."""
    df = make_cohort_triangle(backend_name, SMALL).execute()
    grid = cohort_grid_frame(
        df,
        dev_grain_months=12,
        loss_field="paid_loss",
        segment={"company_code": "0001"},
        measure="cumulative",
    )
    assert grid["segment"] == {"company_code": "0001"}
    assert grid["measure"] == "cumulative"
    assert grid["fields"] == ("paid_loss",)


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


def test_zero_cumulative_keeps_the_factor_and_shrinks_the_sigma_sample(backend_name):
    """An accident year at zero is a usable chain-ladder observation but not a
    usable variance observation, and the two estimators part company there.

    ``SMALL[2, 0] = 0`` puts the zero strictly above the diagonal (origin 2 is
    observed at dev 0 and dev 1, so it is in step 0's pair set). The factor is
    the column total including that origin - 0 into the denominator, its 165
    into the numerator - which is exactly what the retired inline chain-ladder
    benchmark in ``scripts/compare_gallery.py`` computed and what 2 of the 152
    scored Schedule P cohorts need. Sigma drops it and loses a degree of freedom
    with it. Every quantity below is hand-checkable from the matrix.
    """
    bad = SMALL.copy()
    bad[2, 0] = 0.0
    fit = fit_mack(make_cohort_triangle(backend_name, bad), loss_field="paid_loss")

    s0 = 100.0 + 120.0 + 0.0
    f0 = (150.0 + 180.0 + 165.0) / s0
    np.testing.assert_allclose([fit.s[0], fit.f[0]], [s0, f0])  # 220.0, 2.25
    # the two origin counts diverge exactly here, which is the point of n_pos
    np.testing.assert_array_equal(fit.n_obs, [3, 2, 1])
    np.testing.assert_array_equal(fit.n_pos, [2, 2, 1])
    # sigma over origins 0 and 1 only, df = 2 - 1 = 1
    resid = 100.0 * (150.0 / 100.0 - f0) ** 2 + 120.0 * (180.0 / 120.0 - f0) ** 2
    np.testing.assert_allclose(fit.sigma2[0], resid / 1)  # 123.75
    # and the variance machinery stays finite - the zero is above the diagonal,
    # so nothing downstream ever divides by it
    assert np.isfinite(fit.msep_runoff()["msep"]).all()
    assert np.isfinite(fit.msep_runoff()["msep_total"])


def test_rejects_negative_cumulative(backend_name):
    """A negative weight makes sigma_j^2 itself negative, hence a negative msep
    and a NaN standard error - all silently. Hard error, named origin."""
    bad = SMALL.copy()
    bad[2, 0] = -50.0
    with pytest.raises(ValueError, match="negative cumulative"):
        fit_mack(make_cohort_triangle(backend_name, bad), loss_field="paid_loss")


def test_rejects_zero_volume_step(backend_name):
    """S_j = 0 makes the volume-weighted factor 0/0. numpy would return inf or
    nan and carry on, so it is checked rather than divided."""
    bad = SMALL.copy()
    bad[0, 0] = bad[1, 0] = bad[2, 0] = 0.0
    with pytest.raises(ValueError, match="zero volume at dev step 1"):
        fit_mack(make_cohort_triangle(backend_name, bad), loss_field="paid_loss")


def test_rejects_step_with_too_few_positive_origins(backend_name):
    """A step with several pairs but under two positive ones has no sigma, and
    must NOT fall through to the last-step extrapolation.

    This is the trap the relaxation opens: ``_tail_sigma2`` looks only backwards,
    so at j = 0 it finds nothing and returns 0.0 under BOTH sigma rules - which
    would silently declare the first and most volatile development step
    noiseless, and understate the youngest year's CDR accordingly.
    """
    bad = SMALL.copy()
    bad[1, 0] = bad[2, 0] = 0.0  # leaves exactly one positive origin at step 0
    with pytest.raises(ValueError, match="only 1 with a positive cumulative"):
        fit_mack(make_cohort_triangle(backend_name, bad), loss_field="paid_loss")


def test_tail_sigma_never_silently_fills_an_early_step(backend_name):
    """Pins the mechanism behind the test above rather than just its symptom:
    left to itself ``_tail_sigma2`` answers 0.0 at step 0 under both rules."""
    assert _tail_sigma2(np.full(3, np.nan), 0, rule="mack") == 0.0
    assert _tail_sigma2(np.full(3, np.nan), 0, rule="log_linear") == 0.0


def test_point_estimate_survives_a_zero_diagonal_but_the_variance_does_not(backend_name):
    """The cell no factor-side guard can see: an open origin's latest diagonal.

    It has no observed successor, so it enters no development step's estimator -
    yet every variance formula divides by it. Before this guard existed the fit
    was accepted and ``msep_runoff()`` returned NaN for that origin AND a NaN
    total, with no exception. The point estimate genuinely does not need the
    cell, so it stays available; only the variance path refuses.
    """
    bad = SMALL.copy()
    bad[3, 0] = 0.0  # youngest origin: its diagonal IS dev 0
    fit = fit_mack(make_cohort_triangle(backend_name, bad), loss_field="paid_loss")
    np.testing.assert_allclose(fit.ultimate[3], 0.0)  # the projection is still defined
    assert np.isfinite(fit.ultimate).all()
    for call in (
        fit.msep_runoff,
        fit.summary,
        lambda: simulate_ultimates(fit, n_draws=16, seed=0),
    ):
        with pytest.raises(ValueError, match="latest diagonal of open origin"):
            call()


def test_zero_diagonal_on_a_closed_origin_is_fine(backend_name):
    """A fully developed origin has no remaining step, so nothing divides by its
    last cell - the guard is scoped to OPEN origins and must not over-reach."""
    ok = SMALL.copy()
    ok[0, 3] = 0.0  # origin 0 is at the last dev column: closed
    fit = fit_mack(make_cohort_triangle(backend_name, ok), loss_field="paid_loss")
    assert np.isfinite(fit.msep_runoff()["msep"]).all()


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


# -- fit_mack_many: the batch entry point ---------------------------------------


def test_fit_mack_many_matches_loop(backend_name):
    """The batch fit IS the loop, minus the per-cohort engine round-trips: every
    estimated quantity must match fit_mack on the filtered cohort exactly."""
    import ibis

    from .conftest import make_multiline_triangle

    t = make_multiline_triangle(backend_name, {"wkcomp": SMALL, "comauto": SMALL * 1.2 + 5.0})
    panel = fit_mack_many(t, loss_field="paid_loss")
    assert panel.by == ("company_code", "line_of_business")
    assert len(panel) == 2 and not panel.errors
    for lob in ("wkcomp", "comauto"):
        one = fit_mack(t.filter(ibis._.line_of_business == lob), loss_field="paid_loss")
        batch = panel[("0001", lob)]
        np.testing.assert_allclose(batch.f, one.f)
        np.testing.assert_allclose(batch.sigma2, one.sigma2)
        np.testing.assert_allclose(batch.s, one.s)
        np.testing.assert_allclose(np.nan_to_num(batch.cum), np.nan_to_num(one.cum))
        np.testing.assert_allclose(batch.ultimate, one.ultimate)
        assert batch.origin_periods == one.origin_periods


def test_fit_mack_many_as_of_matches_loop(backend_name):
    """as_of slices once, before the grouping; each cohort must see the same
    training staircase the per-cohort loop would."""
    import ibis

    from .conftest import make_multiline_triangle

    t = make_multiline_triangle(backend_name, {"wkcomp": SMALL, "comauto": SMALL * 1.2 + 5.0})
    panel = fit_mack_many(t, loss_field="paid_loss", as_of="2012-12-31")
    for lob in ("wkcomp", "comauto"):
        one = fit_mack(
            t.filter(ibis._.line_of_business == lob), loss_field="paid_loss", as_of="2012-12-31"
        )
        np.testing.assert_allclose(panel[("0001", lob)].f, one.f)
        np.testing.assert_allclose(panel[("0001", lob)].ultimate, one.ultimate)


def test_fit_mack_many_without_segments(backend_name):
    """A segment-less triangle is one anonymous cohort: key () and by ()."""
    panel = fit_mack_many(make_cohort_triangle(backend_name, SMALL), loss_field="paid_loss")
    assert panel.by == ()
    one = fit_mack(make_cohort_triangle(backend_name, SMALL), loss_field="paid_loss")
    np.testing.assert_allclose(panel[()].f, one.f)


def test_fit_mack_many_on_error(backend_name):
    """A broken cohort (interior hole) fails fast by default, naming the cohort;
    on_error='skip' quarantines it in .errors and still fits the rest."""
    from .conftest import make_multiline_triangle

    holed = SMALL.copy()
    holed[0, 1] = np.nan
    t = make_multiline_triangle(backend_name, {"good": SMALL, "holed": holed})
    with pytest.raises(ValueError, match="line_of_business"):
        fit_mack_many(t, loss_field="paid_loss")
    panel = fit_mack_many(t, loss_field="paid_loss", on_error="skip")
    assert set(panel.fits) == {("0001", "good")}
    assert ("0001", "holed") in panel.errors
    assert "run-off" in panel.errors[("0001", "holed")]


def test_fit_mack_many_summary(backend_name):
    """summary() carries the segment keys and the per-cohort point quantities."""
    from .conftest import make_multiline_triangle

    t = make_multiline_triangle(backend_name, {"wkcomp": SMALL, "comauto": SMALL * 1.2 + 5.0})
    panel = fit_mack_many(t, loss_field="paid_loss")
    got = panel.summary().set_index("line_of_business")
    for lob in ("wkcomp", "comauto"):
        assert got.loc[lob, "ultimate"] == pytest.approx(panel[("0001", lob)].ultimate.sum())
        assert got.loc[lob, "ibnr"] == pytest.approx(panel[("0001", lob)].reserve.sum())
