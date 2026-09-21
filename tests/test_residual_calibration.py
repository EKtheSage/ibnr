"""kernels.residual_calibration: size-stratified rolling-origin residual pools.

The R study's uncertainty for a point forecaster, written once: apply the fixed
forecaster at earlier cutoffs, standardise the errors by a floored scale, pool
them by unit size, centre each pool on its median, and resample around the final
point. A synthetic forecaster with a KNOWN error law is the arbiter here - the
draws must recover its quantiles - and every refusal is named.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibnr.kernels.residual_calibration import (
    Calibration,
    calibrate,
    calibrated_draws,
    leave_one_out_coverage,
    ntile,
    rolling_residuals,
)

N_UNITS = 200
SIZES = np.linspace(100.0, 20_000.0, N_UNITS)  # premium-like, strictly increasing
CUTOFFS = (5, 6, 7, 8, 9)
N_PERIODS = 10


def _synthetic(seed=0, spread=0.2):
    """Forecast = 3 x size; actual = forecast x (1 + u), u uniform on [-spread, spread]
    drawn per (cutoff, unit). The error law is known, so the pools must recover it."""
    rng = np.random.default_rng(seed)
    u = {k: rng.uniform(-spread, spread, size=N_UNITS) for k in CUTOFFS}

    def forecast_at(k):
        return 3.0 * SIZES

    def actual_at(k):
        return 3.0 * SIZES * (1.0 + u[k])

    return forecast_at, actual_at, u


def _synthetic_residuals(**over):
    forecast_at, actual_at, _ = _synthetic(**over)
    return rolling_residuals(
        forecast_at, actual_at, cutoffs=CUTOFFS, n_periods=N_PERIODS, size=SIZES
    )


def test_ntile_matches_dplyr():
    """dplyr >= 1.0: bins as equal as possible, the LARGER bins first. Five values
    in two bins are 3 + 2; ten in four are 3 + 3 + 2 + 2."""
    assert ntile(np.array([5.0, 1.0, 3.0, 2.0, 4.0]), 2).tolist() == [2, 1, 1, 1, 2]
    assert ntile(np.arange(10.0), 4).tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    assert ntile(np.array([2.0, 2.0, 1.0]), 3).tolist() == [2, 3, 1]  # ties broken by position
    # the R study's 93 companies fall into premium quartiles of 24, 23, 23, 23
    assert [ntile(np.arange(93.0), 4).tolist().count(b) for b in (1, 2, 3, 4)] == [24, 23, 23, 23]


def test_rolling_residuals_columns_horizon_and_scale():
    forecast_at, actual_at, u = _synthetic()
    rows = rolling_residuals(
        forecast_at, actual_at, cutoffs=CUTOFFS, n_periods=N_PERIODS, size=SIZES
    )
    assert list(rows.columns) == [
        "unit",
        "cutoff",
        "horizon",
        "predicted",
        "actual",
        "size",
        "residual_scale",
        "standardised_error",
    ]
    assert len(rows) == N_UNITS * len(CUTOFFS)
    assert rows.attrs["n_dropped_nonfinite"] == 0
    assert set(rows["horizon"]) == {5, 4, 3, 2, 1}
    first = rows[(rows["cutoff"] == 5) & (rows["unit"] == 0)].iloc[0]
    assert first["horizon"] == 5
    # scale = max(|predicted|, 0.01 * size, 1): here |predicted| = 300 dominates
    assert first["residual_scale"] == pytest.approx(300.0)
    assert first["standardised_error"] == pytest.approx(u[5][0])


def test_rolling_residuals_floors_the_scale_and_drops_nonfinite_rows():
    size = np.array([50.0, 1e6])

    def forecast_at(k):
        return np.array([0.0, np.nan])

    def actual_at(k):
        return np.array([2.0, 1.0])

    rows = rolling_residuals(forecast_at, actual_at, cutoffs=(1,), n_periods=2, size=size)
    assert len(rows) == 1
    assert rows.attrs["n_dropped_nonfinite"] == 1
    assert rows["residual_scale"].iloc[0] == pytest.approx(1.0)  # max(0, 0.5, 1)


def test_rolling_residuals_refuses_a_forecaster_of_the_wrong_length():
    def bad(k):
        return np.ones(3)

    with pytest.raises(ValueError, match="length"):
        rolling_residuals(bad, bad, cutoffs=(1,), n_periods=2, size=SIZES)


def test_calibrate_strata_are_size_quartiles_over_units_and_pools_are_centred():
    rows = _synthetic_residuals()
    cal = calibrate(rows, horizons=(3, 4, 5))
    assert isinstance(cal, Calibration)
    assert cal.horizons == (3, 4, 5)
    strata = np.array([cal.unit_stratum[u] for u in range(N_UNITS)])
    assert strata.tolist() == ntile(SIZES, 4).tolist()
    assert all(len(p) == 50 * 3 for p in cal.pools)  # 50 units x 3 horizons per stratum
    for p in cal.pools:
        assert np.median(p) == pytest.approx(0.0, abs=1e-12)
    assert len(cal.residuals) == N_UNITS * 3
    np.testing.assert_array_equal(cal.strata_for(SIZES), strata)
    assert cal.strata_for(np.array([0.0, 1e9])).tolist() == [1, 4]


def test_calibrate_refuses_a_thin_stratum_and_unknown_horizons():
    rows = _synthetic_residuals()
    with pytest.raises(ValueError, match="stratum"):
        calibrate(rows, horizons=(3,), n_strata=4, min_per_stratum=51)
    with pytest.raises(ValueError, match="horizon"):
        calibrate(rows, horizons=(3, 9))


def test_calibrated_draws_recover_the_known_error_law():
    rows = _synthetic_residuals(spread=0.2)
    cal = calibrate(rows, horizons=(3, 4, 5))
    point = 3.0 * SIZES
    draws = calibrated_draws(
        cal, point=point, size=SIZES, n_draws=20_000, rng=np.random.default_rng(1)
    )
    assert draws.shape == (20_000, N_UNITS)
    # u is uniform on [-0.2, 0.2] with median 0, so the 10th and 90th percentiles of
    # draws / point - 1 sit near -0.16 and +0.16
    rel = draws / point[None, :] - 1.0
    assert np.quantile(rel, 0.10) == pytest.approx(-0.16, abs=0.02)
    assert np.quantile(rel, 0.90) == pytest.approx(0.16, abs=0.02)
    again = calibrated_draws(
        cal, point=point, size=SIZES, n_draws=20_000, rng=np.random.default_rng(1)
    )
    np.testing.assert_array_equal(draws, again)


def test_calibrated_draws_use_the_units_own_stratum():
    """Two strata whose CENTRED pools are disjoint in size: a small unit's draws
    can only be +/- 0.1 of its point and a large unit's only +/- 0.2, so where a
    draw landed says which pool it came from. Centring alone would put both at
    the point and hide the routing."""
    size = np.array([1.0, 2.0, 3.0, 4.0] * 25)  # 100 units, two strata of 50
    small = size <= 2.0
    # stratum 1: -0.6 and -0.4 about a median of -0.5; stratum 2: +0.3 and +0.7
    # about a median of +0.5
    by_cutoff = {
        1: np.where(small, -0.6, 0.3),
        2: np.where(small, -0.4, 0.7),
    }

    def forecast_at(k):
        return np.full(100, 100.0)

    def actual_at(k):
        return 100.0 * (1.0 + by_cutoff[k])

    rows = rolling_residuals(forecast_at, actual_at, cutoffs=(1, 2), n_periods=3, size=size)
    cal = calibrate(rows, horizons=(1, 2), n_strata=2, min_per_stratum=10)
    assert cal.medians.tolist() == [-0.5, 0.5]
    draws = calibrated_draws(
        cal, point=np.full(100, 100.0), size=size, n_draws=50, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(np.abs(draws[:, small] - 100.0), 10.0)
    np.testing.assert_allclose(np.abs(draws[:, ~small] - 100.0), 20.0)


def test_calibrated_draws_refusals():
    rows = _synthetic_residuals()
    cal = calibrate(rows, horizons=(3, 4, 5))
    with pytest.raises(ValueError, match="same length"):
        calibrated_draws(
            cal, point=np.ones(3), size=SIZES, n_draws=10, rng=np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="n_draws"):
        calibrated_draws(
            cal, point=3.0 * SIZES, size=SIZES, n_draws=0, rng=np.random.default_rng(0)
        )


def test_leave_one_out_coverage_leaves_the_unit_out():
    """Five units in one stratum, four with no error and one far out. Built from
    the other four, the interval is a point and the outlier is not covered, so
    coverage is 4/5. Built from all five it would be 5/5."""
    size = np.arange(1.0, 6.0)
    errors = np.array([0.0, 0.0, 0.0, 0.0, 10.0])

    def forecast_at(k):
        return np.full(5, 100.0)

    def actual_at(k):
        return 100.0 * (1.0 + errors)

    rows = rolling_residuals(forecast_at, actual_at, cutoffs=(1,), n_periods=2, size=size)
    cal = calibrate(rows, horizons=(1,), n_strata=1, min_per_stratum=5)
    table = leave_one_out_coverage(cal, levels=(0.8,))
    assert table["empirical_coverage"].tolist() == [0.8]
    assert table["forecasts"].tolist() == [5]


def test_leave_one_out_coverage_is_near_nominal_for_a_uniform_error_law():
    rows = _synthetic_residuals(seed=3, spread=0.2)
    cal = calibrate(rows, horizons=(3, 4, 5))
    table = leave_one_out_coverage(cal, levels=(0.8, 0.95))
    assert list(table.columns) == [
        "horizon",
        "nominal_coverage",
        "empirical_coverage",
        "forecasts",
    ]
    assert set(table["horizon"]) == {3, 4, 5}
    assert (table["forecasts"] == N_UNITS).all()
    for level in (0.8, 0.95):
        sub = table[table["nominal_coverage"] == level]
        # 200 forecasts per horizon: one standard deviation of the coverage is
        # about 0.03 at 0.8, so 0.08 is a little under three
        assert sub["empirical_coverage"].between(level - 0.08, level + 0.08).all()
