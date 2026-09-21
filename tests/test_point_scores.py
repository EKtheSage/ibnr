"""kernels.point_scores: point-error metrics with the aggregation level explicit.

Errors are summed WITHIN a level before the absolute value is taken. That is
where cancellation between lines happens, and the level is named on every call
so the reader can tell a company-level Pool_APE from a line-level one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.point_scores import level_errors, point_metrics, shrink_toward

# Company 671 from the 2026-09-20 reconciliation, USD thousands. The three
# overestimates almost offset the personal-auto underestimate.
COMPANY_671 = pd.DataFrame(
    {
        "company_code": ["671"] * 4,
        "line_of_business": [
            "commercial_auto",
            "other_liability",
            "private_passenger_auto",
            "workers_compensation",
        ],
        "predicted": [9761.03, 2336.71, 92807.64, 27312.97],
        "actual": [9746.0, 2009.0, 93662.0, 26811.0],
    }
)


def test_company_level_sums_before_the_absolute_value():
    """Company error is the sum of signed line errors (-9.65); the sum of absolute
    line errors is 1699.07. Swapping the order of sum and abs must fail this."""
    company = level_errors(
        COMPANY_671, predicted="predicted", actual="actual", level=["company_code"]
    )
    assert list(company.columns) == ["company_code", "n_rows", "predicted", "actual", "error"]
    assert len(company) == 1
    assert company["n_rows"].iloc[0] == 4
    assert company["error"].iloc[0] == pytest.approx(-9.65, abs=0.01)
    pair = level_errors(
        COMPANY_671,
        predicted="predicted",
        actual="actual",
        level=["company_code", "line_of_business"],
    )
    assert len(pair) == 4
    assert pair["error"].abs().sum() == pytest.approx(1699.07, abs=0.01)


def test_level_errors_refuses_a_missing_level_column_and_a_missing_forecast():
    with pytest.raises(KeyError, match="level column"):
        level_errors(COMPANY_671, predicted="predicted", actual="actual", level=["state"])
    holed = COMPANY_671.assign(predicted=[9761.03, np.nan, 92807.64, 27312.97])
    with pytest.raises(ValueError, match="non-finite"):
        level_errors(holed, predicted="predicted", actual="actual", level=["company_code"])


def test_point_metrics_closed_forms():
    m = point_metrics([110.0, 90.0], [100.0, 100.0])
    assert m["n"] == 2
    assert m["mae"] == pytest.approx(10.0)
    assert m["rmse"] == pytest.approx(10.0)
    assert m["wrmse"] == pytest.approx(10.0)
    assert m["mape"] == pytest.approx(0.1)
    assert m["medape"] == pytest.approx(0.1)
    assert m["pool_ape"] == pytest.approx(0.10)
    assert m["pool_pe"] == pytest.approx(0.0)
    assert m["prop_over"] == pytest.approx(0.5)


def test_pool_ape_weights_by_reserve_size_and_mape_does_not():
    """Pool_APE is a dollar-weighted ratio: a 1 unit miss on a 1 unit reserve
    beside a perfect 99 unit reserve is 1 percent, while MAPE calls it 50 percent."""
    m = point_metrics([2.0, 99.0], [1.0, 99.0])
    assert m["pool_ape"] == pytest.approx(0.01)
    assert m["mape"] == pytest.approx(0.5)


def test_mape_and_medape_skip_zero_actuals_and_pool_ape_does_not():
    m = point_metrics([5.0, 3.0], [0.0, 2.0])
    assert m["mape"] == pytest.approx(0.5)
    assert m["medape"] == pytest.approx(0.5)
    assert m["pool_ape"] == pytest.approx(3.0)


def test_wrmse_weights_squared_errors_by_actual():
    """wRMSE = sqrt(sum(actual * e^2) / sum(actual)), the R study's definition."""
    predicted, actual = np.array([12.0, 100.0]), np.array([10.0, 90.0])
    m = point_metrics(predicted, actual)
    want = np.sqrt((10.0 * 4.0 + 90.0 * 100.0) / 100.0)
    assert m["wrmse"] == pytest.approx(want)


def test_point_metrics_refusals():
    with pytest.raises(ValueError, match="non-finite"):
        point_metrics([1.0, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        point_metrics([1.0, 1.0], [1.0, np.inf])
    with pytest.raises(ValueError, match="non-positive"):
        point_metrics([1.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="non-positive"):
        point_metrics([1.0, 1.0], [-3.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        point_metrics([1.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="at least one"):
        point_metrics([], [])


def test_shrink_toward_endpoints_and_the_published_weight():
    point, baseline = np.array([120.0, 80.0]), np.array([100.0, 100.0])
    np.testing.assert_allclose(shrink_toward(point, baseline, 0.0), baseline)
    np.testing.assert_allclose(shrink_toward(point, baseline, 1.0), point)
    np.testing.assert_allclose(shrink_toward(point, baseline, 0.658), [113.16, 86.84])


def test_shrink_toward_refusals():
    with pytest.raises(ValueError, match="alpha"):
        shrink_toward([1.0], [1.0], 1.5)
    with pytest.raises(ValueError, match="same shape"):
        shrink_toward([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(ValueError, match="non-finite"):
        shrink_toward([np.nan], [1.0], 0.5)
