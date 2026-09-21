"""kernels.point_scores: point-error metrics with the aggregation level explicit.

Errors are summed WITHIN a level before the absolute value is taken. That is
where cancellation between lines happens, and the level is named on every call
so the reader can tell a company-level Pool_APE from a line-level one.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle, gallery
from ibnr.kernels.point_scores import level_errors, point_metrics, reserve_rows, shrink_toward

from .conftest import make_cohort_triangle, make_multiline_triangle

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


# A full 4 x 4 square of cumulative paid loss; origin i = 2010 + i. The first
# development step's factors deliberately differ between origins (150/100
# against 168/110). A square whose origins all develop by the identical factor
# gives Mack a sigma of exactly zero, and then every simulated draw equals the
# deterministic ultimate - so a test that reads the draw mean where it meant to
# read the point passes, and so does the reverse.
SQUARE = np.array(
    [
        [100.0, 150.0, 175.0, 185.0],
        [110.0, 168.0, 190.0, 200.0],
        [120.0, 180.0, 210.0, 222.0],
        [130.0, 195.0, 230.0, 240.0],
    ]
)
#: the third diagonal: origins 2010, 2011 and 2012 are written, 2013 is not.
AS_OF = dt.date(2012, 12, 31)
#: the anchor at that valuation: 2010 at dev 36, 2011 at dev 24, 2012 at dev 12.
ANCHOR = SQUARE[0, 2] + SQUARE[1, 1] + SQUARE[2, 0]
#: the fit sees three development steps, so its endpoint is dev 36, not dev 48.
REALIZED = SQUARE[0, 2] + SQUARE[1, 2] + SQUARE[2, 2]


def test_mack_point_is_the_deterministic_ultimate(backend_name):
    tri = make_cohort_triangle(backend_name, SQUARE, start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of=AS_OF)
    frame = entry.point()
    assert list(frame.columns) == ["label", "origin_period", "point"]
    labels = frame["label"].astype(str).tolist()
    assert labels == entry.predict(n_draws=10).targets["label"].astype(str).tolist()
    assert labels[-1] == "total"
    np.testing.assert_allclose(frame["point"].to_numpy()[:-1], entry.fit_.ultimate)
    assert frame["point"].iloc[-1] == pytest.approx(entry.fit_.ultimate.sum())


def test_reserve_rows_for_a_single_cohort_entry(backend_name):
    tri = make_cohort_triangle(backend_name, SQUARE, start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of=AS_OF)
    rows = reserve_rows(
        entry, tri, as_of=AS_OF, loss_field="paid_loss", premium_field=None, point="native"
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["anchor"] == pytest.approx(ANCHOR)
    assert row["realized_ultimate"] == pytest.approx(REALIZED)
    assert row["actual_reserve"] == pytest.approx(row["realized_ultimate"] - row["anchor"])
    assert row["predicted_ultimate"] == pytest.approx(entry.fit_.ultimate.sum())
    assert row["predicted_reserve"] == pytest.approx(row["predicted_ultimate"] - row["anchor"])
    assert row["point_source"] == "native"
    assert np.isnan(row["premium"])

    draws = reserve_rows(
        entry,
        tri,
        as_of=AS_OF,
        loss_field="paid_loss",
        premium_field=None,
        predict_kwargs={"seed": 3, "n_draws": 2000},
    )
    assert draws["point_source"].iloc[0] == "draw_mean"
    assert draws["n_draws"].iloc[0] == 2000
    # the draw mean sits near the deterministic point but is not it
    assert draws["predicted_ultimate"].iloc[0] == pytest.approx(row["predicted_ultimate"], rel=0.05)


# Two lines of one company over five origins. SUR needs the first development
# transition to keep residual degrees of freedom, so the square is 5 x 5 rather
# than the 4 x 4 above, and the two lines are not proportional to each other.
AUTO = np.array(
    [
        [100.0, 152.0, 178.0, 194.0, 201.0],
        [110.0, 163.0, 197.0, 214.0, 223.0],
        [120.0, 184.0, 216.0, 239.0, 246.0],
        [130.0, 191.0, 233.0, 251.0, 262.0],
        [140.0, 217.0, 252.0, 274.0, 284.0],
    ]
)
LIAB = np.array(
    [
        [200.0, 291.0, 355.0, 381.0, 402.0],
        [220.0, 338.0, 393.0, 431.0, 441.0],
        [240.0, 351.0, 425.0, 462.0, 481.0],
        [260.0, 397.0, 464.0, 503.0, 518.0],
        [280.0, 419.0, 501.0, 537.0, 562.0],
    ]
)
AS_OF_ML = dt.date(2014, 12, 31)
PREMIUM_ML = {"auto": np.full(5, 1000.0), "liab": np.full(5, 3000.0)}
#: company 0002 writes five times company 0001 on both lines.
SECOND_COMPANY_SCALE = 5.0


def _two_company_triangle(backend_name):
    """Companies 0001 and 0002, two lines each, and the triangle holding both.

    Two companies rather than one on purpose: the cohort filter inside
    ``reserve_rows`` is what keeps 0002's losses out of 0001's anchor, and a
    one-company triangle cannot tell a working filter from a missing one.
    """
    first = make_multiline_triangle(
        backend_name,
        {"auto": AUTO, "liab": LIAB},
        premium_by_lob=PREMIUM_ML,
        start_year=2010,
        company="0001",
    )
    second = make_multiline_triangle(
        backend_name,
        {"auto": AUTO * SECOND_COMPANY_SCALE, "liab": LIAB * SECOND_COMPANY_SCALE},
        premium_by_lob=PREMIUM_ML,
        start_year=2010,
        company="0002",
    )
    both = pd.concat([first.execute(), second.execute()], ignore_index=True)
    return first, second, Triangle.from_long(both, measure="cumulative", backend=backend_name)


def test_reserve_rows_for_a_multi_line_entry_sums_the_company(backend_name):
    first, second, both = _two_company_triangle(backend_name)
    anchor = sum(AUTO[i, 4 - i] + LIAB[i, 4 - i] for i in range(5))
    realized = AUTO[:, 4].sum() + LIAB[:, 4].sum()

    entry = gallery.fit("sur", first, loss_field="paid_loss", as_of=AS_OF_ML)
    rows = reserve_rows(
        entry, both, as_of=AS_OF_ML, loss_field="paid_loss", predict_kwargs={"seed": 5}
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["company_code"] == "0001"
    assert row["anchor"] == pytest.approx(anchor)
    assert row["realized_ultimate"] == pytest.approx(realized)
    # premium is summed over the origins the valuation has written, on both lines
    assert row["premium"] == pytest.approx(5 * 1000.0 + 5 * 3000.0)
    assert row["point_source"] == "draw_mean"

    other = gallery.fit("sur", second, loss_field="paid_loss", as_of=AS_OF_ML)
    other_rows = reserve_rows(
        other, both, as_of=AS_OF_ML, loss_field="paid_loss", predict_kwargs={"seed": 5}
    )
    assert other_rows["company_code"].iloc[0] == "0002"
    assert other_rows["anchor"].iloc[0] == pytest.approx(SECOND_COMPANY_SCALE * anchor)
    assert other_rows["realized_ultimate"].iloc[0] == pytest.approx(SECOND_COMPANY_SCALE * realized)


def test_reserve_rows_refusals(backend_name):
    tri = make_cohort_triangle(backend_name, SQUARE, start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of=AS_OF)
    with pytest.raises(ValueError, match="no field named 'earned_premium'"):
        reserve_rows(entry, tri, as_of=AS_OF, loss_field="paid_loss")
    with pytest.raises(ValueError, match="point must be"):
        reserve_rows(
            entry, tri, as_of=AS_OF, loss_field="paid_loss", premium_field=None, point="median"
        )
    with pytest.raises(ValueError, match="no field named 'incurred_loss'"):
        reserve_rows(
            entry, tri, as_of=AS_OF, loss_field="incurred_loss", premium_field=None, point="native"
        )

    class NoPoint:
        """An entry-shaped object without point(); the native route must refuse it by name."""

        name = "stub"

        def cohorts(self):
            return entry.cohorts()

        def cohort_index(self, segment):
            return entry.cohort_index(segment)

        def predict(self, segment=None, **kw):
            return entry.predict(segment=segment, **kw)

        def realized_ultimates(self, full, segment=None):
            return entry.realized_ultimates(full, segment=segment)

    with pytest.raises(TypeError, match="does not implement point"):
        reserve_rows(
            NoPoint(), tri, as_of=AS_OF, loss_field="paid_loss", premium_field=None, point="native"
        )
