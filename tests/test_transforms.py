import pandas as pd
import pytest

from ibnr import Triangle

from .conftest import assert_triangles_equal, d, sorted_long


def test_cum_incr_round_trip(small_cumulative):
    incr = small_cumulative.to_incremental()
    assert incr.meta.measure == "incremental"
    wide = incr.to_wide()
    assert wide.loc[d("2020-01-01"), 12] == 100.0
    assert wide.loc[d("2020-01-01"), 24] == 50.0
    assert wide.loc[d("2020-01-01"), 36] == 25.0
    assert wide.loc[d("2021-01-01"), 24] == 55.0
    back = incr.to_cumulative()
    assert back.meta.measure == "cumulative"
    assert_triangles_equal(back, small_cumulative)


def test_cum_incr_idempotent(small_cumulative):
    assert small_cumulative.to_cumulative() is small_cumulative
    incr = small_cumulative.to_incremental()
    assert incr.to_incremental() is incr


def test_as_of(small_cumulative):
    upper = small_cumulative.as_of("2021-12-31")
    df = sorted_long(upper)
    assert len(df) == 3
    assert df["eval_date"].max() == pd.Timestamp("2021-12-31")
    # slicing preserves measure and grain
    assert upper.meta == small_cumulative.meta


def test_latest_diagonal(small_cumulative):
    diag = sorted_long(small_cumulative.latest_diagonal())
    assert len(diag) == 3
    expected = {36: 175.0, 24: 165.0, 12: 120.0}
    assert dict(zip(diag["dev_lag"], diag["value"], strict=True)) == expected
    assert (diag["eval_date"] == pd.Timestamp("2022-12-31")).all()


def _quarterly_dev_triangle(backend_name, measure: str) -> Triangle:
    # one origin year, quarterly devs through 24 months
    cum = [
        (3, 10.0),
        (6, 30.0),
        (9, 45.0),
        (12, 50.0),
        (15, 58.0),
        (18, 64.0),
        (21, 67.0),
        (24, 70.0),
    ]
    rows = []
    prev = 0.0
    for lag, v in cum:
        val = v if measure == "cumulative" else v - prev
        prev = v
        month_end = pd.Timestamp("2020-01-01") + pd.DateOffset(months=lag) - pd.Timedelta(days=1)
        rows.append(("2020-01-01", lag, month_end.date(), "paid_loss", val))
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "field", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    return Triangle.from_long(
        df, measure=measure, origin_grain="Y", dev_grain="Q", backend=backend_name
    )


def test_dev_grain_cumulative(backend_name):
    t = _quarterly_dev_triangle(backend_name, "cumulative")
    out = t.with_dev_grain("Y")
    assert out.meta.grain == "OYDY"
    wide = out.to_wide()
    assert list(wide.columns) == [12, 24]
    assert wide.iloc[0].tolist() == [50.0, 70.0]


def test_dev_grain_incremental(backend_name):
    t = _quarterly_dev_triangle(backend_name, "incremental")
    out = t.with_dev_grain("Y")
    wide = out.to_wide()
    assert wide.iloc[0].tolist() == [50.0, 20.0]
    # bucket eval_date is the latest eval it contains
    df = sorted_long(out)
    assert df["eval_date"].tolist() == [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-12-31")]


def test_dev_grain_errors(small_cumulative):
    with pytest.raises(ValueError, match="refine"):
        small_cumulative.with_dev_grain("Q")
    with pytest.raises(ValueError, match="unknown grain"):
        small_cumulative.with_dev_grain("W")


def test_origin_grain(backend_name):
    # two quarterly origins in 2020, observed at two year-end evals
    rows = [
        ("2020-01-01", 12, "2020-12-31", 40.0),
        ("2020-04-01", 9, "2020-12-31", 30.0),
        ("2020-01-01", 24, "2021-12-31", 60.0),
        ("2020-04-01", 21, "2021-12-31", 45.0),
    ]
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    t = Triangle.from_long(
        df, measure="cumulative", origin_grain="Q", dev_grain="Q", backend=backend_name
    )
    out = t.with_origin_grain("Y")
    assert out.meta.grain == "OYDQ"
    df_out = sorted_long(out)
    assert df_out["origin_period"].unique().tolist() == [pd.Timestamp("2020-01-01")]
    assert dict(zip(df_out["dev_lag"], df_out["value"], strict=True)) == {12: 70.0, 24: 105.0}


def test_as_of_drops_restatements(backend_name):
    # same cell restated at a later eval: as_of keeps the latest surviving one
    rows = [
        ("2020-01-01", 12, "2020-12-31", 100.0),
        ("2020-01-01", 12, "2021-12-31", 95.0),  # restated
    ]
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    t = Triangle.from_long(df, backend=backend_name)
    assert sorted_long(t.as_of("2021-12-31"))["value"].tolist() == [95.0]
    assert sorted_long(t.as_of("2020-12-31"))["value"].tolist() == [100.0]
    assert sorted_long(t.latest_diagonal())["value"].tolist() == [95.0]
