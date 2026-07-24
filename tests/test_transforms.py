"""Behavioral spec for ``triangle/transforms.py`` - the core algebra of the
Triangle layer: cumulative<->incremental, grain changes, and the ``as_of``
backtest slice.

Every test here runs twice, once per ibis backend (see ``conftest.backend_name``).
That parameterization is load-bearing rather than cosmetic: the ibis polars
backend has no window-function support, so the transforms are written as
equi-join + group-by formulations, and this suite is what proves those give the
same answer as the duckdb path. A test that passes only under ``[duckdb]``
means a window function crept back in.

Invariants pinned here:
- cum <-> incr is a lossless round trip and idempotent when already in target form;
- coarsening dev/origin grain aggregates correctly and is refuse-only in the
  refining direction (you cannot invent quarters from years);
- ``as_of`` respects restatement history: a cell restated at a later eval must
  resolve to the value that was on the books at the requested evaluation date.
"""

import pandas as pd
import pytest

from ibnr import Triangle

from .conftest import assert_triangles_equal, d, sorted_long


def test_cum_incr_round_trip(small_cumulative):
    """incr = diff(cum) cell-by-cell within an origin, and to_cumulative undoes it
    exactly - differencing must not lose or fabricate cells."""
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
    """Converting to the measure a triangle already has is a no-op - and returns
    the *same object* (``is``), so callers can convert defensively for free."""
    assert small_cumulative.to_cumulative() is small_cumulative
    incr = small_cumulative.to_incremental()
    assert incr.to_incremental() is incr


def test_as_of(small_cumulative):
    """``as_of`` yields the triangle as it stood at a past evaluation date - the
    slice every backtest trains on - and carries metadata through unchanged."""
    upper = small_cumulative.as_of("2021-12-31")
    df = sorted_long(upper)
    # 3 of the 6 cells had been evaluated by 12/31/2021: (2020, 12/24) and (2021, 12)
    assert len(df) == 3
    assert df["eval_date"].max() == pd.Timestamp("2021-12-31")
    # slicing preserves measure and grain
    assert upper.meta == small_cumulative.meta


def test_latest_diagonal(small_cumulative):
    """The latest diagonal is one cell per origin at its most recent eval - the
    paid-to-date column reserves are measured against (reserve = ultimate - this)."""
    diag = sorted_long(small_cumulative.latest_diagonal())
    assert len(diag) == 3
    expected = {36: 175.0, 24: 165.0, 12: 120.0}
    assert dict(zip(diag["dev_lag"], diag["value"], strict=True)) == expected
    assert (diag["eval_date"] == pd.Timestamp("2022-12-31")).all()


def _quarterly_dev_triangle(backend_name, measure: str) -> Triangle:
    """One origin year observed quarterly through 24 months (OYDQ).

    Built in both measures from the same cumulative ladder so the grain tests can
    check that coarsening picks the *last* value in each year bucket when
    cumulative (50 at dev 12, 70 at dev 24) but *sums* the bucket when incremental
    (50, then 20).
    """
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
        # eval_date convention (CLAUDE.md): last day of the month that
        # origin + dev_lag lands in, so dev 3 on a 1/1/2020 origin -> 3/31/2020.
        month_end = pd.Timestamp("2020-01-01") + pd.DateOffset(months=lag) - pd.Timedelta(days=1)
        rows.append(("2020-01-01", lag, month_end.date(), "paid_loss", val))
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "field", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    return Triangle.from_long(
        df, measure=measure, origin_grain="Y", dev_grain="Q", backend=backend_name
    )


def test_dev_grain_cumulative(backend_name):
    """Coarsening dev grain on a cumulative triangle keeps the last cell of each
    bucket (cumulative values must not be added up)."""
    t = _quarterly_dev_triangle(backend_name, "cumulative")
    out = t.with_dev_grain("Y")
    assert out.meta.grain == "OYDY"
    wide = out.to_wide()
    assert list(wide.columns) == [12, 24]
    assert wide.iloc[0].tolist() == [50.0, 70.0]


def test_dev_grain_incremental(backend_name):
    """Same coarsening on an incremental triangle sums each bucket, and the bucket
    inherits the latest eval_date it covers (not the earliest)."""
    t = _quarterly_dev_triangle(backend_name, "incremental")
    out = t.with_dev_grain("Y")
    wide = out.to_wide()
    assert wide.iloc[0].tolist() == [50.0, 20.0]
    # bucket eval_date is the latest eval it contains
    df = sorted_long(out)
    assert df["eval_date"].tolist() == [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-12-31")]


def test_dev_grain_errors(small_cumulative):
    """Grain changes only ever coarsen: refining Y -> Q would have to invent
    unobserved sub-annual detail, so it raises rather than interpolating."""
    with pytest.raises(ValueError, match="refine"):
        small_cumulative.with_dev_grain("Q")
    with pytest.raises(ValueError, match="unknown grain"):
        small_cumulative.with_dev_grain("W")


def test_origin_grain(backend_name):
    """Coarsening origin grain merges accident quarters into an accident year by
    *calendar diagonal*, not by dev_lag: the two Q origins sit at different lags
    (12 and 9) at the same 12/31/2020 eval, and must combine into one dev-12 cell.
    """
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
    """With restatement history in the table, ``as_of`` must return what was booked
    at that date, not the latest revision - otherwise backtests leak the future.

    ``eval_date`` is a stored first-class column precisely so this is expressible:
    the same (origin, dev) cell legitimately appears twice with different values.
    """
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
