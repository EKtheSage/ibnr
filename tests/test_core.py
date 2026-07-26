import datetime as dt

import pandas as pd
import pytest

from ibnr import Triangle, TriangleMeta


def test_meta_validation():
    assert TriangleMeta().grain == "OYDY"
    assert TriangleMeta(origin_grain="Y", dev_grain="Q").grain == "OYDQ"
    with pytest.raises(ValueError):
        TriangleMeta(origin_grain="W")
    with pytest.raises(ValueError):
        TriangleMeta(measure="paid")


def test_construction_and_introspection(small_cumulative):
    t = small_cumulative
    assert t.segments == ["lob"]
    assert t.fields == ["paid_loss"]
    assert t.origins == [dt.date(2020, 1, 1), dt.date(2021, 1, 1), dt.date(2022, 1, 1)]
    assert t.dev_lags == [12, 24, 36]
    assert t.eval_dates[-1] == dt.date(2022, 12, 31)
    assert t.count() == 6
    assert "OYDY" in repr(t) and "cumulative" in repr(t)


def test_missing_columns_rejected(small_cumulative):
    with pytest.raises(ValueError, match="missing core columns"):
        Triangle(small_cumulative.expr.drop("eval_date"))


def _null_segment_frame() -> pd.DataFrame:
    """Two cohorts, one of which has no segment value at all.

    The shape of a real gold-mart accident: ``dim_company.company_name`` is built
    with a LEFT JOIN onto ``sat_company_details``, so a company present in the hub
    but missing from the satellite arrives here with a null name - and
    ``company_name`` is one of the three Schedule P segment columns.
    """
    rows = []
    for lob in ("auto", None):
        for origin in (2020, 2021):
            for j, dev in enumerate((12, 24)):
                rows.append((lob, dt.date(origin, 1, 1), dev, dt.date(origin + j, 12, 31), 100.0))
    df = pd.DataFrame(rows, columns=["lob", "origin_period", "dev_lag", "eval_date", "value"])
    df["field"] = "paid_loss"
    return df


def test_from_long_rejects_null_segment(backend_name):
    """A null segment key must fail at the door.

    It identifies no cohort, and every transform that equi-joins on the segment
    columns (``as_of``, ``latest_diagonal``, ``to_cumulative``, ``to_incremental``)
    silently drops such rows, because SQL join equality is false for NULL = NULL.
    Losing a whole cohort with no error is strictly worse than refusing the data.
    """
    with pytest.raises(ValueError, match="null segment key"):
        Triangle.from_long(_null_segment_frame(), backend=backend_name)


def test_from_long_null_segment_error_names_the_column(backend_name):
    """The error has to say which column and how many rows, or it cannot be acted on."""
    with pytest.raises(ValueError) as excinfo:
        Triangle.from_long(_null_segment_frame(), backend=backend_name)
    message = str(excinfo.value)
    assert "lob" in message
    assert "4" in message  # 4 of the 8 rows carry the null segment


def test_from_long_allows_an_empty_triangle(backend_name):
    """A frame with no surviving observations is an empty triangle, not an error.

    Reachable whenever every value is null (chainladder's padding half, a field
    absent for one segment) or a filter matches nothing. It is a boundary the
    segment check has to survive rather than define: aggregating over zero rows
    returns SQL NULL on duckdb and NaN on polars, so a guard that counts nulls
    without allowing for that turns a legitimate empty load into a crash.
    """
    df = pd.DataFrame(
        {
            "lob": pd.Series(["auto"], dtype="string"),
            "origin_period": [dt.date(2020, 1, 1)],
            "dev_lag": [12],
            "eval_date": [dt.date(2020, 12, 31)],
            "field": ["paid_loss"],
            "value": [float("nan")],
        }
    )
    t = Triangle.from_long(df, backend=backend_name)
    assert t.count() == 0
    assert t.segments == ["lob"]


def test_select_fields_and_filter(small_cumulative):
    import ibis

    assert small_cumulative.select_fields("paid_loss").count() == 6
    assert small_cumulative.filter(ibis._.dev_lag == 12).count() == 3


def test_to_wide(small_cumulative):
    wide = small_cumulative.to_wide()
    assert wide.shape == (3, 3)
    assert wide.loc[dt.date(2020, 1, 1), 36] == 175.0
    assert wide.loc[dt.date(2022, 1, 1), 12] == 120.0
