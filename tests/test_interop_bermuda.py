"""Interop with Ledger Investing's bermuda (cell-based triangles).

Skipped when bermuda-ledger is not installed; the package core never imports it.
"""

import pandas as pd
import pytest

bermuda = pytest.importorskip("bermuda")

from ibnr import Triangle  # noqa: E402

from .conftest import assert_triangles_equal, sorted_long  # noqa: E402


@pytest.fixture(scope="module")
def meyers():
    return bermuda.meyers_tri


def test_meyers_import(meyers, backend_name):
    t = Triangle.from_bermuda(meyers, backend=backend_name)
    assert t.meta.measure == "cumulative"
    assert t.meta.grain == "OYDY"
    assert t.count() == 300  # 100 cells x 3 fields
    assert t.fields == ["earned_premium", "paid_loss", "reported_loss"]
    assert t.dev_lags == [12 * i for i in range(1, 11)]
    df = sorted_long(t)
    paid = df[df["field"] == "paid_loss"]["value"].sum()
    assert paid == pytest.approx(sum(c.values["paid_loss"] for c in meyers))


def test_meyers_round_trip(meyers, backend_name):
    t = Triangle.from_bermuda(meyers, backend=backend_name)
    back = t.to_bermuda()
    assert len(list(back)) == len(list(meyers))
    assert_triangles_equal(Triangle.from_bermuda(back, backend=backend_name), t)


def test_round_trip_with_segments(small_cumulative):
    back = small_cumulative.to_bermuda()
    cells = list(back)
    assert all(type(c).__name__ == "CumulativeCell" for c in cells)
    assert {c.details["lob"] for c in cells} == {"auto"}
    again = Triangle.from_bermuda(back)
    assert again.segments == ["lob"]
    assert_triangles_equal(again, small_cumulative)


def _month_end(origin: str, dev_lag: int):
    """Last day of the month that ``origin + dev_lag`` months lands in."""
    return (pd.Timestamp(origin) + pd.DateOffset(months=dev_lag) - pd.Timedelta(days=1)).date()


def _tri(rows, backend_name, *, dev_grain: str, origin_grain: str = "Y"):
    """Build a Triangle from a long frame, with both grains declared explicitly."""
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    return Triangle.from_long(
        df,
        measure="cumulative",
        origin_grain=origin_grain,
        dev_grain=dev_grain,
        backend=backend_name,
    )


def test_round_trip_keeps_a_finer_dev_grain(backend_name):
    """A quarterly-development triangle must come back quarterly.

    bermuda stores no dev grain, so ``from_bermuda`` used to give the dev grain the
    ORIGIN's value, and annual periods observed every quarter came back declared
    OYDY. The cells were intact and the label was wrong, which is worse: with a
    12-month step declared, ``to_incremental`` looks for each cell's predecessor 12
    months back, finds nothing, and keeps 2 of the 8 rows.
    """
    rows = [
        (o, lag, _month_end(o, lag), 10.0 * lag)
        for o in ("2019-01-01", "2020-01-01")
        for lag in (3, 6, 9, 12)
    ]
    src = _tri(rows, backend_name, dev_grain="Q")
    back = Triangle.from_bermuda(src.to_bermuda(), backend=backend_name)

    assert back.meta.grain == "OYDQ"
    assert back.validate(strict=False) == []
    assert_triangles_equal(back, src)
    assert back.to_incremental().count() == src.to_incremental().count() == 8


def test_from_bermuda_falls_back_to_the_origin_grain_on_a_single_diagonal(backend_name):
    """One evaluation date leaves bermuda with no dev resolution to report.

    Nothing in the data says how far apart the next diagonal would be, so the
    origin grain is the honest default - and it is used only here, where there is
    genuinely no answer, rather than as the general rule.

    The origins are QUARTERS on purpose. On an annual axis "fall back to the origin
    grain" and "hand back 'Y'" produce the same answer, so an annual fixture cannot
    tell the two apart; here the fallback has to say OQDQ, which no fixed answer
    can give.
    """
    rows = [
        ("2020-01-01", 6, "2020-06-30", 20.0),
        ("2020-04-01", 3, "2020-06-30", 10.0),
    ]
    src = _tri(rows, backend_name, dev_grain="Q", origin_grain="Q")
    assert src.to_bermuda().eval_date_resolution is None
    back = Triangle.from_bermuda(src.to_bermuda(), backend=backend_name)
    assert back.meta.grain == "OQDQ"


def test_from_bermuda_refuses_an_unrepresentable_dev_resolution(backend_name):
    """Six-month spacing is a real bermuda triangle and not a grain we have.

    ``GRAIN_MONTHS`` knows 12, 3 and 1 months. Rounding six months to one of them
    would mislabel every dev lag downstream, so the conversion says what it found
    and stops.
    """
    rows = [
        ("2019-01-01", 6, "2019-06-30", 10.0),
        ("2019-01-01", 12, "2019-12-31", 20.0),
    ]
    src = _tri(rows, backend_name, dev_grain="M")
    assert src.to_bermuda().eval_date_resolution == 6
    with pytest.raises(ValueError, match="dev grain"):
        Triangle.from_bermuda(src.to_bermuda(), backend=backend_name)
