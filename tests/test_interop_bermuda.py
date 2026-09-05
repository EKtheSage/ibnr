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


def test_to_bermuda_refuses_misaligned_rows(backend_name):
    """A bermuda cell is (period, evaluation date) with one value per field, and
    ``from_bermuda`` derives dev_lag from the evaluation date on the way back. So
    rows sharing an eval_date are one cell: the pair below, 95 at dev 12 and 150
    at dev 24 both carried at 12/31/2021, used to become a single cell whose
    ``paid_loss`` is whichever of the two pandas grouped last. The Meyers round
    trip above is the aligned control.
    """
    rows = [
        ("2020-01-01", 12, "2021-12-31", 95.0),  # eval_date says dev 24, not 12
        ("2020-01-01", 24, "2021-12-31", 150.0),
    ]
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    t = Triangle.from_long(df, measure="cumulative", backend=backend_name)
    with pytest.raises(ValueError, match="does not align"):
        t.to_bermuda()


def test_round_trip_with_segments(small_cumulative):
    back = small_cumulative.to_bermuda()
    cells = list(back)
    assert all(type(c).__name__ == "CumulativeCell" for c in cells)
    assert {c.details["lob"] for c in cells} == {"auto"}
    again = Triangle.from_bermuda(back)
    assert again.segments == ["lob"]
    assert_triangles_equal(again, small_cumulative)
