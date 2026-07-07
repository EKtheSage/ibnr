"""Interop with Ledger Investing's bermuda (cell-based triangles).

Skipped when bermuda-ledger is not installed; the package core never imports it.
"""

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
