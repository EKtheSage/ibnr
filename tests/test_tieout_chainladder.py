"""Tie-out: the Triangle layer must reproduce chainladder-python on the public
raa / clrd samples, on both backends. Milestone 1 acceptance tests."""

import chainladder as cl
import numpy as np
import pytest

from ibnr import Triangle

from .conftest import assert_triangles_equal, sorted_long

pytestmark = pytest.mark.tieout


@pytest.fixture(scope="module")
def raa():
    return cl.load_sample("raa")


@pytest.fixture(scope="module")
def clrd():
    return cl.load_sample("clrd")


def cl_total(tri) -> float:
    return float(np.nansum(tri.values))


def test_raa_import(raa, backend_name):
    t = Triangle.from_chainladder(raa, backend=backend_name)
    assert t.meta.grain == "OYDY"
    assert t.meta.measure == "cumulative"
    assert t.count() == 55
    assert t.validate(strict=True) == []
    df = sorted_long(t)
    assert df["value"].sum() == pytest.approx(cl_total(raa))


def test_raa_round_trip(raa, backend_name):
    t = Triangle.from_chainladder(raa, backend=backend_name)
    back = t.to_chainladder()
    assert back.shape == raa.shape
    assert back.origin_grain == raa.origin_grain
    assert back.development_grain == raa.development_grain
    assert back.is_cumulative == raa.is_cumulative
    np.testing.assert_allclose(back.values, raa.values)
    # and the long forms agree exactly
    assert_triangles_equal(Triangle.from_chainladder(back, backend=backend_name), t)


def test_raa_latest_diagonal(raa, backend_name):
    t = Triangle.from_chainladder(raa, backend=backend_name)
    ours = sorted_long(t.latest_diagonal())["value"]
    theirs = raa.latest_diagonal.to_frame(keepdims=True).reset_index()["values"]
    assert ours.sum() == pytest.approx(theirs.sum())
    assert len(ours) == len(theirs)


def test_raa_cum_to_incr(raa, backend_name):
    ours = Triangle.from_chainladder(raa, backend=backend_name).to_incremental()
    theirs = Triangle.from_chainladder(raa.cum_to_incr(), backend=backend_name)
    assert_triangles_equal(ours, theirs)


def test_raa_incr_to_cum(raa, backend_name):
    incr = raa.cum_to_incr()
    ours = Triangle.from_chainladder(incr, backend=backend_name).to_cumulative()
    theirs = Triangle.from_chainladder(incr.incr_to_cum(), backend=backend_name)
    assert_triangles_equal(ours, theirs)


def test_raa_as_of_matches_valuation_slice(raa, backend_name):
    t = Triangle.from_chainladder(raa, backend=backend_name)
    ours = t.as_of("1985-12-31")
    # chainladder valuations are end-of-day timestamps, so `<= "1985-12-31"`
    # would exclude the 1985 diagonal; `< "1986"` is the equivalent slice
    theirs = Triangle.from_chainladder(raa[raa.valuation < "1986"], backend=backend_name)
    assert_triangles_equal(ours, theirs)


def test_clrd_import_multi_segment(clrd, backend_name):
    t = Triangle.from_chainladder(clrd, backend=backend_name)
    assert set(t.segments) == {"GRNAME", "LOB"}
    assert set(t.fields) == {str(c) for c in clrd.columns}
    df = sorted_long(t)
    assert df["value"].sum() == pytest.approx(cl_total(clrd))


def test_clrd_round_trip(clrd, backend_name):
    t = Triangle.from_chainladder(clrd, backend=backend_name)
    back = t.to_chainladder()
    assert back.shape == clrd.shape
    # column/index order can differ; compare via the long form
    assert_triangles_equal(Triangle.from_chainladder(back, backend=backend_name), t)


def test_clrd_cum_to_incr(clrd, backend_name):
    """Ties out modulo chainladder's sparsity conventions, which the long format
    deliberately does not share (absent row = unobserved, zero = explicit):

    - chainladder stores zero increments as NaN, which the long format drops on
      import -> cells only on our side must be exactly the zero increments;
    - chainladder's triangle arithmetic treats missing cells as zero inside the
      observed region, fabricating increments where the cumulative source has no
      cell -> cells only on its side must touch a missing source cell.
    """
    sub = clrd[clrd["LOB"] == "wkcomp"]
    source = Triangle.from_chainladder(sub, backend=backend_name)
    ours = sorted_long(source.to_incremental())
    theirs = sorted_long(Triangle.from_chainladder(sub.cum_to_incr(), backend=backend_name))
    keys = [c for c in ours.columns if c != "value"]
    merged = ours.merge(theirs, on=keys, how="outer", suffixes=("_ours", "_cl"), indicator=True)

    both = merged[merged["_merge"] == "both"]
    assert len(both) > 10_000
    np.testing.assert_allclose(both["value_ours"], both["value_cl"], atol=1e-8)

    ours_only = merged[merged["_merge"] == "left_only"]
    np.testing.assert_allclose(ours_only["value_ours"], 0.0, atol=1e-8)

    cell_keys = ["GRNAME", "LOB", "field", "origin_period", "dev_lag"]
    src = sorted_long(source)
    present = set(map(tuple, src[cell_keys].itertuples(index=False)))
    theirs_only = merged[merged["_merge"] == "right_only"]
    for row in theirs_only[cell_keys].itertuples(index=False):
        cell = tuple(row)
        prev_cell = (*cell[:-1], cell[-1] - 12)
        assert cell not in present or prev_cell not in present, cell


def test_quarterly_dev_grain(backend_name):
    q = cl.load_sample("quarterly")
    ours = Triangle.from_chainladder(q, backend=backend_name).with_dev_grain("Y")
    theirs = Triangle.from_chainladder(q.grain("OYDY"), backend=backend_name)
    assert_triangles_equal(ours, theirs)
