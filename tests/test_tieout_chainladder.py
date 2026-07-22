"""Tie-out: the Triangle layer must reproduce chainladder-python on the public
raa / clrd samples, on both backends. Milestone 1 acceptance tests.

``ibnr`` is positioned as a companion to chainladder-python, not a fork, so
numbers that both libraries can compute must agree exactly — otherwise no
comparison downstream (Mack baselines, ClarkLDF, the gallery leaderboard) means
anything. chainladder is the reference implementation; where we differ it is on
purpose and the difference is asserted here rather than tolerated.

Two deliberate divergences are encoded below:
- **End-of-day valuations.** chainladder valuation timestamps land at the end of
  the day, so ``tri[tri.valuation <= "1985-12-31"]`` *excludes* the 1985 diagonal.
  Our ``as_of("1985-12-31")`` includes it; the equivalent chainladder slice is
  ``< "1986"`` (see test_raa_as_of_matches_valuation_slice).
- **Sparsity.** chainladder's 4D arrays treat missing cells as zero inside the
  observed region and store zero increments as NaN. Our long format says absent =
  unobserved, zero = explicitly observed, and never densifies. test_clrd_cum_to_incr
  ties out modulo exactly that difference.

Marked ``tieout``: run with ``pytest -m tieout``. Requires chainladder-python,
which is a dev dependency (interop is sacred), not an optional extra.
"""

import chainladder as cl
import numpy as np
import pytest

from ibnr import Triangle

from .conftest import assert_triangles_equal, sorted_long

pytestmark = pytest.mark.tieout


@pytest.fixture(scope="module")
def raa():
    """Reinsurance Association of America: the canonical 10x10 single-triangle
    sample. Module-scoped — loading is pure I/O and the object is never mutated."""
    return cl.load_sample("raa")


@pytest.fixture(scope="module")
def clrd():
    """CAS Loss Reserve Database: many companies x lines x fields. Exercises the
    multi-segment / multi-field paths that raa cannot."""
    return cl.load_sample("clrd")


def cl_total(tri) -> float:
    """Grand total over a chainladder 4D array, ignoring the NaN padding outside
    the observed region — the one summary both representations must agree on."""
    return float(np.nansum(tri.values))


def test_raa_import(raa, backend_name):
    """from_chainladder preserves grain, measure, cell count and total: the import
    neither drops observed cells nor materializes the NaN padding as rows."""
    t = Triangle.from_chainladder(raa, backend=backend_name)
    assert t.meta.grain == "OYDY"
    assert t.meta.measure == "cumulative"
    # 10 origins with 10, 9, ... 1 observed devs = 55 cells; the 45 NaN padding
    # cells of the 10x10 array must NOT become rows.
    assert t.count() == 55
    assert t.validate(strict=True) == []
    df = sorted_long(t)
    assert df["value"].sum() == pytest.approx(cl_total(raa))


def test_raa_round_trip(raa, backend_name):
    """chainladder -> Triangle -> chainladder is lossless, down to grain flags and
    cell values. Interop is sacred (CLAUDE.md): users must be able to move a
    triangle between the two libraries without silent degradation."""
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
    """Our latest_diagonal equals chainladder's: same number of cells, same total
    paid-to-date. Compared on the aggregate rather than cell-by-cell because
    chainladder's frame carries its own index/ordering."""
    t = Triangle.from_chainladder(raa, backend=backend_name)
    ours = sorted_long(t.latest_diagonal())["value"]
    theirs = raa.latest_diagonal.to_frame(keepdims=True).reset_index()["values"]
    assert ours.sum() == pytest.approx(theirs.sum())
    assert len(ours) == len(theirs)


def test_raa_cum_to_incr(raa, backend_name):
    """Our differencing matches chainladder's ``cum_to_incr`` exactly on raa.

    raa is a dense square triangle with no interior gaps and no zero increments,
    so the sparsity conventions never bite here — this is the clean case, and
    equality is expected to be total. (The messy case is test_clrd_cum_to_incr.)
    """
    ours = Triangle.from_chainladder(raa, backend=backend_name).to_incremental()
    theirs = Triangle.from_chainladder(raa.cum_to_incr(), backend=backend_name)
    assert_triangles_equal(ours, theirs)


def test_raa_incr_to_cum(raa, backend_name):
    """The reverse direction: our running sum matches chainladder's ``incr_to_cum``.
    Exercises the equi-join formulation that stands in for a window function."""
    incr = raa.cum_to_incr()
    ours = Triangle.from_chainladder(incr, backend=backend_name).to_cumulative()
    theirs = Triangle.from_chainladder(incr.incr_to_cum(), backend=backend_name)
    assert_triangles_equal(ours, theirs)


def test_raa_as_of_matches_valuation_slice(raa, backend_name):
    """``as_of(date)`` is the same backtest slice as chainladder's valuation filter
    — the gotcha being that the two spell the same cutoff differently."""
    t = Triangle.from_chainladder(raa, backend=backend_name)
    ours = t.as_of("1985-12-31")
    # chainladder valuations are end-of-day timestamps (1985-12-31 23:59:59.999),
    # so the intuitive `<= "1985-12-31"` compares against midnight and silently
    # EXCLUDES the whole 1985 diagonal. `< "1986"` is the equivalent slice, and
    # is what any backtest against chainladder must use.
    theirs = Triangle.from_chainladder(raa[raa.valuation < "1986"], backend=backend_name)
    assert_triangles_equal(ours, theirs)


def test_clrd_import_multi_segment(clrd, backend_name):
    """Multi-segment import: chainladder's index levels (GRNAME company, LOB) become
    ordinary segment columns and its columns become ``field`` values, with the grand
    total preserved. This is the shape the Schedule P mart arrives in."""
    t = Triangle.from_chainladder(clrd, backend=backend_name)
    assert set(t.segments) == {"GRNAME", "LOB"}
    assert set(t.fields) == {str(c) for c in clrd.columns}
    df = sorted_long(t)
    assert df["value"].sum() == pytest.approx(cl_total(clrd))


def test_clrd_round_trip(clrd, backend_name):
    """Round trip survives many segments and fields. Shape is checked directly, but
    equality goes through the long form because chainladder's index/column ordering
    is not guaranteed to be reconstructed."""
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

    Restricted to workers' comp purely to keep the outer merge tractable; the
    conventions being tested are not line-specific.
    """
    sub = clrd[clrd["LOB"] == "wkcomp"]
    source = Triangle.from_chainladder(sub, backend=backend_name)
    ours = sorted_long(source.to_incremental())
    theirs = sorted_long(Triangle.from_chainladder(sub.cum_to_incr(), backend=backend_name))
    keys = [c for c in ours.columns if c != "value"]
    merged = ours.merge(theirs, on=keys, how="outer", suffixes=("_ours", "_cl"), indicator=True)

    # Cells both sides agree exist: values must be identical. The >10k floor is a
    # guard against the merge silently degenerating (e.g. a key dtype mismatch
    # sending every row to left_only/right_only, which would make the two
    # one-sided assertions below vacuously true).
    both = merged[merged["_merge"] == "both"]
    assert len(both) > 10_000
    np.testing.assert_allclose(both["value_ours"], both["value_cl"], atol=1e-8)

    # Ours only: chainladder stored these as NaN because the increment is zero, and
    # the long-format import drops NaN. So every such cell must be exactly zero.
    ours_only = merged[merged["_merge"] == "left_only"]
    np.testing.assert_allclose(ours_only["value_ours"], 0.0, atol=1e-8)

    # Theirs only: chainladder fabricated an increment by treating a missing cell as
    # zero. Each must therefore involve a hole in the cumulative source — either the
    # cell itself or its dev-12-months predecessor was never observed.
    cell_keys = ["GRNAME", "LOB", "field", "origin_period", "dev_lag"]
    src = sorted_long(source)
    present = set(map(tuple, src[cell_keys].itertuples(index=False)))
    theirs_only = merged[merged["_merge"] == "right_only"]
    for row in theirs_only[cell_keys].itertuples(index=False):
        cell = tuple(row)
        prev_cell = (*cell[:-1], cell[-1] - 12)
        assert cell not in present or prev_cell not in present, cell


def test_quarterly_dev_grain(backend_name):
    """``with_dev_grain("Y")`` matches chainladder's ``grain("OYDY")``, including its
    anchoring rule: dev buckets are anchored to the LATEST diagonal, so a triangle
    whose latest valuation is Q1 yields ages 3, 15, 27, ... not 12, 24, 36."""
    q = cl.load_sample("quarterly")
    ours = Triangle.from_chainladder(q, backend=backend_name).with_dev_grain("Y")
    theirs = Triangle.from_chainladder(q.grain("OYDY"), backend=backend_name)
    assert_triangles_equal(ours, theirs)
