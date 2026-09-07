import ibis
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.triangle.validate import require_single_observation


def _null_segment_tri(backend_name) -> Triangle:
    """A two-cohort triangle whose second cohort has a null segment value.

    Built through ``from_long`` and then nulled on the expression, deliberately
    bypassing the ingestion guard: this is the state a triangle constructed
    directly as ``Triangle(expr)`` can still reach, which is exactly what the
    validate-level check exists to catch.
    """
    rows = []
    for lob in ("auto", "home"):
        for origin in (2020, 2021):
            for j, dev in enumerate((12, 24)):
                rows.append((lob, f"{origin}-01-01", dev, f"{origin + j}-12-31", 100.0))
    df = pd.DataFrame(rows, columns=["lob", "origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    t = Triangle.from_long(df, backend=backend_name)
    e = t.expr
    return t.with_expr(e.mutate(lob=(e.lob == "home").ifelse(ibis.null().cast("string"), e.lob)))


def _tri(backend_name, rows):
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    return Triangle.from_long(df, backend=backend_name)


def test_clean_triangle_passes(small_cumulative):
    assert small_cumulative.validate(strict=True) == []


def test_duplicate_cells(backend_name):
    t = _tri(
        backend_name,
        [("2020-01-01", 12, "2020-12-31", 1.0), ("2020-01-01", 12, "2020-12-31", 2.0)],
    )
    issues = t.validate(strict=False)
    assert any("duplicated" in i for i in issues)
    with pytest.raises(ValueError, match="validation failed"):
        t.validate(strict=True)


def test_restated_cells_flagged(backend_name):
    t = _tri(
        backend_name,
        [("2020-01-01", 12, "2020-12-31", 1.0), ("2020-01-01", 12, "2021-12-31", 2.0)],
    )
    issues = t.validate(strict=False)
    assert any("multiple eval_dates" in i for i in issues)
    # eval alignment also fires for the restated row, by design
    assert any("does not align" in i for i in issues)


def test_the_refusal_repeats_the_findings_word_for_word(backend_name):
    """``require_single_observation`` and ``validate`` must describe the same
    triangle in the same words.

    Each route's count phrase is written once and formatted by both, the way
    ``_misalignment_count`` is shared by the eval-alignment finding and its
    refusal. A reader who has run ``validate`` should be able to match the line
    they saw against the refusal they got, so this checks the finding strings
    appear in the message verbatim rather than merely saying something similar.
    """
    t = _tri(
        backend_name,
        [
            ("2020-01-01", 12, "2020-12-31", 1.0),
            ("2020-01-01", 12, "2021-12-31", 2.0),  # restated
            ("2020-01-01", 24, "2021-12-31", 3.0),
            ("2020-01-01", 24, "2021-12-31", 4.0),  # duplicated
        ],
    )
    issues = t.validate(strict=False)
    findings = [i for i in issues if "restated history" in i or "duplicated" in i]
    assert len(findings) == 2, issues
    with pytest.raises(ValueError) as excinfo:
        require_single_observation(t, operation="op()", reason="Because.")
    message = str(excinfo.value)
    for finding in findings:
        assert finding in message, f"{finding!r} not repeated in {message!r}"


def test_the_two_route_counts_are_counting_two_different_things(backend_name):
    """One cell duplicated at each of two eval_dates is the case where the two
    numbers in the message must differ, and the only case that says which is which.

    Duplication is counted over (cell, eval_date) groups and restatement over
    cells, so this triangle is one cell stored more than once, one restated cell
    and *two* duplicated groups. Every other fixture in the suite has the two
    counts equal, so a refusal that reported the cell count for both would pass
    everywhere else while telling a caller with two bad rows about one.
    """
    t = _tri(
        backend_name,
        [
            ("2020-01-01", 12, "2020-12-31", 1.0),
            ("2020-01-01", 12, "2020-12-31", 1.0),
            ("2020-01-01", 12, "2021-12-31", 2.0),
            ("2020-01-01", 12, "2021-12-31", 2.0),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        require_single_observation(t, operation="op()", reason="Because.")
    message = str(excinfo.value)
    assert "1 cells are stored more than once" in message
    assert "1 cells observed at multiple eval_dates" in message
    assert "2 duplicated (segment, field, origin, dev_lag, eval_date) cells" in message


def test_a_cell_stored_once_and_an_empty_triangle_both_pass(backend_name):
    """The clean cases, including the boundary that has bitten this module before.

    A per-column ``sum()`` over zero rows is SQL NULL on duckdb and NaN on
    polars, so a multiplicity check written as a sum of the extra rows crashes on
    an empty triangle - which is a legitimate object here, being what a filter
    matching nothing returns. Counting groups answers 0 on both backends, and
    this is what says so.
    """
    clean = _tri(
        backend_name,
        [("2020-01-01", 12, "2020-12-31", 1.0), ("2020-01-01", 24, "2021-12-31", 2.0)],
    )
    require_single_observation(clean, operation="op()", reason="r")
    empty = clean.filter(ibis._.dev_lag == 999)
    assert empty.count() == 0
    require_single_observation(empty, operation="op()", reason="r")


def test_eval_misalignment(backend_name):
    t = _tri(backend_name, [("2020-01-01", 12, "2021-06-30", 1.0)])
    assert any("does not align" in i for i in t.validate(strict=False))


def test_dev_lag_off_grain(backend_name):
    """Two diagonals on DIFFERENT offsets against the declared annual grain, plus a
    non-positive age.

    Mixing a 9-month age with a 12-month one means the two are not a whole number
    of dev steps apart, so ``dev_lag // 12`` floors them onto the same step. And
    dev_lag 0 is not a development age at all: dev_lag counts months from the
    origin period start, so the first annual cell is 12.

    The message is pinned whole, because every part of it is what tells a reader
    which rows to look at: the count, the anchor offset, the offsets found, and
    the evaluation date the anchor came from, written as the date the triangle
    stores rather than as a timestamp.
    """
    mixed = _tri(
        backend_name,
        [("2020-01-01", 9, "2020-09-30", 1.0), ("2020-01-01", 12, "2020-12-31", 2.0)],
    )
    assert mixed.validate(strict=False) == [
        "1 rows whose dev_lag offset against the 12-month dev grain is not 0: "
        "offsets found [0, 9], anchor taken from the latest eval_date 2020-12-31"
    ]

    zero = _tri(
        backend_name,
        [("2020-01-01", 0, "2019-12-31", 1.0), ("2020-01-01", 12, "2020-12-31", 2.0)],
    )
    assert any("not positive" in i for i in zero.validate(strict=False))


def test_the_dev_grain_anchor_comes_from_the_latest_diagonal(backend_name):
    """Which offset is the right one is decided by the LATEST evaluation date.

    The bucketing was anchored there, so that is the offset the triangle means.
    Here the latest diagonal is a March 31 with offset 3 and the two older rows
    sit at offset 0, so both of those are the rows to look at. Reading the anchor
    off the smallest offset, or off the offset most rows carry, would name the
    single newest row instead and send a reader to the wrong end of the triangle.
    """
    t = _tri(
        backend_name,
        [
            ("2019-01-01", 12, "2019-12-31", 1.0),
            ("2019-01-01", 24, "2020-12-31", 2.0),
            ("2019-01-01", 39, "2022-03-31", 3.0),
        ],
    )
    assert t.validate(strict=False) == [
        "2 rows whose dev_lag offset against the 12-month dev grain is not 3: "
        "offsets found [0, 3], anchor taken from the latest eval_date 2022-03-31"
    ]


def test_a_latest_diagonal_carrying_several_offsets_says_so(backend_name):
    """When the anchor diagonal is itself mixed, the message must not pretend it is not.

    Quarterly origins on an annual dev grain put four origins on the same
    evaluation date at four different ages, so the latest diagonal carries several
    offsets and no single one anchors the triangle. The smallest is taken so a
    count can be reported at all, and the message says the diagonal was mixed
    rather than presenting that choice as something the data decided.
    """
    t = _tri(
        backend_name,
        [
            ("2020-01-01", 3, "2020-03-31", 1.0),
            ("2020-01-01", 15, "2021-03-31", 2.0),
            ("2020-04-01", 12, "2021-03-31", 3.0),
        ],
    )
    assert t.validate(strict=False) == [
        "2 rows whose dev_lag offset against the 12-month dev grain is not 0: "
        "offsets found [0, 3], the latest eval_date 2021-03-31 carries offsets [0, 3] "
        "itself, so no single offset anchors the triangle and the smallest of them is taken"
    ]


def test_a_negative_dev_lag_is_reported_once_and_the_same_way_on_both_backends(backend_name):
    """A negative age is a sign problem, and only that.

    ``%`` follows the backend's own sign rule, so an age of -3 against an annual
    grain leaves a remainder of -3 on one backend and 9 on the other. Feeding
    negative ages to the offset rule therefore made the offsets reported for the
    SAME triangle differ by backend, and dressed a negative age up as an anchoring
    problem on top of the positivity finding it already has. Only positive ages get
    an offset now, so this triangle has exactly one finding on both backends.
    """
    t = _tri(
        backend_name,
        [("2020-01-01", -3, "2019-09-30", 1.0), ("2020-01-01", 12, "2020-12-31", 2.0)],
    )
    issues = t.validate(strict=False)
    assert "1 rows where dev_lag is not positive" in issues
    assert not any("offset" in i for i in issues)


def test_anchored_dev_ages_are_on_grain(backend_name):
    """Ages 3, 15, 27 on an ANNUAL dev grain are a clean triangle, not a finding.

    This is what chainladder's ``grain('OYDY')`` and our ``with_dev_grain('Y')``
    both produce when the latest valuation is a March 31: dev buckets are anchored
    to the latest diagonal, so the ages step by 12 months from an offset of 3
    rather than from 0. The rule is that every row shares ONE offset, not that the
    offset is zero - the old rule called chainladder's own output invalid, on all
    156 rows of its quarterly sample.
    """
    rows = [
        ("2019-01-01", 3, "2019-03-31", 1.0),
        ("2019-01-01", 15, "2020-03-31", 2.0),
        ("2019-01-01", 27, "2021-03-31", 3.0),
        ("2020-01-01", 3, "2020-03-31", 4.0),
        ("2020-01-01", 15, "2021-03-31", 5.0),
    ]
    t = _tri(backend_name, rows)
    assert t.dev_lags == [3, 15, 27]
    assert t.validate(strict=False) == []


def test_validate_handles_an_empty_triangle(backend_name):
    """An empty triangle is clean, not a crash - see the from_long counterpart."""
    t = _null_segment_tri(backend_name)
    empty = t.filter(ibis._.dev_lag == 999)
    assert empty.count() == 0
    assert empty.validate(strict=False) == []


def test_null_segments_flagged(backend_name):
    """Null segment keys are a validation finding, not a clean triangle."""
    t = _null_segment_tri(backend_name)
    issues = t.validate(strict=False)
    assert any("null segment key" in i for i in issues)
    assert any("lob" in i for i in issues)
    with pytest.raises(ValueError, match="validation failed"):
        t.validate(strict=True)


def test_null_segment_cohort_is_never_lost_silently(backend_name):
    """The invariant the whole fix exists to protect: a triangle that ``validate``
    calls clean must not lose a cohort when sliced.

    ``as_of``/``latest_diagonal`` collapse restatements with an equi-join on the
    segment columns, and SQL join equality is false for NULL = NULL - so a
    null-segment cohort vanishes with no error and no warning. This asserts the
    contract rather than either implementation of it: it passes if the triangle is
    flagged (the guard we chose) and would equally pass if the joins were instead
    made null-safe. What it refuses to allow is a clean bill of health next to a
    missing cohort.
    """
    t = _null_segment_tri(backend_name)
    before = t.execute()
    sliced = t.as_of("2021-12-31").execute()
    # cohort identity includes the null one; count distinct segment values with nulls kept
    cohorts_before = set(before["lob"].where(before["lob"].notna(), "<null>"))
    cohorts_after = set(sliced["lob"].where(sliced["lob"].notna(), "<null>"))
    lost = cohorts_before - cohorts_after
    assert t.validate(strict=False) or not lost, (
        f"as_of() silently dropped cohort(s) {lost} and validate() reported nothing"
    )
