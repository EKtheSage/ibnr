import ibis
import pandas as pd
import pytest

from ibnr import Triangle


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


def test_eval_misalignment(backend_name):
    t = _tri(backend_name, [("2020-01-01", 12, "2021-06-30", 1.0)])
    assert any("does not align" in i for i in t.validate(strict=False))


def test_dev_lag_off_grain(backend_name):
    t = _tri(backend_name, [("2020-01-01", 9, "2020-09-30", 1.0)])
    assert any("multiple of 12 months" in i for i in t.validate(strict=False))


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
