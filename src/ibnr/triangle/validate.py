"""Triangle consistency checks.

Checks return human-readable issue strings (empty list = clean). ``validate``
aggregates them; with strict=True it raises on any issue. eval_date is stored,
not derived, so misalignment with origin_period + dev_lag is a *warning-level*
finding surfaced here rather than an error enforced at construction.

The operations that cannot live with that misalignment call
:func:`require_eval_alignment` themselves, which raises with the same wording
plus what the operation would otherwise have done. Ingestion stays permissive:
a triangle nobody regrains or exports is usable as it is.
"""

from __future__ import annotations

from ibis import _

from ibnr.triangle.core import GRAIN_MONTHS, Triangle, implied_dev_lag


def duplicate_cells(t: Triangle) -> list[str]:
    """A cell may appear at most once per eval_date."""
    keys = [*t.segments, "field", "origin_period", "dev_lag", "eval_date"]
    e = t.expr
    n = int(e.group_by(keys).agg(n=_.count()).filter(_.n > 1).count().execute())
    return [f"{n} duplicated (segment, field, origin, dev_lag, eval_date) cells"] if n else []


def restated_cells(t: Triangle) -> list[str]:
    """Cells observed at more than one eval_date. Legal for stored history, but
    cum<->incr conversion requires slicing with as_of()/latest_diagonal() first."""
    keys = [*t.segments, "field", "origin_period", "dev_lag"]
    e = t.expr
    n = int(e.group_by(keys).agg(n=_.eval_date.nunique()).filter(_.n > 1).count().execute())
    return [f"{n} cells observed at multiple eval_dates (restated history)"] if n else []


def misaligned_rows(expr) -> int:
    """Rows whose eval_date does not fall in the last month of origin + dev_lag.

    The one query behind both the warning-level :func:`eval_alignment` finding and
    the hard :func:`require_eval_alignment` refusal, so the two can never come to
    different answers about the same triangle.
    """
    return int(
        expr.filter(implied_dev_lag(expr.origin_period, expr.eval_date) != expr.dev_lag)
        .count()
        .execute()
    )


def _misalignment_count(n: int) -> str:
    """The one wording for how many rows disagree, written once so the finding and
    the refusal cannot drift into two sentences that say it differently."""
    return f"{n} rows where eval_date does not align with origin_period + dev_lag"


def eval_alignment(t: Triangle) -> list[str]:
    """eval_date should fall in the last month of origin_period + dev_lag."""
    n = misaligned_rows(t.expr)
    return [_misalignment_count(n)] if n else []


def require_eval_alignment(t: Triangle, *, operation: str, reason: str) -> None:
    """Refuse a triangle whose eval_date and dev_lag disagree on any row.

    Most of the package treats misalignment as the warning-level finding above:
    eval_date is stored rather than derived, so a row that disagrees with the
    convention is odd data, not an impossible state. Three operations cannot be
    that relaxed, because each one reads eval_date as the whole truth about a
    row's development and drops the stored dev_lag: coarsening the origin grain,
    and the two exports. On a misaligned row they do not fail, they answer, and
    the answer puts the row in a different cell from the one it came from. The
    origin regrain and the chainladder export add the two values together, so the
    total is preserved and nothing downstream can tell; to_bermuda keeps one of
    the two values for the field and drops the other.

    ``operation`` names the caller and ``reason`` says what it would have done,
    because a generic message would not tell the caller which of dev_lag and
    eval_date is the wrong one, or where to look for it.

    A row reaches this state in one of two ways, and the way out differs, so the
    message names both. Either dev_lag or eval_date is wrong in the source data,
    where the repair belongs, or the cell was restated at a later eval_date, which
    is legal stored history that ``validate`` reports as its own finding. Slicing
    never changes a row's eval_date, so it cannot align one; it can only drop it,
    which is what a restatement gets from ``latest_diagonal()`` or from ``as_of()``
    at a date before the restatement. ``as_of()`` at or after the restatement keeps
    the restated row and the refusal stands.
    """
    n = misaligned_rows(t.expr)
    if not n:
        return
    raise ValueError(
        f"{_misalignment_count(n)}, which {operation} cannot work with. {reason} "
        "Either dev_lag or eval_date is wrong in the source data, where the repair "
        "belongs, or the cell was restated at a later eval_date. as_of() and "
        "latest_diagonal() never change a row's eval_date, so they cannot align one; "
        "they can only drop it, which is what a restatement gets from latest_diagonal() "
        "or from as_of() at a date before the restatement, while as_of() at or after it "
        "keeps the restated row and this refusal stands. validate() reports the same "
        "rows as a finding."
    )


def dev_lag_on_grain(t: Triangle) -> list[str]:
    """dev_lag (months) should be a positive multiple of the declared dev grain."""
    step = GRAIN_MONTHS[t.meta.dev_grain]
    e = t.expr
    n = int(e.filter((e.dev_lag <= 0) | (e.dev_lag % step != 0)).count().execute())
    return [f"{n} rows where dev_lag is not a positive multiple of {step} months"] if n else []


def null_values(t: Triangle) -> list[str]:
    e = t.expr
    cols = ["origin_period", "dev_lag", "eval_date", "field"]
    n = int(e.filter(ibis_or(*(e[c].isnull() for c in cols))).count().execute())
    return [f"{n} rows with null core keys"] if n else []


def null_segments(t: Triangle) -> list[str]:
    """Segment columns must be non-null: a null segment key names no cohort.

    This is not a cosmetic finding. Every transform that equi-joins on the
    segment columns - ``as_of``, ``latest_diagonal``, ``to_cumulative``,
    ``to_incremental`` - drops such rows silently, because SQL join equality is
    false for NULL = NULL, so the cohort disappears with no error and no warning.
    ``Triangle.from_long`` refuses them at ingestion; this check is the net under
    a triangle built directly from an ibis expression, which bypasses that door.
    """
    total, per_column = null_segment_counts(t.expr, t.segments)
    if not total:
        return []
    detail = ", ".join(f"{s}={n}" for s, n in sorted(per_column.items()))
    return [f"{total} rows with a null segment key ({detail})"]


def null_segment_counts(expr, segments: list[str]) -> tuple[int, dict[str, int]]:
    """Rows with a null in any segment column, plus the per-column breakdown.

    Shared with ``io.from_long``, which refuses such rows outright at ingestion.

    The row count is taken first and on its own because ``count()`` answers 0 on
    an empty table, whereas the per-column ``sum()`` answers SQL NULL on duckdb
    and NaN on polars - an empty triangle is legitimate (every value null, or a
    filter that matched nothing) and must not be turned into a crash by the
    check meant to protect it. Computing the breakdown only when there is
    something to break down also keeps the clean case to a single query.
    """
    if not segments:
        return 0, {}
    total = int(expr.filter(ibis_or(*(expr[s].isnull() for s in segments))).count().execute())
    if not total:
        return 0, {}
    counts = expr.aggregate(**{s: expr[s].isnull().cast("int64").sum() for s in segments}).execute()
    return total, {s: int(counts[s].iloc[0]) for s in segments if int(counts[s].iloc[0])}


def ibis_or(*preds):
    out = preds[0]
    for p in preds[1:]:
        out = out | p
    return out


ALL_CHECKS = (
    null_values,
    null_segments,
    duplicate_cells,
    restated_cells,
    eval_alignment,
    dev_lag_on_grain,
)


def validate(t: Triangle, strict: bool = True) -> list[str]:
    issues = [issue for check in ALL_CHECKS for issue in check(t)]
    if strict and issues:
        raise ValueError("triangle validation failed:\n  " + "\n  ".join(issues))
    return issues
