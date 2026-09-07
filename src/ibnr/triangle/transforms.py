"""Triangle transformations, written once in ibis and run on duckdb and polars.

Every function takes and returns a Triangle. Formulations deliberately prefer
group-by + join over exotic window frames where the result is identical, because
ibis's polars backend has weaker window-function coverage than duckdb (any
divergence found by the cross-backend test suite is documented here).

Cell identity is (segments..., field, origin_period, dev_lag). Triangles are
assumed to hold at most one row per cell per eval_date; cum<->incr conversions
additionally assume one row per cell (slice with as_of()/latest_diagonal()
first if the table holds restated history).

``to_wide`` and ``change_dev_grain`` need that second assumption too, and check
it rather than assuming it: each reduces a cell's rows to one number without
being asked which observation was meant, so a restatement is counted beside the
value it replaced rather than in place of it. Both refuse a triangle storing any
cell twice (see ``validate.require_single_observation``), and there slicing IS
the answer for restated history, while duplicated source rows are a source
repair.

``change_origin_grain`` needs something else again, and checks it rather than
assuming it: eval_date must sit in the last month of origin_period + dev_lag on
every row, because the coarsened dev_lag is derived from eval_date and the
stored one is dropped. A row where the two disagree lands in a different cell
and is added to whatever is already there, so it is refused (see
``validate.require_eval_alignment``). Slicing is a partial answer only: as_of()
and latest_diagonal() pick a stored observation without changing its eval_date,
so they cannot align such a row, but they do drop a restated one, which is the
common way a triangle acquires one.

Every join below is a plain equi-join on those keys, which is only safe because
segment values are guaranteed non-null: SQL join equality is false for
NULL = NULL, so one null segment value would silently delete a whole cohort
here. That guarantee is enforced at the door by ``io.from_long`` and checked by
``validate.null_segments``; do not weaken either without making these joins
null-safe on BOTH backends first.
"""

from __future__ import annotations

import datetime as dt

import ibis
from ibis import _

from ibnr.triangle.core import GRAIN_MONTHS, Triangle, implied_dev_lag
from ibnr.triangle.validate import require_eval_alignment, require_single_observation

ORIGIN_TRUNC_UNIT = {"Y": "Y", "Q": "Q", "M": "M"}

#: what a dev-grain coarsening would do with a cell stored twice, per measure.
#: Both were measured on the same fixture (see ``change_dev_grain``); they differ
#: enough that one shared sentence would be wrong about one of them.
_DEV_GRAIN_REASON = {
    "incremental": (
        "The increments inside a bucket are summed, so a restated increment is added "
        "beside the value it replaced: a bucket whose cells are worth 105 came out as 125."
    ),
    "cumulative": (
        "A cumulative cell is kept when it lands on a bucket boundary, and the boundaries "
        "are counted back from the triangle's latest eval_date - which a restatement moves "
        "- so the row kept for a cell need not be the one that survives it, and both are "
        "kept when both land on a boundary."
    ),
}


def _cell_keys(t: Triangle) -> list[str]:
    return [*t.segments, "field", "origin_period"]


def to_cumulative(t: Triangle) -> Triangle:
    # Running sum as equi-join + filter + group-by rather than a window function:
    # ibis's polars backend has no WindowFunction translation rule at all (ibis
    # 10.x), so window formulations are duckdb-only. Triangles are small; the
    # O(n*devs) pair join is irrelevant.
    if t.meta.measure == "cumulative":
        return t
    e = t.expr
    keys = _cell_keys(t)
    other = e.select(*keys, dev2=e.dev_lag, value2=e.value)
    pairs = e.join(other, keys).filter(_.dev2 <= _.dev_lag)
    expr = pairs.group_by([*keys, "dev_lag", "eval_date"]).agg(value=_.value2.sum())
    return t.with_expr(expr, measure="cumulative")


def to_incremental(t: Triangle) -> Triangle:
    # Strict immediate-predecessor differencing, matching chainladder's
    # cum_to_incr: each cell subtracts the cell exactly one dev-grain step
    # earlier. The triangle's first dev column keeps its cumulative value
    # (increment from zero); a cell after an interior gap has an undefined
    # increment and is dropped (chainladder yields NaN there). Window lag()
    # is avoided deliberately (see to_cumulative).
    if t.meta.measure == "incremental":
        return t
    step = GRAIN_MONTHS[t.meta.dev_grain]
    e = t.expr
    keys = _cell_keys(t)
    first = e.aggregate(_first_dev=_.dev_lag.min())
    cur = e.cross_join(first)
    prev = e.select(*keys, _prev_dev=e.dev_lag, _prev_val=e.value)
    preds = [cur[k] == prev[k] for k in keys]
    preds.append(cur.dev_lag == prev["_prev_dev"] + step)
    joined = cur.left_join(prev, preds)
    new_val = (
        joined["_prev_val"]
        .notnull()
        .ifelse(
            joined.value - joined["_prev_val"],
            (joined.dev_lag == joined["_first_dev"]).ifelse(
                joined.value, ibis.null().cast("float64")
            ),
        )
    )
    kept = [c for c in e.columns if c != "value"]
    expr = joined.select(*kept, value=new_val).filter(_.value.notnull())
    return t.with_expr(expr, measure="incremental")


def as_of(t: Triangle, eval_date: dt.date | str) -> Triangle:
    """Slice the triangle as it was known at ``eval_date``.

    Drops observations after ``eval_date`` and, if a cell was restated at several
    eval dates, keeps only the latest surviving restatement.
    """
    if isinstance(eval_date, str):
        eval_date = dt.date.fromisoformat(eval_date)
    expr = t.expr.filter(t.expr.eval_date <= eval_date)
    return t.with_expr(_keep_latest_eval(expr, [*_cell_keys(t), "dev_lag"]))


def latest_diagonal(t: Triangle) -> Triangle:
    """For each cell key, keep the observation with the greatest dev_lag (and its
    latest eval_date) - the current diagonal of each sub-triangle."""
    keys = _cell_keys(t)
    expr = _keep_latest_eval(t.expr, [*keys, "dev_lag"])  # collapse restatements
    latest = expr.group_by(keys).agg(dev_lag=_.dev_lag.max())
    return t.with_expr(expr.join(latest, [*keys, "dev_lag"], how="inner"))


def _keep_latest_eval(expr, keys: list[str]):
    # keys include the segment columns, so this inner join relies on the
    # non-null-segment guarantee documented at the top of the module: a null
    # there matches nothing and the cohort is dropped without a trace.
    latest = expr.group_by(keys).agg(eval_date=_.eval_date.max())
    return expr.join(latest, [*keys, "eval_date"], how="inner")


def change_dev_grain(t: Triangle, grain: str) -> Triangle:
    """Coarsen the development grain (e.g. quarterly -> yearly).

    Buckets are anchored to the triangle's latest eval_date, matching
    chainladder's ``grain()``: if the latest diagonal sits at Q1, a yearly
    regrain keeps the Q1 valuations (ages 3, 15, 27, ...), not calendar
    year-ends. Cumulative triangles keep the cell at each bucket boundary
    (non-conforming cells are dropped); incremental triangles sum increments
    within each bucket and take the bucket's latest dev_lag/eval_date.

    Asking for the grain the triangle already has returns the same object.
    Nothing is recomputed, so nothing can move, and the check below is skipped
    for the same reason ``change_origin_grain`` skips its own on that path.

    A real coarsening needs one stored observation per cell and refuses a
    triangle carrying more, because both measures answer wrongly rather than
    failing. Measured on one quarterly bucket, both backends: incremental, the
    bucket sum counted a restated increment beside the 20 it replaced and
    answered 125 where the four cells are worth 105. Cumulative, the sum never
    happens - the boundary filter keeps rows rather than adding them - but the
    boundaries are counted back from the triangle's latest eval_date, which the
    restatement moves, so the regrain kept a superseded dev-3 value of 20 and
    dropped the 30 that had replaced it; and when a cell and its restatement are
    exactly one bucket apart both survive, leaving two rows at one age (20 and
    25) for ``to_wide`` to add up to 45. So both are refused, with the reason
    each measure actually has.
    """
    step = _grain_step(t.meta.dev_grain, grain, "dev")
    if step == 1:
        return t
    require_single_observation(
        t, operation="with_dev_grain()", reason=_DEV_GRAIN_REASON[t.meta.measure]
    )
    target = GRAIN_MONTHS[grain]
    e = t.expr
    anchor = e.aggregate(_anchor=_.eval_date.max())
    a = e.cross_join(anchor)
    # whole months this observation sits behind the latest diagonal; mutate it
    # into a real column BEFORE filtering/grouping - the ibis polars backend
    # loses cross-joined columns when predicates reference them directly
    e = a.mutate(
        _behind=a["_anchor"].year() * 12
        + a["_anchor"].month()
        - a.eval_date.year() * 12
        - a.eval_date.month()
    ).drop("_anchor")
    if t.meta.measure == "cumulative":
        expr = e.filter(_._behind % target == 0).drop("_behind")
        return t.with_expr(expr, dev_grain=grain)
    expr = (
        e.mutate(_bucket=e["_behind"] // target)
        .group_by([*_cell_keys(t), "_bucket"])
        .agg(dev_lag=_.dev_lag.max(), eval_date=_.eval_date.max(), value=_.value.sum())
        .drop("_bucket")
    )
    return t.with_expr(expr, dev_grain=grain)


def change_origin_grain(t: Triangle, grain: str) -> Triangle:
    """Coarsen the origin grain (e.g. quarterly origins -> yearly).

    Cells are aligned on eval_date: origins within the same coarser period are
    summed at each eval_date, and dev_lag is recomputed from the new origin
    start to the eval date. Valid for both cumulative and incremental triangles.

    Asking for the grain the triangle already has returns the same object, as
    ``change_dev_grain`` does. Nothing is recomputed, so nothing can move.

    A real coarsening reads eval_date as the whole truth about a row's
    development, because it derives the new dev_lag from it and drops the stored
    one. A row where the two disagree would be summed into whichever cell its
    eval_date names, which is a wrong number rather than a missing one, so such
    rows are refused by name before any of that happens. Restated history is
    refused too, because a restatement keeps its dev_lag and takes a later
    eval_date; there the answer is to slice it away with ``latest_diagonal()`` or
    an ``as_of()`` before the restatement, which the refusal says.
    """
    if _grain_step(t.meta.origin_grain, grain, "origin") == 1:
        return t
    require_eval_alignment(
        t,
        operation="with_origin_grain()",
        reason=(
            "Coarsening sums the origins inside each new period at each eval_date and "
            "derives the new dev_lag from that eval_date, so a row whose stored dev_lag "
            "says something else is added into a different cell."
        ),
    )
    e = t.expr
    new_origin = e.origin_period.truncate(ORIGIN_TRUNC_UNIT[grain])
    e = e.mutate(origin_period=new_origin)
    e = e.mutate(dev_lag=implied_dev_lag(e.origin_period, e.eval_date))
    expr = e.group_by([*_cell_keys(t), "dev_lag", "eval_date"]).agg(value=_.value.sum())
    return t.with_expr(expr, origin_grain=grain)


def _grain_step(current: str, target: str, axis: str) -> int:
    if target not in GRAIN_MONTHS:
        raise ValueError(f"unknown grain {target!r}; expected one of {sorted(GRAIN_MONTHS)}")
    cur, tgt = GRAIN_MONTHS[current], GRAIN_MONTHS[target]
    if tgt < cur:
        raise ValueError(f"cannot refine {axis} grain {current!r} -> {target!r}")
    if tgt % cur:
        raise ValueError(
            f"cannot convert {axis} grain {current!r} -> {target!r}: "
            f"{tgt} months is not a multiple of {cur}"
        )
    return tgt // cur


def to_wide(t: Triangle, field: str | None = None):
    """Materialize one field as an origin x dev_lag pandas matrix (display/export).

    The square has one axis fewer than the triangle, so the pivot adds up
    everything the triangle stores at a cell. That is refused for a cell stored
    more than once - 100 booked and 95 restated a year later displayed as one
    cell of 195, measured on both backends - and the refusal names the slice
    that leaves one view.

    The other thing the pivot adds up is deliberate and stays: the square has no
    segment axis either, so a two-line triangle displays the two lines' total.
    The check keys on the segment columns, exactly as a cell's identity does, and
    so says nothing about that sum; select or aggregate the segments you want
    before pivoting.

    The check runs on the selected field, not on the whole triangle: restated
    premium is no reason to refuse to display paid loss.
    """
    fields = t.fields
    if field is None:
        if len(fields) != 1:
            raise ValueError(f"triangle has fields {fields}; pass field=...")
        field = fields[0]
    elif field not in fields:
        raise ValueError(f"field {field!r} not in {fields}")
    one_field = t.select_fields(field)
    require_single_observation(
        one_field,
        operation="to_wide()",
        reason=(
            "The pivot adds every stored observation of a cell into one square, so a "
            "restated value is displayed on top of the value it replaced rather than in "
            "place of it."
        ),
    )
    df = one_field.expr.execute()
    if str(df["origin_period"].dtype).startswith("datetime64"):
        df["origin_period"] = df["origin_period"].dt.date
    df["dev_lag"] = df["dev_lag"].astype("int64")
    return (
        df.pivot_table(index="origin_period", columns="dev_lag", values="value", aggfunc="sum")
        .rename_axis(index="origin_period", columns="dev_lag")
        .sort_index()
    )
