"""Triangle ingestion and interop.

Every constructor in this module funnels into :func:`from_long`, which is the
one place that normalizes column names, unpivots wide measure columns, converts
dev-lag units and attaches :class:`~ibnr.triangle.core.TriangleMeta`. The two
interop pairs below are thin adapters that reshape a foreign triangle into that
same long frame - they never build a ``Triangle`` directly.

Interop is sacred (CLAUDE.md): ``from_chainladder``/``to_chainladder`` and
``from_bermuda``/``to_bermuda`` must round-trip losslessly (modulo NaN padding
cells, which chainladder materializes and the long format simply omits).

Two conventions differ from the foreign libraries and are handled here, once:

* **bermuda dev_lag origin.** bermuda's ``Cell.dev_lag`` measures months from
  the period *end*, so the first diagonal is 0. Ours measures months from the
  origin *start*, so the first diagonal is 12 (annual grain).
  :func:`from_bermuda` therefore **recomputes** dev_lag from
  ``period_start`` and ``evaluation_date`` and never reads bermuda's value;
  :func:`to_bermuda` symmetrically emits ``period_start``/``period_end`` and
  lets bermuda derive its own dev_lag.
* **chainladder valuation timestamps are end-of-day.** A chainladder
  ``valuation`` is a Timestamp at the end of the valuation period, so filters
  like ``tri[tri.valuation <= "1985-12-31"]`` *exclude* the 1985 diagonal (use
  ``< "1986"``). We store ``eval_date`` as a plain ``datetime.date`` at the
  last day of the period, so both converters normalize the datetime64 columns
  to dates and back rather than carrying timestamps across the boundary.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from uuid import uuid4

import ibis
from ibis import selectors as s
from ibis.backends import BaseBackend
from ibis.expr.types import Table as IbisTable

from ibnr.triangle.core import CORE_COLUMNS, GRAIN_MONTHS, Triangle, TriangleMeta
from ibnr.triangle.validate import null_segment_counts, require_eval_alignment


def _require_interop(module: str, feature: str):
    """Import an optional interop dependency, or say how to install it.

    ``chainladder`` and ``bermuda`` are not core dependencies: they are only
    needed to convert *out* to those libraries, and they live behind the
    ``interop`` extra. Importing them lazily keeps ``import ibnr`` working on a
    bare install; this wrapper makes the failure legible when the conversion is
    actually called.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial branch
        raise ModuleNotFoundError(
            f"{feature} requires the optional '{module}' package, which is not installed. "
            f'Install the interop extra:  pip install "ibnr[interop]"  '
            f'(or: uv add "ibnr[interop]")'
        ) from exc


def resolve_backend(backend: str | BaseBackend | None = None) -> BaseBackend:
    """'duckdb' (default) and 'polars' are the two supported names; an already
    connected ibis backend passes through.

    Both backends are first-class by design (CLAUDE.md #2) - duckdb is the
    default because ibis's polars backend has no window-function support, which
    the transforms work around but which makes duckdb the faster path.

    Only duckdb is installed by default: the polars backend brings a ~176 MB
    runtime that a duckdb-only user never executes, so it sits behind the
    ``polars`` extra. Asking for it without that extra says so.
    """
    if backend is None or backend == "duckdb":
        return ibis.duckdb.connect()
    if backend == "polars":
        try:
            return ibis.polars.connect()
        except Exception as exc:
            if _polars_installed():
                raise
            raise ModuleNotFoundError(
                "backend='polars' requires the optional polars backend, which is not "
                'installed. Install the polars extra:  pip install "ibnr[polars]"  '
                '(or: uv add "ibnr[polars]"). The default duckdb backend needs nothing extra.'
            ) from exc
    if isinstance(backend, BaseBackend):
        return backend
    raise ValueError(f"backend must be 'duckdb', 'polars', or an ibis backend, got {backend!r}")


def _polars_installed() -> bool:
    """Whether the polars backend is actually importable.

    Used to tell "polars is missing" apart from "polars is here and genuinely
    failed", so a real backend error is never masked by the install hint.
    """
    import importlib.util

    return importlib.util.find_spec("polars") is not None


def _register(con: BaseBackend, data) -> IbisTable:
    """Materialize an in-memory frame (pandas/polars/pyarrow) as a backend table.

    The uuid suffix keeps repeated ingestion on one shared connection from
    colliding on table names.
    """
    return con.create_table(f"triangle_{uuid4().hex[:8]}", data)


def from_long(
    data,
    *,
    origin: str = "origin_period",
    dev: str = "dev_lag",
    eval_date: str = "eval_date",
    field: str = "field",
    value: str = "value",
    fields: list[str] | None = None,
    segments: list[str] | None = None,
    measure: str = "cumulative",
    origin_grain: str = "Y",
    dev_grain: str = "Y",
    dev_lag_unit: str = "months",
    units: str | None = None,
    backend: str | BaseBackend | None = None,
) -> Triangle:
    """Build a Triangle from long-format data.

    ``data`` may be a pandas/polars/pyarrow object, an ibis table, or a path to
    a parquet file. Either the data already has ``field``/``value`` columns
    (stacked-long), or ``fields=[...]`` names measure columns to unpivot.
    ``dev_lag_unit`` is ``"months"`` or ``"periods"`` (multiples of dev_grain).
    ``segments`` restricts which extra columns are kept; default keeps all.

    This is the single ingestion path: the chainladder/bermuda adapters build a
    long pandas frame and hand it here so name normalization, unit conversion
    and null dropping happen exactly once.
    """
    if isinstance(data, IbisTable):
        t = data  # already an ibis expr: keep its backend, do not re-register
    else:
        con = resolve_backend(backend)
        is_path = isinstance(data, str | Path)
        t = con.read_parquet(str(data)) if is_path else _register(con, data)

    # Normalize caller column names to the canonical schema. field/value only
    # exist in stacked-long input; with fields=[...] they are created by the
    # unpivot below, so renaming them would fail.
    renames = {"origin_period": origin, "dev_lag": dev, "eval_date": eval_date}
    if fields is None:
        renames |= {"field": field, "value": value}
    renames = {new: old for new, old in renames.items() if new != old}
    if renames:
        t = t.rename(**renames)

    if fields is not None:
        # Wide input: everything that is not a measure column is an identifier.
        keep = segments if segments is not None else None
        id_cols = [c for c in t.columns if c not in fields]
        if keep is not None:
            # Drop unrequested segment columns before the unpivot, but never the
            # three key columns that define a cell.
            id_cols = [c for c in id_cols if c in (*keep, "origin_period", "dev_lag", "eval_date")]
            t = t.select(*id_cols, *fields)
        t = t.pivot_longer(s.cols(*fields), names_to="field", values_to="value")
    elif segments is not None:
        t = t.select(*segments, "origin_period", "dev_lag", "eval_date", "field", "value")

    # dev_lag is stored in months always (CLAUDE.md milestone 1), so "periods"
    # input (1, 2, 3... dev years/quarters) is scaled by the dev grain here.
    if dev_lag_unit == "periods":
        t = t.mutate(dev_lag=t.dev_lag.cast("int64") * GRAIN_MONTHS[dev_grain])
    elif dev_lag_unit != "months":
        raise ValueError(f"dev_lag_unit must be 'months' or 'periods', got {dev_lag_unit!r}")

    # Long format: absent means unobserved. Nulls (e.g. chainladder's lower-half
    # padding, or a field missing for one segment) are dropped rather than
    # stored, so the triangle never carries fabricated cells.
    t = t.filter(t.value.notnull())
    # Segment nulls are checked only after that filter: a padding cell carries no
    # observation, so it must not be able to condemn an otherwise clean load.
    _reject_null_segments(t)
    meta = TriangleMeta(
        origin_grain=origin_grain, dev_grain=dev_grain, measure=measure, units=units
    )
    return Triangle(t, meta)


def _reject_null_segments(t: IbisTable) -> None:
    """Refuse rows whose segment (cohort key) columns are null.

    A segment tuple names the cohort - company, line of business. A null in it
    names no cohort, and it does not sit there inertly: every transform that
    equi-joins on the segment columns (``as_of``, ``latest_diagonal``,
    ``to_cumulative``, ``to_incremental``) drops those rows silently, because SQL
    join equality is false for NULL = NULL. The cohort would vanish from a
    retrospective with no error, no warning and a clean ``validate()`` - a
    result that looks fine and is short a company. Ingestion is the last point
    where the problem is still attributable to the data that caused it, so it is
    refused here rather than diagnosed downstream.

    Reachable from the mart of record, not just hand-built frames: the data
    model's ``dim_company`` derives ``company_name`` through a LEFT JOIN onto
    ``sat_company_details``, and ``company_name`` is one of the three Schedule P
    segment columns, so a hub company missing its satellite row arrives null.
    """
    segs = [c for c in t.columns if c not in CORE_COLUMNS]
    total, per_column = null_segment_counts(t, segs)
    if not total:
        return
    detail = ", ".join(f"{s} ({n} rows)" for s, n in sorted(per_column.items()))
    raise ValueError(
        f"null segment key in {detail}. Segment columns identify the cohort, so a null "
        "in one names no cohort and is silently dropped by every transform that joins "
        "on it (as_of, latest_diagonal, to_cumulative, to_incremental) - the cohort "
        "would disappear from results with no error. Give those rows an explicit value "
        "(e.g. 'unknown'), drop them, or omit the column via segments=[...]."
    )


# -- chainladder ---------------------------------------------------------------


def from_chainladder(tri, backend: str | BaseBackend | None = None) -> Triangle:
    """Convert a chainladder.Triangle (4D duck array) to a long Triangle.

    chainladder stores an (index x column x origin x development) dense array;
    the lower half of the square is materialized as NaN. Those padding cells are
    dropped here - absent means unobserved in the long format - and chainladder
    rebuilds them on the way back, so the round trip is still lossless.

    chainladder's ``development`` axis is already months from origin start, the
    same convention as ours, so dev_lag passes through unchanged (unlike
    bermuda's, see :func:`from_bermuda`). ``valuation`` is an end-of-day
    Timestamp and is narrowed to a plain date.
    """
    # keepdims/implicit_axis keep the index and development axes as real columns
    # even for single-index or single-column triangles, so the melt below sees a
    # uniform frame regardless of triangle shape.
    df = tri.to_frame(keepdims=True, implicit_axis=True, origin_as_datetime=True).reset_index()
    field_cols = [str(c) for c in tri.columns]
    id_cols = [c for c in df.columns if c not in field_cols]
    long = df.melt(id_vars=id_cols, value_vars=field_cols, var_name="field", value_name="value")
    long = long.dropna(subset=["value"])  # drop the NaN padding half
    long = long.rename(
        columns={"origin": "origin_period", "development": "dev_lag", "valuation": "eval_date"}
    )
    # chainladder timestamps -> dates (its valuations are end-of-day timestamps;
    # keeping them would make eval_date comparisons in as_of() timestamp-sensitive)
    long["origin_period"] = long["origin_period"].dt.date
    long["eval_date"] = long["eval_date"].dt.date
    long["dev_lag"] = long["dev_lag"].astype("int64")

    # Fail loudly on grains we cannot represent (e.g. 'S') rather than silently
    # mislabeling the metadata, which every downstream transform trusts.
    for grain, attr in ((tri.origin_grain, "origin"), (tri.development_grain, "development")):
        if grain not in GRAIN_MONTHS:
            raise ValueError(f"unsupported chainladder {attr} grain {grain!r}")
    return from_long(
        long,
        measure="cumulative" if tri.is_cumulative else "incremental",
        origin_grain=tri.origin_grain,
        dev_grain=tri.development_grain,
        backend=backend,
    )


def to_chainladder(t: Triangle):
    """Convert to a chainladder.Triangle (requires the chainladder package).

    The inverse of :func:`from_chainladder`. We hand chainladder the *valuation*
    (``eval_date``) rather than the dev lag as its development axis: chainladder
    derives development from origin and valuation itself, and doing it that way
    reproduces its own dev bucketing (including the latest-diagonal anchoring of
    ``grain('OYDY')``) instead of second-guessing it. chainladder re-materializes
    the NaN padding cells that the long format omitted.

    Handing over the valuation is also why eval_date has to agree with
    origin_period + dev_lag on every row: chainladder never sees the stored
    dev_lag, so a row that disagrees is exported at the development its eval_date
    implies, and two such rows at one eval_date are added together into one cell.
    """
    cl = _require_interop("chainladder", "Triangle.to_chainladder()")
    import pandas as pd

    require_eval_alignment(
        t,
        operation="to_chainladder()",
        reason=(
            "The export hands chainladder the valuation and lets it derive development, "
            "so a row whose stored dev_lag disagrees with its eval_date is exported at a "
            "different development, and two such rows at one eval_date become one cell."
        ),
    )
    df = t.expr.execute()
    fields = sorted(df["field"].unique())
    segments = t.segments
    # Long -> wide: one column per field, one row per (segment, origin, eval).
    # dev_lag is not in that index, so the pivot is only faithful when eval_date
    # implies it, which is exactly what the check above requires. Given it,
    # aggfunc="sum" adds nothing up: two rows can share (segment, origin, eval)
    # only by being the same cell twice, which validate() reports as a duplicate.
    wide = df.pivot_table(
        index=[*segments, "origin_period", "eval_date"],
        columns="field",
        values="value",
        aggfunc="sum",
    ).reset_index()
    # chainladder wants datetime64, not datetime.date; it normalizes the
    # valuation column to end-of-period timestamps internally.
    wide["origin_period"] = pd.to_datetime(wide["origin_period"])
    wide["eval_date"] = pd.to_datetime(wide["eval_date"])
    return cl.Triangle(
        wide,
        origin="origin_period",
        development="eval_date",
        columns=fields,
        index=segments or None,
        cumulative=t.meta.measure == "cumulative",
    )


# -- bermuda -------------------------------------------------------------------


def from_bermuda(tri, backend: str | BaseBackend | None = None) -> Triangle:
    """Convert a bermuda Triangle (cell-based) to a long Triangle.

    Origin grain is inferred from period lengths and DEV grain from bermuda's
    ``eval_date_resolution`` (see :func:`_bermuda_dev_grain`, which also says what
    happens when there is only one diagonal to read it from); bermuda's
    incremental cells are detected via cell type.

    **Never copy bermuda's ``Cell.dev_lag``.** bermuda measures dev lag from the
    period *end*, so its first diagonal is 0; ours measures months from the
    origin *start*, so the first diagonal is 12 at annual grain. dev_lag is
    recomputed below from ``period_start`` -> ``evaluation_date`` (inclusive
    month count, hence the ``+ 1``), which is the only value the rest of the
    package will accept.
    """
    import pandas as pd

    rows = []
    incremental = False
    grains = set()
    for cell in tri:
        # bermuda encodes cumulative/incremental in the cell class, not metadata;
        # any incremental cell makes the whole triangle incremental.
        incremental = incremental or "incremental" in type(cell).__name__.lower()
        start, end = cell.period_start, cell.period_end
        # Origin grain is not stored either - infer it from the period span in
        # months (12 for annual, 3 for quarterly, ...).
        grains.add((end.year - start.year) * 12 + end.month - start.month + 1)
        ev = cell.evaluation_date
        base = {
            # cell.details carries bermuda's per-cell segment keys (company, lob...)
            **cell.details,
            "origin_period": start,
            # our convention: months from origin START, inclusive of the eval
            # month -> first annual diagonal = 12, NOT bermuda's cell.dev_lag
            "dev_lag": (ev.year - start.year) * 12 + ev.month - start.month + 1,
            "eval_date": ev,
        }
        for field_name, val in cell.values.items():
            rows.append({**base, "field": field_name, "value": float(val)})
    df = pd.DataFrame(rows)
    months_to_grain = {v: k for k, v in GRAIN_MONTHS.items()}
    # Take the widest observed period: a partially-developed final period can
    # look shorter than the true grain, and under-calling the grain would scale
    # dev lags wrongly downstream.
    grain = months_to_grain.get(max(grains, default=12), "Y")
    return from_long(
        df,
        measure="incremental" if incremental else "cumulative",
        origin_grain=grain,
        dev_grain=_bermuda_dev_grain(tri, months_to_grain, origin_grain=grain),
        backend=backend,
    )


def _bermuda_dev_grain(tri, months_to_grain: dict[int, str], *, origin_grain: str) -> str:
    """The dev grain of a bermuda triangle: the spacing of its evaluation dates.

    bermuda stores no dev grain, and the ORIGIN grain is the wrong answer for it.
    Annual periods observed every quarter are an ordinary bermuda triangle, and
    calling that OYDY leaves every cell intact under a label that is wrong: the
    step declared here is what ``to_incremental`` looks back by, so a 12-month
    step on quarterly cells finds no predecessor and keeps 2 rows out of 8.

    ``eval_date_resolution`` is bermuda's own answer, the months between
    evaluation dates, and it is ``None`` when the triangle carries a single
    diagonal - where the data genuinely does not say how far apart the next one
    would be. The origin grain is the fallback there, and only there: the
    attribute is read directly, so a bermuda release that renames it raises
    ``AttributeError`` here instead of quietly restoring the wrong label this
    function exists to stop.

    The one limit worth knowing before trusting the round trip: bermuda carries
    cells, not declarations, so the grain that comes back is the spacing bermuda
    could observe. A triangle declared quarterly that holds only annual diagonals
    (ages 3 and 15, say) comes back annual, because nothing in the cells says
    otherwise.
    """
    resolution = tri.eval_date_resolution
    if resolution is None:
        return origin_grain
    resolution = int(resolution)
    if resolution not in months_to_grain:
        supported = ", ".join(f"{m} ({g})" for m, g in sorted(months_to_grain.items()))
        raise ValueError(
            f"bermuda triangle has a {resolution}-month evaluation-date resolution, which "
            f"is not a dev grain this package can represent (supported: {supported} months). "
            "Rounding it to a supported grain would mislabel every dev lag downstream, so "
            "the conversion stops here rather than guessing."
        )
    return months_to_grain[resolution]


def to_bermuda(t: Triangle):
    """Convert to a bermuda Triangle (requires the bermuda-ledger package).

    The inverse of :func:`from_bermuda`. Our long rows are one field per row;
    bermuda's cell is (period, evaluation) with a ``values`` dict of all fields,
    so rows are grouped back into cells here. We emit ``period_start`` /
    ``period_end`` and let bermuda derive its own end-anchored dev lag - the
    dev_lag column is deliberately not exported (see the convention note in the
    module docstring).

    Letting bermuda derive the dev lag is also why eval_date has to agree with
    origin_period + dev_lag on every row: a row that disagrees comes back at the
    dev_lag its evaluation date implies, and two such rows at one evaluation date
    are grouped into a single cell, where one value for a field replaces the
    other.
    """
    bermuda = _require_interop("bermuda", "Triangle.to_bermuda()")

    require_eval_alignment(
        t,
        operation="to_bermuda()",
        reason=(
            "A bermuda cell is (period, evaluation date) and from_bermuda derives dev_lag "
            "from that evaluation date, so a row whose stored dev_lag disagrees comes back "
            "at a different one, and two such rows at one evaluation date are merged into "
            "one cell."
        ),
    )
    # cumulative vs incremental is carried by the cell class on bermuda's side
    cell_cls = bermuda.CumulativeCell if t.meta.measure == "cumulative" else bermuda.IncrementalCell
    months = GRAIN_MONTHS[t.meta.origin_grain]
    df = t.expr.execute()
    for col in ("origin_period", "eval_date"):  # bermuda wants datetime.date
        if str(df[col].dtype).startswith("datetime64"):
            df[col] = df[col].dt.date
    # One bermuda cell per (segment..., origin, evaluation); dropna=False keeps
    # rows whose segment value is null instead of silently losing those cells.
    keys = [*t.segments, "origin_period", "eval_date"]
    cells = []
    for cell_key, group in df.groupby(keys, dropna=False):
        # pandas yields a scalar key when grouping on a single column
        key_vals = cell_key if isinstance(cell_key, tuple) else (cell_key,)
        cell_key = dict(zip(keys, key_vals, strict=True))
        start = cell_key["origin_period"]
        # bermuda's period_end is inclusive: the day before the next period starts
        end = _add_months(start, months) - dt.timedelta(days=1)
        details = {k: v for k, v in cell_key.items() if k in t.segments}
        cells.append(
            cell_cls(
                period_start=start,
                period_end=end,
                evaluation_date=cell_key["eval_date"],
                values=dict(zip(group["field"], group["value"], strict=True)),
                metadata=bermuda.Metadata(details=details),
            )
        )
    return bermuda.Triangle(cells)


def _add_months(d: dt.date, months: int) -> dt.date:
    """Shift a date by whole months, keeping the day-of-month.

    Only ever called on period *start* dates (day 1 for every supported grain),
    so the day-of-month is always valid in the target month - this deliberately
    does not implement end-of-month clamping.
    """
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return dt.date(y, m + 1, d.day)
