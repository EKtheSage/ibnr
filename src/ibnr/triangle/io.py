"""Triangle ingestion and interop.

Interop is sacred: ``from_chainladder``/``to_chainladder`` and
``from_bermuda``/``to_bermuda`` must round-trip losslessly (modulo NaN padding
cells, which chainladder materializes and the long format simply omits).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from uuid import uuid4

import ibis
from ibis import selectors as s
from ibis.backends import BaseBackend
from ibis.expr.types import Table as IbisTable

from ibnr.triangle.core import GRAIN_MONTHS, Triangle, TriangleMeta


def resolve_backend(backend: str | BaseBackend | None = None) -> BaseBackend:
    """'duckdb' (default) and 'polars' are the two supported names; an already
    connected ibis backend passes through."""
    if backend is None or backend == "duckdb":
        return ibis.duckdb.connect()
    if backend == "polars":
        return ibis.polars.connect()
    if isinstance(backend, BaseBackend):
        return backend
    raise ValueError(f"backend must be 'duckdb', 'polars', or an ibis backend, got {backend!r}")


def _register(con: BaseBackend, data) -> IbisTable:
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
    """
    if isinstance(data, IbisTable):
        t = data
    else:
        con = resolve_backend(backend)
        is_path = isinstance(data, str | Path)
        t = con.read_parquet(str(data)) if is_path else _register(con, data)

    renames = {"origin_period": origin, "dev_lag": dev, "eval_date": eval_date}
    if fields is None:
        renames |= {"field": field, "value": value}
    renames = {new: old for new, old in renames.items() if new != old}
    if renames:
        t = t.rename(**renames)

    if fields is not None:
        keep = segments if segments is not None else None
        id_cols = [c for c in t.columns if c not in fields]
        if keep is not None:
            id_cols = [c for c in id_cols if c in (*keep, "origin_period", "dev_lag", "eval_date")]
            t = t.select(*id_cols, *fields)
        t = t.pivot_longer(s.cols(*fields), names_to="field", values_to="value")
    elif segments is not None:
        t = t.select(*segments, "origin_period", "dev_lag", "eval_date", "field", "value")

    if dev_lag_unit == "periods":
        t = t.mutate(dev_lag=t.dev_lag.cast("int64") * GRAIN_MONTHS[dev_grain])
    elif dev_lag_unit != "months":
        raise ValueError(f"dev_lag_unit must be 'months' or 'periods', got {dev_lag_unit!r}")

    t = t.filter(t.value.notnull())
    meta = TriangleMeta(
        origin_grain=origin_grain, dev_grain=dev_grain, measure=measure, units=units
    )
    return Triangle(t, meta)


# -- chainladder ---------------------------------------------------------------


def from_chainladder(tri, backend: str | BaseBackend | None = None) -> Triangle:
    """Convert a chainladder.Triangle (4D duck array) to a long Triangle.

    NaN padding cells are dropped; chainladder rebuilds them on the way back.
    """
    df = tri.to_frame(keepdims=True, implicit_axis=True, origin_as_datetime=True).reset_index()
    field_cols = [str(c) for c in tri.columns]
    id_cols = [c for c in df.columns if c not in field_cols]
    long = df.melt(id_vars=id_cols, value_vars=field_cols, var_name="field", value_name="value")
    long = long.dropna(subset=["value"])
    long = long.rename(
        columns={"origin": "origin_period", "development": "dev_lag", "valuation": "eval_date"}
    )
    long["origin_period"] = long["origin_period"].dt.date
    long["eval_date"] = long["eval_date"].dt.date
    long["dev_lag"] = long["dev_lag"].astype("int64")

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
    """Convert to a chainladder.Triangle (requires the chainladder package)."""
    import chainladder as cl
    import pandas as pd

    df = t.expr.execute()
    fields = sorted(df["field"].unique())
    segments = t.segments
    wide = df.pivot_table(
        index=[*segments, "origin_period", "eval_date"],
        columns="field",
        values="value",
        aggfunc="sum",
    ).reset_index()
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

    Origin grain is inferred from period lengths; bermuda's incremental cells
    are detected via cell type.
    """
    import pandas as pd

    rows = []
    incremental = False
    grains = set()
    for cell in tri:
        incremental = incremental or "incremental" in type(cell).__name__.lower()
        start, end = cell.period_start, cell.period_end
        grains.add((end.year - start.year) * 12 + end.month - start.month + 1)
        ev = cell.evaluation_date
        base = {
            **cell.details,
            "origin_period": start,
            "dev_lag": (ev.year - start.year) * 12 + ev.month - start.month + 1,
            "eval_date": ev,
        }
        for field_name, val in cell.values.items():
            rows.append({**base, "field": field_name, "value": float(val)})
    df = pd.DataFrame(rows)
    months_to_grain = {v: k for k, v in GRAIN_MONTHS.items()}
    grain = months_to_grain.get(max(grains, default=12), "Y")
    return from_long(
        df,
        measure="incremental" if incremental else "cumulative",
        origin_grain=grain,
        dev_grain=grain,
        backend=backend,
    )


def to_bermuda(t: Triangle):
    """Convert to a bermuda Triangle (requires the bermuda-ledger package)."""
    import bermuda

    cell_cls = bermuda.CumulativeCell if t.meta.measure == "cumulative" else bermuda.IncrementalCell
    months = GRAIN_MONTHS[t.meta.origin_grain]
    df = t.expr.execute()
    for col in ("origin_period", "eval_date"):  # bermuda wants datetime.date
        if str(df[col].dtype).startswith("datetime64"):
            df[col] = df[col].dt.date
    keys = [*t.segments, "origin_period", "eval_date"]
    cells = []
    for cell_key, group in df.groupby(keys, dropna=False):
        key_vals = cell_key if isinstance(cell_key, tuple) else (cell_key,)
        cell_key = dict(zip(keys, key_vals, strict=True))
        start = cell_key["origin_period"]
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
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return dt.date(y, m + 1, d.day)
