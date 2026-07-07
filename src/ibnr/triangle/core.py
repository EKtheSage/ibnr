"""Triangle: a long-format reserving triangle over an ibis table expression.

A triangle is a tidy table with the core columns

    origin_period : date   -- start date of the origin (accident/underwriting) period
    dev_lag       : int    -- development lag in MONTHS from origin period start
    eval_date     : date   -- evaluation (valuation) date of the observation; stored,
                             never derived; all backtesting slices on it
    field         : str    -- measure name (paid_loss, incurred_loss, earned_premium, ...)
    value         : float

plus arbitrary additional columns, which are treated as segments (lob, company, ...).

dev_lag is always expressed in months regardless of grain, so grain conversions are
well-defined (chainladder and bermuda use the same convention).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import ibis.expr.types as ir

if TYPE_CHECKING:
    import chainladder as cl
    import pandas as pd
    import polars as pl

CORE_COLUMNS: tuple[str, ...] = ("origin_period", "dev_lag", "eval_date", "field", "value")

#: months per grain step
GRAIN_MONTHS: dict[str, int] = {"Y": 12, "Q": 3, "M": 1}

_CORE_TYPES: dict[str, str] = {
    "origin_period": "date",
    "dev_lag": "int64",
    "eval_date": "date",
    "field": "string",
    "value": "float64",
}

Measure = Literal["cumulative", "incremental"]


@dataclass(frozen=True)
class TriangleMeta:
    """Triangle-level metadata. Carried through every transformation."""

    origin_grain: str = "Y"
    dev_grain: str = "Y"
    measure: Measure = "cumulative"
    units: str | None = None

    def __post_init__(self) -> None:
        for name, grain in (("origin_grain", self.origin_grain), ("dev_grain", self.dev_grain)):
            if grain not in GRAIN_MONTHS:
                raise ValueError(f"{name} must be one of {sorted(GRAIN_MONTHS)}, got {grain!r}")
        if self.measure not in ("cumulative", "incremental"):
            raise ValueError(f"measure must be 'cumulative' or 'incremental', got {self.measure!r}")

    @property
    def grain(self) -> str:
        """chainladder-style grain string, e.g. 'OYDY'."""
        return f"O{self.origin_grain}D{self.dev_grain}"


class Triangle:
    """A reserving triangle over an ibis table expression.

    All transformations are lazy ibis expressions, written once and executable on
    both the duckdb and polars backends. ``execute()`` / ``to_pandas()`` /
    ``to_polars()`` materialize.
    """

    def __init__(self, expr: ir.Table, meta: TriangleMeta | None = None):
        missing = set(CORE_COLUMNS) - set(expr.columns)
        if missing:
            raise ValueError(f"triangle expression missing core columns: {sorted(missing)}")
        self._expr = expr.mutate(
            **{c: expr[c].cast(t) for c, t in _CORE_TYPES.items() if str(expr[c].type()) != t}
        )
        self.meta = meta or TriangleMeta()

    # -- introspection ------------------------------------------------------

    @property
    def expr(self) -> ir.Table:
        return self._expr

    @property
    def segments(self) -> list[str]:
        """Non-core columns; every distinct combination is a separate sub-triangle."""
        return [c for c in self._expr.columns if c not in CORE_COLUMNS]

    def _distinct(self, column: str) -> list:
        ser = self._expr.select(column).distinct().execute()[column]
        if str(ser.dtype).startswith("datetime64"):  # backends may widen date -> timestamp
            ser = ser.dt.date
        return sorted(ser)

    @property
    def fields(self) -> list[str]:
        return self._distinct("field")

    @property
    def eval_dates(self) -> list[dt.date]:
        return self._distinct("eval_date")

    @property
    def origins(self) -> list[dt.date]:
        return self._distinct("origin_period")

    @property
    def dev_lags(self) -> list[int]:
        return [int(d) for d in self._distinct("dev_lag")]

    def count(self) -> int:
        return int(self._expr.count().execute())

    def __repr__(self) -> str:
        seg = ", ".join(self.segments) or "none"
        return (
            f"Triangle(grain={self.meta.grain}, measure={self.meta.measure}, "
            f"units={self.meta.units!r}, segments=[{seg}])"
        )

    # -- materialization ----------------------------------------------------

    def execute(self) -> pd.DataFrame:
        return self._expr.execute()

    def to_pandas(self) -> pd.DataFrame:
        return self._expr.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        return self._expr.to_polars()

    def to_wide(self, field: str | None = None) -> pd.DataFrame:
        """Pivot one field to an origin x dev_lag matrix (materializes; for display)."""
        from ibnr.triangle import transforms

        return transforms.to_wide(self, field=field)

    # -- derivation helpers --------------------------------------------------

    def with_expr(self, expr: ir.Table, **meta_changes) -> Triangle:
        """New Triangle with a replaced expression and optional meta updates."""
        return Triangle(expr, replace(self.meta, **meta_changes) if meta_changes else self.meta)

    def filter(self, *predicates) -> Triangle:
        """Filter rows with ibis deferred predicates, e.g. ``t.filter(ibis._.lob == 'wkcomp')``."""
        return self.with_expr(self._expr.filter(*predicates))

    def select_fields(self, fields: str | list[str]) -> Triangle:
        fields = [fields] if isinstance(fields, str) else list(fields)
        return self.with_expr(self._expr.filter(self._expr.field.isin(fields)))

    # -- transforms (implemented once in transforms.py) ----------------------

    def to_cumulative(self) -> Triangle:
        from ibnr.triangle import transforms

        return transforms.to_cumulative(self)

    def to_incremental(self) -> Triangle:
        from ibnr.triangle import transforms

        return transforms.to_incremental(self)

    def as_of(self, eval_date: dt.date | str) -> Triangle:
        from ibnr.triangle import transforms

        return transforms.as_of(self, eval_date)

    def latest_diagonal(self) -> Triangle:
        from ibnr.triangle import transforms

        return transforms.latest_diagonal(self)

    def with_dev_grain(self, grain: str) -> Triangle:
        from ibnr.triangle import transforms

        return transforms.change_dev_grain(self, grain)

    def with_origin_grain(self, grain: str) -> Triangle:
        from ibnr.triangle import transforms

        return transforms.change_origin_grain(self, grain)

    # -- validation -----------------------------------------------------------

    def validate(self, strict: bool = True) -> list[str]:
        from ibnr.triangle import validate

        return validate.validate(self, strict=strict)

    # -- io / interop ----------------------------------------------------------

    @classmethod
    def from_long(cls, data, **kwargs) -> Triangle:
        from ibnr.triangle import io

        return io.from_long(data, **kwargs)

    @classmethod
    def from_chainladder(cls, tri: cl.Triangle, **kwargs) -> Triangle:
        from ibnr.triangle import io

        return io.from_chainladder(tri, **kwargs)

    def to_chainladder(self) -> cl.Triangle:
        from ibnr.triangle import io

        return io.to_chainladder(self)

    @classmethod
    def from_bermuda(cls, tri, **kwargs) -> Triangle:
        from ibnr.triangle import io

        return io.from_bermuda(tri, **kwargs)

    def to_bermuda(self):
        from ibnr.triangle import io

        return io.to_bermuda(self)


def implied_dev_lag(origin: ir.DateValue, eval_date: ir.DateValue) -> ir.IntegerValue:
    """Development lag in months implied by an origin period start and an eval date.

    Assumes eval_date falls in the last month of the development period, the
    universal statement convention (e.g. origin 1988-01-01 evaluated 1988-12-31
    -> 12 months). Pure year/month arithmetic so it runs on every backend.
    """
    return (
        (eval_date.year() - origin.year()) * 12 + (eval_date.month() - origin.month()) + 1
    ).cast("int64")
