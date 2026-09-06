"""The sqlglot cap in pyproject, exercised through the call it protects.

sqlglot 30.18.0 renamed the ``Drop`` expression's ``this`` argument to
``tables`` (tobymao/sqlglot#8229, listed under BREAKING CHANGES in sqlglot's
v30.18.0 changelog). ibis 12.0.0 still builds ``sge.Drop(kind=..., this=...)``,
which that sqlglot renders with no table name, and duckdb answers ``Parser
Error: syntax error at end of input``. ibis's duckdb ``create_table`` reaches it
for every in-memory frame (pandas, pyarrow or polars) - it registers the frame
as a memtable view, inserts from it, then DROPs the view - and that is the one
call ``triangle/io.py::_register`` makes, so a fresh ``pip install ibnr`` could
not build a Triangle from a frame. pyproject caps ``sqlglot<30.18`` until the
ibis floor reaches a release carrying the fix proposed in
ibis-project/ibis#12104.

This test is the named tripwire. Without the cap the rest of the suite already
fails several hundred times; what none of those failures can do is say WHY in
one line. Two things keep it honest. It goes through the public ingestion path
rather than rendering ``sge.Drop`` itself, because after ibis ships its fix and
the cap is lifted, sqlglot will still render ``Drop(this=...)`` empty - the
behaviour that has to keep working is ibis's, not sqlglot's. And it blames
sqlglot only when the rename is present in the installed sqlglot
(``"tables" in sge.Drop.arg_types``, the predicate ibis's fix keys on); any
other ingestion failure is re-raised untouched, so the next unrelated break on
the floating CI legs is not misdiagnosed. On the locked resolution (sqlglot
30.11.0) it passes and proves nothing new; it exists for the two floating legs,
where the newest sqlglot is what gets installed.
"""

from __future__ import annotations

import datetime as dt

import ibis
import pandas as pd
import pytest
import sqlglot
import sqlglot.expressions as sge

from ibnr import Triangle


def _why() -> str:
    return (
        "ibis could not build a table from a pandas frame on the installed "
        f"sqlglot {sqlglot.__version__} with ibis {ibis.__version__}. "
        "sqlglot>=30.18 renamed Drop's `this` argument to `tables` "
        "(tobymao/sqlglot#8229) and ibis below the fix proposed in "
        "ibis-project/ibis#12104 still passes `this`, so the DROP VIEW that ends "
        "create_table has no name. pyproject caps sqlglot<30.18 for exactly this; "
        "lift the cap only together with an ibis floor that carries the fix."
    )


def test_ingestion_survives_the_installed_sqlglot():
    frame = pd.DataFrame(
        {
            "origin_period": [dt.date(1988, 1, 1), dt.date(1988, 1, 1)],
            "dev_lag": [12, 24],
            "eval_date": [dt.date(1988, 12, 31), dt.date(1989, 12, 31)],
            "field": ["paid_loss", "paid_loss"],
            "value": [100.0, 150.0],
        }
    )
    try:
        tri = Triangle.from_long(frame)
    except Exception as exc:  # duckdb's ParserException is not a Python builtin
        if "tables" not in sge.Drop.arg_types:
            raise  # not the Drop rename: let the real failure surface as itself
        pytest.fail(f"{_why()}\n{type(exc).__name__}: {exc}")
    assert tri.count() == 2
