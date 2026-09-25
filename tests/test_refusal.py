"""``ibnr.errors.Refusal``: every input ibnr will not answer, refused with a reason code.

Four things are checked. The class itself: a ``ValueError`` with fields, a
closed list of codes, a cap on the cells it keeps, a JSON payload with no NaN
and a pickle round trip (a process pool sends exceptions by pickling them).
Every refusal reachable from ``ibnr.methods``, through the front door, with its
code, argument, column and the cells, ages or rows at fault, each origin as the
caller wrote it, in every form the caller may write it. The messages, which
use no word of the kernels underneath. And that nothing but a ``Refusal``
escapes a method on bad input: a seeded fuzz over five public triangles, and a
sweep of every clrd triangle.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
import pickle
import re
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from ibnr import errors, methods
from ibnr.errors import KIND, MAX_CELLS, REASONS, Refusal, RefusedCell

# a 4 x 4 cumulative staircase, integer accident years 2001 to 2004
BASE = [
    (2001, 12, 100.0),
    (2001, 24, 150.0),
    (2001, 36, 170.0),
    (2001, 48, 175.0),
    (2002, 12, 110.0),
    (2002, 24, 168.0),
    (2002, 36, 190.0),
    (2003, 12, 120.0),
    (2003, 24, 175.0),
    (2004, 12, 130.0),
]
PREM = {2001: 200.0, 2002: 210.0, 2003: 220.0, 2004: 230.0}


def tri(rows) -> pa.Table:
    o, d, v = zip(*rows, strict=True)
    return pa.table({"origin_period": list(o), "dev_lag": list(d), "value": list(v)})


def replace(rows, key, value):
    return [(o, d, value) if (o, d) == key else (o, d, v) for o, d, v in rows]


def drop(rows, key):
    return [(o, d, v) for o, d, v in rows if (o, d) != key]


T = tri(BASE)
ZERO_COL = [(o, d, 0.0 if d == 12 else v) for o, d, v in BASE]
Z2 = replace(replace(BASE, (2002, 12), 0.0), (2003, 12), 0.0)
Z3 = replace(replace(BASE, (2002, 36), 0.0), (2002, 24), 0.0)
cl, bf, cc, mk = (
    methods.chain_ladder,
    methods.bornhuetter_ferguson,
    methods.cape_cod,
    methods.mack,
)


def year(y: int) -> dt.date:
    return dt.date(y, 1, 1)


def refusal_of(thunk) -> Refusal:
    with pytest.raises(Refusal) as caught:
        thunk()
    assert type(caught.value) is Refusal
    return caught.value


# -- 1. the class itself -----------------------------------------------------------


def test_a_refusal_is_a_value_error_whose_message_is_its_text():
    refusal = Refusal("invalid_option", "average must be one of three, got {given}", given="x")
    assert issubclass(Refusal, ValueError)
    assert str(refusal) == refusal.args[0] == "average must be one of three, got 'x'"
    with pytest.raises(ValueError, match="got 'x'"):
        raise refusal


def test_an_unknown_reason_is_refused_by_the_constructor():
    with pytest.raises(ValueError, match="unknown refusal reason 'nope'"):
        Refusal("nope", "text")


def test_every_reason_has_a_kind_and_nothing_else_does():
    assert set(KIND) == set(REASONS)
    assert len(REASONS) == len(set(REASONS))
    assert set(KIND.values()) == {"input", "model"}
    # the kinds of a few codes, so that flipping one is caught
    assert KIND["negative_cumulative"] == "input"
    assert KIND["not_finite"] == "input"
    assert KIND["no_link_ratio"] == "model"
    assert KIND["variance_not_estimable"] == "model"
    assert KIND["negative_projection"] == "model"
    assert KIND["result_not_finite"] == "model"
    inputs = [r for r in REASONS if KIND[r] == "input"]
    assert inputs == list(REASONS[: len(inputs)]), "input codes are listed first"
    assert len(inputs) == 13 and len(REASONS) == 27


def full_refusal() -> Refusal:
    return Refusal(
        "negative_cumulative",
        "value is negative in {cells}; given {given}; {links}; {origins}",
        option="cells",
        column="value",
        options=("cells", "zero_cells"),
        given={1, 2},
        cells=[
            RefusedCell("2003", year(2003), 24, -1.0),
            RefusedCell(2002, year(2002), 24, math.nan),
            RefusedCell(dt.date(2004, 12, 31), year(2004), None, math.inf),
        ],
        links=[(24, 36)],
        rows=[7, 3],
        count=12,
        method="chain_ladder",
    )


@pytest.mark.parametrize(
    "round_trip",
    [lambda r: pickle.loads(pickle.dumps(r)), copy.deepcopy, copy.copy],
    ids=["pickle", "deepcopy", "copy"],
)
def test_a_refusal_survives_pickle_and_copy_with_every_field(round_trip):
    refusal = full_refusal()
    back = round_trip(refusal)
    assert type(back) is Refusal
    assert str(back) == str(refusal)
    for name in (
        "reason",
        "kind",
        "method",
        "option",
        "column",
        "options",
        "given",
        "cells",
        "links",
        "rows",
        "count",
    ):
        a, b = getattr(back, name), getattr(refusal, name)
        assert repr(a) == repr(b), name


def _raise_in_a_worker() -> None:
    methods.chain_ladder(tri(replace(BASE, (2002, 24), -5.0)))


def test_a_refusal_crosses_a_process_pool():
    """The harness and tlrn send exceptions between processes by pickling them;
    without ``__reduce__`` the parent gets a failure to unpickle, not the refusal."""
    with ProcessPoolExecutor(max_workers=1) as pool, pytest.raises(Refusal) as caught:
        pool.submit(_raise_in_a_worker).result(timeout=120)
    assert caught.value.reason == "negative_cumulative"
    assert caught.value.cells[0].origin == 2002


def test_the_cells_are_cut_at_the_cap_and_the_count_keeps_the_total():
    cells = [RefusedCell(y, year(y), 12, -1.0) for y in range(1850, 2000)]
    refusal = Refusal("negative_cumulative", "negative in {cells}", cells=cells)
    assert refusal.count == 150
    assert len(refusal.cells) == MAX_CELLS == 100
    assert str(refusal).endswith("(1854, 12 months) and 145 more")
    assert str(refusal).count("months") == 5
    rows = Refusal("missing_value", "rows", rows=range(250))
    assert rows.count == 250 and len(rows.rows) == 100
    with pytest.raises(ValueError, match="count 1 is less than the 2 items"):
        Refusal("duplicate", "x", cells=cells[:2], count=1)


def test_the_cells_are_sorted_by_origin_period_and_age():
    cells = [RefusedCell(y, year(y), lag, 1.0) for y in (2003, 2001, 2002) for lag in (24, 12)]
    refusal = Refusal("duplicate", "{cells}", cells=cells)
    assert [(c.origin, c.dev_lag) for c in refusal.cells] == [
        (y, lag) for y in (2001, 2002, 2003) for lag in (12, 24)
    ]


def test_the_payload_is_json_with_no_nan_or_infinity():
    payload = full_refusal().to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text) == payload
    assert payload["reason"] == "negative_cumulative"
    assert payload["kind"] == "input"
    assert payload["method"] == "chain_ladder"
    assert payload["option"] == "cells" and payload["column"] == "value"
    assert payload["options"] == ["cells", "zero_cells"]
    assert payload["given"] == "{1, 2}"
    assert payload["cells"] == [
        {"origin": 2002, "origin_period": "2002-01-01", "dev_lag": 24, "value": None},
        {"origin": "2003", "origin_period": "2003-01-01", "dev_lag": 24, "value": -1.0},
        {"origin": "2004-12-31", "origin_period": "2004-01-01", "dev_lag": None, "value": None},
    ]
    assert payload["links"] == [[24, 36]]
    assert payload["rows"] == [7, 3]
    assert payload["count"] == 12
    assert payload["message"] == str(full_refusal())
    assert Refusal("invalid_option", "x", given=math.inf).to_dict()["given"] is None
    assert Refusal("invalid_option", "x", given=[1, math.nan]).to_dict()["given"] == [1, None]


def test_the_errors_module_imports_the_standard_library_only():
    source = Path(errors.__file__).read_text(encoding="utf-8")
    imported = set(re.findall(r"^(?:from|import) ([a-z_.]+)", source, flags=re.M))
    assert imported <= {"__future__", "datetime", "math", "numbers", "dataclasses", "typing"}


def test_methods_exports_refusal():
    assert methods.Refusal is Refusal
    assert "Refusal" in methods.__all__


# -- 2. every refusal reachable from ibnr.methods ------------------------------------


def c(origin, lag=None, value=...):
    """An expected cell: (origin as written, dev_lag[, value])."""
    return (origin, lag, value)


#: (id, call, reason, option, column, expected cells, links, rows, extra fields)
CASES = [
    # reading the tables
    ("cells_list", lambda: cl([1, 2, 3]), "invalid_table", "cells", None, [], [], [], {}),
    (
        "cells_no_value",
        lambda: cl(T.drop(["value"])),
        "invalid_table",
        "cells",
        "value",
        [],
        [],
        [],
        {},
    ),
    ("cells_no_rows", lambda: cl(T.slice(0, 0)), "invalid_table", "cells", None, [], [], [], {}),
    (
        "origin_null",
        lambda: cl(
            pa.table(
                {
                    "origin_period": pa.array([2001, None], pa.int64()),
                    "dev_lag": [12, 12],
                    "value": [1.0, 2.0],
                }
            )
        ),
        "missing_value",
        "cells",
        "origin_period",
        [],
        [],
        [1],
        {},
    ),
    (
        "origin_float",
        lambda: cl(pa.table({"origin_period": [2001.0], "dev_lag": [12], "value": [1.0]})),
        "invalid_table",
        "cells",
        "origin_period",
        [],
        [],
        [],
        {},
    ),
    (
        "origin_two_digit",
        lambda: cl(tri([(2001, 12, 1.0), (97, 12, 1.0)])),
        "unreadable_label",
        "cells",
        "origin_period",
        [],
        [],
        [1],
        {"given": 97},
    ),
    (
        "origin_month_13",
        lambda: cl(tri([("2020-13", 12, 1.0)])),
        "unreadable_label",
        "cells",
        "origin_period",
        [],
        [],
        [0],
        {"given": "2020-13"},
    ),
    (
        "origin_mid_month",
        lambda: cl(tri([(dt.date(2020, 6, 15), 12, 1.0)])),
        "unreadable_label",
        "cells",
        "origin_period",
        [],
        [],
        [0],
        {"given": dt.date(2020, 6, 15)},
    ),
    (
        "origin_before_year_1",
        lambda: cl(tri([(dt.date(1, 6, 30), 12, 1.0)])),
        "unreadable_label",
        "cells",
        "origin_period",
        [],
        [],
        [0],
        {"given": dt.date(1, 6, 30)},
    ),
    (
        "origin_quarter_on_annual",
        lambda: cl(tri([("2020Q1", 12, 1.0)])),
        "grain_mismatch",
        "cells",
        "origin_period",
        [],
        [],
        [0],
        {"given": "2020Q1"},
    ),
    (
        "origin_two_spellings",
        lambda: cl(tri([("2020", 12, 1.0), ("2020-01-01", 24, 2.0)])),
        "duplicate",
        "cells",
        "origin_period",
        [c("2020"), c("2020-01-01")],
        [],
        [],
        {},
    ),
    (
        "dev_lag_null",
        lambda: cl(
            pa.table(
                {
                    "origin_period": [2001, 2001],
                    "dev_lag": pa.array([12, None], pa.int64()),
                    "value": [1.0, 2.0],
                }
            )
        ),
        "missing_value",
        "cells",
        "dev_lag",
        [],
        [],
        [1],
        {},
    ),
    (
        "dev_lag_fractional",
        lambda: cl(pa.table({"origin_period": [2001], "dev_lag": [12.5], "value": [1.0]})),
        "invalid_age",
        "cells",
        "dev_lag",
        [],
        [],
        [0],
        {},
    ),
    (
        "dev_lag_text",
        lambda: cl(pa.table({"origin_period": [2001], "dev_lag": ["12"], "value": [1.0]})),
        "invalid_table",
        "cells",
        "dev_lag",
        [],
        [],
        [],
        {},
    ),
    (
        "dev_lag_off_step",
        lambda: cl(tri([*BASE, (2004, 18, 1.0)])),
        "grain_mismatch",
        "cells",
        "dev_lag",
        [c(2004, 18, 1.0)],
        [],
        [],
        {},
    ),
    (
        "dev_lag_zero",
        lambda: cl(tri([*BASE, (2004, 0, 1.0)])),
        "invalid_age",
        "cells",
        "dev_lag",
        [c(2004, 0, 1.0)],
        [],
        [],
        {},
    ),
    (
        "dev_lag_negative",
        lambda: cl(tri([*BASE, (2004, -12, 1.0)])),
        "invalid_age",
        "cells",
        "dev_lag",
        [c(2004, -12, 1.0)],
        [],
        [],
        {},
    ),
    (
        "value_null",
        lambda: cl(
            pa.table(
                {"origin_period": [2001, 2001], "dev_lag": [12, 24], "value": pa.array([1.0, None])}
            )
        ),
        "missing_value",
        "cells",
        "value",
        [c(2001, 24, None)],
        [],
        [],
        {},
    ),
    (
        "value_nan",
        lambda: cl(tri(replace(BASE, (2002, 24), math.nan))),
        "missing_value",
        "cells",
        "value",
        [c(2002, 24, None)],
        [],
        [],
        {},
    ),
    (
        "value_text",
        lambda: cl(pa.table({"origin_period": [2001], "dev_lag": [12], "value": ["1"]})),
        "invalid_table",
        "cells",
        "value",
        [],
        [],
        [],
        {},
    ),
    (
        "value_inf",
        lambda: cl(tri(replace(BASE, (2002, 24), math.inf))),
        "not_finite",
        "cells",
        "value",
        [c(2002, 24, None)],
        [],
        [],
        {},
    ),
    (
        "value_minus_inf",
        lambda: cl(tri(replace(BASE, (2002, 24), -math.inf))),
        "not_finite",
        "cells",
        "value",
        [c(2002, 24, None)],
        [],
        [],
        {},
    ),
    (
        "mack_value_inf_interior",
        lambda: mk(tri(replace(BASE, (2002, 24), math.inf))),
        "not_finite",
        "cells",
        "value",
        [c(2002, 24, None)],
        [],
        [],
        {},
    ),
    (
        "mack_value_inf_observed",
        lambda: mk(tri(replace(BASE, (2002, 24), math.inf)), zero_cells="observed"),
        "not_finite",
        "cells",
        "value",
        [c(2002, 24, None)],
        [],
        [],
        {},
    ),
    (
        "mack_value_inf_latest",
        lambda: mk(tri(replace(BASE, (2004, 12), math.inf))),
        "not_finite",
        "cells",
        "value",
        [c(2004, 12, None)],
        [],
        [],
        {},
    ),
    (
        "value_negative",
        lambda: cl(tri(replace(BASE, (2002, 24), -5.0))),
        "negative_cumulative",
        "cells",
        "value",
        [c(2002, 24, -5.0)],
        [],
        [],
        {},
    ),
    (
        "duplicate_cell",
        lambda: cl(tri([*BASE, (2002, 24, 1.0)])),
        "duplicate",
        "cells",
        None,
        [c(2002, 24, 168.0), c(2002, 24, 1.0)],
        [],
        [],
        {},
    ),
    (
        "origin_gap",
        lambda: cl(tri([r for r in BASE if r[0] != 2003])),
        "origin_gap",
        "cells",
        "origin_period",
        [c(2002), c(None), c(2004)],
        [],
        [],
        {},
    ),
    (
        "origins_not_whole_steps",
        lambda: cl(tri([(dt.date(2001, 1, 1), 12, 1.0), (dt.date(2001, 7, 1), 12, 1.0)])),
        "grain_mismatch",
        "cells",
        "origin_period",
        [c(dt.date(2001, 1, 1)), c(dt.date(2001, 7, 1))],
        [],
        [],
        {},
    ),
    (
        # year labels say the periods are a year long, so two years apart is a gap
        # even with no neighbours one step apart
        "origin_gap_between_two_years",
        lambda: cl(tri([(2001, 12, 1.0), (2001, 24, 2.0), (2003, 12, 1.0)])),
        "origin_gap",
        "cells",
        "origin_period",
        [c(2001), c(None), c(2003)],
        [],
        [],
        {},
    ),
    (
        # dates a year apart on a quarterly step: periods of another length, not
        # three quarters missing between every two years
        "annual_dates_on_a_quarterly_step",
        lambda: cl(
            tri([(dt.date(2001, 1, 1), 3, 1.0), (dt.date(2002, 1, 1), 3, 1.0)]),
            dev_grain_months=3,
        ),
        "grain_mismatch",
        "cells",
        "origin_period",
        [c(dt.date(2001, 1, 1)), c(dt.date(2002, 1, 1))],
        [],
        [],
        {},
    ),
    (
        "no_first_age_cell",
        lambda: cl(tri(drop(BASE, (2003, 12)))),
        "not_run_off",
        "cells",
        None,
        [c(2003, 12, None)],
        [],
        [],
        {},
    ),
    (
        "interior_hole",
        lambda: cl(tri(drop(BASE, (2002, 24)))),
        "not_run_off",
        "cells",
        None,
        [c(2002, 24, None)],
        [],
        [],
        {},
    ),
    (
        "short_of_the_diagonal",
        lambda: cl(tri(drop(BASE, (2002, 36)))),
        "not_run_off",
        "cells",
        None,
        [c(2002, 36, None)],
        [],
        [],
        {},
    ),
    (
        "past_the_diagonal",
        lambda: cl(tri([*BASE, (2004, 24, 140.0)])),
        "not_run_off",
        "cells",
        None,
        [c(2004, 24, 140.0)],
        [],
        [],
        {},
    ),
    (
        "dev_grain_months_0",
        lambda: cl(T, dev_grain_months=0),
        "invalid_option",
        "dev_grain_months",
        None,
        [],
        [],
        [],
        {"given": 0},
    ),
    (
        "dev_grain_months_float",
        lambda: cl(T, dev_grain_months=12.0),
        "invalid_option",
        "dev_grain_months",
        None,
        [],
        [],
        [],
        {"given": 12.0},
    ),
    # development options
    *[
        (
            f"{name}_{value!r}",
            lambda name=name, value=value: cl(T, **{name: value}),
            "invalid_option",
            name,
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for name, value in [
            ("average", "regression"),
            ("average", 1.0),
            ("history_periods", 0),
            ("history_periods", 2.5),
            ("history_periods", True),
            ("drop_high", 1),
            ("drop_low", "yes"),
            ("unsupported_factor", "x"),
            ("exhausted_exclusions", "x"),
            ("zero_cells", "x"),
        ]
    ],
    (
        "exclude_string",
        lambda: cl(T, exclude="2001,12"),
        "invalid_option",
        "exclude",
        None,
        [],
        [],
        [],
        {"given": "2001,12"},
    ),
    (
        "exclude_one_tuple",
        lambda: cl(T, exclude=[(2001,)]),
        "invalid_option",
        "exclude",
        None,
        [],
        [],
        [],
        {"given": (2001,)},
    ),
    (
        "exclude_origin_abc",
        lambda: cl(T, exclude=[("abc", 12)]),
        "unreadable_label",
        "exclude",
        None,
        [],
        [],
        [],
        {"given": "abc"},
    ),
    (
        "exclude_lag_text",
        lambda: cl(T, exclude=[(2001, "12")]),
        "invalid_option",
        "exclude",
        None,
        [],
        [],
        [],
        {"given": (2001, "12")},
    ),
    (
        "exclude_lag_0",
        lambda: cl(T, exclude=[(2001, 0)]),
        "invalid_option",
        "exclude",
        None,
        [],
        [],
        [],
        {"given": (2001, 0)},
    ),
    (
        "exclude_lag_off_step",
        lambda: cl(T, exclude=[(2001, 18)]),
        "grain_mismatch",
        "exclude",
        None,
        [c(2001, 18, None)],
        [],
        [],
        {},
    ),
    (
        "exclude_twice",
        lambda: cl(T, exclude=[(2001, 12), ("2001", 12)]),
        "duplicate",
        "exclude",
        None,
        [c(2001, 12, None), c("2001", 12, None)],
        [],
        [],
        {},
    ),
    (
        "exclude_no_link",
        lambda: cl(T, exclude=[(2004, 12)]),
        "not_in_triangle",
        "exclude",
        None,
        [c(2004, 12, None)],
        [],
        [],
        {},
    ),
    (
        "exclude_origin_not_in_triangle",
        lambda: cl(T, exclude=[(1999, 12)]),
        "not_in_triangle",
        "exclude",
        None,
        [c(1999, 12, None)],
        [],
        [],
        {},
    ),
    (
        "exhausted_raise",
        lambda: cl(T, drop_high=True, exhausted_exclusions="raise"),
        "exclusions_exhausted",
        "exhausted_exclusions",
        None,
        [],
        [(36, 48)],
        [],
        {},
    ),
    (
        "exclusion_empties_an_age",
        lambda: cl(T, exclude=[(2001, 36)]),
        "no_link_ratio",
        "unsupported_factor",
        None,
        [],
        [(36, 48)],
        [],
        {},
    ),
    (
        "zero_column_missing",
        lambda: cl(tri(ZERO_COL)),
        "no_link_ratio",
        "unsupported_factor",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    (
        "zero_column_observed",
        lambda: cl(tri(ZERO_COL), zero_cells="observed"),
        "no_link_ratio",
        "unsupported_factor",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    # Bornhuetter-Ferguson, Cape Cod and premium
    *[
        (
            f"elr_{value!r}",
            lambda value=value: bf(T, premium=PREM, expected_loss_ratio=value),
            "invalid_option",
            "expected_loss_ratio",
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for value in (None, math.nan, math.inf, "0.7", True, -0.1)
    ],
    *[
        (
            f"decay_{value!r}",
            lambda value=value: cc(T, premium=PREM, decay=value),
            "invalid_option",
            "decay",
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for value in (2.0, -0.5, math.nan, None, "0.5", True)
    ],
    (
        "premium_none",
        lambda: bf(T, premium=None, expected_loss_ratio=0.7),
        "invalid_option",
        "premium",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "premium_list",
        lambda: bf(T, premium=[200.0, 210.0], expected_loss_ratio=0.7),
        "invalid_table",
        "premium",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "premium_no_column",
        lambda: bf(T, premium=pa.table({"origin_period": [2001]}), expected_loss_ratio=0.7),
        "invalid_table",
        "premium",
        "premium",
        [],
        [],
        [],
        {},
    ),
    (
        "premium_origin_null",
        lambda: bf(
            T,
            premium=pa.table(
                {"origin_period": pa.array([2001, None], pa.int64()), "premium": [1.0, 2.0]}
            ),
            expected_loss_ratio=0.7,
        ),
        "missing_value",
        "premium",
        "origin_period",
        [],
        [],
        [1],
        {},
    ),
    (
        "premium_origin_quarter",
        lambda: bf(T, premium={**PREM, "2004Q1": 1.0}, expected_loss_ratio=0.7),
        "grain_mismatch",
        "premium",
        None,
        [],
        [],
        [],
        {"given": "2004Q1"},
    ),
    (
        "premium_amount_null",
        lambda: bf(
            T,
            premium=pa.table(
                {
                    "origin_period": [2001, 2002, 2003, 2004],
                    "premium": pa.array([1.0, None, 1.0, 1.0]),
                }
            ),
            expected_loss_ratio=0.7,
        ),
        "missing_value",
        "premium",
        "premium",
        [c(2002, None, None)],
        [],
        [],
        {},
    ),
    (
        "premium_twice",
        lambda: bf(T, premium={**PREM, "2001": 1.0}, expected_loss_ratio=0.7),
        "duplicate",
        "premium",
        None,
        [c(2001, None, 200.0), c("2001", None, 1.0)],
        [],
        [],
        {},
    ),
    (
        "premium_missing_origin",
        lambda: bf(
            T, premium={k: v for k, v in PREM.items() if k != 2004}, expected_loss_ratio=0.7
        ),
        "origin_not_covered",
        "premium",
        None,
        [c(2004)],
        [],
        [],
        {},
    ),
    (
        "premium_extra_origin",
        lambda: bf(T, premium={**PREM, 2005: 1.0}, expected_loss_ratio=0.7),
        "not_in_triangle",
        "premium",
        None,
        [c(2005, None, 1.0)],
        [],
        [],
        {},
    ),
    (
        "premium_zero",
        lambda: bf(T, premium={**PREM, 2003: 0.0}, expected_loss_ratio=0.7),
        "invalid_option",
        "premium",
        None,
        [c(2003, None, 0.0)],
        [],
        [],
        {},
    ),
    (
        "premium_negative",
        lambda: cc(T, premium={**PREM, 2003: -5.0}),
        "invalid_option",
        "premium",
        None,
        [c(2003, None, -5.0)],
        [],
        [],
        {},
    ),
    (
        "premium_nan",
        lambda: bf(T, premium={**PREM, 2003: math.nan}, expected_loss_ratio=0.7),
        "missing_value",
        "premium",
        None,
        [c(2003, None, None)],
        [],
        [],
        {},
    ),
    (
        "premium_inf",
        lambda: bf(T, premium={**PREM, 2003: math.inf}, expected_loss_ratio=0.7),
        "not_finite",
        "premium",
        None,
        [c(2003, None, None)],
        [],
        [],
        {},
    ),
    (
        "premium_text",
        lambda: bf(T, premium={**PREM, 2003: "220"}, expected_loss_ratio=0.7),
        "invalid_option",
        "premium",
        None,
        [c(2003, None, None)],
        [],
        [],
        {"given": "220"},
    ),
    # Mack
    (
        "mack_sigma_rule",
        lambda: mk(T, sigma_rule="x"),
        "invalid_option",
        "sigma_rule",
        None,
        [],
        [],
        [],
        {"given": "x"},
    ),
    (
        "mack_zero_cells",
        lambda: mk(T, zero_cells="x"),
        "invalid_option",
        "zero_cells",
        None,
        [],
        [],
        [],
        {"given": "x"},
    ),
    (
        "mack_one_cell",
        lambda: mk(tri([(2001, 12, 100.0)])),
        "variance_not_estimable",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "mack_two_origins",
        lambda: mk(tri([(2001, 12, 100.0), (2001, 24, 150.0), (2002, 12, 110.0)])),
        "variance_not_estimable",
        "cells",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    (
        "mack_zero_latest_observed",
        lambda: mk(tri(replace(BASE, (2004, 12), 0.0)), zero_cells="observed"),
        "variance_not_estimable",
        "zero_cells",
        None,
        [c(2004, 12, 0.0)],
        [],
        [],
        {},
    ),
    (
        "mack_two_zero_starts_observed",
        lambda: mk(tri(Z2), zero_cells="observed"),
        "variance_not_estimable",
        "zero_cells",
        None,
        [c(2002, 12, 0.0), c(2003, 12, 0.0)],
        [(12, 24)],
        [],
        {},
    ),
    (
        "mack_zero_column_missing",
        lambda: mk(tri(ZERO_COL)),
        "no_link_ratio",
        "zero_cells",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    (
        "mack_zero_column_observed",
        lambda: mk(tri(ZERO_COL), zero_cells="observed"),
        "no_link_ratio",
        "zero_cells",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    (
        "mack_sigma_gap_mack_rule",
        lambda: mk(tri(Z3), sigma_rule="mack"),
        "variance_not_estimable",
        "sigma_rule",
        None,
        [],
        [(24, 36)],
        [],
        {},
    ),
    (
        "mack_sigma_gap_log_linear",
        lambda: mk(tri(Z3)),
        "variance_not_estimable",
        "sigma_rule",
        None,
        [],
        [(24, 36), (36, 48)],
        [],
        {},
    ),
    (
        "mack_overflow",
        lambda: mk(tri([(o, d, v * 1e305) for o, d, v in BASE])),
        "result_not_finite",
        "cells",
        None,
        [c(2001), c(2002), c(2003), c(2004)],
        [],
        [],
        {},
    ),
    (
        "chain_ladder_overflow",
        lambda: cl(tri([(2000, 12, 1e308), (2000, 24, 1.5e308), (2001, 12, 1.2e308)])),
        "result_not_finite",
        "cells",
        None,
        [c(2001)],
        [],
        [],
        {},
    ),
    (
        # each ultimate is finite, their sum is not
        "chain_ladder_total_overflow",
        lambda: cl(tri([(2000, 12, 1e308), (2000, 24, 1e308), (2001, 12, 1.5e308)])),
        "result_not_finite",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        # a link ratio past the largest double, with its age given a factor of 1.0
        "chain_ladder_ratio_overflow",
        lambda: cl(
            tri([(2000, 12, 1e-300), (2000, 24, 1e10), (2001, 12, 1.0)]),
            unsupported_factor="unity",
        ),
        "result_not_finite",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    # the result
    (
        "to_polars_unknown",
        lambda: cl(T).to_polars("nope"),
        "invalid_option",
        "table",
        None,
        [],
        [],
        [],
        {"given": "nope"},
    ),
    (
        "to_polars_mack_link_ratios",
        lambda: mk(T).to_polars("link_ratios"),
        "invalid_option",
        "table",
        None,
        [],
        [],
        [],
        {"given": "link_ratios"},
    ),
]

#: Cases whose message may show an ISO date although the labels are years: a gap
#: names the missing period by its first day, and a few cases write dates as labels.
_DATES_EXPECTED = {
    "origin_gap",
    "origin_mid_month",
    "origin_before_year_1",
    "origin_two_spellings",
    "origins_not_whole_steps",
    "annual_dates_on_a_quarterly_step",
    "origin_gap_between_two_years",
}

_METHOD_OF = {cl: "chain_ladder", bf: "bornhuetter_ferguson", cc: "cape_cod", mk: "mack"}


@pytest.mark.parametrize(
    ("call", "reason", "option", "column", "cells", "links", "rows", "extra"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_every_refusal_through_the_front_door(
    call, reason, option, column, cells, links, rows, extra
):
    refusal = refusal_of(call)
    assert refusal.reason == reason
    assert refusal.kind == KIND[reason]
    assert refusal.option == option
    assert refusal.column == column
    assert refusal.method in _METHOD_OF.values()
    got = [(cell.origin, cell.dev_lag) for cell in refusal.cells]
    assert got == [(o, lag) for o, lag, _ in cells]
    for cell, (origin, _, value) in zip(refusal.cells, cells, strict=True):
        if value is not ...:
            assert cell.value == value or (value is None and cell.value is None), cell
        assert type(cell.origin) is type(origin), cell  # a text label stays text
    assert list(refusal.links) == links
    assert list(refusal.rows) == rows
    for name, value in extra.items():
        assert repr(getattr(refusal, name)) == repr(value), name
    json.dumps(refusal.to_dict(), allow_nan=False)


def test_the_method_is_named_on_every_refusal():
    assert refusal_of(lambda: bf(T, premium={}, expected_loss_ratio=0.7)).method == (
        "bornhuetter_ferguson"
    )
    assert refusal_of(lambda: cc(T, premium=PREM, decay=5)).method == "cape_cod"
    assert refusal_of(lambda: mk(tri(Z3))).method == "mack"
    assert refusal_of(lambda: cl(T, exclude=[(2001, 36)])).method == "chain_ladder"
    assert refusal_of(lambda: mk(T).to_polars("link_ratios")).method == "mack"


# -- the caller's labels, in every form -----------------------------------------------


def _forms():
    """(id, step, label of origin k, Arrow type or None) for each way to write an origin."""
    return [
        ("int_year", 12, lambda k: 2001 + k, None),
        ("text_year", 12, lambda k: str(2001 + k), None),
        ("text_quarter", 3, lambda k: f"2001Q{k + 1}", None),
        ("first_day_date", 12, lambda k: dt.date(2001 + k, 1, 1), pa.date32()),
        ("last_day_date", 12, lambda k: dt.date(2001 + k, 12, 31), pa.date32()),
        (
            "timestamp_with_zone",
            12,
            lambda k: dt.datetime(2001 + k, 12, 31, 23, 0, tzinfo=dt.UTC),
            pa.timestamp("us", tz="America/New_York"),
        ),
        ("dictionary_text", 12, lambda k: str(2001 + k), "dictionary"),
    ]


def _form_table(rows, label, kind) -> pa.Table:
    o, d, v = zip(*rows, strict=True)
    labels = [label(k) for k in o]
    if kind == "dictionary":
        origin = pa.array(labels).dictionary_encode()
    elif kind is not None:
        origin = pa.array(labels, kind)
    else:
        origin = pa.array(labels)
    return pa.table({"origin_period": origin, "dev_lag": list(d), "value": list(v)})


def _form_cases(step):
    """Cell-naming cases on a 4 x 4 staircase of origins 0..3, as (id, rows, call, cells).

    ``cells`` are (origin index, dev_lag) pairs; ``index`` None means premium's own label.
    """
    base = [(k, step * (j + 1), 100.0 + 10 * k + j) for k in range(4) for j in range(4 - k)]
    return base


@pytest.mark.parametrize(
    ("step", "label", "kind"), [f[1:] for f in _forms()], ids=[f[0] for f in _forms()]
)
def test_each_refusal_names_the_origin_as_the_caller_wrote_it(step, label, kind):
    base = _form_cases(step)
    clean = methods.chain_ladder(_form_table(base, label, kind), dev_grain_months=step)
    shown = clean.origins["origin"].to_pylist()  # the labels a result echoes
    premium = {label(k): 200.0 + k for k in range(4)}

    def rows_with(key, value):
        return [(o, d, value if (o, d) == key else v) for o, d, v in base]

    def run(call):
        return refusal_of(call)

    def table(rows):
        return _form_table(rows, label, kind)

    def expect(refusal, reason, pairs, *, own=False):
        """``own``: the origin is named as the argument at fault wrote it (an
        exclusion, a premium key), not as the cells did."""
        written = [label(k) for k in range(4)] if own else shown
        assert refusal.reason == reason, str(refusal)
        got = [(cell.origin, cell.dev_lag) for cell in refusal.cells]
        assert got == [(written[k], lag) for k, lag in pairs], str(refusal)
        for cell in refusal.cells:
            assert type(cell.origin) is type(written[0])
        text = str(refusal)
        for k, _ in pairs:
            label_text = written[k]
            if isinstance(label_text, dt.datetime):
                label_text = label_text.isoformat(sep=" ")
            elif isinstance(label_text, dt.date):
                label_text = label_text.isoformat()
            assert str(label_text) in text, text

    s = step
    kw = {"dev_grain_months": s}
    # row 16: an age of 0
    expect(run(lambda: cl(table([*base, (3, 0, 1.0)]), **kw)), "invalid_age", [(3, 0)])
    # row 21: a negative cumulative
    expect(
        run(lambda: cl(table(rows_with((1, 2 * s), -5.0)), **kw)),
        "negative_cumulative",
        [(1, 2 * s)],
    )
    # row 22: one cell twice
    expect(
        run(lambda: cl(table([*base, (1, 2 * s, 1.0)]), **kw)),
        "duplicate",
        [(1, 2 * s), (1, 2 * s)],
    )
    # row 25: an origin with no first cell (a kernel refusal, relabelled)
    no_first = [r for r in base if (r[0], r[1]) != (2, s)]
    expect(run(lambda: cl(table(no_first), **kw)), "not_run_off", [(2, s)])
    # row 26: a hole
    hole = [r for r in base if (r[0], r[1]) != (1, 2 * s)]
    expect(run(lambda: cl(table(hole), **kw)), "not_run_off", [(1, 2 * s)])
    # row 36: an exclusion off the step
    expect(
        run(lambda: cl(table(base), exclude=[(label(0), s + 1)], **kw)),
        "grain_mismatch",
        [(0, s + 1)],
        own=True,
    )
    # row 38: an exclusion naming no link ratio
    expect(
        run(lambda: cl(table(base), exclude=[(label(3), s)], **kw)),
        "not_in_triangle",
        [(3, s)],
        own=True,
    )
    # row 54: premium lacking an origin
    lacking = {key: v for key, v in premium.items() if key != label(3)}
    expect(
        run(lambda: bf(table(base), premium=lacking, expected_loss_ratio=0.7, **kw)),
        "origin_not_covered",
        [(3, None)],
    )
    # rows 56 and 57: a bad premium amount
    expect(
        run(
            lambda: bf(
                table(base), premium={**premium, label(2): 0.0}, expected_loss_ratio=0.7, **kw
            )
        ),
        "invalid_option",
        [(2, None)],
        own=True,
    )
    expect(
        run(lambda: cc(table(base), premium={**premium, label(2): "220"}, **kw)),
        "invalid_option",
        [(2, None)],
        own=True,
    )
    # row 61: Mack on a zero latest amount
    expect(
        run(lambda: mk(table(rows_with((3, s), 0.0)), zero_cells="observed", **kw)),
        "variance_not_estimable",
        [(3, s)],
    )
    # a kernel refusal naming cells, relabelled: Mack's sigma with one positive start
    z2 = rows_with((1, s), 0.0)
    z2 = [(o, d, 0.0 if (o, d) == (2, s) else v) for o, d, v in z2]
    expect(
        run(lambda: mk(table(z2), zero_cells="observed", **kw)),
        "variance_not_estimable",
        [(1, s), (2, s)],
    )


def test_premium_refusals_name_premiums_own_label():
    # the cells write years, premium writes year ends: an extra premium origin and a
    # text amount are named as premium wrote them, a lacking one as the cells did
    ends = {dt.date(y, 12, 31): v for y, v in PREM.items()}
    extra = refusal_of(
        lambda: bf(T, premium={**ends, dt.date(2005, 12, 31): 1.0}, expected_loss_ratio=0.7)
    )
    assert [cell.origin for cell in extra.cells] == [dt.date(2005, 12, 31)]
    text = refusal_of(
        lambda: bf(T, premium={**ends, dt.date(2003, 12, 31): "220"}, expected_loss_ratio=0.7)
    )
    assert [cell.origin for cell in text.cells] == [dt.date(2003, 12, 31)]
    assert "2003-12-31" in str(text)
    lacking = refusal_of(
        lambda: bf(
            T,
            premium={k: v for k, v in ends.items() if k.year != 2004},
            expected_loss_ratio=0.7,
        )
    )
    assert [cell.origin for cell in lacking.cells] == [2004]


# -- the kernel's refusal, re-raised in the caller's terms ----------------------------


def test_a_kernel_refusal_is_relabelled_by_period_not_by_position(monkeypatch):
    """Origins given out of order and written as year ends: the kernel names the
    periods by their first day, and each comes back with the caller's label."""
    rows = [(k, 12 * (j + 1), 100.0 + k + j) for k in range(4) for j in range(4 - k)]
    rows = rows[::-1]
    ends = {k: dt.date(2004 - k, 12, 31) for k in range(4)}
    # reverse the calendar too, so position and period disagree
    table = pa.table(
        {
            "origin_period": pa.array([ends[3 - o] for o, _, _ in rows], pa.date32()),
            "dev_lag": [d for _, d, _ in rows],
            "value": [v for _, _, v in rows],
        }
    )

    def refuse(grid, candidate, *, premium=None):
        raise Refusal(
            "no_link_ratio",
            "no link ratio for {cells}",
            option="unsupported_factor",
            cells=[RefusedCell(None, dt.date(2003, 1, 1), 12, 1.0)],
        )

    monkeypatch.setattr(methods, "_estimate_grid", refuse)
    refusal = refusal_of(lambda: methods.chain_ladder(table))
    assert refusal.cells[0].origin == dt.date(2003, 12, 31)
    assert "2003-12-31" in str(refusal) and "2003-01-01" not in str(refusal)
    assert refusal.method == "chain_ladder"


def test_a_plain_value_error_from_a_kernel_is_not_turned_into_a_refusal(monkeypatch):
    """Only a Refusal is relabelled; anything else from underneath is an ibnr defect
    and must reach the caller as it was raised."""

    def broken(grid, candidate, *, premium=None):
        raise ValueError("a defect in ibnr")

    monkeypatch.setattr(methods, "_estimate_grid", broken)
    with pytest.raises(ValueError, match="a defect in ibnr") as caught:
        methods.chain_ladder(T)
    assert type(caught.value) is ValueError


def test_a_kernel_refusal_keeps_the_kernel_line_as_its_last_frame():
    with pytest.raises(Refusal) as caught:
        methods.chain_ladder(T, exclude=[(2001, 36)])
    last = traceback.extract_tb(caught.value.__traceback__)[-1]
    assert Path(last.filename).name == "conventional.py", last
    assert caught.value.__suppress_context__


# -- 3. messages -------------------------------------------------------------------

_KERNEL_WORDS = (
    "candidate",
    "grid",
    "cohort",
    "dev step",
    "origin index",
    "sigma_j",
    "datetime.date(",
    "BF ",
    "GCC",
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_no_message_uses_a_kernel_word_or_a_kernel_date(case):
    name, call = case[0], case[1]
    text = str(refusal_of(call))
    for word in _KERNEL_WORDS:
        assert word not in text, (word, text)
    if name not in _DATES_EXPECTED:
        # the two example dates in the list of forms an origin may take are allowed
        examples = text.replace("(2020-01-01)", "").replace("(2020-12-31)", "")
        assert not _ISO_DATE.search(examples), text


@pytest.mark.parametrize(
    ("call", "phrase"),
    [
        (lambda: cl(tri(replace(BASE, (2002, 24), math.inf))), "value is infinite in 1 cell"),
        (lambda: cl(tri(drop(BASE, (2002, 24)))), "(2002, 24 months) is missing"),
        (lambda: cl(tri([*BASE, (2004, 24, 140.0)])), "past the latest diagonal"),
        (
            lambda: cl(T, exclude=[(2001, "12")]),
            "names development age '12', which is not a positive whole number of months",
        ),
        (
            lambda: cl(T, exclude=[(2001, 18)]),
            "18 months is not a development age of this triangle",
        ),
        (
            lambda: cl(T, drop_high=True, exhausted_exclusions="raise"),
            "drop_high would leave no link ratio from 36 to 48 months",
        ),
        (
            lambda: bf(T, premium=PREM, expected_loss_ratio=-0.1),
            "expected_loss_ratio must be a finite number of 0 or more, got -0.1",
        ),
        (lambda: cc(T, premium=PREM, decay=2.0), "decay must be a number from 0 to 1, got 2.0"),
        (
            lambda: bf(T, premium=None, expected_loss_ratio=0.7),
            "bornhuetter_ferguson needs premium",
        ),
        (
            lambda: bf(T, premium={**PREM, 2003: 0.0}, expected_loss_ratio=0.7),
            "premium must be a positive finite number for every origin",
        ),
        (
            lambda: mk(tri([(2001, 12, 100.0)])),
            "mack needs at least two development ages; this triangle has one",
        ),
        (
            lambda: mk(tri(Z2), zero_cells="observed"),
            "only 1 of the 3 link ratios from 12 to 24 months starts from a positive amount",
        ),
        (lambda: mk(tri(ZERO_COL)), "from 12 to 24 months"),
        (
            lambda: mk(tri([(o, d, v * 1e305) for o, d, v in BASE])),
            "is not a finite number: the amounts are too large",
        ),
        (lambda: cl(T, exclude=[(2001, 36)]), "no link ratio is left from 36 to 48 months"),
    ],
)
def test_reworded_messages_say_what_is_wrong(call, phrase):
    assert phrase in str(refusal_of(call))


# -- 4. nothing else escapes ---------------------------------------------------------

_FUZZ = json.loads(
    (Path(__file__).parent / "data" / "refusal_triangles.json").read_text(encoding="utf-8")
)

_OPTION_POOL = {
    "average": ["volume", "simple", "median", "x", 1, None, True],
    "history_periods": [None, 1, 3, 0, -1, 2.5, True, "3", 10**12],
    "drop_high": [False, True, 1, "yes", None],
    "drop_low": [False, True, 0, None],
    "unsupported_factor": ["raise", "unity", "x", None],
    "exhausted_exclusions": ["keep", "raise", "x"],
    "zero_cells": ["missing", "observed", "x", True],
    "dev_grain_months": [12, 12, 12, 3, 0, -12, 2.5, "12", True, 10**12],
    "sigma_rule": ["log_linear", "mack", "x", None],
    "expected_loss_ratio": [0.7, 0.0, 1.2, -0.1, math.nan, math.inf, "0.7", True, None, 1e305],
    "decay": [1.0, 0.0, 0.5, 2.0, -1, math.nan, "0.5", True, None],
}
_PER_METHOD = {
    "chain_ladder": (
        "average",
        "history_periods",
        "drop_high",
        "drop_low",
        "unsupported_factor",
        "exhausted_exclusions",
        "zero_cells",
        "dev_grain_months",
    ),
    "mack": ("sigma_rule", "zero_cells", "dev_grain_months"),
}
_PER_METHOD["bornhuetter_ferguson"] = (*_PER_METHOD["chain_ladder"], "expected_loss_ratio")
_PER_METHOD["cape_cod"] = (*_PER_METHOD["chain_ladder"], "decay")


def _fuzz_case(rng, rows):
    """One to three random edits; the last kind (scaling a cell) usually leaves a
    triangle the methods answer, so the no-NaN half of the check is exercised too."""
    rows = [list(r) for r in rows]
    for _ in range(int(rng.integers(1, 4))):
        edit = int(rng.integers(0, 10))
        if not rows:
            break
        i = int(rng.integers(0, len(rows)))
        if edit >= 7:
            rows[i][2] = rows[i][2] * float(rng.uniform(0.5, 2.0))
        elif edit == 0:
            rows.pop(i)
        elif edit == 1:
            rows.append(list(rows[i]))
        elif edit == 2:
            rows[i][2] = -rows[i][2]
        elif edit == 3:
            rows[i][2] = [math.nan, math.inf, -math.inf, 0.0, 1e305, 1e-300][
                int(rng.integers(0, 6))
            ]
        elif edit == 4:
            rows[i][0] = [rows[i][0] + 1, 97, "abc", str(rows[i][0]), 1850][int(rng.integers(0, 5))]
        elif edit == 5:
            rows[i][1] = rows[i][1] + int(rng.integers(1, 12))
        else:
            # a whole column of zeros at one age
            lag = rows[i][1]
            for row in rows:
                if row[1] == lag:
                    row[2] = 0.0
    return rows


def _as_table(rows):
    kinds = {type(r[0]) for r in rows}
    if kinds <= {int}:
        origin = pa.array([r[0] for r in rows], pa.int64())
    else:
        origin = pa.array([str(r[0]) for r in rows])
    return pa.table(
        {
            "origin_period": origin,
            "dev_lag": pa.array([r[1] for r in rows], pa.int64()),
            "value": pa.array([float(r[2]) for r in rows], pa.float64()),
        }
    )


def _numbers_are_finite(result) -> None:
    for name in methods.TABLES:
        table = getattr(result, name)
        if table is None:
            continue
        for column in table.columns:
            if pa.types.is_floating(column.type):
                values = column.to_numpy(zero_copy_only=False)
                present = values[~column.is_null().to_numpy(zero_copy_only=False)]
                assert np.isfinite(present).all(), (name, column)


def test_a_seeded_fuzz_meets_nothing_but_refusals():
    """2,000 cases from five public triangles with one to three random edits and
    random options, valid and not. Every call returns a result whose numbers are
    all finite (a missing one is a null) or raises exactly Refusal. A RuntimeWarning
    is an error here, so a silent NaN path (an infinite amount subtracted from
    itself, say) fails the test."""
    rng = np.random.default_rng(20260924)
    names = sorted(_FUZZ)
    outcomes: dict[str, int] = {}
    for _ in range(2000):
        rows = _FUZZ[names[int(rng.integers(0, len(names)))]]
        edited = _fuzz_case(rng, rows)
        method = ("chain_ladder", "bornhuetter_ferguson", "cape_cod", "mack")[
            int(rng.integers(0, 4))
        ]
        options = {}
        for name in _PER_METHOD[method]:
            if rng.random() < 0.25:
                pool = _OPTION_POOL[name]
                options[name] = pool[int(rng.integers(0, len(pool)))]
        if method in ("bornhuetter_ferguson", "cape_cod"):
            origins = sorted({r[0] for r in rows})
            premium = {o: 1000.0 * (1 + k) for k, o in enumerate(origins)}
            if rng.random() < 0.2:
                premium[origins[int(rng.integers(0, len(origins)))]] = [
                    0.0,
                    -1.0,
                    math.nan,
                    math.inf,
                    "x",
                    None,
                ][int(rng.integers(0, 6))]
            options["premium"] = premium
        if method == "bornhuetter_ferguson":
            options.setdefault("expected_loss_ratio", 0.7)
        function = getattr(methods, method)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                result = function(_as_table(edited) if edited else pa.table({}), **options)
            except Refusal as refusal:
                assert type(refusal) is Refusal
                json.dumps(refusal.to_dict(), allow_nan=False)
                outcomes[refusal.reason] = outcomes.get(refusal.reason, 0) + 1
                continue
        _numbers_are_finite(result)
        outcomes["answered"] = outcomes.get("answered", 0) + 1
    assert outcomes["answered"] > 100, outcomes
    assert len(outcomes) > 10, outcomes


@pytest.mark.tieout
def test_every_clrd_triangle_is_answered_or_refused():
    """Every clrd company and line, paid and incurred, and the quarterly sample,
    through all four methods: each gives a result or a Refusal, never anything
    else. The 41 paid triangles with a negative cumulative are refused by name."""
    cl_module = pytest.importorskip("chainladder")
    import pandas as pd

    clrd = cl_module.load_sample("clrd")
    frame = clrd.to_frame(keepdims=True).reset_index()
    frame["origin_period"] = pd.to_datetime(frame["origin"]).dt.year
    counts: dict[tuple[str, str], dict[str, int]] = {}
    negatives = 0
    for _, group in frame.groupby(["GRNAME", "LOB"], sort=True):
        premium_rows = group[group["development"] == group["development"].min()]
        premium = dict(
            zip(premium_rows["origin_period"], premium_rows["EarnedPremNet"], strict=True)
        )
        for field in ("CumPaidLoss", "IncurLoss"):
            rows = group[group[field].notna()]
            cells = {
                "origin_period": rows["origin_period"].astype("int64").tolist(),
                "dev_lag": rows["development"].astype("int64").tolist(),
                "value": rows[field].astype(float).tolist(),
            }
            for method, extra in (
                ("chain_ladder", {}),
                ("bornhuetter_ferguson", {"premium": premium, "expected_loss_ratio": 0.7}),
                ("cape_cod", {"premium": premium}),
                ("mack", {}),
            ):
                bucket = counts.setdefault((field, method), {})
                try:
                    result = getattr(methods, method)(cells, **extra)
                except Refusal as refusal:
                    assert type(refusal) is Refusal
                    bucket[refusal.reason] = bucket.get(refusal.reason, 0) + 1
                    if (field, method) == ("CumPaidLoss", "chain_ladder") and (
                        refusal.reason == "negative_cumulative"
                    ):
                        negatives += 1
                    continue
                _numbers_are_finite(result)
                bucket["answered"] = bucket.get("answered", 0) + 1
    assert negatives == 41, counts
    quarterly = cl_module.load_sample("quarterly")
    long = quarterly["paid"].to_frame(keepdims=True).reset_index()
    cells = {
        "origin_period": pd.to_datetime(long["origin"]).dt.date.tolist(),
        "dev_lag": long["development"].astype("int64").tolist(),
        "value": long["paid"].astype(float).tolist(),
    }
    for method in ("chain_ladder", "mack"):
        refusal = refusal_of(
            lambda method=method: getattr(methods, method)(cells, dev_grain_months=3)
        )
        assert refusal.reason == "grain_mismatch"


# -- 5. the kernels a service reaches besides the methods ---------------------------

SMALL = np.array(
    [
        [100.0, 150.0, 165.0, 170.0],
        [120.0, 180.0, 200.0, np.nan],
        [110.0, 165.0, np.nan, np.nan],
        [130.0, np.nan, np.nan, np.nan],
    ]
)


def _grid_of(matrix: np.ndarray, *, step: int = 12) -> dict:
    """The grid of a matrix of cumulatives, origin ``i`` starting ``step * i``
    months after 2010-01-01 (years on 12, quarters on 3)."""
    from ibnr.kernels.grid import grid_from_columns

    def start(i: int) -> dt.date:
        months = step * i
        return dt.date(2010 + months // 12, months % 12 + 1, 1)

    rows = [
        (start(i), step * (j + 1), float(v))
        for i, row in enumerate(matrix)
        for j, v in enumerate(row)
        if not np.isnan(v)
    ]
    o, d, v = zip(*rows, strict=True)
    return grid_from_columns(o, d, v, dev_grain_months=step, measure="cumulative")


def test_fit_mack_many_skips_a_refusal_and_raises_anything_else(monkeypatch):
    """``on_error="skip"`` records a cohort ibnr refuses, with its reason code,
    and raises any other exception: a defect is not a cohort the data rules out.
    Under ``on_error="raise"`` the refusal keeps its class and gains the cohort."""
    from ibnr.kernels import mack as mack_kernels

    from .conftest import make_multiline_triangle

    holed = SMALL.copy()
    holed[0, 1] = np.nan
    t = make_multiline_triangle(None, {"good": SMALL, "holed": holed})
    panel = mack_kernels.fit_mack_many(t, loss_field="paid_loss", on_error="skip")
    assert panel.reasons == {("0001", "holed"): "not_run_off"}
    refused = refusal_of(lambda: mack_kernels.fit_mack_many(t, loss_field="paid_loss"))
    assert refused.reason == "not_run_off"
    assert str(refused).startswith("cohort {") and "holed" in str(refused)

    real = mack_kernels.fit_mack_grid

    def broken(grid, **kwargs):
        if grid["segment"].get("line_of_business") == "good":
            raise ValueError("a defect in ibnr")
        return real(grid, **kwargs)

    monkeypatch.setattr(mack_kernels, "fit_mack_grid", broken)
    with pytest.raises(ValueError, match="a defect in ibnr") as caught:
        mack_kernels.fit_mack_many(t, loss_field="paid_loss", on_error="skip")
    assert not isinstance(caught.value, Refusal)
    with pytest.raises(Refusal, match="sigma_rule must be one of"):
        mack_kernels.fit_mack_many(t, loss_field="paid_loss", on_error="skip", sigma_rule="x")


def test_every_one_year_cdr_refusal_is_a_refusal_with_its_reason():
    """The one-year CDR is on a service's request path too (``/cdr``), so its
    refusals carry reason codes like the methods'."""
    from ibnr.kernels.cdr import (
        MackDiagonal,
        ODPBootstrapDiagonal,
        cdr_risk_measures,
        one_year_cdr,
        rereserve,
        simulate_one_year_cdr,
    )
    from ibnr.kernels.mack import fit_mack_grid

    fit = fit_mack_grid(_grid_of(SMALL))
    quarterly = fit_mack_grid(_grid_of(SMALL, step=3))
    zero_latest = SMALL.copy()
    zero_latest[3, 0] = 0.0
    missing = fit_mack_grid(_grid_of(zero_latest), zero_cells="missing")
    observed_zero = fit_mack_grid(_grid_of(zero_latest))
    backwards = SMALL.copy()
    backwards[1, 2] = 170.0  # 200 at 36 months becomes 170, below 180 at 24
    cases = [
        (lambda: one_year_cdr(quarterly), "not_supported", "dev_grain_months"),
        (lambda: simulate_one_year_cdr(quarterly, n_draws=5), "not_supported", "dev_grain_months"),
        (lambda: one_year_cdr(missing), "not_supported", "zero_cells"),
        (lambda: one_year_cdr(observed_zero), "variance_not_estimable", "zero_cells"),
        (lambda: MackDiagonal(process="x"), "invalid_option", "process"),
        (lambda: ODPBootstrapDiagonal(process="x"), "invalid_option", "process"),
        (
            lambda: ODPBootstrapDiagonal(process_noise=False, resample_residuals=False),
            "invalid_option",
            "process_noise",
        ),
        (lambda: simulate_one_year_cdr(fit, n_draws=0), "invalid_option", "n_draws"),
        (lambda: simulate_one_year_cdr(fit, generator="nope"), "invalid_option", "generator"),
        (lambda: simulate_one_year_cdr(fit, generator=1), "invalid_option", "generator"),
        (
            lambda: simulate_one_year_cdr(fit, generator="merz_wuthrich"),
            "invalid_option",
            "generator",
        ),
        (
            lambda: simulate_one_year_cdr(fit, generator="mack", process="gamma"),
            "invalid_option",
            "generator",
        ),
        (lambda: rereserve(fit, np.zeros((3, 2))), "invalid_option", "next_diagonal"),
        (
            lambda: simulate_one_year_cdr(
                fit_mack_grid(_grid_of(backwards)), n_draws=5, generator="odp_bootstrap"
            ),
            "negative_increment",
            "cells",
        ),
        (
            lambda: cdr_risk_measures(simulate_one_year_cdr(fit, n_draws=5, seed=1), (1.5,)),
            "invalid_option",
            "levels",
        ),
    ]
    for call, reason, option in cases:
        refusal = refusal_of(call)
        assert (refusal.reason, refusal.option) == (reason, option), str(refusal)
        json.dumps(refusal.to_dict(), allow_nan=False)
    # the cells a CDR refusal names are the ones at fault, by period and age
    assert [
        (c.origin_period, c.dev_lag) for c in refusal_of(lambda: one_year_cdr(missing)).cells
    ] == [(dt.date(2013, 1, 1), 12)]
    odp = refusal_of(
        lambda: simulate_one_year_cdr(
            fit_mack_grid(_grid_of(backwards)), n_draws=5, generator="odp_bootstrap"
        )
    )
    assert [(c.origin_period, c.dev_lag, c.value) for c in odp.cells] == [
        (dt.date(2011, 1, 1), 36, -10.0)
    ]


def test_every_reason_is_documented_for_a_caller():
    """The closed list is only useful if a caller can read what each code means:
    the ``ibnr.errors`` docstring says, and the chainladder page lists them."""
    page = (Path(__file__).parents[1] / "docs" / "coming-from-chainladder.md").read_text(
        encoding="utf-8"
    )
    for reason in REASONS:
        assert f"``{reason}``:" in errors.__doc__, reason
        assert f"`{reason}`" in page, reason
