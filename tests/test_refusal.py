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
# the same staircase with quarterly origins and quarterly ages
QT = tri([(f"2001Q{o - 2000}", d // 4, v) for o, d, v in BASE])
ZERO_COL = [(o, d, 0.0 if d == 12 else v) for o, d, v in BASE]
Z2 = replace(replace(BASE, (2002, 12), 0.0), (2003, 12), 0.0)
Z3 = replace(replace(BASE, (2002, 36), 0.0), (2002, 24), 0.0)
cl, bf, bk, cc, mk, tg, cdr, ml = (
    methods.chain_ladder,
    methods.bornhuetter_ferguson,
    methods.benktander,
    methods.cape_cod,
    methods.mack,
    methods.tweedie_glm,
    methods.one_year_cdr,
    methods.ml_development,
)
#: few draws, so the one-year CDR's cases run fast
FEW = {"n_draws": 20, "seed": 1}
RF = {"estimator": "random_forest"}
GB = {"estimator": "gradient_boosting"}


def fitted(call):
    """A case whose refusal comes after a scikit-learn fit: skipped without it.

    The other ``ml_development`` cases are refused before scikit-learn is
    imported, so they run in every CI leg.
    """

    def run():
        pytest.importorskip("sklearn")
        return call()

    return run


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
    assert imported <= {"__future__", "datetime", "math", "numbers", "re", "dataclasses", "typing"}


def test_methods_exports_refusal():
    assert methods.Refusal is Refusal
    assert "Refusal" in methods.__all__


def test_options_is_the_one_option_when_only_one_is_named():
    assert Refusal("invalid_option", "x", option="average").options == ("average",)
    assert Refusal("invalid_option", "x").options == ()
    both = Refusal("invalid_option", "x", option="average", options=("average", "method"))
    assert both.options == ("average", "method")


def test_every_public_name_of_a_refusal_is_documented():
    """The reference page lists a refusal's public names, so each one is part of
    the contract and has to be in the class docstring; anything else is private."""
    refusal = full_refusal()
    inherited = set(dir(ValueError("x")))
    public = {name for name in dir(refusal) if not name.startswith("_")} - inherited
    fields = {"reason", "kind", "method", "option", "column", "options", "given"}
    fields |= {"cells", "links", "rows", "count"}
    assert public == fields | {"to_dict"}
    for name in fields:
        assert f"    {name} :" in Refusal.__doc__, name


def test_a_placeholder_in_the_callers_text_is_not_filled_in():
    """A label or value that spells a placeholder is printed as written, once."""
    refusal = Refusal(
        "duplicate",
        "{cells} twice, given {given}",
        cells=[RefusedCell("{given}", year(2001), 12, 1.0)],
        given="{cells}",
    )
    assert str(refusal) == "({given}, 12 months) twice, given '{cells}'"


# -- 2. every refusal reachable from ibnr.methods ------------------------------------


def c(origin, lag=None, value=...):
    """An expected cell: (origin as written, dev_lag[, value])."""
    return (origin, lag, value)


def fixture_rows(name: str, scale: float = 1.0) -> list[tuple]:
    """One of the public triangles in ``data/refusal_triangles.json``, scaled."""
    return [(o, d, v * scale) for o, d, v in _FUZZ[name]]


def years(cells: dict, step: int = 12) -> pa.Table:
    """A triangle from ``{origin: [cumulative at each age]}``."""
    return tri([(o, (j + 1) * step, float(v)) for o, vs in cells.items() for j, v in enumerate(vs)])


#: an origin with losses only where the others have none: no finite Poisson fit
GLM_BOUNDARY = years({2001: [0] * 5, 2002: [0] * 4, 2003: [0, 0, 50], 2004: [0, 60], 2005: [40]})
#: cumulatives that fall after the first age: the forest (seed 42) fits 2003's
#: cumulative at 24 months at -5.5, though 2003's latest amount is 1
ML_FALLING = years({2001: [136, 3, 2, 6.8], 2002: [75, 3, 90], 2003: [104, 1], 2004: [52]})
#: cumulatives all zero or more; boosting (seed 42) projects 2004 to -25.7
ML_NEGATIVE = years(
    {2001: [100, 2, 3, 4, 5], 2002: [110, 3, 4, 5], 2003: [95, 1, 2], 2004: [30, 1], 2005: [120]}
)


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
        # dates one step apart elsewhere in the axis, so the two years apart is a gap
        "origin_gap_between_dates",
        lambda: cl(
            tri(
                [
                    (dt.date(2001, 1, 1), 12, 1.0),
                    (dt.date(2001, 1, 1), 24, 2.0),
                    (dt.date(2002, 1, 1), 12, 1.0),
                    (dt.date(2004, 1, 1), 12, 1.0),
                ]
            )
        ),
        "origin_gap",
        "cells",
        "origin_period",
        [c(dt.date(2002, 1, 1)), c(None), c(dt.date(2004, 1, 1))],
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
            ("average", "least_squares"),
            ("average", "Regression"),
            ("average", 1.0),
            ("history_periods", 0),
            ("history_periods", 2.5),
            ("history_periods", True),
            ("drop_high", -1),
            ("drop_high", 1.5),
            ("drop_high", [1, 0, 0]),
            ("drop_low", "yes"),
            ("preserve", 0),
            ("preserve", True),
            ("preserve", 2.0),
            ("drop_above", math.nan),
            ("drop_above", True),
            ("drop_below", "1.0"),
            ("drop_below", [1.0]),
            ("trim_ties", "ratio"),
            ("exclude_valuations", "2002"),
            ("exclude_valuations", 2002),
            ("unsupported_factor", "x"),
            ("exhausted_exclusions", "x"),
            ("zero_cells", "x"),
        ]
    ],
    (
        "bounds_crossed",
        lambda: cl(T, drop_below=1.5, drop_above=1.5),
        "invalid_option",
        "drop_below",
        None,
        [],
        [],
        [],
        {"options": ("drop_below", "drop_above")},
    ),
    (
        "bounds_exhausted_raise",
        lambda: cl(T, drop_above=1.2, exhausted_exclusions="raise"),
        "exclusions_exhausted",
        "exhausted_exclusions",
        None,
        [],
        [(12, 24)],
        [],
        {"options": ("exhausted_exclusions", "drop_above", "preserve")},
    ),
    (
        "trims_below_preserve_raise",
        lambda: cl(T, drop_high=1, preserve=2, exhausted_exclusions="raise"),
        "exclusions_exhausted",
        "exhausted_exclusions",
        None,
        [],
        [(24, 36)],
        [],
        {"options": ("exhausted_exclusions", "drop_high", "preserve")},
    ),
    *[
        (
            f"valuation_{label}",
            lambda value=value, dev_grain=dev_grain: cl(
                T if dev_grain == 12 else QT,
                dev_grain_months=dev_grain,
                exclude_valuations=[value],
            ),
            reason,
            "exclude_valuations",
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for label, value, dev_grain, reason in [
            ("text", "abc", 12, "unreadable_label"),
            ("two_digit_year", 97, 12, "unreadable_label"),
            ("float_year", 2002.0, 12, "unreadable_label"),
            ("bool", True, 12, "unreadable_label"),
            ("mid_month", dt.date(2002, 6, 15), 12, "unreadable_label"),
            ("off_diagonal", "2002-06-30", 12, "grain_mismatch"),
            ("quarter_on_annual", "2002Q4", 12, "grain_mismatch"),
            ("year_on_quarterly", 2001, 3, "grain_mismatch"),
            ("first_diagonal", 2001, 12, "not_in_triangle"),
            ("after_the_latest", "2005-12-31", 12, "not_in_triangle"),
        ]
    ],
    (
        "valuation_twice",
        lambda: cl(T, exclude_valuations=[2002, "2002-12-31"]),
        "duplicate",
        "exclude_valuations",
        None,
        [],
        [],
        [],
        {"given": "2002-12-31"},
    ),
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
        {"options": ("exhausted_exclusions", "drop_high")},
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
    *[
        (
            f"n_iters_{name}_{value!r}",
            lambda call=call, value=value: call(n_iters=value),
            "invalid_option",
            "n_iters",
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for name, call in (
            ("benktander", lambda **o: bk(T, premium=PREM, expected_loss_ratio=0.7, **o)),
            ("cape_cod", lambda **o: cc(T, premium=PREM, **o)),
        )
        for value in (0, -1, 2.5, True, "2", None, 10_001, 10**9)
    ],
    *[
        (
            f"trend_{value!r}",
            lambda value=value: cc(T, premium=PREM, trend=value),
            "invalid_option",
            "trend",
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for value in (-1.0, -1.5, math.inf, math.nan, "0.05", True, None)
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
        {"options": ("sigma_rule", "zero_cells")},
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
        {"options": ("sigma_rule", "zero_cells")},
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
        [(12, 24)],
        [],
        {},
    ),
    (
        # each link ratio is finite, their sum is not: the factor itself overflows,
        # and a factor of 1.0 in its place would be a wrong answer, not a fallback
        "chain_ladder_factor_overflow",
        lambda: cl(years({2020: [1.0, 1e308, 1.0], 2021: [1.0, 1e308], 2022: [1.0]})),
        "result_not_finite",
        "cells",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    (
        "chain_ladder_factor_overflow_unity",
        lambda: cl(
            years({2020: [1.0, 1e308, 1.0], 2021: [1.0, 1e308], 2022: [1.0]}),
            unsupported_factor="unity",
        ),
        "result_not_finite",
        "cells",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    (
        # the link ratio 1e10 / 1e-300 is past the largest double; no link has a zero
        "mack_link_ratio_overflow",
        lambda: mk(years({1981: [1e-300, 1e10, 1e10], 1982: [1e-300, 1e10], 1983: [0.0]})),
        "result_not_finite",
        "cells",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    (
        # every origin with a link ratio from 108 to 120 months closes at zero: the
        # factor there is 0, so every ultimate would be 0
        "mack_zero_last_factor_observed",
        lambda: mk(tri(replace(fixture_rows("raa"), (1981, 120), 0.0)), zero_cells="observed"),
        "no_link_ratio",
        "cells",
        None,
        [],
        [(108, 120)],
        [],
        {},
    ),
    (
        # 5e-324 / 170 is below the smallest double, so the factor is 0 here too
        "mack_factor_underflows_to_zero",
        lambda: mk(tri(replace(BASE, (2001, 48), 5e-324))),
        "no_link_ratio",
        "cells",
        None,
        [],
        [(36, 48)],
        [],
        {},
    ),
    # Mack with development options
    (
        "mack_average_median",
        lambda: mk(T, average="median"),
        "not_supported",
        "average",
        None,
        [],
        [],
        [],
        {"given": "median"},
    ),
    (
        "mack_average_geometric",
        lambda: mk(T, average="geometric"),
        "not_supported",
        "average",
        None,
        [],
        [],
        [],
        {"given": "geometric"},
    ),
    (
        "mack_average_number",
        lambda: mk(T, average=1),
        "invalid_option",
        "average",
        None,
        [],
        [],
        [],
        {"given": 1},
    ),
    (
        "mack_history_periods_1",
        lambda: mk(T, history_periods=1),
        "variance_not_estimable",
        "history_periods",
        None,
        [],
        [],
        [],
        {"given": 1},
    ),
    (
        "mack_age_left_with_no_ratio",
        lambda: mk(T, exclude=[(2001, 36)]),
        "no_link_ratio",
        "exclude",
        None,
        [],
        [(36, 48)],
        [],
        {},
    ),
    (
        "mack_options_leave_one_ratio_everywhere",
        lambda: mk(T, exclude=[(2001, 24)], drop_high=2),
        "variance_not_estimable",
        "exclude",
        None,
        [],
        [(12, 24), (24, 36), (36, 48)],
        [],
        {"options": ("exclude", "drop_high")},
    ),
    (
        "mack_sigma_gap_left_by_a_drop",
        lambda: mk(T, drop_low=2, sigma_rule="mack"),
        "variance_not_estimable",
        "sigma_rule",
        None,
        [],
        [(12, 24)],
        [],
        {"options": ("sigma_rule", "drop_low")},
    ),
    (
        "mack_observed_zero_with_an_option",
        lambda: mk(tri(replace(BASE, (2002, 12), 0.0)), zero_cells="observed", average="simple"),
        "not_supported",
        "zero_cells",
        None,
        [c(2002, 12, 0.0)],
        [],
        [],
        {},
    ),
    (
        "mack_regression_zero_latest",
        lambda: mk(tri(replace(BASE, (2004, 12), 0.0)), average="regression"),
        "not_supported",
        "average",
        None,
        [c(2004, 12, 0.0)],
        [],
        [],
        {"options": ("average", "zero_cells")},
    ),
    (
        "mack_trims_exhausted",
        lambda: mk(T, drop_high=1, exhausted_exclusions="raise"),
        "exclusions_exhausted",
        "exhausted_exclusions",
        None,
        [],
        [(36, 48)],
        [],
        {"options": ("exhausted_exclusions", "drop_high")},
    ),
    (
        "mack_exclusion_not_in_triangle",
        lambda: mk(T, exclude=[(2004, 12)]),
        "not_in_triangle",
        "exclude",
        None,
        [c(2004, 12)],
        [],
        [],
        {},
    ),
    (
        "mack_valuation_not_in_triangle",
        lambda: mk(T, exclude_valuations=["2005-12-31"]),
        "not_in_triangle",
        "exclude_valuations",
        None,
        [],
        [],
        [],
        {"given": "2005-12-31"},
    ),
    (
        "dev_lag_past_whole_numbers",
        lambda: cl(pa.table({"origin_period": [2001], "dev_lag": [1e30], "value": [1.0]})),
        "invalid_age",
        "cells",
        "dev_lag",
        [],
        [],
        [0],
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
        "glm_power_between_0_and_1",
        lambda: tg(T, power=0.5),
        "invalid_option",
        "power",
        None,
        [],
        [],
        [],
        {"given": 0.5},
    ),
    (
        "glm_power_negative",
        lambda: tg(T, power=-1),
        "invalid_option",
        "power",
        None,
        [],
        [],
        [],
        {"given": -1},
    ),
    (
        "glm_power_nan",
        lambda: tg(T, power=math.nan),
        "invalid_option",
        "power",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "glm_power_bool",
        lambda: tg(T, power=True),
        "invalid_option",
        "power",
        None,
        [],
        [],
        [],
        {"given": True},
    ),
    (
        "glm_power_text",
        lambda: tg(T, power="1"),
        "invalid_option",
        "power",
        None,
        [],
        [],
        [],
        {"given": "1"},
    ),
    (
        "glm_link",
        lambda: tg(T, link="logit"),
        "invalid_option",
        "link",
        None,
        [],
        [],
        [],
        {"given": "logit"},
    ),
    (
        "glm_origin",
        lambda: tg(T, origin="year"),
        "invalid_option",
        "origin",
        None,
        [],
        [],
        [],
        {"given": "year"},
    ),
    (
        "glm_calendar",
        lambda: tg(T, calendar="linear"),
        "invalid_option",
        "calendar",
        None,
        [],
        [],
        [],
        {"given": "linear"},
    ),
    (
        "glm_projection",
        lambda: tg(T, projection="ultimate"),
        "invalid_option",
        "projection",
        None,
        [],
        [],
        [],
        {"given": "ultimate"},
    ),
    (
        "glm_max_iter_zero",
        lambda: tg(T, max_iter=0),
        "invalid_option",
        "max_iter",
        None,
        [],
        [],
        [],
        {"given": 0},
    ),
    (
        "glm_max_iter_fraction",
        lambda: tg(T, max_iter=2.5),
        "invalid_option",
        "max_iter",
        None,
        [],
        [],
        [],
        {"given": 2.5},
    ),
    (
        "glm_trend_beside_origin_factors",
        lambda: tg(T, calendar="trend"),
        "invalid_option",
        "calendar",
        None,
        [],
        [],
        [],
        {"options": ("calendar", "origin"), "given": "trend"},
    ),
    (
        "glm_increments_without_origin_factors",
        lambda: tg(T, origin="none", projection="increments"),
        "invalid_option",
        "projection",
        None,
        [],
        [],
        [],
        {"options": ("projection", "origin"), "given": "increments"},
    ),
    (
        "glm_tail",
        lambda: tg(T, tail=1.05),
        "not_supported",
        "tail",
        None,
        [],
        [],
        [],
        {"given": 1.05},
    ),
    (
        "glm_negative_increment",
        lambda: tg(tri(replace(BASE, (2001, 48), 160.0))),
        "negative_increment",
        "cells",
        None,
        [c(2001, 48, 160.0)],
        [],
        [],
        {},
    ),
    (
        "glm_zero_increment",
        lambda: tg(tri(replace(BASE, (2001, 48), 170.0)), power=2),
        "zero_increment",
        "cells",
        None,
        [c(2001, 48, 170.0)],
        [],
        [],
        {},
    ),
    (
        "glm_no_losses",
        lambda: tg(tri([(o, d, 0.0) for o, d, _ in BASE])),
        "not_identified",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "glm_not_identified",
        lambda: tg(years({2001: [100, 150, 170]}), origin="none", calendar="trend"),
        "not_identified",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "glm_did_not_converge",
        lambda: tg(T, max_iter=1),
        "did_not_converge",
        "max_iter",
        None,
        [],
        [],
        [],
        {"given": 1},
    ),
    (
        "glm_boundary",
        lambda: tg(GLM_BOUNDARY),
        "degenerate_fit",
        "cells",
        None,
        [c(2003, 12, 0.0), c(2003, 24, 0.0), c(2004, 12, 0.0)],
        [],
        [],
        {},
    ),
    (
        "glm_falls_on_balance_power_0",
        lambda: tg(
            years(
                {2001: [100, 200, 190, 195], 2002: [110, 220, 200], 2003: [120, 250], 2004: [130]}
            ),
            power=0,
        ),
        "degenerate_fit",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "glm_identity_zero_age",
        lambda: tg(
            years(
                {2001: [100, 150, 170, 170], 2002: [110, 160, 185], 2003: [120, 175], 2004: [130]}
            ),
            link="identity",
        ),
        "negative_fitted_mean",
        "link",
        None,
        [c(2001, 48, 170.0)],
        [],
        [],
        {"given": "identity"},
    ),
    (
        "glm_identity_below_zero",
        lambda: tg(years({2001: [100, 101], 2002: [1]}), link="identity"),
        "negative_fitted_mean",
        "link",
        None,
        [c(2002, 24)],
        [],
        [],
        {"given": "identity"},
    ),
    (
        "glm_negative_ultimate",
        lambda: tg(years({2001: [100, 10], 2002: [1]}), power=0, link="identity"),
        "negative_projection",
        "projection",
        None,
        [c(2002)],
        [],
        [],
        {"given": "pattern"},
    ),
    (
        "to_polars_glm_link_ratios",
        lambda: tg(T).to_polars("link_ratios"),
        "invalid_option",
        "table",
        None,
        [],
        [],
        [],
        {"given": "link_ratios"},
    ),
    (
        "to_polars_chain_ladder_cells",
        lambda: cl(T).to_polars("cells"),
        "invalid_option",
        "table",
        None,
        [],
        [],
        [],
        {"given": "cells"},
    ),
    # the one-year claims development result
    *[
        (
            f"cdr_n_draws_{name}",
            lambda value=value: cdr(T, n_draws=value),
            "invalid_option",
            "n_draws",
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for name, value in (
            ("zero", 0),
            ("negative", -5),
            ("bool", True),
            ("float", 2.5),
            # four origins: 25,000,001 draws hold 100,000,004 numbers
            ("past_the_limit", 25_000_001),
            ("huge", 2**63),
        )
    ],
    *[
        (
            f"cdr_seed_{name}",
            lambda value=value: cdr(T, seed=value, n_draws=5),
            "invalid_option",
            "seed",
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for name, value in (("negative", -1), ("bool", False), ("text", "1"), ("float", 1.5))
    ],
    *[
        (
            f"cdr_quantiles_{name}",
            lambda value=value: cdr(T, quantiles=value, **FEW),
            "invalid_option",
            "quantiles",
            None,
            [],
            [],
            [],
            {"given": given},
        )
        for name, value, given in (
            ("one_number", 0.995, 0.995),
            ("percent", [50, 99.5], (50, 99.5)),
            ("one", (0.5, 1.0), (0.5, 1.0)),
            ("zero", (0.0,), (0.0,)),
            ("nan", (math.nan,), (math.nan,)),
            ("text", ("0.5",), ("0.5",)),
            ("bool", (True,), (True,)),
            ("string", "0.5", "0.5"),
        )
    ],
    (
        "cdr_process",
        lambda: cdr(T, process="poisson", **FEW),
        "invalid_option",
        "process",
        None,
        [],
        [],
        [],
        {"given": "poisson"},
    ),
    (
        "cdr_parameter_risk",
        lambda: cdr(T, parameter_risk="no", **FEW),
        "invalid_option",
        "parameter_risk",
        None,
        [],
        [],
        [],
        {"given": "no"},
    ),
    (
        "cdr_sigma_rule",
        lambda: cdr(T, sigma_rule="linear", **FEW),
        "invalid_option",
        "sigma_rule",
        None,
        [],
        [],
        [],
        {"given": "linear"},
    ),
    (
        "cdr_zero_cells",
        lambda: cdr(T, zero_cells="zero", **FEW),
        "invalid_option",
        "zero_cells",
        None,
        [],
        [],
        [],
        {"given": "zero"},
    ),
    (
        "cdr_quarterly",
        lambda: cdr(
            tri(
                [
                    ("2001Q1", 3, 100.0),
                    ("2001Q1", 6, 150.0),
                    ("2001Q1", 9, 170.0),
                    ("2001Q2", 3, 110.0),
                    ("2001Q2", 6, 160.0),
                    ("2001Q3", 3, 120.0),
                ]
            ),
            dev_grain_months=3,
            **FEW,
        ),
        "not_supported",
        "dev_grain_months",
        None,
        [],
        [],
        [],
        {"given": 3},
    ),
    (
        "cdr_zero_latest",
        lambda: cdr(tri(replace(BASE, (2004, 12), 0.0)), **FEW),
        "variance_not_estimable",
        "cells",
        None,
        [c(2004, 12, 0.0)],
        [],
        [],
        {},
    ),
    (
        "cdr_zero_latest_missing",
        lambda: cdr(tri(replace(BASE, (2004, 12), 0.0)), zero_cells="missing", **FEW),
        "not_supported",
        "zero_cells",
        None,
        [c(2004, 12, 0.0)],
        [],
        [],
        {"given": "missing"},
    ),
    (
        "cdr_zero_inside_missing",
        lambda: cdr(tri(replace(BASE, (2001, 12), 0.0)), zero_cells="missing", **FEW),
        "not_supported",
        "zero_cells",
        None,
        [c(2001, 12, 0.0)],
        [],
        [],
        {"given": "missing"},
    ),
    # refused as not supported before the fit, whose own refusal under "missing"
    # (a sigma with nothing to fill it from) would be about a rule not taken here
    (
        "cdr_zeros_missing_before_the_fit",
        lambda: cdr(tri(Z3), zero_cells="missing", **FEW),
        "not_supported",
        "zero_cells",
        None,
        [c(2002, 24, 0.0), c(2002, 36, 0.0)],
        [],
        [],
        {"given": "missing"},
    ),
    (
        "cdr_one_age",
        lambda: cdr(tri([(2001, 12, 100.0)]), **FEW),
        "variance_not_estimable",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "cdr_one_link_ratio_each",
        lambda: cdr(years({2001: [100, 150], 2002: [110]}), **FEW),
        "variance_not_estimable",
        "cells",
        None,
        [],
        [(12, 24)],
        [],
        {},
    ),
    (
        "cdr_negative",
        lambda: cdr(tri(replace(BASE, (2002, 24), -1.0)), **FEW),
        "negative_cumulative",
        "cells",
        "value",
        [c(2002, 24, -1.0)],
        [],
        [],
        {},
    ),
    (
        "to_polars_cdr_link_ratios",
        lambda: cdr(T, **FEW).to_polars("link_ratios"),
        "invalid_option",
        "table",
        None,
        [],
        [],
        [],
        {"given": "link_ratios"},
    ),
    # machine-learning development: every option is checked before scikit-learn
    *[
        (
            f"ml_{option}_{name}",
            lambda option=option, value=value, extra=extra: ml(T, **{**extra, option: value}),
            "invalid_option",
            option,
            None,
            [],
            [],
            [],
            {"given": value},
        )
        for option, name, value, extra in (
            ("estimator", "none", None, {}),
            ("estimator", "tweedie", "tweedie", {}),
            ("seed", "none", None, RF),
            ("seed", "negative", -1, RF),
            ("seed", "too_large", 2**32, RF),
            ("seed", "bool", True, RF),
            ("seed", "float", 1.5, RF),
            ("seed", "text", "1", RF),
            ("n_estimators", "zero", 0, RF),
            ("n_estimators", "float", 10.0, GB),
            ("max_depth", "zero", 0, GB),
            ("max_depth", "bool", True, RF),
            ("min_samples_leaf", "zero", 0, RF),
            ("learning_rate", "zero", 0.0, GB),
            ("learning_rate", "nan", math.nan, GB),
            ("learning_rate", "text", "0.1", GB),
            ("response", "text", "increments", RF),
            ("origin", "text", "year", RF),
            ("calendar", "text", "linear", RF),
            ("zero_cells", "text", "zero", RF),
            ("unsupported_factor", "text", "one", RF),
        )
    ],
    (
        "ml_learning_rate_on_a_forest",
        lambda: ml(T, learning_rate=0.1, **RF),
        "invalid_option",
        "learning_rate",
        None,
        [],
        [],
        [],
        {"given": 0.1, "options": ("learning_rate", "estimator")},
    ),
    (
        "ml_min_samples_leaf_on_boosting",
        lambda: ml(T, min_samples_leaf=2, **GB),
        "invalid_option",
        "min_samples_leaf",
        None,
        [],
        [],
        [],
        {"given": 2, "options": ("min_samples_leaf", "estimator")},
    ),
    (
        "ml_tail",
        lambda: ml(T, tail=1.05, **RF),
        "not_supported",
        "tail",
        None,
        [],
        [],
        [],
        {"given": 1.05},
    ),
    (
        "ml_one_cell",
        lambda: ml(tri([(2001, 12, 100.0)]), **RF),
        "not_identified",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "ml_one_cell_not_zero",
        lambda: ml(years({2001: [0, 0], 2002: [5]}), zero_cells="missing", **RF),
        "not_identified",
        "zero_cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "ml_no_losses",
        lambda: ml(tri([(o, d, 0.0) for o, d, _ in BASE]), **RF),
        "not_identified",
        "cells",
        None,
        [],
        [],
        [],
        {},
    ),
    (
        "ml_an_age_of_zeros",
        lambda: ml(tri(replace(BASE, (2001, 48), 0.0)), zero_cells="missing", **RF),
        "no_link_ratio",
        "unsupported_factor",
        None,
        [c(2001, 48, 0.0)],
        [(36, 48)],
        [],
        {"options": ("unsupported_factor", "zero_cells")},
    ),
    (
        "ml_first_age_of_zeros",
        lambda: ml(tri(ZERO_COL), zero_cells="missing", **GB),
        "no_link_ratio",
        "unsupported_factor",
        None,
        [c(2001, 12, 0.0), c(2002, 12, 0.0), c(2003, 12, 0.0), c(2004, 12, 0.0)],
        [],
        [],
        {"options": ("unsupported_factor", "zero_cells")},
    ),
    (
        "ml_fitted_latest_not_positive",
        fitted(lambda: ml(ML_FALLING, seed=42, **RF)),
        "negative_projection",
        "estimator",
        None,
        [c(2003, 24)],
        [],
        [],
        {"given": "random_forest"},
    ),
    (
        "ml_negative_ultimate",
        fitted(lambda: ml(ML_NEGATIVE, seed=42, **GB)),
        "negative_projection",
        "estimator",
        None,
        [c(2004)],
        [],
        [],
        {"given": "gradient_boosting"},
    ),
    (
        "to_polars_ml_link_ratios",
        fitted(lambda: ml(T, **GB).to_polars("link_ratios")),
        "invalid_option",
        "table",
        None,
        [],
        [],
        [],
        {"given": "link_ratios"},
    ),
    (
        "to_polars_ml_coefficients",
        fitted(lambda: ml(T, **GB).to_polars("coefficients")),
        "invalid_option",
        "table",
        None,
        [],
        [],
        [],
        {"given": "coefficients"},
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
    "origin_gap_between_dates",
    # an excluded valuation is an evaluation date, and the refusal lists the triangle's
    "valuation_mid_month",
    "valuation_off_diagonal",
    "valuation_first_diagonal",
    "valuation_after_the_latest",
    "valuation_twice",
    "mack_valuation_not_in_triangle",
    # the message gives ['2020-12-31'] as an example of a list of valuations
    "exclude_valuations_'2002'",
    "exclude_valuations_2002",
}

_METHOD_OF = {
    cl: "chain_ladder",
    bf: "bornhuetter_ferguson",
    bk: "benktander",
    cc: "cape_cod",
    mk: "mack",
    tg: "tweedie_glm",
    cdr: "one_year_cdr",
    ml: "ml_development",
}


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
    if "options" not in extra:
        # one argument at fault: options is that argument alone
        assert refusal.options == ((option,) if option is not None else ())
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
    assert refusal_of(lambda: mk(T).to_polars("x")).method == "mack"
    assert refusal_of(lambda: cdr(tri(Z3), **FEW)).method == "one_year_cdr"
    assert refusal_of(lambda: cdr(T, n_draws=0)).method == "one_year_cdr"


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
    if s == 12:  # the one-year result refuses any other step before it looks at cells
        # the one-year result on a zero latest amount, under either zero rule
        zero_latest = table(rows_with((3, s), 0.0))
        expect(run(lambda: cdr(zero_latest, **FEW)), "variance_not_estimable", [(3, s)])
        expect(
            run(lambda: cdr(zero_latest, zero_cells="missing", **FEW)),
            "not_supported",
            [(3, s)],
        )
    else:
        refusal = run(lambda: cdr(table(base), **FEW, **kw))
        assert (refusal.reason, refusal.option) == ("not_supported", "dev_grain_months")


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
        (
            lambda: cl(years({1981: [1e200, 1.0, 1e-200], 1982: [1e200, 1.0], 1983: [1e200]})),
            "the link factors are too large or too small to multiply out",
        ),
        (
            lambda: cl(years({2020: [1.0, 1e308, 1.0], 2021: [1.0, 1e308], 2022: [1.0]})),
            "the factor from 12 to 24 months is not a finite number",
        ),
        (
            lambda: mk(tri(replace(fixture_rows("raa"), (1981, 120), 0.0)), zero_cells="observed"),
            "give a factor of 0.0 from 108 to 120 months",
        ),
        (
            lambda: cl(pa.table({"origin_period": [2001], "dev_lag": [1e30], "value": [1.0]})),
            "dev_lag 1e+30 is too large to be a number of months",
        ),
        # numpy scalars are shown as the numbers they hold, not as numpy's repr
        (
            lambda: bf(T, premium=PREM, expected_loss_ratio=np.float64(-0.1)),
            "expected_loss_ratio must be a finite number of 0 or more, got -0.1",
        ),
        (
            lambda: bf(T, premium=PREM, expected_loss_ratio=np.True_),
            "expected_loss_ratio must be a finite number of 0 or more, got True",
        ),
        (lambda: cl(T, dev_grain_months=np.int64(0)), "got 0"),
        (lambda: cl(T, average=np.str_("least_squares")), "got 'least_squares'"),
        (lambda: cl(T, drop_high=np.float64(1.5)), "got 1.5"),
        (
            lambda: cl(T, drop_high=2, drop_low=1, preserve=2, exhausted_exclusions="raise"),
            "drop_high=2 and drop_low would leave 0 of the 3 link ratio(s) from 12 to 24 "
            "months, fewer than preserve=2",
        ),
        (
            lambda: cl(T, drop_above=1.2, exhausted_exclusions="raise"),
            "drop_above=1.2 would leave 0 of the 3 link ratio(s) from 12 to 24 months, fewer "
            "than preserve=1; pass exhausted_exclusions='keep' to apply neither bound",
        ),
        # the trims still run once a bound is kept, and the count is of every ratio
        # the bound saw, whether a trim then left it out or not
        (
            lambda: cl(T, drop_above=1.49, drop_high=1, preserve=2, exhausted_exclusions="raise"),
            "drop_above=1.49 would leave 1 of the 3 link ratio(s) from 12 to 24 months, fewer "
            "than preserve=2",
        ),
        (
            lambda: cl(T, drop_below=1.51, drop_low=1, preserve=2, exhausted_exclusions="raise"),
            "drop_below=1.51 would leave 1 of the 3 link ratio(s) from 12 to 24 months, fewer "
            "than preserve=2",
        ),
        (
            lambda: bk(T, premium=PREM, expected_loss_ratio=0.7, n_iters=0),
            "n_iters=0 would ignore the reported losses",
        ),
        (
            lambda: cc(T, premium=PREM, n_iters=10**9),
            "n_iters must be at most 10000, got 1000000000",
        ),
        (lambda: cc(T, premium=PREM, trend=-1.0), "trend is an annual rate above -1"),
        (
            lambda: cl(QT, dev_grain_months=3, exclude_valuations=[2001]),
            "exclude_valuations 2001 is a year, 12 months long, but the development periods "
            "are 3 months long",
        ),
        (
            lambda: cl(T, exclude_valuations=["2005"]),
            "exclude_valuations names 2005-12-31, but no link ratio develops into that date; "
            "the link ratios develop into 2002-12-31 to 2004-12-31, every 12 months",
        ),
        (
            lambda: cl(T, exclude_valuations=[2002, "2002-12-31"]),
            "exclude_valuations names 2002-12-31 twice, as 2002 and '2002-12-31'",
        ),
        (
            lambda: bf(T, premium={**PREM, 2003: np.float64(-5.0)}, expected_loss_ratio=0.7),
            "it is -5.0 for 2003",
        ),
        (lambda: cl(T, exclude=[(2001, np.str_("12"))]), "names development age '12', which"),
        # the caller's text is shown as written, never read as a placeholder
        (
            lambda: cl({"origin_period": [2001], "dev_lag": [12], "{given}": [1.0]}),
            "it has ['origin_period', 'dev_lag', '{given}']",
        ),
        (lambda: cl(T, exclude=[(2001, "{cells}")]), "names development age '{cells}', which"),
        # the one-year result says what it needs, in the front door's words
        (
            lambda: cdr(tri(replace(BASE, (2004, 12), 0.0)), **FEW),
            "latest cumulative to be positive, and it is zero for (2004, 12 months)",
        ),
        (
            lambda: cdr(tri(replace(BASE, (2001, 12), 0.0)), zero_cells="missing", **FEW),
            "takes zero_cells='missing' only when no cumulative is zero, and (2001, 12 months)",
        ),
        (
            lambda: cdr(
                tri([("2001Q1", 3, 100.0), ("2001Q1", 6, 150.0), ("2001Q2", 3, 110.0)]),
                dev_grain_months=3,
                **FEW,
            ),
            "needs a 12-month development step, and this triangle's is 3 months",
        ),
        (lambda: cdr(T, quantiles=[50, 99.5]), "(divide a percentile by 100), got (50, 99.5)"),
        (lambda: cdr(T, n_draws=np.int64(0)), "n_draws must be a whole number of 1 or more, got 0"),
        (lambda: cdr(T, seed=-1), "seed must be None or a whole number of 0 or more, got -1"),
        (
            lambda: cdr(T, n_draws=2**63),
            "n_draws times the number of origins must be at most 100,000,000",
        ),
        # the Mack checks the two share name the function that was called
        (
            lambda: cdr(tri([(2001, 12, 100.0)]), **FEW),
            "one_year_cdr needs at least two development ages; this triangle has one",
        ),
        (
            lambda: cdr(years({2001: [100, 150], 2002: [110]}), **FEW),
            "one_year_cdr needs at least one development age with two or more link ratios",
        ),
        # amounts near the smallest double are too small, as methods.mack says, not
        # too large: the one-year formula's NaN must not be read as an overflow
        (
            lambda: cdr(
                years(
                    {
                        1981: [1e-300, 1.5e-300, 1.6e-300],
                        1982: [0.9e-300, 1.4e-300],
                        1983: [1.2e-300],
                    }
                ),
                **FEW,
            ),
            "the amounts are too small for Mack's standard errors",
        ),
    ],
)
def test_reworded_messages_say_what_is_wrong(call, phrase):
    text = str(refusal_of(call))
    assert phrase in text, text
    assert "np." not in text, text


def test_a_fraction_is_a_number():
    """A ``fractions.Fraction`` is a real number: it gives the answer its float gives."""
    from fractions import Fraction

    with_fraction = bf(T, premium=PREM, expected_loss_ratio=Fraction(7, 10))
    assert with_fraction.totals.equals(bf(T, premium=PREM, expected_loss_ratio=0.7).totals)
    half = cc(T, premium=PREM, decay=Fraction(1, 2))
    assert half.totals.equals(cc(T, premium=PREM, decay=0.5).totals)
    assert refusal_of(lambda: cc(T, premium=PREM, decay=Fraction(3, 2))).option == "decay"


#: Amounts so large or so small that sums, squares or products leave the doubles.
#: Each is refused with the reason given, never answered with a wrong number, and
#: never with a RuntimeWarning on the way.
_EXTREMES = [
    *[
        (
            f"mack_rule_{name}_1e160",
            lambda name=name: mk(tri(fixture_rows(name, 1e160)), sigma_rule="mack"),
            "result_not_finite",
        )
        for name in ("raa", "genins", "ukmotor", "abc", "mw2014")
    ],
    (
        "mack_log_linear_mw2014_1e303",
        lambda: mk(tri(fixture_rows("mw2014", 1e303)), zero_cells="observed"),
        "result_not_finite",
    ),
    ("mack_raa_1e300", lambda: mk(tri(fixture_rows("raa", 1e300))), "result_not_finite"),
    # every number per origin is finite, but the cdf at 12 months is 1e-480, which
    # is 0, so its pct_reported would be infinite
    (
        "mack_pattern_underflow",
        lambda: mk(
            years(
                {
                    1981: [1e200, 1e40, 1e-120, 1e-280],
                    1982: [1e200, 1e40, 1e-120],
                    1983: [1e200, 1e40],
                    1984: [1e200],
                }
            )
        ),
        "result_not_finite",
    ),
    # the squares underflow, so every standard error would read as 0
    ("mack_raa_1e-310", lambda: mk(tri(fixture_rows("raa", 1e-310))), "result_not_finite"),
    (
        "tiny_factors_chain_ladder",
        lambda: cl(years({1981: [1e200, 1.0, 1e-200], 1982: [1e200, 1.0], 1983: [1e200]})),
        "result_not_finite",
    ),
    (
        "tiny_factors_bornhuetter_ferguson",
        lambda: bf(
            years({1981: [1e200, 1.0, 1e-200], 1982: [1e200, 1.0], 1983: [1e200]}),
            premium={1981: 1.0, 1982: 1.0, 1983: 1.0},
            expected_loss_ratio=0.5,
        ),
        "result_not_finite",
    ),
    (
        "link_ratio_overflow_chain_ladder",
        lambda: cl(years({1981: [1e-300, 1e10, 1e10], 1982: [1e-300, 1e10], 1983: [0.0]})),
        "result_not_finite",
    ),
    (
        "premium_1e308_elr_10",
        lambda: bf(
            tri(fixture_rows("raa")),
            premium={y: 1e308 for y in range(1981, 1991)},
            expected_loss_ratio=10.0,
        ),
        "result_not_finite",
    ),
    (
        "premium_1e308_elr_1e10",
        lambda: bf(
            tri(fixture_rows("raa")),
            premium={y: 1e308 for y in range(1981, 1991)},
            expected_loss_ratio=1e10,
        ),
        "result_not_finite",
    ),
    (
        "subnormal_premium_cape_cod",
        lambda: cc(tri(fixture_rows("raa")), premium={y: 1e-320 for y in range(1981, 1991)}),
        "result_not_finite",
    ),
    # every ultimate is past the largest double, though every cell is finite
    *[
        (
            f"ml_raa_1e303_{estimator}",
            fitted(
                lambda estimator=estimator: ml(
                    tri(fixture_rows("raa", 1e303)), estimator=estimator, n_estimators=5
                )
            ),
            "result_not_finite",
        )
        for estimator in ("random_forest", "gradient_boosting")
    ],
    # boosting's own squared residuals overflow, so every prediction is NaN:
    # not a fitted cumulative of zero or less
    (
        "ml_genins_1e301_gradient_boosting",
        fitted(lambda: ml(tri(fixture_rows("genins", 1e301)), n_estimators=5, **GB)),
        "result_not_finite",
    ),
    # the normal deviance is a sum of squared amounts, past the largest double
    (
        "glm_deviance_overflow_genins_1e200",
        lambda: tg(tri(fixture_rows("genins", 1e200)), power=0),
        "result_not_finite",
    ),
    *[
        (
            f"cdr_{name}_{scale:g}",
            lambda name=name, scale=scale: cdr(tri(fixture_rows(name, scale)), **FEW),
            "result_not_finite",
        )
        for name in ("raa", "mw2014")
        for scale in (1e160, 1e300, 1e-310)
    ],
    # only the check that Mack's squares did not fall to 0 refuses these: without
    # it raa at 1e-166 gives a run-off standard error 17% too high, and finite
    *[
        (
            f"cdr_raa_{scale:g}",
            lambda scale=scale: cdr(tri(fixture_rows("raa", scale)), **FEW),
            "result_not_finite",
        )
        for scale in (1e-165, 1e-166)
    ],
]


@pytest.mark.parametrize(
    ("call", "reason"), [e[1:] for e in _EXTREMES], ids=[e[0] for e in _EXTREMES]
)
def test_amounts_too_large_or_too_small_are_refused_by_name(call, reason):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        refusal = refusal_of(call)
    assert refusal.reason == reason, str(refusal)
    json.dumps(refusal.to_dict(), allow_nan=False)


def test_mack_rule_fills_a_huge_sigma_without_overflow():
    """Mack's rule is min(last**2 / prev, last, prev). Where last**2 passes the
    largest double the ratio is still a number when prev is larger than last."""
    from ibnr.kernels.mack import _tail_sigma2

    assert _tail_sigma2(np.array([1e250, 1e200, np.nan]), 2, rule="mack") == pytest.approx(1e150)
    assert _tail_sigma2(np.array([1e200, 1e250, np.nan]), 2, rule="mack") == 1e200
    # the ordinary case keeps its exact value
    assert _tail_sigma2(np.array([4.0, 2.0, np.nan]), 2, rule="mack") == min(2.0**2 / 4.0, 2.0)


# -- 4. nothing else escapes ---------------------------------------------------------

_FUZZ = json.loads(
    (Path(__file__).parent / "data" / "refusal_triangles.json").read_text(encoding="utf-8")
)

_OPTION_POOL = {
    "average": ["volume", "simple", "regression", "median", "x", 1, None, True],
    "history_periods": [None, 1, 3, 0, -1, 2.5, True, "3", 10**12],
    "drop_high": [False, True, 1, 2, 3, -1, 1.0, "yes", None],
    "drop_low": [False, True, 0, 2, None],
    "preserve": [1, 1, 2, 3, 0, True, None],
    "drop_above": [None, 1.05, 1.5, 3.0, 1.0, math.nan, "x"],
    "drop_below": [None, 0.99, 1.0, 1.2, math.inf],
    "exclude_valuations": [(), [1990], ["1995-12-31"], [2030], "1990", [1990, 1990], [2.5]],
    "trim_ties": ["volume", "origin", "x"],
    "n_iters": [1, 2, 5, 0, -1, 2.5, True],
    "trend": [0.0, 0.05, -0.03, -1.0, math.inf, "0.05", 1e10],
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
        "preserve",
        "drop_above",
        "drop_below",
        "exclude_valuations",
        "trim_ties",
        "unsupported_factor",
        "exhausted_exclusions",
        "zero_cells",
        "dev_grain_months",
    ),
    "mack": (
        "sigma_rule",
        "zero_cells",
        "dev_grain_months",
        "average",
        "history_periods",
        "drop_high",
        "drop_low",
        "preserve",
        "drop_above",
        "drop_below",
        "exclude_valuations",
        "trim_ties",
        "exhausted_exclusions",
    ),
}
_PER_METHOD["bornhuetter_ferguson"] = (*_PER_METHOD["chain_ladder"], "expected_loss_ratio")
_PER_METHOD["benktander"] = (*_PER_METHOD["bornhuetter_ferguson"], "n_iters")
_PER_METHOD["cape_cod"] = (*_PER_METHOD["chain_ladder"], "decay", "trend", "n_iters")
#: Drawn at half the rate of the others, so that with thirteen options a method
#: takes, enough cases still carry no bad value and are answered.
_DRAWN_LESS_OFTEN = {
    "preserve",
    "drop_above",
    "drop_below",
    "exclude_valuations",
    "trim_ties",
    "n_iters",
    "trend",
}


def _fuzz_case(rng, rows):
    """One to three random edits; the last kind (scaling a cell) usually leaves a
    triangle the methods answer, so the no-NaN half of the check is exercised too."""
    rows = [list(r) for r in rows]
    for _ in range(int(rng.integers(1, 4))):
        edit = int(rng.integers(0, 11))
        if not rows:
            break
        i = int(rng.integers(0, len(rows)))
        if edit == 10:
            # the whole triangle near the edges of the doubles
            scale = [1e-300, 1e-200, 1e200, 1e300][int(rng.integers(0, 4))]
            for row in rows:
                row[2] = row[2] * scale
        elif edit >= 7:
            rows[i][2] = rows[i][2] * float(rng.uniform(0.5, 2.0))
        elif edit == 0:
            rows.pop(i)
        elif edit == 1:
            rows.append(list(rows[i]))
        elif edit == 2:
            rows[i][2] = -rows[i][2]
        elif edit == 3:
            rows[i][2] = [math.nan, math.inf, -math.inf, 0.0, 1e305, 1e-300, 5e-324][
                int(rng.integers(0, 7))
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
    """2,500 cases from five public triangles with one to three random edits and
    random options, valid and not. An edit may scale the whole triangle to near
    the smallest or the largest double, and a premium may be 1e-320 or 1e308.
    Every call returns a result whose numbers are all finite (a missing one is
    a null) or raises exactly Refusal. A RuntimeWarning
    is an error here, so a silent NaN path (an infinite amount subtracted from
    itself, say) fails the test."""
    rng = np.random.default_rng(20260924)
    names = sorted(_FUZZ)
    outcomes: dict[str, int] = {}
    for _ in range(2500):
        rows = _FUZZ[names[int(rng.integers(0, len(names)))]]
        edited = _fuzz_case(rng, rows)
        method = ("chain_ladder", "bornhuetter_ferguson", "benktander", "cape_cod", "mack")[
            int(rng.integers(0, 5))
        ]
        options = {}
        for name in _PER_METHOD[method]:
            if rng.random() < (0.12 if name in _DRAWN_LESS_OFTEN else 0.25):
                pool = _OPTION_POOL[name]
                options[name] = pool[int(rng.integers(0, len(pool)))]
        if method in ("bornhuetter_ferguson", "benktander", "cape_cod"):
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
                    1e-320,
                    1e308,
                ][int(rng.integers(0, 8))]
            options["premium"] = premium
        if method in ("bornhuetter_ferguson", "benktander"):
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


_GLM_POOL = {
    "power": [1.0, 1.0, 0.0, 1.5, 2.0, 3.0, 0.5, -1.0, math.nan, True, "1"],
    "link": ["log", "log", "identity", "logit", None],
    "origin": ["factor", "factor", "none", "x"],
    "calendar": ["none", "none", "trend", "x"],
    "projection": ["pattern", "pattern", "increments", "x"],
    "max_iter": [100, 100, 1, 3, 0, 2.5, True],
    "dev_grain_months": [12, 12, 12, 3, 0],
    "tail": [None, None, None, None, 1.05],
}


def test_a_seeded_fuzz_of_the_glm_meets_nothing_but_refusals():
    """The same edits as the fuzz above, through ``tweedie_glm`` with random
    options: every call gives finite numbers (a missing one a null) or exactly a
    Refusal, and a RuntimeWarning is an error."""
    rng = np.random.default_rng(20260925)
    names = sorted(_FUZZ)
    outcomes: dict[str, int] = {}
    for _ in range(1500):
        edited = _fuzz_case(rng, _FUZZ[names[int(rng.integers(0, len(names)))]])
        options = {}
        for name, pool in _GLM_POOL.items():
            if rng.random() < 0.2:
                options[name] = pool[int(rng.integers(0, len(pool)))]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                result = tg(_as_table(edited) if edited else pa.table({}), **options)
            except Refusal as refusal:
                assert type(refusal) is Refusal
                json.dumps(refusal.to_dict(), allow_nan=False)
                outcomes[refusal.reason] = outcomes.get(refusal.reason, 0) + 1
                continue
        _numbers_are_finite(result)
        outcomes["answered"] = outcomes.get("answered", 0) + 1
    assert outcomes["answered"] > 40, sorted(outcomes.items())
    assert len(outcomes) > 10, sorted(outcomes.items())


_CDR_POOL = {
    "sigma_rule": ["log_linear", "mack", "x", None],
    "zero_cells": ["observed", "observed", "missing", "x"],
    "dev_grain_months": [12, 12, 12, 3, 0],
    "n_draws": [30, 30, 1, 2, 0, -1, 2.5, True, 2**63],
    "seed": [None, 0, 7, -1, True, "3"],
    "process": ["gamma", "gamma", "lognormal", "normal", "x", None],
    "parameter_risk": [True, True, False, 1, None],
    "quantiles": [(0.5, 0.995), (), (0.0,), (1.5,), 0.5, (math.nan,), ("0.5",)],
}


def test_a_seeded_fuzz_of_the_one_year_cdr_meets_nothing_but_refusals():
    """The same edits as the fuzz above, through ``one_year_cdr`` with random
    options: every call gives finite numbers (a missing one a null) or exactly a
    Refusal, and a RuntimeWarning is an error."""
    rng = np.random.default_rng(20260926)
    names = sorted(_FUZZ)
    outcomes: dict[str, int] = {}
    for _ in range(1200):
        edited = _fuzz_case(rng, _FUZZ[names[int(rng.integers(0, len(names)))]])
        options = {"n_draws": 30, "seed": 1}
        for name, pool in _CDR_POOL.items():
            if rng.random() < 0.1:
                options[name] = pool[int(rng.integers(0, len(pool)))]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                result = cdr(_as_table(edited) if edited else pa.table({}), **options)
            except Refusal as refusal:
                assert type(refusal) is Refusal
                json.dumps(refusal.to_dict(), allow_nan=False)
                outcomes[refusal.reason] = outcomes.get(refusal.reason, 0) + 1
                continue
        for name in result.TABLES:
            for column in getattr(result, name).columns:
                if pa.types.is_floating(column.type):
                    present = column.drop_null().to_numpy()
                    assert np.isfinite(present).all(), (name, column)
        outcomes["answered"] = outcomes.get("answered", 0) + 1
    assert outcomes["answered"] > 100, sorted(outcomes.items())
    assert len(outcomes) > 8, sorted(outcomes.items())


_ML_POOL = {
    "estimator": ["random_forest", "gradient_boosting", "tweedie", None],
    "seed": [0, 42, None, -1, 2**32, True],
    "n_estimators": [5, 5, 0, 2.5],
    "max_depth": [None, 2, 0],
    "min_samples_leaf": [None, 2, 0],
    "learning_rate": [None, 0.5, 0.0, math.inf],
    "response": ["incremental", "cumulative", "x"],
    "origin": ["factor", "none", "x"],
    "calendar": ["none", "trend", "x"],
    "zero_cells": ["observed", "missing", "x"],
    "unsupported_factor": ["raise", "unity", "x"],
    "dev_grain_months": [12, 12, 3, 0],
    "tail": [None, None, None, 1.05],
}


def test_a_seeded_fuzz_of_ml_development_meets_nothing_but_refusals():
    """The same edits as the fuzz above, through ``ml_development`` with random
    options and 5 trees, so it runs in seconds: every call gives finite numbers
    (a missing one a null) or exactly a Refusal, and a RuntimeWarning is an
    error. Amounts near the largest double are in ``_EXTREMES``: none of the
    1,500 draws here reaches ``result_not_finite``."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(20260927)
    names = sorted(_FUZZ)
    outcomes: dict[str, int] = {}
    for _ in range(1500):
        edited = _fuzz_case(rng, _FUZZ[names[int(rng.integers(0, len(names)))]])
        options = {"estimator": ("random_forest", "gradient_boosting")[int(rng.integers(0, 2))]}
        options["n_estimators"] = 5
        for name, pool in _ML_POOL.items():
            if rng.random() < 0.1:
                options[name] = pool[int(rng.integers(0, len(pool)))]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                result = ml(_as_table(edited) if edited else pa.table({}), **options)
            except Refusal as refusal:
                assert type(refusal) is Refusal
                json.dumps(refusal.to_dict(), allow_nan=False)
                outcomes[refusal.reason] = outcomes.get(refusal.reason, 0) + 1
                continue
        _numbers_are_finite(result)
        outcomes["answered"] = outcomes.get("answered", 0) + 1
    assert outcomes["answered"] > 150, sorted(outcomes.items())
    assert len(outcomes) > 10, sorted(outcomes.items())


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
                ("tweedie_glm", {}),
                ("tweedie_glm", {"power": 0.0}),
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
    for method in ("chain_ladder", "mack", "tweedie_glm"):
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
    fitted = mack_kernels.fit_mack_many(t, loss_field="paid_loss", on_error="skip")
    assert fitted.reasons == {("0001", "holed"): "not_run_off"}
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


def test_a_non_positive_latest_amount_is_refused_by_the_variance_with_its_reason():
    """The Mack kernels fit the point estimate from a triangle whose youngest
    latest amount is negative or zero; the variance refuses it, and the code says
    which: a negative amount is the input's fault, a zero is Mack's limit."""
    from ibnr.kernels.cdr import one_year_cdr
    from ibnr.kernels.mack import fit_mack_grid

    negative = SMALL.copy()
    negative[3, 0] = -5.0
    fit = fit_mack_grid(_grid_of(negative))
    for call in (fit.msep_runoff, lambda: one_year_cdr(fit)):
        refusal = refusal_of(call)
        assert (refusal.reason, refusal.option) == ("negative_cumulative", "cells")
        assert [(c.origin_period, c.dev_lag, c.value) for c in refusal.cells] == [
            (dt.date(2013, 1, 1), 12, -5.0)
        ]
    zero = SMALL.copy()
    zero[3, 0] = 0.0
    refusal = refusal_of(fit_mack_grid(_grid_of(zero)).msep_runoff)
    assert (refusal.reason, refusal.option) == ("variance_not_estimable", "zero_cells")


def test_every_reason_is_documented_for_a_caller():
    """The closed list is only useful if a caller can read what each code means:
    the ``ibnr.errors`` docstring says, and the chainladder page lists them."""
    page = (Path(__file__).parents[1] / "docs" / "coming-from-chainladder.md").read_text(
        encoding="utf-8"
    )
    for reason in REASONS:
        assert f"``{reason}``:" in errors.__doc__, reason
        assert f"`{reason}`" in page, reason
