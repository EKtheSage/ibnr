"""``methods.ml_development`` and ``kernels.ml_development``: tree models on a triangle's cells.

Checked here:

- the tie-out to chainladder-python 0.9.2's ``DevelopmentML`` followed by its
  ``Chainladder``, built as the Reserving app builds it (``tieout``): the same
  estimator, seed and design give the same ultimates, and the same fitted
  value in every cell ``DevelopmentML`` predicts, to 1e-12, on the five public
  sample triangles and raa relabelled as quarters, and the same ultimates to
  1e-10 on the clrd paid triangles with no zero cell (a fixed sample of 31
  here, all 367 under ``-m slow``);
- the Reserving app's own numbers on its example workbook's paid triangle;
- that every option reaches the fit and changes the answer where it should,
  pinned without chainladder too, so the ``ml`` CI leg (which has no
  chainladder) catches a design or row-order defect;
- zero cells: under ``zero_cells="missing"`` the training rows are
  chainladder's, and the ultimates differ from chainladder's where its
  defects are named in ``docs/coming-from-chainladder.md``;
- the result tables, the caller's labels, nulls never NaN, and the import.

Refusals are in ``tests/test_refusal.py`` with every other method's; the ones
that need a fit are repeated here with their messages.
"""

from __future__ import annotations

import csv
import datetime as dt
import random
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ibnr import methods
from ibnr.errors import Refusal
from ibnr.kernels.grid import grid_from_columns
from ibnr.kernels.ml_development import MLDevelopmentSpec, fit_ml_development_grid

# raa, the Mack (1993) triangle: accident years 1981 to 1990.
RAA = [
    [5012, 8269, 10907, 11805, 13539, 16181, 18009, 18608, 18662, 18834],
    [106, 4285, 5396, 10666, 13782, 15599, 15496, 16169, 16704],
    [3410, 8992, 13873, 16141, 18735, 22214, 22863, 23466],
    [5655, 11555, 15766, 21266, 23425, 26083, 27067],
    [1092, 9565, 15836, 22169, 25955, 26180],
    [1513, 6445, 11702, 12935, 15852],
    [557, 4020, 10946, 12314],
    [1351, 6947, 13112],
    [3133, 5395],
    [2063],
]

# GenIns (Taylor and Ashe 1983), accident years 2001 to 2010.
GENINS = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
    [396132, 1333217, 2180715, 2985752, 3691712],
    [440832, 1288463, 2419861, 3483130],
    [359480, 1421128, 2864498],
    [376686, 1363294],
    [344014],
]

#: The Reserving app's example workbook, sheet Data, range PaidTriangle: New
#: Jersey Manufacturers Grp (7080), workers' compensation paid, $000s, CAS
#: Schedule P as of 1997-12-31. ``data/README.md`` says where it came from.
WORKBOOK = Path(__file__).parent / "data" / "ml_workbook_paid.csv"

#: The app's ``POST /ml`` total IBNR on that triangle through chainladder 0.9.2
#: and scikit-learn 1.9.0: forest seed 42 (the workbook's A24 and the notebook),
#: forest seed 0, boosting at any seed (the workbook's A47, the web), and the
#: forest with ``fit_incrementals`` false. The chain ladder is the ``/reserve``
#: answer, for scale.
APP_TOTALS = {
    ("random_forest", 42, "incremental"): 1_002_288.05,
    ("random_forest", 0, "incremental"): 1_047_110.10,
    ("gradient_boosting", 42, "incremental"): 657_942.08,
    ("gradient_boosting", 0, "incremental"): 657_942.08,
    ("gradient_boosting", 7, "incremental"): 657_942.08,
    ("random_forest", 42, "cumulative"): -250_298.82,
}
CHAIN_LADDER_TOTAL = 373_346.30


@pytest.fixture(autouse=True)
def _quiet_joblib(monkeypatch):
    # On Windows 11 joblib prints a harmless traceback while counting physical
    # cores (wmic is gone); a set count skips the count.
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")


@pytest.fixture
def sklearn():
    return pytest.importorskip("sklearn")


def cells_of(rows, first: int = 2001, *, step: int = 12, label=None) -> pa.Table:
    """The cells of a staircase, integer years by default."""
    origin, lag, value = [], [], []
    for i, row in enumerate(rows):
        for j, amount in enumerate(row):
            if amount is None:
                continue
            origin.append(first + i if label is None else label(i))
            lag.append(step * (j + 1))
            value.append(float(amount))
    return pa.table({"origin_period": origin, "dev_lag": lag, "value": value})


def workbook() -> pa.Table:
    with WORKBOOK.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return pa.table(
        {
            "origin_period": [int(r["origin_period"]) for r in rows],
            "dev_lag": [int(r["dev_lag"]) for r in rows],
            "value": [float(r["value"]) for r in rows],
        }
    )


def column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=float)


def total_ibnr(result) -> float:
    return result.totals["ibnr"][0].as_py()


def refusal_of(thunk) -> Refusal:
    with pytest.raises(Refusal) as caught:
        thunk()
    assert type(caught.value) is Refusal
    return caught.value


def ml(cells, estimator="random_forest", **options):
    return methods.ml_development(cells, estimator=estimator, **options)


# -- 1. chainladder-python's DevelopmentML ---------------------------------------------


def _chainladder_fit(triangle, estimator: str, seed: int, *, formula: str, incremental: bool):
    """DevelopmentML then Chainladder, as the app's ``/ml`` route builds them."""
    cl = pytest.importorskip("chainladder")
    from chainladder.utils import PatsyFormula
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.pipeline import Pipeline

    if estimator == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=seed)
    else:
        model = GradientBoostingRegressor(random_state=seed)
    pipe = Pipeline([("design_matrix", PatsyFormula(formula)), ("model", model)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dev = cl.DevelopmentML(pipe, y_ml=list(triangle.columns)[0], fit_incrementals=incremental)
        fitted = cl.Chainladder().fit(dev.fit_transform(triangle))
    return np.asarray(fitted.ultimate_.values[0, 0, :, -1], dtype=float), dev


def _grid_of(triangle) -> np.ndarray:
    return np.asarray(triangle.values[0, 0], dtype=float)


def _predicted(dev, origins: list, step: int) -> dict[tuple[int, int], float]:
    """``DevelopmentML.predicted_data_`` keyed by (origin index, age index)."""
    import pandas as pd

    frame = dev.predicted_data_
    position = {pd.Timestamp(o): i for i, o in enumerate(origins)}
    return {
        (position[pd.Timestamp(o)], int(d) // step - 1): float(v)
        for o, d, v in zip(frame["origin"], frame["development"], frame["values"], strict=True)
    }


def _require_same_cells(result, predicted: dict, n_d: int, *, incremental: bool) -> None:
    """Every cell DevelopmentML predicts has the same fitted value here."""
    name = "fitted_increment" if incremental else "fitted_cumulative"
    fitted = column(result.cells, name).reshape(-1, n_d)
    for (i, j), value in predicted.items():
        assert fitted[i, j] == pytest.approx(value, rel=1e-12, abs=1e-9), (i, j)


#: (estimator, seed, response, origin, calendar, patsy formula)
_CASES = [
    ("random_forest", 0, "incremental", "factor", "none", "C(development) + C(origin)"),
    ("random_forest", 42, "incremental", "factor", "none", "C(development) + C(origin)"),
    ("gradient_boosting", 42, "incremental", "factor", "none", "C(development) + C(origin)"),
    ("random_forest", 42, "cumulative", "factor", "none", "C(development) + C(origin)"),
    ("gradient_boosting", 42, "cumulative", "factor", "none", "C(development) + C(origin)"),
    ("random_forest", 42, "incremental", "none", "none", "C(development)"),
    ("gradient_boosting", 42, "incremental", "none", "trend", "C(development) + valuation"),
    (
        "random_forest",
        42,
        "incremental",
        "factor",
        "trend",
        "C(development) + C(origin) + valuation",
    ),
]


@pytest.mark.tieout
@pytest.mark.parametrize("sample", ["raa", "genins", "ukmotor", "abc", "mw2014"])
@pytest.mark.parametrize("case", _CASES, ids=["-".join(map(str, c[:5])) for c in _CASES])
def test_the_sample_triangles_match_chainladder(sklearn, sample, case):
    """Mutations: drop the column of ones, put the rows in development order,
    count the calendar as origin minus age, or project from ``latest_dev + 1``:
    each fails here, most at the first cell. Numbering the calendar column
    from 1 changes nothing and is not caught: a tree splits halfway between
    two values, so adding a constant to a column moves no split."""
    cl = pytest.importorskip("chainladder")
    estimator, seed, response, origin, calendar, formula = case
    triangle = cl.load_sample(sample)
    grid = _grid_of(triangle)
    rows = [list(row[~np.isnan(row)]) for row in grid]
    first = int(str(triangle.origin[0])[:4])
    expected, dev = _chainladder_fit(
        triangle, estimator, seed, formula=formula, incremental=response == "incremental"
    )
    result = ml(
        cells_of(rows, first),
        estimator,
        seed=seed,
        response=response,
        origin=origin,
        calendar=calendar,
    )
    np.testing.assert_allclose(column(result.origins, "ultimate"), expected, rtol=1e-12)
    predicted = _predicted(dev, list(triangle.origin.to_timestamp()), 12)
    _require_same_cells(result, predicted, grid.shape[1], incremental=response == "incremental")


def _quarters(i: int) -> str:
    return f"{1981 + i // 4}Q{i % 4 + 1}"


@pytest.mark.tieout
@pytest.mark.parametrize("estimator", ["random_forest", "gradient_boosting"])
@pytest.mark.parametrize(
    ("calendar", "formula"),
    [("none", "C(development) + C(origin)"), ("trend", "C(development) + C(origin) + valuation")],
)
def test_quarterly_origins_on_quarterly_ages_match_chainladder(
    sklearn, estimator, calendar, formula
):
    """raa relabelled as ten quarters from 1981Q1, at ``dev_grain_months=3``.
    Mutation: count the calendar column in years rather than development
    steps; the ``"trend"`` cases fail."""
    cl = pytest.importorskip("chainladder")
    import pandas as pd

    frame = []
    for i, row in enumerate(RAA):
        start = pd.Timestamp(1981, 1, 1) + pd.DateOffset(months=3 * i)
        for j, amount in enumerate(row):
            end = start + pd.DateOffset(months=3 * (j + 1)) - pd.Timedelta(days=1)
            frame.append((start, end, float(amount)))
    data = pd.DataFrame(frame, columns=["origin", "valuation", "values"])
    triangle = cl.Triangle(
        data, origin="origin", development="valuation", columns="values", cumulative=True
    )
    assert (triangle.origin_grain, triangle.development_grain) == ("Q", "Q")
    expected, dev = _chainladder_fit(triangle, estimator, 42, formula=formula, incremental=True)
    result = ml(
        cells_of(RAA, step=3, label=_quarters),
        estimator,
        seed=42,
        calendar=calendar,
        dev_grain_months=3,
    )
    np.testing.assert_allclose(column(result.origins, "ultimate"), expected, rtol=1e-12)
    predicted = _predicted(dev, list(triangle.origin.to_timestamp()), 3)
    _require_same_cells(result, predicted, len(RAA), incremental=True)


def test_annual_origins_on_quarterly_ages_are_refused():
    """chainladder's ``quarterly`` sample has annual origins and quarterly ages.
    Its DevelopmentML counts years and quarters as one unit when it looks for
    the latest diagonal, so every ultimate but one equals the latest amount
    (IBNR 0) and the newest origin's is NaN, where the chain ladder gives IBNR
    of 2 to 945 per origin. ibnr refuses the shape before any fit, as every
    method does."""
    cells = pa.table(
        {
            "origin_period": [1995, 1995, 1995, 1996],
            "dev_lag": [3, 6, 9, 3],
            "value": [100.0, 150.0, 170.0, 120.0],
        }
    )
    refusal = refusal_of(lambda: ml(cells, dev_grain_months=3))
    assert refusal.reason == "grain_mismatch"


def _clrd_grids():
    """Every clrd company and line's paid triangle from chainladder's raw
    ``clrd.csv``, so a zero stays a zero (its Triangle stores zeros as missing)."""
    cl = pytest.importorskip("chainladder")

    path = Path(cl.__file__).parent / "utils" / "data" / "clrd.csv"
    grids: dict[tuple[str, str], np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            year, lag = int(row["AccidentYear"]), int(row["DevelopmentLag"])
            if year + lag - 1 > 1997:
                continue
            grid = grids.setdefault((row["GRNAME"], row["LOB"]), np.full((10, 10), np.nan))
            grid[year - 1988, lag - 1] = float(row["CumPaidLoss"])
    return grids


def _cells_of_grid(grid: np.ndarray) -> pa.Table:
    where = np.argwhere(~np.isnan(grid))
    return pa.table(
        {
            "origin_period": (1988 + where[:, 0]).tolist(),
            "dev_lag": (12 * (where[:, 1] + 1)).tolist(),
            "value": grid[~np.isnan(grid)].tolist(),
        }
    )


def _chainladder_triangle(grid: np.ndarray):
    """The triangle as the app builds it: a DataFrame of origin, valuation and value."""
    cl = pytest.importorskip("chainladder")
    import pandas as pd

    rows = [
        (str(1988 + i), pd.Timestamp(1988 + i + j, 12, 31), grid[i, j])
        for i in range(grid.shape[0])
        for j in range(grid.shape[1])
        if not np.isnan(grid[i, j])
    ]
    frame = pd.DataFrame(rows, columns=["origin", "valuation", "values"])
    return cl.Triangle(
        frame, origin="origin", development="valuation", columns="values", cumulative=True
    )


def _no_zero_cell(grids) -> list[tuple[str, str]]:
    return [
        key
        for key, grid in sorted(grids.items())
        if np.nanmax(np.abs(grid)) > 0 and not (grid[~np.isnan(grid)] == 0).any()
    ]


def _clrd_tie_out(keys, grids, estimator: str) -> dict[str, int]:
    """Each triangle matches chainladder to 1e-10 relative, or is refused for a
    reason chainladder's own answer shows."""
    outcomes: dict[str, int] = {}
    for key in keys:
        grid = grids[key]
        expected, _ = _chainladder_fit(
            _chainladder_triangle(grid),
            estimator,
            42,
            formula="C(development) + C(origin)",
            incremental=True,
        )
        try:
            result = ml(_cells_of_grid(grid), estimator, seed=42)
        except Refusal as refusal:
            outcomes[refusal.reason] = outcomes.get(refusal.reason, 0) + 1
            if refusal.reason == "negative_cumulative":
                assert np.nanmin(grid) < 0, key
            else:
                # chainladder answered the same fit with a negative ultimate
                assert refusal.reason == "negative_projection", key
                assert (expected < 0).any(), key
            continue
        ultimate = column(result.origins, "ultimate")
        gap = np.max(np.abs(ultimate - expected) / np.maximum(np.abs(expected), 1.0))
        assert gap <= 1e-10, (key, gap)
        outcomes["matched"] = outcomes.get("matched", 0) + 1
    return outcomes


@pytest.mark.tieout
@pytest.mark.parametrize("estimator", ["random_forest", "gradient_boosting"])
def test_a_sample_of_clrd_paid_triangles_matches_chainladder(sklearn, estimator):
    """Every twelfth of the 367 paid triangles with no zero cell, 31 in all."""
    grids = _clrd_grids()
    keys = _no_zero_cell(grids)
    assert len(keys) == 367
    outcomes = _clrd_tie_out(keys[::12], grids, estimator)
    assert sum(outcomes.values()) == 31
    assert outcomes["matched"] >= 25, outcomes


@pytest.mark.slow
@pytest.mark.tieout
@pytest.mark.parametrize("estimator", ["random_forest", "gradient_boosting"])
def test_every_clrd_paid_triangle_with_no_zero_cell_matches_chainladder(sklearn, estimator):
    """All 367: the forest matches on 353 and boosting on 333; 14 have a
    negative cumulative and are refused, and boosting's other 20 give a
    negative ultimate, which chainladder answers and ibnr refuses. About 3
    minutes for the forest and 2 for boosting, with chainladder's fits."""
    grids = _clrd_grids()
    outcomes = _clrd_tie_out(_no_zero_cell(grids), grids, estimator)
    if estimator == "random_forest":
        assert outcomes == {"matched": 353, "negative_cumulative": 14}
    else:
        assert outcomes == {"matched": 333, "negative_cumulative": 14, "negative_projection": 20}


# -- 2. the Reserving app's numbers --------------------------------------------------


@pytest.mark.parametrize(("key", "expected"), APP_TOTALS.items(), ids=lambda v: str(v))
def test_the_reserving_app_workbook_totals(sklearn, key, expected):
    """What the app's ``/ml`` route returned through chainladder, to the cent.
    The chain ladder is 373,346.30: the forest is 2.7 times it and boosting
    1.8 times, which is the method (see the module docstring), not a defect."""
    estimator, seed, response = key
    result = ml(workbook(), estimator, seed=seed, response=response)
    assert total_ibnr(result) == pytest.approx(expected, abs=0.005)
    chain = total_ibnr(methods.chain_ladder(workbook()))
    assert chain == pytest.approx(CHAIN_LADDER_TOTAL, abs=0.005)


def test_the_seed_moves_the_forest_and_not_boosting(sklearn):
    """Mutation: pass a fixed ``random_state`` instead of ``seed``; the forest's
    two totals are then equal and this fails."""
    forest = [total_ibnr(ml(workbook(), seed=s)) for s in (0, 42)]
    assert forest[0] != forest[1]
    boosting = [total_ibnr(ml(workbook(), "gradient_boosting", seed=s)) for s in (0, 42)]
    assert boosting[0] == pytest.approx(boosting[1], rel=1e-12)
    assert [ml(workbook(), seed=s).totals["seed"][0].as_py() for s in (0, 4_294_967_295)] == [
        0,
        4_294_967_295,
    ]


def test_the_same_call_twice_gives_the_same_bytes(sklearn):
    first = ml(workbook(), seed=42)
    second = ml(workbook(), seed=42)
    for name in ("origins", "development", "cells", "totals"):
        assert getattr(first, name).equals(getattr(second, name)), name


# -- 3. every option reaches the fit -------------------------------------------------


@pytest.mark.parametrize(
    ("estimator", "option", "value"),
    [
        ("random_forest", "n_estimators", 50),
        ("random_forest", "max_depth", 3),
        ("random_forest", "min_samples_leaf", 2),
        ("gradient_boosting", "n_estimators", 50),
        ("gradient_boosting", "max_depth", 2),
        ("gradient_boosting", "learning_rate", 0.05),
        ("random_forest", "response", "cumulative"),
        ("gradient_boosting", "response", "cumulative"),
        ("random_forest", "origin", "none"),
        ("gradient_boosting", "origin", "none"),
        ("random_forest", "calendar", "trend"),
        ("gradient_boosting", "calendar", "trend"),
    ],
)
def test_every_option_changes_the_answer(sklearn, estimator, option, value):
    """A setting that never reaches scikit-learn leaves the answer as it was;
    only a changed answer shows it was delivered. Each is echoed in totals."""
    base = ml(workbook(), estimator, seed=42)
    changed = ml(workbook(), estimator, seed=42, **{option: value})
    assert total_ibnr(changed) != pytest.approx(total_ibnr(base), rel=1e-6)
    echoed = {"origin": "origin_term", "calendar": "calendar_term"}.get(option, option)
    assert changed.totals[echoed][0].as_py() == value


def test_the_settings_left_as_none_are_scikit_learns_defaults(sklearn):
    """``None`` is scikit-learn's own default, and the echo says which."""
    forest = ml(workbook(), seed=42)
    same = ml(workbook(), seed=42, n_estimators=100, min_samples_leaf=1)
    assert total_ibnr(forest) == total_ibnr(same)
    assert forest.totals.select(
        ["n_estimators", "max_depth", "min_samples_leaf", "learning_rate"]
    ).to_pylist() == [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "learning_rate": None}
    ]
    boosting = ml(workbook(), "gradient_boosting", seed=42)
    explicit = ml(workbook(), "gradient_boosting", seed=42, max_depth=3, learning_rate=0.1)
    assert total_ibnr(boosting) == total_ibnr(explicit)
    assert boosting.totals.select(
        ["n_estimators", "max_depth", "min_samples_leaf", "learning_rate"]
    ).to_pylist() == [
        {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": None, "learning_rate": 0.1}
    ]


#: Totals pinned without chainladder, so the ``ml`` CI leg, which has no
#: chainladder, still catches a design or row-order defect. The forest's
#: predictions were byte-identical under scikit-learn 1.4.2, 1.6.1 and 1.9.0.
#: With the column of ones left out, GenIns's forest total is 17,198,277; with
#: the rows in development order, raa's is 73,211.83 and GenIns's 16,966,770.18.
_PINNED = [
    (RAA, 1981, "random_forest", {}, 75_269.77),
    (GENINS, 2001, "random_forest", {}, 17_326_133.79),
    (GENINS, 2001, "random_forest", {"origin": "none", "calendar": "trend"}, 19_544_811.0),
    (GENINS, 2001, "random_forest", {"calendar": "trend"}, 19_647_001.0),
    (GENINS, 2001, "random_forest", {"origin": "none"}, 18_505_131.0),
]


@pytest.mark.parametrize(("rows", "first", "estimator", "options", "expected"), _PINNED)
def test_pinned_totals_catch_a_design_or_row_order_defect(
    sklearn, rows, first, estimator, options, expected
):
    """Mutations: leave out the column of ones, order the rows by development
    age, or add the origin columns when ``origin="none"``; each moves one of
    these totals by more than 0.5."""
    result = ml(cells_of(rows, first), estimator, seed=42, **options)
    assert total_ibnr(result) == pytest.approx(expected, abs=0.5)


def test_a_pinned_quarterly_calendar_total(sklearn):
    """raa relabelled as quarters, with ``calendar="trend"``: the calendar
    column counts development steps (quarters here). Mutation: count it in
    years; the total moves from 79,155.19 to 41,727.86."""
    cells = cells_of(RAA, step=3, label=_quarters)
    result = ml(cells, seed=42, calendar="trend", dev_grain_months=3)
    assert total_ibnr(result) == pytest.approx(79_155.19, abs=0.005)


def test_the_design_columns_follow_the_options(sklearn):
    grid = _grid(RAA)
    names = {
        (origin, calendar): fit_ml_development_grid(
            grid, MLDevelopmentSpec("gradient_boosting", origin=origin, calendar=calendar)
        ).feature_names
        for origin in ("factor", "none")
        for calendar in ("none", "trend")
    }
    ages = tuple(f"development:{12 * j}" for j in range(2, 11))
    years = tuple(f"origin:{1981 + i}-01-01" for i in range(1, 10))
    assert names[("factor", "none")] == ("intercept", *ages, *years)
    assert names[("none", "none")] == ("intercept", *ages)
    assert names[("none", "trend")] == ("intercept", *ages, "calendar")
    assert names[("factor", "trend")] == ("intercept", *ages, *years, "calendar")


def _grid(rows, first: int = 1981, step: int = 12) -> dict:
    origin, lag, value = [], [], []
    for i, row in enumerate(rows):
        for j, amount in enumerate(row):
            origin.append(np.datetime64(dt.date(first + i, 1, 1), "D"))
            lag.append(step * (j + 1))
            value.append(float(amount))
    return grid_from_columns(
        np.array(origin),
        np.array(lag),
        np.array(value),
        dev_grain_months=step,
        measure="cumulative",
    )


def test_the_projection_is_each_origins_own_fitted_pattern(sklearn):
    """``ultimate = latest * fitted_ultimate / fitted_latest``, from each
    origin's actual latest age. Mutation: take the fitted cumulative one age
    later (``latest_dev + 1``); the ratio then no longer matches the cells."""
    result = ml(workbook(), seed=42)
    origins = result.origins
    fitted = column(result.cells, "fitted_cumulative").reshape(10, 10)
    latest_dev = np.asarray(origins["latest_dev_lag"].to_pylist()) // 12 - 1
    at_latest = fitted[np.arange(10), latest_dev]
    np.testing.assert_array_equal(column(origins, "fitted_latest"), at_latest)
    np.testing.assert_array_equal(column(origins, "fitted_ultimate"), fitted[:, -1])
    cdf = np.where(latest_dev == 9, 1.0, fitted[:, -1] / at_latest)
    np.testing.assert_array_equal(column(origins, "cdf"), cdf)
    np.testing.assert_array_equal(column(origins, "ultimate"), column(origins, "latest") * cdf)
    assert origins["cdf"][0].as_py() == 1.0


def test_model_ibnr_is_the_sum_of_the_fitted_future_increments(sklearn):
    """Mutation: report ``ibnr`` as ``model_ibnr``; this fails, since for a tree
    model the fitted cumulative at the latest age is never the actual one."""
    result = ml(workbook(), seed=42)
    fitted = column(result.cells, "fitted_increment").reshape(10, 10)
    latest_dev = np.asarray(result.origins["latest_dev_lag"].to_pylist()) // 12 - 1
    future = np.arange(10)[None, :] > latest_dev[:, None]
    expected = np.where(future, fitted, 0.0).sum(axis=1)
    np.testing.assert_allclose(column(result.origins, "model_ibnr"), expected, rtol=1e-12)
    assert not np.allclose(column(result.origins, "model_ibnr"), column(result.origins, "ibnr"))
    assert result.totals["model_ibnr"][0].as_py() == pytest.approx(expected.sum(), rel=1e-12)


def test_the_cumulative_response_models_the_cumulative(sklearn):
    """Under ``response="cumulative"`` the fitted cumulative is the prediction
    and the fitted increment its difference; the forest's cumulatives fall with
    age on the workbook triangle, so its total IBNR is below zero."""
    result = ml(workbook(), seed=42, response="cumulative")
    fitted = column(result.cells, "fitted_cumulative").reshape(10, 10)
    increments = column(result.cells, "fitted_increment").reshape(10, 10)
    np.testing.assert_allclose(np.cumsum(increments, axis=1), fitted, rtol=1e-12)
    assert total_ibnr(result) < 0
    assert result.totals["response"][0].as_py() == "cumulative"


def test_origins_left_out_share_one_pattern(sklearn):
    """With ``origin="none"`` every origin gets the same predictions, so the
    development table carries the one pattern; with origin indicators it is
    null and each origin's own factors are in cells."""
    shared = ml(workbook(), "gradient_boosting", origin="none")
    factors = column(shared.cells, "factor").reshape(10, 10)[:, :-1]
    np.testing.assert_allclose(factors, np.broadcast_to(factors[0], factors.shape), rtol=1e-12)
    np.testing.assert_allclose(column(shared.development, "factor")[:-1], factors[0], rtol=1e-12)
    cdf = column(shared.development, "cdf")
    np.testing.assert_allclose(cdf[:-1], np.cumprod(factors[0][::-1])[::-1], rtol=1e-12)
    assert cdf[-1] == 1.0
    np.testing.assert_array_equal(column(shared.development, "pct_reported"), 1.0 / cdf)
    own = ml(workbook(), "gradient_boosting")
    for name in ("factor", "cdf", "pct_reported"):
        assert own.development[name].null_count == 10, name
    assert own.development["n_trained"].to_pylist() == list(range(10, 0, -1))


# -- 4. zero cells ---------------------------------------------------------------------


def test_zero_cells_decides_which_cells_are_trained(sklearn):
    """raa with 1986 at 12 months set to 0: 55 training rows, and 54 under
    ``"missing"``. Mutation: ignore ``zero_cells``; the counts are equal."""
    rows = [list(row) for row in RAA]
    rows[5][0] = 0
    kept = ml(cells_of(rows, 1981), seed=42)
    left = ml(cells_of(rows, 1981), seed=42, zero_cells="missing")
    assert kept.totals["n_training_rows"][0].as_py() == 55
    assert left.totals["n_training_rows"][0].as_py() == 54
    assert left.totals["zero_cells"][0].as_py() == "missing"
    trained = left.cells.filter(pc.equal(left.cells["origin"], 1986))["trained"].to_pylist()
    assert trained == [False, True, True, True, True, False, False, False, False, False]
    assert total_ibnr(kept) != pytest.approx(total_ibnr(left), rel=1e-6)
    # the zero is observed though not trained on: its increment is 0, not null.
    # Mutation: null the increment wherever a cell is not trained; this fails.
    zero = left.cells.filter(pc.equal(left.cells["origin"], 1986))
    assert zero["observed"][0].as_py() is True
    assert zero["increment"][0].as_py() == 0.0
    assert zero["increment"].to_pylist()[5:] == [None] * 5


def test_a_zero_cells_fitted_increment_counts_in_its_fitted_cumulative(sklearn):
    """raa with 1983 at 24 months set to 0 (3,410, then 0, then 13,873). Under
    ``"missing"`` that cell is not trained on, but its fitted increment is
    still part of 1983's fitted cumulative; chainladder leaves it out, which
    changes the origin's factors. Mutation: build the fitted cumulative from the
    training rows and the future cells only; this fails."""
    rows = [list(row) for row in RAA]
    rows[2][1] = 0
    result = ml(cells_of(rows, 1981), seed=42, zero_cells="missing")
    row = result.cells.filter(pc.equal(result.cells["origin"], 1983))
    assert row["trained"].to_pylist()[:3] == [True, False, True]
    increments = column(row, "fitted_increment")
    assert increments[1] != 0
    np.testing.assert_allclose(column(row, "fitted_cumulative"), np.cumsum(increments))


def test_a_zero_latest_gives_an_ultimate_of_zero(sklearn):
    """raa with 1990's only cell set to 0: its ultimate is 0 under both rules,
    where chainladder's is NaN (and the app reports that origin's IBNR as 0 and
    its ultimate as null). Under ``"missing"`` the origin has no training row,
    so with origin indicators the model cannot place it and its fitted numbers
    are null."""
    rows = [list(row) for row in RAA]
    rows[9][0] = 0
    for rule in ("observed", "missing"):
        result = ml(cells_of(rows, 1981), seed=42, zero_cells=rule)
        assert result.origins["ultimate"][9].as_py() == 0.0, rule
        assert result.origins["ibnr"][9].as_py() == 0.0, rule
    missing = ml(cells_of(rows, 1981), seed=42, zero_cells="missing")
    for name in ("fitted_latest", "fitted_ultimate", "cdf", "model_ibnr"):
        assert missing.origins[name][9].as_py() is None, name
    last = missing.cells.filter(pc.equal(missing.cells["origin"], 1990))
    assert last["fitted_increment"].null_count == 10
    shared = ml(cells_of(rows, 1981), "gradient_boosting", zero_cells="missing", origin="none")
    assert shared.origins["fitted_latest"][9].as_py() is not None


def test_an_age_with_only_zeros_is_refused_or_developed_by_one(sklearn):
    """raa with 1981 all zero: under ``"missing"`` the 120-month age has no
    training row. ``unsupported_factor="raise"`` refuses it; ``"unity"``
    develops every origin by 1 into it and marks the age."""
    rows = [list(row) for row in RAA]
    rows[0] = [0] * 10
    cells = cells_of(rows, 1981)
    refusal = refusal_of(lambda: ml(cells, seed=42, zero_cells="missing"))
    assert refusal.reason == "no_link_ratio"
    assert refusal.option == "unsupported_factor"
    assert list(refusal.links) == [(108, 120)]
    assert [(c.origin, c.dev_lag, c.value) for c in refusal.cells] == [(1981, 120, 0.0)]
    assert "every cell at dev_lag 120 is zero" in str(refusal)
    result = ml(cells, seed=42, zero_cells="missing", unsupported_factor="unity")
    fallback = result.development["unity_fallback"].to_pylist()
    assert fallback == [False] * 8 + [True, None]
    factors = column(result.cells, "factor").reshape(10, 10)
    np.testing.assert_array_equal(factors[1:, 8], 1.0)
    assert result.development["n_trained"][9].as_py() == 0
    # with every cell trained, the same triangle needs no fallback
    kept = ml(cells, seed=42)
    assert kept.development["unity_fallback"].to_pylist() == [False] * 9 + [None]


def test_the_cumulative_response_develops_by_one_into_an_age_with_only_zeros(sklearn):
    """The same triangle as above, with ``response="cumulative"``: every placed
    origin's fitted cumulative at 120 months is its fitted cumulative at 108,
    so its factor there is 1. Mutation: set the fitted cumulative at the
    unity age to 0; every ultimate but 1981's falls to 0 (total IBNR
    -142,153), and this fails."""
    rows = [list(row) for row in RAA]
    rows[0] = [0] * 10
    result = ml(
        cells_of(rows, 1981),
        "gradient_boosting",
        seed=42,
        response="cumulative",
        zero_cells="missing",
        unsupported_factor="unity",
    )
    assert result.development["unity_fallback"].to_pylist() == [False] * 8 + [True, None]
    fitted = column(result.cells, "fitted_cumulative").reshape(10, 10)
    np.testing.assert_array_equal(fitted[1:, 9], fitted[1:, 8])
    factors = column(result.cells, "factor").reshape(10, 10)
    np.testing.assert_array_equal(factors[1:, 8], 1.0)
    assert column(result.origins, "ultimate")[1:].min() > 0
    assert total_ibnr(result) == pytest.approx(300.32, abs=0.005)


def test_a_first_age_with_only_zeros_shows_as_no_training_row(sklearn):
    """Every 12-month cell is zero: under ``"missing"`` with ``"unity"`` the
    fitted increment there is 0. ``unity_fallback`` says whether the factor
    from an age to the next is the 1 of the fallback, as for every other
    method, so no row marks the first age: it shows as ``n_trained`` 0."""
    cells = cells_of([[0, 10, 20, 25], [0, 12, 22], [0, 15], [0]])
    result = ml(cells, zero_cells="missing", unsupported_factor="unity")
    assert result.development["n_trained"].to_pylist() == [0, 3, 2, 1]
    assert result.development["unity_fallback"].to_pylist() == [False, False, False, None]
    first = column(result.cells, "fitted_increment").reshape(4, 4)[:, 0]
    np.testing.assert_array_equal(first[:3], 0.0)


def test_a_triangle_with_no_losses_is_refused_as_tweedie_glm_refuses_it(sklearn):
    """Every cell zero: ``tweedie_glm`` refuses it, and so does this, in the
    same words, under both zero-cell rules. Mutation: drop the check; under
    ``"observed"`` the forest answers 0 for every origin and this fails."""
    zeros = cells_of([[0, 0], [0]])
    expected = str(refusal_of(lambda: methods.tweedie_glm(zeros)))
    for rule in ("observed", "missing"):
        refusal = refusal_of(lambda rule=rule: ml(zeros, zero_cells=rule))
        assert refusal.reason == "not_identified", rule
        assert refusal.option == "cells", rule
        assert str(refusal) == expected.replace("tweedie_glm", "ml_development"), rule
    assert "cells has no losses to fit: every observed increment is zero" in expected


def test_a_closed_origin_is_never_refused_for_its_fitted_pattern(sklearn):
    """The oldest origin is at the last age, and boosting's fitted cumulative
    for it there is -0.03: nothing is projected from it, so it is answered with
    a factor of 1 and its latest as its ultimate. Mutation: check closed
    origins too; this is refused as ``negative_projection``."""
    cells = cells_of([[69, 3.45, 0.069, 0.00138], [135, 135, 202.5], [134, 201], [81]])
    result = ml(cells, "gradient_boosting", seed=42)
    assert result.origins["fitted_ultimate"][0].as_py() == pytest.approx(-0.0317, abs=5e-5)
    assert result.origins["cdf"][0].as_py() == 1.0
    assert result.origins["ultimate"][0].as_py() == 0.00138


def test_fewer_than_two_training_rows_is_refused(sklearn):
    one = refusal_of(lambda: ml(cells_of([[100.0]])))
    assert one.reason == "not_identified"
    assert "cells has 1 cell(s) to fit" in str(one)
    zeros = cells_of([[0.0, 0.0], [5.0]])
    refusal = refusal_of(lambda: ml(zeros, zero_cells="missing"))
    assert refusal.reason == "not_identified"
    assert refusal.option == "zero_cells"
    assert "not fitted under zero_cells='missing'" in str(refusal)


@pytest.mark.tieout
def test_the_training_rows_under_missing_are_chainladders(sklearn):
    """On the clrd paid triangles with a zero cell, no negative cumulative, and
    a nonzero cell in every origin and every age (so chainladder's fitted
    triangle keeps its shape; where a whole origin or age is zero it shrinks and
    lines its factors up against the wrong origins): the cells trained under
    ``zero_cells="missing"`` and their increments are chainladder's rows, and
    every fitted value it predicts is the same here to 1e-12. The ultimates
    differ on purpose where chainladder leaves a zero cell's fitted increment
    out of the fitted cumulative, or answers NaN for a zero latest amount."""
    grids = _clrd_grids()
    keys = [
        key
        for key, grid in sorted(grids.items())
        if (grid[~np.isnan(grid)] == 0).any()
        and np.nanmin(grid) >= 0
        and (np.nan_to_num(grid) != 0).any(axis=0).all()
        and (np.nan_to_num(grid) != 0).any(axis=1).all()
    ]
    outcomes: dict[str, int] = {}
    for key in keys:
        grid = grids[key]
        try:
            _, dev = _chainladder_fit(
                _chainladder_triangle(grid),
                "gradient_boosting",
                42,
                formula="C(development) + C(origin)",
                incremental=True,
            )
        except Exception:  # chainladder's own crashes on sparse triangles
            outcomes["chainladder_raised"] = outcomes.get("chainladder_raised", 0) + 1
            continue
        assert dev.triangle_ml_.shape[2:] == grid.shape, key
        try:
            result = ml(_cells_of_grid(grid), "gradient_boosting", seed=42, zero_cells="missing")
        except Refusal as refusal:
            assert refusal.reason == "negative_projection", key
            outcomes["refused"] = outcomes.get("refused", 0) + 1
            continue
        frame = dev.df_
        rows = {
            (int(round(o)), int(d) // 12 - 1): float(v)
            for o, d, v in zip(frame["origin"], frame["development"], frame["values"], strict=True)
        }
        trained = result.cells.filter(result.cells["trained"])
        mine = {
            (o - 1988, d // 12 - 1): v
            for o, d, v in zip(
                trained["origin"].to_pylist(),
                trained["dev_lag"].to_pylist(),
                trained["increment"].to_pylist(),
                strict=True,
            )
        }
        assert mine.keys() == rows.keys(), key
        for cell, value in rows.items():
            assert mine[cell] == pytest.approx(value, rel=1e-12), (key, cell)
        origins = [dt.datetime(1988 + i, 1, 1) for i in range(10)]
        _require_same_cells(result, _predicted(dev, origins, 12), 10, incremental=True)
        outcomes["matched"] = outcomes.get("matched", 0) + 1
    assert outcomes == {"matched": 29, "refused": 7}, outcomes


@pytest.mark.tieout
def test_pioneer_state_mutual_private_auto(sklearn):
    """Only 1996 (2 then 75) and 1997 (328) have losses. chainladder's fitted
    triangle shrinks to 2 x 2, so it develops 1996 by the factor meant for 12
    to 24 months: an ultimate of 2,807.76 for an origin whose losses stopped at
    75. Here, under ``"missing"`` with ``unsupported_factor="unity"``, every
    age past 24 months has no training row and develops by 1, so 1996's
    ultimate is its 75. Under ``"observed"`` the zeros are trained on, and the
    trees give 1996 a little development after 24 months."""
    grid = _clrd_grids()[("Pioneer State Mut Ins Co", "ppauto")]
    cells = _cells_of_grid(grid)
    expected, _ = _chainladder_fit(
        _chainladder_triangle(grid),
        "gradient_boosting",
        42,
        formula="C(development) + C(origin)",
        incremental=True,
    )
    assert expected[8] == pytest.approx(2807.76, abs=0.005)
    for estimator in ("random_forest", "gradient_boosting"):
        result = ml(cells, estimator, seed=42, zero_cells="missing", unsupported_factor="unity")
        assert result.origins["ultimate"][8].as_py() == 75.0, estimator
    observed = ml(cells, "gradient_boosting", seed=42)
    assert observed.origins["ultimate"][8].as_py() == pytest.approx(91.0, abs=0.01)


@pytest.mark.slow
@pytest.mark.tieout
def test_every_clrd_paid_triangle_is_answered_or_refused_by_name(sklearn):
    """Boosting with seed 42 on the 725 paid triangles with a nonzero cell, a
    regression count for the refusal order (rows, then ages, then the fitted
    latest amount, then a negative ultimate). About 40 seconds."""
    grids = _clrd_grids()
    counts = {}
    for rule in ("missing", "observed"):
        outcomes: dict[str, int] = {}
        for _key, grid in sorted(grids.items()):
            if np.nanmax(np.abs(grid)) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                try:
                    result = ml(_cells_of_grid(grid), "gradient_boosting", seed=42, zero_cells=rule)
                except Refusal as refusal:
                    outcomes[refusal.reason] = outcomes.get(refusal.reason, 0) + 1
                    continue
            _no_nan(result)
            outcomes["answered"] = outcomes.get("answered", 0) + 1
        counts[rule] = outcomes
    assert counts["missing"] == {
        "answered": 407,
        "negative_cumulative": 41,
        "negative_projection": 46,
        "no_link_ratio": 203,
        "not_identified": 28,
    }
    assert counts["observed"] == {
        "answered": 641,
        "negative_cumulative": 41,
        "negative_projection": 43,
    }


# -- 5. refusals that need a fit ---------------------------------------------------------


def test_a_negative_ultimate_is_refused_naming_the_origin(sklearn):
    """Every cumulative is zero or more, and boosting projects 2004 to -25.7."""
    cells = cells_of([[100, 2, 3, 4, 5], [110, 3, 4, 5], [95, 1, 2], [30, 1], [120]], first=2001)
    refusal = refusal_of(lambda: ml(cells, "gradient_boosting", seed=42))
    assert refusal.reason == "negative_projection"
    assert [(c.origin, c.dev_lag) for c in refusal.cells] == [(2004, None)]
    assert refusal.cells[0].value == pytest.approx(-25.7, abs=0.05)
    assert "the model projects a negative ultimate for 2004 (-25.7185 for the first)" in str(
        refusal
    )
    # the forest projects the same cells above zero
    assert column(ml(cells, seed=42).origins, "ultimate").min() > 0


def test_a_fitted_latest_amount_of_zero_or_less_is_refused(sklearn):
    """Every origin's cumulative falls from its first cell (136 to 3, 104 to 1),
    and the forest's fitted cumulative for 2003 at 24 months is -5.5, so 2003's
    factor to ultimate is undefined though its latest amount is 1. This is
    checked before the sign of any ultimate."""
    cells = cells_of([[136, 3, 2, 6.8], [75, 3, 90], [104, 1], [52]])
    refusal = refusal_of(lambda: ml(cells, seed=42))
    assert refusal.reason == "negative_projection"
    assert refusal.option == "estimator"
    assert [(c.origin, c.dev_lag) for c in refusal.cells] == [(2003, 24)]
    assert refusal.cells[0].value == pytest.approx(-5.508, abs=5e-4)
    assert str(refusal).startswith(
        "the model's fitted cumulative at the latest age of (2003, 24 months) is zero or less "
        "(-5.508 for the first), so the factor from there to ultimate is undefined"
    )


def test_scikit_learn_missing_names_the_extra(monkeypatch):
    """Runs in every CI leg: an absent scikit-learn is simulated."""
    monkeypatch.setitem(sys.modules, "sklearn", None)
    with pytest.raises(ImportError, match=r'pip install "ibnr\[ml\]"'):
        ml(workbook(), seed=42)


# -- 6. the tables ---------------------------------------------------------------------


def test_the_result_schema_is_pinned(sklearn):
    result = ml(workbook(), seed=42)
    assert result.method == "ml_development"
    assert result.link_ratios is None and result.coefficients is None
    assert result.as_of == dt.date(1997, 12, 31)
    f, i, s, b = pa.float64(), pa.int64(), pa.string(), pa.bool_()
    assert result.origins.schema == pa.schema(
        [
            ("origin", i),
            ("origin_period", pa.date32()),
            ("latest_dev_lag", i),
            ("latest", f),
            ("ultimate", f),
            ("ibnr", f),
            ("fitted_latest", f),
            ("fitted_ultimate", f),
            ("cdf", f),
            ("model_ibnr", f),
        ]
    )
    assert result.development.schema == pa.schema(
        [
            ("dev_lag", i),
            ("factor", f),
            ("cdf", f),
            ("pct_reported", f),
            ("n_trained", i),
            ("unity_fallback", b),
        ]
    )
    assert result.cells.schema == pa.schema(
        [
            ("origin", i),
            ("origin_period", pa.date32()),
            ("dev_lag", i),
            ("observed", b),
            ("trained", b),
            ("increment", f),
            ("fitted_increment", f),
            ("fitted_cumulative", f),
            ("factor", f),
            ("cdf", f),
        ]
    )
    assert result.totals.schema == pa.schema(
        [
            ("latest", f),
            ("ultimate", f),
            ("ibnr", f),
            ("model_ibnr", f),
            ("estimator", s),
            ("seed", i),
            ("n_estimators", i),
            ("max_depth", i),
            ("min_samples_leaf", i),
            ("learning_rate", f),
            ("response", s),
            ("origin_term", s),
            ("calendar_term", s),
            ("zero_cells", s),
            ("n_training_rows", i),
            ("scikit_learn_version", s),
        ]
    )
    assert result.cells.num_rows == 100
    assert result.totals["scikit_learn_version"][0].as_py() == sklearn.__version__
    assert result.totals["n_training_rows"][0].as_py() == 55
    observed = result.cells["observed"].to_pylist()
    assert sum(observed) == 55
    assert result.cells["increment"].null_count == 45


def _no_nan(result) -> None:
    for name in methods.TABLES:
        table = getattr(result, name)
        if table is None:
            continue
        for column_name in table.column_names:
            values = table[column_name]
            if pa.types.is_floating(values.type):
                assert not pc.any(pc.is_nan(values)).as_py(), (name, column_name)
                assert pc.all(pc.is_finite(values.drop_null())).as_py() in (True, None)


def test_missing_numbers_are_nulls_never_nan(sklearn):
    rows = [list(row) for row in RAA]
    rows[9][0] = 0
    rows[0] = [0] * 10
    result = ml(cells_of(rows, 1981), seed=42, zero_cells="missing", unsupported_factor="unity")
    _no_nan(result)
    assert result.origins["cdf"].null_count == 2
    _no_nan(ml(workbook(), "gradient_boosting"))


@pytest.mark.parametrize(
    ("label", "kind", "step"),
    [
        (lambda i: 1988 + i, pa.int64(), 12),
        (lambda i: str(1988 + i), pa.string(), 12),
        (lambda i: f"{2020 + (i + 2) // 4}Q{(i + 2) % 4 + 1}", pa.string(), 3),
        (lambda i: dt.date(1988 + i, 12, 31), pa.date32(), 12),
        (lambda i: dt.datetime(1988 + i, 1, 1), pa.timestamp("us"), 12),
    ],
    ids=["int", "text", "quarter", "period_end", "timestamp"],
)
def test_the_callers_labels_are_echoed_by_period_in_every_table(sklearn, label, kind, step):
    """Rows shuffled, so a label matched by position rather than by period
    would land on the wrong origin, and a training order taken from the input
    rows rather than the grid would change the forest's answer."""
    rows = [list(row) for row in RAA]
    ordered = cells_of(rows, step=step, label=label)
    order = list(range(ordered.num_rows))
    random.Random(7).shuffle(order)
    shuffled = ordered.take(pa.array(order))
    result = ml(shuffled, seed=42, dev_grain_months=step)
    reference = ml(ordered, seed=42, dev_grain_months=step)
    labels = [label(i) for i in range(len(rows))]
    assert result.origins["origin"].type == kind
    assert result.origins["origin"].to_pylist() == labels
    assert result.cells["origin"].type == kind
    assert result.cells["origin"].to_pylist() == [lab for lab in labels for _ in rows]
    for name in methods.TABLES:
        assert getattr(result, name) == getattr(reference, name), name


def test_to_polars_gives_the_cells_and_refuses_what_is_absent(sklearn):
    pytest.importorskip("polars")
    result = ml(workbook(), seed=42)
    assert result.to_polars("cells").height == 100
    for table in ("link_ratios", "coefficients"):
        refusal = refusal_of(lambda table=table: result.to_polars(table))
        assert refusal.reason == "invalid_option"
        assert refusal.method == "ml_development"


def test_the_kernel_takes_a_spec_and_refuses_anything_else():
    with pytest.raises(TypeError, match="MLDevelopmentSpec"):
        fit_ml_development_grid(_grid(RAA), {"estimator": "random_forest"})


def test_a_kernel_refusal_names_the_period_start(sklearn):
    rows = [list(row) for row in RAA]
    rows[0] = [0] * 10
    refusal = refusal_of(
        lambda: fit_ml_development_grid(
            _grid(rows), MLDevelopmentSpec("random_forest", zero_cells="missing")
        )
    )
    assert refusal.method is None
    assert [(c.origin, c.origin_period) for c in refusal.cells] == [(None, dt.date(1981, 1, 1))]


# -- 7. the import ---------------------------------------------------------------------


def test_importing_ibnr_its_methods_and_kernels_loads_no_scikit_learn():
    """Mutation: import scikit-learn at the top of ``kernels/ml_development.py``;
    this fails naming it (where it is installed; ``test_import_purity`` checks
    the same in a process that blocks it)."""
    code = (
        "import sys\n"
        "import ibnr, ibnr.methods, ibnr.kernels\n"
        "import ibnr.kernels.ml_development\n"
        "from ibnr.kernels import MLDevelopmentSpec, fit_ml_development_grid\n"
        "assert 'sklearn' not in sys.modules, 'scikit-learn was imported'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
