"""``zero_cells``: what the chain-ladder fits take a cumulative of exactly zero to be.

``"observed"`` is each kernel's own behaviour and stays its default: the
conventional kernel leaves out only the link OUT of a zero (its ratio is
undefined) and keeps the link INTO it (ratio 0), and Mack's factor keeps both
in its volume sums, as R's ``MackChainLadder`` does. ``"missing"`` is
chainladder-python's rule, which stores a zero cell as missing: a link ratio is
used only when neither of its two cells is zero. ``ibnr.methods`` defaults to
``"missing"``.

Five groups of tests:

1. The default is ``"observed"``: an explicit ``"observed"`` gives the same
   bytes as leaving the option out, through every entry point. (That the
   default is also byte-identical to the code before the option existed was
   shown once, outside the suite, against origin/main on raa, genins and clrd
   triangles with zeros; a test here cannot import the old code.)
2. chainladder-python tie-outs under ``"missing"`` on raa with (a) a zero at an
   origin's first age, (b) an interior write-down to zero, and (c) a zero on
   the latest diagonal, newest and older origin. Built from a frame so that
   chainladder itself turns the zeros into missing cells. Then every clrd paid
   triangle (from chainladder's shipped ``clrd.csv``, whose zeros are still
   there), where the rule is what separates the two libraries.
3. Delivery: the option changes the answer where zeros exist, through
   ``ConventionalCandidate`` (``fit_conventional`` and ``fit_conventional_grid``),
   ``conventional_grid``, ``replay_conventional``/``select_conventional``,
   ``fit_mack``, ``fit_mack_grid``, ``fit_mack_many`` and each ``methods``
   function, and each changed number is the one the rule predicts.
4. A zero latest amount under ``"missing"``: ultimate 0 and Mack standard error
   0, the limit of Mack's formula, rather than a division by zero. Then the two
   places ``"missing"`` has to follow chainladder-python beyond the factors: a
   sigma at a step left with one link ratio, and a history window with zeros
   in it.
5. Refusals: a bad setting, a development step the rule leaves with no link
   ratio, the one-year claims development result on a fit the rule changed,
   and the codec round trip of the setting.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ibnr import methods
from ibnr.kernels.cdr import (
    DiagonalGenerator,
    MackDiagonal,
    ODPBootstrapDiagonal,
    one_year_cdr,
    rereserve,
    simulate_one_year_cdr,
)
from ibnr.kernels.contract import cohort_grid_frame
from ibnr.kernels.conventional import (
    ConventionalCandidate,
    conventional_grid,
    fit_conventional,
    fit_conventional_grid,
)
from ibnr.kernels.mack import (
    MackFit,
    MackFitPanel,
    fit_mack,
    fit_mack_grid,
    fit_mack_many,
    simulate_ultimates,
)
from ibnr.kernels.replay import replay_conventional
from ibnr.kernels.selection import select_conventional
from ibnr.triangle import Triangle

from .conftest import make_cohort_triangle

# raa, the Mack (1993) triangle: accident years 1981 to 1990, annual ages 12 to 120.
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
ORIGINS = [dt.date(1981 + i, 1, 1) for i in range(10)]
PREMIUM = dict(zip(ORIGINS, np.linspace(20000.0, 40000.0, 10).tolist(), strict=True))


def zeroed(*cells: tuple[int, int]) -> list[list[float]]:
    """raa with the given (origin index, dev index) cells set to 0."""
    rows = [[float(v) for v in row] for row in RAA]
    for i, j in cells:
        rows[i][j] = 0.0
    return rows


#: (a) zero paid at 12 months for 1982, then positive
FIRST_AGE = zeroed((1, 0))
#: (b) 1983 written down to zero at 60 and 72 months, then positive again at 84
WRITE_DOWN = zeroed((2, 4), (2, 5))
#: (c) a zero on the latest diagonal: the newest origin, 1990, at 12 months
LATEST_NEWEST = zeroed((9, 0))
#: (c) a zero on the latest diagonal of an older origin: 1986 at 60 months
LATEST_OLDER = zeroed((5, 4))
CASES = {
    "first_age": FIRST_AGE,
    "write_down": WRITE_DOWN,
    "latest_newest": LATEST_NEWEST,
    "latest_older": LATEST_OLDER,
}

#: chainladder-python 0.9.2's total Mack standard error for LATEST_NEWEST
#: (``MackChainladder`` with its default log-linear sigma), which leaves 1990's
#: ultimate missing; tied out against chainladder itself below, and pinned here
#: for the core leg, which has no chainladder.
LATEST_NEWEST_TOTAL_SE = 10008.213647336026


def frame(rows=RAA) -> pd.DataFrame:
    out: dict[str, list] = {"origin_period": [], "dev_lag": [], "value": []}
    for origin, values in zip(ORIGINS, rows, strict=True):
        for j, value in enumerate(values):
            out["origin_period"].append(origin)
            out["dev_lag"].append(12 * (j + 1))
            out["value"].append(float(value))
    return pd.DataFrame(out)


def grid(rows=RAA) -> dict:
    return cohort_grid_frame(frame(rows), dev_grain_months=12, measure="cumulative")


def cells(rows=RAA) -> pa.Table:
    data = frame(rows)
    return pa.table(
        {
            "origin_period": pa.array(list(data["origin_period"]), pa.date32()),
            "dev_lag": pa.array(data["dev_lag"].to_numpy(), pa.int64()),
            "value": pa.array(data["value"].to_numpy(), pa.float64()),
        }
    )


def triangle(rows=RAA) -> Triangle:
    matrix = np.full((10, 10), np.nan)
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = row
    return make_cohort_triangle(None, matrix, start_year=1981)


def column(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).to_numpy(zero_copy_only=False)


def by_hand(rows, step: int, keep: str) -> float:
    """The volume-weighted factor from dev index ``step`` to the next, written out.

    ``keep`` is which observed pairs enter the sums: ``"all"`` (Mack under
    "observed"), ``"from_positive"`` (the conventional kernel under "observed",
    whose ratio out of a zero is undefined) or ``"nonzero"`` ("missing")."""
    pairs = [(r[step], r[step + 1]) for r in rows if len(r) > step + 1]
    if keep == "from_positive":
        pairs = [(a, b) for a, b in pairs if a != 0]
    elif keep == "nonzero":
        pairs = [(a, b) for a, b in pairs if a != 0 and b != 0]
    return sum(b for _, b in pairs) / sum(a for a, _ in pairs)


def as_of_rows(rows, year: int):
    """The rows known at the end of ``year``: origin i has its cells up to then."""
    known = [r[: year - 1981 - i + 1] for i, r in enumerate(rows)]
    return [r for r in known if r]


def same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and (left.tobytes() == right.tobytes())
    )


# -- 1. the default is "observed" ----------------------------------------------------


@pytest.mark.parametrize("rows", [RAA, FIRST_AGE, WRITE_DOWN], ids=["raa", "first", "down"])
def test_the_kernels_default_to_observed(rows):
    g = grid(rows)
    for base in (
        ConventionalCandidate(),
        ConventionalCandidate(method="gcc", decay=0.8),
        ConventionalCandidate(average="simple", history_periods=4),
    ):
        premium = None if base.method == "cl" else PREMIUM
        default = fit_conventional_grid(g, base, premium=premium)
        explicit = fit_conventional_grid(
            g, ConventionalCandidate(**{**vars(base), "zero_cells": "observed"}), premium=premium
        )
        assert base.zero_cells == "observed"
        assert same_bytes(default.factors, explicit.factors)
        assert same_bytes(default.origins["ultimate"], explicit.origins["ultimate"])
        assert default.factor_selection.equals(explicit.factor_selection)
    for rule in ("mack", "log_linear"):
        default = fit_mack_grid(g, sigma_rule=rule)
        explicit = fit_mack_grid(g, sigma_rule=rule, zero_cells="observed")
        assert default.zero_cells == "observed"
        for name in ("f", "sigma2", "s", "n_obs", "n_pos", "full"):
            assert same_bytes(getattr(default, name), getattr(explicit, name)), name
        for key, value in default.msep_runoff().items():
            assert same_bytes(value, explicit.msep_runoff()[key]), key
        assert same_bytes(one_year_cdr(default).msep, one_year_cdr(explicit).msep)
        assert same_bytes(
            simulate_one_year_cdr(default, n_draws=200, seed=1).samples,
            simulate_one_year_cdr(explicit, n_draws=200, seed=1).samples,
        )


def test_the_triangle_paths_and_replay_default_to_observed():
    tri = triangle(FIRST_AGE)
    assert same_bytes(
        fit_mack(tri, loss_field="paid_loss").f,
        fit_mack(tri, loss_field="paid_loss", zero_cells="observed").f,
    )
    panel = fit_mack_many(tri, loss_field="paid_loss")
    assert same_bytes(panel[()].f, fit_mack(tri, loss_field="paid_loss").f)
    assert panel[()].zero_cells == "observed"
    tri = triangle(WRITE_DOWN)
    default = replay_conventional(tri, {"cl": replay_candidate()}, REPLAY_DATES)
    explicit = replay_conventional(tri, {"cl": replay_candidate("observed")}, REPLAY_DATES)
    assert default.cells.equals(explicit.cells)
    assert len(default.cells) > 0


REPLAY_DATES = ["1988-12-31", "1989-12-31", "1990-12-31"]


def replay_candidate(zero_cells: str | None = None) -> ConventionalCandidate:
    # early fits have not seen the last ages, so a factor of 1.0 stands in there
    settings = {} if zero_cells is None else {"zero_cells": zero_cells}
    return ConventionalCandidate(horizon=120, unsupported_factor="unity", **settings)


# -- 2. chainladder-python tie-outs under "missing" -----------------------------------


@pytest.fixture(scope="module")
def cl():
    return pytest.importorskip("chainladder")


def cl_triangle(cl, rows):
    """A chainladder triangle from a frame, so chainladder itself drops the zeros."""
    data = [
        {"origin": f"{1981 + i}-01-01", "valuation": f"{1981 + i + j}-12-31", "paid": float(v)}
        for i, row in enumerate(rows)
        for j, v in enumerate(row)
    ]
    return cl.Triangle(
        pd.DataFrame(data),
        origin="origin",
        development="valuation",
        columns="paid",
        cumulative=True,
    )


def cl_weight(cl, tri):
    weight = tri.latest_diagonal * 0
    amounts = np.array(list(PREMIUM.values()))
    weight.values = np.nan_to_num(weight.values) + amounts[None, None, :, None]
    return weight


def last_column(values) -> np.ndarray:
    return np.asarray(values.values if hasattr(values, "values") else values)[0, 0, :, -1]


def assert_matches(ours: np.ndarray, theirs: np.ndarray, *, skip_first: bool = False) -> None:
    """chainladder leaves a number missing where ibnr's answer is 0: an origin whose
    latest cumulative is zero. Everywhere else the two agree."""
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    if skip_first:  # chainladder reports NaN rather than 0 for the fully developed origin
        ours, theirs = ours[1:], theirs[1:]
    missing = np.isnan(theirs)
    np.testing.assert_array_equal(ours[missing], 0.0)
    np.testing.assert_allclose(ours[~missing], theirs[~missing], rtol=1e-9)


@pytest.mark.tieout
@pytest.mark.parametrize("case", list(CASES))
def test_chain_ladder_matches_chainladder(cl, case):
    rows = CASES[case]
    tri = cl_triangle(cl, rows)
    reference = cl.Chainladder().fit(tri)
    result = methods.chain_ladder(cells(rows))
    assert_matches(column(result.origins, "ultimate"), reference.ultimate_.values.ravel())
    np.testing.assert_allclose(
        result.development["factor"].to_pylist()[:-1],
        reference.ldf_.values.ravel()[:9],
        rtol=1e-12,
    )
    kernel = fit_conventional_grid(grid(rows), ConventionalCandidate(zero_cells="missing"))
    np.testing.assert_allclose(kernel.factors, reference.ldf_.values.ravel()[:9], rtol=1e-12)


@pytest.mark.tieout
@pytest.mark.parametrize("case", list(CASES))
def test_bornhuetter_ferguson_and_cape_cod_match_chainladder(cl, case):
    rows = CASES[case]
    tri = cl_triangle(cl, rows)
    weight = cl_weight(cl, tri)
    bf = methods.bornhuetter_ferguson(cells(rows), premium=PREMIUM, expected_loss_ratio=0.6)
    reference = cl.BornhuetterFerguson(apriori=0.6).fit(tri, sample_weight=weight)
    # chainladder's BF reads a missing latest cell as 0 and adds the BF reserve,
    # which is exactly what ibnr does with the zero it keeps
    np.testing.assert_allclose(
        column(bf.origins, "ultimate"), reference.ultimate_.values.ravel(), rtol=1e-9
    )
    cc = methods.cape_cod(cells(rows), premium=PREMIUM, decay=1.0)
    reference = cl.CapeCod(trend=0, decay=1.0).fit(tri, sample_weight=weight)
    np.testing.assert_allclose(
        column(cc.origins, "ultimate"), reference.ultimate_.values.ravel(), rtol=1e-9
    )
    np.testing.assert_allclose(
        column(cc.origins, "expected_loss_ratio"),
        np.asarray(reference.apriori_.values).ravel(),
        rtol=1e-9,
    )


@pytest.mark.tieout
@pytest.mark.parametrize("case", list(CASES))
def test_mack_matches_chainladder(cl, case):
    rows = CASES[case]
    tri = cl_triangle(cl, rows)
    reference = cl.MackChainladder().fit(tri)  # log-linear sigma, chainladder's default
    development = cl.Development().fit(tri)
    result = methods.mack(cells(rows))
    origins, totals = result.origins, result.totals
    assert_matches(column(origins, "ultimate"), reference.ultimate_.values.ravel())
    assert_matches(
        column(origins, "mack_se"), last_column(reference.mack_std_err_), skip_first=True
    )
    assert_matches(column(origins, "parameter_se"), last_column(reference.parameter_risk_))
    assert_matches(column(origins, "process_se"), last_column(reference.process_risk_))
    for ours, theirs in (
        ("mack_se", reference.total_mack_std_err_),
        ("parameter_se", reference.total_parameter_risk_),
        ("process_se", reference.total_process_risk_),
    ):
        assert totals[ours][0].as_py() == pytest.approx(
            float(np.asarray(theirs).ravel()[-1]), rel=1e-9
        ), ours
    np.testing.assert_allclose(
        column(result.development, "sigma")[:-1],
        np.asarray(development.sigma_.values).ravel(),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        column(result.development, "std_err")[:-1],
        np.asarray(development.std_err_.values).ravel(),
        rtol=1e-9,
    )


@pytest.mark.tieout
def test_the_zero_newest_origin_total_se_is_chainladders(cl):
    reference = cl.MackChainladder().fit(cl_triangle(cl, LATEST_NEWEST))
    assert float(np.asarray(reference.total_mack_std_err_).ravel()[0]) == pytest.approx(
        LATEST_NEWEST_TOTAL_SE, rel=1e-12
    )
    assert np.isnan(reference.ultimate_.values.ravel()[-1])


#: the clrd paid triangles on which chainladder and the conventional kernel under
#: "observed" disagree; every one of them has a zero cumulative
CLRD_ZERO_DISAGREEMENTS = {
    ("Amguard Norguard & Eastguard Grp", "prodliab"),
    ("British Amer Ins Co", "wkcomp"),
    ("Crusader Ins Co", "othliab"),
    ("Farmers Home Mut Fire Ins Co", "othliab"),
    ("Golden Bear Ins Co", "othliab"),
    ("IMT Ins Co Mut", "prodliab"),
    ("National Automotive Ins", "ppauto"),
    ("Ocean Harbor Cas Ins Co", "ppauto"),
    ("Otsego Mut Fire Ins Co", "othliab"),
    ("Philadelphia CBSP Grp", "othliab"),
    ("Red Shield Ins Co", "wkcomp"),
}


@pytest.fixture(scope="module")
def clrd(cl):
    """Every clrd paid triangle as a grid, and chainladder's ultimates for it.

    The grids come from chainladder's shipped ``clrd.csv`` rather than from its
    ``Triangle``, which has already turned every zero into a missing cell."""
    raw = pd.read_csv(os.path.join(os.path.dirname(cl.__file__), "utils", "data", "clrd.csv"))
    paid = cl.load_sample("clrd")["CumPaidLoss"]
    ultimates = np.asarray(cl.Chainladder().fit(paid).ultimate_.values)[:, 0, :, -1]
    reference = {tuple(paid.index.iloc[k]): ultimates[k] for k in range(len(paid.index))}
    grids = {}
    for key, rows in raw.groupby(["GRNAME", "LOB"]):
        cohort = pd.DataFrame(
            {
                "origin_period": [dt.date(int(year), 1, 1) for year in rows["AccidentYear"]],
                "dev_lag": rows["DevelopmentLag"].to_numpy() * 12,
                "value": rows["CumPaidLoss"].to_numpy(dtype=float),
            }
        )
        try:
            grids[key] = cohort_grid_frame(cohort, dev_grain_months=12, measure="cumulative")
        except ValueError:
            continue  # a cohort that is not one run-off triangle: nothing to compare
    return grids, reference, raw


@pytest.mark.tieout
def test_every_clrd_paid_triangle_matches_chainladder_under_missing(clrd):
    # chainladder uses a factor of 1.0 where no link ratio is left, hence "unity"
    grids, reference, _ = clrd
    answered = {"observed": 0, "missing": 0}
    disagree = {"observed": set(), "missing": set()}
    for key, cohort in grids.items():
        for rule in answered:
            candidate = ConventionalCandidate(unsupported_factor="unity", zero_cells=rule)
            try:
                ours = fit_conventional_grid(cohort, candidate).origins["ultimate"].to_numpy()
            except ValueError:
                continue  # negative cumulatives, refused by name under either rule
            answered[rule] += 1
            theirs = reference[key]
            missing = np.isnan(theirs)
            if not (
                np.allclose(ours[~missing], theirs[~missing], rtol=1e-9, atol=1e-9)
                and (ours[missing] == 0).all()
            ):
                disagree[rule].add(key)
    assert answered == {"observed": 731, "missing": 731}
    assert disagree["observed"] == CLRD_ZERO_DISAGREEMENTS
    assert disagree["missing"] == set()


@pytest.mark.tieout
def test_mack_on_the_disagreeing_clrd_triangles(cl, clrd):
    """Where chainladder has a total Mack standard error, ibnr under "missing" gives
    the same one; where chainladder's is missing, a development step has no link
    ratio left, and ibnr refuses that step by name."""
    _, _, raw = clrd
    matched = refused = 0
    for name, lob in sorted(CLRD_ZERO_DISAGREEMENTS):
        rows = raw[(raw["GRNAME"] == name) & (raw["LOB"] == lob)]
        reference = cl.MackChainladder().fit(
            cl.Triangle(
                pd.DataFrame(
                    {
                        "origin": rows["AccidentYear"].astype(str),
                        "valuation": rows["DevelopmentYear"].astype(str) + "-12-31",
                        "paid": rows["CumPaidLoss"].astype(float),
                    }
                ),
                origin="origin",
                development="valuation",
                columns="paid",
                cumulative=True,
            )
        )
        theirs = float(np.asarray(reference.total_mack_std_err_).ravel()[0])
        cohort = pd.DataFrame(
            {
                "origin_period": [dt.date(int(y), 1, 1) for y in rows["AccidentYear"]],
                "dev_lag": rows["DevelopmentLag"].to_numpy() * 12,
                "value": rows["CumPaidLoss"].to_numpy(dtype=float),
            }
        )
        g = cohort_grid_frame(cohort, dev_grain_months=12, measure="cumulative")
        if np.isnan(theirs):
            with pytest.raises(ValueError, match="zero_cells='missing' leaves those link"):
                fit_mack_grid(g, sigma_rule="log_linear", zero_cells="missing")
            refused += 1
            continue
        fit = fit_mack_grid(g, sigma_rule="log_linear", zero_cells="missing")
        assert np.sqrt(fit.msep_runoff()["msep_total"]) == pytest.approx(theirs, rel=1e-9)
        matched += 1
    assert (matched, refused) == (7, 4)


# -- 3. delivery: the option changes the answer where zeros exist ---------------------


#: the conventional kernel's two rules differ on the link INTO a zero: WRITE_DOWN's
#: 1983 goes 16141 -> 0 from 48 to 60 months, which is development step 3
INTO_ZERO = 3
CONVENTIONAL_KEEP = {"observed": "from_positive", "missing": "nonzero"}


def test_the_conventional_grid_path_follows_the_rule():
    fits = {
        rule: fit_conventional_grid(grid(WRITE_DOWN), ConventionalCandidate(zero_cells=rule))
        for rule in CONVENTIONAL_KEEP
    }
    for rule, fit in fits.items():
        for step in range(9):
            assert fit.factors[step] == pytest.approx(
                by_hand(WRITE_DOWN, step, CONVENTIONAL_KEEP[rule])
            ), (rule, step)
    assert fits["missing"].factors[INTO_ZERO] != fits["observed"].factors[INTO_ZERO]
    # a zero at an origin's first age drops the same pair under both rules, and
    # says why differently
    reasons = {
        rule: fit_conventional_grid(
            grid(FIRST_AGE), ConventionalCandidate(zero_cells=rule)
        ).factor_selection.set_index(["origin_period", "from_dev_lag"])["reason"]
        for rule in CONVENTIONAL_KEEP
    }
    assert reasons["observed"][(ORIGINS[1], 12)] == "undefined_ratio"
    assert reasons["missing"][(ORIGINS[1], 12)] == "zero_cell"


def test_the_link_into_a_zero_is_used_only_under_observed():
    observed = fit_conventional_grid(grid(WRITE_DOWN), ConventionalCandidate())
    missing = fit_conventional_grid(grid(WRITE_DOWN), ConventionalCandidate(zero_cells="missing"))

    def row(fit, lag):
        table = fit.factor_selection
        return table[(table["origin_period"] == ORIGINS[2]) & (table["from_dev_lag"] == lag)].iloc[
            0
        ]

    # into the zero (48 -> 60): a ratio of 0, used under "observed" only
    assert (row(observed, 48)["ratio"], row(observed, 48)["reason"]) == (0.0, "included")
    assert row(missing, 48)["reason"] == "zero_cell"
    # between the zeros (60 -> 72) and out of them (72 -> 84): undefined, or a zero cell
    for lag in (60, 72):
        assert row(observed, lag)["reason"] == "undefined_ratio"
        assert row(missing, lag)["reason"] == "zero_cell"
    assert missing.factors[3] != observed.factors[3]
    assert (missing.factor_summary["n_selected"] <= observed.factor_summary["n_selected"]).all()


def test_the_triangle_path_follows_the_rule():
    tri = triangle(WRITE_DOWN)
    for rule, keep in CONVENTIONAL_KEEP.items():
        through_triangle = fit_conventional(
            tri, ConventionalCandidate(zero_cells=rule), as_of="1990-12-31"
        )
        through_grid = fit_conventional_grid(
            grid(WRITE_DOWN), ConventionalCandidate(zero_cells=rule)
        )
        np.testing.assert_array_equal(through_triangle.factors, through_grid.factors)
        assert through_triangle.factors[INTO_ZERO] == pytest.approx(
            by_hand(WRITE_DOWN, INTO_ZERO, keep)
        )


def test_conventional_grid_carries_the_rule_to_every_candidate():
    candidates = conventional_grid(history_periods=(None, 5), decays=(0.5,), zero_cells="missing")
    assert len(candidates) == 4
    assert {c.zero_cells for c in candidates} == {"missing"}
    fit = fit_conventional_grid(grid(WRITE_DOWN), candidates[0])
    assert fit.factors[INTO_ZERO] == pytest.approx(by_hand(WRITE_DOWN, INTO_ZERO, "nonzero"))


def test_replay_and_selection_follow_the_rule():
    tri = triangle(WRITE_DOWN)
    runs = {
        rule: replay_conventional(tri, {"cl": replay_candidate(rule)}, REPLAY_DATES)
        for rule in CONVENTIONAL_KEEP
    }
    for rule, run in runs.items():
        assert len(run.fits) == 3
        for fit in run.fits.values():
            assert fit.candidate.zero_cells == rule
            known = as_of_rows(WRITE_DOWN, fit.as_of.year)
            assert fit.factors[INTO_ZERO] == pytest.approx(
                by_hand(known, INTO_ZERO, CONVENTIONAL_KEEP[rule])
            )
    assert not np.allclose(
        runs["observed"].cells["new_ultimate"], runs["missing"].cells["new_ultimate"]
    )
    selection = select_conventional(runs["missing"], selection_as_of="1990-12-31")
    assert selection.candidate.zero_cells == "missing"
    assert selection.fit.factors[INTO_ZERO] == pytest.approx(
        by_hand(WRITE_DOWN, INTO_ZERO, "nonzero")
    )


def test_the_mack_entry_points_follow_the_rule():
    # Mack under "observed" keeps even 1982's 0 -> 4285 in its volume sums
    expected = {
        "observed": by_hand(FIRST_AGE, 0, "all"),
        "missing": by_hand(FIRST_AGE, 0, "nonzero"),
    }
    tri = triangle(FIRST_AGE)
    other = triangle(RAA).execute().assign(company="B")
    both = Triangle.from_long(
        pd.concat([tri.execute().assign(company="A"), other], ignore_index=True),
        segments=["company"],
        measure="cumulative",
    )
    for rule, factor in expected.items():
        fits = {
            "fit_mack_grid": fit_mack_grid(grid(FIRST_AGE), zero_cells=rule),
            "fit_mack": fit_mack(tri, loss_field="paid_loss", zero_cells=rule),
            "fit_mack_many": fit_mack_many(both, loss_field="paid_loss", zero_cells=rule)["A"],
        }
        for name, fit in fits.items():
            assert fit.zero_cells == rule, name
            assert fit.f[0] == pytest.approx(factor), name
            assert fit.n_obs[0] == (8 if rule == "missing" else 9), name
            assert fit.s[0] == pytest.approx(
                sum(r[0] for r in FIRST_AGE[:9]) - (0 if rule == "observed" else FIRST_AGE[1][0])
            ), name
            assert fit.zero_links == (1 if rule == "missing" else 0), name
        # the cohort with no zero is the same fit under either rule
        clean = fit_mack_many(both, loss_field="paid_loss", zero_cells=rule)["B"]
        assert same_bytes(clean.f, fit_mack_grid(grid(RAA)).f)
    # sigma runs over the same kept pairs under "missing", so n_pos == n_obs there
    missing = fit_mack_grid(grid(WRITE_DOWN), zero_cells="missing")
    np.testing.assert_array_equal(missing.n_pos, missing.n_obs)
    observed = fit_mack_grid(grid(WRITE_DOWN))
    assert (observed.n_obs[3:6] - missing.n_obs[3:6]).tolist() == [1, 1, 1]
    assert not np.allclose(missing.sigma2, observed.sigma2)


METHODS = {
    "chain_ladder": {},
    "bornhuetter_ferguson": {"premium": PREMIUM, "expected_loss_ratio": 0.6},
    "cape_cod": {"premium": PREMIUM},
    "mack": {},
}


@pytest.mark.parametrize("method", list(METHODS))
def test_each_method_defaults_to_missing_and_takes_observed(method):
    run = getattr(methods, method)
    kwargs = METHODS[method]
    default = run(cells(WRITE_DOWN), **kwargs)
    missing = run(cells(WRITE_DOWN), zero_cells="missing", **kwargs)
    observed = run(cells(WRITE_DOWN), zero_cells="observed", **kwargs)
    assert default.origins.equals(missing.origins)
    assert default.development.equals(missing.development)
    keep = {"missing": "nonzero", "observed": "all" if method == "mack" else "from_positive"}
    for rule, result in (("missing", missing), ("observed", observed)):
        assert result.development["factor"][INTO_ZERO].as_py() == pytest.approx(
            by_hand(WRITE_DOWN, INTO_ZERO, keep[rule])
        ), rule
    assert not np.allclose(
        column(missing.origins, "ultimate"), column(observed.origins, "ultimate")
    )


@pytest.mark.parametrize("method", ["chain_ladder", "bornhuetter_ferguson", "cape_cod"])
def test_link_ratios_name_the_zero_cells(method):
    result = getattr(methods, method)(cells(WRITE_DOWN), **METHODS[method])
    ratios = result.link_ratios
    zero = ratios.filter(pc.equal(ratios["reason"], "zero_cell"))
    assert zero["origin_period"].to_pylist() == [ORIGINS[2]] * 3
    assert zero["from_dev_lag"].to_pylist() == [48, 60, 72]
    assert zero["included"].to_pylist() == [False] * 3
    # the ratio into the zero is still shown, as 0; the two out of a zero have none
    assert zero["ratio"].to_pylist() == [0.0, None, None]
    assert "undefined_ratio" not in ratios["reason"].to_pylist()


# -- 4. a zero latest amount under "missing" ------------------------------------------


def test_a_zero_latest_amount_has_ultimate_and_standard_error_zero():
    result = methods.mack(cells(LATEST_NEWEST))
    assert column(result.origins, "ultimate")[-1] == 0.0
    for name in ("mack_se", "parameter_se", "process_se"):
        assert column(result.origins, name)[-1] == 0.0, name
    assert result.totals["mack_se"][0].as_py() == pytest.approx(LATEST_NEWEST_TOTAL_SE, rel=1e-12)
    # every other origin is as on raa, whose 1990 row adds no link ratio
    raa = methods.mack(cells(RAA))
    for name in ("ultimate", "mack_se", "parameter_se", "process_se"):
        np.testing.assert_allclose(
            column(result.origins, name)[:-1], column(raa.origins, name)[:-1], rtol=1e-12
        )


def test_the_zero_is_the_limit_of_a_vanishing_latest_amount():
    """Mack's msep for an origin is its latest amount times a finite number, plus
    that amount squared times another, so it shrinks in proportion to the amount
    and the total converges on the one "missing" gives for a zero. 1990's cell
    is the one to shrink: it starts no link ratio, so the factors stay put."""

    def near(amount):
        rows = [list(r) for r in RAA]
        rows[9][0] = amount
        return fit_mack_grid(grid(rows), sigma_rule="log_linear").msep_runoff()

    at_zero = fit_mack_grid(grid(LATEST_NEWEST), zero_cells="missing", sigma_rule="log_linear")
    risk = at_zero.msep_runoff()
    assert risk["msep"][9] == risk["process"][9] == risk["parameter"][9] == 0.0
    assert near(1e-6)["msep"][9] == pytest.approx(100 * near(1e-8)["msep"][9], rel=1e-4)
    for key in ("msep_total", "process_total", "parameter_total"):
        assert risk[key] == pytest.approx(near(1e-8)[key], rel=1e-10), key
        assert risk[key] != near(1e-8)[key], key  # a limit, not a coincidence


def test_a_zero_latest_amount_simulates_as_zero():
    fit = fit_mack_grid(grid(LATEST_NEWEST), zero_cells="missing")
    samples = simulate_ultimates(fit, n_draws=500, seed=0).samples
    np.testing.assert_array_equal(samples[:, 9], 0.0)
    assert np.isfinite(samples).all() and samples[:, 8].std() > 0


def test_observed_still_refuses_a_zero_latest_amount():
    fit = fit_mack_grid(grid(LATEST_NEWEST))
    np.testing.assert_array_equal(fit.ultimate[-1], 0.0)  # the point estimate is fine
    with pytest.raises(ValueError, match="non-positive cumulative on the latest diagonal"):
        fit.msep_runoff()
    # through methods, the refusal names the caller's origin and the way out,
    # not MackFit attributes a ReserveResult does not have
    with pytest.raises(ValueError, match="latest cumulative is zero: 1990-01-01") as refused:
        methods.mack(cells(LATEST_NEWEST), zero_cells="observed")
    assert "zero_cells='missing' (this function's default) gives" in str(refused.value)
    assert ".ultimate" not in str(refused.value)


def test_missing_still_refuses_a_negative_latest_amount():
    rows = [list(r) for r in RAA]
    rows[9][0] = -5.0
    fit = fit_mack_grid(grid(rows), zero_cells="missing")
    with pytest.raises(ValueError, match="non-positive cumulative on the latest diagonal"):
        fit.msep_runoff()


# -- 4b. what "missing" has to do as chainladder does ------------------------------------

#: the first step left with one link ratio: every origin but 1981 is 0 at 12 months
SINGLE_FIRST = zeroed(*[(i, 0) for i in range(1, 10)])
#: a middle step left with one link ratio: 1982 to 1985 are 0 at 48 months and 1986
#: sits at 0 on its latest diagonal, so only 1981 links 48 to 60 months
SINGLE_MIDDLE = zeroed((1, 3), (2, 3), (3, 3), (4, 3), (5, 4))
SIGMA_GAPS = {"single_first": SINGLE_FIRST, "single_middle": SINGLE_MIDDLE}

#: chainladder-python 0.9.2's sigma at the step left with one link ratio and its
#: total Mack standard error (log-linear sigma), tied out below and pinned here for
#: the core leg. The fill comes from one regression over every estimated step; the
#: rule for the last step alone, which regresses on the steps before it, gives 0.0
#: at the first step and 0.3093 at the middle one.
SIGMA_GAP_REFERENCE = {
    "single_first": (0, 55.56947821, 10050.734540473504),
    "single_middle": (3, 11.12389049, 22673.095435750514),
}


@pytest.mark.parametrize("case", list(SIGMA_GAPS))
def test_a_sigma_left_with_one_link_ratio_is_filled_from_every_step(case):
    step, sigma, total = SIGMA_GAP_REFERENCE[case]
    fit = fit_mack_grid(grid(SIGMA_GAPS[case]), sigma_rule="log_linear", zero_cells="missing")
    assert fit.n_obs[step] == 1
    assert np.sqrt(fit.sigma2[step]) == pytest.approx(sigma, rel=1e-8)
    assert np.sqrt(fit.msep_runoff()["msep_total"]) == pytest.approx(total, rel=1e-9)
    result = methods.mack(cells(SIGMA_GAPS[case]))
    assert result.totals["mack_se"][0].as_py() == pytest.approx(total, rel=1e-9)


def test_mack_rule_fills_a_middle_gap_from_the_two_steps_before_and_refuses_the_first():
    fit = fit_mack_grid(grid(SINGLE_MIDDLE), sigma_rule="mack", zero_cells="missing")
    last, prev = fit.sigma2[2], fit.sigma2[1]
    assert fit.sigma2[3] == min(last**2 / prev, last, prev)
    with pytest.raises(
        ValueError,
        match=r"the link ratios from 12 to 24 months kept at most one ratio once",
    ) as refused:
        fit_mack_grid(grid(SINGLE_FIRST), sigma_rule="mack", zero_cells="missing")
    assert "sigma_rule='log_linear'" in str(refused.value)


def _small_grid(rows) -> dict:
    return cohort_grid_frame(
        pd.DataFrame(
            {
                "origin_period": [ORIGINS[i] for i, r in enumerate(rows) for _ in r],
                "dev_lag": [12 * (j + 1) for r in rows for j in range(len(r))],
                "value": [v for r in rows for v in r],
            }
        ),
        dev_grain_months=12,
        measure="cumulative",
    )


def test_a_sigma_of_exactly_zero_stays_zero_and_out_of_the_regression():
    # 12 -> 24 keeps one link ratio (the other origins start at 0), 24 -> 36 has
    # four equal ratios (sigma exactly 0), 36 -> 48 and 48 -> 60 have positive
    # sigmas, and 60 -> 72 is the last step, with one link ratio
    rows = [
        [100.0, 200.0, 300.0, 330.0, 340.0, 345.0],
        [0.0, 150.0, 225.0, 250.0, 262.0],
        [0.0, 120.0, 180.0, 195.0],
        [0.0, 100.0, 150.0],
        [0.0, 80.0],
        [60.0],
    ]
    fit = fit_mack_grid(_small_grid(rows), sigma_rule="log_linear", zero_cells="missing")
    s = fit.sigma2
    assert list(fit.n_obs) == [1, 4, 3, 2, 1]
    assert s[1] == 0.0
    # the straight line through log sigma at the two positive steps, 2 and 3
    assert s[0] == pytest.approx(s[2] * (s[2] / s[3]) ** 2, rel=1e-12)
    assert s[4] == pytest.approx(s[3] * (s[3] / s[2]), rel=1e-12)
    assert np.isfinite(fit.msep_runoff()["msep_total"])


def test_a_sigma_gap_with_too_little_to_regress_on_is_refused_by_name():
    # three origins: the first step keeps one link ratio (the 12-month zero drops
    # the other), and the last step has one by construction, so no step at all
    # has an estimated sigma to fill the first one from
    rows = [[100.0, 150.0, 160.0], [0.0, 170.0], [130.0]]
    g = cohort_grid_frame(
        pd.DataFrame(
            {
                "origin_period": [ORIGINS[i] for i, r in enumerate(rows) for _ in r],
                "dev_lag": [12 * (j + 1) for r in rows for j in range(len(r))],
                "value": [v for r in rows for v in r],
            }
        ),
        dev_grain_months=12,
        measure="cumulative",
    )
    with pytest.raises(ValueError, match="needs at least two other ages with a positive sigma"):
        fit_mack_grid(g, sigma_rule="log_linear", zero_cells="missing")


@pytest.mark.tieout
@pytest.mark.parametrize(
    ("case", "rule", "cl_rule"),
    # not single_first under Mack's rule: chainladder leaves that sigma missing,
    # and ibnr refuses it by name (tested above)
    [
        ("single_first", "log_linear", "log-linear"),
        ("single_middle", "log_linear", "log-linear"),
        ("single_middle", "mack", "mack"),
    ],
)
def test_sigma_gaps_match_chainladder(cl, case, rule, cl_rule):
    rows = SIGMA_GAPS[case]
    tri = cl_triangle(cl, rows)
    development = cl.Development(sigma_interpolation=cl_rule).fit(tri)
    reference = cl.MackChainladder().fit(development.transform(tri))
    fit = fit_mack_grid(grid(rows), sigma_rule=rule, zero_cells="missing")
    np.testing.assert_allclose(
        np.sqrt(fit.sigma2), np.asarray(development.sigma_.values).ravel(), rtol=1e-9
    )
    assert np.sqrt(fit.msep_runoff()["msep_total"]) == pytest.approx(
        float(np.asarray(reference.total_mack_std_err_).ravel()[-1]), rel=1e-9
    )


#: history windows with zeros inside and at the edge of them
WINDOW_CASES = {
    "newest_edge": zeroed((7, 2)),  # 1988's latest cell, inside a window of 3 at 24 months
    "old_write_down": zeroed((1, 6)),
    "two_first_ages": zeroed((5, 0), (6, 0)),
    "interior": zeroed((6, 1)),
    "first_age": FIRST_AGE,
    "write_down": WRITE_DOWN,
}


def test_a_link_left_out_for_a_zero_keeps_its_place_in_the_window():
    rows = WINDOW_CASES["newest_edge"]
    result = methods.chain_ladder(cells(rows), history_periods=3)
    links = result.link_ratios.filter(pc.equal(result.link_ratios["from_dev_lag"], 24))
    reasons = dict(zip(column(links, "origin").tolist(), column(links, "reason"), strict=True))
    assert reasons[dt.date(1988, 1, 1)] == "zero_cell"
    assert reasons[dt.date(1987, 1, 1)] == reasons[dt.date(1986, 1, 1)] == "included"
    assert reasons[dt.date(1985, 1, 1)] == "history_window"  # not pulled in
    factor = result.development["factor"][1].as_py()
    assert factor == (rows[5][2] + rows[6][2]) / (rows[5][1] + rows[6][1])
    # a link left out for a zero OUTSIDE the window is reported for the zero:
    # 1982 is 0 at 84 months, and a window of 2 at 72 months holds 1983 and 1984
    result = methods.chain_ladder(cells(WINDOW_CASES["old_write_down"]), history_periods=2)
    links = result.link_ratios.filter(pc.equal(result.link_ratios["from_dev_lag"], 72))
    reasons = dict(zip(column(links, "origin").tolist(), column(links, "reason"), strict=True))
    assert reasons == {
        dt.date(1981, 1, 1): "history_window",
        dt.date(1982, 1, 1): "zero_cell",
        dt.date(1983, 1, 1): "included",
        dt.date(1984, 1, 1): "included",
    }
    # under "observed" an undefined ratio still gives up its place, as before
    rows = zeroed((7, 1))
    observed = methods.chain_ladder(cells(rows), history_periods=3, zero_cells="observed")
    links = observed.link_ratios.filter(pc.equal(observed.link_ratios["from_dev_lag"], 24))
    reasons = dict(zip(column(links, "origin").tolist(), column(links, "reason"), strict=True))
    assert reasons[dt.date(1988, 1, 1)] == "undefined_ratio"
    assert reasons[dt.date(1985, 1, 1)] == "included"


@pytest.mark.tieout
@pytest.mark.parametrize("case", list(WINDOW_CASES))
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("average", ["volume", "simple"])
def test_history_window_with_zeros_matches_chainladder(cl, case, n, average):
    rows = WINDOW_CASES[case]
    ours = methods.chain_ladder(
        cells(rows), history_periods=n, average=average, unsupported_factor="unity"
    )
    reference = cl.Development(n_periods=n, average=average).fit(cl_triangle(cl, rows))
    np.testing.assert_allclose(
        ours.development["factor"].to_pylist()[:-1],
        np.asarray(reference.ldf_.values).ravel()[:9],
        rtol=1e-12,
    )


def test_an_age_the_rule_empties_names_the_rule_and_the_ways_out():
    rows = zeroed((0, 9))  # 1981 at 120 months: the only link from 108 months ends at 0
    with pytest.raises(ValueError, match="no link ratio is left from 108 to 120 months") as refused:
        methods.chain_ladder(cells(rows))
    for part in ("zero_cells='missing'", "unsupported_factor='unity'", "zero_cells='observed'"):
        assert part in str(refused.value), part
    # under "observed" the link into the zero is used, its factor is 0, and the
    # refusal is the one it always was
    with pytest.raises(
        ValueError, match=r"from 108 to 120 months give a factor of 0.0, which is not"
    ):
        methods.chain_ladder(cells(rows), zero_cells="observed")


# -- 5. refusals ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["Missing", "zero", None, 0])
def test_an_unknown_setting_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="zero_cells must be 'observed' or 'missing'"):
        ConventionalCandidate(zero_cells=bad)
    if bad is None:
        # the Mack kernels read None as "observed", or as the rule their links carry
        assert fit_mack_grid(grid(), zero_cells=None).zero_cells == "observed"
    else:
        with pytest.raises(ValueError, match="zero_cells must be 'observed' or 'missing'"):
            fit_mack_grid(grid(), zero_cells=bad)
        with pytest.raises(ValueError, match="zero_cells must be 'observed' or 'missing'"):
            fit_mack(triangle(), loss_field="paid_loss", zero_cells=bad)
    for method, kwargs in METHODS.items():
        with pytest.raises(ValueError, match="zero_cells must be 'observed' or 'missing'"):
            getattr(methods, method)(cells(), zero_cells=bad, **kwargs)
    # a fit built by hand, as the codec builds one, is checked too
    with pytest.raises(ValueError, match="zero_cells must be 'observed' or 'missing'"):
        dataclasses.replace(fit_mack_grid(grid()), zero_cells=bad)


def test_fit_mack_many_refuses_a_bad_setting_before_any_cohort():
    # under on_error="skip" a per-cohort refusal would be recorded as every
    # cohort's error and the call would "succeed" with no fits
    with pytest.raises(ValueError, match="zero_cells must be 'observed' or 'missing'"):
        fit_mack_many(triangle(), loss_field="paid_loss", on_error="skip", zero_cells="zero")


def test_a_step_the_rule_leaves_empty_is_refused_by_name():
    # the only 24 -> 36 pair ends at a zero
    rows = [[100.0, 150.0, 0.0], [120.0, 170.0], [130.0]]
    g = cohort_grid_frame(
        pd.DataFrame(
            {
                "origin_period": [ORIGINS[i] for i, r in enumerate(rows) for _ in r],
                "dev_lag": [12 * (j + 1) for r in rows for j in range(len(r))],
                "value": [v for r in rows for v in r],
            }
        ),
        dev_grain_months=12,
        measure="cumulative",
    )
    fit_mack_grid(g)  # kept as data, the step has its one pair
    with pytest.raises(
        ValueError,
        match=r"every origin with a link ratio from 24 to 36 months has a zero cumulative",
    ):
        fit_mack_grid(g, zero_cells="missing")
    with pytest.raises(ValueError, match="no link ratio is left from 24 to 36 months"):
        fit_conventional_grid(g, ConventionalCandidate(zero_cells="missing"))
    unity = fit_conventional_grid(
        g, ConventionalCandidate(zero_cells="missing", unsupported_factor="unity")
    )
    assert unity.factors[1] == 1.0 and unity.factor_summary["unity_fallback"].iloc[1]


def test_leaving_a_pair_out_never_hides_a_negative_cumulative():
    # -5 at 24 months starts a pair that ends at a zero: "missing" would leave
    # that pair out, but the negative is refused first
    rows = [[100.0, -5.0, 0.0], [120.0, 170.0, 180.0], [130.0, 150.0], [140.0]]
    g = cohort_grid_frame(
        pd.DataFrame(
            {
                "origin_period": [ORIGINS[i] for i, r in enumerate(rows) for _ in r],
                "dev_lag": [12 * (j + 1) for r in rows for j in range(len(r))],
                "value": [v for r in rows for v in r],
            }
        ),
        dev_grain_months=12,
        measure="cumulative",
    )
    with pytest.raises(ValueError, match=r"negative cumulative loss in \(1981-01-01, 24 months\)"):
        fit_mack_grid(g, zero_cells="missing")


def _cdr_calls(fit):
    yield "one_year_cdr", lambda: one_year_cdr(fit)
    yield "simulate_one_year_cdr", lambda: simulate_one_year_cdr(fit, n_draws=10, seed=0)
    yield (
        "odp_bootstrap",
        lambda: simulate_one_year_cdr(fit, n_draws=10, seed=0, generator="odp_bootstrap"),
    )
    yield "rereserve", lambda: rereserve(fit, np.tile(fit.latest, (3, 1)))
    yield "MackDiagonal.check", lambda: MackDiagonal().check(fit)
    yield "ODPBootstrapDiagonal.check", lambda: ODPBootstrapDiagonal().check(fit)


#: the way out a refusal offers depends on its cause: refitting with "observed"
#: helps only when the rule left out link ratios, since "observed" refuses a zero
#: latest amount too
REFIT = "Refit with zero_cells='observed'"
POSITIVE = "needs every open origin's latest amount to be positive under either setting"


@pytest.mark.parametrize(
    ("rows", "found", "way_out", "not_offered"),
    [
        (FIRST_AGE, r"left out 1 link ratio\(s\)", REFIT, POSITIVE),
        (WRITE_DOWN, r"left out 3 link ratio\(s\)", REFIT, POSITIVE),
        (
            LATEST_NEWEST,
            r"kept a latest amount of zero for open origin\(s\) 1990-01-01",
            POSITIVE,
            REFIT,
        ),
    ],
    ids=["first_age", "write_down", "latest_newest"],
)
def test_the_one_year_result_refuses_a_fit_the_rule_changed(rows, found, way_out, not_offered):
    fit = fit_mack_grid(grid(rows), zero_cells="missing")
    answered = []
    for name, call in _cdr_calls(fit):
        try:
            call()
        except ValueError as exc:
            assert re.search(f"zero_cells='missing' that {found}", str(exc)), (name, str(exc))
            assert way_out in str(exc) and not_offered not in str(exc), (name, str(exc))
        else:
            answered.append(name)
    assert answered == []
    if way_out == REFIT:  # the way out it offers does lead somewhere
        one_year_cdr(fit_mack_grid(grid(rows), zero_cells="observed"))
    else:
        with pytest.raises(ValueError, match="non-positive cumulative on the latest diagonal"):
            one_year_cdr(fit_mack_grid(grid(rows), zero_cells="observed"))


class _NoDrawGenerator(DiagonalGenerator):
    """A generator with no checks of its own, which must never be asked to draw."""

    name = "no_draw"

    def check(self, fit):
        pass

    def draw(self, fit, *, n_draws, rng):
        raise AssertionError("drew a diagonal for a fit the one-year result refuses")


def test_the_simulated_route_refuses_before_any_draw():
    # a third-party generator carries no zero_cells check, so the refusal has
    # to come from simulate_one_year_cdr itself, before the draw
    fit = fit_mack_grid(grid(FIRST_AGE), zero_cells="missing")
    with pytest.raises(ValueError, match="zero_cells='missing' that left out 1 link"):
        simulate_one_year_cdr(fit, generator=_NoDrawGenerator(), n_draws=5)


def test_the_gallery_route_refuses_a_fit_the_rule_changed():
    from ibnr.gallery.cdr import GalleryDiagonal

    from .test_gallery_cdr import FULL, _cells, _MackEchoEntry

    values = FULL.copy()
    values[1, 0] = 0.0
    fit = fit_mack(
        make_cohort_triangle(None, values, start_year=2010),
        loss_field="paid_loss",
        as_of="2014-12-31",
        zero_cells="missing",
    )
    generator = GalleryDiagonal(_MackEchoEntry(fit, n_draws=10, seed=0), _cells())
    with pytest.raises(ValueError, match=r"zero_cells='missing' that left out 1 link"):
        generator.check(fit)
    with pytest.raises(ValueError, match=r"zero_cells='missing' that left out 1 link"):
        simulate_one_year_cdr(fit, generator=generator)


def test_the_one_year_result_accepts_a_missing_fit_with_no_zeros():
    observed = fit_mack_grid(grid(RAA))
    missing = fit_mack_grid(grid(RAA), zero_cells="missing")
    assert missing.zero_links == 0
    assert same_bytes(one_year_cdr(missing).msep, one_year_cdr(observed).msep)
    assert one_year_cdr(missing).msep_total == one_year_cdr(observed).msep_total
    assert same_bytes(
        simulate_one_year_cdr(missing, n_draws=300, seed=2).samples,
        simulate_one_year_cdr(observed, n_draws=300, seed=2).samples,
    )


def test_the_codec_carries_the_rule():
    fit = fit_mack_grid(grid(LATEST_NEWEST), zero_cells="missing")
    back = MackFit.from_arrow(fit.to_arrow())
    assert back.zero_cells == "missing"
    # a fit decoded as "observed" would refuse this, and the CDR would accept
    # a fit it has to refuse
    assert back.msep_runoff()["msep_total"] == fit.msep_runoff()["msep_total"]
    with pytest.raises(ValueError, match="zero_cells='missing'"):
        one_year_cdr(
            MackFit.from_arrow(fit_mack_grid(grid(FIRST_AGE), zero_cells="missing").to_arrow())
        )
    panel = fit_mack_many(triangle(FIRST_AGE), loss_field="paid_loss", zero_cells="missing")
    decoded = MackFitPanel.from_arrow(panel.to_arrow())
    assert decoded[()].zero_cells == "missing"
    assert decoded[()].zero_links == 1
    assert MackFit.from_arrow(fit_mack_grid(grid(RAA)).to_arrow()).zero_cells == "observed"
