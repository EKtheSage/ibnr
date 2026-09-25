"""Mack's chain ladder with development options (Mack 1999, the "generalized Mack").

Six groups of checks:

1. nothing that existed before moved: ``kernels.fit_mack_grid`` with neither
   ``average`` nor ``links`` has the same bytes as the code before this change
   (a frozen pin), and ``methods.mack``'s default numbers are within 1e-14;
2. R ChainLadder's ``MackChainLadder(alpha=, weights=, est.sigma=)`` on five
   public triangles under 27 settings each (frozen JSON; CI has no R);
3. the shared selection: ``methods.mack`` and ``methods.chain_ladder`` give the
   same factors and the same ``link_ratios`` table for the same options;
4. behaviour on hand triangles: the closed form against a transcription of R's
   recursion, the rule that future development keeps its full variance, the
   standard error of a factor kept from one ratio, the sigma fills, zero
   cells, the simulation, the one-year result's refusal, the codec;
5. every refusal, and every option delivered to the answer;
6. chainladder-python 0.9.2 (marker ``tieout``), where it has no defect in
   play, and with its two standard-error defects patched where it has, and the
   example workbook's numbers from the Schedule P mart (marker ``mart``).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ibnr import methods
from ibnr.errors import Refusal
from ibnr.kernels.cdr import (
    ODPBootstrapDiagonal,
    one_year_cdr,
    rereserve,
    simulate_one_year_cdr,
)
from ibnr.kernels.grid import grid_from_columns
from ibnr.kernels.links import LinkRules, is_all_history
from ibnr.kernels.mack import MackFit, draw_next_cells, fit_mack_grid, simulate_ultimates

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import freeze_mack_pin as frozen  # noqa: E402

PUBLIC = json.loads((DATA / "refusal_triangles.json").read_text("utf-8"))
PIN = json.loads((DATA / "mack_default_pin.json").read_text("utf-8"))
R_FIXTURE = json.loads((DATA / "r_mack_alpha_weights.json").read_text("utf-8"))
ALPHA_AVERAGE = {0: "simple", 1: "volume", 2: "regression"}


def cells_of(rows) -> pa.Table:
    """A triangle from [year, dev_lag, value] rows."""
    years, lags, values = zip(*rows, strict=True)
    return pa.table(
        {
            "origin_period": pa.array(years, pa.int64()),
            "dev_lag": pa.array(lags, pa.int64()),
            "value": pa.array(values, pa.float64()),
        }
    )


def rows_of(matrix, first_year: int = 2001, step: int = 12) -> list[list]:
    """[year, dev_lag, value] rows from an (origins, ages) matrix, NaN or None unobserved."""
    return [
        [first_year + i, step * (j + 1), float(v)]
        for i, row in enumerate(matrix)
        for j, v in enumerate(row)
        if v is not None and not np.isnan(v)
    ]


def grid_of(rows, step: int = 12) -> dict:
    years, lags, values = zip(*rows, strict=True)
    return grid_from_columns(
        np.array([dt.date(y, 1, 1) for y in years], dtype="datetime64[D]"),
        np.array(lags),
        np.array(values, dtype=float),
        dev_grain_months=step,
        measure="cumulative",
    )


def total(result, column: str = "mack_se") -> float:
    return result.totals[column][0].as_py()


def column(result, name: str, table: str = "origins") -> np.ndarray:
    return np.array(getattr(result, table)[name].to_pylist(), dtype=float)


RAA = cells_of(PUBLIC["raa"])
GENINS = cells_of(PUBLIC["genins"])
#: the settings the web sends with Mack, and a few more
SETTINGS = [
    {},
    {"average": "simple"},
    {"average": "regression"},
    {"history_periods": 5},
    {"history_periods": 3, "average": "regression"},
    {"drop_high": 1},
    {"drop_low": 1},
    {"drop_high": 2, "drop_low": 1},
    {"history_periods": 5, "drop_high": 1, "average": "regression"},
    {"drop_above": 3.0},
    {"drop_below": 1.05, "average": "simple"},
]


# -- 1. nothing that existed before moved --------------------------------------------


def test_the_default_kernel_fit_did_not_move_on_the_public_triangles():
    """``fit_mack_grid`` with no ``average`` and no ``links``, under both sigma
    rules and both zero rules, on the five public triangles, the same five with a
    zero cell, and a 30 x 30 one: every array of the fit, ``msep_runoff`` and the
    ``to_arrow`` payload have the bytes they had before this change."""
    triangles = {**frozen.public_triangles(), **frozen.with_zeros()}
    now = frozen.pin(triangles)
    pinned = {k: v for k, v in PIN["kernel"].items() if not k.startswith("clrd|")}
    assert len(pinned) == 11 * 4
    assert [key for key in sorted(pinned) if now["kernel"].get(key) != pinned[key]] == []


@pytest.mark.tieout
def test_the_default_kernel_fit_did_not_move_on_clrd():
    pytest.importorskip("chainladder")
    now = frozen.pin(frozen.clrd_triangles())
    pinned = {k: v for k, v in PIN["kernel"].items() if k.startswith("clrd|")}
    assert len(pinned) == 53 * 4
    assert [key for key in sorted(pinned) if now["kernel"].get(key) != pinned[key]] == []


def _same_numbers(now: dict, before: dict) -> list[str]:
    """The keys whose numbers differ by more than 1e-14 relative, or whose refusal moved."""
    moved = []
    for key, old in before.items():
        new = now[key]
        if "refused" in old or "refused" in new:
            if old != new:
                moved.append(key)
            continue
        for name, values in old.items():
            a = np.array(new[name], dtype=float)
            b = np.array(values, dtype=float)
            if not np.allclose(a, b, rtol=1e-14, atol=0.0, equal_nan=True):
                moved.append(f"{key}|{name}")
    return moved


def test_methods_mack_defaults_move_by_rounding_at_most():
    """``methods.mack`` now reads its factors through the shared selection, whose
    sums add in another order: its default numbers (and with ``sigma_rule="mack"``,
    and ``zero_cells="observed"``) are within 1e-14 of the code before, and a
    refusal is still a refusal with the same reason."""
    triangles = {**frozen.public_triangles(), **frozen.with_zeros()}
    now = frozen.pin(triangles)["methods"]
    before = {k: v for k, v in PIN["methods"].items() if not k.startswith("clrd|")}
    assert _same_numbers(now, before) == []


@pytest.mark.tieout
def test_methods_mack_defaults_move_by_rounding_at_most_on_clrd():
    pytest.importorskip("chainladder")
    now = frozen.pin(frozen.clrd_triangles())["methods"]
    before = {k: v for k, v in PIN["methods"].items() if k.startswith("clrd|")}
    assert len(before) == 53 * 3
    assert _same_numbers(now, before) == []


def test_links_with_no_option_is_the_same_fit_as_none():
    """``links=LinkRules()`` is the development options path with every ratio
    kept: on a triangle with no zero it gives 0.7.2's numbers, to rounding."""
    grid = grid_of(PUBLIC["raa"])
    plain = fit_mack_grid(grid, sigma_rule="log_linear")
    selected = fit_mack_grid(grid, sigma_rule="log_linear", links=LinkRules())
    assert plain.links is None and selected.links == LinkRules()
    np.testing.assert_allclose(selected.f, plain.f, rtol=1e-15)
    np.testing.assert_allclose(selected.sigma2, plain.sigma2, rtol=1e-13)
    np.testing.assert_array_equal(selected.n_obs, plain.n_obs)
    assert selected.msep_runoff()["msep_total"] == pytest.approx(
        plain.msep_runoff()["msep_total"], rel=1e-13
    )


# -- 2. R ChainLadder's MackChainLadder(alpha, weights) -------------------------------


def _r_cases():
    cases = []
    for fit in R_FIXTURE["fits"]:
        if "error" in fit or not fit["finite"]:
            continue
        if fit["est_sigma"] == "log-linear" and fit["switched"]:
            continue  # R used Mack's rule under the log-linear name
        name = (
            f"{fit['dataset']}|alpha{fit['alpha']}|h{fit['history_periods']}|"
            f"x{int(fit['exclude_second_origin_first_link'])}|{fit['trim']}|{fit['est_sigma']}"
        )
        cases.append(pytest.param(fit, id=name))
    return cases


R_CASES = _r_cases()


def _r_options(fit: dict) -> dict:
    options = {
        "average": ALPHA_AVERAGE[fit["alpha"]],
        "sigma_rule": "mack" if fit["est_sigma"] == "Mack" else "log_linear",
        "history_periods": fit["history_periods"],
    }
    if fit["exclude_second_origin_first_link"]:
        options["exclude"] = [(2002, 12)]
    if fit["trim"] == "high":
        options["drop_high"] = 1
    elif fit["trim"] == "low":
        options["drop_low"] = 1
    return options


def test_the_r_fixture_covers_what_it_says():
    """Every fit R answered is here: 224 of 270, the rest recorded as R's
    infinite standard errors (30) or its switch to Mack's rule (16)."""
    fits = R_FIXTURE["fits"]
    assert R_FIXTURE["source"].startswith("R 4.5.3, ChainLadder 0.2.21")
    assert len(fits) == 5 * 27 * 2
    assert len(R_CASES) == 224
    assert sum(not f["finite"] for f in fits) == 30
    assert {f["alpha"] for f in fits} == {0, 1, 2}


@pytest.mark.parametrize("fit", R_CASES)
def test_mack_ties_out_to_r(fit):
    """``methods.mack`` with the same options chooses the link ratios R was given
    as weights, and gives R's factors, sigmas, factor standard errors and
    standard errors (total, process and parameter, per origin and in total)."""
    matrix = [
        [np.nan if v is None else v for v in row] for row in R_FIXTURE["triangles"][fit["dataset"]]
    ]
    result = methods.mack(cells_of(rows_of(matrix)), **_r_options(fit))
    # the same link ratios
    ratios = result.link_ratios
    chosen = {
        (o.year - 2001, lag // 12 - 1): used
        for o, lag, used in zip(
            ratios["origin_period"].to_pylist(),
            ratios["from_dev_lag"].to_pylist(),
            ratios["included"].to_pylist(),
            strict=True,
        )
    }
    weights = {
        (i, j): row[j] == "1"
        for i, row in enumerate(fit["weights"])
        for j in range(len(row) - 1)
        if (i, j) in chosen
    }
    assert chosen == weights
    # the same numbers
    np.testing.assert_allclose(column(result, "factor", "development")[:-1], fit["f"], rtol=1e-13)
    # A sigma of 1e-4 on link ratios near 1 is a difference of nearly equal
    # numbers, which R's lm and this sum round differently: at most 3e-14 apart on
    # mw2014, so the sigmas are compared to 1e-12 of the largest as well.
    for name, key in (("sigma", "sigma"), ("std_err", "f_se")):
        np.testing.assert_allclose(
            column(result, name, "development")[:-1],
            fit[key],
            rtol=1e-10,
            atol=1e-12 * max(fit[key]),
            err_msg=name,
        )
    for name, key in (
        ("mack_se", "mack_se"),
        ("process_se", "process_se"),
        ("parameter_se", "parameter_se"),
    ):
        # the oldest open origins run off on those near-zero sigmas alone
        np.testing.assert_allclose(
            column(result, name), fit[key], rtol=1e-10, atol=1e-12 * max(fit[key]), err_msg=name
        )
    assert total(result) == pytest.approx(fit["total_mack_se"], rel=1e-12)
    assert total(result, "process_se") == pytest.approx(fit["total_process_se"], rel=1e-12)
    assert total(result, "parameter_se") == pytest.approx(fit["total_parameter_se"], rel=1e-12)


# -- 3. the shared selection -----------------------------------------------------------

SHARED = [
    {},
    {"average": "simple"},
    {"average": "regression"},
    {"history_periods": 4},
    {"drop_high": 1, "drop_low": 1},
    {"drop_high": 2, "preserve": 2, "average": "regression"},
    {"drop_above": 2.5, "drop_below": 1.02},
    {"exclude_valuations": ["1995-12-31"], "history_periods": 6},
    {"trim_ties": "origin", "drop_low": 1},
    # preserve stops the trims at most ages and the bound at the first (on raa)
    {"drop_high": 2, "drop_low": 2, "preserve": 3, "drop_above": 1.5},
]


@pytest.mark.parametrize("options", SHARED, ids=[str(o) for o in SHARED])
@pytest.mark.parametrize("name", sorted({**frozen.public_triangles(), **frozen.with_zeros()}))
def test_mack_and_the_chain_ladder_select_and_average_alike(name, options):
    """Under ``zero_cells="missing"`` the same options give ``methods.mack`` the
    chain ladder's factors bit for bit and the identical ``link_ratios`` table."""
    rows = {**frozen.public_triangles(), **frozen.with_zeros()}[name]
    cells = cells_of(rows)
    options = dict(options)
    if "exclude_valuations" in options:
        latest = max(y + d // 12 - 1 for y, d, _ in rows)
        options["exclude_valuations"] = [latest - 2]
    try:
        point = methods.chain_ladder(cells, **options)
    except Refusal:
        return  # the chain ladder refuses, so there is nothing to share
    try:
        mack = methods.mack(cells, **options)
    except Refusal as refusal:
        # only Mack's own refusals: no sigma, or a zero latest under regression
        assert refusal.reason in ("variance_not_estimable", "not_supported"), refusal
        return
    assert mack.development["factor"].equals(point.development["factor"])
    assert mack.link_ratios.equals(point.link_ratios)
    for name in ("n_selected", "extreme_trimming_skipped", "bounds_skipped"):
        assert mack.development[name].equals(point.development[name]), name
    np.testing.assert_allclose(column(mack, "ultimate"), column(point, "ultimate"), rtol=1e-12)


def test_mack_says_where_preserve_stopped_a_trim_and_a_bound():
    """On raa, preserve=3 stops drop_high=2 and drop_low=2 at every age from 24
    months on, and stops drop_above=1.5 at 12 months, where it would leave too few
    ratios. Mack reports both, as the chain ladder does."""
    result = methods.mack(RAA, drop_high=2, drop_low=2, preserve=3, drop_above=1.5)
    development = result.development
    assert development["extreme_trimming_skipped"].to_pylist() == [False] + [True] * 8 + [None]
    assert development["bounds_skipped"].to_pylist() == [True] + [False] * 8 + [None]


# -- 4. behaviour ------------------------------------------------------------------------


def _r_recursion(fit: MackFit) -> tuple[np.ndarray, float]:
    """Mack's standard errors by R's ``MackRecursive.S.E`` and ``TotalMack.S.E``,
    transcribed, from a fit's factors, sigmas and completed triangle."""
    f, full, alpha = fit.f, fit.full, fit.alpha
    sigma = np.sqrt(fit.sigma2)
    f_se = np.sqrt(fit.sigma2 / fit.s)
    obs = fit.obs_mask
    n_w, n_d = full.shape
    proc = np.zeros((n_w, n_d))
    par = np.zeros((n_w, n_d))
    step_se = sigma[None, :] / np.sqrt(full[:, :-1] ** alpha)
    for k in range(n_d - 1):
        for i in range(n_w):
            if obs[i, k + 1]:
                continue
            proc[i, k + 1] = np.sqrt(
                full[i, k] ** 2 * step_se[i, k] ** 2 + proc[i, k] ** 2 * f[k] ** 2
            )
            par[i, k + 1] = np.sqrt(full[i, k] ** 2 * f_se[k] ** 2 + par[i, k] ** 2 * f[k] ** 2)
    total_par = np.zeros(n_d)
    for k in range(n_d - 1):
        m = full[~obs[:, k + 1], k].sum()
        total_par[k + 1] = np.sqrt(m**2 * f_se[k] ** 2 + total_par[k] ** 2 * f[k] ** 2)
    total_proc = np.sqrt((proc[:, -1] ** 2).sum())
    return np.sqrt(proc[:, -1] ** 2 + par[:, -1] ** 2), float(np.hypot(total_proc, total_par[-1]))


@pytest.mark.parametrize("average", ["simple", "volume", "regression"])
@pytest.mark.parametrize(
    "rules",
    [
        LinkRules(),
        LinkRules(history_periods=4, drop_high=1, exhausted_exclusions="keep"),
        LinkRules(drop_low=1, exclude=((dt.date(2003, 1, 1), 24),), exhausted_exclusions="keep"),
    ],
    ids=["all", "window_and_trim", "exclusion_and_trim"],
)
def test_the_closed_form_is_rs_recursion(average, rules):
    fit = fit_mack_grid(grid_of(PUBLIC["genins"]), average=average, links=rules)
    per_origin, total_se = _r_recursion(fit)
    risk = fit.msep_runoff()
    np.testing.assert_allclose(np.sqrt(risk["msep"]), per_origin, rtol=1e-12)
    assert np.sqrt(risk["msep_total"]) == pytest.approx(total_se, rel=1e-12)


@pytest.mark.parametrize(
    "options",
    [
        {"drop_above": 1e9},
        {"drop_below": 0.0},
        {"history_periods": 50},
        {"drop_high": 1, "preserve": 20},
        {"drop_high": False, "drop_low": 0},
    ],
)
def test_options_that_remove_nothing_leave_the_standard_errors_alone(options):
    """Future development keeps its full variance whatever the options: an option
    that leaves out no link ratio gives the default standard errors exactly.
    (chainladder-python halves RAA's total with drop_above=100: 13,037.79.)"""
    base = methods.mack(RAA)
    other = methods.mack(RAA, **options)
    assert other.origins.equals(base.origins)
    assert other.totals.equals(base.totals)
    assert total(other) == pytest.approx(26_880.74, abs=0.005)


def test_a_factor_kept_from_one_ratio_has_that_ratios_standard_error():
    """RAA with ``drop_high=2``: at 84 to 96 months only 1983's link ratio is
    left. The factor's standard error is sigma over the square root of 1983's
    own amount there (0.01901), not the first origin's (chainladder: 0.02142)."""
    result = methods.mack(RAA, drop_high=2)
    ratios = result.link_ratios.filter(pc.equal(result.link_ratios["from_dev_lag"], 84))
    kept = ratios.filter(ratios["included"])
    assert kept["origin"].to_pylist() == [1983]
    sigma = column(result, "sigma", "development")[6]
    amount = kept["previous"][0].as_py()
    std_err = column(result, "std_err", "development")[6]
    assert std_err == pytest.approx(sigma / np.sqrt(amount), rel=1e-15)
    assert std_err == pytest.approx(0.01901, abs=5e-6)
    # and at every alpha: sigma / sqrt(amount ** alpha)
    for average, alpha in (("simple", 0), ("regression", 2)):
        result = methods.mack(RAA, drop_high=2, average=average)
        sigma = column(result, "sigma", "development")[6]
        std_err = column(result, "std_err", "development")[6]
        assert std_err == pytest.approx(sigma / np.sqrt(amount**alpha), rel=1e-14)


#: 8 origins; drop_high=1 leaves the 60 to 72 month link with one ratio
GAP = [
    [100.0, 180.0, 230.0, 260.0, 275.0, 282.0, 286.0, 288.0],
    [110.0, 190.0, 245.0, 272.0, 290.0, 300.0, 303.0],
    [120.0, 222.0, 270.0, 300.0, 318.0, 325.0],
    [105.0, 185.0, 240.0, 268.0, 281.0],
    [130.0, 230.0, 300.0, 330.0],
    [125.0, 228.0, 290.0],
    [140.0, 250.0],
    [150.0],
]


def test_a_middle_gap_is_filled_from_both_sides_or_from_the_two_before():
    rules = LinkRules(drop_high=1, exhausted_exclusions="keep")
    grid = grid_of(rows_of(GAP))
    log_linear = fit_mack_grid(grid, sigma_rule="log_linear", links=rules)
    gaps = np.flatnonzero(log_linear.n_pos < 2)
    assert gaps.tolist() == [5, 6]  # 72 to 84 (two ratios, one dropped) and the last
    estimated = np.flatnonzero(log_linear.n_pos >= 2)
    slope, intercept = np.polyfit(estimated, np.log(np.sqrt(log_linear.sigma2[estimated])), 1)
    np.testing.assert_allclose(
        log_linear.sigma2[gaps], np.exp(intercept + slope * gaps) ** 2, rtol=1e-12
    )
    mack_rule = fit_mack_grid(grid, sigma_rule="mack", links=rules)
    s2 = mack_rule.sigma2
    # in link order, the second gap from the first gap's filled value, as R does
    assert s2[5] == min(s2[4] ** 2 / s2[3], s2[4], s2[3])
    assert s2[6] == min(s2[5] ** 2 / s2[4], s2[5], s2[4])
    # the estimated sigmas are the same under either rule
    np.testing.assert_array_equal(s2[estimated], log_linear.sigma2[estimated])


def test_a_gap_with_estimates_after_it_is_filled_from_both_sides():
    """RAA with six of the seven link ratios from 36 to 48 months excluded: that
    age keeps one, and the ages after it keep two to six. The log-linear fill
    regresses on the ages on both sides, not only the ones before."""
    rules = LinkRules(exclude=tuple((dt.date(y, 1, 1), 36) for y in range(1982, 1988)))
    fit = fit_mack_grid(grid_of(PUBLIC["raa"]), sigma_rule="log_linear", links=rules)
    assert np.flatnonzero(fit.n_pos < 2).tolist() == [2, 8]
    estimated = np.flatnonzero(fit.n_pos >= 2)
    assert estimated.tolist() == [0, 1, 3, 4, 5, 6, 7]
    slope, intercept = np.polyfit(estimated, np.log(np.sqrt(fit.sigma2[estimated])), 1)
    np.testing.assert_allclose(
        fit.sigma2[[2, 8]], np.exp(intercept + slope * np.array([2, 8])) ** 2, rtol=1e-12
    )
    # the regression over the two ages before alone would give another number
    before, first = np.polyfit([0, 1], np.log(np.sqrt(fit.sigma2[[0, 1]])), 1)
    assert fit.sigma2[2] != pytest.approx(np.exp(first + before * 2) ** 2, rel=1e-3)


#: drop_high=1 leaves 1.25 and 1.25 from 36 to 48 months (sigma exactly 0), one ratio
#: from 48 to 60, and the one ratio at the last age kept
ZERO_SIGMA = [
    [100.0, 180.0, 400.0, 500.0, 525.0, 540.0],
    [110.0, 230.0, 480.0, 600.0, 650.0],
    [90.0, 200.0, 320.0, 480.0],
    [120.0, 250.0, 450.0],
    [105.0, 260.0],
    [115.0],
]


def test_a_sigma_of_zero_stays_zero_and_is_left_out_of_the_log_linear_fill():
    """The two ratios kept from 36 to 48 months are equal, so sigma there is 0: an
    estimate, not an underflow, and a value with no logarithm. The log-linear fill
    at 48 and 60 months regresses on the two positive sigmas only (R's rule), and
    methods.mack answers rather than refusing the 0 as amounts too small."""
    result = methods.mack(cells_of(rows_of(ZERO_SIGMA)), drop_high=1)
    sigma = column(result, "sigma", "development")[:-1]
    links = result.link_ratios.filter(result.link_ratios["included"]).to_pylist()
    by_hand = []
    for lag in (12, 24):
        kept = [(r["previous"], r["ratio"]) for r in links if r["from_dev_lag"] == lag]
        previous, ratios = np.array(kept).T
        factor = (previous * ratios).sum() / previous.sum()
        by_hand.append(np.sqrt((previous * (ratios - factor) ** 2).sum() / (len(kept) - 1)))
    np.testing.assert_allclose(sigma[:2], by_hand, rtol=1e-12)
    assert sigma[2] == 0.0
    # a line through log sigma at ages 0 and 1, read off at ages 3 and 4
    slope = np.log(by_hand[1]) - np.log(by_hand[0])
    np.testing.assert_allclose(
        sigma[3:], np.exp(np.log(by_hand[0]) + slope * np.array([3, 4])), rtol=1e-12
    )
    assert result.development["sigma_extrapolated"].to_pylist()[:-1] == [False] * 3 + [True] * 2
    assert np.isfinite(total(result)) and total(result) > 0


def test_a_gap_the_rule_cannot_fill_is_refused_naming_the_option():
    # three origins: drop_high=1 leaves the first link with one ratio, and Mack's
    # rule has no two links before it; the regression has only one estimate
    rows = rows_of([[100.0, 150.0, 160.0, 161.0], [110.0, 170.0, 180.0], [120.0, 175.0], [130.0]])
    rules = LinkRules(drop_high=1, exhausted_exclusions="keep", preserve=1)
    with pytest.raises(Refusal, match=r"Mack's rule fills that sigma from the two ages") as refused:
        fit_mack_grid(
            grid_of(rows),
            sigma_rule="mack",
            links=LinkRules(drop_low=2, exhausted_exclusions="keep"),
        )
    assert refused.value.reason == "variance_not_estimable"
    with pytest.raises(Refusal, match=r"the development options \(drop_high\)") as refused:
        fit_mack_grid(grid_of(rows), sigma_rule="log_linear", links=rules)
    assert refused.value.options == ("sigma_rule", "drop_high")


@pytest.mark.parametrize("options", SETTINGS, ids=[str(o) for o in SETTINGS])
def test_sigma_extrapolated_marks_exactly_the_ages_left_with_one_ratio(options):
    result = methods.mack(GENINS, **options)
    development = result.development
    n_selected = development["n_selected"].to_pylist()[:-1]
    extrapolated = development["sigma_extrapolated"].to_pylist()
    assert extrapolated[:-1] == [n < 2 for n in n_selected]
    assert extrapolated[-1] is None


def test_a_zero_latest_amount_under_each_average():
    """Under ``zero_cells="missing"`` an origin whose latest amount is 0 has
    ultimate 0 and standard error 0 under the volume and simple averages, the
    limit of the formula; under regression the variance does not shrink with the
    amount, so it is refused."""
    rows = [[y, d, 0.0 if (y, d) == (1990, 12) else v] for y, d, v in PUBLIC["raa"]]
    for average in ("volume", "simple"):
        result = methods.mack(cells_of(rows), average=average)
        assert result.origins["ultimate"][-1].as_py() == 0.0
        assert result.origins["mack_se"][-1].as_py() == 0.0
        assert np.isfinite(total(result))
    with pytest.raises(Refusal, match=r"under average='regression' Mack's variance") as refused:
        methods.mack(cells_of(rows), average="regression")
    assert refused.value.reason == "not_supported"
    assert [(c.origin, c.dev_lag) for c in refused.value.cells] == [(1990, 12)]


def test_observed_zeros_with_options_are_refused_and_without_keep_0_7_2():
    rows = [[y, d, 0.0 if (y, d) == (1988, 12) else v] for y, d, v in PUBLIC["raa"]]
    plain = methods.mack(cells_of(rows), zero_cells="observed")
    fit = fit_mack_grid(grid_of(rows))
    assert (fit.n_obs != fit.n_pos).any()  # R's reading: the link out of the zero counts
    assert plain.link_ratios["included"].to_pylist() == [True] * 45
    # n_selected counts the included ratios at each age, the one out of the zero too
    lags = plain.link_ratios["from_dev_lag"].to_pylist()
    counts = [lags.count(12 * (j + 1)) for j in range(9)]
    assert counts[0] == 9
    assert plain.development["n_selected"].to_pylist() == [*counts, None]
    for options in ({"average": "simple"}, {"history_periods": 50}, {"drop_high": 1}):
        with pytest.raises(Refusal, match=r"the link ratio out of the zero at") as refused:
            methods.mack(cells_of(rows), zero_cells="observed", **options)
        assert refused.value.reason == "not_supported"
        assert refused.value.option == "zero_cells"
        assert [(c.origin, c.dev_lag) for c in refused.value.cells] == [(1988, 12)]


@pytest.mark.parametrize(("average", "rel"), [("simple", 0.03), ("regression", 0.02)])
def test_the_simulation_matches_the_formula_at_every_alpha(average, rel):
    """``simulate_ultimates`` draws each step with variance sigma^2 * C^(2 - alpha):
    over 200,000 draws on genins, the spread of the total is Mack's standard
    error. (At alpha 0 the formula replaces E[C^2] by C-hat^2, so the two differ
    by a little more.)"""
    fit = fit_mack_grid(grid_of(PUBLIC["genins"]), average=average)
    pred = simulate_ultimates(fit, n_draws=200_000, seed=7, process="normal")
    assert pred.samples[:, -1].std() == pytest.approx(
        np.sqrt(fit.msep_runoff()["msep_total"]), rel=rel
    )


@pytest.mark.parametrize("average", ["simple", "volume", "regression"])
def test_next_cell_draws_carry_the_alphas_variance(average):
    """``draw_next_cells`` draws one step with variance sigma^2 * prev^(2 - alpha)."""
    fit = fit_mack_grid(grid_of(PUBLIC["genins"]), average=average)
    prev = np.array([2_000_000.0, 3_500_000.0])
    draws = draw_next_cells(
        fit,
        _cells(np.array([2, 3]), prev),
        rng=np.random.default_rng(3),
        n_draws=200_000,
        process="normal",
        parameter_risk=False,
    )
    expected = fit.sigma2[[0, 1]] * prev ** (2 - fit.alpha)
    np.testing.assert_allclose(draws.var(axis=0), expected, rtol=0.02)


def _cells(d: np.ndarray, prev: np.ndarray):
    """The two fields of a ``CellIndex`` that ``draw_next_cells`` reads."""
    return type("Cells", (), {"d": d, "prev_value": prev})()


def test_a_quarterly_triangle_takes_a_window_and_the_one_year_result_still_refuses_the_grain():
    rows = [[y, 3 * d // 12, v] for y, d, v in PUBLIC["raa"]]
    quarterly = [[f"{2001 + (y - 1981) // 4}Q{(y - 1981) % 4 + 1}", d, v] for y, d, v in rows]
    cells = pa.table(
        {
            "origin_period": [r[0] for r in quarterly],
            "dev_lag": [r[1] for r in quarterly],
            "value": [r[2] for r in quarterly],
        }
    )
    result = methods.mack(cells, dev_grain_months=3, history_periods=5)
    assert result.dev_grain_months == 3 and np.isfinite(total(result))
    assert total(result) != total(methods.mack(cells, dev_grain_months=3))
    fit = fit_mack_grid(
        grid_from_columns(
            np.array(
                [dt.date(2001 + (y - 1981) // 4, 3 * ((y - 1981) % 4) + 1, 1) for y, _, _ in rows],
                dtype="datetime64[D]",
            ),
            np.array([d for _, d, _ in rows]),
            np.array([v for _, _, v in rows]),
            dev_grain_months=3,
            measure="cumulative",
        ),
        links=LinkRules(history_periods=5),
    )
    with pytest.raises(Refusal, match="needs an annual development grain"):
        one_year_cdr(fit)


# -- the one-year result -----------------------------------------------------------------

CDR_REFUSED = [
    ({"average": "simple"}, None, "average='simple'"),
    (
        {"average": "regression"},
        LinkRules(history_periods=5),
        "average='regression' and history_periods=5",
    ),
    ({}, LinkRules(history_periods=50), "history_periods=50"),
    ({}, LinkRules(drop_high=1, exhausted_exclusions="keep"), "drop_high=1"),
    ({}, LinkRules(drop_low=2, exhausted_exclusions="keep"), "drop_low=2"),
    ({}, LinkRules(drop_above=100.0), "drop_above=100.0"),
    ({}, LinkRules(drop_below=0.5), "drop_below=0.5"),
    ({}, LinkRules(exclude=((dt.date(2002, 1, 1), 12),)), r"exclude \(1 link ratio\)"),
    (
        {},
        LinkRules(exclude_valuations=(dt.date(2005, 12, 31),)),
        r"exclude_valuations \(1 valuation\)",
    ),
]


@pytest.mark.parametrize(
    ("average", "links", "named"), CDR_REFUSED, ids=[c[2] for c in CDR_REFUSED]
)
def test_the_one_year_result_refuses_every_development_option_by_its_settings(
    average, links, named
):
    """Every route to a one-year result refuses a fit with an option set, even one
    that removed nothing from this triangle (``history_periods=50`` on 10 x 10,
    ``drop_above=100``, ``drop_below=0.5``): ``rereserve`` re-runs the volume
    chain ladder over every ratio on NEXT year's triangle."""
    fit = fit_mack_grid(grid_of(PUBLIC["genins"]), links=links, **average)
    assert not fit.all_history_volume
    match = f"This fit used {named}\\."
    calls = [
        lambda: one_year_cdr(fit),
        lambda: simulate_one_year_cdr(fit, n_draws=10, seed=0, generator="mack"),
        lambda: simulate_one_year_cdr(fit, n_draws=10, seed=0, generator="odp_bootstrap"),
        lambda: simulate_one_year_cdr(fit, n_draws=10, seed=0, generator=ODPBootstrapDiagonal()),
        lambda: rereserve(fit, np.tile(fit.latest, (3, 1))),
    ]
    for call in calls:
        with pytest.raises(Refusal, match=match) as refused:
            call()
        assert refused.value.reason == "not_supported"


def test_links_with_no_option_gives_the_one_year_result():
    grid = grid_of(PUBLIC["genins"])
    plain = one_year_cdr(fit_mack_grid(grid))
    selected = one_year_cdr(fit_mack_grid(grid, links=LinkRules()))
    assert selected.msep_total == pytest.approx(plain.msep_total, rel=1e-12)
    assert is_all_history(LinkRules(preserve=3, trim_ties="volume", exhausted_exclusions="keep"))


# -- the codec -----------------------------------------------------------------------------


def test_the_codec_carries_the_options_and_the_selection():
    rules = LinkRules(
        history_periods=6,
        exclude=((dt.date(2002, 1, 1), 12),),
        exclude_valuations=(dt.date(2008, 12, 31),),
        drop_high=1,
        drop_above=4.5,
        preserve=2,
        trim_ties="volume",
        exhausted_exclusions="keep",
        zero_cells="missing",
    )
    fit = fit_mack_grid(grid_of(PUBLIC["genins"]), average="regression", links=rules)
    back = MackFit.from_arrow(fit.to_arrow())
    assert back.average == "regression" and back.links == rules
    for name in ("previous", "following", "ratio", "observed", "used", "reason"):
        a, b = getattr(back.selection, name), getattr(fit.selection, name)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name
    assert back.msep_runoff()["msep_total"] == fit.msep_runoff()["msep_total"]
    # decoded, it is refused by the one-year result exactly as the original
    with pytest.raises(Refusal, match="This fit used average='regression'"):
        one_year_cdr(back)


def test_a_fit_with_options_writes_version_2_and_a_fit_without_writes_1():
    import pyarrow.ipc as ipc

    def version(data: bytes) -> bytes:
        return ipc.open_stream(data).schema.metadata[b"ibnr.version"]

    grid = grid_of(PUBLIC["raa"])
    assert version(fit_mack_grid(grid).to_arrow()) == b"1"
    assert version(fit_mack_grid(grid, average="simple").to_arrow()) == b"2"


def test_a_tampered_average_or_selection_is_refused():
    fit = fit_mack_grid(grid_of(PUBLIC["raa"]), average="simple")
    with pytest.raises(Refusal, match="average must be 'volume', 'simple' or 'regression'"):
        dataclasses.replace(fit, average="cubic")
    with pytest.raises(Refusal, match="carries its link rules and the selection they made"):
        dataclasses.replace(fit, selection=None)
    with pytest.raises(Refusal, match="must carry the link rules it was fitted with"):
        dataclasses.replace(fit, links=None, selection=None)
    with pytest.raises(Refusal, match="must be its link rules'"):
        dataclasses.replace(fit, zero_cells="missing")


# -- 5. refusals ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("average", "reason", "phrase"),
    [
        ("median", "not_supported", "a median average is not one"),
        ("geometric", "not_supported", "a geometric average is not one"),
        ("cubic", "invalid_option", "average must be 'volume', 'simple' or 'regression'"),
        (1, "invalid_option", "average must be 'volume', 'simple' or 'regression'"),
        (None, "invalid_option", "average must be 'volume', 'simple' or 'regression'"),
    ],
)
def test_an_average_mack_has_no_variance_for_is_refused_by_name(average, reason, phrase):
    for call in (
        lambda: methods.mack(RAA, average=average),
        lambda: fit_mack_grid(grid_of(PUBLIC["raa"]), average=average),
    ):
        with pytest.raises(Refusal, match=phrase) as refused:
            call()
        assert refused.value.reason == reason and refused.value.option == "average"
    if average == "median":
        # the chain ladder answers it, and the message says so
        assert "The chain ladder with average='median'" in str(refused.value)
        methods.chain_ladder(RAA, average="median")


def test_history_periods_1_is_refused_by_name():
    for call in (
        lambda: methods.mack(RAA, history_periods=1),
        lambda: fit_mack_grid(grid_of(PUBLIC["raa"]), links=LinkRules(history_periods=1)),
    ):
        with pytest.raises(Refusal, match=r"history_periods=1 keeps one link ratio") as refused:
            call()
        assert refused.value.reason == "variance_not_estimable"
        assert refused.value.option == "history_periods"
    # 2 is answered: every age but the last keeps two
    assert np.isfinite(total(methods.mack(RAA, history_periods=2)))


def test_an_age_the_options_empty_is_refused_not_given_a_unit_factor():
    """RAA's only link ratio from 108 to 120 months is 1981's: excluding it leaves
    that age with none. The chain ladder can use 1.0 there when asked; Mack
    refuses, because a chosen factor has no variance."""
    with pytest.raises(Refusal, match=r"leave no link ratio from 108 to 120 months") as refused:
        methods.mack(RAA, exclude=[(1981, 108)])
    assert refused.value.reason == "no_link_ratio" and refused.value.option == "exclude"
    assert "unsupported_factor='unity'" in str(refused.value)
    point = methods.chain_ladder(RAA, exclude=[(1981, 108)], unsupported_factor="unity")
    assert point.development["factor"][8].as_py() == 1.0
    with pytest.raises(TypeError, match="unsupported_factor"):
        methods.mack(RAA, unsupported_factor="unity")


def test_zero_cells_given_twice_is_refused_and_links_must_be_link_rules():
    grid = grid_of(PUBLIC["raa"])
    with pytest.raises(Refusal, match="zero_cells is given twice") as refused:
        fit_mack_grid(grid, zero_cells="observed", links=LinkRules(zero_cells="missing"))
    assert refused.value.options == ("zero_cells", "links")
    fit = fit_mack_grid(grid, links=LinkRules(zero_cells="missing"))
    assert fit.zero_cells == "missing"  # None takes the rule of links
    with pytest.raises(Refusal, match="links must be an ibnr.kernels.links.LinkRules"):
        fit_mack_grid(grid, links={"history_periods": 5})


def test_the_trims_exhausted_at_the_last_age_under_raise():
    with pytest.raises(Refusal, match="drop_high would leave no link ratio from 108 to 120"):
        methods.mack(RAA, drop_high=1, exhausted_exclusions="raise")
    with pytest.raises(Refusal, match="drop_high would leave no link ratio from 108 to 120"):
        fit_mack_grid(grid_of(PUBLIC["raa"]), links=LinkRules(drop_high=1))


# -- every option reaches the answer ------------------------------------------------------

TIES = cells_of(
    rows_of(
        [
            [300, 900, 950, 960, 965],
            [200, 400, 420, 425],
            [100, 300, 310],
            [100, 200],
            [150],
        ]
    )
)
ZERO_FIRST = cells_of(rows_of([[0, 150, 170, 175], [110, 168, 190], [120, 175], [130]]))

#: (option, value, triangle, the other options both calls share)
DELIVERY = [
    ("average", "simple", RAA, {}),
    ("average", "regression", RAA, {}),
    ("history_periods", 3, RAA, {}),
    ("drop_high", 1, RAA, {}),
    ("drop_low", 1, RAA, {}),
    ("preserve", 3, RAA, {"drop_high": 2}),
    ("drop_above", 2.0, RAA, {}),
    ("drop_below", 1.1, RAA, {}),
    ("exclude", [(1982, 12)], RAA, {}),
    ("exclude_valuations", [1986], RAA, {}),
    ("trim_ties", "origin", TIES, {"drop_high": 1}),
    ("sigma_rule", "mack", RAA, {}),
    ("zero_cells", "observed", ZERO_FIRST, {}),
]


@pytest.mark.parametrize(
    ("option", "value", "cells", "shared"), DELIVERY, ids=[f"{o}={v}" for o, v, _, _ in DELIVERY]
)
def test_each_option_changes_macks_standard_error(option, value, cells, shared):
    """Called through ``methods.mack``, each option moves the total standard error
    away from the same call without it: an argument accepted and not passed on
    would leave it where it was."""
    without = methods.mack(cells, **shared)
    with_it = methods.mack(cells, **shared, **{option: value})
    assert total(with_it) != pytest.approx(total(without), rel=1e-6), option


def test_exhausted_exclusions_is_delivered():
    assert np.isfinite(total(methods.mack(RAA, drop_high=1, exhausted_exclusions="keep")))
    with pytest.raises(Refusal):
        methods.mack(RAA, drop_high=1, exhausted_exclusions="raise")


def _matrix(rows) -> np.ndarray:
    """An (origins, ages) matrix, NaN unobserved, from [year, dev_lag, value] rows."""
    first = min(y for y, _, _ in rows)
    out = np.full(
        (max(y for y, _, _ in rows) - first + 1, max(d for _, d, _ in rows) // 12), np.nan
    )
    for y, d, v in rows:
        out[y - first, d // 12 - 1] = v
    return out


def test_fit_mack_and_fit_mack_many_deliver_average_and_links():
    """The Triangle entry points pass ``average`` and ``links`` on to the fit. On raa,
    a simple average over the last five link ratios gives a total standard error of
    27,485.84 and five link ratios alone 22,290.07, where the default fit gives
    26,909.01: a dropped argument would give the default."""
    from ibnr.kernels.mack import fit_mack, fit_mack_many

    from .conftest import make_cohort_triangle

    rules = LinkRules(history_periods=5)
    se = {}
    for name, rows in (("raa", PUBLIC["raa"]), ("genins", PUBLIC["genins"])):
        matrix = _matrix(rows)
        one = make_cohort_triangle(None, matrix, start_year=int(rows[0][0]))
        default = fit_mack(one)
        for average in ("volume", "simple"):
            fit = fit_mack(one, average=average, links=rules)
            assert fit.average == average and fit.links == rules
            expected = fit_mack_grid(grid_of(rows), average=average, links=rules)
            assert fit.msep_runoff()["msep_total"] == expected.msep_runoff()["msep_total"]
            assert fit.msep_runoff()["msep_total"] != default.msep_runoff()["msep_total"]
            se[name, average] = float(np.sqrt(fit.msep_runoff()["msep_total"]))
        se[name, "default"] = float(np.sqrt(default.msep_runoff()["msep_total"]))
    assert se["raa", "default"] == pytest.approx(26_909.01, abs=0.01)
    assert se["raa", "simple"] == pytest.approx(27_485.84, abs=0.01)
    assert se["raa", "volume"] == pytest.approx(22_290.07, abs=0.01)

    # the same two cohorts in one triangle, fitted in one pass
    import pandas as pd

    from ibnr import Triangle

    frames = [
        make_cohort_triangle(None, _matrix(rows), segment={"lob": name}).execute()
        for name, rows in (("raa", PUBLIC["raa"]), ("genins", PUBLIC["genins"]))
    ]
    both = Triangle.from_long(pd.concat(frames, ignore_index=True), measure="cumulative")
    for average in ("volume", "simple"):
        panel = fit_mack_many(both, average=average, links=rules)
        for name in ("raa", "genins"):
            fit = panel[name]
            assert fit.average == average and fit.links == rules
            total_se = float(np.sqrt(fit.msep_runoff()["msep_total"]))
            assert total_se == pytest.approx(se[name, average], rel=1e-12)
            assert total_se != pytest.approx(se[name, "default"], rel=1e-6)


# -- 6. chainladder-python and the example workbook ----------------------------------------

CL_SAMPLES = ("raa", "genins", "ukmotor", "abc", "mw2014")


@pytest.fixture(scope="module")
def cl():
    return pytest.importorskip("chainladder")


def _cl_cells(tri) -> tuple[pa.Table, int]:
    matrix = np.asarray(tri.values[0, 0], dtype=float)
    first = int(tri.origin[0].year)
    return cells_of(rows_of(matrix, first_year=first)), first


def _cl_mack(cl, tri, dev: dict, *, patch: bool) -> dict:
    """chainladder's Mack under ``cl.Development(**dev)``, optionally with its two
    standard-error defects patched: the latest diagonal's estimation weight put
    back to 1 (defect A), and every factor's standard error recomputed from the
    link ratios chainladder itself used, ``sigma / sqrt(sum C^alpha)`` (defect B,
    whose shortcut reads the first origin's cell)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = cl.Development(**dev).fit_transform(tri)
        cum = np.asarray(tri.values[0, 0], dtype=float)
        n_w, n_d = cum.shape
        if patch:
            obs = ~np.isnan(cum)
            weights = np.nan_to_num(np.array(X.w_[0, 0], dtype=float))
            used = obs[:, :-1] & obs[:, 1:] & (weights > 0)
            latest = [int(np.flatnonzero(obs[i]).max()) for i in range(n_w)]
            w = np.array(X.w_, dtype=float).copy()
            for i in range(n_w):
                if latest[i] < n_d - 1:
                    w[0, 0, i, latest[i]] = 1.0
            alpha = {"volume": 1, "simple": 0, "regression": 2}[dev.get("average", "volume")]
            sigma = np.asarray(X.sigma_.values, dtype=float)[0, 0, 0, : n_d - 1]
            total_weight = np.array([(cum[used[:, j], j] ** alpha).sum() for j in range(n_d - 1)])
            std_err = X.std_err_.copy()
            values = np.array(std_err.values, dtype=float)
            values[0, 0, 0, : n_d - 1] = sigma / np.sqrt(total_weight)
            std_err.values = values
            X.w_ = w
            X.std_err_ = std_err
        m = cl.MackChainladder().fit(X)
    return {
        "ldf": np.asarray(m.X_.ldf_.values[0, 0, 0], dtype=float)[: n_d - 1],
        "total": float(np.asarray(m.total_mack_std_err_.values).ravel()[0]),
        "origins": np.nan_to_num(np.asarray(m.summary_.values[0, 0, :, 3], dtype=float)),
        "ultimate": np.asarray(m.ultimate_.values[0, 0, :, -1], dtype=float),
    }


def _ours(cl_options: dict) -> dict:
    """chainladder's development options in ibnr's names."""
    options = {}
    for name, value in cl_options.items():
        if name == "n_periods":
            options["history_periods"] = None if value == -1 else value
        elif name == "drop":
            options["exclude"] = [(int(o), lag) for o, lag in value]
        elif name == "drop_valuation":
            # chainladder names a link ratio's earlier end, ibnr its later one
            options["exclude_valuations"] = [int(value) + 1]
        else:
            options[name] = value
    return options


def _one_kept_off_the_first_origin(result) -> bool:
    ratios = result.link_ratios.filter(result.link_ratios["included"])
    by_age: dict[int, list] = {}
    for origin, lag in zip(
        ratios["origin"].to_pylist(), ratios["from_dev_lag"].to_pylist(), strict=True
    ):
        by_age.setdefault(lag, []).append(origin)
    first = result.origins["origin"][0].as_py()
    return any(len(kept) == 1 and kept[0] != first for kept in by_age.values())


def _no_defect_settings(first: int) -> list[dict]:
    out = []
    for average, n_periods, drop, valuation in itertools.product(
        ("volume", "simple", "regression"), (-1, 3, 5), (False, True), (False, True)
    ):
        dev = {"average": average, "n_periods": n_periods}
        if drop:
            dev["drop"] = [(str(first + 1), 12)]
        if valuation:
            dev["drop_valuation"] = str(first + 3)
        out.append(dev)
    return out


@pytest.mark.tieout
@pytest.mark.parametrize("sample", CL_SAMPLES)
def test_mack_equals_chainladder_where_its_defects_do_not_fire(cl, sample):
    """No drop or bound, no latest valuation: chainladder's Mack is right, and
    equals ibnr's (the log-linear rule, as chainladder fills sigmas). A setting
    that leaves one link ratio at an age, not the first origin's, fires
    chainladder's second defect and is compared in the next test instead."""
    tri = cl.load_sample(sample)
    cells, first = _cl_cells(tri)
    compared = 0
    for dev in _no_defect_settings(first):
        ours = methods.mack(cells, **_ours(dev))
        if _one_kept_off_the_first_origin(ours):
            continue
        theirs = _cl_mack(cl, tri, dev, patch=False)
        np.testing.assert_allclose(
            column(ours, "factor", "development")[:-1], theirs["ldf"], rtol=1e-12
        )
        np.testing.assert_allclose(column(ours, "mack_se"), theirs["origins"], rtol=1e-8, atol=1e-8)
        assert total(ours) == pytest.approx(theirs["total"], rel=1e-8), dev
        compared += 1
    assert compared >= 24, compared


#: settings under which chainladder's latest-diagonal weight (defect A), or its
#: first-origin standard error (defect B), is in play
DEFECT_SETTINGS = [
    {"drop_high": 1},
    {"drop_high": 2},
    {"drop_low": 1},
    {"drop_high": 2, "drop_low": 1},
    {"drop_above": 100.0},
    {"drop_below": 1.1},
    {"n_periods": 5, "drop_low": 2},
    {"n_periods": 5, "drop_high": 1, "average": "regression"},
    {"drop_high": 1, "average": "simple"},
]


@pytest.mark.tieout
@pytest.mark.parametrize("sample", CL_SAMPLES)
@pytest.mark.parametrize("dev", DEFECT_SETTINGS, ids=[str(d) for d in DEFECT_SETTINGS])
def test_mack_equals_chainladder_with_its_two_defects_patched(cl, sample, dev):
    """With a drop or a bound, chainladder's factors and ultimates are ibnr's, and
    its standard errors are ibnr's once both defects are patched."""
    tri = cl.load_sample(sample)
    cells, _ = _cl_cells(tri)
    try:
        ours = methods.mack(cells, **_ours(dev))
    except Refusal as refusal:
        pytest.skip(f"ibnr refuses: {refusal.reason}")
    raw = _cl_mack(cl, tri, dev, patch=False)
    patched = _cl_mack(cl, tri, dev, patch=True)
    np.testing.assert_allclose(column(ours, "factor", "development")[:-1], raw["ldf"], rtol=1e-12)
    np.testing.assert_allclose(column(ours, "ultimate"), raw["ultimate"], rtol=1e-12)
    assert total(ours) == pytest.approx(patched["total"], rel=1e-8)
    np.testing.assert_allclose(column(ours, "mack_se"), patched["origins"], rtol=1e-8, atol=1e-8)


@pytest.mark.tieout
@pytest.mark.parametrize(
    ("dev", "ours_total", "theirs_total"),
    [
        ({"drop_above": 100.0}, 26_880.74, 13_037.79),  # leaves out nothing
        ({"drop_valuation": "1990"}, 26_880.74, 13_037.79),  # the latest: leaves out nothing
        ({"drop_high": 1}, 14_811.95, 7_796.95),
        ({"drop_high": 2}, 11_359.79, 7_658.91),
        ({"drop_low": 1}, 31_377.00, 15_115.94),
    ],
)
def test_chainladders_named_differences_on_raa(cl, dev, ours_total, theirs_total):
    """The numbers ``docs/coming-from-chainladder.md`` quotes."""
    tri = cl.load_sample("raa")
    assert _cl_mack(cl, tri, dev, patch=False)["total"] == pytest.approx(theirs_total, abs=0.005)
    # chainladder's drop_valuation="1990" names ratios starting on the latest
    # diagonal, and there are none, so ibnr's equivalent is no option at all
    ours = methods.mack(RAA) if "drop_valuation" in dev else methods.mack(RAA, **dev)
    assert total(ours) == pytest.approx(ours_total, abs=0.005)


def _clrd_clean(cl) -> list:
    clrd = cl.load_sample("clrd")["CumPaidLoss"]
    out = []
    for k in range(clrd.shape[0]):
        tri = clrd.iloc[k]
        v = np.asarray(tri.values[0, 0], dtype=float)
        obs = ~np.isnan(v)
        stair = all(obs[i, : 10 - i].all() and not obs[i, 10 - i :].any() for i in range(10))
        if stair and (v[obs] > 0).all():
            out.append((k, tri, v))
    return out


@pytest.mark.tieout
@pytest.mark.parametrize(
    "dev",
    [{"drop_high": 1}, {"drop_low": 1}, {"n_periods": 5, "drop_high": 1, "average": "regression"}],
    ids=["drop_high", "drop_low", "window_high_regression"],
)
def test_mack_equals_patched_chainladder_on_clrd(cl, dev):
    """The 353 clrd paid triangles that are full positive staircases, against
    chainladder with both defects patched. Two kinds of triangle differ, both
    named in ``docs/coming-from-chainladder.md``:

    - a sigma estimated as exactly 0 (every link ratio kept at an age equal, which
      clrd's whole-thousand amounts make common at late ages) when another sigma
      has to be filled in: chainladder's log-linear fill reads the 0 as 1e-320,
      R's regression leaves it out, and ibnr does as R (Mack's rule when only the
      last age is filled, as in 0.7.2). About 108 of 348 under these settings;
    - clrd 377 under ``drop_low=1``, where chainladder projects the 1990 origin's
      108-month cell as 1365 instead of 1365 x 1.001203: R's
      ``MackChainLadder`` with the same weights gives ibnr's 331.554588864.
    """
    clean = _clrd_clean(cl)
    assert len(clean) == 353
    zero_sigma, compared, other = 0, 0, []
    for k, tri, v in clean:
        cells = cells_of(rows_of(v, first_year=1988))
        try:
            ours = methods.mack(cells, **_ours(dev))
        except Refusal as refusal:
            # five triangles are left with at most one link ratio at every age,
            # or a gap before the third age: refused by name
            assert refusal.reason == "variance_not_estimable", k
            continue
        patched = _cl_mack(cl, tri, dev, patch=True)
        if total(ours) == pytest.approx(patched["total"], rel=1e-8):
            compared += 1
            continue
        sigma = column(ours, "sigma", "development")[:-1]
        extrapolated = np.array(ours.development["sigma_extrapolated"].to_pylist()[:-1])
        if (sigma[~extrapolated] == 0).any() and extrapolated.any():
            zero_sigma += 1
        else:
            other.append(k)
    assert compared >= 239, (compared, zero_sigma)
    assert zero_sigma <= 108
    assert other == ([377] if dev == {"drop_low": 1} else [])
    if other:
        v = next(v for k, _, v in clean if k == 377)
        assert total(methods.mack(cells_of(rows_of(v, first_year=1988)), **_ours(dev))) == (
            pytest.approx(331.554588864, rel=1e-11)
        )


# -- the example workbook, from the mart ----------------------------------------------------

PUBLISH = "20260613_041006"
SOURCE = f"github://EKtheSage/cas-schedule-p-data-model@{PUBLISH}"


def _mart_cached() -> bool:
    try:
        from ibnr.data.schedule_p import active_mart_path

        return active_mart_path(SOURCE).exists()
    except Exception:
        return False


@pytest.fixture(scope="module")
def njm_paid() -> pa.Table:
    """New Jersey Manufacturers (NAIC 7080), workers' compensation paid, as of 1997."""
    import pandas as pd

    from ibnr.data.schedule_p import load_schedule_p

    tri = load_schedule_p(SOURCE, companies=["7080"], lines=["workers_compensation"])
    frame = tri.as_of(dt.date(1997, 12, 31)).execute()
    frame["year"] = pd.to_datetime(frame["origin_period"]).dt.year
    paid = frame[(frame["field"] == "paid_loss") & frame["year"].between(1988, 1997)]
    return pa.table(
        {
            "origin_period": paid["year"].tolist(),
            "dev_lag": paid["dev_lag"].astype(int).tolist(),
            "value": paid["value"].astype(float).tolist(),
        }
    )


@pytest.mark.mart
@pytest.mark.skipif(not _mart_cached(), reason=f"Schedule P publish {PUBLISH} is not reachable")
@pytest.mark.parametrize(
    ("options", "total_se"),
    [
        ({"drop_high": 1}, 10_329.10),  # the app's /reserve gives 5,914.68
        ({"drop_high": 1, "drop_low": 1}, 8_342.65),  # 4,266.05
        ({"drop_low": 1}, 9_345.41),  # 5,550.69
        ({"average": "simple"}, 10_058.10),
        ({"average": "regression"}, 12_056.94),
        ({"history_periods": 5}, 12_720.86),
    ],
)
def test_the_example_workbooks_standard_errors(njm_paid, options, total_se):
    assert total(methods.mack(njm_paid, **options)) == pytest.approx(total_se, abs=0.01)
