"""Tails: constant, exponential, inverse power and Weibull, and Mack's tail variance.

Seven groups of checks:

1. nothing untailed moved: every conventional fit, Mack fit and ``ibnr.methods``
   table has the same bytes as before tails (a frozen pin, from the source of
   ``feat/generalized-mack``), except the cases that take log, exp or Cape Cod's
   trend, which are the same numbers to 1e-14;
2. the tail arithmetic against chainladder-python 0.9.2 (marker ``tieout``):
   every factor after the attachment, the steps shown beyond the triangle and
   the rest, on eight public triangles and every clrd triangle, and the four
   point methods' ultimates through ``ibnr.methods``;
3. Mack's tail step against R's ``MackChainLadder(tail=, tail.se=,
   tail.sigma=)`` at alpha 0, 1 and 2 (frozen JSON; CI has no R), against a
   transcription of the closed form, and against chainladder where it has no
   defect in play;
4. the rule for Mack's tail variance where chainladder's is wrong: a
   quarterly triangle whose late factors are exactly 1;
5. behaviour: ``tail_rows`` never moves an ultimate, ``tail_decay`` only with
   an earlier attachment, a tail of 1.0 is no tail, zeros, the unity fallback;
6. every option delivered to the answer it is meant to change, and nothing
   else;
7. the kernel's own refusals, the one-year result and the simulations refusing
   a tailed fit, and the codec.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import inspect
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from ibnr import methods
from ibnr.errors import Refusal
from ibnr.kernels.cdr import (
    DiagonalGenerator,
    MackDiagonal,
    ODPBootstrapDiagonal,
    one_year_cdr,
    rereserve,
    simulate_one_year_cdr,
)
from ibnr.kernels.codec import CODEC_VERSION, from_arrow
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional_grid
from ibnr.kernels.grid import grid_from_columns
from ibnr.kernels.links import LinkRules
from ibnr.kernels.mack import MackFit, draw_next_cells, fit_mack_grid, simulate_ultimates
from ibnr.kernels.tail import MIN_FIT_FACTOR, TailSpec, apply_tail, tail_variance

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import freeze_untailed_pin as frozen  # noqa: E402

PUBLIC = json.loads((DATA / "refusal_triangles.json").read_text("utf-8"))
PIN = json.loads((DATA / "untailed_pin.json").read_text("utf-8"))
R_FIXTURE = json.loads((DATA / "r_mack_tail.json").read_text("utf-8"))
ALPHA_AVERAGE = {0: "simple", 1: "volume", 2: "regression"}
CURVES = ("exponential", "inverse_power", "weibull")


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


def premium_for(rows) -> dict:
    years = sorted({y for y, _, _ in rows})
    top = max(v for _, _, v in rows)
    return dict(zip(years, (np.linspace(1.0, 2.0, len(years)) * top).tolist(), strict=True))


def column(result, name: str, table: str = "origins") -> np.ndarray:
    return np.array(getattr(result, table)[name].to_pylist(), dtype=float)


def total(result, name: str) -> float:
    return result.totals[name][0].as_py()


RAA = cells_of(PUBLIC["raa"])
RAA_PREMIUM = premium_for(PUBLIC["raa"])


def call(method: str, cells=RAA, premium=RAA_PREMIUM, **options):
    """One method with the extra arguments it needs."""
    if method in ("bornhuetter_ferguson", "benktander"):
        options = {"premium": premium, "expected_loss_ratio": 0.75, **options}
    elif method == "cape_cod":
        options = {"premium": premium, **options}
    return getattr(methods, method)(cells, **options)


METHODS = ("chain_ladder", "bornhuetter_ferguson", "benktander", "cape_cod", "mack")


# -- 1. nothing untailed moved -----------------------------------------------------


def _public_triangles() -> dict[str, list]:
    import freeze_conventional_pin
    import freeze_mack_pin

    return {**freeze_conventional_pin.public_triangles(), **freeze_mack_pin.with_zeros()}


def _close(now: list, before: list) -> bool:
    """Numbers within 1e-14 relative, with an absolute tolerance of 1e-14 times
    the largest value in the array, and nulls, NaN and infinities where they were.

    The cases stored as numbers take log, exp or a power with an inexact result,
    whose last bits differ between the Windows the pin was frozen on and Linux.
    """
    if len(now) != len(before) or [v is None for v in now] != [v is None for v in before]:
        return False
    a = np.array([np.nan if v is None else v for v in now], dtype=float)
    b = np.array([np.nan if v is None else v for v in before], dtype=float)
    finite = np.isfinite(b)
    scale = np.abs(b[finite]).max() if finite.any() else 0.0
    return bool(np.allclose(a, b, rtol=1e-14, atol=1e-14 * scale, equal_nan=True))


def _moved(key: str, before: str | dict, now: str | dict) -> list[str]:
    """What moved in one case. A digest (or a refusal) must be the same string. A
    case stored as named parts must keep every part, each digest the same and
    each list of numbers within :func:`_close`; a part added since is not in the
    pin and is not compared."""
    if isinstance(before, str) or isinstance(now, str):
        return [] if before == now else [key]
    missing = set(before) - set(now)
    moved = [f"{key}: lost {sorted(missing)}"] if missing else []
    for name, old in before.items():
        if name not in now:
            continue
        new = now[name]
        same = _close(new, old) if isinstance(old, list) and isinstance(new, list) else new == old
        if not same:
            moved.append(f"{key}: {name}")
    return moved


def _compare(frozen_part: dict, now: dict) -> list[str]:
    return [moved for key, value in now.items() for moved in _moved(key, frozen_part[key], value)]


def test_only_the_log_exp_and_trend_cases_are_pinned_as_numbers():
    """Every case but those that take log, exp or ``(1 + trend) ** t`` is pinned by
    its exact bits, so the looser comparison reaches only where it has to."""

    def labels(part: str) -> set[str]:
        return {
            key.rsplit("|", 1)[1]
            for key, value in PIN[part].items()
            if isinstance(value, dict) and any(isinstance(v, list) for v in value.values())
        }

    assert labels("conventional") == set(frozen.NUMBER_CONVENTIONAL) == {"gcc_trend"}
    assert labels("methods") == set(frozen.NUMBER_METHODS) == {"cape_cod", "mack", "mack_observed"}
    assert all(isinstance(value, str) for value in PIN["mack"].values())
    assert set(PIN["mack"]) == set(PIN["mack_log_linear"])


def test_untailed_answers_did_not_move_on_the_public_triangles():
    now = frozen.pin(_public_triangles())
    assert _compare(PIN["conventional"], now["conventional"]) == []
    assert _compare(PIN["mack"], now["mack"]) == []
    assert _compare(PIN["mack_log_linear"], now["mack_log_linear"]) == []
    assert _compare(PIN["methods"], now["methods"]) == []
    assert len(now["methods"]) == 8 * 11  # every method case of the eleven triangles ran


@pytest.mark.tieout
def test_untailed_answers_did_not_move_on_clrd():
    pytest.importorskip("chainladder")
    import freeze_conventional_pin

    now = frozen.pin(freeze_conventional_pin.clrd_triangles())
    assert _compare(PIN["conventional"], now["conventional"]) == []
    assert _compare(PIN["mack"], now["mack"]) == []
    assert _compare(PIN["mack_log_linear"], now["mack_log_linear"]) == []
    assert _compare(PIN["methods"], now["methods"]) == []
    assert len(now["mack"]) == 36 * 4  # every fourteenth clrd paid triangle, four settings


def test_an_untailed_result_says_it_has_no_tail():
    result = methods.mack(RAA)
    development = result.development
    assert development.num_rows == 10
    assert development["source"].to_pylist() == ["link_ratios"] * 9 + [None]
    assert development["curve_factor"].null_count == 10
    assert development["in_tail_fit"].null_count == 10
    assert total(result, "tail_factor") == 1.0
    for name in ("tail_sigma", "tail_std_err", "tail_position"):
        assert result.totals[name][0].as_py() is None
    assert total(methods.chain_ladder(RAA), "tail_factor") == 1.0


# -- 2. the tail arithmetic against chainladder --------------------------------------


@pytest.fixture(scope="module")
def cl():
    return pytest.importorskip("chainladder")


def _samples(cl) -> dict:
    clrd = cl.load_sample("clrd").groupby("LOB").sum()
    return {
        "raa": cl.load_sample("raa"),
        "genins": cl.load_sample("genins"),
        "ukmotor": cl.load_sample("ukmotor"),
        "abc": cl.load_sample("abc"),
        "mw2014": cl.load_sample("mw2014"),
        "tail_sample.paid": cl.load_sample("tail_sample")["paid"],
        "tail_sample.incurred": cl.load_sample("tail_sample")["incurred"],
        "clrd.comauto.paid": clrd.loc["comauto"]["CumPaidLoss"],
    }


GRAIN = {"Y": 12, "S": 6, "Q": 3, "M": 1}


def _spec_for(kind: str, argument, chainladder_options: dict, step: int) -> TailSpec:
    """chainladder's TailCurve/TailConstant options as a TailSpec, by the spec's mapping."""
    rows = int(chainladder_options.get("projection_period", 12) / 12) * (12 // step)
    if kind == "constant":
        return TailSpec(
            "constant",
            factor=argument,
            decay=chainladder_options.get("decay"),
            attach_lag=chainladder_options.get("attachment_age"),
            rows=rows,
        )
    first, end = chainladder_options.get("fit_period", (None, None))
    return TailSpec(
        argument,
        attach_lag=chainladder_options.get("attachment_age"),
        steps=chainladder_options.get("extrap_periods"),
        rows=rows,
        # chainladder's end is exclusive; ours names the last link fitted
        fit_lags=(first, None if end is None else end - step),
    )


def _tail_cases(step: int) -> list:
    cases = []
    for curve in CURVES:
        for options in (
            {},
            {"attachment_age": 5 * step},
            {"extrap_periods": 30, "projection_period": 36},
            {"fit_period": (3 * step, None)},
            {"fit_period": (2 * step, 7 * step)},
        ):
            cases.append(("curve", curve, options))
    for options in (
        {},
        {"decay": 0.7},
        {"attachment_age": 5 * step},
        {"attachment_age": 5 * step, "decay": 0.9},
        {"projection_period": 36},
    ):
        cases.append(("constant", 1.05, options))
    return cases


def _relative(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


@pytest.mark.tieout
def test_the_tail_arithmetic_matches_chainladder(cl):
    """Every factor after the attachment, each step shown beyond the triangle, the
    rest, and the tail factor, fed chainladder's own untailed factors, so only the
    tail is under test: 25 settings on eight triangles."""
    warnings.simplefilter("ignore")
    worst = 0.0
    for name, triangle in _samples(cl).items():
        development = cl.Development().fit_transform(triangle)
        f = development.ldf_.values.ravel()
        n = f.size
        step = GRAIN[triangle.development_grain]
        for kind, argument, options in _tail_cases(step):
            if kind == "curve":
                reference = cl.TailCurve(argument, **options).fit(development)
            else:
                reference = cl.TailConstant(argument, **options).fit(development)
            fit = apply_tail(f, step, _spec_for(kind, argument, options, step))
            theirs = reference.ldf_.values.ravel()
            ours = np.r_[fit.factors, fit.shown, fit.beyond_cdf[-1]]
            assert ours.size == theirs.size, (name, kind, argument, options)
            error = max(
                _relative(ours, theirs),
                _relative(fit.tail_factor, np.asarray(reference.tail_.values).ravel()[0]),
                _relative(fit.factors[:n], theirs[:n]),
            )
            assert error < 1e-12, (name, kind, argument, options, error)
            worst = max(worst, error)
    assert worst > 0  # the comparison compared numbers


@pytest.mark.tieout
@pytest.mark.parametrize("field", ["CumPaidLoss", "IncurLoss"])
def test_every_clrd_curve_matches_chainladder_or_is_refused(cl, field):
    """Every clrd company and line whose factors chainladder can give (485 paid and
    494 incurred of 775; the rest have a missing factor), one curve fit per
    triangle in one call. Where ours answers, it equals chainladder's tail. Where
    fewer than two factors are above 1.00001, chainladder's tail is its silent 1.0
    and ours is refused. Every other refusal is a line that does not decay, and the
    line is chainladder's own: the same slope, which chainladder extrapolates
    anyway."""
    warnings.simplefilter("ignore")
    clrd = cl.load_sample("clrd")[field]
    development = cl.Development().fit_transform(clrd)
    f = np.asarray(development.ldf_.values)[:, 0, 0, :]
    finite = np.isfinite(f).all(axis=1)
    f = f[finite]
    counts = {}
    for curve in CURVES:
        reference = cl.TailCurve(curve).fit(development)
        theirs = np.asarray(reference.tail_.values).reshape(-1)[finite]
        slope = np.asarray(reference._slope_).reshape(-1)[finite]
        ours = apply_tail(f, 12, TailSpec(curve), on_error="flag")
        few = (f > MIN_FIT_FACTOR).sum(axis=1) < 2
        # chainladder's silent 1.0: nothing to fit
        assert (~ours.ok[few]).all()
        assert np.all(theirs[few] == 1.0)
        np.testing.assert_allclose(ours.tail_factor[ours.ok], theirs[ours.ok], rtol=1e-12)
        grows = ~ours.ok & ~few
        # a flat line (two equal factors) has slope 0 here and NaN or -4e-16 there
        known = grows & np.isfinite(slope)
        np.testing.assert_allclose(ours.slope[known], slope[known], rtol=1e-9, atol=1e-12)
        assert (ours.slope[grows & ~known] == 0).all()
        mine = ours.slope
        decays = {"exponential": mine < 0, "inverse_power": mine < -1, "weibull": mine > 0}
        assert not decays[curve][grows].any()
        counts[curve] = (int(ours.ok.sum()), int(few.sum()), int(grows.sum()))
    expected = {
        "CumPaidLoss": {
            "exponential": (468, 9, 8),
            "inverse_power": (451, 9, 25),
            "weibull": (468, 9, 8),
        },
        "IncurLoss": {
            "exponential": (307, 102, 85),
            "inverse_power": (206, 102, 186),
            "weibull": (308, 102, 84),
        },
    }
    assert counts == expected[field]


@pytest.mark.tieout
@pytest.mark.parametrize(
    "tail",
    [
        {"tail": "constant", "tail_factor": 1.05},
        {"tail": "exponential"},
        {"tail": "inverse_power", "tail_attach_lag": 72},
        {"tail": "weibull"},
    ],
    ids=["constant", "exponential", "inverse_power_at_72", "weibull"],
)
@pytest.mark.parametrize("name", ["raa", "genins", "abc", "mw2014"])
def test_the_point_methods_match_chainladder(cl, name, tail):
    """Ultimates of the chain ladder, Bornhuetter-Ferguson (0.75 of premium),
    Benktander (two iterations) and Cape Cod, through ibnr.methods, against
    chainladder with the same tail."""
    warnings.simplefilter("ignore")
    rows = PUBLIC[name]
    cells, premium = cells_of(rows), premium_for(rows)
    sample = cl.load_sample(name)
    development = cl.Development().fit_transform(sample)
    kind = tail["tail"]
    if kind == "constant":
        transformer = cl.TailConstant(tail["tail_factor"])
    else:
        transformer = cl.TailCurve(kind, attachment_age=tail.get("tail_attach_lag"))
    tailed = transformer.fit_transform(development)
    weight = sample.latest_diagonal * 0
    weight.values = weight.values + np.array(list(premium.values()))[None, None, :, None]
    theirs = {
        "chain_ladder": cl.Chainladder().fit(tailed),
        "bornhuetter_ferguson": cl.BornhuetterFerguson(apriori=0.75).fit(
            tailed, sample_weight=weight
        ),
        "benktander": cl.Benktander(apriori=0.75, n_iters=2).fit(tailed, sample_weight=weight),
        "cape_cod": cl.CapeCod().fit(tailed, sample_weight=weight),
    }
    for method, reference in theirs.items():
        extra = {"n_iters": 2} if method == "benktander" else {}
        result = call(method, cells, premium, **tail, **extra)
        np.testing.assert_allclose(
            column(result, "ultimate"),
            np.asarray(reference.ultimate_.values).ravel(),
            rtol=1e-10,
            err_msg=method,
        )
        assert total(result, "tail_factor") == pytest.approx(
            float(np.asarray(transformer.tail_.values).ravel()[0]), rel=1e-12
        )


def _mack_reference(cl, triangle, transformer):
    warnings.simplefilter("ignore")
    development = cl.Development().fit_transform(triangle)
    return cl.MackChainladder().fit(transformer.fit_transform(development)), development


@pytest.mark.tieout
@pytest.mark.parametrize("kind", ["constant", *CURVES])
@pytest.mark.parametrize(
    "name",
    [
        "raa",
        "genins",
        "ukmotor",
        "abc",
        "mw2014",
        "tail_sample.paid",
        "tail_sample.incurred",
        "clrd.comauto.paid",
    ],
)
def test_mack_with_a_tail_matches_chainladder(cl, name, kind):
    """Where no factor is at or below 1 and no sigma is 0, chainladder's tail
    variance is the masked rule's: total and per-origin standard errors, their two
    parts, and the tail's sigma and standard error, to 1e-9."""
    triangle = _samples(cl)[name]
    transformer = cl.TailConstant(1.05) if kind == "constant" else cl.TailCurve(kind)
    reference, development = _mack_reference(cl, triangle, transformer)
    long = triangle.to_frame(keepdims=True).reset_index()
    cells = pa.table(
        {
            "origin_period": [o.date() for o in long["origin"]],
            "dev_lag": long["development"].astype("int64").tolist(),
            "value": long[long.columns[-1]].astype(float).tolist(),
        }
    )
    tail = {"tail": kind, "tail_factor": 1.05} if kind == "constant" else {"tail": kind}
    result = methods.mack(cells, dev_grain_months=GRAIN[triangle.development_grain], **tail)

    def last(values):
        return np.asarray(values.values)[0, 0, :, -1]

    np.testing.assert_allclose(column(result, "ultimate"), last(reference.ultimate_), rtol=1e-10)
    np.testing.assert_allclose(
        column(result, "mack_se"), np.nan_to_num(last(reference.mack_std_err_)), rtol=1e-9
    )
    np.testing.assert_allclose(
        column(result, "parameter_se"), last(reference.parameter_risk_), rtol=1e-9
    )
    np.testing.assert_allclose(
        column(result, "process_se"), last(reference.process_risk_), rtol=1e-9
    )
    assert total(result, "mack_se") == pytest.approx(
        float(np.asarray(reference.total_mack_std_err_.values).ravel()[0]), rel=1e-9
    )
    tailed = transformer.fit_transform(development)
    assert total(result, "tail_sigma") == pytest.approx(tailed.sigma_.values.ravel()[-1], rel=1e-9)
    assert total(result, "tail_std_err") == pytest.approx(
        tailed.std_err_.values.ravel()[-1], rel=1e-9
    )


# -- 3. Mack's tail step against R and the closed form ------------------------------


def _r_cases():
    cases = []
    for fit in R_FIXTURE["fits"]:
        label = (
            f"{fit['dataset']}-a{fit['alpha']}-{fit['tail']}"
            f"{'-given' if fit['tail_se'] is not None else ''}-{fit['est_sigma']}"
        )
        cases.append(pytest.param(fit, id=label))
    return cases


R_CASES = _r_cases()


def test_the_r_fixture_covers_what_it_says():
    assert R_FIXTURE["source"].startswith("R 4.5.3, ChainLadder 0.2.21")
    assert len(R_FIXTURE["fits"]) == 5 * 5 * 2
    assert not any("error" in fit for fit in R_FIXTURE["fits"])
    assert {fit["alpha"] for fit in R_FIXTURE["fits"]} == {0, 1, 2}
    assert {fit["tail"] for fit in R_FIXTURE["fits"]} == {1.05, "exponential"}


@pytest.mark.parametrize("fit", R_CASES)
def test_mack_with_a_tail_ties_out_to_r(fit):
    """R's MackChainLadder with a tail, alpha 0, 1 and 2, under both sigma rules:
    the ultimates, each origin's standard error and its two parts, the totals, and
    the tail's factor, sigma and standard error."""
    matrix = np.array(
        [
            [np.nan if v is None else v for v in row]
            for row in R_FIXTURE["triangles"][fit["dataset"]]
        ]
    )
    rows = rows_of(matrix)
    switched = fit["switched"] or fit["est_sigma"] == "Mack"
    options = {
        "sigma_rule": "mack" if switched else "log_linear",
        "average": ALPHA_AVERAGE[fit["alpha"]],
        "zero_cells": "observed",
    }
    if fit["tail"] == "exponential":
        options["tail"] = "exponential"
    else:
        options.update(tail="constant", tail_factor=fit["tail"])
    if fit["tail_se"] is not None:
        options.update(tail_std_err=fit["tail_se"], tail_sigma=fit["tail_sigma"])
    result = methods.mack(cells_of(rows), **options)
    np.testing.assert_allclose(column(result, "ultimate"), fit["ultimate"], rtol=1e-10)
    for name in ("mack_se", "process_se", "parameter_se"):
        np.testing.assert_allclose(column(result, name), fit[name], rtol=1e-8, err_msg=name)
        assert total(result, name) == pytest.approx(fit[f"total_{name}"], rel=1e-8)
    assert total(result, "tail_factor") == pytest.approx(fit["f"][-1], rel=1e-10)
    assert total(result, "tail_sigma") == pytest.approx(fit["sigma"][-1], rel=1e-8)
    assert total(result, "tail_std_err") == pytest.approx(fit["f_se"][-1], rel=1e-8)


def _closed_form(fit: MackFit, sigma: float, se: float) -> tuple[np.ndarray, float]:
    """Mack's msep with one more step carrying the tail, written out step by step."""
    alpha = fit.alpha
    full = fit.full
    factor = fit.tail_factor
    f = np.r_[fit.f, factor]
    sigma2 = np.r_[fit.sigma2, sigma**2]
    se2 = np.r_[fit.sigma2 / fit.s, se**2]
    ultimate = full[:, -1] * factor
    msep = np.zeros(fit.n_w)
    for i in range(fit.n_w):
        for j in range(int(fit.latest_dev[i]), fit.n_d):
            msep[i] += sigma2[j] / f[j] ** 2 / full[i, j] ** alpha + se2[j] / f[j] ** 2
        msep[i] *= ultimate[i] ** 2
    cross = 0.0
    for i in range(fit.n_w):
        shared = sum(se2[j] / f[j] ** 2 for j in range(int(fit.latest_dev[i]), fit.n_d))
        cross += 2 * ultimate[i] * ultimate[i + 1 :].sum() * shared
    return msep, msep.sum() + cross


@pytest.mark.parametrize("average", ["simple", "volume", "regression"])
def test_the_tail_step_is_the_closed_form(average):
    """A hand triangle with a known tail sigma and standard error: msep_runoff
    against Mack's formula with one more step, at each alpha."""
    grid = grid_of(PUBLIC["genins"])
    fit = fit_mack_grid(
        grid,
        average=average,
        links=LinkRules(),
        tail=TailSpec("constant", factor=1.07, sigma=30.0, std_err=0.015),
    )
    per_origin, total_msep = _closed_form(fit, 30.0, 0.015)
    risk = fit.msep_runoff()
    np.testing.assert_allclose(risk["msep"], per_origin, rtol=1e-12)
    assert risk["msep_total"] == pytest.approx(total_msep, rel=1e-12)
    # the fully developed origin has a standard error from the tail step alone
    assert risk["msep"][0] > 0


def test_tail_variance_reads_the_masked_lines():
    """The position line goes through the factors above 1 only, the sigma line
    through the positive sigmas only: a factor below 1 and a zero sigma are left
    out of both the sums of x and of y (chainladder leaves their x in)."""
    f = np.array([2.0, 1.5, 1.2, 0.99, 1.05, 1.02])
    sigma2 = np.array([4.0, 2.0, 1.0, 0.0, 0.25, 0.1])
    s = np.array([100.0, 120.0, 130.0, 140.0, 150.0, 160.0])
    got = tail_variance(f, sigma2, s, 1.03)
    t = np.arange(1.0, 7.0)
    keep = f > 1
    b, a = np.polyfit(t[keep], np.log(f[keep] - 1), 1)
    position = (np.log(0.03) - a) / b
    assert got.position == pytest.approx(position, rel=1e-13)
    positive = sigma2 > 0
    b, a = np.polyfit(t[positive], np.log(np.sqrt(sigma2[positive])), 1)
    assert got.sigma2 == pytest.approx(np.exp(a + b * position) ** 2, rel=1e-12)
    b, a = np.polyfit(t[positive], np.log(np.sqrt(sigma2[positive] / s[positive])), 1)
    assert got.se2 == pytest.approx(np.exp(a + b * position) ** 2, rel=1e-12)
    # the unmasked rule gives another number, so the masking is what the test sees
    b, a = np.polyfit(t, np.log(np.maximum(sigma2, 1e-320) ** 0.5), 1)
    assert got.sigma2 != pytest.approx(np.exp(a + b * position) ** 2, rel=1e-3)


# -- 4. a quarterly triangle whose late factors are exactly 1 ------------------------


def _quarterly() -> pa.Table:
    """Twelve quarterly origins from 2020Q1, developed quarterly; the last three
    links are exactly 1.0 for every origin, as on a paid triangle that has stopped."""
    with open(DATA / "tail_quarterly_triangle.csv", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        lags = [int(value) for value in header[1:]]
        origins, dev_lags, values = [], [], []
        for row in reader:
            origin = dt.date.fromisoformat(row[0])
            for lag, value in zip(lags, row[1:], strict=True):
                if value:
                    origins.append(origin)
                    dev_lags.append(lag)
                    values.append(float(value))
    return pa.table({"origin_period": origins, "dev_lag": dev_lags, "value": values})


@pytest.mark.parametrize(
    ("factor", "total_se", "sigma"), [(1.05, 1054.3015, 1.05943), (1.01, 1021.1777, 1.04577)]
)
def test_a_quarterly_tail_reads_the_positive_sigmas_only(factor, total_se, sigma):
    """ibnr's sigmas of the three flat links are 0, and the masked lines leave them
    out. chainladder fills them as 1e-235 and more, and its standard error comes out
    934.35 for 1.05 and 4.8e15 for 1.01."""
    cells = _quarterly()
    untailed = methods.mack(cells, dev_grain_months=3)
    assert total(untailed, "mack_se") == pytest.approx(889.8578, abs=5e-5)
    result = methods.mack(cells, dev_grain_months=3, tail="constant", tail_factor=factor)
    assert total(result, "mack_se") == pytest.approx(total_se, abs=5e-5)
    assert total(result, "tail_sigma") == pytest.approx(sigma, abs=5e-6)
    # the development shows one year beyond the last age: four quarterly steps
    assert result.development.num_rows == 12 + 4
    assert result.development["dev_lag"].to_pylist()[-5:] == [36, 39, 42, 45, 48]


@pytest.mark.tieout
@pytest.mark.parametrize("kind", ["constant", *CURVES])
def test_a_quarterly_triangles_point_tails_match_chainladder(cl, kind):
    import pandas as pd

    warnings.simplefilter("ignore")
    cells = _quarterly()
    frame = pd.DataFrame(
        {
            "origin": pd.to_datetime(cells["origin_period"].to_pylist()),
            "dev_lag": cells["dev_lag"].to_pylist(),
            "value": cells["value"].to_pylist(),
        }
    )
    frame["valuation"] = [
        o + pd.DateOffset(months=int(d)) - pd.Timedelta(days=1)
        for o, d in zip(frame["origin"], frame["dev_lag"], strict=True)
    ]
    triangle = cl.Triangle(
        frame, origin="origin", development="valuation", columns="value", cumulative=True
    )
    development = cl.Development().fit_transform(triangle)
    transformer = cl.TailConstant(1.05) if kind == "constant" else cl.TailCurve(kind)
    reference = cl.Chainladder().fit(transformer.fit_transform(development))
    tail = {"tail": kind, "tail_factor": 1.05} if kind == "constant" else {"tail": kind}
    result = methods.chain_ladder(cells, dev_grain_months=3, zero_cells="observed", **tail)
    np.testing.assert_allclose(
        column(result, "ultimate"), np.asarray(reference.ultimate_.values).ravel(), rtol=1e-12
    )


# -- 5. behaviour ------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["constant", *CURVES])
def test_tail_rows_never_moves_an_ultimate(kind):
    tail = {"tail": kind, "tail_factor": 1.05} if kind == "constant" else {"tail": kind}
    results = [methods.chain_ladder(RAA, tail_rows=rows, **tail) for rows in range(11)]
    first = column(results[0], "ultimate")
    for rows, result in enumerate(results):
        assert column(result, "ultimate").tobytes() == first.tobytes(), rows
        assert result.development.num_rows == 10 + rows
        development = result.development
        factors = np.array(development["factor"].to_pylist()[9:-1], dtype=float)
        rest = development["cdf"].to_pylist()[-1]
        # the steps shown, times the rest the final row carries, are the tail factor
        assert np.prod(factors) * rest == pytest.approx(total(result, "tail_factor"), rel=1e-14)
        assert development["factor"].to_pylist()[-1] is None


def test_tail_decay_moves_ultimates_only_with_an_earlier_attachment():
    base = {"tail": "constant", "tail_factor": 1.05, "tail_rows": 3}
    at_last = [
        column(methods.chain_ladder(RAA, tail_decay=decay, **base), "ultimate")
        for decay in (0.0, 0.5, 0.9, 1.0)
    ]
    assert all(values.tobytes() == at_last[0].tobytes() for values in at_last)
    early = {
        decay: methods.chain_ladder(RAA, tail_decay=decay, tail_attach_lag=72, **base)
        for decay in (0.5, 0.9)
    }
    assert total(early[0.5], "ultimate") == pytest.approx(205_714.76, abs=0.005)
    assert total(early[0.9], "ultimate") == pytest.approx(207_902.47, abs=0.005)


@pytest.mark.parametrize("method", METHODS)
def test_a_tail_of_one_is_no_tail(method):
    plain = call(method)
    one = call(method, tail="constant", tail_factor=1.0)
    for name in plain.origins.column_names:
        assert one.origins[name].equals(plain.origins[name]), name
    if method == "mack":
        assert total(one, "mack_se") == total(plain, "mack_se")
        assert total(one, "tail_sigma") == 0.0
        assert total(one, "tail_std_err") == 0.0
        assert one.totals["tail_position"][0].as_py() is None


def test_a_constant_tail_attached_at_the_first_age_is_every_cdf_from_12():
    result = methods.chain_ladder(RAA, tail="constant", tail_factor=1.05, tail_attach_lag=12)
    cdf = column(result, "cdf", "development")
    assert cdf[0] == pytest.approx(1.05, rel=1e-15)
    ultimates = column(result, "ultimate")
    latest = column(result, "latest")
    assert ultimates[-1] == pytest.approx(latest[-1] * 1.05, rel=1e-15)
    assert result.development["source"].to_pylist() == ["tail"] * 11


def test_a_zero_latest_amount_keeps_an_ultimate_and_a_standard_error_of_zero():
    rows = [[y, d, 0.0 if (y, d) == (1990, 12) else v] for y, d, v in PUBLIC["raa"]]
    result = methods.mack(cells_of(rows), tail="constant", tail_factor=1.05)
    ultimate, se = column(result, "ultimate"), column(result, "mack_se")
    assert ultimate[-1] == 0.0 and se[-1] == 0.0
    assert (se[:-1] > 0).all()
    # A fully developed origin at zero cannot come from cells (its last link would
    # have no ratio), so it is set by hand: with a tail it still develops a step,
    # which under "missing" is the limit 0 and under "observed" divides by zero.
    fit = fit_mack_grid(
        grid_of(PUBLIC["raa"]), zero_cells="missing", tail=TailSpec("constant", factor=1.05)
    )
    cum = fit.cum.copy()
    cum[0, -1] = 0.0
    zeroed = dataclasses.replace(fit, cum=cum)
    assert zeroed.ultimate[0] == 0.0 and zeroed.msep_runoff()["msep"][0] == 0.0
    with pytest.raises(Refusal, match="non-positive cumulative on the latest diagonal"):
        dataclasses.replace(zeroed, zero_cells="observed").msep_runoff()


def test_a_unity_factor_is_left_out_of_the_curve():
    rows = [[y, d, 0.0 if d <= 12 else v] for y, d, v in PUBLIC["raa"] if y != 1990]
    result = methods.chain_ladder(cells_of(rows), unsupported_factor="unity", tail="exponential")
    development = result.development
    assert development["unity_fallback"].to_pylist()[0] is True
    assert development["in_tail_fit"].to_pylist()[0] is False
    assert all(development["in_tail_fit"].to_pylist()[1:8])


def test_a_tail_result_has_nulls_never_nan():
    for method in METHODS:
        result = call(method, tail="weibull", tail_rows=2)
        for table in (result.origins, result.development, result.totals):
            for name in table.column_names:
                values = table[name]
                if pa.types.is_floating(values.type):
                    present = [v for v in values.to_pylist() if v is not None]
                    assert np.isfinite(present).all(), (method, name)


def test_the_oldest_origin_gets_the_tail():
    for method in ("chain_ladder", "bornhuetter_ferguson", "benktander", "cape_cod"):
        result = call(method, tail="constant", tail_factor=1.05)
        ibnr = column(result, "ibnr")
        assert ibnr[0] > 0, method
    result = methods.chain_ladder(RAA, tail="constant", tail_factor=1.05)
    assert column(result, "ibnr")[0] == pytest.approx(18834.0 * 0.05, rel=1e-13)


def test_bornhuetter_ferguson_and_cape_cod_read_the_tailed_pattern():
    """BF's unreported share and Cape Cod's used-up premium both use the cdf with
    the tail in it."""
    result = methods.chain_ladder(RAA, tail="inverse_power")
    cdf = column(result, "cdf", "development")[:10][::-1]  # each origin's latest age
    latest = column(result, "latest")
    premium = np.array(list(RAA_PREMIUM.values()))
    bf = methods.bornhuetter_ferguson(
        RAA, premium=RAA_PREMIUM, expected_loss_ratio=0.75, tail="inverse_power"
    )
    np.testing.assert_allclose(
        column(bf, "ultimate"), latest + 0.75 * premium * (1 - 1 / cdf), rtol=1e-13
    )
    cc = methods.cape_cod(RAA, premium=RAA_PREMIUM, tail="inverse_power")
    ratio = latest.sum() / (premium / cdf).sum()
    np.testing.assert_allclose(
        column(cc, "ultimate"), latest + ratio * premium * (1 - 1 / cdf), rtol=1e-13
    )


def test_the_development_rows_rebuild_chainladders_labels():
    """What the Reserving app rebuilds from development: the ldf values are every
    non-null factor then the final row's cdf, the cdf values every row's cdf."""
    result = methods.chain_ladder(RAA, tail="constant", tail_factor=1.05)
    development = result.development
    lags = development["dev_lag"].to_pylist()
    factors = development["factor"].to_pylist()
    labels = [f"{a}-{a + 12}" for a, f in zip(lags, factors, strict=True) if f is not None]
    labels.append(f"{lags[-1]}-{lags[-1] + 12}")
    assert labels[-3:] == ["108-120", "120-132", "132-144"]
    assert [f"{a}-Ult" for a in lags[-2:]] == ["120-Ult", "132-Ult"]
    assert development["cdf"].to_pylist()[-2] == pytest.approx(1.05, rel=1e-15)
    assert development["cdf"].to_pylist()[-1] == pytest.approx(1.05 / 1.0240113, rel=1e-6)


def test_the_curve_is_fitted_to_the_factors_the_options_chose():
    plain = methods.chain_ladder(RAA, tail="exponential")
    dropped = methods.chain_ladder(RAA, tail="exponential", drop_high=True)
    assert total(plain, "tail_factor") != total(dropped, "tail_factor")
    kernel = apply_tail(
        np.array(dropped.development["factor"].to_pylist()[:9], dtype=float),
        12,
        TailSpec("exponential"),
    )
    assert total(dropped, "tail_factor") == float(kernel.tail_factor)


def test_one_row_per_simulation_is_each_row_on_its_own():
    rng = np.random.default_rng(7)
    f = 1 + np.exp(-0.6 * np.arange(1, 10))[None, :] * rng.uniform(0.7, 1.3, size=(50, 9))
    for kind in CURVES:
        spec = TailSpec(kind, rows=2, steps=40)
        batch = apply_tail(f, 12, spec)
        for i in range(0, 50, 7):
            one = apply_tail(f[i], 12, spec)
            np.testing.assert_allclose(batch.tail_factor[i], one.tail_factor, rtol=1e-13)
            np.testing.assert_allclose(batch.beyond_cdf[i], one.beyond_cdf, rtol=1e-13)
            np.testing.assert_allclose(batch.factors[i], one.factors, rtol=1e-13)
    constant = apply_tail(f, 12, TailSpec("constant", factor=1.05, attach_lag=60))
    assert constant.factors.shape == f.shape
    assert (constant.tail_factor == constant.tail_factor[0]).all()


def test_a_failed_row_is_flagged_not_answered():
    f = np.array(
        [[2.0, 1.5, 1.2, 1.1, 1.05], [1.05, 1.06, 1.07, 1.08, 1.09], [2.0, 1.0, 1.0, 1.0, 1.0]]
    )
    fit = apply_tail(f, 12, TailSpec("exponential"), on_error="flag")
    assert fit.ok.tolist() == [True, False, False]
    assert np.isfinite(fit.tail_factor[0]) and np.isnan(fit.tail_factor[1:]).all()
    with pytest.raises(Refusal, match=r"in simulation row 2 of 3") as caught:
        apply_tail(f, 12, TailSpec("exponential"))
    assert caught.value.reason == "tail_not_decaying"


# -- 6. every option delivered ----------------------------------------------------

#: (option, the value without it, the value with it, what it must change)
DELIVERY = [
    ("tail", {}, {"tail": "exponential"}, "ultimate"),
    (
        "tail_factor",
        {"tail": "constant", "tail_factor": 1.05},
        {"tail": "constant", "tail_factor": 1.1},
        "ultimate",
    ),
    (
        "tail_decay",
        {"tail": "constant", "tail_factor": 1.05, "tail_attach_lag": 72},
        {"tail": "constant", "tail_factor": 1.05, "tail_attach_lag": 72, "tail_decay": 0.9},
        "ultimate",
    ),
    (
        "tail_attach_lag",
        {"tail": "constant", "tail_factor": 1.05},
        {"tail": "constant", "tail_factor": 1.05, "tail_attach_lag": 96},
        "ultimate",
    ),
    (
        "tail_fit_lags",
        {"tail": "weibull"},
        {"tail": "weibull", "tail_fit_lags": (24, 84)},
        "ultimate",
    ),
    (
        "tail_steps",
        {"tail": "inverse_power"},
        {"tail": "inverse_power", "tail_steps": 1000},
        "ultimate",
    ),
    ("tail_rows", {"tail": "exponential"}, {"tail": "exponential", "tail_rows": 4}, "development"),
]


#: every (method, option) pair; mack refuses an attachment before the last age
#: (tested below), so the options that attach early are left out for mack rather
#: than skipped, because the CI job with every extra allows no such skip
DELIVERY_CASES = [
    pytest.param(method, *row, id=f"{row[0]}-{method}")
    for row in DELIVERY
    for method in METHODS
    if not (method == "mack" and "tail_attach_lag" in row[2])
]


@pytest.mark.parametrize(("method", "option", "without", "with_", "changes"), DELIVERY_CASES)
def test_each_tail_option_changes_what_it_should(method, option, without, with_, changes):
    before, after = call(method, **without), call(method, **with_)
    if changes == "ultimate":
        assert not np.array_equal(column(before, "ultimate"), column(after, "ultimate"))
    else:
        # the rows shown change, and nothing else does
        assert after.development.num_rows != before.development.num_rows
        assert after.origins.equals(before.origins)
        assert after.totals.equals(before.totals)


@pytest.mark.parametrize("option", ["tail_sigma", "tail_std_err"])
def test_each_mack_tail_variance_is_delivered(option):
    base = {"tail": "constant", "tail_factor": 1.05}
    before = methods.mack(RAA, **base)
    after = methods.mack(RAA, **base, **{option: 0.5})
    assert total(after, "mack_se") != total(before, "mack_se")
    assert total(after, option) == 0.5
    # the one given is used; the other is still read off the line
    other = "tail_std_err" if option == "tail_sigma" else "tail_sigma"
    assert total(after, other) == total(before, other)
    assert after.totals["tail_position"][0].as_py() == total(before, "tail_position")


def test_the_alpha_reaches_the_tail_step():
    """The tail step's process term is divided by the amount at the last age to the
    power alpha: with the tail sigma given, the tail's share of the variance moves
    with the average."""
    spec = {"tail": "constant", "tail_factor": 1.05, "tail_sigma": 2.0, "tail_std_err": 0.0}
    for average in ("simple", "regression"):
        plain = methods.mack(RAA, average=average)
        tailed = methods.mack(RAA, average=average, **spec)
        extra = (
            column(tailed, "process_se") ** 2 / column(tailed, "ultimate") ** 2
            - column(plain, "process_se") ** 2 / column(plain, "ultimate") ** 2
        )
        alpha = {"simple": 0, "regression": 2}[average]
        amount = column(plain, "ultimate")
        np.testing.assert_allclose(extra, 4.0 / 1.05**2 / amount**alpha, rtol=1e-6)


@pytest.mark.parametrize("method", ["chain_ladder", "mack"])
@pytest.mark.parametrize(
    "tail",
    [{"tail": "constant", "tail_factor": 1.05}, {"tail": "exponential"}],
    ids=["constant", "exponential"],
)
def test_a_tail_attached_at_the_last_age_is_the_default(method, tail):
    """Mack's refusal of an earlier attachment sends the caller to the last age,
    120 months on raa, so that age has to be accepted and change nothing."""
    default = call(method, **tail)
    last = call(method, **tail, tail_attach_lag=120)
    for table in ("origins", "development", "totals"):
        assert getattr(last, table).equals(getattr(default, table)), table


def test_tail_decay_reaches_macks_development_table():
    """With the tail at the last age the decay spreads the tail over the rows
    shown and moves no ultimate, so only the development table can show it."""
    base = {"tail": "constant", "tail_factor": 1.05, "tail_rows": 3}
    before = methods.mack(RAA, **base)
    after = methods.mack(RAA, **base, tail_decay=0.9)
    assert not after.development.equals(before.development)
    assert after.origins.equals(before.origins)
    assert after.totals.equals(before.totals)
    # the rows are the chain ladder's for the same tail options
    for decay in (None, 0.9):
        mack = methods.mack(RAA, **base, tail_decay=decay)
        chain_ladder = methods.chain_ladder(RAA, **base, tail_decay=decay)
        for name in ("dev_lag", "factor", "cdf", "pct_reported", "source"):
            assert mack.development[name].equals(chain_ladder.development[name]), (decay, name)


#: A value each tail option accepts when there is a tail, so the only fault in
#: the call below is that there is none.
_TAIL_VALUES = {
    "tail_factor": 1.05,
    "tail_decay": 0.5,
    "tail_attach_lag": 120,
    "tail_fit_lags": (24, 84),
    "tail_steps": 5,
    "tail_rows": 2,
    "tail_sigma": 0.1,
    "tail_std_err": 0.01,
}


def _options_without_a_tail() -> list[tuple[str, str]]:
    pairs = []
    for method in METHODS:
        accepted = inspect.signature(getattr(methods, method)).parameters
        pairs += [(method, name) for name in methods._TAIL_OPTIONS if name in accepted]
    return pairs


def test_every_tail_option_is_taken_by_some_method():
    taken = {name for _, name in _options_without_a_tail()}
    assert taken == set(methods._TAIL_OPTIONS) == set(_TAIL_VALUES)


@pytest.mark.parametrize(("method", "option"), _options_without_a_tail())
def test_a_tail_option_without_a_tail_is_refused_not_ignored(method, option):
    with pytest.raises(Refusal, match=f"^{option} was given but tail is None") as caught:
        call(method, **{option: _TAIL_VALUES[option]})
    assert caught.value.reason == "invalid_option"
    assert caught.value.option == option


@pytest.mark.parametrize("method", [m for m in METHODS if m != "mack"])
@pytest.mark.parametrize("option", ["tail_sigma", "tail_std_err"])
def test_macks_tail_variances_are_not_options_of_the_point_methods(method, option):
    """They are not in the point methods' signatures, so Python refuses them
    before ibnr sees them, as it does ``sigma_rule``: a TypeError, not a Refusal."""
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{option}'"):
        call(method, tail="constant", tail_factor=1.05, **{option: 0.1})


# -- 7. refusals, the one-year result, the simulations, the codec -----------------


@pytest.mark.parametrize(
    ("options", "reason", "option", "phrase"),
    [
        ({"tail": 1.05}, "invalid_option", "tail", "a constant tail's factor goes in tail_factor"),
        ({"tail": "gamma"}, "invalid_option", "tail", "'constant', 'exponential'"),
        (
            {"tail_decay": 0.5},
            "invalid_option",
            "tail_decay",
            "tail_decay was given but tail is None",
        ),
        ({"tail_rows": 2}, "invalid_option", "tail_rows", "tail_rows was given but tail is None"),
        (
            {"tail": "constant"},
            "invalid_option",
            "tail_factor",
            "a constant tail needs tail_factor",
        ),
        (
            {"tail": "constant", "tail_factor": True},
            "invalid_option",
            "tail_factor",
            "a constant tail needs tail_factor",
        ),
        (
            {"tail": "constant", "tail_factor": float("inf")},
            "invalid_option",
            "tail_factor",
            "positive finite",
        ),
        (
            {"tail": "constant", "tail_factor": 0.0},
            "invalid_option",
            "tail_factor",
            "positive finite",
        ),
        (
            {"tail": "exponential", "tail_factor": 1.05},
            "invalid_option",
            "tail_factor",
            "tail_factor is a constant-tail setting",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_steps": 5},
            "invalid_option",
            "tail_steps",
            "tail_steps is a curve setting",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_fit_lags": (12, 60)},
            "invalid_option",
            "tail_fit_lags",
            "tail_fit_lags is a curve setting",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_fit_lags": (None, None)},
            "invalid_option",
            "tail_fit_lags",
            "tail_fit_lags is a curve setting",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_decay": 1.5},
            "invalid_option",
            "tail_decay",
            "tail_decay must be between 0 and 1",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_attach_lag": 100},
            "grain_mismatch",
            "tail_attach_lag",
            r"tail_attach_lag 100 is not a development age of this triangle \(12, 24, ... 120\)",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_attach_lag": 240},
            "not_in_triangle",
            "tail_attach_lag",
            "is after the last observed age 120",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_attach_lag": 12.5},
            "invalid_option",
            "tail_attach_lag",
            "whole months",
        ),
        (
            {"tail": "exponential", "tail_fit_lags": (30, None)},
            "grain_mismatch",
            "tail_fit_lags",
            "not a development age",
        ),
        (
            {"tail": "exponential", "tail_fit_lags": (24, 120)},
            "not_in_triangle",
            "tail_fit_lags",
            "no link develops from that age",
        ),
        (
            {"tail": "exponential", "tail_fit_lags": (84, 24)},
            "invalid_option",
            "tail_fit_lags",
            "starts after it ends",
        ),
        (
            {"tail": "exponential", "tail_fit_lags": 24},
            "invalid_option",
            "tail_fit_lags",
            r"a pair \(first, last\)",
        ),
        (
            {"tail": "exponential", "tail_fit_lags": (108, None)},
            "not_identified",
            "tail",
            "at 108 months only 1 of the 1 factor is above",
        ),
        (
            {"tail": "exponential", "tail_steps": 0},
            "invalid_option",
            "tail_steps",
            "a whole number from 1 to 10000",
        ),
        (
            {"tail": "exponential", "tail_rows": -1},
            "invalid_option",
            "tail_rows",
            "a whole number from 0",
        ),
        (
            {"tail": "exponential", "tail_steps": 1, "tail_rows": 2},
            "invalid_option",
            "tail_rows",
            "tail_rows=2 asks for 2 rows beyond the triangle, but tail_steps=1 extrapolates 1 step",
        ),
    ],
)
@pytest.mark.parametrize("method", METHODS)
def test_each_tail_option_is_refused_by_name(method, options, reason, option, phrase):
    with pytest.raises(Refusal, match=phrase) as caught:
        call(method, **options)
    assert caught.value.reason == reason
    assert caught.value.option == option
    assert caught.value.method == method


def _growing() -> pa.Table:
    """A staircase whose link factors grow with age: 1.05, 1.06, ... 1.09."""
    factors = np.array([1.05, 1.06, 1.07, 1.08, 1.09])
    matrix = np.full((6, 6), np.nan)
    for i in range(6):
        amount = 100.0 + 10 * i
        for j in range(6 - i):
            matrix[i, j] = amount
            if j < 5:
                amount *= factors[j] * (1 + 0.001 * ((i + j) % 3))
    return cells_of(rows_of(matrix))


@pytest.mark.parametrize("kind", CURVES)
def test_a_curve_that_grows_is_refused(kind):
    with pytest.raises(Refusal, match="does not decay") as caught:
        methods.chain_ladder(_growing(), tail=kind)
    assert caught.value.reason == "tail_not_decaying"


def test_an_inverse_power_slope_between_minus_one_and_zero_is_refused():
    t = np.arange(1, 10, dtype=float)
    f = 1 + 0.3 * t**-0.6
    with pytest.raises(Refusal, match="between -1 and 0, so the product") as caught:
        apply_tail(f, 12, TailSpec("inverse_power"))
    assert caught.value.reason == "tail_not_decaying"
    assert np.isfinite(apply_tail(1 + 0.3 * t**-1.6, 12, TailSpec("inverse_power")).tail_factor)


def test_every_factor_at_or_below_the_threshold_is_refused_not_answered_as_no_tail():
    f = np.array([1.5, 1.2, 1.000005, 1.0, 0.99])
    with pytest.raises(Refusal, match="only 1 of the 4 factors is above") as caught:
        apply_tail(f[1:], 12, TailSpec("weibull"))
    assert caught.value.reason == "not_identified"
    fit = apply_tail(f, 12, TailSpec("exponential"))
    assert fit.in_fit.tolist() == [True, True, False, False, False]


def test_a_factor_of_exactly_the_threshold_is_left_out_of_the_curve():
    f = np.array([1.5, 1.2, 1.1, 1.05, MIN_FIT_FACTOR])
    fit = apply_tail(f, 12, TailSpec("exponential"))
    assert fit.in_fit.tolist() == [True, True, True, True, False]


def test_as_many_rows_as_steps_answers():
    result = methods.chain_ladder(RAA, tail="exponential", tail_steps=3, tail_rows=3)
    assert result.development.num_rows == 13
    assert result.development["cdf"][-1].as_py() == 1.0
    assert total(result, "tail_factor") > 1


def test_the_tail_position_line_keeps_a_factor_just_above_one():
    """R's position line goes through every factor above 1, the ones the curve
    fit leaves out (at or below 1.00001) included."""
    f = np.array([2.0, 1.5, 1.2, 1.1, 1.000005])
    assert f[-1] < MIN_FIT_FACTOR
    sigma2 = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    s = np.array([100.0, 120.0, 130.0, 140.0, 150.0])
    got = tail_variance(f, sigma2, s, 1.03)
    t = np.arange(1.0, 6.0)
    b, a = np.polyfit(t, np.log(f - 1), 1)
    assert got.position == pytest.approx((np.log(0.03) - a) / b, rel=1e-12)


def test_a_constant_tail_the_decay_cannot_spread_is_refused():
    with pytest.raises(Refusal, match="cannot be spread over its steps") as caught:
        methods.chain_ladder(RAA, tail="constant", tail_factor=0.5, tail_decay=0.0)
    assert caught.value.reason == "result_not_finite"
    # with one row holding the whole tail there is nothing to spread
    one = methods.chain_ladder(RAA, tail="constant", tail_factor=0.5, tail_decay=0.0, tail_rows=0)
    assert total(one, "tail_factor") == 0.5


@pytest.mark.parametrize(
    ("options", "reason", "phrase"),
    [
        (
            {"tail": "exponential", "tail_attach_lag": 72},
            "not_supported",
            "(?i)mack cannot attach a tail before the last observed age",
        ),
        (
            {"tail": "constant", "tail_factor": 0.95},
            "variance_not_estimable",
            "below 1.+pass tail_sigma and tail_std_err",
        ),
        (
            {"tail": "constant", "tail_factor": 0.95, "tail_sigma": 0.1},
            "variance_not_estimable",
            "below 1",
        ),
        (
            {"tail": "constant", "tail_factor": 3.0},
            "variance_not_estimable",
            "extrapolated backwards",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_sigma": -1.0},
            "invalid_option",
            "tail_sigma must be zero or a positive finite number",
        ),
        (
            {"tail": "constant", "tail_factor": 1.05, "tail_std_err": float("nan")},
            "invalid_option",
            "tail_std_err must be zero or a positive finite number",
        ),
        ({"tail_sigma": 0.1}, "invalid_option", "tail_sigma was given but tail is None"),
    ],
)
def test_macks_tail_refusals(options, reason, phrase):
    with pytest.raises(Refusal, match=phrase) as caught:
        methods.mack(RAA, **options)
    assert caught.value.reason == reason


def test_a_tail_below_one_moves_ultimates_and_mack_takes_it_with_both_variances():
    below = methods.chain_ladder(RAA, tail="constant", tail_factor=0.95)
    assert total(below, "ultimate") == pytest.approx(202_466.12, abs=0.005)
    result = methods.mack(RAA, tail="constant", tail_factor=0.95, tail_sigma=0.5, tail_std_err=0.01)
    assert total(result, "ultimate") == total(below, "ultimate")
    assert total(result, "mack_se") > 0


def test_mack_refuses_too_few_factors_above_one_for_the_tail_line():
    f = np.array([1.5, 0.98, 0.99])
    with pytest.raises(Refusal, match="needs two factors above 1; 1 of the 3") as caught:
        tail_variance(f, np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), 1.05)
    assert caught.value.reason == "variance_not_estimable"
    with pytest.raises(Refusal, match="needs two positive ones; 1 of the 3"):
        tail_variance(np.array([1.5, 1.2, 1.1]), np.array([1.0, 0.0, 0.0]), np.ones(3), 1.05)


def test_the_kernel_candidate_refuses_mack_settings_and_a_horizon():
    with pytest.raises(Refusal, match="sigma is a Mack setting"):
        ConventionalCandidate(tail=TailSpec("exponential", sigma=0.1))
    with pytest.raises(Refusal, match="horizon fixes the last age") as caught:
        ConventionalCandidate(tail=TailSpec("exponential"), horizon=120)
    assert caught.value.reason == "not_supported"
    with pytest.raises(Refusal, match="TailSpec or None"):
        ConventionalCandidate(tail="exponential")


def test_a_kernel_tail_is_named_in_kernel_terms():
    with pytest.raises(Refusal, match=r"^attach_lag 100 is not") as caught:
        fit_conventional_grid(
            grid_of(PUBLIC["raa"]),
            ConventionalCandidate(tail=TailSpec("constant", factor=1.05, attach_lag=100)),
        )
    assert caught.value.option == "attach_lag"


def test_the_kernel_fit_carries_the_tail():
    grid = grid_of(PUBLIC["raa"])
    fit = fit_conventional_grid(grid, ConventionalCandidate(tail=TailSpec("exponential")))
    assert fit.tail is not None
    assert fit.beta[-1] == pytest.approx(1 / float(fit.tail.tail_factor), rel=1e-15)
    assert fit.factor_summary["tail"].tolist() == [False] * 9
    early = fit_conventional_grid(
        grid, ConventionalCandidate(tail=TailSpec("exponential", attach_lag=84))
    )
    assert early.factor_summary["tail"].tolist() == [False] * 6 + [True] * 3
    # predict_cumulative at the last age is the amount there, not the ultimate
    last = fit.predict_cumulative(dt.date(1990, 1, 1), 120)
    ultimate = fit.origins.loc[fit.origins["origin_period"] == dt.date(1990, 1, 1), "ultimate"]
    assert last * float(fit.tail.tail_factor) == pytest.approx(float(ultimate.iloc[0]), rel=1e-12)


def _tailed_fit() -> MackFit:
    return fit_mack_grid(grid_of(PUBLIC["raa"]), tail=TailSpec("constant", factor=1.05))


@pytest.mark.parametrize("decoded", [False, True], ids=["fitted", "decoded"])
def test_the_one_year_result_and_the_simulations_refuse_a_tailed_fit(decoded):
    fit = _tailed_fit()
    if decoded:
        fit = MackFit.from_arrow(fit.to_arrow())
    one_year = "the one-year claims development result has no tail"
    for thunk in (
        lambda: one_year_cdr(fit),
        lambda: simulate_one_year_cdr(fit, n_draws=10, seed=1),
        lambda: simulate_one_year_cdr(fit, generator="odp_bootstrap", n_draws=10, seed=1),
        lambda: rereserve(fit, np.tile(fit.latest, (2, 1))),
        lambda: MackDiagonal().check(fit),
        lambda: ODPBootstrapDiagonal().check(fit),
    ):
        with pytest.raises(Refusal, match=one_year) as caught:
            thunk()
        assert caught.value.reason == "not_supported"
    with pytest.raises(Refusal, match="simulate_ultimates has no tail step yet") as caught:
        simulate_ultimates(fit, n_draws=10, seed=1)
    assert caught.value.reason == "not_supported"
    from ibnr.kernels.holdout import CellIndex

    cells = CellIndex(
        w=np.array([9]),
        d=np.array([2]),
        value=np.array([2100.0]),
        prev_value=np.array([2063.0]),
        premium=np.array([1.0]),
    )
    with pytest.raises(Refusal, match="draw_next_cells has no tail step yet"):
        draw_next_cells(fit, cells, rng=np.random.default_rng(1), n_draws=5)


class _Consulted(DiagonalGenerator):
    """A generator that accepts any fit and records that it was asked."""

    def __init__(self):
        self.calls = []

    def check(self, fit):
        self.calls.append("check")

    def draw(self, fit, *, n_draws, rng):
        self.calls.append("draw")
        return np.tile(fit.latest, (2, 1))


def test_the_simulated_one_year_result_refuses_a_tail_before_any_generator_is_asked():
    """The refusal is simulate_one_year_cdr's own, not the generator's: a
    generator that takes any fit is never consulted."""
    generator = _Consulted()
    with pytest.raises(Refusal, match="the one-year claims development result has no tail"):
        simulate_one_year_cdr(_tailed_fit(), generator=generator, n_draws=2, seed=1)
    assert generator.calls == []


def test_the_codec_carries_the_tail():
    for spec in (
        TailSpec("constant", factor=1.05, decay=0.7, rows=3),
        TailSpec("exponential", fit_lags=(24, None), steps=50, rows=2),
        TailSpec("constant", factor=1.05, sigma=0.1, std_err=0.02),
    ):
        fit = fit_mack_grid(grid_of(PUBLIC["raa"]), tail=spec, average="simple", links=LinkRules())
        payload = fit.to_arrow()
        back = MackFit.from_arrow(payload)
        assert back.tail == spec
        for name in ("tail_factor", "tail_sigma2", "tail_se2", "tail_position"):
            assert getattr(back, name) == getattr(fit, name), name
        assert back.msep_runoff()["msep_total"] == fit.msep_runoff()["msep_total"]
        assert np.array_equal(back.ultimate, fit.ultimate)
    import pyarrow.ipc as ipc

    version = ipc.open_stream(payload).schema.metadata[b"ibnr.version"]
    assert int(version) == 3 == CODEC_VERSION
    untailed = fit_mack_grid(grid_of(PUBLIC["raa"]))
    assert ipc.open_stream(untailed.to_arrow()).schema.metadata[b"ibnr.version"] == b"1"
    assert from_arrow(untailed.to_arrow()).tail is None


def test_a_tampered_tail_is_refused():
    fit = _tailed_fit()
    with pytest.raises(Refusal, match="without a tail has tail_factor 1.0"):
        dataclasses.replace(fit, tail=None)
    for factor in (float("nan"), 0.0, -1.05):
        with pytest.raises(Refusal, match="positive finite float"):
            dataclasses.replace(fit, tail_factor=factor)


# -- the example workbook's triangle, from the mart -------------------------------

PUBLISH = "20260613_041006"
SOURCE = f"github://EKtheSage/cas-schedule-p-data-model@{PUBLISH}"


def _mart_cached() -> bool:
    try:
        from ibnr.data.schedule_p import active_mart_path

        return active_mart_path(SOURCE).exists()
    except Exception:
        return False


@pytest.mark.mart
@pytest.mark.skipif(not _mart_cached(), reason=f"Schedule P publish {PUBLISH} is not reachable")
@pytest.mark.parametrize(
    ("tail", "ibnr", "mack_se"),
    [
        ({}, 373_346.30, 10_938.84),
        ({"tail": "constant", "tail_factor": 1.05}, 464_776.81, 11_606.89),
        ({"tail": "exponential"}, 420_724.94, 11_239.91),
        ({"tail": "inverse_power"}, 797_871.60, 21_456.55),
        ({"tail": "weibull"}, 447_427.26, 11_447.48),
        # chainladder answers Mack here (11,162.46) with curve factors in place of
        # link ratios and their standard errors kept; mack refuses it
        ({"tail": "exponential", "tail_attach_lag": 108}, 408_806.60, None),
    ],
)
def test_the_example_workbooks_tails(tail, ibnr, mack_se):
    """New Jersey Manufacturers (NAIC 7080), workers' compensation paid, as of 1997:
    the Reserving app's chain-ladder reserve and Mack standard error for each tail."""
    import pandas as pd

    from ibnr.data.schedule_p import load_schedule_p

    triangle = load_schedule_p(SOURCE, companies=["7080"], lines=["workers_compensation"])
    frame = triangle.as_of(dt.date(1997, 12, 31)).execute()
    frame["year"] = pd.to_datetime(frame["origin_period"]).dt.year
    paid = frame[(frame["field"] == "paid_loss") & frame["year"].between(1988, 1997)]
    cells = pa.table(
        {
            "origin_period": paid["year"].tolist(),
            "dev_lag": paid["dev_lag"].astype(int).tolist(),
            "value": paid["value"].astype(float).tolist(),
        }
    )
    assert total(methods.chain_ladder(cells, **tail), "ibnr") == pytest.approx(ibnr, abs=0.005)
    if mack_se is None:
        with pytest.raises(Refusal, match="cannot attach a tail before the last observed age"):
            methods.mack(cells, **tail)
    else:
        assert total(methods.mack(cells, **tail), "mack_se") == pytest.approx(mack_se, abs=0.005)
