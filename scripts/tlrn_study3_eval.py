"""The third tlrn study's evaluator: does the refit help, and is the model effective?

Reads the fits that ``scripts/tlrn_study_colab.py --study companion_final`` saved and scores
them at four valuation dates, each on its own company set, against the chain ladder (``mack``)
and the multivariate chain ladder (``mcl``) fitted by ibnr's own entries on exactly the data
known at each date and scored on exactly the cells the network is scored on. It needs no jax
and no accelerator, only ibnr, the cas-schedule-p package and the saved folder. The design and
its reasons are in ``knowledge/decisions/tlrn-study3-preregistration.md``, which this
docstring repeats; the second study's evaluator is ``scripts/tlrn_study2_eval.py`` and its
benchmarks, scoring window and interval code are reused here.

PRE-REGISTERED, written before any fit of this study existed
============================================================

Two co-primary questions, each judged on each cohort by a written rule. P1: does retraining
each network on everything known up to the cutoff (the refit) lower the company-reserve error
compared with the network as it stood at its chosen epoch? The two arms come from one training
run: ``tlrn_cf_refit`` (the blended refit forecast, ``predicted`` in the saved table) and
``tlrn_cf_norefit`` (the same members before the refit with the same alpha,
``predicted_selected``). P2: is the refit model more accurate than ``mack`` and ``mcl``? The
second study's rule, unchanged, applied to ``tlrn_cf_refit``. Secondary and never part of P1 or
P2: the network against the network-off row ``tlrn_cf_factors``, the blend against the raw refit
network, replication of the companion's reported figures, the blend weight alpha.

Cohorts and company sets. B3: accident years 1998 to 2007, last known diagonal 2007-12-31.
C3: accident years 1988 to 1997, 1997-12-31, the fresh cohort. At valuation ``v`` in 6, 7, 8
and 10 the company set ``S_v`` is ``select_pairs(cells, cutoff=v)``: a complete 10-by-10 square
with positive premium on every cell, then on the cells known at ``v`` positive paid and premium
and a cumulative paid loss ratio below 3, then at least two such lines per company. The
networks of date ``v`` are trained on ``S_v`` only. The forecasts of date ``v`` are scored on
``S_v`` and on the cells of the second study (diagonals after ``v``, development lag at most
``v``, and at valuation 10 every cell past the last diagonal). ``mack`` and ``mcl`` are fitted
and scored on the same ``S_v``. Expected pairs 230, 226, 219 and 212 on B3 and 237, 236, 232 and
232 on C3; companies 90, 89, 87 and 84 on B3. If ibnr's rule returns other counts the study
uses ibnr's counts and the evaluator reports the difference. The dominant company (the largest
actual reserve at valuation 10, found in code) must be in every ``S_v`` or the evaluator stops.
A company that either benchmark cannot answer for at a date it belongs to is left out of every
date and every row and named in the output (the reading of "all of the second study's
definitions are kept").

Draws. Ten independent fits per cohort and row (draw ``k`` has fit seed ``11 + 100003 k``).
A draw set that is not exactly draws 0 to 9 makes the verdict provisional, printed with the
draw count and the missing numbers. Every member of a registered (non-smoke) fit must have run
3,000 epochs and its refit must have run to the epoch the member chose, or the evaluator stops.

Statistics. A company's error at a valuation is predicted minus actual over the scored cells.
Pool_APE at ``v`` is the sum of the absolute errors over the companies of ``S_v`` divided by the
sum of their actuals. The draw average at ``v`` is the mean over draws of each company's
prediction. The dollar-pooled Pool_APE over valuations 6, 7, 8 and 10 is the sum, over the four
valuations and the companies of each date's set, of the absolute company error divided by the
sum of the company actuals over the same company-date pairs. The unweighted mean over the four
valuations is reported and takes part in no rule, except the descriptive label in P1. The
bootstrap draws 10,000 resamples of companies with replacement from the union of the four sets
(seed 2008). A company drawn ``n`` times counts ``n`` times at every date at which it is in the
date's set and not at all at a date where it is not; one resample serves the model and the
comparison and every valuation. The interval is the 2.5th to 97.5th percentile of the
difference. The stratified sensitivity bootstrap keeps the dominant company in every resample
exactly once (seed 2009), is reported beside the primary one for P1 and P2 and replaces
nothing. "Wholly below zero" means the upper end is below zero; "wholly above" means the lower
end is above zero.

The rule for P1. ``D(v)`` is the draw-average Pool_APE of ``tlrn_cf_refit`` minus that of
``tlrn_cf_norefit`` (negative favours the refit), ``D10`` is ``D(10)`` and ``Dpool`` the same
difference of the dollar-pooled statistic. Per cohort, ``helps`` needs all four: (a) the
interval of ``D10`` is wholly below zero; (b) the interval of ``Dpool`` is wholly below zero;
(c) in at least 8 of the 10 draws, the draw's own valuation-10 Pool_APE with the refit is
below its Pool_APE without it; (d) with the dominant company left out, the point value of
``D10`` is at or below zero. ``hurts`` if the interval of ``D10`` or of ``Dpool`` is wholly
above zero. ``not distinguishable`` otherwise. A descriptive label that changes nothing:
``date-dependent`` if the point value of ``D10`` is below zero and the unweighted mean over the
four valuations of ``D(v)`` is above zero. Across cohorts, C3 is the headline: C3 helps and B3
helps gives "the refit helps on the fresh cohort and on the companion's"; C3 helps and B3 does
not gives "the refit helps on the fresh cohort only; B3 is listed"; C3 not distinguishable
gives "not distinguishable", with B3 listed; C3 hurts gives "the refit hurts on the fresh
cohort". If the refit helps on B3 and is not distinguishable or hurts on C3, the statement says
the benefit is not shown beyond the cohort the companion's design was chosen on. A cohort not
run, or with fewer than the registered draws, is labelled provisional, and a cohort not run is
never counted as hurting.

The rule for P2. The second study's rule, with its amendments, applied to ``tlrn_cf_refit``
against ``mack`` and ``mcl`` on each cohort and the per-date sets. ``effective`` needs all four:
(i) at valuation 10 the Pool_APE difference (model minus benchmark) has a 95% interval wholly
below zero against both benchmarks; (ii) the dollar-pooled difference over valuations 6, 7, 8
and 10 has a 95% interval wholly below zero against both; (iii) the median over the draws of
the valuation-10 Pool_APE is below both benchmarks; (iv) without the dominant company, either
the valuation-10 point difference is at or below zero against both benchmarks, or condition (i)
holds without it. ``worse`` if the interval in (i) or (ii) is wholly above zero against either
benchmark. ``not distinguishable`` otherwise. Overall: C3 effective and B3 effective gives
"effective on the fresh cohort, confirmed on the companion's"; C3 effective and B3 not worse
gives "effective on the fresh cohort; on the companion's cohort not worse"; C3 effective and B3
worse gives "mixed", with no claim of effectiveness; C3 not distinguishable gives "not
distinguishable" whatever B3 says; C3 worse gives "worse on the fresh cohort". A verdict on B3
is confirmatory of a fixed specification, not out of sample. A verdict on C3 is out of sample in
accident years only, and weaker than the second study's C because that study's results were
known before this model's rules were written.

Secondary questions (reported, never in P1 or P2). The network: ``tlrn_cf_refit`` minus
``tlrn_cf_factors``; ``network adds value`` only if the valuation-10 and the dollar-pooled
intervals are both wholly below zero and, without the dominant company, the valuation-10
interval is not wholly above zero; ``network hurts`` if either of the first two is wholly above
zero; otherwise ``not distinguishable``. The blend: ``tlrn_cf_refit`` minus the raw refit
network, with the words ``blend adds value`` and ``blend hurts`` by the same conditions. The raw
refit network is also judged by the P2 rule as a secondary row. Also reported: alpha (minimum,
quartiles, median, maximum, standard deviation) per valuation; Pool_PE; the median absolute
percentage error; the mean absolute error; the mean absolute log of predicted over actual
reserve; the share of companies where the model beats each benchmark; Pool_APE within size
quartiles; the draw-to-draw spread; the width of every interval and the gain it could just
detect (half the width). MAPE and the reserve-weighted root mean squared error are descriptive
only.

Replication of the companion's reported figures, on B3 at valuation 10, in percentage points
(the companion's single reported fit, not verified here). Let ``m`` be the draw average's
Pool_APE and ``[lo, hi]`` the range of the ten draws' own. R1: 3.93 lies in ``[lo, hi]`` and
``|m - 3.93| <= 0.5``. R2: ``D10 <= -0.5``. R3 (secondary): ``|m_net - 2.96| <= 0.5`` for the
raw refit network. ``replicates`` if R1 and R2 both hold; ``fails to replicate`` if neither
holds and ``|m - 3.93| > 1.0``; ``partially replicates`` otherwise. R3 does not change the
label. The benchmark side is a tie-out: at valuation 10 ``mack`` and ``mcl`` must equal the
second study's saved values on the same cohort (read from its evaluation folder); at
valuations 6, 7 and 8 the sets differ from the second study's, so ``mack`` is recomputed by an
independent chain ladder written in plain numpy and compared (``mcl`` has no such independent
recomputation here).

What is not claimed. Nothing about why a result comes out as it does; nothing about other
accident years, lines or companies; nothing from B3 as out-of-sample evidence; no multiplicity
correction; no claim that the bootstrap, which resamples companies only, measures uncertainty
about future years. One company holds about two thirds of the reserve at valuation 10, so every
dollar-based statistic depends on it.

Usage::

    python scripts/tlrn_study3_eval.py --fits DIR [--cohort B|C|all] [--out DIR] [--cache DIR]
        [--cas-dir DIR] [--study2-evaluation DIR] [--smoke] [--strict-data-hash]

A smoke folder (``--smoke``) is scored too, so the evaluator can be checked, and its output
carries a banner saying the numbers must never be quoted. When this file was written no fit of
the study existed and no result existed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tlrn_study2_eval as ev2
import tlrn_study_colab as study
import tlrn_study_eval as first

REPO = Path(__file__).resolve().parent.parent

VALUATIONS = (6, 7, 8, 10)
LAST = max(VALUATIONS)
LAST_I = VALUATIONS.index(LAST)
BENCHMARKS = ev2.BENCHMARKS
N_BOOT = first.N_BOOT
BOOT_SEED = first.BOOT_SEED
STRATIFIED_SEED = BOOT_SEED + 1
LEVEL = first.LEVEL
EXPECTED_EPOCHS = ev2.EXPECTED_EPOCHS

#: the rows of the runner, and the arms read from them
MAIN_ROW = "tlrn_cf_refit"
FACTORS_ROW = "tlrn_cf_factors"
ROWS = (MAIN_ROW, FACTORS_ROW)
REFIT = "tlrn_cf_refit"
NOREFIT = "tlrn_cf_norefit"
RAW_NETWORK = "raw_network"
RAW_NETWORK_NOREFIT = "raw_network_norefit"
FACTORS = "tlrn_cf_factors"
#: the column of the saved valuations table that holds each arm
ARM_COLUMN = {
    REFIT: "predicted",
    NOREFIT: "predicted_selected",
    RAW_NETWORK: "predicted_net",
    RAW_NETWORK_NOREFIT: "predicted_net_selected",
}

#: the pre-registration speaks in percentage points; statistics here are fractions
POINT = 0.01
#: the companion's relayed figures at valuation 10, in points (reported, not verified here)
COMPANION_BLEND = 3.93
COMPANION_NETWORK = 2.96
#: R1 and R2 thresholds and the failure distance, in points
R1_TOLERANCE = 0.5
R2_DIFFERENCE = -0.5
FAIL_DISTANCE = 1.0
R3_TOLERANCE = 0.5
#: P1 (c): at least this many of this many draws
P1_DRAWS_REQUIRED = 8
P1_DRAWS_OF = 10

P1_WORDS = ("helps", "not distinguishable", "hurts")
P2_WORDS = ev2.VERDICTS
NETWORK_WORDS = ev2.NETWORK_VERDICTS
BLEND_WORDS = ev2.BLEND_VERDICTS
REPLICATION_WORDS = ("replicates", "partially replicates", "fails to replicate")

#: pairs and companies the pre-registration expects at valuations 6, 7, 8 and 10
EXPECTED_PAIRS = {
    "B": {6: 230, 7: 226, 8: 219, 10: 212},
    "C": {6: 237, 7: 236, 8: 232, 10: 232},
}
EXPECTED_COMPANIES = {"B": {6: 90, 7: 89, 8: 87, 10: 84}}

CONFIRMATORY = "confirmatory of a fixed specification, not out-of-sample"
OUT_OF_SAMPLE = (
    "out-of-sample in accident years only (weaker than the second study's C, whose results "
    "were known before this model's rules were written; companies and calendar years overlap B3)"
)
STATUS = {"B": CONFIRMATORY, "C": OUT_OF_SAMPLE}

REFIT_OVERALL = {
    "none": "no overall statement: cohort C3 has no finished fits",
    "both": "the refit helps on the fresh cohort and on the companion's",
    "fresh_only": "the refit helps on the fresh cohort only; B3 is listed",
    "nd": "not distinguishable",
    "hurts": "the refit hurts on the fresh cohort",
    "not_shown": (
        "the benefit is not shown beyond the cohort the companion's design was chosen on"
    ),
}
EFFECTIVE_OVERALL = {
    "none": "no overall statement: cohort C3 has no finished fits",
    "worse": "worse on the fresh cohort",
    "nd": "not distinguishable",
    "confirmed": "effective on the fresh cohort, confirmed on the companion's",
    "not_worse": "effective on the fresh cohort; on the companion's cohort not worse",
    "mixed": (
        "mixed: effective on the fresh cohort, worse on the companion's; "
        "no overall claim of effectiveness"
    ),
}


# -- the decision rules, as pure functions of numbers -------------------------------------


def _below(i: dict) -> bool:
    return bool(np.isfinite(i["high"]) and i["high"] < 0)


def _above(i: dict) -> bool:
    return bool(np.isfinite(i["low"]) and i["low"] > 0)


def required_draws(n_draws: int) -> int:
    """Draws that must favour the refit in P1 (c): 8 of 10, scaled up for a smaller set."""
    return -(-P1_DRAWS_REQUIRED * n_draws // P1_DRAWS_OF)


def decide_refit(stats: dict) -> tuple[str, dict, bool, str]:
    """P1: ``(word, conditions, date_dependent, reason)`` from numbers.

    ``stats`` has ``last`` and ``pooled`` (``low``, ``high``: the interval of the refit minus
    no-refit Pool_APE difference at valuation 10 and dollar-pooled), ``draws_better`` (how many
    draws have a smaller valuation-10 Pool_APE with the refit) out of ``n_draws``,
    ``excluding_diff`` (the valuation-10 point difference without the dominant company), ``d10``
    (the valuation-10 point difference) and ``rolling_diff`` (the unweighted mean over the four
    valuations of the point difference). Differences are fractions; negative favours the refit.
    """
    conditions = {
        "a_last_below": _below(stats["last"]),
        "b_pooled_below": _below(stats["pooled"]),
        "c_draws_favour": stats["draws_better"] >= required_draws(stats["n_draws"]),
        "d_excluding_at_or_below": bool(
            np.isfinite(stats["excluding_diff"]) and stats["excluding_diff"] <= 0
        ),
    }
    if all(conditions.values()):
        word = "helps"
    elif _above(stats["last"]) or _above(stats["pooled"]):
        word = "hurts"
    else:
        word = "not distinguishable"
    date_dependent = bool(stats["d10"] < 0 and stats["rolling_diff"] > 0)
    reason = (
        f"valuation 10 [{stats['last']['low']:+.4%}, {stats['last']['high']:+.4%}], pooled "
        f"[{stats['pooled']['low']:+.4%}, {stats['pooled']['high']:+.4%}], "
        f"{stats['draws_better']} of {stats['n_draws']} draws favour the refit (need "
        f"{required_draws(stats['n_draws'])}), valuation 10 without the dominant company "
        f"{stats['excluding_diff']:+.4%}; conditions "
        + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in conditions.items())
    )
    return word, conditions, date_dependent, reason


def decide_secondary(stats: dict, words: tuple[str, str, str]) -> tuple[str, dict, str]:
    """The network and the blend rule: valuation 10 AND dollar-pooled, not the unweighted mean.

    ``stats`` has ``last``, ``pooled`` and ``excluding`` (each ``low``, ``high``). ``words`` is
    ``(adds value, hurts, not distinguishable)``.
    """
    conditions = {
        "a_last_below": _below(stats["last"]),
        "b_pooled_below": _below(stats["pooled"]),
        "c_excluding_not_above": bool(
            np.isfinite(stats["excluding"]["low"]) and not _above(stats["excluding"])
        ),
    }
    if all(conditions.values()):
        word = words[0]
    elif _above(stats["last"]) or _above(stats["pooled"]):
        word = words[1]
    else:
        word = words[2]
    reason = (
        "; ".join(
            f"{name} [{stats[key]['low']:+.4%}, {stats[key]['high']:+.4%}]"
            for name, key in (
                ("valuation 10", "last"),
                ("pooled", "pooled"),
                ("valuation 10 without the dominant company", "excluding"),
            )
        )
        + "; conditions "
        + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in conditions.items())
    )
    return word, conditions, reason


def _within(distance: float, bound: float) -> bool:
    """``distance <= bound`` with the float noise of a subtraction rounded away."""
    return round(distance - bound, 9) <= 0


def replication_label(m: float, lo: float, hi: float, d10: float, m_net: float | None) -> dict:
    """R1 to R3 and the label, from percentage points (not fractions).

    ``m`` is the draw average's valuation-10 Pool_APE, ``lo`` and ``hi`` the range of the
    draws' own, ``d10`` the refit minus no-refit difference at valuation 10 and ``m_net`` the
    raw refit network's draw average (or None).
    """
    r1 = bool(
        _within(abs(m - COMPANION_BLEND), R1_TOLERANCE)
        and _within(lo - COMPANION_BLEND, 0.0)
        and _within(COMPANION_BLEND - hi, 0.0)
    )
    r2 = bool(_within(d10 - R2_DIFFERENCE, 0.0))
    r3 = None if m_net is None else bool(_within(abs(m_net - COMPANION_NETWORK), R3_TOLERANCE))
    far = not _within(abs(m - COMPANION_BLEND), FAIL_DISTANCE)
    if r1 and r2:
        label = REPLICATION_WORDS[0]
    elif not r1 and not r2 and far:
        label = REPLICATION_WORDS[2]
    else:
        label = REPLICATION_WORDS[1]
    return {"r1": r1, "r2": r2, "r3": r3, "label": label, "m": m, "lo": lo, "hi": hi, "d10": d10}


def refit_overall_statement(c3, b3, provisional: dict | None = None) -> str:
    """P1 across cohorts from the two verdict words (``None`` for a cohort not run)."""
    if c3 is None:
        return REFIT_OVERALL["none"]
    if c3 == "hurts":
        text = REFIT_OVERALL["hurts"]
    elif c3 == "not distinguishable":
        text = REFIT_OVERALL["nd"]
    elif b3 == "helps":
        text = REFIT_OVERALL["both"]
    else:
        text = REFIT_OVERALL["fresh_only"]
    if b3 == "helps" and c3 != "helps":
        text += "; " + REFIT_OVERALL["not_shown"]
    text += f" (C3 {c3}; B3 {b3 if b3 is not None else 'not run'})"
    return _provisional(text, provisional)


def effective_overall_statement(c3, b3, provisional: dict | None = None) -> str:
    """P2 across cohorts from the two verdict words (``None`` for a cohort not run)."""
    if c3 is None:
        return EFFECTIVE_OVERALL["none"]
    if c3 == "not distinguishable":
        text = EFFECTIVE_OVERALL["nd"]
    elif c3 == "worse":
        text = EFFECTIVE_OVERALL["worse"]
    elif b3 == "worse":
        text = EFFECTIVE_OVERALL["mixed"]
    elif b3 == "effective":
        text = EFFECTIVE_OVERALL["confirmed"]
    else:
        text = EFFECTIVE_OVERALL["not_worse"]
    text += f" (C3 {c3}; B3 {b3 if b3 is not None else 'not run'})"
    return _provisional(text, provisional)


def _provisional(text: str, provisional: dict | None) -> str:
    flagged = sorted(k for k, v in (provisional or {}).items() if v)
    if flagged:
        return f"provisional ({', '.join(flagged)} not exactly draws 0 to 9): {text}"
    return text


# -- companies and arrays: each date has its own set ---------------------------------------


def date_companies(sets: dict) -> dict[int, list[str]]:
    """``{v: sorted companies}`` of ``study.valuation_pairs``' ``{v: pairs}``."""
    return {v: sorted({c for c, _ in sets[v]}) for v in VALUATIONS}


def presence_mask(companies: list[str], by_date: dict[int, list[str]]) -> np.ndarray:
    """``(valuations, companies)`` booleans: the company is in the date's set."""
    return np.array([[c in set(by_date[v]) for c in companies] for v in VALUATIONS])


def wide(frame: pd.DataFrame, companies: list[str], column: str) -> np.ndarray:
    """``(valuations, companies)`` of one column of a long frame, NaN where not in the set."""
    table = frame.pivot(index="valuation", columns="company_code", values=column)
    return table.reindex(index=list(VALUATIONS), columns=companies).to_numpy(dtype=float)


def draw_arrays(fits: list, companies: list[str], column: str, mask: np.ndarray) -> np.ndarray:
    """``(draws, valuations, companies)`` of one saved column; finite wherever ``mask``."""
    out = np.stack([wide(f.valuations, companies, column) for f in fits])
    bad = ~np.isfinite(out[:, mask])
    if bad.any():
        raise SystemExit(f"a fit's valuations table has no finite {column} for some company")
    return out


# -- statistics with a company set per date -------------------------------------------------


def _parts(pred, actual, mask) -> tuple[np.ndarray, np.ndarray]:
    err = np.where(mask, np.abs(np.nan_to_num(pred) - np.nan_to_num(actual)), 0.0)
    act = np.where(mask, np.nan_to_num(actual), 0.0)
    return err, act


def pool_ape_v(pred, actual, mask) -> np.ndarray:
    """Pool_APE at each valuation over the companies of that date's set."""
    err, act = _parts(pred, actual, mask)
    return err.sum(axis=1) / act.sum(axis=1)


def pooled_ape(pred, actual, mask) -> float:
    """The dollar-pooled Pool_APE over the valuations and each date's companies."""
    err, act = _parts(pred, actual, mask)
    return float(err.sum() / act.sum())


def counts_of(idx: np.ndarray) -> np.ndarray:
    """``(resamples, companies)``: how many times each company is drawn in each resample."""
    b, n = idx.shape
    counts = np.zeros((b, n))
    np.add.at(counts, (np.arange(b)[:, None], idx), 1.0)
    return counts


def boot_pool(pred, actual, mask, counts) -> tuple[np.ndarray, np.ndarray]:
    """``(resamples, valuations)`` Pool_APE and ``(resamples,)`` dollar-pooled Pool_APE.

    A company drawn ``n`` times counts ``n`` times at each date whose set holds it and not at
    all at the others; a resample whose actuals add to zero has no value (NaN).
    """
    err, act = _parts(pred, actual, mask)
    top, bottom = counts @ err.T, counts @ act.T
    per_date = top / np.where(bottom != 0, bottom, np.nan)
    bottom_all = bottom.sum(axis=1)
    pooled = top.sum(axis=1) / np.where(bottom_all != 0, bottom_all, np.nan)
    return per_date, pooled


def difference_summary(pred, ref, actual, mask, counts) -> dict:
    """Pool_APE of ``pred`` minus ``ref`` at valuation 10, dollar-pooled and unweighted mean.

    Negative favours ``pred``. ``pooled`` is the rule's statistic, ``rolling`` the plain mean
    over the four valuations (descriptive), and ``by_valuation`` each date's own.
    """
    p_boot, p_pool = boot_pool(pred, actual, mask, counts)
    r_boot, r_pool = boot_pool(ref, actual, mask, counts)
    diff_v = p_boot - r_boot
    point_v = pool_ape_v(pred, actual, mask) - pool_ape_v(ref, actual, mask)
    cells = (
        ("last", point_v[LAST_I], diff_v[:, LAST_I]),
        ("pooled", pooled_ape(pred, actual, mask) - pooled_ape(ref, actual, mask), p_pool - r_pool),
        ("rolling", float(point_v.mean()), p_boot.mean(axis=1) - r_boot.mean(axis=1)),
    )
    out = {}
    for name, point, boot in cells:
        lo, hi = ev2.interval(boot)
        finite = boot[np.isfinite(boot)]
        out[name] = {
            "diff": float(point),
            "low": lo,
            "high": hi,
            "share_favour": float(np.mean(finite < 0)) if finite.size else float("nan"),
        }
    out["by_valuation"] = {
        v: dict(zip(("low", "high"), ev2.interval(diff_v[:, i]), strict=True))
        | {"diff": float(point_v[i])}
        for i, v in enumerate(VALUATIONS)
    }
    return out


def metric_table(series: dict, actual, mask, bench: dict) -> pd.DataFrame:
    """Every reported metric per method and valuation, the rolling mean and the pooled row."""
    rows = []
    for method, pred in series.items():
        per = []
        for i, v in enumerate(VALUATIONS):
            m = mask[i]
            row = ev2.point_metrics(pred[i][m], actual[i][m])
            for b in BENCHMARKS:
                if method != b:
                    row[f"share_better_than_{b}"] = ev2.point_metrics(
                        pred[i][m], actual[i][m], bench=bench[b][i][m]
                    )["share_better_than_benchmark"]
            per.append({"method": method, "valuation": str(v), **row})
        rows += per
        keys = [k for k in per[0] if k not in ("method", "valuation", "n", "n_log")]
        rows.append(
            {"method": method, "valuation": "rolling"}
            | {k: float(np.mean([r[k] for r in per])) for k in keys}
        )
        rows.append(
            {"method": method, "valuation": "pooled", "pool_ape": pooled_ape(pred, actual, mask)}
        )
    return pd.DataFrame(rows)


def quartile_table(series: dict, actual, mask) -> pd.DataFrame:
    """Pool_APE within quartiles of the actual reserve, per method and valuation."""
    rows = []
    for i, v in enumerate(VALUATIONS):
        m = mask[i]
        q = ev2.size_quartiles(actual[i][m])
        for k in (1, 2, 3, 4):
            inside = q == k
            for method, pred in series.items():
                a = actual[i][m][inside]
                rows.append(
                    {
                        "valuation": v,
                        "quartile": k,
                        "n": int(inside.sum()),
                        "actual_reserve": float(a.sum()),
                        "method": method,
                        "pool_ape": float(np.abs(pred[i][m][inside] - a).sum() / a.sum())
                        if a.sum() != 0
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


# -- the saved fits -------------------------------------------------------------------------


@dataclass
class Fit:
    """One saved fit of the third study: its settings and its tables."""

    draw: int
    seed: int
    settings: dict
    run: dict
    reserves: pd.DataFrame
    valuations: pd.DataFrame
    members: pd.DataFrame


def check_members(name: str, members: pd.DataFrame) -> None:
    """Refuse a registered fit in which a member did not run 3,000 epochs or its refit stopped
    anywhere but at the epoch the member chose."""
    for column in ("epochs_run", "best_epoch", "refit_epochs"):
        if column not in members.columns:
            raise SystemExit(f"{name}: the members table has no {column} column to check")
    short = members[members["epochs_run"] != EXPECTED_EPOCHS]
    if len(short):
        row = short.iloc[0]
        raise SystemExit(
            f"{name}: {len(short)} of {len(members)} members did not run {EXPECTED_EPOCHS} "
            f"epochs (for example valuation {row['valuation']} member {row['member']} ran "
            f"{row['epochs_run']}); this is not a registered fit"
        )
    off = members[members["refit_epochs"].astype(int) != members["best_epoch"].astype(int)]
    if len(off):
        row = off.iloc[0]
        raise SystemExit(
            f"{name}: the refit of {len(off)} members did not run to the epoch the member "
            f"chose (for example valuation {row['valuation']} member {row['member']}: "
            f"refit {row['refit_epochs']}, chosen {row['best_epoch']})"
        )


def load_fits(folder: Path, cohort: str, row: str, *, smoke: bool) -> list[Fit]:
    """Every finished fit of one cohort and row, in draw order, checked to be one study.

    A record counts only if it says it is the third study's (``study`` 3 and the name
    ``companion_final``). All fits must share every setting except the draw and its seed, each
    seed must be the one its draw implies, every table must be there, and (outside smoke) every
    member must have run its epochs and its refit to the chosen one. Two ibnr versions of one
    draw are refused rather than one of them picked.
    """
    spec = study.get_study("companion_final")
    found: dict[int, tuple[Path, str]] = {}
    for path in sorted(Path(folder).glob(f"{cohort}_{row}_d*.json")):
        m = ev2.RECORD_NAME.match(path.name)
        if m is None or m["cohort"] != cohort or m["row"] != row or bool(m["smoke"]) != smoke:
            continue
        settings = json.loads(path.read_text(encoding="utf-8")).get("settings", {})
        if settings.get("study") != spec.version or settings.get("study_name") != spec.name:
            continue
        draw = int(m["draw"])
        if draw in found:
            raise SystemExit(
                f"draw {draw} of cohort {cohort} {row} was saved by two ibnr versions "
                f"({found[draw][1]} and {m['version']}) in {folder}; keep one"
            )
        found[draw] = (path, m["version"])
    fits = []
    for draw in sorted(found):
        path, _ = found[draw]
        record = json.loads(path.read_text(encoding="utf-8"))
        settings = record["settings"]
        if settings["seed"] != study.draw_seed(settings["draw"]) or settings["draw"] != draw:
            raise SystemExit(f"{path.name}: the seed or draw in the record is not draw {draw}'s")
        reserves = path.with_name(path.stem + "_reserves.csv")
        tables = study.table_paths(path, spec.tables)
        for needed in (reserves, *tables.values()):
            if not needed.exists():
                raise SystemExit(f"{path.name} has no {needed.name}")
        read = {name: pd.read_csv(p, dtype={"company_code": str}) for name, p in tables.items()}
        if not smoke:
            check_members(path.name, read["members"])
        frame = pd.read_csv(reserves, dtype={"company_code": str}).set_index("company_code")
        fits.append(
            Fit(
                draw,
                settings["seed"],
                settings,
                record.get("run", {}),
                frame,
                read["valuations"],
                read["members"],
            )
        )
    if fits:
        reference = fits[0].settings
        for fit in fits[1:]:
            differ = study.settings_differences(
                {k: v for k, v in reference.items() if k not in ("draw", "seed")},
                {k: v for k, v in fit.settings.items() if k not in ("draw", "seed")},
            )
            if differ:
                raise SystemExit(
                    f"cohort {cohort} {row}: draw {fit.draw} was fitted with different settings "
                    f"from draw {fits[0].draw} (they differ in {', '.join(differ)}); a folder "
                    "must hold one study"
                )
    return fits


def check_fits_against_cohort(
    cohort, row: str, fits: list[Fit], sets: dict, *, strict: bool = False
) -> str:
    """The second study's check (cohort, data, study), plus the company set of every date.

    The hash of each date's pairs recorded in a fit must equal the evaluator's own, so a fit
    trained on other companies at any date is refused by name.
    """
    note = ev2.check_fits_against_cohort(
        cohort, row, fits, strict=strict, study_name="companion_final"
    )
    for fit in fits:
        want = {str(v): study.pairs_hash(sets[v]) for v in VALUATIONS}
        got = fit.settings.get("valuation_hashes")
        if got != want:
            differ = sorted(k for k in want if (got or {}).get(k) != want[k])
            raise SystemExit(
                f"cohort {cohort.name} {row} draw {fit.draw} was trained on other companies "
                f"at valuation(s) {', '.join(differ) or 'all'} than this evaluator selects"
            )
    return note


def check_tables_against_truth(cohort, row: str, fits: list[Fit], truth: pd.DataFrame) -> None:
    """Refuse a fit whose saved tables disagree with the cells this evaluator scores.

    Each valuation's companies must be exactly the date's set, with the evaluator's cell count;
    the saved actual (known before the last diagonal) must equal the evaluator's sum of the
    observed increments; the saved valuation-10 prediction must equal the stored reserve.
    """
    own = truth.set_index(["valuation", "company_code"])
    where = f"cohort {cohort.name} {row}"
    need = {"valuation", "company_code", "n_cells", "actual", *ARM_COLUMN.values()}
    for fit in fits:
        missing = need - set(fit.valuations.columns)
        if missing:
            raise SystemExit(f"{where} draw {fit.draw}: the valuations table lacks {missing}")
        tab = fit.valuations.set_index(["valuation", "company_code"])
        if set(tab.index) != set(own.index):
            extra = sorted(set(tab.index) ^ set(own.index))[:3]
            raise SystemExit(
                f"{where} draw {fit.draw}: the valuations table covers other (valuation, "
                f"company) pairs than the date's company sets, e.g. {extra}"
            )
        mine = own.reindex(tab.index)
        bad = tab["n_cells"].to_numpy() != mine["n_cells"].to_numpy()
        if bad.any():
            at = tab.index[bad][0]
            raise SystemExit(
                f"{where} draw {fit.draw}: the fit scored {tab.loc[at, 'n_cells']} cells for "
                f"(valuation, company) {at} and this evaluator scores {mine.loc[at, 'n_cells']}; "
                "the scoring windows differ"
            )
        known = tab.index.get_level_values("valuation") != LAST
        saved, ours = tab["actual"].to_numpy()[known], mine["actual"].to_numpy()[known]
        off = ~np.isclose(saved, ours, rtol=ev2.TABLE_RTOL, atol=ev2.TABLE_ATOL)
        if off.any():
            at = tab.index[known][off][0]
            raise SystemExit(
                f"{where} draw {fit.draw}: the saved actual for {at} is {tab.loc[at, 'actual']} "
                f"and the cells give {mine.loc[at, 'actual']}"
            )
        last = tab.xs(LAST, level="valuation")["predicted"]
        stored = fit.reserves["reserve"].reindex(last.index)
        if not np.allclose(last, stored, rtol=ev2.TABLE_RTOL, atol=ev2.TABLE_ATOL):
            raise SystemExit(
                f"{where} draw {fit.draw}: the valuation-{LAST} prediction differs from the "
                "reserve the fit stored"
            )


# -- the benchmarks and the selection -------------------------------------------------------


def cached_benchmark_cells(cohort, sets: dict, cache_dir: Path | None, *, log=print):
    """:func:`tlrn_study2_eval.benchmark_cells` on the per-date sets, kept in ``cache_dir``.

    The file name carries the cohort, its data hash and the hash of all the dates' pairs, so a
    cache made from other cells or other sets is never read.
    """
    union = sorted({p for v in VALUATIONS for p in sets[v]})
    if cache_dir is None:
        return ev2.benchmark_cells(cohort, VALUATIONS, log=log, pairs_by_valuation=sets)
    path = (
        Path(cache_dir) / f"benchmarks3_{cohort.name}_{cohort.data_hash}_"
        f"{study.pairs_hash(union)}.csv"
    )
    want = {(v, c) for v, companies in date_companies(sets).items() for c in companies}
    if path.exists():
        frame = pd.read_csv(path, dtype={"company_code": str})
        got = set(zip(frame["valuation"], frame["company_code"], strict=True))
        columns = {"valuation", "company_code", "mack", "mcl", "n_cells", "actual"}
        if set(frame.columns) == columns and got == want:
            log(f"  benchmarks read from {path}")
            return frame
    frame = ev2.benchmark_cells(cohort, VALUATIONS, log=log, pairs_by_valuation=sets)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def selection_report(cohort, sets: dict) -> tuple[pd.DataFrame, list[str]]:
    """Pairs and companies per date against the pre-registration, and any stop reason."""
    by_date = date_companies(sets)
    rows = []
    for v in VALUATIONS:
        rows.append(
            {
                "valuation": v,
                "pairs": len(sets[v]),
                "pairs_expected": EXPECTED_PAIRS.get(cohort.name, {}).get(v),
                "companies": len(by_date[v]),
                "companies_expected": EXPECTED_COMPANIES.get(cohort.name, {}).get(v),
            }
        )
    table = pd.DataFrame(rows)
    problems = []
    ordered = sorted(VALUATIONS)
    for early, late in zip(ordered, ordered[1:], strict=False):
        if not set(by_date[late]) <= set(by_date[early]):
            problems.append(f"the company set at valuation {late} is not inside that at {early}")
    return table, problems


def scored_companies(bench: pd.DataFrame, by_date: dict) -> tuple[list[str], dict]:
    """The companies both benchmarks answered for at every date they belong to, and who is out.

    A company that either benchmark cannot answer for at any of its dates is left out of every
    date and every row (``left_out`` names the benchmark and dates), so one set of companies is
    compared throughout.
    """
    union = sorted({c for v in VALUATIONS for c in by_date[v]})
    left_out, keep = {}, []
    for company in union:
        part = bench[bench["company_code"] == company]
        dates = sorted(v for v in VALUATIONS if company in set(by_date[v]))
        failed = {
            b: sorted(part.loc[part[b].isna(), "valuation"].astype(int).tolist())
            for b in BENCHMARKS
            if part[b].isna().any()
        }
        if failed or sorted(part["valuation"].astype(int).tolist()) != dates:
            left_out[company] = failed or {"rows": len(part)}
        else:
            keep.append(company)
    return keep, left_out


def independent_mack(cohort, sets: dict, by_date: dict, bench: pd.DataFrame) -> dict:
    """``mack`` recomputed by a volume-weighted chain ladder in plain numpy, per date.

    For every pair of each date's set: the cells with calendar diagonal at most ``v``, the
    volume-weighted link ratios, each origin projected from its latest known age, the
    increments over the scored cells added over lines. Compared company by company with the
    gallery entry's totals. Returns the largest relative difference per date.
    """
    out = {}
    for v in VALUATIONS:
        squares, n_w, n_d, _ = ev2.paid_squares(cohort.all_cells, sets[v])
        mask = ev2.scored_mask(v, n_w, n_d)
        calendar = np.arange(n_w)[:, None] + np.arange(n_d)[None, :] + 1
        totals: dict[str, float] = {}
        for (company, _), cum in squares.items():
            known = np.where(calendar <= v, cum, np.nan)
            factors = np.ones(n_d - 1)
            for d in range(n_d - 1):
                both = np.isfinite(known[:, d]) & np.isfinite(known[:, d + 1])
                den = known[both, d].sum()
                factors[d] = known[both, d + 1].sum() / den if den > 0 else 1.0
            full = known.copy()
            for w in range(n_w):
                if not np.isfinite(known[w]).any():
                    continue
                d0 = int(np.nonzero(np.isfinite(known[w]))[0][-1])
                for d in range(d0, n_d - 1):
                    full[w, d + 1] = full[w, d] * factors[d]
            inc = ev2.increments(full)[mask]
            totals[company] = totals.get(company, 0.0) + float(inc.sum())
        part = bench[bench["valuation"] == v].set_index("company_code")["mack"]
        gap = 0.0
        for company, mine in totals.items():
            theirs = part.get(company, np.nan)
            if np.isfinite(theirs) and abs(theirs) > 0:
                gap = max(gap, abs(mine / theirs - 1.0))
        out[v] = gap
    return out


#: how closely the independent chain ladder and the benchmarks must agree
RECOMPUTE_TOL = 1e-6
#: how closely valuation-10 benchmarks must equal the second study's saved values
TIE_OUT_TOL = 1e-8


def second_study_tie_out(cohort_name: str, result: dict, folder: Path | None) -> dict | str:
    """The valuation-10 Pool_APE of ``mack`` and ``mcl`` against the second study's saved ones.

    Read from its ``<cohort>_metrics_by_valuation.csv``. Only meaningful when every company of
    the date's set was scored; otherwise a note says so. A difference beyond rounding stops the
    evaluator: the sets at valuation 10 are the second study's cohort, so the benchmarks must
    be the same numbers.
    """
    path = None if folder is None else Path(folder) / f"{cohort_name}_metrics_by_valuation.csv"
    if path is None or not path.exists():
        return f"NOT TIED OUT: no {cohort_name}_metrics_by_valuation.csv in {folder}"
    if result["left_out"]:
        return "not compared: some companies are left out because a benchmark failed"
    table = pd.read_csv(path)
    out = {}
    for b in BENCHMARKS:
        row = table[(table["method"] == b) & (table["valuation"].astype(str) == str(LAST))]
        if row.empty:
            return f"NOT TIED OUT: {path.name} has no {b} row at valuation {LAST}"
        theirs = float(row["pool_ape"].iloc[0])
        mine = float(result["benchmark_pool_ape"][b][LAST_I])
        out[b] = {"ours": mine, "second_study": theirs, "relative": abs(mine / theirs - 1.0)}
        if out[b]["relative"] > TIE_OUT_TOL:
            raise SystemExit(
                f"{b} at valuation {LAST} on cohort {cohort_name} does not reproduce the second "
                f"study's saved Pool_APE: {mine} against {theirs}"
            )
    return out


# -- one cohort -----------------------------------------------------------------------------


def evaluate_cohort(cohort, fits_by_row: dict, bench: pd.DataFrame, sets: dict) -> dict:
    """Everything the third study reports for one cohort, from fits and benchmarks.

    ``fits_by_row`` holds the study's rows, ``bench`` is the benchmark frame, ``sets`` the
    ``{v: pairs}`` company sets. P1 and P2 are decided from the main row's draw averages, and
    only when it has fits.
    """
    by_date = date_companies(sets)
    companies, left_out = scored_companies(bench, by_date)
    if len(companies) < 3:
        raise SystemExit(f"cohort {cohort.name}: only {len(companies)} companies scored")
    by_kept = {v: [c for c in by_date[v] if c in set(companies)] for v in VALUATIONS}
    mask = presence_mask(companies, by_kept)
    actual = wide(bench, companies, "actual")
    bench_arrays = {b: wide(bench, companies, b) for b in BENCHMARKS}
    in_last = [c for c, m in zip(companies, mask[LAST_I], strict=True) if m]
    top, share = first.dominant_company(
        pd.Series(actual[LAST_I][mask[LAST_I]], index=in_last, dtype=float)
    )
    t = companies.index(top)
    if not mask[:, t].all():
        raise SystemExit(
            f"cohort {cohort.name}: the dominant company {top} is not in every date's set, "
            "so the study stops (pre-registered)"
        )
    keep10 = mask[LAST_I].copy()
    keep10[t] = False
    counts = counts_of(first.bootstrap_indices(len(companies)))
    counts_strat = counts_of(ev2.stratified_indices(len(companies), t))
    idx_ex = first.bootstrap_indices(int(keep10.sum()))

    main_fits = fits_by_row.get(MAIN_ROW) or []
    factor_fits = fits_by_row.get(FACTORS_ROW) or []
    draws, draw_sets = {}, {}
    for arm, column in ARM_COLUMN.items():
        if main_fits:
            draws[arm] = draw_arrays(main_fits, companies, column, mask)
            draw_sets[arm] = [f.draw for f in main_fits]
    if factor_fits:
        draws[FACTORS] = draw_arrays(factor_fits, companies, "predicted", mask)
        draw_sets[FACTORS] = [f.draw for f in factor_fits]
    series = {b: bench_arrays[b] for b in BENCHMARKS}
    series |= {arm: d.mean(axis=0) for arm, d in draws.items()}

    result: dict = {
        "cohort": cohort.name,
        "status": STATUS.get(cohort.name, ""),
        "companies": companies,
        "mask": mask,
        "left_out": left_out,
        "dominant_company": top,
        "dominant_share": share,
        "n_cells_last": int(bench.loc[bench["valuation"] == LAST, "n_cells"].sum()),
        "draw_sets": draw_sets,
        "actual": actual,
        "series": series,
        "draws": draws,
        "benchmark_pool_ape": {b: pool_ape_v(bench_arrays[b], actual, mask) for b in BENCHMARKS},
    }

    def versus(arm, ref, counts_used):
        return difference_summary(series[arm], series[ref], actual, mask, counts_used)

    def excluding(arm, ref):
        return ev2.last_difference(
            series[arm][LAST_I][keep10],
            series[ref][LAST_I][keep10],
            actual[LAST_I][keep10],
            idx_ex,
        )

    # P2 and the raw refit network as a secondary row
    def p2(arm):
        stats_by = {b: versus(arm, b, counts) for b in BENCHMARKS}
        strat_by = {b: versus(arm, b, counts_strat) for b in BENCHMARKS}
        per_draw = np.array([pool_ape_v(d, actual, mask) for d in draws[arm]])
        stats = {
            "last": {b: stats_by[b]["last"] for b in BENCHMARKS},
            "pooled": {b: stats_by[b]["pooled"] for b in BENCHMARKS},
            "excluding": {b: excluding(arm, b) for b in BENCHMARKS},
            "median_draw_last": float(np.median(per_draw[:, LAST_I])),
            "pool_last": {b: float(result["benchmark_pool_ape"][b][LAST_I]) for b in BENCHMARKS},
        }
        word, conditions, reason = ev2.decide(stats)
        sens_word, _, sens_reason = ev2.decide(
            {
                **stats,
                "last": {b: strat_by[b]["last"] for b in BENCHMARKS},
                "pooled": {b: strat_by[b]["pooled"] for b in BENCHMARKS},
            }
        )
        return {
            "word": word,
            "conditions": conditions,
            "reason": reason,
            "stats": stats,
            "differences": stats_by,
            "stratified": strat_by,
            "sensitivity_word": sens_word,
            "sensitivity_reason": sens_reason,
        }

    def completeness(arms):
        notes, provisional, missing = [], False, []
        for arm in arms:
            prov, miss = first.draw_completeness(draw_sets.get(arm, []))
            if prov:
                provisional = True
                missing = miss
                notes.append(f"{arm} has {len(draw_sets.get(arm, []))} of {study.DEFAULT_DRAWS}")
        return provisional, missing, notes

    if main_fits:
        primary = p2(REFIT)
        provisional, missing, notes = completeness([REFIT])
        label = primary["word"]
        reason = primary["reason"]
        if provisional:
            note = "provisional: " + "; ".join(notes)
            label, reason = f"{label} ({note})", f"{note}; {reason}"
        result["p2"] = {**primary, "label": label, "reason": reason, "provisional": provisional}
        result["p2_missing_draws"] = missing
        result["p2_raw_network"] = p2(RAW_NETWORK)

        # P1: the refit minus no refit, one training run
        d = versus(REFIT, NOREFIT, counts)
        d_strat = versus(REFIT, NOREFIT, counts_strat)
        ex = excluding(REFIT, NOREFIT)
        per_refit = np.array([pool_ape_v(x, actual, mask) for x in draws[REFIT]])
        per_norefit = np.array([pool_ape_v(x, actual, mask) for x in draws[NOREFIT]])
        better = int((per_refit[:, LAST_I] < per_norefit[:, LAST_I]).sum())
        p1_stats = {
            "last": d["last"],
            "pooled": d["pooled"],
            "draws_better": better,
            "n_draws": len(main_fits),
            "excluding_diff": ex["diff"],
            "d10": d["last"]["diff"],
            "rolling_diff": d["rolling"]["diff"],
        }
        word, conditions, date_dependent, p1_reason = decide_refit(p1_stats)
        sens_word, _, _, _ = decide_refit(
            {**p1_stats, "last": d_strat["last"], "pooled": d_strat["pooled"]}
        )
        label = word
        if provisional:
            label = f"{word} ({note})"
            p1_reason = f"{note}; {p1_reason}"
        result["p1"] = {
            "word": word,
            "label": label,
            "conditions": conditions,
            "date_dependent": date_dependent,
            "reason": p1_reason,
            "stats": p1_stats,
            "differences": d,
            "stratified": d_strat,
            "excluding": ex,
            "sensitivity_word": sens_word,
            "provisional": provisional,
            "per_draw_refit": per_refit,
            "per_draw_norefit": per_norefit,
        }
        # the blend: the refit blend against the raw refit network
        blend_d = versus(REFIT, RAW_NETWORK, counts)
        blend = decide_secondary(
            {
                "last": blend_d["last"],
                "pooled": blend_d["pooled"],
                "excluding": excluding(REFIT, RAW_NETWORK),
            },
            BLEND_WORDS,
        )
        result["blend"] = {"word": blend[0], "conditions": blend[1], "reason": blend[2]}
        result["blend"]["provisional"] = provisional
        # the replication check (B3 at valuation 10)
        if cohort.name == "B":
            m = float(pool_ape_v(series[REFIT], actual, mask)[LAST_I]) / POINT
            lo = float(per_refit[:, LAST_I].min()) / POINT
            hi = float(per_refit[:, LAST_I].max()) / POINT
            net = float(pool_ape_v(series[RAW_NETWORK], actual, mask)[LAST_I]) / POINT
            result["replication"] = replication_label(m, lo, hi, d["last"]["diff"] / POINT, net)
        result["alpha"] = ev2.alpha_spread(main_fits)
    if main_fits and factor_fits:
        net_d = versus(REFIT, FACTORS, counts)
        net = decide_secondary(
            {
                "last": net_d["last"],
                "pooled": net_d["pooled"],
                "excluding": excluding(REFIT, FACTORS),
            },
            NETWORK_WORDS,
        )
        prov, miss, notes = completeness([REFIT, FACTORS])
        result["network"] = {
            "word": net[0],
            "conditions": net[1],
            "reason": net[2],
            "provisional": prov,
            "differences": net_d,
        }

    # the descriptive tables
    differences_rows = []
    for arm in (REFIT, NOREFIT, RAW_NETWORK, FACTORS):
        if arm not in series:
            continue
        for b in BENCHMARKS:
            d = versus(arm, b, counts)
            for scope in ("last", "pooled", "rolling"):
                differences_rows.append(
                    {"method": arm, "benchmark": b, "statistic": scope, **d[scope]}
                )
            for v, cell in d["by_valuation"].items():
                differences_rows.append(
                    {
                        "method": arm,
                        "benchmark": b,
                        "statistic": f"valuation {v}",
                        **cell,
                        "share_favour": np.nan,
                    }
                )
    result["differences"] = pd.DataFrame(differences_rows)
    result["metrics"] = metric_table(series, actual, mask, bench_arrays)
    result["quartiles"] = quartile_table(series, actual, mask)
    precision = result["differences"][result["differences"]["method"] == REFIT]
    result["precision"] = ev2.precision_table(
        precision[precision["statistic"].isin(["last", "pooled", "rolling"])], "uniform"
    )
    summaries = {}
    for arm, d in draws.items():
        per = np.array([pool_ape_v(x, actual, mask) for x in d])
        summaries[arm] = {
            **{str(v): first.draw_summary(per[:, i]) for i, v in enumerate(VALUATIONS)},
            "rolling": first.draw_summary(per.mean(axis=1)),
        }
    result["draw_summary"] = summaries
    return result


def company_table(result: dict) -> pd.DataFrame:
    """One row per (valuation, company in that date's set): the actual and every series."""
    rows = []
    for i, v in enumerate(VALUATIONS):
        m = result["mask"][i]
        frame = pd.DataFrame(
            {
                "valuation": v,
                "company_code": [c for c, k in zip(result["companies"], m, strict=True) if k],
                "actual": result["actual"][i][m],
            }
        )
        for method, pred in result["series"].items():
            frame[method] = pred[i][m]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


# -- the report -----------------------------------------------------------------------------


def fmt_pct(x: float) -> str:
    return f"{100 * x:.3f}%"


def fmt_extra(value) -> str:
    return ev2.fmt_extra(value)


def render(cohort, result: dict, extras: dict, banner: str) -> str:
    """The text report of one cohort."""
    lines = [banner, ""] if banner else []
    lines.append(f"=== {cohort.description}")
    lines.append(
        f"status of any verdict on this cohort: {result['status']}; valuation {LAST} is "
        f"{cohort.as_of}"
    )
    per_date = [int(result["mask"][i].sum()) for i in range(len(VALUATIONS))]
    lines.append(
        f"scored companies per valuation {list(VALUATIONS)}: {per_date} (union "
        f"{len(result['companies'])}); cells scored at valuation {LAST}: {result['n_cells_last']}"
    )
    if result["left_out"]:
        lines.append(f"left out because a benchmark failed: {result['left_out']}")
    lines.append(
        f"dominant company {result['dominant_company']} holds {result['dominant_share']:.1%} of "
        f"the actual valuation-{LAST} reserve"
    )
    for key, value in extras.items():
        lines.append(f"{key}: {fmt_extra(value)}")
    pools = result["benchmark_pool_ape"]
    lines.append("benchmark Pool_APE by valuation " + str(list(VALUATIONS)) + ":")
    for b in BENCHMARKS:
        lines.append(
            f"   {b}: " + "  ".join(fmt_pct(x) for x in pools[b]) + f"  | rolling "
            f"{fmt_pct(pools[b].mean())}"
        )
    for arm, summary in result["draw_summary"].items():
        n = len(result["draw_sets"][arm])
        flag = ""
        if first.draw_completeness(result["draw_sets"][arm])[0]:
            flag = " (PROVISIONAL, not the registered 10 draws)"
        lines += ["", f"--- {arm}: {n} draws{flag}"]
        for key, s in summary.items():
            lines.append(
                f"   Pool_APE per draw at valuation {key}: min {fmt_pct(s['min'])}  median "
                f"{fmt_pct(s['median'])}  max {fmt_pct(s['max'])}  sd "
                f"{fmt_pct(s['sd']) if s['n'] > 1 else 'n/a'}"
            )
    lines += ["", "Pool_APE of the draw average, and the other reported metrics:"]
    lines.append(result["metrics"].round(5).to_string(index=False))
    lines += ["", "Difference in Pool_APE (row minus benchmark; negative favours the row):"]
    lines.append(result["differences"].round(5).to_string(index=False))
    lines += ["", "Pool_APE within quartiles of the actual reserve:"]
    lines.append(
        result["quartiles"]
        .pivot_table(index=["valuation", "quartile", "n"], columns="method", values="pool_ape")
        .round(5)
        .to_string()
    )
    if len(result["precision"]):
        lines += [
            "",
            "Precision: the width of each 95% interval and the gain it could just detect, about "
            "half the width, in Pool_APE (0.03 is three points):",
            result["precision"][["statistic", "benchmark", "width", "detectable_gain"]]
            .round(4)
            .to_string(index=False),
        ]
    if "p1" in result:
        p1 = result["p1"]
        d = p1["differences"]
        lines += [
            "",
            "P1, does the refit help? Refit minus no refit, same training run (negative favours "
            "the refit):",
            pd.DataFrame(
                [
                    {"statistic": k, **{x: d[k][x] for x in ("diff", "low", "high")}}
                    for k in ("last", "pooled", "rolling")
                ]
            )
            .round(5)
            .to_string(index=False),
            f"   valuation 10 without the dominant company: {p1['excluding']['diff']:+.5f} "
            f"[{p1['excluding']['low']:+.5f}, {p1['excluding']['high']:+.5f}]",
            f"   draws favouring the refit at valuation 10: "
            f"{p1['stats']['draws_better']} of {p1['stats']['n_draws']}",
            f"P1 for cohort {cohort.name} ({result['status']}): {p1['label'].upper()}"
            + (
                "  [date-dependent: better at valuation 10, worse on the unweighted mean]"
                if p1["date_dependent"]
                else ""
            ),
            f"   {p1['reason']}",
            f"   the same rule on the stratified intervals gives: {p1['sensitivity_word'].upper()}",
        ]
    if "p2" in result:
        p2 = result["p2"]
        lines += [
            "",
            f"P2 for cohort {cohort.name} ({result['status']}): {p2['label'].upper()}",
            f"   {p2['reason']}",
            f"   the same rule on the stratified intervals gives: {p2['sensitivity_word'].upper()}",
            f"   (secondary row) the raw refit network by the same rule: "
            f"{result['p2_raw_network']['word'].upper()}",
        ]
    for key, title in (("network", "NETWORK QUESTION"), ("blend", "BLEND QUESTION")):
        if key in result:
            lines += [f"{title} for cohort {cohort.name}: {result[key]['word'].upper()}"]
            lines.append(f"   {result[key]['reason']}")
    if "replication" in result:
        r = result["replication"]
        lines += [
            "",
            f"REPLICATION of the companion's reported figures (B3, valuation 10, points): "
            f"{r['label'].upper()}",
            f"   m {r['m']:.3f} range [{r['lo']:.3f}, {r['hi']:.3f}] against {COMPANION_BLEND}; "
            f"D10 {r['d10']:+.3f}; R1 {r['r1']}, R2 {r['r2']}, R3 {r['r3']} (the raw network "
            f"against {COMPANION_NETWORK})",
        ]
    if "alpha" in result:
        lines += ["", "Spread of the blend weight alpha over members and draws:"]
        lines.append(result["alpha"].round(4).to_string(index=False))
    return "\n".join(lines)


def run_cohort(
    name: str,
    fits_dir: Path,
    *,
    smoke: bool = False,
    cells=None,
    cas_dir: Path | None = None,
    cache_dir: Path | None = None,
    strict: bool = False,
    study2_evaluation: Path | None = None,
    log=print,
):
    """Load cohort ``name``, its fits and benchmarks, and evaluate. None when no fit exists."""
    if cells is None:
        cells = study.cells_from_mart(years=study.YEARS[name])
    cohort = study.load_cohort(name, cas_dir=cas_dir if name == "B" else None, cells=cells)
    log(cohort.description)
    sets = study.valuation_pairs(cohort)
    fits_by_row = {row: load_fits(fits_dir, name, row, smoke=smoke) for row in ROWS}
    if not any(fits_by_row.values()):
        log(f"cohort {name}: no finished fits in {fits_dir}, benchmarks not fitted, skipped")
        return None
    table, problems = selection_report(cohort, sets)
    if problems:
        raise SystemExit(f"cohort {name}: " + "; ".join(problems))
    extras: dict = {"selection against the pre-registration": "\n" + table.to_string(index=False)}
    truth = ev2.truth_table(cohort, VALUATIONS, sets)
    for row, fits in fits_by_row.items():
        if not fits:
            log(f"cohort {name} {row}: no finished fits in {fits_dir}")
            continue
        note = check_fits_against_cohort(cohort, row, fits, sets, strict=strict)
        if note:
            log(note)
            extras[f"data hash, {row}"] = note
        check_tables_against_truth(cohort, row, fits, truth)
    bench = cached_benchmark_cells(cohort, sets, cache_dir, log=log)
    result = evaluate_cohort(cohort, fits_by_row, bench, sets)
    gaps = independent_mack(cohort, sets, date_companies(sets), bench)
    extras["mack against an independent chain ladder, largest relative difference by valuation"] = {
        str(v): g for v, g in gaps.items()
    }
    worst = max(gaps.values())
    if worst > RECOMPUTE_TOL:
        raise SystemExit(
            f"cohort {name}: mack differs from the independent chain ladder by {worst:.3g} "
            f"(relative), more than {RECOMPUTE_TOL}; the benchmarks are not read"
        )
    extras["second study's valuation-10 benchmarks"] = second_study_tie_out(
        name, result, study2_evaluation
    )
    result["extras"] = extras
    result["sets"] = {v: len(sets[v]) for v in VALUATIONS}
    return cohort, result


def save_tables(out: Path, cohort, result: dict) -> None:
    """Write the cohort's tables as CSV files in ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    name = cohort.name
    result["metrics"].to_csv(out / f"{name}_metrics_by_valuation.csv", index=False)
    result["differences"].to_csv(out / f"{name}_differences.csv", index=False)
    result["quartiles"].to_csv(out / f"{name}_size_quartiles.csv", index=False)
    result["precision"].to_csv(out / f"{name}_precision.csv", index=False)
    company_table(result).to_csv(out / f"{name}_company_predictions.csv", index=False)
    if "alpha" in result:
        result["alpha"].to_csv(out / f"{name}_alpha_spread.csv", index=False)
    if "p1" in result:
        p1 = result["p1"]
        pd.DataFrame(
            {
                "draw": result["draw_sets"][REFIT],
                **{
                    f"{kind}_valuation_{v}": p1[f"per_draw_{kind}"][:, i]
                    for kind in ("refit", "norefit")
                    for i, v in enumerate(VALUATIONS)
                },
            }
        ).to_csv(out / f"{name}_draw_pool_ape_refit_norefit.csv", index=False)


def verdict_record(result: dict) -> dict:
    """The JSON-able verdicts of one cohort."""
    record = {
        "status": result["status"],
        "scored_companies": len(result["companies"]),
        "left_out": result["left_out"],
        "dominant_company": result["dominant_company"],
        "dominant_share": result["dominant_share"],
        "p1": None,
        "p2": None,
    }
    if "p1" in result:
        p1 = result["p1"]
        record["p1"] = {
            "verdict": p1["word"],
            "label": p1["label"],
            "date_dependent": p1["date_dependent"],
            "reason": p1["reason"],
            "provisional": p1["provisional"],
            "conditions": p1["conditions"],
            "sensitivity": p1["sensitivity_word"],
        }
        p2 = result["p2"]
        record["p2"] = {
            "verdict": p2["word"],
            "label": p2["label"],
            "reason": p2["reason"],
            "provisional": p2["provisional"],
            "conditions": p2["conditions"],
            "sensitivity": p2["sensitivity_word"],
            "raw_network": result["p2_raw_network"]["word"],
        }
        record["blend"] = result["blend"]["word"]
    if "network" in result:
        record["network"] = result["network"]["word"]
    if "replication" in result:
        record["replication"] = result["replication"]
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fits", type=Path, required=True, help="the runner's --out folder")
    parser.add_argument("--cohort", choices=("B", "C", "all"), default="all")
    parser.add_argument("--out", type=Path, default=None, help="write the report and tables here")
    parser.add_argument("--cache", type=Path, default=None, help="keep benchmark fits here")
    parser.add_argument("--cas-dir", type=Path, default=None, help="the four CAS files, cohort B")
    parser.add_argument(
        "--study2-evaluation",
        type=Path,
        default=REPO / "analysis" / "results" / "tlrn_study2" / "evaluation",
        help="the second study's evaluation folder, for the valuation-10 benchmark tie-out",
    )
    parser.add_argument("--strict-data-hash", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="score a smoke folder")
    args = parser.parse_args(argv)

    names = ["B", "C"] if args.cohort == "all" else [args.cohort]
    banner = (
        "SMOKE RUN. The fits were a few epochs and a few members. Nothing below may be quoted."
        if args.smoke
        else ""
    )
    if banner:
        print(banner)
    reports, verdicts = [], {}
    for name in names:
        t0 = time.perf_counter()
        cells = study.cells_from_mart(years=study.YEARS[name])
        got = run_cohort(
            name,
            args.fits,
            smoke=args.smoke,
            cells=cells,
            cas_dir=args.cas_dir,
            cache_dir=args.cache or args.out,
            strict=args.strict_data_hash,
            study2_evaluation=args.study2_evaluation,
        )
        if got is None:
            continue
        cohort, result = got
        report = render(cohort, result, result["extras"], banner)
        print(report)
        reports.append(report)
        verdicts[name] = verdict_record(result)
        print(f"(cohort {name} evaluated in {time.perf_counter() - t0:.0f}s)")
        if args.out is not None:
            save_tables(args.out, cohort, result)
    overall = None
    if args.cohort == "all":
        words = {
            k: {p: (v[p] or {}).get("verdict") if v.get(p) else None for p in ("p1", "p2")}
            for k, v in verdicts.items()
        }
        flags = {
            k: {
                p: bool((v[p] or {}).get("provisional")) if v.get(p) else True for p in ("p1", "p2")
            }
            for k, v in verdicts.items()
        }
        overall = {
            "p1": refit_overall_statement(
                words.get("C", {}).get("p1"),
                words.get("B", {}).get("p1"),
                {k: f["p1"] for k, f in flags.items()},
            ),
            "p2": effective_overall_statement(
                words.get("C", {}).get("p2"),
                words.get("B", {}).get("p2"),
                {k: f["p2"] for k, f in flags.items()},
            ),
        }
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.txt").write_text("\n\n".join(reports), encoding="utf-8")
        (args.out / "verdicts.json").write_text(
            json.dumps(
                {"smoke": args.smoke, "verdicts": verdicts, "overall": overall},
                indent=2,
                default=str,
            ),
            "utf-8",
        )
    print("\nVERDICTS" + (" (SMOKE, not a result)" if args.smoke else ""))
    for key, value in verdicts.items():
        if value["p1"]:
            print(
                f"  {key}: P1 {value['p1']['label']}; P2 {value['p2']['label']} [{value['status']}]"
            )
            print(
                f"     network question: {value.get('network')}; blend question: {value['blend']}"
            )
    if overall is not None:
        print(f"OVERALL P1: {overall['p1']}")
        print(f"OVERALL P2: {overall['p2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
