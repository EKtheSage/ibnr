"""The fourth tlrn study's evaluator: the companion's revised model, the refit, and the third study.

Reads the fits that ``scripts/tlrn_study_colab.py --study companion_revised`` saved and scores
them at four valuation dates, each on its own company set, against the chain ladder (``mack``)
and the multivariate chain ladder (``mcl``) fitted by ibnr's own entries on exactly the data
known at each date and scored on exactly the cells the network is scored on. It needs no jax
and no accelerator, only ibnr, the cas-schedule-p package and the saved folder (and, for the
descriptive comparison P0 and the rerun control, the third study's saved fits). The design and
its reasons are in ``knowledge/decisions/tlrn-study4-preregistration.md``, which this docstring
repeats; the third study's evaluator is ``scripts/tlrn_study3_eval.py`` and its decision rules,
bootstrap, benchmarks and scoring window are reused here unchanged.

PRE-REGISTERED, written before any fit of this study existed
============================================================

The model. The companion's revised model is the third study's model with one change: the
per-count ("lags observed") embedding is removed (``maturity_embedding=False``, 18,153
parameters against 18,537). The count still reaches the network as a feature channel. Nothing
else differs. The fourth study asks three things: P1 and P2 below, carried unchanged from the
third study, and P0, a new descriptive comparison with no verdict.

Two co-primary questions, each judged on each cohort by a written rule. P1: does retraining
each network on everything known up to the cutoff (the refit) lower the company-reserve error
compared with the network as it stood at its chosen epoch? The two arms come from one training
run: ``tlrn_cr_refit`` (the blended refit forecast, ``predicted`` in the saved table) and
``tlrn_cr_norefit`` (the same members before the refit with the same alpha,
``predicted_selected``). P2: is the refit model more accurate than ``mack`` and ``mcl``? The
third study's rule, unchanged, applied to ``tlrn_cr_refit``. Secondary and never part of P1 or
P2: the network against the network-off row ``tlrn_cr_factors``, the blend against the raw refit
network, replication of the companion's reported figures, the blend weight alpha.

Cohorts and company sets. B4: accident years 1998 to 2007, last known diagonal 2007-12-31, the
companion's main cohort. C4: accident years 1988 to 1997, 1997-12-31, by count the companion's
holdout cohort. At valuation ``v`` in 6, 7, 8 and 10 the company set ``S_v`` is
``select_pairs(cells, cutoff=v)``: a complete 10-by-10 square with positive premium on every
cell, then on the cells known at ``v`` positive paid and premium and a cumulative paid loss
ratio below 3, then at least two such lines per company. The networks of date ``v`` are trained
on ``S_v`` only. The forecasts of date ``v`` are scored on ``S_v`` and on the cells of the
third study (diagonals after ``v``, development lag at most ``v``, and at valuation 10 every
cell past the last diagonal). ``mack`` and ``mcl`` are fitted and scored on the same ``S_v``.
Expected pairs 230, 226, 219 and 212 on B4 and 237, 236, 232 and 232 on C4; companies 90, 89,
87 and 84 on B4 and 84, 84, 83 and 83 on C4. If ibnr's rule returns other counts the study uses
ibnr's counts and the evaluator reports the difference. The dominant company (the largest
actual reserve at valuation 10, found in code) must be in every ``S_v`` or the evaluator stops.
A company that either benchmark cannot answer for at a date it belongs to is left out of every
date and every row and named in the output. At valuation 10 ``mack`` and ``mcl`` must equal the
third study's saved values on the same cohort (mack 5.481 and mcl 5.410 percent on B4, mack
11.947 and mcl 4.969 percent on C4) to rounding, or the evaluator stops.

Draws. Ten independent fits per cohort and row (draw ``k`` has fit seed ``11 + 100003 k``, the
third study's arithmetic, so draw ``k`` is paired with the third study's draw ``k`` in its seeds
and its shared starting weights). A draw set that is not exactly draws 0 to 9 makes the verdict
provisional, printed with the draw count and the missing numbers. Every member of a registered
(non-smoke) fit must have run 3,000 epochs and its refit must have run to the epoch the member
chose, or the evaluator stops. A fit that failed is reported by name and its draw is missing.

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

The rule for P1. ``D(v)`` is the draw-average Pool_APE of ``tlrn_cr_refit`` minus that of
``tlrn_cr_norefit`` (negative favours the refit), ``D10`` is ``D(10)`` and ``Dpool`` the same
difference of the dollar-pooled statistic. Per cohort, ``helps`` needs all four: (a) the
interval of ``D10`` is wholly below zero; (b) the interval of ``Dpool`` is wholly below zero;
(c) in at least 8 of the 10 draws, the draw's own valuation-10 Pool_APE with the refit is
below its Pool_APE without it; (d) with the dominant company left out, the point value of
``D10`` is at or below zero. ``hurts`` if the interval of ``D10`` or of ``Dpool`` is wholly
above zero. ``not distinguishable`` otherwise. A descriptive label that changes nothing:
``date-dependent`` if the point value of ``D10`` is below zero and the unweighted mean over the
four valuations of ``D(v)`` is above zero. Across cohorts, C4 is the headline: C4 helps and B4
helps gives "the refit helps on the fresh cohort and on the companion's"; C4 helps and B4 does
not gives "the refit helps on the fresh cohort only; B4 is listed"; C4 not distinguishable
gives "not distinguishable", with B4 listed for information; C4 hurts gives "the refit hurts on
the fresh cohort". If the refit helps on B4 and is not distinguishable or hurts on C4, the
statement says the benefit is not shown beyond the cohort the companion's design was chosen on.
A cohort not run, or with fewer than the registered draws, is labelled provisional, and a
cohort not run is never counted as hurting.

The rule for P2. The third study's rule applied to ``tlrn_cr_refit`` against ``mack`` and
``mcl`` on each cohort and the per-date sets. ``effective`` needs all four: (i) at valuation 10
the Pool_APE difference (model minus benchmark) has a 95% interval wholly below zero against
both benchmarks; (ii) the dollar-pooled difference over valuations 6, 7, 8 and 10 has a 95%
interval wholly below zero against both; (iii) the median over the draws of the valuation-10
Pool_APE is below both benchmarks; (iv) without the dominant company, either the valuation-10
point difference is at or below zero against both benchmarks, or condition (i) holds without it.
``worse`` if the interval in (i) or (ii) is wholly above zero against either benchmark.
``not distinguishable`` otherwise. Overall: C4 effective and B4 effective gives "effective on
the fresh cohort, confirmed on the companion's"; C4 effective and B4 not worse gives "effective
on the fresh cohort; on the companion's cohort not worse"; C4 effective and B4 worse gives
"mixed", with no claim of effectiveness; C4 not distinguishable gives "not distinguishable"
whatever B4 says; C4 worse gives "worse on the fresh cohort". A verdict on B4 is confirmatory of
a fixed specification, not out of sample. A verdict on C4 is out of sample in accident years
only, and weaker than that: the third study's result on the same cohort existed before these
rules were written, the companion's holdout run is in the same commit as his embedding removal,
and C4 shares 60 of its 83 companies and 155 of its 232 pairs with B4.

P0, embedding removed minus kept (descriptive, no verdict, no threshold, no label). The fourth
study's saved fits minus the third study's, for the same cohort, row, draw, date and company.
Reported: per draw, the valuation-10 Pool_APE of each study's refit blend and the difference
(ten numbers per cohort) with the median and range of the ten differences; for the draw-averaged
forecasts, the difference in valuation-10 and dollar-pooled Pool_APE with the same company
bootstrap interval as P1 (seed 2008), with and without the dominant company, and the number of
companies whose absolute error is smaller in the fourth study; the same for the no-refit arm and
the raw network; and on B4 the companion's own move beside it for reading only (the blend at
valuation 10 went from 3.93 percent to 3.37, a fall of 0.56 points). P0 is computed only if the
per-date pair hashes of the two studies are equal on that cohort; if the third study's fits are
not available, P0 is not reported. Caveats written before any result: the pairing is in seeds
and starting weights, not in training (removing the table changes every gradient, and GPU fits
are not repeatable bit for bit); the rerun control ``tlrn_cf_rerun`` (one fit of the third
study's model on B4 at draw 0, on this study's image) gives one measure of what a rerun of an
unchanged specification moves, printed beside P0's draw 0 on B4, and one number is a scale and
not a distribution; a difference smaller than the draw-to-draw spread of the third study (0.53
points of Pool_APE at valuation 10 across ten refit-blend draws on B3, 1.95 on C3) is not
interpretable; the environments may differ and each record names its versions; P0 is a
statement about ibnr and not about the companion's change; there is no correction for
multiplicity and C4 overlaps B4.

Secondary questions (reported, never in P1 or P2). The network: ``tlrn_cr_refit`` minus
``tlrn_cr_factors``; ``network adds value`` only if the valuation-10 and the dollar-pooled
intervals are both wholly below zero and, without the dominant company, the valuation-10
interval is not wholly above zero; ``network hurts`` if either of the first two is wholly above
zero; otherwise ``not distinguishable``. The blend: ``tlrn_cr_refit`` minus the raw refit
network, with the words ``blend adds value`` and ``blend hurts`` by the same conditions. The raw
refit network is also judged by the P2 rule as a secondary row. Also reported: alpha (minimum,
quartiles, median, maximum, standard deviation) per valuation beside the companion's medians
(0.4618 on B4, 0.6617 on C4); Pool_PE; the median absolute percentage error; the mean absolute
error; the mean absolute log of predicted over actual reserve; the share of companies where the
model beats each benchmark; Pool_APE within size quartiles; the draw-to-draw spread; the width
of every interval and the gain it could just detect (half the width). All descriptive.

Replication of the companion's reported figures, at valuation 10, in percentage points, on each
cohort separately (the companion's single reported fit at his primary seed, not verified here).
Let ``m`` be the draw average's Pool_APE and ``[lo, hi]`` the range of the ten draws' own. On
B4, from his output files: R1 needs 3.37 inside ``[lo, hi]`` and ``|m - 3.37| <= 0.5``; R2 needs
``D10 <= -0.5`` (his difference is -1.00: 3.37 against 4.37 without the refit); R3 (secondary)
is ``|m_net - 4.31| <= 0.5`` for the raw refit network. On C4: R1c needs 5.75 inside
``[lo, hi]`` and ``|m - 5.75| <= 0.5``; R2c needs ``D10 >= +0.5`` (his difference is +1.02: 5.75
against 4.73, the refit hurts, and the rule asks for a hurt of at least half that size); R3c is
``|m_net - 6.29| <= 0.5``. The label per cohort is ``replicates`` if R1 and R2 both hold,
``fails to replicate`` if neither holds and ``|m - target| > 1.0``, and ``partially replicates``
otherwise. R3 is reported beside the label and does not change it. How a miss of R1 on B4 is
read, fixed before any result: (1) if ``m`` is within 0.5 of 3.37 and 3.37 is in the range of the
ten draws, R1 holds and the removal has moved ibnr's level in the direction and by about the
amount the companion's moved; (2) if ``m`` is between 3.87 and 4.37 the label is ``partially
replicates`` and the results note says that ibnr's level did not follow the companion's drop,
which is not read as a defect (candidate causes are listed and none is tested); (3) if ``m`` is
above 4.37 or below 2.37 and R2 also fails the label is ``fails to replicate`` and the evaluator
prints STOP AND ASK ETHAN: nothing is written as a result before he is asked; (4) whatever R1
says, P1 and P2 are decided by their own rules and a miss changes no verdict; (5) if R1 misses
and the P0 difference at valuation 10 on B4 is small compared with the third study's draw spread
(its absolute value no larger than the width of the range of the third study's ten refit-blend
draws), the sentence is "removing the embedding did not detectably move ibnr's level", which
says nothing about why the companion's moved. On C4 a draw average above 6.25 is unsurprising
because the target is the lowest of the companion's three seeds, and R2c is judged on the
direction and size of the refit effect. The companion's three seeds are printed beside ``m``
and change no label: B4 mean 3.62, range 3.37 to 3.88; C4 mean 6.96, range 5.75 to 7.67.

The variant rows (descriptive, no verdict rule, no interval compared with a threshold, no
statement that a component helps or hurts). ``tlrn_cr_noline`` (no cross-line attention),
``tlrn_cr_noay`` (no accident-year attention) and ``tlrn_cr_paid`` (the five incurred and case
reserve input channels set to zero, all 13 inputs kept) are one fit each on B4 at draw 0. The
evaluator prints, for each row and for the main row's draw 0, the valuation-10 Pool_APE and the
mean over the four valuations, for the blend and the raw network, beside the companion's means
over dates (main 4.40 and 4.71; no cross-line 4.10 and 4.47; no accident-year 3.76 for the
network and 4.36 for the blend; paid only 5.28 and 6.73). One draw has a spread of about half a
point at valuation 10, as large as the differences in that table, so the table says nothing
about which component matters. The control row ``tlrn_cf_rerun`` is printed in its own block. A
row that failed is named and delays nothing.

What is not claimed. Nothing about why a result comes out as it does; nothing about other
accident years, lines or companies; nothing from B4 as out-of-sample evidence; nothing from the
variant rows beyond what they print; no multiplicity correction; no claim that the bootstrap,
which resamples companies only, measures uncertainty about future years; no claim of precision.
One company holds about two thirds of the reserve at valuation 10, so every dollar-based
statistic depends on it.

The data hash. Each fit record names the versions of pandas, numpy and duckdb. The third study's
fit records held a data hash that differed from the one this evaluator builds from the same
publish, for a reason nobody could check. Here the tolerant check is kept: a data hash that
differs while the anchors, premiums, lines, companies and per-date pair hashes agree does not
stop the study. The recorded and the built hash are printed for every row whether or not they
match, and nothing in a rule depends on them.

Usage::

    python scripts/tlrn_study4_eval.py --fits DIR [--study3-fits DIR] [--cohort B|C|all]
        [--out DIR] [--cache DIR] [--cas-dir DIR] [--tie-out DIR] [--smoke]
        [--strict-data-hash]

A smoke folder (``--smoke``) is scored too, so the evaluator can be checked, and its output
carries a banner saying the numbers must never be quoted. When this file was written no fit of
the study existed and no result existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tlrn_study2_eval as ev2
import tlrn_study3_eval as ev3
import tlrn_study_colab as study
import tlrn_study_eval as first

REPO = Path(__file__).resolve().parent.parent

VALUATIONS = ev3.VALUATIONS
LAST = ev3.LAST
LAST_I = ev3.LAST_I
BENCHMARKS = ev3.BENCHMARKS
N_BOOT = ev3.N_BOOT
BOOT_SEED = ev3.BOOT_SEED
STRATIFIED_SEED = ev3.STRATIFIED_SEED
LEVEL = ev3.LEVEL
EXPECTED_EPOCHS = ev3.EXPECTED_EPOCHS
POINT = ev3.POINT

#: the rows of the runner and the arms read from them
MAIN_ROW = study.REVISED_MAIN_ROW
FACTORS_ROW = study.REVISED_FACTORS_ROW
VARIANT_ROWS = ("tlrn_cr_noline", "tlrn_cr_noay", "tlrn_cr_paid")
CONTROL_ROW = "tlrn_cf_rerun"
ROWS = study.COMPANION_REVISED_ROWS
REFIT = MAIN_ROW
NOREFIT = "tlrn_cr_norefit"
RAW_NETWORK = ev3.RAW_NETWORK
RAW_NETWORK_NOREFIT = ev3.RAW_NETWORK_NOREFIT
FACTORS = FACTORS_ROW
ARM_COLUMN = {
    REFIT: "predicted",
    NOREFIT: "predicted_selected",
    RAW_NETWORK: "predicted_net",
    RAW_NETWORK_NOREFIT: "predicted_net_selected",
}
#: the third study's row whose fits P0 reads, and the control's own source row
THIRD_MAIN_ROW = ev3.MAIN_ROW
#: the arms P0 compares, with the third study's column for each (the same columns)
P0_ARMS = (REFIT, NOREFIT, RAW_NETWORK)

#: the companion's figures at valuation 10, in percentage points, read from his output files at
#: commit b30ef02 (HEADLINE.csv and N_T2_seed_robustness.csv of the main and holdout runs)
COMPANION = {
    "B": {
        "blend": 3.37,
        "network": 4.31,
        "blend_no_refit": 4.37,
        "classical_best": 5.41,
        "median_alpha": 0.4618,
        "companies_per_date": (90, 89, 87, 84),
        "seeds": {"mean": 3.62, "min": 3.37, "max": 3.88},
    },
    "C": {
        "blend": 5.75,
        "network": 6.29,
        "blend_no_refit": 4.73,
        "classical_best": 4.97,
        "median_alpha": 0.6617,
        "companies_per_date": (84, 84, 83, 83),
        "seeds": {"mean": 6.96, "min": 5.75, "max": 7.67},
    },
}
#: the figure the third study replicated against, and the move the companion made since
COMPANION_EARLIER_BLEND_B = 3.93
#: the companion's means over the four dates, blend and network, for the variant rows
COMPANION_VARIANTS = {
    MAIN_ROW: {"blend": 4.40, "network": 4.71},
    "tlrn_cr_noline": {"blend": 4.10, "network": 4.47},
    "tlrn_cr_noay": {"blend": 4.36, "network": 3.76},
    "tlrn_cr_paid": {"blend": 5.28, "network": 6.73},
}
VARIANT_SETTING = {
    MAIN_ROW: "companion_revised",
    "tlrn_cr_noline": "no cross-line attention",
    "tlrn_cr_noay": "no accident-year attention",
    "tlrn_cr_paid": "paid only (five incurred and case channels zeroed)",
    CONTROL_ROW: "third study's model (control)",
}
#: the third study's draw spread at valuation 10 (points), the scale P0 is read against
THIRD_STUDY_SPREAD = {"B": 0.53, "C": 1.95}

R1_TOLERANCE = 0.5
R2_DIFFERENCE = 0.5
FAIL_DISTANCE = 1.0
R3_TOLERANCE = 0.5
#: R1 reading rule on B4: the edges of the bands, in points
READ_NEAR_LOW, READ_NEAR_HIGH = 3.87, 4.37
READ_FAR_LOW = 2.37
#: C4: a draw average above this is unsurprising (the target is the lowest of the three seeds)
C_UNSURPRISING = 6.25
P1_DRAWS_REQUIRED = ev3.P1_DRAWS_REQUIRED
P1_DRAWS_OF = ev3.P1_DRAWS_OF

P1_WORDS = ev3.P1_WORDS
P2_WORDS = ev3.P2_WORDS
NETWORK_WORDS = ev3.NETWORK_WORDS
BLEND_WORDS = ev3.BLEND_WORDS
REPLICATION_WORDS = ev3.REPLICATION_WORDS

EXPECTED_PAIRS = study.EXPECTED_PAIRS_BY_DATE
EXPECTED_COMPANIES = {
    "B": {6: 90, 7: 89, 8: 87, 10: 84},
    "C": {6: 84, 7: 84, 8: 83, 10: 83},
}

CONFIRMATORY = "confirmatory of a fixed specification, not out-of-sample"
OUT_OF_SAMPLE = (
    "out-of-sample in accident years only, and weaker than that (the third study's result on "
    "this cohort existed before these rules were written, the companion's holdout run is in "
    "the same commit as his embedding removal, and it shares 60 of its 83 companies and 155 of "
    "its 232 pairs with B4)"
)
STATUS = {"B": CONFIRMATORY, "C": OUT_OF_SAMPLE}

REFIT_OVERALL = {
    "none": "no overall statement: cohort C4 has no finished fits",
    "both": "the refit helps on the fresh cohort and on the companion's",
    "fresh_only": "the refit helps on the fresh cohort only; B4 is listed",
    "nd": "not distinguishable; B4 is listed for information",
    "hurts": "the refit hurts on the fresh cohort",
    "not_shown": (
        "the benefit is not shown beyond the cohort the companion's design was chosen on"
    ),
}
EFFECTIVE_OVERALL = {
    "none": "no overall statement: cohort C4 has no finished fits",
    "worse": "worse on the fresh cohort",
    "nd": "not distinguishable",
    "confirmed": "effective on the fresh cohort, confirmed on the companion's",
    "not_worse": "effective on the fresh cohort; on the companion's cohort not worse",
    "mixed": (
        "mixed: effective on the fresh cohort, worse on the companion's; "
        "no overall claim of effectiveness"
    ),
}
STOP_AND_ASK = (
    "STOP AND ASK ETHAN: the draw average is more than 1.0 point from the companion's figure and "
    "the refit effect is not shown. The pre-registration says to ask before writing any result."
)

#: what the evaluator exits with when it held the results back for Ethan's answer
STOP_EXIT_CODE = 3

#: the decision rules are the third study's, unchanged
decide_refit = ev3.decide_refit
decide_secondary = ev3.decide_secondary
required_draws = ev3.required_draws
difference_summary = ev3.difference_summary
pool_ape_v = ev3.pool_ape_v
pooled_ape = ev3.pooled_ape
counts_of = ev3.counts_of
metric_table = ev3.metric_table
quartile_table = ev3.quartile_table
presence_mask = ev3.presence_mask
wide = ev3.wide
draw_arrays = ev3.draw_arrays
date_companies = ev3.date_companies
scored_companies = ev3.scored_companies
independent_mack = ev3.independent_mack
check_members = ev3.check_members
check_tables_against_truth = ev3.check_tables_against_truth
Fit = ev3.Fit
RECOMPUTE_TOL = ev3.RECOMPUTE_TOL
TIE_OUT_TOL = ev3.TIE_OUT_TOL
fmt_pct = ev3.fmt_pct
fmt_extra = ev3.fmt_extra


def docstring_sha256() -> str:
    """The sha256 of this module's pre-registered docstring (recorded before the first fit)."""
    return hashlib.sha256(__doc__.encode("utf-8")).hexdigest()


# -- replication, with its own targets and its own direction on each cohort -----------------


def _within(distance: float, bound: float) -> bool:
    """``distance <= bound`` with the float noise of a subtraction rounded away."""
    return round(distance - bound, 9) <= 0


def replication_label(
    cohort: str, m: float, lo: float, hi: float, d10: float, m_net: float | None
) -> dict:
    """R1 to R3 and the label for a cohort, from percentage points (not fractions).

    ``m`` is the draw average's valuation-10 Pool_APE, ``lo`` and ``hi`` the range of the draws'
    own, ``d10`` the refit minus no-refit difference at valuation 10 and ``m_net`` the raw refit
    network's draw average (or None). The refit effect R2 has the companion's sign on each
    cohort: on B4 the refit must help by at least half a point (``d10 <= -0.5``), on C4 it must
    hurt by at least half a point (``d10 >= +0.5``).
    """
    target = COMPANION[cohort]["blend"]
    network = COMPANION[cohort]["network"]
    r1 = bool(
        _within(abs(m - target), R1_TOLERANCE)
        and _within(lo - target, 0.0)
        and _within(target - hi, 0.0)
    )
    r2 = bool(
        _within(d10 + R2_DIFFERENCE, 0.0) if cohort == "B" else _within(R2_DIFFERENCE - d10, 0.0)
    )
    r3 = None if m_net is None else bool(_within(abs(m_net - network), R3_TOLERANCE))
    far = not _within(abs(m - target), FAIL_DISTANCE)
    if r1 and r2:
        label = REPLICATION_WORDS[0]
    elif not r1 and not r2 and far:
        label = REPLICATION_WORDS[2]
    else:
        label = REPLICATION_WORDS[1]
    return {
        "cohort": cohort,
        "target": target,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "label": label,
        "m": m,
        "lo": lo,
        "hi": hi,
        "d10": d10,
    }


def reading_of_miss(rep: dict, p0_diff: float | None = None, spread: float | None = None) -> dict:
    """How the pre-registration says to read the replication on this cohort.

    ``rep`` is :func:`replication_label`'s result. On B4 the five readings for a miss of R1 are
    applied in the order written; ``stop`` is True only for the third (a draw average above 4.37
    or below 2.37 with R2 failing). ``p0_diff`` (the valuation-10 draw-average difference of the
    fourth study minus the third, in points) and ``spread`` (the width of the third study's ten
    draws, in points) feed the fifth reading when both are known. On C4 only the note on an
    unsurprising level applies.
    """
    m, target = rep["m"], rep["target"]
    lines: list[str] = []
    stop = False
    if rep["cohort"] == "C":
        if m > C_UNSURPRISING:
            lines.append(
                f"the draw average {m:.2f} is above {C_UNSURPRISING}: unsurprising, because the "
                f"target {target} is the lowest of the companion's three seeds (7.45 and 7.67 "
                "for the others); R2c is judged on the direction and size of the refit effect"
            )
        else:
            lines.append("R2c is judged on the direction and size of the refit effect")
        return {"lines": lines, "stop": False}
    if rep["r1"]:
        lines.append(
            "(1) R1 holds: removing the embedding has moved ibnr's level in the direction and by "
            "about the amount the companion's moved"
        )
    else:
        if READ_NEAR_LOW < m <= READ_NEAR_HIGH:
            lines.append(
                "(2) the draw average is between 3.87 and 4.37: partially replicates by the "
                "rule; ibnr's level did not follow the companion's drop, which is not read as a "
                "defect (candidate causes are listed in the results note and none is tested)"
            )
        elif (m > READ_NEAR_HIGH or m < READ_FAR_LOW) and not rep["r2"]:
            lines.append("(3) " + STOP_AND_ASK)
            stop = True
        elif m > READ_NEAR_HIGH or m < READ_FAR_LOW:
            lines.append(
                "(3) the draw average is more than 1.0 point from the target but R2 holds: the "
                "label is partially replicates and the stop does not apply"
            )
        else:
            lines.append(
                "R1 misses without falling in a band the pre-registration read in advance "
                f"(draw average {m:.2f}, target in the range of the draws: "
                f"{rep['lo'] <= target <= rep['hi']}); the label follows the rule"
            )
    lines.append("(4) P1 and P2 are decided by their own rules; a miss of R1 changes no verdict")
    if not rep["r1"] and p0_diff is not None and spread is not None:
        if abs(p0_diff) <= spread:
            lines.append(
                "(5) the P0 difference at valuation 10 is no larger than the third study's draw "
                "spread: removing the embedding did not detectably move ibnr's level (this says "
                "nothing about why the companion's moved)"
            )
        else:
            lines.append(
                "(5) the P0 difference at valuation 10 is larger than the third study's draw "
                "spread; read it beside the rerun control"
            )
    return {"lines": lines, "stop": stop}


def refit_overall_statement(c4, b4, provisional: dict | None = None) -> str:
    """P1 across cohorts from the two verdict words (``None`` for a cohort not run)."""
    if c4 is None:
        return REFIT_OVERALL["none"]
    if c4 == "hurts":
        text = REFIT_OVERALL["hurts"]
    elif c4 == "not distinguishable":
        text = REFIT_OVERALL["nd"]
    elif b4 == "helps":
        text = REFIT_OVERALL["both"]
    else:
        text = REFIT_OVERALL["fresh_only"]
    if b4 == "helps" and c4 != "helps":
        text += "; " + REFIT_OVERALL["not_shown"]
    text += f" (C4 {c4}; B4 {b4 if b4 is not None else 'not run'})"
    return _provisional(text, provisional)


def effective_overall_statement(c4, b4, provisional: dict | None = None) -> str:
    """P2 across cohorts from the two verdict words (``None`` for a cohort not run)."""
    if c4 is None:
        return EFFECTIVE_OVERALL["none"]
    if c4 == "not distinguishable":
        text = EFFECTIVE_OVERALL["nd"]
    elif c4 == "worse":
        text = EFFECTIVE_OVERALL["worse"]
    elif b4 == "worse":
        text = EFFECTIVE_OVERALL["mixed"]
    elif b4 == "effective":
        text = EFFECTIVE_OVERALL["confirmed"]
    else:
        text = EFFECTIVE_OVERALL["not_worse"]
    text += f" (C4 {c4}; B4 {b4 if b4 is not None else 'not run'})"
    return _provisional(text, provisional)


def _provisional(text: str, provisional: dict | None) -> str:
    flagged = sorted(k for k, v in (provisional or {}).items() if v)
    if flagged:
        return f"provisional ({', '.join(flagged)} not exactly draws 0 to 9): {text}"
    return text


# -- P0, the rerun control and the variant rows ---------------------------------------------


def p0_arm(
    new: np.ndarray, old: np.ndarray, actual, mask, counts, idx_ex, keep10, draw_ids
) -> dict:
    """P0 for one arm: the fourth study's draws minus the third study's, same draw, company, date.

    ``new`` and ``old`` are ``(draws, valuations, companies)`` arrays of the same draws. No
    threshold and no label. Returns the per-draw valuation-10 Pool_APE of each study and their
    difference, the median and range of the differences, the draw-averaged difference (valuation
    10 and dollar-pooled, with the P1 bootstrap interval), the valuation-10 difference without the
    dominant company, and how many companies at valuation 10 have a smaller absolute error in the
    fourth study.
    """
    per_new = np.array([pool_ape_v(x, actual, mask)[LAST_I] for x in new])
    per_old = np.array([pool_ape_v(x, actual, mask)[LAST_I] for x in old])
    diffs = per_new - per_old
    mean_new, mean_old = new.mean(axis=0), old.mean(axis=0)
    summary = difference_summary(mean_new, mean_old, actual, mask, counts)
    excluding = ev2.last_difference(
        mean_new[LAST_I][keep10], mean_old[LAST_I][keep10], actual[LAST_I][keep10], idx_ex
    )
    inside = mask[LAST_I]
    err_new = np.abs(mean_new[LAST_I] - actual[LAST_I])[inside]
    err_old = np.abs(mean_old[LAST_I] - actual[LAST_I])[inside]
    return {
        "draws": list(draw_ids),
        "per_draw_new": per_new,
        "per_draw_old": per_old,
        "per_draw_diff": diffs,
        "median_diff": float(np.median(diffs)),
        "range_diff": (float(diffs.min()), float(diffs.max())),
        "last": summary["last"],
        "pooled": summary["pooled"],
        "excluding": excluding,
        "smaller_in_fourth": int((err_new < err_old).sum()),
        "companies": int(inside.sum()),
    }


def selection_report(cohort, sets: dict) -> tuple[pd.DataFrame, list[str]]:
    """The third study's table, with this study's expected company counts on both cohorts."""
    table, problems = ev3.selection_report(cohort, sets)
    expected = EXPECTED_COMPANIES.get(cohort.name, {})
    return table.assign(companies_expected=table["valuation"].map(expected)), problems


def third_study_draws(fits3: list, fits4: list, companies, mask) -> tuple[dict, dict, list[int]]:
    """The two studies' arms over the draws both have: ``(fourth, third, draw ids)``."""
    common = sorted({f.draw for f in fits3} & {f.draw for f in fits4})
    mine = [f for f in fits4 if f.draw in common]
    theirs = [f for f in fits3 if f.draw in common]
    new = {arm: draw_arrays(mine, companies, ARM_COLUMN[arm], mask) for arm in P0_ARMS}
    old = {arm: draw_arrays(theirs, companies, ARM_COLUMN[arm], mask) for arm in P0_ARMS}
    return new, old, common


def variant_table(main_fit, variant_fits: dict, control_fits: list, companies, actual, mask):
    """Valuation-10 and four-date mean Pool_APE of each single-draw row, blend and raw network.

    ``main_fit`` is the main row's draw 0 (the reference). Descriptive only: no interval, no
    rule, no verdict. The companion's means over dates sit beside the four variant rows; the
    valuation-10 figures of his table are not in the pre-registration, so none is shown.
    """

    def levels(fit):
        out = {}
        for key, column in (("blend", "predicted"), ("network", "predicted_net")):
            arr = draw_arrays([fit], companies, column, mask)[0]
            per = pool_ape_v(arr, actual, mask)
            out[f"{key}_valuation_{LAST}"] = float(per[LAST_I])
            out[f"{key}_mean_dates"] = float(per.mean())
        return out

    rows = []
    reference = levels(main_fit)
    entries = [(MAIN_ROW, main_fit)]
    entries += [(row, fits[0]) for row, fits in variant_fits.items() if fits]
    entries += [(CONTROL_ROW, control_fits[0])] if control_fits else []
    for row, fit in entries:
        got = levels(fit)
        base = COMPANION_VARIANTS.get(row, {})
        rows.append(
            {
                "row": row,
                "setting": VARIANT_SETTING[row],
                "draw": fit.draw,
                **got,
                f"blend_valuation_{LAST}_minus_main_draw_0": (
                    got[f"blend_valuation_{LAST}"] - reference[f"blend_valuation_{LAST}"]
                ),
                "companion_blend_mean_dates": base.get("blend", np.nan),
                "companion_network_mean_dates": base.get("network", np.nan),
            }
        )
    return pd.DataFrame(rows)


def rerun_control(main_fit4, control_fit, fit3_draw0, companies, actual, mask) -> dict:
    """The rerun control's move beside P0's draw 0, on B4, in valuation-10 Pool_APE.

    ``control_fit`` is ``tlrn_cf_rerun`` (the third study's model fitted again, this study's
    image). Its valuation-10 Pool_APE minus the third study's B3 draw 0 measures what a rerun of
    an unchanged specification moves; the fourth study's draw 0 minus the same third-study fit is
    P0's draw 0. Both for the blend and the raw network. One fit: a scale, not a distribution.
    """
    out = {}
    for key, column in (("blend", "predicted"), ("network", "predicted_net")):

        def level(fit, _column=column):
            arr = draw_arrays([fit], companies, _column, mask)[0]
            return float(pool_ape_v(arr, actual, mask)[LAST_I])

        third, rerun, fourth = level(fit3_draw0), level(control_fit), level(main_fit4)
        out[key] = {
            "third_study_draw_0": third,
            "rerun": rerun,
            "fourth_study_draw_0": fourth,
            "rerun_minus_third": rerun - third,
            "p0_draw_0": fourth - third,
        }
    return out


# -- the benchmark tie-out against the third study's saved values ---------------------------


def third_study_tie_out(
    cohort_name: str, result: dict, folder: Path | None, *, smoke: bool = False
) -> dict | str:
    """The valuation-10 Pool_APE of ``mack`` and ``mcl`` against the third study's saved ones.

    Read from its ``<cohort>_metrics_by_valuation.csv``. A difference beyond rounding stops the
    evaluator: the benchmarks are the same programs on the same data as the third study's. So
    does a tie-out that cannot be made (no file, no row, or a company left out, which the third
    study's benchmarks never did), because the rule is that they must equal, not that they
    must equal when they can be read. A smoke run only notes these, since nothing in it is read.
    """

    def not_tied_out(why: str) -> str:
        if smoke:
            return why
        raise SystemExit(f"cohort {cohort_name}: the benchmarks must tie out and {why}")

    path = None if folder is None else Path(folder) / f"{cohort_name}_metrics_by_valuation.csv"
    if path is None or not path.exists():
        return not_tied_out(f"NOT TIED OUT: no {cohort_name}_metrics_by_valuation.csv in {folder}")
    if result["left_out"]:
        return not_tied_out("not compared: some companies are left out as a benchmark failed")
    table = pd.read_csv(path)
    out = {}
    for b in BENCHMARKS:
        row = table[(table["method"] == b) & (table["valuation"].astype(str) == str(LAST))]
        if row.empty:
            return not_tied_out(f"NOT TIED OUT: {path.name} has no {b} row at valuation {LAST}")
        theirs = float(row["pool_ape"].iloc[0])
        mine = float(result["benchmark_pool_ape"][b][LAST_I])
        out[b] = {"ours": mine, "third_study": theirs, "relative": abs(mine / theirs - 1.0)}
        if out[b]["relative"] > TIE_OUT_TOL:
            raise SystemExit(
                f"{b} at valuation {LAST} on cohort {cohort_name} does not reproduce the third "
                f"study's saved Pool_APE: {mine} against {theirs}"
            )
    return out


# -- the saved fits -------------------------------------------------------------------------


def load_fits(folder: Path, cohort: str, row: str, *, smoke: bool) -> list[Fit]:
    """Every finished fit of one cohort and row, in draw order, checked to be one study.

    A record counts only if it says it is the fourth study's (``study`` 4 and the name
    ``companion_revised``). All fits must share every setting except the draw and its seed, each
    seed must be the one its draw implies, every table must be there, the row's two defining
    settings must be what the row is (``maturity_embedding`` False except in the control row,
    ``paid_only`` True only in the paid-only row), and (outside smoke) every member must have run
    its epochs and its refit to the chosen one. Two ibnr versions of one draw are refused rather
    than one of them picked. A fit taken from the third study (``run['reused_from']``) is read
    like any other and named by the report.
    """
    spec = study.get_study("companion_revised")
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
        wants_table = row == CONTROL_ROW
        if settings["config"].get("maturity_embedding") is not wants_table:
            raise SystemExit(
                f"{path.name}: maturity_embedding is {settings['config'].get('maturity_embedding')}"
                f" but row {row} is defined with it {'on' if wants_table else 'off'}"
            )
        if settings.get("paid_only") is not (row == "tlrn_cr_paid"):
            raise SystemExit(
                f"{path.name}: paid_only is {settings.get('paid_only')} but row {row} is "
                f"defined with it {row == 'tlrn_cr_paid'}"
            )
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


def failed_fits(folder: Path, *, smoke: bool) -> list[str]:
    """The fits the runner reported as failed (a text file each), as one line apiece."""
    lines = []
    for path in sorted(Path(folder).glob("FAILED_*.txt")):
        if ("_smoke" in path.name) == smoke:
            lines.append(path.read_text(encoding="utf-8").strip())
    return lines


def check_fits_against_cohort(
    cohort, row: str, fits: list[Fit], sets: dict, *, strict: bool = False
) -> str:
    """The third study's check (cohort, data, study), with this study's name.

    The hash of each date's pairs recorded in a fit must equal the evaluator's own, so a fit
    trained on other companies at any date is refused by name. Returns the note about the data
    hash when only it differs.
    """
    note = ev2.check_fits_against_cohort(
        cohort, row, fits, strict=strict, study_name="companion_revised"
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


def data_hash_report(cohort, fits: list[Fit]) -> str:
    """The recorded data hashes and the evaluator's own, printed whether or not they match."""
    recorded = sorted({f.settings["data_hash"] for f in fits})
    verdict = "equal" if recorded == [cohort.data_hash] else "DIFFERENT"
    return f"recorded {', '.join(recorded)}; built here {cohort.data_hash} ({verdict})"


def environments(fits: list[Fit]) -> dict:
    """The distinct environments the fits report, one line each, with reused fits named."""
    seen: dict[str, int] = {}
    for fit in fits:
        run = fit.run.get("reused_from", {}).get("run") or fit.run
        reused = "reused from the third study: " if fit.run.get("reused_from") else ""
        key = reused + ", ".join(
            f"{k} {run.get(k)}"
            for k in ("ibnr", "jax", "python", "pandas", "numpy", "duckdb", "device_kind")
            if run.get(k) is not None
        )
        seen[key] = seen.get(key, 0) + 1
    return {k: f"{n} fits" for k, n in sorted(seen.items())}


# -- one cohort -----------------------------------------------------------------------------


def companion_beside_ours(result: dict, main_fits: list, cohort_name: str) -> pd.DataFrame:
    """The companion's Pool_APE figures next to ibnr's draws (valuation 10). DESCRIPTIVE.

    Not a decision rule and feeds none. The companion reports one fit at his primary seed; ibnr
    has ten draws, each a complete fit, and the random streams cannot be aligned, so the table
    shows where his number falls among ours.
    """
    mask, actual, draws = result["mask"], result["actual"], result["draws"]
    i = LAST_I
    theirs = COMPANION[cohort_name]

    def per_draw(arm):
        return (
            [100 * float(pool_ape_v(d, actual, mask)[i]) for d in draws[arm]]
            if arm in draws
            else None
        )

    rows = [
        ev3._beside("tlrn_cr_refit blended: Pool APE (%)", theirs["blend"], per_draw(REFIT)),
        ev3._beside("raw network alone: Pool APE (%)", theirs["network"], per_draw(RAW_NETWORK)),
        ev3._beside(
            "blended without refit: Pool APE (%)", theirs["blend_no_refit"], per_draw(NOREFIT)
        ),
        ev3._beside(
            "CL+MCL (his best classical): Pool APE (%)",
            theirs["classical_best"],
            [100 * float(result["benchmark_pool_ape"]["mcl"][i])],
            "ibnr's mcl, one fit",
        ),
    ]
    members = [f.members for f in main_fits]
    last = [m[m["valuation"].astype(int) == LAST] for m in members]
    rows.append(
        ev3._beside(
            f"median alpha at valuation {LAST} (per draw)",
            theirs["median_alpha"],
            [float(m["alpha"].median()) for m in last],
        )
    )
    for j, (v, n) in enumerate(zip(VALUATIONS, theirs["companies_per_date"], strict=True)):
        rows.append(ev3._beside(f"companies at valuation {v}", n, [float(mask[j].sum())]))
    return pd.DataFrame(rows)


def evaluate_cohort(
    cohort, fits_by_row: dict, bench: pd.DataFrame, sets: dict, fits3: list | None = None
) -> dict:
    """Everything the fourth study reports for one cohort, from fits and benchmarks.

    ``fits_by_row`` holds the study's rows, ``bench`` is the benchmark frame, ``sets`` the
    ``{v: pairs}`` company sets and ``fits3`` the third study's main-row fits for this cohort (or
    None). P1 and P2 are decided from the main row's draw averages, and only when it has fits.
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
        "label": f"{cohort.name}4",
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
        "reused": sorted({f.draw for f in factor_fits if f.run.get("reused_from")}),
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
        note = ""
        if provisional:
            note = "provisional: " + "; ".join(notes)
            label, reason = f"{label} ({note})", f"{note}; {reason}"
        result["p2"] = {**primary, "label": label, "reason": reason, "provisional": provisional}
        result["p2_missing_draws"] = missing
        result["p2_raw_network"] = p2(RAW_NETWORK)

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

        # P0, only if the third study's fits exist and were trained on the same companies
        result["p0"] = None
        p0_note = "not reported: the third study's fits for this cohort were not given"
        if fits3:
            same = fits3[0].settings.get("valuation_hashes") == main_fits[0].settings.get(
                "valuation_hashes"
            )
            if not same:
                p0_note = "not reported: the per-date pair hashes of the two studies differ"
            else:
                new, old, ids = third_study_draws(fits3, main_fits, companies, mask)
                if not ids:
                    p0_note = "not reported: the two studies have no draw in common"
                else:
                    result["p0"] = {
                        arm: p0_arm(new[arm], old[arm], actual, mask, counts, idx_ex, keep10, ids)
                        for arm in P0_ARMS
                    }
                    # all of the third study's draws, not only those the fourth study has yet
                    third_all = draw_arrays(fits3, companies, ARM_COLUMN[REFIT], mask)
                    third_per = np.array([pool_ape_v(x, actual, mask)[LAST_I] for x in third_all])
                    result["p0_third_spread"] = float(third_per.max() - third_per.min()) / POINT
                    result["p0_third_spread_draws"] = len(fits3)
                    p0_note = ""
        result["p0_note"] = p0_note

        # the replication of the companion's figures, with the reading rule for a miss
        m = float(pool_ape_v(series[REFIT], actual, mask)[LAST_I]) / POINT
        lo = float(per_refit[:, LAST_I].min()) / POINT
        hi = float(per_refit[:, LAST_I].max()) / POINT
        net = float(pool_ape_v(series[RAW_NETWORK], actual, mask)[LAST_I]) / POINT
        rep = replication_label(cohort.name, m, lo, hi, d["last"]["diff"] / POINT, net)
        p0_diff = None
        spread = None
        if result["p0"] is not None:
            p0_diff = result["p0"][REFIT]["last"]["diff"] / POINT
            spread = result["p0_third_spread"]
        rep["reading"] = reading_of_miss(rep, p0_diff, spread)
        result["replication"] = rep
        result["alpha"] = ev2.alpha_spread(main_fits)
        result["companion"] = companion_beside_ours(result, main_fits, cohort.name)

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

    # the single-draw rows and the control
    variant_fits = {row: fits_by_row.get(row) or [] for row in VARIANT_ROWS}
    control_fits = fits_by_row.get(CONTROL_ROW) or []
    draw0 = [f for f in main_fits if f.draw == 0]
    if draw0 and (any(variant_fits.values()) or control_fits):
        result["variants"] = variant_table(
            draw0[0], variant_fits, control_fits, companies, actual, mask
        )
    if draw0 and control_fits and fits3:
        third0 = [f for f in fits3 if f.draw == 0]
        if third0:
            result["rerun_control"] = rerun_control(
                draw0[0], control_fits[0], third0[0], companies, actual, mask
            )

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
    for arm, dr in draws.items():
        per = np.array([pool_ape_v(x, actual, mask) for x in dr])
        summaries[arm] = {
            **{str(v): first.draw_summary(per[:, i]) for i, v in enumerate(VALUATIONS)},
            "rolling": first.draw_summary(per.mean(axis=1)),
        }
    result["draw_summary"] = summaries
    return result


def company_table(result: dict) -> pd.DataFrame:
    """One row per (valuation, company in that date's set): the actual and every series."""
    return ev3.company_table(result)


# -- the report -----------------------------------------------------------------------------


def _p0_lines(result: dict) -> list[str]:
    lines = ["", "P0, embedding removed minus kept (descriptive; no threshold, no label):"]
    if result.get("p0") is None:
        lines.append(f"   {result.get('p0_note', 'not reported')}")
        return lines
    for arm, p in result["p0"].items():
        diffs = p["per_draw_diff"] / POINT
        lines += [
            f"--- {arm}: draws {p['draws']}",
            "   valuation-10 Pool_APE per draw, fourth study (%): "
            + " ".join(f"{100 * x:.2f}" for x in p["per_draw_new"]),
            "   valuation-10 Pool_APE per draw, third study (%):  "
            + " ".join(f"{100 * x:.2f}" for x in p["per_draw_old"]),
            "   fourth minus third per draw (points):             "
            + " ".join(f"{x:+.2f}" for x in diffs),
            f"   median {p['median_diff'] / POINT:+.3f}, range "
            f"[{p['range_diff'][0] / POINT:+.3f}, {p['range_diff'][1] / POINT:+.3f}] points",
            f"   draw averages, valuation 10: {p['last']['diff'] / POINT:+.3f} points "
            f"[{p['last']['low'] / POINT:+.3f}, {p['last']['high'] / POINT:+.3f}]; dollar-pooled "
            f"{p['pooled']['diff'] / POINT:+.3f} "
            f"[{p['pooled']['low'] / POINT:+.3f}, {p['pooled']['high'] / POINT:+.3f}]; without "
            f"the dominant company {p['excluding']['diff'] / POINT:+.3f} "
            f"[{p['excluding']['low'] / POINT:+.3f}, {p['excluding']['high'] / POINT:+.3f}]",
            f"   companies with a smaller absolute error in the fourth study: "
            f"{p['smaller_in_fourth']} of {p['companies']}",
        ]
    spread = result.get("p0_third_spread")
    if spread is not None:
        lines.append(
            f"   the third study's own draw-to-draw spread of the refit blend at valuation 10, "
            f"over its {result['p0_third_spread_draws']} draws: "
            f"{spread:.2f} points (a difference smaller than this is not interpretable)"
        )
    if result["cohort"] == "B":
        lines.append(
            f"   the companion's own move, for reading only: the blend went from "
            f"{COMPANION_EARLIER_BLEND_B} to {COMPANION['B']['blend']} percent, a fall of "
            f"{COMPANION_EARLIER_BLEND_B - COMPANION['B']['blend']:.2f} points"
        )
    return lines


def render(cohort, result: dict, extras: dict, banner: str) -> str:
    """The text report of one cohort."""
    label = result["label"]
    lines = [banner, ""] if banner else []
    lines.append(f"=== {cohort.description}")
    lines.append(
        f"status of any verdict on {label}: {result['status']}; valuation {LAST} is {cohort.as_of}"
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
    if result["reused"]:
        lines.append(
            f"the network-off row for draws {result['reused']} is the third study's "
            "tlrn_cf_factors fit (taken, not refitted; the identity check in the "
            "pre-registration allowed it)"
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
            f"P1 for cohort {label} ({result['status']}): {p1['label'].upper()}"
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
            f"P2 for cohort {label} ({result['status']}): {p2['label'].upper()}",
            f"   {p2['reason']}",
            f"   the same rule on the stratified intervals gives: {p2['sensitivity_word'].upper()}",
            f"   (secondary row) the raw refit network by the same rule: "
            f"{result['p2_raw_network']['word'].upper()}",
        ]
    for key, title in (("network", "NETWORK QUESTION"), ("blend", "BLEND QUESTION")):
        if key in result:
            lines += [f"{title} for cohort {label}: {result[key]['word'].upper()}"]
            lines.append(f"   {result[key]['reason']}")
    if "p1" in result:
        lines += _p0_lines(result)
    if "replication" in result:
        r = result["replication"]
        r2_rule = "D10 <= -0.5" if r["cohort"] == "B" else "D10 >= +0.5"
        lines += [
            "",
            f"REPLICATION of the companion's reported figures ({label}, valuation 10, points): "
            f"{r['label'].upper()}",
            f"   m {r['m']:.3f} range [{r['lo']:.3f}, {r['hi']:.3f}] against {r['target']}; "
            f"D10 {r['d10']:+.3f} (R2 needs {r2_rule}); R1 {r['r1']}, R2 {r['r2']}, R3 {r['r3']} "
            f"(the raw network against {COMPANION[r['cohort']]['network']})",
            "   how the pre-registration says to read it:",
            *[f"      {x}" for x in r["reading"]["lines"]],
        ]
        seeds = COMPANION[r["cohort"]]["seeds"]
        lines.append(
            f"   the companion's three seeds (descriptive, no label): mean {seeds['mean']}, range "
            f"{seeds['min']} to {seeds['max']}; ibnr's draw average {r['m']:.2f}, draws "
            f"{r['lo']:.2f} to {r['hi']:.2f}"
        )
    if "alpha" in result:
        lines += ["", "Spread of the blend weight alpha over members and draws:"]
        lines.append(result["alpha"].round(4).to_string(index=False))
        lines.append(
            f"   the companion's median alpha on {label}: "
            f"{COMPANION[result['cohort']]['median_alpha']}"
        )
    if "companion" in result:
        lines += [
            "",
            "THE COMPANION'S NUMBERS BESIDE OURS (descriptive, not a decision rule): "
            f"{label}, valuation {LAST}. His figures are one fit at his primary seed, the lowest "
            "of his three; ours are the per-draw values of ten complete fits. The random streams "
            "cannot be aligned, so this shows where his number falls in our spread and nothing "
            "more.",
            result["companion"].round(4).to_string(index=False),
        ]
    if "variants" in result:
        lines += [
            "",
            "SINGLE-DRAW ROWS (descriptive; no verdict, no interval, no statement that a "
            f"component helps or hurts; {label}, draw 0). Pool_APE as fractions; the companion's "
            "means over dates beside:",
            result["variants"].round(5).to_string(index=False),
        ]
    if "rerun_control" in result:
        lines += ["", "RERUN CONTROL (one fit, a scale and not a distribution), valuation 10:"]
        for key, c in result["rerun_control"].items():
            lines.append(
                f"   {key}: third study draw 0 {100 * c['third_study_draw_0']:.3f}%; rerun of the "
                f"third study's model {100 * c['rerun']:.3f}% (rerun minus third "
                f"{c['rerun_minus_third'] / POINT:+.3f} points); fourth study draw 0 "
                f"{100 * c['fourth_study_draw_0']:.3f}% (P0 draw 0: {c['p0_draw_0'] / POINT:+.3f} "
                "points)"
            )
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
    third_study_fits: Path | None = None,
    tie_out: Path | None = None,
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
    expected_companies = EXPECTED_COMPANIES[name]
    got_companies = {v: len(c) for v, c in date_companies(sets).items()}
    extras["companies per date against the pre-registration"] = (
        "equal"
        if got_companies == expected_companies
        else f"{got_companies} vs {expected_companies}"
    )
    truth = ev2.truth_table(cohort, VALUATIONS, sets)
    for row, fits in fits_by_row.items():
        if not fits:
            log(f"cohort {name} {row}: no finished fits in {fits_dir}")
            continue
        note = check_fits_against_cohort(cohort, row, fits, sets, strict=strict)
        if note:
            log(note)
        extras[f"data hash, {row}"] = data_hash_report(cohort, fits)
        extras[f"environments, {row}"] = environments(fits)
        check_tables_against_truth(cohort, row, fits, truth)
    fits3 = None
    if third_study_fits is not None:
        fits3 = ev3.load_fits(Path(third_study_fits), name, THIRD_MAIN_ROW, smoke=False) or None
        if fits3 is None:
            log(f"cohort {name}: no third-study {THIRD_MAIN_ROW} fits in {third_study_fits}")
        else:
            extras["environments, third study " + THIRD_MAIN_ROW] = environments(fits3)
    failed = failed_fits(fits_dir, smoke=smoke)
    if failed:
        extras["fits the runner reported as failed"] = "\n   " + "\n   ".join(failed)
    bench = ev3.cached_benchmark_cells(cohort, sets, cache_dir, log=log)
    result = evaluate_cohort(cohort, fits_by_row, bench, sets, fits3)
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
    extras["third study's valuation-10 benchmarks"] = third_study_tie_out(
        name, result, tie_out, smoke=smoke
    )
    extras["docstring sha256"] = docstring_sha256()
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
    if "companion" in result:
        result["companion"].to_csv(out / f"{name}_companion_beside_ours.csv", index=False)
    if "variants" in result:
        result["variants"].to_csv(out / f"{name}_single_draw_rows.csv", index=False)
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
    if result.get("p0"):
        rows = []
        for arm, p in result["p0"].items():
            for k, draw in enumerate(p["draws"]):
                rows.append(
                    {
                        "arm": arm,
                        "draw": draw,
                        "fourth_study": p["per_draw_new"][k],
                        "third_study": p["per_draw_old"][k],
                        "difference": p["per_draw_diff"][k],
                    }
                )
        pd.DataFrame(rows).to_csv(out / f"{name}_p0_per_draw.csv", index=False)


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
        record["p0_reported"] = result.get("p0") is not None
    if "network" in result:
        record["network"] = result["network"]["word"]
    if "replication" in result:
        record["replication"] = {
            k: v for k, v in result["replication"].items() if k != "reading"
        } | {
            "reading": result["replication"]["reading"]["lines"],
            "stop_and_ask": result["replication"]["reading"]["stop"],
        }
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fits", type=Path, required=True, help="the runner's --out folder")
    parser.add_argument(
        "--study3-fits",
        type=Path,
        default=None,
        help="the third study's folder, for P0 and the rerun control (not reported without it)",
    )
    parser.add_argument("--cohort", choices=("B", "C", "all"), default="all")
    parser.add_argument("--out", type=Path, default=None, help="write the report and tables here")
    parser.add_argument("--cache", type=Path, default=None, help="keep benchmark fits here")
    parser.add_argument("--cas-dir", type=Path, default=None, help="the four CAS files, cohort B")
    parser.add_argument(
        "--tie-out",
        type=Path,
        default=REPO / "analysis" / "results" / "tlrn_study3" / "evaluation",
        help="the third study's evaluation folder, for the valuation-10 benchmark tie-out",
    )
    parser.add_argument("--strict-data-hash", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="score a smoke folder")
    parser.add_argument(
        "--stop-answered",
        default=None,
        metavar="ANSWER",
        help="Ethan's answer to a stop-and-ask reading, in his words: the results are written "
        "after all and carry the answer; without it a stop writes nothing to --out",
    )
    args = parser.parse_args(argv)

    names = ["B", "C"] if args.cohort == "all" else [args.cohort]
    banner = (
        "SMOKE RUN. The fits were a few epochs and a few members. Nothing below may be quoted."
        if args.smoke
        else ""
    )
    if banner:
        print(banner)
    reports, verdicts, finished = [], {}, []
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
            third_study_fits=args.study3_fits,
            tie_out=args.tie_out,
        )
        if got is None:
            continue
        cohort, result = got
        report = render(cohort, result, result["extras"], banner)
        print(report)
        reports.append(report)
        verdicts[name] = verdict_record(result)
        print(f"(cohort {name} evaluated in {time.perf_counter() - t0:.0f}s)")
        finished.append((cohort, result))
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
    # the pre-registration's third reading: ask Ethan before writing the results. A smoke run
    # has no stops: its numbers are not results and its level says nothing
    must_ask = [
        name
        for name, value in verdicts.items()
        if not args.smoke and (value.get("replication") or {}).get("stop_and_ask")
    ]
    if args.stop_answered and not must_ask:
        raise SystemExit(
            "--stop-answered was given but no reading asked for an answer, so nothing would "
            "record it; run again without it"
        )
    held = bool(must_ask) and not args.stop_answered
    if args.out is not None and not held:
        for cohort, result in finished:
            save_tables(args.out, cohort, result)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.txt").write_text("\n\n".join(reports), encoding="utf-8")
        answered = {"stop_answered": args.stop_answered} if must_ask else {}
        (args.out / "verdicts.json").write_text(
            json.dumps(
                {"smoke": args.smoke, "verdicts": verdicts, "overall": overall, **answered},
                indent=2,
                default=str,
            ),
            "utf-8",
        )
    print("\nVERDICTS" + (" (SMOKE, not a result)" if args.smoke else ""))
    for key, value in verdicts.items():
        if value["p1"]:
            print(
                f"  {key}4: P1 {value['p1']['label']}; P2 {value['p2']['label']} "
                f"[{value['status']}]"
            )
            print(
                f"     network question: {value.get('network')}; blend question: {value['blend']}"
            )
            rep = value.get("replication")
            if rep and rep["stop_and_ask"]:
                print("     " + STOP_AND_ASK)
    if overall is not None:
        print(f"OVERALL P1: {overall['p1']}")
        print(f"OVERALL P2: {overall['p2']}")
    if held:
        print(
            f"\nNO RESULT WRITTEN: cohort {', '.join(must_ask)} needs Ethan's answer first. "
            'When he has given it, run this again with --stop-answered "<his answer>".'
        )
        return STOP_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
