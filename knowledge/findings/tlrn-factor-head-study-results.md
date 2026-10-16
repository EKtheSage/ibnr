---
type: Finding
title: Results of the second tlrn study
description: The companion's factor-head model, ten draws on three cohorts against the chain ladder and the multivariate chain ladder. Not distinguishable on A and B, worse on the fresh cohort C, so the overall statement is that it is worse on the fresh cohort.
tags: [tlrn, evaluation, colab, gpu, companion-study, results]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-16T09:00:00Z }
stale_after: 2027-01-14T00:00:00Z
sources:
  - id: colab
    resource: analysis/results/tlrn_study2/fits (the 60 settings and run records, company reserve tables, valuation forecasts, backtest cells and member tables, and the runner's manifest, from the Colab runs of 2026-10-15 to 16, ibnr 0.7.4, jax 0.11.1, cas-schedule-p publish 20260613_041006)
    title: The 60 finished fits
  - id: eval
    resource: analysis/results/tlrn_study2/evaluation (verdicts.json, report.txt, evaluator_log.txt and the tables written by scripts/tlrn_study2_eval.py)
    title: Evaluator output
  - id: notebook
    resource: analysis/05_tlrn_factor_head_study.ipynb
    title: Notebook 05, executed on the saved fits
  - id: decision
    resource: /decisions/tlrn-factor-head-study.md
    title: The questions and verdict rules, fixed before any fit
  - id: pr179
    resource: https://github.com/EKtheSage/ibnr/pull/179
    title: PR 179 (open when this was written), the study scripts, the notebook and these results
  - id: pr177
    resource: https://github.com/EKtheSage/ibnr/pull/177
    title: PR 177, the evaluator accepts Colab fits whose data hash text differs
  - id: companion
    resource: the companion study's published rolling-origin table (R_T1_rolling_origin.csv) and member table (T_T1_members.csv), read from outside this repository
    title: The companion's published tables, cohort B only
---

# Result

The pre-registered verdict for the companion's final model (the main row `tlrn_13_fh`) is
**not distinguishable on cohort A, not distinguishable on cohort B and worse on cohort C**.
The overall statement the rules fix for a mixed result is: **worse on the fresh cohort (C
worse; A not distinguishable, B not distinguishable)**.[^decision][^eval]

In plain words:

* On the companion's own cohort (B, and A which contains it) the model's reserve error at the
  last date is a little below the chain ladder and the multivariate chain ladder, by about
  0.4 to 0.7 points of Pool_APE, but the intervals around that gain are several times wider
  than the gain. The study cannot tell it from no gain. This was the expected outcome, not a
  surprise: the decision note says a gain of half a point cannot be detected on one cohort.
* On the fresh cohort C (accident years 1988 to 1997, never tuned on as far as we know) the
  model is clearly behind. At the last date its Pool_APE is 15.9% against 11.9% for the chain
  ladder and 5.0% for the multivariate chain ladder, and every one of its ten draws is above
  both.
* The model is also clearly behind both benchmarks at the earliest date (valuation 6) on
  cohorts A and B, which is the same weakness the companion's own rolling table shows.
* This says nothing about why. The explanations are listed under Limits as untested.

Notebook 05 reads the saved fits and reprints the same tables and verdicts from the
repository's copy of them (all but the comparison with the companion's tables, which need
those tables).[^notebook]

# Cohort tables

Pool_APE is the sum of absolute errors of the company reserves over the sum of the actual
reserves, so it is a dollar-weighted error and company 1767 dominates it. "Median draw" is
the median over the ten draws of each draw's own Pool_APE; "draws" is their minimum to
maximum. Each draw is one fit of 20 members averaged. `mack` and `mcl` are the chain ladder
and the multivariate chain ladder, both fitted by ibnr on the cells known at each date.

## Valuation 10 (the last known diagonal) and the rolling mean over valuations 6, 7, 8 and 10

| Cohort | Statistic | Model, median draw | Model, draws | mack | mcl |
|---|---|---|---|---|---|
| A (82 companies) | valuation 10 | 5.127% | 4.929% to 5.423% | 5.773% | 5.661% |
| A | rolling | 5.532% | 5.259% to 5.704% | 4.898% | 4.947% |
| B (84 companies) | valuation 10 | 5.013% | 4.743% to 5.140% | 5.481% | 5.410% |
| B | rolling | 5.363% | 5.246% to 5.548% | 4.854% | 4.884% |
| C (83 companies) | valuation 10 | 15.886% | 12.951% to 17.857% | 11.947% | 4.969% |
| C | rolling | 16.120% | 15.329% to 16.847% | 14.836% | 13.167% |

## The intervals behind each verdict

These are 95% company-bootstrap intervals for the Pool_APE difference, model minus
benchmark, in points (negative favours the model). The verdict rule needs the
valuation-10 interval and the pooled-over-dates interval wholly below zero against both
benchmarks, the median draw below both, and the valuation-10 difference without company 1767
not behind either benchmark.

| Cohort | Against | Valuation 10 | Pooled over valuations | Valuation 10 without company 1767 |
|---|---|---|---|---|
| A | mack | -2.17 to +2.58 | -0.93 to +2.07 | +0.09, from -3.21 to +3.39 |
| A | mcl | -5.14 to +2.14 | -1.67 to +1.34 | -0.26, from -7.24 to +2.72 |
| B | mack | -1.87 to +3.51 | -1.07 to +2.13 | +0.61, from -2.70 to +4.41 |
| B | mcl | -4.86 to +2.86 | -1.69 to +1.46 | +0.11, from -7.02 to +3.55 |
| C | mack | -0.002 to +10.87 | +0.05 to +5.02 | +4.47, from -0.99 to +12.26 |
| C | mcl | +6.37 to +12.32 | +1.84 to +4.81 | +8.14, from +5.74 to +11.52 |

On A and B the first two conditions fail because the intervals reach above zero, and the
fourth is not met either, so the verdict is not distinguishable. On C both pooled
intervals and the valuation-10 interval against `mcl` are wholly above zero (the model is
behind) and the valuation-10 interval against `mack` just touches zero, so the rule gives
worse. The stratified bootstrap, which keeps company 1767 in every resample and is a
sensitivity analysis and not the verdict, gives not distinguishable on A and B and worse on
C.

## The earlier dates

Median-draw Pool_APE of the model against `mack` and `mcl`, from the evaluator's draw tables:

| Cohort | Valuation 6 | Valuation 7 | Valuation 8 |
|---|---|---|---|
| A | 8.915% vs 6.031% and 6.248% | 4.435% vs 4.363% and 4.383% | 3.773% vs 3.426% and 3.495% |
| B | 9.230% vs 6.012% and 6.106% | 3.947% vs 4.439% and 4.410% | 3.346% vs 3.483% and 3.611% |
| C | 16.092% vs 13.772% and 13.836% | 19.450% vs 16.702% and 16.818% | 13.230% vs 16.923% and 17.044% |

The model is behind at valuation 6 everywhere. On A and B the interval against `mack` at
valuation 6 (A) is wholly above zero; the rest of the early-date intervals on A and B include
zero. On C the model is behind at valuation 7 (the interval is wholly above zero) and ahead
at valuation 8 (wholly below zero), and the bootstrap intervals do not separate it from
`mack` at valuation 6.

# Draw to draw spread

Ten fits of the same row differ only in their random streams. On A and B the spread at
valuation 10 is small (standard deviation 0.17 and 0.13 points). On C it is much larger:
1.50 points at valuation 10, with draws from 12.95% to 17.86%, so the fresh cohort is noisier
as well as worse. The factors-only comparison row (below) is steady on every cohort (a
standard deviation of 0.04 to 0.11 points at valuation 10).

# Does the network add anything? Does the blend?

Two questions are reported beside the verdict and never change it.

* **Network question**: the main row minus the row with the network switched off
  (`tlrn_13_fh_factors`: the head's pooled factors blended with the multivariate chain ladder
  by the fitted weight). The answer is **not distinguishable on A, B and C**. On A and B the
  main row's point estimate is a little better (valuation 10: 5.1% against 6.8% on A, 5.0%
  against 6.7% on B) but the intervals are wide (about -8 to +3 points). On C the
  point estimate goes the other way: the network-off row is at 7.8% and the main row at
  15.9%, again with an interval of about -10 to +13 points. The stratified bootstrap
  reads that C difference as the main row being behind (pooled +2.9 to +6.5 points), but
  that analysis keeps the dominant company fixed and is not the pre-registered answer.
  So the network adds nothing that this study can detect beyond the pooled factors with the
  blend.
* **Blend question**: the main row minus the raw network before the blend. The answer is
  **the blend adds value on C** (valuation 10 interval -1.57 to -0.83 points, rolling -5.73
  to -0.72, without company 1767 -1.77 to -0.77) and **not distinguishable on A and B**,
  where the rolling interval and the interval without company 1767 are below zero but the
  valuation-10 interval reaches just above zero (A -4.19 to +0.10, B -4.73 to +0.22).
  Pulling the network back toward the multivariate chain ladder is what helps, on C clearly.

The blend weight tells the same story from the inside. On C its median rises from 0.35 at
valuation 6 to 0.94 at valuation 10, so at the last date the blend leans almost wholly on
the network there, against about 0.63 and 0.61 on A and B.

# Replication of the companion

On cohort B the evaluator ties out to the companion's published rolling-origin table.[^companion]

* **The benchmarks match exactly.** ibnr's `mack` and `mcl` reproduce the companion's
  Pool_APE at valuations 6, 7, 8 and 10 (6.012, 4.439, 3.483 and 5.481 percent for `mack`;
  6.106, 4.410, 3.611 and 5.410 for `mcl`) to rounding, on the same 84 companies and
  2,968, 3,180, 2,756 and 9,540 scored cells.
* **The final value is reproduced within the draws.** The companion's published transformer
  value at valuation 10 is 4.969%, ours are 4.743% to 5.140% over ten draws (median draw
  5.013%, draw average 4.962%).
* **The early-date weakness is reproduced.** The model is well behind the benchmarks at
  valuation 6 in both (ours 9.2% against 6.0% for `mack`; the companion's published figures
  are 8.2% against 6.1% for `mcl`).
* **The member fidelity table** puts the blend weight and best epoch beside the
  companion's for each date. The medians at valuation 10 are close (blend weight 0.608
  against 0.627, best epoch 1,775 against 1,880); at valuations 6 and 7 our weights and
  epochs are higher (0.609 and 0.814 against 0.527 and 0.702; 130 and 568 against 108
  and 370). This is a report and not a pass or fail test.

The same benchmark tie-out holds on cohort A against notebook 04 (relative difference below
1e-12) and against the first study's saved benchmarks on all three cohorts.

# The dominant company

Company 1767 holds 66.3%, 67.3% and 66.7% of the actual valuation-10 reserve on A, B and C.
The verdict's fourth condition and the `without` columns above remove it. On A and B
removing it moves the model-minus-benchmark gap by about a point or less, and the gap stays
inside a wide interval. On C the gap against `mcl` is +11.07 points with the company and
+8.14 without, with intervals wholly above zero both times (+6.37 to +12.32 and +5.74 to
+11.52). Against `mack` it is +4.10 with the company and +4.47 without, and both intervals
reach zero from below (-0.002 and -0.99), so against `mack` the valuation-10 result is on the
edge either way. The C result is therefore not only company 1767.

By size of reserve at valuation 10, the model is nearer the truth than both benchmarks in
the three smaller size groups on A and B and level in the largest (within 0.2 points), and farther from the truth than both in every size group on C.
On C the model also over-reserves in total (bias +16.0% against +11.3% for `mack` and
+3.8% for `mcl`) and beats `mcl` on only 29% of companies at valuation 10 (A 51%, B 54%).

# The first study's accident-year rows

The first study's models are compared with the main row on the last date, on A and B. The
main row is ahead by 0.6 to 1.1 points of Pool_APE (A: 0.63 against `tlrn_8_ay` and 1.10
against `tlrn_13_ay`; B: 0.82 and 0.66), and every interval includes zero (for example A
against `tlrn_8_ay`: -3.67 to +4.64).[^eval]

# Compute

The 60 fits (3 cohorts, 2 rows, 10 draws) ran on one NVIDIA RTX PRO 6000 Blackwell Server
Edition GPU with ibnr 0.7.4 and jax 0.11.1. Fit time was 1.65 hours in total: 177 to 209
seconds for a main-row fit (30 fits, mean 190 seconds) and 8 to 10 seconds for a
network-off fit (30 fits, mean 9 seconds). The fits' finishing times in the run records span
about 6.8 hours, from 22:41 on 2026-10-15 to 05:27 on 2026-10-16; the records do not say what
filled the time between fits.[^colab] The model files (.pkl, about 425 MB) are in the owner's
Google Drive and not in the repository. The evaluator accepts these Colab fits although the
recorded cell hash text differs from a laptop build, because it checks the anchors,
premiums, lines and companies instead (the change in PR 177).[^pr177]

# Limits

* **C is independent in accident years only.** Its companies and calendar years overlap A
  and B (67 of its 83 companies are in A and 60 in B), so it is not a new economic
  environment or a new set of companies. Whether the companion tuned on C is assumed not
  known.
* **Every cohort needs complete squares.** A company enters only if its whole 10-by-10
  square, outcome cells included, exists. That survivorship applies to the benchmarks as
  well as to the model.
* **Low precision.** With 83 companies and one company holding two thirds of the reserve,
  the 95% intervals are wide: the valuation-10 interval is 5 to 11 points across, and the
  pooled one 3 to 5 points. They could only show a gain of several points.
* **The multivariate chain ladder on C is unusually accurate at valuation 10 (4.969%)** while
  the chain ladder is at 11.947% on the same cells. Neither is tied out to an external table
  on C, unlike B, where both tie out to the companion's published numbers. Much of the C
  verdict at valuation 10 rests on how close `mcl` is to the answer there.
* **One valuation date per cohort for the verdict.** The pooled-over-dates statistic uses
  four dates, but the three earlier ones are scored with later data in view of the company
  selection, as in the companion's design.
* **The reasons for the weaker result on C are not tested.** Candidate explanations only:
  tuning of the design on A and B, the larger share of the blend on the network at the last
  date, the different accident years and mix of lines in cohort C, and chance given the noise of one date. None was measured.

# What would settle it

More fresh cohorts and dates (a rolling set of cutoffs, other accident-year windows), a
size-balanced metric that does not let one company decide, a check of the cohort C
benchmarks against an independent implementation, and running the companion's own notebook
on cohort C so that the comparison does not rest on ibnr's re-implementation of it.

[^decision]: The questions and verdict rules, fixed before any fit
[^eval]: Evaluator output
[^notebook]: Notebook 05, executed on the saved fits
[^colab]: The 60 finished fits
[^pr177]: PR 177, the evaluator accepts Colab fits whose data hash text differs
[^companion]: The companion's published tables, cohort B only
