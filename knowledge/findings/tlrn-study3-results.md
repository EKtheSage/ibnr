---
type: Finding
title: Results of the third tlrn study
description: The companion's 21 October model (premium head, accident-year attention, a refit of every network), ten draws on two cohorts. On the companion's own cohort B3 it replicates the published figures and is not distinguishable from the chain ladder and the multivariate chain ladder; on the fresh cohort C3 the refit hurts.
tags: [tlrn, evaluation, colab, gpu, companion-study, results, pre-registered]
status: draft
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-27T04:00:00Z }
stale_after: 2027-04-27T00:00:00Z
sources:
  - id: eval
    resource: analysis/results/tlrn_study3/evaluation (verdicts.json, report.txt, evaluator_log.txt, benchmarks3_*.csv and one table per cohort and question, written by scripts/tlrn_study3_eval.py from the 40 fits)
    title: Evaluator output
  - id: decision
    resource: /decisions/tlrn-study3-preregistration.md
    title: The questions and verdict rules, fixed before any fit
  - id: evaluator
    resource: scripts/tlrn_study3_eval.py
    title: The evaluator, with the rules in its docstring
  - id: runner
    resource: scripts/tlrn_study_colab.py
    title: The runner, cohort selection (select_pairs) and data hash (cells_hash)
  - id: notebook
    resource: analysis/05_tlrn_study3_on_colab.ipynb
    title: The Colab notebook that made the fits
  - id: fits
    resource: the 40 fits of 2026-10-26 (settings and run records, company reserves, valuation forecasts, member tables and fitted results), kept in the owner's Google Drive folder ibnr_colab_cf/study3 and not in this repository; ibnr 0.7.6, jax 0.11.1, Python 3.13.16, cas-schedule-p publish 20260613_041006, one NVIDIA RTX PRO 6000 Blackwell Server Edition
    title: The 40 finished fits
---

# Result

The pre-registered verdicts for the companion's 21 October model (main row `tlrn_cf_refit`)
are: **P1, the refit hurts on the fresh cohort C3 (and is not distinguishable on B3); P2, not
distinguishable from the chain ladder and the multivariate chain ladder on either cohort.** The
overall statements the rules fix are "the refit hurts on the fresh cohort (C3 hurts; B3 not
distinguishable)" and "not distinguishable (C3 not distinguishable; B3 not
distinguishable)".[^eval]

In plain words:

* **B3, the companion's own cohort, replicates.** At the last known date (2007-12-31) the
  blended refit forecast has a Pool_APE of 4.02% (the ten draws range from 3.73% to 4.26%)
  against the companion's published 3.93%, and the raw network 3.19% (draws 2.93% to 3.69%)
  against his 2.96%. The evaluator's replication rule is met on all three of its conditions.
  The model is a little ahead of the chain ladder (5.48%) and the multivariate chain ladder
  (5.41%), but the interval around that gain reaches above zero, so the study cannot call it a
  gain. The refit lowers the error in every one of the ten draws (4.02% against 4.70% without
  it), yet the pooled interval over the four dates (-1.60 to +0.18 points) includes zero, so
  that is not distinguishable either.
* **C3, accident years 1988 to 1997, does not.** The refit raises the last-date error from 4.15%
  to 5.22% (+1.07 points, interval +0.08 to +1.95) and helps in only 2 of 10 draws. Against the
  benchmarks the model beats the chain ladder (11.95%) but not the multivariate chain ladder
  (4.97%), so the rule gives not distinguishable.
* **The network does something on B3 and nothing detectable on C3.** Against the same model with
  the network switched off (5.64%), the refit blend is better on B3 by 1.6 points (interval
  -5.45 to -1.10, "adds value"); on C3 the interval is -10.10 to +1.33.
* This says nothing about why. B3 is the cohort the model was designed on, so its result is a
  check that a fixed specification behaves as the companion reported. C3 shares most of its
  companies and one dominant company with B3.

# Numbers

Pool_APE is the sum of the absolute errors of the company reserves over the sum of the actual
reserves, so it is a dollar-weighted error and company 1767 dominates it (67.3% of the actual
date-10 reserve on B3, 66.7% on C3). Each model figure is the Pool_APE of the average of the ten
draws' forecasts. "Rolling" is the plain mean over the four dates (6, 7, 8 and 10). All figures
in percent, from `evaluation/{B,C}_metrics_by_valuation.csv`.[^eval]

| Arm | B3 date 10 | B3 rolling | C3 date 10 | C3 rolling |
|---|---|---|---|---|
| `tlrn_cf_refit` (blend with refit; the model) | 4.02 | 3.86 | 5.22 | 10.58 |
| `tlrn_cf_norefit` (same members, blend before the refit) | 4.70 | 4.05 | 4.15 | 9.48 |
| raw network with refit | 3.19 | 3.26 | 5.35 | 7.58 |
| raw network without refit | 4.92 | 8.02 | 4.04 | 5.26 |
| `tlrn_cf_factors` (network switched off) | 5.64 | 5.01 | 5.76 | 10.62 |
| `mack` (chain ladder) | 5.48 | 4.90 | 11.95 | 14.84 |
| `mcl` (multivariate chain ladder) | 5.41 | 4.93 | 4.97 | 13.17 |

The model against the benchmarks at each of the four dates:

| Cohort, arm | date 6 | date 7 | date 8 | date 10 |
|---|---|---|---|---|
| B3 `tlrn_cf_refit` | 4.63 | 3.72 | 3.07 | 4.02 |
| B3 `mack` | 6.10 | 4.50 | 3.50 | 5.48 |
| B3 `mcl` | 6.22 | 4.46 | 3.62 | 5.41 |
| C3 `tlrn_cf_refit` | 12.79 | 13.16 | 11.15 | 5.22 |
| C3 `mack` | 13.78 | 16.70 | 16.92 | 11.95 |
| C3 `mcl` | 13.84 | 16.82 | 17.04 | 4.97 |

Draw spread (one draw's own Pool_APE at date 10, minimum to maximum): B3 refit 3.73 to 4.26,
no refit 4.51 to 5.02, raw network 2.93 to 3.69, network off 5.53 to 5.74; C3 refit 4.25 to
6.20, no refit 3.72 to 5.53, raw network 4.09 to 6.85, network off 4.89 to 6.30. On B3, 54 of
the 84 companies are over-estimated at date 10 by the refit blend.

# The verdicts, with their intervals

Differences are in points of Pool_APE, row minus comparison (negative favours the row), with a
95% company-bootstrap interval, from `evaluation/verdicts.json`.[^eval]

| Question | B3 | C3 |
|---|---|---|
| P1, refit minus no refit, date 10 | -0.68, [-1.83, -0.33] | +1.07, [+0.08, +1.95] |
| P1, pooled over the four dates | -0.17, [-1.60, +0.18] | +1.11, [-0.98, +1.47] |
| P1, draws favouring the refit (8 needed) | 10 of 10 | 2 of 10 |
| P1, date 10 without company 1767 | -1.19 | +1.18, [-0.22, +2.17] |
| P1 verdict | not distinguishable (fails the pooled condition) | **hurts** |
| P2, refit minus `mack`, date 10 | -1.46, [-5.55, +1.97] | -6.73, [-8.07, -0.12] |
| P2, refit minus `mcl`, date 10 | -1.39, [-5.72, -0.65] | +0.25, [-3.12, +2.78] |
| P2, pooled, against `mack` and `mcl` | both wholly below zero | both wholly below zero |
| P2 verdict | not distinguishable (the date-10 interval against `mack` includes zero) | not distinguishable (three of the four conditions fail: the date-10 interval against `mcl`, the median draw against `mcl`, and the result without company 1767) |
| network question | **adds value** | not distinguishable |
| blend question (blend against raw network) | not distinguishable | not distinguishable |

P1 on B3 fails only on the condition that the pooled interval lie wholly below zero (it reaches
+0.18); the other three conditions hold. The same rule on the stratified bootstrap, which keeps
company 1767 in every resample, gives the same verdict on each cohort.

# What the data can and cannot support

* **Supports.** The ibnr implementation of the 21 October model, with the companion's network
  size (18,537 parameters), reproduces his published B3 figures within 0.1 (blend) and 0.3 (raw
  network) points; a fixed specification does what its author reported on the cohort it was
  built on. On B3 the refit reduces error in every draw at the last date.
* **Does not support.** A claim that the model beats the chain ladder or the multivariate chain
  ladder on B3. The date-10 interval against `mack` (-5.55 to +1.97) reaches above zero. Against
  `mcl` the date-10 interval is wholly below zero (-5.72 to -0.65) but the rule needs both
  benchmarks. The study's precision at date 10 is low: the intervals are 7.5 points wide against
  `mack` and 5.1 against `mcl`, so a gain has to be about 3.8 and 2.5 points to be detected
  there. Pooled over the four dates the intervals are narrower (about 2 points wide, a
  detectable gain of about 1.1) and lie wholly below zero against both benchmarks on B3 and on
  C3, but the pre-registered rule does not accept the pooled statistic without the date-10 one,
  and the dates are not independent.
* **Does not support** that the refit generally helps or hurts. The two cohorts disagree, the C3
  hurt rests on a date-10 interval whose lower end is +0.08 and a pooled interval that includes
  zero, and C3 is not an independent sample (below).
* **Does not support** any reason for the C3 result. No explanation was tested.

# Caveats

* **One company decides the dollar figures.** Company 1767 holds about two thirds of the actual
  date-10 reserve on both cohorts, so every dollar-based figure is largely a statement about it.
  The results without it are reported beside each verdict and are in the table above for P1.
* **B3 is a fixed specification on the companion's own window.** It was built with this cohort
  in view, so its verdict is "confirmatory of a fixed specification, not out of sample".
* **C3 is out of sample in accident years only.** At date 10 it shares 155 of its 232
  (company, line) pairs and 60 of its 83 companies with B3 (recomputed with `select_pairs`),
  and the second study's results on the same cohort existed before this model's rules were
  written. See `docs/tlrn-study3.md` for both cohorts in detail.
* **Ten draws sample training randomness thinly**, and GPU fits are not bit for bit repeatable
  (a rerun gives different draws). Date-10 Pool_APE has a draw standard deviation of 0.18
  points (refit) to 0.62 (C3 refit). The bootstrap resamples companies, so it says nothing
  about other years. No correction for the number of questions.
* **The refit's epoch is chosen with the same cells in view that the refit then learns from**, so a gain from
  the refit is an upper bound on what an independent choice would give.
* **The date-average figures of the companion are not matched as closely.** His date-average
  Pool_APE with the refit is 4.36% and without it 4.02% (the refit worse by 0.34 points); here
  they are 3.87% (draws 3.55 to 4.52) and 4.07% (3.90 to 4.30), the refit better by 0.20.
  His 4.36% sits above nine of our ten draws. The date-10 figures, which the replication rule
  uses, agree. We do not know whether this is chance, a difference in the training pool at the
  earlier dates, or something else; it was not investigated.
* **The raw network without the refit under-reserves at the early dates.** On B3 its pooled
  percentage error is -7.1%, -7.3% and -7.9% at dates 6, 7 and 8 (and +0.3% at date 10),
  against -0.9%, +0.5% and -0.4% for the raw network with the refit. The blended no-refit
  arm, which P1 uses, is nearer (-2.1%, -1.6%, -1.7%). So the rolling Pool_APE of 8.02% for
  the raw network without the refit reflects a bias at three dates, not noise.
* **The log-ratio statistic is not defined for every company at date 10**: on B3, company 17884
  has an actual of -2,689 (negative) and company 28886 has negative forecasts from `mcl` and
  the refit blend; on C3, company 38997 has an actual of exactly zero. Their counts are 82 to
  83 of 84 (B3) and 82 of 83 (C3), depending on the arm. Amounts are USD thousands.
* **The parameter count 18,537 is not stored in the saved files.** The notebook asserts it
  before any fit (section 6, `EXPECTED_PARAMETERS = 18537`), and the repository's tests build
  the network and count it: `tests/test_tlrn_refit.py` (`test_the_companion_final_variant_has_the_companions_18537_parameters`),
  `tests/test_tlrn_companion_network.py` and `tests/test_tlrn_study3_pins.py`. A reader of the
  saved fits cannot confirm it from them.
* **Several benchmark and near-tie counts can move by a company on a rerun.** Rerunning the
  evaluator on the same fits reproduced every Pool_APE and verdict to rounding, and moved one
  count ("share of companies better than `mcl`", `mack`, B3, date 6) from 0.311 to 0.300.

# Data hash: unexplained

Every fit record holds a data hash (`bce499efbb6bde4e` for B3, `e52d50e002d50e16` for C3).
The evaluator builds a different one from the same publish (`2d9ecc777b088fcd` and
`92f5d7a74f37a0c2`), prints "data hash differs" and, as the second study's evaluator was
changed to do (pull request 177), scores the fits anyway because the anchors, premiums, lines
and companies of every fit equal the cohort's within tolerance. `--strict-data-hash` makes it
an error. What was checked now:

* The hash of the selected pairs (`cohort_hash`, `de51bae56cca563c` and `4a613cc4cb5f0f94`)
  and the hash of each date's pairs agree between the fits and the evaluator, and the selection
  counts equal the pre-registration, so the same companies and lines were selected.
* The evaluator's hash is not specific to the evaluator: rebuilding the cohort on this
  checkout's locked environment (Python 3.12.13, pandas 2.3.3, duckdb 1.5.3, numpy 2.4.6) gives
  `2d9ecc777b088fcd` and `92f5d7a74f37a0c2` exactly. Only the Colab runs differ.
* The recipe is `sha256` of `DataFrame.to_csv` text of the cohort's cells (`cells_hash` in the
  runner), and the mart's columns are `DOUBLE`, `INTEGER` and `VARCHAR`, so the values read are
  the same doubles; the text can still differ with the pandas, numpy or duckdb version that
  built and wrote the frame.
* Not known: the Colab image's pandas, numpy and duckdb versions, because the run record keeps
  only ibnr, jax, Python and the platform. So the cause is **unexplained**; it is
  consistent with a library difference in how the frame is typed or written, and it is not
  evidence of a difference in the data (nor proof that there is none). Settling it takes one
  cell in a Colab runtime that prints those versions and `cells_hash` of the cohort, and, for
  the next study, adding the versions to the run record.

# Compute

Forty fits on 2026-10-26, first finished 20:58 UTC and last 23:51 UTC (2.9 hours), 2.94 GPU
hours of fit time. A main fit (20 members at each of four dates, 3,000 epochs) took 501 to 535
seconds on B3 (median 521) and 461 to 480 on C3 (median 478); a network-off fit 31 to 33
seconds. The pre-registered planning figure was 400 to 500 seconds and 3 to 6 GPU hours. The
evaluator took about 45 seconds (B3) and 52 seconds (C3) to fit the two benchmarks.[^notebook]

# What would settle it

Fresh cohorts (other accident-year windows and companies, not just other years of the same
companies), a size-balanced metric so one company does not decide, more draws, and recording the
library versions with the data hash. How to repeat B3 is in `docs/tlrn-study3-reproduce.md`.

[^eval]: Evaluator output
[^decision]: The questions and verdict rules, fixed before any fit
[^notebook]: The Colab notebook that made the fits
