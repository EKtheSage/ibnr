---
type: Finding
title: Results of the fourth tlrn study
description: The companion's revised model (the per-count embedding removed, 18,153 parameters), ten draws on two cohorts. Every pre-registered verdict on P1 and P2 is not distinguishable on both cohorts; the network adds value on B4 only; the replication of the companion's figures is partial on both cohorts.
tags: [tlrn, evaluation, colab, gpu, companion-study, results, pre-registered]
status: draft
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-11-25T12:00:00Z }
stale_after: 2027-05-25T00:00:00Z
sources:
  - id: eval
    resource: analysis/results/tlrn_study4/evaluation (verdicts.json, report.txt, evaluator_log.txt, benchmarks3_*.csv and one table per cohort and question, written by scripts/tlrn_study4_eval.py from the 44 fits)
    title: Evaluator output
  - id: decision
    resource: /decisions/tlrn-study4-preregistration.md
    title: The questions and verdict rules, fixed before any fit
  - id: evaluator
    resource: scripts/tlrn_study4_eval.py
    title: The evaluator, with the rules in its docstring
  - id: runner
    resource: scripts/tlrn_study_colab.py
    title: The runner, cohort selection (select_pairs) and data hash (cells_hash)
  - id: notebook
    resource: analysis/06_tlrn_study4_on_colab.ipynb
    title: The Colab notebook that made the fits
  - id: third
    resource: /findings/tlrn-study3-results.md
    title: The third study's results, the comparison for P0 and the first account of the data hash
  - id: fits
    resource: the 44 fits of 2026-11-25 (settings and run records, company reserves, valuation forecasts, member tables and fitted results), kept in the owner's Google Drive folder ibnr_colab_cf/study4 and not in this repository; ibnr 0.7.7, jax 0.11.1, Python 3.13.16, cas-schedule-p publish 20260613_041006, one fit on an NVIDIA L4 and the rest on an NVIDIA A100-SXM4-80GB; the 20 network-off rows are the third study's fits
    title: The 44 finished fits
---

# Result

The pre-registered verdicts for the companion's revised model (main row `tlrn_cr_refit`), as
the evaluator printed them: **P1, not distinguishable on B4 and on C4; P2, not distinguishable
on B4 and on C4.** The overall statements the evaluator printed are "not distinguishable; B4 is
listed for information (C4 not distinguishable; B4 not distinguishable)" for P1 and "not
distinguishable (C4 not distinguishable; B4 not distinguishable)" for P2. The network question
is "network adds value" on B4 and "not distinguishable" on C4; the blend question is "not
distinguishable" on both.[^eval]

Status of each verdict, as printed. B4: "confirmatory of a fixed specification, not
out-of-sample". C4: "out-of-sample in accident years only, and weaker than that (the third
study's result on this cohort existed before these rules were written, the companion's holdout
run is in the same commit as his embedding removal, and it shares 60 of its 83 companies and
155 of its 232 pairs with B4)". No verdict is provisional: every arm has draws 0 to 9.[^eval]

In plain words:

* **P1 on B4: not distinguishable, with the label "date-dependent: better at valuation 10,
  worse on the unweighted mean".** At valuation 10 the refit lowers Pool_APE by 0.72 points
  (interval -2.61 to -0.21) and 10 of 10 draws favour it, but the pooled interval over the four
  valuations (-1.44 to +0.42) includes zero, so the rule's second condition fails. The same
  rule on the stratified intervals gives not distinguishable.
* **P1 on C4: not distinguishable.** The refit blend is worse than the blend without the refit
  by 0.18 points at valuation 10 (interval -0.86 to +3.35) and only 2 of 10 draws favour the
  refit (8 needed). The same rule on the stratified intervals gives "hurts"; the verdict above
  is the one the evaluator prints for the rule on the main intervals.
* **P2 on B4: not distinguishable.** Condition i fails: the valuation 10 interval against `mack`
  is [-5.0128%, +1.7196%]. The other three conditions hold. The same rule on the stratified
  intervals gives "effective".
* **P2 on C4: not distinguishable.** Conditions i, iii and iv fail (against `mcl` the valuation
  10 difference is +0.82 points, interval -2.39 to +3.05). The stratified intervals also give
  not distinguishable.
* **The network** (refit blend against the same model with the network switched off) adds value
  on B4 (valuation 10 interval -5.49 to -1.03) and is not distinguishable on C4 (-8.75 to
  +1.68). The blend against the raw network is not distinguishable on either cohort.
* **Replication of the companion's figures: "partially replicates" on both cohorts.** The
  details are below. A miss of R1 changes no verdict.
* This note says nothing about why any of these results came out as they did.

# Numbers

Pool_APE is the sum of the absolute errors of the company reserves over the sum of the actual
reserves, so it is a dollar-weighted error and company 1767 dominates it (67.3% of the actual
valuation 10 reserve on B4, 66.7% on C4). Each model figure is the Pool_APE of the average of
the ten draws' forecasts. "Rolling" is the plain mean over the four valuations (6, 7, 8 and 10).
All figures in percent, from the "Pool_APE of the draw average" tables of
`report.txt`.[^eval]

| Arm | B4 valuation 10 | B4 rolling | C4 valuation 10 | C4 rolling |
|---|---|---|---|---|
| `tlrn_cr_refit` (blend with refit; the model) | 3.773 | 4.279 | 5.788 | 11.019 |
| `tlrn_cr_norefit` (same members, blend before the refit) | 4.491 | 4.133 | 5.607 | 10.612 |
| `raw_network` (raw network with refit) | 3.098 | 4.276 | 6.128 | 8.277 |
| `raw_network_norefit` | 4.545 | 6.247 | 5.752 | 7.128 |
| `tlrn_cr_factors` (network switched off; the third study's fits) | 5.638 | 5.007 | 5.765 | 10.620 |
| `mack` (chain ladder) | 5.481 | 4.896 | 11.947 | 14.838 |
| `mcl` (multivariate chain ladder) | 5.410 | 4.926 | 4.969 | 13.168 |

The model against the benchmarks at each of the four valuations:

| Cohort, arm | valuation 6 | valuation 7 | valuation 8 | valuation 10 |
|---|---|---|---|---|
| B4 `tlrn_cr_refit` | 5.291 | 5.059 | 2.991 | 3.773 |
| B4 `mack` | 6.102 | 4.498 | 3.503 | 5.481 |
| B4 `mcl` | 6.217 | 4.460 | 3.619 | 5.410 |
| C4 `tlrn_cr_refit` | 13.894 | 13.613 | 10.782 | 5.788 |
| C4 `mack` | 13.781 | 16.701 | 16.923 | 11.947 |
| C4 `mcl` | 13.843 | 16.815 | 17.044 | 4.969 |

Draw spread (one draw's own Pool_APE at valuation 10, minimum to maximum, with the median for
the refit blend): B4 refit 3.377 to 4.101 (median 3.799), no refit 4.155 to 4.620, raw network
3.068 to 4.007, raw network without refit 4.523 to 5.433, network off 5.532 to 5.739; C4 refit
4.012 to 7.185 (median 5.731), no refit 4.881 to 6.350, raw network 3.667 to 8.298, raw network
without refit 4.593 to 7.104, network off 4.888 to 6.300. On B4 the refit blend over-estimates
55 of the 84 companies at valuation 10; on C4, 63 of 83.[^eval]

# The verdicts, with their intervals

Differences are in points of Pool_APE, row minus comparison (negative favours the row), with a
95% company-bootstrap interval, from `verdicts.json` and `report.txt`.[^eval]

| Question | B4 | C4 |
|---|---|---|
| P1, refit minus no refit, valuation 10 | -0.72, [-2.61, -0.21] | +0.18, [-0.86, +3.35] |
| P1, pooled over the four valuations | +0.15, [-1.44, +0.42] | +0.42, [-0.85, +0.99] |
| P1, draws favouring the refit (8 needed) | 10 of 10 | 2 of 10 |
| P1, valuation 10 without company 1767 | -1.75, [-2.90, -0.18] | +1.93, [-1.18, +3.76] |
| P1 conditions a, b, c, d | yes, no, yes, yes | no, no, no, no |
| P1 verdict | not distinguishable ("date-dependent: better at valuation 10, worse on the unweighted mean") | not distinguishable |
| P1, same rule on the stratified intervals | not distinguishable | hurts |
| P2, refit minus `mack`, valuation 10 | -1.71, [-5.01, +1.72] | -6.16, [-7.61, +1.35] |
| P2, refit minus `mcl`, valuation 10 | -1.64, [-5.46, -1.01] | +0.82, [-2.39, +3.05] |
| P2, pooled, against `mack` | -0.69, [-2.22, -0.24] | -3.89, [-4.50, -0.44] |
| P2, pooled, against `mcl` | -0.72, [-3.10, -0.35] | -1.83, [-2.53, -0.83] |
| P2, median draw at valuation 10 against `mack` and `mcl` | 3.799% against 5.481% and 5.410% | 5.731% against 11.947% and 4.969% |
| P2 conditions i, ii, iii, iv | no, yes, yes, yes | no, yes, no, no |
| P2 verdict | not distinguishable | not distinguishable |
| P2, same rule on the stratified intervals | effective | not distinguishable |
| P2, raw refit network by the same rule (secondary row) | not distinguishable | not distinguishable |
| network question (valuation 10; pooled; without 1767) | **network adds value** ([-5.49, -1.03]; [-4.20, -0.27]; [-6.51, -0.69]) | not distinguishable ([-8.75, +1.68]; [-5.21, +1.43]; [-10.00, +1.60]) |
| blend question (valuation 10; pooled; without 1767) | not distinguishable ([-0.57, +2.35]; [-1.77, +1.01]; [-1.24, +2.97]) | not distinguishable ([-2.12, +0.19]; [-0.55, +2.96]; [-2.35, +0.33]) |

Without company 1767, P2 on B4 reads -2.44 [-5.66, +2.56] against `mack` and -2.94
[-6.88, -1.04] against `mcl`; on C4 -2.64 [-6.64, +2.50] against `mack` and +1.03 [-3.16, +3.59]
against `mcl`.

# P0: the fourth study minus the third (descriptive)

P0 has no threshold and no label. It sets the fourth study's saved fits beside the third
study's for the same cohort, draw, valuation and company. "Fourth minus third" is in points of
valuation 10 Pool_APE, per draw; a negative number means a smaller error in the fourth
study.[^eval][^third]

| Row, cohort | median per draw, range | draw averages | dollar-pooled | without 1767 | companies with a smaller error in the fourth study |
|---|---|---|---|---|---|
| refit blend, B4 | -0.215, [-0.728, +0.143] | -0.249 [-0.543, +0.712] | +0.393 [-0.314, +0.508] | +0.220 [-0.657, +0.827] | 29 of 84 |
| no-refit blend, B4 | -0.139, [-0.472, +0.050] | -0.213 [-0.610, +1.331] | +0.081 [-0.650, +0.357] | +0.785 [-0.199, +1.473] | 34 of 84 |
| raw network, B4 | +0.128, [-0.480, +0.917] | -0.091 [-0.978, +1.500] | +0.990 [-0.485, +1.272] | +0.456 [-1.396, +1.764] | 24 of 84 |
| refit blend, C4 | +0.282, [-1.150, +2.643] | +0.571 [-0.469, +1.868] | +0.473 [+0.126, +0.806] | +0.561 [-0.746, +2.357] | 23 of 83 |
| no-refit blend, C4 | +1.653, [-0.649, +2.477] | +1.456 [-1.179, +2.279] | +1.166 [+0.024, +1.474] | -0.186 [-1.429, +2.329] | 18 of 83 |
| raw network, C4 | +0.305, [-1.088, +3.634] | +0.779 [-0.561, +2.560] | +0.758 [+0.153, +0.945] | +0.856 [-0.945, +3.209] | 21 of 83 |

The evaluator prints the third study's own draw-to-draw spread of the refit blend at valuation
10: 0.53 points on B and 1.95 points on C, and says that a difference smaller than this is not
interpretable. The companion's own move, for reading only: the blend went from 3.93 to 3.37
percent, a fall of 0.56 points. The evaluator's reading (5) on B4 is that the P0 difference at
valuation 10 is no larger than the third study's draw spread, so removing the embedding did not
detectably move ibnr's level; this says nothing about why the companion's moved. The per-draw
values are in `B_p0_per_draw.csv` and `C_p0_per_draw.csv`.

# Replication of the companion's reported figures

At valuation 10, per cohort, as the evaluator printed it. `m` is the draw average of the refit
blend and `[lo, hi]` the range of the ten draws' own Pool_APE.[^eval]

| | B4 | C4 |
|---|---|---|
| label | **partially replicates** | **partially replicates** |
| `m`, range | 3.773, [3.377, 4.101] | 5.788, [4.012, 7.185] |
| companion's figure | 3.37 | 5.75 |
| refit effect D10 (refit minus no refit) | -0.718 (R2 needs D10 <= -0.5) | +0.181 (R2 needs D10 >= +0.5) |
| R1 (level) | False | True |
| R2 (refit effect) | True | False |
| R3 (raw network; secondary, changes no label) | False (against 4.31) | True (against 6.29) |
| the companion's three seeds (descriptive) | mean 3.62, range 3.37 to 3.88 | mean 6.96, range 5.75 to 7.67 |

How the evaluator prints the reading, using the pre-registration's wording. B4: "R1 misses
without falling in a band the pre-registration read in advance (draw average 3.77, target in
the range of the draws: False); the label follows the rule"; "P1 and P2 are decided by their
own rules; a miss of R1 changes no verdict". C4: "R2c is judged on the direction and size of
the refit effect".

The companion's figures beside ours (descriptive, not a decision rule). His figures are one fit
at his primary seed, the lowest of his three; ours are the per-draw values of ten complete fits;
the random streams cannot be aligned. "Below" and "above" count the ten draws against his
figure.[^eval]

| Quantity (valuation 10) | companion | ibnr mean, min to max | draws below, above |
|---|---|---|---|
| B4 refit blend, Pool_APE (%) | 3.37 | 3.7745, 3.3770 to 4.1007 | 0, 10 |
| B4 raw network, Pool_APE (%) | 4.31 | 3.3738, 3.0682 to 4.0072 | 10, 0 |
| B4 blend without refit, Pool_APE (%) | 4.37 | 4.4920, 4.1549 to 4.6195 | 2, 8 |
| B4 median alpha | 0.4618 | 0.4694, 0.4347 to 0.5008 | 5, 5 |
| C4 refit blend, Pool_APE (%) | 5.75 | 5.8045, 4.0119 to 7.1851 | 5, 5 |
| C4 raw network, Pool_APE (%) | 6.29 | 6.1423, 3.6674 to 8.2977 | 6, 4 |
| C4 blend without refit, Pool_APE (%) | 4.73 | 5.6255, 4.8811 to 6.3505 | 0, 10 |
| C4 median alpha | 0.6617 | 0.6971, 0.6324 to 0.7556 | 2, 8 |

His best classical figure (CL+MCL) is 5.41 on B4 and 4.97 on C4; ibnr's `mcl` gives 5.4098 and
4.9693. The number of companies matches at every valuation (90, 89, 87, 84 on B4; 84, 84, 83,
83 on C4).

# Single-draw rows (descriptive)

Four fits on B4 at draw 0, with no verdict, no interval and no statement that a component helps
or hurts. Pool_APE as printed, in percent, with the companion's means over the four valuations
beside.[^eval]

| Row | setting | blend, valuation 10 | blend, mean over valuations | network, valuation 10 | network, mean over valuations | blend valuation 10 minus main draw 0 | companion's blend, network (means) |
|---|---|---|---|---|---|---|---|
| `tlrn_cr_refit` | main row, draw 0 | 3.937 | 4.595 | 3.115 | 5.352 | 0.000 | 4.40, 4.71 |
| `tlrn_cr_noline` | no cross-line attention | 4.628 | 4.086 | 4.363 | 4.310 | +0.691 | 4.10, 4.47 |
| `tlrn_cr_noay` | no accident-year attention | 4.375 | 4.442 | 3.781 | 4.493 | +0.438 | 4.36, 3.76 |
| `tlrn_cr_paid` | paid only (five incurred and case channels zeroed) | 5.569 | 5.094 | 6.681 | 6.319 | +1.632 | 5.28, 6.73 |

**Rerun control** (one fit, a scale and not a distribution), valuation 10. Blend: third study
draw 0 4.196%; rerun of the third study's model 4.073% (rerun minus third -0.122 points);
fourth study draw 0 3.937% (P0 draw 0: -0.259 points). Network: third study draw 0 3.467%;
rerun 3.276% (-0.191 points); fourth study draw 0 3.115% (P0 draw 0: -0.352 points).[^eval]

# Precision: what gain could be detected

The width of each 95% interval and the gain it could just detect (about half the width), in
points of Pool_APE, from the evaluator's precision tables.[^eval]

| Statistic, benchmark | B4 width, detectable gain | C4 width, detectable gain |
|---|---|---|
| valuation 10, `mack` | 6.73, 3.37 | 8.96, 4.48 |
| pooled, `mack` | 1.98, 0.99 | 4.06, 2.03 |
| rolling, `mack` | 1.89, 0.95 | 3.79, 1.90 |
| valuation 10, `mcl` | 4.45, 2.23 | 5.45, 2.72 |
| pooled, `mcl` | 2.75, 1.38 | 1.70, 0.85 |
| rolling, `mcl` | 2.70, 1.35 | 1.89, 0.94 |

At valuation 10 the detectable gain against a benchmark is 2.23 to 3.37 points on B4 and 2.72
to 4.48 points on C4.

# What the data can and cannot support

* **Supports.** The listed verdicts and the figures above, as printed. On B4 the refit blend
  has a lower valuation 10 error than `tlrn_cr_norefit` in 10 of 10 draws. On both cohorts the
  draw-average blend is below `mack` at valuation 10; on B4 it is also below `mcl`, on C4 it is
  above `mcl` (5.788% against 4.969%).
* **Does not support** a claim that the model beats the chain ladder or the multivariate chain
  ladder on either cohort under the pre-registered rule, nor that the refit helps or hurts on
  either cohort. The B4 and C4 P1 results differ in sign at valuation 10, and on each cohort
  at least one required interval includes zero.
* **Does not support** any reason for any result, including the effect of removing the
  embedding. P0 is descriptive, and its pairing between studies is in starting weights and
  seeds, not in training (pre-registration, P0 caveat 1).
* **Does not support** a reading of the single-draw rows as the effect of a component: one draw
  of the refit blend at valuation 10 on B4 is one of ten that range over 0.72 points (3.377 to
  4.101), and the differences between the rows are of that order or larger.

# Caveats

* **One company decides the dollar figures.** Company 1767 holds 67.3% (B4) and 66.7% (C4) of
  the actual valuation 10 reserve, so every dollar-based figure is largely a statement about
  it. The results without it are reported beside each verdict.
* **B4 is a fixed specification on the companion's own window**, so its verdict is
  "confirmatory of a fixed specification, not out-of-sample". C4 is out of sample in accident
  years only, and weaker than that (its status line above).
* **Two devices.** One fit, cohort C draw 0 of `tlrn_cr_refit`, ran on an NVIDIA L4; the other
  main fits ran on an NVIDIA A100-SXM4-80GB; the network-off rows are the third study's fits,
  made on an NVIDIA RTX PRO 6000 Blackwell Server Edition with ibnr 0.7.6. GPU fits are not bit
  for bit repeatable, and whether the device changes a result was not tested.
* **The first-fit stop.** The pre-registration says to stop and ask if draw 0 on C4 takes longer
  than 25 minutes. It took 1711.2 seconds (28.5 minutes) on the L4. The owner answered on
  2026-11-25: "switch to a faster GPU and go on (draw 0 took 28.5 minutes on an L4)". That
  answer is recorded in every later fit. The 8 hour budget stop was not reached (6.67 GPU
  hours).
* **Cells driven through a background thread.** The cells of notebook 06 were taken byte for
  byte from tag `v0.7.7` (notebook sha256 `e1b7d2d10d4a648be7b9764696dc8da347aff952edd7113850133000039e6950`)
  and run in the notebook's own kernel through a small driver cell that ran the long cells on a
  background thread.
* **Ten draws sample training randomness thinly**, the bootstrap resamples companies only (so
  it says nothing about other years), and there is no correction for the number of questions.
  The draw standard deviation of the refit blend at valuation 10 is 0.227 points on B4 and
  1.021 points on C4.
* **The factors row is reused.** The 20 `tlrn_cr_factors` rows are the third study's
  `tlrn_cf_factors` fits, not refitted, as the pre-registration's identity check allowed. The
  network question rests on them.
* **The benchmarks tie out.** The evaluator's `mack` and `mcl` valuation 10 values equal the
  third study's to about 1e-15 relative on both cohorts, and its `mack` equals an independent
  chain ladder to 2.2e-16.
* **The log-ratio statistic is not defined for every company at valuation 10** (82 or 83 of 84
  on B4 and 82 of 83 on C4, depending on the arm). Amounts are USD thousands.

# Data hash: still unexplained

Every fit record holds a data hash that differs from the one the evaluator builds from the same
publish, as in the third study: recorded `bce499efbb6bde4e` and built `2d9ecc777b088fcd` for B,
recorded `e52d50e002d50e16` and built `92f5d7a74f37a0c2` for C. The evaluator prints "data hash
differs" for every fit and, as in the third study, scores the fits anyway because the anchors,
premiums, lines and companies of every fit equal the cohort's within tolerance; the selection
counts equal the pre-registration (pairs 230, 226, 219 and 212 on B and 237, 236, 232 and 232
on C, at valuations 6, 7, 8 and 10, with the registered company counts). What is new: the
fourth study's run records hold the library versions of the Colab image (pandas 2.2.3, numpy
2.4.6, duckdb 1.3.2, Python 3.13.16). The third study found that the laptop's locked
environment (Python 3.12.13, pandas 2.3.3, numpy 2.4.6, duckdb 1.5.3) reproduces the
evaluator's hash exactly.[^third] So the versions of pandas and duckdb differ between the two
machines; that is consistent with a library difference in how the cohort's frame is written,
and it is not shown to be the cause, nor is it evidence of a difference in the data (or proof
that there is none). The cause remains unexplained. Nothing in a rule depends on it.

# Compute

44 fits on 2026-11-25, 6.67 GPU hours against the 8 hour budget (the pre-registered planning
figure was about 3.6 GPU hours, from the third study's times). A main fit took about 950
seconds on C and about 1015 seconds on B on the A100; C draw 0 took 1711.2 seconds on the L4.
The evaluator took 59 and 60 seconds to fit the benchmarks of the two cohorts and 68 and 63
seconds per cohort overall.[^notebook]

# What would settle it

Ten draws of the single-draw rows if they look worth it (the pre-registration's study 4b),
fresh cohorts that do not share companies with B4, a size-balanced metric so that one company
does not decide, and the library versions printed beside the data hash in a Colab runtime.

[^eval]: Evaluator output
[^decision]: The questions and verdict rules, fixed before any fit
[^third]: The third study's results, the comparison for P0 and the first account of the data hash
[^notebook]: The Colab notebook that made the fits
