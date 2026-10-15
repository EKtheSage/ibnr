---
type: Finding
title: Results of the tlrn robustness study
description: Ten training draws of the accident-year tlrn model on two cohorts, against the chain ladder and the multivariate chain ladder. All four pre-registered verdicts are not distinguishable, and the model does not beat the benchmarks on pooled company reserves.
tags: [tlrn, evaluation, colab, gpu, companion-study, results]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-15T12:00:00Z }
stale_after: 2027-01-13T00:00:00Z
sources:
  - id: colab
    resource: analysis/results/tlrn_study/fits (the 40 settings and run records, the 40 company reserve tables and the manifest from the Colab runs of 2026-10-14 to 15, ibnr 0.7.4, cas-schedule-p publish 20260613_041006)
    title: The 40 finished fits
  - id: eval
    resource: analysis/results/tlrn_study/evaluation (verdicts.json, report.txt, evaluator_log.txt and the tables written by scripts/tlrn_study_eval.py)
    title: Evaluator output
  - id: decision
    resource: /decisions/tlrn-robustness-study.md
    title: The questions and verdict rules, fixed before any fit
  - id: pr174
    resource: https://github.com/EKtheSage/ibnr/pull/174
    title: PR 174 (merged), the study scripts
  - id: pr177
    resource: https://github.com/EKtheSage/ibnr/pull/177
    title: PR 177 (open when this was written), the evaluator accepts Colab fits whose data hash text differs
  - id: companion
    resource: the companion study's published company reserve table (M_T0_company_reserves.csv), read from outside this repository, nothing copied
    title: The companion's published table, cohort B only
---

# Result

All four pre-registered verdicts are **not distinguishable**, each over the full 10
draws.[^decision][^eval]

| Cohort | Row | Verdict |
|---|---|---|
| A | tlrn_8_ay | not distinguishable |
| A | tlrn_13_ay | not distinguishable |
| B | tlrn_8_ay | not distinguishable |
| B | tlrn_13_ay | not distinguishable |

What that means in plain terms:

* The accident-year model is **steady across training randomness**. The ten fits of one row
  land within 0.15 to 0.4 points of Pool_APE of each other, so it is not a lottery.
* The earlier single run in notebook 04 (`tlrn_ay`, 6.22%) was representative of
  `tlrn_13_ay`, whose draws average 6.24% on the same cohort.
* The study does **not** show the model is more effective than the chain ladder (`mack`) or
  the multivariate chain ladder (`mcl`) on pooled company reserves. Nominally `tlrn_8_ay` ties
  `mack` on cohort A and every other row is 0.1 to 0.6 points worse than `mack` and `mcl`.
  The pre-registered rule needs the 95% interval of the difference to lie wholly below zero,
  and every interval contains zero.

The evaluator ran on these fits with the change in pull request 177, which was open and not
merged when this was written.[^pr177] Without it the evaluator stops at its data hash check,
because Colab's hash text differs from the laptop's. The evaluator printed that the anchors,
premiums, lines and companies of every fit equal the cohort's within tolerance before scoring.

# Pool_APE by draw (company reserve)

Pool_APE is the sum of absolute reserve errors over the scored companies divided by the sum
of the actual reserves. The draw average is the Pool_APE of the mean of the 10 draws' company
reserves.

**Cohort A** (93 companies, 82 scored)

| | min | median | max | sd | draw average |
|---|---|---|---|---|---|
| tlrn_8_ay | 5.537% | 5.769% | 5.939% | 0.129 pts | 5.765% |
| tlrn_13_ay | 6.057% | 6.219% | 6.456% | 0.119 pts | 6.236% |
| mack | 5.773% | | | | |
| mcl | 5.661% | | | | |

`sur` refused two of the 82 companies; on the other 80 it scores 5.938%.

**Cohort B** (84 companies, all scored)

| | min | median | max | sd | draw average |
|---|---|---|---|---|---|
| tlrn_8_ay | 5.688% | 5.792% | 5.837% | 0.045 pts | 5.777% |
| tlrn_13_ay | 5.498% | 5.633% | 5.797% | 0.100 pts | 5.622% |
| mack | 5.481% | | | | |
| mcl | 5.410% | | | | |
| sur | 5.669% | | | | |

**Win rate over draws** (share of the 10 draws whose Pool_APE is below the benchmark's):
`tlrn_8_ay` on A beats `mack` in 50% and `mcl` in 30%. Every other row and cohort beats
neither `mack` nor `mcl` in any draw. `tlrn_13_ay` on B beats `sur` in 80%.

Cohort B lies entirely inside cohort A, so the two cohorts are not independent evidence.

# Draw noise is small

* Between-draw variance is under 1% of a single draw's squared error
  (`draw_noise_share`: 0.0009 to 0.007 across the four rows).
* The median coefficient of variation of a company's reserve across draws is about 0.6 to
  0.7%.
* Averaging draws narrows the spread but does not move the middle. On cohort A `tlrn_8_ay`,
  single draws span 5.54% to 5.94%, averages of 5 draws span 5.66% to 5.87%, and the median
  stays at 5.77% throughout.

So a bigger ensemble of draws will not change the answer. That is why this finding reads
the verdicts as being about the model and not about luck.

# One company dominates

Company 1767 holds 66.3% of the actual reserve in cohort A and 67.3% in cohort B, so
Pool_APE is mostly a statement about that one company.

Excluding it, every model row beats every benchmark in 100% of draws (draw averages for the
model rows):

| | tlrn_8_ay | tlrn_13_ay | mack | mcl |
|---|---|---|---|---|
| Cohort A | 12.49% | 10.98% | 12.89% | 13.24% |
| Cohort B | 11.30% | 9.90% | 12.31% | 12.81% |

But every paired bootstrap interval still contains zero. **This scope is not one of the
pre-registered verdicts.** It is reported to show where the pooled score comes from, and it
does not change the verdicts above.

# Company by company

Across individual companies the models win against `mack` and `mcl` on 51% to 63% of
companies, with median paired absolute percentage error differences of about 0.4 to 3
points in the model's favour. Most two-sided Wilcoxon p values are between 0.06 and 0.6.
The exceptions are cohort B `tlrn_13_ay` against `mcl` (p = 0.0068) and against `sur`
(p = 0.0096), both with the dominant company included. Many tests were run, none is
corrected for multiple comparisons, and none is part of the pre-registered verdict, so
read these as hints about where to look, not as findings.

# Replication of the companion study (cohort B)

* ibnr's `mack` and `mcl` company reserves reproduce the companion's published `IBNR_CL` and
  `IBNR_CL_MCL`: largest relative differences 2.84e-15 and 3.70e-13. The actual reserves
  match exactly.[^companion]
* The companion's published transformer Pool_APE is 5.5541%. None of the `tlrn_8_ay` draws is
  below it and 30% of the `tlrn_13_ay` draws are.
* The companion reported that 48.8% of its bootstrap resamples favoured its transformer
  over `mcl`. The share for the `tlrn_13_ay` draw average here is 47.3%. The two are close,
  and both say the same thing: about a coin flip.

# Compute

40 fits, 4.4 hours of fit time in all. Every fit trained 3,000 epochs with the JAX backend
on a Colab GPU (Python 3.13.15, ibnr 0.7.4).[^colab]

| Device | Fits | Seconds per fit |
|---|---|---|
| NVIDIA RTX PRO 6000 Blackwell (Colab "G4 GPU") | 25 | 247 to 277 |
| NVIDIA A100-SXM4-40GB | 14 | 459 to 477 |
| Tesla T4 | 1 (A, tlrn_8_ay, draw 0) | 3,207 |

The Blackwell GPU was about 1.8 times faster than the A100 and about 13 times faster than
the T4 on this fit. One Colab session ended after about 1 hour 50 minutes on the A100 and
the study resumed (reported by Ethan, not recorded in the fit files). The runner keeps one
saved file per finished fit, so finished fits were not repeated. The model files (about 270
MB) are kept in Drive, not in the repository.

# What this cannot show

* One valuation date (2007-12-31) and one set of calendar years, Schedule P 1998 to 2007.
* One family of random streams: draws differ in seed, not in data or settings.
* Company-level resampling only. The bootstrap says how much the answer would move with a
  different set of companies from the same market, not with different years.
* Cohort B lies inside cohort A, so agreement between them is not a replication.
* One company holds two thirds of the reserve, so the pooled number is nearly a one-company
  test.
* ibnr's `sur` is a different estimator from the companion's SUR and is not in any verdict.

One reading these results are consistent with, but which was not tested: each member is
blended with the multivariate chain ladder (the fitted blend weight alpha was about 0.3 to
0.7 in the accident-year fits recorded in [the GPU run](/findings/tlrn-jax-tpu-run.md)),
which would pull the model toward `mcl` and shrink draw noise.

# What would settle it

More valuation dates (a rolling set of cutoffs rather than one), a cohort in which no
company holds a large share of the reserve, and data from outside Schedule P 1998 to 2007.
With one dominant company, even a model that is better on most companies cannot show it
on the pooled measure.

[^decision]: The questions and verdict rules, fixed before any fit
[^eval]: Evaluator output
[^pr177]: PR 177 (open when this was written), the evaluator accepts Colab fits whose data hash text differs
[^companion]: The companion's published table, cohort B only
[^colab]: The 40 finished fits
