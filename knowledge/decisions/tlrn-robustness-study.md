---
type: Decision
title: The tlrn robustness study, its questions and its verdict rules
description: "How ibnr tests whether the companion study's accident-year model is effective against the classical benchmarks: ten independent training draws, two cohorts, six questions and a verdict rule, all fixed before any result exists (Ethan, 2026-10-12)."
tags: [tlrn, evaluation, preregistration, colab, companion-study, decision]
status: draft
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-12T18:00:00Z }
stale_after: 2027-04-12T00:00:00Z
sources:
  - id: request
    resource: Ethan's instruction in the 2026-10-12 working session
    title: "use ibnr library to run marco's model and truly test its effectiveness against benchmarks"
  - id: gpu
    resource: /findings/tlrn-jax-tpu-run.md
    title: The accident-year fits on a Colab GPU, about 7.5 minutes each
  - id: repro
    resource: /findings/companion-notebook-reproduction.md
    title: Reproducing the companion notebook (cohort of 84 companies, MCL tie-out)
  - id: nb04
    resource: analysis/04_nn_architectures_vs_classical.ipynb
    title: Notebook 04, whose Pool_APE and cohort this study reuses
  - id: runner
    resource: scripts/tlrn_study_colab.py
    title: The Colab runner
  - id: evaluator
    resource: scripts/tlrn_study_eval.py
    title: The evaluator, whose docstring repeats these rules
---

# Decision

One run of a neural network is one draw of its training randomness. The companion study
reports one run of its accident-year model (`tlrn_ay`) and a company-reserve Pool_APE of
5.55% against 5.41% for the chain ladder with the multivariate chain ladder, and says
itself that 48.8% of its bootstrap resamples favour the transformer. That does not say
whether the model is effective. Ethan's call (2026-10-12) is to test it with ibnr: train
the model many times, on a GPU that makes a fit cheap, and compare the whole spread of
outcomes with the benchmarks.[^request]

The study is fixed here, before any real fit exists. The runner is
`scripts/tlrn_study_colab.py`, the evaluator is `scripts/tlrn_study_eval.py`, and the
how-to is `docs/tlrn-study.md`. If a rule below turns out to be wrong, change it in a new
commit that says so, not silently.

# What is run

* **Rows.** `tlrn_8_ay` (paid only) and `tlrn_13_ay` (paid, incurred and case reserve),
  both `TLRNConfig.accident_year_variant()` at its full budget: 20 members at each of 4
  valuation dates, 3,000 epochs, trained with `fit(backend="jax")`.
* **Draws.** 10 independent fits per row and cohort. Draw `k` uses fit seed
  `11 + 100003 k`, so draw 0 has the seed of the earlier single fit and no two draws share
  a member's stream. A draw is one fit; a fit already averages its 20 members.
* **Valuation.** One date, 2007-12-31, accident years 1998 to 2007, as notebook 04.[^nb04]
* **Cohort A.** Notebook 04's: 93 companies, 243 company-line pairs, cohort hash
  `08605b3c80513bca`, built from the cas-schedule-p package. 82 companies are scored,
  because `mack` and `mcl` refuse the other 11.
* **Cohort B.** The companion study's: its selection rule applied with the cutoff at the
  last known diagonal (calendar index 10), which gives 84 companies and 212 company-line
  pairs, all scored. Its cells are identical to the cas-schedule-p mart's for these lines
  and years, so the runner can build it from the mart (the default) or from the four
  cached CAS files (`--cas-dir`).[^repro] Cohort B is a subset of cohort A: its rule asks
  more of every pair, so the two cohorts are not independent evidence (see [the two
  cohorts](/findings/tlrn-study-cohorts.md) for the counts and the tie-outs).
* **Cost.** 10 draws x 2 rows x 2 cohorts is 40 fits, about 7.5 minutes each on a Colab
  GPU, about 5 hours in all, runnable in pieces and resumable.[^gpu]

# Benchmarks

`mack` (the volume-weighted chain ladder, one fit per company-line pair), `mcl` (the
multivariate chain ladder, one fit per company) and `sur`, each from ibnr's own gallery
entry. On cohort B the companion's published `IBNR_CL` and `IBNR_CL_MCL` company reserves
are also read when their table is supplied, as a check that ibnr's `mack` and `mcl` tie
out to them, and its published transformer reserve is shown beside the draws.

# Definitions

* **Reserve.** The 120-month cumulative paid minus the cumulative paid at the valuation
  date, added over the company's lines. The prediction is the model's ultimate minus the
  same anchor.
* **Pool_APE.** The sum of absolute reserve errors over the scored companies divided by
  the sum of actual reserves, as notebook 04 defines it.
* **Draw average.** The mean over the draws of the draw's company reserve.
* **Dominant company.** The company with the largest share of the cohort's actual
  reserve, found in code. Company 1767 holds 66% in cohort A.

# The six questions

Each gets numbers, for every cohort and row.

1. **Draw to draw.** Pool_APE of each draw: minimum, median, maximum and standard
   deviation, and Pool_APE of the draw average, beside `mack`, `mcl` and `sur`.
2. **Win rate.** The fraction of the draws whose Pool_APE is below `mack`'s and below
   `mcl`'s.
3. **Paired company comparison of the draw average** against `mack` and `mcl`: the share
   of companies where the model's absolute error is smaller, the median over companies of
   the paired difference in absolute percentage error, a company-level bootstrap of the
   pooled Pool_APE difference (10,000 resamples, seed 2008, the same resampled companies
   for the model and the benchmark, 95% percentile interval), and a two-sided Wilcoxon
   signed-rank p value on the paired absolute percentage errors. All of it is shown with
   the dominant company and without it.
4. **The companion's bootstrap claim.** The share of bootstrap resamples whose Pool_APE
   for the model is below the benchmark's, for each draw (minimum, median, maximum over
   draws) and for the draw average, so the reported 48.8% can be compared with a spread
   rather than a single run.
5. **Variance decomposition.** Draw noise against company differences: the share of a
   single draw's squared error that is variance between draws (an exact identity: the
   mean squared error of a draw is the squared error of the draw average plus the
   between-draw variance), the intraclass correlation of the log company reserve, and
   the Pool_APE of the average of every subset of k draws, k = 1 to 10, with the share of
   those averages that beat `mack` and `mcl`. This says whether a bigger ensemble changes
   the answer.
6. **Verdict.** One of three words per cohort and row, by the rule below.

# The verdict rule

Let `D(b)` be the 95% bootstrap interval of Pool_APE(draw average) minus Pool_APE(`b`)
on the cohort with the dominant company included, for the benchmark `b`.

* **effective**: the median-draw Pool_APE is below both `mack`'s and `mcl`'s, and `D(mack)`
  and `D(mcl)` both lie wholly below zero.
* **worse**: not effective, and `D(c)` lies wholly above zero for `c`, the benchmark with
  the lower Pool_APE of the two.
* **not distinguishable**: everything else.

The rule is mechanical so that nobody reads the result into it. Eight comparisons are
made (two rows, two benchmarks, two cohorts) and no multiplicity correction is applied;
a single "effective" is therefore weaker evidence than it reads, and an effective on
both cohorts counts once, because cohort B lies inside cohort A.

# What this cannot show

It is one valuation date and one set of calendar years, so it says nothing about other
years. The bootstrap resamples companies, so it measures company-to-company sampling
only. Ten draws sample the training randomness thinly. The model blends the multivariate
chain ladder into each member (alpha is fitted), so it is partly the benchmark. GPU fits
are not bit-for-bit repeatable, so a rerun gives different draws. Nothing has been run
on a GPU when this note is written, and no result exists.

[^request]: Ethan's instruction in the 2026-10-12 working session
[^nb04]: Notebook 04, whose Pool_APE and cohort this study reuses
[^repro]: Reproducing the companion notebook (cohort of 84 companies, MCL tie-out)
[^gpu]: The accident-year fits on a Colab GPU, about 7.5 minutes each
