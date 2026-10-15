---
type: Decision
title: The second tlrn study, on the companion's factor-head model
description: "How ibnr tests whether the companion study's newest model (the factor head with line and lag attention) beats the chain ladder and the multivariate chain ladder: ten draws of two rows on three cohorts, four valuation dates and a verdict rule, all fixed before any result exists (Ethan, 2026-10-15)."
tags: [tlrn, evaluation, preregistration, colab, companion-study, decision]
status: draft
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-15T15:00:00Z }
stale_after: 2027-04-15T00:00:00Z
sources:
  - id: request
    resource: Ethan's instruction in the 2026-10-15 working session
    title: "run the second study on his final architecture - in a new notebook, say 05 - use ibnr to reproduce his final architecture - see if there's any testing improvement we should make"
  - id: first
    resource: /decisions/tlrn-robustness-study.md
    title: The first study, whose machinery and style this one extends
  - id: cohorts
    resource: /findings/tlrn-study-cohorts.md
    title: The two cohorts of the first study (A and B)
  - id: companion
    resource: /references/companion-study-repository.md
    title: The companion study's repository, read through git objects only
  - id: runner
    resource: scripts/tlrn_study_colab.py
    title: The Colab runner, which runs this study with --study factor_head
  - id: evaluator
    resource: scripts/tlrn_study_eval.py
    title: The evaluator, whose docstring repeats these rules
---

# Decision

The first study asked about the companion study's accident-year model. That model is no
longer the companion's final one: its repository moved, on 2026-10-08, to a model with a
factor head and line and lag attention, which it reports at a company-reserve Pool_APE of
4.97% against 5.41% for the multivariate chain ladder and 5.48% for the chain ladder. The
same repository's own paired bootstrap against the multivariate chain ladder gives a 90%
interval from -3.14 to +2.43 points, and 68.4% of its resamples favour the transformer. Ethan's call
(2026-10-15) is to run the second study on that final architecture, in a new notebook
(05), built with ibnr, and to see what the testing itself should do better.[^request]

The study is fixed here, before any fit or any evaluator code exists. If a rule below turns
out to be wrong, change it in a new commit that says so, not silently. The machinery is the
first study's: the runner is `scripts/tlrn_study_colab.py` (now with `--study
factor_head`), the evaluator is `scripts/tlrn_study_eval.py` and its second-study
extension, and the first study's decision note[^first] explains the ideas that carry over
(ten independent draws, a mechanical verdict rule, provisional labelling).

# The model, and what is verified about it

Read from the companion's newest notebook (its repository at commit f74addd, 2026-10-08)
through git objects, without checking anything out. **Verified** means read in that
notebook, its run manifest or its published tables when this note was written (the
constants, the loss, the parameter count, the table figures and the two runs' outputs were
read again by the author of this note); **from the earlier review** means reported by a
read-only review of the repository that this note did not repeat; **inferred** means worked
out from the others.

* **Verified.** 13 features per cell. One axial block with attention across lines and
  across lags only (d_model 32, 2 heads, dropout 0.3); there is no accident-year
  attention, which is what separates it from the first study's model.
* **Verified.** The head reads the network output as a log development factor per
  (line, step): `logf = softplus(phi + 0.5 * net)`. `phi` is a learned line-by-9-steps
  parameter shared by all companies, starting at `softplus(phi) = 0.57 / lag`, a decaying
  start and not the company's own chain ladder factors. The forecast is the cumulative
  projection from the latest paid, with no tail factor. (From the earlier review: a
  transition that no training target supervised falls back to the pooled chain ladder
  factor, which is what `tail_policy="observed_cl"` does in ibnr.)
* **Verified.** Loss: absolute error per (example, line) over the absolute actual, plus
  0.5 times the pooled signed error over the same base, plus 0.1 times a squared error
  scaled by 0.005. Adam, learning rate 0.003 (0.03 for `phi`), 20 warmup epochs then a
  cosine schedule, gradient clipping at 1.0, 3,000 epochs, batches of 64 examples with one
  random training cutoff per epoch. Validation uses diagonals C-1 and C (the cells it
  reaches), every 5 epochs, and the blended validation error picks the checkpoint. Nothing
  stops training early.
* **Verified.** Each member forecasts `MCL + alpha * (network - MCL)` cell by cell, with
  alpha refitted as an exact weighted median clipped to [0, 1]. Twenty members are trained
  at every valuation date and the reported transformer is the plain mean of all twenty
  blended members. No member is selected. (From the earlier review: a table of seed
  selection in the repository is a leftover of an older pipeline and is not used.)
* **Verified.** 14,245 parameters per member in the companion's architecture table. The ibnr
  network built from the configuration below has 14,309 trainable parameters, exactly 64
  more: its line and lag embeddings are 1-based, so each has one extra row of 32 (5 rows
  for 4 lines, 11 for 10 lags). Its random streams differ too, so ibnr trains the same
  procedure, not the same numbers.
* **Verified, about run-to-run noise.** The committed notebook source is identical across
  its versions, and two stored runs of it report a Pool_APE of 5.04% and 4.97%, so
  run-to-run noise of about that size exists.
* **From the earlier review, about the history.** The design moved from the accident-year
  model (5.55%, worse than the multivariate chain ladder) to this one after results were
  visible, and its hyperparameters were chosen with the test data visible. This is why the
  verdicts on A and B are labelled as they are below.
* **Verified, about the headline.** Company 1767 holds 67.3% of the cohort's reserve. Without
  it the transformer is at 13.07% against 12.81% for the multivariate chain ladder and
  12.31% for the chain ladder. The transformer beats the multivariate chain ladder on 52.4%
  of companies. Its rolling table gives Pool_APE at valuations 6, 7, 8 and 10 of 8.24, 3.50,
  3.58 and 4.97 for the transformer against 6.11, 4.41, 3.61 and 5.41 for the multivariate
  chain ladder (means 5.07 and 4.88).
* **Verified, a tie-out the runner already meets.** That table scores 2,968, 3,180, 2,756
  and 9,540 cells at valuations 6, 7, 8 and 10 over its 212 pairs, which is 14, 15, 13 and
  45 cells per pair. The scored-cell rule below (diagonals after the date, development lag
  at most the date, every cell past the last diagonal at valuation 10) gives exactly those
  numbers per pair, and a test pins them; the evaluator should find the same totals in the
  `n_cells` column of cohort B's fits.
* **Inferred.** The companion's selection rule reads data through diagonal 10 at every
  valuation date, so at valuations 6, 7 and 8 the cohort is chosen with later data in view.
  This study keeps that rule unchanged, because the aim is to reproduce the companion's
  design on the same footing, and says so in its output.

# How ibnr runs it

`TLRNConfig(...)` keyword arguments, written out in `scripts/tlrn_colab.py` (no new preset
and no library change, so the study runs on the published ibnr 0.7.4; the library source
the runner was written and smoke-tested against is identical to the tag `v0.7.4`, which
`git diff v0.7.4 HEAD -- src/` shows, while the PyPI wheel itself was not installed here):

| companion choice | ibnr setting |
|---|---|
| factor head, `softplus(phi + 0.5 net)`, start 0.57 per lag | `head="ldf"`, `eps=0.5`, `init_step=0.57`, `cl_anchor=False` |
| line and lag attention, one block | `attention=None`, `cross_line=True`, `n_layers=1`, `d_model=32`, `n_heads=2`, `dropout=0.3`, `mask="unwritten_lines"` |
| pooled fallback for untrained transitions | `tail_policy="observed_cl"` |
| blend with the multivariate chain ladder, 20 members all averaged | `member="mcl_blend"`, `ensemble_size=20`, `keep=None` |
| batches of 64 examples, one cutoff per epoch | `batch_unit="example"`, `batch_size=64`, `cutoff_sampling="per_epoch"` |
| validation on C-1 and C, reached cells | `val_diagonals=2`, `scoring="reached_cells"`, `min_cutoff=2` |
| loss weights | `w_pe=0.5`, `w_mse=0.1`, `mse_scale=0.005` |
| optimiser | `lr=3e-3`, `lr_phi=3e-2`, `warmup=20`, `grad_clip=1.0`, `weight_decay=0.0`, `max_epochs=3000` |
| earlier valuation dates | `calibration="retrain_per_valuation"`, `calibration_cutoffs=(6, 7, 8)`, `calibration_horizons=(2, 3, 4)` |

The 13-feature row is the roles `incurred_field="incurred_loss"` and
`case_field="case_reserve"` at fit time, as `tlrn_13` is configured in notebook 04. Early
stopping cannot fire: `patience=600` counts validation checks and `check_every=5`, so 600
checks without an improvement is the whole 3,000 epochs. The first study found the same
("members still training 80/80"), and a test pins the arithmetic. Draw `k` of a fit uses
seed `11 + 100003 k`; inside a fit member `m` is seeded `seed + 1000 m` and the valuation
date `v` protocol `seed + 10000 v`, so no two draws share a stream. The valuation dates
are 6, 7, 8 and 10 (10 is the last known diagonal, 2007-12-31 on cohorts A and B).

# What is run

* **Rows.** `tlrn_13_fh`, the main row: the model above. `tlrn_13_fh_factors`, a
  comparison row: the same configuration with `factors_only=True`, which switches the
  network off, so the member is the head's pooled factors blended with the multivariate
  chain ladder by the fitted alpha. It answers how much of any gain the network adds
  beyond the pooled factors and the blend. The raw network (`pred_net` of the main fits,
  the forecast before the blend) is read from the main row's fits and needs no extra fit.
* **Draws.** 10 independent fits per cohort and row; draw `k` uses fit seed `11 + 100003 k`,
  as in the first study.
* **Cohort A.** Notebook 04's: 93 companies, 243 pairs, 82 scored (selection cutoff 5).
* **Cohort B.** The companion's: 84 companies, 212 pairs, all scored, inside A.
* **Cohort C.** New, and the only cohort that can support an out-of-sample claim. Accident
  years 1988 to 1997, the same four lines, the companion's selection rule with the cutoff
  at the last known diagonal (10), valuation 1997-12-31. It is built from the same
  cas-schedule-p publish (20260613_041006), which carries those accident years as complete
  10-by-10 squares with every later calendar year observed (statement years to 2006), so the
  outcomes are real later development and not constructed. Selection counts 675, 405, 322
  and 232 pairs (83 companies, all scored; pairs hash `4a613cc4cb5f0f94`), and company 1767
  holds the largest reserve here too. The cohort is not independent of A and B in its
  companies: 67 of its 83 companies also appear in A and 60 in B (155 of the 232 pairs are
  also B's pairs). It is independent in accident years and calendar time.
* **Cost.** 10 draws x 2 rows x 3 cohorts is 60 fits (40 on cohorts A and B alone). A fit
  trains 20 members at each of 4 valuation dates. The first study measured about 250 s on
  a Blackwell GPU (Colab G4), about 460 s on an A100 and about 3,200 s on a T4 for the
  accident-year model; this model has no accident-year attention, so those are estimates,
  not measurements of this one.

# Benchmarks

`mack` (the volume-weighted chain ladder, one fit per company-line pair) and `mcl` (the
multivariate chain ladder, one fit per company), from ibnr's gallery, fitted on the data
known at each valuation date and scored on exactly the cells the transformer is scored on
(diagonals after the valuation date, development lag at most the valuation date, at
valuation 10 every cell past the last diagonal). The first study's `sur` is not repeated.
The companion's published rolling table is read through a path argument as a tie-out of
the benchmarks only.

# The statistics

Let a company's reserve error at valuation `v` be its predicted minus its actual total over
the scored cells. **Pool_APE** at `v` is the sum of the absolute errors over the scored
companies divided by the sum of the actuals. The **draw average** at `v` is the mean over
the draws of each company's prediction. The **dominant company** is the one with the
largest actual reserve, found in code. The **bootstrap** resamples companies 10,000 times
with a fixed seed, takes the same resampled companies for the model and the benchmark, and
reports the 2.5th and 97.5th percentiles of the difference model minus benchmark. For the
rolling statistic the resample keeps a company's valuation dates together and averages
Pool_APE over valuations 6, 7, 8 and 10.

# The verdict rule

For the main row `tlrn_13_fh` only, on each cohort, from the draw average:

* **effective** needs all four of: (i) at valuation 10 the Pool_APE difference has a 95%
  interval wholly below zero against both `mack` and `mcl`; (ii) the mean over valuations
  6, 7, 8 and 10 of the Pool_APE difference has an interval wholly below zero against both;
  (iii) the median over the draws of the valuation-10 Pool_APE is below both benchmarks;
  (iv) with the dominant company left out, the valuation-10 difference interval is not
  wholly above zero against either benchmark.
* **worse** if the interval in (i) or (ii) is wholly above zero against either benchmark.
* **not distinguishable** otherwise.

A labelling rule sits beside it and changes nothing: unless a cohort holds exactly draws 0
to 9, the evaluator prints the word as provisional with the draw count and the missing draw
numbers. Three cohorts are judged and no multiplicity correction is applied. A and B are
not independent (B lies inside A), and C shares companies with both, so agreement across
cohorts counts less than it reads.

**What each cohort's verdict means.** The companion chose its design and hyperparameters
with the A and B test data in view. A verdict on A or B is therefore labelled **confirmatory
of a fixed specification, not out-of-sample**: it says whether that specification, run
many times, beats the benchmarks on the data it was tuned on. The verdict on C is the only
out-of-sample claim, with two caveats that go beside it: C is the older accident-year
window, and it shares most of its companies with A and B.

# Reported, not in the verdict (fixed in advance)

Pool_PE (bias); the median absolute percentage error; the mean absolute error; the mean
absolute log of predicted over actual reserve, over companies with positive reserves; the
share of companies where the model has the smaller absolute percentage error; Pool_APE
within quartiles of actual reserve; and the valuation-10 difference with the dominant
company left out (the company holding 66 to 67% of a reserve in A and B). The **mean
absolute percentage error (MAPE)** and the **reserve-weighted root mean squared error
(wRMSE)** are shown as descriptive only: the companion's claimed gains are largest on
them, and they are driven by very small reserves and by one large company. Comparison rows
are reported against the main row: the raw network, `tlrn_13_fh_factors`, and the first
study's accident-year rows when their fits exist.

# What this cannot show

The companion's design was tuned on A and B, so only C is a test the specification could
fail for a reason it was not tuned against, and C overlaps A and B in companies. The
bootstrap resamples companies, so it measures company-to-company sampling and says nothing
about other years beyond what C adds. Ten draws sample training randomness thinly. The
model blends the multivariate chain ladder into every member (alpha is fitted), so it is
partly the benchmark. GPU fits are not bit-for-bit repeatable, so a rerun gives different
draws. Nothing has been run on a GPU for this study when this note is written, and no
result exists.

[^request]: Ethan's instruction in the 2026-10-15 working session
[^first]: The first study, whose machinery and style this one extends
