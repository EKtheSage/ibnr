---
type: Finding
title: The companion study's final model is the factor-head model
description: On 2026-10-08 the companion study replaced its accident-year notebook with a factor-head model (line and lag attention, a learned starting factor shape, a blend with the multivariate chain ladder, 20 members averaged); what it is, what it scores, and which of those facts were read directly and which were not.
tags: [tlrn, companion-study, architecture, reference]
status: draft
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-15T17:30:00Z }
stale_after: 2027-04-15T00:00:00Z
sources:
  - id: repo
    resource: the companion study's repository, branch master at commit f74addd (2026-10-08), read through git objects only
    title: Code/tlrn_pipeline_final.ipynb, Exhibits/run_manifest.json and Exhibits/tab/*.csv
  - id: decision
    resource: /decisions/tlrn-factor-head-study.md
    title: The second study, which runs this model through ibnr
  - id: reproduction
    resource: /findings/companion-notebook-reproduction.md
    title: The earlier reproduction of the accident-year notebook
  - id: location
    resource: /references/companion-study-repository.md
    title: Where the companion study's files live
---

# Result

The companion study's newest model is not the accident-year variant that ibnr's
0.7.3 work reproduced. Its repository moved on 2026-10-08, the same day as its last run, to
a model with a different head and no accident-year attention. The notebook that earlier
notes called the best one, `Code/tlrn_pipeline_best.ipynb`, was deleted that day (commit
`bf4e52c`); `Code/tlrn_pipeline_final.ipynb` is now the notebook, and commit `f74addd` re-ran
it and regenerated the published tables and figures.[^repo]

How to read the labels below. **Verified** means the author of this note read it in the
notebook source, the run manifest or a published table. **From the earlier review** means a
read-only review of the repository reported it and this note did not repeat the reading.
**Inferred** means worked out from the others.

# The model

* **Verified.** Thirteen features per cell. One block of attention across lines and across
  lags (model width 32, 2 heads, dropout 0.3). There is no attention across accident years.
* **Verified.** The head reads the network output as a log development factor for each line
  and step: `log f = softplus(phi + 0.5 * net)`. `phi` is a learned line-by-9-steps table that
  every company shares. It starts so that `softplus(phi) = 0.57 / lag`, a decaying curve. It
  does not start from a company's own chain ladder factors, and there is no anchor to them.
  The forecast is a cumulative projection from the latest paid amount, with no tail factor.
* **From the earlier review.** A step that no training target supervised falls back to the
  pooled chain ladder factor. ibnr's `tail_policy="observed_cl"` is the same idea.
* **Verified.** The network has 14,245 parameters per member, according to the run manifest
  and the architecture table. ibnr's version of the same configuration has 14,309, which is
  exactly 64 more, because its line and lag embeddings have one extra row each.
* **Verified.** The loss is, per example and line, the absolute error over the absolute
  actual, plus 0.5 times the pooled signed error over the same base, plus 0.1 times a squared
  error divided by 0.005.
* **Verified.** Adam with a learning rate of 0.003 (0.03 for `phi`), 20 warmup epochs then a
  cosine schedule, gradient clipping at 1.0, 3,000 epochs, batches of 64 examples with one
  random training cutoff per epoch. Validation uses the last two diagonals every 5 epochs and
  the blended validation error picks the checkpoint. Nothing stops training early.
* **Verified.** Each member forecasts `MCL + alpha * (network - MCL)` for every cell, where
  `MCL` is the multivariate chain ladder's forecast and `alpha` is refitted as an exact
  weighted median and clipped to [0, 1]. The published median alpha is 0.627.
* **Verified.** 20 members are trained at each valuation date and the reported model is the
  plain average of all 20. No member is selected. **From the earlier review:** a table of seed
  selection in the repository (`T_T1_seed_selection.csv`) is left over from the older R
  pipeline and is not used by this notebook.

# What it scores

* **Verified.** The cohort is the same as ibnr's cohort B: 84 companies, 212 pairs and the
  audit counts 668, 414, 297 and 212. Its recorded hash (`447b7bf355b1ca0d`) differs from
  ibnr's (`de51bae56cca563c`) only because the line names are spelled differently when the
  pairs are hashed.
* **Verified.** Valuation dates 6, 7, 8 and 10, where 10 is the last known year end
  (2007-12-31) and 6, 7 and 8 are the year ends of 2003, 2004 and 2005. At a date before the
  last, the cells scored are those on later diagonals up to the last known one, at
  development lag no greater than the date; the actuals are the known cells.
* **Verified.** Company-reserve Pool_APE (the published table): 4.969% for the model against
  5.410% for the chain ladder with the multivariate chain ladder, 5.456% for the chain ladder
  with SUR, 5.481% for the chain ladder and 5.556% for the network alone. By valuation date
  (6, 7, 8, 10) the model scores 8.24%, 3.50%, 3.58% and 4.97% against 6.11%, 4.41%, 3.61% and
  5.41% for the multivariate chain ladder (means 5.07% and 4.88%).
* **Verified.** Company 1767 holds 67.3% of the reserve. Without it the model scores 13.07%
  against 12.81% for the multivariate chain ladder and 12.31% for the chain ladder, and the
  model beats the multivariate chain ladder on 52.4% of companies.
* **Verified, with a correction to the earlier wording.** The repository's own paired
  bootstrap of the model against the multivariate chain ladder gives a 90% interval from
  -3.14 to +2.43 points, and 68.4% of the resamples favour the model.
* **Verified.** The repository's rolling version of that bootstrap, on the mean Pool_APE over
  the four valuation dates, gives the model 5.07% against 4.88% for the multivariate chain
  ladder: a difference of +0.19 points with a 90% interval from -1.02 to +0.93, and 29.1% of
  the resamples favour the model. So its own rolling comparison does not favour the model.
* **Verified.** The committed notebook source is identical across two stored runs, and they
  report 5.04% and 4.97%, so run-to-run noise of about that size exists.

# What is not known

* **From the earlier review.** The design moved from the accident-year model (5.55%, worse than
  the multivariate chain ladder) to this one after results were visible, and the
  hyperparameters were chosen with the test data visible. Nothing re-read here confirms the
  order of events, so treat it as likely, not settled.
* **Inferred.** The companion's cohort rule reads data through the last diagonal at every
  valuation date, so at the earlier dates the cohort is chosen with later data in view.
* **Inferred.** Because of both points, a result on this cohort is not out-of-sample. That is
  why [the second study](/decisions/tlrn-factor-head-study.md) adds a cohort of other
  accident years (1988 to 1997) and labels the verdicts on the old cohorts as confirmatory of a
  fixed specification.

[^repo]: Code/tlrn_pipeline_final.ipynb, Exhibits/run_manifest.json and Exhibits/tab/*.csv
