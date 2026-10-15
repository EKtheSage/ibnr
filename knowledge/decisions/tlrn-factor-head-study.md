---
type: Decision
title: The second tlrn study, on the companion's factor-head model
description: "How ibnr tests whether the companion study's newest model (the factor head with line and lag attention) beats the chain ladder and the multivariate chain ladder: ten draws of two rows on three cohorts, four valuation dates and a verdict rule, all fixed before any result exists (Ethan, 2026-10-15). Amendment 1, made before any fit, adds the smallest difference the primary verdict can detect, a paired question on whether the network adds value, and cohort C first."
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
    resource: scripts/tlrn_study2_eval.py
    title: The evaluator, whose docstring repeats these rules (it reuses the first study's statistics in scripts/tlrn_study_eval.py)
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
factor_head`), the evaluator is `scripts/tlrn_study2_eval.py`, which reuses the first study's
`scripts/tlrn_study_eval.py` statistics, and the first study's decision note[^first]
explains the ideas that carry over (ten independent draws, a mechanical verdict rule,
provisional labelling).

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
* **Verified, about the two stored runs.** The committed notebook source is identical across its
  versions, and two stored runs of it report a Pool_APE of 5.04% and 4.97%. That change may
  reflect a precision or matrix-multiplication setting or run-to-run variation; git cannot
  say which.
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
* **Cohort C.** New, and out-of-sample in accident years (not tuned on as far as we know,
  assumed, not known; see Amendment 1). Accident
  years 1988 to 1997, the same four lines, the companion's selection rule with the cutoff
  at the last known diagonal (10), valuation 1997-12-31. It is built from the same
  cas-schedule-p publish (20260613_041006), which carries those accident years as complete
  10-by-10 squares with every later calendar year observed (statement years to 2006), so the
  outcomes are real later development and not constructed. Selection counts 675, 405, 322
  and 232 pairs (83 companies, all scored; pairs hash `4a613cc4cb5f0f94`), and company 1767
  holds the largest reserve here too. The cohort is not independent of A and B in its
  companies: 67 of its 83 companies also appear in A and 60 in B (155 of the 232 pairs are
  also B's pairs). It is independent in accident years only: its outcomes sit in statement
  years 1998 to 2006, so it overlaps A and B in calendar time too.
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
claim out-of-sample in accident years, with two caveats that go beside it: C is the older accident-year
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

The companion's design was tuned on A and B. Cohort C is not tuned on as far as we know
(assumed, not known), and it overlaps A and B in companies and calendar time. The
bootstrap resamples companies, so it measures company-to-company sampling and says nothing
about other years beyond what C adds. Ten draws sample training randomness thinly. The
model blends the multivariate chain ladder into every member (alpha is fitted), so it is
partly the benchmark. GPU fits are not bit-for-bit repeatable, so a rerun gives different
draws. Nothing has been run on a GPU for this study when this note is written, and no
result exists.

# Amendment 1 (2026-10-15, before any fit of this study has been run)

This amendment is written before any fit of this study exists on any machine: no fit has
been run on a GPU, no result exists, and the evaluator has only ever seen CPU smoke fits
of a few epochs. It adds to the rules above and changes none of them. The primary verdict
rule, the rows, the draws, the valuation dates and the statistics stand as registered, and
the evaluator's docstring keeps the text it was committed with. Where a sentence above
claims more than this amendment allows (for example the calendar-time sentence on cohort C),
read it as corrected here.

## A1. What the primary verdict can detect

The first study measured the paired company-bootstrap interval of the Pool_APE difference
between the draw average and each benchmark, at valuation 10, on cohorts A and B (its
final run: `verdicts.json` and the log of that run, rows `tlrn_8_ay` and `tlrn_13_ay`).
The 95% intervals, in points of Pool_APE, with the dominant company 1767 holding 66.3% to
67.3% of the reserve:

| cohort and row | against `mack` | width | against `mcl` | width |
|---|---|---|---|---|
| A, `tlrn_8_ay` | -4.82 to +6.13 | 10.95 | -3.46 to +2.42 | 5.88 |
| A, `tlrn_13_ay` | -5.66 to +3.62 | 9.28 | -5.41 to +1.86 | 7.27 |
| B, `tlrn_8_ay` | -5.15 to +5.27 | 10.43 | -3.78 to +1.62 | 5.40 |
| B, `tlrn_13_ay` | -6.30 to +3.40 | 9.70 | -6.28 to +1.52 | 7.81 |

So the intervals were 5.4 to 11.0 points wide: 9.3 to 11.0 against `mack` and 5.4 to 7.8
against `mcl`. For an interval to lie wholly below zero (condition (i)), the point
difference has to sit below zero by about the distance from the point difference to the
interval's upper end. In those runs that distance was 1.3 to 6.1 points (against `mcl`
1.3 to 2.3, against `mack` 3.2 to 6.1). The factor-head model's intervals are not known
until it is run, and its rolling statistic (condition (ii)) may be narrower or wider, so
these figures are the best prior, not a prediction. They do show the scale: the primary
verdict on a single cohort can detect a gain of roughly one and a half to six points of
Pool_APE and not much less. Because `effective` needs both benchmarks cleared, the binding
distance is the one against `mack`: for the primary verdict the figure to remember is about
three points or more. The first study's result files were local to that run and are not
committed to the repository, so the intervals above are quoted from them and cannot be
followed from here.

A gain of about half a point, which is the size of the companion's headline gain over the
multivariate chain ladder (4.97% against 5.41%), cannot be detected at that precision.
"Not distinguishable" on cohort A or B is therefore the **expected** result of this design,
whether or not the model is slightly better, and it must not be read as news, as evidence
that the model does not work, or as evidence that it works the same as the benchmarks. This
is why the paired network question (A2) and cohort C (A3) matter more than a second look at A and B.

## A2. A second pre-registered question: does the network add anything?

**Question.** Does the network add anything beyond shrinking toward the multivariate chain
ladder? The main row `tlrn_13_fh` and the network-off row `tlrn_13_fh_factors` have the same
cohort, the same scored companies and the same ten draws, and both are blended with the
multivariate chain ladder. The only difference is the network.

**Statistic.** The paired company-level bootstrap of the draw-average difference in
Pool_APE, model minus the network-off row (negative favours the network): 10,000 resamples of
companies, seed 2008, the 2.5th and 97.5th percentiles, the same resample for both rows and
for every valuation (a company's dates kept together), as in the primary verdict. It is
computed at valuation 10 and as the mean over valuations 6, 7, 8 and 10. A third interval,
valuation 10 with the dominant company left out, is computed the same way.

**Rule.**

* **network adds value** only if the valuation-10 difference interval AND the
  mean-over-valuations difference interval are both wholly below zero, AND with the
  dominant company left out the valuation-10 interval is not wholly above zero.
* **network hurts** if either of the first two intervals is wholly above zero.
* **not distinguishable** otherwise.

"Wholly below zero" means the upper end is below zero and "wholly above" means the lower
end is above zero, as in the primary rule. Descriptively, and outside the rule, the raw
network (the forecast before the blend, `pred_net`) is also reported against the network-off row
with the same bootstrap. The verdict has the same labelling as the primary one (provisional
unless both rows hold exactly draws 0 to 9; the network-off row's draw set must equal the main
row's). It is reported per cohort next to the primary verdict. It is **not** part of the
primary "effective" rule and cannot change it.

**Why this question is expected to be sharper.** The two rows' errors are highly
correlated: they share the data, the head's pooled factors and the multivariate chain
ladder blend, so the company-to-company noise that makes the primary intervals wide
largely cancels in their difference. No interval width is claimed here, because no fit of
this study exists to measure one.

## A3. Run cohort C first

The run order on Colab is cohort C, then A, then B. If the primary verdict on cohort C is
"not distinguishable" or "worse", no further GPU time is needed to decide the main question:
A and B are confirmation runs on data the specification was tuned on, with a precision too
low to show a half-point gain (A1). Whether to run them anyway is the owner's (Ethan's)
decision.

**What cohort C is, stated without overclaiming.**

* It is independent in accident years (1988 to 1997 against 1998 to 2007).
* It is NOT independent in calendar time. Its outcomes sit in statement years 1998 to 2006,
  which overlap the data of A and B and the companion's valuations 6, 7 and 8.
* It is NOT independent in companies. 67 of its 83 companies are in A and 60 are in B, and
  155 of its 232 pairs are B's pairs.
* So it does not test a new economic environment or other companies.
* Whether the companion tuned on cohort C is **not tuned on as far as we know (assumed, not
  known)**. Wherever this study or its documents say C was not tuned on, they mean that,
  and the phrases "the only cohort the companion did not tune on" and "the one test it was
  not tuned against" are withdrawn as statements of fact.
* Every cohort, C included, is selected on future cells existing: the selection requires a
  complete 10-by-10 square, outcome cells included. That is a shared survivorship (a company
  that stops reporting after the cutoff is not in any cohort). It applies to the benchmarks
  as well as to the model, so it does not favour one over the other, but it tends to flatter the
  level of every method's error.

# Amendment 2 (2026-10-15, before any fit of this study has been run)

This amendment answers an outside review of the branch. The facts are the same as in
Amendment 1: **no fit of this study has ever been run on a GPU, and no real result exists.**
The only fits that exist are tiny CPU smoke fits (a few epochs, a few members) used to
check the code, and nothing from them may be quoted. Because nothing has been seen, the
rules below may still be changed, and this is the last point at which they can be. Every
change is made in this new section and the earlier text is left as written, except for
the one wording change described in B9. Where a sentence above says something this section
changes, read it as corrected here. The reviewer's claims were each checked against the
tables and the code before being written down; B0 gives the result, including where we
disagree.

## B0. The reviewer's claims, checked

| claim | result |
|---|---|
| The companion's own model-minus-multivariate-chain-ladder interval is 90%, from -3.14 to +2.43 points, for a 0.44-point gain (4.97% against 5.41%); ours is 95%, so wider. | **Confirmed.** The 90% interval is 5.57 points wide. Scaled to 95% under a normal approximation it is about 6.6 points. |
| The first study's measured 95% widths at valuation 10 were 9.3 to 11.0 points against `mack` and 5.4 to 7.8 against `mcl`. | **Confirmed** from the first study's log and `verdicts.json` (the table in A1): 9.28, 9.70, 10.43 and 10.95 against `mack`; 5.40, 5.88, 7.27 and 7.81 against `mcl`. These are for the accident-year model, at valuation 10 only. |
| Condition (ii), an unweighted mean over valuations 6, 7, 8 and 10, is thrown off by valuation 6, which flips the sign. | **Confirmed for the sign, qualified for the cause.** On the companion's rolling table the differences (transformer minus multivariate chain ladder) are +2.13, -0.91, -0.03 and -0.44 points, mean +0.19. Against the chain ladder they are +2.23, -0.94, +0.09 and -0.51, mean +0.22. Valuation 6 has 14 cells per pair, but valuation 8 has fewer (13), so the cell count alone does not explain it. Valuation 6 is the date with the shortest history, and it carries a quarter of the weight. |
| Condition (iv) passes even where the model is worse without the dominant company. | **Confirmed.** On the companion's own numbers, without company 1767, the transformer is at 13.07% against 12.81% (multivariate chain ladder) and 12.31% (chain ladder), worse by 0.26 and 0.76 points. With intervals several points wide, "not wholly above zero" holds for such a model almost surely. The interval itself cannot be computed from the public tables, so "passes" rests on the first study's widths. |
| The first study judged "worse" against the stronger benchmark only. | **Confirmed.** In `scripts/tlrn_study_eval.py`, function `verdict`, `strongest = min(...)` picks the benchmark with the lower Pool_APE and only its interval is tested. This study's rule already fired on either benchmark, so the difference is deliberate and is now stated (B5). |
| Selection reads later cells at valuations 6 to 8, on cohorts B and C. | **Confirmed**, and measured (B7). Cohort A (cutoff 5) is not affected in the same way. |
| About 37% of uniform bootstrap resamples leave the dominant company out. | **Confirmed**: the chance is (1 - 1/n)^n, which is 0.366 for n = 82, 83 or 84. |
| One reviewer disputed part of the reasoning for the stratified bootstrap. | **Noted.** It is therefore added as a sensitivity analysis only (B1). |

## B1. Precision, and a stratified bootstrap beside the primary one

**What the design can detect.** The primary verdict needs the 95% interval of the Pool_APE
difference to lie wholly below zero. For an interval of width W around the point
difference, that needs a gain of roughly W / 2 or more. With the measured widths:

* against `mcl`: W from 5.4 to 7.8 points, so a gain of about 2.7 to 3.9 points;
* against `mack`: W from 9.3 to 11.0 points, so a gain of about 4.6 to 5.5 points;
* `effective` needs both, so the binding figure is the one against `mack`: about
  **4.6 to 5.5 points**. Amendment 1's "about three points or more" is the low end of the
  distances actually seen (1.3 to 6.1 points); read it as three to five.

The companion's own headline gain over the multivariate chain ladder is 0.44 point (and
0.51 point over the chain ladder at valuation 10). That is about a tenth of what the design
can detect. As a rough guide (a normal approximation, my arithmetic and not a simulation),
if the true gain is half a point the chance that the interval ends up wholly below zero is
about 4% to 6%, close to the 2.5% it would be if the true gain were zero. So **"not
distinguishable" on A and B is the expected result, whether or not the model is slightly
better**, and it is not evidence that the model fails or that it equals the benchmarks.
The widths come from another model at valuation 10, so the factor-head widths are unknown
until it is run. The dollar-pooled statistic of B3 is dominated by valuation 10 (9,540 of
the 18,444 scored cells on cohort B, 52%, and most of the dollars), so it should be about
as wide, not much narrower. The evaluator prints, for every cohort, the width of each
interval and the gain it implies, so the achieved precision is stated beside every verdict.

**Secondary sensitivity analysis (reported next to the primary one, never replacing it).**
A uniform resample of companies leaves the dominant company out of about 37% of the 10,000
resamples. The stratified bootstrap keeps the dominant company in every resample exactly
once and resamples the other companies with replacement (n - 1 draws from the n - 1
others, random seed 2009, 10,000 resamples), using the same companies for the model and the
benchmark and for all four valuation dates. It is computed for the main row against `mack`
and `mcl` at valuation 10 and dollar-pooled, and for the paired network question. Its
intervals go through the same rule (i), (ii), (iii) as a "sensitivity verdict" printed
beside the primary verdict, with (iv) unchanged. **Caveat:** it conditions on the dominant
company being present. It answers "given that this company is in the book, how uncertain is
the rest?" and not "how uncertain is the whole result?". The uniform bootstrap is the honest
answer to the second question, because the result truly does depend on one company, so a
disagreement between the two is information and not an error.

## B2. Testing improvements: each change to the companion's protocol, and whether this study adopts it

| item in the companion's protocol | this study |
|---|---|
| All 20 members averaged, no member selection | **Adopted** (no selection). The repository's table of seed selection is a leftover and is not used. |
| One trained model | **Replaced** by 10 independent draws per row and cohort, judged on the draw average, with the spread of draws reported. |
| 90% intervals | **Replaced** by 95% intervals, which are wider (about 1.19 times, on a normal approximation). |
| Unweighted mean of Pool_APE over valuations 6, 7, 8 and 10 | **Replaced** in the rule by a dollar-pooled statistic (B3). The unweighted mean stays as a reported descriptive. |
| Dominant company (1767, 67% of the reserve) | **Adopted** as reporting: every key figure with and without it, and in condition (iv) as changed in B4. |
| Cohort B only (his 84 companies) | **Extended** to cohort A (notebook 04's) and a fresh cohort C (accident years 1988 to 1997). C is the headline cohort for the out-of-sample statement (B5). |
| Rolling dates kept together in the bootstrap | **Adopted**: a company's four dates always move together. |
| MAPE and wRMSE as headline metrics | **Descriptive only**: they are driven by very small reserves and by one large company. |
| Benchmarks from his own code | **Replaced** by ibnr's `mack` and `mcl` entries, fitted on the cells known at each date, with a tie-out against his table when it is supplied. |
| Comparison with the raw network and with the network switched off | **Adopted**, as paired bootstraps (B6), reported and not part of the primary verdict. |
| Selection rule at the last known diagonal for every valuation date | **Kept**, because the aim is to reproduce his design. Its look-ahead is named (B7). |
| Hyperparameters chosen with the test data visible | **Cannot be changed.** It is why A and B are labelled confirmatory. Cohort C is not tuned on as far as we know (assumed, not known). |
| Multiplicity | **No correction** (three cohorts). |

## B3. Condition (ii) is replaced by a dollar-pooled statistic

The unweighted mean over valuations is replaced, in condition (ii) and in the "worse"
rule, by the **dollar-pooled Pool_APE over valuations 6, 7, 8 and 10**, defined here before
any result:

> the sum, over the four valuations and all scored companies, of the absolute company
> error (|predicted - actual| of the company's total over the cells scored at that
> valuation), divided by the sum of the company actuals over the same company-valuation
> pairs.

The cells scored at different valuations overlap (a cell on diagonal 9 at development lag 3
is scored at valuations 6, 7 and 8), and each valuation counts its own cells. The statistic
is dollars of error over dollars of reserve, so a date with a large reserve counts in
proportion to it, and valuation 10, the whole realised reserve, dominates. The difference
(model minus benchmark) is bootstrapped with the same resample for all four dates.
**Condition (ii)** is now: the dollar-pooled difference has a 95% interval wholly below
zero against both `mack` and `mcl`. The unweighted mean over the four valuations, with its
interval, is still printed as a descriptive and takes no part in any rule. Because
valuation 10 dominates, (ii) is close to a repeat of (i) and is not an independent second
test. It is kept because it also reads the earlier dates.

## B4. Condition (iv) is replaced

Condition (iv) as registered ("without the dominant company the valuation-10 interval is
not wholly above zero") is true for a model that is worse than both benchmarks. It is
replaced by: **without the dominant company, either the valuation-10 point difference
(model minus benchmark) is at or below zero against both `mack` and `mcl`, or condition (i)
holds without it** (the valuation-10 interval is wholly below zero against both).

## B5. The rule as it now stands, the "worse" rule, and how mixed verdicts read

`effective` needs (i), (ii) as changed in B3, (iii) unchanged, and (iv) as changed in B4,
all four. **`worse`** if the interval in (i) or (ii) is wholly above zero against **either**
benchmark. This differs from the first study, which judged "worse" against the stronger
benchmark only. The difference is deliberate: the model has to beat both benchmarks to be
effective, so being clearly worse than either is a finding in its own right, and judging
against the stronger benchmark alone would let the model be clearly worse than the weaker
one and still read as "not distinguishable". **`not distinguishable`** otherwise.

**Mixed verdicts across cohorts.** Cohort C is the headline cohort for the out-of-sample
statement. A and B are confirmation of a fixed specification (it was tuned on them). The
overall statement follows mechanically from the cohort verdicts, and the evaluator prints
it:

| cohort C | A and B | overall statement |
|---|---|---|
| effective | both effective | "effective on the fresh cohort, confirmed on A and B" |
| effective | neither worse, not both effective | "effective on the fresh cohort; on A and B not worse (their verdicts are listed)" |
| effective | at least one worse | "mixed: effective on the fresh cohort, worse on A or B; no overall claim of effectiveness" |
| not distinguishable | any | "not distinguishable", whatever A and B say (they are listed for information) |
| worse | any | "worse on the fresh cohort", whatever A and B say |
| not run, or no finished fits | any | no overall statement is made |

A cohort verdict that is provisional (not exactly draws 0 to 9) makes the statement
provisional. An A or B that was not run (the owner may decide not to) is listed as not run
and is not counted as worse.

## B6. Comparison rows, the raw network and the first study's rows

* The paired question of Amendment 1 (main row minus the network-off row) keeps its rule.
  The dollar-pooled interval is reported beside it and is not in its rule.
* The same paired bootstrap, **main row minus the raw network** (`pred_net` from the same
  fits), is added with the same interval rules (the words are `blend adds value`, `blend
  hurts` and `not distinguishable`). It is reported and is not part of the primary verdict.
* The spread of the blend weight alpha (minimum, quartiles, median, maximum and standard
  deviation over all members of all draws) is reported for every valuation, from
  `_members.csv`, beside the median the companion publishes.
* The first study's accident-year rows are no longer "compared when their fits exist". The
  fits exist (the first study's `analysis/results/tlrn_study/fits`, on the branch of the
  pull request that holds its results). The evaluator takes their folder as
  `--first-study-fits`, compares both rows on cohorts A and B, and says in its output and in
  `verdicts.json` when a run was made without them, instead of leaving them out silently.
  The first study ran A and B only, so on C they do not apply.

## B7. Selection look-ahead

Step 2 of the companion's selection rule (positive paid and a loss ratio below 3 on every
cell up to the cutoff) is run at the last known diagonal (10) on cohorts B and C, so at
valuations 6, 7 and 8 it reads cells that those valuations score. **Cohorts B and C are
affected.** Cohort A (cutoff 5) reads diagonals 1 to 5 only, all before valuation 6, so it
is not affected in the same way. The effect is one-sided and small. Selecting again at
cutoff 6, 7 or 8 gives a larger set that contains the cutoff-10 set (checked: nothing is
removed) and adds 18, 14 and 7 pairs on cohort B and 5, 4 and 0 pairs on cohort C. An
optional sensitivity ("select again at the valuation's cutoff and score the existing fits")
is **not done**: the fits were trained on the cutoff-10 companies, so the added pairs have
no forecast to score, and re-selection never removes a company. Doing it properly needs new
fits.

## B8. Other statements corrected or added

* Member random seeds repeat across valuation dates within one fit: a member at valuation
  `v` is seeded `seed + 10000 v + 1000 m`, and the final fit at `seed + 1000 m`. Members 10
  to 19 at valuation 6 share seeds with members 0 to 9 at valuation 7, and members 10 to 19
  at valuation 7 with members 0 to 9 at valuation 8. The sentence "no two draws share a
  random stream" is true across draws only. The training data differ at each valuation, so
  the members themselves are not copies.
* The checkpoint improvement threshold is 1e-6 in ibnr's JAX backend against 1e-12 in the
  companion's code. This is recorded as a difference, thought harmless, and not tested here.
* The evaluator refuses any non-smoke fit in which a member did not run 3,000 epochs, and
  prints the medians of alpha and best epoch beside the companion's published member table
  when its path is given (report only).
* If the check of a cohort's selection fails (the counts or the hash differ from the
  published selection), `select_cohort` raises an `AssertionError` and **stops the whole
  call**. It does not drop cohort C and give a reason. The study is three cohorts times two
  rows times ten draws, which is 60 fits.

## B9. Wording

Ethan has banned one word family. Amendment 1 above, the evaluator's `AMENDMENT_1`
constant, the tables the evaluator writes and its tests used one of those words for the row
with the network switched off. They now say "network-off row" or "comparison row", and the
file that carried the word in its name is now `<cohort>_comparison_rows.csv`. No rule changed. The
evaluator's original module docstring is pinned by its sha256 and was not edited, so it
still contains one use of the banned word; the test that looks for the word skips that one
text.


[^request]: Ethan's instruction in the 2026-10-15 working session
[^first]: The first study, whose machinery and style this one extends
