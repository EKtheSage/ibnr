---
type: Decision
title: The fifth tlrn study, descriptive rows and tables around the fourth study
description: "Pre-registration, written before any code or fit exists (2026-11-24), of ibnr's fifth tlrn study. It answers no question with a verdict. It adds four single-draw rows (no attention along the development lag, no attention at all, batches from the union of the date company sets, and the full model with its trained parameters saved), read-outs from those fits, three benchmark rows (Munich chain ladder, a floored multivariate chain ladder, a blend against the floored forecast), and tables (by line, by company, without the largest company, an interval comparison, all reduced rows, the companion's published values beside ibnr's), around the fourth study's fits, which stand unchanged. It also fixes the checks that must pass before any fit, the budget and stops, and the points left open for the owner."
tags: [tlrn, evaluation, preregistration, colab, companion-study, decision]
status: draft
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-11-24T12:00:00Z }
stale_after: 2027-05-24T00:00:00Z
sources:
  - id: fourth
    resource: /decisions/tlrn-study4-preregistration.md
    title: The fourth study's pre-registration, whose fits, rows, questions and verdict rules stand as accepted on 2026-11-03
  - id: proposal
    resource: docs/tlrn-study5-proposal.md in the main checkout (untracked when this note was written)
    title: The background proposal for this study
  - id: companion
    resource: /references/companion-study-repository.md
    title: Where the companion study's files live
  - id: revision
    resource: the companion's repository, revision at commit 1654ee0 (2026-11-17), compared file by file with commit b30ef02
    title: The revision whose reporting this study follows
  - id: scope
    resource: /decisions/tlrn-entry-scope.md
    title: The attribution rule (the companion and the companion's repository are not named in this repository)
---

# Decision

The fourth study targets the companion's commit b30ef02 and is fixed.[^fourth] The companion's
repository has since moved to commit 1654ee0 (17 November 2026).[^revision] Its revision was
compared file by file with b30ef02. The model is unchanged: network, inputs, head, loss,
optimizer, schedule, validation, blend, refit and member count are the same, and the size is
still 18,153 parameters per member. The published valuation 10 figure moved from 3.37% to 3.65%
on the same starting seed, on the same 84 companies, against an unchanged 5.41% for the
multivariate chain ladder, because the study was rerun on a new framework with new random
streams. Its three seeds now give 3.65%, 3.56% and 3.41%.

What changed is reporting:

* the 1988 to 1997 cohort was removed;
* Munich chain ladder moved into the main table (5.62% at valuation 10, 4.54% mean over dates);
* a floored multivariate chain ladder row was added;
* the blend against the floored forecast (3.60% at valuation 10, 4.09% mean over dates) is now
  reported at every date;
* a table without the largest company was added;
* the multivariate chain ladder with the same residual recipe was added to the interval
  comparison.

Other published values at valuation 10: network alone 2.99%; network off 13.55%; pooled
incremental loss ratios 7.17%; pooled chain ladder 9.76%; mean over dates 4.11% against 4.93%
(the best classical method). Interval coverage of the realized outcome at 80% and 95%: 75.0% and
90.5% for the model, 71.4% and 91.7% for the multivariate chain ladder with the same recipe,
60.7% and 73.8% for the ODP bootstrap. Mean CRPS: 9,505, 12,647 and 11,430 in the same order.

Reduced rows published at 1654ee0, mean Pool_APE over dates, blend first and network alone
second:

| row | blend | network alone |
|---|---|---|
| full model | 4.11% | 4.16% |
| no cross-line attention | 4.16% | 3.70% |
| no development attention | 4.65% | 4.87% |
| no accident-year attention | 4.08% | 3.88% |
| no attention | 4.48% | 5.41% |
| paid only | 4.90% | 5.64% |

These are the companion's figures, read from the companion's files. They are not ibnr results and are used
here only as the values ibnr's tables are set beside.

**This study answers no question with a verdict.** It adds descriptive rows and tables. Every
number it prints is a description of one set of fits. If a rule below turns out to be wrong,
change it in a dated amendment at the end of this note, written before the fit it would
affect.

# Relation to the fourth study

The fourth study stands exactly as accepted on 2026-11-03: cohorts B4 and C4, ten draws, its
rows, its two questions and its written verdict rules, pinned to ibnr 0.7.7. It is run first, as
released, from notebook 06. The fifth study reads the fourth study's fits from their own folder
and adds rows and tables around them. No fourth-study row is refitted, and the fourth study's
verdicts are computed only by its own evaluator, `scripts/tlrn_study4_eval.py`.

# New fits (ibnr 0.7.8)

Four single-draw rows, cohort B only, draw 0 only, valuation dates 6, 7, 8 and 10, twenty
members each, with the same seeds as the main row's draw 0 (fit seed 11; date `v` adds
`20000 v`; member `m` adds `1000 m`):

* **`tlrn_cr_nolag`.** Attention along line and accident year, none along the development lag.
* **`tlrn_cr_noattn`.** No attention in any direction.
* **`tlrn_cr_union`.** Batches drawn from the union of the four dates' company sets, as the
  companion does, instead of each date's own set.
* **`tlrn_cr_saved`.** The main row's own configuration, fitted again with every member's
  trained parameters saved. The fourth study's fits do not store parameters, so this row is the
  only source of the full model's read-outs below. It is not a fourth-study fit, it enters no
  verdict, and it is never substituted for the fourth study's draw 0. Its forecasts are printed
  beside the fourth study's draw 0 as a description of how far a refit under 0.7.8 on a GPU
  moves from the fit under 0.7.7.

All four rows save their members' trained parameters. Each is one draw. Each is reported beside the main row's ten-draw minimum and maximum, and is
described only as inside or outside that range. No statement about cause is made from one draw.
With a different network or a different batch pool, the same seeds give the same starting
numbers only where the parts exist in both networks; the rows are not paired with the main row
in any stricter sense.

# Read-outs from saved fits

Two read-outs are wanted: (1) every trained member read out with the network set to zero and
not blended (the companion's control), and (2) the attention weights and the learned baseline
for one example company, for the article's figures. Neither can be computed from the fourth
study's saved fits, which hold no trained parameters (see "Open for the owner to decide"). Both
are computed for the four new rows, which save their members' parameters under 0.7.8. The full
model's read-outs come from `tlrn_cr_saved`, one draw on cohort B. They carry no verdict.

# Benchmark rows added

* **Munich chain ladder** (Quarg and Mack, paid with incurred), per company, on each date's own
  company set.
* **The multivariate chain ladder floored at zero.**
* **The blend against the floored forecast**, with the same blend weights and the same member
  forecasts as the main row.

The chain ladder (`mack`) and the multivariate chain ladder (`mcl`) stay as the fourth study
defines them: `mack` is ibnr's `mack` entry fitted per company and line on the cells known at
the date (`sigma_rule="mack"`), its point forecast over the scored cells added over lines, and
`mcl` is the multivariate chain ladder entry fitted per company on the lines of the date's set.

# Tables added

1. Metrics by line.
2. Per-company errors.
3. Every headline quantity without the largest company, defined in advance as the company with
   the largest actual reserve at valuation 10 on each cohort.
4. **The interval comparison.** For the model and for the multivariate chain ladder, the
   companion's recipe: errors from valuation dates 6, 7 and 8, each scaled by the larger of the
   absolute forecast, 1% of premium and 1; grouped by premium quartile; centred on each group's
   median; 4,000 draws per company; interval equals point plus scale times error; 80% is the
   10th to 90th percentile and 95% is the 2.5th to 97.5th. For the third method, an ODP
   bootstrap with 2,000 draws per triangle, line draws summed within a company. The three are
   scored by coverage of the realized outcome at 80% and 95%, CRPS and interval score,
   computed per draw and reported as the ten-draw mean with minimum and maximum.
5. All five reduced rows in one table (the fourth study's no cross-line attention, no accident-year
   attention and paid-only rows, and the two new rows that remove attention).
6. The companion's 1654ee0 values beside ibnr's.

The recipe is implemented by `kernels/residual_calibration.py` with the differences listed
under "Open for the owner to decide". The evaluator fixes how each is handled before any fit.

# Checks that must pass before any fit

1. **The main row under 0.7.8 equals the main row under 0.7.7** on a small fixture, because
   allowing new attention sets and the batch pool setting touch the network and the training
   loop.
2. **Munich chain ladder ties out per company** to a reference produced by one run of the
   companion's classical stage, stored under a neutral name. The tolerance is a proposal:
   relative 1e-6 on company reserves. The owner may change it before acceptance.
3. **A run-time perturbation check.** Replacing every cell after the origin with noise leaves
   the inputs, the masks, the starting amounts, the benchmark forecast and a random-weight
   network's output unchanged, on 50 example sets.
4. **Parameter counts** for the two new architecture rows are printed and recorded before
   fitting. The main row stays at 18,153.

# Budget and stops

Four single fits at roughly 9 minutes each, from the third study's measured 521 seconds per
main fit. Stop and ask if the first fit takes more than 25 minutes or if the saved fits pass
2 GPU hours.

# What is not done

* No reproduction of the companion's three seeds or random streams.
* No verdicts.
* No torch backend on a GPU.
* No change to the fourth study.

# Where things live

* Runner entry: `scripts/tlrn_study_colab.py`.
* Evaluator: `scripts/tlrn_study5_eval.py`.
* Notebook: `analysis/07_tlrn_study5_on_colab.ipynb`.
* Drive folder: `ibnr_colab_cf/study5`.
* Results: `analysis/results/tlrn_study5/`.
* Munich chain ladder and the interval score live in `kernels/`, with no new top-level export.
  Two config options are added, attention sets without the development lag and the batch pool,
  because the new rows cannot run without them.

# Open for the owner to decide

1. **Acceptance of this note**, in a separate commit before any code, as with the fourth study.
   The status stays draft until the owner accepts it.
2. **The read-outs from saved fits (conflict with the stated plan).** The fourth study's saved
   `.pkl` holds a `ValuationFits` container (`scripts/tlrn_study_colab.py`, class at line 863 and
   `fit_valuations` at line 1076): the per-date forecast tables, the members table, the last
   date's reserves, the company pairs, the seeds and the config. The members table records each
   member's kept flag, best epoch, epochs run, blend weight and refit epochs, and the history
   is reduced to a count of non-finite loss epochs. **No trained parameters and no validation
   histories are stored.** So neither read-out can be computed from the fourth study's fits
   without a new fit. This note registers one extra single-draw row, `tlrn_cr_saved` (the full
   model on cohort B at draw 0, about 9 minutes), and computes the read-outs for the four new
   rows. The alternatives are: refit the twenty fourth-study main fits under 0.7.8 with
   parameter saving (about 2.8 GPU hours by the third study's times, which passes the 2 hour
   stop, and the refit is not guaranteed to reproduce the fourth study's forecasts bit for bit
   on a GPU); add the same row on cohort C; or drop the full model's read-outs. Default: the
   one extra row on cohort B.
3. **The residual recipe against `kernels/residual_calibration.py`.** Differences found:
   * the kernel returns draws and the recipe states percentiles, so the interval is the
     quantile of 4,000 draws (numpy's default linear method) and not a direct quantile of the
     centred errors, which adds sampling error of the draws; the kernel's own coverage table
     uses numpy's `median_unbiased` quantile (R's type 8), a different convention;
   * the kernel refuses a size group with fewer than 40 errors (`min_per_stratum`), and the
     recipe has no such rule;
   * the quartiles are an `ntile` over the units that have at least one finite error at the
     calibration dates, and a company at the final date is placed by the stored upper size
     bounds, not by a quartile of its own date's set, so a company can sit in a different group
     than a quartile of its date would give;
   * errors that are not finite are dropped by the kernel and counted;
   * the horizons are named by the caller: dates 6, 7 and 8 against ten accident years are
     horizons 4, 3 and 2;
   * each company's draws are independent of the other companies' draws, as in the kernel.

   Default: use the kernel as written, with 4,000 draws, the percentiles taken from the draws,
   and print each difference in the evaluator's output.
4. **The union batch pool.** The union pool trains one network over companies that a given date
   does not score. How the date's own mask is applied to the union batches is decided in code
   and recorded in the evaluator's output before the first fit. The fixture check above does
   not cover the new setting, only that the default is unchanged.
5. **The tolerance for Munich chain ladder** (relative 1e-6), proposed above.
6. **The number of draws of the interval comparison per fit** (4,000 and 2,000), which are the
   companion's and so not tuned.

[^fourth]: The fourth study's pre-registration
[^revision]: The companion's revision at commit 1654ee0, compared with commit b30ef02
