---
type: Decision
title: The third tlrn study, on the companion's refit model
description: "Pre-registration, written before any code or fit exists (2026-10-21), of ibnr's third tlrn study: the companion's newest model (premium loss-ratio head, attention across accident years, company batches, networks retrained on everything known up to the cutoff, alpha fitted on one diagonal and the epoch chosen on the next, companies picked separately at each valuation date). Two co-primary questions: does the refit help (paired, with and without refit from the same training run), and is the model effective against the chain ladder and the multivariate chain ladder. Also fixes the cohorts and the per-date selection rule, ten draws, the benchmarks, the statistics, every decision rule as a number, the known deviations from the companion, the budget and stopping rules, the questions for the companion's author and the choices still open."
tags: [tlrn, evaluation, preregistration, colab, companion-study, decision]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-21T12:00:00Z }
stale_after: 2027-04-21T00:00:00Z
sources:
  - id: request
    resource: Ethan's instruction in the 2026-10-21 working session
    title: "write the pre-registration for ibnr's third tlrn study before any code or fit exists, modelled on the second study's"
  - id: second
    resource: /decisions/tlrn-factor-head-study.md
    title: The second study's pre-registration, whose rules this one reuses where it says so
  - id: second-results
    resource: /findings/tlrn-factor-head-study-results.md
    title: The second study's results, which this study's design has seen
  - id: architecture
    resource: /findings/companion-final-architecture.md
    title: The companion's previous final model (commit f74addd)
  - id: reproduction
    resource: /findings/companion-notebook-reproduction.md
    title: The earlier reproduction of the accident-year notebook
  - id: companion
    resource: /references/companion-study-repository.md
    title: Where the companion study's files live
  - id: audit
    resource: the companion's repository at commit f50871d (2026-10-21), files Code/01_functions.py, Code/02_data.py and Code/06_train.py, read through the GitHub API without checking anything out
    title: The audit this note rests on (results quoted from the companion were produced by code at commit 9e118a7)
  - id: runner
    resource: scripts/tlrn_study_colab.py
    title: The Colab runner, which has select_pairs(cutoff=) but fits one cohort per call
  - id: scope
    resource: /decisions/tlrn-entry-scope.md
    title: The attribution rule (the companion and the companion's repository are not named in this repository)
---

# Decision

The second study tested the companion's model as of commit f74addd (a factor head, attention
across lines and lags) and found it not distinguishable from the chain ladder and the
multivariate chain ladder on cohorts A and B and worse on the fresh cohort C.[^second-results]
The companion's repository has since moved (commit f50871d, audited 2026-10-21) to a different
model. Ethan's call (2026-10-21) is a third study on that model, fixed here before any code
or fit exists.[^request]

**Two co-primary questions**, each judged by a written rule on each cohort:

* **P1, the refit.** Does retraining each network on everything known up to the cutoff
  (the companion's refit) lower the company-reserve error compared with using the network as
  it stood at its chosen epoch? This is the new question. It is answered from a paired
  comparison: the two arms come from the same training run.
* **P2, effectiveness.** Is the model with refit more accurate than the chain ladder
  (`mack`) and the multivariate chain ladder (`mcl`)? This is the second study's question,
  asked of the new model with the second study's verdict rule unchanged.

Secondary questions, reported and never part of P1 or P2: does the network add anything
beyond the head's pooled pattern and the blend; does the blend with `mcl` help or hurt the
network alone; and does ibnr replicate the figures the companion reports.

The study is fixed here. If a rule below turns out to be wrong, change it in a dated
amendment at the end of this note, written before the fit it would affect, not silently.

# Why this is a new model and not a rerun of the second study

Five things changed in the companion's pipeline, and ibnr has none of the last three.

| | second study (f74addd) | this study (f50871d) |
|---|---|---|
| head | `log f = softplus(phi + 0.5 net)`, a log development factor per line and step | `inc = exp(min(beta + 0.5 net, 5))`, an incremental loss ratio per line and lag, times premium; `beta` learned, reset at each valuation to the pooled incremental loss ratio (ibnr: `head="premium_lr"`) |
| attention | lines and lags | lines, lags and accident years (ibnr: `attention=("line","lag","ay")`, `accident_year_variant`) |
| batches | 64 examples | 8 whole companies (ibnr: `batch_unit="company"`) |
| networks used | the network at its chosen epoch | a **refit**: each network retrained on all diagonals known at the cutoff and stopped at the epoch chosen earlier (ibnr has no refit today) |
| alpha and epoch | both from diagonals C-1 and C | alpha fitted on diagonal C-1 only, the epoch and checkpoint chosen on diagonal C (ibnr today uses both diagonals for both) |
| companies | one set, picked at the last known diagonal | picked separately at each valuation date (ibnr's runner fits one cohort per call) |

The blend `mcl + alpha * (network - mcl)` is unchanged, and alpha is not refitted after the
refit. So the second study's results say nothing about this model: its head, its inputs'
reach and its final networks are different, and its company sets differ at every date but
the last.

# What is verified about the new model

**Verified** means read in the companion's source at f50871d when this note was written.
**Reported** means the companion's own claim about its results, relayed to ibnr and not
checked. **Assumed** means carried from f74addd or inferred and still to be checked before
code is written (the checklist is under "Before any fit").

* **Verified.** The head: `inc = exp(min(beta + EPS * net, 5))`, cumulative amounts built
  lag by lag from the latest observed cumulative as `previous + premium * inc`, with
  `EPS = 0.5`. A lag no training target supervised uses a pooled loss ratio fallback
  (`lr_fallback`).
* **Verified.** The company rule (`select_cohort` in `Code/02_data.py`, lines 76 to 100): a
  complete 10-by-10 square with positive premium on every cell, including cells after the
  cutoff; then, on the cells known at diagonal `cutoff`, positive paid and premium and a
  cumulative paid loss ratio below 300%; then at least two such lines per company. The
  cutoff is a parameter, and `COHORT_BY_C` builds one company set per backtest valuation.
* **Verified.** Training: 3,000 epochs, 20 warmup epochs, learning rate 0.003 for the
  network and 0.03 for `beta`, gradient clipping 1.0, batches of 8 whole companies (all their
  accident years together), validation every 5 epochs. The alpha is fitted on the first
  validation diagonal (C-1) and the blended error on the second (C) picks the checkpoint.
* **Verified.** The refit: every member is retrained from the same initial key and schedule on
  targets up to diagonal C, for as many epochs as the longest selected epoch among all
  members, and the weights at exactly the member's own selected epoch are kept. Forecasts use
  the lags the refit trained. Alpha is not refitted.
* **Verified.** The primary seed is 2025; seeds 2026 and 2027 repeat the whole protocol as a
  robustness check, and the reported run (`FITS`) is the primary seed's 20 members per
  valuation date. How the three seeds enter any published figure is a question (below).
* **Reported, unverified by ibnr.** Companies at valuations 6, 7, 8 and 10: 90, 89, 87 and
  84. Valuation 10 Pool_APE of the blend: 3.93% with the refit and 4.88% without. The
  unweighted mean over the four valuations: 4.36% with the refit and 4.02% without. The
  network alone at valuation 10: 2.96%. Median alpha about 0.45. Parameters per network:
  18,537.
* **Assumed.** The loss weights, the cosine schedule, dropout 0.3, d_model 32, 2 heads, 13
  features, 20 members averaged and the paired bootstrap are those of f74addd, which the
  second study took from the companion's older notebook; they are not re-read here except the
  constants listed as verified.
* **Inferred, from ibnr's own building blocks.** ibnr's network for the same configuration will
  have about 18,601 parameters, 64 more than the companion's 18,537, for the reason the
  second study found (1-based line and lag embeddings, one extra row each). To be confirmed by
  building the network, which needs no fit. (Amended 2026-10-22: the preset now builds
  18,537, see the amendments below; 18,601 is the accident-year variant's count.)
* **Inferred, from the second study's note.** Selecting again at cutoffs 6, 7 and 8 on cohort
  B adds 18, 14 and 7 pairs to the 212 of cutoff 10, and on cohort C adds 5, 4 and 0 to the
  232; the larger sets contain the smaller (that note, B7, measured it). So the expected pair
  counts are 230, 226, 219 and 212 on B and 237, 236, 232 and 232 on C. The companion's company counts
  above (90, 89, 87, 84) are consistent with that on B and are checked by a selection run
  that fits nothing.

# How ibnr runs it

Planned for ibnr 0.7.6. **None of the code below exists when this note is written.** The
names are decided; the behaviour is what the rules below assume.

| companion choice | ibnr setting |
|---|---|
| incremental loss ratio head, premium times `exp(min(beta + 0.5 net, 5))` | `head="premium_lr"`, `eps=0.5` |
| attention over lines, lags and accident years, one block, d_model 32, 2 heads, dropout 0.3 | `attention=("line","lag","ay")` with the accident-year variant of the network, `n_layers=1` |
| batches of 8 whole companies | `batch_unit="company"`, `batch_size=8` |
| refit on everything known, stop at the selected epoch, alpha not refitted | new `TLRNConfig.refit: bool = False` |
| alpha fitted on diagonal C-1, epoch chosen on diagonal C | new `TLRNConfig.alpha_on` (default: today's behaviour; new value `"first_diagonal"`) |
| all of the above together | classmethod `TLRNConfig.companion_final_variant()` bundling `accident_year_variant`, `premium_lr`, `refit` and `alpha_on="first_diagonal"` |
| 13 features | `incurred_field="incurred_loss"`, `case_field="case_reserve"` as in the second study |
| pooled fallback for untrained lags | `tail_policy` set so that the premium_lr head falls back to the pooled incremental loss ratio (matched to `lr_fallback`: see the amendment of 2026-10-21) |
| 3,000 epochs, 20 members averaged, no member selected | `max_epochs=3000`, `ensemble_size=20`, `keep=None`, `member="mcl_blend"`, early stopping unable to fire (`patience=600`, `check_every=5`) |
| companies picked at each valuation date | a study-runner wrapper that fits each date on its own `select_pairs(cutoff=v)` cohort; the existing `select_pairs` already takes `cutoff` |
| the runner | a new config name in `scripts/tlrn_colab.py` and a new `Study` entry in `scripts/tlrn_study_colab.py`; the evaluator is a new `scripts/tlrn_study3_eval.py` that reuses the second study's statistics |

# What is run (the arms)

All arms are judged on the same companies at each date.

* **`tlrn_cf_refit`, the main arm.** The new model with the refit: the blended forecast
  `mcl + alpha * (network_refit - mcl)`, 20 members averaged. This is the companion's
  published headline construction and is the arm P2 judges.
* **`tlrn_cf_norefit`, the comparison arm for P1.** The same training run, scored with each
  member's network at its chosen epoch (before the refit) and the same alpha. The two arms are
  therefore paired by construction: same data, same initial weights, same epochs, same alpha;
  only the final networks differ. This needs one implementation that keeps both forecasts
  from one run (as the raw network is kept from the main fit today). If it cannot be built,
  the fallback is two separate fits with `refit=True` and `refit=False` and the same draw
  seed; the P1 rule is unchanged, but the pairing is then weaker and the cost higher (open
  decision 3).
* **`tlrn_cf_factors`, the network-off arm.** The same configuration with `factors_only=True`:
  the head's pooled pattern (`exp(beta)` times premium) blended with `mcl` by its own fitted
  alpha. About 10 seconds a fit in the second study. It answers the secondary question on
  the network. It is run only if the code check in "Before any fit" shows that
  `factors_only` is defined for this head.
* **The raw network.** The refit networks before the blend, read from the main fits at no
  extra cost.

**Not in the registered arms, and why.**

* **A control with `head="ldf"`** (the second study's head, with attention across accident
  years, the refit, the alpha split and the per-date companies). It would separate the effect
  of the head from the other changes. It costs one main fit per draw and cohort, roughly 1
  to 1.3 GPU hours on a Blackwell-class card for ten draws on two cohorts (unmeasured, from
  the same estimate as the budget below). The questions here are about the companion's final
  model as a whole and about the refit, not about why it differs from the previous one, so it
  is left out of the rules and offered as an exploratory addendum (open decision 4).
* **A comparison on the alpha split** (alpha on both diagonals as ibnr does today against
  alpha on diagonal C-1 only). It changes checkpoint selection, so it needs a separate
  training run for every draw, the same cost as the control above, and it bears on how
  ibnr's existing behaviour compares with the companion's, which is a replication
  question and not one of the two registered ones. Left out of the rules and offered as an
  exploratory addendum (open decision 4).
* **Cohort A of the second study** (selected at cutoff 5). The companion's selection now
  moves with the valuation date, so a fixed-cutoff cohort does not describe the model.

# Cohorts, and the selection rule at each date

Two cohorts, both built from the same cas-schedule-p publish as the second study
(20260613_041006).

* **Cohort B3, the companion's.** Accident years 1998 to 2007, last known diagonal 2007-12-31,
  as the second study's cohort B.
* **Cohort C3, the fresh cohort.** Accident years 1988 to 1997, valuation 1997-12-31, as the
  second study's cohort C. It is independent in accident years only, as that study
  recorded: it overlaps B3 in calendar time and in companies (67 of its 83 companies are in the second
  study's cohort A and 60 in its B, and 155 of its 232 pairs are B's pairs).

**The rule at valuation `v` in {6, 7, 8, 10}.** The company set `S_v` is the output of
`select_pairs(cells, cutoff=v)` on the cohort's cells, which is the companion's
`select_cohort(cutoff=v)`: (1) a complete 10-by-10 square with a finite positive net earned
premium on every cell, post-cutoff cells included; (2) on the cells with calendar index at
most `v`, finite positive paid and premium and a cumulative paid loss ratio below 3; (3) at
least two such lines per company. Pairs are the unit, companies the unit of scoring. Valuation
10 uses the last known diagonal and gives exactly the second study's cohort (212 pairs and 84
companies on B3; 232 pairs and 83 companies on C3).

* **Training.** The networks of date `v` are trained, validated and refit on the companies of
  `S_v` only, with data known at `v`. (Assumed from the companion's code, to be checked; see
  "Before any fit".)
* **Scoring.** The forecasts of date `v` are scored on the companies of `S_v` and on the
  cells the second study scored: diagonals after `v`, development lag at most `v`, and at
  valuation 10 every cell past the last diagonal. That is 14, 15, 13 and 45 cells per pair.
* **Benchmarks.** `mack` and `mcl` are fitted and scored on exactly the same `S_v`.
* **Expected sizes.** Pairs 230, 226, 219 and 212 on B3 and 237, 236, 232 and 232 on C3 at
  valuations 6, 7, 8 and 10, from the second study's note; companies 90, 89, 87 and 84 on B3
  as the companion reports. If ibnr's rule returns other counts, the study uses ibnr's
  counts and the evaluator reports the difference. The companion's list is not used.
* **Nesting.** `S_10` is contained in `S_8`, in `S_7` and in `S_6` (the second study checked
  it). A company is therefore either in all later sets or enters at an earlier date.

# Draws and seeds

Ten independent fits per cohort. Draw `k` uses fit seed `11 + 100003 k`, the second study's
seeds, so a rerun is possible and no seed was chosen after any result. Inside a fit, each date
trains its 20 members with seeds `seed + 20000 v + 1000 m` (amended 2026-10-21 from `10000 v`, see below). The companion's seeds
(2025 to 2027) are not used: the random streams of JAX in ibnr and in the companion differ, so
equal seeds would not give equal numbers. The ten draws are compared with the companion's
reported run, which is one fit of 20 members at its primary seed. 2 cohorts x 10 draws = 20
main fits, each giving both the refit and the no-refit forecasts, plus 20 network-off fits.

# Benchmarks

`mack` (one fit per company-line pair) and `mcl` (one fit per company) from ibnr's gallery,
fitted on the data known at each valuation date and scored on the cells above. At valuation
10 they must reproduce the second study's saved values exactly: `mack` 5.481% and `mcl`
5.410% on B3, `mack` 11.947% and `mcl` 4.969% on C3, to rounding, and the tie-out to the
companion's published table of valuation 10 (the second study's tie-out). At valuations 6, 7
and 8 the sets differ from the second study's, so there is nothing stored to tie to; the
evaluator checks them against an independent recomputation (the second study's check of cohort
C is the model for it) before the verdicts are read.

# The statistics

All of the second study's definitions are kept. A company's reserve error at valuation `v` is
its predicted minus its actual total over the scored cells. **Pool_APE** at `v` is the sum of
the absolute errors over the scored companies of `S_v` divided by the sum of their actuals.
The **draw average** at `v` is the mean over the ten draws of each company's prediction. The
**dominant company** is the one with the largest actual reserve at valuation 10, found in
code, and the study stops if it is not in every `S_v`. The **dollar-pooled Pool_APE** over
valuations 6, 7, 8 and 10 is the sum, over the four valuations and all companies of that
date's set, of the absolute company error divided by the sum of the company actuals over the
same company-date pairs. The **unweighted mean** over the four valuations is reported and
takes part in no rule.

**The bootstrap.** 10,000 resamples of companies with replacement, taken from the union of the
four sets (the cutoff-6 set contains the others). A company drawn `n` times counts `n` times at
every date at which it is in the date's set and not at all at a date where it is not. The same
resample is used for the model and for the benchmark (or for the two arms of a pairing), for
every valuation, and a company's dates stay together. The interval is the 2.5th to 97.5th
percentile of the difference. Seed 2008 for the primary intervals. The **stratified
sensitivity bootstrap** keeps the dominant company in every resample exactly once and
resamples the other companies (seed 2009); it is reported beside the primary one for P1 and
P2 and replaces nothing. "Wholly below zero" means the upper end is below zero; "wholly
above" means the lower end is above zero.

# The rule for P1: does the refit help?

Let `D(v)` be the draw-average Pool_APE of `tlrn_cf_refit` minus that of `tlrn_cf_norefit`
(points; negative favours the refit), `D10 = D(10)`, and `Dpool` the same difference for the
dollar-pooled statistic. Per cohort:

* **refit helps** needs all four: (a) the bootstrap interval of `D10` is wholly below zero;
  (b) the interval of `Dpool` is wholly below zero; (c) in at least 8 of the 10 draws, each
  draw's own valuation-10 Pool_APE with the refit is below its Pool_APE without it; (d) with
  the dominant company left out, the point value of `D10` is at or below zero.
* **refit hurts** if the interval of `D10` or of `Dpool` is wholly above zero.
* **not distinguishable** otherwise.

**A descriptive label that changes nothing**, because the companion's reported pattern is
mixed: **date-dependent** if the point value of `D10` is below zero and the unweighted mean
over the four valuations of `D(v)` is above zero. On the figures relayed (-0.95 points at
valuation 10 and +0.34 points on the unweighted mean) the companion's own run would carry
this label.

**Across cohorts.** C3 is the headline for a claim beyond the companion's data; B3 is the
companion's own cohort, where the design was chosen. Overall statement, printed by the
evaluator:

| C3 | B3 | overall statement |
|---|---|---|
| helps | helps | "the refit helps on the fresh cohort and on the companion's" |
| helps | not helps | "the refit helps on the fresh cohort only; B3 is listed" |
| not distinguishable | any | "not distinguishable", with B3 listed for information |
| hurts | any | "the refit hurts on the fresh cohort" |

If the refit helps on B3 and is not distinguishable or hurts on C3, the statement is that the
benefit is not shown beyond the cohort the companion's design was chosen on. A cohort not
run, or with fewer than the registered draws, is labelled provisional and not counted as hurting.

# The rule for P2: is the model effective?

The second study's rule, with its amendments, unchanged, applied to `tlrn_cf_refit` against
`mack` and `mcl` on each cohort and the per-date company sets above. `effective` needs all
four: (i) at valuation 10 the Pool_APE difference (model minus benchmark) has a 95% interval
wholly below zero against both benchmarks; (ii) the same for the dollar-pooled difference over
valuations 6, 7, 8 and 10 against both; (iii) the median over the draws of the valuation-10
Pool_APE is below both benchmarks; (iv) without the dominant company, either the valuation-10
point difference is at or below zero against both, or condition (i) holds without it. `worse`
if the interval in (i) or (ii) is wholly above zero against **either** benchmark.
`not distinguishable` otherwise. The overall statement follows the second study's table, with
cohort B3 in the place of "A and B": C3 effective and B3 effective gives "effective on the
fresh cohort, confirmed on the companion's"; C3 effective and B3 not worse gives "effective on
the fresh cohort; on the companion's cohort not worse"; C3 effective and B3 worse gives
"mixed", with no claim of effectiveness; C3 not distinguishable gives "not distinguishable"
whatever B3 says; C3 worse gives "worse on the fresh cohort".

**What each cohort's verdict means.** B3 is the cohort the companion's design was chosen on,
and the refit and the new head were, as far as we know, adopted with B's results in view, so a
verdict on B3 is **confirmatory of a fixed specification, not out of sample**. A verdict on C3
is out of sample in accident years only. Beyond what the second study said of C, one more
caveat applies here: the second study's results on C were written down (and a findings note
prepared for the companion's author) before this model's rules were written, and whether the
companion saw them before adopting these changes is not known (a question below), so "not
tuned on as far as we know" is weaker for C3 than it was for C.

# Secondary questions, reported and never in P1 or P2

* **The network.** `tlrn_cf_refit` minus `tlrn_cf_factors`, with the second study's rule:
  *network adds value* only if the valuation-10 and the dollar-pooled intervals are both wholly
  below zero and, without the dominant company, the valuation-10 interval is not wholly
  above zero; *network hurts* if either of the first two is wholly above zero; otherwise not
  distinguishable.
* **The blend.** `tlrn_cf_refit` minus the raw refit network: *blend adds value* on the same
  condition as above; *blend hurts* if the valuation-10 or the dollar-pooled interval is wholly
  above zero; otherwise not distinguishable. On the relayed figures (3.93% blended against
  2.96% for the network alone) the companion's run would read "blend hurts" at valuation 10.
  The raw network is also judged by the P2 rule as a secondary row; the blended arm is the
  registered one because it is the companion's headline construction, and no choice between
  the two is made after the results.
* **Replication of the companion**, below.
* **Alpha.** Minimum, quartiles, median, maximum and standard deviation of the fitted blend
  weight over all members of all draws, per valuation, beside the companion's median of about
  0.45.
* **Also reported:** Pool_PE (bias); the median absolute percentage error; the mean absolute
  error; the mean absolute log of predicted over actual reserve (companies with positive
  reserves); the share of companies where the model has the smaller absolute percentage error;
  Pool_APE within quartiles of actual reserve; every key figure with and without the dominant
  company; the draw-to-draw spread (standard deviation and range of the ten valuation-10
  Pool_APE values); the width of every interval and the gain it could just detect (half the
  width). MAPE and the reserve-weighted root mean squared error are descriptive only, as in
  the second study.

# Replication of the companion's reported figures

Judged on cohort B3 at valuation 10 against the figures as relayed, which are the companion's
single reported fit (primary seed) and not verified by ibnr. Let `m` be ibnr's draw average
and `[lo, hi]` the range of the ten draws' own Pool_APE.

* **R1, the level.** `3.93` lies in `[lo, hi]` of `tlrn_cf_refit` and `|m - 3.93| <= 0.5`.
* **R2, the refit effect.** `D10 <= -0.5` (the companion's relayed difference is -0.95).
* **R3, the network alone** (secondary). `|m_net - 2.96| <= 0.5` for the raw refit network.
* **replicates** if R1 and R2 both hold. **fails to replicate** if neither holds and
  `|m - 3.93| > 1.0`. **partially replicates** otherwise (this includes R1 alone, R2 alone,
  and the case where neither holds but `m` is within 1.0 point of 3.93). R3 is reported beside
  the label and does not change it.
* The two numbers 0.5 and 1.0 are set before any result: 0.5 is a little more than the whole
  range of the ten draws on B in the second study (4.743% to 5.140%, 0.40 points wide), and
  1.0 is about the relayed gain from the refit. They test whether a single published run
  could be one more draw from this procedure, not whether ibnr and the companion agree to the
  digit.
* The benchmark side is a tie-out and not a replication test: `mack` and `mcl` at valuation 10
  must match the second study's saved values (above), which already match the companion's
  published table.
* If the relayed figures turn out to be wrong, the corrected figures replace them by an
  amendment written before the first fit.

# Known deviations from the companion

Each is disclosed in the evaluator's output and the results note.

1. **Random streams.** ibnr's JAX streams differ from the companion's, so the same
   procedure gives different numbers; replication is judged on distributions (R1 to R3),
   not on equal numbers.
2. **1-based embeddings. Removed 2026-10-22.** The default line and lag embeddings have one
   unused row each, 64 parameters more than the companion's count. The preset now sets
   `compact_embeddings=True` and builds the companion's 18,537 (built and counted, see the
   amendments below). Not a deviation of this study any more.
3. **Draw design.** Ten ibnr draws against one reported fit of the companion's, and no use of
   its three seeds.
4. **Selection reuse.** The refit trains on diagonals C-1 and C and stops at the epoch that
   the network before the refit chose on diagonal C, so the choice was made with the same
   cells in view that the refit then learns from. The companion's design has this too, and it
   is judged to favour the refit slightly (a judgement of the earlier audit, not measured by
   ibnr). The comparison in P1 inherits it, so a gain from the refit is an upper bound on
   what an independent epoch choice would give. This is disclosed and not corrected.
5. **Alpha not refitted.** Alpha is fitted for the networks before the refit and applied
   unchanged to the refit networks, as in the companion's code; the refit networks have seen
   the cells alpha was fitted on. A property of the design, shared by ibnr, and not tested.
6. **Per-date training in one program.** The companion trains all dates in one vmapped
   program with the training sets padded to the largest date; ibnr will fit each date on its
   own cohort. Assumed equivalent in expectation and not tested.
7. **The refit length.** The companion runs the refit for the longest selected epoch over
   all members and keeps each member's own epoch; ibnr copies that. Any difference in
   which epoch is kept is a defect, not a deviation, and is covered by a test of the refit
   against a hand-rolled one-member case.
8. **Checkpoint threshold.** 1e-6 in ibnr's JAX backend against 1e-12 in the companion's
   code, from the second study's list; thought harmless, not tested.
9. **Survivorship.** Step 1 of the rule requires post-cutoff cells to exist, so a company
   must have a complete square whatever the date. This applies to the benchmarks as well as
   the model, as in the second study, and tends to flatter every method's level of error. The
   second study's look-ahead (sets chosen at diagonal 10 for every date) is gone by design,
   because the sets now move with the date.
10. **Fallback for untrained lags.** The companion uses a pooled loss ratio; ibnr's
    `tail_policy` for this head is matched to it (see the amendment of 2026-10-21).

# What is NOT claimed

* Nothing about why a result comes out as it does. Explanations (the head, the attention,
  the refit, the per-date companies, tuning on the data) are untested unless an exploratory
  row is run.
* Nothing about other accident-year windows, other lines, other companies or a new economic
  environment. Cohort C3 shares most of its companies with B3 and its calendar years.
* Nothing from a verdict on B3 as out-of-sample evidence.
* Nothing about the companion's code beyond commit f50871d, or about the run at commit
  9e118a7 other than the figures relayed, which are unverified here.
* No claim that ten draws sample training randomness well, or that the intervals measure
  uncertainty about future years: the bootstrap resamples companies only. One company holds
  about two thirds of the reserve at valuation 10 (66.3% to 67.3% in the second study), so
  every dollar-based statistic depends on it; hence the without-company-1767 figures.
* No claim of precision. P1 is a paired comparison and its interval should be narrower than
  the second study's (which were 5 to 11 points wide at valuation 10), but no width is
  known until a fit exists. The evaluator prints the width of every interval and half of it
  as the smallest difference that could be called clear.
* No statement that the second study's result transfers, in either direction.
* No multiplicity correction. Two cohorts judged on two co-primary questions, plus the
  secondary ones; agreement across cohorts counts less than it reads because they overlap.

# Budget, stopping rules and compute

**Unmeasured estimate.** The second study measured 177 to 209 s for a main fit on a Blackwell
GPU (mean 190 s; that model had no accident-year attention); the first study measured about 250
s for the accident-year model on the same card (that study's note, not measured here for this
model). This model has attention across accident years, so the nearer figure is 250 s for
training alone. The refit adds a second training run of up to the longest selected epoch over
all members, taking 0.6 to 1.0 of the training time if the median selected epoch is near the
second study's (about 1,800 of 3,000), but the longest epoch over 20 members can approach
3,000, which would make it about 1.0. So 400 to 500 s per fit, times 20 fits, is 2.2 to 2.8 GPU hours,
plus 20 network-off fits of about 10 s. Per-date cohorts mean four programs with different
shapes where there was one, so compile time may rise; no figure for that exists. A
planning figure of **3 to 6 GPU hours on a Blackwell-class card**, and about 460 s of
base time on an A100 and 3,200 s on a T4 in the first study (about 2 times and 13 times
Blackwell) so that a T4 is not a reasonable target. These are estimates, not
measurements.

**Order.** Draw 0 of C3 first, as a pipeline check; then draws 1 to 9 of C3; then B3 draws 0
to 9; then the network-off fits; then any exploratory rows. No result of any draw may change a
rule: the only reasons to stop or change course are a defect, non-finite values or the budget.

**Stops.**

* Stop and ask if draw 0 on C3 takes longer than 25 minutes (three times the upper estimate),
  or if the cumulative GPU time passes 8 hours before all registered fits are done.
* A member with a non-finite loss stops its fit; the same fit is rerun once with the same
  seed; if it recurs the cohort is reported without that draw and the verdict is provisional.
* Every non-smoke fit must show that every member ran 3,000 epochs (the evaluator refuses
  otherwise), and that the refit ran to each member's own selected epoch.
* A cohort without exactly draws 0 to 9 of every registered arm has a provisional verdict,
  printed with the draw count and the missing numbers.

# Before any fit

Checks that fit nothing and must pass before the first fit, listed here so they are not
chosen after the fact:

1. Build the network for `companion_final_variant()` and record its parameter count (expected
   18,537 after the amendment of 2026-10-22; it was about 18,601 when this note was written).
2. Run `select_pairs` on both cohorts at cutoffs 6, 7, 8 and 10 and record the pair and
   company counts against the expected ones (a selection run, not a fit).
3. Confirm in the companion's source at f50871d that the training set of date `v` is the
   cohort selected at `v` (assumed above), that the extra seeds enter no headline figure, and
   what `lr_fallback` is, and match the fallback in ibnr by test (done: see the amendment of 2026-10-21).
4. Test the refit against a hand-rolled single member (the weights kept are those at the
   member's own epoch) and test that the no-refit forecast equals what `refit=False`
   produces.
5. Check that `factors_only` is defined for the `premium_lr` head, or drop the network-off
   arm by an amendment.
6. Tiny CPU smoke fits (a few epochs, a few members) may be run to check the code. Nothing
   from them may be quoted.
7. The benchmark tie-outs at valuation 10 above.

The evaluator's module docstring will repeat the rules of this note, and its sha256 will be
recorded in an amendment before the first real fit, as the second study did.

# Questions for the companion's author

Sent, if Ethan agrees, before the fits; answers that change the design go in an amendment
before the first fit. None of them blocks the plan, and the plan is not changed on a guess.

1. How is the reported figure built from the three seeds (2025 to 2027): the primary seed's 20
   members only, the average over seeds, or something else? Are figures for the other two
   seeds available?
2. Are the 3.93% and 4.88% at valuation 10, the 4.36% and 4.02% averages over valuations, the
   2.96% for the network alone and the median alpha of about 0.45 from the same run (commit
   9e118a7), and which table holds them?
3. Is the training set at valuation `v` the company set selected at cutoff `v`, and are the 90,
   89, 87 and 84 companies the counts of those sets?
4. Was the choice of refit, of the premium head, of accident-year attention and of company
   batches made with cohort B's later cells in view, and were results on the fresh accident
   years 1988 to 1997 (or the second study's findings) seen before they were adopted?
5. Why is alpha fitted on diagonal C-1 and the epoch chosen on diagonal C, and was alpha
   considered for refitting after the refit?
6. What is `lr_fallback` exactly, and does the refit's support set ever differ from the
   pre-refit one?
7. What changed between the 5.04% and the 4.97% of the earlier model's two stored runs (a
   precision setting, run-to-run variation)? This tells us how much of a difference between
   ibnr's draws and a published run is noise.

# Known context, disclosed

* This design has seen the second study's results (worse on C, not distinguishable on A and
  B). The choice of cohort C3 and of running it first follows from that study, and the rule
  of P2 is the second study's. Nothing in the rules was tuned to a number of the second
  study; P1's thresholds (8 of 10 draws, the 0.5 and 1.0 of the replication test) are set from
  the second study's measured spread of draws and the relayed figures, not from any result of
  this model.
* The benchmark values at valuation 10 on B3 and C3 are known from the second study. They
  are not blind. The benchmark values at valuations 6, 7 and 8 on the new per-date sets are
  not known.
* The relayed figures of the companion are used only for the replication labels, and a
  different figure would change that label's thresholds, not P1 or P2.

# Amendments before any fit (2026-10-21)

No fit of this study exists, so these changes need no rerun. Each one is a correction found by
an independent review of the library and runner diff, and none moves a registered number.

1. **The fallback for untrained lags is matched.** Items 10 and the third verification step
   said it was "to be matched to `lr_fallback`". It is: `tail_policy="observed_cl"` with the
   `premium_lr` head substitutes, at every development step no training target supervised, the
   pooled incremental loss ratio observable at the example's cutoff
   (`network.py`, `apply_support` with `fallback_lr`; the same arithmetic in `jax_backend.py`).
   That is the companion's `prev + p * lr_fallback` at those lags. The rest of the model is
   unchanged.
2. **The per-date seed offset is 20,000, not 10,000.** Date `v` member `m` was seeded
   `seed + 10000 v + 1000 m`, so member `m + 10` of date `v` repeated member `m` of date
   `v + 1` (dates 6 to 7 and 7 to 8). The study's `date_seed` is now `seed + 20000 v`, so date
   `v` member `m` has seed `seed + 20000 v + 1000 m`. With 20 members and dates 6, 7, 8 and 10
   all (date, member) streams are distinct, and none meets a draw seed `11 + 100003 k` for
   `k` below 10 (`tests/test_tlrn_study.py`, the study 3 seed test). The library's own
   `calibration="retrain_per_valuation"` keeps `seed + 10000 v`, because changing it would move
   the published default fits.
3. **The refit stops where the weights were stored.** The training loop stores a member's
   weights at the last check that improved on the best score by more than 1e-6, and the
   selection table's `best_epoch` is the check with the lowest score; late in the cosine
   schedule the second can come after the first. The refit now stops at the stored epoch
   (`refit_epochs`); `best_epoch` keeps its meaning and its values, and a default fit does not
   move. The companion records the epoch it stores (threshold 1e-12), so this makes the two
   rules agree on which epoch a refit trains to, with ibnr's threshold left as it was.
4. **The evaluator's rules are fixed from this date.** The sha256 of the evaluator's module
   docstring, unchanged by these amendments, is
   `552bacd548ea1237da3a2ae13e8d32a9541b3a4fc49b41da6c69d1cbfc6f4df2`
   (`tests/test_tlrn_study3_eval.py` holds it). A rule change after this date needs an
   amendment here first.
5. **Evaluator interpretation choices, as the docstring states them.** A company a benchmark
   refuses at any date is left out of every date and every row, and is named in the output.
   With fewer than 10 draws the P1 condition (c) is scaled to `ceil(8 n / 10)` draws and the
   verdict is prefixed provisional. A cohort that was not run is reported as "not run" and is
   never counted as the refit hurting.
6. **A check against looking ahead.** The refit path now refuses by name a feature set whose
   scored cells include a training target, as the companion asserts in its feature step.

# Amendment before any fit (2026-10-22)

No fit of this study exists, so this change needs no rerun. It is a correction found by an
independent comparison of ibnr's network with the companion's code at commit `f50871d`
(`Code/01_functions.py`). The comparison found the 13 features and the structure identical,
and two differences in the network, both now removable by an option that is off by default.

1. **A query with nothing to read.** In the companion's attention a query whose keys are all
   unreadable gets zero weights, and the output projection still adds its bias, so the output
   there is the learned output bias. ibnr returned zero, which also dropped that bias. This
   touches fully hidden groups, such as the line attention at a future lag where no line is
   visible, an accident year with nothing revealed, and a column of accident years none of
   which reveals the cell. New option `TLRNConfig.empty_query_bias` (default `False`, so every
   earlier fit is unchanged): on, a query with no readable key returns the attention's output
   bias exactly, on the line, lag and accident-year attentions, in torch and in JAX.
2. **The extra 64 parameters were unreachable rows.** The earlier explanation (one extra row in
   each of the line and lag tables because ibnr indexes them from 1) was right about the cause
   and wrong to present it as a fixed property: row 0 of each table is never read and gets no
   gradient. New option `TLRNConfig.compact_embeddings` (default `False`): on, the line table
   has one row per line and the lag table one per lag, read 0-based, and the visible-lag table
   (twelve rows in both codes) is read at `lk` as the companion reads it, where ibnr read it at
   `lk + 1`. That is a relabeling of the same count, so the parameter count is the companion's.
   The initial distributions are the companion's (N(0, 1) embeddings, Xavier-uniform attention
   in-projections, torch-default linear layers, zero output biases), so only the random stream
   differs.
3. **The preset.** `TLRNConfig.companion_final_variant()` now sets both, and the runner's
   `companion_final_config()` writes them out. The preset's network has **18,537 parameters**,
   built and counted by a test (projection and embeddings 1,280; each of three attention blocks
   4,288; feed-forward 4,256; final layer norm and output layer 97; `beta` 40). The notebook's
   check cell and its pin test expect 18,537. The accident-year variant, which is not the
   preset, keeps 18,601.
4. **Verification.** A hand-written numpy copy of the companion's attention and block agrees
   with ibnr's network (torch) on a batch with hidden groups, and the JAX forward and its
   training trajectory agree with torch with the options on
   (`tests/test_tlrn_companion_network.py`, `tests/test_tlrn_companion_jax.py`). A default fit
   is unchanged weight for weight (the pinned 0.7.5 fits pass on both backends).
5. **Deviations that remain.** The random streams (item 1 of the deviations list); the study
   design (ten pooled fits of twenty members against the companion's one seed, item 3); and
   float32 feature rounding here against the companion's float64 (about 2e-7 relative). The
   earlier text that named 18,601 and the 1-based embeddings as a deviation is struck above.
6. **No fit exists yet.** The registered rules, cohorts, draws, statistics and decision rules
   are untouched, and so is the evaluator's docstring hash in the amendment of 2026-10-21.

# Amendment before any fit (2026-10-22), second

No fit of this study exists. This amendment changes how the fits are run and what the evaluator
prints. It changes no decision rule and no number the rules use.

1. **The notebook's install and script flow.** The repository is private and ibnr 0.7.6 will be
   on PyPI. The notebook (`analysis/05_tlrn_study3_on_colab.ipynb`) now installs
   `ibnr[jax]==0.7.6` from PyPI with no token (`SOURCE = "pypi"`, the default), downloads the
   two runner scripts (`scripts/tlrn_colab.py`, `scripts/tlrn_study_colab.py`) through the
   GitHub contents API at the release tag `v0.7.6` with a fine-grained read-only token kept as
   the Colab secret `GITHUB_TOKEN`, and saves a script only if it parses as Python. `SOURCE =
   "git"` stays as a fallback for an unreleased branch. The token is never printed, never
   written to Drive and masked in every error. The scripts are not packaged into the wheel.
   Tested against a local fake of the API; not yet run against the real private repository or
   on Colab.
2. **No zip.** The packaging step is gone. The fits already sit, flat, in the Drive folder, and
   the notebook's last cell prints the number of finished fits, the GPU hours and the evaluator
   command for the Drive folder as the laptop sees it.
3. **Three added descriptive metrics.** The metric tables and the report gain RMSE (square root
   of the mean squared error), SMAPE (mean of `200 |error| / (|predicted| + |actual|)`, a 0 to
   200 scale, a company with both zero skipped) and the number of companies whose predicted
   reserve is above the actual one. They follow the companion's own definitions (commit
   `f50871d`, `Code/07_estimates.py` lines 20 to 36) and are in the units of the mart, USD
   thousands, as his table is. They feed no rule.
4. **The companion's numbers beside ours.** For B3 at valuation 10 the report and a CSV set his
   published Table 10 figures (blend, network alone, chain ladder, the date averages, the
   median alpha, the companies per date) beside ibnr's ten draws: his value, ibnr's mean,
   minimum and maximum, and how many draws lie below and above his value. The table is
   descriptive, is not a decision rule, and says that the random streams cannot be aligned. The
   methods ibnr did not run (CL+MCL as one blend, CL+SUR, Cape Cod, Cape Cod + SUR/MCL) are
   printed as "not run in ibnr". His constants sit in one module-level dictionary in the
   evaluator, with a test that holds them equal to the published numbers.
5. **The median alpha.** The note's text above says the companion's median blend weight is about
   0.45; his Table 10 says 0.476. The table in item 4 compares ibnr's median alpha (at
   valuation 10, and over all four dates) with 0.476 and says so. The existing alpha spread is
   unchanged.
6. **Nothing else moved.** The decision rules, the thresholds, the cohorts, the draws and the
   evaluator's module docstring are unchanged: the docstring's sha256 test passes, and on the
   synthetic cohort the verdict record, the differences table and the old columns of the metrics
   table hash the same before and after (`tests/test_tlrn_study3_eval.py`, the test named
   `test_the_added_metrics_leave_every_decision_output_unchanged`).

# Frozen before any fit

When this note was written (2026-10-21) **no fit with the new options exists, on any
machine.** The fields `refit` and `alpha_on`, the `companion_final_variant` preset, the
per-date study wrapper, the config name in `scripts/tlrn_colab.py`, the `Study` entry in
`scripts/tlrn_study_colab.py` and the evaluator `scripts/tlrn_study3_eval.py` do not exist.
Nothing of this study has been run; the only work behind the note is a read of the companion's
repository at commit f50871d and of this repository's second-study notes. Pair counts at
cutoffs 6, 7 and 8 are taken from the second study's note and have not been recomputed for this
study. The rules above may still be changed by a dated amendment before the first real fit,
and not after.

# Open for the parent to decide

None of these is decided in this note; each has a stated default.

1. **Cohorts.** B3 and C3 only (default), or also a third window (a shifted accident-year
   window would need complete squares with outcomes in the mart; not checked here), or the
   second study's A.
2. **Draws.** Ten (default, as the second study) or fewer to save GPU time. Five would cut the
   estimate to about half and widen the draw average's noise.
3. **One run or two.** Produce the refit and no-refit forecasts from one training run
   (default, paired, cheaper) or fit the two arms separately.
4. **Exploratory rows.** Add a `head="ldf"` control (about 1 to 1.3 GPU hours) and a
   comparison on the alpha split (same cost), run last and outside every rule (default: not
   included).
5. **Which arm is primary.** The blended refit arm (default, the companion's headline) or the
   raw network. The note judges both by the same rule and registers the blended one.
6. **Thresholds.** The 8 of 10 draws in P1, the 0.5 and 1.0 of the replication test, the
   25-minute and 8-hour stops. All set here as defaults.
7. **The questions.** Send the questions to the companion's author before the fits, and
   whether to wait for the answers to 1, 3 and 4 before freezing. The author's name and
   repository stay out of this repository, as the attribution decision requires.
8. **The network-off arm.** Keep it if `factors_only` works with the `premium_lr` head, drop it
   otherwise.
9. **Seeds.** Reuse the second study's draw seeds (default) or choose new ones now.
10. **The sha256 of the evaluator's docstring** and the commit that freezes this note: the
    note is a draft until Ethan accepts it, and the fits should not start before it is
    committed.

## Run (2026-10-26)

The study was run on 2026-10-26 on a Colab GPU (40 fits, draws 0 to 9 of both rows on cohorts B3 and C3, ibnr 0.7.6). The results are in [the findings note](/findings/tlrn-study3-results.md). The rules above were not changed, and the two stopping rules were not triggered (the first C3 fit took about 8 minutes, the 40 fits 2.94 GPU hours).

[^request]: Ethan's instruction in the 2026-10-21 working session
[^second-results]: The results of the second tlrn study

**Accepted 2026-10-22.** Ethan accepted this note as written, taking the defaults for all ten open items (ten draws, cohorts B3 and C3, no `head=ldf` control, the blended forecast as the primary arm), and kept the library's own per-date seed spacing unchanged.
