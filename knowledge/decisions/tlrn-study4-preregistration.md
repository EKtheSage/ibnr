---
type: Decision
title: The fourth tlrn study, on the companion's revised model
description: "Pre-registration, written before any code or fit exists (2026-11-03), of ibnr's fourth tlrn study: the companion's revised model (commit b30ef02), which differs from the third study's model by one change, the removal of the per-count 'lags observed' embedding. Two co-primary questions carried unchanged from the third study (does the refit help; is the model effective against the chain ladder and the multivariate chain ladder), a new descriptive comparison with the third study's fits (P0), replication targets read from the companion's new output on both cohorts, three single-draw ablation rows that carry no verdict, the pre-fit checks, the run plan, the budget and stopping rules, and the choices still open."
tags: [tlrn, evaluation, preregistration, colab, companion-study, decision]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-11-03T12:00:00Z }
stale_after: 2027-05-03T00:00:00Z
sources:
  - id: request
    resource: Ethan's instruction in the 2026-11-03 working session
    title: "write a draft pre-registration for the fourth tlrn study, on the companion's revised model, before any code or fit exists, modelled on the third study's, with ten settled decisions"
  - id: third
    resource: /decisions/tlrn-study3-preregistration.md
    title: The third study's pre-registration, whose structure, rules and thresholds this one reuses where it says so
  - id: third-results
    resource: /findings/tlrn-study3-results.md
    title: The third study's results, which this study's design has seen
  - id: companion
    resource: /references/companion-study-repository.md
    title: Where the companion study's files live
  - id: exhibits
    resource: the companion's repository at commit b30ef02 (2026-10-30), files Exhibits/tab/HEADLINE.csv, Exhibits/holdout/tab/HEADLINE.csv, Exhibits/tab/N_T2_seed_robustness.csv, Exhibits/holdout/tab/N_T2_seed_robustness.csv, Exhibits/tab/N_T3_ablation.csv, Exhibits/holdout/tab/N_T3_ablation.csv, Exhibits/full.log and Exhibits/holdout/full.log
    title: The companion's new output, from which every target number in this note was read directly
  - id: diff
    resource: the companion's repository, Code/ directory, diff from commit f50871d (2026-10-21) to commit b30ef02 (2026-10-30), read by a delegated worker in the 2026-11-03 session and not re-read by the writer of this note
    title: The change between the two commits (the embedding removal, the ablation switches, the controls and the holdout run)
  - id: runner
    resource: scripts/tlrn_study_colab.py
    title: The Colab runner that Study 4 extends (the Study entry, the valuation dates as a field, the run record)
  - id: scope
    resource: /decisions/tlrn-entry-scope.md
    title: The attribution rule (the companion and the companion's repository are not named in this repository)
---

# Decision

The third study ran the companion's 21 October model (commit f50871d) and found that ibnr
reproduces his published figures on his own cohort B3 and that the refit hurts on the fresh
cohort C3.[^third-results] The companion's repository has since moved to commit b30ef02
(2026-10-30). Its author reports the model with one change, the removal of the per-count
"lags observed" embedding, and re-runs the main experiment on 1998 to 2007 and a holdout run on
1988 to 1997. Ethan's call (2026-11-03) is a fourth study on that model, fixed here before any
code or fit exists.[^request]

**Two co-primary questions**, carried from the third study with the same rules:

* **P1, the refit.** Does retraining each network on everything known up to the cutoff lower
  the company-reserve error compared with the network at its chosen epoch? Answered from a
  paired comparison: the two arms come from one training run.
* **P2, effectiveness.** Is the model with refit more accurate than the chain ladder (`mack`)
  and the multivariate chain ladder (`mcl`)?

**One new descriptive comparison, P0**, with no verdict: the fourth study's fits minus the
third study's, by company and by draw, labelled "embedding removed minus kept". It describes
what removing the embedding did to ibnr's own numbers, with the caveats written below.

**Descriptive ablation rows**, one draw each, no verdict: no cross-line attention, no
accident-year attention, and paid only, as the companion runs them.

Secondary questions, reported and never part of P1 or P2: does the network add anything beyond
the head's pooled pattern; does the blend with `mcl` help or hurt the network alone; and does
ibnr replicate the figures the companion reports, now on both cohorts.

The study is fixed here. If a rule below turns out to be wrong, change it in a dated amendment
at the end of this note, written before the fit it would affect, not silently.

# What changed, and what did not

The diff between the two companion commits was read by a delegated worker. Its finding, relayed
here and not re-read by the writer of this note, is that the model changed in one place that
matters to the network.[^diff]

| | third study (f50871d) | this study (b30ef02) |
|---|---|---|
| input to the first layer | features, plus line embedding, lag embedding, **and a per-count "lags observed" embedding** (a table of 12 rows by 32) | features, plus line embedding and lag embedding |
| the count as a feature | channel 6 (`lk / n_d`) present | channel 6 present, unchanged |
| parameters per member | 18,537 | 18,153 (the companion's output, `Exhibits/tab/HEADLINE.csv` line 25) |
| head, attention, features, mask, loss, optimiser, schedule, 20 members all averaged, alpha on the first diagonal, best epoch on the second, refit to the stored epoch | as in the third study | unchanged, per the diff |
| cohort rule (loss ratio below 3, at least two lines) | as in the third study | unchanged, per the diff |

The companion's reason for the removal, as the diff relays his comment: the count already
reaches the network as the feature in channel 6, and a per-count embedding leaves the oldest
accident year's count at the forecast origin at its random starting value, because no training
origin reaches that count. This is the companion's explanation and is not tested here.

Other changes in the commit are not part of this study and are listed so a reader knows they
were seen. The companion trains all 640 of its members in one compiled program (a change in
last-bit floating point at most, per the diff); it adds a Munich chain ladder, a pooled chain
ladder, a pooled incremental loss ratio and a floored blend as controls; it adds ablations with
fixed attention switches; and it reports new tables. ibnr has no Munich chain ladder, and none of
these is reproduced.

# What is verified about the new model

**Verified** means read by the writer of this note in the companion's files at b30ef02.
**Relayed** means the delegated worker's reading of the code diff. **Assumed** means inferred and
still to be checked before the first fit (the list is under "Before any fit").

* **Verified.** The companion's headline figures on his main cohort (1998 to 2007, 84 companies
  at the last date): blend Pool_APE at the last valuation 3.37%, the refitted network alone
  4.31%, the same blend without the refit 4.37%, the best classical method (chain ladder with
  the multivariate chain ladder) 5.41%, the median blend weight 0.4618, 18,153 trainable
  parameters per member.[^exhibits] On his holdout (1988 to 1997, 83 companies): blend 5.75%,
  network alone 6.29%, without the refit 4.73%, best classical 4.97%, median blend weight
  0.6617, 18,153 parameters.
* **Verified.** The companion's three seeds disagree at the last valuation, and the primary
  seed (2025), the one every headline figure uses, is the lowest of the three on both cohorts.
  Blend Pool_APE at the last valuation, main cohort: 3.37%, 3.88%, 3.62% for seeds 2025, 2026,
  2027. Holdout: 5.75%, 7.45%, 7.67% (`N_T2_seed_robustness.csv`, lines 2 to 4 of each file).
  So the companion's own spread across seeds at the last date is 0.51 points on the main cohort
  and 1.92 on the holdout.
* **Verified.** Company counts and pair counts by valuation date in the companion's logs equal
  the counts ibnr's selection rule gave in the third study: main cohort 90, 89, 87 and 84
  companies and 230, 226, 219 and 212 pairs; holdout 84, 84, 83 and 83 companies and 237, 236,
  232 and 232 pairs (`Exhibits/full.log`, `Exhibits/holdout/full.log`, `D_T0b_cohort_by_valuation.csv`).
  Equal counts do not show equal pairs; the pairs were compared and are equal (see the cohort
  section).
* **Relayed, from a delegated check.** The companion's cohorts equal ibnr's pair for pair, and his data files equal the mart cell for cell (cohort section).
* **Verified.** The refit hurt on the holdout in the companion's own run (5.75% against 4.73%
  without it, +1.02 points), as it did in the third study's C3 (+1.07).
* **Relayed.** The embedding is removed and nothing else in the network changes; the other
  parameters start from the same random draws as before in the companion's JAX code.
* **Assumed.** That removing the table in ibnr, while still consuming the random draws that
  would have filled it, leaves every other starting weight equal to the third study's for the
  same seed. Checked before any fit (test 2 below).

# How ibnr runs it

Planned for ibnr 0.7.7. **None of the code below exists when this note is written.** The names
are decided; the behaviour is what the rules below assume.

| companion choice | ibnr setting |
|---|---|
| no per-count embedding, channel 6 kept | new boolean `TLRNConfig.maturity_embedding`, default `True` (so every earlier fit and pin is unchanged), and a new preset `companion_revised` that is `companion_final` with only `maturity_embedding=False`; 18,153 parameters expected |
| same starting weights as the third study | the table's random draws are still consumed but the table is not registered or used, so for the same seed every shared parameter starts equal to the third study's |
| no cross-line attention | `attention=("lag","ay")`, `cross_line=False` |
| no accident-year attention | `attention=("line","lag")` |
| paid only | a new small option that sets the five incurred and case-reserve input features of the 13 to zero before the first layer, as the companion does (the feature channels to be read from `FEATURE_NAMES` and confirmed against the companion's channels 9 to 13 counted from 1, before any fit); the existing `incurred_field=None, case_field=None` route is not used because it gives 8 inputs, a different network |
| the runner | a new config name in `scripts/tlrn_colab.py` written out field by field, a new `Study` entry in `scripts/tlrn_study_colab.py` with the valuation dates as a field of `Study` and a branch for the paid-only row, a copy of the evaluator as `scripts/tlrn_study4_eval.py`, a new notebook `analysis/06_tlrn_study4_on_colab.ipynb` |

**Not run, and why.** "No development attention" and "no attention at all" are refused by ibnr
("attention must include 'lag'") and need a larger change; they are not part of this study. The
companion's Munich chain ladder is not in ibnr. His bit-for-bit training is out of reach (ibnr
starts from torch's weights under its own seeds and its dropout draws differ), so replication is
judged on distributions, as in the third study.

# What is run (the rows)

* **`tlrn_cr_refit`, the main row.** The `companion_revised` preset, 3,000 epochs, 20 members at
  each of four valuation dates, refit, alpha on the first diagonal. One run gives the refit
  blend (the model, judged by P2), the blend without the refit (the comparison arm of P1,
  paired by construction), the refitted network alone and the network alone without the refit,
  as columns of the same run, exactly as in the third study.
* **`tlrn_cr_factors`, the network-off row.** The same configuration with `factors_only=True`.
  **Conditional rule, fixed now.** With the network off the embedding is never read, so the row
  may be identical to the third study's `tlrn_cf_factors`. Before any fit, run a small fit
  (two members, ten epochs, on the laptop's CPU, on both cohorts at date 10, the same seed) with
  `companion_final` and with `companion_revised` and compare the forecast tables. If every
  forecast is equal to the last bit, the third study's `tlrn_cf_factors` fits are reused as this
  study's `tlrn_cr_factors` (read from the third study's folder, recorded in the fourth study's
  records as reused, with their own environment named), and no factors fit is run. If any
  forecast differs, the row is fitted fresh, ten draws per cohort. The answer is recorded in an
  amendment before the first real fit.
* **Ablation rows, descriptive only.** `tlrn_cr_noline`, `tlrn_cr_noay` and `tlrn_cr_paid`, draw
  0 only, cohort B4 only. No rule below uses them. See "The ablation rows".
* **`tlrn_cf_rerun`, the rerun control, descriptive only.** One fit of the third study's
  `companion_final` preset on B4 at draw 0, on the same Colab image as this study's fits. Its
  date-10 Pool_APE minus the third study's B3 draw 0 measures what a rerun of an unchanged
  specification moves, in this environment. P0 is read against it (caveat 2 there). No rule
  uses it.
* **The raw network.** The refit networks before the blend, read from the main fits at no
  extra cost.

# Cohorts, and the selection rule at each date

* **Cohort B4.** The definition of B3: accident years 1998 to 2007, last known diagonal
  2007-12-31, cas-schedule-p publish 20260613_041006, valuation dates 6, 7, 8 and 10
  (year ends 2003, 2004, 2005 and 2007). This is the companion's main cohort.
* **Cohort C4.** The definition of C3: accident years 1988 to 1997, last known diagonal
  1997-12-31, same publish, dates 6, 7, 8 and 10 (year ends 1993, 1994, 1995 and 1997). By
  count this is the companion's holdout cohort.

The rule at each valuation date is the third study's, unchanged: (1) a complete 10-by-10 square
with a finite positive net earned premium on every cell, post-cutoff cells included; (2) on the
cells with calendar index at most `v`, finite positive paid and premium and a cumulative paid
loss ratio below 3; (3) at least two such lines per company. The company set `S_v` is nested
(`S_10` inside `S_8`, `S_7` and `S_6`). Expected sizes, from the third study and equal to the
companion's logs: pairs 230, 226, 219 and 212 on B4 and 237, 236, 232 and 232 on C4; companies
90, 89, 87 and 84 on B4 and 84, 84, 83 and 83 on C4. If ibnr's rule returns other counts, the
study uses ibnr's counts and the evaluator reports the difference.

**The pre-fit pair comparison (registered, and done 2026-11-03: passed).** The rule was to
compare the companion's company-line pairs at each of the four dates, pair by pair, with ibnr's
`select_pairs` on both cohorts, and to **proceed only if the pairs are identical at every date
on both cohorts**; if they differed, record the difference in an amendment and ask Ethan before
any fit. A separate read-only worker re-ran the companion's own `download_triangle` and
`select_cohort` (`Code/02_data.py` at b30ef02) on his cached raw CAS files and compared the
result with ibnr's `select_pairs` on the cas-schedule-p publish. The pairs are identical at
dates 6, 7, 8 and 10 on both releases, with empty set differences in both directions. Pairs
(companies): 1998 to 2007: 230 (90), 226 (89), 219 (87), 212 (84); 1988 to 1997: 237 (84), 236
(84), 232 (83), 232 (83). The same worker found the companion's CAS 2021-04 `*_pos.csv` files
and the mart equal cell for cell (62,010 cells on the main release and 67,500 on the holdout
release; largest absolute difference 0.0 for paid, premium and incurred), and found the date-10
benchmarks equal to the printed digits (chain ladder 0.054810 and multivariate chain ladder
0.054098 on the main release; 0.119470 and 0.049693 on the holdout; unrounded differences at
most 1e-13). So B4 is the companion's main cohort and C4 his holdout cohort, literally. The
rule stays in force: if the selection code or the publish changes before the first fit, the
comparison is repeated.

**What each cohort's verdict means.** B4 is the cohort the companion's design was chosen on, so
a verdict on B4 is **confirmatory of a fixed specification, not out of sample**. C4 is out of
sample in accident years only, as C3 was, and it is weaker than that: the third study's result
on the same cohort (the refit hurts) existed before these rules were written, the companion's
holdout run is in the same commit as his embedding removal so the C4 targets below are not
blind, and C4 shares 60 of its 83 companies and 155 of its 232 pairs with B4.

# Draws and seeds

Ten independent fits per cohort and row, the third study's arithmetic unchanged: draw `k` uses
fit seed `11 + 100003 k`, date `v` adds `20000 v`, member `m` adds `1000 m`. Keeping the
arithmetic is what makes P0 a pairing: draw `k` of the fourth study starts from the same
random seeds, and (see the embedding rule above) the same shared starting weights, as draw `k`
of the third. The companion's seeds (2025 to 2027) are not used. 2 cohorts x 10 draws = 20 main
fits. The ablation rows use draw 0 only (seed 11), the companion's primary-seed-only protocol.

# Benchmarks

`mack` and `mcl` from ibnr's gallery, fitted and scored on the same `S_v`, exactly as in the
third study. At date 10 they must reproduce the third study's saved values to rounding: `mack`
5.481% and `mcl` 5.410% on B4, `mack` 11.947% and `mcl` 4.969% on C4. The companion's best
classical figure at date 10 (chain ladder with the multivariate chain ladder) is 5.41% on B4
(`HEADLINE.csv` line 5) and 4.97% on C4 (holdout `HEADLINE.csv` line 5), which agree with the
`mcl` figures. The benchmarks are the same programs on the same data as the third study, so
they are a tie-out and not a result.

# The statistics

The third study's definitions, unchanged. A company's reserve error at valuation `v` is its
predicted minus its actual total over the scored cells (diagonals after `v`, development lag at
most `v`, and at valuation 10 every cell past the last diagonal: 14, 15, 13 and 45 cells per
pair). **Pool_APE** at `v` is the sum of the absolute errors over the scored companies of `S_v`
divided by the sum of their actuals. The **draw average** at `v` is the mean over the ten draws
of each company's prediction. The **dominant company** is the one with the largest actual
reserve at valuation 10, found in code (company 1767 in the third study), and the study stops
if it is not in every `S_v`. The **dollar-pooled Pool_APE** over valuations 6, 7, 8 and 10 is
the sum, over the four valuations and all companies of that date's set, of the absolute company
error divided by the sum of the company actuals over the same company-date pairs. The
**unweighted mean** over the four valuations is reported and takes part in no rule.

**The bootstrap.** 10,000 resamples of companies with replacement, taken from the union of the
four sets. A company drawn `n` times counts `n` times at every date at which it is in the date's
set and not at all at a date where it is not. The same resample is used for the model and the
benchmark (or the two arms of a pairing), for every valuation, and a company's dates stay
together. The interval is the 2.5th to 97.5th percentile of the difference. Seed 2008 for the
primary intervals. The **stratified sensitivity bootstrap** keeps the dominant company in every
resample exactly once and resamples the others (seed 2009); it is reported beside the primary
one for P1 and P2 and replaces nothing. "Wholly below zero" means the upper end is below zero;
"wholly above" means the lower end is above zero.

# The rule for P1: does the refit help?

Carried word for word from the third study, with B4 and C4 for B3 and C3 and `tlrn_cr_refit`
and the no-refit arm of the same run for `tlrn_cf_refit` and `tlrn_cf_norefit`.

Let `D(v)` be the draw-average Pool_APE of the refit arm minus that of the no-refit arm (points;
negative favours the refit), `D10 = D(10)`, and `Dpool` the same difference for the
dollar-pooled statistic. Per cohort:

* **refit helps** needs all four: (a) the bootstrap interval of `D10` is wholly below zero;
  (b) the interval of `Dpool` is wholly below zero; (c) in at least 8 of the 10 draws, each
  draw's own valuation-10 Pool_APE with the refit is below its Pool_APE without it; (d) with
  the dominant company left out, the point value of `D10` is at or below zero.
* **refit hurts** if the interval of `D10` or of `Dpool` is wholly above zero.
* **not distinguishable** otherwise.

**A descriptive label that changes nothing**: **date-dependent** if the point value of `D10` is
below zero and the unweighted mean over the four valuations of `D(v)` is above zero.

**Across cohorts.** C4 is the headline for a claim beyond the companion's data; B4 is the
companion's own cohort. Overall statement, printed by the evaluator:

| C4 | B4 | overall statement |
|---|---|---|
| helps | helps | "the refit helps on the fresh cohort and on the companion's" |
| helps | not helps | "the refit helps on the fresh cohort only; B4 is listed" |
| not distinguishable | any | "not distinguishable", with B4 listed for information |
| hurts | any | "the refit hurts on the fresh cohort" |

If the refit helps on B4 and is not distinguishable or hurts on C4, the statement is that the
benefit is not shown beyond the cohort the companion's design was chosen on. A cohort not run,
or with fewer than the registered draws, is labelled provisional and not counted as hurting.

# The rule for P2: is the model effective?

Carried word for word from the third study, applied to `tlrn_cr_refit` against `mack` and `mcl`
on each cohort and the per-date company sets above. `effective` needs all four: (i) at valuation
10 the Pool_APE difference (model minus benchmark) has a 95% interval wholly below zero against
both benchmarks; (ii) the same for the dollar-pooled difference over valuations 6, 7, 8 and 10
against both; (iii) the median over the draws of the valuation-10 Pool_APE is below both
benchmarks; (iv) without the dominant company, either the valuation-10 point difference is at or
below zero against both, or condition (i) holds without it. `worse` if the interval in (i) or
(ii) is wholly above zero against **either** benchmark. `not distinguishable` otherwise. The
overall statement: C4 effective and B4 effective gives "effective on the fresh cohort,
confirmed on the companion's"; C4 effective and B4 not worse gives "effective on the fresh
cohort; on the companion's cohort not worse"; C4 effective and B4 worse gives "mixed", with no
claim of effectiveness; C4 not distinguishable gives "not distinguishable" whatever B4 says; C4
worse gives "worse on the fresh cohort".

# P0: embedding removed minus kept (descriptive, no verdict)

The fourth study's saved fits minus the third study's, for the same cohort, row, draw, date and
company. No threshold, no label, no interval is compared with a rule. The evaluator reports:

* per draw, the date-10 Pool_APE of each study's refit blend and the difference (ten numbers per
  cohort), and the median and range of the ten differences;
* for the draw-averaged forecasts, the difference in date-10 Pool_APE and in dollar-pooled
  Pool_APE with the same company-bootstrap interval as P1 (seed 2008), with and without the
  dominant company, and the number of companies whose absolute error is smaller in the fourth
  study;
* the same for the no-refit arm and the raw network;
* the companion's own move, set beside it for reading only: the blend at date 10 went from 3.93%
  (the figure the third study replicated against) to 3.37% on the main cohort, a fall of 0.56
  points.

P0 is computed only if the per-date pair hashes of the two studies are equal on that cohort. If
the third study's fits are not available, P0 is not reported.

**Caveats, written before any result.**

1. **The pairing is in starting weights and seeds, not in training.** Draw `k` shares its seed
   arithmetic and, by the embedding rule, every shared starting weight with the third study's
   draw `k`. But removing the embedding changes every gradient, so the trajectories part
   immediately. GPU fits are not repeatable bit for bit (the third study's note), so even an
   unchanged specification would not reproduce the third study's draws.
2. **P0 can separate the effect of the embedding from rerun-to-rerun variation only roughly.**
   The rerun control `tlrn_cf_rerun` gives one measurement of that variation (one fit, one
   draw, one cohort), printed beside P0's draw-0 difference on B4; one number is a scale, not a
   distribution. A difference smaller than the draw-to-draw spread of the third study
   (0.53 points of Pool_APE at date 10 across the ten refit-blend draws on B3, from 3.73% to
   4.26%; 1.95 points on C3, from 4.25% to 6.20%) is not interpretable.
3. **The environments may differ.** The third study ran on ibnr 0.7.6, jax 0.11.1 and Python
   3.13.16 on a Colab RTX PRO 6000. The fourth runs on ibnr 0.7.7 and whatever the Colab image
   then holds. Each fit record names its versions; a difference is disclosed, not corrected.
4. **It is a statement about ibnr, not about the companion's change.** The companion's commit
   changed other things (the batching of members, the controls, the data source files), none of
   which ibnr copies. The difference measures what removing this one table does to ibnr's
   network.
5. **It is a many-way comparison on one cohort window and ten draws.** There is no correction
   for multiplicity, and C4 and B4 overlap in companies.

# Secondary questions, reported and never in P1 or P2

* **The network.** `tlrn_cr_refit` minus `tlrn_cr_factors`, with the third study's rule: *network
  adds value* only if the valuation-10 and the dollar-pooled intervals are both wholly below
  zero and, without the dominant company, the valuation-10 interval is not wholly above zero;
  *network hurts* if either of the first two is wholly above zero; otherwise not
  distinguishable.
* **The blend.** `tlrn_cr_refit` minus the raw refit network: *blend adds value* on the same
  condition; *blend hurts* if the valuation-10 or the dollar-pooled interval is wholly above
  zero; otherwise not distinguishable. On the companion's new figures the blend is better than
  the network alone at date 10 on both cohorts (3.37% against 4.31% on the main cohort, 5.75%
  against 6.29% on the holdout); the third study's relayed figures had the
  opposite order on the main cohort (3.93% against 2.96%).
* **Replication of the companion**, below.
* **Alpha.** Minimum, quartiles, median, maximum and standard deviation of the fitted blend
  weight over all members of all draws, per valuation, beside the companion's medians (0.4618
  main, 0.6617 holdout).
* **Also reported:** Pool_PE (bias); the median absolute percentage error; the mean absolute
  error; RMSE; SMAPE; the number of companies over-estimated; the mean absolute log of
  predicted over actual reserve; the share of companies where the model has the smaller
  absolute percentage error; Pool_APE within quartiles of actual reserve; every key figure with
  and without the dominant company; the draw-to-draw spread; the width of every interval and the
  gain it could just detect (half the width). All as in the third study, all descriptive.

# Replication of the companion's reported figures

Judged at valuation 10, on both cohorts separately, against the figures in the companion's
`HEADLINE.csv` files at b30ef02 (his primary seed, a single fit, not verified by ibnr). Let `m`
be ibnr's draw average and `[lo, hi]` the range of the ten draws' own Pool_APE.

**B4, targets from `Exhibits/tab/HEADLINE.csv`:**

* **R1, the level.** `3.37` lies in `[lo, hi]` of `tlrn_cr_refit` and `|m - 3.37| <= 0.5`.
* **R2, the refit effect.** `D10 <= -0.5`. The companion's difference is -1.00 (3.37% against
  4.37% without the refit).
* **R3, the network alone** (secondary). `|m_net - 4.31| <= 0.5` for the raw refit network.

**C4, targets from `Exhibits/holdout/tab/HEADLINE.csv`:**

* **R1c, the level.** `5.75` lies in `[lo, hi]` and `|m - 5.75| <= 0.5`.
* **R2c, the refit effect.** `D10 >= +0.5`. The companion's difference is +1.02 (5.75% against
  4.73% without the refit): the refit hurts, and the rule asks that ibnr show a hurt of at
  least half that size.
* **R3c, the network alone** (secondary). `|m_net - 6.29| <= 0.5`.

**Labels, per cohort, as in the third study.** *replicates* if R1 and R2 both hold (R1c and R2c
on C4). *fails to replicate* if neither holds and `|m - target| > 1.0`. *partially replicates*
otherwise. R3 is reported beside the label and does not change it. The widths 0.5 and 1.0 are
kept from the third study for the reasons given there: 0.5 is a little more than the range of
the ten draws of the second study on B (0.40 points wide), and 1.0 is about the size of the
companion's refit effect. They test whether a single published run could be one more draw from
this procedure, not whether ibnr and the companion agree to the digit.

**How a miss of R1 will be read, written before any result.** The third study found 4.02%
against the companion's then 3.93%. The companion now reports 3.37%, which is 0.65 below the
third study's ibnr figure and 0.56 below his own earlier figure, both more than the 0.5 width.
So R1 on B4 can fail even if ibnr is correct, and the following readings are fixed now:

1. If `m` is within 0.5 of 3.37 and 3.37 is in the range of the ten draws, R1 holds, and the
   embedding removal has moved ibnr's level in the direction and by about the amount the
   companion's moved.
2. If `m` is between 3.87 and 4.37 (a miss of up to 1.0 on the high side), the label is
   *partially replicates* by the rule above, and the results note will say that ibnr's level did
   not follow the companion's drop. This is not read as a defect. Three facts bound how much it
   means: the companion's own seed 2026 gives 3.88%, 0.51 above his headline; the third study's
   ibnr draws span 0.53 points; and the forward pass is checked against the companion's before
   any fit (test 4 below), so a defect would have to be in training, not in the network.
   Candidate causes are listed in the results note and none is tested: the random streams, the
   companion's joint training of 640 members, and changes in the commit that the diff reading
   judged equivalent without running them.
3. If `m` is above 4.37 or below 2.37 and R2 also fails, the label is *fails to replicate*.
   Because this would differ from the companion by more than anything his seeds or ours
   produce, **stop and ask Ethan** before writing the results; the pre-fit checks make an
   implementation defect in the forward pass unlikely, so the first things to look at are the
   run record and the training logs.
4. Whatever R1 says, P1 and P2 are decided by their own rules, unchanged, and a miss of R1
   changes no verdict.
5. P0 is read beside R1 and not into it. If R1 misses and the P0 difference at date 10 on B4 is
   small compared with the third study's draw spread, the sentence is "removing the embedding
   did not detectably move ibnr's level", which says nothing about why the companion's moved.

**The companion's three seeds, beside the targets (descriptive, no label).** His headline is his
primary seed, and it is the lowest of his three seeds at date 10 on both cohorts. So the
evaluator also prints `m` beside the mean and range of his three seeds (`N_T2_seed_robustness.csv`
lines 2 to 4, column `TLRN_Pool_APE_10`): on B4 mean 3.62, range 3.37 to 3.88; on C4 mean 6.96,
range 5.75 to 7.67. This comparison changes no label; it says where ibnr's draws sit against
everything the companion ran rather than his best run.

**How a miss on C4 will be read.** The companion's primary seed is the lowest of his three on
the holdout (5.75%, against 7.45% and 7.67% for the others, a spread of 1.92 points), and the
third study's C3 draws spanned 1.95 points. The widths are kept as decided, but a draw average
of more than 6.25% on C4 is unsurprising and the results note will say that the target is the
lowest of the companion's three seeds. R2c is judged on the direction and size of the refit
effect, where the third study's C3 already found +1.07.

# The ablation rows

Three fits on cohort B4 at draw 0, mirroring the companion's protocol (primary seed only, each
variant with its own fitted blend weights, blend scored against blend and network against
network). **Descriptive only. No verdict rule, no interval compared with a threshold, no
statement that a component helps or hurts.** The evaluator prints, for each row, the date-10
Pool_APE and the mean over the four dates, for the blend and the raw network, beside the same
quantities for the main row's draw 0, and beside the companion's figures.

| row | setting | companion's blend / network, mean over dates (`HEADLINE.csv` lines 28 to 33) |
|---|---|---|
| main row, for reference | `companion_revised` | 4.40 / 4.71 |
| `tlrn_cr_noline` | no cross-line attention | 4.10 / 4.47 |
| `tlrn_cr_noay` | no accident-year attention | 4.36 / 3.76 |
| `tlrn_cr_paid` | paid only (the five incurred and case-reserve features set to zero) | 5.28 / 6.73 |

**Why they carry no verdict.** One draw has a spread of about half a point at date 10 on B3
(the third study), which is as large as the differences in the table; the companion's table
rests on one seed too. A comparison of one fit with one fit cannot say which component matters.
They are run so that a reader can see ibnr's version of the companion's table, and so that a
later study can decide whether ten draws of any of them are worth buying.

**Differences from the companion's ablations, disclosed.** The companion multiplies the removed
attention residual by zero inside one network, so every other parameter starts where the main
model's starts. ibnr removes the cross-line attention by skipping it while still building the
module, so the starting weights of that row are unchanged; but ibnr builds the accident-year
attention module only when it is used, so in `tlrn_cr_noay` the weights of every module built
after it start at different values from the main row's. The paid-only row keeps 13 inputs, as
the companion's does, so its first layer starts as the main row's does. Whether the added
option zeroes the same five channels as the companion's is a registered check below. "No
development attention" and "no attention at all" are not run because ibnr refuses them.

# Known deviations from the companion

Each is disclosed in the evaluator's output and the results note.

1. **Random streams.** ibnr's JAX streams differ from the companion's, so the same procedure
   gives different numbers; replication is judged on distributions.
2. **Draw design.** Ten ibnr draws per cohort against one reported fit of the companion's at its
   primary seed, which is the lowest of his three at the last date on both cohorts.
3. **Training in one program.** The companion trains all members of all dates and variants in
   one compiled program; ibnr fits each date on its own cohort. Assumed equivalent in
   expectation, not tested.
4. **Selection reuse, alpha not refitted, the refit length, the checkpoint threshold.** As in
   the third study (its deviations 4, 5, 7 and 8).
5. **Survivorship.** The selection rule requires post-cutoff cells to exist. It applies to the
   benchmarks as well, and tends to flatter every method's level of error.
6. **Data files.** The companion reads the CAS 2021-04 files; ibnr reads the cas-schedule-p
   publish 20260613_041006. Pairs and cell values were compared and are identical (above).
7. **Float precision.** float32 feature rounding in ibnr against the companion's float64.
8. **Controls not run.** The Munich chain ladder, the pooled chain ladder, the pooled
   incremental loss ratio and the floored blend are not reproduced (the network-off row is).

# What is NOT claimed

* Nothing about why a result comes out as it does. The companion's explanation for removing the
  embedding is untested here, and so is every explanation of P0.
* Nothing from a verdict on B4 as out-of-sample evidence.
* Nothing about other accident-year windows, other lines, other companies or a new economic
  environment. C4 shares most of its companies with B4.
* Nothing from the ablation rows beyond what they print.
* Nothing about the companion's code beyond commit b30ef02, and nothing about the figures he
  reports beyond the files named in the sources.
* No claim that ten draws sample training randomness well, or that the intervals measure
  uncertainty about future years: the bootstrap resamples companies only. One company holds
  about two thirds of the reserve at valuation 10, so every dollar-based statistic depends on
  it, and the without-that-company figures are printed.
* No claim of precision. The third study's intervals at date 10 were 5 to 8 points wide against
  the benchmarks and 2 to 3 points wide for the refit comparison.
* No multiplicity correction.

# Budget, run plan and stopping rules

**Where and what.** Colab, G4 with an RTX PRO 6000 Blackwell Server Edition, as the third study.
Drive folder `ibnr_colab_cf/study4` (smoke runs in `study4_smoke`). The notebook installs
`ibnr[jax]==0.7.7` from PyPI and fetches the runner scripts at the tag `v0.7.7`. **ibnr 0.7.7 is
not released when this note is written; the fits do not start before it is.**

**Order.** (1) C4 draw 0, as a pipeline check; (2) C4 draws 1 to 9; (3) B4 draws 0 to 9; (4) the
factors rows, if they are fitted fresh; (5) the three ablation rows and the rerun control (B4,
draw 0). No result of
any draw may change a rule: the only reasons to stop or change course are a defect, non-finite
values or the budget.

**Estimate, from the third study's measured times.** A main fit took a median of 521 s on B3 and
478 s on C3; a network-off fit 31 to 33 s; 40 fits took 2.94 GPU hours. Removing a table of 384
parameters should not change the time. So: 10 x 478 s + 10 x 521 s is 2.78 hours; the factors
rows (if fitted) 20 x 32 s is 0.18 hours; the three ablation rows and the rerun control at
about 9 minutes each is about 0.6 hours. **Planning figure: about 3.6 GPU hours.** An estimate, not a measurement.

**Stops.**

* Stop and ask if draw 0 on C4 takes longer than 25 minutes, or if the cumulative GPU time passes
  8 hours before all registered fits are done.
* A member with a non-finite loss stops its fit; the same fit is rerun once with the same seed;
  if it recurs the cohort is reported without that draw and its verdict is provisional.
* Every non-smoke fit must show that every member ran 3,000 epochs (the evaluator refuses
  otherwise) and that the refit ran to each member's own selected epoch.
* A cohort without exactly draws 0 to 9 of every registered arm has a provisional verdict, printed
  with the draw count and the missing numbers.
* An ablation row that fails is reported as failed; it does not delay the verdicts.

# Before any fit

Checks that fit nothing and must pass before the first fit:

1. **Parameters.** `companion_revised` builds 18,153 parameters and has no `nobs_emb`.
   `dataclasses.asdict` of its config and of `companion_final`'s differ in exactly
   `maturity_embedding`. `maturity_embedding` defaults to `True`, refuses a non-boolean, and
   every existing pin on `companion_final` (including 18,537) is unchanged.
2. **Starting weights.** For the same seed, every parameter the two presets share starts with
   exactly the same values (this pins the rule that the table's draws are consumed).
3. **The pair comparison** against the companion's pairs, both releases, all four dates: done
   2026-11-03 and passed (above); repeat it if the selection code or the publish changes.
4. **Forward parity.** With shared weights loaded, ibnr's forward pass (torch, and JAX) equals a
   numpy transcription of the companion's b30ef02 forward pass, after the pattern of the
   existing parity test for `companion_final`. Bitwise training parity is not offered.
5. **The option is connected.** The option reaches `gallery.fit` on both backends: trained
   networks lack the table and forecasts differ from `companion_final`'s. A test of the preset's
   config alone would pass if the option were accepted and then ignored.
6. **The factors-row identity check** (above), and its result recorded in an amendment.
7. **The ablation switches.** The new paid-only option zeroes the channels the companion's does,
   read from `FEATURE_NAMES`; the other two rows build the expected parameter counts.
8. **The benchmark tie-outs** at date 10 (above).
9. **Run records.** The run record of each fit gains the versions of pandas, numpy and duckdb
   (see below), checked by a test of the record writer.
10. Tiny CPU smoke fits (a few epochs, a few members) may be run to check the code. Nothing from
    them may be quoted.

The evaluator's module docstring will repeat the rules of this note, and its sha256 will be
recorded in an amendment before the first real fit, as the third study did. **This note is
committed alone, before any code, and accepted by a second dated commit**, so that git shows
that the rules came first (the third study's note was committed in the same commit as its code
and git cannot show the order).

# The data hash, and the versions to record

In the third study every fit record holds a data hash that differs from the one the evaluator
builds from the same publish: `bce499efbb6bde4e` (fits) against `2d9ecc777b088fcd` (evaluator)
for B3, and `e52d50e002d50e16` against `92f5d7a74f37a0c2` for C3. The hash is a sha256 of the
text `DataFrame.to_csv` writes for the cohort's cells. The hashes of the selected pairs and of
each date's pairs agreed, so the same companies and lines were selected; the evaluator's hash
was reproduced exactly on the laptop's locked environment, so only the Colab runs differ; and
the cause is **unexplained**, because the run record kept only the versions of ibnr, jax,
Python and the platform, not of pandas, numpy or duckdb (the third study's results note).

For this study:

* Each fit record gains the versions of pandas, numpy and duckdb, and the notebook prints them
  with the cohort's data hash in the first cell, before any fit, so the question is answered on
  the machine where it arises.
* The evaluator's tolerant data check is kept as the third study's was: a data hash that differs
  while the anchors, premiums, lines, companies and per-date pair hashes agree does not stop the
  study, and `--strict-data-hash` is not used as a stop.
* The fourth study's own fits are compared with the evaluator's hash, and the result is reported
  in the results note whether or not it matches. Nothing in a rule depends on it.

# Questions for the companion's author

None blocks the plan. Sent, if Ethan agrees, before the fits; answers that change the design go
in an amendment before the first fit.

1. How are the other two seeds meant to enter a published figure? At the last date the primary
   seed is the lowest on both cohorts.
2. Was the holdout cohort run after the design in the commit was fixed, and was the embedding
   removal decided with the holdout's results in view?
3. Was the embedding's effect measured alone (the same seeds with and without it), or did other
   things change between the two runs besides the removal?

# Known context, disclosed

* This design has seen the third study's results (the refit helps on B3 in every draw but is
  not distinguishable by the pooled interval; the refit hurts on C3) and the companion's new
  output. The thresholds of P1 and P2 are the third study's, set before they were seen; the
  replication targets are the companion's new figures, read from his files by the writer of this
  note and not tuned to any ibnr result.
* The benchmark values at date 10 on both cohorts are known from the third study. They are not
  blind.
* The third study's fits are the comparison for P0, so P0 is not blind to them.

# Frozen before any fit

When this note was written (2026-11-03) **no code for this study exists and no fit has been
made, on any machine.** The field `maturity_embedding`, the preset `companion_revised`, the
paid-only option, the runner entries, the evaluator `scripts/tlrn_study4_eval.py` and the
notebook do not exist. The work behind the note is a read of the companion's output files at
b30ef02, a delegated read of the code diff between the two commits, a delegated pair-by-pair
cohort comparison (passed, above), and this repository's
third-study notes. The rules above may still be changed by a dated amendment before the first
real fit, and not after.

# Open for the parent to decide

None of these is decided in this note; each has a stated default.

1. **The status of this note.** A draft until Ethan accepts it. Accept it in a separate commit
   after the note is committed alone, and only then write code.
2. **Release.** The fits need ibnr 0.7.7 on PyPI with the new field and preset. Default: release
   it as a patch, as the versioning decision asks.
3. **A rerun control.** Decided 2026-11-03: included, as the row `tlrn_cf_rerun` (about 9
   minutes).
4. **The ablation rows at ten draws.** Run later as a separate study 4b if the draw-0 rows look
   worth it (about 1.4 GPU hours each per cohort). Default: draw 0 only.
5. **The factors row.** Reuse the third study's fits if the identity check passes (default), or
   fit it fresh.
6. **The questions** to the companion's author: send before the fits, or not. His name and
   repository stay out of this repository, as the attribution decision requires.
7. **The sha256 of the evaluator's docstring** and the commit that freezes this note.

[^request]: Ethan's instruction in the 2026-11-03 working session
[^third-results]: The results of the third tlrn study
[^exhibits]: The companion's new output at commit b30ef02
[^diff]: The diff between the companion's commits f50871d and b30ef02, read by a delegated worker

**Accepted 2026-11-03.** Ethan accepted this note as written.

# Amendment before any fit (2026-11-03): the identity check, the evaluator's hash and the code

Written after the code exists and before any real fit. The original text above is unchanged.
No decision rule, threshold, target or width above was changed.

**1. The factors-row identity check, run.** The note said the third study's `tlrn_cf_factors`
fits may stand in for `tlrn_cr_factors` if the two configurations give the same answer. The check
was run with the new code: cohorts B (84 companies) and C (83 companies) from the pinned mart,
valuation 10, 2 members, 10 epochs, the seed that `date_seed(11, 10)` gives, `factors_only=True`,
through `fit_valuation`, for `companion_final_config()` and for the same configuration with
`maturity_embedding=False`, on torch (CPU) and on jax (CPU). The valuations table (all four
forecast columns) and the members table were bit-identical (`np.array_equal`, maximum absolute
difference 0.0) in all four cases. With the network off the per-count table is not part of the
model, so the two configurations are the same model; this is what the check shows. **Decision:**
the third study's `tlrn_cf_factors` fits are reused for `tlrn_cr_factors` (open item 5 above,
default taken). The runner takes them with `--reuse-factors-from` and writes a record with this
study's own settings, `fit_seconds` 0.0 and the source record, version, environment and seconds
under `run["reused_from"]`; it refuses a source fit whose study, row, cohort, draw, seed,
publish, valuation dates or per-date pair hashes differ, or whose configuration differs in
anything but `maturity_embedding`. What the check does not show: it was a ten-epoch, two-member
CPU run, and the third study's fits ran on a GPU, whose floating-point sums can differ in the
last bits; the two are the same model, not the same bits on that hardware.

**2. The evaluator's docstring.** `scripts/tlrn_study4_eval.py` repeats this note's rules in its
module docstring. The sha256 of that docstring (UTF-8, as `ast.get_docstring(..., clean=False)`
returns it) is
`7357336acaf0f854b35d3b547c47f360b9c2964ec4ee6c7b7607603449fba029`, pinned in
`tests/test_tlrn_study4_eval.py`. It carries one wording change from this note: the rows without
a verdict are called "variant rows" (the note uses the banned word family).

**3. Choices the code made where the note left a detail open.**

* *The version pin.* Only this study pins ibnr 0.7.7 (`COMPANION_REVISED_PINNED_IBNR`). The
  module constant `PINNED_IBNR` stays 0.7.6 for the third study, whose notebook, how-to and
  pin tests are history. The notebook stops if the installed version is not 0.7.7.
* *The stops.* The runner applies both stops itself (8 GPU hours before every fit, counting
  trained fits only; 25 minutes for draw 0 of the main row on C, checked the moment it is saved
  and again on a rerun). A fit with a non-finite forecast or stored epoch is rerun once with the
  same seed and then reported in a `FAILED_<fit>.txt` file; a failed single-draw row delays
  nothing; a failed main-row or factors fit makes its cohort's verdict provisional. A smoke run
  has no stops.
* *Single-draw rows* are planned once, at draw 0 on B. An explicit request for one with more
  than one draw, or on C, is refused by name.
* *Reading rule (5).* "Small compared with the third study's draw spread" is made exact as: the
  absolute valuation-10 difference of the draw averages is no larger than the width of the range
  of the third study's ten refit-blend valuation-10 Pool_APE values. The third study's published
  spreads (0.53 and 1.95 points) are printed beside it.
* *The companion's figures for the variant rows* exist only as means over the four dates (the
  note's table), so the table prints those beside ours and no valuation-10 figure of his.
* *The data hash.* The evaluator prints the recorded and the built hash for every row and does
  not stop on a difference unless `--strict-data-hash` is given; each record names the versions
  of pandas, numpy and duckdb, which the third study's records did not.
* *Environment on a reused fit.* The evaluator prints the source fit's own environment and
  marks the row as taken from the third study.
