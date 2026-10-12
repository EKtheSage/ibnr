---
type: Finding
title: Notebook 04 results on 0.7.0, 0.7.1 and 0.7.3
description: The company-reserve scores of analysis/04 in its four published runs (2026-09-22, 2026-09-24, 2026-09-25, 2026-10-08), with the reference numbers it is checked against and what the breakdowns showed.
tags: [notebook, tlrn, mcl, results, companion-study]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries dated 2026-09-22 (lines 143-201), 2026-09-24 (lines 43-81) and 2026-09-25 (lines 11-41)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: nb04
    resource: analysis/04_nn_architectures_vs_classical.ipynb
    title: The notebook and its saved outputs
  - id: pr-146
    resource: https://github.com/EKtheSage/ibnr/pull/146
    title: Notebook 04 first published run (0.7.0)
  - id: pr-150
    resource: https://github.com/EKtheSage/ibnr/pull/150
    title: Notebook 04 on 0.7.1, forty tlrn members, and the board broken down
  - id: run-073
    resource: analysis/04_nn_architectures_vs_classical.ipynb
    title: The saved outputs of the 0.7.3 run (company board and run manifest of 2026-10-08)
  - id: pr-145
    resource: https://github.com/EKtheSage/ibnr/pull/145
    title: reserve_rows fixes found by the notebook's first smoke run
---

# What the notebook does

`analysis/04_nn_architectures_vs_classical.ipynb` scores ibnr's methods on the companion
study's own cohort and protocol: accident years 1998 to 2007, valuation 2007-12-31,
**82 companies** scored. The headline number is the company-reserve **Pool_APE** (an error score
on each company's reserve; lower is better; the note does not spell out its formula).[^note] How the
cohort was selected and how ibnr's `tlrn` compares with the reference implementation is in
[the tlrn reference study](/findings/tlrn-reference-study.md); how long the notebook takes
and how to run it is in [re-run notebook 04](/playbooks/rerun-notebook-04.md).

Reference constants the notebook checks against (company Pool_APE, in percent): chain
ladder 5.7732, chain ladder blended with the multivariate chain ladder (MCL) 5.6606, raw
TLRN 5.6256, blended TLRN 5.0880.[^note]

Naming used below: `tlrn_13` is the fit with 13 features. The note calls the other fit
"paid only" and lists it as `tlrn_8`; that `tlrn_8` is the paid-only fit is read from how
the note pairs them, not stated outright. A `_blend` row mixes the network's reserve with
MCL; a `_keep2` row keeps the best two members by validation score, as the reference does.

# First run, 0.7.0 (finished 2026-09-22)

599 minutes of wall clock, 41 of 41 code cells, no error. Ten members per tlrn fit, best
two kept.[^pr-146]

| row | company Pool_APE |
|---|---|
| tlrn_8_blend | 5.26% |
| mcl | 5.66% (equals the reference to 7.5e-13) |
| tlrn_13_blend | 5.77% |
| mack | 5.77% (equals the reference to 7.1e-15) |
| tlrn_13_early_blend | 5.88% |
| sur (80 companies) | 5.94% |
| raw tlrn rows | 6.51%, 6.77%, 6.91% |
| pooled neural entries | 19.3% to 84.1% |

`tlrn_13` missed the reference: raw 6.77% against 5.63%, blend 5.77% against 5.09%. Its
blend was 1.9% worse than MCL, where the reference's blend was 10.1% better. The paid-only
fit beat the 13-feature fit. One company, 1767, is 70.4% of the premium.[^note]

The first smoke run of this notebook found three defects in `reserve_rows`, fixed before
0.7.0 shipped: the native route never called `predict()`, anchor and premium had to be
summed over the fitted origins only, and `sur` needed a `point()`. Reading `sur`'s native
point moved its Pool_APE from 6.0212% to 5.9382%; mack and MCL did not move.[^pr-145]

# Second run, 0.7.1, forty members (finished 2026-09-24)

614 minutes, 44 of 44 cells, no error. Each tlrn variant trained 40 members and averaged
all of them; the `_keep2` rows keep the best two of members 0 to 9.[^note]

* `tlrn_13_blend`, the average of 40: **5.16%, the best row**, 8.8% better than MCL's
  5.66%. The reference's keep-two blend is 5.09%.
* `tlrn_13` raw average of 40: 5.76%, about level with chain ladder's 5.77%.
* `tlrn_8` average of 40: 7.14% raw, 5.64% blended.
* The `_keep2` rows reproduce the 0.7.0 run exactly, because members 0 to 9 are
  identical between the two runs.

So 0.7.0's "paid only beats 13 features" came from which two members were kept, not from
the features. See [which tlrn members are kept](/findings/tlrn-member-choice.md).

# Third run, the board broken down (finished 2026-09-25)

618 minutes, 52 of 52 code cells, no error. The board and all 1,513 company reserve rows
are bit-identical to the 2026-09-24 run.[^pr-150] This round added the reference study's
seven metrics (SMAPE through a helper in the notebook, because ibnr 0.7.1 has none), and
broke the board down by line, accident year, premium quartile and company, over six rows
(`tlrn_13_blend`, `mcl`, `tlrn_13`, `tlrn_8_blend`, `tlrn_13_keep2_blend`, `mack`). `sur`
was left out because it covers 80 of the 82 companies.[^note]

Tie-outs the notebook asserts: mack and MCL on all seven metrics against the reference's
`M_T1` table (4.7e-13 in the smoke run, 5e-13 measured on 2026-09-24), by line against
`M_T8` (7.6e-13), MCL by accident year against `M_T10` (4.0e-13), and the pieces adding up
to the total (9.8e-16). 85 company-line-accident-year units have a negative actual
reserve.[^note]

What the breakdown showed for `tlrn_13_blend` against MCL:[^note]

* Never worse than third of six on any of the seven metrics. First on Pool_APE, MAE and
  SMAPE; MCL is first on none.
* Better than mack on all four lines, and better than MCL on three (not private passenger
  auto: 3.64 against 3.37).
* Ahead of MCL in 8 of 9 accident years and in 31 of 36 line-year cells. The 5 losing
  cells hold 3.5% of the money.
* The lead comes from the 61 smaller companies. In the top premium quartile (95% of the
  money) MCL is slightly ahead: 937,858 against 959,808 in absolute error.
* Company by company it is an even split, 41 against 41.

# Fourth run, 0.7.3, with the accident-year variant (finished 2026-10-08)

1,438 minutes (started 2026-10-07 19:05, finished 2026-10-08 19:02), 53 of 53 code cells,
no error, in a fresh environment with `ibnr` 0.7.3 from PyPI, `cas-schedule-p` 2026.6.13
and gold publish `20260613_041006`, protocol `published`. Forty members per tlrn fit. New
against 0.7.1: a third fit, `tlrn_ay`, the accident-year variant of the 13-feature
network.[^run-073]

Company Pool_APE over the 82 scored companies, lowest first. Rows marked ref are the
reference study's constants, not ibnr fits:[^run-073]

| row | Pool_APE | Pool_PE |
|---|---|---|
| tlrn_13_blend | 5.16% | +1.95% |
| ref: blended transformer | 5.09% | |
| tlrn_8_keep2_blend | 5.26% | +2.29% |
| ref: raw transformer | 5.63% | |
| tlrn_8_blend | 5.64% | +3.20% |
| mcl | 5.66% | -0.19% |
| ref: chain ladder + MCL | 5.66% | |
| tlrn_13 | 5.76% | +3.05% |
| tlrn_13_keep2_blend | 5.77% | +1.30% |
| mack | 5.77% | +1.03% |
| ref: chain ladder | 5.77% | |
| sur (80 companies) | 5.94% | +1.26% |
| **tlrn_ay** | **6.22%** | -1.46% |
| tlrn_8_keep2 | 6.51% | +3.58% |
| tlrn_13_keep2 | 6.77% | +2.08% |
| tlrn_8 | 7.14% | +4.95% |
| nn_transformer_ml | 19.3% | +14.4% |
| deeptriangle | 35.3% | +32.1% |
| resnet | 40.3% | +37.1% |
| nn_transformer | 60.3% | +56.1% |
| mdn | 61.1% | +58.9% |
| nn_transformer_pic | 84.1% | +81.7% |

* `tlrn_13_blend` (0.0516) is the best ibnr row, 8.8% below MCL (0.0566) and 10.6% below
  mack (0.0577), and 0.7 tenths of a percentage point above the reference's blended
  transformer (0.0509). It equals the 0.7.1 run's value, as do the `tlrn_8` and `tlrn_13`
  rows.
* The accident-year variant at full budget, `tlrn_ay` (0.0622), is **behind the chain
  ladder** on this board: 7.8% above mack, 9.9% above MCL, and above the reference's raw
  transformer (0.0563). Its bias is the only negative one among the tlrn rows (-1.46%,
  against +3.05% for `tlrn_13`). It is ahead of the keep-two rows and of `tlrn_8` raw
  (0.0651 to 0.0714) and far ahead of the pooled neural entries, and behind `sur`.
* Nothing here was tested statistically. The differences between rows near 5.2% to 5.8%
  are a few tenths of a point on 82 companies, one of which (1767) holds 70.4% of the
  premium, and the run is one random draw of the training (fit seed 11, draw seed
  7).[^run-073]
* The notebook asserts that its `mack` and `mcl` rows equal the reference company
  reserves, and the saved board agrees with the reference constants (0.057732 and
  0.056606).

# How the runs relate

The 0.7.0 outputs were reproduced by a later run to 3e-8, apart from the `tlrn_13_early`
rows, which the 0.7.1 notebook dropped.[^note]

# Disagreement on the tlrn fit times

The note records the 2026-09-24 run's tlrn fits as 224 and 233 minutes (four processes of
four threads). [Re-run notebook 04](/playbooks/rerun-notebook-04.md) gives 12,111 s and
13,224 s (about 202 and 220 minutes) for `tlrn_8` and `tlrn_13`, read from the saved
outputs of the 618-minute run of 2026-09-25. The two are different runs; the note does not
say which fit each of its two numbers belongs to.

[^note]: Project status log, entries dated 2026-09-22 (lines 143-201), 2026-09-24 (lines 43-81) and 2026-09-25 (lines 11-41)
[^pr-146]: Notebook 04 first published run (0.7.0)
[^pr-150]: Notebook 04 on 0.7.1, forty tlrn members, and the board broken down
[^pr-145]: reserve_rows fixes found by the notebook's first smoke run
[^run-073]: The saved outputs of the 0.7.3 run (company board and run manifest of 2026-10-08)
