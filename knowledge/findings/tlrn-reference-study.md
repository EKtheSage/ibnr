---
type: Finding
title: The tlrn reference study and ibnr's port of it
description: What the reference transformer (TLRN) is and scores, how ibnr's 0.7.0 port and its multivariate chain ladder were checked against it, and where the note and the newer bundle concepts describe the reference differently.
tags: [tlrn, mcl, companion-study, reproduction, tie-out]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries dated 2026-09-20 and 2026-09-21 (lines 203-261)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: spec
    resource: docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md
    title: The approved design for point scores, the training scheme, mcl and tlrn
  - id: pr-141
    resource: https://github.com/EKtheSage/ibnr/pull/141
    title: The mcl entry and its tie-out
  - id: pr-143
    resource: https://github.com/EKtheSage/ibnr/pull/143
    title: The tlrn entry
---

# The reference

A transformer for loss reserving, TLRN, written by a colleague of Ethan's for a study
that was not yet published on 2026-09-20 (see [tlrn entry scope](/decisions/tlrn-entry-scope.md)
for why ibnr does not name it). As the note describes it (2026-09-20):[^note]

* Examples are company by accident year, made of line-by-lag tokens, with axial attention.
* A softplus development-factor head.
* Point loss: accident-year and line APE, plus 0.5 times PE, plus 0.1 times MSE divided by
  0.005.
* 10 seeds of 3,000 epochs; the best 2 by validation accident-year and line APE are kept.
* The reserve is blended as 0.342 times MCL plus 0.658 times the raw network.
* Uncertainty comes from size-stratified historical residual draws.

On 82 companies (accident years 1998 to 2007, valuation 2007-12-31; the study's selection
rule with cutoff 5 gives 93 companies and 243 company-line pairs), it scored a
company-reserve Pool_APE of **5.088%** against 5.661% for chain ladder blended with MCL.
ibnr 0.5.5's `nn_transformer` scored 60.3% and 84.1% on the same set. The 11 companies
missing from the 82 failed the reference's Mack fit on a zero or negative other-liability
cell; its MCL was finite on all 93.[^note]

# How ibnr's port was checked

* **Cohort.** ibnr reproduces the study's cohort selection exactly: 93 companies and 243
  pairs.[^note]
* **Network.** Reviewed module by module (head, network, features, model) as faithful
  transcriptions of the reference code, with departures documented.[^note] 14,309
  parameters at `d_model` 32 with 13 features. The embedding tables carry one unused row.
  This is not "row 0 as in the reference": the reference's torch is 1-based, so its unused
  row is the last one.[^note]
* **MCL.** ibnr's `mcl` ties out to the reference's replay on all 82 companies to 1.3e-12
  relative, once two conventions were found:[^pr-141]
  1. The reference's `solve()` refuses a matrix whose reciprocal condition number is below
     machine epsilon, where numpy inverts it. This affects 6 companies.
  2. The positivity check applies only to cells a transition divides by. 2 companies
     inside the 82 have a zero 12-month paid on accident year 2007.
  The reference tables used are vendored as `tests/data/tlrn_study_*.csv`, with neutral
  names.

How each `tlrn` result compares with the reference is in [notebook 04
results](/findings/notebook-04-results.md), [which tlrn members are
kept](/findings/tlrn-member-choice.md) and [ibnr's tlrn matches the R
reference](/findings/tlrn-port-matches-r-reference.md).

# Where this note and the newer concepts differ

* **Language of the reference.** This note (2026-09-20 to 2026-09-23), like [ibnr's tlrn
  matches the R reference](/findings/tlrn-port-matches-r-reference.md), calls the
  reference an R implementation running R torch, with R checkpoints. [The companion study's
  repository](/references/companion-study-repository.md) and [reproducing the companion
  notebook](/findings/companion-notebook-reproduction.md) (2026-10-05) describe a Python and
  JAX notebook. Whether these are two implementations of the study or one was not
  established in either source.
* **Cohort size.** This note's cohort is 93 companies and 243 pairs, 82 of them scored.
  The 2026-10-05 reproduction concept counts 84 companies and 212 company-line pairs for
  the JAX notebook's selection rule. The two may be different selections; neither source
  reconciles them.
* **Embedding rows.** Both agree that the reference's embedding tables and ibnr's are
  offset by one row. This note says the reference is 1-based; the 2026-10-05 concept says
  the JAX notebook is 0-based and ibnr 1-based.

[^note]: Project status log, entries dated 2026-09-20 and 2026-09-21 (lines 203-261)
[^pr-141]: The mcl entry and its tie-out
