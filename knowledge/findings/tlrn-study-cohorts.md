---
type: Finding
title: The two cohorts of the tlrn robustness study and how the benchmarks tie out
description: Cohort B (the companion study's) can be built from the cas-schedule-p mart because its cells equal the cached CAS files' cell for cell; B lies inside cohort A; ibnr's mack and mcl reproduce notebook 04 on A and the companion's chain ladder and MCL reserves on B.
tags: [tlrn, companion-study, cohort, tie-out, data]
status: draft
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-12T20:00:00Z }
stale_after: 2027-04-12T00:00:00Z
sources:
  - id: decision
    resource: /decisions/tlrn-robustness-study.md
    title: The study's questions and verdict rule
  - id: runner
    resource: scripts/tlrn_study_colab.py
    title: The cohort loaders and selection (cells_from_mart, cells_from_cas, select_pairs)
  - id: evaluator
    resource: scripts/tlrn_study_eval.py
    title: The benchmark fits and the tie-out (benchmark_reserves, tie_out)
  - id: repro
    resource: /findings/companion-notebook-reproduction.md
    title: The earlier reproduction that found 84 companies and the MCL tie-out
---

# Result

The study's second cohort does not need the companion study's four cached CAS files. The
cas-schedule-p mart (publish `20260613_041006`) holds the same cells for those lines and
years, so `scripts/tlrn_study_colab.py` builds cohort B from the package by default, and
`--cas-dir` builds it from the files as a check.[^runner]

# What was measured

* **The cells are identical.** For the four lines and accident years 1998 to 2007 both
  sources hold 62,010 rows, and a join on company, line, accident year and development age
  matches every row with a difference of exactly 0.0 in cumulative paid, incurred, bulk
  and net earned premium. The mart's case reserve equals incurred minus paid minus bulk on
  every cell, which is how the CAS-file route derives it. The two routes' cell hash is the
  same (`83b7a9dde5aa167f` for all 62,010 rows).
* **One rule, two cutoffs.** Notebook 04's selection rule and the companion study's are
  the same rule with the calendar-index cutoff at 5 and at the last known diagonal (10).
  Cutoff 5 gives 668, 414, 330 and 243 pairs after the four steps (93 companies, hash
  `08605b3c80513bca`); cutoff 10 gives 668, 414, 297 and 212 (84 companies, hash
  `de51bae56cca563c`, the companion's own audit table's counts). The stricter cutoff
  asks more of every pair, so **cohort B lies inside cohort A**: all 212 pairs and all 84
  companies are in A. The two cohorts are therefore not independent evidence.
* **All 84 companies are scored.** The cutoff-10 rule requires a positive cumulative paid
  on every known cell, so `mack` and `mcl` fit every company; cohort A scores 82 of its 93
  for the reason notebook 04 gives.
* **Company 1767 dominates both.** It holds 66.3% of cohort A's scored actual reserve and
  67.3% of cohort B's. Without it the chain ladder's company-reserve Pool_APE is 12.9% on A
  and 12.3% on B, against 5.77% and 5.48% with it, so the pooled number is mostly one
  company.

# The benchmarks tie out

| check | result |
|---|---|
| `mack` Pool_APE on A against notebook 04's reference | 0.0577322331, relative difference 6.7e-16 |
| `mcl` Pool_APE on A against notebook 04's reference | 0.0566061422, relative difference 2.3e-13 |
| `sur` on A | 80 of 82 companies, Pool_APE 5.94% (notebook 04's count) |
| `mack` reserves on B against the companion's chain ladder | 84 companies, max relative difference 2.8e-15 |
| `mcl` reserves on B against the companion's chain ladder + MCL | 84 companies, max relative difference 3.7e-13 |
| actual reserves on B against the companion's observed | identical |
| Pool_APE on B | `mack` 5.481%, `mcl` 5.410%, `sur` 5.669% |

`sur` is not compared with the companion's SUR on B. The two are different estimators with
different defaults (the largest company difference is 55%), and `sur` is not part of the
verdict.[^evaluator]

# Cost

Fitting the benchmarks takes 86 s on cohort A and 96 s on cohort B on the 16-core laptop,
almost all of it the per-company `reserve_rows` calls, once the cohort's cells are held in
an in-memory triangle. The mart's own triangle gave the same reserves (Pool_APE identical
to 1e-15) but took 539 to 609 s on cohort A, because every query re-applies a filter that
ORs the 243 pairs. The evaluator keeps the fits in a cache keyed by cohort and data hash.

On a CPU with two members and ten epochs, the runner's fits take 52 to 122 s each (the
machine was shared), almost all of it compilation and setup. Both cohorts train: the
cohort B triangle built from the CAS files was fitted and scored as well as cohort A's.

[^runner]: The cohort loaders and selection (cells_from_mart, cells_from_cas, select_pairs)
[^evaluator]: The benchmark fits and the tie-out (benchmark_reserves, tie_out)
