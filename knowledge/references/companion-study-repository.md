---
type: Reference
title: The companion study's repository
description: Where the companion study's notebooks, cached CAS data and published tables live, and which notebook is the best one.
tags: [companion-study, tlrn, data]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:50:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: repo
    resource: the companion study's repository (sibling checkout, Python and JAX)
    title: Local checkout next to this repository
---

# What is there

A sibling checkout of the companion study's repository, next to this repository.[^repo]

* `Code/tlrn_pipeline_best.ipynb` is the best notebook: the accident-year variant with
  the multivariate chain ladder blend and retrained calibration. `tlrn_pipeline_final.ipynb`
  is the earlier version. `04_nn_architectures_vs_classical.ipynb` and a reconciliation notebook
  are comparison notebooks.
* `Exhibits/data/*.csv`: the four cached CAS Schedule P files the notebook downloads,
  which can be read from disk instead (`ppauto`, `wkcomp`, `comauto`, `othliab`).
* `Exhibits/tab/*.csv`: every published table, among them `M_T0_company_reserves.csv`
  (per-company reserves of each method and the observed value), `R_T1_rolling_origin.csv`
  and `T_T1_members.csv` (each member's alpha and best epoch).
* `Exhibits/run_manifest.json`: the run's settings, the data hashes and the 1,667-second
  training time on a Colab TPU.

# What it is

Python with JAX and `optax`, not R. See [reproducing the companion
notebook](/findings/companion-notebook-reproduction.md) for how ibnr's variant compares with it.

[^repo]: Local checkout next to this repository
