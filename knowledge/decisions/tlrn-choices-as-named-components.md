---
type: Decision
title: tlrn design choices as named components
description: A tlrn variant is a config naming its head, attention, mask, batching, member and calibration, not a copy of the entry; the defaults reproduce the published fits bit for bit.
tags: [tlrn, design, nn, api]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:25:00Z }
sources:
  - id: pr-164
    resource: https://github.com/EKtheSage/ibnr/pull/164
    title: tlrn design choices (merged)
  - id: components
    resource: https://github.com/EKtheSage/ibnr/blob/main/src/ibnr/gallery/nn/tlrn/components.py
    title: The choices and which of them can be combined
---

# Question

Ethan: a neural network can be adjusted in nearly unlimited ways (heads, attentions,
factors or loss ratios). What should ibnr do to accommodate that?

# Decision

Do not keep adding booleans to `TLRNConfig`. Name each design choice, check the
combinations together when the config is built, and let a preset set several at once.[^components]

* `head`: `ldf` (published) or `premium_lr`.
* `attention`: the axes in order; adding `"ay"` attends across a company's accident years.
* `mask`: `unwritten_lines` or `observed_cells`.
* `batch_unit`: `example` or `company`.
* `member`: `network` or `mcl_blend`.
* `scoring`: `all_cells` or `reached_cells`.
* `calibration`: `rescore_final` or `retrain_per_valuation`.
* `keep=None` averages every trained member.

A combination that cannot work (accident-year attention on a batch of unrelated examples,
the anchored variant on the premium head) is refused at construction, naming the two
choices that disagree. `TLRNConfig.accident_year_variant()` is the companion study's
variant in one call; an override switches any single choice back to the published model.

# Why

The choices are not independent, and a flat set of switches lets a caller build every
wrong combination and learn it from a shape error deep in a forward pass. A copy of the
entry for each variant is the stale-template problem the repo has met before.

# Evidence the defaults did not move

Ten fits (the network-only, anchored, single-line, factors-only and legacy-tail variants,
each with and without the incurred features) were compared as raw bytes before and after
every refactor: weights, histories, reserves, cumulative grids and draws. All identical.
That snapshot was taken outside the repo, so it is not a committed test.

# Rejected

* **A second implementation as a JAX backend**: no faster on the CPU this runs on, see
  [JAX versus torch on a CPU](/findings/jax-versus-torch-on-cpu.md).
* **Torch batching of members**: the step is compute-bound, see [tlrn CPU training
  throughput](/findings/tlrn-cpu-training-throughput.md).
* **A registered gallery entry for the variant**: touches the docs and every
  registry-wide test; left to Ethan.

[^components]: The choices and which of them can be combined
