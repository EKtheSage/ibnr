---
type: Measurement
title: Torch thread count for small models
description: "On Ethan's 16-core laptop, tlrn (about 14k parameters) trained 3x faster at 4 torch threads than at the default 16 (2026-09-22); set the thread count in the notebook or script, never in the library."
tags: [tlrn, performance, cpu, torch, threads]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: note
    resource: agent memory note torch-threads-small-models.md (private, outside the repository)
    title: Torch threads for small models
    last_modified: 2026-09-22T08:06:33.801Z
  - id: pr-143
    resource: https://github.com/EKtheSage/ibnr/pull/143
    title: The earlier 1.1 s per epoch figure
---

# Result

Measured 2026-09-22 on the Intel Core Ultra 9 285H (16 cores, a mix of performance and
efficiency cores), torch 2.14.0+cpu, `tlrn` on the 93-company study set with 13 features
(`tests/data/tlrn_study_pairs.csv`). Marginal cost per epoch from a 20-epoch and a
60-epoch fit:[^note]

| torch threads | s per epoch | 3000-epoch seed | 10-seed fit |
|---|---|---|---|
| 16 (default) | 0.77-1.0 | about 45 min | about 8-10 h |
| 8 | about 0.5-0.6 (noisy) | | |
| 4 | about 0.25 | about 12.5 min | about 2.1 h |
| 2 | about 0.32 | | |

The 16-thread figure is what PR #143 measured ("~1.1 s/epoch") and what put one published
fit near ten hours.[^pr-143] It is oversubscription of a tiny model, not the model's cost.
Some runs shared the CPU with a background pytest, so the 8-thread row is noisy; the
4-against-16 gap (3x) held in every pairing.[^note]

# Why the setting lives outside the library

The other NN entries (`nn_transformer`, `deeptriangle`, `mdn`, `resnet`) are the same size
class, so the same setting likely helps them. The library must not set global torch
threads itself (a process-wide side effect), so the notebook or script sets it and records
`torch.get_num_threads()` in its timing table.[^note]

# What to do (as of 2026-09-22)

In notebook 04 and any NN script on this machine, call `torch.set_num_threads(4)` before
the fits. Budget tlrn at about 2-2.5 hours per 10-seed fit, not about 10 hours, and run
the three published fits one after another rather than through a process pool.[^note]

Later measurements (2026-10-05) of the accident-year variant, with workers and threads
traded against each other, are in [tlrn CPU training
throughput](/findings/tlrn-cpu-training-throughput.md); there the starting point is one
thread per worker and as many workers as cores.

[^note]: Torch threads for small models
[^pr-143]: The earlier 1.1 s per epoch figure
