---
type: Measurement
title: tlrn CPU training throughput
description: What one tlrn training step costs on the 16-core Windows laptop, how that cost scales with threads and processes, and what a full accident-year-variant run takes.
tags: [tlrn, performance, cpu, torch]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:10:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: pr-164
    resource: https://github.com/EKtheSage/ibnr/pull/164
    title: tlrn design choices (the network and training loop measured here)
  - id: session
    resource: measurements taken in the 2026-10-05 working session on the dev laptop (16 logical cores, Windows 11, torch from uv.lock)
    title: Profiling runs, not kept as scripts
---

# Result

A tlrn member is **compute-bound, not overhead-bound**, and the laptop delivers about
**10 to 15 member-epochs per second in total** however the cores are split between
processes and threads. A full accident-year-variant run is 240,000 member-epochs
(20 members, 3,000 epochs, four valuation dates), so it takes roughly **4.4 to 6.7
hours**, not the 15 hours first estimated from a contended run.

# What was measured

One forward and backward step of the accident-year-variant network on one thread:[^session]

| companies per step | ms per step | ms per 8 companies |
|---|---|---|
| 8 | 26.5 | 26.5 |
| 32 | 97.9 | 24.5 |
| 64 | 217.3 | 27.2 |
| 128 | 452.2 | 28.3 |

The cost per 8 companies does not fall as the batch grows, so batching members into
one program (the idea behind the JAX speed-up, see [JAX versus torch on a
CPU](/findings/jax-versus-torch-on-cpu.md)) has nothing to amortise.

Seconds per epoch for one member, by torch threads: 0.287 (1), 0.155 (2), 0.121 (4),
0.124 (8). The published (paid-only, thirteen-feature) form is 0.080 at 4 threads.
Each fit also pays about 8 to 10 seconds of fixed cost.

Aggregate throughput, 16 members of 150 epochs each (processes x threads): 11.3
member-epochs per second at 4 x 4, 12.3 at 8 x 2, 15.1 at 16 x 1. One thread per worker
and as many workers as cores is the setting to start from.

# Where a step's time goes

One thread, op by op: random-number generation for dropout 15%, softmax 11% plus 5%
backward, matrix multiplies 11%, batched matrix multiplies 9%, copies 7%, layer-norm
backward 7%.

# What did not help

* `scaled_dot_product_attention` in place of `nn.MultiheadAttention`: 5.3 ms against
  5.8 ms for the lag-attention shape of one batch. Not worth the loss of bit-identical
  defaults.
* A cheaper dropout mask (`rand_like`) is 35% faster per tensor, about 5% of a step.
* `torch.compile` cannot be used here: no C++ compiler (`cl`, `gcc`, `clang`) is on the
  path.
* Pooling the members of every valuation date's protocol into one worker pool
  ([#164](https://github.com/EKtheSage/ibnr/pull/164)) keeps all workers busy when the
  ensemble does not divide the process count, but it did not change the aggregate
  throughput ceiling.

# Why the ceiling

Not established. Sixteen workers on one thread each are slower per worker than one
worker alone (0.29 s per epoch alone, about 1.1 s per member-epoch per worker at 16),
so something shared saturates: memory bandwidth, the mix of performance and efficiency
cores, or power limits. A hypothesis, not a measurement.

[^session]: Profiling runs, not kept as scripts
