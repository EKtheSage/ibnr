---
type: Measurement
title: JAX versus torch on a CPU
description: The companion notebook's JAX training code, run unchanged on this CPU, against ibnr's torch loop; the TPU speed does not transfer.
tags: [tlrn, performance, jax, torch, cpu]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:15:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: notebook
    resource: Code/tlrn_pipeline_best.ipynb in the companion study's repository (sibling checkout)
    title: The companion notebook, its recorded run and its manifest
  - id: session
    resource: the 2026-10-05 working session on the dev laptop (jax 0.11.2, cpu backend)
    title: JAX spike run
---

# Result

The companion notebook is **Python with JAX, not R**, and its 1,667-second training
time is a TPU result. On this 16-core CPU the same code ran at about **1.5
member-epochs per second**, against about **6.4** for ibnr's torch loop with four worker
processes. A JAX backend for ibnr would not make CPU runs faster, so none was built.

# The notebook's run

Colab TPU, JAX 0.7.2 (`backend: tpu` in its run manifest). All 80 members (20 at each
of four valuation dates) train in one compiled program: `jit` over `vmap` over
members, with the epoch loop and the minibatch loop inside `lax.scan`. 3,000 epochs in
1,667 s.[^notebook]

# The same code on this CPU

The notebook's cells run unchanged except that the cohort file paths pointed at the
cached CAS files and the members and epochs were cut to 8 and 600 (32 member jobs).
After 100 epochs, 2,115 s had elapsed, including compilation, and the run was stopped.
Extrapolated, 600 epochs would have taken about 3.5 hours.[^session] ibnr's torch loop
did the same 19,200 member-epochs in 2,981 s.

# Why

On a TPU the win comes from running many small networks at once. On this CPU a step is
already compute-bound (see [tlrn CPU training
throughput](/findings/tlrn-cpu-training-throughput.md)), so one large batched program
loses to simple process-level parallelism.

[^notebook]: The companion notebook, its recorded run and its manifest
[^session]: JAX spike run
