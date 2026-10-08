---
type: Measurement
title: The JAX training backend matches torch
description: How closely tlrn's JAX training backend reproduces the torch path on a CPU (forward pass, gradients, optimiser, whole trajectories), the one way the two can part with neither wrong, and what is still unmeasured (the TPU).
tags: [tlrn, jax, torch, parity, testing]
status: draft
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-08T12:00:00Z }
stale_after: 2027-04-08T00:00:00Z
sources:
  - id: tests
    resource: tests/test_tlrn_jax.py (branch feat/tlrn-jax)
    title: The parity tests
  - id: session
    resource: the 2026-10-08 working session on the dev laptop (jax 0.7.1, torch 2.12.0, CPU, two threads)
    title: Measurements and mutation run
---

# Result

On a CPU, with dropout off, `fit(backend="jax")` and the torch path train the same
members to float32 rounding, when no batch puts a parameter at a true-zero gradient (see
below). Every piece and both whole-fit forms were compared against torch, and 21
deliberate breakages of the backend were each caught by the tests.[^tests]

# Numbers (largest relative difference, torch against JAX)

* **Forward pass**, torch weights loaded into JAX, on nine layouts (published, anchored,
  legacy tail, factors only, lag only, observed cells, the accident-year variant, accident
  years with unwritten lines, two layers): within the test tolerance of 2e-5 relative.
* **Loss and every parameter's gradient** on one batch: loss within 1e-5, gradients within
  1e-4 of each parameter's largest; padding the batch with zero-weight rows moves neither.
* **Six clipped AdamW steps** with two learning rates under the warmup cosine: within
  1e-5 (largest gap 4e-7 on values of order 1).
* **Fifty Adam updates**, compared as updates (the parameters reset to zero before each
  step, so a parameter's own size cannot hide an error in the step): within 1e-6
  relative. Before the bias corrections `1 - beta**t` were computed from double-precision
  logarithms, raising a float32 beta to `t` rounded 0.999 to 0.99900001, and every JAX
  update was about 6.5e-6 smaller than torch's, from the first step on.[^tests]
* **Whole fits, dropout off**:[^session]
  * published model, 3 members, 4 epochs, clipping binding, weight decay on: member
    reserves 2.4e-6, validation scores 3.2e-6, training losses 6.2e-6, same members kept;
  * accident-year variant, four valuation dates in one program (1 to 4 training cutoffs,
    a padded last batch), 2 members, 5 epochs: reserves 1.8e-7, validation 1.9e-6;
  * early stopping, 4 members stopping after 6, 8, 6 and 4 epochs in both backends:
    reserves 3.4e-6, validation 8.4e-6.

  These three were measured before the Adam fix above; after it the same trajectory
  tests pass at the same tolerances, and the figures were not re-measured. Two more
  trajectory tests, with weight decay at 0.05, cover the variants that leave parameters
  no forward pass reads (attention across lags only, and the factors alone): torch hands
  those no gradient and AdamW skips them, decay included, and the whole state dict must
  agree.
* **Dropout on** (0.3), 8 members, 30 epochs, seed 5: best validation scores 0.2506 to
  0.2636 under torch (median 0.2565) and 0.2508 to 0.2622 under JAX (median 0.2574). The
  members differ, as different dropout masks should make them, and look like the same
  procedure; eight members of a test fixture cannot say more than that.

# Where the two part with neither wrong

Adam divides each step by the gradient's own running size, so it takes a full-size step
on a gradient of any size, including rounding. Where a parameter's true gradient over a
batch is exactly zero, torch's autograd can leave rounding there and JAX leaves exactly
zero. Measured at seed 11 on the test fixture: a batch whose scored cells all start past
the first development step gave `phi[:, 0]` a torch gradient of 5e-9 and a JAX gradient
of 0, and torch then moved that parameter by about a third of the learning rate. From
there the two members follow different trajectories. The attention's key bias
(`in_proj_bias[d:2d]`) always has a zero true gradient, because the softmax cancels a
constant added to every key's score, so it drifts on rounding in both backends and
changes no output. The trajectory tests use seeds where the first does not happen and
leave the key bias out of the weight comparison.[^tests]

# Mutations caught

Exact GELU replaced by the tanh form; the empty-sequence guard removed; Adam's bias
correction removed; clipping removed; the head trained at the network's rate; the epoch's
cutoff drawn before the permutation; padded rows counted; an empty batch stepping; the
Adam count advancing on a skipped batch; patience never reset; `min_epochs` ignored; the
blend weight not kept with its checkpoint; validation ignoring the blend; dropout off in
training; the factor support ignored; the observed-count embedding off by one; the final
blend weight taken from the last check; `backend` ignored by `fit`; `processes > 1`
accepted with `backend="jax"`.[^session] Added with the review fixes: Adam's bias
corrections computed from a float32 beta (6.5e-6 per update); weight decay applied to
the parameters torch gives no gradient (up to 9e-4 relative on those weights after four
epochs).

# Not measured

The TPU. The program was compiled and run on a CPU only. On the test fixture (8 tiny
members, 30 epochs) the JAX fit took 11 s with one thread against 23 s for torch with
two, compilation included. The Colab runner's ten-epoch check on notebook 04's real cohort
(93 companies, 4 members) took 25 to 37 s per fit on one CPU thread. Neither says anything
about the full-size fit on a CPU, where the
earlier measurement found JAX slower (see [JAX versus torch on a
CPU](/findings/jax-versus-torch-on-cpu.md)). Related: [tlrn gets an optional JAX training
backend](/decisions/tlrn-jax-training-backend.md).

[^tests]: The parity tests
[^session]: Measurements and mutation run
