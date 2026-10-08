---
type: Decision
title: tlrn gets an optional JAX training backend
description: "fit(backend='jax') trains every tlrn member of every valuation date as one compiled JAX program for a TPU or GPU; training is the only seam, torch stays the default and does everything else (Ethan, 2026-10-08)."
tags: [tlrn, jax, torch, performance, colab, decision]
status: draft
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-08T12:00:00Z }
stale_after: 2027-04-08T00:00:00Z
sources:
  - id: request
    resource: Ethan's request in the 2026-10-08 working session
    title: "make a jax backend optional and run this on google colab"
  - id: module
    resource: src/ibnr/gallery/nn/tlrn/jax_backend.py (branch feat/tlrn-jax)
    title: The JAX training backend
  - id: howto
    resource: docs/tlrn-on-colab.md (branch feat/tlrn-jax)
    title: Training tlrn on a Colab TPU
---

# Decision

`TLRN.fit` takes `backend="torch"` (the default) or `backend="jax"`, an execution
choice like `processes=`. The JAX backend trains every member of every valuation date at
once as one compiled program, for a Colab TPU or a GPU. It ships in an optional extra,
`ibnr[jax]`, which carries torch as well because everything after training still runs
on torch.[^request]

# Why

On the dev laptop's CPU one 20-member accident-year fit takes about 8 hours with the
torch loop, while the companion study trained all 80 members of that variant in 1,667 s
on a Colab TPU with a JAX program of this shape. On a CPU the same JAX code is slower
than torch (see [JAX versus torch on a CPU](/findings/jax-versus-torch-on-cpu.md)), so
JAX is only worth having on an accelerator, and torch stays the default.

# How it is built

* **Training is the only seam.** The backend returns exactly what the torch training
  path returns: per valuation date, torch modules carrying each member's best-validation
  weights, and one history list per member in the same record format. Selection, the
  point, the calibration, `predict` and the attention read-outs are unchanged torch
  code.[^module]
* **ibnr's torch code is the specification.** Each member starts from the weights torch's
  `make_model` gives it under the member's seed, and draws its batch order and training
  cutoffs from the same numpy generator in the same order. Only the dropout masks come
  from JAX's own random numbers.
* **Different valuation dates in one program** by padding: the cutoff axis is padded to
  the longest date's count (a member never reads a padded set), and a short last batch is
  padded with zero-weight copies of one unit, which add nothing to any loss term.
* **Refusals by name**: `processes > 1` (the program runs in one process, so the argument
  would do nothing) and `cutoff_sampling="per_example"`.
* **Fits survive a disconnect** by saving each fitted entry as soon as it is done
  (`scripts/tlrn_colab.py`); a fitted entry pickles and reloads with the same
  predictions. A saved entry carries a record of its settings and is reused only by a
  run with the same settings, so a resumed run cannot pass off an old fit of another
  budget as its own.[^howto]

# Not yet known

The TPU speed. Everything was built and checked on a CPU; the first Colab run decides
whether the program comes near the companion study's 1,667 s. The parity evidence is in
[the JAX backend matches torch](/findings/tlrn-jax-backend-parity.md).

[^request]: "make a jax backend optional and run this on google colab"
[^module]: The JAX training backend
[^howto]: Training tlrn on a Colab TPU
