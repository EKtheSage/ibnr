---
type: Measurement
title: PyMC speed on the Windows laptop
description: How Stan, NumPyro and PyMC compared on meyers_ccl in July 2026, why PyMC is slow here, and which ways of speeding it up were tried and failed on this Windows machine.
tags: [pymc, numpyro, stan, performance, windows, milestone-4]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, milestone-4 entries of 2026-07-08 to 2026-07-20 (lines 594-687) and 2026-07-25 (lines 1127-1132)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: card
    resource: src/ibnr/gallery/bayesian/meyers_ccl/card.md
    title: The meyers_ccl card's "Accelerating PyMC" section
  - id: pr-23
    resource: https://github.com/EKtheSage/ibnr/pull/23
    title: nuts_sampler reachable through every Bayesian entry
---

# Result (2026-07-08)

On `meyers_ccl`, with the cache warm, chains run one after another on a single core, all
three backends reach the same posterior at R-hat at most 1.02 and ESS about 400 to 600.
Wall clock: **Stan about 7 s, NumPyro about 15 to 24 s (about 2.5 times), PyMC about 41
to 59 s (about 7 times)**. Stan ran with `parallel_chains=1` for a fair comparison.[^note]

The parity figures measured in the same session (4 x 500 draws) came through a summary
that rounded to three decimals; `CLAUDE.md` records them as superseded (see [the parity
check rounded its z-scores away](/gotchas/parity-check-rounding.md)).

# Why PyMC is slow here

The gap is the backend, not the model: PyTensor's C backend runs many small operations
per leapfrog step, where JAX/XLA fuses the whole graph. Missing BLAS is only a minor
factor (the main matrix product is 55 x 55). The first PyTensor C compile takes about 4 to
5 minutes on Windows with RTools and is then cached by graph shape; it is excluded from
the times above.[^note]

A closed form helped: the Stan `prev_idx` recurrence equals `mu = P(rho) @ B`
(`kernels.contract.ccl_mu_index`), an N x N matrix product. Unrolling the recurrence as an
N-deep scalar chain makes PyTensor's C compile explode (about 4 minutes). Never
reintroduce the scalar unroll in the ports.[^note]

# Tried and failed on this machine (2026-07-08)

* **BLAS through pip.** `scipy-openblas64` is built with MSVC and will not link against
  the RTools MinGW g++ that PyTensor uses, so PyTensor rejects the flag. Reinstalling
  pytensor with uv or pip gives the same wheel without BLAS.[^note]
* **nutpie.** Installs only by forcing numpy 2 (do not put it in the shared `.venv`; use a
  separate one). Sampling hung: 1 chain of 300 draws timed out after more than 8 minutes,
  where NumPyro does the same model in about 27 s. A numba or nutpie problem on Windows
  with this graph. That separate environment also carried pytensor 3.1.2 and numpy 2.5,
  where native PyMC on company 11347 also went pathological, so the versions are a
  confound.[^note]
* **conda BLAS through pixi** (pixi 0.67.1 installed). conda-forge resolves the coherent
  stack pip cannot (MKL 2026, gcc 15.2, m2w64-sysroot), but the environment would not
  finish installing: Windows Defender locks freshly extracted packages, giving "Access is
  denied (os error 5)" on the rename, on a different package each retry, even with a clean
  cache and a short path (`C:\pxb`, needed because the long scratch path also exceeded
  Windows' path limit). Needs a Defender exclusion or admin rights. pixi cannot install
  into uv's `.venv`.[^note]

The plan written down: retry on a Linux container, where conda-forge PyTensor links MKL
by default and the nutpie hang should not apply, and record the numbers in the card's
"Accelerating PyMC" section.[^card]

# What does work

`model_pymc.sample` takes `nuts_sampler=` ("pymc" by default, or "nutpie", "numpyro",
"blackjax"), and since PR #23 (2026-07-25) every Bayesian entry's `fit()` passes it
through. The default stays "pymc" on purpose: a faster foreign default would turn the
cross-backend tables into NumPyro measured against NumPyro. For compartmental, the
NumPyro sampler on the PyMC graph took 34.5 s against more than 2,000 s with native
PyTensor.[^pr-23]

# target_accept and the centered parameterization

At `target_accept` 0.8, PyMC under-adapted company 11347 badly: R-hat 1.17, 182
divergences, ESS 18, and slow, since divergent trees are long. Stan and NumPyro tolerate
0.8 (0 of 38 divergences). 0.9 with 1,000 warmup draws took PyMC from 182 to 11
divergences. Use 0.9 or more for hard companies. Why PyMC alone fails at 0.8 is explained
only as under-adaptation of the centered parameterization (still open on
2026-07-20).[^note]

[^note]: Project status log, milestone-4 entries of 2026-07-08 to 2026-07-20 (lines 594-687) and 2026-07-25 (lines 1127-1132)
[^card]: The meyers_ccl card's "Accelerating PyMC" section
[^pr-23]: nuts_sampler reachable through every Bayesian entry
