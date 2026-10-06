---
type: Gotcha
title: The Meyers a_ig bound is load-bearing
description: "Stan's (0, 1e5) bound on a_ig in the Meyers CCL and CSR models is not cosmetic; NumPyro and PyMC ports without it biased sig 5-12% low, and restoring it needs an explicit a_ig init."
tags: [bayesian, parity, meyers, numpyro, pymc, stan]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note meyers-a-ig-bound.md (private, outside the repository)
    title: Meyers a_ig bound
    last_modified: 2026-07-24T16:42:17.399Z
---

# The trap

`model.stan` for both CCL and CSR declares `vector<lower=0, upper=100000>[n_d] a_ig`. The
NumPyro and PyMC ports originally used an **unbounded** `InverseGamma(1,1)`, documented in
both model cards as a harmless deviation because the truncated **prior** mass is about
1e-5.[^note]

That reasoning was wrong (found 2026-07-24 while porting CSR). The *posterior*
concentrates in exactly the truncated corner at deep development lags, where the data
barely constrain sigma. Measured on workers' compensation company 11347, `P(a_ig > 1e5)`
by development index: 0.1% at d=0, rising monotonically to **10.6% at d=8**. That is the
`a -> 0` region, so unbounded ports put mass at smaller variance increments than Stan
can, dragging `sig` **5-12% below** the Stan reference (worst at the deepest lags) and
failing parity at z_mean 6.8-8.1. `gamma` and `logelr` were unaffected (z about 0.1-0.9),
which is what located the problem.[^note]

# The fix: reproduce Stan's constraint

* **NumPyro**: `dist.ImproperUniform(constraints.interval(0, 1e5))` plus `numpyro.factor`
  carrying the `InverseGamma(1,1)` density. This is literally Stan's construction: the
  constraint fixes the transform and Jacobian, and the factor supplies the unnormalized
  density. NumPyro's `TruncatedDistribution` does not accept an InverseGamma base.
* **PyMC**: `pm.Truncated("a_ig", pm.InverseGamma.dist(1,1), upper=1e5)`.[^note]

**Both need an explicit init at `a_ig = 1e3`** (`init_to_value` / `initval`). The (0, 1e5)
interval transform maps the default unconstrained uniform(-2, 2) init onto `a_ig` in
(1.2e4, 8.8e4), hard against the upper bound. Stan starts there happily; the ports do
not. Without the init, NumPyro gets max R-hat 1.59 / ESS 7, and PyMC dies outright
(`_init_jitter` steps past the bound, giving `logp = -inf` and a `SamplingError` before
warmup). This is a documented init deviation; only that one site's warmup path
changes.[^note]

# After the fix

`sig` sits about 1.6-2% **above** Stan (z about 1.1-1.9) instead of 5-12% below.[^note]

`a_ig`'s own R-hat and ESS look terrible in every backend, **including at convergence**:
the individual `a_ig[j]` are weakly identified because the likelihood only sees their
cumulative sums through `sig2`. Judge these models on `sig`, not on `a_ig`.[^note]

# Rejected alternative

Kept as the documented fallback if the init deviation ever becomes unacceptable: sample
`a ~ Uniform(1 - exp(-1e-5), 1)` directly. That is algebraically the identical model,
with no init trick and 0 divergences, but a different transform from Stan's, which weakens
the "same parameterization" basis of the convergence comparison (design decision 7 in
CLAUDE.md).[^note]

# How it was found

Only because [the parity check rounding bug](/gotchas/parity-check-rounding.md) was fixed
first: the rounding was scoring these parameters z = 0.[^note]

[^note]: Meyers a_ig bound
