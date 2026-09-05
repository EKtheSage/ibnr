# clark - Clark (2003) growth-curve MLE

**Family:** statistical · **Reference:** Clark, "LDF Curve-Fitting and
Stochastic Reserving: A Maximum Likelihood Approach", *CAS Forum* (Fall
2003). chainladder-python's `ClarkLDF` is the tieout reference for the MLE;
Clark's paper is ground truth for the variance decomposition.

Smooths the development pattern with a two-parameter growth curve instead of
free per-lag factors, giving a parsimonious likelihood-based reserve with
Clark's process/parameter variance split. The Bayesian twin
(`bayesian/clark_growth_curve`) shares this parameterization.

## Model

Incremental losses are over-dispersed Poisson around a growth-curve share of
an origin-level ultimate:

```
E[X[w,d]]   = U[w] * (G(x_hi[d]) - G(x_lo[d]))
Var[X[w,d]] = phi * E[X[w,d]]
```

Growth curves (`growth_curve=`):

| curve | G(x \| omega, theta) |
|---|---|
| loglogistic (default) | x^omega / (x^omega + theta^omega) |
| weibull | 1 - exp(-(x/theta)^omega) |

**Ages** are measured from the origin period's average accident date
(uniform-writing assumption): a cell at dev index `d` (annual grain) spans
`[max(12(d-1) - 6, 0), 12d - 6]` months. This is Clark's convention and -
verified empirically - exactly what chainladder's `ClarkLDF` does; plain
end-of-period ages give a visibly different curve (omega 2.04 vs 1.44 on
genins).

Methods (`method=`):

- **`cape_cod`** (default, Clark's recommendation): `U[w] = ELR * premium[w]`
  with one profiled ELR - 3 parameters total.
- **`ldf`**: free `U[w]` per origin (profile MLE
  `U[w] = paid_to_date[w] / G(age[w])`), so the point ultimate is the
  truncated-LDF answer `paid * G(x_max)/G(age[w])` - n_w + 2 parameters.

**`premium_field` is a `cape_cod` requirement, not an entry requirement.** The
`ldf` method never reads exposure, so its contract is built without premium: it
fits a premium-free triangle with no extra arguments, and on a triangle that
has premium it ignores the column, giving the same answer as the older
`premium_field=None` spelling. An `ldf` fit therefore carries no `premium` in
its contract and `predict()` reports NaN exposure in its targets. `cape_cod`
demands the field up front and names itself in the error.

The curve MLE is a 2-D Nelder-Mead over `(log omega, log theta)` with the
level parameters profiled out in closed form (Poisson MLE given the curve).

**Convergence is scale-aware, and has to be.** The objective is a Poisson
deviance over loss *amounts*, so its magnitude is the data's - about -4.3e8 on
genins, more on a real Schedule P cohort - and one ULP there is ~6e-8. An
*absolute* `fatol` below that gap is satisfiable only when every simplex vertex
evaluates bit-identically, which made termination a lottery on floating-point
rounding: the CI leg that installs torch, jax and pymc together shifts BLAS/OMP
thread counts, hence the summation order inside the objective, and stalled a
weibull fit that had converged in 90 of 2000 iterations into a spurious
`RuntimeError`. So `fatol = 1e-12 * |f(x0)|` (~7e3 ULPs, reachable) while
`xatol = 1e-8` on the log-parameters stays absolute and is the criterion that
actually pins the fit - those are O(1) in any currency, and 1e-8 in log space
is 1e-8 relative on omega and theta. Rescaling or shifting the objective is
*not* an alternative: the resolution floor comes from summing 1e8-magnitude
terms, so subtracting a constant afterwards moves the value without recovering
a single bit.

**`res.success` is honored, with a fallback for `maxiter` only.** With a
reachable `fatol` it is reachable again, and scipy's own pair of criteria is
strictly tighter than anything below, so a successful result is simply accepted.
The fallback exists for the stall above - Nelder-Mead reporting `maxiter`
because the function criterion stayed unmet for numerical reasons while the fit
had long since arrived - and accepting one of those takes **both** halves of the
convergence claim:

- the simplex collapsed in *coordinates*, spread <= 1e-6 in log space (1e-6
  relative on omega/theta, three orders tighter than the 2e-3 the tieout asks) -
  the parameters have stopped moving; **and**
- the simplex collapsed in *objective value*, scale-relatively: spread <= 1e3 x
  the `fatol` this objective's magnitude warrants, i.e. 1e-9 relative - the
  vertices agree about what they found.

Coordinates alone is not convergence, and that is the sharp edge here. The
shrink steps can contract the simplex to a point against the infeasibility
plateau, where the objective is flat because every move is *rejected*; the
optimizer is giving up, not finishing. So every vertex of the final simplex is
also required to be finite and off the rejection sentinel - checked whatever
`res.success` says, since a simplex can collapse *inside* the rejection region
and report success. Reading `res.fun` is not enough for that: it is the *best*
vertex, so a simplex straddling the feasibility boundary (one healthy vertex,
the rest parked on the sentinel) looks perfectly fine through it. The
optimizer's report (`params_["optimizer"]`: objective, tolerances, iterations,
final spread) is kept rather than discarded, so a stalling cohort in a
200-company retrospective is visible instead of merely loud.

## Uncertainty (Clark's decomposition, simulated)

- `phi`: Pearson chi-square / (n - p), Clark's scale estimate.
- **Parameter risk**: MVN draws on the log-parameters with covariance
  `phi * inverse observed Poisson information` (numerical Hessian at the
  MLE) - the quasi-likelihood delta method of the paper, in log space so
  levels stay positive.
- **Process risk**: scaled-Poisson ODP draws `phi * Poisson(mu/phi)` per
  future cell, the same process distribution as the `england_verrall_odp`
  entry and the bootstrap ODP baselines. Every one of those draws goes
  through the shared `kernels.densities.odp_draw`, which returns the mean
  exactly when the dispersion has collapsed so far that `mu / phi` is past
  the largest Poisson rate numpy can draw at - reachable here, because a
  triangle sitting exactly on the fitted curve drives `phi` to about 1e-29.

**Truncation:** increments are projected only to the triangle's final age
(`n_d`) - no tail beyond the curve's support in the data. The backtest
scores `C[w, n_d]`, and chainladder's `ClarkLDF` ultimate truncates
identically (its fully-developed origin gets ultimate = latest). Extending
`G` to infinity is a deliberate non-goal here; Clark's own truncation
discussion applies.

## Held-out evaluation: CRPS only, never ELPD

This entry has no MCMC posterior but it does have genuine draws, so it sits
on the milestone-6 CRPS board: `scorer.py` samples the log parameters from
the asymptotic MVN above (the entry's stand-in for a posterior, one sample
per draw), then draws each next-diagonal increment as
`X = phi * Poisson(mu / phi)` - the identical two-stage recipe `predict()`
runs for the ultimates, factored into one place so the two cannot drift.
Both `ldf` and `cape_cod` level recoveries are supported, exactly as
`predict()` defines them. The entry declares
`heldout_draw_scale = "incremental"`, so `PredictsHeldout.predict_at` adds
each cell's training-diagonal cumulative before scoring; the draw count is
the instance attribute `n_heldout_draws` (default 10,000, matching
`predict()`).

The ELPD column is permanently empty, on principle: the quasi-likelihood
this entry maximizes is not a normalized density on any scale -
`exp(odp_lpdf)/phi` integrates to 0.69 at `mu/phi = 0.5` and the defect
varies with `mu/phi` (`kernels/densities.py`, the odp-not-a-density note).
An ELPD would require declaring a real predictive distribution (negative
binomial, Tweedie) - a modelling decision, not a units conversion.

## Data contract

`kernels.contract.odp_stan_data` - incremental cells, `paid_to_date` /
`latest_d` anchors, and premium by origin *for `cape_cod` only* (see the
methods above). Negative increments are rejected (same ODP limitation as the
bootstrap; failures are recorded, not patched). Zero increments are fine. The
`ldf` method additionally requires positive paid-to-date per origin;
`cape_cod` does not.

## Validation

- Tieout (`tests/test_clark.py`, fast): omega/theta, ultimates, ELR, and
  scale match chainladder's `ClarkLDF` on genins for both growth curves.
  The same file pins convergence itself - that the requested `fatol` is
  reachable given where the objective sits on the float64 grid, and that the
  fit lands far short of the iteration cap. Asserting only that `fit()`
  returns passed throughout the bug above.
- Retrospective Meyers protocol on paid: `scripts/meyers_validation.py
  --model clark`. Results in `analysis/results/clark_validation.csv`.
  No published Meyers-monograph bar exists for Clark; the comparison set is
  the paid panel (england_verrall_odp, meyers_csr) on identical cohorts.

**Result (2026-07-20, cape_cod, 95/200 companies completed - 105 rejected
for negative paid increments, PPA worst): fails uniformity catastrophically.
Combined KS D = 61.3* vs crit 14.0 with outcome percentiles piled at ~0 -
systematic paid over-prediction.** Two stacked causes: (1) the post-1997
settlement speedup that sinks every no-speedup paid model in this window
(Meyers' bootstrap ODP: D = 24.1*; our Bayesian ODP panel: same story);
(2) the loglogistic tail - G still holds several percent of ultimate beyond
the ages where short-tail books have finished paying, so even books the ODP
scores mid-range get dragged to percentile ~0. The **weibull arm** isolates
cause 2: D = 49.6* (`clark_validation_weibull.csv`,
`--growth-curve weibull`), OL passes (25.8 < 27.8), CA improves 54→39 -
materially better, still failing on the regime. Use weibull for short-tail
lines; treat this entry's role in the paid panel as the curve-fit baseline,
not a calibrated reserve.
