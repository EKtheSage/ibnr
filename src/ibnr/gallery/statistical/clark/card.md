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

The curve MLE is a 2-D Nelder-Mead over `(log omega, log theta)` with the
level parameters profiled out in closed form (Poisson MLE given the curve).

## Uncertainty (Clark's decomposition, simulated)

- `phi`: Pearson chi-square / (n - p), Clark's scale estimate.
- **Parameter risk**: MVN draws on the log-parameters with covariance
  `phi * inverse observed Poisson information` (numerical Hessian at the
  MLE) - the quasi-likelihood delta method of the paper, in log space so
  levels stay positive.
- **Process risk**: scaled-Poisson ODP draws `phi * Poisson(mu/phi)` per
  future cell, the same process distribution as the `england_verrall_odp`
  entry and the bootstrap ODP baselines.

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
`latest_d` anchors, premium by origin (resolved under `cape_cod` only; `ldf`
never reads premium, so a losses-only triangle fits it with no
`premium_field` argument). Negative increments are rejected
(same ODP limitation as the bootstrap; failures are recorded, not patched).
Zero increments are fine. The `ldf` method additionally requires positive
paid-to-date per origin; `cape_cod` does not.

## Validation

- Tieout (`tests/test_clark.py`, fast): omega/theta, ultimates, ELR, and
  scale match chainladder's `ClarkLDF` on genins for both growth curves.
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
