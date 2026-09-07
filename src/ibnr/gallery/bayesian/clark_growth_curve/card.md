# clark_growth_curve - Bayesian Clark (Cape Cod)

**Family:** bayesian · **Reference:** Clark, "LDF Curve-Fitting and
Stochastic Reserving: A Maximum Likelihood Approach", *CAS Forum* (Fall
2003) for the likelihood. **There is no published Stan ground truth for a
Bayesian Clark** - the priors below are this package's specification, and
the milestone-5 NumPyro/PyMC ports must hold them (and the plug-in phi)
constant. The statistical `clark` entry is the MLE twin sharing the exact
likelihood, parameterization, and age convention.

## Model

Incremental paid losses:

```
X[w,d] ~ ODP:  E[X] = elr * premium[w] * (G(x_hi) - G(x_lo)),  Var[X] = phi * E[X]
G = loglogistic (default) or weibull, ages from the average accident date
    (x = 12d - 6 for annual grains, first period starts at 0)
```

Cape Cod form only - one ELR against earned premium, Clark's recommendation
for triangle-sized data. A free-ultimates (LDF) Bayesian variant is a
scaffold exercise: eject this entry and swap `logelr` for a vector.

Priors:

| parameter | prior | rationale |
|---|---|---|
| logelr | normal(-0.4, sqrt(10)) | the Meyers-family variance-10 ELR prior |
| omega | lognormal(log 1.5, 0.5) | curve shape; mass on ~0.6-4 |
| theta | lognormal(log(4 * grain), 1) | curve scale; median at 4 dev periods, tracks the grain |

## Dispersion phi - plug-in from the MLE twin

`fit()` first runs the statistical `clark` entry (same curve, cape_cod) and
passes its Pearson scale into Stan as data - the same treatment as
`england_verrall_odp`'s phi. With phi fixed, the ODP quasi-likelihood is a
proper likelihood for `(logelr, omega, theta)` up to a constant.

## Data contract

`kernels.contract.odp_stan_data` (incremental cells, premium required,
negative increments rejected). The Stan data block additionally carries the
per-cell `age_lo`/`age_hi` and the curve code - ages are data, not model
logic, so ports cannot drift on the convention.

## Predictive distribution

Posterior draws of `(logelr, omega, theta)` × scaled-Poisson process draws
`phi * Poisson(mu/phi)` per future cell, truncated at the triangle's final
age (no tail), paid-to-date anchored - identical mechanics to the MLE twin
with the posterior replacing the delta-method MVN. Fully-developed origins
are constant. The process draw is the shared `kernels.densities.odp_draw`,
which returns the mean exactly when the dispersion has collapsed so far
that `mu / phi` is past the largest Poisson rate numpy can draw at.

## Held-out evaluation: CRPS only, never ELPD

On the milestone-6 board this entry is scored by CRPS from one-step draws:
`scorer.py` (beside `model.stan`) draws each next-diagonal increment as
`X = phi * Poisson(mu / phi)`, one per posterior draw of
`(logelr, omega, theta)`, with the growth-curve share taken over the cell's
mid-period age interval from the one shared `age_interval` (the same
convention the fit put in the Stan data block). The entry declares
`heldout_draw_scale = "incremental"`, so `PredictsHeldout.predict_at` adds
each cell's training-diagonal cumulative before the draws meet the outcomes.
A cell whose age interval spans no emergence has `mu` exactly 0 (floored at
1e-12 so the Poisson stays legal) and draws all zeros - the model's own
statement, handled by the panel's bookkeeping rather than a refusal here.

The ELPD column is permanently empty, on principle: the ODP
quasi-likelihood is not a normalized density on any scale -
`exp(odp_lpdf)/phi` integrates to 0.69 at `mu/phi = 0.5` and the defect
varies with `mu/phi`, so no change of variable can fix it
(`kernels/densities.py`, the odp-not-a-density note). An ELPD would require
a real predictive distribution (negative binomial, Tweedie) - a different
model, not a conversion.

## Validation

Retrospective Meyers protocol on paid: `scripts/meyers_validation.py
--model clark_growth_curve`. Results in
`analysis/results/clark_growth_curve_validation.csv`. No monograph bar
exists; compare against the paid panel (england_verrall_odp, meyers_csr,
clark MLE) on identical cohorts.

**Result (2026-07-20, loglogistic, 96/200 completed - the rest rejected for
negative paid increments): combined KS D = 63.4* vs crit 13.9, percentiles
piled at ~0 - within noise of the MLE twin's 61.3*, confirming the
posterior tracks the MLE.** The failure is the model, not the inference:
the post-1997 settlement speedup plus the loglogistic tail mass (see the
MLE twin's card for the weibull arm, D = 49.6*). In the gallery this
entry is the growth-curve baseline on the CRPS board (it is ELPD-ineligible,
see the held-out section above); meyers_csr (D = 4.1, passes) is what
calibrated paid reserving looks like in this window.

## Backends (three ports, one data block)

| file | backend | sampler |
|---|---|---|
| `model.stan` | `stan` (reference, ground truth) | cmdstanpy NUTS |
| `model_numpyro.py` | `numpyro` | NumPyro NUTS (JAX) |
| `model_pymc.py` | `pymc` | PyMC NUTS (PyTensor) |

All three consume the identical Stan `data` block the entry assembles - ages,
the plug-in `phi` and the integer curve code included - so no port can drift on
the age convention or quietly fit the other curve. `parallel_chains` /
`max_treedepth` are cmdstan-level controls and are **rejected** by the ports
rather than silently ignored.

### The growth curve at age zero (the one real hazard)

`model.stan` writes `if (x <= 0) return 0;`, and that branch is **not** an edge
case: `age_lo` is exactly 0 for every origin's first development cell, by
construction - the mid-period age shift `max(step(d-1) - step/2, 0)` clamps
there. Every triangle hits it, on every fit.

The obvious port, `where(x > 0, formula, 0.0)`, produces **the correct value and
a NaN gradient**. Both PPLs evaluate both branches of a `where`/`switch`, so the
NaN generated inside the unselected branch (`theta/0 -> inf` for loglogistic;
differentiating `0**omega` for Weibull) propagates through the reverse pass.
Measured at the prior medians on ages `[0, 6, 18, 42]`:

| curve | form | G values | dG/dtheta | dG/domega |
|---|---|---|---|---|
| loglogistic | naive `where` | correct | **NaN** | **NaN** |
| loglogistic | safe (shipped) | identical | -0.00601 | -0.2333 |
| weibull | either | identical | -0.00703 | -0.2669 |

The **loglogistic curve is the default**, so the naive port breaks the common
path while looking right on inspection - NUTS fails on the first leapfrog with
nothing informative to report. Both ports therefore use the standard
double-`where`: a safe dummy age is substituted *inside* the formula as well as
masking the result. `x` is data, so the mask is static and costs nothing.

`tests/test_parity_clark.py` asserts the **gradient**, not just the value. A
value-only test passes cleanly on the broken implementation, which makes it
worse than no test at all.

### Likelihood attachment

Stan's `odp_lpdf` is duplicated here rather than imported from
`england_verrall_odp`: `model.stan` duplicates it too, and each model directory
must stand alone for `gallery.scaffold()` (decision 6). It is attached as a
`numpyro.factor` / `pm.Potential` rather than an observed distribution, because
`mu` can legitimately reach 0 when a cell's age interval spans no emergence -
where a distribution's support check would reject outright instead of letting
the density go to `-inf`. ELPD for this entry is read off the Stan fit, whose
`generated quantities` carries `log_lik`.

## Cross-backend parity & convergence (milestone 5)

Compared parameters are `CLARK_PARITY_VARS` = `logelr, omega, theta` - the whole
model. `omega` and `theta` trade off along a ridge (shape against scale), so a
port that got the curve subtly wrong would show up as a shifted pair even when
the fitted development pattern looked similar.

**Result: 4 of 4 PASS, with the tightest agreement in the gallery** - every
z-score below 1.6 against a tolerance of 4. Two WC companies as of 1997-12-31,
4 chains x 2500 draws after 1000 warmup, `target_accept = 0.9`, common seed,
loglogistic curve. Full data in
`analysis/results/{parity,convergence}_clark.csv`; reproduce with

```
uv run python scripts/parity_gallery.py --model clark_growth_curve \
    --line workers_compensation --companies 11347 1538
```

| company | backend | max &#124;z_mean&#124; | max &#124;z_sd&#124; | max KS | verdict |
|---|---|---|---|---|---|
| 11347 | numpyro | 0.95 | 1.52 | 0.025 | PASS |
| 11347 | pymc | 1.28 | 1.20 | 0.026 | PASS |
| 1538 | numpyro | 0.71 | 0.92 | 0.015 | PASS |
| 1538 | pymc | 1.07 | 0.85 | 0.021 | PASS |

| company | backend | runtime | max R-hat | min ESS-bulk | divergences /10000 |
|---|---|---|---|---|---|
| 11347 | stan | 7.7s | 1.00 | 2796 | 0 |
| 11347 | numpyro | 7.6s | 1.00 | 2698 | 0 |
| 11347 | pymc | 53.2s* | 1.00 | 3193 | 0 |
| 1538 | stan | 4.1s | 1.00 | 4710 | 0 |
| 1538 | numpyro | 4.3s | 1.00 | 4517 | 0 |
| 1538 | pymc | 12.9s | 1.00 | 4150 | 0 |

\* cold PyTensor compile; 1538's 12.9s is the warm figure, so the compile is
roughly 40s of the difference - the same split seen on every other entry.

Zero divergences everywhere and ESS up to 4700 of 10000 draws. With only three
parameters the sampler has very little to get wrong once the gradient is
correct, which is precisely why the gradient is the thing this entry's tests
guard. NumPyro matches Stan almost exactly on wall-clock here (7.6s vs 7.7s,
4.3s vs 4.1s) - the smallest parameter space in the gallery gives JAX's fused
graph the least to amortize over, and PyMC's warm cost stays ~3x.
