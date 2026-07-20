# clark_growth_curve — Bayesian Clark (Cape Cod)

**Family:** bayesian · **Reference:** Clark, "LDF Curve-Fitting and
Stochastic Reserving: A Maximum Likelihood Approach", *CAS Forum* (Fall
2003) for the likelihood. **There is no published Stan ground truth for a
Bayesian Clark** — the priors below are this package's specification, and
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

Cape Cod form only — one ELR against earned premium, Clark's recommendation
for triangle-sized data. A free-ultimates (LDF) Bayesian variant is a
scaffold exercise: eject this entry and swap `logelr` for a vector.

Priors:

| parameter | prior | rationale |
|---|---|---|
| logelr | normal(-0.4, sqrt(10)) | the Meyers-family variance-10 ELR prior |
| omega | lognormal(log 1.5, 0.5) | curve shape; mass on ~0.6–4 |
| theta | lognormal(log(4 * grain), 1) | curve scale; median at 4 dev periods, tracks the grain |

## Dispersion phi — plug-in from the MLE twin

`fit()` first runs the statistical `clark` entry (same curve, cape_cod) and
passes its Pearson scale into Stan as data — the same treatment as
`england_verrall_odp`'s phi. With phi fixed, the ODP quasi-likelihood is a
proper likelihood for `(logelr, omega, theta)` up to a constant.

## Data contract

`kernels.contract.odp_stan_data` (incremental cells, premium required,
negative increments rejected). The Stan data block additionally carries the
per-cell `age_lo`/`age_hi` and the curve code — ages are data, not model
logic, so ports cannot drift on the convention.

## Predictive distribution

Posterior draws of `(logelr, omega, theta)` × scaled-Poisson process draws
`phi * Poisson(mu/phi)` per future cell, truncated at the triangle's final
age (no tail), paid-to-date anchored — identical mechanics to the MLE twin
with the posterior replacing the delta-method MVN. Fully-developed origins
are constant.

## Validation

Retrospective Meyers protocol on paid: `scripts/meyers_validation.py
--model clark_growth_curve`. Results in
`analysis/results/clark_growth_curve_validation.csv`. No monograph bar
exists; compare against the paid panel (england_verrall_odp, meyers_csr,
clark MLE) on identical cohorts. Expect close agreement with the MLE twin
when the data are informative (the posterior concentrates near the MLE) and
regularization from the priors on thin books.
