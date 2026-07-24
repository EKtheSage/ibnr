# england_verrall_odp - Bayesian over-dispersed Poisson chain ladder

**Family:** bayesian · **Reference:** England & Verrall, "Stochastic Claims
Reserving in General Insurance", *British Actuarial Journal* 8 III (2002),
sections 3.2 (the ODP model) and 7.11 (the Bayesian implementation, originally
in WinBUGS with vague priors).

The Bayesian form of the cross-classified over-dispersed Poisson model whose
maximum-likelihood fitted values reproduce chain-ladder reserves exactly. The
monograph-companion of our retrospective studies: Meyers' frequentist ODP
baseline (`ODP.R`, `BootChainLadder` with od-Poisson process) is the bootstrap
twin of this posterior.

## Model

Incremental losses `X[w,d]` (differenced within origin from the cumulative
triangle):

```
E[X[w,d]]   = m[w,d]
Var[X[w,d]] = phi * m[w,d]
log m[w,d]  = logprem[w] + c + alpha[w] + beta[d],   alpha[1] = beta[1] = 0
```

England & Verrall present the multiplicative form `m = x_i * y_j` and note
(7.11.8) that the log-link linear predictor is the stable formulation - we
use the log link. The premium offset only re-centers `alpha` (each origin
keeps a free level), so the model family and its chain-ladder-reproducing
MLE are unchanged; the offset makes the vague priors exchangeable across
origins of different sizes.

Priors (E&V's "vague priors", concretized to the family's variance-10
convention): `c, alpha[w], beta[d] ~ normal(0, sqrt(10))`.

## Dispersion phi - plug-in, not sampled

`phi` enters Stan as **data**: the GLM Pearson chi-square estimate
`sum((x - m_MLE)^2 / m_MLE) / (n - p)` with `p = n_w + n_d - 1`, where
`m_MLE` comes from iterative proportional fitting (= the chain-ladder fitted
incrementals). This is exactly how England & Verrall treat the scale
parameter (quasi-likelihood; the scale is estimated outside the model).
With `phi` fixed, the od-Poisson quasi-likelihood
`(x/phi) log(mu/phi) - mu/phi - lgamma(x/phi + 1)` is the exact Poisson
mass at `x/phi` up to a constant in the parameters, so the posterior is a
proper Bayesian posterior for `(c, alpha, beta)`.

## Data contract

`kernels.contract.odp_stan_data`: same `(w, d)` conventions as the Meyers
family but **incremental** `inc_loss`, plus `paid_to_date` / `latest_d`
anchors per origin. Zero increments are legitimate (Poisson mass at zero);
**negative increments are rejected** - the ODP likelihood is undefined
there, the same limitation as the bootstrap ODP. Failures are recorded, not
patched. Fits paid_loss by default (the ODP ≡ chain-ladder equivalence is a
paid result; reported increments go negative far too often).

## Predictive distribution

For each origin, ultimate = observed paid-to-date + simulated future
increments `X ~ phi * Poisson(m/phi)` for each unobserved dev through
`n_d` - the od-Poisson process draw England & Verrall obtain by imputing
future cells as missing values (7.11.6), and the same process distribution
as `BootChainLadder(process.distr = "od.pois")`. Parameter uncertainty comes
from the posterior draws of `(c, alpha, beta)`; process variance from the
scaled-Poisson draw. The first origin is fully developed at the cutoff, so
its ultimate is constant (zero SE), aligning the summary table with the
rest of the family.

## Parameterization notes (for the milestone-5 ports)

- Log-link linear predictor, corner constraints `alpha[1] = beta[1] = 0`.
- `phi` plug-in as data - ports must consume the identical value from the
  contract dict, never re-estimate it internally.
- No hierarchical structure, no truncation; the posterior is smooth and
  log-concave, `target_accept = 0.8` is comfortable.

## Validation

Retrospective Meyers protocol on paid losses: run via
`scripts/meyers_validation.py --model england_verrall_odp`. Results land in
`analysis/results/england_verrall_odp_validation.csv`. The comparison bar
from Meyers (2nd ed., Figure 4.1, bootstrap ODP on paid): combined KS
D = 24.1* vs critical 9.6, failing three of four lines individually
(CA 23.1*, PA 44.9*, WC 28.4*, OL 6.8 vs critical 19.2) - the outcome
percentiles skew low because the ODP, like the plain cross-classified
model, over-predicts paid losses when settlement speeds up; meyers_csr's
settlement-rate term is the monograph's fix (combined D = 3.1).

**Result (2026-07-20, 96/200 companies completed - 104 rejected for
negative paid increments, the honest ODP-family coverage limit; Meyers'
bootstrap tolerated them in fitting, which is one reason his n = 200):
combined KS D = 47.9* vs crit 13.9, all four lines fail (CA 56.7*,
PA 63.6*, WC 49.7*, OL 29.8* marginal), percentiles piled low.** The
monograph's failure mode, amplified: the completed subset is
selection-biased toward longer-tailed books (a book must still be paying
at late lags to avoid both rejection modes), where the no-speedup
over-prediction is worst. Same window, same cohorts: meyers_csr passes at
D = 4.1. This entry's role is the calibrated Bayesian ODP baseline for the
ELPD/stacking harness, not a model expected to survive this backtest.
