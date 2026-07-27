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

## Held-out evaluation: CRPS only, never ELPD

On the milestone-6 board this entry is scored by CRPS from one-step draws:
`scorer.py` (beside `model.stan`) draws each next-diagonal increment as
`X = phi * Poisson(mu / phi)`, one draw per posterior draw, with `mu` built
from the held-out cell's origin premium and the posterior `(c, alpha, beta)`.
The entry declares `heldout_draw_scale = "incremental"`, so
`PredictsHeldout.predict_at` adds each cell's training-diagonal cumulative
before the draws meet the outcomes - without that anchor the draws would be
finite, plausible, and wrong by the whole cumulative-to-date.

The ELPD column is permanently empty, and that is a property of the model,
not missing work. The ODP quasi-likelihood is not a normalized density on
any scale: `exp(odp_lpdf)/phi` integrates to 0.69 at `mu/phi = 0.5` and 0.83
at 1.0, and the defect varies with `mu/phi`, so it cannot be fixed by a
change of variable and does not cancel between models
(`kernels/densities.py`, the odp-not-a-density note). Giving this entry an
ELPD would mean replacing the quasi-likelihood with a real distribution
(negative binomial, Tweedie) - a different model, not a units conversion.

## Backends (three ports, one contract)

| file | backend | sampler |
|---|---|---|
| `model.stan` | `stan` (reference, ground truth) | cmdstanpy NUTS |
| `model_numpyro.py` | `numpyro` | NumPyro NUTS (JAX) |
| `model_pymc.py` | `pymc` | PyMC NUTS (PyTensor; `nuts_sampler=` swaps in nutpie/numpyro/blackjax over the same graph) |

All three consume the identical `kernels.contract.odp_stan_data` dict - the
same plug-in `phi` included - and expose the same deterministic quantities, so
`predict()`, `convergence()` and the evaluation harness are backend-agnostic.
`scripts/parity_gallery.py --model england_verrall_odp` runs the full
comparison. `parallel_chains` / `max_treedepth` are cmdstan-level controls and
are **rejected** by the ports rather than silently ignored.

### The likelihood is hand-written, and that is the interesting part

ODP is the first gallery entry whose likelihood **neither PPL provides**. Stan
declares its own `odp_lpdf`; each port therefore transcribes a density by hand,
which is the one thing here that could be subtly wrong while still sampling
happily. Two facts keep it honest:

- **`odp_lpdf(x | mu, phi)` is exactly `Poisson(mu/phi).log_prob(x/phi)`.** The
  od-Poisson quasi-likelihood is a Poisson mass at the scaled data with the
  scaled mean - which is *why* it is a proper density despite `x/phi` not being
  an integer. `tests/test_parity_odp.py` pins both ports against that closed
  form and against each other. It is written out literally in each port rather
  than delegated to the built-in Poisson, because the gallery ships readable
  ejectable source (decision 6) and because both PPLs' `Poisson` is a
  *discrete* distribution whose support would reject the continuous `x/phi`.
- **The parameter-free terms are kept.** `-lgamma(x/phi + 1)` and
  `(x/phi) log(1/phi)` are constants in the parameters (`phi` and `x` are
  data), so dropping them could not move the posterior - but it would shift
  every `log_lik` by a constant and silently break ELPD comparability with
  Stan's `generated quantities`. Stan keeps them too: the constant-dropping
  that `~` performs for built-in distributions does not apply to a user-defined
  `_lpdf`.

**How the density is attached decides whether ELPD is possible at all.** Both
ports use a genuine observed variable - a `dist.Distribution` subclass in
NumPyro, `pm.CustomDist(..., observed=)` in PyMC - not a `numpyro.factor` /
`pm.Potential`. `pm.Potential` yields **no** `log_likelihood` group whatsoever
(verified directly); `numpyro.factor` does yield one, since it is itself an
observed `Unit` site, but leaves `observed_data` degenerate and `Predictive`
unusable. Only a real distribution keeps likelihood, observed data and forward
simulation all correct at once.

## Parameterization notes

- Log-link linear predictor, corner constraints `alpha[1] = beta[1] = 0`. Note
  ODP pins the **first** beta where the Meyers family pins the **last**; the
  conventions are not interchangeable.
- `phi` plug-in as data - every backend consumes the identical value from the
  contract dict and never re-estimates it internally.
- No hierarchical structure, no truncation; the posterior is smooth and
  log-concave, and `target_accept = 0.8` is comfortable in all three backends.
  None of the `a_ig` boundary trouble that forced an explicit init in the
  Meyers ports applies here - there is no bounded parameter.
- **Float precision:** JAX runs in float32 (x64 deliberately not enabled,
  matching the other ports). This density is a small difference of large terms,
  so float32 leaves ~3e-5 absolute on a log-density of order 1 - orders of
  magnitude below the MCSE parity is measured in, but the reason the density
  tests carry a 1e-4 tolerance rather than machine epsilon.

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

## Cross-backend parity & convergence (milestone 5)

`kernels.parity.compare_posteriors` compares each port's posterior to the Stan
reference marginal-by-marginal in **Monte-Carlo-error units** - the mean
difference over the combined MCSE of the mean, the SD difference over the
combined MCSE of the SD - both required within `z_tol = 4`. Compared
parameters are `ODP_PARITY_VARS` = `c, alpha, beta`: the whole log-link linear
predictor. There is no scale parameter to compare, because `phi` is a plug-in
constant rather than something the sampler explores.

**Result: 4 of 4 PASS**, and every fit converged cleanly. Two WC companies as
of 1997-12-31, 4 chains x 2500 draws after 1000 warmup, `target_accept = 0.8`,
common seed. Full data in `analysis/results/{parity,convergence}_odp.csv`;
reproduce with

```
uv run python scripts/parity_gallery.py --model england_verrall_odp \
    --line workers_compensation --companies 11347 1538
```

| company | backend | max &#124;z_mean&#124; | max &#124;z_sd&#124; | max KS | verdict |
|---|---|---|---|---|---|
| 11347 | numpyro | 2.27 | 1.94 | 0.024 | PASS |
| 11347 | pymc | 1.02 | 2.27 | 0.020 | PASS |
| 1538 | numpyro | 1.53 | 1.94 | 0.021 | PASS |
| 1538 | pymc | 1.51 | 2.57 | 0.025 | PASS |

| company | backend | runtime | max R-hat | min ESS-bulk | divergences /10000 |
|---|---|---|---|---|---|
| 11347 | stan | 6.8s | 1.00 | 2982 | 0 |
| 11347 | numpyro | 8.6s | 1.00 | 2977 | 0 |
| 11347 | pymc | 13.9s | 1.00 | 2779 | 0 |
| 1538 | stan | 6.0s | 1.00 | 2543 | 0 |
| 1538 | numpyro | 3.8s | 1.00 | 2363 | 0 |
| 1538 | pymc | 17.5s | 1.00 | 2355 | 0 |

This is the **best-behaved entry in the gallery so far**, and the reason is the
model rather than the ports: a smooth log-concave GLM with no bounded
parameter, no hierarchy and no scale to co-adapt. Every backend reaches R-hat
1.00 with ESS ~2400-3000 of 10000 draws and **zero divergences** - where CCL
and CSR both need `target_accept` 0.9 and still show some. It also closes the
runtime gap: PyMC is only ~2-3x Stan here rather than the ~7-10x seen on the
Meyers family, and NumPyro actually *beats* Stan on 1538 (3.8s vs 6.0s), the
first time any port has. Note 11347's PyMC figure is a cold PyTensor compile;
its warm cost is the ~14s shown, and 1538's 17.5s is warm throughout.

### Cohort selection is constrained by the model, not chosen for flattery

ODP requires **non-negative incremental losses**, so it legitimately refuses
triangles with a negative paid increment - about half the Schedule P mart, a
documented coverage limit of the ODP family (see Validation below). The
second-largest WC company by the standard size ranking is one such cohort, and
the contract raises rather than fitting it. The two companies above are the
top two of that ranking that ODP can actually take (9 of the top 12 qualify),
named explicitly via `--companies` so the constraint is visible in the command
rather than hidden behind a silent skip.
