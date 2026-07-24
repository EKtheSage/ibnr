# meyers_csr - Changing Settlement Rate (Meyers)

**Family:** bayesian · **Reference:** Meyers, *Stochastic Loss Reserving Using
Bayesian MCMC Models*, CAS Monograph 1 (2015) section 8; 2nd ed. Monograph 8
(2019) section 7. Meyers' published `CSR.R` Stan script is ground truth.

The monograph's **paid-loss** model: a cross-classified lognormal whose
log-development factors trend across accident years with a settlement-rate
parameter. Meyers built CSR because the plain cross-classified model (CRC)
systematically overestimates paid ultimates when claim settlement speeds up -
a positive `gamma` shrinks `beta[d] * (1-gamma)^(w-1)` toward zero for later
origins, absorbing the speedup.

## Model

```
log(C[w,d]) ~ normal(mu[w,d], sig[d])
  mu[w,d] = logprem[w] + logelr + alpha[w] + beta[d] * speedup[w]
  speedup[w] = (1 - gamma)^(w-1)
```

with `alpha[1] = 0` and `beta[n_d] = 0` (so `e^logelr` reads as the expected
loss ratio and the fully-developed origin defines the level). `gamma == 0`
recovers the CRC model exactly; CSR differs from CCL/CAY in having **no
across-origin correlation term** (`rho`) - origins are conditionally
independent given the parameters.

Priors (variance-10 normals, exactly as in the monograph):

| parameter | prior | notes |
|---|---|---|
| logelr | normal(-0.4, sqrt(10)) | log expected loss ratio |
| alpha[w], w>=2 | normal(0, sqrt(10)) | AY level offsets |
| beta[d], d<n_d | normal(0, sqrt(10)) | development profile |
| gamma | normal(0, **0.05**) | settlement-rate trend; **sd 0.05**, not variance - Meyers' Stan code says `gamma ~ normal(0, 0.05)` while his prose convention elsewhere quotes variances |
| sig2[d] | sum_{i=d}^{n_d} a_i, a_i ~ uniform(0,1) | forces sig2 decreasing in d |

## Backends

| file | backend | sampler |
|---|---|---|
| `model.stan` | `stan` (reference, ground truth) | cmdstanpy NUTS |

NumPyro/PyMC ports and the `kernels.parity` gate arrive with milestone 5
(the roadmap builds every gallery model in Stan first); `fit(backend=...)`
already reserves the dispatch seam.

## Parameterization

- **Centered** parameterization throughout (as published).
- a_i ~ uniform(0,1) via the monograph's inverse-gamma trick:
  a_ig ~ inv_gamma(1,1) bounded to (0, 1e5), a_i = gamma_cdf(1/a_ig | 1, 1),
  sig2[d] = reverse cumsum → sig decreasing in dev lag. **Note:** Meyers'
  CSR.R indexes the loop increments as `a_ig[i]` where we (and our CCL) use
  `a_ig[n_d-i]`; the a_ig are iid, so this is a relabeling with an identical
  joint distribution.
- `speedup` is the explicit recurrence `speedup[w] = speedup[w-1]*(1-gamma)`,
  matching the published code (algebraically `(1-gamma)^(w-1)`).
- Generalized to `n_w` x `n_d` from the published hard-coded 10x10.
- Sampling defaults: 4 chains x 2500 draws after 1000 warmup (the monograph
  uses a posterior sample of 10,000), `adapt_delta = 0.9`. Meyers ran
  `adapt_delta = 0.9999, max_treedepth = 50` with an R-hat-triggered
  thinning escalation; we default lower and surface divergences via
  `convergence()` instead.

## Data contract

`kernels.contract.stan_data` - the same dict as the whole Meyers family;
CSR consumes `(len_data, n_w, n_d, w, d, logprem, logloss)` and ignores
`prev_idx`. Fits **paid_loss** (`loss_field` default) against net earned
premium, per the monograph. Lognormal likelihood requires positive training
cells; Meyers clamps paid cells to a floor of 1 (in $000s) - the validation
harness reproduces that clamp so the paid study keeps the same company
cohort as the incurred (CCL) study.

## Predictive distribution

The monograph's simulation, exactly as in `CSR.R`:

1. Origin 1 (fully developed at the training cutoff) is its observed
   `C[1, n_d]` - zero predictive variance.
2. For w >= 2: `C[w, n_d] ~ lognormal(logprem[w] + logelr + alpha[w], sig[n_d])`,
   independent across origins (`beta[n_d] = 0` kills the speedup term at
   the ultimate, which is why estimates coincide with CRC's).
3. Total = observed C[1] + sum of simulated ultimates; the outcome
   percentile is the CDF of the realized total under these draws.

Documented deviation: Meyers applies `ceiling()` to each simulated ultimate
(whole-$1000 rounding); we keep the draws continuous (immaterial at study
scale, and CRPS prefers continuous samples).

## Validation

Retrospective Meyers protocol (train on the 1988–1997 upper triangle as of
1997-12-31, score the realized paid ultimate percentile, KS/PIT uniformity
across companies): run via `scripts/meyers_validation.py --model meyers_csr`.
Results land in `analysis/results/meyers_csr_validation.csv`. The monograph's
own paid CSR result (2nd ed. Figure 7.4) **passes uniformity outright**:
combined KS D = 3.1 vs critical 9.6, and every line individually (CA 5.9,
PA 18.5, WC 12.0, OL 10.4 vs critical 19.2) - versus the plain CRC's
combined D = 25.5*. That is the bar to compare against.

**Our result (2026-07-20, 200 companies, 4x2500 draws, adapt_delta 0.9,
zero failures): combined KS D = 4.1 (crit 9.6, p = 0.885) - passes, in line
with Meyers' 3.1.** Per line (crit 19.2): CA 7.4, WC 11.9, OL 13.3 all
pass; PA fails at 25.3* (Meyers' PA was 18.5, just under - private
passenger auto's post-1997 settlement regime is the shared weak spot).
Occasional divergences (typically <0.2%, worst ~1.3% on one chain) -
the centered-parameterization property already documented for CCL.
