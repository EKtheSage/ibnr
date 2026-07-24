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

## Backends (three ports, one contract)

| file | backend | sampler |
|---|---|---|
| `model.stan` | `stan` (reference, ground truth) | cmdstanpy NUTS |
| `model_numpyro.py` | `numpyro` | NumPyro NUTS (JAX) |
| `model_pymc.py` | `pymc` | PyMC NUTS (PyTensor; `nuts_sampler=` swaps in nutpie/numpyro/blackjax over the same graph) |

All three consume the identical `kernels.contract.stan_data` dict and expose
the same deterministic quantities, so `predict()`, `convergence()` and the
evaluation harness are backend-agnostic. `kernels.parity.compare_posteriors`
gates the ports against the Stan reference before any convergence claim is
made; `scripts/parity_meyers.py --model meyers_csr` runs the full comparison.

`parallel_chains` / `max_treedepth` are cmdstan-level controls (the retro
harness escalates on them) and are **rejected** by the ports rather than
silently ignored - an ignored escalation would be reported as a fit that never
happened.

## Parameterization (held constant across backends for parity)

- **Centered** parameterization throughout (as published), identical in all
  three backends - "same model" across PPLs only holds when parameterization is
  held constant, so convergence differences measure the samplers, not the code.
- a_i ~ uniform(0,1) via the monograph's inverse-gamma trick:
  a_ig ~ inv_gamma(1,1) bounded to (0, 1e5), a_i = gamma_cdf(1/a_ig | 1, 1),
  sig2[d] = reverse cumsum → sig decreasing in dev lag. **Note:** Meyers'
  CSR.R indexes the loop increments as `a_ig[i]` where we (and our CCL) use
  `a_ig[n_d-i]`; the a_ig are iid, so this is a relabeling with an identical
  joint distribution. **Documented deviation:** Stan bounds a_ig to (0, 1e5);
  the ports use an *unbounded* `InverseGamma(1,1)` (as in CCL). The truncated
  mass is ~1e-5, far below MCMC noise, and parity confirms it.
- `speedup` is the explicit recurrence `speedup[w] = speedup[w-1]*(1-gamma)`,
  matching the published code (algebraically `(1-gamma)^(w-1)`). The ports
  build it as a **cumulative product**, not a power: `gamma` is unconstrained,
  so a warmup excursion past 1 makes the base negative, and a float exponent on
  a negative base is NaN - which would poison the chain rather than fail
  loudly. `tests/test_parity_meyers_csr.py` pins the cumprod against the literal
  Stan loop for gamma on both sides of 1.
- **Init / adaptation** (documented, held equal): each backend uses its native
  default init (Stan uniform(-2,2); NumPyro `init_to_uniform`; PyMC
  `jitter+adapt_diag`) with a common `target_accept = 0.9`. Init affects the
  warmup path, not the stationary posterior.
- **Float precision:** JAX runs in float32 by default and x64 is deliberately
  NOT enabled, matching the CCL ports. The resulting ~1e-7 relative error is
  orders of magnitude below the MCSE that parity is measured in.
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

Retrospective Meyers protocol (train on the 1988-1997 upper triangle as of
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
