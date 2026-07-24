# meyers_ccl - Correlated Chain Ladder (Stan reference)

## Provenance

Meyers, *Stochastic Loss Reserving Using Bayesian MCMC Models*, CAS Monograph 1
(2015), where the model is named **CCL**; identical to the **CAY** (Correlated
Accident Year) model of the 2nd edition, CAS Monograph 8 (2019), section 8.
The monograph's published Stan code is ground truth; `model.stan` modernizes
the syntax for Stan >= 2.33 (`array` declarations, `gamma_cdf(x | a, b)`)
without changing the math or priors. Meyers found CCL/CAY to be the best
performing of his models on **incurred** loss triangles.

## Model

For cumulative losses C[w, d] (origin year w = 1..10, dev year d = 1..10):

    log C[w, d] ~ normal(mu[w, d], sig[d])
    mu[1, d] = log Premium[1] + logelr + beta[d]
    mu[w, d] = log Premium[w] + logelr + alpha[w] + beta[d]
               + rho * (log C[w-1, d] - mu[w-1, d])        w > 1

Priors (variance-10 normals, exactly as in the monograph):

| parameter | prior | notes |
|---|---|---|
| logelr | normal(-0.4, sqrt(10)) | log expected loss ratio |
| alpha[w] | normal(0, sqrt(10)), alpha[1] = 0 | origin-year level |
| beta[d] | normal(0, sqrt(10)), beta[10] = 0 | development level |
| rho | 2*beta(2,2) - 1 | AY correlation; Corr[log C[w,d], log C[w-1,d]] = rho/(1+rho^2) |
| sig2[d] | sum_{i=d}^{10} a_i, a_i ~ uniform(0,1) | forces sig2 decreasing in d |

## Backends (three ports, one contract)

The entry dispatches to three interchangeable posterior samplers behind
`fit(..., backend=...)`, all consuming the identical `kernels.contract` data
dict and producing a common `arviz.InferenceData` that `predict()` reads off:

| file | backend | sampler |
|---|---|---|
| `model.stan` | `stan` (reference, ground truth) | cmdstanpy NUTS |
| `model_numpyro.py` | `numpyro` | NumPyro NUTS (JAX) |
| `model_pymc.py` | `pymc` | PyMC NUTS (PyTensor) |

`kernels.parity.compare_posteriors` gates the ports against the Stan reference
(mean agreement in MCSE units, SD ratio, marginal KS) before any convergence
claim is made; `scripts/parity_gallery.py` runs the full comparison.

## Parameterization (held constant across backends for parity)

- **Centered** parameterization throughout (as published), identical in all
  three backends - "same model" across PPLs only holds when parameterization is
  held constant, so convergence differences measure the samplers, not the code.
- a_i ~ uniform(0,1) is implemented via the monograph's inverse-gamma trick:
  a_ig ~ inv_gamma(1,1), a_i = gamma_cdf(1/a_ig | 1, 1) = 1 - exp(-1/a_ig),
  sig2[d] = sum_{i>=d} a_i (reverse cumsum → sig decreasing in dev lag). This
  avoids hard zero boundaries that stall the sampler. **Stan's `a_ig` upper
  bound is reproduced in both ports, and it is load-bearing.** Until 2026-07-24
  the ports used an *unbounded* `InverseGamma(1,1)`, on the argument that
  Stan's (0, 1e5) truncates only ~1e-5 of the **prior**. That was wrong: the
  **posterior** concentrates in exactly that corner at deep dev lags, where the
  data barely constrain sigma. Measured on WC 11347, `P(a_ig > 1e5)` runs from
  0.1% at d=0 to **10.6% at d=8** - the `a -> 0` region Stan forbids - which
  pulled `sig` **5-12% below** the Stan reference (worst at the deepest lags)
  while leaving `logelr` and `rho` untouched. The ports now carry Stan's
  constraint: an interval-constrained parameter with the `InverseGamma(1,1)`
  density as a factor (NumPyro) / `pm.Truncated` (PyMC).
  In PyMC, **both** bounds must be declared (`lower=0, upper=1e5`): with
  `upper=` alone it picks a one-sided `log(upper - x)` transform that knows
  nothing of the InverseGamma's own lower bound, so the default jitter pushes
  `x` negative, out of support, for a `logp = -inf` SamplingError before warmup.
- **NumPyro init deviation (documented, and necessary).** The NumPyro port
  seeds `a_ig` at 1e3: the (0, 1e5) interval transform maps the default
  unconstrained uniform(-2,2) init onto a_ig in (1.2e4, 8.8e4), hard against
  the upper bound. Stan starts there without trouble; NumPyro reaches max R-hat
  1.59 / ESS 7. Only that one site's warmup path is affected, and the
  stationary posterior is not. PyMC needs no override once both bounds are set.
- **Judge these models on `sig`, not `a_ig`.** The individual `a_ig[j]` are
  weakly identified in *every* backend including Stan - the likelihood only
  sees their cumulative sums through `sig2` - so their R-hat/ESS look alarming
  even at convergence.
- The rho residual term references the *previous origin, same dev* cell. Stan
  builds `mu` with a forward recurrence over an explicit `prev_idx` array; the
  ports use its exact closed form `mu = P(rho) @ B` (see
  `kernels.contract.ccl_mu_index`) so the JAX/PyTensor autodiff graphs stay a
  small matmul instead of an N-deep scalar chain - otherwise the PyTensor
  C-compile dominates PyMC's wall-clock. The two forms are algebraically
  identical (unit-tested to <1e-10).
- **Init / adaptation** (documented, held equal): all backends use their native
  default init (Stan uniform(-2,2); NumPyro `init_to_uniform`; PyMC
  `jitter+adapt_diag`) with a common `target_accept = 0.8` (Stan `adapt_delta`).
  Init strategy affects the warmup path, not the stationary posterior.
- Sampling defaults: 4 chains x 2500 draws after 1000 warmup (the monograph
  uses a posterior sample of 10,000). NumPyro/PyMC default to sequential /
  single-core chain execution (Windows multiprocessing for these samplers is
  fragile).

## Data contract

`kernels.contract.stan_data(triangle, loss_field=..., premium_field=...)`:
`len_data, n_w, n_d, w, d, prev_idx, logprem, logloss`. Requires a cumulative,
single-cohort triangle with positive losses and premiums; slice training data
with `triangle.as_of(...)` first.

## Prediction

Ultimates = losses at the final dev period, simulated per posterior draw
exactly as in the monograph (Monograph 8, p. 38): the first origin's ultimate
is its observed value; for w = 2..10, mu[w] = log Premium[w] + logelr +
alpha[w] + rho * (log C~[w-1, 10] - mu[w-1]) and C~[w, 10] ~
lognormal(mu[w], sig[10]). Targets: one per origin year plus the total.

## Validation protocol (Meyers retrospective test)

Train on the Schedule P upper triangle as of 1997-12-31 (incurred), score the
realized C[w, 10] from subsequent statements, record the predictive percentile
of the total outcome per insurer, and test the percentiles across insurers for
uniformity (KS / p-p plots; 5% critical value 19.2 for n=50 per line, 9.6 for
n=200 pooled). Company selection follows the monograph appendix mechanically:
complete 10x10 triangles, all-year premium > $20,000k and incurred loss >
$4,000k, then the 50 lowest premium-CV companies per line under the Table A.1
CV1 limits.

The model fits **reported_loss = IncurLoss - BulkLoss** (incurred net of
bulk+IBNR, i.e. paid + case), exactly as in the monograph. Company selection
applies both Table A.1 screens (CV1 on net premium, CV2 on the net/direct
premium ratio); the exact company set may still differ marginally from the
monograph's Table A.2 list (selection re-derived mechanically, not
transcribed).

## Validation results (2026-06-13, net-of-bulk incurred, n=200)

200/200 selected insurers fitted (4 chains x 2500 draws; max per-company
R-hat 1.08). KS D x100 of total-outcome percentiles vs Meyers' published
CAY-on-incurred values (Monograph 8, Figure 8.5):

| line | ours | Meyers | 5% crit |
|---|---|---|---|
| commercial_auto | 13.2 | 8.3 | 19.2 |
| private_passenger_auto | 10.1 | 15.9 | 19.2 |
| workers_compensation | 13.1 | 12.5 | 19.2 |
| other_liability | 9.8 | 20.8* | 19.2 |
| combined (n=200) | **6.8** | 10.8* | 9.6 |

All four lines pass at 5%, as does the combined test (p = 0.31). Outcome
percentiles are centered (per-line means 46-52). An earlier run against
gross-of-bulk incurred failed WC at D = 36.7 with a strong over-prediction
signature - fitting bulk-inclusive incurred is NOT equivalent; the net-of-bulk
definition is load-bearing for calibration.

Occasional divergent transitions (typically <1%, worst ~3%) occur on
individual companies - a known property of the centered parameterization; the
monograph used the same parameterization. Documented here for the parity
comparison (milestone 3).

## Cross-backend parity & convergence (milestone 4)

### Parity (the correctness gate)

`kernels.parity.compare_posteriors` compares each port's posterior to the Stan
reference marginal-by-marginal, in **Monte-Carlo-error units**: the mean
difference over the combined MCSE of the mean, and the SD difference over the
combined MCSE of the SD. Both are ~N(0, 1) under the null (same posterior,
independent runs); a parameter passes when both z-scores are within `z_tol = 4`.
Scaling the SD check by `mcse_sd` (not a flat fractional band) is what keeps the
gate honest on short chains - the deepest-dev `sig`, identified by a single
observation, has a large `mcse_sd`, so its noisier SD estimate is tolerated
automatically. KS on the pooled marginals is reported for context only (MCMC
autocorrelation inflates it). The comparison runs against Stan when a cmdstan
toolchain is present, and NumPyro-vs-PyMC otherwise (so CI can gate parity
without Stan). `scripts/parity_gallery.py` runs it on real Schedule P companies.

**Result, re-measured 2026-07-24 (4 chains x 2500 draws after 1000 warmup,
`target_accept = 0.9`, two WC companies; `analysis/results/parity_meyers.csv`):
4 of 4 PASS.**

| company | backend | max &#124;z_mean&#124; | max &#124;z_sd&#124; | max KS | verdict |
|---|---|---|---|---|---|
| 11347 | numpyro | 1.65 | 2.45 | 0.025 | PASS |
| 11347 | pymc | 1.53 | 3.24 | 0.031 | PASS |
| 38687 | numpyro | 2.18 | 1.58 | 0.022 | PASS |
| 38687 | pymc | 1.93 | 2.07 | 0.018 | PASS |

The mu closed form matches the Stan `prev_idx` recurrence exactly (unit-tested
to <1e-10).

**This supersedes the original milestone-4 parity claim, which was not
reproducible.** Two corrections, both found in 2026-07-24 while porting CSR:

1. **The gate was silently lenient.** `kernels.parity` read `az.summary`, which
   rounds to 3 decimals; every z-score is a difference of two summaries over
   their combined MCSE, so any parameter whose MCSE rounded to `0.000` took the
   `if mcse > 0 else 0.0` branch and scored `z = 0` - an automatic pass,
   precisely for the best-identified parameters. Fixed (`round_to="none"`) and
   guarded by a test. The old headline figures (max|z_mean| 2.8, max|z_sd| 3.6)
   were produced through that bug.
2. **The unbounded `a_ig` was a real defect**, not the harmless deviation
   claimed above - see the Parameterization section. Restoring Stan's bound cut
   max KS from ~0.10 to ~0.03 and dropped the ports' divergences by an order of
   magnitude (numpyro 24 -> 1, pymc 11 -> 8 on 11347).

**Draw budget matters in both directions - measure, don't assume.** At the
original 500-draw protocol the post-fix SD check straddled the tolerance with
the *failing backend alternating by cohort* (numpyro failed 38687 at 4.03, pymc
failed 11347 at 5.83) - the signature of noise, since at low ESS `mcse_sd` is
itself poorly estimated. Re-running the identical fits at 2500 draws dropped
pymc's max|z_sd| from **5.83 to 3.24** and numpyro's from 3.28 to 2.45, giving
the 4/4 above; a *systematic* gap would instead have grown by ~sqrt(5). Parity
is therefore quoted at 2500 draws, and the table below is the matching run.
Practical rule: do not gate parity below ~1000 draws.

### Convergence & runtime

Identical settings across backends (4 chains x 2500 draws after 1000 warmup,
single-core / sequential chain execution for a fair per-chain runtime,
`target_accept = 0.9`, common seed) on two Schedule P WC companies as of
1997-12-31 - the same run the parity table above comes from. Diagnostics via
arviz. Runtimes are cache-warm (PyTensor/JAX compiled, Stan binary built) - see
the compile note below. Full data in
`analysis/results/convergence_meyers.csv`; reproduce with
`scripts/parity_gallery.py --line workers_compensation --n-companies 2 --draws 2500 --target-accept 0.9`.

| company | backend | runtime | max R-hat | min ESS-bulk | divergences /10000 |
|---|---|---|---|---|---|
| 11347 | stan | 9.3s | 1.00 | 2897 | 0 |
| 11347 | numpyro | 22.0s | 1.00 | 3107 | 1 |
| 11347 | pymc | 67.1s | 1.00 | 3569 | 8 |
| 38687 | stan | 6.8s | 1.00 | 2873 | 27 |
| 38687 | numpyro | 16.7s | 1.00 | 2981 | 3 |
| 38687 | pymc | 66.7s | 1.00 | 2861 | 23 |

(The earlier 500-draw table is superseded: those fits predate the `a_ig` fix, so
their divergence counts - numpyro 24, pymc 11 on 11347 - reflect the missing
bound rather than the samplers.)

**All three backends reach the same posterior** (parity passes, above) at
comparable R-hat (1.00) and ESS (~2900-3600 from 10000 draws). The difference is
wall-clock: **Stan** is fastest (~7-9s), **NumPyro (JAX/XLA)** ~2.5x that,
**PyMC** ~7-10x. The gap is the backend, not the model or parameterization: PyTensor's C
backend evaluates the logp/gradient as many small ops per leapfrog, whereas
NumPyro's JAX/XLA fuses the whole graph and Stan emits one tight translation
unit. PyTensor also links no BLAS on this box (it warns as much), but that is a
minor contributor at this size - the `P(rho) @ B` matmul is only 55x55. The
pip BLAS route (`scipy-openblas64`) does **not** help here: its MSVC-built
library will not link against the RTools MinGW g++ PyTensor compiles with, so
PyTensor rejects the flag. The real levers are a conda-installed PyTensor
(matching toolchain + MKL/OpenBLAS) or a faster PyMC sampler (`nutpie` /
`numba`), which cut the per-leapfrog overhead directly. Recorded this way so the
comparison measures the samplers, not the build.

**`target_accept` matters for the centered parameterization.** At the Stan
default 0.8, Stan and NumPyro sample 11347 fine (0 / 38 divergences) but PyMC
under-adapts badly (R-hat 1.17, 182 divergences, ESS 18) - and slows down,
since divergent / max-treedepth trajectories are long. Raising `adapt_delta` to
0.9 with 1000 warmup fixes it (11347 PyMC: 182 -> 11 divergences), consistent
with the occasional-divergence note above. This is the documented adaptation
setting for the parity/convergence comparison; production fits of hard companies
should use 0.9+. The 0.8 arm is kept as
`analysis/results/convergence_meyers_ta08.csv`.

**One-time compile** (excluded from the runtimes above): PyTensor C-compiles
PyMC's logp/dlogp on the first fit of a given triangle shape (~4-5 min on this
Windows/RTools box), then caches by graph shape - every same-shape company after
is cache-warm. The vectorized `mu` (an N x N matmul, not an N-deep scalar chain)
is what keeps this compile bounded; the unrolled form pushed it far higher.

### Accelerating PyMC (attempted; unresolved on this Windows box)

`model_pymc.sample` takes a `nuts_sampler=` argument ("pymc" default |
"nutpie" | "numpyro" | "blackjax") that swaps the NUTS implementation over the
*same* model graph - posterior and parity are unaffected, only runtime changes.
The known levers to close PyMC's ~7x gap are (a) a **conda/pixi PyTensor** with a
matching toolchain + MKL/OpenBLAS, or (b) **`nutpie`** (Rust NUTS, compiles the
logp via numba). Neither could be made to work on this Windows/RTools machine
(2026-07-08):

- **`nutpie` (pip/uv-installable):** installs (pulls numpy 2.x, so use an
  isolated env), but its sampling *hangs* here - even 1 chain x 300 draws did not
  finish in 8 min, while NumPyro samples the same model in ~27s. A numba /
  nutpie-on-Windows issue with this graph.
- **conda/pixi BLAS:** `pixi` resolves the coherent stack pip cannot (MKL + a
  matching gcc), but the environment will not finalize in this sandbox -
  Windows Defender file-locks freshly-extracted conda packages, failing the
  atomic rename (`Access is denied`). Needs a Defender exclusion / admin.
- **pip BLAS (`scipy-openblas64`):** blocked as noted above (MSVC lib vs MinGW).

Both accelerator paths are expected to work on **Linux** (conda-forge PyTensor
links MKL out of the box; nutpie's numba/Windows problem does not apply), so the
plan is to re-benchmark `nuts_sampler="nutpie"` and a conda-BLAS PyTensor in a
Linux container and record the numbers here. Until then, NumPyro is the fast
backend on this machine and PyMC is the readable, parity-checked reference port.
