# compartmental - Hierarchical Compartmental Reserving (Gesmann & Morris)

**Family:** bayesian · **Reference:** Gesmann & Morris, *Hierarchical
Compartmental Reserving Models*, CAS Research Paper (2020). The published
brms/Stan code of the Section 5 case study (appendix 7.2) is the ground
truth; both variants below hold its parameterization and priors verbatim,
and the milestone-5 NumPyro/PyMC ports must hold them constant. Original
single-triangle case study: Morris, *Hierarchical Compartmental Models for
Loss Reserving* (CAS E-Forum, 2016).

## Model

The only entry in the gallery that fits **paid and case outstanding
jointly**. Premium flows through three compartments,

```
EX' = -ker * EX          (exposure earns and is reported at rate ker)
OS' =  ker * RLR * EX - kp * OS      (case reserves settle at rate kp)
PD' =  kp * RRF * OS                 (RRF = reserve robustness factor)
```

with EX(0) = 1 per unit premium and the closed-form solution

```
OS(t) = RLR * ker/(ker-kp) * (e^(-kp t) - e^(-ker t))
PD(t) = RLR * RRF/(ker-kp) * (ker(1 - e^(-kp t)) - kp(1 - e^(-ker t)))
```

so the ultimate loss ratio is RLR·RRF. `t` is the development age in
**years at the cell's period end** (the case study's `Lag` = 1..10; ker/kp
are per-year rates; no mid-period shift - gradual earning is what the EX
compartment models). Ages, premiums and the delta indicator are data via
`kernels.contract.compartmental_stan_data` (outstanding = `reported_loss` -
`paid_loss`, i.e. net of bulk on both sides; the monograph's wkcomp case
study used direct premium - we use net earned premium like the rest of the
gallery, consistent with the net loss basis).

## Variants (switchable, both from the monograph's case study)

### `variant="gaussian"` (default) - case-study Model 1

Gaussian likelihood on **amounts**: OS levels (delta=0) and cumulative paid
(delta=1), `sigma` per delta on the log link. ker, kp fixed across accident
years; (RLR, RRF) carry **correlated accident-year effects** - the
monograph's signature reserving-cycle structure (posterior RLR-RRF
correlation > 0 means prudent case reserves in hard markets).

```
ker    = 3   exp(0.1 oker)                  oker ~ N(0,1)   [LN(log 3, 0.1)]
kp     = 1   exp(0.1 okp)                   okp  ~ N(0,1)   [LN(log 1, 0.1)]
RLR[w] = 0.7 exp(0.2 (b_RLR + u_RLR[w]))    b_RLR ~ N(0,1)  [LN(log .7, .2)]
RRF[w] = 0.8 exp(0.1 (b_RRF + u_RRF[w]))    b_RRF ~ N(0,1)  [LN(log .8, .1)]
(u_RLR, u_RRF) ~ MVN(0, D Ω D),  Ω ~ LKJ(1)
sd(u_RLR) ~ Student-t(10,0,0.2)+,  sd(u_RRF) ~ Student-t(10,0,0.1)+
log sigma[os], log sigma[paid] ~ Student-t(1,0,1000)
```

Takes zero/negative cells natively - what a mechanical 200-company
retrospective needs. The monograph excludes Model 1 from its own model
*selection* (a Gaussian can pay negative claims); it remains the robust
reference fit and the Morris (2016) original.

### `variant="lognormal"` - case-study Model 2

Lognormal likelihood on **loss ratios**: OS levels and **incremental** paid
(differenced with the same cell parameters at `t` and `t - devfreq`). All
four compartmental parameters get varying effects by accident year **and**
development year ("row" and "column" effects on the parameters, not the
outcome); (RLR, RRF) accident-year effects correlated as above, everything
else independent.

```
sd priors: oRLR 0.7, oRRF 0.5, oker 0.3, okp 0.3   (all Student-t(10,0,·)+,
           same prior for the AY and dev grouping of each parameter)
log sigma[δ] ~ N(log 0.2, 0.2)
```

(The monograph text quotes `sigma ~ LN(log 0.1, 0.2)`; its appendix code -
which produced the published results - uses `normal(log(0.2), 0.2)` on the
log-sigma coefficients. The code wins; documented here so ports don't
drift.) Non-positive OS or incremental-paid cells are dropped before
sampling and counted in `dropped_cells_` - the lognormal analogue of the
ODP entries' negative-increment failures.

Both variants sample at the monograph's `adapt_delta = 0.99`,
`max_treedepth = 15`.

## Parameterization notes

Non-centered accident-year (and dev-year) effects via
`diag_pre_multiply(sd, L_chol) * z` - brms's own default; ports must keep
it. Initialization is Stan's default U(-2,2) on the unconstrained scale;
all compartmental parameters are strictly positive by construction
(lognormal transforms of unconstrained Gaussians), the monograph's stated
reason for this parameterization.

## Predictive distribution

Cumulative paid at the triangle's final development age per origin + total
(no tail extrapolation):

- **gaussian**: `Normal(premium[w] * PD(t_final; RLR[w], RRF[w], ker, kp),
  sigma[paid])` per posterior draw - the model's unconditional-given-
  parameters predictive; what the origin's observed cells taught the
  posterior enters through the accident-year effects. Fully developed
  origins anchor at their observed value (zero variance), matching the
  Meyers-family retrospective protocol.
- **lognormal**: paid-to-date plus lognormal incremental draws cell-by-cell
  through the final age, with per-(origin, dev) parameters.

## Schedule P retrospective (2026-07-21, milestone 4)

Meyers-protocol paid retro: 200 companies (Table A.1 screens, 50 per line),
train as-of 1997-12-31, score the realized paid at dev 10, KS on the
total-outcome percentiles. No paid clamp (the gaussian arm takes zeros
natively; the lognormal arm drops its own non-positive cells). Both arms:
**zero company failures**; R-hat > 1.05 on 2/200 (gaussian) and 6/200
(lognormal) companies. `analysis/results/compartmental_validation.csv` +
`_lognormal.csv`; seed 20260612, 4 x 2500 draws.

| KS D (crit 19.2 / 9.6) | gaussian (M1) | lognormal (M2) |
|---|---|---|
| commercial_auto | 51.2* | **12.5 (passes)** |
| other_liability | 69.3* | 20.0* (marginal) |
| private_passenger_auto | 37.2* | 36.5* |
| workers_compensation | 28.9* | 28.2* |
| **combined (n=200)** | **39.9*** | **16.1*** |

The two failure modes separate cleanly:

- **gaussian fails on bias + sharpness** - the monograph's own critique of
  Model 1, reproduced at scale. Constant amount-scale sigma → median total
  CV 2.5%, 110/200 outcomes outside the 5-95 band; and the case-study
  priors (ULR median ≈ 0.7·0.8 = 0.56; kp ~ LN(0, 0.1), i.e. ~63% of
  outstanding paid within a year, ±20% wiggle) were tuned to one
  fast-settling WC book - on long-tailed other liability the median
  estimate/outcome is **0.84** with 35/50 outcomes above the 95th
  percentile. Priors are load-bearing when transferred mechanically.
- **lognormal removes the bias entirely** (median estimate/outcome
  0.99-1.02 on every line - the AY + dev varying effects give the curve
  enough freedom to escape the binding priors) and outside-band outcomes
  drop to 63/200. What remains is the residual too-sharp/regime problem
  concentrated in PPA + WC (percentiles leaning low = mild over-prediction
  into the post-1997 favorable-development + settlement-speedup regime that
  only meyers_csr's gamma term absorbs).

Context in the paid panel (identical cohorts): meyers_csr 4.1 (passes) <
**compartmental lognormal 16.1*** < compartmental gaussian 39.9* <
england_verrall_odp 47.9* < clark weibull 49.6* < clark loglogistic 61.3*.
The lognormal variant is the best-calibrated paid entry after CSR.

Runtime honesty: the monograph's `adapt_delta = 0.99, max_treedepth = 15`
cost ~160 s (gaussian) / ~250 s (lognormal; WC median 488 s) per company
sequential-chain - ~20x the Meyers-family entries. Nothing here needed a
model change; see the roadmap for the parallel retro harness and a
two-stage escalation policy (fast settings, retry hard companies at the
monograph settings).

Open comparison (not built): a `hierarchical` variant with per-line,
mart-derived prior medians (or cross-company partial pooling) to test
whether the gaussian arm's failure is purely the single-company priors.

## Held-out scoring (milestone 6)

The entry subclasses `ScoresHeldout` and `PredictsHeldout`; `scorer.py`
evaluates both variants' likelihoods and draws at arbitrary cells, in plain
numpy, over the posterior of any backend. Because the two variants sit on
different observation scales, the declarations are per-variant instance
attributes set by `fit()`:

| variant | `heldout_measure` (density of) | `heldout_draw_scale` (draws are) |
|---|---|---|
| `gaussian` | `amount` - OS levels + cumulative paid amounts | `cumulative` paid amounts |
| `lognormal` | `loss_ratio` - OS-level + incremental paid ratios | `incremental` paid amounts |

The carries are the base classes' job, once: `log_lik_at` subtracts
`log premium` from the lognormal ratio density (the gaussian one is already
on amounts), and `predict_at` adds the training-diagonal anchor to the
lognormal variant's incremental draws. Both variants' single-cell densities
are normalization-checked on the amount scale in
`tests/test_heldout_scorer_compartmental.py`.

**Board scope: paid only.** The held-out `CohortForecast` covers
`field='paid_loss'` - the one field intersectable with the other entries.
`index_into` resolves exactly that field (`'outstanding'` is derived,
reported minus paid, not a raw field), hands back a plain `CellIndex`, and
the scorer reads a plain index as the paid block (`scorer.cell_deltas`).
The OS block is not board-scored; it exists for the in-sample agreement
gate, where `training_index` returns a `DeltaCellIndex` over BOTH stacked
blocks with per-block predecessors - the (w, d) ambiguity that used to make
`training_index` refuse this contract outright.

**Lognormal refusals.** A held-out cell whose OS level or paid increment is
non-positive has no lognormal density: the scorer raises (the same family
limit `_lognormal_stan_data` applies at fit time, counted in
`dropped_cells_`), and a retro run maps that to a cohort-level
`scoring_refused` absence. Draws do NOT inherit the refusal - a cohort with
a negative held-out increment is still CRPS-scorable. The gaussian variant
takes non-positive cells natively on both axes, deliberately. The
in-sample gate aligns with Stan's `log_lik` (which covers only surviving
rows) via `entry._kept_rows_`, the stored keep mask.

The lognormal scorer reconstructs the per-cell parameters from the sampled
`sd_*`/`z_*` sites (`u = sd * z`, the Stan file's own identities) rather
than reading the Stan-only `u_*` transformed parameters, so held-out
scoring is backend-blind across all three ports. Note the gap that remains:
`predict()` for the lognormal variant still reads the Stan-only
`u_*_dev`/`u_*_ay` names (`model.py::_predict_lognormal`), so full-triangle
prediction is Stan-only while held-out scoring is not.

## Data contract

`kernels.contract.compartmental_stan_data(paid_field, reported_field,
premium_field)`: stacked (delta=0 outstanding, delta=1 paid) cells with
`w`, `d`, `t`, per-origin premium and paid-to-date anchors. Paid and
reported must be present on identical cells; dev lags contiguous per
origin. No positivity enforced at the contract level.

## Backends (three ports, one data block)

| file | backend | sampler |
|---|---|---|
| `model.stan` / `model_lognormal.stan` | `stan` (reference, ground truth) | cmdstanpy NUTS |
| `model_numpyro.py` | `numpyro` | NumPyro NUTS (JAX) |
| `model_pymc.py` | `pymc` | PyMC NUTS (PyTensor, or `nuts_sampler="numpyro"` over the same graph) |

Both variants are ported to both PPLs and consume the identical Stan `data`
block the entry assembles. `parallel_chains` is cmdstan-only and is rejected by
the ports; `max_treedepth` is a genuine NUTS control in all three and is passed
through.

### Three constructs that had to be reproduced exactly

**Half-Student-t scales.** Stan's `vector<lower=0>[2] sd_ay` +
`student_t(10, 0, 0.2)` is a *half* Student-t (the constraint truncates; Stan
does not renormalize, and it does not need to - the missing factor is the
constant 1/2). PyMC's `pm.HalfStudentT(nu, sigma)` matches directly. NumPyro
has no half-Student-t, and the obvious `TruncatedDistribution(StudentT, low=0)`
**fails outright here** - it needs the Student-t CDF and raises
`ImportError: install tensorflow_probability`. The port therefore uses Stan's
own construction: a positive-constrained `ImproperUniform` site plus the
density as a `numpyro.factor`. Checked against the analytic half-t quantiles
(scale 0.2: median 0.138 vs 0.140, q90 0.358 vs 0.363).

**The positivity of `sd_ay` is load-bearing, not decoration.** Drop the
constraint and the model gains an exact sign symmetry: with `u = diag(sd) L z`
and `L[0,0] = 1`, the map `(sd_0, z[0,:], L[1,0]) -> (-sd_0, -z[0,:], -L[1,0])`
leaves `u_ay` and hence the entire likelihood invariant. The posterior becomes
bimodal and symmetric in rho, so `rho_ay` - the reserving-cycle correlation
that is this model's headline actuarial output - averages to ~0. **Nothing
raises.** Same failure family as the `a_ig` bound in the Meyers family, and
`tests/test_parity_compartmental.py` guards it in both PPLs.

**LKJ on a 2x2 Cholesky factor.** `lkj_corr_cholesky(1)` contributes *literally
zero* to Stan's target (its kernel is `L[1,1]^0 = 1`); all the geometry lives in
the `cholesky_factor_corr` constraint transform's Jacobian. A port can get the
density term right, get the transform wrong, and look correct on inspection -
so the prior is checked by sampling: rho comes back with sd 0.5774 in both
PPLs, i.e. exactly uniform on (-1, 1), as LKJ(1) implies in two dimensions.
PyMC's `LKJCholeskyCov(sd_dist=HalfStudentT(...))` reproduces Stan's *separate*
`sd_ay` + `L_ay` priors (scales' medians 0.1398 / 0.0703 against the analytic
0.1400 / 0.0700), and its `chol` output **is** `diag_pre_multiply(sd_ay, L_ay)`,
so the non-centered line transcribes verbatim.

## Cross-backend parity & convergence (milestone 5)

Compared parameters are `COMPARTMENTAL_PARITY_VARS` - the population-level
scalars both variants expose under identical names (`b_oRLR`, `b_oRRF`,
`b_oker`, `b_okp`, `sigma_os`, `sigma_paid`, `rho_ay`). The per-accident-year
`RLR`/`RRF` are excluded because Model 1 carries one set per accident year while
Model 2 resolves them per cell - not the same quantity across variants - and the
raw `(sd_ay, L_ay)` block has no common name, since PyMC bundles it into one
`LKJCholeskyCov` variable. `rho_ay` is the interpretable summary of that block
and stands in for it.

**Result vs the Stan reference** (WC company 11347 as of 1997-12-31, gaussian,
4 chains x 2500 draws after 1000 warmup, monograph settings `adapt_delta = 0.99`
/ `max_treedepth = 15`; `analysis/results/{parity,convergence}_compartmental.csv`):

| backend | runtime | max R-hat | min ESS-bulk | div /10000 | max &#124;z_mean&#124; | max &#124;z_sd&#124; | max KS |
|---|---|---|---|---|---|---|---|
| stan | 229.0s | 1.00 | 2697 | 0 | reference | | |
| numpyro | **59.4s** | 1.00 | 2570 | 0 | **0.67** | **0.84** | **0.011** |

**The tightest agreement anywhere in the gallery**, and the first entry where a
port decisively BEATS Stan on wall clock - NumPyro is 3.9x faster here. That is
the opposite of the Meyers family's ordering and it is `max_treedepth = 15` that
does it: long trajectories mean many gradient evaluations per iteration, which
is exactly where JAX's fused graph amortizes and Stan's per-leapfrog cost does
not. On the small synthetic triangles used in the unit tests the ordering is
reversed again (NumPyro ~2x Stan), so quote this comparison at production size
or not at all.

**PyMC: correct, but not runnable natively at this size.** Both variants pass
parity against the NumPyro port on a real-shaped problem (gaussian
z_mean 1.28 / z_sd 2.29 / KS 0.029; lognormal 2.53 / 1.47 / 0.039), which
establishes that the PyMC graph defines the same posterior. But PyTensor's
sampler costs ~0.57-0.94 s per iteration on this model against NumPyro's
~0.007 s - a ~60-80x gap that does NOT come from the model:

| the SAME PyMC graph | sampler | 2 chains x 3000 iters |
|---|---|---|
| `nuts_sampler="numpyro"` | JAX | **34.5s** |
| `nuts_sampler="pymc"` (default) | PyTensor | **>2000s, did not finish** |

Nor is it `max_treedepth`: capping at 10 instead of 15 barely helped, because
trajectories run ~35-60 leapfrog steps and never approach either cap. It is
per-gradient cost on a BLAS-less pip PyTensor - the effect the meyers_ccl card
already documents, amplified by this model's larger gradient graph (LKJ Cholesky
transform, matrix products, exp/where over stacked rows).

**Practical recommendation: run this entry's PyMC graph with
`nuts_sampler="numpyro"`.** It is the same model at the same cost as the
NumPyro port, and it is reachable straight from the gallery:

```python
gallery.fit("compartmental", triangle, backend="pymc", nuts_sampler="numpyro")
```

The default remains `"pymc"` deliberately. Defaulting to a faster foreign
sampler would turn the cross-backend convergence comparison - the entire point
of milestone 5 - into NumPyro measured against NumPyro over two graph
representations, and would bury a real, fixable environment problem (the
BLAS-less pip PyTensor documented on the meyers_ccl card, whose actual fix is a
conda/MKL PyTensor). The swapped run is also labelled distinctly,
`backend = "pymc:numpyro"` rather than `"pymc"`, so it can never masquerade as
a native fit in a results CSV. The native PyTensor path is retained because it
is the parity reference for the graph itself, not because anyone should sample
production fits with it here.
