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

## Variants (ablatable, both from the monograph's case study)

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

Open ablation (not built): a `hierarchical` variant with per-line,
mart-derived prior medians (or cross-company partial pooling) to test
whether the gaussian arm's failure is purely the single-company priors.

## Data contract

`kernels.contract.compartmental_stan_data(paid_field, reported_field,
premium_field)`: stacked (delta=0 outstanding, delta=1 paid) cells with
`w`, `d`, `t`, per-origin premium and paid-to-date anchors. Paid and
reported must be present on identical cells; dev lags contiguous per
origin. No positivity enforced at the contract level.
