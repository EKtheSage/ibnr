# meyers_ccl — Correlated Chain Ladder (Stan reference)

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

## Parameterization (held constant across backends for parity)

- **Centered** parameterization throughout (as published).
- a_i ~ uniform(0,1) is implemented via the monograph's inverse-gamma trick:
  a_ig ~ inv_gamma(1,1) bounded to (0, 1e5), a_i = gamma_cdf(1/a_ig | 1, 1).
  This avoids hard zero boundaries that stall the sampler.
- The rho residual term references the *previous origin, same dev* cell via an
  explicit `prev_idx` index array (the original code relied on row ordering).
  Rows are sorted by (w, d); training on the upper triangle guarantees
  (w-1, d) is observed whenever (w, d) is.
- Sampling defaults: 4 chains x 2500 draws after 1000 warmup (the monograph
  uses a posterior sample of 10,000).

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

The model fits **reported_loss = IncurLoss − BulkLoss** (incurred net of
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
signature — fitting bulk-inclusive incurred is NOT equivalent; the net-of-bulk
definition is load-bearing for calibration.

Occasional divergent transitions (typically <1%, worst ~3%) occur on
individual companies — a known property of the centered parameterization; the
monograph used the same parameterization. Documented here for the parity
comparison (milestone 3).
