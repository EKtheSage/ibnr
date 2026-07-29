# mack - distribution-free chain ladder + one-year CDR

**Family:** deterministic · **References:** Mack, "Distribution-free
calculation of the standard error of chain ladder reserve estimates", *ASTIN
Bulletin* 23/2 (1993); Merz & Wuthrich, "Modelling the claims development
result for solvency purposes", *CAS E-Forum* Fall 2008.

The gallery's reference point, and the only entry that answers the **one-year**
question. Mack's MSEP asks how wrong the ultimate can be; the claims
development result (CDR) asks how far next year's re-estimate can move, which
is the quantity a Solvency II reserve-risk capital charge is calibrated to.
Both are computed natively over `Triangle` - chainladder-python is an optional
interop extra here and has no CDR at all, so nothing is delegated to it.

## Model

Mack fixes the first two conditional moments of the development and assumes
accident years independent - no distribution:

```
E[C[i,j+1] | C[i,0..j]]   = f[j] * C[i,j]
Var(C[i,j+1] | C[i,0..j]) = sigma[j]^2 * C[i,j]
```

Estimated on the observed run-off triangle, over the origins `i` observing both
ends of each step:

```
S[j]        = sum_i C[i,j]                        (volume behind the factor)
f[j]        = sum_i C[i,j+1] / S[j]               (volume-weighted, alpha = 1)
sigma[j]^2  = 1/(n[j]-1) * sum_i C[i,j] * (C[i,j+1]/C[i,j] - f[j])^2
```

The two sums do not always run over the same origins. `sigma[j]^2`'s summand
carries a `1/C[i,j]` - the weighted residual is `(C[i,j+1] - f[j] C[i,j])^2 /
C[i,j]` - so it is estimable only where `C[i,j] > 0`, while `f[j]` needs nothing
beyond `S[j] > 0`. `n[j]` above is therefore the count of *positive* pair
origins, not the pair count, and the degrees of freedom follow it. On a triangle
with strictly positive cumulatives the two sets coincide and this says nothing;
they part on an accident year with zero paid at 12 months, whose chain-ladder
ultimate is perfectly well defined and whose sigma simply has one fewer
observation behind it. `MackFit.n_obs` and `MackFit.n_pos` report the two counts
separately, so the divergence is visible rather than inferred.

`alpha = 1` is not a default but a requirement: both Mack's and
Merz-Wuthrich's variance formulas are derived for the volume-weighted factor,
so no other averaging is offered.

**Last-step variance** (`sigma_rule=`). The final step has one observation and
no residual degrees of freedom:

| rule | sigma[J-1]^2 |
|---|---|
| `mack` (default) | `min(sigma[J-2]^4/sigma[J-3]^2, sigma[J-3]^2, sigma[J-2]^2)` - Mack's own, always shrinking |
| `log_linear` | regress `log sigma[j]` on `j`, extrapolate - what chainladder-python and R's `est.sigma="log-linear"` default to, and the convention behind most published raa numbers |

Both are pinned against `cl.Development(sigma_interpolation=...)` in
`tests/test_mack.py`. No tail factor is fitted: development stops at the
triangle's last observed lag, which is also the restriction R's
`CDR.MackChainLadder` enforces.

## Run-off uncertainty

Mack (1993) formula (3), with `k[i]` the dev index of origin `i`'s latest cell:

```
msep[i]  = Chat[i,J]^2 * sum_{j>=k[i]} (sigma[j]^2/f[j]^2) * (1/Chat[i,j] + 1/S[j])
```

the `1/Chat` half being process risk and the `1/S` half estimation risk. The
aggregate adds the estimation covariance between accident years, which share
the estimated factors; process risk carries no cross term.

## One-year claims development result

One year on, the new diagonal has emerged and every factor is re-estimated:

```
CDR[i](I+1) = Chat[i,J]^{I} - Chat[i,J]^{I+1}
```

positive = release, negative = strengthening, and `E[CDR | D_I] = 0` under the
model, so the risk measure is `msep = E[CDR^2 | D_I]`.

The key quantity is the **leverage** of the single new observation that joins
dev step `j` next year (the cell of the origin `i_j` sitting on the diagonal at
`j`):

```
a[j] = C[i_j,j] / S^{I+1}[j],      S^{I+1}[j] = S[j] + C[i_j,j]
```

Within twelve months an accident year learns its own next cell *in full*, but
learns about every later factor *only through that one new observation*. So
with `ratio[j] = sigma[j]^2/f[j]^2` and `k = k[i]`:

```
Phi[i]   = ratio[k]/C[i,k] + sum_{j>k} ratio[j]*C[i_j,j]/S^{I+1}[j]^2    (process)
Delta[i] = ratio[k]/S[k]   + sum_{j>k} a[j]^2 * ratio[j]/S[j]            (estimation)
msep[i]  = Chat[i,J]^2 * (Phi[i] + Delta[i])
```

Set against Mack's run-off formula, the one-year version keeps the `j = k`
terms whole and damps every later one by `a[j]`. Two consequences worth
knowing:

- an accident year **one step from ultimate** has one-year msep *exactly equal*
  to its run-off msep (nothing is left to learn afterwards);
- the one-year msep never exceeds the run-off msep, and the ratio of the two -
  reported as `one_year_share` - is the share of reserve risk that emerges in
  the first year. It is high for short-tailed business and low for long-tailed.

Aggregation carries a cross term, because accident years share the factors they
have yet to run through; the older year of each pair determines the shared
range.

**On the split.** `Phi`/`Delta` above is the paper's grouping, and `Phi` is
exactly what a re-reserving simulation with `parameter_risk=False` measures.
R's `CDR.MackChainLadder` labels its two pieces differently - only
`ratio[k]/C[i,k]` is called process variance there, and everything else,
including the later-step terms that are genuinely process risk, is called
parameter uncertainty. The **totals are algebraically identical** (`tests/
test_cdr.py::test_phi_delta_split_matches_the_reference_grouping` asserts it);
only the labels differ. Quote the convention when reporting a decomposition.

### The one-year CDR is not one method. Two axes, and they are separate

Every simulated one-year CDR is the same two-step recipe, and the steps are
independent choices:

1. **what generates next year's diagonal** - a `DiagonalGenerator`;
2. **how the reserve is re-estimated once it exists** - the volume-weighted
   chain ladder, `kernels.cdr.rereserve`.

Axis 2 is *not* a choice. It is the market convention and it is what R's
`ChainLadder` uses for its Mack CDR and its bootstrap CDR alike (`getAvDFs`
with the cumulative triangle as weights is the volume-weighted factor written
out), so every generator shares one implementation. Axis 1 is the choice:

```python
from ibnr.kernels.cdr import cdr_methods

cdr_methods()  # name, route, generates, re_estimates, returns, requires, validated
```

| method | route | generates next diagonal by | returns |
|---|---|---|---|
| `merz_wuthrich` | analytic | nothing - the factor update is linearized | `CDRResult` (msep, no quantiles) |
| `mack` | simulation | Mack's conditional moments, `E = f*C`, `Var = sigma^2*C` | `PredictiveDistribution` |
| `odp_bootstrap` | simulation | England-Verrall Pearson-residual bootstrap + over-dispersed Poisson noise | `PredictiveDistribution` |

```python
entry.one_year_cdr()  # merz_wuthrich (Mack only)
entry.cdr_distribution()  # mack, defaults
entry.cdr_distribution(generator="odp_bootstrap")  # England-Verrall
```

**`merz_wuthrich` is deliberately not reachable through `generator=`.** Every
term in it is `sigma[j]^2/f[j]^2`: it is a linearization of the chain-ladder
factor update *around Mack's conditional moments*, so there is no ODP version
of it and no version for any other model. Asking for it as a generator is
refused by name and redirected to `one_year_cdr()`, in the same spirit as the
held-out capability mixins - a method that cannot answer says so.

#### The `mack` generator

"Actuary in the box": simulate only the next diagonal from the fitted model,
append it, **re-run the chain ladder** on the extended triangle, and difference
the ultimates. With `parameter_risk=True` and `process="normal"` it reproduces
the analytic msep to Monte Carlo error (asserted in `tests/test_cdr.py`);
`parameter_risk=False` isolates `Phi`.

Because Mack's model constrains only two moments, the shape of the shock is an
assumption of the *simulation*, not of the model: `gamma` (default, positive
support), `lognormal`, or `normal` (matches the analytic linearization, can go
negative). The choice moves the tail quantiles - not the mean, and only
slightly the variance - and it is the first thing to vary when the CDR
distribution is used for capital.

#### The `odp_bootstrap` generator

The generator behind R's `CDR.BootChainLadder`: resample degrees-of-freedom
adjusted Pearson residuals into a pseudo-triangle, refit the chain ladder on
it, project the next diagonal off that refit and add over-dispersed Poisson
process noise (`kernels/odp_bootstrap.py`, written to be read against R's
source). `process=` picks the noise law - `od_poisson` (default,
`phi*Poisson(mu/phi)`, the England-Verrall construction `england_verrall_odp`
also draws) or `gamma`, both with mean `mu` and variance `phi*mu`. **R's
`BootChainLadder` defaults to `gamma`**, so a like-for-like comparison needs it
set explicitly.

`resample_residuals` and `process_noise` are the two ablation switches: R's
`NYCost` arm is both on, its `NYParamDist` arm is `process_noise=False`, and
the process-only arm R derives by subtraction is `resample_residuals=False`.
All three are simulated here, and their variances compose in quadrature to
Monte Carlo error - which is the assumption R's `CDR.Process.S.E` rests on.

**Validated to Monte Carlo error against the algorithm, not to published
digits, because no published digits exist.** R's `CDR.BootChainLadder` help
page prints no output for its example, and a bootstrap is stochastic. So the
route is pinned by (a) a literal transcription of R's `getNYCost` in the test
suite, which must reproduce the draws to 1e-10 when fed the same diagonal;
(b) a delta-method variance reference computed off the re-reserving Jacobian
for the process-only arm; (c) the Poisson-MLE property of the fitted values,
cross-checked against `england_verrall_odp`'s iterative proportional fit. Do
not quote this route as tied out; the Mack route is.

**Family limit, refused by name.** The ODP quasi-likelihood is defined on
non-negative increments, so a cohort with a negative paid increment cannot be
bootstrapped - roughly half the Schedule P mart, the same limit
`england_verrall_odp` carries. The error names the offending cells and points
at the `mack` generator, which has no such restriction. The reverse also holds
and is a real capability difference rather than an accident: an accident year
with **zero paid at 12 months** has a bootstrap CDR (the variance comes off the
fitted mean) and no Mack CDR (Mack's variance is proportional to that cell, so
`require_positive_open_diagonals` refuses).

#### One Mack assumption survives the split, and it is worth knowing

`rereserve` differences against the *deterministic* chain-ladder ultimate at
time I. Under Mack that is exactly right and is what makes `E[CDR | D_I] = 0`,
so `kernels.cdr.simulated_msep`'s mean square **about zero** is the risk
measure. A residual bootstrap centres its diagonal on the pseudo-triangle's
refit, so its draws carry a small bias (measured at ~0.5% of a standard
deviation on the test fixture, ~1.7% on MW2014) and the mean square about zero
is no longer quite the variance. R draws the same distinction from the other
side: `CDR.MackChainLadder` reports the analytic msep (about zero) and
`CDR.BootChainLadder` reports `sd()` of the re-reserved amount (about its own
mean). On a non-Mack generator, report both.

#### Plugging in a third generator

`rereserve(fit, next_diagonal)` is public. Anything that can draw next year's
cumulative cells - including a gallery entry with `PredictsHeldout` - can be
re-reserved through the same volume-weighted chain ladder and produce a CDR
directly comparable to these two.

### Capital

`kernels.cdr.cdr_risk_measures(pred, levels=(0.995,))` reports VaR and TVaR on
the **loss** `-CDR` (the strengthening), which is the Solvency II reserve-risk
basis:

```python
from ibnr.kernels.cdr import cdr_risk_measures

cdr_risk_measures(entry.cdr_distribution(n_draws=100_000, seed=1))
```

Quantiles are exact empirical order statistics, so a 99.5th percentile is only
as good as the draws behind it (at 20k draws it rests on 100 observations).
This needs the simulation, not the closed form - a second moment does not imply
a quantile - and it is where the choice of `process` law bites hardest.

## Prediction

`predict()` returns simulated **full run-off ultimates** (per origin, plus a
total column that is the row-sum of the same draws, so diversification is in
the samples). Parameter risk is drawn once per draw and shared across accident
years, which is what correlates them; process noise is independent. This is the
bootstrap wrapper a deterministic method needs to enter the gallery - the point
estimate alone could not.

## Held-out one-step draws (CRPS board)

The entry subclasses `PredictsHeldout`: `predict_at(cells)` draws the outcome
of next-diagonal cells the fit never saw, so mack sits on the leaderboard's
CRPS panel. Each cell's draw carries Mack's one-step conditional moments off
its own training predecessor - for a cell at 1-based dev index `d`,

```
E   = f[d-2] * prev_value
Var = sigma[d-2]^2 * prev_value
```

with `prev_value` on the training diagonal (data the model already had, so no
leakage). The draws are `kernels.mack.draw_next_cells`, which **shares its
core with `cdr_distribution()`'s simulated next diagonal** - the board and the
CDR carry the identical noise assumption by construction, not by discipline.
Draws are cumulative (`heldout_draw_scale = "cumulative"`), the triangle's own
basis, so the base class passes them through unchanged.

Knobs, set at fit time (`fit(..., heldout_n_draws=10_000,
heldout_process="gamma", heldout_parameter_risk=True)`):

- `heldout_process` - the step shock's shape, one of `gamma` (default,
  positive support), `lognormal`, `normal`. As everywhere in this entry, the
  law is an assumption of the *simulation*: Mack's model fixes two moments and
  nothing else. `gamma`/`lognormal` need a positive conditional mean, and
  `require_positive_open_diagonals` is the guard that makes that well-posed -
  a fit whose open diagonal carries a zero or negative cell refuses to draw
  (loudly, naming the origin) rather than degenerate.
- `heldout_parameter_risk` - draw the "true" factors from their estimation
  error once per draw, **shared across the cells**. That shared draw is what
  correlates the held-out diagonal, exactly as it correlates accident years in
  the CDR; off, the cells are independent pure process noise.

One caveat worth knowing: a cell whose development step has `sigma^2 = 0`
(possible only via the extrapolated last-step rule, e.g. a 2-column triangle)
draws a **point mass at its mean, silently** - `draw_step`'s documented
degenerate case. Zero estimated variance is the model's answer there, however
implausible the triangle that produced it, so it is left to stand rather than
patched.

**ELPD is a permanent N/A, by design.** Mack's model states two conditional
moments and no distribution, so there is no predictive density to evaluate an
outcome under - which is exactly what an ELPD is. The gamma law above *would*
technically define a one-step density, but claiming it would promote an
assumption of the simulation into a claim about the model, and that promotion
is an explicitly reserved decision (do not make it in passing; ask). The entry
therefore does **not** subclass `ScoresHeldout`, and the board prints its ELPD
as `na: no_predictive_density` - a statement about the density axis only. Its
draws axis is first-class: CRPS and PIT work fine.

## Data contract

`kernels/contract.py::cohort_grid` - one cohort, cumulative, a genuine run-off
staircase (each origin observed from dev 1 to one common calendar diagonal).
Anything else is a hard error rather than a repair: the factors are estimated
from exactly those cells, so a fabricated one would silently change them. Slice
the backtest window with `as_of=` before fitting.

## Limitations

- **Volume-weighted only**, and **no tail factor**. Both are restrictions of
  the underlying formulas, not of the implementation.
- **Positivity is required unevenly**, and the fit says where. Mack's variance
  is proportional to `C`, so a zero cumulative carries no conditional variance -
  but the volume-weighted *factor* divides only by the column total. So a zero
  above the diagonal costs that step's sigma one observation and nothing else,
  while a **negative** cumulative is refused outright (it would drive
  `sigma[j]^2` itself negative, hence a negative msep and a NaN standard error),
  as is a step with zero volume or with fewer than two positive origins. The
  errors name the dev step and the offending origins.
- **The latest diagonal is checked on the variance path, not at fit time, and
  the check belongs to Mack rather than to the CDR.** Those cells have no
  observed successor, so they enter no step's estimator - yet `msep_runoff`,
  the closed form and every *Mack* simulation divide by them. `fit_mack`
  therefore succeeds on a cohort with a non-positive diagonal and gives a valid
  point estimate; `msep_runoff()` / `one_year_cdr()` / `simulate_*()` on the
  `mack` generator raise, naming the origin. Incurred triangles net of bulk
  reserves are where this bites. (Before this split the same cohort returned a
  silent `NaN` msep and a `NaN` total.) The guard now lives on
  `MackDiagonal.check`, not on `simulate_one_year_cdr`: the `odp_bootstrap`
  generator has a *different* precondition (non-negative increments) and used
  to inherit Mack's, which refused cohorts it could perfectly well answer for.
- **The last step's sigma is an extrapolation**, and on a small triangle it can
  dominate the youngest accident year's uncertainty. The two rules disagree by
  design; if the answer is sensitive to which one is chosen, say so rather than
  picking silently.
- The analytic CDR is a **linearization** (as in the paper). The simulation is
  not, which is why the two differ by a fraction of a percent on real triangles
  and by more when a step's coefficient of variation is large.
- One-year CDR is defined per cohort. Cross-line diversification of the CDR is
  **not** modelled here; that belongs with the multiline entries
  (`statistical/sur`, `nn/transformer_ml`).
