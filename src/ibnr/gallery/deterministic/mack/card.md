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

### Two routes, deliberately comparable

| call | what it gives | risk components |
|---|---|---|
| `one_year_cdr()` | closed-form msep per origin + total | `Phi` (process) + `Delta` (estimation) |
| `cdr_distribution()` | the full CDR *distribution* (quantiles, tails) | `process=` law; `parameter_risk=` on/off |

`cdr_distribution()` is "actuary in the box": simulate only the next diagonal
from the fitted model, append it, **re-run the chain ladder** on the extended
triangle, and difference the ultimates. With `parameter_risk=True` and
`process="normal"` it reproduces the analytic msep to Monte Carlo error
(asserted in `tests/test_cdr.py`); `parameter_risk=False` isolates `Phi`.

Because Mack's model constrains only two moments, the shape of the shock is an
assumption of the *simulation*, not of the model: `gamma` (default, positive
support), `lognormal`, or `normal` (matches the analytic linearization, can go
negative). The choice moves the tail quantiles - not the mean, and only
slightly the variance - and it is the first thing to vary when the CDR
distribution is used for capital.

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
- **The latest diagonal is checked on the variance path, not at fit time.**
  Those cells have no observed successor, so they enter no step's estimator -
  yet `msep_runoff`, the CDR and the simulations all divide by them. `fit_mack`
  therefore succeeds on a cohort with a non-positive diagonal and gives a valid
  point estimate; `msep_runoff()` / `one_year_cdr()` / `simulate_*()` raise,
  naming the origin. Incurred triangles net of bulk reserves are where this
  bites. (Before this split the same cohort returned a silent `NaN` msep and a
  `NaN` total.)
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
