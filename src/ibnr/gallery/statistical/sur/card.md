# sur - Seemingly Unrelated Regression (multivariate chain ladder)

**Family:** statistical (frequentist stochastic; simulation-based predictive)
**Provenance:** Zhang (2010), *A general multivariate framework for predicting
reserves*, and the multivariate chain ladder literature (Prohl & Schmidt).
This entry is the no-intercept special case whose point estimates collapse to
the volume-weighted chain ladder per line, plus a contemporaneous cross-line
error correlation - "chain ladder + dependence", estimated jointly.

## Model

One regression system per development transition `d -> d+1`, jointly over the
`K` lines of business of a single company:

```
C_{k,w,d+1} = b_{k,d} * C_{k,w,d} + e_{k,w,d}          (default: no intercept)
Var(e_{k,w,d})            = sigma^2_{k,d} * C_{k,w,d}   (Mack's 1/C weighting)
Corr(e_{k,w,d}, e_{l,w,d}) = R_d[k,l]                    (same origin, across lines)
```

Whitening each equation by `1/sqrt(C_d)` makes the system homoskedastic; the
whitened residual covariance `Sigma_d` (diag `sigma^2_{k,d}`) carries the
cross-line dependence. With `intercept=True` the design becomes `[1, C_d]`
(Zhang's general form) at the cost of one extra parameter per line per
transition. Two columns need at least two origin pairs at every development
step, and a square triangle's last step has exactly one, so `intercept=True`
there is refused by name rather than fitted.

## Estimation

Hand-rolled iterated **feasible GLS** per transition (`model.py:_fgls`):
whitened per-line OLS -> residual covariance `Sigma_d` -> stacked GLS with
`Omega = Sigma_d (x) I` -> iterate to convergence. No statsmodels or
linearmodels. `coef_cov` is the FGLS asymptotic covariance `(X' Omega^-1 X)^-1`.

### Small-sample ladder (explicit df guards)

The ladder is entered only by a transition that is identified at all: `n >= p`
usable origin pairs for `p` design columns. Below that the ladder has nothing
to degrade, so `fit()` refuses the transition by name. It bites only under
`intercept=True` (`p = 2`), where the single origin pair at a square triangle's
last step is one equation for two unknowns.

A transition with `n >= p` usable origin pairs is then estimated by:

| condition | slopes | Sigma_d | method tag |
|---|---|---|---|
| `n >= max(K + 2, p + 1)` | full FGLS | full residual covariance | `fgls` |
| `n - p >= 2` | per-line WLS | own variances x pooled correlation `R_bar` | `pooled_corr` |
| otherwise | per-line WLS | Mack tail rule variances x `R_bar` | `tail` |

`R_bar` is the correlation of standardized whitened OLS residuals pooled
across all transitions with `n >= 3`. Every covariance passes an
eigenvalue-floor PD repair (`_nearest_pd`); floored dimensions bias
correlations toward zero.

**On a 10x10 Schedule P square with K=4 lines, the full 4x4 covariance is
estimable only for the first ~4 transitions; later-transition correlations are
pooled, not estimated.** This is a documented limitation, not a bug.

## Point

`point()` rolls each origin forward from its latest observed diagonal with no
noise: `C_{d+1} = b_d C_d` per line, or `b0 + b1 C_d` under `intercept=True`.
That is exactly the conditional mean `predict()` builds before it adds a shock,
including that line's floor at zero, and both start from the same cell. It
returns the `kernels.multiline` target frame - per-(lob, origin) ultimates,
per-lob totals, grand total - with a `point` column, so a point board and a
draw board read the same rows. Reserves are the caller's subtraction: ultimate
minus the latest observed cumulative.

## Prediction

`predict()` simulates each origin forward from its latest observed diagonal:
per draw, one coefficient vector per transition sampled from
`N(beta_hat, coef_cov)` (parameter risk, common across origins - set
`param_uncertainty=False` for process-only), plus cross-line correlated
process noise `e_k = sqrt(C_k) * (L_d z)_k` with `L_d = chol(Sigma_d)`,
independent across origins and transitions. Ultimates are the simulated
cumulatives at the last development step.

Targets: per-(lob, origin) ultimates, per-lob totals, grand total (layout from
`kernels.multiline`). The grand-total spread vs the sum of per-lob spreads is
the diversification readout.

## Data contract

`kernels.multiline.multiline_data`: one company, >= 2 lines of business,
cumulative, identical observed-cell pattern across lines, positive
cumulatives (the 1/C weighting requires it). Default `loss_field="paid_loss"`.

## Limitations

- Additive Gaussian errors can simulate negative cumulatives on small books;
  draws are floored at zero, which truncates the left tail and biases means
  slightly upward.
- Late-transition cross-line correlations come from the pooled `R_bar`
  (assumed, not estimated); with very short triangles `R_bar` may itself rest
  on few residuals.
- No tail development beyond the last observed dev step (as with the chain
  ladder it generalizes).
- `intercept=True` needs a wider triangle than the default does: a square
  triangle's last development step has one origin pair against two columns, so
  the fit is refused rather than answered. One extra origin period is enough
  (7 origins x 6 development steps leaves two pairs at the last step).
- The eigenvalue floor repairs near-singular covariances at the price of
  attenuating extreme correlations.
