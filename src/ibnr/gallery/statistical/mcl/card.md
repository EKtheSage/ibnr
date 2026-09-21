# mcl - the full-matrix multivariate chain ladder

**Family:** statistical (frequentist stochastic; simulation-based predictive)
**Provenance:** Zhang (2010), *A general multivariate framework for predicting
reserves*, in its general form: every line's next cumulative is regressed on
every line's current cumulative. `sur` is the special case where that
coefficient matrix is diagonal.

## Model

One system per development transition `d -> d+1`, jointly over the `K` lines of
business of a single company:

```
C_{k,w,d+1} = sum_l B_d[k,l] * C_{l,w,d} + e_{k,w,d}
Var(e_{k,w,d})             = sigma^2_{k,d} * C_{k,w,d}   (Mack's 1/C weighting)
Corr(e_{k,w,d}, e_{l,w,d}) = R_d[k,l]                     (same origin, cross-line)
```

`B_d` is `K x K` with no intercept. Its diagonal plays the part of a
development factor and its off-diagonal entries let one line's emergence
predict another's, which is what separates this entry from `sur`: `sur` only
lets the two lines' errors be correlated.

Mack's weighting is applied per equation, so equation `k` is divided through by
`sqrt(C_{k,w,d})` - its own line's current cumulative. The other lines'
regressors are divided by that same number, not by their own.

## Estimation

The R `systemfit` conventions for `method = "SUR"`, reproduced so the entry
ties out to the reference R implementation (`model.py:system_estimate`):

1. OLS per equation on the whitened data.
2. Residual covariance from those OLS residuals, uncentred, with the `geomean`
   denominator `sqrt((n - p_k)(n - p_m))`. Every equation here has the same `K`
   columns, so it is `n - K`.
3. One feasible-GLS solve - no iteration to a fixed point. The stacked system
   has covariance `Omega = Sigma (x) I_n`, so block `(k, m)` of the normal
   matrix is `Sigma^-1[k,m] * (X_k' X_m)`.

`coef_cov` is `(X' Omega^-1 X)^-1`, row-major over `(equation, line)`, and is
what parameter risk draws from.

### When a transition is a system, and when it is not

A full `B_d` costs `K^2` coefficients where a diagonal one costs `K`. So a
transition is estimated as a system only when it has strictly more than
`min_obs_mult * K` origin pairs (default `min_obs_mult = 2`, the reference
implementation's). Otherwise, and when the solve returns a singular matrix or a
non-finite coefficient, the transition is the diagonal volume-weighted chain
ladder, `diag(sum(y_k) / sum(x_k))`. `transitions_[d]["method"]` says which
happened and `fallback_reason` says why.

The rule bites hard on a Schedule P triangle, and that is the truth of the
method on this data rather than a limitation of the port. A 10 by 10 square has
9 origin pairs at its first transition and 1 at its last, so:

| lines | transitions estimated as a system, of 9 |
|---|---|
| 2 | the first 5 |
| 3 | the first 3 |
| 4 | the first 1 |

For the draws, a fallback transition keeps each line's own whitened residual
variance while residual degrees of freedom remain, and imports the cross-line
correlation from `R_bar`, the correlation of standardized own-line residuals
pooled over every transition with at least 3 origin pairs. With no degrees of
freedom left the variances come from Mack's tail rule over the transitions
already fitted. Every covariance passes an eigenvalue-floor repair
(`kernels.multiline.nearest_pd`); floored dimensions bias correlations toward
zero.

## Point

`point()` is the vector recursion: each origin starts at its latest observed
diagonal, `C_{d+1} = B_d C_d` to the last development step. It returns the
`kernels.multiline` target frame - per-(lob, origin) ultimates, per-lob totals,
grand total - with a `point` column. Reserves are the caller's subtraction:
ultimate minus latest observed cumulative.

## Prediction

`predict()` simulates the same recursion. Per draw, one coefficient matrix per
transition: a system transition draws the whole flattened matrix from
`N(vec(B_d), coef_cov)`, a fallback transition draws only its `K` diagonal
factors and leaves the off-diagonal zeros alone, because those coefficients
were never estimated. `param_uncertainty=False` reuses the point estimate.
Process noise is `e_k = sqrt(C_k) * (L_d z)_k` with `L_d = chol(Sigma_d)`,
independent across origins and transitions.

Targets: per-(lob, origin) ultimates, per-lob totals, grand total (layout from
`kernels.multiline`). The grand-total spread against the sum of the per-lob
spreads is the diversification readout.

## Data contract

`kernels.multiline.multiline_data`: one company, >= 2 lines of business,
cumulative, identical observed-cell pattern across lines. Default
`loss_field="paid_loss"`. No premium anywhere in the API.

Every cumulative a development transition divides by must be positive - that
is, every cell with an observed successor. A cell on the latest diagonal is
never divided by, so a zero there is accepted and carried forward by the other
lines' coefficients. This is narrower than `sur`, which refuses any
non-positive cumulative: `sur` has no cross-line coefficient with which to
carry such a cell, and a zero latest diagonal would leave its whole origin at
zero.

## Limitations

- Late transitions are the diagonal chain ladder, so the cross-line
  coefficients this entry exists for are estimated only on the early part of
  the development, where the origin pairs are.
- Additive Gaussian errors can simulate negative cumulatives on small books;
  draws are floored at zero, which truncates the left tail and biases means
  slightly upward.
- Fallback-transition cross-line correlations come from the pooled `R_bar`
  (assumed, not estimated).
- The eigenvalue floor repairs near-singular covariances at the price of
  attenuating extreme correlations.
- No tail development beyond the last observed dev step, as with the chain
  ladder it generalizes.

## References

- Zhang, Y. (2010). A general multivariate chain ladder model. *Insurance:
  Mathematics and Economics* 46(3), 588-599.
- Henningsen, A. and Hamann, J. D. (2007). systemfit: A Package for Estimating
  Systems of Simultaneous Equations in R. *Journal of Statistical Software*
  23(4).
- Mack, T. (1993). Distribution-free calculation of the standard error of chain
  ladder reserve estimates. *ASTIN Bulletin* 23(2).
