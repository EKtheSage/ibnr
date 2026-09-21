# mcl - the full-matrix multivariate chain ladder

**Family:** statistical (frequentist stochastic; simulation-based predictive)
**Provenance:** Zhang (2010), *A general multivariate chain ladder model*, in
its general form: every line's next cumulative is regressed on every line's
current cumulative. `sur` is the special case where that coefficient matrix is
diagonal.

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

Only step 3 reaches the point estimate. Multiplying the residual covariance by
a constant divides the normal matrix and its right-hand side by the same
constant, so the coefficients do not move - the `geomean` denominator changes
`coef_cov`, and therefore the width of the predictive distribution, and nothing
else. The tie-out below cannot see it; the unit test in `tests/test_mcl.py`
that solves a two-line system by hand can, and does.

### A matrix that is singular to working precision

Both of the solves above refuse a matrix whose reciprocal condition number is
below machine epsilon, which is the rule R's `solve` applies and the tolerance
`systemfit` leaves it at. This is not a nicety. A line whose paid loss has
stopped developing has an exactly zero residual in every origin, so the
residual covariance has a zero row and column; `numpy` inverts such a matrix
happily unless a pivot is exactly zero, and the coefficients that come back are
rounding error multiplied by 1e30 - finite, plausibly sized and wrong. Six of
the 82 companies in the tie-out below have such a line, and before the check
they were the only six that did not tie out. On one of them the reserve came
out at 558 against the reference's 2600.

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

## Tie-out

`tests/test_mcl_tieout.py` refits the companion study's R replay: 93 multi-line
Schedule P companies at 31 December 2007, accident years 1998 to 2007, paid
loss, publish `20260613_041006`. The reserve compared is ultimate minus the
latest observed cumulative, summed over lines. The R company tables are
vendored at `tests/data/tlrn_study_company_reserves.csv` and
`tests/data/tlrn_study_pairs.csv`; `tests/data/README.md` says how they were
produced.

On the 82 companies where every R method returned a finite number, `mcl`
reproduces the R reserve to 1.3e-12 relative at worst (1.1e-06 in USD
thousands), against a tolerance of 1e-6 relative. `mack` is checked in the same
test and reproduces the same study's chain ladder to 8.6e-15 relative; it is
the control, because the two implementations have to agree on the data and on
the reserve definition before agreeing on the harder estimator means anything.

The other 11 companies are the ones the R chain ladder could not score, on a
line with a zero or negative paid cell. This entry refuses all 11 by name, for
the same reason, and the test asserts the refusals rather than skipping them.

Two conventions had to be matched before the 82 agreed, and both are recorded
above: the singular-matrix rule, which accounts for 6 of them, and the scope of
the positivity refusal, which accounts for 2 more. The third possible
difference, the residual covariance's denominator, turns out not to reach the
point estimate at all.

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
