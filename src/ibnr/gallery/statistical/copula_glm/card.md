# copula_glm — Copula-linked lognormal regressions (dependent loss reserving)

**Family:** statistical (frequentist stochastic; simulation-based predictive)
**Provenance:** Shi & Frees (2011), *Dependent loss reserving using copulas*
(ASTIN Bulletin 41(2)). This entry is their preferred lognormal case: one
parametric regression per line of business on incremental losses, joined by a
Gaussian copula on the residuals cell by cell.

## Model

Per line `k`, on incremental loss ratios of a single company:

```
log( incr_{k,w,d} / premium_{k,w} ) = x'_{w,d} beta_k + sigma_k eps_{k,w,d}
x_{w,d} = [1, origin dummies (w >= 2), dev dummies (d >= 2)]     (default)
        = [1, origin dummies, ln(d), d]                          (dev_effect="hoerl")
```

Dependence: within each (origin, dev) cell, `(eps_1, ..., eps_K)` follows a
**Gaussian copula** with correlation `R`; cells are independent of each other
(Shi & Frees' baseline dependence structure). On the standardized-residual
scale the copula reduces to a correlation matrix of normal scores.

The factor design spends `1 + (n_w - 1) + (n_d - 1)` parameters per line
(19 on a 10x10 square with 55 points — Shi & Frees' own budget). The Hoerl
curve option collapses the dev effects to 2 parameters for thin triangles.

## Estimation

- Marginals: per-line OLS on logs (lognormal regression needs no IRLS/GLM
  machinery). `sigma_k` from residuals with `df = n_obs - p`.
- Copula: Pearson correlation of standardized log-residuals across lines,
  matched on common cells, eigenvalue-floored to PD.
- Identification guards: every origin and (factor case) every dev step must
  have at least one usable cell; `n_obs > p` enforced with a pointer to
  `dev_effect="hoerl"`.

## Prediction

`predict()` simulates future increments cell by cell: correlated normals
within a cell across lines -> lognormal increments -> summed onto the latest
observed cumulative. Ultimates per (lob, origin), per-lob totals, grand total
(layout from `kernels.multiline`).

**Parameter uncertainty:** `param_uncertainty="bootstrap"` (default) refits
marginals + copula on `n_boot=200` triangles simulated from the fitted model
and assigns each predictive draw a random replicate's parameters — a
parametric bootstrap. Refits are one `p x n_obs` pseudo-inverse multiply per
replicate, so this costs milliseconds. `"plugin"` uses point estimates only
and visibly understates reserve variability at triangle sample sizes.

## Data contract

`kernels.multiline.multiline_data`: one company, >= 2 lines of business,
cumulative input, aligned observed cells, positive premium. Default
`loss_field="paid_loss"`: **incremental reported losses routinely go negative
at late lags (case releases), which a lognormal marginal cannot represent.**
`nonpositive="error"` (default) fails loudly, reporting the offending cell
count; `nonpositive="drop"` censors those cells in every line (documented as
biased — it truncates the left tail of the marginals).

## Extensions (scaffold ideas, not implemented)

- Student-t copula (tail dependence) — swap the Gaussian scores for t scores.
- Gamma GLM marginals — replace the OLS-on-logs with IRLS.
- Cell-level covariates (calendar-year effects) — extra design columns.

## Limitations

- Cell-level independence: dependence acts only within an (origin, dev) cell
  across lines, not along calendar years within a line.
- The copula correlation is treated as re-estimated per bootstrap replicate,
  but replicates are simulated from the point-estimate copula — correlation
  uncertainty is approximated, not fully propagated.
- The last origin's level effect rests on a single observed cell (inherent to
  the design, as in Shi & Frees).
- No tail development beyond the last observed dev step.
