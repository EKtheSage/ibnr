# Design: point scores, the TLRN entry, full-matrix MCL, and the reproduction notebook

Date: 2026-09-20. Status: approved 2026-09-20 (Ethan). Target release: 0.7.0.

## 1. Context

A reconciliation run on 2026-09-20 (`transformers_reserving/Code/marco_vs_ibnr.ipynb`,
ibnr 0.5.5) compared a colleague's Transformer Loss Reserving Network (TLRN, an R torch
implementation) with ibnr's `nn_transformer` on the same Schedule P data. The data
reconciled exactly and ibnr's Mack reproduced R's chain ladder to 4e-8. The model
comparison did not transfer: TLRN blended with the multivariate chain ladder (MCL)
scored a company-reserve Pool_APE of 5.088 percent against 5.661 percent for chain
ladder plus MCL, while `nn_transformer` scored 60.3 percent paid-only and 84.1 percent
with incurred and case inputs, on the same 82 companies.

The reconciliation had to leave R running for three reasons: ibnr has no point-error
metrics, no development-factor network, and no MCL. This document designs the pieces
that remove those reasons, so that the whole study runs through `ibnr.gallery`, and
the notebook that reproduces the study on the same company set with the same training
protocol.

Everything below was checked against the R code (`05_transformers.R`,
`01_functions.R`, `04_traditional.R`, `06_uncertainty.R` at commit `0d21d0e`) and
against ibnr 0.6.0 as it stands on `main` at 482ff9b.

## 2. Goals

1. Point-error metrics (Pool_APE, Pool_PE, MAE, RMSE, and the secondary ones the R
   study reports) implemented once in `kernels/`, with the aggregation level explicit,
   reachable from `evaluate()` and from a cross-model reserve table.
2. A `tlrn` gallery entry that reproduces the R model's architecture, features,
   objective, checkpoint protocol, seed selection, factor fallback and historical
   residual calibration, on ibnr's data contract.
3. An `mcl` gallery entry, the full-matrix multivariate chain ladder, whose point
   reserves tie out to the R `systemfit` implementation on the 82 reconciled companies.
4. The shared NN training loop and contract extended so that a new architecture is a
   directory holding a network, a head, a loss and a config, with everything else
   reused. No formula DSL (decision 6).
5. A standalone notebook that fits every row on the R study's company set (93
   companies, 243 company-line pairs, accident years 1998 to 2007, valuation
   2007-12-31), under the R study's protocol (10 seeds, 3000 epochs, validation
   checkpoint selection), and reports CRPS, PIT and the point metrics at three levels.

## 3. Non-goals

- Per-cell predictive draws for `tlrn`. Its uncertainty is calibrated at company
  reserve level, as in the R study, and that is where it is scored. Spreading company
  draws over cells would invent cell uncertainty the method never claimed.
- The R study's Cape Cod variants (`IBNR_CC`, `IBNR_CC_SUR`, `IBNR_CC_MCL`). Not needed
  for the blend or the headline.
- A full run-off over-dispersed Poisson bootstrap (the R study's uncertainty
  comparator). `kernels/odp_bootstrap.py` draws one diagonal, because that is all the
  one-year CDR reads. Out of scope; the notebook says so where the comparison would go.
- Re-tuning the shrinkage weight 0.658. The R repository supplies the constant with a
  comment that it was selected on historical origins, and no search code. The notebook
  reproduces the constant and says so.
- A hosted or parallel training service. Member-parallel training is a local option
  measured in this work, not a new harness.

## 4. Vocabulary

- **Cohort set**: the fixed list of companies and company-line pairs every model is fit
  and scored on. Never shrunk silently when a model refuses a fit.
- **Level**: the unit errors are summed within before an absolute value is taken:
  `cell`, `pair` (company-line), or `company`. Summing before the absolute value is
  where cancellation between lines happens; the level is therefore always named.
- **Point**: a deterministic forecast number. Every entry has draws; some also have a
  native point (`mack`, `mcl`, `sur`, `tlrn`). The reserve table records which was used.
- **Reserve**: predicted or realized ultimate at the grid endpoint minus the cumulative
  observed at the valuation date (the anchor). The R study's target is outstanding paid
  through 120 months, so "ultimate" everywhere below means the 120-month endpoint.

## 5. Design

### 5.1 `kernels/point_scores.py`

Plain numpy and pandas, no torch, no gallery import.

```python
def point_metrics(predicted, actual, *, weights=None) -> dict
```

Over aligned 1-D arrays of one level's units. Returns `n`, `mae`, `rmse`, `wrmse`
(actual-weighted root mean square error, weights `actual / sum(actual)`), `mape` and
`medape` (over units with nonzero actual), `pool_ape` (`sum |e| / sum actual`),
`pool_pe` (`sum e / sum actual`) and `prop_over`. Refuses any non-finite `predicted`
(the R rule: a missing forecast on a fixed cohort set is an error, not a smaller
cohort set) and `sum(actual) <= 0`.

```python
def level_errors(frame, *, predicted, actual, level) -> pd.DataFrame
```

Sums `predicted` and `actual` within `level` (a list of columns) and returns one row per
unit with `error = predicted - actual`. `point_metrics` over its rows is the level's
score. The docstring carries the worked cancellation example from the reconciliation
(company 671: line errors +15.03, +327.71, -854.36, +501.97; company error -9.66; sum
of absolute line errors 1699.07).

```python
def shrink_toward(point, baseline, alpha) -> np.ndarray
```

`baseline + alpha * (point - baseline)`, with `alpha` in `[0, 1]` checked. The R study's
final estimator is this with `alpha = 0.658`, `baseline` the MCL reserve.

```python
def reserve_rows(entry, full_triangle, *, as_of, point="draw_mean" | "native",
                 segment=None) -> pd.DataFrame
```

One row per cohort of the entry (or the named one): the entry's segment columns,
`predicted_ultimate`, `realized_ultimate`, `anchor`, `predicted_reserve`,
`actual_reserve`, `premium`, `n_draws`, `point_source`. `anchor` is the sum over the
cohort's origins of the cumulative at `as_of`, read from `full_triangle.as_of(as_of)`
for the entry's loss field. With `point="native"` the entry must implement
`point(segment)` (section 5.5); otherwise the draw mean of `predict()` is used, and the
row says which. Rows are what `level_errors` and `point_metrics` consume; the notebook's
company board is `point_metrics(level_errors(rows, level=["company_code"]))` per model.

`GalleryEntry.evaluate()` gains a `"point"` key beside `summary`, `percentiles` and
`crps`: per-target `error` and `pct_error` (NaN where the outcome is 0 or missing), and
`point_metrics` over the targets whose label is not `"total"`, because the total is the
sum of the others and would count every error twice.

Exports: `point_metrics`, `level_errors`, `shrink_toward`, `reserve_rows` on
`ibnr.kernels`. On `ibnr.gallery` only `reserve_rows`, by the decision 8 rule: it is the
one a caller must call to get from fitted entries to a published point board.

### 5.2 `gallery/nn/_training.py`: the training scheme

`train_ensemble` keeps its signature and gains keyword-only, defaulted arguments. Every
default reproduces the current loop bit for bit; a fixed-seed test pins the final
weights of a tiny model against a hash recorded from `main` before the change.

- `schedule: Callable[[int], float] | None = None`. Multiplier on every parameter
  group's base learning rate at epoch `e` (1-based). `warmup_cosine(schedule_epochs,
  warmup=20)` is the R study's `learning_rate_multiplier`, ported with its edge cases
  (one-epoch run still updates; warmup capped at `schedule_epochs - 1`; cosine finishes
  at zero).
- `param_groups: Callable[[model], list[dict]] | None = None`. Returns AdamW parameter
  groups with their own `lr`. `None` is today's single group. The R study trains `phi`
  at ten times the network rate.
- `min_epochs: int = 0`, `check_every: int = 1`. Validation runs every `check_every`
  epochs and on the last; early stopping cannot fire before `min_epochs`. `patience` is
  counted in checks, which at `check_every = 1` is today's count in epochs.
- `cutoff_sampling: "per_example" | "per_epoch" = "per_example"`. Today one augmented
  cutoff is drawn per cohort per batch. The R study draws one cutoff per epoch for every
  batch of that epoch. Both are drawn from the same member generator so streams stay
  aligned; `per_epoch` consumes one integer per epoch.
- `keep: int | None = None`. After all members train, keep the `keep` members with the
  lowest best validation score (ties by member index), in member order. `None` keeps
  all, today's behaviour. The R study trains 10 and keeps 2.
- `n_workers: int = 1`. Members are independent given their seeds, so they can train in
  spawned processes with one torch thread each. Whether the result is bit-identical to
  sequential training is measured, not assumed; if it is not, the card of every entry
  that uses it says so and the notebook records which was used.

`train_ensemble` returns `(models, histories)` as today; the per-member best validation
score is the minimum finite `val` in each history, which `kernels.tuning.validation_score`
already reads.

### 5.3 `kernels/residual_calibration.py`

The R study's uncertainty, written once for any point forecaster.

```python
def rolling_residuals(forecast_at, actual_at, *, cutoffs, size, scale_floor=(0.01, 1.0))
    -> pd.DataFrame
```

For each cutoff `K` calls `forecast_at(K)` and `actual_at(K)`, both returning one value
per unit in a fixed unit order, and records `unit`, `cutoff`, `horizon`, `predicted`,
`actual`, `size`, `residual_scale = max(|predicted|, 0.01 * size, 1)` and
`standardised_error = (actual - predicted) / residual_scale`. The units here are
companies and `size` is company premium; nothing in the function knows that.

```python
def calibrate(residuals, *, horizons, n_strata=4, min_per_stratum=40) -> Calibration
```

Keeps the horizons named, assigns each unit a size stratum by quantile of `size`, centres
each stratum's errors on its median, refuses a stratum with fewer than `min_per_stratum`
residuals (the R check). `Calibration` carries the pools and the stratum edges.

```python
def calibrated_draws(calibration, *, point, size, n_draws, rng) -> np.ndarray
```

`(n_draws, n_units)`: `point + scale * resampled centred residual` from the unit's
stratum, `scale = max(|point|, 0.01 * size, 1)`.

Also `leave_one_out_coverage(residuals, calibration, levels)`, the R study's
cross-company historical coverage table, since it costs nothing once the pools exist.

The R study calibrates on horizons 3 to 5 from cutoffs 5 to 9 using the SAME
checkpoints applied at earlier cutoffs, so some calibration residuals are in-sample for
the network (cutoffs 5 to 7 predict cells that were training targets). That is the R
study's choice; the kernel takes `cutoffs` and `horizons` as arguments and the `tlrn`
card discloses the overlap.

### 5.4 Contract addition and the feature builder

`kernels.nn_contract.nn_data` and `nn_company_data` gain one key, `values`: the field's
raw grid as the triangle reported it, `(n_c, n_f, n_w, n_d)` and `(n_c, L, n_f, n_w,
n_d)`, NaN where absent, in dollars. `x` stays the increment (or level) ratio. Additive;
no existing consumer changes. The feature builder needs cumulative amounts and
cumulative loss ratios at arbitrary visible cells, and a cumulative sum of increments
over padding would fabricate them.

`kernels/nn_features.py`:

```python
def tlrn_features(contract, *, cutoff, clamp=5.0) -> dict
```

From a company contract and a 1-based cutoff diagonal `K`, builds the R study's example
tensors for every (company, origin): `feat (n_c * n_w, L * n_d, n_feat)`, `target`,
`target_mask`, `premium`, `c_lk`, `anchor_start`, `p_lk`, `lk_ix`, `nobs_ix`,
`fallback_logf`, `anchor_logf`, `factor_support` inputs, and `n_dropped`. Feature
order and definitions follow `build_examples` exactly: 8 channels with paid only, 13
with incurred and case. Every statistic (per-line-per-lag mean and sd of increment
ratios, own increment-to-cumulative ratios, latest cumulative ratios) is computed from
cells on or before `K` only, clamped at `clamp` standard deviations. Origins with no
visible cell are kept in the tensors and dropped from the loss, as in R.

The R study's perturbation check becomes a test: multiply every hidden cell (calendar
index above `K`) by 1.37 and add 0.123, rebuild, and every input tensor must be
unchanged while the targets differ. This is the check that the features leak nothing.

`nn_features` imports no torch; the entry converts.

### 5.5 `gallery/nn/tlrn/`

Directory: `card.md`, `config.py`, `network.py`, `head.py`, `model.py`. Torch imported
inside fit and predict only.

**Data.** `nn_company_data(loss_field="paid_loss", feature_fields=(), ...)` for the
8-feature form; `feature_fields=("incurred_loss", "case_reserve")`,
`level_fields=("case_reserve",)` for the 13-feature form. The mart's `case_reserve` is
`incurred - paid - bulk`, which is the R study's `cum_am_case` (verified in the
reconciliation). A cohort is a company; each (company, origin) is one training example
of `L * n_d` tokens.

**Network** (`network.py`). `TLRN(nn.Module)`: `Linear(n_feat, d)` plus line, lag and
observed-count embeddings; `n_layers` axial blocks, each pre-LayerNorm attention across
lines within a lag (key padding on unwritten lines) then across lags within a line then
a GELU feed-forward of width `2d`, with residual paths; LayerNorm and `Linear(d, 1)`
head initialised at 0.01 times default weights and zero bias; `phi` parameter of shape
`(L, n_d - 1)` initialised so that `softplus(phi)` is `init_step / m` at step `m`.
Attention weights are returnable for the interpretability read-out.

**Head** (`head.py`). Free functions over tensors: `log_factors(phi, net, eps)` is
`softplus(phi + eps * net)`; `apply_support(logf, support, fallback)` substitutes
as-of-cutoff chain ladder log factors where no training target ever supervised a step;
`anchor(logf, anchor_logf, initial, width)` is the optional chain-ladder-anchored
variant `anchor + width * tanh((logf - initial) / width)`; `project(logf, c_lk, lk_ix,
p_lk)` is the cumulative projection `C_j = C_lk * exp(S_j - S_lk)` with observed cells
held, returning predicted increment ratios. The `factors_only`, `cl_anchor` and
`cross_line` switches are kept so each design choice can be compared with and without.

**Config** (`config.py`), defaults the R study's checkpoint protocol: `d_model=32`,
`n_heads=2` (1 when `d_model < 16`), `n_layers=1`, `dropout=0.3`, `eps=0.5`,
`init_step=0.57`, `cross_line=True`, `factors_only=False`, `cl_anchor=False`,
`anchor_width=0.1`, `tail_policy="observed_cl"`, `loss_aggregation="ay_line"`,
`w_pe=0.5`, `w_mse=0.1`, `mse_scale=0.005`, `lr=3e-3`, `lr_phi=3e-2`, `warmup=20`,
`batch_size=64` (examples, not companies), `max_epochs=3000`, `min_epochs=500`,
`check_every=5`, `patience_checks=600`, `keep=2`, `val_diagonals=2`,
`min_cutoff=2`, `cutoff_sampling="per_epoch"`, `ensemble_size=10` (the loop reads this name; it is the seed count), `calibration_cutoffs=(5, 6, 7, 8, 9)`,
`calibration_horizons=(3, 4, 5)`, `n_strata=4`, `n_draws=4000`, `n_workers=1`.
`patience_checks=600` at `check_every=5` never fires inside 3000 epochs, which is what
the R run did (every seed ran 3000 epochs; the retained state is the best validation
checkpoint). Early stopping is the notebook's choice to make, and it records it.

**Fit.** Validation split: `splits` with `val_diagonals=2` gives training targets on
diagonals up to `c_max - 2` and validation targets on the last two. Training cutoffs
run from `min_cutoff` to `c_max - 3` inclusive; validation is scored at cutoff `c_max
- 2` predicting the last two diagonals and at cutoff `c_max - 1` predicting the last
one, the R study's two validation sets. Feature tensors are built once per cutoff and
indexed by the loop's drawn cutoff. Loss per batch: AY/line APE with minibatch
denominators plus `w_pe` times pooled PE plus `w_mse` times masked MSE over
`mse_scale`, the `train_checkpoint_tlrn` objective. `factor_support` is computed from
the training masks once per fit. Members are seeds; `train_ensemble(keep=2)` keeps the
two with the lowest validation AY/line APE.

**Predict.** `point(segment)` returns the deterministic per-(line, origin) ultimates,
per-line totals and grand total of one company in the `kernels.multiline` layout, the
mean over the kept checkpoints. `predict(segment)` returns a `PredictiveDistribution`
with ONE target, the company's total ultimate, whose draws come from
`residual_calibration.draw` around the point, calibrated once per fit by applying the
kept checkpoints at `calibration_cutoffs` (forward passes only, no refits) and pooling
by company premium quartile. `predict()` without a segment returns every company's
total. `cohorts()` lists companies, as `nn_transformer_ml` does.

**Held-out capability.** Neither mixin. `tlrn` does not join the per-cell CRPS board
(non-goal). It joins the point board through `reserve_rows(point="native")` and the
company-level CRPS and PIT through `predict`.

**Card.** States what is the R study's and what is not: the distribution is
historically calibrated, not native; calibration residuals overlap training targets at
the earlier cutoffs; the blend weight is supplied, not searched; `mcl` and not the R
`systemfit` code is the blend baseline; seeds in torch and in R torch differ, so a
reproduction reproduces the protocol and the magnitude, never the digits. Cites Zhang
(2010) for MCL and the companion manuscript for TLRN (attribution wording is an open
question, section 9).

### 5.6 `gallery/statistical/mcl/`

Full-matrix multivariate chain ladder: per development transition `d`, each line's next
cumulative is regressed on EVERY line's current cumulative, one equation per line,
Mack-weighted by the equation's own current cumulative, estimated as a system.

```
C_{k,w,d+1} = sum_l B_d[k,l] * C_{l,w,d} + e_{k,w,d}
Var(e_{k,w,d}) = sigma^2_{k,d} * C_{k,w,d}
```

Zhang's (2010) general form with a full `B_d`; `sur` is the diagonal special case.

**Estimator.** The R `systemfit` SUR conventions, so that the entry ties out: per-line
whitened OLS, residual covariance with the `geomean` denominator
`sqrt((n - p_i)(n - p_j))`, ONE feasible GLS step (no iteration), coefficient
covariance `(X' (Sigma^-1 x I) X)^-1`. A transition is estimated as a system only when
`n > min_obs_mult * K` origin pairs (default `min_obs_mult = 2`); otherwise, and when the
system estimate is non-finite, the transition falls back to the diagonal
volume-weighted chain ladder. The last transition of a square triangle is the single
observed ratio per line. On a 10 by 10 square with four lines only the first transition
is a system estimate; the card says so, because it is the truth of the method rather
than a limitation of the port.

**Point.** Vector recursion from each origin's latest diagonal: `C_{d+1} = B_d C_d`.
`point()` returns the `kernels.multiline` layout.

**Draws.** As `sur`: parameter risk from `N(beta, coef_cov)` per transition, correlated
process noise `sqrt(C) * (L_d z)` with `L_d` the Cholesky factor of the residual
covariance, independent across origins and transitions; fallback transitions use
per-line Mack variances and the pooled residual correlation. Draws are floored at zero,
as `sur` does, and the card carries the same caveat.

**Contract.** `kernels.multiline.multiline_data`, one company, at least two lines,
positive cumulatives. The 82 reconciled companies have positive paid cumulatives on
every line (the 11 excluded ones failed the R Mack on a line with a zero or negative
cell), so the tie-out set needs no relaxation. Companies the contract refuses are
refused by name.

**Tie-out.** `tests/data/marco_reconciliation_classical.csv` vendors the R replay's
company table (93 rows; `IBNR_CL`, `IBNR_CL_MCL`, `IBNR_obs`, `point_ok`), produced from
ibnr-loaded data on 2026-09-20. A `mart` test asserts `mcl`'s point reserve equals
`IBNR_CL_MCL` on the 82 `point_ok` companies to 1e-6 relative, and `mack`'s equals
`IBNR_CL`. The remaining 11 rows are listed with the reason the contract refuses them.

### 5.7 Public surface and version

New on `ibnr.kernels`: `point_metrics`, `level_errors`, `shrink_toward`,
`reserve_rows`, `rolling_residuals`, `calibrate`, `calibrated_draws`, `Calibration`, `warmup_cosine`. New on `ibnr.gallery`:
`reserve_rows`. New entries: `tlrn` (family `nn`), `mcl` (family `statistical`).
`GalleryEntry.evaluate()` output gains a key. `tests/test_entry_contract.py`'s buckets
gain both names. Minor version bump to 0.7.0 under the CHANGELOG's own rule.

### 5.8 Notebook: `analysis/04_nn_architectures_vs_classical.ipynb`

A new file. `03b` stays as the ten-company study pinned to 0.5.6 and `03c` as the
multi-line study pinned to 0.5.8; `analysis/README.md` gains the new row.

Standalone: `pip install "ibnr[nn]==0.7.0" cas-schedule-p==2026.6.13 matplotlib`,
version and publish asserted, cache seeded from the wheel, no repository paths. Runs
from any directory.

**Cohort set.** The R study's `select_cohort(cutoff=5)` ported to pandas: complete 10 by
10 company-line grids with positive premium on every origin; finite positive
cumulative paid and cumulative loss ratio below 3 on calendar indices up to 5; at least
two eligible lines per company. Asserted: 668 input pairs, 414 after completeness, 330
after loss eligibility, 243 pairs over 93 companies. The 82-company point board is the
R study's `point_ok` set, vendored as a constant and re-derived as "`mack` fits every
line of the company".

**Window.** Accident years 1998 to 2007, valuation 2007-12-31, `dev_lag` in months
(the R `DevelopmentLag` 1 to 10 times 12). Targets: the 2008 diagonal (9 cells per
pair), the 120-month endpoint per pair, and the company reserve summed over lines.

**Rows.** `mack` per pair; `mcl` and `sur` per company; `nn_transformer` paid-only and
with incurred and case inputs; `nn_transformer_ml`; `deeptriangle`, `mdn`, `resnet` at
their 3b budgets; `tlrn` 8-feature and 13-feature under the R protocol; `tlrn` blended
toward `mcl` with `alpha = 0.658`. Every fit under one `PROTOCOL` switch, `"published"`
(the R budgets) or `"smoke"` (a two-seed, few-epoch run that must not be quoted), and
the notebook prints which ran.

**Scores.** At the next diagonal and the endpoint: CRPS and PIT for entries with cell
draws, point metrics for every entry with a point (including `tlrn`'s cell points). At
company level: point metrics for every row, and for every entry with draws CRPS, PIT,
80 and 95 percent interval coverage and interval score, and the leave-one-out historical
coverage for `tlrn`. Dollar and premium-normalised views side by side, with the
concentration of absolute error by company shown (company 1767 carried 75 percent of
the paid-only transformer's error in the reconciliation).

**Reproduction target.** The R headline (5.088 versus 5.661 percent, a 10.12 percent
relative reduction on 82 companies) is printed beside the notebook's own `tlrn` blend
and `mcl` numbers. `mcl` must match to 1e-6 relative. `tlrn` reports its own number; a
gap is a finding.

**Diagram.** One matplotlib figure, two columns: `nn_transformer` (cell tokens, encoder,
mixture head, diagonal rollout) and `tlrn` (company-origin examples of line-by-lag
tokens, axial attention, factor head, cumulative projection, checkpoint ensemble,
residual calibration, blend). Boxes and parameter counts read from the live module
trees of the fitted networks, so nothing in it is typed.

**Inside `tlrn`.** A section like 3b's transformer section: the feature grid at one
cutoff, the axial block's two attention patterns on a live forward pass, the learned
factors against chain ladder factors, and the chain-ladder reproduction test run
inline.

## 6. Data flow

```
Triangle (as_of)
  -> nn_company_data(...)                       contract with values, x, masks, premium
  -> nn_features.tlrn_features(contract, K)      one tensor set per cutoff K
  -> train_ensemble(keep=2, schedule, groups)    10 seeds -> 2 checkpoints
  -> head.project(...) at K = as_of              point per (company, line, origin)
  -> calibrated_draws over K in 5..9              company-level draws
  -> PredictiveDistribution (company total)  +  point() layout
  -> reserve_rows(...) -> level_errors(...) -> point_metrics(...)
  -> shrink_toward(tlrn point, mcl point, 0.658)
```

## 7. Refusals and error handling

- `point_metrics`: non-finite forecast, non-positive total actual, misaligned lengths.
- `level_errors`: a level column missing from the frame; a unit with no actual.
- `reserve_rows(point="native")` on an entry without `point()`: refused by name.
- `train_ensemble`: `keep > ensemble_size`; `min_epochs > max_epochs`; `check_every <
  1`; a `schedule` returning a non-finite or negative multiplier; `n_workers > 1` with
  a model that cannot be constructed in a spawned process (the error names the member).
- `tlrn_features`: a cohort with a hole inside its visible region (the R data has
  none; a cumulative over padding is fabricated development); a cutoff outside `[1,
  n_d]`; a company with no written line.
- `tlrn.fit`: fewer than `keep` seeds; validation diagonals fewer than 2 when the
  protocol needs both validation sets; a stratum below `min_per_stratum` at
  calibration, naming the stratum and the count.
- `tlrn.predict_at` / `log_lik_at`: not present, so an `isinstance` check reads the
  entry as neither capability and the board records `no_cell_sampler` /
  `no_predictive_density`, as it does for `sur`.
- `mcl.fit`: the `multiline_data` refusals unchanged; a transition whose system
  estimate is non-finite falls back and the fit records the fallback per transition.

## 8. Testing

- `tests/test_point_scores.py`: closed forms on hand-built frames (the company 671
  example), refusal cases, `shrink_toward` at `alpha` 0 and 1, `evaluate()`'s new key on
  a tiny entry, the total row excluded. Mutation-checked: swap sum-then-abs for
  abs-then-sum and the 671 example must fail.
- `tests/test_nn_training_scheme.py`: bit-identical defaults against the recorded
  hash; `warmup_cosine` against the R function's values at epochs 1, warmup, and the
  end; `keep` selects by validation and preserves member order; `per_epoch` draws one
  cutoff per epoch; parameter groups receive their own rates; `n_workers=2` against
  sequential, with the result recorded either way.
- `tests/test_residual_calibration.py`: a synthetic point forecaster with known error
  law recovers its quantiles; stratum refusal; leave-one-out coverage on a case with
  known answer.
- `tests/test_nn_features.py`: the perturbation check; feature values against a
  hand-computed 2-line, 4-lag example; the cumulative-over-padding refusal.
- `tests/test_tlrn.py`: the chain-ladder reproduction (phi at inverse softplus of the
  as-of chain ladder log factors, network output zeroed, projection equals the direct
  chain ladder to 1e-3 relative, and the `factors_only` switch gives the same);
  determinism by seed; `predict` returns one target per company; `point` layout;
  `cohorts`; factor support fallback fires where no target supervised a step; the
  13-feature form changes the input width and nothing else; parameter count 14,309 with 13 features at
  `d_model=32` on a 4-line 10-lag grid, pinned in `test_nn_parameter_counts.py`.
- `tests/test_mcl.py`: FGLS against a hand-solved two-line system; the fallback rule
  by transition; the tie-out (`mart` marker) on the vendored 82 companies for `mcl` and
  `mack`.
- `tests/test_entry_contract.py`, `test_gallery.py`, `test_import_purity.py`: the two
  new entries register without torch, land in the right buckets, and carry cards.
- Every new test file is mutation-checked before it is committed: each claimed
  refusal or identity is broken on purpose once and the test must go red.

## 9. Sequence and open questions

Pull requests, each on its own branch and worktree, landing on `main` in order:

1. A: `point_scores` and `evaluate()` (independent).
2. B: training scheme extensions and `residual_calibration` (independent of A).
3. C1: contract `values` key and `nn_features` (depends on nothing; small).
4. C2: `tlrn` entry, card, tests (depends on B and C1).
5. M: `mcl` entry, card, tie-out (independent; can run beside C2).
6. R: release 0.7.0.
7. N: notebook 04 and the README row (depends on R for the pin; developed against the
   local checkout).

A and B run in parallel with two workers; then C2 and M in parallel.

Decisions taken 2026-09-20 (Ethan):

1. Attribution: neutral. The `tlrn` card and the notebook describe the method as "the
   TLRN protocol of a companion manuscript in preparation, reproduced from its R
   implementation" and name nobody, until the manuscript is published. No private
   repository URL anywhere in this repository.
2. Early stopping: the entry default is the R behaviour (validation every 5 epochs,
   best checkpoint restored, `patience_checks=600`, so no stop inside 3000 epochs). The
   notebook runs that default AND a `patience_checks=120` (600 epochs) variant and
   reports both, side by side.
3. Notebook: new file `analysis/04_nn_architectures_vs_classical.ipynb`; `03b` and
   `03c` untouched.
4. Execution: A and B in parallel with two Opus workers; C1 and C2 after B; M beside
   C2; then the release and the notebook.

## 10. Risks

- Runtime. Ten seeds at 3000 epochs over 930 examples of 40 tokens in Python torch on
  CPU is unmeasured. First task of C2 is to time one seed; the `n_workers` option and
  the `PROTOCOL` switch exist so the notebook can still be executed end to end.
- Reproduction gap. R torch and Python torch draw different random streams, so the
  `tlrn` Pool_APE will not equal 5.088 percent. The notebook states the protocol match
  and reports the gap; equality is claimed only for `mcl` and `mack`.
- Calibration overlap. The R calibration residuals at cutoffs 5 to 7 are partly
  in-sample. Reproduced as specified and disclosed; a non-overlapping variant is one
  config change (`calibration_cutoffs=(8, 9)`) and is worth a row in the notebook.
- `systemfit` conventions. The tie-out depends on matching its residual covariance
  denominator and single FGLS step exactly. The vendored R output is the arbiter; if a
  convention differs, the entry follows R and the card names the convention.
