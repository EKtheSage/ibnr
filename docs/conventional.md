# Conventional reserving candidates

`ibnr.kernels.conventional` implements point forecasts for chain ladder (CL),
Bornhuetter-Ferguson (BF), and Gluck's generalized Cape Cod (GCC). The formulas
follow sections 2.3.1-2.3.3 of [The Actuary and IBNR Techniques: A Machine
Learning Approach](https://ssrn.com/abstract=3697256) by Caesar Balona and
Ronald Richman, the 14 August 2020 manuscript. These candidates support
research into forecasting procedures.

```python
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional

candidate = ConventionalCandidate(
    method="gcc",
    decay=0.75,
    history_periods=5,
    horizon=120,  # months; fixed final development age, with no additional tail
)
fit = fit_conventional(tri, candidate, as_of="2022-12-31")
print(fit.origins[["origin_period", "ultimate", "reserve"]])
print(fit.factor_selection)
print(fit.factor_summary)
```

## What is fixed and what is estimated

The frozen `ConventionalCandidate` specifies method, averaging rule, history
window, exclusions, horizon, and fallback policies. BF also fixes an a priori
expected loss ratio; GCC fixes a decay parameter. A refit recalculates factors,
development proportions and GCC loss ratios from the information available at
its cutoff. It does not change these settings.

`conventional_grid` builds an explicit Cartesian grid over history windows,
high/low exclusion flags, BF loss ratios and GCC decays. It always includes CL;
empty loss-ratio/decay sequences omit the corresponding family. There is no
default grid purporting to reproduce every published example.

## Formulas

Let `C_i` be the latest cumulative loss, `P_i` premium, and `beta_j` the
proportion developed at age `j`: the reciprocal product of subsequent selected
factors, ending at `beta_horizon = 1`.

- **CL:** prior ultimate `A_i = C_i / beta_i`.
- **BF:** `A_i = P_i × expected_loss_ratio`.
- **GCC:** `A_i = P_i × LR_i`, where
  `LR_i = sum_n(C_n × T_n × decay^distance(i,n)) / sum_n(P_n × beta_n × decay^distance(i,n)) / T_i`
  and `T_n = (1 + trend)^(m_n / 12)`, with `m_n` the whole months from the end
  of origin period `n` to the cutoff.

All three have ultimate `C_i + A_i × (1 - beta_i)` and forecast cumulative
loss at a later age `j` of `C_i + A_i × (beta_j - beta_i)`. GCC distances are
calendar distances in origin periods; its loss-ratio estimate uses all origins
available at the cutoff, independently of the factor history window. Decay 0
uses only the same origin (and equals CL, whatever the trend); decay 1 gives
ordinary Cape Cod.

`trend` (GCC only, default 0) is Gluck's: each origin's losses are moved to the
cutoff's level by `T_n` before they are pooled, and the pooled ratio is brought
back to origin `i`'s level by `T_i`. The origins table carries `T_i` as
`trend_factor` and `LR_i × T_i` as `trended_loss_ratio` (chainladder-python's
`apriori_`; `expected_loss_ratio` is its `detrended_apriori_`). At trend 0 every
`T_n` is exactly 1.0, and the loss ratios are the same numbers, bit for bit, as
before the option existed. chainladder-python's `CapeCod` defaults to
`trend=0.05`, so a comparison with it has to pass the same trend.

`n_iters` (BF and GCC, default 1) iterates the method as Benktander does (Mack
2000): with `q_i = 1 - beta_i`, `U_0 = A_i` and `U_k = C_i + q_i × U_(k-1)`, the
ultimate is `U_n`. `n_iters=1` is the method itself; as `n_iters` grows the
ultimate moves to CL's. The origins table's `prior_ultimate` is then `U_(n-1)`,
so the forecast formula above holds for every `n`, and `expected_ultimate`
carries `A_i`. `n_iters=0`, the expected loss method, is refused.

Factors below 1, proportions above 1, negative increments and negative reserves
are permitted. Cumulative losses must be finite and non-negative. Premiums must
be finite and strictly positive. Uncertainty distributions are not inferred
from these point estimates: Mack's uncertainty formulas retain their own
estimator assumptions, and these candidates are not gallery registrations.

## Observation selection and sparse data

The input is one cohort, cumulative, on the existing complete run-off staircase
contract, with matching origin/development grains. Annual, quarterly and monthly
grains are supported. Interior holes are errors. The cutoff uses stored
`eval_date`, including historical restatements. BF/GCC premiums are obtained
from that same slice; a caller-supplied BF ratio must also have been available
at the cutoff.

At each development age:

1. Gather origins observing both ends of the link. Under
   `zero_cells="observed"`, remove pairs with undefined individual ratios (zero
   current cumulative loss, reason `undefined_ratio`) and keep the most recent
   `history_periods` of the pairs left. Under `zero_cells="missing"`, keep the
   most recent `history_periods` of all the pairs and then remove every pair
   with a zero at either end (reason `zero_cell`), so a removed pair keeps its
   place in the window, as chainladder-python's `n_periods` counts it.
   `history_periods=None` keeps every pair.
2. Remove explicit `(origin_period, from_dev_lag)` exclusions. The lag is in
   months. Exclusions beyond the current information set remain fixed for later
   refits. Excluding a link does not delete that origin's observed losses.
3. Remove every link whose LATER cell is valued on one of
   `exclude_valuations` (month-end dates on the grid's diagonals), reason
   `valuation_exclusion`: excluding a year removes the development during it.
   A valuation the cutoff has not reached stays fixed for later refits.
4. Remove ratios strictly above `drop_above` or strictly below `drop_below`; a
   ratio equal to a bound is kept.
5. Remove the `drop_low` lowest and the `drop_high` highest ratios (counts;
   `True` is 1). `trim_ties="origin"` (the default here) breaks ties by origin,
   the oldest low and the newest high; `"volume"` (the `ibnr.methods` default,
   chainladder-python's rule) breaks them by the earlier cumulative first, the
   smallest low and the largest high, then by origin. The two sides never pick
   the same ratio. The history window is not backfilled after exclusions.
6. Calculate the volume (ratio of column sums), simple, regression
   (`sum(previous × following) / sum(previous²)`, least squares through the
   origin) or median factor on the same selected pairs. Every removed
   observation is recorded with the first rule that removed it.

Steps 4 and 5 each obey `preserve` (default 1), the fewest ratios each may
leave at an age, all or nothing. By default an unavailable/non-positive/
non-finite factor raises an error. `unsupported_factor="unity"` explicitly
substitutes 1 and marks the age in `factor_summary`. By default, a rule that
would leave fewer than `preserve` ratios raises an error.
`exhausted_exclusions="keep"` skips that rule at that age and records it
(`extreme_trimming_skipped` for the trims, `bounds_skipped` for the bounds); it
never reverses explicit or valuation exclusions. An age with no ratio left
before step 4 or 5 is not an exhausted rule: `unsupported_factor` decides it.

The selection lives in `ibnr.kernels.links` (`LinkRules`, `select_links`,
`link_factors`, `ALPHA`), numpy only, written once so that the Mack fit and the
bootstrap refit can use the same one.

On any complete run-off triangle the deepest link has exactly **one** origin
pair, so removing an extreme ratio there would leave nothing to average: under
the default `exhausted_exclusions="raise"` a candidate carrying `drop_high` or
`drop_low` always raises on such a triangle. That is why the benchmark scripts
set `exhausted_exclusions="keep"`, which is the setting that reproduces the
published chainladder-derived numbers rather than stopping at the deepest link.

The final age is either the explicitly supplied `horizon` or the deepest age
observed in a standalone fit. An explicit horizon cannot omit observed ages, so
a replay has to stop before the data develops past the declared horizon: a fit
refuses a horizon shorter than the development it can see. An age beyond the
available data requires the explicit unity policy. There is no fitted tail
beyond the horizon.

The paper does not prescribe all of these boundary conventions, so this package
declares its own and documents them here: the pair, window and tie rules above,
and the unity and skipped-trimming fallbacks, which are never automatic and
have to be requested explicitly.

A cumulative of exactly zero is read one of two ways, set by `zero_cells` on
the candidate and on `fit_mack`, `fit_mack_grid` and `fit_mack_many`:

- `"observed"`, the default of both kernels, keeps it as data, and the two
  kernels then differ. This candidate family needs a defined individual link
  ratio, so it leaves out the ratio out of a zero but uses the ratio into one
  (a ratio of 0). Mack's untrimmed volume estimator keeps both pairs in its
  column sums, as R's `MackChainLadder` does, and estimates sigma only from
  pairs that start from a positive amount.
- `"missing"` is chainladder-python's rule, which stores a zero cell as
  missing: a link ratio is used only when neither of its two cells is zero, in
  both kernels, for the factor and Mack's sigma alike. `ibnr.methods` defaults
  to it. A zero on an origin's latest diagonal is still that origin's latest
  amount, so its chain-ladder ultimate is 0 and, under this rule, its Mack
  standard error is 0 too (chainladder-python leaves both missing). Where the
  rule leaves an age before the last with one link ratio, Mack's sigma there is
  filled in as chainladder-python fills it: the log-linear rule from one
  regression over every age with a positive estimate, Mack's rule from the two
  ages just before. The one-year claims development result refuses a Mack fit
  on which this rule left anything out.

On a triangle with no zero cumulative the two settings give the same answer.

## Fitting from arrays

`fit_conventional_grid` is the same estimator as `fit_conventional`, started
from plain arrays instead of a `Triangle`, so no database query runs on the
path. Importing it still imports ibis, because the kernels modules import the
Triangle layer; only the queries are gone. Use it in a service that fits one
small triangle per request, where the database queries inside
`fit_conventional` take longer than the fit itself. On the same cells it
returns exactly what `fit_conventional` returns.

```python
import datetime as dt

import pandas as pd

from ibnr.kernels import ConventionalCandidate, cohort_grid_frame, fit_conventional_grid

a, b, c = dt.date(2021, 1, 1), dt.date(2022, 1, 1), dt.date(2023, 1, 1)
rows = pd.DataFrame(
    {
        "origin_period": [a, a, a, b, b, c],
        "dev_lag": [12, 24, 36, 12, 24, 12],  # months from the origin's start
        "value": [100.0, 180.0, 200.0, 120.0, 210.0, 130.0],  # cumulative
    }
)
grid = cohort_grid_frame(rows, dev_grain_months=12, measure="cumulative")

chain_ladder = fit_conventional_grid(grid, ConventionalCandidate())
print(chain_ladder.as_of)  # 2023-12-31, read from the latest cell
print(chain_ladder.origins[["origin_period", "ultimate", "reserve"]])

premium = {a: 250.0, b: 260.0, c: 280.0}  # keyed by origin, never by position
bf = fit_conventional_grid(
    grid, ConventionalCandidate("bf", expected_loss_ratio=0.8), premium=premium
)
print(bf.origins[["origin_period", "ultimate", "reserve"]])
```

There is no `as_of` argument: the information date is the evaluation date of
the grid's latest cell (origin 2023-01-01 at 12 months is 2023-12-31), so each
origin period must be the first day of its period. The grid carries a
development step but no origin grain, so the fit reads the origin spacing from
the dates and refuses origins that are not one development step apart. A
missing origin period is accepted only where every origin before it has
already reached the last development step: the run-off check counts origins by
position, so a missing year among origins still developing is refused as not a
run-off triangle, as it is by `fit_conventional`. Premium is needed for BF and
GCC only, and passing it to a chain ladder candidate is refused rather than
ignored. `fit_mack_grid`, which gives Mack's standard errors from the same
grid, checks the grid the same way.

Measured on RAA (ten accident years, chain ladder), as the median of 100 to
200 warm runs, repeated in three separate processes on one Windows laptop
(Intel Core Ultra 9 285H, Python 3.12, duckdb backend): `fit_conventional` took
about 28 ms, `cohort_grid_frame` plus `fit_conventional_grid` about 2 ms, and
chainladder-python's `Chainladder().fit` about 28 ms. The ratio is the finding;
the milliseconds move with the machine.

## Observed replay

```python
from ibnr.kernels.replay import replay_conventional

replay = replay_conventional(
    tri,
    {"gcc_075": candidate},
    dates=["2020-12-31", "2021-12-31", "2022-12-31"],
    on_error="record",
)
print(replay.cells)
print(replay.errors)
```

Each supplied date is an information cutoff. Successive dates must be one
development period apart, preserving the day or month-end anchor (including
February's shorter month). All candidates must have the same explicit horizon.
At each date the same settings are refitted on the historically available
triangle and premium. Every pair of dates compares only origins present at the
first date; newly appearing origins enter the second fit but are listed in
`exclusions` and do not enter that interval's outcome score.

For each existing origin:

- `actual_increment = new latest cumulative - old latest cumulative`.
- `expected_increment = old forecast of the next cumulative - old latest cumulative`.
- `ave = actual_increment - expected_increment` (adverse positive).
- `cdr = new fitted ultimate - old fitted ultimate` (adverse positive).
- `remaining_revision = new reserve - old expected reserve remaining after the next diagonal`.

The signed identity is `cdr = ave + remaining_revision`. Add components before
squaring any CDR error. This sign agrees with the paper and is the opposite of
the favorable-positive simulated/analytic CDR convention in `kernels.cdr`.

`actual_increment` is an observed **cumulative movement**. It can include
restatements and, for incurred data, movements in case reserves; it is not
necessarily cash paid. Both actual and expected increments subtract the same
old-date predecessor. New-date restatements affect the new fit; subsequent
restatements cannot alter an earlier replay. Even an origin already at the
fixed horizon can have a nonzero movement/CDR after a terminal-value revision.

Every open prior origin must observe exactly its next development age at the
second date. Interior holes, delayed origins, skipped ages, a lost origin or an
unfittable candidate fail the **whole candidate interval**. A pure-restatement
interval in which open origins do not develop is unsupported. With
`on_error="raise"` the replay stops; with `"record"` it retains a named reason
and emits no partial outcome rows for that candidate interval. Fully developed
unchanged origins remain explicit zero rows. `fits` retains each successful fit
and its selection/fallback diagnostics under `(candidate_name, cutoff)`.

This is observed replay of the conventional point procedures. It is separate
from a simulated CDR distribution and from replaying every distributional
gallery entry through its own re-reserving method.

## Select, then evaluate later

```python
from ibnr.kernels.selection import evaluate_conventional, select_conventional

decision = select_conventional(replay, selection_as_of="2022-12-31", metric="ave")
print(decision.ranking)
# This does not refit the selected model on the later data.
evaluation = evaluate_conventional(decision, full_triangle, as_of="2032-12-31")
print(evaluation.summary)
```

`score_replay` implements the paper's Equation 2. On each diagonal, for error
`e_i` (AvE or CDR), it computes
`sqrt(sum(abs(actual_increment_i) × e_i²) / sum(abs(actual_increment_i)))`.
Negative actual movements receive positive weights. Zero movements receive
zero weight. A whole diagonal with zero weight has an undefined score, retained
with status `zero_weight`. The signed CDR is squared after combining AvE and
the remaining-reserve revision. Some worked paper tables are inconsistent with
its stated equations; this implementation follows the equations.

Selection uses the arithmetic **mean of diagonal RMSEs**, giving each date
equal weight. It does not pool all cells into one weighted RMSE. Only intervals
whose ending information date is on or before `selection_as_of` enter the
ranking. Later replay results and failures have no influence. The selection
date must be a replay cutoff with at least one completed historical interval
and a successful fit at that date.

Every candidate must have a valid score for **every** interval in this history.
One failed, incomplete, nonfinite or zero-weight interval makes it ineligible;
its ranking row retains counts and reasons, with no partial mean. If none is
eligible, selection raises an error. There is no automatic complete-case
intersection or missing-score imputation. Supply an explicitly shorter replay
to change the historical window, before inspecting later evaluation results.
Exact score ties use candidate names in lexical order. Candidate names and
the full settings remain in the decision record.

Later evaluation compares the frozen ultimate prediction to the cumulative
observation at the **same declared horizon**, using only records available by
its later cutoff. This is a terminal-development-age comparison: the data may
not represent settled economic ultimate losses. It reports unweighted RMSE
over the training origins whose terminal values were unknown at selection.
New origins and terminal values already known at selection do not enter this
test. Every target and its observation date remain visible. Missing terminal
observations suppress the overall RMSE rather than improve it by dropping
harder targets. A supposedly unknown target with an observation dated before
selection raises a history-mismatch error.

This measures historical point-forecast performance. It does not establish
probability calibration, nor that selecting on AvE or on CDR is superior in
general.
