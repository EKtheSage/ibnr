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
  `LR_i = sum_n(C_n × decay^distance(i,n)) / sum_n(P_n × beta_n × decay^distance(i,n))`.

All three have ultimate `C_i + A_i × (1 - beta_i)` and forecast cumulative
loss at a later age `j` of `C_i + A_i × (beta_j - beta_i)`. GCC distances are
calendar distances in origin periods; its loss-ratio estimate uses all origins
available at the cutoff, independently of the factor history window. Decay 0
uses only the same origin (and equals CL); decay 1 gives ordinary Cape Cod.

This generalized Cape Cod carries **no trend parameter**: origins are combined
by decay alone, and no loss-ratio trend is applied across origin periods.
chainladder-python's `CapeCod` defaults to `trend=0.05`, so a comparison with it
has to pass `trend=0` or the two answers differ for that reason alone.

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

1. Gather origins observing both ends of the link. Remove pairs with undefined
   individual ratios (zero current cumulative loss). Keep the most recent
   `history_periods` paired origins, or all if `None`.
2. Remove explicit `(origin_period, from_dev_lag)` exclusions. The lag is in
   months. Exclusions beyond the current information set remain fixed for later
   refits. Excluding a link does not delete that origin's observed losses.
3. Remove one lowest and/or one highest ratio. Ties use the oldest low and
   newest high origin; low is removed first, so both flags remove distinct
   observations. The history window is not backfilled after exclusions.
4. Calculate the volume (ratio of column sums), simple, or median factor on
   the same selected pairs. Every removed observation is recorded.

By default an unavailable/non-positive/non-finite factor raises an error.
`unsupported_factor="unity"` explicitly substitutes 1 and marks the age in
`factor_summary`. By default, extreme trimming that would exhaust the pairs
raises an error. `exhausted_exclusions="keep"` skips both extreme removals and
records that fact; it never reverses explicit exclusions.

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
have to be requested explicitly. Mack's existing untrimmed volume estimator
can include zero-current pairs; this candidate family consistently requires
defined individual link ratios, so the two can differ on zero-valued data.

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
