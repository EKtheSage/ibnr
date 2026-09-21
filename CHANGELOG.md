# Changelog

This file starts at 0.5.0. For 0.1.0 through 0.4.0, read `git log v0.1.0..v0.4.0` -
those releases predate the file and reconstructing them now would be a summary of
a summary.

Versions follow [semantic versioning](https://semver.org/), loosely: while the
package is `Development Status :: 3 - Alpha`, a minor bump is free to change a
kernel signature. The public surface named in CLAUDE.md decision 8 (`Triangle`,
`gallery.list/fit/evaluate/stack/scaffold/leaderboard`) is the part treated as
stable. Nothing in it was removed or renamed in this release; it GREW (the
held-out evaluation pipeline, and a `segment` argument on three entry methods).
(As built that surface is `Triangle` plus
`gallery.list/get/fit/stack/leaderboard/next_diagonal/CohortForecast/Absence/align_panel/SCORE_DIRECTION`
- `evaluate` is a method on a fitted entry and `scaffold` is planned, per the
corrected decision 8.)

## Unreleased

New gallery entry `mcl` (family `statistical`): the full-matrix multivariate
chain ladder, Zhang's (2010) general form, where each line's next cumulative is
regressed on every line's current cumulative rather than only on its own.
`sur`, its sibling, is the special case where that coefficient matrix is
diagonal. Each development transition is one system, estimated in a single
feasible-GLS step with the R `systemfit` conventions - per-equation OLS, an
uncentred residual covariance with the geomean denominator, and Mack's
weighting applied per equation by its own line's current cumulative. A
transition with no more origin pairs than twice the line count, and one whose
matrices are singular to working precision, falls back to the diagonal
volume-weighted chain ladder; `transitions_[d]["method"]` and
`fallback_reason` say which happened and why. `point()` is the vector
recursion and `predict()` simulates it with parameter risk and correlated
process risk, as `sur` does.

The entry's point reserves tie out to the companion study's R implementation on
82 Schedule P companies at 31 December 2007, to 1.3e-12 relative at worst, and
`mack`'s tie out to the same study's chain ladder to 8.6e-15. The company
tables that replay produced are vendored at `tests/data/`, and
`tests/test_mcl_tieout.py` (marker `mart`) is the comparison. On the 11
companies the R chain ladder could not score, this package refuses the fit by
name for the same reason the R run failed: a paid cumulative at or below zero
on a cell a development transition divides by.

`nearest_pd` and `mack_tail_variance` moved out of the `sur` entry into
`kernels.multiline`, with `EIG_FLOOR`, so the two entries share one copy. `sur`
keeps the underscored spellings as module aliases.

The appendix data behind the published conventional examples is now carried in
the repository at `analysis/data/balona_richman_2020_appendix.json`, transcribed
from the freely distributed 14 August 2020 manuscript, so the benchmark fetches
nothing at runtime and needs no network access. The documentation cites that
paper directly - Caesar Balona and Ronald Richman, "The Actuary and IBNR
Techniques: A Machine Learning Approach", https://ssrn.com/abstract=3697256 -
rather than a copy hosted elsewhere.

New: point-error metrics. `kernels/point_scores.py` implements `point_metrics`
(Pool_APE, Pool_PE, MAE, RMSE, wRMSE, MAPE, MedAPE, prop_over), `level_errors`
(sum within a named level before the absolute value), `shrink_toward` (a point
pulled toward a baseline) and `reserve_rows` (predicted against actual reserve
per cohort for any fitted entry, from the draw mean or a native point).
`GalleryEntry.evaluate()` gains a `point` key carrying the per-target error and
those metrics over the targets that are not the total. `mack` gains `point()`,
the deterministic ultimates in `predict()`'s target order. `reserve_rows` is
exported from `ibnr.gallery`; the metrics themselves stay on `ibnr.kernels`,
because they consume the table and never touch an entry.

New: the shared NN training loop (`gallery/nn/_training.py::train_ensemble`)
takes a learning-rate `schedule` (`warmup_cosine` ports the R study's warmup
plus cosine decay), `param_groups`, `min_epochs`, `check_every` (patience now
counts validation checks), `cutoff_sampling="per_epoch"` and `keep` (the best k
members by validation score). Every default is bit-identical to 0.6.0, checked
against a verbatim copy of the old loop. Every history record now also carries
the `member` it belongs to. New: `kernels/residual_calibration.py`, the
size-stratified rolling-origin residual calibration of a point forecaster
(`rolling_residuals`, `calibrate`, `calibrated_draws`,
`leave_one_out_coverage`), re-exported from `ibnr.kernels`.

New gallery entry `tlrn`, the transformer loss reserving network, family `nn`,
fitted on company cohorts. It reproduces a companion study's R model: one
training example is a company at one accident year, its tokens are that year's
(line, development lag) cells, and axial attention runs across lines within a
lag and across lags within a line. The network does not predict a cell - it
predicts a positive log development factor per (line, step), and the cells
follow by projecting each origin's cumulative forward, so setting the factors to
the chain ladder's reproduces the chain ladder exactly. 14,309 parameters at the
published shape. Trained under the study's checkpoint protocol: one cutoff drawn
per epoch, the trailing calendar diagonals held out, ten seeds run for the whole
schedule and the two that validated best kept with no refit. `point()` gives a
company's ultimates in the multi-line layout; `predict()` gives its total
ultimate as historically calibrated draws, which are not a native predictive
distribution and are at company level only, so the entry takes neither held-out
mixin. The card discloses both, and that the calibration cutoffs overlap the
training targets at the earlier ones.

Supporting it: `kernels/nn_features.py` (`tlrn_features`, `pooled_cl_factors`,
`origin_cl_log_factors`), which builds the study's engineered example tensors -
eight features from paid alone, thirteen with an incurred channel and a case
reserve channel - from cells on or before the cutoff and nothing later. And
`kernels.nn_contract.nn_data`/`nn_company_data` gain one key, `values`: the
field's raw grid in the units the triangle reported, NaN where absent. Additive;
no existing consumer reads it.

## 0.6.0 - 2026-09-15

A new family of estimators and the application that sits on it (#136), plus
the two review fixes that landed on main after 0.5.9 (#134, #135).

New: `kernels/conventional.py` fits chain ladder, Bornhuetter-Ferguson and
Gluck's generalized Cape Cod as frozen point candidates; `kernels/replay.py`
refits them at successive information dates and reports the observed AvE and
CDR of each interval; `kernels/selection.py` selects on the completed history
and evaluates the frozen forecast on later terminal-age outcomes. All three
are re-exported from `ibnr.kernels`. These are point forecasts and stay out of
the gallery. `apps/reserving_review/` is a local reference application over
them and is not in the wheel. The benchmark against Balona and Richman (2021)
and a 30-seed synthetic study are under `analysis/results/conventional/`; the
Swiss fixed baselines match the paper and the selected winners do not, which
the summaries record as an open gap.

Inputs 0.5.9 accepted that now raise:

- `gallery.stack()` on a weights table whose weight-fitting outcomes were
  observed after the evaluation cutoff, or whose used cells lack an
  unambiguous `eval_date` (#136);
- `to_wide` and the dev-grain coarsening on a cell stored twice (#134);
- `convergence()` asked for a parameter the posterior does not carry (#135).

Numbered 0.6.0 rather than 0.5.10 because the public surface grew: three
kernel modules with eleven new names on `ibnr.kernels`, and a new refusal on
`gallery.stack()`, which is part of the surface CLAUDE.md decision 8 treats as
stable.

Still open: reconciling the paper's selected winners (recorded in
`analysis/results/conventional/published/summary.md`), and the 0.5.9 backlog
in issues #120-#129.

### Review fixes to the conventional work and the reference application (#136)

From a review of the entries below, before any of them is released. In the
application: a decided run can be revised once, and a second request names the
revision that already exists instead of creating another approvable position
for the same cutoff, which the listing now marks as superseded; one record
whose hash chain no longer verifies is listed as `UNVERIFIABLE` rather than
turning the whole listing into an error and locking everyone out of the
workspace; a CSV holding more than one cohort is refused by name instead of
failing once per candidate per date, and the text of any error is bounded
before it reaches the browser; `--demo` beside `IBNR_REVIEW_USERS` is refused
rather than quietly replacing the configured accounts; the server binds the
host it was given, logs an unexpected failure to stderr, and writes the
approved export in the canonical form the stored hash covers. In the library:
the unity fallback is reachable together with the high/low exclusion flags,
`evaluate_conventional` refuses an evaluation triangle whose history as of the
selection date differs from the one the selected candidate was fitted on and a
candidate carrying no explicit horizon, `score_replay` is linear in the number
of candidates rather than quadratic, and `premium_by_origin` and `as_date` are
public names.

### Reserving review reference application (#136)

`python -m apps.reserving_review --demo` runs a local browser application with
real historical analysis, source and engine hashes, candidate and factor
evidence, reasoned origin-level overrides, submission, independent approval or
rejection, linked revisions, and approved JSON exports. SQLite persists
immutable snapshots and hash-linked decisions; revision checks reject stale
edits. Named analyst/reviewer accounts enforce ownership and separation of
review. See [the application guide](apps/reserving_review/README.md) for setup
and the local reference deployment's limits.

### Published and independent conventional benchmarks (#136)

`scripts/benchmark_conventional.py` runs the full literal published grids and
prespecified stable/noisy/drift/shock synthetic portfolios. The appendix loader
checks a pinned source hash; data remain a runtime download. Outputs retain
every candidate ranking, selected settings, terminal-coverage counts, paper
differences and implementation hashes. Candidate batching is tested against
whole-grid selection. The protocol documents source grid/count ambiguities,
ultimate-derived Swiss premiums and declared information-date conventions.

### Historical selection and later evaluation (#136)

`ibnr.kernels.selection` scores each replay diagonal using the paper's absolute
actual-weighted RMSE, then selects on mean diagonal RMSE available by an
explicit cutoff. Incomplete or undefined histories are ineligible, with
coverage and reasons retained. Later evaluation freezes the selected forecast
and compares only unknown-at-selection terminal-age targets; missing targets
suppress the aggregate score. Outcome dates, units, grains and cohort identity
are checked. See [the scoring rules](docs/conventional.md#select-then-evaluate-later).

### Observed AvE/CDR replay (#136)

`ibnr.kernels.replay.replay_conventional` refits fixed conventional candidates
at successive information dates, using the same explicit development horizon.
Origin-level results expose adverse-positive AvE, CDR and the revision to
remaining reserves. New origins enter the refit but are excluded from that
interval's score. Historical restatements use the correct information cutoff;
incomplete or unfittable candidate intervals raise or retain explicit failures.

### Conventional CL, BF and generalized Cape Cod point candidates (#136)

`ibnr.kernels.conventional` adds frozen candidate settings, historical fitting,
volume/simple/median factors, per-age history windows, explicit and high/low
link exclusions, and an explicit candidate grid. Fits retain selected-pair
diagnostics and any requested sparse-factor fallback. BF uses a supplied loss
ratio; GCC estimates origin-specific loss ratios, with decay 0 matching CL and
decay 1 matching ordinary Cape Cod. These point forecasts are separate from
the distributional gallery and Mack's uncertainty formulas. See
[the conventions and API](docs/conventional.md).

### Stacking requires weight-fitting outcomes to be available at evaluation (#136)

**This refuses an input 0.5.9 accepted.** `gallery.stack()` now refuses an
earlier `ForecastPanel` whose weight-fitting outcomes were observed after the
evaluation cutoff. Previously, January and February forecasts could both target
December outcomes: the cutoff-order check passed while the stack learned its
weights from the same future outcomes it was evaluated on, and nothing raised.

The check uses the dates of the ELPD cells actually used to fit weights, matched
by cell key. Outcomes dated exactly on the evaluation cutoff remain valid;
later CRPS-only outcomes do not affect weight fitting. Missing or ambiguous
availability metadata is refused, including on a `ForecastPanel` restored from
Arrow.

### `to_wide` and the dev-grain coarsening refuse a cell stored twice (#118)

Both operations reduce a cell's rows to one number without being asked which
observation was meant, so a cell restated at a later `eval_date` - legal stored
history, and the reason `as_of` can answer what was on the books at a past date
- was counted beside the value it replaced. Measured on both backends: 100
booked and 95 restated a year later displayed through `to_wide` as one cell of
195, and an annual bucket whose four quarterly increments are worth 105 came out
of `with_dev_grain("Y")` as 125. Both are plausible numbers and neither raised.
This is the pair #112 deliberately left open.

Following #112's idiom, `validate.require_single_observation` now refuses such a
triangle by name at both operations, naming the operation, how many cells are
affected, and the way out for each of the two routes into that state: a cell at
several eval_dates is restated history, where `latest_diagonal()` or an `as_of()`
before the restatement each leave a single view, while a cell recorded twice at
one eval_date is duplicated source data, which no slice resolves - `as_of` picks
an eval_date and keeps every row carrying it - so that repair belongs in the
source. Neither operation slices for the caller: which view was wanted is the
caller's to choose, and that choice changes the answer (on the bucket above,
`as_of("2020-12-31")` gives 105 and `latest_diagonal()` gives 20).

Inputs 0.5.9 accepted that now raise, and the ones that do not:

- `Triangle.to_wide()` on a triangle storing the pivoted field's cell twice.
  The check is on the selected field, so restated premium does not stop paid
  loss being displayed; and it keys on the segment columns, so the pivot's other
  sum, across segments, is unchanged - a two-line triangle still displays the
  two lines' total.
- `Triangle.with_dev_grain()` in both measures, once it actually coarsens. The
  cumulative path never summed, so it is refused for its own reasons rather than
  that one: bucket boundaries are counted back from the triangle's latest
  eval_date, which a restatement moves, so the row kept for a cell need not be
  the one that survives it (measured: the regrain kept a superseded dev-3 value
  of 20 and dropped the 30 that had replaced it), and when a cell and its
  restatement land one bucket apart both survive, leaving two rows at one age.
- Asking for the dev grain a triangle already has is still a no-op and still
  returns the same object, refused triangles included: it recomputes nothing, so
  nothing can move. Same rule as #112 gave the origin regrain.
- The Schedule P mart is unaffected - it stores one row per cell, so its
  `to_wide` tie-out is unchanged - and so are `as_of`, `latest_diagonal`,
  `to_cumulative`, `to_incremental` and the origin regrain, none of which
  reduces a cell's rows to one number this way.

### `convergence()` refuses a parameter the posterior does not carry (#119)

Each of the six Bayesian entries filtered its default parameter list down to the
names the fitted posterior happened to carry, then reported max R-hat and min ESS
over whatever survived. Nothing said the set had shrunk, and
`kernels.harness.ConvergenceGates` decides sampler escalation from those two
numbers, so a fit missing a parameter read as converged over a strictly smaller
set than was asked for and was never re-run. Measured on a CCL-shaped posterior
whose `a_ig` sits at a different level in every chain: with `a_ig` present max
R-hat is 2.84 and min bulk ESS 5, and the fit is escalated; drop `a_ig` and the
same call answers 1.00 and 1833, and the fit is accepted. Same family as the
parity leniency #113 fixed in 0.5.9, but not the same fix - these filters acted
on each entry's DEFAULT list, so the per-backend lists had to be designed first.

- Every Bayesian entry now declares `CONVERGENCE_VARS` at module level, keyed by
  the backend argument its `fit()` took. For five entries the three lists are the
  same names written out three times, so a port that renames a site fails rather
  than being inferred around.
- `compartmental` is why the mapping is per backend at all: Stan and NumPyro
  declare the correlated accident-year block as a `sd_ay` / `L_ay` pair, while
  PyMC's `LKJCholeskyCov` is both at once and reports the scales as
  `ay_chol_stds`. Its lists are per variant as well - Model 1 gives `ker` and
  `kp` no varying effects, so `sd_dev` / `sd_ker` / `sd_kp` exist only under
  `lognormal`.
- The summary itself moves to the new `kernels.diagnostics.convergence_report`,
  which the six entries call instead of each computing the identical dict. It
  also refuses a name that reaches arviz and leaves no summary row, and an
  unknown backend rather than falling back to a neighbour's list.
- An explicit `var_names` naming a parameter the fit lacks is refused too. It
  used to come back as an ordinary diagnostics dict over the names that did
  exist; a request naming only absent parameters died inside xarray on
  `Dimension(s) 'chain', 'draw' do not exist`.

## 0.5.9 - 2026-09-06

Every fix from the 2026-09-04 external review of 0.5.8 (ten PRs, #107-#116),
led by an install repair: a fresh `pip install ibnr` of 0.5.8 resolves sqlglot
30.18.0, which breaks `Triangle.from_long` from any in-memory frame on the
default duckdb backend. Upgrading to this release is the fix; a user staying on
0.5.8 or earlier must pin `sqlglot<30.18` beside ibnr themselves. Details in
the first entry below.

Most of the other fixes replace a silently wrong answer with a refusal, so
inputs 0.5.8 accepted now raise:

- a one-year CDR on a quarterly or monthly development grain, at
  `one_year_cdr`, `simulate_one_year_cdr` and `rereserve` (#108);
- a pooled origin axis whose step is not one development step - a gapped
  accident-year axis, or annual origins on a quarterly dev grain - at the data
  contracts serving `meyers_ccl`, `meyers_csr`, `guszcza_growth_curve` and the
  NN entries (#115);
- a row whose `eval_date` and `dev_lag` disagree, restated history included,
  at `change_origin_grain`, `to_chainladder` and `to_bermuda`, until the
  triangle is sliced to a single view (#112);
- a bermuda object whose evaluation-date resolution no dev grain of ours can
  represent, at `from_bermuda` (#115);
- an unidentified design: `sur` on a transition with fewer origin pairs than
  design columns, `copula_glm` on a design that is rank-deficient after
  exclusions (#110) - on the pinned mart this refuses two `reported_loss`
  cohorts that previously fitted, and nothing on the published `paid_loss`
  rows;
- a parity request naming a variable any posterior lacks, an element shape
  differing from the reference, a non-finite draw, a dimension-labelled
  posterior, an empty request, or two unequal point masses (#113);
- a stacking solve scipy reports unsuccessful, and a `+inf` pointwise ELPD
  (#111);
- a non-finite or negative mean, or a negative or non-finite dispersion, at
  the shared over-dispersed Poisson draw (#114).

Two changes move numbers rather than raise. A missing outcome or a missing
draw now gives a missing percentile instead of 0.0 (#107) - one published
compartmental figure is re-read, see that entry. Draws whose Poisson rate
exceeds numpy's own cap come back as a point mass at the mean (#114) - every
ordinary seeded fit is byte for byte unchanged. And the validator moves in the
opposite direction from the kernels: it now accepts consistently anchored dev
ages it used to reject wholesale, while the kernel doors refuse them by name
(#115).

Numbered 0.5.9 rather than 0.6.0 because every refusal above replaces an
answer that was already wrong, which is this changelog's practice for
corrections (0.5.1's refusals, 0.5.7's draw change).

What this release does not fix is now tracked rather than remembered: the
follow-up backlog is filed as issues #118-#129. The two that can still put a
misleading number in front of a user: `to_wide` and the incremental dev
regrain still silently sum a restated cell into the value it replaced (#118,
the one silent-summation route left after #112), and the Bayesian entries'
`convergence()` summaries still drop a requested parameter the fit does not
carry (#119). Also open: no true twelve-month CDR on sub-annual grains (#125 -
refused, not answered), no export route that carries restated history (#124),
and the published leaderboard and compartmental retrospective are not yet
rerun on the current draw paths (#129).

### `pip install ibnr` works again: sqlglot capped below 30.18 (#116)

sqlglot 30.18.0 (2026-09-03; tobymao/sqlglot#8229, listed under BREAKING
CHANGES in its changelog) renamed the `Drop` expression's `this` argument to
`tables`. ibis 12.0.0 still passes `this=`, so the SQL it renders to drop a
memtable view loses the view's name and duckdb answers `Parser Error: syntax
error at end of input`. ibis's duckdb `create_table` takes that path for every
in-memory frame it is handed - register the frame as a view, insert from it,
drop the view - and that is the one call `Triangle.from_long` makes to ingest
a pandas, polars or pyarrow frame. So on any fresh install resolving today's
sqlglot, every `from_long` from an in-memory frame on the default duckdb
backend died at ingestion. Parquet paths, ibis expressions (`load_schedule_p`
included) and the polars backend were unaffected.

The core dependencies now carry `sqlglot<30.18`: a cap on a transitive
dependency, and a temporary one. It lifts only together with an ibis floor
carrying the upstream fix (proposed in ibis-project/ibis#12104), because the
rename is deliberate, so a later sqlglot will not restore the old argument -
which is also why `!=30.18.0` would be the wrong shape. The cap went into the
package metadata rather than into CI's install lines because a CI-only pin
would have greened the build while leaving `pip install ibnr` broken - the one
thing the unpinned CI legs exist to catch. `tests/test_sqlglot_cap.py` names
the cause through the public ingestion path if the cap is ever lifted before
that ibis floor exists.

### A missing outcome or a missing draw gives a missing percentile, not 0 (#107)

`PredictiveDistribution.cdf` counted the draws at or below the outcome and
returned that fraction. Every comparison against NaN is False, so a target whose
outcome had not emerged yet came back as exactly 0.0: the lowest percentile
there is, and to the uniformity test the worst possible over-prediction. A
missing draw compares False in the same way, so every one of them was counted as
lying above the outcome and that target's percentile came out too low. `cdf` now
returns NaN at those targets, which is what its own docstring already promised
for a missing outcome. `kernels.scores.crps` was already answering NaN for a
missing outcome, and `mean`, `std` and `crps` were all already answering NaN for
a missing draw, so the percentile now agrees with the numbers printed beside it.
The mask is per target, so a finite neighbour keeps its percentile, and infinite
outcomes and draws are left alone: an outcome below every draw is a real verdict
of 0.0 and has to stay distinguishable from a missing one.

One published figure moves, and it is a missing-draw row rather than a
missing-outcome one. Every scored row in the Schedule P results carries an
outcome, so that half of the fix changes nothing there; it changes a backtest on
a triangle whose cutoff leaves an origin short of the fit's final development
lag, the case the `mack` entry documents by name. But one row of
`analysis/results/compartmental_validation_lognormal.csv` (other_liability,
company 16373) came from a fit that never converged (R-hat 1.25, bulk ESS 12)
and whose draws hold missing values, which is why its estimate, standard error
and CV are already blank there while its percentile reads 0.0. That 0.0 was the
old `cdf` counting missing draws, not a real over-prediction. Under the fix the
row has no percentile and leaves the uniformity test. Recomputed from the stored
percentiles, other_liability goes from D = 20.2 (rejects, n = 50) to 18.6
(passes, n = 49) and the combined figure from 16.2 to 15.9 (still rejects,
n = 199). The compartmental card records both readings.

`scripts/meyers_validation.py` used to define `failed` as a missing percentile,
which would now also count such a row; it counts fits that raised instead, and
reports `missing_outcome` and `missing_draws` beside it, so a row that leaves
the test is never invisible.

### Stacking weights survive deep log densities, and an unconverged solve is refused (#111)

`MleStacking`, the default stacking method, does its arithmetic in linear
space: it exponentiates the pointwise ELPD and hands SLSQP the Jacobian
`1 / (Y @ w)`. Once the mixture `Y @ w` falls below `1 / DBL_MAX`, about
5.6e-309, that reciprocal overflows to infinity, SLSQP stops at its first
iteration and hands back the uniform vector it started from. That vector is
finite, non-negative and sums to 1, so every check the weights faced accepted
it as a fitted even split, and only three numpy warnings naming bayesblend's
own lines reached stderr. One shared cell below the boundary among sixteen
board-like cells was enough: a 1.0/0.0 fit came back 0.5/0.5, which scores 10.3
nats below the answer the fit should have given and is not the optimum of
anything.

`kernels/stacking.py` now passes bayesblend each cell's ELPD relative to that
cell's best finite member, floored at `LPD_FLOOR`, so every value it sees is
between -700 and 0 whatever the absolute level of the densities. A common
per-cell offset cannot move the optimum, so no correct answer changes. One
ranking does change, in the direction of the milestone 6 rule that zero
density ranks last: on a cell where one member gave the outcome zero density
and another gave it a tiny positive density, the old absolute floor put the
zero-density member above the finite one, and now it does not. `_fit_weights`
also reads the scipy result bayesblend stores and refuses an unsuccessful
solve by name, with its status and message, and a `+inf` pointwise ELPD is
refused where it arrives instead of reaching the solve as a NaN objective.

### The parity comparison covers every requested parameter, and a point mass is no longer free agreement (#113)

`kernels.parity.compare_posteriors` used to drop any requested variable a
posterior did not carry and then report a pass over the rows that were left, and
it scored a zero or undefined Monte Carlo error as `z = 0`, the best score there
is, whatever the two summaries said. It now refuses, by name, a requested
variable missing from any posterior including the reference, an element shape
that differs from the reference's, a non-finite draw and an empty request. Point
masses are handled explicitly: two equal constants agree at `z = 0` (the
identifiability anchors every CCL, CSR and ODP run carries), two unequal
constants score `z_mean = inf` and fail, and a Monte Carlo error that is not a
positive finite number on a marginal that is not constant raises. Whether two
point masses are equal is read off the pinned value itself, not off the two
summary means, which for a constant that is not exactly representable in binary
depend on how many draws were summed. A posterior that names its dimensions
(`coords=`/`dims=`) labels its elements with coordinates rather than integer
positions, and those are now refused by name too, where before the string form
crashed inside `int()` and an out-of-range integer form read the wrong element.
Every ordinary row keeps the identical formula, so no published parity number
moves.

### The compute image installs the data package its own default command needs (#109)

The Dockerfile installed `.[bayesian]` and nothing else, so the image's own
`CMD`, `python scripts/meyers_validation.py --help`, would stop at
`ModuleNotFoundError: No module named 'cas_schedule_p'` before printing a line
of help. The image was not built to find this: there is no Docker on the
development machine, so the command was run with `cas_schedule_p` absent from
the import path, which is the same failure one line earlier. That script and
the three other study scripts read the Meyers company selection rule out of
`cas_schedule_p.screens`, and the package is deliberately absent from the wheel:
ibnr itself never imports it, only `scripts/` does. The image now installs it,
pinned to the version `uv.lock` resolves, because the wheel carries the mart and
its vintage decides which companies a run selects.

`tests/test_compute_image.py` is what keeps this true. It reads the install
lines out of the Dockerfile, works out which distributions those lines put in
the image, then starts each promised script's `--help` in a subprocess that
refuses any module the image would not have, and separately reads every import
in those scripts at any nesting depth. Nothing in it can skip. It stands in for
building the image, which no machine here can do, and it is exact for the
failure that mattered.

### `sur` and `copula_glm` refuse designs the data cannot pin down (#110)

Both frequentist dependence entries could return a fit whose coefficients the
usable cells never determined. `sur` with `intercept=True` fits a two-column
`[1, C_d]` design at every development transition, and a square triangle's last
transition has exactly one origin pair: one equation for two unknowns. Depending
on the data that either raised `LinAlgError: Singular matrix`, which names a
matrix rather than the model, or rounded through and returned a coefficient
covariance of order 1e15, which parameter risk turned into a predicted grand
total hundreds of times too large with every draw finite. `copula_glm` checked
that every origin and every development step had a usable cell, which counts
cells per column and misses the shape they form: after `nonpositive="drop"`
exclusions the usable cells can split into groups sharing no origin and no
development step, and `np.linalg.pinv` then answers with the minimum-norm member
of an unbounded family. `sur` now refuses a transition with fewer origin pairs
than design columns before estimation starts, and `copula_glm` checks the built
design's rank and names the columns a null direction touches. Both refusals
happen before anything is stamped on the entry, so a rejected fit leaves the
entry unfitted.

What this reaches on the Schedule P retrospective, measured on the pinned mart
publish `20260613_041006` across all 60 selected multiline companies with
`scripts/compare_gallery.py`'s own default of `--copula-nonpositive drop`: on the
published `paid_loss` field nothing changes, all 60 companies still fit (45 of
them through the existing fallback to the Hoerl curve). On `reported_loss` the
new rank check refuses companies 13439 and 16373, which previously fitted at rank
11 of 12 under the Hoerl curve. The published copula rows are `paid_loss` only,
so no published result changes. `sur` is unaffected on the mart because the
retrospective runs its `intercept=False` default, where the design has one column
and the pre-existing no-origin-pair refusal already covers it.

### One shared over-dispersed Poisson draw, with numpy's real rate cap (#114)

The three ODP gallery entries (`clark`, `clark_growth_curve`,
`england_verrall_odp`) each wrote `phi * rng.poisson(mu / phi)` in two places,
six copies in all, and none of them handled the case numpy refuses: a Poisson
rate larger than about 9.2e18. A triangle that develops exactly on its own
fitted curve gets there: it fits itself to rounding error, the Pearson
dispersion collapses to about 1e-29, and the rate runs to 1e30, so `predict()`
raised `ValueError: lam value too large` on a fit that had nothing wrong with
it. That is a degenerate fit rather than an everyday one, and the measurements
say how far: the same triangle with its amounts rounded to whole units sits at
1e-14 of the limit, and with a relative noise of one part in a million, 1e-8 of
it. The bootstrap kernel already had a cap for this, but its constant was
the int64 maximum rather than numpy's own limit, which is 30 billion lower, so
rates in between passed the check and made numpy raise anyway.

All six sites and the kernel now call one function,
`ibnr.kernels.densities.odp_draw`. Cells past `POISSON_RATE_MAX` come back at
their mean, which is what the law says there: the draw's coefficient of
variation is below 3.3e-10, so it is a point mass. A mean that is not finite, a
negative mean, and a negative or non-finite dispersion are each refused by name
and by count instead of being reported by numpy as a rate problem, and they are
refused separately, because a mean that overflowed upstream and a mean the
caller forgot to floor or reflect are two different defects. Nothing else moves:
the generator is consumed only for the cells that are actually drawn, so every
seeded output of every ordinary fit is byte for byte what it was.

### The one-year CDR refuses a non-annual development grain (#108)

Every route to a one-year claims development result advances the triangle by
exactly one development step: the Merz-Wuthrich closed form, each
`DiagonalGenerator`, and `rereserve`. That step is a year only when the triangle
develops in twelve-month steps. On a quarterly or a monthly fit the same code
answered anyway, so a three-month or a one-month development result came back
labelled as a one-year figure, finite and plausible, with nothing in the output
saying otherwise. `one_year_cdr`, `simulate_one_year_cdr` and `rereserve` now
refuse such a fit, naming the grain they measured, saying that one step is that
many months rather than twelve, and giving the two ways forward: aggregate the
triangle to an annual grain first, from a year-end valuation, because the annual
buckets are anchored to the latest diagonal and a mid-year one gives development
lags the annual grain rejects; or read the run-off uncertainty from
`MackFit.msep_runoff()`, which does not depend on the grain. The gallery `mack`
entry's `one_year_cdr()` and `cdr_distribution()` inherit the refusal. Annual
triangles are untouched, the MW2014 tie-out against R included.

### An origin axis that is not one dev step apart is refused by name (#115)

`nn_data`'s calendar index `cal_idx = w + d + 1` and `stan_data`'s `prev_idx`
both read the origin index and the dev index as one shared clock, which they are
only while one origin step equals one dev step. Two geometries break that
without being malformed data: a gap in the pooled origin axis (accident years
2010, 2012 and 2013, with 2011 missing) and annual origins on a quarterly dev
grain. On the first of those, three cells whose real evaluation date was
2013-12-31 were given calendar indices 4, 3 and 3, so the validation split held
one of them out and trained on the other two, which is training on the diagonal
it is scored on. Nothing raised and nothing about the result looked wrong. Both
geometries are now refused, with a message naming the offending pair of origins,
the gap in months and what to do about it. The check is on the pooled origin
axis, so a single cohort that skips an accident year its neighbours carry is
unaffected: it keeps its row on the shared axis and is masked out as before. What
is asked for is one origin step per dev step, not an annual grain: a quarterly
origin axis on a quarterly dev grain is accepted, and so is a monthly one on a
monthly grain.

One entry pays for sharing a door, and it is named here so the decision is
visible. `stan_data` serves three gallery entries: `meyers_ccl` and `meyers_csr`
read the origin index as a clock, and `guszcza_growth_curve` does not, using it
only to index `ulr[w]` and `premium[w-1]`, exactly as `odp_stan_data` and
`compartmental_stan_data` use theirs (which is why those two contracts are not
checked at all). So Guszcza will now refuse a gapped origin axis it could in
principle fit. That is the cost of putting the check on the shared contract
rather than in two model files, and it is written down in
`require_origin_axis_step` so it can be revisited rather than discovered.

The geometry is reachable from an installed sample, in two lines: 50 of
chainladder's 725 clrd paid-loss cohorts have a gapped origin axis, and fitted
one at a time all 50 are now refused by name. Before this change none of them
reached a wrong calendar index either, but none of them said why: 47 died on a
raw `KeyError` from the premium lookup and 3 on "no usable cohorts". Fitting the
whole sample at once is unaffected, because the pooled axis is then 1988 to 1997
with no hole.

### Anchored dev ages validate, and are refused by name at the kernel doors (#115)

`with_dev_grain('Y')` on a triangle whose latest valuation is a March 31 gives
dev ages 3, 15, 27, matching chainladder's `grain('OYDY')` exactly, which is what
the tie-out test asserts. `validate` then reported all 156 rows of that triangle
as broken and `validate(strict=True)` raised on it, because the rule asked for a
multiple of the grain rather than for one shared offset. The rule now asks every
row to share ONE offset against the declared grain, with the offset at the latest
evaluation date as the anchor, and reports the offsets it found when they are
mixed. When the latest evaluation date carries several offsets itself, which
quarterly origins on an annual dev grain do, the message says so instead of
presenting the smallest as the triangle's own answer. Only positive ages are
given an offset, because `%` follows the backend's sign rule and a negative age
otherwise had a different offset reported on duckdb than on polars for the same
triangle. That is a strict relaxation: a triangle mixing 9-month and 12-month
ages, or carrying a dev age of zero, is still reported.

The kernels take the opposite decision and now say so. Ages off the grain
boundary are a coherent triangle and are not a grid a contract can index, since
every contract stores a cell at `dev_lag // step`, so one helper,
`kernels.contract.dev_step_index`, replaces the six copies of that check and
refuses them by name: it gives the step, the offsets it measured, example ages,
the cause and the two ways out. `kernels.holdout` keeps its own check, which is
about a fit rather than a triangle.

`from_bermuda` now reads the dev grain from bermuda's `eval_date_resolution`
instead of reusing the origin grain, and refuses a resolution it cannot represent
(six months, say) rather than rounding it. Annual periods observed every quarter
used to come back declared OYDY with every cell intact under a wrong label, and
`to_incremental` then kept 2 rows out of 8, looking for each cell's predecessor
12 months back. The origin grain remains the fallback for a single-diagonal
triangle, where the data does not say; the attribute is read directly rather than
through a default, so a bermuda release that renames it raises instead of quietly
restoring the wrong label. One limit of the round trip is now written down: bermuda
carries cells, not declarations, so a triangle declared quarterly that holds only
annual diagonals comes back annual.

### Origin regrain and the two exports refuse a misaligned eval_date (#112)

`change_origin_grain`, `to_chainladder` and `to_bermuda` all work out a row's
development from its `eval_date` and drop the stored `dev_lag`. A row where the
two disagree, which `validate()` had only ever reported as a finding, was
therefore moved into the cell its `eval_date` names: the regrain and the
chainladder export added it to whatever was already there, so two cells of 95
and 150 carried at one evaluation date came out as a single 245 on both
backends, with the triangle total unchanged so nothing downstream could notice,
while `to_bermuda` kept one of the two values for the field and dropped the
other. All three now refuse such a row by name, through the new
`validate.require_eval_alignment`, which shares its wording and its query with
the `validate()` finding. Ingestion is unchanged and still permissive: a
triangle nobody regrains or exports is usable as it is.

A row gets into that state in one of two ways and the refusal names both,
because the way out differs. Either `dev_lag` or `eval_date` is wrong in the
source data, where the repair belongs, or the cell was restated at a later
`eval_date`, which is legal stored history. Slicing never changes a row's
`eval_date`, so it cannot align one, but it can drop it: a restatement goes
under `latest_diagonal()`, or under `as_of()` at a date before the restatement,
and the operation then runs. `as_of()` at or after the restatement keeps the
restated row and the refusal stands.

Separately, `with_origin_grain` at the grain a triangle already has is now a
no-op that returns the same object, as `with_dev_grain` already was.

## 0.5.8 - 2026-08-24

A one-feature release: `nn_transformer_ml` gains held-out scoring (#105), the
adapter notebook 3c's first execution had to work without. Purely additive -
no existing entry's draws or densities change on this version.

### The multi-line transformer scores held-out cells (#105)

The entry's fitted cohort is a company - all of its lines are one training
example - while a held-out cohort built by `next_diagonal` is a single
(company, line) pair. The new `gallery/nn/_heldout_ml.py` bridges the two: it
slices a per-(company, line) contract out of the company-shaped one (putting
`line_of_business` back into the segment key, with the same training-closure
and identity guards every other entry answers to), and the entry now
subclasses `ScoresHeldout` and `PredictsHeldout`, so `log_lik_at` and
`predict_at` work through the ordinary calls. The forward pass conditions on
everything the company observed across all its lines; the scored line's
predictive is read off the head - under the `"joint"` head as that line's
marginal of the multivariate mixture, whose weights are unchanged and whose
per-component scale is the matching row of the Cholesky factor. The density
algebra and draw loop were extracted from `gallery/nn/_heldout.py` and are
shared, not copied; the single-line entries' behavior was verified unchanged
byte for byte. `TransformerMLConfig` gains `heldout_n_draws = 10_000` to match
its siblings, and per-cohort draw streams (0.5.7) apply to the new path
automatically.

## 0.5.7 - 2026-08-24

A one-fix release: per-cohort draw streams (#104), found by notebook 3c's
multi-company study (#103). Draws from entry-level `predict`, `predict_at` and
`cdr_distribution` change for a given integer seed - a different sample from
the same distribution - so any pipeline pinning those draws byte-for-byte
re-pins on this version. The kernel functions' own seed behavior is untouched.

### Gallery draws stop sharing one noise stream across cohorts (#104)

Notebook 3c measured a defect in how the gallery turned a seed into random
numbers. A gallery entry is fitted to one cohort, so a study over twenty-five
companies is twenty-five separate fits, and a script that wants reproducible
results passes all of them the same seed - which is what notebooks 3b and 3c
do. Every `predict` and the shared `predict_at` answered that seed with
`np.random.default_rng(seed)`, starting the same generator from scratch each
time, so draw `i` of every company read the same underlying random numbers and
the companies' simulated ultimates rose and fell together. For `mack` the mean
implied correlation between cohorts came out at 0.255, where independent draws
at 10,000 draws would sit near 0.01 of sampling noise.

Each cohort's own distribution was never affected. What was affected is every
quantity read across cohorts *within a draw*: a company total, a panel total,
the spread of either, and any calibration statistic computed from those sums.

* **The fix.** Entries now derive a stream from the caller's seed before drawing
  anything. The new `kernels/rng.py` builds a short text naming what is being
  drawn - the method (`predict`, `predict_at`, `cdr_distribution`), the cohort's
  segment identity, the loss field, the training cutoff - hashes it with sha256,
  and folds eight 32-bit numbers from that digest into a
  `numpy.random.SeedSequence` behind the seed. sha256 rather than Python's own
  `hash()`, which is salted differently in every process and would make a rerun
  of the same script produce different numbers. Wired into
  `PredictsHeldout.predict_at` (the one shared held-out path, so it covers the
  pooled NN entries too), the nine single-cohort `predict` methods, and both of
  the `mack` entry's kernel calls.
* **What changes for users.** Draws from an entry's `predict`, `predict_at` and
  `cdr_distribution` differ from 0.5.6 for the same seed. Each cohort's own
  distribution is statistically unchanged - the numbers are a different sample
  from the same distribution, not a different distribution. The same seed and
  the same cohort still reproduce bit for bit, in any process and on any
  machine. A `Generator` or `SeedSequence` passed as `seed` is used exactly as
  given: a caller who built their own stream gets that stream, not one derived
  from it, which is also how two entries can still be held on common random
  numbers when a comparison wants that.
* **What does not change.** The kernel functions are untouched:
  `kernels.mack.simulate_ultimates`, `kernels.mack.draw_next_cells` and
  `kernels.cdr.simulate_one_year_cdr` called with a plain integer seed are
  byte-identical to 0.5.6, so the CDR byte pins and the R `ChainLadder` tie-outs
  still hold. Their `seed` annotation widened to
  `int | np.random.SeedSequence | None`, which is documentation of what
  `np.random.default_rng` already accepted, not a behavior change. The NN
  entries' `predict()` is deliberately untouched as well: one pooled fit runs a
  single cached rollout and slices it per cohort, so its cohorts already read
  different positions of one stream and the defect never arose there.

## 0.5.6 - 2026-08-12

A one-fix release: the case-level floor in `nn_paid_case` (#100), on `main`
since 2026-08-06 but not in any published wheel until now. Also ships the
`schedule_p` docstring correction (#99): `case_reserve` is reported minus paid
(net of bulk), not incurred minus paid - the stored values were always right,
the description of them was not.

### `nn_paid_case` floors the simulated case level at zero (#100)

A case reserve is booked down TO zero and never past it, and 0.5.5's rollout did
not say so: it advanced the state as `level += movement` with nothing stopping
the walk crossing zero, and roughly half the simulated terminal levels on the
Schedule P panel landed below it (55% with the transformer body, 45% with the
GRU, against zero of that panel's 900 observed case cells). The state update is
now `level = max(level + movement, 0)`.

* **New config field** `NNPaidCaseConfig.floor_case_at_zero`, default `True`. It
  is a rollout knob shared by both backbones, so it is not in `BACKBONE_KNOBS`
  and neither body refuses it. It is also in the rollout cache key, unlike every
  other config field, because it is the one knob a caller is meant to flip on a
  FITTED entry - keyed on `(n_draws, seed)` alone, that flip returned the other
  arm's cached array, byte-identical and with no error.
* **The draw is untouched.** The clamp is arithmetic on the case STATE, applied
  after the joint (paid increment, case movement) sample; the paid coordinate is
  written unchanged and the generator has already advanced. The two arms
  therefore consume the same random stream - same generator, same call order, a
  common-random-numbers pairing - and are byte-identical **until the first
  bind**, which is why `floor_case_at_zero=False` reproduces the 0.5.5 walk
  exactly (verified against the published 0.5.5 wheel in a clean venv, on
  `case_paths()` and `predict().samples`, on both backbones). Past the first
  bind the floored arm feeds the network a different case level, so its later
  samples legitimately differ.
* **The floored level is the only copy of the state**: it is what feeds channel 1
  forward on the next step AND what `case_paths()` reports, terminal and
  per-diagonal alike. A rollout that clamped the read-out while handing the
  network the unfloored level would pass every diagnostic; it is caught by test.
  One exception, pre-existing and unchanged by the floor: at a PINNED dev the
  channel-1 input is masked to the pin (standardized 0, the pooled level mean),
  so the network sees the pin there and not the level, floored or not.
* **The starting level is not floored.** `_initial_case_level` carries the
  deepest observed case level as the triangle reported it, negative included; a
  recovery can outrun the case estimate, and restating an observation is not
  constraining a simulation. The consequence is narrow: every origin the rollout
  projects has a cell on the first future diagonal, so a negative start is
  clamped there and never reaches `case_paths()`. Only an origin with no future
  cell at all can show one.
* **What moves and what does not.** `predict()`'s rollout ultimates CAN move
  with the floor on - the floored level feeds back and changes the next
  diagonal's mixture - and on the small test fixtures that movement is nonzero
  but small. The held-out board columns do NOT move: `predict_at` /
  `log_lik_at` are a single forward pass at observed features with no level walk
  in them, so they are byte-identical between the two arms (asserted, which
  doubles as proof that the two fits trained identically).

`floor_case_at_zero=False` is kept so the change can be measured with and
without it, not as a fallback. Measure it on `predict()`'s ultimates and on a
Meyers-style retrospective - **not** on the held-out board, whose rows are
identical between the arms by construction.

## 0.5.5 - 2026-08-06

The case-reserve arc built on 0.5.4's channel machinery lands: deeptriangle
learns from the case-reserve level it previously refused, and `nn_paid_case` -
the sixth NN entry - models paid development and case-reserve dynamics jointly.
Three PRs (#94, #95, #96) plus a repo-wide wording pass.

### The case-reserve head (#94)

0.5.4 made deeptriangle refuse a level at channel 1 of its auxiliary task by
name, because level-minus-increment is neither the outstanding increment nor
the outstanding level. That refusal is now replaced by the case-reserve head:
when channel 1 carries a level (inferred from `field_kinds`, no new knob), the
auxiliary MDN trains on the case-reserve level itself, masked to cells where
both channels are real. The increment path is byte-identical to 0.5.4 for the
same seed (measured), so existing fits are unchanged.

### `nn_paid_case` (#95)

A joint model of paid development and case-reserve dynamics, motivated by what
a case reserve is: a state with dynamics, not a static covariate - it runs down
toward zero as payments replace it and jumps when new information arrives. Per
cell, a K-component bivariate Gaussian mixture predicts (paid increment, case
movement) with full per-component covariance, so the correlation between payment
and case run-off is a learned per-cell quantity. The case LEVEL is an input
channel the rollout advances (`level += movement`, in ratio space) and feeds
back - both channels write back, both flags promote, so the frozen-feature
rollout limitation the other five NN entries disclose does not apply here. Two
switchable backbones (`config.backbone = "transformer" | "gru"`) share one head
module; foreign knobs are refused by name. Training is mixed-observedness: the
joint density where both targets are real, the closed-form margin where one is -
no cell discarded, no target fabricated. Held-out scoring takes the paid margin
of the joint density (a bivariate mixture's margin is a univariate mixture,
test-pinned against the raw head output) through the shared pooled-MDN path, so
the entry joins the board column-comparable at 10,000 draws. `case_paths()` is
the case run-off diagnostic: terminal simulated case levels per draw, or the
full walk over projected diagonals with `per_diagonal=True`. The case path's
calibration is unvalidated in this release (no realized-case board column) and
the card says so.

### Card fixes (#96)

The mdn card's evaluate example used a one-key `segment` that raises whenever
the pooled fit carries more than one line for a company (the normal case); it
now uses the two-key form, matching the `predict` call above it and the other
NN cards. The transformer card's placeholder set literal in the same slot is
spelled out the same way. `transformer_ml`'s one-key example is correct as it
stands - that entry's cohort unit is the company - and is deliberately
unchanged.

### Wording pass

Prose that described with-and-without comparisons through a lab-jargon term
now says what it means: "switchable", "comparison arm", "variant". Cards,
docstrings, comments and test prose across the package; one test function in
`tests/test_odp_bootstrap.py` renamed to `test_both_switches_off_is_refused`.
No behavior change.

## 0.5.4 - 2026-08-05

Case reserves (and any eval-date snapshot) become usable NN input channels
with honest semantics, and the NN entries' held-out CRPS draws stop being
rationed by the rollout budget. One PR (#91), adversarially reviewed.

### Per-channel observedness: `x_obs`

`kernels.nn_contract.nn_data` now returns `x_obs`, a per-channel usable-value
mask whose channel 0 equals `obs_mask` exactly, and every NN entry conditions
per channel: tokens are `[values * chan_flags, chan_flags]`, element-identical
to the old per-cell form on a single-channel fit (seeded draws are
byte-identical). Two defects the cards used to disclose as limitations are
closed. A rollout cell promoted to context no longer presents the contract's
padding zero as an observed zero feature increment - promotion raises only the
target channel's flag, so next year's features stay what they are, unobserved.
And deeptriangle's auxiliary outstanding head now trains only where BOTH its
channels are real; the fabricated pre-fix target at a punched fixture cell was
measurably negative outstanding, paid exceeding reported.

### `level_fields`: snapshots carried undifferenced

`fit(feature_fields=("case_reserve",), level_fields=("case_reserve",))` is the
new consumer spelling. A field named in `level_fields` skips differencing -
`case_reserve` is a snapshot whose difference is the case movement while the
informative quantity is the level - and `field_kinds` records each channel's
kind. The target cannot be a level, a level must also be a feature, and an
absent field is refused by name (`select_fields` is a filter, so the old
behaviour fit a dead all-masked channel silently - the refusal immediately
caught two test fixtures that had been doing exactly that since the
deeptriangle entry landed). deeptriangle refuses a LEVEL at channel 1 for its
auxiliary task by name - level-minus-increment is neither the outstanding
increment nor the outstanding level - with `config(aux_weight=0.0)` as the
single-task escape.

### `heldout_n_draws`

The four held-out-capable NN configs gain `heldout_n_draws = 10_000`, read by
the held-out draw path instead of the rollout's `n_draws` (a held-out diagonal
is one forward pass per ensemble member regardless of draw count, so 10,000 is
cheap there and was not in a rollout). The NN rows of a CRPS board now rest on
the same draw count as every other entry; they carried 1,000 before, the
rollout default inherited by coincidence. `transformer_ml`'s config does not
gain the field - it has no held-out surface, and an unread field would be an
inert parameter.

### One disclosed edge, kept deliberately

Training gates every channel at the drawn augmentation cutoff; the rollout and
held-out paths gate features by observedness alone. On a ragged cohort whose
feature is booked on a deeper calendar diagonal than every target cell, the
network therefore conditions on a feature cell at a distance training never
showed. Kept, because that contemporaneous cell is genuinely informative; all
five cards state it.

## 0.5.3 - 2026-08-04

Released without a changelog section at the time; backfilled here from the
commit body. One consumer-visible change: Schedule P gold-mart downloads go
over anonymous HTTPS (stdlib urllib) with the `gh` CLI demoted to a fallback,
so a fresh clone with no GitHub tooling works.

## 0.5.2 - 2026-07-29

Two review findings against 0.5.1's own work, no public surface change and no
shipped behaviour change - both defects are in a study script and a test guard.
Released rather than held because one of them would have put unreliable MCMC
draws on a published leaderboard.

### The parameter-count guard could not see an unpinned count

0.5.1 pinned every parameter count the NN cards disclose. The regex that read
those values was also answering "does this card disclose a count?", and in that
role it could not see `~30k parameters` - the exact prose format the guard was
written to stamp out, and the one `mdn/card.md` had carried alongside a
transformer figure the transformer's own card had already retracted. A card
could revert to prose, sit in the "discloses nothing" list, and pass.

Detection is now a separate, wider pattern, used both ways: a card listed as
disclosing nothing must contain no count-shaped phrase at all, and a card that
DOES disclose must have every such phrase inside a bolded claim that a builder
rebuilds. It requires the literal word "parameters" after the number, which is
what keeps it off the historical mentions the cards legitimately carry ("quoted
~120k, which was never the number").

### Four defects between the worker pool and the published board

All in `scripts/heldout_leaderboard.py`, found by review of the script that
lands milestone 6's study. Two would have corrupted its output and two would
have aborted or misreported it.

* **A fit that failed its convergence gates at the final escalation stage was
  scored onto the board.** `run_retro` uses the gates to decide what to
  *re-run*; it does not discard a fit that still fails at the last stage,
  because that is the caller's decision - and this script was not making it. A
  fit with R-hat 1.4 contributed draws to a published board with nothing saying
  so. It is now downgraded to a `fit_failed` absence rather than dropped:
  `align_panel` intersects, so a silent drop would delete those cells from
  every OTHER model's column, making a badly-converged fit look good while
  costing everyone coverage. One `ConvergenceGates` instance is shared with
  `run_retro` so escalation and judgement cannot drift.
* **`heldout_stacking.json` was written only on success**, leaving a previous
  run's weights beside freshly-overwritten CSVs - an output directory that
  looked complete and self-consistent and was not. Written either way now, with
  the failure payload naming the reason.
* **A pool-machinery failure crashed the progress callback.** Those rows carry
  no `as_of` (the task never got far enough to have one), so indexing it raised
  `KeyError` inside the callback, aborting the study and hiding the real
  failure behind a missing-key traceback.
* **The documented default invocation could not run.** It selects every board
  entry, needing `[bayesian]` and `[nn]`, and a plain `uv sync` installs
  neither; it failed late inside a spawned worker as an `ImportError` about
  cmdstanpy. A preflight now names the extra and the models that want it and
  exits before the mart is touched, checking with `find_spec` so torch is never
  imported. The entry-to-extra map is derived from the registry.

## 0.5.1 - 2026-07-29

A patch in version number only where the CDR is concerned: the one-year CDR
became multi-method and then gallery-wide, `guszcza_growth_curve` closed the
last parity gap, and two drift guards landed. Nothing was removed or renamed and
no published number moved; `ibnr.gallery.__all__` grew by one name.

### Multi-method one-year CDR (out of band)

"The one-year CDR" was one method - Mack's - because the two things it does were
fused. They are now two axes, and only one of them is a choice.

* **Axis 1, what generates next year's diagonal, is now selectable.**
  `kernels.cdr.DiagonalGenerator` with two implementations: `MackDiagonal`
  (Mack's conditional moments, the previous behaviour) and
  `ODPBootstrapDiagonal` (England & Verrall's Pearson-residual bootstrap with
  over-dispersed Poisson process noise - the generator behind R's
  `CDR.BootChainLadder`). `simulate_one_year_cdr(fit, generator=...)` takes a
  method name or a configured instance.
* **Axis 2, how the reserve is re-estimated afterwards, is not a choice.**
  `kernels.cdr.rereserve(fit, next_diagonal)` re-runs the volume-weighted chain
  ladder on the extended triangle and differences the ultimates. One
  implementation shared by every generator - it is the market convention and
  what R uses for *both* of its CDR methods. It is public, so draws from any
  model that predicts next year's cells can be re-reserved into a directly
  comparable CDR.
* **`kernels/odp_bootstrap.py` (new)**: the bootstrap engine as free functions
  over plain arrays, written to be read against R's `BootstrapReserve.R` and
  documenting each of the four places it deviates.
* **The option surface**: `cdr_methods()` lists every route with what it
  generates, how it re-estimates, what it returns, what it requires of the
  cohort and - route by route - what it has actually been validated against.
  `get_cdr_method(name)` returns the descriptor, carrying the generator
  **class**, mirroring `gallery.get`.
* **`merz_wuthrich` is listed but is not a generator, and asking for it as one
  is refused by name.** The closed form linearizes the chain-ladder factor
  update around Mack's conditional moments; there is no version of it for
  another model. `one_year_cdr(fit)` is unchanged and remains the only way to
  reach it.
* **Mack's precondition moved off the shared path onto its own generator.**
  `require_positive_open_diagonals` was applied to every simulated CDR; it is
  Mack's (his conditional variance is proportional to the diagonal cell) and the
  bootstrap has the opposite requirement (non-negative increments, and it
  answers happily for an accident year with zero paid at 12 months). Each
  generator now states and enforces its own.
* Honest limits, stated in the card and the docstrings: the bootstrap route is
  validated **to Monte Carlo error against R's algorithm**, not to published
  digits - R's `CDR.BootChainLadder` example prints none, and a bootstrap is
  stochastic. The test suite transcribes `getNYCost` literally and requires
  agreement to 1e-10 on a shared diagonal, checks the process-only standard
  error against a delta-method reference computed off the re-reserving Jacobian,
  and cross-checks the fitted values against `england_verrall_odp`'s iterative
  proportional fit. Also documented: `E[CDR] = 0` is Mack's, so
  `simulated_msep`'s mean square about zero is the variance only on the `mack`
  generator - R reports `sd()` for its bootstrap route for the same reason.

**No published number moved.** `simulate_one_year_cdr`'s Mack path is
bit-identical to 0.5.0's for the same seed - verified over 49 arrays spanning
two triangles, both sigma rules, all three process laws, both `parameter_risk`
settings and two draw budgets, compared as raw bytes - and the R MW2014 golden
tie-out is untouched. `process`/`parameter_risk` default to `None` on the
signature instead of `"gamma"`/`True` so that "not supplied" is distinguishable
from "supplied"; the applied defaults are unchanged, and combining either with
an explicit `generator=` is refused rather than left inert.

### The one-year CDR opens to the gallery

`rereserve` being public was the door. **`ibnr.gallery.GalleryDiagonal`** is a
third `DiagonalGenerator` that takes a fitted entry with `PredictsHeldout` and
re-reserves the draws it already produces for the leaderboard's CRPS column, so
CCL, CSR, ODP, Clark and Mack reach a one-year CDR with no new theory. It lives
in the gallery rather than `kernels`, since it imports `PredictsHeldout` and the
re-export direction is gallery -> kernels only; `cdr_methods()` still lists the
route, naming the class as a string.

* **What the number is, because it will be misquoted.** It is the *chain
  ladder's* one-year CDR under model M's view of next year, not "model M's
  one-year CDR". Both differenced ultimates are chain-ladder ultimates and only
  the diagonal between them is the model's - the structure R's
  `CDR.BootChainLadder` already has. The honest alternative refits M on the
  extended triangle once per draw; it is not offered rather than approximated.
  Consequently `E[CDR|D_I] = 0` does **not** hold here (it is Mack's result), so
  read `mean_cdr` and `sd_cdr` from `cdr_risk_measures` and not
  `simulated_msep`, which folds a disagreement between two methods into
  something that reads as volatility.
* **Two limits, both refused by name rather than assumed.** The route is
  **backtest only** - `next_diagonal` builds cells only from observations that
  already exist after the cutoff, so a *current* valuation is not reachable this
  way (the `mack` and `odp_bootstrap` generators are unaffected and remain
  prospective). And it excludes the **NN entries and `compartmental`**, whose
  contracts keep no raw cumulatives for the training-history check below.
* **All three objects are bound to each other, not two of them.** A `MackFit`, a
  `HoldoutCells` and a fitted entry come from three calls. Two checks tie the
  cells to the fit; the third ties the entry to both, by comparing its own
  contract values against `fit.cum`. Without it, an entry refitted on restated
  *interior* history - latest diagonal untouched, so every other check passed -
  was accepted, and the total CDR mean moved from 0.27 to -367.69 with seven
  times the spread, every number finite. Found by review.
* `simulate_one_year_cdr(n_draws=...)` and `Mack.cdr_distribution(n_draws=...)`
  now default to `None`, meaning "this generator's own count".
  `DiagonalGenerator.resolve_n_draws` is the seam: a Monte Carlo budget for the
  two simulating generators (both resolve `None` to the previous literal 20,000,
  so no existing call changes) and the source's own size for a fitted posterior,
  which refuses a mismatched explicit count instead of resampling to it.

### Documentation and drift guards

* **A "Chain ladder & reserve risk" reference section** on the docs site. The
  CDR and Mack kernels had no reference page at all, so `cdr_methods()` - the
  discoverability feature - was not discoverable. Thirteen entries, all
  resolving statically under `dynamic: false`.
* **Every parameter count an NN card discloses is pinned to the network that
  builds it.** `transformer/card.md` had retracted a `~120k` figure and the
  retraction never reached `mdn/card.md`, which had copied the comparison; mdn's
  own "~30k" was 26,353. Cross-card references get a builder each, every `nn`
  entry must be classified as disclosing a count or not, and the "does not
  disclose" list is verified against the cards rather than trusted.

### Milestone 5 - `guszcza_growth_curve` joins the parity gate

The entry landed 2026-07-27, two days after milestone 5 was declared complete, so
it was the only Bayesian entry without NumPyro and PyMC ports and the only one
absent from `scripts/parity_gallery.py`. Its card had recorded the ports as a
follow-up task, so this was a deferral rather than a deliberate exclusion.

* **Both ports added** - `model_numpyro.py` and `model_pymc.py`, plus `_shared.py`
  for the pieces that are not PPL-specific (curve codes, tree depth, the data
  guard) so the two cannot drift apart. `BACKENDS` is now
  `("stan", "numpyro", "pymc")`.
* **4 of 4 parity against the Stan reference**, two WC companies as of
  1997-12-31, 4 chains x 2500 draws: every z-score below 2.4 against a tolerance
  of 4, zero divergences in every backend on every cohort, R-hat 1.00. Published
  in `analysis/results/{parity,convergence}_guszcza.csv` and tabulated in the
  card. New `kernels.parity.GUSZCZA_PARITY_VARS`.
* **`scripts/parity_gallery.py` gained `--nuts-sampler`**, which reaches the
  `pymc` leg only. This entry needs it: at the `adapt_delta = 0.999` its source
  specifies, PyMC's native PyTensor NUTS runs ~130x slower than the identical
  graph through JAX (measured 860 s against 6.7 s), the same situation
  `compartmental` documents. `convergence()` now reports the sampler's own label
  (`pymc:numpyro`) rather than the backend argument, so a published row says
  what produced it.
* **`max_treedepth` is a shared control on this entry**, not a cmdstan-only one:
  its default of 15 differs from both PPLs' default of 10, and holding it
  constant is part of what parity means. Only `parallel_chains` is refused by
  the ports.
* A standing requirement is now recorded in CLAUDE.md: a new `bayesian` entry is
  not done until it has both ports and a published parity row. What expired with
  this gap was the milestone number, not decision 7.

## 0.5.0 - 2026-07-27

The 0.4.0 wheel on PyPI was 49 commits behind `main`, so this release is mostly a
catch-up: milestone 5 finished, milestones 6 and 7 opened, the one-year CDR landed
out of band, and the package moved to numpy 2 and grew a test CI.

### Public API - the cohort vocabulary

Five gaps found by building `analysis/03` through the public API alone, and they
were one gap seen five times: **a fitted entry could not say which cohorts it
answers for**, so every caller reconstructed that fact by hand.

* **`GalleryEntry.cohorts()` (new, abstract)** returns every cohort the fit
  answers for, in `predict()`'s target order, as the cohort's FULL segment
  identity - including a column the fit's own key does not carry.
  **`cohort_index(segment)`** is the one resolver behind it: `segment` is a
  *filter* on this fit's cohorts, so any subset naming exactly one is accepted
  and one naming none raises (naming the fit's key and the supplied dict) rather
  than quietly scoring the fitted cohort.
* **`predict`, `realized_ultimates` and `evaluate` now take the identical
  leading `segment: Mapping | None = None`** on all 15 entries.
  `realized_ultimates` moved onto the ABC. Previously the NN entries took a
  segment dict and the other ten took none, so a cross-model outcome table
  needed a `family == "nn"` branch. Every existing call site passes its
  arguments by keyword, so no 0.4.0 caller moves.
* **A pooled NN fit can be scored on the mart's own cells.**
  `kernels.nn_contract` keeps display-only segments (`company_name`) out of the
  cohort key, so a pooled fit was keyed on two columns while `next_diagonal`
  built cells on three - and both `entry.log_lik_at(cells)` and
  `entry.at_cohort({...}).log_lik_at(cells)` failed, the first naming a column
  the caller had just passed. The cells are now re-keyed onto the fit's own
  schema at the mixin boundary (`HoldoutCells.narrowed_to`), with each dropped
  value **verified** against the fitted cohort first; `index_into`'s schema
  equality is left exact, and the narrowing never escapes, so a shared board
  still sees one segment schema. Backed by a new refusal in `nn_data`: a segment
  column dropped from the key must be a *function* of the key, or two cohorts
  would collapse onto one grid - silently, whenever their cells are disjoint.
  `nn_data`/`nn_company_data` gained `segment_columns` and `display` keys. That
  refusal was measured against the real mart before shipping, since the data
  model derives `company_name` through a LEFT JOIN and a null or second spelling
  would fire it on every pooled fit: on publish `20260613_041006` all four
  Meyers lines carry 353 company codes with zero null names and zero codes
  spelled two ways, and both the study's pooled panel (60 companies /
  152 cohorts) and the full `--nn-pool market` pool (221 / 405) build clean.
* **`gallery.get(name).config_class`** is the dataclass an entry's
  `fit(config=...)` takes, or `None`. Registration checks the declaration both
  ways - missing when `fit` takes a config, and stale when it does not.
* **`ibnr.gallery.__all__` gained `next_diagonal`, `CohortForecast`, `Absence`,
  `align_panel` and `SCORE_DIRECTION`**, the four steps that BUILD the panel
  `leaderboard()` consumes plus the direction the board has no default sort for.
  The rule, now written in the module docstring and CLAUDE.md decision 8: a name
  is exported if a caller must construct or call it to get from a fitted entry
  to a board row. `ibnr.kernels` re-exports the same names plus `ForecastPanel`;
  `kernels` still never imports the gallery, and there is a subprocess test for it.
* **For anyone subclassing `GalleryEntry` out of tree:** `cohorts()` and
  `realized_ultimates()` are now abstract, so a subclass that implements neither
  will not register. There are no such subclasses (`gallery.scaffold()` is
  unbuilt), and the CALL surface CLAUDE.md protects is unchanged.
* `kernels.multiline.multiline_data` now stamps `segment` and `measure`, the
  cohort identity SUR and the copula GLM answer for.
* Note on `analysis/03`: its committed run predates all of this. Its
  `drop("company_name")` workaround and `family == "nn"` branch still run
  correctly - they are simply no longer necessary.

### Packaging

* **Python 3.11 and 3.12.** `requires-python` is now `>=3.11,<3.13`, and the
  classifiers say the same. The ceiling is a deliberate choice rather than a
  limit of the code: the core install, `[polars]`, `[nn]` and `[viz]` were all
  measured resolving wheels-only on 3.13 and 3.14. `[bayesian]` and
  `[interop]` cannot follow, because both transitively pin numpy below 2 -
  through `arviz` 0.18 (pulled by `bayesblend` 0.0.8) and through
  `bermuda-ledger` 2.3.0 - and the newest numpy under 2 is 1.26.4, which does not
  support Python 3.13 at all. Since packaging metadata cannot say "3.11 to 3.14
  unless you asked for `[interop]`", the package claims one range for everything
  it ships and a 3.13 user gets the standard "requires a different Python"
  refusal (`pip` on a clean 3.13 venv: `Package 'ibnr' requires a different
  Python: 3.13.13 not in '<3.13,>=3.11'`). Revisit the cap when those two
  upstreams move. (An alternative
  that gated the two extras on a deliberately unregistered package name was
  built and then rejected: it only holds while nobody uploads that name.)
  Caveat worth knowing on the versions that are supported: on 3.12 `[bayesian]`
  installs but not wheels-only, because `bayesblend` pins `matplotlib==3.7.2`
  whose newest wheel is cp311, so 3.12 builds it from source.
* **numpy 2.** Development and CI now run numpy 2.4.6; a plain `pip install ibnr`
  resolves numpy 2.5.1 and pandas 3.0.5. The core floor stays `numpy>=1.26`
  because raising it to 2 would make `ibnr[interop]` and `ibnr[bayesian]`
  unsatisfiable on PyPI - the blockers are upstream metadata on code that works,
  and this repo bridges them with `[tool.uv] override-dependencies` rather than
  shipping metadata nobody can install. Measured: the core suite gives the same
  557 passed / 0 failed on the locked numpy 2.4.6 + pandas 2.3.3 and on numpy
  2.5.1 + pandas 3.0.5, and all extras together give 0 failed.
* **pymc 5.28.5 / pytensor 2.38.3** were not an optional upgrade. pytensor
  2.31.7 unpacks numpy's `einsum_path` result as a five-tuple and numpy 2.4
  returns three, so every LKJ-based compartmental test died on "not enough
  values to unpack" the moment numpy moved. Worth knowing before anyone pins
  pymc back.
* **`ibis-framework` is capped below 13.** 12.0.0 is what is locked and what every
  CI leg runs. Transforms are written around backend-specific ibis behaviour, so a
  major bump has to be a deliberate change with the dual-backend suite re-run.
* **`arviz<1` and `pymc<6`** in the `[bayesian]` extra. Both are measured breaks:
  arviz 1.x drops the `az.from_dict(posterior=...)` signature the parity tests
  use, and pymc 6.x changed `LKJCorrRV.rv_op`.
* **`chainladder>=0.9.2`** in `[interop]`, raised from 0.8.18 - the old floor let a
  fresh install resolve 0.8.26, which is not the version the tie-outs were
  validated against.
* **`py.typed`.** The package now ships the PEP 561 marker, so type checkers use
  the annotations instead of treating `ibnr` as untyped.

### Testing and CI

* **A pytest workflow, and gates that make a green run mean something.** Nine
  legs: core, core on 3.11, one per extra, everything at once, plus two that
  install without the lockfile and force numpy and pandas to their newest
  releases - the only legs that grade the resolution a downstream consumer
  actually gets. Each leg declares what it installed and which files it exists to
  exercise, and fails if a test was skipped because a package that leg installed
  could not be imported, if a named file contributed zero executed tests, or if
  the total falls below a floor.
* **`tests/test_import_purity.py`.** Walks every submodule under a blocker that
  refuses the optional extras, and checks the four public import paths pull in
  none of them. This is what makes "the core install stays light" a checked claim
  rather than a convention.
* A `test` dependency group carved out of `dev`, so a genuinely core-only
  environment can be built at all.

### Milestone 5 - cross-backend parity (complete)

* NumPyro and PyMC ports for `meyers_csr`, `england_verrall_odp`,
  `clark_growth_curve` and `compartmental`, each gated against the Stan reference
  posterior.
* **The parity gate was silently lenient.** It read `az.summary`, which rounds to
  three decimals, so any parameter whose MCSE rounded to zero scored a perfect
  z - precisely for the best-identified parameters. The published milestone-4 CCL
  figures came through this bug and are superseded.
* **`meyers_ccl`'s `a_ig` bound is load-bearing.** Stan declares
  `<lower=0, upper=1e5>` and both ports had left it unbounded, because the
  truncated *prior* mass is negligible - but the *posterior* piles into that
  corner at deep development lags and pulled `sig` 5-12% low. Restored in all four
  ports.
* `nuts_sampler` is exposed through every Bayesian entry's `fit()`.

### Milestone 6 - held-out scoring, stacking, leaderboard (in progress)

* `kernels/holdout.py` (which cells a fit at a given `as_of` is scored on),
  `kernels/densities.py` (the one place a density changes measure),
  `kernels/forecast.py` (the forecast object and the leaderboard) and
  `kernels/stacking.py` (bayesblend stacking over two panels).
* A forecast carries two independent capabilities - a density, giving ELPD, and
  draws, giving CRPS - each with its own panel membership, so one model's refusal
  cannot delete cells from a column it does not appear in.
* Held-out scorers and predictors for CSR, CCL, compartmental, guszcza, ODP,
  Clark, Mack and the NN family.
* `kernels/codec.py` - the wire format. `to_arrow`/`from_arrow` round-trip
  `PredictiveDistribution`, `Triangle`, `MackFit`/`MackFitPanel`, the CDR
  result, forecast panels and plain frames losslessly over Arrow IPC, with a
  summary-only JSON mode for callers that cannot take megabytes of draws;
  `peek_kind` routes a payload without decoding it. This promoted `pyarrow>=15`
  from the dev group to a core dependency - no new weight, since
  `ibis-framework[duckdb]` has always pulled it transitively, but the codec
  imports it directly and an inherited requirement can vanish in an upstream
  release.

### Milestone 7 - the rest of the NN family (in progress)

* New entries: `mdn`, `deeptriangle`, `resnet`.
* A shared NN training scheme and one shared held-out mixin across all four NN
  entries.
* `kernels/tuning.py`: random-search hyper-parameter search over NN configs.

### New gallery entries

* `guszcza_growth_curve` - hierarchical growth curve reserving (Gesmann/Guszcza).
* `mdn`, `deeptriangle`, `resnet` (see above).

### One-year CDR (out of band)

* `kernels/cdr.py`: the Merz-Wuthrich 2008 analytic one-year claims development
  result per accident year and in total, plus an "actuary in the box" re-reserving
  simulation, plus VaR/TVaR of the CDR loss. Ties out to R ChainLadder's published
  `CDR()` output to seven decimal places.
* `kernels/mack.py` gained a native distribution-free chain ladder and
  `fit_mack_many`, a batch fit that closes the multi-cohort gap against
  chainladder-python's vectorized point estimate.

### Milestone 9 - speed benchmark (complete)

* `scripts/benchmark_speed.py` times construction, cumulative/incremental
  conversion, `as_of`, grain changes, aggregation, parquet ingestion and Mack fits
  against chainladder-python on both ibis backends. Headline: scale decides -
  chainladder wins small in-memory transforms, ibnr wins at mart scale and on
  every Mack fit.

### Fixes

* **`fit()` is atomic across every gallery entry.** A failed refit used to leave a
  half-updated entry behind; it now leaves the previous state untouched.
* **Null segment keys are rejected at ingestion.** Every join in
  `triangle/transforms.py` is a plain equi-join, and SQL join equality is false
  for `NULL = NULL`, so a single null segment value silently deleted a whole
  cohort from `as_of`, `latest_diagonal` and `to_incremental` - identically on
  both backends, and reachable from the real mart.
* Premium must match the loss cohort, and fields must share a diagonal.
* The zero-variance guard survives density columns that mix finite and `-inf`.
* Two inert parameters removed (`line_embedding_dim`, and `n_lob` in
  `_mack_tail_variance`) - both accepted, neither ever read.

### Tooling

* ruff floor raised to 0.16 across the repo, and CI checks formatting again.
* Python code blocks inside markdown are linted and formatted like source
  (`scripts/lint_md_snippets.py`), which also rejects doc samples that do not
  parse.
