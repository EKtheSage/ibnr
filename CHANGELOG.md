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
`gallery.list/get/fit/stack/leaderboard/next_diagonal/CohortForecast/Absence/align_panel/SCORE_DIRECTION/GalleryDiagonal/reserve_rows`
- `evaluate` is a method on a fitted entry and `scaffold` is planned, per the
corrected decision 8 - and, from 0.7.2,
`methods.chain_ladder/bornhuetter_ferguson/cape_cod/mack/ReserveResult`.)

From 0.7.2 the minor number is kept for milestones: additions and fixes ship as
patch releases, and 0.8 comes when the roadmap's goals are done.

## Unreleased

### Mack with development options

`methods.mack` takes the chain ladder's development options, and Mack's
standard errors follow them (Mack 1999, "the standard error of chain ladder
reserve estimates: recursive calculation and inclusion of a tail factor").

**New in `ibnr.methods`:**
- `methods.mack(cells, average=..., history_periods=..., drop_high=...,
  drop_low=..., preserve=..., drop_above=..., drop_below=..., exclude=...,
  exclude_valuations=..., trim_ties=..., exhausted_exclusions=...)`: every
  option `chain_ladder` takes except `unsupported_factor`, with the same
  defaults. The same options choose the same link ratios and give the same
  factors, bit for bit, as `methods.chain_ladder`.
- `average` is Mack's alpha: `"simple"` 0, `"volume"` 1, `"regression"` 2. A
  link ratio is weighted by the amount it starts from to that power, in the
  factor and in sigma, and one step's variance is sigma squared times the
  amount to the power 2 - alpha.
- A Mack result has a `link_ratios` table, the chain ladder's for the same
  options (0.7.2 returned `None`, and `to_polars("link_ratios")` refused).
- `development` gains `n_selected`, `extreme_trimming_skipped`,
  `bounds_skipped` and `sigma_extrapolated` for Mack: the sigma came from
  `sigma_rule` because the age kept fewer than two link ratios.

**Refused, by name:** `average="median"` and `"geometric"` (`not_supported`: a
median or a geometric average is not a weighted mean of the link ratios, so
Mack's variance does not exist for it; the chain ladder still takes the
median); `history_periods=1`, and any options that leave at most one link ratio
at every age (`variance_not_estimable`); an age the options leave with no link
ratio (`no_link_ratio`: a factor of 1.0 there would be chosen, not estimated,
and would have no variance); `zero_cells="observed"` with any development
option on a triangle with a link ratio out of a zero (`not_supported`: that
ratio has no value to rank, bound or window); `average="regression"` with an
origin whose latest amount is 0 under `zero_cells="missing"` (`not_supported`:
the variance does not shrink with the amount, so it would get a mean of 0 and
a positive standard error); and a sigma the fill cannot reach, naming the
options that thinned the ages. No reason code was added.

**In `ibnr.kernels`:**
- `fit_mack_grid(grid, *, sigma_rule="mack", zero_cells=None, average="volume",
  links=None)`, and the same `average` and `links` on `fit_mack` and
  `fit_mack_many`. `links` is an `ibnr.kernels.links.LinkRules`. With neither
  set, the fit is 0.7.2's, byte for byte. `zero_cells=None` means
  `"observed"`, or the rule `links` carries; the two given with different
  values are refused.
- `MackFit` gains `average`, `links` and `selection` (the
  `kernels.links.LinkSelection` the rules made), and the properties `alpha`
  and `all_history_volume`. With options, `s` is the weight total behind each
  factor and `n_obs` and `n_pos` count the link ratios used.
- `msep_runoff`, `simulate_ultimates` and `draw_next_cells` use each fit's
  alpha. A sigma at an age before the last that keeps one link ratio is filled
  from one log-linear regression over every age with a positive sigma, or by
  Mack's rule from the two ages before, in age order, so an age after a filled
  one uses the filled value; both are R's.
- The one-year claims development result (`one_year_cdr`,
  `simulate_one_year_cdr` with either generator or `GalleryDiagonal`, and
  `rereserve`) refuses a fit made with any development option, naming each
  (`not_supported`). It reads the settings, not what they removed:
  `rereserve` re-runs the volume chain ladder over every link ratio on next
  year's triangle, where `history_periods=9` on a 9 x 9 triangle does remove a
  ratio. `cdr_methods()` lists the condition.
- `kernels.links` gains `is_all_history` and `settings_named`.
- The wire format: a `MackFit` with options carries them and its selection, and
  is written as codec version 2, which 0.7.2 refuses rather than read as a fit
  without them. `CODEC_VERSION` is 2. Every other payload, a `MackFit` without
  options included, keeps version 1 and its bytes.

**Checked against R ChainLadder 0.2.21** (`MackChainLadder(Triangle, weights,
alpha, est.sigma)`, frozen in `tests/data/r_mack_alpha_weights.json` by
`scripts/r_mack_alpha_weights.R`, because CI has no R): raa, genins, ukmotor,
abc and mw2014, each under the three averages, with every link ratio, the
latest 5 and the latest 3, with and without an exclusion, and with the highest
or the lowest ratio dropped, 27 settings, under both of R's sigma rules: 270
fits, of which R answers 224 under its own rule (30 are R's infinite standard
error at an age that keeps one ratio not the first origin's, 16 its silent
switch from the log-linear rule to Mack's). On those 224 the link ratios chosen
equal R's weights, the factors agree to 1e-13, and the total standard error and
its process and parameter parts to 1e-12.

**Checked against chainladder-python 0.9.2**, which has two standard-error
defects when a drop or a bound is set, both named in
`docs/coming-from-chainladder.md`: it removes the first projected year's process
variance (raa with `drop_above=100`, which leaves out nothing, gives 13,037.79
for 26,880.74; `drop_high=1` 7,796.95 for 14,811.95), and it takes the standard
error of a factor kept from one link ratio from the first origin's cell. Where
neither fires (the three averages, `n_periods` -1, 3 and 5, an explicit drop,
an interior `drop_valuation`) ibnr equals chainladder on raa, genins, ukmotor,
abc and mw2014; with both patched, it equals chainladder under nine drop and
bound settings on the same five, and on the clrd paid triangles that are full
positive staircases: 240 of the 348 ibnr answers under `drop_high=1`, the rest
a sigma of exactly 0 that chainladder reads as 1e-320 when it fills another
(R and ibnr leave it out), and one triangle (clrd 377 under `drop_low=1`) where
chainladder projects a cell off the chain ladder and R agrees with ibnr. The
example workbook's triangle (New Jersey Manufacturers, workers' compensation
paid, 1988-1997) gives total standard errors of 10,329.10 with `drop_high=1`
(the app's `/reserve` gives 5,914.68 today), 8,342.65 with both drops, 9,345.41
with `drop_low=1`, 10,058.10 simple, 12,056.94 regression and 12,720.86 with
`history_periods=5`.

**No kernel number moved.** `fit_mack_grid` without `average` or `links`, under
both sigma rules and both zero rules, on raa, genins, ukmotor, abc, mw2014, the
same five with a zero cell, a 30 x 30 triangle and 53 clrd paid triangles (256
cases), has the bytes of every array, of `msep_runoff`, and of the `to_arrow()`
payload that the code before this change gave; digests are in
`tests/data/mack_default_pin.json`, written by `scripts/freeze_mack_pin.py`
against that code.

**What can change for a caller:**
- `methods.mack` now reads its factors through the shared link selection even
  with no option set (except under `zero_cells="observed"`), which adds in
  another order: its numbers move by at most 1e-14 relative. Checked on the 64
  triangles above under the default, `sigma_rule="mack"` and
  `zero_cells="observed"` (192 cases, the same refusals with the same reasons).
- `methods.mack(sigma_rule="mack")` fills a sigma after an age whose own sigma
  was filled from that filled value, as R does; 0.7.2 refused it. None of the
  pinned triangles reaches it.
- `ReserveResult.to_polars("link_ratios")` answers for Mack, and Mack's
  `development` has four more columns.
- `kernels.fit_mack`, `fit_mack_many` and `fit_mack_grid` accept
  `zero_cells=None` (the default now), which they refused.
- The seeded fuzz in `tests/test_refusal.py` runs 2,500 cases (was 2,000),
  since Mack draws from thirteen options now, like the chain ladder.

Speed: `methods.mack` on raa takes about 1.65 ms where it took 1.2 to 1.4 ms,
and about 13 to 19 ms on a 40 x 40 quarterly triangle, as before (best of 7
runs of 50, two alternating rounds on the dev box).

### Development options, Benktander, and Cape Cod's trend

The chain ladder, Bornhuetter-Ferguson and Cape Cod take the development
options chainladder-python users send, and there is a new method,
`methods.benktander`.

**New in `ibnr.methods`:**
- `average="regression"`: least squares through the origin,
  `sum(previous * following) / sum(previous ** 2)`. With `"volume"` and
  `"simple"` it is R ChainLadder's `delta` 0, 1 and 2, which the tests check
  on RAA and GenIns to a relative 1e-12 (the R script and its output are in
  the repository, since CI has no R).
- `drop_high` and `drop_low` are counts: `drop_high=2` leaves out the two
  highest link ratios at each age. `True` still means 1.
- `preserve`: the fewest link ratios the drops may leave at an age (1 by
  default). At an age where they would leave fewer, neither drop is made, as in
  chainladder-python.
- `drop_above` and `drop_below`: leave out link ratios above or below a number.
- `exclude_valuations`: leave out whole diagonals, written as dates
  (`"2020-12-31"`) or as the period they end (`2020`, `"2020Q4"`, `"2020-12"`).
- `trim_ties`: which of two equal link ratios the drops leave out. The default,
  `"volume"`, is chainladder-python's rule; `"origin"` is 0.7.2's.
- `methods.benktander(cells, premium=..., expected_loss_ratio=..., n_iters=...)`:
  Bornhuetter-Ferguson repeated from its own ultimate (Mack 2000); `n_iters=1`
  is `bornhuetter_ferguson` byte for byte.
- `methods.cape_cod(..., trend=0.05, n_iters=2)`: Gluck's trend, and
  Benktander's iteration. `origins` gains `trended_loss_ratio`
  (chainladder-python's `apriori_`) and `trend_factor`; `expected_loss_ratio`
  is `detrended_apriori_`.
- `development` gains `bounds_skipped`, and `link_ratios.reason` gains
  `valuation_exclusion`, `drop_above` and `drop_below`. The `ibnr.methods`
  docstring has one table of every `development` column and the methods that
  carry it.

The rules that leave link ratios out run in one order, each on what the ones
before left: zero cells, `history_periods`, `exclude`, `exclude_valuations`,
the bounds, then the drops. They live once, in a new numpy-only module,
`kernels/links.py` (`LinkRules`, `select_links`, `link_factors`, `ALPHA`),
which the Mack fit and the bootstrap are to reuse. `ConventionalCandidate`
gains `preserve`, `drop_above`, `drop_below`, `exclude_valuations`,
`trim_ties`, `n_iters` and `trend`, and a `link_rules` property.

**Checked against chainladder-python 0.9.2:** every combination of
`history_periods`, `drop_high` (0 to 3), `drop_low` (0 to 2), `preserve` (1 to
3) and the three averages on raa, and a sample of them on genins, ukmotor, abc,
mw2014 and a 40 x 40 quarterly triangle (prism); bounds on raa and genins;
every diagonal of raa, genins and ukmotor as an excluded valuation; Benktander
with 1, 2, 5 and 10 iterations and Cape Cod with three trends, three decays and
two iterations, on raa, genins, prism and 20 clrd triangles. A sweep outside
the test suite compared 35,406 fits on every clrd paid and incurred triangle
(the grid of `n_periods` -1, 3, 5, `drop_high` 0 to 2, `drop_low` 0, 1, 3,
`preserve` 1, 2 and the three averages): none differ. The example workbook's
totals come out to the cent from the Schedule P mart (New Jersey Manufacturers,
workers' compensation paid, 1988-1997): Benktander with two iterations
422,098.51, Cape Cod at a 5% trend 520,719.58, `history_periods=5` with one
high ratio dropped 356,958.59, regression 373,469.10.

**No kernel number moved.** The kernel defaults are 0.7.2's (`trim_ties="origin"`,
`preserve=1`, no bounds, `n_iters=1`, `trend=0`). Compared as raw bytes with the
code before this change: `fit_conventional_grid` under 13 sets of the options
that existed before, the methods under the same sets with
`trim_ties="origin"`, and `methods.mack`, on raa, genins, ukmotor, abc, mw2014
and 122 clrd paid and incurred triangles, 10,414 cases (3,112 of them
refusals, with the same messages), all identical. 1,764 kernel cases of the
same kind (14 sets, the three methods, the five public triangles, a 30 x 30 one
and 36 clrd triangles) are frozen as digests of their bytes in
`tests/data/conventional_selection_pin.json`, which the tests check.

**What can change for a caller:**
- In `ibnr.methods`, `drop_high` and `drop_low` now break ties as
  chainladder-python does, so on tied link ratios the factors move to
  chainladder's. ibnr answers 731 of the 775 clrd paid triangles; 50 of those
  are zero in every cell, and chainladder, which stores a zero as a missing
  cell, holds no cells for them. On the other 681 the factors change on 23 for
  `drop_high=True`, 124 for `drop_low=True` and 83 for both (on incurred, 70,
  51 and 40 of 726: 752 answered, 26 of them all zero). Pass
  `trim_ties="origin"` for 0.7.2's answer. The kernels keep 0.7.2's rule.
- `average="regression"` and a whole-number `drop_high`/`drop_low` above 1 were
  refused and are answered now. The message refusing a bad `drop_high` or
  `drop_low` names counts instead of True or False, and the message refusing a
  bad `average` lists `"regression"`.
- `ConventionalCandidate.drop_high` and `drop_low` are stored as whole
  numbers, so `True` reads back as `1` (it compares and hashes the same), and
  `dataclasses.asdict` of a candidate, as `scripts/benchmark_conventional.py`
  writes into its results, has the seven new fields. `ConventionalFit.origins`
  gains `expected_ultimate`, `trend_factor` and `trended_loss_ratio`, and
  `factor_summary` gains `bounds_skipped`.
- `exclusions_exhausted` now also covers bounds that would leave fewer than
  `preserve` link ratios. No reason code was added.

**Where ibnr differs from chainladder-python on purpose**, each in
`docs/coming-from-chainladder.md`: `exclude_valuations` names the later end of
a link ratio (chainladder's `drop_valuation="1994"` is
`exclude_valuations=[1995]`, and the latest valuation can be excluded); the
rules act in order rather than independently, so two exclusion rules together
cannot empty an age `preserve` protects; a ratio equal to a bound is kept;
`n_iters=0`, `n_iters` above 10,000, per-age lists and `trend <= -1` are
refused; and Cape Cod keeps fully developed origins that are not on the latest
diagonal.

Speed: the conventional estimator (what `ibnr.methods` runs after reading the
cells) takes about 0.3 ms on raa where 0.7.2 took 0.15 ms, and 1.2 to 1.7 ms on
a 40 x 40 quarterly triangle where it took 1.1 to 1.3 ms (best of 7 runs of
50, three alternating rounds on the dev box, with and without drops).

### `ibnr.methods` loads no ibis, pandas or scipy

`from ibnr import methods` used to load ibis, pandas and scipy, although the four
methods use none of them. A service that starts a new process per request paid
for all three on every cold start. Now importing the module loads numpy and
pyarrow only, and running `chain_ladder`, `bornhuetter_ferguson`, `cape_cod` or
`mack` loads nothing heavy: a first call imports `numpy.ma` (3 modules), and
later calls import nothing. That holds for cells given as a polars DataFrame,
an Arrow table or a dict of Python lists or numpy arrays (what a service reading
JSON has). Two inputs still make pyarrow load pandas: a pandas DataFrame, and a
dict with a list that is not all strings, all bools, all dates or all numbers
(one holding a null or a datetime, for example).

Times from one run on the dev box on 2026-09-25: for each row, 11 fresh
interpreters per arm, the two arms taking turns, every row in the same run. The
box was busy (0.7.2's `from ibnr import methods` measured 2.35 to 2.57 s there
the day before), so read the ratios. The rows with a call include reading the
cells, and are not slower than the bare import within the spread: a first call
costs less than the difference between two runs of the same row.

| What was timed | 0.7.2, median (range) | now, median (range) | modules loaded |
|---|---|---|---|
| `from ibnr import methods` | 4.10 s (3.38-4.44) | 1.22 s (0.90-1.41) | 1,050 -> 286 |
| that, then a first `chain_ladder` on raa read from an Arrow file | 3.90 s (3.73-4.44) | 1.13 s (1.00-1.38) | 1,053 -> 289 |
| that, then a first `mack` on raa read from an Arrow file | 4.04 s (3.10-5.04) | 1.09 s (0.92-1.33) | 1,053 -> 289 |
| that, then a first `chain_ladder` on raa as a dict of lists | 4.07 s (3.39-4.79) | 1.06 s (0.95-1.35) | 1,053 -> 289 |
| `import ibnr` | 1.26 s (1.06-1.63) | 0.54 s (0.41-0.65) | 360 -> 154 |
| `from ibnr import gallery` | 4.46 s (3.92-4.77) | 4.00 s (3.70-5.28) | 1,117 -> 1,113 |

What is left is numpy (about 0.45 s on this box), pyarrow (about 0.25 s) and the
lookup of ibnr's installed version (0.1 to 0.2 s); ibnr's own modules take about
0.03 s. The gallery needs pandas and scipy, so its import time does not change.

**No number moved.** Every result was compared with 0.7.2's code, as exact float
bits and exact refusal messages: the four methods with eight sets of options,
`fit_conventional_grid` and `fit_mack_grid` (with `msep_runoff` and `summary`)
under both zero rules, the grid itself, and the Triangle path (`fit_mack`,
`fit_mack_many`, `fit_conventional`), on raa, genins and 130 clrd paid
triangles (every sixth company and line, in name order): 2,511 cases, 1,509 answers and 1,002 refusals, all identical.
A dict of lists or numpy arrays is read into the same Arrow table `pa.table`
made of it, type and value.

**What changed:**
- `ibnr/__init__.py` imports `Triangle` and `TriangleMeta` the first time they
  are read, so `import ibnr` no longer loads ibis. `ibnr.triangle` still
  answers after `import ibnr`, and `ibnr.gallery` is still an `AttributeError`
  until something imports it.
- `ibnr/kernels/__init__.py` imports each name the first time it is read (PEP
  562). `__all__` is unchanged, `kernels.codec` and the other submodules still
  answer as attributes, and a `TYPE_CHECKING` block repeats every import for
  type checkers and the docs build.
- The numpy-only grid helpers (`grid_from_columns`, `check_grid`,
  `require_run_off`, `dev_step_index`, `as_date`, `month_end`, `ZERO_CELLS`)
  moved from `kernels/contract.py` to a new `kernels/grid.py`, beside a new
  `TRIANGLE_MEASURES`. `kernels.contract` re-exports the first seven, so imports from there
  keep working.
- `kernels/conventional.py` and `kernels/mack.py` import pandas, the Triangle
  layer and `PredictiveDistribution` inside the functions that use them.
  `fit_conventional_grid` still returns pandas tables, and so imports pandas
  when it runs; `ibnr.methods` reads the same fit before those tables are built.
- pyarrow imports pandas, when it is installed, the first time `pa.array`,
  `pa.scalar` or `Array.to_numpy` runs, and `pa.table` does too for nearly
  every input (it asks whether the input is a pandas DataFrame, or builds
  arrays with `pa.array`). So `ibnr.methods` builds its result
  arrays from numpy memory (`ibnr/_arrow.py`), reads its input columns the same
  way, and reads a polars DataFrame (or any other object offering the Arrow
  stream interface, except a pandas DataFrame) with
  `pa.RecordBatchReader.from_stream` instead of `pa.table`. A dict of Python
  lists or numpy arrays has its columns built the same way, and any other dict
  still goes to `pa.table`.

**What can break:** code that counted on `import ibnr` or `import ibnr.kernels`
having loaded something as a side effect. `vars(ibnr.kernels)` lists a name only
once it has been read, and `sys.modules` no longer holds ibis after a bare
`import ibnr`. Reading the names works as before.

### Refusals carry a reason code: `ibnr.errors.Refusal`

Every input `ibnr.methods` will not answer is now refused with
`ibnr.errors.Refusal`, a `ValueError`, so `except ValueError` still catches it.
Before, the refusals were plain `ValueError`s whose text was partly in the
kernels' own terms (`dev step 1`, `sigma_j^2`, `datetime.date(2003, 1, 1)`), so a
service could tell one from another only by matching text, and could not tell
bad input from a defect in ibnr.

A refusal carries:
- `reason`, one code from a closed list of 27 (`ibnr.errors.REASONS`), such as
  `negative_cumulative`, `not_run_off` or `no_link_ratio`;
- `kind`, `"input"` (13 codes: the request must change) or `"model"` (14 codes:
  another option or method can answer);
- `method`, `option` and `column`: the function, the argument and the column at
  fault, in the function's own terms;
- `cells` (origin, first day of the period, `dev_lag`, value), `links`
  (`(from_dev_lag, to_dev_lag)` in months) and `rows` (0-based rows of a
  table): at most 100 cells and 100 rows, with `count` holding the total;
- each origin as the caller wrote it, value and type, the same value the
  result's `origin` column would show; the kernels name an origin by its
  period's first day and `ibnr.methods` puts the caller's label back.

`refusal.to_dict()` is JSON with no NaN or infinity, and a refusal pickles with
every field, so it crosses a process pool. `ibnr.methods.__all__` gains
`Refusal`. The contract, in the `ibnr.methods` docstring: a `Refusal` is ibnr
declining the input; any other exception from these functions, a plain
`ValueError` included, is a defect in ibnr. A seeded fuzz of 2,000 edited
triangles and every clrd company and line (paid and incurred, all four methods)
meet nothing but `Refusal`.

Codes are never renamed or removed; a new one may be added in a patch release
and will be named here. Handle a code you do not know by its `kind`.

The one-year claims development result in `ibnr.kernels` (`one_year_cdr`,
`simulate_one_year_cdr`, `rereserve`, the diagonal generators, the ODP
bootstrap and `cdr_risk_measures`) raises `Refusal` too, and so do the kernel
checks of data and options in `kernels/grid.py`, `kernels/conventional.py` and
`kernels/mack.py`. A grid dict with the wrong keys, shapes or types, which only
a grid built by hand can have, is still refused with a plain `ValueError`.

**No number moved.** Every result was compared with the code before this change
as raw bytes: the four methods (Arrow tables) and the kernels
(`fit_conventional_grid` for chain ladder, Bornhuetter-Ferguson and Cape Cod
under seven sets of options, `fit_mack_grid` and `msep_runoff` under both sigma
rules and both zero rules, `one_year_cdr`, and seeded `simulate_one_year_cdr`
draws from the Mack and ODP bootstrap generators) on raa, genins, ukmotor, abc,
mw2014 and 62 clrd triangles, and `fit_mack_many` over 120 clrd cohorts: 2,153
cases, 230 of them refusals, all identical (the refusals in the same places).

**Refused now, answered before** (each gave a wrong or meaningless answer):
- an infinite amount in the cells. The chain ladder refused it without naming
  the cell, and `mack` either refused it for a wrong reason (zero cells, on a
  triangle with none) or answered NaN. Now `not_finite`, naming the cells;
- `expected_loss_ratio=True` and `decay=True`, which were read as 1. Now
  `invalid_option`, in `ibnr.methods` and in `kernels.ConventionalCandidate`;
- an answer with a number that is not finite: a `methods.mack` standard error
  (amounts near the largest double, whose squares overflow), which came back as
  NaN, and a total or a link ratio of the other three methods that overflows
  although each ultimate is finite, which came back as infinity. Now
  `result_not_finite`, naming the origins where it can. A result's numbers are
  all finite, and a missing one is a null;
- `methods.mack(zero_cells="observed")` where the last age's link ratios give a
  factor of 0 (every origin there closing at zero; raa with 1981 at 120 months
  set to 0, say), or where a factor is below the smallest double. Every ultimate
  came back 0 and `pct_reported` infinite. Now `no_link_ratio`, as
  `chain_ladder` already refused a factor of 0;
- a factor or a link ratio past the largest double. `unsupported_factor="unity"`
  put 1.0 in place of an infinite factor, which gave a wrong answer with every
  number finite (a total ultimate of 1.0 against a latest of 1e308), and without it
  the refusal was `no_link_ratio` and suggested that option. In `mack` the
  sigma came out NaN and was filled in as if the age had too few link ratios,
  so the refusal was `variance_not_estimable`, blaming zero cells the triangle
  did not have. Now `result_not_finite` naming the age, in
  `kernels.fit_conventional_grid` and `kernels.fit_mack_grid` too;
- `methods.mack` on amounts near the smallest double (raa times 1e-310), whose
  squares are 0: the standard errors came back 0. Now `result_not_finite`;
- a float `dev_lag` of 2**63 months or more, which numpy turned into
  -9223372036854775808 with a warning and which was then refused as that
  negative age. Now `invalid_age`, quoting the value sent.

**Other changes a caller can see:**
- Mack's rule for a sigma (`sigma_rule="mack"`) raised `OverflowError` once a
  sigma passed about 1e154 (raa times 1e154, say), in `methods.mack` and in
  `kernels.fit_mack_grid`. It now computes `last * (last / prev)` there; below
  that nothing changes, bit for bit. Such a triangle's standard errors are then
  refused as `result_not_finite`;
- `ibnr.methods` gives no `RuntimeWarning` on amounts near the largest or the
  smallest double; the answer is refused by name instead. The seeded fuzz now
  scales whole triangles to 1e-300 and 1e300 and sends premiums of 1e-320 and
  1e308, with a warning counted as a failure;
- a `fractions.Fraction` `expected_loss_ratio` or `decay` raised a `TypeError`
  from inside numpy; it is a number and is now used as its float;
- messages show a numpy scalar as the number it holds (`got -0.1`, not
  `got np.float64(-0.1)`), and a column name or label spelling a placeholder
  such as `{given}` is shown as written;
- a text `expected_loss_ratio` or `decay` raised a `TypeError` from inside
  numpy; now `invalid_option`;
- `simulate_one_year_cdr(generator=...)` with a name that is no method raised a
  `KeyError`, and with a value of the wrong type a `TypeError`; both are now
  `invalid_option`. `get_cdr_method` still raises `KeyError`, as a lookup does;
- `fit_mack_many(on_error="skip")` records only a `Refusal` and raises anything
  else, since that is a defect rather than a cohort the data rules out. The
  returned `MackFitPanel` gains `reasons`, each skipped cohort's code beside
  its message in `errors`, and the codec carries it. An unknown `sigma_rule` is
  refused before any cohort instead of being recorded against every one. Under
  `on_error="raise"` a cohort's refusal stays a `Refusal`, its message led by the
  cohort;
- many messages are reworded to name ages in months and origins as written:
  `from 36 to 48 months` for `dev step 3` or `dev lag 36`, `(2002, 24 months)`
  for `(2002, 24)`, and a text label without quotes except where two spellings
  of one period are compared. Code that matched message text should match the
  `reason` instead;
- `kernels.grid.require_run_off` takes `step=` and `cum=` (both optional) so a
  refusal names cells in months with their amounts, and
  `kernels.odp_bootstrap.fit_odp_bootstrap` takes `origins=` and `dev_grain_months=` for the same reason. Where cells are
  not a run-off triangle, the message names the cells against the diagonal that
  leaves the fewest cells wrong; which triangles are refused is unchanged.

### `methods.tweedie_glm`: a Tweedie GLM fitted to the increments

A new method, `methods.tweedie_glm(cells, power=1.0, link="log", origin="factor",
calendar="none", projection="pattern", dev_grain_months=12, max_iter=100)`, for
a service moving off chainladder-python's `TweedieGLM`. It takes the same
cumulative cells as the other methods, fits a GLM to their increments by
iteratively reweighted least squares (Fisher scoring, as R's `glm` does, in
numpy; the kernel is `kernels.fit_tweedie_grid` with `kernels.TweedieSpec`) and
returns a `ReserveResult`.

- `power` 0 (normal), 1 (over-dispersed Poisson), between 1 and 2 (compound
  Poisson-gamma), 2 (gamma) or above; a power between 0 and 1 has no Tweedie
  distribution and is refused. `link` is `"log"` or `"identity"`. `origin`
  is `"factor"` (a level per origin) or `"none"`. `calendar="trend"` adds a
  straight-line calendar trend, and only beside `origin="none"`: with origin
  and development factors it cannot be estimated (R returns `NA` for it).
- `projection="pattern"` (the default) is the latest cumulative times the
  fitted development from the latest age, as chainladder reports it;
  `"increments"` is the latest cumulative plus the fitted future increments, as
  R's `glmReserve` reports it. `origins.model_ibnr` always carries the second.
- `tail` is accepted only as `None` for now (`not_supported`).
- No penalty and no `tol`: the fit stops when no fitted increment moves by
  more than 1e-10 of the largest increment, so the answer scales exactly with
  the units. A fit that has not settled within `max_iter` is refused
  (`did_not_converge`), never returned.
- A fitted mean below 1e-8 of the largest increment, on a cell whose increment
  is zero or less, means the fit exists only in the limit. It is refused, as
  `degenerate_fit` under the log link and as `negative_fitted_mean` under the
  identity link at a power above 0, so whether a triangle is answered does not
  depend on its units either. A small mean fitted to a small positive
  increment (GenIns with one increment of 0.01) is answered.
- At power 0 the log link keeps every fitted increment above zero, so an age or
  an origin whose increments sum to zero or less can leave it with no finite
  fit. Such a refusal names the age or origin and points to `link="identity"`,
  which answers it (417 of the 775 clrd incurred-less-bulk triangles at power 0,
  against 128 under the log link).
- Under the log link an origin or an age whose increments are all zero is
  fitted at exactly zero (its coefficient null, `fitted_zero` true), which is
  the limit of the fit and the chain ladder's answer.

**Checked:** power 1 with the defaults equals `chain_ladder(cells,
zero_cells="observed")` to 1e-10 by both projections on GenIns, UKMotor, ABC
and MW2014, and chainladder-python's chain ladder too. Powers 0, 1, 1.5 and 2
match R's `glm` with `statmod::tweedie` on the same four triangles (reserves to
at most 9.3e-8 of the largest origin's, plus the coefficients, standard errors,
deviance and dispersion). The identity link matches R at power 0 on GenIns and
at powers 1, 1.5 and 2 on GenIns, UKMotor and ABC (reserves and fitted
increments to 2e-7 of the largest, standard errors to 1e-6), and on a small
triangle whose fit has to halve a step.
MW2014 under the identity link is refused (`negative_fitted_mean`) at powers 1,
1.5 and 2: R finds no valid coefficients at 1 and 1.5, and at 2 returns
negative future increments (a total reserve of -29,932). The numbers are frozen in
`tests/data/tweedie_glm_r.json` by `scripts/r/tweedie_glm_reference.R`, because
CI has no R. On the 775 clrd paid triangles, built from chainladder's raw
`clrd.csv` so zeros stay zeros, the power-1 GLM equals `chain_ladder(cells,
unsupported_factor="unity")` under both zero rules to 1e-8 on all 273 with
increments of zero or more and no cumulative going from zero to a positive
amount; of the other 82 with increments of zero or more, 13 have no finite fit
and are refused (`degenerate_fit`) and 69 are answered.

**What moves for a chainladder-python user.** chainladder 0.9.2's `TweedieGLM`
never passes `alpha` to scikit-learn, so every fit carries scikit-learn's ridge
penalty of 1.0 whatever `alpha` says, and its answer depends on the units.
Total IBNR on genins, chainladder's `TweedieGLM` (with any `alpha`) against
`tweedie_glm` (pattern route): power 1 18,683,659.86 against 18,680,855.61 (the
chain ladder's); power 1.5 19,696,737.66 against 18,472,367.35; power 2
24,252,472.98 against 18,257,520.17; power 0 19,115,349.05 against
19,115,055.22. On ukmotor at power 1, chainladder's total over the chain
ladder's is 1.0422 as stored and 2.9197 with the amounts divided by 1,000;
ibnr's is 1 at every scale. `docs/coming-from-chainladder.md` lists the other
differences.

**`ReserveResult` gains two tables**, `cells` and `coefficients`, both `None`
except on a `tweedie_glm` result, and `methods.TABLES` lists them; `to_polars`
refuses a table a result does not carry with a message naming that table.
`methods.__all__` gains `tweedie_glm`, and `ibnr.kernels` gains
`TweedieSpec`, `TweedieFit` and `fit_tweedie_grid`. No refusal code was added:
the GLM's refusals use `negative_increment`, `zero_increment`,
`not_identified`, `did_not_converge`, `degenerate_fit`,
`negative_fitted_mean`, `negative_projection`, `not_supported` and
`invalid_option`. Nothing else moved: the chain ladder's,
Bornhuetter-Ferguson's, Cape Cod's and Mack's results (measured before
Benktander and the development options landed) were compared with the code
before this change as raw Arrow bytes and exact refusal messages, eight sets of
options on raa, GenIns, UKMotor, ABC, MW2014 and 130 clrd paid triangles, 1,218
cases, all identical.

### `methods.one_year_cdr`: the one-year claims development result

A new function, `methods.one_year_cdr(cells, dev_grain_months=12,
sigma_rule="log_linear", zero_cells="observed", n_draws=20_000, seed=None,
process="gamma", parameter_risk=True, quantiles=(0.5, 0.75, 0.9, 0.95, 0.99,
0.995, 0.999))`, so a service can answer a one-year question without
chainladder-python, the Triangle layer or pandas. It fits Mack's chain ladder
(`kernels.fit_mack_grid`) and returns a new `OneYearCDRResult` with four
pyarrow Tables and `to_polars()`:

- `origins`: the latest amount, today's chain-ladder ultimate and IBNR, the
  simulated mean and standard deviation of `ultimate_change`, the
  Merz-Wuthrich standard error `cdr_se` and Mack's run-off standard error
  `runoff_se`;
- `totals`: the same for the sum over the origins, and `n_draws`;
- `quantiles`: the quantile and the mean beyond it (`tvar`) at each level, for
  the total (a null origin) and for each origin;
- `draws`: every draw of every origin.

`ultimate_change` is next year's ultimate minus today's, so positive is a
strengthening. `kernels.simulate_one_year_cdr` has the opposite sign.

**The draws are the Reserving app's.** The seed is turned into a random stream
the same way the gallery's `mack` entry does for a Triangle with the one
segment `Total` and the field `values`, which is what the app's `/cdr` route
fits after `Triangle.from_chainladder`. For the same seed, draw count and
options, the draws are equal to the last bit, and so is every number `/cdr`
returns (latest, ultimates, IBNR, the per-origin and total means and standard
deviations, the total's percentiles and tail means, and the four analytic
standard errors). One difference is on purpose: an origin at its last age
never moves, and the app's negated draws hold -0.0 for it, so `/cdr` shows its
mean as -0.0; this function has 0.0 there, in the draws and in every summary.
Checked on raa and GenIns under six sets of options, from 2 to 20,000 draws,
both through chainladder exactly as the app parses its request
(`tieout`) and through an ibnr Triangle (on the core leg). For the app: pass
`sigma_rule` through (its default is `"mack"`, this function's is
`"log_linear"`), and send `quantiles=[p / 100 for p in percentiles]` rather than
relying on the default, because 99.9 / 100 and 0.999 are not the same double.
The Merz-Wuthrich figures match R's `CDR(MackChainLadder(MW2014,
est.sigma="Mack"))` to 6 decimals through this function too.

Importing `ibnr.methods` still loads no ibis, pandas or scipy, and neither does
this function when it runs: its first call imports `numpy.random` and
`numpy.ma` (16 modules), and 20,000 draws of raa took about 0.1 s on the dev
box (0.05 s the second time). To make that hold, `kernels/cdr.py` now imports
pandas, `PredictiveDistribution` and the ODP bootstrap (which needs scipy)
inside the functions that use them, and `kernels/rng.py` imports
`HoldoutCells` (which needs ibis and pandas) inside `heldout_stream`. The
draws come from a new private `kernels.cdr._one_year_cdr_draws`, which
`simulate_one_year_cdr` now calls before wrapping its answer in a
`PredictiveDistribution`.

Refusals, all in the caller's terms: a `dev_grain_months` other than 12
(`not_supported`: one step is a year only on an annual triangle); a zero
cumulative under `zero_cells="missing"` (`not_supported`, checked before the
fit, because the one-year formulas have not been checked with the link ratios
that rule leaves out; on a triangle with no zero the two rules are the same
fit); a still-developing origin whose latest cumulative is zero
(`variance_not_estimable`, under either rule); everything `methods.mack`
refuses; and `n_draws`, `seed`, `process`, `parameter_risk` and `quantiles`
outside their values (`invalid_option`; a bare number for `quantiles`, and
percentages such as 99.5, are refused rather than dropped or read). `n_draws`
times the number of origins above 100,000,000 is refused too
(`invalid_option`): each number takes about 70 bytes at the peak, so that is
about 7 GB, and a larger count would otherwise reach numpy and fail there with
a `MemoryError` or a `ValueError`. Amounts near the smallest double are
refused with the words `methods.mack` uses, "too small"; the one-year formula
turns them into NaN, which the check for numbers that are not finite would
have called too large. No refusal code was added. The kernels refuse the grain
and the zero cases too, in words about `MackFit` and Triangle methods; the
front door checks first so the message names its own options.

**One kernel number moves, in its last one or two binary digits.**
`kernels.one_year_cdr` computed each origin's own-process term of the total
as `ult**2 * ratio / C`, and `ult**2 * ratio` is on the scale of an amount
cubed. Below about 1e-103 it read 0, so the total `cdr_se` came out too low
with nothing refused (16% low on a 3 x 3 triangle scaled by 1e-112), and above
about 1e103 it was infinite, so the total was refused though every number in
it fits. It is now `ult**2 * (ratio / C)`, and the standard errors now scale
with the amounts from 1e-140 to 1e120 (tested on raa and a 3 x 3 triangle).
The per-origin and run-off figures do not move; the total's msep moved in 51
of 374 fits (raa, GenIns, UKMotor, ABC, MW2014 and every sixth clrd paid and
incurred triangle, under both sigma rules), by at most 3.3e-16 of its value.
The MW2014 tie-out to R is unchanged.

`methods.__all__` gains `one_year_cdr` and `OneYearCDRResult`. **Nothing
else moved** apart from that total: with the code before this change,
one-year CDR draws (`simulate_one_year_cdr` with three process laws, with and
without parameter risk, for an integer seed and a derived stream, and the ODP
bootstrap with two process laws), the Merz-Wuthrich results and their
`summary()`, `cdr_risk_measures`, `cdr_methods()` and `methods.mack` under
both sigma rules and both zero rules (raw Arrow bytes, or the exact refusal)
were hashed on raa, GenIns and clrd paid triangles (every sixth company and
line: 130, of which 90 come out of chainladder's frame as a run-off triangle;
the other 40 are empty or have holes where chainladder dropped a zero), and
the gallery `mack` entry's `cdr_distribution` and `predict` draws on raa and
GenIns: 9,381 hashed answers and 330 refusals, all identical.

## 0.7.2 - 2026-09-24

Three pull requests (#149, #152, #153) for moving a reserving service off
chainladder-python:
- a fast array path for the conventional point fits;
- a front door named after the traditional methods (`ibnr.methods`, with Arrow
  tables in and out);
- a setting that reads a zero cumulative as missing, as chainladder does.

This is a patch release although it adds a public module. From this release the
minor number is kept for when the roadmap's goals are done, and additions and
fixes ship as patch releases.

**Changes that can break existing code.** All of them refuse input that used to
be accepted:
- `fit_mack`, `fit_mack_grid` and `fit_mack_many` now check their grid. They
  refuse an incremental grid, origin periods that are not the first day of their
  period, and origins whose spacing does not match the development step.
- `cohort_grid_frame` reads ISO date strings as dates; it used to keep them as
  text and sort them as text. It refuses an unknown measure, a row with no
  origin period, and origin periods that are not dates.

No number changes for input that is still accepted: with default settings the
kernels' results were compared byte for byte against 0.7.1's code.

**`zero_cells`: a cumulative of zero can be read as missing, as chainladder-python
reads it.** chainladder-python stores every zero cell as missing, so a link ratio
is used only when neither of its two cells is zero. ibnr kept zeros as data, and
not even the same way in its two kernels: the conventional kernel left out the
link ratio out of a zero (undefined) but kept the one into it (a ratio of 0),
while Mack's factor kept both in its volume sums, as R's `MackChainLadder` does.

- New option `zero_cells`, `"observed"` or `"missing"`, on
  `kernels.ConventionalCandidate` (so on `fit_conventional`,
  `fit_conventional_grid`, `conventional_grid(zero_cells=...)`,
  `replay_conventional` and `select_conventional`) and on `kernels.fit_mack`,
  `fit_mack_grid` and `fit_mack_many`. **The kernels default to `"observed"`
  and their answers do not change**: compared byte for byte against the code
  before the option, on raa, genins and eight clrd paid triangles (factors,
  sigmas, ultimates, run-off and one-year standard errors, simulations, replay
  and selection, and every refusal message), nothing moved. Under
  `"missing"` a link ratio with a zero at either end is left out, of the
  factor, its volume, `n_obs` and Mack's sigma alike; the conventional kernel
  reports it in `factor_selection` with the new reason `"zero_cell"`
  (`"undefined_ratio"` stays the `"observed"` reason). `MackFit` records the
  setting as `zero_cells` and counts the link ratios it left out as
  `zero_links`, and the codec carries the setting.
- **`ibnr.methods` defaults to `"missing"`**: `chain_ladder`,
  `bornhuetter_ferguson`, `cape_cod` and `mack` take `zero_cells` and follow
  chainladder-python unless told `zero_cells="observed"`. On a triangle with no
  zero cumulative the two settings give the same answer, so raa, genins and
  every earlier tie-out are unchanged.
- A zero on a still-developing origin's latest diagonal, under `"missing"`, is
  kept as that origin's latest amount: its chain-ladder ultimate is 0 and its
  Mack standard error is 0, the limit of Mack's formula as the amount goes to
  zero, set directly rather than divided through. chainladder-python leaves
  that origin's ultimate and standard error missing and gives the same total:
  on raa with 1990's 12-month cell set to 0, the total Mack standard error is
  10,008.21 in both, and every other origin's ultimate and standard error
  agree. Under `"observed"` the standard errors of such a triangle are still
  refused, as before.
- Tied out to chainladder-python 0.9.2 under `"missing"`, from a frame so that
  chainladder itself turns the zeros into missing cells, on raa with a zero at
  an origin's first age, an interior write-down to zero, and a zero on the
  latest diagonal of the newest and of an older origin: chain-ladder ultimates
  and factors, Bornhuetter-Ferguson, Cape Cod (`decay=1`, `trend=0`) and Mack's
  per-origin and total standard errors with the parameter and process parts,
  sigma and std_err (log-linear sigma), all to a relative 1e-9. On the clrd
  paid triangles built from chainladder's `clrd.csv` (which still has its
  zeros), 731 are answered by the conventional kernel with a factor of 1.0
  where no link ratio is left, as chainladder does; under `"observed"` 720
  agree with chainladder and 11 do not, and under `"missing"` all 731 agree.
  Of those 11, Mack under `"missing"` gives chainladder's total standard error
  on 7, and on the other 4, where chainladder's is missing, refuses by name a
  development step with no link ratio left.
- Two things `"missing"` has to do the way chainladder-python does, or it
  would not match it with zeros present. The history window
  (`history_periods`) counts the most recent origins, so a link ratio left out
  for a zero keeps its place and the window holds fewer ratios; it does not
  pull an older origin in (under `"observed"` an undefined ratio still gives up
  its place, as before). And a development step before the last can be left
  with a single link ratio, where sigma has nothing to be estimated from: the
  log-linear rule then fills it from one regression over every step with a
  positive estimate, before and after it, and Mack's rule from the two steps
  just before it, refused by name when those have no estimate of their own.
  Both are tied out to chainladder-python (a single link ratio at the first
  step and at a middle step; windows of 2 to 5 with zeros inside and at the
  edge).
- Known differences from chainladder-python that remain under `"missing"`: an
  origin whose latest amount is zero gets ultimate and standard error 0 where
  chainladder leaves them missing; chainladder also stores a sigma of exactly 0
  (every link ratio at an age equal) as missing and fills it, which ibnr does
  not, so Mack's standard errors can differ there; and ibnr refuses a Mack step
  with no link ratio left even when only origins at a latest amount of zero
  cross it, where chainladder leaves that factor missing and still answers. On
  the clrd paid triangles that contain a zero and that both libraries answer
  (85), Mack's total standard error agrees on 79; each of the other 6 has an
  estimated sigma of exactly 0.
- The codec carries `zero_cells` in the `MackFit` header without a version
  change, so a payload written before it decodes as `"observed"`, which is what
  it was. The other direction is not guarded: a reader from 0.7.1 or earlier
  drops the setting and would give a one-year result for a `"missing"` fit that
  this version refuses.
- `cdr_methods()` lists the `zero_cells` precondition in every row's
  `requires`.
- Refused by name: an unknown `zero_cells` value (by `fit_mack_many` before
  any cohort, so `on_error="skip"` cannot record it against every cohort), a
  Mack development step where `"missing"` leaves no link ratio, and the
  one-year claims development result (`one_year_cdr`,
  `simulate_one_year_cdr`, `rereserve`, and the `mack`, `odp_bootstrap` and
  `GalleryDiagonal` generators' checks) on a `"missing"` fit that left out a
  link ratio or kept a zero latest amount, because the Merz-Wuthrich formulas
  have not been checked under that rule. A `"missing"` fit with no zero in it
  is the same fit as an `"observed"` one and is accepted.

**`ibnr.methods`, the front door for the traditional methods.** One function per
method, named after it: `methods.chain_ladder`, `methods.bornhuetter_ferguson`,
`methods.cape_cod` (Gluck's generalized Cape Cod; `decay=1` is the classic one)
and `methods.mack`. Import it with `from ibnr import methods`; a bare
`import ibnr` does not load it.

- Each takes one triangle's cells as any Arrow-readable table (a polars
  DataFrame, a pyarrow Table or RecordBatch, anything with
  `__arrow_c_stream__`) with columns `origin_period`, `dev_lag` (months) and
  `value` (cumulative loss). ibnr reads it with pyarrow and never imports
  polars to do so.
- `origin_period` is written the way the caller writes it: an integer year
  (`2020`), a text label for a year, quarter or month (`"2020"`, `"2020Q3"`,
  `"2020-03"`), or a date, timestamp (with or without a time zone; the date
  read is the one in that zone, and a time of day is dropped) or ISO date
  string, any of them dictionary-encoded (a polars Categorical or Enum). A date
  is read with `dev_grain_months`, which is also the origin period's length:
  the first day of a month starts a period and the last day of a month ends
  one, so `2020-12-31` is the accident year 2020 and, annually, `2021-06-30` is
  the year from July 2020 to June 2021. Premium's origins and the origins in
  `exclude` take the same forms and need not match the cells' spelling. The
  results echo the caller's label in a new first column `origin` of `origins`
  and `link_ratios` (int64, string, date32 or the input's timestamp type),
  beside `origin_period`, which stays the first day of the period.
- Each returns a `methods.ReserveResult` holding pyarrow Tables with fixed
  column types: `origins` (latest, ultimate, ibnr per origin; Mack adds
  `mack_se`, `parameter_se`, `process_se`), `development` (factor, cdf,
  pct_reported per age; null factor at the last age), `link_ratios` (every
  observed link ratio and why any was left out; none for Mack) and `totals`
  (for Mack, the total standard error, which is not the sum of the origins').
  `ReserveResult.to_polars(name)` gives any table as a polars DataFrame and
  names the `polars` extra when polars is missing. There are no pandas objects
  on the result.
- Development options use ibnr's names: `average`, `history_periods`,
  `drop_high`, `drop_low`, `exclude`, `unsupported_factor` and
  `exhausted_exclusions`. `exhausted_exclusions` defaults to `"keep"` here (the
  kernel candidate keeps `"raise"`), because on a complete triangle the last
  age has one link ratio and `drop_high=True` would otherwise always be
  refused; the skip is recorded in `development.extreme_trimming_skipped`.
  `methods.mack` takes no development options (the Mack kernel has none yet)
  and defaults to `sigma_rule="log_linear"`, chainladder-python's default;
  `kernels.fit_mack` keeps `"mack"`, so published numbers do not move.
- Refused by name: a missing column, a `dev_lag` that is not whole months, a
  null or NaN `value` (an unobserved cell is left out, not null), a null
  `origin_period` or one that names no period (`"2020Q5"`, `"FY20"`, a date
  that is neither a month's first day nor its last, an integer year without
  its century such as `97`), a year, quarter or month label whose length is
  not `dev_grain_months` (checked on the label, so even a one-origin triangle
  cannot be misread; a premium or `exclude` origin of the wrong length is told
  to follow the cells), one period written two ways in one column (the label
  is echoed back, so it must be one), two rows for one cell (the message says
  the methods fit one cohort at a time; ibnr refuses rather than sums them),
  premium without `origin_period` and `premium` columns, with two rows for one
  origin, missing an origin or carrying an extra one, an `exclude` pair that
  names no link ratio of the triangle, and one link ratio excluded twice (two
  spellings of one origin included). Also refused: `dev_lag` values that are
  not multiples of `dev_grain_months` (quarterly ages with the default annual
  grain are told to pass `dev_grain_months=3`), a negative cumulative (naming
  the cells), a missing origin period or origin periods longer than a
  development step (naming the periods), and a `methods.mack` triangle with at
  most one link ratio at every age, where every sigma would be 0 and the
  standard errors would read as no uncertainty. Every message names an origin
  as the caller wrote it; a missing origin period, which has no label, is
  named by its first day.
- Tied out from polars frames to chainladder-python 0.9.2 on raa and genins,
  to a relative 1e-9: chain-ladder ultimates, factors and cdfs; Mack's
  per-origin and total standard errors with the parameter and process parts,
  sigma and std_err; Bornhuetter-Ferguson with a premium that rises across the
  origins; Cape Cod with `decay=1` and `trend=0`; and on raa, the chain-ladder
  factors under `history_periods`, `average="simple"`, `drop_high`, both drops
  and `exclude`. The chain ladder and Mack's total standard error also tie out
  from integer accident years.
- `docs/coming-from-chainladder.md`: a lookup table from chainladder-python's
  classes and attributes to these functions, what is not there yet, and the
  behaviours that differ on purpose.
- `kernels.contract.grid_from_columns`: the grid builder, now taking three
  plain columns with no pandas inside it. `cohort_grid_frame` is a thin wrapper
  over it with the same checks and messages. As a side effect `fit_mack_many`
  on clrd's paid losses (725 cohorts, 496 fitted and 229 refused, the same
  split as before) went from about 0.34 s to about 0.23 s (three interleaved
  pairs of processes, median of seven warm runs each, one loaded Windows
  laptop), because origin periods are now factorized as numpy dates rather
  than as Python date objects. `kernels.contract.as_date` no longer uses
  pandas.

A public array entry point for the conventional point estimators, for a
service that fits one small triangle per request. Mostly additive; the few new
refusals on existing functions are listed at the end.

- `kernels.fit_conventional_grid(grid, candidate, *, premium=None)`: the same
  estimator as `fit_conventional` (chain ladder, Bornhuetter-Ferguson,
  generalized Cape Cod, with every factor setting), started from a grid of plain
  arrays, with no database query on the path (importing it still imports ibis,
  because the kernels modules import the Triangle layer). On the same cells it
  returns exactly what `fit_conventional` returns, and from plain arrays it
  matches chainladder-python's `Chainladder`, `BornhuetterFerguson` and `CapeCod` (with
  `trend=0`) on raa and genins to a relative 1e-9. On RAA, building the grid
  and fitting took about 2 ms, against about 28 ms for `fit_conventional` and
  about 28 ms for chainladder-python's `Chainladder().fit` (medians of warm
  runs, three separate processes, one Windows laptop with an Intel Core Ultra
  9 285H; the ratio is the finding, the milliseconds move with the machine).
  It takes no `as_of`: the information date is read from the grid's latest cell,
  because a date the caller supplied would only be stamped on the result, never
  checked. Premium is keyed by origin period (a dict or a pandas Series), never
  by position. Refused by name: a grid missing a key or whose arrays disagree
  or have the wrong type, a grid that is not a run-off triangle, an incremental
  grid, origin periods that are not the first day of their period, origins
  whose spacing does not match the development step (the grid carries no
  origin grain, so without this quarterly origins on an annual step would pass
  and GCC would measure distances in the wrong unit), an origin still
  developing but observed only to an earlier date than the rest, premium
  passed to a chain ladder candidate or given twice, premium amounts that are
  not numbers, and premium keys that are not dates, collide, miss an origin or
  name one the grid does not have. A missing origin period is accepted only
  where every origin before it has already run off, as in `fit_conventional`.
- `kernels.cohort_grid_frame` (builds that grid from a pandas frame of
  `origin_period`, `dev_lag`, `value`) and `kernels.fit_mack_grid` (Mack from
  the same grid, for standard errors) are now exported and in the API
  reference. `fit_mack_grid` always estimates Mack's sigmas, so it refuses
  triangles a point fit does not need to refuse; it is not the point path.
- New refusals on existing functions. `fit_mack_grid` (and so `fit_mack` and
  `fit_mack_many`) now checks its grid exactly as `fit_conventional_grid` does;
  before, it read an incremental grid, quarterly origins on an annual step, or
  a hand-edited grid without complaint. `cohort_grid_frame` refuses a measure
  other than `"cumulative"` or `"incremental"`, a row with no origin period, and
  origin periods that are not dates; it now reads ISO strings such as
  `"2010-01-01"` as dates, where before it kept them as text and sorted them as
  text. `kernels.conventional.as_date` also accepts numpy `datetime64` values.
  A grid built from a triangle whose origin and development grains match is
  not affected, and the rest of the test suite passes unchanged.

## 0.7.1 - 2026-09-23

One pull request (#147), for notebook 04's next run: it trains forty `tlrn`
members per fit and compares the study's keep-two-of-ten rule with averaging, from
the members' own reserves. Additive only, hence a patch release.

`tlrn` keeps every trained member's reserves, and can train its members in
parallel. Nothing it produced before changes: a one-process fit is bit for bit
the 0.7.0 fit (checked on the test configuration, both feature sets, two seeds).

- `TLRN.member_company_reserves()`: `(n_members, n_companies)`, every TRAINED
  member's point reserve per company, in `selection_` row order, the members the
  selection dropped included. The kept ensemble is exactly the mean of the kept
  rows, so any group of members is scored without refitting. On the study's data
  the choice of which two members to keep moved the company Pool_APE of one
  ten-member run anywhere from 4.7% to 7.7%, which is why the dropped members are
  now kept. Averaging every member was already available: `keep = ensemble_size`.
- `TLRN.fit(..., processes=n)` trains the members in `n` worker processes
  (`spawn`, CPU only). Member `m` is seeded `seed + 1000 * m` wherever it runs and
  each worker uses the caller's torch thread count, so the members are the ones
  `processes=1` trains, weight for weight; the pool refuses a worker that trained
  on a different thread count, because on a small problem nothing else would show
  it. A script must call it under `if __name__ == "__main__":`, since every worker
  re-imports the script that started it; without the guard `fit` raises an error
  that names the guard. The workers' setup data goes through a temporary file
  rather than down each worker's pipe, because a worker that stops while starting
  never reads its pipe, and on Windows the caller then waited forever writing
  into it (measured while building this, before it shipped).
- The shared NN training loop gains `members=`: train only the named member
  indices, each exactly as the whole loop would. It cannot be combined with
  `keep`.

## 0.7.0 - 2026-09-22

The pieces a companion transformer reserving study needed before it could run
through `ibnr.gallery` end to end (#139, #140, #141, #143, #145), plus the citation
fix for the conventional examples (#138). Design and plans:
`docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md`.

New: `tlrn`, the transformer loss reserving network of that study, reproduced
from its R implementation - company x accident-year examples of line-by-lag
tokens, axial attention, a positive development-factor head with cumulative
projection, the study's point objective and checkpoint protocol, best-two-of-
ten seed selection, and company-level draws from a size-stratified historical
residual calibration. New: `mcl`, the full-matrix multivariate chain ladder,
tied out to the study's R reserves on 82 Schedule P companies. New:
`kernels/point_scores.py` (Pool_APE, Pool_PE, MAE, RMSE and the rest, with the
aggregation level explicit, `reserve_rows` and `shrink_toward`),
`kernels/residual_calibration.py`, `kernels/nn_features.py`, and six
extensions of the shared NN training loop. `GalleryEntry.evaluate()` gains a
`point` key, `mack` gains `point()`, and the NN contract carries `values`.

Numbered 0.7.0 rather than 0.6.1 because the public surface grew: two gallery
entries, `reserve_rows` on `ibnr.gallery`, eleven names on `ibnr.kernels`, and
a new key in every `evaluate()` result.

Still open, in the next notebook rather than in the package: the reproduction
of the study's headline on its own 93-company cohort set under its 10-seed,
3000-epoch protocol (`analysis/04_nn_architectures_vs_classical.ipynb`,
planned).

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

Fixed, both in `reserve_rows`, both surfaced by a notebook built on this code.
`point="native"` called `predict()` before it read `point()`, so an entry whose
draws refuse a cohort contributed no row at all even though its point estimate
was defined: `mack` on a Schedule P cohort whose newest accident year sits at
zero on the valuation diagonal raises from `predict`, and that company's total
then came out one line short, 12.42 percent from the reference, with every
number plausible. The native route now never calls `predict`, reads the realized
total as the last element of `realized_ultimates` (checking that layout rather
than assuming it), records `n_draws = 0`, and refuses `predict_kwargs` beside
itself rather than leaving them inert. Separately, the anchor and the premium
were summed over every origin the valuation observes rather than over the
origins the entry was fitted on. On the Schedule P mart, which carries accident
years 1988 to 2007 with all of them observed at a 2007 valuation, that more than
doubled company 10022's anchor - 123,543 against 54,153 - while
`realized_ultimates` stayed on the study window, so the reserve was wrong by the
accident years the fit never saw. Both are now restricted to the fitted origins,
read from `contract_["origin_periods"]` or `fit_.origin_periods`.

`sur` gains `point()`: the conditional-mean recursion of `predict()` with the
noise removed, in the `kernels.multiline` layout with a `point` column, so
`reserve_rows(..., point="native")` accepts it as it already accepted `mack`,
`mcl` and `tlrn`.

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
