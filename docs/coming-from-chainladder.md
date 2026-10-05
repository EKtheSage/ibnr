# Coming from chainladder-python

`ibnr.methods` runs the traditional reserving methods on one triangle at a time,
one function per method, named after the method. This page maps
chainladder-python's classes and attributes onto those functions. The numbers
agree: on chainladder's `raa` and `genins` samples the ultimates, factors, Mack
standard errors (with their parameter and process parts), Bornhuetter-Ferguson
and Cape Cod match chainladder-python to a relative 1e-9, which
`tests/test_methods.py` checks, and `tests/test_zero_cells.py` checks the same
on triangles with zero cells. `tests/test_development_options.py` checks the
development options, Benktander and Cape Cod's trend against chainladder on
raa, genins, ukmotor, abc, mw2014, a 40 x 40 quarterly triangle and clrd,
`tests/test_generalized_mack.py` checks Mack with development options against
R's `MackChainLadder` and chainladder (see
[Mack with development options](#mack-with-development-options)), and
`tests/test_tail.py` checks tails against both (see [Tails](#tails)).

## The shape of a call

chainladder-python builds a `Triangle` object first and fits estimators to it.
ibnr takes the cells directly: a table with one row per observed cell and the
columns `origin_period` (the origin period, such as the accident year `2020`;
see [Origin labels](#origin-labels)), `dev_lag` (months from the start of the
origin period, so an annual triangle starts at 12) and `value` (the cumulative
loss). A polars DataFrame works, and so does a pyarrow Table or anything else
Arrow can read. Each call returns a `ReserveResult` whose tables are pyarrow
Tables; `.to_polars(name)` gives any of them as a polars DataFrame
(`pip install "ibnr[polars]"`). `from ibnr import methods` loads numpy and
pyarrow only, not pandas, scipy or ibis, and the methods do not load them
when they run; `import chainladder` (0.9.2) loads pandas, scipy and
scikit-learn.

```python
import polars as pl

from ibnr import methods

cells = pl.DataFrame(
    {
        "origin_period": [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
        "dev_lag": [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
        "value": [100.0, 180.0, 220.0, 240.0, 110.0, 200.0, 250.0, 90.0, 170.0, 120.0],
    }
)
premium = pl.DataFrame(
    {"origin_period": [2020, 2021, 2022, 2023], "premium": [400.0, 420.0, 450.0, 480.0]}
)

# cl.Chainladder().fit(cl.Development(n_periods=3, drop_high=True).fit_transform(tri))
chain_ladder = methods.chain_ladder(cells, history_periods=3, drop_high=True)
chain_ladder.to_polars()  # ultimate_, ibnr_ and the latest diagonal, by origin
chain_ladder.to_polars("development")  # ldf_, cdf_, and 1 / cdf_

# cl.MackChainladder().fit(tri)
mack = methods.mack(cells)
mack.to_polars("totals")  # total_mack_std_err_, with its parameter and process parts

# cl.MackChainladder().fit(cl.Development(average="simple", n_periods=3).fit_transform(tri))
mack = methods.mack(cells, average="simple", history_periods=3)

# cl.BornhuetterFerguson(apriori=0.6).fit(tri, sample_weight=premium_triangle)
bf = methods.bornhuetter_ferguson(cells, premium=premium, expected_loss_ratio=0.6)

# cl.Benktander(apriori=0.6, n_iters=2).fit(tri, sample_weight=premium_triangle)
benktander = methods.benktander(cells, premium=premium, expected_loss_ratio=0.6, n_iters=2)

# cl.CapeCod(decay=0.75, trend=0.05).fit(tri, sample_weight=premium_triangle)
cape_cod = methods.cape_cod(cells, premium=premium, decay=0.75, trend=0.05)
cape_cod.to_polars()["trended_loss_ratio"]  # apriori_
cape_cod.to_polars()["expected_loss_ratio"]  # detrended_apriori_
```

## Lookup table

| chainladder-python | ibnr |
|---|---|
| `cl.Chainladder()` | `methods.chain_ladder(cells)` |
| `cl.MackChainladder()` | `methods.mack(cells)` |
| `cl.MackChainladder()` on `cl.Development(**options)`'s output | `methods.mack(cells, **options)`, with the options below (not `average="median"`) |
| `cl.BornhuetterFerguson(apriori=r)` with `sample_weight=` a premium triangle | `methods.bornhuetter_ferguson(cells, premium=..., expected_loss_ratio=r)` |
| `cl.Benktander(apriori=r, n_iters=n)` with `sample_weight=` | `methods.benktander(cells, premium=..., expected_loss_ratio=r, n_iters=n)` |
| `cl.CapeCod(decay=d, trend=t, n_iters=n)` with `sample_weight=` | `methods.cape_cod(cells, premium=..., decay=d, trend=t, n_iters=n)` |
| `cl.Development(n_periods=n)` (`-1` for all) | `history_periods=n` (`None` for all) |
| `cl.Development(average="volume" / "simple" / "regression")` | `average="volume"` / `"simple"` / `"regression"`, and also `"median"` |
| `cl.Development(drop_high=k, drop_low=k)` (`True` is 1) | `drop_high=k, drop_low=k` (`True` is 1) |
| `cl.Development(preserve=p)` | `preserve=p` |
| `cl.Development(drop_above=a, drop_below=b)` | `drop_above=a, drop_below=b` (a ratio equal to a bound is kept; below) |
| `cl.Development(drop=("1982", 12))` | `exclude=[(1982, 12)]` |
| `cl.Development(drop_valuation="1994")` | `exclude_valuations=[1995]` (the same link ratios; below) |
| `cl.Development(sigma_interpolation="log-linear" / "mack")` | `methods.mack(cells, sigma_rule="log_linear" / "mack")` |
| `.ultimate_` | `result.origins["ultimate"]` |
| `.ibnr_` | `result.origins["ibnr"]` |
| `.latest_diagonal` | `result.origins["latest"]` |
| `.ldf_` | `result.development["factor"]` |
| `.cdf_` | `result.development["cdf"]` |
| `1 / .cdf_` | `result.development["pct_reported"]` |
| `.sigma_`, `.std_err_` | `result.development["sigma"]`, `["std_err"]` (Mack only) |
| `.mack_std_err_` (its ultimate column; the `Mack Std Err` column of `.summary_`) | `result.origins["mack_se"]` |
| `.parameter_risk_`, `.process_risk_` (their ultimate columns) | `result.origins["parameter_se"]`, `["process_se"]` |
| `.total_mack_std_err_` | `result.totals["mack_se"]` |
| `.total_parameter_risk_`, `.total_process_risk_` (their ultimate columns) | `result.totals["parameter_se"]`, `["process_se"]` |
| `.apriori_` (Cape Cod) | `result.origins["trended_loss_ratio"]` |
| `.detrended_apriori_` (Cape Cod) | `result.origins["expected_loss_ratio"]` |

Every link ratio, and whether the factor used it, is in `result.link_ratios`,
with the reason for any that were left out: `zero_cell` when either cumulative
is zero (or, with `zero_cells="observed"`, `undefined_ratio` when the earlier
one is), `history_window`, `explicit_exclusion`, `valuation_exclusion`,
`drop_above`, `drop_below`, `drop_low` or `drop_high`. A ratio carries the
first rule that left it out, in that order.

Five details of the correspondence:

- A zero cumulative is read as chainladder-python reads it, as a missing
  cell: a link ratio is used only when neither of its two cells is zero, so
  the ratio into a zero and the ratio out of it both drop, from the factors
  and from Mack's sigmas. This is the `zero_cells="missing"` default. As in
  chainladder, a ratio left out for a zero keeps its place in a
  `history_periods` window, so the window holds fewer ratios rather than
  reaching back to an older origin, and a sigma at an age left with one link
  ratio is filled in by the `sigma_rule`, as chainladder's
  `sigma_interpolation` fills it. `tests/test_zero_cells.py` checks these
  against chainladder-python; the differences that remain are listed below.
- `exclude` names a link ratio by its origin and the age it develops FROM,
  the same as chainladder's `drop`. A pair that is not a link ratio of the
  triangle is refused rather than ignored.
- With `drop_high` or `drop_low` on a complete triangle, the last age has a
  single link ratio. chainladder-python keeps it; so does ibnr, and it marks
  that age in `development["extreme_trimming_skipped"]`. Pass
  `exhausted_exclusions="raise"` to refuse instead. `preserve` works as
  chainladder's does for the trims: it counts the ratios both trims would
  leave, and at an age where that is fewer, neither trim is made.
- Equal link ratios are trimmed as chainladder-python trims them
  (`trim_ties="volume"`, the default): among equal ratios `drop_high` leaves out
  the one with the larger earlier cumulative and `drop_low` the smaller, and
  equal amounts too go by origin, the newer for `drop_high` and the older for
  `drop_low`. `trim_ties="origin"` ranks equal ratios by origin alone, the rule
  of ibnr 0.7.2 and of `kernels.ConventionalCandidate`, which disagrees with
  chainladder on 23 of 681 clrd paid triangles for `drop_high=True`, on 124
  for `drop_low=True` and on 83 for both. The 681 are the 731 ibnr answers less
  the 50 that are zero in every cell, which chainladder holds no cells for.
- chainladder-python reports a fully developed origin's IBNR and Mack standard
  error as NaN; ibnr reports 0.0, because nothing is left to develop.

`methods.mack` defaults to `sigma_rule="log_linear"`, as chainladder-python
does. (`kernels.fit_mack`, the Triangle path, keeps `"mack"` as its default so
published numbers do not move.)

## Behaviour that differs on purpose

- **An excluded valuation names the later end of a link ratio.**
  chainladder-python's `drop_valuation` leaves out every link ratio whose
  EARLIER cell is valued on that date. ibnr's `exclude_valuations` leaves out
  every link ratio that develops INTO that date, so excluding 2020 removes the
  development that happened during 2020. The same link ratios are one year
  apart: chainladder's `drop_valuation="1994"` is `exclude_valuations=[1995]`.
  So passing chainladder's value through unchanged moves the answer (on the
  example workbook's triangle, New Jersey Manufacturers' workers' compensation
  paid, `"1994"` gives a total IBNR of 376,546.99 in chainladder and 375,199.78
  here), and naming the latest valuation, which chainladder ignores without a
  word, now leaves out the latest diagonal's link ratios. Shifting the value one
  year keeps chainladder's numbers, except that chainladder's latest year then
  names a date after the triangle, which ibnr refuses (`not_in_triangle`). A
  valuation no link ratio develops into, and a year label on a quarterly or
  monthly triangle (chainladder reads `"1994"` there as its first quarter), are
  refused by name.
- **The rules act one after another.** chainladder-python applies its drop
  rules independently: `drop_high` ranks ratios that `drop`, `drop_valuation`
  or a bound already left out, and each rule checks `preserve` on its own, so
  two rules together can empty an age that `preserve` was set to protect. ibnr
  runs them in order (the zero rule, `history_periods`, `exclude`,
  `exclude_valuations`, `drop_above`/`drop_below`, `drop_high`/`drop_low`),
  each on the ratios the ones before left, and `preserve` counts what is
  really left. With `history_periods`, the trims and `average` alone the two
  agree; they differ only when two exclusion rules are combined. On raa,
  `exclude=[(1982, 12)]` with `drop_high=True` leaves out 1982's 40.4 and then
  1985's 8.76 here, where chainladder leaves out only 1982's.
- **A ratio equal to a bound is kept.** `drop_above` leaves out ratios strictly
  above it and `drop_below` those strictly below it, as chainladder's docstring
  describes them; chainladder itself leaves out a ratio equal to the bound too.
  It matters for `drop_below=1.0`: 498 of the 725 clrd paid triangles have a
  link ratio of exactly 1.0. The bounds' `preserve` also counts the ratios in
  the `history_periods` window, where chainladder counts the whole column and
  can empty the window.
- **`n_iters` is a whole number from 1 to 10,000.** chainladder-python reads
  `n_iters=0` as the expected loss method (every ultimate is the a priori
  one, a fully developed origin included) and accepts `2.5`; ibnr refuses both.
  It also refuses a count above 10,000, because each iteration is one more pass
  of a loop: a million took 1.3 seconds on raa, so a count typed by a user could
  hold a request for minutes.
- **One setting for every age.** chainladder-python takes a list per age for
  the drop options, `preserve` and `average`; ibnr refuses a list, and applies
  one setting at every age.
- **`trend` must be above -1.** chainladder-python answers `trend=-1` with
  missing loss ratios and `trend=-1.5` with negative ones.
- **Cape Cod keeps every origin.** On a triangle with more origins than ages,
  chainladder-python leaves the fully developed origins that are not on the
  latest diagonal out of the ultimates and out of the Cape Cod pool. ibnr keeps
  them: their ultimate is their latest amount, and their losses count in the
  pooled loss ratio, trended from their own period's end.
- **Zeros can be kept as data.** chainladder-python stores a zero cell as
  missing, so a zero cumulative at 12 months disappears from its triangle.
  `ibnr.methods` follows it by default (`zero_cells="missing"`, above). Pass
  `zero_cells="observed"` to keep a zero as an observed cell instead: the
  chain ladder, Bornhuetter-Ferguson and Cape Cod then use the link ratio
  into a zero (a ratio of 0) and leave out only the one out of it, which has
  no value, with reason `undefined_ratio`; Mack keeps both in its volume sums,
  as R's `MackChainLadder` does, and refuses standard errors when a
  still-developing origin's latest cumulative is zero. The kernels
  (`kernels.ConventionalCandidate`, `kernels.fit_mack`) default to
  `"observed"`. Either way an unobserved cell is one you leave out.
- **A zero latest cumulative gives 0, not a missing number.** Under the default
  `zero_cells="missing"`, an origin whose latest cumulative is zero still has
  0 as its latest amount, so ibnr reports its ultimate as 0 and its Mack
  standard error as 0 (the limit of Mack's formula as that amount goes to
  zero), where chainladder-python leaves both missing. The total standard error
  is the same in both.
- **Two rarer Mack cases still differ with zeros present.** chainladder-python
  also stores a sigma of exactly 0 (every link ratio at an age equal) as
  missing and fills it in, which ibnr does not, so Mack's standard errors can
  differ at such an age. And where no link ratio is left at an age, ibnr's Mack
  refuses the triangle by name even when the only origins that need that age
  sit at a latest amount of zero; chainladder-python leaves that factor missing
  and still gives a total standard error.
- **Duplicate cells are refused, not added together.** chainladder-python sums
  rows that land in the same cell. ibnr refuses them, naming the cells, because
  two rows for one cell are usually two companies or two lines passed at once,
  and the methods fit one cohort at a time.
- **Negative cumulatives are refused.** chainladder-python fits a triangle
  with a negative cumulative loss. ibnr refuses it and names the cells, because
  a chain-ladder factor is a ratio of cumulatives and Mack's variance is
  weighted by them.
- **A missing origin period is refused.** Every origin period from the first
  to the last needs cells, one development step apart. The refusal names the
  missing periods; give a period with no business its cells as zeros.
- **`methods.mack` needs a sigma it can estimate.** A triangle with at most one
  link ratio at every age (two origins, for example) gives Mack's sigma
  nothing to measure. ibnr refuses it rather than report a standard error of
  0 that would read as no uncertainty; `methods.chain_ladder` still gives the
  ultimates.
- **Infinite amounts are refused.** chainladder-python fits a triangle with an
  infinite cumulative and reports a finite, wrong total. ibnr refuses it and
  names the cells.

## Mack with development options

`methods.mack` takes the chain ladder's development options: `average`,
`history_periods`, `drop_high`, `drop_low`, `preserve`, `drop_above`,
`drop_below`, `exclude`, `exclude_valuations`, `trim_ties` and
`exhausted_exclusions`. The same options choose the same link ratios and give
the same factors, bit for bit, as `methods.chain_ladder`, and the result carries
the same `link_ratios` table. `average` is Mack's alpha: `"simple"` is 0,
`"volume"` 1 and `"regression"` 2, so a link ratio is weighted by the amount it
starts from to that power, and a step's variance is sigma squared times the
amount to the power 2 - alpha (Mack 1999). R ChainLadder's
`MackChainLadder(Triangle, weights, alpha, est.sigma)` is the reference: on
raa, genins, ukmotor, abc and mw2014 under 27 settings each (every average;
all link ratios, the latest 5 or the latest 3; with and without an exclusion;
the highest or the lowest dropped), the 224 fits R answers under its own rules
agree with ibnr to a relative 1e-12 on the total standard error.
chainladder-python agrees too, except in three places, each a difference on
purpose:

- **Future development keeps its full variance.** chainladder multiplies the
  process variance of each projected cell by the estimation weight of the cell
  it starts from. The first projected step starts from the latest diagonal,
  which has no link ratio, and any `drop_high`, `drop_low`, `drop_above` or
  `drop_below`, or a `drop_valuation` naming the latest valuation, sets that
  weight to 0, so a whole year of process variance goes missing. On raa,
  `drop_above=100` leaves out no link ratio at all and still takes the total
  standard error from 26,880.74 to 13,037.79; `drop_high=1` gives 7,796.95
  where the right figure is 14,811.95, `drop_high=2` 7,658.91 for 11,359.79,
  and `drop_low=1` 15,115.94 for 31,377.00. On the example workbook's triangle
  (New Jersey Manufacturers, workers' compensation paid, 1988-1997)
  `drop_high=1` gives 10,329.10 here, where chainladder, and so the Reserving
  app's `/reserve` today, gives 5,914.68. R has no such step: its weights
  choose link ratios, and the latest diagonal keeps a weight of 1.
- **A factor kept from one link ratio has that ratio's standard error.** When
  the options leave one link ratio at an age, chainladder computes the
  factor's standard error from the first origin's amount there, whichever
  ratio was kept (R does the same, and answers infinity when the first
  origin's ratio was left out). On raa with `drop_high=2`, 84 to 96 months
  keeps 1983's ratio only: its standard error is 0.01901 here and 0.02142 in
  chainladder.
- **A sigma of exactly 0 is not read as 1e-320.** When every link ratio kept
  at an age is equal, sigma there is 0. When another age's sigma has to be
  filled in, chainladder's log-linear fill takes the logarithm of 1e-320 for
  it, which drags the fill toward 0; ibnr, like R, fits the line through the
  positive sigmas only, and when only the last age is filled uses Mack's rule,
  as it did before. Whole-thousand amounts make equal ratios common: with
  `drop_high=1`, of the 353 clrd paid triangles that are full positive
  staircases, ibnr answers 348, and this is the difference on 108 of them; the
  other 240 agree with chainladder once its two defects above are patched. On
  one more (clrd triangle 377, `drop_low=1`) chainladder projects the 1990
  origin's 108-month cell at 1365 rather than 1365 x 1.001203, and R's total,
  331.554588864, is ibnr's.

Two more things differ from chainladder:

- `history_periods=1` is refused by name: one link ratio at every age leaves
  no sigma to estimate, and chainladder answers NaN with a warning. So are any
  options that leave at most one link ratio at every age.
- An age the options leave with no link ratio is refused (`no_link_ratio`).
  chainladder projects it at 1.0 and gives NaN standard errors for every
  origin; `methods.chain_ladder(..., unsupported_factor="unity")` gives the
  ultimates at 1.0 if that is what you want, and `methods.mack` has no
  `unsupported_factor`.

`average="median"` is refused for Mack (`not_supported`): a median is not a
weighted mean of the link ratios, so Mack's variance does not exist for it.
chainladder has no geometric average, and ibnr refuses `"geometric"` for the
same reason. With `drop_high`, `drop_low`, `drop_above` or `drop_below` the
ratios are chosen after looking at them, which Mack's formulas do not allow
for, so the standard errors are approximate and tend to be low; R and
chainladder apply the formulas the same way. `development` gains
`n_selected`, `extreme_trimming_skipped`, `bounds_skipped` and
`sigma_extrapolated` (the sigma came from `sigma_rule`, because the age kept
fewer than two link ratios). The one-year claims development result in
`ibnr.kernels` refuses a fit with any development option, naming each: the
Merz-Wuthrich formulas are derived for the volume average over every link
ratio, and the re-reserving re-runs that estimator on next year's triangle.

## Tails

Every method in `ibnr.methods` takes a tail: `chain_ladder`,
`bornhuetter_ferguson`, `benktander`, `cape_cod` and `mack`. chainladder puts a
`cl.TailConstant` or a `cl.TailCurve` between `cl.Development` and the method;
ibnr takes the same settings as keyword options on the method.

```python
# cl.Chainladder().fit(cl.TailConstant(1.05).fit_transform(cl.Development().fit_transform(tri)))
methods.chain_ladder(cells, tail="constant", tail_factor=1.05)

# cl.TailCurve("exponential", extrap_periods=50, projection_period=24, attachment_age=36)
methods.chain_ladder(cells, tail="exponential", tail_steps=50, tail_rows=2, tail_attach_lag=36)

# cl.TailCurve("weibull", fit_period=(12, 48)) in front of cl.MackChainladder()
methods.mack(cells, tail="weibull", tail_fit_lags=(12, 36))
```

| chainladder-python | ibnr |
|---|---|
| no tail | `tail=None` (the default) |
| `cl.TailConstant(tail=t)` | `tail="constant", tail_factor=t` |
| `cl.TailConstant(decay=d)` | `tail_decay=d` (0.5 by default in both) |
| `cl.TailCurve(curve=c)`, `c` one of `"exponential"`, `"inverse_power"`, `"weibull"` | `tail=c` |
| `attachment_age=a` | `tail_attach_lag=a` |
| `extrap_periods=n` | `tail_steps=n` (100 by default in both; development steps, not months) |
| `projection_period=p` | `tail_rows=int(p / 12) * (12 // dev_grain_months)` (one year of steps by default in both) |
| `fit_period=(s, e)` | `tail_fit_lags=(s, e - dev_grain_months)`: chainladder leaves out the link from `e` |
| `errors="ignore"`, `reg_threshold=(1.00001, None)` | fixed: a factor at or below 1.00001 is left out of the curve fit; not an option |
| `.tail_` | `result.totals["tail_factor"]` |
| `.ldf_`, `.cdf_` with the tail columns | `result.development`, which goes on past the last observed age (below) |
| `MackChainladder` on the tailed `Development` | `methods.mack(cells, tail=...)`, and `tail_sigma`, `tail_std_err` (R's `tail.sigma`, `tail.se`) |

`development` has one row per observed age and then `tail_rows` rows beyond the
last one, at the ages that follow. On each row `factor` is the factor to the
next row (null on the final row), and `cdf` is the development to ultimate with
the tail in it, so on the final row it is what is left of the tail. `source`
says whether the factor came from the link ratios or the tail, `curve_factor`
is a fitted curve's factor at every row but the final one, for plotting fitted
against selected, and `in_tail_fit` says which observed ages' factors the curve
went through. chainladder's `ldf_` values are every `factor` that is not null
and then the final row's `cdf`, and its `cdf_` values are every row's `cdf`: on
raa with a constant 1.05, `108-120`, `120-132`, `132-144` and `120-Ult`,
`132-Ult`. `totals["tail_factor"]` is the development beyond the last observed
age, which with an earlier attachment is chainladder's `tail_`, the part of the
constant left after the attached ages, not the constant itself. Mack adds
`tail_sigma`, `tail_std_err` and `tail_position` to `totals`. Only `mack` takes
`tail_sigma` and `tail_std_err`: passed to the other four methods, either is a
Python `TypeError`, as `sigma_rule` is, not an `ibnr.errors.Refusal`.

The numbers agree: the factors after the attachment, the steps shown beyond
the triangle, the rest and the tail factor match chainladder to a relative
1e-12 on raa, genins, ukmotor, abc, mw2014, the tail sample's paid and incurred
triangles and clrd's commercial auto paid, for each curve with and without an
attachment, a fit range and `extrap_periods`/`projection_period`, and each
constant with a decay and an attachment. The four point methods' ultimates
match to 1e-10, and Mack's standard errors, their two parts and the tail's
sigma and standard error to 1e-9, wherever no factor is at or below 1 and no
sigma is 0. Mack's also match R's `MackChainLadder(tail=, tail.se=,
tail.sigma=)` on raa, genins, ukmotor, abc and mw2014 at alpha 0, 1 and 2, to
1e-8 (`tests/test_tail.py`).

Where they differ, on purpose:

- **An attachment at the first age is honoured.** chainladder ignores
  `TailConstant(attachment_age=12)` on an annual triangle and attaches at the
  last age (its test for an attachment is false for index 0); `TailCurve`
  honours it. ibnr honours it for both.
- **`tail_fit_lags` includes both ends.** chainladder's `fit_period=(12, 108)`
  leaves out the link from 108 (raa: tail 1.00864 against 1.00944 for
  `(12, 120)`); ibnr's `(12, 108)` fits it.
- **Every case chainladder rounds, crashes on or answers as no tail is
  refused by name.** An attachment age off the grid (chainladder rounds it up)
  or after the last age (a numpy error); a curve with fewer than two factors
  above 1.00001 to fit, such as a fit range holding one link (chainladder's
  tail is NaN, read as 1.0, so the result silently equals no tail); a decay
  above 1 (chainladder's tail overflows to NaN, also no tail); more rows shown
  than steps extrapolated (chainladder fits, then fails to build its tables); a
  curve whose factors grow with age (chainladder gives an exponential tail of
  5.8e229 on factors 1.05 to 1.09), and an inverse power curve whose slope is
  between -1 and 0, whose product never converges, so the tail would be set by
  `tail_steps` alone (`tail_not_decaying`).
- **Mack's tail variance reads the logarithms it can take.** The tail's sigma
  and standard error are read off straight lines through the logarithms of the
  sigmas and the factors' standard errors, at the age where a line through
  `log(f - 1)` reaches the tail. ibnr, like R, fits the first line through the
  factors above 1 only and the other two through the positive values only.
  chainladder leaves a left-out point's age in its sums, and fills a sigma of
  0 with 1e-320, so its lines are not the least-squares lines whenever a
  factor is at or below 1 or a sigma is 0 or missing: on a quarterly triangle
  whose last three factors are exactly 1, a constant 1.05 gives a total
  standard error of 1,054.30 here and 934.35 in chainladder, and 1.01 gives
  1,021.18 here and 4.8e15 in chainladder (`tests/test_tail.py`). Where
  nothing is left out, the two agree.
- **Mack refuses what has no variance.** A tail attached before the last age
  (`not_supported`: chainladder keeps the link ratios' sigmas and standard
  errors for the curve factors that replace them, which no formula derives); a
  tail factor below 1 without `tail_sigma` and `tail_std_err`
  (`variance_not_estimable`: chainladder applies the tail but reads its
  variance as if it were 1.001, and R ignores a tail below 1 altogether); and a
  tail larger than the line through the factors gives even at the first link,
  whose sigma would be extrapolated backwards (`variance_not_estimable`; on raa
  any tail above about 2.3). `tail_sigma` and `tail_std_err` answer all but the
  first.
- **A tail below 1 moves the ultimates**, as in chainladder (raa with 0.95:
  202,466.12). R ignores it.

The one-year claims development result, `kernels.simulate_ultimates` and the
held-out draws refuse a tailed fit (`not_supported`): the Merz-Wuthrich
formulas and R's `CDR.MackChainLadder` cover the development inside the
triangle only.

## When a method refuses

chainladder-python has no error class of its own: bad input raises whatever
the numpy, pandas or scikit-learn line underneath raises (`KeyError` for an
unknown `average`, `IndexError` for a `drop` pair the triangle does not have,
pandas' `DateParseError` for an origin it cannot read), or is answered.

Every input an `ibnr.methods` function will not answer is refused with
`methods.Refusal` (the class lives in `ibnr.errors`), a `ValueError`. Any other
exception from these functions, a plain `ValueError` included, is a defect in
ibnr and worth reporting; a `TypeError` from a missing or misspelled keyword is
Python's own. So a service can answer a `Refusal` as bad input and anything
else as its own failure:

```python
from ibnr import methods

try:
    result = methods.chain_ladder(cells)
except methods.Refusal as refusal:
    body = refusal.to_dict()  # JSON: reason, kind, option, column, cells, ...
```

The fields:

| Field | What it holds |
|---|---|
| `reason` | one code from `ibnr.errors.REASONS`, such as `"negative_cumulative"` |
| `kind` | `"input"`: the request must change; `"model"`: another option or method can answer |
| `method` | the function that refused, such as `"mack"` |
| `option`, `column` | the argument at fault in the function's own terms (`"cells"`, `"premium"`, `"exclude"`), and the column of a table argument (`"origin_period"`, `"dev_lag"`, `"value"`) |
| `cells` | the cells or origins at fault: `origin` as you wrote it (value and type), `origin_period` (the period's first day), `dev_lag`, `value` |
| `links` | `(from_dev_lag, to_dev_lag)` in months, for a refusal about a development age |
| `rows` | 0-based rows of the table, where a cell cannot be named (a missing origin) |
| `count` | how many are at fault in all; `cells` and `rows` keep at most 100 |

The codes of kind `"input"`: `invalid_option`, `invalid_table`,
`missing_value`, `not_finite`, `unreadable_label`, `invalid_age`,
`grain_mismatch`, `duplicate`, `negative_cumulative`, `origin_gap`,
`not_run_off`, `not_in_triangle`, `origin_not_covered`. Of kind `"model"`:
`no_link_ratio`, `exclusions_exhausted`, `variance_not_estimable`,
`negative_increment`, `zero_increment`, `negative_fitted_mean`,
`not_identified`, `degenerate_fit`, `did_not_converge`, `tail_not_decaying`,
`empty_residual_pool`, `result_not_finite`, `not_supported`,
`negative_projection`. The `ibnr.errors` module lists what each means. Codes
are never renamed or removed; a new one may be added in a patch release, so
handle a code you do not know by its `kind`.

What some of chainladder-python's answers become:

| chainladder-python | ibnr |
|---|---|
| `Development(average=1.0)`: `KeyError` | `invalid_option`, `option="average"` |
| `Development(drop_high=np.int64(2))` or `2.0`: `TypeError` | `np.int64(2)` is answered as 2; `2.0` is `invalid_option` |
| `Development(drop_high=-1)`: answered, nothing dropped | `invalid_option`, `option="drop_high"` |
| `Benktander(n_iters=-1)`: `IndexError` | `invalid_option`, `option="n_iters"` |
| `Development(drop_valuation="2005")` past the triangle: a warning, then answered | `not_in_triangle`, `option="exclude_valuations"` |
| `Development(drop=("1850", 12))`: `IndexError` | `not_in_triangle`, `option="exclude"`, the pair in `cells` |
| an origin `"abc"`: pandas `DateParseError` | `unreadable_label`, `column="origin_period"`, the rows in `rows` |
| a negative cumulative: answered | `negative_cumulative`, the cells in `cells` |
| a cell sent twice: summed | `duplicate`, one entry in `cells` per row |
| `dev_lag` 18 on an annual triangle: regridded | `grain_mismatch`, `column="dev_lag"` |
| `MackChainladder` on two origins: NaN standard error | `variance_not_estimable` |
| `Development(n_periods=1)`, then `MackChainladder`: NaN standard errors | `variance_not_estimable`, `option="history_periods"` |
| `Development(drop=("1981", 108))` on raa, then `MackChainladder`: factor 1.0, NaN standard errors | `no_link_ratio`, `option="exclude"` |

## Origin labels

`origin_period` accepts the ways an origin period is usually written, and the
premium table (or dict) and the origins in `exclude` accept the same forms,
each independently of how the cells write theirs:

- an integer year with its century: `2020` is the calendar year 2020 (a
  two-digit year such as `97` is refused rather than read as the year 97);
- a text label: a year `"2020"`, a quarter `"2020Q3"` or a month `"2020-03"`;
- a date, a timestamp (read as its date, in its own time zone when it has one)
  or an ISO date string such as `"2020-12-31"`.

A date must be the first day or the last day of its period, and it is read
with `dev_grain_months`, which is also the length of an origin period. The
first day of a month starts a period, and the last day of a month ends a
period that began `dev_grain_months` months earlier. So with
`dev_grain_months=12`, `2020-01-01` and `2020-12-31` are both the accident year
2020, and `2021-06-30` is the year from July 2020 to June 2021, a fiscal or
treaty year that an integer cannot name; with `dev_grain_months=3`,
`2020-12-31` is the fourth quarter of 2020. Any other day of the month is
refused. A year, quarter or month label must name a period `dev_grain_months`
long, so `2020` with `dev_grain_months=3` is refused rather than read as a
quarter. A label that names no period (`"2020Q5"`, `"FY20"`) is refused by
name, and so is a period written two ways in one column (`"2020"` and
`"2020-12-31"` together), because the results echo the label back.

```python
from datetime import date

import polars as pl

from ibnr import methods

fiscal = pl.DataFrame(
    {
        "origin_period": [date(2021, 6, 30)] * 2 + [date(2022, 6, 30)],
        "dev_lag": [12, 24, 12],
        "value": [100.0, 150.0, 120.0],
    }
)
result = methods.chain_ladder(fiscal)
result.to_polars()  # origin 2021-06-30 has origin_period 2020-07-01
print(result.as_of)  # 2022-06-30
```

The results show your label in a column `origin`, the first column of
`origins` and `link_ratios`, with the type you passed it in: integers stay
integers (int64), text stays text (string), dates stay dates (date32) and
timestamps keep their own type. Beside it, `origin_period` is always the first
day of the period, as everywhere else in ibnr, while a valuation date is the
last day of one: the accident year 2020 at 12 months is valued 2020-12-31, and
`result.as_of` is the valuation date of the latest diagonal. `dev_lag` still
counts from the period's first day whichever day names it, so the accident
year written `2020-12-31` has its first cell at `dev_lag` 12, valued
2020-12-31, not at `dev_lag` 0.

## Not there yet

- `fillna`. chainladder's `drop_below` defaults to 0, which leaves out
  negative link ratios; ibnr has none to leave out, since it refuses negative
  cumulatives.
- Origin periods longer than a development step, such as annual origins
  developed quarterly: `dev_grain_months` must equal the origin period length.
- The bootstrap (`cl.BootstrapODPSample`) through `ibnr.methods`. An
  over-dispersed Poisson bootstrap of next year's diagonal is in
  `ibnr.kernels` for the one-year claims development result.
