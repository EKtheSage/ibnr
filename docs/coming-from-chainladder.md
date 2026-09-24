# Coming from chainladder-python

`ibnr.methods` runs the traditional reserving methods on one triangle at a time,
one function per method, named after the method. This page maps
chainladder-python's classes and attributes onto those functions. The numbers
agree: on chainladder's `raa` and `genins` samples the ultimates, factors, Mack
standard errors (with their parameter and process parts), Bornhuetter-Ferguson
and Cape Cod match chainladder-python to a relative 1e-9, which
`tests/test_methods.py` checks.

## The shape of a call

chainladder-python builds a `Triangle` object first and fits estimators to it.
ibnr takes the cells directly: a table with one row per observed cell and the
columns `origin_period` (the first day of the origin period), `dev_lag` (months
from that day, so an annual triangle starts at 12) and `value` (the cumulative
loss). A polars DataFrame works, and so does a pyarrow Table or anything else
Arrow can read. Each call returns a `ReserveResult` whose tables are pyarrow
Tables; `.to_polars(name)` gives any of them as a polars DataFrame
(`pip install "ibnr[polars]"`).

```python
from datetime import date

import polars as pl

from ibnr import methods

cells = pl.DataFrame(
    {
        "origin_period": [date(2020, 1, 1)] * 4
        + [date(2021, 1, 1)] * 3
        + [date(2022, 1, 1)] * 2
        + [date(2023, 1, 1)],
        "dev_lag": [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
        "value": [100.0, 180.0, 220.0, 240.0, 110.0, 200.0, 250.0, 90.0, 170.0, 120.0],
    }
)
premium = pl.DataFrame(
    {
        "origin_period": [date(2020 + i, 1, 1) for i in range(4)],
        "premium": [400.0, 420.0, 450.0, 480.0],
    }
)

# cl.Chainladder().fit(cl.Development(n_periods=3, drop_high=True).fit_transform(tri))
chain_ladder = methods.chain_ladder(cells, history_periods=3, drop_high=True)
chain_ladder.to_polars()  # ultimate_, ibnr_ and the latest diagonal, by origin
chain_ladder.to_polars("development")  # ldf_, cdf_, and 1 / cdf_

# cl.MackChainladder().fit(tri)
mack = methods.mack(cells)
mack.to_polars("totals")  # total_mack_std_err_, with its parameter and process parts

# cl.BornhuetterFerguson(apriori=0.6).fit(tri, sample_weight=premium_triangle)
bf = methods.bornhuetter_ferguson(cells, premium=premium, expected_loss_ratio=0.6)

# cl.CapeCod(decay=0.75, trend=0).fit(tri, sample_weight=premium_triangle)
cape_cod = methods.cape_cod(cells, premium=premium, decay=0.75)
cape_cod.to_polars()["expected_loss_ratio"]  # apriori_
```

## Lookup table

| chainladder-python | ibnr |
|---|---|
| `cl.Chainladder()` | `methods.chain_ladder(cells)` |
| `cl.MackChainladder()` | `methods.mack(cells)` |
| `cl.BornhuetterFerguson(apriori=r)` with `sample_weight=` a premium triangle | `methods.bornhuetter_ferguson(cells, premium=..., expected_loss_ratio=r)` |
| `cl.CapeCod(decay=d, trend=0)` with `sample_weight=` | `methods.cape_cod(cells, premium=..., decay=d)` |
| `cl.Development(n_periods=n)` | `history_periods=n` |
| `cl.Development(average="volume" / "simple")` | `average="volume"` / `"simple"`, and also `"median"` |
| `cl.Development(drop_high=True, drop_low=True)` | `drop_high=True, drop_low=True` |
| `cl.Development(drop=("1982", 12))` | `exclude=[(date(1982, 1, 1), 12)]` |
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
| `.apriori_` (Cape Cod) | `result.origins["expected_loss_ratio"]` |

Every link ratio, and whether the factor used it, is in `result.link_ratios`,
with the reason for any that were left out (`history_window`, `drop_high`,
`drop_low`, `explicit_exclusion`, or `undefined_ratio` when the earlier
cumulative is zero).

Three details of the correspondence:

- `exclude` names a link ratio by its origin and the age it develops FROM,
  the same as chainladder's `drop`. A pair that is not a link ratio of the
  triangle is refused rather than ignored.
- With `drop_high` or `drop_low` on a complete triangle, the last age has a
  single link ratio. chainladder-python keeps it; so does ibnr, and it marks
  that age in `development["extreme_trimming_skipped"]`. Pass
  `exhausted_exclusions="raise"` to refuse instead.
- chainladder-python reports a fully developed origin's IBNR and Mack standard
  error as NaN; ibnr reports 0.0, because nothing is left to develop.

`methods.mack` defaults to `sigma_rule="log_linear"`, as chainladder-python
does. (`kernels.fit_mack`, the Triangle path, keeps `"mack"` as its default so
published numbers do not move.)

## Behaviour that differs on purpose

- **Explicit zeros stay zeros.** chainladder-python stores a zero cell as
  missing, so a zero cumulative at 12 months disappears from its triangle. In
  ibnr a cell you pass is observed, zero or not; an unobserved cell is one you
  leave out. An option to read zeros as missing is planned.
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
- **Integer accident years are not dates.** `origin_period` is the first day of
  the period, so the accident year 2020 is `date(2020, 1, 1)`; in polars,
  `pl.date(pl.col("year"), 1, 1)` makes that column. A timestamp is read as its
  date, in its own time zone when it has one.

## Not there yet

- Tail factors (`cl.TailCurve`, `cl.TailConstant`): every method projects to
  the last observed development age.
- Benktander (`cl.Benktander`).
- Cape Cod's `trend`: amounts are compared as they are.
- The `"regression"` average.
- `drop_valuation`, and integer counts for `drop_high` / `drop_low` (ibnr drops
  one ratio at each end).
- `drop_above` / `drop_below` / `preserve` / `fillna`, and per-age lists for
  `n_periods`, `average` and the drop options (ibnr applies one setting at
  every age). chainladder's `drop_below` defaults to 0, which leaves out
  negative link ratios; ibnr has none to leave out, since it refuses negative
  cumulatives.
- Origin periods longer than a development step, such as annual origins
  developed quarterly: `dev_grain_months` must equal the origin period length.
- Development options under Mack: `methods.mack` uses every link ratio,
  volume-weighted, because Mack's standard-error formulas are derived for that
  estimator.
- The bootstrap (`cl.BootstrapODPSample`) through `ibnr.methods`. An
  over-dispersed Poisson bootstrap of next year's diagonal is in
  `ibnr.kernels` for the one-year claims development result.
