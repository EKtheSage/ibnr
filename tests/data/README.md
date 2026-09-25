# Vendored test data

Small reference tables that a test compares against. Each file records where it
came from, so a number in it can be re-derived rather than trusted.

## `tweedie_glm_r.json`

R's `glm` with `statmod::tweedie(var.power = p, link.power = 0)` on the
observed increments of R ChainLadder's `GenIns`, `UKMotor`, `ABC` and `MW2014`,
`value ~ factor(origin) + factor(dev)`, at p = 0, 1, 1.5 and 2, with
`glm.control(epsilon = 1e-12)`: the reserves by origin (the fitted future
increments), every fitted increment, the coefficients and their standard
errors, the deviance, Pearson chi-squared, dispersion and residual degrees of
freedom. Also the identity link at p = 0 on `GenIns`, development factors with
a calendar trend at p = 1 on `GenIns`, `ChainLadder::glmReserve`'s total IBNR
and the chain ladder's IBNR by origin. Each triangle's cumulative cells are in
the file too, so the tests need no other copy.

Written on 2026-09-25 by `scripts/r/tweedie_glm_reference.R` (R 4.5.3,
ChainLadder 0.2.21, statmod 1.5.2): from the repository root, `Rscript
scripts/r/tweedie_glm_reference.R > tests/data/tweedie_glm_r.json`. CI has no
R, so the output is committed. Read by `tests/test_glm.py`.

## `tlrn_study_company_reserves.csv`, `tlrn_study_pairs.csv`

Produced on 2026-09-20 by replaying the companion study's R implementation of
four classical reserving methods on data loaded through `ibnr.data.schedule_p`
at Schedule P publish `20260613_041006`. Accident years 1998 to 2007, valuation
31 December 2007. Reserves are outstanding paid loss through 120 months, in USD
thousands, summed over a company's lines of business.

`tlrn_study_company_reserves.csv` has one row per company (93 of them):

| column | meaning |
|---|---|
| `company_code` | the mart's own spelling of the company code |
| `r_chain_ladder_reserve` | the R Mack chain ladder's reserve, blank where it failed |
| `r_mcl_reserve` | the R full-matrix multivariate chain ladder's reserve |
| `actual_reserve` | paid after the valuation date, the outcome |
| `point_ok` | true on the 82 companies where every R method returned a finite number |

`tlrn_study_pairs.csv` has one row per (company, line of business) the study
selected, 243 in all. A company's lines are a subset of what the mart carries:
the study kept a line only when its paid history over the first five diagonals
was complete and strictly positive, so a company in this file with two lines
usually has four in the mart.

Read by `tests/test_mcl_tieout.py`, which refits the same companies through
this package and compares. That test carries the `mart` marker and skips when
the pinned publish is not cached.

## `r_chainladder_delta.json`

R ChainLadder 0.2.21's development factors on RAA and GenIns from
`chainladder(Triangle, delta=...)` with delta 0, 1 and 2, and with delta 0 after
the highest link ratio from 12 to 24 months is weighted out. Written by
`scripts/r_chainladder_delta.R` (R 4.5.3), because CI has no R. Read by
`tests/test_development_options.py`, which requires `average="regression"`,
`"volume"` and `"simple"` to give them.

## `conventional_selection_pin.json`

A digest of every answer the conventional fit gave before `kernels/links.py`:
14 option sets, chain ladder, Bornhuetter-Ferguson and Cape Cod, on the five
public triangles in `refusal_triangles.json`, a 30 x 30 triangle built from a
formula (big enough that a change in the order of a sum shows in its bits) and
36 clrd paid triangles, 1,764 cases. Each digest covers the raw bytes of the factors, the pattern, the origin
columns, every link-ratio row and the summary flags, or a refusal's reason and
message. Written by `scripts/freeze_conventional_pin.py` run against commit
01e5c2f's source; running it against the current source must write the same
file. Read by `tests/test_development_options.py`.

## `clrd_tie_cohorts.json`

The clrd paid triangles (chainladder 0.9.2's `clrd.csv`, zeros kept) on which
ibnr 0.7.2's tie rule, `trim_ties="origin"`, gives different factors from
chainladder-python for `drop_high=1`, `drop_low=1`, both, and `drop_high=2` with
`preserve=2`: 23, 124, 83 and 30 of 681 triangles. ibnr answers 731 of the 775
paid triangles, and 50 of those are zero in every cell; chainladder stores a
zero as a missing cell, so it holds no cells for them, and the 681 are the
rest. Found by running both on every triangle;
`tests/test_development_options.py` requires the default rule to match chainladder on each, and 0.7.2's rule still not to.

## `r_mack_alpha_weights.json`

R ChainLadder 0.2.21's `MackChainLadder(Triangle, weights, alpha, est.sigma)`
on RAA, GenIns, UKMotor, ABC and MW2014 under 27 development settings each
(alpha 0, 1 and 2; every link ratio, the latest 5 or the latest 3, with and
without the second origin's first ratio left out; the highest or the lowest
ratio dropped at every age), under both `est.sigma = "Mack"` and
`"log-linear"`: 270 fits, with the triangles, the 0/1 weights, the factors,
sigmas, factor standard errors and the standard errors per origin and in
total. A fit R answers with an infinite standard error is recorded as
`"finite": false`, and one where R switched the log-linear rule to Mack's as
`"switched": true`. Written by `scripts/r_mack_alpha_weights.R` (R 4.5.3),
because CI has no R. Read by `tests/test_generalized_mack.py`, which requires
`methods.mack` with the same options to choose the same link ratios and give
the same numbers.

## `mack_default_pin.json`

Mack's answers before development options: digests of the raw bytes of
`kernels.fit_mack_grid`'s arrays, `msep_runoff` and the `to_arrow()` payload
with no `average` and no `links`, under both sigma rules and both zero rules
(256 cases), and `methods.mack`'s numbers under its defaults,
`sigma_rule="mack"` and `zero_cells="observed"` (192 cases), on the five
public triangles, the same five with one zero cell, a 30 x 30 triangle and 53
clrd paid triangles. Written by `scripts/freeze_mack_pin.py` run against the
source of the branch before this change (`feat/development-options`); running
it against that source again writes the same file. Read by
`tests/test_generalized_mack.py`.
