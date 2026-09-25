# Vendored test data

Small reference tables that a test compares against. Each file records where it
came from, so a number in it can be re-derived rather than trusted.

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
