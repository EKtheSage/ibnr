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
