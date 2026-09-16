# Conventional procedure benchmark

This benchmark evaluates the implemented procedure: fixed candidate settings,
successive historical refits, selection on observed AvE/CDR, and evaluation of
the frozen selected forecast against later terminal-development observations.
The paper behind it is [The Actuary and IBNR Techniques: A Machine Learning
Approach](https://ibnr.co/research/balona-richman-2021.pdf), the 23 April 2021
manuscript by Caesar Balona and Ronald Richman, published on https://ibnr.co,
an educational site by Ron Richman, one of the paper's authors and the founder
of insureAI. That site is unrelated to this package, which shares the name by
coincidence. The benchmark does not compare against ReserveAI, insureAI's
commercial reserving platform built on the paper's ideas: its methods are
proprietary, so they are neither implemented nor compared here.

## Reproduce

From a checkout with the core package and development environment installed:

```sh
.venv/Scripts/python.exe scripts/benchmark_conventional.py --published --output analysis/results/conventional/published
.venv/Scripts/python.exe scripts/benchmark_conventional.py --synthetic-seeds 30 --output analysis/results/conventional/synthetic
```

Each output directory contains `results.csv` and a `manifest.json` with package
version, seeds and SHA256 hashes of the implementation and benchmark scripts.
The published run also retains every candidate's eligibility and historical
mean score in `published_rankings.csv`. Candidate settings are serialized in
each result row. Failed intervals are ineligible under the library's strict
scoring rules; no partial-history mean can win.

Candidate batches bound memory. Selection is performed by the library within
each batch, then the global winner is the smallest of those batch minima, with
the same lexical tie rule. A test compares this to selecting the whole grid at
once. Batching does not change the candidate population or scoring window.

## Published examples

Source: [Balona and Richman (2021)](https://ibnr.co/research/balona-richman-2021.pdf),
Appendix C, via the [structured appendix](https://ibnr.co/research/appendix-data.json)
published on https://ibnr.co.
The loader downloads the bytes at runtime, caches outside the repository and
checks SHA256 `7333caea41fecc907dc0c98e46f38c64559cec2158cdacdb4bfc22f604dad455`.
The original dataset is not redistributed in the repository. The adapter's
constants record tables, pages, fields, grains and source notes.

| Case | Historical fit dates | Scored intervals | Selection and final fit | Terminal age | Evaluation date |
|---|---|---:|---|---:|---|
| Swiss private liability | 1984-1997 year ends | 13 | 1997-12-31 | 240 months | 2016-12-31 |
| Quarterly liability | 2012Q1-2014Q4 ends | 11 | 2014-12-31 | 63 months | 2019-12-31 |
| Quarterly property | 2012Q1-2014Q4 ends | 11 | 2014-12-31 | 63 months | 2019-12-31 |

For example, the final quarterly score compares a Q3 forecast with the Q4
outcome, then the chosen settings are refitted using Q4 information. The
selection date is Q4. Appendix coloring and the plotted origin populations
support this interpretation; it does not assign Q4 information to a Q3 decision.

**Literal table grids**, all volume weighted:

- Swiss: history windows 10-19, both high/low flags, BF loss ratios 0.50-0.70
  by 0.01, GCC decays 0-1 by 0.05. There are 40 CL, 840 BF and 840 GCC
  candidates. Selection is conducted separately within each family, as in the
  first case study.
- Quarterly: history windows 5-21 with the same exclusion flags, BF ratios
  0.40-0.60 by 0.01, and GCC decays 0-1 by 0.05. There are 2,924 candidates,
  selected jointly across methods.
- Basic comparisons use all available history, no exclusions, BF ratio 0.60
  and GCC decay 0.75. Exact ties among equivalent history windows use candidate
  names, not an inferred preference for a particular reported setting.
- Sparse factors explicitly use unity; extreme trimming is skipped if it would
  exhaust observations. On a complete run-off triangle the deepest link has one
  origin pair, so a candidate with a drop flag needs
  `exhausted_exclusions="keep"` to reach the end at all. These are the
  conventions disclosed on https://ibnr.co, not undocumented claims about the
  original chainladder package version.

### Source limitations

1. The prose and ranking tables give inconsistent candidate counts. In
   particular the quarterly paper reports 1,848, whereas the literal Cartesian
   grid contains 2,924. The benchmark does not remove candidates to force a
   reported count.
2. The appendix provides origin and development axes, not transaction-level
   availability timestamps. The adapter assigns corresponding calendar-period
   ends. Swiss AY1997 at 240 months maps to 2016, although the prose says
   development ends in 2015.
3. Swiss premiums were simulated **from eventual ultimate claims**. Assigning
   them historical booking dates reproduces an input convention; it does not
   establish that those features were independently available at the time.
   Quarterly premiums were normalized to an average 50% loss ratio.
4. The final fitted diagonal stops one age before the appendix endpoint, so
   reaching the declared horizon requires the explicit unity extension. The
   last published age is not verified as a fully settled economic ultimate.
5. Some worked CDR tables conflict with the stated equations; this code uses
   the signed CDR equation and Equation 2. Published GCC(0)/CL results differ
   despite equivalent displayed settings, and a BF baseline appears with two
   slightly different values. Numerical differences are retained, not tuned away.

## Independent synthetic portfolios

The generator does not import or call a reserving estimator. Every seed from
0 through 29 is retained under four prespecified scenarios. Each portfolio has
24 annual origins (2000-2023), eight development ages and exogenous premiums
booked at the first age. Losses combine a random portfolio development pattern,
origin-specific development, correlated origin severity and gamma process noise.
The complete distribution is documented in `scripts/conventional_synthetic.py`.

| Scenario | Change from common generator |
|---|---|
| Stable | Increment noise coefficient of variation 0.15 |
| Noisy | Increment noise coefficient of variation 0.50 |
| Drift | Origin severity grows 2.5% in log terms and calendar inflation 4% annually after 2010 |
| Shock | Future increments from 2021 increase by 40% |

Same-seed stable and shock portfolios have **identical information through
selection**. This tests an unanticipated change that historical selection has
no opportunity to discover. The benchmark neither selects favorable seeds nor
modifies the candidate grid after observing terminal losses.

Replay runs from 2011 through 2020 year ends (nine intervals); selection occurs
at 2020-12-31; terminal-age evaluation is at 2030-12-31. The 2021-2023 origins
are excluded from the frozen forecast population. Origins already terminal at
selection are excluded from later RMSE, leaving seven genuinely unknown targets
per portfolio. This prevents mature zero errors from diluting test performance.

The same 42-candidate grid is used in every scenario: windows all/3/5,
high exclusion off/on, no low exclusion, BF ratios 0.40/0.50/0.60 and GCC decays
0.25/0.75/1. CL is included for each factor configuration. Fixed all-history
CL, BF(0.60) and GCC(0.75) are reported alongside both selection rules.

Results are paired by seed within each scenario. Thirty independent portfolio
seeds provide an exploratory stress test, not precise evidence about rare tails
or an assurance that a historical winner will remain best after a regime change.
