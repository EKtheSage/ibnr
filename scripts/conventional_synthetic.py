"""Independent synthetic portfolios for conventional procedure experiments.

These choices are fixed before benchmarking; the generator never imports a
reserving estimator, searches candidates, or selects a favorable seed. This is
a stress experiment, not a reproduction of the paper's empirical datasets.

Geometry and evaluation protocol
-------------------------------
* 24 annual origins, 2000--2023, each observed through eight annual ages. The
  last observation is December 2030. Amounts are synthetic USD, undiscounted.
* Replay dates: December 31, 2011--2020 (nine observed one-year intervals).
  Select at December 2020; evaluate frozen age-96 forecasts at December 2030.
  The 2021--2023 origins exist in the full data but were unknown at selection.
* Earned premium is booked once, at age 12 in the origin year. It has a
  $1 million median starting level, lognormal standard deviation 0.35, and
  2% nominal growth per origin year. Premium is exogenous to all scenarios.

Common latent structure
-----------------------
* Baseline expected loss ratio 0.65. Log severity has a stationary Gaussian
  AR(1) origin effect, correlation 0.40 and marginal standard deviation 0.15;
  subtracting half its variance gives a unit-mean multiplicative effect.
* A portfolio development pattern is drawn from Dirichlet(80*p), where
  p=(.12,.24,.21,.16,.11,.08,.05,.03). Each origin independently draws its own
  pattern from Dirichlet(120*portfolio_pattern). Thus different origins have
  different emergence, rather than one exact chain-ladder development curve.
* Incremental amounts are the expected ultimate times that origin's pattern,
  multiplied by independent mean-one gamma noise. All increments and premiums
  are positive; cumulative losses are their sums. The age 96 value is the
  realized terminal amount, not a noise-free latent expectation.

Prespecified scenarios
---------------------
stable: common structure, gamma coefficient of variation 0.15.
noisy:  common structure, gamma coefficient of variation 0.50.
drift:  stable noise, plus exp(.025*max(origin_year-2010,0)) severity growth
        and 1.04**max(calendar_year-2010,0) incremental calendar inflation.
shock:  stable noise, with all increments in calendar 2021 and later multiplied
        by 1.40. The shock is unanticipated at the selection date.

For the same seed, stable/drift/shock share every underlying random draw.
In particular stable and shock have identical histories through 2020. The
different scenarios change loss experience without changing booked premiums.
No scenario promises that historical selection will beat a fixed baseline.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ibnr import Triangle

SCENARIOS = ("stable", "noisy", "drift", "shock")
START_YEAR = 2000
N_ORIGINS = 24
N_DEVELOPMENT = 8
HORIZON = 12 * N_DEVELOPMENT
REPLAY_DATES = tuple(dt.date(year, 12, 31) for year in range(2011, 2021))
SELECTION_DATE = dt.date(2020, 12, 31)
EVALUATION_DATE = dt.date(2030, 12, 31)
BASE_DEVELOPMENT = (0.12, 0.24, 0.21, 0.16, 0.11, 0.08, 0.05, 0.03)


def synthetic_portfolio(seed: int, scenario: str, backend: str = "duckdb") -> Triangle:
    """Return one fully emerged, long-format portfolio under a fixed scenario.

    Use ``as_of`` to expose historical information. The seed determines the
    portfolio and process draws; no model or outcome enters their construction.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    origins = START_YEAR + np.arange(N_ORIGINS)
    calendar_year = origins[:, None] + np.arange(N_DEVELOPMENT)[None, :]

    premium = (
        1_000_000
        * 1.02 ** np.arange(N_ORIGINS)
        * rng.lognormal(mean=0.0, sigma=0.35, size=N_ORIGINS)
    )
    portfolio_pattern = rng.dirichlet(80 * np.asarray(BASE_DEVELOPMENT))
    origin_pattern = rng.dirichlet(120 * portfolio_pattern, size=N_ORIGINS)

    severity = np.empty(N_ORIGINS)
    severity[0] = rng.normal(scale=0.15)
    innovations = rng.normal(scale=0.15 * np.sqrt(1 - 0.4**2), size=N_ORIGINS - 1)
    for i, innovation in enumerate(innovations, start=1):
        severity[i] = 0.4 * severity[i - 1] + innovation
    mean_ultimate = premium * 0.65 * np.exp(severity - 0.5 * 0.15**2)

    cv = 0.50 if scenario == "noisy" else 0.15
    process = rng.gamma(shape=1 / cv**2, scale=cv**2, size=(N_ORIGINS, N_DEVELOPMENT))
    increments = mean_ultimate[:, None] * origin_pattern * process
    if scenario == "drift":
        increments *= np.exp(0.025 * np.maximum(origins - 2010, 0))[:, None]
        increments *= 1.04 ** np.maximum(calendar_year - 2010, 0)
    elif scenario == "shock":
        increments *= np.where(calendar_year >= 2021, 1.40, 1.0)
    cumulative = np.cumsum(increments, axis=1)

    rows = []
    for i, year in enumerate(origins):
        origin = dt.date(int(year), 1, 1)
        rows.append(
            {
                "portfolio": "synthetic",
                "origin_period": origin,
                "dev_lag": 12,
                "eval_date": dt.date(int(year), 12, 31),
                "field": "earned_premium",
                "value": float(premium[i]),
            }
        )
        for j in range(N_DEVELOPMENT):
            rows.append(
                {
                    "portfolio": "synthetic",
                    "origin_period": origin,
                    "dev_lag": 12 * (j + 1),
                    "eval_date": dt.date(int(calendar_year[i, j]), 12, 31),
                    "field": "paid_loss",
                    "value": float(cumulative[i, j]),
                }
            )
    return Triangle.from_long(
        pd.DataFrame(rows),
        measure="cumulative",
        origin_grain="Y",
        dev_grain="Y",
        units="USD",
        backend=backend,
    )
