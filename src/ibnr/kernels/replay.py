"""Observed AvE/CDR replay of conventional candidates at successive dates.

This measures realized forecast revisions, not a simulated one-year risk
distribution. The ``cdr`` column is the change in a candidate's fitted ultimate
over ONE DEVELOPMENT PERIOD, which is one year only on an annual grain, and it
uses the paper's adverse-positive convention (new ultimate minus old ultimate),
opposite the favorable-positive convention in ``kernels.cdr``. That module's
one-year CDR is a different measurement: a distribution of where next year's
re-estimate could land, rather than one observed number per origin.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from ibnr.kernels.contract import cohort_grid, premium_by_origin
from ibnr.kernels.conventional import (
    ConventionalCandidate,
    ConventionalFit,
    _fit_grid,
    as_date,
)
from ibnr.triangle import Triangle
from ibnr.triangle.core import GRAIN_MONTHS

CELL_COLUMNS = [
    "candidate",
    "as_of",
    "eval_date",
    "origin_period",
    "from_dev_lag",
    "to_dev_lag",
    "actual_increment",
    "expected_increment",
    "ave",
    "old_ultimate",
    "new_ultimate",
    "cdr",
    "remaining_revision",
]


@dataclass(frozen=True)
class ConventionalReplay:
    """Historical fits, origin-level outcomes, and explicit failures/exclusions.

    ``fits`` is keyed by (candidate name, information date). A failed candidate
    interval has no cell rows, and a reason in ``errors`` when recording errors.
    This prevents partial origin coverage from looking like a complete score.
    """

    candidates: dict[str, ConventionalCandidate]
    dates: tuple[dt.date, ...]
    fits: dict[tuple[str, dt.date], ConventionalFit]
    cells: pd.DataFrame
    errors: pd.DataFrame
    exclusions: pd.DataFrame
    loss_field: str
    premium_field: str
    units: str | None
    segment: dict


def replay_conventional(
    triangle: Triangle,
    candidates: Mapping[str, ConventionalCandidate],
    dates: Sequence[dt.date | str],
    *,
    loss_field: str = "paid_loss",
    premium_field: str = "earned_premium",
    on_error: str = "raise",
) -> ConventionalReplay:
    """Refit fixed candidates, then measure each successive observed diagonal.

    Dates must be strictly increasing, one development period apart. Every
    candidate must declare the same final development ``horizon``. The same
    cohort, field, units, and previous-origin population are used throughout.
    ``on_error='record'`` retains interval-level failures for later selection
    to disqualify; it never silently scores a surviving subset of origins.
    """
    specs = dict(candidates)
    cutoffs = tuple(as_date(d) for d in dates)
    if not specs or any(not isinstance(n, str) or not n for n in specs):
        raise ValueError("candidates must have nonempty string names")
    if any(not isinstance(c, ConventionalCandidate) for c in specs.values()):
        raise ValueError("candidates must contain ConventionalCandidate settings")
    horizons = {c.horizon for c in specs.values()}
    if None in horizons or len(horizons) != 1:
        raise ValueError("replay candidates must have the same explicit horizon")
    if len(cutoffs) < 2 or any(b <= a for a, b in pairwise(cutoffs)):
        raise ValueError("dates must be strictly increasing and contain at least two cutoffs")
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    if triangle.meta.origin_grain != triangle.meta.dev_grain:
        raise ValueError("replay requires matching origin and development grains")
    for a, b in pairwise(cutoffs):
        expected = pd.Timestamp(a) + pd.DateOffset(months=step)
        if pd.Timestamp(a).is_month_end:
            expected += pd.offsets.MonthEnd(0)
        if b != expected.date():
            raise ValueError("successive dates must be one development period apart")

    fits, fit_errors, grids, segment = {}, {}, {}, {}
    need_premium = any(c.method != "cl" for c in specs.values())
    # Materialize each historical grid once, independent of the candidate count.
    # Both data and premiums are cut BEFORE fitting, including restatements.
    for cutoff in cutoffs:
        train = triangle.as_of(cutoff)
        try:
            grid = cohort_grid(train, loss_field=loss_field)
        except ValueError as exc:
            for name in specs:
                fit_errors[name, cutoff] = str(exc)
            continue
        grids[cutoff] = grid
        if segment and segment != grid["segment"]:
            raise ValueError("cohort identity changed between replay dates")
        segment = grid["segment"]
        premium_error = None
        if need_premium:
            try:
                premium = premium_by_origin(
                    train, premium_field, grid["origin_periods"], segment=segment
                )
            except ValueError as exc:
                premium_error = str(exc)
        for name, spec in specs.items():
            try:
                if spec.method != "cl":
                    if premium_error:
                        raise ValueError(premium_error)
                    fit_grid = {**grid, "premium": premium}
                else:
                    fit_grid = grid
                fits[name, cutoff] = _fit_grid(fit_grid, spec, cutoff)
            except ValueError as exc:
                fit_errors[name, cutoff] = str(exc)

    cells, errors, exclusions = [], [], []
    for before, after in pairwise(cutoffs):
        if before in grids and after in grids:
            new_origins = set(grids[after]["origin_periods"]) - set(grids[before]["origin_periods"])
            exclusions.extend(
                {"as_of": before, "eval_date": after, "origin_period": o, "reason": "new_origin"}
                for o in sorted(new_origins)
            )
        for name in specs:
            try:
                for date in (before, after):
                    if (name, date) in fit_errors:
                        raise ValueError(f"fit at {date}: {fit_errors[name, date]}")
                rows = _interval(name, fits[name, before], fits[name, after])
                cells.extend(rows)
            except ValueError as exc:
                if on_error == "raise":
                    raise ValueError(f"candidate {name!r}, {before} to {after}: {exc}") from exc
                errors.append(
                    {"candidate": name, "as_of": before, "eval_date": after, "reason": str(exc)}
                )
    return ConventionalReplay(
        specs,
        cutoffs,
        fits,
        pd.DataFrame(cells, columns=CELL_COLUMNS),
        pd.DataFrame(errors, columns=["candidate", "as_of", "eval_date", "reason"]),
        pd.DataFrame(exclusions, columns=["as_of", "eval_date", "origin_period", "reason"]),
        loss_field,
        premium_field,
        triangle.meta.units,
        segment,
    )


def _interval(name: str, old: ConventionalFit, new: ConventionalFit) -> list[dict]:
    before = old.origins.set_index("origin_period")
    after = new.origins.set_index("origin_period")
    if not before.index.isin(after.index).all():
        raise ValueError("a prior origin is missing from the later information set")
    step = old.grid["dev_grain_months"]
    horizon = old.candidate.horizon
    rows = []
    for origin, previous in before.iterrows():
        following = after.loc[origin]
        next_lag = min(int(previous["latest_dev_lag"]) + step, horizon)
        if following["latest_dev_lag"] != next_lag:
            raise ValueError(
                f"origin {origin} does not observe exactly the next development age {next_lag}"
            )
        predicted_cumulative = old.predict_cumulative(origin, next_lag)
        actual = float(following["latest"] - previous["latest"])
        expected = float(predicted_cumulative - previous["latest"])
        ave = actual - expected
        cdr = float(following["ultimate"] - previous["ultimate"])
        # Calculate independently of CDR-AvE so this is an inspectable identity:
        # new remaining reserve minus old expected reserve after this diagonal.
        remaining_revision = float(
            following["reserve"] - (previous["ultimate"] - predicted_cumulative)
        )
        values = [actual, expected, ave, cdr, remaining_revision]
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite replay amounts for origin {origin}")
        rows.append(
            {
                "candidate": name,
                "as_of": old.as_of,
                "eval_date": new.as_of,
                "origin_period": origin,
                "from_dev_lag": int(previous["latest_dev_lag"]),
                "to_dev_lag": next_lag,
                "actual_increment": actual,
                "expected_increment": expected,
                "ave": ave,
                "old_ultimate": float(previous["ultimate"]),
                "new_ultimate": float(following["ultimate"]),
                "cdr": cdr,
                "remaining_revision": remaining_revision,
            }
        )
    return rows
