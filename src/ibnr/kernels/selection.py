"""Historical conventional selection followed by untouched later evaluation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from ibnr.kernels.conventional import ConventionalCandidate, ConventionalFit, _date
from ibnr.kernels.replay import ConventionalReplay
from ibnr.triangle import Triangle
from ibnr.triangle.core import GRAIN_MONTHS


@dataclass(frozen=True)
class ConventionalSelection:
    """A decision based solely on replay outcomes available by ``as_of``."""

    name: str
    candidate: ConventionalCandidate
    as_of: dt.date
    metric: str
    fit: ConventionalFit
    ranking: pd.DataFrame
    scores: pd.DataFrame
    loss_field: str
    premium_field: str
    units: str | None
    segment: dict


@dataclass(frozen=True)
class ConventionalEvaluation:
    """Frozen forecasts against later terminal-age observations, in input units.

    Terminal age is the declared horizon; these observations need not be final
    economic ultimates. ``summary['rmse']`` is defined only with full coverage
    of origins whose terminal value was unknown at the selection date.
    """

    origins: pd.DataFrame
    summary: dict
    selection_as_of: dt.date
    evaluation_as_of: dt.date


def _weighted_rmse(error: np.ndarray, weight: np.ndarray) -> float:
    # Normalize in two stages to avoid squaring large currency amounts or
    # overflowing a sum of otherwise finite weights.
    positive = weight > 0
    error, weight = error[positive], weight[positive]
    scale = np.max(np.abs(error))
    if scale == 0:
        return 0.0
    normalized = weight / np.max(weight)
    normalized /= normalized.sum()
    return float(scale * np.sqrt(np.sum(normalized * (error / scale) ** 2)))


def score_replay(
    replay: ConventionalReplay,
    *,
    metric: str = "ave",
    through: dt.date | str | None = None,
) -> pd.DataFrame:
    """Equation 2: absolute-actual-weighted RMSE separately on each diagonal.

    Zero actual movements have zero weight. An all-zero diagonal is undefined,
    as is a failed or incomplete candidate interval. All are retained as rows
    with a status and reason, never converted to zero errors.
    """
    if metric not in ("ave", "cdr"):
        raise ValueError("metric must be 'ave' or 'cdr'")
    cutoff = replay.dates[-1] if through is None else _date(through)
    rows = []
    for before, after in pairwise(replay.dates):
        if after > cutoff:
            continue
        for name in replay.candidates:
            cells = replay.cells.loc[
                (replay.cells["candidate"] == name)
                & (replay.cells["as_of"] == before)
                & (replay.cells["eval_date"] == after)
            ]
            failures = replay.errors.loc[
                (replay.errors["candidate"] == name)
                & (replay.errors["as_of"] == before)
                & (replay.errors["eval_date"] == after)
            ]
            status, reason, rmse = "ok", "", np.nan
            fit = replay.fits.get((name, before))
            expected_origins = set() if fit is None else set(fit.origins["origin_period"])
            errors = cells[metric].to_numpy(dtype=float)
            weights = np.abs(cells["actual_increment"].to_numpy(dtype=float))
            weight_sum = float(weights.sum())
            if not failures.empty:
                status, reason = "failed", "; ".join(failures["reason"].astype(str))
            elif (
                cells.empty
                or set(cells["origin_period"]) != expected_origins
                or cells["origin_period"].duplicated().any()
            ):
                status, reason = "incomplete", "missing, duplicate, or unexpected origin outcomes"
            elif not np.isfinite(errors).all() or not np.isfinite(weights).all():
                status, reason = "nonfinite", "non-finite outcome or forecast error"
            elif not np.any(weights > 0):
                status, reason = (
                    "zero_weight",
                    "all actual movements are zero; weighted RMSE undefined",
                )
            else:
                rmse = _weighted_rmse(errors, weights)
            rows.append(
                {
                    "candidate": name,
                    "as_of": before,
                    "eval_date": after,
                    "metric": metric,
                    "n_cells": len(cells),
                    "weight_sum": weight_sum,
                    "rmse": rmse,
                    "status": status,
                    "reason": reason,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "candidate",
            "as_of",
            "eval_date",
            "metric",
            "n_cells",
            "weight_sum",
            "rmse",
            "status",
            "reason",
        ],
    )


def select_conventional(
    replay: ConventionalReplay,
    *,
    selection_as_of: dt.date | str,
    metric: str = "ave",
) -> ConventionalSelection:
    """Choose the lowest mean diagonal RMSE over a complete historical window.

    The history is all replay intervals ending on/before the selection cutoff.
    Choose a shorter window by supplying a shorter replay. One missing or
    undefined interval disqualifies that candidate; ties use lexical names.
    No eligible candidate is an error, with no fallback to incomplete scores.
    """
    cutoff = _date(selection_as_of)
    if cutoff not in replay.dates:
        raise ValueError("selection_as_of must be a replay information date")
    scores = score_replay(replay, metric=metric, through=cutoff)
    if scores.empty:
        raise ValueError("selection needs at least one outcome interval available by its cutoff")
    rows = []
    for name in replay.candidates:
        group = scores.loc[scores["candidate"] == name]
        ok = group["status"] == "ok"
        has_fit = (name, cutoff) in replay.fits
        eligible = bool(ok.all() and has_fit)
        reason = "; ".join(dict.fromkeys(group.loc[~ok, "reason"].astype(str)))
        if not has_fit:
            reason = (reason + "; no fit at the selection date").lstrip("; ")
        mean_rmse = np.nan
        if eligible:
            values = group["rmse"].to_numpy(dtype=float)
            scale = values.max()
            mean_rmse = float(scale * np.mean(values / scale)) if scale else 0.0
        rows.append(
            {
                "candidate": name,
                "eligible": eligible,
                "n_scored": int(ok.sum()),
                "n_required": len(group),
                "mean_rmse": mean_rmse,
                "reason": reason,
            }
        )
    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            ["eligible", "mean_rmse", "candidate"], ascending=[False, True, True], kind="stable"
        )
        .reset_index(drop=True)
    )
    if not ranking["eligible"].any():
        reasons = "; ".join(f"{r['candidate']}: {r['reason']}" for r in rows)
        raise ValueError(f"no eligible candidate with complete historical scores: {reasons}")
    winner = ranking.iloc[0]["candidate"]
    return ConventionalSelection(
        winner,
        replay.candidates[winner],
        cutoff,
        metric,
        replay.fits[winner, cutoff],
        ranking,
        scores,
        replay.loss_field,
        replay.premium_field,
        replay.units,
        dict(replay.segment),
    )


def evaluate_conventional(
    selection: ConventionalSelection,
    triangle: Triangle,
    *,
    as_of: dt.date | str,
) -> ConventionalEvaluation:
    """Evaluate the frozen selected forecast at its declared terminal age.

    Later data are sliced at ``as_of`` and never used to refit or reselect. New
    origins and terminal values already known at selection are not test targets.
    Missing targets remain visible and suppress the overall RMSE.
    """
    cutoff = _date(as_of)
    if cutoff <= selection.as_of:
        raise ValueError("evaluation as_of must be later than the selection date")
    if triangle.meta.units != selection.units or triangle.meta.measure != "cumulative":
        raise ValueError("evaluation units and cumulative measure must match the selection")
    if (
        GRAIN_MONTHS[triangle.meta.dev_grain] != selection.fit.grid["dev_grain_months"]
        or triangle.meta.origin_grain != triangle.meta.dev_grain
    ):
        raise ValueError("evaluation grains must match the selection")
    if set(triangle.segments) != set(selection.segment):
        raise ValueError("evaluation segment columns must match the selected cohort")
    data = triangle.as_of(cutoff).select_fields(selection.loss_field).execute().copy()
    if data.empty:
        raise ValueError("evaluation has no observations of the selected loss field")
    for name, value in selection.segment.items():
        if not data[name].eq(value).all():
            raise ValueError("evaluation data must contain exactly the selected cohort")
    data["origin_period"] = pd.to_datetime(data["origin_period"]).dt.date
    data["eval_date"] = pd.to_datetime(data["eval_date"]).dt.date
    horizon = selection.candidate.horizon
    # Check the earlier snapshot too: a later restatement cannot turn an
    # already available terminal outcome into a new test observation.
    earlier = triangle.as_of(selection.as_of).select_fields(selection.loss_field).execute()
    earlier_terminal = set(
        pd.to_datetime(earlier.loc[earlier["dev_lag"] == horizon, "origin_period"]).dt.date
    )
    terminal = data.loc[data["dev_lag"] == horizon]
    if terminal["origin_period"].duplicated().any():
        raise ValueError("duplicate terminal-age outcomes")
    observed = terminal.set_index("origin_period")["value"]
    observed_dates = terminal.set_index("origin_period")["eval_date"]
    rows = []
    for _, row in selection.fit.origins.iterrows():
        origin = row["origin_period"]
        known = bool(row["latest_dev_lag"] == horizon)
        actual = float(observed.get(origin, np.nan))
        observed_at = observed_dates.get(origin)
        if not known and origin in earlier_terminal:
            raise ValueError(
                "terminal outcome was available by selection; evaluation history differs from fit"
            )
        available = np.isfinite(actual)
        rows.append(
            {
                "origin_period": origin,
                "predicted": float(row["ultimate"]),
                "observed": actual,
                "observed_at": observed_at,
                "error": actual - float(row["ultimate"]) if available else np.nan,
                "known_at_selection": known,
                "included": not known and available,
                "status": "known_at_selection"
                if known
                else "ok"
                if available
                else "missing_terminal",
            }
        )
    origins = pd.DataFrame(rows)
    targets = origins.loc[~origins["known_at_selection"]]
    n_observed = int(targets["included"].sum())
    if targets.empty:
        rmse, reason = np.nan, "no unknown terminal-age targets at selection"
    elif n_observed != len(targets):
        rmse, reason = np.nan, "missing terminal-age outcomes; full target coverage required"
    else:
        rmse = _weighted_rmse(targets["error"].to_numpy(), np.ones(len(targets)))
        reason = ""
    return ConventionalEvaluation(
        origins,
        {"n_targets": len(targets), "n_observed": n_observed, "rmse": rmse, "reason": reason},
        selection.as_of,
        cutoff,
    )
