"""Triangle -> standardized multi-line model data.

The multiline analogue of ``contract.py``: one canonical mapping from a
one-company, many-lines-of-business cumulative Triangle to dense arrays,
consumed by the statistical dependence models (SUR, copula). Both entries
share this prep — neither grows its own.

Conventions:
- ``cum`` is (n_lob, n_w, n_d) with NaN at unobserved cells; dev step ``d``
  lives at index d-1 (1-based steps, as in ``stan_data``).
- All lines must share the same origin set and the same observed-cell mask:
  cross-line dependence is estimated cell-by-cell, so misaligned triangles
  are a hard error, not something to silently intersect.
- The predictive target layout is fixed: per-(lob, origin) ultimates in
  lob-major order, then one total per lob, then the grand total — so
  cross-line diversification is visible in the samples.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from ibnr.kernels.contract import _as_date
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import GRAIN_MONTHS, Triangle

LOB_COLUMN = "line_of_business"


def multiline_data(
    triangle: Triangle,
    *,
    loss_field: str,
    premium_field: str | None = None,
    lob_column: str = LOB_COLUMN,
) -> dict[str, Any]:
    """Map a one-company multi-LOB cumulative Triangle to dense arrays.

    Slice training data with ``triangle.as_of(...)`` before calling — this
    function uses every row.
    """
    if triangle.meta.measure != "cumulative":
        raise ValueError("multiline_data requires a cumulative triangle")
    df = triangle.select_fields(loss_field).execute()
    if df.empty:
        raise ValueError(f"no rows for loss field {loss_field!r}")
    segs = triangle.segments
    if lob_column not in segs:
        raise ValueError(f"triangle has no {lob_column!r} segment column; segments: {segs}")
    for col in (c for c in segs if c != lob_column):
        values = df[col].unique()
        if len(values) > 1:
            raise ValueError(
                f"multiple {col!r} values {sorted(map(str, values))}; "
                "multiline models fit one company at a time — filter first"
            )

    df = df.copy()
    df["origin_period"] = _as_date(df["origin_period"])
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    if (df["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")
    df["d"] = (df["dev_lag"] // step).astype(int)
    if (df["d"] < 1).any():
        raise ValueError("dev_lag must be positive")
    if df.duplicated([lob_column, "origin_period", "d"]).any():
        raise ValueError(
            "multiple rows per (lob, origin, dev) cell; slice with as_of()/latest_diagonal() first"
        )

    lobs = sorted(df[lob_column].unique())
    if len(lobs) < 2:
        raise ValueError(f"multiline models need >= 2 lines of business, got {lobs}")
    origins = sorted(df["origin_period"].unique())
    n_lob, n_w, n_d = len(lobs), len(origins), int(df["d"].max())

    k_of = {lob: k for k, lob in enumerate(lobs)}
    w_of = {o: w for w, o in enumerate(origins)}
    cum = np.full((n_lob, n_w, n_d), np.nan)
    for row in df.itertuples():
        lob = getattr(row, lob_column)
        cum[k_of[lob], w_of[row.origin_period], int(row.d) - 1] = float(row.value)
    obs_mask = ~np.isnan(cum)

    misaligned = [lobs[k] for k in range(1, n_lob) if not (obs_mask[k] == obs_mask[0]).all()]
    if misaligned:
        raise ValueError(
            f"lines {misaligned} have a different observed-cell pattern than {lobs[0]!r}; "
            "cell-wise dependence models need aligned triangles"
        )

    data: dict[str, Any] = {
        "n_lob": n_lob,
        "n_w": n_w,
        "n_d": n_d,
        "lobs": lobs,
        "cum": cum,
        "obs_mask": obs_mask,
        "origin_periods": origins,
        "dev_grain_months": step,
        "lob_column": lob_column,
        "units": triangle.meta.units,
    }
    if premium_field is not None:
        data["premium"] = _premium_by_lob_origin(triangle, premium_field, lobs, origins, lob_column)
    return data


def _premium_by_lob_origin(
    triangle: Triangle,
    premium_field: str,
    lobs: list,
    origins: list,
    lob_column: str,
) -> np.ndarray:
    pdf = triangle.select_fields(premium_field).latest_diagonal().execute()
    if pdf.empty:
        raise ValueError(f"no rows for premium field {premium_field!r}")
    pdf = pdf.copy()
    pdf["origin_period"] = _as_date(pdf["origin_period"])
    by_key = pdf.set_index([lob_column, "origin_period"])["value"]
    premium = np.full((len(lobs), len(origins)), np.nan)
    for k, lob in enumerate(lobs):
        for w, origin in enumerate(origins):
            if (lob, origin) in by_key.index:
                premium[k, w] = float(by_key.loc[(lob, origin)])
    if np.isnan(premium).any():
        missing = [
            (lobs[k], origins[w].isoformat())
            for k, w in zip(*np.nonzero(np.isnan(premium)), strict=True)
        ]
        raise ValueError(f"premium missing for (lob, origin) pairs {missing}")
    if (premium <= 0).any():
        raise ValueError("non-positive premium; exposure-scaled models need positive premium")
    return premium


def realized_multiline(
    full_triangle: Triangle,
    *,
    loss_field: str,
    dev_lag: int,
    lobs: list,
    origins: list[dt.date],
    lob_column: str = LOB_COLUMN,
) -> np.ndarray:
    """Realized cumulative losses at ``dev_lag`` months, (n_lob, n_w), from the
    FULL (unsliced) triangle. NaN where the outcome is not (yet) observed."""
    df = full_triangle.select_fields(loss_field).execute()
    df = df[df["dev_lag"] == dev_lag].copy()
    df["origin_period"] = _as_date(df["origin_period"])
    by_key = df.set_index([lob_column, "origin_period"])["value"]
    out = np.full((len(lobs), len(origins)), np.nan)
    for k, lob in enumerate(lobs):
        for w, origin in enumerate(origins):
            if (lob, origin) in by_key.index:
                out[k, w] = float(by_key.loc[(lob, origin)])
    return out


def multiline_targets(
    lobs: list,
    origins: list[dt.date],
    premium: np.ndarray | None = None,
    lob_column: str = LOB_COLUMN,
) -> pd.DataFrame:
    """Target metadata matching ``flatten_with_totals`` column order:
    per-(lob, origin) rows lob-major, then per-lob totals, then grand total."""
    origin_labels = _origin_labels(origins)
    rows = []
    for k, lob in enumerate(lobs):
        for w, origin in enumerate(origins):
            rows.append(
                {
                    "label": f"{lob}/{origin_labels[w]}",
                    lob_column: lob,
                    "origin_period": origin,
                    "premium": float(premium[k, w]) if premium is not None else np.nan,
                }
            )
    for lob in lobs:
        rows.append(
            {"label": f"{lob}/total", lob_column: lob, "origin_period": None, "premium": np.nan}
        )
    rows.append({"label": "total", lob_column: None, "origin_period": None, "premium": np.nan})
    return pd.DataFrame(rows)


def flatten_with_totals(arr: np.ndarray) -> np.ndarray:
    """(..., n_lob, n_w) -> (..., n_lob*n_w + n_lob + 1): per-cell values
    lob-major, per-lob sums, grand sum. Works for realized values (2-D) and
    sample stacks (3-D) alike; NaN outcomes propagate into their sums."""
    per_cell = arr.reshape(*arr.shape[:-2], -1)
    lob_totals = arr.sum(axis=-1)
    grand = lob_totals.sum(axis=-1, keepdims=True)
    return np.concatenate([per_cell, lob_totals, grand], axis=-1)


def assemble_predictive(
    ults: np.ndarray,
    targets: pd.DataFrame,
    units: str | None = None,
) -> PredictiveDistribution:
    """Build the PredictiveDistribution from per-(lob, origin) ultimate draws.

    ults: (n_draws, n_lob, n_w) — totals are derived here so they are always
    row-sums of the same draws (diversification stays coherent)."""
    if ults.ndim != 3:
        raise ValueError(f"ults must be (n_draws, n_lob, n_w), got {ults.shape}")
    samples = flatten_with_totals(ults)
    if samples.shape[1] != len(targets):
        raise ValueError(
            f"{samples.shape[1]} sample columns but {len(targets)} target rows; "
            "build targets with multiline_targets(lobs, origins)"
        )
    return PredictiveDistribution(samples=samples, targets=targets, units=units)


def _origin_labels(origins: list[dt.date]) -> list[str]:
    years = [o.year for o in origins]
    if len(set(years)) == len(years):
        return [str(y) for y in years]
    return [o.isoformat() for o in origins]
