"""Triangle -> standardized model data. The Stan ``data`` block is the contract.

``stan_data`` produces the dict consumed by the cross-classified lognormal
family (Meyers CRC/CCL/CSR). NumPyro and PyMC implementations consume the
IDENTICAL dict — no backend grows its own data prep.

Conventions:
- ``w``/``d`` are 1-based origin/dev indices (Stan style), sorted by (w, d).
- ``prev_idx[i]`` is the 1-based row index of the observation at
  (w[i]-1, d[i]), or 0 when w[i] == 1. Because rows are sorted by (w, d),
  prev_idx[i] < i+1 always, so mu can be built in one forward pass.
- ``logprem`` is per-observation; ``premium`` is per-origin (each origin's
  premium at its latest eval in the triangle, i.e. the booked value).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from ibnr.triangle.core import GRAIN_MONTHS, Triangle


def stan_data(
    triangle: Triangle,
    *,
    loss_field: str,
    premium_field: str | None = None,
) -> dict[str, Any]:
    """Map a single-cohort cumulative Triangle to the standardized data dict.

    The triangle must contain exactly one segment combination (one company x
    line); slice with ``triangle.filter`` first. Slice training data with
    ``triangle.as_of(...)`` before calling — this function uses every row.
    """
    if triangle.meta.measure != "cumulative":
        raise ValueError("stan_data requires a cumulative triangle")
    df = triangle.select_fields(loss_field).execute()
    if df.empty:
        raise ValueError(f"no rows for loss field {loss_field!r}")
    segs = triangle.segments
    if segs and len(df.drop_duplicates(segs)) > 1:
        raise ValueError(
            f"triangle has multiple segment combinations on {segs}; filter to one cohort first"
        )

    df = df.copy()
    df["origin_period"] = _as_date(df["origin_period"])
    df["eval_date"] = _as_date(df["eval_date"])
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    if (df["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")

    origins = sorted(df["origin_period"].unique())
    dev_steps = sorted((df["dev_lag"] // step).unique())
    if dev_steps[0] < 1:
        raise ValueError("dev_lag must be positive")
    n_w, n_d = len(origins), int(dev_steps[-1])
    w_of = {o: i + 1 for i, o in enumerate(origins)}

    df["w"] = df["origin_period"].map(w_of)
    df["d"] = (df["dev_lag"] // step).astype(int)
    if df.duplicated(["w", "d"]).any():
        raise ValueError(
            "multiple rows per (origin, dev) cell; slice with as_of()/latest_diagonal() first"
        )
    nonpos = df["value"] <= 0
    if nonpos.any():
        raise ValueError(
            f"{int(nonpos.sum())} cells have non-positive {loss_field!r}; "
            "lognormal models need positive losses (Meyers' selection excludes these)"
        )
    df = df.sort_values(["w", "d"]).reset_index(drop=True)

    row_of = {(int(r.w), int(r.d)): i + 1 for i, r in df.iterrows()}
    prev_idx = np.array(
        [row_of.get((int(r.w) - 1, int(r.d)), 0) if r.w > 1 else 0 for r in df.itertuples()],
        dtype=int,
    )

    data: dict[str, Any] = {
        "len_data": len(df),
        "n_w": n_w,
        "n_d": n_d,
        "w": df["w"].to_numpy(dtype=int),
        "d": df["d"].to_numpy(dtype=int),
        "prev_idx": prev_idx,
        "logloss": np.log(df["value"].to_numpy(dtype=float)),
        # metadata (not part of the Stan data block proper)
        "origin_periods": origins,
        "dev_grain_months": step,
        "loss": df["value"].to_numpy(dtype=float),
    }

    if premium_field is not None:
        premium = _premium_by_origin(triangle, premium_field, origins)
        data["premium"] = premium
        data["logprem"] = np.log(premium)[data["w"] - 1]
    return data


def _premium_by_origin(triangle: Triangle, premium_field: str, origins: list) -> np.ndarray:
    pdf = triangle.select_fields(premium_field).latest_diagonal().execute()
    if pdf.empty:
        raise ValueError(f"no rows for premium field {premium_field!r}")
    pdf = pdf.copy()
    pdf["origin_period"] = _as_date(pdf["origin_period"])
    by_origin = pdf.set_index("origin_period")["value"]
    missing = [o for o in origins if o not in by_origin.index]
    if missing:
        raise ValueError(f"premium missing for origins {missing}")
    premium = by_origin.loc[origins].to_numpy(dtype=float)
    if (premium <= 0).any():
        raise ValueError("non-positive premium; lognormal exposure models need positive premium")
    return premium


def ccl_mu_index(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Static arrays that turn the CCL ``mu`` recurrence into one matmul.

    ``model.stan`` builds mu with a forward recurrence over ``prev_idx``:

        mu[w, d] = base[w, d] + rho * (logloss[w-1, d] - mu[w-1, d])   (w > 1)

    where base[w, d] = logprem + logelr + alpha[w] + beta[d]. Within a fixed
    dev column this linear recurrence has the exact closed form

        mu[w, d] = sum_{k=1..w} (-rho)^(w-k) * B[k, d],
        B[k, d]  = base[k, d] + rho * logloss[k-1, d]   (B[1, d] = base[1, d])

    so mu = P(rho) @ B with P[i, j] = (-rho)^(w_i - w_j) for cells j in the same
    column as i with w_j <= w_i (else 0). ``P`` is the only rho-dependent piece;
    everything here is data, precomputed once. This form keeps the NumPyro/PyMC
    autodiff graphs tiny (an N x N matmul) instead of an N-deep scalar chain —
    the PyTensor C-compile of the unrolled chain is the dominant cost otherwise.
    Returns ``expo`` (N x N, the exponents w_i - w_j), ``colmask`` (N x N, 1.0
    where j contributes to i), and ``logloss_prev`` (N, the observed previous-
    origin loss, 0 where absent).
    """
    w = np.asarray(data["w"], dtype=int)
    d = np.asarray(data["d"], dtype=int)
    prev0 = np.asarray(data["prev_idx"], dtype=int) - 1  # -1 = no previous cell
    logloss = np.asarray(data["logloss"], dtype=float)

    same_col = (d[:, None] == d[None, :]) & (w[None, :] <= w[:, None])
    expo = np.where(same_col, w[:, None] - w[None, :], 0).astype(int)
    colmask = same_col.astype(float)
    logloss_prev = np.where(prev0 >= 0, logloss[np.where(prev0 >= 0, prev0, 0)], 0.0)
    return {"expo": expo, "colmask": colmask, "logloss_prev": logloss_prev}


def realized_values(
    triangle: Triangle,
    *,
    loss_field: str,
    dev_lag: int,
    origins: list[dt.date],
) -> np.ndarray:
    """Realized cumulative losses at ``dev_lag`` months for the given origins,
    taken from the FULL (unsliced) triangle — the scoring targets for
    backtests. NaN where the outcome is not (yet) observed."""
    df = triangle.select_fields(loss_field).execute()
    df = df[df["dev_lag"] == dev_lag].copy()
    df["origin_period"] = _as_date(df["origin_period"])
    by_origin = df.set_index("origin_period")["value"]
    return np.array([float(by_origin.get(o, np.nan)) for o in origins])


def _as_date(series):
    if str(series.dtype).startswith("datetime64"):
        return series.dt.date
    return series
