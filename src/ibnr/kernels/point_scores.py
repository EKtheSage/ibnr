"""Point-error metrics, implemented once, with the aggregation level explicit.

Every function here scores a POINT forecast against a realized value. Nothing
here touches draws; CRPS and PIT live in ``kernels.scores`` and
``kernels.predictive``.

The one idea to carry away: errors are summed WITHIN a level before the
absolute value is taken. On company 671 of the 2026-09-20 reconciliation the
four line errors were +15.03, +327.71, -854.36 and +501.97 (USD thousands): the
company error is their signed sum, -9.65, while the sum of their absolute
values is 1699.07. A company-level Pool_APE therefore credits a model for
overestimating one line and underestimating another, and a line-level one does
not. Neither is wrong; they answer different questions, which is why
:func:`level_errors` takes the level as an argument and names it in its output.

Definitions (``e = predicted - actual`` per unit of the chosen level):

- ``mae``       mean |e|
- ``rmse``      sqrt(mean e^2)
- ``wrmse``     sqrt(sum(actual * e^2) / sum(actual)) - the actual-weighted form
- ``mape``      mean |e / actual| over units with a nonzero actual
- ``medape``    median |e / actual| over the same units
- ``pool_ape``  sum |e| / sum actual - the R study's headline
- ``pool_pe``   sum e / sum actual - the signed version, a bias measure
- ``prop_over`` share of units with e > 0

A missing forecast is refused, never dropped: on a fixed cohort set a model that
cannot forecast a unit must not score on a smaller set than its neighbours.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["level_errors", "point_metrics", "shrink_toward"]


def point_metrics(predicted, actual) -> dict[str, float]:
    """Point-error metrics over aligned units of one level. See the module docstring."""
    p = np.asarray(predicted, dtype=float).reshape(-1)
    a = np.asarray(actual, dtype=float).reshape(-1)
    if p.shape != a.shape:
        raise ValueError(
            f"predicted and actual must have the same length, got {p.size} and {a.size}"
        )
    if p.size == 0:
        raise ValueError("point_metrics needs at least one unit")
    bad_p, bad_a = ~np.isfinite(p), ~np.isfinite(a)
    if bad_p.any() or bad_a.any():
        raise ValueError(
            f"{int(bad_p.sum())} non-finite predicted and {int(bad_a.sum())} non-finite actual "
            "value(s); a missing forecast on a fixed cohort set is an error, not a smaller set"
        )
    total = float(a.sum())
    if total <= 0:
        raise ValueError(
            f"sum of actual is non-positive ({total}); pool_ape and pool_pe divide by it"
        )
    e = p - a
    nonzero = a != 0
    ape = np.abs(e[nonzero] / a[nonzero])
    return {
        "n": int(p.size),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "wrmse": float(np.sqrt(np.sum(a * e**2) / total)),
        "mape": float(np.mean(ape)) if ape.size else float("nan"),
        "medape": float(np.median(ape)) if ape.size else float("nan"),
        "pool_ape": float(np.sum(np.abs(e)) / total),
        "pool_pe": float(np.sum(e) / total),
        "prop_over": float(np.mean(e > 0)),
    }


def level_errors(
    frame: pd.DataFrame, *, predicted: str, actual: str, level: Sequence[str]
) -> pd.DataFrame:
    """Sum ``predicted`` and ``actual`` within ``level`` and return one row per unit.

    Columns: the ``level`` columns, ``n_rows`` (rows summed into the unit),
    ``predicted``, ``actual``, ``error``. Feed the result to :func:`point_metrics`.
    """
    level = list(level)
    missing = [c for c in level if c not in frame.columns]
    if missing:
        raise KeyError(f"level column(s) {missing} not in the frame; it has {list(frame.columns)}")
    for name in (predicted, actual):
        if name not in frame.columns:
            raise KeyError(f"column {name!r} not in the frame; it has {list(frame.columns)}")
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{int((~np.isfinite(values)).sum())} non-finite value(s) in {name!r}; a "
                "missing forecast on a fixed cohort set is an error, not a smaller set"
            )
    grouped = frame.groupby(level, sort=True, dropna=False)
    out = grouped.agg(
        n_rows=(predicted, "size"), predicted=(predicted, "sum"), actual=(actual, "sum")
    ).reset_index()
    out["error"] = out["predicted"] - out["actual"]
    return out[[*level, "n_rows", "predicted", "actual", "error"]]


def shrink_toward(point, baseline, alpha: float) -> np.ndarray:
    """``baseline + alpha * (point - baseline)``: a point pulled toward a baseline.

    The R study's final estimator is ``shrink_toward(raw TLRN, MCL, 0.658)``.
    ``alpha = 0`` is the baseline, ``alpha = 1`` the point itself.
    """
    if not (isinstance(alpha, (int, float)) and np.isfinite(alpha) and 0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be a finite number in [0, 1], got {alpha!r}")
    p = np.asarray(point, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if p.shape != b.shape:
        raise ValueError(
            f"point and baseline must have the same shape, got {p.shape} and {b.shape}"
        )
    if not (np.isfinite(p).all() and np.isfinite(b).all()):
        raise ValueError("point and baseline must be finite; a non-finite entry cannot be blended")
    return b + alpha * (p - b)
