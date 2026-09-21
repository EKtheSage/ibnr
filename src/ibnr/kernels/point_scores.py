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

__all__ = ["level_errors", "point_metrics", "reserve_rows", "shrink_toward"]

#: what ``reserve_rows`` may read a cohort's predicted ultimate from.
POINT_SOURCES: tuple[str, ...] = ("draw_mean", "native")


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


def _cohort_triangle(triangle, cohort: dict):
    """``triangle`` restricted to one cohort's rows.

    ``cohort`` may carry columns the triangle does not (a display column a
    pooled fit kept beside its key); only the shared columns filter, and every
    shared column must match. A cohort with no shared column - a single-cohort
    fit on a triangle that has no segment columns at all - filters nothing,
    which is that triangle's only cohort.
    """
    segments = triangle.segments
    predicates = [triangle.expr[c] == v for c, v in cohort.items() if c in segments]
    return triangle.filter(*predicates) if predicates else triangle


def _latest_total(triangle, field: str, cohort: dict) -> float:
    """The cohort's latest observed ``field`` value per origin, summed."""
    frame = _cohort_triangle(triangle, cohort).select_fields(field).latest_diagonal().execute()
    return float(frame["value"].sum())


def _total_index(labels: list[str], what: str) -> int:
    if "total" not in labels:
        raise ValueError(
            f"{what} carries no 'total' row (labels: {labels[:6]}...); reserve_rows reads the "
            "cohort's total ultimate off that row"
        )
    return labels.index("total")


def reserve_rows(
    entry,
    full_triangle,
    *,
    as_of,
    loss_field: str,
    premium_field: str | None = "earned_premium",
    point: str = "draw_mean",
    segment=None,
    predict_kwargs: dict | None = None,
) -> pd.DataFrame:
    """One row per cohort: predicted and actual reserve at the grid endpoint.

    ``reserve = ultimate - anchor``, the anchor being the cumulative observed at
    ``as_of`` summed over the cohort's origins, read from
    ``full_triangle.as_of(as_of)`` - the same slice the entry was fitted on.
    ``realized_ultimate`` comes from ``entry.realized_ultimates``, which
    restricts itself to the training origins, so both sides of the subtraction
    cover the same accident years.

    ``full_triangle`` may carry cohorts this entry was not fitted on: every read
    of it here is filtered to the cohort first, so a company's anchor cannot
    pick up its neighbour's losses and a single-cohort entry is handed only the
    rows it answers for.

    ``point="draw_mean"`` reads the mean of ``entry.predict()``'s ``total`` row;
    ``point="native"`` reads the ``total`` row of ``entry.point()`` and refuses
    an entry that has none. ``point_source`` records which. ``predict_kwargs``
    are handed to ``predict`` unchanged (``seed``, ``n_draws``), so an entry
    that does not accept one raises rather than quietly ignoring it.

    ``premium`` is the premium field's latest value per origin, summed over the
    origins written at ``as_of``; ``premium_field=None`` records NaN. Feed the
    rows to :func:`level_errors` with ``level=["company_code"]`` for a company
    board, or with the full cohort key for a company-line board.
    """
    if point not in POINT_SOURCES:
        raise ValueError(f"point must be one of {POINT_SOURCES}, got {point!r}")
    if point == "native" and not callable(getattr(entry, "point", None)):
        name = getattr(entry, "name", type(entry).__name__)
        raise TypeError(
            f"{name} does not implement point(); it has draws only, so use point='draw_mean'"
        )
    training = full_triangle.as_of(as_of)
    fields = list(training.fields)
    if loss_field not in fields:
        raise ValueError(f"no field named {loss_field!r}; the triangle carries {sorted(fields)}")
    if premium_field is not None and premium_field not in fields:
        raise ValueError(
            f"no field named {premium_field!r}; the triangle carries {sorted(fields)}. "
            "Pass premium_field=None to record no premium"
        )
    kwargs = dict(predict_kwargs or {})
    cohorts = entry.cohorts()
    index = entry.cohort_index(segment)
    chosen = cohorts if index is None else [cohorts[index]]
    name = getattr(entry, "name", type(entry).__name__)
    rows = []
    for cohort in chosen:
        pred = entry.predict(segment=cohort, **kwargs)
        labels = pred.targets["label"].astype(str).tolist()
        i = _total_index(labels, f"{name}.predict targets")
        outcomes = entry.realized_ultimates(_cohort_triangle(full_triangle, cohort), segment=cohort)
        realized = np.asarray(outcomes, dtype=float)
        if point == "native":
            frame = entry.point(segment=cohort)
            j = _total_index(frame["label"].astype(str).tolist(), f"{name}.point() frame")
            predicted = float(frame["point"].iloc[j])
        else:
            predicted = float(pred.mean()[i])
        anchor = _latest_total(training, loss_field, cohort)
        premium = (
            _latest_total(training, premium_field, cohort)
            if premium_field is not None
            else float("nan")
        )
        rows.append(
            {
                **cohort,
                "predicted_ultimate": predicted,
                "realized_ultimate": float(realized[i]),
                "anchor": anchor,
                "predicted_reserve": predicted - anchor,
                "actual_reserve": float(realized[i]) - anchor,
                "premium": premium,
                "n_draws": int(pred.n_draws),
                "point_source": point,
            }
        )
    return pd.DataFrame(rows)
