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

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["level_errors", "point_metrics", "point_summary", "reserve_rows", "shrink_toward"]

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


def point_summary(pred, observed) -> dict:
    """Point errors of a ``PredictiveDistribution``'s draw means against outcomes.

    ``errors`` has one row per target: the target metadata plus ``estimate``
    (the draw mean), ``outcome``, ``error`` and ``pct_error`` (missing where the
    outcome is missing or zero).

    ``metrics`` is :func:`point_metrics` over the scorable targets - those whose
    ``label`` is not ``"total"`` (the total is the sum of the others and would
    count every error twice) and whose outcome is a finite number. It is
    ``None`` when the cohort cannot be scored at all, which happens three ways:
    no scorable target, a scorable target whose draw mean is not finite (a fit
    with missing draws; scoring the rest would put this cohort on fewer targets
    than its neighbours), and scorable outcomes summing to zero or less
    (``pool_ape`` divides by that sum). ``excluded`` counts each reason, so a
    caller reading ``None`` can tell which one it was without re-deriving it.
    """
    obs = np.asarray(observed, dtype=float).reshape(-1)
    if obs.shape != (pred.n_targets,):
        raise ValueError(f"observed must have shape ({pred.n_targets},), got {obs.shape}")
    estimate = pred.mean()
    errors = pred.targets.copy()
    errors["estimate"] = estimate
    errors["outcome"] = obs
    errors["error"] = estimate - obs
    with np.errstate(divide="ignore", invalid="ignore"):
        errors["pct_error"] = np.where(obs != 0, (estimate - obs) / obs, np.nan)
    is_total = (
        errors["label"].astype(str).to_numpy() == "total"
        if "label" in errors.columns
        else np.zeros(len(errors), dtype=bool)
    )
    missing = ~np.isfinite(obs)
    scorable = ~is_total & ~missing
    no_estimate = int((~np.isfinite(estimate[scorable])).sum())
    non_positive = int(bool(scorable.any()) and float(obs[scorable].sum()) <= 0)
    metrics = (
        point_metrics(estimate[scorable], obs[scorable])
        if scorable.any() and not no_estimate and not non_positive
        else None
    )
    return {
        "errors": errors,
        "metrics": metrics,
        "excluded": {
            "total": int(is_total.sum()),
            "missing_outcome": int(missing.sum()),
            "missing_estimate": no_estimate,
            "non_positive_actual": non_positive,
        },
    }


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


def _fitted_origins(entry) -> list:
    """The origin periods ``entry`` was fitted on.

    Every contract in this package carries them under ``origin_periods`` - the
    three Stan grids, the dense cohort grid, the multi-LOB grid and the neural
    grids alike - and Mack's fit object carries them too, which is the fallback.

    ``reserve_rows`` reads the anchor and the premium off the full triangle,
    which can hold accident years the fit never saw: the Schedule P mart holds
    1988 to 2007 and a 2007 valuation observes all of them, while a study fits
    1988 to 1997. ``realized_ultimates`` already restricts itself to these
    origins, so the anchor has to as well or the two sides of
    ``ultimate - anchor`` cover different accident years.
    """
    contract = getattr(entry, "contract_", None)
    if isinstance(contract, Mapping) and "origin_periods" in contract:
        return list(contract["origin_periods"])
    origins = getattr(getattr(entry, "fit_", None), "origin_periods", None)
    if origins is not None:
        return list(origins)
    name = getattr(entry, "name", type(entry).__name__)
    raise AttributeError(
        f"{name} exposes neither contract_['origin_periods'] nor fit_.origin_periods; "
        "reserve_rows restricts the anchor and the premium to the origins the entry was "
        "fitted on and cannot tell which those are"
    )


def _latest_total(triangle, field: str, cohort: dict, origins) -> float:
    """The cohort's latest observed ``field`` value per origin, summed over ``origins``.

    The restriction is the whole point: see :func:`_fitted_origins`.
    """
    frame = _cohort_triangle(triangle, cohort).select_fields(field).latest_diagonal().execute()
    wanted = set(pd.to_datetime(pd.Index(list(origins))))
    keep = pd.to_datetime(frame["origin_period"]).isin(wanted)
    return float(frame.loc[keep, "value"].sum())


def _total_index(labels: list[str], what: str) -> int:
    if "total" not in labels:
        raise ValueError(
            f"{what} carries no 'total' row (labels: {labels[:6]}...); reserve_rows reads the "
            "cohort's total ultimate off that row"
        )
    return labels.index("total")


def _realized_total_index(n_rows: int, total_row: int, n_realized: int, what: str) -> int:
    """Where a cohort's realized total sits in its ``realized_ultimates`` array.

    Every entry puts it last: the single-line and pooled entries return one
    value per origin and then the total, the multi-LOB layout ends with the
    grand total, and ``tlrn`` returns the company total on its own. The native
    route never calls ``predict``, so it cannot read the index off the draws'
    labels; it checks that rule here instead of assuming it.
    """
    if n_realized == 1:
        return 0
    if n_rows == n_realized and total_row == n_realized - 1:
        return n_realized - 1
    raise ValueError(
        f"{what} has {n_rows} row(s) with 'total' at index {total_row}, against {n_realized} "
        "realized value(s); reserve_rows reads the realized total as the last element and "
        "cannot line the two up"
    )


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
    ``as_of`` summed over the origins the entry was FITTED on, read from
    ``full_triangle.as_of(as_of)``. ``realized_ultimate`` comes from
    ``entry.realized_ultimates``, which restricts itself to those same origins,
    so both sides of the subtraction cover the same accident years even when the
    triangle carries more history than the fit saw.

    ``full_triangle`` may carry cohorts this entry was not fitted on: every read
    of it here is filtered to the cohort first, so a company's anchor cannot
    pick up its neighbour's losses and a single-cohort entry is handed only the
    rows it answers for.

    ``point="draw_mean"`` reads the mean of ``entry.predict()``'s ``total`` row;
    ``point="native"`` reads the ``total`` row of ``entry.point()``, refuses an
    entry that has none, and does not call ``predict`` at all - an entry whose
    DRAWS refuse a cohort (Mack on a cohort whose newest accident year has a
    zero on the valuation diagonal) still has a point, and dropping its row
    leaves a company total one line short and entirely plausible. Such a row
    records ``n_draws = 0``. ``point_source`` records which route was taken.
    ``predict_kwargs`` are handed to ``predict`` unchanged (``seed``,
    ``n_draws``), so an entry that does not accept one raises rather than
    quietly ignoring it; on the native route they are refused instead, because
    nothing would read them.

    ``premium`` is the premium field's latest value per origin, summed over the
    same fitted origins; ``premium_field=None`` records NaN. Feed the rows to
    :func:`level_errors` with ``level=["company_code"]`` for a company board, or
    with the full cohort key for a company-line board.
    """
    if point not in POINT_SOURCES:
        raise ValueError(f"point must be one of {POINT_SOURCES}, got {point!r}")
    if point == "native" and not callable(getattr(entry, "point", None)):
        name = getattr(entry, "name", type(entry).__name__)
        raise TypeError(
            f"{name} does not implement point(); it has draws only, so use point='draw_mean'"
        )
    if point == "native" and predict_kwargs:
        raise ValueError(
            f"predict_kwargs {sorted(predict_kwargs)} were passed beside point='native', which "
            "never calls predict(); they would be inert. Drop them or use point='draw_mean'"
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
    origins = _fitted_origins(entry)
    rows = []
    for cohort in chosen:
        outcomes = entry.realized_ultimates(_cohort_triangle(full_triangle, cohort), segment=cohort)
        realized = np.asarray(outcomes, dtype=float)
        if point == "native":
            frame = entry.point(segment=cohort)
            j = _total_index(frame["label"].astype(str).tolist(), f"{name}.point() frame")
            predicted = float(frame["point"].iloc[j])
            i = _realized_total_index(len(frame), j, realized.size, f"{name}.point() frame")
            n_draws = 0
        else:
            pred = entry.predict(segment=cohort, **kwargs)
            labels = pred.targets["label"].astype(str).tolist()
            i = _total_index(labels, f"{name}.predict targets")
            predicted = float(pred.mean()[i])
            n_draws = int(pred.n_draws)
        anchor = _latest_total(training, loss_field, cohort, origins)
        premium = (
            _latest_total(training, premium_field, cohort, origins)
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
                "n_draws": n_draws,
                "point_source": point,
            }
        )
    return pd.DataFrame(rows)
