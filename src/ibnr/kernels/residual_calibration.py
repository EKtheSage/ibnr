"""Size-stratified rolling-origin residual calibration for a point forecaster.

The R transformer study's uncertainty for its point model, written once and
generic: any callable that forecasts a fixed set of units at a cutoff can be
calibrated.

1. :func:`rolling_residuals` applies the forecaster at each earlier cutoff and
   standardises ``actual - predicted`` by ``max(|predicted|, 0.01 * size, 1)``.
2. :func:`calibrate` keeps the named horizons, assigns every unit a size
   stratum (dplyr's ``ntile`` over the units), and centres each stratum's
   errors on its median.
3. :func:`calibrated_draws` resamples a unit's stratum pool around its final
   point, on the same floored scale.
4. :func:`leave_one_out_coverage` is the R study's cross-unit historical
   coverage table: each calibration forecast is covered by an interval built
   from the OTHER units of its stratum.

The units in the R study are companies and ``size`` is company premium, but
nothing here knows that.

What this is not: a native predictive distribution. Whoever consumes the draws
must say they are historically calibrated. It is also blind to what the
forecaster trained on: the R study calibrates on cutoffs whose targets overlap
the network's training targets, and that choice belongs to the caller, who
picks ``cutoffs`` and ``horizons``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Calibration",
    "calibrate",
    "calibrated_draws",
    "leave_one_out_coverage",
    "ntile",
    "rolling_residuals",
]

_COLUMNS = [
    "unit",
    "cutoff",
    "horizon",
    "predicted",
    "actual",
    "size",
    "residual_scale",
    "standardised_error",
]


def ntile(values, n: int) -> np.ndarray:
    """dplyr's ``ntile`` (1.0 and later), 1-based.

    Rank by value with ties broken by position, then split into bins as equal
    in size as possible with the LARGER bins first. ``len = 93, n = 4`` gives
    bins of 24, 23, 23, 23, which is how the R study's companies fall into
    premium quartiles. With fewer values than bins the trailing bins are empty,
    as in dplyr.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    order = np.argsort(v, kind="stable")
    rank = np.empty(v.size, dtype=int)
    rank[order] = np.arange(1, v.size + 1)
    smaller = v.size // n
    n_larger = v.size % n
    larger = smaller + 1
    threshold = larger * n_larger  # ranks up to here fall in the n_larger larger bins
    bins = np.where(
        rank <= threshold,
        np.ceil(rank / max(larger, 1)),
        n_larger + np.ceil((rank - threshold) / max(smaller, 1)),
    )
    return bins.astype(int)


def _scale(predicted: np.ndarray, size: np.ndarray, floor: tuple[float, float]) -> np.ndarray:
    """``max(|predicted|, floor[0] * size, floor[1])``, elementwise."""
    return np.maximum.reduce([np.abs(predicted), floor[0] * size, np.full_like(size, floor[1])])


def rolling_residuals(
    forecast_at: Callable[[int], np.ndarray],
    actual_at: Callable[[int], np.ndarray],
    *,
    cutoffs: Sequence[int],
    n_periods: int,
    size,
    scale_floor: tuple[float, float] = (0.01, 1.0),
) -> pd.DataFrame:
    """Standardised forecast errors of a fixed forecaster at earlier cutoffs.

    ``forecast_at(k)`` and ``actual_at(k)`` return one value per unit, in one
    fixed unit order, for cutoff ``k``; ``horizon = n_periods - k``. Rows whose
    standardised error is not finite are dropped and counted in
    ``frame.attrs["n_dropped_nonfinite"]``, as the R study filters them.
    """
    size = np.asarray(size, dtype=float).reshape(-1)
    rows = []
    dropped = 0
    for k in cutoffs:
        predicted = np.asarray(forecast_at(k), dtype=float).reshape(-1)
        actual = np.asarray(actual_at(k), dtype=float).reshape(-1)
        if predicted.size != size.size or actual.size != size.size:
            raise ValueError(
                f"cutoff {k}: forecast_at returned length {predicted.size} and actual_at "
                f"length {actual.size}; both must match size's length {size.size}"
            )
        scale = _scale(predicted, size, scale_floor)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (actual - predicted) / scale
        keep = np.isfinite(z)
        dropped += int((~keep).sum())
        for u in np.nonzero(keep)[0]:
            rows.append(
                {
                    "unit": int(u),
                    "cutoff": int(k),
                    "horizon": int(n_periods - k),
                    "predicted": float(predicted[u]),
                    "actual": float(actual[u]),
                    "size": float(size[u]),
                    "residual_scale": float(scale[u]),
                    "standardised_error": float(z[u]),
                }
            )
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    frame.attrs["n_dropped_nonfinite"] = dropped
    return frame


@dataclass(frozen=True)
class Calibration:
    """Centred residual pools per size stratum; see the module docstring."""

    horizons: tuple[int, ...]
    n_strata: int
    unit_stratum: dict
    size_edges: np.ndarray  # upper size bound of each stratum; the last is +inf
    medians: np.ndarray  # (n_strata,) the median subtracted from each pool
    pools: tuple  # n_strata arrays of centred standardised errors
    residuals: pd.DataFrame  # the rows used, with a `stratum` column

    def strata_for(self, size) -> np.ndarray:
        """1-based stratum per size, by the stored upper bounds."""
        s = np.asarray(size, dtype=float).reshape(-1)
        return np.searchsorted(self.size_edges[:-1], s, side="left") + 1


def calibrate(
    residuals: pd.DataFrame,
    *,
    horizons: Sequence[int],
    n_strata: int = 4,
    min_per_stratum: int = 40,
) -> Calibration:
    """Pool the named horizons' errors by unit size and centre each pool.

    A stratum holding fewer than ``min_per_stratum`` residuals is refused by
    name: resampling from a handful of errors would report a spread that is
    mostly an accident of which units fell into that stratum.
    """
    horizons = tuple(int(h) for h in horizons)
    have = set(residuals["horizon"].unique())
    missing = [h for h in horizons if h not in have]
    if missing:
        raise ValueError(f"horizon(s) {missing} not in the residuals, which carry {sorted(have)}")
    units = residuals.groupby("unit")["size"].first().sort_index()
    stratum = ntile(units.to_numpy(), n_strata)
    unit_stratum = dict(zip(units.index.tolist(), stratum.tolist(), strict=True))
    sizes = units.to_numpy()
    edges = np.array(
        [
            sizes[stratum == s].max() if (stratum == s).any() else -np.inf
            for s in range(1, n_strata + 1)
        ]
    )
    edges[-1] = np.inf
    used = residuals[residuals["horizon"].isin(horizons)].copy()
    used["stratum"] = used["unit"].map(unit_stratum).astype(int)
    medians, pools = [], []
    for s in range(1, n_strata + 1):
        errors = used.loc[used["stratum"] == s, "standardised_error"].to_numpy(dtype=float)
        if errors.size < min_per_stratum:
            raise ValueError(
                f"stratum {s} of {n_strata} has {errors.size} residual(s) over horizons "
                f"{horizons}; fewer than {min_per_stratum} is too few to resample from"
            )
        med = float(np.median(errors))
        medians.append(med)
        pools.append(errors - med)
    return Calibration(
        horizons=horizons,
        n_strata=n_strata,
        unit_stratum=unit_stratum,
        size_edges=edges,
        medians=np.array(medians),
        pools=tuple(pools),
        residuals=used.reset_index(drop=True),
    )


def calibrated_draws(
    calibration: Calibration,
    *,
    point,
    size,
    n_draws: int,
    rng: np.random.Generator,
    scale_floor: tuple[float, float] = (0.01, 1.0),
) -> np.ndarray:
    """``(n_draws, n_units)``: ``point + scale * resampled centred residual``."""
    p = np.asarray(point, dtype=float).reshape(-1)
    s = np.asarray(size, dtype=float).reshape(-1)
    if p.size != s.size:
        raise ValueError(f"point and size must have the same length, got {p.size} and {s.size}")
    if not isinstance(n_draws, int) or n_draws < 1:
        raise ValueError(f"n_draws must be an int >= 1, got {n_draws!r}")
    scale = _scale(p, s, scale_floor)
    strata = calibration.strata_for(s)
    out = np.empty((n_draws, p.size))
    for u in range(p.size):
        pool = calibration.pools[strata[u] - 1]
        out[:, u] = p[u] + scale[u] * rng.choice(pool, size=n_draws, replace=True)
    return out


def leave_one_out_coverage(
    calibration: Calibration, *, levels: Sequence[float] = (0.8, 0.95)
) -> pd.DataFrame:
    """Coverage of intervals built for each calibration forecast from the OTHER
    units of its stratum (the R study's cross-company historical coverage).

    The quantiles are numpy's ``median_unbiased`` method, which is R's type 8.
    """
    used = calibration.residuals
    rows = []
    for level in levels:
        alpha = (1.0 - level) / 2.0
        covered = []
        for _, r in used.iterrows():
            pool = used.loc[
                (used["stratum"] == r["stratum"]) & (used["unit"] != r["unit"]),
                "standardised_error",
            ].to_numpy(dtype=float)
            centred = pool - np.median(pool)
            lo, hi = np.quantile(centred, [alpha, 1 - alpha], method="median_unbiased")
            covered.append(
                r["predicted"] + r["residual_scale"] * lo
                <= r["actual"]
                <= r["predicted"] + r["residual_scale"] * hi
            )
        cov = used.assign(covered=covered).groupby("horizon")["covered"].agg(["mean", "size"])
        for horizon, (mean, n) in cov.iterrows():
            rows.append(
                {
                    "horizon": int(horizon),
                    "nominal_coverage": float(level),
                    "empirical_coverage": float(mean),
                    "forecasts": int(n),
                }
            )
    return pd.DataFrame(
        rows, columns=["horizon", "nominal_coverage", "empirical_coverage", "forecasts"]
    )
