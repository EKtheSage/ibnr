"""Proper scoring rules, implemented once - gallery entries call these.

Only sample-based estimators live here: they apply uniformly to every
PredictiveDistribution regardless of the model family that produced it.
Density-based scores (ELPD/log-score) are deferred: estimating a density
from samples needs bandwidth choices that would leak into model rankings.
"""

from __future__ import annotations

import numpy as np


def crps(samples: np.ndarray, observed) -> np.ndarray:
    """Sample-based CRPS, one value per target. Lower is better.

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'| with both expectations under the
    empirical distribution of the draws (the standard 1/n^2 estimator),
    computed in O(n log n) per target via the sorted-sample identity
    sum_ij |x_i - x_j| = 2 * sum_i (2i - n - 1) x_(i).

    samples:  (n_draws, n_targets) predictive draws.
    observed: (n_targets,) realized outcomes; NaN propagates.
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2-D (draws x targets), got {x.shape}")
    y = np.asarray(observed, dtype=float)
    if y.shape != (x.shape[1],):
        raise ValueError(f"observed must have shape ({x.shape[1]},), got {y.shape}")
    n = x.shape[0]
    if n < 2:
        raise ValueError("CRPS needs at least 2 draws")

    mad = np.abs(x - y).mean(axis=0)
    x_sorted = np.sort(x, axis=0)
    weights = 2.0 * np.arange(1, n + 1) - n - 1  # (2i - n - 1), i = 1..n
    spread = (weights @ x_sorted) / (n * n)  # 0.5 * E|X - X'|
    return mad - spread
