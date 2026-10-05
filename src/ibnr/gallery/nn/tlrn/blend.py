"""The weight of the ``mcl_blend`` member. Numpy only.

A member forecasts ``m + alpha * (n - m)``: ``m`` the multivariate chain ladder's
forecast, ``n`` the network's, ``alpha`` in [0, 1]. The validation score sums the
error within each (accident year, line) before taking its absolute value, so with
``a_g`` the error of ``m`` summed over group ``g`` and ``b_g`` the difference
``n - m`` summed over it, the score is ``sum_g |a_g + alpha * b_g|``, a convex
piecewise-linear function of alpha.

Each term is ``|b_g| * |alpha - t_g|`` with ``t_g = -a_g / b_g``, so the
minimiser is the weighted median of the ``t_g`` with weights ``|b_g|``, and
clipping it to [0, 1] is exactly the constrained minimum because the function is
convex. That is an exact answer, not a grid search: see
``tests/test_tlrn_components.py``, which compares it with a fine grid.
"""

from __future__ import annotations

import numpy as np

__all__ = ["blend_weight"]


def blend_weight(a: np.ndarray, b: np.ndarray) -> float:
    """``argmin over alpha in [0, 1]`` of ``sum |a + alpha * b|``, exactly.

    When no group has a difference between the two forecasts (every ``b`` is zero
    or not finite) the score does not depend on alpha, and the network is taken
    whole: ``alpha = 1``.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b) & (b != 0)
    if not keep.any():
        return 1.0
    a, b = a[keep], b[keep]
    t = -a / b
    weight = np.abs(b)
    order = np.argsort(t, kind="stable")
    t, weight = t[order], weight[order]
    reached = np.cumsum(weight)
    median = t[int(np.searchsorted(reached, 0.5 * reached[-1], side="left"))]
    return float(np.clip(median, 0.0, 1.0))
