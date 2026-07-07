"""PIT calibration: uniformity testing of outcome percentiles.

If a model's predictive distributions are well calibrated, the PIT values of
realized outcomes across many triangles are uniform on [0, 1]. Meyers tests
this with p-p plots and the Kolmogorov-Smirnov statistic (D x 100 against the
5% critical value 1.36/sqrt(n) x 100 — 19.2 for n=50, 9.6 for n=200).

Implemented with numpy only (no scipy in the core).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KSResult:
    statistic: float  # sup |ecdf - uniform cdf|, in [0, 1]
    p_value: float  # asymptotic Kolmogorov distribution
    n: int
    critical_value_5pct: float  # 1.36 / sqrt(n)

    @property
    def reject_5pct(self) -> bool:
        return self.statistic > self.critical_value_5pct

    def __repr__(self) -> str:  # Meyers reports D and the critical value x100
        flag = " *" if self.reject_5pct else ""
        return (
            f"KS D = {self.statistic * 100:.1f}{flag} "
            f"(crit. val. = {self.critical_value_5pct * 100:.1f}, "
            f"n = {self.n}, p = {self.p_value:.3f})"
        )


def ks_uniformity(pits) -> KSResult:
    """Two-sided KS test of PIT values (in [0, 1]) against Uniform(0, 1)."""
    p = np.sort(np.asarray(pits, dtype=float))
    if p.size == 0:
        raise ValueError("no PIT values")
    if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("PIT values must be in [0, 1] and non-NaN")
    n = p.size
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - p)
    d_minus = np.max(p - (i - 1) / n)
    d = max(d_plus, d_minus)
    return KSResult(
        statistic=float(d),
        p_value=_kolmogorov_sf(np.sqrt(n) * d),
        n=n,
        critical_value_5pct=1.36 / np.sqrt(n),
    )


def _kolmogorov_sf(x: float) -> float:
    """Asymptotic Kolmogorov distribution survival function."""
    if x <= 0:
        return 1.0
    k = np.arange(1, 101)
    return float(np.clip(2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * (k * x) ** 2)), 0.0, 1.0))


def pp_points(pits) -> tuple[np.ndarray, np.ndarray]:
    """(expected, predicted) percentile pairs for a Meyers-style p-p plot."""
    p = np.sort(np.asarray(pits, dtype=float))
    n = p.size
    expected = np.arange(1, n + 1) / (n + 1)
    return expected * 100.0, p * 100.0
