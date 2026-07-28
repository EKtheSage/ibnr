"""PredictiveDistribution: the unifying output type for every gallery entry.

Every model - Bayesian, NN, or bootstrapped deterministic - must produce one.
It is just a matrix of posterior/predictive samples over named targets, plus
the methods the evaluation harness needs (means, quantiles, outcome
percentiles a la Meyers, summary tables).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PredictiveDistribution:
    """Sampled predictive distribution over a set of targets.

    samples: (n_draws, n_targets) array of predictive samples.
    targets: one row of metadata per target (e.g. origin_period, premium,
             label). Index is positional.
    units:   carried from the source triangle, purely informational.
    """

    samples: np.ndarray
    targets: pd.DataFrame
    units: str | None = field(default=None)

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim != 2:
            raise ValueError(f"samples must be 2-D (draws x targets), got {self.samples.shape}")
        if len(self.targets) != self.samples.shape[1]:
            raise ValueError(
                f"{self.samples.shape[1]} sample columns but {len(self.targets)} target rows"
            )
        self.targets = self.targets.reset_index(drop=True)

    @property
    def n_draws(self) -> int:
        return self.samples.shape[0]

    @property
    def n_targets(self) -> int:
        return self.samples.shape[1]

    # -- moments and quantiles -------------------------------------------------

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def std(self) -> np.ndarray:
        return self.samples.std(axis=0, ddof=1)

    def quantile(self, q: float | list[float]) -> np.ndarray:
        return np.quantile(self.samples, q, axis=0)

    # -- scoring ---------------------------------------------------------------

    def cdf(self, observed) -> np.ndarray:
        """Empirical predictive CDF at the observed outcomes, in [0, 1].

        This is the PIT value / "outcome percentile" of Meyers' validation
        tables (he reports it x100). One value per target; NaN observations
        propagate.
        """
        obs = np.asarray(observed, dtype=float)
        if obs.shape != (self.n_targets,):
            raise ValueError(f"observed must have shape ({self.n_targets},), got {obs.shape}")
        return (self.samples <= obs).mean(axis=0)

    def summary(self, observed=None) -> pd.DataFrame:
        """Meyers-style output table: one row per target with posterior mean,
        SE, CV, and (when outcomes are given) the outcome and its percentile."""
        mean = self.mean()
        se = self.std()
        out = self.targets.copy()
        out["estimate"] = mean
        out["se"] = se
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cv"] = np.where(mean != 0, se / mean, np.nan)
        if observed is not None:
            obs = np.asarray(observed, dtype=float)
            out["outcome"] = obs
            out["percentile"] = self.cdf(obs) * 100.0
        return out

    # -- serialization -----------------------------------------------------------
    #
    # Thin delegation on purpose: ``kernels/codec.py`` owns the format and this
    # class owns nothing about it. pyarrow is imported there and never here, so
    # the type every gallery entry returns keeps a numpy/pandas-only import path.

    def to_arrow(self, *, compression: str | None = None) -> bytes:
        """Arrow IPC bytes carrying every draw, bit for bit. See ``kernels.codec``."""
        from ibnr.kernels import codec

        return codec.to_arrow(self, compression=compression)

    @classmethod
    def from_arrow(cls, data: bytes) -> PredictiveDistribution:
        from ibnr.kernels import codec

        return codec.from_arrow(data)

    def to_summary(self, *, quantiles: Sequence[float] | None = None) -> dict:
        """JSON-safe digest - moments, quantiles, target metadata, no draws.

        Two orders of magnitude smaller than the draws and deliberately
        one-way: there is no ``from_summary``, because a summary is not a
        distribution (CLAUDE.md decision 4).
        """
        from ibnr.kernels import codec

        kwargs = {} if quantiles is None else {"quantiles": quantiles}
        return codec.to_summary(self, **kwargs)

    # -- composition -------------------------------------------------------------

    def with_total(
        self, label: str = "total", label_column: str = "label"
    ) -> PredictiveDistribution:
        """Append a target that is the row-wise sum of all current targets."""
        total = self.samples.sum(axis=1, keepdims=True)
        targets = self.targets.copy()
        if label_column not in targets.columns:
            targets[label_column] = [str(i) for i in range(len(targets))]
        total_row = pd.DataFrame([{label_column: label}])
        return PredictiveDistribution(
            samples=np.hstack([self.samples, total]),
            targets=pd.concat([targets, total_row], ignore_index=True),
            units=self.units,
        )
