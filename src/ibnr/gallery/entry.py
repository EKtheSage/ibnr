"""GalleryEntry: the contract every gallery model must satisfy.

`.fit()`, `.predict()`, `.evaluate()`, `.card()` are mandatory. Evaluation
logic lives in ``kernels`` - entries call it, never reimplement it. An entry
that cannot satisfy this interface does not register.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np

from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.kernels.scores import crps
from ibnr.triangle.core import Triangle


class GalleryEntry(ABC):
    name: ClassVar[str]
    family: ClassVar[str]  # "bayesian" | "nn" | "statistical" | "deterministic"

    @abstractmethod
    def fit(self, triangle: Triangle, **kwargs) -> GalleryEntry:
        """Fit on a training triangle (already sliced with as_of)."""

    @abstractmethod
    def predict(self) -> PredictiveDistribution:
        """Predictive distribution of the fitted quantities (e.g. ultimates)."""

    def evaluate(self, observed) -> dict:
        """Score realized outcomes against the predictive distribution.

        Default implementation (kernels-backed): the Meyers-style summary
        table plus the outcome percentile of each target. Richer harnesses
        (ELPD, stacking) extend this in kernels, not in entries.
        """
        pred = self.predict()
        obs = np.asarray(observed, dtype=float)
        table = pred.summary(observed=obs)
        return {
            "summary": table,
            "percentiles": pred.cdf(obs) * 100.0,
            "crps": crps(pred.samples, obs),
        }

    @classmethod
    def card(cls) -> str:
        """The model card (card.md next to the entry's module)."""
        path = Path(inspect.getfile(cls)).parent / "card.md"
        if not path.exists():
            raise FileNotFoundError(f"{cls.__name__} has no card.md at {path}")
        return path.read_text(encoding="utf-8")
