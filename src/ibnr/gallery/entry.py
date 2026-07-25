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

from ibnr.kernels.densities import MEASURES, to_amount_scale
from ibnr.kernels.holdout import CellIndex, HoldoutCells, index_into, training_index
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


class ScoresHeldout(ABC):
    """Opt-in capability: this entry can evaluate its own likelihood at cells it
    was not trained on.

    A **mixin**, not an addition to :class:`GalleryEntry`, so no existing entry
    breaks and no entry has to pretend. Entries that cannot do this - the NN,
    statistical and deterministic families today - simply do not subclass it,
    and the leaderboard reports their ELPD as missing rather than as zero.

    Two structural choices, both there to stop a whole class of silent error:

    **The measure conversion happens here, in the base.** Subclasses implement
    :meth:`_log_lik_native`, which returns the density on the entry's *own*
    scale, and :meth:`log_lik_at` carries it to Lebesgue-on-amount. An entry
    cannot skip the carry or apply its own, which matters because the five
    Bayesian entries sit on four different measures and a wrong Jacobian
    produces numbers that still rank (see ``kernels/densities.py``).

    **The scorer reads ``idata.posterior``, never ``idata.log_likelihood``.**
    That group is not uniform: Stan names it ``log_lik`` and both ports name it
    ``obs``; NumPyro pollutes it with scalar ``*_prior`` factor sites; and
    ``clark_growth_curve``'s PyMC port attaches via ``pm.Potential`` so it has no
    such group at all. The posterior is the same in every backend.
    """

    #: which measure :meth:`_log_lik_native` returns. One of ``MEASURES``.
    heldout_measure: ClassVar[str]

    @abstractmethod
    def _log_lik_native(self, cells: CellIndex) -> np.ndarray:
        """``(n_draws, n_cells)`` log density on this entry's own measure."""

    def heldout_dispersion(self) -> np.ndarray | float:
        """Lattice spacing for ``odp_lattice`` entries. Unused otherwise."""
        raise NotImplementedError(
            f"{type(self).__name__} declares heldout_measure='odp_lattice' but does "
            "not implement heldout_dispersion()"
        )

    def log_lik_at(self, cells: HoldoutCells | CellIndex, *, field: str | None = None):
        """``(n_draws, n_cells)`` log density, carried to Lebesgue-on-amount.

        Accepts either a :class:`~ibnr.kernels.holdout.HoldoutCells` (which is
        indexed against this fit's contract) or an already-built
        :class:`~ibnr.kernels.holdout.CellIndex`.
        """
        idx = (
            cells
            if isinstance(cells, CellIndex)
            else index_into(cells, self.contract_, field=field)
        )
        native = np.asarray(self._log_lik_native(idx), dtype=float)
        if native.shape[1] != idx.n_cells:
            raise ValueError(
                f"{type(self).__name__}._log_lik_native returned {native.shape[1]} columns "
                f"for {idx.n_cells} cells"
            )
        return to_amount_scale(native, **self._measure_covariates(idx))

    def training_cells(self) -> CellIndex:
        """The cells this fit was trained on, for the in-sample agreement gate."""
        return training_index(self.contract_)

    def _measure_covariates(self, idx: CellIndex) -> dict:
        """The covariate this entry's measure needs, resolved from the cells.

        Deliberately explicit per measure rather than "pass everything and let
        the callee pick": ``to_amount_scale`` refuses an unused covariate, so a
        mismatch between a declared measure and what is supplied is an error
        instead of an unconverted density.
        """
        measure = getattr(self, "heldout_measure", None)
        if measure not in MEASURES:
            raise ValueError(
                f"{type(self).__name__}.heldout_measure must be one of "
                f"{sorted(MEASURES)}, got {measure!r}"
            )
        if measure == "amount":
            return {"measure": measure}
        if measure == "log_amount":
            return {"measure": measure, "value": idx.value}
        if measure == "loss_ratio":
            if np.isnan(idx.premium).any():
                raise ValueError(
                    f"{type(self).__name__} scores on loss ratios but the fit carries no "
                    "premium, so its density cannot be carried to the amount scale"
                )
            return {"measure": measure, "premium": idx.premium}
        return {"measure": measure, "phi": self.heldout_dispersion()}
