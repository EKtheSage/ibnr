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
    breaks and no entry has to pretend. Entries that cannot do this simply do
    not subclass it, and the leaderboard reports their ELPD as missing rather
    than as zero.

    **Subclassing this is a claim that the entry has a normalized predictive
    density, and that claim is the whole basis of its ELPD.** It is not a
    Bayesian/non-Bayesian distinction - a distributional NN head or a GLM with a
    declared observation model qualifies, and being fitted by MCMC does not. It
    excludes ``england_verrall_odp`` and ``clark_growth_curve``, whose ODP
    *quasi*-likelihood is Poisson only up to proportionality and is not a
    density on any scale (see :ref:`odp-not-a-density`); they are scored by CRPS
    and PIT until given a proper predictive law. It also excludes point and
    quantile predictors, and ``deterministic/mack``, whose bootstrap has no
    stated observation model.

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
        if np.isnan(idx.premium).any():
            raise ValueError(
                f"{type(self).__name__} scores on loss ratios but the fit carries no "
                "premium, so its density cannot be carried to the amount scale"
            )
        return {"measure": measure, "premium": idx.premium}


#: the bases a held-out draw can be on. The TRIANGLE's vocabulary
#: (``TriangleMeta.measure``), deliberately - not ``densities.MEASURES``, which
#: is the density's scale. Two different words spelled "measure"; see
#: :class:`PredictsHeldout`.
DRAW_SCALES: tuple[str, ...] = ("cumulative", "incremental")


class PredictsHeldout(ABC):
    """Opt-in capability: this entry can **draw the outcome** at cells it was not
    trained on.

    The sibling of :class:`ScoresHeldout`, and deliberately a separate mixin
    because the two capabilities are genuinely independent. A density gives ELPD;
    draws give CRPS and PIT. ``england_verrall_odp`` has draws but no usable
    density (its quasi-likelihood is not normalized, see
    :ref:`odp-not-a-density`), so it belongs on the CRPS board and not the ELPD
    one; a Gaussian-head NN could be the reverse. Folding them into one mixin
    would force every entry to claim both or neither.

    **What the declaration is for.** ``heldout_draw_scale`` says whether
    :meth:`_draws_native` draws a **cumulative** loss or an **incremental** one.
    That is not bookkeeping. Three of the five Bayesian entries model increments
    while the Schedule P triangles are cumulative, so undeclared increment draws
    scored against ``HoldoutCells.values`` are wrong by the whole training-diagonal
    anchor - measured on a synthetic cell with anchor 1000: CRPS 996 where the
    truth is 3.4, both finite, both smooth, both entirely plausible on a board.
    It is exactly the bug class an unconverted Jacobian is for a density, which
    is why the conversion lives here, in the base, and not in any entry.

    **Why the conversion is free of leakage.** ``C = X + C_prev`` with ``C_prev``
    on the *training* diagonal, so it is data the model already had, not a
    prediction. That is the same fact that makes the increment/cumulative
    Jacobian 1 for a density (``kernels/densities.py``).

    **Why comparing across scales is legitimate at all.** CRPS is
    translation-equivariant - ``CRPS(F + c, y + c) == CRPS(F, y)`` - so once each
    entry's draws are put on the triangle's own basis against the matching
    outcome, an increment-drawing entry and a cumulative-drawing one produce
    directly comparable numbers. Verified to 5.5e-12 on lognormal draws.

    :meth:`predict_at` takes a :class:`~ibnr.kernels.holdout.HoldoutCells` and
    not a bare ``CellIndex``, unlike :meth:`ScoresHeldout.log_lik_at`. The target
    basis is a property of the *triangle* and only ``HoldoutCells`` carries it; a
    ``target_scale=`` argument would be a knob whose wrong setting is precisely
    the 996-versus-3.4 error above.
    """

    #: the basis :meth:`_draws_native` returns. One of :data:`DRAW_SCALES`.
    heldout_draw_scale: ClassVar[str]

    @abstractmethod
    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        """``(n_draws, n_cells)`` predictive draws on this entry's own scale.

        One draw per posterior draw - the posterior predictive, not a plug-in at
        the posterior mean. ``rng`` is supplied so the caller controls the seed.
        """

    def predict_at(
        self,
        cells: HoldoutCells,
        *,
        field: str | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """``(n_draws, n_cells)`` draws on the TRIANGLE's basis, in cell order.

        Aligned with :attr:`~ibnr.kernels.holdout.HoldoutCells.values`, so the
        caller scores against those and never has to know what the entry drew.
        """
        if not isinstance(cells, HoldoutCells):
            raise TypeError(
                "predict_at needs a HoldoutCells: the draws must be carried to the "
                "TRIANGLE's basis and only HoldoutCells records which that is. A bare "
                f"CellIndex cannot say; got {type(cells).__name__}"
            )
        scale = getattr(self, "heldout_draw_scale", None)
        if scale not in DRAW_SCALES:
            raise ValueError(
                f"{type(self).__name__}.heldout_draw_scale must be one of "
                f"{list(DRAW_SCALES)}, got {scale!r}"
            )

        idx = index_into(cells, self.contract_, field=field)
        rng = np.random.default_rng(seed)
        draws = np.asarray(self._draws_native(idx, rng=rng), dtype=float)
        if draws.ndim != 2 or draws.shape[1] != idx.n_cells:
            raise ValueError(
                f"{type(self).__name__}._draws_native returned {draws.shape} for "
                f"{idx.n_cells} cells; expected (n_draws, {idx.n_cells})"
            )
        if draws.shape[0] < 2:
            raise ValueError(
                f"{type(self).__name__}._draws_native returned {draws.shape[0]} draw(s); "
                "CRPS needs at least 2, and one draw is a plug-in estimate rather than a "
                "predictive distribution"
            )
        if scale == cells.measure:
            return draws

        # The only conversion either direction is the training-diagonal anchor.
        anchor = np.asarray(idx.prev_value, dtype=float)
        if np.isnan(anchor).any():
            raise ValueError(
                f"{type(self).__name__} draws {scale} values but these cells are "
                f"{cells.measure!r}, and the conversion needs each cell's predecessor - "
                f"{int(np.isnan(anchor).sum())} of {len(anchor)} are missing. On an "
                "incremental triangle there is no predecessor to add at all, so an entry "
                "drawing cumulatives cannot be scored on one"
            )
        return draws + anchor if cells.measure == "cumulative" else draws - anchor
