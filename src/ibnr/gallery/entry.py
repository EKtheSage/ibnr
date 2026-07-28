"""GalleryEntry: the contract every gallery model must satisfy.

`.fit()`, `.cohorts()`, `.predict()`, `.realized_ultimates()`, `.evaluate()` and
`.card()` are mandatory. Evaluation logic lives in ``kernels`` - entries call it,
never reimplement it. An entry that cannot satisfy this interface does not
register.

**A fitted entry says which cohorts it answers for.** :meth:`GalleryEntry.cohorts`
is the vocabulary the rest of the contract is written in: ``predict``,
``realized_ultimates`` and ``evaluate`` all take the same ``segment`` argument
and mean the same thing by it on every entry in the gallery, so::

    for seg in entry.cohorts():
        pred = entry.predict(segment=seg)
        outcome = entry.realized_ultimates(full_triangle, segment=seg)

is one loop over any entry, with no ``family == "nn"`` branch. Before 0.5.0 the
NN entries took a segment dict and everything else took none, and
``realized_ultimates`` was not on the ABC at all - so a cross-model outcome
table depended on an undeclared convention plus caller-side family branching.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ibnr.kernels.densities import MEASURES, to_amount_scale
from ibnr.kernels.holdout import CellIndex, HoldoutCells, index_into, training_index
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.kernels.scores import crps
from ibnr.triangle.core import Triangle


class GalleryEntry(ABC):
    name: ClassVar[str]
    family: ClassVar[str]  # "bayesian" | "nn" | "statistical" | "deterministic"

    #: the dataclass a caller builds to configure ``fit(config=...)``, or None
    #: when the entry takes no config object. On the CLASS, so it is reachable
    #: from ``gallery.get(name)`` without constructing or fitting anything - the
    #: same reason ``get()`` returns the class rather than an instance::
    #:
    #:     cfg = gallery.get("mdn").config_class(ensemble_size=1)
    #:
    #: ``registry.register`` checks the declaration BOTH ways: an entry whose
    #: ``fit`` takes ``config=`` must declare one (or ``gallery.get(name)`` gives
    #: a caller no route to the type it must build), and an entry that declares
    #: one whose ``fit`` does not take ``config=`` is refused too, because a
    #: stale declaration is exactly as misleading as a missing one.
    config_class: ClassVar[type | None] = None

    @abstractmethod
    def fit(self, triangle: Triangle, **kwargs) -> GalleryEntry:
        """Fit on a training triangle (already sliced with as_of)."""

    @abstractmethod
    def cohorts(self) -> list[dict[str, Any]]:
        """Every cohort this fit answers for, in :meth:`predict`'s target order.

        Length 1 for a single-cohort fit, one entry per pooled cohort otherwise,
        so ``for seg in entry.cohorts(): entry.predict(segment=seg)`` loops over
        any entry in the gallery with no knowledge of its family.

        Each dict is the cohort's FULL segment identity as the triangle carried
        it - **including any column the fit's own cohort KEY does not carry**
        (``kernels.nn_contract.DISPLAY_COLUMNS``: a pooled NN fit is keyed on
        ``company_code`` and drops ``company_name``, so that two spellings of one
        company do not become two cohorts). A caller must not be told a cohort is
        unidentifiable when the fit knows exactly which one it is.

        Abstract rather than defaulted to ``contract_["segment"]``: a pooled
        entry that forgot to override would then get a plausible answer from a
        key that happens to exist, which is the entries-cloned-from-stale-templates
        failure. ``register()`` refuses an entry that skips it.
        """

    def cohort_index(self, segment: Mapping[str, Any] | None) -> int | None:
        """Which of :meth:`cohorts` a ``segment=`` argument names, or None.

        ``None`` in, ``None`` out - the entry then does whatever ``segment=None``
        means for it. Otherwise every supplied pair must match and exactly one
        cohort must survive: ``segment`` is a **filter on this fit's cohorts**,
        not a description of a triangle. A subset that identifies one cohort is
        accepted, which is what lets one loop pass the same dict to a fit keyed
        on ``(company_code, company_name, line_of_business)`` and to a pooled fit
        keyed on ``(company_code, line_of_business)``.

        Implemented once here and used by all 15 entries, which is why a mismatch
        reports the same way everywhere: naming the fit's own cohort key, and
        never silently scoring the fitted cohort for a typo'd one.
        """
        if segment is None:
            return None
        if not isinstance(segment, Mapping):
            raise TypeError(
                f"{self.name}: segment must be a mapping of segment column -> value, "
                f"got {type(segment).__name__}"
            )
        cohorts = self.cohorts()
        schema = list(cohorts[0]) if cohorts else []
        unknown = [k for k in segment if k not in schema]
        if unknown:
            raise KeyError(
                f"{self.name}: unknown segment column(s) {unknown}; this fit's cohort key "
                f"is {schema}. A segment names a cohort of THIS fit, so the columns must "
                "be the fit's own - not the triangle's, which can be wider"
            )
        hits = [i for i, c in enumerate(cohorts) if all(c[k] == v for k, v in segment.items())]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            more = f" ... ({len(cohorts) - 3} more)" if len(cohorts) > 3 else ""
            raise ValueError(
                f"{self.name}: segment {dict(segment)} matches 0 cohorts of this fit. It "
                f"is keyed on {schema} and covers {cohorts[:3]}{more}"
            )
        raise ValueError(
            f"{self.name}: segment {dict(segment)} matches {len(hits)} cohorts, need "
            f"exactly 1. This fit is keyed on {schema}; e.g. {cohorts[hits[0]]} and "
            f"{cohorts[hits[1]]} both match - name enough columns to identify one"
        )

    @abstractmethod
    def predict(self, segment: Mapping[str, Any] | None = None, **kwargs) -> PredictiveDistribution:
        """Predictive distribution of the fitted quantities (e.g. ultimates).

        ``segment`` names one cohort of this fit (see :meth:`cohorts`), and means
        the same thing on every entry in the gallery: the returned targets
        describe that cohort and nothing else. A single-cohort fit accepts None
        or a matching key and returns the identical object either way; a pooled
        fit returns that one cohort's targets plus its total, and returns the
        whole panel (no total - a total across companies is not a quantity) when
        the segment is omitted.

        A segment that names no cohort of this fit raises rather than being
        ignored: accepting and discarding it would make a typo'd cohort key
        score the fitted cohort and return a plausible number.
        """

    @abstractmethod
    def realized_ultimates(
        self, full_triangle: Triangle, segment: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        """Outcomes aligned, element for element, to ``predict(segment)``'s targets.

        Read from the FULL (unsliced) triangle, restricted to the origins the
        training slice actually had - the Schedule P mart carries accident years
        past the study window and an unrestricted aggregate is silently inflated
        (measured at 2.4x once, in ``scripts/compare_gallery.py``).

        On the ABC because it is what makes a cross-model outcome table possible:
        it was implemented on all 15 entries under a convention nothing enforced,
        and the convention had already drifted across two families.
        """

    def evaluate(self, observed, segment: Mapping[str, Any] | None = None) -> dict:
        """Score realized outcomes against the predictive distribution.

        Default implementation (kernels-backed): the Meyers-style summary
        table plus the outcome percentile of each target. Richer harnesses
        (ELPD, stacking) extend this in kernels, not in entries.

        ``segment`` is passed straight to :meth:`predict`, so it selects exactly
        the same cohort here as there. Without it a pooled fit could only ever be
        evaluated on its whole panel.
        """
        pred = self.predict(segment=segment)
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


class _HeldoutCapability(ABC):  # noqa: B024 - ABC for the metaclass, not for abstract methods
    """Shared plumbing of :class:`ScoresHeldout` and :class:`PredictsHeldout`.

    One job: put the supplied cells on the fit's OWN segment schema before
    ``index_into`` sees them, so that check can keep comparing schemas for
    equality. That equality is what stops a fit keyed on ``lob`` alone from
    vouching for cells keyed on ``(lob, company)`` - cells that would index
    cleanly and score the wrong cohort - so it is narrowed *around*, never
    relaxed.

    The case that needs it is a pooled NN fit: ``kernels.nn_contract`` keeps
    display-only segment columns out of the cohort key (two spellings of one
    company must not become two cohorts), so its key is a strict subset of the
    triangle's segments while ``next_diagonal`` builds cells on all of them.
    Before 0.5.0 that combination simply had no working route - ``log_lik_at``
    raised "unknown segment column 'company_name'", naming a column the caller
    had passed, and binding the cohort by hand failed one level deeper inside
    ``index_into``.
    """

    def _cell_identity(self) -> dict:
        """Full segment identity of the cohort this scorer answers for, including
        columns the fit's KEY does not carry.

        Default: the contract's own key, which IS the identity for every
        single-cohort fit. ``gallery/nn/_heldout.CohortHeldout`` overrides it
        with the pooled entry's ``cohorts()[i]``, which is where a dropped
        display value actually gets checked.
        """
        contract = getattr(self, "contract_", None)
        if not isinstance(contract, dict) or "segment" not in contract:
            raise RuntimeError(
                f"{type(self).__name__} carries no single-cohort identity, so held-out "
                "cells cannot be re-keyed onto it; bind a cohort first"
            )
        return dict(contract["segment"])

    def _keyed_to_fit(self, cells):
        """``cells`` re-keyed onto this fit's own segment schema.

        Unchanged when the schemas already agree (every single-cohort entry, and
        a caller who already dropped the display column). When the fit's key is a
        strict SUBSET of the cells', each extra column's value is **checked**
        against :meth:`_cell_identity` and then dropped - verified, never
        discarded, because ``entry.at_cohort(segment).log_lik_at(cells)`` does
        not otherwise look at those columns at all. When the fit is keyed on
        something the cells lack, the cells pass through untouched so
        ``index_into`` raises its own message, which already names both schemas.

        Narrowing drops columns, not rows, so the ``(n_draws, n_cells)`` column
        order is untouched - and it is internal to ``log_lik_at``/``predict_at``,
        so the caller keeps handing the original wide ``HoldoutCells`` to
        ``CohortForecast`` and ``align_panel`` still sees one schema across every
        model on the board.
        """
        if not isinstance(cells, HoldoutCells):
            return cells  # a bare CellIndex is already in the fit's index space
        contract = getattr(self, "contract_", None)
        if not isinstance(contract, dict) or "segment" not in contract:
            return cells
        fitted = list(contract["segment"])
        have = list(cells.segments)
        if set(fitted) == set(have) or not set(fitted) < set(have):
            return cells

        identity = self._cell_identity()
        who = getattr(self, "name", None) or type(self).__name__
        frame = cells.frame
        for column in (c for c in have if c not in fitted):
            if column not in identity:
                raise ValueError(
                    f"{who}: held-out cells are keyed on {have} and this "
                    f"fit's contract on {fitted}. The extra column {column!r} is not part "
                    "of this fit's cohort identity either, so dropping it would be "
                    "unchecked - and a cell that belongs to another cohort along it would "
                    "index cleanly and score this one"
                )
            values = sorted({str(v) for v in frame[column].unique()})
            if values != [str(identity[column])]:
                raise ValueError(
                    f"{who}: held-out cells are keyed on {have} and this "
                    f"fit's contract on {fitted}. The extra column(s) can be dropped only "
                    f"if they agree with the fitted cohort, and {column!r} does not: the "
                    f"cells say {', '.join(values)} where this fit's cohort is "
                    f"{identity[column]!r}"
                )
        return cells.narrowed_to([c for c in have if c in fitted])


class ScoresHeldout(_HeldoutCapability):
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
            else index_into(self._keyed_to_fit(cells), self.contract_, field=field)
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


class PredictsHeldout(_HeldoutCapability):
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

        idx = index_into(self._keyed_to_fit(cells), self.contract_, field=field)
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
