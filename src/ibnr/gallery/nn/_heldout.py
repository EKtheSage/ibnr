"""Bridge from the multi-cohort NN contract to the held-out scoring machinery.

``kernels.holdout.index_into`` demands a single-cohort contract carrying an
identity block (``segment``/``fields``/``models``/``measure``) plus the
``(w, d)`` training cells, none of which ``kernels.nn_contract.nn_data``
provides - its dict is deliberately multi-cohort. Rather than teach
``index_into`` a second contract shape (and re-implement its cohort-identity
and training-overlap guards), :func:`cohort_contract` builds a PER-COHORT
adapter dict from the pooled contract so ``index_into`` works unchanged and
every one of its refusals - wrong cohort, wrong segment schema, wrong measure,
unknown origin, cells in the fit's own training data - applies to an NN entry
exactly as it does to a Stan one.

:class:`CohortHeldout` is the matching scorer view: a pooled NN fit scores one
cohort at a time (``next_diagonal`` builds cells one cohort at a time on
principle), so the entry hands out a light per-cohort object that IS a
``ScoresHeldout``/``PredictsHeldout`` - its ``log_lik_at``/``predict_at`` are
the unmodified base-class implementations, so the measure carry, the
draw-scale conversion and every shape check happen in exactly one place.

Torch-free: the view delegates the actual forward passes back to the entry,
whose methods import torch lazily. ``ibnr.gallery`` must import without the
``[nn]`` extra (subprocess-tested in ``tests/test_gallery.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout
from ibnr.kernels.holdout import CellIndex

__all__ = ["CohortHeldout", "cohort_contract"]


def cohort_contract(contract: dict, cohort: int, *, models: Sequence[str]) -> dict:
    """One cohort of an ``nn_data`` contract, in ``index_into``'s shape.

    contract: the pooled dict from ``kernels.nn_contract.nn_data``.
    cohort:   row index into ``contract["cohorts"]`` (axis 0 of every array).
    models:   the field(s) this entry puts a likelihood on - for the MDN
              entries the target channel, ``contract["fields"][0]``.

    The training cells declared as ``(w, d)`` are the union of the cohort's
    usable increments (``obs_mask``) and its per-origin anchors
    (``latest_dev``): an anchor cell whose own increment was unusable (a
    predecessor hole) is absent from ``obs_mask``, but its VALUE was training
    data all the same - the rollout and the held-out draws are anchored on it -
    so scoring it as "held out" would report in-sample fit. ``premium`` is the
    cohort's per-origin booked premium, 1-D as ``index_into`` indexes it.

    ``measure`` is ``"cumulative"`` unconditionally because ``nn_data`` refuses
    anything else at construction.
    """
    cohorts = contract["cohorts"]
    ci = int(cohort)
    if not 0 <= ci < len(cohorts):
        raise IndexError(f"cohort {ci} out of range; the contract has {len(cohorts)}")
    row = cohorts.iloc[ci]
    segment = {col: row[col] for col in cohorts.columns}

    obs = np.asarray(contract["obs_mask"][ci], dtype=bool)  # (n_w, n_d)
    w_obs, d_obs = np.nonzero(obs)
    trained = set(zip((w_obs + 1).tolist(), (d_obs + 1).tolist(), strict=True))
    latest = np.asarray(contract["latest_dev"][ci], dtype=int)  # (n_w,) 1-based, 0 = none
    for w0 in np.nonzero(latest > 0)[0]:
        trained.add((int(w0) + 1, int(latest[w0])))
    pairs = sorted(trained)
    return {
        "segment": segment,
        "fields": list(contract["fields"]),
        "models": list(models),
        "measure": "cumulative",
        "origin_periods": list(contract["origin_periods"]),
        "dev_grain_months": int(contract["dev_grain_months"]),
        "n_w": int(contract["n_w"]),
        "n_d": int(contract["n_d"]),
        "w": np.array([p[0] for p in pairs], dtype=int),
        "d": np.array([p[1] for p in pairs], dtype=int),
        "premium": np.asarray(contract["premium"][ci], dtype=float),  # (n_w,)
    }


class CohortHeldout(ScoresHeldout, PredictsHeldout):
    """Per-cohort held-out scorer view over a pooled NN fit.

    Constructed by the entry (``entry.at_cohort(segment)``), never directly.
    ``log_lik_at`` and ``predict_at`` are inherited from the mixins untouched:
    they call ``index_into`` with this view's per-cohort :func:`cohort_contract`
    (so the identity and training-overlap guards apply) and do the measure
    carry / draw-scale conversion in the base class. Only the two native hooks
    delegate back to the entry, which knows how to run its network for one
    cohort:

    - ``entry._heldout_log_lik(cohort, cells) -> (n_members, n_cells)``
    - ``entry._heldout_draws(cohort, cells, rng=...) -> (n_draws, n_cells)``

    ``heldout_measure`` / ``heldout_draw_scale`` are read off the ENTRY class
    so an entry declares its scales exactly once.
    """

    def __init__(self, entry: Any, cohort: int) -> None:
        self._entry = entry
        self._cohort = int(cohort)
        # instance attributes shadow the mixins' ClassVars; the entry is the
        # single source of truth for both declarations
        self.heldout_measure = entry.heldout_measure
        self.heldout_draw_scale = entry.heldout_draw_scale
        self.contract_ = cohort_contract(
            entry.contract_, cohort, models=(entry._loss_field,)
        )

    @property
    def cohort(self) -> int:
        return self._cohort

    @property
    def segment(self) -> dict:
        return dict(self.contract_["segment"])

    def _log_lik_native(self, cells: CellIndex) -> np.ndarray:
        return self._entry._heldout_log_lik(self._cohort, cells)

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        return self._entry._heldout_draws(self._cohort, cells, rng=rng)

    def training_cells(self) -> CellIndex:
        raise NotImplementedError(
            "the NN contract carries loss ratios, not per-cell loss values, so the "
            "in-sample agreement gate's CellIndex cannot be built from it; the fast "
            "closed-form tests play that role for the NN entries"
        )
