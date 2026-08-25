"""Held-out adapter for the multi-line NN contract: one (company, line) pair.

``gallery/nn/_heldout.py`` maps one cohort of the flat ``nn_data`` contract into
the shape ``kernels.holdout.index_into`` demands. That works for every NN entry
whose cohort is already a (company, line) pair, and it is exactly what
``nn_transformer_ml`` is not: its cohort is a COMPANY, all of whose lines are
one training example, while a held-out cohort built by ``next_diagonal`` is a
single (company, line) pair. This module is the sibling that closes the gap -
the adapter, not a second copy of the scoring code, which still lives next door
and is called from here.

Three pieces, mirroring the single-line module one for one:

:func:`company_line_contract`
    The ``index_into``-shaped dict for one (company, line) pair, sliced out of
    the company-shaped contract. Same three-set training closure as
    ``cohort_contract``, same refusals downstream.
:func:`company_line_cutoff`
    Where as_of sits for the line being scored.
:class:`MLCohortHeldout`
    The per-pair scorer view, whose ``log_lik_at``/``predict_at`` are the
    unmodified base-class implementations.

Torch-free at MODULE level, like everything else on this path: the entry
imports torch inside the methods that run a forward pass, and ``ibnr.gallery``
must import without the ``[nn]`` extra (subprocess-tested in
``tests/test_gallery.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout
from ibnr.kernels.holdout import CellIndex

__all__ = ["MLCohortHeldout", "company_line_contract", "company_line_cutoff"]

#: the segment column the line axis is keyed on. The company contract drops it
#: from its own key (a company is a cohort key with the line removed), so the
#: adapter has to put it back to name a held-out cohort.
LOB_COLUMN = "line_of_business"


def _resolve_pair(contract: dict, company: int, line: int) -> tuple[int, int]:
    """Range-check a (company, line) pair and refuse a line the company omits.

    A company that does not write a line has all-zero arrays and a False
    ``line_mask`` there (padding, not data), so every array slice below would
    return zeros and the scoring path would answer with plausible numbers for a
    triangle that does not exist.
    """
    companies = contract["companies"]
    ci = int(company)
    if not 0 <= ci < len(companies):
        raise IndexError(f"company {ci} out of range; the contract has {len(companies)}")
    levels = list(contract["lob_levels"])
    li = int(line)
    if not 0 <= li < len(levels):
        raise IndexError(f"line {li} out of range; the contract has {len(levels)}")
    if not bool(contract["line_mask"][ci, li]):
        written = [levels[j] for j in np.nonzero(contract["line_mask"][ci])[0]]
        raise ValueError(
            f"company {dict(companies.iloc[ci])} does not write {levels[li]!r}; it writes "
            f"{written}. That line's arrays are padding zeros, so scoring it would return "
            "numbers for a triangle the fit never saw"
        )
    return ci, li


def company_line_cutoff(contract: dict, company: int, line: int) -> int:
    """The SCORED LINE's own as_of calendar diagonal: the deepest one it HELD a
    cell on.

    **The conditioning convention this belongs to, in one sentence:** the
    forward context is everything the COMPANY observed at as_of across ALL its
    lines, while this cutoff scalar is the SCORED LINE's own as_of diagonal.

    Both halves are deliberate and they answer different questions. The context
    is gated by observedness alone - the rollout's own rule - because the whole
    point of this entry is that one line's held-out cell is informed by what the
    company's other lines have already reported. The cutoff, by contrast, feeds
    the relative calendar embedding, whose job is to place the cell being
    predicted one step past the conditioning boundary: distance 1, the
    most-supervised position and the rollout's first step. Reading it off the
    company's deepest line instead would put a slower-reporting line's held-out
    diagonal two or three steps out, into a different embedding row and
    therefore a different predictive distribution. On complete squares every
    line of a company reaches the same diagonal and the two readings coincide.

    The value itself is the max calendar index over the line's ``obs_mask``
    UNION its per-origin anchors, the same union
    ``gallery.nn._heldout.heldout_cutoff`` takes on the flat contract and for
    the same reason: ``obs_mask`` marks usable *increments*, so an anchor whose
    predecessor is missing is absent from it even though the line plainly held
    that cell.
    """
    ci, li = _resolve_pair(contract, company, line)
    obs = np.array(contract["obs_mask"][ci, li], dtype=bool)  # (n_w, n_d); copied to mutate
    latest = np.asarray(contract["latest_dev"][ci, li], dtype=int)  # (n_w,) 1-based, 0 = none
    anchored = np.nonzero(latest > 0)[0]
    obs[anchored, latest[anchored] - 1] = True
    if not obs.any():
        raise ValueError(
            f"({ci}, {li}) has no observed cell and no anchor, so there is no as_of "
            "diagonal to condition on"
        )
    return int(np.asarray(contract["cal_idx"])[obs].max())


def company_line_contract(
    contract: dict, company: int, line: int, *, models: Sequence[str]
) -> dict:
    """One (company, line) pair of an ``nn_company_data`` contract, in
    ``index_into``'s shape.

    contract: the pooled dict from ``kernels.nn_contract.nn_company_data``.
    company:  row index into ``contract["companies"]`` (axis 0 of every array).
    line:     index into ``contract["lob_levels"]`` (axis 1 of every array).
    models:   the field(s) this entry puts a likelihood on - for a mixture-head
              NN entry the target channel, ``contract["fields"][0]``.

    ``segment`` is the company's key columns plus ``line_of_business``, which is
    what makes this a held-out cohort at all: the fit is keyed on the company
    and ``next_diagonal`` builds cells on the full (company, line) identity, so
    the adapter is where the line rejoins the key. Everything downstream then
    works unchanged - ``index_into`` compares this segment against the cells'
    and refuses another company, another line, another field or another measure
    exactly as it does for a Stan fit.

    The training cells declared as ``(w, d)`` are the same honest closure the
    flat adapter declares, taken on this line's slices: the line's usable
    increments (``obs_mask``); its per-origin anchors (``latest_dev``); and each
    observed cell's immediate predecessor ``(w, d - 1)``. The latter two cover
    cells whose VALUE was training information even though their own increment
    was unusable. Scoring any of them as "held out" would report in-sample fit.

    Note what is NOT in the closure: the company's OTHER lines. They were
    training data too, and the forward pass conditions on them - but they sit at
    a different ``line_of_business``, so they are a different cohort with its
    own held-out diagonal, and ``index_into`` would refuse a cell of theirs on
    identity before the overlap check ever ran.

    ``premium`` is this pair's per-origin booked premium, 1-D as ``index_into``
    indexes it. ``measure`` is ``"cumulative"`` unconditionally because
    ``nn_data`` - which ``nn_company_data`` regroups - refuses anything else at
    construction.
    """
    ci, li = _resolve_pair(contract, company, line)
    row = contract["companies"].iloc[ci]
    segment = {col: row[col] for col in contract["companies"].columns}
    segment[LOB_COLUMN] = contract["lob_levels"][li]

    obs = np.asarray(contract["obs_mask"][ci, li], dtype=bool)  # (n_w, n_d)
    w_obs, d_obs = np.nonzero(obs)
    trained = set(zip((w_obs + 1).tolist(), (d_obs + 1).tolist(), strict=True))
    # predecessor closure: an obs cell's increment was differenced against
    # (w, d - 1), so that cell's value is training information even when its
    # own increment was unusable (0-based d_obs IS the 1-based predecessor dev)
    trained |= {(int(w) + 1, int(d)) for w, d in zip(w_obs, d_obs, strict=True) if d >= 1}
    latest = np.asarray(contract["latest_dev"][ci, li], dtype=int)  # (n_w,) 1-based, 0 = none
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
        "premium": np.asarray(contract["premium"][ci, li], dtype=float),  # (n_w,)
    }


class MLCohortHeldout(ScoresHeldout, PredictsHeldout):
    """Per-(company, line) held-out scorer view over a pooled multi-line fit.

    Constructed by the entry (``entry.at_cohort(segment)``), never directly.
    ``log_lik_at`` and ``predict_at`` are inherited from the mixins untouched:
    they call ``index_into`` with this view's :func:`company_line_contract` (so
    the identity and training-overlap guards apply) and do the measure carry and
    the increment-to-cumulative anchoring in the base class. Only the two native
    hooks delegate back to the entry, which knows how to run its network for one
    company and read one line out of the result:

    - ``entry._heldout_log_lik((ci, li), cells) -> (n_members, n_cells)``
    - ``entry._heldout_draws((ci, li), cells, rng=...) -> (n_draws, n_cells)``

    ``heldout_measure`` / ``heldout_draw_scale`` are read off the ENTRY class so
    an entry declares its scales exactly once.
    """

    def __init__(self, entry: Any, company: int, line: int) -> None:
        self._entry = entry
        self._company = int(company)
        self._line = int(line)
        # instance attributes shadow the mixins' ClassVars; the entry is the
        # single source of truth for all three declarations. `name` is what the
        # base class's refusals identify themselves by, and a user who asked for
        # "nn_transformer_ml" should read that back, not the name of a view they
        # never built.
        self.name = entry.name
        self.heldout_measure = entry.heldout_measure
        self.heldout_draw_scale = entry.heldout_draw_scale
        self.contract_ = company_line_contract(
            entry.contract_, company, line, models=(entry._loss_field,)
        )

    @property
    def company(self) -> int:
        return self._company

    @property
    def line(self) -> int:
        return self._line

    @property
    def cohort(self) -> tuple[int, int]:
        """The ``(company, line)`` pair this view answers for."""
        return (self._company, self._line)

    @property
    def segment(self) -> dict:
        return dict(self.contract_["segment"])

    def _cell_identity(self) -> dict:
        """This pair's FULL segment identity, wider than its contract key.

        The company contract keys on ``nn_company_data``'s company columns,
        which exclude display-only segments, while ``next_diagonal`` builds
        cells on all of the triangle's. ``_keyed_to_fit`` narrows the cells onto
        the contract's key and CHECKS each dropped column against this dict on
        the way - so ``at_cohort(segment).log_lik_at(cells)`` verifies a display
        value like ``company_name`` rather than discarding it, which is the
        whole reason it is not simply ignored.

        The entry's ``cohorts()`` is company-level, so the line has to be added
        here; without it, ``line_of_business`` would be an extra column with no
        entry in the identity and the narrowing would refuse every call.
        """
        identity = dict(self._entry.cohorts()[self._company])
        identity[LOB_COLUMN] = self.contract_["segment"][LOB_COLUMN]
        return identity

    def _log_lik_native(self, cells: CellIndex) -> np.ndarray:
        return self._entry._heldout_log_lik(self.cohort, cells)

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        return self._entry._heldout_draws(self.cohort, cells, rng=rng)

    def training_cells(self) -> CellIndex:
        raise NotImplementedError(
            "the NN contract carries loss ratios, not per-cell loss values, so the "
            "in-sample agreement gate's CellIndex cannot be built from it; the fast "
            "closed-form tests play that role for the NN entries"
        )
