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

:class:`PooledMDNHeldout` is the ENTRY side of that same protocol, and the two
halves are why held-out scoring is written once rather than once per entry.
Every NN entry in the gallery is the same shape - a pooled multi-cohort fit
whose per-cell predictive is a Gaussian mixture over the standardized
incremental loss ratio - so the cohort resolution, the density algebra, the
member-draw split and the pinned-dev guards are identical code. What actually
differs between entries is TWO things, and they are the mixin's abstract
hooks: how a cohort's forward inputs are assembled (``_heldout_inputs``) and
how the network is called to get mixture parameters out (``_forward_mixture``,
which absorbs both the argument list and the return shape - the transformer
returns the 3-tuple bare, DeepTriangle returns it alongside its auxiliary
head). Entry-identifying text in the error messages reads ``self.name``, so
an entry parameterizes those by existing rather than by declaring anything.

Torch-free at MODULE level: every forward pass imports torch inside the method
that runs it, here as much as in the entries. ``ibnr.gallery`` must import
(and NN entries must register) without the ``[nn]`` extra - subprocess-tested
in ``tests/test_gallery.py``.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import logsumexp

from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout
from ibnr.kernels.holdout import CellIndex, HoldoutCells

__all__ = ["CohortHeldout", "PooledMDNHeldout", "cohort_contract", "heldout_cutoff"]

LOG_2PI = math.log(2.0 * math.pi)


def heldout_cutoff(contract: dict, cohort: int) -> int:
    """The cohort's as_of calendar diagonal: the deepest one it HELD a cell on.

    Every entry whose network reads a relative calendar position derives it as
    ``cal_idx - cutoff``, so this scalar is what places the held-out diagonal
    at distance 1 - the most-supervised position, and the one ``_rollout``
    steps through (``cut_b = lv - 1``).

    It is the max over ``obs_mask`` UNION the per-origin anchors, and the union
    is the whole point: ``obs_mask`` marks usable *increments*, so an anchor
    whose predecessor is missing is absent from it even though the cohort
    plainly held that cell - the same fact :func:`cohort_contract` unions
    ``latest_dev`` back in for when it declares the training cells. Reading the
    cutoff off ``obs_mask`` alone parked as_of at the last hole-free diagonal
    instead: a cohort with cumulative values at devs {1, 2, 4} was cut at dev
    2's diagonal, so its held-out cell arrived three diagonals out rather than
    one, into a different ``dist_emb`` row and therefore a different predictive.
    Anchors are the same quantity ``_rollout`` reads from ``latest_dev`` for its
    future mask, so entry and rollout now agree on where as_of sits.

    Predecessors - ``cohort_contract``'s third training set - cannot extend the
    boundary, sitting one dev step before a cell already counted, so obs union
    anchors is all of it. The context mask stays ``obs_mask`` alone and is not
    widened to match: ``x`` carries incremental ratios, and a hole-anchored
    cell has no usable increment to condition on, only a cumulative value the
    network has no channel for.
    """
    obs = np.array(contract["obs_mask"][cohort], dtype=bool)  # (n_w, n_d); copied to mutate
    latest = np.asarray(contract["latest_dev"][cohort], dtype=int)  # (n_w,) 1-based, 0 = none
    anchored = np.nonzero(latest > 0)[0]
    obs[anchored, latest[anchored] - 1] = True
    # nn_data screens out cohorts with no usable increment, so obs is non-empty
    return int(np.asarray(contract["cal_idx"])[obs].max())


def cohort_contract(contract: dict, cohort: int, *, models: Sequence[str]) -> dict:
    """One cohort of an ``nn_data`` contract, in ``index_into``'s shape.

    contract: the pooled dict from ``kernels.nn_contract.nn_data``.
    cohort:   row index into ``contract["cohorts"]`` (axis 0 of every array).
    models:   the field(s) this entry puts a likelihood on - for the MDN
              entries the target channel, ``contract["fields"][0]``.

    The training cells declared as ``(w, d)`` are the honest closure of what
    the fit consumed, the union of three sets: the cohort's usable increments
    (``obs_mask``); its per-origin anchors (``latest_dev``); and each obs
    cell's immediate predecessor ``(w, d - 1)``. The latter two cover cells
    whose VALUE was training information even though their own increment was
    unusable: an anchor with a predecessor hole is absent from ``obs_mask``
    yet the rollout and the held-out draws are anchored on it, and with
    cumulative values at devs {1, 2, 4, 5} the dev-4 cell feeds the dev-5
    increment while being neither obs nor anchor. Scoring any of these as
    "held out" would report in-sample fit, so they are declared here rather
    than left to downstream guards (``next_diagonal`` happens to exclude the
    dev-4 case as ``no_predecessor`` today, but the overlap check should not
    depend on that). ``premium`` is the cohort's per-origin booked premium,
    1-D as ``index_into`` indexes it.

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
    # predecessor closure: an obs cell's increment was differenced against
    # (w, d - 1), so that cell's value is training information even when its
    # own increment was unusable (0-based d_obs IS the 1-based predecessor dev)
    trained |= {(int(w) + 1, int(d)) for w, d in zip(w_obs, d_obs, strict=True) if d >= 1}
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
        # single source of truth for all three declarations. `name` is what the
        # base class's refusals identify themselves by, and a user who asked for
        # "mdn" should read "mdn" back, not the name of a view they never built.
        self.name = entry.name
        self.heldout_measure = entry.heldout_measure
        self.heldout_draw_scale = entry.heldout_draw_scale
        self.contract_ = cohort_contract(entry.contract_, cohort, models=(entry._loss_field,))

    @property
    def cohort(self) -> int:
        return self._cohort

    @property
    def segment(self) -> dict:
        return dict(self.contract_["segment"])

    def _cell_identity(self) -> dict:
        """This cohort's FULL segment identity, wider than its contract key.

        ``cohort_contract`` keys on ``nn_data``'s cohort columns, which exclude
        display-only segments, while ``next_diagonal`` builds cells on all of the
        triangle's. ``_keyed_to_fit`` narrows the cells onto the contract's key
        and CHECKS each dropped column against this dict on the way - so
        ``at_cohort(segment).log_lik_at(cells)`` verifies the display value
        rather than discarding it, which is the whole reason it is not simply
        ignored.
        """
        return dict(self._entry.cohorts()[self._cohort])

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


class PooledMDNHeldout(ScoresHeldout, PredictsHeldout):
    """Held-out scoring for a pooled multi-cohort NN fit with an MDN head.

    Mixed into an NN gallery entry (beside ``GalleryEntry``), this supplies the
    whole held-out surface: ``at_cohort``/``log_lik_at``/``predict_at``, the
    two native hooks the ABCs demand, and the per-cohort machinery
    :class:`CohortHeldout` calls back into. See the module docstring for why
    it is shared.

    **What the entry must provide.** The two abstract hooks below, plus the
    attributes every NN entry already has: ``contract_`` (the pooled
    ``nn_data`` dict), ``models_`` (the ensemble), ``norm_`` (per-(channel,
    dev) normalization with its ``pinned`` mask), ``config_.n_draws``,
    ``_device``, ``_loss_field``, and the class-level ``name``,
    ``heldout_measure`` and ``heldout_draw_scale`` declarations. The scale
    declarations stay on the ENTRY rather than here because they are claims
    about that entry's head, and the base classes read them to run the measure
    carry and the increment->cumulative anchoring.

    **The density's draw axis is the ENSEMBLE MEMBERS**, not posterior draws:
    ``logmeanexp`` over it is the ensemble-average predictive density, the
    deep-ensemble analogue of averaging a posterior's per-draw likelihoods.

    **The exposure is the CONTRACT's**, on both hooks: they divide and multiply
    by the premium the fit standardized against and merely VERIFY the caller's
    against it (:meth:`_cell_premium`) - the rule ``guszcza_growth_curve`` and
    ``compartmental`` state on the Bayesian side.
    """

    # -- per-entry hooks ---------------------------------------------------------

    @abstractmethod
    def _heldout_inputs(self, ci: int) -> dict:
        """One cohort's forward inputs, conditioned on everything it had at
        as_of, as a dict of already-batched torch tensors.

        Entry-specific because the conditioning IS the architecture: what the
        network needs to place the held-out diagonal one step past the
        cohort's observed context differs by entry (a calendar cutoff level,
        a company embedding index, ...). The contract this must honour is only
        that the returned inputs describe a single cohort with a leading batch
        axis of 1, and that :meth:`_forward_mixture` accepts them.
        """

    @abstractmethod
    def _forward_mixture(self, model: Any, inputs: dict) -> tuple:
        """``(log_pi, mu, sigma)`` from one ensemble member, each ``(1, n_w,
        n_d, K)`` over the cohort's full grid.

        The one place an entry's forward signature and return shape are known.
        Absorbs both differences: the argument list (which keys of
        :meth:`_heldout_inputs` the network takes, and in what order) and any
        extra returns an entry's network carries - an auxiliary head is
        dropped here, since it is a training-time regularizer and never part
        of the predictive density.
        """

    # -- the held-out surface ----------------------------------------------------

    def at_cohort(self, segment: dict[str, str]) -> CohortHeldout:
        """A per-cohort held-out scorer view (see this module's docstring).

        The pooled fit is multi-cohort while ``kernels.holdout`` scores one
        cohort at a time, so held-out capability is handed out per cohort: the
        view carries a single-cohort adapter contract that ``index_into``
        accepts unchanged (cohort-identity and training-overlap guards
        included), and its ``log_lik_at``/``predict_at`` are the unmodified
        mixin implementations."""
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        ci = self.cohort_index(segment)
        if ci is None:
            raise ValueError(
                f"{self.name}: at_cohort needs a segment naming one cohort of this fit; "
                "None names the whole pool, which cannot be scored at one diagonal"
            )
        return CohortHeldout(self, ci)

    def log_lik_at(self, cells, *, field: str | None = None) -> np.ndarray:
        """``(n_members, n_cells)`` log density on Lebesgue-on-amount.

        Resolves the cohort from the cells' own segment values, then delegates
        to :meth:`at_cohort`'s view, whose ``log_lik_at`` IS
        ``ScoresHeldout.log_lik_at`` - the measure carry and every guard run in
        the base class against the per-cohort contract."""
        return self.at_cohort(self._heldout_segment(cells)).log_lik_at(cells, field=field)

    def predict_at(
        self, cells: HoldoutCells, *, field: str | None = None, seed: int | None = None
    ) -> np.ndarray:
        """``(n_draws, n_cells)`` draws on the TRIANGLE's basis, in cell order.

        Same delegation as :meth:`log_lik_at`: the view's ``predict_at`` is
        ``PredictsHeldout.predict_at`` unchanged, so the incremental draws are
        anchored onto each cell's training-diagonal predecessor in the base
        class, never here."""
        return self.at_cohort(self._heldout_segment(cells)).predict_at(
            cells, field=field, seed=seed
        )

    def training_cells(self) -> CellIndex:
        raise NotImplementedError(
            "the NN contract carries loss ratios, not per-cell loss values, so the "
            "in-sample agreement gate's CellIndex cannot be built from it; the fast "
            "closed-form tests play that role for the NN entries"
        )

    def _log_lik_native(self, cells: CellIndex) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name}'s fit spans many cohorts and a bare CellIndex cannot name "
            "one; call log_lik_at(HoldoutCells) or at_cohort(segment).log_lik_at(...)"
        )

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name}'s fit spans many cohorts and a bare CellIndex cannot name "
            "one; call predict_at(HoldoutCells) or at_cohort(segment).predict_at(...)"
        )

    def _heldout_segment(self, cells) -> dict[str, str]:
        """The one segment combination the held-out cells describe."""
        if not isinstance(cells, HoldoutCells):
            raise TypeError(
                f"{self.name} resolves which cohort to score from the cells' segment "
                f"values, and only HoldoutCells carries them; got {type(cells).__name__}. "
                "For a bare CellIndex, bind the cohort first: at_cohort(segment)"
            )
        combos = cells.frame[list(cells.segments)].drop_duplicates()
        if len(combos) != 1:
            raise ValueError(
                f"held-out cells span {len(combos)} segment combinations; "
                "next_diagonal scores one cohort at a time"
            )
        return {col: combos.iloc[0][col] for col in combos.columns}

    def _heldout_log_lik(self, ci: int, cells: CellIndex) -> np.ndarray:
        """``(n_members, n_cells)`` log density of the loss RATIO at the cells.

        Per ensemble member: the MDN mixture density of the STANDARDIZED
        increment ratio ``z = (increment/premium - mean0[d]) / std0[d]``, with
        the standardization Jacobian folded in (``- log std0[d]``), leaving a
        density on the ratio - which is what ``heldout_measure = "loss_ratio"``
        declares, so the base class's ``- log premium`` completes the carry to
        Lebesgue-on-amount (the increment/cumulative step has Jacobian 1). The
        divisor is the CONTRACT's premium (:meth:`_cell_premium`), the one the
        per-dev normalizer was estimated on.

        The draw axis is the ENSEMBLE MEMBERS: ``logmeanexp`` over it is the
        ensemble-average predictive density, the deep-ensemble analogue of
        averaging a posterior's per-draw likelihoods. Two members minimum -
        one member is a plug-in density, not an ensemble.

        Cells at a PINNED dev are refused: no trained head exists there and
        the standardized scale is degenerate, so a density would be dishonest.
        The same cells remain CRPS-scorable through :meth:`predict_at` - the
        documented asymmetry (card.md "Held-out scoring").
        """
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        if len(self.models_) < 2:
            raise ValueError(
                f"the ensemble has {len(self.models_)} member(s); the members are the "
                "density's draw axis and one member is a plug-in, not a predictive "
                "distribution - fit with ensemble_size >= 2"
            )
        d0 = np.asarray(cells.d, dtype=int) - 1
        pinned0 = self.norm_["pinned"][0]
        bad = pinned0[d0]
        if bad.any():
            devs = sorted({int(v) + 1 for v in d0[bad]})
            raise ValueError(
                f"{int(bad.sum())} cell(s) sit at pinned dev step(s) {devs}: fewer than "
                "two training-context values reached the per-dev normalizer there, so "
                f"the MDN head is untrained and its scale is degenerate. {self.name} "
                "refuses to score a density at a pinned dev; the cells remain "
                "CRPS-scorable via predict_at, where a pinned draw is the pooled dev mean"
            )
        prev = np.asarray(cells.prev_value, dtype=float)
        if np.isnan(prev).any():
            raise ValueError(
                f"{int(np.isnan(prev).sum())} cell(s) have no training predecessor, so "
                "no increment can be formed to evaluate the density at"
            )
        premium = self._cell_premium(ci, cells)
        increment = np.asarray(cells.value, dtype=float) - prev
        mean0, std0 = self.norm_["mean"][0], self.norm_["std"][0]  # (n_d,)
        z = (increment / premium - mean0[d0]) / std0[d0]  # (n_cells,)

        log_pi, mu, sigma = self._heldout_mixture(ci, cells)  # (n_members, n_cells, K)
        comp = -0.5 * ((z[None, :, None] - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * LOG_2PI
        ll_z = logsumexp(log_pi + comp, axis=-1)  # (n_members, n_cells) density of z
        # z -> ratio change of variable: r = z * std0 + ..., so divide by std0
        return ll_z - np.log(std0[d0])[None, :]

    def _heldout_draws(self, ci: int, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        """``(config.n_draws, n_cells)`` INCREMENTAL dollar draws at the cells.

        One forward pass per ensemble member, conditioned on everything the
        cohort had at as_of (:meth:`_heldout_inputs`) - the held-out diagonal
        is one step past that context, the most-supervised position and the
        rollout's first step, so no rollout is needed here. Then ``mdn_sample``
        at the requested cells, un-standardized (``z * std0[d] + mean0[d]``)
        and scaled by the CONTRACT's premium (:meth:`_cell_premium`) - the
        exposure the sampled ratio is a ratio TO. Draws are split across members
        exactly as ``_rollout`` splits them; per-member torch seeds derive from
        ``rng``, so ``predict_at(seed=...)`` is reproducible.

        Pinned devs keep rollout semantics: the sampled ``z`` is forced to 0,
        i.e. the pooled dev mean after un-standardizing - a point-mass column,
        legal for CRPS as long as some requested cell is live. If EVERY
        requested cell is pinned the result would be a point mass everywhere,
        which is not a predictive distribution, so that is refused.
        """
        import torch

        # the MDN sampler is shared by every mixture-head NN entry; it lives in
        # the transformer's network module, which was simply the first to need it
        from ibnr.gallery.nn.transformer.network import mdn_sample

        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        d0 = np.asarray(cells.d, dtype=int) - 1
        pin_cells = self.norm_["pinned"][0][d0]  # (n_cells,)
        if pin_cells.all():
            raise ValueError(
                "every requested cell sits at a pinned dev step, so every draw column "
                "would be the point mass at the pooled dev mean - not a predictive "
                "distribution. A pinned dev has no trained head (fewer than two "
                "training-context values reached the per-dev normalizer)"
            )
        premium = self._cell_premium(ci, cells)
        mean0, std0 = self.norm_["mean"][0], self.norm_["std"][0]  # (n_d,)
        dev = torch.device(self._device)
        inputs = self._heldout_inputs(ci)
        w0_t = torch.as_tensor(np.asarray(cells.w, dtype=int) - 1, device=dev)
        d0_t = torch.as_tensor(d0, device=dev)
        pin_t = torch.as_tensor(pin_cells, device=dev)

        # split the requested draws across members exactly like _rollout
        n_draws = self.config_.n_draws
        n_members = len(self.models_)
        member_draws = [n_draws // n_members] * n_members
        for i in range(n_draws % n_members):
            member_draws[i] += 1
        # one torch seed per member, drawn from the caller's generator for
        # EVERY member (zero-draw ones too) so the numpy stream is identical
        # whatever the split - predict_at(seed=) stays reproducible
        member_seeds = [int(rng.integers(0, 2**63 - 1)) for _ in range(n_members)]

        pieces: list[np.ndarray] = []
        with torch.no_grad():
            for model, m_draws, m_seed in zip(
                self.models_, member_draws, member_seeds, strict=True
            ):
                if m_draws == 0:
                    continue
                log_pi, mu, sigma = self._forward_mixture(model, inputs)
                # index the grid at the requested cells, replicate per draw
                log_pi_c = log_pi[0, w0_t, d0_t].unsqueeze(0).expand(m_draws, -1, -1)
                mu_c = mu[0, w0_t, d0_t].unsqueeze(0).expand(m_draws, -1, -1)
                sigma_c = sigma[0, w0_t, d0_t].unsqueeze(0).expand(m_draws, -1, -1)
                gen = torch.Generator(device=dev)
                gen.manual_seed(m_seed)
                sample = mdn_sample(log_pi_c, mu_c, sigma_c, generator=gen)
                # pinned devs -> 0 (pooled dev mean after un-standardizing),
                # the same rule _rollout applies
                sample = sample.masked_fill(pin_t[None, :], 0.0)
                pieces.append(sample.cpu().numpy())
        draws = np.concatenate(pieces, axis=0)  # (n_draws, n_cells) standardized
        ratios = draws * std0[d0][None, :] + mean0[d0][None, :]
        return ratios * premium[None, :]  # incremental dollars

    def _cell_premium(self, ci: int, cells: CellIndex) -> np.ndarray:
        """``(n_cells,)`` exposure the ratio math uses: the CONTRACT's
        per-origin premium, with the cells' own premium VERIFIED against it.

        **The number the fit used is the only one either hook may divide or
        multiply by.** ``nn_data`` forms this entry's target as
        ``increment / premium`` from exactly this array, and ``norm_stats``
        then estimates ``mean0``/``std0`` over the result, so an exposure from
        anywhere else standardizes the observation on a scale the network was
        never trained on and rescales every draw by the same factor - silently,
        both finite, both plausible (measured on the test fixture: a x1.5
        premium moved the ensemble log density from -3.8 to -28.7 nats/cell and
        multiplied every incremental draw by 1.5, with no error on either path).

        Verified rather than ignored, which is the rule
        ``guszcza_growth_curve``/``compartmental`` already state: the base
        class's measure carry legitimately divides by the CELLS' premium (the
        holdout frame's, attached by ``next_diagonal`` from the training
        slice), so the two sources must agree or the carried density silently
        stops integrating to 1 - a wrong Jacobian, the bug class nothing
        downstream can see. Cells carrying no premium (NaN) are exempt from the
        comparison: ``index_into`` falls back to this same contract array when
        the frame has no premium column, and the carry has its own refusal for
        a genuinely absent one.

        A NaN in the CONTRACT's premium is refused outright. ``nn_data`` writes
        NaN for an origin the premium field never observed, and both hooks
        would otherwise return NaN for that cell rather than say so - which is
        exactly the read the cells' premium could paper over, since a holdout
        frame can carry an exposure at an origin the fit had none for.
        """
        w0 = np.asarray(cells.w, dtype=int) - 1
        premium = np.asarray(self.contract_["premium"][ci], dtype=float)[w0]
        if np.isnan(premium).any():
            raise ValueError(
                f"{int(np.isnan(premium).sum())} cell(s) sit at an origin the fitted "
                f"contract carries no premium for, so {self.name}'s loss ratio is "
                "undefined there and neither a density nor a draw can be formed"
            )
        supplied = np.asarray(cells.premium, dtype=float)
        mismatched = ~np.isnan(supplied) & ~np.isclose(supplied, premium)
        if mismatched.any():
            raise ValueError(
                f"{int(mismatched.sum())} cell(s) carry a premium that disagrees with the "
                "fitted contract's per-origin premium (the cells' comes from the holdout "
                "frame, attached by next_diagonal from the training slice; the contract's "
                "is the nn_data grid this fit standardized against). The loss ratio is "
                "formed with the contract's number while the measure carry divides by the "
                "cells', so a mismatch would rescale every draw and leave a density that "
                "no longer integrates to 1"
            )
        return premium

    def _heldout_mixture(self, ci: int, cells: CellIndex) -> tuple[np.ndarray, ...]:
        """``(n_members, n_cells, K)`` MDN parameters at the cells, one forward
        pass per ensemble member at the cohort's as_of conditioning."""
        import torch

        inputs = self._heldout_inputs(ci)
        dev = torch.device(self._device)
        w0_t = torch.as_tensor(np.asarray(cells.w, dtype=int) - 1, device=dev)
        d0_t = torch.as_tensor(np.asarray(cells.d, dtype=int) - 1, device=dev)
        acc: tuple[list, list, list] = ([], [], [])
        with torch.no_grad():
            for model in self.models_:
                params = self._forward_mixture(model, inputs)
                for out, t in zip(acc, params, strict=True):
                    out.append(t[0, w0_t, d0_t].cpu().numpy())  # (n_cells, K)
        return tuple(np.stack(a) for a in acc)

    def _cell_identity(self) -> dict:
        """The pooled entry cannot be scored without binding a cohort first.

        Its ``contract_`` is the multi-cohort ``nn_data`` dict, which carries no
        single ``segment`` - and ``log_lik_at``/``predict_at`` here delegate to a
        :class:`CohortHeldout`, which supplies the real identity.
        """
        raise RuntimeError(
            f"{self.name}'s fit spans many cohorts and has no single identity; "
            "bind one with at_cohort(segment) first"
        )
