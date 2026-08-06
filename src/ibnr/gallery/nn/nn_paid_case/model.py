"""nn_paid_case gallery entry: the joint paid + case-reserve model. See card.md.

Case reserves are a STATE with dynamics - they run down toward zero as payments
replace them and jump upward when new information arrives - so a frozen input
channel cannot carry them into a multi-year projection. Every other NN entry in
the gallery discloses exactly that limitation ("features are not simulated
forward"). This entry removes it: per cell it predicts (paid increment, case
movement) JOINTLY from one bivariate mixture, samples both, feeds the sampled
paid increment and the UPDATED case level back as context, and rolls both
forward diagonal by diagonal. The paid projection therefore conditions on a live
case position at every step.

Two backbones over one head, one loss and one contract (``config.backbone``):
the attention body of ``nn_transformer`` and the per-origin GRU recurrence of
``deeptriangle``. The entry's claim is about the data, not about attention, so
the encoder is switchable rather than assumed.

Torch is imported inside fit()/predict() only: the entry must register (and
``ibnr.gallery`` must import) without the [nn] extra installed. The training
scheme is the family's shared machinery (``gallery/nn/_scheme.py``,
``gallery/nn/_training.py``); what is written out here is what differs - the
movement target's derivation, the mixed-observedness loss and the state-update
rollout.

This is a SINGLE-LINE entry: each cohort (company x line of business) is encoded
independently, so a company's per-line predictive draws carry NO cross-line
dependence (that is ``nn_transformer_ml``'s job).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.nn._heldout import PooledMDNHeldout, heldout_cutoff
from ibnr.gallery.nn._scheme import norm_stats, splits
from ibnr.gallery.nn._training import train_ensemble
from ibnr.gallery.nn.nn_paid_case.config import NNPaidCaseConfig
from ibnr.gallery.registry import register
from ibnr.kernels.contract import _as_date
from ibnr.kernels.nn_contract import cohort_identities, nn_data
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

#: cap on (draw chunk x cohorts) per rollout forward pass
MAX_ROLLOUT_BATCH = 4096


def case_movement(level: np.ndarray, level_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The case MOVEMENT target and its own observedness, from the level channel.

    ``level`` / ``level_obs`` are the contract's channel-1 grids (``x[:, 1]`` and
    ``x_obs[:, 1]``) with dev last: the case reserve as a ratio to premium,
    carried UNDIFFERENCED because ``fit`` names the case field in
    ``level_fields``. Input and target are two readings of that one channel -
    the network conditions on the LEVEL (the state) and predicts the MOVEMENT
    (the dynamics), so nothing is stored twice and no second field is needed.

    ``move[d] = level[d] - level[d - 1]``, usable where BOTH cells are present;
    at dev 1 the movement IS the level, the case position having been zero
    before the accident year opened, so it is usable wherever that cell is. That
    asymmetry is the same one ``nn_contract`` applies to increments, for the same
    reason, and it is why the movement mask is NOT ``level_obs``: a hole at dev d
    costs the movement target at dev d AND at dev d+1, while the paid increment
    at both cells is perfectly observable. Feeding those cells to the joint
    density anyway would score a movement computed against contract padding -
    which is what the mixed-observedness loss exists to avoid (``head.nll_mixed``).

    Unusable cells are zero-filled, exactly as the contract zero-fills its own:
    the value is padding for the tensor and every consumer gates on the mask.
    """
    move = np.zeros(level.shape, dtype=float)
    move[..., 0] = level[..., 0]  # dev 1: the movement from a zero case position
    move[..., 1:] = level[..., 1:] - level[..., :-1]
    obs = np.zeros(level_obs.shape, dtype=bool)
    obs[..., 0] = level_obs[..., 0]
    obs[..., 1:] = level_obs[..., 1:] & level_obs[..., :-1]
    return np.where(obs, move, 0.0), obs


def call_backbone(model: Any, inputs: dict) -> tuple:
    """Call a backbone with exactly the inputs its ``forward`` takes, in order.

    The two bodies genuinely differ in what they consume - the transformer needs
    a calendar cutoff and has no company embedding, the GRU is the reverse - so
    each declares ``INPUT_KEYS`` and this is the one place that is honoured.
    Giving both a union signature and letting each ignore half would be the
    repo's named inert-parameter bug class, one layer below the config's.
    """
    missing = [k for k in model.INPUT_KEYS if k not in inputs]
    if missing:
        raise KeyError(
            f"{type(model).__name__} takes {list(model.INPUT_KEYS)} and the inputs are "
            f"missing {missing}"
        )
    return model(*(inputs[k] for k in model.INPUT_KEYS))


def _network_class(backbone: str) -> Any:
    """The nn.Module class for a backbone name. Imports torch - call it late."""
    if backbone == "transformer":
        from ibnr.gallery.nn.nn_paid_case.network_transformer import PaidCaseTransformer

        return PaidCaseTransformer
    from ibnr.gallery.nn.nn_paid_case.network_gru import PaidCaseGRU

    return PaidCaseGRU


@register
class NNPaidCase(GalleryEntry, PooledMDNHeldout):
    name = "nn_paid_case"
    family = "nn"
    #: the dataclass ``fit(config=...)`` takes, reachable through
    #: ``gallery.get("nn_paid_case").config_class`` without importing it by path
    config_class = NNPaidCaseConfig

    #: the head is a JOINT density and the board scores its PAID MARGIN - a
    #: mixture of the component margins, exact rather than approximate
    #: (``head.paid_margin``) - of the STANDARDIZED incremental paid loss ratio.
    #: ``_heldout_log_lik`` folds the standardization Jacobian (``-log std0[d]``)
    #: in, leaving a density on the loss RATIO; this declaration then makes
    #: ``ScoresHeldout.log_lik_at`` subtract ``log premium`` to reach
    #: Lebesgue-on-amount. See card.md "Held-out scoring".
    heldout_measure = "loss_ratio"

    #: a draw is ``premium x un-standardized paid ratio`` - an INCREMENTAL dollar
    #: amount, one dev step's emergence. The Schedule P triangles are cumulative,
    #: so ``PredictsHeldout.predict_at`` adds each cell's training-diagonal
    #: anchor; declaring the scale is what makes that conversion the base class's
    #: job rather than a silent 996-vs-3.4 bug.
    heldout_draw_scale = "incremental"

    def __init__(self) -> None:
        self.contract_: dict | None = None  # nn_data() grids/masks for the fit slice
        self.config_: NNPaidCaseConfig | None = None
        self.models_: list | None = None  # one backbone per ensemble member
        self.norm_: dict | None = None  # per-(channel, dev) + movement + premium stats
        self.history_: list[list[dict]] | None = None  # per-member per-epoch train/val NLL
        self._loss_field: str | None = None  # the PAID field; the board's column
        self._case_field: str | None = None
        self._device: str = "cpu"
        # the rollout is expensive and global; cache it keyed by (n_draws, seed)
        # so per-segment predict() calls reuse one shared set of draws. The case
        # path rides the same cache - it comes out of the same simulation.
        self._rollout_key: tuple | None = None
        self._rollout_ults: np.ndarray | None = None  # (n_draws, n_c, n_w)
        self._rollout_case: np.ndarray | None = None  # (n_draws, L, n_c, n_w) level walk

    def fit(
        self,
        triangle: Triangle,
        *,
        paid_field: str = "paid_loss",
        case_field: str = "case_reserve",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        config: NNPaidCaseConfig | None = None,
        device: str | None = None,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> NNPaidCase:
        """Pooled fit across every cohort (segment combination) in the triangle.

        Never fit this on a single triangle - the whole point is cross-cohort
        pooling.

        **There is deliberately no ``feature_fields``/``level_fields`` here, and
        no ``loss_field``.** The two channels ARE the entry: channel 0 is the
        paid increment ratio (the emergence being predicted and the board's
        column) and channel 1 the case reserve LEVEL ratio, carried undifferenced
        because a case reserve is an eval-date snapshot rather than an amount
        that accumulates. A caller who wants a free choice of channels wants
        ``nn_transformer`` or ``deeptriangle``. The loss field is spelled
        ``paid_field`` because this entry models two fields and "loss_field"
        would underdescribe it; the registry constrains no fit signature beyond
        ``config=``.

        A triangle that does not carry ``case_field`` is refused by name, by
        ``nn_data`` - the same refusal every channel gets, rather than a fit that
        silently trains on one all-masked channel.
        """
        import torch

        from ibnr.gallery.nn.nn_paid_case.head import nll_mixed

        cfg = config or NNPaidCaseConfig()
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # BUILD FIRST, ASSIGN AFTER TRAINING SUCCEEDED - fit() must be atomic.
        # train_ensemble is the fallible step, and assigning contract_/norm_
        # before it leaves a failed refit TORN: the new pool's contract and
        # normalizer over the old pool's networks. index_into checks identity
        # against the contract, so at_cohort(...).predict_at would then pass
        # every guard and score one cohort's cells from another pool's fit.
        # See gallery/nn/transformer/model.py and deterministic/mack.
        contract = nn_data(
            train,
            loss_field=paid_field,
            feature_fields=(case_field,),
            level_fields=(case_field,),
            premium_field=premium_field,
        )
        device_str = device or "cpu"
        c = contract
        # n_c cohorts, n_f = 2 channels (paid increment, case level), n_w
        # origins, n_d dev lags
        n_c, n_f, n_w, n_d = c["x"].shape

        # eval_date validation split + per-(channel, dev) normalization, both
        # computed from training-context cells only so the held-out diagonal
        # never leaks into the split, the normalizer, or the premium stats.
        paid_elig, paid_val, val_cutoff = splits(c["obs_mask"], c["cal_idx"], cfg.val_diagonals)
        cal_gate = c["cal_idx"] <= val_cutoff
        # the CONTEXT is per channel: every channel gets the target's calendar
        # constraint but its OWN observedness, so the case level is conditioned
        # on (and standardized) only where the triangle reported one.
        context_elig = c["x_obs"] & cal_gate[None, None]  # (n_c, n_f, n_w, n_d)
        mean, std, pinned = norm_stats(c["x"], context_elig, c["x_obs"])
        prem_mean = float(np.mean(c["log_premium"]))
        prem_std = float(np.std(c["log_premium"]))
        if prem_std < 1e-8:
            prem_std = 1.0
        norm = {
            "mean": mean,
            "std": std,
            "pinned": pinned,
            "prem_mean": prem_mean,
            "prem_std": prem_std,
        }

        # THE MOVEMENT TARGET. Derived in-entry from channel 1 (the level), with
        # its own observedness, and given its OWN per-dev standardization under
        # the same pinning rule - stored beside the input stats exactly as
        # deeptriangle stores its auxiliary head's. It needs separate statistics
        # because a movement and a level are different quantities on the same
        # ratio scale: the level decays across dev while the movement is centred
        # near zero and changes sign, so standardizing one with the other's mean
        # would put every target at a location the head has to undo.
        move, move_obs = case_movement(c["x"][:, 1], c["x_obs"][:, 1])
        move_mean, move_std, move_pinned = norm_stats(move[:, None], move_obs & cal_gate, move_obs)
        norm["move_mean"] = move_mean[0]
        norm["move_std"] = move_std[0]
        norm["move_pinned"] = move_pinned[0]
        # the movement's training targets get the same calendar treatment as the
        # paid channel's: trailing diagonals are validation, everything earlier
        # is trainable.
        move_elig = move_obs & cal_gate
        move_val = move_obs & ~cal_gate

        # standardize per (channel, dev); pinned devs (too few context values,
        # in practice the deepest) are forced to standardized 0 so they carry no
        # spurious signal. mean/std/pinned are (n_f, n_d), broadcast over cohorts
        # (axis 0) and origins (axis 2); x_norm is (n_c, n_f, n_w, n_d).
        x_norm = (c["x"] - mean[None, :, None, :]) / std[None, :, None, :]
        x_norm = np.where(pinned[None, :, None, :], 0.0, x_norm)
        move_norm = (move - norm["move_mean"][None, None, :]) / norm["move_std"][None, None, :]
        move_norm = np.where(norm["move_pinned"][None, None, :], 0.0, move_norm)

        dev = torch.device(device_str)
        xt = torch.tensor(x_norm, dtype=torch.float32, device=dev)  # (n_c, n_f, n_w, n_d)
        y_paid_t = xt[:, 0]  # (n_c, n_w, n_d) standardized paid increment ratio
        y_move_t = torch.tensor(move_norm, dtype=torch.float32, device=dev)  # (n_c, n_w, n_d)
        xobs_t = torch.tensor(c["x_obs"], device=dev)  # (n_c, n_f, n_w, n_d)
        cal_t = torch.tensor(c["cal_idx"], device=dev)  # (n_w, n_d)
        ctx_elig_t = torch.tensor(context_elig, device=dev)  # (n_c, n_f, n_w, n_d)
        paid_elig_t = torch.tensor(paid_elig, device=dev)  # (n_c, n_w, n_d)
        paid_val_t = torch.tensor(paid_val, device=dev)
        move_elig_t = torch.tensor(move_elig, device=dev)
        move_val_t = torch.tensor(move_val, device=dev)
        lob_t = torch.tensor(c["lob_idx"], dtype=torch.long, device=dev)
        comp_t = torch.tensor(c["company_idx"], dtype=torch.long, device=dev)
        prem_t = torch.tensor(
            (c["log_premium"] - prem_mean) / prem_std, dtype=torch.float32, device=dev
        )

        # augmented cutoffs are drawn from [min_cutoff, val_cutoff); clamp the
        # floor so at least one earlier diagonal remains to condition on.
        min_cutoff = max(1, min(cfg.min_cutoff, val_cutoff - 1))
        network = _network_class(cfg.backbone)

        def make_model():
            if cfg.backbone == "transformer":
                return network(
                    cfg, n_lob=len(c["lob_levels"]), n_features=n_f, n_w=n_w, n_d=n_d
                ).to(dev)
            return network(
                cfg,
                n_lob=len(c["lob_levels"]),
                n_company=len(c["company_levels"]),
                n_features=n_f,
                n_w=n_w,
                n_d=n_d,
            ).to(dev)

        def loss_at(model, idx, ctx, cutoffs, paid_tgt, move_tgt):
            """The mixed-observedness NLL over one batch, or None if nothing scores.

            The three masks PARTITION the scored cells: a cell with both targets
            trains on the joint bivariate density, a cell with only the paid
            increment on the closed-form paid margin, a cell with only the case
            movement on the case margin. All three are the SAME fitted head,
            marginalized - not a second model - which is what lets a case hole
            cost the joint term without costing the paid signal the board scores.
            """
            joint = paid_tgt & move_tgt
            if not bool((paid_tgt | move_tgt).any()):
                return None
            inputs = {
                "x": xt[idx],
                "ctx": ctx,
                "lob": lob_t[idx],
                "comp": comp_t[idx],
                "prem": prem_t[idx],
                "cutoff": cutoffs,
            }
            log_pi, mu, chol = call_backbone(model, inputs)
            return nll_mixed(
                log_pi,
                mu,
                chol,
                y_paid=y_paid_t[idx],
                y_case=y_move_t[idx],
                joint_mask=joint,
                paid_only_mask=paid_tgt & ~move_tgt,
                case_only_mask=move_tgt & ~paid_tgt,
            )

        def train_loss(model, idx, cutoffs):
            # condition on cells on/before the augmented cutoff, score the
            # observed training cells strictly after it (card.md "Training").
            # PER CHANNEL: a case level beyond the cutoff is masked, not read.
            ctx = xobs_t[idx] & (cal_t[None, None] <= cutoffs[:, None, None, None])  # (B,F,W,D)
            past = cal_t[None] > cutoffs[:, None, None]  # (B, W, D)
            return loss_at(
                model,
                idx,
                ctx,
                cutoffs,
                paid_elig_t[idx] & past,
                move_elig_t[idx] & past,
            )

        def val_loss(model):
            # early stopping tracks the SAME mixed loss, not the paid margin
            # alone. Unlike deeptriangle's auxiliary head - a regularizer whose
            # weight model selection must not be coupled to - the case
            # coordinate here is consumed at prediction time: the rollout feeds
            # the simulated level back, so a member that fits paid well and case
            # badly degrades the paid projection it is being selected for. One
            # density, one number, no weight to be coupled to.
            idx = torch.arange(n_c, device=dev)
            cut = torch.full((n_c,), val_cutoff, dtype=torch.long, device=dev)
            loss = loss_at(model, idx, ctx_elig_t, cut, paid_val_t, move_val_t)
            return float("inf") if loss is None else float(loss)

        # deep ensemble via the shared loop (gallery/nn/_training.py): member
        # seeding (seed + 1000 * member), cutoff augmentation batching, AdamW,
        # early stopping. Pooling the members' draws at rollout adds epistemic
        # spread on top of the mixture's aleatoric spread.
        models, history = train_ensemble(
            n_c,
            config=cfg,
            seed=seed,
            make_model=make_model,
            train_loss=train_loss,
            val_loss=val_loss,
            min_cutoff=min_cutoff,
            val_cutoff=val_cutoff,
            device=dev,
            show_progress=show_progress,
        )

        self.contract_ = contract
        self._loss_field = paid_field
        self._case_field = case_field
        self.config_ = cfg
        self._device = device_str
        self.norm_ = norm
        self.models_, self.history_ = models, history
        # the cached rollout belongs to the previous fit; drop it whole
        self._rollout_key = None
        self._rollout_ults = None
        self._rollout_case = None
        return self

    def cohorts(self) -> list[dict]:
        """One dict per pooled cohort, in ``contract_["cohorts"]`` row order.

        The cohort KEY plus any display-only segment column ``nn_data`` kept out
        of it, so a caller sees the identity the triangle carried rather than the
        narrower key the pooling required (see :meth:`GalleryEntry.cohorts`).
        """
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        return cohort_identities(self.contract_)

    def predict(
        self,
        segment: Mapping | None = None,
        n_draws: int | None = None,
        seed: int | None = None,
    ) -> PredictiveDistribution:
        """Predictive PAID ultimates from the cached global rollout.

        With ``segment``: per-origin ultimates for that one cohort plus their
        total - the same target layout as every other entry. Without: every
        (cohort, origin) ultimate, no grand total (a total across companies is
        meaningless, and this entry's per-line draws are independent anyway).

        Paid only, on purpose. The simulated case path is a diagnostic
        (:meth:`case_paths`), not a second predictive distribution: the board
        compares paid ultimates across models and there is no realized-case
        column to score a case predictive against.
        """
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        # resolve the cohort BEFORE the rollout: a bad segment must not cost one
        ci = self.cohort_index(segment)
        ults = self._ensure_rollout(n_draws, seed)[0]
        c = self.contract_

        if ci is not None:
            targets = pd.DataFrame(
                {
                    "label": [str(o.year) for o in c["origin_periods"]],
                    "origin_period": c["origin_periods"],
                    "premium": c["premium"][ci],
                }
            )
            pred = PredictiveDistribution(samples=ults[:, ci, :], targets=targets)
            return pred.with_total()

        n_c, n_w = ults.shape[1], ults.shape[2]
        cohorts = c["cohorts"]
        targets = cohorts.loc[cohorts.index.repeat(n_w)].reset_index(drop=True)
        targets["origin_period"] = list(c["origin_periods"]) * n_c
        targets["premium"] = c["premium"].reshape(-1)
        return PredictiveDistribution(samples=ults.reshape(ults.shape[0], -1), targets=targets)

    def case_paths(
        self,
        n_draws: int | None = None,
        seed: int | None = None,
        *,
        per_diagonal: bool = False,
    ) -> np.ndarray:
        """Simulated case level ratios - a DIAGNOSTIC, not a forecast.

        Default: ``(n_draws, n_c, n_w)`` TERMINAL levels, the state after the
        deepest projected diagonal. With ``per_diagonal=True``: the FULL walk,
        ``(n_draws, n_levels, n_c, n_w)`` - axis 1 is the rollout's future
        calendar diagonals in ascending order, its last step identical to the
        terminal read, so the run-off TRAJECTORY (where the level steps down,
        where a shock lands) is inspectable, not only its endpoint.

        Not a ``PredictiveDistribution`` and deliberately not one: it is the
        rollout's internal state read out, in the same (draw, cohort, origin)
        layout as :meth:`predict`'s samples and off the same cached simulation,
        so the two are the same draws. The value is the case reserve as a RATIO
        to that origin's premium - multiply by ``contract_["premium"]`` for
        dollars.

        **The case run-off diagnostic** (card.md): a case reserve that has done
        its job is nearly exhausted by the end of the projection, so this should
        concentrate near zero. Mass exactly AT zero is the reserve having fully
        run down - the walk is floored there (``config.floor_case_at_zero``), so
        zero is an absorbing value the simulation can reach and not pass, the
        way a booked reserve is. A fat positive tail is the model saying
        development continues past the triangle's window - useful information
        about the tail, and a reason to distrust the ultimate at face value.

        With ``floor_case_at_zero=False`` (the 0.5.5 walk, kept so the change
        can be measured with and without it) the movement head is unconstrained
        and a run-down can overshoot into a case reserve below zero; roughly
        half the simulated terminal levels did on the Schedule P panel. Read
        negative mass there as the defect the floor exists for, not a finding.

        The case path's own calibration is unvalidated in v1 - no realized-case
        board column exists to score it against - so read it as a diagnostic of
        the simulation, never as a case-reserve forecast.
        """
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        path = self._ensure_rollout(n_draws, seed)[1]  # (n_draws, L, n_c, n_w)
        return path if per_diagonal else path[:, -1]

    def realized_ultimates(
        self, full_triangle: Triangle, segment: Mapping | None = None
    ) -> np.ndarray:
        """Outcomes aligned to predict(segment)'s targets, from the full
        triangle at the final dev lag (+ total when a segment is given)."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        ci = self.cohort_index(segment)
        c = self.contract_
        seg_cols = list(c["cohorts"].columns)
        df = full_triangle.select_fields(self._loss_field).execute()
        df = df[df["dev_lag"] == c["n_d"] * c["dev_grain_months"]].copy()
        df["origin_period"] = _as_date(df["origin_period"])
        by_key = df.set_index([*seg_cols, "origin_period"])["value"]

        def lookup(row: pd.Series) -> np.ndarray:
            vals = [float(by_key.get((*row.tolist(), o), np.nan)) for o in c["origin_periods"]]
            return np.asarray(vals)

        if ci is not None:
            per_origin = lookup(c["cohorts"].iloc[ci])
            return np.append(per_origin, per_origin.sum())
        return np.concatenate([lookup(row) for _, row in c["cohorts"].iterrows()])

    # -- held-out scoring (milestone 6 wiring) -------------------------------------

    def _forward_mixture(self, model, inputs: dict) -> tuple:
        """``(log_pi, mu_p, sigma_p)`` - the head's PAID MARGIN over the grid.

        A Gaussian mixture's margin is the mixture of its components' margins
        with the same weights, so this is the fitted joint head marginalized
        exactly, not a second model and not an approximation. It is also
        precisely the univariate ``(log_pi, mu, sigma)`` shape
        ``PooledMDNHeldout`` scores and samples, which is what lets this entry
        join the board column-for-column beside the four univariate NN entries.

        The case coordinate is dropped here because the board has no
        realized-case column to score it against - not because it is a nuisance.
        """
        from ibnr.gallery.nn.nn_paid_case.head import paid_margin

        return paid_margin(*call_backbone(model, inputs))

    def _heldout_inputs(self, ci: int) -> dict:
        """One cohort's forward inputs, conditioned on everything it had at as_of.

        Context = every cell each CHANNEL has a usable value at (``x_obs``), so
        the paid increments and the case levels each condition on their own
        observed cells. For the transformer backbone the cutoff is the deepest
        calendar diagonal the cohort HELD a cell on, so the held-out diagonal
        sits at distance 1 - the most-supervised relative-calendar position and
        the rollout's first step (see ``_heldout.heldout_cutoff`` for why
        ``obs_mask`` alone is not that boundary).

        The dict is filtered to the backbone's own ``INPUT_KEYS``, so the GRU
        arm carries NO ``cutoff`` key at all: it has no calendar boundary to
        place (relative position is structural in the recurrence), and
        advertising one it never reads would be an inert parameter that
        ``tests/test_nn_heldout_cutoff.py`` would grade as live.
        """
        import torch

        c = self.contract_
        # same standardize + pin as fit()/_rollout(), for this cohort only
        x_norm = (c["x"][ci] - self.norm_["mean"][:, None, :]) / self.norm_["std"][:, None, :]
        x_norm = np.where(self.norm_["pinned"][:, None, :], 0.0, x_norm)
        prem_norm = (c["log_premium"][ci] - self.norm_["prem_mean"]) / self.norm_["prem_std"]
        dev = torch.device(self._device)
        available = {
            "x": torch.tensor(x_norm[None], dtype=torch.float32, device=dev),
            "ctx": torch.tensor(c["x_obs"][ci][None], device=dev),  # per-channel conditioning
            "lob": torch.tensor([c["lob_idx"][ci]], dtype=torch.long, device=dev),
            "comp": torch.tensor([c["company_idx"][ci]], dtype=torch.long, device=dev),
            "prem": torch.tensor([prem_norm], dtype=torch.float32, device=dev),
            "cutoff": torch.tensor([heldout_cutoff(c, ci)], dtype=torch.long, device=dev),
        }
        return {k: available[k] for k in _network_class(self.config_.backbone).INPUT_KEYS}

    # -- internals ---------------------------------------------------------------

    def _ensure_rollout(self, n_draws: int | None, seed: int | None) -> tuple:
        """The cached ``(ultimates, terminal case levels)`` for this (n_draws, seed)."""
        n_draws = n_draws or self.config_.n_draws
        key = (n_draws, seed)
        if self._rollout_key != key:
            self._rollout_ults, self._rollout_case = self._rollout(n_draws, seed)
            self._rollout_key = key
        return self._rollout_ults, self._rollout_case

    def _initial_case_level(self) -> np.ndarray:
        """``(n_c, n_w)`` case level ratio the rollout starts each origin from.

        The cohort's deepest OBSERVED case level at a dev strictly before that
        origin's first projected cell, or 0.0 where the triangle reported none.
        "Strictly before" ties the state to the rollout's own notion of the past
        (``future = d_index >= latest_dev``), so a case cell booked on a deeper
        diagonal than the paid anchor cannot be counted twice - once as the
        starting level and again as a cell the rollout simulates over.

        Known approximation, disclosed in the card: when the case cell AT the
        paid anchor is missing but an earlier one exists, the state starts from
        that stale level and the movements over the skipped devs are never
        sampled (the rollout only visits ``d >= latest_dev``), so for a reserve
        that is running down the start is overstated by the run-down that was
        skipped - silently, every number finite. Bridging it would mean sampling
        movements on pre-anchor diagonals, which the double-count rule above
        exists to forbid.
        """
        c = self.contract_
        level, obs = c["x"][:, 1], c["x_obs"][:, 1]  # (n_c, n_w, n_d)
        d_idx = np.arange(level.shape[2])[None, None, :]
        past = obs & (d_idx < c["latest_dev"][:, :, None])  # (n_c, n_w, n_d)
        any_past = past.any(axis=2)
        # index of the LAST True per (cohort, origin): first True from the right
        deepest = past.shape[2] - 1 - past[:, :, ::-1].argmax(axis=2)
        picked = np.take_along_axis(level, deepest[:, :, None], axis=2)[:, :, 0]
        return np.where(any_past, picked, 0.0)

    def _rollout(self, n_draws: int, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Autoregressive rollout of BOTH channels, diagonal by diagonal.

        Per future calendar diagonal, per draw: one forward pass, one JOINT
        sample of (paid increment, case movement) at that diagonal's future cells
        - same mixture component, same ``z``, so the learned correlation between
        payment and case run-off survives into the simulated diagonal - then

        1. the un-standardized paid increment ratio is written into channel 0;
        2. the case LEVEL STATE is advanced, ``level = max(level + movement, 0)``,
           in RATIO space, and the re-standardized level is written into
           channel 1;
        3. BOTH channels' flags are promoted at those cells;
        4. the grid is re-encoded and the next diagonal follows.

        Step 2 is the entry. The network conditions on a LEVEL, so a sampled
        MOVEMENT has to be integrated before it can be fed back, and the
        integration happens in ratio space - the only scale on which the two are
        the same quantity - before being re-standardized with the LEVEL channel's
        own per-dev statistics (which are not the movement's). Standardized
        values cannot simply be added: each dev has its own mean and spread, so
        ``z_level[d] + z_move[d]`` is not the standardized new level, and it
        would look entirely plausible.

        The ``max(..., 0)`` is ``config.floor_case_at_zero`` (default on): a
        case reserve is booked down TO zero and never past it, so the walk is
        truncated there. It is applied to the STATE only, AFTER the joint draw,
        which is what keeps it out of the random stream - the paid coordinate is
        written unchanged and the generator has already advanced, so a floored
        and an unfloored run share every draw for a given seed and
        ``floor_case_at_zero=False`` reproduces the 0.5.5 walk exactly. The
        floored value is what feeds channel 1 forward AND what
        :meth:`case_paths` reports; there is no second, unfloored copy of the
        state.

        What is NOT floored is the STARTING level: :meth:`_initial_case_level`
        carries the deepest observed case level exactly as the triangle reported
        it, negative included (a recovery can outrun the case estimate, and
        restating an observation is not constraining a simulation). So an origin
        that never steps - one already at its deepest dev - can still show a
        negative level in :meth:`case_paths`. Every level the SIMULATION
        produced is >= 0.

        Step 3 is what the per-channel promotion of 0.5.4 exists for: this
        rollout SIMULATED both channels at those cells, so both flags rise -
        the exact opposite of the other four NN entries, which promote channel 0
        alone precisely because they simulated nothing else.

        Returns ``(ultimates, case_level_path)`` - (n_draws, n_c, n_w) and
        (n_draws, L, n_c, n_w), L = future calendar diagonals ascending, the
        path's last step being the terminal level:
        ultimate = anchor cumulative + premium x summed future paid ratios, and
        the case state as it stands after the last projected diagonal.
        """
        import torch

        from ibnr.gallery.nn.nn_paid_case import head as H

        c = self.contract_
        n_c, _, n_w, n_d = c["x"].shape
        norm = self.norm_
        mean0, std0 = norm["mean"][0], norm["std"][0]  # (n_d,) paid channel
        dev = torch.device(self._device)

        # future cells: at/beyond each origin's latest observed dev (its anchor)
        # - everything to be predicted. latest_dev is 1-based, d_grid 0-based,
        # so `>=` includes the first unobserved dev.
        d_grid = np.arange(n_d)[None, None, :]
        future = d_grid >= c["latest_dev"][:, :, None]  # (n_c, n_w, n_d) bool
        cal_levels = sorted(np.unique(c["cal_idx"][future.any(axis=0)]))

        # same standardize + pin as fit(); (n_c, n_f, n_w, n_d)
        x_norm = (c["x"] - norm["mean"][None, :, None, :]) / norm["std"][None, :, None, :]
        x_norm = np.where(norm["pinned"][None, :, None, :], 0.0, x_norm)
        prem_norm = (c["log_premium"] - norm["prem_mean"]) / norm["prem_std"]

        def t(a, dtype=None):
            return torch.tensor(a, dtype=dtype, device=dev)

        # pinned devs have no trained head: their standardized value is 0 by
        # definition, so sampled draws there are forced to 0 -> the pooled dev
        # mean once un-standardized. Applied to both coordinates, each against
        # its own pin mask (the paid channel's and the movement target's).
        pin_paid = t(norm["pinned"][0])  # (n_d,)
        pin_move = t(norm["move_pinned"])  # (n_d,)
        pin_level = t(norm["pinned"][1])  # (n_d,)
        move_mean_t = t(norm["move_mean"], torch.float32)
        move_std_t = t(norm["move_std"], torch.float32)
        level_mean_t = t(norm["mean"][1], torch.float32)
        level_std_t = t(norm["std"][1], torch.float32)

        xt = t(x_norm, torch.float32)  # (n_c, n_f, n_w, n_d)
        xobs_t = t(c["x_obs"])  # (n_c, n_f, n_w, n_d)
        fut_t = t(future)  # (n_c, n_w, n_d)
        cal_t = t(c["cal_idx"])  # (n_w, n_d)
        lob_t = t(c["lob_idx"], torch.long)  # (n_c,)
        comp_t = t(c["company_idx"], torch.long)  # (n_c,)
        prem_t = t(prem_norm, torch.float32)  # (n_c,)
        level0_t = t(self._initial_case_level(), torch.float32)  # (n_c, n_w) ratio
        floor_case = bool(self.config_.floor_case_at_zero)

        # split the requested draws as evenly as possible across ensemble
        # members (remainder spread over the first few members).
        n_members = len(self.models_)
        member_draws = [n_draws // n_members] * n_members
        for i in range(n_draws % n_members):
            member_draws[i] += 1
        # a forward pass stacks (chunk draws) x (n_c cohorts) rows; cap the row
        # count at MAX_ROLLOUT_BATCH so wide draw counts don't blow up memory.
        chunk_size = max(1, MAX_ROLLOUT_BATCH // n_c)

        ult_pieces: list[np.ndarray] = []
        case_pieces: list[np.ndarray] = []
        for member, (model, m_draws) in enumerate(zip(self.models_, member_draws, strict=True)):
            if m_draws == 0:
                continue
            gen = torch.Generator(device=dev)
            gen.manual_seed((0 if seed is None else seed) * 100003 + member)
            done = 0
            while done < m_draws:
                chunk = min(chunk_size, m_draws - done)
                # replicate each cohort `chunk` times (draws interleaved within a
                # cohort block): row = cohort*chunk + draw. xb/ctx/level are
                # mutated as the rollout fills future cells, so they are cloned.
                xb = xt.repeat_interleave(chunk, dim=0).clone()  # (chunk*n_c, n_f, n_w, n_d)
                ctx = xobs_t.repeat_interleave(chunk, dim=0).clone()
                futb = fut_t.repeat_interleave(chunk, dim=0)  # (chunk*n_c, n_w, n_d)
                lobb = lob_t.repeat_interleave(chunk, dim=0)
                compb = comp_t.repeat_interleave(chunk, dim=0)
                premb = prem_t.repeat_interleave(chunk, dim=0)
                level = level0_t.repeat_interleave(chunk, dim=0).clone()  # (chunk*n_c, n_w)
                # one state snapshot per future calendar diagonal, sampled or
                # not, so the path axis is uniform across chunks and members
                path_steps: list[torch.Tensor] = []
                with torch.no_grad():
                    for lv in cal_levels:
                        cells = futb & (cal_t[None] == lv)  # (chunk*n_c, n_w, n_d)
                        if not bool(cells.any()):
                            path_steps.append(level.clone())
                            continue
                        # the context boundary advances with each sampled
                        # diagonal, so the predicted diagonal always sits at
                        # distance 1 - the most-supervised position.
                        cut_b = torch.full(
                            (xb.shape[0],), int(lv) - 1, dtype=torch.long, device=dev
                        )
                        log_pi, mu, chol = call_backbone(
                            model,
                            {
                                "x": xb,
                                "ctx": ctx,
                                "lob": lobb,
                                "comp": compb,
                                "prem": premb,
                                "cutoff": cut_b,
                            },
                        )
                        # ONE joint draw per cell: (chunk*n_c, n_w, n_d, 2)
                        sample = H.sample_joint(log_pi, mu, chol, generator=gen)
                        paid_z = sample[..., 0].masked_fill(pin_paid[None, None, :], 0.0)
                        move_z = sample[..., 1].masked_fill(pin_move[None, None, :], 0.0)
                        # movement back onto the RATIO scale, then integrated
                        # into the level state. At most one cell per (row,
                        # origin) sits on a diagonal, so the masked sum over dev
                        # picks that one movement (and 0 for origins with none).
                        move_ratio = move_z * move_std_t + move_mean_t  # (B, W, D)
                        delta = (move_ratio * cells.to(move_ratio.dtype)).sum(dim=2)  # (B, W)
                        stepped = level + delta
                        if floor_case:
                            # a case reserve is booked down TO zero, never past
                            # it. Post-draw arithmetic on the STATE only, after
                            # the sample: the paid coordinate is untouched and
                            # the generator has already advanced, so a floored
                            # and an unfloored rollout share every draw.
                            stepped = stepped.clamp(min=0.0)
                        level = torch.where(cells.any(dim=2), stepped, level)
                        # re-standardize the NEW level with the LEVEL channel's
                        # own per-dev statistics - never by adding standardized
                        # movements to standardized levels.
                        level_z = (stepped[:, :, None] - level_mean_t) / level_std_t
                        level_z = level_z.masked_fill(pin_level[None, None, :], 0.0)
                        # write both channels at the sampled cells and promote
                        # BOTH flags: the rollout simulated both, which is what
                        # per-channel promotion exists for.
                        xb[:, 0][cells] = paid_z[cells]
                        xb[:, 1][cells] = level_z[cells]
                        ctx[:, 0][cells] = True
                        ctx[:, 1][cells] = True
                        path_steps.append(level.clone())
                # un-standardize the paid channel back to loss ratios, then sum
                # only the future increments per (row, origin).
                ratios = xb[:, 0].cpu().numpy() * std0[None, None, :] + mean0[None, None, :]
                contrib = (ratios * future.repeat(chunk, axis=0)).sum(axis=2)  # (chunk*n_c, n_w)
                contrib = contrib.reshape(n_c, chunk, n_w, order="C")  # undo interleave
                # ultimate = anchor cumulative + premium * summed future increments
                ults = c["latest_cum"][:, None, :] + contrib * c["premium"][:, None, :]
                ult_pieces.append(np.moveaxis(ults, 1, 0))  # -> (chunk, n_c, n_w)
                # the full path: (L, chunk*n_c, n_w) -> (L, n_c, chunk, n_w)
                # undoing the interleave, then the chunk axis to the front. Its
                # last step IS the terminal level, so nothing is stored twice.
                path = torch.stack(path_steps).cpu().numpy()
                path = path.reshape(len(cal_levels), n_c, chunk, n_w, order="C")
                case_pieces.append(np.moveaxis(path, 2, 0))  # (chunk, L, n_c, n_w)
                done += chunk
        return np.concatenate(ult_pieces, axis=0), np.concatenate(case_pieces, axis=0)
