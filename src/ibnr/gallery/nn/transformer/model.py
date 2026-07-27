"""Triangle transformer gallery entry. See card.md for the model card.

This wires the ``TriangleTransformer`` network (network.py) and the NN data
contract (kernels/nn_contract.py) into the ``GalleryEntry`` ABC: fit() ->
predict() -> a ``PredictiveDistribution`` of ultimate losses, scored by the
same Meyers-style PIT/CRPS harness as every other entry.

Torch is imported inside fit()/predict() only: the entry must register (and
`ibnr.gallery` must import) without the [nn] extra installed. The training
scheme - calendar-cutoff augmentation, trailing-diagonal validation, deep
ensembling, diagonal-by-diagonal autoregressive rollout - lives here in
literal source, per the gallery's eject pattern.

This is the SINGLE-LINE transformer: each cohort (company x line of business)
is encoded independently, so a company's per-line predictive draws carry NO
cross-line dependence. Modeling that dependence (contemporaneous cross-line
correlation, the learned analogue of SUR/copula) is the separate
``nn_transformer_ml`` entry; nothing here does it."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ibnr.gallery.entry import GalleryEntry, PredictsHeldout, ScoresHeldout
from ibnr.gallery.nn._heldout import CohortHeldout

# the training-scheme helpers moved to gallery/nn/_scheme.py (shared by every
# NN entry); the private aliases keep this module's historical import surface
# (`from ...transformer.model import _norm_stats, _splits`) working.
from ibnr.gallery.nn._scheme import norm_stats as _norm_stats
from ibnr.gallery.nn._scheme import splits as _splits
from ibnr.gallery.nn._training import train_ensemble
from ibnr.gallery.nn.transformer.config import TransformerConfig
from ibnr.gallery.registry import register
from ibnr.kernels.contract import _as_date
from ibnr.kernels.holdout import CellIndex, HoldoutCells
from ibnr.kernels.nn_contract import nn_data
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

#: cap on (draw chunk x cohorts) per rollout forward pass
MAX_ROLLOUT_BATCH = 4096

LOG_2PI = math.log(2.0 * math.pi)


@register
class NNTransformer(GalleryEntry, ScoresHeldout, PredictsHeldout):
    name = "nn_transformer"
    family = "nn"

    #: the MDN head is a density of the STANDARDIZED incremental loss ratio;
    #: ``_heldout_log_lik`` folds the standardization Jacobian (``-log std0[d]``)
    #: in, leaving a density on the loss RATIO - this declaration then makes
    #: ``ScoresHeldout.log_lik_at`` subtract ``log premium`` to reach
    #: Lebesgue-on-amount. See card.md "Held-out scoring".
    heldout_measure = "loss_ratio"

    #: a draw is ``premium x un-standardized ratio`` - an INCREMENTAL dollar
    #: amount, one dev step's emergence. The Schedule P triangles are
    #: cumulative, so ``PredictsHeldout.predict_at`` adds each cell's
    #: training-diagonal anchor; declaring the scale is what makes that
    #: conversion the base class's job rather than a silent 996-vs-3.4 bug.
    heldout_draw_scale = "incremental"

    def __init__(self) -> None:
        self.contract_: dict | None = None  # nn_data() grids/masks for the fit slice
        self.config_: TransformerConfig | None = None
        self.models_: list | None = None  # one TriangleTransformer per ensemble member
        self.norm_: dict | None = None  # per-(channel, dev) + premium normalization stats
        self.history_: list[list[dict]] | None = None  # per-member per-epoch train/val NLL
        self._loss_field: str | None = None
        self._device: str = "cpu"
        # rollout is expensive and global; cache it keyed by (n_draws, seed) so
        # per-segment predict() calls reuse one shared set of draws.
        self._rollout_key: tuple | None = None
        self._rollout_ults: np.ndarray | None = None  # (n_draws, n_c, n_w)

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "reported_loss",
        feature_fields: tuple[str, ...] = (),
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        config: TransformerConfig | None = None,
        device: str | None = None,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> NNTransformer:
        """Pooled fit across every cohort (segment combination) in the
        triangle. Never fit this on a single triangle - the whole point is
        cross-cohort pooling."""
        import torch

        from ibnr.gallery.nn.transformer import network as net

        cfg = config or TransformerConfig()
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # BUILD FIRST, ASSIGN AFTER TRAINING SUCCEEDED - fit() must be atomic.
        # The ensemble trainer is the fallible step (bad loss surfaces, OOM,
        # interrupts), and assigning contract_/norm_ before it leaves a failed
        # refit TORN: the new pool's contract and normalizer over the old
        # pool's networks, which at_cohort/predict then serve under the wrong
        # identity. See mack.
        contract = nn_data(
            train,
            loss_field=loss_field,
            feature_fields=feature_fields,
            premium_field=premium_field,
        )
        device_str = device or "cpu"
        c = contract
        # n_c cohorts, n_f channels (target first), n_w origins, n_d dev lags
        n_c, n_f, n_w, n_d = c["x"].shape

        # eval_date validation split + per-(channel, dev) normalization, both
        # computed from training-context cells only so the held-out diagonal
        # never leaks into the split, the normalizer, or the premium stats.
        context_elig, val_target, val_cutoff = _splits(
            c["obs_mask"], c["cal_idx"], cfg.val_diagonals
        )
        mean, std, pinned = _norm_stats(c["x"], context_elig, c["obs_mask"])
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

        # standardize per (channel, dev); pinned devs (too few context values,
        # in practice the deepest) are forced to standardized 0 so they carry
        # no spurious signal. Shapes: mean/std/pinned are (n_f, n_d), broadcast
        # over cohorts (axis 0) and origins (axis 2). x_norm is (n_c, n_f, n_w, n_d).
        x_norm = (c["x"] - mean[None, :, None, :]) / std[None, :, None, :]
        x_norm = np.where(pinned[None, :, None, :], 0.0, x_norm)
        dev = torch.device(device_str)
        xt = torch.tensor(x_norm, dtype=torch.float32, device=dev)  # (n_c, n_f, n_w, n_d)
        yt = xt[:, 0]  # (n_c, n_w, n_d) - target channel, normalized (channel 0)
        obs_t = torch.tensor(c["obs_mask"], device=dev)
        cal_t = torch.tensor(c["cal_idx"], device=dev)
        ctx_elig_t = torch.tensor(context_elig, device=dev)
        val_tgt_t = torch.tensor(val_target, device=dev)
        lob_t = torch.tensor(c["lob_idx"], dtype=torch.long, device=dev)
        prem_t = torch.tensor(
            (c["log_premium"] - prem_mean) / prem_std, dtype=torch.float32, device=dev
        )

        # augmented cutoffs are drawn from [min_cutoff, val_cutoff); clamp the
        # floor so at least one earlier diagonal remains to condition on.
        min_cutoff = max(1, min(cfg.min_cutoff, val_cutoff - 1))

        def make_model():
            model = net.TriangleTransformer(
                cfg, n_lob=len(c["lob_levels"]), n_features=n_f, n_w=n_w, n_d=n_d
            ).to(dev)
            if cfg.exposure_sigma:
                # the exposure factor un-normalizes log_premium; hand it the
                # pooled spread so p is a true log-dollar power.
                model.prem_log_std.fill_(prem_std)
            return model

        def train_loss(model, idx, cutoffs):
            # condition on cells on/before the augmented cutoff, score the
            # observed training cells strictly after it (card.md "Training")
            ctx = obs_t[idx] & (cal_t[None] <= cutoffs[:, None, None])  # (B, W, D)
            # targets are context-eligible (never validation) cells past the cutoff
            tgt = ctx_elig_t[idx] & (cal_t[None] > cutoffs[:, None, None])  # (B, W, D)
            if not bool(tgt.any()):
                return None
            log_pi, mu, sigma = model(xt[idx], ctx, lob_t[idx], prem_t[idx], cutoffs)
            return net.mdn_nll(log_pi, mu, sigma, yt[idx], tgt)

        def val_loss(model):
            val_cut = torch.full((n_c,), val_cutoff, dtype=torch.long, device=dev)
            log_pi, mu, sigma = model(xt, ctx_elig_t, lob_t, prem_t, val_cut)
            return float(net.mdn_nll(log_pi, mu, sigma, yt, val_tgt_t))

        # deep ensemble via the shared loop (gallery/nn/_training.py): member
        # seeding, cutoff augmentation batching, AdamW, early stopping. Pooling
        # the members' draws at rollout adds epistemic spread on top of the
        # MDN's aleatoric spread (card.md "Why an MDN head").
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
        self._loss_field = loss_field
        self.config_ = cfg
        self._device = device_str
        self.norm_ = norm
        self.models_, self.history_ = models, history
        # the cached rollout belongs to the previous fit; drop it whole
        self._rollout_key = None
        self._rollout_ults = None
        return self

    def predict(
        self,
        segment: dict[str, str] | None = None,
        n_draws: int | None = None,
        seed: int | None = None,
    ) -> PredictiveDistribution:
        """Predictive ultimates from the cached global rollout.

        With ``segment`` (e.g. {"company_code": ..., "line_of_business": ...}):
        per-origin ultimates for that one cohort plus their total - the same
        target layout as meyers_ccl. Without: every (cohort, origin) ultimate,
        no grand total (a total across companies is meaningless)."""
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        cfg = self.config_
        n_draws = n_draws or cfg.n_draws
        key = (n_draws, seed)
        if self._rollout_key != key:
            self._rollout_ults = self._rollout(n_draws, seed)
            self._rollout_key = key
        ults = self._rollout_ults
        c = self.contract_

        if segment is not None:
            ci = self._cohort_index(segment)
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
        return PredictiveDistribution(samples=ults.reshape(n_draws, -1), targets=targets)

    def realized_ultimates(
        self, full_triangle: Triangle, segment: dict[str, str] | None = None
    ) -> np.ndarray:
        """Outcomes aligned to predict(segment)'s targets, from the full
        triangle at the final dev lag (+ total when a segment is given)."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        seg_cols = list(c["cohorts"].columns)
        df = full_triangle.select_fields(self._loss_field).execute()
        df = df[df["dev_lag"] == c["n_d"] * c["dev_grain_months"]].copy()
        df["origin_period"] = _as_date(df["origin_period"])
        by_key = df.set_index([*seg_cols, "origin_period"])["value"]

        def lookup(row: pd.Series) -> np.ndarray:
            vals = [float(by_key.get((*row.tolist(), o), np.nan)) for o in c["origin_periods"]]
            return np.asarray(vals)

        if segment is not None:
            ci = self._cohort_index(segment)
            per_origin = lookup(c["cohorts"].iloc[ci])
            return np.append(per_origin, per_origin.sum())
        return np.concatenate([lookup(row) for _, row in c["cohorts"].iterrows()])

    # -- held-out scoring (milestone 6) ------------------------------------------

    def at_cohort(self, segment: dict[str, str]) -> CohortHeldout:
        """A per-cohort held-out scorer view (see ``gallery/nn/_heldout.py``).

        The pooled fit is multi-cohort while ``kernels.holdout`` scores one
        cohort at a time, so held-out capability is handed out per cohort: the
        view carries a single-cohort adapter contract that ``index_into``
        accepts unchanged (cohort-identity and training-overlap guards
        included), and its ``log_lik_at``/``predict_at`` are the unmodified
        mixin implementations."""
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        return CohortHeldout(self, self._cohort_index(segment))

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
            "nn_transformer's fit spans many cohorts and a bare CellIndex cannot name "
            "one; call log_lik_at(HoldoutCells) or at_cohort(segment).log_lik_at(...)"
        )

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError(
            "nn_transformer's fit spans many cohorts and a bare CellIndex cannot name "
            "one; call predict_at(HoldoutCells) or at_cohort(segment).predict_at(...)"
        )

    def _heldout_segment(self, cells) -> dict[str, str]:
        """The one segment combination the held-out cells describe."""
        if not isinstance(cells, HoldoutCells):
            raise TypeError(
                "nn_transformer resolves which cohort to score from the cells' segment "
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
        Lebesgue-on-amount (the increment/cumulative step has Jacobian 1).

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
                "the MDN head is untrained and its scale is degenerate. nn_transformer "
                "refuses to score a density at a pinned dev; the cells remain "
                "CRPS-scorable via predict_at, where a pinned draw is the pooled dev mean"
            )
        prev = np.asarray(cells.prev_value, dtype=float)
        if np.isnan(prev).any():
            raise ValueError(
                f"{int(np.isnan(prev).sum())} cell(s) have no training predecessor, so "
                "no increment can be formed to evaluate the density at"
            )
        premium = np.asarray(cells.premium, dtype=float)
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

        One forward pass per ensemble member at cutoff = the cohort's as_of
        diagonal (the held-out diagonal sits one step past it - the
        most-supervised relative-calendar position; no rollout needed), then
        ``mdn_sample`` at the requested cells, un-standardized
        (``z * std0[d] + mean0[d]``) and scaled by the cell's premium. Draws
        are split across members exactly as ``_rollout`` splits them; per-
        member torch seeds derive from ``rng``, so ``predict_at(seed=...)`` is
        reproducible.

        Pinned devs keep rollout semantics: the sampled ``z`` is forced to 0,
        i.e. the pooled dev mean after un-standardizing - a point-mass column,
        legal for CRPS as long as some requested cell is live. If EVERY
        requested cell is pinned the result would be a point mass everywhere,
        which is not a predictive distribution, so that is refused.
        """
        import torch

        from ibnr.gallery.nn.transformer import network as net

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
        premium = np.asarray(cells.premium, dtype=float)
        if np.isnan(premium).any():
            raise ValueError(
                f"{int(np.isnan(premium).sum())} cell(s) have no premium; draws are "
                "premium x sampled ratio, so they cannot be formed"
            )
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
                log_pi, mu, sigma = model(
                    inputs["x"], inputs["ctx"], inputs["lob"], inputs["prem"], inputs["cutoff"]
                )
                # index the grid at the requested cells, replicate per draw
                log_pi_c = log_pi[0, w0_t, d0_t].unsqueeze(0).expand(m_draws, -1, -1)
                mu_c = mu[0, w0_t, d0_t].unsqueeze(0).expand(m_draws, -1, -1)
                sigma_c = sigma[0, w0_t, d0_t].unsqueeze(0).expand(m_draws, -1, -1)
                gen = torch.Generator(device=dev)
                gen.manual_seed(m_seed)
                sample = net.mdn_sample(log_pi_c, mu_c, sigma_c, generator=gen)
                # pinned devs -> 0 (pooled dev mean after un-standardizing),
                # the same rule _rollout applies
                sample = sample.masked_fill(pin_t[None, :], 0.0)
                pieces.append(sample.cpu().numpy())
        draws = np.concatenate(pieces, axis=0)  # (n_draws, n_cells) standardized
        ratios = draws * std0[d0][None, :] + mean0[d0][None, :]
        return ratios * premium[None, :]  # incremental dollars

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
                params = model(
                    inputs["x"], inputs["ctx"], inputs["lob"], inputs["prem"], inputs["cutoff"]
                )
                for out, t in zip(acc, params, strict=True):
                    out.append(t[0, w0_t, d0_t].cpu().numpy())  # (n_cells, K)
        return tuple(np.stack(a) for a in acc)

    def _heldout_inputs(self, ci: int) -> dict:
        """One cohort's forward inputs, conditioned on everything it had at
        as_of: context = all its observed cells, cutoff = its deepest observed
        calendar diagonal (so the held-out diagonal sits at distance 1, the
        most-supervised relative-calendar position - the rollout's first step)."""
        import torch

        c = self.contract_
        # same standardize + pin as fit()/_rollout(), for this cohort only
        x_norm = (c["x"][ci] - self.norm_["mean"][:, None, :]) / self.norm_["std"][:, None, :]
        x_norm = np.where(self.norm_["pinned"][:, None, :], 0.0, x_norm)
        prem_norm = (c["log_premium"][ci] - self.norm_["prem_mean"]) / self.norm_["prem_std"]
        obs = c["obs_mask"][ci]  # (n_w, n_d)
        cut_level = int(c["cal_idx"][obs].max())
        dev = torch.device(self._device)
        return {
            "x": torch.tensor(x_norm[None], dtype=torch.float32, device=dev),
            "ctx": torch.tensor(obs[None], device=dev),
            "lob": torch.tensor([c["lob_idx"][ci]], dtype=torch.long, device=dev),
            "prem": torch.tensor([prem_norm], dtype=torch.float32, device=dev),
            "cutoff": torch.tensor([cut_level], dtype=torch.long, device=dev),
        }

    # -- internals ---------------------------------------------------------------

    def _cohort_index(self, segment: dict[str, str]) -> int:
        cohorts = self.contract_["cohorts"]
        mask = np.ones(len(cohorts), dtype=bool)
        for col, value in segment.items():
            if col not in cohorts.columns:
                raise KeyError(f"unknown segment column {col!r}; have {list(cohorts.columns)}")
            mask &= (cohorts[col] == value).to_numpy()
        idx = np.nonzero(mask)[0]
        if len(idx) != 1:
            raise ValueError(f"segment {segment} matches {len(idx)} cohorts, need exactly 1")
        return int(idx[0])

    def _rollout(self, n_draws: int, seed: int | None) -> np.ndarray:
        """Autoregressive rollout, diagonal by diagonal: sample every future
        cell on the next calendar diagonal, feed the samples back as context,
        re-encode, continue. Draws are pooled over the ensemble members.

        This is the transformer analogue of the chain-ladder recursion: the
        sequential re-encoding makes draws jointly coherent across cells within
        a cohort (dependence flows through the shared encoded context). Returns
        (n_draws, n_c, n_w): a draw of ultimate loss per (cohort, origin)."""
        import torch

        from ibnr.gallery.nn.transformer import network as net

        c = self.contract_
        n_c, _, n_w, n_d = c["x"].shape
        # channel-0 (target) unstandardization stats per dev, to turn sampled
        # normalized ratios back into loss-ratio dollars-per-premium.
        mean0, std0 = self.norm_["mean"][0], self.norm_["std"][0]  # (n_d,)
        dev = torch.device(self._device)

        # future cells: at/beyond each origin's latest observed dev (its anchor)
        # - everything to be predicted. latest_dev is 1-based, d_grid 0-based,
        # so `>=` includes the first unobserved dev.
        d_grid = np.arange(n_d)[None, None, :]
        future = d_grid >= c["latest_dev"][:, :, None]  # (n_c, n_w, n_d) bool
        # calendar diagonals that contain any future cell, in ascending order -
        # the rollout advances through these one at a time.
        cal_levels = sorted(np.unique(c["cal_idx"][future.any(axis=0)]))

        # same standardize + pin as fit(); (n_c, n_f, n_w, n_d)
        x_norm = (c["x"] - self.norm_["mean"][None, :, None, :]) / self.norm_["std"][
            None, :, None, :
        ]
        x_norm = np.where(self.norm_["pinned"][None, :, None, :], 0.0, x_norm)
        prem_norm = (c["log_premium"] - self.norm_["prem_mean"]) / self.norm_["prem_std"]
        # pinned devs have no trained head: their standardized value is 0 by
        # definition, so sampled draws are forced to 0 -> pooled dev mean
        pin_t = torch.tensor(self.norm_["pinned"][0], device=dev)  # (n_d,) target-channel pins
        xt = torch.tensor(x_norm, dtype=torch.float32, device=dev)  # (n_c, n_f, n_w, n_d)
        obs_t = torch.tensor(c["obs_mask"], device=dev)  # (n_c, n_w, n_d)
        fut_t = torch.tensor(future, device=dev)  # (n_c, n_w, n_d)
        cal_t = torch.tensor(c["cal_idx"], device=dev)  # (n_w, n_d)
        lob_t = torch.tensor(c["lob_idx"], dtype=torch.long, device=dev)  # (n_c,)
        prem_t = torch.tensor(prem_norm, dtype=torch.float32, device=dev)  # (n_c,)

        # split the requested draws as evenly as possible across ensemble
        # members (remainder spread over the first few members).
        n_members = len(self.models_)
        member_draws = [n_draws // n_members] * n_members
        for i in range(n_draws % n_members):
            member_draws[i] += 1
        # a forward pass stacks (chunk draws) x (n_c cohorts) rows; cap the row
        # count at MAX_ROLLOUT_BATCH so wide draw counts don't blow up memory.
        chunk_size = max(1, MAX_ROLLOUT_BATCH // n_c)

        pieces: list[np.ndarray] = []
        for member, (model, m_draws) in enumerate(zip(self.models_, member_draws, strict=True)):
            if m_draws == 0:
                continue
            gen = torch.Generator(device=dev)
            gen.manual_seed((0 if seed is None else seed) * 100003 + member)
            done = 0
            while done < m_draws:
                chunk = min(chunk_size, m_draws - done)
                # replicate each cohort `chunk` times (draws interleaved within
                # a cohort block): row = cohort*chunk + draw. Batch is
                # (chunk*n_c, n_f, n_w, n_d); xb/ctx are mutated in-place as the
                # rollout fills future cells, so they are cloned.
                xb = xt.repeat_interleave(chunk, dim=0).clone()  # (chunk*n_c, n_f, n_w, n_d)
                ctx = obs_t.repeat_interleave(chunk, dim=0).clone()  # (chunk*n_c, n_w, n_d)
                futb = fut_t.repeat_interleave(chunk, dim=0)  # (chunk*n_c, n_w, n_d)
                lobb = lob_t.repeat_interleave(chunk, dim=0)  # (chunk*n_c,)
                premb = prem_t.repeat_interleave(chunk, dim=0)  # (chunk*n_c,)
                with torch.no_grad():
                    for lv in cal_levels:
                        # future cells sitting exactly on this diagonal
                        cells = futb & (cal_t[None] == lv)  # (chunk*n_c, n_w, n_d)
                        if not bool(cells.any()):
                            continue
                        # context boundary advances with each sampled diagonal,
                        # so the predicted diagonal always sits at distance 1 -
                        # the most-supervised relative-calendar position.
                        cut_b = torch.full(
                            (xb.shape[0],), int(lv) - 1, dtype=torch.long, device=dev
                        )
                        log_pi, mu, sigma = model(xb, ctx, lobb, premb, cut_b)
                        # sample: (chunk*n_c, n_w, n_d) - one draw per cell
                        sample = net.mdn_sample(log_pi, mu, sigma, generator=gen)
                        # pinned devs -> 0 (pooled dev mean after unstandardizing)
                        sample = sample.masked_fill(pin_t[None, None, :], 0.0)
                        # write sampled cells into the target channel and promote
                        # them to context for the next diagonal (autoregression)
                        xb[:, 0][cells] = sample[cells]
                        ctx = ctx | cells
                # unstandardize the target channel back to loss ratios, then sum
                # only the future increments per (row, origin).
                ratios = xb[:, 0].cpu().numpy() * std0[None, None, :] + mean0[None, None, :]
                contrib = (ratios * future.repeat(chunk, axis=0)).sum(axis=2)  # (chunk*n_c, n_w)
                contrib = contrib.reshape(n_c, chunk, n_w, order="C")  # undo interleave
                # ultimate = anchor cumulative + premium * summed future increments
                ults = c["latest_cum"][:, None, :] + contrib * c["premium"][:, None, :]
                pieces.append(np.moveaxis(ults, 1, 0))  # (n_c, chunk, n_w) -> (chunk, n_c, n_w)
                done += chunk
        return np.concatenate(pieces, axis=0)  # (n_draws, n_c, n_w)
