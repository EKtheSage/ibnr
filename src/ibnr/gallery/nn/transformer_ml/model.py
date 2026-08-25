"""Multi-line triangle transformer gallery entry. See card.md.

The research entry: one encoder attends over every (line, origin, dev) cell
of a COMPANY, and cross-line dependence enters the draws either by
within-diagonal line-by-line autoregression ("ar") or an explicit joint
Gaussian-mixture head ("joint") - config.dependence. Torch is imported
inside fit()/predict() only. Training scheme (calendar-cutoff augmentation,
trailing-diagonal validation, pinned per-dev standardization, deep
ensembling, diagonal-by-diagonal rollout) mirrors nn_transformer; the
normalization and split helpers are imported from it - one implementation.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ibnr.gallery.entry import GalleryEntry, PredictsHeldout, ScoresHeldout
from ibnr.gallery.nn._heldout import (
    cell_premium,
    heldout_segment,
    mixture_log_density,
    refuse_all_pinned_draws,
    refuse_missing_predecessor,
    refuse_pinned_density,
    require_ensemble,
    sample_mixture_draws,
    standardized_increment,
    unstandardize_to_amounts,
)
from ibnr.gallery.nn._heldout_ml import LOB_COLUMN, MLCohortHeldout, company_line_cutoff
from ibnr.gallery.nn._scheme import norm_stats, splits
from ibnr.gallery.nn._training import train_ensemble
from ibnr.gallery.nn.transformer_ml.config import TransformerMLConfig
from ibnr.gallery.registry import register
from ibnr.kernels.contract import _as_date
from ibnr.kernels.holdout import CellIndex, HoldoutCells
from ibnr.kernels.multiline import assemble_predictive, flatten_with_totals, multiline_targets
from ibnr.kernels.nn_contract import cohort_identities, nn_company_data
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

#: cap on (draw chunk x companies) per rollout forward pass
MAX_ROLLOUT_BATCH = 2048


@register
class NNTransformerML(GalleryEntry, ScoresHeldout, PredictsHeldout):
    """Multi-line triangle transformer entry (see card.md).

    One encoder is fit across every company in the training triangle (each
    company = one multi-line example); ``predict(segment=company)`` then rolls
    the future diagonals of that company forward and returns the SUR target
    layout (per-(lob, origin) ultimates, per-lob totals, grand total) so
    cross-line diversification is visible in the draws and directly comparable
    to the ``sur`` / ``copula_glm`` baselines.

    Held-out scoring is served per (company, line) pair, not per company: the
    fit's cohort is a company while ``next_diagonal``'s is a (company, line)
    pair. ``gallery/nn/_heldout_ml.py`` is the adapter that bridges the two, and
    the density algebra and draw loop come from ``gallery/nn/_heldout.py``, so
    this entry scores by the same code as the single-line ones.
    """

    name = "nn_transformer_ml"
    family = "nn"
    #: the dataclass ``fit(config=...)`` takes, reachable through
    #: ``gallery.get("nn_transformer_ml").config_class`` without importing it by path
    config_class = TransformerMLConfig

    #: both heads are densities of the STANDARDIZED incremental loss ratio;
    #: ``_heldout_log_lik`` folds the standardization Jacobian (``-log std0[d]``)
    #: in, leaving a density on the loss RATIO - this declaration then makes
    #: ``ScoresHeldout.log_lik_at`` subtract ``log premium`` to reach
    #: Lebesgue-on-amount. See card.md "Held-out scoring".
    heldout_measure = "loss_ratio"

    #: a draw is ``premium x un-standardized ratio`` - an INCREMENTAL dollar
    #: amount, one dev step's emergence for one line. The Schedule P triangles
    #: are cumulative, so ``PredictsHeldout.predict_at`` adds each cell's
    #: training-diagonal anchor; declaring the scale is what makes that
    #: conversion the base class's job rather than a silent 996-vs-3.4 bug.
    heldout_draw_scale = "incremental"

    def __init__(self) -> None:
        self.contract_: dict | None = None  # nn_company_data() dict
        self.config_: TransformerMLConfig | None = None
        self.models_: list | None = None  # the deep-ensemble members
        self.norm_: dict | None = None  # pinned per-(line, channel, dev) standardization
        self.history_: list[list[dict]] | None = None  # per-member train/val NLL curves
        self._loss_field: str | None = None
        self._device: str = "cpu"
        # rollout is expensive; cache it keyed on (n_draws, seed)
        self._rollout_key: tuple | None = None
        self._rollout_ults: np.ndarray | None = None  # (n_draws, n_c, n_l, n_w)

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "reported_loss",
        feature_fields: tuple[str, ...] = (),
        level_fields: tuple[str, ...] = (),
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        config: TransformerMLConfig | None = None,
        device: str | None = None,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> NNTransformerML:
        """Pooled fit across every company in the triangle; each company's
        lines are one joint training example.

        ``feature_fields`` names further triangle fields carried as input
        channels beside the target (channel 0); ``level_fields`` declares
        which of them are eval-date SNAPSHOTS carried undifferenced -
        ``case_reserve`` is the motivating one, whose difference is the case
        movement while the informative quantity is the outstanding level.
        Both are the contract's vocabulary; see
        ``kernels.nn_contract.nn_data`` for the semantics and the refusals."""
        import torch

        from ibnr.gallery.nn.transformer import network as net1
        from ibnr.gallery.nn.transformer_ml import network as net

        cfg = config or TransformerMLConfig()
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # BUILD FIRST, ASSIGN AFTER TRAINING SUCCEEDED - fit() must be atomic:
        # a failed refit must not leave the new pool's contract/normalizer over
        # the old pool's networks (see the single-line entry and mack).
        contract = nn_company_data(
            train,
            loss_field=loss_field,
            feature_fields=feature_fields,
            level_fields=level_fields,
            premium_field=premium_field,
        )
        device_str = device or "cpu"
        c = contract
        n_c, n_l, n_f, n_w, n_d = c["x"].shape

        # validation split: hold out the last cfg.val_diagonals calendar
        # diagonals (an eval_date-style split). A cutoff is per company but
        # spans all its lines - calendar is shared across a company's lines.
        obs_any = c["obs_mask"].any(axis=1)  # (n_c, n_w, n_d): any line observed
        _, _, val_cutoff = splits(obs_any, c["cal_idx"], cfg.val_diagonals)
        cal = c["cal_idx"]  # (n_w, n_d) 1-based diagonal index
        ctx_elig = c["obs_mask"] & (cal[None, None] <= val_cutoff)  # target cells trainable
        val_tgt = c["obs_mask"] & (cal[None, None] > val_cutoff)  # held-out validation targets
        # what the network CONDITIONS on: each channel's own usable values under
        # the same calendar constraint the target context uses. The target
        # vocabulary above stays target-only - it selects prediction targets,
        # which are always channel 0.
        chan_elig = c["x_obs"] & (cal[None, None, None] <= val_cutoff)  # (n_c, L, F, W, D)

        # pinned per-(line, channel, dev) standardization - run the single-line
        # helper once per line so the rules are identical by construction. A dev
        # with <2 context values is "pinned": standardized value forced to 0 and
        # rollout draws there replaced by the pooled dev mean (v2 fix, see card).
        # Stats come from each channel's OWN observed cells, so a feature that is
        # missing where the target is observed is not standardized against the
        # contract's padding zeros.
        mean = np.zeros((n_l, n_f, n_d))
        std = np.ones((n_l, n_f, n_d))
        pinned = np.zeros((n_l, n_f, n_d), dtype=bool)
        for li in range(n_l):
            mean[li], std[li], pinned[li] = norm_stats(
                c["x"][:, li], chan_elig[:, li], c["x_obs"][:, li]
            )
        # normalize log premium over the lines actually present (absent = 0 later)
        lp = c["log_premium"][c["line_mask"]]
        prem_mean, prem_std = float(lp.mean()), float(lp.std())
        if prem_std < 1e-8:
            prem_std = 1.0
        norm = {
            "mean": mean,
            "std": std,
            "pinned": pinned,
            "prem_mean": prem_mean,
            "prem_std": prem_std,
        }

        # standardize channels; pinned devs are forced to 0; target channel is
        # x[:, :, 0] (the incremental loss ratio the model predicts)
        x_norm = (c["x"] - mean[None, :, :, None, :]) / std[None, :, :, None, :]
        x_norm = np.where(pinned[None, :, :, None, :], 0.0, x_norm)
        prem_norm = np.where(c["line_mask"], (c["log_premium"] - prem_mean) / prem_std, 0.0)

        dev = torch.device(device_str)
        xt = torch.tensor(x_norm, dtype=torch.float32, device=dev)  # (n_c, L, F, W, D)
        yt = xt[:, :, 0]  # (n_c, L, W, D) target channel
        xobs_t = torch.tensor(c["x_obs"], device=dev)  # (n_c, L, F, W, D)
        cal_t = torch.tensor(cal, device=dev)
        ctx_elig_t = torch.tensor(ctx_elig, device=dev)
        chan_elig_t = torch.tensor(chan_elig, device=dev)
        val_tgt_t = torch.tensor(val_tgt, device=dev)
        lm_t = torch.tensor(c["line_mask"], device=dev)
        prem_t = torch.tensor(prem_norm, dtype=torch.float32, device=dev)
        val_cut_t = torch.full((n_c,), val_cutoff, dtype=torch.long, device=dev)

        # dispatch the loss to the head the config selected: the single-line
        # univariate MDN NLL ("ar") vs the joint MVN-mixture NLL ("joint")
        def nll(model, xb, ctxb, lmb, premb, cutb, yb, tgtb):
            if cfg.dependence == "ar":
                log_pi, mu, sigma = model.forward_ar(xb, ctxb, lmb, premb, cutb)
                return net1.mdn_nll(log_pi, mu, sigma, yb, tgtb)
            log_pi, mu, scale = model.forward_joint(xb, ctxb, lmb, premb, cutb)
            return net.joint_mdn_nll(log_pi, mu, scale, yb, tgtb)

        # earliest cutoff the augmentation may draw (leave >= 1 target diagonal)
        min_cutoff = max(1, min(cfg.min_cutoff, val_cutoff - 1))

        def make_model():
            return net.TriangleTransformerML(cfg, n_lines=n_l, n_features=n_f, n_w=n_w, n_d=n_d).to(
                dev
            )

        def train_loss(model, idx, cutoffs):
            # cells on/before the augmented cutoff are context, later observed
            # cells are the prediction targets - this teaches the model to
            # forecast future diagonals. The context is per channel and the
            # targets are channel 0's, under one calendar gate. The rollout
            # gates features by observedness alone (no calendar clamp), so a
            # feature booked past every target cell reaches it at a distance
            # training never shows - the disclosed edge (card.md).
            ctx = xobs_t[idx] & (cal_t[None, None, None] <= cutoffs[:, None, None, None, None])
            tgt = ctx_elig_t[idx] & (cal_t[None, None] > cutoffs[:, None, None, None])
            if not bool(tgt.any()):
                return None
            return nll(model, xt[idx], ctx, lm_t[idx], prem_t[idx], cutoffs, yt[idx], tgt)

        def val_loss(model):
            # condition on all trainable cells, score the held-out trailing
            # diagonals at the fixed val_cutoff
            return float(nll(model, xt, chan_elig_t, lm_t, prem_t, val_cut_t, yt, val_tgt_t))

        # deep ensemble via the shared loop (gallery/nn/_training.py):
        # cfg.ensemble_size independently-seeded fits; their draws are pooled
        # at predict time to widen the predictive distribution
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

    def cohorts(self) -> list[dict]:
        """One dict per COMPANY, in ``contract_["companies"]`` row order.

        The cohort unit here is the company (all its lines at once), so the
        identity carries no ``line_of_business`` - and it does carry any
        display-only segment column the key dropped (see
        :meth:`GalleryEntry.cohorts`).
        """
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        return cohort_identities(self.contract_, key="companies")

    def predict(
        self,
        segment: Mapping | None = None,
        n_draws: int | None = None,
        seed: int | None = None,
    ) -> PredictiveDistribution:
        """With ``segment`` identifying one company: the SUR layout -
        per-(lob, origin) ultimates for its present lines, per-lob totals,
        grand total - so cross-line diversification is visible in the draws.
        Without: every (company, line, origin) ultimate, no totals."""
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        # resolve the company BEFORE the rollout: a bad segment must not cost one
        ci = self.cohort_index(segment)
        cfg = self.config_
        n_draws = n_draws or cfg.n_draws
        # one rollout serves every company; reuse it across predict() calls
        key = (n_draws, seed)
        if self._rollout_key != key:
            self._rollout_ults = self._rollout(n_draws, seed)
            self._rollout_key = key
        ults = self._rollout_ults  # (n_draws, n_c, n_l, n_w)
        c = self.contract_

        if ci is not None:
            # one company: emit the SUR layout over its present lines only.
            # assemble_predictive derives lob/grand totals as row-sums of the
            # SAME draws, so cross-line diversification stays coherent.
            present = np.nonzero(c["line_mask"][ci])[0]  # line indices this company writes
            lobs = [c["lob_levels"][li] for li in present]
            targets = multiline_targets(
                lobs, c["origin_periods"], premium=c["premium"][ci, present]
            )
            return assemble_predictive(ults[:, ci, present], targets)

        # no segment: flat per-(company, line, origin) ultimates, no totals
        n_c, n_l, n_w = ults.shape[1:]
        rows = []
        for ci in range(n_c):
            for li in np.nonzero(c["line_mask"][ci])[0]:
                for w, origin in enumerate(c["origin_periods"]):
                    rows.append(
                        {
                            **dict(c["companies"].iloc[ci]),
                            "line_of_business": c["lob_levels"][li],
                            "origin_period": origin,
                            "premium": c["premium"][ci, li, w],
                        }
                    )
        # drop the padding columns of absent (company, line) pairs, matching
        # the row order built above
        keep = c["line_mask"][:, :, None] & np.ones((1, 1, n_w), dtype=bool)
        samples = ults.reshape(n_draws, -1)[:, keep.reshape(-1)]
        return PredictiveDistribution(samples=samples, targets=pd.DataFrame(rows))

    def realized_ultimates(
        self, full_triangle: Triangle, segment: Mapping | None = None
    ) -> np.ndarray:
        """Outcomes aligned to predict()'s targets, from the full triangle
        at the final dev lag (with lob/grand totals when a segment is given)."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        ci = self.cohort_index(segment)
        c = self.contract_
        company_cols = list(c["companies"].columns)
        df = full_triangle.select_fields(self._loss_field).execute()
        df = df[df["dev_lag"] == c["n_d"] * c["dev_grain_months"]].copy()
        df["origin_period"] = _as_date(df["origin_period"])
        by_key = df.set_index([*company_cols, "line_of_business", "origin_period"])["value"]

        def lookup(company_row, lob) -> np.ndarray:
            vals = [float(by_key.get((*company_row, lob, o), np.nan)) for o in c["origin_periods"]]
            return np.asarray(vals)

        if ci is not None:
            crow = tuple(c["companies"].iloc[ci])
            present = np.nonzero(c["line_mask"][ci])[0]
            grid = np.stack([lookup(crow, c["lob_levels"][li]) for li in present])
            return flatten_with_totals(grid)
        out = []
        for ci in range(len(c["companies"])):
            crow = tuple(c["companies"].iloc[ci])
            for li in np.nonzero(c["line_mask"][ci])[0]:
                out.append(lookup(crow, c["lob_levels"][li]))
        return np.concatenate(out)

    # -- held-out scoring --------------------------------------------------------

    def at_cohort(self, segment: Mapping) -> MLCohortHeldout:
        """A per-(company, line) held-out scorer view.

        The fit's cohort is a COMPANY - all its lines are one training example -
        while ``kernels.holdout`` scores one (company, line) pair at a time, so
        the segment has to name both. The view carries a single-pair adapter
        contract that ``index_into`` accepts unchanged (cohort-identity and
        training-overlap guards included), and its ``log_lik_at``/``predict_at``
        are the unmodified mixin implementations.
        """
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        if not isinstance(segment, Mapping):
            raise TypeError(
                f"{self.name}: at_cohort needs a mapping of segment column -> value, "
                f"got {type(segment).__name__}"
            )
        if LOB_COLUMN not in segment:
            raise ValueError(
                f"{self.name}: at_cohort needs {LOB_COLUMN!r} in the segment. This fit's "
                f"cohort is a company (keyed on {list(self.contract_['segment_columns'])}), "
                "but a held-out cohort is one (company, line) pair, because next_diagonal "
                "builds cells for a single line at a time - so the line has to be named. "
                f"Got {sorted(segment)}"
            )
        company_keys = {k: v for k, v in segment.items() if k != LOB_COLUMN}
        ci = self.cohort_index(company_keys)
        if ci is None:  # pragma: no cover - cohort_index answers None only to None
            raise ValueError(f"{self.name}: at_cohort needs a segment naming one company")
        levels = list(self.contract_["lob_levels"])
        wanted = segment[LOB_COLUMN]
        if wanted not in levels:
            raise ValueError(
                f"{self.name}: unknown {LOB_COLUMN} {wanted!r}; this fit's line axis is "
                f"{levels}. The line axis is the training triangle's whole vocabulary, so "
                "a line that is not on it was never fitted for any company"
            )
        return MLCohortHeldout(self, ci, levels.index(wanted))

    def log_lik_at(self, cells, *, field: str | None = None) -> np.ndarray:
        """``(n_members, n_cells)`` log density on Lebesgue-on-amount.

        Resolves the (company, line) pair from the cells' own segment values -
        which include ``line_of_business``, exactly what :meth:`at_cohort` needs
        - then delegates to that view, whose ``log_lik_at`` IS
        ``ScoresHeldout.log_lik_at``: the measure carry and every guard run in
        the base class against the per-pair contract."""
        return self.at_cohort(heldout_segment(self.name, cells)).log_lik_at(cells, field=field)

    def predict_at(
        self, cells: HoldoutCells, *, field: str | None = None, seed: int | None = None
    ) -> np.ndarray:
        """``(n_draws, n_cells)`` draws on the TRIANGLE's basis, in cell order.

        Same delegation as :meth:`log_lik_at`: the view's ``predict_at`` is
        ``PredictsHeldout.predict_at`` unchanged, so the incremental draws are
        anchored onto each cell's training-diagonal predecessor in the base
        class, and the draw stream is derived from ``seed`` together with the
        cells' cohort identity there rather than here - which is what gives two
        lines of one company different random numbers under one study seed."""
        return self.at_cohort(heldout_segment(self.name, cells)).predict_at(
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
            f"{self.name}'s fit spans many (company, line) pairs and a bare CellIndex "
            "cannot name one; call log_lik_at(HoldoutCells) or "
            "at_cohort(segment).log_lik_at(...)"
        )

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name}'s fit spans many (company, line) pairs and a bare CellIndex "
            "cannot name one; call predict_at(HoldoutCells) or "
            "at_cohort(segment).predict_at(...)"
        )

    def _cell_identity(self) -> dict:
        """The pooled entry cannot be scored without binding a pair first.

        Its ``contract_`` is the multi-company ``nn_company_data`` dict, which
        carries no single ``segment`` - and ``log_lik_at``/``predict_at`` here
        delegate to an :class:`MLCohortHeldout`, which supplies the real
        identity.
        """
        raise RuntimeError(
            f"{self.name}'s fit spans many companies and lines and has no single "
            "identity; bind one with at_cohort(segment) first"
        )

    def _heldout_log_lik(self, cohort: tuple[int, int], cells: CellIndex) -> np.ndarray:
        """``(n_members, n_cells)`` log density of one line's loss RATIO.

        Per ensemble member: the mixture density of the STANDARDIZED increment
        ratio ``z = (increment/premium - mean0[d]) / std0[d]``, with the
        standardization Jacobian folded in, leaving a density on the ratio -
        which is what ``heldout_measure = "loss_ratio"`` declares, so the base
        class's ``- log premium`` completes the carry to Lebesgue-on-amount.
        Every per-dev statistic is read at the SCORED LINE's row of ``norm_``,
        because the normalizer is estimated per (line, channel, dev) and another
        line's scale would standardize the observation against a spread the head
        never saw.

        The draw axis is the ENSEMBLE MEMBERS, so ``logmeanexp`` over it is the
        ensemble-average predictive density - two members minimum.

        Cells at a PINNED dev are refused: no trained head exists there and the
        standardized scale is degenerate. The same cells remain CRPS-scorable
        through :meth:`predict_at` - the documented asymmetry (card.md "Held-out
        scoring").
        """
        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        ci, li = int(cohort[0]), int(cohort[1])
        require_ensemble(len(self.models_))
        d0 = np.asarray(cells.d, dtype=int) - 1
        refuse_pinned_density(self.name, d0, self.norm_["pinned"][li, 0])
        prev = np.asarray(cells.prev_value, dtype=float)
        refuse_missing_predecessor(prev)
        premium = self._cell_premium((ci, li), cells)
        mean0, std0 = self.norm_["mean"][li, 0], self.norm_["std"][li, 0]  # (n_d,)
        z = standardized_increment(
            np.asarray(cells.value, dtype=float),
            prev,
            premium,
            mean=mean0[d0],
            std=std0[d0],
        )
        log_pi, mu, sigma = self._heldout_mixture((ci, li), cells)  # (n_members, n_cells, K)
        return mixture_log_density(log_pi, mu, sigma, z, std=std0[d0])

    def _heldout_draws(
        self, cohort: tuple[int, int], cells: CellIndex, *, rng: np.random.Generator
    ) -> np.ndarray:
        """``(config.heldout_n_draws, n_cells)`` INCREMENTAL dollar draws.

        One forward pass per ensemble member at the company's as_of conditioning
        (:meth:`_heldout_inputs`) - the held-out diagonal is one step past that
        context, the most-supervised position and the rollout's first step, so
        no rollout is needed here. Then mixture sampling at the requested cells,
        un-standardized on the scored line's own per-dev statistics and scaled
        by the contract's premium.

        Pinned devs keep rollout semantics: the sampled value is forced to the
        pooled dev mean, a point-mass column, legal for CRPS as long as some
        requested cell is live. A request whose every cell is pinned is refused.
        """
        import torch

        if self.models_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        ci, li = int(cohort[0]), int(cohort[1])
        d0 = np.asarray(cells.d, dtype=int) - 1
        pin_cells = self.norm_["pinned"][li, 0][d0]  # (n_cells,)
        refuse_all_pinned_draws(pin_cells)
        premium = self._cell_premium((ci, li), cells)
        mean0, std0 = self.norm_["mean"][li, 0], self.norm_["std"][li, 0]  # (n_d,)
        dev = torch.device(self._device)
        inputs = self._heldout_inputs(ci, li)
        w0_t = torch.as_tensor(np.asarray(cells.w, dtype=int) - 1, device=dev)
        d0_t = torch.as_tensor(d0, device=dev)

        def cell_params(model):
            return self._cell_mixture(model, inputs, li, w0_t, d0_t)

        # config_.n_draws is the ROLLOUT's budget and is not read here: a
        # held-out diagonal costs one forward pass per member whatever the draw
        # count, so the two are sized independently.
        draws = sample_mixture_draws(
            self.models_,
            n_draws=self.config_.heldout_n_draws,
            rng=rng,
            cell_params=cell_params,
            pinned_cells=pin_cells,
            device=dev,
        )
        return unstandardize_to_amounts(
            draws, mean=mean0[d0], std=std0[d0], premium=premium
        )  # incremental dollars

    def _heldout_mixture(self, cohort: tuple[int, int], cells: CellIndex) -> tuple[np.ndarray, ...]:
        """``(n_members, n_cells, K)`` univariate mixture parameters of the
        SCORED LINE, one forward pass per ensemble member at the company's as_of
        conditioning."""
        import torch

        ci, li = int(cohort[0]), int(cohort[1])
        inputs = self._heldout_inputs(ci, li)
        dev = torch.device(self._device)
        w0_t = torch.as_tensor(np.asarray(cells.w, dtype=int) - 1, device=dev)
        d0_t = torch.as_tensor(np.asarray(cells.d, dtype=int) - 1, device=dev)
        acc: tuple[list, list, list] = ([], [], [])
        with torch.no_grad():
            for model in self.models_:
                params = self._cell_mixture(model, inputs, li, w0_t, d0_t)
                for out, t in zip(acc, params, strict=True):
                    out.append(t.cpu().numpy())  # (n_cells, K)
        return tuple(np.stack(a) for a in acc)

    def _cell_mixture(self, model, inputs: dict, li: int, w0_t, d0_t) -> tuple:
        """One member's ``(log_pi, mu, sigma)`` for one line at the requested
        cells, each ``(n_cells, K)``.

        The one place the two heads differ on the held-out path.

        - ``"ar"``: the head is already univariate per (line, origin, dev) cell,
          so the line is just an axis to index.
        - ``"joint"``: the head is a mixture of L-variate Gaussians over the
          whole line vector, and what a one-line score needs is its MARGINAL.
          The marginal of a mixture of multivariate Gaussians is the mixture of
          the components' marginals with the weights UNCHANGED, so the mixture
          weights carry over as they are, the mean is component k's entry for
          this line, and the standard deviation is the square root of that
          line's diagonal entry of the covariance. With ``Cov = L L'`` that
          entry is ``sum_j L[li, j]^2``, i.e. the length of row ``li`` of the
          Cholesky factor - taken directly rather than by forming ``Cov``,
          which would build an ``(L, L)`` matrix per component per cell to read
          one number off it.
        """
        import torch

        args = (
            inputs["x"],
            inputs["ctx"],
            inputs["line_mask"],
            inputs["prem"],
            inputs["cutoff"],
        )
        if self.config_.dependence == "ar":
            log_pi, mu, sigma = model.forward_ar(*args)  # each (1, L, W, D, K)
            return log_pi[0, li, w0_t, d0_t], mu[0, li, w0_t, d0_t], sigma[0, li, w0_t, d0_t]
        # (1, W, D, K), (1, W, D, K, L), (1, W, D, K, L, L)
        log_pi, mu, scale_tril = model.forward_joint(*args)
        weights = log_pi[0, w0_t, d0_t]  # (n_cells, K)
        mu_li = mu[0, w0_t, d0_t][..., li]  # (n_cells, K)
        sigma_li = torch.linalg.vector_norm(scale_tril[0, w0_t, d0_t][:, :, li, :], dim=-1)
        return weights, mu_li, sigma_li

    def _heldout_inputs(self, ci: int, li: int) -> dict:
        """One company's forward inputs, as already-batched torch tensors.

        Context is everything the COMPANY observed at as_of, per channel and
        across ALL its lines (the contract's ``x_obs``), because a held-out cell
        of one line is exactly what the other lines' reported experience is
        supposed to inform - that is the entry's reason to exist. The cutoff is
        the SCORED LINE's own as_of diagonal, so its held-out cell sits at
        distance 1 in the relative calendar embedding; see
        :func:`~ibnr.gallery.nn._heldout_ml.company_line_cutoff` for why the two
        are read differently and why they coincide on complete squares.
        """
        import torch

        c, n = self.contract_, self.norm_
        # same standardize + pin as fit()/_rollout(), for this company only
        x_norm = (c["x"][ci] - n["mean"][:, :, None, :]) / n["std"][:, :, None, :]
        x_norm = np.where(n["pinned"][:, :, None, :], 0.0, x_norm)
        prem_norm = np.where(
            c["line_mask"][ci],
            (c["log_premium"][ci] - n["prem_mean"]) / n["prem_std"],
            0.0,
        )
        cut_level = company_line_cutoff(c, ci, li)
        dev = torch.device(self._device)
        return {
            "x": torch.tensor(x_norm[None], dtype=torch.float32, device=dev),
            "ctx": torch.tensor(c["x_obs"][ci][None], device=dev),
            "line_mask": torch.tensor(c["line_mask"][ci][None], device=dev),
            "prem": torch.tensor(prem_norm[None], dtype=torch.float32, device=dev),
            "cutoff": torch.tensor([cut_level], dtype=torch.long, device=dev),
        }

    def _cell_premium(self, cohort: tuple[int, int], cells: CellIndex) -> np.ndarray:
        """This (company, line) pair's row of the contract's premium grid,
        checked against the cells' own. See
        :func:`~ibnr.gallery.nn._heldout.cell_premium` for why it is checked and
        not ignored."""
        ci, li = int(cohort[0]), int(cohort[1])
        return cell_premium(self.name, self.contract_["premium"][ci, li], cells)

    # -- internals ---------------------------------------------------------------

    def _rollout(self, n_draws: int, seed: int | None) -> np.ndarray:
        """Diagonal-by-diagonal autoregressive rollout over every company.

        "ar": within a diagonal the lines are sampled one at a time in a
        seeded random order (fresh order per diagonal per chunk), each fed
        back before the next - cross-line dependence via conditioning.
        "joint": one forward per diagonal; every (origin, dev) cell-group's
        line vector is drawn jointly from the multivariate mixture.

        Either way only the TARGET channel is fed back: feature channels are
        not simulated forward, so the rollout conditions on progressively
        fewer observed channels as it goes deeper (see card.md).
        """
        import torch

        from ibnr.gallery.nn.transformer import network as net1
        from ibnr.gallery.nn.transformer_ml import network as net

        c = self.contract_
        cfg = self.config_
        n_c, n_l, _, n_w, n_d = c["x"].shape
        # channel 0 = target; rollout only ever samples/denormalizes that channel
        mean0, std0 = self.norm_["mean"][:, 0], self.norm_["std"][:, 0]  # (n_l, n_d)
        pin0 = self.norm_["pinned"][:, 0]  # (n_l, n_d)
        dev = torch.device(self._device)

        # future = cells strictly past each (company, line)'s latest observed dev
        # and belonging to a written line - the increments we must simulate
        d_grid = np.arange(n_d)[None, None, None, :]
        future = (d_grid >= c["latest_dev"][:, :, :, None]) & c["line_mask"][:, :, None, None]
        # calendar diagonals to fill, in order - the rollout is autoregressive
        # across these (each conditions on all earlier ones)
        cal_levels = sorted(np.unique(c["cal_idx"][future.any(axis=(0, 1))]))

        x_norm = (c["x"] - self.norm_["mean"][None, :, :, None, :]) / self.norm_["std"][
            None, :, :, None, :
        ]
        x_norm = np.where(self.norm_["pinned"][None, :, :, None, :], 0.0, x_norm)
        prem_norm = np.where(
            c["line_mask"],
            (c["log_premium"] - self.norm_["prem_mean"]) / self.norm_["prem_std"],
            0.0,
        )
        xt = torch.tensor(x_norm, dtype=torch.float32, device=dev)
        xobs_t = torch.tensor(c["x_obs"], device=dev)  # (n_c, L, F, W, D)
        fut_t = torch.tensor(future, device=dev)
        cal_t = torch.tensor(c["cal_idx"], device=dev)
        lm_t = torch.tensor(c["line_mask"], device=dev)
        prem_t = torch.tensor(prem_norm, dtype=torch.float32, device=dev)
        pin_t = torch.tensor(pin0, device=dev)  # (n_l, n_d)

        # split the requested draws roughly evenly over ensemble members
        n_members = len(self.models_)
        member_draws = [n_draws // n_members] * n_members
        for i in range(n_draws % n_members):
            member_draws[i] += 1
        # each forward pass replicates all n_c companies, so cap the draw chunk
        chunk_size = max(1, MAX_ROLLOUT_BATCH // n_c)
        premium = np.where(np.isnan(c["premium"]), 0.0, c["premium"])  # (n_c, n_l, n_w)

        pieces: list[np.ndarray] = []
        for member, (model, m_draws) in enumerate(zip(self.models_, member_draws, strict=True)):
            if m_draws == 0:
                continue
            gen = torch.Generator(device=dev)
            gen.manual_seed((0 if seed is None else seed) * 100003 + member)
            # separate rng seeds the per-diagonal line order for the "ar" head
            order_rng = np.random.default_rng((0 if seed is None else seed) * 7919 + member)
            done = 0
            while done < m_draws:
                chunk = min(chunk_size, m_draws - done)
                # replicate each company `chunk` times -> batch B = n_c * chunk,
                # one independent posterior draw per replica
                xb = xt.repeat_interleave(chunk, dim=0).clone()
                ctx = xobs_t.repeat_interleave(chunk, dim=0).clone()  # (B, L, F, W, D)
                futb = fut_t.repeat_interleave(chunk, dim=0)
                lmb = lm_t.repeat_interleave(chunk, dim=0)
                premb = prem_t.repeat_interleave(chunk, dim=0)
                with torch.no_grad():
                    # fill one calendar diagonal at a time; a sampled cell's
                    # TARGET channel becomes context before the next diagonal is
                    # encoded - its feature channels stay unobserved, since
                    # nothing simulates them forward
                    for lv in cal_levels:
                        cells = futb & (cal_t[None, None] == lv)  # (B, L, W, D) cells on this diag
                        if not bool(cells.any()):
                            continue
                        # cutoff = lv - 1: everything strictly before this diagonal
                        # is the conditioning set for this encode
                        cut_b = torch.full(
                            (xb.shape[0],), int(lv) - 1, dtype=torch.long, device=dev
                        )
                        if cfg.dependence == "ar":
                            # dependence via conditioning: draw the lines of this
                            # diagonal ONE AT A TIME in a fresh random order, each
                            # fed back as context before the next. Re-encoding per
                            # line is what lets attention couple them.
                            for li in order_rng.permutation(n_l):
                                li = int(li)
                                cells_l = cells[:, li]
                                if not bool(cells_l.any()):
                                    continue
                                log_pi, mu, sigma = model.forward_ar(xb, ctx, lmb, premb, cut_b)
                                sample = net1.mdn_sample(log_pi, mu, sigma, generator=gen)
                                # pinned devs are unsupervised -> force pooled mean (0 std)
                                sample = sample.masked_fill(pin_t[None, :, None, :], 0.0)
                                xb[:, li, 0][cells_l] = sample[:, li][cells_l]
                                # promote the TARGET channel only: next year's
                                # features are genuinely unobserved, and flagging
                                # them would present the contract's padding zero
                                # as an observed value
                                ctx[:, li, 0][cells_l] = True
                        else:
                            # explicit dependence: one forward draws every cell-
                            # group's whole line vector jointly, so all lines of
                            # the diagonal are sampled together
                            log_pi, mu, scale = model.forward_joint(xb, ctx, lmb, premb, cut_b)
                            sample = net.joint_mdn_sample(log_pi, mu, scale, generator=gen)
                            sample = sample.permute(0, 3, 1, 2)  # (B, W, D, L) -> (B, L, W, D)
                            sample = sample.masked_fill(pin_t[None, :, None, :], 0.0)
                            xb[:, :, 0][cells] = sample[cells]
                            # target channel only, as in the "ar" branch
                            ctx[:, :, 0][cells] = True
                # denormalize the target channel back to incremental loss ratios
                ratios = (
                    xb[:, :, 0].cpu().numpy() * std0[None, :, None, :] + mean0[None, :, None, :]
                )
                # sum the simulated future increment ratios per (company, line, origin),
                # scale by premium (dollars), and add the latest observed cumulative
                contrib = (ratios * future.repeat(chunk, axis=0)).sum(axis=3)  # (B, L, W)
                contrib = contrib.reshape(n_c, chunk, n_l, n_w, order="C")
                ults = premium[:, None] * contrib + c["latest_cum"][:, None]
                pieces.append(np.moveaxis(ults, 1, 0))  # (chunk, n_c, n_l, n_w)
                done += chunk
        return np.concatenate(pieces, axis=0)  # (n_draws, n_c, n_l, n_w)
