"""Multi-line triangle transformer gallery entry. See card.md.

The research entry: one encoder attends over every (line, origin, dev) cell
of a COMPANY, and cross-line dependence enters the draws either by
within-diagonal line-by-line autoregression ("ar") or an explicit joint
Gaussian-mixture head ("joint") — config.dependence. Torch is imported
inside fit()/predict() only. Training scheme (calendar-cutoff augmentation,
trailing-diagonal validation, pinned per-dev standardization, deep
ensembling, diagonal-by-diagonal rollout) mirrors nn_transformer; the
normalization and split helpers are imported from it — one implementation.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.nn.transformer.model import _norm_stats, _splits
from ibnr.gallery.nn.transformer_ml.config import TransformerMLConfig
from ibnr.gallery.registry import register
from ibnr.kernels.contract import _as_date
from ibnr.kernels.multiline import assemble_predictive, flatten_with_totals, multiline_targets
from ibnr.kernels.nn_contract import nn_company_data
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

#: cap on (draw chunk x companies) per rollout forward pass
MAX_ROLLOUT_BATCH = 2048


@register
class NNTransformerML(GalleryEntry):
    name = "nn_transformer_ml"
    family = "nn"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.config_: TransformerMLConfig | None = None
        self.models_: list | None = None
        self.norm_: dict | None = None
        self.history_: list[list[dict]] | None = None
        self._loss_field: str | None = None
        self._device: str = "cpu"
        self._rollout_key: tuple | None = None
        self._rollout_ults: np.ndarray | None = None  # (n_draws, n_c, n_l, n_w)

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "reported_loss",
        feature_fields: tuple[str, ...] = (),
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        config: TransformerMLConfig | None = None,
        device: str | None = None,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> NNTransformerML:
        """Pooled fit across every company in the triangle; each company's
        lines are one joint training example."""
        import torch

        from ibnr.gallery.nn.transformer import network as net1
        from ibnr.gallery.nn.transformer_ml import network as net

        cfg = config or TransformerMLConfig()
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = nn_company_data(
            train,
            loss_field=loss_field,
            feature_fields=feature_fields,
            premium_field=premium_field,
        )
        self._loss_field = loss_field
        self.config_ = cfg
        self._device = device or "cpu"
        c = self.contract_
        n_c, n_l, n_f, n_w, n_d = c["x"].shape

        obs_any = c["obs_mask"].any(axis=1)
        _, _, val_cutoff = _splits(obs_any, c["cal_idx"], cfg.val_diagonals)
        cal = c["cal_idx"]
        ctx_elig = c["obs_mask"] & (cal[None, None] <= val_cutoff)
        val_tgt = c["obs_mask"] & (cal[None, None] > val_cutoff)

        # pinned per-(line, channel, dev) standardization — one line at a time
        # through the single-line helper (identical rules by construction)
        mean = np.zeros((n_l, n_f, n_d))
        std = np.ones((n_l, n_f, n_d))
        pinned = np.zeros((n_l, n_f, n_d), dtype=bool)
        for li in range(n_l):
            mean[li], std[li], pinned[li] = _norm_stats(
                c["x"][:, li], ctx_elig[:, li], c["obs_mask"][:, li]
            )
        lp = c["log_premium"][c["line_mask"]]
        prem_mean, prem_std = float(lp.mean()), float(lp.std())
        if prem_std < 1e-8:
            prem_std = 1.0
        self.norm_ = {
            "mean": mean,
            "std": std,
            "pinned": pinned,
            "prem_mean": prem_mean,
            "prem_std": prem_std,
        }

        x_norm = (c["x"] - mean[None, :, :, None, :]) / std[None, :, :, None, :]
        x_norm = np.where(pinned[None, :, :, None, :], 0.0, x_norm)
        prem_norm = np.where(c["line_mask"], (c["log_premium"] - prem_mean) / prem_std, 0.0)

        dev = torch.device(self._device)
        xt = torch.tensor(x_norm, dtype=torch.float32, device=dev)
        yt = xt[:, :, 0]  # (n_c, n_l, n_w, n_d)
        obs_t = torch.tensor(c["obs_mask"], device=dev)
        cal_t = torch.tensor(cal, device=dev)
        ctx_elig_t = torch.tensor(ctx_elig, device=dev)
        val_tgt_t = torch.tensor(val_tgt, device=dev)
        lm_t = torch.tensor(c["line_mask"], device=dev)
        prem_t = torch.tensor(prem_norm, dtype=torch.float32, device=dev)
        val_cut_t = torch.full((n_c,), val_cutoff, dtype=torch.long, device=dev)

        def nll(model, xb, ctxb, lmb, premb, cutb, yb, tgtb):
            if cfg.dependence == "ar":
                log_pi, mu, sigma = model.forward_ar(xb, ctxb, lmb, premb, cutb)
                return net1.mdn_nll(log_pi, mu, sigma, yb, tgtb)
            log_pi, mu, scale = model.forward_joint(xb, ctxb, lmb, premb, cutb)
            return net.joint_mdn_nll(log_pi, mu, scale, yb, tgtb)

        min_cutoff = max(1, min(cfg.min_cutoff, val_cutoff - 1))
        self.models_, self.history_ = [], []
        for member in range(cfg.ensemble_size):
            member_seed = None if seed is None else seed + 1000 * member
            if member_seed is not None:
                torch.manual_seed(member_seed)
            rng = np.random.default_rng(member_seed)
            model = net.TriangleTransformerML(
                cfg, n_lines=n_l, n_features=n_f, n_w=n_w, n_d=n_d
            ).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

            best_val, best_state, patience_left = math.inf, None, cfg.patience
            history: list[dict] = []
            for epoch in range(cfg.max_epochs):
                model.train()
                epoch_loss, n_batches = 0.0, 0
                perm = rng.permutation(n_c)
                for start in range(0, n_c, cfg.batch_size):
                    idx = torch.tensor(perm[start : start + cfg.batch_size], device=dev)
                    cutoffs = torch.tensor(
                        rng.integers(min_cutoff, val_cutoff, size=len(idx)), device=dev
                    )
                    ctx = obs_t[idx] & (cal_t[None, None] <= cutoffs[:, None, None, None])
                    tgt = ctx_elig_t[idx] & (cal_t[None, None] > cutoffs[:, None, None, None])
                    if not bool(tgt.any()):
                        continue
                    loss = nll(model, xt[idx], ctx, lm_t[idx], prem_t[idx], cutoffs, yt[idx], tgt)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    epoch_loss += float(loss.detach())
                    n_batches += 1

                model.eval()
                with torch.no_grad():
                    val_loss = float(
                        nll(model, xt, ctx_elig_t, lm_t, prem_t, val_cut_t, yt, val_tgt_t)
                    )
                history.append(
                    {"epoch": epoch, "train": epoch_loss / max(n_batches, 1), "val": val_loss}
                )
                if show_progress:
                    print(f"member {member} epoch {epoch}: val {val_loss:.4f}")
                if val_loss < best_val - 1e-6:
                    best_val, patience_left = val_loss, cfg.patience
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models_.append(model)
            self.history_.append(history)

        self._rollout_key = None
        return self

    def predict(
        self,
        segment: dict[str, str] | None = None,
        n_draws: int | None = None,
        seed: int | None = None,
    ) -> PredictiveDistribution:
        """With ``segment`` identifying one company: the SUR layout —
        per-(lob, origin) ultimates for its present lines, per-lob totals,
        grand total — so cross-line diversification is visible in the draws.
        Without: every (company, line, origin) ultimate, no totals."""
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
            ci = self._company_index(segment)
            present = np.nonzero(c["line_mask"][ci])[0]
            lobs = [c["lob_levels"][li] for li in present]
            targets = multiline_targets(
                lobs, c["origin_periods"], premium=c["premium"][ci, present]
            )
            return assemble_predictive(ults[:, ci, present], targets)

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
        keep = c["line_mask"][:, :, None] & np.ones((1, 1, n_w), dtype=bool)
        samples = ults.reshape(n_draws, -1)[:, keep.reshape(-1)]
        return PredictiveDistribution(samples=samples, targets=pd.DataFrame(rows))

    def realized_ultimates(
        self, full_triangle: Triangle, segment: dict[str, str] | None = None
    ) -> np.ndarray:
        """Outcomes aligned to predict()'s targets, from the full triangle
        at the final dev lag (with lob/grand totals when a segment is given)."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        company_cols = list(c["companies"].columns)
        df = full_triangle.select_fields(self._loss_field).execute()
        df = df[df["dev_lag"] == c["n_d"] * c["dev_grain_months"]].copy()
        df["origin_period"] = _as_date(df["origin_period"])
        by_key = df.set_index([*company_cols, "line_of_business", "origin_period"])["value"]

        def lookup(company_row, lob) -> np.ndarray:
            vals = [float(by_key.get((*company_row, lob, o), np.nan)) for o in c["origin_periods"]]
            return np.asarray(vals)

        if segment is not None:
            ci = self._company_index(segment)
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

    # -- internals ---------------------------------------------------------------

    def _company_index(self, segment: dict[str, str]) -> int:
        companies = self.contract_["companies"]
        mask = np.ones(len(companies), dtype=bool)
        for col, value in segment.items():
            if col not in companies.columns:
                raise KeyError(f"unknown segment column {col!r}; have {list(companies.columns)}")
            mask &= (companies[col] == value).to_numpy()
        idx = np.nonzero(mask)[0]
        if len(idx) != 1:
            raise ValueError(f"segment {segment} matches {len(idx)} companies, need exactly 1")
        return int(idx[0])

    def _rollout(self, n_draws: int, seed: int | None) -> np.ndarray:
        """Diagonal-by-diagonal autoregressive rollout over every company.

        "ar": within a diagonal the lines are sampled one at a time in a
        seeded random order (fresh order per diagonal per chunk), each fed
        back before the next — cross-line dependence via conditioning.
        "joint": one forward per diagonal; every (origin, dev) cell-group's
        line vector is drawn jointly from the multivariate mixture.
        """
        import torch

        from ibnr.gallery.nn.transformer import network as net1
        from ibnr.gallery.nn.transformer_ml import network as net

        c = self.contract_
        cfg = self.config_
        n_c, n_l, _, n_w, n_d = c["x"].shape
        mean0, std0 = self.norm_["mean"][:, 0], self.norm_["std"][:, 0]  # (n_l, n_d)
        pin0 = self.norm_["pinned"][:, 0]  # (n_l, n_d)
        dev = torch.device(self._device)

        d_grid = np.arange(n_d)[None, None, None, :]
        future = (d_grid >= c["latest_dev"][:, :, :, None]) & c["line_mask"][:, :, None, None]
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
        obs_t = torch.tensor(c["obs_mask"], device=dev)
        fut_t = torch.tensor(future, device=dev)
        cal_t = torch.tensor(c["cal_idx"], device=dev)
        lm_t = torch.tensor(c["line_mask"], device=dev)
        prem_t = torch.tensor(prem_norm, dtype=torch.float32, device=dev)
        pin_t = torch.tensor(pin0, device=dev)  # (n_l, n_d)

        n_members = len(self.models_)
        member_draws = [n_draws // n_members] * n_members
        for i in range(n_draws % n_members):
            member_draws[i] += 1
        chunk_size = max(1, MAX_ROLLOUT_BATCH // n_c)
        premium = np.where(np.isnan(c["premium"]), 0.0, c["premium"])

        pieces: list[np.ndarray] = []
        for member, (model, m_draws) in enumerate(zip(self.models_, member_draws, strict=True)):
            if m_draws == 0:
                continue
            gen = torch.Generator(device=dev)
            gen.manual_seed((0 if seed is None else seed) * 100003 + member)
            order_rng = np.random.default_rng((0 if seed is None else seed) * 7919 + member)
            done = 0
            while done < m_draws:
                chunk = min(chunk_size, m_draws - done)
                xb = xt.repeat_interleave(chunk, dim=0).clone()
                ctx = obs_t.repeat_interleave(chunk, dim=0).clone()
                futb = fut_t.repeat_interleave(chunk, dim=0)
                lmb = lm_t.repeat_interleave(chunk, dim=0)
                premb = prem_t.repeat_interleave(chunk, dim=0)
                with torch.no_grad():
                    for lv in cal_levels:
                        cells = futb & (cal_t[None, None] == lv)
                        if not bool(cells.any()):
                            continue
                        cut_b = torch.full(
                            (xb.shape[0],), int(lv) - 1, dtype=torch.long, device=dev
                        )
                        if cfg.dependence == "ar":
                            for li in order_rng.permutation(n_l):
                                li = int(li)
                                cells_l = cells[:, li]
                                if not bool(cells_l.any()):
                                    continue
                                log_pi, mu, sigma = model.forward_ar(xb, ctx, lmb, premb, cut_b)
                                sample = net1.mdn_sample(log_pi, mu, sigma, generator=gen)
                                sample = sample.masked_fill(pin_t[None, :, None, :], 0.0)
                                xb[:, li, 0][cells_l] = sample[:, li][cells_l]
                                ctx[:, li] |= cells_l
                        else:
                            log_pi, mu, scale = model.forward_joint(xb, ctx, lmb, premb, cut_b)
                            sample = net.joint_mdn_sample(log_pi, mu, scale, generator=gen)
                            sample = sample.permute(0, 3, 1, 2)  # (B, L, W, D)
                            sample = sample.masked_fill(pin_t[None, :, None, :], 0.0)
                            xb[:, :, 0][cells] = sample[cells]
                            ctx = ctx | cells
                ratios = (
                    xb[:, :, 0].cpu().numpy() * std0[None, :, None, :] + mean0[None, :, None, :]
                )
                contrib = (ratios * future.repeat(chunk, axis=0)).sum(axis=3)  # (B, L, W)
                contrib = contrib.reshape(n_c, chunk, n_l, n_w, order="C")
                ults = premium[:, None] * contrib + c["latest_cum"][:, None]
                pieces.append(np.moveaxis(ults, 1, 0))  # (chunk, n_c, n_l, n_w)
                done += chunk
        return np.concatenate(pieces, axis=0)
