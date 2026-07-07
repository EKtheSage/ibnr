"""Triangle transformer gallery entry. See card.md for the model card.

Torch is imported inside fit()/predict() only: the entry must register (and
`ibnr.gallery` must import) without the [nn] extra installed. The training
scheme — calendar-cutoff augmentation, trailing-diagonal validation, deep
ensembling, diagonal-by-diagonal autoregressive rollout — lives here in
literal source, per the gallery's eject pattern."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.nn.transformer.config import TransformerConfig
from ibnr.gallery.registry import register
from ibnr.kernels.contract import _as_date
from ibnr.kernels.nn_contract import nn_data
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

#: cap on (draw chunk x cohorts) per rollout forward pass
MAX_ROLLOUT_BATCH = 4096


def _splits(
    obs_mask: np.ndarray, cal_idx: np.ndarray, val_diagonals: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """(context_eligible, val_target, val_cutoff): the trailing
    ``val_diagonals`` observed calendar diagonals are validation targets and
    are excluded from every training context and every training target."""
    c_max = int(cal_idx[obs_mask.any(axis=0)].max())
    val_cutoff = c_max - val_diagonals
    if val_cutoff < 2:
        raise ValueError(
            f"latest observed diagonal is {c_max}; need at least {2 + val_diagonals} "
            "diagonals to hold one out for validation and still train"
        )
    context_eligible = obs_mask & (cal_idx <= val_cutoff)
    val_target = obs_mask & (cal_idx > val_cutoff)
    if not val_target.any():
        raise ValueError("no cells on the validation diagonal(s)")
    return context_eligible, val_target, val_cutoff


def _norm_stats(x: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-(channel, dev) mean/std over the given cells ONLY — validation
    diagonals must not leak through the normalizer.

    Devs with fewer than two cells (or zero spread) inherit the nearest
    EARLIER dev's stats, not the channel's global stats: the deepest dev may
    appear only on the held-out diagonal, and development is smooth in dev,
    so the previous lag's scale is the sane prior — global stats would
    denormalize a tail cell at first-diagonal magnitudes."""
    _, n_f, _, n_d = x.shape
    mean, std = np.zeros((n_f, n_d)), np.ones((n_f, n_d))
    for f in range(n_f):
        pooled = x[:, f][cells]
        g_mean = float(pooled.mean()) if pooled.size else 0.0
        g_std = float(pooled.std()) if pooled.size else 1.0
        if not np.isfinite(g_std) or g_std < 1e-8:
            g_std = 1.0
        prev: tuple[float, float] | None = None
        for d in range(n_d):
            vals = x[:, f, :, d][cells[:, :, d]]
            if vals.size >= 2 and float(vals.std()) > 1e-8:
                prev = (float(vals.mean()), float(vals.std()))
                mean[f, d], std[f, d] = prev
            else:
                mean[f, d], std[f, d] = prev if prev is not None else (g_mean, g_std)
    return mean, std


@register
class NNTransformer(GalleryEntry):
    name = "nn_transformer"
    family = "nn"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.config_: TransformerConfig | None = None
        self.models_: list | None = None  # one TriangleTransformer per ensemble member
        self.norm_: dict | None = None
        self.history_: list[list[dict]] | None = None
        self._loss_field: str | None = None
        self._device: str = "cpu"
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
        triangle. Never fit this on a single triangle — the whole point is
        cross-cohort pooling."""
        import torch

        from ibnr.gallery.nn.transformer import network as net

        cfg = config or TransformerConfig()
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = nn_data(
            train,
            loss_field=loss_field,
            feature_fields=feature_fields,
            premium_field=premium_field,
        )
        self._loss_field = loss_field
        self.config_ = cfg
        self._device = device or "cpu"
        c = self.contract_
        n_c, n_f, n_w, n_d = c["x"].shape

        context_elig, val_target, val_cutoff = _splits(
            c["obs_mask"], c["cal_idx"], cfg.val_diagonals
        )
        mean, std = _norm_stats(c["x"], context_elig)
        prem_mean = float(np.mean(c["log_premium"]))
        prem_std = float(np.std(c["log_premium"]))
        if prem_std < 1e-8:
            prem_std = 1.0
        self.norm_ = {"mean": mean, "std": std, "prem_mean": prem_mean, "prem_std": prem_std}

        x_norm = (c["x"] - mean[None, :, None, :]) / std[None, :, None, :]
        dev = torch.device(self._device)
        xt = torch.tensor(x_norm, dtype=torch.float32, device=dev)
        yt = xt[:, 0]  # target channel, normalized
        obs_t = torch.tensor(c["obs_mask"], device=dev)
        cal_t = torch.tensor(c["cal_idx"], device=dev)
        ctx_elig_t = torch.tensor(context_elig, device=dev)
        val_tgt_t = torch.tensor(val_target, device=dev)
        lob_t = torch.tensor(c["lob_idx"], dtype=torch.long, device=dev)
        prem_t = torch.tensor(
            (c["log_premium"] - prem_mean) / prem_std, dtype=torch.float32, device=dev
        )

        min_cutoff = max(1, min(cfg.min_cutoff, val_cutoff - 1))
        self.models_, self.history_ = [], []
        for member in range(cfg.ensemble_size):
            member_seed = None if seed is None else seed + 1000 * member
            if member_seed is not None:
                torch.manual_seed(member_seed)
            rng = np.random.default_rng(member_seed)
            model = net.TriangleTransformer(
                cfg, n_lob=len(c["lob_levels"]), n_features=n_f, n_w=n_w, n_d=n_d
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
                    ctx = obs_t[idx] & (cal_t[None] <= cutoffs[:, None, None])
                    tgt = ctx_elig_t[idx] & (cal_t[None] > cutoffs[:, None, None])
                    if not bool(tgt.any()):
                        continue
                    log_pi, mu, sigma = model(xt[idx], ctx, lob_t[idx], prem_t[idx])
                    loss = net.mdn_nll(log_pi, mu, sigma, yt[idx], tgt)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    epoch_loss += float(loss.detach())
                    n_batches += 1

                model.eval()
                with torch.no_grad():
                    log_pi, mu, sigma = model(xt, ctx_elig_t, lob_t, prem_t)
                    val_loss = float(net.mdn_nll(log_pi, mu, sigma, yt, val_tgt_t))
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
        """Predictive ultimates from the cached global rollout.

        With ``segment`` (e.g. {"company_code": ..., "line_of_business": ...}):
        per-origin ultimates for that one cohort plus their total — the same
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
        re-encode, continue. Draws are pooled over the ensemble members."""
        import torch

        from ibnr.gallery.nn.transformer import network as net

        c = self.contract_
        n_c, _, n_w, n_d = c["x"].shape
        mean0, std0 = self.norm_["mean"][0], self.norm_["std"][0]  # (n_d,)
        dev = torch.device(self._device)

        # future cells: strictly beyond each origin's anchor
        d_grid = np.arange(n_d)[None, None, :]
        future = d_grid >= c["latest_dev"][:, :, None]  # (n_c, n_w, n_d)
        cal_levels = sorted(np.unique(c["cal_idx"][future.any(axis=0)]))

        x_norm = (c["x"] - self.norm_["mean"][None, :, None, :]) / self.norm_["std"][
            None, :, None, :
        ]
        prem_norm = (c["log_premium"] - self.norm_["prem_mean"]) / self.norm_["prem_std"]
        xt = torch.tensor(x_norm, dtype=torch.float32, device=dev)
        obs_t = torch.tensor(c["obs_mask"], device=dev)
        fut_t = torch.tensor(future, device=dev)
        cal_t = torch.tensor(c["cal_idx"], device=dev)
        lob_t = torch.tensor(c["lob_idx"], dtype=torch.long, device=dev)
        prem_t = torch.tensor(prem_norm, dtype=torch.float32, device=dev)

        n_members = len(self.models_)
        member_draws = [n_draws // n_members] * n_members
        for i in range(n_draws % n_members):
            member_draws[i] += 1
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
                xb = xt.repeat_interleave(chunk, dim=0).clone()  # (chunk*n_c interleaved)
                ctx = obs_t.repeat_interleave(chunk, dim=0).clone()
                futb = fut_t.repeat_interleave(chunk, dim=0)
                lobb = lob_t.repeat_interleave(chunk, dim=0)
                premb = prem_t.repeat_interleave(chunk, dim=0)
                with torch.no_grad():
                    for lv in cal_levels:
                        cells = futb & (cal_t[None] == lv)
                        if not bool(cells.any()):
                            continue
                        log_pi, mu, sigma = model(xb, ctx, lobb, premb)
                        sample = net.mdn_sample(log_pi, mu, sigma, generator=gen)
                        xb[:, 0][cells] = sample[cells]
                        ctx = ctx | cells
                ratios = xb[:, 0].cpu().numpy() * std0[None, None, :] + mean0[None, None, :]
                contrib = (ratios * future.repeat(chunk, axis=0)).sum(axis=2)
                contrib = contrib.reshape(n_c, chunk, n_w, order="C")  # undo interleave
                ults = c["latest_cum"][:, None, :] + contrib * c["premium"][:, None, :]
                pieces.append(np.moveaxis(ults, 1, 0))  # (chunk, n_c, n_w)
                done += chunk
        return np.concatenate(pieces, axis=0)
