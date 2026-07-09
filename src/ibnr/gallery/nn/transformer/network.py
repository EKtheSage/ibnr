"""The triangle transformer network: masked-cell encoder with a mixture
density head. This module imports torch — only import it from inside the
entry's fit/predict paths."""

from __future__ import annotations

import math

import torch
from torch import nn

from ibnr.gallery.nn.transformer.config import TransformerConfig

LOG_2PI = math.log(2.0 * math.pi)
#: raw_p init so softplus(raw_p) == 1.0 -> exposure factor is 1 everywhere,
#: i.e. the exposure-aware head starts identical to the flat-sigma baseline.
RAW_P_INIT = math.log(math.expm1(1.0))


class TriangleTransformer(nn.Module):
    """Encoder over the full origin x dev grid of one cohort.

    Every cell is a token: [channel values * context flag, context flag]
    projected to d_model, plus learned origin / dev embeddings, a *relative*
    calendar embedding (distance past the conditioning cutoff, clamped to
    [0, n_d]) and broadcast cohort conditioning (LOB embedding + normalized
    log premium). The MDN head reads a distribution over the *normalized
    incremental loss ratio* of every cell; the loss is evaluated only where
    the target mask says so.

    Why relative, not absolute, calendar position: forecasting happens on
    calendar diagonals beyond the training window, where an absolute learned
    embedding never received a gradient. Distance-past-cutoff is supervised
    directly by the cutoff augmentation, and the autoregressive rollout
    re-encodes after every sampled diagonal, so inference only ever consumes
    distance-1 predictions — the most supervised case.
    """

    def __init__(
        self, cfg: TransformerConfig, *, n_lob: int, n_features: int, n_w: int, n_d: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_w, self.n_d = n_w, n_d
        d = cfg.d_model
        self.value_proj = nn.Linear(n_features + 1, d)
        self.origin_emb = nn.Embedding(n_w, d)
        self.dev_emb = nn.Embedding(n_d, d)
        self.dist_emb = nn.Embedding(n_d + 1, d)  # distance past cutoff, clamped
        self.lob_emb = nn.Embedding(n_lob, cfg.lob_embedding_dim)
        self.cond_proj = nn.Linear(cfg.lob_embedding_dim + 1, d)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3 * cfg.n_components)
        if cfg.exposure_sigma:
            # p = softplus(raw_p); a single trained log-dollar exposure power.
            # prem_log_std is filled from the training premiums by the entry
            # after construction, so the factor is expressed in true log-dollar
            # units rather than normalized-premium units.
            self.raw_p = nn.Parameter(torch.tensor(RAW_P_INIT))
            self.register_buffer("prem_log_std", torch.ones(()), persistent=True)

        w_idx, d_idx = torch.meshgrid(torch.arange(n_w), torch.arange(n_d), indexing="ij")
        self.register_buffer("w_idx", w_idx.reshape(-1), persistent=False)
        self.register_buffer("d_idx", d_idx.reshape(-1), persistent=False)
        # 1-based calendar diagonal, matching nn_contract's cal_idx convention
        self.register_buffer("cal_idx", (w_idx + d_idx + 1).reshape(-1), persistent=False)

    def forward(
        self,
        x: torch.Tensor,  # (B, F, W, D) normalized values (junk allowed off-context)
        context_mask: torch.Tensor,  # (B, W, D) bool
        lob_idx: torch.Tensor,  # (B,) long
        log_premium: torch.Tensor,  # (B,) normalized
        cutoff: torch.Tensor,  # (B,) long — 1-based conditioning diagonal
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns MDN parameters over every cell:
        log_pi, mu, sigma each (B, W, D, K)."""
        b = x.shape[0]
        flag = context_mask.unsqueeze(1).to(x.dtype)  # (B, 1, W, D)
        vals = (x * flag).flatten(2).transpose(1, 2)  # (B, T, F)
        tok = self.value_proj(torch.cat([vals, flag.flatten(2).transpose(1, 2)], dim=-1))
        dist = (self.cal_idx[None, :] - cutoff[:, None]).clamp(0, self.n_d)
        tok = tok + self.origin_emb(self.w_idx) + self.dev_emb(self.d_idx) + self.dist_emb(dist)
        cond = self.cond_proj(torch.cat([self.lob_emb(lob_idx), log_premium.unsqueeze(-1)], -1))
        tok = tok + cond.unsqueeze(1)
        h = self.encoder(self.drop(tok))
        out = self.head(self.out_norm(h)).reshape(b, self.n_w, self.n_d, 3, self.cfg.n_components)
        log_pi = out[..., 0, :].log_softmax(dim=-1)
        mu = out[..., 1, :]
        sigma = nn.functional.softplus(out[..., 2, :]) + 1e-3
        if self.cfg.exposure_sigma:
            # log_premium arrives normalized ((log_prem - mean) / std); rescale
            # to the centered log premium so the scale carries a factor of
            # premium**(p - 1) about the pooled mean premium. Combined with the
            # premium**1 the ratio normalization already contributes, the
            # predictive dollar sd scales as premium**p. p = 1 -> factor == 1.
            p = nn.functional.softplus(self.raw_p)
            centered_log_prem = log_premium * self.prem_log_std  # (B,)
            factor = torch.exp((p - 1.0) * centered_log_prem)  # (B,)
            sigma = sigma * factor[:, None, None, None]
        return log_pi, mu, sigma


def mdn_nll(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,  # (B, W, D)
    mask: torch.Tensor,  # (B, W, D) bool — cells that count
) -> torch.Tensor:
    """Mean negative log likelihood of y over masked cells."""
    y_ = y.unsqueeze(-1)
    comp = -0.5 * ((y_ - mu) / sigma) ** 2 - sigma.log() - 0.5 * LOG_2PI
    ll = torch.logsumexp(log_pi + comp, dim=-1)
    n = mask.sum().clamp(min=1)
    return -(ll * mask).sum() / n


def mdn_sample(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One draw of every cell's mixture: (..., K) params -> (...) values."""
    k = mu.shape[-1]
    u = torch.rand(*mu.shape[:-1], 1, generator=generator, device=mu.device)
    cum = log_pi.exp().cumsum(dim=-1)
    comp = (u > cum).sum(dim=-1, keepdim=True).clamp(max=k - 1)
    mu_s = mu.gather(-1, comp).squeeze(-1)
    sigma_s = sigma.gather(-1, comp).squeeze(-1)
    z = torch.randn(mu_s.shape, generator=generator, device=mu.device)
    return mu_s + sigma_s * z
