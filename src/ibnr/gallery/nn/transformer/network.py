"""The triangle transformer network: masked-cell encoder with a mixture
density head. This module imports torch — only import it from inside the
entry's fit/predict paths (``ibnr.gallery`` must import without the [nn]
extra; see model.py and CLAUDE.md's "torch must never be imported at module
level" rule).

Three pieces, all consumed by ``model.py``:
- ``TriangleTransformer`` — the encoder + MDN head (the ``nn.Module``);
- ``mdn_nll`` — the training loss (masked mixture negative log likelihood);
- ``mdn_sample`` — one draw per cell, used by the autoregressive rollout.

This is the SINGLE-LINE transformer: each cohort (company x LOB) is encoded
on its own, so its per-line predictive draws are independent. Cross-line
dependence is a separate entry (``nn_transformer_ml``); nothing here models
it. Model writeup: card.md. Data contract feeding it: kernels/nn_contract.py."""

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
        # n_w = number of origin periods (accident years), n_d = number of dev
        # lags; the grid has n_w * n_d cells and each cell is one token.
        self.n_w, self.n_d = n_w, n_d
        d = cfg.d_model
        # value_proj: per-cell [channel values, context flag] -> d_model. The
        # +1 input dim is the context flag itself, fed as a feature so the token
        # encodes "this cell is/ isn't conditioning data", not just its value.
        self.value_proj = nn.Linear(n_features + 1, d)
        # learned positional structure: an origin (row) and a dev-lag (column)
        # embedding placed additively on every token.
        self.origin_emb = nn.Embedding(n_w, d)
        self.dev_emb = nn.Embedding(n_d, d)
        # RELATIVE calendar position: distance (in diagonals) past the
        # conditioning cutoff, clamped to [0, n_d] (hence n_d + 1 rows). Chosen
        # over an absolute calendar embedding because forecasts land on
        # diagonals past the training window that an absolute embedding never
        # trained on; see the class docstring and card.md "Why relative".
        self.dist_emb = nn.Embedding(n_d + 1, d)  # distance past cutoff, clamped
        # cohort conditioning: LOB identity + size (normalized log premium),
        # projected to d and broadcast onto every token. No company embedding
        # on purpose (~600 companies x ~55 cells would just memorize; card.md).
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
        # MDN head: per cell, 3 params (mixture logit, mu, raw sigma) x K
        # components. Reshaped to (..., 3, K) in forward().
        self.head = nn.Linear(d, 3 * cfg.n_components)
        if cfg.exposure_sigma:
            # p = softplus(raw_p); a single trained log-dollar exposure power.
            # prem_log_std is filled from the training premiums by the entry
            # after construction, so the factor is expressed in true log-dollar
            # units rather than normalized-premium units.
            self.raw_p = nn.Parameter(torch.tensor(RAW_P_INIT))
            self.register_buffer("prem_log_std", torch.ones(()), persistent=True)

        # Precompute per-token (origin, dev, calendar-diagonal) indices for the
        # flattened T = n_w * n_d token sequence, in the same row-major
        # (origin-major) order forward() flattens x with. Buffers, not
        # parameters: fixed lookups that must follow .to(device) but never
        # train. non-persistent -> not written to checkpoints (pure functions
        # of n_w, n_d).
        w_idx, d_idx = torch.meshgrid(torch.arange(n_w), torch.arange(n_d), indexing="ij")
        self.register_buffer("w_idx", w_idx.reshape(-1), persistent=False)  # (T,)
        self.register_buffer("d_idx", d_idx.reshape(-1), persistent=False)  # (T,)
        # 1-based calendar diagonal, matching nn_contract's cal_idx convention
        self.register_buffer("cal_idx", (w_idx + d_idx + 1).reshape(-1), persistent=False)  # (T,)

    def forward(
        self,
        x: torch.Tensor,  # (B, F, W, D) normalized values (junk allowed off-context)
        context_mask: torch.Tensor,  # (B, W, D) bool
        lob_idx: torch.Tensor,  # (B,) long
        log_premium: torch.Tensor,  # (B,) normalized
        cutoff: torch.Tensor,  # (B,) long — 1-based conditioning diagonal
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one batch of cohort grids and read an MDN over every cell.

        B = cohorts in the batch, F = channels, W = origins, D = dev lags,
        T = W*D tokens, K = mixture components. Returns log_pi, mu, sigma each
        (B, W, D, K) — a K-Gaussian mixture over the normalized incremental
        loss ratio of every cell. Only ``context_mask``-true cells contribute
        their values; the rest are zeroed, so their tokens carry position +
        conditioning only (the model must predict them, not read them)."""
        b = x.shape[0]
        flag = context_mask.unsqueeze(1).to(x.dtype)  # (B, 1, W, D)
        # zero out non-context values, then flatten grid -> token sequence:
        # (B, F, W, D) -> (B, F, T) -> (B, T, F). Row-major over (W, D) so token
        # order matches the w_idx/d_idx/cal_idx buffers built in __init__.
        vals = (x * flag).flatten(2).transpose(1, 2)  # (B, T, F)
        # append the context flag as an extra input feature -> (B, T, F+1),
        # projected to (B, T, d_model).
        tok = self.value_proj(torch.cat([vals, flag.flatten(2).transpose(1, 2)], dim=-1))
        # relative calendar position of each token: diagonals past the cutoff,
        # clamped to [0, n_d]. cutoff is the last conditioning diagonal.
        dist = (self.cal_idx[None, :] - cutoff[:, None]).clamp(0, self.n_d)  # (B, T)
        # additive positional signal: origin + dev + distance-past-cutoff, each
        # (T, d) / (B, T, d), broadcast onto the (B, T, d) tokens.
        tok = tok + self.origin_emb(self.w_idx) + self.dev_emb(self.d_idx) + self.dist_emb(dist)
        # per-cohort conditioning (LOB + size) -> (B, d), broadcast to all tokens
        cond = self.cond_proj(torch.cat([self.lob_emb(lob_idx), log_premium.unsqueeze(-1)], -1))
        tok = tok + cond.unsqueeze(1)  # (B, T, d)
        h = self.encoder(self.drop(tok))  # (B, T, d) — full self-attention over the grid
        # head -> (B, T, 3K), reshaped to per-cell (B, W, D, 3, K): the 3 slot
        # splits into mixture logits / means / raw scales.
        out = self.head(self.out_norm(h)).reshape(b, self.n_w, self.n_d, 3, self.cfg.n_components)
        log_pi = out[..., 0, :].log_softmax(dim=-1)  # (B, W, D, K) normalized log weights
        mu = out[..., 1, :]  # (B, W, D, K) component means (normalized ratio scale)
        # softplus keeps sigma > 0; the 1e-3 floor prevents a collapsing
        # component from driving the NLL to -inf.
        sigma = nn.functional.softplus(out[..., 2, :]) + 1e-3  # (B, W, D, K)
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
    """Mean negative log likelihood of the mixture over masked cells.

    Training objective. ``mask`` is the set of scored cells — under cutoff
    augmentation, the observed training cells strictly past the drawn cutoff
    (the "predict the next diagonals" task). Per cell the log density is the
    log-sum-exp over K components of log_pi + log N(y; mu, sigma); the loss is
    the negative mean over masked cells only."""
    y_ = y.unsqueeze(-1)  # (B, W, D, 1) to broadcast against the K component axis
    # per-component Gaussian log density (the -0.5*log(2pi) is the constant)
    comp = -0.5 * ((y_ - mu) / sigma) ** 2 - sigma.log() - 0.5 * LOG_2PI  # (B, W, D, K)
    ll = torch.logsumexp(log_pi + comp, dim=-1)  # (B, W, D) log mixture density
    n = mask.sum().clamp(min=1)  # clamp: never divide by zero if a batch has no targets
    return -(ll * mask).sum() / n


def mdn_sample(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One draw of every cell's mixture: (..., K) params -> (...) values.

    Ancestral sampling per cell: pick a component by its weight, then draw from
    that Gaussian. The rollout calls this once per calendar diagonal per draw;
    ``generator`` makes the draw reproducible/seed-controlled."""
    k = mu.shape[-1]
    # inverse-CDF component pick: u in [0,1); count how many cumulative weights
    # it exceeds -> the chosen component index (clamped to the last for u==1).
    u = torch.rand(*mu.shape[:-1], 1, generator=generator, device=mu.device)  # (..., 1)
    cum = log_pi.exp().cumsum(dim=-1)  # (..., K) cumulative mixture weights
    comp = (u > cum).sum(dim=-1, keepdim=True).clamp(max=k - 1)  # (..., 1) component index
    mu_s = mu.gather(-1, comp).squeeze(-1)  # (...) selected means
    sigma_s = sigma.gather(-1, comp).squeeze(-1)  # (...) selected scales
    z = torch.randn(mu_s.shape, generator=generator, device=mu.device)
    return mu_s + sigma_s * z  # (...) one normalized-ratio draw per cell
