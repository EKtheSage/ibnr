"""Multi-line triangle transformer: one encoder over every (line, origin,
dev) cell of a COMPANY, so attention spans a company's lines — the learned
analogue of SUR's contemporaneous correlation. Imports torch — only import
this module from inside the entry's fit/predict paths.

Two dependence heads (config.dependence):
- "ar":    univariate MDN per token (as the single-line model); cross-line
           dependence enters through within-diagonal line-by-line
           autoregressive sampling in the entry's rollout.
- "joint": per-(origin, dev) multivariate Gaussian mixture across lines,
           Cholesky-parameterized; absent lines are marginalized out (a
           marginal of a Gaussian mixture is the mixture of marginals).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ibnr.gallery.nn.transformer_ml.config import TransformerMLConfig

LOG_2PI = math.log(2.0 * math.pi)


class TriangleTransformerML(nn.Module):
    """Encoder over the full (line, origin, dev) grid of one company.

    Token = [channel values * context flag, context flag] projected to
    d_model, plus line / origin / dev embeddings, a relative calendar
    embedding (distance past the conditioning cutoff — supervised by the
    cutoff augmentation, unlike absolute positions), and per-line premium
    conditioning. Absent lines are excluded from attention via the padding
    mask and from every loss via the masks the entry passes in.
    """

    def __init__(
        self,
        cfg: TransformerMLConfig,
        *,
        n_lines: int,
        n_features: int,
        n_w: int,
        n_d: int,
    ) -> None:
        super().__init__()
        if cfg.dependence not in ("ar", "joint"):
            raise ValueError(f"dependence must be 'ar' or 'joint', got {cfg.dependence!r}")
        self.cfg = cfg
        self.n_l, self.n_w, self.n_d = n_lines, n_w, n_d
        d = cfg.d_model
        self.value_proj = nn.Linear(n_features + 1, d)
        self.line_emb = nn.Embedding(n_lines, d)
        self.origin_emb = nn.Embedding(n_w, d)
        self.dev_emb = nn.Embedding(n_d, d)
        self.dist_emb = nn.Embedding(n_d + 1, d)  # distance past cutoff, clamped
        self.prem_proj = nn.Linear(1, d)  # per-line normalized log premium
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
        k = cfg.n_components
        if cfg.dependence == "ar":
            self.head = nn.Linear(d, 3 * k)
        else:
            # per-(w, d) joint head over the line vector: K logits,
            # K*L means, K*L*(L+1)/2 scale_tril entries
            self.n_tril = n_lines * (n_lines + 1) // 2
            self.joint_head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, k * (1 + n_lines + self.n_tril)),
            )
            tril = torch.tril_indices(n_lines, n_lines)
            self.register_buffer("tril_r", tril[0], persistent=False)
            self.register_buffer("tril_c", tril[1], persistent=False)

        l_idx, w_idx, d_idx = torch.meshgrid(
            torch.arange(n_lines), torch.arange(n_w), torch.arange(n_d), indexing="ij"
        )
        self.register_buffer("l_idx", l_idx.reshape(-1), persistent=False)
        self.register_buffer("w_idx", w_idx.reshape(-1), persistent=False)
        self.register_buffer("d_idx", d_idx.reshape(-1), persistent=False)
        self.register_buffer("cal_idx", (w_idx + d_idx + 1).reshape(-1), persistent=False)

    def encode(
        self,
        x: torch.Tensor,  # (B, L, F, W, D) normalized values
        context_mask: torch.Tensor,  # (B, L, W, D) bool
        line_mask: torch.Tensor,  # (B, L) bool
        log_premium: torch.Tensor,  # (B, L) normalized (0 where absent)
        cutoff: torch.Tensor,  # (B,) long, 1-based conditioning diagonal
    ) -> torch.Tensor:
        """Token states h: (B, L, W, D, d_model)."""
        b = x.shape[0]
        flag = context_mask.unsqueeze(2).to(x.dtype)  # (B, L, 1, W, D)
        vals = (x * flag).permute(0, 1, 3, 4, 2).reshape(b, -1, x.shape[2])  # (B, T, F)
        flags = flag.permute(0, 1, 3, 4, 2).reshape(b, -1, 1)
        tok = self.value_proj(torch.cat([vals, flags], dim=-1))
        dist = (self.cal_idx[None, :] - cutoff[:, None]).clamp(0, self.n_d)
        prem_tok = self.prem_proj(log_premium.unsqueeze(-1))  # (B, L, d)
        tok = (
            tok
            + self.line_emb(self.l_idx)
            + self.origin_emb(self.w_idx)
            + self.dev_emb(self.d_idx)
            + self.dist_emb(dist)
            + prem_tok[:, self.l_idx]
        )
        pad = ~line_mask[:, self.l_idx]  # (B, T): absent lines leave attention
        h = self.encoder(self.drop(tok), src_key_padding_mask=pad)
        h = self.out_norm(h)
        return h.reshape(b, self.n_l, self.n_w, self.n_d, -1)

    def forward_ar(self, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Univariate MDN params per cell: log_pi, mu, sigma (B, L, W, D, K)."""
        h = self.encode(*args)
        out = self.head(h).reshape(*h.shape[:-1], 3, self.cfg.n_components)
        log_pi = out[..., 0, :].log_softmax(dim=-1)
        mu = out[..., 1, :]
        sigma = nn.functional.softplus(out[..., 2, :]) + 1e-3
        return log_pi, mu, sigma

    def forward_joint(self, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint MVN-mixture params per (w, d) cell-group:
        log_pi (B, W, D, K), mu (B, W, D, K, L), scale_tril (B, W, D, K, L, L).

        The group state is the mean of the present lines' token states —
        absent lines' tokens never attend, but masking them here keeps the
        head input independent of the padding values."""
        x, context_mask, line_mask, log_premium, cutoff = args
        h = self.encode(x, context_mask, line_mask, log_premium, cutoff)
        m = line_mask[:, :, None, None, None].to(h.dtype)
        group = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B, W, D, d)
        k, n_l = self.cfg.n_components, self.n_l
        out = self.joint_head(group).reshape(*group.shape[:-1], k, 1 + n_l + self.n_tril)
        log_pi = out[..., 0].log_softmax(dim=-1)
        mu = out[..., 1 : 1 + n_l]
        tril_vals = out[..., 1 + n_l :]
        scale = torch.zeros(*mu.shape, n_l, device=mu.device, dtype=mu.dtype)
        scale[..., self.tril_r, self.tril_c] = tril_vals
        diag = torch.arange(n_l, device=mu.device)
        scale[..., diag, diag] = nn.functional.softplus(scale[..., diag, diag]) + 1e-3
        return log_pi, mu, scale


def joint_mdn_nll(
    log_pi: torch.Tensor,  # (B, W, D, K)
    mu: torch.Tensor,  # (B, W, D, K, L)
    scale_tril: torch.Tensor,  # (B, W, D, K, L, L)
    y: torch.Tensor,  # (B, L, W, D)
    target_mask: torch.Tensor,  # (B, L, W, D) bool — cells that count
) -> torch.Tensor:
    """Mean NLL of the line vector over target cell-groups, marginalizing
    lines that are not targets in a group (absent lines, ragged masks).

    Marginalization uses the covariance submatrix per line-pattern; groups
    are batched by pattern (few distinct patterns in practice).
    """
    b, n_l, n_w, n_d = y.shape
    group_mask = target_mask.permute(0, 2, 3, 1)  # (B, W, D, L)
    flat_mask = group_mask.reshape(-1, n_l)
    active = flat_mask.any(dim=-1)
    if not bool(active.any()):
        return torch.zeros((), device=y.device)

    k = log_pi.shape[-1]
    y_flat = y.permute(0, 2, 3, 1).reshape(-1, n_l)[active]
    pi_flat = log_pi.reshape(-1, k)[active]
    mu_flat = mu.reshape(-1, k, n_l)[active]
    scale_flat = scale_tril.reshape(-1, k, n_l, n_l)[active]
    pattern = flat_mask[active]

    total, n_groups = y_flat.new_zeros(()), 0
    for pat in pattern.unique(dim=0):
        rows = (pattern == pat[None]).all(dim=-1)
        idx = pat.nonzero(as_tuple=True)[0]
        yp = y_flat[rows][:, idx]  # (G, P)
        mup = mu_flat[rows][:, :, idx]  # (G, K, P)
        lp = scale_flat[rows]  # (G, K, L, L)
        cov = lp @ lp.transpose(-1, -2)
        sub = cov[:, :, idx][:, :, :, idx]  # (G, K, P, P)
        chol = torch.linalg.cholesky(sub)
        diff = (yp[:, None, :] - mup).unsqueeze(-1)  # (G, K, P, 1)
        z = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
        logdet = chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        comp = -0.5 * (z**2).sum(-1) - logdet - 0.5 * len(idx) * LOG_2PI  # (G, K)
        ll = torch.logsumexp(pi_flat[rows] + comp, dim=-1)
        total = total - ll.sum()
        n_groups += int(rows.sum())
    return total / max(n_groups, 1)


def joint_mdn_sample(
    log_pi: torch.Tensor,  # (B, W, D, K)
    mu: torch.Tensor,  # (B, W, D, K, L)
    scale_tril: torch.Tensor,  # (B, W, D, K, L, L)
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One joint draw of every cell-group's line vector: (B, W, D, L).
    Callers select the present lines / target cells from the result."""
    k, n_l = mu.shape[-2], mu.shape[-1]
    u = torch.rand(*log_pi.shape[:-1], 1, generator=generator, device=mu.device)
    cum = log_pi.exp().cumsum(dim=-1)
    comp = (u > cum).sum(dim=-1).clamp(max=k - 1)  # (B, W, D)
    gather = comp[..., None, None, None].expand(*comp.shape, 1, n_l, n_l)
    scale_s = scale_tril.gather(-3, gather).squeeze(-3)  # (B, W, D, L, L)
    mu_s = mu.gather(-2, comp[..., None, None].expand(*comp.shape, 1, n_l)).squeeze(-2)
    z = torch.randn(*mu_s.shape, 1, generator=generator, device=mu.device)
    return mu_s + (scale_s @ z).squeeze(-1)
