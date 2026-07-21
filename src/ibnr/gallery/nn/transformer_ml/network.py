"""Multi-line triangle transformer network. See card.md and model.py.

One encoder over every (line, origin, dev) cell of a COMPANY, so attention
spans a company's lines of business — the learned analogue of SUR's
contemporaneous correlation (Zhang 2010) and the copula's cell dependence
(Shi & Frees 2011), the two explicit baselines this entry is compared against.
Contrast the single-line ``nn.transformer.network``: that encoder sees one
line at a time, so its per-line draws are independent; here every line is a
token in the same sequence and dependence can be learned. Imports torch —
only import this module from inside the entry's fit/predict paths.

Two dependence heads (config.dependence) — the research question is which
mechanism carries cross-line dependence better:
- "ar":    univariate MDN per token (identical head to the single-line model);
           cross-line dependence enters only through within-diagonal
           line-by-line autoregressive sampling in the entry's rollout, i.e.
           via conditioning, not via the head itself.
- "joint": per-(origin, dev) multivariate Gaussian mixture over the LINE
           vector, Cholesky-parameterized so any covariance is representable.
           At training/eval time absent or non-target lines are marginalized
           out — a marginal of a Gaussian mixture is the mixture of the
           components' marginals, i.e. just drop the missing rows/cols of each
           component's mean and covariance.
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
        # value_proj takes [F normalized channel values * context flag, flag] -> d
        self.value_proj = nn.Linear(n_features + 1, d)
        # additive positional/identity embeddings, one per token (all shape d):
        self.line_emb = nn.Embedding(n_lines, d)  # which line of business
        self.origin_emb = nn.Embedding(n_w, d)  # accident/origin period
        self.dev_emb = nn.Embedding(n_d, d)  # development lag
        # relative calendar position: distance past the conditioning cutoff,
        # clamped to [0, n_d]. Relative (not absolute) so it is supervised at
        # the same distances the rollout later queries — see single-line card.
        self.dist_emb = nn.Embedding(n_d + 1, d)
        self.prem_proj = nn.Linear(1, d)  # per-line normalized log premium -> d
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
            # one univariate K-component MDN per token: pi, mu, sigma each K
            self.head = nn.Linear(d, 3 * k)
        else:
            # per-(w, d) joint head over the whole line vector: per component
            # K logits, K*L means, and K * L(L+1)/2 lower-triangular scale
            # entries (the Cholesky factor of the covariance)
            self.n_tril = n_lines * (n_lines + 1) // 2
            self.joint_head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, k * (1 + n_lines + self.n_tril)),
            )
            # (row, col) index of each lower-triangular entry, to scatter the
            # flat scale_tril outputs back into an (L, L) matrix
            tril = torch.tril_indices(n_lines, n_lines)
            self.register_buffer("tril_r", tril[0], persistent=False)
            self.register_buffer("tril_c", tril[1], persistent=False)

        # flat (line, origin, dev) index of every token position, in the token
        # order used by encode(): line-major, then origin, then dev. Length
        # T = L*W*D. Used to gather the right embedding per token.
        l_idx, w_idx, d_idx = torch.meshgrid(
            torch.arange(n_lines), torch.arange(n_w), torch.arange(n_d), indexing="ij"
        )
        self.register_buffer("l_idx", l_idx.reshape(-1), persistent=False)  # (T,)
        self.register_buffer("w_idx", w_idx.reshape(-1), persistent=False)  # (T,)
        self.register_buffer("d_idx", d_idx.reshape(-1), persistent=False)  # (T,)
        # 1-based calendar diagonal per token (w + d + 1), matching nn_contract
        self.register_buffer("cal_idx", (w_idx + d_idx + 1).reshape(-1), persistent=False)  # (T,)

    def encode(
        self,
        x: torch.Tensor,  # (B, L, F, W, D) normalized values
        context_mask: torch.Tensor,  # (B, L, W, D) bool
        line_mask: torch.Tensor,  # (B, L) bool
        log_premium: torch.Tensor,  # (B, L) normalized (0 where absent)
        cutoff: torch.Tensor,  # (B,) long, 1-based conditioning diagonal
    ) -> torch.Tensor:
        """Run the encoder; return token states h: (B, L, W, D, d_model).

        The (L, W, D) grid is flattened to a length-T sequence (T = L*W*D),
        one token per cell. Off-context cells are zeroed and flagged so the
        model can tell a masked-out cell from a genuine zero increment.
        """
        b = x.shape[0]
        # zero the channel values wherever the cell is not in context, and
        # carry the flag itself as an extra channel
        flag = context_mask.unsqueeze(2).to(x.dtype)  # (B, L, 1, W, D)
        vals = (x * flag).permute(0, 1, 3, 4, 2).reshape(b, -1, x.shape[2])  # (B, T, F)
        flags = flag.permute(0, 1, 3, 4, 2).reshape(b, -1, 1)  # (B, T, 1)
        tok = self.value_proj(torch.cat([vals, flags], dim=-1))  # (B, T, d)
        # distance of each token's diagonal past the conditioning cutoff
        dist = (self.cal_idx[None, :] - cutoff[:, None]).clamp(0, self.n_d)  # (B, T)
        prem_tok = self.prem_proj(log_premium.unsqueeze(-1))  # (B, L, d)
        # add per-token identity/position embeddings; prem_tok[:, l_idx]
        # broadcasts each line's premium token to all its (w, d) cells
        tok = (
            tok
            + self.line_emb(self.l_idx)  # (T, d) -> broadcast over B
            + self.origin_emb(self.w_idx)
            + self.dev_emb(self.d_idx)
            + self.dist_emb(dist)  # (B, T, d)
            + prem_tok[:, self.l_idx]  # (B, T, d)
        )
        # padding mask keeps tokens of lines the company does not write out of
        # attention entirely (they must not leak into present lines' states)
        pad = ~line_mask[:, self.l_idx]  # (B, T): True where the line is absent
        h = self.encoder(self.drop(tok), src_key_padding_mask=pad)  # (B, T, d)
        h = self.out_norm(h)
        return h.reshape(b, self.n_l, self.n_w, self.n_d, -1)  # (B, L, W, D, d)

    def forward_ar(self, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Univariate MDN params per cell: log_pi, mu, sigma each (B, L, W, D, K).

        The "ar" head: every cell gets its own independent 1-D mixture, exactly
        as in the single-line model. Cross-line dependence is NOT in these
        params — it is injected later by the rollout's line-by-line sampling.
        """
        h = self.encode(*args)  # (B, L, W, D, d)
        # last dim d -> 3*K, split into (logits, means, log-scales) x K
        out = self.head(h).reshape(*h.shape[:-1], 3, self.cfg.n_components)
        log_pi = out[..., 0, :].log_softmax(dim=-1)
        mu = out[..., 1, :]
        sigma = nn.functional.softplus(out[..., 2, :]) + 1e-3  # floor keeps it > 0
        return log_pi, mu, sigma

    def forward_joint(self, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint MVN-mixture params per (w, d) cell-group:
        log_pi (B, W, D, K), mu (B, W, D, K, L), scale_tril (B, W, D, K, L, L).

        The "joint" head: one mixture of L-variate Gaussians per (origin, dev)
        cell-group, covering the whole line vector at once, so covariance is
        modeled explicitly rather than through conditioning.

        The group's input state is the mean of the PRESENT lines' token states
        (one d-vector per cell-group). Absent lines' tokens never attend to
        anything (padding mask), but masking them again here keeps the pooled
        input independent of whatever junk sits in their padding positions."""
        x, context_mask, line_mask, log_premium, cutoff = args
        h = self.encode(x, context_mask, line_mask, log_premium, cutoff)  # (B, L, W, D, d)
        m = line_mask[:, :, None, None, None].to(h.dtype)  # (B, L, 1, 1, 1)
        group = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B, W, D, d)
        k, n_l = self.cfg.n_components, self.n_l
        # d -> K*(1 + L + n_tril): logit, mean vector, scale_tril entries per K
        out = self.joint_head(group).reshape(*group.shape[:-1], k, 1 + n_l + self.n_tril)
        log_pi = out[..., 0].log_softmax(dim=-1)  # (B, W, D, K)
        mu = out[..., 1 : 1 + n_l]  # (B, W, D, K, L)
        tril_vals = out[..., 1 + n_l :]  # (B, W, D, K, n_tril)
        # scatter the flat entries into the lower triangle of an (L, L) matrix
        scale = torch.zeros(*mu.shape, n_l, device=mu.device, dtype=mu.dtype)  # (..., L, L)
        scale[..., self.tril_r, self.tril_c] = tril_vals
        # positive diagonal -> a valid Cholesky factor (covariance = LL^T)
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

    A marginal of a mixture of Gaussians is the mixture of the marginals, so
    for each cell-group we drop the non-target rows/cols from every component's
    mean and covariance and score the reduced-dimension density. Groups with
    the same target pattern (which lines are present) share those index ops,
    so we batch by distinct pattern — few distinct patterns in practice.
    """
    b, n_l, n_w, n_d = y.shape
    # collapse the (B, W, D) grid to one row per cell-group; L is the line axis
    group_mask = target_mask.permute(0, 2, 3, 1)  # (B, W, D, L)
    flat_mask = group_mask.reshape(-1, n_l)  # (B*W*D, L)
    active = flat_mask.any(dim=-1)  # groups with >= 1 target line
    if not bool(active.any()):
        return torch.zeros((), device=y.device)

    k = log_pi.shape[-1]
    # keep only active cell-groups, flattened to (n_active, ...)
    y_flat = y.permute(0, 2, 3, 1).reshape(-1, n_l)[active]  # (A, L)
    pi_flat = log_pi.reshape(-1, k)[active]  # (A, K)
    mu_flat = mu.reshape(-1, k, n_l)[active]  # (A, K, L)
    scale_flat = scale_tril.reshape(-1, k, n_l, n_l)[active]  # (A, K, L, L)
    pattern = flat_mask[active]  # (A, L) which lines are targets

    total, n_groups = y_flat.new_zeros(()), 0
    for pat in pattern.unique(dim=0):
        rows = (pattern == pat[None]).all(dim=-1)  # (A,) groups with this pattern
        idx = pat.nonzero(as_tuple=True)[0]  # (P,) target line indices, P = |pat|
        yp = y_flat[rows][:, idx]  # (G, P)
        mup = mu_flat[rows][:, :, idx]  # (G, K, P)
        lp = scale_flat[rows]  # (G, K, L, L)
        # marginalize to the target lines: covariance submatrix on idx x idx
        cov = lp @ lp.transpose(-1, -2)  # (G, K, L, L)
        sub = cov[:, :, idx][:, :, :, idx]  # (G, K, P, P)
        chol = torch.linalg.cholesky(sub)  # re-factor the reduced covariance
        # Mahalanobis term per component via a triangular solve (no explicit inverse)
        diff = (yp[:, None, :] - mup).unsqueeze(-1)  # (G, K, P, 1)
        z = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)  # (G, K, P)
        logdet = chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (G, K); log|chol| = .5 log|cov|
        # log N(yp; mup, sub) per component
        comp = -0.5 * (z**2).sum(-1) - logdet - 0.5 * len(idx) * LOG_2PI  # (G, K)
        ll = torch.logsumexp(pi_flat[rows] + comp, dim=-1)  # mixture log-density, (G,)
        total = total - ll.sum()
        n_groups += int(rows.sum())
    return total / max(n_groups, 1)  # mean NLL per scored cell-group


def joint_mdn_sample(
    log_pi: torch.Tensor,  # (B, W, D, K)
    mu: torch.Tensor,  # (B, W, D, K, L)
    scale_tril: torch.Tensor,  # (B, W, D, K, L, L)
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One joint draw of every cell-group's line vector: (B, W, D, L).
    Callers select the present lines / target cells from the result.

    Sampling is full-dimension here (all L lines drawn together); marginals
    are taken by the caller. Each group picks a component by inverse-CDF on the
    mixture weights, then draws x = mu + L z with z ~ N(0, I) (the reparameter-
    ization of an MVN with Cholesky factor L)."""
    k, n_l = mu.shape[-2], mu.shape[-1]
    # inverse-CDF component selection per (B, W, D) cell-group
    u = torch.rand(*log_pi.shape[:-1], 1, generator=generator, device=mu.device)
    cum = log_pi.exp().cumsum(dim=-1)
    comp = (u > cum).sum(dim=-1).clamp(max=k - 1)  # (B, W, D) chosen component
    # gather the selected component's mean and scale factor for every group
    gather = comp[..., None, None, None].expand(*comp.shape, 1, n_l, n_l)
    scale_s = scale_tril.gather(-3, gather).squeeze(-3)  # (B, W, D, L, L)
    mu_s = mu.gather(-2, comp[..., None, None].expand(*comp.shape, 1, n_l)).squeeze(-2)  # (B,W,D,L)
    z = torch.randn(*mu_s.shape, 1, generator=generator, device=mu.device)  # (B, W, D, L, 1)
    return mu_s + (scale_s @ z).squeeze(-1)  # (B, W, D, L)
