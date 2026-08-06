"""The mdn network: a per-cell MLP with a mixture density head - deliberately
NO sequence model and NO attention. This module imports torch - only import it
from inside the entry's fit/predict paths (``ibnr.gallery`` must import
without the [nn] extra; see model.py and CLAUDE.md's "torch must never be
imported at module level" rule).

This entry is the transformer's architecture comparison: it consumes the same
contract, the same masked-cell objective, the same training scheme and the
same MDN head, and differs ONLY in how a cell sees the rest of its triangle.
Where the transformer attends over every (origin, dev) token, the MLP here
reads a FIXED-SIZE summary of the context cells (see
:class:`TriangleMDN.forward`). Whatever attention buys shows up as the gap
between the two entries on an otherwise identical rig.

``mdn_nll`` and ``mdn_sample`` are the transformer's own head loss and
sampler, IMPORTED rather than copied - re-exported here so the entry code
reads ``net.mdn_nll`` like every other NN entry while the implementation
stays single-sourced in ``gallery/nn/transformer/network.py``."""

from __future__ import annotations

import torch
from torch import nn

from ibnr.gallery.nn.mdn.config import MDNConfig
from ibnr.gallery.nn.transformer.network import mdn_nll, mdn_sample

__all__ = ["TriangleMDN", "mdn_nll", "mdn_sample"]


class TriangleMDN(nn.Module):
    """Per-cell MLP over one cohort's grid. Same forward signature as
    ``TriangleTransformer``, so the entry's training loop, rollout and
    held-out passes are structurally identical to the transformer's.

    Each target cell's feature vector concatenates:

    - **cohort summary** - the masked mean of every (channel, dev)'s
      standardized context values, pooled over origins, each channel averaged
      over ITS OWN context cells, plus each dev's context-cell fraction
      (channel 0's count / n_w). Fixed size, recomputed from the context mask
      on every forward, so the rollout's promotion of sampled diagonals to
      context flows through it;
    - **the target origin's own masked row** - its standardized context values
      across devs (zeroed off-context) plus per-dev context flags: the
      chain-ladder-natural conditioning on the origin's own history;
    - **origin embedding + dev embedding + distance-past-cutoff embedding** -
      the calendar encoding is RELATIVE (distance past the conditioning
      cutoff, clamped to [0, n_d]), exactly like the transformer's: an
      absolute calendar embedding is the documented v1/v2 defect (forecast
      diagonals lie past the training window where an absolute embedding
      never received a gradient);
    - **LOB embedding + normalized log premium** - cohort conditioning.

    Non-context values are zeroed BEFORE any summary is taken (``x * flags``),
    which is the entry's no-leak gate: a value at a cell past the cutoff
    cannot influence any prediction, its own included. The gate is PER
    CHANNEL - a value is never consumed without its own channel's flag, so a
    feature the triangle never reported is masked rather than read as an
    observed zero. Where one structural flag is needed instead (the per-dev
    context fraction, the origin row's flag vector) it is channel 0's, the
    target's. The head is the same K-Gaussian MDN as the transformer's, over
    the normalized incremental loss ratio of every cell.
    """

    def __init__(self, cfg: MDNConfig, *, n_lob: int, n_features: int, n_w: int, n_d: int) -> None:
        super().__init__()
        self.cfg = cfg
        # n_w = number of origin periods (accident years), n_d = number of dev
        # lags; the MLP is applied independently at each of the n_w * n_d cells.
        self.n_w, self.n_d = n_w, n_d
        e = cfg.embedding_dim
        self.origin_emb = nn.Embedding(n_w, e)
        self.dev_emb = nn.Embedding(n_d, e)
        # RELATIVE calendar position: distance (in diagonals) past the
        # conditioning cutoff, clamped to [0, n_d] (hence n_d + 1 rows).
        self.dist_emb = nn.Embedding(n_d + 1, e)
        self.lob_emb = nn.Embedding(n_lob, cfg.lob_embedding_dim)
        # feature vector: cohort summary (n_f*n_d values + n_d fractions) +
        # origin row (n_f*n_d values + n_d flags) + 3 positional embeddings +
        # LOB embedding + normalized log premium (1 scalar)
        summary_dim = n_features * n_d + n_d
        row_dim = n_features * n_d + n_d
        feat_dim = summary_dim + row_dim + 3 * e + cfg.lob_embedding_dim + 1
        layers: list[nn.Module] = [
            nn.Linear(feat_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        ]
        for _ in range(cfg.n_layers - 1):
            layers += [
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
        self.body = nn.Sequential(*layers)
        # MDN head: per cell, 3 params (mixture logit, mu, raw sigma) x K
        # components. Reshaped to (..., 3, K) in forward().
        self.head = nn.Linear(cfg.hidden_dim, 3 * cfg.n_components)

        # Fixed per-cell index grids. Buffers, not parameters: they follow
        # .to(device) but never train; non-persistent because they are pure
        # functions of (n_w, n_d).
        w_grid, d_grid = torch.meshgrid(torch.arange(n_w), torch.arange(n_d), indexing="ij")
        self.register_buffer("w_grid", w_grid, persistent=False)  # (n_w, n_d)
        self.register_buffer("d_grid", d_grid, persistent=False)  # (n_w, n_d)
        # 1-based calendar diagonal, matching nn_contract's cal_idx convention
        self.register_buffer("cal_grid", w_grid + d_grid + 1, persistent=False)  # (n_w, n_d)

    def forward(
        self,
        x: torch.Tensor,  # (B, F, W, D) normalized values (junk allowed off-context)
        context_mask: torch.Tensor,  # (B, F, W, D) bool - PER CHANNEL
        lob_idx: torch.Tensor,  # (B,) long
        log_premium: torch.Tensor,  # (B,) normalized
        cutoff: torch.Tensor,  # (B,) long - 1-based conditioning diagonal
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One MDN per cell from a fixed-size view of the context.

        B = cohorts in the batch, F = channels, W = origins, D = dev lags,
        K = mixture components. Returns log_pi, mu, sigma each (B, W, D, K) -
        a K-Gaussian mixture over the normalized incremental loss ratio of
        every cell. Only cells ``context_mask`` marks true ON THEIR OWN CHANNEL
        contribute their values; the rest are zeroed before any summary, so no
        cell can read a value it is supposed to predict and no channel can read
        a value its own triangle never carried."""
        b = x.shape[0]
        n_w, n_d = self.n_w, self.n_d
        # a per-cell mask would broadcast against x rather than raise (B against
        # F on the channel axis), so its rank is checked rather than trusted
        if context_mask.dim() != x.dim():
            raise ValueError(
                f"context_mask must be per channel, shaped {tuple(x.shape)}; got "
                f"{tuple(context_mask.shape)}. Channel 0 is the target's own mask"
            )
        flags = context_mask.to(x.dtype)  # (B, F, W, D)
        vals = x * flags  # the no-leak gate: a value without ITS channel's flag dies here

        # cohort summary: masked mean per (channel, dev) over origins, each
        # channel divided by its OWN context count, plus the per-dev
        # context-cell fraction of channel 0 - a structural "how much of this
        # column exists" flag, which is the target's
        counts = flags.sum(dim=2)  # (B, F, D)
        mean_fd = vals.sum(dim=2) / counts.clamp(min=1.0)  # (B, F, D)
        frac_d = counts[:, 0] / float(n_w)  # (B, D)
        cohort = torch.cat([mean_fd.flatten(1), frac_d], dim=1)  # (B, F*D + D)

        # the target origin's own masked row: values across devs + channel 0's flags
        row_vals = vals.permute(0, 2, 1, 3).flatten(2)  # (B, W, F*D)
        rows = torch.cat([row_vals, flags[:, 0]], dim=2)  # (B, W, F*D + D)

        # relative calendar position of each cell, clamped to [0, n_d]
        dist = (self.cal_grid[None] - cutoff[:, None, None]).clamp(0, n_d)  # (B, W, D)
        # per-cohort conditioning (LOB + size), broadcast to every cell
        cond = torch.cat([self.lob_emb(lob_idx), log_premium.unsqueeze(-1)], dim=-1)  # (B, c)

        feat = torch.cat(
            [
                cohort[:, None, None, :].expand(b, n_w, n_d, cohort.shape[-1]),
                rows[:, :, None, :].expand(b, n_w, n_d, rows.shape[-1]),
                self.origin_emb(self.w_grid)[None].expand(b, n_w, n_d, -1),
                self.dev_emb(self.d_grid)[None].expand(b, n_w, n_d, -1),
                self.dist_emb(dist),
                cond[:, None, None, :].expand(b, n_w, n_d, cond.shape[-1]),
            ],
            dim=-1,
        )  # (B, W, D, feat_dim)
        out = self.head(self.body(feat)).reshape(b, n_w, n_d, 3, self.cfg.n_components)
        log_pi = out[..., 0, :].log_softmax(dim=-1)  # (B, W, D, K) normalized log weights
        mu = out[..., 1, :]  # (B, W, D, K) component means (normalized ratio scale)
        # softplus keeps sigma > 0; the 1e-3 floor prevents a collapsing
        # component from driving the NLL to -inf. Same head math as the
        # transformer's - the comparison shares the distribution family exactly.
        sigma = nn.functional.softplus(out[..., 2, :]) + 1e-3  # (B, W, D, K)
        return log_pi, mu, sigma
