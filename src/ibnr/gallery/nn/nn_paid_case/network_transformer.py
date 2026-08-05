"""The ATTENTION backbone of ``nn_paid_case``: masked-cell encoder over the whole
origin x dev grid, ending in the shared bivariate mixture head.

This module imports torch - only import it from inside the entry's fit/predict
paths (``ibnr.gallery`` must import without the [nn] extra; see model.py and
CLAUDE.md's "torch must never be imported at module level" rule).

It is ``gallery/nn/transformer/network.py``'s encoder, written out literally
rather than imported, for the eject-pattern reason (decision 6: each model
directory stands alone and readable) and for one substantive one: the output
layer is not the same object. Everything up to ``out_norm`` is the transformer's
shape - per-cell tokens of ``[channel values * channel flags, channel flags]``,
learned origin/dev embeddings, a RELATIVE calendar embedding (distance past the
conditioning cutoff), broadcast LOB + log-premium conditioning - and then the
univariate MDN head is replaced by :class:`~ibnr.gallery.nn.nn_paid_case.head.
BivariateMixtureHead`, which reads a joint law over (paid increment, case
movement) at every cell.

What is deliberately NOT carried over: the transformer entry's optional
``exposure_sigma`` power. Its lever is the scale of a univariate sigma, and the
scale here is a 2x2 Cholesky factor whose off-diagonal is the quantity this
entry exists to learn; widening it by a premium power is a design question of
its own, not a flag to copy. See card.md "Limitations".
"""

from __future__ import annotations

import torch
from torch import nn

from ibnr.gallery.nn.nn_paid_case.config import NNPaidCaseConfig
from ibnr.gallery.nn.nn_paid_case.head import BivariateMixtureHead


class PaidCaseTransformer(nn.Module):
    """Encoder over the full origin x dev grid of one cohort, bivariate head.

    Every cell is a token: ``[channel values * channel flags, channel flags]``
    projected to ``d_model``, plus learned origin / dev embeddings, a *relative*
    calendar embedding (distance past the conditioning cutoff, clamped to
    ``[0, n_d]``) and broadcast cohort conditioning (LOB embedding + normalized
    log premium). The head reads a bivariate mixture per cell; the loss is
    evaluated only where the entry's target masks say so.

    The context flags are PER CHANNEL, and for this entry that is load-bearing
    rather than merely correct: channel 1 is the case LEVEL, which the rollout
    SIMULATES and promotes, so "is this channel's value at this cell real (or
    simulated) rather than contract padding" is a per-(channel, cell) fact that
    changes as the rollout advances. A per-cell mask cannot express it, and is
    refused by name in :meth:`forward`.

    Why relative, not absolute, calendar position: forecasting happens on
    calendar diagonals beyond the training window, where an absolute learned
    embedding never received a gradient. Distance-past-cutoff is supervised
    directly by the cutoff augmentation, and the rollout re-encodes after every
    sampled diagonal, so inference only ever consumes distance-1 predictions.
    """

    #: ``forward``'s positional arguments, in order, as keys of the entry's
    #: input dict (``model.py::call_backbone``). Declared rather than inferred so
    #: the two backbones can take genuinely different inputs - this one needs a
    #: calendar cutoff and has no company embedding; the GRU is the reverse -
    #: without either growing a parameter it ignores.
    INPUT_KEYS = ("x", "ctx", "lob", "prem", "cutoff")

    def __init__(
        self, cfg: NNPaidCaseConfig, *, n_lob: int, n_features: int, n_w: int, n_d: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # n_w = origin periods (accident years), n_d = dev lags; the grid has
        # n_w * n_d cells and each cell is one token.
        self.n_w, self.n_d = n_w, n_d
        d = cfg.d_model
        # per-cell [masked channel values, channel flags] -> d_model. 2F inputs,
        # one flag per channel, fed as features so a token encodes "this channel
        # is / is not conditioning data here", not just its value.
        self.value_proj = nn.Linear(2 * n_features, d)
        self.origin_emb = nn.Embedding(n_w, d)
        self.dev_emb = nn.Embedding(n_d, d)
        # RELATIVE calendar position: diagonals past the conditioning cutoff,
        # clamped to [0, n_d] (hence n_d + 1 rows).
        self.dist_emb = nn.Embedding(n_d + 1, d)
        # cohort conditioning: LOB identity + size (normalized log premium).
        # No company embedding on purpose - see the config's note.
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
        # THE head: one bivariate mixture per cell over (paid increment, case
        # movement) on the standardized scale. Shared with the GRU backbone -
        # same module, same weights shape - which is what makes the two a clean
        # encoder ablation.
        self.head = BivariateMixtureHead(d, cfg.n_components)

        # Per-token (origin, dev, calendar-diagonal) indices for the flattened
        # T = n_w * n_d sequence, in the row-major (origin-major) order forward()
        # flattens x with. Buffers, not parameters: fixed lookups that follow
        # .to(device) but never train, and non-persistent so they stay out of
        # checkpoints (pure functions of n_w, n_d).
        w_idx, d_idx = torch.meshgrid(torch.arange(n_w), torch.arange(n_d), indexing="ij")
        self.register_buffer("w_idx", w_idx.reshape(-1), persistent=False)  # (T,)
        self.register_buffer("d_idx", d_idx.reshape(-1), persistent=False)  # (T,)
        # 1-based calendar diagonal, matching nn_contract's cal_idx convention
        self.register_buffer("cal_idx", (w_idx + d_idx + 1).reshape(-1), persistent=False)  # (T,)

    def forward(
        self,
        x: torch.Tensor,  # (B, F, W, D) normalized values (junk allowed off-context)
        context_mask: torch.Tensor,  # (B, F, W, D) bool - PER CHANNEL
        lob_idx: torch.Tensor,  # (B,) long
        log_premium: torch.Tensor,  # (B,) normalized
        cutoff: torch.Tensor,  # (B,) long - 1-based conditioning diagonal
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one batch of cohort grids and read a bivariate mixture per cell.

        B = cohorts in the batch, F = channels, W = origins, D = dev lags,
        T = W*D tokens, K = mixture components. Returns the head's triple:
        ``log_pi (B, W, D, K)``, ``mu (B, W, D, K, 2)``,
        ``chol (B, W, D, K, 2, 2)`` - coordinate 0 is the paid increment,
        coordinate 1 the case movement, both on the entry's standardized scale.
        Only ``context_mask``-true (channel, cell) pairs contribute values.
        """
        b = x.shape[0]
        if context_mask.dim() != x.dim():
            raise ValueError(
                f"context_mask has {context_mask.dim()} dimensions; it is PER CHANNEL, so "
                f"it must be (B, F, W, D) = {tuple(x.shape)} like x. A (B, W, D) per-cell "
                "mask cannot say that a cell's paid increment is simulated while its case "
                "level is not (or the reverse), which is exactly what this entry's rollout "
                "manipulates"
            )
        flag = context_mask.to(x.dtype)  # (B, F, W, D)
        # zero each channel's non-context values, then flatten grid -> token
        # sequence: (B, F, W, D) -> (B, F, T) -> (B, T, F), row-major over (W, D)
        # so token order matches the w_idx/d_idx/cal_idx buffers.
        vals = (x * flag).flatten(2).transpose(1, 2)  # (B, T, F)
        # append every channel's own flag -> (B, T, 2F) -> (B, T, d_model). A
        # value never travels without the flag that says whether it is real.
        tok = self.value_proj(torch.cat([vals, flag.flatten(2).transpose(1, 2)], dim=-1))
        dist = (self.cal_idx[None, :] - cutoff[:, None]).clamp(0, self.n_d)  # (B, T)
        tok = tok + self.origin_emb(self.w_idx) + self.dev_emb(self.d_idx) + self.dist_emb(dist)
        cond = self.cond_proj(torch.cat([self.lob_emb(lob_idx), log_premium.unsqueeze(-1)], -1))
        tok = tok + cond.unsqueeze(1)  # (B, T, d)
        h = self.encoder(self.drop(tok))  # (B, T, d) - full self-attention over the grid
        h = self.out_norm(h).reshape(b, self.n_w, self.n_d, -1)  # (B, W, D, d)
        return self.head(h)
