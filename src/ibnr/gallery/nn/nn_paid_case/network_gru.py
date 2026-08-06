"""The RECURRENT backbone of ``nn_paid_case``: a GRU encoder/decoder per origin,
ending in the same bivariate mixture head as the attention backbone.

This module imports torch - only import it from inside the entry's fit/predict
paths (``ibnr.gallery`` must import without the [nn] extra; see model.py and
CLAUDE.md's "torch must never be imported at module level" rule).

Adapted from ``gallery/nn/deeptriangle/network.py`` (Kuo's DeepTriangle shape:
each origin's development history is a sequence; a GRU ENCODER consumes the
observed steps and a GRU DECODER takes over from the encoder's state to emit the
remaining ones, dispatched per step by the TARGET channel's context flag, which
generalizes the paper's clean encode-then-decode split to ragged conditioning
boundaries and predecessor holes). Two differences, and they are the entry:

- the two univariate MDN heads (target + auxiliary) are replaced by ONE
  :class:`~ibnr.gallery.nn.nn_paid_case.head.BivariateMixtureHead`. DeepTriangle's
  auxiliary claims-outstanding head is a training-time regularizer that nothing
  downstream consumes; here the second coordinate is a JOINT one, sampled with
  the paid increment from the same component and the same ``z``, and fed back
  into the rollout. There is no auxiliary weight to tune because there is no
  second loss - one density covers both quantities.
- the encoder/decoder dispatch still reads CHANNEL 0's flag, and that is now a
  choice worth stating: the rollout promotes BOTH channels at a sampled cell, so
  the two flags agree at every cell the rollout has filled, and channel 0's is
  the one that also marks genuine training observations of the emergence being
  predicted.

Why a second backbone at all: the entry's claim is about the DATA (case reserves
are a state with dynamics, and simulating them forward is worth doing), not
about attention. Two bodies over one head, one loss and one contract is what
makes that claim separable from the encoder - see card.md "Backbones".
"""

from __future__ import annotations

import torch
from torch import nn

from ibnr.gallery.nn.nn_paid_case.config import NNPaidCaseConfig
from ibnr.gallery.nn.nn_paid_case.head import BivariateMixtureHead


class PaidCaseGRU(nn.Module):
    """GRU encoder/decoder over each origin's dev sequence, bivariate head.

    Per dev step the input is ``Linear([channel values * channel flags, channel
    flags])`` plus a dev-lag embedding and broadcast cohort conditioning (LOB
    embedding + optional company embedding + normalized log premium). The
    context mask is PER CHANNEL, so a channel's value is consumed only where
    that channel has a usable (or simulated) value of its own. Steps where
    channel 0's flag is True run the encoder cell; steps where it is False run
    the decoder cell on the same input, so the state rolls forward open-loop -
    Kuo's decoder emitting the remaining dev steps. The head reads the state
    AFTER each step, and only non-context cells are ever scored, so it never
    reads a state that consumed the cell's own value.

    **Each origin is an INDEPENDENT sequence at inference.** The recurrence runs
    along the development axis only - origins are folded into the batch - so
    nothing flows between origins inside a forward pass; they share information
    only through the trained weights. The transformer backbone is the opposite,
    and that is the substance of the comparison between them.

    There is NO calendar input of any kind, and none is needed: relative
    position arises structurally from the recurrence (the decoder knows how far
    past the boundary it is by how many steps it has run since the flag
    dropped), so the absolute-calendar failure mode is unrepresentable rather
    than merely avoided. This is why :data:`INPUT_KEYS` carries no ``cutoff``.
    """

    #: ``forward``'s positional arguments, in order, as keys of the entry's
    #: input dict (``model.py::call_backbone``). No ``cutoff``: this body has no
    #: calendar boundary to place, and advertising one it ignored would be an
    #: inert parameter - the bug class the repo names.
    INPUT_KEYS = ("x", "ctx", "lob", "comp", "prem")

    def __init__(
        self,
        cfg: NNPaidCaseConfig,
        *,
        n_lob: int,
        n_company: int,
        n_features: int,
        n_w: int,
        n_d: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_w, self.n_d = n_w, n_d
        d = cfg.hidden_dim
        # 2F, not F+1: every channel carries its OWN flag, so a step encodes
        # "this channel is / is not conditioning data here" per channel.
        self.value_proj = nn.Linear(2 * n_features, d)
        # dev-lag identity per step. Development position, not calendar.
        self.dev_emb = nn.Embedding(n_d, d)
        self.lob_emb = nn.Embedding(n_lob, cfg.lob_embedding_dim)
        cond_in = cfg.lob_embedding_dim + 1
        if cfg.company_embedding:
            self.company_emb = nn.Embedding(n_company, cfg.company_embedding_dim)
            cond_in += cfg.company_embedding_dim
        self.cond_proj = nn.Linear(cond_in, d)
        self.drop = nn.Dropout(cfg.dropout)
        # the encoder/decoder pair; a learned shared initial state seeds both
        self.encoder = nn.GRUCell(d, d)
        self.decoder = nn.GRUCell(d, d)
        self.h0 = nn.Parameter(torch.zeros(d))
        self.out_norm = nn.LayerNorm(d)
        self.head = BivariateMixtureHead(d, cfg.n_components)

    def forward(
        self,
        x: torch.Tensor,  # (B, F, W, D) normalized values (junk allowed off-context)
        context_mask: torch.Tensor,  # (B, F, W, D) bool - PER CHANNEL
        lob_idx: torch.Tensor,  # (B,) long
        company_idx: torch.Tensor,  # (B,) long (ignored when the flag is off)
        log_premium: torch.Tensor,  # (B,) normalized
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run every origin's sequence and read a bivariate mixture per cell.

        B = cohorts in the batch, F = channels, W = origins, D = dev lags,
        K = mixture components. Returns the head's triple:
        ``log_pi (B, W, D, K)``, ``mu (B, W, D, K, 2)``,
        ``chol (B, W, D, K, 2, 2)`` - coordinate 0 the paid increment,
        coordinate 1 the case movement, on the entry's standardized scale.
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
        d_model = self.cfg.hidden_dim
        flags = context_mask.to(x.dtype)  # (B, F, W, D)
        # zero off-context values CHANNEL BY CHANNEL; per-cell input =
        # [values, flags] with channels last: (B, F, W, D) -> (B, W, D, 2F)
        vals = (x * flags).permute(0, 2, 3, 1)  # (B, W, D, F)
        inp = torch.cat([vals, flags.permute(0, 2, 3, 1)], dim=-1)  # (B, W, D, 2F)
        tok = self.value_proj(inp) + self.dev_emb.weight[None, None, :, :]  # (B, W, D, d)
        cond_parts = [self.lob_emb(lob_idx), log_premium.unsqueeze(-1)]
        if self.cfg.company_embedding:
            cond_parts.insert(1, self.company_emb(company_idx))
        cond = self.cond_proj(torch.cat(cond_parts, dim=-1))  # (B, d)
        tok = self.drop(tok + cond[:, None, None, :])  # (B, W, D, d)

        # each origin is an independent sequence: fold origins into the batch
        seq = tok.reshape(b * self.n_w, self.n_d, d_model)
        # channel 0's flag: one cell is one encoder-or-decoder step, and the
        # paid increment is what the step is a step of
        is_ctx = context_mask[:, 0].reshape(b * self.n_w, self.n_d)
        h = self.h0.expand(b * self.n_w, d_model)
        states = []
        for t in range(self.n_d):
            # both cells advance from the same state; the mask picks which
            # transition is real
            h_enc = self.encoder(seq[:, t], h)
            h_dec = self.decoder(seq[:, t], h)
            h = torch.where(is_ctx[:, t].unsqueeze(-1), h_enc, h_dec)
            states.append(h)
        hs = torch.stack(states, dim=1)  # (B*W, D, d)
        hs = self.out_norm(hs).reshape(b, self.n_w, self.n_d, d_model)
        return self.head(hs)
