"""The DeepTriangle network: GRU encoder/decoder per origin with mixture
density heads. This module imports torch - only import it from inside the
entry's fit/predict paths (``ibnr.gallery`` must import without the [nn]
extra; see model.py and CLAUDE.md's "torch must never be imported at module
level" rule).

One piece, consumed by ``model.py``: ``DeepTriangleGRU`` - the encoder/decoder
plus two MDN heads (target + auxiliary claims outstanding). The MDN loss and
sampler are NOT redefined here: the entry imports ``mdn_nll``/``mdn_sample``
from ``gallery/nn/transformer/network.py`` - one implementation of the
mixture math per package, per the shared-machinery rule.

Kuo's DeepTriangle, adapted to this package's contract: each origin's
development history is a sequence; a GRU ENCODER consumes the observed steps
and a GRU DECODER takes over from the encoder's state to emit the remaining
steps. The two cells are dispatched per dev step by the context mask, which
generalizes the paper's clean encode-then-decode split to ragged conditioning
boundaries (each origin's context ends at a different dev for a given
calendar cutoff) and to predecessor holes. Model writeup: card.md. Data
contract feeding it: kernels/nn_contract.py."""

from __future__ import annotations

import torch
from torch import nn

from ibnr.gallery.nn.deeptriangle.config import DeepTriangleConfig


class DeepTriangleGRU(nn.Module):
    """GRU encoder/decoder over each origin's dev sequence of one cohort.

    Per dev step the input is ``Linear([channel values * flag, flag])`` plus a
    dev-lag embedding and broadcast cohort conditioning (LOB embedding +
    optional company embedding + normalized log premium). Steps where the
    context mask is True run the ENCODER cell on that input (the true values);
    steps where it is False run the DECODER cell on the same input with the
    values zeroed - position + conditioning only, so the state rolls forward
    open-loop, which IS Kuo's decoder emitting the remaining dev steps. Both
    MDN heads read the state AFTER each step; only decoded (non-context) cells
    are ever scored, so a head never reads a state that consumed the cell's
    own value.

    There is NO calendar input of any kind - no absolute calendar embedding
    (the v1/v2 transformer defect: untrained parameters injected exactly at
    forecast cells) and no explicit distance-past-cutoff either. Relative
    calendar position arises structurally from recurrence: the decoder knows
    how far past the conditioning boundary it is by how many steps it has run
    since the flag dropped, so the absolute-position failure mode is
    unrepresentable rather than merely avoided.
    """

    def __init__(
        self,
        cfg: DeepTriangleConfig,
        *,
        n_lob: int,
        n_company: int,
        n_features: int,
        n_w: int,
        n_d: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # n_w = number of origin periods (accident years), n_d = number of dev
        # lags; each origin is one sequence of n_d steps.
        self.n_w, self.n_d = n_w, n_d
        d = cfg.hidden_dim
        # value_proj: per-step [channel values, context flag] -> hidden_dim.
        # The +1 input dim is the context flag itself, fed as a feature so the
        # step encodes "this cell is / is not conditioning data".
        self.value_proj = nn.Linear(n_features + 1, d)
        # dev-lag identity per step. Development position, not calendar - a
        # dev embedding is trained wherever any cohort has data at that dev.
        self.dev_emb = nn.Embedding(n_d, d)
        # cohort conditioning: LOB identity + size (normalized log premium) +
        # optionally company identity (Kuo's design; ablatable via the config
        # flag), projected to d and broadcast onto every step.
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
        # two MDN heads over the same trunk: per cell, 3 params (mixture
        # logit, mu, raw sigma) x K components. ``head`` is the target channel
        # (the only head rollout/scoring consume); ``aux_head`` is the
        # auxiliary claims-outstanding task, training-time only.
        self.head = nn.Linear(d, 3 * cfg.n_components)
        self.aux_head = nn.Linear(d, 3 * cfg.n_components)

    def _mdn_params(
        self, raw: torch.Tensor, b: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, W, D, 3K) head output -> (log_pi, mu, sigma) each (B, W, D, K)."""
        out = raw.reshape(b, self.n_w, self.n_d, 3, self.cfg.n_components)
        log_pi = out[..., 0, :].log_softmax(dim=-1)  # normalized log weights
        mu = out[..., 1, :]  # component means (normalized ratio scale)
        # softplus keeps sigma > 0; the 1e-3 floor prevents a collapsing
        # component from driving the NLL to -inf.
        sigma = nn.functional.softplus(out[..., 2, :]) + 1e-3
        return log_pi, mu, sigma

    def forward(
        self,
        x: torch.Tensor,  # (B, F, W, D) normalized values (junk allowed off-context)
        context_mask: torch.Tensor,  # (B, W, D) bool
        lob_idx: torch.Tensor,  # (B,) long
        company_idx: torch.Tensor,  # (B,) long (ignored when the flag is off)
        log_premium: torch.Tensor,  # (B,) normalized
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Run every origin's sequence and read an MDN pair over every cell.

        B = cohorts in the batch, F = channels, W = origins, D = dev lags,
        K = mixture components. Returns ``(target, aux)`` where each is
        ``(log_pi, mu, sigma)`` with shape (B, W, D, K): the target head is a
        K-Gaussian mixture over the normalized incremental loss ratio of every
        cell, the aux head the same over the normalized incremental
        outstanding ratio. Only ``context_mask``-true cells contribute their
        values (encoder steps); the rest are zeroed and run the decoder."""
        b = x.shape[0]
        d_model = self.cfg.hidden_dim
        flag = context_mask.unsqueeze(1).to(x.dtype)  # (B, 1, W, D)
        # zero out non-context values; per-cell input = [values, flag] with
        # channels last: (B, F, W, D) -> (B, W, D, F+1)
        vals = (x * flag).permute(0, 2, 3, 1)  # (B, W, D, F)
        inp = torch.cat([vals, flag.permute(0, 2, 3, 1)], dim=-1)  # (B, W, D, F+1)
        tok = self.value_proj(inp) + self.dev_emb.weight[None, None, :, :]  # (B, W, D, d)
        # per-cohort conditioning, broadcast to every step of every origin
        cond_parts = [self.lob_emb(lob_idx), log_premium.unsqueeze(-1)]
        if self.cfg.company_embedding:
            cond_parts.insert(1, self.company_emb(company_idx))
        cond = self.cond_proj(torch.cat(cond_parts, dim=-1))  # (B, d)
        tok = self.drop(tok + cond[:, None, None, :])  # (B, W, D, d)

        # each origin is an independent sequence: fold origins into the batch
        seq = tok.reshape(b * self.n_w, self.n_d, d_model)
        is_ctx = context_mask.reshape(b * self.n_w, self.n_d)
        h = self.h0.expand(b * self.n_w, d_model)
        states = []
        for t in range(self.n_d):
            # both cells advance from the same state; the mask picks which
            # transition is real. Encoder consumes the true (flagged) values,
            # decoder the zeroed position-only input - Kuo's encode/decode
            # split, generalized to ragged boundaries and holes.
            h_enc = self.encoder(seq[:, t], h)
            h_dec = self.decoder(seq[:, t], h)
            h = torch.where(is_ctx[:, t].unsqueeze(-1), h_enc, h_dec)
            states.append(h)
        hs = torch.stack(states, dim=1)  # (B*W, D, d)
        hs = self.out_norm(hs).reshape(b, self.n_w, self.n_d, d_model)
        return self._mdn_params(self.head(hs), b), self._mdn_params(self.aux_head(hs), b)
