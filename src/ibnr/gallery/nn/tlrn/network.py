"""The axial attention body of ``tlrn``.

This module imports torch - only import it from inside the entry's fit/predict
paths (``ibnr.gallery`` must import without the [nn] extra; see CLAUDE.md's
"torch must never be imported at module level" rule).

THE TOKEN GRID. One example is a company at one accident year, and its tokens
are that year's (line, development lag) cells, line-major with the lag varying
fastest. Attention over all ``n_lines * n_lag`` tokens at once would spend most
of its weight on pairs that have nothing to say to each other, so each block
attends along the two axes in turn: first across the LINES within a lag, then
across the LAGS within a line. Two small attention matrices instead of one large
one, which on a grid of forty tokens is the difference between a model that fits
Schedule P triangles and one that memorises them.

WHAT IS MASKED AND WHAT IS NOT. A company does not write every line, and an
absent line's tokens are contract padding. The line attention therefore carries
a key padding mask, so a written line's output cannot depend on an absent one -
tested by perturbing the absent line's inputs and requiring the written lines'
outputs not to move. The lag attention needs no mask: it only ever sees one
line's own lags.

An example with NO written line is refused rather than answered. Every key would
be masked, the attention weights would be a softmax over nothing, and the output
would be NaN at every token of that example, which then poisons the whole
batch's gradient.

THE OUTPUT IS NOT A CELL. The body ends in one number per token, which becomes a
per (line, step) correction to the learned log development factors in
``head.py``; the cells follow from projecting the cumulative forward. See that
module for why.

Two embedding tables and a third for position: the line, the lag, and how many
lags of this accident year are already observed. All three are indexed 1-based
with row 0 unused, matching the reference implementation, which is where the
parameter count of 14,309 at ``d_model = 32`` comes from.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ibnr.gallery.nn.tlrn import head as tlrn_head

__all__ = ["AxialBlock", "TLRNNetwork"]

#: the head's output layer starts this much smaller than the default
#: initialisation, so a fresh network is a small correction to the learned log
#: factors rather than a competing prediction
HEAD_INIT_SCALE = 0.01


class AxialBlock(nn.Module):
    """Pre-norm attention across lines, then across lags, then a feed-forward.

    ``cross_line=False`` skips the line attention. The module is still built, so
    the parameter count does not move between the two arms and a comparison of
    them is a comparison of the attention rather than of the model size; the
    unused weights simply never receive a gradient.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        n_lines: int,
        n_lag: int,
        cross_line: bool = True,
    ) -> None:
        super().__init__()
        self.cross_line = cross_line
        self.n_lines = n_lines
        self.n_lag = n_lag
        self.attn_line = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_lag = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, written: Tensor, need_weights: bool = False
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """``x`` (B, n_lines * n_lag, d), ``written`` (B, n_lines) bool."""
        batch, _, d = x.shape
        n_l, n_j = self.n_lines, self.n_lag
        attn_line = attn_lag = None

        if self.cross_line:
            if not bool(written.any(dim=1).all()):
                raise ValueError(
                    "an example has no written line, so every key of the cross-line "
                    "attention is padded and its output is NaN at every token. Drop the "
                    "company from the cohort rather than carrying an example with no data"
                )
            # (B, T, d) -> (B, n_lag, n_lines, d) -> one sequence of lines per lag
            h = self.ln1(x).reshape(batch, n_l, n_j, d).permute(0, 2, 1, 3)
            h = h.reshape(batch * n_j, n_l, d)
            pad = (~written).unsqueeze(1).expand(batch, n_j, n_l).reshape(batch * n_j, n_l)
            a, attn_line = self.attn_line(h, h, h, key_padding_mask=pad, need_weights=need_weights)
            a = a.reshape(batch, n_j, n_l, d).permute(0, 2, 1, 3).reshape(batch, n_l * n_j, d)
            x = x + self.drop(a)

        # (B, T, d) -> one sequence of lags per line; no mask, a line only sees
        # its own lags and every lag of a written line is a real position
        h = self.ln2(x).reshape(batch * n_l, n_j, d)
        a, attn_lag = self.attn_lag(h, h, h, need_weights=need_weights)
        x = x + self.drop(a.reshape(batch, n_l * n_j, d))
        x = x + self.ff(self.ln3(x))
        return x, attn_line, attn_lag


class TLRNNetwork(nn.Module):
    """The development-factor network: axial body, learned log factors, head.

    ``forward`` returns a dict carrying ``pred`` (B, n_lines * n_lag) predicted
    incremental loss ratios, ``logf`` (B, n_lines, n_lag - 1) the log
    development factors it used, ``C`` (B, n_lines, n_lag) the projected
    cumulative grid in amounts, and the two attention weight stacks when they
    were asked for.
    """

    def __init__(self, cfg: Any, *, n_lines: int, n_lag: int, n_feat: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_lines = n_lines
        self.n_lag = n_lag
        d = cfg.d_model
        self.inp = nn.Linear(n_feat, d)
        # row 0 of each table is unused: the indices are 1-based, as the
        # reference implementation's are, and the count depends on it
        self.line_emb = nn.Embedding(n_lines + 1, d)
        self.lag_emb = nn.Embedding(n_lag + 1, d)
        self.nobs_emb = nn.Embedding(n_lag + 2, d)
        self.blocks = nn.ModuleList(
            [
                AxialBlock(d, cfg.n_heads, cfg.dropout, n_lines, n_lag, cfg.cross_line)
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        # the log factor at step m starts at init_step / m, a decaying pattern
        # whose shape is the reference's and whose magnitudes are learned
        steps = cfg.init_step / torch.arange(1, n_lag, dtype=torch.float32)
        self.phi = nn.Parameter(torch.log(torch.exp(steps) - 1.0).repeat(n_lines, 1))
        # what softplus(phi) is at that initialisation, which the anchored
        # variant measures its correction from. Not persistent: it is a function
        # of the config, so a reloaded state dict must not be able to disagree
        # with the config it is reloaded under.
        self.register_buffer("initial_logf", steps.repeat(n_lines, 1), persistent=False)
        with torch.no_grad():
            self.head.weight.mul_(HEAD_INIT_SCALE)
            self.head.bias.zero_()
            if cfg.cl_anchor:
                # the anchored variant starts AT the chain ladder, not near it
                self.head.weight.zero_()

    def forward(
        self,
        feat: Tensor,
        c_lk: Tensor,
        p_lk: Tensor,
        lk: Tensor,
        line_ix: Tensor,
        lag_ix: Tensor,
        written: Tensor,
        *,
        use_residual: bool = True,
        need_weights: bool = False,
        factor_support: Tensor | None = None,
        fallback_logf: Tensor | None = None,
        anchor_logf: Tensor | None = None,
        anchor_start: Tensor | None = None,
    ) -> dict[str, Any]:
        """One forward pass over a batch of (company, accident year) examples.

        ``use_residual=False`` skips the body entirely and projects with the
        learned log factors alone, which is how the head is checked against the
        chain ladder and how the ``factors_only`` variant runs.
        """
        batch = feat.shape[0]
        n_l, n_j = self.n_lines, self.n_lag
        use_residual = use_residual and not self.cfg.factors_only
        net = None
        attn_line: list = []
        attn_lag: list = []
        if use_residual:
            h = self.inp(feat) + self.line_emb(line_ix) + self.lag_emb(lag_ix)
            # one more position: how many lags of this accident year are already
            # observed, which is the same for every token of the example
            h = h + self.nobs_emb(lk + 1).unsqueeze(-2)
            for block in self.blocks:
                h, w_line, w_lag = block(h, written, need_weights=need_weights)
                attn_line.append(w_line)
                attn_lag.append(w_lag)
            # one number per token, read as a correction per (line, step); the
            # last lag has no step past it, so its column is dropped
            net = self.head(self.ln(h)).squeeze(-1).reshape(batch, n_l, n_j)[:, :, : n_j - 1]

        phi = self.phi.unsqueeze(0).expand(batch, n_l, n_j - 1)
        logf = tlrn_head.log_factors(phi, net if use_residual else None, self.cfg.eps)

        if self.cfg.cl_anchor:
            if anchor_logf is None:
                raise ValueError(
                    "cl_anchor needs anchor_logf, the company's own chain ladder log "
                    "factors as of this example's cutoff: the variant learns a bounded "
                    "correction around them and has no meaning without them"
                )
            logf = tlrn_head.anchor(
                logf, anchor_logf, self.initial_logf.unsqueeze(0), self.cfg.anchor_width
            )

        if factor_support is not None:
            if fallback_logf is None:
                raise ValueError(
                    "a factor support mask needs fallback_logf, the chain ladder log "
                    "factors observable at this example's cutoff, to substitute at the "
                    "steps no training target supervised"
                )
            # under the anchored variant the company's own factors already ARE
            # the fallback, and substituting the pooled ones would move a step
            # the correction was measured from
            fallback = anchor_logf if self.cfg.cl_anchor else fallback_logf.unsqueeze(0)
            logf = tlrn_head.apply_support(logf, factor_support, fallback)

        if self.cfg.cl_anchor and anchor_start is None:
            raise ValueError(
                "cl_anchor needs anchor_start, the UNFLOORED cumulative each origin "
                "starts from: the variant projects from the company's own balance, "
                "including a balance of zero, which the floored c_lk cannot represent"
            )
        pred, c = tlrn_head.project(
            logf,
            c_lk,
            p_lk,
            lk,
            anchor_start=anchor_start if self.cfg.cl_anchor else None,
        )
        return {
            "pred": pred,
            "logf": logf,
            "C": c,
            "attn_line": attn_line,
            "attn_lag": attn_lag,
        }
