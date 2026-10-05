"""The axial attention body of ``tlrn``.

This module imports torch - only import it from inside the entry's fit/predict
paths (``ibnr.gallery`` must import without the [nn] extra; see CLAUDE.md's
"torch must never be imported at module level" rule).

THE TOKEN GRID. One example is a company at one accident year, and its tokens
are that year's (line, development lag) cells, line-major with the lag varying
fastest. Attention over all ``n_lines * n_lag`` tokens at once would spend most
of its weight on pairs that have nothing to say to each other, so each block
attends along the axes in turn: first across the LINES within a lag, then across
the LAGS within a line, and - when ``attention`` names it - across the ACCIDENT
YEARS of one company within a (line, lag). Small attention matrices instead of
one large one, which on a grid of forty tokens is the difference between a model
that fits Schedule P triangles and one that memorises them.

THE ACCIDENT-YEAR AXIS NEEDS WHOLE COMPANIES. Examples are laid out company-major
(``example = company * n_origin + origin``), and the accident-year attention
reads each run of ``n_origin`` consecutive examples as one company. So a batch
fed to a network with that axis must be whole companies in origin order, which is
what ``batch_unit="company"`` produces; a batch that breaks the run is refused by
name rather than attended across companies. That separation is tested by
perturbing one company's inputs and requiring every other company's outputs not
to move.

WHAT IS MASKED AND WHAT IS NOT. ``mask="unwritten_lines"`` (the published model)
masks the keys of a line the company does not write: an absent line's tokens are
contract padding, so a written line's output cannot depend on them - tested by
perturbing the absent line's inputs and requiring the written lines' outputs not
to move. The lag attention needs no mask there: it only ever sees one line's own
lags. ``mask="observed_cells"`` masks every key the forecast date has not
revealed, on all three axes, so a token can only read cells that exist at the
cutoff. A token with nothing left to read gets nothing: its attention output is
zero, not the NaN a softmax over no keys would give.

An example with NO written line is refused rather than answered, whenever the
line attention is on. Every key would be masked, the attention weights would be a
softmax over nothing, and the output would be NaN at every token of that example,
which then poisons the whole batch's gradient.

THE OUTPUT IS NOT A CELL. The body ends in one number per token, which the head
reads. Under ``head="ldf"`` it is a per (line, step) correction to the learned log
development factors and the cells follow from projecting the cumulative forward;
under ``head="premium_lr"`` it is a correction to the learned log incremental
loss ratio of that (line, lag) and a cell is that ratio times premium. See
``head.py`` for why.

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
from ibnr.gallery.nn.tlrn.components import attention_axes

__all__ = ["AxialBlock", "TLRNNetwork"]

#: the head's output layer starts this much smaller than the default
#: initialisation, so a fresh network is a small correction to the learned log
#: factors rather than a competing prediction
HEAD_INIT_SCALE = 0.01


def attend(
    attn: nn.MultiheadAttention,
    h: Tensor,
    readable: Tensor | None,
    need_weights: bool,
    *,
    guard: bool,
) -> tuple[Tensor, Tensor | None]:
    """One attention over ``h`` (N, S, d) reading only the keys ``readable`` (N, S) marks.

    With ``guard`` a sequence whose keys are all unreadable gets a zero output
    instead of the NaN a softmax over nothing returns: its first key is let
    through so the softmax is finite, and the output is then multiplied by zero
    for that sequence. Without ``guard`` the call is the plain masked attention,
    which is what keeps the published model's arithmetic exactly as it was.
    """
    if readable is None:
        return attn(h, h, h, need_weights=need_weights)
    pad = ~readable
    if not guard:
        return attn(h, h, h, key_padding_mask=pad, need_weights=need_weights)
    any_key = readable.any(dim=1)
    pad = pad.clone()
    pad[:, 0] &= any_key  # a sequence with no key lets its first one through
    out, weights = attn(h, h, h, key_padding_mask=pad, need_weights=need_weights)
    return out * any_key.view(-1, 1, 1).to(out.dtype), weights


class AxialBlock(nn.Module):
    """Pre-norm attention across lines, lags and (optionally) accident years, then a feed-forward.

    ``cross_line=False`` skips the line attention. The module is still built, so
    the parameter count does not move between the two arms and a comparison of
    them is a comparison of the attention rather than of the model size; the
    unused weights simply never receive a gradient. ``axes`` says it directly and
    wins when given. The accident-year attention is built only when ``axes`` names
    it, and after everything else, so a network without it draws exactly the
    initial weights it always drew.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        n_lines: int,
        n_lag: int,
        cross_line: bool = True,
        *,
        axes: tuple[str, ...] | None = None,
        n_origin: int | None = None,
        mask: str = "unwritten_lines",
    ) -> None:
        super().__init__()
        self.axes = (
            tuple(axes) if axes is not None else (("line", "lag") if cross_line else ("lag",))
        )
        self.cross_line = "line" in self.axes
        self.mask = mask
        self.n_lines = n_lines
        self.n_lag = n_lag
        self.n_origin = n_origin
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
        if "ay" in self.axes:
            if n_origin is None or n_origin < 1:
                raise ValueError(
                    "the accident-year attention reads runs of n_origin examples as one "
                    f"company, so it needs n_origin >= 1, got {n_origin!r}"
                )
            self.attn_ay = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.ln_ay = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        written: Tensor,
        need_weights: bool = False,
        visible: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """``x`` (B, n_lines * n_lag, d), ``written`` (B, n_lines) bool.

        ``visible`` (B, n_lines * n_lag) bool is the tokens the forecast date has
        revealed, and is what ``mask="observed_cells"`` reads.
        """
        batch, n_tok, d = x.shape
        n_l, n_j = self.n_lines, self.n_lag
        attn_line = attn_lag = attn_ay = None
        observed = self.mask == "observed_cells"
        if observed and visible is None:
            raise ValueError(
                "mask='observed_cells' reads which tokens the forecast date has revealed, "
                "and this call carried no `visible`"
            )
        vis = visible.bool().reshape(batch, n_l, n_j) if observed else None

        if "line" in self.axes:
            if not bool(written.any(dim=1).all()):
                raise ValueError(
                    "an example has no written line, so every key of the cross-line "
                    "attention is padded and its output is NaN at every token. Drop the "
                    "company from the cohort rather than carrying an example with no data"
                )
            # (B, T, d) -> (B, n_lag, n_lines, d) -> one sequence of lines per lag
            h = self.ln1(x).reshape(batch, n_l, n_j, d).permute(0, 2, 1, 3)
            h = h.reshape(batch * n_j, n_l, d)
            if observed:
                readable = vis.permute(0, 2, 1).reshape(batch * n_j, n_l)
            else:
                readable = written.unsqueeze(1).expand(batch, n_j, n_l).reshape(batch * n_j, n_l)
            a, attn_line = attend(self.attn_line, h, readable, need_weights, guard=observed)
            a = a.reshape(batch, n_j, n_l, d).permute(0, 2, 1, 3).reshape(batch, n_l * n_j, d)
            x = x + self.drop(a)

        # (B, T, d) -> one sequence of lags per line. Under the published mask no
        # mask at all: a line only sees its own lags and every lag of a written
        # line is a real position
        h = self.ln2(x).reshape(batch * n_l, n_j, d)
        readable = vis.reshape(batch * n_l, n_j) if observed else None
        a, attn_lag = attend(self.attn_lag, h, readable, need_weights, guard=observed)
        x = x + self.drop(a.reshape(batch, n_l * n_j, d))

        if "ay" in self.axes:
            n_w = self.n_origin
            if batch % n_w:
                raise ValueError(
                    f"the accident-year attention reads runs of {n_w} consecutive examples as "
                    f"one company, and this batch has {batch} examples, which is not a whole "
                    "number of companies. Batch whole companies: batch_unit='company'"
                )
            n_c = batch // n_w
            # (B, T, d) -> (n_c, n_w, T, d) -> one sequence of accident years per
            # (company, token)
            h = self.ln_ay(x).reshape(n_c, n_w, n_tok, d).permute(0, 2, 1, 3)
            h = h.reshape(n_c * n_tok, n_w, d)
            if observed:
                token_ok = vis.reshape(batch, n_tok)
            else:
                token_ok = written.unsqueeze(2).expand(batch, n_l, n_j).reshape(batch, n_tok)
            readable = token_ok.reshape(n_c, n_w, n_tok).permute(0, 2, 1).reshape(n_c * n_tok, n_w)
            a, attn_ay = attend(self.attn_ay, h, readable, need_weights, guard=True)
            a = a.reshape(n_c, n_tok, n_w, d).permute(0, 2, 1, 3).reshape(batch, n_tok, d)
            x = x + self.drop(a)

        x = x + self.ff(self.ln3(x))
        return x, attn_line, attn_lag, attn_ay


class TLRNNetwork(nn.Module):
    """The loss reserving network: axial body, learned head parameters, head.

    ``forward`` returns a dict carrying ``pred`` (B, n_lines * n_lag) predicted
    incremental loss ratios, ``C`` (B, n_lines, n_lag) the projected cumulative
    grid in amounts, ``logf`` (B, n_lines, n_lag - 1) the log development
    factors it used (``head="ldf"``) or ``logr`` (B, n_lines, n_lag) the log
    incremental loss ratios (``head="premium_lr"``), and the attention weight
    stacks when they were asked for.

    ``n_origin`` is how many accident years make one company and is needed by the
    accident-year attention only. ``head_init`` is the premium head's starting
    log incremental loss ratio, (n_lines, n_lag), which has no default because
    the right one is a statement about the data.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        n_lines: int,
        n_lag: int,
        n_feat: int,
        n_origin: int | None = None,
        head_init: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_lines = n_lines
        self.n_lag = n_lag
        self.n_origin = n_origin
        d = cfg.d_model
        self.inp = nn.Linear(n_feat, d)
        # row 0 of each table is unused: the indices are 1-based, as the
        # reference implementation's are, and the count depends on it
        self.line_emb = nn.Embedding(n_lines + 1, d)
        self.lag_emb = nn.Embedding(n_lag + 1, d)
        self.nobs_emb = nn.Embedding(n_lag + 2, d)
        axes = attention_axes(cfg)
        self.blocks = nn.ModuleList(
            [
                AxialBlock(
                    d,
                    cfg.n_heads,
                    cfg.dropout,
                    n_lines,
                    n_lag,
                    cfg.cross_line,
                    axes=axes,
                    n_origin=n_origin,
                    mask=cfg.mask,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        if cfg.head == "premium_lr":
            if head_init is None:
                raise ValueError(
                    "head='premium_lr' starts from a log incremental loss ratio per "
                    "(line, lag), which has no default: pass head_init, the log of the "
                    "pooled incremental loss ratio known when training starts"
                )
            start = torch.as_tensor(head_init, dtype=torch.float32)
            if tuple(start.shape) != (n_lines, n_lag):
                raise ValueError(
                    f"head_init must be (n_lines, n_lag) = {(n_lines, n_lag)}, got "
                    f"{tuple(start.shape)}"
                )
            # the learned log incremental loss ratio of each (line, lag)
            self.beta = nn.Parameter(start.clone())
            #: the parameters the head owns, which train at their own rate
            self.head_param_names: tuple[str, ...] = ("beta",)
        else:
            # the log factor at step m starts at init_step / m, a decaying pattern
            # whose shape is the reference's and whose magnitudes are learned
            steps = cfg.init_step / torch.arange(1, n_lag, dtype=torch.float32)
            self.phi = nn.Parameter(torch.log(torch.exp(steps) - 1.0).repeat(n_lines, 1))
            # what softplus(phi) is at that initialisation, which the anchored
            # variant measures its correction from. Not persistent: it is a function
            # of the config, so a reloaded state dict must not be able to disagree
            # with the config it is reloaded under.
            self.register_buffer("initial_logf", steps.repeat(n_lines, 1), persistent=False)
            self.head_param_names = ("phi",)
        if cfg.member == "mcl_blend":
            # the weight of the network in the member's blend with the multivariate
            # chain ladder. A buffer, so it lives in the state dict: the validation
            # check that picks the best checkpoint also fits this, and the weight
            # that goes with that checkpoint comes back with its weights.
            self.register_buffer("alpha", torch.ones(()))
        with torch.no_grad():
            self.head.weight.mul_(HEAD_INIT_SCALE)
            self.head.bias.zero_()
            if cfg.cl_anchor:
                # the anchored variant starts AT the chain ladder, not near it
                self.head.weight.zero_()

    def head_parameters(self) -> list[nn.Parameter]:
        """The parameters the head owns, which train at ``lr_phi``."""
        return [getattr(self, name) for name in self.head_param_names]

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
        fallback_lr: Tensor | None = None,
        anchor_logf: Tensor | None = None,
        anchor_start: Tensor | None = None,
        visible: Tensor | None = None,
    ) -> dict[str, Any]:
        """One forward pass over a batch of (company, accident year) examples.

        ``use_residual=False`` skips the body entirely and projects with the
        learned head parameters alone, which is how the head is checked against
        the chain ladder (``ldf``) or the pooled loss ratio (``premium_lr``) and
        how the ``factors_only`` variant runs.
        """
        batch = feat.shape[0]
        n_l, n_j = self.n_lines, self.n_lag
        use_residual = use_residual and not self.cfg.factors_only
        net = None
        attn_line: list = []
        attn_lag: list = []
        attn_ay: list = []
        if use_residual:
            h = self.inp(feat) + self.line_emb(line_ix) + self.lag_emb(lag_ix)
            # one more position: how many lags of this accident year are already
            # observed, which is the same for every token of the example
            h = h + self.nobs_emb(lk + 1).unsqueeze(-2)
            for block in self.blocks:
                h, w_line, w_lag, w_ay = block(
                    h, written, need_weights=need_weights, visible=visible
                )
                attn_line.append(w_line)
                attn_lag.append(w_lag)
                attn_ay.append(w_ay)
            # one number per token, read per (line, lag) by the head
            net = self.head(self.ln(h)).squeeze(-1).reshape(batch, n_l, n_j)

        attn = {"attn_line": attn_line, "attn_lag": attn_lag, "attn_ay": attn_ay}
        if self.cfg.head == "premium_lr":
            return self._premium_head(net, c_lk, p_lk, lk, factor_support, fallback_lr) | attn
        return (
            self._ldf_head(
                net,
                c_lk,
                p_lk,
                lk,
                factor_support,
                fallback_logf,
                anchor_logf,
                anchor_start,
            )
            | attn
        )

    def _premium_head(self, net, c_lk, p_lk, lk, factor_support, fallback_lr) -> dict[str, Any]:
        """Incremental loss ratio ``exp(min(beta + eps * net, cap))`` times premium."""
        batch = c_lk.shape[0]
        beta = self.beta.unsqueeze(0).expand(batch, self.n_lines, self.n_lag)
        logr = tlrn_head.log_ratios(beta, net, self.cfg.eps, self.cfg.lr_cap)
        if factor_support is not None:
            if fallback_lr is None:
                raise ValueError(
                    "a support mask needs fallback_lr, the pooled incremental loss ratios "
                    "observable at this example's cutoff, to substitute at the lags no "
                    "training target supervised"
                )
            logr = tlrn_head.apply_support(
                logr, factor_support, torch.log(fallback_lr).unsqueeze(0)
            )
        pred, c = tlrn_head.project_ratios(logr, c_lk, p_lk, lk)
        return {"pred": pred, "logr": logr, "C": c}

    def _ldf_head(
        self, net, c_lk, p_lk, lk, factor_support, fallback_logf, anchor_logf, anchor_start
    ) -> dict[str, Any]:
        """The published head: ``softplus(phi + eps * net)`` log factors, projected."""
        batch = c_lk.shape[0]
        n_l, n_j = self.n_lines, self.n_lag
        # the last lag has no step past it, so its column is dropped
        net = None if net is None else net[:, :, : n_j - 1]
        phi = self.phi.unsqueeze(0).expand(batch, n_l, n_j - 1)
        logf = tlrn_head.log_factors(phi, net, self.cfg.eps)

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
        return {"pred": pred, "logf": logf, "C": c}
