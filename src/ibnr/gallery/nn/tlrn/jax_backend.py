"""The JAX training backend of ``tlrn``: every member of every valuation date in one program.

This module imports jax - only import it from inside ``TLRN.fit`` when the caller asked
for ``backend="jax"`` (``ibnr.gallery`` must import without the [jax] extra, as it does
without the [nn] one; ``tests/test_gallery.py`` checks both in a clean interpreter).

WHAT IT REPLACES AND WHAT IT DOES NOT. Training only. The torch path trains one member
at a time through ``gallery/nn/_training.py::train_ensemble``; this one trains every
member of every protocol (the final fit and, under ``calibration="retrain_per_valuation"``,
each earlier valuation date) at once, as one compiled program: ``vmap`` over protocols,
``vmap`` over members, and ``lax.scan`` over epochs and over the minibatches of an
epoch. It returns what the torch path returns - per protocol, the trained members as
torch modules carrying their best-validation weights, and one history list per member
with the same ``{"member", "epoch", "train", "val"}`` records - so the selection, the
point, the calibration, ``predict`` and every read-out stay torch code and do not know
which backend trained the weights. On a TPU or a GPU the one program is what is fast;
on a CPU the torch loop is faster (see ``knowledge/findings/jax-versus-torch-on-cpu.md``),
which is why torch stays the default.

WHAT IS THE SAME AS THE TORCH PATH, ON PURPOSE. ibnr's torch code is the specification,
so this module mirrors it piece by piece: the network (``network.py``: input projection,
three 1-based embedding tables, the axial blocks with their masks and their guard, exact
GELU, LayerNorm at 1e-5, dropout on the attention weights and on each residual branch),
the two heads and the projection (``head.py``), the three-term loss on each batch's own
denominators, gradient clipping on the global norm, AdamW with torch's update and its two
learning rates, the warmup-cosine schedule, and the checkpoint rule (validate every
``check_every`` epochs and on the last, keep the best weights, stop after ``patience``
checks without improvement once ``min_epochs`` have run). Three things are taken from the
torch path rather than re-made, so the two backends agree on them exactly:

- the starting weights: member ``m`` of a protocol is built by the torch ``make_model``
  under ``torch.manual_seed(seed + 1000 * m)``, exactly as ``train_ensemble`` builds it,
  and its state dict becomes this program's starting parameters;
- the batch order and the training cutoffs: drawn from the member's own numpy generator
  in the order ``train_ensemble`` draws them (a permutation, then the epoch's cutoff);
- the shapes of everything a forward pass reads: the feature sets are the torch path's.

What differs is the dropout stream (jax's random numbers, not torch's) and the order of
floating-point additions inside a matrix product. With ``dropout = 0`` the two backends
follow the same trajectory to float32 rounding, which ``tests/test_tlrn_jax.py`` checks
epoch by epoch; with dropout on, the members are different draws of the same procedure.

HOW ONE PROGRAM SERVES PROTOCOLS OF DIFFERENT SHAPES. Two things vary between the
valuation dates of one fit, and both are made uniform by padding plus a mask:

- the number of training cutoffs (a later valuation date has more): each protocol's
  training sets are stacked along a cutoff axis padded to the longest, by repeating its
  first set. A member only ever indexes the cutoffs its own protocol drew, so a padded
  set is never read;
- the last minibatch of an epoch, which is short whenever the batch size does not
  divide the number of units (84 companies in batches of 8 leave 4): it is padded to
  the full batch size with copies of unit 0 whose rows carry a weight of zero. Every
  loss term is a sum over masked cells, so a zero-weight row adds nothing to any
  numerator or denominator and receives no gradient, and the accident-year attention
  only reads within one company, so a padded copy cannot leak into a real one. The
  check that skips a batch with nothing to score reads the real rows only, as the
  torch path does, and a skipped batch takes no optimiser step and advances no Adam
  step count.

Everything else - examples per set, tokens, features, lines, lags, the validation sets
(``val_diagonals`` of them in every protocol) - has one shape across the protocols of a
fit, and the program refuses a fit where it does not rather than padding a shape it was
not designed to pad.

EARLY STOPPING IN A BATCHED PROGRAM. A member that has stopped keeps riding along: its
updates are discarded (``where`` on the whole state), its epochs are not recorded and its
best checkpoint is frozen. Its history therefore ends where the torch loop's ``break``
would have ended it, and ``epochs_run`` in the selection table means the same thing.

PRECISION. Every matrix product asks for ``Precision.HIGHEST``. On a TPU the default
would round the operands to bfloat16, which is a different model from the float32 torch
trains; the networks are small enough that the full-precision products cost little.

THE TPU PATH IS UNMEASURED HERE: this module was written and tested on a CPU, where it
is correct and slow. The speed it is for is the companion notebook's (all 80 members of
the accident-year variant in 1,667 s on a Colab TPU), and the first TPU run is what will
say whether this program reaches it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ibnr.gallery.nn._training import warmup_cosine
from ibnr.gallery.nn.tlrn.components import attention_axes

__all__ = [
    "Spec",
    "adamw_update",
    "as_set",
    "blend_weight",
    "clip_by_global_norm",
    "forward",
    "params_from_state",
    "point_loss",
    "spec_of",
    "train_protocols",
]

#: every matrix product at full float32 precision (see the module docstring)
PRECISION = jax.lax.Precision.HIGHEST
#: torch's LayerNorm default
LN_EPS = 1e-5
#: guards a ratio whose denominator is zero, as ``head.RATIO_EPS`` does
RATIO_EPS = 1e-8
#: torch's AdamW defaults, which ``train_ensemble`` does not override
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
#: ``log(beta)`` for each beta, computed in double precision and only then made float32.
#: torch forms its bias corrections ``1 - beta**t`` in Python floats; raising a float32
#: beta to ``t`` instead rounds 0.999 to 0.99900001 first, which made every JAX Adam step
#: about 6.5e-6 (relative) off torch's. ``1 - beta**t = -expm1(t * log(beta))`` keeps it
#: to float32 rounding (``test_adam_steps_are_torch_steps_to_float32_rounding``).
ADAM_LOG_BETAS = tuple(np.float32(math.log(b)) for b in ADAM_BETAS)
#: ``torch.nn.utils.clip_grad_norm_`` adds this to the norm before dividing
CLIP_EPS = 1e-6
#: a validation check improves on the best only by more than this, as in ``train_ensemble``
IMPROVEMENT = 1e-6
#: about this many epochs per compiled call: between calls the program reports progress,
#: and a call that is too long would leave a Colab cell silent for an hour
EPOCHS_PER_CALL = 100

#: the per-example arrays of a feature set, gathered by row for a batch
ROW_KEYS = (
    "feat",
    "target",
    "target_mask",
    "premium",
    "c_lk",
    "p_lk",
    "lk",
    "written",
    "visible",
    "anchor_logf",
    "anchor_start",
)
#: the per-set arrays, shared by every example of the set
SET_KEYS = ("fallback_logf", "fallback_lr")
#: the token layout, one per fit rather than one per set
_LAYOUT_KEYS = ("line_ix", "lag_ix")
_INT_KEYS = ("lk",)
_BOOL_KEYS = ("written", "visible")


class Spec(NamedTuple):
    """The static shape of one network and one objective: everything ``jit`` compiles on."""

    n_l: int
    n_d: int
    n_w: int
    d_model: int
    n_heads: int
    n_layers: int
    axes: tuple[str, ...]
    observed: bool  # mask="observed_cells"
    head: str
    eps: float
    lr_cap: float
    cl_anchor: bool
    anchor_width: float
    factors_only: bool
    dropout: float
    support: bool  # tail_policy="observed_cl": a factor support mask is applied
    blend: bool  # member="mcl_blend"
    init_step: float
    w_pe: float
    w_mse: float
    mse_scale: float


def spec_of(cfg: Any, *, n_l: int, n_d: int, n_w: int) -> Spec:
    """The :class:`Spec` of a ``TLRNConfig`` on a contract of these sizes."""
    return Spec(
        n_l=int(n_l),
        n_d=int(n_d),
        n_w=int(n_w),
        d_model=int(cfg.d_model),
        n_heads=int(cfg.n_heads),
        n_layers=int(cfg.n_layers),
        axes=tuple(attention_axes(cfg)),
        observed=cfg.mask == "observed_cells",
        head=cfg.head,
        eps=float(cfg.eps),
        lr_cap=float(cfg.lr_cap),
        cl_anchor=bool(cfg.cl_anchor),
        anchor_width=float(cfg.anchor_width),
        factors_only=bool(cfg.factors_only),
        dropout=float(cfg.dropout),
        support=cfg.tail_policy == "observed_cl",
        blend=cfg.member == "mcl_blend",
        init_step=float(cfg.init_step),
        w_pe=float(cfg.w_pe),
        w_mse=float(cfg.w_mse),
        mse_scale=float(cfg.mse_scale),
    )


# -- parameters ------------------------------------------------------------------------


def params_from_state(state: Mapping[str, Any], names) -> dict[str, jnp.ndarray]:
    """The named trainable tensors of a torch state dict, as float32 jax arrays."""
    return {n: jnp.asarray(np.asarray(_numpy(state[n]), dtype=np.float32)) for n in names}


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def head_names(spec: Spec) -> tuple[str, ...]:
    """The parameters the head owns, which train at ``lr_phi`` (``head_param_names``)."""
    return ("beta",) if spec.head == "premium_lr" else ("phi",)


def unused_names(spec: Spec, names) -> frozenset[str]:
    """Parameters no forward pass reads, which torch never hands a gradient.

    AdamW skips a parameter whose gradient is ``None``, weight decay included, so these
    must not decay here either; their zero gradient already leaves Adam's update at zero.
    """
    if spec.factors_only:
        return frozenset(n for n in names if n not in head_names(spec))
    if "line" not in spec.axes:
        return frozenset(n for n in names if ".attn_line." in n or ".ln1." in n)
    return frozenset()


# -- feature sets ----------------------------------------------------------------------


def as_set(features: Mapping[str, Any], *, with_mcl: bool = False) -> dict[str, jnp.ndarray]:
    """One numpy feature set (``tlrn_features``) as the arrays :func:`forward` reads."""
    out = {}
    for k in (*ROW_KEYS, *SET_KEYS):
        out[k] = _cast(k, features[k])
    out["line_ix"] = jnp.asarray(features["line_ix"], dtype=jnp.int32)
    out["lag_ix"] = jnp.asarray(features["lag_ix"], dtype=jnp.int32)
    if with_mcl:
        out["mcl_pred"] = jnp.asarray(features["mcl_pred"], dtype=jnp.float32)
    return out


def _cast(key: str, value):
    if key in _INT_KEYS:
        return jnp.asarray(value, dtype=jnp.int32)
    if key in _BOOL_KEYS:
        return jnp.asarray(np.asarray(value).astype(bool))
    return jnp.asarray(value, dtype=jnp.float32)


# -- the network (network.py) ------------------------------------------------------------


def _linear(x, weight, bias=None):
    out = jnp.einsum("...i,oi->...o", x, weight, precision=PRECISION)
    return out if bias is None else out + bias


def _layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + LN_EPS) * weight + bias


def _dropout(x, key, rate: float, train: bool):
    """torch's dropout: zero with probability ``rate``, scale the rest by ``1 / (1 - rate)``."""
    if not train or rate == 0.0:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def _attend(p, prefix, h, readable, *, guard, spec, train, key):
    """``network.attend`` over ``nn.MultiheadAttention``'s packed weights.

    ``readable`` (N, S) marks the keys a sequence may read, or is None for no mask. With
    ``guard`` a sequence with no readable key lets its first key through so the softmax is
    finite, and its output is then multiplied by zero.
    """
    n, s, d = h.shape
    heads = spec.n_heads
    hd = d // heads
    qkv = _linear(h, p[f"{prefix}.in_proj_weight"], p[f"{prefix}.in_proj_bias"])
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(n, s, heads, hd)
    k = k.reshape(n, s, heads, hd)
    v = v.reshape(n, s, heads, hd)
    scores = jnp.einsum("nshd,nthd->nhst", q, k, precision=PRECISION) / math.sqrt(hd)
    any_key = None
    if readable is not None:
        if guard:
            any_key = readable.any(axis=1)
            readable = readable.at[:, 0].set(readable[:, 0] | ~any_key)
        scores = jnp.where(readable[:, None, None, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    weights = _dropout(weights, key, spec.dropout, train)
    out = jnp.einsum("nhst,nthd->nshd", weights, v, precision=PRECISION).reshape(n, s, d)
    out = _linear(out, p[f"{prefix}.out_proj.weight"], p[f"{prefix}.out_proj.bias"])
    if any_key is not None:
        out = out * any_key[:, None, None].astype(out.dtype)
    return out


def _block(p, i, x, written, visible, *, spec, train, keys):
    """``AxialBlock.forward``: lines, lags, accident years, then the feed-forward."""
    pre = f"blocks.{i}"
    batch, n_tok, d = x.shape
    n_l, n_j = spec.n_l, spec.n_d
    vis = visible.reshape(batch, n_l, n_j) if spec.observed else None

    if "line" in spec.axes:
        h = _layer_norm(x, p[f"{pre}.ln1.weight"], p[f"{pre}.ln1.bias"])
        h = h.reshape(batch, n_l, n_j, d).transpose(0, 2, 1, 3).reshape(batch * n_j, n_l, d)
        if spec.observed:
            readable = vis.transpose(0, 2, 1).reshape(batch * n_j, n_l)
        else:
            readable = jnp.broadcast_to(written[:, None, :], (batch, n_j, n_l)).reshape(
                batch * n_j, n_l
            )
        a = _attend(
            p,
            f"{pre}.attn_line",
            h,
            readable,
            guard=spec.observed,
            spec=spec,
            train=train,
            key=keys[0],
        )
        a = a.reshape(batch, n_j, n_l, d).transpose(0, 2, 1, 3).reshape(batch, n_l * n_j, d)
        x = x + _dropout(a, keys[1], spec.dropout, train)

    h = _layer_norm(x, p[f"{pre}.ln2.weight"], p[f"{pre}.ln2.bias"]).reshape(batch * n_l, n_j, d)
    readable = vis.reshape(batch * n_l, n_j) if spec.observed else None
    a = _attend(
        p,
        f"{pre}.attn_lag",
        h,
        readable,
        guard=spec.observed,
        spec=spec,
        train=train,
        key=keys[2],
    )
    x = x + _dropout(a.reshape(batch, n_l * n_j, d), keys[3], spec.dropout, train)

    if "ay" in spec.axes:
        n_w = spec.n_w
        n_c = batch // n_w
        h = _layer_norm(x, p[f"{pre}.ln_ay.weight"], p[f"{pre}.ln_ay.bias"])
        h = h.reshape(n_c, n_w, n_tok, d).transpose(0, 2, 1, 3).reshape(n_c * n_tok, n_w, d)
        if spec.observed:
            token_ok = vis.reshape(batch, n_tok)
        else:
            token_ok = jnp.broadcast_to(written[:, :, None], (batch, n_l, n_j)).reshape(
                batch, n_tok
            )
        readable = token_ok.reshape(n_c, n_w, n_tok).transpose(0, 2, 1).reshape(n_c * n_tok, n_w)
        a = _attend(
            p, f"{pre}.attn_ay", h, readable, guard=True, spec=spec, train=train, key=keys[4]
        )
        a = a.reshape(n_c, n_tok, n_w, d).transpose(0, 2, 1, 3).reshape(batch, n_tok, d)
        x = x + _dropout(a, keys[5], spec.dropout, train)

    h = _layer_norm(x, p[f"{pre}.ln3.weight"], p[f"{pre}.ln3.bias"])
    h = _linear(h, p[f"{pre}.ff.0.weight"], p[f"{pre}.ff.0.bias"])
    h = jax.nn.gelu(h, approximate=False)  # torch's nn.GELU() is the exact one
    h = _dropout(h, keys[6], spec.dropout, train)
    h = _linear(h, p[f"{pre}.ff.3.weight"], p[f"{pre}.ff.3.bias"])
    return x + h


#: dropout sites per block (two per attention, one in the feed-forward)
_KEYS_PER_BLOCK = 7


def forward(p, s, spec: Spec, *, train: bool = False, key=None, support=None) -> dict:
    """``TLRNNetwork.forward`` on one feature set (or a batch of its rows).

    ``s`` carries the arrays :func:`as_set` builds, ``support`` the (n_l, n_d - 1) factor
    support mask when ``spec.support``. Returns ``pred`` (B, n_l * n_d) and ``C``
    (B, n_l, n_d), the network alone: the blend with the chain ladder is a validation
    read-out, as ``forward(..., blend=False)`` is in ``model.py``'s training loss.
    """
    batch = s["feat"].shape[0]
    n_l, n_j = spec.n_l, spec.n_d
    net = None
    if not spec.factors_only:
        if train and spec.dropout > 0.0:
            keys = jax.random.split(key, _KEYS_PER_BLOCK * spec.n_layers)
        else:
            keys = [None] * (_KEYS_PER_BLOCK * spec.n_layers)
        h = _linear(s["feat"], p["inp.weight"], p["inp.bias"])
        h = h + p["line_emb.weight"][s["line_ix"]] + p["lag_emb.weight"][s["lag_ix"]]
        h = h + p["nobs_emb.weight"][s["lk"] + 1][:, None, :]
        for i in range(spec.n_layers):
            block_keys = keys[_KEYS_PER_BLOCK * i : _KEYS_PER_BLOCK * (i + 1)]
            h = _block(p, i, h, s["written"], s["visible"], spec=spec, train=train, keys=block_keys)
        h = _layer_norm(h, p["ln.weight"], p["ln.bias"])
        net = _linear(h, p["head.weight"], p["head.bias"])[..., 0].reshape(batch, n_l, n_j)
    if spec.head == "premium_lr":
        return _premium_head(p, net, s, spec, support)
    return _ldf_head(p, net, s, spec, support)


def _premium_head(p, net, s, spec, support):
    beta = jnp.broadcast_to(p["beta"][None], (s["c_lk"].shape[0], spec.n_l, spec.n_d))
    raw = beta if net is None else beta + spec.eps * net
    # torch.clamp(max=cap): the gradient passes where raw <= cap and stops above it
    logr = jnp.where(raw <= spec.lr_cap, raw, spec.lr_cap)
    if spec.support:
        fallback = jnp.log(s["fallback_lr"])[:, 1:][None]
        tail = jnp.where(support[None], logr[:, :, 1:], fallback)
        logr = jnp.concatenate([logr[:, :, :1], tail], axis=2)
    pred, c = project_ratios(logr, s["c_lk"], s["p_lk"], s["lk"])
    return {"pred": pred, "C": c}


def _initial_logf(spec):
    steps = spec.init_step / np.arange(1, spec.n_d, dtype=np.float32)
    return jnp.asarray(np.tile(steps.astype(np.float32), (spec.n_l, 1)))


def _ldf_head(p, net, s, spec, support):
    batch = s["c_lk"].shape[0]
    n_l, n_j = spec.n_l, spec.n_d
    net = None if net is None else net[:, :, : n_j - 1]
    phi = jnp.broadcast_to(p["phi"][None], (batch, n_l, n_j - 1))
    logf = jax.nn.softplus(phi if net is None else phi + spec.eps * net)
    if spec.cl_anchor:
        initial = _initial_logf(spec)[None]
        logf = s["anchor_logf"] + spec.anchor_width * jnp.tanh((logf - initial) / spec.anchor_width)
    if spec.support:
        fallback = s["anchor_logf"] if spec.cl_anchor else s["fallback_logf"][None]
        logf = jnp.where(support[None], logf, jnp.broadcast_to(fallback, logf.shape))
    pred, c = project(
        logf,
        s["c_lk"],
        s["p_lk"],
        s["lk"],
        anchor_start=s["anchor_start"] if spec.cl_anchor else None,
    )
    return {"pred": pred, "C": c}


def project(logf, c_lk, p_lk, lk, *, anchor_start=None):
    """``head.project``: cumulative forward from each origin's latest visible lag."""
    batch, n_l, n_steps = logf.shape
    n_d = n_steps + 1
    zero = jnp.zeros((batch, n_l, 1), logf.dtype)
    cumulative_logf = jnp.concatenate([zero, jnp.cumsum(logf, axis=2)], axis=2)
    index = jnp.broadcast_to((lk - 1)[:, None, None], (batch, n_l, 1))
    at_lk = jnp.take_along_axis(cumulative_logf, index, axis=2)
    lag = jnp.arange(n_d)[None, None, :]
    ahead = (lag >= lk[:, None, None]).astype(logf.dtype)
    if anchor_start is None:
        grown = jnp.exp(jnp.log(c_lk)[:, :, None] + cumulative_logf - at_lk)
        c = c_lk[:, :, None] * (1 - ahead) + grown * ahead
    else:
        grown = jnp.exp(cumulative_logf - at_lk)
        c = anchor_start[:, :, None] * ((1 - ahead) + grown * ahead)
    previous = jnp.concatenate([jnp.zeros_like(c[:, :, :1]), c[:, :, :-1]], axis=2)
    pred = ((c - previous) / p_lk[:, :, None]).reshape(batch, n_l * n_d)
    return pred, c


def project_ratios(logr, c_lk, p_lk, lk):
    """``head.project_ratios``: premium times an incremental loss ratio, forward from ``lk``."""
    batch, n_l, n_d = logr.shape
    lag = jnp.arange(n_d)[None, None, :]
    ahead = lag >= lk[:, None, None]
    ratio = jnp.exp(logr)
    increment = ratio * p_lk[:, :, None] * ahead.astype(logr.dtype)
    c = c_lk[:, :, None] + jnp.cumsum(increment, axis=2)
    previous = jnp.concatenate([jnp.zeros_like(c[:, :, :1]), c[:, :, :-1]], axis=2)
    held = (c - previous) / p_lk[:, :, None]
    pred = jnp.where(ahead, ratio, held).reshape(batch, n_l * n_d)
    return pred, c


# -- the objective (head.py) -------------------------------------------------------------


def point_loss(pred, targ, mask, prem, spec: Spec):
    """``head.point_loss``: the three terms on this batch's own denominators."""
    n_l, n_d = spec.n_l, spec.n_d
    error = (pred - targ) * mask * prem
    actual = targ * mask * prem
    by_group = error.reshape(-1, n_l, n_d).sum(2)
    actual_group = actual.reshape(-1, n_l, n_d).sum(2)
    ay_line = jnp.abs(by_group).sum() / (jnp.abs(actual_group).sum() + RATIO_EPS)
    pool = jnp.abs(error.sum() / (jnp.abs(actual.sum()) + RATIO_EPS))
    n = mask.sum()
    mse = jnp.where(n > 0, (((pred - targ) ** 2) * mask).sum() / jnp.where(n > 0, n, 1.0), 0.0)
    return ay_line + spec.w_pe * pool + spec.w_mse * mse / spec.mse_scale


# -- validation (model.py's _ay_line_ape and _blend_ay_line_ape) ---------------------------


def blend_weight(a, b):
    """``blend.blend_weight`` in jax: the weighted median of ``-a / b``, clipped to [0, 1]."""
    keep = jnp.isfinite(a) & jnp.isfinite(b) & (b != 0)
    t = jnp.where(keep, -a / jnp.where(keep, b, 1.0), jnp.inf)
    weight = jnp.where(keep, jnp.abs(b), 0.0)
    order = jnp.argsort(t, stable=True)
    t, weight = t[order], weight[order]
    reached = jnp.cumsum(weight)
    index = jnp.searchsorted(reached, 0.5 * reached[-1], side="left")
    median = t[jnp.minimum(index, t.shape[0] - 1)]
    return jnp.where(keep.any(), jnp.clip(median, 0.0, 1.0), 0.0)


def _validate(p, val, n_sets, spec, support):
    """``(score, alpha)`` of one member on its protocol's validation sets, in eval mode."""
    groups = []
    for v in range(n_sets):
        s = {k: x[v] for k, x in val.items() if k not in _LAYOUT_KEYS}
        s["line_ix"], s["lag_ix"] = val["line_ix"], val["lag_ix"]
        pred = forward(p, s, spec, train=False, support=support)["pred"]
        weight = s["target_mask"] * s["premium"]

        def by_group(cells, weight=weight):
            return (cells * weight).reshape(-1, spec.n_l, spec.n_d).sum(2).reshape(-1)

        if spec.blend:
            groups.append(
                (
                    by_group(s["mcl_pred"] - s["target"]),
                    by_group(pred - s["mcl_pred"]),
                    by_group(s["target"]),
                )
            )
        else:
            groups.append((by_group(pred - s["target"]), None, by_group(s["target"])))
    actual = jnp.concatenate([g[2] for g in groups])
    denominator = jnp.maximum(jnp.abs(actual).sum(), 1e-8)
    error = jnp.concatenate([g[0] for g in groups])
    if not spec.blend:
        return jnp.abs(error).sum() / denominator, jnp.float32(1.0)
    gap = jnp.concatenate([g[1] for g in groups])
    alpha = blend_weight(error, gap)
    return jnp.abs(error + alpha * gap).sum() / denominator, alpha


# -- the optimiser (train_ensemble's clip_grad_norm_ and AdamW) ----------------------------


def clip_by_global_norm(grads: dict, max_norm: float) -> dict:
    """``torch.nn.utils.clip_grad_norm_``: every gradient times ``min(1, max / (norm + 1e-6))``."""
    total = jnp.sqrt(sum(jnp.sum(g * g) for g in grads.values()))
    scale = jnp.minimum(max_norm / (total + CLIP_EPS), 1.0)
    return {n: g * scale for n, g in grads.items()}


def adamw_update(params, m, v, t, grads, *, base_lr, mult, weight_decay, decays):
    """One ``torch.optim.AdamW`` step: decoupled decay, then the bias-corrected Adam step.

    ``base_lr`` maps each parameter to its group's rate and ``mult`` is the schedule's
    multiplier for this epoch, so a parameter steps at ``base_lr * mult`` as torch's
    ``group["lr"] = group["base_lr"] * mult`` makes it. ``t`` is the step count before
    this step. Returns ``(params, m, v, t + 1)``.
    """
    b1, b2 = ADAM_BETAS
    log_b1, log_b2 = ADAM_LOG_BETAS
    t1 = t + 1.0
    bias1 = -jnp.expm1(t1 * log_b1)
    bias2 = -jnp.expm1(t1 * log_b2)
    new_p, new_m, new_v = {}, {}, {}
    for n in params:
        g = grads[n]
        lr = base_lr[n] * mult
        q = params[n] * (1.0 - lr * weight_decay) if (decays[n] and weight_decay) else params[n]
        new_m[n] = b1 * m[n] + (1.0 - b1) * g
        new_v[n] = b2 * v[n] + (1.0 - b2) * g * g
        denom = jnp.sqrt(new_v[n]) / jnp.sqrt(bias2) + ADAM_EPS
        new_p[n] = q - (lr / bias1) * new_m[n] / denom
    return new_p, new_m, new_v, t1


# -- the training program -----------------------------------------------------------------


def _program(spec, cfg, names, *, n_val, rows_per_unit, segment_epochs):
    """The jitted call that trains every member through some whole segments.

    A segment is ``segment_epochs`` epochs followed by one validation check, which is the
    torch loop's rhythm: a check every ``check_every`` epochs and one on the last epoch, so
    a budget that ``check_every`` does not divide ends in one shorter segment.
    """
    heads = set(head_names(spec))
    base_lr = {n: float(cfg.lr_phi if n in heads else cfg.lr) for n in names}
    decays = {n: n not in unused_names(spec, names) for n in names}
    wd = float(cfg.weight_decay)
    clip = float(cfg.grad_clip)
    n_w = spec.n_w

    def rows_of(units, real):
        if rows_per_unit == 1:
            return units, real
        rows = (units[:, None] * n_w + jnp.arange(n_w)[None, :]).reshape(-1)
        return rows, jnp.repeat(real, n_w)

    def batch_step(carry, inputs, train, support, k):
        params, m, v, t, epoch_loss, n_batches = carry
        units, real, key, mult = inputs
        rows, weight = rows_of(units, real)
        s = {name: train[name][k, rows] for name in ROW_KEYS}
        for name in SET_KEYS:
            s[name] = train[name][k]
        s["line_ix"], s["lag_ix"] = train["line_ix"], train["lag_ix"]
        mask = s["target_mask"] * weight[:, None]
        has_cells = mask.sum() > 0  # the torch loop skips a batch with nothing to score

        def loss_of(q):
            pred = forward(q, s, spec, train=True, key=key, support=support)["pred"]
            return point_loss(pred, s["target"], mask, s["premium"], spec)

        loss, grads = jax.value_and_grad(loss_of)(params)
        grads = clip_by_global_norm(grads, clip)
        stepped_p, stepped_m, stepped_v, t1 = adamw_update(
            params, m, v, t, grads, base_lr=base_lr, mult=mult, weight_decay=wd, decays=decays
        )

        def taken(new, old):
            return {n: jnp.where(has_cells, new[n], old[n]) for n in names}

        return (
            taken(stepped_p, params),
            taken(stepped_m, m),
            taken(stepped_v, v),
            jnp.where(has_cells, t1, t),
            epoch_loss + jnp.where(has_cells, loss, 0.0),
            n_batches + has_cells.astype(jnp.float32),
        ), None

    def member_segment(state, data, units, real, k, member_key, mult, epochs):
        """One member through one segment: ``segment_epochs`` epochs, then a check."""
        train, val, support = data["train"], data["val"], data.get("support")
        running = ~state["done"]

        def one_epoch(carry, inputs):
            params, m, v, t = carry
            units_e, real_e, k_e, mult_e, epoch_e = inputs
            n_batch = units_e.shape[0]
            keys = jax.vmap(lambda b: jax.random.fold_in(member_key, epoch_e * n_batch + b))(
                jnp.arange(n_batch)
            )
            (params, m, v, t, total, count), _ = jax.lax.scan(
                lambda c, x: batch_step(c, (*x, mult_e), train, support, k_e),
                (params, m, v, t, jnp.float32(0.0), jnp.float32(0.0)),
                (units_e, real_e, keys),
            )
            return (params, m, v, t), total / jnp.maximum(count, 1.0)

        (params, m, v, t), losses = jax.lax.scan(
            one_epoch,
            (state["params"], state["m"], state["v"], state["t"]),
            (units, real, k, mult, epochs),
        )
        score, alpha = _validate(params, val, n_val, spec, support)
        better = running & (score < state["best_val"] - IMPROVEMENT)
        patience = jnp.where(better, cfg.patience, state["patience"] - running.astype(jnp.int32))
        last_epoch = epochs[-1] + 1  # 1-based count of epochs run once this segment ends
        stop = running & ~better & (patience <= 0) & (last_epoch >= cfg.min_epochs)

        def keep_running(new, old):
            return jnp.where(running, new, old)

        new_state = {
            "params": {n: keep_running(params[n], state["params"][n]) for n in names},
            "m": {n: keep_running(m[n], state["m"][n]) for n in names},
            "v": {n: keep_running(v[n], state["v"][n]) for n in names},
            "t": keep_running(t, state["t"]),
            "alpha": keep_running(alpha, state["alpha"]),
            "best": {n: jnp.where(better, params[n], state["best"][n]) for n in names},
            "best_alpha": jnp.where(better, alpha, state["best_alpha"]),
            "best_val": jnp.where(better, score, state["best_val"]),
            "has_best": state["has_best"] | better,
            "patience": patience,
            "done": state["done"] | stop,
        }
        return new_state, (losses, score, running)

    members = jax.vmap(
        member_segment, in_axes=(0, None, 0, 0, 0, 0, None, None)
    )  # over the members of one protocol
    protocols = jax.vmap(members, in_axes=(0, 0, 0, 0, 0, 0, None, None))  # over protocols

    def call(state, data, units, real, k, member_keys, mult, epochs):
        def one_segment(s, x):
            u, r, kk, mu, ep = x
            return protocols(s, data, u, r, kk, member_keys, mu, ep)

        return jax.lax.scan(one_segment, state, (units, real, k, mult, epochs))

    return jax.jit(call)


def _stack_sets(sets_per_protocol: list[list[dict]], keys, *, pad_to: int | None = None):
    """``{key: (P, n_sets, ...)}``, each protocol's list padded by repeating its first set."""
    out = {}
    for key in keys:
        per = []
        for sets in sets_per_protocol:
            arrays = [np.asarray(s[key]) for s in sets]
            if pad_to is not None:
                arrays += [arrays[0]] * (pad_to - len(arrays))
            per.append(np.stack(arrays))
        out[key] = _cast(key, np.stack(per))
    return out


def _check_uniform(preps: list[dict]) -> None:
    """Every protocol of a fit must share the shapes the program does not pad."""
    first = next(iter(preps[0]["train_sets"].values()))
    for i, prep in enumerate(preps):
        for s in (*prep["train_sets"].values(), *prep["val_sets"]):
            for key in ("feat", "target", "c_lk", "written", "anchor_logf"):
                if np.shape(s[key]) != np.shape(first[key]):
                    raise ValueError(
                        f"backend='jax' trains every protocol in one program, and protocol {i}'s "
                        f"{key!r} is {np.shape(s[key])} where the first is {np.shape(first[key])}"
                    )
            if not (
                np.array_equal(s["line_ix"], first["line_ix"])
                and np.array_equal(s["lag_ix"], first["lag_ix"])
            ):
                raise ValueError("backend='jax' needs one token layout across every protocol")
        if prep["n_units"] != preps[0]["n_units"]:
            raise ValueError("backend='jax' needs one number of batch units across every protocol")
        if len(prep["val_sets"]) != len(preps[0]["val_sets"]):
            raise ValueError("backend='jax' needs one number of validation sets per protocol")


def _member_seed(seed: int | None, member: int) -> int | None:
    return None if seed is None else seed + 1000 * member


def train_protocols(preps: list[dict], cfg, torch, *, epochs_per_call: int = EPOCHS_PER_CALL):
    """Train every member of every prepared protocol in one program.

    ``preps`` are ``TLRN._prepare_protocol`` results. Returns ``(models, history)`` per
    protocol, each in member order, exactly the shape ``train_ensemble(keep=None)`` gives:
    the models are torch modules in eval mode carrying their best-validation weights.
    """
    if cfg.cutoff_sampling != "per_epoch":
        raise ValueError(
            "tlrn needs one cutoff for the whole batch, because a batch is scored against one "
            "feature set: set cutoff_sampling='per_epoch'"
        )
    _check_uniform(preps)
    first = next(iter(preps[0]["train_sets"].values()))
    n_ex, n_tok = first["target"].shape
    n_l = first["c_lk"].shape[1]
    n_d = n_tok // n_l
    n_w = n_ex // int(np.max(first["example_company"]) + 1)
    spec = spec_of(cfg, n_l=n_l, n_d=n_d, n_w=n_w)
    if "line" in spec.axes and not spec.observed:
        for prep in preps:
            for s in prep["train_sets"].values():
                if not bool(np.asarray(s["written"]).any(axis=1).all()):
                    raise ValueError(
                        "an example has no written line, so every key of the cross-line "
                        "attention is padded and its output is NaN at every token. Drop the "
                        "company from the cohort rather than carrying an example with no data"
                    )

    n_p, n_m = len(preps), int(cfg.ensemble_size)
    cutoffs = [sorted(prep["train_sets"]) for prep in preps]
    n_k = max(len(c) for c in cutoffs)
    data = {
        "train": _stack_sets(
            [[prep["train_sets"][k] for k in c] for prep, c in zip(preps, cutoffs, strict=True)],
            (*ROW_KEYS, *SET_KEYS),
            pad_to=n_k,
        ),
        "val": _stack_sets(
            [prep["val_sets"] for prep in preps],
            (*ROW_KEYS, *SET_KEYS, *(("mcl_pred",) if spec.blend else ())),
        ),
    }
    for part in ("train", "val"):
        data[part]["line_ix"] = jnp.asarray(np.tile(first["line_ix"], (n_p, 1)), dtype=jnp.int32)
        data[part]["lag_ix"] = jnp.asarray(np.tile(first["lag_ix"], (n_p, 1)), dtype=jnp.int32)
    if spec.support:
        data["support"] = jnp.asarray(np.stack([np.asarray(p["support_np"]) for p in preps]))
    n_val = len(preps[0]["val_sets"])

    # the starting weights, the batch order and the cutoffs: the torch path's own
    names: list[str] | None = None
    start: list[list[dict]] = []
    rngs: list[list[np.random.Generator]] = []
    keys = np.zeros((n_p, n_m, 2), dtype=np.uint32)
    for p, prep in enumerate(preps):
        start.append([])
        rngs.append([])
        for m in range(n_m):
            member_seed = _member_seed(prep["seed"], m)
            if member_seed is not None:
                torch.manual_seed(member_seed)
            model = prep["make_model"]()
            if names is None:
                names = [n for n, _ in model.named_parameters()]
            start[p].append({k: _numpy(v) for k, v in model.state_dict().items()})
            rngs[p].append(np.random.default_rng(member_seed))
            entropy = np.random.SeedSequence(member_seed).generate_state(2)
            keys[p, m] = entropy
    assert names is not None
    params = {n: jnp.asarray(np.stack([[s[n] for s in row] for row in start])) for n in names}
    zeros = {n: jnp.zeros_like(x) for n, x in params.items()}
    alpha0 = np.array([[float(s.get("alpha", 1.0)) for s in row] for row in start], np.float32)
    state = {
        "params": params,
        "m": zeros,
        "v": dict(zeros),
        "t": jnp.zeros((n_p, n_m), jnp.float32),
        "alpha": jnp.asarray(alpha0),
        "best": dict(params),
        "best_alpha": jnp.asarray(alpha0),
        "best_val": jnp.full((n_p, n_m), jnp.inf, jnp.float32),
        "has_best": jnp.zeros((n_p, n_m), bool),
        "patience": jnp.full((n_p, n_m), cfg.patience, jnp.int32),
        "done": jnp.zeros((n_p, n_m), bool),
    }
    member_keys = jax.vmap(jax.vmap(jax.random.wrap_key_data))(jnp.asarray(keys))

    n_units = preps[0]["n_units"]
    batch = int(cfg.batch_size)
    n_batches = -(-n_units // batch)
    rows_per_unit = n_w if cfg.batch_unit == "company" else 1
    schedule = warmup_cosine(cfg.max_epochs, cfg.warmup)
    every = int(cfg.check_every)
    lengths = [every] * (cfg.max_epochs // every)
    if cfg.max_epochs % every:
        lengths.append(cfg.max_epochs % every)

    # draws one epoch for one member from its own generator, in train_ensemble's order
    def draw(p: int, m: int) -> tuple[np.ndarray, np.ndarray, int]:
        rng = rngs[p][m]
        perm = rng.permutation(n_units)
        cutoff = int(rng.integers(cfg.min_cutoff, preps[p]["train_end"]))
        units = np.zeros(n_batches * batch, dtype=np.int32)
        real = np.zeros(n_batches * batch, dtype=np.float32)
        units[:n_units] = perm
        real[:n_units] = 1.0
        return (
            units.reshape(n_batches, batch),
            real.reshape(n_batches, batch),
            cutoffs[p].index(cutoff),
        )

    programs: dict[int, Any] = {}
    losses: list[np.ndarray] = []  # (P, M, epochs) per segment
    scores: list[np.ndarray] = []
    ran: list[np.ndarray] = []
    seg_per_call = max(1, epochs_per_call // every)
    done_epochs = 0
    started = time.perf_counter()
    show = bool(preps[0].get("show_progress"))
    i = 0
    while i < len(lengths):
        length = lengths[i]
        group = [length]
        while len(group) < seg_per_call and i + len(group) < len(lengths):
            if lengths[i + len(group)] != length:
                break
            group.append(length)
        g = len(group)
        units = np.zeros((g, n_p, n_m, length, n_batches, batch), np.int32)
        real = np.zeros((g, n_p, n_m, length, n_batches, batch), np.float32)
        kidx = np.zeros((g, n_p, n_m, length), np.int32)
        epochs = done_epochs + np.arange(g * length, dtype=np.int32).reshape(g, length)
        mult = np.array([[schedule(int(e) + 1) for e in row] for row in epochs], np.float32)
        for j in range(g):
            for e in range(length):
                for p in range(n_p):
                    for m in range(n_m):
                        units[j, p, m, e], real[j, p, m, e], kidx[j, p, m, e] = draw(p, m)
        if length not in programs:
            programs[length] = _program(
                spec,
                cfg,
                names,
                n_val=n_val,
                rows_per_unit=rows_per_unit,
                segment_epochs=length,
            )
        state, (seg_losses, seg_scores, seg_ran) = programs[length](
            state,
            data,
            jnp.asarray(units),
            jnp.asarray(real),
            jnp.asarray(kidx),
            member_keys,
            jnp.asarray(mult),
            jnp.asarray(epochs),
        )
        seg_losses = np.asarray(seg_losses)  # (g, P, M, length)
        seg_scores = np.asarray(seg_scores)  # (g, P, M)
        seg_ran = np.asarray(seg_ran)
        for j in range(g):
            losses.append(seg_losses[j])
            scores.append(seg_scores[j])
            ran.append(seg_ran[j])
        done_epochs += g * length
        i += g
        if show:
            live = seg_ran[-1]
            median = float(np.median(seg_scores[-1][live])) if live.any() else math.nan
            print(
                f"jax: epoch {done_epochs}/{cfg.max_epochs}, "
                f"{time.perf_counter() - started:.0f}s, members still training "
                f"{int(live.sum())}/{live.size}, median validation {median:.4f}",
                flush=True,
            )

    final = {
        n: np.asarray(
            jnp.where(
                _expand(state["has_best"], state["best"][n]), state["best"][n], state["params"][n]
            )
        )
        for n in names
    }
    alpha = np.asarray(jnp.where(state["has_best"], state["best_alpha"], state["alpha"]))
    out: list[tuple[list, list]] = []
    for p, prep in enumerate(preps):
        models, histories = [], []
        for m in range(n_m):
            model = prep["make_model"]()
            weights = {n: torch.from_numpy(np.array(final[n][p, m])) for n in names}
            if "alpha" in start[p][m]:
                weights["alpha"] = torch.tensor(float(alpha[p, m]), dtype=torch.float32)
            model.load_state_dict(weights)
            model.eval()
            models.append(model)
            histories.append(_history(m, lengths, losses, scores, ran, p))
        out.append((models, histories))
    return out


def _expand(flag, like):
    return flag.reshape(flag.shape + (1,) * (like.ndim - flag.ndim))


def _history(member: int, lengths, losses, scores, ran, p: int) -> list[dict]:
    """One member's ``train_ensemble`` history: a record per epoch it ran."""
    records: list[dict] = []
    epoch = 0
    for length, seg_losses, seg_scores, seg_ran in zip(lengths, losses, scores, ran, strict=True):
        if not bool(seg_ran[p, member]):
            break
        for e in range(length):
            last = e == length - 1
            records.append(
                {
                    "member": member,
                    "epoch": epoch,
                    "train": float(seg_losses[p, member, e]),
                    "val": float(seg_scores[p, member]) if last else math.nan,
                }
            )
            epoch += 1
    return records
