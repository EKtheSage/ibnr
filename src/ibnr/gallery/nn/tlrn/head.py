"""The positive development-factor head, its projection and the point objective.

This module imports torch - only import it from inside the entry's fit/predict
paths (``ibnr.gallery`` must import without the [nn] extra; see CLAUDE.md's
"torch must never be imported at module level" rule). Free functions over plain
tensors, like ``transformer/network.py``'s loss pair, so each piece can be
tested against a closed form on hand-written numbers.

WHAT THE HEAD IS. The network does not predict a cell. It predicts a log
development factor per (line, step), and the cells follow from projecting the
cumulative forward with those factors. So the model cannot produce a negative
factor, cannot disagree with itself about two cells of one origin, and reduces
exactly to the chain ladder when the factors are the chain ladder's - which is
the check ``tests/test_tlrn.py`` runs before anything else, because a head that
fails it is mis-specified and no amount of training fixes that.

The pieces, in the order a forward pass uses them:

- :func:`log_factors` - ``softplus(phi + eps * net)``. ``phi`` is the learned
  per (line, step) parameter and ``net`` the network's per-cell correction to
  it, scaled by ``eps``. The softplus is what keeps every factor above 1, so a
  cumulative can only grow. Paid recoveries are therefore outside what this head
  can express, which the card discloses.
- :func:`anchor` - the switchable variant: instead of learning a factor
  outright, start at the company's own chain ladder factors and learn a bounded
  correction, ``anchor + width * tanh((logf - initial) / width)``. At the
  initialisation ``tanh(0) = 0``, so the model starts exactly at the chain
  ladder rather than arriving there.
- :func:`apply_support` - a step no training target ever projected across has
  not been estimated; the network's value there is whatever the optimiser left.
  Those steps take the chain ladder factors observable at the example's own
  cutoff instead.
- :func:`project` - cumulative forward from each origin's latest visible lag,
  then differenced back into the increment ratios the loss compares.

THE OBJECTIVE. Three terms, from the study's checkpoint protocol:
:func:`ay_line_ape_loss` (absolute error summed within each (example, line)
before the absolute value, so development errors inside one accident year and
line offset and errors across years do not), :func:`pool_pe_loss` (a signed
pooled bias penalty) and :func:`masked_mse` (a plain squared error that keeps
the cell-level fit honest). The absolute-value-after-summing is the whole point
of the first term: it scores the quantity a reserving actuary books, not the
cell-by-cell fit.

The denominators default to the MINIBATCH's own, which is what the protocol
does. That makes each batch's ratio a ratio of that batch, so the gradient is
not exactly the gradient of the full-cohort objective; the study's reason is
that it keeps the scale of the three terms comparable within a batch, and the
fixed-denominator form is what its earlier training function used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "anchor",
    "apply_support",
    "ay_line_ape_loss",
    "factor_support",
    "log_factors",
    "masked_mse",
    "point_loss",
    "pool_pe_loss",
    "project",
]

#: guards a ratio whose denominator is zero, as the reference implementation does
RATIO_EPS = 1e-8


def log_factors(phi: Tensor, net: Tensor | None, eps: float) -> Tensor:
    """``softplus(phi + eps * net)``, or ``softplus(phi)`` when ``net`` is None.

    ``phi`` is (n_l, n_d - 1) or already batched; ``net`` is (B, n_l, n_d - 1).
    The result is strictly positive, so every development factor is above one.
    """
    if net is None:
        return torch.nn.functional.softplus(phi)
    return torch.nn.functional.softplus(phi + eps * net)


def anchor(logf: Tensor, anchor_logf: Tensor, initial_logf: Tensor, width: float) -> Tensor:
    """Bound the learned factors to a window around the chain ladder's.

    ``anchor_logf + width * tanh((logf - initial_logf) / width)``. The
    correction is zero at the initialisation and saturates at ``+/- width``, so
    the chain ladder is where this variant starts rather than something it is
    compared against after the fact.
    """
    return anchor_logf + width * torch.tanh((logf - initial_logf) / width)


def apply_support(logf: Tensor, support: Tensor, fallback: Tensor) -> Tensor:
    """Substitute ``fallback`` at every (line, step) ``support`` marks False.

    ``support`` is (n_l, n_d - 1) and shared by the batch; ``fallback`` may be
    the same shape or already batched.
    """
    return torch.where(support.unsqueeze(0).expand_as(logf), logf, fallback.expand_as(logf))


def project(
    logf: Tensor,
    c_lk: Tensor,
    p_lk: Tensor,
    lk: Tensor,
    *,
    anchor_start: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Project each origin's cumulative forward and difference it back.

    ``logf`` (B, n_l, n_d - 1), ``c_lk`` and ``p_lk`` (B, n_l), ``lk`` (B,) the
    1-based latest visible lag. Returns ``(pred, C)``: the predicted incremental
    loss ratios (B, n_l * n_d) in the token layout, and the cumulative grid
    (B, n_l, n_d) in amounts.

    Cells at or before ``lk`` are OBSERVED and are held at the starting balance,
    so their predicted increment is zero except at the starting cell itself;
    only cells past it are projected. ``anchor_start`` is the unfloored starting
    balance the chain-ladder-anchored variant uses in place of ``c_lk``, which
    is floored at one dollar - so that variant can start an origin at zero and
    this one cannot.
    """
    batch, n_l, n_steps = logf.shape
    n_d = n_steps + 1
    zero = logf.new_zeros(batch, n_l, 1)
    # S[..., j] is the sum of the log factors BEFORE lag j, so the factor from
    # lag a to lag b is exp(S[b] - S[a])
    cumulative_logf = torch.cat([zero, logf.cumsum(dim=2)], dim=2)
    at_lk = cumulative_logf.gather(2, (lk - 1).view(batch, 1, 1).expand(batch, n_l, 1))
    lag = torch.arange(n_d, device=logf.device).view(1, 1, n_d)
    forward = (lag >= lk.view(batch, 1, 1)).to(logf.dtype)
    if anchor_start is None:
        grown = torch.exp(c_lk.log().unsqueeze(2) + cumulative_logf - at_lk)
        c = c_lk.unsqueeze(2) * (1 - forward) + grown * forward
    else:
        grown = torch.exp(cumulative_logf - at_lk)
        c = anchor_start.unsqueeze(2) * ((1 - forward) + grown * forward)
    previous = torch.cat([torch.zeros_like(c[:, :, :1]), c[:, :, :-1]], dim=2)
    pred = ((c - previous) / p_lk.unsqueeze(2)).reshape(batch, n_l * n_d)
    return pred, c


def masked_mse(
    pred: Tensor, targ: Tensor, mask: Tensor, denominator: Tensor | float | None = None
) -> Tensor:
    """Mean squared error over the masked cells of the loss ratio."""
    n = mask.sum() if denominator is None else denominator
    if float(n) == 0:
        return pred.new_zeros(())
    return (((pred - targ) ** 2) * mask).sum() / n


def pool_pe_loss(
    pred: Tensor,
    targ: Tensor,
    mask: Tensor,
    prem: Tensor,
    denominator: Tensor | float | None = None,
) -> Tensor:
    """Absolute pooled percentage error: one signed total over everything.

    Every error in the batch is added up in dollars before the absolute value,
    so over-reserving one company and under-reserving another cancels. That is
    the point: this term penalises the BIAS of the batch, which the term below
    cannot see because it takes an absolute value first.
    """
    error = ((pred - targ) * mask * prem).sum()
    actual = (targ * mask * prem).sum()
    den = actual.abs() if denominator is None else denominator
    return (error / (den + RATIO_EPS)).abs()


def ay_line_ape_loss(
    pred: Tensor,
    targ: Tensor,
    mask: Tensor,
    prem: Tensor,
    n_l: int,
    n_d: int,
    denominator: Tensor | float | None = None,
) -> Tensor:
    """Absolute percentage error at the (accident year, line) level.

    Errors are summed in dollars WITHIN each (example, line) before the absolute
    value and across them after, so being early on one development lag and late
    on the next costs nothing while being wrong about the accident year costs
    the difference. It is a finer level than the company reserve the study
    validates on, and deliberately so - a company-level training loss lets two
    lines cancel each other all the way through training.
    """
    error = ((pred - targ) * mask * prem).reshape(-1, n_l, n_d).sum(2)
    actual = (targ * mask * prem).reshape(-1, n_l, n_d).sum(2)
    den = actual.abs().sum() if denominator is None else denominator
    return error.abs().sum() / (den + RATIO_EPS)


def point_loss(
    pred: Tensor,
    targ: Tensor,
    mask: Tensor,
    prem: Tensor,
    n_l: int,
    n_d: int,
    *,
    w_pe: float,
    w_mse: float,
    mse_scale: float,
) -> Tensor:
    """The study's checkpoint objective, on this batch's own denominators."""
    return (
        ay_line_ape_loss(pred, targ, mask, prem, n_l, n_d)
        + w_pe * pool_pe_loss(pred, targ, mask, prem)
        + w_mse * masked_mse(pred, targ, mask) / mse_scale
    )


def factor_support(target_masks, lks, n_l: int, n_d: int) -> np.ndarray:
    """Which (line, step) factors any training target was projected across.

    ``target_masks`` is one (n_ex, n_l * n_d) scoring mask per training set and
    ``lks`` the matching (n_ex,) latest visible lags. A factor from lag ``d`` to
    lag ``d + 1`` is supervised when some example starts at or before ``d + 1``
    and has a scored cell past ``d`` on that line.

    Read off the masks and the starting lags alone - never off a value - so it
    is a statement about which parameters the training data could move, not
    about what the validation or the test data look like. Numpy, because it is
    computed once per fit before any tensor exists.
    """
    support = np.zeros((n_l, n_d - 1), dtype=bool)
    for mask, lk in zip(target_masks, lks, strict=True):
        mask = np.asarray(mask)
        lk = np.asarray(lk).reshape(-1)
        for li in range(n_l):
            line = mask[:, li * n_d : (li + 1) * n_d]
            for d in range(n_d - 1):
                scored_beyond = line[:, d + 1 :].sum(axis=1) > 0
                support[li, d] |= bool(((lk <= d + 1) & scored_beyond).any())
    return support
