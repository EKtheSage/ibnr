"""The bivariate mixture head: one Gaussian mixture over (paid increment, case
movement) per cell, shared by both of the entry's backbones.

This module imports torch - only import it from inside the entry's fit/predict
paths (``ibnr.gallery`` must import without the [nn] extra; see CLAUDE.md's
"torch must never be imported at module level" rule). It is the exact analogue
of ``transformer/network.py``'s ``mdn_nll``/``mdn_sample`` pair, one dimension
wider, and follows those conventions deliberately: free functions over plain
tensors, an explicit ``torch.Generator`` on every sampler, mean-over-masked-
cells losses.

Why a FULL 2x2 covariance rather than two independent heads: the correlation
between payment and case run-off - paid up, case down - is the quantity this
entry exists to learn, and it is a per-cell quantity (strong late in
development, weak at dev 1). Two marginal heads would give the same means and
the wrong joint, so a sampled diagonal fed back into the rollout would carry
paid and case movements that do not offset each other.

The pieces, all consumed by the entry's networks and training loop:

- :class:`BivariateMixtureHead` - ``d_model`` -> mixture parameters (the layer
  both backbones end with);
- :func:`chol_from_raw` - the 3-raw-parameter Cholesky construction;
- :func:`joint_log_prob` / :func:`margin_log_prob` - per-cell log densities
  (the primitives; everything below is built from them);
- :func:`nll_joint` - the pure bivariate training loss;
- :func:`nll_mixed` - the MIXED-OBSERVEDNESS loss (joint where both targets are
  observed, the closed-form paid margin where only paid is, the case margin
  where only case is - one scalar, no data discarded);
- :func:`paid_margin` / :func:`case_margin` - the closed-form univariate
  margins; the paid one is what held-out scoring consumes, and it is exactly
  the ``(log_pi, mu, sigma)`` shape ``PooledMDNHeldout`` already scores;
- :func:`sample_joint` - one (paid, case) draw per cell for the rollout;
- :func:`implied_correlation` / :func:`mixture_moments` - read-outs of the
  fitted head (the card's case run-off diagnostic, and the tests' reference).

PARAMETERIZATION. Per cell and component: a mixture logit, a mean vector
``mu (2,)``, and a lower-triangular Cholesky factor built from 3 free numbers,
``L = [[softplus(a) + floor, 0], [b, softplus(c) + floor]]``. Covariance is
``L L^T``, so ``b`` is unconstrained and the correlation is free over
``(-1, 1)`` while the covariance stays positive definite by construction. No
matrix is ever inverted: the Mahalanobis term is a triangular solve.
"""

from __future__ import annotations

import math

import torch
from torch import nn

LOG_2PI = math.log(2.0 * math.pi)

#: Floor on BOTH Cholesky diagonal entries, matching the family's univariate
#: heads (``transformer/network.py`` floors sigma at the same 1e-3). It does
#: three jobs here, all load-bearing:
#:
#: 1. a collapsing component cannot drive the NLL to -inf;
#: 2. ``L`` is never singular, so :func:`joint_log_prob`'s triangular solve
#:    never divides by zero and ``log|L|`` is always finite, however large ``b``
#:    grows (in exact arithmetic the implied correlation stays inside
#:    ``(-1, 1)``; see :func:`implied_correlation` for what float64 does with
#:    that at absurd ratios, which is a read-out concern, not a solve one);
#: 3. :func:`case_margin`'s ``sqrt(b**2 + L11**2)`` has argument >= 1e-6 > 0,
#:    so its gradient is finite (``d sqrt / dx`` is infinite at 0).
#:
#: The targets are per-dev standardized loss ratios, i.e. O(1), so 1e-3 is
#: ~0.1% of a typical scale: real protection, negligible bias.
SIGMA_FLOOR = 1e-3

#: Head outputs per component: 1 mixture logit + 2 means + 3 Cholesky raws.
PARAMS_PER_COMPONENT = 6


def chol_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """``(..., K, 3)`` unconstrained -> ``(..., K, 2, 2)`` Cholesky factors.

    ``L = [[softplus(a) + floor, 0], [b, softplus(c) + floor]]``. The diagonal
    is positive by construction and ``b`` is free, which is the whole point:
    the correlation is unconstrained while ``L L^T`` stays a valid covariance.

    Built with ``stack`` rather than by assigning into a zeros tensor so the
    graph carries no in-place writes.
    """
    if raw.shape[-1] != 3:
        raise ValueError(
            f"chol_from_raw expects 3 raw parameters per component (a, b, c), got "
            f"{raw.shape[-1]} in a tensor of shape {tuple(raw.shape)}"
        )
    l00 = nn.functional.softplus(raw[..., 0]) + SIGMA_FLOOR
    l10 = raw[..., 1]
    l11 = nn.functional.softplus(raw[..., 2]) + SIGMA_FLOOR
    zero = torch.zeros_like(l00)
    return torch.stack([torch.stack([l00, zero], dim=-1), torch.stack([l10, l11], dim=-1)], dim=-2)


class BivariateMixtureHead(nn.Module):
    """``(..., d_model)`` hidden states -> a bivariate mixture per cell.

    The final layer of both backbones. ``forward`` returns
    ``log_pi (..., K)``, ``mu (..., K, 2)``, ``chol (..., K, 2, 2)`` - the
    argument triple every function in this module takes, in that order.

    Coordinate 0 is the paid increment, coordinate 1 the case movement, on
    whatever normalized scale the entry trains on. The head knows nothing about
    that scale; un-standardization is the entry's job.
    """

    def __init__(self, d_model: int, n_components: int) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = n_components
        self.proj = nn.Linear(d_model, n_components * PARAMS_PER_COMPONENT)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.proj(h).reshape(*h.shape[:-1], self.n_components, PARAMS_PER_COMPONENT)
        log_pi = out[..., 0].log_softmax(dim=-1)  # (..., K) normalized log weights
        mu = out[..., 1:3]  # (..., K, 2) component means (paid, case)
        chol = chol_from_raw(out[..., 3:])  # (..., K, 2, 2)
        return log_pi, mu, chol


def _mask_safe(y: torch.Tensor, mask: torch.Tensor | None, *, event: bool) -> torch.Tensor:
    """Substitute a finite dummy wherever ``mask`` is false, INSIDE the formula.

    The repo's Clark/guszcza rule: masking the RESULT of a density gives the
    right value and a poisoned gradient, because both branches are evaluated
    and ``0 * nan`` is ``nan``. An unobserved cell may carry contract padding,
    a stale value, or nothing at all, so the substitution happens before the
    residual is ever formed. Any finite substitute would do - the term is
    multiplied by zero afterwards - and zero is the cheapest.

    ``event=True`` when ``y`` carries a trailing size-2 event axis the mask
    does not (the joint case), so the mask broadcasts over both coordinates:
    a cell is jointly observed or it is not, never half.
    """
    if mask is None:
        return y
    m = mask.unsqueeze(-1) if event else mask
    return torch.where(m, y, torch.zeros_like(y))


def joint_log_prob(
    log_pi: torch.Tensor,  # (..., K)
    mu: torch.Tensor,  # (..., K, 2)
    chol: torch.Tensor,  # (..., K, 2, 2)
    y: torch.Tensor,  # (..., 2)
    mask: torch.Tensor | None = None,  # (...) bool
) -> torch.Tensor:
    """Per-cell log density of the bivariate mixture at ``y``: ``(...)``.

    ``log sum_k exp(log_pi_k + log N2(y; mu_k, L_k L_k^T))``, with the
    Mahalanobis term from a triangular solve against ``L`` - no matrix inverse,
    and ``log|L L^T| / 2 = sum(log diag(L))`` for free.

    Where ``mask`` is false the returned value is meaningless (a density at a
    substituted dummy point) but always finite; callers zero it. That is the
    contract that makes :func:`nll_mixed`'s poisoned cells bit-identical.
    """
    if y.shape[-1] != 2:
        raise ValueError(
            f"joint_log_prob expects a trailing event axis of 2 (paid, case), got {tuple(y.shape)}"
        )
    y_ = _mask_safe(y, mask, event=True)
    # (..., 1, 2) - (..., K, 2) -> (..., K, 2), then a column vector to solve against
    diff = (y_.unsqueeze(-2) - mu).unsqueeze(-1)  # (..., K, 2, 1)
    z = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)  # (..., K, 2)
    log_det = chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # (..., K) = 0.5 log|cov|
    comp = -0.5 * (z**2).sum(dim=-1) - log_det - LOG_2PI  # (..., K); 2 dims -> 2 * 0.5 * LOG_2PI
    return torch.logsumexp(log_pi + comp, dim=-1)  # (...)


def margin_log_prob(
    log_pi: torch.Tensor,  # (..., K)
    mu_m: torch.Tensor,  # (..., K)
    sigma_m: torch.Tensor,  # (..., K)
    y: torch.Tensor,  # (...)
    mask: torch.Tensor | None = None,  # (...) bool
) -> torch.Tensor:
    """Per-cell log density of a UNIVARIATE mixture: ``(...)``.

    Consumed with the output of :func:`paid_margin` / :func:`case_margin`. Same
    formula as ``transformer.network.mdn_nll``'s per-component term (pinned by
    a test), split out here because :func:`nll_mixed` needs the per-cell
    density rather than a mean over its own mask.
    """
    y_ = _mask_safe(y, mask, event=False)
    comp = -0.5 * ((y_.unsqueeze(-1) - mu_m) / sigma_m) ** 2 - sigma_m.log() - 0.5 * LOG_2PI
    return torch.logsumexp(log_pi + comp, dim=-1)


def nll_joint(
    log_pi: torch.Tensor,  # (..., K)
    mu: torch.Tensor,  # (..., K, 2)
    chol: torch.Tensor,  # (..., K, 2, 2)
    y: torch.Tensor,  # (..., 2)
    mask: torch.Tensor,  # (...) bool - cells where BOTH targets are observed
) -> torch.Tensor:
    """Mean negative joint log likelihood over masked cells.

    The training objective at cells where paid increment and case movement are
    both observed. ``mask`` is the set of scored cells - under cutoff
    augmentation, the observed training cells strictly past the drawn cutoff.
    """
    ll = joint_log_prob(log_pi, mu, chol, y, mask=mask)
    n = mask.sum().clamp(min=1)  # clamp: never divide by zero if a batch has no targets
    return -(ll * mask).sum() / n


def nll_mixed(
    log_pi: torch.Tensor,  # (..., K)
    mu: torch.Tensor,  # (..., K, 2)
    chol: torch.Tensor,  # (..., K, 2, 2)
    y_paid: torch.Tensor,  # (...)
    y_case: torch.Tensor,  # (...)
    joint_mask: torch.Tensor,  # (...) bool - both targets observed
    paid_only_mask: torch.Tensor,  # (...) bool - paid observed, case movement not
    case_only_mask: torch.Tensor,  # (...) bool - case movement observed, paid not
) -> torch.Tensor:
    """The mixed-observedness loss: one scalar, three masks, no data discarded.

    A cell's case MOVEMENT needs two present level cells (``d`` and ``d-1``), so
    a hole in the case field costs the movement target at two cells while the
    paid increment at both is perfectly observable. Discarding those cells would
    throw away the majority of the signal on the field the board actually scores.
    Instead each cell is scored under the density it has evidence for:

    - both observed -> the joint bivariate density;
    - paid only -> the closed-form PAID margin (a mixture of Gaussians'
      margin is the mixture of the component margins, exactly, so this is the
      same fitted head marginalized - not a second model);
    - case only -> the CASE margin, likewise.

    The three contributions are summed and divided by the total number of
    scored cells, so a cell counts once whichever branch scored it and the loss
    does not silently reweight itself as holes come and go.

    The masks must be disjoint, and that is CHECKED rather than documented: an
    overlap double-counts a cell under two densities, which no output would
    reveal. The check is a device sync per call, which is free at this repo's
    grid sizes (~55 cells per cohort).
    """
    for name, m in (
        ("joint_mask", joint_mask),
        ("paid_only_mask", paid_only_mask),
        ("case_only_mask", case_only_mask),
    ):
        if m.shape != y_paid.shape:
            raise ValueError(
                f"{name} has shape {tuple(m.shape)}; every mask must match y_paid "
                f"{tuple(y_paid.shape)} one flag per cell"
            )
    if y_case.shape != y_paid.shape:
        raise ValueError(
            f"y_case {tuple(y_case.shape)} and y_paid {tuple(y_paid.shape)} must be the "
            "same grid: one paid increment and one case movement per cell"
        )
    overlap = (joint_mask & paid_only_mask) | (joint_mask & case_only_mask)
    overlap = overlap | (paid_only_mask & case_only_mask)
    if bool(overlap.any()):
        raise ValueError(
            f"the three masks overlap at {int(overlap.sum())} cell(s); joint / paid-only / "
            "case-only partition the scored cells, and an overlap scores a cell twice"
        )

    y2 = torch.stack([y_paid, y_case], dim=-1)  # (..., 2)
    ll_joint = joint_log_prob(log_pi, mu, chol, y2, mask=joint_mask)
    ll_paid = margin_log_prob(*paid_margin(log_pi, mu, chol), y_paid, mask=paid_only_mask)
    ll_case = margin_log_prob(*case_margin(log_pi, mu, chol), y_case, mask=case_only_mask)

    total = (
        (ll_joint * joint_mask).sum()
        + (ll_paid * paid_only_mask).sum()
        + (ll_case * case_only_mask).sum()
    )
    n = (joint_mask.sum() + paid_only_mask.sum() + case_only_mask.sum()).clamp(min=1)
    return -total / n


def paid_margin(
    log_pi: torch.Tensor,  # (..., K)
    mu: torch.Tensor,  # (..., K, 2)
    chol: torch.Tensor,  # (..., K, 2, 2)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form paid margin: ``(log_pi, mu_p, sigma_p)``, each ``(..., K)``.

    A Gaussian mixture's margin is the mixture of the component margins with the
    SAME weights, so this is exact, not an approximation. Coordinate 0's
    variance is ``(L L^T)_00 = L00**2``, hence ``sigma_p = L00`` - the one
    entry the second row never touches.

    This is what held-out scoring consumes: a univariate mixture in exactly the
    ``(log_pi, mu, sigma)`` shape ``PooledMDNHeldout`` already scores and
    samples, so the entry joins the board with a true marginal paid density.
    """
    return log_pi, mu[..., 0], chol[..., 0, 0]


def case_margin(
    log_pi: torch.Tensor,  # (..., K)
    mu: torch.Tensor,  # (..., K, 2)
    chol: torch.Tensor,  # (..., K, 2, 2)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form case margin: ``(log_pi, mu_c, sigma_c)``, each ``(..., K)``.

    Coordinate 1's variance is ``(L L^T)_11 = L10**2 + L11**2``: BOTH entries of
    the second row, because the off-diagonal is the part of the case movement
    shared with the paid increment. Dropping it would understate every case
    scale by the size of the correlation - and would look perfectly healthy,
    which is why the margin has its own quadrature test.
    """
    return log_pi, mu[..., 1], torch.sqrt(chol[..., 1, 0] ** 2 + chol[..., 1, 1] ** 2)


def sample_joint(
    log_pi: torch.Tensor,  # (..., K)
    mu: torch.Tensor,  # (..., K, 2)
    chol: torch.Tensor,  # (..., K, 2, 2)
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One joint draw of every cell: ``(..., 2)`` = (paid increment, case movement).

    Ancestral sampling per cell, exactly as ``mdn_sample`` does it one dimension
    down: pick a component by inverse-CDF on its weight, then draw
    ``x = mu + L z`` with ``z ~ N(0, I)`` - the reparameterization of a
    multivariate normal with Cholesky factor ``L``. The rollout calls this once
    per calendar diagonal per draw, and the two coordinates come out of the SAME
    component and the SAME ``z``, which is what carries the learned correlation
    into the simulated diagonal. ``generator`` makes the draw
    reproducible/seed-controlled.
    """
    k = log_pi.shape[-1]
    # inverse-CDF component pick: u in [0,1); count how many cumulative weights
    # it exceeds -> the chosen component index (clamped to the last for u==1).
    u = torch.rand(*log_pi.shape[:-1], 1, generator=generator, device=mu.device, dtype=mu.dtype)
    cum = log_pi.exp().cumsum(dim=-1)  # (..., K)
    comp = (u > cum).sum(dim=-1).clamp(max=k - 1)  # (...) component index
    # gather that component's mean vector and 2x2 factor for every cell
    mu_s = mu.gather(-2, comp[..., None, None].expand(*comp.shape, 1, 2)).squeeze(-2)  # (..., 2)
    gather = comp[..., None, None, None].expand(*comp.shape, 1, 2, 2)
    chol_s = chol.gather(-3, gather).squeeze(-3)  # (..., 2, 2)
    z = torch.randn(*mu_s.shape, 1, generator=generator, device=mu.device, dtype=mu.dtype)
    return mu_s + (chol_s @ z).squeeze(-1)  # (..., 2)


def implied_correlation(chol: torch.Tensor) -> torch.Tensor:
    """Per-component correlation of ``L L^T``: ``(..., K)``.

    ``cov_01 / sqrt(cov_00 cov_11) = L00 L10 / (L00 sqrt(L10**2 + L11**2))``,
    where ``L00`` cancels. The entry's headline learned quantity: negative is
    payment replacing case reserve (paid up, case down). The
    ``L11 >= SIGMA_FLOOR`` floor keeps the denominator positive, so the value is
    inside ``(-1, 1)`` in exact arithmetic for every ``L`` this head can build.

    In float64 it can still ROUND to exactly ``+-1``: at ``|b| / L11 = 1e9`` the
    true value is ``1 - 5e-19``, below the spacing of 1.0. That is a property of
    the read-out, not of the density - the same ``L`` solves and scores
    perfectly (pinned by a test), because nothing downstream consumes this
    number. Do not build a positive-definiteness check on it.
    """
    return chol[..., 1, 0] / torch.sqrt(chol[..., 1, 0] ** 2 + chol[..., 1, 1] ** 2)


def mixture_moments(
    log_pi: torch.Tensor,  # (..., K)
    mu: torch.Tensor,  # (..., K, 2)
    chol: torch.Tensor,  # (..., K, 2, 2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean ``(..., 2)`` and covariance ``(..., 2, 2)`` of the whole mixture.

    Law of total covariance: ``Cov = sum_k pi_k (Sigma_k + d_k d_k^T)`` with
    ``d_k = mu_k - mean``. Used by the card's case run-off diagnostic and as the
    reference the sampler's empirical moments are checked against; the mixture
    correlation it implies is NOT any component's :func:`implied_correlation`
    unless K is 1.
    """
    pi = log_pi.exp()  # (..., K)
    mean = (pi.unsqueeze(-1) * mu).sum(dim=-2)  # (..., 2)
    cov_k = chol @ chol.transpose(-1, -2)  # (..., K, 2, 2)
    d = mu - mean.unsqueeze(-2)  # (..., K, 2)
    spread = d.unsqueeze(-1) @ d.unsqueeze(-2)  # (..., K, 2, 2) outer products
    cov = (pi[..., None, None] * (cov_k + spread)).sum(dim=-3)  # (..., 2, 2)
    return mean, cov
