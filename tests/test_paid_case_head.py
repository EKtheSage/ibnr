"""gallery.nn.nn_paid_case.head: the bivariate mixture. Skips without torch.

This file protects the mathematical core of the entry, before any entry wiring
exists. Four claims, and they fail for different reasons:

1. **The gradient is the one the density implies.** Autograd against central
   finite differences over a parameter grid that includes near-singular
   components (correlation +-0.999) and a masked cell poisoned with NaN. A
   VALUE-only test here would be worse than none: the repo has been bitten
   twice by densities that return the right number and a NaN gradient (Clark's
   ``theta/0`` in an unselected ``where`` branch, guszcza's ``log`` of a
   negative mean), and in both cases NUTS or the optimizer died with no useful
   message while every value assertion stayed green. The masking is where this
   head can repeat it: ``0 * nan`` is ``nan``, so a mask applied to the RESULT
   of the density rather than substituted INSIDE it poisons the whole backward
   pass from one unobserved cell.
2. **The margins are the joint's margins.** Checked against numerical
   quadrature of the joint density over the other coordinate, at a grid of
   evaluation points, for BOTH margins. The paid margin is what the leaderboard
   scores, so if it is not the fitted head's true margin the entry's board row
   is a number from a model nobody fitted. The case margin is the one that can
   be wrong quietly: ``sqrt(L10**2 + L11**2)`` degrades to ``L11`` if the
   off-diagonal is dropped, which is a perfectly valid-looking density that
   understates every case scale by the size of the correlation.
3. **The correlation is learnable, with sign.** Synthetic data from a known
   strongly NEGATIVE bivariate normal - the sign the correlation between
   payment and case run-off carries, and what this entry exists to capture -
   recovered by gradient descent on ``nll_joint`` alone.
4. **The mixed-observedness loss reads exactly the cells it claims to.** Its
   three branches are pinned against a hand-computed scipy reference, and a
   cell no mask selects is poisoned with NaN to prove it is never read: the
   loss must come back BIT-identical, not merely close.

Every test here runs on tiny CPU tensors in float64 - this is the head's math,
not the entry's plumbing, so nothing constructs a triangle or a config.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

torch = pytest.importorskip("torch")

from ibnr.gallery.nn.nn_paid_case import head  # noqa: E402
from ibnr.gallery.nn.transformer.network import mdn_nll  # noqa: E402

DTYPE = torch.float64


def _inv_softplus(x: float) -> float:
    """The raw value whose ``softplus`` is ``x`` (x > 0)."""
    return math.log(math.expm1(x))


def _raw_for(sd_p: float, sd_c: float, rho: float) -> list[float]:
    """Raw ``(a, b, c)`` giving marginal sds ``(sd_p, sd_c)`` and correlation ``rho``.

    Inverts :func:`head.chol_from_raw`: ``L00 = sd_p``, ``L10 = rho * sd_c``,
    ``L11 = sd_c * sqrt(1 - rho**2)``, then undoes the softplus and the floor on
    the two diagonal entries. Tests state the covariance they mean rather than
    three unconstrained numbers whose implied correlation nobody can read.
    """
    l00, l10 = sd_p, rho * sd_c
    l11 = sd_c * math.sqrt(1.0 - rho**2)
    return [_inv_softplus(l00 - head.SIGMA_FLOOR), l10, _inv_softplus(l11 - head.SIGMA_FLOOR)]


def _params(rows: list[list[list[float]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[[[w, mu_p, mu_c, sd_p, sd_c, rho], ...component...], ...cell...]`` ->
    ``(logits, mu, raw)`` ready for ``log_softmax`` / :func:`head.chol_from_raw`."""
    logits, mu, raw = [], [], []
    for cell in rows:
        logits.append([c[0] for c in cell])
        mu.append([[c[1], c[2]] for c in cell])
        raw.append([_raw_for(c[3], c[4], c[5]) for c in cell])
    return (
        torch.tensor(logits, dtype=DTYPE),
        torch.tensor(mu, dtype=DTYPE),
        torch.tensor(raw, dtype=DTYPE),
    )


def _covariances(chol: torch.Tensor) -> np.ndarray:
    """``L L^T`` per component as numpy, for the scipy references."""
    return (chol @ chol.transpose(-1, -2)).detach().numpy()


# --------------------------------------------------------------------------
# 1. gradients
# --------------------------------------------------------------------------


def _flat_loss(v: torch.Tensor, n_cells: int, k: int, y: torch.Tensor, mask: torch.Tensor):
    """``nll_joint`` as a function of one flat parameter vector.

    Unpacks ``v`` into logits / means / Cholesky raws and rebuilds the whole
    head math, so the finite-difference check covers ``log_softmax`` and
    :func:`head.chol_from_raw`'s softplus as well as the density itself.
    """
    n_logit, n_mu = n_cells * k, n_cells * k * 2
    logits = v[:n_logit].reshape(n_cells, k)
    mu = v[n_logit : n_logit + n_mu].reshape(n_cells, k, 2)
    raw = v[n_logit + n_mu :].reshape(n_cells, k, 3)
    return head.nll_joint(logits.log_softmax(dim=-1), mu, head.chol_from_raw(raw), y, mask)


def _fd_grad(f, v: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
    """Central differences, step scaled to each coordinate's own magnitude.

    An absolute step is a lottery on a parameter vector whose entries span
    orders of magnitude - the repo's tolerance rule, one level down.
    """
    g = torch.zeros_like(v)
    for i in range(v.numel()):
        step = h * max(1.0, abs(float(v[i])))
        up, down = v.clone(), v.clone()
        up[i] += step
        down[i] -= step
        g[i] = (f(up) - f(down)) / (2.0 * step)
    return g


@pytest.mark.parametrize("rho", [-0.999, -0.6, 0.0, 0.6, 0.999])
def test_nll_joint_gradient_matches_finite_differences(rho):
    """Autograd vs central differences, including near-singular components.

    A value-only test of this loss is worse than no test: it passes on an
    implementation whose backward pass is entirely NaN (CLAUDE.md, milestone 5,
    twice). So the assertion is on the GRADIENT of every parameter, and the
    grid includes correlation +-0.999, where the Cholesky is nearly singular and
    the triangular solve is closest to dividing by something tiny.

    Cell 1 is masked off and its target poisoned with NaN. A correct
    implementation substitutes a finite dummy INSIDE the formula, so the
    poisoned cell contributes nothing to either the value or the gradient; an
    implementation that masks the density's RESULT instead returns the same
    value and an all-NaN gradient, which this comparison catches and no value
    assertion can.
    """
    k, n_cells = 2, 3
    logits, mu, raw = _params(
        [
            [[0.3, 0.5, -0.4, 0.7, 0.5, rho], [-0.2, -0.6, 0.9, 0.4, 0.8, -0.5 * rho]],
            [[0.0, 0.1, 0.2, 1.1, 0.6, -rho], [0.5, 0.9, -1.2, 0.3, 0.35, 0.2]],
            [[-0.4, -0.3, 0.4, 0.6, 0.9, 0.8 * rho], [0.1, 0.4, 0.1, 0.9, 0.5, -0.3]],
        ]
    )
    y = torch.tensor([[0.6, -0.2], [float("nan"), float("nan")], [-0.5, 0.7]], dtype=DTYPE)
    mask = torch.tensor([True, False, True])

    v = torch.cat([logits.reshape(-1), mu.reshape(-1), raw.reshape(-1)]).requires_grad_(True)
    loss = _flat_loss(v, n_cells, k, y, mask)
    assert torch.isfinite(loss), "the poisoned masked cell reached the loss value"
    loss.backward()
    auto = v.grad.detach().clone()
    assert torch.isfinite(auto).all(), "the poisoned masked cell reached the gradient"

    with torch.no_grad():
        fd = _fd_grad(lambda w: _flat_loss(w, n_cells, k, y, mask), v.detach())

    scale = float(auto.abs().max().clamp(min=1.0))
    worst = float((auto - fd).abs().max())
    assert worst <= 1e-6 * scale, (
        f"autograd and central differences disagree by {worst:.3e} against a gradient scale of "
        f"{scale:.3e} at rho={rho}"
    )


def test_nll_mixed_gradient_matches_finite_differences():
    """The same check on the three-branch loss, which has three ways to leak.

    ``nll_mixed`` substitutes in three places (the joint, and each margin), and
    a miss in any one of them poisons the whole backward pass from a single
    cell that branch does not score. Every cell here is unobserved in at least
    one coordinate and carries NaN there, so all three substitutions are live.
    """
    k, n_cells = 2, 3
    logits, mu, raw = _params(
        [
            [[0.2, 0.4, -0.5, 0.8, 0.6, -0.9], [0.1, -0.3, 0.7, 0.5, 0.4, 0.3]],
            [[-0.5, 0.9, 0.2, 0.6, 1.0, 0.5], [0.4, 0.2, -0.8, 1.2, 0.3, -0.4]],
            [[0.0, -0.7, 0.3, 0.9, 0.7, -0.6], [0.3, 0.5, 0.5, 0.4, 0.9, 0.95]],
        ]
    )
    nan = float("nan")
    y_paid = torch.tensor([0.3, -0.4, nan], dtype=DTYPE)
    y_case = torch.tensor([-0.2, nan, 0.8], dtype=DTYPE)
    joint = torch.tensor([True, False, False])
    paid_only = torch.tensor([False, True, False])
    case_only = torch.tensor([False, False, True])

    def loss_of(v):
        n_logit, n_mu = n_cells * k, n_cells * k * 2
        return head.nll_mixed(
            v[:n_logit].reshape(n_cells, k).log_softmax(dim=-1),
            v[n_logit : n_logit + n_mu].reshape(n_cells, k, 2),
            head.chol_from_raw(v[n_logit + n_mu :].reshape(n_cells, k, 3)),
            y_paid,
            y_case,
            joint,
            paid_only,
            case_only,
        )

    v = torch.cat([logits.reshape(-1), mu.reshape(-1), raw.reshape(-1)]).requires_grad_(True)
    loss = loss_of(v)
    loss.backward()
    auto = v.grad.detach().clone()
    assert torch.isfinite(auto).all(), "an unobserved coordinate reached the gradient"

    with torch.no_grad():
        fd = _fd_grad(loss_of, v.detach())

    scale = float(auto.abs().max().clamp(min=1.0))
    assert float((auto - fd).abs().max()) <= 1e-6 * scale


# --------------------------------------------------------------------------
# 2. margins
# --------------------------------------------------------------------------

#: A deliberately awkward 3-component mixture: mixed correlation signs, one
#: nearly singular component, marginal scales spanning 3x.
MARGIN_ROWS = [
    [
        [0.4, 0.5, -0.3, 0.40, 0.60, -0.70],
        [-0.2, -1.0, 0.8, 0.90, 0.30, 0.50],
        [0.9, 0.2, 0.1, 0.50, 0.45, -0.95],
    ]
]


def _margin_by_quadrature(axis: int, points: torch.Tensor, n_nodes: int = 20_001):
    """Log density of the joint marginalized over the OTHER axis, at ``points``.

    Trapezoidal quadrature over a +-14 sd window. That rule is exponentially
    accurate for a smooth density decaying at both ends (Euler-Maclaurin: every
    boundary term vanishes), so the residual error is the truncation at the
    window edge, ~1e-40 here - which is what lets the comparison below be
    asserted at 1e-9 rather than at "close enough for a plot".
    """
    logits, mu, raw = _params(MARGIN_ROWS)
    log_pi, chol = logits.log_softmax(dim=-1), head.chol_from_raw(raw)
    other = 1 - axis
    sd = torch.sqrt((chol @ chol.transpose(-1, -2))[..., other, other])
    lo = float((mu[..., other] - 14.0 * sd).min())
    hi = float((mu[..., other] + 14.0 * sd).max())
    nodes = torch.linspace(lo, hi, n_nodes, dtype=DTYPE)  # (Q,)

    # (P, Q, 2): every evaluation point crossed with every quadrature node
    p_grid, q_grid = torch.meshgrid(points, nodes, indexing="ij")
    y = torch.stack([p_grid, q_grid] if axis == 0 else [q_grid, p_grid], dim=-1)
    # params carry a leading (1, 1) so they broadcast over (P, Q)
    dens = head.joint_log_prob(log_pi[:, None], mu[:, None], chol[:, None], y).exp()
    return torch.trapezoid(dens, nodes, dim=-1).log()  # (P,)


@pytest.mark.parametrize("axis, margin_fn", [(0, head.paid_margin), (1, head.case_margin)])
def test_margin_equals_the_numerically_marginalized_joint(axis, margin_fn):
    """The closed-form margin IS the joint's margin, to quadrature accuracy.

    A mixture of Gaussians marginalizes exactly - same weights, component
    margins - so this is an identity, not an approximation, and it is asserted
    as one. It is the only check that catches a wrong variance entry: the case
    margin's scale is ``sqrt(L10**2 + L11**2)`` and every plausible-looking
    mistake (using ``L11``, or ``L00``, or the covariance in place of the sd)
    still yields a valid density that scores, samples and trains.
    """
    logits, mu, raw = _params(MARGIN_ROWS)
    log_pi_m, mu_m, sigma_m = margin_fn(logits.log_softmax(dim=-1), mu, head.chol_from_raw(raw))

    lo = float((mu_m - 3.0 * sigma_m).min())
    hi = float((mu_m + 3.0 * sigma_m).max())
    points = torch.linspace(lo, hi, 9, dtype=DTYPE)

    closed = head.margin_log_prob(log_pi_m, mu_m, sigma_m, points[:, None]).squeeze(-1)
    numeric = _margin_by_quadrature(axis, points)

    assert torch.allclose(closed, numeric, rtol=1e-9, atol=1e-12), (
        f"axis {axis}: closed-form margin {closed.tolist()} vs quadrature {numeric.tolist()}"
    )


def test_margin_log_prob_is_the_familys_univariate_mixture():
    """The margin's density is the same formula the family already trains on.

    ``PooledMDNHeldout`` scores a ``(log_pi, mu, sigma)`` triple with the
    transformer's ``mdn_nll``, and this entry hands it :func:`head.paid_margin`
    output. If the two formulas ever drifted, the entry's board row would be
    computed by a density its own loss never used. Checked as an identity of
    numbers, not of code paths, because the two implementations differ (per-cell
    density here, mean over a mask there).
    """
    logits, mu, raw = _params(MARGIN_ROWS * 3)
    log_pi_m, mu_m, sigma_m = head.paid_margin(
        logits.log_softmax(dim=-1), mu, head.chol_from_raw(raw)
    )
    y = torch.tensor([0.4, -0.9, 0.1], dtype=DTYPE)
    mask = torch.tensor([True, True, False])

    mine = -(head.margin_log_prob(log_pi_m, mu_m, sigma_m, y, mask=mask) * mask).sum() / mask.sum()
    theirs = mdn_nll(log_pi_m, mu_m, sigma_m, y, mask)
    assert torch.allclose(mine, theirs, rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------
# 3. correlation recovery
# --------------------------------------------------------------------------


def _fit_head(y: torch.Tensor, k: int, steps: int = 800, seed: int = 0):
    """Gradient descent on ``nll_joint`` alone over one cell's parameters.

    No entry machinery: this is the head fitting a pile of draws directly, which
    is the sharpest available statement that the loss is the data's likelihood
    and that its gradient points somewhere useful.
    """
    gen = torch.Generator().manual_seed(seed)
    logits = torch.zeros(1, k, dtype=DTYPE, requires_grad=True)
    mu = (0.1 * torch.randn(1, k, 2, generator=gen, dtype=DTYPE)).requires_grad_(True)
    raw = (0.1 * torch.randn(1, k, 3, generator=gen, dtype=DTYPE)).requires_grad_(True)
    mask = torch.ones(y.shape[0], dtype=torch.bool)
    opt = torch.optim.Adam([logits, mu, raw], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = head.nll_joint(logits.log_softmax(dim=-1), mu, head.chol_from_raw(raw), y, mask)
        loss.backward()
        opt.step()
    return logits.detach().log_softmax(dim=-1), mu.detach(), head.chol_from_raw(raw.detach())


#: The truth the recovery tests fit: paid up, case DOWN - payment replacing case.
TRUE_MU = (1.2, -0.7)
TRUE_SD = (0.5, 0.4)
TRUE_RHO = -0.75


def _draw_truth(n: int = 4000, seed: int = 11) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    chol = head.chol_from_raw(torch.tensor([[_raw_for(*TRUE_SD, TRUE_RHO)]], dtype=DTYPE))
    z = torch.randn(n, 2, 1, generator=gen, dtype=DTYPE)
    return torch.tensor(TRUE_MU, dtype=DTYPE) + (chol[0, 0] @ z).squeeze(-1)


def test_single_component_recovers_the_negative_correlation():
    """Fit K=1 to draws from a known strongly negative bivariate normal.

    The sign is the entry's whole claim (paid up, case down), so it is asserted
    separately from the magnitude: a head that fitted ``+0.75`` would be within
    a symmetric band of the truth and would model the opposite physics. The
    magnitude band is loose (0.1) against an MLE standard error of ~0.007 at
    n=4000 - this test is about the optimizer reaching the likelihood's mode,
    not about sampling noise.
    """
    log_pi, mu, chol = _fit_head(_draw_truth(), k=1)
    rho = float(head.implied_correlation(chol)[0, 0])

    assert rho < 0.0, f"recovered correlation {rho:.3f} has the wrong sign"
    assert abs(rho - TRUE_RHO) < 0.1, f"recovered correlation {rho:.3f} vs truth {TRUE_RHO}"
    assert torch.allclose(mu[0, 0], torch.tensor(TRUE_MU, dtype=DTYPE), atol=0.05)
    sd = torch.sqrt((chol @ chol.transpose(-1, -2))[0, 0].diagonal())
    assert torch.allclose(sd, torch.tensor(TRUE_SD, dtype=DTYPE), rtol=0.1)


def test_mixture_recovers_the_negative_correlation():
    """Same data, K=3: the MIXTURE's correlation, not a component's.

    With spare components the fit can split the single Gaussian several ways, so
    the quantity that must come back is the total-covariance correlation from
    :func:`head.mixture_moments` - which is what the card's case run-off
    diagnostic reports and what the rollout's sampled diagonal actually carries.
    """
    log_pi, mu, chol = _fit_head(_draw_truth(), k=3)
    _, cov = head.mixture_moments(log_pi, mu, chol)
    rho = float(cov[0, 0, 1] / torch.sqrt(cov[0, 0, 0] * cov[0, 1, 1]))

    assert rho < 0.0, f"recovered mixture correlation {rho:.3f} has the wrong sign"
    assert abs(rho - TRUE_RHO) < 0.15, f"recovered mixture correlation {rho:.3f}"


# --------------------------------------------------------------------------
# 4. the mixed-observedness loss
# --------------------------------------------------------------------------

#: Four cells x two components. Cell 3 is scored by NO mask - the poisoned one.
MIXED_ROWS = [
    [[0.3, 0.4, -0.5, 0.80, 0.60, -0.90], [-0.1, -0.3, 0.7, 0.50, 0.40, 0.30]],
    [[-0.5, 0.9, 0.2, 0.60, 1.00, 0.50], [0.4, 0.2, -0.8, 1.20, 0.30, -0.40]],
    [[0.0, -0.7, 0.3, 0.90, 0.70, -0.60], [0.3, 0.5, 0.5, 0.40, 0.90, 0.95]],
    [[0.2, 0.1, 0.1, 0.70, 0.70, 0.00], [0.1, 0.2, 0.2, 0.60, 0.60, 0.10]],
]
#: cell 0 joint, cell 1 paid-only, cell 2 case-only, cell 3 unscored
MIXED_JOINT = torch.tensor([True, False, False, False])
MIXED_PAID_ONLY = torch.tensor([False, True, False, False])
MIXED_CASE_ONLY = torch.tensor([False, False, True, False])
OBS_PAID, OBS_CASE = 0.35, -0.45


def _mixed_targets(poison: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """The four cells' targets, with every UNOBSERVED coordinate set to ``poison``.

    Unobserved here means exactly what the masks say: the case value at the
    paid-only cell, the paid value at the case-only cell, and both values at the
    unscored cell. ``poison=None`` fills them with a plain number instead, which
    is the control the bit-identity test differences against.
    """
    fill = 0.0 if poison is None else poison
    paid = torch.tensor([OBS_PAID, OBS_PAID, fill, fill], dtype=DTYPE)
    case = torch.tensor([OBS_CASE, fill, OBS_CASE, fill], dtype=DTYPE)
    return paid, case


def test_nll_mixed_equals_the_three_hand_computed_pieces():
    """Each branch against an independent scipy reference, summed by hand.

    scipy's ``multivariate_normal`` and ``norm`` share no code with this module,
    so this pins the joint density, both margins, and - the part only arithmetic
    can catch - the denominator: the three branches are summed and divided by
    the TOTAL number of scored cells, so a cell counts once whichever branch
    scored it. Averaging the three branch means instead would reweight the loss
    by how the holes happen to fall.
    """
    logits, mu, raw = _params(MIXED_ROWS)
    log_pi, chol = logits.log_softmax(dim=-1), head.chol_from_raw(raw)
    y_paid, y_case = _mixed_targets(float("nan"))

    pi = log_pi.exp().numpy()
    mu_np, cov = mu.numpy(), _covariances(chol)
    joint = np.log(
        sum(
            pi[0, k] * stats.multivariate_normal(mu_np[0, k], cov[0, k]).pdf([OBS_PAID, OBS_CASE])
            for k in range(pi.shape[1])
        )
    )
    paid = np.log(
        sum(
            pi[1, k] * stats.norm(mu_np[1, k, 0], math.sqrt(cov[1, k, 0, 0])).pdf(OBS_PAID)
            for k in range(pi.shape[1])
        )
    )
    case = np.log(
        sum(
            pi[2, k] * stats.norm(mu_np[2, k, 1], math.sqrt(cov[2, k, 1, 1])).pdf(OBS_CASE)
            for k in range(pi.shape[1])
        )
    )
    expected = -(joint + paid + case) / 3.0

    got = head.nll_mixed(
        log_pi, mu, chol, y_paid, y_case, MIXED_JOINT, MIXED_PAID_ONLY, MIXED_CASE_ONLY
    )
    assert float(got) == pytest.approx(expected, rel=1e-12)


def test_nll_mixed_never_reads_an_unobserved_value():
    """Poisoning every unobserved coordinate changes the loss BIT for bit.

    Not "within a tolerance": a value that is never read cannot move the last
    bit either. NaN is the poison because it is the failure the substitution
    exists to prevent - a real hole carries contract padding or a stale number,
    and multiplying a NaN density by a zero mask gives NaN, not zero.
    """
    logits, mu, raw = _params(MIXED_ROWS)
    log_pi, chol = logits.log_softmax(dim=-1), head.chol_from_raw(raw)

    def loss(poison):
        y_paid, y_case = _mixed_targets(poison)
        return head.nll_mixed(
            log_pi, mu, chol, y_paid, y_case, MIXED_JOINT, MIXED_PAID_ONLY, MIXED_CASE_ONLY
        )

    control = loss(None)
    assert torch.isfinite(control)
    for poison in (float("nan"), float("inf"), -1e300):
        assert loss(poison).item() == control.item(), f"poisoning with {poison} moved the loss"


def test_nll_joint_never_reads_an_unmasked_cell():
    """The same guarantee one branch down, where the substitution actually lives."""
    logits, mu, raw = _params(MIXED_ROWS)
    log_pi, chol = logits.log_softmax(dim=-1), head.chol_from_raw(raw)
    mask = torch.tensor([True, True, False, False])
    clean = torch.tensor([[0.3, -0.2], [0.1, 0.4], [0.0, 0.0], [0.0, 0.0]], dtype=DTYPE)
    dirty = clean.clone()
    dirty[2:] = float("nan")

    a = head.nll_joint(log_pi, mu, chol, clean, mask)
    b = head.nll_joint(log_pi, mu, chol, dirty, mask)
    assert torch.isfinite(a) and a.item() == b.item()


def test_nll_mixed_refuses_overlapping_masks():
    """A cell in two masks is scored twice, and no output would show it."""
    logits, mu, raw = _params(MIXED_ROWS)
    log_pi, chol = logits.log_softmax(dim=-1), head.chol_from_raw(raw)
    y_paid, y_case = _mixed_targets(None)
    both = torch.tensor([True, True, False, False])

    with pytest.raises(ValueError, match="overlap"):
        head.nll_mixed(log_pi, mu, chol, y_paid, y_case, both, MIXED_PAID_ONLY, MIXED_CASE_ONLY)


def test_nll_mixed_refuses_a_mask_of_the_wrong_shape():
    logits, mu, raw = _params(MIXED_ROWS)
    log_pi, chol = logits.log_softmax(dim=-1), head.chol_from_raw(raw)
    y_paid, y_case = _mixed_targets(None)

    with pytest.raises(ValueError, match="case_only_mask"):
        head.nll_mixed(
            log_pi,
            mu,
            chol,
            y_paid,
            y_case,
            MIXED_JOINT,
            MIXED_PAID_ONLY,
            torch.zeros(3, dtype=torch.bool),
        )


def test_nll_mixed_with_no_scored_cells_is_finite():
    """An all-empty batch divides by a clamped 1 rather than by zero."""
    logits, mu, raw = _params(MIXED_ROWS)
    log_pi, chol = logits.log_softmax(dim=-1), head.chol_from_raw(raw)
    y_paid, y_case = _mixed_targets(float("nan"))
    none = torch.zeros(4, dtype=torch.bool)

    assert float(head.nll_mixed(log_pi, mu, chol, y_paid, y_case, none, none, none)) == 0.0


# --------------------------------------------------------------------------
# 5. the sampler
# --------------------------------------------------------------------------


def _broadcast_params(rows, n_cells: int):
    """One cell's parameters broadcast over ``n_cells`` draws."""
    logits, mu, raw = _params(rows)
    log_pi, chol = logits.log_softmax(dim=-1), head.chol_from_raw(raw)
    return (
        log_pi.expand(n_cells, -1),
        mu.expand(n_cells, -1, -1),
        chol.expand(n_cells, -1, -1, -1),
    )


def test_sample_joint_is_seed_reproducible():
    """Same generator seed -> bit-identical draws; a different seed -> different.

    Without this a deep ensemble's spread is unreproducible and no calibration
    result off this entry is citable (the family's standing rule).
    """
    log_pi, mu, chol = _broadcast_params(MARGIN_ROWS, 64)
    a = head.sample_joint(log_pi, mu, chol, torch.Generator().manual_seed(3))
    b = head.sample_joint(log_pi, mu, chol, torch.Generator().manual_seed(3))
    c = head.sample_joint(log_pi, mu, chol, torch.Generator().manual_seed(4))

    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (64, 2)


def test_sample_joint_reproduces_a_single_components_moments():
    """Empirical mean, covariance and correlation of 200k draws vs ``(mu, L L^T)``.

    The sampler is where the learned dependence either reaches the rollout or
    quietly does not: drawing the two coordinates from independent normals would
    leave both margins perfect and the correlation at zero, so the correlation
    is asserted directly rather than inferred from the marginal moments.
    """
    n = 200_000
    log_pi, mu, chol = _broadcast_params([[[0.0, 0.9, -0.4, 0.7, 1.3, -0.8]]], n)
    draws = head.sample_joint(log_pi, mu, chol, torch.Generator().manual_seed(0))

    assert torch.allclose(draws.mean(0), mu[0, 0], atol=0.02)
    cov = torch.cov(draws.T)
    assert torch.allclose(cov, (chol @ chol.transpose(-1, -2))[0, 0], atol=0.03)
    rho = float(cov[0, 1] / torch.sqrt(cov[0, 0] * cov[1, 1]))
    assert rho == pytest.approx(float(head.implied_correlation(chol)[0, 0]), abs=0.01)


def test_sample_joint_reproduces_the_mixtures_moments():
    """A real K=3 mixture against :func:`head.mixture_moments`.

    Catches the component pick as well as the draw: a sampler that ignored the
    weights, or gathered the wrong component's Cholesky factor, still produces
    tidy Gaussian-looking draws whose moments are somebody else's.
    """
    n = 400_000
    log_pi, mu, chol = _broadcast_params(MARGIN_ROWS, n)
    draws = head.sample_joint(log_pi, mu, chol, torch.Generator().manual_seed(1))
    mean, cov = head.mixture_moments(log_pi[:1], mu[:1], chol[:1])

    assert torch.allclose(draws.mean(0), mean[0], atol=0.02)
    assert torch.allclose(torch.cov(draws.T), cov[0], atol=0.03)


# --------------------------------------------------------------------------
# 6. the parameterization itself
# --------------------------------------------------------------------------


def test_head_module_emits_valid_mixture_parameters():
    """Weights normalize, the factor is lower triangular with a floored diagonal.

    A point estimator cannot enter the gallery, so the validity of the
    distributional head is pinned directly rather than inferred from a
    downstream metric (the family's rule, one dimension up).
    """
    torch.manual_seed(0)
    module = head.BivariateMixtureHead(d_model=8, n_components=3).to(DTYPE)
    log_pi, mu, chol = module(torch.randn(5, 7, 8, dtype=DTYPE))

    assert log_pi.shape == (5, 7, 3) and mu.shape == (5, 7, 3, 2)
    assert chol.shape == (5, 7, 3, 2, 2)
    assert torch.allclose(log_pi.exp().sum(-1), torch.ones(5, 7, dtype=DTYPE))
    assert torch.equal(chol[..., 0, 1], torch.zeros_like(chol[..., 0, 1]))
    assert (chol[..., 0, 0] >= head.SIGMA_FLOOR).all()
    assert (chol[..., 1, 1] >= head.SIGMA_FLOOR).all()
    assert (head.implied_correlation(chol).abs() < 1.0).all()


def test_the_sigma_floor_keeps_an_extreme_factor_solvable():
    """An enormous off-diagonal still leaves a non-singular, solvable factor.

    This is what makes the +-0.999 arm of the gradient test a statement about
    the head rather than about the numbers that arm happened to pick: however
    hard training pushes the off-diagonal, the floor keeps ``L`` invertible, so
    the triangular solve, ``log|L|`` and their gradients stay finite.

    The correlation READ-OUT is the part that does not survive, and the split is
    deliberate. At ``|b| / L11 = 1e9`` the true correlation is ``1 - 5e-19``,
    which float64 rounds to exactly 1.0 - so ``implied_correlation`` reports an
    endpoint for a covariance that is still strictly positive definite. Nothing
    downstream consumes that number, and asserting ``< 1`` here would be
    asserting a property of float64 rather than of the head; three decades
    lower it holds comfortably, which is what the second half checks.
    """
    log_pi = torch.zeros(1, 1, dtype=DTYPE)
    mu = torch.zeros(1, 1, 2, dtype=DTYPE)
    y = torch.tensor([[0.4, -0.2]], dtype=DTYPE)

    raw = torch.tensor([[[0.0, 1e6, -30.0]]], dtype=DTYPE, requires_grad=True)
    chol = head.chol_from_raw(raw)
    ll = head.joint_log_prob(log_pi, mu, chol, y)
    assert torch.isfinite(ll).all()
    ll.sum().backward()
    assert torch.isfinite(raw.grad).all()
    rho = head.implied_correlation(chol).detach()
    assert float(rho[0, 0]) == 1.0  # the float64 round, documented above

    modest = head.chol_from_raw(torch.tensor([[[0.0, 1e3, -30.0]]], dtype=DTYPE))
    assert float(head.implied_correlation(modest)[0, 0]) < 1.0


def test_chol_from_raw_refuses_the_wrong_parameter_count():
    with pytest.raises(ValueError, match="3 raw parameters"):
        head.chol_from_raw(torch.zeros(2, 3, 4))


def test_joint_log_prob_refuses_a_non_bivariate_target():
    log_pi, mu, chol = _broadcast_params(MARGIN_ROWS, 2)
    with pytest.raises(ValueError, match="event axis of 2"):
        head.joint_log_prob(log_pi, mu, chol, torch.zeros(2, 3, dtype=DTYPE))
