"""Guszcza growth-curve log density and draws at arbitrary cells, plain numpy.

The counterpart of ``model.stan``'s ``generated quantities`` block, except that
it evaluates at cells the fit never saw. Free functions over plain arrays, per
CLAUDE.md decision 6: backend-blind, unit-testable without a sampler, living
beside the Stan program they mirror.

``model.stan``, the two lines that matter::

    mu[i]      = log(ulr[w[i]] * growth_curve(t[i], omega, theta, curve));
    log_lik[i] = lognormal_lpdf(y[i] | mu[i], sigma);

with ``y = C / premium`` the cumulative paid loss ratio. The growth curve is
**reused** from ``gallery/statistical/clark/model.py::growth`` - one algebra,
three readers (that module, this one, the Stan ``functions`` block) - and the
equivalence is pinned by test. ``growth`` is unit-agnostic: it only needs ``t``
and ``theta`` on the same scale, which here is YEARS (the post's ``dev_year``).

**``t`` comes from the CONTRACT's grain** (``d * dev_grain_months / 12``),
never from the cells: on the annual Schedule P grain ``t == d`` numerically, so
a scorer that used the dev index as an age would pass every mart-shaped test
and evaluate a quarterly triangle at 4x the true age (the compartmental
scorer's lesson, kept here).

The density is on the **loss-ratio** measure - the model's own scale. Carrying
it to Lebesgue-on-amount (``- log premium``) is ``ScoresHeldout.log_lik_at``'s
job, once, so this file cannot disagree with the other entries about the
Jacobian. The draws, by contrast, are scaled to AMOUNTS here: draws carry no
measure declaration, and the panel scores against the triangle's amounts.
"""

from __future__ import annotations

import numpy as np

from ibnr.gallery.statistical.clark.model import GROWTH_CURVES, growth
from ibnr.kernels.densities import lognormal_lpdf
from ibnr.kernels.holdout import CellIndex

__all__ = ["draw_cells", "log_lik_cells", "mu_cells", "sigma_cells"]

#: posterior variables this scorer needs. ``ulr`` is a Stan transformed
#: parameter (the ports must re-expose it as a deterministic); the rest are
#: sampled sites with identical names in every backend.
REQUIRED_DRAWS: tuple[str, ...] = ("ulr", "omega", "theta", "sigma")


def mu_cells(
    contract: dict, post: dict[str, np.ndarray], cells: CellIndex, *, curve: str
) -> np.ndarray:
    """``(n_draws, n_cells)`` lognormal location: ``log(ulr[w] * G(t))``.

    ``ulr[w]`` must be strictly positive: every retained posterior draw
    satisfies that for every trained origin (a non-positive ``ulr`` makes the
    training ``mu`` NaN and the proposal is rejected), and held-out cells only
    ever index trained origins (``new_origin`` is excluded upstream). A
    non-positive value reaching here is therefore a malformed posterior, and
    refusing beats returning NaN that pandas would silently drop.
    """
    _require(post, curve)
    w0 = np.asarray(cells.w, dtype=int) - 1  # contract w/d are 1-based (Stan)
    t = _t_years(contract, cells)

    ulr = np.asarray(post["ulr"], dtype=float)[:, w0]
    if np.any(ulr <= 0):
        raise ValueError(
            f"{int(np.sum(np.any(ulr <= 0, axis=0)))} cell(s) index an accident year with "
            "non-positive ulr draws; the sampler rejects such draws for every trained "
            "origin, so this posterior cannot have come from a fit of these cells"
        )
    omega = np.asarray(post["omega"], dtype=float).reshape(-1, 1)
    theta = np.asarray(post["theta"], dtype=float).reshape(-1, 1)
    return np.log(ulr * growth(t[None, :], omega, theta, curve))


def log_lik_cells(
    contract: dict, post: dict[str, np.ndarray], cells: CellIndex, *, curve: str
) -> np.ndarray:
    """``(n_draws, n_cells)`` log density of the loss RATIO, per ``model.stan``.

    The observed ratio is formed exactly as ``fit()`` forms Stan's ``y``: the
    cell's cumulative value over the CONTRACT's per-origin premium, the same
    number Stan divided by - the agreement gate compares elementwise. A
    non-positive cell is refused, the same rule the contract applies at fit
    time: the lognormal has no density there.
    """
    ratio = _observed_ratios(contract, cells)
    mu = mu_cells(contract, post, cells, curve=curve)
    return lognormal_lpdf(ratio[None, :], mu, sigma_cells(post, cells))


def draw_cells(
    contract: dict,
    post: dict[str, np.ndarray],
    cells: CellIndex,
    *,
    rng: np.random.Generator,
    curve: str,
) -> np.ndarray:
    """``(n_draws, n_cells)`` draws of **cumulative** loss AMOUNTS at the cells.

    One draw per posterior draw - the posterior predictive, carrying parameter
    uncertainty and process noise; a plug-in at the posterior mean would be
    systematically too sharp. The ratio draw is scaled by premium here because
    draws, unlike densities, carry no measure declaration: the panel scores
    against the triangle's amounts, so the draws must arrive in those units.
    Cumulative because ``y`` is the CUMULATIVE paid loss ratio, hence the
    entry's ``heldout_draw_scale = "cumulative"`` and ``predict_at``'s
    pass-through on the (cumulative) Schedule P triangles.

    Unlike :func:`log_lik_cells` this never reads ``cells.value``: it is a
    forecast, not an evaluation, so a cohort with a zero-paid held-out cell can
    still be CRPS-scored where it cannot be ELPD-scored - exactly why the two
    capabilities are separate mixins.
    """
    mu = mu_cells(contract, post, cells, curve=curve)
    premium = np.asarray(contract["premium"], dtype=float)[np.asarray(cells.w, dtype=int) - 1]
    return premium[None, :] * rng.lognormal(mu, sigma_cells(post, cells))


def sigma_cells(post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` lognormal scale: one ``sigma`` per draw, shared by
    every cell (the post's model has a single residual scale - no per-dev
    ``sig[d]`` like the Meyers entries).

    One expression, used by both the density and the draws, so the two cannot
    disagree about the scale a cell is evaluated at.
    """
    sigma = np.asarray(post["sigma"], dtype=float).reshape(-1, 1)
    return np.broadcast_to(sigma, (sigma.shape[0], cells.n_cells))


def _require(post: dict[str, np.ndarray], curve: str) -> None:
    if curve not in GROWTH_CURVES:
        raise ValueError(f"curve must be one of {GROWTH_CURVES}, got {curve!r}")
    missing = [name for name in REQUIRED_DRAWS if name not in post]
    if missing:
        raise KeyError(f"posterior is missing {missing}; have {sorted(post)}")


def _t_years(contract: dict, cells: CellIndex) -> np.ndarray:
    """Development age in YEARS per cell, from the contract's grain -
    ``d * dev_grain_months / 12``, identical to ``fit()``'s assembly of Stan's
    ``t``. Derived from the contract and never read off the cells (see the
    module docstring)."""
    return np.asarray(cells.d, dtype=float) * float(contract["dev_grain_months"]) / 12.0


def _observed_ratios(contract: dict, cells: CellIndex) -> np.ndarray:
    """The observed loss ratio per cell: cumulative value over the CONTRACT's
    per-origin premium - exactly ``fit()``'s ``y``, cell by cell.

    **The two premium sources must agree, and that is checked here** (the
    compartmental scorer's rule): the ratio divides by the contract's premium
    while the base class's measure carry divides by the CELLS' premium (the
    holdout frame's, when ``next_diagonal`` attached one). Both are the
    training slice's booked value, so today they cannot differ; if they ever
    did, the carried density would silently stop integrating to 1 - a wrong
    Jacobian, the bug class nothing downstream can see. Cells carrying no
    premium (NaN) are exempt: the carry has its own refusal for those.
    """
    value = np.asarray(cells.value, dtype=float)
    if np.any(value <= 0):
        raise ValueError(
            f"{int(np.sum(value <= 0))} cell(s) have non-positive loss; the lognormal "
            "likelihood cannot score them (the same rule the contract applies at fit time)"
        )
    premium = np.asarray(contract["premium"], dtype=float)[np.asarray(cells.w, dtype=int) - 1]
    cell_premium = np.asarray(cells.premium, dtype=float)
    mismatched = ~np.isnan(cell_premium) & ~np.isclose(cell_premium, premium)
    if mismatched.any():
        raise ValueError(
            f"{int(mismatched.sum())} cell(s) carry a premium that disagrees with the "
            "fitted contract's per-origin premium. The observed ratio divides by the "
            "contract's number while the measure carry divides by the cells', so a "
            "mismatch would produce a density that silently no longer integrates to 1"
        )
    return value / premium
