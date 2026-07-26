"""CSR's log density at arbitrary cells, in plain numpy.

The counterpart of ``model.stan``'s ``generated quantities`` block, except that
it evaluates at cells the fit never saw. Written literally, next to the Stan
program it mirrors, per CLAUDE.md decision 6 - and deliberately as a free
function taking arrays rather than a method on the entry, so it is
backend-blind and unit-testable without a sampler.

``model.stan``, lines 58-61 and 76-79::

    mu[i] = logprem[i] + logelr + alpha[w[i]] + beta[d[i]] * speedup[w[i]];
    ...
    log_lik[i] = normal_lpdf(logloss[i] | mu[i], sig[d[i]]);

which is what :func:`log_lik_cells` computes, for every posterior draw at once.

Every quantity it reads - ``alpha``, ``beta``, ``speedup``, ``sig`` - is a Stan
``transformed parameter``, and both ports re-expose all four as deterministics
(``model_numpyro.py:92-109``, ``model_pymc.py``). So this reads ``posterior``
identically in all three backends, which the ``log_likelihood`` group does not:
Stan calls it ``log_lik`` and the ports call it ``obs``, and NumPyro adds
scalar ``*_prior`` factor sites to it.

**Nothing here extrapolates.** ``alpha[w]`` and ``sig[d]`` are read at indices
the fit already estimated; a cell needing ``alpha[n_w + 1]`` or ``sig[n_d + 1]``
is excluded upstream by ``kernels.holdout.next_diagonal``. This function is
therefore a re-evaluation of the fitted model, not a forecast of its parameters.

The value returned is a density on **log** cumulative loss - CSR's own measure.
Carrying it to the amount scale is ``ScoresHeldout.log_lik_at``'s job, once, so
that this file cannot disagree with the other entries about the Jacobian.
"""

from __future__ import annotations

import numpy as np

from ibnr.kernels.densities import normal_lpdf
from ibnr.kernels.holdout import CellIndex

__all__ = ["draw_cells", "log_lik_cells", "mu_cells"]

#: posterior variables this scorer needs, in every backend
REQUIRED_DRAWS: tuple[str, ...] = ("logelr", "alpha", "beta", "speedup", "sig")


def mu_cells(contract: dict, post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` lognormal location, per ``model.stan:58-61``."""
    missing = [name for name in REQUIRED_DRAWS if name not in post]
    if missing:
        raise KeyError(f"posterior is missing {missing}; have {sorted(post)}")

    w0 = cells.w - 1  # the contract's w/d are 1-based, matching the Stan data block
    d0 = cells.d - 1
    logprem = np.log(np.asarray(contract["premium"], dtype=float))[w0]

    logelr = np.asarray(post["logelr"], dtype=float).reshape(-1, 1)
    alpha = np.asarray(post["alpha"], dtype=float)[:, w0]
    beta = np.asarray(post["beta"], dtype=float)[:, d0]
    speedup = np.asarray(post["speedup"], dtype=float)[:, w0]
    return logprem[None, :] + logelr + alpha + beta * speedup


def log_lik_cells(contract: dict, post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` log density of ``log(loss)``, per ``model.stan:76-79``."""
    value = np.asarray(cells.value, dtype=float)
    if np.any(value <= 0):
        raise ValueError(
            f"{int(np.sum(value <= 0))} cell(s) have non-positive loss; CSR is lognormal "
            "and cannot score them (the same rule the contract applies at fit time)"
        )
    mu = mu_cells(contract, post, cells)
    sig = _sig_cells(post, cells)
    return normal_lpdf(np.log(value)[None, :], mu, sig)


def draw_cells(
    contract: dict,
    post: dict[str, np.ndarray],
    cells: CellIndex,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(n_draws, n_cells)`` draws of **cumulative** loss at the cells.

    The sampling counterpart of :func:`log_lik_cells`, and the same two lines of
    ``model.stan``: ``logloss[i] ~ normal(mu[i], sig[d[i]])`` read forwards
    instead of backwards, so a draw is ``exp(normal(mu, sig))``.

    **One draw per posterior draw**, paired row for row with ``mu`` and ``sig``.
    That is the posterior predictive: it carries parameter uncertainty *and*
    process noise. Drawing repeatedly from the posterior mean instead would give
    a plug-in predictive that is systematically too sharp - narrower intervals,
    a better-looking CRPS, and nothing about the output that says so.

    Cumulative because CSR's ``logloss`` is ``log`` of the cumulative paid loss,
    which is why the entry declares ``heldout_draw_scale = "cumulative"``. On the
    Schedule P triangles that matches the triangle's own basis, so
    ``PredictsHeldout.predict_at`` passes these through unchanged.

    Unlike :func:`log_lik_cells` this does **not** need ``cells.value``: it is a
    forecast, not an evaluation. So it does not inherit that function's
    non-positive-loss refusal - a cohort whose held-out cell is zero-paid still
    has a perfectly well-defined lognormal predictive, it just has no lognormal
    *density* at the outcome. An entry can therefore be CRPS-scorable on a cohort
    where it is not ELPD-scorable, which is exactly why the two capabilities are
    separate mixins.
    """
    mu = mu_cells(contract, post, cells)
    sig = _sig_cells(post, cells)
    return np.exp(rng.normal(mu, sig))


def _sig_cells(post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` lognormal scale, ``sig[d]`` per ``model.stan``.

    One expression, used by both the density and the draws, so the two cannot
    disagree about which development lag a cell reads its scale at. An index slip
    here shows up in a density as a mis-scored cell and in draws as a
    mis-calibrated one, and reading them off separate lines is how those two stay
    consistent with each other while both being wrong.
    """
    return np.asarray(post["sig"], dtype=float)[:, cells.d - 1]
