"""Bayesian Clark's held-out draws at arbitrary cells, in plain numpy.

The sampling counterpart of ``model.stan``'s likelihood, evaluated at cells the
fit never saw. Written literally, next to the Stan program it mirrors, per
CLAUDE.md decision 6 - free functions over plain arrays, backend-blind,
testable without a sampler.

``model.stan``, lines 51-53 and 61::

    mu[i] = exp(logprem_w[w[i]] + logelr)
            * (growth_curve(age_hi[i], omega, theta, curve)
               - growth_curve(age_lo[i], omega, theta, curve));
    ...
    inc_loss[i] ~ odp(mu[i], phi);

:func:`mu_cells` is the first expression at arbitrary ``(w, d)``;
:func:`draw_cells` is the second read forwards - ``X = phi * Poisson(mu/phi)``
with the contract's plug-in Pearson ``phi`` from the MLE twin.

**Draws only - there is deliberately no ``log_lik_cells`` here.** The ODP
quasi-likelihood is not a normalized density on any scale
(``kernels/densities.py``, :ref:`odp-not-a-density`), so this entry never
subclasses ``ScoresHeldout``: it is CRPS-scorable and permanently
ELPD-ineligible.

The age convention and the growth curves are imported from the statistical
``clark`` entry - one shared :func:`age_interval` rather than a fifth copy of
the mid-period formula, because a copy drifting by ``step/2`` produces
perfectly plausible mis-scaled draws. The ``curve`` name is threaded in by the
entry (its ``_curve`` fitted state); the contract does not carry it.

``logelr``, ``omega`` and ``theta`` are the model's three sampled parameters,
exposed under the same names by Stan and both ports, so this reads
``posterior`` identically in every backend. (Never ``log_likelihood``: the
PyMC port attaches via ``pm.Potential`` and has no such group at all.)
"""

from __future__ import annotations

import numpy as np

from ibnr.gallery.statistical.clark.model import GROWTH_CURVES, age_interval, growth
from ibnr.gallery.statistical.clark.scorer import MU_FLOOR
from ibnr.kernels.holdout import CellIndex

__all__ = ["draw_cells", "mu_cells"]

#: posterior variables this scorer needs, in every backend
REQUIRED_DRAWS: tuple[str, ...] = ("logelr", "omega", "theta")


def mu_cells(
    contract: dict, post: dict[str, np.ndarray], cells: CellIndex, *, curve: str
) -> np.ndarray:
    """``(n_draws, n_cells)`` od-Poisson mean, per ``model.stan:51-53``.

    The Cape Cod expected increment: ``exp(logelr) * premium[w]`` times the
    growth-curve share falling in the cell's mid-period age interval. Floored
    at :data:`MU_FLOOR` exactly as ``predict()`` floors it - a deep-age growth
    increment can round to 0 or a hair below, and ``rng.poisson`` refuses a
    negative rate. A genuinely emergence-free cell then draws all zeros, which
    is the model's own statement (``model.stan``'s ``mu`` can be exactly 0
    there - the reason its ports attach via ``factor``/``Potential``); the
    zero-variance bookkeeping that follows is ``kernels/forecast.py``'s job.
    """
    missing = [name for name in REQUIRED_DRAWS if name not in post]
    if missing:
        raise KeyError(f"posterior is missing {missing}; have {sorted(post)}")
    if curve not in GROWTH_CURVES:
        raise ValueError(f"curve must be one of {GROWTH_CURVES}, got {curve!r}")

    w0 = cells.w - 1  # the contract's w/d are 1-based, matching the Stan data block
    step = float(contract["dev_grain_months"])
    lo, hi = age_interval(cells.d, step)  # each (n_cells,)
    logprem = np.log(np.asarray(contract["premium"], dtype=float))[w0]  # (n_cells,)

    logelr = np.asarray(post["logelr"], dtype=float).reshape(-1, 1)
    omega = np.asarray(post["omega"], dtype=float).reshape(-1, 1)
    theta = np.asarray(post["theta"], dtype=float).reshape(-1, 1)
    elr_prem = np.exp(logprem[None, :] + logelr)  # (n_draws, n_cells)
    ginc = growth(hi[None, :], omega, theta, curve) - growth(lo[None, :], omega, theta, curve)
    return np.maximum(elr_prem * ginc, MU_FLOOR)


def draw_cells(
    contract: dict,
    post: dict[str, np.ndarray],
    cells: CellIndex,
    *,
    curve: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(n_draws, n_cells)`` draws of **incremental** loss at the cells.

    ``model.stan:61`` read forwards: ``X = phi * Poisson(mu / phi)``, mean
    ``mu`` and variance ``phi * mu`` - the same process law ``predict()``
    simulates future increments with, and the same scaled-Poisson twin the MLE
    entry draws.

    **One draw per posterior draw**, paired row for row with ``mu``. That is
    the posterior predictive: parameter uncertainty *and* process noise.
    Repeated draws at the posterior mean would be a plug-in predictive -
    systematically too sharp, and nothing about the output says so.

    Incremental because Clark models incremental emergence, which is why the
    entry declares ``heldout_draw_scale = "incremental"``:
    ``PredictsHeldout.predict_at`` adds each cell's training-diagonal anchor
    to reach the cumulative triangle's basis.
    """
    if "phi" not in contract:
        raise KeyError(
            "contract has no 'phi'; the entry injects the MLE twin's Pearson dispersion "
            "at fit() time, and draws are undefined without it"
        )
    mu = mu_cells(contract, post, cells, curve=curve)
    phi = float(contract["phi"])
    return phi * rng.poisson(mu / phi)
