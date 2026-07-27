"""Clark MLE held-out draws at arbitrary cells, in plain numpy.

The statistical ``clark`` entry has no MCMC posterior, but it does have
genuine draws: parameter risk from the asymptotic MVN of the log parameters
(delta method, covariance ``phi * pinv(observed Hessian)`` - ``model.py``),
process risk as od-Poisson noise. That is exactly what ``predict()`` simulates
for the ultimates; this module is the same recipe factored out per cell, so
``predict()`` and ``PredictsHeldout.predict_at()`` cannot drift apart.

Draws only, deliberately - there is no ``log_lik_cells`` here and the entry
never subclasses ``ScoresHeldout``. The quasi-likelihood this entry maximizes
is not a normalized density on any scale (``kernels/densities.py``,
:ref:`odp-not-a-density`): ``exp(odp_lpdf)/phi`` integrates to 0.69 at
``mu/phi = 0.5`` and the defect varies with ``mu/phi``, so it cannot back an
ELPD. The entry is CRPS-scorable and permanently ELPD-ineligible.

Free functions over plain arrays per CLAUDE.md decision 6, testable without
fitting anything: the fitted state arrives as the entry's ``params_`` dict and
the ``kernels.contract`` dict, nothing else.
"""

from __future__ import annotations

import numpy as np

from ibnr.gallery.statistical.clark.model import GROWTH_CURVES, METHODS, age_interval, growth
from ibnr.kernels.holdout import CellIndex

__all__ = ["draw_cells", "mu_cells", "param_draws"]

#: what a parameter sample carries, the shape :func:`mu_cells` consumes:
#: ``level`` (n_draws, n_w) ultimates per origin, ``omega``/``theta``
#: (n_draws,) growth-curve parameters. :func:`param_draws` produces exactly
#: these from the entry's fitted MVN - this entry's stand-in for a posterior.
REQUIRED_DRAWS: tuple[str, ...] = ("level", "omega", "theta")

#: fitted-state keys :func:`draw_cells` needs from the entry's ``params_``
REQUIRED_PARAMS: tuple[str, ...] = ("growth_curve", "method", "log_params", "log_cov", "phi")

#: the floor ``predict()`` applies before every process draw, shared so the
#: held-out path cannot disagree. A deep-age growth increment can round to
#: exactly 0 or a hair below, and ``rng.poisson`` refuses a negative rate; at
#: the floor the draws are all zero, which is the model's own statement that
#: the cell has no expected emergence (the zero-variance bookkeeping that
#: follows is ``kernels/forecast.py``'s job, not a refusal here).
MU_FLOOR: float = 1e-12


def param_draws(
    contract: dict, params: dict, *, n_draws: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """The entry's parameter sample: ``model.py``'s MVN recipe, verbatim.

    Draw the whole log-parameter vector from its asymptotic MVN (mean = MLE,
    cov = the delta-method covariance), exponentiate back, and recover the
    per-origin ultimates the way the fitted ``method`` defines them: ``ldf``
    keeps a free ultimate per origin (``exp`` of the first ``n_w`` entries),
    ``cape_cod`` scales one ELR draw by each origin's premium.
    """
    missing = [k for k in REQUIRED_PARAMS if k not in params]
    if missing:
        raise KeyError(f"params is missing {missing}; have {sorted(params)}")
    method = params["method"]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    log_params = np.asarray(params["log_params"], dtype=float)
    log_cov = np.asarray(params["log_cov"], dtype=float)
    draws = rng.multivariate_normal(log_params, log_cov, size=n_draws)  # (n_draws, p)
    omega = np.exp(draws[:, -2])
    theta = np.exp(draws[:, -1])
    if method == "ldf":
        level = np.exp(draws[:, : int(contract["n_w"])])  # (n_draws, n_w)
    else:
        # cape_cod: single ELR draw scaled by each origin's premium.
        premium = np.asarray(contract["premium"], dtype=float)
        level = np.exp(draws[:, [0]]) * premium[None, :]  # (n_draws, n_w)
    return {"level": level, "omega": omega, "theta": theta}


def mu_cells(
    contract: dict, post: dict[str, np.ndarray], cells: CellIndex, *, curve: str
) -> np.ndarray:
    """``(n_draws, n_cells)`` od-Poisson mean ``U[w] * (G(age_hi) - G(age_lo))``.

    The cell's expected incremental emergence: the origin's ultimate times the
    growth-curve share falling in the cell's mid-period age interval - the same
    expression ``fit()`` maximized and ``predict()`` simulates, with the ages
    coming from the one shared :func:`~...model.age_interval`. Floored at
    :data:`MU_FLOOR`, exactly as ``predict()`` floors it.

    One function for the mean so any future density and the draws cannot
    disagree about what the model expects at a cell.
    """
    missing = [name for name in REQUIRED_DRAWS if name not in post]
    if missing:
        raise KeyError(f"parameter sample is missing {missing}; have {sorted(post)}")
    if curve not in GROWTH_CURVES:
        raise ValueError(f"curve must be one of {GROWTH_CURVES}, got {curve!r}")

    w0 = cells.w - 1  # the contract's w/d are 1-based, matching the Stan data block
    step = float(contract["dev_grain_months"])
    lo, hi = age_interval(cells.d, step)  # each (n_cells,)

    omega = np.asarray(post["omega"], dtype=float).reshape(-1, 1)
    theta = np.asarray(post["theta"], dtype=float).reshape(-1, 1)
    level = np.asarray(post["level"], dtype=float)[:, w0]  # (n_draws, n_cells)
    ginc = growth(hi[None, :], omega, theta, curve) - growth(lo[None, :], omega, theta, curve)
    return np.maximum(level * ginc, MU_FLOOR)


def draw_cells(
    contract: dict,
    params: dict,
    cells: CellIndex,
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(n_draws, n_cells)`` draws of **incremental** loss at the cells.

    Clark's own uncertainty decomposition, per cell: parameter risk from the
    MVN sample (:func:`param_draws`), process risk as the od-Poisson draw
    ``X = phi * Poisson(mu / phi)`` (mean ``mu``, variance ``phi * mu``) - the
    same two stages ``predict()`` runs for the ultimates.

    **One process draw per parameter draw**, so the output carries parameter
    uncertainty *and* process noise. Repeated process draws at the MLE would be
    a plug-in predictive: systematically too sharp, a better-looking CRPS, and
    nothing about the output that says so.

    Incremental because Clark models incremental emergence, which is why the
    entry declares ``heldout_draw_scale = "incremental"``:
    ``PredictsHeldout.predict_at`` adds each cell's training-diagonal anchor to
    reach the cumulative triangle's basis.

    ``n_draws`` is the caller's choice (the entry's ``n_heldout_draws``) - this
    entry has no posterior whose size decides it.
    """
    post = param_draws(contract, params, n_draws=n_draws, rng=rng)
    mu = mu_cells(contract, post, cells, curve=params["growth_curve"])
    phi = float(params["phi"])
    return phi * rng.poisson(mu / phi)
