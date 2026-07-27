"""ODP's held-out draws at arbitrary cells, in plain numpy.

The sampling counterpart of ``model.stan``'s likelihood, evaluated at cells the
fit never saw. Written literally, next to the Stan program it mirrors, per
CLAUDE.md decision 6 - and deliberately as free functions taking arrays rather
than methods on the entry, so it is backend-blind and unit-testable without a
sampler.

``model.stan``, lines 51 and 59::

    log_mu[i] = logprem[i] + c + alpha[w[i]] + beta[d[i]];
    ...
    inc_loss[i] ~ odp(exp(log_mu[i]), phi);

:func:`mu_cells` is the first line at arbitrary ``(w, d)``; :func:`draw_cells`
is the second read forwards - the od-Poisson process draw
``X = phi * Poisson(mu / phi)`` with mean ``mu`` and variance ``phi * mu``,
the same law ``predict()`` simulates future increments with.

**Draws only - there is deliberately no ``log_lik_cells`` here.** The ODP
quasi-likelihood is not a normalized density on any scale
(``kernels/densities.py``, :ref:`odp-not-a-density`): ``exp(odp_lpdf)/phi``
integrates to 0.69 at ``mu/phi = 0.5`` and 0.83 at 1.0, and the defect varies
with ``mu/phi`` so it does not cancel between models. So this entry never
subclasses ``ScoresHeldout``: it is CRPS-scorable and permanently
ELPD-ineligible until given a proper predictive law (negative binomial,
Tweedie - a modelling decision, not a units conversion).

Every quantity read here - ``c``, ``alpha``, ``beta`` - is sampled or a Stan
``transformed parameter``, and both ports re-expose ``alpha``/``beta`` as
deterministics under the same names (``model_numpyro.py``, ``model_pymc.py``).
So this reads ``posterior`` identically in all three backends, which the
``log_likelihood`` group does not: Stan calls it ``log_lik`` and the ports
call it ``obs``.

**Per-cell conditional independence is exact here.** Given a posterior draw,
ODP increments are independent across cells (no AR term, unlike CCL), so one
independent draw per cell IS the model's joint predictive for a held-out
diagonal - no rollout, no approximation.
"""

from __future__ import annotations

import numpy as np

from ibnr.kernels.holdout import CellIndex

__all__ = ["draw_cells", "mu_cells"]

#: posterior variables this scorer needs, in every backend
REQUIRED_DRAWS: tuple[str, ...] = ("c", "alpha", "beta")


def mu_cells(contract: dict, post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` od-Poisson mean, per ``model.stan:51``."""
    missing = [name for name in REQUIRED_DRAWS if name not in post]
    if missing:
        raise KeyError(f"posterior is missing {missing}; have {sorted(post)}")

    w0 = cells.w - 1  # the contract's w/d are 1-based, matching the Stan data block
    d0 = cells.d - 1
    # PER-ORIGIN premium, never contract["logprem"]: that vector is per
    # TRAINING ROW (length len_data, the Stan data block's shape), so indexing
    # it by origin reads whatever row happens to sit at that position - wrong
    # everywhere the premiums differ, and silently right when they do not.
    logprem = np.log(np.asarray(contract["premium"], dtype=float))[w0]  # (n_cells,)

    const = np.asarray(post["c"], dtype=float).reshape(-1, 1)  # (n_draws, 1)
    alpha = np.asarray(post["alpha"], dtype=float)[:, w0]  # (n_draws, n_cells)
    beta = np.asarray(post["beta"], dtype=float)[:, d0]  # (n_draws, n_cells)
    return np.exp(logprem[None, :] + const + alpha + beta)


def draw_cells(
    contract: dict,
    post: dict[str, np.ndarray],
    cells: CellIndex,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(n_draws, n_cells)`` draws of **incremental** loss at the cells.

    ``model.stan:59`` read forwards: ``X = phi * Poisson(mu / phi)`` with the
    contract's plug-in Pearson ``phi`` (data, never a parameter - exactly as
    England & Verrall treat the scale, and exactly what ``predict()`` draws
    future increments with).

    **One draw per posterior draw**, paired row for row with ``mu``. That is
    the posterior predictive: it carries parameter uncertainty *and* process
    noise. Drawing repeatedly at the posterior mean would be a plug-in
    predictive - systematically too sharp, a better-looking CRPS, and nothing
    about the output that says so.

    Incremental because ODP models incremental losses, which is why the entry
    declares ``heldout_draw_scale = "incremental"``: on the cumulative
    Schedule P triangles ``PredictsHeldout.predict_at`` adds each cell's
    training-diagonal anchor, and an undeclared increment draw would be wrong
    by that whole anchor while staying finite and plausible.

    The draws are non-negative multiples of ``phi`` by construction. A
    held-out realized increment can still be negative (the same feature that
    rejects a cohort at fit time); CRPS is finite and well-defined there, so
    that is a large score rather than a refusal.
    """
    if "phi" not in contract:
        raise KeyError(
            "contract has no 'phi'; the entry injects the plug-in Pearson dispersion "
            "at fit() time (pearson_phi), and draws are undefined without it"
        )
    mu = mu_cells(contract, post, cells)
    phi = float(contract["phi"])
    return phi * rng.poisson(mu / phi)
