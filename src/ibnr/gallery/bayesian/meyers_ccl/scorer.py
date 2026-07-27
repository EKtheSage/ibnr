"""CCL's log density at arbitrary cells, in plain numpy.

The counterpart of ``model.stan``'s ``generated quantities`` block, except that
it evaluates at cells the fit never saw. Written literally, next to the Stan
program it mirrors, per CLAUDE.md decision 6 - and deliberately as a free
function taking arrays rather than a method on the entry, so it is
backend-blind and unit-testable without a sampler.

``model.stan``, lines 56-61 and 74::

    mu[i] = logprem[i] + logelr + alpha[w[i]] + beta[d[i]];
    if (prev_idx[i] > 0) {
      mu[i] += rho * (logloss[prev_idx[i]] - mu[prev_idx[i]]);
    }
    ...
    log_lik[i] = normal_lpdf(logloss[i] | mu[i], sig[d[i]]);

which is what :func:`log_lik_cells` computes, for every posterior draw at once.

**The AR term is what separates this scorer from CSR's.** A cell's location
depends on the previous *origin*'s residual at the same dev lag - ``rho *
(logloss[w-1, d] - mu[w-1, d])`` - and both pieces of that residual are
training-row quantities: ``logloss`` comes from the contract and ``mu`` from
the posterior, where all three backends expose it (Stan saves transformed
parameters; both ports declare ``mu`` a deterministic over the same contract
rows). Reading the fit's own ``mu`` rather than re-running the recurrence
means the residual is exactly the one the model conditioned on, and the
agreement gate can hold elementwise.

**No leakage, and no rollout.** A held-out cell on the next diagonal sits at
``(w, d)``; its predecessor ``(w - 1, d)`` sits on the last *training*
diagonal, so the residual is data the fit already had - the same fact that
makes ``prev_value`` safe for the increment carry. Scoring two diagonals ahead
would need the unobserved diagonal between, which is why
``kernels.holdout`` stops at one.

**A missing predecessor drops the AR term, exactly as the fitted model does.**
``kernels.contract`` sets ``prev_idx = 0`` both for ``w == 1`` and for a
``(w - 1, d)`` hole, and ``model.stan`` then skips the adjustment - so a cell
in that position has density ``normal(base, sig[d])`` by the model's own
definition, and the scorer mirrors it rather than inventing a stricter rule.

Every quantity read here - ``alpha``, ``beta``, ``rho``, ``sig``, ``mu`` - is
a Stan ``transformed parameter``, and both ports re-expose all five as
deterministics (``model_numpyro.py:89-107``, ``model_pymc.py:78-96``). So this
reads ``posterior`` identically in all three backends, which the
``log_likelihood`` group does not: Stan calls it ``log_lik`` and the ports call
it ``obs``, and NumPyro adds scalar ``*_prior`` factor sites to it.

The value returned is a density on **log** cumulative loss - CCL's own
measure. Carrying it to the amount scale is ``ScoresHeldout.log_lik_at``'s
job, once, so that this file cannot disagree with the other entries about the
Jacobian.
"""

from __future__ import annotations

import numpy as np

from ibnr.kernels.densities import normal_lpdf
from ibnr.kernels.holdout import CellIndex

__all__ = ["draw_cells", "log_lik_cells", "mu_cells"]

#: posterior variables this scorer needs, in every backend. ``mu`` is the
#: fit's own location at the TRAINING rows - the AR residual reads it at the
#: predecessor, so the scorer conditions on exactly what the model did.
REQUIRED_DRAWS: tuple[str, ...] = ("logelr", "alpha", "beta", "rho", "sig", "mu")


def mu_cells(contract: dict, post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` lognormal location, per ``model.stan:56-61``."""
    missing = [name for name in REQUIRED_DRAWS if name not in post]
    if missing:
        raise KeyError(f"posterior is missing {missing}; have {sorted(post)}")

    w0 = cells.w - 1  # the contract's w/d are 1-based, matching the Stan data block
    d0 = cells.d - 1
    logprem = np.log(np.asarray(contract["premium"], dtype=float))[w0]

    logelr = np.asarray(post["logelr"], dtype=float).reshape(-1, 1)
    alpha = np.asarray(post["alpha"], dtype=float)[:, w0]
    beta = np.asarray(post["beta"], dtype=float)[:, d0]
    base = logprem[None, :] + logelr + alpha + beta

    # rho * (logloss[w-1, d] - mu[w-1, d]): the previous ORIGIN's residual at
    # the same dev, resolved against the TRAINING rows - the same lookup
    # contract.py encodes as prev_idx, extended to cells outside the fit.
    # 0-based here; -1 marks "no predecessor", which is model.stan's
    # prev_idx == 0 branch: the AR term is skipped, never approximated.
    row_of = {
        (int(a), int(b)): i
        for i, (a, b) in enumerate(
            zip(
                np.asarray(contract["w"], dtype=int),
                np.asarray(contract["d"], dtype=int),
                strict=True,
            )
        )
    }
    prev_row = np.array(
        [row_of.get((int(a) - 1, int(b)), -1) for a, b in zip(cells.w, cells.d, strict=True)],
        dtype=int,
    )
    has_prev = prev_row >= 0
    if not has_prev.any():
        return base

    rho = np.asarray(post["rho"], dtype=float).reshape(-1, 1)
    logloss = np.asarray(contract["logloss"], dtype=float)
    mu_train = np.asarray(post["mu"], dtype=float)
    if mu_train.shape[1] != logloss.shape[0]:
        raise ValueError(
            f"posterior mu has {mu_train.shape[1]} columns but the contract has "
            f"{logloss.shape[0]} training rows; the AR residual gathers mu by training-row "
            "index, so a mismatched width would read the wrong rows"
        )
    # clip the -1 sentinels to a real index for the gather, then mask the
    # result: fancy-indexing with -1 would silently wrap to the LAST row
    safe = np.where(has_prev, prev_row, 0)
    residual = np.where(has_prev[None, :], logloss[safe][None, :] - mu_train[:, safe], 0.0)
    return base + rho * residual


def log_lik_cells(contract: dict, post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` log density of ``log(loss)``, per ``model.stan:74``."""
    value = np.asarray(cells.value, dtype=float)
    if np.any(value <= 0):
        raise ValueError(
            f"{int(np.sum(value <= 0))} cell(s) have non-positive loss; CCL is lognormal "
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

    The sampling counterpart of :func:`log_lik_cells`, and the same lines of
    ``model.stan``: ``logloss[i] ~ normal(mu[i], sig[d[i]])`` read forwards
    instead of backwards, so a draw is ``exp(normal(mu, sig))``.

    **Per-cell conditionals, not a joint rollout - and on the next diagonal
    that is exact.** CCL's dependence runs down a dev column (each origin's
    ``mu`` reads the residual of the origin above it at the *same* dev), and a
    diagonal never contains two cells in the same column, so no held-out
    cell's location depends on another held-out cell's outcome. Every residual
    entering ``mu`` here is a training-row quantity, which makes the draws at
    one diagonal's cells conditionally independent given the posterior draw -
    the model's own joint predictive, not an approximation of it. (Scoring
    deeper than one diagonal would break this, and ``kernels.holdout``
    deliberately refuses to go there.)

    **One draw per posterior draw**, paired row for row with ``mu`` and
    ``sig``. That is the posterior predictive: it carries parameter
    uncertainty *and* process noise. Drawing repeatedly from the posterior
    mean instead would give a plug-in predictive that is systematically too
    sharp - narrower intervals, a better-looking CRPS, and nothing about the
    output that says so.

    Cumulative because CCL's ``logloss`` is ``log`` of the cumulative loss,
    which is why the entry declares ``heldout_draw_scale = "cumulative"``. On
    the Schedule P triangles that matches the triangle's own basis, so
    ``PredictsHeldout.predict_at`` passes these through unchanged.

    Unlike :func:`log_lik_cells` this does **not** need ``cells.value``: it is
    a forecast, not an evaluation. So it does not inherit that function's
    non-positive-loss refusal - a cohort whose held-out cell is zero still has
    a perfectly well-defined lognormal predictive, it just has no lognormal
    *density* at the outcome. An entry can therefore be CRPS-scorable on a
    cohort where it is not ELPD-scorable, which is exactly why the two
    capabilities are separate mixins.
    """
    mu = mu_cells(contract, post, cells)
    sig = _sig_cells(post, cells)
    return np.exp(rng.normal(mu, sig))


def _sig_cells(post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` lognormal scale, ``sig[d]`` per ``model.stan``.

    One expression, used by both the density and the draws, so the two cannot
    disagree about which development lag a cell reads its scale at. An index
    slip here shows up in a density as a mis-scored cell and in draws as a
    mis-calibrated one, and reading them off separate lines is how those two
    stay consistent with each other while both being wrong.
    """
    return np.asarray(post["sig"], dtype=float)[:, cells.d - 1]
