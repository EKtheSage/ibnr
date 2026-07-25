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

__all__ = ["log_lik_cells", "mu_cells"]

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
    sig = np.asarray(post["sig"], dtype=float)[:, cells.d - 1]
    return normal_lpdf(np.log(value)[None, :], mu, sig)
