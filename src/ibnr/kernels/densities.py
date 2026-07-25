"""Model-free log densities, and the one place a density changes measure.

Two jobs, and the second is the reason this module exists.

**The catalogue.** Closed-form log densities the gallery's Bayesian entries
evaluate at held-out cells: normal, lognormal, and the over-dispersed Poisson
quasi-likelihood England & Verrall use. Plain numpy in, plain numpy out - no
sampler, no PPL, no entry-specific anything.

**The measure change.** The five Bayesian entries do not put their densities on
the same scale, and none of them applies a Jacobian:

===========================  ==========================================
entry                        density is of
===========================  ==========================================
meyers_ccl, meyers_csr       ``log`` cumulative loss
england_verrall_odp, clark   incremental loss, on an ODP **lattice**
compartmental (gaussian)     outstanding and cumulative paid **amounts**
compartmental (lognormal)    outstanding and incremental paid **ratios**
===========================  ==========================================

So the numbers in each entry's ``log_likelihood`` group are not comparable
across entries as they stand - the compartmental lognormal Stan file says so
about its own two variants. Ethan's milestone-6 call is that the leaderboard
publishes a **single global ELPD ranking**, which means every density has to be
carried to one common measure first. :func:`to_amount_scale` is that carry, and
it is deliberately the only place it happens: an entry that applied its own
Jacobian would be a second implementation of a thing that must not disagree.

The common measure is **Lebesgue on the loss amount**. Getting there:

``amount``
    Nothing to do.
``log_amount``
    ``y = exp(u)`` with a density on ``u``, so ``log p(y) = log p(u) - log y``.
``loss_ratio``
    ``y = r * premium`` with a density on ``r``, so subtract ``log premium``.
``odp_lattice``
    The ODP quasi-likelihood is a probability **mass** on ``{0, phi, 2phi, ...}``
    - it is ``Poisson(mu/phi)`` evaluated at ``x/phi``. Mass to density is mass
    divided by the lattice spacing, so subtract ``log phi``.

The increment/cumulative step needs **no** Jacobian: ``X = C - C_prev`` with
``C_prev`` sitting on the training diagonal, so it is data and ``dC/dX = 1``.
That is what lets an entry modelling cumulative loss and one modelling
increments meet on this scale at all.

**A caveat to carry into the model cards, because it is a real limitation
rather than a rounding error.** The ODP conversion is a convention, not a
theorem. Dividing a lattice mass by its spacing is the natural density
approximation and it is what makes the units commensurable, but the ODP
entries' likelihood remains a discretization, so their converted ELPD is not
quite the same kind of object as a genuinely continuous one. The alternative
considered was to give them their own comparability group and never sum across
- rejected in favour of one global ranking.

**On testing this.** A wrong Jacobian does not look like a wrong answer. It
looks like a differently-shaped model: the ranking still ranks, the numbers
still look like log densities, and every value-only comparison passes. The only
thing that catches it is normalization - :func:`check_normalization` - so every
measure in this module has a test that integrates its converted density to 1,
and those tests are verified to fail when the conversion is removed.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import gammaln

__all__ = [
    "MEASURES",
    "check_normalization",
    "lognormal_lpdf",
    "normal_lpdf",
    "odp_lpdf",
    "to_amount_scale",
]

_LOG_2PI = float(np.log(2.0 * np.pi))

#: every measure a gallery density may declare, and the covariate its carry to
#: the amount scale needs. ``None`` means the density is already there.
MEASURES: dict[str, str | None] = {
    "amount": None,
    "log_amount": "value",
    "loss_ratio": "premium",
    "odp_lattice": "phi",
}


def normal_lpdf(y, mu, sigma) -> np.ndarray:
    """log N(y | mu, sigma). Full normalization kept, no dropped constants."""
    y, mu, sigma = np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    z = (y - mu) / sigma
    return -0.5 * (z * z + _LOG_2PI) - np.log(sigma)


def lognormal_lpdf(y, mu_log, sigma) -> np.ndarray:
    """log density **of y**, where ``log y ~ N(mu_log, sigma)``.

    Includes the ``-log y`` term, i.e. this is already on the amount scale -
    matching Stan's ``lognormal_lpdf`` and unlike a ``normal_lpdf`` applied to
    ``log y``, which is a density on the log scale and needs ``log_amount``.
    """
    y = np.asarray(y, float)
    if np.any(y <= 0):
        raise ValueError("lognormal_lpdf needs strictly positive y")
    return normal_lpdf(np.log(y), mu_log, sigma) - np.log(y)


def odp_lpdf(y, mu, phi) -> np.ndarray:
    """Stan's ``odp_lpdf``, term for term::

        (y/phi) log(mu/phi) - mu/phi - lgamma(y/phi + 1)

    which is exactly ``Poisson(mu/phi).log_prob(y/phi)``.

    The parameter-free ``-lgamma(y/phi + 1)`` is KEPT. It cannot move a
    posterior, but dropping it shifts every value by a constant and breaks ELPD
    comparability with the Stan reference - which does not drop it either,
    because ``~``'s constant-dropping applies to built-in distributions, not to
    a user-defined ``_lpdf``.

    This is a probability MASS on the lattice ``{0, phi, 2phi, ...}``. Use
    ``to_amount_scale(..., measure="odp_lattice", phi=phi)`` before comparing it
    with a continuous density.
    """
    y, mu, phi = np.asarray(y, float), np.asarray(mu, float), np.asarray(phi, float)
    if np.any(phi <= 0):
        raise ValueError("phi must be positive")
    if np.any(mu <= 0):
        raise ValueError("odp_lpdf needs strictly positive mu")
    if np.any(y < 0):
        raise ValueError("odp_lpdf needs non-negative y")
    scaled = y / phi
    return scaled * np.log(mu / phi) - mu / phi - gammaln(scaled + 1.0)


def to_amount_scale(
    log_density,
    *,
    measure: str,
    value=None,
    premium=None,
    phi=None,
) -> np.ndarray:
    """Carry a log density to Lebesgue-on-the-loss-amount.

    log_density: values as the model reports them, on ``measure``.
    measure:     one of :data:`MEASURES`.
    value:       the observed loss amount. Required by ``log_amount``.
    premium:     exposure at the cell. Required by ``loss_ratio``.
    phi:         ODP dispersion, the lattice spacing. Required by ``odp_lattice``.

    The covariate a measure does not use may not be supplied. That is not
    pedantry: a covariate accepted and ignored is a parameter that looks
    connected and is not, and it would silently produce an unconverted density
    that still ranks.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {sorted(MEASURES)}, got {measure!r}")

    supplied = {"value": value, "premium": premium, "phi": phi}
    needed = MEASURES[measure]
    for name, given in supplied.items():
        if given is not None and name != needed:
            raise ValueError(
                f"measure={measure!r} does not use {name!r}"
                + (f"; it needs {needed!r}" if needed else "; it needs no covariate")
            )
    if needed is not None and supplied[needed] is None:
        raise ValueError(f"measure={measure!r} needs {needed!r} to change measure")

    out = np.asarray(log_density, float)
    if measure == "amount":
        return out
    covariate = np.asarray(supplied[needed], float)
    if np.any(covariate <= 0):
        raise ValueError(f"{needed!r} must be positive to take its log (got a non-positive value)")
    # every carry is a division by |dy/du|, hence a subtraction of its log
    return out - np.log(covariate)


def check_normalization(
    logpdf,
    *,
    lo: float,
    hi: float,
    tol: float = 1e-4,
    lattice: float | None = None,
) -> float:
    """Integrate ``exp(logpdf)`` and check it is 1. Returns the mass found.

    ``logpdf`` takes an array of points and returns log densities at them.
    ``lattice`` switches to a spacing-weighted sum for a converted lattice mass,
    where the "integral" is ``sum(exp(logpdf) * spacing)``.

    This is the only check that catches a wrong change of variable. A missing or
    sign-flipped Jacobian leaves a function that is still smooth, still
    unimodal, still orders points the same way, and integrates to something
    other than 1.
    """
    if lattice is not None:
        points = np.arange(lo, hi, lattice)
        mass = float(np.sum(np.exp(logpdf(points)) * lattice))
    else:
        mass = float(integrate.quad(lambda x: float(np.exp(logpdf(np.array([x])))[0]), lo, hi)[0])
    if not abs(mass - 1.0) <= tol:
        raise AssertionError(f"density integrates to {mass:.6f}, not 1 (tol {tol})")
    return mass
