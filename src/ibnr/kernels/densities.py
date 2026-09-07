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
about its own two variants. Carrying them to one common measure is what makes a
cross-entry log score meaningful at all, and :func:`to_amount_scale` is
deliberately the only place it happens: an entry applying its own Jacobian would
be a second implementation of something that must not disagree.

The common measure is **Lebesgue on the loss amount**. Getting there:

``amount``
    Nothing to do.
``log_amount``
    ``y = exp(u)`` with a density on ``u``, so ``log p(y) = log p(u) - log y``.
``loss_ratio``
    ``y = r * premium`` with a density on ``r``, so subtract ``log premium``.

The increment/cumulative step needs **no** Jacobian: ``X = C - C_prev`` with
``C_prev`` sitting on the training diagonal, so it is data and ``dC/dX = 1``.
That is what lets an entry modelling cumulative loss and one modelling
increments meet on this scale at all.

.. _odp-not-a-density:

**Why there is no ``odp_lattice`` measure.** An earlier version of this module
carried the ODP quasi-likelihood to the amount scale by subtracting
``log phi``, on the reasoning that it is a probability mass on the lattice
``{0, phi, 2phi, ...}`` and mass-to-density is mass divided by spacing. **That
is wrong, and it was wrong in a way the first round of tests could not see.**

``exp(odp_lpdf(x | mu, phi)) / phi`` does not integrate to 1 over ``x``.
Substituting ``z = x/phi`` leaves ``\\int lambda^z e^{-lambda} / Gamma(z+1) dz``
with ``lambda = mu/phi``, and that integral is not 1 - it is a function of
``lambda``:

======================  =========================================
``lambda = mu / phi``   ``\\int`` of the carried "density"
======================  =========================================
0.5                     0.688
1.0                     0.834
2.0                     0.947
5.0                     0.998
20 and above            1.000 (to 6 dp)
======================  =========================================

Two things follow, and the second is the fatal one. First, it is not a density.
Second, **the defect varies with** ``mu/phi``, so it is not even a constant
offset that cancels when two models are compared on the same cells - it would
tilt a ranking towards whichever model happens to put more mass in low-``mu``
cells, which on a reserving triangle means the tail.

The deeper reason is that ODP is a *quasi*-likelihood: with non-integer
``x/phi`` it is Poisson only up to proportionality (England & Verrall, section
2.3.5), so it never was a normalized predictive law and no change of variable
can make it one. Giving it one means declaring an actual distribution -
negative binomial or Tweedie are the usual choices - which is a modelling
decision, not a units conversion.

So ``odp_lpdf`` stays here as a **likelihood** (the entries need it, and its
Stan-identical form is pinned by test), but ODP and Clark are **not
ELPD-eligible** and are scored by CRPS and PIT only until they are given a
proper predictive distribution.

**On testing this - the lesson that produced the paragraph above.** A wrong
Jacobian does not look like a wrong answer. It looks like a differently-shaped
model: the ranking still ranks, the numbers still look like log densities, and
every value-only comparison passes. Normalization is the only thing that sees
it, so every measure here integrates to 1 in a test, and those tests are
verified to fail when the carry is dropped or its sign flipped.

The original ODP test *looked* like exactly that check and was not. It summed
``exp(.) * phi`` over the lattice points, which recovers the Poisson pmf sum and
is 1 by construction for every ``mu`` and ``phi``. It confirmed a true and
irrelevant statement: real losses are not on the lattice, so the sum was never
the quantity that had to be 1. When adding a measure, integrate over the
**observation space the data actually lives in**, not over a grid chosen to make
the arithmetic come out.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = [
    "MEASURES",
    "POISSON_RATE_MAX",
    "check_normalization",
    "lognormal_lpdf",
    "normal_lpdf",
    "odp_draw",
    "odp_lpdf",
    "to_amount_scale",
]

_LOG_2PI = float(np.log(2.0 * np.pi))

_INT64_MAX = float(np.iinfo(np.int64).max)

#: the largest rate ``numpy.random.Generator.poisson`` will draw at. It is
#: numpy's own ``POISSON_LAM_MAX``, which numpy does not export, so it is
#: rebuilt from its definition rather than copied as a digit string:
#: ``int64max - 10 * sqrt(int64max)``, about 9.2233720064847708e18.
#:
#: Writing it out by hand is how this went wrong once already. An earlier copy
#: in ``kernels/odp_bootstrap.py`` said ``9.223372036854776e18``, the int64
#: maximum itself, which is about 30 billion too high: every rate in between
#: passed the check and then made numpy raise anyway.
#: ``tests/test_odp_bootstrap.py::test_poisson_rate_cap_is_numpys`` finds the
#: real boundary by bisection and refuses anything else.
POISSON_RATE_MAX: float = _INT64_MAX - 10.0 * float(np.sqrt(_INT64_MAX))

#: every measure a gallery density may declare, and the covariate its carry to
#: the amount scale needs. ``None`` means the density is already there.
#:
#: Only genuine changes of variable belong here. A family that is not a
#: normalized density on ANY scale does not get an entry - see
#: :ref:`odp-not-a-density`.
MEASURES: dict[str, str | None] = {
    "amount": None,
    "log_amount": "value",
    "loss_ratio": "premium",
}

#: measures that were tried and rejected, mapped to why. Named rather than
#: deleted so the reasoning is discoverable from the error a caller gets.
REJECTED_MEASURES: dict[str, str] = {
    "odp_lattice": (
        "the ODP quasi-likelihood is not a normalized density: exp(odp_lpdf)/phi "
        "integrates to 0.69 at mu/phi=0.5 and 0.83 at mu/phi=1, and the defect "
        "VARIES with mu/phi so it does not cancel between models. ODP is Poisson "
        "only up to proportionality for non-integer x/phi (England & Verrall "
        "2.3.5). Score ODP/Clark with CRPS and PIT, or give them a proper "
        "predictive law (negative binomial, Tweedie) - which is a modelling "
        "decision, not a change of variable"
    ),
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

    **This is not a normalized density and cannot be made into one by a change
    of variable** - see :ref:`odp-not-a-density`. It is here because the ODP
    entries need their likelihood, not because their ELPD is computable.
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


def odp_draw(rng: np.random.Generator, mu, phi) -> np.ndarray:
    """The over-dispersed Poisson draw ``X = phi * Poisson(mu / phi)``.

    Mean ``mu``, variance ``phi * mu`` - England & Verrall's process law, and
    the one every ODP entry in the gallery simulates with. One implementation,
    shared, because the three gallery entries and the bootstrap kernel wrote
    the same expression seven times - twice in each entry and once in the
    kernel - and only the kernel handled its one hard edge.

    Returns an array shaped like ``mu``. ``rng`` is consumed only for the cells
    that are actually drawn, so a fit with nothing at the limit reads exactly
    the random numbers it always did.

    **The limit.** numpy refuses a Poisson rate above :data:`POISSON_RATE_MAX`,
    and ``mu / phi`` runs past it exactly when the over-dispersion collapses
    against the mean. There the draw's coefficient of variation
    ``sqrt(phi / mu) = 1 / sqrt(rate)`` is below 3.3e-10: the law is a point
    mass to any precision that matters, so those cells come back at ``mu``
    exactly. ``phi == 0`` is the same statement with no dispersion left at all
    and returns a copy of ``mu``. Both follow the convention
    ``kernels.mack.draw_step`` already documents for a zero-variance step.

    How far the collapse has to go, measured rather than asserted: a triangle
    that develops exactly on its own fitted curve fits itself to rounding
    error, and the Pearson scale falls to about 1e-29 while the means stay in
    the thousands, which puts the rate 4e11 times past the limit
    (``tests/test_clark.py`` and ``tests/test_odp_bootstrap.py`` both build one
    and pin the answer). Any noise at all keeps a fit well clear: the same
    triangle with its amounts rounded to whole units sits at 1e-14 of the
    limit, and with a relative noise of one part in a million, at 1e-8 of it.
    So this is the answer for a degenerate fit, not a branch an ordinary one
    takes.

    **The refusals.** A mean that is not finite never came from the model: it
    is a parameter sample that overflowed before any draw was asked for, and it
    is named as such rather than left for numpy to report as a rate problem. A
    negative mean is a different defect - a sign the caller was supposed to
    handle, by flooring the mean (the gallery entries use
    ``clark.scorer.MU_FLOOR``) or by reflecting it (the bootstrap kernel) - so
    it is refused separately and its message says which. Same for a negative or
    non-finite ``phi``. All the checks run before the ``phi == 0`` shortcut, so
    a broken mean is refused whatever the dispersion is.

    A mean of exactly 0 is none of those things: it is a legal Poisson rate,
    drawn like any other and answering 0.
    """
    mu = np.asarray(mu, dtype=float)
    phi = float(phi)

    not_finite = ~np.isfinite(mu)
    if not_finite.any():
        raise ValueError(
            f"odp_draw needs a finite mean at every cell: {int(not_finite.sum())} of "
            f"{mu.size} are not finite. A mean like that is a parameter sample that "
            "overflowed upstream, before any draw was asked for"
        )
    negative = mu < 0.0
    if negative.any():
        raise ValueError(
            f"odp_draw needs a non-negative mean at every cell: {int(negative.sum())} of "
            f"{mu.size} are negative, the smallest {float(mu.min())!r}. An over-dispersed "
            "Poisson draw has no negative mean: floor it (the gallery entries use "
            "clark.scorer.MU_FLOOR) or reflect its sign (the bootstrap kernel does) "
            "before asking for the draw"
        )
    if not np.isfinite(phi) or phi < 0.0:
        raise ValueError(f"odp_draw needs a finite, non-negative dispersion phi, got {phi!r}")

    out = mu.astype(float, copy=True)
    if phi == 0.0:
        return out  # no dispersion left: the mean, and no random numbers taken
    live = mu / phi <= POISSON_RATE_MAX
    if live.any():
        out[live] = phi * rng.poisson(mu[live] / phi)
    return out


def to_amount_scale(
    log_density,
    *,
    measure: str,
    value=None,
    premium=None,
) -> np.ndarray:
    """Carry a log density to Lebesgue-on-the-loss-amount.

    log_density: values as the model reports them, on ``measure``.
    measure:     one of :data:`MEASURES`.
    value:       the observed loss amount. Required by ``log_amount``.
    premium:     exposure at the cell. Required by ``loss_ratio``.

    The covariate a measure does not use may not be supplied. That is not
    pedantry: a covariate accepted and ignored is a parameter that looks
    connected and is not, and it would silently produce an unconverted density
    that still ranks.
    """
    if measure in REJECTED_MEASURES:
        raise ValueError(
            f"measure={measure!r} is not a valid change of variable: {REJECTED_MEASURES[measure]}"
        )
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {sorted(MEASURES)}, got {measure!r}")

    supplied = {"value": value, "premium": premium}
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


def check_normalization(logpdf, *, lo: float, hi: float, tol: float = 1e-4) -> float:
    """Integrate ``exp(logpdf)`` over ``[lo, hi]`` and check it is 1.

    ``logpdf`` takes an array of points and returns log densities at them.
    Returns the mass found, so a caller can report it.

    This is the only check that catches a wrong change of variable: a missing or
    sign-flipped Jacobian leaves a function that is still smooth, still
    unimodal, still orders points the same way, and integrates to something
    other than 1.

    It integrates over a **continuum**, deliberately, and there is no option to
    sum over a grid instead. An earlier version had a ``lattice=`` mode, and it
    is how the ODP defect in :ref:`odp-not-a-density` survived review: summing
    ``exp(.) * spacing`` over lattice points recovers the underlying pmf sum and
    returns 1.000000 for any parameters at all, whether or not the function is a
    density anywhere the data actually lives. A check that cannot fail is worse
    than no check, because it is quoted as evidence.
    """
    # Imported here, not at module scope: this module is on the import path of
    # every gallery entry (via ``gallery.entry``), while ``check_normalization``
    # is a test-time guard rail no production path calls. ``scipy.integrate``
    # drags ``scipy.optimize`` and ``scipy.sparse.linalg`` in behind it (its
    # ``_bvp`` submodule needs both), so at module scope it charges that to
    # every caller that only ever evaluates one of the lpdfs above.
    from scipy import integrate

    mass = float(integrate.quad(lambda x: float(np.exp(logpdf(np.array([x])))[0]), lo, hi)[0])
    if not abs(mass - 1.0) <= tol:
        raise AssertionError(f"density integrates to {mass:.6f}, not 1 (tol {tol})")
    return mass
