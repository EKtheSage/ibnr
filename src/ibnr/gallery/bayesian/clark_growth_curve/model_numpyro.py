"""Bayesian Clark growth curve - NumPyro port of the Stan reference (``model.stan``).

Same model, same parameterization, held constant for parity (design decision
7): the same Stan ``data`` block the entry assembles is consumed verbatim -
ages, the plug-in ``phi`` and the curve code included, so a port cannot drift
on the age convention or fit the other curve.

Two things need care here, and both are about *zero*.

**1. The growth curve at age 0.** Stan writes

    if (x <= 0) return 0;

and that branch is not an edge case - ``age_lo`` is exactly 0 for every
origin's first development cell, by construction (the mid-period shift clamps
at 0). A literal ``jnp.where(x > 0, formula, 0.0)`` gets the VALUE right and
the GRADIENT wrong: ``jnp.where`` evaluates both branches, so the NaN produced
inside the unselected one (``theta/0 -> inf`` for loglogistic, ``0**omega``
differentiated for Weibull) propagates through the reverse pass and poisons
every gradient. The fix is the standard double-``where``: substitute a safe
dummy age *inside* the formula as well as masking the result. Without it NUTS
diverges immediately on the very first leapfrog, with no error message.

**2. The od-Poisson quasi-likelihood** is Stan's own ``odp_lpdf``, identical to
the one in ``england_verrall_odp``. It is deliberately duplicated rather than
imported: ``model.stan`` duplicates it too, and each gallery model directory
has to stand alone so ``gallery.scaffold()`` can copy it into a user's project
(decision 6). See that entry's card for why the density is written literally
and why the parameter-free terms are kept.

``sample()`` returns an ``arviz.InferenceData`` whose ``posterior`` group
carries the same deterministic ``mu`` Stan puts in ``transformed parameters``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

#: Stan's integer curve codes (model.stan ``curve``), mirrored so the port
#: cannot silently fit the other curve.
LOGLOGISTIC, WEIBULL = 1, 2


def growth_curve(x, omega, theta, curve: int):
    """Clark's G(x): the fraction of ultimate emerged by age ``x`` months.

    ``loglogistic``  G = 1 / (1 + (theta/x)^omega)
    ``weibull``      G = 1 - exp(-(x/theta)^omega)

    with G(x) = 0 for x <= 0, as in ``model.stan``.

    ``curve`` is DATA (a Python int), so the branch is resolved at trace time
    and never enters the computation graph.

    The ``safe_x`` substitution is load-bearing, not defensive: see the module
    docstring. ``x`` is data, so the mask is static and costs nothing.
    """
    import jax.numpy as jnp

    positive = x > 0
    # A dummy age of 1.0 wherever x <= 0: keeps theta/x and x/theta finite so
    # the reverse pass sees no inf/NaN. The value is then masked out below, so
    # the choice of dummy is arbitrary as long as it is strictly positive.
    safe_x = jnp.where(positive, x, 1.0)
    if curve == LOGLOGISTIC:
        g = 1.0 / (1.0 + (theta / safe_x) ** omega)
    elif curve == WEIBULL:
        g = 1.0 - jnp.exp(-((safe_x / theta) ** omega))
    else:  # pragma: no cover - the entry validates before dispatch
        raise ValueError(f"curve must be {LOGLOGISTIC} (loglogistic) or {WEIBULL} (weibull)")
    return jnp.where(positive, g, 0.0)


def clark_model(data: dict[str, Any]) -> None:
    """NumPyro model consuming the Stan data block the entry assembles.

    Mirrors ``model.stan``: same priors, same Cape Cod mean, same plug-in
    dispersion.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import gammaln

    w0 = np.asarray(data["w"], dtype=int) - 1  # 0-based origin index
    age_lo = jnp.asarray(np.asarray(data["age_lo"], dtype=float))
    age_hi = jnp.asarray(np.asarray(data["age_hi"], dtype=float))
    inc_loss = jnp.asarray(np.asarray(data["inc_loss"], dtype=float))
    logprem_w = jnp.asarray(np.asarray(data["logprem_w"], dtype=float))
    phi = float(data["phi"])  # plug-in Pearson dispersion from the MLE twin: DATA
    curve = int(data["curve"])
    theta_prior_median = float(data["theta_prior_median"])

    # Priors - byte-for-byte the Stan `model` block. logelr carries the
    # family's variance-10 convention; omega/theta are the curve's shape and
    # scale, both positive, with the scale's prior median tracking the dev
    # grain so it means the same thing on annual and quarterly triangles.
    logelr = numpyro.sample("logelr", dist.Normal(-0.4, np.sqrt(10.0)))
    omega = numpyro.sample("omega", dist.LogNormal(np.log(1.5), 0.5))
    theta = numpyro.sample("theta", dist.LogNormal(np.log(theta_prior_median), 1.0))

    # Cape Cod mean: an origin's ultimate is elr * premium, and the cell claims
    # the slice of the growth curve its age interval spans.
    emerged = growth_curve(age_hi, omega, theta, curve) - growth_curve(
        age_lo, omega, theta, curve
    )
    mu = numpyro.deterministic("mu", jnp.exp(logprem_w[w0] + logelr) * emerged)

    # Stan's odp_lpdf, term for term (see the module docstring on duplication).
    # Attached with numpyro.factor rather than a Distribution subclass: this
    # entry's Stan reference exposes log_lik in generated quantities, and the
    # port's log_likelihood group is rebuilt by arviz from observed sites -
    # a factor IS an observed site, so the group exists either way. Kept as a
    # factor here because `mu` can legitimately reach 0 when a cell's age
    # interval spans no emergence, where a Distribution's support check would
    # reject rather than return -inf.
    scaled = inc_loss / phi
    numpyro.factor(
        "obs",
        scaled * jnp.log(mu / phi) - mu / phi - gammaln(scaled + 1.0),
    )


def sample(
    data: dict[str, Any],
    *,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2500,
    seed: int | None = None,
    target_accept: float = 0.9,
    chain_method: str = "sequential",
    progress_bar: bool = False,
):
    """Sample the Clark posterior with NUTS; return an ``arviz.InferenceData``.

    ``target_accept`` defaults to 0.9, matching the Stan entry: a growth curve's
    (omega, theta) are strongly correlated - the shape and scale trade off
    against each other along a ridge - which is harder geometry than the ODP
    GLM despite having only three parameters.
    """
    import arviz as az
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(chains)
    seed = 0 if seed is None else int(seed)

    kernel = NUTS(clark_model, target_accept_prob=target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=iter_warmup,
        num_samples=iter_sampling,
        num_chains=chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), data)
    runtime_s = time.perf_counter() - t0

    idata = az.from_numpyro(mcmc)
    idata.attrs["runtime_s"] = runtime_s
    idata.attrs["backend"] = "numpyro"
    return idata
