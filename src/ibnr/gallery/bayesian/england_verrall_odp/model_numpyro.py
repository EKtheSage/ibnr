"""England & Verrall ODP - NumPyro port of the Stan reference (``model.stan``).

Same model, same parameterization, held constant for parity (design decision
7): the ``kernels.contract.odp_stan_data`` dict is consumed verbatim, including
the plug-in dispersion ``phi``, which is DATA here exactly as it is in Stan -
never a sampled parameter.

The only thing worth care in this port is the likelihood. Stan declares a
user-defined density

    odp_lpdf(x | mu, phi) = (x/phi) log(mu/phi) - mu/phi - lgamma(x/phi + 1)

which has no built-in counterpart in either PPL. It is, however, exactly

    Poisson(mu/phi).log_prob(x/phi)

- the od-Poisson quasi-likelihood is a Poisson mass evaluated at the scaled
data ``x/phi`` with the scaled mean ``mu/phi``. That identity is why the
posterior is proper despite ``x/phi`` not being an integer, and it is the
cheapest way to convince yourself the port is faithful. It is nonetheless
written out explicitly below rather than delegated to ``dist.Poisson``: the
gallery ships literal, ejectable source (decision 6), a reader should see the
density the card documents, and NumPyro's ``Poisson`` is a *discrete*
distribution whose support constraint would reject the continuous ``x/phi``
under ``validate_args``.

Implemented as a real ``dist.Distribution`` subclass rather than a
``numpyro.factor``. Note the reason is NOT that a factor loses the
``log_likelihood`` group - it does not: ``numpyro.factor`` is itself an
observed sample site carrying a ``Unit`` distribution, and
``numpyro.infer.log_likelihood`` collects every observed site, so a batched
factor does yield a per-observation array. The reasons are that a factor's
``observed_data`` group comes back degenerate (the ``Unit`` site stores an
empty value, not the losses) and ``Predictive`` cannot draw from it, which
would break the ``PredictiveDistribution`` contract this gallery is built on.
A genuine distribution keeps all three - likelihood, observed data, and
forward simulation - honest.

``sample()`` returns an ``arviz.InferenceData`` whose ``posterior`` group
carries the same deterministic quantities Stan puts in ``transformed
parameters`` (``alpha``, ``beta``, ``log_mu``), so the backend dispatch in
``model.py`` reads predictions off it identically to cmdstanpy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _odp_distribution():
    """Build the ODP quasi-likelihood distribution class.

    Deferred into a function because ``numpyro`` is an optional extra: the
    gallery must import (and this entry must register) without it installed.
    """
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import random
    from jax.scipy.special import gammaln

    class ODP(dist.Distribution):
        """Over-dispersed Poisson quasi-likelihood with a PLUG-IN dispersion.

        ``log_prob`` is Stan's ``odp_lpdf`` term for term. ``phi`` is data, so
        the ``-lgamma(x/phi + 1)`` and ``(x/phi) log(1/phi)`` pieces are
        constants in the parameters and cannot move the posterior; they are
        kept anyway so the ``log_likelihood`` group matches Stan's
        ``generated quantities`` block value for value.
        """

        # x is a non-negative incremental loss; the contract already rejects
        # negative increments upstream (odp_stan_data), so this is a guard, not
        # a filter.
        support = dist.constraints.nonnegative
        arg_constraints = {
            "mu": dist.constraints.positive,
            "phi": dist.constraints.positive,
        }

        def __init__(self, mu, phi, *, validate_args=None):
            self.mu, self.phi = mu, phi
            batch_shape = jnp.broadcast_shapes(jnp.shape(mu), jnp.shape(phi))
            super().__init__(batch_shape=batch_shape, validate_args=validate_args)

        def log_prob(self, value):
            scaled = value / self.phi  # x/phi: the Poisson-equivalent "count"
            return scaled * jnp.log(self.mu / self.phi) - self.mu / self.phi - gammaln(scaled + 1.0)

        def sample(self, key, sample_shape=()):
            """Scaled-Poisson forward draw: ``phi * Poisson(mu / phi)``.

            Matches the density's own moments (mean ``mu``, variance
            ``phi * mu``) and is E&V's own od-Poisson simulation. Not needed to
            *fit* - this site is always observed - but implementing it keeps
            ``Predictive`` working and stops arviz's converter from tripping if
            it ever traces the model under an init-by-sampling strategy.
            """
            shape = sample_shape + self.batch_shape
            return self.phi * random.poisson(key, self.mu / self.phi, shape=shape)

    return ODP


def odp_model(data: dict[str, Any]) -> None:
    """NumPyro model consuming the standardized ODP contract dict.

    Mirrors ``model.stan``: identical vague priors, the same two identifiability
    pins, the same log-link linear predictor with the premium offset.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n_w = int(data["n_w"])
    n_d = int(data["n_d"])
    # 0-based static index arrays for the vectorized linear predictor
    w0 = np.asarray(data["w"], dtype=int) - 1
    d0 = np.asarray(data["d"], dtype=int) - 1
    logprem = jnp.asarray(np.asarray(data["logprem"], dtype=float))
    inc_loss = jnp.asarray(np.asarray(data["inc_loss"], dtype=float))
    phi = float(data["phi"])  # plug-in Pearson dispersion: DATA, never sampled
    root10 = np.sqrt(10.0)  # prior SD shared by all three normals: Stan sqrt(10.0)

    # Priors - byte-for-byte the Stan `model` block, held constant for parity.
    # r_alpha / r_beta are the FREE accident-year / dev-lag effects; the first
    # element of each is pinned to 0 below.
    c = numpyro.sample("c", dist.Normal(0.0, root10))  # log scale / intercept
    r_alpha = numpyro.sample("r_alpha", dist.Normal(0.0, root10).expand([n_w - 1]))
    r_beta = numpyro.sample("r_beta", dist.Normal(0.0, root10).expand([n_d - 1]))

    # Identifiability pinning, exactly as Stan's transformed-parameters block:
    # alpha[1] = 0 AND beta[1] = 0 (note ODP pins the FIRST beta, where the
    # Meyers family pins the LAST - the two conventions are not interchangeable).
    alpha = numpyro.deterministic("alpha", jnp.concatenate([jnp.zeros(1), r_alpha]))
    beta = numpyro.deterministic("beta", jnp.concatenate([jnp.zeros(1), r_beta]))

    # log m[w,d] = logprem[w] + c + alpha[w] + beta[d] (E&V 7.11.8 log link).
    # The premium offset only re-centers alpha - each origin keeps a free level -
    # so the family, and its chain-ladder-reproducing MLE, is unchanged.
    log_mu = numpyro.deterministic("log_mu", logprem + c + alpha[w0] + beta[d0])

    odp = _odp_distribution()
    numpyro.sample("obs", odp(jnp.exp(log_mu), phi), obs=inc_loss)


def sample(
    data: dict[str, Any],
    *,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2500,
    seed: int | None = None,
    target_accept: float = 0.8,
    chain_method: str = "sequential",
    progress_bar: bool = False,
):
    """Sample the ODP posterior with NUTS and return an ``arviz.InferenceData``.

    ``target_accept`` defaults to 0.8, matching the Stan entry: the ODP
    posterior is a well-conditioned log-linear GLM and needs no extra
    adaptation. ``chain_method="sequential"`` is the Windows-robust default, as
    in the other ports.
    """
    import arviz as az
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(chains)
    seed = 0 if seed is None else int(seed)

    # Init left at NUTS's default (init_to_uniform): no custom init, so the
    # card's convergence comparison reflects the model rather than an init
    # trick. Unlike the Meyers family there is no bounded parameter here, so
    # the default init has no boundary to land against.
    kernel = NUTS(odp_model, target_accept_prob=target_accept)
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
