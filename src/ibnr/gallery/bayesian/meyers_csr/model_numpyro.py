"""Meyers CSR - NumPyro port of the Stan reference (``model.stan``).

Same model, same **centered** parameterization, held constant for parity
(design decision 7): the Stan ``data`` block dict from
``kernels.contract.stan_data`` is consumed verbatim. The one documented
deviation, shared with the CCL ports, is that ``a_ig`` is an *unbounded*
``InverseGamma(1, 1)`` here - Stan bounds it to (0, 1e5) only to keep
``a = gamma_cdf(1/a_ig | 1, 1)`` off a hard 0/1 boundary, and the mass above
1e5 is ~1e-5, far below MCMC noise.

CSR is structurally easier to port than CCL: ``mu`` has no across-origin
recurrence (no ``rho``, so ``prev_idx`` goes unused), and every quantity is a
plain vectorized expression. The one construction that needs care is the
settlement-rate ``speedup`` - see ``csr_model``.

``sample()`` returns an ``arviz.InferenceData`` whose ``posterior`` group
carries the same deterministic quantities Stan puts in ``transformed
parameters`` (``alpha``, ``beta``, ``speedup``, ``sig``, ``mu``), so the
backend dispatch in ``model.py`` reads predictions off it identically to
cmdstanpy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def csr_model(data: dict[str, Any]) -> None:
    """NumPyro model consuming the standardized Stan data dict.

    Mirrors ``model.stan``: identical priors, the reverse-cumsum construction
    of the decreasing ``sig2``, and the settlement-rate trend on the
    development profile.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n_w = int(data["n_w"])
    n_d = int(data["n_d"])
    # 0-based static index arrays for the vectorized mu construction
    w0 = np.asarray(data["w"], dtype=int) - 1
    d0 = np.asarray(data["d"], dtype=int) - 1
    logprem = jnp.asarray(np.asarray(data["logprem"], dtype=float))
    logloss = jnp.asarray(np.asarray(data["logloss"], dtype=float))
    root10 = np.sqrt(10.0)  # prior SD shared by the three diffuse normals

    # Priors - the monograph values, byte-for-byte the Stan `model` block,
    # held constant for parity. r_alpha/r_beta are the *free* accident-year /
    # dev-lag effects (the remaining element of each is pinned to 0 below);
    # a_ig is the per-dev variance seed; gamma is CSR's settlement-rate trend,
    # given the monograph's tight normal(0, 0.05) prior.
    logelr = numpyro.sample("logelr", dist.Normal(-0.4, root10))  # log expected loss ratio
    r_alpha = numpyro.sample("r_alpha", dist.Normal(0.0, root10).expand([n_w - 1]))
    r_beta = numpyro.sample("r_beta", dist.Normal(0.0, root10).expand([n_d - 1]))
    a_ig = numpyro.sample("a_ig", dist.InverseGamma(1.0, 1.0).expand([n_d]))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.05))

    # Identifiability pinning, exactly as Stan's transformed-parameters block:
    # alpha[1] = 0 anchors the accident-year level, beta[n_d] = 0 anchors the
    # development profile at the last lag (which is why gamma drops out of the
    # ultimate - see the entry's predict()).
    alpha = numpyro.deterministic("alpha", jnp.concatenate([jnp.zeros(1), r_alpha]))
    beta = numpyro.deterministic("beta", jnp.concatenate([r_beta, jnp.zeros(1)]))

    # speedup[w] = (1 - gamma)^(w-1), Stan's forward recurrence. Written as a
    # cumulative PRODUCT rather than a power: gamma is unconstrained, so an
    # init or warmup excursion past 1 makes the base negative, and a float
    # exponent on a negative base is NaN. cumprod is exact for either sign and
    # reproduces the recurrence term for term.
    speedup = numpyro.deterministic(
        "speedup",
        jnp.cumprod(jnp.concatenate([jnp.ones(1), jnp.full(n_w - 1, 1.0 - gamma)])),
    )

    # a_i = gamma_cdf(1/a_ig | 1, 1) = 1 - exp(-1/a_ig) ~ uniform(0, 1);
    # sig2[d] = sum_{i>=d} a_i forces sig2 (and sig) decreasing in dev lag:
    # more settled claims carry less process variance.
    a = 1.0 - jnp.exp(-1.0 / a_ig)
    sig2 = numpyro.deterministic("sig2", jnp.cumsum(a[::-1])[::-1])
    sig = numpyro.deterministic("sig", jnp.sqrt(sig2))

    # mu[w,d] = logprem + logelr + alpha[w] + beta[d] * speedup[w]. The
    # settlement-rate trend multiplies the whole development profile, shrinking
    # it toward zero for later origins when gamma > 0 (a speedup).
    mu = numpyro.deterministic("mu", logprem + logelr + alpha[w0] + beta[d0] * speedup[w0])

    # Likelihood: log(C[w,d]) ~ Normal(mu, sig[d]) - Stan's model block.
    numpyro.sample("obs", dist.Normal(mu, sig[d0]), obs=logloss)


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
    """Sample the CSR posterior with NUTS and return an ``arviz.InferenceData``.

    ``target_accept`` defaults to 0.9, matching the Stan entry: the gamma-times-beta
    interaction makes CSR's centered geometry harder than CCL's.
    ``chain_method="sequential"`` is the Windows-robust default, as in the CCL port.
    """
    import arviz as az
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(chains)
    seed = 0 if seed is None else int(seed)

    # Init strategy left at NUTS's default (init_to_uniform): no custom init, so
    # the convergence comparison in card.md reflects the centered
    # parameterization itself rather than an init trick - identical across backends.
    kernel = NUTS(csr_model, target_accept_prob=target_accept)
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
