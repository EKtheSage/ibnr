"""Meyers CSR - NumPyro port of the Stan reference (``model.stan``).

Same model, same **centered** parameterization, held constant for parity
(design decision 7): the Stan ``data`` block dict from
``kernels.contract.stan_data`` is consumed verbatim, **including Stan's
``a_ig`` upper bound**.

That bound is not cosmetic, contrary to what these ports (and CCL's) assumed
until 2026-07-24. Stan declares ``vector<lower=0, upper=1e5>[n_d] a_ig``; the
truncated *prior* mass is ~1e-5, which is where the "harmless" reading came
from - but the *posterior* piles into exactly that corner at deep development
lags, where the data barely constrain sigma. Measured on a real WC triangle,
up to **10.6%** of an unbounded port's posterior draws for ``a_ig`` sat above
1e5, i.e. in the ``a -> 0`` region Stan forbids, dragging ``sig`` 5-12% below
the Stan reference and failing parity. So the bound is reproduced here the way
Stan itself implements it: an interval-constrained parameter carrying the
``InverseGamma(1, 1)`` density as a factor (NumPyro applies the same
logit/sigmoid transform and Jacobian for an ``interval`` constraint, and the
truncation's normalizing constant is a true constant, dropped by both).

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

#: Stan's ``upper=`` bound on a_ig (model.stan). Load-bearing, not cosmetic -
#: see the module docstring; an unbounded port biases sig low at deep dev lags.
A_IG_MAX = 1e5

#: Where a_ig chains start. The default unconstrained init lands against
#: A_IG_MAX (see ``sample``); this is inside the posterior bulk instead.
A_IG_INIT = 1e3


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
    # a_ig exactly as Stan declares it: interval-constrained to (0, A_IG_MAX)
    # with the InverseGamma(1, 1) density supplied as a factor. numpyro.factor
    # is the direct analogue of Stan's `a_ig ~ inv_gamma(1, 1)` under a
    # `<lower=0, upper=1e5>` declaration - the constraint fixes the transform
    # and Jacobian, the factor supplies the (unnormalized) density.
    a_ig = numpyro.sample(
        "a_ig", dist.ImproperUniform(dist.constraints.interval(0.0, A_IG_MAX), (), (n_d,))
    )
    numpyro.factor("a_ig_prior", dist.InverseGamma(1.0, 1.0).log_prob(a_ig).sum())
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
    from numpyro.infer import MCMC, NUTS, init_to_value

    numpyro.set_host_device_count(chains)
    seed = 0 if seed is None else int(seed)

    # Every site keeps NUTS's default init (init_to_uniform) EXCEPT a_ig, which
    # is seeded in the bulk of its posterior. This is a documented deviation
    # from "native default init everywhere" and it is necessary, not cosmetic:
    # the (0, 1e5) interval transform maps uniform(-2, 2) on the unconstrained
    # scale onto a_ig in (1.2e4, 8.8e4), i.e. hard against the upper bound.
    # Stan survives starting there; NumPyro does not (max R-hat 1.59, ESS 7 -
    # measured). init_to_value falls back to init_to_uniform for every other
    # site, so only this one parameter's warmup path changes, and the
    # stationary posterior is unaffected.
    init = init_to_value(values={"a_ig": np.full(int(data["n_d"]), A_IG_INIT)})
    kernel = NUTS(csr_model, target_accept_prob=target_accept, init_strategy=init)
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
