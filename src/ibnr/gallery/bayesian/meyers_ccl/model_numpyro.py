"""Meyers CCL - NumPyro port of the Stan reference (``model.stan``).

Same model, same **centered** parameterization, held constant for parity: the
Stan ``data`` block dict from ``kernels.contract.stan_data`` is consumed
verbatim, **including Stan's ``a_ig`` upper bound**.

This port previously left ``a_ig`` an *unbounded* ``InverseGamma(1, 1)``,
documented as a harmless deviation on the grounds that Stan's (0, 1e5) bound
truncates only ~1e-5 of the PRIOR. That reasoning was wrong, and it was caught
on 2026-07-24 while porting CSR: the POSTERIOR concentrates in exactly that
corner at deep development lags, where the data barely constrain sigma. On a
real WC triangle, up to 10.6% of an unbounded port's ``a_ig`` draws sat above
1e5 - the ``a -> 0`` region Stan forbids - pulling ``sig`` 5-12% below the Stan
reference. The bound is now reproduced the way Stan implements it: an
interval-constrained parameter carrying the ``InverseGamma(1, 1)`` density as a
factor (same transform, same Jacobian; the truncation constant is a true
constant and is dropped by both).

``sample()`` returns an ``arviz.InferenceData`` whose ``posterior`` group
carries the same deterministic quantities Stan puts in ``transformed
parameters`` (``alpha``, ``beta``, ``rho``, ``sig``, ``mu``), so the backend
dispatch in ``model.py`` reads predictions off it identically to cmdstanpy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

#: Stan's ``upper=`` bound on a_ig (model.stan). Load-bearing at deep dev lags -
#: see the module docstring.
A_IG_MAX = 1e5

#: Where a_ig chains start. The default unconstrained init lands against
#: A_IG_MAX (see ``sample``); this is inside the posterior bulk instead.
A_IG_INIT = 1e3


def ccl_model(data: dict[str, Any]) -> None:
    """NumPyro model consuming the standardized Stan data dict.

    Mirrors ``model.stan``: identical priors, the reverse-cumsum construction of
    the decreasing ``sig2``, and the rho residual term - the latter via the
    vectorized closed form of Stan's ``prev_idx`` recurrence (``ccl_mu_index``).
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from ibnr.kernels.contract import ccl_mu_index

    n_w = int(data["n_w"])
    n_d = int(data["n_d"])
    # 0-based static index arrays for the (vectorized) mu construction
    w0 = np.asarray(data["w"], dtype=int) - 1
    d0 = np.asarray(data["d"], dtype=int) - 1
    logprem = jnp.asarray(np.asarray(data["logprem"], dtype=float))
    logloss = jnp.asarray(np.asarray(data["logloss"], dtype=float))
    idx = ccl_mu_index(data)
    expo = jnp.asarray(idx["expo"])  # (N, N) int exponents
    colmask = jnp.asarray(idx["colmask"])  # (N, N) contribution mask
    logloss_prev = jnp.asarray(idx["logloss_prev"])  # (N,)
    max_expo = int(idx["expo"].max())
    root10 = np.sqrt(10.0)  # prior SD shared by all three normals: Stan sqrt(10.0)

    # Priors - the monograph values, byte-for-byte the Stan `model` block
    # (model.stan lines 64-68), held constant for parity. r_alpha/r_beta are the
    # *free* accident-year / dev-lag effects (n-1 of each; the last is pinned to 0
    # below). a_ig is the per-dev variance seed (see reparam note in the module
    # docstring); r_rho is the raw correlation on (0, 1).
    logelr = numpyro.sample("logelr", dist.Normal(-0.4, root10))  # log expected loss ratio
    r_alpha = numpyro.sample("r_alpha", dist.Normal(0.0, root10).expand([n_w - 1]))
    r_beta = numpyro.sample("r_beta", dist.Normal(0.0, root10).expand([n_d - 1]))
    # a_ig exactly as Stan declares it: interval-constrained to (0, A_IG_MAX)
    # with the InverseGamma(1, 1) density supplied as a factor - the direct
    # analogue of `a_ig ~ inv_gamma(1, 1)` under `<lower=0, upper=1e5>`.
    a_ig = numpyro.sample(
        "a_ig", dist.ImproperUniform(dist.constraints.interval(0.0, A_IG_MAX), (), (n_d,))
    )
    numpyro.factor("a_ig_prior", dist.InverseGamma(1.0, 1.0).log_prob(a_ig).sum())
    r_rho = numpyro.sample("r_rho", dist.Beta(2.0, 2.0))

    # Identifiability pinning, exactly as Stan's transformed-parameters block:
    # alpha[1] = 0 (model.stan 42-43) and beta[n_d] = 0 (44-45) anchor the AY/dev
    # effects. rho = 2*r_rho - 1 maps Beta(2,2) on (0,1) onto (-1, 1) (model.stan
    # 46): rho is the CCL correlation between successive accident years' log-losses.
    alpha = numpyro.deterministic("alpha", jnp.concatenate([jnp.zeros(1), r_alpha]))
    beta = numpyro.deterministic("beta", jnp.concatenate([r_beta, jnp.zeros(1)]))
    rho = numpyro.deterministic("rho", 2.0 * r_rho - 1.0)

    # a_i = gamma_cdf(1/a_ig | 1, 1) = 1 - exp(-1/a_ig) ~ uniform(0, 1);
    # sig2[d] = sum_{i>=d} a_i forces sig2 (and sig) decreasing in dev lag.
    a = 1.0 - jnp.exp(-1.0 / a_ig)
    sig2 = numpyro.deterministic("sig2", jnp.cumsum(a[::-1])[::-1])
    sig = numpyro.deterministic("sig", jnp.sqrt(sig2))

    # mu = P(rho) @ B, the closed form of the Stan prev_idx recurrence (see
    # kernels.contract.ccl_mu_index). Powers via a table indexed by the static
    # exponents so JAX uses integer_pow (jnp.power of a negative base with a
    # float exponent is NaN); table depth is max_expo <= n_w - 1.
    base = logprem + logelr + alpha[w0] + beta[d0]
    big_b = base + rho * logloss_prev
    pow_table = jnp.stack([(-rho) ** k for k in range(max_expo + 1)])
    p_mat = colmask * pow_table[expo]
    mu = numpyro.deterministic("mu", p_mat @ big_b)

    # Likelihood: log(C[w,d]) ~ Normal(mu, sig[d]) - Stan model block line 69,
    # `logloss ~ normal(mu, sig[d])`. sig[d0] broadcasts the per-dev SD to cells.
    numpyro.sample("obs", dist.Normal(mu, sig[d0]), obs=logloss)


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
    """Sample the CCL posterior with NUTS and return an ``arviz.InferenceData``.

    ``chain_method="sequential"`` by default - robust on Windows, where JAX's
    parallel/multiprocessing chain execution is fragile. Returns the same
    diagnostics arviz derives for any backend (``sample_stats.diverging`` etc.)
    plus a ``log_likelihood`` group for the observed cells (ELPD-ready).
    """
    import arviz as az
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS, init_to_value

    numpyro.set_host_device_count(chains)
    seed = 0 if seed is None else int(seed)

    # Every site keeps NUTS's default init (init_to_uniform) EXCEPT a_ig, which
    # is seeded inside its posterior bulk. Documented deviation, and necessary:
    # the (0, 1e5) interval transform maps uniform(-2, 2) on the unconstrained
    # scale onto a_ig in (1.2e4, 8.8e4), hard against the upper bound. Stan
    # survives starting there; NumPyro does not. Only this site's warmup path
    # changes - the stationary posterior is unaffected.
    init = init_to_value(values={"a_ig": np.full(int(data["n_d"]), A_IG_INIT)})
    kernel = NUTS(ccl_model, target_accept_prob=target_accept, init_strategy=init)
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

    # Stamp runtime + backend onto the idata attrs; model.py::convergence() and
    # the card's cross-backend table read wall-clock and backend label from here.
    idata = az.from_numpyro(mcmc)
    idata.attrs["runtime_s"] = runtime_s
    idata.attrs["backend"] = "numpyro"
    return idata
