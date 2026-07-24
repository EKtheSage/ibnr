"""Meyers CSR - PyMC port of the Stan reference (``model.stan``).

Same model, same **centered** parameterization, held constant for parity. The
Stan ``data`` block dict from ``kernels.contract.stan_data`` is consumed
verbatim; ``a_ig`` is an unbounded ``InverseGamma(1, 1)`` (see the note in
``model_numpyro.py`` - the (0, 1e5) Stan bound is numerically irrelevant).

``build_model()`` returns the ``pm.Model`` graph (readable, ejectable source);
``sample()`` runs NUTS and returns an ``arviz.InferenceData`` carrying the same
deterministic quantities Stan exposes in ``transformed parameters`` so the
backend dispatch in ``model.py`` reads predictions off it identically.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def build_model(data: dict[str, Any]):
    """Construct the CSR ``pm.Model`` from the standardized Stan data dict."""
    import pymc as pm
    import pytensor.tensor as pt

    n_w = int(data["n_w"])
    n_d = int(data["n_d"])
    w0 = np.asarray(data["w"], dtype=int) - 1
    d0 = np.asarray(data["d"], dtype=int) - 1
    logprem = np.asarray(data["logprem"], dtype=float)
    logloss = np.asarray(data["logloss"], dtype=float)
    root10 = np.sqrt(10.0)  # prior SD shared by the three diffuse normals

    model = pm.Model()
    with model:
        # Priors - identical to model_numpyro.py and to Stan's `model` block,
        # held constant for parity. gamma carries the monograph's tight
        # normal(0, 0.05) prior; everything else is diffuse.
        logelr = pm.Normal("logelr", -0.4, root10)  # log expected loss ratio
        r_alpha = pm.Normal("r_alpha", 0.0, root10, shape=n_w - 1)
        r_beta = pm.Normal("r_beta", 0.0, root10, shape=n_d - 1)
        a_ig = pm.InverseGamma("a_ig", alpha=1.0, beta=1.0, shape=n_d)
        gamma = pm.Normal("gamma", 0.0, 0.05)

        # Identifiability pinning, as in Stan's transformed parameters:
        # alpha[1] = 0, beta[n_d] = 0.
        alpha = pm.Deterministic("alpha", pt.concatenate([pt.zeros(1), r_alpha]))
        beta = pm.Deterministic("beta", pt.concatenate([r_beta, pt.zeros(1)]))

        # speedup[w] = (1 - gamma)^(w-1) as a cumulative PRODUCT, mirroring
        # Stan's recurrence: gamma is unconstrained, so a warmup excursion past
        # 1 would make the base negative and a float power NaN. cumprod is exact
        # for either sign.
        speedup = pm.Deterministic(
            "speedup",
            pt.cumprod(pt.concatenate([pt.ones(1), pt.full((n_w - 1,), 1.0 - gamma)])),
        )

        # a_i = gamma_cdf(1/a_ig | 1, 1) = 1 - exp(-1/a_ig); reverse cumsum ->
        # sig2 decreasing in dev lag (same construction as model.stan).
        a = 1.0 - pt.exp(-1.0 / a_ig)
        sig2 = pm.Deterministic("sig2", pt.cumsum(a[::-1])[::-1])
        sig = pm.Deterministic("sig", pt.sqrt(sig2))

        # mu[w,d] = logprem + logelr + alpha[w] + beta[d] * speedup[w]: the
        # settlement-rate trend scales the whole development profile.
        mu = pm.Deterministic("mu", logprem + logelr + alpha[w0] + beta[d0] * speedup[w0])

        # Likelihood: log(C[w,d]) ~ Normal(mu, sig[d]) - Stan's model block.
        pm.Normal("obs", mu, sig[d0], observed=logloss)
    return model


def sample(
    data: dict[str, Any],
    *,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2500,
    seed: int | None = None,
    target_accept: float = 0.9,
    cores: int = 1,
    nuts_sampler: str = "pymc",
    progressbar: bool = False,
):
    """Sample the CSR posterior with PyMC's NUTS; return ``arviz.InferenceData``.

    ``cores=1`` by default - PyMC's multiprocessing chain execution is fragile
    on Windows. ``target_accept`` defaults to 0.9 to match the Stan entry.

    ``nuts_sampler`` selects the NUTS implementation over the *same* PyMC model
    graph (``"pymc"`` native is the parity reference; ``"nutpie"`` / ``"numpyro"``
    / ``"blackjax"`` are faster alternatives where installed). The graph is
    identical either way, so only runtime changes.
    """
    import pymc as pm

    from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx

    ensure_pytensor_cxx()
    model = build_model(data)
    t0 = time.perf_counter()
    with model:
        # `init` left at pm.sample's default (jitter+adapt_diag): no custom init,
        # so the card's convergence comparison measures the centered
        # parameterization. R-hat/ESS are computed once via arviz in
        # model.py::convergence(), so every backend reports identical diagnostics.
        idata = pm.sample(
            draws=iter_sampling,
            tune=iter_warmup,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            nuts_sampler=nuts_sampler,
            progressbar=progressbar,
            idata_kwargs={"log_likelihood": True},
            compute_convergence_checks=False,
        )
    runtime_s = time.perf_counter() - t0
    idata.attrs["runtime_s"] = runtime_s
    idata.attrs["backend"] = f"pymc:{nuts_sampler}" if nuts_sampler != "pymc" else "pymc"
    return idata
