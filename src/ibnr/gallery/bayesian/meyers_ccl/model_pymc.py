"""Meyers CCL — PyMC port of the Stan reference (``model.stan``).

Same model, same **centered** parameterization, held constant for parity. The
Stan ``data`` block dict from ``kernels.contract.stan_data`` is consumed
verbatim; ``a_ig`` is an unbounded ``InverseGamma(1, 1)`` (see the note in
``model_numpyro.py`` — the (0, 1e5) Stan bound is numerically irrelevant).

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
    """Construct the CCL ``pm.Model`` from the standardized Stan data dict."""
    import pymc as pm
    import pytensor.tensor as pt

    from ibnr.kernels.contract import ccl_mu_index

    n_w = int(data["n_w"])
    n_d = int(data["n_d"])
    w0 = np.asarray(data["w"], dtype=int) - 1
    d0 = np.asarray(data["d"], dtype=int) - 1
    logprem = np.asarray(data["logprem"], dtype=float)
    logloss = np.asarray(data["logloss"], dtype=float)
    idx = ccl_mu_index(data)
    expo = idx["expo"]  # (N, N) static int exponents
    colmask = idx["colmask"]  # (N, N) contribution mask
    logloss_prev = idx["logloss_prev"]  # (N,)
    max_expo = int(expo.max())
    root10 = np.sqrt(10.0)

    model = pm.Model()
    with model:
        logelr = pm.Normal("logelr", -0.4, root10)
        r_alpha = pm.Normal("r_alpha", 0.0, root10, shape=n_w - 1)
        r_beta = pm.Normal("r_beta", 0.0, root10, shape=n_d - 1)
        a_ig = pm.InverseGamma("a_ig", alpha=1.0, beta=1.0, shape=n_d)
        r_rho = pm.Beta("r_rho", 2.0, 2.0)

        alpha = pm.Deterministic("alpha", pt.concatenate([pt.zeros(1), r_alpha]))
        beta = pm.Deterministic("beta", pt.concatenate([r_beta, pt.zeros(1)]))
        rho = pm.Deterministic("rho", 2.0 * r_rho - 1.0)

        # a_i = gamma_cdf(1/a_ig | 1, 1) = 1 - exp(-1/a_ig); reverse cumsum ->
        # decreasing sig2 in dev lag (same construction as model.stan).
        a = 1.0 - pt.exp(-1.0 / a_ig)
        sig2 = pm.Deterministic("sig2", pt.cumsum(a[::-1])[::-1])
        sig = pm.Deterministic("sig", pt.sqrt(sig2))

        # mu = P(rho) @ B, the closed form of the Stan prev_idx recurrence (see
        # kernels.contract.ccl_mu_index) — an N x N matmul keeps the PyTensor
        # C-graph tiny vs. the N-deep unrolled scalar chain (which dominates
        # compile time). Powers via a table indexed by the static exponents.
        base = logprem + logelr + alpha[w0] + beta[d0]
        big_b = base + rho * logloss_prev
        pow_table = pt.stack([(-rho) ** k for k in range(max_expo + 1)])
        p_mat = colmask * pow_table[expo]
        mu = pm.Deterministic("mu", p_mat @ big_b)

        pm.Normal("obs", mu, sig[d0], observed=logloss)
    return model


def sample(
    data: dict[str, Any],
    *,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2500,
    seed: int | None = None,
    target_accept: float = 0.8,
    cores: int = 1,
    nuts_sampler: str = "pymc",
    progressbar: bool = False,
):
    """Sample the CCL posterior with PyMC's NUTS; return ``arviz.InferenceData``.

    ``cores=1`` by default — PyMC's multiprocessing chain execution is fragile
    on Windows. The returned idata carries ``sample_stats`` (divergences etc.)
    and a ``log_likelihood`` group for the observed cells (ELPD-ready).

    ``nuts_sampler`` selects the NUTS implementation over the *same* PyMC model
    graph: ``"pymc"`` (native, PyTensor C backend — the default and the parity
    reference), or a faster alternative such as ``"nutpie"`` (Rust, compiles the
    logp via numba) / ``"numpyro"`` / ``"blackjax"`` when that package is
    installed. Because the graph is identical, only runtime changes — the
    posterior (and parity) are unaffected. See the model card's convergence note
    on why native PyMC is slow on a BLAS-less pip PyTensor.
    """
    import pymc as pm

    from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx

    ensure_pytensor_cxx()
    model = build_model(data)
    t0 = time.perf_counter()
    with model:
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
