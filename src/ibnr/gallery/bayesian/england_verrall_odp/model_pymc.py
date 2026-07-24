"""England & Verrall ODP - PyMC port of the Stan reference (``model.stan``).

Same model, same parameterization, held constant for parity. The
``kernels.contract.odp_stan_data`` dict is consumed verbatim, including the
plug-in dispersion ``phi``, which is DATA here exactly as in Stan.

The likelihood is Stan's user-defined

    odp_lpdf(x | mu, phi) = (x/phi) log(mu/phi) - mu/phi - lgamma(x/phi + 1)

which is exactly ``Poisson(mu/phi).log_prob(x/phi)`` - see ``model_numpyro``
for why that identity matters and why the density is nonetheless written out
literally instead of delegated to ``pm.Poisson`` (whose discrete support would
reject the continuous ``x/phi``).

Supplied through ``pm.CustomDist(..., observed=...)`` rather than
``pm.Potential``: only a genuine observed variable makes PyMC emit a
per-observation ``log_likelihood`` group under
``idata_kwargs={"log_likelihood": True}``, which the ELPD/LOO harness needs. A
``Potential`` contributes to the posterior identically and silently yields no
``log_likelihood`` at all.

``build_model()`` returns the ``pm.Model`` graph (readable, ejectable source);
``sample()`` runs NUTS and returns an ``arviz.InferenceData`` carrying the same
deterministic quantities Stan exposes in ``transformed parameters``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _odp_logp(value, mu, phi):
    """Stan's ``odp_lpdf``, term for term, on PyTensor tensors.

    ``phi`` is data, so ``-gammaln(x/phi + 1)`` and ``(x/phi) log(1/phi)`` are
    constants in the parameters and cannot move the posterior; they are kept so
    the ``log_likelihood`` group matches Stan's ``generated quantities`` value
    for value.
    """
    import pytensor.tensor as pt

    scaled = value / phi  # x/phi: the Poisson-equivalent "count"
    return scaled * pt.log(mu / phi) - mu / phi - pt.gammaln(scaled + 1.0)


def build_model(data: dict[str, Any]):
    """Construct the ODP ``pm.Model`` from the standardized contract dict."""
    import pymc as pm
    import pytensor.tensor as pt

    n_w = int(data["n_w"])
    n_d = int(data["n_d"])
    w0 = np.asarray(data["w"], dtype=int) - 1
    d0 = np.asarray(data["d"], dtype=int) - 1
    logprem = np.asarray(data["logprem"], dtype=float)
    inc_loss = np.asarray(data["inc_loss"], dtype=float)
    phi = float(data["phi"])  # plug-in Pearson dispersion: DATA, never sampled
    root10 = np.sqrt(10.0)  # prior SD shared by all three normals: Stan sqrt(10.0)

    model = pm.Model()
    with model:
        # Priors - identical to model_numpyro.py and to Stan's `model` block.
        c = pm.Normal("c", 0.0, root10)  # log scale / intercept
        r_alpha = pm.Normal("r_alpha", 0.0, root10, shape=n_w - 1)
        r_beta = pm.Normal("r_beta", 0.0, root10, shape=n_d - 1)

        # Identifiability pinning, as in Stan's transformed parameters:
        # alpha[1] = 0 AND beta[1] = 0. Note ODP pins the FIRST beta where the
        # Meyers family pins the LAST; the conventions are not interchangeable.
        alpha = pm.Deterministic("alpha", pt.concatenate([pt.zeros(1), r_alpha]))
        beta = pm.Deterministic("beta", pt.concatenate([pt.zeros(1), r_beta]))

        # log m[w,d] = logprem[w] + c + alpha[w] + beta[d] (E&V 7.11.8 log link).
        log_mu = pm.Deterministic("log_mu", logprem + c + alpha[w0] + beta[d0])

        # CustomDist with observed= : an honest observed variable, so PyMC can
        # emit log_likelihood. No `random=` is supplied because this variable is
        # never forward-sampled - prior/posterior predictive would need a true
        # od-Poisson sampler, which lives in the entry's predict().
        pm.CustomDist(
            "obs",
            pt.exp(log_mu),
            phi,
            logp=_odp_logp,
            observed=inc_loss,
        )
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
    """Sample the ODP posterior with PyMC's NUTS; return ``arviz.InferenceData``.

    ``cores=1`` by default - PyMC's multiprocessing chain execution is fragile
    on Windows. ``target_accept`` defaults to 0.8 to match the Stan entry.

    ``nuts_sampler`` selects the NUTS implementation over the *same* PyMC model
    graph (``"pymc"`` native is the parity reference; ``"nutpie"`` / ``"numpyro"``
    / ``"blackjax"`` are faster alternatives where installed).
    """
    import pymc as pm

    from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx

    ensure_pytensor_cxx()
    model = build_model(data)
    t0 = time.perf_counter()
    with model:
        # `init` left at pm.sample's default (jitter+adapt_diag): no custom
        # init, so the card's convergence comparison measures the model, not an
        # init trick. R-hat/ESS are computed once via arviz in
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
