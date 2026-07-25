"""Bayesian Clark growth curve - PyMC port of the Stan reference (``model.stan``).

Same model, same parameterization, held constant for parity. The Stan ``data``
block the entry assembles is consumed verbatim - ages, plug-in ``phi`` and the
curve code included.

The two zero-related hazards are identical to the NumPyro port's, and so are
the fixes; see ``model_numpyro`` for the full account. In short:

1. ``age_lo`` is exactly 0 for every origin's first cell, and a bare
   ``pt.switch(x > 0, formula, 0.0)`` computes both branches, so the NaN from
   ``theta/0`` inside the unselected one propagates through the gradient. The
   loglogistic curve - the default - is affected. Both branches therefore
   evaluate at a safe substituted age.
2. Stan's ``odp_lpdf`` is duplicated rather than imported from
   ``england_verrall_odp``: ``model.stan`` duplicates it too, and each model
   directory must stand alone for ``gallery.scaffold()`` (decision 6).

``build_model()`` returns the ``pm.Model`` graph; ``sample()`` runs NUTS and
returns an ``arviz.InferenceData`` carrying the same deterministic ``mu`` Stan
exposes in ``transformed parameters``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

#: Stan's integer curve codes (model.stan ``curve``).
LOGLOGISTIC, WEIBULL = 1, 2


def growth_curve(x, omega, theta, curve: int):
    """Clark's G(x) on PyTensor tensors; G(x) = 0 for x <= 0, as in Stan.

    ``curve`` is data (a Python int), so the branch is resolved when the graph
    is built and never becomes a symbolic switch. The ``safe_x`` substitution
    keeps the gradient finite at x = 0 - see the module docstring.
    """
    import pytensor.tensor as pt

    positive = pt.gt(x, 0.0)
    safe_x = pt.switch(positive, x, 1.0)
    if curve == LOGLOGISTIC:
        g = 1.0 / (1.0 + (theta / safe_x) ** omega)
    elif curve == WEIBULL:
        g = 1.0 - pt.exp(-((safe_x / theta) ** omega))
    else:  # pragma: no cover - the entry validates before dispatch
        raise ValueError(f"curve must be {LOGLOGISTIC} (loglogistic) or {WEIBULL} (weibull)")
    return pt.switch(positive, g, 0.0)


def build_model(data: dict[str, Any]):
    """Construct the Clark ``pm.Model`` from the Stan data block."""
    import pymc as pm
    import pytensor.tensor as pt

    w0 = np.asarray(data["w"], dtype=int) - 1
    age_lo = np.asarray(data["age_lo"], dtype=float)
    age_hi = np.asarray(data["age_hi"], dtype=float)
    inc_loss = np.asarray(data["inc_loss"], dtype=float)
    logprem_w = np.asarray(data["logprem_w"], dtype=float)
    phi = float(data["phi"])
    curve = int(data["curve"])
    theta_prior_median = float(data["theta_prior_median"])

    model = pm.Model()
    with model:
        # Priors - identical to model_numpyro.py and to Stan's `model` block.
        logelr = pm.Normal("logelr", -0.4, np.sqrt(10.0))
        omega = pm.LogNormal("omega", np.log(1.5), 0.5)
        theta = pm.LogNormal("theta", np.log(theta_prior_median), 1.0)

        # Cape Cod mean times the growth-curve slice the cell's age span covers.
        emerged = growth_curve(age_hi, omega, theta, curve) - growth_curve(
            age_lo, omega, theta, curve
        )
        mu = pm.Deterministic("mu", pt.exp(logprem_w[w0] + logelr) * emerged)

        # Stan's odp_lpdf, term for term. pm.Potential rather than
        # pm.CustomDist(observed=): `mu` can legitimately reach 0 for a cell
        # whose age interval spans no emergence, which a distribution's support
        # check would reject outright instead of letting the density go to
        # -inf. The Stan reference carries its own `log_lik` in generated
        # quantities, so ELPD for this entry is read off the Stan fit.
        scaled = inc_loss / phi
        pm.Potential(
            "obs",
            pt.sum(scaled * pt.log(mu / phi) - mu / phi - pt.gammaln(scaled + 1.0)),
        )
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
    """Sample the Clark posterior with PyMC's NUTS; return ``arviz.InferenceData``.

    ``cores=1`` by default - PyMC's multiprocessing chain execution is fragile
    on Windows. ``target_accept`` defaults to 0.9 to match the Stan entry: the
    curve's shape and scale trade off along a ridge.
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
            compute_convergence_checks=False,
        )
    runtime_s = time.perf_counter() - t0
    idata.attrs["runtime_s"] = runtime_s
    idata.attrs["backend"] = f"pymc:{nuts_sampler}" if nuts_sampler != "pymc" else "pymc"
    return idata
