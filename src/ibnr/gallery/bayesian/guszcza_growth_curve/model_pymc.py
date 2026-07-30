"""Hierarchical growth-curve reserving (Guszcza / Gesmann) - PyMC port of the
Stan reference (``model.stan``).

Same model, same parameterization, held constant for parity. The Stan ``data``
block the entry assembles is consumed verbatim - ages in years, the cumulative
paid loss ratios and the curve code included.

The hazards are the NumPyro port's, and so are the fixes; see ``model_numpyro``
for the full account. In short:

1. **No zero-age branch, deliberately.** ``t = d * dev_grain_months / 12`` with
   ``d >= 1``, so every age is strictly positive and ``model.stan``'s
   ``growth_curve`` omits the ``if (x <= 0)`` guard the Clark entries need.
   ``_shared.check_data`` refuses a non-positive age rather than letting the
   loglogistic's NaN gradient at ``t = 0`` surface as an unexplained NUTS death.
2. **``ulr`` is unbounded and Stan rejects the draws that go non-positive.**
   That rejection is part of the specification - it is what makes every retained
   draw safe for ``predict()`` and the scorer - so the port reproduces it as a
   ``-inf`` ``pm.Potential``, with a safe dummy substituted INSIDE the ``log``.
   The substitution has to be inside because a NaN ``mu`` would otherwise reach
   the observed ``LogNormal`` and make the whole density NaN instead of ``-inf``;
   masking only the ``log``'s result is NOT enough for that, though it would be
   enough to keep the gradient finite. The mechanism is therefore not Clark's
   ``theta/0 -> inf``, and ``model_numpyro`` records the measurement.
3. **Half-Student-t scales.** ``pm.HalfStudentT(nu=3, sigma=1)`` is exactly
   Stan's ``real<lower=0> x;`` plus ``x ~ student_t(3, 0, 1);`` - the factor of
   two is a constant and cannot move the posterior. PyMC has the distribution
   natively, so unlike the NumPyro port this file needs no
   ``ImproperUniform``-plus-factor construction.

``build_model()`` returns the ``pm.Model`` graph; ``sample()`` runs NUTS and
returns an ``arviz.InferenceData`` carrying the same ``ulr`` Stan exposes in
``transformed parameters`` - the held-out scorer reads it from ``posterior``, so
a port that left it out would fit but not score.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ibnr.gallery.bayesian.guszcza_growth_curve._shared import (
    LOGLOGISTIC,
    MAX_TREE_DEPTH,
    WEIBULL,
    check_data,
)

__all__ = ["MAX_TREE_DEPTH", "build_model", "growth_curve", "sample"]


def growth_curve(t, omega, theta, curve: int):
    """G(t) on PyTensor tensors, written exactly as ``model.stan``'s.

    No zero-age branch - see the module docstring and ``_check_data``. ``curve``
    is data (a Python int), so the branch is resolved when the graph is built
    and never becomes a symbolic switch.
    """
    import pytensor.tensor as pt

    if curve == LOGLOGISTIC:
        return 1.0 / (1.0 + (theta / t) ** omega)
    if curve == WEIBULL:
        return 1.0 - pt.exp(-((t / theta) ** omega))
    raise ValueError(f"curve must be {LOGLOGISTIC} (loglogistic) or {WEIBULL} (weibull)")


def build_model(data: dict[str, Any]):
    """Construct the Guszcza ``pm.Model`` from the Stan data block."""
    import pymc as pm
    import pytensor.tensor as pt

    w0 = np.asarray(data["w"], dtype=int) - 1
    t = np.asarray(data["t"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    n_w = int(data["n_w"])
    curve = int(data["curve"])
    check_data(t, y)

    model = pm.Model()
    with model:
        # Priors - identical to model_numpyro.py and to Stan's `model` block.
        # Every <lower=0> bound is reproduced; they are load-bearing in this
        # family (CLAUDE.md on sd_ay and a_ig).
        ulr_pop = pm.LogNormal("ulr_pop", np.log(0.6), np.log(2.0))
        # normal(2, 1) under <lower=0> keeps its location at 2 - a truncated
        # normal, not a half-normal, which is the easiest thing to mis-port here
        # and is pinned by test_ports_match_the_stan_target's offset check.
        #
        # PyMC gives these an INTERVAL transform, not a log one - the value vars
        # are omega_interval__ / theta_interval__ (measured). CLAUDE.md's warning
        # about a one-sided bound concerns `pm.Truncated` with `upper=` alone,
        # whose transform jitter can push out of support; that does not arise
        # here, and jitter is disabled on both sampling paths anyway (see
        # `sample`), which is what the -inf ulr potential requires.
        omega = pm.TruncatedNormal("omega", mu=2.0, sigma=1.0, lower=0.0)
        theta = pm.TruncatedNormal("theta", mu=4.0, sigma=1.0, lower=0.0)
        sd_ulr = pm.HalfStudentT("sd_ulr", nu=3, sigma=1.0)
        z_ulr = pm.Normal("z_ulr", 0.0, 1.0, shape=n_w)
        sigma = pm.HalfStudentT("sigma", nu=3, sigma=1.0)

        # Stan's `transformed parameters`, and the one variable the held-out
        # scorer reads out of `posterior` - a Deterministic, not a local.
        ulr = pm.Deterministic("ulr", ulr_pop + sd_ulr * z_ulr)

        # Stan's implicit truncation, made explicit: the dummy is substituted
        # INSIDE the log so no NaN mu can reach the observed LogNormal (which
        # would make the density NaN rather than -inf), and the rejection rides
        # on a separate potential that is piecewise constant in every
        # parameter, so its gradient is zero.
        positive = pt.gt(ulr, 0.0)
        safe_ulr = pt.switch(positive, ulr, 1.0)
        mu = pt.log(safe_ulr[w0] * growth_curve(t, omega, theta, curve))
        pm.Potential("ulr_support", pt.switch(pt.all(positive), 0.0, -np.inf))

        pm.LogNormal("y", mu=mu, sigma=sigma, observed=y)
    return model


def sample(
    data: dict[str, Any],
    *,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2500,
    seed: int | None = None,
    target_accept: float = 0.999,
    max_treedepth: int = MAX_TREE_DEPTH,
    cores: int = 1,
    nuts_sampler: str = "pymc",
    progressbar: bool = False,
):
    """Sample the Guszcza posterior with PyMC's NUTS; return ``arviz.InferenceData``.

    ``cores=1`` by default - PyMC's multiprocessing chain execution is fragile
    on Windows. ``target_accept`` and ``max_treedepth`` default to the Stan
    entry's own settings (the post's ``control`` list) so a parity run holds the
    sampler constant.

    **PyMC's native NUTS is impractical for this entry at ``adapt_delta =
    0.999``, and the graph is fine** - the same situation ``compartmental``
    documents. Measured on the 8-origin synthetic fixture, 2 chains x (300 warmup
    + 300 draws): ``nuts_sampler="pymc"`` 860 s against NumPyro's own 6.7 s on
    the identical model, i.e. ~130x, because a step size tuned to 0.999 needs
    long trajectories and PyTensor pays per leapfrog. The default is left at
    ``"pymc"`` so this argument reports what actually ran; pass
    ``nuts_sampler="numpyro"`` to sample the SAME graph through JAX, which is
    what ``scripts/parity_gallery.py --nuts-sampler numpyro`` does for the
    published run.

    Two init details mirror ``model.stan``'s partial init, and both matter:
    ``z_ulr`` starts at 0 so the initial ``ulr = ulr_pop`` is positive and the
    first ``mu`` is finite, and ``init="adapt_diag"`` is chosen over PyMC's
    default ``jitter+adapt_diag`` because the jitter would perturb exactly that
    starting value - a large enough kick puts ``ulr`` below zero, where the
    potential is ``-inf`` and PyMC refuses to start at all. Everything else
    keeps its dispersed default, so R-hat keeps its meaning.
    """
    import pymc as pm

    from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx

    ensure_pytensor_cxx()
    model = build_model(data)
    initvals = {"z_ulr": np.zeros(int(data["n_w"]))}
    # The two sampler paths spell BOTH of these differently, and getting either
    # wrong is silent rather than loud, so they are written out per path:
    #
    #   max_treedepth  a NUTS step-method kwarg for PyMC's own sampler; for a
    #                  foreign one it has to travel as
    #                  nuts_sampler_kwargs["nuts_kwargs"]["max_tree_depth"],
    #                  because pm.sample hands nuts_sampler_kwargs straight to
    #                  sample_jax_nuts, whose own signature has no tree-depth
    #                  argument (passing it there is a TypeError, which is how
    #                  this was found).
    #   no jitter      PyMC's default init "jitter+adapt_diag" and
    #                  sample_jax_nuts's default jitter=True both perturb the
    #                  z_ulr = 0 start; a large enough kick puts ulr below zero,
    #                  where the potential is -inf and sampling cannot begin.
    extra: dict[str, Any] = (
        {"max_treedepth": max_treedepth, "init": "adapt_diag"}
        if nuts_sampler == "pymc"
        else {
            "nuts_sampler_kwargs": {
                "nuts_kwargs": {"max_tree_depth": max_treedepth},
                "jitter": False,
            }
        }
    )
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
            initvals=initvals,
            progressbar=progressbar,
            compute_convergence_checks=False,
            **extra,
        )
    runtime_s = time.perf_counter() - t0
    idata.attrs["runtime_s"] = runtime_s
    idata.attrs["backend"] = f"pymc:{nuts_sampler}" if nuts_sampler != "pymc" else "pymc"
    return idata
