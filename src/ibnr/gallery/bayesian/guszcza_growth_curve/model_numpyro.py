"""Hierarchical growth-curve reserving (Guszcza / Gesmann) - NumPyro port of
the Stan reference (``model.stan``).

Same model, same parameterization, held constant for parity (design decision
7): the same Stan ``data`` block the entry assembles is consumed verbatim -
ages in years, the cumulative paid loss ratios and the curve code included, so
a port cannot drift on the age convention or fit the other curve.

Three constructs needed care, and two of them are about *zero*.

**The zero this model does NOT have.** ``clark_growth_curve`` needs a
gradient-safe zero-age branch because its mid-period ``age_lo`` clamps to
exactly 0 on every origin's first cell. This model has no such branch and needs
none: ``t = d * dev_grain_months / 12`` with ``d >= 1``, so every age is
strictly positive, and ``model.stan``'s ``growth_curve`` is written without the
``if (x <= 0)`` guard Clark's carries. The port mirrors that literally rather
than importing Clark's guarded version, so the two files can be read against
each other line by line; ``tests/test_parity_guszcza.py`` pins the two curves
elementwise for ``t > 0``.

Stan's ``data`` block nonetheless declares ``vector<lower=0>[len_data] t``,
which is LOOSER than the invariant the function relies on. At ``t = 0`` the
loglogistic's ``(theta/t)^omega`` is ``inf``, and while the VALUE survives
(``G -> 0``) the derivative does not: ``dG/dtheta`` and ``dG/domega`` are both
NaN, and NUTS dies on the first leapfrog with no useful message.
``_shared.check_data`` therefore refuses a non-positive age by name. ``t`` is
data, so the check is static and costs nothing.

**The zero that DOES bite here is ``ulr``.** The accident-year effect is
additive on the ulr scale::

    ulr[w] = ulr_pop + sd_ulr * z_ulr[w]

with no lower bound - exactly as brms builds an nlpar's linear predictor - so a
draw can push ``ulr[w]`` non-positive. Stan then evaluates ``log()`` of a
non-positive number, raises a domain error and REJECTS the proposal, and that
rejection is what confines the posterior to the positive orthant. It is a real
part of the specification, not an edge case: it is why every retained draw has
``ulr[w] > 0``, which ``predict()`` and the held-out scorer both rely on (see
``scorer.check_ulr_positive``).

A port has to reproduce the rejection, and the only faithful spelling is a
log-density of ``-inf``. Getting there needs a safe dummy substituted INSIDE the
``log``, because the NaN that ``log()`` of a non-positive number produces does
not stay put: ``mu`` feeds ``LogNormal(mu, sigma).log_prob(y)``, so a single NaN
cell makes the whole observed site NaN, and ``NaN + (-inf)`` is ``NaN``, not
``-inf``. Measured on the shipped model against an unsubstituted one at a point
where every ``ulr`` is negative - substituted: density ``-inf``, all six
gradients finite; unsubstituted: density NaN and all six gradients NaN.

**The mechanism differs from ``clark_growth_curve``'s, even though the fix is
the same shape**, and the difference decides where the substitution has to go.
There the NaN comes from ``theta/0 -> inf`` inside an unselected ``where``
branch, so ``0 * inf`` poisons the reverse pass and masking the result is not
enough. Here ``d/dx log(x) = 1/x`` is perfectly finite for ``x < 0``, so masking
only the RESULT of the ``log`` does keep the gradient finite - it is the NaN
reaching the LIKELIHOOD that does the damage. Both halves of that are pinned by
a negative control in ``tests/test_parity_guszcza.py``; the first version of
that control masked ``mu.sum()`` instead of feeding the likelihood, and it
passed on the broken model.

The rejection itself rides on a separate factor that is piecewise constant in
every parameter, so its own gradient is identically zero.

**Half-Student-t scales.** Stan writes ``real<lower=0> sd_ulr;`` with
``sd_ulr ~ student_t(3, 0, 1);`` - a constrained parameter carrying an
unnormalized density, i.e. a half Student-t. ``dist.TruncatedDistribution(
dist.StudentT(...), low=0)`` does not work: it needs the Student-t CDF, which
pulls in ``betaincinv`` and raises ``ImportError: Please install
tensorflow_probability>=0.18``. So this port uses Stan's own construction, as
``compartmental``'s does - a positive-constrained ``ImproperUniform`` site plus
the density as a ``numpyro.factor``.

``sample()`` returns an ``arviz.InferenceData`` whose ``posterior`` group
carries the same ``ulr`` Stan exposes in ``transformed parameters`` - the
held-out scorer reads it from there, so a port that left it out would fit but
not score.
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

__all__ = ["MAX_TREE_DEPTH", "growth_curve", "guszcza_model", "sample"]


def growth_curve(t, omega, theta, curve: int):
    """G(t): the fraction of ultimate emerged by development age ``t`` YEARS.

    ``loglogistic``  G = 1 / (1 + (theta/t)^omega)
    ``weibull``      G = 1 - exp(-(t/theta)^omega)

    Written exactly as ``model.stan``'s ``growth_curve``, which has **no**
    zero-age branch because ``t > 0`` always holds here - unlike the Clark
    entries' curve, whose ``age_lo`` clamps to 0. ``_shared.check_data``
    enforces the invariant this relies on; see the module docstring.

    ``curve`` is DATA (a Python int), so the branch is resolved at trace time
    and never enters the computation graph.
    """
    import jax.numpy as jnp

    if curve == LOGLOGISTIC:
        return 1.0 / (1.0 + (theta / t) ** omega)
    if curve == WEIBULL:
        return 1.0 - jnp.exp(-((t / theta) ** omega))
    raise ValueError(f"curve must be {LOGLOGISTIC} (loglogistic) or {WEIBULL} (weibull)")


def _half_student_t(name: str, nu: float, scale: float):
    """A scalar half-Student-t site, spelled the way Stan spells it.

    ``ImproperUniform`` on the positive half-line supplies the constraint (and
    so the log transform and its Jacobian); ``numpyro.factor`` supplies the
    unnormalized density. That is exactly what ``real<lower=0> x;`` plus
    ``x ~ student_t(nu, 0, scale);`` means in Stan, and it sidesteps
    ``TruncatedDistribution``'s missing Student-t CDF (module docstring). The
    truncation constant is a constant, so it cannot move the posterior.
    """
    import numpyro
    import numpyro.distributions as dist

    x = numpyro.sample(name, dist.ImproperUniform(dist.constraints.positive, (), ()))
    numpyro.factor(f"{name}_prior", dist.StudentT(nu, 0.0, scale).log_prob(x))
    return x


def guszcza_model(data: dict[str, Any]) -> None:
    """NumPyro model consuming the Stan data block the entry assembles.

    Mirrors ``model.stan``: the same priors (the magesblog post's ``my_priors``,
    less the company level that collapses for a single cohort), the same
    non-centered accident-year effect, and the same lognormal likelihood on
    cumulative paid loss ratios.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    w0 = np.asarray(data["w"], dtype=int) - 1  # 0-based accident-year index
    t_np = np.asarray(data["t"], dtype=float)
    y_np = np.asarray(data["y"], dtype=float)
    n_w = int(data["n_w"])
    curve = int(data["curve"])
    check_data(t_np, y_np)
    t = jnp.asarray(t_np)
    y = jnp.asarray(y_np)

    # Priors - the Stan `model` block, term for term. The <lower=0> bounds are
    # load-bearing in this family (CLAUDE.md on sd_ay and a_ig), so every one
    # of them is reproduced: ulr_pop and the two curve parameters carry theirs
    # through the distribution's own support, the two scales through
    # _half_student_t's positive constraint.
    ulr_pop = numpyro.sample("ulr_pop", dist.LogNormal(np.log(0.6), np.log(2.0)))
    # normal(2, 1) under <lower=0> is a truncated normal, NOT a half-normal
    # about 0 - the location stays at 2. TruncatedNormal is native here (it
    # needs only the normal CDF), and its normalizing constant is a constant,
    # so it cannot move the posterior away from Stan's unnormalized version.
    omega = numpyro.sample("omega", dist.TruncatedNormal(2.0, 1.0, low=0.0))
    theta = numpyro.sample("theta", dist.TruncatedNormal(4.0, 1.0, low=0.0))
    sd_ulr = _half_student_t("sd_ulr", 3.0, 1.0)
    z_ulr = numpyro.sample("z_ulr", dist.Normal(0.0, 1.0).expand([n_w]).to_event(1))
    sigma = _half_student_t("sigma", 3.0, 1.0)

    # Stan's `transformed parameters`, and the one variable the held-out scorer
    # reads out of `posterior` - so it must be a deterministic, not a local.
    ulr = numpyro.deterministic("ulr", ulr_pop + sd_ulr * z_ulr)

    # Stan's implicit truncation, made explicit. See the module docstring: the
    # substitution is INSIDE the log because a NaN mu would otherwise reach
    # LogNormal.log_prob and make the whole density NaN rather than -inf, and
    # the rejection rides on a separate factor that is piecewise constant in
    # every parameter.
    #
    # The condition covers EVERY origin, not just the ones a given cell touches
    # - the same choice scorer.check_ulr_positive makes and for the same reason:
    # the invariant being claimed is that the sampler rejected such draws for
    # all trained origins. Here they coincide, since every origin contributes at
    # least its first cell to `w`.
    positive = ulr > 0
    safe_ulr = jnp.where(positive, ulr, 1.0)
    mu = jnp.log(safe_ulr[w0] * growth_curve(t, omega, theta, curve))
    numpyro.factor("ulr_support", jnp.where(jnp.all(positive), 0.0, -jnp.inf))

    numpyro.sample("y", dist.LogNormal(mu, sigma), obs=y)


def sample(
    data: dict[str, Any],
    *,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2500,
    seed: int | None = None,
    target_accept: float = 0.999,
    max_treedepth: int = MAX_TREE_DEPTH,
    chain_method: str = "sequential",
    progress_bar: bool = False,
):
    """Sample the Guszcza posterior with NUTS; return an ``arviz.InferenceData``.

    ``target_accept`` and ``max_treedepth`` default to the Stan entry's own
    settings (the post's ``control`` list), so a parity run holds the sampler
    constant and any difference it finds is the model.

    Init strategy, mirroring ``model.stan``'s: ``z_ulr`` starts at 0, so the
    initial ``ulr = ulr_pop`` is positive for every chain and the first ``mu``
    is finite; everything else keeps NumPyro's dispersed default so R-hat keeps
    its meaning. A PARTIAL init on purpose - the same one the Stan entry uses.
    """
    import arviz as az
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS, init_to_value

    numpyro.set_host_device_count(chains)
    seed = 0 if seed is None else int(seed)

    kernel = NUTS(
        guszcza_model,
        target_accept_prob=target_accept,
        max_tree_depth=max_treedepth,
        init_strategy=init_to_value(values={"z_ulr": np.zeros(int(data["n_w"]))}),
    )
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
