"""Hierarchical compartmental reserving - NumPyro ports of both Stan variants.

``gaussian_model`` mirrors ``model.stan`` (case-study Model 1) and
``lognormal_model`` mirrors ``model_lognormal.stan`` (Model 2). Both consume
the Stan ``data`` block the entry assembles, verbatim, and hold the monograph's
priors and parameterization constant for parity (design decision 7) - these are
the comparison baseline, so nothing here may be retuned.

Three constructs needed care, all verified empirically before use:

**Half-Student-t scales.** Stan writes ``vector<lower=0>[2] sd_ay;`` with
``sd_ay[1] ~ student_t(10, 0, 0.2);`` - a constrained parameter carrying an
unnormalized density, i.e. a HALF Student-t. The obvious NumPyro spelling,
``dist.TruncatedDistribution(dist.StudentT(...), low=0)``, **does not work
here**: it needs the Student-t CDF, which pulls in ``betaincinv`` and raises
``ImportError: Please install tensorflow_probability>=0.18``. So the ports use
Stan's own construction instead - a positive-constrained ``ImproperUniform``
site plus the density as a ``numpyro.factor``. Checked against the analytic
half-t quantiles (sd 0.2: median 0.138 vs 0.140, q90 0.358 vs 0.363).

**LKJ on a 2x2 Cholesky factor.** ``dist.LKJCholesky(2, concentration=1)``
matches Stan's ``lkj_corr_cholesky(1)``: sampled rho has mean 0.000 and sd
0.5774, i.e. exactly uniform on (-1, 1) (sd = 1/sqrt(3)), as LKJ(1) implies in
two dimensions.

**Non-centered varying effects.** ``u = diag_pre_multiply(sd, L) @ z`` is kept
exactly as brms writes it. A centered version is a different geometry at finite
sample size, not the same model - the Stan file says so explicitly and the
ports must not "simplify" it.

The compartmental ODE system is solved in CLOSED FORM (the monograph's own
simplification for two rates), so there is no integrator in any backend. The
curves below are the jnp mirrors of the Stan ``functions`` block and of
``model.py``'s numpy pair; all three must stay in step.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

#: Stan's variant names, mirrored so a caller cannot ask for a third.
VARIANTS = ("gaussian", "lognormal")


def os_curve(t, ker, kp, rlr):
    """Outstanding loss ratio at age ``t`` years - the Stan ``os_curve``.

    A hump: case reserves build at the reporting rate ``ker`` and empty at the
    settlement rate ``kp``, so OS -> 0 as t -> inf. Singular at ker == kp; the
    priors (medians 3 and 1) keep the sampler away from that ridge.
    """
    import jax.numpy as jnp

    return rlr * ker / (ker - kp) * (jnp.exp(-kp * t) - jnp.exp(-ker * t))


def paid_curve(t, ker, kp, rlr, rrf):
    """Cumulative paid loss ratio at age ``t`` years - the Stan ``paid_curve``.

    The integral of ``kp * RRF * OS``; tends to ``RLR * RRF`` (the ultimate
    loss ratio) as t -> inf, and is exactly 0 at t = 0.
    """
    import jax.numpy as jnp

    return rlr * rrf / (ker - kp) * (ker * (1 - jnp.exp(-kp * t)) - kp * (1 - jnp.exp(-ker * t)))


def _half_student_t(name: str, nu: float, scale, shape: tuple[int, ...]):
    """A half-Student-t site, spelled the way Stan spells it.

    ``ImproperUniform`` on the positive orthant supplies the constraint (and so
    the log transform and its Jacobian); ``numpyro.factor`` supplies the
    unnormalized density. That is exactly what ``vector<lower=0> x;`` plus
    ``x ~ student_t(nu, 0, scale);`` means in Stan, and it sidesteps
    ``TruncatedDistribution``'s missing Student-t CDF (see the module
    docstring). The truncation constant is a constant, so it cannot move the
    posterior.
    """
    import numpyro
    import numpyro.distributions as dist

    x = numpyro.sample(name, dist.ImproperUniform(dist.constraints.positive, (), shape))
    numpyro.factor(f"{name}_prior", dist.StudentT(nu, 0.0, scale).log_prob(x).sum())
    return x


def _correlated_ay_effects(n_w: int, sd_scales, nu: float = 10.0):
    """The shared (oRLR, oRRF) accident-year block: half-t scales, an LKJ(1)
    correlation and non-centered standard normals.

    Identical in both variants - Model 2 only widens the scales - so it lives
    in one place. Returns ``(u_ay, sd_ay)`` with ``u_ay`` of shape (2, n_w).
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    sd_ay = _half_student_t("sd_ay", nu, jnp.asarray(sd_scales), (2,))
    l_ay = numpyro.sample("L_ay", dist.LKJCholesky(2, concentration=1.0))
    z_ay = numpyro.sample("z_ay", dist.Normal(0.0, 1.0).expand([2, n_w]).to_event(2))
    # diag_pre_multiply(sd_ay, L_ay) @ z_ay, exactly as Stan writes it
    u_ay = numpyro.deterministic("u_ay", (sd_ay[:, None] * l_ay) @ z_ay)
    # the reserving-cycle correlation the monograph reports
    numpyro.deterministic("rho_ay", (l_ay @ l_ay.T)[0, 1])
    return u_ay, sd_ay


def gaussian_model(data: dict[str, Any]) -> None:
    """Case-study Model 1: Gaussian on OS levels + cumulative paid AMOUNTS.

    Mirrors ``model.stan`` term for term - ker and kp are fixed across accident
    years here, only (RLR, RRF) carry varying effects, and the residual scale
    is a constant per block on the amount scale.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n_w = int(data["n_w"])
    w0 = np.asarray(data["w"], dtype=int) - 1
    t = jnp.asarray(np.asarray(data["t"], dtype=float))
    delta = jnp.asarray(np.asarray(data["delta"], dtype=float))
    loss = jnp.asarray(np.asarray(data["loss"], dtype=float))
    premium = jnp.asarray(np.asarray(data["premium"], dtype=float))

    # population-level effects on the unconstrained (o-prefixed) scale
    b_orlr = numpyro.sample("b_oRLR", dist.Normal(0.0, 1.0))
    b_orrf = numpyro.sample("b_oRRF", dist.Normal(0.0, 1.0))
    b_oker = numpyro.sample("b_oker", dist.Normal(0.0, 1.0))
    b_okp = numpyro.sample("b_okp", dist.Normal(0.0, 1.0))

    u_ay, _ = _correlated_ay_effects(n_w, [0.2, 0.1])

    # brms nlf transforms, verbatim: lognormal medians (3, 1, 0.7, 0.8)
    ker = numpyro.deterministic("ker", 3.0 * jnp.exp(0.1 * b_oker))
    kp = numpyro.deterministic("kp", 1.0 * jnp.exp(0.1 * b_okp))
    rlr = numpyro.deterministic("RLR", 0.7 * jnp.exp(0.2 * (b_orlr + u_ay[0])))
    rrf = numpyro.deterministic("RRF", 0.8 * jnp.exp(0.1 * (b_orrf + u_ay[1])))

    # brms class-b prior on the log-sigma coefficients: student_t(1, .) is a
    # Cauchy, and at scale 1000 on the AMOUNT scale it is effectively flat.
    log_sigma_os = numpyro.sample("log_sigma_os", dist.StudentT(1.0, 0.0, 1000.0))
    log_sigma_paid = numpyro.sample("log_sigma_paid", dist.StudentT(1.0, 0.0, 1000.0))
    numpyro.deterministic("sigma_os", jnp.exp(log_sigma_os))
    numpyro.deterministic("sigma_paid", jnp.exp(log_sigma_paid))

    # One shared curve system; delta picks the compartment the row observes.
    # Both branches are finite everywhere the priors reach, so a plain where is
    # safe here (unlike the Clark entry's zero-age curve).
    os_lr = os_curve(t, ker, kp, rlr[w0])
    pd_lr = paid_curve(t, ker, kp, rlr[w0], rrf[w0])
    mu = numpyro.deterministic("mu", premium[w0] * jnp.where(delta == 0, os_lr, pd_lr))
    sigma = jnp.where(delta == 0, jnp.exp(log_sigma_os), jnp.exp(log_sigma_paid))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=loss)


def lognormal_model(data: dict[str, Any]) -> None:
    """Case-study Model 2: lognormal on OS levels + INCREMENTAL paid LOSS RATIOS.

    Mirrors ``model_lognormal.stan``: ker and kp now carry accident-year AND
    development effects, every non-(RLR, RRF) effect is independent, and the
    residual scale is a relative (log-scale) CV rather than a dollar width.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n_w, n_d = int(data["n_w"]), int(data["n_d"])
    w0 = np.asarray(data["w"], dtype=int) - 1
    d0 = np.asarray(data["d"], dtype=int) - 1
    t = jnp.asarray(np.asarray(data["t"], dtype=float))
    delta = jnp.asarray(np.asarray(data["delta"], dtype=float))
    y = jnp.asarray(np.asarray(data["y"], dtype=float))
    devfreq = float(data["devfreq"])

    b_orlr = numpyro.sample("b_oRLR", dist.Normal(0.0, 1.0))
    b_orrf = numpyro.sample("b_oRRF", dist.Normal(0.0, 1.0))
    b_oker = numpyro.sample("b_oker", dist.Normal(0.0, 1.0))
    b_okp = numpyro.sample("b_okp", dist.Normal(0.0, 1.0))

    # Model 2 widens the AY scales (0.7/0.5 vs Model 1's 0.2/0.1) and uses the
    # same prior for the dev grouping of each parameter.
    u_ay, _ = _correlated_ay_effects(n_w, [0.7, 0.5])
    sd_dev = _half_student_t("sd_dev", 10.0, jnp.asarray([0.7, 0.5]), (2,))
    sd_ker = _half_student_t("sd_ker", 10.0, 0.3, (2,))
    sd_kp = _half_student_t("sd_kp", 10.0, 0.3, (2,))

    z_rlr_dev = numpyro.sample("z_RLR_dev", dist.Normal(0.0, 1.0).expand([n_d]).to_event(1))
    z_rrf_dev = numpyro.sample("z_RRF_dev", dist.Normal(0.0, 1.0).expand([n_d]).to_event(1))
    z_ker_ay = numpyro.sample("z_ker_ay", dist.Normal(0.0, 1.0).expand([n_w]).to_event(1))
    z_ker_dev = numpyro.sample("z_ker_dev", dist.Normal(0.0, 1.0).expand([n_d]).to_event(1))
    z_kp_ay = numpyro.sample("z_kp_ay", dist.Normal(0.0, 1.0).expand([n_w]).to_event(1))
    z_kp_dev = numpyro.sample("z_kp_dev", dist.Normal(0.0, 1.0).expand([n_d]).to_event(1))

    u_rlr_dev = sd_dev[0] * z_rlr_dev
    u_rrf_dev = sd_dev[1] * z_rrf_dev
    u_ker_ay, u_ker_dev = sd_ker[0] * z_ker_ay, sd_ker[1] * z_ker_dev
    u_kp_ay, u_kp_dev = sd_kp[0] * z_kp_ay, sd_kp[1] * z_kp_dev

    # per-CELL compartmental parameters (Model 1 has one set per accident year)
    ker = 3.0 * jnp.exp(0.1 * (b_oker + u_ker_ay[w0] + u_ker_dev[d0]))
    kp = 1.0 * jnp.exp(0.1 * (b_okp + u_kp_ay[w0] + u_kp_dev[d0]))
    rlr = 0.7 * jnp.exp(0.2 * (b_orlr + u_ay[0][w0] + u_rlr_dev[d0]))
    rrf = 0.8 * jnp.exp(0.1 * (b_orrf + u_ay[1][w0] + u_rrf_dev[d0]))

    # sigma is a relative CV here, so the monograph gives it a genuinely
    # informative prior. NOTE its appendix CODE uses normal(log 0.2, 0.2) while
    # the TEXT quotes LN(log 0.1, 0.2); the code produced the published
    # results and wins - the Stan file says so and the port must not drift.
    log_sigma_os = numpyro.sample("log_sigma_os", dist.Normal(np.log(0.2), 0.2))
    log_sigma_paid = numpyro.sample("log_sigma_paid", dist.Normal(np.log(0.2), 0.2))
    # exposed under the same names as Stan's generated quantities, so parity
    # compares the same scalars in both variants
    numpyro.deterministic("sigma_os", jnp.exp(log_sigma_os))
    numpyro.deterministic("sigma_paid", jnp.exp(log_sigma_paid))

    # Incremental paid over (t - devfreq, t], differenced with the SAME cell's
    # parameters at both ends. Stan branches on ``t > devfreq``; clamping the
    # earlier age at 0 is algebraically identical, because paid_curve(0) = 0
    # exactly, and it keeps the expression branch-free.
    prev_age = jnp.maximum(t - devfreq, 0.0)
    paid_incr = paid_curve(t, ker, kp, rlr, rrf) - paid_curve(prev_age, ker, kp, rlr, rrf)
    mu = numpyro.deterministic("mu", jnp.where(delta == 0, os_curve(t, ker, kp, rlr), paid_incr))
    sigma = jnp.where(delta == 0, jnp.exp(log_sigma_os), jnp.exp(log_sigma_paid))
    numpyro.sample("obs", dist.LogNormal(jnp.log(mu), sigma), obs=y)


def sample(
    data: dict[str, Any],
    *,
    variant: str = "gaussian",
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2500,
    seed: int | None = None,
    target_accept: float = 0.99,
    max_treedepth: int = 15,
    chain_method: str = "sequential",
    progress_bar: bool = False,
):
    """Sample either variant with NUTS; return an ``arviz.InferenceData``.

    ``target_accept = 0.99`` and ``max_treedepth = 15`` are the MONOGRAPH's own
    sampler settings, and they are the defaults here for the same reason the
    Stan entry uses them: this is the stiffest posterior in the gallery
    (correlated varying effects over a compartmental ODE solution), and the
    published results were produced at these settings.
    """
    import arviz as az
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    numpyro.set_host_device_count(chains)
    seed = 0 if seed is None else int(seed)

    model = gaussian_model if variant == "gaussian" else lognormal_model
    kernel = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_treedepth)
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
