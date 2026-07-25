"""Hierarchical compartmental reserving - PyMC ports of both Stan variants.

``build_gaussian`` mirrors ``model.stan`` (case-study Model 1) and
``build_lognormal`` mirrors ``model_lognormal.stan`` (Model 2). Both consume
the Stan ``data`` block the entry assembles, verbatim, and hold the monograph's
priors and parameterization constant for parity.

The correlated accident-year block maps onto PyMC more neatly than onto
NumPyro. Stan writes

    vector<lower=0>[2] sd_ay;  cholesky_factor_corr[2] L_ay;
    u_ay = diag_pre_multiply(sd_ay, L_ay) * z_ay;

and ``pm.LKJCholeskyCov(..., sd_dist=pm.HalfStudentT.dist(...))`` is exactly
that pair: it places LKJ(eta) on the correlation and the given ``sd_dist``
independently on the scales, and its ``chol`` output *is*
``diag_pre_multiply(sd, L_corr)``. Verified against Stan's prior before use -
sampled rho has sd 0.5774 (uniform on (-1, 1), as LKJ(1) implies in two
dimensions) and the scales' medians come back at 0.1398 / 0.0703 against the
analytic half-Student-t 0.1400 / 0.0700.

``pm.HalfStudentT(nu, sigma)`` is Stan's ``student_t(nu, 0, sigma)`` truncated
at 0: same parameterization, checked to three decimals on the quantiles.

The ODE system is solved in CLOSED FORM (no integrator in any backend); the
curves below mirror the Stan ``functions`` block and ``model.py``'s numpy pair.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

VARIANTS = ("gaussian", "lognormal")


def os_curve(t, ker, kp, rlr):
    """Outstanding loss ratio at age ``t`` years - the Stan ``os_curve``."""
    import pytensor.tensor as pt

    return rlr * ker / (ker - kp) * (pt.exp(-kp * t) - pt.exp(-ker * t))


def paid_curve(t, ker, kp, rlr, rrf):
    """Cumulative paid loss ratio at age ``t`` years - the Stan ``paid_curve``.
    Exactly 0 at t = 0, which the incremental differencing below relies on."""
    import pytensor.tensor as pt

    return (
        rlr * rrf / (ker - kp) * (ker * (1 - pt.exp(-kp * t)) - kp * (1 - pt.exp(-ker * t)))
    )


def _correlated_ay_effects(n_w: int, sd_scales, nu: float = 10.0):
    """The shared (oRLR, oRRF) accident-year block, as Stan spells it.

    ``LKJCholeskyCov``'s ``chol`` is ``diag_pre_multiply(sd_ay, L_ay)``, so the
    non-centered product below is Stan's line verbatim. Returns
    ``u_ay`` of shape (2, n_w); ``rho_ay`` is registered as a deterministic.
    """
    import pymc as pm

    chol, corr, _sds = pm.LKJCholeskyCov(
        "ay_chol",
        n=2,
        eta=1.0,  # Stan's lkj_corr_cholesky(1): uniform over correlations
        sd_dist=pm.HalfStudentT.dist(nu=nu, sigma=np.asarray(sd_scales, dtype=float)),
        compute_corr=True,
    )
    z_ay = pm.Normal("z_ay", 0.0, 1.0, shape=(2, n_w))
    u_ay = pm.Deterministic("u_ay", chol @ z_ay)
    pm.Deterministic("rho_ay", corr[0, 1])
    return u_ay


def build_gaussian(data: dict[str, Any]):
    """Case-study Model 1: Gaussian on OS levels + cumulative paid AMOUNTS."""
    import pymc as pm
    import pytensor.tensor as pt

    n_w = int(data["n_w"])
    w0 = np.asarray(data["w"], dtype=int) - 1
    t = np.asarray(data["t"], dtype=float)
    delta = np.asarray(data["delta"], dtype=int)
    loss = np.asarray(data["loss"], dtype=float)
    premium = np.asarray(data["premium"], dtype=float)
    is_os = delta == 0  # static mask: delta is data

    model = pm.Model()
    with model:
        b_orlr = pm.Normal("b_oRLR", 0.0, 1.0)
        b_orrf = pm.Normal("b_oRRF", 0.0, 1.0)
        b_oker = pm.Normal("b_oker", 0.0, 1.0)
        b_okp = pm.Normal("b_okp", 0.0, 1.0)

        u_ay = _correlated_ay_effects(n_w, [0.2, 0.1])

        # brms nlf transforms, verbatim
        ker = pm.Deterministic("ker", 3.0 * pt.exp(0.1 * b_oker))
        kp = pm.Deterministic("kp", 1.0 * pt.exp(0.1 * b_okp))
        rlr = pm.Deterministic("RLR", 0.7 * pt.exp(0.2 * (b_orlr + u_ay[0])))
        rrf = pm.Deterministic("RRF", 0.8 * pt.exp(0.1 * (b_orrf + u_ay[1])))

        # brms class-b prior on the log-sigma coefficients (Cauchy, effectively
        # flat on the amount scale)
        log_sigma_os = pm.StudentT("log_sigma_os", nu=1.0, mu=0.0, sigma=1000.0)
        log_sigma_paid = pm.StudentT("log_sigma_paid", nu=1.0, mu=0.0, sigma=1000.0)
        pm.Deterministic("sigma_os", pt.exp(log_sigma_os))
        pm.Deterministic("sigma_paid", pt.exp(log_sigma_paid))

        os_lr = os_curve(t, ker, kp, rlr[w0])
        pd_lr = paid_curve(t, ker, kp, rlr[w0], rrf[w0])
        mu = pm.Deterministic("mu", premium[w0] * pt.where(is_os, os_lr, pd_lr))
        sigma = pt.where(is_os, pt.exp(log_sigma_os), pt.exp(log_sigma_paid))
        pm.Normal("obs", mu=mu, sigma=sigma, observed=loss)
    return model


def build_lognormal(data: dict[str, Any]):
    """Case-study Model 2: lognormal on OS levels + INCREMENTAL paid LOSS RATIOS."""
    import pymc as pm
    import pytensor.tensor as pt

    n_w, n_d = int(data["n_w"]), int(data["n_d"])
    w0 = np.asarray(data["w"], dtype=int) - 1
    d0 = np.asarray(data["d"], dtype=int) - 1
    t = np.asarray(data["t"], dtype=float)
    delta = np.asarray(data["delta"], dtype=int)
    y = np.asarray(data["y"], dtype=float)
    devfreq = float(data["devfreq"])
    is_os = delta == 0

    model = pm.Model()
    with model:
        b_orlr = pm.Normal("b_oRLR", 0.0, 1.0)
        b_orrf = pm.Normal("b_oRRF", 0.0, 1.0)
        b_oker = pm.Normal("b_oker", 0.0, 1.0)
        b_okp = pm.Normal("b_okp", 0.0, 1.0)

        # Model 2 widens the AY scales and reuses each parameter's prior for
        # its dev grouping.
        u_ay = _correlated_ay_effects(n_w, [0.7, 0.5])
        sd_dev = pm.HalfStudentT("sd_dev", nu=10.0, sigma=np.array([0.7, 0.5]), shape=2)
        sd_ker = pm.HalfStudentT("sd_ker", nu=10.0, sigma=0.3, shape=2)
        sd_kp = pm.HalfStudentT("sd_kp", nu=10.0, sigma=0.3, shape=2)

        z_rlr_dev = pm.Normal("z_RLR_dev", 0.0, 1.0, shape=n_d)
        z_rrf_dev = pm.Normal("z_RRF_dev", 0.0, 1.0, shape=n_d)
        z_ker_ay = pm.Normal("z_ker_ay", 0.0, 1.0, shape=n_w)
        z_ker_dev = pm.Normal("z_ker_dev", 0.0, 1.0, shape=n_d)
        z_kp_ay = pm.Normal("z_kp_ay", 0.0, 1.0, shape=n_w)
        z_kp_dev = pm.Normal("z_kp_dev", 0.0, 1.0, shape=n_d)

        u_rlr_dev = sd_dev[0] * z_rlr_dev
        u_rrf_dev = sd_dev[1] * z_rrf_dev
        u_ker_ay, u_ker_dev = sd_ker[0] * z_ker_ay, sd_ker[1] * z_ker_dev
        u_kp_ay, u_kp_dev = sd_kp[0] * z_kp_ay, sd_kp[1] * z_kp_dev

        # per-CELL compartmental parameters
        ker = 3.0 * pt.exp(0.1 * (b_oker + u_ker_ay[w0] + u_ker_dev[d0]))
        kp = 1.0 * pt.exp(0.1 * (b_okp + u_kp_ay[w0] + u_kp_dev[d0]))
        rlr = 0.7 * pt.exp(0.2 * (b_orlr + u_ay[0][w0] + u_rlr_dev[d0]))
        rrf = 0.8 * pt.exp(0.1 * (b_orrf + u_ay[1][w0] + u_rrf_dev[d0]))

        # The monograph's appendix CODE, not its text: normal(log 0.2, 0.2).
        log_sigma_os = pm.Normal("log_sigma_os", np.log(0.2), 0.2)
        log_sigma_paid = pm.Normal("log_sigma_paid", np.log(0.2), 0.2)
        # same names as Stan's generated quantities, so parity compares the
        # same scalars in both variants
        pm.Deterministic("sigma_os", pt.exp(log_sigma_os))
        pm.Deterministic("sigma_paid", pt.exp(log_sigma_paid))

        # Incremental paid over (t - devfreq, t], same cell's parameters both
        # ends. Clamping the earlier age at 0 is identical to Stan's
        # ``t > devfreq`` branch because paid_curve(0) = 0 exactly.
        prev_age = np.maximum(t - devfreq, 0.0)
        paid_incr = paid_curve(t, ker, kp, rlr, rrf) - paid_curve(prev_age, ker, kp, rlr, rrf)
        mu = pm.Deterministic("mu", pt.where(is_os, os_curve(t, ker, kp, rlr), paid_incr))
        sigma = pt.where(is_os, pt.exp(log_sigma_os), pt.exp(log_sigma_paid))
        pm.LogNormal("obs", mu=pt.log(mu), sigma=sigma, observed=y)
    return model


def build_model(data: dict[str, Any], *, variant: str = "gaussian"):
    """Dispatch to the requested variant's graph."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return build_gaussian(data) if variant == "gaussian" else build_lognormal(data)


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
    cores: int = 1,
    nuts_sampler: str = "pymc",
    progressbar: bool = False,
):
    """Sample either variant with PyMC's NUTS; return ``arviz.InferenceData``.

    ``target_accept = 0.99`` / ``max_treedepth = 15`` are the MONOGRAPH's own
    settings, matching the Stan entry - this is the stiffest posterior in the
    gallery and the published results were produced at them.
    """
    import pymc as pm

    from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx

    ensure_pytensor_cxx()
    model = build_model(data, variant=variant)
    t0 = time.perf_counter()
    with model:
        idata = pm.sample(
            draws=iter_sampling,
            tune=iter_warmup,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            nuts={"max_treedepth": max_treedepth},
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
