"""Cross-backend parity for guszcza_growth_curve (milestone 5).

Same two tiers as the other parity files (fast structural checks in the default
suite; a sampling gate under ``parity``/``slow``). The three-way run with Stan as
ground truth is ``scripts/parity_gallery.py --model guszcza_growth_curve``.

**This entry's hazard is not Clark's, and that is the point.** Both are growth
curves, so the obvious assumption is that the zero-age NaN-gradient trap carries
over. It does not: ``t = d * dev_grain_months / 12`` with ``d >= 1``, so every
age is strictly positive and ``model.stan``'s ``growth_curve`` is written without
the ``if (x <= 0)`` guard Clark's carries. What the ports guard instead is
``_shared.check_data``, because Stan's ``vector<lower=0>[len_data] t`` is looser
than the invariant the missing branch relies on.

**The zero that does bite here is ``ulr``.** The accident-year effect is additive
on the ulr scale and unbounded::

    ulr[w] = ulr_pop + sd_ulr * z_ulr[w]

so a draw can push it non-positive, and Stan then evaluates ``log()`` of a
non-positive number, raises a domain error and REJECTS the proposal. That
rejection is part of the specification - it is what makes every retained draw
safe for ``predict()`` and the held-out scorer - so a port has to reproduce it,
as a log-density of ``-inf`` rather than as a NaN.

Getting there needs a safe dummy substituted INSIDE the ``log``, and the tests
below pin both what that buys and what it does not - because the mechanism is
NOT Clark's, and the first version of the negative control here got it wrong.

Clark's trap is ``theta/0 -> inf`` in an unselected ``where`` branch, where
``0 * inf`` poisons the reverse pass, so masking the result loses the GRADIENT.
Here the offending operation is ``log`` of a negative number, whose derivative
``1/x`` is finite - so masking only the result keeps the gradient, and a control
built that way passes on the broken model. What actually breaks is the DENSITY:
``mu`` feeds ``LogNormal(mu, sigma).log_prob(y)``, so one NaN cell makes the
observed site NaN, and ``NaN + (-inf) = NaN``. Measured on the unsubstituted
model at a fully-negative ``ulr``: density NaN and all six gradients NaN, against
``-inf`` and six finite gradients for the shipped one. NaN is strictly worse -
``-inf`` is a rejection NUTS understands.

The strongest test here is neither of those: ``test_ports_match_the_stan_target``
reimplements ``model.stan``'s whole target in numpy and asserts each port
differs from it by a CONSTANT across many parameter points. Every prior's
location and scale, the non-centered effect and the likelihood are pinned at
once - a mis-ported ``normal(2, 1)`` under ``<lower=0>`` read as a half-normal
about 0, say, moves the offset rather than breaking a shape assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

#: what the numpyro-vs-pymc gate in this file compares. The published
#: three-backend run uses ``kernels.parity.GUSZCZA_PARITY_VARS``.
GUSZCZA_VARS = ("ulr_pop", "sd_ulr", "ulr", "omega", "theta", "sigma")

#: parameters at the priors' own centres, and ages on the entry's YEARS scale.
OMEGA, THETA = 2.0, 4.0
ULR_POP, SD_ULR, SIGMA = 0.65, 0.05, 0.08
AGES = np.array([1.0, 2.0, 5.0, 10.0])

CURVES = [("loglogistic", 1), ("weibull", 2)]

#: the two sites this entry's NumPyro port spells as ImproperUniform + factor
#: (Stan's own half-Student-t construction), so they cannot be forward-sampled
#: and a trace has to substitute them.
IMPROPER = {"sd_ulr": np.array(SD_ULR), "sigma": np.array(SIGMA)}


def simulate_guszcza_stan_data(n_w: int = 8, seed: int = 0, curve: int = 1, sigma: float = 0.08):
    """A Guszcza-generated Stan data block, self-contained (no mart, no Triangle).

    Cumulative paid loss RATIOS are lognormal about ``ulr[w] * G(t)`` - drawn
    from the model being fitted, so it is correctly specified and any parity
    failure can only be an implementation bug. Ages follow the entry's own
    convention: ``t = d`` years on an annual grain, so ``t >= 1`` and the
    zero-age case this curve does not handle never arises.
    """
    rng = np.random.default_rng(seed)
    ulr = ULR_POP + SD_ULR * rng.normal(size=n_w)
    w, t, y = [], [], []
    for wi in range(1, n_w + 1):
        for d in range(1, n_w - wi + 2):  # upper triangle
            age = float(d)
            g = (
                1.0 / (1.0 + (THETA / age) ** OMEGA)
                if curve == 1
                else 1.0 - np.exp(-((age / THETA) ** OMEGA))
            )
            w.append(wi)
            t.append(age)
            y.append(float(ulr[wi - 1] * g * np.exp(rng.normal(0.0, sigma))))
    return {
        "len_data": len(w),
        "n_w": n_w,
        "w": np.array(w, dtype=int),
        "t": np.array(t, dtype=float),
        "y": np.array(y, dtype=float),
        "curve": int(curve),
    }


def stan_target(params: dict, data: dict) -> float:
    """``model.stan``'s target in plain numpy, term for term, in CONSTRAINED space.

    The reference the ports are checked against. Written from the Stan file
    rather than from either port, and using fully normalized densities - each
    port drops or adds its own constants (Stan's ``~`` drops them, NumPyro's
    ``TruncatedNormal`` adds a truncation term, PyMC's ``HalfStudentT`` adds a
    ``log 2``), which is why the assertion is that the difference is constant
    rather than zero.

    No Jacobians: this is the density at given constrained values, which is what
    ``numpyro``'s ``log_density`` and PyMC's ``compile_logp(jacobian=False)``
    both return.
    """
    from scipy import stats

    ulr_pop = float(params["ulr_pop"])
    omega, theta = float(params["omega"]), float(params["theta"])
    sd_ulr, sigma = float(params["sd_ulr"]), float(params["sigma"])
    z_ulr = np.asarray(params["z_ulr"], dtype=float)

    lp = stats.lognorm.logpdf(ulr_pop, s=np.log(2.0), scale=0.6)  # lognormal(log .6, log 2)
    lp += stats.norm.logpdf(omega, 2.0, 1.0)  # normal(2, 1) under <lower=0>
    lp += stats.norm.logpdf(theta, 4.0, 1.0)  # normal(4, 1) under <lower=0>
    lp += stats.t.logpdf(sd_ulr, 3.0)  # student_t(3, 0, 1) under <lower=0>
    lp += stats.norm.logpdf(z_ulr, 0.0, 1.0).sum()  # std_normal()
    lp += stats.t.logpdf(sigma, 3.0)  # student_t(3, 0, 1) under <lower=0>

    ulr = ulr_pop + sd_ulr * z_ulr
    if np.any(ulr <= 0):
        # log() of a non-positive ulr: Stan raises a domain error and rejects
        return -np.inf
    t, y = np.asarray(data["t"], dtype=float), np.asarray(data["y"], dtype=float)
    g = (
        1.0 / (1.0 + (theta / t) ** omega)
        if int(data["curve"]) == 1
        else 1.0 - np.exp(-((t / theta) ** omega))
    )
    mu = np.log(ulr[np.asarray(data["w"], dtype=int) - 1] * g)
    return float(lp + stats.lognorm.logpdf(y, s=sigma, scale=np.exp(mu)).sum())


def param_points(n_w: int, *, include_rejected: bool = False) -> list[dict]:
    """Parameter points to compare the ports against :func:`stan_target` at.

    Deliberately spread over each prior's support rather than clustered at the
    centre, so a wrong location or scale moves the offset at some of them.
    """
    rng = np.random.default_rng(7)
    points = []
    for i in range(8):
        points.append(
            {
                "ulr_pop": 0.4 + 0.1 * i,
                "omega": 1.2 + 0.25 * i,
                "theta": 2.5 + 0.4 * i,
                "sd_ulr": 0.02 + 0.03 * i,
                "sigma": 0.05 + 0.02 * i,
                "z_ulr": rng.normal(size=n_w),
            }
        )
    if include_rejected:
        # sd_ulr * z_ulr large and negative: ulr goes below zero, Stan rejects
        points.append(
            {
                "ulr_pop": 0.65,
                "omega": 2.0,
                "theta": 4.0,
                "sd_ulr": 2.0,
                "sigma": 0.08,
                "z_ulr": np.full(n_w, -1.0),
            }
        )
    return points


def _numpyro_lp(params: dict, data: dict) -> float:
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    from ibnr.gallery.bayesian.guszcza_growth_curve import model_numpyro

    jp = {k: jnp.asarray(v, dtype=float) for k, v in params.items()}
    lp, _ = log_density(model_numpyro.guszcza_model, (data,), {}, jp)
    return float(lp)


def _pymc_lp(params: dict, data: dict) -> float:
    from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx
    from ibnr.gallery.bayesian.guszcza_growth_curve import model_pymc

    ensure_pytensor_cxx()
    model = model_pymc.build_model(data)
    # PyMC evaluates logp on VALUE vars, which live in the transformed space
    # (measured names here: ulr_pop_log__, omega_interval__, theta_interval__,
    # sd_ulr_log__, sigma_log__, z_ulr). `forward` needs the RV's own inputs -
    # an interval transform reads its bounds from them - so passing the value
    # alone raises IndexError rather than transforming wrongly.
    point = {}
    for rv in model.free_RVs:
        value_var = model.rvs_to_values[rv]
        raw = np.asarray(params[rv.name], dtype=float)
        transform = model.rvs_to_transforms.get(rv)
        point[value_var.name] = (
            np.asarray(transform.forward(raw, *rv.owner.inputs).eval())
            if transform is not None
            else raw
        )
    # jacobian=False: the density AT the constrained values, which is what
    # stan_target computes and what numpyro's log_density returns
    return float(model.compile_logp(jacobian=False)(point))


# -- fast: the growth curve --------------------------------------------------


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_growth_curve_matches_stans_closed_form(name, curve):
    """Values against the literal formulas in ``model.stan``, plus the shape a
    growth curve must have. No G(0) case here - unlike Clark's curve, this one
    is never evaluated at zero and does not define it."""
    pytest.importorskip("numpyro")
    from ibnr.gallery.bayesian.guszcza_growth_curve.model_numpyro import growth_curve

    got = np.asarray(growth_curve(AGES, OMEGA, THETA, curve))
    ref = (
        1.0 / (1.0 + (THETA / AGES) ** OMEGA)
        if curve == 1
        else 1.0 - np.exp(-((AGES / THETA) ** OMEGA))
    )
    np.testing.assert_allclose(got, ref, rtol=1e-6)
    assert np.all(np.diff(got) > 0), "G must be increasing in age"
    assert np.all((got > 0) & (got < 1)), "G is a fraction of ultimate"


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_ports_agree_on_the_growth_curve(name, curve):
    """The two hand-written curves must agree elementwise - an
    implementation-vs-implementation check no single-port test can make."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    import pytensor.tensor as pt

    from ibnr.gallery.bayesian.guszcza_growth_curve.model_numpyro import (
        growth_curve as g_numpyro,
    )
    from ibnr.gallery.bayesian.guszcza_growth_curve.model_pymc import growth_curve as g_pymc

    npy = np.asarray(g_numpyro(AGES, OMEGA, THETA, curve))
    pmc = np.asarray(g_pymc(pt.as_tensor(AGES), OMEGA, THETA, curve).eval())
    np.testing.assert_allclose(npy, pmc, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_ports_agree_with_the_shared_numpy_curve(name, curve):
    """``model.stan`` claims shared algebra with
    ``gallery/statistical/clark/model.py::growth`` - the function this entry's
    own ``predict()`` uses - so the claim is pinned rather than trusted. Ages are
    strictly positive, which is the region where the two agree; Clark's carries a
    zero-age branch this one deliberately lacks."""
    pytest.importorskip("numpyro")
    from ibnr.gallery.bayesian.guszcza_growth_curve.model_numpyro import growth_curve
    from ibnr.gallery.statistical.clark.model import growth

    assert np.all(AGES > 0), "the shared region is t > 0"
    # rtol at float32's resolution: JAX runs in single precision by default
    # (the repo does not enable jax_enable_x64), so the port's curve agrees with
    # the float64 numpy one to ~1e-7 and no closer. Measured max relative
    # difference on the weibull branch: 1.3e-7.
    np.testing.assert_allclose(
        np.asarray(growth_curve(AGES, OMEGA, THETA, curve)),
        growth(AGES, OMEGA, THETA, name),
        rtol=1e-6,
    )


def test_growth_curve_rejects_an_unknown_code():
    """The curve is data, so an out-of-range code must fail loudly rather than
    silently fitting the other curve."""
    pytest.importorskip("numpyro")
    from ibnr.gallery.bayesian.guszcza_growth_curve.model_numpyro import growth_curve

    with pytest.raises(ValueError, match="curve must be"):
        growth_curve(AGES, OMEGA, THETA, 3)


# -- fast: the data guard that stands in for Stan's missing zero-age branch ---


def test_check_data_refuses_a_zero_age():
    """``t = 0`` gives the loglogistic a finite VALUE and a NaN gradient, so it
    has to be refused by name; Stan's ``<lower=0>`` declaration would allow it."""
    from ibnr.gallery.bayesian.guszcza_growth_curve._shared import check_data

    with pytest.raises(ValueError, match="non-positive"):
        check_data(np.array([1.0, 0.0]), np.array([0.5, 0.6]))


def test_check_data_refuses_a_non_positive_loss_ratio():
    from ibnr.gallery.bayesian.guszcza_growth_curve._shared import check_data

    with pytest.raises(ValueError, match="lognormal"):
        check_data(np.array([1.0, 2.0]), np.array([0.5, 0.0]))


def test_the_zero_age_that_would_nan_the_gradient_is_real():
    """The reason ``check_data`` exists, demonstrated rather than asserted: at
    ``t = 0`` the loglogistic's value is finite and its derivatives are NaN. This
    is what the Clark entry's double-``where`` exists to prevent, and what this
    entry avoids by construction instead."""
    pytest.importorskip("numpyro")
    import jax

    from ibnr.gallery.bayesian.guszcza_growth_curve.model_numpyro import growth_curve

    at_zero = np.array([0.0])
    # theta/0 is the whole point, so the divide-by-zero warning is the expected
    # observation here rather than a problem to fix
    with np.errstate(divide="ignore"):
        value = float(np.asarray(growth_curve(at_zero, OMEGA, THETA, 1))[0])
        d_theta = jax.grad(lambda th: growth_curve(at_zero, OMEGA, th, 1).sum())(THETA)
    assert np.isfinite(value), (
        "the VALUE survives at t = 0, which is what makes a value-only test useless"
    )
    assert not np.isfinite(float(d_theta)), (
        "if this ever becomes finite, the loglogistic no longer needs the age guard"
    )


# -- fast: THE hazard - the gradient in the ulr rejection region --------------


def test_gradient_survives_the_ulr_rejection_region():
    """THE test for this entry. Where ``ulr`` goes non-positive the density must
    be ``-inf`` (Stan rejects the proposal) and every gradient must stay finite:
    NUTS consumes the gradient, and a NaN there kills the chain with no message."""
    pytest.importorskip("numpyro")
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    from ibnr.gallery.bayesian.guszcza_growth_curve import model_numpyro

    data = simulate_guszcza_stan_data(seed=2)
    rejected = param_points(data["n_w"], include_rejected=True)[-1]
    ulr = rejected["ulr_pop"] + rejected["sd_ulr"] * rejected["z_ulr"]
    assert np.all(ulr < 0), "this point must actually be in the rejection region"

    jp = {k: jnp.asarray(v, dtype=float) for k, v in rejected.items()}

    def lp(p):
        out, _ = log_density(model_numpyro.guszcza_model, (data,), {}, p)
        return out

    assert float(lp(jp)) == -np.inf, "Stan rejects here, so the port's density must be zero"
    grads = jax.grad(lp)(jp)
    for name, g in grads.items():
        assert np.all(np.isfinite(np.asarray(g))), (
            f"d/d{name} is not finite in the rejection region"
        )


def _unsubstituted_model(data: dict):
    """``guszcza_model`` with the safe substitution REMOVED, and nothing else
    changed - the negative control for the test above.

    Kept in the shape the real model has, with ``mu`` reaching the observed
    ``LogNormal``, because that is where the damage happens and a smaller
    stand-in does not reproduce it (see
    :func:`test_masking_only_the_log_result_is_not_enough`).
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from ibnr.gallery.bayesian.guszcza_growth_curve.model_numpyro import (
        _half_student_t,
        growth_curve,
    )

    w0 = np.asarray(data["w"], dtype=int) - 1
    t = jnp.asarray(np.asarray(data["t"], dtype=float))
    y = jnp.asarray(np.asarray(data["y"], dtype=float))
    n_w, curve = int(data["n_w"]), int(data["curve"])

    ulr_pop = numpyro.sample("ulr_pop", dist.LogNormal(np.log(0.6), np.log(2.0)))
    omega = numpyro.sample("omega", dist.TruncatedNormal(2.0, 1.0, low=0.0))
    theta = numpyro.sample("theta", dist.TruncatedNormal(4.0, 1.0, low=0.0))
    sd_ulr = _half_student_t("sd_ulr", 3.0, 1.0)
    z_ulr = numpyro.sample("z_ulr", dist.Normal(0.0, 1.0).expand([n_w]).to_event(1))
    sigma = _half_student_t("sigma", 3.0, 1.0)
    ulr = numpyro.deterministic("ulr", ulr_pop + sd_ulr * z_ulr)

    mu = jnp.log(ulr[w0] * growth_curve(t, omega, theta, curve))  # <- no substitution
    numpyro.factor("ulr_support", jnp.where(jnp.all(ulr > 0), 0.0, -jnp.inf))
    numpyro.sample("y", dist.LogNormal(mu, sigma), obs=y)


def test_the_unsubstituted_model_loses_both_density_and_gradient():
    """The negative control that makes the gradient test above meaningful.

    Drop the safe substitution and the density in the rejection region becomes
    **NaN instead of -inf**, because the NaN ``mu`` reaches
    ``LogNormal.log_prob`` and ``NaN + (-inf) = NaN``; every gradient goes with
    it. That is what the shipped model's substitution buys, and NaN is strictly
    worse than ``-inf``: ``-inf`` is a rejection NUTS understands.
    """
    pytest.importorskip("numpyro")
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    from ibnr.gallery.bayesian.guszcza_growth_curve import model_numpyro

    data = simulate_guszcza_stan_data(seed=2)
    rejected = param_points(data["n_w"], include_rejected=True)[-1]
    jp = {k: jnp.asarray(v, dtype=float) for k, v in rejected.items()}

    def lp_of(model):
        def f(p):
            out, _ = log_density(model, (data,), {}, p)
            return out

        return f

    shipped, naive = lp_of(model_numpyro.guszcza_model), lp_of(_unsubstituted_model)

    assert float(shipped(jp)) == -np.inf, "the shipped model rejects, as Stan does"
    assert np.isnan(float(naive(jp))), "the unsubstituted model is supposed to go NaN"

    shipped_g = jax.grad(shipped)(jp)
    naive_g = jax.grad(naive)(jp)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in shipped_g.values())
    assert not any(np.all(np.isfinite(np.asarray(v))) for v in naive_g.values()), (
        "every gradient of the unsubstituted model should be NaN here"
    )


def test_masking_only_the_log_result_is_not_enough():
    """Why the substitution goes INSIDE the ``log``, and why this entry's hazard
    is not ``clark_growth_curve``'s.

    Clark's trap is ``theta/0 -> inf`` in an unselected ``where`` branch, where
    ``0 * inf`` poisons the reverse pass, so masking the result is not enough to
    save the GRADIENT. Here the offending operation is ``log`` of a negative
    number, whose derivative ``1/x`` is perfectly finite - so masking the result
    DOES keep the gradient finite, and the first version of the control above
    passed on the broken model because of it. The substitution is load-bearing
    for the DENSITY (see the test above), not for the gradient of this
    expression, and the two must not be conflated.
    """
    pytest.importorskip("numpyro")
    import jax
    import jax.numpy as jnp

    from ibnr.gallery.bayesian.guszcza_growth_curve.model_numpyro import growth_curve

    data = simulate_guszcza_stan_data(seed=2)
    w0 = data["w"] - 1
    z_bad = jnp.full(data["n_w"], -1.0)

    def masked_result(ulr_pop, sd_ulr):
        ulr = ulr_pop + sd_ulr * z_bad
        mu = jnp.log(ulr[w0] * growth_curve(data["t"], OMEGA, THETA, 1))
        return jnp.where(jnp.all(ulr > 0), mu.sum(), -jnp.inf)

    args = (jnp.array(0.65), jnp.array(2.0))
    assert float(masked_result(*args)) == -np.inf
    assert np.isfinite(float(jax.grad(masked_result, argnums=0)(*args))), (
        "d/dx log(x) = 1/x is finite for x < 0, so this spelling keeps the gradient - "
        "if it ever stops doing so, this entry's hazard has become Clark's"
    )


def test_pymc_gradient_survives_the_ulr_rejection_region():
    """Same guard on the PyTensor side - ``pt.switch`` has the identical
    both-branches gradient behaviour as ``jnp.where``."""
    pytest.importorskip("pymc")

    from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx
    from ibnr.gallery.bayesian.guszcza_growth_curve import model_pymc

    ensure_pytensor_cxx()
    data = simulate_guszcza_stan_data(seed=2)
    model = model_pymc.build_model(data)
    point = model.initial_point()
    # push z_ulr hard negative in the transformed space; sd_ulr is log-transformed
    point["z_ulr"] = np.full(data["n_w"], -1.0)
    point["sd_ulr_log__"] = np.log(2.0)
    assert float(model.compile_logp()(point)) == -np.inf, "must be the rejection region"
    dlogp = np.asarray(model.compile_dlogp()(point))
    assert np.all(np.isfinite(dlogp)), (
        f"PyMC gradient is not finite in the rejection region: {dlogp}"
    )


# -- fast: both ports reproduce model.stan's whole target --------------------


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_ports_match_the_stan_target(name, curve):
    """The strongest check here: each port's log-density must differ from a
    literal numpy transcription of ``model.stan``'s target by a CONSTANT across
    the whole parameter grid.

    A constant, not zero, because each backend keeps its own normalizing terms
    (Stan's ``~`` drops them; ``TruncatedNormal`` adds a truncation constant;
    ``HalfStudentT`` adds a ``log 2``). Those cannot depend on the parameters, so
    a spread in the offset is a transcription error - a wrong prior location or
    scale, a missing ``sd_ulr`` multiplication, the wrong curve. The exact
    offsets are checked separately, by
    :func:`test_the_port_offsets_are_the_constants_we_can_predict`.

    Tolerances are per backend and RELATIVE to the size of the density, which is
    the same rule ``gallery/statistical/clark`` learned the hard way: an absolute
    tolerance on a likelihood is a lottery, because the magnitude belongs to the
    data. These densities run to ~2000 on this fixture, and JAX is
    single-precision by default (the repo does not enable ``jax_enable_x64``), so
    the NumPyro floor is ~2000 * 1.2e-7 ~ 2e-4 of pure rounding; PyMC is float64
    throughout. A real transcription error moves an offset by O(1) or more, so
    both gates stay orders of magnitude tighter than they need to be.
    """
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")

    data = simulate_guszcza_stan_data(n_w=7, seed=5, curve=curve)
    points = param_points(data["n_w"])
    ref = np.array([stan_target(p, data) for p in points])
    assert np.all(np.isfinite(ref)), "the reference grid must avoid the rejection region"
    scale = max(1.0, float(np.abs(ref).max()))

    for backend, fn, rtol in (("numpyro", _numpyro_lp, 1e-6), ("pymc", _pymc_lp, 1e-10)):
        got = np.array([fn(p, data) for p in points])
        offset = got - ref
        assert np.all(np.isfinite(got)), f"{backend} produced a non-finite density"
        assert offset.std() < rtol * scale, (
            f"{name}/{backend} does not track model.stan's target: offsets vary by "
            f"{offset.std():.3g} against a tolerance of {rtol * scale:.3g} "
            f"(min {offset.min():.6f}, max {offset.max():.6f})"
        )


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_the_port_offsets_are_the_constants_we_can_predict(name, curve):
    """The offsets above are not just constant, they are the constants the two
    constructions imply - so they are asserted rather than accepted.

    NumPyro: two ``TruncatedNormal`` normalizers, ``-log P(N(2,1) > 0)`` and
    ``-log P(N(4,1) > 0)``, against a reference that uses the untruncated
    normal. PyMC: the same two, plus ``log 2`` for each of the two
    ``HalfStudentT`` scales, which the reference writes as a full Student-t.

    This is what tells the difference between "both ports agree with the
    reference up to a constant" and "both ports normalize the way I think they
    do" - a port could pass the constancy check while quietly fitting a
    half-normal about zero in place of ``normal(2, 1)`` truncated at zero, and
    the offset would then be wrong by a knowable amount.

    Tolerances are relative to the density's magnitude, for the reason given in
    the test above - the offsets themselves are ~0.023 and ~1.409 while the
    densities are ~2000, so a float32 backend cannot resolve them absolutely.
    """
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from scipy import stats

    data = simulate_guszcza_stan_data(n_w=7, seed=5, curve=curve)
    point = param_points(data["n_w"])[0]
    ref = stan_target(point, data)
    scale = max(1.0, abs(ref))

    truncation = -np.log(stats.norm.sf(-2.0)) - np.log(stats.norm.sf(-4.0))
    expected = {"numpyro": truncation, "pymc": truncation + 2.0 * np.log(2.0)}
    for backend, fn, rtol in (("numpyro", _numpyro_lp, 1e-6), ("pymc", _pymc_lp, 1e-10)):
        got = fn(point, data) - ref
        assert abs(got - expected[backend]) < rtol * scale, (
            f"{name}/{backend} offset is {got:.8f}, expected {expected[backend]:.8f} "
            f"(tolerance {rtol * scale:.3g})"
        )


def test_the_stan_target_reference_is_sensitive_to_each_prior():
    """The reference itself must be able to fail. If a prior's location were
    dropped from :func:`stan_target`, the offsets above would still be constant
    for a port that dropped it too - so check that moving each parameter really
    moves the reference."""
    data = simulate_guszcza_stan_data(n_w=6, seed=4)
    base = param_points(data["n_w"])[0]
    lp0 = stan_target(base, data)
    for key in ("ulr_pop", "omega", "theta", "sd_ulr", "sigma"):
        moved = dict(base)
        moved[key] = float(base[key]) * 1.1
        assert stan_target(moved, data) != lp0, f"the reference ignores {key}"


def test_the_rejection_region_is_zero_density_in_both_ports():
    """``ulr <= 0`` must be refused identically by the reference and both ports -
    the boundary of the posterior's support, not a numerical accident."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")

    data = simulate_guszcza_stan_data(n_w=6, seed=6)
    rejected = param_points(data["n_w"], include_rejected=True)[-1]
    assert stan_target(rejected, data) == -np.inf
    assert _numpyro_lp(rejected, data) == -np.inf
    assert _pymc_lp(rejected, data) == -np.inf


# -- fast: the graphs are Stan's parameter block -----------------------------


def test_numpyro_graph_matches_stan():
    """Trace once with the improper sites substituted: Stan's sampled parameters,
    ``ulr`` exposed as a deterministic (the scorer reads it), one ``mu`` per cell
    implied by the observed site."""
    pytest.importorskip("numpyro")
    import jax
    from numpyro import handlers

    from ibnr.gallery.bayesian.guszcza_growth_curve import model_numpyro

    data = simulate_guszcza_stan_data(seed=3)
    tr = handlers.trace(
        handlers.seed(
            handlers.substitute(model_numpyro.guszcza_model, IMPROPER), jax.random.PRNGKey(0)
        )
    ).get_trace(data)

    assert {"ulr_pop", "omega", "theta", "sd_ulr", "z_ulr", "sigma"} <= set(tr)
    assert np.asarray(tr["z_ulr"]["value"]).shape == (data["n_w"],)
    assert np.asarray(tr["ulr"]["value"]).shape == (data["n_w"],)
    assert tr["ulr"]["type"] == "deterministic"
    assert float(np.asarray(tr["omega"]["value"])) > 0
    assert float(np.asarray(tr["theta"]["value"])) > 0
    assert np.asarray(tr["y"]["value"]).shape == (data["len_data"],)


def test_pymc_graph_matches_stan():
    """The free RVs are exactly Stan's sampled parameters - no more (a port that
    sampled ``ulr`` directly would have lost the non-centering) and no fewer."""
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.guszcza_growth_curve import model_pymc

    data = simulate_guszcza_stan_data(seed=3)
    model = model_pymc.build_model(data)
    assert {v.name for v in model.free_RVs} == {
        "ulr_pop",
        "omega",
        "theta",
        "sd_ulr",
        "z_ulr",
        "sigma",
    }
    assert "ulr" in model.named_vars, "the held-out scorer reads ulr out of posterior"


@pytest.mark.parametrize("port", ["model_numpyro", "model_pymc"])
def test_ports_expose_every_variable_the_scorer_reads(port):
    """``scorer.REQUIRED_DRAWS`` is what the held-out path reads out of
    ``posterior``. A port that fits but omits one of them would raise only when
    something tried to score it, which is far from here."""
    pytest.importorskip("numpyro" if port == "model_numpyro" else "pymc")
    import importlib

    from ibnr.gallery.bayesian.guszcza_growth_curve import scorer

    data = simulate_guszcza_stan_data(seed=3)
    mod = importlib.import_module(f"ibnr.gallery.bayesian.guszcza_growth_curve.{port}")
    if port == "model_pymc":
        from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx

        ensure_pytensor_cxx()
        exposed = set(mod.build_model(data).named_vars)
    else:
        import jax
        from numpyro import handlers

        exposed = set(
            handlers.trace(
                handlers.seed(
                    handlers.substitute(mod.guszcza_model, IMPROPER), jax.random.PRNGKey(0)
                )
            ).get_trace(data)
        )
    missing = [name for name in scorer.REQUIRED_DRAWS if name not in exposed]
    assert not missing, f"{port} does not expose {missing}"


# -- fast: the half-Student-t construction -----------------------------------


def test_numpyro_half_student_t_is_the_density_stan_writes():
    """``real<lower=0> x;`` plus ``x ~ student_t(3, 0, 1);`` is a half-t, and the
    port spells it as ImproperUniform + factor because
    ``TruncatedDistribution(StudentT)`` needs a CDF NumPyro cannot supply
    (CLAUDE.md). Check the density it actually adds, against scipy, up to the
    constant that separates the half-t from the full one."""
    pytest.importorskip("numpyro")
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from scipy import stats

    grid = np.array([0.05, 0.2, 0.8, 1.0, 3.0])
    got = np.asarray(dist.StudentT(3.0, 0.0, 1.0).log_prob(jnp.asarray(grid)))
    ref = stats.t.logpdf(grid, 3.0)
    np.testing.assert_allclose(got, ref, rtol=1e-6)
    # and the site is positively constrained, which is what makes it HALF - the
    # constraint carries the whole truncation, so a port that dropped it would
    # sample a full Student-t and put mass on negative scales. ImproperUniform
    # wraps the constraint in an IndependentConstraint, so unwrap before testing.
    site = dist.ImproperUniform(dist.constraints.positive, (), ())
    base = getattr(site.support, "base_constraint", site.support)
    assert isinstance(base, dist.constraints._Positive | dist.constraints._GreaterThan), (
        f"the half-t site must be positively constrained, got {base}"
    )


def test_pymc_half_student_t_matches_numpyros_construction():
    """PyMC has the distribution natively; the two ports must nonetheless agree
    on the density up to the ``log 2`` that normalizes the half."""
    pytest.importorskip("pymc")
    import pymc as pm
    from scipy import stats

    grid = np.array([0.05, 0.2, 0.8, 1.0, 3.0])
    with pm.Model():
        x = pm.HalfStudentT("x", nu=3, sigma=1.0)
        logp = np.asarray(pm.logp(x, grid).eval())
    np.testing.assert_allclose(logp, stats.t.logpdf(grid, 3.0) + np.log(2.0), rtol=1e-6)


# -- fast: the entry's backend surface ---------------------------------------


def test_entry_rejects_unknown_backend():
    from ibnr.gallery.bayesian.guszcza_growth_curve.model import GuszczaGrowthCurve

    with pytest.raises(ValueError, match="backend must be one of"):
        GuszczaGrowthCurve().fit(None, backend="jags")


def test_all_three_backends_are_registered():
    """The seam the card promised is actually open - and this is the assertion
    that would have caught the milestone-5 gap, where the entry sat at
    ``BACKENDS = ("stan",)`` while every sibling had three."""
    from ibnr.gallery.bayesian.guszcza_growth_curve.model import BACKENDS

    assert BACKENDS == ("stan", "numpyro", "pymc")


def test_ports_reject_stan_only_controls():
    """``parallel_chains`` is a cmdstan control; accepting it on a port would
    report an escalated fit that never ran."""
    from ibnr.gallery.bayesian.guszcza_growth_curve.model import GuszczaGrowthCurve

    with pytest.raises(ValueError, match="stan-backend control"):
        GuszczaGrowthCurve().fit(None, backend="numpyro", parallel_chains=4)


def test_non_pymc_backends_reject_nuts_sampler():
    from ibnr.gallery.bayesian.guszcza_growth_curve.model import GuszczaGrowthCurve

    with pytest.raises(ValueError, match="pymc-backend control"):
        GuszczaGrowthCurve().fit(None, backend="numpyro", nuts_sampler="numpyro")


def test_max_treedepth_is_shared_by_every_backend():
    """It is NOT a stan-only control: all three samplers have it, this entry's
    default of 15 differs from NumPyro's and PyMC's own default of 10, and
    parity depends on holding it constant. So the ports must accept it - the
    delivery of the value is checked by ``test_max_treedepth_reaches_the_ports``.
    """
    import inspect

    from ibnr.gallery.bayesian.guszcza_growth_curve import model
    from ibnr.gallery.bayesian.guszcza_growth_curve._shared import MAX_TREE_DEPTH

    assert MAX_TREE_DEPTH == 15
    for name in ("_sample_stan", "_sample_numpyro", "_sample_pymc"):
        sig = inspect.signature(getattr(model.GuszczaGrowthCurve, name))
        assert "max_treedepth" in sig.parameters, f"{name} does not take max_treedepth"
    # and asking for it on a port must NOT raise, unlike parallel_chains
    entry = model.GuszczaGrowthCurve()
    with pytest.raises(AttributeError):
        # fails inside as_of on the None triangle, i.e. it got PAST validation
        entry.fit(None, backend="numpyro", max_treedepth=12, as_of="1997-12-31")


# -- slow: the samplers ------------------------------------------------------


def test_max_treedepth_reaches_numpyros_kernel(monkeypatch):
    """A signature test proves a wire exists, not that it is connected (this
    repo's named inert-parameter bug class).

    ``arviz.from_numpyro`` gives this model only ``diverging`` in
    ``sample_stats`` - no tree depth, no step count - so unlike the PyMC test
    below there is no sampler statistic to read the cap back from. Spy on the
    NUTS constructor instead, which is the exact join the argument has to cross.
    """
    pytest.importorskip("numpyro")
    import numpyro.infer

    from ibnr.gallery.bayesian.guszcza_growth_curve import model_numpyro

    seen = {}
    real_nuts = numpyro.infer.NUTS

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop before sampling")

    monkeypatch.setattr(numpyro.infer, "NUTS", spy)
    try:
        with pytest.raises(RuntimeError, match="stop before sampling"):
            model_numpyro.sample(simulate_guszcza_stan_data(n_w=5, seed=8), max_treedepth=2)
    finally:
        monkeypatch.setattr(numpyro.infer, "NUTS", real_nuts)

    assert seen.get("max_tree_depth") == 2, (
        f"max_treedepth did not reach NUTS; it saw {seen.get('max_tree_depth')!r}"
    )
    assert seen.get("target_accept_prob") == pytest.approx(0.999), (
        "the entry's adapt_delta must reach NUTS too"
    )


@pytest.mark.slow
def test_max_treedepth_reaches_pymcs_sampler():
    """The behavioural half of the check above: PyMC does record ``tree_depth``,
    so cap the tree at 2 and read it back. An inert argument leaves the sampler
    at its own default of 10.

    Marked slow because it samples. Runs through PyMC's native NUTS deliberately
    - that is the path ``max_treedepth`` is forwarded to as a step-method kwarg,
    and the foreign-sampler path takes a different keyword.
    """
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.guszcza_growth_curve import model_pymc

    idata = model_pymc.sample(
        simulate_guszcza_stan_data(n_w=5, seed=8),
        chains=1,
        iter_warmup=80,
        iter_sampling=80,
        seed=3,
        max_treedepth=2,
    )
    observed = int(np.asarray(idata.sample_stats["tree_depth"]).max())
    assert observed <= 2, f"tree_depth reached {observed}, so max_treedepth=2 was ignored"


@pytest.mark.parity
@pytest.mark.slow
@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_numpyro_pymc_parity(name, curve):
    """The parity gate, run on BOTH curves: they share every line except the
    curve itself, so a port that transcribed one and not the other is caught
    here. The three-way run against Stan is ``scripts/parity_gallery.py``.

    PyMC runs through ``nuts_sampler="numpyro"``: at this entry's
    ``adapt_delta = 0.999`` the native PyTensor sampler takes ~130x as long
    (measured 860 s against 6.7 s for 2 x 600 draws), and it is the GRAPH being
    compared here, not the NUTS implementation. Same choice, same reason, as
    ``compartmental``.
    """
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.guszcza_growth_curve import model_numpyro, model_pymc
    from ibnr.kernels.parity import compare_posteriors

    data = simulate_guszcza_stan_data(n_w=8, seed=7, curve=curve)
    idn = model_numpyro.sample(data, chains=2, iter_warmup=1000, iter_sampling=2500, seed=11)
    idp = model_pymc.sample(
        data,
        chains=2,
        iter_warmup=1000,
        iter_sampling=2500,
        seed=11,
        nuts_sampler="numpyro",
    )

    report = compare_posteriors(
        {"numpyro": idn, "pymc": idp}, reference="numpyro", var_names=GUSZCZA_VARS
    )
    assert report.passed, f"{name} parity failed:\n{report.failures()}"
