"""Cross-backend parity for clark_growth_curve (milestone 5).

Same two tiers as the other parity files (fast structural checks in the default
suite; a sampling gate under ``parity``/``slow``). The three-way run with Stan
as ground truth is ``scripts/parity_gallery.py --model clark_growth_curve``.

**The hazard this file exists for is the growth curve at age zero.** Stan writes

    if (x <= 0) return 0;

and that is not an edge case: ``age_lo`` is exactly 0 for every origin's first
development cell, by construction (the mid-period age shift clamps at 0). The
obvious port - ``where(x > 0, formula, 0.0)`` - produces the RIGHT VALUE and a
**NaN gradient**, because both PPLs evaluate both branches of a where/switch and
the NaN from ``theta/0`` in the unselected branch propagates through the reverse
pass. On the loglogistic curve, which is the default, that is exactly what
happens (measured: `dG/dtheta` and `dG/domega` both NaN). NUTS then fails with
no useful message.

So the tests below check the GRADIENT, not just the value - a value-only test
passes cleanly on the broken implementation and is worse than no test at all.
"""

from __future__ import annotations

import numpy as np
import pytest

CLARK_VARS = ("logelr", "omega", "theta")

#: (omega, theta) at the prior medians, and ages including the load-bearing 0.
OMEGA, THETA = 1.5, 48.0
AGES = np.array([0.0, 6.0, 18.0, 42.0])

CURVES = [("loglogistic", 1), ("weibull", 2)]


def simulate_clark_stan_data(n_w: int = 6, seed: int = 0, curve: int = 1, phi: float = 50.0):
    """A Clark-generated Stan data block, self-contained (no mart, no Triangle).

    Increments are od-Poisson about the Cape Cod growth-curve mean, so the
    fitted model is correctly specified and any parity failure is an
    implementation bug. Ages follow the entry's own convention: cell d spans
    ``(step(d-1) - step/2, step*d - step/2]`` clamped at 0, which is what makes
    ``age_lo[0] == 0`` for every origin.
    """
    from ibnr.gallery.bayesian.clark_growth_curve.model_numpyro import growth_curve

    rng = np.random.default_rng(seed)
    step, n_d = 12.0, n_w
    prem = rng.uniform(8000, 20000, n_w)
    logelr = -0.5

    w, age_lo, age_hi = [], [], []
    for wi in range(1, n_w + 1):
        for d in range(1, n_d - wi + 2):  # upper triangle
            w.append(wi)
            age_lo.append(max(step * (d - 1) - step / 2, 0.0))
            age_hi.append(step * d - step / 2)
    w = np.array(w, dtype=int)
    age_lo = np.array(age_lo)
    age_hi = np.array(age_hi)

    emerged = np.asarray(growth_curve(age_hi, OMEGA, THETA, curve)) - np.asarray(
        growth_curve(age_lo, OMEGA, THETA, curve)
    )
    mean_inc = np.exp(np.log(prem)[w - 1] + logelr) * emerged
    inc = rng.poisson(np.maximum(mean_inc, 1e-9) / phi) * phi
    return {
        "len_data": len(w),
        "n_w": n_w,
        "w": w,
        "age_lo": age_lo,
        "age_hi": age_hi,
        "inc_loss": inc.astype(float),
        "logprem_w": np.log(prem),
        "phi": float(phi),
        "curve": int(curve),
        "theta_prior_median": 4.0 * step,
    }


# -- fast: the zero-age branch must not poison the gradient ------------------


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_growth_curve_gradient_is_finite_at_age_zero(name, curve):
    """THE test for this entry. ``age_lo = 0`` occurs in every triangle, and a
    naive ``where`` returns the correct value with a NaN gradient - so this
    checks the derivative, which is what the sampler actually consumes."""
    pytest.importorskip("numpyro")
    import jax

    from ibnr.gallery.bayesian.clark_growth_curve.model_numpyro import growth_curve

    assert AGES[0] == 0.0, "the zero age is the whole point of this test"
    d_theta = jax.grad(lambda t: growth_curve(AGES, OMEGA, t, curve).sum())(THETA)
    d_omega = jax.grad(lambda o: growth_curve(AGES, o, THETA, curve).sum())(OMEGA)
    assert np.isfinite(float(d_theta)), f"{name}: dG/dtheta is not finite"
    assert np.isfinite(float(d_omega)), f"{name}: dG/domega is not finite"
    # non-degenerate: the curve genuinely responds to both parameters
    assert abs(float(d_theta)) > 0 and abs(float(d_omega)) > 0


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_pymc_growth_curve_gradient_is_finite_at_age_zero(name, curve):
    """Same guard on the PyTensor side - ``pt.switch`` has the identical
    both-branches gradient behaviour as ``jnp.where``."""
    pytest.importorskip("pymc")
    import pytensor
    import pytensor.tensor as pt

    from ibnr.gallery.bayesian._toolchain import ensure_pytensor_cxx
    from ibnr.gallery.bayesian.clark_growth_curve.model_pymc import growth_curve

    ensure_pytensor_cxx()
    om, th = pt.scalar("om"), pt.scalar("th")
    out = growth_curve(pt.as_tensor(AGES), om, th, curve).sum()
    d_theta = pytensor.function([om, th], pt.grad(out, th))(OMEGA, THETA)
    d_omega = pytensor.function([om, th], pt.grad(out, om))(OMEGA, THETA)
    assert np.isfinite(float(d_theta)), f"{name}: dG/dtheta is not finite"
    assert np.isfinite(float(d_omega)), f"{name}: dG/domega is not finite"


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_growth_curve_matches_stans_closed_form(name, curve):
    """Values against the literal formulas in ``model.stan``, including
    G(0) = 0 and the monotone, bounded shape Clark's curves must have."""
    pytest.importorskip("numpyro")
    from ibnr.gallery.bayesian.clark_growth_curve.model_numpyro import growth_curve

    got = np.asarray(growth_curve(AGES, OMEGA, THETA, curve))
    x = AGES[1:]
    ref = (
        1.0 / (1.0 + (THETA / x) ** OMEGA)
        if curve == 1
        else 1.0 - np.exp(-((x / THETA) ** OMEGA))
    )
    assert got[0] == 0.0, "G(0) must be exactly 0"
    np.testing.assert_allclose(got[1:], ref, rtol=1e-5)
    assert np.all(np.diff(got) > 0), "G must be increasing in age"
    assert np.all((got >= 0) & (got < 1)), "G is a fraction of ultimate"


@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_ports_agree_on_the_growth_curve(name, curve):
    """The two hand-written curves must agree elementwise - an
    implementation-vs-implementation check no single-port test can make."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    import pytensor.tensor as pt

    from ibnr.gallery.bayesian.clark_growth_curve.model_numpyro import (
        growth_curve as g_numpyro,
    )
    from ibnr.gallery.bayesian.clark_growth_curve.model_pymc import growth_curve as g_pymc

    npy = np.asarray(g_numpyro(AGES, OMEGA, THETA, curve))
    pmc = np.asarray(g_pymc(pt.as_tensor(AGES), OMEGA, THETA, curve).eval())
    np.testing.assert_allclose(npy, pmc, rtol=1e-5, atol=1e-6)


def test_growth_curve_rejects_an_unknown_code():
    """The curve is data, so an out-of-range code must fail loudly rather than
    silently fitting the other curve."""
    pytest.importorskip("numpyro")
    from ibnr.gallery.bayesian.clark_growth_curve.model_numpyro import growth_curve

    with pytest.raises(ValueError, match="curve must be"):
        growth_curve(AGES, OMEGA, THETA, 3)


# -- fast: graphs build with Stan's parameterization -------------------------


def test_numpyro_model_shapes():
    """Trace once (no sampling): three scalar parameters, positive-constrained
    shape and scale, and one ``mu`` per observed cell."""
    pytest.importorskip("numpyro")
    import jax
    from numpyro import handlers

    from ibnr.gallery.bayesian.clark_growth_curve import model_numpyro

    data = simulate_clark_stan_data(seed=2)
    tr = handlers.trace(handlers.seed(model_numpyro.clark_model, jax.random.PRNGKey(0))).get_trace(
        data
    )
    assert np.asarray(tr["mu"]["value"]).shape == (data["len_data"],)
    assert float(np.asarray(tr["omega"]["value"])) > 0
    assert float(np.asarray(tr["theta"]["value"])) > 0
    assert (np.asarray(tr["mu"]["value"]) >= 0).all(), "a Cape Cod mean cannot be negative"


def test_pymc_model_builds():
    """The free RVs are exactly Stan's three sampled parameters, with ``mu``
    deterministic."""
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.clark_growth_curve import model_pymc

    data = simulate_clark_stan_data(seed=3)
    model = model_pymc.build_model(data)
    assert {v.name for v in model.free_RVs} == {"logelr", "omega", "theta"}
    assert "mu" in model.named_vars


def test_entry_rejects_unknown_backend():
    from ibnr.gallery.bayesian.clark_growth_curve.model import ClarkGrowthCurve

    with pytest.raises(ValueError, match="backend must be one of"):
        ClarkGrowthCurve().fit(None, backend="jags")


def test_ports_reject_stan_only_controls():
    from ibnr.gallery.bayesian.clark_growth_curve.model import ClarkGrowthCurve

    with pytest.raises(ValueError, match="stan-backend controls"):
        ClarkGrowthCurve().fit(None, backend="numpyro", max_treedepth=12)


# -- slow: NumPyro and PyMC sample the same posterior ------------------------


@pytest.mark.parity
@pytest.mark.slow
@pytest.mark.parametrize(("name", "curve"), CURVES)
def test_numpyro_pymc_parity(name, curve):
    """The parity gate, run on BOTH curves: the loglogistic branch is the one
    the gradient trap affects, and the Weibull branch shares the machinery, so
    a port that fixed only one would be caught here."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.clark_growth_curve import model_numpyro, model_pymc
    from ibnr.kernels.parity import compare_posteriors

    data = simulate_clark_stan_data(n_w=8, seed=7, curve=curve)
    idn = model_numpyro.sample(data, chains=2, iter_warmup=1000, iter_sampling=2500, seed=11)
    idp = model_pymc.sample(data, chains=2, iter_warmup=1000, iter_sampling=2500, seed=11)

    report = compare_posteriors(
        {"numpyro": idn, "pymc": idp}, reference="numpyro", var_names=CLARK_VARS
    )
    assert report.passed, f"{name} parity failed:\n{report.failures()}"
