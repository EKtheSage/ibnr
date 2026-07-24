"""Cross-backend parity for england_verrall_odp (milestone 5).

Same two tiers as the Meyers parity files:

* **Fast structural tests** (default suite): both graphs build with Stan's
  parameterization, and the hand-written quasi-likelihood is pinned against the
  closed form it is equal to.
* **Sampling parity** (``parity`` + ``slow``): both ports sample the same small
  synthetic posterior and agree within MCSE via ``kernels.parity``. Needs no
  cmdstan, so CI can gate parity without a Stan toolchain; the three-way run
  with Stan as ground truth is
  ``scripts/parity_gallery.py --model england_verrall_odp``.

What is distinctive here, and what these tests are really for: **ODP is the
first entry whose likelihood neither PPL provides.** Stan declares its own

    odp_lpdf(x | mu, phi) = (x/phi) log(mu/phi) - mu/phi - lgamma(x/phi + 1)

so each port hand-writes a density, and a hand-written density is exactly the
kind of thing that can be subtly wrong while still sampling happily. Two
independent checks pin it:

1. it equals ``Poisson(mu/phi).log_prob(x/phi)`` - the identity that makes the
   quasi-likelihood proper - to floating point;
2. the two ports agree with each other elementwise at matched parameters.

The parameter-free terms are deliberately kept. Dropping them would leave the
posterior untouched (``phi`` and ``x`` are data) but shift every ``log_lik`` by
a constant, so ELPD/LOO would silently stop matching Stan's
``generated quantities``. Stan does not drop them either: the constant-dropping
that ``~`` performs for BUILT-IN distributions does not apply to a user-defined
``_lpdf``.
"""

from __future__ import annotations

import numpy as np
import pytest

ODP_VARS = ("c", "alpha", "beta")
PHI = 50.0

#: Tolerance for comparing the NumPyro density against a float64 reference.
#: JAX runs in float32 (x64 deliberately not enabled, matching every other port)
#: and this density is a SMALL DIFFERENCE OF LARGE TERMS: at x = 12000, phi = 50
#: the first term is ~1358 and the gammaln ~1112, cancelling to ~-3.8. Float32
#: therefore leaves ~3e-5 absolute, which is ~1e-5 relative on the result and
#: many orders of magnitude below the MCSE the parity gate works in. Anything
#: materially larger than this would be a transcription error, not arithmetic.
DENSITY_RTOL, DENSITY_ATOL = 1e-4, 1e-4


def simulate_odp_contract(n_w: int = 7, n_d: int = 7, seed: int = 0, phi: float = PHI) -> dict:
    """An ODP-generated upper-triangle contract dict (the ``odp_stan_data``
    schema), self-contained so parity depends on neither the mart nor a Triangle.

    Increments are genuinely over-dispersed Poisson - ``phi * Poisson(mean/phi)``
    about a log-link mean - so they are non-negative by construction, which the
    incremental contract requires, and the fitted model is correctly specified.
    ``alpha[0] = 0`` and ``beta[0] = 0`` are E&V's pins (the FIRST beta, unlike
    the Meyers family's last).
    """
    rng = np.random.default_rng(seed)
    prem = rng.uniform(8000, 20000, n_w)
    c = -0.5
    alpha = np.concatenate([[0.0], rng.normal(0, 0.2, n_w - 1)])
    beta = np.concatenate([[0.0], np.sort(rng.uniform(-2.5, -0.2, n_d - 1))[::-1]])

    cells = [
        (w, d) for w in range(1, n_w + 1) for d in range(1, n_d + 1) if (w - 1) + (d - 1) < n_d
    ]
    cells.sort()
    w = np.array([x[0] for x in cells])
    d = np.array([x[1] for x in cells])
    logprem = np.log(prem)[w - 1]
    mean_inc = np.exp(logprem + c + alpha[w - 1] + beta[d - 1])
    inc = rng.poisson(mean_inc / phi) * float(phi)
    return {
        "len_data": len(cells),
        "n_w": n_w,
        "n_d": n_d,
        "w": w,
        "d": d,
        "inc_loss": inc.astype(float),
        "logprem": logprem,
        "phi": float(phi),
    }


# -- fast: the hand-written density is the one Stan declares -----------------


def test_numpyro_density_matches_the_poisson_identity():
    """``odp_lpdf(x | mu, phi)`` must equal ``Poisson(mu/phi).log_prob(x/phi)``.

    That identity is what makes an od-Poisson quasi-likelihood a proper density
    despite ``x/phi`` not being an integer, and it is the single most effective
    guard on a hand-transcribed log-density: any dropped or mistyped term moves
    the value by far more than the float32 floor documented at
    ``DENSITY_RTOL``.
    """
    pytest.importorskip("numpyro")
    import numpyro.distributions as dist

    from ibnr.gallery.bayesian.england_verrall_odp.model_numpyro import _odp_distribution

    odp = _odp_distribution()
    mu = np.array([1000.0, 2000.0, 12345.0])
    x = np.array([950.0, 2100.0, 12000.0])
    got = np.asarray(odp(mu, PHI).log_prob(x))
    ref = np.asarray(dist.Poisson(mu / PHI).log_prob(x / PHI))
    np.testing.assert_allclose(got, ref, rtol=DENSITY_RTOL, atol=DENSITY_ATOL)


def test_density_keeps_the_parameter_free_terms():
    """The ``-lgamma(x/phi + 1)`` and ``(x/phi) log(1/phi)`` terms are constants
    in the parameters, so dropping them would not move the posterior - but it
    WOULD shift every ``log_lik`` by a constant and silently break ELPD
    comparability with Stan's ``generated quantities``. Checked by evaluating
    the density against the literal formula rather than a proportional one."""
    pytest.importorskip("numpyro")
    from scipy.special import gammaln

    from ibnr.gallery.bayesian.england_verrall_odp.model_numpyro import _odp_distribution

    odp = _odp_distribution()
    mu, x = np.array([1500.0]), np.array([1400.0])
    literal = (x / PHI) * np.log(mu / PHI) - mu / PHI - gammaln(x / PHI + 1.0)
    np.testing.assert_allclose(
        np.asarray(odp(mu, PHI).log_prob(x)), literal, rtol=DENSITY_RTOL, atol=DENSITY_ATOL
    )
    # and it is NOT the proportional form: dropping the constants would shift
    # the value by a large, easily detected amount
    proportional = (x / PHI) * np.log(mu) - mu / PHI
    assert abs(float(literal[0] - proportional[0])) > 1.0


def test_zero_increment_is_a_valid_cell():
    """An incremental loss of exactly 0 is a real, common cell (a quarter with
    no payments), not a degenerate one: ``log_prob(0) = -mu/phi`` is finite. A
    density that returned -inf there would reject perfectly good triangles."""
    pytest.importorskip("numpyro")
    from ibnr.gallery.bayesian.england_verrall_odp.model_numpyro import _odp_distribution

    odp = _odp_distribution()
    mu = np.array([1000.0])
    lp = np.asarray(odp(mu, PHI).log_prob(np.array([0.0])))
    assert np.isfinite(lp).all()
    np.testing.assert_allclose(lp, -mu / PHI, rtol=DENSITY_RTOL, atol=DENSITY_ATOL)


def test_forward_sampler_matches_the_density_moments():
    """``sample()`` is never used to fit (the site is always observed) but it
    backs ``Predictive``, so it must be the same distribution the density
    describes: mean ``mu``, variance ``phi * mu`` - genuinely over-dispersed."""
    pytest.importorskip("numpyro")
    import jax

    from ibnr.gallery.bayesian.england_verrall_odp.model_numpyro import _odp_distribution

    odp = _odp_distribution()
    mu = np.array([1000.0, 4000.0])
    draws = np.asarray(odp(mu, PHI).sample(jax.random.PRNGKey(0), (200_000,)))
    np.testing.assert_allclose(draws.mean(axis=0), mu, rtol=0.02)
    np.testing.assert_allclose(draws.var(axis=0), PHI * mu, rtol=0.05)


def test_ports_agree_on_the_density_elementwise():
    """The two hand-written densities must agree with each other, cell by cell,
    at the same parameter values - an implementation-vs-implementation check
    that no single-port test can make."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    import pytensor.tensor as pt

    from ibnr.gallery.bayesian.england_verrall_odp.model_numpyro import _odp_distribution
    from ibnr.gallery.bayesian.england_verrall_odp.model_pymc import _odp_logp

    mu = np.array([1000.0, 2000.0, 500.0, 12345.0])
    x = np.array([950.0, 2100.0, 0.0, 12000.0])
    npy = np.asarray(_odp_distribution()(mu, PHI).log_prob(x))
    pmc = np.asarray(_odp_logp(pt.as_tensor(x), pt.as_tensor(mu), PHI).eval())
    # PyMC/PyTensor is float64 and NumPyro float32, so this inherits the same
    # cancellation floor as the identity check above.
    np.testing.assert_allclose(npy, pmc, rtol=DENSITY_RTOL, atol=DENSITY_ATOL)


# -- fast: graphs build with Stan's parameterization -------------------------


def test_numpyro_model_shapes():
    """Structural guard: zero-pinned ``alpha[0]`` AND ``beta[0]`` (ODP pins the
    FIRST beta where the Meyers family pins the last - getting this backwards
    would still sample, just fit a different model), and one ``log_mu`` per
    observed cell."""
    pytest.importorskip("numpyro")
    import jax
    from numpyro import handlers

    from ibnr.gallery.bayesian.england_verrall_odp import model_numpyro

    data = simulate_odp_contract(n_w=6, n_d=6, seed=2)
    tr = handlers.trace(handlers.seed(model_numpyro.odp_model, jax.random.PRNGKey(0))).get_trace(
        data
    )
    alpha = np.asarray(tr["alpha"]["value"])
    beta = np.asarray(tr["beta"]["value"])
    assert alpha.shape == (6,) and float(alpha[0]) == 0.0
    assert beta.shape == (6,) and float(beta[0]) == 0.0
    assert np.asarray(tr["log_mu"]["value"]).shape == (data["len_data"],)
    assert tr["obs"]["is_observed"]


def test_pymc_model_builds():
    """Same guard through the PyMC graph: the free RVs are exactly Stan's
    sampled parameters, with alpha/beta/log_mu deterministic, and `obs` is a
    genuine OBSERVED variable rather than a Potential - which is what makes a
    log_likelihood group possible at all."""
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.england_verrall_odp import model_pymc

    data = simulate_odp_contract(n_w=6, n_d=6, seed=3)
    model = model_pymc.build_model(data)
    assert {v.name for v in model.free_RVs} == {"c", "r_alpha", "r_beta"}
    assert {"alpha", "beta", "log_mu", "obs"} <= set(model.named_vars)
    assert [v.name for v in model.observed_RVs] == ["obs"]


def test_entry_rejects_unknown_backend():
    from ibnr.gallery.bayesian.england_verrall_odp.model import EnglandVerrallODP

    with pytest.raises(ValueError, match="backend must be one of"):
        EnglandVerrallODP().fit(None, backend="jags")


def test_ports_reject_stan_only_controls():
    """``parallel_chains`` / ``max_treedepth`` are cmdstan controls the retro
    harness escalates on; a port that ignored them would report an escalated fit
    that never happened."""
    from ibnr.gallery.bayesian.england_verrall_odp.model import EnglandVerrallODP

    with pytest.raises(ValueError, match="stan-backend controls"):
        EnglandVerrallODP().fit(None, backend="pymc", max_treedepth=15)


# -- slow: NumPyro and PyMC sample the same posterior ------------------------


@pytest.mark.parity
@pytest.mark.slow
def test_numpyro_pymc_parity():
    """The parity gate: both ports sample the same synthetic ODP posterior and
    every compared parameter must agree in mean AND SD within ``z_tol`` MCSE
    units. 2500 draws, not fewer - below ~1000 the SD check is noise-dominated
    (see the CCL card's draw-budget note)."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.england_verrall_odp import model_numpyro, model_pymc
    from ibnr.kernels.parity import compare_posteriors

    data = simulate_odp_contract(n_w=8, n_d=8, seed=7)
    idn = model_numpyro.sample(data, chains=2, iter_warmup=1000, iter_sampling=2500, seed=11)
    idp = model_pymc.sample(data, chains=2, iter_warmup=1000, iter_sampling=2500, seed=11)

    report = compare_posteriors(
        {"numpyro": idn, "pymc": idp}, reference="numpyro", var_names=ODP_VARS
    )
    assert report.passed, f"parity failed:\n{report.failures()}"


@pytest.mark.parity
@pytest.mark.slow
def test_both_ports_emit_a_log_likelihood_group():
    """ELPD/LOO needs a per-observation ``log_likelihood``, and the way a custom
    density is attached decides whether one exists. ``pm.Potential`` yields none
    at all; a ``numpyro.factor`` yields one but leaves ``observed_data``
    degenerate. Both ports therefore use genuine observed variables, and this
    asserts the consequence rather than the mechanism."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.england_verrall_odp import model_numpyro, model_pymc

    data = simulate_odp_contract(n_w=6, n_d=6, seed=5)
    for mod, kwargs in ((model_numpyro, {}), (model_pymc, {})):
        idata = mod.sample(data, chains=1, iter_warmup=200, iter_sampling=200, seed=3, **kwargs)
        assert "log_likelihood" in idata.groups(), f"{mod.__name__} has no log_likelihood"
        ll = np.asarray(next(iter(idata.log_likelihood.data_vars.values())).values)
        assert ll.shape[-1] == data["len_data"]
        assert np.isfinite(ll).all()
