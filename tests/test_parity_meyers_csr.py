"""Cross-backend parity for meyers_csr (milestone 5).

Same two tiers as ``test_parity_meyers.py`` (which covers CCL):

* **Fast structural tests** (default suite): the NumPyro and PyMC graphs build
  with Stan's parameterization - zero-pinned ``alpha[0]``/``beta[-1]``, the
  decreasing-sigma reverse cumsum, and the settlement-rate ``speedup`` - with no
  sampling and no C-compile.
* **Sampling parity** (``parity`` + ``slow``): both ports sample the same small
  synthetic posterior and agree within MCSE via ``kernels.parity``. It needs no
  cmdstan, so CI can gate parity without a Stan toolchain; the three-way run
  with Stan as ground truth lives in ``scripts/parity_meyers.py --model
  meyers_csr``.

CSR-specific risk this file exists to catch: ``speedup[w] = (1 - gamma)^(w-1)``
is a *recurrence* in Stan. Written naively as a power it is NaN whenever a
warmup excursion pushes gamma past 1 (the base goes negative under a float
exponent), and written as a scan it is slow. Both ports use a cumulative
product; ``test_speedup_matches_stan_recurrence`` pins it against the literal
loop, and the negative-gamma case is checked explicitly because that is the one
a plain power silently turns into NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

CSR_VARS = ("logelr", "alpha", "beta", "gamma", "sig")


def simulate_csr_contract(n_w: int = 8, n_d: int = 8, seed: int = 0, gamma: float = 0.03) -> dict:
    """A CSR-generated upper-triangle contract dict (the ``stan_data`` schema),
    self-contained so parity depends on neither the mart nor a Triangle.

    Draws from Meyers' Changing Settlement Rate model itself: log loss for cell
    (w, d) is normal around ``logprem + logelr + alpha_w + beta_d * (1-gamma)^(w-1)``.
    A positive ``gamma`` is a settlement speedup - later accident years' whole
    development profile shrinks toward zero. ``sig`` is a reverse cumsum of
    positive increments, so volatility decreases in dev lag as the model assumes.

    ``prev_idx`` is included for schema fidelity with the shared Meyers contract
    even though CSR never reads it (no across-origin correlation term).
    """
    rng = np.random.default_rng(seed)
    logprem_o = np.log(rng.uniform(8000, 20000, n_w))
    logelr = -0.5
    # Identifiability pins from Meyers' Stan code: alpha[0] = 0 (level absorbed
    # by logelr) and beta[-1] = 0 (scale absorbed by the last dev lag).
    alpha = np.concatenate([[0.0], rng.normal(0, 0.2, n_w - 1)])
    beta = np.concatenate([np.sort(rng.uniform(-1.5, 0.0, n_d - 1)), [0.0]])
    a = rng.uniform(0.2, 0.6, n_d)
    sig = np.sqrt(np.cumsum(a[::-1])[::-1] * 0.02)
    speedup = (1.0 - gamma) ** np.arange(n_w)

    cells = [
        (w, d) for w in range(1, n_w + 1) for d in range(1, n_d + 1) if (w - 1) + (d - 1) < n_d
    ]
    cells.sort()
    row_of = {c: i for i, c in enumerate(cells)}
    w = np.array([c[0] for c in cells])
    d = np.array([c[1] for c in cells])
    prev_idx = np.array(
        [
            row_of[(c[0] - 1, c[1])] + 1 if c[0] > 1 and (c[0] - 1, c[1]) in row_of else 0
            for c in cells
        ]
    )
    logprem = logprem_o[w - 1]
    mu = logprem + logelr + alpha[w - 1] + beta[d - 1] * speedup[w - 1]
    logloss = rng.normal(mu, sig[d - 1])
    return {
        "len_data": len(cells),
        "n_w": n_w,
        "n_d": n_d,
        "w": w,
        "d": d,
        "prev_idx": prev_idx,
        "logprem": logprem,
        "logloss": logloss,
    }


# -- fast: the cumulative-product speedup equals the Stan recurrence ----------


@pytest.mark.parametrize("gamma", [0.0, 0.05, -0.05, 0.9, 1.4, 2.0])
def test_speedup_matches_stan_recurrence(gamma):
    """``speedup`` is the one construction the ports rewrite, so it is pinned
    against Stan's literal forward loop - including gamma > 1, where the base
    ``1 - gamma`` turns negative. That case is not academic: gamma is an
    unconstrained parameter, so warmup can visit it, and a float power of a
    negative base is NaN, which would poison the whole chain."""
    n_w = 9
    ref = np.empty(n_w)  # Stan: speedup[1] = 1; speedup[i] = speedup[i-1] * (1 - gamma)
    ref[0] = 1.0
    for i in range(1, n_w):
        ref[i] = ref[i - 1] * (1.0 - gamma)
    got = np.cumprod(np.concatenate([np.ones(1), np.full(n_w - 1, 1.0 - gamma)]))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)
    assert np.isfinite(got).all()


# -- fast: NumPyro graph builds with correct shapes --------------------------


def _trace_numpyro(data, seed: int = 0, a_ig: float = 1.0):
    """Trace the NumPyro CSR model once, with ``a_ig`` substituted.

    ``a_ig`` is an interval-constrained ``ImproperUniform`` carrying its density
    as a factor - Stan's own construction (see ``model_numpyro``). NumPyro can
    *infer* such a site but cannot FORWARD-SAMPLE it, so tracing must supply a
    value. Everything else is sampled from its prior as usual.
    """
    import jax
    import numpy as _np
    from numpyro import handlers

    from ibnr.gallery.bayesian.meyers_csr import model_numpyro

    fixed = handlers.substitute(
        model_numpyro.csr_model, {"a_ig": _np.full(data["n_d"], a_ig, dtype=float)}
    )
    return handlers.trace(handlers.seed(fixed, jax.random.PRNGKey(seed))).get_trace(data)


def test_numpyro_a_ig_respects_stans_upper_bound():
    """The ``a_ig`` site must be constrained to Stan's ``(0, 1e5)``.

    This is the bug that broke parity: an *unbounded* ``InverseGamma(1,1)`` looks
    harmless (the truncated PRIOR mass is ~1e-5) but the POSTERIOR piles into the
    ``a -> 0`` corner at deep development lags - measured at up to 10.6% of draws
    above the bound - which dragged ``sig`` 5-12% below the Stan reference.
    """
    pytest.importorskip("numpyro")
    from numpyro.distributions import constraints

    from ibnr.gallery.bayesian.meyers_csr import model_numpyro

    data = simulate_csr_contract(n_w=6, n_d=6, seed=1)
    tr = _trace_numpyro(data)
    support = tr["a_ig"]["fn"].support
    # the per-element constraint sits inside an IndependentConstraint (event dim)
    interval = getattr(support, "base_constraint", support)
    assert isinstance(interval, constraints._Interval)
    assert interval.lower_bound == 0.0
    assert interval.upper_bound == model_numpyro.A_IG_MAX == 1e5


def test_pymc_a_ig_respects_stans_upper_bound():
    """Same guard for the PyMC port: draws from the prior never exceed Stan's
    bound, and the logp above it is -inf rather than merely small."""
    pytest.importorskip("pymc")
    import pymc as pm

    from ibnr.gallery.bayesian.meyers_csr import model_pymc

    data = simulate_csr_contract(n_w=6, n_d=6, seed=1)
    model = model_pymc.build_model(data)
    a_ig = model["a_ig"]
    draws = pm.draw(a_ig, draws=20_000, random_seed=0)
    assert draws.max() <= model_pymc.A_IG_MAX == 1e5
    over = pm.logp(a_ig, np.full(data["n_d"], 2e5)).eval()
    assert np.isneginf(over).all()


def test_numpyro_model_shapes():
    """Structural guard on the NumPyro port: trace once (no sampling) and check
    the parameterization matches Stan's - zero-pinned alpha[0]/beta[-1], one
    sigma per dev lag strictly decreasing in dev, speedup starting at 1, and mu
    one value per observed cell."""
    pytest.importorskip("numpyro")

    data = simulate_csr_contract(n_w=7, n_d=7, seed=2)
    tr = _trace_numpyro(data, seed=0)
    alpha = np.asarray(tr["alpha"]["value"])
    beta = np.asarray(tr["beta"]["value"])
    sig = np.asarray(tr["sig"]["value"])
    speedup = np.asarray(tr["speedup"]["value"])
    mu = np.asarray(tr["mu"]["value"])
    assert alpha.shape == (7,) and float(alpha[0]) == 0.0
    assert beta.shape == (7,) and float(beta[-1]) == 0.0
    assert sig.shape == (7,)
    assert speedup.shape == (7,) and float(speedup[0]) == 1.0
    assert mu.shape == (data["len_data"],)
    # sig2 = reverse cumsum of positive a -> sig strictly decreasing in dev lag
    assert np.all(np.diff(sig) < 0)


def test_numpyro_speedup_is_the_geometric_trend():
    """The traced ``speedup`` must equal (1 - gamma)^(w-1) for the gamma that was
    actually drawn - i.e. the cumprod really is the recurrence, inside the model
    rather than only in the standalone check above."""
    pytest.importorskip("numpyro")

    data = simulate_csr_contract(n_w=6, n_d=6, seed=5)
    tr = _trace_numpyro(data, seed=3)
    gamma = float(np.asarray(tr["gamma"]["value"]))
    speedup = np.asarray(tr["speedup"]["value"])
    # rtol 1e-5, not machine epsilon: JAX runs in float32 by default (as does the
    # CCL port - x64 is deliberately NOT enabled, so the two ports stay
    # comparable), and float32 gives ~1e-7 relative error here. That is orders of
    # magnitude below the MCSE the parity gate works in.
    np.testing.assert_allclose(speedup, (1.0 - gamma) ** np.arange(6), rtol=1e-5)


# -- fast: PyMC graph builds with the expected variables ---------------------


def test_pymc_model_builds():
    """Same structural guard for the PyMC port, through the model graph: the
    free RV set must be exactly Stan's sampled parameters, with everything else
    deterministic. Pinning free-vs-deterministic is how a re-parameterization -
    which would invalidate any convergence comparison - gets caught."""
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.meyers_csr import model_pymc

    data = simulate_csr_contract(n_w=7, n_d=7, seed=3)
    model = model_pymc.build_model(data)
    names = set(model.named_vars)
    assert {"logelr", "r_alpha", "r_beta", "a_ig", "gamma"} <= names  # sampled
    assert {"alpha", "beta", "speedup", "sig", "sig2", "mu"} <= names  # deterministics
    assert "obs" in names
    free = {v.name for v in model.free_RVs}
    assert free == {"logelr", "r_alpha", "r_beta", "a_ig", "gamma"}


def test_ports_agree_on_mu_at_fixed_parameters():
    """The two ports must compute the SAME mu from the same parameter values -
    a pure-algebra check that needs no sampler and would catch a transcription
    slip (e.g. adding the trend instead of multiplying) that structural shape
    tests cannot see."""
    pytest.importorskip("numpyro")
    data = simulate_csr_contract(n_w=6, n_d=6, seed=9)
    rng = np.random.default_rng(0)
    n_w, n_d = data["n_w"], data["n_d"]
    logelr = float(rng.normal(-0.4))
    alpha = np.concatenate([[0.0], rng.normal(0, 0.3, n_w - 1)])
    beta = np.concatenate([rng.normal(0, 0.3, n_d - 1), [0.0]])
    gamma = 0.07

    w0, d0 = data["w"] - 1, data["d"] - 1
    speedup = np.cumprod(np.concatenate([np.ones(1), np.full(n_w - 1, 1.0 - gamma)]))
    expected = data["logprem"] + logelr + alpha[w0] + beta[d0] * speedup[w0]
    # the literal Stan loop, cell by cell
    ref = np.array(
        [
            data["logprem"][i] + logelr + alpha[w0[i]] + beta[d0[i]] * speedup[w0[i]]
            for i in range(data["len_data"])
        ]
    )
    np.testing.assert_allclose(expected, ref, atol=1e-12)


def test_parity_summaries_are_not_rounded():
    """``kernels.parity`` must read FULL-PRECISION summaries.

    ``az.summary`` rounds to 3 decimals by default, and every parity z-score is
    a difference of two summaries divided by their MCSE. Rounding the numerator
    onto a 1e-3 grid inflates z for any parameter whose MCSE is smaller than
    that - a parity failure invented by the formatter rather than the sampler.
    (Symptom when this regressed: reported z-scores landing on exact multiples
    of sqrt(2).) Guarded here by requiring at least one summary entry to carry
    more precision than the 3-decimal default would leave.
    """
    pytest.importorskip("arviz")
    import arviz as az

    from ibnr.kernels import parity

    rng = np.random.default_rng(0)
    idata = az.from_dict(posterior={"logelr": rng.normal(size=(4, 500))})
    summ = parity._summ(idata, ["logelr"])
    values = summ.loc["logelr", ["mean", "sd", "mcse_mean", "mcse_sd"]].astype(float)
    assert (values != values.round(3)).any(), f"summaries look rounded: {values.to_dict()}"


def _parity_script():
    """Import ``scripts/parity_meyers.py`` by path (scripts/ is not a package)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "parity_meyers.py"
    spec = importlib.util.spec_from_file_location("parity_meyers_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_script_model_registry():
    """``scripts/parity_meyers.py`` grew a ``--model`` flag with CSR; its
    registry is the only model-specific wiring in an otherwise model-agnostic
    script, so it is worth pinning.

    Two things matter. Every key must be a real gallery entry (a typo would only
    surface after a multi-hour fit), and **meyers_ccl must keep the ``meyers``
    filename stem** - the milestone-4 results at
    ``analysis/results/{parity,convergence}_meyers.csv`` are published and cited
    from the CCL card, so the generalization must not silently move them.
    """
    from ibnr import gallery

    module = _parity_script()
    assert set(module.MODELS) <= set(gallery.list())
    assert module.MODELS["meyers_ccl"]["stem"] == "meyers"
    assert module.MODELS["meyers_csr"]["stem"] == "meyers_csr"
    # each model's parity vars must name its own signature parameter: rho is
    # CCL's across-origin correlation, gamma is CSR's settlement-rate trend
    assert "rho" in module.MODELS["meyers_ccl"]["parity_vars"]
    assert "gamma" in module.MODELS["meyers_csr"]["parity_vars"]
    # the reference posterior of each entry is defined on ONE loss field
    assert module.MODELS["meyers_ccl"]["loss_field"] == "reported_loss"
    assert module.MODELS["meyers_csr"]["loss_field"] == "paid_loss"


def test_parity_script_synthetic_csr_triangle_is_a_run_off_square():
    """The CSR simulator must emit a full square on the shared conventions
    (dev_lag in months from origin start, year-end eval dates) - ``fit()``
    slices the upper triangle out of it with ``as_of``, so a mis-built square
    would quietly change what every backend is fit to."""
    module = _parity_script()
    tri = module._synthetic_csr_triangle(seed=3)
    assert tri.meta.measure == "cumulative"
    assert set(tri.fields) == {"paid_loss", "earned_premium"}
    assert tri.dev_lags == [12 * (d + 1) for d in range(10)]
    assert len(tri.origins) == 10
    train = tri.as_of("1997-12-31").select_fields("paid_loss")
    assert train.count() == 55  # the 10x10 upper triangle


def test_entry_rejects_unknown_backend():
    """The backend selector is validated up front, so a typo fails before any
    data prep or sampler startup cost."""
    from ibnr.gallery.bayesian.meyers_csr.model import MeyersCSR

    with pytest.raises(ValueError, match="backend must be one of"):
        MeyersCSR().fit(None, backend="jags")


def test_ports_reject_stan_only_controls():
    """``parallel_chains`` / ``max_treedepth`` are cmdstan controls the retro
    harness escalates on. A port that silently ignored them would report an
    escalated fit that never happened, so they are a hard error instead."""
    from ibnr.gallery.bayesian.meyers_csr.model import MeyersCSR

    with pytest.raises(ValueError, match="stan-backend controls"):
        MeyersCSR().fit(None, backend="numpyro", parallel_chains=4)


# -- slow: NumPyro and PyMC sample the same posterior ------------------------


@pytest.mark.parity
@pytest.mark.slow
def test_numpyro_pymc_parity():
    """The parity gate: both ports sample the same synthetic CSR posterior and
    every parameter element must agree in mean AND SD within ``z_tol`` MCSE
    units. Marked ``parity`` + ``slow`` because it runs two real MCMC fits; it
    needs no cmdstan, so CI can enforce parity without a Stan toolchain."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.meyers_csr import model_numpyro, model_pymc
    from ibnr.kernels.parity import compare_posteriors

    data = simulate_csr_contract(n_w=8, n_d=8, seed=7)
    idn = model_numpyro.sample(data, chains=2, iter_warmup=600, iter_sampling=600, seed=11)
    idp = model_pymc.sample(data, chains=2, iter_warmup=600, iter_sampling=600, seed=11)

    report = compare_posteriors(
        {"numpyro": idn, "pymc": idp}, reference="numpyro", var_names=CSR_VARS
    )
    assert report.passed, f"parity failed:\n{report.failures()}"
