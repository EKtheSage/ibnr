"""Cross-backend parity for meyers_ccl (milestone 4).

Two tiers:

* **Fast structural tests** (default suite): the NumPyro and PyMC model graphs
  build with the right deterministic shapes and the decreasing-sigma
  construction, and ``ccl_mu_index`` reproduces the Stan ``prev_idx`` recurrence
  exactly. No sampling, no C-compile.
* **Sampling parity** (``parity`` + ``slow``): NumPyro and PyMC sample the same
  small synthetic posterior and agree within MCSE via ``kernels.parity``. This
  runs without cmdstan, so CI can gate parity without a Stan toolchain. Stan is
  the ground truth but its compile is behind ``slow``; the full three-way
  comparison lives in ``scripts/parity_gallery.py``.

Why parity is a gate and not a nicety (CLAUDE.md design decision 7): the point
of the ports is a cross-backend convergence/speed comparison (R-hat, ESS,
divergences, runtime) published in the model card. That comparison is
meaningless until the ports are shown to target the *same posterior* as the
Stan reference - otherwise you are timing two different models. So parity is a
correctness gate that must pass BEFORE any performance number is quoted.

What ``kernels.parity.compare_posteriors`` actually gates: per parameter
element, |mean_ref - mean_port| / combined MCSE-of-mean and
|sd_ref - sd_port| / combined MCSE-of-sd, both required within ``z_tol`` (4 by
default). Scaling by MCSE rather than a flat percentage is what makes this
robust on short chains - a weakly identified parameter (the deepest-dev ``sig``
sees one observation) has a large MCSE and is tolerated automatically. The
marginal two-sample KS statistic is computed and reported for context but does
NOT gate: MCMC autocorrelation inflates it, so it is diagnostic only.

"Same model" is only same if the parameterization is held constant, so these
tests also pin the structural details (zero-pinned ``alpha[0]`` / ``beta[-1]``,
the decreasing-sigma reverse-cumsum construction, which variables are free vs
deterministic) that a re-parameterized port would silently break.
"""

from __future__ import annotations

import numpy as np
import pytest


def simulate_ccl_contract(n_w: int = 8, n_d: int = 8, seed: int = 0) -> dict:
    """A CCL-generated upper-triangle contract dict (the stan_data schema),
    self-contained so parity does not depend on the mart or a Triangle.

    Draws from Meyers' Correlated Chain Ladder itself: log loss for cell (w, d)
    is normal around logprem + logelr + alpha_w + beta_d plus rho times the
    previous accident year's residual at the same dev lag - that AR(1)-across-
    accident-years term is the "correlated" in CCL, and ``prev_idx`` is the
    Stan-side pointer implementing it. ``sig`` is built as a reverse cumsum of
    positive increments so volatility is monotonically decreasing in dev lag.
    """
    rng = np.random.default_rng(seed)
    logprem_o = np.log(rng.uniform(8000, 20000, n_w))
    logelr, rho = -0.5, 0.3
    # Identifiability pins from Meyers' Stan code: alpha[0] and beta[-1] are
    # fixed at 0 (level absorbed by logelr, scale by the last dev lag).
    alpha = np.concatenate([[0.0], rng.normal(0, 0.2, n_w - 1)])
    beta = np.concatenate([np.sort(rng.uniform(-1.5, 0.0, n_d - 1)), [0.0]])
    a = rng.uniform(0.2, 0.6, n_d)
    sig = np.sqrt(np.cumsum(a[::-1])[::-1] * 0.02)

    # Upper triangle only: cell (w, d) is observed when (w-1) + (d-1) < n_d.
    cells = [
        (w, d) for w in range(1, n_w + 1) for d in range(1, n_d + 1) if (w - 1) + (d - 1) < n_d
    ]
    cells.sort()
    # 1-based row pointer to the same dev lag one accident year earlier, 0 when
    # there is none - Stan has no null, so 0 is the sentinel.
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
    logloss = np.zeros(len(cells))
    mu = np.zeros(len(cells))
    for i in range(len(cells)):
        m = logprem[i] + logelr + alpha[w[i] - 1] + beta[d[i] - 1]
        if prev_idx[i] > 0:
            m += rho * (logloss[prev_idx[i] - 1] - mu[prev_idx[i] - 1])
        mu[i] = m
        logloss[i] = rng.normal(m, sig[d[i] - 1])
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


# -- fast: the vectorized mu equals the Stan recurrence ----------------------


def test_ccl_mu_index_matches_recurrence():
    """``ccl_mu_index`` is the load-bearing shared prep: it unrolls Stan's
    sequential mu recurrence into a closed form the vectorized NumPyro/PyMC
    ports can evaluate in one matmul. This pins it against the literal forward
    loop over several triangle shapes - if it drifts, both ports silently
    sample a different model than Stan and parity means nothing."""
    from ibnr.kernels.contract import ccl_mu_index

    rng = np.random.default_rng(1)
    for n_w, n_d in [(6, 6), (10, 10), (4, 7)]:
        data = simulate_ccl_contract(n_w, n_d, seed=n_w + n_d)
        logelr = float(rng.normal(-0.4))
        alpha = np.concatenate([[0.0], rng.normal(0, 0.3, n_w - 1)])
        beta = np.concatenate([rng.normal(0, 0.3, n_d - 1), [0.0]])
        rho = float(rng.uniform(-0.9, 0.9))

        w0, d0 = data["w"] - 1, data["d"] - 1
        prev0 = data["prev_idx"] - 1
        # reference: the literal forward recurrence
        mu_ref = np.zeros(data["len_data"])
        for i in range(data["len_data"]):
            m = data["logprem"][i] + logelr + alpha[w0[i]] + beta[d0[i]]
            if prev0[i] >= 0:
                m += rho * (data["logloss"][prev0[i]] - mu_ref[prev0[i]])
            mu_ref[i] = m
        # Vectorized closed form: substituting mu_prev repeatedly turns the
        # recurrence into an alternating (-rho)**k weighted sum over each
        # cell's accident-year chain. colmask selects the chain, expo the power.
        idx = ccl_mu_index(data)
        base = data["logprem"] + logelr + alpha[w0] + beta[d0]
        big_b = base + rho * idx["logloss_prev"]
        pow_table = np.stack([(-rho) ** k for k in range(int(idx["expo"].max()) + 1)])
        mu_vec = (idx["colmask"] * pow_table[idx["expo"]]) @ big_b
        # atol 1e-10: this is an exact algebraic identity, only float assoc.
        assert np.allclose(mu_ref, mu_vec, atol=1e-10)


# -- fast: NumPyro graph builds with correct shapes --------------------------


def test_numpyro_model_shapes():
    """Cheap structural guard on the NumPyro port: trace the model once (no
    sampling) and check the parameterization matches Stan's - zero-pinned
    alpha[0]/beta[-1], one sigma per dev lag, strictly decreasing in dev, and
    rho inside (-1, 1). Runs in the default suite because it needs no
    compilation, so re-parameterization regressions surface immediately rather
    than only in the slow parity job."""
    pytest.importorskip("numpyro")
    import jax
    from numpyro import handlers

    from ibnr.gallery.bayesian.meyers_ccl import model_numpyro

    data = simulate_ccl_contract(n_w=7, n_d=7, seed=2)
    # a_ig is an interval-constrained ImproperUniform carrying its density as a
    # factor (Stan's construction, matching model.stan's upper bound). NumPyro
    # can infer such a site but not FORWARD-SAMPLE it, so tracing supplies a
    # value; every other site is drawn from its prior as before.
    fixed = handlers.substitute(model_numpyro.ccl_model, {"a_ig": np.full(7, 1.0)})
    tr = handlers.trace(handlers.seed(fixed, jax.random.PRNGKey(0))).get_trace(data)
    alpha = np.asarray(tr["alpha"]["value"])
    beta = np.asarray(tr["beta"]["value"])
    sig = np.asarray(tr["sig"]["value"])
    mu = np.asarray(tr["mu"]["value"])
    assert alpha.shape == (7,) and float(alpha[0]) == 0.0
    assert beta.shape == (7,) and float(beta[-1]) == 0.0
    assert sig.shape == (7,)
    assert mu.shape == (data["len_data"],)
    # sig2 = reverse cumsum of positive a -> sig strictly decreasing in dev lag
    assert np.all(np.diff(sig) < 0)
    assert -1.0 < float(np.asarray(tr["rho"]["value"])) < 1.0


# -- fast: PyMC graph builds with the expected variables ---------------------


def test_pymc_model_builds():
    """Same structural guard for the PyMC port, expressed through the model
    graph: the free RV set must be exactly the non-centered raw parameters
    (r_alpha/r_beta/a_ig/r_rho + logelr), with alpha/beta/rho/sig/mu as
    deterministics. Pinning free-vs-deterministic is how a centered/non-centered
    drift - which would invalidate any convergence comparison - gets caught."""
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.meyers_ccl import model_pymc

    data = simulate_ccl_contract(n_w=7, n_d=7, seed=3)
    model = model_pymc.build_model(data)
    names = set(model.named_vars)
    assert {"logelr", "r_alpha", "r_beta", "a_ig", "r_rho"} <= names  # sampled
    assert {"alpha", "beta", "rho", "sig", "sig2", "mu"} <= names  # deterministics
    assert "obs" in names
    # free (sampled) RVs only; deterministics are not free
    free = {v.name for v in model.free_RVs}
    assert free == {"logelr", "r_alpha", "r_beta", "a_ig", "r_rho"}


def test_entry_rejects_unknown_backend():
    """The backend selector is validated up front, so a typo fails before any
    data prep or sampler startup cost."""
    from ibnr.gallery.bayesian.meyers_ccl.model import MeyersCCL

    with pytest.raises(ValueError, match="backend must be one of"):
        MeyersCCL().fit(None, backend="jags")


def test_ports_reject_stan_only_controls():
    """``parallel_chains`` / ``max_treedepth`` are cmdstan controls the retro
    harness escalates on. A port that silently ignored them would report an
    escalated fit that never happened, so they are a hard error instead.

    ``triangle=None`` is the assertion: the guard has to fire BEFORE any data
    prep, so the ValueError arrives rather than the AttributeError that
    ``None.as_of(...)`` would raise. CCL validated these AFTER building its
    contract until this test was added, which is why it reported a missing
    premium field where its three siblings reported the bad argument."""
    from ibnr.gallery.bayesian.meyers_ccl.model import MeyersCCL

    with pytest.raises(ValueError, match="stan-backend controls"):
        MeyersCCL().fit(None, backend="numpyro", parallel_chains=4)
    with pytest.raises(ValueError, match="stan-backend controls"):
        MeyersCCL().fit(None, backend="numpyro", max_treedepth=12)


# -- slow: NumPyro and PyMC sample the same posterior ------------------------


@pytest.mark.parity
@pytest.mark.slow
def test_numpyro_pymc_parity():
    """The actual parity gate: both ports sample the same synthetic CCL
    posterior and every parameter element must agree in mean AND SD within
    ``z_tol`` MCSE units (``kernels.parity.compare_posteriors``). Marked
    ``parity`` + ``slow`` because it runs two real MCMC fits; it needs no
    cmdstan, so CI can enforce parity without a Stan toolchain - the
    Stan-as-ground-truth three-way run lives in ``scripts/parity_gallery.py``.
    """
    # Small triangle + modest draws keep PyMC's PyTensor sampler tractable on a
    # BLAS-less install. Short chains inflate MCSE, but that is exactly what the
    # z-score scaling absorbs, so shrinking the budget loosens the test rather
    # than making it flaky. Identical seed on both sides is cosmetic - the
    # samplers differ, so the check is statistical either way.
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.meyers_ccl import model_numpyro, model_pymc
    from ibnr.kernels.parity import compare_posteriors

    data = simulate_ccl_contract(n_w=8, n_d=8, seed=7)
    idn = model_numpyro.sample(data, chains=2, iter_warmup=600, iter_sampling=600, seed=11)
    idp = model_pymc.sample(data, chains=2, iter_warmup=600, iter_sampling=600, seed=11)

    # NumPyro stands in as reference here only because Stan is unavailable
    # without a toolchain; Stan remains ground truth in the scripted run.
    report = compare_posteriors({"numpyro": idn, "pymc": idp}, reference="numpyro")
    assert report.passed, f"parity failed:\n{report.failures()}"
