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
  comparison lives in ``scripts/parity_meyers.py``.
"""

from __future__ import annotations

import numpy as np
import pytest


def simulate_ccl_contract(n_w: int = 8, n_d: int = 8, seed: int = 0) -> dict:
    """A CCL-generated upper-triangle contract dict (the stan_data schema),
    self-contained so parity does not depend on the mart or a Triangle."""
    rng = np.random.default_rng(seed)
    logprem_o = np.log(rng.uniform(8000, 20000, n_w))
    logelr, rho = -0.5, 0.3
    alpha = np.concatenate([[0.0], rng.normal(0, 0.2, n_w - 1)])
    beta = np.concatenate([np.sort(rng.uniform(-1.5, 0.0, n_d - 1)), [0.0]])
    a = rng.uniform(0.2, 0.6, n_d)
    sig = np.sqrt(np.cumsum(a[::-1])[::-1] * 0.02)

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
        # vectorized closed form
        idx = ccl_mu_index(data)
        base = data["logprem"] + logelr + alpha[w0] + beta[d0]
        big_b = base + rho * idx["logloss_prev"]
        pow_table = np.stack([(-rho) ** k for k in range(int(idx["expo"].max()) + 1)])
        mu_vec = (idx["colmask"] * pow_table[idx["expo"]]) @ big_b
        assert np.allclose(mu_ref, mu_vec, atol=1e-10)


# -- fast: NumPyro graph builds with correct shapes --------------------------


def test_numpyro_model_shapes():
    pytest.importorskip("numpyro")
    import jax
    from numpyro import handlers

    from ibnr.gallery.bayesian.meyers_ccl import model_numpyro

    data = simulate_ccl_contract(n_w=7, n_d=7, seed=2)
    seeded = handlers.seed(model_numpyro.ccl_model, jax.random.PRNGKey(0))
    tr = handlers.trace(seeded).get_trace(data)
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
    from ibnr.gallery.bayesian.meyers_ccl.model import MeyersCCL

    with pytest.raises(ValueError, match="backend must be one of"):
        MeyersCCL().fit(None, backend="jags")


# -- slow: NumPyro and PyMC sample the same posterior ------------------------


@pytest.mark.parity
@pytest.mark.slow
def test_numpyro_pymc_parity():
    # Confirms the two ports target the same posterior via kernels.parity (MCSE
    # z-scores). Small triangle + modest draws keep PyMC's PyTensor sampler
    # tractable on a BLAS-less install; the wider MCSE is handled by the check.
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.meyers_ccl import model_numpyro, model_pymc
    from ibnr.kernels.parity import compare_posteriors

    data = simulate_ccl_contract(n_w=8, n_d=8, seed=7)
    idn = model_numpyro.sample(data, chains=2, iter_warmup=600, iter_sampling=600, seed=11)
    idp = model_pymc.sample(data, chains=2, iter_warmup=600, iter_sampling=600, seed=11)

    report = compare_posteriors({"numpyro": idn, "pymc": idp}, reference="numpyro")
    assert report.passed, f"parity failed:\n{report.failures()}"
