"""gallery.statistical.copula_glm: copula-linked lognormal regressions."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from ibnr import gallery
from ibnr.gallery.statistical.copula_glm.model import CopulaGLM

from .conftest import make_multiline_triangle

START = 2000
SIGMAS = np.array([0.10, 0.15])
DEV_LEVEL = np.log(np.array([0.45, 0.25, 0.15, 0.08, 0.05]))  # incremental LR by dev


def simulate_lognormal_square(rng, *, n_w=8, rho=0.6, sigmas=SIGMAS, premium=1000.0):
    """Full (K, n_w, n_d) cumulative square simulated FROM the copula model."""
    n_lob, n_d = len(sigmas), len(DEV_LEVEL)
    corr = np.full((n_lob, n_lob), rho)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    alpha = rng.normal(0.0, 0.05, size=n_w)  # origin level effects
    incr = np.empty((n_lob, n_w, n_d))
    for w in range(n_w):
        for d in range(n_d):
            z = chol @ rng.standard_normal(n_lob)
            incr[:, w, d] = premium * np.exp(DEV_LEVEL[d] + alpha[w] + sigmas * z)
    return np.cumsum(incr, axis=2)


def fit_on_upper(backend_name, cum, *, premium=1000.0, **fit_kwargs):
    n_lob, n_w, _ = cum.shape
    lobs = {f"lob_{k}": cum[k] for k in range(n_lob)}
    prem = {f"lob_{k}": np.full(n_w, premium) for k in range(n_lob)}
    full = make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)
    cutoff = dt.date(START + n_w - 1, 12, 31)
    entry = CopulaGLM().fit(full, as_of=cutoff, **fit_kwargs)
    return entry, full


def test_copula_registered():
    assert "copula_glm" in gallery.list()
    assert gallery.get("copula_glm").family == "statistical"
    assert "Shi & Frees" in gallery.get("copula_glm").card()


def test_copula_recovers_planted_correlation(backend_name):
    rng = np.random.default_rng(21)
    cum = simulate_lognormal_square(rng, rho=0.6)
    entry, _ = fit_on_upper(backend_name, cum)
    assert abs(entry.corr_[0, 1] - 0.6) < 0.25, f"corr {entry.corr_[0, 1]:.2f} not near 0.6"


def test_copula_recovers_marginal_scale(backend_name):
    rng = np.random.default_rng(22)
    cum = simulate_lognormal_square(rng, rho=0.3)
    entry, full = fit_on_upper(backend_name, cum)
    pred = entry.predict(n_draws=4000, seed=0, param_uncertainty="plugin")
    realized = entry.realized_ultimates(full)
    # point estimates should land within ~10% of the true ultimates
    estimate = pred.mean()[-1]
    np.testing.assert_allclose(estimate, realized[-1], rtol=0.10)
    assert 0.6 < np.mean(entry.sigma_ / SIGMAS) < 1.5


def test_copula_nonpositive_error_and_drop(backend_name):
    rng = np.random.default_rng(23)
    cum = simulate_lognormal_square(rng, rho=0.2)
    cum[0, 1, 2] = cum[0, 1, 1] - 5.0  # negative increment in lob_0
    cum[0, 1, 3:] = cum[0, 1, 2] + np.diff(cum[0, 1, 2:])  # keep later cells consistent

    with pytest.raises(ValueError, match="non-positive increment"):
        fit_on_upper(backend_name, cum)

    entry, _ = fit_on_upper(backend_name, cum, nonpositive="drop")
    pred = entry.predict(n_draws=200, seed=0, param_uncertainty="plugin")
    assert np.isfinite(pred.samples).all()


def test_copula_bootstrap_widens_intervals(backend_name):
    rng = np.random.default_rng(24)
    cum = simulate_lognormal_square(rng, rho=0.4)
    entry, _ = fit_on_upper(backend_name, cum)
    sd_boot = entry.predict(n_draws=8000, seed=1, param_uncertainty="bootstrap").std()[-1]
    sd_plug = entry.predict(n_draws=8000, seed=1, param_uncertainty="plugin").std()[-1]
    assert sd_boot > sd_plug


def test_copula_positive_correlation_widens_grand_total(backend_name):
    rng = np.random.default_rng(25)
    cum = simulate_lognormal_square(rng, rho=0.7)
    entry, _ = fit_on_upper(backend_name, cum)
    independent = CopulaGLM()
    independent.__dict__.update({**entry.__dict__, "corr_": np.eye(2)})
    sd_dep = entry.predict(n_draws=8000, seed=2, param_uncertainty="plugin").std()[-1]
    sd_ind = independent.predict(n_draws=8000, seed=2, param_uncertainty="plugin").std()[-1]
    assert sd_dep > sd_ind * 1.02


def test_copula_target_layout(backend_name):
    rng = np.random.default_rng(26)
    cum = simulate_lognormal_square(rng, rho=0.2)
    entry, full = fit_on_upper(backend_name, cum)
    pred = entry.predict(n_draws=200, seed=0, param_uncertainty="plugin")
    n_lob, n_w = 2, 8
    assert pred.n_targets == n_lob * n_w + n_lob + 1
    assert pred.targets["label"].tolist()[-1] == "total"
    realized = entry.realized_ultimates(full)
    np.testing.assert_allclose(realized[: n_lob * n_w], cum[:, :, -1].reshape(-1))
    # fully developed first origin is a constant target equal to its outcome
    np.testing.assert_allclose(pred.samples[:, 0], cum[0, 0, -1])


def test_copula_df_guard_suggests_hoerl(backend_name):
    # staircase: origin 1 observes all 6 devs, origins 2-3 only dev 1 ->
    # 8 usable cells vs 8 factor params (1 + 2 origin + 5 dev); hoerl has 5.
    rng = np.random.default_rng(27)
    n_w, n_d = 3, 6
    incr = 1000.0 * np.exp(rng.normal(0.0, 0.05, size=(2, n_w, n_d)) + np.linspace(-0.5, -2.5, n_d))
    cum = np.cumsum(incr, axis=2)
    cum[:, 1:, 1:] = np.nan  # later origins: first dev only
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    t = make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)

    with pytest.raises(ValueError, match="hoerl"):
        CopulaGLM().fit(t)
    entry = CopulaGLM().fit(t, dev_effect="hoerl")
    pred = entry.predict(n_draws=200, seed=0, param_uncertainty="plugin")
    assert np.isfinite(pred.samples).all()


def test_copula_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        CopulaGLM().predict()


def test_copula_rejects_bad_options(backend_name):
    rng = np.random.default_rng(28)
    cum = simulate_lognormal_square(rng)
    with pytest.raises(ValueError, match="dev_effect"):
        fit_on_upper(backend_name, cum, dev_effect="spline")
    entry, _ = fit_on_upper(backend_name, cum)
    with pytest.raises(ValueError, match="param_uncertainty"):
        entry.predict(param_uncertainty="exact")
