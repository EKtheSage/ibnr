"""gallery.statistical.sur: FGLS multivariate chain ladder."""

from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pytest

from ibnr import gallery
from ibnr.gallery.statistical.sur.model import SUR

from .conftest import make_multiline_triangle

START = 2000
FACTORS = np.array([1.5, 1.2, 1.1, 1.05])
SIGMAS = np.array([2.0, 3.0])


def simulate_cl_square(rng, *, n_w=12, rho=0.0, factors=FACTORS, sigmas=SIGMAS):
    """Full (K, n_w, n_d) cumulative square from a chain-ladder process with
    cross-line error correlation ``rho`` (Mack variance scaling sqrt(C))."""
    n_lob, n_d = len(sigmas), len(factors) + 1
    corr = np.full((n_lob, n_lob), rho)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    cum = np.empty((n_lob, n_w, n_d))
    cum[:, :, 0] = 1000.0 * rng.uniform(0.8, 1.2, size=(n_lob, n_w))
    for d in range(n_d - 1):
        eta = (rng.standard_normal((n_w, n_lob)) @ chol.T).T * sigmas[:, None]
        cum[:, :, d + 1] = factors[d] * cum[:, :, d] + np.sqrt(cum[:, :, d]) * eta
    assert (cum > 0).all(), "simulation produced non-positive cumulatives"
    return cum


def fit_on_upper(backend_name, cum, **fit_kwargs):
    """Build the full-square triangle, fit on the upper-triangle as_of slice."""
    n_lob, n_w, _ = cum.shape
    lobs = {f"lob_{k}": cum[k] for k in range(n_lob)}
    full = make_multiline_triangle(backend_name, lobs, start_year=START)
    cutoff = dt.date(START + n_w - 1, 12, 31)
    entry = SUR().fit(full, as_of=cutoff, **fit_kwargs)
    return entry, full


def volume_weighted_cl_ultimates(cum, obs_mask):
    """Hand-rolled volume-weighted chain ladder per line on masked data."""
    n_lob, n_w, n_d = cum.shape
    ults = np.empty((n_lob, n_w))
    for k in range(n_lob):
        factors = []
        for d in range(n_d - 1):
            pair = obs_mask[k, :, d] & obs_mask[k, :, d + 1]
            factors.append(cum[k, pair, d + 1].sum() / cum[k, pair, d].sum())
        for w in range(n_w):
            latest = np.nonzero(obs_mask[k, w])[0][-1]
            ult = cum[k, w, latest]
            for d in range(latest, n_d - 1):
                ult *= factors[d]
            ults[k, w] = ult
    return ults


def test_sur_registered():
    assert "sur" in gallery.list()
    assert gallery.get("sur").family == "statistical"
    assert "Seemingly Unrelated Regression" in gallery.get("sur").card()


def test_sur_reduces_to_chain_ladder_at_zero_correlation(backend_name):
    rng = np.random.default_rng(42)
    cum = simulate_cl_square(rng, rho=0.0)
    entry, _ = fit_on_upper(backend_name, cum)
    c = entry.contract_
    pred = entry.predict(n_draws=20_000, seed=1)

    cl = volume_weighted_cl_ultimates(c["cum"], c["obs_mask"])
    got = pred.mean()[: c["n_lob"] * c["n_w"]].reshape(c["n_lob"], c["n_w"])
    np.testing.assert_allclose(got, cl, rtol=0.02)


def test_sur_recovers_planted_correlation(backend_name):
    rng = np.random.default_rng(7)
    cum = simulate_cl_square(rng, rho=0.7)
    entry, _ = fit_on_upper(backend_name, cum)

    fgls_corrs = []
    for tr in entry.transitions_:
        if tr["method"] == "fgls":
            sd = np.sqrt(np.diag(tr["sigma"]))
            fgls_corrs.append(tr["sigma"][0, 1] / (sd[0] * sd[1]))
    assert fgls_corrs, "no transition was estimated by full FGLS"
    mean_corr = np.mean(fgls_corrs)
    assert 0.4 < mean_corr < 1.0, f"recovered correlation {mean_corr:.2f} not near 0.7"
    assert entry.pooled_corr_[0, 1] > 0.3


def test_sur_positive_correlation_widens_grand_total(backend_name):
    rng = np.random.default_rng(11)
    cum = simulate_cl_square(rng, rho=0.7)
    entry, _ = fit_on_upper(backend_name, cum)

    independent = copy.deepcopy(entry)
    for tr in independent.transitions_:
        tr["sigma"] = np.diag(np.diag(tr["sigma"]))

    sd_dep = entry.predict(n_draws=20_000, seed=3, param_uncertainty=False).std()[-1]
    sd_ind = independent.predict(n_draws=20_000, seed=3, param_uncertainty=False).std()[-1]
    assert sd_dep > sd_ind * 1.05


def test_sur_parameter_uncertainty_widens_intervals(backend_name):
    rng = np.random.default_rng(5)
    cum = simulate_cl_square(rng, rho=0.3)
    entry, _ = fit_on_upper(backend_name, cum)
    sd_with = entry.predict(n_draws=20_000, seed=2, param_uncertainty=True).std()[-1]
    sd_without = entry.predict(n_draws=20_000, seed=2, param_uncertainty=False).std()[-1]
    assert sd_with > sd_without


def test_sur_target_layout_and_realized(backend_name):
    rng = np.random.default_rng(3)
    cum = simulate_cl_square(rng, rho=0.2)
    entry, full = fit_on_upper(backend_name, cum)
    c = entry.contract_
    pred = entry.predict(n_draws=500, seed=0)

    n_cells = c["n_lob"] * c["n_w"]
    assert pred.n_targets == n_cells + c["n_lob"] + 1
    labels = pred.targets["label"].tolist()
    assert labels[-1] == "total"
    assert labels[n_cells] == "lob_0/total"
    assert np.isfinite(pred.samples).all()

    realized = entry.realized_ultimates(full)
    assert realized.shape == (pred.n_targets,)
    np.testing.assert_allclose(realized[:n_cells], cum[:, :, -1].reshape(-1))
    np.testing.assert_allclose(realized[-1], cum[:, :, -1].sum())

    table = pred.summary(observed=realized)
    assert {"estimate", "se", "cv", "outcome", "percentile"} <= set(table.columns)


def test_sur_small_sample_ladder_kicks_in(backend_name):
    # 6 origins x 6 devs, K=2: late transitions have < K+2 = 4 pairs
    rng = np.random.default_rng(9)
    cum = simulate_cl_square(rng, n_w=6, rho=0.0, factors=np.array([1.4, 1.2, 1.1, 1.05, 1.02]))
    entry, _ = fit_on_upper(backend_name, cum)
    methods = [tr["method"] for tr in entry.transitions_]
    assert methods[0] == "fgls"
    assert "pooled_corr" in methods
    assert methods[-1] == "tail"  # last transition has a single pair
    pred = entry.predict(n_draws=500, seed=0)
    assert np.isfinite(pred.samples).all()


def test_sur_intercept_variant_fits(backend_name):
    rng = np.random.default_rng(13)
    cum = simulate_cl_square(rng, rho=0.2)
    entry, _ = fit_on_upper(backend_name, cum, intercept=True)
    pred = entry.predict(n_draws=500, seed=0)
    assert np.isfinite(pred.samples).all()
    assert entry.transitions_[0]["beta"].shape == (2, 2)


def test_sur_rejects_nonpositive_cumulatives(backend_name):
    cum = np.array([[[100.0, 150.0], [110.0, np.nan]]])
    bad = np.concatenate([cum, -cum])  # second lob negative
    t = make_multiline_triangle(backend_name, {"lob_a": bad[0], "lob_b": bad[1]}, start_year=START)
    with pytest.raises(ValueError, match="non-positive cumulative"):
        SUR().fit(t)


def test_sur_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        SUR().predict()
