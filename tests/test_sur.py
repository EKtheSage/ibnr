"""gallery.statistical.sur: FGLS multivariate chain ladder (Zhang 2010).

What this file protects: SUR is the frequentist *dependence* baseline the ML
entries are measured against, so the invariants that matter are (a) it collapses
to the volume-weighted chain ladder when lines are independent - otherwise a
"multivariate" reserve is not comparable to the univariate benchmark - and (b)
positive cross-line correlation actually widens the diversified grand total,
which is the whole reason a multi-line model exists.

Strategy: simulate a *full* (K, n_w, n_d) chain-ladder square with a planted
cross-line error correlation, hand it to the entry through ``as_of()`` so it
only sees the upper triangle, then score against the square's last column (the
known realized ultimates). Data is generated in-test rather than loaded from the
Schedule P mart so these run everywhere (CLAUDE.md: in-repo fixtures use public
/ synthetic samples; mart-backed smokes live in ``test_statistical_mart.py``).

Every test runs on BOTH ibis backends via the ``backend_name`` fixture.
"""

from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pytest

from ibnr import gallery
from ibnr.gallery.statistical.sur.model import SUR

from .conftest import make_multiline_triangle

START = 2000
# Decaying age-to-age factors: 5 dev steps, a realistic paid-loss development
# shape (fast early growth, near-flat tail).
FACTORS = np.array([1.5, 1.2, 1.1, 1.05])
# Per-line Mack sigmas; deliberately different across lines so the FGLS
# covariance has to recover unequal scales, not just a correlation.
SIGMAS = np.array([2.0, 3.0])


def simulate_cl_square(rng, *, n_w=12, rho=0.0, factors=FACTORS, sigmas=SIGMAS):
    """Full (K, n_w, n_d) cumulative square from a chain-ladder process with
    cross-line error correlation ``rho`` (Mack variance scaling sqrt(C)).

    Generating under Mack's assumptions (C_{d+1} = f_d C_d + sqrt(C_d) eps) is
    what makes the zero-correlation test a real identity check: the volume-
    weighted chain ladder is the efficient estimator for exactly this process,
    so SUR must reproduce it rather than merely come close.
    """
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
    """Build the full-square triangle, fit on the upper-triangle as_of slice.

    The full square is kept in the returned Triangle so ``realized_ultimates``
    can score against the true last column; the model only ever sees cells at or
    before ``cutoff`` (the last diagonal of a square n_w x n_w triangle).
    """
    n_lob, n_w, _ = cum.shape
    lobs = {f"lob_{k}": cum[k] for k in range(n_lob)}
    full = make_multiline_triangle(backend_name, lobs, start_year=START)
    cutoff = dt.date(START + n_w - 1, 12, 31)
    entry = SUR().fit(full, as_of=cutoff, **fit_kwargs)
    return entry, full


def volume_weighted_cl_ultimates(cum, obs_mask):
    """Hand-rolled volume-weighted chain ladder per line on masked data.

    Deliberately independent of both ``ibnr`` and chainladder-python: an
    oracle written from the textbook definition, so the reduction test cannot
    pass by two implementations sharing a bug.
    """
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
    """The entry is discoverable through the public gallery API under the
    ``statistical`` family, and its card names the method it implements."""
    assert "sur" in gallery.list()
    assert gallery.get("sur").family == "statistical"
    assert "Seemingly Unrelated Regression" in gallery.get("sur").card()


def test_sur_reduces_to_chain_ladder_at_zero_correlation(backend_name):
    """The defining sanity check: with independent lines, FGLS gains nothing
    over equation-by-equation OLS, so per-origin ultimates must match the
    volume-weighted chain ladder."""
    rng = np.random.default_rng(42)
    cum = simulate_cl_square(rng, rho=0.0)
    entry, _ = fit_on_upper(backend_name, cum)
    c = entry.contract_
    pred = entry.predict(n_draws=20_000, seed=1)

    cl = volume_weighted_cl_ultimates(c["cum"], c["obs_mask"])
    # first n_lob*n_w targets are the per-(lob, origin) ultimates; the lob
    # subtotals and grand total follow.
    got = pred.mean()[: c["n_lob"] * c["n_w"]].reshape(c["n_lob"], c["n_w"])
    # 2% is Monte-Carlo slack on 20k draws plus the small-sample ladder used on
    # the last transitions; the estimators are equal in expectation, not in
    # closed form on a finite draw set.
    np.testing.assert_allclose(got, cl, rtol=0.02)


def test_sur_recovers_planted_correlation(backend_name):
    """Estimation check: the FGLS residual covariance recovers a planted
    rho=0.7 between lines, and the pooled fallback correlation agrees in sign
    and rough magnitude."""
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
    # Wide band on purpose: each transition estimates a 2x2 covariance from at
    # most ~11 residual pairs, so the sampling error on rho is large. The test
    # pins "clearly positive and not degenerate", not a precise value.
    assert 0.4 < mean_corr < 1.0, f"recovered correlation {mean_corr:.2f} not near 0.7"
    assert entry.pooled_corr_[0, 1] > 0.3


def test_sur_positive_correlation_widens_grand_total(backend_name):
    """The actuarial payoff: correlated lines diversify less, so the grand
    total's SD must exceed the same model with cross-line covariance zeroed."""
    rng = np.random.default_rng(11)
    cum = simulate_cl_square(rng, rho=0.7)
    entry, _ = fit_on_upper(backend_name, cum)

    # Ablate ONLY the off-diagonals of each transition covariance: same point
    # estimates and same marginal variances, so any SD difference on the total
    # is attributable to dependence alone.
    independent = copy.deepcopy(entry)
    for tr in independent.transitions_:
        tr["sigma"] = np.diag(np.diag(tr["sigma"]))

    # param_uncertainty=False isolates process variance; parameter draws would
    # add a shared component that muddies the comparison. Same seed both sides.
    sd_dep = entry.predict(n_draws=20_000, seed=3, param_uncertainty=False).std()[-1]
    sd_ind = independent.predict(n_draws=20_000, seed=3, param_uncertainty=False).std()[-1]
    # 5% margin, not >: keeps the assertion above Monte-Carlo noise on 20k draws.
    assert sd_dep > sd_ind * 1.05


def test_sur_parameter_uncertainty_widens_intervals(backend_name):
    """Estimation error in the development factors is a real source of reserve
    risk: switching it on can only widen the predictive distribution."""
    rng = np.random.default_rng(5)
    cum = simulate_cl_square(rng, rho=0.3)
    entry, _ = fit_on_upper(backend_name, cum)
    sd_with = entry.predict(n_draws=20_000, seed=2, param_uncertainty=True).std()[-1]
    sd_without = entry.predict(n_draws=20_000, seed=2, param_uncertainty=False).std()[-1]
    assert sd_with > sd_without


def test_sur_target_layout_and_realized(backend_name):
    """Pins the PredictiveDistribution contract every consumer relies on: the
    SUR target layout is [per-(lob, origin) cells | per-lob totals | grand
    total], and ``realized_ultimates`` lines up element-for-element with it."""
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

    # Realized ultimates come from the full square's last dev column - the
    # quantity a backtest scores the predictive distribution against.
    realized = entry.realized_ultimates(full)
    assert realized.shape == (pred.n_targets,)
    np.testing.assert_allclose(realized[:n_cells], cum[:, :, -1].reshape(-1))
    np.testing.assert_allclose(realized[-1], cum[:, :, -1].sum())

    table = pred.summary(observed=realized)
    assert {"estimate", "se", "cv", "outcome", "percentile"} <= set(table.columns)


def test_sur_small_sample_ladder_kicks_in(backend_name):
    """Schedule P triangles are tiny, so the deepest transitions cannot support
    a full K x K covariance. This pins the graceful degradation ladder:
    full FGLS -> pooled correlation -> tail (single pair)."""
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
    """The ablatable intercept variant (regression through a free constant
    rather than the origin) fits and yields a 2-parameter beta per line."""
    rng = np.random.default_rng(13)
    cum = simulate_cl_square(rng, rho=0.2)
    entry, _ = fit_on_upper(backend_name, cum, intercept=True)
    pred = entry.predict(n_draws=500, seed=0)
    assert np.isfinite(pred.samples).all()
    assert entry.transitions_[0]["beta"].shape == (2, 2)


def test_sur_rejects_nonpositive_cumulatives(backend_name):
    """Mack's sqrt(C) variance scaling is undefined for non-positive
    cumulatives, so the entry must refuse rather than emit NaNs downstream."""
    cum = np.array([[[100.0, 150.0], [110.0, np.nan]]])
    bad = np.concatenate([cum, -cum])  # second lob negative
    t = make_multiline_triangle(backend_name, {"lob_a": bad[0], "lob_b": bad[1]}, start_year=START)
    with pytest.raises(ValueError, match="non-positive cumulative"):
        SUR().fit(t)


def test_sur_predict_before_fit_raises():
    """GalleryEntry lifecycle: predict() before fit() is a clear RuntimeError,
    not an AttributeError from a missing fitted attribute."""
    with pytest.raises(RuntimeError, match="fit"):
        SUR().predict()
