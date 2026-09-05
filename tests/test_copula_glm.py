"""gallery.statistical.copula_glm: copula-linked lognormal regressions
(Shi & Frees 2011).

What this file protects: the second frequentist dependence baseline. Lognormal
marginals on *incremental* loss ratios per line, tied together by a Gaussian
copula on the cell-level residuals, with a parametric bootstrap for parameter
uncertainty. The invariants that matter are recovery of the planted copula
correlation and marginal scale, and that dependence and bootstrap each widen the
diversified total (a dependence model that doesn't is worthless).

Tests simulate FROM the model's own generative process, so a failure is an
estimation/plumbing bug rather than a model-misspecification artifact. Data is
synthetic and in-repo - the package must be testable without the Schedule P
mart (mart-backed smokes live in ``test_statistical_mart.py``).

Note the headline-field constraint from CLAUDE.md: lognormal marginals cannot
take negative increments, hence the explicit non-positive guard tests below.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from ibnr import gallery
from ibnr.gallery.statistical.copula_glm.model import CopulaGLM

from .conftest import make_multiline_triangle

START = 2000
# Per-line lognormal residual SDs on the log scale; unequal so the fit has to
# recover two distinct marginal scales, not one shared one.
SIGMAS = np.array([0.10, 0.15])
# Decaying incremental paid loss ratios: 45% of premium in dev 1 down to 5% in
# dev 5 - a plausible short-tail paid pattern.
DEV_LEVEL = np.log(np.array([0.45, 0.25, 0.15, 0.08, 0.05]))  # incremental LR by dev


def simulate_lognormal_square(rng, *, n_w=8, rho=0.6, sigmas=SIGMAS, premium=1000.0):
    """Full (K, n_w, n_d) cumulative square simulated FROM the copula model.

    Increment (k, w, d) = premium * exp(dev_d + alpha_w + sigma_k z_k), with z
    drawn from a Gaussian copula of correlation ``rho`` shared across lines
    within a cell. Cumulating is the last step, so dependence is planted where
    the model looks for it: on incremental residuals.
    """
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
    """Build the full-square triangle (premium included - the marginals are
    loss *ratios*, so exposure is required) and fit on the upper-triangle
    ``as_of`` slice; the square is retained for realized-ultimate scoring."""
    n_lob, n_w, _ = cum.shape
    lobs = {f"lob_{k}": cum[k] for k in range(n_lob)}
    prem = {f"lob_{k}": np.full(n_w, premium) for k in range(n_lob)}
    full = make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)
    cutoff = dt.date(START + n_w - 1, 12, 31)
    entry = CopulaGLM().fit(full, as_of=cutoff, **fit_kwargs)
    return entry, full


def test_copula_registered():
    """Discoverable via the public gallery API in the ``statistical`` family,
    with a card that credits the source paper."""
    assert "copula_glm" in gallery.list()
    assert gallery.get("copula_glm").family == "statistical"
    assert "Shi & Frees" in gallery.get("copula_glm").card()


def test_copula_recovers_planted_correlation(backend_name):
    """Estimation check: the Gaussian copula correlation is recovered from a
    simulation with a known rho."""
    rng = np.random.default_rng(21)
    cum = simulate_lognormal_square(rng, rho=0.6)
    entry, _ = fit_on_upper(backend_name, cum)
    # +/-0.25 is loose because the correlation is estimated from ~36 upper-
    # triangle residual pairs after the marginal regressions consume df; the
    # test pins "recovered, strongly positive", not a precise point estimate.
    assert abs(entry.corr_[0, 1] - 0.6) < 0.25, f"corr {entry.corr_[0, 1]:.2f} not near 0.6"


def test_copula_recovers_marginal_scale(backend_name):
    """Both first and second moments are right: the grand-total point estimate
    lands near the true ultimate and the fitted marginal sigmas recover the
    planted ones."""
    rng = np.random.default_rng(22)
    cum = simulate_lognormal_square(rng, rho=0.3)
    entry, full = fit_on_upper(backend_name, cum)
    pred = entry.predict(n_draws=4000, seed=0, param_uncertainty="plugin")
    realized = entry.realized_ultimates(full)
    # point estimates should land within ~10% of the true ultimates
    estimate = pred.mean()[-1]
    np.testing.assert_allclose(estimate, realized[-1], rtol=0.10)
    # sigma is estimated from few residuals and is biased low by the fitted
    # effects, so the ratio band is wide and asymmetric-tolerant by design.
    assert 0.6 < np.mean(entry.sigma_ / SIGMAS) < 1.5


def test_copula_nonpositive_error_and_drop(backend_name):
    """A negative paid increment has no lognormal density, so the default is a
    hard error; ``nonpositive="drop"`` is the documented escape hatch and must
    still produce finite draws."""
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
    """The parametric bootstrap adds parameter uncertainty on top of process
    variance, so it must be strictly wider than the plug-in predictive."""
    rng = np.random.default_rng(24)
    cum = simulate_lognormal_square(rng, rho=0.4)
    entry, _ = fit_on_upper(backend_name, cum)
    sd_boot = entry.predict(n_draws=8000, seed=1, param_uncertainty="bootstrap").std()[-1]
    sd_plug = entry.predict(n_draws=8000, seed=1, param_uncertainty="plugin").std()[-1]
    assert sd_boot > sd_plug


def test_copula_positive_correlation_widens_grand_total(backend_name):
    """Same diversification invariant as SUR: cross-line dependence must show
    up as extra spread on the grand total, not just in a reported statistic."""
    rng = np.random.default_rng(25)
    cum = simulate_lognormal_square(rng, rho=0.7)
    entry, _ = fit_on_upper(backend_name, cum)
    # Clone the fitted entry and swap ONLY the copula correlation for identity:
    # marginals are untouched, so the SD gap isolates dependence.
    independent = CopulaGLM()
    independent.__dict__.update({**entry.__dict__, "corr_": np.eye(2)})
    sd_dep = entry.predict(n_draws=8000, seed=2, param_uncertainty="plugin").std()[-1]
    sd_ind = independent.predict(n_draws=8000, seed=2, param_uncertainty="plugin").std()[-1]
    # Only 2% margin (vs SUR's 5%): the same seed drives both draw sets, so the
    # comparison is near-paired and needs less Monte-Carlo headroom.
    assert sd_dep > sd_ind * 1.02


def test_copula_target_layout(backend_name):
    """The entry emits the same SUR-style target layout (cells | lob totals |
    grand total) so the two dependence baselines are directly comparable, and
    an already-developed origin is a degenerate (constant) target."""
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
    """Ragged real-world triangles can leave the free dev-factor
    parameterization saturated. The guard must refuse rather than fit a
    degenerate model, and name the parsimonious Hoerl curve as the way out."""
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


def zeroed_square(rng, *, n_w, n_d, zero_cells, premium=1000.0):
    """Cumulative square whose named cells carry a ZERO increment in lob_0.

    Zero is non-positive, so ``nonpositive="drop"`` removes those cells in every
    line (the mask stays aligned across lines). That is how a caller reaches a
    usable-cell pattern the counting checks accept and the design cannot
    support: cells are excluded one at a time, with nothing looking at the shape
    they leave behind.
    """
    levels = np.log(np.linspace(0.45, 0.02, n_d))  # decaying incremental loss ratios
    incr = premium * np.exp(levels + rng.normal(0.0, 0.05, size=(2, n_w, n_d)))
    for w, d in zero_cells:
        incr[0, w, d] = 0.0
    return np.cumsum(incr, axis=2)


def test_copula_drop_refuses_disconnected_design(backend_name):
    """Dropping every cell of one origin except its last splits the usable cells
    into two groups that share no origin and no dev step, so the level of one
    group can be traded against the other with every fitted training mean
    unchanged. The counting checks all pass: 29 usable cells against 15 columns,
    every origin covered, every dev step covered. Only the rank sees it, at 14
    of 15.

    Before this check the fit returned, and a whole direction of the parameter
    space was decided by the pseudo-inverse's minimum-norm tie-break rather than
    by the data; ``predict()`` then reported one plausible answer out of an
    unbounded family.
    """
    rng = np.random.default_rng(31)
    n_w = n_d = 8
    # origin 1 (index 0) keeps only its dev-8 cell, and dev 8 is reached by no
    # other origin, so that cell is an island.
    zero_cells = [(0, d) for d in range(n_d - 1)]
    cum = zeroed_square(rng, n_w=n_w, n_d=n_d, zero_cells=zero_cells)
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    t = make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)
    cutoff = dt.date(START + n_w - 1, 12, 31)

    entry = CopulaGLM()
    with pytest.raises(ValueError, match="rank 14 of 15") as raised:
        entry.fit(t, as_of=cutoff, nonpositive="drop")
    assert entry.contract_ is None  # fit() is atomic
    message = str(raised.value)
    # The columns that trade off against each other are named, and ONLY those:
    # the isolated cell's dev step is in, the dev steps the rest of the triangle
    # pins down are out. Without the second half the message would pass while
    # reading a direction the data CAN see (a row-space vector names nearly
    # every column, dev[8] included).
    assert "dev[8]" in message
    assert "dev[2]" not in message
    # keeps the pointer scripts/compare_gallery.py keys its Hoerl retry on
    assert "hoerl" in message


def test_copula_drop_refuses_collinear_hoerl_design(backend_name):
    """The Hoerl curve is the documented escape from a saturated factor design,
    but it has its own failure: ``ln(dev)`` and ``dev`` are two different shapes
    only when the usable cells span three or more development steps. Drop
    everything past dev 2 and the two columns fall on one straight line through
    the intercept, so no data separates them.

    There is no coarser marginal below Hoerl, so the message says that instead
    of pointing at one.
    """
    rng = np.random.default_rng(32)
    n_w = n_d = 10
    # every origin keeps devs 1 and 2 only
    zero_cells = [(w, d) for w in range(n_w) for d in range(2, n_d - w)]
    cum = zeroed_square(rng, n_w=n_w, n_d=n_d, zero_cells=zero_cells)
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    t = make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)

    entry = CopulaGLM()
    with pytest.raises(ValueError, match=r"ln\(dev\)") as raised:
        entry.fit(t, as_of=dt.date(START + n_w - 1, 12, 31), dev_effect="hoerl", nonpositive="drop")
    assert entry.contract_ is None  # fit() is atomic
    message = str(raised.value)
    assert "rank 11 of 12" in message
    # The origin levels are perfectly well identified here, so naming one would
    # mean the message is reading a direction the data can see.
    assert "origin[" not in message


def test_copula_predict_before_fit_raises():
    """GalleryEntry lifecycle: predict() before fit() raises a clear
    RuntimeError rather than an AttributeError."""
    with pytest.raises(RuntimeError, match="fit"):
        CopulaGLM().predict()


def test_copula_rejects_bad_options(backend_name):
    """Unsupported option strings fail loudly at the call site instead of
    silently falling back to a default the user did not ask for."""
    rng = np.random.default_rng(28)
    cum = simulate_lognormal_square(rng)
    with pytest.raises(ValueError, match="dev_effect"):
        fit_on_upper(backend_name, cum, dev_effect="spline")
    entry, _ = fit_on_upper(backend_name, cum)
    with pytest.raises(ValueError, match="param_uncertainty"):
        entry.predict(param_uncertainty="exact")
