"""gallery.statistical.mcl: the full-matrix multivariate chain ladder.

Two things are protected here. The estimator follows the R systemfit SUR
conventions (one feasible-GLS step, geomean residual covariance), which a
hand-solved two-line system pins; and which transitions fall back to the
diagonal chain ladder (the system is used only where origin pairs exceed twice
the line count) is what makes the entry the reference implementation's twin
rather than a cousin. The Schedule P tie-out lives in tests/test_mcl_tieout.py.

Data is simulated in-test rather than loaded from the Schedule P mart so these
run everywhere, and every test runs on both ibis backends via ``backend_name``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from ibnr import gallery
from ibnr.gallery.statistical.mcl.model import MCL, system_estimate, volume_weighted

from .conftest import make_multiline_triangle
from .test_sur import simulate_cl_square, volume_weighted_cl_ultimates

START = 2000
#: Development factors long enough to make a simulated square actually square.
#: ``simulate_cl_square`` derives ``n_d`` from the factor count, so a 12-origin
#: square needs 11 of them; the shorter default in test_sur gives 5 development
#: steps whatever the origin count, and then no transition is thin enough to
#: reach the fallback rule this file is about.
SQUARE_FACTORS = np.array([1.5, 1.35, 1.25, 1.18, 1.12, 1.08, 1.06, 1.04, 1.03, 1.02, 1.01])


def upper_triangle(backend_name, cum, start_year=START):
    """The full simulated square as a Triangle, plus the as_of cutoff that
    leaves the model only the upper triangle.

    ``tests/test_sur.py::fit_on_upper`` does the same but fits a SUR entry on
    the way, which this file has no use for.
    """
    n_lob, n_w, _ = cum.shape
    lobs = {f"lob_{k}": cum[k] for k in range(n_lob)}
    full = make_multiline_triangle(backend_name, lobs, start_year=start_year)
    return full, dt.date(start_year + n_w - 1, 12, 31)


def test_mcl_registered():
    """Discoverable through the public gallery API, in the statistical family,
    with a card that names the method."""
    assert "mcl" in gallery.list()
    assert gallery.get("mcl").family == "statistical"
    assert "multivariate chain ladder" in gallery.get("mcl").card()


def test_system_estimate_matches_a_hand_solved_two_line_system():
    """K = 2, n = 6: OLS per equation, geomean residual covariance, one GLS step.

    The reference values are computed here with dense numpy on the stacked
    system, which is a different code path from the block assembly the
    estimator uses.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(100.0, 200.0, size=(2, 6))
    y = np.stack([1.4 * x[0] + 0.1 * x[1], 0.2 * x[0] + 1.2 * x[1]]) + rng.normal(0, 3.0, (2, 6))
    B, sigma, coef_cov = system_estimate(x, y)
    # dense reference: stacked whitened system
    root = np.sqrt(x)
    X = [np.stack([x[i] / root[k] for i in range(2)], axis=1) for k in range(2)]  # (n, K) each
    Y = [y[k] / root[k] for k in range(2)]
    beta_ols = [np.linalg.lstsq(X[k], Y[k], rcond=None)[0] for k in range(2)]
    e = np.stack([Y[k] - X[k] @ beta_ols[k] for k in range(2)])
    S = e @ e.T / (6 - 2)
    Sinv = np.linalg.inv(S)
    Xs = np.zeros((12, 4))
    Xs[:6, :2], Xs[6:, 2:] = X[0], X[1]
    Ys = np.concatenate(Y)
    omega_inv = np.kron(Sinv, np.eye(6))
    A = Xs.T @ omega_inv @ Xs
    beta_gls = np.linalg.solve(A, Xs.T @ omega_inv @ Ys)
    np.testing.assert_allclose(B.reshape(-1), beta_gls, rtol=1e-10)
    np.testing.assert_allclose(coef_cov, np.linalg.inv(A), rtol=1e-8)
    assert sigma.shape == (2, 2)
    # the GLS step moved the answer: otherwise this test would pass on an
    # implementation that stopped after the per-equation OLS
    assert not np.allclose(B.reshape(-1), np.concatenate(beta_ols), rtol=1e-6)


def test_volume_weighted_is_the_diagonal_of_column_totals():
    """The fallback is the per-line volume-weighted chain ladder and nothing
    else: column total over column total, zero off the diagonal."""
    x = np.array([[100.0, 200.0, 300.0], [50.0, 60.0, 70.0]])
    y = np.array([[150.0, 260.0, 430.0], [55.0, 72.0, 77.0]])
    B = volume_weighted(x, y)
    np.testing.assert_allclose(np.diag(B), [840.0 / 600.0, 204.0 / 180.0])
    assert np.count_nonzero(B - np.diag(np.diag(B))) == 0


def test_zero_cross_terms_collapse_to_the_volume_weighted_chain_ladder(backend_name):
    """Lines that develop independently give exactly the volume-weighted
    factors on the fallback transitions, and a projection close to the per-line
    chain ladder everywhere."""
    rng = np.random.default_rng(1)
    cum = simulate_cl_square(rng, n_w=12, rho=0.0, factors=SQUARE_FACTORS)
    tri, cutoff = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri, as_of=cutoff)
    methods = [t["method"] for t in entry.transitions_]
    # 12 origins, K = 2: transition d has 11 - d pairs; the system needs > 4
    assert methods == ["system"] * 7 + ["volume_weighted"] * 4
    c = entry.contract_
    for d, t in enumerate(entry.transitions_[7:], start=7):
        assert np.count_nonzero(t["B"] - np.diag(np.diag(t["B"]))) == 0
        pair = c["obs_mask"][0, :, d] & c["obs_mask"][0, :, d + 1]
        expected = c["cum"][:, pair, d + 1].sum(axis=1) / c["cum"][:, pair, d].sum(axis=1)
        np.testing.assert_allclose(np.diag(t["B"]), expected, rtol=1e-12)
    # the whole projection stays within a few percent of the per-line chain
    # ladder, computed by the oracle test_sur.py writes from the textbook
    # definition rather than by another ibnr estimator
    point = entry.point()
    oracle = volume_weighted_cl_ultimates(c["cum"], c["obs_mask"])
    assert point["point"].iloc[-1] == pytest.approx(float(oracle.sum()), rel=0.02)


def test_fallback_rule_by_transition_and_min_obs_mult(backend_name):
    """``min_obs_mult`` is the only knob on the rule, and it moves which
    transitions are systems - nothing else."""
    rng = np.random.default_rng(2)
    cum = simulate_cl_square(rng, n_w=8, rho=0.3, factors=SQUARE_FACTORS[:7])
    tri, cutoff = upper_triangle(backend_name, cum)
    default = MCL().fit(tri, as_of=cutoff)
    # 8 origins: pairs 7, 6, 5, 4, 3, 2, 1 -> the system needs > 4 -> first three
    assert [t["method"] for t in default.transitions_] == ["system"] * 3 + ["volume_weighted"] * 4
    strict = MCL().fit(tri, as_of=cutoff, min_obs_mult=3)
    assert [t["method"] for t in strict.transitions_] == ["system"] + ["volume_weighted"] * 6
    assert [t["n"] for t in default.transitions_] == [7, 6, 5, 4, 3, 2, 1]
    # exactly 2 * K pairs is NOT enough: the rule is strict
    boundary = MCL().fit(tri, as_of=cutoff, min_obs_mult=2)
    assert boundary.transitions_[3]["n"] == 4
    assert boundary.transitions_[3]["method"] == "volume_weighted"
    assert boundary.transitions_[2]["n"] == 5
    assert boundary.transitions_[2]["method"] == "system"


def test_point_layout_and_recursion(backend_name):
    """``point()`` is the vector recursion in the shared multiline layout."""
    rng = np.random.default_rng(3)
    cum = simulate_cl_square(rng, n_w=6, rho=0.2)
    tri, cutoff = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri, as_of=cutoff)
    frame = entry.point()
    labels = frame["label"].astype(str).tolist()
    assert labels[-1] == "total" and labels[-3:-1] == ["lob_0/total", "lob_1/total"]
    assert len(frame) == 2 * 6 + 2 + 1
    # the grand total is the sum of the per-lob totals, which are sums over origins
    assert frame["point"].iloc[-1] == pytest.approx(frame["point"].iloc[-3:-1].sum())
    # hand recursion for the last origin: its latest dev is 0, so every
    # transition applies, and the matrix product mixes the lines
    state = entry.contract_["cum"][:, -1, 0]
    for t in entry.transitions_:
        state = t["B"] @ state
    np.testing.assert_allclose(frame["point"].to_numpy()[[5, 11]], state, rtol=1e-10)


def test_point_carries_a_fully_developed_origin_through_unchanged(backend_name):
    """The oldest origin observes every development step, so its ultimate is
    its own latest cumulative and no factor touches it."""
    rng = np.random.default_rng(8)
    cum = simulate_cl_square(rng, n_w=6, rho=0.0)
    tri, cutoff = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri, as_of=cutoff)
    frame = entry.point()
    np.testing.assert_allclose(frame["point"].to_numpy()[[0, 6]], cum[:, 0, -1], rtol=1e-12)


def test_refusals(backend_name):
    """Both refusals name what is wrong: a cumulative the whitening cannot
    divide by, and a rule setting that would ask K columns of K observations."""
    rng = np.random.default_rng(4)
    cum = simulate_cl_square(rng, n_w=6)
    cum[0, 2, 1] = 0.0  # a zero cumulative with an observed successor
    tri, cutoff = upper_triangle(backend_name, cum)
    with pytest.raises(ValueError, match="non-positive cumulative"):
        MCL().fit(tri, as_of=cutoff)
    clean, cutoff = upper_triangle(backend_name, simulate_cl_square(rng, n_w=6))
    with pytest.raises(ValueError, match="min_obs_mult"):
        MCL().fit(clean, as_of=cutoff, min_obs_mult=0)


def test_a_non_positive_latest_diagonal_cell_is_accepted(backend_name):
    """A cell with no observed successor is never divided by, so it is not
    refused - it only starts that origin's recursion.

    This is the boundary the guard above sits next to, and it is load-bearing
    rather than pedantic: two of the 82 Schedule P companies the entry ties out
    against carry a zero paid cumulative at 12 months on their newest accident
    year, and the reference R implementation projects them exactly this way.
    The other lines still push the cell forward, because the recursion is a
    matrix product rather than a per-line factor.
    """
    rng = np.random.default_rng(12)
    cum = simulate_cl_square(rng, n_w=6, rho=0.5)
    cum[0, -1, 0] = 0.0  # newest origin, first dev step: on the latest diagonal
    tri, cutoff = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri, as_of=cutoff)
    frame = entry.point()
    assert np.isfinite(frame["point"]).all()
    assert entry.contract_["cum"][0, -1, 0] == 0.0
    # line 0's newest origin is carried by line 1 alone through the first
    # transition, so its ultimate is not zero
    assert frame["point"].iloc[5] != 0.0


def test_failed_positivity_guard_leaves_the_previous_fit_intact(backend_name):
    """fit() is atomic: a refusal must leave the earlier fit untouched, not a
    new contract over old transitions."""
    rng = np.random.default_rng(5)
    cum = simulate_cl_square(rng, n_w=6, rho=0.2)
    tri_a, cutoff_a = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri_a, as_of=cutoff_a)
    before = dict(vars(entry))

    bad = cum.copy()
    bad[0, 0, 0] = 0.0
    tri_b, cutoff_b = upper_triangle(backend_name, bad, start_year=1990)
    with pytest.raises(ValueError, match="non-positive cumulative"):
        entry.fit(tri_b, as_of=cutoff_b)

    after = vars(entry)
    assert set(after) == set(before)
    assert not [k for k in before if after[k] is not before[k]]


def test_predict_layout_mean_and_determinism(backend_name):
    """The predictive draws share point()'s layout, are centred on it when only
    process risk is switched on, and repeat exactly for one seed."""
    rng = np.random.default_rng(5)
    cum = simulate_cl_square(rng, n_w=8, rho=0.3, factors=SQUARE_FACTORS[:7])
    tri, cutoff = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri, as_of=cutoff)
    # both draw paths are exercised: three system transitions, four fallbacks
    assert {t["method"] for t in entry.transitions_} == {"system", "volume_weighted"}
    pred = entry.predict(n_draws=4000, seed=9)
    assert pred.samples.shape == (4000, 2 * 8 + 2 + 1)
    assert pred.targets["label"].astype(str).tolist() == entry.point()["label"].astype(str).tolist()
    assert np.isfinite(pred.samples).all()
    # process-only draws are centred on the point recursion
    only_process = entry.predict(n_draws=4000, seed=9, param_uncertainty=False)
    np.testing.assert_allclose(only_process.mean()[-1], entry.point()["point"].iloc[-1], rtol=0.02)
    again = entry.predict(n_draws=4000, seed=9)
    np.testing.assert_array_equal(pred.samples, again.samples)
    assert not np.array_equal(pred.samples, entry.predict(n_draws=4000, seed=10).samples)
    # the grand total is the within-draw sum of the per-lob totals
    np.testing.assert_allclose(pred.samples[:, -1], pred.samples[:, -3:-1].sum(axis=1))


def test_parameter_risk_widens_the_total(backend_name):
    """Estimation error in the coefficient matrices is real reserve risk, so
    switching it on can only widen the predictive distribution."""
    rng = np.random.default_rng(7)
    cum = simulate_cl_square(rng, n_w=8, rho=0.3, factors=SQUARE_FACTORS[:7])
    tri, cutoff = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri, as_of=cutoff)
    with_param = entry.predict(n_draws=8000, seed=2, param_uncertainty=True).std()[-1]
    without = entry.predict(n_draws=8000, seed=2, param_uncertainty=False).std()[-1]
    assert with_param > without


def test_correlated_process_noise_reaches_the_grand_total(backend_name):
    """The off-diagonal of each transition's residual covariance has to reach
    the draws, and the grand total is where that shows.

    Summing the per-lob totals draw by draw is not enough to see it: the grand
    total is their sum whether or not the two lines move together. What
    distinguishes the two is the total's VARIANCE against the sum of the lines'
    own variances, which are equal under independence and further apart the
    stronger the dependence. At a planted correlation of 0.9 the gap is large,
    so this fails outright on draws that use only the diagonal of sigma.
    """
    rng = np.random.default_rng(6)
    cum = simulate_cl_square(rng, n_w=8, rho=0.9, factors=SQUARE_FACTORS[:7])
    tri, cutoff = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri, as_of=cutoff)
    samples = entry.predict(n_draws=20_000, seed=4, param_uncertainty=False).samples
    total_var = samples[:, -1].var()
    independent_var = samples[:, -3:-1].var(axis=0).sum()
    assert total_var > 1.2 * independent_var


def test_process_noise_scales_with_the_cumulative_it_sits_on(backend_name):
    """Multiplying every loss amount by four must multiply every draw by four.

    That is what Mack's variance assumption says, and it is the one thing a
    test of the layout or of the total cannot see. The coefficients are
    dimensionless and do not move; the whitened residual covariance scales by
    16, so its Cholesky factor scales by 4; and the noise is un-whitened by
    ``sqrt(C)``, which scales by 2 - so the noise added to a cumulative four
    times as large is four times as large, not twice. Dropping the ``sqrt(C)``
    factor leaves the draws scaling by two, which this catches and which a
    standard deviation read on one triangle alone cannot.
    """
    rng = np.random.default_rng(15)
    cum = simulate_cl_square(rng, n_w=8, rho=0.3, factors=SQUARE_FACTORS[:7])
    tri, cutoff = upper_triangle(backend_name, cum)
    big, _ = upper_triangle(backend_name, 4.0 * cum)
    base = MCL().fit(tri, as_of=cutoff).predict(n_draws=2000, seed=3, param_uncertainty=False)
    scaled = MCL().fit(big, as_of=cutoff).predict(n_draws=2000, seed=3, param_uncertainty=False)
    np.testing.assert_allclose(scaled.samples, 4.0 * base.samples, rtol=1e-8)


def test_realized_ultimates_align_to_predict(backend_name):
    """Outcomes line up element for element with the predictive targets, and
    the default scoring runs over them."""
    rng = np.random.default_rng(6)
    cum = simulate_cl_square(rng, n_w=6, rho=0.1)
    tri, cutoff = upper_triangle(backend_name, cum)
    entry = MCL().fit(tri, as_of=cutoff)
    realized = entry.realized_ultimates(tri)
    assert realized.shape == (2 * 6 + 2 + 1,)
    np.testing.assert_allclose(realized[:12].reshape(2, 6), cum[:, :, -1])
    np.testing.assert_allclose(realized[-1], cum[:, :, -1].sum())
    scored = entry.evaluate(realized)
    assert set(scored) == {"summary", "percentiles", "crps"}
    assert np.isfinite(scored["crps"]).all()
