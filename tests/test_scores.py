"""kernels.scores: sample-based CRPS against closed forms.

CRPS is the headline proper score on the gallery leaderboard - it rewards
sharpness subject to calibration, so a model cannot win by widening its
intervals. It is computed once here from raw draws (every entry produces a
``PredictiveDistribution``, never a parametric family), which means the estimator
itself needs pinning down: this file checks it against the analytic Gaussian
CRPS, against the degenerate limit where CRPS collapses to absolute error, and on
the ordering property the leaderboard actually depends on.

NaN propagation matters operationally: retrospectives score origins whose outcome
has not emerged, and those must come back NaN to be dropped rather than silently
scored as zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibnr.kernels.scores import crps


def gaussian_crps(y: float, mu: float, sigma: float) -> float:
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting & Raftery 2007)."""
    from scipy.stats import norm

    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_crps_matches_gaussian_closed_form():
    """The sample estimator converges to the analytic CRPS - correctness check on
    the estimator itself, at the center, at +1 sigma and far out in the tail."""
    rng = np.random.default_rng(7)
    mu, sigma = 3.0, 2.0
    # 200k draws: the sample CRPS estimator converges at O(1/n), so this buys well
    # under the 1% rtol below even at the -2.5 sigma outcome where draws are sparse
    samples = rng.normal(mu, sigma, size=(200_000, 3))
    observed = np.array([mu, mu + sigma, mu - 2.5 * sigma])
    got = crps(samples, observed)
    want = np.array([gaussian_crps(y, mu, sigma) for y in observed])
    np.testing.assert_allclose(got, want, rtol=0.01)


def test_crps_degenerate_forecast_is_absolute_error():
    """A point forecast dressed as draws scores as plain absolute error. This is
    what makes CRPS a fair common currency between the distributional entries and
    a deterministic chain-ladder point benchmark."""
    # All draws (nearly) identical: CRPS -> |estimate - outcome|.
    samples = np.full((100, 2), 5.0)
    got = crps(samples, np.array([5.0, 8.0]))
    np.testing.assert_allclose(got, [0.0, 3.0], atol=1e-12)


def test_crps_nan_outcome_propagates():
    """An unemerged outcome yields NaN for that target only, leaving its neighbours
    scored - retrospectives rely on this to drop unscoreable origins rather than
    treat them as perfectly predicted."""
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(1000, 2))
    got = crps(samples, np.array([0.0, np.nan]))
    assert np.isfinite(got[0])
    assert np.isnan(got[1])


def test_crps_shape_validation():
    """Guards on the (draws, targets) layout: 2-D required, one outcome per target,
    and at least 2 draws (the spread term is undefined for a single draw)."""
    with pytest.raises(ValueError, match="2-D"):
        crps(np.zeros(10), np.zeros(1))
    with pytest.raises(ValueError, match="shape"):
        crps(np.zeros((10, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="at least 2"):
        crps(np.zeros((1, 3)), np.zeros(3))


def test_crps_rewards_sharpness():
    """The ordering property the leaderboard depends on: with both forecasts
    centered on the outcome, the sharper one wins. Without this a model could
    always improve its score by inflating uncertainty."""
    # Same central coverage, tighter forecast scores better at the center.
    rng = np.random.default_rng(1)
    tight = rng.normal(0.0, 1.0, size=(50_000, 1))
    wide = rng.normal(0.0, 3.0, size=(50_000, 1))
    y = np.array([0.0])
    assert crps(tight, y)[0] < crps(wide, y)[0]
