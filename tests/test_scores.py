"""kernels.scores: sample-based CRPS against closed forms."""

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
    rng = np.random.default_rng(7)
    mu, sigma = 3.0, 2.0
    samples = rng.normal(mu, sigma, size=(200_000, 3))
    observed = np.array([mu, mu + sigma, mu - 2.5 * sigma])
    got = crps(samples, observed)
    want = np.array([gaussian_crps(y, mu, sigma) for y in observed])
    np.testing.assert_allclose(got, want, rtol=0.01)


def test_crps_degenerate_forecast_is_absolute_error():
    # All draws (nearly) identical: CRPS -> |estimate - outcome|.
    samples = np.full((100, 2), 5.0)
    got = crps(samples, np.array([5.0, 8.0]))
    np.testing.assert_allclose(got, [0.0, 3.0], atol=1e-12)


def test_crps_nan_outcome_propagates():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(1000, 2))
    got = crps(samples, np.array([0.0, np.nan]))
    assert np.isfinite(got[0])
    assert np.isnan(got[1])


def test_crps_shape_validation():
    with pytest.raises(ValueError, match="2-D"):
        crps(np.zeros(10), np.zeros(1))
    with pytest.raises(ValueError, match="shape"):
        crps(np.zeros((10, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="at least 2"):
        crps(np.zeros((1, 3)), np.zeros(3))


def test_crps_rewards_sharpness():
    # Same central coverage, tighter forecast scores better at the center.
    rng = np.random.default_rng(1)
    tight = rng.normal(0.0, 1.0, size=(50_000, 1))
    wide = rng.normal(0.0, 3.0, size=(50_000, 1))
    y = np.array([0.0])
    assert crps(tight, y)[0] < crps(wide, y)[0]
