import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.calibration import ks_uniformity, pp_points
from ibnr.kernels.predictive import PredictiveDistribution


@pytest.fixture
def pred():
    rng = np.random.default_rng(7)
    samples = np.column_stack([rng.normal(100, 10, 20_000), rng.normal(200, 20, 20_000)])
    targets = pd.DataFrame({"label": ["a", "b"]})
    return PredictiveDistribution(samples=samples, targets=targets)


def test_moments_and_quantiles(pred):
    np.testing.assert_allclose(pred.mean(), [100, 200], rtol=0.01)
    np.testing.assert_allclose(pred.std(), [10, 20], rtol=0.05)
    q = pred.quantile([0.025, 0.975])
    assert q.shape == (2, 2)
    np.testing.assert_allclose(q[:, 0], [100 - 1.96 * 10, 100 + 1.96 * 10], rtol=0.02)


def test_cdf_is_pit(pred):
    pits = pred.cdf([100.0, 240.0])
    assert pits[0] == pytest.approx(0.5, abs=0.02)
    assert pits[1] == pytest.approx(0.977, abs=0.01)
    with pytest.raises(ValueError, match="shape"):
        pred.cdf([1.0])


def test_summary_table(pred):
    table = pred.summary(observed=[110.0, 180.0])
    assert list(table["label"]) == ["a", "b"]
    assert table["estimate"].iloc[0] == pytest.approx(100, rel=0.01)
    assert table["cv"].iloc[1] == pytest.approx(0.1, rel=0.06)
    assert table["percentile"].iloc[0] == pytest.approx(84.1, abs=2)


def test_with_total(pred):
    tot = pred.with_total()
    assert tot.n_targets == 3
    assert tot.targets["label"].tolist() == ["a", "b", "total"]
    np.testing.assert_allclose(tot.samples[:, 2], pred.samples.sum(axis=1))


def test_shape_validation():
    with pytest.raises(ValueError, match="2-D"):
        PredictiveDistribution(samples=np.zeros(5), targets=pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="target rows"):
        PredictiveDistribution(samples=np.zeros((5, 2)), targets=pd.DataFrame({"x": [1]}))


def test_ks_uniformity_accepts_uniform():
    rng = np.random.default_rng(11)
    res = ks_uniformity(rng.uniform(size=200))
    assert res.critical_value_5pct == pytest.approx(1.36 / np.sqrt(200))
    assert not res.reject_5pct
    assert res.p_value > 0.05


def test_ks_uniformity_rejects_concentrated():
    rng = np.random.default_rng(11)
    res = ks_uniformity(np.clip(rng.normal(0.5, 0.08, 200), 0, 1))
    assert res.reject_5pct
    assert res.p_value < 0.01
    assert "*" in repr(res)


def test_meyers_critical_values():
    # the monograph's p-p plot critical values: 19.2 (n=50), 9.6 (n=200)
    assert ks_uniformity(np.linspace(0.01, 0.99, 50)).critical_value_5pct * 100 == pytest.approx(
        19.2, abs=0.05
    )
    assert ks_uniformity(np.linspace(0.01, 0.99, 200)).critical_value_5pct * 100 == pytest.approx(
        9.6, abs=0.05
    )


def test_pp_points_sorted():
    expected, predicted = pp_points([0.9, 0.1, 0.5])
    assert predicted.tolist() == [10.0, 50.0, 90.0]
    assert expected.tolist() == [25.0, 50.0, 75.0]
