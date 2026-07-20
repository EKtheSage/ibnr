"""End-to-end smoke test of the clark_growth_curve entry against the gold mart.

Slow (compiles the Stan model on first run, then samples) and mart-dependent;
runs only when cmdstan and the local warehouse are both available.
"""

import numpy as np
import pytest

from .test_meyers_ccl import _cmdstan_ready
from .test_schedule_p import MART_AVAILABLE, WAREHOUSE

pytestmark = [
    pytest.mark.slow,
    pytest.mark.mart,
    pytest.mark.skipif(
        not MART_AVAILABLE or not _cmdstan_ready(),
        reason="needs the Schedule P gold mart and a cmdstan installation",
    ),
]


@pytest.fixture(scope="module")
def fitted():
    from ibnr import gallery
    from ibnr.data.schedule_p import load_schedule_p

    tri = load_schedule_p(WAREHOUSE, lines=["workers_compensation"], companies=["11347"])
    entry = gallery.fit(
        "clark_growth_curve",
        tri,
        as_of="1997-12-31",
        chains=2,
        iter_warmup=500,
        iter_sampling=1000,
        seed=20260612,
    )
    return tri, entry


def test_fit_converges(fitted):
    _, entry = fitted
    summary = entry.fit_.summary()
    core = summary.loc[summary.index.str.match(r"logelr|omega|theta")]
    assert core["R_hat"].max() < 1.05
    assert (core["ESS_bulk"].dropna() > 200).all()


def test_posterior_centers_on_mle(fitted):
    """Vague-ish priors + 55 cells => posterior near the MLE twin."""
    _, entry = fitted
    from ibnr.gallery.bayesian.clark_growth_curve.model import pooled

    mle = entry.mle_.params_
    om = pooled(entry.idata_, "omega").mean()
    th = pooled(entry.idata_, "theta").mean()
    assert om == pytest.approx(mle["omega"], rel=0.25)
    assert th == pytest.approx(mle["theta"], rel=0.25)


def test_predict_and_score(fitted):
    tri, entry = fitted
    pred = entry.predict(seed=1)
    assert pred.n_targets == 11
    assert pred.std()[0] == 0.0  # AY 1988 fully developed

    realized = entry.realized_ultimates(tri)
    assert not np.isnan(realized).any()
    table = pred.summary(observed=realized)
    total = table.iloc[-1]
    assert 0.5 * total["outcome"] < total["estimate"] < 2.0 * total["outcome"]
    assert 0.0 <= total["percentile"] <= 100.0
