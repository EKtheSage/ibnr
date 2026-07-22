"""End-to-end smoke test of the clark_growth_curve entry against the gold mart.

The Bayesian twin of the deterministic Clark entry: the same loglogistic growth
curve and ODP likelihood, sampled instead of maximized. Because the MLE version
is separately tied out against chainladder's ``ClarkLDF`` (``test_clark.py``,
fast), the load-bearing check here is that the posterior agrees with that
already-validated MLE — which chains this entry to the external reference
without re-deriving it.

Slow (compiles the Stan model on first run, then samples) and mart-dependent;
runs only when cmdstan and the local warehouse are both available — see
``test_meyers_ccl`` for the ``slow``/``mart`` marker rationale.
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


# Module-scoped: the Stan fit dominates runtime; every test reuses it.
@pytest.fixture(scope="module")
def fitted():
    """Same company / 1997 cutoff / pinned seed as the other Bayesian smoke tests.

    Fitting the entry also fits its MLE twin (``entry.mle_``), which
    ``test_posterior_centers_on_mle`` compares against.
    """
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
    """The sampler converged on the level and the two growth-curve parameters.

    Only 3 free parameters over 55 cells, no hierarchy, so the tighter
    thresholds (R-hat < 1.05, ESS > 200) are appropriate — a low ESS here would
    mean the omega/theta ridge is being sampled badly, which is the known
    failure mode of growth-curve models (the two are strongly correlated).
    """
    _, entry = fitted
    summary = entry.fit_.summary()
    core = summary.loc[summary.index.str.match(r"logelr|omega|theta")]
    assert core["R_hat"].max() < 1.05
    assert (core["ESS_bulk"].dropna() > 200).all()


def test_posterior_centers_on_mle(fitted):
    """Vague-ish priors + 55 cells => posterior near the MLE twin.

    This is the entry's anchor to the external ClarkLDF reference (via the MLE,
    which test_clark.py ties out). 25% is deliberately generous: omega and theta
    trade off along a likelihood ridge, so posterior *means* can drift well away
    from the mode individually while describing near-identical growth curves —
    the parameters are only weakly identified, the curve is not.
    """
    _, entry = fitted
    from ibnr.gallery.bayesian.clark_growth_curve.model import pooled

    mle = entry.mle_.params_
    om = pooled(entry.idata_, "omega").mean()
    th = pooled(entry.idata_, "theta").mean()
    assert om == pytest.approx(mle["omega"], rel=0.25)
    assert th == pytest.approx(mle["theta"], rel=0.25)


def test_predict_and_score(fitted):
    """predict() honors the PredictiveDistribution contract and lands in the
    right ballpark.

    Wiring fence only (units, anchor, prior runaway) — Clark's true calibration
    (D=49.6-63.4* on the 200-company retro, the weakest paid entry) is measured
    by the retrospective, not here.
    """
    tri, entry = fitted
    pred = entry.predict(seed=1)
    assert pred.n_targets == 11  # 10 accident years + total
    assert pred.std()[0] == 0.0  # AY 1988 fully developed

    realized = entry.realized_ultimates(tri)
    assert not np.isnan(realized).any()
    table = pred.summary(observed=realized)
    total = table.iloc[-1]
    assert 0.5 * total["outcome"] < total["estimate"] < 2.0 * total["outcome"]
    assert 0.0 <= total["percentile"] <= 100.0
