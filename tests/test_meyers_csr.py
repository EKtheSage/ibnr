"""End-to-end smoke test of the meyers_csr entry against the gold mart.

Same shape as ``test_meyers_ccl.py`` (mart → as_of slice → Stan fit →
PredictiveDistribution → score), plus one CSR-specific structural check:
the ``speedup`` term. CSR (Changing Settlement Rate) replaces CCL's AY
correlation with a claim-settlement-speed trend, so ``beta[d]`` is scaled by
``speedup[w] = (1 - gamma)^(w-1)`` — accident years settle progressively
faster (gamma > 0) or slower (gamma < 0) than the first.

Slow (compiles the Stan model on first run, then samples) and mart-dependent;
runs only when cmdstan and the local warehouse are both available. See
``test_meyers_ccl`` for why cmdstan work sits behind the ``slow`` marker;
``mart`` auto-skips when the local Schedule P gold mart is missing.
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


# Module-scoped: one sample serves every test here — the fit is the slow part.
@pytest.fixture(scope="module")
def fitted():
    """Fit CSR on the Meyers setup: 1988-1997 accident years, 1997 diagonal.

    Same company, cutoff, and pinned seed as the CCL smoke test so the two
    entries are directly comparable; chains/iterations are trimmed from the
    production retro settings to keep runtime tolerable.
    """
    from ibnr import gallery
    from ibnr.data.schedule_p import load_schedule_p

    # group 11347: top of the mechanical selection for WC, and in Meyers' Table A.2
    tri = load_schedule_p(WAREHOUSE, lines=["workers_compensation"], companies=["11347"])
    entry = gallery.fit(
        "meyers_csr",
        tri,
        as_of="1997-12-31",
        chains=2,
        iter_warmup=500,
        iter_sampling=1000,
        seed=20260612,
    )
    return tri, entry


def test_fit_converges(fitted):
    """The sampler converged on CSR's core parameters.

    Same conventions as the CCL test: restrict to the model's own parameters
    (level, settlement-rate trend gamma, AY/dev effects, dev sigmas), R-hat
    < 1.1 and ESS > 100 loosened for a 2-chain short run.
    """
    _, entry = fitted
    summary = entry.fit_.summary()
    core = summary.loc[summary.index.str.match(r"logelr|gamma|alpha|beta|sig\[")]
    assert core["R_hat"].max() < 1.1
    # alpha[1], beta[n_d] and speedup[1] are pinned, so their ESS is NaN
    assert (core["ESS_bulk"].dropna() > 100).all()


def test_speedup_shape(fitted):
    """speedup is the exact deterministic transform of gamma, draw by draw.

    Pins Meyers' CSR definition ``speedup[w] = (1 - gamma)^(w-1)``: the first
    accident year is the reference (speedup == 1 exactly, not approximately),
    and later years scale geometrically. Checked per draw rather than on
    posterior means because it is an algebraic identity inside the model, so
    the tolerance is float-noise tight (1e-6), not statistical.
    """
    _, entry = fitted
    from ibnr.gallery.bayesian.meyers_csr.model import pooled

    gamma = pooled(entry.idata_, "gamma")
    speedup = pooled(entry.idata_, "speedup")  # (draws, n_w)
    assert speedup.shape[1] == entry.contract_["n_w"]
    np.testing.assert_allclose(speedup[:, 0], 1.0)
    # speedup[w] = (1-gamma)^(w-1), draw by draw
    np.testing.assert_allclose(speedup[:, 3], (1 - gamma) ** 3, rtol=1e-6)


def test_predict_and_score(fitted):
    """predict() honors the PredictiveDistribution contract and lands in the
    right ballpark.

    The 2x/0.5x band on the total is a wiring fence (units, anchor, prior
    runaway), not a calibration claim — calibration is settled by the
    200-company retrospective, where CSR is the best-calibrated paid entry.
    """
    tri, entry = fitted
    pred = entry.predict(seed=1)
    assert pred.n_targets == 11  # 10 accident years + total
    assert pred.targets["label"].tolist()[:3] == ["1988", "1989", "1990"]
    # AY 1988 is fully developed at the training cutoff: constant, zero SE
    assert pred.std()[0] == 0.0

    realized = entry.realized_ultimates(tri)
    assert not np.isnan(realized).any()
    table = pred.summary(observed=realized)
    total = table.iloc[-1]
    assert 0.5 * total["outcome"] < total["estimate"] < 2.0 * total["outcome"]
    assert 0.0 <= total["percentile"] <= 100.0


def test_evaluate_contract(fitted):
    """evaluate() returns the kernels-backed dict every entry must produce.

    Superset check so kernels can add scores without breaking entries; the
    per-target percentiles feed the PIT/KS calibration harness.
    """
    tri, entry = fitted
    result = entry.evaluate(entry.realized_ultimates(tri))
    assert set(result) >= {"summary", "percentiles"}
    assert len(result["percentiles"]) == 11
