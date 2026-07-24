"""End-to-end smoke test of the meyers_ccl entry against the gold mart.

Protects the whole Bayesian pipeline for one real company: mart load →
``as_of`` training slice → Stan fit → ``PredictiveDistribution`` → scoring
against realized ultimates. It is a *smoke* test, not the validation run -
statistical calibration is settled by the 200-company retrospective in
``scripts/meyers_validation.py``, so the assertions here are deliberately
loose and only catch wiring breakage.

Slow (compiles the Stan model on first run, then samples) and mart-dependent;
runs only when cmdstan and the local warehouse are both available.

Markers: ``slow`` keeps cmdstan compilation out of the default suite (excluded
via addopts; it needs an RTools/MSVC toolchain on Windows and runs in a
separate CI job) and ``mart`` auto-skips when the local Schedule P gold mart
is absent. ``_cmdstan_ready`` here is the shared probe reused by the other
Bayesian entries' smoke tests.
"""

import numpy as np
import pytest

from .test_schedule_p import MART_AVAILABLE, WAREHOUSE

pytestmark = [pytest.mark.slow, pytest.mark.mart]


def _cmdstan_ready() -> bool:
    """True only if cmdstanpy is installed AND points at a real cmdstan tree.

    Importing cmdstanpy succeeds without a toolchain; ``cmdstan_path()`` is what
    actually raises when cmdstan was never installed or built.
    """
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


pytestmark.append(
    pytest.mark.skipif(
        not MART_AVAILABLE or not _cmdstan_ready(),
        reason="needs the Schedule P gold mart and a cmdstan installation",
    )
)


# Module-scoped: one sample for the whole file - the fit is the expensive part.
@pytest.fixture(scope="module")
def fitted():
    """Fit CCL on the Meyers setup: 1988-1997 accident years, 1997 diagonal.

    ``as_of="1997-12-31"`` is the Meyers monograph's training cutoff - a square
    10x10 upper triangle whose ultimates are realized ten years later in the same
    mart, which is what makes the retrospective possible. Chains/iterations are
    cut down from the production retro settings to keep the smoke test to a few
    minutes; the seed is pinned so a failure here is reproducible rather than a
    sampler lottery.
    """
    from ibnr import gallery
    from ibnr.data.schedule_p import load_schedule_p

    # group 11347: top of the mechanical selection for WC, and in Meyers' Table A.2
    tri = load_schedule_p(WAREHOUSE, lines=["workers_compensation"], companies=["11347"])
    entry = gallery.fit(
        "meyers_ccl",
        tri,
        as_of="1997-12-31",
        chains=2,
        iter_warmup=500,
        iter_sampling=1000,
        seed=20260612,
    )
    return tri, entry


def test_fit_converges(fitted):
    """The sampler actually converged on the core CCL parameters.

    Restricted to the model's own parameters (level, AY/dev effects, the AY
    correlation rho, the dev-varying sigmas) - generated quantities and
    transformed parameters would add noise to the max/min without saying
    anything about the sampler. Thresholds are the usual diagnostic
    conventions (R-hat < 1.1, ESS > 100), loosened for a 2-chain short run.
    """
    _, entry = fitted
    summary = entry.fit_.summary()
    core = summary.loc[summary.index.str.match(r"logelr|rho|alpha|beta|sig\[")]
    assert core["R_hat"].max() < 1.1
    # alpha[1] and beta[10] are pinned to 0, so their ESS is NaN by construction
    assert (core["ESS_bulk"].dropna() > 100).all()


def test_predict_and_score(fitted):
    """predict() honors the PredictiveDistribution contract and lands in the
    right ballpark.

    The 2x/0.5x band on the total is a sanity fence, not a calibration claim:
    it catches unit errors, a wrong anchor, or an unconditioned posterior,
    while staying wide enough that ordinary reserving error never trips it.
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
    # loose accuracy: the posterior mean total should be in the realized
    # outcome's ballpark for a stable workers' comp book
    assert 0.5 * total["outcome"] < total["estimate"] < 2.0 * total["outcome"]
    assert 0.0 <= total["percentile"] <= 100.0


def test_evaluate_contract(fitted):
    """evaluate() returns the kernels-backed dict every entry must produce.

    Superset (``>=``) so kernels can add scores (crps, ELPD, ...) without
    breaking entries; one percentile per target is the piece the PIT/KS
    calibration harness consumes.
    """
    tri, entry = fitted
    result = entry.evaluate(entry.realized_ultimates(tri))
    assert set(result) >= {"summary", "percentiles"}
    assert len(result["percentiles"]) == 11
