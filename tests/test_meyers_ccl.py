"""End-to-end smoke test of the meyers_ccl entry against the gold mart.

Slow (compiles the Stan model on first run, then samples) and mart-dependent;
runs only when cmdstan and the local warehouse are both available.
"""

import numpy as np
import pytest

from .test_schedule_p import WAREHOUSE

pytestmark = [pytest.mark.slow, pytest.mark.mart]


def _cmdstan_ready() -> bool:
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


pytestmark.append(
    pytest.mark.skipif(
        not (WAREHOUSE / "_active_manifest.json").exists() or not _cmdstan_ready(),
        reason="needs the Schedule P gold mart and a cmdstan installation",
    )
)


@pytest.fixture(scope="module")
def fitted():
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
    _, entry = fitted
    summary = entry.fit_.summary()
    core = summary.loc[summary.index.str.match(r"logelr|rho|alpha|beta|sig\[")]
    assert core["R_hat"].max() < 1.1
    # alpha[1] and beta[10] are pinned to 0, so their ESS is NaN by construction
    assert (core["ESS_bulk"].dropna() > 100).all()


def test_predict_and_score(fitted):
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
    tri, entry = fitted
    result = entry.evaluate(entry.realized_ultimates(tri))
    assert set(result) >= {"summary", "percentiles"}
    assert len(result["percentiles"]) == 11
