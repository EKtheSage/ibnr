"""End-to-end smoke test of the england_verrall_odp entry against the gold mart.

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
        "england_verrall_odp",
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
    core = summary.loc[summary.index.str.match(r"c$|alpha|beta")]
    assert core["R_hat"].max() < 1.05  # smooth GLM posterior: should converge easily
    assert (core["ESS_bulk"].dropna() > 200).all()
    assert entry.contract_["phi"] > 0


def test_posterior_centers_on_chainladder(fitted):
    """Vague priors => the posterior mean reserve sits near the ODP MLE
    (= chain-ladder) point forecast."""
    from ibnr.gallery.bayesian.england_verrall_odp.model import odp_mle_fitted

    _, entry = fitted
    c = entry.contract_
    m = odp_mle_fitted(c["w"], c["d"], c["inc_loss"], c["n_w"], c["n_d"])
    future = np.array([m[wi, int(c["latest_d"][wi]) :].sum() for wi in range(c["n_w"])])
    cl_total_ult = (c["paid_to_date"] + future).sum()
    pred = entry.predict(seed=1)
    bayes_total = pred.mean()[-1]
    assert abs(bayes_total / cl_total_ult - 1) < 0.05


def test_predict_and_score(fitted):
    tri, entry = fitted
    pred = entry.predict(seed=1)
    assert pred.n_targets == 11
    # AY 1988 is fully developed at the training cutoff: constant, zero SE
    assert pred.std()[0] == 0.0

    realized = entry.realized_ultimates(tri)
    assert not np.isnan(realized).any()
    table = pred.summary(observed=realized)
    total = table.iloc[-1]
    assert 0.5 * total["outcome"] < total["estimate"] < 2.0 * total["outcome"]
    assert 0.0 <= total["percentile"] <= 100.0
