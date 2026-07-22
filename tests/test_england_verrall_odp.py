"""End-to-end smoke test of the england_verrall_odp entry against the gold mart.

Same pipeline check as the other Bayesian smoke tests, plus the one property
that makes this entry recognizable as England & Verrall's ODP model: with vague
priors the posterior mean reserve must sit on the chain-ladder point estimate,
because the over-dispersed Poisson MLE *is* the chain ladder. If that link
breaks, the model has stopped being ODP whatever else still passes.

Fast numeric checks of the same contract (on the Taylor & Ashe genins sample,
no cmdstan needed) live in ``test_odp_contract.py``.

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


# Module-scoped: the Stan fit is the expensive part; every test reuses it.
@pytest.fixture(scope="module")
def fitted():
    """Fit ODP on the same company/cutoff/seed as the CCL and CSR smoke tests.

    Group 11347 workers' comp, trained on the 1997 diagonal so the 1988-1997
    ultimates are realized later in the same mart. Keeping the setup identical
    across entries makes cross-entry differences attributable to the model.
    """
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
    """The sampler converged on the GLM's intercept and AY/dev effects.

    Tighter thresholds than CCL/CSR (R-hat < 1.05, ESS > 200) on purpose: this
    is a log-link GLM with a smooth, near-Gaussian posterior and no hierarchical
    funnel, so anything worse signals a real problem rather than a short run.
    ``phi`` is the Pearson dispersion estimated *outside* the sampler and
    plugged into the likelihood; a non-positive value would silently disable
    over-dispersion (this bit us once — see the phi fix in the retro history).
    """
    _, entry = fitted
    summary = entry.fit_.summary()
    core = summary.loc[summary.index.str.match(r"c$|alpha|beta")]
    assert core["R_hat"].max() < 1.05  # smooth GLM posterior: should converge easily
    assert (core["ESS_bulk"].dropna() > 200).all()
    assert entry.contract_["phi"] > 0


def test_posterior_centers_on_chainladder(fitted):
    """Vague priors => the posterior mean reserve sits near the ODP MLE
    (= chain-ladder) point forecast.

    The defining England & Verrall property: the ODP GLM's maximum-likelihood
    fitted values reproduce chain-ladder ultimates exactly, so with priors that
    carry no information the Bayesian posterior mean must land on the same
    number. 5% is the tolerance because the two are not identical even in
    theory — MCMC noise, mild prior pull, and mean-vs-mode on a right-skewed
    predictive all contribute — but a genuine model error moves it far more.
    """
    from ibnr.gallery.bayesian.england_verrall_odp.model import odp_mle_fitted

    _, entry = fitted
    c = entry.contract_
    # fitted incremental means on the full square, (n_w, n_d)
    m = odp_mle_fitted(c["w"], c["d"], c["inc_loss"], c["n_w"], c["n_d"])
    # per origin, sum the cells past the latest observed dev = the reserve
    future = np.array([m[wi, int(c["latest_d"][wi]) :].sum() for wi in range(c["n_w"])])
    cl_total_ult = (c["paid_to_date"] + future).sum()
    pred = entry.predict(seed=1)
    bayes_total = pred.mean()[-1]  # last target is the all-origins total
    assert abs(bayes_total / cl_total_ult - 1) < 0.05


def test_predict_and_score(fitted):
    """predict() honors the PredictiveDistribution contract and lands in the
    right ballpark.

    The 2x/0.5x fence catches wiring faults only; ODP's actual calibration
    (D=47.9* on the 200-company retro) is measured by the retrospective script.
    """
    tri, entry = fitted
    pred = entry.predict(seed=1)
    assert pred.n_targets == 11  # 10 accident years + total
    # AY 1988 is fully developed at the training cutoff: constant, zero SE
    assert pred.std()[0] == 0.0

    realized = entry.realized_ultimates(tri)
    assert not np.isnan(realized).any()
    table = pred.summary(observed=realized)
    total = table.iloc[-1]
    assert 0.5 * total["outcome"] < total["estimate"] < 2.0 * total["outcome"]
    assert 0.0 <= total["percentile"] <= 100.0
