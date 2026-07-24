"""Clark (2003) MLE tieouts against chainladder-python's ClarkLDF, plus
predictive-contract checks. Fast - no cmdstan involved.

Clark fits a parametric *growth curve* G(age; omega, theta) - loglogistic or
Weibull - to the emergence pattern instead of estimating a free LDF per dev
age, under an over-dispersed Poisson likelihood. Two methods: ``ldf``
(each origin's ultimate anchored on its own paid-to-date) and ``cape_cod``
(one expected loss ratio shared across origins, requiring premium).

chainladder-python's ``ClarkLDF`` is the MLE reference, so this file is marked
``tieout``: it pins our optimizer to an independent implementation of the same
published method. The Bayesian twin of this entry is exercised in
``test_clark_growth_curve.py`` (slow, cmdstan).
"""

import chainladder as cl
import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.gallery.statistical.clark.model import Clark, growth

pytestmark = pytest.mark.tieout


@pytest.fixture(scope="module")
def genins_tri():
    """genins in both representations, so both sides fit identical numbers."""
    tri_cl = cl.load_sample("genins")
    t = Triangle.from_chainladder(tri_cl)
    return tri_cl, t, t.fields[0]


# Both growth curves from the paper: loglogistic (Clark's default) and Weibull.
@pytest.mark.parametrize("curve", ["loglogistic", "weibull"])
def test_ldf_ties_to_chainladder(genins_tri, curve):
    """Our MLE recovers ClarkLDF's omega/theta/sigma^2 and its point ultimates.

    Parameters first (the optimizer found the same likelihood mode), then the
    downstream quantity that actually matters. rel=2e-3 throughout: both sides
    are numerical optimizations of the same likelihood with different
    optimizers and starting points, so exact agreement is not available, but
    a genuine parameterization difference would show up far above 0.2%.
    """
    tri_cl, t, field = genins_tri
    ref = cl.ClarkLDF(growth=curve).fit(tri_cl)
    entry = Clark().fit(t, loss_field=field, premium_field=None, method="ldf", growth_curve=curve)
    prm = entry.params_
    assert prm["omega"] == pytest.approx(float(ref.omega_.values.ravel()[0]), rel=2e-3)
    assert prm["theta"] == pytest.approx(float(ref.theta_.values.ravel()[0]), rel=2e-3)
    assert prm["phi"] == pytest.approx(float(np.asarray(ref.scale_).ravel()[0]), rel=2e-3)

    # point ultimates (truncated at the final age) match the ClarkLDF pipeline
    ult_ref = (
        cl.Chainladder()
        .fit(cl.ClarkLDF(growth=curve).fit_transform(tri_cl))
        .ultimate_.to_frame(origin_as_datetime=False)
        .iloc[:, 0]
        .to_numpy()
    )
    c = entry.contract_
    step = c["dev_grain_months"]
    # Clark measures age from the *average* date of loss, so a cell evaluated at
    # dev age d months is really d - step/2 months of exposure-weighted maturity.
    age = c["latest_d"] * step - step / 2
    # ClarkLDF truncates at the oldest observed age rather than extrapolating to
    # G(inf) = 1, so the reference ultimate is paid * G(max_age) / G(own_age).
    g_max = growth(c["n_d"] * step - step / 2, prm["omega"], prm["theta"], curve)
    ours = c["paid_to_date"] * g_max / growth(age, prm["omega"], prm["theta"], curve)
    np.testing.assert_allclose(ours, ult_ref, rtol=2e-3)


def test_cape_cod_elr_ties_to_chainladder(genins_tri):
    """Cape Cod recovers ClarkLDF's expected loss ratio.

    Under Cape Cod the fitted ``level`` is expected *loss*, so dividing by the
    (constant) premium recovers the ELR that ClarkLDF reports directly. A flat
    premium keeps the comparison unambiguous - with varying premium the two
    implementations' exposure weighting would also be under test.
    """
    tri_cl, t, field = genins_tri
    prem_const = 10_000_000.0  # arbitrary; only the loss/premium ratio is compared
    ref = cl.ClarkLDF(growth="loglogistic").fit(
        tri_cl, sample_weight=tri_cl.latest_diagonal * 0 + prem_const
    )
    elr_ref = float(np.asarray(ref.elr_).ravel()[0])

    # genins carries no premium; inject a constant premium field into the long table
    long = t.execute()
    prem_rows = long.copy()
    prem_rows["field"] = "premium"
    prem_rows["value"] = prem_const
    both = Triangle.from_long(pd.concat([long, prem_rows], ignore_index=True))
    entry = Clark().fit(both, loss_field=field, premium_field="premium", method="cape_cod")
    assert entry.params_["level"][0] / prem_const == pytest.approx(elr_ref, rel=2e-3)


def test_cape_cod_without_premium_raises(genins_tri):
    """Cape Cod is exposure-based: no premium means no model, so fail loudly
    rather than silently falling back to the LDF method."""
    _, t, field = genins_tri
    with pytest.raises(ValueError, match="cape_cod needs a premium_field"):
        Clark().fit(t, loss_field=field, premium_field=None, method="cape_cod")


def test_predict_contract(genins_tri):
    """The simulated predictive is coherent with the deterministic MLE it wraps.

    Clark is a point estimator, so the entry enters the gallery via simulation
    (design decision 4). Three properties: the total target is the sum of the
    origin targets, the fully-developed first origin carries no uncertainty, and
    the Monte Carlo mean recovers the analytic MLE total. rel=0.05 on that last
    one is Monte Carlo error at 2000 draws on a right-skewed reserve
    distribution - the seed is pinned so it is reproducible rather than flaky.
    """
    _, t, field = genins_tri
    entry = Clark().fit(t, loss_field=field, premium_field=None, method="ldf")
    pred = entry.predict(n_draws=2000, seed=1)
    assert pred.n_targets == 11  # 10 origins + total
    est = pred.mean()
    # first origin fully developed: only process/parameter noise beyond age 114 is nil
    assert pred.std()[0] == 0.0
    # totals are the row sums
    assert est[-1] == pytest.approx(est[:-1].sum(), rel=1e-9)
    # the simulated mean total should sit near the MLE point total
    c = entry.contract_
    step = c["dev_grain_months"]
    prm = entry.params_
    age = c["latest_d"] * step - step / 2
    g_max = growth(c["n_d"] * step - step / 2, prm["omega"], prm["theta"], "loglogistic")
    g_age = growth(age, prm["omega"], prm["theta"], "loglogistic")
    point = (c["paid_to_date"] * g_max / g_age).sum()
    assert est[-1] == pytest.approx(point, rel=0.05)
