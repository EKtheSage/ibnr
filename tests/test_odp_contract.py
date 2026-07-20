"""Fast checks of the ODP contract and its chain-ladder equivalence.

The genins (Taylor & Ashe) sample is England & Verrall's own worked example;
all increments are positive, so it exercises the contract without tripping
the negative-increment guard. No cmdstan needed — everything here is the
plug-in numpy layer under the Bayesian entry.
"""

import chainladder as cl
import numpy as np
import pytest

from ibnr import Triangle
from ibnr.gallery.bayesian.england_verrall_odp.model import odp_mle_fitted, pearson_phi
from ibnr.kernels.contract import odp_stan_data

pytestmark = pytest.mark.tieout


@pytest.fixture(scope="module")
def genins_contract():
    tri_cl = cl.load_sample("genins")
    t = Triangle.from_chainladder(tri_cl)
    field = t.fields[0]
    return tri_cl, odp_stan_data(t, loss_field=field)


def test_incrementals_rebuild_cumulative(genins_contract):
    _, c = genins_contract
    assert c["len_data"] == 55
    assert c["n_w"] == c["n_d"] == 10
    # increments cumulate back to the stored cumulative losses
    rebuilt = np.zeros_like(c["inc_loss"])
    for wi in range(1, c["n_w"] + 1):
        sel = c["w"] == wi
        rebuilt[sel] = np.cumsum(c["inc_loss"][sel])
    np.testing.assert_allclose(rebuilt, c["loss"], rtol=1e-12)
    # paid_to_date is the last diagonal of each origin
    np.testing.assert_allclose(
        c["paid_to_date"], [c["loss"][c["w"] == wi][-1] for wi in range(1, 11)]
    )
    assert c["latest_d"].tolist() == list(range(10, 0, -1))


def test_ipf_reproduces_chainladder_ultimates(genins_contract):
    tri_cl, c = genins_contract
    m = odp_mle_fitted(c["w"], c["d"], c["inc_loss"], c["n_w"], c["n_d"])
    # observed row totals are matched exactly (Poisson MLE property)
    for wi in range(1, c["n_w"] + 1):
        sel = c["w"] == wi
        assert m[wi - 1, : int(c["latest_d"][wi - 1])].sum() == pytest.approx(
            c["inc_loss"][sel].sum(), rel=1e-9
        )
    # paid-to-date + future fitted increments = chain-ladder ultimates
    future = np.zeros(c["n_w"])
    for wi in range(c["n_w"]):
        future[wi] = m[wi, int(c["latest_d"][wi]) :].sum()
    ult_cl = cl.Chainladder().fit(tri_cl).ultimate_.to_frame(origin_as_datetime=False)
    np.testing.assert_allclose(c["paid_to_date"] + future, ult_cl.iloc[:, 0].to_numpy(), rtol=1e-7)


def test_pearson_phi_positive(genins_contract):
    _, c = genins_contract
    phi = pearson_phi(c["w"], c["d"], c["inc_loss"], c["n_w"], c["n_d"])
    assert phi > 0
    # England & Verrall report phi ~= 52,601 for Taylor & Ashe (B.A.J. 8 III,
    # section 3.2.3 context); accept a loose band around it
    assert 40_000 < phi < 70_000


def test_negative_increment_rejected():
    tri_cl = cl.load_sample("raa")  # raa has a famous negative increment
    t = Triangle.from_chainladder(tri_cl)
    with pytest.raises(ValueError, match="negative incremental"):
        odp_stan_data(t, loss_field=t.fields[0])
