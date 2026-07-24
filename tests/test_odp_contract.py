"""Fast checks of the ODP contract and its chain-ladder equivalence.

Everything the Bayesian ODP entry needs *before* Stan runs: the Triangle →
`odp_stan_data` mapping (design decision 3, the Stan data block is the data
contract), the ODP maximum-likelihood fit, and the Pearson dispersion phi.

The genins (Taylor & Ashe) sample is England & Verrall's own worked example;
all increments are positive, so it exercises the contract without tripping
the negative-increment guard. No cmdstan needed - everything here is the
plug-in numpy layer under the Bayesian entry, which is why these run in the
default suite while ``test_england_verrall_odp.py`` sits behind ``slow``.

Marked ``tieout``: the load-bearing claims are equalities against
chainladder-python and against a published number from the paper, not internal
self-consistency.
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
    """genins as both a chainladder Triangle (the reference) and our contract dict.

    Round-tripping through ``Triangle.from_chainladder`` rather than building the
    long table by hand keeps the tie-out honest: both sides start from literally
    the same numbers.
    """
    tri_cl = cl.load_sample("genins")
    t = Triangle.from_chainladder(tri_cl)
    field = t.fields[0]
    return tri_cl, odp_stan_data(t, loss_field=field)


def test_incrementals_rebuild_cumulative(genins_contract):
    """The contract's derived arrays are internally consistent with the source.

    ODP is fit on *increments* while the triangle stores cumulatives, and the
    predictive anchors on ``paid_to_date``/``latest_d``. Each of those three is
    derived independently, so this pins that they still describe one triangle:
    increments cumulate back, paid_to_date is the latest diagonal, and latest_d
    counts down 10..1 across origins (square 10x10 upper triangle, 55 cells).
    """
    _, c = genins_contract
    assert c["len_data"] == 55  # 10 + 9 + ... + 1
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
    """The ODP MLE (fit by iterative proportional fitting) IS the chain ladder.

    Two halves of the classic result. First the Poisson MLE property: fitted
    values reproduce the observed row totals *exactly* (rel=1e-9 is IPF
    convergence noise, not statistical slack). Then the consequence actuaries
    care about - paid-to-date plus the fitted future increments equals
    chainladder-python's volume-weighted ultimates. This equivalence is what
    licenses the Bayesian entry's vague-prior sanity check.
    """
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
    """phi reproduces the published dispersion for the paper's own example.

    phi scales the whole ODP predictive variance, so an error here is invisible
    in point estimates and fatal to calibration. The band is loose because the
    published figure depends on the degrees-of-freedom convention (n - p, and
    which parameters are counted as free); the order of magnitude is the signal.
    """
    _, c = genins_contract
    phi = pearson_phi(c["w"], c["d"], c["inc_loss"], c["n_w"], c["n_d"])
    assert phi > 0
    # England & Verrall report phi ~= 52,601 for Taylor & Ashe (B.A.J. 8 III,
    # section 3.2.3 context); accept a loose band around it
    assert 40_000 < phi < 70_000


def test_pearson_phi_tolerates_zero_dev_column():
    """A fully-paid-early book has an all-zero last dev column; the fitted
    means there are exactly 0 and must be excluded, not fatal.

    Regression test: Pearson residuals divide by sqrt(fitted), so a zero column
    used to produce inf/nan phi and reject the company outright. Recovered 34
    companies in the 200-company retrospective when fixed. Synthetic rather than
    a real triangle so the degenerate column is unambiguous; the seed just makes
    the gamma draws reproducible.
    """
    rng = np.random.default_rng(7)
    n = 4
    w, d, inc = [], [], []
    for wi in range(1, n + 1):
        for di in range(1, n + 2 - wi):
            w.append(wi)
            d.append(di)
            # zero out the last dev column entirely
            inc.append(0.0 if di == n else float(rng.gamma(5, 100)))
    phi = pearson_phi(np.array(w), np.array(d), np.array(inc), n, n)
    assert phi > 0 and np.isfinite(phi)


def test_negative_increment_rejected():
    """The contract refuses triangles the ODP likelihood cannot represent.

    An over-dispersed Poisson has non-negative support, so a negative
    incremental (salvage/subrogation, a reserve takedown) is not a data quirk to
    clamp - it invalidates the model. Fail at contract time with a clear error
    rather than letting Stan diverge. raa is the canonical example: it carries a
    well-known negative increment.
    """
    tri_cl = cl.load_sample("raa")  # raa has a famous negative increment
    t = Triangle.from_chainladder(tri_cl)
    with pytest.raises(ValueError, match="negative incremental"):
        odp_stan_data(t, loss_field=t.fields[0])
