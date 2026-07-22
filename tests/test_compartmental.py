"""Compartmental entry: fast contract/closed-form checks + slow mart fits.

The fast half needs no cmdstan: the joint paid+OS contract, the Model 2
(lognormal) data assembly with its non-positive-cell drops, and the
closed-form ODE solution verified against scipy's numerical integrator.
The slow half fits the real Schedule P mart like the other Bayesian
entries' smoke tests.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.gallery.bayesian.compartmental.model import Compartmental, os_curve, paid_curve
from ibnr.kernels.contract import compartmental_stan_data

KER, KP, RLR, RRF = 3.0, 1.0, 0.7, 0.8


def make_joint_triangle(n_w=4, n_d=4, premium=1000.0, os_floor=0.0):
    """Upper triangle simulated exactly on the compartmental curves (no
    noise): paid_loss, reported_loss = paid + OS, constant premium."""
    rows = []
    for w in range(n_w):
        for dev in range(n_d - w):
            t = float(dev + 1)
            origin = dt.date(2000 + w, 1, 1)
            eval_date = dt.date(2000 + w + dev, 12, 31)
            paid = premium * paid_curve(t, KER, KP, RLR, RRF)
            os = max(premium * os_curve(t, KER, KP, RLR), os_floor)
            for field, value in [
                ("paid_loss", paid),
                ("reported_loss", paid + os),
                ("earned_premium", premium),
            ]:
                rows.append((origin, 12 * (dev + 1), eval_date, field, value))
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "field", "value"])
    return Triangle.from_long(df, measure="cumulative", backend="duckdb")


@pytest.fixture(scope="module")
def contract():
    tri = make_joint_triangle()
    return compartmental_stan_data(
        tri,
        paid_field="paid_loss",
        reported_field="reported_loss",
        premium_field="earned_premium",
    )


def test_registered():
    from ibnr import gallery

    assert "compartmental" in gallery.list()
    assert gallery.registry.get("compartmental") is Compartmental


def test_curves_match_numerical_ode():
    """The closed forms are the EX->OS->PD system's solution: integrate the
    ODEs numerically and compare."""
    from scipy.integrate import solve_ivp

    def rhs(_t, y):
        ex, os, pd_ = y
        return [-KER * ex, KER * RLR * ex - KP * os, KP * RRF * os]

    ts = np.linspace(0.5, 10.0, 20)
    sol = solve_ivp(rhs, (0.0, 10.0), [1.0, 0.0, 0.0], t_eval=ts, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol.y[1], os_curve(ts, KER, KP, RLR), rtol=1e-6)
    np.testing.assert_allclose(sol.y[2], paid_curve(ts, KER, KP, RLR, RRF), rtol=1e-6)


def test_contract_shapes_and_stacking(contract):
    c = contract
    n_cells = 4 + 3 + 2 + 1
    assert c["len_data"] == 2 * n_cells
    assert c["n_w"] == c["n_d"] == 4
    # outstanding block (delta = 0) first, then the paid block
    assert c["delta"].tolist() == [0] * n_cells + [1] * n_cells
    # t is the dev age in years at period end
    np.testing.assert_allclose(c["t"], c["d"].astype(float))
    # OS rows are reported - paid = premium * os_curve
    os_rows = c["delta"] == 0
    np.testing.assert_allclose(
        c["loss"][os_rows], 1000.0 * os_curve(c["t"][os_rows], KER, KP, RLR), rtol=1e-12
    )
    # anchors: latest observed paid per origin
    assert c["latest_d"].tolist() == [4, 3, 2, 1]
    np.testing.assert_allclose(
        c["paid_to_date"],
        [1000.0 * paid_curve(float(d), KER, KP, RLR, RRF) for d in (4, 3, 2, 1)],
        rtol=1e-12,
    )


def test_contract_rejects_mismatched_cells():
    tri = make_joint_triangle()
    df = tri.execute()
    # drop one reported cell -> paid/reported no longer on identical cells
    drop = (df["field"] == "reported_loss") & (df["dev_lag"] == 24)
    tri_bad = Triangle.from_long(df[~drop], measure="cumulative", backend="duckdb")
    with pytest.raises(ValueError, match="identical cells"):
        compartmental_stan_data(
            tri_bad,
            paid_field="paid_loss",
            reported_field="reported_loss",
            premium_field="earned_premium",
        )


def test_lognormal_data_assembly(contract):
    entry = Compartmental()
    entry.contract_ = contract
    entry.variant_ = "lognormal"
    data = entry._lognormal_stan_data()
    # noise-free curves: every OS level and paid increment is positive
    assert entry.dropped_cells_ == {"outstanding": 0, "paid_incremental": 0}
    assert data["len_data"] == contract["len_data"]
    assert data["devfreq"] == 1.0
    # paid rows are incremental loss ratios: they cumulate back to the curve
    paid_rows = data["delta"] == 1
    w1 = paid_rows & (data["w"] == 1)
    np.testing.assert_allclose(
        np.cumsum(data["y"][w1]),
        paid_curve(data["t"][w1], KER, KP, RLR, RRF),
        rtol=1e-12,
    )
    # OS rows are levels, not increments
    os_rows = data["delta"] == 0
    np.testing.assert_allclose(
        data["y"][os_rows], os_curve(data["t"][os_rows], KER, KP, RLR), rtol=1e-12
    )


def test_lognormal_drops_nonpositive_cells():
    # force zero OS everywhere: reported == paid
    tri = make_joint_triangle()
    df = tri.execute()
    paid = df[df["field"] == "paid_loss"].set_index(["origin_period", "dev_lag"])["value"]
    rep = df["field"] == "reported_loss"
    df.loc[rep, "value"] = paid.loc[
        list(df.loc[rep, ["origin_period", "dev_lag"]].itertuples(index=False))
    ].to_numpy()
    tri0 = Triangle.from_long(df, measure="cumulative", backend="duckdb")
    c = compartmental_stan_data(
        tri0,
        paid_field="paid_loss",
        reported_field="reported_loss",
        premium_field="earned_premium",
    )
    entry = Compartmental()
    entry.contract_ = c
    entry.variant_ = "lognormal"
    data = entry._lognormal_stan_data()
    assert entry.dropped_cells_["outstanding"] == c["len_data"] // 2
    assert (data["delta"] == 1).all()


# ---------------------------------------------------------------------------
# slow: cmdstan + gold mart

from .test_meyers_ccl import _cmdstan_ready  # noqa: E402
from .test_schedule_p import MART_AVAILABLE, WAREHOUSE  # noqa: E402

slow = pytest.mark.skipif(
    not MART_AVAILABLE or not _cmdstan_ready(),
    reason="needs the Schedule P gold mart and a cmdstan installation",
)


@pytest.fixture(scope="module")
def fitted():
    from ibnr import gallery
    from ibnr.data.schedule_p import load_schedule_p

    tri = load_schedule_p(WAREHOUSE, lines=["workers_compensation"], companies=["11347"])
    entry = gallery.fit(
        "compartmental",
        tri,
        as_of="1997-12-31",
        chains=2,
        iter_warmup=500,
        iter_sampling=1000,
        seed=20260612,
    )
    return tri, entry


@pytest.mark.slow
@pytest.mark.mart
@slow
def test_fit_converges(fitted):
    _, entry = fitted
    conv = entry.convergence()
    assert conv["max_rhat"] < 1.05
    assert conv["min_ess_bulk"] > 200


@pytest.mark.slow
@pytest.mark.mart
@slow
def test_predict_and_score(fitted):
    tri, entry = fitted
    pred = entry.predict(seed=1)
    assert pred.n_targets == 11
    assert pred.std()[0] == 0.0  # AY 1988 fully developed -> observed anchor

    realized = entry.realized_ultimates(tri)
    assert not np.isnan(realized).any()
    table = pred.summary(observed=realized)
    total = table.iloc[-1]
    assert 0.5 * total["outcome"] < total["estimate"] < 2.0 * total["outcome"]
    assert 0.0 <= total["percentile"] <= 100.0


@pytest.mark.slow
@pytest.mark.mart
@slow
def test_lognormal_variant_fits(fitted):
    """Model 2 on the same company, small run: converges loosely and its
    anchored predictive stays near the gaussian variant's."""
    from ibnr import gallery

    tri, entry = fitted
    e2 = gallery.fit(
        "compartmental",
        tri,
        as_of="1997-12-31",
        variant="lognormal",
        chains=2,
        iter_warmup=500,
        iter_sampling=500,
        seed=20260612,
    )
    conv = e2.convergence()
    assert conv["max_rhat"] < 1.1
    p1 = entry.predict(seed=1).mean()[-1]
    p2 = e2.predict(seed=1).mean()[-1]
    assert 0.7 * p1 < p2 < 1.3 * p1
