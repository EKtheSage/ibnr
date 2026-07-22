"""Compartmental entry: fast contract/closed-form checks + slow mart fits.

The fast half needs no cmdstan: the joint paid+OS contract, the Model 2
(lognormal) data assembly with its non-positive-cell drops, and the
closed-form ODE solution verified against scipy's numerical integrator.
The slow half fits the real Schedule P mart like the other Bayesian
entries' smoke tests (``slow`` keeps cmdstan compilation out of the default
suite, ``mart`` auto-skips without the local warehouse).

Gesmann & Morris model claims as a three-compartment flow
EX (exposure) -> OS (outstanding) -> PD (paid), with rates ker (reporting) and
kp (payment) and scale factors RLR (reported loss ratio) and RRF (reserve
robustness factor). The entry solves the ODE system in closed form; the fast
tests below are what make that closed form trustworthy without paying for MCMC.
Two ablatable variants from the monograph: ``gaussian`` (Model 1, OS + cumulative
paid amounts) and ``lognormal`` (Model 2, OS + incremental paid loss ratios).
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.gallery.bayesian.compartmental.model import Compartmental, os_curve, paid_curve
from ibnr.kernels.contract import compartmental_stan_data

# Ground-truth compartmental parameters for the synthetic fixture. Chosen to be
# unremarkable and well separated: reporting (ker=3/yr) faster than payment
# (kp=1/yr), a 70% reported loss ratio, and 80% of reported ultimately paid.
KER, KP, RLR, RRF = 3.0, 1.0, 0.7, 0.8


def make_joint_triangle(n_w=4, n_d=4, premium=1000.0, os_floor=0.0):
    """Upper triangle simulated exactly on the compartmental curves (no
    noise): paid_loss, reported_loss = paid + OS, constant premium.

    Noise-free by design: every contract assertion below can then be an exact
    algebraic identity (rtol=1e-12) rather than a statistical tolerance, so a
    failure means the assembly is wrong, not that the fit was unlucky.
    ``dev_lag`` is months from origin start, so the first diagonal is 12 and
    dev year t = dev + 1 (CLAUDE.md, triangle conventions).
    """
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
    """The joint paid+outstanding Stan contract built from the noise-free triangle.

    OS is not a stored field — the contract derives it as reported minus paid.
    """
    tri = make_joint_triangle()
    return compartmental_stan_data(
        tri,
        paid_field="paid_loss",
        reported_field="reported_loss",
        premium_field="earned_premium",
    )


def test_registered():
    """The entry is in the gallery under its documented name."""
    from ibnr import gallery

    assert "compartmental" in gallery.list()
    assert gallery.registry.get("compartmental") is Compartmental


def test_curves_match_numerical_ode():
    """The closed forms are the EX->OS->PD system's solution: integrate the
    ODEs numerically and compare.

    The entry solves the compartment ODEs analytically for speed (Stan would
    otherwise need an ODE solver in the likelihood, the monograph's hardest
    parity case). This is the only test that proves the algebra is right — an
    independent numerical integration of the same system. rtol=1e-6 is the
    integrator's own accuracy floor, not modelling slack; the solver is run
    tight (rtol 1e-10) so any gap is attributable to the closed form.
    """
    from scipy.integrate import solve_ivp

    def rhs(_t, y):
        # dEX/dt = -ker*EX;  dOS/dt = ker*RLR*EX - kp*OS;  dPD/dt = kp*RRF*OS
        ex, os, pd_ = y
        return [-KER * ex, KER * RLR * ex - KP * os, KP * RRF * os]

    ts = np.linspace(0.5, 10.0, 20)
    # unit exposure at t=0, nothing yet outstanding or paid
    sol = solve_ivp(rhs, (0.0, 10.0), [1.0, 0.0, 0.0], t_eval=ts, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol.y[1], os_curve(ts, KER, KP, RLR), rtol=1e-6)
    np.testing.assert_allclose(sol.y[2], paid_curve(ts, KER, KP, RLR, RRF), rtol=1e-6)


def test_contract_shapes_and_stacking(contract):
    """The joint contract delta-stacks OS then paid over the same cells.

    Unlike the single-field entries, this contract feeds two observation types
    to one likelihood, distinguished by ``delta`` (0 = outstanding, 1 = paid).
    Pins the stacking order, that ``t`` is dev age in *years* (Stan's rate
    parameters are per year while dev_lag is months), that the OS block really
    is reported minus paid, and that the predictive anchors on the latest
    observed paid per origin.
    """
    c = contract
    n_cells = 4 + 3 + 2 + 1  # upper triangle of a 4x4
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
    """Paid and reported must be observed on identical cells, or OS is nonsense.

    OS = reported - paid is only defined cell-by-cell; if one field is missing a
    cell, a silent join would either drop data or subtract mismatched maturities.
    Fail at contract time instead.
    """
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
    """Model 2 rescales the same contract into loss ratios, paid incrementally.

    The lognormal variant models OS as a *level* loss ratio but paid as
    *incremental* loss ratios (monograph appendix 7.2), so the transformation is
    asymmetric across the delta blocks and easy to get backwards. Verified by
    cumulating the paid rows back onto the known curve. On noise-free data no
    cell is non-positive, so the lognormal support drops nothing — that
    baseline is what makes the next test meaningful.
    """
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
    """Non-positive cells are dropped, not clamped — lognormal has no mass at 0.

    Real books do produce zero or negative OS (a line closed out, or paid
    overtaking reported). Clamping would fabricate a tiny positive observation
    and bias the fit; dropping loses information but stays honest, and the count
    is recorded in ``dropped_cells_`` so the retrospective can report it.
    Degenerate case here: reported forced equal to paid, so every OS row goes.
    """
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

# Imported down here (E402 waived) so the fast tests above stay importable and
# runnable even where cmdstan and the mart are missing.
from .test_meyers_ccl import _cmdstan_ready  # noqa: E402
from .test_schedule_p import MART_AVAILABLE, WAREHOUSE  # noqa: E402

# Applied per-test rather than via pytestmark: this module's fast half must not
# be skipped. Note this local name shadows nothing — pytest.mark.slow is still
# applied separately alongside it.
slow = pytest.mark.skipif(
    not MART_AVAILABLE or not _cmdstan_ready(),
    reason="needs the Schedule P gold mart and a cmdstan installation",
)


@pytest.fixture(scope="module")
def fitted():
    """Gaussian variant (Model 1) on the same company/cutoff/seed as the other
    Bayesian smoke tests. Iterations are far below the monograph's production
    settings (adapt_delta .99, treedepth 15, ~200s/company) — this is a wiring
    check, not the retrospective.
    """
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
    """The gaussian variant converged.

    Uses the entry's own ``convergence()`` summary rather than regex-filtering
    the Stan table, since the correlated-AY-effects parameterization (LKJ prior
    on (RLR, RRF)) has too many derived quantities to name individually.
    """
    _, entry = fitted
    conv = entry.convergence()
    assert conv["max_rhat"] < 1.05
    assert conv["min_ess_bulk"] > 200


@pytest.mark.slow
@pytest.mark.mart
@slow
def test_predict_and_score(fitted):
    """predict() honors the PredictiveDistribution contract and lands in the
    right ballpark.

    A fence only. The retrospective already knows this variant is miscalibrated
    (combined D=39.9*, too sharp at CV~2.5%, and the monograph's single-company
    priors bias the loss ratio) — that is a calibration finding, not something
    this smoke test should re-litigate.
    """
    tri, entry = fitted
    pred = entry.predict(seed=1)
    assert pred.n_targets == 11  # 10 accident years + total
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
    anchored predictive stays near the gaussian variant's.

    The variants must remain ablatable (same data, same anchor, one modelling
    choice apart), so their point predictions should not diverge wildly even
    though their spreads differ a lot. The 0.7-1.3x band and the looser
    R-hat < 1.1 both allow for the shorter run used here.
    """
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
