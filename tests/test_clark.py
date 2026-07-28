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

import numpy as np
import pandas as pd
import pytest

# chainladder ships in the [interop] extra, so this has to SKIP when it is
# absent. Left unguarded it raises during collection, which pytest treats as a
# whole-run error rather than a skipped file - the core-only CI leg then reports
# nothing at all instead of the tests it can run.
cl = pytest.importorskip("chainladder")

from ibnr import Triangle  # noqa: E402
from ibnr.gallery.statistical.clark.model import (  # noqa: E402
    _REJECT_PENALTY,
    _SIMPLEX_XTOL,
    Clark,
    _check_converged,
    growth,
)

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


@pytest.mark.parametrize("curve", ["loglogistic", "weibull"])
def test_curve_mle_converges_on_its_merits(genins_tri, curve):
    """The MLE terminates because it converged, not by a floating-point accident.

    Nelder-Mead stops only when the simplex spread <= ``xatol`` AND
    ``max|f_i - f_0| <= fatol``. The objective here is a Poisson deviance over
    loss *amounts* - about -4.3e8 on genins - so adjacent representable float64s
    there are ~6e-8 apart. A ``fatol`` below that gap is satisfiable only when
    every simplex vertex evaluates BIT-IDENTICALLY, which makes termination a
    lottery on rounding the model does not control: the ``pytest (all)`` CI leg
    installs torch, jax and pymc together, which changes BLAS/OMP thread counts
    and hence the summation order inside the objective, and it stalled to
    ``maxiter`` on weibull - a spurious RuntimeError from a fit sitting at its
    optimum - while the same test passed in the ``interop`` leg and locally.

    So pin the property rather than the symptom. Asserting only that ``fit()``
    returns passed throughout the bug, on this machine and in most CI legs.
    """
    _, t, field = genins_tri
    entry = Clark().fit(t, loss_field=field, premium_field=None, method="ldf", growth_curve=curve)
    opt = entry.params_["optimizer"]

    # 1. The requested function tolerance must be reachable given where the
    #    objective actually sits on the float64 grid. This is the assertion that
    #    fails on the pre-fix absolute fatol=1e-10 (1e-10 vs a 6e-5 bar).
    ulp = float(np.spacing(abs(opt["objective"])))
    assert ulp > 1e-9, "the fixture must exercise a large-magnitude objective"
    assert opt["fatol"] > 1e3 * ulp

    # 2. Converged on the simplex, which is what the gate now reads.
    assert opt["simplex_spread"] <= _SIMPLEX_XTOL

    # 3. Substantial iteration headroom: the failure mode is a STALL, and a fit
    #    that genuinely needs 2000 iterations is a different problem from one
    #    that needs 90. Local runs: 65 (loglogistic), 90 (weibull).
    assert opt["iterations"] < opt["max_iterations"] // 4


class _FakeResult:
    """Minimal stand-in for scipy's OptimizeResult.

    ``final_simplex`` is scipy's ``(vertices, objective values)`` pair, and the
    values half is load-bearing: the gate has to see EVERY vertex's objective,
    not only the best one. A fake that drops them (``(sim, None)``) can only
    exercise the coordinate half of the acceptance rule, which is how a simplex
    parked on the infeasibility plateau - coordinates collapsed, objectives 1e11
    apart - once got through. ``fun`` is derived rather than passed for the same
    reason scipy derives it: it IS the best vertex, so the two cannot disagree.
    """

    def __init__(
        self, *, success, simplex, fvals, nit=2000, message="Maximum number of iterations"
    ):
        self.success = success
        self.final_simplex = (np.asarray(simplex, dtype=float), np.asarray(fvals, dtype=float))
        self.fun = float(np.min(fvals))
        self.nit = nit
        self.message = message


#: A simplex that has stopped moving in the parameters - the CI failure mode.
_COLLAPSED = [[0.4, 3.9], [0.4 + 1e-12, 3.9], [0.4, 3.9 + 1e-12]]
#: Healthy objective values: genins scale, disagreeing only at rounding noise
#: (~4e-6 absolute is tens of ULPs on -4.3e8).
_QUIET_FVALS = [-4.3e8, -4.3e8 + 3e-6, -4.3e8 + 1e-6]


def test_gate_accepts_a_stalled_but_collapsed_simplex():
    """``res.success`` is not the only gate; a stalled-but-finished fit passes.

    This is precisely the CI failure: Nelder-Mead reports ``maxiter`` because
    one of its two criteria stayed unmet, while both the parameters AND the
    objective have long since stopped moving. Trusting ``res.success`` alone
    raises on a converged fit. Mutation checked: dropping the ``maxiter``
    fallback (``if not res.success: raise``) fails here.
    """
    spread = _check_converged(_FakeResult(success=False, simplex=_COLLAPSED, fvals=_QUIET_FVALS))
    assert spread == pytest.approx(1e-12)


def test_gate_honors_success():
    """When scipy says it converged, that is the end of it.

    Its own criteria (coordinate spread <= ``xatol`` AND ``max|f_i - f_0| <=
    fatol``) are strictly tighter than the ``maxiter`` fallback below, so a
    successful result is accepted without re-deriving them - the relaxed rules
    exist only for the case scipy refuses to bless. Mutation checked: applying
    the fallback unconditionally fails here.
    """
    wandering = [[0.4, 3.9], [0.9, 3.9], [0.4, 4.7]]
    spread = _check_converged(
        _FakeResult(success=True, simplex=wandering, fvals=_QUIET_FVALS, message="converged")
    )
    assert spread == pytest.approx(0.8)


def test_gate_still_refuses_a_simplex_that_never_collapsed():
    """The loosened gate must not become a rubber stamp - a genuinely wandering
    simplex is still a failed fit. Mutation checked: the naive over-correction
    (accept every non-success result) passes the test above and fails here."""
    wandering = [[0.4, 3.9], [0.9, 3.9], [0.4, 4.7]]
    with pytest.raises(RuntimeError, match="simplex spread"):
        _check_converged(_FakeResult(success=False, simplex=wandering, fvals=_QUIET_FVALS))


def test_gate_refuses_a_collapsed_simplex_whose_objective_still_disagrees():
    """Coordinate collapse is NOT convergence on its own.

    Nelder-Mead can contract the simplex to a point while its vertices still
    evaluate to wildly different objective values - the shrink steps ran out of
    room, not out of disagreement. Accepting on coordinates alone would take
    that fit as finished. So the objective has to have collapsed too, and
    SCALE-RELATIVELY: 5e3 is nothing next to 4.3e8 in absolute terms and a
    catastrophe next to the ~4e-4 ``fatol`` this fit asks for. Mutation checked:
    deleting the objective-spread branch fails here and nowhere else.
    """
    noisy = [-4.3e8, -4.3e8 + 5e3, -4.3e8 + 5e3]
    with pytest.raises(RuntimeError, match="objective spread"):
        _check_converged(_FakeResult(success=False, simplex=_COLLAPSED, fvals=noisy))


@pytest.mark.parametrize("sentinel", [_REJECT_PENALTY, np.inf, np.nan])
def test_gate_refuses_an_optimum_stuck_on_the_rejection_penalty(sentinel):
    """A simplex can collapse INSIDE the infeasible region and report success -
    a converged answer to the wrong question. ``res.success`` never saw this and
    neither does the coordinate-spread check, so it is asserted separately.
    Parametrized over every shape an unusable vertex can take: the objective's
    own sentinel, and the non-finite values a pathological cell could produce."""
    with pytest.raises(RuntimeError, match="non-positive expected increment"):
        _check_converged(
            _FakeResult(success=True, simplex=_COLLAPSED, fvals=[sentinel] * 3, message="converged")
        )


def test_gate_refuses_a_simplex_straddling_the_rejection_penalty():
    """The review's reproduction, verbatim: coordinates collapsed to 5e-7, but
    the vertex objectives are [1e11, 1e12, 1e12].

    Two of the three vertices sit exactly ON the infeasibility sentinel and the
    objective spread is 9e11 - the optimizer gave up against the rejection
    region, it did not converge. Only ``res.fun`` (1e11, the one feasible-looking
    vertex) is healthy, which is why a gate reading the best vertex alone
    ACCEPTED this. Both new criteria reject it independently; the sentinel check
    is the one that fires first and names the actual problem.
    """
    straddling = [[0.4, 3.9], [0.4 + 5e-7, 3.9], [0.4, 3.9 + 5e-7]]
    with pytest.raises(RuntimeError, match="non-positive expected increment"):
        _check_converged(_FakeResult(success=False, simplex=straddling, fvals=[1e11, 1e12, 1e12]))


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
