"""kernels.cdr: the one-year claims development result.

What this file protects, in descending order of strength:

1. **A golden tie-out.** The analytic msep is checked against the published
   output of R ChainLadder's ``CDR(MackChainLadder(MW2014, est.sigma="Mack"))``
   - Mario Wuthrich's own reference implementation of his 2008 paper - to the
   full precision the reference prints. chainladder-python has no CDR at all,
   so this is the only external reference that exists for these numbers, and it
   needs no optional dependency: the triangle and the expected output are both
   literal data below.
2. **Invariants that hold regardless of any reference**, above all that an
   accident year one development step from ultimate has one-year msep EXACTLY
   equal to its run-off msep. That single identity pins the boundary of the
   formula, where the Phi and Delta damping weights must both collapse to one.
3. **Analytic against simulation.** Two independent routes to the same quantity
   - a closed-form linearization and a re-reserving Monte Carlo - agreeing to
   Monte Carlo error is what makes either believable.

Sources for the literal data: Merz & Wuthrich, *Claims Run-Off Uncertainty: The
Full Picture* (SFI 14-69, 2014) via the R ChainLadder ``MW2014`` dataset;
expected values from that package's published ``CDR`` reference output.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle, gallery
from ibnr.errors import Refusal
from ibnr.kernels.cdr import (
    CDR_METHODS,
    DiagonalGenerator,
    MackDiagonal,
    cdr_methods,
    cdr_risk_measures,
    get_cdr_method,
    one_year_cdr,
    rereserve,
    simulate_one_year_cdr,
    simulated_msep,
)
from ibnr.kernels.mack import PROCESS_LAWS, fit_mack

from .conftest import assert_triangles_equal, make_cohort_triangle

# -- the MW2014 triangle (17 accident years, cumulative) ----------------------
# fmt: off
MW2014_ROWS = [
    [13109, 20355, 21337, 22043, 22401, 22658, 22997, 23158, 23492, 23664, 23699, 23904, 23960,
     23992, 23994, 24001, 24002],
    [14457, 22038, 22627, 23114, 23238, 23312, 23440, 23490, 23964, 23976, 24048, 24111, 24252,
     24538, 24540, 24550],
    [16075, 22672, 23753, 24052, 24206, 24757, 24786, 24807, 24823, 24888, 24986, 25401, 25681,
     25705, 25732],
    [15682, 23464, 24465, 25052, 25529, 25708, 25752, 25770, 25835, 26075, 26082, 26146, 26150,
     26167],
    [16551, 23706, 24627, 25573, 26046, 26115, 26283, 26481, 26701, 26718, 26724, 26728, 26735],
    [15439, 23796, 24866, 25317, 26139, 26154, 26175, 26205, 26764, 26818, 26836, 26959],
    [14629, 21645, 22826, 23599, 24992, 25434, 25476, 25549, 25604, 25709, 25723],
    [17585, 26288, 27623, 27939, 28335, 28638, 28715, 28759, 29525, 30302],
    [17419, 25941, 27066, 27761, 28043, 28477, 28721, 28878, 28948],
    [16665, 25370, 26909, 27611, 27729, 27861, 29830, 29844],
    [15471, 23745, 25117, 26378, 26971, 27396, 27480],
    [15103, 23393, 26809, 27691, 28061, 29183],
    [14540, 22642, 23571, 24127, 24210],
    [14590, 22336, 23440, 24029],
    [13967, 21515, 22603],
    [12930, 20111],
    [12539],
]

# published: CDR(MackChainLadder(MW2014, est.sigma="Mack")) - IBNR, CDR(1)S.E., Mack.S.E.
MW2014_IBNR = [
    0.0, 1.022874, 10.085643, 21.187574, 117.662565, 223.279748, 361.808180, 469.408830,
    653.504225, 1008.763182, 1011.859648, 1406.702133, 1492.903495, 1917.636398, 2458.152208,
    3384.341045, 9596.552341,
]
MW2014_CDR_SE = [
    0.0, 0.4083149, 2.5393857, 16.7232632, 156.4022713, 137.6522771, 171.1812092, 70.3161155,
    271.6352221, 310.1268449, 103.3834357, 632.6388191, 315.0489135, 406.1424672, 285.2076540,
    668.2337878, 733.2222786,
]
MW2014_MACK_SE = [
    0.0, 0.4083149, 2.5652899, 16.8984949, 157.2756452, 207.1650862, 261.9266093, 292.2622285,
    390.5874717, 502.0606072, 486.0911099, 806.9028971, 793.9381916, 891.6613403, 916.4940218,
    1106.1262716, 1295.6909824,
]
# fmt: on
MW2014_TOTAL = {"ibnr": 24134.870088, "cdr_se": 1842.8507073, "runoff_se": 3233.6807352}


def mw2014_matrix() -> np.ndarray:
    grid = np.full((17, 17), np.nan)
    for i, row in enumerate(MW2014_ROWS):
        grid[i, : len(row)] = row
    return grid


def synthetic_triangle(backend_name, n_w: int = 8, seed: int = 12345) -> np.ndarray:
    """A run-off square generated under Mack's own assumptions, then cut to the
    upper triangle. Moderate coefficients of variation on purpose: the
    analytic-vs-simulation comparison converges slowly when a step's CV is near
    one, and this file is not the place to measure that."""
    rng = np.random.default_rng(seed)
    factors = np.array([1.6, 1.25, 1.12, 1.06, 1.03, 1.015, 1.005])
    sigmas = np.array([6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])
    cum = np.empty((n_w, n_w))
    cum[:, 0] = rng.uniform(900.0, 1100.0, size=n_w)
    for j in range(n_w - 1):
        eps = rng.standard_normal(n_w) * sigmas[j]
        cum[:, j + 1] = factors[j] * cum[:, j] + np.sqrt(cum[:, j]) * eps
    cum[np.arange(n_w)[:, None] + np.arange(n_w)[None, :] >= n_w] = np.nan
    return cum


@pytest.fixture(scope="module")
def mw2014_fit():
    """Fitted on the default backend: this block is about agreeing with R's
    numbers, and the backend matrix is exercised by the invariant tests below."""
    tri = make_cohort_triangle(None, mw2014_matrix(), start_year=1990)
    return fit_mack(tri, loss_field="paid_loss", sigma_rule="mack")


@pytest.mark.tieout
def test_mw2014_matches_published_r_output(mw2014_fit):
    """The golden tie-out: every accident year and the total, against R
    ChainLadder's published CDR output for MW2014. Tolerance is the precision
    the reference prints (7 decimals), not a fudge factor."""
    out = one_year_cdr(mw2014_fit).summary()
    np.testing.assert_allclose(out["ibnr"][:-1], MW2014_IBNR, atol=1e-5)
    np.testing.assert_allclose(out["cdr_se"][:-1], MW2014_CDR_SE, atol=1e-6)
    np.testing.assert_allclose(out["runoff_se"][:-1], MW2014_MACK_SE, atol=1e-6)
    for column, expected in MW2014_TOTAL.items():
        assert out[column].iloc[-1] == pytest.approx(expected, abs=1e-5)


def test_mw2014_second_origin_one_year_equals_runoff(mw2014_fit):
    """R's own documentation calls this out: for the accident year with a single
    development step left, CDR(1)S.E. and Mack.S.E. are the same number
    (0.4083149 here). Nothing is left to learn after the first year, so the two
    views of the risk coincide - the sharpest single check of the damping."""
    res = one_year_cdr(mw2014_fit)
    assert res.msep[1] == pytest.approx(res.runoff_msep[1], rel=1e-12)
    assert np.sqrt(res.msep[1]) == pytest.approx(0.4083149, abs=1e-6)


def test_one_year_never_exceeds_runoff(backend_name):
    """Structural: the one year is part of the run-off, so its msep is bounded
    by it - per accident year and in aggregate."""
    tri = make_cohort_triangle(backend_name, synthetic_triangle(backend_name))
    res = one_year_cdr(fit_mack(tri, loss_field="paid_loss"))
    assert (res.msep <= res.runoff_msep + 1e-9).all()
    assert res.msep_total <= res.runoff_msep_total + 1e-9
    share = res.summary()["one_year_share"].to_numpy()[1:]  # origin 0 is fully developed
    assert ((share > 0) & (share <= 1 + 1e-12)).all()


def test_fully_developed_origin_has_no_cdr(backend_name):
    """The oldest accident year is at ultimate already: no next diagonal, no
    re-estimate, no CDR - analytically and in the draws."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    assert one_year_cdr(fit).msep[0] == 0.0
    pred = simulate_one_year_cdr(fit, n_draws=200, seed=1)
    assert (pred.samples[:, 0] == 0.0).all()


def test_zero_process_variance_gives_zero_cdr(backend_name):
    """A triangle that develops by exactly constant factors has sigma = 0
    everywhere, so both halves of the formula vanish. Exercises the degenerate
    paths (zero variance, extrapolated last sigma) that a real triangle never
    reaches."""
    factors = [1.5, 1.2, 1.1]
    cum = np.full((4, 4), np.nan)
    for i in range(4):
        cum[i, 0] = 100.0 * (i + 1)
        for j in range(3 - i):
            cum[i, j + 1] = cum[i, j] * factors[j]
    fit = fit_mack(make_cohort_triangle(backend_name, cum), loss_field="paid_loss")
    np.testing.assert_allclose(fit.sigma2, 0.0)
    res = one_year_cdr(fit)
    np.testing.assert_allclose(res.msep, 0.0)
    assert res.msep_total == pytest.approx(0.0)
    pred = simulate_one_year_cdr(fit, n_draws=100, seed=2)
    np.testing.assert_allclose(pred.samples, 0.0, atol=1e-9)


def test_scale_equivariance(backend_name):
    """Reporting the triangle in thousands must scale every standard error by
    the same factor and leave the one-year share untouched - the check that the
    sigma^2 / f^2 / volume bookkeeping is dimensionally consistent."""
    cum = synthetic_triangle(backend_name)
    base = one_year_cdr(fit_mack(make_cohort_triangle(backend_name, cum), loss_field="paid_loss"))
    scaled = one_year_cdr(
        fit_mack(make_cohort_triangle(backend_name, cum * 1000.0), loss_field="paid_loss")
    )
    np.testing.assert_allclose(scaled.msep, base.msep * 1000.0**2, rtol=1e-10)
    assert scaled.msep_total == pytest.approx(base.msep_total * 1000.0**2, rel=1e-10)


def test_aggregate_exceeds_sum_of_accident_years(backend_name):
    """Accident years share the factors they have yet to run through, so the
    aggregate msep carries positive cross terms: no diversification credit is
    available against estimation error."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    res = one_year_cdr(fit)
    assert res.msep_total > res.msep.sum()


def test_phi_delta_split_matches_the_reference_grouping(backend_name):
    """Re-derives the per-accident-year msep in the OTHER algebraic grouping -
    the one Wuthrich's R implementation uses, where the later dev steps appear
    once as a_j * ratio_j / S_j rather than split into process and estimation
    parts - and requires the two to agree. Catches an error in either
    arrangement that a self-consistent refactor would hide."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    res = one_year_cdr(fit)
    last = fit.n_d - 1
    ratio = fit.sigma2 / fit.f**2
    c_new = np.array([fit.cum[i, int(fit.latest_dev[i])] for i in range(fit.n_w)])
    expected = np.zeros(fit.n_w)
    for i in range(fit.n_w):
        j0 = int(fit.latest_dev[i])
        if j0 >= last:
            continue
        total = ratio[j0] / fit.cum[i, j0] + ratio[j0] / fit.s[j0]
        for j in range(j0 + 1, last):
            m = int(np.nonzero(fit.latest_dev == j)[0][0])
            alpha = c_new[m] / (fit.s[j] + c_new[m])
            total += alpha * ratio[j] / fit.s[j]
        expected[i] = fit.ultimate[i] ** 2 * total
    np.testing.assert_allclose(res.msep, expected, rtol=1e-12)


def test_simulation_isolates_process_risk(backend_name):
    """With parameter risk switched off the draws must reproduce Phi alone -
    the process half - computed here directly from the fit. This is what makes
    the risk-source switch meaningful rather than decorative."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    last = fit.n_d - 1
    ratio = fit.sigma2 / fit.f**2
    phi = np.zeros(fit.n_w)
    for i in range(fit.n_w):
        j0 = int(fit.latest_dev[i])
        if j0 >= last:
            continue
        total = ratio[j0] / fit.cum[i, j0]
        for j in range(j0 + 1, last):
            m = int(np.nonzero(fit.latest_dev == j)[0][0])
            c = fit.cum[m, j]
            total += ratio[j] * c / (fit.s[j] + c) ** 2
        phi[i] = fit.ultimate[i] ** 2 * total
    pred = simulate_one_year_cdr(
        fit, n_draws=60_000, seed=17, process="normal", parameter_risk=False
    )
    np.testing.assert_allclose(simulated_msep(pred)[:-1], phi, rtol=0.06)


def test_analytic_matches_simulation(backend_name):
    """The two routes are independent implementations of the same quantity: a
    closed-form linearization versus a re-reserving Monte Carlo. ``normal``
    noise is the fair comparison (the analytic formula is itself a linear,
    unfloored approximation); the tolerance is Monte Carlo error at this draw
    count, not a fitted fudge."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    res = one_year_cdr(fit)
    pred = simulate_one_year_cdr(
        fit, n_draws=60_000, seed=23, process="normal", parameter_risk=True
    )
    sim = simulated_msep(pred)
    np.testing.assert_allclose(sim[:-1], res.msep, rtol=0.08)
    assert sim[-1] == pytest.approx(res.msep_total, rel=0.08)
    # and the model's own claim: the CDR is centred on zero
    assert abs(pred.samples[:, -1].mean()) < 0.1 * np.sqrt(res.msep_total)


def test_cdr_distribution_layout(backend_name):
    """One column per origin plus a total that is the row-sum of the same draws
    - so the aggregate keeps whatever dependence the draws have, rather than
    assuming independence."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    pred = simulate_one_year_cdr(fit, n_draws=500, seed=5)
    assert pred.n_targets == fit.n_w + 1
    assert list(pred.targets["label"])[-1] == "total"
    np.testing.assert_allclose(pred.samples[:, -1], pred.samples[:, :-1].sum(axis=1), rtol=1e-12)


def test_risk_measures_are_stated_on_the_loss(backend_name):
    """VaR/TVaR are reported on ``-CDR`` (the strengthening), which is the side
    capital is held against, and TVaR must sit beyond VaR at every level."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    pred = simulate_one_year_cdr(fit, n_draws=40_000, seed=41)
    table = cdr_risk_measures(pred, levels=(0.95, 0.995))
    assert len(table) == pred.n_targets
    assert (table["tvar_0.995"] >= table["var_0.995"]).all()
    assert (table["var_0.995"] >= table["var_0.95"]).all()
    # the reported VaR is the empirical quantile of the loss, nothing smoothed
    total_loss = -pred.samples[:, -1]
    assert table["var_0.995"].iloc[-1] == pytest.approx(np.quantile(total_loss, 0.995))
    # a fully developed origin cannot move, so it consumes no capital
    assert table["var_0.995"].iloc[0] == pytest.approx(0.0)


def test_risk_measures_reject_degenerate_levels(backend_name):
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    pred = simulate_one_year_cdr(fit, n_draws=100, seed=1)
    with pytest.raises(ValueError, match="strictly inside"):
        cdr_risk_measures(pred, levels=(1.0,))


@pytest.mark.parametrize("law", ["gamma", "normal", "lognormal"])
def test_process_laws_are_moment_matched(backend_name, law):
    """All three shapes match Mack's two conditional moments, so they may move
    the tail of the CDR but not its scale. Anything more than a few percent
    apart in msep would mean a mis-parameterized law rather than a modelling
    choice."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    res = one_year_cdr(fit)
    pred = simulate_one_year_cdr(fit, n_draws=40_000, seed=31, process=law)
    assert simulated_msep(pred)[-1] == pytest.approx(res.msep_total, rel=0.12)


def test_rejects_unknown_process_law(backend_name):
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    with pytest.raises(ValueError, match="process must be one of"):
        simulate_one_year_cdr(fit, n_draws=10, process="cauchy")


# -- the two axes, and the option surface over them ---------------------------


@pytest.mark.parametrize("process", PROCESS_LAWS)
@pytest.mark.parametrize("parameter_risk", [True, False])
def test_simulate_is_exactly_the_generator_composed_with_rereserve(process, parameter_risk):
    """The refactor's safety property, and it is BIT identity rather than
    agreement: the published Mack route must be the ``MackDiagonal`` generator
    followed by :func:`rereserve`, with the same seed producing the same bytes.

    A tolerance-based check would pass on a re-ordered RNG consumption that
    quietly changes every published number, which is exactly what a refactor of
    a released simulation must not be allowed to do.
    """
    fit = fit_mack(make_cohort_triangle(None, synthetic_triangle(None)), loss_field="paid_loss")
    convenience = simulate_one_year_cdr(
        fit, n_draws=257, seed=7, process=process, parameter_risk=parameter_risk
    )
    rng = np.random.default_rng(7)
    composed = rereserve(
        fit,
        MackDiagonal(process=process, parameter_risk=parameter_risk).draw(
            fit, n_draws=257, rng=rng
        ),
    )
    per_origin = np.ascontiguousarray(convenience.samples[:, :-1])
    assert np.array_equal(per_origin.view(np.uint8), composed.view(np.uint8))
    # ... and the same numbers again through generator=, which must not re-seed
    # or re-order anything either
    through_generator = simulate_one_year_cdr(
        fit,
        n_draws=257,
        seed=7,
        generator=MackDiagonal(process=process, parameter_risk=parameter_risk),
    )
    assert np.array_equal(
        convenience.samples.view(np.uint8), through_generator.samples.view(np.uint8)
    )


def test_generator_name_string_is_the_default_configured_generator(backend_name):
    """``generator="mack"`` must be the same thing as the no-argument default,
    not a differently-configured instance that happens to be close."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    a = simulate_one_year_cdr(fit, n_draws=200, seed=3)
    b = simulate_one_year_cdr(fit, n_draws=200, seed=3, generator="mack")
    assert np.array_equal(a.samples.view(np.uint8), b.samples.view(np.uint8))


def test_cdr_methods_is_the_discoverable_option_surface():
    """One row per route, and the table must describe every registered method -
    a method cannot join ``CDR_METHODS`` without appearing in the listing a user
    reads to choose one."""
    table = cdr_methods()
    assert list(table["name"]) == list(CDR_METHODS)
    for column in ("route", "generates", "re_estimates", "returns", "requires", "validated"):
        assert column in table.columns
        assert table[column].map(bool).all()
    # the analytic route is listed even though generator= cannot take it, and it
    # says where it does live
    row = table.set_index("name").loc["merz_wuthrich"]
    assert row["route"] == "analytic"
    assert row["entry_point"] == "one_year_cdr(fit)"
    # every simulation route re-estimates the reserve the same way: axis 2 is
    # not a choice, and the table must not imply that it is
    simulated = table[table["route"] == "simulation"]
    assert len(simulated) >= 2
    assert simulated["re_estimates"].nunique() == 1


def test_registry_keys_agree_with_the_generators_they_name():
    """A registry key, a ``CDRMethod.name`` and the generator class's own
    ``name`` are three places one string is written; they must be one string.

    Since 0.5.1 a row can be a simulation route and STILL carry no generator
    class - ``gallery`` wraps a fitted entry, which no name can supply. So the
    invariant is stated on ``why_not_by_name`` rather than on ``route``: every
    row either builds from its name or says why it does not, never both and
    never neither. Read against ``route``, ``analytic`` and ``gallery`` are
    nameless for opposite reasons, and only one of them is not a generator.
    """
    for key, method in CDR_METHODS.items():
        assert key == method.name
        assert method.route in ("analytic", "simulation")
        assert get_cdr_method(key) is method
        assert (method.generator is None) == bool(method.why_not_by_name)
        if method.route == "analytic":
            assert method.generator is None
            continue
        if method.generator is None:
            # nameless but still a simulation: it must say how it IS reached
            assert "simulate_one_year_cdr" in method.entry_point
            continue
        assert issubclass(method.generator, DiagonalGenerator)
        assert method.generator.name == key


def test_merz_wuthrich_is_refused_as_a_generator(backend_name):
    """The hard constraint: the closed form is a linearization around Mack's
    conditional moments, so it is not one option among many that any fit can
    take. Asking for it as a generator must be refused by name and redirected,
    not silently substituted."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    with pytest.raises(ValueError, match="analytic method, not a diagonal generator"):
        simulate_one_year_cdr(fit, n_draws=10, generator="merz_wuthrich")
    try:
        simulate_one_year_cdr(fit, n_draws=10, generator="merz_wuthrich")
    except ValueError as exc:
        assert "one_year_cdr(fit)" in str(exc)


def test_mack_knobs_cannot_ride_along_with_another_generator(backend_name):
    """The inert-parameter refusal. ``process=``/``parameter_risk=`` are
    MackDiagonal's vocabulary; accepting them beside ``generator=`` would leave
    them doing nothing while the answer still looked perfect - this repo's named
    bug class. Refused by name instead."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    for kwargs in ({"process": "normal"}, {"parameter_risk": False}):
        with pytest.raises(ValueError, match="cannot be combined with generator="):
            simulate_one_year_cdr(fit, n_draws=10, generator="mack", **kwargs)
    # explicitly passing the DEFAULT value is refused too: "the same as the
    # default" is not the same as "not supplied", and only the second is inert-safe
    with pytest.raises(ValueError, match="cannot be combined with generator="):
        simulate_one_year_cdr(fit, n_draws=10, generator="mack", process="gamma")


def test_unknown_generator_is_refused_by_name(backend_name):
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    # an argument naming no method is a refused option; the lookup itself is a KeyError
    with pytest.raises(Refusal, match="no CDR method named") as named:
        simulate_one_year_cdr(fit, n_draws=10, generator="bootstrap")
    assert named.value.reason == "invalid_option" and named.value.option == "generator"
    with pytest.raises(Refusal, match="must be a DiagonalGenerator"):
        simulate_one_year_cdr(fit, n_draws=10, generator=object())
    with pytest.raises(KeyError, match="no CDR method named"):
        get_cdr_method("odp")


def test_rereserve_refuses_a_diagonal_that_is_not_the_fits_shape(backend_name):
    """``rereserve`` is public so a caller can re-reserve draws from any model
    that predicts next year's cells. A column count that is not the fit's is the
    one mistake that would otherwise index cleanly and answer for the wrong
    origins."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    with pytest.raises(ValueError, match="must be .*n_w="):
        rereserve(fit, np.zeros((5, fit.n_w - 1)))
    with pytest.raises(ValueError, match="must be .*n_w="):
        rereserve(fit, np.zeros(fit.n_w))


def test_rereserve_accepts_a_hand_built_diagonal(backend_name):
    """The generic path: hand it the diagonal the chain ladder itself projects
    and every CDR must be exactly zero, because nothing was learned that the
    time-I estimate did not already assume."""
    fit = fit_mack(
        make_cohort_triangle(backend_name, synthetic_triangle(backend_name)),
        loss_field="paid_loss",
    )
    k = fit.latest_dev
    expected = np.where(k < fit.n_d - 1, fit.latest * fit.f[np.minimum(k, fit.n_d - 2)], 0.0)
    cdr = rereserve(fit, np.tile(expected, (3, 1)))
    np.testing.assert_allclose(cdr, 0.0, atol=1e-8)


# -- the development grain the "one year" in the name refers to ---------------


def test_the_annual_cohort_helper_builds_the_rows_it_always_did(backend_name):
    """``make_cohort_triangle`` gained a ``dev_grain=`` knob so the tests below
    can build a quarterly and a monthly cohort. Widening a shared fixture is only
    safe if its default output is unchanged, so this pins the annual rows against
    the rule the helper used before the knob existed: origin ``i`` = Jan 1 of
    ``start_year + i``, dev_lag ``12*(j+1)``, eval date Dec 31 of
    ``start_year + i + j``, on the annual grain."""
    cum = synthetic_triangle(backend_name)
    n_w, n_d = cum.shape
    expected = Triangle.from_long(
        pd.DataFrame(
            [
                {
                    "origin_period": dt.date(2010 + i, 1, 1),
                    "dev_lag": 12 * (j + 1),
                    "eval_date": dt.date(2010 + i + j, 12, 31),
                    "field": "paid_loss",
                    "value": float(cum[i, j]),
                }
                for i in range(n_w)
                for j in range(n_d)
                if not np.isnan(cum[i, j])
            ]
        ),
        measure="cumulative",
        backend=backend_name,
    )
    got = make_cohort_triangle(backend_name, cum)
    assert got.meta == expected.meta
    assert_triangles_equal(got, expected)


class _ExplodingDiagonal(DiagonalGenerator):
    """A generator that must never be reached. The grain refusal in
    ``simulate_one_year_cdr`` has to come before the generator is consulted at
    all, so a stub whose every method raises is what shows where the refusal
    sits: if either method runs, the check was placed too late."""

    name = "exploding"

    def check(self, fit):
        raise AssertionError("check() ran, so the grain refusal came too late")

    def draw(self, fit, *, n_draws, rng):
        raise AssertionError("draw() ran, so the grain refusal came too late")


@pytest.mark.parametrize(("dev_grain", "step"), [("Q", 3), ("M", 1)])
def test_a_non_annual_dev_grain_is_refused_by_name(backend_name, dev_grain, step):
    """One development step is one year only on an annual triangle.

    Every route to a one-year CDR advances the triangle by exactly one
    development step: the Merz-Wuthrich closed form, each ``DiagonalGenerator``
    and ``rereserve``. On a quarterly or monthly triangle that step is three
    months or one, so the answer would be a three-month or one-month claims
    development result reported under a one-year name, with no sign that
    anything was off. The three functions that name a year therefore refuse,
    naming the grain they measured and the two ways out, and so do the two
    gallery methods that call them."""
    cum = synthetic_triangle(backend_name)
    tri = make_cohort_triangle(backend_name, cum, dev_grain=dev_grain)
    assert tri.validate(strict=False) == []
    fit = fit_mack(tri, loss_field="paid_loss")
    assert fit.dev_grain_months == step
    # the remedy is pinned call by call, not just by the name of the method: an
    # aggregation to the wrong grain letter, or one that forgets the year-end
    # slice the annual buckets need, would send the caller into a second error.
    expected = (
        rf"{step}-month development grain"
        r".*one development step"
        r".*as_of\(\)"
        r'.*with_origin_grain\("Y"\)\.with_dev_grain\("Y"\)'
        r".*msep_runoff\(\)"
    )

    with pytest.raises(ValueError, match=expected):
        one_year_cdr(fit)
    with pytest.raises(ValueError, match=expected):
        simulate_one_year_cdr(fit, n_draws=10, seed=0, generator=_ExplodingDiagonal())
    with pytest.raises(ValueError, match=expected):
        rereserve(fit, np.zeros((5, fit.n_w)))

    entry = gallery.get("mack")().fit(tri, loss_field="paid_loss")
    with pytest.raises(ValueError, match=expected):
        entry.one_year_cdr()
    with pytest.raises(ValueError, match=expected):
        entry.cdr_distribution(n_draws=10, seed=0)

    # the control, on the identical numbers: the annual grain still answers, so
    # the refusal is about the grain and not about this cohort.
    annual = fit_mack(make_cohort_triangle(backend_name, cum), loss_field="paid_loss")
    assert np.isfinite(one_year_cdr(annual).msep_total)


def test_the_aggregation_remedy_works_only_from_a_year_end(backend_name):
    """The refusal names a way out, so the way out has to work.

    ``with_dev_grain("Y")`` anchors the annual development buckets to the latest
    diagonal. A quarterly triangle whose latest valuation is a September
    therefore aggregates to development lags 9, 21, 33 and 45, which the
    chain-ladder kernels refuse (they index dev steps as ``dev_lag // 12``), so
    a caller who followed an unqualified "aggregate first" would meet a second
    error on the triangle they had just been told to build. Slicing to a year
    end first is what the message names and what this pins. The kernel refusal
    is matched on the one phrase every wording of it carries, "12-month", because
    the validator's own verdict on anchored ages and the exact refusal text are
    the triangle layer's business, not this test's."""
    n = 15
    rng = np.random.default_rng(7)
    square = np.cumsum(rng.uniform(100.0, 200.0, size=(n, n)), axis=1)
    square[np.arange(n)[:, None] + np.arange(n)[None, :] >= n] = np.nan
    quarterly = make_cohort_triangle(backend_name, square, dev_grain="Q")
    assert str(quarterly.to_pandas()["eval_date"].max().date()) == "2013-09-30"

    straight = quarterly.with_origin_grain("Y").with_dev_grain("Y")
    assert sorted({int(x) for x in straight.to_pandas()["dev_lag"]}) == [9, 21, 33, 45]
    with pytest.raises(ValueError, match="12-month"):
        fit_mack(straight, loss_field="paid_loss")

    sliced = quarterly.as_of("2012-12-31").with_origin_grain("Y").with_dev_grain("Y")
    assert sliced.validate(strict=False) == []
    fit = fit_mack(sliced, loss_field="paid_loss")
    assert fit.dev_grain_months == 12
    assert np.isfinite(one_year_cdr(fit).msep_total)
