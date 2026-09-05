"""kernels.odp_bootstrap + the odp_bootstrap one-year CDR route.

**There are no published digits to tie out to, and this file does not pretend
otherwise.** R's ``CDR.BootChainLadder`` help page prints no output for its
example (unlike ``CDR.MackChainLadder``, whose MW2014 table is the golden
fixture in ``test_cdr.py``), and a bootstrap is stochastic in any case. So the
route is pinned three other ways, in descending order of strength:

1. **Against R's ALGORITHM, exactly.** ``test_rereserving_reproduces_the_r_algorithm``
   transcribes ``getNYCost`` line for line - build the extended triangle, take
   every individual development factor, volume-weight them with the cumulative
   triangle, cumulate the ultimate factors, reproject - and requires it to
   reproduce ``rereserve``'s draws to floating point. That is a structurally
   different computation (a matrix of ratios versus the one-new-observation
   update) fed the identical diagonal, so agreement is not self-consistency.
2. **Against a computed reference for the standard error.** The process-only
   arm has an exactly known input distribution, so ``Var(CDR)`` follows from
   the re-reserving Jacobian by the delta method - no Merz-Wuthrich algebra and
   no second simulation. ``test_process_only_se_matches_the_delta_method``
   requires agreement to Monte Carlo error.
3. **Against invariants**: the fitted values are the Poisson MLE (checked
   against ``england_verrall_odp``'s iterative proportional fit, an independent
   implementation), the Pearson scale reproduces R's formula on a square
   triangle, the three arms' variances decompose, and a triangle with
   no residual variation has no CDR at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibnr.kernels.cdr import (
    ODPBootstrapDiagonal,
    cdr_risk_measures,
    rereserve,
    simulate_one_year_cdr,
)
from ibnr.kernels.densities import POISSON_RATE_MAX, odp_draw
from ibnr.kernels.mack import fit_mack
from ibnr.kernels.odp_bootstrap import (
    ODP_PROCESS_LAWS,
    _od_process_noise,
    draw_next_increments,
    fit_odp_bootstrap,
)

from .conftest import make_cohort_triangle
from .test_cdr import mw2014_matrix

#: Monte Carlo draw budget for the two agreement tests. The relative error of a
#: sample standard deviation is ~1/sqrt(2N), so 60k buys ~0.3% - the tolerances
#: below are that, rounded up to leave room for the delta method's own
#: linearization error (measured at 0.08% on the total at 200k draws).
N_MC = 60_000


def odp_triangle(n_w: int = 8, n_d: int = 8, seed: int = 4) -> np.ndarray:
    """A run-off triangle generated FROM an over-dispersed Poisson model.

    Every increment is a scaled Poisson draw, so all of them are non-negative by
    construction - the ODP family's own support, and the reason ``test_cdr.py``'s
    Mack-generated ``synthetic_triangle`` cannot be reused here: its normal step
    noise routinely produces a small negative increment at the shallow tail
    factors, which this family refuses.
    """
    rng = np.random.default_rng(seed)
    row = np.linspace(9000.0, 12000.0, n_w)
    col = np.array([0.42, 0.26, 0.14, 0.08, 0.05, 0.025, 0.015, 0.01])[:n_d]
    col = col / col.sum()
    phi = 40.0
    cum = np.full((n_w, n_d), np.nan)
    for i in range(n_w):
        running = 0.0
        for j in range(min(n_d, n_w - i)):
            running += phi * rng.poisson(row[i] * col[j] / phi)
            cum[i, j] = running
    return cum


@pytest.fixture(scope="module")
def odp_fit():
    """Fitted on the default backend: this file is about the bootstrap
    arithmetic, and the backend matrix is exercised by the tests below that
    take ``backend_name``."""
    return fit_mack(make_cohort_triangle(None, odp_triangle()), loss_field="paid_loss")


@pytest.fixture(scope="module")
def odp_boot(odp_fit):
    return fit_odp_bootstrap(odp_fit.cum, odp_fit.obs_mask, odp_fit.latest_dev, odp_fit.f)


# -- the deterministic half ---------------------------------------------------


def test_fitted_increments_are_the_poisson_mle(odp_boot):
    """The bootstrap resamples residuals against the chain-ladder fitted
    incrementals, which are also the cross-classified Poisson MLE's fitted means
    (Renshaw & Verrall 1998) - the result that makes the ODP GLM and the chain
    ladder give the same reserve.

    Checked against ``england_verrall_odp.odp_mle_fitted``, which reaches the
    same numbers by iterative proportional fitting: a genuinely different
    algorithm (alternating margin scaling versus one backwards recursion off the
    latest diagonal), so agreement pins both.
    """
    from ibnr.gallery.bayesian.england_verrall_odp.model import odp_mle_fitted

    mask = odp_boot.obs_mask
    w, d = np.nonzero(mask)
    ipf = odp_mle_fitted(w + 1, d + 1, odp_boot.inc[mask], odp_boot.n_w, odp_boot.n_d)
    np.testing.assert_allclose(odp_boot.fitted[mask], ipf[w, d], rtol=1e-9)


def test_fitted_values_reproduce_the_observed_margins(odp_boot):
    """The Poisson MLE's defining property, and the invariant behind the test
    above: fitted row totals and fitted column totals equal the observed ones
    on the triangle's support."""
    mask, fitted, inc = odp_boot.obs_mask, odp_boot.fitted, odp_boot.inc
    got = np.where(mask, fitted, 0.0)
    want = np.where(mask, inc, 0.0)
    np.testing.assert_allclose(got.sum(axis=1), want.sum(axis=1), rtol=1e-9)
    np.testing.assert_allclose(got.sum(axis=0), want.sum(axis=0), rtol=1e-9)


def test_pearson_scale_matches_the_r_formula(odp_boot):
    """R computes ``nobs = n(n+1)/2``, ``scale.factor = nobs - 2n + 1``,
    ``scale.phi = sum(r^2)/scale.factor`` and adjusts the residuals by
    ``sqrt(nobs/scale.factor)``. On a square triangle our general counts - all
    observed cells, and ``p = n_w + n_d - 1`` - must be those same two numbers,
    and the scale and adjustment must follow from them."""
    n = odp_boot.n_d
    assert odp_boot.n_cells == n * (n + 1) // 2
    assert odp_boot.n_params == 2 * n - 1
    dof = odp_boot.n_cells - odp_boot.n_params

    pool = odp_boot.pool_mask
    unscaled = (odp_boot.inc[pool] - odp_boot.fitted[pool]) / np.sqrt(odp_boot.fitted[pool])
    assert odp_boot.phi == pytest.approx(float((unscaled**2).sum() / dof))
    np.testing.assert_allclose(
        odp_boot.residuals[pool], unscaled * np.sqrt(odp_boot.n_cells / dof), rtol=1e-12
    )


def test_negative_increment_is_refused_by_name(backend_name):
    """The ODP family limit, and it is a limit of the family rather than of the
    implementation: R reflects a negative increment through ``sign()``/``abs()``
    and carries on, we refuse. The message must name the cause and point at the
    generator that CAN answer, because roughly half the Schedule P mart's paid
    cohorts land here."""
    cum = odp_triangle()
    cum[3, 2] = cum[3, 1] - 50.0  # one cumulative that goes backwards
    fit = fit_mack(make_cohort_triangle(backend_name, cum), loss_field="paid_loss")
    with pytest.raises(ValueError, match="negative incremental cell"):
        simulate_one_year_cdr(fit, n_draws=10, seed=1, generator="odp_bootstrap")
    try:
        simulate_one_year_cdr(fit, n_draws=10, seed=1, generator="odp_bootstrap")
    except ValueError as exc:
        assert "dev step 3" in str(exc)
        assert "mack" in str(exc)  # the route that still answers here
    # ... and the mack generator does still answer, on the same fit
    assert simulate_one_year_cdr(fit, n_draws=10, seed=1).samples.shape == (10, cum.shape[0] + 1)


# -- the simulation -----------------------------------------------------------


def expected_next_payment(fit) -> np.ndarray:
    """``(n_w,)`` mean payment of the process-only arm, straight off the fit.

    ``resample_residuals=False`` makes the pseudo-triangle the FITTED triangle,
    whose cumulative column ratio is ``ultdf[j]/ultdf[j+1] = f[j]`` exactly, so
    the refit returns the original factors. And the backwards recursion is
    anchored at each origin's own diagonal (``Chat[i,k_i] = ult_i/ultdf[k_i] =
    C[i,k_i]``), so the fitted diagonal IS the observed one. The expected next
    increment therefore collapses to ``C[i,k_i] * (f[k_i] - 1)``.
    """
    k, n_d = fit.latest_dev, fit.n_d
    return np.where(k < n_d - 1, fit.latest * (fit.f[np.minimum(k, n_d - 2)] - 1.0), 0.0)


def test_expectation_only_pseudo_triangle_refits_to_the_original_factors(odp_fit, odp_boot):
    """The identity above, tested: with both noise sources off, every draw is
    that deterministic mean. It is what makes the ``resample_residuals=False``
    arm pure process risk, and what the delta-method reference below rests on."""
    mu = draw_next_increments(
        odp_boot,
        n_draws=3,
        rng=np.random.default_rng(0),
        process_noise=False,
        resample_residuals=False,
    )
    for row in mu:
        np.testing.assert_allclose(row, expected_next_payment(odp_fit), rtol=1e-9, atol=1e-9)


def r_get_ny_cost(fit, payments: np.ndarray) -> np.ndarray:
    """Literal transcription of R ChainLadder's ``getNYCost``, draw by draw.

    Deliberately slow and unvectorized, and deliberately NOT written in
    ``rereserve``'s one-new-observation form: R rebuilds the whole extended
    triangle, takes every individual development factor ``C[i,j+1]/C[i,j]``,
    volume-weights them with the cumulative triangle as weights
    (``colSums(dfs*wghts*include)/colSums(wghts*include)``), cumulates the
    ultimate factors and reprojects. Same answer by a different route, or one of
    the two is wrong.

    Returns ``(n_draws, n_w)`` NYCost = ``NYIBNR + NYPayments``.
    """
    cum, mask, k = fit.cum, fit.obs_mask, fit.latest_dev
    n_w, n_d = cum.shape
    out = np.zeros((payments.shape[0], n_w))
    for r in range(payments.shape[0]):
        # getTriangleNextYear: the ORIGINAL triangle, overwritten at the next diagonal
        tri = np.where(mask, cum, np.nan)
        for i in range(n_w):
            if k[i] < n_d - 1:
                tri[i, k[i] + 1] = tri[i, k[i]] + payments[r, i]
        # getIndivDFs
        with np.errstate(divide="ignore", invalid="ignore"):
            dfs = tri[:, 1:] / tri[:, :-1]
        # getAvDFs, weights = triangleNY
        av = np.ones(n_d - 1)
        for j in range(n_d - 1):
            keep = np.isfinite(dfs[:, j]) & np.isfinite(tri[:, j])
            den = tri[keep, j].sum()
            av[j] = (dfs[keep, j] * tri[keep, j]).sum() / den if den != 0 else 1.0
        # getUltDFs
        ultdf = np.ones(n_d)
        for j in range(n_d - 2, -1, -1):
            ultdf[j] = ultdf[j + 1] * av[j]
        # getUltimates -> NYIBNR + NYPayments
        for i in range(n_w):
            if k[i] >= n_d - 1:
                continue  # R zeroes the fully developed origin
            ny_latest = tri[i, k[i] + 1]
            out[r, i] = (ny_latest * ultdf[k[i] + 1] - ny_latest) + payments[r, i]
    return out


@pytest.mark.parametrize("matrix_name", ["odp", "mw2014"])
def test_rereserving_reproduces_the_r_algorithm(matrix_name):
    """The strongest check this route has: ``rereserve`` fed the bootstrap's own
    simulated payments must equal ``R_i^I - NYCost_i`` from the transcription
    above, for every draw and every origin.

    That identity is also the proof that our CDR is R's quantity:
    ``NYCost = NYUlts - OldLatest``, so ``R^I - NYCost`` telescopes to
    ``Ult^I - Ult^{I+1}`` - the ultimate difference this package reports.
    """
    cum = odp_triangle() if matrix_name == "odp" else mw2014_matrix()
    fit = fit_mack(make_cohort_triangle(None, cum), loss_field="paid_loss")
    boot = fit_odp_bootstrap(fit.cum, fit.obs_mask, fit.latest_dev, fit.f)
    payments = draw_next_increments(boot, n_draws=40, rng=np.random.default_rng(11))

    ours = rereserve(fit, np.where(fit.latest_dev < fit.n_d - 1, fit.latest + payments, 0.0))
    theirs = fit.reserve[None, :] - r_get_ny_cost(fit, payments)
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-8)


def delta_method_cdr_variance(fit, boot) -> np.ndarray:
    """Var(CDR) of the PROCESS-ONLY arm, per origin then in total, by the delta
    method on the exact re-reserving map.

    Justification for using it as a reference. With ``resample_residuals=False``
    the pseudo-triangle is the fitted triangle, so the refit factors are exactly
    ``f`` and the simulated payment of origin ``i`` is an independent draw with
    a KNOWN mean ``mu_i = Chat[i,k_i](f[k_i] - 1)`` and a KNOWN variance
    ``phi*|mu_i|`` (both process laws are moment-matched to it). ``rereserve`` is
    a deterministic function of those payments, so linearizing it about their
    mean gives ``Var(CDR) = J diag(phi|mu|) J^T``. The Jacobian is taken by
    central differences of ``rereserve`` itself, so nothing here reuses the
    simulation's RNG, its estimator, or Merz-Wuthrich's algebra - the only
    shared ingredient is the map being linearized, which is the thing under
    test only through its derivative.

    Exact to first order; the residual is the map's curvature, measured at 0.08%
    on the total for this fixture (200k draws, three seeds).
    """
    n_w, n_d = fit.n_w, fit.n_d
    k = fit.latest_dev
    open_ = k < n_d - 1
    mu = expected_next_payment(fit)
    var_payment = boot.phi * np.abs(mu)

    x0 = np.where(open_, fit.latest + mu, 0.0)
    jac = np.zeros((n_w, n_w))  # d CDR_i / d x_j
    for j in np.nonzero(open_)[0]:
        h = max(abs(x0[j]) * 1e-6, 1e-6)
        plus, minus = x0.copy(), x0.copy()
        plus[j] += h
        minus[j] -= h
        moved = rereserve(fit, np.vstack([plus, minus]))
        jac[:, j] = (moved[0] - moved[1]) / (2 * h)
    per_origin = (jac**2 * var_payment[None, :]).sum(axis=1)
    return np.append(per_origin, (jac.sum(axis=0) ** 2 * var_payment).sum())


def test_process_only_se_matches_the_delta_method(odp_fit, odp_boot):
    """The computed reference above, against the simulation. Tolerances are
    Monte Carlo error at ``N_MC`` draws plus the linearization residual, both
    stated in the docstrings - not fitted to the observed numbers."""
    reference = np.sqrt(delta_method_cdr_variance(odp_fit, odp_boot))
    pred = simulate_one_year_cdr(
        odp_fit,
        n_draws=N_MC,
        seed=5,
        generator=ODPBootstrapDiagonal(resample_residuals=False),
    )
    simulated = pred.samples.std(axis=0, ddof=1)
    assert simulated[-1] == pytest.approx(reference[-1], rel=0.02)
    live = reference[:-1] > 0
    np.testing.assert_allclose(simulated[:-1][live], reference[:-1][live], rtol=0.04)
    # the fully developed origin cannot move under any generator
    assert simulated[0] == 0.0


def test_the_three_arms_variances_decompose(odp_fit):
    """R never simulates its process arm: ``CDR.BootChainLadder`` reports
    ``CDR.Process.S.E = sqrt(CDR.S.E^2 - CDR.Param.S.E^2)``, which is only
    legitimate if the two sources compose in quadrature. We simulate all three,
    so the identity becomes a test of the risk-source switches rather than an
    assumption behind a reported number."""
    sd = {}
    for label, kwargs in (
        ("total", {}),
        ("parameter", {"process_noise": False}),
        ("process", {"resample_residuals": False}),
    ):
        pred = simulate_one_year_cdr(
            odp_fit, n_draws=N_MC, seed=3, generator=ODPBootstrapDiagonal(**kwargs)
        )
        sd[label] = pred.samples[:, -1].std(ddof=1)
    assert np.hypot(sd["parameter"], sd["process"]) == pytest.approx(sd["total"], rel=0.02)
    # and each arm is strictly smaller than the total it composes into
    assert 0 < sd["parameter"] < sd["total"]
    assert 0 < sd["process"] < sd["total"]


@pytest.mark.parametrize("law", ODP_PROCESS_LAWS)
def test_zero_residual_triangle_has_no_cdr(backend_name, law):
    """A triangle developing by exactly constant factors fits itself perfectly,
    so the Pearson scale collapses to rounding error (3.3e-29 here, measured)
    and both the resampling and the process noise vanish.

    This is the branch :data:`POISSON_RATE_MAX` exists for:
    ``phi * Poisson(mu/phi)`` at that scale asks numpy for a rate of 1e30 and
    gets a ``ValueError``, where the honest answer is a point mass at the mean.
    Exercised on both laws, since only one of them has the limit."""
    factors = [1.5, 1.2, 1.1]
    cum = np.full((4, 4), np.nan)
    for i in range(4):
        cum[i, 0] = 100.0 * (i + 1)
        for j in range(3 - i):
            cum[i, j + 1] = cum[i, j] * factors[j]
    fit = fit_mack(make_cohort_triangle(backend_name, cum), loss_field="paid_loss")
    boot = fit_odp_bootstrap(fit.cum, fit.obs_mask, fit.latest_dev, fit.f)
    assert 0.0 <= boot.phi < 1e-20
    pred = simulate_one_year_cdr(
        fit, n_draws=100, seed=2, generator=ODPBootstrapDiagonal(process=law)
    )
    np.testing.assert_allclose(pred.samples, 0.0, atol=1e-8)


# -- the shared over-dispersed Poisson draw -----------------------------------


def _numpy_accepts(rate: float) -> bool:
    """Does ``Generator.poisson`` take this rate, or refuse it by raising?"""
    try:
        np.random.default_rng(0).poisson(np.array([rate]))
    except ValueError:
        return False
    return True


def test_poisson_rate_cap_is_numpys():
    """:data:`POISSON_RATE_MAX` must be the largest rate numpy will draw at,
    found by bisection rather than trusted from a comment.

    numpy does not export its ``POISSON_LAM_MAX``, and copying it by hand is
    exactly how this went wrong: the first version of this constant was
    9.223372036854776e18, the int64 maximum, while numpy's own value is
    ``int64max - 10 * sqrt(int64max)`` - about 30 billion lower. Every rate in
    between passed the check and then made numpy raise, which is this defect's
    whole kernel half.

    Mutation (verified): setting POISSON_RATE_MAX back to the int64 maximum."""
    lo, hi = 1e18, 1e19
    assert _numpy_accepts(lo) and not _numpy_accepts(hi)
    for _ in range(200):
        mid = lo + (hi - lo) / 2.0
        if mid <= lo or mid >= hi:
            break
        if _numpy_accepts(mid):
            lo = mid
        else:
            hi = mid
    assert np.nextafter(lo, np.inf) == hi  # bisection ran to adjacent doubles
    assert lo == POISSON_RATE_MAX
    assert _numpy_accepts(POISSON_RATE_MAX)
    assert not _numpy_accepts(np.nextafter(POISSON_RATE_MAX, np.inf))
    assert float(np.iinfo(np.int64).max) > POISSON_RATE_MAX


def test_odp_draw_refuses_a_non_finite_mean():
    """A mean that is not a finite non-negative number never came from the
    model, and the error says which defect it is, by name and by count.

    The two are refused separately because they are not the same mistake. A
    mean that is not finite is a parameter sample that overflowed before any
    draw was asked for. A negative mean is a sign the caller was supposed to
    handle - by flooring, as the gallery entries do, or by reflecting, as the
    bootstrap kernel does - so that message names the value and the remedy
    rather than blaming an overflow that did not happen.

    numpy's own answer here is ``lam value too large``, which names neither
    how many cells are broken nor that the cause is upstream of the draw.

    Mutations (verified): dropping the non-finite check (numpy raises its own
    message); dropping the negative-mu check (numpy raises `lam < 0`);
    dropping the phi check (a NaN phi silently returns every cell at its mean,
    because ``mu / nan`` is never below the cap); putting the ``phi == 0``
    shortcut ahead of the checks (a broken mean is then returned as itself);
    giving both refusals the one overflow message."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="2 of 4 are not finite.*overflowed"):
        odp_draw(rng, np.array([100.0, np.inf, 50.0, np.nan]), 2.0)
    with pytest.raises(ValueError, match="1 of 2 are negative, the smallest -1.5"):
        odp_draw(rng, np.array([100.0, -1.5]), 2.0)
    for bad_phi in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="phi"):
            odp_draw(rng, np.array([100.0, 50.0]), bad_phi)
    # the checks run first, so no dispersion at all is not a way past them
    with pytest.raises(ValueError, match="1 of 2"):
        odp_draw(rng, np.array([100.0, np.nan]), 0.0)


def test_odp_draw_caps_only_the_cells_past_the_rate():
    """A mixed array: the cells numpy can draw at are drawn, the ones past the
    cap come back at their mean, and the generator is consumed ONLY for the
    live cells - so a fit with nothing capped reads the same random numbers it
    always did.

    ``phi = 1`` so the mean IS the rate and the boundary can be named exactly:
    one cell sits on the cap (drawn) and one a single double above it (not).
    The other boundary is in there too: a mean of exactly 0 is a legal Poisson
    rate, drawn like any other, and is not to be mistaken for a bad mean.

    The generator's own STATE is compared, not only the values it produced. A
    Poisson draw takes a variable number of random doubles, so an extra one
    usually shifts everything after it - but only usually: the first version of
    this test compared values alone, and the mutation that hands the capped
    cells to the generator anyway passed it, because the shifted draw landed on
    the same integer by chance. Comparing states cannot land on anything.

    Mutations (verified): passing the whole array to ``rng.poisson`` (numpy
    raises); handing the capped cells to the generator and discarding the
    result afterwards; the boundary tested with ``<`` instead of ``<=``;
    returning the caller's own array at ``phi == 0``; refusing a mean of
    exactly 0 along with the negative ones."""
    phi = 1.0
    mu = np.array(
        [
            3.0,
            POISSON_RATE_MAX,
            7.0,
            np.nextafter(POISSON_RATE_MAX, np.inf),
            11.0,
            1e30,
            0.0,
            5.0,
            40.0,
            2.5,
        ]
    )
    live = np.array([True, True, True, False, True, False, True, True, True, True])

    rng = np.random.default_rng(7)
    got = odp_draw(rng, mu, phi)
    reference = np.random.default_rng(7)
    want_live = phi * reference.poisson(mu[live] / phi)
    assert np.array_equal(got[live].view(np.uint8), want_live.view(np.uint8))
    assert np.array_equal(got[~live].view(np.uint8), mu[~live].view(np.uint8))
    assert rng.bit_generator.state == reference.bit_generator.state
    assert got[6] == 0.0  # the zero-mean cell: drawn, and Poisson(0) is 0

    # phi == 0 is the same statement with no dispersion left at all: the mean,
    # and not one random number taken
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state
    zero = odp_draw(rng, mu, 0.0)
    assert np.array_equal(zero.view(np.uint8), mu.view(np.uint8))
    assert rng.bit_generator.state == before
    assert zero is not mu  # a copy, never the caller's array


@pytest.mark.parametrize("law", ODP_PROCESS_LAWS)
def test_process_noise_reflects_the_sign_and_leaves_a_zero_mean_alone(law):
    """The two things the bootstrap's noise step does around the shared draw,
    tested directly rather than through a CDR that averages them away.

    1. **The sign reflection**, R's ``processTriangle``: the draw is taken at
       ``|mu|`` and the sign put back, so a resampled triangle that refits to
       ``f* < 1`` and projects a negative increment still gets noise about its
       own mean instead of an error. Nothing else in this file sees it - the
       fixtures never project a negative increment - so without this test the
       reflection could be deleted and every other test would pass.
    2. **A zero mean comes back at 0, and no random number is spent on it.**

    Point 2 is pinned here and NOT attributed to the ``live`` mask, which is an
    optimization with no observable effect: numpy answers a Poisson rate of 0,
    and a gamma shape of 0, with 0 and takes nothing off the generator, so
    widening the mask to every cell leaves both the values and the generator's
    state identical (mutation run, escaped as it must). What this test pins is
    the answer and the consumption, which is what a caller can see.

    The capped cell is here as well, so this also shows the reflection and the
    cap composing: ``od_poisson`` returns the mean at ``-1e30``, sign and all.

    Mutations (verified): dropping the ``sign *`` on either law's branch."""
    phi = 18.55
    mu = np.array([-100.0, 100.0, 0.0, -1e30])
    live = mu != 0.0

    rng = np.random.default_rng(3)
    got = _od_process_noise(rng, mu, phi, law=law)

    reference = np.random.default_rng(3)
    magnitude = np.abs(mu[live])
    if law == "od_poisson":
        drawn = magnitude.copy()
        under = magnitude / phi <= POISSON_RATE_MAX
        assert under.tolist() == [True, True, False]  # the case really is mixed
        drawn[under] = phi * reference.poisson(magnitude[under] / phi)
    else:
        drawn = reference.gamma(shape=magnitude / phi, scale=phi)
    want = mu.copy()
    want[live] = np.sign(mu[live]) * drawn

    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))
    assert got[0] < 0.0 and got[1] > 0.0  # reflected, not folded onto one side
    assert got[0] != mu[0] and got[1] != mu[1]  # and genuinely drawn, not the mean
    assert got[2] == 0.0
    assert rng.bit_generator.state == reference.bit_generator.state


def test_process_laws_are_moment_matched(odp_fit):
    """Both laws have mean ``mu`` and variance ``phi*mu``, so they may move the
    tail of the CDR but not its scale. R's ``BootChainLadder`` defaults to
    ``gamma`` and ours to ``od_poisson``; this is what makes that difference a
    tail choice rather than a different answer.

    Parametrized over ``ODP_PROCESS_LAWS`` itself so a law cannot join the tuple
    without joining this check."""
    sd = {
        law: simulate_one_year_cdr(
            odp_fit, n_draws=N_MC, seed=13, generator=ODPBootstrapDiagonal(process=law)
        )
        .samples[:, -1]
        .std(ddof=1)
        for law in ODP_PROCESS_LAWS
    }
    assert set(sd) == set(ODP_PROCESS_LAWS)
    assert max(sd.values()) == pytest.approx(min(sd.values()), rel=0.03)


def test_cdr_distribution_layout_and_capital(odp_fit):
    """The bootstrap route returns the same object as the Mack route, so
    everything downstream - the total column, VaR/TVaR on the loss - works
    unchanged."""
    pred = simulate_one_year_cdr(odp_fit, n_draws=20_000, seed=8, generator="odp_bootstrap")
    assert pred.n_targets == odp_fit.n_w + 1
    assert list(pred.targets["label"])[-1] == "total"
    np.testing.assert_allclose(pred.samples[:, -1], pred.samples[:, :-1].sum(axis=1), rtol=1e-12)
    table = cdr_risk_measures(pred, levels=(0.995,))
    assert (table["tvar_0.995"] >= table["var_0.995"]).all()
    assert table["var_0.995"].iloc[0] == pytest.approx(0.0)


def test_the_bootstrap_cdr_is_not_exactly_centred_on_zero(odp_fit):
    """The Mack assumption ``rereserve`` cannot shed, made visible.

    ``E[CDR | D_I] = 0`` is a consequence of Mack's conditional moments, and it
    is what lets ``simulated_msep``'s mean square about zero be the risk
    measure. A residual bootstrap centres its diagonal on the pseudo-triangle's
    refit, so its mean drifts - small (well under a tenth of a standard
    deviation here) but real, which is why R reports ``sd()`` for this route and
    an msep for the Mack one. The test pins the size of the drift so a change
    in it cannot pass unnoticed.
    """
    mack = simulate_one_year_cdr(odp_fit, n_draws=N_MC, seed=21).samples[:, -1]
    boot = simulate_one_year_cdr(odp_fit, n_draws=N_MC, seed=21, generator="odp_bootstrap").samples[
        :, -1
    ]
    assert abs(mack.mean()) < 0.02 * mack.std(ddof=1)
    assert abs(boot.mean()) < 0.10 * boot.std(ddof=1)


# -- refusals -----------------------------------------------------------------


def test_both_switches_off_is_refused():
    with pytest.raises(ValueError, match="no risk source left"):
        ODPBootstrapDiagonal(process_noise=False, resample_residuals=False)


def test_rejects_unknown_process_law():
    with pytest.raises(ValueError, match="process must be one of"):
        ODPBootstrapDiagonal(process="lognormal")


def test_refuses_a_triangle_with_no_residual_degrees_of_freedom(backend_name):
    """``phi`` divides by ``n_cells - n_params``; a 2x2 triangle has 3 cells and
    3 parameters, so there is nothing to estimate the scale from."""
    cum = np.array([[100.0, 150.0], [120.0, np.nan]])
    fit = fit_mack(make_cohort_triangle(backend_name, cum), loss_field="paid_loss")
    with pytest.raises(ValueError, match="residual degrees of freedom"):
        simulate_one_year_cdr(fit, n_draws=10, seed=1, generator="odp_bootstrap")


def test_a_zero_diagonal_cell_is_fine_for_the_bootstrap(backend_name):
    """Where the Mack route refuses. Mack's conditional variance is proportional
    to the diagonal cell, so a zero there is a hard error
    (``require_positive_open_diagonals``); the bootstrap's variance comes from
    the fitted mean instead, so an accident year with nothing paid at 12 months
    keeps a one-year CDR. A real capability difference between the two
    generators, not an accident of implementation."""
    cum = odp_triangle()
    cum[7, 0] = 0.0  # youngest origin, nothing paid yet
    assert np.isnan(cum[7, 1])  # and it is the whole of that origin's data
    fit = fit_mack(make_cohort_triangle(backend_name, cum), loss_field="paid_loss")
    with pytest.raises(ValueError, match="non-positive cumulative on the latest diagonal"):
        simulate_one_year_cdr(fit, n_draws=10, seed=1)
    pred = simulate_one_year_cdr(fit, n_draws=200, seed=1, generator="odp_bootstrap")
    assert np.isfinite(pred.samples).all()
