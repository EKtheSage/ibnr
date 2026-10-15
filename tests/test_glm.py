"""``methods.tweedie_glm`` and ``kernels.glm``: a Tweedie GLM on a triangle's increments.

Checked here:

- power 1 with a log link and origin and development factors is the
  volume-weighted chain ladder, to 1e-10, by both projections, against ibnr's
  own ``chain_ladder`` and (with the ``tieout`` marker) chainladder-python;
- every power (0, 1, 1.5, 2) against R's ``glm`` with ``statmod::tweedie`` on
  four public triangles, frozen in ``data/tweedie_glm_r.json`` by
  ``scripts/r/tweedie_glm_reference.R`` (CI has no R): reserves, coefficients,
  standard errors, deviance and dispersion; the identity link and a calendar
  trend too;
- a second solver (scikit-learn's Newton-Cholesky) and chainladder-python's
  pipeline with its penalty actually removed;
- that the answer scales exactly with the units (there is no penalty, and the
  stopping rule is relative), every option reaching the fit, the zero rows and
  columns, the refusals, the caller's labels and the nulls;
- the clrd sweep: on every paid triangle with non-negative increments and no
  zero-to-positive link the GLM equals the chain ladder, and every other one is
  answered or refused with a ``Refusal``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ibnr import methods
from ibnr.errors import Refusal
from ibnr.kernels.glm import TweedieSpec, fit_tweedie_grid
from ibnr.kernels.grid import grid_from_columns

R = json.loads((Path(__file__).parent / "data" / "tweedie_glm_r.json").read_text(encoding="utf-8"))
NAMES = ("GenIns", "UKMotor", "ABC", "MW2014")
POWERS = (0.0, 1.0, 1.5, 2.0)

# raa, the Mack (1993) triangle: accident years 1981 to 1990. Its one negative
# increment is 1982 at 84 months (15,496 after 15,599).
RAA = [
    [5012, 8269, 10907, 11805, 13539, 16181, 18009, 18608, 18662, 18834],
    [106, 4285, 5396, 10666, 13782, 15599, 15496, 16169, 16704],
    [3410, 8992, 13873, 16141, 18735, 22214, 22863, 23466],
    [5655, 11555, 15766, 21266, 23425, 26083, 27067],
    [1092, 9565, 15836, 22169, 25955, 26180],
    [1513, 6445, 11702, 12935, 15852],
    [557, 4020, 10946, 12314],
    [1351, 6947, 13112],
    [3133, 5395],
    [2063],
]


def rows_of(name: str) -> list[list[float]]:
    return R["triangles"][name]["cumulative"]


def first_year(name: str) -> int:
    first = R["triangles"][name]["first_origin"]
    return first if first >= 1000 else 2001  # R numbers GenIns and MW2014 from 1


def cells_of(rows, first: int = 2001, *, step: int = 12, scale: float = 1.0, label=None):
    """The cells of a staircase as a pyarrow Table, integer years by default."""
    origin, lag, value = [], [], []
    for i, row in enumerate(rows):
        for j, amount in enumerate(row):
            origin.append(first + i if label is None else label(i))
            lag.append(step * (j + 1))
            value.append(float(amount) * scale)
    return pa.table({"origin_period": origin, "dev_lag": lag, "value": value})


def named(name: str, **kw) -> pa.Table:
    return cells_of(rows_of(name), first_year(name), **kw)


def column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=float)


def grid_of(rows, *, step: int = 12) -> dict:
    origin, lag, value = [], [], []
    for i, row in enumerate(rows):
        for j, amount in enumerate(row):
            origin.append(np.datetime64(dt.date(2001 + i, 1, 1), "D"))
            lag.append(step * (j + 1))
            value.append(float(amount))
    return grid_from_columns(
        np.array(origin),
        np.array(lag),
        np.array(value),
        dev_grain_months=step,
        measure="cumulative",
    )


def refusal_of(thunk) -> Refusal:
    with pytest.raises(Refusal) as caught:
        thunk()
    assert type(caught.value) is Refusal
    return caught.value


# -- 1. power 1 is the chain ladder ------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("projection", ["pattern", "increments"])
def test_power_one_is_the_chain_ladder(name, projection):
    """Mutation: difference from the second column only (the first increment
    dropped), or drop the first development column's increments: this fails."""
    cells = named(name)
    glm = methods.tweedie_glm(cells, projection=projection)
    chain = methods.chain_ladder(cells, zero_cells="observed")
    np.testing.assert_allclose(
        column(glm.origins, "ultimate"), column(chain.origins, "ultimate"), rtol=1e-10
    )
    np.testing.assert_allclose(
        column(glm.origins, "model_ibnr"), column(chain.origins, "ibnr"), rtol=1e-9, atol=1e-6
    )
    factors = glm.development["factor"].to_pylist()
    np.testing.assert_allclose(
        factors[:-1], chain.development["factor"].to_pylist()[:-1], rtol=1e-10
    )
    np.testing.assert_allclose(
        column(glm.development, "cdf"), column(chain.development, "cdf"), rtol=1e-10
    )


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["genins", "ukmotor", "abc", "mw2014"])
def test_power_one_matches_chainladder_python(name):
    cl = pytest.importorskip("chainladder")
    sample = cl.load_sample(name)
    grid = np.asarray(sample.values[0, 0], dtype=float)
    rows = [list(row[~np.isnan(row)]) for row in grid]
    first = int(str(sample.origin[0])[:4])
    result = methods.tweedie_glm(cells_of(rows, first))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = cl.Chainladder().fit(cl.Development().fit_transform(sample))
    expected = np.asarray(reference.ultimate_.values[0, 0, :, -1], dtype=float)
    np.testing.assert_allclose(column(result.origins, "ultimate"), expected, rtol=1e-10)


# -- 2. every power against R -------------------------------------------------------


@pytest.mark.tieout
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("index", range(len(POWERS)), ids=[f"p{p:g}" for p in POWERS])
def test_every_power_matches_r(name, index):
    """R's glm with statmod::tweedie(var.power = p, link.power = 0), origin and
    development factors. The reserves are the fitted future increments (R's
    ``glmReserve`` route), compared to 1e-6 of the largest origin's; measured
    at most 9.3e-8, where R stops at a relative deviance change of 1e-12 and
    ibnr later. Mutation: weights ``mu ** (2 - 1)`` whatever the power, or a
    ridge penalty on the coefficients: this fails at 1.5 and 2 (and every power)."""
    fit_r = R["triangles"][name]["fits"][index]
    assert fit_r["power"] == POWERS[index] and fit_r["converged"]
    result = methods.tweedie_glm(named(name), power=POWERS[index], projection="increments")
    reserve = np.asarray(fit_r["reserve_by_origin"])
    got = column(result.origins, "model_ibnr")
    assert np.max(np.abs(got - reserve)) <= 1e-6 * np.max(np.abs(reserve))
    # the two routes sum the same increments in different orders: on Linux
    # MW2014's second origin (a reserve of about 1.09) differs by 1.2e-12 of
    # itself, so the tolerance is scaled to the largest origin's reserve
    np.testing.assert_allclose(
        column(result.origins, "ibnr"), got, rtol=1e-12, atol=1e-12 * np.max(np.abs(got))
    )
    coefficients = result.coefficients
    assert coefficients.num_rows == len(fit_r["coefficient_names"])
    np.testing.assert_allclose(column(coefficients, "estimate"), fit_r["coefficients"], atol=2e-6)
    np.testing.assert_allclose(column(coefficients, "std_error"), fit_r["std_errors"], rtol=2e-6)
    totals = result.totals.to_pylist()[0]
    assert totals["deviance"] == pytest.approx(fit_r["deviance"], rel=1e-9)
    assert totals["pearson_chi2"] == pytest.approx(fit_r["pearson_chi2"], rel=1e-6)
    assert totals["dispersion"] == pytest.approx(fit_r["dispersion"], rel=1e-6)
    assert totals["n_observed"] - totals["n_parameters"] == fit_r["df_residual"]
    # the fitted increment in every cell, observed or not
    fitted = column(result.cells, "fitted_increment")
    assert np.max(np.abs(fitted - fit_r["fitted"])) <= 1e-6 * np.max(np.abs(fitted))


def test_the_dispersion_divides_by_the_residual_degrees_of_freedom():
    """GenIns at power 1: R's summary(glm)$dispersion is 52,601.4, the Pearson
    chi-squared over 55 cells less 19 terms. Mutation: divide by the cells (55):
    this fails."""
    totals = methods.tweedie_glm(named("GenIns")).totals.to_pylist()[0]
    assert (totals["n_observed"], totals["n_parameters"]) == (55, 19)
    assert totals["dispersion"] == pytest.approx(52601.36, rel=1e-6)
    assert totals["dispersion"] == pytest.approx(totals["pearson_chi2"] / 36, rel=1e-14)


@pytest.mark.tieout
def test_the_identity_link_matches_r_origin_by_origin():
    """GenIns at power 0 with the identity link: R's reserves total 21,522,632.61.
    The pattern differs by origin, so ``development`` has nulls and each
    origin's ultimate is its latest times its own cdf in ``cells``. Mutation:
    use the first origin's cdf for every origin: this fails."""
    reference = R["extras"]["genins_identity_p0"]
    for projection in ("pattern", "increments"):
        result = methods.tweedie_glm(
            named("GenIns"), power=0, link="identity", projection=projection
        )
        reserve = np.asarray(reference["reserve_by_origin"])
        got = column(result.origins, "model_ibnr")
        assert np.max(np.abs(got - reserve)) <= 1e-8 * np.max(reserve)
        assert result.totals["model_ibnr"][0].as_py() == pytest.approx(21522632.61, abs=0.01)
        assert result.development["factor"].null_count == 10
        assert result.development["cdf"].null_count == 10
        cells = result.cells
        at_latest = pc.equal(
            cells["dev_lag"],
            pa.chunked_array(
                [np.repeat(column(result.origins, "latest_dev_lag"), 10).astype(np.int64)]
            ),
        )
        cdf = column(cells.filter(at_latest), "cdf")
        latest = column(result.origins, "latest")
        if projection == "pattern":
            np.testing.assert_allclose(column(result.origins, "ultimate"), latest * cdf, rtol=1e-12)
        assert len(set(np.round(column(cells, "factor")[:9], 12))) > 1
        # the coefficients, their standard errors (in the units of the amounts)
        # and the dispersion. Mutation: leave the standard errors in the fit's
        # internal units (divided by the largest increment): this fails.
        np.testing.assert_allclose(
            column(result.coefficients, "estimate"), reference["coefficients"], rtol=1e-8
        )
        np.testing.assert_allclose(
            column(result.coefficients, "std_error"), reference["std_errors"], rtol=1e-8
        )
        dispersion = result.totals["dispersion"][0].as_py()
        assert dispersion == pytest.approx(reference["dispersion"], rel=1e-10)


IDENTITY_POWERS = (1.0, 1.5, 2.0)


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["GenIns", "UKMotor", "ABC"])
@pytest.mark.parametrize("index", range(3), ids=[f"p{p:g}" for p in IDENTITY_POWERS])
def test_the_identity_link_matches_r_at_every_positive_power(name, index):
    """R's glm with statmod::tweedie(var.power = p, link.power = 1), origin and
    development factors. Mutation: weights of 1 whatever the power (the normal
    fit), or the standard errors left in the fit's internal units: this fails."""
    fit_r = R["extras"]["identity"][name][index]
    assert fit_r["power"] == IDENTITY_POWERS[index] and fit_r["converged"]
    result = methods.tweedie_glm(
        named(name), power=IDENTITY_POWERS[index], link="identity", projection="increments"
    )
    reserve = np.asarray(fit_r["reserve_by_origin"])
    got = column(result.origins, "model_ibnr")
    assert np.max(np.abs(got - reserve)) <= 1e-7 * np.max(np.abs(reserve))
    estimate = column(result.coefficients, "estimate")
    # R stops at a relative deviance change of 1e-12, before ibnr: measured
    # at most 4e-8 of the largest estimate apart
    np.testing.assert_allclose(
        estimate, fit_r["coefficients"], rtol=1e-6, atol=1e-7 * np.max(np.abs(estimate))
    )
    np.testing.assert_allclose(
        column(result.coefficients, "std_error"), fit_r["std_errors"], rtol=1e-6
    )
    totals = result.totals.to_pylist()[0]
    assert totals["deviance"] == pytest.approx(fit_r["deviance"], rel=1e-8)
    assert totals["dispersion"] == pytest.approx(fit_r["dispersion"], rel=1e-6)
    fitted = column(result.cells, "fitted_increment")
    assert np.max(np.abs(fitted - fit_r["fitted"])) <= 2e-7 * np.max(np.abs(fitted))


@pytest.mark.tieout
def test_mw2014_under_the_identity_link_is_refused_where_r_fails_or_goes_below_zero():
    """R finds no valid coefficients at powers 1 and 1.5, and at power 2 returns
    a fit whose future increments are negative (a total reserve of -29,932),
    which a gamma mean cannot be. ibnr refuses all three by name."""
    runs = R["extras"]["identity"]["MW2014"]
    assert "no valid set of coefficients" in runs[0]["error"]
    assert "no valid set of coefficients" in runs[1]["error"]
    assert runs[2]["converged"] and sum(runs[2]["reserve_by_origin"]) < 0
    for power in IDENTITY_POWERS:
        refusal = refusal_of(
            lambda power=power: methods.tweedie_glm(named("MW2014"), power=power, link="identity")
        )
        assert (refusal.reason, refusal.option) == ("negative_fitted_mean", "link"), power
        assert "link='log'" in str(refusal)


@pytest.mark.tieout
@pytest.mark.parametrize("index", range(2), ids=["p1", "p1.5"])
def test_a_step_that_leaves_the_positive_means_is_halved(index):
    """A 4x4 triangle whose identity-link fit takes a Fisher step past zero and
    has to halve it back towards the last valid one, as R's glm.fit does ("step
    size truncated due to divergence"). The fit converges slowly and R stops at
    a relative deviance change of 1e-12, before ibnr, so the fitted increments
    agree to 5e-7 of the largest; ibnr's are the maximum-likelihood ones, whose
    score (the derivative of the log-likelihood in each coefficient) is 0.
    Mutation: no halving, or halving towards zero instead of towards the last
    step: this fails."""
    fit_r = R["extras"]["halving_identity"][index]
    assert fit_r["converged"]
    rows = R["extras"]["halving_triangle"]
    power = fit_r["power"]
    result = methods.tweedie_glm(
        cells_of(rows), power=power, link="identity", projection="increments"
    )
    fitted = column(result.cells, "fitted_increment")
    assert np.max(np.abs(fitted - fit_r["fitted"])) <= 5e-7 * np.max(fitted)
    reserve = column(result.origins, "model_ibnr")
    assert np.max(np.abs(reserve - fit_r["reserve_by_origin"])) <= 5e-7 * np.max(reserve)
    # the score at ibnr's fit: sum over the observed cells of x * (y - mu) / mu ** p
    n = len(rows)
    observed = [(i, j) for i in range(n) for j in range(len(rows[i]))]
    y = np.array([rows[i][j] - (rows[i][j - 1] if j else 0.0) for i, j in observed])
    mu = fitted.reshape(n, n)[tuple(np.array(observed).T)]
    x = np.array(
        [
            [1.0] + [float(i == k) for k in range(1, n)] + [float(j == k) for k in range(1, n)]
            for i, j in observed
        ]
    )
    score = x.T @ ((y - mu) / mu**power)
    assert np.max(np.abs(score)) <= 1e-8


@pytest.mark.tieout
def test_a_calendar_trend_matches_r_and_a_linear_origin_trend():
    """GenIns at power 1, development factors and a calendar trend: R's
    coefficient is 0.03483559. Under the log link the same means come from a
    straight line across origins (fitted here with plain numpy), because
    ``g * (i + j)`` is ``g * i`` plus a development term."""
    reference = R["extras"]["genins_calendar_p1"]
    result = methods.tweedie_glm(named("GenIns"), origin="none", calendar="trend")
    trend = result.coefficients.filter(pc.equal(result.coefficients["term"], "calendar"))
    assert trend.num_rows == 1
    assert trend["estimate"][0].as_py() == pytest.approx(0.0348355857961, rel=1e-9)
    assert trend["std_error"][0].as_py() == pytest.approx(reference["std_errors"][-1], rel=1e-6)
    fitted = column(result.cells, "fitted_increment")
    np.testing.assert_allclose(fitted, reference["fitted"], rtol=1e-9)

    # the same means from development factors and a linear origin trend
    rows = rows_of("GenIns")
    n = len(rows)
    cells = [(i, j) for i in range(n) for j in range(len(rows[i]))]
    y = np.array([rows[i][j] - (rows[i][j - 1] if j else 0.0) for i, j in cells])
    x = np.array([[1.0, i] + [float(j == k) for k in range(1, n)] for i, j in cells])
    beta = np.linalg.lstsq(x, np.log(y), rcond=None)[0]
    for _ in range(50):  # Poisson IRLS, log link
        mu = np.exp(x @ beta)
        z = x @ beta + (y - mu) / mu
        a = x * np.sqrt(mu)[:, None]
        beta = np.linalg.lstsq(a, np.sqrt(mu) * z, rcond=None)[0]
    everywhere = np.array(
        [[1.0, i] + [float(j == k) for k in range(1, n)] for i in range(n) for j in range(n)]
    )
    np.testing.assert_allclose(fitted, np.exp(everywhere @ beta), rtol=1e-8)
    assert beta[1] == pytest.approx(0.0348355857961, rel=1e-7)


@pytest.mark.tieout
def test_r_s_glm_reserve_agrees_with_the_increments_route():
    """ChainLadder::glmReserve, a second R route, rounds its totals to whole units."""
    for name in NAMES:
        for power, total in R["triangles"][name]["glm_reserve_total_ibnr"].items():
            result = methods.tweedie_glm(named(name), power=float(power), projection="increments")
            assert result.totals["ibnr"][0].as_py() == pytest.approx(total, abs=40.0), name


# -- 3. other references ------------------------------------------------------------


@pytest.mark.tieout
@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
def test_a_second_solver_gives_the_same_fitted_means(power):
    """scikit-learn's TweedieRegressor with no penalty and its Newton-Cholesky
    solver, on the same design. Not at power 0, where it does not converge."""
    linear_model = pytest.importorskip("sklearn.linear_model")
    rows = rows_of("GenIns")
    n = len(rows)
    cells = [(i, j) for i in range(n) for j in range(len(rows[i]))]
    y = np.array([rows[i][j] - (rows[i][j - 1] if j else 0.0) for i, j in cells])
    x = np.array(
        [
            [float(i == k) for k in range(1, n)] + [float(j == k) for k in range(1, n)]
            for i, j in cells
        ]
    )
    model = linear_model.TweedieRegressor(
        power=power, link="log", alpha=0.0, solver="newton-cholesky", tol=1e-12, max_iter=1000
    ).fit(x, y)
    result = methods.tweedie_glm(named("GenIns"), power=power)
    observed = result.cells.filter(result.cells["observed"])
    np.testing.assert_allclose(column(observed, "fitted_increment"), model.predict(x), rtol=1e-7)


@pytest.mark.tieout
@pytest.mark.parametrize("power", [1.5, 2.0])
def test_the_pattern_route_is_chainladders_pipeline_without_its_penalty(power):
    """chainladder-python's TweedieGLM with ``alpha`` really passed (0) and a
    tight tolerance gives the pattern route. Its own TweedieGLM ignores alpha
    and always fits scikit-learn's penalty of 1.0."""
    cl = pytest.importorskip("chainladder")
    pytest.importorskip("sklearn")
    from chainladder.development.learning import DevelopmentML
    from chainladder.utils.utility_functions import PatsyFormula
    from sklearn.linear_model import TweedieRegressor
    from sklearn.pipeline import Pipeline

    sample = cl.load_sample("genins")
    model = TweedieRegressor(
        link="log", power=power, max_iter=1000, tol=1e-10, alpha=0.0, fit_intercept=False
    )
    steps = [("design_matrix", PatsyFormula("C(development) + C(origin)")), ("model", model)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        developed = DevelopmentML(Pipeline(steps=steps), y_ml=sample.columns[0]).fit_transform(
            sample
        )
        chain = cl.Chainladder().fit(developed)
    expected = float(np.nansum(chain.ibnr_.values))
    result = methods.tweedie_glm(named("GenIns"), power=power)
    assert result.totals["ibnr"][0].as_py() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("power", POWERS)
def test_a_development_only_design_fits_each_age_at_its_mean(power):
    """With no origin term the log-link fit has a closed form: each age's fitted
    increment is the mean of its observed increments, at every power."""
    rows = rows_of("GenIns")
    result = methods.tweedie_glm(named("GenIns"), power=power, origin="none")
    increments = [[row[j] - (row[j - 1] if j else 0.0) for j in range(len(row))] for row in rows]
    means = [np.mean([r[j] for r in increments if len(r) > j]) for j in range(10)]
    fitted = column(result.cells, "fitted_increment").reshape(10, 10)
    np.testing.assert_allclose(fitted, np.tile(means, (10, 1)), rtol=1e-9)
    assert result.coefficients["term"].to_pylist().count("origin") == 0


def test_a_quarterly_triangle_is_the_quarterly_chain_ladder():
    """Quarterly origins developed quarterly: GenIns' cumulatives read as ten
    quarters from 2001Q1, at ages 3 to 30 months."""
    cells = cells_of(rows_of("GenIns"), step=3, label=lambda i: f"{2001 + i // 4}Q{i % 4 + 1}")
    glm = methods.tweedie_glm(cells, dev_grain_months=3)
    chain = methods.chain_ladder(cells, dev_grain_months=3, zero_cells="observed")
    np.testing.assert_allclose(
        column(glm.origins, "ultimate"), column(chain.origins, "ultimate"), rtol=1e-10
    )
    assert glm.origins["origin"].to_pylist()[:2] == ["2001Q1", "2001Q2"]
    assert glm.as_of == dt.date(2003, 6, 30)
    assert glm.development["dev_lag"].to_pylist() == list(range(3, 31, 3))


@pytest.mark.tieout
def test_annual_origins_developed_quarterly_are_refused_before_the_fit():
    """chainladder's ``quarterly`` sample, whose GLM chainladder answers with a
    total of NaN at power 0."""
    cl = pytest.importorskip("chainladder")
    import pandas as pd

    long = cl.load_sample("quarterly")["paid"].to_frame(keepdims=True).reset_index()
    cells = {
        "origin_period": pd.to_datetime(long["origin"]).dt.date.tolist(),
        "dev_lag": long["development"].astype("int64").tolist(),
        "value": long["paid"].astype(float).tolist(),
    }
    refusal = refusal_of(lambda: methods.tweedie_glm(cells, power=0, dev_grain_months=3))
    assert (refusal.reason, refusal.method) == ("grain_mismatch", "tweedie_glm")


# -- 4. the units, and every option reaching the fit --------------------------------


@pytest.mark.parametrize("power", POWERS)
@pytest.mark.parametrize("link", ["log", "identity"])
def test_the_answer_scales_exactly_with_the_units(power, link):
    """No penalty and a relative stopping rule: the triangle times 1e-3 or 1e6
    gives every amount times the same, to 1e-10. chainladder's GLM moves by up
    to a factor of 3 over the same range. Mutation: a ridge penalty of 1.0, or
    stopping at an absolute change of 1e-6: this fails."""
    base = methods.tweedie_glm(named("UKMotor"), power=power, link=link)
    base_coef = base.coefficients
    for scale in (1e-3, 1e6):
        scaled = methods.tweedie_glm(named("UKMotor", scale=scale), power=power, link=link)
        for table, name in [
            ("origins", "ultimate"),
            ("origins", "model_ibnr"),
            ("cells", "fitted_increment"),
            ("totals", "ibnr"),
        ]:
            np.testing.assert_allclose(
                column(getattr(scaled, table), name),
                column(getattr(base, table), name) * scale,
                rtol=1e-10,
                atol=1e-9 * scale,
                err_msg=f"{table}.{name}",
            )
        np.testing.assert_allclose(
            column(scaled.development, "factor")[:-1] if link == "log" else [0.0],
            column(base.development, "factor")[:-1] if link == "log" else [0.0],
            rtol=1e-10,
        )
        estimate = column(scaled.coefficients, "estimate")
        expected = column(base_coef, "estimate")
        if link == "log":
            expected[0] += math.log(scale)
        else:
            expected *= scale
        np.testing.assert_allclose(estimate, expected, rtol=1e-9, atol=1e-9)
        # a standard error on the log scale does not depend on the units; on the
        # identity scale it is in the units of the amounts
        np.testing.assert_allclose(
            column(scaled.coefficients, "std_error"),
            column(base_coef, "std_error") * (1.0 if link == "log" else scale),
            rtol=1e-8,
        )
        totals, base_totals = scaled.totals.to_pylist()[0], base.totals.to_pylist()[0]
        assert totals["deviance"] == pytest.approx(
            base_totals["deviance"] * scale ** (2 - power), rel=1e-8
        )


def test_each_power_gives_its_own_answer():
    totals = {
        p: methods.tweedie_glm(named("GenIns"), power=p).totals["ibnr"][0].as_py()
        for p in (1.0, 1.5, 2.0)
    }
    assert len({round(v, 2) for v in totals.values()}) == 3, totals
    # the pattern route (chainladder's) at 1.5 and 2 on GenIns
    assert totals[1.5] == pytest.approx(18472367, abs=1)
    assert totals[2.0] == pytest.approx(18257520, abs=1)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("link", "identity"),
        ("origin", "none"),
        ("projection", "increments"),
        ("calendar", "trend"),
    ],
)
def test_every_option_reaches_the_fit(option, value):
    """One call with the option at its default and one with it changed, at power
    0 (which takes every link) on UKMotor: the answer moves. A calendar trend
    needs origin='none' in both."""
    base = {"power": 0.0}
    if option == "calendar":
        base["origin"] = "none"
    before = methods.tweedie_glm(named("UKMotor"), **base)
    after = methods.tweedie_glm(named("UKMotor"), **base, **{option: value})
    moved = [
        name
        for name in ("origins", "cells", "coefficients", "totals")
        if not getattr(before, name).equals(getattr(after, name))
    ]
    assert "origins" in moved or "cells" in moved, option
    assert before.totals["ibnr"][0].as_py() != after.totals["ibnr"][0].as_py(), option


def test_max_iter_reaches_the_fit():
    """Mutation: return the fit when max_iter runs out: this fails."""
    refusal = refusal_of(lambda: methods.tweedie_glm(named("GenIns"), max_iter=1))
    assert (refusal.reason, refusal.option, refusal.given) == ("did_not_converge", "max_iter", 1)
    assert "22.7% of the largest increment" in str(refusal)
    iterations = methods.tweedie_glm(named("GenIns")).totals["iterations"][0].as_py()
    assert methods.tweedie_glm(named("GenIns"), max_iter=iterations).totals.equals(
        methods.tweedie_glm(named("GenIns")).totals
    )
    refusal_of(lambda: methods.tweedie_glm(named("GenIns"), max_iter=iterations - 1))


def test_a_tail_is_not_supported_yet():
    refusal = refusal_of(lambda: methods.tweedie_glm(named("GenIns"), tail=1.05))
    assert (refusal.reason, refusal.option, refusal.given) == ("not_supported", "tail", 1.05)


# -- 5. zeros, edges and refusals ---------------------------------------------------


ZERO_ROW_AND_COLUMN = [
    [100, 150, 170, 175, 175],
    [0, 0, 0, 0],
    [120, 175, 190],
    [130, 180],
    [140],
]


def test_a_zero_origin_and_a_zero_age_are_fitted_at_zero():
    """An origin and a last age with no losses, under the log link: fitted at
    exactly 0, their coefficients null with ``fitted_zero``, the origin's
    ultimate 0 and the factor into the zero age 1, as the chain ladder gives.
    Mutation: leave the zero rows and columns in the regression: the fit then
    runs out its iterations (or stops only by luck) and this fails."""
    cells = cells_of(ZERO_ROW_AND_COLUMN)
    result = methods.tweedie_glm(cells)
    assert result.totals["iterations"][0].as_py() <= 25
    fitted = column(result.cells, "fitted_increment").reshape(5, 5)
    assert (fitted[1] == 0.0).all() and (fitted[:, 4] == 0.0).all()
    assert (fitted[np.ix_([0, 2, 3, 4], range(4))] > 0).all()
    coefficients = result.coefficients.to_pylist()
    zero = [
        (row["term"], row["origin"], row["dev_lag"]) for row in coefficients if row["fitted_zero"]
    ]
    assert zero == [("origin", 2002, None), ("development", None, 60)]
    for row in coefficients:
        assert (row["estimate"] is None) == row["fitted_zero"]
        assert (row["std_error"] is None) == row["fitted_zero"]
    assert result.origins["ultimate"][1].as_py() == 0.0
    assert result.development["factor"].to_pylist()[3] == 1.0
    chain = methods.chain_ladder(cells, zero_cells="observed")
    np.testing.assert_allclose(
        column(result.origins, "ultimate"), column(chain.origins, "ultimate"), rtol=1e-10
    )
    totals = result.totals.to_pylist()[0]
    assert (totals["n_observed"], totals["n_parameters"]) == (10, 7)  # 15 cells, 5 fitted at zero


def test_zeros_at_the_start_leave_the_first_factors_null():
    """A first age with no losses anywhere: its fitted cumulative is 0, so the
    factor out of it and its cdf are null, never NaN or infinite."""
    rows = [[0, 100, 150, 160], [0, 110, 170], [0, 120], [0]]
    result = methods.tweedie_glm(cells_of(rows))
    assert result.development["factor"].to_pylist()[0] is None
    assert result.development["cdf"].to_pylist()[0] is None
    assert result.development["pct_reported"].to_pylist()[0] is None
    assert result.origins["ultimate"].to_pylist()[3] == 0.0
    _no_nan(result)


@pytest.mark.parametrize("power", [1.0, 1.5])
def test_a_zero_increment_is_answered_below_power_two(power):
    result = methods.tweedie_glm(cells_of(ZERO_ROW_AND_COLUMN), power=power)
    assert result.origins.num_rows == 5


def test_a_zero_increment_is_refused_at_power_two_by_its_cell():
    refusal = refusal_of(lambda: methods.tweedie_glm(cells_of(ZERO_ROW_AND_COLUMN), power=2))
    assert refusal.reason == "zero_increment"
    assert [(c.origin, c.dev_lag) for c in refusal.cells] == [
        (2001, 60),
        (2002, 12),
        (2002, 24),
        (2002, 36),
        (2002, 48),
    ]


def test_raa_is_refused_by_its_negative_increment():
    raa = cells_of(RAA, 1981)
    for power in (1, 1.5, 2):
        refusal = refusal_of(lambda power=power: methods.tweedie_glm(raa, power=power))
        assert refusal.reason == "negative_increment"
        assert [(c.origin, c.dev_lag, c.value) for c in refusal.cells] == [(1982, 84, 15496.0)]
        assert "power=0 (normal) accepts them" in str(refusal)


def test_raa_at_power_zero_takes_its_negative_increment():
    """Normal errors accept a negative increment: R's route (the fitted future
    increments) gives 52,557.47 with the log link and 57,561.44 with the
    identity link, whose fitted future increments include negative ones."""
    raa = cells_of(RAA, 1981)
    log = methods.tweedie_glm(raa, power=0, projection="increments")
    assert log.totals["ibnr"][0].as_py() == pytest.approx(52557.47, abs=0.01)
    identity = methods.tweedie_glm(raa, power=0, link="identity")
    assert identity.totals["ibnr"][0].as_py() == pytest.approx(57561.44, abs=0.01)
    future = identity.cells.filter(pc.invert(identity.cells["observed"]))
    assert pc.sum(pc.less(future["fitted_increment"], 0)).as_py() == 6


@pytest.mark.parametrize(
    "rows", [[[100, 150, 170]], [[100, 150], [120]]], ids=["one_origin", "2x2"]
)
def test_as_many_terms_as_cells_leaves_the_dispersion_undefined(rows):
    result = methods.tweedie_glm(cells_of(rows))
    totals = result.totals.to_pylist()[0]
    assert totals["n_observed"] == totals["n_parameters"]
    assert totals["dispersion"] is None
    assert result.coefficients["std_error"].null_count == result.coefficients.num_rows
    assert result.coefficients["estimate"].null_count == 0
    assert result.origins["ultimate"].null_count == 0
    _no_nan(result)


def test_a_boundary_fit_is_refused_as_degenerate():
    """An origin with losses only on the latest diagonal and zeros before them
    (GA Resaurant's paid triangle in clrd): the Poisson fit has no finite
    solution, its fitted means on the zero cells falling towards 0."""
    increments = [[0] * (10 - i) for i in range(5)] + [
        [0, 0, 0, 0, 122],
        [0, 0, 0, 378],
        [0, 0, 478],
        [0, 278],
        [126],
    ]
    rows = [np.cumsum(row).tolist() for row in increments]
    refusal = refusal_of(lambda: methods.tweedie_glm(cells_of(rows, 1988)))
    assert refusal.reason == "degenerate_fit"
    assert refusal.cells and all(cell.value == 0.0 for cell in refusal.cells)
    assert "chain_ladder(cells, unsupported_factor='unity')" in str(refusal)


def test_an_identity_fit_below_zero_is_refused_at_a_positive_power():
    """Three cells and three terms: the identity fit puts the future cell at
    100 - 99 - 99 = -98, which a Poisson mean cannot be."""
    refusal = refusal_of(
        lambda: methods.tweedie_glm(cells_of([[100, 101], [1]]), power=1, link="identity")
    )
    assert refusal.reason == "negative_fitted_mean"
    assert [(c.origin, c.dev_lag) for c in refusal.cells] == [(2002, 24)]


FLAT_LAST_AGE = [[100, 150, 170, 170], [110, 160, 185], [120, 175], [130]]


@pytest.mark.parametrize("power", [1.0, 1.5])
@pytest.mark.parametrize("origin", ["factor", "none"])
def test_the_identity_link_gives_one_verdict_at_every_scale(power, origin):
    """The last age's only increment is 0, so the identity fit's mean there
    runs to exactly 0, which rounding leaves a hair above or below zero
    depending on the units. Before the fix the same triangle was answered at
    one scale and refused at another (as ``negative_fitted_mean`` or
    ``did_not_converge``). Mutation: test the fitted means against 0 instead of
    ``BOUNDARY`` times the largest increment: this fails."""
    seen = set()
    for scale in (1.0, 1e-3, 0.7, 3.0, 7.0, 1e3):
        refusal = refusal_of(
            lambda scale=scale: methods.tweedie_glm(
                cells_of(FLAT_LAST_AGE, scale=scale), power=power, link="identity", origin=origin
            )
        )
        seen.add((refusal.reason, tuple((c.origin, c.dev_lag) for c in refusal.cells)))
        assert "link='log'" in str(refusal)
    assert seen == {("negative_fitted_mean", ((2001, 48),))}
    # the log link fits the zero age at exactly 0 at every scale
    for scale in (1.0, 0.7, 7.0):
        result = methods.tweedie_glm(
            cells_of(FLAT_LAST_AGE, scale=scale), power=power, origin=origin
        )
        assert column(result.cells, "fitted_increment")[3::4].tolist() == [0.0] * 4


@pytest.mark.parametrize("power", [1.0, 1.5])
def test_a_small_positive_increment_is_an_answer(power):
    """GenIns with its one 120-month increment set to 0.01, 6.4e-9 of the
    largest: the fitted mean there is small because the increment is, not
    because the fit runs off to a limit, so it is answered (at power 1 the
    chain ladder's answer). Mutation: test every observed cell against
    ``BOUNDARY``, not only those whose increment is zero or less: this fails."""
    rows = [list(row) for row in rows_of("GenIns")]
    rows[0][-1] = rows[0][-2] + 0.01
    result = methods.tweedie_glm(cells_of(rows), power=power)
    fitted = column(result.cells, "fitted_increment")
    assert fitted[9] == pytest.approx(0.01, rel=1e-3)
    if power == 1.0:
        chain = methods.chain_ladder(cells_of(rows), zero_cells="observed")
        np.testing.assert_allclose(
            column(result.origins, "ultimate"), column(chain.origins, "ultimate"), rtol=1e-10
        )


FALLS_AT_36 = [[100, 200, 190, 195], [110, 220, 200], [120, 250], [130]]


def test_an_age_that_falls_on_balance_is_named_at_power_zero():
    """At power 0 the increments at 36 months are -10 and -20. Under the log
    link every fitted increment is above zero, so the fit has no finite
    answer; the message names the age and the identity link, which answers."""
    refusal = refusal_of(lambda: methods.tweedie_glm(cells_of(FALLS_AT_36), power=0))
    assert refusal.reason == "degenerate_fit"
    text = str(refusal)
    assert "increments at 36 months sum to zero or less" in text
    assert "link='identity'" in text
    assert "only zeros" not in text
    identity = methods.tweedie_glm(cells_of(FALLS_AT_36), power=0, link="identity")
    assert identity.origins.num_rows == 4


def test_an_age_that_falls_on_balance_can_still_have_a_log_link_fit():
    """At 36 months +40 on a large origin and -60 on a small one sum to -20,
    and the power-0 log-link fit still exists (12 iterations): a sum of zero or
    less does not prove there is no fit, so a fit stopped by ``max_iter`` stays
    ``did_not_converge``, with the age named as a possible cause."""
    rows = [[1000, 1500, 1540, 1560], [100, 200, 140], [1100, 1600], [1200]]
    result = methods.tweedie_glm(cells_of(rows), power=0)
    assert result.totals["iterations"][0].as_py() == 12
    refusal = refusal_of(lambda: methods.tweedie_glm(cells_of(rows), power=0, max_iter=2))
    assert refusal.reason == "did_not_converge"
    assert "Pass a larger max_iter" in str(refusal)
    assert "increments at 36 months sum to zero or less" in str(refusal)


@pytest.mark.parametrize("power", [0.0, 1.0, 1.5, 2.0])
def test_the_pearson_residuals_sum_to_the_chi_squared(power):
    """Each residual is ``(y - mu) / sqrt(mu ** power)``, so their squares over
    the observed cells add up to ``totals.pearson_chi2``. Mutation: divide by
    ``mu ** power`` rather than its square root: this fails at every power but
    0."""
    result = methods.tweedie_glm(named("GenIns"), power=power)
    residual = np.asarray(result.cells["pearson_residual"].drop_null().to_pylist())
    chi2 = result.totals["pearson_chi2"][0].as_py()
    assert np.sum(residual**2) == pytest.approx(chi2, rel=1e-10)


def test_tiny_amounts_at_power_two_give_the_same_residuals():
    """At power 2 a residual does not depend on the units, but ``mu ** 2``
    underflows to 0 near 1e-300. Mutation: raise the mean to the full power
    and take the square root: this is refused as ``result_not_finite``."""
    base = methods.tweedie_glm(named("GenIns"), power=2)
    tiny = methods.tweedie_glm(named("GenIns", scale=1e-300), power=2)
    np.testing.assert_allclose(
        column(tiny.cells, "pearson_residual")[~np.isnan(column(base.cells, "pearson_residual"))],
        column(base.cells, "pearson_residual")[~np.isnan(column(base.cells, "pearson_residual"))],
        rtol=1e-9,
        atol=1e-14,
    )


@pytest.mark.parametrize("power", [1.0, 1.5])
def test_pct_reported_is_the_reciprocal_of_the_cdf(power):
    """At power 1 it is the chain ladder's too. Mutation: report the cdf itself:
    this fails."""
    development = methods.tweedie_glm(named("GenIns"), power=power).development
    np.testing.assert_allclose(
        column(development, "pct_reported"), 1.0 / column(development, "cdf"), rtol=1e-14
    )
    assert column(development, "pct_reported")[-1] == 1.0
    if power == 1.0:
        chain = methods.chain_ladder(named("GenIns"), zero_cells="observed").development
        np.testing.assert_allclose(
            column(development, "pct_reported"), column(chain, "pct_reported"), rtol=1e-10
        )


@pytest.mark.parametrize("projection", ["pattern", "increments"])
def test_an_ultimate_below_zero_is_refused(projection):
    refusal = refusal_of(
        lambda: methods.tweedie_glm(
            cells_of([[100, 10], [1]]), power=0, link="identity", projection=projection
        )
    )
    assert refusal.reason == "negative_projection"
    assert [c.origin for c in refusal.cells] == [2002]


# -- 6. labels and nulls ------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "kind", "step"),
    [
        (lambda i: 1988 + i, pa.int64(), 12),
        (lambda i: str(1988 + i), pa.string(), 12),
        (lambda i: f"{2020 + (i + 2) // 4}Q{(i + 2) % 4 + 1}", pa.string(), 3),
        (lambda i: dt.date(1988 + i, 12, 31), pa.date32(), 12),
    ],
    ids=["int", "text", "quarter", "period_end"],
)
def test_the_callers_labels_are_echoed_by_period_in_every_table(label, kind, step):
    """Rows shuffled, so a label matched by position rather than by period
    would land on the wrong origin."""
    rows = rows_of("UKMotor")
    ordered = cells_of(rows, step=step, label=label)
    order = list(range(ordered.num_rows))
    random.Random(7).shuffle(order)
    shuffled = ordered.take(pa.array(order))
    result = methods.tweedie_glm(shuffled, dev_grain_months=step)
    reference = methods.tweedie_glm(ordered, dev_grain_months=step)
    labels = [label(i) for i in range(len(rows))]
    assert result.origins["origin"].type == kind
    assert result.origins["origin"].to_pylist() == labels
    assert result.cells["origin"].type == kind
    assert result.cells["origin"].to_pylist() == [lab for lab in labels for _ in rows]
    terms = result.coefficients.filter(pc.equal(result.coefficients["term"], "origin"))
    assert terms["origin"].type == kind
    assert terms["origin"].to_pylist() == labels[1:]
    others = result.coefficients.filter(pc.not_equal(result.coefficients["term"], "origin"))
    assert others["origin"].null_count == others.num_rows
    for name in methods.TABLES:
        assert getattr(result, name) == getattr(reference, name), name


def _no_nan(result) -> None:
    for name in methods.TABLES:
        table = getattr(result, name)
        if table is None:
            continue
        for column_name in table.column_names:
            values = table[column_name]
            if pa.types.is_floating(values.type):
                assert not pc.any(pc.is_nan(values)).as_py(), (name, column_name)
                assert pc.all(pc.is_finite(values.drop_null())).as_py() in (True, None)


@pytest.mark.parametrize("link", ["log", "identity"])
def test_missing_numbers_are_nulls_never_nan(link):
    result = methods.tweedie_glm(named("GenIns"), power=0, link=link)
    _no_nan(result)
    cells = result.cells
    assert cells.num_rows == 100
    assert cells["increment"].null_count == 45  # the unobserved cells
    assert cells["pearson_residual"].null_count == 45
    assert cells["factor"].null_count == 10  # the last age
    assert cells["cdf"].null_count == 0
    assert result.development["factor"].null_count == (1 if link == "log" else 10)
    assert result.coefficients["estimate"].null_count == 0
    assert result.link_ratios is None


def test_the_result_schema_is_pinned():
    result = methods.tweedie_glm(named("GenIns"))
    assert result.method == "tweedie_glm"
    assert result.origins.schema.equals(
        pa.schema(
            [
                ("origin", pa.int64()),
                ("origin_period", pa.date32()),
                ("latest_dev_lag", pa.int64()),
                ("latest", pa.float64()),
                ("ultimate", pa.float64()),
                ("ibnr", pa.float64()),
                ("model_ibnr", pa.float64()),
            ]
        )
    )
    assert result.development.schema.equals(
        pa.schema(
            [
                ("dev_lag", pa.int64()),
                ("factor", pa.float64()),
                ("cdf", pa.float64()),
                ("pct_reported", pa.float64()),
                ("n_observed", pa.int64()),
            ]
        )
    )
    assert result.cells.schema.equals(
        pa.schema(
            [
                ("origin", pa.int64()),
                ("origin_period", pa.date32()),
                ("dev_lag", pa.int64()),
                ("observed", pa.bool_()),
                ("increment", pa.float64()),
                ("fitted_increment", pa.float64()),
                ("fitted_cumulative", pa.float64()),
                ("factor", pa.float64()),
                ("cdf", pa.float64()),
                ("pearson_residual", pa.float64()),
            ]
        )
    )
    assert result.coefficients.schema.equals(
        pa.schema(
            [
                ("term", pa.string()),
                ("origin", pa.int64()),
                ("origin_period", pa.date32()),
                ("dev_lag", pa.int64()),
                ("estimate", pa.float64()),
                ("std_error", pa.float64()),
                ("fitted_zero", pa.bool_()),
            ]
        )
    )
    assert result.totals.schema.equals(
        pa.schema(
            [
                ("latest", pa.float64()),
                ("ultimate", pa.float64()),
                ("ibnr", pa.float64()),
                ("model_ibnr", pa.float64()),
                ("power", pa.float64()),
                ("link", pa.string()),
                ("deviance", pa.float64()),
                ("pearson_chi2", pa.float64()),
                ("dispersion", pa.float64()),
                ("n_observed", pa.int64()),
                ("n_parameters", pa.int64()),
                ("iterations", pa.int64()),
            ]
        )
    )
    assert result.development["n_observed"].to_pylist() == list(range(10, 0, -1))
    assert result.coefficients["term"].to_pylist() == (
        ["intercept"] + ["origin"] * 9 + ["development"] * 9
    )


def test_to_polars_gives_the_new_tables_and_refuses_what_is_absent():
    pl = pytest.importorskip("polars")
    result = methods.tweedie_glm(named("GenIns"))
    for name in ("cells", "coefficients"):
        assert result.to_polars(name).equals(pl.from_arrow(getattr(result, name)))
    refusal = refusal_of(lambda: result.to_polars("link_ratios"))
    assert "fitted to the increments, not to link ratios" in str(refusal)
    chain = methods.chain_ladder(named("GenIns"))
    assert chain.cells is None and chain.coefficients is None
    refusal = refusal_of(lambda: chain.to_polars("cells"))
    assert "a chain_ladder result has no cells table" in str(refusal)


# -- 7. the kernel directly -----------------------------------------------------------


def test_the_kernel_takes_a_spec_and_refuses_anything_else():
    fit = fit_tweedie_grid(grid_of(rows_of("GenIns")))
    assert fit.spec == TweedieSpec()
    assert fit.common_pattern
    assert fit.pattern.shape == (9,)
    with pytest.raises(TypeError, match="TweedieSpec"):
        fit_tweedie_grid(grid_of(rows_of("GenIns")), {"power": 1})


def test_the_identity_link_has_no_common_pattern():
    """Each origin's fitted factors differ under the identity link, so there is
    no one pattern. Mutation: ``common_pattern`` always true: this fails."""
    fit = fit_tweedie_grid(grid_of(rows_of("GenIns")), TweedieSpec(power=0, link="identity"))
    assert not fit.common_pattern
    assert fit.pattern is None


def test_a_kernel_refusal_names_the_period_start():
    refusal = refusal_of(lambda: fit_tweedie_grid(grid_of(RAA), TweedieSpec(power=1)))
    assert refusal.method is None
    assert [(c.origin, c.origin_period, c.dev_lag) for c in refusal.cells] == [
        (None, dt.date(2002, 1, 1), 84)
    ]


# -- 8. the clrd sweep ----------------------------------------------------------------


def _clrd_grids():
    """Every clrd company and line's paid triangle from chainladder's raw
    ``clrd.csv``, so a zero stays a zero (its Triangle stores zeros as missing)."""
    cl = pytest.importorskip("chainladder")
    import csv

    path = Path(cl.__file__).parent / "utils" / "data" / "clrd.csv"
    grids: dict[tuple[str, str], np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            year, lag = int(row["AccidentYear"]), int(row["DevelopmentLag"])
            if year + lag - 1 > 1997:
                continue
            grid = grids.setdefault((row["GRNAME"], row["LOB"]), np.full((10, 10), np.nan))
            grid[year - 1988, lag - 1] = float(row["CumPaidLoss"])
    return grids


def _cells_of_grid(grid: np.ndarray) -> pa.Table:
    where = np.argwhere(~np.isnan(grid))
    return pa.table(
        {
            "origin_period": (1988 + where[:, 0]).tolist(),
            "dev_lag": (12 * (where[:, 1] + 1)).tolist(),
            "value": grid[~np.isnan(grid)].tolist(),
        }
    )


@pytest.mark.tieout
def test_every_clrd_paid_triangle_is_the_chain_ladder_or_refused_by_name():
    """On the 273 paid triangles with increments of zero or more and no
    cumulative going from zero to a positive amount, the power-1 GLM equals
    the chain ladder (with a factor of 1 where an age has no link ratio) under
    both zero rules, to 1e-8. Every other triangle is answered or refused with
    a Refusal, never another exception. Of the 82 other triangles with
    increments of zero or more, 13 have no finite Poisson fit (a fitted mean on
    an observed cell falls below 1e-8 of the largest increment, 1e-11 to 1e-42
    measured) and are refused as ``degenerate_fit``; the other 69 are answered."""
    grids = _clrd_grids()
    assert len(grids) == 775
    outcomes: dict[str, int] = {}
    matched = 0
    for (company, line), grid in sorted(grids.items()):
        cells = _cells_of_grid(grid)
        increments = np.diff(np.nan_to_num(grid, nan=0.0), axis=1, prepend=0.0)
        increments[np.isnan(grid)] = np.nan
        plain = (
            np.nanmax(np.abs(grid)) > 0
            and np.nanmin(increments) >= 0
            and not np.nansum((grid[:, :-1] == 0) & (grid[:, 1:] > 0))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                result = methods.tweedie_glm(cells)
            except Refusal as refusal:
                assert type(refusal) is Refusal
                assert not plain, (company, line, str(refusal))
                outcomes[refusal.reason] = outcomes.get(refusal.reason, 0) + 1
                continue
        outcomes["answered"] = outcomes.get("answered", 0) + 1
        _no_nan(result)
        if plain:
            ultimate = column(result.origins, "ultimate")
            for zero_cells in ("observed", "missing"):
                chain = methods.chain_ladder(
                    cells, zero_cells=zero_cells, unsupported_factor="unity"
                )
                expected = column(chain.origins, "ultimate")
                gap = np.max(np.abs(ultimate - expected) / np.maximum(np.abs(expected), 1.0))
                assert gap <= 1e-8, (company, line, zero_cells, gap)
            matched += 1
    assert matched == 273, outcomes
    assert outcomes["answered"] == 342, outcomes
    assert outcomes["degenerate_fit"] == 13, outcomes


@pytest.mark.tieout
def test_the_documented_clrd_cases():
    """Two paid triangles with no finite Poisson fit are refused, and one where
    the GLM and the chain ladder differ because an origin starts from zero is
    answered with the documented numbers."""
    grids = _clrd_grids()
    for key in [
        ("GA Resaurant Mut Captive Ins Co", "wkcomp"),
        ("Campmed Cas & Ind Co Inc MD", "medmal"),
    ]:
        refusal = refusal_of(lambda key=key: methods.tweedie_glm(_cells_of_grid(grids[key])))
        assert refusal.reason == "degenerate_fit", key
    cells = _cells_of_grid(grids[("American Assoc Of Othodontists RRG", "medmal")])
    glm = column(methods.tweedie_glm(cells).origins, "ultimate")
    chain = column(methods.chain_ladder(cells, unsupported_factor="unity").origins, "ultimate")
    assert glm[-1] == pytest.approx(284.9, abs=0.05)
    assert chain[-1] == pytest.approx(275.9, abs=0.05)
