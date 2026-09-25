"""``methods.one_year_cdr``: the one-year claims development result at the front door.

Four things are checked.

1. **The draws are the gallery's.** For the same seed, draw count and options,
   the front door's draws equal, byte for byte, the ones the gallery's ``mack``
   entry gives for a Triangle with the one segment ``Total`` and the field
   ``values``, which is what the Reserving app's ``/cdr`` route fits today.
   Once through chainladder, exactly as the app builds it (``tieout``), and
   once through ibnr's own Triangle, so the core leg checks it too.
2. **The analytic figures are the kernel's**, and through the kernel R's
   published MW2014 numbers.
3. **Every option changes the answer where it should**, so none of them is
   accepted and then ignored.
4. **The tables have the columns, types and orientation the docstring says**:
   ``ultimate_change`` is next year's ultimate minus today's, positive when the
   reserve is strengthened, the opposite sign of ``kernels.simulate_one_year_cdr``.

The refusals are in ``tests/test_refusal.py`` with every other method's.
"""

from __future__ import annotations

import datetime as dt
import inspect
import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ibnr import methods
from ibnr.errors import Refusal
from ibnr.kernels.cdr import one_year_cdr as kernel_one_year_cdr
from ibnr.kernels.cdr import simulate_one_year_cdr
from ibnr.kernels.mack import fit_mack_grid
from ibnr.kernels.rng import cohort_stream

from .conftest import make_cohort_triangle
from .test_cdr import MW2014_CDR_SE, MW2014_IBNR, MW2014_MACK_SE, MW2014_ROWS, MW2014_TOTAL
from .test_methods import RAA, arrow_cells, kernel_grid

# genins, the Taylor-Ashe triangle, accident years 2001 to 2010.
GENINS = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
    [396132, 1333217, 2180715, 2985752, 3691712],
    [440832, 1288463, 2419861, 3483130],
    [359480, 1421128, 2864498],
    [376686, 1363294],
    [344014],
]
TRIANGLES = {"raa": (RAA, 1981), "genins": (GENINS, 2001)}

#: The Reserving app's default percentiles (``DEFAULT_CDR_PERCENTILES``).
APP_PERCENTILES = [50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9]

#: (sigma_rule, process, parameter_risk, seed, n_draws): the settings the
#: byte-for-byte comparisons run under, so each option is compared off its default.
#: The first is the app's own default request with the workbook's seed; three
#: draws put the median on a draw, where a tail mean's ">=" matters.
SETTINGS = [
    ("mack", "gamma", True, 42, 20_000),
    ("mack", "gamma", True, 42, 2_000),
    ("log_linear", "lognormal", False, 7, 1_500),
    ("mack", "normal", True, 12345, 1_000),
    ("log_linear", "gamma", True, 0, 3),
]


def matrix(rows) -> np.ndarray:
    out = np.full((len(rows), len(rows[0])), np.nan)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def year_cells(rows, first: int) -> pa.Table:
    """The cells with integer accident years, as a service sends them."""
    origin, lag, value = [], [], []
    for i, row in enumerate(rows):
        for j, amount in enumerate(row):
            origin.append(first + i)
            lag.append(12 * (j + 1))
            value.append(float(amount))
    return pa.table({"origin_period": origin, "dev_lag": lag, "value": value})


def app_style_answer(entry, *, n_draws, seed, process, parameter_risk, percentiles):
    """What the app's ``/cdr`` computes from a fitted gallery entry.

    Copied from ``function_app.py`` (``cdr``, ``_exact_tvar``): the draws are
    negated once, so positive is adverse, and every summary is taken from the
    negated draws.
    """
    pred = entry.cdr_distribution(
        n_draws=n_draws, seed=seed, process=process, parameter_risk=parameter_risk
    )
    fit = entry.fit_
    delta = -pred.samples
    total = delta[:, -1]
    per_origin = delta[:, : fit.n_w]

    def tvar(sample, p):
        threshold = np.percentile(sample, p)
        tail = sample[sample >= threshold]
        return float(tail.mean()) if tail.size else float("nan")

    analytic = entry.one_year_cdr()
    return {
        "per_origin": per_origin,
        "total": total,
        "mean": float(total.mean()),
        "std": float(total.std(ddof=1)) if n_draws > 1 else None,
        "percentiles": [float(np.percentile(total, p)) for p in percentiles],
        "tvar": [tvar(total, p) for p in percentiles],
        "origin_mean": [float(per_origin[:, i].mean()) for i in range(fit.n_w)],
        "origin_std": [
            float(per_origin[:, i].std(ddof=1)) if n_draws > 1 else None for i in range(fit.n_w)
        ],
        "central_ultimate": fit.ultimate.tolist(),
        "central_ibnr": fit.reserve.tolist(),
        "latest": fit.latest.tolist(),
        "cdr_se": np.sqrt(analytic.msep).tolist(),
        "total_cdr_se": math.sqrt(analytic.msep_total),
        "runoff_se": np.sqrt(analytic.runoff_msep).tolist(),
        "total_runoff_se": math.sqrt(analytic.runoff_msep_total),
    }


def assert_same_as_the_app(result, app, *, n_draws, percentiles) -> None:
    """Every number the app's ``/cdr`` response carries, equal to the last bit."""
    n_w = app["per_origin"].shape[1]
    draws = result.draws.column("ultimate_change").to_numpy().reshape(n_draws, n_w)
    assert draws.tobytes() == np.ascontiguousarray(app["per_origin"]).tobytes()
    assert result.draws.column("draw").to_pylist() == np.repeat(np.arange(n_draws), n_w).tolist()

    totals = result.totals.to_pylist()[0]
    assert totals["mean_ultimate_change"] == app["mean"]
    assert totals["sd_ultimate_change"] == app["std"]
    assert totals["cdr_se"] == app["total_cdr_se"]
    assert totals["runoff_se"] == app["total_runoff_se"]
    assert totals["ultimate"] == float(np.sum(app["central_ultimate"]))
    assert totals["n_draws"] == n_draws

    origins = result.origins
    assert origins.column("latest").to_pylist() == app["latest"]
    assert origins.column("ultimate").to_pylist() == app["central_ultimate"]
    assert origins.column("ibnr").to_pylist() == app["central_ibnr"]
    assert origins.column("mean_ultimate_change").to_pylist() == app["origin_mean"]
    assert origins.column("sd_ultimate_change").to_pylist() == app["origin_std"]
    assert origins.column("cdr_se").to_pylist() == app["cdr_se"]
    assert origins.column("runoff_se").to_pylist() == app["runoff_se"]

    rows = result.quantiles.to_pylist()
    total_rows = [row for row in rows if row["origin"] is None]
    # the app sends percentiles; the front door takes them divided by 100
    assert [row["level"] for row in total_rows] == [p / 100 for p in percentiles]
    assert [row["ultimate_change"] for row in total_rows] == app["percentiles"]
    assert [row["tvar"] for row in total_rows] == app["tvar"]


@pytest.mark.tieout
@pytest.mark.parametrize("name", sorted(TRIANGLES))
@pytest.mark.parametrize(("sigma_rule", "process", "parameter_risk", "seed", "n_draws"), SETTINGS)
def test_the_draws_are_the_apps_draws_through_chainladder(
    name, sigma_rule, process, parameter_risk, seed, n_draws
):
    """The app's ``/cdr`` as it runs today: its own parser builds a chainladder
    Triangle from origin strings, ``Triangle.from_chainladder`` converts it, and
    the gallery's ``mack`` entry is fitted on the field ``values``."""
    cl = pytest.importorskip("chainladder")
    pd = pytest.importorskip("pandas")
    from ibnr import Triangle
    from ibnr.gallery.deterministic.mack import Mack

    rows, first = TRIANGLES[name]
    origin, lags, values = [], [], []
    for i, row in enumerate(rows):
        for j, amount in enumerate(row):
            origin.append(str(first + i))
            lags.append(12 * (j + 1))
            values.append(float(amount))
    # function_app._parse_long_triangle
    start = [pd.Period(o).to_timestamp(how="start") for o in origin]
    valuation = [
        s + pd.DateOffset(months=lag) - pd.Timedelta(days=1)
        for s, lag in zip(start, lags, strict=True)
    ]
    frame = pd.DataFrame({"origin": origin, "valuation": valuation, "values": values})
    tri = cl.Triangle(
        frame, origin="origin", development="valuation", columns="values", cumulative=True
    )
    entry = Mack().fit(
        Triangle.from_chainladder(tri), loss_field=str(tri.columns[0]), sigma_rule=sigma_rule
    )
    options = {"process": process, "parameter_risk": parameter_risk}
    app = app_style_answer(
        entry, n_draws=n_draws, seed=seed, percentiles=APP_PERCENTILES, **options
    )
    # the app's latest comes from chainladder's own latest diagonal
    assert tri.latest_diagonal.to_frame().iloc[:, 0].tolist() == app["latest"]

    cells = {"origin_period": origin, "dev_lag": lags, "value": values}
    result = methods.one_year_cdr(
        cells,
        sigma_rule=sigma_rule,
        n_draws=n_draws,
        seed=seed,
        quantiles=[p / 100 for p in APP_PERCENTILES],
        **options,
    )
    assert_same_as_the_app(result, app, n_draws=n_draws, percentiles=APP_PERCENTILES)
    assert result.origins.column("origin").to_pylist() == [str(o) for o in tri.origin]


@pytest.mark.parametrize("name", sorted(TRIANGLES))
@pytest.mark.parametrize(("sigma_rule", "process", "parameter_risk", "seed", "n_draws"), SETTINGS)
def test_the_draws_are_the_gallery_entrys_on_an_ibnr_triangle(
    name, sigma_rule, process, parameter_risk, seed, n_draws
):
    """The same comparison with no chainladder: the Triangle is built by ibnr
    with the one segment ``Total`` and the field ``values``, which is what
    ``Triangle.from_chainladder`` gives the app."""
    from ibnr.gallery.deterministic.mack import Mack

    rows, first = TRIANGLES[name]
    tri = make_cohort_triangle(
        None, matrix(rows), start_year=first, loss_field="values", segment={"Total": "Total"}
    )
    entry = Mack().fit(tri, loss_field="values", sigma_rule=sigma_rule)
    options = {"process": process, "parameter_risk": parameter_risk}
    percentiles = [10.0, 50.0, 99.5]
    app = app_style_answer(entry, n_draws=n_draws, seed=seed, percentiles=percentiles, **options)
    result = methods.one_year_cdr(
        year_cells(rows, first),
        sigma_rule=sigma_rule,
        n_draws=n_draws,
        seed=seed,
        quantiles=[p / 100 for p in percentiles],
        **options,
    )
    assert_same_as_the_app(result, app, n_draws=n_draws, percentiles=percentiles)


def test_the_draws_are_the_kernels_for_the_derived_stream():
    """The front door is ``simulate_one_year_cdr`` on ``fit_mack_grid``, with the
    seed turned into the stream the gallery entry uses, and the sign flipped."""
    grid = kernel_grid()
    fit = fit_mack_grid(grid, sigma_rule="log_linear")
    stream = cohort_stream(
        5, label="cdr_distribution", cohorts=[{"Total": "Total"}], field="values"
    )
    pred = simulate_one_year_cdr(fit, n_draws=500, seed=stream)
    result = methods.one_year_cdr(arrow_cells(), n_draws=500, seed=5)
    draws = result.draws.column("ultimate_change").to_numpy().reshape(500, fit.n_w)
    assert draws.tobytes() == (-pred.samples[:, : fit.n_w]).tobytes()
    # a plain integer seed handed to the kernel is a different stream
    plain = simulate_one_year_cdr(fit, n_draws=500, seed=5)
    assert not np.array_equal(-plain.samples[:, : fit.n_w], draws)


def test_the_same_seed_gives_the_same_answer_and_none_gives_fresh_draws():
    first = methods.one_year_cdr(arrow_cells(), n_draws=200, seed=11)
    again = methods.one_year_cdr(arrow_cells(), n_draws=200, seed=11)
    for name in methods.OneYearCDRResult.TABLES:
        assert getattr(first, name).equals(getattr(again, name)), name
    fresh = [methods.one_year_cdr(arrow_cells(), n_draws=200) for _ in range(2)]
    assert not fresh[0].draws.equals(fresh[1].draws)
    assert fresh[0].seed is None
    # the analytic figures do not depend on the draws
    assert (
        fresh[0]
        .origins.select(["cdr_se", "runoff_se"])
        .equals(fresh[1].origins.select(["cdr_se", "runoff_se"]))
    )


# -- the analytic figures ------------------------------------------------------------


@pytest.mark.parametrize("rule", ["mack", "log_linear"])
def test_the_standard_errors_are_the_kernels(rule):
    fit = fit_mack_grid(kernel_grid(), sigma_rule=rule)
    analytic = kernel_one_year_cdr(fit)
    result = methods.one_year_cdr(arrow_cells(), sigma_rule=rule, n_draws=10, seed=1)
    assert result.origins.column("cdr_se").to_pylist() == np.sqrt(analytic.msep).tolist()
    assert result.origins.column("runoff_se").to_pylist() == (
        np.sqrt(analytic.runoff_msep).tolist()
    )
    assert result.totals["cdr_se"][0].as_py() == math.sqrt(analytic.msep_total)
    assert result.totals["runoff_se"][0].as_py() == math.sqrt(analytic.runoff_msep_total)
    # the run-off standard error is methods.mack's, under the same rule and zero rule
    mack = methods.mack(arrow_cells(), sigma_rule=rule, zero_cells="observed")
    assert result.origins.column("runoff_se").equals(mack.origins.column("mack_se"))
    assert result.totals["runoff_se"].equals(mack.totals["mack_se"])


@pytest.mark.tieout
def test_mw2014_matches_published_r_output():
    """R ChainLadder's ``CDR(MackChainLadder(MW2014, est.sigma="Mack"))``, through
    the front door (``tests/test_cdr.py`` holds the same numbers for the kernel)."""
    result = methods.one_year_cdr(year_cells(MW2014_ROWS, 1990), sigma_rule="mack", n_draws=10)
    origins = result.origins
    np.testing.assert_allclose(origins.column("ibnr").to_numpy(), MW2014_IBNR, atol=1e-5)
    np.testing.assert_allclose(origins.column("cdr_se").to_numpy(), MW2014_CDR_SE, atol=1e-6)
    np.testing.assert_allclose(origins.column("runoff_se").to_numpy(), MW2014_MACK_SE, atol=1e-6)
    totals = result.totals.to_pylist()[0]
    assert totals["ibnr"] == pytest.approx(MW2014_TOTAL["ibnr"], abs=1e-5)
    assert totals["cdr_se"] == pytest.approx(MW2014_TOTAL["cdr_se"], abs=1e-6)
    assert totals["runoff_se"] == pytest.approx(MW2014_TOTAL["runoff_se"], abs=1e-6)


def test_the_simulated_spread_agrees_with_the_closed_form():
    """Mack's generator with parameter risk against Merz-Wuthrich, to Monte Carlo
    error: 40,000 draws put the standard error of an sd near 0.4%."""
    result = methods.one_year_cdr(arrow_cells(), sigma_rule="mack", n_draws=40_000, seed=3)
    totals = result.totals.to_pylist()[0]
    assert totals["sd_ultimate_change"] == pytest.approx(totals["cdr_se"], rel=0.03)


# -- each option reaches the answer ----------------------------------------------------


def _draws(result) -> np.ndarray:
    return result.draws.column("ultimate_change").to_numpy()


def test_every_option_changes_the_answer():
    base = {"n_draws": 300, "seed": 4}
    ref = methods.one_year_cdr(arrow_cells(), **base)

    # sigma_rule moves the standard errors, and through the sigmas the draws
    rule = methods.one_year_cdr(arrow_cells(), sigma_rule="mack", **base)
    assert rule.totals["cdr_se"][0].as_py() != ref.totals["cdr_se"][0].as_py()
    assert rule.totals["runoff_se"][0].as_py() != ref.totals["runoff_se"][0].as_py()
    assert not np.array_equal(_draws(rule), _draws(ref))

    for option, value in (("process", "lognormal"), ("process", "normal")):
        other = methods.one_year_cdr(arrow_cells(), **{option: value}, **base)
        assert not np.array_equal(_draws(other), _draws(ref)), (option, value)
        # the closed form does not depend on how the draws are made
        assert other.totals["cdr_se"].equals(ref.totals["cdr_se"])

    # without parameter risk only the process half is left, so the spread falls
    no_risk = methods.one_year_cdr(arrow_cells(), parameter_risk=False, **base)
    assert no_risk.totals["sd_ultimate_change"][0].as_py() < (
        0.95 * ref.totals["sd_ultimate_change"][0].as_py()
    )

    seeded = methods.one_year_cdr(arrow_cells(), n_draws=300, seed=5)
    assert not np.array_equal(_draws(seeded), _draws(ref))

    more = methods.one_year_cdr(arrow_cells(), n_draws=301, seed=4)
    assert more.draws.num_rows == 301 * 10
    assert more.totals["n_draws"][0].as_py() == 301

    levels = methods.one_year_cdr(arrow_cells(), quantiles=(0.25, 0.995), **base)
    assert sorted(set(levels.quantiles.column("level").to_pylist())) == [0.25, 0.995]
    assert levels.quantiles.num_rows == 2 * 11


def test_zero_cells_keeps_zeros_as_data_or_refuses_what_it_has_not_checked():
    """raa with 1982's 12-month cumulative zeroed: ``"observed"`` (the default)
    keeps it as data and answers; ``"missing"`` would leave out a link ratio,
    which the one-year formulas have not been checked with, so it refuses. On a
    triangle with no zero the two are the same fit."""
    rows = [list(row) for row in RAA]
    rows[1][0] = 0
    cells = year_cells(rows, 1981)
    observed = methods.one_year_cdr(cells, n_draws=50, seed=1)
    assert math.isfinite(observed.totals["cdr_se"][0].as_py())
    refusal = pytest.raises(Refusal, methods.one_year_cdr, cells, zero_cells="missing").value
    assert (refusal.reason, refusal.option, refusal.method) == (
        "not_supported",
        "zero_cells",
        "one_year_cdr",
    )
    plain = year_cells(RAA, 1981)
    same = [
        methods.one_year_cdr(plain, zero_cells=rule, n_draws=50, seed=1)
        for rule in ("observed", "missing")
    ]
    for name in methods.OneYearCDRResult.TABLES:
        assert getattr(same[0], name).equals(getattr(same[1], name)), name


def test_a_quarterly_triangle_is_read_and_then_refused_as_not_one_year():
    """``dev_grain_months=3`` is delivered: the cells are read on a three-month
    step (with the default of 12 their ages are refused as off the step), and
    the one-year result then refuses a step that is not a year."""
    rows = [[100.0, 150.0, 170.0], [110.0, 160.0], [120.0]]
    cells = {
        "origin_period": ["2020Q1", "2020Q1", "2020Q1", "2020Q2", "2020Q2", "2020Q3"],
        "dev_lag": [3, 6, 9, 3, 6, 3],
        "value": [v for row in rows for v in row],
    }
    at_twelve = pytest.raises(Refusal, methods.one_year_cdr, cells).value
    assert at_twelve.reason in ("grain_mismatch", "unreadable_label")
    refusal = pytest.raises(Refusal, methods.one_year_cdr, cells, dev_grain_months=3).value
    assert (refusal.reason, refusal.option) == ("not_supported", "dev_grain_months")
    assert "3-month" in str(refusal)


# -- the tables ------------------------------------------------------------------------


def test_the_tables_have_the_documented_columns_and_types():
    result = methods.one_year_cdr(year_cells(RAA, 1981), n_draws=20, seed=2, quantiles=(0.5, 0.9))
    assert result.method == "one_year_cdr"
    assert result.as_of == dt.date(1990, 12, 31)
    assert (result.dev_grain_months, result.n_draws, result.seed) == (12, 20, 2)
    f64, i64 = pa.float64(), pa.int64()
    assert result.origins.schema == pa.schema(
        [
            ("origin", i64),
            ("origin_period", pa.date32()),
            ("latest_dev_lag", i64),
            ("latest", f64),
            ("ultimate", f64),
            ("ibnr", f64),
            ("mean_ultimate_change", f64),
            ("sd_ultimate_change", f64),
            ("cdr_se", f64),
            ("runoff_se", f64),
        ]
    )
    assert result.totals.schema == pa.schema(
        [
            ("latest", f64),
            ("ultimate", f64),
            ("ibnr", f64),
            ("mean_ultimate_change", f64),
            ("sd_ultimate_change", f64),
            ("cdr_se", f64),
            ("runoff_se", f64),
            ("n_draws", i64),
        ]
    )
    assert result.quantiles.schema == pa.schema(
        [
            ("origin", i64),
            ("origin_period", pa.date32()),
            ("level", f64),
            ("ultimate_change", f64),
            ("tvar", f64),
        ]
    )
    assert result.draws.schema == pa.schema(
        [("draw", i64), ("origin", i64), ("origin_period", pa.date32()), ("ultimate_change", f64)]
    )
    # total rows first (null origin), then each origin in order, each at every level
    rows = result.quantiles.to_pylist()
    assert [(r["origin"], r["level"]) for r in rows] == [
        (origin, level) for origin in [None, *range(1981, 1991)] for level in (0.5, 0.9)
    ]
    assert rows[0]["origin_period"] is None
    assert result.draws.column("origin").to_pylist() == list(range(1981, 1991)) * 20
    assert result.origins.column("origin").to_pylist() == list(range(1981, 1991))
    assert result.origins.column("latest_dev_lag").to_pylist() == [120 - 12 * i for i in range(10)]
    # the per-origin quantiles and tail means are those of each origin's own draws
    draws = _draws(result).reshape(20, 10)
    per_origin = rows[2:]
    for i in range(10):
        for k, level in enumerate((0.5, 0.9)):
            row = per_origin[2 * i + k]
            assert row["ultimate_change"] == float(np.quantile(draws[:, i], level))
            tail = draws[:, i][draws[:, i] >= row["ultimate_change"]]
            assert row["tvar"] == float(tail.mean())


def test_the_callers_labels_are_echoed_in_every_table():
    labels = [str(1981 + i) for i in range(10)]
    cells = year_cells(RAA, 1981)
    text = cells.set_column(0, "origin_period", pa.array([str(v) for v in cells[0].to_pylist()]))
    result = methods.one_year_cdr(text, n_draws=5, seed=1)
    assert result.origins.column("origin").to_pylist() == labels
    assert result.draws.column("origin").to_pylist() == labels * 5
    assert result.quantiles.column("origin").to_pylist() == [
        label for label in [None, *labels] for _ in range(7)
    ]
    ends = cells.set_column(
        0,
        "origin_period",
        pa.array([dt.date(v, 12, 31) for v in cells[0].to_pylist()], pa.date32()),
    )
    by_date = methods.one_year_cdr(ends, n_draws=5, seed=1)
    assert by_date.origins.column("origin").type == pa.date32()
    # the labels change nothing else
    for name in ("totals",):
        assert getattr(by_date, name).equals(getattr(result, name))
    assert by_date.draws.column("ultimate_change").equals(result.draws.column("ultimate_change"))


def test_ultimate_change_is_adverse_positive():
    """A strengthening is positive: each draw is next year's re-estimated
    ultimate minus today's, the negative of the kernel's claims development result.
    A fully developed origin never moves."""
    result = methods.one_year_cdr(arrow_cells(), n_draws=2000, seed=9)
    draws = _draws(result).reshape(2000, 10)
    assert (draws[:, 0] == 0.0).all()  # 1981 is at its last age
    stream = cohort_stream(
        9, label="cdr_distribution", cohorts=[{"Total": "Total"}], field="values"
    )
    fit = fit_mack_grid(kernel_grid(), sigma_rule="log_linear")
    kernel = simulate_one_year_cdr(fit, n_draws=2000, seed=stream)
    # the kernel's result is today's ultimate minus next year's (rereserve's step 3)
    assert np.array_equal(draws, -kernel.samples[:, :10])
    # the 99.5% level of an adverse-positive change is a strengthening
    total = result.quantiles.filter(pc.is_null(result.quantiles["origin"]))
    q = dict(zip(total["level"].to_pylist(), total["ultimate_change"].to_pylist(), strict=True))
    assert q[0.995] > 0 and q[0.995] > q[0.5]


def test_one_draw_gives_a_null_spread_and_no_nan():
    result = methods.one_year_cdr(arrow_cells(), n_draws=1, seed=1)
    assert result.totals["sd_ultimate_change"].null_count == 1
    assert result.origins["sd_ultimate_change"].null_count == 10
    for name in methods.OneYearCDRResult.TABLES:
        for column in getattr(result, name).columns:
            if pa.types.is_floating(column.type):
                present = column.drop_null().to_numpy()
                assert np.isfinite(present).all(), name


def test_to_polars_returns_each_table_and_refuses_others():
    pl = pytest.importorskip("polars")
    result = methods.one_year_cdr(arrow_cells(), n_draws=5, seed=1)
    for name in methods.OneYearCDRResult.TABLES:
        frame = result.to_polars(name)
        assert isinstance(frame, pl.DataFrame)
        assert frame.columns == getattr(result, name).column_names
    refusal = pytest.raises(Refusal, result.to_polars, "link_ratios").value
    assert (refusal.reason, refusal.option, refusal.method) == (
        "invalid_option",
        "table",
        "one_year_cdr",
    )


def test_the_default_quantiles_are_the_apps_default_percentiles():
    default = inspect.signature(methods.one_year_cdr).parameters["quantiles"].default
    assert default == (0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999)
    result = methods.one_year_cdr(arrow_cells(), n_draws=5, seed=1)
    total = result.quantiles.filter(pc.is_null(result.quantiles["origin"]))
    assert total["level"].to_pylist() == list(default)
    assert result.quantiles.num_rows == 11 * 7
