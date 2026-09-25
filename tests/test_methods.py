"""``ibnr.methods``: the front door for the chain ladder, Bornhuetter-Ferguson, Cape Cod and Mack.

Four things are checked. Starting from a polars DataFrame, each method gives
chainladder-python's numbers on raa and genins. Starting from a pyarrow Table,
with polars never imported, each gives exactly what the kernel path
(``cohort_grid_frame`` then ``fit_conventional_grid`` or ``fit_mack_grid``)
gives, and raa's published chain-ladder and Mack totals; those tests need
neither polars nor chainladder, so they run on the core-only leg. Every keyword
option changes the answer on a triangle where it must. And every input the
functions refuse is refused with a message that names the problem.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ibnr import methods
from ibnr.errors import Refusal
from ibnr.kernels.contract import cohort_grid_frame, grid_from_columns
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional_grid
from ibnr.kernels.mack import fit_mack_grid

# raa, the Mack (1993) triangle: accident years 1981 to 1990, annual ages 12 to 120.
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
RAA_ORIGINS = [dt.date(1981 + i, 1, 1) for i in range(10)]
#: A premium rising across the origins, so Cape Cod's decay has something to act on.
RAA_PREMIUM = dict(zip(RAA_ORIGINS, np.linspace(20000.0, 40000.0, 10).tolist(), strict=True))

#: chainladder-python 0.9.2 on raa (``cl.Chainladder``, ``cl.MackChainladder``
#: with its default log-linear sigma). The per-origin figures are tied out
#: against chainladder itself in the tieout tests below; these two totals let
#: the core leg, which has no chainladder, check the same numbers.
RAA_TOTAL_ULTIMATE = 213122.22826121017
RAA_TOTAL_MACK_SE = 26880.74032989


def columns(rows=RAA, origins=RAA_ORIGINS, *, step: int = 12) -> dict[str, list]:
    """origin_period / dev_lag / value lists, one entry per observed cell."""
    out: dict[str, list] = {"origin_period": [], "dev_lag": [], "value": []}
    for origin, values in zip(origins, rows, strict=True):
        for j, value in enumerate(values):
            out["origin_period"].append(origin)
            out["dev_lag"].append(step * (j + 1))
            out["value"].append(float(value))
    return out


def arrow_cells(rows=RAA, origins=RAA_ORIGINS, *, step: int = 12) -> pa.Table:
    data = columns(rows, origins, step=step)
    return pa.table(
        {
            "origin_period": pa.array(data["origin_period"], pa.date32()),
            "dev_lag": pa.array(data["dev_lag"], pa.int64()),
            "value": pa.array(data["value"], pa.float64()),
        }
    )


def kernel_grid(rows=RAA, origins=RAA_ORIGINS, *, step: int = 12) -> dict:
    """The same cells through the pandas grid builder, the kernel path."""
    return cohort_grid_frame(
        pd.DataFrame(columns(rows, origins, step=step)),
        dev_grain_months=step,
        measure="cumulative",
    )


def premium_table(premium: dict) -> pa.Table:
    return pa.table(
        {
            "origin_period": pa.array(list(premium), pa.date32()),
            "premium": pa.array(list(premium.values()), pa.float64()),
        }
    )


def ultimates(result) -> np.ndarray:
    return result.origins.column("ultimate").to_numpy()


def factors(result) -> list:
    return result.development.column("factor").to_pylist()


# -- chainladder-python, from polars frames -----------------------------------------


@pytest.fixture(scope="module")
def cl():
    return pytest.importorskip("chainladder")


@pytest.fixture(scope="module")
def pl():
    return pytest.importorskip("polars")


def polars_cells(cl, pl, name: str):
    """A chainladder sample as the polars frame a user would build, and the sample."""
    sample = cl.load_sample(name)
    long = sample.to_frame(keepdims=True).reset_index()
    frame = pl.DataFrame(
        {
            "origin_period": pd.to_datetime(long["origin"]).dt.date.tolist(),
            "dev_lag": long["development"].to_numpy(dtype=np.int64),
            "value": long["values"].to_numpy(dtype=float),
        }
    )
    return frame, sample


def sloped_weight(cl, sample, frame, pl):
    """A premium rising 1x to 2x, as the polars table methods take and as the
    one-column chainladder triangle its methods take as ``sample_weight``."""
    origins = sorted(set(frame["origin_period"].to_list()))
    amounts = np.linspace(1.0, 2.0, len(origins)) * float(frame["value"].max())
    weight = sample.latest_diagonal * 0
    weight.values = weight.values + amounts[None, None, :, None]
    return pl.DataFrame({"origin_period": origins, "premium": amounts}), weight


def last_column(triangle) -> np.ndarray:
    return np.asarray(triangle.values)[0, 0, :, -1]


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_chain_ladder_matches_chainladder(cl, pl, name):
    frame, sample = polars_cells(cl, pl, name)
    result = methods.chain_ladder(frame)
    reference = cl.Chainladder().fit(sample)
    np.testing.assert_allclose(ultimates(result), reference.ultimate_.values.ravel(), rtol=1e-9)
    # chainladder pads its patterns past the last age with factors of 1.0
    np.testing.assert_allclose(factors(result)[:-1], reference.ldf_.values.ravel()[:9], rtol=1e-9)
    np.testing.assert_allclose(
        result.development.column("cdf").to_numpy()[:-1],
        reference.cdf_.values.ravel()[:9],
        rtol=1e-9,
    )


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_integer_accident_years_match_chainladder(cl, pl, name):
    # the form a chainladder user most likely has: the accident year as a whole number
    frame, sample = polars_cells(cl, pl, name)
    years = frame.with_columns(pl.col("origin_period").dt.year().cast(pl.Int64))
    result = methods.chain_ladder(years)
    np.testing.assert_allclose(
        ultimates(result), cl.Chainladder().fit(sample).ultimate_.values.ravel(), rtol=1e-9
    )
    reference = cl.MackChainladder().fit(sample)
    assert methods.mack(years).totals["mack_se"][0].as_py() == pytest.approx(
        float(np.asarray(reference.total_mack_std_err_).ravel()[0]), rel=1e-9
    )
    starts = sorted(set(frame["origin_period"].to_list()))
    assert result.origins["origin"].type == pa.int64()
    assert result.origins["origin"].to_pylist() == [start.year for start in starts]
    assert result.origins["origin_period"].to_pylist() == starts
    assert result.as_of == dt.date(starts[-1].year, 12, 31)


@pytest.mark.tieout
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        ({"history_periods": 3}, {"n_periods": 3}),
        ({"average": "simple"}, {"average": "simple"}),
        ({"drop_high": True}, {"drop_high": True}),
        ({"drop_high": True, "drop_low": True}, {"drop_high": True, "drop_low": True}),
        ({"exclude": [(dt.date(1982, 1, 1), 12)]}, {"drop": ("1982", 12)}),
    ],
    ids=["history_periods", "simple", "drop_high", "drop_both", "exclude"],
)
def test_development_options_match_chainladder(cl, pl, ours, theirs):
    # drop_high on raa's last age, which has one ratio, keeps it, as chainladder does
    frame, sample = polars_cells(cl, pl, "raa")
    result = methods.chain_ladder(frame, **ours)
    development = cl.Development(**theirs).fit(sample)
    reference = cl.Chainladder().fit(development.transform(sample))
    np.testing.assert_allclose(
        factors(result)[:-1], np.asarray(development.ldf_.values).ravel()[:9], rtol=1e-12
    )
    np.testing.assert_allclose(ultimates(result), reference.ultimate_.values.ravel(), rtol=1e-9)


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_mack_matches_chainladder(cl, pl, name):
    frame, sample = polars_cells(cl, pl, name)
    result = methods.mack(frame)
    development = cl.Development().fit(sample)  # log-linear sigma, chainladder's default
    reference = cl.MackChainladder().fit(sample)
    origins = result.origins
    np.testing.assert_allclose(ultimates(result), reference.ultimate_.values.ravel(), rtol=1e-9)
    # chainladder reports NaN rather than 0 for the fully developed first origin
    np.testing.assert_allclose(
        origins.column("mack_se").to_numpy()[1:],
        last_column(reference.mack_std_err_)[1:],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        origins.column("parameter_se").to_numpy(),
        last_column(reference.parameter_risk_),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        origins.column("process_se").to_numpy(), last_column(reference.process_risk_), rtol=1e-9
    )
    totals = result.totals
    assert totals["mack_se"][0].as_py() == pytest.approx(
        float(np.asarray(reference.total_mack_std_err_).ravel()[0]), rel=1e-9
    )
    assert totals["parameter_se"][0].as_py() == pytest.approx(
        float(np.asarray(reference.total_parameter_risk_).ravel()[-1]), rel=1e-9
    )
    assert totals["process_se"][0].as_py() == pytest.approx(
        float(np.asarray(reference.total_process_risk_).ravel()[-1]), rel=1e-9
    )
    np.testing.assert_allclose(
        result.development.column("sigma").to_numpy()[:-1],
        np.asarray(development.sigma_.values).ravel(),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        result.development.column("std_err").to_numpy()[:-1],
        np.asarray(development.std_err_.values).ravel(),
        rtol=1e-9,
    )


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_bornhuetter_ferguson_matches_chainladder(cl, pl, name):
    frame, sample = polars_cells(cl, pl, name)
    premium, weight = sloped_weight(cl, sample, frame, pl)
    result = methods.bornhuetter_ferguson(frame, premium=premium, expected_loss_ratio=0.6)
    reference = cl.BornhuetterFerguson(apriori=0.6).fit(sample, sample_weight=weight)
    np.testing.assert_allclose(ultimates(result), reference.ultimate_.values.ravel(), rtol=1e-9)


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_cape_cod_matches_chainladder(cl, pl, name):
    frame, sample = polars_cells(cl, pl, name)
    premium, weight = sloped_weight(cl, sample, frame, pl)
    result = methods.cape_cod(frame, premium=premium, decay=1.0)
    # trend=0 written out (it is chainladder 0.9.2's default): this Cape Cod has no trend.
    reference = cl.CapeCod(trend=0, decay=1.0).fit(sample, sample_weight=weight)
    np.testing.assert_allclose(ultimates(result), reference.ultimate_.values.ravel(), rtol=1e-9)
    np.testing.assert_allclose(
        result.origins.column("expected_loss_ratio").to_numpy(),
        np.asarray(reference.apriori_.values).ravel(),
        rtol=1e-9,
    )


# -- from a pyarrow Table: the kernel path and raa's published totals ------------


def test_chain_ladder_from_a_pyarrow_table_gives_raa_published_total():
    result = methods.chain_ladder(arrow_cells())
    assert result.totals["ultimate"][0].as_py() == pytest.approx(RAA_TOTAL_ULTIMATE, rel=1e-12)
    assert result.as_of == dt.date(1990, 12, 31)
    assert result.method == "chain_ladder"
    assert result.dev_grain_months == 12


def test_mack_from_a_pyarrow_table_gives_raa_published_total():
    result = methods.mack(arrow_cells())
    assert result.totals["mack_se"][0].as_py() == pytest.approx(RAA_TOTAL_MACK_SE, rel=1e-9)
    assert result.totals["ultimate"][0].as_py() == pytest.approx(RAA_TOTAL_ULTIMATE, rel=1e-12)


def assert_equals_conventional_fit(result, fit) -> None:
    """Every number the result carries is the kernel fit's, exactly."""
    origins = result.origins
    assert origins.column("origin_period").to_pylist() == list(fit.origins["origin_period"])
    np.testing.assert_array_equal(
        origins.column("latest_dev_lag").to_numpy(), fit.origins["latest_dev_lag"]
    )
    np.testing.assert_array_equal(origins.column("latest").to_numpy(), fit.origins["latest"])
    np.testing.assert_array_equal(origins.column("ultimate").to_numpy(), fit.origins["ultimate"])
    np.testing.assert_array_equal(
        origins.column("ibnr").to_numpy(), fit.origins["ultimate"] - fit.origins["latest"]
    )
    totals = result.totals
    assert totals["ultimate"][0].as_py() == pytest.approx(fit.origins["ultimate"].sum(), rel=1e-12)
    assert totals["ibnr"][0].as_py() == pytest.approx(
        (fit.origins["ultimate"] - fit.origins["latest"]).sum(), rel=1e-12
    )
    if "expected_loss_ratio" in origins.column_names:
        np.testing.assert_array_equal(
            origins.column("expected_loss_ratio").to_numpy(), fit.origins["expected_loss_ratio"]
        )
    assert factors(result) == [*fit.factors.tolist(), None]
    np.testing.assert_allclose(
        result.development.column("pct_reported").to_numpy(), fit.beta, rtol=1e-15
    )
    summary = fit.factor_summary
    assert (
        result.development.column("n_selected").to_pylist()[:-1] == summary["n_selected"].tolist()
    )
    selection = fit.factor_selection
    ratios = result.link_ratios
    assert ratios.column("reason").to_pylist() == selection["reason"].tolist()
    assert ratios.column("included").to_pylist() == selection["included"].tolist()
    assert ratios.column("origin_period").to_pylist() == selection["origin_period"].tolist()
    np.testing.assert_array_equal(
        ratios.column("ratio").to_numpy(zero_copy_only=False), selection["ratio"]
    )
    assert result.as_of == fit.as_of


def test_chain_ladder_equals_the_kernel_path():
    options = dict(history_periods=5, drop_high=True, average="simple")
    result = methods.chain_ladder(arrow_cells(), **options)
    candidate = ConventionalCandidate(exhausted_exclusions="keep", **options)
    assert_equals_conventional_fit(result, fit_conventional_grid(kernel_grid(), candidate))


def test_bornhuetter_ferguson_equals_the_kernel_path():
    result = methods.bornhuetter_ferguson(
        arrow_cells(), premium=premium_table(RAA_PREMIUM), expected_loss_ratio=0.7
    )
    candidate = ConventionalCandidate("bf", expected_loss_ratio=0.7, exhausted_exclusions="keep")
    fit = fit_conventional_grid(kernel_grid(), candidate, premium=RAA_PREMIUM)
    assert_equals_conventional_fit(result, fit)


def test_cape_cod_equals_the_kernel_path():
    result = methods.cape_cod(arrow_cells(), premium=premium_table(RAA_PREMIUM), decay=0.5)
    candidate = ConventionalCandidate("gcc", decay=0.5, exhausted_exclusions="keep")
    fit = fit_conventional_grid(kernel_grid(), candidate, premium=RAA_PREMIUM)
    assert_equals_conventional_fit(result, fit)


@pytest.mark.parametrize("rule", ["log_linear", "mack"])
def test_mack_equals_the_kernel_path(rule):
    result = methods.mack(arrow_cells(), sigma_rule=rule)
    fit = fit_mack_grid(kernel_grid(), sigma_rule=rule)
    risk = fit.msep_runoff()
    origins = result.origins
    np.testing.assert_array_equal(origins.column("ultimate").to_numpy(), fit.ultimate)
    np.testing.assert_array_equal(origins.column("latest").to_numpy(), fit.latest)
    np.testing.assert_array_equal(origins.column("ibnr").to_numpy(), fit.ultimate - fit.latest)
    totals = result.totals
    assert totals["ibnr"][0].as_py() == pytest.approx(
        totals["ultimate"][0].as_py() - totals["latest"][0].as_py(), rel=1e-15
    )
    assert totals["ibnr"][0].as_py() == pytest.approx((fit.ultimate - fit.latest).sum(), rel=1e-12)
    np.testing.assert_array_equal(origins.column("mack_se").to_numpy(), np.sqrt(risk["msep"]))
    np.testing.assert_array_equal(
        origins.column("parameter_se").to_numpy(), np.sqrt(risk["parameter"])
    )
    np.testing.assert_array_equal(origins.column("process_se").to_numpy(), np.sqrt(risk["process"]))
    assert result.totals["mack_se"][0].as_py() == np.sqrt(risk["msep_total"])
    assert result.totals["parameter_se"][0].as_py() == np.sqrt(risk["parameter_total"])
    assert result.totals["process_se"][0].as_py() == np.sqrt(risk["process_total"])
    assert factors(result) == [*fit.f.tolist(), None]
    assert result.development.column("sigma").to_pylist() == [*np.sqrt(fit.sigma2).tolist(), None]
    # the factor's standard error, sqrt(sigma^2 / S_j), S_j the volume behind the factor
    assert result.development.column("std_err").to_pylist() == [
        *np.sqrt(fit.sigma2 / fit.s).tolist(),
        None,
    ]
    assert result.as_of == dt.date(1990, 12, 31)


def test_the_total_mack_se_is_not_the_sum_of_the_origins():
    # The origins share the estimated factors, so their errors are correlated.
    result = methods.mack(arrow_cells())
    origins, totals = result.origins, result.totals
    total = totals["mack_se"][0].as_py()
    assert total > np.sqrt((origins.column("mack_se").to_numpy() ** 2).sum()) * 1.01
    assert total**2 == pytest.approx(
        totals["parameter_se"][0].as_py() ** 2 + totals["process_se"][0].as_py() ** 2, rel=1e-12
    )


def test_methods_does_not_import_polars():
    script = textwrap.dedent(
        """
        import sys
        import pyarrow as pa
        from ibnr import methods
        cells = pa.table({
            "origin_period": pa.array(["2010-01-01", "2010-01-01", "2011-01-01"]),
            "dev_lag": [12, 24, 12],
            "value": [100.0, 150.0, 120.0],
        })
        result = methods.chain_ladder(cells)
        assert result.totals["ultimate"][0].as_py() == 150.0 + 120.0 * 1.5
        print("polars" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout
    assert out.strip() == "False"


def test_a_bare_import_of_ibnr_does_not_load_methods():
    script = "import sys, ibnr; print('ibnr.methods' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout
    assert out.strip() == "False"
    assert "methods" not in __import__("ibnr").__all__


def test_the_public_names_are_pinned():
    assert sorted(methods.__all__) == [
        "Refusal",
        "ReserveResult",
        "benktander",
        "bornhuetter_ferguson",
        "cape_cod",
        "chain_ladder",
        "mack",
    ]


# -- what the functions accept ------------------------------------------------------


def without_origin(table: pa.Table) -> pa.Table:
    return table.drop_columns(["origin"])


def assert_same_numbers(result, expected) -> None:
    """Every table and the information date agree, apart from the echoed labels."""
    assert result.as_of == expected.as_of
    assert without_origin(result.origins).equals(without_origin(expected.origins))
    assert result.development.equals(expected.development)
    assert result.totals.equals(expected.totals)
    if expected.link_ratios is None:
        assert result.link_ratios is None
    else:
        assert without_origin(result.link_ratios).equals(without_origin(expected.link_ratios))


def assert_echoes(result, labels: pa.Array) -> None:
    """``origin`` is the caller's label for each period, one per origin in origin order,
    in both tables that carry it, with the Arrow type the caller's column decodes to."""
    assert result.origins.column_names[0] == "origin"
    assert result.origins["origin"].type == labels.type
    assert result.origins["origin"].equals(pa.chunked_array([labels]))
    if result.link_ratios is not None:
        assert result.link_ratios.column_names[0] == "origin"
        starts = result.origins["origin_period"].to_pylist()
        position = [starts.index(o) for o in result.link_ratios["origin_period"].to_pylist()]
        expected = labels.take(pa.array(position, pa.int64()))
        assert result.link_ratios["origin"].equals(pa.chunked_array([expected]))


def first_rows(origins: list) -> list[int]:
    """The row each origin first appears in, in origin order."""
    return [origins.index(o) for o in sorted(set(origins))]


def one_label_per_origin(origin: pa.Array, kind: pa.DataType) -> pa.Array:
    """What the ``origin`` column should hold for a cells column ``origin``."""
    if pa.types.is_dictionary(origin.type):
        origin = origin.dictionary_decode()
    rows = first_rows(columns()["origin_period"])
    # cast first: pyarrow cannot take from a string_view array
    return origin.cast(kind).take(pa.array(rows, pa.int64()))


def origin_variants():
    """The same accident years, 1981 to 1990, in each form the functions read, with
    the Arrow type the result echoes each in."""
    dates = columns()["origin_period"]
    ends = [dt.date(o.year, 12, 31) for o in dates]
    years = [o.year for o in dates]
    yield "date32", pa.array(dates, pa.date32()), pa.date32()
    yield "date64", pa.array(dates, pa.date64()), pa.date32()
    yield "year_end_date32", pa.array(ends, pa.date32()), pa.date32()
    yield (
        "timestamp",
        pa.array([dt.datetime(o.year, 1, 1) for o in dates], pa.timestamp("us")),
        pa.timestamp("us"),
    )
    # Midnight on 1 January in Tokyo is 15:00 on 31 December in UTC. The date
    # read is Tokyo's, the zone the column is in; reading the UTC date would
    # put every origin on 31 December.
    tokyo = pa.timestamp("us", tz="Asia/Tokyo")
    yield (
        "timestamp_tz",
        pa.array([dt.datetime(o.year - 1, 12, 31, 15, tzinfo=dt.UTC) for o in dates], tokyo),
        tokyo,
    )
    # 23:00 on 31 December in Tokyo is 14:00 the same day in UTC: the year's last day
    yield (
        "year_end_timestamp_tz",
        pa.array([dt.datetime(o.year, 12, 31, 14, tzinfo=dt.UTC) for o in dates], tokyo),
        tokyo,
    )
    # a time of day is dropped: the date part is the origin period
    yield (
        "timestamp_time_of_day",
        pa.array([dt.datetime(o.year, 1, 1, 6) for o in dates], pa.timestamp("s")),
        pa.timestamp("s"),
    )
    yield "string", pa.array([o.isoformat() for o in dates], pa.string()), pa.string()
    yield (
        "year_end_string",
        pa.array([o.isoformat() for o in ends], pa.string()),
        pa.string(),
    )
    yield "year_string", pa.array([str(y) for y in years], pa.string()), pa.string()
    yield (
        "large_string",
        pa.array([o.isoformat() for o in dates], pa.large_string()),
        pa.string(),
    )
    if hasattr(pa, "string_view"):  # pyarrow 16 and later; the floor is 15
        yield (
            "string_view",
            pa.array([str(y) for y in years], pa.string()).cast(pa.string_view()),
            pa.string(),
        )
    yield (
        "dictionary_string",
        pa.array([o.isoformat() for o in dates], pa.string()).dictionary_encode(),
        pa.string(),
    )
    yield "dictionary_date", pa.array(dates, pa.date32()).dictionary_encode(), pa.date32()
    yield "int64_year", pa.array(years, pa.int64()), pa.int64()
    yield "int16_year", pa.array(years, pa.int16()), pa.int64()
    yield "uint32_year", pa.array(years, pa.uint32()), pa.int64()
    yield "dictionary_year", pa.array(years, pa.int64()).dictionary_encode(), pa.int64()


VARIANTS = list(origin_variants())


@pytest.mark.parametrize(
    ("kind", "origin", "echo"), VARIANTS, ids=[kind for kind, _, _ in VARIANTS]
)
@pytest.mark.parametrize("method", ["chain_ladder", "bornhuetter_ferguson", "cape_cod", "mack"])
def test_every_accepted_origin_form_gives_the_same_answer(method, kind, origin, echo):
    expected = run(method)
    result = run(method, arrow_cells().set_column(0, "origin_period", origin))
    assert_same_numbers(result, expected)
    assert result.origins["origin_period"].to_pylist() == RAA_ORIGINS
    assert_echoes(result, one_label_per_origin(origin, echo))


POLARS_FORMS = [
    "date",
    "year_end_date",
    "datetime",
    "string",
    "year_end_string",
    "year_string",
    "categorical",
    "enum",
    "int64_year",
]


@pytest.mark.parametrize("kind", POLARS_FORMS)
def test_polars_origin_forms_give_the_same_answer(pl, kind):
    expected = methods.chain_ladder(arrow_cells())
    data = columns()
    starts = data["origin_period"]
    origins = {
        "date": starts,
        "year_end_date": [dt.date(o.year, 12, 31) for o in starts],
        "datetime": [dt.datetime(o.year, o.month, o.day) for o in starts],
        "year_end_string": [f"{o.year}-12-31" for o in starts],
        "year_string": [str(o.year) for o in starts],
        "int64_year": [o.year for o in starts],
    }.get(kind, [o.isoformat() for o in starts])
    frame = pl.DataFrame({**data, "origin_period": origins})
    if kind == "categorical":
        frame = frame.with_columns(pl.col("origin_period").cast(pl.Categorical))
    elif kind == "enum":
        frame = frame.with_columns(pl.col("origin_period").cast(pl.Enum(sorted(set(origins)))))
    result = methods.chain_ladder(frame)
    assert_same_numbers(result, expected)
    labels = [origins[i] for i in first_rows(starts)]
    assert result.origins["origin"].to_pylist() == labels
    kind_echoed = {
        "date": pa.date32(),
        "year_end_date": pa.date32(),
        "datetime": pa.timestamp("us"),
        "int64_year": pa.int64(),
    }.get(kind, pa.string())
    assert result.origins["origin"].type == kind_echoed


def test_a_record_batch_and_extra_columns_are_accepted():
    expected = methods.chain_ladder(arrow_cells())
    table = arrow_cells().append_column("company", pa.array(["a"] * arrow_cells().num_rows))
    assert methods.chain_ladder(table).origins.equals(expected.origins)
    batch = arrow_cells().to_batches()[0]
    assert methods.chain_ladder(batch).origins.equals(expected.origins)


def test_whole_float_ages_and_integer_values_are_accepted():
    expected = methods.chain_ladder(arrow_cells())
    table = arrow_cells()
    table = table.set_column(1, "dev_lag", pc.cast(table["dev_lag"], pa.float64()))
    table = table.set_column(2, "value", pc.cast(table["value"], pa.int64()))
    assert methods.chain_ladder(table).origins.equals(expected.origins)


def test_premium_as_a_dict_or_a_table_gives_the_same_answer(pl):
    by_dict = methods.bornhuetter_ferguson(
        arrow_cells(), premium=RAA_PREMIUM, expected_loss_ratio=0.7
    )
    by_table = methods.bornhuetter_ferguson(
        arrow_cells(), premium=premium_table(RAA_PREMIUM), expected_loss_ratio=0.7
    )
    frame = pl.DataFrame(
        {"origin_period": list(RAA_PREMIUM), "premium": list(RAA_PREMIUM.values())}
    )
    by_polars = methods.bornhuetter_ferguson(arrow_cells(), premium=frame, expected_loss_ratio=0.7)
    assert by_dict.origins.equals(by_table.origins)
    assert by_dict.origins.equals(by_polars.origins)


# -- what the functions refuse ------------------------------------------------------


def with_column(name: str, values, kind=None) -> pa.Table:
    table = arrow_cells()
    return table.set_column(table.column_names.index(name), name, pa.array(values, kind))


N = sum(len(row) for row in RAA)
#: A row in the middle of the table (1984 at 60 months), so a check that looks
#: only at the first row cannot pass.
MIDDLE = N // 2


def one_bad(good, bad, at: int = MIDDLE) -> list:
    return [bad if i == at else good for i in range(N)]


@pytest.mark.parametrize("missing", ["origin_period", "dev_lag", "value"])
def test_a_missing_column_is_refused_by_name(missing):
    cells = arrow_cells().drop_columns([missing])
    with pytest.raises(ValueError, match=rf"cells is missing column\(s\) \['{missing}'\]"):
        methods.chain_ladder(cells)


@pytest.mark.parametrize(
    ("cells", "match"),
    [
        (
            with_column("dev_lag", one_bad(12.0, 12.5)),
            r"whole numbers of months, got \[12.5\]",
        ),
        (with_column("dev_lag", ["12"] * N), "dev_lag must be a column of whole numbers"),
        (with_column("dev_lag", one_bad(12, None), pa.int64()), "dev_lag has 1 missing"),
        (
            with_column("value", one_bad(1.0, None), pa.float64()),
            "value has 1 null value.*left out of cells, not given a null value",
        ),
        (
            with_column("value", one_bad(1.0, float("nan")), pa.float64()),
            "value has 1 NaN value.*left out of cells",
        ),
        (with_column("value", ["1"] * N), "value must be a numeric column, got string"),
        (
            with_column("origin_period", one_bad(dt.date(1981, 1, 1), None), pa.date32()),
            "origin_period has 1 missing value",
        ),
        (
            with_column("origin_period", one_bad(1981, None), pa.int64()),
            "origin_period has 1 missing value",
        ),
        (
            with_column("origin_period", one_bad("1981", None), pa.string()),
            "origin_period has 1 missing value",
        ),
        (
            with_column("origin_period", [1981.0] * N),
            "origin_period must be a column of integer years, text labels, dates or "
            r"timestamps, got double; write it as a year \(2020\)",
        ),
    ],
    ids=[
        "fractional_age",
        "text_age",
        "null_age",
        "null_value",
        "nan_value",
        "text_value",
        "null_origin",
        "null_year",
        "null_label",
        "float_origin",
    ],
)
def test_a_bad_column_is_refused_by_name(cells, match):
    with pytest.raises(ValueError, match=match):
        methods.chain_ladder(cells)


def test_two_cohorts_rows_are_refused_one_cohort_at_a_time():
    both = pa.concat_tables([arrow_cells(), arrow_cells()])
    with pytest.raises(ValueError, match="one triangle at a time.*refused rather than added"):
        methods.chain_ladder(both)


def years(*first_years: int) -> list:
    return [dt.date(y, 1, 1) for y in first_years]


#: raa's origins written three ways, with how a message names the 1985 origin in each
#: (a text label is shown without quotes, as the caller's text)
ORIGIN_SPELLINGS = pytest.mark.parametrize(
    ("spell", "shown"),
    [(lambda o: o, "1985-01-01"), (lambda o: o.year, "1985"), (lambda o: str(o.year), "1985")],
    ids=["date", "int_year", "text_year"],
)


def relabelled(cells: pa.Table, spell) -> pa.Table:
    """``cells`` with each row's origin written by ``spell``."""
    by_row = [spell(o) for o in cells["origin_period"].to_pylist()]
    return cells.set_column(0, "origin_period", pa.array(by_row))


@ORIGIN_SPELLINGS
@pytest.mark.parametrize("method", ["chain_ladder", "mack"])
def test_a_negative_cumulative_is_refused_naming_the_cells(method, spell, shown):
    rows = [list(row) for row in RAA]
    rows[4][1], rows[4][2] = -50, -80
    with pytest.raises(
        Refusal,
        match=rf"value is negative in 2 cell\(s\): \({shown}, 24 months\) and "
        rf"\({shown}, 36 months\)\.",
    ) as refused:
        getattr(methods, method)(relabelled(arrow_cells(rows), spell))
    assert refused.value.reason == "negative_cumulative"
    assert [cell.origin for cell in refused.value.cells] == [spell(dt.date(1985, 1, 1))] * 2
    assert [cell.value for cell in refused.value.cells] == [-50.0, -80.0]


@ORIGIN_SPELLINGS
def test_two_rows_for_one_cell_are_named_in_the_callers_labels(spell, shown):
    cells = arrow_cells()
    data = columns()
    at = [*zip(data["origin_period"], data["dev_lag"], strict=True)].index(
        (dt.date(1985, 1, 1), 24)
    )
    doubled = pa.concat_tables([cells, cells.slice(at, 1)])
    with pytest.raises(
        Refusal,
        match=rf"more than one row for the same cell: \({shown}, 24 months\) and "
        rf"\({shown}, 24 months\)",
    ) as refused:
        methods.chain_ladder(relabelled(doubled, spell))
    assert refused.value.reason == "duplicate"


@pytest.mark.parametrize(
    ("rows", "origins"),
    [
        # 2020 on the 2022 diagonal, one step short of where 2023's cell puts the triangle
        ([[5, 7, 8], [6, 9], [4]], years(2020, 2022, 2023)),
        # 2020 on the 2023 diagonal, a correct staircase in calendar terms
        ([[5, 7, 8, 9], [6, 9], [4]], years(2020, 2022, 2023)),
    ],
    ids=["off_the_diagonal", "on_the_diagonal"],
)
@pytest.mark.parametrize(
    ("spell", "first", "second"),
    [
        (lambda o: o, "2020-01-01", "2022-01-01"),
        (lambda o: o.year, "2020", "2022"),
        (lambda o: dt.date(o.year, 12, 31), "2020-12-31", "2022-12-31"),
    ],
    ids=["date", "int_year", "year_end"],
)
@pytest.mark.parametrize("method", ["chain_ladder", "mack"])
def test_a_missing_origin_period_is_refused_by_name(method, rows, origins, spell, first, second):
    # the two neighbours are named as the caller wrote them; the missing period has
    # no label, so it is named by its first day
    with pytest.raises(
        ValueError,
        match=rf"origin periods {first} and {second} are 24 months apart, but "
        r"dev_grain_months=12.*cells has no rows for the period\(s\) starting 2021-01-01: ",
    ):
        getattr(methods, method)(relabelled(arrow_cells(rows, origins), spell))


def test_annual_origins_developed_quarterly_are_refused_by_name():
    rows = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5], [1]]
    cells = arrow_cells(rows, years(2020, 2021, 2022), step=3)
    with pytest.raises(
        ValueError,
        match=r"are 12 months apart, but dev_grain_months=3.*annual origins developed "
        "quarterly, for example\\) are not supported yet",
    ):
        methods.chain_ladder(cells, dev_grain_months=3)


def test_mack_refuses_a_triangle_with_no_sigma_to_estimate():
    # two origins: one link ratio at the only age, so Mack's sigma has no spread to measure
    cells = arrow_cells([[5, 7], [6]], years(2010, 2011))
    with pytest.raises(ValueError, match="mack needs at least one development age with two or"):
        methods.mack(cells)
    assert methods.chain_ladder(cells).totals["ultimate"][0].as_py() == pytest.approx(7 + 6 * 1.4)
    # one more origin gives the first age two ratios, and every sigma a value
    three = methods.mack(arrow_cells([[5, 7], [6, 9], [4]], years(2010, 2011, 2012)))
    assert three.totals["mack_se"][0].as_py() > 0


def test_cells_that_are_not_a_table_are_refused():
    with pytest.raises(ValueError, match="cells must be a table Arrow can read.*not list"):
        methods.chain_ladder([1, 2, 3])


def test_empty_cells_are_refused():
    with pytest.raises(ValueError, match="cells has no rows"):
        methods.chain_ladder(arrow_cells().slice(0, 0))


@pytest.mark.parametrize("grain", [0, 12.0, True, "12"])
def test_a_bad_dev_grain_is_refused(grain):
    with pytest.raises(ValueError, match="dev_grain_months must be a positive whole number"):
        methods.chain_ladder(arrow_cells(), dev_grain_months=grain)


@pytest.mark.parametrize(
    ("premium", "match"),
    [
        (
            pa.table({"origin": list(RAA_PREMIUM), "premium": list(RAA_PREMIUM.values())}),
            r"premium is missing column\(s\) \['origin_period'\]",
        ),
        (
            pa.table({"origin_period": list(RAA_PREMIUM), "amount": list(RAA_PREMIUM.values())}),
            r"premium is missing column\(s\) \['premium'\]",
        ),
        (
            pa.concat_tables([premium_table(RAA_PREMIUM), premium_table(RAA_PREMIUM).slice(0, 1)]),
            r"premium has more than one row for origin period\(s\) 1981-01-01$",
        ),
        (
            premium_table(RAA_PREMIUM).slice(1),
            r"premium has no amount for origin\(s\) 1981-01-01$",
        ),
        (
            pa.table(
                {
                    "origin_period": pa.array(list(RAA_PREMIUM), pa.date32()),
                    "premium": pa.array(
                        [None if i == 5 else p for i, p in enumerate(RAA_PREMIUM.values())],
                        pa.float64(),
                    ),
                }
            ),
            "premium has 1 null value",
        ),
        (
            pa.table(
                {
                    "origin_period": pa.array(list(RAA_PREMIUM), pa.date32()),
                    "premium": pa.array(
                        [np.nan if i == 5 else p for i, p in enumerate(RAA_PREMIUM.values())],
                        pa.float64(),
                    ),
                }
            ),
            "premium has 1 NaN value",
        ),
        ([1.0] * 10, "premium must be a table Arrow can read"),
    ],
    ids=["no_origin", "no_premium", "repeated", "short", "null", "nan", "list"],
)
def test_bad_premium_is_refused_by_name(premium, match):
    with pytest.raises(ValueError, match=match):
        methods.cape_cod(arrow_cells(), premium=premium)


@pytest.mark.parametrize(
    ("origin", "shown"),
    [
        (dt.date(1990, 1, 1), "1990-01-01"),
        # named as the caller wrote it, not as the date it was read as
        (1990, "1990"),
        (dt.date(1990, 12, 31), "1990-12-31"),
        ("1990", "1990"),
    ],
    ids=["date", "int_year", "year_end", "text_year"],
)
def test_an_exclusion_the_triangle_does_not_have_is_refused(origin, shown):
    with pytest.raises(
        Refusal,
        match=rf"exclude names link ratio\(s\) \({shown}, 12 months\) that the triangle",
    ) as refused:
        methods.chain_ladder(arrow_cells(), exclude=[(origin, 12)])
    assert refused.value.reason == "not_in_triangle"
    assert refused.value.cells[0].origin == origin  # as written in exclude, value and type
    assert type(refused.value.cells[0].origin) is type(origin)


def test_every_link_ratio_of_raa_can_be_excluded():
    """Each of raa's 45 link ratios, the first and the last included, is accepted
    as an exclusion and marked as excluded. Mutation: build the set of known
    links from all but the first row of the selection; the first is refused."""
    plain = methods.chain_ladder(arrow_cells())
    links = plain.link_ratios.select(["origin_period", "from_dev_lag"]).to_pylist()
    assert len(links) == 45
    assert {"origin_period": dt.date(1981, 1, 1), "from_dev_lag": 12} in links
    assert {"origin_period": dt.date(1981, 1, 1), "from_dev_lag": 108} in links
    for link in links:
        pair = (link["origin_period"], link["from_dev_lag"])
        result = methods.chain_ladder(arrow_cells(), exclude=[pair], unsupported_factor="unity")
        rows = [
            row
            for row in result.link_ratios.to_pylist()
            if (row["origin_period"], row["from_dev_lag"]) == pair
        ]
        assert [(row["included"], row["reason"]) for row in rows] == [
            (False, "explicit_exclusion")
        ], pair


def test_a_forecast_that_overflows_is_refused():
    """Cumulatives near the largest double give a link ratio that is finite and
    an ultimate that is not (1.2e308 x 1.5 overflows). Mutation: delete the
    finiteness check at the end of ``conventional._estimate``; the chain ladder
    then returns an infinite ultimate."""
    cells = pa.table(
        {
            "origin_period": pa.array([2000, 2000, 2001], pa.int64()),
            "dev_lag": pa.array([12, 24, 12], pa.int64()),
            "value": pa.array([1e308, 1.5e308, 1.2e308], pa.float64()),
        }
    )
    with pytest.raises(Refusal, match="the ultimate for 2001 is not a finite number") as refused:
        methods.chain_ladder(cells)
    assert refused.value.reason == "result_not_finite"
    assert [cell.origin for cell in refused.value.cells] == [2001]


@pytest.mark.parametrize(
    ("exclude", "match"),
    [
        (
            [(1982, 12), ("1982-12-31", 12)],
            r"exclude names one link ratio twice, as \(1982, 12 months\) and "
            r"\('1982-12-31', 12 months\); list each \(origin_period, dev_lag\) pair once",
        ),
        (
            [(1982, 12), (dt.date(1983, 1, 1), 24), (1982, np.int64(12))],
            r"twice, as \(1982, 12 months\) and \(1982, 12 months\)",
        ),
        (
            # two equal literal tuples, which Python may hand over as one object
            [(1982, 12), (1982, 12)],
            r"twice, as \(1982, 12 months\) and \(1982, 12 months\)",
        ),
    ],
    ids=["two_spellings", "the_same_pair_again", "the_same_literal_twice"],
)
def test_one_link_ratio_excluded_twice_is_refused_as_written(exclude, match):
    with pytest.raises(ValueError, match=match):
        methods.chain_ladder(spelled(lambda o: o.year), exclude=exclude)


@pytest.mark.parametrize(
    "exclude", ["1981-01-01", [dt.date(1981, 1, 1)], (dt.date(1981, 1, 1), 12)]
)
def test_an_exclusion_that_is_not_a_list_of_pairs_is_refused(exclude):
    with pytest.raises(ValueError, match="exclu"):
        methods.chain_ladder(arrow_cells(), exclude=exclude)


def test_mack_refuses_an_unknown_sigma_rule():
    with pytest.raises(ValueError, match="sigma_rule must be one of"):
        methods.mack(arrow_cells(), sigma_rule="log-linear")


# -- origin labels: the forms, the echo, and what is refused ----------------------

#: raa's accident years written as whole numbers, the form a chainladder user has
RAA_YEARS = [o.year for o in RAA_ORIGINS]


def spelled(spell, kind=None, *, bad=None, at: int = MIDDLE) -> pa.Table:
    """raa's cells with each origin written by ``spell``, one row optionally replaced."""
    labels = [spell(o) for o in columns()["origin_period"]]
    if bad is not None:
        labels[at] = bad
    return with_column("origin_period", labels, kind)


def year_end(origin: dt.date) -> dt.date:
    return dt.date(origin.year, 12, 31)


def last_day(first: dt.date) -> dt.date:
    """The last day of the month ``first`` starts."""
    following = dt.date(first.year + first.month // 12, first.month % 12 + 1, 1)
    return following - dt.timedelta(days=1)


def quarter_end(start: dt.date) -> dt.date:
    return last_day(dt.date(start.year, start.month + 2, 1))


#: Five quarterly origins, 2020Q1 to 2021Q1, developed quarterly.
QUARTERS = [dt.date(2020, 1 + 3 * q, 1) for q in range(4)] + [dt.date(2021, 1, 1)]
QUARTER_ROWS = [
    [100.0, 180.0, 220.0, 240.0, 250.0],
    [110.0, 200.0, 230.0, 250.0],
    [90.0, 170.0, 190.0],
    [120.0, 210.0],
    [130.0],
]
QUARTER_PREMIUM = [500.0, 600.0, 700.0, 800.0, 900.0]


def quarter_label(start: dt.date) -> str:
    return f"{start.year}Q{(start.month + 2) // 3}"


@pytest.mark.parametrize(
    ("spell", "echo"),
    [
        (quarter_label, pa.string()),
        (lambda o: o, pa.date32()),
        (quarter_end, pa.date32()),
        (lambda o: quarter_end(o).isoformat(), pa.string()),
    ],
    ids=["quarter_label", "quarter_start", "quarter_end", "quarter_end_string"],
)
@pytest.mark.parametrize("method", ["chain_ladder", "mack"])
def test_quarterly_origins_written_each_way_agree(method, spell, echo):
    function = getattr(methods, method)
    cells = arrow_cells(QUARTER_ROWS, QUARTERS, step=3)
    expected = function(cells, dev_grain_months=3)
    written = [spell(o) for o in QUARTERS]
    by_row = [spell(o) for o in cells["origin_period"].to_pylist()]
    result = function(cells.set_column(0, "origin_period", pa.array(by_row)), dev_grain_months=3)
    assert_same_numbers(result, expected)
    assert result.origins["origin_period"].to_pylist() == QUARTERS
    assert result.as_of == dt.date(2021, 3, 31)
    assert_echoes(result, pa.array(written, echo))


@pytest.mark.parametrize(
    "premium",
    [
        {quarter_label(o): p for o, p in zip(QUARTERS, QUARTER_PREMIUM, strict=True)},
        {quarter_end(o): p for o, p in zip(QUARTERS, QUARTER_PREMIUM, strict=True)},
        pa.table(
            {
                "origin_period": [quarter_end(o).isoformat() for o in QUARTERS],
                "premium": QUARTER_PREMIUM,
            }
        ),
    ],
    ids=["dict_of_labels", "dict_of_quarter_ends", "table_of_quarter_ends"],
)
def test_quarterly_premium_is_read_with_the_quarterly_step(premium):
    # a quarter's last day is read with dev_grain_months=3, never as a year's end
    cells = arrow_cells(QUARTER_ROWS, QUARTERS, step=3)
    by_start = dict(zip(QUARTERS, QUARTER_PREMIUM, strict=True))
    expected = methods.cape_cod(cells, premium=by_start, dev_grain_months=3)
    result = methods.cape_cod(cells, premium=premium, dev_grain_months=3)
    assert_same_numbers(result, expected)


def monthly_cells() -> pa.Table:
    origins = [dt.date(2020, 1 + k, 1) for k in range(4)]
    rows = [[100.0, 180.0, 220.0, 240.0], [110.0, 200.0, 230.0], [90.0, 170.0], [120.0]]
    return arrow_cells(rows, origins, step=1)


@pytest.mark.parametrize(
    ("step", "origin"),
    [
        (3, "2020Q2"),
        (3, dt.date(2020, 4, 1)),
        (3, dt.date(2020, 6, 30)),
        (3, "2020-06-30"),
        (1, "2020-02"),
        (1, dt.date(2020, 2, 1)),
        (1, dt.date(2020, 2, 29)),
    ],
    ids=[
        "quarter_label",
        "quarter_start",
        "quarter_end",
        "quarter_end_string",
        "month_label",
        "month_start",
        "month_end",
    ],
)
def test_an_exclusion_is_read_with_the_triangles_step(step, origin):
    # the second origin's first link ratio, however the exclusion writes that origin
    cells = arrow_cells(QUARTER_ROWS, QUARTERS, step=3) if step == 3 else monthly_cells()
    second = cells["origin_period"].to_pylist()[len(QUARTER_ROWS[0]) if step == 3 else 4]
    expected = methods.chain_ladder(cells, dev_grain_months=step, exclude=[(second, step)])
    result = methods.chain_ladder(cells, dev_grain_months=step, exclude=[(origin, step)])
    assert_same_numbers(result, expected)
    excluded = result.link_ratios.filter(
        pc.equal(result.link_ratios["reason"], "explicit_exclusion")
    )
    assert excluded["origin_period"].to_pylist() == [second]
    assert excluded["from_dev_lag"].to_pylist() == [step]
    plain = methods.chain_ladder(cells, dev_grain_months=step)
    assert factors(result) != factors(plain)


@pytest.mark.parametrize(
    ("premium", "exclude", "match"),
    [
        (
            {2020: 1.0},
            (),
            r"premium origin 2020 is a year, 12 months long, but the triangle's origin periods "
            r"are 3 months long \(dev_grain_months=3\)\. Write it as one of the triangle's "
            "origin periods, as a label of that length or as the period's first or last day$",
        ),
        (
            dict(zip(QUARTERS, QUARTER_PREMIUM, strict=True)),
            [("2020-03", 3)],
            r"exclude origin '2020-03' is a month, 1 month long, but the triangle's origin "
            r"periods are 3 months long \(dev_grain_months=3\)",
        ),
    ],
    ids=["premium_by_year", "exclude_by_month"],
)
def test_premium_and_exclusions_are_told_to_follow_the_cells(premium, exclude, match):
    # the cells set the period length, so the advice is to rewrite the origin, not the step
    cells = arrow_cells(QUARTER_ROWS, QUARTERS, step=3)
    with pytest.raises(ValueError, match=match) as refused:
        methods.cape_cod(cells, premium=premium, dev_grain_months=3, exclude=exclude)
    assert "pass dev_grain_months" not in str(refused.value)


@pytest.mark.parametrize("spell", [lambda o: o.year, year_end, lambda o: str(o.year)])
@pytest.mark.parametrize("order", ["newest_first", "shuffled"])
def test_the_echoed_label_follows_each_rows_period_not_its_position(order, spell):
    # rows arrive out of order, so the first label seen is not the first period
    cells = arrow_cells()
    if order == "newest_first":
        rows = np.arange(cells.num_rows)[::-1]
    else:
        rows = np.random.default_rng(2020).permutation(cells.num_rows)
    shuffled = relabelled(cells.take(pa.array(rows)), spell)
    premium = premium_table(RAA_PREMIUM).take(pa.array([9, 3, 0, 7, 1, 8, 2, 6, 4, 5]))
    expected = methods.cape_cod(arrow_cells(), premium=RAA_PREMIUM)
    result = methods.cape_cod(shuffled, premium=premium)
    assert_same_numbers(result, expected)
    assert result.origins["origin"].to_pylist() == [spell(o) for o in RAA_ORIGINS]
    for table in (result.origins, result.link_ratios):
        starts = table["origin_period"].to_pylist()
        assert table["origin"].to_pylist() == [spell(o) for o in starts]


@pytest.mark.parametrize(
    ("spell", "echo"),
    [
        (lambda o: f"{o.year}-{o.month:02d}", pa.string()),
        (lambda o: o, pa.date32()),
        (last_day, pa.date32()),
    ],
    ids=["month_label", "month_start", "month_end"],
)
# January to April 2020 has 29 February; December 2020 to March 2021 has 28 February
@pytest.mark.parametrize(
    ("first", "as_of"),
    [(dt.date(2020, 1, 1), dt.date(2020, 4, 30)), (dt.date(2020, 12, 1), dt.date(2021, 3, 31))],
    ids=["leap_year", "common_year"],
)
def test_monthly_origins_written_each_way_agree(spell, echo, first, as_of):
    origins = [first]
    for _ in range(3):
        origins.append(last_day(origins[-1]) + dt.timedelta(days=1))
    rows = [[100.0, 180.0, 220.0, 240.0], [110.0, 200.0, 230.0], [90.0, 170.0], [120.0]]
    cells = arrow_cells(rows, origins, step=1)
    expected = methods.chain_ladder(cells, dev_grain_months=1)
    result = methods.chain_ladder(relabelled(cells, spell), dev_grain_months=1)
    assert_same_numbers(result, expected)
    assert result.origins["origin_period"].to_pylist() == origins
    assert result.as_of == as_of
    assert_echoes(result, pa.array([spell(o) for o in origins], echo))


def test_a_june_fiscal_year_is_read_from_its_last_day():
    # 30 June 1982 ends the year that began 1 July 1981, which no integer year can name
    ends = [dt.date(o.year + 1, 6, 30) for o in RAA_ORIGINS]
    calendar = methods.mack(arrow_cells())
    fiscal = methods.mack(spelled(lambda o: dt.date(o.year + 1, 6, 30)))
    assert fiscal.origins["origin_period"].to_pylist() == [
        dt.date(o.year, 7, 1) for o in RAA_ORIGINS
    ]
    assert fiscal.origins["origin"].to_pylist() == ends
    assert fiscal.as_of == dt.date(1991, 6, 30)
    # the same cells, so the same numbers: only the calendar moved
    for column in ("latest", "ultimate", "ibnr", "mack_se"):
        assert fiscal.origins[column].equals(calendar.origins[column])
    assert fiscal.totals.equals(calendar.totals)


@pytest.mark.parametrize(
    ("cells", "premium"),
    [
        (spelled(lambda o: o.year), {year_end(o): p for o, p in RAA_PREMIUM.items()}),
        (spelled(year_end), {o.year: p for o, p in RAA_PREMIUM.items()}),
        (spelled(lambda o: o.year), {str(o.year): p for o, p in RAA_PREMIUM.items()}),
        (
            spelled(lambda o: o.year),
            pa.table(
                {
                    "origin_period": [f"{o.year}-12-31" for o in RAA_PREMIUM],
                    "premium": list(RAA_PREMIUM.values()),
                }
            ),
        ),
        (
            spelled(lambda o: o.isoformat()),
            pa.table({"origin_period": RAA_YEARS, "premium": list(RAA_PREMIUM.values())}),
        ),
    ],
    ids=[
        "years_by_year_end",
        "year_ends_by_year",
        "years_by_label",
        "years_by_table_of_ends",
        "strings_by_table_of_years",
    ],
)
@pytest.mark.parametrize("method", ["bornhuetter_ferguson", "cape_cod"])
def test_premium_may_be_written_differently_from_the_cells(method, cells, premium):
    expected = run(method)
    extra = {"expected_loss_ratio": 0.7} if method == "bornhuetter_ferguson" else {}
    result = getattr(methods, method)(cells, premium=premium, **extra)
    assert_same_numbers(result, expected)


@pytest.mark.parametrize(
    "origin",
    [1982, np.int64(1982), "1982", "1982-12-31", dt.date(1982, 12, 31), dt.datetime(1982, 1, 1, 6)],
    ids=["int", "numpy_int", "year_string", "year_end_string", "year_end_date", "datetime"],
)
def test_an_exclusion_origin_may_be_written_any_way(origin):
    expected = methods.chain_ladder(arrow_cells(), exclude=[(dt.date(1982, 1, 1), 12)])
    for cells in (arrow_cells(), spelled(lambda o: o.year)):
        result = methods.chain_ladder(cells, exclude=[(origin, 12)])
        assert_same_numbers(result, expected)


LABEL_FORMS = (
    r"write it as a year \(2020\), a quarter \(2020Q3\), a month \(2020-03\), or a date that "
    r"is the period's first day \(2020-01-01\) or last day \(2020-12-31\)"
)
NOT_FIRST_OR_LAST = "is neither the first nor the last day of a month"


@pytest.mark.parametrize(
    ("cells", "match"),
    [
        (
            spelled(lambda o: str(o.year), bad="1984Q5"),
            f"origin_period '1984Q5' is not an origin period: {LABEL_FORMS}",
        ),
        (
            spelled(lambda o: str(o.year), bad="FY84"),
            "origin_period 'FY84' is not an origin period",
        ),
        (
            spelled(lambda o: str(o.year), bad="1984-13"),
            "origin_period '1984-13' is not an origin period",
        ),
        (
            spelled(lambda o: str(o.year), bad="1984-02-30"),
            "origin_period '1984-02-30' is not an origin period",
        ),
        (
            spelled(lambda o: o.isoformat(), bad="1984-06-15"),
            f"origin_period '1984-06-15' {NOT_FIRST_OR_LAST}",
        ),
        (
            spelled(lambda o: o, pa.date32(), bad=dt.date(1984, 6, 15)),
            f"origin_period 1984-06-15 {NOT_FIRST_OR_LAST}. A date must be an origin period's "
            r"first day \(2020-01-01\) or its last day \(2020-12-31\), read with "
            "dev_grain_months=12",
        ),
        (
            spelled(
                lambda o: dt.datetime(o.year, 1, 1),
                pa.timestamp("s"),
                bad=dt.datetime(1984, 6, 15, 6),
            ),
            f"origin_period 1984-06-15 06:00:00 falls on 1984-06-15, which {NOT_FIRST_OR_LAST}",
        ),
        (spelled(lambda o: o.year, bad=0), "origin_period 0 is not a four-digit year"),
        (spelled(lambda o: o.year, bad=10000), "origin_period 10000 is not a four-digit year"),
        (
            spelled(lambda o: o.year % 100, pa.int8()),
            r"origin_period 81 is not a four-digit year: write a year with its century "
            r"\(1997, not 97\)",
        ),
        (
            spelled(lambda o: o.year, pa.uint64(), bad=2**64 - 1),
            f"origin_period {2**64 - 1} is not a four-digit year",
        ),
        (
            spelled(lambda o: o, pa.date32(), bad=dt.date(1984, 2, 28)),
            f"origin_period 1984-02-28 {NOT_FIRST_OR_LAST}",
        ),
        (
            spelled(lambda o: o, pa.date32(), bad=dt.date(1984, 4, 29)),
            f"origin_period 1984-04-29 {NOT_FIRST_OR_LAST}",
        ),
        (
            spelled(lambda o: o, pa.date32(), bad=dt.date(1984, 1, 30)),
            f"origin_period 1984-01-30 {NOT_FIRST_OR_LAST}",
        ),
        (
            spelled(lambda o: o, pa.date32(), bad=dt.date(9999, 12, 31)),
            "origin_period 9999-12-31 would end an origin period of 12 months that starts or "
            "ends outside the years 1 to 9999",
        ),
        (
            spelled(lambda o: o, pa.date32(), bad=dt.date(1, 1, 31)),
            "origin_period 0001-01-31 would end an origin period of 12 months",
        ),
        (
            spelled(lambda o: str(o.year), bad="1984-12-31"),
            "origin_period writes one origin period in more than one way: "
            "'1984-12-31' and '1984'. Write each period one way throughout the column, because "
            "the results echo its label back",
        ),
        (
            spelled(lambda o: o, pa.date32(), bad=dt.date(1984, 12, 31)),
            "in more than one way: 1984-12-31 and 1984-01-01",
        ),
        (
            spelled(
                lambda o: dt.datetime(o.year, 1, 1),
                pa.timestamp("s"),
                bad=dt.datetime(1984, 1, 1, 6),
            ),
            "in more than one way: 1984-01-01 06:00:00 and 1984-01-01 00:00:00",
        ),
        (
            spelled(lambda o: f"{o.year}Q1"),
            r"origin_period '1981Q1' is a quarter, 3 months long, but dev_grain_months=12\. "
            "The methods need origin periods one development step long: pass "
            "dev_grain_months=3 if the triangle develops 3 months at a time",
        ),
        (
            spelled(lambda o: f"{o.year}-01"),
            "origin_period '1981-01' is a month, 1 month long, but dev_grain_months=12",
        ),
    ],
    ids=[
        "quarter_5",
        "fiscal_label",
        "month_13",
        "february_30",
        "mid_month_string",
        "mid_month_date",
        "mid_month_timestamp",
        "year_0",
        "year_10000",
        "two_digit_years",
        "uint64_past_int64",
        "leap_february_28",
        "april_29",
        "january_30",
        "last_day_of_9999",
        "end_of_january_1",
        "year_written_two_ways",
        "date_written_two_ways",
        "timestamp_written_two_ways",
        "quarters_on_an_annual_step",
        "months_on_an_annual_step",
    ],
)
def test_an_origin_label_that_names_no_period_is_refused(cells, match):
    with pytest.raises(ValueError, match=match):
        methods.chain_ladder(cells)


@pytest.mark.parametrize(
    "origins",
    [[2020, 2021, 2022], [2020]],
    ids=["three_years", "one_year"],
)
@pytest.mark.parametrize(("spell", "shown"), [(int, "2020"), (str, "'2020'")], ids=["int", "text"])
def test_years_on_a_quarterly_step_are_refused_by_name(origins, spell, shown):
    # one origin has no neighbour to be 12 months away from, so only the label can tell
    rows = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5], [1]][: len(origins)]
    cells = arrow_cells(rows, years(*origins), step=3)
    labelled = cells.set_column(
        0, "origin_period", pa.array([spell(o.year) for o in cells["origin_period"].to_pylist()])
    )
    with pytest.raises(
        ValueError,
        match=rf"origin_period {shown} is a year, 12 months long, but dev_grain_months=3\. The "
        "methods need origin periods one development step long: pass dev_grain_months=12 if "
        r"the triangle develops 12 months at a time\. Origin periods longer than a development "
        r"step \(annual origins developed quarterly, for example\) are not supported yet",
    ):
        methods.chain_ladder(labelled, dev_grain_months=3)


YEAR_PREMIUM = {o.year: p for o, p in RAA_PREMIUM.items()}


@pytest.mark.parametrize(
    ("premium", "match"),
    [
        (
            {**YEAR_PREMIUM, "1981-12-31": 1.0},
            r"premium has more than one amount for origin period\(s\) 1981 and '1981-12-31'$",
        ),
        (
            pa.table(
                {
                    "origin_period": [*map(str, RAA_YEARS), "1981-01-01"],
                    "premium": [*RAA_PREMIUM.values(), 1.0],
                }
            ),
            r"premium has more than one row for origin period\(s\) '1981' and '1981-01-01'$",
        ),
        ({**YEAR_PREMIUM, "FY81": 1.0}, "premium origin 'FY81' is not an origin period"),
        (
            {**YEAR_PREMIUM, dt.date(1981, 6, 15): 1.0},
            f"premium origin 1981-06-15 {NOT_FIRST_OR_LAST}",
        ),
        ({**YEAR_PREMIUM, True: 1.0}, "premium origin True is not an origin period"),
        (
            {**YEAR_PREMIUM, 1.5: 1.0},
            "premium origin 1.5 is not an origin period",
        ),
        (
            {y: p for y, p in YEAR_PREMIUM.items() if y not in (1981, 1985)},
            r"premium has no amount for origin\(s\) 1981 and 1985$",
        ),
        (
            {**{str(y): p for y, p in YEAR_PREMIUM.items()}, "1991-12-31": 1.0},
            r"premium has amounts for origin\(s\) 1991-12-31 that are not in the triangle",
        ),
        (
            pa.table({"origin_period": [1981.0] * 10, "premium": list(RAA_PREMIUM.values())}),
            "premium origin_period must be a column of integer years, text labels, dates or "
            "timestamps, got double",
        ),
        (
            pa.table(
                {
                    "origin_period": [f"{y}Q1" for y in RAA_YEARS],
                    "premium": list(RAA_PREMIUM.values()),
                }
            ),
            r"premium origin_period '1981Q1' is a quarter, 3 months long, but the triangle's "
            r"origin periods are 12 months long \(dev_grain_months=12\)\. Write it as one of "
            "the triangle's origin periods",
        ),
    ],
    ids=[
        "dict_two_ways",
        "table_two_ways",
        "dict_unparseable",
        "dict_mid_month",
        "dict_bool",
        "dict_float",
        "missing_in_the_cells_labels",
        "extra_in_premiums_labels",
        "table_float",
        "table_quarters",
    ],
)
def test_premium_origins_are_refused_in_the_callers_labels(premium, match):
    with pytest.raises(ValueError, match=match):
        methods.cape_cod(spelled(lambda o: o.year), premium=premium)


@pytest.mark.parametrize(
    ("origin", "match"),
    [
        ("FY82", "exclude origin 'FY82' is not an origin period"),
        (dt.date(1982, 6, 15), f"exclude origin 1982-06-15 {NOT_FIRST_OR_LAST}"),
        ("1982Q1", "exclude origin '1982Q1' is a quarter"),
        (None, "exclude origin None is not an origin period"),
    ],
    ids=["unparseable", "mid_month", "quarter", "none"],
)
def test_an_exclusion_origin_that_names_no_period_is_refused(origin, match):
    with pytest.raises(ValueError, match=match):
        methods.chain_ladder(arrow_cells(), exclude=[(origin, 12)])


# -- every option is delivered ----------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"average": "simple"},
        {"average": "median"},
        {"history_periods": 3},
        {"drop_high": True},
        {"drop_low": True},
        {"exclude": [(dt.date(1982, 1, 1), 12)]},
    ],
    ids=["simple", "median", "history", "drop_high", "drop_low", "exclude"],
)
@pytest.mark.parametrize("method", ["chain_ladder", "bornhuetter_ferguson", "cape_cod"])
def test_each_development_option_changes_the_answer(method, options):
    function = getattr(methods, method)
    extra = {} if method == "chain_ladder" else {"premium": RAA_PREMIUM}
    if method == "bornhuetter_ferguson":
        extra["expected_loss_ratio"] = 0.7
    plain = function(arrow_cells(), **extra)
    varied = function(arrow_cells(), **extra, **options)
    assert factors(varied) != factors(plain)
    assert not np.array_equal(ultimates(varied), ultimates(plain))


def test_the_named_link_ratio_is_the_one_excluded():
    result = methods.chain_ladder(arrow_cells(), exclude=[(dt.date(1982, 1, 1), 12)])
    reasons = result.link_ratios
    excluded = reasons.filter(pc.equal(reasons["reason"], "explicit_exclusion"))
    assert excluded["origin_period"].to_pylist() == [dt.date(1982, 1, 1)]
    assert excluded["from_dev_lag"].to_pylist() == [12]


def test_expected_loss_ratio_is_delivered():
    low = methods.bornhuetter_ferguson(arrow_cells(), premium=RAA_PREMIUM, expected_loss_ratio=0.5)
    high = methods.bornhuetter_ferguson(arrow_cells(), premium=RAA_PREMIUM, expected_loss_ratio=0.8)
    assert (ultimates(high)[1:] > ultimates(low)[1:]).all()
    assert high.origins["expected_loss_ratio"].to_pylist() == [0.8] * 10


def test_decay_is_delivered():
    one = methods.cape_cod(arrow_cells(), premium=RAA_PREMIUM, decay=1.0)
    half = methods.cape_cod(arrow_cells(), premium=RAA_PREMIUM, decay=0.5)
    default = methods.cape_cod(arrow_cells(), premium=RAA_PREMIUM)
    assert not np.allclose(ultimates(one), ultimates(half))
    assert default.origins.equals(one.origins)
    # decay 1 is one loss ratio for the whole triangle
    assert len(set(one.origins["expected_loss_ratio"].to_pylist())) == 1


def test_sigma_rule_is_delivered():
    log_linear = methods.mack(arrow_cells(), sigma_rule="log_linear")
    by_mack = methods.mack(arrow_cells(), sigma_rule="mack")
    default = methods.mack(arrow_cells())
    assert default.origins.equals(log_linear.origins)
    np.testing.assert_array_equal(ultimates(log_linear), ultimates(by_mack))
    assert log_linear.totals["mack_se"][0].as_py() != by_mack.totals["mack_se"][0].as_py()


POINT_METHODS = ["chain_ladder", "bornhuetter_ferguson", "cape_cod"]


def point_method(method: str):
    """The method as a function of the cells and the development options alone."""
    function = getattr(methods, method)
    extra = {}
    if method != "chain_ladder":
        extra["premium"] = RAA_PREMIUM
    if method == "bornhuetter_ferguson":
        extra["expected_loss_ratio"] = 0.7
    return lambda cells, **options: function(cells, **extra, **options)


@pytest.mark.parametrize("method", POINT_METHODS)
def test_unsupported_factor_is_delivered(method):
    fit = point_method(method)
    # Excluding the only link ratio at 108 months leaves that age with nothing.
    lonely = [(dt.date(1981, 1, 1), 108)]
    with pytest.raises(Refusal, match="no link ratio is left from 108 to 120 months") as refused:
        fit(arrow_cells(), exclude=lonely)
    assert refused.value.reason == "no_link_ratio"
    assert refused.value.links == ((108, 120),)
    unity = fit(arrow_cells(), exclude=lonely, unsupported_factor="unity")
    assert factors(unity)[8] == 1.0
    assert unity.development["unity_fallback"].to_pylist()[8] is True


@pytest.mark.parametrize("method", POINT_METHODS)
def test_exhausted_exclusions_is_delivered(method):
    fit = point_method(method)
    kept = fit(arrow_cells(), drop_high=True)
    # the last link has a single ratio, so dropping the highest leaves none there
    assert kept.development["extreme_trimming_skipped"].to_pylist()[-2] is True
    assert factors(kept)[-2] == pytest.approx(18834 / 18662)
    with pytest.raises(
        Refusal, match="drop_high would leave no link ratio from 108 to 120 months"
    ) as refused:
        fit(arrow_cells(), drop_high=True, exhausted_exclusions="raise")
    assert refused.value.reason == "exclusions_exhausted"
    assert refused.value.links == ((108, 120),)


def test_an_exclusion_age_may_be_a_numpy_integer():
    # iterating a numpy or polars column gives numpy integers, not ints
    plain = methods.chain_ladder(arrow_cells(), exclude=[(dt.date(1982, 1, 1), 12)])
    for age in (np.int64(12), np.int32(12)):
        numpy_age = methods.chain_ladder(arrow_cells(), exclude=[(dt.date(1982, 1, 1), age)])
        assert numpy_age.origins.equals(plain.origins)
        assert numpy_age.link_ratios.equals(plain.link_ratios)


def quarterly_rows() -> tuple[list, list]:
    origins = [dt.date(2020, 1 + 3 * q, 1) for q in range(4)]
    rows = [[100.0, 180.0, 220.0, 240.0], [110.0, 200.0, 230.0], [90.0, 170.0], [120.0]]
    return rows, origins


@pytest.mark.parametrize("method", ["chain_ladder", "mack"])
def test_dev_grain_months_is_delivered(method):
    rows, origins = quarterly_rows()
    cells = arrow_cells(rows, origins, step=3)
    function = getattr(methods, method)
    result = function(cells, dev_grain_months=3)
    assert result.dev_grain_months == 3
    assert result.development["dev_lag"].to_pylist() == [3, 6, 9, 12]
    assert result.origins["latest_dev_lag"].to_pylist() == [12, 9, 6, 3]
    assert result.as_of == dt.date(2020, 12, 31)
    # the likely mistake, quarterly ages with the default grain, is named in the caller's terms
    with pytest.raises(
        ValueError,
        match=r"dev_lag values \[3, 6, 9\] are not multiples of dev_grain_months=12.*"
        "for a quarterly triangle pass dev_grain_months=3",
    ):
        function(cells)
    grid = kernel_grid(rows, origins, step=3)
    if method == "mack":
        np.testing.assert_array_equal(ultimates(result), fit_mack_grid(grid).ultimate)
    else:
        fit = fit_conventional_grid(grid, ConventionalCandidate())
        np.testing.assert_array_equal(ultimates(result), fit.origins["ultimate"])


# -- the result's tables -----------------------------------------------------------

BASE_ORIGINS = [
    ("origin", pa.date32()),  # the cells' own labels, which arrow_cells writes as dates
    ("origin_period", pa.date32()),
    ("latest_dev_lag", pa.int64()),
    ("latest", pa.float64()),
    ("ultimate", pa.float64()),
    ("ibnr", pa.float64()),
]
PATTERN = [
    ("dev_lag", pa.int64()),
    ("factor", pa.float64()),
    ("cdf", pa.float64()),
    ("pct_reported", pa.float64()),
]
SELECTION = [
    ("n_selected", pa.int64()),
    ("unity_fallback", pa.bool_()),
    ("extreme_trimming_skipped", pa.bool_()),
    ("bounds_skipped", pa.bool_()),
]
LINK_RATIOS = pa.schema(
    [
        ("origin", pa.date32()),
        ("origin_period", pa.date32()),
        ("from_dev_lag", pa.int64()),
        ("previous", pa.float64()),
        ("following", pa.float64()),
        ("ratio", pa.float64()),
        ("included", pa.bool_()),
        ("reason", pa.string()),
    ]
)
TOTALS = [("latest", pa.float64()), ("ultimate", pa.float64()), ("ibnr", pa.float64())]
MACK_SE = [("mack_se", pa.float64()), ("parameter_se", pa.float64()), ("process_se", pa.float64())]
ELR = [("expected_loss_ratio", pa.float64())]
TREND = [("trended_loss_ratio", pa.float64()), ("trend_factor", pa.float64())]

SCHEMAS = {
    "chain_ladder": (BASE_ORIGINS, PATTERN + SELECTION, LINK_RATIOS, TOTALS),
    "bornhuetter_ferguson": (BASE_ORIGINS + ELR, PATTERN + SELECTION, LINK_RATIOS, TOTALS),
    "benktander": (BASE_ORIGINS + ELR, PATTERN + SELECTION, LINK_RATIOS, TOTALS),
    "cape_cod": (BASE_ORIGINS + ELR + TREND, PATTERN + SELECTION, LINK_RATIOS, TOTALS),
    "mack": (
        BASE_ORIGINS + MACK_SE,
        PATTERN + [("sigma", pa.float64()), ("std_err", pa.float64())],
        None,
        TOTALS + MACK_SE,
    ),
}


def run(method: str, cells=None):
    cells = arrow_cells() if cells is None else cells
    if method == "bornhuetter_ferguson":
        return methods.bornhuetter_ferguson(cells, premium=RAA_PREMIUM, expected_loss_ratio=0.7)
    if method == "benktander":
        return methods.benktander(cells, premium=RAA_PREMIUM, expected_loss_ratio=0.7)
    if method == "cape_cod":
        return methods.cape_cod(cells, premium=RAA_PREMIUM)
    return getattr(methods, method)(cells)


def run_with_options(method: str):
    """``run``, with every option that adds or fills a column set away from its default."""
    if method == "mack":
        return run(method)
    options = {"drop_above": 4.0, "drop_high": 2, "preserve": 2, "exclude_valuations": [1989]}
    extra = {}
    if method != "chain_ladder":
        extra["premium"] = RAA_PREMIUM
    if method in ("bornhuetter_ferguson", "benktander"):
        extra["expected_loss_ratio"] = 0.7
    if method == "benktander":
        extra["n_iters"] = 3
    if method == "cape_cod":
        extra.update(trend=0.05, n_iters=2, decay=0.5)
    return getattr(methods, method)(arrow_cells(), **options, **extra)


@pytest.mark.parametrize("options", [False, True], ids=["defaults", "options"])
@pytest.mark.parametrize("method", list(SCHEMAS))
def test_the_result_schema_is_pinned(method, options):
    result = run_with_options(method) if options else run(method)
    origins, development, link_ratios, totals = SCHEMAS[method]
    assert result.method == method
    assert result.origins.schema.equals(pa.schema(origins))
    assert result.development.schema.equals(pa.schema(development))
    assert result.totals.schema.equals(pa.schema(totals))
    assert result.totals.num_rows == 1
    assert result.origins.num_rows == 10
    assert result.development.num_rows == 10
    if link_ratios is None:
        assert result.link_ratios is None
    else:
        assert result.link_ratios.schema.equals(link_ratios)
        assert result.link_ratios.num_rows == 45


@pytest.mark.parametrize("options", [False, True], ids=["defaults", "options"])
@pytest.mark.parametrize("method", list(SCHEMAS))
def test_missing_numbers_are_nulls_never_nan(method, options):
    result = run_with_options(method) if options else run(method)
    development = result.development
    # the last age has no next age: its factor is null
    assert development["factor"].to_pylist()[-1] is None
    assert development["factor"].null_count == 1
    assert development["cdf"].to_pylist()[-1] == 1.0
    assert development["pct_reported"].to_pylist()[-1] == 1.0
    tables = [result.origins, development, result.totals]
    if result.link_ratios is not None:
        tables.append(result.link_ratios)
    for table in tables:
        for name in table.column_names:
            column = table[name]
            if pa.types.is_floating(column.type):
                assert not pc.any(pc.is_nan(column)).as_py(), name


def test_an_undefined_link_ratio_is_null_and_says_why():
    # 1982 starts at zero, so its first ratio has no value; kept as data, the
    # zero is observed and only the ratio out of it is left out
    rows = [list(row) for row in RAA]
    rows[1][0] = 0
    result = methods.chain_ladder(arrow_cells(rows), zero_cells="observed")
    ratios = result.link_ratios
    undefined = ratios.filter(pc.equal(ratios["reason"], "undefined_ratio"))
    assert undefined["origin_period"].to_pylist() == [dt.date(1982, 1, 1)]
    assert undefined["ratio"].to_pylist() == [None]
    assert ratios["ratio"].null_count == 1


@pytest.mark.parametrize("method", list(SCHEMAS))
def test_to_polars_returns_each_table(pl, method):
    result = run(method)
    for name in methods.TABLES:
        table = getattr(result, name)
        if table is None:
            continue
        frame = result.to_polars(name)
        assert isinstance(frame, pl.DataFrame)
        assert frame.equals(pl.from_arrow(table))
    assert result.to_polars().equals(pl.from_arrow(result.origins))


def test_to_polars_refuses_an_unknown_table():
    with pytest.raises(ValueError, match="table must be one of .* got 'origin'"):
        run("chain_ladder").to_polars("origin")


def test_to_polars_refuses_link_ratios_on_a_mack_result():
    with pytest.raises(ValueError, match="a mack result has no link_ratios table"):
        run("mack").to_polars("link_ratios")


def test_to_polars_names_the_extra_when_polars_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "polars", None)  # import polars now raises ImportError
    with pytest.raises(ImportError, match=r'pip install "ibnr\[polars\]"'):
        run("chain_ladder").to_polars()


def test_the_result_holds_no_pandas_objects():
    for method in SCHEMAS:
        result = run(method)
        for value in vars(result).values():
            assert not isinstance(value, pd.DataFrame | pd.Series), method


@pytest.mark.parametrize("method", list(SCHEMAS))
def test_columns_in_several_pieces_or_part_way_into_memory_read_the_same(method):
    """The methods read input columns straight from Arrow's memory buffers, so as
    not to load pandas. A column held in several chunks, or one that starts part
    way into its memory (a slice of a longer table), must read as the same
    numbers. Mutation: read only the first chunk in ``ibnr._arrow.to_numpy``;
    this fails. (On pyarrow 24 joining the chunks copies a slice to the start of
    new memory, so the slice cannot reach ``to_numpy`` with an offset from here;
    ``test_to_numpy_reads_a_slice_from_where_it_starts`` checks that case.)"""
    plain = arrow_cells()
    before = arrow_cells(
        rows=[[7.0, 9.0], [5.0]], origins=[dt.date(1960, 1, 1), dt.date(1961, 1, 1)]
    )
    sliced = pa.concat_tables([before, plain]).combine_chunks().slice(before.num_rows)
    pieces = pa.concat_tables([plain.slice(0, 20), plain.slice(20)])
    assert sliced.column("value").chunk(0).offset == before.num_rows
    assert pieces.column("value").num_chunks == 2
    expected = run(method, plain)
    for cells in (sliced, pieces):
        got = run(method, cells)
        for table in ("origins", "development", "link_ratios", "totals"):
            assert getattr(got, table) == getattr(expected, table), table


def test_to_numpy_reads_a_slice_from_where_it_starts():
    """``ibnr._arrow.to_numpy`` on an array that starts part way into its memory.
    Mutation: ignore the offset; this fails, reading the values before the slice."""
    from ibnr import _arrow

    whole = pa.table({"x": [10, 11, 12, 13, 14]}).column("x").chunk(0)
    part = whole.slice(2)
    assert part.offset == 2
    out = _arrow.to_numpy(part)
    assert out.dtype == np.int64
    assert out.tolist() == [12, 13, 14]
    assert _arrow.to_numpy(part.cast(pa.float64())).tolist() == [12.0, 13.0, 14.0]


# -- the numpy grid builder ---------------------------------------------------------


GRID_KEYS = ("n_w", "n_d", "latest_dev", "origin_periods", "dev_grain_months", "w", "d")


@pytest.mark.parametrize(
    "spell",
    [
        lambda o: o,
        lambda o: np.datetime64(o, "D"),
        lambda o: np.datetime64(o, "ns"),
        dt.date.isoformat,
        lambda o: dt.datetime(o.year, o.month, o.day, 6),
    ],
    ids=["date", "datetime64_day", "datetime64_ns", "iso", "datetime"],
)
def test_the_numpy_builder_equals_the_frame_builder(spell):
    data = columns()
    expected = cohort_grid_frame(pd.DataFrame(data), dev_grain_months=12, measure="cumulative")
    grid = grid_from_columns(
        [spell(o) for o in data["origin_period"]],
        np.array(data["dev_lag"]),
        data["value"],
        dev_grain_months=12,
        measure="cumulative",
    )
    np.testing.assert_array_equal(grid["cum"], expected["cum"])
    np.testing.assert_array_equal(grid["obs_mask"], expected["obs_mask"])
    for key in GRID_KEYS:
        np.testing.assert_array_equal(grid[key], expected[key])
    assert all(type(o) is dt.date for o in grid["origin_periods"])


def test_the_numpy_builder_merges_two_spellings_of_one_origin():
    grid = grid_from_columns(
        [dt.date(2010, 1, 1), "2010-01-01", np.datetime64("2011-01-01")],
        [12, 24, 12],
        [100.0, 200.0, 150.0],
        dev_grain_months=12,
        measure="cumulative",
    )
    assert grid["origin_periods"] == [dt.date(2010, 1, 1), dt.date(2011, 1, 1)]
    np.testing.assert_array_equal(grid["cum"], [[100.0, 200.0], [150.0, np.nan]])


@pytest.mark.parametrize(
    "missing",
    [None, np.datetime64("NaT"), float("nan"), pd.NaT, pd.NA],
    ids=["none", "nat", "nan", "pandas_nat", "pandas_na"],
)
def test_the_numpy_builder_refuses_a_missing_origin(missing):
    with pytest.raises(ValueError, match="origin_period has missing values"):
        grid_from_columns(
            [dt.date(2010, 1, 1), missing, dt.date(2011, 1, 1)],
            [12, 24, 12],
            [1.0, 2.0, 3.0],
            dev_grain_months=12,
            measure="cumulative",
        )


def test_the_numpy_builder_refuses_a_missing_float_origin():
    # a float column (as pandas makes of numbers with a gap) is refused as missing,
    # not as "not a date", both through the builder and through the frame wrapper
    origins = np.array([2010.0, np.nan])
    with pytest.raises(ValueError, match="origin_period has missing values"):
        grid_from_columns(origins, [12, 12], [1.0, 2.0], dev_grain_months=12, measure="cumulative")
    frame = pd.DataFrame({"origin_period": origins, "dev_lag": [12, 12], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="origin_period has missing values"):
        cohort_grid_frame(frame, dev_grain_months=12, measure="cumulative")


@pytest.mark.parametrize(
    ("origins", "named"),
    [([1995, 1990, 1995], "1995"), ([1995.0, 1990.0], "1995.0"), (["x", "2010-01-01"], "'x'")],
    ids=["int", "float", "text"],
)
def test_a_non_date_origin_is_named_as_the_caller_wrote_it(origins, named):
    # the first value in the column, as a plain Python value, as pandas.factorize gave it
    frame = pd.DataFrame(
        {"origin_period": origins, "dev_lag": [12] * len(origins), "value": [1.0] * len(origins)}
    )
    with pytest.raises(ValueError, match=rf"origin_period values must be dates: .*{named}$"):
        cohort_grid_frame(frame, dev_grain_months=12, measure="cumulative")


def test_the_numpy_builder_refuses_a_missing_datetime64_origin():
    with pytest.raises(ValueError, match="origin_period has missing values"):
        grid_from_columns(
            np.array(["2010-01-01", "NaT"], "datetime64[D]"),
            [12, 12],
            [1.0, 2.0],
            dev_grain_months=12,
            measure="cumulative",
        )


def test_the_numpy_builder_refuses_two_rows_for_one_cell():
    # the same origin spelled two ways is still one cell
    with pytest.raises(ValueError, match=r"multiple rows per \(origin, dev\) cell"):
        grid_from_columns(
            [dt.date(2010, 1, 1), "2010-01-01"],
            [12, 12],
            [1.0, 2.0],
            dev_grain_months=12,
            measure="cumulative",
        )


@pytest.mark.parametrize(
    "value",
    [np.datetime64("NaT"), np.datetime64("NaT", "ns"), np.datetime64("10000-01-01")],
    ids=["nat", "nat_ns", "year_10000"],
)
def test_a_datetime64_that_is_not_a_date_is_refused(value):
    from ibnr.kernels.contract import as_date

    with pytest.raises(ValueError, match="expected an ISO date or date object"):
        as_date(value)


def test_the_numpy_builder_refuses_columns_of_different_lengths():
    with pytest.raises(ValueError, match="origin_period has 2 values and dev_lag has 3"):
        grid_from_columns(
            ["2010-01-01", "2010-01-01"],
            [12, 24, 36],
            [1.0, 2.0, 3.0],
            dev_grain_months=12,
            measure="cumulative",
        )
    with pytest.raises(ValueError, match="value has 2 values and dev_lag has 3"):
        grid_from_columns(
            ["2010-01-01"] * 3,
            [12, 24, 36],
            [1.0, 2.0],
            dev_grain_months=12,
            measure="cumulative",
        )
