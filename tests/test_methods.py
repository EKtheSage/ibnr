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
        "ReserveResult",
        "bornhuetter_ferguson",
        "cape_cod",
        "chain_ladder",
        "mack",
    ]


# -- what the functions accept ------------------------------------------------------


def origin_variants():
    """The same origin column in each type the functions read."""
    dates = columns()["origin_period"]
    yield "date32", pa.array(dates, pa.date32())
    yield "date64", pa.array(dates, pa.date64())
    yield "timestamp", pa.array([dt.datetime(o.year, 1, 1) for o in dates], pa.timestamp("us"))
    # Midnight on 1 January in Tokyo is 15:00 on 31 December in UTC. The date
    # read is Tokyo's, the zone the column is in; reading the UTC date would
    # put every origin on 31 December.
    yield (
        "timestamp_tz",
        pa.array(
            [dt.datetime(o.year - 1, 12, 31, 15, tzinfo=dt.UTC) for o in dates],
            pa.timestamp("us", tz="Asia/Tokyo"),
        ),
    )
    # a time of day is dropped: the date part is the origin period
    yield (
        "timestamp_time_of_day",
        pa.array([dt.datetime(o.year, 1, 1, 6) for o in dates], pa.timestamp("s")),
    )
    yield "string", pa.array([o.isoformat() for o in dates], pa.string())
    yield "large_string", pa.array([o.isoformat() for o in dates], pa.large_string())
    yield (
        "dictionary_string",
        pa.array([o.isoformat() for o in dates], pa.string()).dictionary_encode(),
    )
    yield "dictionary_date", pa.array(dates, pa.date32()).dictionary_encode()


@pytest.mark.parametrize(
    ("kind", "origin"), list(origin_variants()), ids=[kind for kind, _ in origin_variants()]
)
def test_every_accepted_origin_type_gives_the_same_answer(kind, origin):
    expected = methods.chain_ladder(arrow_cells())
    cells = arrow_cells().set_column(0, "origin_period", origin)
    result = methods.chain_ladder(cells)
    assert result.origins.equals(expected.origins), kind
    assert result.as_of == expected.as_of


@pytest.mark.parametrize("kind", ["date", "datetime", "string", "categorical", "enum"])
def test_polars_origin_types_give_the_same_answer(pl, kind):
    expected = methods.chain_ladder(arrow_cells())
    data = columns()
    origins = data["origin_period"]
    if kind == "datetime":
        origins = [dt.datetime(o.year, o.month, o.day) for o in origins]
    elif kind != "date":
        origins = [o.isoformat() for o in origins]
    frame = pl.DataFrame({**data, "origin_period": origins})
    if kind == "categorical":
        frame = frame.with_columns(pl.col("origin_period").cast(pl.Categorical))
    elif kind == "enum":
        labels = sorted(set(origins))
        frame = frame.with_columns(pl.col("origin_period").cast(pl.Enum(labels)))
    assert methods.chain_ladder(frame).origins.equals(expected.origins)


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
            with_column("origin_period", ["1981-13-01"] * N),
            "origin_period strings must be ISO dates",
        ),
        (
            with_column("origin_period", [1981] * N),
            "origin_period must be a date, timestamp or ISO date string column, got int64. "
            r"An accident year 2020 is the date 2020-01-01; in polars, pl.date\(",
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
        "bad_iso",
        "integer_origin",
    ],
)
def test_a_bad_column_is_refused_by_name(cells, match):
    with pytest.raises(ValueError, match=match):
        methods.chain_ladder(cells)


def test_two_cohorts_rows_are_refused_one_cohort_at_a_time():
    both = pa.concat_tables([arrow_cells(), arrow_cells()])
    with pytest.raises(ValueError, match="one cohort at a time.*refused rather than added"):
        methods.chain_ladder(both)


def years(*first_years: int) -> list:
    return [dt.date(y, 1, 1) for y in first_years]


@pytest.mark.parametrize("method", ["chain_ladder", "mack"])
def test_a_negative_cumulative_is_refused_naming_the_cells(method):
    rows = [list(row) for row in RAA]
    rows[4][1], rows[4][2] = -50, -80
    with pytest.raises(
        ValueError,
        match=r"value is negative in 2 cell\(s\), \(origin_period, dev_lag\) "
        r"\[\(datetime.date\(1985, 1, 1\), 24\), \(datetime.date\(1985, 1, 1\), 36\)\]",
    ):
        getattr(methods, method)(arrow_cells(rows))


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
@pytest.mark.parametrize("method", ["chain_ladder", "mack"])
def test_a_missing_origin_period_is_refused_by_name(method, rows, origins):
    with pytest.raises(
        ValueError,
        match=r"2020-01-01 and 2022-01-01 are 24 months apart, but dev_grain_months=12.*"
        r"cells has no rows for \[datetime.date\(2021, 1, 1\)\]",
    ):
        getattr(methods, method)(arrow_cells(rows, origins))


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
            r"premium has more than one row for origin period\(s\) \[datetime.date\(1981, 1, 1\)\]",
        ),
        (
            premium_table(RAA_PREMIUM).slice(1),
            r"premium has no amount for origin\(s\) \[datetime.date\(1981, 1, 1\)\]",
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


def test_an_exclusion_the_triangle_does_not_have_is_refused():
    with pytest.raises(ValueError, match="exclude names link ratio.*1990, 1, 1\\), 12"):
        methods.chain_ladder(arrow_cells(), exclude=[(dt.date(1990, 1, 1), 12)])


@pytest.mark.parametrize(
    "exclude", ["1981-01-01", [dt.date(1981, 1, 1)], (dt.date(1981, 1, 1), 12)]
)
def test_an_exclusion_that_is_not_a_list_of_pairs_is_refused(exclude):
    with pytest.raises(ValueError, match="exclu"):
        methods.chain_ladder(arrow_cells(), exclude=exclude)


def test_mack_refuses_an_unknown_sigma_rule():
    with pytest.raises(ValueError, match="sigma_rule must be one of"):
        methods.mack(arrow_cells(), sigma_rule="log-linear")


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
    with pytest.raises(ValueError, match="no estimable positive factor at dev lag 108"):
        fit(arrow_cells(), exclude=lonely)
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
        ValueError, match="extreme exclusions exhaust paired origins at dev lag 108"
    ):
        fit(arrow_cells(), drop_high=True, exhausted_exclusions="raise")


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
]
LINK_RATIOS = pa.schema(
    [
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

SCHEMAS = {
    "chain_ladder": (BASE_ORIGINS, PATTERN + SELECTION, LINK_RATIOS, TOTALS),
    "bornhuetter_ferguson": (BASE_ORIGINS + ELR, PATTERN + SELECTION, LINK_RATIOS, TOTALS),
    "cape_cod": (BASE_ORIGINS + ELR, PATTERN + SELECTION, LINK_RATIOS, TOTALS),
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
    if method == "cape_cod":
        return methods.cape_cod(cells, premium=RAA_PREMIUM)
    return getattr(methods, method)(cells)


@pytest.mark.parametrize("method", list(SCHEMAS))
def test_the_result_schema_is_pinned(method):
    result = run(method)
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


@pytest.mark.parametrize("method", list(SCHEMAS))
def test_missing_numbers_are_nulls_never_nan(method):
    result = run(method)
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
    # 1982 starts at zero, so its first ratio has no value
    rows = [list(row) for row in RAA]
    rows[1][0] = 0
    result = methods.chain_ladder(arrow_cells(rows))
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
