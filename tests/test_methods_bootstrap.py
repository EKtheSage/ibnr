"""``methods.odp_bootstrap``: the result's tables, the labels, and every option's delivery.

The arithmetic is tested in ``tests/test_odp_runoff.py`` (the kernel, against
chainladder-python and R); here the front door: the five tables and their
types, nulls never NaN, the caller's origin labels, the quantile layout,
``central`` equal to the point method, the seed, and that every option reaches
the kernel, which a test of the answer alone could not see.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from ibnr import methods
from ibnr.errors import Refusal

DATA = Path(__file__).parent / "data"
PUBLIC = json.loads((DATA / "refusal_triangles.json").read_text("utf-8"))


def cells_of(rows, origin=None) -> pa.Table:
    years, lags, values = zip(*rows, strict=True)
    labels = pa.array(years, pa.int64()) if origin is None else origin(years)
    return pa.table(
        {
            "origin_period": labels,
            "dev_lag": pa.array(lags, pa.int64()),
            "value": pa.array(values, pa.float64()),
        }
    )


RAA = cells_of(PUBLIC["raa"])
GENINS = cells_of(PUBLIC["genins"])
#: raa with 1988's 12-month amount 0: under zero_cells="missing" the ratio out
#: of it keeps its place in a history window, under "observed" it gives it up.
RAA_ZERO = cells_of([(o, d, 0.0 if (o, d) == (1988, 12) else v) for o, d, v in PUBLIC["raa"]])
PREMIUM = {1981 + i: 20000.0 + 2000.0 * i for i in range(10)}

#: Every method with the arguments it needs.
METHODS = {
    "chain_ladder": {},
    "bornhuetter_ferguson": {"premium": PREMIUM, "expected_loss_ratio": 0.8},
    "benktander": {"premium": PREMIUM, "expected_loss_ratio": 0.8, "n_iters": 2},
    "cape_cod": {"premium": PREMIUM, "decay": 0.5, "trend": 0.02},
}


def boot(cells=RAA, **options):
    options.setdefault("negative_increments", "reflect")
    options.setdefault("n_draws", 300)
    options.setdefault("seed", 1)
    return methods.odp_bootstrap(cells, **options)


def floats(table: pa.Table, name: str) -> np.ndarray:
    return np.array(table[name].to_pylist(), dtype=float)


SCHEMA = {
    "origins": [
        ("origin", pa.int64()),
        ("origin_period", pa.date32()),
        ("latest_dev_lag", pa.int64()),
        ("latest", pa.float64()),
        ("central_ultimate", pa.float64()),
        ("central_ibnr", pa.float64()),
        ("mean_ibnr", pa.float64()),
        ("sd_ibnr", pa.float64()),
        ("mean_ultimate", pa.float64()),
    ],
    "totals": [
        ("latest", pa.float64()),
        ("central_ultimate", pa.float64()),
        ("central_ibnr", pa.float64()),
        ("mean_ibnr", pa.float64()),
        ("sd_ibnr", pa.float64()),
        ("mean_ultimate", pa.float64()),
        ("n_draws", pa.int64()),
        ("n_residuals", pa.int64()),
        ("degrees_of_freedom", pa.int64()),
        ("n_negative_fitted", pa.int64()),
        ("n_draws_unit_factor", pa.int64()),
        ("n_draws_negative_ibnr", pa.int64()),
        ("n_draws_tail_fallback", pa.int64()),
        ("phi", pa.float64()),
        ("residual_adjustment", pa.string()),
        ("residual_pool", pa.string()),
        ("negative_increments", pa.string()),
        ("process", pa.string()),
    ],
    "quantiles": [
        ("origin", pa.int64()),
        ("origin_period", pa.date32()),
        ("level", pa.float64()),
        ("ibnr", pa.float64()),
        ("tvar", pa.float64()),
    ],
    "draws": [
        ("draw", pa.int64()),
        ("origin", pa.int64()),
        ("origin_period", pa.date32()),
        ("ibnr", pa.float64()),
    ],
    "residuals": [
        ("origin", pa.int64()),
        ("origin_period", pa.date32()),
        ("dev_lag", pa.int64()),
        ("increment", pa.float64()),
        ("fitted", pa.float64()),
        ("residual", pa.float64()),
        ("leverage", pa.float64()),
        ("adjusted", pa.float64()),
        ("in_pool", pa.bool_()),
        ("reason", pa.string()),
        ("link_reason", pa.string()),
    ],
}


@pytest.mark.parametrize("method", METHODS)
def test_the_five_tables_have_fixed_columns_for_every_method(method):
    result = boot(method=method, **METHODS[method])
    assert tuple(SCHEMA) == result.TABLES
    for name, columns in SCHEMA.items():
        table = getattr(result, name)
        assert [(f.name, f.type) for f in table.schema] == columns, name
    assert result.method == method and result.central.method == method
    assert result.n_draws == 300 and result.seed == 1
    assert result.draws.num_rows == 300 * 10
    assert result.residuals.num_rows == 55
    assert result.as_of == dt.date(1990, 12, 31) and result.dev_grain_months == 12


@pytest.mark.parametrize("method", METHODS)
def test_central_is_the_point_method_for_the_same_options(method):
    """``central`` is exactly what ``methods.<method>`` returns, every table,
    tails and development options included."""
    options = {"history_periods": 7, "drop_high": 1, "tail": "constant", "tail_factor": 1.03}
    result = boot(method=method, **METHODS[method], **options)
    point = getattr(methods, method)(RAA, **METHODS[method], **options)
    for name in methods.TABLES:
        assert getattr(result.central, name).equals(getattr(point, name)), name
    origins = result.origins
    assert origins["central_ultimate"].equals(point.origins["ultimate"])
    assert origins["central_ibnr"].equals(point.origins["ibnr"])
    assert result.totals["central_ibnr"].equals(point.totals["ibnr"])


def test_the_summaries_are_the_draws_summarised():
    result = boot(n_draws=400)
    draws = floats(result.draws, "ibnr").reshape(400, 10)
    assert result.draws["draw"].to_pylist()[:12] == [0] * 10 + [1, 1]
    np.testing.assert_array_equal(floats(result.origins, "mean_ibnr"), draws.mean(axis=0))
    np.testing.assert_array_equal(floats(result.origins, "sd_ibnr"), draws.std(axis=0, ddof=1))
    latest = floats(result.origins, "latest")
    np.testing.assert_array_equal(
        floats(result.origins, "mean_ultimate"), latest + draws.mean(axis=0)
    )
    total = draws.sum(axis=1)
    totals = result.totals.to_pylist()[0]
    assert totals["mean_ibnr"] == total.mean()
    assert totals["sd_ibnr"] == total.std(ddof=1)
    assert totals["mean_ultimate"] == latest.sum() + total.mean()
    assert totals["n_draws_negative_ibnr"] == int((total < 0).sum())
    assert totals["n_residuals"] == int(sum(result.residuals["in_pool"].to_pylist()))


def test_the_quantile_table_puts_the_total_first_with_a_null_origin():
    """Linear quantiles (numpy's default, np.percentile, R's type 7) and the mean
    of the draws at or above each."""
    levels = (0.5, 0.9, 0.995)
    result = boot(n_draws=500, quantiles=levels)
    table = result.quantiles
    assert table.num_rows == 3 * 11
    assert table["origin"].to_pylist()[:4] == [None, None, None, 1981]
    assert table["origin_period"].null_count == 3
    assert table["level"].to_pylist()[:6] == [0.5, 0.9, 0.995, 0.5, 0.9, 0.995]
    draws = floats(result.draws, "ibnr").reshape(500, 10)
    samples = [draws.sum(axis=1)] + [draws[:, i] for i in range(10)]
    expected, tvar = [], []
    for sample in samples:
        for level in levels:
            q = np.quantile(sample, level)
            expected.append(q)
            tvar.append(sample[sample >= q].mean())
    assert floats(table, "ibnr").tolist() == expected
    assert floats(table, "tvar").tolist() == tvar
    assert np.percentile(samples[0], 99.5) == expected[2]


def test_one_draw_leaves_the_standard_deviations_null_not_nan():
    result = boot(n_draws=1)
    assert result.origins["sd_ibnr"].null_count == 10
    assert result.totals["sd_ibnr"].null_count == 1
    for name in result.TABLES:
        table = getattr(result, name)
        for column in table.columns:
            if pa.types.is_floating(column.type):
                values = np.array(column.drop_null().to_pylist(), dtype=float)
                assert np.isfinite(values).all(), (name, column)


@pytest.mark.parametrize(
    ("label", "arrow_type"),
    [
        (lambda years: pa.array([str(y) for y in years]), pa.string()),
        (lambda years: pa.array([dt.date(y, 12, 31) for y in years], pa.date32()), pa.date32()),
        (
            lambda years: pa.array([dt.datetime(y, 1, 1) for y in years], pa.timestamp("us")),
            pa.timestamp("us"),
        ),
        (lambda years: pa.array(years, pa.int32()), pa.int64()),
    ],
    ids=["text", "year_end_date", "timestamp", "int32"],
)
def test_the_origin_is_echoed_as_the_caller_wrote_it(label, arrow_type):
    cells = cells_of(PUBLIC["raa"], origin=label)
    result = boot(cells)
    written = label(list(range(1981, 1991)))
    for name in ("origins", "quantiles", "draws", "residuals"):
        column = getattr(result, name)["origin"]
        assert column.type == arrow_type, name
    assert result.origins["origin"].to_pylist() == pa.array(written).cast(arrow_type).to_pylist()
    same = boot(RAA)
    assert result.draws["ibnr"].equals(same.draws["ibnr"])


def test_the_residual_table_says_which_cells_were_resampled_and_why():
    result = boot(exclude=[(1981, 12)], residual_adjustment="hat", residual_pool="centred")
    rows = result.residuals.to_pylist()
    by_cell = {(r["origin"], r["dev_lag"]): r for r in rows}
    for cell in ((1981, 12), (1981, 24)):
        assert by_cell[cell]["reason"] == "excluded_link"
        assert by_cell[cell]["link_reason"] == "explicit_exclusion"
        assert not by_cell[cell]["in_pool"] and by_cell[cell]["adjusted"] is None
    assert by_cell[(1990, 12)]["reason"] == "leverage_one"
    assert by_cell[(1981, 120)]["reason"] == "leverage_one"
    pooled = [r for r in rows if r["in_pool"]]
    assert all(r["adjusted"] is not None and r["link_reason"] is None for r in pooled)
    assert abs(sum(r["adjusted"] for r in pooled)) < 1e-9  # the centred pool
    assert len(pooled) == result.totals["n_residuals"][0].as_py() == 51


def test_a_seed_of_none_is_recorded_and_replays():
    first = boot(seed=None, n_draws=50)
    assert isinstance(first.seed, int) and first.seed > 2**32
    again = boot(seed=first.seed, n_draws=50)
    assert again.draws.equals(first.draws)
    zero = boot(seed=0, n_draws=50)
    assert boot(seed=0, n_draws=50).draws.equals(zero.draws)


def test_to_polars_gives_each_table():
    pl = pytest.importorskip("polars")
    result = boot(n_draws=20)
    for name in result.TABLES:
        frame = result.to_polars(name)
        assert isinstance(frame, pl.DataFrame) and frame.height == getattr(result, name).num_rows
    with pytest.raises(Refusal, match="table must be one of"):
        result.to_polars("development")


# -- every option reaches the kernel -----------------------------------------------

#: A 6 x 6 triangle with two equal link ratios from 12 to 24 months (1.5 for
#: 2002 and 2004, whose 12-month amounts differ) and a zero cumulative, for
#: trim_ties and zero_cells.
TIES = cells_of(
    [
        (2001, 12, 100.0),
        (2001, 24, 170.0),
        (2001, 36, 190.0),
        (2001, 48, 200.0),
        (2001, 60, 204.0),
        (2001, 72, 205.0),
        (2002, 12, 120.0),
        (2002, 24, 180.0),
        (2002, 36, 205.0),
        (2002, 48, 214.0),
        (2002, 60, 219.0),
        (2003, 12, 90.0),
        (2003, 24, 150.0),
        (2003, 36, 171.0),
        (2003, 48, 180.0),
        (2004, 12, 140.0),
        (2004, 24, 210.0),
        (2004, 36, 236.0),
        (2005, 12, 0.0),
        (2005, 24, 75.0),
        (2006, 12, 150.0),
    ]
)
TIES_PREMIUM = {2001 + i: 300.0 + 10 * i for i in range(6)}


def _answer(result) -> tuple:
    return (
        result.draws["ibnr"].to_pylist(),
        result.central.origins["ultimate"].to_pylist(),
        result.quantiles["level"].to_pylist(),
    )


#: (option, value, other value, extra arguments, cells)
DELIVERY = [
    ("residual_adjustment", "hat", "dof", {}, RAA),
    ("residual_adjustment", "hat", "none", {}, RAA),
    ("residual_pool", "centred", "all", {}, RAA),
    ("process", "gamma", "od_poisson", {}, RAA),
    ("process", "gamma", "none", {}, RAA),
    ("prior_cv", 0.0, 0.2, {"method": "cape_cod", "premium": PREMIUM}, RAA),
    (
        "prior_cv",
        0.0,
        0.2,
        {"method": "bornhuetter_ferguson", **METHODS["bornhuetter_ferguson"]},
        RAA,
    ),
    ("seed", 1, 2, {}, RAA),
    ("n_draws", 300, 301, {}, RAA),
    ("quantiles", (0.5,), (0.9,), {}, RAA),
    (
        "n_iters",
        1,
        3,
        {"premium": PREMIUM, "expected_loss_ratio": 0.8, "method": "benktander"},
        RAA,
    ),
    ("expected_loss_ratio", 0.8, 0.6, {"premium": PREMIUM, "method": "bornhuetter_ferguson"}, RAA),
    ("decay", 1.0, 0.3, {"premium": PREMIUM, "method": "cape_cod"}, RAA),
    ("trend", 0.0, 0.05, {"premium": PREMIUM, "method": "cape_cod"}, RAA),
    ("n_iters", 1, 2, {"premium": PREMIUM, "method": "cape_cod"}, RAA),
    ("average", "volume", "simple", {}, RAA),
    ("history_periods", None, 5, {}, RAA),
    ("drop_high", False, 1, {}, RAA),
    ("drop_low", False, 1, {}, RAA),
    ("preserve", 1, 3, {"drop_high": 1}, RAA),
    ("drop_above", None, 3.0, {}, RAA),
    ("drop_below", None, 1.1, {}, RAA),
    ("exclude", (), [(1985, 24)], {}, RAA),
    ("exclude_valuations", (), [1988], {}, RAA),
    ("trim_ties", "volume", "origin", {"drop_high": 1}, TIES),
    ("zero_cells", "missing", "observed", {"history_periods": 2}, RAA_ZERO),
    ("tail", None, "exponential", {}, RAA),
    ("tail_factor", 1.05, 1.10, {"tail": "constant"}, RAA),
    ("tail_decay", 0.5, 0.9, {"tail": "constant", "tail_factor": 1.05, "tail_attach_lag": 72}, RAA),
    ("tail_attach_lag", None, 84, {"tail": "exponential"}, RAA),
    ("tail_fit_lags", None, (24, None), {"tail": "exponential"}, RAA),
    ("tail_steps", None, 5, {"tail": "exponential"}, RAA),
]


@pytest.mark.parametrize(
    ("option", "one", "other", "extra", "cells"),
    DELIVERY,
    ids=[f"{row[0]}={row[2]!r}" for row in DELIVERY],
)
def test_every_option_changes_the_answer(option, one, other, extra, cells):
    """Changing the option changes the draws, the central fit or the quantile
    levels, so a build that drops it fails here."""
    extra = dict(extra)
    if cells is TIES and "premium" in extra:
        extra["premium"] = TIES_PREMIUM
    first = boot(cells, **extra, **{option: one})
    second = boot(cells, **extra, **{option: other})
    assert _answer(first) != _answer(second)


def test_negative_increments_is_delivered():
    with pytest.raises(Refusal) as caught:
        methods.odp_bootstrap(RAA, n_draws=10)
    assert caught.value.reason == "negative_increment"
    assert caught.value.option == "negative_increments"
    assert [(c.origin, c.dev_lag) for c in caught.value.cells] == [(1982, 84)]
    assert "negative_increments='reflect'" in str(caught.value)
    assert boot(RAA).totals["negative_increments"][0].as_py() == "reflect"


def test_unsupported_factor_and_exhausted_exclusions_are_delivered():
    with pytest.raises(Refusal) as caught:
        boot(exclude=[(1981, 108)], unsupported_factor="raise")
    assert caught.value.reason == "no_link_ratio"
    unity = boot(exclude=[(1981, 108)], unsupported_factor="unity")
    assert unity.central.development["unity_fallback"].to_pylist()[8] is True
    with pytest.raises(Refusal) as caught:
        boot(drop_high=1, exhausted_exclusions="raise")
    assert caught.value.reason == "exclusions_exhausted"
    assert (
        boot(drop_high=1, exhausted_exclusions="keep")
        .central.development["extreme_trimming_skipped"]
        .to_pylist()[8]
    )


def test_tail_rows_reaches_the_central_fit():
    one = boot(tail="constant", tail_factor=1.05, tail_rows=1)
    three = boot(tail="constant", tail_factor=1.05, tail_rows=3)
    assert one.central.development.num_rows == 11 and three.central.development.num_rows == 13
    assert one.draws.equals(three.draws)  # rows shown never move an ultimate


def test_a_tail_is_one_more_future_cell_for_every_origin():
    result = boot(tail="constant", tail_factor=1.05, n_draws=2000)
    oldest = floats(result.draws, "ibnr").reshape(2000, 10)[:, 0]
    central = result.origins["central_ibnr"][0].as_py()
    assert central > 0 and oldest.std() > 0
    assert oldest.mean() == pytest.approx(central, rel=0.1)
    assert result.totals["n_draws_tail_fallback"][0].as_py() == 0


def test_process_none_draws_the_refitted_means():
    noiseless = boot(process="none", n_draws=100)
    noisy = boot(process="gamma", n_draws=100)
    a = floats(noiseless.draws, "ibnr").reshape(100, 10)
    b = floats(noisy.draws, "ibnr").reshape(100, 10)
    assert a.std(axis=0)[-1] < b.std(axis=0)[-1]
    assert noiseless.totals["process"][0].as_py() == "none"


def test_the_bootstrap_mean_sits_near_the_central_estimate():
    """3% above on raa and 1% on genins at 20,000 draws, from the refit's
    non-linearity, as in R and chainladder-python."""
    for cells, above in ((RAA, 0.03), (GENINS, 0.01)):
        totals = boot(cells, n_draws=20_000, seed=5).totals.to_pylist()[0]
        ratio = totals["mean_ibnr"] / totals["central_ibnr"] - 1
        assert abs(ratio - above) < 0.02, ratio
