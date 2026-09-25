"""The array path to the conventional point estimators, ``fit_conventional_grid``.

Three things are checked. On the same cells the array path gives exactly what
the Triangle path (``fit_conventional``) gives, so a service can use one and a
study the other without the numbers moving. Starting from plain arrays, with no
Triangle at all, it reproduces chainladder-python's chain ladder,
Bornhuetter-Ferguson and Cape Cod. And each refusal fires with a message that
names the problem, because a grid is a plain dict a caller can build by hand.
"""

from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle, kernels
from ibnr.errors import Refusal
from ibnr.kernels.contract import cohort_grid, cohort_grid_frame
from ibnr.kernels.conventional import (
    ConventionalCandidate,
    fit_conventional,
    fit_conventional_grid,
)
from ibnr.kernels.mack import fit_mack_grid
from ibnr.triangle.core import GRAIN_MONTHS

from .conftest import make_cohort_triangle

# Six annual origins, 2010 to 2015, observed to 2015-12-31. The first link has
# two tied ratios of 3.0 (2012 and 2014), so the trimming tie rule is exercised.
SIX = np.array(
    [
        [100.0, 250.0, 330.0, 380.0, 400.0, 410.0],
        [120.0, 260.0, 360.0, 395.0, 420.0, np.nan],
        [90.0, 270.0, 340.0, 400.0, np.nan, np.nan],
        [150.0, 300.0, 420.0, np.nan, np.nan, np.nan],
        [110.0, 330.0, np.nan, np.nan, np.nan, np.nan],
        [140.0, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
)
SIX_ORIGINS = [dt.date(2010 + i, 1, 1) for i in range(6)]
SIX_AS_OF = dt.date(2015, 12, 31)
SIX_PREMIUM = np.array([500.0, 520.0, 480.0, 600.0, 650.0, 700.0])


def with_premium(tri: Triangle, premium: dict[dt.date, float], backend_name) -> Triangle:
    """The same triangle with an ``earned_premium`` row at each origin's first age."""
    frame = tri.execute()
    for column in ("origin_period", "eval_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.date
    step = GRAIN_MONTHS[tri.meta.dev_grain]
    first = frame.loc[frame["dev_lag"] == step, ["origin_period", "eval_date"]]
    rows = first.assign(dev_lag=step, field="earned_premium")
    rows["value"] = [premium[o] for o in rows["origin_period"]]
    return Triangle.from_long(
        pd.concat([frame, rows], ignore_index=True),
        measure="cumulative",
        origin_grain=tri.meta.origin_grain,
        dev_grain=tri.meta.dev_grain,
        backend=backend_name,
    )


def grid_from(tri: Triangle, *, field: str = "paid_loss", measure: str = "cumulative") -> dict:
    """The grid a service would build: the same rows, as a plain pandas frame."""
    frame = tri.execute()
    frame = frame.loc[frame["field"] == field, ["origin_period", "dev_lag", "value"]]
    return cohort_grid_frame(
        frame, dev_grain_months=GRAIN_MONTHS[tri.meta.dev_grain], measure=measure
    )


def rows_frame(cells: dict[dt.date, list[float]], *, step: int = 12) -> pd.DataFrame:
    """A plain (origin_period, dev_lag, value) frame from each origin's values."""
    return pd.DataFrame(
        [
            {"origin_period": origin, "dev_lag": step * (j + 1), "value": value}
            for origin, values in cells.items()
            for j, value in enumerate(values)
        ]
    )


def long_triangle(cells: dict[dt.date, list[float]], backend_name, *, step: int = 12) -> Triangle:
    """A Triangle holding exactly the rows of ``rows_frame(cells)``."""
    frame = rows_frame(cells, step=step)
    frame["eval_date"] = [
        (pd.Timestamp(o) + pd.DateOffset(months=int(lag)) - pd.Timedelta(days=1)).date()
        for o, lag in zip(frame["origin_period"], frame["dev_lag"], strict=True)
    ]
    frame["field"] = "paid_loss"
    return Triangle.from_long(frame, measure="cumulative", backend=backend_name)


def assert_same_fit(grid_fit, triangle_fit) -> None:
    """Every published output of the two paths is identical, not merely close."""
    assert grid_fit.candidate == triangle_fit.candidate
    assert grid_fit.as_of == triangle_fit.as_of
    np.testing.assert_array_equal(grid_fit.factors, triangle_fit.factors)
    np.testing.assert_array_equal(grid_fit.beta, triangle_fit.beta)
    pd.testing.assert_frame_equal(grid_fit.origins, triangle_fit.origins, check_exact=True)
    pd.testing.assert_frame_equal(
        grid_fit.factor_selection, triangle_fit.factor_selection, check_exact=True
    )
    pd.testing.assert_frame_equal(
        grid_fit.factor_summary, triangle_fit.factor_summary, check_exact=True
    )


@pytest.fixture
def six(backend_name) -> Triangle:
    tri = make_cohort_triangle(backend_name, SIX)
    return with_premium(tri, dict(zip(SIX_ORIGINS, SIX_PREMIUM, strict=True)), backend_name)


def six_premium() -> dict[dt.date, float]:
    return dict(zip(SIX_ORIGINS, SIX_PREMIUM, strict=True))


CC = ConventionalCandidate
CANDIDATES = {
    "cl": CC(),
    "cl_window_trimmed": CC(
        history_periods=3, drop_high=True, drop_low=True, exhausted_exclusions="keep"
    ),
    "cl_simple": CC(average="simple"),
    "cl_median": CC(average="median"),
    "cl_excluded": CC(exclude=((dt.date(2012, 1, 1), 12),)),
    "cl_beyond_data": CC(horizon=84, unsupported_factor="unity"),
    "bf": CC("bf", expected_loss_ratio=0.7),
    "gcc_0": CC("gcc", decay=0.0),
    "gcc_half": CC("gcc", decay=0.5),
    "gcc_1": CC("gcc", decay=1.0),
}


# -- the array path equals the Triangle path ----------------------------------


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_the_array_path_equals_the_triangle_path(six, name):
    spec = CANDIDATES[name]
    premium = None if spec.method == "cl" else six_premium()
    by_grid = fit_conventional_grid(grid_from(six), spec, premium=premium)
    by_triangle = fit_conventional(six, spec, as_of=SIX_AS_OF)
    assert_same_fit(by_grid, by_triangle)


def test_the_trims_and_the_window_really_changed_the_answer(six):
    # The equality above is only worth something if the settings it covers
    # move the answer; otherwise two paths that both ignore them would agree.
    plain = fit_conventional_grid(grid_from(six), CANDIDATES["cl"])
    for name in ("cl_window_trimmed", "cl_simple", "cl_median", "cl_excluded"):
        varied = fit_conventional_grid(grid_from(six), CANDIDATES[name])
        assert not np.array_equal(varied.factors, plain.factors), name


def test_premium_already_on_the_grid_is_used(six):
    spec = CANDIDATES["gcc_half"]
    carried = cohort_grid(
        six.as_of(SIX_AS_OF), loss_field="paid_loss", premium_field="earned_premium"
    )
    by_carried = fit_conventional_grid(carried, spec)
    by_keyword = fit_conventional_grid(grid_from(six), spec, premium=six_premium())
    assert_same_fit(by_carried, by_keyword)


# -- chainladder-python, from plain arrays ------------------------------------


@pytest.fixture(scope="module")
def chainladder():
    return pytest.importorskip("chainladder")


def plain_arrays(chainladder, name: str):
    """Origin dates, development ages and cumulative values, as numpy arrays.

    Taken out of the sample on purpose, so nothing of ibnr's Triangle layer is on
    the path: this is what a service holding its own data would start from.
    """
    sample = chainladder.load_sample(name)
    long = sample.to_frame(keepdims=True).reset_index()
    origins = pd.to_datetime(long["origin"]).dt.date.to_numpy()
    lags = long["development"].to_numpy(dtype=int)
    values = long["values"].to_numpy(dtype=float)
    return sample, origins, lags, values


def plain_grid(origins, lags, values) -> dict:
    frame = pd.DataFrame({"origin_period": origins, "dev_lag": lags, "value": values})
    return cohort_grid_frame(frame, dev_grain_months=12, measure="cumulative")


def sloped_premium(chainladder, sample, grid):
    """A premium rising 1x to 2x across the origins, keyed by origin, and the same
    numbers as the one-column chainladder triangle its methods take as weight."""
    amounts = np.linspace(1.0, 2.0, grid["n_w"]) * np.nanmax(grid["cum"])
    weight = sample.latest_diagonal * 0
    weight.values = weight.values + amounts[None, None, :, None]
    return dict(zip(grid["origin_periods"], amounts, strict=True)), weight


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_chain_ladder_from_plain_arrays_matches_chainladder(chainladder, name):
    sample, origins, lags, values = plain_arrays(chainladder, name)
    fitted = fit_conventional_grid(plain_grid(origins, lags, values), CC())
    reference = chainladder.Chainladder().fit(sample)
    np.testing.assert_allclose(
        fitted.origins["ultimate"], reference.ultimate_.values.ravel(), rtol=1e-9
    )
    np.testing.assert_allclose(
        fitted.factors, reference.ldf_.values.ravel()[: len(fitted.factors)], rtol=1e-9
    )


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_bornhuetter_ferguson_from_plain_arrays_matches_chainladder(chainladder, name):
    sample, origins, lags, values = plain_arrays(chainladder, name)
    grid = plain_grid(origins, lags, values)
    premium, weight = sloped_premium(chainladder, sample, grid)
    fitted = fit_conventional_grid(grid, CC("bf", expected_loss_ratio=0.6), premium=premium)
    reference = chainladder.BornhuetterFerguson(apriori=0.6).fit(sample, sample_weight=weight)
    np.testing.assert_allclose(
        fitted.origins["ultimate"], reference.ultimate_.values.ravel(), rtol=1e-9
    )


@pytest.mark.tieout
@pytest.mark.parametrize("decay", [1.0, 0.8])
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_cape_cod_from_plain_arrays_matches_chainladder(chainladder, name, decay):
    sample, origins, lags, values = plain_arrays(chainladder, name)
    grid = plain_grid(origins, lags, values)
    premium, weight = sloped_premium(chainladder, sample, grid)
    fitted = fit_conventional_grid(grid, CC("gcc", decay=decay), premium=premium)
    # trend=0: this Cape Cod has no trend, and chainladder's default is 5%.
    reference = chainladder.CapeCod(trend=0, decay=decay).fit(sample, sample_weight=weight)
    np.testing.assert_allclose(
        fitted.origins["ultimate"], reference.ultimate_.values.ravel(), rtol=1e-9
    )
    # A premium that moves across the origins is what makes decay matter.
    flat = fit_conventional_grid(grid, CC("gcc", decay=decay), premium=dict.fromkeys(premium, 1.0))
    assert not np.allclose(flat.origins["ultimate"], fitted.origins["ultimate"])


@pytest.mark.tieout
def test_the_raa_information_date_is_its_last_diagonal(chainladder):
    _, origins, lags, values = plain_arrays(chainladder, "raa")
    assert fit_conventional_grid(plain_grid(origins, lags, values), CC()).as_of == dt.date(
        1990, 12, 31
    )


# -- premium --------------------------------------------------------------------


@pytest.fixture
def six_grid(six) -> dict:
    return grid_from(six)


BF = CC("bf", expected_loss_ratio=0.7)


@pytest.mark.parametrize(
    "spell",
    [
        pytest.param(lambda p: dict(reversed(list(p.items()))), id="shuffled_dict"),
        pytest.param(lambda p: pd.Series(p), id="series"),
        pytest.param(
            lambda p: pd.Series(list(p.values()), index=pd.to_datetime(list(p))), id="dates"
        ),
        pytest.param(lambda p: {o.isoformat(): v for o, v in p.items()}, id="iso_strings"),
        pytest.param(
            lambda p: {dt.datetime(o.year, o.month, o.day): v for o, v in p.items()}, id="datetimes"
        ),
        # What dict(zip(frame["origin"].values, ...)) gives on a datetime64 column.
        pytest.param(
            lambda p: dict(
                zip(pd.to_datetime(list(p)).values, np.array(list(p.values())), strict=True)
            ),
            id="numpy_datetime64",
        ),
    ],
)
def test_premium_is_matched_by_origin_however_it_is_spelled(six_grid, spell):
    expected = fit_conventional_grid(six_grid, BF, premium=six_premium())
    fitted = fit_conventional_grid(six_grid, BF, premium=spell(six_premium()))
    pd.testing.assert_frame_equal(fitted.origins, expected.origins, check_exact=True)


def test_premium_follows_its_origin_not_its_position(six_grid):
    # Premium given in reverse order must not be read in reverse order.
    reversed_premium = dict(reversed(list(six_premium().items())))
    fitted = fit_conventional_grid(six_grid, BF, premium=reversed_premium)
    np.testing.assert_array_equal(fitted.grid["premium"], SIX_PREMIUM)


def test_premium_refusals(six_grid):
    premium = six_premium()
    colliding = {**premium, SIX_ORIGINS[0].isoformat(): 1.0}
    # each refusal names the origin by its first day and carries it as a cell
    with pytest.raises(Refusal, match=r"more than one amount for origin\(s\) 2010-01-01") as got:
        fit_conventional_grid(six_grid, BF, premium=colliding)
    assert got.value.reason == "duplicate"
    assert [c.origin_period for c in got.value.cells] == [dt.date(2010, 1, 1)]

    short = {o: v for o, v in premium.items() if o != SIX_ORIGINS[3]}
    with pytest.raises(Refusal, match=r"no amount for origin\(s\) 2013-01-01$") as got:
        fit_conventional_grid(six_grid, BF, premium=short)
    assert got.value.reason == "origin_not_covered"

    extra = {**premium, dt.date(2016, 1, 1): 800.0}
    with pytest.raises(Refusal, match=r"2016-01-01 that are not in the grid") as got:
        fit_conventional_grid(six_grid, BF, premium=extra)
    assert got.value.reason == "not_in_triangle"
    assert [c.value for c in got.value.cells] == [800.0]

    with pytest.raises(ValueError, match="keyed by origin period"):
        fit_conventional_grid(six_grid, BF, premium=list(SIX_PREMIUM))

    with pytest.raises(ValueError, match="premium key 2010 is not an origin period"):
        fit_conventional_grid(six_grid, BF, premium={**premium, 2010: 1.0})

    with pytest.raises(ValueError, match="premium key 'garbage' is not an origin period"):
        fit_conventional_grid(six_grid, BF, premium={**premium, "garbage": 1.0})


@pytest.mark.parametrize("amount", ["500", pd.NA, None, True], ids=["text", "na", "none", "bool"])
def test_a_premium_amount_that_is_not_a_number_is_refused(six_grid, amount):
    premium = {**six_premium(), SIX_ORIGINS[2]: amount}
    with pytest.raises(ValueError, match=r"premium for origin 2012-01-01 is .* not a number"):
        fit_conventional_grid(six_grid, BF, premium=premium)


@pytest.mark.parametrize(
    "carried",
    [
        pytest.param(lambda p: list(p), id="list"),
        pytest.param(lambda p: p.astype(str), id="text_array"),
    ],
)
def test_premium_on_the_grid_must_be_a_float_array(six, carried):
    grid = cohort_grid(six.as_of(SIX_AS_OF), loss_field="paid_loss", premium_field="earned_premium")
    grid["premium"] = carried(grid["premium"])
    with pytest.raises(ValueError, match="grid premium must be a float array"):
        fit_conventional_grid(grid, BF)


def test_premium_with_a_chain_ladder_candidate_is_refused_not_ignored(six_grid):
    with pytest.raises(ValueError, match="'cl' candidate, which never reads it"):
        fit_conventional_grid(six_grid, CC(), premium=six_premium())


def test_premium_given_twice_is_refused(six):
    carried = cohort_grid(
        six.as_of(SIX_AS_OF), loss_field="paid_loss", premium_field="earned_premium"
    )
    with pytest.raises(ValueError, match="premium was given twice"):
        fit_conventional_grid(carried, BF, premium=six_premium())


@pytest.mark.parametrize("spec", [BF, CC("gcc", decay=0.5)], ids=["bf", "gcc"])
def test_exposure_methods_refuse_a_missing_premium(six_grid, spec):
    with pytest.raises(ValueError, match=f"a '{spec.method}' candidate needs premium"):
        fit_conventional_grid(six_grid, spec)


def test_premium_on_the_grid_must_have_one_value_per_origin(six):
    carried = cohort_grid(
        six.as_of(SIX_AS_OF), loss_field="paid_loss", premium_field="earned_premium"
    )
    carried["premium"] = carried["premium"][:-1]
    with pytest.raises(ValueError, match="one value per origin, 6 in all"):
        fit_conventional_grid(carried, BF)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_unusable_premium_is_refused_through_the_array_path(six_grid, bad):
    premium = {**six_premium(), SIX_ORIGINS[-1]: bad}
    with pytest.raises(ValueError, match="premium must be finite and positive"):
        fit_conventional_grid(six_grid, BF, premium=premium)


# -- origin and development grains ---------------------------------------------


def test_quarterly_origins_on_an_annual_step_are_refused():
    # Two origins three months apart, depths two and one: a run-off staircase
    # by index, so the grid builder accepts it. But the second origin's twelve
    # months end a quarter after the first's twenty-four, and GCC would count
    # the distance between them as one whole period.
    cells = {dt.date(2010, 1, 1): [100.0, 200.0], dt.date(2010, 4, 1): [150.0]}
    grid = cohort_grid_frame(rows_frame(cells), dev_grain_months=12, measure="cumulative")
    with pytest.raises(ValueError, match=r"\[3\] months apart but the development step is 12"):
        fit_conventional_grid(grid, CC())


def test_a_gap_that_is_not_a_whole_number_of_steps_is_refused():
    cells = {
        dt.date(2010, 1, 1): [100.0, 200.0],
        dt.date(2011, 1, 1): [110.0, 210.0],
        dt.date(2012, 7, 1): [120.0],
    }
    grid = cohort_grid_frame(rows_frame(cells), dev_grain_months=12, measure="cumulative")
    with pytest.raises(ValueError, match=r"\[12, 18\] months apart"):
        fit_conventional_grid(grid, CC())


def test_origins_all_two_steps_apart_are_refused():
    cells = {
        dt.date(2010, 1, 1): [100.0, 200.0],
        dt.date(2012, 1, 1): [110.0, 210.0],
        dt.date(2014, 1, 1): [120.0],
    }
    grid = cohort_grid_frame(rows_frame(cells), dev_grain_months=12, measure="cumulative")
    with pytest.raises(ValueError, match=r"\[24\] months apart"):
        fit_conventional_grid(grid, CC())


@pytest.mark.parametrize(
    "grain,as_of",
    [("Q", dt.date(2010, 12, 31)), ("M", dt.date(2010, 4, 30))],
    ids=["quarterly", "monthly"],
)
def test_matching_finer_grains_are_accepted_and_dated_at_the_period_end(backend_name, grain, as_of):
    values = SIX[2:, :4]  # a four-origin staircase
    tri = make_cohort_triangle(backend_name, values, dev_grain=grain)
    by_grid = fit_conventional_grid(grid_from(tri), CC())
    # The youngest origin starts in October (quarterly) or April (monthly) and
    # has one period of development, so the fit is dated at that period's end.
    assert by_grid.as_of == as_of
    assert_same_fit(by_grid, fit_conventional(tri, CC(), as_of=as_of))


GAPPED = {
    dt.date(2001, 1, 1): [80.0, 170.0, 200.0],
    dt.date(2003, 1, 1): [100.0, 210.0, 260.0],
    dt.date(2004, 1, 1): [120.0, 250.0],
    dt.date(2005, 1, 1): [130.0],
}
GAPPED_PREMIUM = {
    dt.date(2001, 1, 1): 300.0,
    dt.date(2003, 1, 1): 360.0,
    dt.date(2004, 1, 1): 420.0,
    dt.date(2005, 1, 1): 450.0,
}


@pytest.mark.parametrize("spec", [CC(), CC("gcc", decay=0.5)], ids=["cl", "gcc_half"])
def test_a_gap_in_the_origin_axis_is_accepted_and_equals_the_triangle_path(backend_name, spec):
    # 2002 is missing; 2001 has run off to the last dev step, so the rest is a
    # staircase. GCC must measure 2001 to 2003 as two periods, not one.
    tri = long_triangle(GAPPED, backend_name)
    premium = None
    if spec.method != "cl":
        tri = with_premium(tri, GAPPED_PREMIUM, backend_name)
        premium = GAPPED_PREMIUM
    by_grid = fit_conventional_grid(grid_from(tri), spec, premium=premium)
    assert by_grid.as_of == dt.date(2005, 12, 31)
    assert_same_fit(by_grid, fit_conventional(tri, spec, as_of="2005-12-31"))


def test_cape_cod_counts_a_missing_origin_year_in_its_distances():
    # Both paths above share one estimator, so their agreement cannot show how
    # GCC measures distance. Here the answer is worked out by hand: 2001 to 2003
    # is two years apart, so their weight is 0.5 ** 2, not 0.5 ** 1.
    grid = cohort_grid_frame(rows_frame(GAPPED), dev_grain_months=12, measure="cumulative")
    fitted = fit_conventional_grid(grid, CC("gcc", decay=0.5), premium=GAPPED_PREMIUM)

    f1 = (170.0 + 210.0 + 250.0) / (80.0 + 100.0 + 120.0)  # 2001, 2003, 2004
    f2 = (200.0 + 260.0) / (170.0 + 210.0)  # 2001, 2003
    beta = np.array([1 / (f1 * f2), 1 / f2, 1.0])  # share emerged at 12, 24, 36
    latest = np.array([200.0, 260.0, 250.0, 130.0])
    developed = beta[[2, 2, 1, 0]]
    premium = np.array(list(GAPPED_PREMIUM.values()))
    years = np.array([2001, 2003, 2004, 2005])
    weights = 0.5 ** np.abs(years[:, None] - years[None, :])
    elr = (weights @ latest) / (weights @ (premium * developed))
    ultimate = latest + premium * elr * (1 - developed)

    np.testing.assert_allclose(fitted.origins["expected_loss_ratio"], elr, rtol=1e-12)
    np.testing.assert_allclose(fitted.origins["ultimate"], ultimate, rtol=1e-12)
    # Counting rows instead of years gives a different answer, so the check
    # above can tell the two apart.
    rows = np.arange(4)
    by_row = 0.5 ** np.abs(rows[:, None] - rows[None, :])
    assert not np.allclose(elr, (by_row @ latest) / (by_row @ (premium * developed)))


def test_an_open_origin_behind_the_others_is_refused():
    # Index-wise a staircase, so the grid builder accepts it. But 2002 has not
    # reached the last dev step and its latest cell is dated 2004-12-31 while
    # 2004 and 2005 are observed to 2005-12-31: 2002's 2005 cell is missing.
    cells = {
        dt.date(2001, 1, 1): [80.0, 170.0, 200.0, 210.0],
        dt.date(2002, 1, 1): [90.0, 180.0, 220.0],
        dt.date(2004, 1, 1): [120.0, 250.0],
        dt.date(2005, 1, 1): [130.0],
    }
    grid = cohort_grid_frame(rows_frame(cells), dev_grain_months=12, measure="cumulative")
    with pytest.raises(
        ValueError, match=r"origins \[datetime.date\(2002, 1, 1\)\] are still developing"
    ):
        fit_conventional_grid(grid, CC())


# -- the information date --------------------------------------------------------


def test_the_information_date_is_the_latest_diagonal(six_grid):
    assert fit_conventional_grid(six_grid, CC()).as_of == SIX_AS_OF


def test_run_off_origins_do_not_pull_the_information_date_back():
    # Every origin before 2013 has run off at 36 months, well before 2014.
    cells = {dt.date(2010 + i, 1, 1): [100.0, 180.0, 200.0][: min(3, 5 - i)] for i in range(5)}
    grid = cohort_grid_frame(rows_frame(cells), dev_grain_months=12, measure="cumulative")
    assert fit_conventional_grid(grid, CC()).as_of == dt.date(2014, 12, 31)


# -- the grid itself ---------------------------------------------------------------


def test_an_incremental_grid_is_refused(six):
    with pytest.raises(ValueError, match="grid measure is 'incremental'"):
        fit_conventional_grid(grid_from(six, measure="incremental"), CC())


def _set(key, value):
    def change(grid):
        grid[key] = value(grid) if callable(value) else value

    return change


def _without(key):
    def change(grid):
        del grid[key]

    return change


def _flip_one_mask_cell(grid):
    grid["obs_mask"] = grid["obs_mask"].copy()
    grid["obs_mask"][0, -1] = False


def _one_latest_dev_short(grid):
    grid["latest_dev"] = grid["latest_dev"].copy()
    grid["latest_dev"][0] -= 1


def _hole(grid):
    grid["cum"] = grid["cum"].copy()
    grid["cum"][0, 2] = np.nan
    grid["obs_mask"] = ~np.isnan(grid["cum"])


def _empty(grid):
    grid.update(
        n_w=0,
        cum=np.empty((0, grid["n_d"])),
        obs_mask=np.empty((0, grid["n_d"]), dtype=bool),
        latest_dev=np.empty(0, dtype=int),
        origin_periods=[],
    )


CORRUPTIONS = {
    "missing_key": (_without("latest_dev"), r"grid is missing \['latest_dev'\]"),
    "cum_wrong_shape": (_set("cum", lambda g: g["cum"][:, :-1]), "shape"),
    "cum_not_an_array": (_set("cum", lambda g: g["cum"].tolist()), "float array"),
    "mask_disagrees": (_flip_one_mask_cell, "obs_mask"),
    "mask_not_boolean": (_set("obs_mask", lambda g: g["obs_mask"].astype(int)), "obs_mask"),
    "latest_dev_wrong": (_one_latest_dev_short, "latest_dev"),
    "latest_dev_not_integer": (
        _set("latest_dev", lambda g: g["latest_dev"].astype(float)),
        "latest_dev",
    ),
    "origins_too_few": (_set("origin_periods", lambda g: g["origin_periods"][:-1]), "5 origin"),
    "origins_out_of_order": (
        _set("origin_periods", lambda g: g["origin_periods"][::-1]),
        "increasing order",
    ),
    "hole_in_the_triangle": (_hole, "not a run-off triangle"),
    "no_dev_step": (_set("dev_grain_months", 0), "dev_grain_months"),
    # True == 1, so without its own check this would read as a one-month step.
    "dev_step_is_bool": (_set("dev_grain_months", True), "dev_grain_months"),
    "n_w_not_whole": (_set("n_w", lambda g: float(g["n_w"])), "n_w and n_d must be whole"),
    "n_d_not_whole": (_set("n_d", lambda g: float(g["n_d"])), "n_w and n_d must be whole"),
    "origins_not_dates": (
        _set("origin_periods", lambda g: [o.year for o in g["origin_periods"]]),
        "origin_periods must be dates",
    ),
    "origins_at_year_end": (
        _set("origin_periods", lambda g: [dt.date(o.year, 12, 31) for o in g["origin_periods"]]),
        "first day of their period",
    ),
    "measure_misspelled": (_set("measure", "cumulativ"), "grid measure is 'cumulativ'"),
    "empty": (_empty, "no cells"),
}

#: Both array fits read the same grid and must refuse the same corruptions.
ARRAY_FITS = {
    "conventional": lambda grid: fit_conventional_grid(grid, CC()),
    "mack": fit_mack_grid,
}


@pytest.mark.parametrize("fit", list(ARRAY_FITS))
@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_a_corrupted_grid_is_refused(six_grid, name, fit):
    corrupt, message = CORRUPTIONS[name]
    corrupt(six_grid)
    with pytest.raises(ValueError, match=message):
        ARRAY_FITS[fit](six_grid)


@pytest.mark.parametrize("fit", list(ARRAY_FITS))
def test_both_array_fits_accept_the_uncorrupted_grid(six_grid, fit):
    # The refusals above are only worth something if the same grid, untouched,
    # is accepted.
    ARRAY_FITS[fit](six_grid)


def test_mack_refuses_an_incremental_grid(six):
    with pytest.raises(ValueError, match="grid measure is 'incremental'"):
        fit_mack_grid(grid_from(six, measure="incremental"))


def test_mack_refuses_quarterly_origins_on_an_annual_step():
    cells = {dt.date(2010, 1, 1): [100.0, 200.0], dt.date(2010, 4, 1): [150.0]}
    grid = cohort_grid_frame(rows_frame(cells), dev_grain_months=12, measure="cumulative")
    with pytest.raises(ValueError, match=r"\[3\] months apart but the development step is 12"):
        fit_mack_grid(grid)


def test_mack_reads_origin_periods_written_as_iso_strings_as_dates(six_grid):
    expected = fit_mack_grid(six_grid)
    six_grid["origin_periods"] = [o.isoformat() for o in six_grid["origin_periods"]]
    fitted = fit_mack_grid(six_grid)
    assert fitted.origin_periods == SIX_ORIGINS
    np.testing.assert_array_equal(fitted.f, expected.f)


# -- building the grid from a frame -----------------------------------------------


def test_the_grid_builder_reads_iso_string_origins_as_dates(six_grid):
    # A JSON request carries dates as text.
    frame = rows_frame(dict(zip(SIX_ORIGINS, [row[~np.isnan(row)] for row in SIX], strict=True)))
    frame["origin_period"] = frame["origin_period"].map(dt.date.isoformat)
    grid = cohort_grid_frame(frame, dev_grain_months=12, measure="cumulative")
    assert grid["origin_periods"] == SIX_ORIGINS
    assert all(type(o) is dt.date for o in grid["origin_periods"])
    np.testing.assert_array_equal(grid["cum"], six_grid["cum"])


def test_the_grid_builder_merges_two_spellings_of_one_origin():
    # One origin written as a date in one row and as text in another is one
    # origin; its two cells land in one row of the grid.
    frame = rows_frame({dt.date(2010, 1, 1): [100.0, 200.0], dt.date(2011, 1, 1): [150.0]})
    frame.loc[1, "origin_period"] = "2010-01-01"
    grid = cohort_grid_frame(frame, dev_grain_months=12, measure="cumulative")
    assert grid["origin_periods"] == [dt.date(2010, 1, 1), dt.date(2011, 1, 1)]
    np.testing.assert_array_equal(grid["cum"], [[100.0, 200.0], [150.0, np.nan]])


def test_the_grid_builder_refuses_a_measure_it_does_not_know():
    frame = rows_frame({dt.date(2010, 1, 1): [100.0, 200.0], dt.date(2011, 1, 1): [150.0]})
    with pytest.raises(ValueError, match="measure must be one of .* got 'cumulativ'"):
        cohort_grid_frame(frame, dev_grain_months=12, measure="cumulativ")


def test_the_grid_builder_refuses_a_row_with_no_origin():
    frame = rows_frame({dt.date(2010, 1, 1): [100.0, 200.0], dt.date(2011, 1, 1): [150.0]})
    frame["origin_period"] = frame["origin_period"].astype(object)
    frame.loc[2, "origin_period"] = None
    with pytest.raises(ValueError, match="origin_period has missing values"):
        cohort_grid_frame(frame, dev_grain_months=12, measure="cumulative")


def test_the_grid_builder_refuses_origins_that_are_not_dates():
    frame = rows_frame({dt.date(2010, 1, 1): [100.0, 200.0], dt.date(2011, 1, 1): [150.0]})
    frame["origin_period"] = [2010, 2010, 2011]
    with pytest.raises(ValueError, match="origin_period values must be dates"):
        cohort_grid_frame(frame, dev_grain_months=12, measure="cumulative")


def test_origin_periods_written_as_iso_strings_are_read_as_dates(six_grid):
    spec = CC("gcc", decay=0.5)
    expected = fit_conventional_grid(six_grid, spec, premium=six_premium())
    six_grid["origin_periods"] = [o.isoformat() for o in six_grid["origin_periods"]]
    fitted = fit_conventional_grid(six_grid, spec, premium=six_premium())
    pd.testing.assert_frame_equal(fitted.origins, expected.origins, check_exact=True)
    assert fitted.origins["origin_period"].tolist() == SIX_ORIGINS


def test_the_callers_grid_is_not_changed(six_grid):
    before = copy.deepcopy(six_grid)
    fitted = fit_conventional_grid(six_grid, BF, premium=six_premium())
    assert "premium" not in six_grid
    assert fitted.grid is not six_grid
    assert six_grid.keys() == before.keys()
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(six_grid[key], value)
        else:
            assert six_grid[key] == value


def test_the_array_entry_points_are_exported():
    assert kernels.fit_conventional_grid is fit_conventional_grid
    assert kernels.cohort_grid_frame is cohort_grid_frame
    assert kernels.fit_mack_grid is fit_mack_grid
    assert {"fit_conventional_grid", "cohort_grid_frame", "fit_mack_grid"} <= set(kernels.__all__)
