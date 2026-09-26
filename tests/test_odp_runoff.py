"""The full run-off ODP bootstrap's kernel: residual options, the draws, the refit, tails.

Sections:

1. the one-year CDR's ODP route did not move: every digest of
   ``tests/data/odp_cdr_pin.json`` (frozen from ``feat/tails`` by
   ``scripts/freeze_odp_cdr_pin.py``) is recomputed and must match bit for bit;
2. the residual options of ``fit_odp_bootstrap``: the adjustment, the pool,
   leverage, excluded cells, the scale, negative increments;
3. the draws: seeds, chunks, streams, the prior multiplier, finiteness;
4. the refit: one specification with the central fit, the position rules,
   the unit factor, tails;
5. chainladder-python 0.9.2 on shared random numbers (marker ``tieout``);
6. Monte Carlo agreement with R's ``BootChainLadder`` (frozen in
   ``tests/data/r_bootchainladder.json`` by ``scripts/r_bootchainladder.R``),
   with chainladder-python's own runs, and with the Reserving app's workbook.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ibnr.errors import Refusal
from ibnr.kernels import odp_bootstrap as kernel
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional_grid
from ibnr.kernels.grid import grid_from_columns
from ibnr.kernels.links import (
    LinkRules,
    link_factors,
    link_factors_many,
    position_rules,
    select_links,
)
from ibnr.kernels.odp_bootstrap import (
    LEVERAGE_ONE,
    POOL_REASONS,
    draw_runoff,
    fit_odp_bootstrap,
    future_cell_means,
    prepare_runoff,
)
from ibnr.kernels.tail import TailSpec

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import freeze_odp_cdr_pin as frozen  # noqa: E402

PUBLIC = json.loads((DATA / "refusal_triangles.json").read_text("utf-8"))
PIN = json.loads((DATA / "odp_cdr_pin.json").read_text("utf-8"))
R_BOOT = json.loads((DATA / "r_bootchainladder.json").read_text("utf-8"))
WORKBOOK = json.loads((DATA / "workbook_paid_bootstrap.json").read_text("utf-8"))

CL = ConventionalCandidate("cl")


def grid_of(rows, step: int = 12) -> dict:
    years, lags, values = zip(*rows, strict=True)
    return grid_from_columns(
        np.array([dt.date(y, 1, 1) for y in years], dtype="datetime64[D]"),
        np.array(lags),
        np.array(values, dtype=float),
        dev_grain_months=step,
        measure="cumulative",
    )


def rows_of(matrix, first_year: int = 2001, step: int = 12) -> list[list]:
    return [
        [first_year + i, step * (j + 1), float(v)]
        for i, row in enumerate(matrix)
        for j, v in enumerate(row)
        if v is not None and not np.isnan(v)
    ]


def premium_of(grid) -> dict:
    """Premium keyed by origin period, rising from 1 to 2 times the largest amount."""
    top = float(np.nanmax(grid["cum"]))
    amounts = np.linspace(1.0, 2.0, grid["n_w"]) * top
    return {
        dt.date.fromisoformat(str(o)[:10]): float(a)
        for o, a in zip(grid["origin_periods"], amounts, strict=True)
    }


def setup_of(grid, candidate=CL, premium=None, **options):
    options.setdefault("negative_increments", "reflect")
    return prepare_runoff(grid, candidate, premium=premium, **options)


def pseudo_of(boot, index) -> np.ndarray:
    """The simulated cumulative triangles for residual indices, as draw_runoff builds them."""
    mask = boot.obs_mask
    mean = np.where(mask, boot.fitted, 0.0)
    return np.cumsum(np.where(mask, boot.pool[index] * np.sqrt(np.abs(mean)) + mean, 0.0), axis=2)


def fitted_triangle(boot) -> np.ndarray:
    """The fitted cumulatives on the observed cells, as one simulated triangle."""
    return np.cumsum(np.where(boot.obs_mask, boot.fitted, 0.0), axis=1)[None]


def glm_leverage(mask, fitted) -> np.ndarray:
    """chainladder-python's hat matrix, written out: X (X'WX)^-1 X'W, W = diag(m)."""
    n_w, n_d = mask.shape
    cells = [(i, j) for i in range(n_w) for j in range(n_d) if mask[i, j] and fitted[i, j] != 0]
    x = np.zeros((len(cells), n_w + n_d - 1))
    for r, (i, j) in enumerate(cells):
        x[r, i] = 1.0
        if j:
            x[r, n_w + j - 1] = 1.0
    w = np.array([abs(fitted[i, j]) for i, j in cells])
    hat = x @ np.linalg.inv(x.T @ (w[:, None] * x)) @ x.T * w[None, :]
    out = np.full((n_w, n_d), np.nan)
    for r, (i, j) in enumerate(cells):
        out[i, j] = hat[r, r]
    return out


RAA = grid_of(PUBLIC["raa"])
GENINS = grid_of(PUBLIC["genins"])


def odp_matrix(n: int = 8, seed: int = 4) -> np.ndarray:
    """A run-off triangle drawn from an over-dispersed Poisson model: every increment >= 0."""
    rng = np.random.default_rng(seed)
    row = np.linspace(9000.0, 12000.0, n)
    col = np.array([0.42, 0.26, 0.14, 0.08, 0.05, 0.025, 0.015, 0.01, 0.005])[:n]
    col = col / col.sum()
    cum = np.full((n, n), np.nan)
    for i in range(n):
        running = 0.0
        for j in range(n - i):
            running += 40.0 * rng.poisson(row[i] * col[j] / 40.0)
            cum[i, j] = running
    return cum


ODP = grid_of(rows_of(odp_matrix()))


# -- 1. the one-year CDR's ODP route did not move ---------------------------------


def _moved(now: dict) -> list[str]:
    return [
        f"{name}|{key}"
        for name, digests in now.items()
        for key, value in digests.items()
        if PIN[name].get(key) != value
    ]


def test_the_one_year_cdr_odp_route_did_not_move_on_the_public_triangles():
    """The defaults of fit_odp_bootstrap, draw_next_increments and the CDR's
    odp_bootstrap generator give the bits feat/tails gave, refusals included.
    Mutation: make the kernel's default pool centred, or its default
    adjustment 'none'; the digests move."""
    import freeze_conventional_pin

    now = frozen.pin(freeze_conventional_pin.public_triangles())
    assert set(now) == {"raa", "genins", "ukmotor", "abc", "mw2014", "formula_30"}
    assert _moved(now) == []


@pytest.mark.tieout
def test_the_one_year_cdr_odp_route_did_not_move_on_clrd():
    pytest.importorskip("chainladder")
    import freeze_conventional_pin

    now = frozen.pin(freeze_conventional_pin.clrd_triangles())
    assert len(now) == 36
    assert _moved(now) == []


# -- 2. the residual options -------------------------------------------------------


def test_the_three_adjustments_scale_the_same_residuals_differently():
    """``none`` is the unscaled residual, ``dof`` that times sqrt(n / (n - p)) bit
    for bit, and ``hat`` each divided by sqrt(1 - h), 0 where h is one.
    Mutations: have ``none`` apply the dof factor, or ``hat`` fall back to dof;
    this fails."""
    boots = {
        adjustment: setup_of(RAA, adjustment=adjustment, pool="all").boot
        for adjustment in ("none", "dof", "hat")
    }
    none, dof, hat = boots["none"], boots["dof"], boots["hat"]
    n, p = none.n_cells, none.n_params
    assert (n, p) == (55, 19)
    assert np.array_equal(none.pool, none.unscaled[none.pool_mask])
    assert np.array_equal(dof.pool, none.pool * np.sqrt(n / (n - p)))
    exact = hat.pool_mask & (hat.leverage > LEVERAGE_ONE)
    assert exact.sum() == 2  # the first origin's last cell and the last origin's first
    assert np.array_equal(hat.residuals[exact], [0.0, 0.0])
    rest = hat.pool_mask & ~exact
    expected = none.unscaled[rest] / np.sqrt(1 - hat.leverage[rest])
    np.testing.assert_allclose(hat.residuals[rest], expected, rtol=1e-14)
    assert not np.allclose(hat.pool, dof.pool)
    # the scale is the same whatever the adjustment: it is of the unscaled residuals
    assert none.phi == dof.phi == hat.phi


def test_the_leverage_is_the_odp_glm_hat_matrix():
    """The pseudo-inverse leverage equals chainladder-python's explicit inverse
    where that exists, and the leverages add up to the GLM's parameter count."""
    for grid in (RAA, GENINS):
        boot = setup_of(grid, adjustment="hat").boot
        cells = boot.obs_mask & (boot.fitted != 0)
        expected = glm_leverage(boot.obs_mask, boot.fitted)
        np.testing.assert_allclose(boot.leverage[cells], expected[cells], rtol=1e-9, atol=1e-12)
        assert boot.leverage[cells].sum() == pytest.approx(boot.n_params, rel=1e-12)
        assert ((boot.leverage[cells] >= -1e-12) & (boot.leverage[cells] <= 1 + 1e-12)).all()


def test_cells_fitted_exactly_are_found_by_leverage_not_by_a_zero_residual():
    """On genins without an adjustment the first origin's last residual is
    1.8e-12, not 0, from rounding; chainladder-python tests the residual for
    zero and resamples it. Found by leverage, it leaves the centred pool.
    Mutation: find those cells with ``residual != 0``; the pool has 54."""
    boot = setup_of(GENINS, adjustment="none", pool="centred").boot
    assert 0 < abs(boot.unscaled[0, -1]) < 1e-9  # the rounding is real
    assert boot.pool.size == 53
    assert POOL_REASONS[boot.pool_reason[0, -1]] == "leverage_one"
    assert POOL_REASONS[boot.pool_reason[-1, 0]] == "leverage_one"
    everything = setup_of(GENINS, adjustment="none", pool="all").boot
    assert everything.pool.size == 55


def test_the_centred_pool_has_mean_zero_and_the_whole_pool_is_rs():
    """``centred`` subtracts the pool's mean; ``all`` is R's pool, uncentred, with
    R's size and mean on RAA. Mutations: centre under ``all``, or not under
    ``centred``; this fails."""
    centred = setup_of(RAA, adjustment="hat", pool="centred").boot
    assert abs(centred.pool.mean()) < 1e-12 * np.abs(centred.pool).max()
    r = next(run for run in R_BOOT["runs"] if run["triangle"] == "RAA")
    whole = setup_of(RAA, adjustment="dof", pool="all").boot
    assert whole.pool.size == r["pool_size"] == 55
    assert (np.abs(whole.pool) < 1e-9).sum() == r["pool_zeros"] == 2
    assert whole.pool.mean() == pytest.approx(r["pool_mean"], rel=1e-9)


def test_a_zero_increment_is_data_and_stays_in_the_pool():
    """A cumulative that does not move is an increment of 0 against a positive
    fitted mean: its residual is -sqrt(m) and it is resampled (chainladder-python
    stores it as missing and drops it). Mutation: leave zero increments out of
    the pool; this fails."""
    matrix = odp_matrix()
    matrix[1, 3] = matrix[1, 2]  # nothing paid in that year
    matrix[1, 4:] = matrix[1, 4:] - (odp_matrix()[1, 3] - odp_matrix()[1, 2])
    boot = setup_of(grid_of(rows_of(matrix)), negative_increments="refuse").boot
    assert boot.inc[1, 3] == 0.0 and boot.fitted[1, 3] > 0
    assert boot.unscaled[1, 3] == pytest.approx(-np.sqrt(boot.fitted[1, 3]))
    for pool in ("all", "centred"):
        found = setup_of(grid_of(rows_of(matrix)), pool=pool).boot
        assert found.pool_mask[1, 3], pool
        assert POOL_REASONS[found.pool_reason[1, 3]] == "pooled"


def test_an_excluded_link_takes_its_later_cell_out_and_a_first_age_link_both_cells():
    """chainladder-python's cell rule: excluding (1981, 12) removes 1981's cells
    at 12 and 24; excluding (1985, 36) removes only 1985 at 48."""
    first = ConventionalCandidate("cl", exclude=((dt.date(1981, 1, 1), 12),))
    setup = setup_of(RAA, first)
    assert set(zip(*np.nonzero(setup.excluded), strict=True)) == {(0, 0), (0, 1)}
    assert POOL_REASONS[setup.boot.pool_reason[0, 0]] == "excluded_link"
    later = ConventionalCandidate("cl", exclude=((dt.date(1985, 1, 1), 36),))
    setup = setup_of(RAA, later)
    assert set(zip(*np.nonzero(setup.excluded), strict=True)) == {(4, 3)}
    assert setup.boot.pool.size == 54  # 55 cells, one out


def test_a_cell_whose_ratios_are_each_the_only_one_left_is_fitted_exactly():
    """With a 5-year window and the highest ratio dropped, 1981's ratio is the
    only one left from 96 months on, so the central factors reproduce 1981 at
    108 months as well as at 120: both are out of the centred pool, as
    chainladder-python drops their zero residuals, though 1981 at 108 months
    has a leverage below one among all the cells. Mutation: find the exact
    cells by leverage alone; 1981 at 108 months is resampled."""
    candidate = ConventionalCandidate(
        "cl", history_periods=5, drop_high=1, trim_ties="volume", exhausted_exclusions="keep"
    )
    boot = setup_of(RAA, candidate, adjustment="hat", pool="centred").boot
    exact = boot.pool_reason == POOL_REASONS.index("leverage_one")
    assert sorted(zip(*np.nonzero(exact), strict=True)) == [(0, 8), (0, 9), (9, 0)]
    assert boot.leverage[0, 8] < LEVERAGE_ONE
    assert abs(boot.unscaled[0, 8]) < 1e-9
    assert boot.pool.size == 29


def test_a_factor_of_exactly_one_does_not_switch_the_hat_adjustment_off():
    """A development age with no payments makes its fitted increments 0 and the
    GLM's weighted design singular; chainladder-python then drops the hat
    adjustment with a warning. The pseudo-inverse leaves that column out.
    Mutation: invert X'WX directly; this is refused or not finite."""
    matrix = odp_matrix()
    matrix[0, 7] = matrix[0, 6]  # the last age develops by exactly 1.0
    boot = setup_of(grid_of(rows_of(matrix)), adjustment="hat").boot
    assert boot.fitted[0, 7] == 0.0
    assert POOL_REASONS[boot.pool_reason[0, 7]] == "zero_fitted_mean"
    cells = boot.obs_mask & (boot.fitted != 0)
    assert np.isfinite(boot.leverage[cells]).all()
    assert boot.leverage[cells].sum() == pytest.approx(boot.n_params - 1, rel=1e-12)
    assert np.isfinite(boot.pool).all() and boot.adjustment == "hat"


def test_the_scale_counts_every_observed_cell_and_sums_only_the_pooled_ones():
    """``phi`` sums the squared unscaled residuals of the cells no development
    option took out, over every observed cell less the parameters.
    Mutation: compute phi from the adjusted residuals; this fails."""
    candidate = ConventionalCandidate("cl", history_periods=5)
    boot = setup_of(RAA, candidate, adjustment="hat").boot
    kept = (
        boot.obs_mask
        & (boot.fitted != 0)
        & (boot.pool_reason != POOL_REASONS.index("excluded_link"))
    )
    expected = np.sum(boot.unscaled[kept] ** 2) / (55 - 19)
    assert boot.phi == pytest.approx(expected, rel=1e-13)
    assert boot.degrees_of_freedom == 36
    assert boot.phi == pytest.approx(544.1, abs=0.05)  # chainladder's scale_ (spec 2.1)


def test_negative_increments_are_refused_or_reflected():
    """raa's 1982 falls from 72 to 84 months; ``refuse`` names that cell, and
    ``reflect`` bootstraps it with |m|."""
    with pytest.raises(Refusal, match="84 months") as caught:
        setup_of(RAA, negative_increments="refuse")
    assert caught.value.reason == "negative_increment"
    boot = setup_of(RAA).boot
    assert boot.inc[1, 6] < 0 and boot.n_negative_fitted == 0


def _falling_matrix() -> np.ndarray:
    """A triangle whose total falls from 60 to 72 months: factor below 1."""
    matrix = odp_matrix()
    matrix[:3, 5:] -= 900.0
    return matrix


def test_a_negative_fitted_mean_is_refused_or_counted():
    """Only a factor below 1 makes a fitted mean negative. With every increment
    zero or more no link factor of the data is below 1, so fit_odp_bootstrap
    is handed one directly here. A constant tail of 0.9 attached at 60 months
    stays out of the fitted values, but puts the refit's steps below 1 and
    the future cells' means below 0, so under ``refuse`` it is refused with
    the same reason, naming the future cells; under ``reflect`` it runs and
    the fitted values are the untailed ones. A triangle that falls is refused
    for its negative increments first; under ``reflect`` its negative fitted
    means are counted. Mutation: drop _refuse_a_falling_tail; the second
    refusal is not raised."""
    factors = fit_conventional_grid(ODP, CL).factors.copy()
    factors[4:] = 0.95
    with pytest.raises(Refusal, match="below 1") as caught:
        fit_odp_bootstrap(
            ODP["cum"], ODP["obs_mask"], ODP["latest_dev"], factors, dev_grain_months=12
        )
    assert caught.value.reason == "negative_fitted_mean"
    assert caught.value.links == ((60, 72), (72, 84), (84, 96))
    shrinking = ConventionalCandidate("cl", tail=TailSpec("constant", factor=0.9, attach_lag=60))
    with pytest.raises(Refusal, match="below 1") as caught:
        setup_of(ODP, shrinking, negative_increments="refuse")
    assert caught.value.reason == "negative_fitted_mean"
    assert caught.value.links == ((60, 72), (72, 84), (84, 96))
    assert caught.value.cells and all(cell.dev_lag > 60 for cell in caught.value.cells)
    reflected = setup_of(ODP, shrinking)
    assert reflected.boot.fitted.tobytes() == setup_of(ODP).boot.fitted.tobytes()
    assert reflected.boot.n_negative_fitted == 0
    grid = grid_of(rows_of(_falling_matrix()))
    with pytest.raises(Refusal) as caught:
        setup_of(grid, negative_increments="refuse")
    assert caught.value.reason == "negative_increment"
    boot = setup_of(grid).boot
    assert boot.n_negative_fitted == int((boot.obs_mask & (boot.fitted < 0)).sum()) > 0
    # the noise keeps a negative mean's sign: each draw's cell has the mean's sign
    setup = setup_of(grid)
    means = future_cell_means(setup.projection, fitted_triangle(boot)).means
    negative = means[0] < 0
    assert negative.any()
    rng = np.random.default_rng(3)
    noisy = kernel._od_process_noise(rng, np.repeat(means, 200, axis=0), boot.phi, law="gamma")
    assert (noisy[:, negative] <= 0).all()


def test_a_nonzero_increment_against_a_zero_fitted_mean_is_degenerate():
    """72 to 84 months: one origin falls by 500 and the other rises by 500, so
    the factor is exactly 1, the fitted increments are 0, and the observed ones
    are not."""
    matrix = odp_matrix()
    matrix[0, 6] = matrix[0, 5] - 500.0
    matrix[1, 6] = matrix[1, 5] + 500.0
    matrix[0, 7] = matrix[0, 6] + 100.0
    with pytest.raises(Refusal, match="zero fitted mean") as caught:
        setup_of(grid_of(rows_of(matrix)))
    assert caught.value.reason == "degenerate_fit"


def test_an_excluded_cell_with_a_zero_fitted_mean_is_not_degenerate():
    """Excluding the only ratio at the last age with a factor of 1.0 in its place
    makes 1981's fitted increment at 120 months 0 against 172 paid; that cell's
    residual is out of the pool, so nothing is undefined."""
    candidate = ConventionalCandidate(
        "cl", exclude=((dt.date(1981, 1, 1), 108),), unsupported_factor="unity"
    )
    boot = setup_of(RAA, candidate).boot
    assert boot.fitted[0, 9] == 0.0 and boot.inc[0, 9] == 172.0
    assert POOL_REASONS[boot.pool_reason[0, 9]] == "excluded_link"
    assert np.isfinite(boot.pool).all()


@pytest.mark.parametrize("adjustment", ["hat", "dof", "none"])
@pytest.mark.parametrize("pool", ["centred", "all"])
def test_a_tail_before_the_last_age_leaves_the_fitted_values_and_the_pool_alone(adjustment, pool):
    """The fitted values and the residuals come from the factors before the tail,
    so a curve attached at 84 months gives the untailed bootstrap bit for bit:
    1981's last cell, alone in its development column (leverage one), is
    fitted exactly, its residual is 0 and it is out of the centred pool, and
    no hat factor is undefined. Mutation: pass the tailed factors to
    fit_odp_bootstrap; the fitted values move and the cell keeps a residual."""
    tail = TailSpec("exponential", attach_lag=84)
    options = {"adjustment": adjustment, "pool": pool}
    boot = setup_of(RAA, ConventionalCandidate("cl", tail=tail), **options).boot
    plain = setup_of(RAA, CL, **options).boot
    assert boot.fitted.tobytes() == plain.fitted.tobytes()
    assert boot.pool.tobytes() == plain.pool.tobytes()
    assert boot.phi == plain.phi
    assert boot.leverage[0, 9] > LEVERAGE_ONE and abs(boot.unscaled[0, 9]) < 1e-9
    assert np.isfinite(boot.pool).all()
    if pool == "centred":
        assert not boot.pool_mask[0, 9]
        assert POOL_REASONS[boot.pool_reason[0, 9]] == "leverage_one"
        assert boot.pool.size == 53 and abs(boot.pool.mean()) < 1e-12


def test_too_few_cells_and_an_empty_pool_are_refused():
    tiny = grid_of(rows_of([[100.0, 150.0], [120.0, np.nan]]))
    with pytest.raises(Refusal, match="more cells than parameters") as caught:
        setup_of(tiny)
    assert caught.value.reason == "not_identified"
    grid = ODP
    with pytest.raises(Refusal) as caught:
        fit_odp_bootstrap(
            grid["cum"],
            grid["obs_mask"],
            grid["latest_dev"],
            fit_conventional_grid(grid, CL).factors,
            excluded=grid["obs_mask"],
        )
    assert caught.value.reason == "empty_residual_pool"


@pytest.mark.parametrize("broken", ["outside", "nan"])
def test_a_leverage_that_cannot_be_computed_is_refused_under_the_hat_only(monkeypatch, broken):
    def leverage(cells, weight):
        out = np.full(cells.shape, np.nan)
        out[cells] = 1.5 if broken == "outside" else np.nan
        return out

    monkeypatch.setattr(kernel, "_leverage", leverage)
    for pool in ("all", "centred"):
        with pytest.raises(Refusal, match="leverage") as caught:
            setup_of(ODP, adjustment="hat", pool=pool)
        assert caught.value.reason == "degenerate_fit"
    # the run-off bootstrap finds the exactly fitted cells from its link ratios,
    # so only the hat adjustment needs the leverage; a direct call with no
    # ``exact`` finds them by leverage, and refuses too
    for pool in ("all", "centred"):
        assert setup_of(ODP, adjustment="dof", pool=pool).boot.pool.size > 0
    grid = ODP
    factors = fit_conventional_grid(grid, CL).factors
    with pytest.raises(Refusal, match="leverage"):
        fit_odp_bootstrap(
            grid["cum"], grid["obs_mask"], grid["latest_dev"], factors, pool="centred"
        )


def test_unknown_residual_options_are_refused():
    for option, value in (
        ("adjustment", "leverage"),
        ("pool", "median"),
        ("negative_increments", "keep"),
    ):
        with pytest.raises(Refusal) as caught:
            setup_of(ODP, **{option: value})
        assert caught.value.reason == "invalid_option" and caught.value.option == option


# -- 3. the draws ------------------------------------------------------------------


def _run(setup, n=40, seed=0, **options):
    return draw_runoff(setup.boot, setup.projection, n_draws=n, seed=seed, **options)


def test_seed_zero_reproduces_and_a_seed_sequence_is_the_same_as_its_integer():
    """chainladder-python seeds the noise with ``seed + 1`` only when the seed is
    truthy, so seed 0 is unseeded there. Mutation: seed a stream only when the
    seed is truthy; this fails."""
    setup = setup_of(RAA, adjustment="hat", pool="centred")
    first, second = _run(setup, seed=0), _run(setup, seed=0)
    assert first.ibnr.tobytes() == second.ibnr.tobytes()
    third = _run(setup, seed=np.random.SeedSequence(0))
    assert first.ibnr.tobytes() == third.ibnr.tobytes()
    assert _run(setup, seed=1).ibnr.tobytes() != first.ibnr.tobytes()


def test_the_draws_do_not_depend_on_the_chunk_size_and_a_run_is_a_prefix_of_a_longer_one():
    """Each of the three streams is read on from where the last chunk left it,
    so the chunk size moves no bit (9 x 9 cells, an odd count per draw, with a
    priori multipliers). Mutation: start a chunk's residual generator afresh
    from the seed; the chunk sizes disagree."""
    setup = setup_of(
        grid_of(rows_of(odp_matrix(9))),
        ConventionalCandidate("bf", expected_loss_ratio=0.9),
        premium=premium_of(grid_of(rows_of(odp_matrix(9)))),
    )
    whole = _run(setup, n=100, seed=5, chunk_draws=100, prior_cv=0.2)
    for chunk in (1, 7, 33):
        part = _run(setup, n=100, seed=5, chunk_draws=chunk, prior_cv=0.2)
        assert part.ibnr.tobytes() == whole.ibnr.tobytes(), chunk
        assert part.prior_multiplier.tobytes() == whole.prior_multiplier.tobytes()
    shorter = _run(setup, n=37, seed=5, prior_cv=0.2)
    assert shorter.ibnr.tobytes() == whole.ibnr[:37].tobytes()


def test_the_three_streams_are_independent(monkeypatch):
    """The simulated triangles are the same whatever the process law, with the
    noise off, and with or without an a priori multiplier. Mutation: draw the
    process noise from the residual stream; the triangles of the second chunk
    move."""
    seen = []
    real = kernel.future_cell_means

    def spy(projection, pseudo, **options):
        seen.append(pseudo.copy())
        return real(projection, pseudo, **options)

    monkeypatch.setattr(kernel, "future_cell_means", spy)
    grid = ODP
    setup = setup_of(
        grid, ConventionalCandidate("bf", expected_loss_ratio=0.8), premium=premium_of(grid)
    )
    runs = [
        {"process": "gamma"},
        {"process": "od_poisson"},
        {"process_noise": False},
        {"process": "gamma", "prior_cv": 0.3},
    ]
    triangles = []
    for options in runs:
        seen.clear()
        _run(setup, n=12, seed=9, chunk_draws=4, **options)
        triangles.append(np.concatenate(seen))
    for other in triangles[1:]:
        assert other.tobytes() == triangles[0].tobytes()


def test_a_triangle_that_fits_exactly_draws_its_central_reserve_every_time():
    """No residual, no noise: every draw is the central fit's reserve."""
    matrix = np.outer(np.linspace(1000, 1500, 6), np.cumsum([0.5, 0.25, 0.12, 0.08, 0.03, 0.02]))
    matrix[np.add.outer(np.arange(6), np.arange(6)) > 5] = np.nan
    grid = grid_of(rows_of(matrix))
    setup = setup_of(grid)
    central = setup.estimate.origins["ultimate"] - setup.estimate.origins["latest"]
    drawn = _run(setup, n=20, process_noise=False)
    np.testing.assert_allclose(
        drawn.ibnr, np.broadcast_to(central, drawn.ibnr.shape), rtol=1e-9, atol=1e-9
    )


def test_the_prior_multiplier_has_mean_one_is_shared_by_origins_and_zero_draws_nothing():
    """One mean-one lognormal number per draw, with coefficient of variation
    prior_cv, multiplying every origin's a priori ultimate of that draw.
    Mutations: draw one multiplier per origin, or use exp(s Z) (mean
    exp(s^2 / 2)); this fails."""
    grid = ODP
    bf = ConventionalCandidate("bf", expected_loss_ratio=0.8)
    setup = setup_of(grid, bf, premium=premium_of(grid))
    drawn = _run(setup, n=40_000, seed=2, prior_cv=0.25, process_noise=False, chunk_draws=10_000)
    m = drawn.prior_multiplier
    se = 0.25 / np.sqrt(m.size)
    assert abs(m.mean() - 1) < 4 * se
    assert m.std(ddof=1) == pytest.approx(0.25, rel=0.03)
    # every origin's reserve moves by that draw's multiplier: BF's reserve is E (1 - p)
    index = np.zeros((6, grid["n_w"], grid["n_d"]), dtype=np.intp)
    mult = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    base = draw_runoff(
        setup.boot, setup.projection, n_draws=6, seed=0, residual_index=index, process_noise=False
    )
    scaled = draw_runoff(
        setup.boot,
        setup.projection,
        n_draws=6,
        seed=0,
        residual_index=index,
        process_noise=False,
        prior_multiplier=mult,
    )
    open_ = base.ibnr[0] != 0
    ratio = scaled.ibnr[:, open_] / base.ibnr[:, open_]
    np.testing.assert_allclose(ratio, np.broadcast_to(mult[:, None], ratio.shape), rtol=1e-12)
    # prior_cv=0 reads nothing from the multiplier stream and draws nothing
    plain = _run(setup, n=50, seed=4)
    zero = _run(setup, n=50, seed=4, prior_cv=0.0)
    assert plain.ibnr.tobytes() == zero.ibnr.tobytes() and zero.prior_multiplier is None


def test_prior_cv_is_refused_for_the_chain_ladder_and_when_negative():
    setup = setup_of(ODP)
    with pytest.raises(Refusal, match="chain ladder has none"):
        _run(setup, prior_cv=0.1)
    grid = ODP
    bf = setup_of(
        grid, ConventionalCandidate("bf", expected_loss_ratio=0.8), premium=premium_of(grid)
    )
    for cv in (-0.1, float("nan"), float("inf")):
        with pytest.raises(Refusal, match="prior_cv"):
            _run(bf, prior_cv=cv)


def test_a_draw_that_is_not_finite_is_refused_never_zero(monkeypatch):
    """Mutation: turn the draw that is not finite into 0 (the Reserving app's
    ``np.nan_to_num``); this fails."""
    real = kernel.project_ultimates

    def broken(*args, **options):
        out = real(*args, **options)
        out["ultimate"] = out["ultimate"].copy()
        out["ultimate"][1, 2] = np.nan
        return out

    monkeypatch.setattr(kernel, "project_ultimates", broken)
    setup = setup_of(ODP)
    with pytest.raises(Refusal, match="3 of the 12 simulated draws") as caught:
        _run(setup, n=12, chunk_draws=4)
    assert caught.value.reason == "result_not_finite"


def test_every_draw_is_finite_and_the_counts_are_right():
    setup = setup_of(RAA, adjustment="hat", pool="centred")
    drawn = _run(setup, n=2000, seed=3)
    assert np.isfinite(drawn.ibnr).all()
    assert drawn.ibnr.shape == (2000, 10)
    assert (drawn.ibnr[:, 0] == 0).all()  # the oldest origin has nothing left, no tail
    assert not drawn.tail_fallback.any()


# -- 4. the refit ------------------------------------------------------------------

AVERAGES = ("volume", "simple", "regression", "median")


def _candidate(method: str, **options) -> ConventionalCandidate:
    extra = {
        "cl": {},
        "bf": {"expected_loss_ratio": 0.8},
        "benktander": {"expected_loss_ratio": 0.8, "n_iters": 3},
        "gcc": {"decay": 0.6, "trend": 0.03, "n_iters": 2},
    }
    kind = "bf" if method == "benktander" else method
    return ConventionalCandidate(kind, **extra[method], **options)


@pytest.mark.parametrize("method", ["cl", "bf", "benktander", "gcc"])
@pytest.mark.parametrize("average", AVERAGES)
def test_refitting_the_fitted_triangle_gives_the_central_fit(method, average):
    """Every link ratio of the fitted triangle is the central factor, so the refit
    returns it for every average, and the refit's ultimates are the central
    ones: the central fit and every draw share one arithmetic. With a trim, the
    refit keeps every ratio (the trim acted once) and still returns the central
    factors. So does every tail: the fitted values come from the factors
    before it, so a curve refits to the central curve wherever it is attached,
    and a constant tail reads no ratio (next test for a curve attached
    before the last age)."""
    for options in (
        {},
        {"drop_high": 1, "exhausted_exclusions": "keep"},
        {"history_periods": 4},
        {"tail": TailSpec("exponential")},
        {"tail": TailSpec("exponential", attach_lag=60)},
        {"tail": TailSpec("weibull", attach_lag=72)},
        {"tail": TailSpec("constant", factor=1.05, attach_lag=60)},
    ):
        candidate = _candidate(method, average=average, **options)
        setup = setup_of(ODP, candidate, premium=None if method == "cl" else premium_of(ODP))
        found = future_cell_means(setup.projection, fitted_triangle(setup.boot))
        np.testing.assert_allclose(found.factors[0], setup.estimate.factors, rtol=1e-12)
        np.testing.assert_allclose(
            found.ultimate[0], setup.estimate.origins["ultimate"], rtol=1e-12
        )
        assert not found.unit.any()
        if "tail" in options:
            assert found.tail_factor[0] == pytest.approx(setup.estimate.tail.tail_factor, 1e-12)


def test_a_curve_attached_before_the_last_age_refits_to_the_central_curve():
    """The curve stays out of the fitted values (as in chainladder-python), so
    the fitted triangle's ratios past the attachment age are the data's
    factors, and the curve fitted to them is the one fitted to the data.
    When the curve made the fitted values, genins's tail factor refitted to
    1.02353 against the central 1.02950. The methods docstring and
    docs/coming-from-chainladder.md say this; change them with this test.
    Mutation: pass the tailed factors to fit_odp_bootstrap; this fails."""
    for spec in (
        TailSpec("exponential"),
        TailSpec("exponential", attach_lag=72),
        TailSpec("weibull", attach_lag=72),
        TailSpec("inverse_power", attach_lag=60),
    ):
        setup = setup_of(GENINS, ConventionalCandidate("cl", tail=spec))
        found = future_cell_means(setup.projection, fitted_triangle(setup.boot))
        moved = abs(found.tail_factor[0] / setup.estimate.tail.tail_factor - 1)
        assert moved < 1e-12, spec
        np.testing.assert_allclose(found.factors[0], setup.estimate.factors, rtol=1e-12)


#: The simulated mean ultimate (20,000 draws, the front door's hat and
#: centred pool, gamma process), as a share of the central IBNR, against
#: chainladder-python 0.9.2's own offset on the same setting (BootstrapODPSample
#: at 20,000 draws, then Development, TailCurve and Chainladder), each at two
#: seeds; the Monte Carlo standard error is about 0.13% on genins and 0.03% on
#: abc. Measured 2026-09-25:
#:
#: - genins, exponential at 72: ibnr +0.44%, +0.34%; chainladder +0.13%, +0.50%
#:   (-6.2% in ibnr when the curve made the fitted values);
#: - abc, exponential at 72: ibnr -0.30%, -0.28%; chainladder -0.32%, -0.29%
#:   (-3.4% before).
#:
#: The band is 1% of the central IBNR: at least four standard errors from
#: either measured offset, and far inside the old ones. A curve is not a
#: straight line in the factors, so neither library's mean sits exactly on
#: the central estimate; on raa with an inverse power curve at 60 months both
#: sit far above it (ibnr +26%, chainladder +27% to +28%), which is the curve
#: and not the fitted values, and is why raa is not in this test.
TAILED_MEAN_BAND = 0.01


@pytest.mark.parametrize("name", ["genins", "abc"])
def test_a_curve_attached_before_the_last_age_keeps_the_mean_on_the_central_estimate(name):
    """With the curve out of the fitted values, the simulated mean ultimate
    stays on the central estimate, as it does in chainladder-python (numbers
    at TAILED_MEAN_BAND). Mutation: pass the tailed factors to
    fit_odp_bootstrap; the mean falls 6.2% (genins) and 3.4% (abc) of the
    central IBNR below it."""
    grid = grid_of(PUBLIC[name])
    spec = TailSpec("exponential", attach_lag=72)
    setup = setup_of(grid, ConventionalCandidate("cl", tail=spec), adjustment="hat", pool="centred")
    central = np.asarray(setup.estimate.origins["ultimate"], dtype=float).sum()
    latest = np.nansum(grid["cum"][np.arange(grid["n_w"]), grid["latest_dev"]])
    drawn = _run(setup, n=20_000, seed=1)
    assert not drawn.tail_fallback.any()
    offset = (latest + drawn.ibnr.sum(axis=1).mean() - central) / (central - latest)
    assert abs(offset) < TAILED_MEAN_BAND, offset


@pytest.mark.parametrize("name", ["raa", "genins", "mw2014", "abc"])
def test_the_refit_factors_are_the_link_factors_of_one_triangle(name):
    """On the real triangle the vectorised averages give link_factors' numbers:
    the volume and regression averages bit for bit (the same sums in the same
    order), the simple average and the median to rounding."""
    grid = grid_of(PUBLIC[name])
    cum = np.nan_to_num(grid["cum"])[None]
    selection = select_links(
        grid["cum"], grid["obs_mask"], grid["origin_periods"], 12, LinkRules(), grid["n_d"] - 1
    )
    for average in AVERAGES:
        expected, _ = link_factors(selection, average)
        got = link_factors_many(cum[:, :, :-1], cum[:, :, 1:], selection.used, average)[0]
        if average in ("volume", "regression"):
            assert got.tobytes() == expected.tobytes(), average
        else:
            np.testing.assert_allclose(got, expected, rtol=1e-15)


def test_the_refit_uses_the_position_rules_decided_on_the_real_triangle():
    """The refit keeps the ratios the window, the explicit exclusions and the
    excluded valuations keep, and every ratio the trims and bounds removed. The
    expected pairs are selected with the position rules written out, not with
    position_rules itself, so a field it drops is seen. Mutation: leave
    exclude_valuations out of position_rules; this fails."""
    valuation = (dt.date(1988, 12, 31),)
    candidate = ConventionalCandidate(
        "cl",
        history_periods=6,
        exclude=((dt.date(1983, 1, 1), 24),),
        exclude_valuations=valuation,
        drop_high=1,
        drop_below=1.05,
        exhausted_exclusions="keep",
    )
    setup = setup_of(RAA, candidate)

    def used(rules: LinkRules) -> np.ndarray:
        return select_links(RAA["cum"], RAA["obs_mask"], RAA["origin_periods"], 12, rules, 9).used

    written = dict(
        history_periods=6, exclude=((dt.date(1983, 1, 1), 24),), exhausted_exclusions="keep"
    )
    positions = used(LinkRules(**written, exclude_valuations=valuation))
    assert np.array_equal(setup.projection.keep, positions)
    assert np.array_equal(positions, used(position_rules(candidate.link_rules)))
    assert (positions & ~used(candidate.link_rules)).any()  # trims removed ratios the refit keeps
    assert (used(LinkRules(**written)) & ~positions).any()  # the valuation left ratios out


def test_the_refit_keeps_the_zero_rule_decided_on_the_real_triangle():
    """Under zero_cells='missing' (the front door's default) a link ratio out of
    or into a zero cumulative is left out but keeps its place in the history
    window, so the window reaches one origin less far back than under
    'observed'. The refit keeps the 'missing' pairs. Mutation: leave zero_cells
    out of position_rules, so the refit falls back to 'observed'; this fails."""
    rows = [[o, d, 0.0 if (o, d) == (1988, 12) else v] for o, d, v in PUBLIC["raa"]]
    grid = grid_of(rows)
    candidate = ConventionalCandidate("cl", history_periods=2, zero_cells="missing")
    setup = setup_of(grid, candidate)

    def used(zero_cells: str) -> np.ndarray:
        rules = LinkRules(history_periods=2, zero_cells=zero_cells)
        periods = grid["origin_periods"]
        return select_links(grid["cum"], grid["obs_mask"], periods, 12, rules, 9).used

    assert not np.array_equal(used("missing"), used("observed"))
    assert np.array_equal(setup.projection.keep, used("missing"))


def test_the_window_is_not_recomputed_on_simulated_triangles_under_observed():
    """Under zero_cells='observed' the window counts only the ratios in use, so
    an origin with a zero cumulative gives up its place. A simulated triangle
    has no zero there; recomputing the window on it would move the window.
    Mutation: select the refit's pairs on each simulated triangle; the factors
    move off the central ones."""
    matrix = odp_matrix()
    matrix[5, 0] = 0.0  # 12-month cumulative of zero: undefined ratio 12 -> 24
    grid = grid_of(rows_of(matrix))
    candidate = ConventionalCandidate("cl", history_periods=3, zero_cells="observed")
    setup = setup_of(grid, candidate)
    # the window at 12 months is origins 3, 4 and 6: 5 gave up its place
    assert np.flatnonzero(setup.projection.keep[:, 0]).tolist() == [3, 4, 6]
    # a simulated triangle with an amount where the zero was
    pseudo = np.nan_to_num(grid["cum"])[None].copy()
    pseudo[0, 5, 0] = 50.0
    found = future_cell_means(setup.projection, pseudo)
    assert found.factors[0].tobytes() == setup.estimate.factors.tobytes()


def test_a_link_with_no_positive_volume_takes_factor_one_and_is_counted():
    setup = setup_of(ODP)
    pseudo = np.repeat(fitted_triangle(setup.boot), 3, axis=0)
    rows = np.flatnonzero(setup.projection.keep[:, 0])
    pseudo[1, rows, 0] = -1.0  # the 12-month volume of draw 1 is negative
    found = future_cell_means(setup.projection, pseudo)
    assert found.unit.tolist() == [False, True, False]
    assert found.factors[1, 0] == 1.0


@pytest.mark.parametrize(("average", "reducer"), [("simple", np.mean), ("median", np.median)])
def test_the_simple_and_median_refits_leave_out_a_non_positive_earlier_amount(average, reducer):
    """The simple average and the median are of link ratios, and an earlier
    amount of zero or less gives none, so that pair is left out of that draw's
    average only; with every earlier amount negative the link takes 1.0 and the
    draw is counted. Mutation: keep every pair; the negative ratio enters."""
    setup = setup_of(ODP, ConventionalCandidate("cl", average=average))
    pseudo = np.repeat(fitted_triangle(setup.boot), 3, axis=0)
    rows = np.flatnonzero(setup.projection.keep[:, 0])
    pseudo[1, rows[0], 0] = -1.0  # one negative 12-month amount in draw 1
    pseudo[2, rows, 0] = -1.0  # every one in draw 2
    found = future_cell_means(setup.projection, pseudo)
    rest = pseudo[1, rows[1:], 1] / pseudo[1, rows[1:], 0]
    assert found.factors[1, 0] == pytest.approx(float(reducer(rest)), rel=1e-14)
    assert found.factors[2, 0] == 1.0
    assert found.unit.tolist() == [False, False, True]


def test_the_value_rules_act_once_so_the_mean_stays_near_the_central_estimate():
    """Measured on chainladder-python: trimming each simulated triangle again
    puts the mean 22% below the central estimate on raa. Here the trim acts on
    the central fit and the pool only. Mutation: refit with the full rules;
    the mean falls about 20% below."""
    candidate = ConventionalCandidate(
        "cl", drop_high=1, trim_ties="volume", exhausted_exclusions="keep"
    )
    setup = setup_of(RAA, candidate, adjustment="hat", pool="centred")
    central = float((setup.estimate.origins["ultimate"] - setup.estimate.origins["latest"]).sum())
    drawn = _run(setup, n=5000, seed=42)
    mean = drawn.ibnr.sum(axis=1).mean()
    assert abs(mean / central - 1) < 0.05


def test_a_constant_tail_enters_every_refit_as_one_more_future_cell():
    tail = TailSpec("constant", factor=1.05)
    setup = setup_of(RAA, ConventionalCandidate("cl", tail=tail))
    found = future_cell_means(setup.projection, fitted_triangle(setup.boot))
    assert found.means.shape == (1, 10, 11)
    assert found.tail_factor.tolist() == [1.05]
    oldest = setup.estimate.origins
    assert found.means[0, 0, 10] == pytest.approx(
        oldest["ultimate"][0] - oldest["latest"][0], rel=1e-12
    )
    np.testing.assert_allclose(found.means[0].sum(axis=1), oldest["reserve"], rtol=1e-12)


def test_a_curve_is_refitted_on_every_draw_and_falls_back_when_it_fails():
    """The exponential tail's factor varies from draw to draw, as
    chainladder-python's does (1.0001 to 1.0631 on raa). A draw whose own
    factors cannot carry the curve uses the central fit's curve and is
    counted. Mutation: apply the central curve to every draw; the tail
    factors do not vary."""
    tail = TailSpec("exponential")
    setup = setup_of(RAA, ConventionalCandidate("cl", tail=tail), adjustment="hat", pool="centred")
    rng = np.random.default_rng(1)
    index = np.floor(rng.random((300, 10, 10)) * setup.boot.pool.size).astype(int)
    found = future_cell_means(setup.projection, pseudo_of(setup.boot, index))
    assert found.tail_factor.std() > 1e-3
    assert not found.tail_fallback.any()
    # a draw developing by exactly 1.0 has no factor above 1.00001 to fit
    flat = np.repeat(fitted_triangle(setup.boot), 2, axis=0)
    flat[1] = np.where(setup.boot.obs_mask, flat[1][:, :1], 0.0)
    found = future_cell_means(setup.projection, flat)
    assert found.tail_fallback.tolist() == [False, True]
    assert found.tail_factor[1] == float(setup.projection.central_tail.tail_factor)


def test_a_fallback_takes_the_central_curve_from_the_attachment_age():
    """With the curve attached at 84 months it replaces the factors from 84 on
    as well as giving the tail factor; a draw that falls back takes the central
    curve's factors there, and keeps its own before 84. Mutation: copy only the
    tail factor; the draw keeps factors of 1.0 from 84 on."""
    tail = TailSpec("exponential", attach_lag=84)
    setup = setup_of(RAA, ConventionalCandidate("cl", tail=tail), adjustment="hat", pool="centred")
    central = setup.projection.central_tail
    k = central.attach_index
    assert k == 6 and (central.factors[k:9] > 1).all()
    flat = np.repeat(fitted_triangle(setup.boot), 2, axis=0)
    flat[1] = np.where(setup.boot.obs_mask, flat[1][:, :1], 0.0)
    found = future_cell_means(setup.projection, flat)
    assert found.tail_fallback.tolist() == [False, True]
    assert found.factors[1, k:].tobytes() == np.asarray(central.factors[k:9]).tobytes()
    assert (found.factors[1, :k] == 1.0).all()


def test_a_drawn_run_counts_the_tail_fallbacks():
    tail = TailSpec("exponential", fit_lags=(60, None))
    setup = setup_of(RAA, ConventionalCandidate("cl", tail=tail), adjustment="hat", pool="centred")
    drawn = _run(setup, n=500, seed=6)
    assert 0 < drawn.tail_fallback.sum() < 500
    assert np.isfinite(drawn.ibnr).all()


# -- 5. chainladder-python on shared random numbers ---------------------------------


def _chainladder_noise(means, mask, phi, seed) -> np.ndarray:
    """chainladder-python's process noise on these means: ``RandomState(seed + 1)``
    gamma over a ``(n, 1, n_w, n_d + 2)`` array, NaN on the observed cells and
    the two columns past the last age, which still take random numbers."""
    n, n_w, _ = means.shape
    n_d = mask.shape[1]
    lower = np.full((n, n_w, n_d + 2), np.nan)
    lower[:, :, :n_d] = np.where(mask[None], np.nan, means[:, :, :n_d])
    rs = np.random.RandomState(None if not seed else seed + 1)
    noisy = rs.gamma(np.abs(lower[:, None]) / phi, phi)[:, 0] * np.sign(np.nan_to_num(lower))
    return np.nansum(noisy, axis=2)


CHAINLADDER_SAMPLES = ("raa", "genins", "ukmotor", "abc", "mw2014")


@pytest.mark.tieout
@pytest.mark.parametrize("name", CHAINLADDER_SAMPLES)
@pytest.mark.parametrize("hat", [True, False])
def test_chainladder_draw_for_draw(name, hat):
    """Fed chainladder-python's residual stream, the simulated triangles, the
    scale and the refitted ultimates are chainladder's, and with its gamma
    layout applied to the future-cell means so is every IBNR draw. genins
    without the hat is the named exception (next test). Mutation: project from
    the actual latest diagonal instead of the simulated one; this fails."""
    cl = pytest.importorskip("chainladder")
    if name == "genins" and not hat:
        pytest.skip("chainladder resamples a rounding residual here; see the next test")
    tri = cl.load_sample(name)
    grid = grid_of(PUBLIC[name])
    setup = setup_of(grid, adjustment="hat" if hat else "none", pool="centred")
    boot = setup.boot
    n, seed = 200, 42
    sampler = cl.BootstrapODPSample(n_sims=n, random_state=seed, hat_adj=hat).fit(tri)
    assert boot.phi == pytest.approx(sampler.scale_, rel=1e-12)
    index = np.random.RandomState(seed).randint(0, boot.pool.size, size=(n, boot.n_w, boot.n_d))
    pseudo = pseudo_of(boot, index)
    theirs = np.nan_to_num(sampler.resampled_triangles_.values[:, 0])
    scale = np.abs(theirs).max()
    assert np.abs(np.where(boot.obs_mask, pseudo - theirs, 0)).max() < 1e-12 * scale
    found = future_cell_means(setup.projection, pseudo)
    ultimate = cl.Chainladder().fit(sampler.resampled_triangles_).ultimate_.values[:, 0, :, 0]
    np.testing.assert_allclose(found.ultimate, ultimate, rtol=1e-12)
    ibnr = cl.Chainladder().fit(sampler.transform(tri)).ibnr_.values[:, 0, :, 0]
    ours = _chainladder_noise(found.means, boot.obs_mask, boot.phi, seed)
    np.testing.assert_allclose(ours, np.nan_to_num(ibnr), rtol=1e-9, atol=1e-9 * scale)


@pytest.mark.tieout
def test_chainladder_resamples_a_rounding_residual_on_genins_without_the_hat():
    """Named difference: chainladder-python's pool has 54 residuals, ours 53."""
    cl = pytest.importorskip("chainladder")
    boot = setup_of(GENINS, adjustment="none", pool="centred").boot
    sampler = cl.BootstrapODPSample(n_sims=5, random_state=1, hat_adj=False).fit(
        cl.load_sample("genins")
    )
    theirs = sampler.resampled_triangles_  # built from their pool of 54
    assert theirs is not None and boot.pool.size == 53
    assert boot.phi == pytest.approx(sampler.scale_, rel=1e-12)


@pytest.mark.tieout
@pytest.mark.parametrize(
    "label",
    ["bornhuetter_ferguson", "benktander", "cape_cod", "cape_cod_decay", "bf_sigma", "cc_sigma"],
)
def test_chainladder_draw_for_draw_for_the_a_priori_methods(label):
    """Bornhuetter-Ferguson, Benktander (``n_iters`` 2), Cape Cod (``decay`` 1, and
    0.5 with ``n_iters`` 2) and ``apriori_sigma``, each refitted on
    chainladder's simulated triangles with its noise: equal to 1e-9. The
    injected multipliers (one per draw, shared by every origin) reproduce
    ``apriori_sigma``."""
    cl = pytest.importorskip("chainladder")
    tri = cl.load_sample("raa")
    n, seed = 300, 42
    amounts = np.linspace(40000, 58000, 10)
    weight = cl.Chainladder().fit(tri).ultimate_ * 0 + amounts[None, None, :, None]
    premium = {dt.date(1981 + i, 1, 1): float(a) for i, a in enumerate(amounts)}
    cases = {
        "bornhuetter_ferguson": (cl.BornhuetterFerguson(apriori=0.8), _candidate("bf"), None),
        "benktander": (
            cl.Benktander(apriori=0.8, n_iters=2),
            ConventionalCandidate("bf", expected_loss_ratio=0.8, n_iters=2),
            None,
        ),
        "cape_cod": (cl.CapeCod(), ConventionalCandidate("gcc", decay=1.0), None),
        "cape_cod_decay": (
            cl.CapeCod(decay=0.5, n_iters=2),
            ConventionalCandidate("gcc", decay=0.5, n_iters=2),
            None,
        ),
        "bf_sigma": (
            cl.BornhuetterFerguson(apriori=0.8, apriori_sigma=0.1, random_state=42),
            _candidate("bf"),
            np.random.RandomState(42).normal(0.8, 0.1, n) / 0.8,
        ),
        "cc_sigma": (
            cl.CapeCod(apriori_sigma=0.1, random_state=7),
            ConventionalCandidate("gcc", decay=1.0),
            np.random.RandomState(7).normal(1.0, 0.1, n),
        ),
    }
    estimator, candidate, multiplier = cases[label]
    sampler = cl.BootstrapODPSample(n_sims=n, random_state=seed).fit(tri)
    ibnr = estimator.fit(sampler.transform(tri), sample_weight=weight).ibnr_.values[:, 0, :, 0]
    setup = setup_of(RAA, candidate, premium=premium, adjustment="hat", pool="centred")
    index = np.random.RandomState(seed).randint(0, setup.boot.pool.size, size=(n, 10, 10))
    found = future_cell_means(
        setup.projection, pseudo_of(setup.boot, index), prior_multiplier=multiplier
    )
    ours = _chainladder_noise(found.means, setup.boot.obs_mask, setup.boot.phi, seed)
    np.testing.assert_allclose(ours, np.nan_to_num(ibnr), rtol=1e-9, atol=1e-6)


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_chainladder_draw_for_draw_with_development_options(name):
    """The combination the one specification reaches in chainladder-python: the
    sampler with ``n_periods=5, drop_high=True`` and the refit with
    ``n_periods=5``, which is ``history_periods=5, drop_high=1`` here. And an
    excluded link from the first age, which zeroes two residual cells."""
    cl = pytest.importorskip("chainladder")
    tri = cl.load_sample(name)
    grid = grid_of(PUBLIC[name])
    first = str(grid["origin_periods"][0])[:4]
    cases = [
        (
            {"n_periods": 5, "drop_high": True},
            {"n_periods": 5},
            ConventionalCandidate(
                "cl",
                history_periods=5,
                drop_high=1,
                trim_ties="volume",
                exhausted_exclusions="keep",
            ),
        ),
        (
            {"drop": [(first, 12)]},
            {"drop": [(first, 12)]},
            ConventionalCandidate("cl", exclude=((dt.date(int(first), 1, 1), 12),)),
        ),
    ]
    n, seed = 200, 7
    for sampler_options, refit_options, candidate in cases:
        sampler = cl.BootstrapODPSample(n_sims=n, random_state=seed, **sampler_options).fit(tri)
        setup = setup_of(grid, candidate, adjustment="hat", pool="centred")
        assert setup.boot.phi == pytest.approx(sampler.scale_, rel=1e-12)
        index = np.random.RandomState(seed).randint(
            0, setup.boot.pool.size, size=(n, grid["n_w"], grid["n_d"])
        )
        found = future_cell_means(setup.projection, pseudo_of(setup.boot, index))
        refit = cl.Development(**refit_options).fit_transform(sampler.resampled_triangles_)
        ultimate = cl.Chainladder().fit(refit).ultimate_.values[:, 0, :, 0]
        np.testing.assert_allclose(found.ultimate, ultimate, rtol=1e-12)


# -- 6. Monte Carlo agreement ------------------------------------------------------


def _agrees(ours: np.ndarray, mean: float, sd: float, n: int) -> None:
    """Means within 4 combined standard errors, standard deviations within 3%."""
    total = ours.sum(axis=1)
    se = np.sqrt(total.var(ddof=1) / total.size + sd**2 / n)
    assert abs(total.mean() - mean) < 4 * se, (total.mean(), mean, se)
    assert total.std(ddof=1) == pytest.approx(sd, rel=0.03)


@pytest.mark.parametrize(
    "run", R_BOOT["runs"], ids=lambda r: f"{r['triangle']}-{r['process']}-{r['seed']}"
)
def test_r_bootchainladder_within_monte_carlo_error(run):
    """R's conventions (degrees of freedom, every residual, not centred) at
    20,000 draws, two seeds each, against R's frozen runs; od_poisson against
    R's od.pois. Measured: every mean within 2 standard errors and every sd
    within 1%."""
    grid = RAA if run["triangle"] == "RAA" else GENINS
    setup = setup_of(grid, adjustment="dof", pool="all")
    process = "gamma" if run["process"] == "gamma" else "od_poisson"
    drawn = _run(setup, n=20_000, seed=run["seed"], process=process)
    _agrees(drawn.ibnr, run["mean"], run["sd"], run["n_draws"])
    n = run["n_draws"]
    for i in range(1, drawn.ibnr.shape[1]):  # the oldest origin has nothing left
        mine = drawn.ibnr[:, i]
        se = np.sqrt(mine.var(ddof=1) / mine.size + run["origin_sd"][i] ** 2 / n)
        assert abs(mine.mean() - run["origin_mean"][i]) < 4 * se, i
        assert mine.std(ddof=1) == pytest.approx(run["origin_sd"][i], rel=0.05), i


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
def test_chainladder_within_monte_carlo_error(name):
    """chainladder-python's own run and ibnr's (hat, centred) at 20,000 draws."""
    cl = pytest.importorskip("chainladder")
    tri = cl.load_sample(name)
    x = cl.BootstrapODPSample(n_sims=20_000, random_state=1).fit_transform(tri)
    theirs = np.nan_to_num(cl.Chainladder().fit(x).ibnr_.values[:, 0, :, 0]).sum(axis=1)
    setup = setup_of(grid_of(PUBLIC[name]), adjustment="hat", pool="centred")
    drawn = _run(setup, n=20_000, seed=1)
    _agrees(drawn.ibnr, theirs.mean(), theirs.std(ddof=1), theirs.size)


def test_the_reserving_apps_workbook_within_monte_carlo_error():
    """The Reserving app's cached bootstrap (chainladder-python, 2,000 draws, seed
    42) on its own paid triangle, against ibnr's at 20,000 draws."""
    grid = grid_of(WORKBOOK["cells"])
    cached = WORKBOOK["cached"]
    setup = setup_of(grid, adjustment="hat", pool="centred")
    central = float((setup.estimate.origins["ultimate"] - setup.estimate.origins["latest"]).sum())
    assert central == pytest.approx(cached["central_ibnr"], rel=1e-12)
    drawn = _run(setup, n=20_000, seed=42)
    _agrees(drawn.ibnr, cached["mean_ibnr"], cached["sd_ibnr"], cached["n_draws"])
