"""The second tlrn study's evaluator: scoring windows, benchmarks, statistics and the verdict.

``scripts/tlrn_study2_eval.py`` scores what ``scripts/tlrn_study_colab.py --study factor_head``
saved. No network is fitted here. The tests drive the pieces whose mistakes would quietly
change an answer, each with small inputs worked out another way:

* the scoring window of each valuation date (the companion's 14, 15, 13 and 45 cells per pair);
* the benchmarks read per cell from fits that saw only the cells known at the date (a leak of
  later cells is the mistake that matters, so one test changes the later cells and requires
  the benchmark not to move);
* the bootstrap, with a company's valuation dates resampled together;
* the verdict, one case for each condition and each word;
* the checks that refuse a fit scored on other cells than the evaluator's own;
* the whole path from fits on disk to ``verdicts.json``, on a synthetic cohort.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import tlrn_colab  # noqa: E402
import tlrn_study2_eval as ev  # noqa: E402
import tlrn_study_colab as study  # noqa: E402
import tlrn_study_eval as first  # noqa: E402

N = 10
AY0 = 2000
V = len(ev.VALUATIONS)
LAST_I = ev.VALUATIONS.index(ev.LAST)

# -- a synthetic cohort ------------------------------------------------------------------


def synthetic_cells(
    companies=("c1", "c2", "c3"), lines=("L1", "L2"), *, seed=0, later=None, after=N
):
    """Complete 10 x 10 squares of cumulative paid in the cell layout, with noise.

    ``later`` multiplies every cell on a calendar diagonal past ``after`` (default 10, the
    last known one), so a test can change only the development a valuation date has not
    seen.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ci, company in enumerate(companies):
        for li, line in enumerate(lines):
            size = 500.0 * (ci + 1) * (li + 1)
            for w in range(N):
                level = size * (1 + 0.05 * w) * rng.uniform(0.9, 1.1)
                cum = 0.0
                for d in range(1, N + 1):
                    cum += level * 0.6 ** (d - 1) * 0.4 * rng.uniform(0.9, 1.1)
                    shown = cum
                    if later is not None and w + d > after:
                        shown = cum * later
                    rows.append(
                        {
                            "company_code": company,
                            "line_of_business": line,
                            "statement_year": AY0 + w + d - 1,
                            "accident_year": AY0 + w,
                            "dev_age": d,
                            "paid": shown,
                            "incurred": shown * 1.1,
                            "bulk": 0.0,
                            "case": shown * 0.1,
                            "premium": 5000.0 * (ci + 1),
                        }
                    )
    return study._tidy(pd.DataFrame(rows))


def cohort_of(cells, name="A", *, data_hash="d" * 16):
    pairs = sorted(
        {(c, ln) for c, ln in zip(cells["company_code"], cells["line_of_business"], strict=True)}
    )
    actual = study.actual_reserves(cells, pairs, as_of_year=AY0 + N - 1)
    board = sorted({c for c, _ in pairs})
    return SimpleNamespace(
        name=name,
        cells=cells,
        pairs=pairs,
        board=board,
        companies=board,
        actual=actual,
        as_of=study.AS_OF[name],
        pairs_hash="p" * 16,
        data_hash=data_hash,
        description=f"a synthetic cohort {name}",
    )


def volume_weighted_completion(cum_known: np.ndarray, known: np.ndarray) -> np.ndarray:
    """A chain ladder written out directly: volume-weighted factors, latest amount rolled on.

    ``cum_known`` is ``(n_w, n_d)`` cumulative paid and ``known`` the booleans of the cells
    the date has seen. Written for the test, not shared with the evaluator.
    """
    n_w, n_d = cum_known.shape
    f = np.ones(n_d - 1)
    for d in range(n_d - 1):
        both = known[:, d] & known[:, d + 1]
        if both.any():
            f[d] = cum_known[both, d + 1].sum() / cum_known[both, d].sum()
    full = np.where(known, cum_known, np.nan)
    for w in range(n_w):
        seen = np.nonzero(known[w])[0]
        if seen.size == 0:
            continue
        for d in range(seen[-1], n_d - 1):
            full[w, d + 1] = full[w, d] * f[d]
    return full


# -- the scoring window ------------------------------------------------------------------


def test_the_valuation_windows_score_the_companions_cells_per_pair():
    got = [int(ev.scored_mask(v, N, N).sum()) for v in ev.VALUATIONS]
    assert got == [14, 15, 13, 45]
    # over the companion's 212 pairs these are the cells column of his rolling table
    assert [212 * g for g in got] == [2968, 3180, 2756, 9540]


def test_the_window_matches_a_cell_by_cell_definition():
    for v in ev.VALUATIONS:
        mask = ev.scored_mask(v, N, N)
        for w in range(1, N + 1):  # 1-based origin
            for d in range(1, N + 1):  # 1-based lag
                diagonal = (w - 1) + (d - 1) + 1  # calendar diagonal of the cell, 1-based
                elapsed = w <= v  # the origin has begun by the valuation
                later = diagonal > v
                if v < ev.LAST:
                    expected = elapsed and later and diagonal <= ev.LAST and d <= v
                else:
                    expected = elapsed and later
                assert mask[w - 1, d - 1] == expected, (v, w, d)


def test_the_window_at_the_last_valuation_is_the_unobserved_triangle():
    mask = ev.scored_mask(ev.LAST, N, N)
    w, d = np.nonzero(mask)
    assert (w + d + 1 > N).all()
    assert mask.sum() == N * (N - 1) // 2


def test_the_scored_cells_never_include_a_known_cell():
    for v in ev.VALUATIONS:
        w, d = np.nonzero(ev.scored_mask(v, N, N))
        assert ((w + d + 1) > v).all()


# -- what the cells are worth -------------------------------------------------------------


def test_the_truth_table_sums_the_observed_increments_over_the_window():
    cells = synthetic_cells(("c1",), ("L1", "L2"))
    cohort = cohort_of(cells)
    truth = ev.truth_table(cohort).set_index(["valuation", "company_code"])
    squares, *_ = ev.paid_squares(cells, cohort.pairs)
    for v in ev.VALUATIONS:
        want = 0.0
        for cum in squares.values():
            for w in range(N):
                for d in range(N):
                    if ev.scored_mask(v, N, N)[w, d]:
                        want += cum[w, d] - (cum[w, d - 1] if d else 0.0)
        assert truth.loc[(v, "c1"), "actual"] == pytest.approx(want)
        assert truth.loc[(v, "c1"), "n_cells"] == 2 * ev.scored_mask(v, N, N).sum()
    # at the last valuation it is the cohort's realised reserve
    assert truth.loc[(ev.LAST, "c1"), "actual"] == pytest.approx(
        cohort.actual.loc["c1", "actual_reserve"]
    )


def test_the_truth_table_refuses_cells_that_disagree_with_the_realised_reserve():
    cohort = cohort_of(synthetic_cells(("c1",), ("L1", "L2")))
    cohort.actual.loc["c1", "actual_reserve"] *= 1.5
    with pytest.raises(SystemExit, match="realised reserve"):
        ev.truth_table(cohort)


# -- the benchmarks -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmarks():
    cells = synthetic_cells(("c1", "c2"), ("L1", "L2"), seed=3)
    cohort = cohort_of(cells)
    return cohort, ev.benchmark_cells(cohort, log=lambda *_: None)


def test_mack_at_each_valuation_is_a_chain_ladder_on_the_cells_known_then(benchmarks):
    cohort, bench = benchmarks
    squares, *_ = ev.paid_squares(cohort.cells, cohort.pairs)
    for v in ev.VALUATIONS:
        mask = ev.scored_mask(v, N, N)
        known = (np.arange(N)[:, None] + np.arange(N)[None, :] + 1) <= v
        for company in cohort.board:
            total = 0.0
            for (c, _), cum in squares.items():
                if c != company:
                    continue
                visible = known[:v, :v]
                full = volume_weighted_completion(np.where(known, cum, np.nan)[:v, :v], visible)
                grid = np.full((N, N), np.nan)
                grid[:v, :v] = full
                total += np.diff(grid, axis=1, prepend=0.0)[mask].sum()
            got = bench[(bench["valuation"] == v) & (bench["company_code"] == company)]["mack"]
            assert got.iloc[0] == pytest.approx(total, rel=1e-9)


@pytest.mark.parametrize("after", [6, 7, 8, 10])
def test_the_benchmarks_do_not_see_cells_past_the_valuation(after):
    """Changing only the diagonals past ``after`` moves the benchmarks at later valuations only.

    A valuation at or before ``after`` has not seen the changed cells, so both benchmarks
    there must not move (a leak of a later diagonal into a fit is the mistake that matters,
    and it needs a test of the multivariate chain ladder as well as the chain ladder). A
    valuation past ``after`` has seen them, which shows the test can fail.
    """
    base = synthetic_cells(("c1", "c2"), ("L1", "L2"), seed=5)
    moved = synthetic_cells(("c1", "c2"), ("L1", "L2"), seed=5, later=1.7, after=after)
    a = ev.benchmark_cells(cohort_of(base), log=lambda *_: None)
    b = ev.benchmark_cells(cohort_of(moved), log=lambda *_: None)
    seen = a["valuation"] <= after
    for column in ("mack", "mcl"):
        np.testing.assert_allclose(a.loc[seen, column], b.loc[seen, column], rtol=1e-12)
        if (~seen).any():
            assert not np.allclose(a.loc[~seen, column], b.loc[~seen, column])
    # the actual is the development the date has not seen: with every changed cell past the
    # last known diagonal it moves only at the last valuation
    if after == ev.LAST:
        early = a["valuation"] < ev.LAST
        np.testing.assert_allclose(a.loc[early, "actual"], b.loc[early, "actual"])
        assert not np.allclose(a.loc[~early, "actual"], b.loc[~early, "actual"])


def test_the_benchmarks_at_the_last_valuation_are_the_whole_run_off(benchmarks):
    from ibnr import gallery

    cohort, bench = benchmarks
    tri = study.triangle_from_cells(cohort.cells, cohort.pairs)
    e = tri.expr
    company = "c1"
    entry = gallery.fit(
        "mcl",
        tri.filter(e.company_code == company),
        loss_field="paid_loss",
        as_of=ev.valuation_date(AY0, ev.LAST),
    )
    anchor = float(cohort.actual.loc[company, "anchor"])
    ultimate = float(entry.point()["point"].iloc[-1])
    got = bench[(bench["valuation"] == ev.LAST) & (bench["company_code"] == company)]
    assert got["mcl"].iloc[0] == pytest.approx(ultimate - anchor, rel=1e-9)


def test_a_refused_company_is_nan_for_that_benchmark_only(benchmarks):
    cohort, _ = benchmarks
    zero = cohort.cells.copy()
    mask = (zero["company_code"] == "c1") & (zero["line_of_business"] == "L1")
    zero.loc[mask & (zero["accident_year"] == AY0), "paid"] = 0.0  # a zero first row
    out = ev.benchmark_cells(cohort_of(zero), log=lambda *_: None)
    assert len(out) == V * 2
    assert out["company_code"].nunique() == 2


def test_benchmark_fits_are_cached_by_cohort_and_data_hash(tmp_path, monkeypatch):
    cohort = cohort_of(synthetic_cells(("c1", "c2"), ("L1", "L2")))
    calls = []

    def fake(c, valuations=ev.VALUATIONS, *, log=print):
        calls.append(c.name)
        rows = [
            {"valuation": v, "company_code": co, "mack": 1.0, "mcl": 2.0, "n_cells": 3, "actual": 4}
            for v in valuations
            for co in c.board
        ]
        return pd.DataFrame(rows)

    monkeypatch.setattr(ev, "benchmark_cells", fake)
    ev.cached_benchmark_cells(cohort, tmp_path, log=lambda *_: None)
    ev.cached_benchmark_cells(cohort, tmp_path, log=lambda *_: None)
    assert calls == ["A"]
    # other cells are another cache entry, and no cache directory means always compute
    ev.cached_benchmark_cells(
        cohort_of(synthetic_cells(), data_hash="e" * 16), tmp_path, log=lambda *_: None
    )
    ev.cached_benchmark_cells(cohort, None, log=lambda *_: None)
    assert calls == ["A", "A", "A"]


def test_a_cache_for_other_companies_is_not_trusted(tmp_path, monkeypatch):
    cohort = cohort_of(synthetic_cells(("c1", "c2"), ("L1", "L2")))
    stale = pd.DataFrame(
        [
            {"valuation": v, "company_code": "c1", "mack": 1, "mcl": 1, "n_cells": 1, "actual": 1}
            for v in ev.VALUATIONS
        ]
    )
    stale.to_csv(tmp_path / f"benchmarks2_A_{cohort.data_hash}.csv", index=False)
    seen = []

    def fake(c, valuations=ev.VALUATIONS, *, log=print):
        seen.append(1)
        return stale.assign(company_code="c2")

    monkeypatch.setattr(ev, "benchmark_cells", fake)
    ev.cached_benchmark_cells(cohort, tmp_path, log=lambda *_: None)
    assert seen == [1]


# -- statistics ---------------------------------------------------------------------------


def test_pool_ape_by_valuation_is_the_sum_of_errors_over_the_sum_of_actuals():
    actual = np.array([[10.0, 20.0, 30.0]] * V)
    pred = actual * np.array([[1.1, 0.9, 1.0]] * V)
    got = ev.pool_ape_by_valuation(pred, actual)
    np.testing.assert_allclose(got, (1 + 2 + 0) / 60)


def test_the_difference_summary_resamples_a_companys_valuations_together():
    rng = np.random.default_rng(1)
    c = 30
    actual = rng.uniform(100, 1000, size=(V, c))
    model = actual * rng.uniform(0.8, 1.2, size=(V, c))
    bench = actual * rng.uniform(0.8, 1.2, size=(V, c))
    idx = first.bootstrap_indices(c, 2000, 7)
    got = ev.difference_summary(model, bench, actual, idx)

    def pa(p, a, rows):
        return np.abs(p[rows] - a[rows]).sum() / a[rows].sum()

    diff_last, diff_roll = [], []
    for rows in idx:
        m = [pa(model[i], actual[i], rows) for i in range(V)]
        b = [pa(bench[i], actual[i], rows) for i in range(V)]
        diff_last.append(m[LAST_I] - b[LAST_I])
        diff_roll.append(np.mean(m) - np.mean(b))
    for key, boot in (("last", diff_last), ("rolling", diff_roll)):
        assert got[key]["low"] == pytest.approx(np.quantile(boot, 0.025))
        assert got[key]["high"] == pytest.approx(np.quantile(boot, 0.975))
        assert got[key]["share_favour"] == pytest.approx(np.mean(np.array(boot) < 0))
    point = ev.pool_ape_by_valuation(model, actual) - ev.pool_ape_by_valuation(bench, actual)
    assert got["last"]["diff"] == pytest.approx(point[LAST_I])
    assert got["rolling"]["diff"] == pytest.approx(point.mean())


def test_the_rolling_statistic_is_the_plain_mean_over_the_four_valuations():
    assert ev.rolling(np.array([0.08, 0.04, 0.04, 0.04])) == pytest.approx(0.05)
    by_resample = np.array([[0.08, 0.1], [0.04, 0.2], [0.04, 0.3], [0.04, 0.4]])
    np.testing.assert_allclose(ev.rolling(by_resample, axis=0), [0.05, 0.25])
    assert ev.VALUATIONS == (6, 7, 8, 10)


def test_an_identical_forecast_has_a_zero_interval_and_a_better_one_is_wholly_below():
    rng = np.random.default_rng(2)
    c = 25
    actual = rng.uniform(100, 1000, size=(V, c))
    bench = actual * rng.uniform(0.7, 1.3, size=(V, c))
    idx = first.bootstrap_indices(c)
    same = ev.difference_summary(bench, bench, actual, idx)
    assert same["last"]["low"] == same["last"]["high"] == 0.0
    better = ev.difference_summary(actual, bench, actual, idx)
    assert better["last"]["high"] < 0 and better["rolling"]["high"] < 0
    assert better["last"]["share_favour"] == 1.0


def test_the_interval_leaves_out_resamples_with_no_pool_ape():
    diff = np.array([np.nan, -1.0, 1.0, np.nan])
    assert ev.interval(diff) == (pytest.approx(-0.95), pytest.approx(0.95))


def test_point_metrics_match_a_hand_worked_example():
    actual = np.array([100.0, 0.0, 50.0, 10.0])
    pred = np.array([110.0, 5.0, 25.0, 10.0])
    got = ev.point_metrics(pred, actual, bench=np.array([100.0, 0.0, 50.0, 12.0]))
    err = pred - actual
    assert got["pool_ape"] == pytest.approx(np.abs(err).sum() / 160)
    assert got["pool_pe"] == pytest.approx(err.sum() / 160)
    ape = np.array([10 / 100, 25 / 50, 0 / 10])  # the zero actual has no percentage error
    assert got["medape"] == pytest.approx(np.median(ape))
    assert got["mape"] == pytest.approx(ape.mean())
    assert got["mae"] == pytest.approx(np.abs(err).mean())
    # log ratio only where both are positive: companies 0, 2, 3
    assert got["n_log"] == 3
    assert got["log_ratio"] == pytest.approx(np.mean(np.abs(np.log([1.1, 0.5, 1.0]))))
    assert got["wrmse"] == pytest.approx(np.sqrt((actual * err**2).sum() / actual.sum()))
    # the benchmark's errors are 0, 0, 0 and 2 and the model's are 10, 5, 25 and 0, so the
    # model's error is the smaller at company 3 only
    assert got["share_better_than_benchmark"] == pytest.approx(1 / 4)


def test_size_quartiles_split_into_four_equal_groups_in_order_of_actual():
    actual = np.array([5.0, 1.0, 8.0, 3.0, 7.0, 2.0, 6.0, 4.0])
    q = ev.size_quartiles(actual)
    assert sorted(q.tolist()) == [1, 1, 2, 2, 3, 3, 4, 4]
    assert q[np.argmax(actual)] == 4 and q[np.argmin(actual)] == 1
    # equal actuals are split by position, not lumped
    assert sorted(ev.size_quartiles(np.ones(8)).tolist()) == [1, 1, 2, 2, 3, 3, 4, 4]


# -- the verdict --------------------------------------------------------------------------


def interval(low, high):
    return {"low": low, "high": high}


def stats_for(**changes):
    base = {
        "last": {b: interval(-0.03, -0.005) for b in ev.BENCHMARKS},
        "rolling": {b: interval(-0.02, -0.002) for b in ev.BENCHMARKS},
        "excluding": {b: interval(-0.02, 0.01) for b in ev.BENCHMARKS},
        "median_draw_last": 0.04,
        "pool_last": {"mack": 0.055, "mcl": 0.054},
    }
    base.update(changes)
    return base


def test_all_four_conditions_give_effective():
    word, conditions, reason = ev.decide(stats_for())
    assert word == "effective" and all(conditions.values())
    assert "valuation 10" in reason and "median-draw" in reason


@pytest.mark.parametrize(
    "changes, failed",
    [
        ({"last": {"mack": interval(-0.03, -0.005), "mcl": interval(-0.03, 0.001)}}, "i_"),
        (
            {"rolling": {"mack": interval(-0.03, 0.0), "mcl": interval(-0.03, -0.002)}},
            "ii_",
        ),
        ({"median_draw_last": 0.0545}, "iii_"),
        ({"excluding": {"mack": interval(0.001, 0.02), "mcl": interval(-0.02, 0.01)}}, "iv_"),
    ],
)
def test_each_condition_alone_stops_effective(changes, failed):
    word, conditions, _ = ev.decide(stats_for(**changes))
    assert word == "not distinguishable"
    assert [k for k, v in conditions.items() if not v][0].startswith(failed)
    assert sum(not v for v in conditions.values()) == 1


def test_median_draw_must_be_below_both_benchmarks_not_just_one():
    word, conditions, _ = ev.decide(stats_for(median_draw_last=0.0541))
    assert word == "not distinguishable" and not conditions["iii_median_draw_below_both"]


def test_worse_needs_an_interval_wholly_above_zero_at_the_last_valuation_or_rolling():
    above = {"mack": interval(-0.01, 0.02), "mcl": interval(0.001, 0.02)}
    assert ev.decide(stats_for(last=above))[0] == "worse"
    assert ev.decide(stats_for(rolling=above))[0] == "worse"
    # either benchmark counts, the weaker one too (the first study's rule used the stronger)
    weaker = {"mack": interval(0.001, 0.02), "mcl": interval(-0.01, 0.02)}
    assert stats_for()["pool_last"]["mack"] > stats_for()["pool_last"]["mcl"]
    assert ev.decide(stats_for(last=weaker))[0] == "worse"
    assert ev.decide(stats_for(rolling=weaker))[0] == "worse"
    # touching zero is not wholly above
    touching = {"mack": interval(-0.01, 0.02), "mcl": interval(0.0, 0.02)}
    assert ev.decide(stats_for(last=touching, rolling=touching))[0] == "not distinguishable"
    # an interval above zero in the dominant-excluded check alone does not make it worse
    ex = {"mack": interval(0.001, 0.02), "mcl": interval(0.001, 0.02)}
    assert ev.decide(stats_for(excluding=ex))[0] == "not distinguishable"


def test_an_interval_that_could_not_be_computed_is_neither_effective_nor_worse():
    nan = {b: interval(float("nan"), float("nan")) for b in ev.BENCHMARKS}
    for key in ("last", "rolling", "excluding"):
        word, _, _ = ev.decide(stats_for(**{key: nan}))
        assert word == "not distinguishable"


def test_the_pre_registered_constants():
    assert (ev.N_BOOT, ev.BOOT_SEED, ev.LEVEL) == (10_000, 2008, 0.95)
    assert ev.VALUATIONS == (6, 7, 8, 10) and ev.LAST == 10
    assert ev.BENCHMARKS == ("mack", "mcl")
    assert ev.ROWS == ("tlrn_13_fh", "tlrn_13_fh_factors")
    assert ev.VERDICTS == ("effective", "not distinguishable", "worse")
    assert "out-of-sample" in ev.STATUS["C"] and "not out-of-sample" in ev.STATUS["A"]
    doc = " ".join(ev.__doc__.split())
    for phrase in (
        "wholly below zero",
        "confirmatory of a fixed specification, not out-of-sample",
        "10,000 times",
        "dominant company",
    ):
        assert phrase in doc


# -- fits on disk -------------------------------------------------------------------------


def fit_settings(cohort, row, draw, **override):
    s = {
        "study": 2,
        "study_name": "factor_head",
        "cohort": cohort.name,
        "row": row,
        "draw": draw,
        "seed": study.draw_seed(draw),
        "smoke": False,
        "config": {"d_model": 32},
        "roles": {"incurred_field": "incurred_loss"},
        "backend": "jax",
        "as_of": cohort.as_of.isoformat(),
        "loss_field": "paid_loss",
        "publish_id": tlrn_colab.PUBLISH,
        "cohort_hash": cohort.pairs_hash,
        "data_hash": cohort.data_hash,
    }
    s.update(override)
    return s


def tables_for(cohort, predict, *, truth=None):
    """The valuations, backtest and members tables of one fit.

    ``predict(valuation_index, actual)`` gives the draw's prediction for the companies.
    """
    truth = ev.truth_table(cohort) if truth is None else truth
    frames = []
    for i, v in enumerate(ev.VALUATIONS):
        part = truth[truth["valuation"] == v].copy()
        part["predicted"] = predict(i, part["actual"].to_numpy())
        part["predicted_net"] = part["predicted"] * 1.01
        if v == ev.LAST:
            part["actual"] = np.nan
        frames.append(part)
    valuations = pd.concat(frames, ignore_index=True)[
        ["valuation", "company_code", "n_cells", "predicted", "predicted_net", "actual"]
    ]
    members = pd.DataFrame(
        {
            "valuation": np.repeat(ev.VALUATIONS, 2),
            "member": [0, 1] * V,
            "alpha": 0.5,
            "kept": True,
        }
    )
    backtest = pd.DataFrame({"unit": [0], "cutoff": [6], "horizon": [4]})
    return valuations, backtest, members


def write_fit(folder, cohort, row, draw, predict, *, smoke=False, version="0.7.4", **override):
    valuations, backtest, members = tables_for(cohort, predict)
    stem = f"{cohort.name}_{row}_d{draw:02d}{'_smoke' if smoke else ''}-ibnr{version}"
    reserve = valuations[valuations["valuation"] == ev.LAST].set_index("company_code")["predicted"]
    reserves = pd.DataFrame(
        {
            "company_code": reserve.index,
            "anchor": cohort.actual["anchor"].reindex(reserve.index).to_numpy(),
            "premium": cohort.actual["premium"].reindex(reserve.index).to_numpy(),
            "n_lines": 2,
            "reserve": reserve.to_numpy(),
        }
    )
    folder.mkdir(parents=True, exist_ok=True)
    reserves.to_csv(folder / f"{stem}_reserves.csv", index=False)
    valuations.to_csv(folder / f"{stem}_valuations.csv", index=False)
    backtest.to_csv(folder / f"{stem}_backtest.csv", index=False)
    members.to_csv(folder / f"{stem}_members.csv", index=False)
    settings = fit_settings(cohort, row, draw, smoke=smoke, **override)
    (folder / f"{stem}.json").write_text(json.dumps({"settings": settings, "run": {}}))
    return stem


@pytest.fixture(scope="module")
def cohort():
    return cohort_of(synthetic_cells(("c1", "c2", "c3", "c4"), ("L1", "L2"), seed=9))


def perfect(i, actual):
    return actual


def test_load_fits_reads_the_second_studys_fits_in_draw_order(tmp_path, cohort):
    for draw in (2, 0, 1):
        write_fit(tmp_path, cohort, ev.MAIN_ROW, draw, perfect)
    fits = ev.load_fits(tmp_path, "A", ev.MAIN_ROW, smoke=False)
    assert [f.draw for f in fits] == [0, 1, 2]
    assert fits[0].valuations.shape[0] == V * len(cohort.board)
    assert fits[0].reserves.index.name == "company_code"
    assert ev.load_fits(tmp_path, "A", ev.MAIN_ROW, smoke=True) == []
    assert ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False) == []
    # the comparison row is a different row even though its name starts the same
    assert ev.load_fits(tmp_path, "A", ev.FACTORS_ROW, smoke=False) == []


def test_load_fits_ignores_a_record_that_is_not_the_second_studys(tmp_path, cohort):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, perfect, study=1)
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 1, perfect, study_name="accident_year")
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 2, perfect)
    assert [f.draw for f in ev.load_fits(tmp_path, "A", ev.MAIN_ROW, smoke=False)] == [2]


def test_load_fits_refuses_a_seed_that_is_not_the_draws(tmp_path, cohort):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 3, perfect, seed=11)
    with pytest.raises(SystemExit, match="not draw 3's"):
        ev.load_fits(tmp_path, "A", ev.MAIN_ROW, smoke=False)


def test_load_fits_refuses_draws_fitted_with_other_settings(tmp_path, cohort):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, perfect)
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 1, perfect, config={"d_model": 64})
    with pytest.raises(SystemExit, match="different settings.*config.d_model"):
        ev.load_fits(tmp_path, "A", ev.MAIN_ROW, smoke=False)


def test_load_fits_refuses_a_fit_with_a_table_missing(tmp_path, cohort):
    stem = write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, perfect)
    (tmp_path / f"{stem}_members.csv").unlink()
    with pytest.raises(SystemExit, match="_members.csv"):
        ev.load_fits(tmp_path, "A", ev.MAIN_ROW, smoke=False)


def test_load_fits_refuses_a_draw_saved_by_two_versions(tmp_path, cohort):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, perfect, version="0.7.4")
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, perfect, version="0.7.5")
    with pytest.raises(SystemExit, match="two ibnr versions"):
        ev.load_fits(tmp_path, "A", ev.MAIN_ROW, smoke=False)


def loaded(tmp_path, cohort, row=ev.MAIN_ROW, draws=(0, 1), predict=perfect, **override):
    for draw in draws:
        write_fit(tmp_path, cohort, row, draw, predict, **override)
    return ev.load_fits(tmp_path, cohort.name, row, smoke=False)


def test_a_fit_for_this_cohort_passes_the_cohort_check_with_no_note(tmp_path, cohort):
    fits = loaded(tmp_path, cohort)
    assert ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits) == ""


@pytest.mark.parametrize(
    "override, key",
    [
        ({"cohort_hash": "x" * 16}, "cohort_hash"),
        ({"publish_id": "other"}, "publish_id"),
        ({"study": 1}, "study"),
        ({"study_name": "accident_year"}, "study_name"),
        ({"cohort": "B"}, "cohort"),
        ({"as_of": "2001-12-31"}, "as_of"),
    ],
)
def test_a_fit_made_for_another_cohort_or_study_is_refused(tmp_path, cohort, override, key):
    fits = loaded(tmp_path, cohort, draws=(0,))
    fits[0].settings.update(override)
    with pytest.raises(SystemExit, match=f"different cohort or data.*{key}"):
        ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits)


def test_a_different_cell_hash_is_accepted_when_the_plumbing_agrees(tmp_path, cohort):
    fits = loaded(tmp_path, cohort, draws=(0,), data_hash="f" * 16)
    note = ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits)
    assert "data hash differs" in note and "so the fits are scored" in note
    with pytest.raises(SystemExit, match="strict-data-hash"):
        ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits, strict=True)


def test_a_different_cell_hash_with_other_anchors_is_refused(tmp_path, cohort):
    fits = loaded(tmp_path, cohort, draws=(0,), data_hash="f" * 16)
    fits[0].reserves.loc["c2", "anchor"] += 5.0
    with pytest.raises(SystemExit, match="anchor for company c2"):
        ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits)
    fits = loaded(tmp_path / "b", cohort, draws=(0,), data_hash="f" * 16)
    fits[0].reserves.loc["c3", "n_lines"] = 1
    with pytest.raises(SystemExit, match="1 lines for company c3"):
        ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits)
    fits = loaded(tmp_path / "c", cohort, draws=(0,), data_hash="f" * 16)
    fits[0].reserves = fits[0].reserves.drop("c4")
    with pytest.raises(SystemExit, match="companies differ"):
        ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits)


def test_the_saved_tables_must_agree_with_the_evaluators_own_cells(tmp_path, cohort):
    truth = ev.truth_table(cohort)
    fits = loaded(tmp_path, cohort, draws=(0,))
    ev.check_tables_against_truth(cohort, ev.MAIN_ROW, fits, truth)  # agrees

    wrong_cells = loaded(tmp_path / "a", cohort, draws=(0,))
    wrong_cells[0].valuations.loc[3, "n_cells"] += 1
    with pytest.raises(SystemExit, match="scoring windows differ"):
        ev.check_tables_against_truth(cohort, ev.MAIN_ROW, wrong_cells, truth)

    wrong_actual = loaded(tmp_path / "b", cohort, draws=(0,))
    row = wrong_actual[0].valuations.index[wrong_actual[0].valuations["valuation"] == 6][0]
    wrong_actual[0].valuations.loc[row, "actual"] *= 1.01
    with pytest.raises(SystemExit, match="saved actual"):
        ev.check_tables_against_truth(cohort, ev.MAIN_ROW, wrong_actual, truth)

    wrong_reserve = loaded(tmp_path / "c", cohort, draws=(0,))
    wrong_reserve[0].reserves["reserve"] *= 1.1
    with pytest.raises(SystemExit, match="differs from the reserve the fit stored"):
        ev.check_tables_against_truth(cohort, ev.MAIN_ROW, wrong_reserve, truth)

    missing = loaded(tmp_path / "d", cohort, draws=(0,))
    missing[0].valuations = missing[0].valuations[missing[0].valuations["valuation"] != 7]
    with pytest.raises(SystemExit, match="valuations table covers"):
        ev.check_tables_against_truth(cohort, ev.MAIN_ROW, missing, truth)


def test_draw_arrays_put_valuation_and_company_on_the_right_axes(tmp_path, cohort):
    fits = loaded(tmp_path, cohort, draws=(0, 1), predict=lambda i, a: a * (1 + 0.1 * i))
    arr = ev.draw_arrays(fits, ["c3", "c1"], "predicted")
    truth = ev.truth_table(cohort).set_index(["valuation", "company_code"])["actual"]
    assert arr.shape == (2, V, 2)
    for i, v in enumerate(ev.VALUATIONS):
        np.testing.assert_allclose(arr[0, i, 0], truth[(v, "c3")] * (1 + 0.1 * i))
        np.testing.assert_allclose(arr[1, i, 1], truth[(v, "c1")] * (1 + 0.1 * i))


def test_alpha_summary_is_the_median_over_members_and_draws(tmp_path, cohort):
    fits = loaded(tmp_path, cohort)
    # four members, two per draw; the median is 0.3 and the mean 0.55
    fits[0].members["alpha"] = [0.2, 0.3] * V
    fits[1].members["alpha"] = [0.3, 1.4] * V
    got = ev.alpha_summary(fits).set_index("valuation")["alpha_median"]
    assert got.loc[6] == pytest.approx(0.3)


# -- one cohort, end to end ---------------------------------------------------------------


def bench_for(cohort, mack_scale, mcl_scale, *, drop=None):
    truth = ev.truth_table(cohort)
    bench = truth.copy()
    rng = np.random.default_rng(0)
    noise = lambda: rng.uniform(-1, 1, size=len(bench))  # noqa: E731
    bench["mack"] = bench["actual"] * (1 + mack_scale * noise())
    bench["mcl"] = bench["actual"] * (1 + mcl_scale * noise())
    if drop is not None:
        bench.loc[(bench["valuation"] == drop[0]) & (bench["company_code"] == drop[1]), "mcl"] = (
            np.nan
        )
    return bench


def evaluate(tmp_path, cohort, bench, predicts, *, draws=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
    by_row = {}
    for row, predict in predicts.items():
        folder = tmp_path / row
        by_row[row] = loaded(folder, cohort, row=row, draws=tuple(draws), predict=predict)
    return ev.evaluate_cohort(cohort, by_row, bench)


def test_a_perfect_model_against_noisy_benchmarks_is_effective(tmp_path, cohort):
    res = evaluate(tmp_path, cohort, bench_for(cohort, 0.3, 0.3), {ev.MAIN_ROW: perfect})
    assert res["verdict"] == "effective" and not res["provisional"]
    assert res["n_draws"] == 10 and all(res["conditions"].values())


def test_a_model_that_is_far_off_against_good_benchmarks_is_worse(tmp_path, cohort):
    bad = lambda i, a: a * 1.5  # noqa: E731
    res = evaluate(tmp_path, cohort, bench_for(cohort, 0.01, 0.01), {ev.MAIN_ROW: bad})
    assert res["verdict"] == "worse"


def test_a_model_equal_to_a_benchmark_is_not_distinguishable(tmp_path, cohort):
    bench = bench_for(cohort, 0.2, 0.2)
    bench["mcl"] = bench["mack"]  # one benchmark, so the model equals both
    mack = bench.pivot(index="valuation", columns="company_code", values="mack")

    def like_mack(i, actual):
        return mack.loc[ev.VALUATIONS[i], sorted(cohort.board)].to_numpy()

    res = evaluate(tmp_path, cohort, bench, {ev.MAIN_ROW: like_mack})
    assert res["verdict"] == "not distinguishable"
    assert res["verdict_statistics"]["last"]["mack"]["low"] == pytest.approx(0.0, abs=1e-12)


def test_fewer_than_ten_draws_make_the_verdict_provisional_and_say_which_are_missing(
    tmp_path, cohort
):
    res = evaluate(
        tmp_path, cohort, bench_for(cohort, 0.3, 0.3), {ev.MAIN_ROW: perfect}, draws=(0, 1, 3)
    )
    assert res["provisional"] and res["missing_draws"] == [2, 4, 5, 6, 7, 8, 9]
    assert "provisional: 3 of 10 draws" in res["verdict_label"]
    assert "missing draws 2, 4" in res["verdict_reason"]


def test_a_company_a_benchmark_refused_is_left_out_of_every_row_and_named(tmp_path, cohort):
    bench = bench_for(cohort, 0.3, 0.3, drop=(7, "c2"))
    res = evaluate(tmp_path, cohort, bench, {ev.MAIN_ROW: perfect, ev.FACTORS_ROW: perfect})
    assert "c2" not in res["companies"] and len(res["companies"]) == 3
    assert res["left_out"] == {"c2": {"mcl": [7]}}
    for row in (ev.MAIN_ROW, ev.FACTORS_ROW, ev.RAW_NETWORK):
        assert res["series"][row].shape == (V, 3)
    assert res["actual"].shape == (V, 3)


def test_the_dominant_company_is_the_largest_actual_reserve_at_the_last_valuation(tmp_path, cohort):
    res = evaluate(tmp_path, cohort, bench_for(cohort, 0.3, 0.3), {ev.MAIN_ROW: perfect})
    last = cohort.actual.loc[res["companies"], "actual_reserve"]
    assert res["dominant_company"] == last.idxmax()
    assert res["dominant_share"] == pytest.approx(last.max() / last.sum())


def test_the_draw_average_is_the_mean_over_draws_not_the_median(tmp_path, cohort):
    for draw, factor in enumerate((0.9, 1.0, 1.8)):
        write_fit(tmp_path, cohort, ev.MAIN_ROW, draw, lambda i, a, f=factor: a * f)
    fits = ev.load_fits(tmp_path, "A", ev.MAIN_ROW, smoke=False)
    res = ev.evaluate_cohort(cohort, {ev.MAIN_ROW: fits}, bench_for(cohort, 0.3, 0.3))
    np.testing.assert_allclose(res["series"][ev.MAIN_ROW], res["actual"] * (0.9 + 1.0 + 1.8) / 3)
    # the per-draw Pool_APE is the draw's own, so its median is the middle draw's 0
    table = res["draw_pool_ape"]
    per = table[table["row"] == ev.MAIN_ROW].set_index("draw")["valuation_10"]
    np.testing.assert_allclose(per.to_numpy(), [0.1, 0.0, 0.8], atol=1e-12)


def test_excluding_the_dominant_company_removes_it_from_the_model_and_the_benchmark(
    tmp_path, cohort
):
    ramp = np.linspace(0.7, 1.5, len(cohort.board))
    res = evaluate(
        tmp_path,
        cohort,
        bench_for(cohort, 0.3, 0.3),
        {ev.MAIN_ROW: lambda i, a: a * ramp},
        draws=(0,),
    )
    keep = np.array([c != res["dominant_company"] for c in res["companies"]])
    assert keep.sum() == len(keep) - 1
    got = res["excluding_dominant"].set_index(["method", "benchmark"]).loc[(ev.MAIN_ROW, "mack")]
    model, bench = res["series"][ev.MAIN_ROW][LAST_I][keep], res["series"]["mack"][LAST_I][keep]
    actual = res["actual"][LAST_I][keep]
    assert got["diff"] == pytest.approx(ev.pool_ape(model, actual) - ev.pool_ape(bench, actual))
    want = ev.last_difference(model, bench, actual, first.bootstrap_indices(int(keep.sum())))
    assert (got["low"], got["high"]) == pytest.approx((want["low"], want["high"]))


def test_the_raw_network_is_the_main_fits_pre_blend_forecast_and_needs_no_extra_row(
    tmp_path, cohort
):
    res = evaluate(tmp_path, cohort, bench_for(cohort, 0.3, 0.3), {ev.MAIN_ROW: perfect})
    # tables_for writes predicted_net as 1% above predicted
    np.testing.assert_allclose(res["series"][ev.RAW_NETWORK], res["series"][ev.MAIN_ROW] * 1.01)
    assert ev.FACTORS_ROW not in res["series"]
    ablated = set(res["ablations"]["row"])
    assert ablated == {ev.RAW_NETWORK}


def test_the_ablations_are_the_difference_from_the_main_row(tmp_path, cohort):
    res = evaluate(
        tmp_path,
        cohort,
        bench_for(cohort, 0.3, 0.3),
        {ev.MAIN_ROW: perfect, ev.FACTORS_ROW: lambda i, a: a * 1.2},
    )
    abl = res["ablations"].set_index(["row", "statistic"])
    # the factors row is 20% off, the main row exact: row minus main is its Pool_APE, 0.2
    assert abl.loc[(ev.FACTORS_ROW, "last"), "diff"] == pytest.approx(0.2)
    assert abl.loc[(ev.FACTORS_ROW, "rolling"), "diff"] == pytest.approx(0.2)
    assert abl.loc[(ev.FACTORS_ROW, "last"), "low"] > 0


def test_a_first_study_row_is_compared_at_the_last_valuation_only(tmp_path, cohort):
    bench = bench_for(cohort, 0.3, 0.3)
    by_row = {ev.MAIN_ROW: loaded(tmp_path, cohort, draws=(0, 1))}
    reserve = cohort.actual["actual_reserve"].to_numpy() * 1.3
    res = ev.evaluate_cohort(cohort, by_row, bench, first_study={"first_study:tlrn_13_ay": reserve})
    row = res["ablations"].set_index("row").loc["first_study:tlrn_13_ay"]
    assert row["statistic"] == "last" and row["diff"] == pytest.approx(0.3)


def test_without_main_fits_there_is_no_verdict_but_the_benchmarks_are_still_reported(
    tmp_path, cohort
):
    bench = bench_for(cohort, 0.3, 0.3)
    by_row = {ev.FACTORS_ROW: loaded(tmp_path, cohort, row=ev.FACTORS_ROW, draws=(0,))}
    res = ev.evaluate_cohort(cohort, by_row, bench)
    assert res["verdict"] is None and "not scored" in res["verdict_label"]
    assert ev.FACTORS_ROW in res["series"] and set(res["benchmark_pool_ape"]) == {"mack", "mcl"}


def test_the_company_table_lists_every_method_for_every_valuation(tmp_path, cohort):
    res = evaluate(
        tmp_path, cohort, bench_for(cohort, 0.3, 0.3), {ev.MAIN_ROW: perfect}, draws=(0,)
    )
    table = ev.company_table(res)
    assert len(table) == V * len(cohort.board)
    assert {"actual", "mack", "mcl", ev.MAIN_ROW, ev.RAW_NETWORK} <= set(table.columns)
    np.testing.assert_allclose(table[ev.MAIN_ROW], table["actual"])


def test_the_companion_rolling_tie_out_reports_differences(tmp_path, cohort):
    res = evaluate(
        tmp_path, cohort, bench_for(cohort, 0.3, 0.3), {ev.MAIN_ROW: perfect}, draws=(0,)
    )
    res["n_cells_by_valuation"] = dict(zip(ev.VALUATIONS, (70, 71, 72, 73), strict=True))
    theirs = pd.DataFrame(
        {
            "valuation": list(ev.VALUATIONS),
            "companies": len(res["companies"]),
            "cells": [70, 71, 72, 73],
            "alpha_median": [0.5, 0.6, 0.7, 0.8],
            "TLRN": 0.05,
            "TLRN network only": 0.06,
            "Chain ladder + MCL": ev.pool_ape_by_valuation(res["series"]["mcl"], res["actual"])
            + 0.001,
            "Chain ladder": ev.pool_ape_by_valuation(res["series"]["mack"], res["actual"]),
        }
    )
    path = tmp_path / "rolling.csv"
    theirs.to_csv(path, index=False)
    table = ev.companion_rolling_tie_out(path, res)
    np.testing.assert_allclose(table["mack_difference"], 0.0, atol=1e-12)
    np.testing.assert_allclose(table["mcl_difference"], -0.001, atol=1e-12)
    assert list(table["valuation"]) == list(ev.VALUATIONS)
    assert list(table["cells_theirs"]) == [70, 71, 72, 73]
    assert list(table["alpha_median_theirs"]) == [0.5, 0.6, 0.7, 0.8]
    assert list(table["alpha_median_ours"]) == [0.5] * V  # the synthetic fits' alpha
    # a scoring window that scores other cells than the companion's is refused
    res["n_cells_by_valuation"][7] = 99
    with pytest.raises(SystemExit, match="valuation 7: this evaluator scores 99 cells"):
        ev.companion_rolling_tie_out(path, res)
    res["n_cells_by_valuation"][7] = 71
    theirs.drop(theirs.index[2]).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="no row for valuation 8"):
        ev.companion_rolling_tie_out(path, res)


def test_the_notebook_04_tie_out_refuses_a_pool_ape_that_moved(tmp_path, cohort):
    a = cohort_of(cohort.cells, "A")
    a.board = list(cohort.board)
    a.pairs_hash = study.COHORT_HASH["A"]  # the real cohort A is the one with a reference
    res = evaluate(tmp_path, a, bench_for(a, 0.3, 0.3), {ev.MAIN_ROW: perfect}, draws=(0,))
    with pytest.raises(SystemExit, match="does not reproduce notebook 04"):
        ev.reference_tie_out(a, res)
    b = cohort_of(cohort.cells, "B")
    assert ev.reference_tie_out(b, res) == {}


# -- the command line ---------------------------------------------------------------------


def _main(tmp_path, monkeypatch, cohort, *, cohort_name="A", draws=(0, 1, 2), args=()):
    named = SimpleNamespace(
        **{**vars(cohort), "name": cohort_name, "as_of": study.AS_OF[cohort_name]}
    )
    fits = tmp_path / "fits"
    for draw in draws:
        write_fit(fits, named, ev.MAIN_ROW, draw, perfect, smoke=True)
        write_fit(fits, named, ev.FACTORS_ROW, draw, lambda i, a: a * 1.1, smoke=True)
    monkeypatch.setattr(ev.study, "cells_from_mart", lambda years=None: None)
    monkeypatch.setattr(ev.study, "load_cohort", lambda name, **kw: named)
    bench = bench_for(named, 0.3, 0.3)
    monkeypatch.setattr(ev, "cached_benchmark_cells", lambda c, cache, *, log=print: bench)
    out = tmp_path / "out"
    ev.main(["--fits", str(fits), "--cohort", cohort_name, "--smoke", "--out", str(out), *args])
    return out


def test_main_writes_the_report_the_tables_and_verdicts(tmp_path, monkeypatch, cohort, capsys):
    out = _main(tmp_path, monkeypatch, cohort)
    verdicts = json.loads((out / "verdicts.json").read_text())
    assert verdicts["smoke"] is True
    record = verdicts["verdicts"]["A"]
    assert record["verdict"] == "effective" and record["provisional"] is True
    assert "provisional: 3 of 10 draws" in record["label"]
    assert record["status"] == ev.CONFIRMATORY and record["conditions"]
    for name in (
        "A_metrics_by_valuation.csv",
        "A_differences.csv",
        "A_excluding_dominant.csv",
        "A_size_quartiles.csv",
        "A_ablations.csv",
        "A_draw_pool_ape.csv",
        "A_company_predictions.csv",
        "report.txt",
    ):
        assert (out / name).exists(), name
    text = (out / "report.txt").read_text()
    assert text.startswith("SMOKE RUN") and "VERDICT for tlrn_13_fh on cohort A" in text
    assert "not out-of-sample" in text
    assert "SMOKE, not a result" in capsys.readouterr().out


def test_cohort_c_is_labelled_the_out_of_sample_one(tmp_path, monkeypatch, cohort):
    out = _main(tmp_path, monkeypatch, cohort, cohort_name="C")
    record = json.loads((out / "verdicts.json").read_text())["verdicts"]["C"]
    assert record["status"] == ev.OUT_OF_SAMPLE
    assert "out-of-sample" in (out / "report.txt").read_text()


def test_main_skips_a_cohort_with_no_fits_and_does_not_fit_its_benchmarks(
    tmp_path, monkeypatch, cohort
):
    named = SimpleNamespace(**{**vars(cohort), "name": "B", "as_of": study.AS_OF["B"]})
    monkeypatch.setattr(ev.study, "cells_from_mart", lambda years=None: None)
    monkeypatch.setattr(ev.study, "load_cohort", lambda name, **kw: named)
    fitted = []
    monkeypatch.setattr(
        ev, "cached_benchmark_cells", lambda c, cache, *, log=print: fitted.append(1)
    )
    (tmp_path / "fits").mkdir()
    ev.main(["--fits", str(tmp_path / "fits"), "--cohort", "B", "--smoke"])
    assert fitted == []


def test_the_first_evaluators_statistics_are_the_ones_used_here():
    assert ev.pool_ape is first.pool_ape
    assert ev.N_BOOT == first.N_BOOT and ev.BOOT_SEED == first.BOOT_SEED
