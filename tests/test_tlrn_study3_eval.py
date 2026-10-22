"""The third tlrn study's evaluator: the decision rules at their thresholds, and the plumbing.

``scripts/tlrn_study3_eval.py`` scores what ``scripts/tlrn_study_colab.py --study
companion_final`` saved. No network is fitted here. The tests drive the things whose mistakes
would quietly change a verdict:

* every pre-registered rule at the number it is stated at: just inside and just outside (an
  interval end at zero, 8 of 10 draws against 7, a difference of exactly 0.0, replication at
  0.5 and 1.0 points), as pure functions of numbers;
* the bootstrap with a company set per date: a company absent at a date adds nothing there, and
  with every company present it equals the second study's bootstrap;
* the checks that refuse a fit that is not the registered one (epochs, refit epoch, study,
  company sets);
* the whole path from fits on disk to ``verdicts.json`` on a synthetic cohort, with the cohort
  loader replaced so no mart is needed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import tlrn_colab  # noqa: E402
import tlrn_study2_eval as ev2  # noqa: E402
import tlrn_study3_eval as ev  # noqa: E402
import tlrn_study_colab as study  # noqa: E402
import tlrn_study_eval as first  # noqa: E402

from .test_tlrn_study3 import cohort_from, synthetic_cells  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EPS = 1e-9
V = len(ev.VALUATIONS)


def iv(low, high):
    return {"low": low, "high": high}


# -- P1 at its thresholds ------------------------------------------------------------------


def p1(**override):
    """A set of P1 statistics in which the refit helps, with one field changed at a time."""
    stats = {
        "last": iv(-0.03, -EPS),
        "pooled": iv(-0.02, -EPS),
        "draws_better": 8,
        "n_draws": 10,
        "excluding_diff": 0.0,
        "d10": -0.01,
        "rolling_diff": -0.01,
    }
    stats.update(override)
    return stats


def test_helps_needs_all_four_and_each_sits_at_its_threshold():
    word, conditions, _, _ = ev.decide_refit(p1())
    assert word == "helps" and all(conditions.values())
    # (a) the valuation-10 interval: an upper end at zero is not "wholly below"
    word, c, *_ = ev.decide_refit(p1(last=iv(-0.03, 0.0)))
    assert word == "not distinguishable" and not c["a_last_below"]
    # (b) the same for the dollar-pooled interval
    word, c, *_ = ev.decide_refit(p1(pooled=iv(-0.02, 0.0)))
    assert word == "not distinguishable" and not c["b_pooled_below"]
    # (c) 8 of 10 draws is enough and 7 of 10 is not
    assert ev.decide_refit(p1(draws_better=8))[0] == "helps"
    word, c, *_ = ev.decide_refit(p1(draws_better=7))
    assert word == "not distinguishable" and not c["c_draws_favour"]
    # (d) a point difference of exactly zero without the dominant company passes, a hair above fails
    assert ev.decide_refit(p1(excluding_diff=0.0))[0] == "helps"
    word, c, *_ = ev.decide_refit(p1(excluding_diff=EPS))
    assert word == "not distinguishable" and not c["d_excluding_at_or_below"]
    # an interval that is not a number is never a pass
    assert ev.decide_refit(p1(last=iv(np.nan, np.nan)))[0] == "not distinguishable"
    assert not ev.decide_refit(p1(excluding_diff=np.nan))[1]["d_excluding_at_or_below"]


def test_hurts_is_a_lower_end_above_zero_on_either_interval_and_only_then():
    assert ev.decide_refit(p1(last=iv(EPS, 0.03), draws_better=0))[0] == "hurts"
    assert ev.decide_refit(p1(pooled=iv(EPS, 0.03), draws_better=0))[0] == "hurts"
    # a lower end at zero is not wholly above
    assert ev.decide_refit(p1(last=iv(0.0, 0.03), draws_better=0))[0] == "not distinguishable"
    assert ev.decide_refit(p1(pooled=iv(0.0, 0.03), draws_better=0))[0] == "not distinguishable"
    # an interval across zero never hurts
    assert ev.decide_refit(p1(last=iv(-0.01, 0.01), pooled=iv(-0.01, 0.01)))[0] == (
        "not distinguishable"
    )


def test_the_draw_count_needed_is_eight_of_ten_and_scales_for_a_smaller_set():
    assert ev.required_draws(10) == 8
    assert ev.required_draws(5) == 4 and ev.required_draws(1) == 1
    assert ev.decide_refit(p1(draws_better=4, n_draws=5))[0] == "helps"
    assert ev.decide_refit(p1(draws_better=3, n_draws=5))[0] == "not distinguishable"


def test_date_dependent_is_better_at_ten_and_worse_on_the_unweighted_mean():
    assert ev.decide_refit(p1(d10=-0.0095, rolling_diff=0.0034))[2] is True
    # the companion's relayed figures: -0.95 points at valuation 10, +0.34 on the mean
    assert ev.decide_refit(p1(d10=-0.0095, rolling_diff=EPS))[2] is True
    assert ev.decide_refit(p1(d10=-0.0095, rolling_diff=0.0))[2] is False
    assert ev.decide_refit(p1(d10=0.0, rolling_diff=0.01))[2] is False
    assert ev.decide_refit(p1(d10=-EPS, rolling_diff=0.01))[2] is True
    # it is a label: the verdict is the same word with or without it
    assert ev.decide_refit(p1(d10=-0.0095, rolling_diff=0.0034))[0] == "helps"


# -- the secondary rules use the dollar-pooled interval, not the unweighted mean ------------


def secondary(**override):
    stats = {"last": iv(-0.02, -EPS), "pooled": iv(-0.02, -EPS), "excluding": iv(-0.02, 0.01)}
    stats.update(override)
    return stats


def test_the_network_and_blend_rules_at_their_thresholds():
    words = ev.NETWORK_WORDS
    assert ev.decide_secondary(secondary(), words)[0] == "network adds value"
    # pooled at zero: no longer wholly below
    assert ev.decide_secondary(secondary(pooled=iv(-0.02, 0.0)), words)[0] == "not distinguishable"
    # without the dominant company wholly above zero blocks it
    blocked = secondary(excluding=iv(EPS, 0.03))
    assert ev.decide_secondary(blocked, words)[0] == "not distinguishable"
    assert ev.decide_secondary(secondary(excluding=iv(0.0, 0.03)), words)[0] == words[0]
    # hurts: either of the first two wholly above zero
    assert ev.decide_secondary(secondary(last=iv(EPS, 0.02)), words)[0] == "network hurts"
    assert ev.decide_secondary(secondary(pooled=iv(EPS, 0.02)), words)[0] == "network hurts"
    assert ev.decide_secondary(secondary(pooled=iv(0.0, 0.02)), words)[0] == "not distinguishable"
    blend = ev.decide_secondary(secondary(last=iv(EPS, 0.02)), ev.BLEND_WORDS)
    assert blend[0] == "blend hurts"
    assert ev.decide_secondary(secondary(), ev.BLEND_WORDS)[0] == "blend adds value"


# -- P2 is the second study's rule, applied unchanged ----------------------------------------


def p2_stats(*, last_high=-EPS, pooled_high=-EPS, median=0.03, bench_pool=0.05, excl=-0.001):
    cell = lambda hi: {"low": hi - 0.04, "high": hi}  # noqa: E731
    return {
        "last": {b: cell(last_high) for b in ev.BENCHMARKS},
        "pooled": {b: cell(pooled_high) for b in ev.BENCHMARKS},
        "excluding": {
            b: {"diff": excl, "low": excl - 0.04, "high": excl + 0.04} for b in ev.BENCHMARKS
        },
        "median_draw_last": median,
        "pool_last": {b: bench_pool for b in ev.BENCHMARKS},
    }


def test_p2_is_the_second_studys_rule_at_its_thresholds():
    assert ev2.decide(p2_stats())[0] == "effective"
    assert ev2.decide(p2_stats(last_high=0.0))[0] == "not distinguishable"
    assert ev2.decide(p2_stats(pooled_high=0.0))[0] == "not distinguishable"
    assert ev2.decide(p2_stats(median=0.05))[0] == "not distinguishable"
    assert ev2.decide(p2_stats(excl=0.0))[0] == "effective"
    assert ev2.decide(p2_stats(excl=EPS))[0] == "not distinguishable"
    worse = p2_stats()
    worse["pooled"]["mcl"] = {"low": EPS, "high": 0.04}
    assert ev2.decide(worse)[0] == "worse"


# -- the replication check, in points ---------------------------------------------------------


def rep(m=3.93, lo=3.5, hi=4.4, d10=-0.95, m_net=2.96):
    return ev.replication_label(m, lo, hi, d10, m_net)


def test_r1_is_the_range_and_a_half_point_level_gap():
    assert rep()["r1"] and rep()["label"] == "replicates"
    # the level: exactly 0.5 points away is inside, a hair more is outside
    assert rep(m=3.93 + 0.5, hi=5.0)["r1"]
    assert not rep(m=3.93 + 0.5 + 1e-6, hi=5.0)["r1"]
    assert rep(m=3.93 - 0.5, lo=3.0)["r1"]
    assert not rep(m=3.93 - 0.5 - 1e-6, lo=3.0)["r1"]
    # 3.93 must lie in the range of the draws: an end exactly on it counts
    assert rep(lo=3.93)["r1"] and rep(hi=3.93)["r1"]
    assert not rep(lo=3.93 + 1e-6)["r1"] and not rep(hi=3.93 - 1e-6)["r1"]


def test_r2_is_a_refit_gain_of_at_least_half_a_point():
    assert rep(d10=-0.5)["r2"]
    assert not rep(d10=-0.5 + 1e-6)["r2"]
    assert rep(d10=-3.0)["r2"] and not rep(d10=0.0)["r2"]


def test_r3_is_the_network_alone_within_half_a_point_and_never_moves_the_label():
    assert rep(m_net=2.96 + 0.5)["r3"] and not rep(m_net=2.96 + 0.5 + 1e-6)["r3"]
    assert rep(m_net=None)["r3"] is None
    assert rep(m_net=9.0)["label"] == rep(m_net=2.96)["label"] == "replicates"


def test_the_replication_label_at_one_point():
    # neither R1 nor R2: fails only beyond 1.0 point from 3.93
    far = rep(m=3.93 + 1.0 + 1e-6, lo=5.0, hi=6.0, d10=0.0)
    assert not far["r1"] and not far["r2"] and far["label"] == "fails to replicate"
    exactly = rep(m=3.93 + 1.0, lo=5.0, hi=6.0, d10=0.0)
    assert exactly["label"] == "partially replicates"
    below = rep(m=3.93 - 1.0 - 1e-6, lo=2.0, hi=3.0, d10=0.0)
    assert below["label"] == "fails to replicate"
    # R1 alone, R2 alone, and the level within 1.0 of the figure with neither
    assert rep(d10=0.0)["label"] == "partially replicates"
    assert rep(m=3.93 + 1.0 + 1e-6, lo=3.0, hi=6.0, d10=-0.95)["label"] == "partially replicates"
    assert rep(m=3.93 + 0.7, lo=3.0, hi=6.0, d10=0.0)["label"] == "partially replicates"
    # far away but with the refit effect: R2 holds, so it only partially replicates
    assert rep(m=9.0, lo=8.0, hi=10.0, d10=-0.95)["label"] == "partially replicates"


def test_points_and_fractions_are_not_mixed_up():
    assert ev.POINT == 0.01
    assert ev.COMPANION_BLEND == 3.93 and ev.COMPANION_NETWORK == 2.96
    assert (ev.R1_TOLERANCE, ev.FAIL_DISTANCE, ev.R2_DIFFERENCE) == (0.5, 1.0, -0.5)
    # a Pool_APE of 0.0443 is 4.43 points: exactly 0.5 points above the companion's figure
    assert ev.replication_label(0.0443 / ev.POINT, 3.9, 4.9, -0.95, None)["r1"]
    assert not ev.replication_label(0.0444 / ev.POINT, 3.9, 4.9, -0.95, None)["r1"]


# -- the statements across cohorts -----------------------------------------------------------


@pytest.mark.parametrize(
    "c3, b3, key",
    [
        ("helps", "helps", "both"),
        ("helps", "not distinguishable", "fresh_only"),
        ("helps", "hurts", "fresh_only"),
        ("helps", None, "fresh_only"),
        ("not distinguishable", "helps", "nd"),
        ("not distinguishable", "hurts", "nd"),
        ("hurts", "helps", "hurts"),
        ("hurts", "not distinguishable", "hurts"),
    ],
)
def test_the_refit_statement_follows_the_preregistered_table(c3, b3, key):
    text = ev.refit_overall_statement(c3, b3)
    assert text.startswith(ev.REFIT_OVERALL[key])
    assert f"C3 {c3}" in text and "B3 " in text
    beyond = b3 == "helps" and c3 != "helps"
    assert (ev.REFIT_OVERALL["not_shown"] in text) is beyond


def test_a_cohort_not_run_is_listed_and_a_provisional_one_says_so():
    assert ev.refit_overall_statement(None, "helps") == ev.REFIT_OVERALL["none"]
    assert "B3 not run" in ev.refit_overall_statement("helps", None)
    assert ev.refit_overall_statement("helps", "helps", {"C": True}).startswith("provisional (C")
    assert not ev.refit_overall_statement("helps", "helps", {"C": False}).startswith("provisional")


@pytest.mark.parametrize(
    "c3, b3, key",
    [
        ("effective", "effective", "confirmed"),
        ("effective", "not distinguishable", "not_worse"),
        ("effective", None, "not_worse"),
        ("effective", "worse", "mixed"),
        ("not distinguishable", "effective", "nd"),
        ("not distinguishable", "worse", "nd"),
        ("worse", "effective", "worse"),
    ],
)
def test_the_effective_statement_follows_the_second_studys_table_with_b3(c3, b3, key):
    assert ev.effective_overall_statement(c3, b3).startswith(ev.EFFECTIVE_OVERALL[key])
    assert ev.effective_overall_statement(None, b3) == ev.EFFECTIVE_OVERALL["none"]


# -- the bootstrap with a company set per date ------------------------------------------------


def test_a_company_absent_at_a_date_adds_nothing_there_and_counts_n_times_elsewhere():
    pred = np.array([[5.0, 1.0, 9.0], [4.0, 2.0, 7.0]])
    actual = np.array([[4.0, 2.0, 6.0], [4.0, 3.0, 5.0]])
    mask = np.array([[True, True, False], [True, False, True]])
    # one resample draws company 0 twice and company 2 once
    counts = ev.counts_of(np.array([[0, 0, 2]]))
    assert counts.tolist() == [[2.0, 0.0, 1.0]]
    # date 0 has companies 0 and 1: company 0 counts twice, company 2 is absent
    # date 1 has companies 0 and 2: both count (twice, once)
    per_date, pooled = ev.boot_pool(pred, actual, mask, counts)
    top0, bottom0 = 2 * abs(5 - 4), 2 * 4
    top1, bottom1 = 2 * abs(4 - 4) + abs(7 - 5), 2 * 4 + 5
    assert per_date[0, 0] == pytest.approx(top0 / bottom0)
    assert per_date[0, 1] == pytest.approx(top1 / bottom1)
    assert pooled[0] == pytest.approx((top0 + top1) / (bottom0 + bottom1))
    # the point statistics sum each date's own companies
    assert ev.pool_ape_v(pred, actual, mask) == pytest.approx(
        [(1 + 1) / (4 + 2), (0 + 2) / (4 + 5)]
    )
    assert ev.pooled_ape(pred, actual, mask) == pytest.approx((1 + 1 + 0 + 2) / (4 + 2 + 4 + 5))
    # moving a prediction of the absent company changes nothing at that date
    moved = pred.copy()
    moved[0, 2] = 1e6
    assert ev.pool_ape_v(moved, actual, mask)[0] == ev.pool_ape_v(pred, actual, mask)[0]
    assert ev.boot_pool(moved, actual, mask, counts)[0][0, 0] == per_date[0, 0]


def test_with_every_company_present_it_is_the_second_studys_bootstrap():
    rng = np.random.default_rng(4)
    pred = rng.uniform(1, 9, (V, 12))
    actual = rng.uniform(1, 9, (V, 12))
    idx = first.bootstrap_indices(12, n_boot=200)
    mask = np.ones((V, 12), dtype=bool)
    per_date, pooled = ev.boot_pool(pred, actual, mask, ev.counts_of(idx))
    old_per_date = ev2.resampled_pool_ape(pred, actual, idx)
    assert per_date == pytest.approx(np.asarray(old_per_date).T)
    assert pooled == pytest.approx(ev2.resampled_pooled_ape(pred, actual, idx))
    assert ev.pooled_ape(pred, actual, mask) == pytest.approx(ev2.pooled_ape(pred, actual))
    assert ev.pool_ape_v(pred, actual, mask) == pytest.approx(
        ev2.pool_ape_by_valuation(pred, actual)
    )


def test_the_difference_summary_signs_and_intervals():
    rng = np.random.default_rng(1)
    actual = rng.uniform(5, 9, (V, 30))
    mask = np.ones((V, 30), dtype=bool)
    good = actual * 1.02
    bad = actual * 1.20
    counts = ev.counts_of(first.bootstrap_indices(30, n_boot=300))
    d = ev.difference_summary(good, bad, actual, mask, counts)
    for scope in ("last", "pooled", "rolling"):
        assert d[scope]["diff"] < 0 and d[scope]["high"] < 0
        assert d[scope]["share_favour"] == 1.0
    assert d["last"]["diff"] == pytest.approx(0.02 - 0.20)
    reverse = ev.difference_summary(bad, good, actual, mask, counts)
    assert reverse["last"]["low"] > 0


# -- fits on disk -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cohort():
    return cohort_from(synthetic_cells(spike=True))


def roles_and_config(row):
    return {r: (ro, c) for r, ro, c in tlrn_colab.configs("companion_final", 20, False)}[row]


def settings_for(cohort, row, draw, **override):
    roles, cfg = roles_and_config(row)
    s = study.fit_settings(cohort, row, roles, cfg, draw=draw, smoke=False, study="companion_final")
    s.update(override)
    return s


def members_table(*, epochs=3000, refit_gap=0):
    rows = []
    for v in ev.VALUATIONS:
        for m in range(2):
            best = 100 + m
            rows.append(
                {
                    "valuation": v,
                    "member": m,
                    "best_epoch": best,
                    "epochs_run": epochs,
                    "kept": True,
                    "alpha": 0.4 + 0.1 * m,
                    "refit_epochs": best + refit_gap,
                }
            )
    return pd.DataFrame(rows)


def write_fit(
    folder,
    cohort,
    row,
    draw,
    errors,
    *,
    sets,
    version="0.7.6",
    members=None,
    smoke=False,
    **override,
):
    """A fit on disk whose predictions are the truth times ``1 + error`` for each arm.

    ``errors`` maps a table column to the relative error of that arm.
    """
    truth = ev2.truth_table(cohort, ev.VALUATIONS, sets)
    table = truth.copy()
    for column in ev.ARM_COLUMN.values():
        table[column] = truth["actual"] * (1.0 + errors.get(column, errors.get("predicted", 0.0)))
    table.loc[table["valuation"] == ev.LAST, "actual"] = np.nan
    stem = f"{cohort.name}_{row}_d{draw:02d}{'_smoke' if smoke else ''}-ibnr{version}"
    last = table[table["valuation"] == ev.LAST].set_index("company_code")["predicted"]
    reserves = pd.DataFrame(
        {
            "company_code": last.index,
            "anchor": cohort.actual["anchor"].reindex(last.index).to_numpy(),
            "premium": cohort.actual["premium"].reindex(last.index).to_numpy(),
            "n_lines": 2,
            "reserve": last.to_numpy(),
        }
    )
    folder.mkdir(parents=True, exist_ok=True)
    reserves.to_csv(folder / f"{stem}_reserves.csv", index=False)
    table.to_csv(folder / f"{stem}_valuations.csv", index=False)
    (members if members is not None else members_table()).to_csv(
        folder / f"{stem}_members.csv", index=False
    )
    settings = settings_for(cohort, row, draw, **override)
    if smoke:
        settings["smoke"] = True
    (folder / f"{stem}.json").write_text(json.dumps({"settings": settings, "run": {}}))
    return stem


@pytest.fixture(scope="module")
def sets(cohort):
    return study.valuation_pairs(cohort)


def test_load_fits_reads_the_third_studys_fits_in_draw_order(tmp_path, cohort, sets):
    for draw in (2, 0, 1):
        write_fit(tmp_path, cohort, ev.MAIN_ROW, draw, {}, sets=sets)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    assert [f.draw for f in fits] == [0, 1, 2]
    assert ev.load_fits(tmp_path, "B", ev.FACTORS_ROW, smoke=False) == []
    assert ev.load_fits(tmp_path, "C", ev.MAIN_ROW, smoke=False) == []
    assert ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=True) == []


def test_load_fits_ignores_another_studys_record(tmp_path, cohort, sets):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets, study=2, study_name="factor_head")
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 1, {}, sets=sets)
    assert [f.draw for f in ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)] == [1]


def test_load_fits_refuses_a_draw_saved_by_two_versions_and_a_wrong_seed(tmp_path, cohort, sets):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets, version="0.7.6")
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets, version="0.7.7")
    with pytest.raises(SystemExit, match="two ibnr versions"):
        ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    other = tmp_path / "seed"
    write_fit(other, cohort, ev.MAIN_ROW, 3, {}, sets=sets, seed=11)
    with pytest.raises(SystemExit, match="not draw 3's"):
        ev.load_fits(other, "B", ev.MAIN_ROW, smoke=False)


def test_every_member_must_run_3000_epochs_and_the_refit_to_the_chosen_one(tmp_path, cohort, sets):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets, members=members_table(epochs=2999))
    with pytest.raises(SystemExit, match="did not run 3000 epochs"):
        ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    other = tmp_path / "refit"
    write_fit(other, cohort, ev.MAIN_ROW, 0, {}, sets=sets, members=members_table(refit_gap=1))
    with pytest.raises(SystemExit, match="refit .* did not run to the epoch"):
        ev.load_fits(other, "B", ev.MAIN_ROW, smoke=False)
    # a smoke folder is not held to the registered budget
    smoke = tmp_path / "smoke"
    write_fit(
        smoke, cohort, ev.MAIN_ROW, 0, {}, sets=sets, members=members_table(epochs=10), smoke=True
    )
    assert len(ev.load_fits(smoke, "B", ev.MAIN_ROW, smoke=True)) == 1
    # and a members table without the refit column is refused, not assumed
    bare = tmp_path / "bare"
    write_fit(
        bare,
        cohort,
        ev.MAIN_ROW,
        0,
        {},
        sets=sets,
        members=members_table().drop(columns="refit_epochs"),
    )
    with pytest.raises(SystemExit, match="no refit_epochs column"):
        ev.load_fits(bare, "B", ev.MAIN_ROW, smoke=False)


def test_a_fit_trained_on_other_companies_at_any_date_is_refused(tmp_path, cohort, sets):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    assert ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits, sets) == ""
    fits[0].settings["valuation_hashes"]["7"] = "0" * 16
    with pytest.raises(SystemExit, match="other companies at valuation.*7"):
        ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits, sets)


def test_a_fit_scored_on_other_cells_is_refused(tmp_path, cohort, sets):
    truth = ev2.truth_table(cohort, ev.VALUATIONS, sets)
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    ev.check_tables_against_truth(cohort, ev.MAIN_ROW, fits, truth)
    fits[0].valuations.loc[0, "n_cells"] += 1
    with pytest.raises(SystemExit, match="the scoring windows differ"):
        ev.check_tables_against_truth(cohort, ev.MAIN_ROW, fits, truth)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    fits[0].valuations = fits[0].valuations.iloc[1:]
    with pytest.raises(SystemExit, match="other \\(valuation, company\\) pairs"):
        ev.check_tables_against_truth(cohort, ev.MAIN_ROW, fits, truth)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    fits[0].valuations.loc[fits[0].valuations["valuation"] == 6, "actual"] *= 1.5
    with pytest.raises(SystemExit, match="saved actual"):
        ev.check_tables_against_truth(cohort, ev.MAIN_ROW, fits, truth)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    fits[0].reserves["reserve"] *= 2
    with pytest.raises(SystemExit, match="differs from the reserve the fit stored"):
        ev.check_tables_against_truth(cohort, ev.MAIN_ROW, fits, truth)


# -- the whole path ---------------------------------------------------------------------------


def populate(folder, cohort, sets, *, draws=None, refit_error=0.0, norefit_error=0.2):
    """Ten draws of each row: the refit near the truth, the no-refit arm further, the rest far."""
    draws = range(10) if draws is None else draws
    rng = np.random.default_rng(7)
    for draw in draws:
        jitter = rng.normal(0.0, 0.002)
        errors = {
            "predicted": refit_error + jitter,
            "predicted_selected": norefit_error + jitter,
            "predicted_net": refit_error + 0.03,
            "predicted_net_selected": norefit_error + 0.03,
        }
        write_fit(folder, cohort, ev.MAIN_ROW, draw, errors, sets=sets)
        write_fit(folder, cohort, ev.FACTORS_ROW, draw, {"predicted": 0.3}, sets=sets)


def run_synthetic(monkeypatch, folder, cohort, **kwargs):
    monkeypatch.setattr(study, "load_cohort", lambda name, **kw: cohort)
    return ev.run_cohort("B", folder, cells=cohort.all_cells, log=lambda *_: None, **kwargs)


def test_the_whole_path_on_a_synthetic_cohort(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path, cohort, sets)
    cohort_, result = run_synthetic(monkeypatch, tmp_path, cohort, cache_dir=tmp_path / "cache")
    # the company sets differ by date: the company the early dates select is scored there only
    assert "0004" in result["companies"]
    assert result["mask"][ev.VALUATIONS.index(6), result["companies"].index("0004")]
    assert not result["mask"][ev.LAST_I, result["companies"].index("0004")]
    # the refit is close to the truth and the no-refit arm is not: the refit helps
    assert result["p1"]["word"] == "helps" and result["p1"]["conditions"]["c_draws_favour"]
    assert result["p1"]["stats"]["draws_better"] == 10
    assert result["p1"]["differences"]["last"]["diff"] < 0
    assert result["p2"]["word"] == "effective"
    assert result["network"]["word"] == "network adds value"
    assert result["blend"]["word"] == "blend adds value"
    assert result["p2"]["provisional"] is False
    # the independent chain ladder agrees with the gallery entry, and the tie-out says it is off
    gaps = result["extras"][next(k for k in result["extras"] if k.startswith("mack against"))]
    assert max(gaps.values()) < ev.RECOMPUTE_TOL
    assert "NOT TIED OUT" in result["extras"]["second study's valuation-10 benchmarks"]
    # the benchmark cache is used on the second call
    again = run_synthetic(monkeypatch, tmp_path, cohort, cache_dir=tmp_path / "cache")
    assert again[1]["p1"]["word"] == "helps"
    assert any((tmp_path / "cache").glob("benchmarks3_B_*.csv"))
    # the report and the tables come out
    text = ev.render(cohort_, result, result["extras"], "")
    assert "P1 for cohort B" in text and "REPLICATION" in text
    out = tmp_path / "out"
    ev.save_tables(out, cohort_, result)
    assert (out / "B_company_predictions.csv").exists()
    record = ev.verdict_record(result)
    assert record["p1"]["verdict"] == "helps" and record["p2"]["verdict"] == "effective"
    json.dumps(record, default=str)


def test_a_refit_that_is_worse_is_reported_as_hurting(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path, cohort, sets, refit_error=0.2, norefit_error=0.0)
    _, result = run_synthetic(monkeypatch, tmp_path, cohort)
    assert result["p1"]["word"] == "hurts"
    assert result["p1"]["stats"]["draws_better"] == 0


def test_fewer_than_ten_draws_makes_every_verdict_provisional(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path, cohort, sets, draws=range(3))
    _, result = run_synthetic(monkeypatch, tmp_path, cohort)
    assert result["p1"]["provisional"] and "provisional: " in result["p1"]["label"]
    assert result["p2"]["provisional"]
    assert result["p2_missing_draws"] == list(range(3, 10))


def test_a_benchmark_failure_leaves_a_company_out_of_every_date():
    by_date = {6: ["a", "b", "c"], 7: ["a", "b", "c"], 8: ["a", "b"], 10: ["a", "b"]}
    rows = [
        {"valuation": v, "company_code": c, "mack": 1.0, "mcl": 1.0}
        for v, cs in by_date.items()
        for c in cs
    ]
    bench = pd.DataFrame(rows)
    bench.loc[(bench["company_code"] == "c") & (bench["valuation"] == 7), "mcl"] = np.nan
    keep, left_out = ev.scored_companies(bench, by_date)
    assert keep == ["a", "b"] and left_out == {"c": {"mcl": [7]}}


def test_the_dominant_company_must_be_in_every_date(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path, cohort, sets)
    monkeypatch.setattr(first, "dominant_company", lambda actual: ("0004", 0.5))
    with pytest.raises(SystemExit, match="dominant company 0004 is not in every date"):
        run_synthetic(monkeypatch, tmp_path, cohort)


def test_the_valuation_10_tie_out_stops_on_a_different_benchmark(tmp_path):
    table = pd.DataFrame(
        {
            "method": ["mack", "mcl", "mack", "mcl"],
            "valuation": ["10", "10", "rolling", "rolling"],
            "pool_ape": [0.0548098, 0.0540981, 0.0, 0.0],
        }
    )
    table.to_csv(tmp_path / "B_metrics_by_valuation.csv", index=False)
    pools = {
        "mack": np.array([0.1, 0.1, 0.1, 0.0548098]),
        "mcl": np.array([0.1, 0.1, 0.1, 0.0540981]),
    }
    good = {"left_out": {}, "benchmark_pool_ape": pools}
    out = ev.second_study_tie_out("B", good, tmp_path)
    assert out["mack"]["relative"] < 1e-12
    bad = {"left_out": {}, "benchmark_pool_ape": {**pools, "mack": pools["mack"] + 0.001}}
    with pytest.raises(SystemExit, match="does not reproduce the second study"):
        ev.second_study_tie_out("B", bad, tmp_path)
    assert "NOT TIED OUT" in ev.second_study_tie_out("C", good, tmp_path)
    assert "not compared" in ev.second_study_tie_out("B", {**good, "left_out": {"x": 1}}, tmp_path)


def test_the_second_studys_saved_benchmarks_are_what_the_tie_out_reads():
    folder = REPO / "analysis" / "results" / "tlrn_study2" / "evaluation"
    for name in ("B", "C"):
        table = pd.read_csv(folder / f"{name}_metrics_by_valuation.csv")
        got = table[
            (table["valuation"].astype(str) == "10") & table["method"].isin(["mack", "mcl"])
        ]
        assert len(got) == 2
    b = pd.read_csv(folder / "B_metrics_by_valuation.csv")
    mack = b[(b["method"] == "mack") & (b["valuation"].astype(str) == "10")]["pool_ape"].iloc[0]
    assert round(100 * mack, 3) == 5.481


# -- the pre-registered text and constants ----------------------------------------------------

#: sha256 of the evaluator's module docstring as pre-registered (recorded before any real fit,
#: as the second study's was). A change to a rule changes this; change it only by a dated
#: amendment in knowledge/decisions/tlrn-study3-preregistration.md written before the fit.
DOCSTRING_SHA256 = "552bacd548ea1237da3a2ae13e8d32a9541b3a4fc49b41da6c69d1cbfc6f4df2"


def test_the_preregistered_docstring_is_pinned():
    source = (REPO / "scripts" / "tlrn_study3_eval.py").read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(source), clean=False)
    assert hashlib.sha256(doc.encode("utf-8")).hexdigest() == DOCSTRING_SHA256


def test_the_preregistered_constants():
    assert (ev.N_BOOT, ev.BOOT_SEED, ev.STRATIFIED_SEED, ev.LEVEL) == (10_000, 2008, 2009, 0.95)
    assert ev.VALUATIONS == (6, 7, 8, 10) and ev.LAST == 10
    assert ev.ROWS == ("tlrn_cf_refit", "tlrn_cf_factors")
    assert ev.EXPECTED_EPOCHS == 3000
    assert ev.EXPECTED_PAIRS["B"] == {6: 230, 7: 226, 8: 219, 10: 212}
    assert ev.EXPECTED_PAIRS["C"] == {6: 237, 7: 236, 8: 232, 10: 232}
    assert ev.EXPECTED_COMPANIES["B"] == {6: 90, 7: 89, 8: 87, 10: 84}
    assert (ev.P1_DRAWS_REQUIRED, ev.P1_DRAWS_OF) == (8, 10)
    assert "not out-of-sample" in ev.STATUS["B"] and "out-of-sample" in ev.STATUS["C"]
    # the runner's rows and tables are the ones the evaluator reads
    spec = study.get_study("companion_final")
    assert spec.rows == ev.ROWS and set(spec.tables) == {"valuations", "members"}
    assert set(ev.ARM_COLUMN.values()) == {
        "predicted",
        "predicted_selected",
        "predicted_net",
        "predicted_net_selected",
    }
    doc = " ".join(ev.__doc__.split())
    for phrase in (
        "wholly below zero",
        "at least 8 of the 10 draws",
        "10,000 resamples",
        "Replication of the companion's reported figures",
        "never counted as hurting",
    ):
        assert phrase in doc


# -- a company that loses a line at a later date ------------------------------------------------


def test_mcl_at_a_date_fits_only_that_dates_lines():
    # company 0002 has three lines at the early dates and two at the later ones (the third
    # line's loss ratio passes 3 at calendar index 8). The triangle holds the union of all dates'
    # pairs, so a multivariate fit that took every line in it would be refused for the lines
    # the date does not score and the company would be left out of the study at valuation 10.
    cells = synthetic_cells(spike=False)
    extra = cells[(cells["company_code"] == "0002") & (cells["line_of_business"] == "lob_b")].copy()
    extra["line_of_business"] = "lob_c"
    spiked = (extra["accident_year"] == 2001) & (extra["dev_age"] == 7)
    extra.loc[spiked, "paid"] *= 8.0
    cells = study._tidy(pd.concat([cells, extra], ignore_index=True))
    cohort = cohort_from(cells)
    sets = study.valuation_pairs(cohort)
    lines = lambda v: sorted(ln for c, ln in sets[v] if c == "0002")  # noqa: E731
    assert lines(6) == ["lob_a", "lob_b", "lob_c"] and lines(10) == ["lob_a", "lob_b"]
    got = ev2.benchmark_cells(cohort, pairs_by_valuation=sets, log=lambda *_: None)
    at10 = got[(got["valuation"] == 10) & (got["company_code"] == "0002")]
    assert np.isfinite(at10[["mack", "mcl"]].to_numpy()).all()
    early = got[(got["valuation"] == 6) & (got["company_code"] == "0002")]
    assert np.isfinite(early[["mack", "mcl"]].to_numpy()).all()
    # the same company on a cohort that never had the third line gives the same valuation-10 numbers
    plain = cohort_from(synthetic_cells(spike=False))
    ref = ev2.benchmark_cells(plain, log=lambda *_: None)
    ref10 = ref[(ref["valuation"] == 10) & (ref["company_code"] == "0002")]
    assert at10[["mack", "mcl"]].to_numpy() == pytest.approx(ref10[["mack", "mcl"]].to_numpy())
