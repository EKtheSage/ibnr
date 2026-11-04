"""The fourth tlrn study's evaluator: the rules at their thresholds, P0, and the plumbing.

``scripts/tlrn_study4_eval.py`` scores what ``scripts/tlrn_study_colab.py --study
companion_revised`` saved. No network is fitted here. The tests drive what would quietly change a
verdict or a published comparison:

* the replication rule on each cohort at the numbers it is stated at, with the sign of the
  refit effect opposite on C4 (a hurt of at least half a point) to B4 (a gain);
* the reading rule for a miss of R1, including the one case that stops the work;
* P0 on small arrays with known answers, and the conditions under which it is not reported;
* the rerun control and the single-fit rows;
* the checks that refuse a fit that is not the registered one, and the whole path from fits on
  disk to ``verdicts.json`` on a synthetic cohort, with the cohort loader replaced;
* the pre-registered docstring, pinned by its sha256.
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
import tlrn_study3_eval as ev3  # noqa: E402
import tlrn_study4_eval as ev  # noqa: E402
import tlrn_study_colab as study  # noqa: E402
import tlrn_study_eval as first  # noqa: E402

from .test_tlrn_study3 import cohort_from, synthetic_cells  # noqa: E402
from .test_tlrn_study3_eval import members_table  # noqa: E402
from .test_tlrn_study3_eval import write_fit as write_third  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NAME = "companion_revised"
V = len(ev.VALUATIONS)


# -- the replication rule on each cohort -----------------------------------------------------


def rep(cohort, m, lo=None, hi=None, d10=0.0, m_net=None):
    target = ev.COMPANION[cohort]["blend"]
    return ev.replication_label(
        cohort,
        m,
        target - 0.1 if lo is None else lo,
        target + 0.1 if hi is None else hi,
        d10,
        m_net,
    )


def test_the_targets_are_the_companions_and_the_third_studys_are_not_reused():
    assert ev.COMPANION["B"]["blend"] == 3.37 and ev.COMPANION["B"]["network"] == 4.31
    assert (
        ev.COMPANION["B"]["blend_no_refit"] == 4.37 and ev.COMPANION["B"]["median_alpha"] == 0.4618
    )
    assert ev.COMPANION["C"]["blend"] == 5.75 and ev.COMPANION["C"]["network"] == 6.29
    assert (
        ev.COMPANION["C"]["blend_no_refit"] == 4.73 and ev.COMPANION["C"]["median_alpha"] == 0.6617
    )
    assert ev.COMPANION["B"]["seeds"] == {"mean": 3.62, "min": 3.37, "max": 3.88}
    assert ev.COMPANION["C"]["seeds"] == {"mean": 6.96, "min": 5.75, "max": 7.67}
    assert ev.COMPANION["B"]["blend"] != ev3.COMPANION_BLEND  # the third study's was 3.93
    assert (ev.R1_TOLERANCE, ev.FAIL_DISTANCE, ev.R2_DIFFERENCE, ev.R3_TOLERANCE) == (
        0.5,
        1.0,
        0.5,
        0.5,
    )


@pytest.mark.parametrize("cohort", ["B", "C"])
def test_r1_is_the_range_and_a_half_point_level_gap_on_each_cohort(cohort):
    t = ev.COMPANION[cohort]["blend"]
    d10 = -0.9 if cohort == "B" else 0.9
    assert rep(cohort, t, d10=d10)["r1"]
    assert rep(cohort, t + 0.5, hi=t + 1.0, d10=d10)["r1"]
    assert not rep(cohort, t + 0.5 + 1e-6, hi=t + 1.0, d10=d10)["r1"]
    assert rep(cohort, t - 0.5, lo=t - 1.0, d10=d10)["r1"]
    assert not rep(cohort, t - 0.5 - 1e-6, lo=t - 1.0, d10=d10)["r1"]
    # the figure itself has to lie in the draws' range: an end exactly on it counts
    assert rep(cohort, t, lo=t, d10=d10)["r1"] and rep(cohort, t, hi=t, d10=d10)["r1"]
    assert not rep(cohort, t, lo=t + 1e-6, d10=d10)["r1"]
    assert not rep(cohort, t, hi=t - 1e-6, d10=d10)["r1"]


def test_r2_has_the_opposite_sign_on_the_two_cohorts():
    # B4: the refit has to help by at least half a point
    assert rep("B", 3.4, d10=-0.5)["r2"] and not rep("B", 3.4, d10=-0.5 + 1e-6)["r2"]
    assert not rep("B", 3.4, d10=+0.9)["r2"]
    # C4: his refit hurt (5.75 against 4.73); the rule asks for a hurt of at least half a point
    assert rep("C", 5.8, d10=+0.5)["r2"] and not rep("C", 5.8, d10=+0.5 - 1e-6)["r2"]
    assert not rep("C", 5.8, d10=-0.9)["r2"] and not rep("C", 5.8, d10=0.0)["r2"]


def test_r3_is_the_network_alone_within_half_a_point_against_each_cohorts_figure():
    assert rep("B", 3.4, m_net=4.31 + 0.5)["r3"] and not rep("B", 3.4, m_net=4.31 + 0.51)["r3"]
    assert rep("C", 5.8, m_net=6.29 - 0.5)["r3"] and not rep("C", 5.8, m_net=6.29 - 0.51)["r3"]
    assert rep("B", 3.4, m_net=None)["r3"] is None
    assert (
        rep("B", 3.4, d10=-0.9, m_net=9.0)["label"] == rep("B", 3.4, d10=-0.9, m_net=4.3)["label"]
    )


@pytest.mark.parametrize("cohort, sign", [("B", -1.0), ("C", +1.0)])
def test_the_label_at_one_point_on_each_cohort(cohort, sign):
    t = ev.COMPANION[cohort]["blend"]
    good = sign * 0.9
    assert rep(cohort, t, d10=good)["label"] == "replicates"
    # neither R1 nor R2: fails only beyond 1.0 point from the figure
    far = rep(cohort, t + 1.0 + 1e-6, lo=t + 2, hi=t + 3, d10=0.0)
    assert not far["r1"] and not far["r2"] and far["label"] == "fails to replicate"
    assert rep(cohort, t + 1.0, lo=t + 2, hi=t + 3, d10=0.0)["label"] == "partially replicates"
    assert rep(cohort, t - 1.0 - 1e-6, lo=t - 3, hi=t - 2, d10=0.0)["label"] == "fails to replicate"
    # one of the two holding is only a partial replication, whichever it is
    assert rep(cohort, t, d10=0.0)["label"] == "partially replicates"
    assert (
        rep(cohort, t + 2.0, lo=t, hi=t + 3, d10=good)["label"] == "partially replicates"
    )  # R2 holds, level far away


def test_the_third_studys_wrong_direction_does_not_replicate_c4():
    # a C4 refit that helps (negative D10) is the opposite of the companion's: R2c fails
    out = rep("C", 5.75, d10=-1.0)
    assert out["r1"] and not out["r2"] and out["label"] == "partially replicates"


def test_points_and_fractions_are_not_mixed_up():
    assert ev.POINT == 0.01
    assert ev.replication_label("B", 0.0387 / ev.POINT, 3.0, 4.0, -0.9, None)["r1"]
    assert not ev.replication_label("B", 0.0388 / ev.POINT, 3.0, 4.0, -0.9, None)["r1"]


# -- how a miss of R1 is read ------------------------------------------------------------------


def reading(m, d10=0.0, cohort="B", p0=None, spread=None):
    r = rep(cohort, m, lo=m - 0.1, hi=m + 0.1, d10=d10)
    return ev.reading_of_miss(r, p0, spread)


def test_reading_1_when_r1_holds():
    out = reading(3.4, d10=-0.9)
    assert out["stop"] is False and out["lines"][0].startswith("(1) R1 holds")


@pytest.mark.parametrize("m", [3.88, 4.0, 4.37])
def test_reading_2_in_the_middle_band_is_partial_and_does_not_stop(m):
    out = reading(m, d10=0.0)
    assert out["stop"] is False and out["lines"][0].startswith("(2)")
    assert "not read as a defect" in out["lines"][0]


@pytest.mark.parametrize("m", [4.38, 6.0, 2.36, 1.0])
def test_reading_3_above_4_37_or_below_2_37_with_r2_failing_stops_and_asks(m):
    out = reading(m, d10=0.0)
    assert out["stop"] is True and ev.STOP_AND_ASK in out["lines"][0]
    # if R2 holds the stop does not apply
    assert reading(m, d10=-0.9)["stop"] is False


def test_the_band_edges_are_the_preregistered_ones():
    assert (ev.READ_NEAR_LOW, ev.READ_NEAR_HIGH, ev.READ_FAR_LOW) == (3.87, 4.37, 2.37)
    assert reading(3.87, d10=0.0)["stop"] is False  # not above 3.87: R1 misses, no stop, no (2)
    assert reading(2.37, d10=0.0)["stop"] is False


def test_reading_5_compares_the_p0_difference_with_the_third_studys_spread():
    small = reading(4.6, d10=0.0, p0=0.3, spread=0.53)
    big = reading(4.6, d10=0.0, p0=-0.9, spread=0.53)
    text_small = " ".join(small["lines"])
    text_big = " ".join(big["lines"])
    assert "did not detectably move ibnr's level" in text_small
    assert "larger than the third study's draw spread" in text_big
    # it is only said when R1 misses and both numbers are known
    assert "(5)" not in " ".join(reading(3.4, d10=-0.9, p0=0.0, spread=0.5)["lines"])
    assert "(5)" not in " ".join(reading(4.6, d10=0.0)["lines"])
    # (4): a miss changes no verdict, always printed
    assert any(line.startswith("(4)") for line in small["lines"])


def test_on_c4_a_draw_average_above_6_25_is_called_unsurprising_and_nothing_stops():
    high = reading(6.5, d10=0.9, cohort="C")
    assert (
        "unsurprising" in high["lines"][0]
        and "lowest of the companion's three seeds" in high["lines"][0]
    )
    assert high["stop"] is False
    low = reading(5.9, d10=0.9, cohort="C")
    assert "unsurprising" not in low["lines"][0] and low["stop"] is False
    # on C4 even a far miss with R2 failing never stops: the stop rule is B4's
    assert reading(9.0, d10=0.0, cohort="C")["stop"] is False


# -- the statements across cohorts -------------------------------------------------------------


@pytest.mark.parametrize(
    "c4, b4, key",
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
def test_the_refit_statement_follows_the_preregistered_table(c4, b4, key):
    text = ev.refit_overall_statement(c4, b4)
    assert text.startswith(ev.REFIT_OVERALL[key])
    assert f"C4 {c4}" in text and "B4 " in text
    beyond = b4 == "helps" and c4 != "helps"
    assert (ev.REFIT_OVERALL["not_shown"] in text) is beyond


def test_a_cohort_not_run_is_listed_and_a_provisional_one_says_so():
    assert ev.refit_overall_statement(None, "helps") == ev.REFIT_OVERALL["none"]
    assert "B4 not run" in ev.refit_overall_statement("helps", None)
    assert ev.refit_overall_statement("helps", "helps", {"C": True}).startswith("provisional (C")
    assert not ev.refit_overall_statement("helps", "helps", {"C": False}).startswith("provisional")
    assert ev.effective_overall_statement("effective", "effective", {"B": True}).startswith(
        "provisional (B"
    )


@pytest.mark.parametrize(
    "c4, b4, key",
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
def test_the_effective_statement_follows_the_table_with_b4(c4, b4, key):
    assert ev.effective_overall_statement(c4, b4).startswith(ev.EFFECTIVE_OVERALL[key])
    assert ev.effective_overall_statement(None, b4) == ev.EFFECTIVE_OVERALL["none"]


# -- P0 on small arrays with known answers ---------------------------------------------------


def test_p0_arm_is_fourth_minus_third_by_draw_and_by_draw_average():
    rng = np.random.default_rng(3)
    n_dr, n_co = 4, 30
    actual = rng.uniform(5, 9, (V, n_co))
    mask = np.ones((V, n_co), dtype=bool)
    new = np.stack([actual * (1.02 + 0.001 * k) for k in range(n_dr)])
    old = np.stack([actual * (1.05 + 0.001 * k) for k in range(n_dr)])
    counts = ev.counts_of(first.bootstrap_indices(n_co, n_boot=300))
    keep10 = mask[ev.LAST_I].copy()
    keep10[0] = False
    idx_ex = first.bootstrap_indices(int(keep10.sum()), n_boot=300)
    out = ev.p0_arm(new, old, actual, mask, counts, idx_ex, keep10, [0, 1, 2, 3])
    expected = [(0.02 + 0.001 * k) - (0.05 + 0.001 * k) for k in range(n_dr)]
    assert out["per_draw_diff"] == pytest.approx(expected)
    assert out["median_diff"] == pytest.approx(-0.03)
    assert out["range_diff"] == pytest.approx((-0.03, -0.03))
    # draw averages: 1.0215 against 1.0515
    assert out["last"]["diff"] == pytest.approx(0.0215 - 0.0515)
    assert out["last"]["high"] < 0 and out["pooled"]["high"] < 0
    assert out["excluding"]["diff"] == pytest.approx(-0.03, abs=1e-9)
    assert out["smaller_in_fourth"] == out["companies"] == n_co
    swapped = ev.p0_arm(old, new, actual, mask, counts, idx_ex, keep10, [0, 1, 2, 3])
    assert swapped["last"]["diff"] == pytest.approx(-out["last"]["diff"])
    assert swapped["smaller_in_fourth"] == 0


def test_the_third_study_spread_constants_are_the_preregistered_ones():
    assert ev.THIRD_STUDY_SPREAD == {"B": 0.53, "C": 1.95}
    assert ev.COMPANION_EARLIER_BLEND_B == 3.93
    assert ev.COMPANION_VARIANTS["tlrn_cr_paid"] == {"blend": 5.28, "network": 6.73}
    assert ev.COMPANION_VARIANTS["tlrn_cr_noay"] == {"blend": 4.36, "network": 3.76}
    assert ev.COMPANION_VARIANTS["tlrn_cr_noline"] == {"blend": 4.10, "network": 4.47}
    assert ev.COMPANION_VARIANTS["tlrn_cr_refit"] == {"blend": 4.40, "network": 4.71}


# -- fits on disk ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cohort():
    return cohort_from(synthetic_cells(spike=True))


@pytest.fixture(scope="module")
def sets(cohort):
    return study.valuation_pairs(cohort)


def roles_and_config(row):
    return {r: (ro, c) for r, ro, c in tlrn_colab.configs(NAME, 20, False)}[row]


def settings_for(cohort, row, draw, **override):
    roles, cfg = roles_and_config(row)
    s = study.fit_settings(cohort, row, roles, cfg, draw=draw, smoke=False, study=NAME)
    s.update(override)
    return s


def write_fit(folder, cohort, row, draw, errors, *, sets, smoke=False, run=None, **override):
    """A fourth-study fit on disk whose predictions are the truth times ``1 + error``."""
    truth = ev2.truth_table(cohort, ev.VALUATIONS, sets)
    table = truth.copy()
    for column in ev.ARM_COLUMN.values():
        table[column] = truth["actual"] * (1.0 + errors.get(column, errors.get("predicted", 0.0)))
    table.loc[table["valuation"] == ev.LAST, "actual"] = np.nan
    stem = f"{cohort.name}_{row}_d{draw:02d}{'_smoke' if smoke else ''}-ibnr0.7.7"
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
    members_table().to_csv(folder / f"{stem}_members.csv", index=False)
    settings = settings_for(cohort, row, draw, **override)
    if smoke:
        settings["smoke"] = True
    (folder / f"{stem}.json").write_text(json.dumps({"settings": settings, "run": run or {}}))
    return stem


def test_load_fits_reads_the_fourth_studys_fits_in_draw_order(tmp_path, cohort, sets):
    for draw in (2, 0, 1):
        write_fit(tmp_path, cohort, ev.MAIN_ROW, draw, {}, sets=sets)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    assert [f.draw for f in fits] == [0, 1, 2]
    assert ev.load_fits(tmp_path, "B", ev.FACTORS_ROW, smoke=False) == []
    assert ev.load_fits(tmp_path, "C", ev.MAIN_ROW, smoke=False) == []
    assert ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=True) == []


def test_load_fits_ignores_the_third_studys_record_with_the_same_file_pattern(
    tmp_path, cohort, sets
):
    # a third-study record under a fourth-study row name is not read as this study's
    write_fit(
        tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets, study=3, study_name="companion_final"
    )
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 1, {}, sets=sets)
    assert [f.draw for f in ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)] == [1]


def test_load_fits_refuses_two_versions_a_wrong_seed_and_a_wrong_embedding(tmp_path, cohort, sets):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets)
    other = tmp_path / "v"
    write_fit(other, cohort, ev.MAIN_ROW, 0, {}, sets=sets)
    old = next(other.glob("*_d00-ibnr0.7.7.json"))
    twin = old.with_name(old.name.replace("0.7.7", "0.7.6"))
    twin.write_text(old.read_text())
    with pytest.raises(SystemExit, match="two ibnr versions"):
        ev.load_fits(other, "B", ev.MAIN_ROW, smoke=False)
    seed = tmp_path / "seed"
    write_fit(seed, cohort, ev.MAIN_ROW, 3, {}, sets=sets, seed=11)
    with pytest.raises(SystemExit, match="not draw 3's"):
        ev.load_fits(seed, "B", ev.MAIN_ROW, smoke=False)
    # the main row must have the per-count table off; the control row must have it on
    wrong = settings_for(cohort, ev.MAIN_ROW, 0)
    wrong["config"]["maturity_embedding"] = True
    table = tmp_path / "emb"
    write_fit(table, cohort, ev.MAIN_ROW, 0, {}, sets=sets, config=wrong["config"])
    with pytest.raises(SystemExit, match="maturity_embedding is True but row tlrn_cr_refit"):
        ev.load_fits(table, "B", ev.MAIN_ROW, smoke=False)
    control = tmp_path / "control"
    write_fit(control, cohort, ev.CONTROL_ROW, 0, {}, sets=sets)
    assert len(ev.load_fits(control, "B", ev.CONTROL_ROW, smoke=False)) == 1
    off = settings_for(cohort, ev.CONTROL_ROW, 0)
    off["config"]["maturity_embedding"] = False
    bad_control = tmp_path / "bad_control"
    write_fit(bad_control, cohort, ev.CONTROL_ROW, 0, {}, sets=sets, config=off["config"])
    with pytest.raises(SystemExit, match="maturity_embedding is False but row tlrn_cf_rerun"):
        ev.load_fits(bad_control, "B", ev.CONTROL_ROW, smoke=False)


def test_load_fits_refuses_paid_only_where_the_row_is_not_paid_only(tmp_path, cohort, sets):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets, paid_only=True)
    with pytest.raises(SystemExit, match="paid_only is True but row tlrn_cr_refit"):
        ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    ok = tmp_path / "paid"
    write_fit(ok, cohort, "tlrn_cr_paid", 0, {}, sets=sets)
    assert len(ev.load_fits(ok, "B", "tlrn_cr_paid", smoke=False)) == 1
    missing = tmp_path / "missing"
    write_fit(missing, cohort, "tlrn_cr_paid", 0, {}, sets=sets, paid_only=False)
    with pytest.raises(SystemExit, match="paid_only is False but row tlrn_cr_paid"):
        ev.load_fits(missing, "B", "tlrn_cr_paid", smoke=False)


def test_every_member_must_run_its_epochs_but_a_smoke_folder_is_not_held_to_them(
    tmp_path, cohort, sets
):
    stem = write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets)
    members_table(epochs=2999).to_csv(tmp_path / f"{stem}_members.csv", index=False)
    with pytest.raises(SystemExit, match="did not run 3000 epochs"):
        ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    smoke = tmp_path / "smoke"
    stem = write_fit(smoke, cohort, ev.MAIN_ROW, 0, {}, sets=sets, smoke=True)
    members_table(epochs=10).to_csv(smoke / f"{stem}_members.csv", index=False)
    assert len(ev.load_fits(smoke, "B", ev.MAIN_ROW, smoke=True)) == 1


def test_a_reused_fit_is_read_like_any_other_and_named_in_the_environments(tmp_path, cohort, sets):
    run = {"fit_seconds": 0.0, "reused_from": {"record": "x.json", "run": {"jax": "0.7.1"}}}
    write_fit(tmp_path, cohort, ev.FACTORS_ROW, 0, {"predicted": 0.3}, sets=sets, run=run)
    (fit,) = ev.load_fits(tmp_path, "B", ev.FACTORS_ROW, smoke=False)
    env = ev.environments([fit])
    assert list(env) == ["reused from the third study: jax 0.7.1"]


def test_a_fit_trained_on_other_companies_at_any_date_is_refused(tmp_path, cohort, sets):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    assert ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits, sets) == ""
    fits[0].settings["valuation_hashes"]["7"] = "0" * 16
    with pytest.raises(SystemExit, match="other companies at valuation.*7"):
        ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits, sets)


def test_the_data_hash_is_printed_whether_or_not_it_matches(tmp_path, cohort, sets):
    write_fit(tmp_path, cohort, ev.MAIN_ROW, 0, {}, sets=sets)
    fits = ev.load_fits(tmp_path, "B", ev.MAIN_ROW, smoke=False)
    assert ev.data_hash_report(cohort, fits).endswith("(equal)")
    fits[0].settings["data_hash"] = "f" * 16
    text = ev.data_hash_report(cohort, fits)
    assert text.endswith("(DIFFERENT)") and "f" * 16 in text and cohort.data_hash in text
    # and a data hash that differs alone does not stop the study
    assert ev.check_fits_against_cohort(cohort, ev.MAIN_ROW, fits, sets) is not None


def test_failed_fits_are_listed_from_their_text_files(tmp_path):
    (tmp_path / "FAILED_B_tlrn_cr_paid_d00-ibnr0.7.7.txt").write_text(
        "B tlrn_cr_paid draw 0 failed\n"
    )
    (tmp_path / "FAILED_B_tlrn_cr_noay_d00_smoke-ibnr0.7.7.txt").write_text("smoke one\n")
    assert ev.failed_fits(tmp_path, smoke=False) == ["B tlrn_cr_paid draw 0 failed"]
    assert ev.failed_fits(tmp_path, smoke=True) == ["smoke one"]


# -- the whole path --------------------------------------------------------------------------


def populate(folder, cohort, sets, *, draws=None, refit_error=0.0, norefit_error=0.2):
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
    for row, err in (("tlrn_cr_noline", 0.01), ("tlrn_cr_noay", 0.02), ("tlrn_cr_paid", 0.04)):
        write_fit(
            folder, cohort, row, 0, {"predicted": err, "predicted_net": err + 0.01}, sets=sets
        )
    write_fit(
        folder, cohort, ev.CONTROL_ROW, 0, {"predicted": 0.09, "predicted_net": 0.1}, sets=sets
    )


def populate_third(folder, cohort, sets, *, error=0.1):
    for draw in range(10):
        errors = {
            "predicted": error,
            "predicted_selected": 0.2,
            "predicted_net": error + 0.03,
            "predicted_net_selected": 0.23,
        }
        write_third(folder, cohort, ev3.MAIN_ROW, draw, errors, sets=sets)


def run_synthetic(monkeypatch, folder, cohort, **kwargs):
    monkeypatch.setattr(study, "load_cohort", lambda name, **kw: cohort)
    # a synthetic cohort has no third-study values to equal; that stop has its own tests
    monkeypatch.setattr(ev, "third_study_tie_out", lambda *a, **k: "not compared: synthetic")
    return ev.run_cohort("B", folder, cells=cohort.all_cells, log=lambda *_: None, **kwargs)


def test_the_whole_path_on_a_synthetic_cohort(tmp_path, monkeypatch, cohort, sets):
    mine, third = tmp_path / "mine", tmp_path / "third"
    populate(mine, cohort, sets)
    populate_third(third, cohort, sets)
    cohort_, result = run_synthetic(
        monkeypatch, mine, cohort, cache_dir=tmp_path / "cache", third_study_fits=third
    )
    assert result["label"] == "B4" and "not out-of-sample" in result["status"]
    # the refit is close to the truth and the no-refit arm is not: the refit helps
    assert result["p1"]["word"] == "helps" and result["p1"]["stats"]["draws_better"] == 10
    assert result["p2"]["word"] == "effective" and result["p2"]["provisional"] is False
    assert result["network"]["word"] == "network adds value"
    assert result["blend"]["word"] == "blend adds value"
    # P0: the fourth study's refit (error ~0) against the third's (error 0.1)
    p0 = result["p0"][ev.REFIT]
    assert p0["median_diff"] == pytest.approx(-0.1, abs=0.01)
    assert p0["last"]["high"] < 0 and p0["smaller_in_fourth"] == p0["companies"]
    assert set(result["p0"]) == set(ev.P0_ARMS) and result["p0_note"] == ""
    assert result["p0_third_spread"] == pytest.approx(0.0, abs=1e-6)
    # the single fits and the control
    variants = result["variants"].set_index("row")
    assert list(variants.index) == [ev.MAIN_ROW, *ev.VARIANT_ROWS, ev.CONTROL_ROW]
    assert variants.loc["tlrn_cr_noline", "blend_valuation_10"] == pytest.approx(0.01)
    assert variants.loc["tlrn_cr_paid", "network_valuation_10"] == pytest.approx(0.05)
    assert variants.loc["tlrn_cr_noay", "companion_blend_mean_dates"] == 4.36
    control = result["rerun_control"]["blend"]
    assert control["rerun"] == pytest.approx(0.09)
    assert control["third_study_draw_0"] == pytest.approx(0.1)
    assert control["rerun_minus_third"] == pytest.approx(-0.01)
    # R1 misses here (the synthetic figure is nowhere near 3.37), with a reading printed
    assert result["replication"]["label"] in ev3.REPLICATION_WORDS
    assert result["replication"]["reading"]["lines"]
    # the tie-out's answer is carried to the report; the docstring hash is recorded
    assert "synthetic" in result["extras"]["third study's valuation-10 benchmarks"]
    assert result["extras"]["docstring sha256"] == ev.docstring_sha256()
    text = ev.render(cohort_, result, result["extras"], "")
    for needle in (
        "P1 for cohort B4",
        "P0, embedding removed minus kept",
        "REPLICATION of the companion's reported figures",
        "SINGLE-DRAW ROWS",
        "RERUN CONTROL",
        "companion's three seeds",
    ):
        assert needle in text, needle
    out = tmp_path / "out"
    ev.save_tables(out, cohort_, result)
    for name in ("B_single_draw_rows.csv", "B_p0_per_draw.csv", "B_company_predictions.csv"):
        assert (out / name).exists(), name
    record = ev.verdict_record(result)
    assert record["p1"]["verdict"] == "helps" and record["p0_reported"] is True
    json.dumps(record, default=str)


def test_p0_is_not_reported_without_the_third_studys_fits_or_with_other_companies(
    tmp_path, monkeypatch, cohort, sets
):
    mine, third = tmp_path / "mine", tmp_path / "third"
    populate(mine, cohort, sets)
    _, without = run_synthetic(monkeypatch, mine, cohort, cache_dir=tmp_path / "cache")
    assert without["p0"] is None and "were not given" in without["p0_note"]
    assert "rerun_control" not in without
    populate_third(third, cohort, sets)
    for path in third.glob("*.json"):
        body = json.loads(path.read_text())
        body["settings"]["valuation_hashes"]["7"] = "0" * 16
        path.write_text(json.dumps(body))
    _, other = run_synthetic(
        monkeypatch, mine, cohort, cache_dir=tmp_path / "cache", third_study_fits=third
    )
    assert other["p0"] is None and "pair hashes of the two studies differ" in other["p0_note"]
    assert "P0, embedding removed minus kept" in ev.render(
        cohort, other, other["extras"], ""
    ) and "not reported" in ev.render(cohort, other, other["extras"], "")


def test_a_refit_that_is_worse_is_reported_as_hurting(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path, cohort, sets, refit_error=0.2, norefit_error=0.0)
    _, result = run_synthetic(monkeypatch, tmp_path, cohort)
    assert result["p1"]["word"] == "hurts" and result["p1"]["stats"]["draws_better"] == 0


def test_fewer_than_ten_draws_makes_every_verdict_provisional(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path, cohort, sets, draws=range(3))
    _, result = run_synthetic(monkeypatch, tmp_path, cohort)
    assert result["p1"]["provisional"] and "provisional: " in result["p1"]["label"]
    assert result["p2"]["provisional"] and result["p2_missing_draws"] == list(range(3, 10))


def test_a_row_that_failed_leaves_the_rest_of_the_table_in_place(
    tmp_path, monkeypatch, cohort, sets
):
    populate(tmp_path, cohort, sets)
    for path in tmp_path.glob("B_tlrn_cr_paid_*"):
        path.unlink()
    (tmp_path / "FAILED_B_tlrn_cr_paid_d00-ibnr0.7.7.txt").write_text("B tlrn_cr_paid failed\n")
    _, result = run_synthetic(monkeypatch, tmp_path, cohort)
    assert "tlrn_cr_paid" not in set(result["variants"]["row"])
    assert "B tlrn_cr_paid failed" in " ".join(
        str(v) for v in result["extras"]["fits the runner reported as failed"].split("\n")
    )


def test_the_dominant_company_must_be_in_every_date(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path, cohort, sets)
    monkeypatch.setattr(first, "dominant_company", lambda actual: ("0004", 0.5))
    with pytest.raises(SystemExit, match="dominant company 0004 is not in every date"):
        run_synthetic(monkeypatch, tmp_path, cohort)


def test_the_valuation_10_tie_out_reads_the_third_studys_saved_values_and_stops_on_a_gap(tmp_path):
    table = pd.DataFrame(
        {
            "method": ["mack", "mcl", "mack", "mcl"],
            "valuation": ["10", "10", "rolling", "rolling"],
            "pool_ape": [0.05481, 0.05410, 0.0, 0.0],
        }
    )
    table.to_csv(tmp_path / "B_metrics_by_valuation.csv", index=False)
    pools = {"mack": np.array([0.1, 0.1, 0.1, 0.05481]), "mcl": np.array([0.1, 0.1, 0.1, 0.0541])}
    good = {"left_out": {}, "benchmark_pool_ape": pools}
    assert ev.third_study_tie_out("B", good, tmp_path)["mack"]["relative"] < 1e-12
    bad = {"left_out": {}, "benchmark_pool_ape": {**pools, "mack": pools["mack"] + 0.001}}
    with pytest.raises(SystemExit, match="does not reproduce the third study"):
        ev.third_study_tie_out("B", bad, tmp_path)


def test_the_third_studys_saved_benchmarks_are_what_the_tie_out_reads():
    folder = REPO / "analysis" / "results" / "tlrn_study3" / "evaluation"
    expect = {("B", "mack"): 5.481, ("B", "mcl"): 5.410, ("C", "mack"): 11.947, ("C", "mcl"): 4.969}
    for (name, method), want in expect.items():
        table = pd.read_csv(folder / f"{name}_metrics_by_valuation.csv")
        row = table[(table["method"] == method) & (table["valuation"].astype(str) == "10")]
        assert round(100 * row["pool_ape"].iloc[0], 3) == want, (name, method)


# -- the pre-registered text and constants ---------------------------------------------------

#: sha256 of the evaluator's module docstring as pre-registered (recorded in the amendment in
#: knowledge/decisions/tlrn-study4-preregistration.md before any real fit). Change it only by a
#: dated amendment written before the fit.
DOCSTRING_SHA256 = "7357336acaf0f854b35d3b547c47f360b9c2964ec4ee6c7b7607603449fba029"


def test_the_preregistered_docstring_is_pinned():
    source = (REPO / "scripts" / "tlrn_study4_eval.py").read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(source), clean=False)
    assert hashlib.sha256(doc.encode("utf-8")).hexdigest() == DOCSTRING_SHA256
    assert ev.docstring_sha256() == DOCSTRING_SHA256


def test_the_preregistered_constants():
    assert (ev.N_BOOT, ev.BOOT_SEED, ev.STRATIFIED_SEED, ev.LEVEL) == (10_000, 2008, 2009, 0.95)
    assert ev.VALUATIONS == (6, 7, 8, 10) and ev.LAST == 10
    assert study.get_study(NAME).rows == ev.ROWS
    assert ev.EXPECTED_EPOCHS == 3000
    assert ev.EXPECTED_PAIRS["B"] == {6: 230, 7: 226, 8: 219, 10: 212}
    assert ev.EXPECTED_PAIRS["C"] == {6: 237, 7: 236, 8: 232, 10: 232}
    assert ev.EXPECTED_COMPANIES["B"] == {6: 90, 7: 89, 8: 87, 10: 84}
    assert ev.EXPECTED_COMPANIES["C"] == {6: 84, 7: 84, 8: 83, 10: 83}
    assert (ev.P1_DRAWS_REQUIRED, ev.P1_DRAWS_OF) == (8, 10)
    assert "not out-of-sample" in ev.STATUS["B"] and "155 of its 232 pairs" in ev.STATUS["C"]
    assert ev.COMPANION["B"]["companies_per_date"] == (90, 89, 87, 84)
    assert ev.COMPANION["C"]["companies_per_date"] == (84, 84, 83, 83)
    assert set(ev.ARM_COLUMN.values()) == {
        "predicted",
        "predicted_selected",
        "predicted_net",
        "predicted_net_selected",
    }
    assert tuple(ev.VARIANT_ROWS) + (ev.CONTROL_ROW,) == study.get_study(NAME).single_draw_rows
    doc = " ".join(ev.__doc__.split())
    for phrase in (
        "wholly below zero",
        "at least 8 of the 10 draws",
        "10,000 resamples",
        "R2c needs ``D10 >= +0.5``",
        "STOP AND ASK ETHAN",
        "above 4.37 or below 2.37",
        "never counted as hurting",
        "Expected pairs 230, 226, 219 and 212 on B4 and 237, 236, 232 and 232 on C4",
        "0.53 points of Pool_APE at valuation 10",
        "mack 5.481 and mcl 5.410 percent on B4, mack 11.947 and mcl 4.969 percent on C4",
        "B4 mean 3.62, range 3.37 to 3.88; C4 mean 6.96, range 5.75 to 7.67",
    ):
        assert phrase in doc, phrase


def test_the_docstring_numbers_are_the_modules_constants():
    doc = " ".join(ev.__doc__.split())
    for cohort in ("B", "C"):
        c = ev.COMPANION[cohort]
        for key in ("blend", "network", "blend_no_refit"):
            assert f"{c[key]}" in doc, (cohort, key)
        assert f"{c['median_alpha']}" in doc


def test_the_evaluator_reads_the_runners_rows_tables_and_names():
    spec = study.get_study(NAME)
    assert spec.rows[0] == ev.MAIN_ROW and spec.rows[1] == ev.FACTORS_ROW
    assert set(spec.tables) == {"valuations", "members"}
    assert ev.NOREFIT == "tlrn_cr_norefit" and ev.CONTROL_ROW in spec.rows
    assert ev.EXPECTED_PAIRS is study.EXPECTED_PAIRS_BY_DATE


# -- the command line --------------------------------------------------------------------------


def test_main_prints_a_smoke_banner_and_writes_the_verdicts(
    tmp_path, monkeypatch, cohort, sets, capsys
):
    populate(tmp_path / "fits", cohort, sets)
    monkeypatch.setattr(study, "load_cohort", lambda name, **kw: cohort)
    monkeypatch.setattr(study, "cells_from_mart", lambda years=None: cohort.all_cells)
    monkeypatch.setattr(ev, "third_study_tie_out", lambda *a, **k: "not compared: synthetic")
    # the fits are registered, not smoke ones: with --smoke nothing is found and nothing is written
    out = tmp_path / "out"
    nothing = ["--tie-out", str(tmp_path / "no_tie_out")]
    code = ev.main(
        ["--fits", str(tmp_path / "fits"), "--cohort", "B", "--out", str(out), "--smoke", *nothing]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "SMOKE RUN" in text and "(SMOKE, not a result)" in text
    assert json.loads((out / "verdicts.json").read_text())["smoke"] is True
    code = ev.main(
        ["--fits", str(tmp_path / "fits"), "--cohort", "B", "--out", str(out / "r"), *nothing]
    )
    assert code == 0
    verdicts = json.loads((out / "r" / "verdicts.json").read_text())
    assert verdicts["smoke"] is False and verdicts["verdicts"]["B"]["p1"]["verdict"] == "helps"
    assert (out / "r" / "report.txt").exists()


# -- the rules the review held the evaluator to --------------------------------------------------


def test_a_tie_out_that_cannot_be_made_stops_outside_a_smoke_run(tmp_path):
    pools = {"mack": np.array([0.1, 0.1, 0.1, 0.05481]), "mcl": np.array([0.1, 0.1, 0.1, 0.0541])}
    good = {"left_out": {}, "benchmark_pool_ape": pools}
    with pytest.raises(SystemExit, match="must tie out and NOT TIED OUT: no B_metrics"):
        ev.third_study_tie_out("B", good, tmp_path)
    with pytest.raises(SystemExit, match="must tie out and NOT TIED OUT"):
        ev.third_study_tie_out("B", good, None)
    pd.DataFrame({"method": ["mack"], "valuation": ["10"], "pool_ape": [0.05481]}).to_csv(
        tmp_path / "B_metrics_by_valuation.csv", index=False
    )
    with pytest.raises(SystemExit, match="has no mcl row at valuation 10"):
        ev.third_study_tie_out("B", good, tmp_path)
    with pytest.raises(SystemExit, match="some companies are left out"):
        ev.third_study_tie_out("B", {**good, "left_out": {"x": 1}}, tmp_path)


def test_a_smoke_run_only_notes_a_tie_out_it_cannot_make(tmp_path):
    good = {"left_out": {}, "benchmark_pool_ape": {}}
    got = ev.third_study_tie_out("B", good, tmp_path, smoke=True)
    assert "NOT TIED OUT" in got


def test_run_cohort_holds_a_full_run_to_the_tie_out(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path, cohort, sets)
    monkeypatch.setattr(study, "load_cohort", lambda name, **kw: cohort)
    with pytest.raises(SystemExit, match="must tie out"):
        ev.run_cohort("B", tmp_path, cells=cohort.all_cells, log=lambda *_: None)


def evaluate_for_main(monkeypatch, cohort):
    monkeypatch.setattr(study, "load_cohort", lambda name, **kw: cohort)
    monkeypatch.setattr(study, "cells_from_mart", lambda years=None: cohort.all_cells)
    monkeypatch.setattr(ev, "third_study_tie_out", lambda *a, **k: "not compared: synthetic")


def test_a_stop_and_ask_reading_writes_no_result_until_ethan_has_answered(
    tmp_path, monkeypatch, cohort, sets, capsys
):
    # 20 points from the target with no refit effect: the third reading
    populate(tmp_path / "fits", cohort, sets, refit_error=0.2, norefit_error=0.2)
    evaluate_for_main(monkeypatch, cohort)
    out = tmp_path / "out"
    base = ["--fits", str(tmp_path / "fits"), "--cohort", "B", "--out", str(out)]
    assert ev.main(base) == ev.STOP_EXIT_CODE
    text = capsys.readouterr().out
    assert ev.STOP_AND_ASK in text and "NO RESULT WRITTEN" in text
    written = {p.name for p in out.glob("*")} if out.exists() else set()
    assert not {n for n in written if not n.startswith("benchmarks")}  # only the benchmark cache

    assert ev.main([*base, "--stop-answered", "Ethan, 2026-11-05: write them up"]) == 0
    verdicts = json.loads((out / "verdicts.json").read_text())
    assert verdicts["stop_answered"] == "Ethan, 2026-11-05: write them up"
    assert verdicts["verdicts"]["B"]["replication"]["stop_and_ask"] is True
    assert (out / "report.txt").exists() and (out / "B_metrics_by_valuation.csv").exists()


def test_a_result_that_needs_no_answer_records_none(tmp_path, monkeypatch, cohort, sets):
    populate(tmp_path / "fits", cohort, sets)
    evaluate_for_main(monkeypatch, cohort)
    out = tmp_path / "out"
    assert ev.main(["--fits", str(tmp_path / "fits"), "--cohort", "B", "--out", str(out)]) == 0
    assert "stop_answered" not in json.loads((out / "verdicts.json").read_text())


def test_a_draw_average_in_the_unread_band_takes_the_label_rule_and_does_not_stop():
    for m in (2.4, 2.86):
        got = reading(m)
        assert got["stop"] is False
        assert any("without falling in a band" in line for line in got["lines"])
