"""The fourth tlrn study's runner: the revised model, its rows, its stops and its reuse.

Nothing here trains a real network except one tiny fit, which checks that ``paid_only`` reaches
the features. The tests drive the things whose mistakes would cost a Colab afternoon or quietly
change an answer: the rows' configurations written out and compared with the main row, the
settings a record carries (and what Study 3's records still carry), the rule that a single-draw
row is fitted once on one cohort, the two pre-registered stops, the retry and the failure file,
the reuse of the third study's network-off fits, the run record's environment fields, and the
pre-fit checks.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import tlrn_colab  # noqa: E402
import tlrn_study_colab as study  # noqa: E402

from .test_tlrn_study3 import BUDGET, cohort_from, synthetic_cells  # noqa: E402

NAME = "companion_revised"
PREDICTED = ("predicted", "predicted_selected", "predicted_net", "predicted_net_selected")


def rows(members=20, smoke=False):
    return {r: (roles, cfg) for r, roles, cfg in tlrn_colab.configs(NAME, members, smoke)}


def differing(a, b) -> set:
    x, y = asdict(a), asdict(b)
    return {k for k in x if x[k] != y[k]}


# -- the rows ---------------------------------------------------------------------------------


def test_the_rows_come_in_the_preregistered_order():
    expected = (
        "tlrn_cr_refit",
        "tlrn_cr_factors",
        "tlrn_cr_noline",
        "tlrn_cr_noay",
        "tlrn_cr_paid",
        "tlrn_cf_rerun",
    )
    assert list(rows()) == list(expected) == list(study.COMPANION_REVISED_ROWS)
    assert study.get_study(NAME).rows == expected


def test_the_revised_config_is_the_final_one_with_only_the_embedding_removed():
    revised, final = tlrn_colab.companion_revised_config(), tlrn_colab.companion_final_config()
    assert differing(revised, final) == {"maturity_embedding"}
    assert revised.maturity_embedding is False and final.maturity_embedding is True
    # an override still reaches the config, and the default is not an override's to flip back
    assert tlrn_colab.companion_revised_config(ensemble_size=3).ensemble_size == 3
    assert tlrn_colab.companion_revised_config().maturity_embedding is False


def test_the_main_row_is_the_revised_config_and_the_rest_differ_as_registered():
    r = rows()
    main = r["tlrn_cr_refit"][1]
    assert main == tlrn_colab.companion_revised_config()
    assert differing(main, r["tlrn_cr_factors"][1]) == {"factors_only"}
    assert r["tlrn_cr_factors"][1].factors_only is True
    assert differing(main, r["tlrn_cr_noline"][1]) == {"attention", "cross_line"}
    assert (
        r["tlrn_cr_noline"][1].attention == ("lag", "ay") and not r["tlrn_cr_noline"][1].cross_line
    )
    assert differing(main, r["tlrn_cr_noay"][1]) == {"attention"}
    assert r["tlrn_cr_noay"][1].attention == ("line", "lag")
    assert differing(main, r["tlrn_cr_paid"][1]) == set()  # paid-only is a feature option
    control = r["tlrn_cf_rerun"][1]
    assert control == tlrn_colab.companion_final_config() and control.maturity_embedding is True
    # every other row has the embedding off
    for name in (
        "tlrn_cr_refit",
        "tlrn_cr_factors",
        "tlrn_cr_noline",
        "tlrn_cr_noay",
        "tlrn_cr_paid",
    ):
        assert r[name][1].maturity_embedding is False, name


def test_only_the_paid_row_has_the_paid_only_role():
    for name, (roles, _) in rows().items():
        assert roles.get("paid_only", False) is (name == "tlrn_cr_paid"), name


def test_the_smoke_budget_reaches_every_row():
    for name, (_, cfg) in rows(members=3, smoke=True).items():
        assert (cfg.max_epochs, cfg.ensemble_size) == (10, 3), name


def test_the_earlier_studies_rows_are_untouched():
    assert [r for r, _, _ in tlrn_colab.configs("companion_final", 20, False)] == [
        "tlrn_cf_refit",
        "tlrn_cf_factors",
    ]
    assert tlrn_colab.companion_final_config().maturity_embedding is True


# -- the registry and the settings -----------------------------------------------------------


def test_the_registry_names_the_fourth_study():
    s = study.get_study(NAME)
    assert (s.version, s.cohorts, s.per_valuation, s.stop_rules) == (4, ("B", "C"), True, True)
    assert s.tables == ("valuations", "members")
    assert s.pinned_ibnr == "0.7.7" and study.PINNED_IBNR == "0.7.6"
    assert s.single_draw_rows == ("tlrn_cr_noline", "tlrn_cr_noay", "tlrn_cr_paid", "tlrn_cf_rerun")
    assert s.single_draw_cohorts == ("B",)
    assert not study.get_study("companion_final").stop_rules
    assert study.STOP_FIRST_FIT_SECONDS == 1500 and study.BUDGET_HOURS == 8.0


def test_settings_carry_the_study_the_embedding_and_paid_only():
    cohort = cohort_from(synthetic_cells(spike=False))
    r = rows()
    got = {
        name: study.fit_settings(cohort, name, *r[name], draw=1, smoke=False, study=NAME)
        for name in r
    }
    for name, settings in got.items():
        assert settings["study"] == 4 and settings["study_name"] == NAME
        assert settings["seed"] == 11 + 100_003
        assert settings["paid_only"] is (name == "tlrn_cr_paid")
        assert settings["config"]["maturity_embedding"] is (name == "tlrn_cf_rerun")
    assert (
        got["tlrn_cr_noline"]["config"]["attention"] != got["tlrn_cr_refit"]["config"]["attention"]
    )


def test_the_third_studys_settings_do_not_gain_a_field_so_its_saved_fits_still_resume():
    cohort = cohort_from(synthetic_cells(spike=False))
    roles, cfg = {r: (ro, c) for r, ro, c in study.row_configs(20, False, "companion_final")}[
        "tlrn_cf_refit"
    ]
    got = study.fit_settings(
        cohort, "tlrn_cf_refit", roles, cfg, draw=0, smoke=False, study="companion_final"
    )
    assert "maturity_embedding" not in got["config"] and "paid_only" not in got
    # the fourth study's settings do list it, even when True, so a rerun row cannot pass for another
    again = study.fit_settings(
        cohort, "tlrn_cf_rerun", *rows()["tlrn_cf_rerun"], draw=0, smoke=False, study=NAME
    )
    assert again["config"]["maturity_embedding"] is True


def test_the_fourth_studys_records_are_never_read_as_the_thirds(tmp_path):
    cohort = cohort_from(synthetic_cells(spike=False))
    for name in ("companion_final", NAME):
        row = study.get_study(name).rows[0]
        roles, cfg = {r: (ro, c) for r, ro, c in study.row_configs(20, False, name)}[row]
        settings = study.fit_settings(cohort, row, roles, cfg, draw=0, smoke=False, study=name)
        (tmp_path / f"{name}-ibnr9.9.9.json").write_text(
            json.dumps({"settings": settings, "run": {}}), encoding="utf-8"
        )
    assert len(study.collect_records(tmp_path, smoke=False, study=NAME)) == 1
    assert len(study.collect_records(tmp_path, smoke=False, study="companion_final")) == 1


# -- single-draw rows ------------------------------------------------------------------------


def plan(tmp_path, cohorts, rows_, draws, **kw):
    return study.plan_study(
        tmp_path,
        cohorts,
        rows_,
        draws,
        members=2,
        smoke=kw.pop("smoke", True),
        version="t",
        study=NAME,
        **kw,
    )


def test_a_single_draw_row_is_planned_once_on_cohort_b_whatever_the_draw_count(tmp_path):
    cohorts = {
        "B": cohort_from(synthetic_cells(spike=False), "B"),
        "C": cohort_from(synthetic_cells(spike=False), "C"),
    }
    todo, _ = plan(tmp_path, cohorts, study.get_study(NAME).rows, 3)
    count = {}
    for item in todo:
        count.setdefault(item["row"], []).append((item["cohort"], item["draw"]))
    for row in study.get_study(NAME).single_draw_rows:
        assert count[row] == [("B", 0)], row
    assert len(count["tlrn_cr_refit"]) == 6 and len(count["tlrn_cr_factors"]) == 6


def test_an_explicit_request_for_a_single_draw_row_with_more_draws_or_cohort_c_is_refused():
    spec = study.get_study(NAME)
    with pytest.raises(ValueError, match="tlrn_cr_paid is fitted once, at draw 0.*--draws 2"):
        study.check_row_request(spec, ["tlrn_cr_paid"], ["B"], 2)
    with pytest.raises(ValueError, match="tlrn_cr_noay is fitted on cohort B only.*C is refused"):
        study.check_row_request(spec, ["tlrn_cr_noay"], ["B", "C"], 1)
    study.check_row_request(spec, ["tlrn_cr_noay", "tlrn_cr_refit"], ["B"], 1)
    study.check_row_request(spec, ["tlrn_cr_refit"], ["B", "C"], 10)  # a verdict row is free


def test_main_refuses_a_single_draw_row_with_two_draws(tmp_path):
    with pytest.raises(SystemExit):
        study.main(
            ["--out", str(tmp_path), "--study", NAME, "--cohort", "B", "--smoke",
             "--rows", "tlrn_cr_paid", "--draws", "2"]
        )  # fmt: skip


# -- the stops -------------------------------------------------------------------------------


def test_the_budget_stop_fires_at_eight_hours_and_not_before():
    assert study.check_budget(7.9 * 3600) == pytest.approx(7.9)
    with pytest.raises(SystemExit, match="STOP: the saved fits add up to 8.00 GPU hours"):
        study.check_budget(8 * 3600)


def test_the_first_fit_stop_fires_just_over_25_minutes():
    assert study.check_first_fit(1500.0) == 1500.0
    with pytest.raises(SystemExit, match="STOP: draw 0 on C4 took 25.0 minutes"):
        study.check_first_fit(1500.1)
    assert study.is_first_fit({"cohort": "C", "row": "tlrn_cr_refit", "draw": 0})
    for other in (
        ("B", "tlrn_cr_refit", 0),
        ("C", "tlrn_cr_factors", 0),
        ("C", "tlrn_cr_refit", 1),
    ):
        assert not study.is_first_fit(dict(zip(("cohort", "row", "draw"), other, strict=True)))


def write_saved(
    out: Path, cohort_name, row, draw, seconds, *, reused=False, study_name=NAME, smoke=False
):
    stem = f"{cohort_name}_{row}_d{draw:02d}{'_smoke' if smoke else ''}-ibnr0.7.7"
    run = {"fit_seconds": seconds}
    if reused:
        run["reused_from"] = {"record": "x.json"}
    record = {
        "settings": {"study": 4, "study_name": study_name, "smoke": smoke, "cohort": cohort_name,
                     "row": row, "draw": draw},
        "run": run,
    }  # fmt: skip
    (out / f"{stem}.json").write_text(json.dumps(record))


def test_fit_seconds_used_counts_this_studys_full_fits_and_not_a_reused_one(tmp_path):
    write_saved(tmp_path, "C", "tlrn_cr_refit", 0, 1000.0)
    write_saved(tmp_path, "C", "tlrn_cr_refit", 1, 500.0, smoke=True)
    write_saved(tmp_path, "C", "tlrn_cr_refit", 2, 700.0, study_name="companion_final")
    write_saved(tmp_path, "C", "tlrn_cr_factors", 0, 9999.0, reused=True)
    assert study.fit_seconds_used(tmp_path, study=NAME) == 1000.0


def fake_cohort(name):
    return cohort_from(synthetic_cells(spike=False), name)


def stub_fit(seconds=10.0, reserve=100.0, record=None):
    def fit(cohort, row, roles, config, seed):
        if record is not None:
            record.append((cohort.name, row, seed))
        codes = list(cohort.board)
        frame = pd.DataFrame({"company_code": codes, "reserve": [reserve] * len(codes)})
        forecast = {c: [reserve] * len(codes) for c in PREDICTED}
        tables = {
            "valuations": pd.DataFrame({"valuation": 10, "company_code": codes, **forecast}),
            "members": pd.DataFrame(
                {"valuation": [10], "alpha": [0.5], "best_epoch": [5], "refit_epochs": [5]}
            ),
        }
        return SimpleNamespace(frame=frame), {"fit_seconds": seconds}, tables

    return fit


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(study, "reserves_frame", lambda entry: entry.frame)
    monkeypatch.setattr(study, "save_entry", lambda *a, **k: None)


def run_stub(out, fit_fn, *, cohorts=None, rows_=("tlrn_cr_refit",), draws=1, smoke=False, **kw):
    return study.run_study(
        out,
        cohorts or {"C": fake_cohort("C")},
        rows_,
        draws,
        members=2,
        smoke=smoke,
        version="0.7.7",
        fit_fn=fit_fn,
        study=NAME,
        log=lambda *_: None,
        **kw,
    )


def test_run_study_stops_before_a_fit_when_the_saved_seconds_reach_the_budget(tmp_path, stubbed):
    write_saved(tmp_path, "B", "tlrn_cr_refit", 0, 8 * 3600.0)
    calls = []
    with pytest.raises(SystemExit, match="STOP: the saved fits add up to 8.00"):
        run_stub(tmp_path, stub_fit(record=calls))
    assert calls == []


def test_a_smoke_run_is_not_stopped_by_the_budget(tmp_path, stubbed):
    write_saved(tmp_path, "B", "tlrn_cr_refit", 0, 9 * 3600.0)
    assert run_stub(tmp_path, stub_fit(), smoke=True)["fitted"] == 1


def test_run_study_stops_after_a_slow_first_fit_but_saves_it_first(tmp_path, stubbed):
    with pytest.raises(SystemExit, match="STOP: draw 0 on C4"):
        run_stub(tmp_path, stub_fit(seconds=1501.0))
    assert len(list(tmp_path.glob("C_tlrn_cr_refit_d00-ibnr0.7.7.json"))) == 1  # kept, not lost
    # the rule is also applied on a rerun that finds the slow fit already saved
    with pytest.raises(SystemExit, match="STOP: draw 0 on C4"):
        run_stub(tmp_path, stub_fit())


def test_run_study_summary_has_the_new_counts_only_for_the_fourth_study(tmp_path, stubbed):
    got = run_stub(tmp_path, stub_fit())
    assert got == {"fitted": 1, "skipped": 0, "failed": 0, "reused": 0}
    assert run_stub(tmp_path, stub_fit()) == {"fitted": 0, "skipped": 1, "failed": 0, "reused": 0}


# -- the retry and the failure file -----------------------------------------------------------


def test_a_non_finite_fit_is_rerun_once_with_the_same_seed_then_kept_if_it_recovers(
    tmp_path, stubbed
):
    seen = []
    state = {"n": 0}

    def flaky(cohort, row, roles, config, seed):
        seen.append(seed)
        state["n"] += 1
        return stub_fit(reserve=np.nan if state["n"] == 1 else 100.0)(
            cohort, row, roles, config, seed
        )

    got = run_stub(tmp_path, flaky)
    assert got["fitted"] == 1 and got["failed"] == 0
    assert seen == [11, 11]  # the same seed twice


def test_a_fit_that_stays_non_finite_is_reported_in_a_failed_file_and_left_out(tmp_path, stubbed):
    got = run_stub(tmp_path, stub_fit(reserve=np.inf))
    assert got["failed"] == 1 and got["fitted"] == 0
    (note,) = list(tmp_path.glob("FAILED_*.txt"))
    assert "non-finite" in note.read_text() and note.suffix == ".txt"
    assert not list(tmp_path.glob("*.json"))  # nothing marks it finished


def test_a_single_draw_row_that_raises_is_reported_and_the_rest_goes_on(tmp_path, stubbed):
    def fit(cohort, row, roles, config, seed):
        if row == "tlrn_cr_paid":
            raise RuntimeError("out of memory")
        return stub_fit()(cohort, row, roles, config, seed)

    got = run_stub(
        tmp_path, fit, cohorts={"B": fake_cohort("B")}, rows_=("tlrn_cr_paid", "tlrn_cr_noay"),
    )  # fmt: skip
    assert got == {"fitted": 1, "skipped": 0, "failed": 1, "reused": 0}
    assert "out of memory" in next(tmp_path.glob("FAILED_*.txt")).read_text()


def test_a_row_that_carries_a_verdict_does_not_swallow_an_exception(tmp_path, stubbed):
    def fit(*a):
        raise RuntimeError("defect")

    with pytest.raises(RuntimeError, match="defect"):
        run_stub(tmp_path, fit)


def test_resume_skips_what_is_saved_and_refuses_changed_settings(tmp_path, stubbed):
    run_stub(tmp_path, stub_fit())
    calls = []
    assert run_stub(tmp_path, stub_fit(record=calls))["skipped"] == 1 and calls == []
    # the same record name with other settings is refused before any training
    record = next(tmp_path.glob("C_tlrn_cr_refit_d00-ibnr0.7.7.json"))
    body = json.loads(record.read_text())
    body["settings"]["config"]["maturity_embedding"] = True
    record.write_text(json.dumps(body))
    with pytest.raises(SystemExit):
        run_stub(tmp_path, stub_fit(record=calls))
    assert calls == []


# -- the reuse of the third study's factors fits ---------------------------------------------


def make_third_study_folder(folder: Path, cohort, draw=0, **change):
    """A third-study ``tlrn_cf_factors`` fit that matches what this study would plan."""
    third = study.get_study("companion_final")
    row = "tlrn_cf_factors"
    roles, cfg = {r: (ro, c) for r, ro, c in study.row_configs(2, False, "companion_final")}[row]
    settings = study.fit_settings(
        cohort, row, roles, cfg, draw=draw, smoke=False, study="companion_final"
    )
    settings.update(change)
    stem = f"{cohort.name}_{row}_d{draw:02d}-ibnr0.7.6"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.json").write_text(
        json.dumps({"settings": settings, "run": {"fit_seconds": 321.0, "jax": "0.7.1"}})
    )
    pd.DataFrame({"company_code": ["1"], "reserve": [5.0]}).to_csv(
        folder / f"{stem}_reserves.csv", index=False
    )
    for name in third.tables:
        pd.DataFrame({"valuation": [10], "company_code": ["1"], "x": [1.0]}).to_csv(
            folder / f"{stem}_{name}.csv", index=False
        )
    return settings


def planned_factors_item(tmp_path, cohort, reuse_dir):
    todo, _ = study.plan_study(
        tmp_path, {cohort.name: cohort}, ("tlrn_cr_factors",), 1, members=2, smoke=False,
        version="0.7.7", study=NAME, reuse_dir=reuse_dir,
    )  # fmt: skip
    (item,) = todo
    return item


def test_reuse_takes_the_third_studys_tables_and_records_where_they_came_from(tmp_path):
    cohort = cohort_from(synthetic_cells(spike=False))
    third_dir = tmp_path / "third"
    make_third_study_folder(third_dir, cohort)
    item = planned_factors_item(tmp_path / "out", cohort, third_dir)
    assert item["reuse"] is True
    frame, run, tables = study.reuse_factors_fit(item, third_dir)
    assert frame["reserve"].tolist() == [5.0]
    assert set(tables) == {"valuations", "members"} and tables["valuations"]["x"].tolist() == [1.0]
    assert run["fit_seconds"] == 0.0 and run["paid_only"] is False
    source = run["reused_from"]
    assert source["study"] == "companion_final" and source["row"] == "tlrn_cf_factors"
    assert source["ibnr"] == "0.7.6" and source["fit_seconds"] == 321.0
    assert source["run"]["jax"] == "0.7.1" and source["record"].endswith("-ibnr0.7.6.json")


def test_reuse_is_off_without_a_folder_and_in_a_smoke_run(tmp_path):
    cohort = cohort_from(synthetic_cells(spike=False))
    off = planned_factors_item(tmp_path / "a", cohort, None)
    assert off["reuse"] is False
    todo, _ = study.plan_study(
        tmp_path / "b", {"B": cohort}, ("tlrn_cr_factors",), 1, members=2, smoke=True,
        version="t", study=NAME, reuse_dir=tmp_path,
    )  # fmt: skip
    assert todo[0]["reuse"] is False  # a smoke run trains every row


@pytest.mark.parametrize(
    "change, message",
    [
        ({"study": 2}, "study is 2"),
        ({"seed": 5}, "seed is 5"),
        ({"smoke": True}, "smoke is True"),
        ({"cohort_hash": "x" * 16}, "cohort_hash"),
        ({"valuation_hashes": {"6": "a"}}, "valuation_hashes"),
        ({"publish_id": "other"}, "publish_id"),
    ],
)
def test_reuse_refuses_a_fit_that_is_not_this_cohorts_and_draws(tmp_path, change, message):
    cohort = cohort_from(synthetic_cells(spike=False))
    third_dir = tmp_path / "third"
    make_third_study_folder(third_dir, cohort, **change)
    item = planned_factors_item(tmp_path / "out", cohort, third_dir)
    with pytest.raises(SystemExit, match=message):
        study.reuse_factors_fit(item, third_dir)


def test_reuse_refuses_a_fit_with_other_model_settings_but_allows_the_embedding_difference(
    tmp_path,
):
    cohort = cohort_from(synthetic_cells(spike=False))
    third_dir = tmp_path / "third"
    settings = make_third_study_folder(third_dir, cohort)
    item = planned_factors_item(tmp_path / "out", cohort, third_dir)
    assert settings["config"].get("maturity_embedding") is None  # the third study omitted it
    study.reuse_factors_fit(item, third_dir)  # passes: the one field covered by the identity check
    record = next(third_dir.glob("*.json"))
    body = json.loads(record.read_text())
    body["settings"]["config"]["d_model"] = 99
    record.write_text(json.dumps(body))
    with pytest.raises(SystemExit, match="d_model differ"):
        study.reuse_factors_fit(item, third_dir)


def test_reuse_needs_exactly_one_finished_fit(tmp_path):
    cohort = cohort_from(synthetic_cells(spike=False))
    third_dir = tmp_path / "third"
    third_dir.mkdir()
    item = planned_factors_item(tmp_path / "out", cohort, third_dir)
    with pytest.raises(SystemExit, match="expected one finished third-study.*found 0"):
        study.reuse_factors_fit(item, third_dir)


def test_run_study_writes_a_reused_fit_with_zero_seconds_and_counts_it(tmp_path, stubbed):
    cohort = cohort_from(synthetic_cells(spike=False))
    third_dir = tmp_path / "third"
    make_third_study_folder(third_dir, cohort)
    cohort.board = ["1"]
    cohort.actual = pd.DataFrame(
        {"actual_reserve": [100.0]}, index=pd.Index(["1"], name="company_code")
    )
    out = tmp_path / "out"
    out.mkdir()
    calls = []
    got = study.run_study(
        out, {"B": cohort}, ("tlrn_cr_factors",), 1, members=2, smoke=False, version="0.7.7",
        fit_fn=stub_fit(record=calls), study=NAME, log=lambda *_: None, reuse_dir=third_dir,
    )  # fmt: skip
    assert got["reused"] == 1 and got["fitted"] == 0 and calls == []
    record = json.loads(next(out.glob("B_tlrn_cr_factors_d00-ibnr0.7.7.json")).read_text())
    assert record["run"]["fit_seconds"] == 0.0 and "reused_from" in record["run"]
    assert record["settings"]["study"] == 4  # the record is this study's, with its own settings
    assert study.fit_seconds_used(out, study=NAME) == 0.0


# -- the run record and the pre-fit checks ---------------------------------------------------


def test_the_library_versions_are_the_installed_ones():
    import duckdb

    got = study.library_versions()
    assert got == {"pandas": pd.__version__, "numpy": np.__version__, "duckdb": duckdb.__version__}


def test_main_puts_the_versions_and_paid_only_in_every_run_record_and_threads_paid_only(
    tmp_path, monkeypatch
):
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from .test_tlrn_study3 import fake_fits

    sent, built = [], []

    def fake_fit_valuations(cohort, config, seed, **kw):
        sent.append({"config": config, "seed": seed, **kw})
        return fake_fits(seed)

    def build(name, **kw):
        built.append(name)
        return SimpleNamespace(
            name=name, pairs_hash="p" * 16, data_hash="d" * 16, board=["1", "2", "3"],
            actual=pd.DataFrame({"actual_reserve": [100.0, 200.0, 300.0]},
                                index=pd.Index(["1", "2", "3"], name="company_code")),
            source="stub", description="stub", as_of=study.AS_OF[name],
            by_valuation={v: [(name, str(v))] for v in study.VALUATIONS},
        )  # fmt: skip

    checked = []
    monkeypatch.setattr(study, "fit_valuations", fake_fit_valuations)
    monkeypatch.setattr(study, "cells_from_mart", lambda years=None: ("cells", years))
    monkeypatch.setattr(study, "build_cohort", build)
    monkeypatch.setattr(
        study, "prefit_checks", lambda cohorts, **k: checked.append(sorted(cohorts))
    )
    study.main(
        ["--out", str(tmp_path), "--study", NAME, "--cohort", "B", "--smoke", "--draws", "1",
         "--members", "2"]
    )  # fmt: skip
    # every row of the study ran on B at draw 0, in the registry order
    assert checked == [["B"]]
    by_embedding = [(c["config"].factors_only, c["paid_only"]) for c in sent]
    assert len(sent) == 6 and by_embedding.count((False, True)) == 1  # one paid-only row
    for call in sent:
        assert call["backend"] == "jax" and call["seed"] == 11
    records = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("B_*.json"))]
    assert len(records) == 6
    for record in records:
        run = record["run"]
        assert run["pandas"] == pd.__version__ and run["numpy"] == np.__version__
        assert "duckdb" in run and run["paid_only"] is (record["settings"]["row"] == "tlrn_cr_paid")


# -- the paid-only option reaches the features -------------------------------------------------


def test_fit_valuation_passes_paid_only_to_the_feature_builder(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    from ibnr.kernels import nn_features

    cohort = cohort_from(synthetic_cells(spike=False))
    triangle = study.triangle_from_cells(cohort.all_cells, cohort.pairs)
    seen = []
    real = nn_features.tlrn_features

    def spy(*args, **kwargs):
        seen.append(kwargs.get("paid_only"))
        return real(*args, **kwargs)

    monkeypatch.setattr(nn_features, "tlrn_features", spy)
    config = tlrn_colab.companion_revised_config(ensemble_size=1, **BUDGET)
    study.fit_valuation(
        triangle, 10, config, 3, as_of=cohort.as_of, backend="torch", paid_only=True
    )
    assert seen and all(x is True for x in seen)
    seen.clear()
    study.fit_valuation(triangle, 10, config, 3, as_of=cohort.as_of, backend="torch")
    assert seen and all(x is False for x in seen)
    with pytest.raises(ValueError, match="paid_only must be True or False"):
        study.fit_valuation(
            triangle, 10, config, 3, as_of=cohort.as_of, backend="torch", paid_only=1
        )


def test_the_paid_only_channels_are_zero_in_the_features_the_runner_builds():
    from ibnr.kernels.nn_features import FEATURE_NAMES, N_FEATURES_PAID_ONLY

    assert len(FEATURE_NAMES[N_FEATURES_PAID_ONLY:]) == 5
    assert all(("incurred" in n or "case" in n) for n in FEATURE_NAMES[N_FEATURES_PAID_ONLY:])


# -- the revised network ---------------------------------------------------------------------


def test_the_revised_network_has_no_count_table_and_shares_the_starting_weights():
    torch = pytest.importorskip("torch")
    from ibnr.gallery.nn.tlrn.network import TLRNNetwork

    def build(cfg):
        torch.manual_seed(11)
        return TLRNNetwork(
            cfg, n_lines=4, n_lag=10, n_feat=13, n_origin=10, head_init=torch.zeros(4, 10)
        )

    revised, final = (
        build(tlrn_colab.companion_revised_config()),
        build(tlrn_colab.companion_final_config()),
    )
    counts = (
        sum(p.numel() for p in revised.parameters()),
        sum(p.numel() for p in final.parameters()),
    )
    assert (
        counts
        == (18153, 18537)
        == (study.EXPECTED_PARAMETERS["revised"], study.EXPECTED_PARAMETERS["final"])
    )
    assert "nobs_emb.weight" not in revised.state_dict() and "nobs_emb.weight" in final.state_dict()
    final_state = final.state_dict()
    assert all(torch.equal(v, final_state[k]) for k, v in revised.state_dict().items())


def test_prefit_checks_pass_on_the_two_real_cohorts_counts_or_refuse_by_name(monkeypatch):
    pytest.importorskip("torch")
    cohort = SimpleNamespace(name="B")
    monkeypatch.setattr(
        study,
        "valuation_pairs",
        lambda c: {v: [("a", "b")] * n for v, n in study.EXPECTED_PAIRS_BY_DATE["B"].items()},
    )
    found = study.prefit_checks({"B": cohort}, log=lambda *_: None)
    assert found["parameters"] == {"revised": 18153, "final": 18537}
    assert (
        found["pairs"]["B"] == study.EXPECTED_PAIRS_BY_DATE["B"]
        and len(found["paid_only_channels"]) == 5
    )
    monkeypatch.setattr(study, "valuation_pairs", lambda c: {6: [("a", "b")], 7: [], 8: [], 10: []})
    with pytest.raises(
        SystemExit, match="pairs by valuation date.*differ from the pre-registration"
    ):
        study.prefit_checks({"B": cohort}, log=lambda *_: None)
