"""The second tlrn study's runner: the factor-head model, cohort C and the extra tables.

``scripts/tlrn_study_colab.py --study factor_head`` trains the companion study's newest
model (the factor head) ten times on cohorts A, B and C, and saves besides the reserves
three tables per fit that the evaluator reads (``_valuations.csv``, ``_backtest.csv`` and
``_members.csv``). Nothing here fits a network on a GPU. The tests drive the things whose
mistakes would cost a Colab afternoon or quietly change an answer:

* the model's configuration, written out value by value against the companion's constants
  (a change of the library's defaults must not change the model);
* the two rows and what separates them;
* the first study's settings staying exactly what they were, and the second study's records
  never being read as the first's (and the other way round);
* cohort C's counts, its valuation date and its triangle, where the mart is available;
* parameter delivery through ``main``: the study name, the cohort's own valuation date, the
  row's configuration, the budget and the seed each reach ``gallery.fit``;
* resuming, and refusing a fit whose extra tables are missing;
* the valuation tables of a real tiny fit, checked against a route of their own (the
  multivariate chain ladder rebuilt from the features, and a count of the scored cells).
"""

from __future__ import annotations

import datetime as dt
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

# -- the model's configuration ---------------------------------------------------------------

#: the companion study's constants, as its notebook sets them (see the decision note)
COMPANION = {
    "head": "ldf",
    "eps": 0.5,
    "init_step": 0.57,
    "cl_anchor": False,
    "tail_policy": "observed_cl",
    "cross_line": True,
    "attention": None,
    "n_layers": 1,
    "d_model": 32,
    "n_heads": 2,
    "dropout": 0.3,
    "mask": "unwritten_lines",
    "member": "mcl_blend",
    "ensemble_size": 20,
    "keep": None,
    "batch_unit": "example",
    "batch_size": 64,
    "cutoff_sampling": "per_epoch",
    "min_cutoff": 2,
    "val_diagonals": 2,
    "scoring": "reached_cells",
    "w_pe": 0.5,
    "w_mse": 0.1,
    "mse_scale": 0.005,
    "lr": 0.003,
    "lr_phi": 0.03,
    "weight_decay": 0.0,
    "warmup": 20,
    "grad_clip": 1.0,
    "max_epochs": 3000,
    "check_every": 5,
    "calibration": "retrain_per_valuation",
    "calibration_cutoffs": (6, 7, 8),
    "calibration_horizons": (2, 3, 4),
}


def rows(members=20, smoke=False):
    return {r: (roles, cfg) for r, roles, cfg in tlrn_colab.configs("factor_head", members, smoke)}


def test_the_main_row_is_the_companions_model_value_for_value():
    roles, cfg = rows()["tlrn_13_fh"]
    got = asdict(cfg)
    for name, value in COMPANION.items():
        assert got[name] == value, name
    assert cfg.factors_only is False
    assert roles == {"incurred_field": "incurred_loss", "case_field": "case_reserve"}


def test_the_comparison_row_differs_only_by_switching_the_network_off():
    (roles_a, main_cfg), (roles_b, factors_cfg) = rows()["tlrn_13_fh"], rows()["tlrn_13_fh_factors"]
    assert roles_a == roles_b
    a, b = asdict(main_cfg), asdict(factors_cfg)
    assert {k for k in a if a[k] != b[k]} == {"factors_only"}
    assert b["factors_only"] is True


def test_the_row_order_is_the_registry_order():
    assert list(rows()) == ["tlrn_13_fh", "tlrn_13_fh_factors"] == list(study.FACTOR_HEAD_ROWS)


def test_the_smoke_budget_reaches_both_rows_and_the_members_argument_too():
    for _, cfg in rows(members=3, smoke=True).values():
        assert (cfg.max_epochs, cfg.min_epochs, cfg.check_every, cfg.patience) == (10, 0, 5, 100)
        assert cfg.ensemble_size == 3
    for _, cfg in rows(members=7).values():
        assert cfg.max_epochs == 3000 and cfg.ensemble_size == 7


def test_early_stopping_cannot_end_a_member_before_the_last_epoch():
    # patience counts validation checks, so the whole budget has to be needed to exhaust it
    cfg = rows()["tlrn_13_fh"][1]
    assert cfg.patience * cfg.check_every >= cfg.max_epochs
    assert cfg.max_epochs % cfg.check_every == 0


def test_the_first_study_still_gets_its_own_rows():
    configs = {r: (ro, c) for r, ro, c in tlrn_colab.configs("accident_year", 20, False)}
    assert list(configs) == ["tlrn_8_ay", "tlrn_13_ay"]
    assert all(c.head == "premium_lr" for _, c in configs.values())


# -- the study registry and the settings -----------------------------------------------------


def cohort_stub(name="A", *, as_of=None, board=("1", "2", "3")):
    actual = pd.DataFrame(
        {"actual_reserve": [100.0, 200.0, 300.0], "anchor": [1000.0, 2000.0, 3000.0]},
        index=pd.Index(list(board), name="company_code"),
    )
    return SimpleNamespace(
        name=name,
        pairs_hash="p" * 16,
        data_hash="d" * 16,
        board=list(board),
        actual=actual,
        triangle=f"triangle {name}",
        source="stub",
        description=f"stub {name}",
        as_of=as_of or study.AS_OF[name],
    )


def settings(cohort, row="tlrn_13_fh", draw=0, *, which="factor_head", members=20, smoke=False):
    roles, cfg = {r: (ro, c) for r, ro, c in study.row_configs(members, smoke, which)}[row]
    return study.fit_settings(cohort, row, roles, cfg, draw=draw, smoke=smoke, study=which)


def test_the_registry_names_each_studys_cohorts_rows_and_tables():
    first, second = study.get_study("accident_year"), study.get_study("factor_head")
    assert (first.cohorts, first.rows, first.tables) == (("A", "B"), study.ROWS, ())
    assert second.cohorts == ("A", "B", "C")
    assert second.rows == ("tlrn_13_fh", "tlrn_13_fh_factors")
    assert second.tables == ("valuations", "backtest", "members")
    assert first.version != second.version
    with pytest.raises(ValueError, match="study must be one of"):
        study.get_study("nonsense")
    # the first study's public constants are untouched
    assert study.COHORTS == ("A", "B") and study.ROWS == ("tlrn_8_ay", "tlrn_13_ay")


def test_the_first_studys_settings_are_what_they_were_before_the_second_existed():
    got = settings(cohort_stub("A"), "tlrn_8_ay", which="accident_year")
    assert got["study"] == 1 and "study_name" not in got
    assert set(got) == {
        "study",
        "cohort",
        "row",
        "draw",
        "seed",
        "smoke",
        "config",
        "roles",
        "backend",
        "as_of",
        "loss_field",
        "publish_id",
        "cohort_hash",
        "data_hash",
    }
    assert got["as_of"] == "2007-12-31"


def test_the_second_studys_settings_say_which_study_and_the_cohorts_own_date():
    a = settings(cohort_stub("A"))
    c = settings(cohort_stub("C"))
    assert a["study"] == 2 and a["study_name"] == "factor_head"
    assert a["as_of"] == "2007-12-31" and c["as_of"] == "1997-12-31"
    assert c["cohort"] == "C" and c["seed"] == 11
    assert settings(cohort_stub("A"), draw=3)["seed"] == 11 + 3 * 100_003
    # nothing a first-study record could be mistaken for: every key of one studys settings
    # that differs from the other is in the comparison
    differ = study.settings_differences(
        a, settings(cohort_stub("A"), "tlrn_8_ay", which="accident_year")
    )
    assert "study" in differ and "study_name" in differ and "row" in differ
    # the main row and the comparison row are different fits
    assert settings(cohort_stub("A"), "tlrn_13_fh_factors") != a
    assert "config.factors_only" in study.settings_differences(
        a, settings(cohort_stub("A"), "tlrn_13_fh_factors")
    )


def test_cohort_c_is_the_meyers_window_valued_at_the_end_of_1997():
    assert study.YEARS["C"] == (1988, 1997)
    assert study.AS_OF["C"] == dt.date(1997, 12, 31)
    assert study.CUTOFF["C"] == 10
    assert study.EXPECTED["C"] == {"audit": [675, 405, 322, 232], "companies": 83, "board": 83}
    assert study.COHORT_HASH["C"] == "4a613cc4cb5f0f94"
    # A and B keep their dates and years
    assert study.AS_OF["A"] == study.AS_OF["B"] == dt.date(2007, 12, 31)
    assert study.YEARS["A"] == study.YEARS["B"] == (1998, 2007)


def test_cohort_c_cannot_be_built_from_the_cas_files():
    with pytest.raises(ValueError, match="--cas-dir is for cohort B"):
        study.load_cohort("C", cas_dir="anywhere")
    with pytest.raises(ValueError, match="cohort must be one of"):
        study.load_cohort("D")


def test_cohort_c_from_the_mart_reproduces_its_counts_and_is_valued_in_1997():
    try:
        cohort = study.build_cohort("C")
    except Exception as err:  # the Schedule P mart is not in the local cache
        pytest.skip(f"the Schedule P mart is not available: {err}")
    assert cohort.audit == [675, 405, 322, 232]
    assert (len(cohort.companies), len(cohort.pairs), len(cohort.board)) == (83, 232, 83)
    assert cohort.pairs_hash == "4a613cc4cb5f0f94"
    assert cohort.as_of == dt.date(1997, 12, 31)
    assert sorted(cohort.triangle.origins) == [dt.date(y, 1, 1) for y in range(1988, 1998)]
    # the actual reserve is the deepest paid minus the cumulative paid on the 1997 diagonal,
    # and the dominant company is the one the other two cohorts have
    reserve = cohort.actual["actual_reserve"]
    assert reserve.idxmax() == "1767" and 0.6 < reserve.max() / reserve.sum() < 0.7
    # the lower triangle is real later development: every pair has all 100 cells
    assert len(cohort.cells) == 100 * len(cohort.pairs)
    assert cohort.cells["statement_year"].max() == 2006


# -- the files of a fit and the runner -------------------------------------------------------


class FakeEntry:
    """Just enough of a fitted TLRN for the runner: a few attributes, picklable."""

    def __init__(self, seed: int, codes=("1", "2", "3")):
        rng = np.random.default_rng(seed)
        self._codes = list(codes)
        self.members = rng.uniform(80.0, 320.0, size=(2, len(codes)))
        self.company_size_ = np.array([10.0, 20.0, 30.0])
        self.contract_ = {"line_mask": np.array([[True, True], [True, False], [True, True]])}

    def cohorts(self):
        return [{"company_code": c} for c in self._codes]

    def member_company_reserves(self):
        return self.members

    def company_reserves(self):
        return self.members.mean(axis=0)

    def company_anchors(self):
        return np.array([1000.0, 2000.0, 3000.0])


def fake_tables(seed: int = 0) -> dict[str, pd.DataFrame]:
    return {
        "valuations": pd.DataFrame(
            {
                "valuation": [10, 10, 10, 6, 6, 6],
                "company_code": ["1", "2", "3"] * 2,
                "n_cells": [90, 45, 90, 28, 14, 28],
                "predicted": np.arange(6.0) + seed,
                "predicted_net": np.arange(6.0) + 0.5,
                "actual": [np.nan] * 3 + [1.0, 2.0, 3.0],
            }
        ),
        "backtest": pd.DataFrame(
            {"unit": [0, 1], "cutoff": [6, 6], "horizon": [4, 4], "company_code": ["1", "2"]}
        ),
        "members": pd.DataFrame({"valuation": [10, 10], "member": [0, 1], "alpha": [0.5, 0.7]}),
    }


class Recorder:
    """A fit_fn that records its calls, can fail on one, and returns the extra tables."""

    def __init__(self, fail_on=None, tables=True):
        self.calls, self.fail_on, self.tables = [], fail_on, tables

    def __call__(self, cohort, row, roles, config, seed):
        if self.fail_on == len(self.calls):
            raise KeyboardInterrupt("the runtime disconnected")
        self.calls.append((cohort.name, row, seed, config.factors_only))
        out = (FakeEntry(seed), {"fit_seconds": 1.5, "jax_backend": "gpu"})
        return (*out, fake_tables(seed)) if self.tables else out


def run(out, fit_fn, *, draws=2, members=20, cohorts=None, which="factor_head", **kw):
    cohorts = cohorts or {"A": cohort_stub("A"), "C": cohort_stub("C")}
    return study.run_study(
        out,
        cohorts,
        kw.pop("rows", study.get_study(which).rows),
        draws,
        members=members,
        smoke=kw.pop("smoke", False),
        version="9.9.9",
        fit_fn=fit_fn,
        log=lambda *_: None,
        study=which,
        **kw,
    )


def test_a_fit_is_saved_as_a_reserves_file_three_tables_an_entry_and_a_record(tmp_path):
    run(tmp_path, Recorder(), draws=1, cohorts={"C": cohort_stub("C")}, rows=["tlrn_13_fh"])
    names = sorted(p.name for p in tmp_path.iterdir())
    stem = "C_tlrn_13_fh_d00-ibnr9.9.9"
    assert names == sorted(
        [
            f"{stem}.json",
            f"{stem}.pkl",
            f"{stem}_reserves.csv",
            f"{stem}_valuations.csv",
            f"{stem}_backtest.csv",
            f"{stem}_members.csv",
            "manifest_factor_head.json",
        ]
    )
    record = json.loads((tmp_path / f"{stem}.json").read_text(encoding="utf-8"))
    assert record["settings"]["study_name"] == "factor_head" and record["settings"]["study"] == 2
    assert record["settings"]["as_of"] == "1997-12-31"
    valuations = pd.read_csv(tmp_path / f"{stem}_valuations.csv", dtype={"company_code": str})
    assert list(valuations.columns) == [
        "valuation",
        "company_code",
        "n_cells",
        "predicted",
        "predicted_net",
        "actual",
    ]
    assert valuations.loc[valuations["valuation"] == 10, "actual"].isna().all()
    manifest = json.loads((tmp_path / "manifest_factor_head.json").read_text(encoding="utf-8"))
    assert manifest["study_name"] == "factor_head" and list(manifest["fits"]) == [stem]


def test_a_second_run_fits_nothing_and_an_interrupted_one_resumes(tmp_path):
    flaky = Recorder(fail_on=3)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, flaky)
    assert len(flaky.calls) == 3
    resumed = Recorder()
    assert run(tmp_path, resumed) == {"fitted": 5, "skipped": 3}
    # draw 0 for every cohort and row before draw 1 starts, and the seeds follow the draws
    assert [(c, r) for c, r, *_ in flaky.calls] == [
        ("A", "tlrn_13_fh"),
        ("A", "tlrn_13_fh_factors"),
        ("C", "tlrn_13_fh"),
    ]
    assert [seed for _, _, seed, _ in resumed.calls] == [11] + [11 + 100_003] * 4
    assert run(tmp_path, Recorder()) == {"fitted": 0, "skipped": 8}
    assert not list(tmp_path.glob("*.part"))


def test_the_comparison_row_is_the_one_fitted_with_the_network_off(tmp_path):
    fit = Recorder()
    run(tmp_path, fit, draws=1, cohorts={"A": cohort_stub("A")})
    assert fit.calls == [("A", "tlrn_13_fh", 11, False), ("A", "tlrn_13_fh_factors", 11, True)]


def test_a_fit_missing_one_of_its_tables_is_refused_not_reused(tmp_path):
    cohorts = {"A": cohort_stub("A")}
    run(tmp_path, Recorder(), draws=1, cohorts=cohorts, rows=["tlrn_13_fh"])
    record = tmp_path / "A_tlrn_13_fh_d00-ibnr9.9.9.json"
    for name in ("valuations", "backtest", "members"):
        table = record.with_name(record.stem + f"_{name}.csv")
        saved = table.read_bytes()
        table.unlink()
        again = Recorder()
        with pytest.raises(SystemExit, match=f"_{name}.csv is missing"):
            run(tmp_path, again, draws=1, cohorts=cohorts, rows=["tlrn_13_fh"])
        assert again.calls == []
        table.write_bytes(saved)
    assert run(tmp_path, Recorder(), draws=1, cohorts=cohorts, rows=["tlrn_13_fh"])["fitted"] == 0


def test_tables_with_no_record_are_refused(tmp_path):
    cohorts = {"A": cohort_stub("A")}
    run(tmp_path, Recorder(), draws=1, cohorts=cohorts, rows=["tlrn_13_fh"])
    record = tmp_path / "A_tlrn_13_fh_d00-ibnr9.9.9.json"
    record.unlink()
    with pytest.raises(SystemExit, match="exist with no settings record"):
        run(tmp_path, Recorder(), draws=1, cohorts=cohorts, rows=["tlrn_13_fh"])


def test_the_record_is_written_last_so_a_failed_table_leaves_no_finished_fit(tmp_path, monkeypatch):
    real = study.save_reserves
    calls = {"n": 0}

    def flaky(frame, path):
        calls["n"] += 1
        if calls["n"] == 3:  # the reserves file and the first table are written, then it dies
            raise OSError("the disk filled up")
        return real(frame, path)

    monkeypatch.setattr(study, "save_reserves", flaky)
    with pytest.raises(OSError):
        run(tmp_path, Recorder(), draws=1, cohorts={"A": cohort_stub("A")}, rows=["tlrn_13_fh"])
    assert not list(tmp_path.glob("*.json"))
    monkeypatch.setattr(study, "save_reserves", real)
    with pytest.raises(SystemExit, match="no settings record"):
        run(tmp_path, Recorder(), draws=1, cohorts={"A": cohort_stub("A")}, rows=["tlrn_13_fh"])


def test_a_fit_function_that_forgets_the_tables_or_adds_one_is_refused(tmp_path):
    with pytest.raises(ValueError, match="saves the tables"):
        run(tmp_path, Recorder(tables=False), draws=1, cohorts={"A": cohort_stub("A")})
    # the first study takes no tables, and says so when handed some
    with pytest.raises(ValueError, match="saves the tables"):
        run(
            tmp_path / "first",
            Recorder(),
            draws=1,
            cohorts={"A": cohort_stub("A")},
            which="accident_year",
        )


def test_settings_that_differ_stop_the_second_study_before_any_training(tmp_path):
    run(tmp_path, Recorder(), draws=1, cohorts={"A": cohort_stub("A")})
    again = Recorder()
    with pytest.raises(SystemExit, match="config.ensemble_size"):
        run(tmp_path, again, draws=1, cohorts={"A": cohort_stub("A")}, members=19)
    with pytest.raises(SystemExit, match="as_of"):
        run(
            tmp_path,
            again,
            draws=1,
            cohorts={"A": cohort_stub("A", as_of=dt.date(2006, 12, 31))},
        )
    assert again.calls == []


def test_the_two_studies_share_a_folder_without_reading_each_others_records(tmp_path):
    cohorts = {"A": cohort_stub("A")}
    run(tmp_path, Recorder(), draws=1, cohorts=cohorts)
    run(tmp_path, Recorder(tables=False), draws=1, cohorts=cohorts, which="accident_year")
    second = study.collect_records(tmp_path, smoke=False, study="factor_head")
    first = study.collect_records(tmp_path, smoke=False, study="accident_year")
    assert sorted(second) == [
        "A_tlrn_13_fh_d00-ibnr9.9.9",
        "A_tlrn_13_fh_factors_d00-ibnr9.9.9",
    ]
    assert sorted(first) == ["A_tlrn_13_ay_d00-ibnr9.9.9", "A_tlrn_8_ay_d00-ibnr9.9.9"]
    assert (tmp_path / "manifest_study.json").exists()
    assert (tmp_path / "manifest_factor_head.json").exists()
    assert len(json.loads((tmp_path / "manifest_study.json").read_text())["fits"]) == 2
    # smoke records of the second study are a third set again
    run(tmp_path, Recorder(), draws=1, cohorts=cohorts, smoke=True)
    assert len(study.collect_records(tmp_path, smoke=True, study="factor_head")) == 2
    assert len(study.collect_records(tmp_path, smoke=False, study="factor_head")) == 2
    assert (tmp_path / "manifest_factor_head_smoke.json").exists()


# -- parameter delivery through main ---------------------------------------------------------


def _main_with_stubs(tmp_path, monkeypatch, argv):
    """Run ``main`` with the cohorts and the fit stubbed; returns what reached gallery.fit."""
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from ibnr import gallery
    from ibnr.gallery.nn.tlrn.model import TLRN

    sent, built = [], []

    def fake_fit(name, triangle, **kw):
        wrapped = getattr(TLRN._finish_protocol, "__name__", "") == "finish"
        sent.append({"name": name, "triangle": triangle, "wrapped": wrapped, **kw})
        return FakeEntry(kw["seed"])

    monkeypatch.setattr(gallery, "fit", fake_fit)
    monkeypatch.setattr(study, "cells_from_mart", lambda years=None: ("cells", years))

    def build(name, **kw):
        built.append((name, kw))
        return cohort_stub(name)

    monkeypatch.setattr(study, "build_cohort", build)
    monkeypatch.setattr(study, "valuation_tables", lambda entry, runs: fake_tables())
    original = TLRN._finish_protocol
    study.main(["--out", str(tmp_path), *argv])
    assert TLRN._finish_protocol is original  # the wrapper is gone once the fit is done
    return sent, built


def test_main_delivers_the_study_the_cohorts_date_the_row_and_the_budget(tmp_path, monkeypatch):
    sent, built = _main_with_stubs(
        tmp_path,
        monkeypatch,
        [
            "--study",
            "factor_head",
            "--cohort",
            "C",
            "--smoke",
            "--draws",
            "2",
            "--members",
            "3",
            "--rows",
            "tlrn_13_fh_factors",
        ],
    )
    assert [b[0] for b in built] == ["C"]
    # cohort C is built from the 1988 to 1997 cells and not from the recent ones
    assert built[0][1]["cells"] == ("cells", (1988, 1997)) and built[0][1]["cas_dir"] is None
    assert len(sent) == 2
    for call, draw in zip(sent, (0, 1), strict=True):
        assert call["name"] == "tlrn" and call["backend"] == "jax"
        assert call["as_of"] == dt.date(1997, 12, 31)
        assert call["seed"] == 11 + 100_003 * draw
        assert call["triangle"] == "triangle C"
        assert call["incurred_field"] == "incurred_loss" and call["case_field"] == "case_reserve"
        assert call["loss_field"] == "paid_loss" and call["show_progress"] is True
        assert call["wrapped"] is True  # the valuation dates are collected during the fit
        cfg = call["config"]
        assert cfg.factors_only is True and cfg.ensemble_size == 3 and cfg.max_epochs == 10
    stem = "C_tlrn_13_fh_factors_d01_smoke-ibnr"
    assert any(
        p.name.startswith(stem) and p.name.endswith("_members.csv") for p in tmp_path.iterdir()
    )
    assert (tmp_path / "manifest_factor_head_smoke.json").exists()


def test_main_runs_a_and_b_at_the_end_of_2007_and_all_means_every_cohort(tmp_path, monkeypatch):
    sent, built = _main_with_stubs(
        tmp_path, monkeypatch, ["--study", "factor_head", "--smoke", "--draws", "1", "--quiet"]
    )
    assert [b[0] for b in built] == ["A", "B", "C"]
    assert [c["as_of"] for c in sent] == [dt.date(2007, 12, 31)] * 4 + [dt.date(1997, 12, 31)] * 2
    assert [b[1]["cells"] for b in built[:2]] == [("cells", None)] * 2
    assert all(c["show_progress"] is False for c in sent)
    # both is A and B
    _, built_both = _main_with_stubs(
        tmp_path / "both",
        monkeypatch,
        ["--study", "factor_head", "--cohort", "both", "--smoke", "--draws", "1"],
    )
    assert [b[0] for b in built_both] == ["A", "B"]


def test_main_keeps_the_first_study_on_its_own_rows_and_refuses_cohort_c_there(
    tmp_path, monkeypatch
):
    sent, built = _main_with_stubs(
        tmp_path, monkeypatch, ["--smoke", "--draws", "1", "--cohort", "both"]
    )
    assert [b[0] for b in built] == ["A", "B"]
    assert len(sent) == 4 and all(c["config"].head == "premium_lr" for c in sent)
    assert not [p for p in tmp_path.iterdir() if "valuations" in p.name]
    assert (tmp_path / "manifest_study_smoke.json").exists()
    with pytest.raises(SystemExit):
        study.main(["--out", str(tmp_path / "x"), "--cohort", "C", "--smoke"])
    with pytest.raises(SystemExit):
        study.main(["--out", str(tmp_path / "y"), "--study", "factor_head", "--rows", "tlrn_8_ay"])


# -- the valuation tables of a real tiny fit -------------------------------------------------


@pytest.fixture(scope="module")
def tiny_fits():
    """Two real fits (torch, a few epochs): one member with the network, two without it."""
    pytest.importorskip("torch")
    from ibnr import gallery

    from .test_nn_features import START, study_arrays, study_triangle

    triangle = study_triangle("duckdb", study_arrays(n_companies=5, n_w=10, seed=3))
    budget = {
        "d_model": 8,
        "dropout": 0.0,
        "max_epochs": 4,
        "min_epochs": 0,
        "check_every": 2,
        "patience": 5,
        "batch_size": 8,
        "n_strata": 1,
        "min_per_stratum": 3,
        "n_draws": 20,
        "warmup": 1,
    }
    out = {}
    for label, cfg in (
        ("main", tlrn_colab.factor_head_config(ensemble_size=1, **budget)),
        ("factors", tlrn_colab.factor_head_config(ensemble_size=2, factors_only=True, **budget)),
    ):
        with study.capture_protocols() as runs:
            entry = gallery.fit(
                "tlrn",
                triangle,
                loss_field="paid_loss",
                incurred_field="incurred_loss",
                case_field="case_reserve",
                as_of=f"{START + 9}-12-31",
                config=cfg,
                seed=7,
            )
        out[label] = (entry, runs, study.valuation_tables(entry, runs))
    return out


def test_a_real_fit_gives_a_protocol_per_valuation_date_in_fit_order(tiny_fits):
    entry, runs, tables = tiny_fits["main"]
    assert [int(r["final_set"]["cutoff"]) for r in runs] == [10, 6, 7, 8]
    valuations = tables["valuations"]
    assert valuations["valuation"].drop_duplicates().tolist() == [10, 6, 7, 8]
    assert (valuations.groupby("valuation").size() == 5).all()
    assert valuations["company_code"].iloc[:5].tolist() == [
        c["company_code"] for c in entry.cohorts()
    ]


def test_the_last_valuation_is_the_entrys_own_reserve_and_has_no_actual(tiny_fits):
    entry, _, tables = tiny_fits["main"]
    last = tables["valuations"].query("valuation == 10")
    np.testing.assert_allclose(last["predicted"], entry.company_reserves(), rtol=1e-4)
    assert last["actual"].isna().all()
    earlier = tables["valuations"].query("valuation < 10")
    assert earlier["actual"].notna().all()


def test_the_cells_scored_are_the_diagonals_after_the_date_within_its_development_lags(tiny_fits):
    _, _, tables = tiny_fits["main"]
    n = 10
    for v, got in tables["valuations"].groupby("valuation"):
        if v == n:
            per_line = [(w, d) for w in range(n) for d in range(n) if w + d + 1 > n]
        else:
            per_line = [
                (w, d)
                for w in range(n)
                for d in range(n)
                if v < w + d + 1 <= n and d + 1 <= v and w <= v - 1
            ]
        n_lines = 2  # the synthetic companies write both of the fixture's lines
        assert (got["n_cells"] == n_lines * len(per_line)).all(), v


def test_the_blend_is_the_chain_ladder_plus_alpha_times_the_networks_difference(tiny_fits):
    # one member, so the company totals are linear in the forecast: predicted equals
    # mcl + alpha * (network - mcl), with the chain ladder rebuilt straight from the features
    from ibnr.gallery.nn.tlrn.model import TLRN
    from ibnr.kernels.nn_features import tlrn_features

    entry, _, tables = tiny_fits["main"]
    n_c = len(entry.cohorts())
    features = tlrn_features(
        entry.contract_,
        cutoff=10,
        target_lo=11,
        target_hi=19,
        incurred_field="incurred_loss",
        case_field="case_reserve",
        with_mcl=True,
    )
    mcl = TLRN._company_totals(features["mcl_pred"], features, n_c)
    alpha = float(tables["members"].query("valuation == 10")["alpha"].iloc[0])
    last = tables["valuations"].query("valuation == 10")
    expected = mcl + alpha * (last["predicted_net"].to_numpy() - mcl)
    np.testing.assert_allclose(last["predicted"], expected, rtol=1e-4)
    # and the network and the blend are different forecasts unless alpha is one
    assert 0.0 <= alpha <= 1.0
    if alpha < 0.99:
        assert not np.allclose(last["predicted"], last["predicted_net"], rtol=1e-6)


def test_the_backtest_table_is_the_entrys_own_with_the_company_code_added(tiny_fits):
    entry, _, tables = tiny_fits["main"]
    back = tables["backtest"]
    pd.testing.assert_frame_equal(back, entry.backtest_)
    assert "company_code" in back.columns
    assert back["cutoff"].drop_duplicates().tolist() == [6, 7, 8]
    assert back["horizon"].drop_duplicates().tolist() == [4, 3, 2]


def test_the_members_table_has_one_row_per_valuation_and_member_with_alpha(tiny_fits):
    _, _, tables = tiny_fits["factors"]
    members = tables["members"]
    assert members.groupby("valuation").size().to_dict() == {6: 2, 7: 2, 8: 2, 10: 2}
    assert {"member", "best_epoch", "epochs_run", "alpha", "kept"} <= set(members.columns)
    assert members["alpha"].between(0.0, 1.0).all() and members["kept"].all()


def _raw_total_after_moving_the_network(run, model, n_c):
    """The raw forecast's company totals, before and after the network weights are shifted."""
    import copy

    import torch

    from ibnr.gallery.nn.tlrn.model import TLRN

    final, tensors, forward = run["final_set"], run["final_tensors"], run["forward"]

    def total(m):
        with torch.no_grad():
            pred = forward(m, tensors, blend=False)["pred"].cpu().numpy()
        return TLRN._company_totals(pred, final, n_c)

    moved = copy.deepcopy(model)
    owned = set(moved.head_param_names)
    with torch.no_grad():
        for name, p in moved.named_parameters():
            if name not in owned:
                p.add_(torch.randn_like(p))
    return total(model), total(moved)


def test_with_the_network_off_its_weights_cannot_move_the_raw_forecast(tiny_fits):
    # the comparison row's flag has to reach the model: moving every weight outside the
    # head leaves the forecast alone with the flag set, and changes it without the flag
    pytest.importorskip("torch")
    for label, moves in (("factors", False), ("main", True)):
        entry, runs, _ = tiny_fits[label]
        n_c = len(entry.cohorts())
        before, after = _raw_total_after_moving_the_network(runs[0], runs[0]["kept"][0], n_c)
        assert np.allclose(before, after, rtol=1e-6) is (not moves), label


def test_the_collected_protocols_are_checked_against_the_entry_before_anything_is_saved(
    tiny_fits,
):
    entry, runs, _ = tiny_fits["main"]
    with pytest.raises(ValueError, match="one protocol per valuation date"):
        study.valuation_tables(entry, runs[:2])
    # a run swapped in from another place is not the entry's own
    other = tiny_fits["factors"][1]
    with pytest.raises(RuntimeError, match="differ from the fitted entry"):
        study.valuation_tables(entry, other)


def test_capturing_changes_nothing_the_fit_computes_and_puts_the_method_back(tiny_fits):
    pytest.importorskip("torch")
    from ibnr.gallery.nn.tlrn.model import TLRN

    entry, runs, _ = tiny_fits["main"]
    assert TLRN._finish_protocol.__name__ == "_finish_protocol"
    # the run the wrapper kept is the very dict the entry read its selection from
    assert runs[0]["selection"].equals(entry.selection_)
    with pytest.raises(ZeroDivisionError), study.capture_protocols():
        raise ZeroDivisionError
    assert TLRN._finish_protocol.__name__ == "_finish_protocol"
