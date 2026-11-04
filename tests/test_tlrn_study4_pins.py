"""The fourth study's Colab notebook agrees with the runner, the evaluator and the rules.

The notebook is read as JSON (no Colab, no network). Its settings, fetch and helper cells are
executed in a throwaway namespace on a tiny fixture, so a helper that stopped enforcing a
pre-registered rule, or a setting that drifted from the runner's, fails here. The fetch cell is
the third study's, unchanged; its behaviour is tested in ``tests/test_tlrn_study3_pins.py``, and
this file holds the two cells word for word equal.
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import tlrn_study4_eval as ev  # noqa: E402
import tlrn_study_colab as study  # noqa: E402

NOTEBOOK = REPO / "analysis" / "06_tlrn_study4_on_colab.ipynb"
THIRD = REPO / "analysis" / "05_tlrn_study3_on_colab.ipynb"
PIN = study.COMPANION_REVISED_PINNED_IBNR


def cells(path=NOTEBOOK):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def source(cell_id: str, path=NOTEBOOK) -> str:
    (found,) = [c for c in cells(path) if c["id"] == cell_id]
    return "".join(found["source"])


def run_cell(cell_id: str, namespace: dict) -> dict:
    exec(compile(source(cell_id), f"<cell {cell_id}>", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def settings() -> dict:
    return run_cell("settings", {})


@pytest.fixture
def helpers(settings):
    return run_cell("helpers", run_cell("fetch", dict(settings)))


def write_record(folder: Path, cohort, row, draw, seconds, *, smoke=False, study_name=None):
    study_name = study_name or "companion_revised"
    stem = f"{cohort}_{row}_d{draw:02d}{'_smoke' if smoke else ''}-ibnr{PIN}"
    record = {
        "settings": {
            "study": 4,
            "study_name": study_name,
            "cohort": cohort,
            "row": row,
            "draw": draw,
            "smoke": smoke,
        },
        "run": {"fit_seconds": seconds, "device_kind": "fixture"},
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.json").write_text(json.dumps(record))
    (folder / f"{stem}_reserves.csv").write_text("company_code\n1\n")
    return stem


# -- the settings and the order ---------------------------------------------------------------


def test_the_pin_is_the_fourth_studys_alone_and_the_third_keeps_its_own():
    assert PIN == "0.7.7"
    assert study.get_study("companion_revised").pinned_ibnr == "0.7.7"
    assert study.get_study("companion_final").pinned_ibnr == study.PINNED_IBNR == "0.7.6"


def test_the_notebooks_settings_agree_with_the_runner_and_the_study_entry(settings):
    spec = study.get_study("companion_revised")
    assert settings["EXPECTED_VERSION"] == spec.pinned_ibnr
    assert settings["STUDY"] == spec.name
    assert (settings["MAIN_ROW"], settings["FACTORS_ROW"]) == spec.rows[:2]
    assert set(settings["SINGLE_ROWS"]) == set(spec.single_draw_rows)
    assert settings["DRAWS"] == study.DEFAULT_DRAWS == 10
    assert settings["MEMBERS"] == study.DEFAULT_MEMBERS == 20
    assert set(settings["COHORT_ORDER"]) == set(spec.cohorts)
    assert settings["COHORT_ORDER"][0] == "C"
    assert settings["SOURCE"] == "pypi"
    assert settings["SCRIPTS_REF"] == f"v{spec.pinned_ibnr}"
    assert settings["STOP_DRAW0_SECONDS"] == study.STOP_FIRST_FIT_SECONDS == 25 * 60
    assert settings["BUDGET_HOURS"] == study.BUDGET_HOURS == 8.0
    assert settings["DRIVE_FOLDER"] == "/content/drive/MyDrive/ibnr_colab_cf"
    assert settings["STUDY3_FOLDER"] == "/content/drive/MyDrive/ibnr_colab_cf/study3"


def test_the_cells_come_in_the_preregistered_order():
    order = [c["id"] for c in cells()]
    at = order.index
    assert at("setup") < at("settings") < at("fetch") < at("token") < at("install")
    assert at("install") < at("scripts") < at("drive") < at("checks") < at("smoke")
    assert at("smoke") < at("draw0") < at("c3") < at("b3") < at("factors") < at("singles")
    assert at("singles") < at("results") and order[-1] == "results"


def test_every_runner_call_names_the_study_and_the_rows():
    assert '"--study", STUDY' in source("smoke") and "--smoke" in source("smoke")
    assert '"--cohort", "B"' in source("smoke") and "OUT_SMOKE" in source("smoke")
    assert "MAIN_ROW" in source("draw0") and '"C"' in source("draw0")
    assert '"C", MAIN_ROW' in source("c3") and '"B", MAIN_ROW' in source("b3")
    assert "OUT_SMOKE" not in source("draw0") + source("c3") + source("b3")
    factors = source("factors")
    assert "FACTORS_ROW" in factors and "--reuse-factors-from" in factors
    assert "STUDY3_FOLDER" in factors
    singles = source("singles")
    assert "SINGLE_ROWS" in singles and '"B"' in singles and "last=1" in singles


def test_the_scripts_cell_checks_the_studys_pin_not_the_third_studys():
    text = source("scripts")
    assert "get_study(STUDY).pinned_ibnr == EXPECTED_VERSION" in text
    assert "PINNED_IBNR ==" not in text.replace("pinned_ibnr", "")


def test_the_install_cell_is_the_third_studys_and_names_no_literal_version():
    # the third study's cell, but for one line it lacked: its "git" route with a token had no
    # EXTRAS. It ran on "pypi", so the repair is here alone and its notebook stays as it ran
    repair = (
        '    EXTRAS = "ibnr[jax]"  # already installed above: this adds only its dependencies\n'
    )
    assert source("install").replace(repair, "", 1).count(repair) == 1
    assert source("install").replace(repair, "", 1) == source("install", THIRD)
    assert not re.search(
        r"\d+\.\d+\.\d+", source("install").replace("cas-schedule-p==2026.6.13", "")
    )
    assert source("fetch") == source("fetch", THIRD) and source("token") == source("token", THIRD)


def test_the_checks_cell_calls_the_prefit_checks_and_prints_the_versions_and_hashes():
    text = source("checks")
    assert "prefit_checks(_cohorts)" in text and "library_versions()" in text
    assert "data_hash" in text
    tree = ast.parse(text)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.comparators[0], ast.Dict):
            with contextlib.suppress(ValueError):
                found["rhs"] = ast.literal_eval(node.comparators[0])
    assert found["rhs"] == {"revised": 18153, "final": 18537}
    assert study.EXPECTED_PARAMETERS == {"revised": 18153, "final": 18537}


def test_the_token_is_never_printed_or_stored_by_any_cell():
    for cell in cells():
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            if "token" in text.lower():
                assert not re.search(r"^\s*[!%]", text, re.M), cell["id"]
                assert "print(_TOKEN" not in text and "len(_TOKEN" not in text
    others = [c["id"] for c in cells() if "GITHUB_TOKEN" in "".join(c["source"])]
    assert others == ["setup", "fetch"]


def test_the_notebook_is_valid_clean_and_secret_free():
    raw = NOTEBOOK.read_text(encoding="utf-8")
    nb = json.loads(raw)
    assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    try:
        import nbformat
    except ImportError:
        pass
    else:
        nbformat.validate(nbformat.reads(raw, as_version=4))
    for pattern in (r"hf_[A-Za-z0-9]{10,}", r"ghp_[A-Za-z0-9]{10,}", r"sk-[A-Za-z0-9]{20,}"):
        assert not re.search(pattern, raw)
    assert chr(0x2014) not in raw and chr(0x2013) not in raw


def test_the_notebook_states_the_time_the_stops_and_how_to_resume():
    intro = source("intro")
    assert "3 to 4 GPU hours" in intro
    flat = " ".join(intro.split())
    assert "25 minutes" in flat and "8 GPU hours" in flat
    assert "A rerun skips everything already saved" in " ".join(intro.split())
    assert "Python 3.13" in source("install") or "3.13" in source("install")


# -- the helper cell, executed on a tiny fixture ---------------------------------------------


def test_the_draw_0_rule_stops_at_25_minutes_on_cohort_c(tmp_path, helpers):
    with pytest.raises(SystemExit, match="not saved yet"):
        helpers["check_draw0"](tmp_path)
    write_record(tmp_path, "C", "tlrn_cr_refit", 0, 1500.0)
    assert helpers["check_draw0"](tmp_path) == 1500.0
    write_record(tmp_path, "C", "tlrn_cr_refit", 0, 1500.1)
    with pytest.raises(SystemExit, match="STOP: draw 0 on C4 took 25.0 minutes"):
        helpers["check_draw0"](tmp_path)
    other = tmp_path / "other"
    write_record(other, "B", "tlrn_cr_refit", 0, 9999.0)
    with pytest.raises(SystemExit, match="not saved yet"):
        helpers["check_draw0"](other)


def test_the_budget_counts_this_studys_full_fits_and_a_reused_fit_adds_nothing(tmp_path, helpers):
    write_record(tmp_path, "C", "tlrn_cr_refit", 0, 28440.0)
    write_record(tmp_path, "C", "tlrn_cr_refit", 1, 99999.0, smoke=True)
    write_record(tmp_path, "C", "tlrn_cr_refit", 2, 99999.0, study_name="companion_final")
    write_record(tmp_path, "C", "tlrn_cr_factors", 0, 0.0)  # a fit taken from the third study
    assert helpers["gpu_hours"](tmp_path) == pytest.approx(7.9)
    write_record(tmp_path, "B", "tlrn_cr_refit", 0, 360.0)
    with pytest.raises(SystemExit, match="STOP: the saved fits add up to 8.00 GPU hours"):
        helpers["check_budget"](tmp_path)


def test_run_piece_forwards_extra_arguments_and_checks_the_budget_before_every_fit(
    tmp_path, helpers
):
    calls = []

    def runner(argv):
        calls.append(argv)
        draws = int(argv[argv.index("--draws") + 1])
        write_record(tmp_path, "C", "tlrn_cr_factors", draws - 1, 0.0)

    helpers["run_piece"](
        runner,
        tmp_path,
        "C",
        "tlrn_cr_factors",
        first=1,
        last=2,
        extra=("--reuse-factors-from", "/third"),
    )
    assert len(calls) == 2
    for argv in calls:
        assert argv[:2] == ["--study", "companion_revised"]
        assert argv[-2:] == ["--reuse-factors-from", "/third"]
        assert argv[argv.index("--rows") + 1] == "tlrn_cr_factors"
    for draw in range(2, 10):
        write_record(tmp_path, "C", "tlrn_cr_refit", draw, 3600.0)
    calls.clear()
    with pytest.raises(SystemExit, match="STOP"):
        helpers["run_piece"](runner, tmp_path, "C", "tlrn_cr_factors", first=3, last=3)
    assert calls == []


def test_the_evaluator_command_reads_both_drive_folders_and_follows_the_settings(helpers):
    out = "/content/drive/MyDrive/ibnr_colab_cf/study4"
    assert helpers["evaluator_command"](out) == (
        'uv run python scripts/tlrn_study4_eval.py --fits "G:\\My Drive\\ibnr_colab_cf\\study4" '
        '--study3-fits "G:\\My Drive\\ibnr_colab_cf\\study3" --out C:\\Temp\\study4_report'
    )
    moved = run_cell(
        "helpers", run_cell("fetch", dict(run_cell("settings", {}), LAPTOP_DRIVE_ROOT="H:\\D"))
    )
    assert moved["evaluator_command"](out).count("H:\\D\\ibnr_colab_cf") == 2
    assert 'OUT = f"{DRIVE_FOLDER}/study4"' in source("drive")
    assert 'OUT_SMOKE = f"{DRIVE_FOLDER}/study4_smoke"' in source("drive")


def test_failed_files_names_the_fits_the_runner_gave_up_on(tmp_path, helpers):
    assert helpers["failed_files"](tmp_path) == []
    (tmp_path / "FAILED_B_tlrn_cr_paid_d00-ibnr0.7.7.txt").write_text("failed\n")
    assert helpers["failed_files"](tmp_path) == ["FAILED_B_tlrn_cr_paid_d00-ibnr0.7.7.txt"]


def test_missing_tables_names_a_fit_the_evaluator_could_not_read(tmp_path, helpers):
    stem = write_record(tmp_path, "C", "tlrn_cr_refit", 0, 400.0)
    assert helpers["missing_tables"](tmp_path) == [f"{stem}_valuations.csv", f"{stem}_members.csv"]


def test_the_results_cell_expects_every_fit_of_the_study(settings):
    text = source("results")
    assert "len(SINGLE_ROWS)" in text and "DRAWS * len(COHORT_ORDER) * 2" in text
    assert (
        settings["DRAWS"] * len(settings["COHORT_ORDER"]) * 2 + len(settings["SINGLE_ROWS"]) == 44
    )


def test_the_notebooks_row_names_are_the_evaluators_and_the_runners(settings):
    assert settings["MAIN_ROW"] == ev.MAIN_ROW == study.REVISED_MAIN_ROW
    assert settings["FACTORS_ROW"] == ev.FACTORS_ROW == study.REVISED_FACTORS_ROW
    assert tuple(settings["SINGLE_ROWS"]) == tuple(
        study.get_study("companion_revised").single_draw_rows
    )


# -- the install routes and the answers to the stops ---------------------------------------------


def install_namespace(settings, *, source_kind, token, expected):
    import sys as system

    calls = []
    namespace = {
        **settings,
        "SOURCE": source_kind,
        "EXPECTED_VERSION": expected,
        "_TOKEN": token,
        "sys": system,
        "pip_install": lambda *args, token=None: calls.append(args),
    }
    return namespace, calls


@pytest.mark.parametrize("token", [None, "a-token"])
def test_the_git_route_installs_with_and_without_a_token(settings, token):
    import ibnr

    namespace, calls = install_namespace(
        settings, source_kind="git", token=token, expected=ibnr.__version__
    )
    run_cell("install", namespace)
    assert "ibnr[jax]" in calls[1]
    assert (f"x-access-token:{token}@" in calls[0][-1]) is bool(token)


def test_every_route_still_stops_on_a_version_that_is_not_the_pinned_one(settings):
    namespace, calls = install_namespace(settings, source_kind="git", token=None, expected="9.9.9")
    with pytest.raises(AssertionError, match="is installed, not 9.9.9"):
        run_cell("install", namespace)
    assert len(calls) == 2


def test_the_notebook_does_not_offer_git_as_a_way_round_the_release():
    """The fits wait for the release; only a branch that says 0.7.7 passes the check."""
    assert "not released yet" not in source("setup")
    assert "`pyproject.toml` already says" in source("h3")


def test_the_notebook_says_what_to_do_with_a_save_cut_short():
    assert "Delete the files it names" in source("h4")


def test_the_budget_counts_the_seconds_of_failed_fits_and_of_failed_attempts(tmp_path, helpers):
    stem = write_record(tmp_path, "C", "tlrn_cr_refit", 0, 3600.0)
    record = tmp_path / f"{stem}.json"
    saved = json.loads(record.read_text())
    saved["run"]["failed_attempt_seconds"] = 1800.0
    record.write_text(json.dumps(saved))
    (tmp_path / f"FAILED_C_tlrn_cr_refit_d01-ibnr{PIN}.txt").write_text(
        "C tlrn_cr_refit draw 1 (seed 100014) failed: non-finite\nseconds: 5400.0\n"
    )
    (tmp_path / f"FAILED_C_tlrn_cr_refit_d02_smoke-ibnr{PIN}.txt").write_text(
        "x\nseconds: 7200.0\n"
    )
    assert helpers["gpu_hours"](tmp_path) == pytest.approx(3.0)


def test_the_notebooks_budget_equals_the_runners_on_the_same_folder(tmp_path, helpers):
    stem = write_record(tmp_path, "C", "tlrn_cr_refit", 0, 3600.0)
    (tmp_path / f"FAILED_{stem}.txt").write_text("failed\nseconds: 1800.0\n")
    used = study.fit_seconds_used(tmp_path, study="companion_revised") / 3600.0
    assert helpers["gpu_hours"](tmp_path) == pytest.approx(used) == pytest.approx(1.5)


def answered_helpers(**answers):
    namespace = dict(run_cell("settings", {}), **answers)
    return run_cell("helpers", run_cell("fetch", namespace))


def test_no_answer_means_no_argument_and_the_stops_stand(helpers):
    assert helpers["stop_answers"]() == []


def test_the_first_fit_answer_reaches_the_helpers_and_the_runner(tmp_path):
    answered = answered_helpers(FIRST_FIT_ANSWER="Ethan: go on")
    write_record(tmp_path, "C", "tlrn_cr_refit", 0, 1600.0)
    assert answered["check_draw0"](tmp_path) == 1600.0
    calls = []
    answered["run_piece"](calls.append, tmp_path, "C", "tlrn_cr_refit", first=1, last=1)
    assert calls[0][calls[0].index("--answered-first-fit") + 1] == "Ethan: go on"
    assert "stop_answers()" in source("draw0")


def test_the_budget_answer_reaches_the_helpers_and_the_runner(tmp_path):
    answered = answered_helpers(BUDGET_ANSWER_HOURS=9.0)
    write_record(tmp_path, "C", "tlrn_cr_refit", 0, 8.5 * 3600)
    assert answered["check_budget"](tmp_path) == pytest.approx(8.5)
    calls = []
    answered["run_piece"](calls.append, tmp_path, "C", "tlrn_cr_refit", first=1, last=1)
    assert calls[0][calls[0].index("--answered-budget-hours") + 1] == "9.0"
    write_record(tmp_path, "C", "tlrn_cr_refit", 1, 3600.0)
    with pytest.raises(SystemExit, match="STOP"):
        answered["check_budget"](tmp_path)


def test_the_runner_understands_the_arguments_the_notebook_sends(tmp_path, capsys):
    answered = answered_helpers(FIRST_FIT_ANSWER="yes", BUDGET_ANSWER_HOURS=9.0)
    argv = ["--study", "companion_revised", "--out", str(tmp_path), "--rows", "not_a_row"]
    with pytest.raises(SystemExit):
        study.main(argv + answered["stop_answers"]())
    # the parser got as far as the rows, so it knew both of the notebook's arguments
    assert "--rows must be a comma list" in capsys.readouterr().err
