"""The third study's Colab notebook, how-to and runner agree on the ibnr version and the rules.

The third study runs on ibnr 0.7.6, which may not be on PyPI when the fits are made, so the
notebook installs either from PyPI or from a branch on GitHub and stops if the version it got is
not the one the runner pins. The first two studies keep 0.7.4 (their records and how-to documents
are history), so this file holds the current pin and ``tests/test_tlrn_study2_pins.py`` holds the
old one. The notebook is read as JSON (no Colab, no network); its settings cell, its checks cell
and its helper cell are executed in a throwaway namespace on a tiny fixture.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import tlrn_study3_eval as ev  # noqa: E402
import tlrn_study_colab as study  # noqa: E402

NOTEBOOK = REPO / "analysis" / "05_tlrn_study3_on_colab.ipynb"
DOC = REPO / "docs" / "tlrn-study3.md"
PIN = study.PINNED_IBNR
VERSIONS = re.compile(
    r"ibnr(?:\[[a-z,]+\])?==(\d+\.\d+\.\d+)|ibnr-(\d+\.\d+\.\d+)-py3|-ibnr(\d+\.\d+\.\d+)"
)


def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def source(cell_id: str) -> str:
    (found,) = [c for c in cells() if c["id"] == cell_id]
    return "".join(found["source"])


def run_cell(cell_id: str, namespace: dict) -> dict:
    """Execute a notebook cell (one without magics) in ``namespace``."""
    exec(compile(source(cell_id), f"<cell {cell_id}>", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def settings() -> dict:
    return run_cell("settings", {})


# -- the version -----------------------------------------------------------------------------


def test_the_current_pin_is_the_third_studys_and_the_earlier_ones_keep_theirs():
    assert PIN == "0.7.6"
    assert study.get_study("companion_final").pinned_ibnr == PIN
    for name in ("accident_year", "factor_head"):
        assert study.get_study(name).pinned_ibnr == "0.7.4" == study.EARLIER_PINNED_IBNR


def test_the_pin_matches_the_package_this_checkout_builds():
    import ibnr

    assert ibnr.__version__ == PIN


def test_the_notebooks_settings_agree_with_the_runner_and_the_study_entry(settings):
    spec = study.get_study("companion_final")
    assert settings["EXPECTED_VERSION"] == PIN
    assert settings["STUDY"] == spec.name
    assert (settings["MAIN_ROW"], settings["FACTORS_ROW"]) == spec.rows
    assert settings["DRAWS"] == study.DEFAULT_DRAWS == 10
    assert settings["MEMBERS"] == study.DEFAULT_MEMBERS == 20
    assert set(settings["COHORT_ORDER"]) == set(spec.cohorts)
    assert settings["COHORT_ORDER"][0] == "C"  # the fresh cohort is fitted first
    assert settings["SOURCE"] in ("git", "pypi")
    assert settings["REF"] == "tlrn-study3"
    # the pre-registered stopping rules
    assert settings["STOP_DRAW0_SECONDS"] == 25 * 60
    assert settings["BUDGET_HOURS"] == 8.0


def test_the_install_cell_follows_the_existing_flags_and_checks_the_version_after_it():
    install = source("install")
    for needle in (
        "--no-deps",
        "--ignore-requires-python",
        "--only-binary=:all:",
        "cas-schedule-p==2026.6.13",
        "ibnr[jax]",
        "git+https://github.com/EKtheSage/ibnr@{REF}",
        "f'\"ibnr=={EXPECTED_VERSION}\"'",
        "ibnr[jax]=={EXPECTED_VERSION}",
    ):
        assert needle in install, needle
    assert install.index("--no-deps") < install.index("--only-binary=:all:")
    check = "assert ibnr.__version__ == EXPECTED_VERSION"
    assert check in install and install.index("import ibnr") < install.index(check)
    assert install.index("pip install") < install.index(check)
    # no literal version in the cell: it takes it from the one settings cell
    assert not re.search(r"\d+\.\d+\.\d+", install.replace("cas-schedule-p==2026.6.13", ""))


def test_the_scripts_come_from_the_ref_the_install_used_and_are_checked_against_the_pin():
    scripts = source("scripts")
    assert 'SCRIPTS_REF = REF if SOURCE == "git"' in scripts
    for name in ("tlrn_colab.py", "tlrn_study_colab.py"):
        assert name in scripts
    assert "raw.githubusercontent.com/EKtheSage/ibnr/{SCRIPTS_REF}/scripts/" in scripts
    assert "tlrn_study_colab.PINNED_IBNR == EXPECTED_VERSION" in scripts
    order = [c["id"] for c in cells()]
    assert order.index("settings") < order.index("install") < order.index("scripts")
    assert order.index("scripts") < order.index("drive") < order.index("smoke")
    assert order.index("smoke") < order.index("draw0") < order.index("c3") < order.index("b3")
    assert order.index("b3") < order.index("factors") < order.index("package")


def test_every_runner_call_names_the_study_and_the_rows_in_the_preregistered_order():
    for cell_id in ("smoke", "draw0"):
        assert '"--study", STUDY' in source(cell_id)
    assert "MAIN_ROW" in source("draw0") and '"C"' in source("draw0")
    assert '"C", MAIN_ROW' in source("c3")
    assert '"B", MAIN_ROW' in source("b3")
    assert "FACTORS_ROW" in source("factors")
    assert "OUT_SMOKE" in source("smoke") and "--smoke" in source("smoke")
    assert "OUT_SMOKE" not in source("draw0") + source("c3") + source("b3")


def test_the_checks_cell_holds_the_pre_registered_counts():
    tree = ast.parse(source("checks"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            with contextlib.suppress(ValueError):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    assert found["EXPECTED_PAIRS"] == ev.EXPECTED_PAIRS
    assert found["EXPECTED_COMPANIES_B"] == ev.EXPECTED_COMPANIES["B"]
    assert found["EXPECTED_PARAMETERS"] == 18537


def test_the_network_the_checks_cell_builds_has_the_expected_parameter_count():
    torch = pytest.importorskip("torch")
    import tlrn_colab

    from ibnr.gallery.nn.tlrn.network import TLRNNetwork

    net = TLRNNetwork(
        tlrn_colab.companion_final_config(),
        n_lines=4,
        n_lag=10,
        n_feat=13,
        n_origin=10,
        head_init=torch.zeros(4, 10),
    )
    assert sum(p.numel() for p in net.parameters()) == 18537


def test_the_how_to_names_only_the_pinned_version():
    text = DOC.read_text(encoding="utf-8")
    found = [next(g for g in m.groups() if g) for m in VERSIONS.finditer(text)]
    assert found and set(found) == {PIN}
    assert "tlrn-study3" in text and "05_tlrn_study3_on_colab.ipynb" in text
    assert "tlrn_study3_eval.py" in text


# -- the notebook is a valid, clean notebook ---------------------------------------------------


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
    except ImportError:  # the checks above are the same essentials by hand
        pass
    else:
        nbformat.validate(nbformat.reads(raw, as_version=4))
    for pattern in (
        r"hf_[A-Za-z0-9]{10,}",
        r"ghp_[A-Za-z0-9]{10,}",
        r"sk-[A-Za-z0-9]{20,}",
        r"AIza",
    ):
        assert not re.search(pattern, raw)
    assert chr(0x2014) not in raw and chr(0x2013) not in raw


# -- the helper cell: executed on a tiny fixture -----------------------------------------------


@pytest.fixture
def helpers(settings):
    """The helper cell's namespace, with the settings cell's names, for a fixture folder."""
    return run_cell("helpers", dict(settings))


def write_record(
    folder: Path, cohort, row, draw, seconds, *, smoke=False, study_name="companion_final"
):
    stem = f"{cohort}_{row}_d{draw:02d}{'_smoke' if smoke else ''}-ibnr{PIN}"
    record = {
        "settings": {
            "study": 3,
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
    (folder / f"{stem}.pkl").write_bytes(b"x")
    return stem


def test_the_draw_0_rule_stops_at_25_minutes_and_not_before(tmp_path, helpers):
    with pytest.raises(SystemExit, match="not saved yet"):
        helpers["check_draw0"](tmp_path)
    write_record(tmp_path, "C", "tlrn_cf_refit", 0, 1500.0)
    assert helpers["check_draw0"](tmp_path) == 1500.0
    write_record(tmp_path, "C", "tlrn_cf_refit", 0, 1500.1)
    with pytest.raises(SystemExit, match="STOP: draw 0 on C3 took 25.0 minutes"):
        helpers["check_draw0"](tmp_path)
    # another cohort's or row's draw 0 is not the one the rule is about
    other = tmp_path / "other"
    write_record(other, "B", "tlrn_cf_refit", 0, 9999.0)
    write_record(other, "C", "tlrn_cf_factors", 0, 9999.0)
    with pytest.raises(SystemExit, match="not saved yet"):
        helpers["check_draw0"](other)


def test_the_budget_rule_stops_at_eight_gpu_hours_and_counts_only_this_studys_full_fits(
    tmp_path, helpers
):
    write_record(tmp_path, "C", "tlrn_cf_refit", 0, 28440.0)
    write_record(tmp_path, "C", "tlrn_cf_refit", 1, 99999.0, smoke=True)
    write_record(tmp_path, "C", "tlrn_cf_refit", 2, 99999.0, study_name="factor_head")
    assert helpers["gpu_hours"](tmp_path) == pytest.approx(7.9)
    assert helpers["check_budget"](tmp_path) == pytest.approx(7.9)
    write_record(tmp_path, "B", "tlrn_cf_refit", 0, 360.0)
    with pytest.raises(SystemExit, match="STOP: the saved fits add up to 8.00 GPU hours"):
        helpers["check_budget"](tmp_path)
    assert helpers["check_budget"](tmp_path, limit=8.1) == pytest.approx(8.0)


def test_run_piece_checks_the_budget_before_every_fit_and_asks_for_one_new_draw_each_time(
    tmp_path, helpers
):
    calls = []

    def runner(argv):
        calls.append(argv)
        # the runner saves the draw it was asked for (the last of --draws), taking an hour
        draws = int(argv[argv.index("--draws") + 1])
        write_record(tmp_path, "C", "tlrn_cf_refit", draws - 1, 3600.0)

    helpers["run_piece"](runner, tmp_path, "C", "tlrn_cf_refit", first=1, last=3)
    assert [a[a.index("--draws") + 1] for a in calls] == ["1", "2", "3"]
    for argv in calls:
        assert argv[:2] == ["--study", "companion_final"]
        assert argv[argv.index("--cohort") + 1] == "C"
        assert argv[argv.index("--rows") + 1] == "tlrn_cf_refit"
        assert "--smoke" not in argv
    # eight hours used: the next pass is refused before the runner is called
    for draw in range(3, 8):
        write_record(tmp_path, "C", "tlrn_cf_refit", draw, 3600.0)
    calls.clear()
    with pytest.raises(SystemExit, match="STOP"):
        helpers["run_piece"](runner, tmp_path, "C", "tlrn_cf_refit", first=9, last=10)
    assert calls == []


def test_the_timing_table_prints_every_fit_and_the_mean_per_row(tmp_path, helpers):
    write_record(tmp_path, "C", "tlrn_cf_refit", 0, 400.0)
    write_record(tmp_path, "C", "tlrn_cf_refit", 1, 600.0)
    write_record(tmp_path, "B", "tlrn_cf_factors", 0, 10.0)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rows = helpers["timing_table"](tmp_path)
    text = buffer.getvalue()
    assert len(rows) == 3
    assert "mean C tlrn_cf_refit: 500.0 s over 2 fits" in text
    assert "mean B tlrn_cf_factors: 10.0 s over 1 fits" in text
    assert "total 0.28 GPU hours in 3 fits" in text


def test_package_results_zips_records_and_tables_and_leaves_out_the_pickles(tmp_path, helpers):
    stem = write_record(tmp_path / "fits", "C", "tlrn_cf_refit", 0, 400.0)
    (tmp_path / "fits" / f"{stem}.json.part").write_text("half written")
    target = helpers["package_results"](tmp_path / "fits", tmp_path / "download")
    names = set(zipfile.ZipFile(target).namelist())
    assert names == {f"study3/{stem}.json", f"study3/{stem}_reserves.csv", "study3/timing.json"}
    kept = helpers["package_results"](tmp_path / "fits", tmp_path / "download2", entries=True)
    assert f"study3/{stem}.pkl" in set(zipfile.ZipFile(kept).namelist())
    timing = json.loads((tmp_path / "download" / "timing.json").read_text())
    assert timing["fits"][0]["seconds"] == 400.0


def test_the_kit_check_in_the_second_studys_tests_still_refuses_a_stale_notebook(tmp_path):
    # the earlier studies' kit check is untouched; a kit must carry its own study's pin
    from . import test_tlrn_study2_pins as old

    old.write_kit(tmp_path / "kit", "0.7.4")
    old.check_kit(tmp_path / "kit")
    with pytest.raises(AssertionError):
        old.check_kit(tmp_path / "kit", pin=PIN)
