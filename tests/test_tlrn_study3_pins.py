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
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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
    # the default is the public release; "git" is the fallback for an unreleased branch
    assert settings["SOURCE"] == "pypi"
    assert settings["REF"] == "tlrn-study3"
    # the scripts come from the release tag, which is named like the earlier releases (v0.7.5)
    assert settings["SCRIPTS_REF"] == f"v{PIN}" == "v0.7.6"
    assert settings["DRIVE_FOLDER"] == "/content/drive/MyDrive/ibnr_colab_cf"
    assert settings["LAPTOP_DRIVE_ROOT"] == "G:\\My Drive"
    # the pre-registered stopping rules
    assert settings["STOP_DRAW0_SECONDS"] == 25 * 60
    assert settings["BUDGET_HOURS"] == 8.0


def test_the_install_cell_follows_the_existing_flags_and_checks_the_version_after_it():
    install = source("install")
    for needle in (
        '"--no-deps"',
        '"--ignore-requires-python"',
        '"--only-binary=:all:"',
        "cas-schedule-p==2026.6.13",
        '"ibnr[jax]"',
        "git+https://x-access-token:{_TOKEN}@github.com/EKtheSage/ibnr@{REF}",
        'f"ibnr=={EXPECTED_VERSION}"',
        "ibnr[jax]=={EXPECTED_VERSION}",
    ):
        assert needle in install, needle
    assert install.index("--no-deps") < install.index("--only-binary=:all:")
    check = "assert ibnr.__version__ == EXPECTED_VERSION"
    assert check in install and install.index("import ibnr") < install.index(check)
    assert install.index("pip_install(") < install.index(check)
    assert 'ibis.__version__.startswith("12.")' in install
    # no literal version in the cell: it takes it from the one settings cell
    assert not re.search(r"\d+\.\d+\.\d+", install.replace("cas-schedule-p==2026.6.13", ""))


def test_the_token_is_never_printed_echoed_or_stored_by_any_cell():
    for cell in cells():
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        # pip echoes a git URL, so installs run through pip_install (quiet, masked), not "!pip"
        if "_TOKEN" in text or "token" in text.lower():
            assert not re.search(r"^\s*[!%]", text, re.M), cell["id"]
            assert "set -x" not in text and "print(_TOKEN" not in text and "print(token" not in text
            assert "len(_TOKEN" not in text and "len(token" not in text
    assert not re.search(r"^\s*!pip", source("install"), re.M)
    assert "curl" not in source("scripts")
    fetch = source("fetch")
    assert '"-q"' in fetch and '"--no-input"' in fetch
    assert "mask(" in fetch and "GITHUB_TOKEN" in fetch
    # the secret is read in the fetch cell only; nothing writes it anywhere
    others = [c["id"] for c in cells() if "GITHUB_TOKEN" in "".join(c["source"])]
    assert others == ["setup", "fetch"]
    for cell in cells():
        text = "".join(cell["source"])
        assert "write_text(_TOKEN" not in text and "environ[" not in text.replace(
            source("fetch"), ""
        )


def test_the_scripts_come_from_the_release_tag_and_are_checked_against_the_pin():
    scripts = source("scripts")
    assert "ensure_scripts(SCRIPTS_REF, _TOKEN)" in scripts
    assert "tlrn_study_colab.PINNED_IBNR == EXPECTED_VERSION" in scripts
    fetch = source("fetch")
    assert 'SCRIPT_NAMES = ("tlrn_colab.py", "tlrn_study_colab.py")' in fetch
    assert 'REPO_SLUG = "EKtheSage/ibnr"' in fetch
    assert "/contents/scripts/{name}?ref={ref}" in fetch
    assert '"Authorization": f"Bearer {token}"' in fetch
    assert '"Accept": "application/vnd.github.raw"' in fetch
    order = [c["id"] for c in cells()]
    assert order.index("setup") < order.index("settings") < order.index("fetch")
    assert order.index("fetch") < order.index("token") < order.index("install")
    assert order.index("install") < order.index("scripts")
    assert order.index("scripts") < order.index("drive") < order.index("smoke")
    assert order.index("smoke") < order.index("draw0") < order.index("c3") < order.index("b3")
    assert order.index("b3") < order.index("factors") < order.index("results")
    assert "package" not in order and order[-1] == "results"


def test_the_setup_cell_lists_the_one_time_steps():
    text = source("setup")
    for needle in (
        "https://github.com/settings/personal-access-tokens/new",
        "Read-only",
        "Only select repositories",
        "30 days",
        "key icon",
        "GITHUB_TOKEN",
        "Notebook access",
        "delete the token",
    ):
        assert needle in text, needle


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
    return run_cell("helpers", run_cell("fetch", dict(settings)))


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


def test_nothing_in_the_notebook_zips_or_downloads_a_bundle():
    text = "\n".join("".join(c["source"]) for c in cells())
    assert "package_results" not in text and "zipfile" not in text and ".zip" not in text
    assert "package_results" not in helpers_namespace_names()


def helpers_namespace_names():
    return set(run_cell("helpers", run_cell("fetch", run_cell("settings", {}))))


def test_the_results_cell_prints_the_drive_folder_command_the_evaluator_reads(tmp_path, helpers):
    out = "/content/drive/MyDrive/ibnr_colab_cf/study3"
    assert helpers["laptop_folder"](out) == "G:\\My Drive\\ibnr_colab_cf\\study3"
    command = helpers["evaluator_command"](out)
    assert command == (
        'uv run python scripts/tlrn_study3_eval.py --fits "G:\\My Drive\\ibnr_colab_cf\\study3" '
        "--out C:\\Temp\\study3_report"
    )
    assert "--fits" in command and "zip" not in command
    # the printed folder follows the settings, not a literal in the helper
    other = dict(run_cell("settings", {}), LAPTOP_DRIVE_ROOT="H:\\Drive")
    moved = run_cell("helpers", run_cell("fetch", other))
    assert moved["evaluator_command"](out).count("H:\\Drive\\ibnr_colab_cf\\study3") == 1
    # the cell prints the summary and the command from the notebook's own OUT
    text = source("results")
    assert "timing_table(OUT)" in text and "gpu_hours(OUT)" in text
    assert "evaluator_command(OUT)" in text and "missing_tables(OUT)" in text
    assert 'OUT = f"{DRIVE_FOLDER}/study3"' in source("drive")


def test_missing_tables_names_a_fit_the_evaluator_could_not_read(tmp_path, helpers):
    stem = write_record(tmp_path, "C", "tlrn_cf_refit", 0, 400.0)
    assert helpers["missing_tables"](tmp_path) == [
        f"{stem}_valuations.csv",
        f"{stem}_members.csv",
    ]
    (tmp_path / f"{stem}_valuations.csv").write_text("x\n")
    (tmp_path / f"{stem}_members.csv").write_text("x\n")
    assert helpers["missing_tables"](tmp_path) == []


# -- the download helper, against a local fake of the GitHub contents API ------------------------

TOKEN = "github_pat_FAKE0123456789abcdefghijklmnopqrstuvwxyz"
GOOD = "import sys\n\n\ndef main():\n    return 0\n"


class FakeGitHub:
    """A local HTTP server that answers like the contents API for a few paths."""

    def __init__(self, files):
        self.files, self.seen = files, []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                owner.seen.append((self.path, dict(self.headers)))
                path, _, query = self.path.partition("?")
                name = path.rsplit("/", 1)[-1]
                auth = self.headers.get("Authorization", "")
                if auth == "Bearer forbidden":
                    return self.reply(403, b"forbidden")
                if auth != f"Bearer {TOKEN}":
                    return self.reply(401, b"bad credentials")
                if query != "ref=v0.7.6" or name not in owner.files:
                    return self.reply(404, b"not found")
                self.reply(200, owner.files[name])

            def reply(self, code, body):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep the test output quiet
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def github():
    server = FakeGitHub({"tlrn_colab.py": GOOD.encode(), "tlrn_study_colab.py": GOOD.encode()})
    yield server
    server.close()


@pytest.fixture
def fetch(settings):
    return run_cell("fetch", dict(settings))


def test_a_good_script_is_downloaded_with_the_documented_headers(tmp_path, github, fetch, capsys):
    target = fetch["fetch_script"]("tlrn_colab.py", "v0.7.6", TOKEN, tmp_path, api=github.url)
    assert target == tmp_path / "tlrn_colab.py" and target.read_text() == GOOD
    assert not list(tmp_path.glob("*.part"))
    path, headers = github.seen[0]
    assert path == "/repos/EKtheSage/ibnr/contents/scripts/tlrn_colab.py?ref=v0.7.6"
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert headers["Accept"] == "application/vnd.github.raw"
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


def test_ensure_scripts_fetches_both_and_prints_no_token(
    tmp_path, github, fetch, capsys, monkeypatch
):
    monkeypatch.setattr(sys, "path", list(sys.path))  # ensure_scripts adds the folder
    real = fetch["fetch_script"]
    monkeypatch.setitem(fetch, "fetch_script", lambda *a, **k: real(*a, **{**k, "api": github.url}))
    fetch["ensure_scripts"]("v0.7.6", TOKEN, tmp_path)
    assert (tmp_path / "tlrn_colab.py").exists() and (tmp_path / "tlrn_study_colab.py").exists()
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err and "downloaded" in out.out


@pytest.mark.parametrize("bad, code", [("wrong", 401), ("forbidden", 403)])
def test_a_refused_token_gets_the_named_message_without_the_token(
    tmp_path, github, fetch, bad, code
):
    with pytest.raises(SystemExit) as caught:
        fetch["fetch_script"]("tlrn_colab.py", "v0.7.6", bad, tmp_path, api=github.url)
    message = str(caught.value)
    assert f"HTTP {code}" in message
    assert "the token is missing, expired, or lacks Contents: Read on EKtheSage/ibnr" in message
    assert not list(tmp_path.iterdir())


def test_a_missing_ref_or_path_says_so_and_names_the_private_repository(tmp_path, github, fetch):
    with pytest.raises(SystemExit) as caught:
        fetch["fetch_script"]("tlrn_colab.py", "v9.9.9", TOKEN, tmp_path, api=github.url)
    message = str(caught.value)
    assert "ref v9.9.9" in message and "path not found" not in message
    assert (
        "the ref or the path does not exist, or the token cannot see the private repository"
        in message
    )
    assert TOKEN not in message
    with pytest.raises(SystemExit, match="scripts/other.py at ref v0.7.6: the ref or the path"):
        fetch["fetch_script"]("other.py", "v0.7.6", TOKEN, tmp_path, api=github.url)
    assert not list(tmp_path.iterdir())


def test_a_body_that_is_not_python_is_refused_and_never_saved(tmp_path, fetch):
    bad = FakeGitHub({"tlrn_colab.py": b"<html>Sign in</html>", "ok.py": GOOD.encode()})
    try:
        existing = tmp_path / "tlrn_colab.py"
        existing.write_text(GOOD)
        with pytest.raises(SystemExit, match="did not come back as Python, so it was not saved"):
            fetch["fetch_script"]("tlrn_colab.py", "v0.7.6", TOKEN, tmp_path, api=bad.url)
        assert existing.read_text() == GOOD  # a good copy is not overwritten by a bad body
        assert not list(tmp_path.glob("*.part"))
        bad.files["bin.py"] = b"\xff\xfe\x00"
        with pytest.raises(SystemExit, match="did not come back as Python"):
            fetch["fetch_script"]("bin.py", "v0.7.6", TOKEN, tmp_path, api=bad.url)
    finally:
        bad.close()


def test_an_unreachable_github_names_the_network_not_the_token(tmp_path, fetch):
    with pytest.raises(SystemExit, match="could not reach GitHub for scripts/tlrn_colab.py") as e:
        fetch["fetch_script"]("tlrn_colab.py", "v0.7.6", TOKEN, tmp_path, api="http://127.0.0.1:9")
    assert TOKEN not in str(e.value)


def test_a_token_that_reaches_an_error_text_is_masked(fetch):
    assert fetch["mask"](f"boom {TOKEN} and {TOKEN}", TOKEN) == "boom *** and ***"
    assert fetch["mask"]("no token here", None) == "no token here"


def test_the_token_is_read_from_the_environment_when_colab_has_none(monkeypatch, fetch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert fetch["get_token"]() is None
    monkeypatch.setenv("GITHUB_TOKEN", f"  {TOKEN}\n")
    assert fetch["get_token"]() == TOKEN
    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    assert fetch["get_token"]() is None


def test_a_colab_secret_wins_and_a_failing_secret_falls_back(monkeypatch, fetch):
    import types

    class Secrets:
        value = TOKEN

        @classmethod
        def get(cls, name):
            assert name == "GITHUB_TOKEN"
            if cls.value is None:
                raise RuntimeError("SecretNotFoundError")
            return cls.value

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = Secrets
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setenv("GITHUB_TOKEN", "from-environment")
    assert fetch["get_token"]() == TOKEN
    Secrets.value = None
    assert fetch["get_token"]() == "from-environment"


def test_a_missing_secret_stops_with_a_short_message_before_any_install(
    tmp_path, monkeypatch, fetch
):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit) as caught:
        fetch["require_setup"]("pypi", tmp_path)
    message = str(caught.value)
    assert message.startswith("GITHUB_TOKEN is not set") and "Notebook access" in message
    assert len(message.splitlines()) == 1
    with pytest.raises(SystemExit, match="GITHUB_TOKEN is not set"):
        fetch["require_setup"]("git", tmp_path)
    # local copies of both scripts are enough for the release install, but not for "git"
    for name in fetch["SCRIPT_NAMES"]:
        (tmp_path / name).write_text(GOOD)
    assert fetch["require_setup"]("pypi", tmp_path) is None
    with pytest.raises(SystemExit, match="GITHUB_TOKEN is not set"):
        fetch["require_setup"]("git", tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert fetch["require_setup"]("git", tmp_path) == TOKEN
    # the token cell calls it first, and prints neither the token nor its length
    token_cell = source("token")
    assert token_cell.startswith("_TOKEN = require_setup(SOURCE)")
    assert "print(_TOKEN" not in token_cell and "len(" not in token_cell
    # without a token and without local copies, ensure_scripts stops with the same message
    with pytest.raises(SystemExit, match="GITHUB_TOKEN is not set"):
        fetch["ensure_scripts"]("v0.7.6", None, tmp_path / "empty")


def test_pip_install_is_quiet_and_hides_the_token_when_pip_fails(monkeypatch, fetch, capsys):
    import subprocess

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        url = f"git+https://x-access-token:{TOKEN}@github.com/EKtheSage/ibnr@tlrn-study3"
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=f"fatal: bad {url}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    url = f"ibnr @ git+https://x-access-token:{TOKEN}@github.com/EKtheSage/ibnr@tlrn-study3"
    with pytest.raises(SystemExit) as caught:
        fetch["pip_install"]("--no-deps", url, token=TOKEN)
    assert TOKEN not in str(caught.value) and "x-access-token:***@github.com" in str(caught.value)
    command, kwargs = calls[0]
    assert command[1:5] == ["-m", "pip", "install", "-q"] and "--no-input" in command
    assert kwargs["capture_output"] is True  # pip's own echo of the URL is never shown
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


def test_pip_install_prints_nothing_when_it_succeeds(monkeypatch, fetch, capsys):
    import subprocess

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=f"saw {TOKEN}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    fetch["pip_install"]("ibnr==0.7.6", token=TOKEN)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_the_kit_check_in_the_second_studys_tests_still_refuses_a_stale_notebook(tmp_path):
    # the earlier studies' kit check is untouched; a kit must carry its own study's pin
    from . import test_tlrn_study2_pins as old

    old.write_kit(tmp_path / "kit", "0.7.4")
    old.check_kit(tmp_path / "kit")
    with pytest.raises(AssertionError):
        old.check_kit(tmp_path / "kit", pin=PIN)
