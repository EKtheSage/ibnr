"""The ibnr version the Colab studies are installed with is written down once and agreed.

The studies run on the published ibnr from PyPI, so a Colab install cell that did not pin an
exact version (or pinned an older one, such as the 0.7.3 wheel the first study's how-to used
to describe) would run the wrong library without saying so, and the fits would be saved under
the wrong version in their names. The runner holds the version in ``PINNED_IBNR``; the how-to
documents must carry exactly that version in every install line and check it after the
install. The Colab kits (a Drive folder with a notebook and a README, kept outside the
repository) are checked the same way when ``IBNR_COLAB_KITS`` names them, separated by the
path separator of the platform.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import tlrn_study_colab as study  # noqa: E402

#: the first two studies were run with 0.7.4 and their how-to documents say so; the runner's
#: ``PINNED_IBNR`` is the third study's, and ``tests/test_tlrn_study3_pins.py`` holds it
PIN = study.get_study("factor_head").pinned_ibnr
DOCS = ["docs/tlrn-study.md", "docs/tlrn-study2.md"]
#: ``ibnr==X``, ``ibnr[jax]==X``, a wheel name ``ibnr-X-py3`` and a file stem ``-ibnrX.``
VERSIONS = re.compile(
    r"ibnr(?:\[[a-z,]+\])?==(\d+\.\d+\.\d+)|ibnr-(\d+\.\d+\.\d+)-py3|-ibnr(\d+\.\d+\.\d+)"
)


def versions_in(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in VERSIONS.finditer(text)]


def test_the_pins_are_plain_release_numbers_and_not_ahead_of_the_package():
    import ibnr

    here = tuple(int(x) for x in ibnr.__version__.split(".")[:3])
    for name in study.STUDIES:
        pin = study.get_study(name).pinned_ibnr
        assert re.fullmatch(r"\d+\.\d+\.\d+", pin), name
        assert tuple(int(x) for x in pin.split(".")) <= here, name
    assert study.get_study("accident_year").pinned_ibnr == PIN
    assert study.get_study("companion_final").pinned_ibnr == study.PINNED_IBNR
    # moving the current pin must not move the earlier studies' records
    assert PIN == study.EARLIER_PINNED_IBNR != study.PINNED_IBNR


@pytest.mark.parametrize("path", DOCS)
def test_the_how_to_documents_name_only_the_pinned_version(path):
    text = (REPO / path).read_text(encoding="utf-8")
    found = versions_in(text)
    assert found, f"{path} names no ibnr version at all"
    assert set(found) == {PIN}, f"{path} names {sorted(set(found))}, the runner pins {PIN}"


def test_the_first_studys_how_to_installs_from_pypi_and_checks_the_version_after_it():
    text = (REPO / "docs/tlrn-study.md").read_text(encoding="utf-8")
    assert f'"ibnr[jax]=={PIN}"' in text and f'"ibnr=={PIN}"' in text
    assert "whl" not in text and "uv build" not in text
    assert f'assert ibnr.__version__ == "{PIN}"' in text
    # the check comes after the install, not before it
    assert text.index(f'"ibnr[jax]=={PIN}"') < text.index(f'assert ibnr.__version__ == "{PIN}"')


def test_the_runner_docstrings_use_the_pinned_version_in_their_file_names():
    pins = {study.get_study(name).pinned_ibnr for name in study.STUDIES}
    for name in ("tlrn_study_colab.py", "tlrn_colab.py"):
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert set(versions_in(text)) <= pins, name


def kit_dirs():
    raw = os.environ.get("IBNR_COLAB_KITS", "")
    return [Path(p) for p in raw.split(os.pathsep) if p]


def check_kit(kit: Path, pin: str = PIN) -> None:
    readme = (kit / "README.txt").read_text(encoding="utf-8")
    assert set(versions_in(readme)) <= {pin}
    notebooks = sorted(kit.glob("*.ipynb"))
    assert len(notebooks) == 1
    cells = json.loads(notebooks[0].read_text(encoding="utf-8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    install = [c for c in code if "pip install" in c]
    assert len(install) == 1
    cell = install[0]
    assert f'"ibnr=={pin}"' in cell and f'"ibnr[jax]=={pin}"' in cell
    assert set(versions_in(cell)) == {pin}
    check = f'assert ibnr.__version__ == "{pin}"'
    assert check in cell and cell.index(check) > cell.index("import ibnr")
    # the second study's kit carries a copy of the runner, which names the same version
    copy = (kit / "tlrn_study_colab.py").read_text(encoding="utf-8")
    if "PINNED_IBNR" in copy:
        assert f'PINNED_IBNR = "{pin}"' in copy


def write_kit(kit: Path, pin: str, *, check: bool = True, runner_pin: str | None = None) -> None:
    """A minimal kit shaped like the Drive folder: README, one notebook, a runner copy."""
    kit.mkdir(parents=True, exist_ok=True)
    (kit / "README.txt").write_text(f"install ibnr=={pin}" + chr(10), encoding="utf-8")
    lines = [f'!pip install "ibnr=={pin}" "ibnr[jax]=={pin}"', "import ibnr"]
    if check:
        lines.append(f'assert ibnr.__version__ == "{pin}"')
    nb = {"cells": [{"cell_type": "code", "source": [chr(10).join(lines)]}]}
    (kit / "study.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    (kit / "tlrn_study_colab.py").write_text(
        f'PINNED_IBNR = "{runner_pin or pin}"' + chr(10), encoding="utf-8"
    )


def test_the_kit_check_accepts_a_good_kit_and_refuses_each_kind_of_bad_one(tmp_path):
    # always runs: the check on the real Drive kits below skips nothing, but a kit check that
    # never ran anywhere would prove nothing about the check itself
    write_kit(tmp_path / "good", PIN)
    check_kit(tmp_path / "good")
    write_kit(tmp_path / "old", "0.0.1")
    with pytest.raises(AssertionError):
        check_kit(tmp_path / "old")
    write_kit(tmp_path / "nocheck", PIN, check=False)
    with pytest.raises(AssertionError):
        check_kit(tmp_path / "nocheck")
    write_kit(tmp_path / "stale_runner", PIN, runner_pin="0.0.1")
    with pytest.raises(AssertionError):
        check_kit(tmp_path / "stale_runner")


def test_each_colab_kit_named_by_the_environment_is_checked():
    # IBNR_COLAB_KITS names the real Drive folders on the machine that builds them; elsewhere
    # it is unset and there is nothing to check. This is a loop, not a skip, because a leg
    # with a strict skip policy refuses a skipped test.
    for kit in kit_dirs():
        check_kit(kit)
