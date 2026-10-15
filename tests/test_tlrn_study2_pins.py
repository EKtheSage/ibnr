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

PIN = study.PINNED_IBNR
DOCS = ["docs/tlrn-study.md", "docs/tlrn-study2.md"]
#: ``ibnr==X``, ``ibnr[jax]==X``, a wheel name ``ibnr-X-py3`` and a file stem ``-ibnrX.``
VERSIONS = re.compile(
    r"ibnr(?:\[[a-z,]+\])?==(\d+\.\d+\.\d+)|ibnr-(\d+\.\d+\.\d+)-py3|-ibnr(\d+\.\d+\.\d+)"
)


def versions_in(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in VERSIONS.finditer(text)]


def test_the_pin_is_a_plain_release_number_and_not_ahead_of_the_package():
    assert re.fullmatch(r"\d+\.\d+\.\d+", PIN)
    import ibnr

    here = tuple(int(x) for x in ibnr.__version__.split(".")[:3])
    assert tuple(int(x) for x in PIN.split(".")) <= here


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
    for name in ("tlrn_study_colab.py", "tlrn_colab.py"):
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert set(versions_in(text)) <= {PIN}, name


def kit_dirs():
    raw = os.environ.get("IBNR_COLAB_KITS", "")
    return [Path(p) for p in raw.split(os.pathsep) if p]


@pytest.mark.skipif(not kit_dirs(), reason="IBNR_COLAB_KITS is not set")
@pytest.mark.parametrize("kit", kit_dirs() or [Path(".")])
def test_each_colab_kit_pins_the_version_and_asserts_it_after_the_install(kit):
    readme = (kit / "README.txt").read_text(encoding="utf-8")
    assert set(versions_in(readme)) <= {PIN}
    notebooks = sorted(kit.glob("*.ipynb"))
    assert len(notebooks) == 1
    cells = json.loads(notebooks[0].read_text(encoding="utf-8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    install = [c for c in code if "pip install" in c]
    assert len(install) == 1
    cell = install[0]
    assert f'"ibnr=={PIN}"' in cell and f'"ibnr[jax]=={PIN}"' in cell
    assert set(versions_in(cell)) == {PIN}
    check = f'assert ibnr.__version__ == "{PIN}"'
    assert check in cell and cell.index(check) > cell.index("import ibnr")
    # the second study's kit carries a copy of the runner, which names the same version
    copy = (kit / "tlrn_study_colab.py").read_text(encoding="utf-8")
    if "PINNED_IBNR" in copy:
        assert f'PINNED_IBNR = "{PIN}"' in copy
