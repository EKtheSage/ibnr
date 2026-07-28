"""No Unicode dash characters anywhere in the tracked text of this repo.

Prose here is deliberately plain ASCII for dashes. The first sweep only removed
U+2014, leaving U+2212 MINUS SIGN and U+2013 EN DASH behind, which render almost
identically and survived in the model cards. This guard covers the whole dash
family so the next lookalike cannot slip through.

The codepoints are spelled with chr() rather than as literals, so this file is
itself pure ASCII and does not trip its own assertion.

Box drawing (U+2500 and friends) is *not* a dash and is explicitly allowed: the
three-repo diagram is built from it. Other non-ASCII (arrows, Greek, math
operators such as U+00B1 PLUS-MINUS) is fine too; only dashes are policed.
"""

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: every codepoint that renders as a dash, including ones not currently present
UNICODE_DASHES = {
    chr(0x2010): "HYPHEN",
    chr(0x2011): "NON-BREAKING HYPHEN",
    chr(0x2012): "FIGURE DASH",
    chr(0x2013): "EN DASH",
    chr(0x2014): "EM DASH",
    chr(0x2015): "HORIZONTAL BAR",
    chr(0x2043): "HYPHEN BULLET",
    chr(0x2212): "MINUS SIGN",
    chr(0xFE31): "VERTICAL EM DASH",
    chr(0xFE32): "VERTICAL EN DASH",
    chr(0xFE58): "SMALL EM DASH",
    chr(0xFE63): "SMALL HYPHEN-MINUS",
    chr(0xFF0D): "FULLWIDTH HYPHEN-MINUS",
    chr(0x00AD): "SOFT HYPHEN",
}

BOX_DRAWINGS_LIGHT_HORIZONTAL = chr(0x2500)

SUFFIXES = {".md", ".py", ".stan", ".toml", ".yml", ".yaml", ".ipynb", ".cfg", ".txt"}
#: ".claude" holds the worktrees of the branch-per-task workflow, each a full
#: copy of this repo - without it the walk scans every sibling branch as well.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".claude",
    "great-docs",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
}
EXCLUDE_FILES = {"uv.lock", "LICENSE"}


def _repo_text_files(root: Path = REPO):
    """Walk ``root`` once, pruning excluded directories as we descend.

    Pruning inside the walk rather than filtering afterwards is what makes this
    cheap: ``.venv`` alone holds ~32k files against the repo's ~1.4k, and
    ``rglob("*")`` enumerated every one of them before discarding it.

    The suffix test is case-insensitive because ``Path.suffix`` reports the name
    as written while ``SUFFIXES`` holds nine lowercase entries, so ``".MD" in
    SUFFIXES`` is false and a file named ``README.MD`` or ``MODEL.STAN`` would
    never be opened. A file this walk does not yield is a file this guard never
    reads, so a dash in it would survive every future run.

    ``root`` is a parameter so the walk itself can be exercised over a fixture
    tree; the excluded names are exact matches and stay case-sensitive.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            if name in EXCLUDE_FILES:
                continue
            p = Path(dirpath) / name
            if p.suffix.casefold() in SUFFIXES:
                yield p


def _read_texts(root: Path) -> list[tuple[Path, str]]:
    """Every policed file under ``root`` as (path relative to ``root``, text).

    Unreadable files are skipped rather than failed on: a binary that happens to
    carry a policed suffix, or a file that vanished mid-walk, is not evidence of
    a dash either way.
    """
    texts = []
    for p in _repo_text_files(root):
        try:
            texts.append((p.relative_to(root), p.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return texts


def _offenders(dash: str, texts: list[tuple[Path, str]]) -> list[str]:
    """Every "path:lineno" where ``dash`` appears, in walk order."""
    offenders = []
    for rel, text in texts:
        if dash in text:
            offenders += [
                f"{rel}:{n}" for n, line in enumerate(text.splitlines(), 1) if dash in line
            ]
    return offenders


@pytest.fixture(scope="session")
def repo_texts():
    """Every policed file as (relative path, text), read once per session.

    The check below is parametrized over 14 codepoints. Reading here instead of
    inside the test turns 14 walks and 14 full re-reads of the repo into one.
    """
    return _read_texts(REPO)


@pytest.mark.parametrize("dash", sorted(UNICODE_DASHES), ids=lambda d: UNICODE_DASHES[d])
def test_no_unicode_dash(dash, repo_texts):
    offenders = _offenders(dash, repo_texts)
    assert not offenders, (
        f"{UNICODE_DASHES[dash]} (U+{ord(dash):04X}) found; use a plain '-':\n  "
        + "\n  ".join(offenders[:20])
    )


@pytest.mark.parametrize("name", ["README.MD", "Notes.Md", "MODEL.STAN"])
def test_the_walk_matches_the_suffix_case_insensitively(tmp_path, name):
    """A file named README.MD is text of this repo and has to be policed too.

    The walk filters on ``Path.suffix``, which reports the name as written, so a
    plain ``in SUFFIXES`` compared it against nine lowercase entries and answered
    false for every capitalised name. Such a file is never opened, and a guard
    that never opens a file cannot fail on it - the dash would sit there through
    every green run. The names are spelled out rather than derived, so the check
    is exercised on a case-sensitive filesystem too, not only on Windows.
    """
    dash = chr(0x2013)  # EN DASH; spelled with chr() to keep this file ASCII
    doc = tmp_path / name
    doc.write_text(f"a clean first line\nbudget {dash} 100\n", encoding="utf-8")

    assert list(_repo_text_files(tmp_path)) == [doc]
    assert _offenders(dash, _read_texts(tmp_path)) == [f"{name}:2"]


def test_box_drawing_is_still_allowed():
    """Guard the guard: the three-repo diagram must not be collateral damage."""
    diagram = REPO / "docs" / "three-repo-workflow.md"
    assert BOX_DRAWINGS_LIGHT_HORIZONTAL in diagram.read_text(encoding="utf-8")
