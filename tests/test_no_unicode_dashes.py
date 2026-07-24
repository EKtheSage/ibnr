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
EXCLUDE_DIRS = {".git", ".venv", "great-docs", "__pycache__", ".ruff_cache", ".pytest_cache"}
EXCLUDE_FILES = {"uv.lock", "LICENSE"}


def _repo_text_files():
    for p in REPO.rglob("*"):
        if not p.is_file() or p.suffix not in SUFFIXES or p.name in EXCLUDE_FILES:
            continue
        if EXCLUDE_DIRS & set(p.relative_to(REPO).parts):
            continue
        yield p


@pytest.mark.parametrize("dash", sorted(UNICODE_DASHES), ids=lambda d: UNICODE_DASHES[d])
def test_no_unicode_dash(dash):
    offenders = []
    for p in _repo_text_files():
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if dash in text:
            offenders += [
                f"{p.relative_to(REPO)}:{n}"
                for n, line in enumerate(text.splitlines(), 1)
                if dash in line
            ]
    assert not offenders, (
        f"{UNICODE_DASHES[dash]} (U+{ord(dash):04X}) found; use a plain '-':\n  "
        + "\n  ".join(offenders[:20])
    )


def test_box_drawing_is_still_allowed():
    """Guard the guard: the three-repo diagram must not be collateral damage."""
    diagram = REPO / "docs" / "three-repo-workflow.md"
    assert BOX_DRAWINGS_LIGHT_HORIZONTAL in diagram.read_text(encoding="utf-8")
