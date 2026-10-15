"""The second tlrn study's own files must not use the words Ethan has banned.

He banned one word family outright (everything built on the stem that means switching a part
of a model off to see what it did) and, earlier, a list of jargon words. This checks the
files of the second study: its evaluator, its runner, its tests, its how-to, the decision
note, the findings written for it and the source of notebook 05.

One thing is left out on purpose: this file is not scanned, because it has to name the
words. The evaluator's module docstring is scanned like the rest: its one use of the banned
word was replaced before any fit existed (a wording-only change, proved by a test in
``tests/test_tlrn_study2_eval.py``).

The words are built from pieces so that this file does not contain them.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

STEM = "abl" + "at"
BANNED = [
    # the word family, in every form (including the one used as a column or file name)
    STEM,
    "seed" + "-stable",
    "seed " + "noise",
    r"\bchips?\b",
    "dra" + "in",
    r"\bscr" + "een",
    "member" + "ship",
    "finger" + "print",
    r"\bga" + r"tes?\b",
    r"\bpa" + r"nels?\b",
]
PATTERN = re.compile("|".join(BANNED), re.IGNORECASE)

FILES = [
    "scripts/tlrn_study2_eval.py",
    "scripts/tlrn_study_colab.py",
    "docs/tlrn-study2.md",
    "knowledge/decisions/tlrn-factor-head-study.md",
    "knowledge/findings/companion-final-architecture.md",
    "knowledge/findings/companion-notebook-reproduction.md",
    "knowledge/findings/jax-versus-torch-on-cpu.md",
    "knowledge/references/companion-study-repository.md",
    "tests/test_tlrn_study2.py",
    "tests/test_tlrn_study2_eval.py",
    "tests/test_tlrn_study2_jax.py",
    "tests/test_tlrn_study2_pins.py",
]


def without_module_docstring(source: str) -> str:
    """The source with the module docstring blanked, line for line."""
    tree = ast.parse(source)
    first = tree.body[0] if tree.body else None
    is_text = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    if is_text:
        lines = source.splitlines()
        for n in range(first.lineno - 1, first.end_lineno):
            lines[n] = ""
        return "\n".join(lines)
    return source


def hits(text: str) -> list[str]:
    return [
        f"line {n}: {line.strip()[:100]}"
        for n, line in enumerate(text.splitlines(), 1)
        if PATTERN.search(line)
    ]


@pytest.mark.parametrize("path", FILES)
def test_the_study_files_do_not_use_banned_words(path):
    file = REPO / path
    if not file.exists():
        pytest.skip(f"{path} is not part of this checkout")
    text = file.read_text(encoding="utf-8")
    assert hits(text) == [], f"{path} uses a banned word"


def test_notebook_05_source_does_not_use_banned_words():
    nb = json.loads((REPO / "analysis/05_tlrn_factor_head_study.ipynb").read_text("utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert hits(text) == []


def test_the_check_itself_finds_each_word_and_can_blank_a_module_docstring():
    word = STEM + "ion"
    assert hits(f"the {word} row") and hits("a seed" + " noise problem")
    assert hits("the " + "pa" + "nel here") and not hits("the cabinet is fine")
    source = f'"""the {word} is pinned."""\nx = 1  # {word}\n'
    assert hits(without_module_docstring(source)) == ["line 2: x = 1  # " + word]
    # the evaluator's pinned docstring no longer holds the word
    ev_source = (REPO / "scripts/tlrn_study2_eval.py").read_text(encoding="utf-8")
    assert hits(ev_source) == []
