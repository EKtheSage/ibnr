"""The fourth tlrn study's new files must not use the words Ethan has banned.

The scan is the second study's (``tests/test_tlrn_study2_vocabulary.py``), applied to the files
this study added or changed. The decision note is not scanned: the accepted pre-registration
names the banned word family on purpose and is not edited for wording after acceptance. This
file is not scanned either, because it has to name the words, and its pieces are built from
fragments so that it does not contain them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .test_tlrn_study2_vocabulary import PATTERN, hits

REPO = Path(__file__).resolve().parent.parent

FILES = [
    "scripts/tlrn_colab.py",
    "scripts/tlrn_study_colab.py",
    "scripts/tlrn_study4_eval.py",
    "tests/test_tlrn_study4.py",
    "tests/test_tlrn_study4_eval.py",
    "tests/test_tlrn_study4_pins.py",
]


@pytest.mark.parametrize("path", FILES)
def test_the_fourth_studys_files_do_not_use_banned_words(path):
    text = (REPO / path).read_text(encoding="utf-8")
    assert hits(text) == [], f"{path} uses a banned word"


def test_notebook_06_does_not_use_banned_words_or_dashes():
    nb = json.loads((REPO / "analysis/06_tlrn_study4_on_colab.ipynb").read_text("utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert hits(text) == []
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_the_pattern_is_the_second_studys():
    assert PATTERN.search("a " + "pa" + "nel here") and not PATTERN.search("a cabinet")
