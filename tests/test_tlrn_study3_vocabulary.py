"""The third tlrn study's own files must not use banned words, em dashes or barred names.

Same word list as the second study's check (``tests/test_tlrn_study2_vocabulary.py``, which
holds the pieces the words are built from, so this file does not contain them). Also barred
in every file of the study: the dash that Ethan does not want in prose, and the names of the
outside sites the repository never cites.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .test_tlrn_study2_vocabulary import hits

REPO = Path(__file__).resolve().parent.parent

FILES = [
    "scripts/tlrn_study3_eval.py",
    "docs/tlrn-study3.md",
    "knowledge/decisions/tlrn-study3-preregistration.md",
    "tests/test_tlrn_study3_eval.py",
    "tests/test_tlrn_study3_pins.py",
]
OUTSIDE_NAMES = ["ibnr" + ".co", "insure" + "AI", "Reserve" + "AI"]
EM_DASH = chr(0x2014)


def study_texts():
    for path in FILES:
        file = REPO / path
        if file.exists():
            yield path, file.read_text(encoding="utf-8")
    nb = json.loads((REPO / "analysis/05_tlrn_study3_on_colab.ipynb").read_text("utf-8"))
    yield "notebook 05 (study 3)", "\n".join("".join(c["source"]) for c in nb["cells"])


@pytest.mark.parametrize("path,text", list(study_texts()), ids=[p for p, _ in study_texts()])
def test_the_study_files_use_no_banned_word_dash_or_outside_name(path, text):
    assert hits(text) == [], f"{path} uses a banned word"
    assert EM_DASH not in text, f"{path} has an em dash"
    for name in OUTSIDE_NAMES:
        assert name.lower() not in text.lower(), f"{path} names {name}"
