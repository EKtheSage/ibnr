"""The knowledge bundle is a conformant OKF v0.2 bundle, and the checker really checks.

``knowledge/`` holds what has been learned while working here, in the Open Knowledge
Format (https://github.com/GoogleCloudPlatform/open-knowledge-format). ``scripts/check_okf.py``
enforces the format's conformance bar and this repository's conventions on top of it.

Every negative case below builds a small valid bundle in a temp directory, breaks exactly
one thing, and requires the checker to name it: a checker that was only ever run on the
bundle it passes proves nothing about the bundle that would fail it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from check_okf import check_bundle  # noqa: E402

CONCEPT = """---
type: Finding
title: A finding
description: One line.
tags: [a]
status: stable
generated: {{ by: claude-code/test, at: 2026-10-05T22:00:00Z }}
sources:
  - id: src
    resource: https://example.com/page
{extra}---

Body.[^src]

[^src]: A source
"""


def make_bundle(tmp_path: Path, *, extra: str = "", body_swap=None) -> Path:
    root = tmp_path / "bundle"
    (root / "findings").mkdir(parents=True)
    (root / "index.md").write_text(
        '---\nokf_version: "0.2"\n---\n\n# Findings\n\n* [A finding](findings/a.md) - one line.\n',
        encoding="utf-8",
    )
    (root / "log.md").write_text(
        "# Log\n\n## 2026-10-05\n* **Creation**: a.\n\n## 2026-10-01\n* **Initialization**: b.\n",
        encoding="utf-8",
    )
    (root / "findings" / "index.md").write_text(
        "# Findings\n\n* [A finding](a.md) - one line.\n", encoding="utf-8"
    )
    text = CONCEPT.format(extra=extra)
    if body_swap:
        text = text.replace(*body_swap)
    (root / "findings" / "a.md").write_text(text, encoding="utf-8")
    return root


def test_the_repository_bundle_is_clean():
    assert check_bundle(REPO / "knowledge") == []


def test_a_well_formed_bundle_is_clean(tmp_path):
    assert check_bundle(make_bundle(tmp_path)) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("type: Finding\n", "", "`type` is missing or empty"),
        ("type: Finding\n", "type: ''\n", "`type` is missing or empty"),
        ("description: One line.\n", "", "`description` is missing"),
        ("generated:", "xgenerated:", "`generated` needs a `by` actor"),
        ("2026-10-05T22:00:00Z", "2026-10-05T22:00:00", "`generated.at`"),
        ("status: stable", "status: finished", "`status` must be one of"),
        (
            "    resource: https://example.com/page\n",
            "    title: no resource\n",
            "needs a `resource`",
        ),
        ("[^src]: A source", "[^other]: A source", "is not a `sources` id"),
        ("Body.[^src]", "Body.[^missing]", "used but never defined"),
        ("Body.", "Body, see [x](/findings/none.md).", "does not resolve"),
        ("type: Finding", "type: Finding\nstatus: [unclosed", "not valid YAML"),
    ],
)
def test_each_broken_rule_is_named(tmp_path, old, new, expected):
    root = make_bundle(tmp_path)
    path = root / "findings" / "a.md"
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    problems = check_bundle(root)
    assert any(expected in p for p in problems), problems


def test_frontmatter_that_is_not_a_mapping_is_named(tmp_path):
    root = make_bundle(tmp_path)
    (root / "findings" / "a.md").write_text("---\n- a\n- b\n---\n\nBody.\n", encoding="utf-8")
    assert any("not a mapping" in p for p in check_bundle(root))


def test_a_file_with_no_frontmatter_is_named(tmp_path):
    root = make_bundle(tmp_path)
    (root / "findings" / "a.md").write_text("Just prose.\n", encoding="utf-8")
    assert any("no frontmatter block" in p for p in check_bundle(root))


def test_a_stale_after_without_an_offset_is_refused(tmp_path):
    root = make_bundle(tmp_path, extra="stale_after: 2027-01-01\n")
    assert any("`stale_after`" in p for p in check_bundle(root))


def test_a_verified_entry_needs_both_fields(tmp_path):
    root = make_bundle(tmp_path, extra="verified: { by: human:ethan }\n")
    assert any("`verified` entry needs" in p for p in check_bundle(root))
    ok = make_bundle(
        tmp_path / "ok", extra="verified: { by: human:ethan, at: 2026-10-05T23:00:00Z }\n"
    )
    assert check_bundle(ok) == []


def test_a_concept_missing_from_its_index_is_named(tmp_path):
    root = make_bundle(tmp_path)
    (root / "findings" / "b.md").write_text(
        (root / "findings" / "a.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert any("does not list b.md" in p for p in check_bundle(root))


def test_index_files_carry_no_frontmatter_outside_the_root(tmp_path):
    root = make_bundle(tmp_path)
    index = root / "findings" / "index.md"
    index.write_text(
        "---\ntype: Index\n---\n" + index.read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert any("no frontmatter outside the bundle root" in p for p in check_bundle(root))


def test_the_root_index_declares_the_version(tmp_path):
    root = make_bundle(tmp_path)
    index = root / "index.md"
    index.write_text(
        index.read_text(encoding="utf-8").replace('---\nokf_version: "0.2"\n---\n', "")
    )
    assert any("okf_version" in p for p in check_bundle(root))


def test_log_dates_must_run_newest_first(tmp_path):
    root = make_bundle(tmp_path)
    (root / "log.md").write_text(
        "# Log\n\n## 2026-10-01\n* **Creation**: b.\n\n## 2026-10-05\n* **Creation**: a.\n",
        encoding="utf-8",
    )
    assert any("not newest first" in p for p in check_bundle(root))


def test_a_directory_of_concepts_needs_an_index(tmp_path):
    root = make_bundle(tmp_path)
    (root / "findings" / "index.md").unlink()
    assert any("no index.md" in p for p in check_bundle(root))
