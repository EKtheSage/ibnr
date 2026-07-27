"""The python code blocks in the docs are linted, and the linter really bites.

`ruff check` ignores markdown, so scripts/lint_md_snippets.py extracts the
fenced python blocks and lints those. The gate lives in the CI lint job; this
module runs the same check under `uv run pytest` and pins the two behaviours a
future edit could plausibly break: what the extractor counts as a python block,
and which findings are real rather than artefacts of extraction.

Every negative case here fails on a fixture, not on the repo - a gate that can
only be tested by breaking the README is a gate nobody tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from lint_md_snippets import (  # noqa: E402
    check,
    extract_snippets,
    iter_markdown_files,
    repo_snippets,
)


def _md(tmp_path: Path, body: str, name: str = "doc.md") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_the_repo_docs_are_clean():
    """The gate itself, over README.md and the model cards."""
    failures = check(repo_snippets())
    assert not failures, "markdown snippets no longer lint clean:\n  " + "\n  ".join(failures)


def test_the_repo_actually_has_snippets_to_check():
    """Guard the guard: an extractor that finds nothing would also pass above."""
    snippets = repo_snippets()
    assert len(snippets) >= 5
    assert any(s.rel.replace("\\", "/") == "README.md" for s in snippets)


def test_the_walk_skips_the_branch_per_task_worktrees(tmp_path):
    """`.claude/worktrees/<name>/` is a full copy of the repo, so without the
    exclusion this gate lints every sibling branch too: a broken snippet on an
    unrelated branch would fail the branch you are actually working on, and the
    real files would each be linted once per worktree on top of that."""
    (tmp_path / "README.md").write_text("```python\na = 1\n```\n", encoding="utf-8")
    other = tmp_path / ".claude" / "worktrees" / "some-branch"
    other.mkdir(parents=True)
    (other / "README.md").write_text("```python\ndef f(:\n```\n", encoding="utf-8")

    assert iter_markdown_files(tmp_path) == [tmp_path / "README.md"]
    assert not check([s for p in iter_markdown_files(tmp_path) for s in extract_snippets(p)])


def test_the_walk_still_reaches_nested_real_docs(tmp_path):
    """Guard the guard: pruning must not cost the model cards, which live several
    directories down under src/. An exclusion that found nothing would also make
    the test above pass."""
    card = tmp_path / "src" / "ibnr" / "gallery" / "bayesian" / "meyers_ccl"
    card.mkdir(parents=True)
    (card / "card.md").write_text("```python\na = 1\n```\n", encoding="utf-8")
    assert iter_markdown_files(tmp_path) == [card / "card.md"]


def test_a_block_that_does_not_parse_is_a_failure(tmp_path):
    """The formatter skips unparseable blocks silently, so this check is the only
    thing standing between a broken doc sample and a green pipeline."""
    doc = _md(tmp_path, "text\n\n```python\ndef f(:\n    return 1\n```\n")
    failures = check(extract_snippets(doc))
    assert len(failures) == 1
    assert "does not parse" in failures[0]
    assert failures[0].startswith(f"{doc}:3")  # the opening fence, line 3


def test_ruff_format_still_does_not_catch_that(tmp_path):
    """Why the parse check exists, kept executable: ruff's markdown formatter
    reports an unparseable block as already formatted. If a future ruff starts
    rejecting it, this test fails and the parse check above becomes redundant."""
    doc = _md(tmp_path, "text\n\n```python\ndef f(:\n    return 1\n```\n")
    proc = subprocess.run(
        ["ruff", "format", "--check", "--isolated", str(doc)], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"ruff now catches this: {proc.stdout}{proc.stderr}"


def test_a_real_finding_is_reported_at_its_markdown_line(tmp_path):
    """An unused import in a snippet is a defect in the docs, and the message has
    to name the markdown line - a temp-file path would be useless to the reader."""
    doc = _md(tmp_path, "prose\n\n```python\nimport os\n\nprint(1)\n```\n")
    failures = check(extract_snippets(doc))
    assert len(failures) == 1
    # fence on line 3, so `import os` is line 4, column 8 is the name
    assert failures[0].startswith(f"{doc}:4:8: F401")


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("```python\nentry.summary()\n```\n", "F821: prose above the block built `entry`"),
        (
            "```python\nx = 1\nimport os\n\nprint(os, x)\n```\n",
            "E402/I001: an excerpt, not a module",
        ),
    ],
)
def test_findings_that_are_artefacts_of_extraction_are_ignored(tmp_path, body, reason):
    assert not check(extract_snippets(_md(tmp_path, body))), reason


def test_only_python_tagged_fences_are_extracted(tmp_path):
    """Matching ruff's own set (python/py) keeps one definition of "a python block
    in the docs" across the formatter gate and this one. `sh` blocks and the 21
    untagged blocks in the repo are prose, and several are not python at all."""
    doc = _md(
        tmp_path,
        "```sh\nuv run pytest\n```\n\n"
        "```\nnot tagged: could be output\n```\n\n"
        "```python\na = 1\n```\n\n"
        "```py\nb = 2\n```\n",
    )
    assert [s.code for s in extract_snippets(doc)] == ["a = 1", "b = 2"]


def test_an_indented_block_is_dedented(tmp_path):
    """A fence nested under a list item arrives indented; linting it as-is would
    report a spurious IndentationError on every line."""
    doc = _md(tmp_path, "- a step:\n\n  ```python\n  if True:\n      a = 1\n  ```\n")
    (snippet,) = extract_snippets(doc)
    assert snippet.code == "if True:\n    a = 1"
    assert not check([snippet])
