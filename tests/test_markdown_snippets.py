"""The python code blocks in the docs are linted, and the linter really bites.

`ruff check` ignores markdown, so scripts/lint_md_snippets.py extracts the
fenced python blocks and lints those. The gate lives in the CI lint job; this
module runs the same check under `uv run pytest` and pins the two behaviours a
future edit could plausibly break: what the extractor counts as a python block,
and which findings are real rather than artefacts of extraction.

Every negative case here fails on a fixture, not on the repo - a gate that can
only be tested by breaking the README is a gate nobody tests.

The last test goes one step further and *runs* the README's end-to-end example,
comparing what it prints against the output blocks pasted beneath it. Linting
only proves a snippet parses and imports cleanly; it cannot notice that the
numbers under it went stale. See the section header there for why that example
in particular.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from lint_md_snippets import (  # noqa: E402
    PYTHON_TAGS,
    Snippet,
    check,
    extract_snippets,
    iter_blocks,
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


@pytest.mark.parametrize("name", ["README.MD", "Notes.Md"])
def test_the_walk_matches_the_suffix_case_insensitively(tmp_path, name):
    """A doc named README.MD is a markdown file and has to be linted like one.

    The pruned walk tests each filename itself, and a plain ``endswith(".md")``
    made that test case-sensitive - where the ``rglob("*.md")`` it replaced had
    matched either case on Windows. A file the walk does not return is a file
    whose broken python block nothing ever reads, so the gate stays green over
    a doc sample that cannot run. The fixture spells the name out, so the check
    is exercised on a case-sensitive filesystem too, not only on Windows.
    """
    doc = tmp_path / name
    doc.write_text("```python\ndef f(:\n```\n", encoding="utf-8")

    assert iter_markdown_files(tmp_path) == [doc]
    failures = check([s for p in iter_markdown_files(tmp_path) for s in extract_snippets(p)])
    assert len(failures) == 1
    assert "does not parse" in failures[0]


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


def test_iter_blocks_keeps_untagged_blocks_that_the_linter_drops(tmp_path):
    """The README's *output* blocks are untagged, so the shared parser has to
    hand them back - while `extract_snippets`, which feeds ruff, still sees only
    python. One parser, two consumers: a second fence reader would be free to
    disagree with this one about where a block starts."""
    doc = _md(tmp_path, "```python\na = 1\n```\n\n```\n1\n```\n\n```sh\nls\n```\n")
    assert [(b.tag, b.code) for b in iter_blocks(doc)] == [
        ("python", "a = 1"),
        ("", "1"),
        ("sh", "ls"),
    ]
    assert [s.code for s in extract_snippets(doc)] == ["a = 1"]


# --- the README's example, actually executed ---------------------------------
#
# Everything above proves the doc snippets parse and lint. None of it can notice
# that the output pasted underneath one went stale, and the README has exactly
# one end-to-end example whose numbers a reader will check theirs against: build
# a Triangle from a hand-written frame with `from_long`, then fit `mack` on it.
# Between them those two blocks pin `from_long`, `Triangle.__repr__`, `to_wide`,
# the registry lookup and the Mack kernel's summary - a wide surface for a doc
# to be silently wrong about. It is core-only and sub-second, so just run it.


def _readme_example(marker: str) -> tuple[Snippet, str]:
    """The README python block containing ``marker``, paired with the untagged
    block that follows it - the output the doc promises its reader.

    ``marker`` has to identify exactly one block. A substring that matches two
    silently grabs the wrong example and fails somewhere confusing later, so the
    ambiguity is the error: `gallery.fit("mack", tri` also prefix-matches
    `gallery.fit("mack", triangle` up in the Status section.
    """
    blocks = iter_blocks(REPO / "README.md")
    hits = [i for i, b in enumerate(blocks) if b.tag in PYTHON_TAGS and marker in b.code]
    where = [f"README.md:{blocks[i].line}" for i in hits]
    assert len(hits) == 1, f"{marker!r} matches {len(hits)} README python blocks: {where}"
    block = blocks[hits[0]]
    output = next((b for b in blocks[hits[0] + 1 :] if not b.tag), None)
    assert output is not None, f"no output block follows README.md:{block.line}"
    return block, output.code


def _comparable(text: str) -> list[str]:
    """Lines, with trailing whitespace dropped and nothing else touched.

    That one allowance is not laziness about float formatting: pandas pads the
    index-name row of a wide frame out to the full column width, and trailing
    spaces do not survive a round trip through an editor or a formatter into
    markdown. Every digit is still compared exactly.
    """
    return [line.rstrip() for line in text.strip("\n").splitlines()]


def test_the_readme_example_still_prints_the_output_it_pastes(capsys):
    """Run the README's example and diff it against the two pasted blocks.

    The second block is executed in the *same* namespace as the first, because
    the README expects the reader to have `tri` in hand by then - so this also
    checks that the narrative order works.
    """
    build, build_output = _readme_example("Triangle.from_long(pd.DataFrame(rows)")
    fit, fit_output = _readme_example('entry_cls = gallery.get("mack")')

    namespace: dict = {}
    exec(compile(build.code, f"README.md:{build.line}", "exec"), namespace)
    assert _comparable(capsys.readouterr().out) == _comparable(build_output)

    exec(compile(fit.code, f"README.md:{fit.line}", "exec"), namespace)
    assert _comparable(capsys.readouterr().out) == _comparable(fit_output)
