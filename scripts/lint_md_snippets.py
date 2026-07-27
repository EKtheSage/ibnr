"""Lint the python code blocks inside this repo's markdown (README, model cards).

`ruff format` (0.16 and later) reaches into markdown and formats fenced python
blocks, so `ruff format --check` keeps their layout honest. `ruff check` does
NOT: pointed at a .md file it answers "No Python files found under the given
path(s)" and exits 0, in stable and in preview alike, with no setting to turn
markdown linting on. The lint rules the rest of the repo is held to therefore
stop at the docs, which is where a reader most often copies code from.

This script closes that gap the only way available: extract each fenced python
block to a temp file and run the project's own ruff configuration over it. Two
kinds of defect are caught that nothing else in the toolchain sees.

**Blocks that do not parse.** The formatter skips an unparseable block in
silence - measured, not assumed: a markdown file whose only python block is
``def f(:`` reports "1 file already formatted". So a doc sample can be broken
Python forever and every gate stays green. Here it is a hard failure.

**Blocks that lint clean-ish.** Unused imports, dead idioms (UP), mutable
default arguments (B) - the same rules that apply to the source apply to a
snippet a reader is invited to run.

Three rules are ignored, because they fire on the FORM of an excerpt rather
than on a defect in it:

* ``F821`` undefined-name - a snippet legitimately says ``entry.summary()``
  where the prose above it built ``entry``. 15 of the 19 findings on the first
  run were this.
* ``E402`` module-import-not-at-top-of-file and ``I001`` unsorted-imports - an
  excerpt is a fragment of a session, not a module, so an import that appears
  where the narrative needs it is correct. Extraction is what makes the import
  look misplaced.

Everything else comes from ``pyproject.toml`` via ``--config``, so the rule set
is not duplicated here and cannot drift from the source rules.

Untagged fences (```` ``` ````) and other languages (``sh``) are left alone:
ruff's formatter only recognises ``python``/``py``, and matching that keeps one
definition of "a python block in the docs" across the two gates.

Usage
-----
    uv run python scripts/lint_md_snippets.py
    uv run python scripts/lint_md_snippets.py --ruff /path/to/ruff

Exits 1 with a file:line report on the first failing block. tests/
test_markdown_snippets.py runs the same check, so `uv run pytest` covers it too.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: fence opener, capturing indentation (fences nest inside list items) and tag
FENCE = re.compile(r"^(?P<indent>\s*)```(?P<tag>[\w+-]*)\s*$")

#: only these tags are python to ruff's markdown formatter, so only these here
PYTHON_TAGS = {"python", "py"}

#: rules that flag the shape of an excerpt, not a defect in it (see docstring)
EXCERPT_IGNORES = ("F821", "E402", "I001")

#: ".claude" holds the worktrees of the branch-per-task workflow, each a full
#: copy of this repo - without it the walk lints every sibling branch as well,
#: so a broken snippet on an unrelated branch fails the gate on this one.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".claude",
    "great-docs",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
}


@dataclass(frozen=True)
class Snippet:
    """One fenced python block, located well enough to point a reader at it."""

    path: Path
    line: int
    """1-based line of the opening fence in ``path``."""
    code: str

    @property
    def rel(self) -> str:
        """Repo-relative path where possible, so failures are clickable."""
        try:
            return str(self.path.relative_to(REPO))
        except ValueError:
            return str(self.path)

    @property
    def where(self) -> str:
        return f"{self.rel}:{self.line}"


def iter_markdown_files(root: Path) -> list[Path]:
    """Every markdown file under ``root``, pruning excluded directories as we go.

    Pruning inside the walk rather than filtering after ``rglob`` is what makes
    this cheap: ``rglob("*.md")`` still descends into ``.venv`` and into each
    worktree's own ``.venv`` before discarding what it finds there. Measured on
    this checkout with two worktrees present: 1788 ms to reach the same 16 files
    that the pruned walk reaches in 2 ms.
    """
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        out += [Path(dirpath) / f for f in filenames if f.endswith(".md")]
    return sorted(out)


def extract_snippets(path: Path) -> list[Snippet]:
    """Fenced python blocks in one markdown file, dedented to column 0.

    A block indented under a list item is dedented by the opening fence's own
    indentation rather than by ``textwrap.dedent``, which would also strip the
    snippet's meaningful leading whitespace if every line happened to share it.
    """
    out: list[Snippet] = []
    indent: str | None = None
    tag = ""
    start = 0
    buf: list[str] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = FENCE.match(line)
        if indent is None:
            if m:
                indent, tag, start, buf = m.group("indent"), m.group("tag"), n, []
            continue
        # inside a block: only a fence at the opener's indentation closes it
        if m and m.group("indent") == indent and not m.group("tag"):
            if tag in PYTHON_TAGS:
                out.append(Snippet(path, start, "\n".join(buf)))
            indent = None
            continue
        buf.append(line.removeprefix(indent))
    return out


def repo_snippets(root: Path = REPO) -> list[Snippet]:
    return [s for p in iter_markdown_files(root) for s in extract_snippets(p)]


def find_ruff() -> str:
    """The ruff to lint with: the venv's, or whatever is on PATH."""
    for candidate in (Path(sys.executable).parent / "ruff", "ruff"):
        found = shutil.which(str(candidate))
        if found:
            return found
    raise SystemExit(
        "ruff not found. It is a dev dependency: run this under `uv run`, or pass --ruff."
    )


def check(
    snippets: list[Snippet], ruff: str | None = None, config: Path | None = None
) -> list[str]:
    """Failures across ``snippets``, one string per problem, empty when clean."""
    failures = [
        f"{s.where}: does not parse - {exc.msg} (line {exc.lineno} of the block)"
        for s in snippets
        for exc in [_syntax_error(s)]
        if exc is not None
    ]
    parsed = [s for s in snippets if _syntax_error(s) is None]
    if not parsed:
        return failures

    with tempfile.TemporaryDirectory() as tmp:
        by_file = {}
        for n, s in enumerate(parsed):
            f = Path(tmp) / f"block_{n:03d}.py"
            f.write_text(s.code + "\n", encoding="utf-8")
            by_file[f.name] = s
        proc = subprocess.run(
            [
                ruff or find_ruff(),
                "check",
                "--config",
                str(config or REPO / "pyproject.toml"),
                "--config",
                f"lint.extend-ignore={list(EXCERPT_IGNORES)!r}",
                "--output-format=concise",
                "--no-cache",
                tmp,
            ],
            capture_output=True,
            text=True,
        )
    if proc.returncode:
        failures += _relocate(proc.stdout, by_file) or [proc.stderr.strip()]
    return failures


#: ruff --output-format=concise line: "path:line:col: CODE message"
CONCISE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+): (?P<rest>.*)$")


def _relocate(stdout: str, by_file: dict[str, Snippet]) -> list[str]:
    """Rewrite ruff's temp-file diagnostics onto the markdown they came from.

    A diagnostic naming ``/tmp/xyz/block_003.py:2:8`` is useless on its own; the
    reader needs ``README.md:63:8``. The block's own line 1 sits one line below
    its opening fence, so the markdown line is ``fence + block_line``.
    """
    out = []
    for line in stdout.splitlines():
        m = CONCISE.match(line.strip())
        if not m:
            continue
        s = by_file.get(Path(m.group("path")).name)
        if s is None:
            out.append(line.strip())
            continue
        out.append(f"{s.rel}:{s.line + int(m.group('line'))}:{m.group('col')}: {m.group('rest')}")
    return out


def _syntax_error(s: Snippet) -> SyntaxError | None:
    try:
        ast.parse(s.code)
    except SyntaxError as exc:
        return exc
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruff", help="ruff binary to use (default: the venv's, else PATH)")
    ap.add_argument("paths", nargs="*", type=Path, help="markdown files (default: the whole repo)")
    args = ap.parse_args()

    snippets = (
        [s for p in args.paths for s in extract_snippets(p)] if args.paths else repo_snippets()
    )
    failures = check(snippets, ruff=args.ruff)
    files = len({s.path for s in snippets})
    scope = f"{len(snippets)} python blocks in {files} markdown file{'s' if files != 1 else ''}"
    if failures:
        print(f"{scope}; problems:\n")
        for f in failures:
            print(f)
        return 1
    print(f"{scope}: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
