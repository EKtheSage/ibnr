"""One ruff version, in one place: uv.lock.

Formatter output is version-dependent, so a floating linter is a real failure
mode rather than a tidiness point. `uvx ruff` (what CI ran until now) always
fetches the newest release: a ruff that reformats one more construct turns a
green branch red with no change to the branch, and disagrees with the `uv run
ruff` the contributor ran locally against the lockfile. The fix was to have CI
read the version out of uv.lock; these tests keep that arrangement intact.

The floor in pyproject is not free either. Markdown formatting arrived in ruff
0.16, and the CI format check now covers the code blocks in README.md and the
model cards - under 0.15 those blocks are invisible, so the check would pass
locally and fail in CI.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: the release whose formatter first reached python code blocks inside markdown
MARKDOWN_FORMATTING_SINCE = (0, 16)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def _pyproject_ruff_floor() -> tuple[int, ...]:
    with open(REPO / "pyproject.toml", "rb") as fh:
        dev = tomllib.load(fh)["dependency-groups"]["dev"]
    (spec,) = [d for d in dev if d.replace("-", "_").startswith("ruff")]
    assert ">=" in spec, f"expected a floor for ruff, got {spec!r}"
    return _version(spec)


def _locked_ruff_version() -> tuple[int, ...]:
    with open(REPO / "uv.lock", "rb") as fh:
        lock = tomllib.load(fh)
    (pkg,) = [p for p in lock["package"] if p["name"] == "ruff"]
    return _version(pkg["version"])


def test_the_locked_ruff_satisfies_the_declared_floor():
    """A raised floor with a stale lock means CI and `uv run ruff` differ again."""
    assert _locked_ruff_version() >= _pyproject_ruff_floor()


def test_the_floor_is_new_enough_to_format_markdown():
    floor = _pyproject_ruff_floor()
    assert floor >= MARKDOWN_FORMATTING_SINCE, (
        f"ruff>={'.'.join(map(str, floor))} predates markdown formatting; the CI "
        "format check covers README.md and the model cards, so an older ruff would "
        "pass locally and fail in CI"
    )


def test_the_lint_workflow_takes_its_version_from_the_lockfile():
    """No literal `ruff@0.16.0` in the workflow: two places to bump is how they
    drift. The version is read from uv.lock at job time instead."""
    workflow = (REPO / ".github" / "workflows" / "lint.yml").read_text(encoding="utf-8")
    hardcoded = re.findall(r"ruff(?:@|==)\d[\w.]*", workflow)
    assert not hardcoded, f"pinned ruff version literal(s) in lint.yml: {hardcoded}"
    assert "uv.lock" in workflow


def test_the_lint_workflow_checks_formatting_and_markdown():
    """The three gates the lint job is supposed to run. `ruff check` alone was
    the state that let unformatted files - and unlinted doc snippets - land."""
    workflow = (REPO / ".github" / "workflows" / "lint.yml").read_text(encoding="utf-8")
    assert "check --output-format=github ." in workflow
    assert "format --check ." in workflow
    assert "scripts/lint_md_snippets.py" in workflow
