"""Check that ``knowledge/`` is a conformant Open Knowledge Format v0.2 bundle.

The format is https://github.com/GoogleCloudPlatform/open-knowledge-format (SPEC.md). Its
own conformance bar is low: every non-reserved ``.md`` file has parseable frontmatter with a
non-empty ``type``. This script checks that bar and then the stricter conventions this
repository's bundle follows (``knowledge/decisions/knowledge-in-okf-format.md``):

* ``description`` and ``generated: {by, at}`` on every concept;
* the optional families, when present, are well formed: ``sources`` entries carry a
  ``resource`` and unique ids, ``verified`` entries carry ``by`` and ``at``, ``status`` is
  one of draft / stable / deprecated, and every timestamp is ISO 8601 with a UTC offset;
* footnote labels are ``sources`` ids, and every footnote that is used is defined;
* ``index.md`` has no frontmatter (the root's may carry ``okf_version``), and lists every
  concept in its directory;
* ``log.md`` has ISO date headings, newest first;
* every markdown link to a ``.md`` file resolves. The spec tolerates a broken link in a
  consumer; a bundle that is written here should not contain one.

Usage: ``python scripts/check_okf.py [bundle-dir]`` (default ``knowledge``). Exits 1 and
prints one line per problem when the bundle is not clean. Needs PyYAML.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

RESERVED = {"index.md", "log.md"}
STATUSES = {"draft", "stable", "deprecated"}
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")
FOOTNOTE_DEF = re.compile(r"^\[\^([^\]]+)\]:", re.MULTILINE)
FOOTNOTE_USE = re.compile(r"\[\^([^\]]+)\](?!:)")
ISO_WITH_OFFSET = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})$"
)
DATE_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """``(frontmatter text, body)``; the frontmatter is None when the file has none."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text
    return text[4:end], text[end + 5 :]


def is_timestamp(value) -> bool:
    """An ISO 8601 datetime with an explicit UTC offset, as YAML hands it back or as text."""
    if isinstance(value, dt.datetime):
        return value.tzinfo is not None
    return isinstance(value, str) and bool(ISO_WITH_OFFSET.match(value))


def check_concept(path: Path, root: Path, yaml) -> list[str]:
    where = path.relative_to(root).as_posix()
    problems: list[str] = []
    text = path.read_text(encoding="utf-8")
    raw, body = split_frontmatter(text)
    if raw is None:
        return [f"{where}: no frontmatter block"]
    try:
        meta = yaml.safe_load(raw)
    except yaml.YAMLError as err:
        return [f"{where}: frontmatter is not valid YAML ({err})"]
    if not isinstance(meta, dict):
        return [f"{where}: frontmatter is not a mapping"]
    if not isinstance(meta.get("type"), str) or not meta["type"].strip():
        problems.append(f"{where}: `type` is missing or empty")
    if not isinstance(meta.get("description"), str) or not meta["description"].strip():
        problems.append(f"{where}: `description` is missing (this bundle requires one)")

    generated = meta.get("generated")
    if not isinstance(generated, dict) or not generated.get("by"):
        problems.append(f"{where}: `generated` needs a `by` actor (this bundle requires it)")
    elif "at" in generated and not is_timestamp(generated["at"]):
        problems.append(f"{where}: `generated.at` is not an ISO 8601 datetime with an offset")
    elif "at" not in generated:
        problems.append(f"{where}: `generated` needs `at` (this bundle requires it)")

    verified = meta.get("verified")
    if verified is not None:
        for entry in verified if isinstance(verified, list) else [verified]:
            if not isinstance(entry, dict) or not entry.get("by") or "at" not in entry:
                problems.append(f"{where}: a `verified` entry needs `by` and `at`")
            elif not is_timestamp(entry["at"]):
                problems.append(
                    f"{where}: `verified.at` is not an ISO 8601 datetime with an offset"
                )

    if "status" in meta and meta["status"] not in STATUSES:
        problems.append(f"{where}: `status` must be one of {sorted(STATUSES)}")
    if "stale_after" in meta and not is_timestamp(meta["stale_after"]):
        problems.append(f"{where}: `stale_after` is not an ISO 8601 datetime with an offset")

    ids: set[str] = set()
    for source in meta.get("sources") or []:
        if not isinstance(source, dict) or not source.get("resource"):
            problems.append(f"{where}: a `sources` entry needs a `resource`")
            continue
        for key in ("last_modified",):
            if key in source and not is_timestamp(source[key]):
                problems.append(f"{where}: `sources[].{key}` is not an ISO 8601 datetime")
        if "id" in source:
            if source["id"] in ids:
                problems.append(f"{where}: source id {source['id']!r} is used twice")
            ids.add(str(source["id"]))

    defined = set(FOOTNOTE_DEF.findall(body))
    for label in sorted(defined - ids):
        problems.append(f"{where}: footnote [^{label}] is not a `sources` id")
    for label in sorted(set(FOOTNOTE_USE.findall(body)) - defined):
        problems.append(f"{where}: footnote [^{label}] is used but never defined")
    problems += check_links(path, root, body)
    return problems


def check_links(path: Path, root: Path, body: str) -> list[str]:
    where = path.relative_to(root).as_posix()
    problems: list[str] = []
    for target in LINK.findall(body):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith("#"):
            continue  # a URL or an in-page anchor
        target = target.split("#", 1)[0]
        if not target.endswith(".md") and not target.endswith("/"):
            continue
        base = root if target.startswith("/") else path.parent
        resolved = (base / target.lstrip("/")).resolve()
        exists = resolved.is_dir() if target.endswith("/") else resolved.is_file()
        if not exists:
            problems.append(f"{where}: link to {target!r} does not resolve")
    return problems


def check_index(path: Path, root: Path, concepts: list[Path]) -> list[str]:
    where = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    raw, body = split_frontmatter(text)
    problems: list[str] = []
    if raw is not None and path.parent != root:
        problems.append(f"{where}: an index.md has no frontmatter outside the bundle root")
    if raw is not None and path.parent == root:
        for line in raw.splitlines():
            if line.split(":", 1)[0].strip() != "okf_version":
                problems.append(f"{where}: the root index may only carry `okf_version`")
    problems += check_links(path, root, body)
    linked = {
        ((root if t.startswith("/") else path.parent) / t.lstrip("/")).resolve()
        for t in LINK.findall(body)
    }
    for concept in concepts:
        if concept.parent == path.parent and concept.resolve() not in linked:
            problems.append(f"{where}: does not list {concept.name}")
    return problems


def check_log(path: Path, root: Path) -> list[str]:
    where = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    dates = DATE_HEADING.findall(text)
    problems = []
    if not dates:
        problems.append(f"{where}: no `## YYYY-MM-DD` date heading")
    if dates != sorted(dates, reverse=True):
        problems.append(f"{where}: date headings are not newest first")
    problems += check_links(path, root, text)
    return problems


def check_bundle(root: Path) -> list[str]:
    """Every problem found in the bundle at ``root``; empty when it is clean."""
    try:
        import yaml
    except ImportError as err:  # pragma: no cover - exercised only without PyYAML
        raise SystemExit("check_okf needs PyYAML: uv pip install pyyaml") from err
    root = root.resolve()
    if not root.is_dir():
        return [f"{root}: not a directory"]
    files = sorted(
        p for p in root.rglob("*.md") if not any(part.startswith(".") for part in p.parts)
    )
    concepts = [p for p in files if p.name not in RESERVED]
    problems: list[str] = []
    if not (root / "index.md").is_file():
        problems.append("index.md: the bundle root has no index.md")
    for path in files:
        if path.name == "index.md":
            problems += check_index(path, root, concepts)
        elif path.name == "log.md":
            problems += check_log(path, root)
        else:
            problems += check_concept(path, root, yaml)
    root_index = root / "index.md"
    if root_index.is_file():
        raw, _ = split_frontmatter(root_index.read_text(encoding="utf-8"))
        meta = yaml.safe_load(raw) if raw else None
        if not isinstance(meta, dict) or "okf_version" not in meta:
            problems.append("index.md: the bundle root should declare `okf_version`")
    for directory in sorted({p.parent for p in concepts}):
        if not (directory / "index.md").is_file():
            problems.append(f"{directory.relative_to(root).as_posix() or '.'}: no index.md")
    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "knowledge"
    problems = check_bundle(root)
    for line in problems:
        print(line)
    if problems:
        print(f"\n{len(problems)} problem(s) in {root}")
        return 1
    print(f"{root}: a clean OKF v0.2 bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
