"""Adapter for the CAS Schedule P gold mart (cas-schedule-p-data-model repo).

The data pipeline lives in that repo; this package only consumes its published
gold mart. The mart of record is ``mart_reserving_model_training``: long-format
loss observations keyed by company x line_of_business x accident_year x
development_age x statement_year. Schema (verified 2026-06-12 rebuild):

    company_code, company_name, line_of_business, statement_year, accident_year,
    development_age (years, 1-10), calendar_age, incurred_loss, cum_paid_loss,
    bulk_loss, earned_prem_net, earned_prem_direct, case_reserve, loss_ratio,
    paid_to_incurred_ratio, ata_factor_into_this_age, publish_id

Note: the mart's ``incurred_loss`` is gross of bulk+IBNR and its
``case_reserve`` is incurred - paid (so it also contains bulk). The adapter
derives ``reported_loss`` = incurred_loss - bulk_loss (paid + true case),
which is what Meyers' monograph calls "incurred".

Two ways to point this adapter at data (either directly or via the
IBNR_SCHEDULE_P_WAREHOUSE environment variable):

1. a local warehouse directory — parquet under ``warehouse/`` with the active
   publish chosen by ``warehouse/_active_manifest.json`` (the sibling checkout
   of cas-schedule-p-data-model);
2. a GitHub release spec ``github://<owner>/<repo>@<publish_id>`` — the data
   repo publishes each gold promote as a release tagged with its publish_id,
   carrying every gold table plus a ``manifest.json`` (asset name, sha256,
   bytes per table). Assets are downloaded once via the ``gh`` CLI (which
   supplies auth for the private repo) into a local cache
   (``~/.cache/ibnr`` or IBNR_CACHE_DIR), sha256-verified, then read locally.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import ibis

from ibnr.triangle.core import Triangle
from ibnr.triangle.io import from_long, resolve_backend

ENV_VAR = "IBNR_SCHEDULE_P_WAREHOUSE"
CACHE_ENV_VAR = "IBNR_CACHE_DIR"
GITHUB_SCHEME = "github://"
TRAINING_MART = "mart_reserving_model_training"

#: mart column -> triangle field name. Loss fields are cumulative; premiums,
#: reserves and bulk are eval-date snapshots carried along as fields.
DEFAULT_FIELDS = {
    "cum_paid_loss": "paid_loss",
    "incurred_loss": "incurred_loss",  # gross of bulk+IBNR
    "bulk_loss": "bulk_loss",
    "case_reserve": "case_reserve",
    "earned_prem_net": "earned_premium",
    "earned_prem_direct": "earned_premium_direct",
}

#: derived field: triangle field name -> ibis expression over the mart table
DERIVED_FIELDS = {
    # Meyers' "incurred": net of bulk+IBNR, i.e. paid + case reserves
    "reported_loss": lambda t: t.incurred_loss - t.bulk_loss,
}

SEGMENTS = ["company_code", "company_name", "line_of_business"]


def active_mart_path(warehouse: str | Path | None = None, mart: str = TRAINING_MART) -> Path:
    """Resolve a mart's parquet path — from a local warehouse's active-publish
    manifest, or from a cached (downloading if needed) GitHub release when
    ``warehouse`` is a ``github://owner/repo@publish_id`` spec."""
    source = _resolve_source(warehouse)
    if _is_github_spec(source):
        repo, tag = _parse_github_spec(str(source))
        return _release_asset(repo, tag, mart)
    warehouse = Path(source)
    manifest = json.loads((warehouse / "_active_manifest.json").read_text())
    rel = Path(manifest["tables"][mart])
    # manifest paths are relative to the data-model repo root ("warehouse\...")
    return warehouse.parent / rel


def active_publish_id(warehouse: str | Path | None = None) -> str:
    """The gold publish's version stamp — stamp this into every results
    artifact so figures trace back to an exact data publish."""
    source = _resolve_source(warehouse)
    if _is_github_spec(source):
        repo, tag = _parse_github_spec(str(source))
        manifest, _ = _release_manifest(repo, tag)
        return str(manifest["publish_id"])
    warehouse = Path(source)
    manifest = json.loads((warehouse / "_active_manifest.json").read_text())
    return str(manifest["publish_id"])


def _resolve_source(warehouse: str | Path | None) -> str | Path:
    if warehouse is None:
        warehouse = os.environ.get(ENV_VAR)
        if warehouse is None:
            raise ValueError(
                f"no warehouse given and {ENV_VAR} is not set; point at the "
                "cas-schedule-p-data-model warehouse directory or a "
                f"{GITHUB_SCHEME}owner/repo@publish_id release spec"
            )
    return warehouse


# -- GitHub release consumption ---------------------------------------------------
#
# The data repo publishes each gold promote as an immutable release tagged with
# its publish_id; assets are the gold tables plus manifest.json. We download
# through the gh CLI (it carries auth for the private repo), cache per
# (repo, publish_id), and verify sha256 against the manifest.


def _is_github_spec(source: str | Path) -> bool:
    return isinstance(source, str) and source.startswith(GITHUB_SCHEME)


def _parse_github_spec(spec: str) -> tuple[str, str]:
    """``github://owner/repo@publish_id`` -> (``owner/repo``, ``publish_id``)."""
    body = spec[len(GITHUB_SCHEME) :]
    repo, _, tag = body.partition("@")
    if not tag or repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(
            f"bad GitHub release spec {spec!r}; expected {GITHUB_SCHEME}owner/repo@publish_id"
        )
    return repo, tag


def _cache_dir(repo: str, tag: str) -> Path:
    root = os.environ.get(CACHE_ENV_VAR)
    root = Path(root) if root else Path.home() / ".cache" / "ibnr"
    return root / repo.replace("/", "__") / tag


def _gh_download(repo: str, tag: str, pattern: str, dest: Path) -> None:
    if shutil.which("gh") is None:
        raise RuntimeError(
            "the GitHub CLI (gh) is required to fetch data releases from the "
            f"private {repo} repo; install it and run `gh auth login`"
        )
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--pattern",
            pattern,
            "--dir",
            str(dest),
            "--clobber",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh release download failed for {repo}@{tag} ({pattern}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _release_manifest(repo: str, tag: str) -> tuple[dict, Path]:
    cache = _cache_dir(repo, tag)
    path = cache / "manifest.json"
    if not path.exists():
        _gh_download(repo, tag, "manifest.json", cache)
    manifest = json.loads(path.read_text())
    if str(manifest["publish_id"]) != tag:
        raise ValueError(
            f"release {repo}@{tag} carries manifest for publish "
            f"{manifest['publish_id']!r} — publishes are immutable, refusing to mix"
        )
    return manifest, cache


def _release_asset(repo: str, tag: str, table: str) -> Path:
    manifest, cache = _release_manifest(repo, tag)
    try:
        entry = manifest["tables"][table]
    except KeyError:
        raise KeyError(
            f"release {repo}@{tag} has no table {table!r}; available: {sorted(manifest['tables'])}"
        ) from None
    path = cache / entry["asset"]
    if not path.exists() or path.stat().st_size != int(entry["bytes"]):
        _gh_download(repo, tag, entry["asset"], cache)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {entry['asset']} from {repo}@{tag}: "
                f"got {digest}, manifest says {entry['sha256']}"
            )
    return path


def load_schedule_p(
    warehouse: str | Path | None = None,
    *,
    lines: list[str] | None = None,
    companies: list[str] | None = None,
    fields: dict[str, str] | None = None,
    derived: dict | None = None,
    backend: str | None = None,
) -> Triangle:
    """Load the reserving training mart as a Triangle.

    Mapping: origin_period = Jan 1 of accident_year; dev_lag = development_age
    in months; eval_date = Dec 31 of statement_year (annual statement date).
    Values are USD thousands. ``lines``/``companies`` filter line_of_business /
    company_code before materializing.
    """
    fields = fields or DEFAULT_FIELDS
    derived = DERIVED_FIELDS if derived is None else derived
    con = resolve_backend(backend)
    t = con.read_parquet(str(active_mart_path(warehouse)))
    if lines:
        t = t.filter(t.line_of_business.isin(lines))
    if companies:
        t = t.filter(t.company_code.isin(companies))
    t = t.select(
        *SEGMENTS,
        origin_period=ibis.date(t.accident_year, 1, 1),
        dev_lag=(t.development_age * 12).cast("int64"),
        eval_date=ibis.date(t.statement_year, 12, 31),
        **{new: t[old] for old, new in fields.items()},
        **{name: expr(t) for name, expr in derived.items()},
    )
    return from_long(
        t,
        fields=[*fields.values(), *derived],
        measure="cumulative",
        origin_grain="Y",
        dev_grain="Y",
        units="USD thousands",
    )
