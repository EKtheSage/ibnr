"""Adapter for the CAS Schedule P gold mart (cas-schedule-p-data-model repo).

The data pipeline lives in that repo; this package only consumes its published
gold mart. The mart of record is ``mart_reserving_model_training``: long-format
loss observations keyed by company x line_of_business x accident_year x
development_age x statement_year. Schema (verified 2026-06-12 rebuild):

    company_code, company_name, line_of_business, statement_year, accident_year,
    development_age (years, 1-10), calendar_age, incurred_loss, cum_paid_loss,
    bulk_loss, earned_prem_net, earned_prem_direct, case_reserve, loss_ratio,
    paid_to_incurred_ratio, ata_factor_into_this_age, publish_id

Note: the mart's ``incurred_loss`` is gross of bulk+IBNR; its ``case_reserve``
is reported minus paid, i.e. (incurred_loss - bulk_loss) - cum_paid_loss, so
it contains NO bulk (measured cell-exact on all 149,550 mart cells of publish
20260613_041006; incurred - paid matches only on the cells where bulk is
zero). The adapter derives ``reported_loss`` = incurred_loss - bulk_loss
(equivalently cum_paid_loss + case_reserve), which is what Meyers' monograph
calls "incurred".

Two ways to point this adapter at data (either directly or via the
IBNR_SCHEDULE_P_WAREHOUSE environment variable; when neither is given the
default is ``DEFAULT_SOURCE`` - the newest GitHub release, ``@latest``
resolved to its concrete publish_id up front):

1. a local warehouse directory - parquet under ``warehouse/`` with the active
   publish chosen by ``warehouse/_active_manifest.json`` (the sibling checkout
   of cas-schedule-p-data-model);
2. a GitHub release spec ``github://<owner>/<repo>@<publish_id>`` - the data
   repo publishes each gold promote as a release tagged with its publish_id,
   carrying every gold table plus a ``manifest.json`` (asset name, sha256,
   bytes per table). Assets are downloaded once into a local cache
   (``~/.cache/ibnr`` or IBNR_CACHE_DIR), sha256-verified, then read locally.

Transport: the data repo is PUBLIC, so the primary path is plain anonymous
HTTPS from the standard library - no ``gh``, no auth, no login. The ``gh`` CLI
is kept only as a FALLBACK, tried when the anonymous request fails, which in
practice means one of two things: GitHub's API rate limit on unauthenticated
callers (60 requests/hour/IP, and only ``@latest`` resolution touches the API
at all), or a repo that has gone private. So a fresh clone with no GitHub
tooling installed works, and an authenticated developer keeps a way through a
rate limit. When both paths fail the error names both and the fix for each.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import ibis

from ibnr.triangle.core import Triangle
from ibnr.triangle.io import from_long, resolve_backend

ENV_VAR = "IBNR_SCHEDULE_P_WAREHOUSE"
CACHE_ENV_VAR = "IBNR_CACHE_DIR"
GITHUB_SCHEME = "github://"
TRAINING_MART = "mart_reserving_model_training"

#: where the data comes from when nothing else is specified: the newest gold
#: publish on GitHub. ``@latest`` resolves to a concrete publish_id before
#: anything is cached or read, so results artifacts always stamp the exact
#: publish even in default-configured dev.
DEFAULT_SOURCE = "github://EKtheSage/cas-schedule-p-data-model@latest"
LATEST = "latest"

#: GitHub's API rejects requests that send no User-Agent.
_USER_AGENT = "ibnr (python-urllib)"
#: seconds before a stalled connection raises instead of hanging a study run
_HTTP_TIMEOUT = 60
_CHUNK = 1 << 20

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
    """Resolve a mart's parquet path - from a local warehouse's active-publish
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
    """The gold publish's version stamp - stamp this into every results
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
    """Explicit argument > env var > the GitHub data repo's newest publish."""
    if warehouse is None:
        warehouse = os.environ.get(ENV_VAR) or DEFAULT_SOURCE
    return warehouse


def pinned_source(warehouse: str | Path | None = None, mart: str = TRAINING_MART) -> str:
    """Resolve a warehouse argument to a CONCRETE, worker-safe source string.

    For a GitHub spec, ``@latest`` is pinned to its publish_id and the mart
    asset is downloaded into the cache up front. The parallel harness hands
    THIS string to its workers, so they never re-resolve ``@latest`` (a race
    against a release published mid-run would split the study across two data
    versions) and never hit the network concurrently - they only read the
    local cache. Local warehouse paths pass through unchanged.
    """
    source = _resolve_source(warehouse)
    if _is_github_spec(source):
        repo, tag = _parse_github_spec(str(source))
        _release_asset(repo, tag, mart)  # warm the cache before workers spawn
        return f"{GITHUB_SCHEME}{repo}@{tag}"
    return str(source)


# -- GitHub release consumption ---------------------------------------------------
#
# The data repo publishes each gold promote as an immutable release tagged with
# its publish_id; assets are the gold tables plus manifest.json. We cache per
# (repo, publish_id) and verify sha256 against the manifest.
#
# Two transports, in this order, behind `_latest_tag` and `_download_asset`:
#   1. anonymous HTTPS (stdlib urllib) - the repo is public, so this needs no
#      tooling and no credentials;
#   2. the gh CLI - only when (1) raised, which is the API rate limit or a repo
#      that has gone private.
# Nothing else in this module knows which one answered.


def _is_github_spec(source: str | Path) -> bool:
    return isinstance(source, str) and source.startswith(GITHUB_SCHEME)


def _parse_github_spec(spec: str) -> tuple[str, str]:
    """``github://owner/repo@publish_id`` -> (``owner/repo``, ``publish_id``).

    ``@latest`` is resolved to the repo's newest release tag (one gh call,
    cached per process) so caching and provenance always see a concrete id.
    """
    body = spec[len(GITHUB_SCHEME) :]
    repo, _, tag = body.partition("@")
    if not tag or repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(
            f"bad GitHub release spec {spec!r}; expected {GITHUB_SCHEME}owner/repo@publish_id"
        )
    if tag == LATEST:
        tag = _latest_tag(repo)
    return repo, tag


_LATEST_TAGS: dict[str, str] = {}


def _api_latest_tag(repo: str) -> str:
    """The newest release's tag, straight off the public API. No auth."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/latest",
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:  # noqa: S310
        payload = json.loads(response.read())
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        raise RuntimeError(f"the latest release of {repo} carries no tag_name")
    return tag


def _gh_latest_tag(repo: str) -> str:
    """The same answer through the gh CLI - the authenticated fallback."""
    if shutil.which("gh") is None:
        raise RuntimeError("the GitHub CLI (gh) is not installed")
    result = subprocess.run(
        ["gh", "release", "view", "--repo", repo, "--json", "tagName", "--jq", ".tagName"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "gh returned nothing")
    return result.stdout.strip()


def _latest_tag(repo: str) -> str:
    """Resolve ``@latest`` to a concrete publish_id, cached per process.

    Anonymous HTTPS first; gh only if that raised. The API call is the ONLY
    unauthenticated request this module makes against api.github.com, so it is
    also the only one exposed to the 60/hour rate limit - which is exactly why
    a published run should pin ``@<publish_id>`` and never resolve at all.
    """
    if repo not in _LATEST_TAGS:
        try:
            _LATEST_TAGS[repo] = _api_latest_tag(repo)
        except (OSError, http.client.HTTPException, ValueError, RuntimeError) as http_error:
            # Deliberately wide: URLError and mid-stream ConnectionResetError are
            # OSError, a garbage API payload is json's ValueError, and a payload
            # with no tag_name is _api_latest_tag's own RuntimeError - every way
            # the anonymous route can fail should reach the fallback, not the user.
            try:
                _LATEST_TAGS[repo] = _gh_latest_tag(repo)
            except Exception as gh_error:
                raise RuntimeError(
                    f"could not resolve @{LATEST} for {repo} by either route.\n"
                    f"  anonymous HTTPS (api.github.com) failed: {http_error}\n"
                    f"    fix: pin a concrete publish_id instead of @{LATEST} "
                    "(no API call at all), or wait out GitHub's 60/hour "
                    "unauthenticated rate limit, or check network access.\n"
                    f"  gh CLI fallback failed: {gh_error}\n"
                    "    fix: install the GitHub CLI and run `gh auth login`."
                ) from gh_error
    return _LATEST_TAGS[repo]


def _cache_dir(repo: str, tag: str) -> Path:
    root = os.environ.get(CACHE_ENV_VAR)
    root = Path(root) if root else Path.home() / ".cache" / "ibnr"
    return root / repo.replace("/", "__") / tag


def _asset_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def _http_download(url: str, dest: Path) -> None:
    """Stream one asset to ``dest``, written atomically.

    The bytes land in a ``.part`` sibling and are renamed only after the
    response has been read to the end, so an interrupted download can never be
    mistaken for a complete one by the size pre-check in ``_release_asset``.
    Content is NOT verified here - the manifest is the authority for that, and
    it lives one level up where it always has.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response,  # noqa: S310
            tmp.open("wb") as fh,
        ):
            while chunk := response.read(_CHUNK):
                fh.write(chunk)
        tmp.replace(dest)
    finally:
        # after a successful replace the temp name is gone, so this only ever
        # clears the leavings of a failed or interrupted download
        tmp.unlink(missing_ok=True)


def _gh_download(repo: str, tag: str, pattern: str, dest: Path) -> None:
    """The authenticated fallback transport. Not the default path."""
    if shutil.which("gh") is None:
        raise RuntimeError("the GitHub CLI (gh) is not installed")
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
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "gh returned nothing")


def _download_asset(repo: str, tag: str, asset: str, dest: Path) -> None:
    """Fetch one release asset into the cache directory ``dest``.

    Anonymous HTTPS first (the repo is public); gh only if that raised. Both
    write the file under its published name, so the caller - and the sha256
    check in ``_release_asset`` - cannot tell which route answered.
    """
    dest.mkdir(parents=True, exist_ok=True)
    url = _asset_url(repo, tag, asset)
    try:
        _http_download(url, dest / asset)
        return
    except (OSError, http.client.HTTPException, ValueError) as http_error:
        # Wide on purpose: a connection reset mid-stream is OSError and an
        # IncompleteRead is HTTPException - the flaky-network cases the fallback
        # exists for must actually reach it. Atomicity holds regardless
        # (_http_download's finally clears the .part file).
        try:
            _gh_download(repo, tag, asset, dest)
        except Exception as gh_error:
            raise RuntimeError(
                f"could not download {asset!r} from release {repo}@{tag} by either route.\n"
                f"  anonymous HTTPS ({url}) failed: {http_error}\n"
                "    fix: check network/proxy access to github.com, and that the "
                "release and that asset exist.\n"
                f"  gh CLI fallback failed: {gh_error}\n"
                "    fix: install the GitHub CLI and run `gh auth login` (needed "
                "only if the repo is private or anonymous access is blocked)."
            ) from gh_error


def _release_manifest(repo: str, tag: str) -> tuple[dict, Path]:
    cache = _cache_dir(repo, tag)
    path = cache / "manifest.json"
    if not path.exists():
        _download_asset(repo, tag, "manifest.json", cache)
    manifest = json.loads(path.read_text())
    if str(manifest["publish_id"]) != tag:
        raise ValueError(
            f"release {repo}@{tag} carries manifest for publish "
            f"{manifest['publish_id']!r} - publishes are immutable, refusing to mix"
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
        _download_asset(repo, tag, entry["asset"], cache)
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
