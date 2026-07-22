"""GitHub-release consumption of the Schedule P gold publishes.

Unit tests run against a fabricated local cache (no network, no gh). The one
live test downloads only the tiny manifest.json and skips without gh auth.

What this file protects: the default data path. ``load_schedule_p()`` with no
source resolves ``github://EKtheSage/cas-schedule-p-data-model@latest`` through
the gh CLI, caches under ``IBNR_CACHE_DIR`` (default ``~/.cache/ibnr``), and
verifies each asset against the release's ``manifest.json``. The invariants
that matter for reproducible experiments are:

* **Immutability.** A release tag names one publish forever. The manifest's
  ``publish_id`` must equal the tag it was fetched under, and a resolved
  ``@latest`` is always reported as the concrete id — never the string
  "latest" — so results CSVs can be stamped with something re-fetchable.
* **Integrity.** Size mismatch on disk triggers a re-download; a sha256
  mismatch after download raises AND deletes the corrupt file, so a poisoned
  cache can never be silently reused.
* **Offline-first.** A complete, verified cache must serve without touching the
  network at all.

Test design: every unit test fabricates its own cache directory under
``tmp_path`` and monkeypatches ``_gh_download`` to a function that fails loudly
if called (``forbid_download``) or to a controlled stub. That makes the "no
network" claims assertions rather than assumptions, and keeps the whole file
runnable in CI without gh auth.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from ibnr.data import schedule_p as sp

REPO = "EKtheSage/cas-schedule-p-data-model"
# A real, already-published release tag (= its publish_id). Concrete rather
# than "latest" so tests never depend on what the newest publish happens to be.
TAG = "20260613_041006"
SPEC = f"github://{REPO}@{TAG}"


def make_cache(tmp_path, monkeypatch, *, payload=b"parquet-bytes", tag=TAG, lie_sha=False):
    """Fabricate a valid (or deliberately corrupt) release cache.

    Mirrors the on-disk layout the loader expects: ``$IBNR_CACHE_DIR/
    <owner>__<repo>/<tag>/{manifest.json, <asset>}``. ``lie_sha`` writes a
    manifest digest that cannot match the payload (integrity failure path);
    ``tag`` differing from ``TAG`` fakes a mutated release (immutability path).
    """
    monkeypatch.setenv(sp.CACHE_ENV_VAR, str(tmp_path))
    cache = tmp_path / REPO.replace("/", "__") / TAG
    cache.mkdir(parents=True)
    sha = hashlib.sha256(payload).hexdigest()
    if lie_sha:
        sha = "0" * 64
    manifest = {
        "publish_id": tag,
        "tables": {
            sp.TRAINING_MART: {"asset": "mart.parquet", "sha256": sha, "bytes": len(payload)}
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    (cache / "mart.parquet").write_bytes(payload)
    return cache


def forbid_download(monkeypatch):
    """Turn any network fetch into a hard failure, so "served from cache" is
    asserted rather than merely plausible."""

    def boom(*a, **kw):
        raise AssertionError("network download attempted")

    monkeypatch.setattr(sp, "_gh_download", boom)


def test_parse_github_spec():
    """The ``github://owner/repo@tag`` spec grammar: valid specs split into
    (repo, tag), and every malformed shape is rejected up front rather than
    producing a nonsense cache directory."""
    assert sp._parse_github_spec(SPEC) == (REPO, TAG)
    # missing tag / missing repo half / too many path parts / empty owner
    for bad in ["github://no-tag", "github://only-owner@x", "github://a/b/c@x", "github://@x"]:
        with pytest.raises(ValueError, match="release spec"):
            sp._parse_github_spec(bad)


def test_cached_release_needs_no_network(tmp_path, monkeypatch):
    """Offline-first: a complete, sha-valid cache resolves both the asset path
    and the publish id without any download."""
    cache = make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    path = sp.active_mart_path(SPEC)
    assert path == cache / "mart.parquet"
    assert sp.active_publish_id(SPEC) == TAG


def test_env_var_accepts_github_spec(tmp_path, monkeypatch):
    """``IBNR_SCHEDULE_P_WAREHOUSE`` is not local-path-only: it accepts a
    release spec, so pinning an experiment to a publish needs no code change."""
    make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    monkeypatch.setenv(sp.ENV_VAR, SPEC)
    assert sp.active_publish_id() == TAG


def test_size_mismatch_triggers_redownload(tmp_path, monkeypatch):
    """A truncated/stale cached asset (byte count disagrees with the manifest)
    is re-fetched rather than used. Size is the cheap pre-check; sha256 is the
    authoritative one, verified after the download."""
    payload = b"the-real-payload"
    cache = make_cache(tmp_path, monkeypatch, payload=payload)
    (cache / "mart.parquet").write_bytes(b"stale")  # wrong size on disk

    calls = []

    def fake_download(repo, tag, pattern, dest):
        calls.append(pattern)
        (dest / "mart.parquet").write_bytes(payload)

    monkeypatch.setattr(sp, "_gh_download", fake_download)
    path = sp.active_mart_path(SPEC)
    # exactly one fetch, and only for the asset actually needed
    assert calls == ["mart.parquet"]
    assert path.read_bytes() == payload


def test_sha_mismatch_after_download_raises(tmp_path, monkeypatch):
    """Integrity gate: content that fails the manifest's sha256 raises, and the
    bad file is unlinked so a later run cannot pick up a poisoned cache."""
    cache = make_cache(tmp_path, monkeypatch, lie_sha=True)
    (cache / "mart.parquet").unlink()  # force the download path

    monkeypatch.setattr(
        sp, "_gh_download", lambda r, t, p, d: (d / "mart.parquet").write_bytes(b"parquet-bytes")
    )
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        sp.active_mart_path(SPEC)
    assert not (cache / "mart.parquet").exists()  # corrupt file removed


def test_publish_id_tag_mismatch_refused(tmp_path, monkeypatch):
    """Immutability gate: if the manifest's ``publish_id`` disagrees with the
    tag it was fetched under, the release was mutated and any experiment
    stamped with that id would be unreproducible — so refuse."""
    make_cache(tmp_path, monkeypatch, tag="some_other_publish")
    forbid_download(monkeypatch)
    with pytest.raises(ValueError, match="immutable"):
        sp.active_publish_id(SPEC)


def test_unknown_table_lists_available(tmp_path, monkeypatch):
    """Asking for a table the publish does not contain fails with a KeyError
    that names it (and, in the message, what is available) — publishes evolve,
    so this is the discoverability path."""
    make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    with pytest.raises(KeyError, match="no table 'nope'"):
        sp.active_mart_path(SPEC, mart="nope")


def test_default_source_is_latest_github_release(tmp_path, monkeypatch):
    """With no env var set, resolution falls through to ``DEFAULT_SOURCE``
    (the GitHub release), not to a local warehouse — the local checkout is only
    ever an override."""
    make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    monkeypatch.delenv(sp.ENV_VAR, raising=False)
    monkeypatch.setattr(sp, "DEFAULT_SOURCE", SPEC)  # concrete tag: no gh needed
    assert sp.active_publish_id() == TAG


def test_latest_resolves_once_then_uses_cache(tmp_path, monkeypatch):
    """``@latest`` resolves to a concrete tag and then serves from cache: the
    only network call is the cheap tag lookup, and the reported publish id is
    always the concrete one so runs stay reproducible."""
    make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    calls = []

    def fake_latest(repo):
        calls.append(repo)
        return TAG

    monkeypatch.setattr(sp, "_latest_tag", fake_latest)
    spec = f"github://{REPO}@latest"
    assert sp.active_mart_path(spec).name == "mart.parquet"
    assert sp.active_publish_id(spec) == TAG  # concrete id, never "latest"
    # one tag lookup per public call; the asset itself never leaves the cache
    assert calls == [REPO, REPO]


def _gh_authed() -> bool:
    """gh CLI present AND authenticated — the only precondition for the one
    test below that really talks to GitHub."""
    if shutil.which("gh") is None:
        return False
    return subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _gh_authed(), reason="needs gh CLI authenticated to GitHub")
def test_live_manifest_fetch(tmp_path, monkeypatch):
    """The one end-to-end check that the real release actually exists and its
    manifest parses. Deliberately asks only for the publish id so the fetch is
    the tiny manifest, not the multi-MB parquet; a throwaway cache dir keeps it
    from touching the developer's real cache."""
    monkeypatch.setenv(sp.CACHE_ENV_VAR, str(tmp_path))
    assert sp.active_publish_id(SPEC) == TAG  # downloads only manifest.json (<5 KB)
