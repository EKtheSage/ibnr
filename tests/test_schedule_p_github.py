"""GitHub-release consumption of the Schedule P gold publishes.

Unit tests run against a fabricated local cache (no network, no gh). The one
live test downloads only the tiny manifest.json and skips without gh auth.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from ibnr.data import schedule_p as sp

REPO = "EKtheSage/cas-schedule-p-data-model"
TAG = "20260613_041006"
SPEC = f"github://{REPO}@{TAG}"


def make_cache(tmp_path, monkeypatch, *, payload=b"parquet-bytes", tag=TAG, lie_sha=False):
    """Fabricate a valid (or deliberately corrupt) release cache."""
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
    def boom(*a, **kw):
        raise AssertionError("network download attempted")

    monkeypatch.setattr(sp, "_gh_download", boom)


def test_parse_github_spec():
    assert sp._parse_github_spec(SPEC) == (REPO, TAG)
    for bad in ["github://no-tag", "github://only-owner@x", "github://a/b/c@x", "github://@x"]:
        with pytest.raises(ValueError, match="release spec"):
            sp._parse_github_spec(bad)


def test_cached_release_needs_no_network(tmp_path, monkeypatch):
    cache = make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    path = sp.active_mart_path(SPEC)
    assert path == cache / "mart.parquet"
    assert sp.active_publish_id(SPEC) == TAG


def test_env_var_accepts_github_spec(tmp_path, monkeypatch):
    make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    monkeypatch.setenv(sp.ENV_VAR, SPEC)
    assert sp.active_publish_id() == TAG


def test_size_mismatch_triggers_redownload(tmp_path, monkeypatch):
    payload = b"the-real-payload"
    cache = make_cache(tmp_path, monkeypatch, payload=payload)
    (cache / "mart.parquet").write_bytes(b"stale")  # wrong size on disk

    calls = []

    def fake_download(repo, tag, pattern, dest):
        calls.append(pattern)
        (dest / "mart.parquet").write_bytes(payload)

    monkeypatch.setattr(sp, "_gh_download", fake_download)
    path = sp.active_mart_path(SPEC)
    assert calls == ["mart.parquet"]
    assert path.read_bytes() == payload


def test_sha_mismatch_after_download_raises(tmp_path, monkeypatch):
    cache = make_cache(tmp_path, monkeypatch, lie_sha=True)
    (cache / "mart.parquet").unlink()  # force the download path

    monkeypatch.setattr(
        sp, "_gh_download", lambda r, t, p, d: (d / "mart.parquet").write_bytes(b"parquet-bytes")
    )
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        sp.active_mart_path(SPEC)
    assert not (cache / "mart.parquet").exists()  # corrupt file removed


def test_publish_id_tag_mismatch_refused(tmp_path, monkeypatch):
    make_cache(tmp_path, monkeypatch, tag="some_other_publish")
    forbid_download(monkeypatch)
    with pytest.raises(ValueError, match="immutable"):
        sp.active_publish_id(SPEC)


def test_unknown_table_lists_available(tmp_path, monkeypatch):
    make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    with pytest.raises(KeyError, match="no table 'nope'"):
        sp.active_mart_path(SPEC, mart="nope")


def _gh_authed() -> bool:
    if shutil.which("gh") is None:
        return False
    return subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _gh_authed(), reason="needs gh CLI authenticated to GitHub")
def test_live_manifest_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv(sp.CACHE_ENV_VAR, str(tmp_path))
    assert sp.active_publish_id(SPEC) == TAG  # downloads only manifest.json (<5 KB)
