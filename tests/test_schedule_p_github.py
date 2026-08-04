"""GitHub-release consumption of the Schedule P gold publishes.

Unit tests run against a fabricated local cache (no network, no gh). The one
live test downloads only the tiny manifest.json and is marked ``network``.

What this file protects: the default data path. ``load_schedule_p()`` with no
source resolves ``github://EKtheSage/cas-schedule-p-data-model@latest``, caches
under ``IBNR_CACHE_DIR`` (default ``~/.cache/ibnr``), and verifies each asset
against the release's ``manifest.json``. The invariants that matter for
reproducible experiments are:

* **Immutability.** A release tag names one publish forever. The manifest's
  ``publish_id`` must equal the tag it was fetched under, and a resolved
  ``@latest`` is always reported as the concrete id - never the string
  "latest" - so results CSVs can be stamped with something re-fetchable.
* **Integrity.** Size mismatch on disk triggers a re-download; a sha256
  mismatch after download raises AND deletes the corrupt file, so a poisoned
  cache can never be silently reused.
* **Offline-first.** A complete, verified cache must serve without touching the
  network at all.
* **Transport.** The data repo is public, so anonymous HTTPS is the primary
  route and gh is only a fallback. Covered in ``TestTransport`` below, which
  patches ``urllib``/``subprocess`` rather than the dispatcher, because a test
  that stubs the dispatcher cannot see which route was chosen.

Test design: every unit test fabricates its own cache directory under
``tmp_path`` and monkeypatches ``_download_asset`` - the transport dispatcher -
to a function that fails loudly if called (``forbid_download``) or to a
controlled stub. That makes the "no network" claims assertions rather than
assumptions, and keeps the whole file runnable in CI without gh auth. The
transport tests are the deliberate exception: they leave the dispatcher alone
so that the real route selection runs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
from pathlib import Path

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
    asserted rather than merely plausible.

    Patched at ``_download_asset``, the transport dispatcher, so this covers
    BOTH routes (anonymous HTTPS and the gh fallback) with one stub.
    """

    def boom(*a, **kw):
        raise AssertionError("network download attempted")

    monkeypatch.setattr(sp, "_download_asset", boom)


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

    monkeypatch.setattr(sp, "_download_asset", fake_download)
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
        sp, "_download_asset", lambda r, t, p, d: (d / "mart.parquet").write_bytes(b"parquet-bytes")
    )
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        sp.active_mart_path(SPEC)
    assert not (cache / "mart.parquet").exists()  # corrupt file removed


def test_publish_id_tag_mismatch_refused(tmp_path, monkeypatch):
    """Immutability gate: if the manifest's ``publish_id`` disagrees with the
    tag it was fetched under, the release was mutated and any experiment
    stamped with that id would be unreproducible - so refuse."""
    make_cache(tmp_path, monkeypatch, tag="some_other_publish")
    forbid_download(monkeypatch)
    with pytest.raises(ValueError, match="immutable"):
        sp.active_publish_id(SPEC)


def test_unknown_table_lists_available(tmp_path, monkeypatch):
    """Asking for a table the publish does not contain fails with a KeyError
    that names it (and, in the message, what is available) - publishes evolve,
    so this is the discoverability path."""
    make_cache(tmp_path, monkeypatch)
    forbid_download(monkeypatch)
    with pytest.raises(KeyError, match="no table 'nope'"):
        sp.active_mart_path(SPEC, mart="nope")


def test_default_source_is_latest_github_release(tmp_path, monkeypatch):
    """With no env var set, resolution falls through to ``DEFAULT_SOURCE``
    (the GitHub release), not to a local warehouse - the local checkout is only
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


# -- transport: anonymous HTTPS first, gh only as a fallback ---------------------
#
# These tests patch `urllib.request.urlopen`, `subprocess.run` and `shutil.which`
# - the two transports' OWN primitives - and deliberately leave
# `_download_asset` / `_latest_tag` alone, so the real route selection runs.
# Stubbing the dispatcher instead would make every one of them pass whichever
# route the code actually took, which is the whole question. Each was confirmed
# to FAIL against the pre-change gh-only module.


class FakeResponse:
    """The slice of an ``http.client.HTTPResponse`` this module uses: a context
    manager with a chunked ``read``."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        end = len(self._payload) if size is None or size < 0 else self._pos + size
        chunk = self._payload[self._pos : end]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def urls(monkeypatch):
    """Record every URL urlopen is asked for. Tests append to ``urls.serve`` /
    set ``urls.error`` to script the response."""

    class Recorder(list):
        serve: bytes | None = None
        error: Exception | None = None

    recorder = Recorder()

    def fake_urlopen(request, timeout=None):
        recorder.append(request.full_url)
        assert timeout == sp._HTTP_TIMEOUT, "every request must carry the module's timeout"
        assert request.get_header("User-agent"), "GitHub rejects requests with no User-Agent"
        if recorder.error is not None:
            raise recorder.error
        return FakeResponse(recorder.serve or b"")

    monkeypatch.setattr(sp.urllib.request, "urlopen", fake_urlopen)
    return recorder


@pytest.fixture
def no_gh(monkeypatch):
    """gh is neither installed nor runnable: any attempt to use it fails the
    test rather than silently succeeding or silently skipping."""

    def boom(*a, **kw):
        raise AssertionError("the gh CLI was invoked")

    monkeypatch.setattr(sp.shutil, "which", boom)
    monkeypatch.setattr(sp.subprocess, "run", boom)


@pytest.fixture(autouse=True)
def fresh_latest_cache(monkeypatch):
    """``_LATEST_TAGS`` is a process-global memo; without this a test that
    resolves ``@latest`` would be answered by whatever an earlier test (or an
    earlier run in the same session) left behind."""
    monkeypatch.setattr(sp, "_LATEST_TAGS", {})


def test_latest_resolves_anonymously_without_gh(tmp_path, monkeypatch, urls, no_gh):
    """(a) ``@latest`` goes to the public API over plain HTTPS and gh is never
    reached. The repo is public, so needing gh installed to answer this was the
    whole defect. Driven through the public ``active_publish_id`` so the real
    source-parsing and dispatch run, not the private helper alone."""
    make_cache(tmp_path, monkeypatch)
    urls.serve = json.dumps({"tag_name": TAG}).encode()
    spec = f"github://{REPO}@latest"
    assert sp.active_publish_id(spec) == TAG
    assert urls == [f"https://api.github.com/repos/{REPO}/releases/latest"]
    # memoized: a second public call must not re-hit the API
    assert sp.active_publish_id(spec) == TAG
    assert len(urls) == 1


def test_asset_downloads_anonymously_and_verifies(tmp_path, monkeypatch, urls, no_gh):
    """(b) A real fetch goes through urllib, lands under the published asset
    name, and passes the manifest's sha256 - the verification path is untouched
    by the transport change, so it must still be what admits the file."""
    payload = b"the-real-payload"
    cache = make_cache(tmp_path, monkeypatch, payload=payload)
    (cache / "mart.parquet").unlink()  # force the download branch
    urls.serve = payload

    path = sp.active_mart_path(SPEC)

    assert urls == [f"https://github.com/{REPO}/releases/download/{TAG}/mart.parquet"]
    assert path == cache / "mart.parquet"
    assert path.read_bytes() == payload
    # atomic write: the temp file is not left behind for the size pre-check to
    # trip over on the next run
    assert not list(cache.glob("*.part"))


def test_download_is_atomic_when_the_stream_dies(tmp_path, monkeypatch, urls, no_gh):
    """A half-read response leaves NOTHING at the final name. Without the
    temp-then-rename dance the partial file would sit there, and the size
    pre-check would re-fetch it - but a sha of the wrong length is the only
    thing standing between a truncated parquet and a study."""
    payload = b"the-real-payload"
    cache = make_cache(tmp_path, monkeypatch, payload=payload)
    (cache / "mart.parquet").unlink()

    class DyingResponse(FakeResponse):
        def read(self, size=-1):
            if self._pos:  # first chunk lands, then the connection drops
                raise TimeoutError("connection reset")
            return super().read(size)

    monkeypatch.setattr(
        sp.urllib.request, "urlopen", lambda request, timeout=None: DyingResponse(payload)
    )
    with pytest.raises(RuntimeError, match="either route"):
        sp.active_mart_path(SPEC)
    assert not (cache / "mart.parquet").exists()
    assert not list(cache.glob("*.part"))


def test_gh_is_the_fallback_when_https_fails(tmp_path, monkeypatch, urls):
    """(c) When anonymous HTTPS raises - a rate limit, or a repo gone private -
    gh is tried, and its download is verified by exactly the same sha256 gate."""
    payload = b"the-real-payload"
    cache = make_cache(tmp_path, monkeypatch, payload=payload)
    (cache / "mart.parquet").unlink()
    urls.error = urllib.error.HTTPError(
        "https://github.com/...", 403, "rate limit exceeded", {}, None
    )

    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        # gh writes into the directory named by --dir, under the asset's own name
        Path(cmd[cmd.index("--dir") + 1]).joinpath("mart.parquet").write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sp.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sp.subprocess, "run", fake_run)

    path = sp.active_mart_path(SPEC)

    assert len(urls) == 1, "HTTPS must be TRIED first, not skipped"
    assert commands and commands[0][:3] == ["gh", "release", "download"]
    assert TAG in commands[0] and "mart.parquet" in commands[0]
    assert path.read_bytes() == payload


def test_gh_is_the_fallback_when_the_stream_dies_mid_download(tmp_path, monkeypatch, urls):
    """(c), flaky-network half. A connection reset AFTER a 200 - the failure the
    fallback most exists for - is OSError, not URLError, so a fallback keyed to
    ``(URLError, TimeoutError)`` alone let it surface raw. Confirmed failing
    before the except tuple was widened."""
    payload = b"the-real-payload"
    cache = make_cache(tmp_path, monkeypatch, payload=payload)
    (cache / "mart.parquet").unlink()

    class DyingResponse(FakeResponse):
        def read(self, size=-1):
            if self._pos:
                raise ConnectionResetError("connection reset by peer")
            return super().read(size)

    monkeypatch.setattr(
        sp.urllib.request, "urlopen", lambda request, timeout=None: DyingResponse(payload)
    )

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--dir") + 1]).joinpath("mart.parquet").write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sp.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sp.subprocess, "run", fake_run)

    path = sp.active_mart_path(SPEC)

    assert path.read_bytes() == payload
    assert not list(cache.glob("*.part"))  # the dead HTTPS attempt left nothing behind


def test_gh_is_the_fallback_when_the_api_answers_garbage(monkeypatch, urls):
    """(c), garbage-payload half. A proxy's HTML error page is a 200 that fails
    json parsing (ValueError), which must reach the fallback rather than surface
    as a bare JSONDecodeError. Confirmed failing before the widening."""
    urls.serve = b"<html>proxy says no</html>"
    monkeypatch.setattr(sp.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        sp.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, f"{TAG}\n", ""),
    )
    assert sp._latest_tag(REPO) == TAG
    assert len(urls) == 1, "HTTPS must be TRIED first, not skipped"


def test_latest_falls_back_to_gh(monkeypatch, urls):
    """(c), tag-resolution half. The API's 60/hour unauthenticated rate limit is
    the realistic failure here, and an authenticated gh is the way through it."""
    urls.error = urllib.error.HTTPError(
        "https://api.github.com/...", 403, "rate limit exceeded", {}, None
    )
    monkeypatch.setattr(sp.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        sp.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, f"{TAG}\n", ""),
    )
    assert sp._latest_tag(REPO) == TAG
    assert len(urls) == 1


def test_both_routes_failing_names_both(tmp_path, monkeypatch, urls):
    """(d) The terminal error is the only thing a stuck user reads, so it names
    both routes, the URL that was tried, and the distinct fix for each. Naming
    only gh would send a user to `gh auth login` for what is usually a proxy."""
    make_cache(tmp_path, monkeypatch)
    (tmp_path / REPO.replace("/", "__") / TAG / "mart.parquet").unlink()
    urls.error = urllib.error.URLError("getaddrinfo failed")
    monkeypatch.setattr(sp.shutil, "which", lambda name: None)  # gh not installed

    with pytest.raises(RuntimeError) as excinfo:
        sp.active_mart_path(SPEC)

    message = str(excinfo.value)
    assert "anonymous HTTPS" in message and "gh CLI fallback" in message
    assert f"https://github.com/{REPO}/releases/download/{TAG}/mart.parquet" in message
    assert "getaddrinfo failed" in message  # what actually went wrong
    assert "github.com" in message and "gh auth login" in message  # a fix for each


def test_both_routes_failing_on_latest_names_both(monkeypatch, urls):
    """(d), tag-resolution half - and the fix differs from the download one:
    the way past a rate limit is to pin a publish_id, which needs no API call
    at all, so the message says that rather than only 'install gh'."""
    urls.error = urllib.error.URLError("getaddrinfo failed")
    monkeypatch.setattr(sp.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError) as excinfo:
        sp._latest_tag(REPO)

    message = str(excinfo.value)
    assert "anonymous HTTPS" in message and "gh CLI fallback" in message
    assert "publish_id" in message and "gh auth login" in message


@pytest.mark.network
def test_live_anonymous_manifest_fetch(tmp_path, monkeypatch):
    """The one end-to-end check that the real release exists, is reachable
    WITHOUT credentials, and parses.

    Deselected by default (``addopts``); run it with ``pytest -m network``.
    ``shutil.which`` is stubbed to None so gh cannot rescue the fetch - that is
    what makes this a test of anonymous access rather than of the developer's
    ``gh auth login``. Asks only for the publish id, so the fetch is the tiny
    manifest and not the multi-MB parquet, into a throwaway cache dir.
    """
    monkeypatch.setenv(sp.CACHE_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(sp.shutil, "which", lambda name: None)
    assert sp.active_publish_id(SPEC) == TAG  # downloads only manifest.json (<5 KB)
