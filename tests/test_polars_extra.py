"""duckdb is the default backend and needs no extra; polars is opt-in.

The polars backend brings a ~176 MB runtime (about 35% of the install) that a
duckdb-only user never executes, so it moved to the ``ibnr[polars]`` extra in
0.4.0. Both backends remain first-class and tested (CLAUDE.md #2); the dev group
installs polars, which is why the parametrized suite still covers it.

These tests pin the two things that split can break: duckdb must keep working
with nothing extra installed, and asking for polars without the extra must say
how to fix it rather than raise something opaque.
"""

import importlib.util

import pytest

import ibnr
from ibnr.triangle.io import _polars_installed, resolve_backend


def test_duckdb_is_the_default_and_needs_no_extra():
    assert resolve_backend() is not None
    assert resolve_backend("duckdb") is not None


def test_polars_installed_matches_reality():
    assert _polars_installed() == (importlib.util.find_spec("polars") is not None)


def test_unknown_backend_still_raises_value_error():
    with pytest.raises(ValueError, match="duckdb"):
        resolve_backend("sqlite")


@pytest.mark.skipif(
    importlib.util.find_spec("polars") is None,
    reason="polars not installed; `uv sync` installs it via the dev group",
)
def test_polars_resolves_when_the_extra_is_present():
    assert resolve_backend("polars") is not None


class _FailingBackend:
    """Stand-in for ``ibis.polars`` whose ``connect`` raises.

    ``ibis.polars`` is a lazily-resolved module attribute, so the substitution
    happens on the ``ibis`` module object that ``io`` actually holds.
    """

    def __init__(self, exc):
        self._exc = exc

    def connect(self, *args, **kwargs):
        raise self._exc


def test_missing_polars_names_the_extra(monkeypatch):
    """Simulate a core-only install: the error must name the fix."""
    from ibnr.triangle import io

    monkeypatch.setattr(
        io.ibis, "polars", _FailingBackend(ModuleNotFoundError("No module named 'polars'"))
    )
    monkeypatch.setattr(io, "_polars_installed", lambda: False)

    with pytest.raises(ModuleNotFoundError) as exc:
        resolve_backend("polars")

    msg = str(exc.value)
    assert "ibnr[polars]" in msg  # how to fix it
    assert "duckdb" in msg  # and that the default needs nothing
    assert exc.value.__cause__ is not None  # original error preserved


def test_a_real_polars_error_is_not_masked_as_missing(monkeypatch):
    """If polars IS installed, a genuine failure must surface unchanged."""
    from ibnr.triangle import io

    sentinel = RuntimeError("genuine backend failure")
    monkeypatch.setattr(io.ibis, "polars", _FailingBackend(sentinel))
    monkeypatch.setattr(io, "_polars_installed", lambda: True)

    with pytest.raises(RuntimeError) as exc:
        resolve_backend("polars")
    assert exc.value is sentinel


def test_triangle_still_importable_from_top_level():
    assert ibnr.Triangle is not None
