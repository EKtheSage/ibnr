"""ibnr: gallery-centric probabilistic loss reserving on a long-format triangle layer."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

from ibnr.triangle import Triangle, TriangleMeta

# Read from the installed distribution so pyproject.toml stays the single source
# of truth. A hardcoded literal here silently drifted once already: 0.2.0 shipped
# to PyPI reporting __version__ == "0.1.0", which matters because callers stamp
# this value as engine provenance for a run.
try:
    __version__ = _dist_version("ibnr")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["Triangle", "TriangleMeta", "__version__"]
