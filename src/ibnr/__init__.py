"""ibnr: gallery-centric probabilistic loss reserving on a long-format triangle layer.

``Triangle`` is re-exported here; the model gallery is **not**. ``import ibnr``
followed by ``ibnr.gallery`` raises ``AttributeError``, because a submodule
becomes an attribute of its package only once something imports it, and
importing it eagerly here would pull every entry (and its optional dependency
probing) into a bare ``import ibnr``. Use ``from ibnr import gallery``.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

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
