"""ibnr: gallery-centric probabilistic loss reserving on a long-format triangle layer.

``Triangle`` is re-exported here; the model gallery is **not**. ``import ibnr``
followed by ``ibnr.gallery`` raises ``AttributeError``, because a submodule
becomes an attribute of its package only once something imports it, and
importing it eagerly here would pull every entry (and its optional dependency
probing) into a bare ``import ibnr``. Use ``from ibnr import gallery``.

``Triangle`` and ``TriangleMeta`` are imported the first time they are read, not
by ``import ibnr``. The Triangle layer imports ibis and pandas, and
``from ibnr import methods`` runs this file first, so importing them here made
every service that only calls ``ibnr.methods`` pay for both.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # read by type checkers and by the docs build, which does not run code
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

#: The names imported on first use, and the module each comes from.
_LAZY = {"Triangle": "ibnr.triangle", "TriangleMeta": "ibnr.triangle"}


def __getattr__(name: str):
    # Called only for a name the module does not already have.
    import importlib

    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value  # later reads find it without coming back here
        return value
    if name == "triangle":
        # ``import ibnr`` used to load the Triangle layer, which made
        # ``ibnr.triangle`` an attribute; it still is. No other submodule is:
        # ``ibnr.gallery`` stays an AttributeError until something imports it.
        return importlib.import_module("ibnr.triangle")
    raise AttributeError(f"module 'ibnr' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
