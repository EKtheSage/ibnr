"""``ibnr.__version__`` must track the installed distribution metadata.

Regression guard: 0.2.0 shipped to PyPI with a hardcoded ``__version__ = "0.1.0"``
in ``ibnr/__init__.py`` that had drifted from ``pyproject.toml``. Callers stamp
this value as the engine provenance of a run, so a stale literal silently
mislabels results.
"""

from importlib.metadata import version

import ibnr


def test_version_matches_distribution_metadata():
    assert ibnr.__version__ == version("ibnr")


def test_version_is_not_a_placeholder():
    # the source-tree fallback must never reach an installed environment
    assert ibnr.__version__ != "0.0.0+unknown"
    assert ibnr.__version__[0].isdigit()
