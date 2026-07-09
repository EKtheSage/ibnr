"""The model gallery. Public API: list(), get(), fit()."""

from __future__ import annotations

import builtins

from ibnr.gallery import registry as _registry

# importing the subpackages self-registers their entries
from ibnr.gallery import statistical as _statistical  # noqa: F401
from ibnr.gallery.bayesian import meyers_ccl as _meyers_ccl  # noqa: F401
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.nn import transformer as _nn_transformer  # noqa: F401
from ibnr.gallery.nn import transformer_ml as _nn_transformer_ml  # noqa: F401

get = _registry.get
fit = _registry.fit


def list() -> builtins.list[str]:  # noqa: A001 - mirrors the designed public API
    """Names of all registered gallery entries."""
    return sorted(_registry.entries())


__all__ = ["GalleryEntry", "fit", "get", "list"]
