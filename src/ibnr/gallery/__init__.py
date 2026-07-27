"""The model gallery. Public API: list(), get(), fit(), stack(), leaderboard()."""

from __future__ import annotations

import builtins

# importing the subpackages self-registers their entries
from ibnr.gallery import deterministic as _deterministic  # noqa: F401
from ibnr.gallery import registry as _registry
from ibnr.gallery import statistical as _statistical  # noqa: F401
from ibnr.gallery.bayesian import clark_growth_curve as _clark_growth_curve  # noqa: F401
from ibnr.gallery.bayesian import compartmental as _compartmental  # noqa: F401
from ibnr.gallery.bayesian import england_verrall_odp as _england_verrall_odp  # noqa: F401
from ibnr.gallery.bayesian import meyers_ccl as _meyers_ccl  # noqa: F401
from ibnr.gallery.bayesian import meyers_csr as _meyers_csr  # noqa: F401
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.nn import deeptriangle as _nn_deeptriangle  # noqa: F401
from ibnr.gallery.nn import mdn as _nn_mdn  # noqa: F401
from ibnr.gallery.nn import transformer as _nn_transformer  # noqa: F401
from ibnr.gallery.nn import transformer_ml as _nn_transformer_ml  # noqa: F401

# the evaluation layer is implemented once in kernels/ (design decision 5);
# these re-exports are the gallery-facing names decision 8 promises
from ibnr.kernels.forecast import leaderboard
from ibnr.kernels.stacking import stack

get = _registry.get
fit = _registry.fit


def list() -> builtins.list[str]:  # noqa: A001 - mirrors the designed public API
    """Names of all registered gallery entries."""
    return sorted(_registry.entries())


__all__ = ["GalleryEntry", "fit", "get", "leaderboard", "list", "stack"]
