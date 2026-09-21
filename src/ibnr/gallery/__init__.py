"""The model gallery.

Public API: ``list()``, ``get()``, ``fit()``, ``GalleryEntry``, plus the
held-out evaluation pipeline - ``next_diagonal``, ``CohortForecast``,
``Absence``, ``align_panel``, ``leaderboard``, ``SCORE_DIRECTION`` -
``GalleryDiagonal``, and ``stack()``.

**The rule this export set follows**, so a later addition has to argue against
it: a name is exported if a caller has to CONSTRUCT or CALL it to get from a
fitted entry to a **published result** - a leaderboard row or a one-year CDR.
That is ``next_diagonal`` (which cells), ``CohortForecast`` (one model's arrays
at them), ``Absence`` (why an array is missing - and ``CohortForecast`` refuses
an absent array without one, so it is not optional), ``align_panel`` (the
cross-model intersection) and ``leaderboard``. ``SCORE_DIRECTION`` joins them
because ``leaderboard()`` has no default sort and no ``sort_by=`` by design,
which moves the direction question onto the caller.

``GalleryDiagonal`` (0.5.1) is the one name the rule admitted on its second
clause. It is the same shape of thing as ``CohortForecast`` - the caller builds
it out of a fitted entry and the held-out cells - but it feeds
``kernels.simulate_one_year_cdr`` rather than a board. It cannot be reached by
name from ``kernels.cdr.cdr_methods()`` the way ``"mack"`` can, precisely
because it carries a fitted entry, so if it were not exported here there would
be no short way to write it down at all.

``reserve_rows`` (0.7.0) joins on the first clause: it is the call a caller
makes to get from fitted entries to a published point board - the company
reserve table that ``kernels.point_metrics`` scores. ``point_metrics`` and
``level_errors`` stay on ``ibnr.kernels``: they consume that table and never
touch an entry.

Deliberately NOT here: ``ForecastPanel`` and ``HoldoutCells`` (returned, never
constructed), ``logmeanexp`` / ``resolve_field`` / ``CAPABILITIES`` /
``ABSENCE_REASONS`` / ``index_into`` / ``CellIndex`` (implementation seams the
pipeline never asks a caller to touch), and ``ScoresHeldout`` /
``PredictsHeldout`` - a caller tests capability with ``isinstance``, and those
two names are how an ENTRY declares one, so they stay at ``ibnr.gallery.entry``.
They are all one import away at ``ibnr.kernels``.

Re-export direction is gallery -> kernels only: ``kernels`` never imports the
gallery, which is what lets the harness's spawn-based workers import it alone.
"""

from __future__ import annotations

import builtins

# importing the subpackages self-registers their entries
from ibnr.gallery import deterministic as _deterministic  # noqa: F401
from ibnr.gallery import registry as _registry
from ibnr.gallery import statistical as _statistical  # noqa: F401
from ibnr.gallery.bayesian import clark_growth_curve as _clark_growth_curve  # noqa: F401
from ibnr.gallery.bayesian import compartmental as _compartmental  # noqa: F401
from ibnr.gallery.bayesian import england_verrall_odp as _england_verrall_odp  # noqa: F401
from ibnr.gallery.bayesian import guszcza_growth_curve as _guszcza_growth_curve  # noqa: F401
from ibnr.gallery.bayesian import meyers_ccl as _meyers_ccl  # noqa: F401
from ibnr.gallery.bayesian import meyers_csr as _meyers_csr  # noqa: F401
from ibnr.gallery.cdr import GalleryDiagonal
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.nn import deeptriangle as _nn_deeptriangle  # noqa: F401
from ibnr.gallery.nn import mdn as _nn_mdn  # noqa: F401
from ibnr.gallery.nn import nn_paid_case as _nn_paid_case  # noqa: F401
from ibnr.gallery.nn import resnet as _nn_resnet  # noqa: F401
from ibnr.gallery.nn import tlrn as _nn_tlrn  # noqa: F401
from ibnr.gallery.nn import transformer as _nn_transformer  # noqa: F401
from ibnr.gallery.nn import transformer_ml as _nn_transformer_ml  # noqa: F401

# the evaluation layer is implemented once in kernels/ (design decision 5);
# these re-exports are the gallery-facing names decision 8 promises
from ibnr.kernels.forecast import (
    SCORE_DIRECTION,
    Absence,
    CohortForecast,
    align_panel,
    leaderboard,
)
from ibnr.kernels.holdout import next_diagonal
from ibnr.kernels.point_scores import reserve_rows
from ibnr.kernels.stacking import stack

get = _registry.get
fit = _registry.fit


def list() -> builtins.list[str]:  # noqa: A001 - mirrors the designed public API
    """Names of all registered gallery entries."""
    return sorted(_registry.entries())


__all__ = [
    "SCORE_DIRECTION",
    "Absence",
    "CohortForecast",
    "GalleryDiagonal",
    "GalleryEntry",
    "align_panel",
    "fit",
    "get",
    "leaderboard",
    "list",
    "next_diagonal",
    "reserve_rows",
    "stack",
]
