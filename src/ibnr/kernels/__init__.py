"""Evaluation, contract and wire-format kernels.

Every name in ``__all__`` is imported the first time it is read, through the
module ``__getattr__`` below (PEP 562), not when this package is imported.
``ibnr.methods`` imports ``kernels.grid``, ``kernels.conventional`` and
``kernels.mack``, and Python runs this file before any of them. When this file
imported every name eagerly, that one import loaded ibis, pandas and scipy
(``scipy.special`` through ``kernels.densities`` and ``kernels.cdr``), none of
which the four methods use. Measured with 11 fresh interpreters per arm,
interleaved, on 2026-09-25: ``from ibnr import methods`` took a median of 4.8 s
with 0.7.2's code and 1.3 s after this change, on a busy dev box; the CHANGELOG
has the full figures.

``from ibnr.kernels import to_arrow`` and ``kernels.to_arrow`` work as before;
the first read of a name imports its module and keeps the value here, so later
reads cost nothing. A submodule is also imported when it is read as an
attribute (``kernels.codec``), as it was when every submodule was loaded
eagerly.

The ``TYPE_CHECKING`` block repeats every import. Type checkers read it, and so
does the docs build, which runs with ``dynamic: false`` and finds names by
reading source, not by running it.

An earlier version of this docstring recorded that deferring only the codec
bought nothing for ``from ibnr import gallery``, because pandas imports pyarrow
itself. That is still true of the gallery, which needs pandas; it was never
true of ``ibnr.methods``, which needs pyarrow and not pandas.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ibnr.kernels.cdr import (
        CDRResult,
        MackDiagonal,
        ODPBootstrapDiagonal,
        cdr_methods,
        cdr_risk_measures,
        get_cdr_method,
        one_year_cdr,
        rereserve,
        simulate_one_year_cdr,
    )
    from ibnr.kernels.codec import (
        CODEC_VERSION,
        CONTENT_TYPE_ARROW,
        CONTENT_TYPE_JSON,
        DEFAULT_QUANTILES,
        from_arrow,
        peek_kind,
        to_arrow,
        to_summary,
    )
    from ibnr.kernels.contract import cohort_grid_frame
    from ibnr.kernels.conventional import (
        ConventionalCandidate,
        ConventionalFit,
        conventional_grid,
        fit_conventional,
        fit_conventional_grid,
    )
    from ibnr.kernels.densities import to_amount_scale
    from ibnr.kernels.forecast import (
        SCORE_DIRECTION,
        Absence,
        CohortForecast,
        ForecastPanel,
        align_panel,
        leaderboard,
    )
    from ibnr.kernels.glm import TweedieFit, TweedieSpec, fit_tweedie_grid
    from ibnr.kernels.holdout import HoldoutCells, next_diagonal
    from ibnr.kernels.mack import MackFit, fit_mack, fit_mack_grid, simulate_ultimates
    from ibnr.kernels.parity import ParityReport, compare_posteriors
    from ibnr.kernels.point_scores import (
        level_errors,
        point_metrics,
        point_summary,
        reserve_rows,
        shrink_toward,
    )
    from ibnr.kernels.predictive import PredictiveDistribution
    from ibnr.kernels.replay import ConventionalReplay, replay_conventional
    from ibnr.kernels.residual_calibration import (
        Calibration,
        calibrate,
        calibrated_draws,
        leave_one_out_coverage,
        rolling_residuals,
    )
    from ibnr.kernels.selection import (
        ConventionalEvaluation,
        ConventionalSelection,
        evaluate_conventional,
        score_replay,
        select_conventional,
    )

#: Each exported name and the module it is imported from on first read. It must
#: hold exactly the names of ``__all__``; ``tests/test_import_purity.py`` checks
#: that, and that the block above imports the same names from the same modules.
_LAZY = {
    "CDRResult": "cdr",
    "MackDiagonal": "cdr",
    "ODPBootstrapDiagonal": "cdr",
    "cdr_methods": "cdr",
    "cdr_risk_measures": "cdr",
    "get_cdr_method": "cdr",
    "one_year_cdr": "cdr",
    "rereserve": "cdr",
    "simulate_one_year_cdr": "cdr",
    "CODEC_VERSION": "codec",
    "CONTENT_TYPE_ARROW": "codec",
    "CONTENT_TYPE_JSON": "codec",
    "DEFAULT_QUANTILES": "codec",
    "from_arrow": "codec",
    "peek_kind": "codec",
    "to_arrow": "codec",
    "to_summary": "codec",
    "cohort_grid_frame": "contract",
    "ConventionalCandidate": "conventional",
    "ConventionalFit": "conventional",
    "conventional_grid": "conventional",
    "fit_conventional": "conventional",
    "fit_conventional_grid": "conventional",
    "to_amount_scale": "densities",
    "Absence": "forecast",
    "CohortForecast": "forecast",
    "ForecastPanel": "forecast",
    "SCORE_DIRECTION": "forecast",
    "align_panel": "forecast",
    "leaderboard": "forecast",
    "TweedieFit": "glm",
    "TweedieSpec": "glm",
    "fit_tweedie_grid": "glm",
    "HoldoutCells": "holdout",
    "next_diagonal": "holdout",
    "MackFit": "mack",
    "fit_mack": "mack",
    "fit_mack_grid": "mack",
    "simulate_ultimates": "mack",
    "ParityReport": "parity",
    "compare_posteriors": "parity",
    "level_errors": "point_scores",
    "point_metrics": "point_scores",
    "point_summary": "point_scores",
    "reserve_rows": "point_scores",
    "shrink_toward": "point_scores",
    "PredictiveDistribution": "predictive",
    "ConventionalReplay": "replay",
    "replay_conventional": "replay",
    "Calibration": "residual_calibration",
    "calibrate": "residual_calibration",
    "calibrated_draws": "residual_calibration",
    "leave_one_out_coverage": "residual_calibration",
    "rolling_residuals": "residual_calibration",
    "ConventionalEvaluation": "selection",
    "ConventionalSelection": "selection",
    "evaluate_conventional": "selection",
    "score_replay": "selection",
    "select_conventional": "selection",
}


def __getattr__(name: str):
    # Called only for a name this module does not have yet.
    module = _LAZY.get(name)
    if module is not None:
        value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
        globals()[name] = value  # later reads find it without coming back here
        return value
    if not name.startswith("_") and importlib.util.find_spec(f"{__name__}.{name}") is not None:
        # a submodule read as an attribute; importing it also sets the attribute
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "CDRResult",
    "CODEC_VERSION",
    "CONTENT_TYPE_ARROW",
    "CONTENT_TYPE_JSON",
    "DEFAULT_QUANTILES",
    "SCORE_DIRECTION",
    "Absence",
    "Calibration",
    "CohortForecast",
    "ConventionalCandidate",
    "ConventionalEvaluation",
    "ConventionalFit",
    "ConventionalReplay",
    "ConventionalSelection",
    "ForecastPanel",
    "HoldoutCells",
    "MackDiagonal",
    "MackFit",
    "ODPBootstrapDiagonal",
    "ParityReport",
    "PredictiveDistribution",
    "TweedieFit",
    "TweedieSpec",
    "align_panel",
    "calibrate",
    "calibrated_draws",
    "cdr_methods",
    "cdr_risk_measures",
    "cohort_grid_frame",
    "compare_posteriors",
    "conventional_grid",
    "evaluate_conventional",
    "fit_conventional",
    "fit_conventional_grid",
    "fit_mack",
    "fit_mack_grid",
    "fit_tweedie_grid",
    "from_arrow",
    "get_cdr_method",
    "leaderboard",
    "leave_one_out_coverage",
    "level_errors",
    "next_diagonal",
    "one_year_cdr",
    "peek_kind",
    "point_metrics",
    "point_summary",
    "replay_conventional",
    "rereserve",
    "reserve_rows",
    "rolling_residuals",
    "score_replay",
    "select_conventional",
    "shrink_toward",
    "simulate_one_year_cdr",
    "simulate_ultimates",
    "to_amount_scale",
    "to_arrow",
    "to_summary",
]
