"""Held-out next-diagonal leaderboard over the Schedule P gold mart (milestone 6).

The capstone study of the milestone-6 object layer: per (company, line) cohort
and per training cutoff, every board entry is fitted to the upper triangle and
scored on the diagonal that first becomes observable AFTER the cutoff
(``kernels.holdout.next_diagonal``). The per-cohort arrays become
``CohortForecast`` objects, ``align_panel`` puts every model on one identical
cell set per score, and ``leaderboard`` reduces the panels to the board.

Protocol (the task string ``paid_next_diagonal_v1`` names it):

* field: ``paid_loss``, the one field every entry can score (compartmental's
  outstanding block is derived, not a raw field a panel can intersect on).
* cutoffs: 1996-12-31 (the weights panel) and 1997-12-31 (the evaluation
  panel). Two cutoffs is two panels - ``align_panel`` refuses a mix - and the
  earlier one exists so ``kernels.stacking.stack`` can fit its weights on
  cells that never graded them.
* origins: accident years 1988-1997 only. The mart runs to 2007 and an
  unrestricted aggregate silently includes post-study accident years
  (the 2.4x gotcha, CLAUDE.md); ``next_diagonal(origins=...)`` is the guard.
* cohort: the milestone-3 screened multiline panel
  (``compare_gallery.screened_company_lines``) - companies passing the Table
  A.1 screens on >= 2 lines, every model scored on exactly those pairs.

THE UNIFORM PAID CLAMP. Meyers' ``pmax(paid_loss, 1)`` floor is applied ONCE
in the worker, to the cohort triangle, before BOTH the fit and
``next_diagonal`` - for EVERY entry on this board, not just the lognormal
ones. ``align_panel`` refuses disagreeing observed values at the same key,
and a per-entry clamp is exactly the documented way to trip it: a clamped run
and an unclamped one produce identical keys with different values at the
cells that matter. The board-wide floor is the price of one shared panel; it
is recorded in the ``clamp`` column of every output CSV. (This deliberately
diverges from ``meyers_validation.py``, where ODP and compartmental see the
unclamped series - there each model is validated alone, here they share a
panel.)

Membership, per entry (the reasons are printed on the board, so a wrong one
is a false statement - see ``forecast.MODEL_ABSENCE_REASONS``):

* density + draws: meyers_ccl, meyers_csr, compartmental_gaussian,
  compartmental_lognormal (variants are distinct board names; ``variant=``
  rides in ``RetroTask.fit_kwargs``), nn_transformer (pooled fit, below).
* draws only (``no_predictive_density`` on principle - ODP quasi-likelihood /
  bootstrap): england_verrall_odp, clark_growth_curve, clark, mack.
* neither yet (``scorer_not_implemented``, both axes): sur, copula_glm,
  nn_transformer_ml. They still occupy board rows.

nn_transformer is a POOLED fit: once per cutoff, in the parent process, on
exactly the scored (company, line) pairs (the cohort-data-parity rule - the
pool is the comparison), then per-cohort forecasts through the same
``log_lik_at`` / ``predict_at`` -> absence mapping every worker uses. Its
pinned-dev density refusal lands as ``scoring_refused``.

Outputs (``analysis/results/``, every row stamped with ``mart_publish_id``
and the clamp policy):

* ``heldout_fits.csv``        raw per-fit rows, written INCREMENTALLY as pool
                              results land, so a long run survives a crash
                              before panel assembly. One row per attempt
                              (``stage`` distinguishes escalations).
* ``heldout_leaderboard.csv`` the boards: both cutoffs plus the 1997 board
                              re-aligned WITH the stacked pseudo-model
                              (``board`` column distinguishes them).
* ``heldout_pointwise.csv``   the panels' per-(model, cell) long frame plus a
                              cell-level PIT column (mean(draws <= value)).
* ``heldout_coverage.csv``, ``heldout_absences.csv``, ``heldout_dropped.csv``,
  ``heldout_excluded.csv``    the panel censuses, per board.
* ``heldout_stacking.json``   the fitted weights and their provenance.

Usage:
    uv run python scripts/heldout_leaderboard.py --per-line 3
    uv run python scripts/heldout_leaderboard.py --models meyers_csr mack --serial
    uv run python scripts/heldout_leaderboard.py --warehouse github://owner/repo@publish_id
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import ibis
import numpy as np
import pandas as pd

from ibnr import gallery
from ibnr.data.schedule_p import (
    active_mart_path,
    active_publish_id,
    load_schedule_p,
    pinned_source,
)
from ibnr.kernels.forecast import Absence, CohortForecast, align_panel, leaderboard
from ibnr.kernels.harness import (
    _DIAG_KEYS,
    RetroTask,
    SamplerSettings,
    precompile,
    run_retro,
)
from ibnr.kernels.holdout import HoldoutCells, next_diagonal
from ibnr.kernels.stacking import StackingResult, stack
from ibnr.triangle.core import Triangle

sys.path.insert(0, str(Path(__file__).parent))
from compare_gallery import pair_filter, screened_company_lines  # noqa: E402
from meyers_validation import DEFAULT_WAREHOUSE, MEYERS_LINES, TRAIN_AYS, stages_for  # noqa: E402

#: the protocol string stamped on every forecast; a different protocol (other
#: field, other holdout rule) is a different task and a different board
TASK = "paid_next_diagonal_v1"
FIELD = "paid_loss"
PREMIUM_FIELD = "earned_premium"

#: weights cutoff first, evaluation cutoff second (stack() enforces the order)
CUTOFFS = ("1996-12-31", "1997-12-31")

#: the Meyers study window as origin dates - next_diagonal(origins=...) is the
#: guard against the mart's post-study accident years (CLAUDE.md, the 2.4x bug)
STUDY_ORIGINS = tuple(dt.date(y, 1, 1) for y in range(TRAIN_AYS[0], TRAIN_AYS[1] + 1))

#: the ONE data alteration, applied uniformly (module docstring) and recorded
#: in a column of every output CSV
CLAMP = "pmax(paid_loss, 1)"

UNITS = "USD thousands"


@dataclass(frozen=True)
class BoardModel:
    """One per-cohort board entry: registry name, fit arguments, capabilities.

    ``density_reason`` None means the entry scores held-out densities
    (``ScoresHeldout``); a reason string is the MODEL-LEVEL absence stamped on
    the density axis of every one of its forecasts - it must stay uniform
    across cohorts (``align_panel`` refuses a model-level reason mixed with
    cohort-level ones), which is why a fit failure never overwrites it.
    """

    entry: str
    fit_kwargs: dict = field(default_factory=dict)
    density_reason: str | None = None
    density_detail: str = ""


_ODP_DETAIL = (
    "ODP quasi-likelihood: Poisson only up to proportionality, not a density on "
    "any scale (densities.py odp-not-a-density)"
)

#: entries fitted per cohort in the worker pool, keyed by BOARD name
WORKER_MODELS: dict[str, BoardModel] = {
    "meyers_ccl": BoardModel("meyers_ccl"),
    "meyers_csr": BoardModel("meyers_csr"),
    "compartmental_gaussian": BoardModel("compartmental", fit_kwargs={"variant": "gaussian"}),
    "compartmental_lognormal": BoardModel("compartmental", fit_kwargs={"variant": "lognormal"}),
    "england_verrall_odp": BoardModel(
        "england_verrall_odp", density_reason="no_predictive_density", density_detail=_ODP_DETAIL
    ),
    "clark_growth_curve": BoardModel(
        "clark_growth_curve", density_reason="no_predictive_density", density_detail=_ODP_DETAIL
    ),
    "clark": BoardModel(
        "clark", density_reason="no_predictive_density", density_detail=_ODP_DETAIL
    ),
    "mack": BoardModel(
        "mack",
        density_reason="no_predictive_density",
        density_detail="distribution-free bootstrap; no stated observation model",
    ),
}

#: fitted ONCE per cutoff in the parent (pooled across cohorts), scored per cohort
POOLED_MODEL = "nn_transformer"
POOLED_SPEC = BoardModel("nn_transformer")

#: entries with no held-out scorer yet: board rows with scorer_not_implemented
#: on BOTH axes (NOT no_predictive_density - sur and copula_glm state
#: observation models; the reason would be a false claim about the model)
UNAVAILABLE_MODELS: dict[str, str] = {
    "sur": "multivariate-normal density exists in principle; no held-out scorer yet",
    "copula_glm": "lognormal marginals exist in principle; no held-out scorer yet",
    "nn_transformer_ml": "multi-line heads have no per-cell held-out scorer yet",
}

ALL_MODELS = [*WORKER_MODELS, POOLED_MODEL, *UNAVAILABLE_MODELS]


# -- the clamp and the cells ---------------------------------------------------


def clamp_paid_floor(tri: Triangle) -> Triangle:
    """Meyers' pmax(paid, 1) floor, the harness's own ibis mutate pattern.

    Applied to the WHOLE triangle before both the fit and ``next_diagonal``,
    so the training slice and the held-out outcomes are on one basis - the
    align_panel value-agreement check is what makes a partial clamp an error
    rather than a quiet bias.
    """
    e = tri.expr
    return tri.with_expr(
        e.mutate(value=ibis.ifelse(e.field == FIELD, ibis.greatest(e.value, 1), e.value))
    )


def study_cells(tri: Triangle, as_of: dt.date | str) -> HoldoutCells:
    """The held-out cells of one cohort at one cutoff, study window enforced."""
    return next_diagonal(
        tri,
        as_of=as_of,
        fields=FIELD,
        premium_field=PREMIUM_FIELD,
        origins=STUDY_ORIGINS,
    )


# -- absence mapping -----------------------------------------------------------


def _refusal(e: Exception) -> Absence:
    return Absence("scoring_refused", f"{type(e).__name__}: {e}")


def _axis_refusal(
    model: str,
    cells: HoldoutCells,
    *,
    log_density: np.ndarray | None = None,
    draws: np.ndarray | None = None,
) -> Absence | None:
    """Probe-construct one axis so a constructor refusal demotes ONLY that axis.

    ``CohortForecast`` validates both arrays in one ``__post_init__``; building
    a throwaway forecast with a single axis present tells us which array a
    raise (zero variance across draws, NaN, width mismatch) belongs to, so an
    all-degenerate draws array becomes a ``scoring_refused`` on the draws axis
    instead of an error row that hides the model's perfectly good density.
    """
    try:
        CohortForecast(
            model=model,
            task=TASK,
            cells=cells,
            field=FIELD,
            log_density=log_density,
            draws=draws,
            density_absence=None if log_density is not None else Absence("not_offered"),
            draws_absence=None if draws is not None else Absence("not_offered"),
        )
    except Exception as e:  # noqa: BLE001 - every refusal maps to an Absence
        return _refusal(e)
    return None


def forecast_from_fit(
    model: str,
    spec: BoardModel,
    entry,
    cells: HoldoutCells,
    *,
    seed: int | None,
) -> CohortForecast:
    """One fitted entry -> one CohortForecast, each axis in its own try/except.

    The mapping (the board prints these, so each reason must be the true one):

    * ``spec.density_reason`` set -> the model-level absence, uniformly - the
      entry has no density on ANY cohort, and align_panel enforces uniformity;
    * ``log_lik_at`` / ``predict_at`` raising -> ``scoring_refused`` with the
      exception text on that axis only (compartmental-lognormal's non-positive
      increments, nn_transformer's pinned-dev refusal, the zero-variance
      constructor raise all land here);
    * both arrays surviving -> a forecast offering both capabilities.
    """
    if spec.density_reason is not None:
        log_density = None
        density_absence: Absence | None = Absence(spec.density_reason, spec.density_detail)
    else:
        try:
            log_density = np.asarray(entry.log_lik_at(cells, field=FIELD), dtype=float)
            density_absence = _axis_refusal(model, cells, log_density=log_density)
            if density_absence is not None:
                log_density = None
        except Exception as e:  # noqa: BLE001 - a refusal is a verdict, not a crash
            log_density, density_absence = None, _refusal(e)

    try:
        draws = np.asarray(entry.predict_at(cells, field=FIELD, seed=seed), dtype=float)
        draws_absence = _axis_refusal(model, cells, draws=draws)
        if draws_absence is not None:
            draws = None
    except Exception as e:  # noqa: BLE001
        draws, draws_absence = None, _refusal(e)

    return CohortForecast(
        model=model,
        task=TASK,
        cells=cells,
        field=FIELD,
        log_density=log_density,
        draws=draws,
        density_absence=density_absence,
        draws_absence=draws_absence,
    )


def absence_forecast(
    model: str, spec: BoardModel, cells: HoldoutCells, absence: Absence
) -> CohortForecast:
    """A no-array forecast for a failed fit, capabilities respected.

    The draws axis carries ``absence`` (``fit_failed``). The density axis does
    too - UNLESS the spec declares a model-level reason, which must stay
    uniform across every cohort: stamping ``fit_failed`` on ODP's density axis
    for one cohort would make align_panel refuse the whole model (a
    model-level claim contradicted by its own census).
    """
    density_absence = (
        Absence(spec.density_reason, spec.density_detail)
        if spec.density_reason is not None
        else absence
    )
    return CohortForecast(
        model=model,
        task=TASK,
        cells=cells,
        field=FIELD,
        density_absence=density_absence,
        draws_absence=absence,
    )


# -- the worker ----------------------------------------------------------------


def _load_cohort(task: RetroTask) -> Triangle:
    return load_schedule_p(task.warehouse, lines=[task.line], companies=[task.company_code])


def _resolve_entry(name: str):
    return gallery.get(name)


def _stamp_forecast_columns(row: dict) -> None:
    """Flat CSV columns describing the row's forecast (or its absence)."""
    f = row.get("forecast")
    if f is None:
        return
    row["n_cells"] = f.n_cells
    row["has_density"] = f.has_density
    row["has_draws"] = f.has_draws
    row["density_absence"] = str(f.density_absence) if f.density_absence is not None else ""
    row["draws_absence"] = str(f.draws_absence) if f.draws_absence is not None else ""
    row["n_draws_density"] = f.n_draws_for("density")
    row["n_draws_sample"] = f.n_draws_for("draws")


def run_cohort(task: RetroTask, settings: SamplerSettings, *, load=None, resolve=None) -> dict:
    """Fit + score one (board model, cohort, cutoff): the injected pool runner.

    Returns a picklable dict: identity, the harness diagnostics (so the
    convergence gates and staged escalation work unchanged), ``seconds``,
    ``error``, and ``forecast`` - a :class:`CohortForecast` or None. Never
    raises; a fit failure becomes an error row whose forecast (when the cells
    were built) records ``fit_failed``, so the absence census still counts it.

    ``load`` / ``resolve`` are injectable for mart-free tests; the defaults
    (None) resolve inside so the function pickles by reference into spawned
    workers.
    """
    row: dict = {
        "model": task.model,
        "line": task.line,
        "company_code": task.company_code,
        "as_of": task.as_of,
        "error": None,
    }
    t0 = time.perf_counter()
    cells: HoldoutCells | None = None
    spec = WORKER_MODELS[task.model]
    try:
        tri = (load or _load_cohort)(task)
        # the uniform board clamp - BEFORE both the fit and the cells, for
        # every entry (module docstring; task.clamp_paid is stamped for the
        # record but this board never runs unclamped)
        tri = clamp_paid_floor(tri)
        cells = study_cells(tri, task.as_of)
        cls = (resolve or _resolve_entry)(spec.entry)
        kwargs = settings.fit_kwargs(cls.fit)
        if "seed" in inspect.signature(cls.fit).parameters:
            kwargs["seed"] = task.seed
        if task.loss_field is not None:
            kwargs["loss_field"] = task.loss_field
        kwargs.update(task.fit_kwargs)
        entry = cls().fit(tri, as_of=task.as_of, **kwargs)
        if hasattr(entry, "convergence"):
            diag = entry.convergence()
            row.update({k: diag.get(k) for k in _DIAG_KEYS})
        row["forecast"] = forecast_from_fit(task.model, spec, entry, cells, seed=task.seed)
    except Exception as e:  # noqa: BLE001 - the study must run to completion
        row["error"] = f"{type(e).__name__}: {e}"
        if cells is not None:
            try:
                row["forecast"] = absence_forecast(
                    task.model, spec, cells, Absence("fit_failed", row["error"])
                )
            except Exception:  # noqa: BLE001 - e.g. a zero-cell cohort
                row["forecast"] = None
    row["seconds"] = time.perf_counter() - t0
    _stamp_forecast_columns(row)
    return row


# -- the pooled NN entry and the not-implemented rows --------------------------


def cells_for_pairs(
    tri_pool: Triangle, pairs: list[tuple[str, str]], as_of: str
) -> dict[tuple[str, str], HoldoutCells | str]:
    """Held-out cells per (company, line) pair; a failure is its error string.

    Built from the SAME clamped pool triangle the NN fits on, so the observed
    values agree cell-for-cell with the worker cohorts' (same parquet, same
    mutate) - which align_panel then verifies rather than assumes.
    """
    out: dict[tuple[str, str], HoldoutCells | str] = {}
    for code, line in pairs:
        tri = tri_pool.filter((ibis._.company_code == code) & (ibis._.line_of_business == line))
        try:
            out[(code, line)] = study_cells(tri, as_of)
        except Exception as e:  # noqa: BLE001
            out[(code, line)] = f"{type(e).__name__}: {e}"
    return out


def fit_pooled_nn(tri_pool: Triangle, *, as_of: str, seed: int | None, nn_features):
    """One pooled transformer fit on the clamped scored pairs (torch)."""
    return _resolve_entry(POOLED_SPEC.entry)().fit(
        tri_pool,
        loss_field=FIELD,
        feature_fields=tuple(nn_features),
        premium_field=PREMIUM_FIELD,
        as_of=as_of,
        seed=seed,
    )


def score_pooled(
    entry,
    cellmap: dict[tuple[str, str], HoldoutCells | str],
    *,
    as_of: str,
    seed: int | None,
    fit_seconds: float,
    fit_error: str | None = None,
    model: str = POOLED_MODEL,
) -> tuple[list[dict], list[CohortForecast]]:
    """Per-cohort forecasts from one pooled fit, same absence mapping as the pool.

    A pooled-fit failure (``fit_error``) becomes ``fit_failed`` on every
    cohort - the fit IS the cohort's fit for a pooled entry (forecast.py's
    clustering note). ``fit_seconds`` is stamped on every row because the cost
    is shared, not per cohort.
    """
    rows: list[dict] = []
    forecasts: list[CohortForecast] = []
    for (code, line), cells in cellmap.items():
        row: dict = {
            "model": model,
            "line": line,
            "company_code": code,
            "as_of": as_of,
            "error": None,
            "fit_seconds": fit_seconds,
        }
        t0 = time.perf_counter()
        if isinstance(cells, str):
            row["error"] = cells
        elif fit_error is not None:
            row["error"] = fit_error
            row["forecast"] = absence_forecast(
                model, POOLED_SPEC, cells, Absence("fit_failed", fit_error)
            )
        else:
            try:
                row["forecast"] = forecast_from_fit(model, POOLED_SPEC, entry, cells, seed=seed)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
        row["seconds"] = time.perf_counter() - t0
        _stamp_forecast_columns(row)
        if row.get("forecast") is not None:
            forecasts.append(row["forecast"])
        rows.append(row)
    return rows, forecasts


def unavailable_forecasts(
    cellmap: dict[tuple[str, str], HoldoutCells | str],
    models: list[str],
    *,
    as_of: str,
) -> tuple[list[dict], list[CohortForecast]]:
    """Board rows for the entries with no held-out scorer yet.

    ``scorer_not_implemented`` on BOTH axes - the honest reason (these entries
    state observation models; ``no_predictive_density`` would be a false claim
    the board then prints). They ride on the same cells as everyone else, so
    the panel's absence census counts them per cohort.
    """
    rows: list[dict] = []
    forecasts: list[CohortForecast] = []
    for model in models:
        for (code, line), cells in cellmap.items():
            row: dict = {
                "model": model,
                "line": line,
                "company_code": code,
                "as_of": as_of,
                "error": None,
                "seconds": 0.0,
            }
            if isinstance(cells, str):
                row["error"] = cells
            else:
                row["forecast"] = CohortForecast.unavailable(
                    model=model,
                    task=TASK,
                    cells=cells,
                    density_reason="scorer_not_implemented",
                    draws_reason="scorer_not_implemented",
                    detail=UNAVAILABLE_MODELS[model],
                    field=FIELD,
                )
                forecasts.append(row["forecast"])
            _stamp_forecast_columns(row)
            rows.append(row)
    return rows, forecasts


# -- PIT (cell level only) -----------------------------------------------------


def pit_values(draws: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``(n_cells,)`` PIT per cell: mean(draws <= value) over the draw axis."""
    return (draws <= np.asarray(values, dtype=float)[None, :]).mean(axis=0)


def pointwise_with_pit(panel, forecasts: list[CohortForecast]) -> pd.DataFrame:
    """``panel.pointwise`` plus a per-(model, cell) ``pit`` column.

    PIT is emitted at CELL level in this long frame ONLY, never as a board
    column: forecast.py's leaderboard docstring records why - the KS critical
    value ``1.36/sqrt(n)`` assumes independence, and the cells of one cohort
    share one posterior and one calendar year, so a cell-level KS flags
    calibrated models as broken. The cell-level PITs are published for
    analyses that model that dependence; models without draws get ``pd.NA``.
    """
    pit_map: dict[tuple[str, tuple], float] = {}
    for f in forecasts:
        if not f.has_draws:
            continue
        values = f.key_frame["value"].to_numpy(dtype=float)
        for key, p in zip(f.keys, pit_values(f.draws, values), strict=True):
            pit_map[(f.model, key)] = float(p)
    out = panel.pointwise.copy()
    out["pit"] = pd.array(
        [pit_map.get((m, k), pd.NA) for m, k in zip(out["model"], out["key"], strict=True)],
        dtype="Float64",
    )
    return out.drop(columns=["key"])


# -- panel assembly, boards, stacking ------------------------------------------


@dataclass
class Outputs:
    """Everything the study writes, before provenance stamping."""

    boards: pd.DataFrame
    pointwise: pd.DataFrame
    coverage: pd.DataFrame
    absences: pd.DataFrame
    dropped: pd.DataFrame
    excluded: pd.DataFrame
    stacking: StackingResult | None
    stack_error: str | None


def build_outputs(
    forecasts_by_cutoff: dict[str, list[CohortForecast]],
    *,
    units: str | None = UNITS,
    stack_method: str = "mle",
    stack_seed: int | None = None,
) -> Outputs:
    """Align each cutoff's panel, reduce the boards, stack across the cutoffs.

    Cutoff keys are ISO date strings, so their sorted order is chronological:
    the earliest is the weights panel, the latest the evaluation panel. The
    stacked pseudo-model is scored by re-aligning the evaluation forecasts
    WITH the stacked ones (stacking.py: no private scoring path), which is the
    third board. Stacking failures are recorded, not fatal - the two base
    boards are results in their own right.
    """
    cutoffs = sorted(forecasts_by_cutoff)
    panels = {}
    boards, pointwise, censuses = (
        [],
        [],
        {n: [] for n in ("coverage", "absences", "dropped", "excluded")},
    )

    def add(label: str, panel, forecasts: list[CohortForecast]) -> None:
        board = leaderboard(panel)
        board.insert(0, "board", label)
        boards.append(board)
        pw = pointwise_with_pit(panel, forecasts)
        pw.insert(0, "board", label)
        pointwise.append(pw)
        for name, frames in censuses.items():
            frame = getattr(panel, name).copy()
            frame.insert(0, "board", label)
            frames.append(frame)

    for cutoff in cutoffs:
        panels[cutoff] = align_panel(forecasts_by_cutoff[cutoff], units=units)
        add(cutoff, panels[cutoff], forecasts_by_cutoff[cutoff])

    stacking: StackingResult | None = None
    stack_error: str | None = None
    if len(cutoffs) >= 2:
        weights_cutoff, eval_cutoff = cutoffs[0], cutoffs[-1]
        try:
            # only the weight-fitted members enter: stack() refuses a member-set
            # mismatch, and a cohort-level density refusal at evaluation (e.g.
            # nn_transformer's pinned dev) would otherwise count as a member
            members = set(panels[weights_cutoff].elpd_members)
            evaluation = [f for f in forecasts_by_cutoff[eval_cutoff] if f.model in members]
            stacking = stack(
                panels[weights_cutoff], evaluation, method=stack_method, seed=stack_seed
            )
            stacked_all = [*forecasts_by_cutoff[eval_cutoff], *stacking.forecasts]
            add(
                f"{eval_cutoff}+stacked_{stack_method}",
                align_panel(stacked_all, units=units),
                stacked_all,
            )
        except Exception as e:  # noqa: BLE001 - the base boards stand on their own
            stack_error = f"{type(e).__name__}: {e}"

    def cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return Outputs(
        boards=cat(boards),
        pointwise=cat(pointwise),
        coverage=cat(censuses["coverage"]),
        absences=cat(censuses["absences"]),
        dropped=cat(censuses["dropped"]),
        excluded=cat(censuses["excluded"]),
        stacking=stacking,
        stack_error=stack_error,
    )


# -- the incremental fits log --------------------------------------------------

FIT_COLUMNS = [
    "model",
    "line",
    "company_code",
    "as_of",
    "stage",
    "seconds",
    "fit_seconds",
    "error",
    *_DIAG_KEYS,
    "n_cells",
    "has_density",
    "has_draws",
    "density_absence",
    "draws_absence",
    "n_draws_density",
    "n_draws_sample",
    "mart_publish_id",
    "clamp",
]


class FitLog:
    """Append-one-row-per-fit CSV, flushed as pool results land.

    Written BEFORE panel assembly by construction: the progress callback feeds
    it, so a crash hours in still leaves every completed fit on disk. Every
    attempt is logged (``stage`` distinguishes an escalated re-fit from its
    stage-1 failure); the ``forecast`` payload itself is not CSV-able and is
    dropped here.
    """

    def __init__(self, path: Path, *, mart_publish_id: str) -> None:
        self.path = path
        self._publish = mart_publish_id
        self._file = None
        self._writer = None

    def write(self, row: dict, *, stage: int | None = None) -> None:
        flat = {k: row.get(k) for k in FIT_COLUMNS}
        if stage is not None:
            flat["stage"] = stage
        flat["mart_publish_id"] = self._publish
        flat["clamp"] = CLAMP
        if self._writer is None:
            # long-lived handle, flushed per row and closed by close() - a
            # context manager per row would defeat the incremental log
            self._file = open(self.path, "w", newline="", encoding="utf-8")  # noqa: SIM115
            self._writer = csv.DictWriter(self._file, fieldnames=FIT_COLUMNS)
            self._writer.writeheader()
        self._writer.writerow(flat)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


# -- main ----------------------------------------------------------------------


def _print_progress(fitlog: FitLog):
    def progress(row: dict, stage: int, done: int, total: int) -> None:
        fitlog.write(row, stage=stage)
        where = (
            f"  [{done}/{total} stage{stage}] {row['model']} {row['as_of']} "
            f"{row['line']} {row['company_code']}"
        )
        if row.get("error") is not None:
            print(f"{where}: FAILED {row['error']}", flush=True)
            return
        status = []
        for label, flag, absence in (
            ("elpd", "has_density", "density_absence"),
            ("crps", "has_draws", "draws_absence"),
        ):
            status.append(label if row.get(flag) else f"no-{label}({row.get(absence, '')})")
        print(f"{where}: {' '.join(status)} ({row.get('seconds', 0.0):.1f}s)", flush=True)

    return progress


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--warehouse",
        default=DEFAULT_WAREHOUSE,
        help="local warehouse path or github://owner/repo@publish_id "
        "(default: the latest GitHub release)",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_MODELS),
        choices=ALL_MODELS,
        help="board subset (default: every board entry)",
    )
    ap.add_argument(
        "--per-line", type=int, default=0, help="cap screened companies (0 = all passing)"
    )
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=2500)
    # one seed drives sampler streams and every predict_at, so a published run
    # reproduces cell for cell
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--nn-features", nargs="*", default=["reported_loss"])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parents[1] / "analysis" / "results",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="process-pool size (default: IBNR_MAX_WORKERS env var, else all cores minus one)",
    )
    ap.add_argument(
        "--serial",
        action="store_true",
        help="run everything in this process (debugging; real tracebacks)",
    )
    ap.add_argument(
        "--no-escalate",
        action="store_true",
        help="single stage at the entry's default sampler settings",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    worker_models = [m for m in args.models if m in WORKER_MODELS]
    run_nn = POOLED_MODEL in args.models
    unavailable = [m for m in args.models if m in UNAVAILABLE_MODELS]

    mart = active_mart_path(args.warehouse)
    # pinned + cache-warmed so pool workers never resolve @latest or call gh
    source = pinned_source(args.warehouse)
    publish = active_publish_id(args.warehouse)
    print(f"task {TASK} | cutoffs {CUTOFFS} | clamp {CLAMP} | mart publish {publish}", flush=True)

    print("selecting companies (screens per line):", flush=True)
    scored, pools = screened_company_lines(mart, MEYERS_LINES, args.per_line)
    pairs = pools["multiline"]
    if not pairs:
        print("no companies pass the screens on >=2 lines")
        return 1

    fitlog = FitLog(args.out_dir / "heldout_fits.csv", mart_publish_id=publish)
    progress = _print_progress(fitlog)

    # run_retro only precompiles for its OWN runner; with a custom one the
    # spawned workers would race the cmdstan compiler over one executable
    if worker_models:
        precompile({WORKER_MODELS[m].entry for m in worker_models})

    forecasts_by_cutoff: dict[str, list[CohortForecast]] = {c: [] for c in CUTOFFS}
    for model in worker_models:
        spec = WORKER_MODELS[model]
        tasks = [
            RetroTask(
                model=model,
                warehouse=source,
                line=line,
                company_code=code,
                as_of=cutoff,
                loss_field=FIELD,
                clamp_paid=True,  # the uniform board clamp (module docstring)
                seed=args.seed,
                fit_kwargs=dict(spec.fit_kwargs),
            )
            for cutoff in CUTOFFS
            for code, line in pairs
        ]
        print(f"\n{model}: {len(tasks)} cohort fits (entry {spec.entry})", flush=True)
        rows = run_retro(
            tasks,
            stages=stages_for(spec.entry, args),
            max_workers=args.workers,
            executor="serial" if args.serial else "process",
            runner=run_cohort,
            progress=progress,
        )
        for row in rows:
            forecast = row.get("forecast")
            if forecast is not None:
                forecasts_by_cutoff[row["as_of"]].append(forecast)

    if run_nn or unavailable:
        lines_needed = sorted({line for _, line in pairs})
        tri_pool = clamp_paid_floor(pair_filter(load_schedule_p(source, lines=lines_needed), pairs))
        for cutoff in CUTOFFS:
            cellmap = cells_for_pairs(tri_pool, pairs, cutoff)
            if run_nn:
                print(f"\n{POOLED_MODEL}: pooled fit at {cutoff} on {len(pairs)} pairs", flush=True)
                t0 = time.perf_counter()
                entry, fit_error = None, None
                try:
                    entry = fit_pooled_nn(
                        tri_pool, as_of=cutoff, seed=args.seed, nn_features=args.nn_features
                    )
                except Exception as e:  # noqa: BLE001
                    fit_error = f"{type(e).__name__}: {e}"
                    print(f"  pooled fit FAILED: {fit_error}", flush=True)
                fit_seconds = time.perf_counter() - t0
                rows, forecasts = score_pooled(
                    entry,
                    cellmap,
                    as_of=cutoff,
                    seed=args.seed,
                    fit_seconds=fit_seconds,
                    fit_error=fit_error,
                )
                for i, row in enumerate(rows, 1):
                    progress(row, 1, i, len(rows))
                forecasts_by_cutoff[cutoff].extend(forecasts)
            if unavailable:
                rows, forecasts = unavailable_forecasts(cellmap, unavailable, as_of=cutoff)
                for row in rows:
                    fitlog.write(row, stage=1)
                forecasts_by_cutoff[cutoff].extend(forecasts)
    fitlog.close()

    print("\nassembling panels and boards", flush=True)
    out = build_outputs(
        {c: fs for c, fs in forecasts_by_cutoff.items() if fs}, stack_seed=args.seed
    )

    named = {
        "heldout_leaderboard.csv": out.boards,
        "heldout_pointwise.csv": out.pointwise,
        "heldout_coverage.csv": out.coverage,
        "heldout_absences.csv": out.absences,
        "heldout_dropped.csv": out.dropped,
        "heldout_excluded.csv": out.excluded,
    }
    for name, frame in named.items():
        frame = frame.copy()
        frame["mart_publish_id"] = publish
        frame["clamp"] = CLAMP
        path = args.out_dir / name
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")

    if out.stacking is not None:
        s = out.stacking
        payload = {
            "method": s.method,
            "model": s.model,
            "weights": s.weights,
            "n_cells_weight_fit": s.n_cells_weight_fit,
            "n_floored_neg_inf": s.n_floored_neg_inf,
            "weights_as_of": str(s.weights_as_of),
            "weights_fingerprint": s.weights_fingerprint,
            "mart_publish_id": publish,
            "clamp": CLAMP,
            "task": TASK,
        }
        path = args.out_dir / "heldout_stacking.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {path}")
        print(f"stacking weights ({s.method}): {s.weights}")
    if out.stack_error is not None:
        print(f"STACKING FAILED (base boards unaffected): {out.stack_error}")

    show = [
        "board",
        "model",
        "elpd",
        "elpd_per_cell",
        "n_cells_elpd",
        "elpd_status",
        "crps",
        "crps_per_cell",
        "n_cells_crps",
        "crps_status",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(f"\n{out.boards[show].to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
