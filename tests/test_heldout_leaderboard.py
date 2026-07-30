"""scripts/heldout_leaderboard.py - the absence mapping, the uniform clamp,
panel assembly and the PIT column, all mart-free.

Why a test file for a script: the runner's absence mapping DECIDES what the
board prints beside every missing number, and forecast.py's vocabulary makes a
wrong reason a false statement about a model (``no_predictive_density`` beside
an entry that merely failed to fit libels it permanently). Every mapping test
below asserts the exact reason string on the exact axis, because the adjacent
mutants - the other reason, the other axis, a swallowed detail - all produce a
board that still renders.

The stub entries follow ``tests/test_harness.py``'s ScriptedRunner pattern:
``run_cohort`` takes injectable ``load``/``resolve`` seams, so none of this
needs the mart, cmdstan, or torch.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.forecast import (
    MODEL_ABSENCE_REASONS,
    Absence,
    CohortForecast,
    align_panel,
)
from ibnr.kernels.harness import RetroTask, SamplerSettings, run_retro
from ibnr.kernels.holdout import HoldoutCells
from ibnr.triangle.core import Triangle

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import heldout_leaderboard as hl  # noqa: E402

AS_OF = "1992-12-31"  # inside the study window, so STUDY_ORIGINS keeps every origin
EARLIER = "1991-12-31"
N_DRAWS = 200


# -- fixtures (the test_forecast pattern, shifted into the study window) -------


def _rows(company, *, n=5, through=6, start=1988, scale=1.0, premium=1000.0):
    out = []
    for w in range(1, n + 1):
        for d in range(1, n + 1):
            if w + d - 1 > through:
                continue
            out.append(
                {
                    "company": company,
                    "origin_period": dt.date(start + w - 1, 1, 1),
                    "dev_lag": 12 * d,
                    "eval_date": dt.date(start + w + d - 2, 12, 31),
                    "field": "paid_loss",
                    "value": float((1000 * w + 100 * d) * scale),
                }
            )
        out.append(
            {
                "company": company,
                "origin_period": dt.date(start + w - 1, 1, 1),
                "dev_lag": 12,
                "eval_date": dt.date(start + w - 1, 12, 31),
                "field": "earned_premium",
                "value": float(premium),
            }
        )
    return out


def _triangle(company, *, doctor=None, **kw) -> Triangle:
    frame = pd.DataFrame(_rows(company, **kw))
    if doctor is not None:
        frame = doctor(frame)
    return Triangle.from_long(frame, measure="cumulative")


def _cells(company="CO_A", *, as_of=AS_OF, **kw) -> HoldoutCells:
    return hl.study_cells(_triangle(company, **kw), as_of)


def _density(cells, *, rng, shift=0.0, n_draws=N_DRAWS):
    return rng.normal(-np.log(cells.values) - 2.0 + shift, 0.3, size=(n_draws, cells.n_cells))


def _draws(cells, *, rng, sd=0.25, n_draws=N_DRAWS):
    return np.exp(rng.normal(np.log(cells.values), sd, size=(n_draws, cells.n_cells)))


def _forecast(model, cells, *, rng, shift=0.0, density=True, draws=True):
    kw = {}
    kw["log_density"] = _density(cells, rng=rng, shift=shift) if density else None
    kw["draws"] = _draws(cells, rng=rng) if draws else None
    if not density:
        kw["density_absence"] = Absence("no_predictive_density", "test")
    if not draws:
        kw["draws_absence"] = Absence("no_cell_sampler", "test")
    return CohortForecast(model=model, task=hl.TASK, cells=cells, field="paid_loss", **kw)


@pytest.fixture
def rng():
    return np.random.default_rng(20260727)


# -- stub entries (duck-typed: run_cohort resolves classes, calls the mixin API)


class StubEntry:
    """Well-behaved density + draws entry."""

    def fit(self, triangle, *, as_of=None, loss_field="paid_loss", seed=None):
        self.seed = seed
        return self

    def convergence(self):
        return {
            "max_rhat": 1.001,
            "min_ess_bulk": 900.0,
            "min_ess_tail": 800.0,
            "divergence_frac": 0.0,
        }

    def log_lik_at(self, cells, *, field=None):
        rng = np.random.default_rng(0)
        return _density(cells, rng=rng, n_draws=50)

    def predict_at(self, cells, *, field=None, seed=None):
        rng = np.random.default_rng(seed if seed is not None else 1)
        return _draws(cells, rng=rng, n_draws=50)


class DensityRefuses(StubEntry):
    def log_lik_at(self, cells, *, field=None):
        raise ValueError("non-positive loss under a lognormal")


class DrawsRefuse(StubEntry):
    def predict_at(self, cells, *, field=None, seed=None):
        raise ValueError("every requested cell sits at a pinned dev step")


class FitFails(StubEntry):
    def fit(self, triangle, *, as_of=None, loss_field="paid_loss", seed=None):
        raise RuntimeError("sampler blew up")


class PointMassDraws(StubEntry):
    def predict_at(self, cells, *, field=None, seed=None):
        return np.full((50, cells.n_cells), 7.0)


class NaNDensity(StubEntry):
    def log_lik_at(self, cells, *, field=None):
        out = _density(cells, rng=np.random.default_rng(0), n_draws=50)
        out[0, 0] = np.nan
        return out


STUBS = {
    "stub_good": StubEntry,
    "stub_density_refuses": DensityRefuses,
    "stub_draws_refuse": DrawsRefuse,
    "stub_fit_fails": FitFails,
    "stub_point_mass": PointMassDraws,
    "stub_nan_density": NaNDensity,
}


def _load(task: RetroTask) -> Triangle:
    return _triangle(task.company_code)


def _task(model, company="CO_A", as_of=AS_OF):
    return RetroTask(
        model=model,
        warehouse="unused",
        line="wc",
        company_code=company,
        as_of=as_of,
        loss_field="paid_loss",
        clamp_paid=True,
        seed=7,
    )


def _run(monkeypatch, model, entry_key, *, density_reason=None, density_detail="", load=_load):
    """One run_cohort call against a stubbed board model."""
    monkeypatch.setitem(
        hl.WORKER_MODELS,
        model,
        hl.BoardModel(entry_key, density_reason=density_reason, density_detail=density_detail),
    )
    return hl.run_cohort(
        _task(model), SamplerSettings(), load=load, resolve=lambda name: STUBS[name]
    )


# =============================================================================
# 1. the absence mapping (each raise class -> the right reason on the right axis)
# =============================================================================


def test_good_fit_offers_both_axes_and_diagnostics(monkeypatch):
    row = _run(monkeypatch, "m1", "stub_good")
    assert row["error"] is None
    f = row["forecast"]
    assert f.has_density and f.has_draws
    assert f.model == "m1" and f.task == hl.TASK and f.field == "paid_loss"
    assert row["max_rhat"] == 1.001 and row["divergence_frac"] == 0.0
    assert row["seconds"] > 0
    # the flat CSV columns mirror the forecast
    assert row["has_density"] and row["has_draws"] and row["n_cells"] == f.n_cells


def test_density_raise_is_scoring_refused_on_the_density_axis_only(monkeypatch):
    """Mutants: 'fit_failed' (wrong scope - the fit exists), the draws axis
    (the raise came from log_lik_at), a swallowed exception text."""
    row = _run(monkeypatch, "m1", "stub_density_refuses")
    f = row["forecast"]
    assert row["error"] is None  # a refusal is a verdict, not a crash
    assert not f.has_density
    assert f.density_absence.reason == "scoring_refused"
    assert "ValueError" in f.density_absence.detail
    assert "non-positive loss" in f.density_absence.detail
    assert f.has_draws  # the other axis is untouched


def test_draws_raise_is_scoring_refused_on_the_draws_axis_only(monkeypatch):
    row = _run(monkeypatch, "m1", "stub_draws_refuse")
    f = row["forecast"]
    assert f.has_density
    assert not f.has_draws
    assert f.draws_absence.reason == "scoring_refused"
    assert "pinned dev" in f.draws_absence.detail


def test_fit_raise_is_fit_failed_on_both_axes_for_a_density_entry(monkeypatch):
    row = _run(monkeypatch, "m1", "stub_fit_fails")
    assert "RuntimeError" in row["error"]
    f = row["forecast"]
    assert f.density_absence.reason == "fit_failed"
    assert f.draws_absence.reason == "fit_failed"
    assert "sampler blew up" in f.draws_absence.detail


def test_fit_raise_keeps_the_model_level_density_reason_of_a_draws_only_entry(monkeypatch):
    """THE uniformity trap. A model-level reason must hold on EVERY cohort:
    stamping fit_failed on ODP's density axis for one failed cohort makes
    align_panel refuse the whole model (its own census would contradict the
    permanent N/A). The mutant that maps fit_failed onto both axes
    unconditionally is exactly one branch away."""
    row = _run(
        monkeypatch,
        "m1",
        "stub_fit_fails",
        density_reason="no_predictive_density",
        density_detail="quasi-likelihood",
    )
    f = row["forecast"]
    assert f.density_absence.reason == "no_predictive_density"
    assert f.density_absence.detail == "quasi-likelihood"
    assert f.draws_absence.reason == "fit_failed"

    # the property the mapping protects: a mixed good/failed cohort set for the
    # draws-only model still aligns beside a density model
    good = _run(monkeypatch, "m1", "stub_good", density_reason="no_predictive_density")
    other = _run(monkeypatch, "m2", "stub_good")
    other_b = hl.run_cohort(
        _task("m2", company="CO_B"),
        SamplerSettings(),
        load=_load,
        resolve=lambda name: STUBS[name],
    )
    failed_b = hl.run_cohort(
        RetroTask(
            model="m1",
            warehouse="unused",
            line="wc",
            company_code="CO_B",
            as_of=AS_OF,
            loss_field="paid_loss",
            seed=7,
        ),
        SamplerSettings(),
        load=_load,
        resolve=lambda name: FitFails,
    )
    panel = align_panel(
        [good["forecast"], failed_b["forecast"], other["forecast"], other_b["forecast"]]
    )
    assert "m1" not in panel.elpd_members
    assert "m1" in panel.crps_members


def test_draws_only_success_carries_the_model_level_reason_and_detail(monkeypatch):
    row = _run(
        monkeypatch,
        "m1",
        "stub_good",
        density_reason="no_predictive_density",
        density_detail="bootstrap, no observation model",
    )
    f = row["forecast"]
    assert not f.has_density and f.has_draws
    assert f.density_absence.reason == "no_predictive_density"
    assert f.density_absence.detail == "bootstrap, no observation model"


def test_point_mass_draws_demote_to_scoring_refused_not_an_error_row(monkeypatch):
    """The CohortForecast zero-variance raise, mapped per axis by the probe:
    an all-degenerate draws array must not hide the model's good density."""
    row = _run(monkeypatch, "m1", "stub_point_mass")
    f = row["forecast"]
    assert row["error"] is None
    assert f.has_density  # untouched
    assert not f.has_draws
    assert f.draws_absence.reason == "scoring_refused"
    assert "zero variance" in f.draws_absence.detail


def test_nan_density_demotes_to_scoring_refused_and_keeps_draws(monkeypatch):
    row = _run(monkeypatch, "m1", "stub_nan_density")
    f = row["forecast"]
    assert not f.has_density
    assert f.density_absence.reason == "scoring_refused"
    assert "NaN" in f.density_absence.detail
    assert f.has_draws


def test_runner_through_run_retro_serial_keeps_order_and_escalates(monkeypatch):
    """The ScriptedRunner pattern, with the real run_cohort as the runner: an
    error row escalates to stage 2 and the final row replaces the failure."""

    class FlakyEntry(StubEntry):
        calls = []

        def fit(self, triangle, *, as_of=None, loss_field="paid_loss", seed=None):
            FlakyEntry.calls.append(1)
            if len(FlakyEntry.calls) == 1:
                raise RuntimeError("stage-1 blowup")
            return self

    monkeypatch.setitem(hl.WORKER_MODELS, "m_flaky", hl.BoardModel("flaky"))
    monkeypatch.setitem(hl.WORKER_MODELS, "m_ok", hl.BoardModel("stub_good"))
    resolve = lambda name: FlakyEntry if name == "flaky" else STUBS[name]  # noqa: E731
    rows = run_retro(
        [_task("m_flaky"), _task("m_ok", company="CO_B")],
        stages=(SamplerSettings(), SamplerSettings(target_accept=0.99)),
        executor="serial",
        runner=lambda task, settings: hl.run_cohort(task, settings, load=_load, resolve=resolve),
    )
    assert [r["model"] for r in rows] == ["m_flaky", "m_ok"]
    assert rows[0]["stage"] == 2 and rows[0]["error"] is None
    assert rows[0]["forecast"].has_density and rows[0]["forecast"].has_draws
    assert rows[1]["stage"] == 1


# =============================================================================
# 2. the uniform paid clamp
# =============================================================================


def _doctored_load(task: RetroTask) -> Triangle:
    """Cohort triangle with a sub-1 held-out cell and a negative training cell."""

    def doctor(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        held_out = (
            (frame["field"] == "paid_loss")
            & (frame["origin_period"] == dt.date(1989, 1, 1))
            & (frame["dev_lag"] == 60)
        )
        train = (
            (frame["field"] == "paid_loss")
            & (frame["origin_period"] == dt.date(1990, 1, 1))
            & (frame["dev_lag"] == 12)
        )
        frame.loc[held_out, "value"] = 0.25
        frame.loc[train, "value"] = -3.0
        return frame

    return _triangle(task.company_code, doctor=doctor)


def test_clamp_is_applied_once_and_uniformly_across_entries(monkeypatch):
    """Two entries' cells carry IDENTICAL clamped values - the align_panel
    value-agreement property the per-entry clamp is documented to break.
    Mutant: dropping ibis.greatest leaves 0.25 in the held-out cell."""
    raw = hl.study_cells(_doctored_load(_task("x")), AS_OF)
    key = raw.frame["origin_period"] == dt.date(1989, 1, 1)
    assert raw.frame.loc[key, "value"].item() == 0.25  # pre-clamp: genuinely sub-1

    monkeypatch.setitem(hl.WORKER_MODELS, "m1", hl.BoardModel("stub_good"))
    monkeypatch.setitem(hl.WORKER_MODELS, "m2", hl.BoardModel("stub_draws_refuse"))
    resolve = lambda name: STUBS[name]  # noqa: E731
    a = hl.run_cohort(_task("m1"), SamplerSettings(), load=_doctored_load, resolve=resolve)
    b = hl.run_cohort(_task("m2"), SamplerSettings(), load=_doctored_load, resolve=resolve)

    fa, fb = a["forecast"], b["forecast"]
    va = fa.key_frame.set_index(["origin_period", "dev_lag"])["value"]
    assert va[(dt.date(1989, 1, 1), 60)] == 1.0  # floored, not 0.25
    pd.testing.assert_frame_equal(fa.key_frame, fb.key_frame)
    panel = align_panel([fa, fb])  # the check the clamp exists to satisfy
    assert panel.n_cells == fa.n_cells


def test_clamped_triangle_floors_training_cells_too():
    tri = hl.clamp_paid_floor(_doctored_load(_task("x")))
    frame = tri.execute()
    paid = frame[frame["field"] == "paid_loss"]
    assert paid["value"].min() >= 1.0
    # the premium field is untouched by a PAID clamp
    prem = frame[frame["field"] == "earned_premium"]
    assert (prem["value"] == 1000.0).all()


# =============================================================================
# 3. the not-implemented rows and the pooled scorer
# =============================================================================


def test_unavailable_forecasts_name_scorer_not_implemented_on_both_axes():
    cellmap = {("A", "wc"): _cells("A"), ("B", "wc"): "ValueError: no cells"}
    rows, forecasts = hl.unavailable_forecasts(cellmap, ["sur"], as_of=AS_OF)
    assert len(rows) == 2 and len(forecasts) == 1
    f = forecasts[0]
    assert f.model == "sur"
    assert f.density_absence.reason == "scorer_not_implemented"
    assert f.draws_absence.reason == "scorer_not_implemented"
    assert f.density_absence.detail == hl.UNAVAILABLE_MODELS["sur"]
    failed = next(r for r in rows if r["company_code"] == "B")
    assert failed["error"] == "ValueError: no cells"
    assert failed.get("forecast") is None


def test_score_pooled_maps_a_pooled_fit_failure_to_every_cohort():
    cellmap = {("A", "wc"): _cells("A"), ("B", "wc"): _cells("B")}
    rows, forecasts = hl.score_pooled(
        None, cellmap, as_of=AS_OF, seed=1, fit_seconds=2.5, fit_error="ImportError: torch"
    )
    assert len(forecasts) == 2
    for f in forecasts:
        assert f.density_absence.reason == "fit_failed"
        assert f.draws_absence.reason == "fit_failed"
        assert "torch" in f.draws_absence.detail
    assert all(r["error"] == "ImportError: torch" and r["fit_seconds"] == 2.5 for r in rows)


def test_score_pooled_success_uses_the_same_absence_mapping():
    cellmap = {("A", "wc"): _cells("A"), ("B", "wc"): "ValueError: empty"}
    rows, forecasts = hl.score_pooled(StubEntry(), cellmap, as_of=AS_OF, seed=3, fit_seconds=1.0)
    assert len(forecasts) == 1
    assert forecasts[0].has_density and forecasts[0].has_draws
    assert forecasts[0].model == hl.POOLED_MODEL
    assert next(r for r in rows if r["company_code"] == "B")["error"] == "ValueError: empty"


# =============================================================================
# 4. PIT - cell level only, mean(draws <= value)
# =============================================================================


def test_pit_values_exact_on_a_hand_built_array():
    """Mutants: strict '<' (ties dropped: 0.5 -> 0.25), '>=' (complement),
    axis flipped (draws averaged per draw rather than per cell)."""
    draws = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    got = hl.pit_values(draws, np.array([3.0, 1.0]))
    np.testing.assert_allclose(got, [0.75, 0.0])
    # ties count as <=: two of four draws at exactly the value
    np.testing.assert_allclose(hl.pit_values(np.array([[2.0], [2.0], [3.0], [5.0]]), [2.0]), [0.5])
    np.testing.assert_allclose(hl.pit_values(draws, np.array([10.0, 10.0])), [1.0, 1.0])


def test_pointwise_pit_lands_on_the_right_model_and_cell(rng):
    cells = _cells("CO_A")
    with_draws = _forecast("m_draws", cells, rng=rng)
    density_only = _forecast("m_density", cells, rng=rng, draws=False)
    panel = align_panel([with_draws, density_only])
    pw = hl.pointwise_with_pit(panel, [with_draws, density_only])

    assert "key" not in pw.columns
    no_draws = pw[pw["model"] == "m_density"]
    assert no_draws["pit"].isna().all()

    mine = pw[pw["model"] == "m_draws"].sort_values(["origin_period", "dev_lag"])
    expected = hl.pit_values(with_draws.draws, cells.values)
    np.testing.assert_allclose(mine["pit"].to_numpy(dtype=float), expected)
    # a PIT is a probability; the draws are centered on the outcome so the
    # interior values are strictly between the degenerate endpoints
    assert ((mine["pit"] > 0) & (mine["pit"] < 1)).all()


# =============================================================================
# 5. panel assembly -> boards, stacking across the two cutoffs
# =============================================================================


def _study_forecasts(rng):
    """Two cutoffs x two cohorts x five models, one shared cells per cohort.

    ``m_refused`` mirrors the real nn_transformer shape: draws everywhere, a
    COHORT-level density refusal on every cohort (so it is not an ELPD member
    anywhere). It exists to pin build_outputs' evaluation filter: unfiltered,
    its cohort-level absences would count it as an evaluation density model and
    stack() would refuse the member-set mismatch.
    """
    out = {}
    for cutoff in (EARLIER, AS_OF):
        forecasts = []
        for company in ("CO_A", "CO_B"):
            cells = _cells(company, as_of=cutoff)
            forecasts += [
                _forecast("m_good", cells, rng=rng),
                _forecast("m_shift", cells, rng=rng, shift=-0.5),
                _forecast("m_drawsonly", cells, rng=rng, density=False),
                CohortForecast(
                    model="m_refused",
                    task=hl.TASK,
                    cells=cells,
                    field="paid_loss",
                    draws=_draws(cells, rng=rng),
                    density_absence=Absence("scoring_refused", "pinned dev"),
                ),
                CohortForecast.unavailable(
                    model="m_unavail",
                    task=hl.TASK,
                    cells=cells,
                    density_reason="scorer_not_implemented",
                    draws_reason="scorer_not_implemented",
                    detail="test",
                    field="paid_loss",
                ),
            ]
        out[cutoff] = forecasts
    return out


def test_build_outputs_boards_and_the_stacked_pseudo_model(rng):
    pytest.importorskip("bayesblend")
    out = hl.build_outputs(_study_forecasts(rng), units=None)

    labels = list(dict.fromkeys(out.boards["board"]))
    assert labels == [EARLIER, AS_OF, f"{AS_OF}+stacked_mle"]
    assert out.stack_error is None

    # the stacked pseudo-model is a real board row on the third board only
    stacked = out.boards[out.boards["model"] == "stacked_mle"]
    assert list(stacked["board"]) == [f"{AS_OF}+stacked_mle"]
    assert stacked["elpd"].notna().all() and stacked["crps"].notna().all()

    s = out.stacking
    assert s is not None and s.method == "mle"
    assert set(s.weights) == {"m_good", "m_shift"}
    assert abs(sum(s.weights.values()) - 1.0) < 1e-6
    assert s.weights_as_of == dt.date(1991, 12, 31)

    # every model occupies a row on every board it was offered to; the
    # cohort-level refuser is on the CRPS side only
    for label in (EARLIER, AS_OF):
        board = out.boards[out.boards["board"] == label].set_index("model")
        assert set(board.index) == {"m_good", "m_shift", "m_drawsonly", "m_refused", "m_unavail"}
        assert board.loc["m_refused", "elpd_status"] == "na: no_scored_cohorts"
        assert board.loc["m_refused", "crps"].item() > 0.0


def test_missing_scores_are_na_never_zero(rng):
    """0.0 is simultaneously the best ELPD and the best CRPS on this board -
    the one imputation the whole milestone forbids."""
    pytest.importorskip("bayesblend")
    out = hl.build_outputs(_study_forecasts(rng), units=None)
    for label in (EARLIER, AS_OF):
        board = out.boards[out.boards["board"] == label].set_index("model")
        assert pd.isna(board.loc["m_unavail", "elpd"])
        assert pd.isna(board.loc["m_unavail", "crps"])
        assert pd.isna(board.loc["m_drawsonly", "elpd"])
        assert board.loc["m_drawsonly", "crps"].item() > 0.0  # scored, and CRPS is positive
        assert board.loc["m_good", "elpd"].item() != 0.0
        assert board.loc["m_unavail", "elpd_status"] == "na: scorer_not_implemented"
    # and the pointwise pit column exists everywhere, NA for the draw-less
    assert out.pointwise.loc[out.pointwise["model"] == "m_unavail", "pit"].isna().all()


def test_build_outputs_single_cutoff_skips_stacking(rng):
    forecasts = _study_forecasts(rng)
    out = hl.build_outputs({AS_OF: forecasts[AS_OF]}, units=None)
    assert list(dict.fromkeys(out.boards["board"])) == [AS_OF]
    assert out.stacking is None and out.stack_error is None


def test_stacking_failure_is_recorded_not_fatal(rng, monkeypatch):
    monkeypatch.setattr(hl, "stack", _raise_stack)
    out = hl.build_outputs(_study_forecasts(rng), units=None)
    assert out.stacking is None
    assert "boom" in out.stack_error
    assert list(dict.fromkeys(out.boards["board"])) == [EARLIER, AS_OF]


def _raise_stack(*args, **kwargs):
    raise ValueError("boom")


# =============================================================================
# 6. the board registry itself
# =============================================================================


def test_board_registry_reasons_are_axis_valid_and_names_unique():
    """A wrong-axis model-level reason would raise at CohortForecast time deep
    in a worker; catch it at test time instead. The board names must be unique
    because they ARE the panel's model identity."""
    assert len(hl.ALL_MODELS) == len(set(hl.ALL_MODELS))
    for name, spec in hl.WORKER_MODELS.items():
        if spec.density_reason is not None:
            assert spec.density_reason in MODEL_ABSENCE_REASONS, name
            Absence(spec.density_reason, spec.density_detail).check_axis("density")
    # the two compartmental arms are one entry under two board names
    assert hl.WORKER_MODELS["compartmental_gaussian"].entry == "compartmental"
    assert hl.WORKER_MODELS["compartmental_gaussian"].fit_kwargs == {"variant": "gaussian"}
    assert hl.WORKER_MODELS["compartmental_lognormal"].fit_kwargs == {"variant": "lognormal"}
    for reason in ("scorer_not_implemented",):
        assert reason in MODEL_ABSENCE_REASONS


def test_study_constants_match_the_locked_design():
    assert hl.TASK == "paid_next_diagonal_v1"
    assert hl.CUTOFFS == ("1996-12-31", "1997-12-31")
    assert hl.STUDY_ORIGINS[0] == dt.date(1988, 1, 1)
    assert hl.STUDY_ORIGINS[-1] == dt.date(1997, 1, 1)
    assert len(hl.STUDY_ORIGINS) == 10
    # stages_for consumes the CLI namespace shape main() builds
    stages = hl.stages_for(
        "compartmental",
        SimpleNamespace(no_escalate=False, chains=4, warmup=1000, draws=2500),
    )
    assert len(stages) == 2 and stages[0].target_accept == 0.9


# -- review of #77: four defects between the pool and the published board ------


def test_a_fit_failing_the_gates_at_the_final_stage_becomes_an_absence(rng):
    """THE P1. ``run_retro`` uses the gates to decide what to RE-RUN; it does not
    discard a fit that still fails at the last escalation stage, and this script
    used to append that forecast to a PUBLISHED board with nothing saying so.

    Downgraded rather than dropped, and the difference is not cosmetic: dropping
    shrinks the panel, and ``align_panel`` INTERSECTS, so one model's silent drop
    deletes those cells from every other model's column - a badly-converged fit
    would make the board look better while costing everyone coverage.
    """
    cells = _cells()
    good = _forecast("meyers_ccl", cells, rng=rng)
    gates = hl.ConvergenceGates()
    spec = hl.WORKER_MODELS["meyers_ccl"]

    passing = {"forecast": good, "max_rhat": 1.001, "divergence_frac": 0.0, "min_ess_bulk": 900.0}
    assert hl.as_gated_forecast("meyers_ccl", spec, passing, gates) is good

    failed = {**passing, "max_rhat": 1.42}
    out = hl.as_gated_forecast("meyers_ccl", spec, failed, gates)
    assert out is not good
    assert out.draws is None and out.draws_absence.reason == "fit_failed"
    assert "max_rhat=1.42" in out.draws_absence.detail
    # the cohort is KEPT, on the same cells, so the panel does not shrink
    assert out.cells is cells


def test_the_gated_downgrade_respects_a_model_level_density_reason(rng):
    """A draws-only entry keeps ``no_predictive_density`` on the density axis.

    Stamping ``fit_failed`` there for one cohort would contradict a model-level
    claim and make align_panel refuse the whole model - the same rule
    ``absence_forecast`` already encodes, which is why this reuses it.
    """
    cells = _cells()
    forecast = _forecast("mack", cells, rng=rng, density=False)
    row = {"forecast": forecast, "max_rhat": 9.9}
    out = hl.as_gated_forecast("mack", hl.WORKER_MODELS["mack"], row, hl.ConvergenceGates())
    assert out.density_absence.reason == "no_predictive_density"
    assert out.draws_absence.reason == "fit_failed"


def test_progress_survives_a_pool_machinery_row(capsys):
    """``run_retro`` emits a row for POOL machinery failures carrying only
    model/line/company_code/error - no ``as_of``, because the task never got one.
    Indexing it raised KeyError inside the progress callback, which aborted the
    whole study and hid the real failure behind a missing-key traceback."""
    written = []
    fitlog = SimpleNamespace(write=lambda row, stage: written.append(row))
    progress = hl._print_progress(fitlog)

    machinery = {
        "model": "meyers_ccl",
        "line": "workers_compensation",
        "company_code": "CO_A",
        "error": "worker: BrokenProcessPool: killed",
    }
    progress(machinery, 1, 1, 1)  # must not raise
    out = capsys.readouterr().out
    assert "FAILED" in out and "BrokenProcessPool" in out
    assert written == [machinery]


def test_the_stacking_json_always_describes_this_run(tmp_path, rng):
    """Written only on success, it left the PREVIOUS run's weights beside this
    run's freshly-overwritten CSVs - a directory that looked complete and
    self-consistent and was not. Now the failure case writes a payload naming
    the reason, so the file always describes the run that produced the CSVs."""
    stale = tmp_path / "heldout_stacking.json"
    stale.write_text(json.dumps({"weights": {"meyers_csr": 1.0}}), encoding="utf-8")

    payload = hl.stacking_payload(
        stacking=None, stack_error="bayesblend blew up", publish="pub1", models=["mack"]
    )
    stale.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    back = json.loads(stale.read_text(encoding="utf-8"))
    assert back["weights"] is None
    assert back["error"] == "bayesblend blew up"
    assert back["mart_publish_id"] == "pub1" and back["models_requested"] == ["mack"]

    # and with no stacking AND no error - a single-cutoff run - it still says so
    quiet = hl.stacking_payload(stacking=None, stack_error=None, publish="pub1", models=["mack"])
    assert quiet["weights"] is None and "no stacking was attempted" in quiet["error"]


def test_missing_extras_names_the_extra_and_the_models():
    """The documented default invocation runs every board entry, which needs
    [bayesian] AND [nn] - and a plain `uv sync` installs neither. It used to
    fail late, inside a spawned worker, as an ImportError about cmdstanpy with
    no hint that an extra was the answer."""
    needed = dict((extra, models) for extra, _, models in hl.missing_extras(["mack"], run_nn=False))
    assert needed == {}, "mack is a core entry and must not demand an extra"

    bayes = hl.missing_extras(["meyers_ccl", "mack"], run_nn=False)
    have_cmdstanpy = importlib.util.find_spec("cmdstanpy") is not None
    if have_cmdstanpy:
        assert bayes == []
    else:
        assert [e for e, _, _ in bayes] == ["bayesian"]
        assert bayes[0][2] == ["meyers_ccl"]  # named, and mack is not implicated

    # the entry->extra map is DERIVED from the registry, not hand-listed
    assert "guszcza_growth_curve" in hl.BAYESIAN_ENTRIES
    assert "mack" not in hl.BAYESIAN_ENTRIES


def test_main_actually_runs_the_preflight_before_touching_the_mart(monkeypatch, capsys):
    """The DELIVERY test, and the mutation pass is why it exists.

    Testing ``missing_extras`` alone left the preflight unwired: deleting its
    call from ``main`` kept every other test green. That is this repo's named
    inert-parameter bug class - a signature test proves a wire exists, not that
    it is connected - so this drives the real entry point and asserts main
    STOPS, before ``active_mart_path`` is reached.
    """
    called = {}

    def fake_missing(worker_models, *, run_nn):
        called["args"] = (list(worker_models), run_nn)
        return [("bayesian", "cmdstanpy", ["meyers_ccl"])]

    def exploding_mart(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("main reached the mart despite a missing extra")

    monkeypatch.setattr(hl, "missing_extras", fake_missing)
    monkeypatch.setattr(hl, "active_mart_path", exploding_mart)
    monkeypatch.setattr(
        sys, "argv", ["heldout_leaderboard.py", "--models", "meyers_ccl", "--out-dir", "."]
    )

    assert hl.main() == 2
    assert called["args"] == (["meyers_ccl"], False)
    assert "uv sync --extra bayesian" in capsys.readouterr().err
