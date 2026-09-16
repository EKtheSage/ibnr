"""Candidate selection uses completed diagonals; outer evaluation freezes forecasts.

Hand-authored errors below isolate the published weighted, per-diagonal score
from its temporal aggregation. Real fits remain attached for selection and the
separate later-ultimate check, so a winning name must resolve to a usable fit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.conventional import ConventionalCandidate
from ibnr.kernels.replay import replay_conventional
from ibnr.kernels.selection import (
    evaluate_conventional,
    score_replay,
    select_conventional,
)

from .test_conventional_replay import DATES, ORIGINS, from_frame, full_triangle


@pytest.fixture
def history(backend_name):
    tri = full_triangle(backend_name)
    replay = replay_conventional(
        tri,
        {
            "a": ConventionalCandidate(horizon=48),
            "b": ConventionalCandidate(horizon=48, average="simple", history_periods=2),
        },
        DATES,
    )
    return tri, replay


def with_errors(replay, *, errors, weights):
    """Specify constant error/weight per candidate and interval for hand checks."""
    frame = replay.cells.copy()
    for (name, end), error in errors.items():
        selected = (frame["candidate"] == name) & (frame["eval_date"] == end)
        frame.loc[selected, "actual_increment"] = weights[end]
        frame.loc[selected, "expected_increment"] = weights[end] - error
        frame.loc[selected, "ave"] = error
        frame.loc[selected, "cdr"] = error
        frame.loc[selected, "remaining_revision"] = 0.0
        frame.loc[selected, "new_ultimate"] = frame.loc[selected, "old_ultimate"] + error
    return replace(replay, cells=frame)


def mean_not_pool(replay):
    # a: two diagonal RMSEs 0 and 8 => mean 4, versus b's 5 and 5 => mean 5.
    # Pooling cells instead overweights the second diagonal and wrongly picks b.
    return with_errors(
        replay,
        errors={
            ("a", DATES[1]): 0.0,
            ("a", DATES[2]): 8.0,
            ("b", DATES[1]): 5.0,
            ("b", DATES[2]): 5.0,
        },
        weights={DATES[1]: 1.0, DATES[2]: 100.0},
    )


def choose(replay, date=DATES[-1], **kwargs):
    return select_conventional(replay, selection_as_of=date, **kwargs)


def test_selection_averages_diagonal_rmses_instead_of_pooling_cells(history):
    _, base = history
    replay = mean_not_pool(base)
    result = choose(replay)
    ranking = result.ranking.set_index("candidate")
    assert result.name == "a"
    assert ranking.loc["a", "mean_rmse"] == pytest.approx(4.0)
    assert ranking.loc["b", "mean_rmse"] == pytest.approx(5.0)
    assert ranking["eligible"].all()
    assert ranking["n_scored"].eq(2).all()
    assert ranking["n_required"].eq(2).all()
    a = replay.cells.loc[replay.cells["candidate"] == "a"]
    pooled = np.sqrt(
        np.sum(np.abs(a["actual_increment"]) * a["ave"] ** 2)
        / np.sum(np.abs(a["actual_increment"]))
    )
    assert pooled > 5.0  # the wrong aggregation would choose the other model


@pytest.mark.parametrize("metric", ["ave", "cdr"])
def test_diagonal_score_weights_squared_errors_by_absolute_actual_increment(history, metric):
    _, replay = history
    frame = replay.cells.copy()
    chosen = (frame["candidate"] == "a") & (frame["eval_date"] == DATES[1])
    assert chosen.sum() == 4
    frame.loc[chosen, "actual_increment"] = [-1.0, 3.0, 0.0, 0.0]
    frame.loc[chosen, metric] = [2.0, 4.0, 100.0, -100.0]
    scores = score_replay(replace(replay, cells=frame), metric=metric)
    score = scores.loc[(scores["candidate"] == "a") & (scores["eval_date"] == DATES[1])].iloc[0]
    assert score["rmse"] == pytest.approx(np.sqrt(13.0))
    assert score["weight_sum"] == pytest.approx(4.0)
    assert score["n_cells"] == 4
    assert score["status"] == "ok"
    assert score["metric"] == metric


def test_zero_weight_extreme_error_does_not_erase_the_positive_weight_score(history):
    _, replay = history
    frame = replay.cells.copy()
    chosen = (frame["candidate"] == "a") & (frame["eval_date"] == DATES[1])
    frame.loc[chosen, "actual_increment"] = [0.0, 1.0, 0.0, 0.0]
    frame.loc[chosen, "ave"] = [1e308, 1.0, 0.0, 0.0]
    scores = score_replay(replace(replay, cells=frame))
    score = scores.loc[(scores["candidate"] == "a") & (scores["eval_date"] == DATES[1])].iloc[0]
    assert score["status"] == "ok"
    assert score["rmse"] == pytest.approx(1.0)


def test_finite_large_diagonal_scores_do_not_overflow_into_a_false_tie(history):
    _, replay = history
    large = with_errors(
        replay,
        errors={
            (name, end): value for name, value in [("a", 1e308), ("b", 9e307)] for end in DATES[1:]
        },
        weights={end: 1.0 for end in DATES[1:]},
    )
    result = choose(large)
    assert result.name == "b"
    means = result.ranking.set_index("candidate")["mean_rmse"]
    assert np.isfinite(means).all()
    assert means["a"] == pytest.approx(1e308)
    assert means["b"] == pytest.approx(9e307)


def test_metric_choice_uses_the_corresponding_error_column(history):
    _, base = history
    replay = mean_not_pool(base)
    frame = replay.cells.copy()
    frame.loc[frame["candidate"] == "a", "cdr"] = 20.0
    frame.loc[frame["candidate"] == "b", "cdr"] = 1.0
    different = replace(replay, cells=frame)
    assert choose(different, metric="ave").name == "a"
    assert choose(different, metric="cdr").name == "b"


def test_zero_weight_diagonal_is_undefined_even_with_zero_forecast_error(history):
    _, replay = history
    frame = replay.cells.copy()
    frame["actual_increment"] = 0.0
    frame["expected_increment"] = 0.0
    frame["ave"] = 0.0
    frame["cdr"] = 0.0
    zero = replace(replay, cells=frame)
    scores = score_replay(zero)
    assert scores["status"].eq("zero_weight").all()
    assert scores["weight_sum"].eq(0.0).all()
    assert scores["rmse"].isna().all()
    with pytest.raises(ValueError, match="(?i)(candidate|eligible|scor)"):
        choose(zero)


def test_through_and_selection_date_ignore_future_scores_and_failures(history):
    _, base = history
    replay = mean_not_pool(base)
    before = choose(replay, date=DATES[1])
    frame = replay.cells.copy()
    frame.loc[frame["eval_date"] == DATES[2], ["ave", "cdr"]] = 1e12
    future_error = pd.DataFrame(
        [{"candidate": "a", "as_of": DATES[1], "eval_date": DATES[2], "reason": "future failure"}]
    )
    changed = replace(replay, cells=frame, errors=future_error)
    after = choose(changed, date=DATES[1])
    assert before.name == after.name == "a"
    pd.testing.assert_frame_equal(before.ranking, after.ranking)
    np.testing.assert_allclose(before.fit.factors, after.fit.factors)
    scores = score_replay(changed, through=DATES[1])
    assert set(scores["eval_date"]) == {DATES[1]}
    assert scores["status"].eq("ok").all()


def test_one_failed_interval_disqualifies_a_candidate_instead_of_nanmean(history):
    _, base = history
    replay = mean_not_pool(base)
    failed = (replay.cells["candidate"] == "a") & (replay.cells["eval_date"] == DATES[2])
    errors = pd.DataFrame(
        [{"candidate": "a", "as_of": DATES[1], "eval_date": DATES[2], "reason": "missing cell"}]
    )
    broken = replace(replay, cells=replay.cells.loc[~failed].copy(), errors=errors)
    result = choose(broken)
    row = result.ranking.set_index("candidate").loc["a"]
    assert result.name == "b"
    assert not row["eligible"]
    assert row["n_scored"] == 1
    assert row["n_required"] == 2
    assert np.isnan(row["mean_rmse"])
    assert isinstance(row["reason"], str) and row["reason"]
    failed_score = result.scores.loc[
        (result.scores["candidate"] == "a") & (result.scores["eval_date"] == DATES[2])
    ].iloc[0]
    assert failed_score["status"] == "failed"


def test_missing_interval_without_an_error_record_is_not_a_smaller_valid_history(history):
    _, base = history
    replay = mean_not_pool(base)
    omitted = (replay.cells["candidate"] == "a") & (replay.cells["eval_date"] == DATES[2])
    result = choose(replace(replay, cells=replay.cells.loc[~omitted].copy()))
    assert result.name == "b"
    row = result.ranking.set_index("candidate").loc["a"]
    assert not row["eligible"]
    assert np.isnan(row["mean_rmse"])


def test_equal_scores_break_ties_by_name_not_mapping_order(history):
    _, replay = history
    equal = with_errors(
        replay,
        errors={(name, end): 1.0 for name in ("a", "b") for end in DATES[1:]},
        weights={end: 1.0 for end in DATES[1:]},
    )
    reverse = replace(equal, candidates=dict(reversed(list(equal.candidates.items()))))
    assert choose(equal).name == choose(reverse).name == "a"


def test_selected_fit_is_from_the_selection_date_with_the_winning_settings(history):
    _, replay = history
    result = choose(mean_not_pool(replay), date=DATES[1])
    assert result.as_of == DATES[1]
    assert result.candidate == replay.candidates[result.name]
    assert result.fit.candidate == result.candidate
    assert result.fit.as_of == DATES[1]
    np.testing.assert_allclose(result.fit.factors, replay.fits[(result.name, DATES[1])].factors)


def test_a_candidate_without_a_selection_date_fit_cannot_win(history):
    _, base = history
    replay = mean_not_pool(base)
    fits = dict(replay.fits)
    del fits[("a", DATES[-1])]
    result = choose(replace(replay, fits=fits))
    assert result.name == "b"
    row = result.ranking.set_index("candidate").loc["a"]
    assert not row["eligible"]
    assert "fit" in row["reason"].lower()


@pytest.mark.parametrize("date", [DATES[0], dt.date(2014, 6, 30)])
def test_selection_needs_a_replay_date_with_completed_history(history, date):
    _, replay = history
    with pytest.raises(ValueError, match="(?i)(date|history|interval|scor)"):
        choose(replay, date=date)


def test_outer_evaluation_freezes_selection_and_uses_unweighted_ultimate_rmse(history):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    evaluation = evaluate_conventional(selected, tri, as_of="2017-12-31")
    rows = evaluation.origins.set_index("origin_period")
    predicted = selected.fit.origins.set_index("origin_period")["ultimate"]
    np.testing.assert_allclose(rows.loc[ORIGINS, "predicted"], predicted.loc[ORIGINS])
    assert set(rows.index) == set(ORIGINS)
    assert rows.loc[ORIGINS[:2], "known_at_selection"].all()
    assert not rows.loc[ORIGINS[:2], "included"].any()
    assert not rows.loc[ORIGINS[2:], "known_at_selection"].any()
    assert rows.loc[ORIGINS[2:], "included"].all()
    actual = np.array([1280.0, 1700.0, 2350.0])
    np.testing.assert_allclose(rows.loc[ORIGINS[2:], "observed"], actual)
    expected_rmse = np.sqrt(np.mean((predicted.loc[ORIGINS[2:]].to_numpy() - actual) ** 2))
    assert evaluation.summary["n_targets"] == 3
    assert evaluation.summary["n_observed"] == 3
    assert evaluation.summary["rmse"] == pytest.approx(expected_rmse)


def test_outer_evaluation_keeps_missing_terminal_outcomes_visible(history):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    evaluation = evaluate_conventional(selected, tri, as_of="2015-12-31")
    rows = evaluation.origins.set_index("origin_period")
    assert evaluation.summary["n_targets"] == 3
    assert evaluation.summary["n_observed"] == 1
    assert np.isnan(evaluation.summary["rmse"])
    assert "missing" in evaluation.summary["reason"].lower()
    assert rows.loc[ORIGINS[3:], "observed"].isna().all()


def test_outer_truth_is_as_of_and_a_later_restatement_never_refits_predictions(
    history, backend_name
):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    revision = {
        "origin_period": ORIGINS[2],
        "dev_lag": 48,
        "eval_date": dt.date(2018, 12, 31),
        "field": "paid_loss",
        "value": 3000.0,
    }
    changed = from_frame(
        pd.concat([tri.execute(), pd.DataFrame([revision])], ignore_index=True), backend_name
    )
    original = evaluate_conventional(selected, tri, as_of="2017-12-31")
    before_revision = evaluate_conventional(selected, changed, as_of="2017-12-31")
    after_revision = evaluate_conventional(selected, changed, as_of="2018-12-31")
    pd.testing.assert_frame_equal(original.origins, before_revision.origins)
    np.testing.assert_allclose(original.origins["predicted"], after_revision.origins["predicted"])
    row = after_revision.origins.set_index("origin_period").loc[ORIGINS[2]]
    assert row["observed"] == pytest.approx(3000.0)


def test_outer_evaluation_excludes_origins_not_present_when_selected(history, backend_name):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    new_origin = dt.date(2015, 1, 1)
    extra = pd.DataFrame(
        [
            {
                "origin_period": new_origin,
                "dev_lag": lag,
                "eval_date": dt.date(2014 + lag // 12, 12, 31),
                "field": "paid_loss",
                "value": amount,
            }
            for lag, amount in [(12, 600.0), (24, 1200.0), (36, 1800.0), (48, 2400.0)]
        ]
    )
    future = from_frame(pd.concat([tri.execute(), extra], ignore_index=True), backend_name)
    evaluation = evaluate_conventional(selected, future, as_of="2018-12-31")
    assert new_origin not in evaluation.origins["origin_period"].tolist()
    assert evaluation.summary["n_targets"] == 3


def test_outer_evaluation_date_must_follow_selection(history):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    with pytest.raises(ValueError, match="(?i)(date|after|later|selection)"):
        evaluate_conventional(selected, tri, as_of=selected.as_of)


def test_outer_evaluation_refuses_another_books_losses_under_the_same_shape(history, backend_name):
    # Same origins, same grains, same field, no segment columns, every loss
    # doubled. Nothing else checked here reads an amount, so this used to report
    # full coverage, an empty reason and a finite RMSE against another book.
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    frame = tri.execute()
    frame.loc[frame["field"] == "paid_loss", "value"] *= 2.0
    doubled = from_frame(frame, backend_name)
    with pytest.raises(ValueError, match="(?i)(history|estimated on)"):
        evaluate_conventional(selected, doubled, as_of="2017-12-31")


def test_outer_evaluation_refuses_one_changed_interior_cell_and_names_it(history, backend_name):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    frame = tri.execute()
    interior = (
        (pd.to_datetime(frame["origin_period"]).dt.date == ORIGINS[1])
        & (frame["dev_lag"] == 24)
        & (frame["field"] == "paid_loss")
    )
    assert interior.sum() == 1
    frame.loc[interior, "value"] = 410.0  # the fit was estimated from 400
    changed = from_frame(frame, backend_name)
    with pytest.raises(ValueError, match="(?i)(history|estimated on)") as raised:
        evaluate_conventional(selected, changed, as_of="2017-12-31")
    message = str(raised.value)
    assert str(ORIGINS[1]) in message
    assert "410" in message and "400" in message


def test_outer_evaluation_refuses_a_triangle_that_carries_no_history_at_selection(
    history, backend_name
):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    frame = tri.execute()
    later_only = frame.loc[pd.to_datetime(frame["eval_date"]).dt.date > selected.as_of]
    assert not later_only.empty
    with pytest.raises(ValueError, match="(?i)(no such observation|different loss histories)"):
        evaluate_conventional(selected, from_frame(later_only, backend_name), as_of="2017-12-31")


def test_outer_evaluation_refuses_a_history_it_cannot_read_a_single_cell_of(history, backend_name):
    # Every loss row recorded twice at one evaluation date. A restatement would
    # be collapsed by as_of, but two readings at one date are not a restatement
    # and there is no way to say which one the fit saw, so no cell is
    # comparable - which must be a refusal rather than a pass for free.
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    frame = tri.execute()
    twice = pd.concat([frame, frame.loc[frame["field"] == "paid_loss"]], ignore_index=True)
    with pytest.raises(ValueError, match="(?i)(shares no|no way to confirm)"):
        evaluate_conventional(selected, from_frame(twice, backend_name), as_of="2017-12-31")


def test_outer_evaluation_needs_an_explicitly_declared_horizon(history):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    implicit = replace(selected, candidate=replace(selected.candidate, horizon=None))
    with pytest.raises(ValueError, match="(?i)horizon"):
        evaluate_conventional(implicit, tri, as_of="2017-12-31")


@pytest.mark.parametrize("later_restatement", [False, True])
def test_outer_history_cannot_reclassify_an_already_available_terminal_as_unseen(
    history, backend_name, later_restatement
):
    tri, replay = history
    selected = choose(replay, date=DATES[1])
    frame = tri.execute()
    terminal = (
        (pd.to_datetime(frame["origin_period"]).dt.date == ORIGINS[2])
        & (frame["dev_lag"] == 48)
        & (frame["field"] == "paid_loss")
    )
    # The selected fit had this origin only through age36. Evaluation data
    # that claim age48 was already available then describe a different history.
    frame.loc[terminal, "eval_date"] = pd.Timestamp(DATES[1])
    if later_restatement:
        # Looking only at the terminal row's newest eval_date would miss the
        # older observation that proves this target was known at selection.
        revision = frame.loc[terminal].copy()
        revision["eval_date"] = pd.Timestamp("2016-12-31")
        revision["value"] = 1300.0
        frame = pd.concat([frame, revision], ignore_index=True)
    incompatible = from_frame(frame, backend_name)
    with pytest.raises(ValueError, match="(?i)(history|available|selection)"):
        evaluate_conventional(selected, incompatible, as_of="2017-12-31")
