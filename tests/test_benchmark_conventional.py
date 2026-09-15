"""Bounded-memory benchmarking must preserve one-shot library decisions.

No published datasets, network access, or full benchmark runs are needed:
the small fully emerged fixture tests batching, score retention and evaluation.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.conventional import ConventionalCandidate
from ibnr.kernels.replay import replay_conventional
from ibnr.kernels.selection import evaluate_conventional, select_conventional

from .test_conventional_replay import DATES, ORIGINS, full_triangle

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from benchmark_conventional import named_grid, select_batched  # noqa: E402


def direct_decisions(triangle, candidates):
    replay = replay_conventional(triangle, candidates, DATES, on_error="record")
    return {
        metric: select_conventional(replay, selection_as_of=DATES[-1], metric=metric)
        for metric in ("ave", "cdr")
    }


def ordered_scores(decision):
    return decision.scores.sort_values(["candidate", "as_of", "eval_date"]).reset_index(drop=True)


def assert_same_decisions(expected, actual, triangle):
    assert set(actual) == set(expected) == {"ave", "cdr"}
    for metric in expected:
        want, got = expected[metric], actual[metric]
        assert got.name == want.name
        assert got.candidate == want.candidate
        assert got.as_of == want.as_of
        assert got.metric == want.metric
        pd.testing.assert_frame_equal(got.ranking, want.ranking)
        pd.testing.assert_frame_equal(ordered_scores(got), ordered_scores(want))
        np.testing.assert_array_equal(got.fit.factors, want.fit.factors)
        pd.testing.assert_frame_equal(got.fit.origins, want.fit.origins)
        evaluated_want = evaluate_conventional(want, triangle, as_of="2017-12-31")
        evaluated_got = evaluate_conventional(got, triangle, as_of="2017-12-31")
        pd.testing.assert_frame_equal(evaluated_got.origins, evaluated_want.origins)
        assert evaluated_got.summary == evaluated_want.summary


@pytest.mark.parametrize("batch_size", [1, 2, 3, 64])
def test_batching_preserves_both_winners_global_rankings_scores_and_evaluations(
    backend_name, batch_size
):
    triangle = full_triangle(backend_name)
    candidates = {
        "volume": ConventionalCandidate(horizon=48),
        "bf": ConventionalCandidate(method="bf", horizon=48, expected_loss_ratio=0.6),
        "gcc": ConventionalCandidate(method="gcc", horizon=48, decay=0.5),
        "simple_recent": ConventionalCandidate(horizon=48, average="simple", history_periods=2),
        "median_recent": ConventionalCandidate(horizon=48, average="median", history_periods=1),
    }
    expected = direct_decisions(triangle, candidates)
    actual = select_batched(
        triangle, candidates, DATES, loss_field="paid_loss", batch_size=batch_size
    )
    assert_same_decisions(expected, actual, triangle)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_lexical_ties_are_global_and_independent_of_batch_boundaries(backend_name, batch_size):
    triangle = full_triangle(backend_name)
    candidate = ConventionalCandidate(horizon=48)
    candidates = dict.fromkeys(["zeta", "mu", "alpha", "beta"], candidate)
    expected = direct_decisions(triangle, candidates)
    actual = select_batched(
        triangle, candidates, DATES, loss_field="paid_loss", batch_size=batch_size
    )
    assert_same_decisions(expected, actual, triangle)
    assert all(decision.name == "alpha" for decision in actual.values())


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_an_entire_failed_chunk_remains_in_global_rankings_and_scores(backend_name, batch_size):
    triangle = full_triangle(backend_name)
    failed = ConventionalCandidate(horizon=48, exclude=tuple((origin, 12) for origin in ORIGINS))
    candidates = {
        "failed_a": failed,
        "failed_b": failed,
        "valid_cl": ConventionalCandidate(horizon=48),
        "valid_gcc": ConventionalCandidate(method="gcc", horizon=48, decay=0.5),
    }
    expected = direct_decisions(triangle, candidates)
    actual = select_batched(
        triangle, candidates, DATES, loss_field="paid_loss", batch_size=batch_size
    )
    assert_same_decisions(expected, actual, triangle)
    for decision in actual.values():
        failed_rows = decision.ranking.set_index("candidate").loc[["failed_a", "failed_b"]]
        assert not failed_rows["eligible"].any()
        assert failed_rows["mean_rmse"].isna().all()
        assert failed_rows["n_required"].eq(2).all()
        assert failed_rows["reason"].str.len().gt(0).all()
        failed_scores = decision.scores.loc[
            decision.scores["candidate"].isin(["failed_a", "failed_b"])
        ]
        assert len(failed_scores) == 4
        assert failed_scores["status"].eq("failed").all()


def test_all_failed_chunks_raise_instead_of_returning_a_partial_winner(backend_name):
    triangle = full_triangle(backend_name)
    failed = ConventionalCandidate(horizon=48, exclude=tuple((origin, 12) for origin in ORIGINS))
    with pytest.raises(ValueError, match="no eligible candidate"):
        select_batched(triangle, {"failed": failed}, DATES, loss_field="paid_loss", batch_size=1)


@pytest.mark.parametrize(
    "windows,ratios,horizon,expected_count,method_counts",
    [
        (
            tuple(range(10, 20)),
            tuple(i / 100 for i in range(50, 71)),
            240,
            1720,
            {"cl": 40, "bf": 840, "gcc": 840},
        ),
        (
            tuple(range(5, 22)),
            tuple(i / 100 for i in range(40, 61)),
            63,
            2924,
            {"cl": 68, "bf": 1428, "gcc": 1428},
        ),
    ],
)
def test_literal_published_grid_counts_have_distinct_complete_settings(
    windows, ratios, horizon, expected_count, method_counts
):
    decays = tuple(i / 20 for i in range(21))
    grid = named_grid(
        history_periods=windows,
        drop_high=(False, True),
        drop_low=(False, True),
        expected_loss_ratios=ratios,
        decays=decays,
        horizon=horizon,
        unsupported_factor="unity",
        exhausted_exclusions="keep",
    )
    assert len(grid) == expected_count
    assert len(set(grid.values())) == expected_count
    assert Counter(c.method for c in grid.values()) == method_counts
    for window in windows:
        for high in (False, True):
            for low in (False, True):
                settings = [
                    c
                    for c in grid.values()
                    if c.history_periods == window and c.drop_high == high and c.drop_low == low
                ]
                assert len(settings) == 43
                assert {c.expected_loss_ratio for c in settings if c.method == "bf"} == set(ratios)
                assert {c.decay for c in settings if c.method == "gcc"} == set(decays)
                assert all(c.horizon == horizon for c in settings)
    # Names themselves remain sufficient to identify the parameter choices;
    # generating the same ranges again must not reorder or rename candidates.
    repeated = named_grid(
        history_periods=windows,
        drop_high=(False, True),
        drop_low=(False, True),
        expected_loss_ratios=ratios,
        decays=decays,
        horizon=horizon,
        unsupported_factor="unity",
        exhausted_exclusions="keep",
    )
    assert list(grid.items()) == list(repeated.items())
