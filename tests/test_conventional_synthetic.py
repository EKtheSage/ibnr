"""Synthetic benchmarks expose honest historical slices and reproducible noise."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.contract import cohort_grid
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from conventional_synthetic import (  # noqa: E402
    EVALUATION_DATE,
    HORIZON,
    N_DEVELOPMENT,
    N_ORIGINS,
    REPLAY_DATES,
    SCENARIOS,
    SELECTION_DATE,
    synthetic_portfolio,
)


def frame(triangle):
    out = triangle.execute().copy()
    for col in ("origin_period", "eval_date"):
        out[col] = pd.to_datetime(out[col]).dt.date
    return out.sort_values(["field", "origin_period", "dev_lag"]).reset_index(drop=True)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_same_seed_and_scenario_reproduce_the_complete_portfolio(backend_name, scenario):
    first = synthetic_portfolio(12, scenario, backend_name)
    second = synthetic_portfolio(12, scenario, backend_name)
    pd.testing.assert_frame_equal(frame(first), frame(second))


def test_different_seeds_change_losses_and_exposures(backend_name):
    first = frame(synthetic_portfolio(12, "stable", backend_name))
    second = frame(synthetic_portfolio(13, "stable", backend_name))
    for field in ("paid_loss", "earned_premium"):
        a = first.loc[first["field"] == field, "value"].to_numpy()
        b = second.loc[second["field"] == field, "value"].to_numpy()
        assert not np.array_equal(a, b)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_geometry_positive_premium_and_stochastic_development(backend_name, scenario):
    tri = synthetic_portfolio(27, scenario, backend_name)
    data = frame(tri)
    paid = data.loc[data["field"] == "paid_loss"]
    premium = data.loc[data["field"] == "earned_premium"]
    assert len(paid) == N_ORIGINS * N_DEVELOPMENT
    assert len(premium) == N_ORIGINS
    assert premium["origin_period"].is_unique
    assert premium["dev_lag"].eq(12).all()
    assert premium["value"].gt(0).all()
    assert np.isfinite(data["value"]).all()
    assert min(paid["origin_period"]) == dt.date(2000, 1, 1)
    assert max(paid["origin_period"]) == dt.date(2023, 1, 1)
    assert max(paid["eval_date"]) == EVALUATION_DATE
    assert max(paid["dev_lag"]) == HORIZON == 96
    assert tri.meta.units == "USD"
    assert tri.meta.measure == "cumulative"
    assert tri.meta.origin_grain == tri.meta.dev_grain == "Y"
    assert all(
        o.year == e.year
        for o, e in zip(premium["origin_period"], premium["eval_date"], strict=True)
    )
    square = paid.pivot(index="origin_period", columns="dev_lag", values="value").to_numpy()
    increments = np.diff(square, axis=1, prepend=np.zeros((N_ORIGINS, 1)))
    assert (increments > 0).all()
    # There is no single noiseless LDF copied across every origin.
    link_ratios = square[:, 1] / square[:, 0]
    assert np.ptp(link_ratios) > 0.05


def test_replay_and_selection_slices_have_only_then_known_origins_and_premium(backend_name):
    tri = synthetic_portfolio(6, "stable", backend_name)
    for date in REPLAY_DATES:
        sliced = tri.as_of(date)
        grid = cohort_grid(sliced, loss_field="paid_loss", premium_field="earned_premium")
        assert len(grid["origin_periods"]) == date.year - 2000 + 1
        assert max(grid["origin_periods"]) == dt.date(date.year, 1, 1)
        assert len(grid["premium"]) == len(grid["origin_periods"])
        assert max(sliced.eval_dates) <= date
    selected = tri.as_of(SELECTION_DATE)
    assert len(selected.origins) == 21
    assert not any(origin.year >= 2021 for origin in selected.origins)
    truth = frame(tri.as_of(EVALUATION_DATE))
    terminal = truth.loc[(truth["field"] == "paid_loss") & (truth["dev_lag"] == HORIZON)]
    assert set(selected.origins) <= set(terminal["origin_period"])


def test_unanticipated_shock_changes_future_losses_but_not_selection_history(backend_name):
    stable = synthetic_portfolio(81, "stable", backend_name)
    shock = synthetic_portfolio(81, "shock", backend_name)
    pd.testing.assert_frame_equal(
        frame(stable.as_of(SELECTION_DATE)), frame(shock.as_of(SELECTION_DATE))
    )
    a, b = frame(stable), frame(shock)
    prem = a["field"] == "earned_premium"
    np.testing.assert_array_equal(a.loc[prem, "value"], b.loc[prem, "value"])
    old = a.loc[a["field"] == "paid_loss"].pivot(
        index="origin_period", columns="dev_lag", values="value"
    )
    new = b.loc[b["field"] == "paid_loss"].pivot(
        index="origin_period", columns="dev_lag", values="value"
    )
    old_increment = np.diff(old.to_numpy(), axis=1, prepend=np.zeros((N_ORIGINS, 1)))
    new_increment = np.diff(new.to_numpy(), axis=1, prepend=np.zeros((N_ORIGINS, 1)))
    calendar = np.array([o.year for o in old.index])[:, None] + np.arange(N_DEVELOPMENT)
    np.testing.assert_allclose(new_increment[calendar <= 2020], old_increment[calendar <= 2020])
    np.testing.assert_allclose(
        new_increment[calendar >= 2021], 1.4 * old_increment[calendar >= 2021]
    )


def test_scenarios_change_experience_without_changing_booked_premiums(backend_name):
    generated = {s: frame(synthetic_portfolio(81, s, backend_name)) for s in SCENARIOS}
    stable = generated["stable"]
    loss = stable["field"] == "paid_loss"
    prem = stable["field"] == "earned_premium"
    for scenario in ("noisy", "drift", "shock"):
        compared = generated[scenario]
        np.testing.assert_array_equal(stable.loc[prem, "value"], compared.loc[prem, "value"])
        assert not np.array_equal(stable.loc[loss, "value"], compared.loc[loss, "value"])
    recent_terminal = (
        loss & (stable["origin_period"] >= dt.date(2020, 1, 1)) & (stable["dev_lag"] == HORIZON)
    )
    assert (
        generated["drift"].loc[recent_terminal, "value"] > stable.loc[recent_terminal, "value"]
    ).all()


def test_future_premiums_cannot_change_the_selected_date_fit(backend_name):
    tri = synthetic_portfolio(6, "stable", backend_name)
    data = frame(tri)
    future_prem = (data["field"] == "earned_premium") & (data["eval_date"] > SELECTION_DATE)
    assert future_prem.sum() == 3
    data.loc[future_prem, "value"] *= 1000
    changed = Triangle.from_long(data, measure="cumulative", units="USD", backend=backend_name)
    candidate = ConventionalCandidate(method="gcc", decay=0.5, horizon=HORIZON)
    original = fit_conventional(tri, candidate, as_of=SELECTION_DATE)
    revised = fit_conventional(changed, candidate, as_of=SELECTION_DATE)
    np.testing.assert_array_equal(original.grid["premium"], revised.grid["premium"])
    pd.testing.assert_frame_equal(original.origins, revised.origins)


def test_duckdb_and_polars_return_the_same_seeded_portfolio(backend_name):
    expected = frame(synthetic_portfolio(33, "drift", "duckdb"))
    actual = frame(synthetic_portfolio(33, "drift", backend_name))
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("seed", [-1, 1.5, True, None])
def test_invalid_seed_is_refused(seed):
    with pytest.raises(ValueError, match="seed"):
        synthetic_portfolio(seed, "stable")


def test_unknown_scenario_is_refused():
    with pytest.raises(ValueError, match="scenario"):
        synthetic_portfolio(1, "winner_only")
