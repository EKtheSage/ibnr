"""Behavioral checks for conventional point estimators, on both data backends.

The references here are hand calculations, not copies of the implementation.
In particular GCC's two endpoints are different on the heterogeneous exposures,
and changing factor selection never borrows Mack's variance formulas.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional
from ibnr.kernels.mack import fit_mack

from .conftest import make_cohort_triangle


@pytest.mark.parametrize("origin", [2020, None, pd.NaT, float("nan")])
def test_exclusion_rejects_invalid_dates(origin):
    with pytest.raises(ValueError, match="date"):
        ConventionalCandidate(exclude=((origin, 12),))


@pytest.mark.parametrize("lag", [48, 60])
def test_exclusion_must_identify_a_link_within_fixed_horizon(lag):
    with pytest.raises(ValueError, match="horizon"):
        ConventionalCandidate(horizon=48, exclude=(("2010-01-01", lag),))


AS_OF = "2013-12-31"
ORIGINS = [dt.date(2010 + i, 1, 1) for i in range(4)]
PREMIUM = np.array([1000.0, 1000.0, 2000.0, 2000.0])
SMALL = np.array(
    [
        [100.0, 200.0, 300.0, 360.0],
        [200.0, 400.0, 600.0, np.nan],
        [300.0, 600.0, np.nan, np.nan],
        [400.0, np.nan, np.nan, np.nan],
    ]
)
LATEST = np.array([360.0, 600.0, 600.0, 400.0])
BETA = np.array([1.0, 5 / 6, 5 / 9, 5 / 18])
CL_ULTIMATE = np.array([360.0, 720.0, 1080.0, 1440.0])


def candidate(**kwargs) -> ConventionalCandidate:
    return ConventionalCandidate(horizon=48, **kwargs)


def from_frame(frame, backend_name):
    frame = frame.copy()
    for column in ("origin_period", "eval_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.date
    return Triangle.from_long(frame, measure="cumulative", backend=backend_name)


def triangle(backend_name, *, values=SMALL, premium=None, segment=None) -> Triangle:
    tri = make_cohort_triangle(backend_name, values, segment=segment)
    if premium is None:
        return tri
    rows = tri.execute().to_dict("records")
    for origin, amount in zip(ORIGINS, premium, strict=True):
        rows.append(
            {
                "origin_period": origin,
                "dev_lag": 12,
                "eval_date": dt.date(origin.year, 12, 31),
                "field": "earned_premium",
                "value": amount,
                **(segment or {}),
            }
        )
    return from_frame(pd.DataFrame(rows), backend_name)


def fit(tri, spec=None, **kwargs):
    return fit_conventional(
        tri,
        candidate=candidate() if spec is None else spec,
        as_of=AS_OF,
        loss_field="paid_loss",
        **kwargs,
    )


def origin_table(fitted):
    return fitted.origins.sort_values("origin_period").reset_index(drop=True)


def selection(fitted, lag=12):
    return fitted.factor_selection.loc[fitted.factor_selection["from_dev_lag"] == lag].sort_values(
        "origin_period"
    )


def test_chain_ladder_factors_completion_and_reserve_by_hand(backend_name):
    fitted = fit(triangle(backend_name))
    table = origin_table(fitted)
    np.testing.assert_allclose(fitted.factors, [2.0, 1.5, 1.2])
    np.testing.assert_allclose(table["latest"], LATEST)
    np.testing.assert_allclose(table["beta"], BETA)
    np.testing.assert_allclose(table["ultimate"], CL_ULTIMATE)
    np.testing.assert_allclose(table["reserve"], CL_ULTIMATE - LATEST)
    assert table["latest_dev_lag"].tolist() == [48, 36, 24, 12]
    assert fitted.predict_cumulative(ORIGINS[-1], 24) == pytest.approx(800.0)
    assert fitted.predict_cumulative(ORIGINS[-1], 48) == pytest.approx(1440.0)


def test_default_point_estimate_matches_existing_mack_without_changing_it(backend_name):
    tri = triangle(backend_name)
    reference = fit_mack(tri, loss_field="paid_loss", as_of=AS_OF)
    fitted = fit(tri)
    np.testing.assert_allclose(fitted.factors, reference.f)
    np.testing.assert_allclose(origin_table(fitted)["ultimate"], reference.ultimate)


def test_bf_combines_observed_losses_with_only_the_unemerged_prior(backend_name):
    fitted = fit(
        triangle(backend_name, premium=PREMIUM),
        candidate(method="bf", expected_loss_ratio=0.5),
        premium_field="earned_premium",
    )
    table = origin_table(fitted)
    expected = LATEST + (1 - BETA) * 0.5 * PREMIUM
    np.testing.assert_allclose(table["ultimate"], expected)
    np.testing.assert_allclose(table["expected_loss_ratio"], 0.5)
    assert table.loc[0, "reserve"] == 0.0
    # A BF projection earns the prior from the current development fraction,
    # rather than multiplying the observed latest value by a CL factor.
    assert fitted.predict_cumulative(ORIGINS[-1], 24) == pytest.approx(
        400 + (5 / 9 - 5 / 18) * 0.5 * 2000
    )


def test_gcc_zero_decay_recovers_chain_ladder(backend_name):
    fitted = fit(
        triangle(backend_name, premium=PREMIUM),
        candidate(method="gcc", decay=0.0),
        premium_field="earned_premium",
    )
    table = origin_table(fitted)
    np.testing.assert_allclose(table["ultimate"], CL_ULTIMATE)
    np.testing.assert_allclose(table["expected_loss_ratio"], [0.36, 0.72, 0.54, 0.72])


def test_gcc_unit_decay_recovers_one_cape_cod_loss_ratio(backend_name):
    fitted = fit(
        triangle(backend_name, premium=PREMIUM),
        candidate(method="gcc", decay=1.0),
        premium_field="earned_premium",
    )
    table = origin_table(fitted)
    # All origins contribute: 1960 paid / 3500 developed premium = 0.56.
    np.testing.assert_allclose(table["expected_loss_ratio"], 0.56)
    np.testing.assert_allclose(table["ultimate"], LATEST + (1 - BETA) * 0.56 * PREMIUM)
    assert not np.allclose(table["ultimate"], CL_ULTIMATE)


def test_gcc_interior_decay_weights_neighbours_in_both_directions(backend_name):
    fitted = fit(
        triangle(backend_name, premium=PREMIUM),
        candidate(method="gcc", decay=0.5),
        premium_field="earned_premium",
    )
    # For origin 2012 the four distances are 2, 1, 0, 1. This explicit
    # calculation detects one-sided histories and weighting ratios directly.
    expected_elr = (360 / 4 + 600 / 2 + 600 + 400 / 2) / (
        1000 / 4 + (1000 * 5 / 6) / 2 + 2000 * 5 / 9 + (2000 * 5 / 18) / 2
    )
    row = origin_table(fitted).iloc[2]
    assert row["expected_loss_ratio"] == pytest.approx(expected_elr)
    assert row["ultimate"] == pytest.approx(600 + (1 - 5 / 9) * 2000 * expected_elr)


def varied_triangle(backend_name):
    values = SMALL.copy()
    values[1, 1:3] = [600.0, 900.0]
    values[2, 1] = 2400.0
    # First-transition individual ratios are 2, 3, 8, on weights 100, 200, 300.
    return triangle(backend_name, values=values)


@pytest.mark.parametrize(
    "average,expected", [("volume", 16 / 3), ("simple", 13 / 3), ("median", 3)]
)
def test_averaging_uses_the_selected_ratios_and_correct_weights(backend_name, average, expected):
    fitted = fit(varied_triangle(backend_name), candidate(average=average))
    assert fitted.factors[0] == pytest.approx(expected)


def test_history_window_uses_the_recent_pairs_at_each_age(backend_name):
    fitted = fit(varied_triangle(backend_name), candidate(history_periods=2))
    # The newest origin has no successor at 12 months and is NOT a pair.
    # Select the two most recent actual pairs, from 2011 and 2012.
    assert fitted.factors[0] == pytest.approx((600 + 2400) / (200 + 300))
    chosen = selection(fitted).query("included")["origin_period"].tolist()
    assert chosen == ORIGINS[1:3]


@pytest.mark.parametrize("average,expected", [("volume", 8 / 3), ("simple", 2.5), ("median", 2.5)])
def test_undefined_ratios_are_removed_before_the_history_window(backend_name, average, expected):
    values = SMALL.copy()
    values[1, 1:3] = [600.0, 900.0]
    values[2, :2] = [0.0, 2400.0]
    fitted = fit(
        triangle(backend_name, values=values),
        candidate(history_periods=2, average=average),
    )
    # The most recent pair has no defined link ratio. It must not consume
    # a slot in the two-observation window, on any averaging method.
    assert fitted.factors[0] == pytest.approx(expected)
    assert selection(fitted).query("included")["origin_period"].tolist() == ORIGINS[:2]
    assert origin_table(fitted).loc[2, "latest"] == 2400.0


def test_gcc_elr_uses_all_valuation_known_origins_not_the_factor_history_window(backend_name):
    values = SMALL.copy()
    values[1, 1:3] = [600.0, 900.0]
    values[2, 1] = 2400.0
    fitted = fit(
        triangle(backend_name, values=values, premium=PREMIUM),
        candidate(method="gcc", decay=1.0, history_periods=1),
        premium_field="earned_premium",
    )
    # The newest eligible link ratio gives f12=8. GCC still pools all four
    # origins for its ELR, whose developed exposure is worked out here.
    expected_elr = (360 + 900 + 2400 + 400) / (1000 + 1000 * 5 / 6 + 2000 * 5 / 9 + 2000 * 5 / 72)
    np.testing.assert_allclose(fitted.factors, [8.0, 1.5, 1.2])
    np.testing.assert_allclose(origin_table(fitted)["expected_loss_ratio"], expected_elr)


@pytest.mark.parametrize(
    "drop_high,drop_low,expected,kept",
    [(True, False, 8 / 3, [0, 1]), (False, True, 6.0, [1, 2]), (True, True, 3.0, [1])],
)
def test_high_low_trimming_changes_the_factors_and_discloses_pairs(
    backend_name, drop_high, drop_low, expected, kept
):
    fitted = fit(
        varied_triangle(backend_name),
        candidate(drop_high=drop_high, drop_low=drop_low, exhausted_exclusions="keep"),
    )
    assert fitted.factors[0] == pytest.approx(expected)
    chosen = selection(fitted).query("included")["origin_period"].tolist()
    assert chosen == [ORIGINS[i] for i in kept]


@pytest.mark.parametrize("side,kept", [("low", [1, 2]), ("high", [0, 1])])
def test_tied_extremes_remove_one_pair_reproducibly(backend_name, side, kept):
    tri = triangle(backend_name)  # all three first-transition ratios equal two
    spec = candidate(**{f"drop_{side}": True}, exhausted_exclusions="keep")
    first, second = fit(tri, spec), fit(tri, spec)
    retained = selection(first).query("included")
    assert len(retained) == 2  # never drop every ratio equal to the minimum
    assert retained["origin_period"].tolist() == [ORIGINS[i] for i in kept]
    assert first.factors[0] == pytest.approx(2.0)
    pd.testing.assert_frame_equal(selection(first), selection(second))


def test_keep_policy_skips_the_whole_trim_when_high_and_low_would_exhaust(backend_name):
    fitted = fit(
        varied_triangle(backend_name),
        candidate(drop_high=True, drop_low=True, exhausted_exclusions="keep"),
    )
    # At 24 months there are two pairs; at 36 months just one. Both retain
    # their complete pre-trimming sample when the requested trim cannot run.
    assert selection(fitted, 24)["included"].sum() == 2
    assert selection(fitted, 36)["included"].sum() == 1
    np.testing.assert_allclose(fitted.factors[1:], [1.5, 1.2])


def test_raise_policy_names_an_exhausted_factor_selection(backend_name):
    with pytest.raises(ValueError, match="(?i)(exclu|trim|remaining|exhaust)"):
        fit(
            varied_triangle(backend_name),
            candidate(drop_high=True, drop_low=True, exhausted_exclusions="raise"),
        )


def test_explicit_exclusion_names_a_pair_without_removing_its_observed_losses(backend_name):
    fitted = fit(
        varied_triangle(backend_name),
        candidate(exclude=((ORIGINS[2], 12),)),
    )
    assert fitted.factors[0] == pytest.approx(8 / 3)
    assert selection(fitted).query("included")["origin_period"].tolist() == ORIGINS[:2]
    # Excluding a development-factor observation is not deleting that origin
    # or changing the paid-to-date balance from which it must be projected.
    assert origin_table(fitted).loc[2, "latest"] == 2400.0


def test_explicit_exclusion_precedes_automatic_extreme_selection(backend_name):
    fitted = fit(
        varied_triangle(backend_name),
        candidate(exclude=((ORIGINS[2], 12),), drop_high=True, exhausted_exclusions="keep"),
    )
    # Explicitly remove ratio 8; the highest remaining ratio is now 3.
    assert fitted.factors[0] == pytest.approx(2.0)
    assert selection(fitted).query("included")["origin_period"].tolist() == ORIGINS[:1]


@pytest.mark.parametrize(
    "method,extra", [("bf", {"expected_loss_ratio": 0.5}), ("gcc", {"decay": 0.5})]
)
def test_exposure_based_methods_refuse_missing_premium(backend_name, method, extra):
    with pytest.raises(ValueError, match="(?i)premium"):
        fit(
            triangle(backend_name),
            candidate(method=method, **extra),
            premium_field="earned_premium",
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_exposure_based_methods_refuse_unusable_premium(backend_name, bad):
    premium = PREMIUM.copy()
    premium[-1] = bad
    with pytest.raises(ValueError, match="(?i)(premium|finite)"):
        fit(
            triangle(backend_name, premium=premium),
            candidate(method="bf", expected_loss_ratio=0.5),
            premium_field="earned_premium",
        )


def test_premium_from_another_cohort_is_refused(backend_name):
    tri = triangle(backend_name, premium=PREMIUM, segment={"company": "one"})
    frame = tri.execute()
    frame.loc[frame["field"] == "earned_premium", "company"] = "two"
    wrong = from_frame(frame, backend_name)
    with pytest.raises(ValueError, match="(?i)(premium|cohort|segment)"):
        fit(
            wrong,
            candidate(method="bf", expected_loss_ratio=0.5),
            premium_field="earned_premium",
        )


def test_future_loss_observations_and_premium_revisions_do_not_change_a_past_fit(backend_name):
    tri = triangle(backend_name, premium=PREMIUM)
    frame = tri.execute()
    extra = []
    # Later observations include both a future loss and a revised old premium.
    for field, lag, amount in [("paid_loss", 24, 10000.0), ("earned_premium", 12, 99999.0)]:
        extra.append(
            {
                "origin_period": ORIGINS[-1],
                "dev_lag": lag,
                "eval_date": dt.date(2014, 12, 31),
                "field": field,
                "value": amount,
            }
        )
    full = from_frame(
        pd.concat([frame, pd.DataFrame(extra)], ignore_index=True),
        backend_name,
    )
    spec = candidate(method="bf", expected_loss_ratio=0.5)
    before = fit(tri, spec, premium_field="earned_premium")
    after = fit(full, spec, premium_field="earned_premium")
    np.testing.assert_allclose(before.factors, after.factors)
    pd.testing.assert_frame_equal(origin_table(before), origin_table(after))


def test_missing_future_development_factor_requires_an_explicit_policy(backend_name):
    spec = replace(candidate(), horizon=60, unsupported_factor="raise")
    with pytest.raises(ValueError, match="(?i)(factor|support|observation)"):
        fit(triangle(backend_name), spec)
    extended = fit(triangle(backend_name), replace(spec, unsupported_factor="unity"))
    np.testing.assert_allclose(extended.factors, [2.0, 1.5, 1.2, 1.0])
    np.testing.assert_allclose(origin_table(extended)["ultimate"], CL_ULTIMATE)
    assert extended.predict_cumulative(ORIGINS[-1], 60) == pytest.approx(1440.0)


def test_candidate_is_immutable_and_the_fit_retains_it(backend_name):
    spec = candidate(history_periods=2, average="simple")
    fitted = fit(triangle(backend_name), spec)
    assert fitted.candidate == spec
    with pytest.raises(FrozenInstanceError):
        spec.history_periods = 1


def test_an_unrelated_origin_cannot_be_projected_as_this_cohort(backend_name):
    fitted = fit(triangle(backend_name))
    with pytest.raises((ValueError, KeyError), match="(?i)origin"):
        fitted.predict_cumulative(dt.date(2099, 1, 1), 24)
