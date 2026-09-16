"""Through-time conventional forecasts, checked against actual books and refits.

Positive AvE and CDR mean adverse development. Their difference is the change
to the remaining reserve after the newly observed increment. The references
below use independent fits and explicit hand calculations on a small rectangle.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional
from ibnr.kernels.replay import replay_conventional

from .conftest import make_cohort_triangle

ORIGINS = [dt.date(2010 + i, 1, 1) for i in range(5)]
DATES = [dt.date(year, 12, 31) for year in (2013, 2014, 2015)]
VALUES = np.array(
    [
        [100.0, 200.0, 300.0, 360.0],
        [200.0, 400.0, 640.0, 800.0],
        [300.0, 660.0, 990.0, 1280.0],
        [400.0, 840.0, 1344.0, 1700.0],
        [500.0, 1150.0, 1840.0, 2350.0],
    ]
)
PREMIUM = [1000.0, 1200.0, 1800.0, 2200.0, 3000.0]


def from_frame(frame, backend_name):
    frame = frame.copy()
    for col in ("origin_period", "eval_date"):
        frame[col] = pd.to_datetime(frame[col]).dt.date
    return Triangle.from_long(frame, measure="cumulative", backend=backend_name)


def full_triangle(backend_name, *, values=VALUES):
    tri = make_cohort_triangle(backend_name, values)
    frame = tri.execute()
    rows = [
        {
            "origin_period": origin,
            "dev_lag": 12,
            "eval_date": dt.date(origin.year, 12, 31),
            "field": "earned_premium",
            "value": amount,
        }
        for origin, amount in zip(ORIGINS, PREMIUM, strict=True)
    ]
    return from_frame(pd.concat([frame, pd.DataFrame(rows)], ignore_index=True), backend_name)


def specs():
    return {
        "cl": ConventionalCandidate(horizon=48),
        "bf": ConventionalCandidate(method="bf", horizon=48, expected_loss_ratio=0.6),
        "gcc": ConventionalCandidate(method="gcc", horizon=48, decay=0.5),
    }


def replay(tri, candidates=None, dates=DATES, **kwargs):
    return replay_conventional(
        tri,
        candidates=specs() if candidates is None else candidates,
        dates=dates,
        loss_field="paid_loss",
        premium_field="earned_premium",
        **kwargs,
    )


def sorted_cells(result):
    return result.cells.sort_values(["candidate", "as_of", "origin_period"]).reset_index(drop=True)


def cell(result, *, origin, name="cl", date=DATES[0]):
    rows = result.cells.loc[
        (result.cells["candidate"] == name)
        & (result.cells["origin_period"] == origin)
        & (result.cells["as_of"] == date)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_cl_first_interval_hand_calculation_and_adverse_sign(backend_name):
    result = replay(full_triangle(backend_name), {"cl": specs()["cl"]}, dates=DATES[:2])
    closing = cell(result, origin=ORIGINS[1])
    # At 2013: f36=360/300=1.2, latest=640, ultimate=768.
    # At 2014: ultimate is observed at 800, so adverse CDR=32.
    assert closing["actual_increment"] == pytest.approx(160.0)
    assert closing["expected_increment"] == pytest.approx(128.0)
    assert closing["ave"] == pytest.approx(32.0)
    assert closing["old_ultimate"] == pytest.approx(768.0)
    assert closing["new_ultimate"] == pytest.approx(800.0)
    assert closing["cdr"] == pytest.approx(32.0)
    assert closing["remaining_revision"] == pytest.approx(0.0)

    continuing = cell(result, origin=ORIGINS[2])
    # At 2013: f24=(300+640)/(200+400)=47/30; expected next=374.
    # Actual next=990-660=330. The following-age factor changes to 58/47.
    assert continuing["actual_increment"] == pytest.approx(330.0)
    assert continuing["expected_increment"] == pytest.approx(374.0)
    assert continuing["ave"] == pytest.approx(-44.0)
    assert continuing["old_ultimate"] == pytest.approx(1240.8)
    assert continuing["new_ultimate"] == pytest.approx(990 * 58 / 47)
    assert continuing["cdr"] == pytest.approx(990 * 58 / 47 - 1240.8)


def test_bf_and_gcc_first_interval_hand_calculations(backend_name):
    """One BF and one GCC cell against numbers worked out away from the code.

    The comparison below against independent refits cannot see an error the fit
    and the replay share, and the CL hand calculation above does not touch the
    two exposure-based priors at all.
    """
    result = replay(full_triangle(backend_name), dates=DATES[:2])

    # BF(0.60), origin 2012, 2013 -> 2014.
    # At 12/2013 the paid triangle is 2010: 100 200 300 360, 2011: 200 400 640,
    # 2012: 300 660, 2013: 400, so the volume-weighted factors are
    #   f12 = 1260/600 = 21/10, f24 = 940/600 = 47/30, f36 = 360/300 = 6/5
    # and the developed fractions, read back from the 48-month end, are
    #   beta = (250/987, 25/47, 5/6, 1).
    # Origin 2012 is two periods old, so it is 25/47 developed on a fixed prior
    # of 0.60 * 1800 = 1080:
    #   ultimate      = 660 + 1080 * 22/47              = 54780/47 = 1165.53...
    #   36-month forecast = 660 + 1080 * (5/6 - 25/47)  = 46320/47 =  985.53...
    #   expected increment = 1080 * 85/282              = 15300/47 =  325.53...
    # The cell actually developed 990 - 660 = 330, so AvE = 330 - 15300/47.
    # At 12/2014, f36 becomes 1160/940 = 58/47, the prior is still 1080 and
    # 2012 is 47/58 developed:
    #   ultimate = 990 + 1080 * 11/58 = 34650/29 = 1194.83...
    #   CDR = 34650/29 - 54780/47 = 39930/1363, and the remaining revision is
    #   1080 * 11/58 - (54780/47 - 46320/47) = 720/29.
    bf = cell(result, origin=ORIGINS[2], name="bf")
    assert bf["actual_increment"] == pytest.approx(330.0)
    assert bf["expected_increment"] == pytest.approx(15300 / 47)
    assert bf["ave"] == pytest.approx(210 / 47)
    assert bf["old_ultimate"] == pytest.approx(54780 / 47)
    assert bf["new_ultimate"] == pytest.approx(34650 / 29)
    assert bf["cdr"] == pytest.approx(39930 / 1363)
    assert bf["remaining_revision"] == pytest.approx(720 / 29)

    # GCC(decay 0.5), origin 2012, the same interval. Same factors and beta;
    # what changes is the prior, which GCC estimates from a distance-weighted
    # loss ratio. The weights halve per origin of distance, so for 2012 they are
    # (1/4, 1/2, 1, 1/2) over the four origins known at 12/2013:
    #   losses   = .25*360 + .5*640 + 660 + .5*400                  = 1270
    #   exposure = .25*1000 + .5*1200*(5/6) + 1800*(25/47)
    #              + .5*2200*(250/987)                  = 750 + 1220000/987
    #   loss ratio = 1270 / (750 + 1220000/987) = 1253490/1960250 = 0.63945...
    #   prior      = 1800 * that                                   = 1151.017...
    # so the ultimate is 660 + prior*22/47 = 1198.774..., the 36-month forecast
    # is 660 + prior*85/282 = 1006.938... and AvE is 330 - 346.938...
    # At 12/2014 the same arithmetic over five origins, with weights
    # (1/4, 1/2, 1, 1/2, 1/4) and losses .25*360 + .5*800 + 990 + .5*840
    # + .25*500 = 2025, gives a prior of 1183.634..., and 2012 is 47/58
    # developed, so the ultimate is 990 + prior*11/58 = 1214.482...
    gcc = cell(result, origin=ORIGINS[2], name="gcc")
    assert gcc["actual_increment"] == pytest.approx(330.0)
    assert gcc["expected_increment"] == pytest.approx(346.9378905751817)
    assert gcc["ave"] == pytest.approx(-16.937890575181736)
    assert gcc["old_ultimate"] == pytest.approx(1198.774135952047)
    assert gcc["new_ultimate"] == pytest.approx(1214.48240450918)
    assert gcc["cdr"] == pytest.approx(15.708268557133009)
    assert gcc["remaining_revision"] == pytest.approx(32.64615913231474)
    assert gcc["ave"] + gcc["remaining_revision"] == pytest.approx(gcc["cdr"])


def test_each_method_matches_independent_fits_and_exact_calendar_outcomes(backend_name):
    tri = full_triangle(backend_name)
    candidates = specs()
    result = replay(tri, candidates)
    assert result.errors.empty
    for name, spec in candidates.items():
        direct = {
            date: fit_conventional(tri, spec, as_of=date, premium_field="earned_premium")
            for date in DATES
        }
        for before, after in zip(DATES[:-1], DATES[1:], strict=True):
            old = direct[before]
            new = direct[after].origins.set_index("origin_period")
            for prior in old.origins.itertuples(index=False):
                row = cell(result, origin=prior.origin_period, name=name, date=before)
                age = min(int(prior.latest_dev_lag) + 12, 48)
                origin_i = ORIGINS.index(prior.origin_period)
                actual_next = VALUES[origin_i, age // 12 - 1]
                expected_next = old.predict_cumulative(prior.origin_period, age)
                actual_increment = actual_next - prior.latest
                expected_increment = expected_next - prior.latest
                new_ultimate = float(new.loc[prior.origin_period, "ultimate"])
                assert row["eval_date"] == after
                assert row["actual_increment"] == pytest.approx(actual_increment)
                assert row["expected_increment"] == pytest.approx(expected_increment)
                assert row["ave"] == pytest.approx(actual_increment - expected_increment)
                assert row["old_ultimate"] == pytest.approx(prior.ultimate)
                assert row["new_ultimate"] == pytest.approx(new_ultimate)
                assert row["cdr"] == pytest.approx(new_ultimate - prior.ultimate)
                expected_remaining = prior.ultimate - expected_next
                actual_remaining = new_ultimate - actual_next
                assert row["remaining_revision"] == pytest.approx(
                    actual_remaining - expected_remaining
                )


def test_cdr_decomposes_and_opposing_components_can_cancel(backend_name):
    result = replay(full_triangle(backend_name))
    frame = result.cells
    np.testing.assert_allclose(frame["cdr"], frame["ave"] + frame["remaining_revision"], atol=1e-10)
    opposing = cell(result, origin=ORIGINS[2])
    assert opposing["ave"] < 0 < opposing["remaining_revision"]
    assert abs(opposing["cdr"]) < abs(opposing["ave"])
    # The same identity survives aggregation over the held-fixed old origins.
    total = frame.groupby(["candidate", "as_of"])[["ave", "cdr", "remaining_revision"]].sum()
    np.testing.assert_allclose(total["cdr"], total["ave"] + total["remaining_revision"], atol=1e-10)


def test_every_prior_origin_is_scored_but_new_origins_only_enter_the_refit(backend_name):
    result = replay(full_triangle(backend_name))
    for name in specs():
        first = result.cells.loc[
            (result.cells["candidate"] == name) & (result.cells["as_of"] == DATES[0])
        ]
        assert set(first["origin_period"]) == set(ORIGINS[:4])
        assert ORIGINS[4] in result.fits[(name, DATES[1])].origins["origin_period"].tolist()
        second = result.cells.loc[
            (result.cells["candidate"] == name) & (result.cells["as_of"] == DATES[1])
        ]
        assert set(second["origin_period"]) == set(ORIGINS)
    assert set(result.exclusions["origin_period"]) == {ORIGINS[4]}


def test_mature_origins_are_retained_as_zero_development_rows(backend_name):
    result = replay(full_triangle(backend_name), dates=DATES[:2])
    for name in specs():
        mature = cell(result, origin=ORIGINS[0], name=name)
        for column in (
            "actual_increment",
            "expected_increment",
            "ave",
            "cdr",
            "remaining_revision",
        ):
            assert mature[column] == pytest.approx(0.0)


def test_fitted_factors_change_while_the_procedure_settings_remain_fixed(backend_name):
    spec = ConventionalCandidate(horizon=48, history_periods=2, average="simple")
    result = replay(full_triangle(backend_name), {"fixed": spec})
    first = result.fits[("fixed", DATES[0])]
    second = result.fits[("fixed", DATES[1])]
    assert first.candidate == second.candidate == spec
    assert not np.allclose(first.factors, second.factors)
    assert set(result.fits) == {("fixed", date) for date in DATES}


def test_later_history_cannot_rewrite_an_earlier_interval(backend_name):
    original = full_triangle(backend_name)
    frame = original.execute()
    later = pd.to_datetime(frame["eval_date"]).dt.date > DATES[1]
    frame.loc[later & (frame["field"] == "paid_loss"), "value"] *= 3
    # A premium revision at the third cutoff is later than BOTH dates of the
    # interval being scored; no fit, forecast, or actual for that interval moves.
    revision = {
        "origin_period": ORIGINS[3],
        "dev_lag": 12,
        "eval_date": DATES[2],
        "field": "earned_premium",
        "value": 99999.0,
    }
    changed = from_frame(
        pd.concat([frame, pd.DataFrame([revision])], ignore_index=True), backend_name
    )
    before = replay(original, dates=DATES[:2])
    after = replay(changed, dates=DATES[:2])
    pd.testing.assert_frame_equal(sorted_cells(before), sorted_cells(after))


def test_booked_prior_anchor_survives_a_later_restatement(backend_name):
    tri = full_triangle(backend_name)
    revision = {
        "origin_period": ORIGINS[3],
        "dev_lag": 12,
        "eval_date": DATES[1],
        "field": "paid_loss",
        "value": 450.0,
    }
    changed = from_frame(
        pd.concat([tri.execute(), pd.DataFrame([revision])], ignore_index=True), backend_name
    )
    result = replay(changed, {"cl": specs()["cl"]}, dates=DATES[:2])
    row = cell(result, origin=ORIGINS[3])
    # The old books said 400. Reading the newly restated 450 as the old anchor
    # would erase 50 from measured experience, although no old forecast used it.
    assert row["actual_increment"] == pytest.approx(840 - 400)
    assert row["expected_increment"] == pytest.approx(440.0)


def test_a_mature_origins_terminal_restatement_is_measured_not_dropped(backend_name):
    tri = full_triangle(backend_name)
    revision = {
        "origin_period": ORIGINS[0],
        "dev_lag": 48,
        "eval_date": DATES[1],
        "field": "paid_loss",
        "value": 390.0,
    }
    changed = from_frame(
        pd.concat([tri.execute(), pd.DataFrame([revision])], ignore_index=True), backend_name
    )
    result = replay(changed, dates=DATES[:2])
    for name in specs():
        row = cell(result, origin=ORIGINS[0], name=name)
        assert row["actual_increment"] == pytest.approx(30.0)
        assert row["expected_increment"] == pytest.approx(0.0)
        assert row["ave"] == pytest.approx(30.0)
        assert row["cdr"] == pytest.approx(30.0)
        assert row["remaining_revision"] == pytest.approx(0.0)


def missing_next_cell(backend_name):
    tri = full_triangle(backend_name)
    frame = tri.execute()
    missing = (
        (pd.to_datetime(frame["origin_period"]).dt.date == ORIGINS[3])
        & (frame["dev_lag"] == 24)
        & (frame["field"] == "paid_loss")
    )
    return from_frame(frame.loc[~missing], backend_name)


def test_a_missing_next_observation_cannot_be_silently_scored_as_zero(backend_name):
    with pytest.raises(ValueError, match="(?i)(missing|observ|triangle|diagonal)"):
        replay(missing_next_cell(backend_name), dates=DATES[:2], on_error="raise")


def test_record_mode_retains_interval_failures_instead_of_a_partial_score(backend_name):
    result = replay(missing_next_cell(backend_name), dates=DATES[:2], on_error="record")
    assert result.cells.empty
    assert set(result.errors["candidate"]) == set(specs())
    assert set(result.errors["as_of"]) == {DATES[0]}
    assert set(result.errors["eval_date"]) == {DATES[1]}
    assert result.errors["reason"].str.len().gt(0).all()


def test_replay_refuses_a_moving_or_inconsistent_ultimate_horizon(backend_name):
    tri = full_triangle(backend_name)
    with pytest.raises(ValueError, match="(?i)horizon"):
        replay(tri, {"implicit": ConventionalCandidate()})
    with pytest.raises(ValueError, match="(?i)horizon"):
        replay(
            tri,
            {
                "48": specs()["cl"],
                "60": replace(specs()["cl"], horizon=60, unsupported_factor="unity"),
            },
        )


@pytest.mark.parametrize(
    "dates", [[DATES[1], DATES[0]], [DATES[0], DATES[0]], [DATES[0], DATES[2]]]
)
def test_replay_refuses_nonconsecutive_or_reversed_dates(backend_name, dates):
    with pytest.raises(ValueError, match="(?i)(date|interval|consecutive|step|grain)"):
        replay(full_triangle(backend_name), dates=dates)


def test_month_difference_alone_does_not_make_a_full_development_interval(backend_name):
    with pytest.raises(ValueError, match="(?i)(date|interval|consecutive|step|grain|month)"):
        replay(full_triangle(backend_name), dates=[DATES[0], dt.date(2014, 12, 1)])


def test_month_end_rollover_preserves_a_monthly_replay(backend_name):
    # Thirteen monthly origins, January 2009 through January 2010, developing
    # over four months. All three cutoff dates are month ends despite different
    # day numbers; strict day-of-month equality would incorrectly refuse them.
    values = (100 + 10 * np.arange(13))[:, None] * np.array([1.0, 2.0, 3.0, 3.6])[None, :]
    tri = make_cohort_triangle(backend_name, values, start_year=2009, dev_grain="M")
    dates = [dt.date(2010, 1, 31), dt.date(2010, 2, 28), dt.date(2010, 3, 31)]
    result = replay_conventional(
        tri, {"monthly": ConventionalCandidate(horizon=4)}, dates, loss_field="paid_loss"
    )
    assert result.errors.empty
    assert len(result.cells) == 26
    assert set(result.cells["as_of"]) == set(dates[:2])
    assert set(result.cells["eval_date"]) == set(dates[1:])
    np.testing.assert_allclose(result.cells["ave"], 0.0, atol=1e-10)
    np.testing.assert_allclose(result.cells["cdr"], 0.0, atol=1e-10)
