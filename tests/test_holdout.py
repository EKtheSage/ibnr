"""``kernels/holdout.py``: which cells the next diagonal actually holds out.

Every test here targets a way of getting this quietly wrong rather than a way of
crashing. The three that matter, because each passes a naive implementation on a
clean square triangle:

* ``D_next`` computed as ``as_of + 12 months`` instead of read from the data -
  correct on annual Schedule P, wrong on quarterly and wrong across a gap.
* a **restated** training cell counted as held out - it looks like a new
  observation and is really a correction to one the model trained on.
* exposure read from the post-cutoff slice, leaking a restated premium.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.holdout import next_diagonal
from ibnr.triangle.core import Triangle


def _rows(cum, *, start_year=2010, field="paid_loss", step=12, segment=None):
    """Long-format rows from a ``{(w, d): value}`` map. ``d`` is 1-based."""
    out = []
    for (w, d), value in cum.items():
        origin = dt.date(start_year + w - 1, 1, 1)
        dev_lag = step * d
        out.append(
            {
                **(segment or {}),
                "origin_period": origin,
                "dev_lag": dev_lag,
                "eval_date": dt.date(start_year + w - 1 + (d - 1) * step // 12, 12, 31),
                "field": field,
                "value": float(value),
            }
        )
    return out


def staircase(n_w, n_d, *, through=None, start_year=2010, field="paid_loss", segment=None):
    """Run-off staircase: origin ``w`` observed at devs 1..(``through`` - w + 1).

    ``through`` is the diagonal index; ``through=n_w`` is the usual square upper
    triangle. Cell value is ``100*w + d`` so every cell is identifiable on sight.
    """
    through = through or n_w
    cum = {
        (w, d): 100 * w + d
        for w in range(1, n_w + 1)
        for d in range(1, n_d + 1)
        if w + d - 1 <= through
    }
    return _rows(cum, start_year=start_year, field=field, segment=segment)


@pytest.fixture
def tri8(backend_name) -> Triangle:
    """8 origins, observed through diagonal 9 - so the diagonal AFTER the
    square 8x8 training triangle is fully present, including a 9th origin."""
    rows = staircase(9, 9, through=9)
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)


def test_eight_by_eight_scores_seven_and_names_both_exclusions(tri8):
    """The headline shape of the definition.

    Diagonal 9 holds 9 cells. Origin 1 reaches dev 9, deeper than training ever
    went; origin 9 appears for the first time. Both are structurally unscorable
    for a model with per-origin and per-dev parameters, so 7 are scored - and
    the 2 are reported, not dropped on the floor.
    """
    cells = next_diagonal(tri8, as_of="2017-12-31", fields="paid_loss")

    assert cells.n_cells == 7
    assert cells.eval_date == dt.date(2018, 12, 31)
    assert cells.exclusion_counts() == {
        "new_origin": 1,
        "dev_beyond_trained": 1,
        "no_predecessor": 0,
    }

    scored = cells.frame
    assert scored["dev_lag"].tolist() == [12 * d for d in range(8, 1, -1)]
    assert [o.year for o in scored["origin_period"]] == list(range(2011, 2018))
    # value is 100*w + d by construction, so this pins cell identity exactly
    assert scored["value"].tolist() == [100 * w + (10 - w) for w in range(2, 9)]

    excluded = cells.excluded.set_index("reason")
    assert excluded.loc["dev_beyond_trained", "dev_lag"] == 108
    assert excluded.loc["new_origin", "origin_period"].year == 2018


def test_next_eval_date_is_read_from_the_data_not_computed(backend_name):
    """``D_next`` must be the next eval_date PRESENT, not ``as_of`` + one grain.

    A triangle with a missing calendar year is the discriminating case: adding
    12 months lands on a date that does not exist, silently holding out nothing
    (or everything, depending on the comparison). Reading the data skips the gap.
    """
    rows = staircase(4, 4, through=4)
    # Skip the 2014 diagonal entirely. dev_lag must stay consistent with
    # eval_date: at 2015 the cell for origin w sits at dev index 2015-origin+1,
    # i.e. dev_lag 12*(7-w). Labelling these 12*(6-w) - as an earlier version of
    # this test did - describes cells at 2014 and quietly tests nothing.
    rows += [
        {
            "origin_period": dt.date(2010 + w - 1, 1, 1),
            "dev_lag": 12 * (7 - w),
            "eval_date": dt.date(2015, 12, 31),
            "field": "paid_loss",
            "value": float(100 * w + (7 - w)),
        }
        for w in range(2, 5)
    ]
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)

    cells = next_diagonal(tri, as_of="2013-12-31", fields="paid_loss")

    assert dt.date(2014, 12, 31) not in tri.eval_dates
    assert cells.eval_date == dt.date(2015, 12, 31), "skipped the gap year"

    # And the honest consequence, which is the point of using a REAL gap: with a
    # whole diagonal missing, every cell on the next observed one is two
    # development steps from its last training value, so no increment can be
    # formed and nothing is scorable. The module reports that with reasons
    # rather than differencing against the wrong cell.
    assert cells.n_cells == 0
    counts = cells.exclusion_counts()
    assert counts["no_predecessor"] == 2
    assert counts["dev_beyond_trained"] == 1
    assert not cells.frame["prev_value"].isna().any()  # vacuously - none survived


def test_quarterly_triangle_steps_one_quarter(backend_name):
    """The same guard on a grain where ``+12 months`` is wrong by 3 diagonals.

    Schedule P is annual, so an arithmetic ``D_next`` passes every other test in
    this file. Here it would skip three real diagonals.
    """
    cum = {(w, d): 100 * w + d for w in range(1, 5) for d in range(1, 5) if w + d - 1 <= 5}
    rows = []
    for (w, d), value in cum.items():
        origin = dt.date(2010 + w - 1, 1, 1)
        months = 3 * d
        end = dt.date(2010 + w - 1, 1, 1) + pd.DateOffset(months=months) - pd.Timedelta(days=1)
        rows.append(
            {
                "origin_period": origin,
                "dev_lag": months,
                "eval_date": end.date(),
                "field": "paid_loss",
                "value": float(value),
            }
        )
    tri = Triangle.from_long(
        pd.DataFrame(rows), measure="cumulative", dev_grain="Q", backend=backend_name
    )

    as_of = sorted(tri.eval_dates)[3]
    cells = next_diagonal(tri, as_of=as_of, fields="paid_loss")

    assert cells.eval_date == sorted(tri.eval_dates)[4]
    gap_months = (cells.eval_date.year - as_of.year) * 12 + cells.eval_date.month - as_of.month
    assert gap_months < 12, "a quarterly step must not advance a whole year"


def test_a_restated_training_cell_is_not_held_out(backend_name):
    """The subtlest failure in the module.

    A cell observed at D and RESTATED at D_next carries a new eval_date, so a
    filter like ``eval_date == D_next`` calls it held out. It is not: the model
    trained on that cell. Scoring it credits the model for predicting its own
    input, and it inflates the cell count on exactly the cohorts whose data is
    most revised.

    Anti-joining on the full key (origin, dev, field, segments) is what makes
    this right, because ``Triangle.as_of`` collapses restatements to the latest
    surviving one - so the restated cell appears in BOTH slices under one key.
    """
    rows = staircase(4, 4, through=5)
    restated = {
        "origin_period": dt.date(2010, 1, 1),
        "dev_lag": 24,  # observed at the 2011 diagonal, inside training
        "eval_date": dt.date(2014, 12, 31),  # ... and restated on the held-out one
        "field": "paid_loss",
        "value": 999.0,
    }
    tri = Triangle.from_long(
        pd.DataFrame([*rows, restated]), measure="cumulative", backend=backend_name
    )

    cells = next_diagonal(tri, as_of="2013-12-31", fields="paid_loss")

    key = list(zip(cells.frame["origin_period"], cells.frame["dev_lag"], strict=True))
    assert (dt.date(2010, 1, 1), 24) not in key, "restatement scored as if it were new"
    assert 999.0 not in cells.frame["value"].tolist()
    # and it is not smuggled into the exclusions either - it is simply not a
    # held-out cell at all
    assert cells.excluded["value"].tolist().count(999.0) == 0


def test_premium_comes_from_training_never_from_the_held_out_slice(backend_name):
    """Exposure leakage guard.

    Premium is restated alongside losses. If the attached value moves when only
    the POST-cutoff premium changes, the model is being handed an exposure it
    could not have had - and premium normalizes nearly every metric downstream,
    so the leak would flow into CRPS, ELPD and the reserve-basis points alike.
    """

    def build(premium_after):
        rows = staircase(4, 4, through=5)
        prem = [
            {
                "origin_period": dt.date(2010 + w - 1, 1, 1),
                "dev_lag": 12,
                "eval_date": dt.date(2010 + w - 1, 12, 31),
                "field": "earned_premium",
                "value": 1000.0,
            }
            for w in range(1, 5)
        ]
        # the same origins, restated on the held-out diagonal
        prem += [
            {
                "origin_period": dt.date(2010 + w - 1, 1, 1),
                "dev_lag": 12 * (6 - w),
                "eval_date": dt.date(2014, 12, 31),
                "field": "earned_premium",
                "value": premium_after,
            }
            for w in range(2, 5)
        ]
        tri = Triangle.from_long(
            pd.DataFrame([*rows, *prem]), measure="cumulative", backend=backend_name
        )
        return next_diagonal(
            tri, as_of="2013-12-31", fields="paid_loss", premium_field="earned_premium"
        )

    baseline = build(1000.0)
    leaked = build(7777.0)

    assert (baseline.frame["premium"] == 1000.0).all()
    pd.testing.assert_series_equal(baseline.frame["premium"], leaked.frame["premium"])


def test_premium_may_not_also_be_scored(tri8):
    """Exposure is attached to cells, never an outcome. Silently scoring premium
    as a loss field would be a plausible-looking, entirely wrong ELPD."""
    with pytest.raises(ValueError, match="not scored as an outcome"):
        next_diagonal(
            tri8,
            as_of="2017-12-31",
            fields=["paid_loss", "earned_premium"],
            premium_field="earned_premium",
        )


def test_prev_value_is_the_training_predecessor(tri8):
    """``prev_value`` must come from the training slice, so that
    ``value - prev_value`` is an increment against data rather than against
    another prediction - which is what makes the increment/cumulative change of
    variable have Jacobian 1."""
    cells = next_diagonal(tri8, as_of="2017-12-31", fields="paid_loss")

    # cell (w, d) has value 100w + d, so its predecessor is 100w + d - 1
    assert cells.frame["prev_value"].tolist() == [100 * w + (10 - w) - 1 for w in range(2, 9)]
    assert np.allclose(cells.increments, 1.0)


def test_ragged_triangle_excludes_cells_with_no_predecessor(backend_name):
    """On a real (ragged) triangle a next-diagonal cell can have a hole behind
    it. No increment can be formed, so it is excluded WITH A REASON rather than
    silently carrying a NaN into a log density."""
    rows = staircase(4, 4, through=5)
    # punch out origin 3's dev-2 cell, which sits inside the training triangle
    rows = [r for r in rows if not (r["origin_period"].year == 2012 and r["dev_lag"] == 24)]
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)

    cells = next_diagonal(tri, as_of="2013-12-31", fields="paid_loss")

    assert cells.exclusion_counts()["no_predecessor"] == 1
    hole = cells.excluded[cells.excluded["reason"] == "no_predecessor"].iloc[0]
    assert hole["origin_period"] == dt.date(2012, 1, 1)
    assert hole["dev_lag"] == 36
    assert not cells.frame["prev_value"].isna().any()


def test_multiple_fields_and_segments_are_keyed_independently(backend_name):
    """compartmental scores paid AND reported jointly, and the mart is
    multi-cohort. The cell key has to carry both, or two lines' cells collide."""
    rows = []
    for lob in ("wc", "ca"):
        for fld in ("paid_loss", "reported_loss"):
            rows += staircase(4, 4, through=5, field=fld, segment={"lob": lob})
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)

    cells = next_diagonal(tri, as_of="2013-12-31", fields=["paid_loss", "reported_loss"])

    assert cells.segments == ("lob",)
    assert cells.n_cells == 3 * 2 * 2  # 3 scorable cells x 2 fields x 2 lobs
    assert cells.key().nunique() == cells.n_cells, "cell keys must be unique"
    assert set(cells.frame["lob"]) == {"wc", "ca"}
    assert set(cells.frame["field"]) == {"paid_loss", "reported_loss"}


def test_origins_restricts_to_the_study_window(tri8):
    """CLAUDE.md's standing gotcha: the mart carries accident years past the
    Meyers window, and anything aggregated over origins the training slice never
    had is silently wrong."""
    window = [dt.date(y, 1, 1) for y in range(2011, 2015)]
    cells = next_diagonal(tri8, as_of="2017-12-31", fields="paid_loss", origins=window)

    assert set(cells.frame["origin_period"]) <= set(window)
    assert cells.train_origins == tuple(window)

    # 4 origins are on the diagonal, but only 3 are scorable: dropping the older
    # origins also drops the DEEPEST training cells, so the window's oldest
    # origin now develops past anything its own training slice reached. That is
    # the honest answer - a model fit on these origins has no parameter at that
    # dev - and it is why the restriction has to happen before depth is measured.
    assert cells.n_cells == 3
    assert cells.exclusion_counts()["dev_beyond_trained"] == 1
    assert cells.excluded.iloc[0]["origin_period"] == dt.date(2011, 1, 1)

    full = next_diagonal(tri8, as_of="2017-12-31", fields="paid_loss")
    assert dt.date(2011, 1, 1) in set(full.frame["origin_period"]), (
        "the same cell IS scorable when the older origins are in training"
    )


def test_a_premium_only_update_does_not_become_the_next_diagonal(backend_name):
    """``D_next`` must be read from the eval dates of the LOSS rows being
    scored, not from the triangle's dates as a whole.

    Premium is restated on its own schedule. If an exposure-only update lands
    between the cutoff and the next loss diagonal, taking the triangle-wide
    minimum selects that date, holds out an empty diagonal, and skips the real
    one - reporting "0 cells scored" for a model that had a perfectly good
    diagonal waiting.
    """
    rows = staircase(4, 4, through=5)
    rows += [
        {
            "origin_period": dt.date(2010 + w - 1, 1, 1),
            "dev_lag": 12,
            "eval_date": dt.date(2010 + w - 1, 12, 31),
            "field": "earned_premium",
            "value": 1000.0,
        }
        for w in range(1, 5)
    ]
    # an exposure-only restatement dated BETWEEN the cutoff and the loss diagonal
    rows.append(
        {
            "origin_period": dt.date(2010, 1, 1),
            "dev_lag": 12,
            "eval_date": dt.date(2014, 6, 30),
            "field": "earned_premium",
            "value": 1100.0,
        }
    )
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)
    assert dt.date(2014, 6, 30) in tri.eval_dates

    cells = next_diagonal(
        tri, as_of="2013-12-31", fields="paid_loss", premium_field="earned_premium"
    )

    assert cells.eval_date == dt.date(2014, 12, 31), "premium-only date became the diagonal"
    assert cells.n_cells == 3


def test_a_loss_field_restatement_does_not_become_the_next_diagonal(backend_name):
    """Filtering to the scored fields is not enough on its own.

    A restatement of an ALREADY-TRAINED paid-loss cell carries a real paid_loss
    row, so it survives any field filter and looks like the next diagonal. The
    anti-join then correctly removes it and nothing is left - an empty hold-out
    reported as "0 cells scored" while the actual new diagonal sits one date
    later, untouched.

    So a date counts only if it introduces a cell key training did not have.
    """
    rows = staircase(4, 4, through=5)
    rows.append(
        {
            "origin_period": dt.date(2010, 1, 1),
            "dev_lag": 24,  # observed at the 2011 diagonal: inside training
            "eval_date": dt.date(2014, 6, 30),  # restated between cutoff and diagonal
            "field": "paid_loss",
            "value": 999.0,
        }
    )
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)
    assert dt.date(2014, 6, 30) in tri.eval_dates

    cells = next_diagonal(tri, as_of="2013-12-31", fields="paid_loss")

    assert cells.eval_date == dt.date(2014, 12, 31), "a restatement became the diagonal"
    assert cells.n_cells == 3
    assert 999.0 not in cells.frame["value"].tolist()


def test_a_diagonal_of_pure_restatements_is_an_error_not_an_empty_panel(backend_name):
    """If NOTHING after the cutoff is new, say so.

    Returning an empty panel would read downstream as "this model scored
    nothing", which is a statement about the model rather than about the data.
    """
    rows = staircase(4, 4, through=4)
    rows += [
        {
            "origin_period": dt.date(2010, 1, 1),
            "dev_lag": 12 * d,
            "eval_date": dt.date(2015, 12, 31),
            "field": "paid_loss",
            "value": 900.0 + d,
        }
        for d in (1, 2)  # both already in training
    ]
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)

    with pytest.raises(ValueError, match="only restatements"):
        next_diagonal(tri, as_of="2013-12-31", fields="paid_loss")


def test_new_origin_is_decided_per_cohort_not_triangle_wide(backend_name):
    """A cohort's first accident year is new to THAT cohort's fit, whatever the
    other cohorts in the triangle wrote.

    One company writing a line since 2010 must not make another company's 2010
    look like trained history. The cell would be scored against an ``alpha[w]``
    the second fit never estimated - and it is the oldest, largest-reserve
    origin, so the error lands where it matters most.
    """
    rows = staircase(4, 4, through=5, segment={"lob": "wc"})
    # 'ca' starts a year later, so 2010 is absent from it entirely
    rows += [
        r
        for r in staircase(4, 4, through=5, start_year=2011, segment={"lob": "ca"})
        if r["origin_period"].year >= 2011
    ]
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)

    cells = next_diagonal(tri, as_of="2013-12-31", fields="paid_loss")

    scored = cells.frame
    ca_origins = set(scored.loc[scored["lob"] == "ca", "origin_period"])
    assert dt.date(2010, 1, 1) not in ca_origins
    # wc did train on 2010, so ITS 2010 cell is unaffected by ca's absence
    wc_origins = set(scored.loc[scored["lob"] == "wc", "origin_period"])
    assert wc_origins != ca_origins


def test_incremental_triangles_do_not_get_cumulative_treatment(backend_name):
    """``prev_value`` and ``increments`` are cumulative-only concepts.

    On an incremental triangle ``value`` IS the increment, so differencing again
    is wrong - and wrong quietly, giving a smaller number of the right sign.
    Equally, every cell would look like it had a missing predecessor and the
    whole diagonal would vanish behind a spurious exclusion reason.
    """
    rows = staircase(4, 4, through=5)
    tri = Triangle.from_long(pd.DataFrame(rows), measure="incremental", backend=backend_name)

    cells = next_diagonal(tri, as_of="2013-12-31", fields="paid_loss")

    assert cells.measure == "incremental"
    assert cells.n_cells == 3, "cells must not be excluded for lacking a predecessor"
    assert cells.exclusion_counts()["no_predecessor"] == 0
    with pytest.raises(ValueError, match="for cumulative triangles"):
        _ = cells.increments


def test_no_next_diagonal_is_an_error_not_an_empty_frame(tri8):
    """Asking to hold out past the end of the data is a mistake in the caller,
    and an empty frame would read downstream as "the model scored nothing", not
    as "you asked for nothing"."""
    with pytest.raises(ValueError, match="no next diagonal"):
        next_diagonal(tri8, as_of="2018-12-31", fields="paid_loss")


def test_unknown_field_names_what_is_available(tri8):
    with pytest.raises(ValueError, match="has no field"):
        next_diagonal(tri8, as_of="2017-12-31", fields="incurred_loss")
