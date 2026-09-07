"""Behavioral spec for ``triangle/transforms.py`` - the core algebra of the
Triangle layer: cumulative<->incremental, grain changes, and the ``as_of``
backtest slice.

Every test here runs twice, once per ibis backend (see ``conftest.backend_name``).
That parameterization is load-bearing rather than cosmetic: the ibis polars
backend has no window-function support, so the transforms are written as
equi-join + group-by formulations, and this suite is what proves those give the
same answer as the duckdb path. A test that passes only under ``[duckdb]``
means a window function crept back in.

Invariants pinned here:
- cum <-> incr is a lossless round trip and idempotent when already in target form;
- coarsening dev/origin grain aggregates correctly and is refuse-only in the
  refining direction (you cannot invent quarters from years);
- ``as_of`` respects restatement history: a cell restated at a later eval must
  resolve to the value that was on the books at the requested evaluation date.
"""

import pandas as pd
import pytest

from ibnr import Triangle

from .conftest import assert_triangles_equal, d, sorted_long


def test_cum_incr_round_trip(small_cumulative):
    """incr = diff(cum) cell-by-cell within an origin, and to_cumulative undoes it
    exactly - differencing must not lose or fabricate cells."""
    incr = small_cumulative.to_incremental()
    assert incr.meta.measure == "incremental"
    wide = incr.to_wide()
    assert wide.loc[d("2020-01-01"), 12] == 100.0
    assert wide.loc[d("2020-01-01"), 24] == 50.0
    assert wide.loc[d("2020-01-01"), 36] == 25.0
    assert wide.loc[d("2021-01-01"), 24] == 55.0
    back = incr.to_cumulative()
    assert back.meta.measure == "cumulative"
    assert_triangles_equal(back, small_cumulative)


def test_cum_incr_idempotent(small_cumulative):
    """Converting to the measure a triangle already has is a no-op - and returns
    the *same object* (``is``), so callers can convert defensively for free."""
    assert small_cumulative.to_cumulative() is small_cumulative
    incr = small_cumulative.to_incremental()
    assert incr.to_incremental() is incr


def test_as_of(small_cumulative):
    """``as_of`` yields the triangle as it stood at a past evaluation date - the
    slice every backtest trains on - and carries metadata through unchanged."""
    upper = small_cumulative.as_of("2021-12-31")
    df = sorted_long(upper)
    # 3 of the 6 cells had been evaluated by 12/31/2021: (2020, 12/24) and (2021, 12)
    assert len(df) == 3
    assert df["eval_date"].max() == pd.Timestamp("2021-12-31")
    # slicing preserves measure and grain
    assert upper.meta == small_cumulative.meta


def test_latest_diagonal(small_cumulative):
    """The latest diagonal is one cell per origin at its most recent eval - the
    paid-to-date column reserves are measured against (reserve = ultimate - this)."""
    diag = sorted_long(small_cumulative.latest_diagonal())
    assert len(diag) == 3
    expected = {36: 175.0, 24: 165.0, 12: 120.0}
    assert dict(zip(diag["dev_lag"], diag["value"], strict=True)) == expected
    assert (diag["eval_date"] == pd.Timestamp("2022-12-31")).all()


def _quarterly_dev_triangle(backend_name, measure: str) -> Triangle:
    """One origin year observed quarterly through 24 months (OYDQ).

    Built in both measures from the same cumulative ladder so the grain tests can
    check that coarsening picks the *last* value in each year bucket when
    cumulative (50 at dev 12, 70 at dev 24) but *sums* the bucket when incremental
    (50, then 20).
    """
    cum = [
        (3, 10.0),
        (6, 30.0),
        (9, 45.0),
        (12, 50.0),
        (15, 58.0),
        (18, 64.0),
        (21, 67.0),
        (24, 70.0),
    ]
    rows = []
    prev = 0.0
    for lag, v in cum:
        val = v if measure == "cumulative" else v - prev
        prev = v
        # eval_date convention (CLAUDE.md): last day of the month that
        # origin + dev_lag lands in, so dev 3 on a 1/1/2020 origin -> 3/31/2020.
        month_end = pd.Timestamp("2020-01-01") + pd.DateOffset(months=lag) - pd.Timedelta(days=1)
        rows.append(("2020-01-01", lag, month_end.date(), "paid_loss", val))
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "field", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    return Triangle.from_long(
        df, measure=measure, origin_grain="Y", dev_grain="Q", backend=backend_name
    )


def test_dev_grain_cumulative(backend_name):
    """Coarsening dev grain on a cumulative triangle keeps the last cell of each
    bucket (cumulative values must not be added up)."""
    t = _quarterly_dev_triangle(backend_name, "cumulative")
    out = t.with_dev_grain("Y")
    assert out.meta.grain == "OYDY"
    wide = out.to_wide()
    assert list(wide.columns) == [12, 24]
    assert wide.iloc[0].tolist() == [50.0, 70.0]


def test_dev_grain_incremental(backend_name):
    """Same coarsening on an incremental triangle sums each bucket, and the bucket
    inherits the latest eval_date it covers (not the earliest)."""
    t = _quarterly_dev_triangle(backend_name, "incremental")
    out = t.with_dev_grain("Y")
    wide = out.to_wide()
    assert wide.iloc[0].tolist() == [50.0, 20.0]
    # bucket eval_date is the latest eval it contains
    df = sorted_long(out)
    assert df["eval_date"].tolist() == [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-12-31")]


def test_dev_grain_errors(small_cumulative):
    """Grain changes only ever coarsen: refining Y -> Q would have to invent
    unobserved sub-annual detail, so it raises rather than interpolating."""
    with pytest.raises(ValueError, match="refine"):
        small_cumulative.with_dev_grain("Q")
    with pytest.raises(ValueError, match="unknown grain"):
        small_cumulative.with_dev_grain("W")


def test_origin_grain(backend_name):
    """Coarsening origin grain merges accident quarters into an accident year by
    *calendar diagonal*, not by dev_lag: the two Q origins sit at different lags
    (12 and 9) at the same 12/31/2020 eval, and must combine into one dev-12 cell.
    """
    # two quarterly origins in 2020, observed at two year-end evals
    rows = [
        ("2020-01-01", 12, "2020-12-31", 40.0),
        ("2020-04-01", 9, "2020-12-31", 30.0),
        ("2020-01-01", 24, "2021-12-31", 60.0),
        ("2020-04-01", 21, "2021-12-31", 45.0),
    ]
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    t = Triangle.from_long(
        df, measure="cumulative", origin_grain="Q", dev_grain="Q", backend=backend_name
    )
    out = t.with_origin_grain("Y")
    assert out.meta.grain == "OYDQ"
    df_out = sorted_long(out)
    assert df_out["origin_period"].unique().tolist() == [pd.Timestamp("2020-01-01")]
    assert dict(zip(df_out["dev_lag"], df_out["value"], strict=True)) == {12: 70.0, 24: 105.0}


def _origin_grain_frame(rows):
    """Long frame for the origin-grain tests, one field, dates as ``date``."""
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    return df


def test_origin_grain_unchanged_is_a_no_op(backend_name):
    """Asking for the origin grain the triangle already has returns the *same
    object*, as ``with_dev_grain`` does: nothing is recomputed, so nothing moves.

    The input here is one a coarsening would have to refuse: both rows sit at the
    12/31/2021 eval, one of them stored at dev 12, so eval_date and dev_lag
    disagree on that row. Recomputing dev_lag from eval_date turned the two cells
    into one cell of 245.0, and the total was preserved, so nothing downstream
    could tell. Doing nothing cannot corrupt anything, so the unchanged grain is
    still a no-op on data a coarsening refuses.
    """
    rows = [
        ("2020-01-01", 12, "2021-12-31", 95.0),  # eval_date says dev 24, not 12
        ("2020-01-01", 24, "2021-12-31", 150.0),
    ]
    t = Triangle.from_long(
        _origin_grain_frame(rows),
        measure="cumulative",
        origin_grain="Y",
        dev_grain="Y",
        backend=backend_name,
    )
    before = sorted_long(t)
    out = t.with_origin_grain("Y")
    assert out is t
    pd.testing.assert_frame_equal(sorted_long(out), before)


def test_origin_grain_refuses_misaligned_rows(backend_name):
    """A real coarsening re-derives dev_lag from eval_date, so it reads eval_date
    as the whole truth about a row's development. A row whose stored dev_lag says
    something else is therefore summed into whichever cell its eval_date names,
    which is a wrong number rather than a missing one. Refuse it by name.

    The rows are ``test_origin_grain``'s quarterly fixture (which still passes,
    unchanged, as the aligned control) plus one row that disagrees with its
    eval_date. Both directions of that disagreement are checked, because a
    comparison written one way round would refuse the first and accept the
    second: the extra row is carried either later than its stored dev_lag says
    (dev 9 at the 12/31/2021 eval, where 21 is the aligned lag) or earlier (dev
    21 at the 12/31/2020 eval, where 9 is).

    The message is checked past its first phrase as well. ``operation`` and
    ``reason`` are arguments the caller passes in, so a test that only looked for
    the shared "does not align" wording could not tell whether they arrived: the
    pattern below reaches the operation's own name, a phrase only this caller's
    reason carries, and the note about slicing that follows both.

    The last case puts both extra rows in one triangle and pins the count the
    message opens with. Every other misaligned fixture in the suite carries
    exactly one row, so a counter that answered "is there any" rather than "how
    many" would tell a user with three hundred bad rows that there is one.
    """
    aligned = [
        ("2020-01-01", 12, "2020-12-31", 40.0),
        ("2020-04-01", 9, "2020-12-31", 30.0),
        ("2020-01-01", 24, "2021-12-31", 60.0),
        ("2020-04-01", 21, "2021-12-31", 45.0),
    ]
    carried_later = ("2020-04-01", 9, "2021-12-31", 33.0)  # eval_date says dev 21
    carried_earlier = ("2020-04-01", 21, "2020-12-31", 33.0)  # eval_date says dev 9
    one_row = r"^1 rows where eval_date does not align"
    two_rows = r"^2 rows where eval_date does not align"
    reason = r"does not align.*with_origin_grain\(\).*Coarsening sums the origins.*as_of\(\)"
    cases = [
        ([carried_later], one_row),
        ([carried_earlier], one_row),
        ([carried_later, carried_earlier], two_rows),
    ]
    for extra, count in cases:
        t = Triangle.from_long(
            _origin_grain_frame([*aligned, *extra]),
            measure="cumulative",
            origin_grain="Q",
            dev_grain="Q",
            backend=backend_name,
        )
        with pytest.raises(ValueError, match=reason):
            t.with_origin_grain("Y")
        with pytest.raises(ValueError, match=count):
            t.with_origin_grain("Y")


def test_origin_grain_takes_restated_history_once_it_is_sliced(backend_name):
    """The second way a row becomes misaligned is restatement, and there the way
    out is the slice the refusal names rather than a correction to the source.

    A restated cell keeps its dev_lag and gets a later eval_date, so it disagrees
    with the convention by construction and ``validate`` reports it under both
    findings. Slicing cannot align such a row, because neither ``as_of`` nor
    ``latest_diagonal`` ever changes an eval_date, but it can drop it, and that is
    what the refusal tells the caller to try. This checks the three outcomes the
    message promises: ``latest_diagonal()`` and an ``as_of`` before the restatement
    both leave a triangle that coarsens, and an ``as_of`` at the restatement keeps
    the row and is refused again.
    """
    rows = [
        ("2020-01-01", 12, "2020-12-31", 40.0),
        ("2020-04-01", 9, "2020-12-31", 30.0),
        ("2020-01-01", 24, "2021-12-31", 60.0),
        ("2020-04-01", 21, "2021-12-31", 45.0),
        ("2020-01-01", 12, "2021-12-31", 42.0),  # the 40 above, restated a year later
    ]
    t = Triangle.from_long(
        _origin_grain_frame(rows),
        measure="cumulative",
        origin_grain="Q",
        dev_grain="Q",
        backend=backend_name,
    )
    with pytest.raises(ValueError, match="does not align"):
        t.with_origin_grain("Y")

    # latest_diagonal keeps each cohort's greatest dev_lag, which the restatement
    # is not, so the misaligned row goes and the coarsening runs.
    assert t.latest_diagonal().with_origin_grain("Y").count() == 1

    # as_of before the restatement drops it and leaves the aligned 12/31/2020
    # diagonal, which coarsens to the same 70.0 the aligned fixture above gives.
    early = sorted_long(t.as_of("2020-12-31").with_origin_grain("Y"))
    assert dict(zip(early["dev_lag"], early["value"], strict=True)) == {12: 70.0}

    # as_of at the restatement keeps it instead of the value it replaced.
    with pytest.raises(ValueError, match="does not align"):
        t.as_of("2021-12-31").with_origin_grain("Y")


def _restated_wide_triangle(backend_name) -> Triangle:
    """The issue's first fixture: one cell booked at 100 and restated to 95 a year
    later, which ``to_wide`` used to display as a single cell of 195."""
    rows = [
        ("2020-01-01", 12, "2020-12-31", 100.0),
        ("2020-01-01", 12, "2021-12-31", 95.0),  # the same cell, restated
    ]
    return Triangle.from_long(_origin_grain_frame(rows), measure="cumulative", backend=backend_name)


def test_to_wide_refuses_a_cell_stored_twice(backend_name):
    """The square has one axis fewer than the triangle, so the pivot adds up
    everything stored at a cell - including a restatement and the value it
    replaced. 100 booked and 95 restated came out as one cell of 195 on both
    backends, a plausible number with nothing raised.

    The refusal is read past its first phrase, for the reason the origin-grain
    tests give: ``operation`` and ``reason`` are arguments the caller passes in,
    so a pattern anchored only on the shared count could not tell whether either
    arrived. This one reaches the operation's own name, a phrase only this
    caller's reason carries, and the restated route with the slice it names.
    """
    t = _restated_wide_triangle(backend_name)
    pattern = (
        r"1 cells are stored more than once.*to_wide\(\).*"
        r"adds every stored observation of a cell into one square.*"
        r"multiple eval_dates \(restated history\).*latest_diagonal\(\)"
    )
    with pytest.raises(ValueError, match=pattern):
        t.to_wide()


def test_to_wide_takes_a_restated_cell_once_it_is_sliced(backend_name):
    """The way out the refusal names, end to end: each slice leaves one view, and
    every one of them displays a value the triangle actually stored.

    195 is not one of them, which is the whole point - the three answers here are
    the two booked values, picked by which date the caller asked about.
    """
    t = _restated_wide_triangle(backend_name)
    with pytest.raises(ValueError, match="stored more than once"):
        t.to_wide()
    assert t.as_of("2020-12-31").to_wide().to_numpy().tolist() == [[100.0]]
    assert t.as_of("2021-12-31").to_wide().to_numpy().tolist() == [[95.0]]
    assert t.latest_diagonal().to_wide().to_numpy().tolist() == [[95.0]]


def test_to_wide_tells_a_duplicated_source_row_from_a_restatement(backend_name):
    """The second way a cell gets stored twice is the same eval_date twice, and it
    needs the other half of the message: slicing does not resolve it.

    ``as_of`` and ``latest_diagonal`` choose an eval_date and keep every row
    carrying it, so both still hand the pivot two rows - measured, 200 where the
    row says 100. Telling that caller to slice would send them round a loop, so
    the refusal sends them to the source data instead, and says nothing about
    restatement.
    """
    rows = [
        ("2020-01-01", 12, "2020-12-31", 100.0),
        ("2020-01-01", 12, "2020-12-31", 100.0),  # the same row twice, not a restatement
    ]
    t = Triangle.from_long(_origin_grain_frame(rows), measure="cumulative", backend=backend_name)
    pattern = (
        r"1 cells are stored more than once.*to_wide\(\).*"
        r"adds every stored observation of a cell into one square.*"
        r"1 duplicated \(segment, field, origin, dev_lag, eval_date\) cells.*"
        r"slicing cannot resolve.*source data"
    )
    with pytest.raises(ValueError, match=pattern) as excinfo:
        t.to_wide()
    assert "restated history" not in str(excinfo.value)
    # and the claim the message makes about slicing is true, so the caller is not
    # sent round a loop: both slices still carry the pair.
    with pytest.raises(ValueError, match="duplicated"):
        t.as_of("2020-12-31").to_wide()
    with pytest.raises(ValueError, match="duplicated"):
        t.latest_diagonal().to_wide()


def test_to_wide_names_both_routes_when_both_are_present(backend_name):
    """A triangle can hold both states at once, and then the message has to carry
    both ways out with the count each one accounts for - one message that told
    this caller only about slicing would leave the duplicate behind after they
    sliced."""
    rows = [
        ("2020-01-01", 12, "2020-12-31", 100.0),
        ("2020-01-01", 12, "2021-12-31", 95.0),  # restated
        ("2020-01-01", 24, "2021-12-31", 150.0),
        ("2020-01-01", 24, "2021-12-31", 150.0),  # duplicated
    ]
    t = Triangle.from_long(_origin_grain_frame(rows), measure="cumulative", backend=backend_name)
    pattern = (
        r"2 cells are stored more than once.*"
        r"1 cells observed at multiple eval_dates \(restated history\).*latest_diagonal\(\).*"
        r"1 duplicated \(segment, field, origin, dev_lag, eval_date\) cells.*source data"
    )
    with pytest.raises(ValueError, match=pattern):
        t.to_wide()


def test_to_wide_checks_the_field_it_pivots_and_not_the_others(backend_name):
    """Restated premium is no reason to refuse to display paid loss.

    ``to_wide`` selects one field and then pivots, so the check belongs on the
    selection: a triangle carrying the mart's several fields would otherwise be
    undisplayable because one of them is restated on its own schedule.
    """
    loss = _origin_grain_frame([("2020-01-01", 12, "2020-12-31", 100.0)])
    premium = _origin_grain_frame(
        [
            ("2020-01-01", 12, "2020-12-31", 700.0),
            ("2020-01-01", 12, "2021-12-31", 720.0),  # premium restated on its own
        ]
    ).assign(field="earned_premium")
    t = Triangle.from_long(
        pd.concat([loss, premium], ignore_index=True),
        measure="cumulative",
        backend=backend_name,
    )
    assert t.to_wide("paid_loss").to_numpy().tolist() == [[100.0]]
    with pytest.raises(ValueError, match="stored more than once"):
        t.to_wide("earned_premium")


def test_to_wide_still_adds_the_segments_up(backend_name):
    """The other sum the pivot performs is deliberate and is left alone.

    The square has no segment axis either, so two lines display as their total.
    The new check keys on the segment columns, exactly as a cell's identity does,
    so it says nothing about that - which is worth pinning, because a check that
    dropped the segments from its key would refuse every multi-line triangle and
    look like the same fix.
    """
    df = _origin_grain_frame(
        [
            ("2020-01-01", 12, "2020-12-31", 100.0),
            ("2020-01-01", 12, "2020-12-31", 95.0),
        ]
    ).assign(lob=["auto", "home"])
    t = Triangle.from_long(df, measure="cumulative", segments=["lob"], backend=backend_name)
    assert t.to_wide().to_numpy().tolist() == [[195.0]]


def _restated_bucket_triangle(backend_name, measure: str) -> Triangle:
    """The issue's second fixture: one annual bucket of quarterly cells whose
    dev-3 observation was restated three months after it was first booked.

    Incremental, the four cells as they now stand are worth 105 and the bucket
    sum answered 125, because the superseded 20 was added beside the 30 that
    replaced it. Cumulative, the same shape one step further out: the dev-3 cell
    is restated exactly one bucket later, so the original and the restatement
    both land on a bucket boundary and both survive the filter.
    """
    if measure == "incremental":
        rows = [
            ("2020-01-01", 3, "2020-03-31", 20.0),
            ("2020-01-01", 3, "2020-06-30", 30.0),  # the 20 above, restated
            ("2020-01-01", 6, "2020-06-30", 30.0),
            ("2020-01-01", 9, "2020-09-30", 25.0),
            ("2020-01-01", 12, "2020-12-31", 20.0),
        ]
    else:
        rows = [
            ("2020-01-01", 3, "2020-03-31", 20.0),
            ("2020-01-01", 6, "2020-06-30", 50.0),
            ("2020-01-01", 9, "2020-09-30", 75.0),
            ("2020-01-01", 12, "2020-12-31", 95.0),
            ("2020-01-01", 15, "2021-03-31", 110.0),
            ("2020-01-01", 3, "2021-03-31", 25.0),  # the 20 above, restated a year later
        ]
    return Triangle.from_long(
        _origin_grain_frame(rows),
        measure=measure,
        origin_grain="Y",
        dev_grain="Q",
        backend=backend_name,
    )


def test_dev_grain_refuses_a_restated_increment(backend_name):
    """Coarsening an incremental triangle sums each bucket, so a restatement is
    added beside the value it replaced: 125 where the four cells are worth 105,
    on both backends, with nothing raised.

    The reason in the message is the incremental one, not a sentence covering
    both measures - the cumulative path does not sum at all, so one shared
    wording would be wrong about it.
    """
    t = _restated_bucket_triangle(backend_name, "incremental")
    pattern = (
        r"1 cells are stored more than once.*with_dev_grain\(\).*"
        r"increments inside a bucket are summed.*worth 105 came out as 125.*"
        r"multiple eval_dates \(restated history\).*latest_diagonal\(\)"
    )
    with pytest.raises(ValueError, match=pattern):
        t.with_dev_grain("Y")


def test_dev_grain_takes_a_restated_increment_once_it_is_sliced(backend_name):
    """The way out, end to end, and the number it produces: 105, the four cells as
    they stand once the superseded 20 is dropped.

    ``latest_diagonal`` also works and answers something else entirely - it keeps
    one cell per origin, so the bucket is the dev-12 increment alone. Both are
    single views; which one a caller wants is theirs to choose, which is exactly
    why neither is applied for them inside the operation.
    """
    t = _restated_bucket_triangle(backend_name, "incremental")
    with pytest.raises(ValueError, match="stored more than once"):
        t.with_dev_grain("Y")
    sliced = sorted_long(t.as_of("2020-12-31").with_dev_grain("Y"))
    assert dict(zip(sliced["dev_lag"], sliced["value"], strict=True)) == {12: 105.0}
    diag = sorted_long(t.latest_diagonal().with_dev_grain("Y"))
    assert dict(zip(diag["dev_lag"], diag["value"], strict=True)) == {12: 20.0}


def test_dev_grain_unchanged_is_a_no_op_on_a_triangle_it_would_refuse(backend_name):
    """The refusal sits after the step == 1 short-circuit, matching the origin
    regrain: a no-op recomputes nothing, so nothing can move, so there is nothing
    to protect the caller from.

    The first assertion is what makes the test's name true - this is a triangle a
    real coarsening refuses - and without it the rest would pass just as well on a
    build where the check never fires at all.
    """
    t = _restated_bucket_triangle(backend_name, "incremental")
    with pytest.raises(ValueError, match="stored more than once"):
        t.with_dev_grain("Y")
    before = sorted_long(t)
    out = t.with_dev_grain("Q")
    assert out is t
    pd.testing.assert_frame_equal(sorted_long(out), before)


def test_dev_grain_refuses_a_restated_cumulative_cell(backend_name):
    """The cumulative path never sums, and is refused anyway, for the two reasons
    it has of its own.

    Bucket boundaries are counted back from the triangle's latest eval_date,
    which a restatement moves, so the row kept for a cell need not be the one
    that survives it - measured on a neighbouring fixture, the regrain kept a
    superseded dev-3 value of 20 and dropped the 30 that had replaced it. And
    when a cell and its restatement sit exactly one bucket apart, as here, both
    survive: the coarsened triangle carried dev 3 twice, at 20 and 25, which
    ``to_wide`` then displayed as 45.
    """
    t = _restated_bucket_triangle(backend_name, "cumulative")
    pattern = (
        r"1 cells are stored more than once.*with_dev_grain\(\).*"
        r"lands on a bucket boundary.*latest eval_date.*"
        r"multiple eval_dates \(restated history\).*latest_diagonal\(\)"
    )
    with pytest.raises(ValueError, match=pattern) as excinfo:
        t.with_dev_grain("Y")
    assert "increments inside a bucket are summed" not in str(excinfo.value)
    # sliced, it coarsens, and the dev-3 cell shows the 25 that replaced the 20
    out = sorted_long(t.as_of("2021-03-31").with_dev_grain("Y"))
    assert dict(zip(out["dev_lag"], out["value"], strict=True)) == {3: 25.0, 15: 110.0}


def test_as_of_drops_restatements(backend_name):
    """With restatement history in the table, ``as_of`` must return what was booked
    at that date, not the latest revision - otherwise backtests leak the future.

    ``eval_date`` is a stored first-class column precisely so this is expressible:
    the same (origin, dev) cell legitimately appears twice with different values.
    """
    rows = [
        ("2020-01-01", 12, "2020-12-31", 100.0),
        ("2020-01-01", 12, "2021-12-31", 95.0),  # restated
    ]
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    t = Triangle.from_long(df, backend=backend_name)
    assert sorted_long(t.as_of("2021-12-31"))["value"].tolist() == [95.0]
    assert sorted_long(t.as_of("2020-12-31"))["value"].tolist() == [100.0]
    assert sorted_long(t.latest_diagonal())["value"].tolist() == [95.0]
