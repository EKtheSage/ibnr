"""kernels.nn_contract: the NN data contract. Pure numpy - no torch needed.

The neural entries consume dense masked grids rather than long rows: loss ratios
(value / earned premium) on a (cohort, channel, n_w, n_d) tensor, paired with
masks marking which cells the network is allowed to attend to - ``obs_mask`` for
the prediction target, ``x_obs`` per channel. Normalizing by premium is what
makes triangles from companies of wildly different size poolable - Schedule P
triangles are small and overfitting is the central risk, so training always
spans many company x line cohorts.

The file matters because these are the failure modes that quietly corrupt a
training set rather than crashing it: a cell whose predecessor is missing has no
usable increment and must be masked out; a feature missing where the target is
present is not a feature of zero, and one present where the target is missing
must not be destroyed - both are what per-channel observedness exists for; a
field that is an eval-date SNAPSHOT (``case_reserve``) must not be differenced
into its own movement; a cohort with non-positive premium produces meaningless
ratios and must be dropped (reported, not hidden); the anchor used for the
autoregressive rollout (``latest_cum``/``latest_dev``) must stay the raw
cumulative even where the mask says the increment is unusable.

Deliberately torch-free - ``ibnr.gallery`` and its contracts must import without
the ``[nn]`` extra (CLAUDE.md), so these run in the core CI job. Both ibis
backends via ``backend_name``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.nn_contract import cutoff_masks, nn_data

from .conftest import make_multiline_triangle, upper_mask

CUM = np.array(
    [
        [100.0, 150.0, 175.0],
        [110.0, 165.0, np.nan],
        [120.0, np.nan, np.nan],
    ]
)
PREMIUM = {"lob_a": np.array([200.0, 210.0, 220.0]), "lob_b": np.array([400.0, 420.0, 440.0])}
#: an eval-date SNAPSHOT on the same grid as CUM - the shape ``level_fields`` exists for
CASE = np.array(
    [
        [10.0, 20.0, 30.0],
        [15.0, 25.0, np.nan],
        [18.0, np.nan, np.nan],
    ]
)


def _cell(w: int, dev: int, field: str, value: float) -> dict:
    return {
        "company_code": "0001",
        "line_of_business": "lob_a",
        "origin_period": dt.date(2010 + w, 1, 1),
        "dev_lag": 12 * (dev + 1),
        "eval_date": dt.date(2010 + w + dev, 12, 31),
        "field": field,
        "value": value,
    }


def _fields_triangle(backend_name: str, cum_by_field: dict[str, np.ndarray]) -> Triangle:
    """One cohort whose fields sit on DIFFERENT cells.

    ``make_multiline_triangle`` emits every field on the same cells, so a
    triangle where the feature is missing exactly where the target is present
    (or the reverse) has to be built here - and those two cases are what
    per-channel observedness exists to represent. Conventions match the shared
    builder: NaN = not emitted, yearly grain, origin w = Jan 1 of 2010 + w, dev
    index ``dev`` = dev_lag ``12 * (dev + 1)``. Premium is emitted wherever ANY
    field has a cell, so it never becomes the reason a cohort is screened out.
    """
    rows: list[dict] = []
    present: set[tuple[int, int]] = set()
    for field, cum in cum_by_field.items():
        n_w, n_d = cum.shape
        for w in range(n_w):
            for dev in range(n_d):
                if np.isnan(cum[w, dev]):
                    continue
                present.add((w, dev))
                rows.append(_cell(w, dev, field, float(cum[w, dev])))
    for w, dev in sorted(present):
        rows.append(_cell(w, dev, "earned_premium", float(PREMIUM["lob_a"][w])))
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)


@pytest.fixture
def two_cohorts(backend_name) -> Triangle:
    return make_multiline_triangle(
        backend_name,
        {"lob_a": CUM, "lob_b": CUM * 2.0},
        premium_by_lob=PREMIUM,
    )


def test_nn_data_shapes_and_values(two_cohorts):
    """Full spec of the emitted tensors, with the loss-ratio features hand-computed
    so a reviewer can verify the premium normalization rather than trust it."""
    data = nn_data(two_cohorts, loss_field="paid_loss", premium_field="earned_premium")
    assert data["x"].shape == (2, 1, 3, 3)  # (cohorts, channels, n_w, n_d)
    assert data["obs_mask"].shape == (2, 3, 3)
    assert data["n_w"] == 3 and data["n_d"] == 3
    assert data["lob_levels"] == ["lob_a", "lob_b"]
    np.testing.assert_array_equal(data["lob_idx"], [0, 1])
    np.testing.assert_array_equal(data["obs_mask"][0], upper_mask(3, 3))

    # hand-computed incremental loss ratios for lob_a
    incr = np.array([[100.0, 50.0, 25.0], [110.0, 55.0, 0.0], [120.0, 0.0, 0.0]])
    # features are incremental LOSS RATIOS, not dollars - this is what makes
    # differently-sized companies poolable into one training set
    want = incr / PREMIUM["lob_a"][:, None]
    # unobserved cells are zero-filled; obs_mask (not the value) tells the network
    # what to ignore, so a real zero increment stays distinguishable from padding
    want[~upper_mask(3, 3)] = 0.0
    np.testing.assert_allclose(data["x"][0, 0], want)
    # lob_b: double the losses, double the premium -> identical ratios
    np.testing.assert_allclose(data["x"][1, 0], want)

    # rollout anchor: cumulative at each origin's latest observed dev, and that dev
    np.testing.assert_allclose(data["latest_cum"][0], [175.0, 165.0, 120.0])
    np.testing.assert_array_equal(data["latest_dev"][0], [3, 2, 1])
    np.testing.assert_allclose(data["premium"][0], PREMIUM["lob_a"])

    # calendar index w + d (1-based): constant along each diagonal, which is what
    # the cutoff augmentation and the relative calendar encoding slice on
    cal = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(data["cal_idx"], cal)
    assert data["cohorts"].columns.tolist() == ["company_code", "line_of_business"]
    assert len(data["dropped"]) == 0


def test_nn_data_feature_channels(two_cohorts, backend_name):
    """Extra fields (e.g. reported loss alongside paid) become additional channels
    on the same grid, normalized identically - here reported = 3x paid, so channel 1
    must be exactly 3x channel 0."""
    df = two_cohorts.execute().copy()
    for col in ("origin_period", "eval_date"):
        df[col] = pd.to_datetime(df[col]).dt.date
    reported = df[df["field"] == "paid_loss"].assign(field="reported_loss")
    reported["value"] = reported["value"] * 3.0
    t = Triangle.from_long(
        pd.concat([df, reported], ignore_index=True), measure="cumulative", backend=backend_name
    )
    data = nn_data(
        t,
        loss_field="paid_loss",
        feature_fields=("reported_loss",),
        premium_field="earned_premium",
    )
    assert data["x"].shape == (2, 2, 3, 3)
    np.testing.assert_allclose(data["x"][:, 1], data["x"][:, 0] * 3.0)


# -- x_obs: per-channel observedness ---------------------------------------------


def test_x_obs_channel_zero_is_exactly_the_obs_mask(two_cohorts):
    """The invariant every consumer leans on: channel 0's per-channel mask IS
    ``obs_mask``. Without it there would be two answers to "can the network see
    the target here?" and a network gating on ``x_obs`` would silently train on a
    different cell set than the splits, cutoffs and held-out scorers use."""
    data = nn_data(two_cohorts, loss_field="paid_loss", premium_field="earned_premium")
    assert data["x_obs"].shape == (2, 1, 3, 3)
    assert data["x_obs"].dtype == bool
    np.testing.assert_array_equal(data["x_obs"][:, 0], data["obs_mask"])


def test_x_obs_marks_a_feature_hole_and_its_successor(backend_name):
    """A feature missing where the TARGET is observed. ``obs_mask`` cannot see it -
    it speaks only for channel 0 - so before ``x_obs`` the network read the
    contract's padding zero as an observed zero increment at both cells: the hole
    itself and the successor whose increment has nothing to difference against."""
    holed = (CUM * 3.0).copy()
    holed[0, 0] = np.nan
    t = _fields_triangle(backend_name, {"paid_loss": CUM, "reported_loss": holed})
    data = nn_data(
        t,
        loss_field="paid_loss",
        feature_fields=("reported_loss",),
        premium_field="earned_premium",
    )
    # the target is fully observed on the upper triangle either way
    np.testing.assert_array_equal(data["obs_mask"][0], upper_mask(3, 3))
    np.testing.assert_array_equal(data["x_obs"][0, 0], upper_mask(3, 3))
    want = upper_mask(3, 3).copy()
    want[0, 0] = False  # the missing cell
    want[0, 1] = False  # its successor: no predecessor to difference against
    np.testing.assert_array_equal(data["x_obs"][0, 1], want)
    # and the values at those two cells are padding, not a zero increment
    np.testing.assert_allclose(data["x"][0, 1, 0, :2], 0.0)


def test_a_feature_observed_where_the_target_is_not_keeps_its_value(backend_name):
    """The reverse hole, and the reason the padding rule is PER CHANNEL.

    Zero-filling every channel wherever the target's increment is unusable
    destroyed a feature value that was perfectly well observed - and once
    ``x_obs`` calls that cell usable, the zero is no longer padding a consumer
    can gate away, it is a fabricated observation.
    """
    holed = CUM.copy()
    holed[0, 1] = np.nan
    t = _fields_triangle(backend_name, {"paid_loss": holed, "reported_loss": CUM * 3.0})
    data = nn_data(
        t,
        loss_field="paid_loss",
        feature_fields=("reported_loss",),
        premium_field="earned_premium",
    )
    assert not data["obs_mask"][0, 0, 1]  # the target hole
    assert not data["obs_mask"][0, 0, 2]  # and its successor
    # the feature channel is untouched by the target's hole
    np.testing.assert_array_equal(data["x_obs"][0, 1], upper_mask(3, 3))
    # 3x the paid increment over premium, at a cell where the target is unusable
    np.testing.assert_allclose(data["x"][0, 1, 0, 1], 3.0 * 50.0 / 200.0)
    np.testing.assert_allclose(data["x"][0, 1, 0, 2], 3.0 * 25.0 / 200.0)


# -- level_fields: a channel carried undifferenced --------------------------------


def test_level_channel_is_undifferenced_and_premium_divided(backend_name):
    """A level channel is the snapshot over premium - a reserve-to-premium ratio -
    not the movement in it. Asserted against the differenced reading of the SAME
    field, so the test cannot pass on a contract that ignored ``level_fields``."""
    t = _fields_triangle(backend_name, {"paid_loss": CUM, "case_reserve": CASE})
    kwargs = {
        "loss_field": "paid_loss",
        "feature_fields": ("case_reserve",),
        "premium_field": "earned_premium",
    }
    data = nn_data(t, level_fields=("case_reserve",), **kwargs)
    want = CASE / PREMIUM["lob_a"][:, None]
    want[np.isnan(want)] = 0.0  # padding, gated by x_obs
    np.testing.assert_allclose(data["x"][0, 1], want)
    assert data["field_kinds"] == ("increment", "level")
    # the target channel is unaffected by a feature's kind
    as_incr = nn_data(t, **kwargs)
    np.testing.assert_allclose(data["x"][0, 0], as_incr["x"][0, 0])
    assert not np.allclose(as_incr["x"][0, 1], want)


def test_a_level_channel_needs_no_predecessor(backend_name):
    """Level observedness is CELL PRESENCE: a snapshot stands on its own, so the
    successor of a hole is usable where an increment channel's would not be. Read
    both ways off one triangle - the only difference is ``level_fields``."""
    holed = CASE.copy()
    holed[0, 0] = np.nan
    t = _fields_triangle(backend_name, {"paid_loss": CUM, "case_reserve": holed})
    kwargs = {
        "loss_field": "paid_loss",
        "feature_fields": ("case_reserve",),
        "premium_field": "earned_premium",
    }
    as_incr = nn_data(t, **kwargs)
    as_level = nn_data(t, level_fields=("case_reserve",), **kwargs)
    # the missing cell itself is unusable under either reading
    assert not as_incr["x_obs"][0, 1, 0, 0]
    assert not as_level["x_obs"][0, 1, 0, 0]
    # its successor is the whole difference
    assert not as_incr["x_obs"][0, 1, 0, 1]
    assert as_level["x_obs"][0, 1, 0, 1]
    np.testing.assert_allclose(as_level["x"][0, 1, 0, 1], CASE[0, 1] / PREMIUM["lob_a"][0])
    # neither reading touches the target channel
    np.testing.assert_array_equal(as_incr["obs_mask"], as_level["obs_mask"])
    np.testing.assert_array_equal(as_level["x_obs"][:, 0], as_level["obs_mask"])


def test_field_kinds_is_aligned_with_fields(two_cohorts, backend_name):
    """``field_kinds`` is read at the same index as the channel it describes, and
    channel 0 is an increment in every configuration."""
    plain = nn_data(two_cohorts, loss_field="paid_loss", premium_field="earned_premium")
    assert plain["fields"] == ["paid_loss"]
    assert plain["field_kinds"] == ("increment",)
    t = _fields_triangle(
        backend_name, {"paid_loss": CUM, "reported_loss": CUM * 3.0, "case_reserve": CASE}
    )
    data = nn_data(
        t,
        loss_field="paid_loss",
        feature_fields=("reported_loss", "case_reserve"),
        level_fields=("case_reserve",),
        premium_field="earned_premium",
    )
    assert data["fields"] == ["paid_loss", "reported_loss", "case_reserve"]
    assert data["field_kinds"] == ("increment", "increment", "level")


def test_level_fields_must_name_a_feature_field(backend_name):
    """``level_fields`` declares the KIND of a channel; it does not add one. A
    field named only there would otherwise be silently absent from the grid."""
    t = _fields_triangle(backend_name, {"paid_loss": CUM, "case_reserve": CASE})
    with pytest.raises(ValueError, match="not in feature_fields"):
        nn_data(
            t,
            loss_field="paid_loss",
            level_fields=("case_reserve",),
            premium_field="earned_premium",
        )


def test_the_target_cannot_be_a_level(backend_name):
    """Channel 0 is the emergence being predicted and the ultimate is rebuilt by
    summing it onto ``latest_cum``; a level target would make that sum meaningless,
    so it is refused by name rather than differenced anyway."""
    t = _fields_triangle(backend_name, {"paid_loss": CUM, "case_reserve": CASE})
    with pytest.raises(ValueError, match="cannot be in level_fields"):
        nn_data(
            t,
            loss_field="paid_loss",
            feature_fields=("case_reserve",),
            level_fields=("paid_loss", "case_reserve"),
            premium_field="earned_premium",
        )


def test_nn_data_rejects_duplicate_fields(two_cohorts):
    """Listing the target field as a feature must raise - it would hand the network
    the answer as an input channel."""
    with pytest.raises(ValueError, match="duplicate fields"):
        nn_data(two_cohorts, loss_field="paid_loss", feature_fields=("paid_loss",))


def test_an_absent_field_is_refused_by_name(two_cohorts):
    """``select_fields`` is a filter, so a field the triangle does not carry
    yields no rows rather than an error - and the fit would then run with an
    all-masked channel that conditions nothing, indistinguishable from one the
    feature genuinely reached. The premium field already gets this refusal;
    the loss and feature fields must answer the same way."""
    with pytest.raises(ValueError, match="case_reserv"):
        nn_data(two_cohorts, loss_field="paid_loss", feature_fields=("case_reserv",))
    with pytest.raises(ValueError, match="no_such_field"):
        nn_data(two_cohorts, loss_field="no_such_field")


def test_nn_data_gap_predecessor_is_unobserved(backend_name):
    """An interior hole masks out two cells, not one: the missing cell itself and
    its successor, whose increment would otherwise silently span two dev periods.

    But the rollout anchor is unaffected - it reads the raw cumulative at the latest
    observed dev, which is still a valid paid-to-date even across a gap.
    """
    gappy = CUM.copy()
    gappy[0, 1] = np.nan  # dev 2 missing -> dev 3 has no usable increment
    t = make_multiline_triangle(
        backend_name, {"lob_a": gappy, "lob_b": CUM * 2.0}, premium_by_lob=PREMIUM
    )
    data = nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    k = data["cohorts"]["line_of_business"].tolist().index("lob_a")
    assert not data["obs_mask"][k, 0, 1]  # missing cell
    assert not data["obs_mask"][k, 0, 2]  # gap predecessor
    # but the anchor still uses the raw cumulative at the latest observed dev
    assert data["latest_dev"][k, 0] == 3
    assert data["latest_cum"][k, 0] == gappy[0, 2]


def test_nn_data_drops_bad_premium_cohorts(backend_name):
    """A cohort with non-positive premium is dropped rather than divided by, and
    lands in ``dropped`` with a reason - exclusions must be auditable, since a
    silently shrinking training pool changes results without changing any metric."""
    bad_premium = {"lob_a": PREMIUM["lob_a"], "lob_b": np.array([400.0, -1.0, 440.0])}
    t = make_multiline_triangle(
        backend_name, {"lob_a": CUM, "lob_b": CUM * 2.0}, premium_by_lob=bad_premium
    )
    data = nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert data["cohorts"]["line_of_business"].tolist() == ["lob_a"]
    assert data["dropped"]["line_of_business"].tolist() == ["lob_b"]
    assert "premium" in data["dropped"]["reason"].iloc[0]


def test_nn_data_rejects_restated_cells(backend_name):
    """Restatement history must raise: picking a vintage is the caller's decision
    (``as_of()``), and choosing silently would leak post-cutoff information into
    training features."""
    t = make_multiline_triangle(
        backend_name, {"lob_a": CUM, "lob_b": CUM * 2.0}, premium_by_lob=PREMIUM
    )
    df = t.execute().copy()
    for col in ("origin_period", "eval_date"):
        df[col] = pd.to_datetime(df[col]).dt.date
    loss = df[df["field"] == "paid_loss"]
    restated = loss.iloc[[0]].assign(eval_date=dt.date(2023, 12, 31), value=999.0)
    hist = Triangle.from_long(
        pd.concat([df, restated], ignore_index=True), measure="cumulative", backend=backend_name
    )
    with pytest.raises(ValueError, match="multiple rows per"):
        nn_data(hist, loss_field="paid_loss", premium_field="earned_premium")


def test_nn_data_rejects_incremental(two_cohorts):
    """The contract differences the triangle itself, so it demands cumulative input;
    an incremental triangle would be differenced a second time."""
    with pytest.raises(ValueError, match="cumulative"):
        nn_data(two_cohorts.to_incremental(), loss_field="paid_loss")


def test_nn_data_obs_mask_matches_as_of_slice(two_cohorts):
    """Backtest composition: ``as_of`` then ``nn_data`` yields a grid containing only
    pre-cutoff information - the grid shrinks to the origins and devs that existed at
    the cutoff, so nothing after it can reach the network."""
    sliced = two_cohorts.as_of(dt.date(2011, 12, 31))  # keeps calendar diagonals 1-2
    data = nn_data(sliced, loss_field="paid_loss", premium_field="earned_premium")
    # the 2012 origin has no rows on/before the cutoff, so the grid shrinks
    assert data["n_w"] == 2 and data["n_d"] == 2
    np.testing.assert_array_equal(data["obs_mask"][0], [[True, True], [True, False]])
    np.testing.assert_allclose(data["latest_cum"][0], [150.0, 110.0])
    np.testing.assert_array_equal(data["latest_dev"][0], [2, 1])


def test_nn_data_origin_missing_from_one_cohort(backend_name):
    """Cohorts share one origin axis built from the union, so a line that never wrote
    a given accident year still occupies its row - fully masked, with a zero anchor
    at dev 0. Keeping the axis aligned is what lets cohorts batch together."""
    partial = CUM.copy()
    partial[2, :] = np.nan  # lob_a has no 2012 origin; lob_b does
    t = make_multiline_triangle(
        backend_name, {"lob_a": partial, "lob_b": CUM * 2.0}, premium_by_lob=PREMIUM
    )
    data = nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert data["n_w"] == 3  # shared origin axis comes from the union
    k = data["cohorts"]["line_of_business"].tolist().index("lob_a")
    assert not data["obs_mask"][k, 2].any()
    assert data["latest_dev"][k, 2] == 0
    assert data["latest_cum"][k, 2] == 0.0


def test_a_cohort_may_skip_an_origin_its_neighbour_carries(backend_name):
    """The origin-spacing rule is about the POOLED axis, and only that.

    lob_a here has no 2011 at all while lob_b does, so lob_a's own origins are
    2010 and 2012, two years apart. That is not a hole in the axis: the axis is
    the union, 2010, 2011, 2012, one year apart throughout, and lob_a simply
    occupies the 2011 row masked out. Reading the rule per cohort instead would
    refuse an ordinary triangle - the case above removes the LAST origin, which
    leaves a contiguous run either way, so it cannot tell the two rules apart.
    """
    partial = CUM.copy()
    partial[1, :] = np.nan  # lob_a has no 2011 origin; lob_b does
    t = make_multiline_triangle(
        backend_name, {"lob_a": partial, "lob_b": CUM * 2.0}, premium_by_lob=PREMIUM
    )
    data = nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert data["n_w"] == 3
    k = data["cohorts"]["line_of_business"].tolist().index("lob_a")
    assert not data["obs_mask"][k, 1].any()
    assert data["obs_mask"][k, 0].any() and data["obs_mask"][k, 2].any()


def test_nn_company_data_regroups_lines(two_cohorts):
    """``nn_company_data`` is the multi-line layout (company, line, channel, n_w, n_d)
    used by ``nn_transformer_ml`` to attend across lines. Asserted to be a pure
    reshape of the flat contract, cohort by cohort, so the two entries provably train
    on identical numbers and any comparison between them is about the model only."""
    from ibnr.kernels.nn_contract import nn_company_data

    flat = nn_data(two_cohorts, loss_field="paid_loss", premium_field="earned_premium")
    data = nn_company_data(two_cohorts, loss_field="paid_loss", premium_field="earned_premium")
    assert data["x"].shape == (1, 2, 1, 3, 3)
    assert data["obs_mask"].shape == (1, 2, 3, 3)
    np.testing.assert_array_equal(data["line_mask"], [[True, True]])
    assert data["companies"].columns.tolist() == ["company_code"]
    # regrouping is a pure reshape of the flat contract
    for li, lob in enumerate(data["lob_levels"]):
        k = flat["cohorts"]["line_of_business"].tolist().index(lob)
        np.testing.assert_allclose(data["x"][0, li], flat["x"][k])
        np.testing.assert_array_equal(data["obs_mask"][0, li], flat["obs_mask"][k])
        np.testing.assert_allclose(data["latest_cum"][0, li], flat["latest_cum"][k])
        np.testing.assert_allclose(data["premium"][0, li], flat["premium"][k])


def test_nn_company_data_regroups_x_obs_and_forwards_level_fields(two_cohorts, backend_name):
    """``x_obs`` lands on the line axis exactly as ``x`` does, and ``level_fields``
    reaches the flat contract underneath.

    The case-reserve channel is punched out on one line only, so a regrouping that
    dropped the line index (or forgot the argument, making the channel differenced
    and the two lines' masks agree again) fails here rather than looking healthy.
    """
    from ibnr.kernels.nn_contract import nn_company_data

    df = two_cohorts.execute().copy()
    for col in ("origin_period", "eval_date"):
        df[col] = pd.to_datetime(df[col]).dt.date
    case = df[df["field"] == "paid_loss"].assign(field="case_reserve")
    case["value"] = case["value"] * 0.4
    # lob_a carries no case reserve at the first dev; lob_b carries all of it
    case = case[~((case["line_of_business"] == "lob_a") & (case["dev_lag"] == 12))]
    t = Triangle.from_long(
        pd.concat([df, case], ignore_index=True), measure="cumulative", backend=backend_name
    )
    kwargs = {
        "loss_field": "paid_loss",
        "feature_fields": ("case_reserve",),
        "level_fields": ("case_reserve",),
        "premium_field": "earned_premium",
    }
    flat = nn_data(t, **kwargs)
    data = nn_company_data(t, **kwargs)
    assert data["x_obs"].shape == (1, 2, 2, 3, 3)  # (company, line, channel, n_w, n_d)
    assert data["field_kinds"] == ("increment", "level")
    for li, lob in enumerate(data["lob_levels"]):
        k = flat["cohorts"]["line_of_business"].tolist().index(lob)
        np.testing.assert_array_equal(data["x_obs"][0, li], flat["x_obs"][k])
        np.testing.assert_array_equal(data["x_obs"][0, li, 0], data["obs_mask"][0, li])
    a, b = (data["lob_levels"].index(lob) for lob in ("lob_a", "lob_b"))
    assert not np.array_equal(data["x_obs"][0, a, 1], data["x_obs"][0, b, 1])
    # a level channel is usable at the successor of the hole; an increment is not
    assert data["x_obs"][0, a, 1, 0, 1]


def test_nn_company_data_masks_dropped_lines(backend_name):
    """A dropped line disappears from ``lob_levels`` entirely when it is the only
    company holding it, and ``line_mask`` shrinks accordingly. ``line_mask`` exists
    because in a multi-company batch other companies still need that slot."""
    from ibnr.kernels.nn_contract import nn_company_data

    bad_premium = {"lob_a": PREMIUM["lob_a"], "lob_b": np.array([400.0, -1.0, 440.0])}
    t = make_multiline_triangle(
        backend_name, {"lob_a": CUM, "lob_b": CUM * 2.0}, premium_by_lob=bad_premium
    )
    data = nn_company_data(t, loss_field="paid_loss", premium_field="earned_premium")
    # lob_b dropped -> absent from the levels entirely (single-company case)
    assert data["lob_levels"] == ["lob_a"]
    np.testing.assert_array_equal(data["line_mask"], [[True]])
    assert data["dropped"]["line_of_business"].tolist() == ["lob_b"]


# -- the display column: kept out of the KEY, kept in the IDENTITY ---------------


def test_display_column_rides_alongside_the_cohort_key(backend_name):
    """``company_name`` stays out of the key and comes back in ``display``.

    Keeping it out of the key is what stops two spellings of one company from
    becoming two cohorts. Carrying it in ``display`` is what lets a held-out
    scorer VERIFY the value instead of discarding it - without that, a pooled fit
    simply had no working route from cells the mart's three segments carry.
    """
    t = make_multiline_triangle(
        backend_name,
        {"lob_a": CUM, "lob_b": CUM * 2.0},
        premium_by_lob=PREMIUM,
        company_name="Acme Insurance Co",
    )
    data = nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert data["segment_columns"] == ("company_code", "line_of_business")
    assert list(data["cohorts"].columns) == ["company_code", "line_of_business"]
    assert list(data["display"].columns) == ["company_name"]
    assert len(data["display"]) == len(data["cohorts"])
    assert data["display"]["company_name"].tolist() == ["Acme Insurance Co"] * 2


def test_display_is_an_empty_frame_when_nothing_was_dropped(two_cohorts):
    """The no-display case is still row-aligned, so consumers need no branch."""
    data = nn_data(two_cohorts, loss_field="paid_loss", premium_field="earned_premium")
    assert data["display"].shape == (2, 0)
    assert data["segment_columns"] == ("company_code", "line_of_business")


def test_cohort_identities_merge_key_and_display(backend_name):
    """``cohort_identities`` is the full identity, which is what ``cohorts()`` hands
    back - the key a caller can filter on plus the label the triangle carried."""
    from ibnr.kernels.nn_contract import cohort_identities

    t = make_multiline_triangle(
        backend_name,
        {"lob_a": CUM, "lob_b": CUM * 2.0},
        premium_by_lob=PREMIUM,
        company_name="Acme Insurance Co",
    )
    data = nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert cohort_identities(data) == [
        {
            "company_code": "0001",
            "line_of_business": lob,
            "company_name": "Acme Insurance Co",
        }
        for lob in ("lob_a", "lob_b")
    ]


def _two_spellings(backend_name) -> Triangle:
    """One (company, line) cohort whose ``company_name`` differs by ORIGIN.

    Deliberately disjoint cells: the two spellings never share an
    (origin, dev) cell, so the duplicate-cell guard cannot fire and the collapse
    would be completely silent.
    """
    rows = []
    for w in range(3):
        name = "ACME" if w < 2 else "Acme Insurance Co"
        for dev in range(3 - w):
            for field, value in (
                ("paid_loss", float(CUM[w, dev])),
                ("earned_premium", float(PREMIUM["lob_a"][w])),
            ):
                rows.append(
                    {
                        "company_code": "0001",
                        "company_name": name,
                        "line_of_business": "lob_a",
                        "origin_period": dt.date(2010 + w, 1, 1),
                        "dev_lag": 12 * (dev + 1),
                        "eval_date": dt.date(2010 + w + dev, 12, 31),
                        "field": field,
                        "value": value,
                    }
                )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)


def test_a_display_column_that_varies_within_a_key_is_refused(backend_name):
    """Two spellings under one cohort key must RAISE, not merge silently.

    The two spellings' cells are disjoint here, so the duplicate-cell guard
    cannot see the collapse: without this refusal the two would be pooled into a
    single grid and the contract would look healthy. That is why the narrowing
    which fixes the pooled-fit scoring gap is only legitimate with this check
    beside it - otherwise the narrowing is safe by accident, invisibly.
    """
    with pytest.raises(ValueError) as excinfo:
        nn_data(_two_spellings(backend_name), loss_field="paid_loss")
    message = str(excinfo.value)
    assert "company_name" in message
    assert "not determined by the cohort key" in message
    assert "ACME" in message and "Acme Insurance Co" in message


def test_without_the_refusal_the_two_spellings_would_be_one_grid(backend_name):
    """What the refusal is protecting: keeping the column makes them TWO cohorts.

    Pins the size of the mistake rather than only its message - one grid where
    the triangle holds two distinct cohorts.
    """
    kept = nn_data(
        _two_spellings(backend_name),
        loss_field="paid_loss",
        segment_columns=("company_code", "company_name", "line_of_business"),
    )
    assert len(kept["cohorts"]) == 2
    assert kept["display"].shape == (2, 0)


def test_the_refusal_also_covers_an_explicit_narrow_segment_columns(backend_name):
    """``segment_columns=`` merged cohorts silently too, so it gets the same rule."""
    t = make_multiline_triangle(
        backend_name,
        {"lob_a": CUM, "lob_b": CUM * 2.0},
        premium_by_lob=PREMIUM,
    )
    # line_of_business plainly varies within company_code alone
    with pytest.raises(ValueError, match="not determined by the cohort key"):
        nn_data(
            t,
            loss_field="paid_loss",
            premium_field="earned_premium",
            segment_columns=("company_code",),
        )


def test_nn_company_data_applies_the_rule_at_the_company_level(backend_name):
    """The key narrows a SECOND time there (cohort minus LOB), so the rule runs again.

    A ``company_name`` that is 1:1 with (company_code, line_of_business) can still
    vary within the COMPANY, which would put two companies on one row of the line
    axis - and the flat contract, whose key still holds the line, cannot see it.
    """
    from ibnr.kernels.nn_contract import nn_company_data

    t = make_multiline_triangle(
        backend_name,
        {"lob_a": CUM, "lob_b": CUM * 2.0},
        premium_by_lob=PREMIUM,
        company_name={"lob_a": "ACME", "lob_b": "Acme Insurance Co"},
    )
    # fine at cohort level - one name per (company, line) - and refused above it
    flat = nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert flat["display"]["company_name"].tolist() == ["ACME", "Acme Insurance Co"]
    with pytest.raises(ValueError, match="not determined by the cohort key"):
        nn_company_data(t, loss_field="paid_loss", premium_field="earned_premium")


def test_nn_company_data_carries_the_company_level_display(backend_name):
    from ibnr.kernels.nn_contract import cohort_identities, nn_company_data

    t = make_multiline_triangle(
        backend_name,
        {"lob_a": CUM, "lob_b": CUM * 2.0},
        premium_by_lob=PREMIUM,
        company_name="Acme Insurance Co",
    )
    data = nn_company_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert data["segment_columns"] == ("company_code",)
    assert cohort_identities(data, key="companies") == [
        {"company_code": "0001", "company_name": "Acme Insurance Co"}
    ]


def test_cutoff_masks_partition():
    """Calendar-cutoff augmentation splits the observed cells into context (on/before
    the cutoff diagonal) and target (after), and the split must be a true partition:
    disjoint and covering. Overlap would be label leakage; a gap would silently
    discard training signal. Synthetic cutoffs on past diagonals are how one small
    triangle yields several training examples."""
    obs = upper_mask(3, 3)
    cal = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    context, target = cutoff_masks(obs, cal, cutoff=2)
    np.testing.assert_array_equal(context, cal <= 2)
    np.testing.assert_array_equal(target, obs & (cal > 2))
    assert not (context & target).any()
    np.testing.assert_array_equal(context | target, obs)


# -- the pooled origin axis has to be one dev step apart --------------------------


def _pooled_origin_triangle(
    backend_name: str,
    origin_years,
    dev_lags,
    *,
    dev_grain: str = "Y",
    through: str = "2013-12-31",
) -> Triangle:
    """One cohort on a caller-chosen origin/dev geometry, with premium on every cell.

    ``nn_data`` builds ONE origin axis shared by every cohort, so the geometry that
    matters is the pooled set of origins and the declared dev grain - which is what
    this varies. Cells past ``through`` are absent, the usual run-off shape.
    """
    limit = dt.date.fromisoformat(through)
    rows = []
    for year in origin_years:
        origin = dt.date(year, 1, 1)
        for lag in dev_lags:
            ev = (pd.Timestamp(origin) + pd.DateOffset(months=lag) - pd.Timedelta(days=1)).date()
            if ev > limit:
                continue
            for field, value in (("paid_loss", 100.0 + lag), ("earned_premium", 1000.0)):
                rows.append(
                    {
                        "company_code": "0001",
                        "line_of_business": "lob_a",
                        "origin_period": origin,
                        "dev_lag": lag,
                        "eval_date": ev,
                        "field": field,
                        "value": value,
                    }
                )
    return Triangle.from_long(
        pd.DataFrame(rows),
        measure="cumulative",
        origin_grain="Y",
        dev_grain=dev_grain,
        backend=backend_name,
    )


def test_nn_data_refuses_a_gap_in_the_pooled_origin_axis(backend_name):
    """Accident years 2010, 2012, 2013 - 2011 is missing from every cohort.

    ``cal_idx = w + d + 1`` is the evaluation date the validation split, the cutoff
    augmentation and the held-out cutoff all slice on, and it is calendar time only
    while one origin step is one dev step. With 2011 absent it is not: measured on
    this triangle, the three cells that really sit on the 2013-12-31 diagonal get
    cal_idx 4, 3 and 3, so the split holds out one of them and trains on the other
    two - training on data from the diagonal it is scored on.
    """
    t = _pooled_origin_triangle(backend_name, (2010, 2012, 2013), (12, 24, 36, 48))
    with pytest.raises(ValueError, match="origin axis") as exc:
        nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert "2010-01-01" in str(exc.value) and "2012-01-01" in str(exc.value)


def test_nn_data_refuses_annual_origins_on_a_quarterly_dev_grain(backend_name):
    """The same fault without any gap: annual origins, quarterly development.

    Every origin step is 12 months and every dev step is 3, so moving one row down
    the grid advances four diagonals while ``cal_idx`` advances one. The origins are
    contiguous and the ages are on the grain; the geometry is still not one the
    calendar index can describe.
    """
    t = _pooled_origin_triangle(
        backend_name, (2010, 2011, 2012, 2013), [3 * k for k in range(1, 17)], dev_grain="Q"
    )
    with pytest.raises(ValueError, match="origin axis"):
        nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
