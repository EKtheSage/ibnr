"""kernels.nn_contract: the NN data contract. Pure numpy - no torch needed.

The neural entries consume dense masked grids rather than long rows: features are
incremental loss ratios (increment / earned premium) on a (cohort, channel,
n_w, n_d) tensor, paired with an ``obs_mask`` marking which cells the network is
allowed to attend to. Normalizing by premium is what makes triangles from
companies of wildly different size poolable - Schedule P triangles are small and
overfitting is the central risk, so training always spans many company x line
cohorts.

The file matters because these are the failure modes that quietly corrupt a
training set rather than crashing it: a cell whose predecessor is missing has no
usable increment and must be masked out; a cohort with non-positive premium
produces meaningless ratios and must be dropped (reported, not hidden); the
anchor used for the autoregressive rollout (``latest_cum``/``latest_dev``) must
stay the raw cumulative even where the mask says the increment is unusable.

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


def test_nn_data_rejects_duplicate_fields(two_cohorts):
    """Listing the target field as a feature must raise - it would hand the network
    the answer as an input channel."""
    with pytest.raises(ValueError, match="duplicate fields"):
        nn_data(two_cohorts, loss_field="paid_loss", feature_fields=("paid_loss",))


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
