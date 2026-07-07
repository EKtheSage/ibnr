"""kernels.nn_contract: the NN data contract. Pure numpy — no torch needed."""

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
    data = nn_data(two_cohorts, loss_field="paid_loss", premium_field="earned_premium")
    assert data["x"].shape == (2, 1, 3, 3)
    assert data["obs_mask"].shape == (2, 3, 3)
    assert data["n_w"] == 3 and data["n_d"] == 3
    assert data["lob_levels"] == ["lob_a", "lob_b"]
    np.testing.assert_array_equal(data["lob_idx"], [0, 1])
    np.testing.assert_array_equal(data["obs_mask"][0], upper_mask(3, 3))

    # hand-computed incremental loss ratios for lob_a
    incr = np.array([[100.0, 50.0, 25.0], [110.0, 55.0, 0.0], [120.0, 0.0, 0.0]])
    want = incr / PREMIUM["lob_a"][:, None]
    want[~upper_mask(3, 3)] = 0.0
    np.testing.assert_allclose(data["x"][0, 0], want)
    # lob_b: double the losses, double the premium -> identical ratios
    np.testing.assert_allclose(data["x"][1, 0], want)

    np.testing.assert_allclose(data["latest_cum"][0], [175.0, 165.0, 120.0])
    np.testing.assert_array_equal(data["latest_dev"][0], [3, 2, 1])
    np.testing.assert_allclose(data["premium"][0], PREMIUM["lob_a"])

    cal = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(data["cal_idx"], cal)
    assert data["cohorts"].columns.tolist() == ["company_code", "line_of_business"]
    assert len(data["dropped"]) == 0


def test_nn_data_feature_channels(two_cohorts, backend_name):
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
    with pytest.raises(ValueError, match="duplicate fields"):
        nn_data(two_cohorts, loss_field="paid_loss", feature_fields=("paid_loss",))


def test_nn_data_gap_predecessor_is_unobserved(backend_name):
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
    bad_premium = {"lob_a": PREMIUM["lob_a"], "lob_b": np.array([400.0, -1.0, 440.0])}
    t = make_multiline_triangle(
        backend_name, {"lob_a": CUM, "lob_b": CUM * 2.0}, premium_by_lob=bad_premium
    )
    data = nn_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert data["cohorts"]["line_of_business"].tolist() == ["lob_a"]
    assert data["dropped"]["line_of_business"].tolist() == ["lob_b"]
    assert "premium" in data["dropped"]["reason"].iloc[0]


def test_nn_data_rejects_restated_cells(backend_name):
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
    with pytest.raises(ValueError, match="cumulative"):
        nn_data(two_cohorts.to_incremental(), loss_field="paid_loss")


def test_nn_data_obs_mask_matches_as_of_slice(two_cohorts):
    sliced = two_cohorts.as_of(dt.date(2011, 12, 31))  # keeps calendar diagonals 1-2
    data = nn_data(sliced, loss_field="paid_loss", premium_field="earned_premium")
    # the 2012 origin has no rows on/before the cutoff, so the grid shrinks
    assert data["n_w"] == 2 and data["n_d"] == 2
    np.testing.assert_array_equal(data["obs_mask"][0], [[True, True], [True, False]])
    np.testing.assert_allclose(data["latest_cum"][0], [150.0, 110.0])
    np.testing.assert_array_equal(data["latest_dev"][0], [2, 1])


def test_nn_data_origin_missing_from_one_cohort(backend_name):
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


def test_cutoff_masks_partition():
    obs = upper_mask(3, 3)
    cal = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    context, target = cutoff_masks(obs, cal, cutoff=2)
    np.testing.assert_array_equal(context, cal <= 2)
    np.testing.assert_array_equal(target, obs & (cal > 2))
    assert not (context & target).any()
    np.testing.assert_array_equal(context | target, obs)
