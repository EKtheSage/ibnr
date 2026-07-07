import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.contract import realized_values, stan_data


def _cohort_triangle(extra_segment_rows=False):
    rows = []
    losses = {
        ("2020-01-01", 12, "2020-12-31"): 100.0,
        ("2020-01-01", 24, "2021-12-31"): 150.0,
        ("2020-01-01", 36, "2022-12-31"): 175.0,
        ("2021-01-01", 12, "2021-12-31"): 110.0,
        ("2021-01-01", 24, "2022-12-31"): 165.0,
        ("2022-01-01", 12, "2022-12-31"): 120.0,
    }
    premium = {"2020-01-01": 500.0, "2021-01-01": 550.0, "2022-01-01": 600.0}
    for (o, dev, e), v in losses.items():
        rows.append(("co1", o, dev, e, "paid_loss", v))
        rows.append(("co1", o, dev, e, "earned_premium", premium[o]))
    if extra_segment_rows:
        rows.append(("co2", "2020-01-01", 12, "2020-12-31", "paid_loss", 1.0))
    df = pd.DataFrame(
        rows, columns=["company", "origin_period", "dev_lag", "eval_date", "field", "value"]
    )
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    return Triangle.from_long(df)


def test_stan_data_mapping():
    t = _cohort_triangle()
    c = stan_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert c["len_data"] == 6
    assert c["n_w"] == 3 and c["n_d"] == 3
    # sorted by (w, d)
    assert c["w"].tolist() == [1, 1, 1, 2, 2, 3]
    assert c["d"].tolist() == [1, 2, 3, 1, 2, 1]
    # prev_idx: 1-based pointer to (w-1, d); 0 for the first origin
    assert c["prev_idx"].tolist() == [0, 0, 0, 1, 2, 4]
    np.testing.assert_allclose(c["loss"], [100, 150, 175, 110, 165, 120])
    np.testing.assert_allclose(c["logloss"], np.log(c["loss"]))
    np.testing.assert_allclose(c["premium"], [500, 550, 600])
    np.testing.assert_allclose(c["logprem"], np.log(np.array([500, 550, 600]))[c["w"] - 1])
    assert [o.year for o in c["origin_periods"]] == [2020, 2021, 2022]
    assert c["dev_grain_months"] == 12


def test_stan_data_rejects_multi_cohort():
    t = _cohort_triangle(extra_segment_rows=True)
    with pytest.raises(ValueError, match="multiple segment combinations"):
        stan_data(t, loss_field="paid_loss")


def test_stan_data_rejects_nonpositive():
    t = _cohort_triangle()
    bad = t.with_expr(t.expr.mutate(value=t.expr.value - 100.0))
    with pytest.raises(ValueError, match="non-positive"):
        stan_data(bad, loss_field="paid_loss")


def test_stan_data_rejects_incremental():
    t = _cohort_triangle().select_fields("paid_loss").to_incremental()
    with pytest.raises(ValueError, match="cumulative"):
        stan_data(t, loss_field="paid_loss")


def test_realized_values():
    t = _cohort_triangle()
    c = stan_data(t, loss_field="paid_loss", premium_field="earned_premium")
    realized = realized_values(t, loss_field="paid_loss", dev_lag=36, origins=c["origin_periods"])
    np.testing.assert_allclose(realized[0], 175.0)
    assert np.isnan(realized[1]) and np.isnan(realized[2])
