"""kernels.contract: Triangle -> the standardized Stan ``data`` dict.

The Stan ``data`` block is the data contract (CLAUDE.md design decision 3): every
Bayesian backend - cmdstanpy, NumPyro, PyMC - consumes the identical dict, so a
bug here is a bug in every model at once and would show up as a spurious
cross-backend parity failure rather than as a data bug. Nothing here needs a
sampler, so these run in the fast suite.

What is pinned:
- the (w, d) sort order and 1-based Stan indexing;
- ``prev_idx``, the pointer to the same-dev cell in the previous origin, which the
  cross-classified models (Meyers CRC/CCL/CSR) walk in a single forward pass;
- ``premium`` being per-origin (the booked value) while ``logprem`` is broadcast
  per-observation;
- the guard rails: one cohort only, positive losses only (the family is lognormal),
  cumulative only.
"""

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.contract import realized_values, stan_data


def _cohort_triangle(extra_segment_rows=False):
    """One company, 3 origins x up-to-3 devs, paid loss + earned premium.

    Premium is repeated on every cell of an origin (as the mart carries it); the
    contract must collapse it back to one value per origin. ``extra_segment_rows``
    adds a second company so the single-cohort guard can be exercised.
    """
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
    """Full field-by-field spec of the emitted dict - the reference a reviewer can
    read to know what the Stan ``data`` block will receive."""
    t = _cohort_triangle()
    c = stan_data(t, loss_field="paid_loss", premium_field="earned_premium")
    assert c["len_data"] == 6
    assert c["n_w"] == 3 and c["n_d"] == 3
    # sorted by (w, d), 1-based: origin-major, so all of origin 1 precedes origin 2
    assert c["w"].tolist() == [1, 1, 1, 2, 2, 3]
    assert c["d"].tolist() == [1, 2, 3, 1, 2, 1]
    # prev_idx: 1-based pointer to (w-1, d); 0 for the first origin. Because rows
    # are (w, d)-sorted, prev_idx[i] < i+1 always -> the CCL/CSR mu recursion is a
    # single forward pass with no lookahead. Row 4 is (w=2,d=1) -> points at row 1.
    assert c["prev_idx"].tolist() == [0, 0, 0, 1, 2, 4]
    np.testing.assert_allclose(c["loss"], [100, 150, 175, 110, 165, 120])
    np.testing.assert_allclose(c["logloss"], np.log(c["loss"]))
    # premium: one per origin (booked value at its latest eval), NOT one per cell
    np.testing.assert_allclose(c["premium"], [500, 550, 600])
    # logprem: the same premium broadcast back out per observation via w
    np.testing.assert_allclose(c["logprem"], np.log(np.array([500, 550, 600]))[c["w"] - 1])
    assert [o.year for o in c["origin_periods"]] == [2020, 2021, 2022]
    assert c["dev_grain_months"] == 12


def test_stan_data_rejects_multi_cohort():
    """Two companies in one triangle must raise, not silently pool. Fitting a
    single-cohort model to interleaved companies would produce plausible-looking
    nonsense, so the contract refuses and tells the caller to filter first."""
    t = _cohort_triangle(extra_segment_rows=True)
    with pytest.raises(ValueError, match="multiple segment combinations"):
        stan_data(t, loss_field="paid_loss")


def test_stan_data_rejects_nonpositive():
    """Non-positive losses must raise: the family is lognormal, so ``logloss``
    would be -inf/NaN and the sampler would fail far from the cause. Real Schedule P
    triangles do contain non-positive cumulative paid for small/immature cells."""
    t = _cohort_triangle()
    bad = t.with_expr(t.expr.mutate(value=t.expr.value - 100.0))
    with pytest.raises(ValueError, match="non-positive"):
        stan_data(bad, loss_field="paid_loss")


def test_stan_data_rejects_incremental():
    """An incremental triangle must raise. The cross-classified models are written
    on cumulative losses; handing them increments would fit quietly and wrongly."""
    t = _cohort_triangle().select_fields("paid_loss").to_incremental()
    with pytest.raises(ValueError, match="cumulative"):
        stan_data(t, loss_field="paid_loss")


def test_realized_values():
    """Scoring outcomes align to the training grid by origin, with NaN where the
    outcome has not emerged yet.

    A retrospective scores model draws against the realized value at some settled
    dev lag. Positional alignment to ``origin_periods`` (rather than an implicit
    order) plus explicit NaN is what lets the harness drop unscoreable origins
    instead of silently shifting outcomes onto the wrong accident years.
    """
    t = _cohort_triangle()
    c = stan_data(t, loss_field="paid_loss", premium_field="earned_premium")
    realized = realized_values(t, loss_field="paid_loss", dev_lag=36, origins=c["origin_periods"])
    # only the 2020 origin has reached dev 36 in this triangle
    np.testing.assert_allclose(realized[0], 175.0)
    assert np.isnan(realized[1]) and np.isnan(realized[2])
