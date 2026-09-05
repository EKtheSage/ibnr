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


def test_every_contract_records_the_cohort_and_fields_it_was_built_from():
    """``segment``/``fields`` are what let a scorer refuse another company's
    cells. ``(w, d)`` alone does not identify a cell, so a contract that cannot
    say which cohort it describes cannot be defended against one.

    All three builders are checked together on purpose. The compartmental one
    read its identity off the PIVOTED frame, which indexes on (origin_period,
    dev_lag) and has therefore already dropped the segment columns - so every
    segmented compartmental fit raised ``KeyError: 'company'``, and a test
    covering only ``stan_data`` would never have found it.
    """
    from ibnr.kernels.contract import compartmental_stan_data, odp_stan_data

    tri = _cohort_triangle()
    paid = stan_data(tri, loss_field="paid_loss", premium_field="earned_premium")
    assert paid["segment"] == {"company": "co1"}
    assert paid["fields"] == ("paid_loss",)

    odp = odp_stan_data(tri, loss_field="paid_loss", premium_field="earned_premium")
    assert odp["segment"] == {"company": "co1"}
    assert odp["fields"] == ("paid_loss",)

    # compartmental needs a second field on the identical cells
    df = tri.execute()
    reported = df[df["field"] == "paid_loss"].copy()
    reported["field"] = "reported_loss"
    reported["value"] = reported["value"] * 1.25
    joint = Triangle.from_long(pd.concat([df, reported], ignore_index=True))

    comp = compartmental_stan_data(
        joint,
        paid_field="paid_loss",
        reported_field="reported_loss",
        premium_field="earned_premium",
    )
    assert comp["segment"] == {"company": "co1"}
    assert comp["fields"] == ("paid_loss", "reported_loss")

    # `fields` is what was READ; `models` is what carries a likelihood. They
    # differ here because compartmental models outstanding = reported - paid, so
    # a raw reported-loss cell is not something this fit can score even though
    # reported_loss is one of its source fields.
    assert comp["models"] == ("paid_loss", "outstanding")
    assert paid["models"] == ("paid_loss",)
    for c in (paid, odp, comp):
        assert c["measure"] == "cumulative"


def _joint_contract():
    """The compartmental (delta-stacked) contract over the shared cohort
    fixture. reported = 1.25 * paid, so outstanding = 0.25 * paid - every OS
    value differs from its paid twin at the same (w, d), which is what makes a
    predecessor handed across blocks detectable rather than plausible."""
    from ibnr.kernels.contract import compartmental_stan_data

    df = _cohort_triangle().execute()
    reported = df[df["field"] == "paid_loss"].copy()
    reported["field"] = "reported_loss"
    reported["value"] = reported["value"] * 1.25
    joint = Triangle.from_long(pd.concat([df, reported], ignore_index=True))
    return compartmental_stan_data(
        joint,
        paid_field="paid_loss",
        reported_field="reported_loss",
        premium_field="earned_premium",
    )


def test_training_index_carries_delta_for_the_compartmental_contract():
    """compartmental stacks outstanding (delta=0) and paid (delta=1) over
    IDENTICAL (w, d), so (w, d) alone cannot identify a cell - an earlier
    version of ``training_index`` refused this contract outright for exactly
    that reason. The delta-aware index restores identity: each block keeps its
    OWN predecessors, keyed on (delta, w, d-1).

    The bug the refusal used to guard against - an outstanding cell handed the
    PAID predecessor, same sign, same order of magnitude - is asserted away
    here cell by cell, on a fixture where the two candidates always differ.
    """
    from ibnr.kernels.holdout import DeltaCellIndex, training_index

    comp = _joint_contract()

    # the duplication is real: every (w, d) appears twice
    pairs = list(zip(comp["w"].tolist(), comp["d"].tolist(), strict=True))
    assert len(pairs) == 2 * len(set(pairs))

    idx = training_index(comp)
    assert isinstance(idx, DeltaCellIndex)
    np.testing.assert_array_equal(idx.delta, comp["delta"])
    np.testing.assert_array_equal(idx.value, comp["loss"])
    np.testing.assert_allclose(idx.premium, np.asarray(comp["premium"])[idx.w - 1])

    row_of = {
        (int(dl), int(w), int(d)): i
        for i, (dl, w, d) in enumerate(zip(comp["delta"], comp["w"], comp["d"], strict=True))
    }
    checked_across_blocks = 0
    for i in range(idx.n_cells):
        dl, w, d = int(idx.delta[i]), int(idx.w[i]), int(idx.d[i])
        if d == 1:
            # first dev: a cumulative value IS its increment, and OS(0) = 0
            assert idx.prev_value[i] == 0.0
            continue
        own_block = float(comp["loss"][row_of[(dl, w, d - 1)]])
        other_block = float(comp["loss"][row_of[(1 - dl, w, d - 1)]])
        assert idx.prev_value[i] == own_block
        # the fixture must be able to tell the two candidates apart, or the
        # assertion above is vacuous
        assert own_block != other_block
        checked_across_blocks += 1
    assert checked_across_blocks > 0


def test_training_index_still_refuses_duplicates_without_a_delta_axis():
    """The delta axis is the LICENSE for duplicate (w, d), not an amnesty.

    A contract that duplicates cells while carrying no block indicator is
    genuinely ambiguous - there is nothing to key the predecessor on - and it
    must stay refused, or the silent-wrong-predecessor bug the old
    NotImplementedError guarded against comes back without a trace."""
    from ibnr.kernels.holdout import training_index

    comp = _joint_contract()
    stripped = {k: v for k, v in comp.items() if k != "delta"}
    with pytest.raises(ValueError, match=r"duplicate \(w, d\)"):
        training_index(stripped)


def test_premium_from_another_cohort_is_refused():
    """The single-cohort guards run on the LOSS rows, because that is the frame
    they are built from. Premium is a different field and was unchecked.

    A triangle with one company's losses and two companies' premium therefore
    passed every guard, and then picked up whichever premium row sorted first -
    measured, half the origins scored against another cohort's exposure, 7.8x
    out, while ``contract["segment"]`` still named the right cohort. Wrong
    premium is wrong ``mu`` for the whole lognormal family.
    """
    df = _cohort_triangle().execute()
    intruder = df[(df["field"] == "earned_premium") & (df["dev_lag"] == 12)].copy()
    intruder["company"] = "co2"
    intruder["value"] = 7777.0
    tri = Triangle.from_long(pd.concat([df, intruder], ignore_index=True))

    # the loss rows are still single-cohort, so the existing guard says nothing
    assert len(tri.select_fields("paid_loss").execute().drop_duplicates(["company"])) == 1

    with pytest.raises(ValueError, match="spans multiple segment combinations"):
        stan_data(tri, loss_field="paid_loss", premium_field="earned_premium")


def test_premium_that_is_entirely_the_wrong_cohort_is_refused():
    """The harder half, and the one a uniqueness check cannot see.

    A premium set that is ENTIRELY another company's is perfectly consistent
    with itself: one cohort, one row per origin, nothing contradictory. It
    passes every "is this internally coherent" test and is completely wrong.
    Measured before the fix: a contract recording ``segment={'company': 'co1'}``
    while carrying co2's premium of 7777 on every origin.

    The check has to be that the premium cohort MATCHES the loss cohort. This is
    the same lesson as the held-out cells - consistency is not identity.
    """
    df = _cohort_triangle().execute()
    losses = df[df["field"] == "paid_loss"]
    premium = df[df["field"] == "earned_premium"].copy()
    premium["company"] = "co2"  # ALL the premium belongs to another company
    tri = Triangle.from_long(pd.concat([losses, premium], ignore_index=True))

    # internally consistent: exactly one premium cohort, one row per origin
    prem_rows = tri.select_fields("earned_premium").execute()
    assert set(prem_rows["company"]) == {"co2"}

    with pytest.raises(ValueError, match="belongs to cohort"):
        stan_data(tri, loss_field="paid_loss", premium_field="earned_premium")


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


# -- grid geometry: the origin axis and the dev-age boundary ---------------------


def _month_end(origin: str, dev_lag: int):
    """Last day of the month that ``origin + dev_lag`` months lands in."""
    return (pd.Timestamp(origin) + pd.DateOffset(months=dev_lag) - pd.Timedelta(days=1)).date()


def _geometry_triangle(origin_years, dev_lags, *, lines=("lob_a",), through="2021-03-31"):
    """One company's cells on a caller-chosen origin/dev geometry.

    Carries paid_loss, reported_loss and earned_premium on every cell, and takes a
    second line on request, so the same triangle can be handed to all six contract
    builders. Cells whose evaluation date is past ``through`` are simply absent,
    which is what makes the shape a run-off staircase.
    """
    rows = []
    limit = pd.Timestamp(through).date()
    for lob in lines:
        for year in origin_years:
            origin = f"{year}-01-01"
            for lag in dev_lags:
                ev = _month_end(origin, lag)
                if ev > limit:
                    continue
                for field, value in (
                    ("paid_loss", 100.0 + lag),
                    ("reported_loss", 125.0 + lag),
                    ("earned_premium", 1000.0),
                ):
                    rows.append(
                        {
                            "company": "co1",
                            "line_of_business": lob,
                            "origin_period": pd.Timestamp(origin).date(),
                            "dev_lag": lag,
                            "eval_date": ev,
                            "field": field,
                            "value": value,
                        }
                    )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


def test_stan_data_refuses_a_gapped_origin_axis():
    """An origin axis with a hole in it, through the cross-classified contract.

    The origin index ``w`` and the dev index ``d`` are one shared calendar clock:
    stepping from origin w to w+1 is meant to be one elapsed development period.
    With 2020 absent, origin 2021 sits two years after 2019 but one index step
    after it, so ``prev_idx`` links cells two calendar years apart as if they were
    neighbours. The triangle is refused rather than quietly re-indexed.
    """
    t = _geometry_triangle((2019, 2021, 2022), (12, 24, 36), through="2022-12-31")
    with pytest.raises(ValueError, match="origin axis") as exc:
        stan_data(t, loss_field="paid_loss")
    assert "2019-01-01" in str(exc.value) and "2021-01-01" in str(exc.value)
