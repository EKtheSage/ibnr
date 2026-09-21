"""kernels.nn_features: the engineered example tensors of the tlrn entry.

Pure numpy over a company contract - no torch, so this runs in the core CI leg
beside ``test_nn_contract.py``.

What this file protects. The feature builder turns a company's triangles into
one training example per (company, accident year), and every number in it is
engineered by hand from cells the cutoff diagonal leaves visible. Three things
can go wrong without any shape error and without the fit ever complaining:

* a statistic estimated over the whole grid instead of the visible part, which
  hands the network the answer it is being asked to predict;
* an index convention off by one - the latest visible lag, the number of steps
  ahead, which development step a factor belongs to;
* a chain ladder factor computed from pairs of cells one of which is in the
  future.

So the expected values here are computed a second time, from the fixture's own
matrices with plain loops, rather than by calling the module. The one test that
would catch all three at once is
:func:`test_hidden_cells_do_not_reach_the_inputs`: every cell past the cutoff is
multiplied by 1.37 and shifted by 0.123, and every returned array except the
targets has to come back bit for bit the same.

Both ibis backends, because the contract underneath is built from a Triangle.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.nn_contract import nn_company_data
from ibnr.kernels.nn_features import origin_cl_log_factors, pooled_cl_factors, tlrn_features

START = 2000
LOBS = ("lob_a", "lob_b")
#: the three loss fields the fixture emits, and the contract channel each becomes
LOSS_FIELDS = ("paid_loss", "incurred_loss", "case_reserve")


def study_arrays(n_companies: int = 3, n_w: int = 6, seed: int = 0) -> dict:
    """Cumulative paid, cumulative incurred, case reserve and premium.

    ``(n_c, n_lines, n_w, n_d)`` matrices for a full square: paid increments
    decay with development, incurred is paid divided by a rising reported share
    so ``incurred >= paid > 0``, and ``case = incurred - paid`` is a real
    outstanding balance rather than a second copy of the loss. Premium is
    constant over development, one value per (company, line, origin), and
    company size grows with the company index so the size strata and the
    premium-normalised features have something to separate.

    Returned as matrices, not as a Triangle, because every test below
    recomputes its expected values from them.
    """
    rng = np.random.default_rng(seed)
    n_d, n_l = n_w, len(LOBS)
    decay = np.exp(np.linspace(-0.6, -3.0, n_d))
    reported_share = np.linspace(0.55, 1.0, n_d)
    paid = np.zeros((n_companies, n_l, n_w, n_d))
    premium = np.zeros((n_companies, n_l, n_w))
    for ci in range(n_companies):
        for li in range(n_l):
            p = 1000.0 * (ci + 1) * (1.0 + 0.5 * li)
            premium[ci, li] = p
            incr = 0.6 * p * decay[None, :] * rng.lognormal(0.0, 0.15, size=(n_w, n_d))
            paid[ci, li] = np.cumsum(incr, axis=1)
    incurred = paid / reported_share[None, None, None, :]
    return {"paid": paid, "incurred": incurred, "case": incurred - paid, "premium": premium}


def study_triangle(backend_name, arrays: dict | None = None, *, lines: dict | None = None):
    """The long frame for :func:`study_arrays`, as a cumulative Triangle.

    Yearly grain, origin ``w`` = Jan 1 of ``START + w``, dev index ``d`` =
    dev_lag ``12 * (d + 1)``, evaluation date Dec 31 of ``START + w + d`` - the
    same conventions as ``tests/conftest.py``'s shared builder. A NaN in a loss
    matrix is simply not emitted, so a hole in one field leaves the others
    alone. ``lines`` maps a company index to the line indices it writes, for the
    one test that needs a company not writing every line.
    """
    a = arrays if arrays is not None else study_arrays()
    n_c, n_l, n_w, n_d = a["paid"].shape
    rows: list[dict] = []
    for ci in range(n_c):
        for li in range(n_l):
            if lines is not None and li not in lines.get(ci, range(n_l)):
                continue
            for w in range(n_w):
                for d in range(n_d):
                    cell = {
                        "company_code": f"{ci + 1:04d}",
                        "line_of_business": LOBS[li],
                        "origin_period": dt.date(START + w, 1, 1),
                        "dev_lag": 12 * (d + 1),
                        "eval_date": dt.date(START + w + d, 12, 31),
                    }
                    for field, key in zip(LOSS_FIELDS, ("paid", "incurred", "case"), strict=True):
                        value = a[key][ci, li, w, d]
                        if np.isnan(value):
                            continue
                        rows.append({**cell, "field": field, "value": float(value)})
                    rows.append(
                        {
                            **cell,
                            "field": "earned_premium",
                            "value": float(a["premium"][ci, li, w]),
                        }
                    )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative", backend=backend_name)


def company_contract(backend_name, arrays=None, *, lines=None, with_incurred=True) -> dict:
    """The ``nn_company_data`` contract the feature builder consumes."""
    t = study_triangle(backend_name, arrays, lines=lines)
    if not with_incurred:
        return nn_company_data(t, loss_field="paid_loss")
    return nn_company_data(
        t,
        loss_field="paid_loss",
        feature_fields=("incurred_loss", "case_reserve"),
        level_fields=("case_reserve",),
    )


def build(contract, cutoff, target_lo, target_hi, *, incurred=True):
    kwargs = {"incurred_field": "incurred_loss", "case_field": "case_reserve"} if incurred else {}
    return tlrn_features(
        contract, cutoff=cutoff, target_lo=target_lo, target_hi=target_hi, **kwargs
    )


def increment_ratios(arrays: dict) -> np.ndarray:
    """``(n_c, n_l, n_w, n_d)`` incremental paid over premium, the target channel.

    Recomputed here rather than read off the contract: the first development
    step's increment is the cumulative itself, which is the convention the
    contract uses and the one a test must not take on trust.
    """
    paid = arrays["paid"]
    incr = np.diff(paid, axis=3, prepend=0.0)
    return incr / arrays["premium"][:, :, :, None]


def clip_z(value, mu, sd, clamp=5.0):
    return float(np.clip((value - mu) / sd, -clamp, clamp))


def sample_stats(values) -> tuple[float, float]:
    """R's ``mean``/``sd`` pair with the module's degenerate-case rules."""
    v = np.asarray([x for x in np.asarray(values).reshape(-1) if np.isfinite(x)])
    if v.size < 2:
        return 0.0, 1.0
    s = float(v.std(ddof=1))
    return float(v.mean()), (s if np.isfinite(s) and s > 1e-8 else 1.0)


# -- shapes, order and the example axis -----------------------------------------


def test_shapes_and_feature_order(backend_name):
    """Eight features without incurred, thirteen with, in one fixed order.

    The order is part of the contract with the network: the input projection is
    a single linear layer, so two features swapped is a silently different model
    that still trains.
    """
    arrays = study_arrays()
    thirteen = build(company_contract(backend_name, arrays), 4, 5, 6)
    eight = build(
        company_contract(backend_name, arrays, with_incurred=False), 4, 5, 6, incurred=False
    )
    assert thirteen["n_feat"] == 13
    assert eight["n_feat"] == 8
    assert thirteen["feat"].shape == (18, 12, 13)  # 3 companies x 6 origins, 2 lines x 6 lags
    assert eight["feat"].shape == (18, 12, 8)
    assert thirteen["feature_names"] == (
        "observed incremental paid loss ratio",
        "observed flag",
        "line written",
        "context incremental paid loss ratio",
        "own incremental to cumulative ratio",
        "cumulative paid loss ratio at the latest visible lag",
        "fraction of lags observed",
        "steps ahead",
        "context incremental incurred loss ratio",
        "observed incremental incurred loss ratio",
        "paid to incurred",
        "incurred loss ratio",
        "case reserve loss ratio",
    )
    assert eight["feature_names"] == thirteen["feature_names"][:8]
    # the eight shared features are the same numbers in both forms
    np.testing.assert_allclose(eight["feat"], thirteen["feat"][:, :, :8])

    assert (thirteen["n_l"], thirteen["n_d"]) == (2, 6)
    # tokens are line-major with the lag varying fastest, 1-based as the
    # embedding tables index them
    np.testing.assert_array_equal(thirteen["line_ix"], [1] * 6 + [2] * 6)
    np.testing.assert_array_equal(thirteen["lag_ix"], list(range(1, 7)) * 2)
    # examples are company-major with the origin varying fastest
    np.testing.assert_array_equal(thirteen["example_company"], np.repeat(np.arange(3), 6))
    np.testing.assert_array_equal(thirteen["example_origin"], np.tile(np.arange(6), 3))

    want_lk = [max(min(4 - w, 6), 1) for _ in range(3) for w in range(6)]
    np.testing.assert_array_equal(thirteen["lk"], want_lk)
    np.testing.assert_array_equal(
        thirteen["has_history"], [(4 - w) >= 1 for _ in range(3) for w in range(6)]
    )
    assert thirteen["written"].shape == (18, 2)
    assert thirteen["written"].all()


def test_visibility_and_masks_at_a_cutoff(backend_name):
    """The observed flag is the cutoff, the target mask is the scoring window.

    An origin whose first development period has not elapsed at the cutoff has
    no cumulative to project from, so its target cells are dropped from the mask
    entirely and counted instead - the count is what tells a caller how much of
    the scoring window a cutoff costs.
    """
    cutoff, target_lo, target_hi = 4, 5, 6
    f = build(company_contract(backend_name), cutoff, target_lo, target_hi)
    observed = np.zeros((18, 12))
    targets = np.zeros((18, 12))
    dropped = 0
    for ci in range(3):
        for w in range(6):
            e = ci * 6 + w
            has_history = (cutoff - w) >= 1
            for li in range(2):
                for d in range(6):
                    t = li * 6 + d
                    cal = w + d + 1
                    observed[e, t] = float(cal <= cutoff)
                    if target_lo <= cal <= target_hi:
                        if has_history:
                            targets[e, t] = 1.0
                        else:
                            dropped += 1
    np.testing.assert_array_equal(f["feat"][:, :, 1], observed)
    np.testing.assert_array_equal(f["target_mask"], targets)
    assert f["n_dropped"] == dropped > 0
    # feature 3 is the written flag, which this fixture has on everywhere
    assert (f["feat"][:, :, 2] == 1.0).all()
    # the premium of a token is its own line's origin premium, on every token
    prem = study_arrays()["premium"]
    want = np.stack([np.repeat(prem[ci, :, w], 6) for ci in range(3) for w in range(6)])
    np.testing.assert_allclose(f["premium"], want)


# -- the statistics --------------------------------------------------------------


def test_standardisation_uses_visible_cells_only(backend_name):
    """Per-(line, lag) mean and spread come from cells on or before the cutoff.

    Estimating them over the whole grid would standardise an input using the
    very cells the network is asked to predict - and the resulting feature is
    finite, smooth and wrong, so only a value check can see it.
    """
    cutoff = 4
    arrays = study_arrays()
    f = build(company_contract(backend_name, arrays), cutoff, 5, 6)
    y = increment_ratios(arrays)
    li, d = 0, 2
    visible = [y[ci, li, w, d] for ci in range(3) for w in range(6) if w + d + 1 <= cutoff]
    mu, sd = sample_stats(visible)
    # origins 0 and 1 have reached this lag by diagonal 4, on 3 companies; the
    # other four origins have not, and their cells must not be in the estimate
    assert len(visible) == 6

    ci, w = 1, 1
    e, t = ci * 6 + w, li * 6 + d
    assert f["feat"][e, t, 0] == pytest.approx(clip_z(y[ci, li, w, d], mu, sd))
    # feature 4 is the same standardisation applied to the company's own mean
    # over the visible origins at that lag
    context = np.mean([y[ci, li, w2, d] for w2 in range(6) if w2 + d + 1 <= cutoff])
    assert f["feat"][e, t, 3] == pytest.approx(clip_z(context, mu, sd))

    # and the same numbers standardised over EVERY cell are different, which is
    # what makes the two assertions above a test rather than a coincidence
    every = sample_stats(y[:, li, :, d])
    assert not np.isclose(clip_z(y[ci, li, w, d], *every), f["feat"][e, t, 0])

    # a cell past the cutoff carries feature 1 = 0 whatever its value, because
    # the observed flag multiplies it away
    hidden_e, hidden_t = ci * 6 + 5, li * 6 + 5
    assert f["feat"][hidden_e, hidden_t, 0] == 0.0


def test_own_ratio_and_latest_balances(backend_name):
    """Features 5, 6, 11, 12 and 13 against a hand-built reference.

    Each is a per-company quantity repeated over the lags of its line, and each
    is standardised over the whole array of such quantities rather than per
    (line, lag) - two different standardisations in one builder, which is the
    thing worth pinning.
    """
    cutoff = 4
    arrays = study_arrays()
    f = build(company_contract(backend_name, arrays), cutoff, 5, 6)
    paid, incurred, case, prem = (arrays[k] for k in ("paid", "incurred", "case", "premium"))
    n_c, n_l, n_w, n_d = paid.shape

    own = np.zeros((n_c, n_l, n_d))
    for li in range(n_l):
        for d in range(1, n_d):
            amax = min(cutoff - d, n_w)
            if amax < 1:
                continue
            for ci in range(n_c):
                num = sum(paid[ci, li, w, d] - paid[ci, li, w, d - 1] for w in range(amax))
                den = sum(paid[ci, li, w, d - 1] for w in range(amax))
                own[ci, li, d] = np.clip(num / den if den > 0 else 0.0, -0.5, 3.0)

    cum_lr = np.zeros((n_c, n_l, n_w))
    paid_to_incurred = np.ones((n_c, n_l, n_w))
    incurred_lr = np.zeros((n_c, n_l, n_w))
    case_lr = np.zeros((n_c, n_l, n_w))
    for w in range(n_w):
        lk = min(cutoff - w, n_d)
        if lk < 1:
            continue
        j = lk - 1
        cum_lr[:, :, w] = np.clip(paid[:, :, w, j] / prem[:, :, w], 0.0, 5.0)
        paid_to_incurred[:, :, w] = np.clip(paid[:, :, w, j] / incurred[:, :, w, j], 0.0, 2.0)
        incurred_lr[:, :, w] = np.clip(incurred[:, :, w, j] / prem[:, :, w], 0.0, 5.0)
        case_lr[:, :, w] = np.clip(case[:, :, w, j] / prem[:, :, w], -1.0, 3.0)

    ci, w, li, d = 2, 1, 1, 3
    e, t = ci * n_w + w, li * n_d + d
    assert f["feat"][e, t, 4] == pytest.approx(clip_z(own[ci, li, d], *sample_stats(own)))
    for channel, quantity in (
        (5, cum_lr),
        (10, paid_to_incurred),
        (11, incurred_lr),
        (12, case_lr),
    ):
        assert f["feat"][e, t, channel] == pytest.approx(
            clip_z(quantity[ci, li, w], *sample_stats(quantity))
        ), f"feature {channel + 1}"

    # the per-company quantities are the SAME on every lag of their line, which
    # is what "repeated over the lags" means
    for channel in (4, 5, 10, 11, 12):
        row = f["feat"][e, li * n_d : (li + 1) * n_d, channel]
        if channel != 4:  # feature 5 varies by lag; the other four do not
            assert len(set(np.round(row, 12))) == 1

    # features 7 and 8: the fraction of lags observed and the steps ahead
    lk = max(min(cutoff - w, n_d), 1)
    assert (f["feat"][e, :, 6] == lk / n_d).all()
    np.testing.assert_allclose(
        f["feat"][e, li * n_d : (li + 1) * n_d, 7],
        [max(dd + 1 - lk, 0) / n_d for dd in range(n_d)],
    )

    # the projection anchors: the cumulative at the latest visible lag, floored
    # at one, and the origin's premium floored just above zero
    np.testing.assert_allclose(f["c_lk"][e], np.maximum(paid[ci, :, w, lk - 1], 1.0))
    np.testing.assert_allclose(f["anchor_start"][e], paid[ci, :, w, lk - 1])
    np.testing.assert_allclose(f["p_lk"][e], prem[ci, :, w])


# -- chain ladder factors ---------------------------------------------------------


def test_pooled_factors_and_fallback_tail(backend_name):
    """Pooled volume-weighted factors from visible PAIRS of cells only.

    A factor for step ``d`` needs both ``d`` and ``d + 1`` to have elapsed, so
    it sees one origin fewer than the cells at ``d`` alone - the off-by-one this
    test exists for. Steps no visible pair can reach are flat, and a flat factor
    is a log factor of zero rather than a guessed tail.
    """
    cutoff = 6
    arrays = study_arrays()
    contract = company_contract(backend_name, arrays)
    paid = arrays["paid"]
    n_c, n_l, n_w, n_d = paid.shape

    want = np.ones((n_l, n_d - 1))
    for li in range(n_l):
        for d in range(n_d - 1):
            amax = min(cutoff - d - 1, n_w)
            if amax < 1:
                continue
            num = sum(paid[ci, li, w, d + 1] for ci in range(n_c) for w in range(amax))
            den = sum(paid[ci, li, w, d] for ci in range(n_c) for w in range(amax))
            if den > 0:
                want[li, d] = num / den
    want = np.maximum(want, 1.0 + 1e-6)
    np.testing.assert_allclose(pooled_cl_factors(contract["values"][:, :, 0], cutoff), want)

    f = build(contract, cutoff, 7, 11)
    np.testing.assert_allclose(f["fallback_logf"], np.log(want))
    assert (f["fallback_logf"] > 0).all()  # nothing is unreachable at this cutoff

    # at an early cutoff the deep steps have no visible pair at all, and those
    # are exactly the ones held flat
    early = build(contract, 3, 4, 6)
    unreachable = np.arange(1, n_d) >= 3  # 1-based step index j = d + 1
    assert (early["fallback_logf"][:, unreachable] == 0.0).all()
    assert (early["fallback_logf"][:, ~unreachable] > 0.0).all()


def test_a_flat_development_step_is_floored_just_above_one(backend_name):
    """A factor of exactly one has no finite parameter behind it.

    The head is a softplus over a log factor, so it can only produce factors
    above one, and the inverse a caller needs to initialise it at the chain
    ladder, ``log(exp(log f) - 1)``, is minus infinity at ``f = 1``. So a step
    where nothing develops is floored just above one rather than left flat, and
    a line that develops backwards is floored the same way. Ordinary data never
    reaches the floor, which is why it needs a case of its own.
    """
    cutoff = 6
    arrays = study_arrays()
    flat, backwards = 2, 3  # steps of the second line, by 0-based index
    arrays["paid"][:, 1, :, flat + 1] = arrays["paid"][:, 1, :, flat]
    arrays["paid"][:, 1, :, backwards + 1] = 0.97 * arrays["paid"][:, 1, :, backwards]
    arrays["incurred"] = np.maximum(arrays["incurred"], arrays["paid"])
    arrays["case"] = arrays["incurred"] - arrays["paid"]
    contract = company_contract(backend_name, arrays)

    factors = pooled_cl_factors(contract["values"][:, :, 0], cutoff)
    assert factors[1, flat] == 1.0 + 1e-6
    assert factors[1, backwards] == 1.0 + 1e-6
    assert (factors > 1.0).all()
    # and the log factors the head is initialised from are finite either way
    assert np.isfinite(np.log(np.exp(np.log(factors)) - 1.0)).all()


def test_anchor_factors_fall_back_to_pooled_on_thin_lines(backend_name):
    """A company's own factors, with two different answers when it has none.

    A line the company does not write keeps a flat factor of one: it has no
    development to describe, and borrowing the market's would put a projection
    on a line that does not exist. A line the company DOES write but whose step
    has no positive starting balance borrows the pooled factor instead, which is
    the only estimate available for a step the company will still be projected
    across.

    The plan for this test called both cases the pooled fallback; the reference
    implementation skips an unwritten line before the fallback can apply, so the
    two are checked apart here.
    """
    cutoff = 6
    arrays = study_arrays()
    # company 0's second line is written but has paid nothing: every step's
    # starting balance sums to zero, so no own factor can be formed
    arrays["paid"][0, 1] = 0.0
    arrays["incurred"][0, 1] = 0.0
    arrays["case"][0, 1] = 0.0
    # company 2 does not write the second line at all
    contract = company_contract(backend_name, arrays, lines={2: (0,)})
    np.testing.assert_array_equal(
        contract["line_mask"], [[True, True], [True, True], [True, False]]
    )

    cp = contract["values"][:, :, 0]
    pooled = pooled_cl_factors(cp, cutoff)
    step = np.arange(1, cp.shape[3])  # 1-based step index
    pooled_tail_one = np.where(step[None, :] >= cutoff, 1.0, pooled)
    logf = origin_cl_log_factors(cp, contract["line_mask"], cutoff, pooled_tail_one)

    # the written but unpaid line borrows the pooled factor at every step a
    # visible pair could have reached, and is flat beyond
    reachable = np.array([min(cutoff - d - 1, cp.shape[2]) >= 1 for d in range(cp.shape[3] - 1)])
    np.testing.assert_allclose(logf[0, 1, reachable], np.log(pooled_tail_one[1, reachable]))
    np.testing.assert_allclose(logf[0, 1, ~reachable], 0.0)
    # the unwritten line is flat everywhere, pooled factor or not
    np.testing.assert_allclose(logf[2, 1], 0.0)
    assert (pooled_tail_one[1, reachable] > 1.0).any()  # the two answers really differ
    # a line with its own history keeps its own factors
    assert not np.allclose(logf[0, 0], np.log(pooled_tail_one[0]))

    # and the builder hands the same array back, one row per example
    f = build(contract, cutoff, 7, 11)
    np.testing.assert_allclose(f["anchor_logf"], np.repeat(logf, cp.shape[2], axis=0))


# -- the leak check ---------------------------------------------------------------


def test_hidden_cells_do_not_reach_the_inputs(backend_name):
    """Perturb every cell past the cutoff; only the targets may move.

    This is the whole file in one assertion. The contract's own arrays are
    rewritten rather than the triangle, so the observed flags, the premium and
    the line coverage are untouched and the only thing that changes is the value
    of a cell the cutoff hides.
    """
    cutoff = 4
    contract = company_contract(backend_name)
    hidden = contract["cal_idx"] > cutoff
    perturbed = dict(contract)
    perturbed["values"] = np.where(hidden, contract["values"] * 1.37 + 0.123, contract["values"])
    perturbed["x"] = np.where(hidden, contract["x"] * 1.37 + 0.123, contract["x"])

    before = build(contract, cutoff, 5, 6)
    after = build(perturbed, cutoff, 5, 6)
    assert set(before) == set(after)
    for key, value in before.items():
        if key == "target":
            continue
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, after[key], err_msg=f"{key} moved")
        else:
            assert value == after[key], key
    # the targets are the one thing that must move: they are the hidden cells
    assert not np.array_equal(before["target"], after["target"])


# -- refusals ----------------------------------------------------------------------


def test_incurred_without_case_is_refused(backend_name):
    contract = company_contract(backend_name)
    with pytest.raises(ValueError, match="both"):
        tlrn_features(contract, cutoff=4, target_lo=5, target_hi=6, incurred_field="incurred_loss")
    with pytest.raises(ValueError, match="both"):
        tlrn_features(contract, cutoff=4, target_lo=5, target_hi=6, case_field="case_reserve")


def test_a_field_the_contract_does_not_carry_is_refused(backend_name):
    contract = company_contract(backend_name)
    with pytest.raises(ValueError, match="not a channel"):
        tlrn_features(
            contract,
            cutoff=4,
            target_lo=5,
            target_hi=6,
            incurred_field="reported_loss",
            case_field="case_reserve",
        )


def test_a_case_field_carried_as_an_increment_is_refused(backend_name):
    """``case_reserve`` differenced is the case MOVEMENT, a different quantity.

    The builder reads the level straight out of the contract, so a contract that
    differenced it would hand the features a movement under the name of a
    balance - finite, plausible and wrong.
    """
    t = study_triangle(backend_name)
    contract = nn_company_data(
        t, loss_field="paid_loss", feature_fields=("incurred_loss", "case_reserve")
    )
    with pytest.raises(ValueError, match="level"):
        tlrn_features(
            contract,
            cutoff=4,
            target_lo=5,
            target_hi=6,
            incurred_field="incurred_loss",
            case_field="case_reserve",
        )


def test_the_cutoff_and_the_scoring_window_are_checked(backend_name):
    contract = company_contract(backend_name)
    with pytest.raises(ValueError, match="cutoff"):
        build(contract, 0, 1, 6)
    with pytest.raises(ValueError, match="cutoff"):
        build(contract, 12, 13, 13)
    with pytest.raises(ValueError, match="target_lo"):
        build(contract, 4, 4, 6)
    with pytest.raises(ValueError, match="target_hi"):
        build(contract, 4, 5, 3)  # inverted by more than one: a swapped pair
    with pytest.raises(ValueError, match="target_hi"):
        build(contract, 4, 5, 12)


def test_an_empty_scoring_window_is_legal_and_scores_nothing(backend_name):
    """A fully developed triangle has nothing past its cutoff to forecast.

    ``target_hi == target_lo - 1`` says exactly that, and it is a state a real
    triangle reaches rather than a swapped pair of arguments, so the mask comes
    back all zero instead of the call being refused. Everything else about the
    example tensors is unchanged, which is what lets a caller build its final
    set the same way whatever the triangle's shape.
    """
    contract = company_contract(backend_name)
    last = contract["n_w"] + contract["n_d"] - 1
    empty = build(contract, last, last + 1, last)
    assert empty["target_mask"].sum() == 0.0
    assert empty["n_dropped"] == 0
    # every origin is fully developed at this cutoff, so nothing projects
    np.testing.assert_array_equal(empty["lk"], contract["n_d"])
    assert empty["has_history"].all()
    assert empty["feat"].shape == build(contract, 4, 5, 6)["feat"].shape


def test_a_hole_inside_the_visible_region_is_refused(backend_name):
    """A missing cell the cutoff does not hide has no honest reading.

    The projection starts from the cumulative at each origin's latest visible
    lag, and a missing one is read as a zero balance floored to one dollar - so
    the fit runs, the numbers are finite, and one company's whole reserve is
    built on a one-dollar anchor. The reference implementation never meets this
    because its cohort selection requires complete grids; here it is refused by
    name.
    """
    arrays = study_arrays()
    arrays["paid"][0, 0, 1, 1] = np.nan  # calendar diagonal 3
    contract = company_contract(backend_name, arrays)
    with pytest.raises(ValueError, match="hole"):
        build(contract, 4, 5, 6)
    # and a cutoff that hides the same cell is fine
    build(contract, 2, 3, 6)


def test_a_flat_contract_is_refused(backend_name):
    """The features are engineered across a company's lines at once."""
    flat = nn_company_data(study_triangle(backend_name), loss_field="paid_loss")
    flat = {**flat, "x": flat["x"][:, 0], "values": flat["values"][:, 0]}
    with pytest.raises(ValueError, match="nn_company_data"):
        build(flat, 4, 5, 6, incurred=False)
