"""Compartmental can score cells it was not trained on - both variants, checkable.

The last Bayesian entry to gain held-out scoring, and the one that needed new
machinery: its contract delta-stacks an outstanding block (delta = 0) and a paid
block (delta = 1) over IDENTICAL (w, d), so ``training_index`` used to refuse it
outright and it had no in-sample agreement gate. ``DeltaCellIndex`` carries the
block axis now, and these tests are what make the resulting numbers checkable.

Layers mirror ``test_heldout_scorer_csr.py``:

1. **Closed form.** ``log_lik_cells`` against a hand-written scalar loop over
   each Stan program's formula (``model.stan`` for gaussian amounts,
   ``model_lognormal.stan`` for the ratio reconstruction), at draws chosen so a
   transposed index, a swapped sd column or a delta slip lands on a visibly
   wrong number.
2. **The agreement gate** (``slow``): score the training cells through the
   held-out code path and reproduce the fit's own ``log_lik`` elementwise -
   for the lognormal variant over exactly the rows ``_kept_rows_`` says
   survived, in Stan's order. Paired with negative controls.
3. **The carries.** ``log_lik_at`` applies the per-variant measure carry
   (none for gaussian amounts, ``-log premium`` for lognormal ratios), and
   ``predict_at`` adds the training-diagonal anchor to the lognormal variant's
   incremental draws. Normalization on the amount scale is checked for BOTH
   variants, with mutants that integrate to something else.

Plus what this entry alone needs: the delta dispatch. Training cells arrive as
a ``DeltaCellIndex`` carrying both blocks; held-out cells arrive as a plain
``CellIndex`` from ``index_into`` and ARE the paid block - a rule
``scorer.cell_deltas`` encodes and these tests pin from both sides.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr.gallery.bayesian.compartmental import scorer
from ibnr.gallery.bayesian.compartmental.model import (
    HELDOUT_DECLARATIONS,
    Compartmental,
    os_curve,
    paid_curve,
)
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout
from ibnr.kernels.contract import compartmental_stan_data
from ibnr.kernels.densities import check_normalization, lognormal_lpdf, normal_lpdf
from ibnr.kernels.holdout import (
    CellIndex,
    DeltaCellIndex,
    index_into,
    next_diagonal,
    training_index,
)
from ibnr.triangle.core import Triangle

#: ground-truth curve parameters for the synthetic cohort (same values as
#: tests/test_compartmental.py: reporting 3x faster than payment, ULR 0.56)
KER, KP, RLR, RRF = 3.0, 1.0, 0.7, 0.8
N_W = N_D = 6
PREMIUM = 1000.0


def _joint_triangle(*, through: int, noise: float = 0.0, seed: int = 0) -> Triangle:
    """Paid + reported on the compartmental curves for one cohort (``FIT_CO``),
    with premium.

    ``noise`` multiplies each level by ``exp(N(0, noise))`` - used by the slow
    Stan gate, where noise-free curves would let the sampler chase sigma toward
    zero AND (deliberately) makes some deep-dev paid increments negative, so
    the lognormal keep mask has real work to do.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, N_W + 1):
        for d in range(1, N_D + 1):
            if w + d - 1 > through:
                continue
            t = float(d)
            paid = PREMIUM * paid_curve(t, KER, KP, RLR, RRF)
            outstanding = PREMIUM * os_curve(t, KER, KP, RLR)
            if noise:
                paid *= float(np.exp(rng.normal(0.0, noise)))
                outstanding *= float(np.exp(rng.normal(0.0, noise)))
            for field, value in (
                ("paid_loss", paid),
                ("reported_loss", paid + outstanding),
                ("earned_premium", PREMIUM),
            ):
                rows.append(
                    {
                        "company": "FIT_CO",
                        "origin_period": dt.date(2010 + w - 1, 1, 1),
                        "dev_lag": 12 * d,
                        "eval_date": dt.date(2010 + w - 1 + d - 1, 12, 31),
                        "field": field,
                        "value": float(value),
                    }
                )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


def _contract_for(triangle: Triangle) -> dict:
    return compartmental_stan_data(
        triangle,
        paid_field="paid_loss",
        reported_field="reported_loss",
        premium_field="earned_premium",
    )


@pytest.fixture(scope="module")
def contract() -> dict:
    return _contract_for(_joint_triangle(through=N_W))


@pytest.fixture(scope="module")
def heldout():
    """The diagonal after the training cutoff, carrying BOTH source fields -
    the shape a compartmental panel run hands the entry. Only the paid rows
    are scorable; ``index_into`` narrows to them."""
    return next_diagonal(
        _joint_triangle(through=N_W + 1),
        as_of=dt.date(2010 + N_W - 1, 12, 31),
        fields=["paid_loss", "reported_loss"],
        premium_field="earned_premium",
    )


def fake_posterior_gaussian(contract: dict, n_draws: int = 7, seed: int = 0) -> dict:
    """Distinct, non-symmetric values per index (the CSR-fixture discipline).

    RLR rises in w while RRF falls, ker and kp stay well apart (the curves are
    singular at ker == kp), and sigma_os is an order of magnitude above
    sigma_paid so a delta slip lands on a visibly wrong scale rather than a
    plausible neighbour.
    """
    rng = np.random.default_rng(seed)
    n_w = contract["n_w"]
    return {
        "RLR": np.tile(np.linspace(0.60, 0.85, n_w), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_w)),
        "RRF": np.tile(np.linspace(0.95, 0.70, n_w), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_w)),
        "ker": rng.uniform(2.7, 3.3, n_draws),
        "kp": rng.uniform(0.8, 1.2, n_draws),
        "sigma_os": rng.uniform(20.0, 30.0, n_draws),
        "sigma_paid": rng.uniform(2.0, 3.0, n_draws),
    }


def fake_posterior_lognormal(contract: dict, n_draws: int = 7, seed: int = 0) -> dict:
    """Every varying-effect vector distinct and direction-revealing: dev
    effects rise where AY effects fall, the two columns of each sd pair
    differ, so a transposed index, a swapped sd column or an AY-for-dev
    mix-up each move the answer."""
    rng = np.random.default_rng(seed)
    n_w, n_d = contract["n_w"], contract["n_d"]

    def vec(lo: float, hi: float, n: int) -> np.ndarray:
        return np.tile(np.linspace(lo, hi, n), (n_draws, 1)) + rng.normal(0, 1e-3, (n_draws, n))

    return {
        "b_oRLR": rng.normal(0.0, 0.3, n_draws),
        "b_oRRF": rng.normal(0.0, 0.3, n_draws),
        "b_oker": rng.normal(0.0, 0.3, n_draws),
        "b_okp": rng.normal(0.0, 0.3, n_draws),
        "u_ay": np.stack([vec(0.4, -0.4, n_w), vec(-0.3, 0.3, n_w)], axis=1),
        "sd_dev": np.column_stack([rng.uniform(0.5, 0.7, n_draws), rng.uniform(0.2, 0.3, n_draws)]),
        "z_RLR_dev": vec(-0.5, 0.5, n_d),
        "z_RRF_dev": vec(0.6, -0.6, n_d),
        "sd_ker": np.column_stack(
            [rng.uniform(0.1, 0.2, n_draws), rng.uniform(0.25, 0.35, n_draws)]
        ),
        "z_ker_ay": vec(0.3, -0.3, n_w),
        "z_ker_dev": vec(-0.2, 0.4, n_d),
        "sd_kp": np.column_stack(
            [rng.uniform(0.15, 0.25, n_draws), rng.uniform(0.05, 0.1, n_draws)]
        ),
        "z_kp_ay": vec(-0.4, 0.2, n_w),
        "z_kp_dev": vec(0.25, -0.35, n_d),
        "sigma_os": rng.uniform(0.15, 0.25, n_draws),
        "sigma_paid": rng.uniform(0.08, 0.12, n_draws),
    }


def _subset(cells: DeltaCellIndex, rows, value: np.ndarray | None = None) -> DeltaCellIndex:
    """A DeltaCellIndex over ``rows`` of ``cells``, optionally with the
    observed values replaced (for the normalization sweeps)."""
    rows = np.asarray(rows, dtype=int)
    return DeltaCellIndex(
        w=cells.w[rows],
        d=cells.d[rows],
        value=cells.value[rows] if value is None else np.asarray(value, dtype=float),
        prev_value=cells.prev_value[rows],
        premium=cells.premium[rows],
        delta=cells.delta[rows],
    )


def _lognormal_cell_params(post: dict, s: int, w0: int, d0: int) -> tuple:
    """model_lognormal.stan:92-97 and 104-107, one cell at a time: the
    reconstruction identities u = sd * z written scalar-wise, independently of
    the scorer's vectorized version."""
    ker = 3.0 * np.exp(
        0.1
        * (
            post["b_oker"][s]
            + post["sd_ker"][s, 0] * post["z_ker_ay"][s, w0]
            + post["sd_ker"][s, 1] * post["z_ker_dev"][s, d0]
        )
    )
    kp = 1.0 * np.exp(
        0.1
        * (
            post["b_okp"][s]
            + post["sd_kp"][s, 0] * post["z_kp_ay"][s, w0]
            + post["sd_kp"][s, 1] * post["z_kp_dev"][s, d0]
        )
    )
    rlr = 0.7 * np.exp(
        0.2
        * (
            post["b_oRLR"][s]
            + post["u_ay"][s, 0, w0]
            + post["sd_dev"][s, 0] * post["z_RLR_dev"][s, d0]
        )
    )
    rrf = 0.8 * np.exp(
        0.1
        * (
            post["b_oRRF"][s]
            + post["u_ay"][s, 1, w0]
            + post["sd_dev"][s, 1] * post["z_RRF_dev"][s, d0]
        )
    )
    return ker, kp, rlr, rrf


class _StubCompartmental(Compartmental):
    """A fitted compartmental without a sampler: the scorer only needs the
    contract, pooled draws and the variant."""

    def __init__(self, contract, post, variant):
        super().__init__()
        self.contract_ = contract
        self._post = post
        self.variant_ = variant
        for name, value in HELDOUT_DECLARATIONS[variant].items():
            setattr(self, name, value)

    def _posterior(self):
        return self._post

    def _log_lik_native(self, cells):
        return scorer.log_lik_cells(self.contract_, self._post, cells, variant=self.variant_)

    def _draws_native(self, cells, *, rng):
        return scorer.draw_cells(self.contract_, self._post, cells, rng=rng, variant=self.variant_)


# -- layer 1: closed form -----------------------------------------------------


def test_gaussian_matches_a_scalar_loop_over_the_stan_formula(contract):
    """``mu[i] = premium[w[i]] * (delta==0 ? os_curve : paid_curve)`` and
    ``log_lik[i] = normal_lpdf(loss[i] | mu[i], delta==0 ? sigma_os :
    sigma_paid)`` (model.stan:102-110, 156-159), written out one cell at a
    time so the vectorized version has something independent to disagree
    with. ``t`` is spelled ``d * step / 12`` in the loop, the contract
    definition, not read off anything the scorer computed."""
    post = fake_posterior_gaussian(contract)
    cells = training_index(contract)
    got = scorer.log_lik_cells(contract, post, cells, variant="gaussian")

    step_years = contract["dev_grain_months"] / 12.0
    prem = np.asarray(contract["premium"], dtype=float)
    expected = np.empty((len(post["ker"]), cells.n_cells))
    for s in range(expected.shape[0]):
        for i in range(cells.n_cells):
            w0, d0 = int(cells.w[i]) - 1, int(cells.d[i]) - 1
            t = (d0 + 1) * step_years
            ker, kp = post["ker"][s], post["kp"][s]
            if int(cells.delta[i]) == 0:
                lr = os_curve(t, ker, kp, post["RLR"][s, w0])
                sig = post["sigma_os"][s]
            else:
                lr = paid_curve(t, ker, kp, post["RLR"][s, w0], post["RRF"][s, w0])
                sig = post["sigma_paid"][s]
            expected[s, i] = normal_lpdf(cells.value[i], prem[w0] * lr, sig)

    assert got.shape == expected.shape
    assert np.allclose(got, expected)


def test_lognormal_matches_a_scalar_loop_over_the_stan_formula(contract):
    """model_lognormal.stan:104-122 and 179-183 one cell at a time: per-cell
    parameters from the u = sd * z reconstruction, outstanding read off the
    curve, paid differenced over (t - devfreq, t], and the observed ratio
    formed exactly as ``_lognormal_stan_data`` forms Stan's ``y`` (increment
    against the predecessor, over the contract's premium)."""
    post = fake_posterior_lognormal(contract)
    cells = training_index(contract)
    got = scorer.log_lik_cells(contract, post, cells, variant="lognormal")

    step_years = contract["dev_grain_months"] / 12.0
    prem = np.asarray(contract["premium"], dtype=float)
    n_draws = len(post["b_oker"])
    expected = np.empty((n_draws, cells.n_cells))
    for s in range(n_draws):
        for i in range(cells.n_cells):
            w0, d0 = int(cells.w[i]) - 1, int(cells.d[i]) - 1
            ker, kp, rlr, rrf = _lognormal_cell_params(post, s, w0, d0)
            t = (d0 + 1) * step_years
            if int(cells.delta[i]) == 0:
                mu = os_curve(t, ker, kp, rlr)
                y = cells.value[i] / prem[w0]
                sig = post["sigma_os"][s]
            else:
                mu = paid_curve(t, ker, kp, rlr, rrf)
                if t > step_years:
                    mu -= paid_curve(t - step_years, ker, kp, rlr, rrf)
                y = (cells.value[i] - cells.prev_value[i]) / prem[w0]
                sig = post["sigma_paid"][s]
            expected[s, i] = lognormal_lpdf(y, np.log(mu), sig)

    assert got.shape == expected.shape
    assert np.allclose(got, expected)


@pytest.mark.parametrize("variant", ["gaussian", "lognormal"])
def test_indices_are_not_interchangeable(contract, variant):
    """A guard on the guard: if swapping w for d changed nothing, the
    closed-form tests would be vacuous."""
    fake = fake_posterior_gaussian if variant == "gaussian" else fake_posterior_lognormal
    post = fake(contract)
    cells = training_index(contract)
    swapped = DeltaCellIndex(
        w=cells.d,
        d=cells.w,
        value=cells.value,
        prev_value=cells.prev_value,
        premium=cells.premium,
        delta=cells.delta,
    )
    assert (cells.w != cells.d).any(), "fixture cannot distinguish w from d"

    a = scorer.log_lik_cells(contract, post, cells, variant=variant)
    b = scorer.log_lik_cells(contract, post, swapped, variant=variant)
    assert not np.allclose(a, b)


def test_a_plain_cell_index_is_the_paid_block(contract):
    """THE delta dispatch, from both sides.

    Held-out cells come through ``index_into`` as a plain ``CellIndex``, and
    the only raw field that path can resolve is the paid one - so the scorer
    must read a plain index as delta = 1 everywhere. Equal to the same rows
    under an explicit paid label, different from the same rows under an
    outstanding label (sigma_os is an order of magnitude off sigma_paid, so
    the mutant that defaults delta to 0 is far away, not adjacent)."""
    cells = training_index(contract)
    paid_rows = np.flatnonzero(cells.delta == 1)
    labelled = _subset(cells, paid_rows)
    plain = CellIndex(
        w=labelled.w,
        d=labelled.d,
        value=labelled.value,
        prev_value=labelled.prev_value,
        premium=labelled.premium,
    )
    np.testing.assert_array_equal(scorer.cell_deltas(plain), np.ones(plain.n_cells, dtype=int))
    np.testing.assert_array_equal(scorer.cell_deltas(cells), cells.delta)

    for variant, fake in (
        ("gaussian", fake_posterior_gaussian),
        ("lognormal", fake_posterior_lognormal),
    ):
        post = fake(contract)
        as_plain = scorer.log_lik_cells(contract, post, plain, variant=variant)
        as_paid = scorer.log_lik_cells(contract, post, labelled, variant=variant)
        np.testing.assert_array_equal(as_plain, as_paid)

    mislabelled = DeltaCellIndex(
        w=labelled.w,
        d=labelled.d,
        value=labelled.value,
        prev_value=labelled.prev_value,
        premium=labelled.premium,
        delta=np.zeros(labelled.n_cells, dtype=int),
    )
    post = fake_posterior_gaussian(contract)
    assert not np.allclose(
        scorer.log_lik_cells(contract, post, plain, variant="gaussian"),
        scorer.log_lik_cells(contract, post, mislabelled, variant="gaussian"),
    )


def test_the_reconstruction_is_backend_blind(contract):
    """The lognormal per-cell parameters must come from the SAMPLED sd_*/z_*
    sites via u = sd * z, because only those exist in every backend.

    Two posteriors with identical products - one carrying (sd, z), the other
    carrying (1, sd * z), i.e. the u_* vectors precomputed by hand - must
    score identically; and the sd factors must genuinely matter, or a scorer
    that ignored them (reading z as if it were u) would pass this test."""
    post = fake_posterior_lognormal(contract)
    cells = training_index(contract)
    a = scorer.mu_cells(contract, post, cells, variant="lognormal")

    folded = dict(post)
    for sd_name, z_names in (
        ("sd_dev", ("z_RLR_dev", "z_RRF_dev")),
        ("sd_ker", ("z_ker_ay", "z_ker_dev")),
        ("sd_kp", ("z_kp_ay", "z_kp_dev")),
    ):
        for col, z_name in enumerate(z_names):
            folded[z_name] = post[sd_name][:, [col]] * post[z_name]
        folded[sd_name] = np.ones_like(post[sd_name])
    b = scorer.mu_cells(contract, folded, cells, variant="lognormal")
    np.testing.assert_allclose(a, b)

    # the guard on the guard: dropping the sd multiplication must move mu
    unscaled = dict(post)
    unscaled["sd_dev"] = np.ones_like(post["sd_dev"])
    assert not np.allclose(scorer.mu_cells(contract, unscaled, cells, variant="lognormal"), a)


def test_missing_posterior_variable_names_what_is_needed(contract):
    cells = training_index(contract)
    post = fake_posterior_gaussian(contract)
    del post["RRF"]
    with pytest.raises(KeyError, match="RRF"):
        scorer.log_lik_cells(contract, post, cells, variant="gaussian")

    post = fake_posterior_lognormal(contract)
    del post["z_kp_dev"]
    with pytest.raises(KeyError, match="z_kp_dev"):
        scorer.log_lik_cells(contract, post, cells, variant="lognormal")


def test_unknown_variant_is_refused(contract):
    with pytest.raises(ValueError, match="variant must be one of"):
        scorer.log_lik_cells(
            contract,
            fake_posterior_gaussian(contract),
            training_index(contract),
            variant="hierarchical",
        )


def test_lognormal_refuses_nonpositive_cells_and_gaussian_takes_them(contract):
    """The two variants' documented family limits, side by side.

    A paid increment flattened to zero has no lognormal density - the same
    rule ``_lognormal_stan_data`` applies at fit time - and must raise, not
    clamp. The gaussian variant takes the identical cells natively; that is
    deliberately why the gaussian arm survives a mechanical retrospective and
    it must not grow a guard here."""
    cells = training_index(contract)
    paid_rows = np.flatnonzero((cells.delta == 1) & (cells.d > 1))
    target = int(paid_rows[0])
    value = cells.value.copy()
    value[target] = cells.prev_value[target]  # zero increment
    flat = DeltaCellIndex(
        w=cells.w,
        d=cells.d,
        value=value,
        prev_value=cells.prev_value,
        premium=cells.premium,
        delta=cells.delta,
    )
    with pytest.raises(ValueError, match="non-positive"):
        scorer.log_lik_cells(
            contract, fake_posterior_lognormal(contract), flat, variant="lognormal"
        )
    out = scorer.log_lik_cells(
        contract, fake_posterior_gaussian(contract), flat, variant="gaussian"
    )
    assert np.isfinite(out).all()

    # a NEGATIVE outstanding level (redundant case reserves) likewise
    os_rows = np.flatnonzero(cells.delta == 0)
    value = cells.value.copy()
    value[os_rows[0]] = -25.0
    redundant = DeltaCellIndex(
        w=cells.w,
        d=cells.d,
        value=value,
        prev_value=cells.prev_value,
        premium=cells.premium,
        delta=cells.delta,
    )
    assert np.isfinite(
        scorer.log_lik_cells(
            contract, fake_posterior_gaussian(contract), redundant, variant="gaussian"
        )
    ).all()
    with pytest.raises(ValueError, match="non-positive"):
        scorer.log_lik_cells(
            contract, fake_posterior_lognormal(contract), redundant, variant="lognormal"
        )


def test_t_comes_from_the_contract_grain_not_the_dev_index():
    """On the annual grain t == d numerically, so a scorer that reads the dev
    index as an age passes every mart-shaped test. A quarterly contract
    separates them: d = 2 must evaluate the curves at t = 0.5 years (and the
    lognormal increment over (0.25, 0.5]), not at t = 2.0."""
    quarterly = {
        "n_w": 1,
        "n_d": 2,
        "w": np.array([1, 1, 1, 1]),
        "d": np.array([1, 2, 1, 2]),
        "delta": np.array([0, 0, 1, 1]),
        "loss": np.array([100.0, 120.0, 30.0, 80.0]),
        "premium": np.array([PREMIUM]),
        "dev_grain_months": 3,
    }
    annual = {**quarterly, "dev_grain_months": 12}
    cells = training_index(quarterly)

    post = fake_posterior_gaussian(quarterly, n_draws=3, seed=1)
    got = scorer.mu_cells(quarterly, post, cells, variant="gaussian")
    right = PREMIUM * paid_curve(0.5, post["ker"], post["kp"], post["RLR"][:, 0], post["RRF"][:, 0])
    wrong = PREMIUM * paid_curve(2.0, post["ker"], post["kp"], post["RLR"][:, 0], post["RRF"][:, 0])
    np.testing.assert_allclose(got[:, 3], right)
    assert not np.allclose(got[:, 3], wrong)

    lpost = fake_posterior_lognormal(quarterly, n_draws=3, seed=1)
    lgot = scorer.mu_cells(quarterly, lpost, cells, variant="lognormal")
    for s in range(3):
        ker, kp, rlr, rrf = _lognormal_cell_params(lpost, s, 0, 1)
        want = paid_curve(0.5, ker, kp, rlr, rrf) - paid_curve(0.25, ker, kp, rlr, rrf)
        np.testing.assert_allclose(lgot[s, 3], want)
    # and the grain genuinely flows through: the same cells on an annual
    # contract must give different mus
    assert not np.allclose(
        scorer.mu_cells(annual, lpost, training_index(annual), variant="lognormal"), lgot
    )


# -- the kept-rows bookkeeping ------------------------------------------------


def test_lognormal_stan_data_stores_the_surviving_rows():
    """``_kept_rows_`` is what aligns a scorer elementwise with Stan's log_lik
    vector, which covers ONLY the kept rows in contract order. One paid cell
    is pushed below its predecessor, so exactly one row must drop - and the
    stored indices must reproduce Stan's ``y`` from the training index's own
    arrays, elementwise. ``dropped_cells_`` keeps its published meaning."""
    tri = _joint_triangle(through=N_W)
    df = tri.execute()
    # execute() hands back datetime64, so compare on a normalized column
    first_origin = pd.to_datetime(df["origin_period"]) == pd.Timestamp(2010, 1, 1)
    hit = (df["field"] == "paid_loss") & first_origin & (df["dev_lag"] == 48)
    prev = df.loc[(df["field"] == "paid_loss") & first_origin & (df["dev_lag"] == 36), "value"]
    assert hit.sum() == 1 and len(prev) == 1
    df.loc[hit, "value"] = float(prev.iloc[0]) * 0.9  # a negative increment at (w=1, d=4)
    contract = _contract_for(Triangle.from_long(df, measure="cumulative"))

    entry = Compartmental()
    entry.contract_ = contract
    entry.variant_ = "lognormal"
    data = entry._lognormal_stan_data()

    assert entry.dropped_cells_ == {"outstanding": 0, "paid_incremental": 1}
    kept = entry._kept_rows_
    assert kept is not None
    assert len(kept) == data["len_data"] == contract["len_data"] - 1
    np.testing.assert_array_equal(contract["w"][kept], data["w"])
    np.testing.assert_array_equal(contract["d"][kept], data["d"])
    np.testing.assert_array_equal(contract["delta"][kept], data["delta"])

    cells = training_index(contract)
    amount = np.where(cells.delta == 1, cells.value - cells.prev_value, cells.value)
    ratio = amount / np.asarray(contract["premium"], dtype=float)[cells.w - 1]
    np.testing.assert_allclose(data["y"], ratio[kept])

    # the scorer refuses the full training index (the dropped cell has no
    # density) and accepts exactly the kept subset
    post = fake_posterior_lognormal(contract)
    with pytest.raises(ValueError, match="non-positive"):
        scorer.log_lik_cells(contract, post, cells, variant="lognormal")
    out = scorer.log_lik_cells(contract, post, _subset(cells, kept), variant="lognormal")
    assert out.shape == (7, len(kept))
    assert np.isfinite(out).all()


# -- layer 3: the carries -----------------------------------------------------


def test_the_entry_declares_capabilities_per_variant():
    entry = Compartmental()
    assert isinstance(entry, ScoresHeldout)
    assert isinstance(entry, PredictsHeldout)
    # no class-level declaration: the variant decides, at fit time
    assert getattr(entry, "heldout_measure", None) is None
    assert getattr(entry, "heldout_draw_scale", None) is None
    assert HELDOUT_DECLARATIONS == {
        "gaussian": {"heldout_measure": "amount", "heldout_draw_scale": "cumulative"},
        "lognormal": {"heldout_measure": "loss_ratio", "heldout_draw_scale": "incremental"},
    }


@pytest.mark.parametrize("variant", ["gaussian", "lognormal"])
def test_fit_applies_the_variant_declarations(monkeypatch, variant):
    """Parameter delivery through the public entry point, not a signature
    check: fit() must actually stamp the declarations on the instance, or
    log_lik_at / predict_at refuse to run. The sampler is stubbed out - the
    declarations must not depend on it."""
    entry = Compartmental()
    monkeypatch.setattr(entry, "_sample_stan", lambda *a, **k: "sentinel-idata")
    entry.fit(_joint_triangle(through=N_W), variant=variant, seed=1)
    assert entry.idata_ == "sentinel-idata"
    for name, value in HELDOUT_DECLARATIONS[variant].items():
        assert getattr(entry, name) == value
    if variant == "lognormal":
        # noise-free curves drop nothing: every stacked row survives
        np.testing.assert_array_equal(entry._kept_rows_, np.arange(entry.contract_["len_data"]))


def test_gaussian_log_lik_at_is_already_on_the_amount_scale(contract):
    """Model 1's density is of amounts, so the measure carry is the identity -
    and it must be the base class saying so (measure='amount'), not the entry
    skipping the carry."""
    entry = _StubCompartmental(contract, fake_posterior_gaussian(contract), "gaussian")
    cells = training_index(contract)
    np.testing.assert_array_equal(entry.log_lik_at(cells), entry._log_lik_native(cells))


def test_lognormal_log_lik_at_carries_by_exactly_log_premium(contract):
    """Model 2's density is of loss RATIOS; the leaderboard sums densities on
    amounts. The carry is exactly -log(premium) per cell, applied once, by
    the base class. The increment->cumulative step adds NOTHING (Jacobian 1:
    the anchor is training data), so any further difference is a bug."""
    entry = _StubCompartmental(contract, fake_posterior_lognormal(contract), "lognormal")
    cells = training_index(contract)
    native = entry._log_lik_native(cells)
    carried = entry.log_lik_at(cells)
    np.testing.assert_allclose(carried, native - np.log(cells.premium)[None, :])


def test_heldout_paid_cells_index_and_score_end_to_end(contract, heldout):
    """The board story: the held-out forecast covers the PAID field only,
    through the standard ``index_into`` path, which hands back a plain
    CellIndex (= the paid block). Reported-loss cells are not scorable even
    though reported is a source field - the fit models outstanding, which is
    not a raw field - and the OS block exists for the training gate alone."""
    idx = index_into(heldout, contract, field="paid_loss")
    assert not isinstance(idx, DeltaCellIndex)
    assert idx.n_cells == int((heldout.frame["field"] == "paid_loss").sum())

    for variant, fake in (
        ("gaussian", fake_posterior_gaussian),
        ("lognormal", fake_posterior_lognormal),
    ):
        entry = _StubCompartmental(contract, fake(contract), variant)
        out = entry.log_lik_at(heldout, field="paid_loss")
        assert out.shape == (7, idx.n_cells)
        assert np.isfinite(out).all()

    with pytest.raises(ValueError, match="not what this fit models"):
        index_into(heldout, contract, field="reported_loss")


def test_lognormal_draws_are_anchored_to_cumulative_and_seeded(contract, heldout):
    """The lognormal variant draws paid INCREMENTS (as amounts); the triangle
    is cumulative, so predict_at must add each cell's training-diagonal
    anchor - and the seed must reach the scorer's generator, or two calls
    cannot be compared."""
    post = fake_posterior_lognormal(contract, n_draws=64)
    entry = _StubCompartmental(contract, post, "lognormal")

    got = entry.predict_at(heldout, field="paid_loss", seed=7)
    np.testing.assert_array_equal(got, entry.predict_at(heldout, field="paid_loss", seed=7))
    assert not np.array_equal(got, entry.predict_at(heldout, field="paid_loss", seed=8))

    idx = index_into(heldout, contract, field="paid_loss")
    raw = scorer.draw_cells(contract, post, idx, rng=np.random.default_rng(7), variant="lognormal")
    np.testing.assert_allclose(got, raw + idx.prev_value[None, :])
    # the anchor is two orders of magnitude above the deep-dev increments, so
    # the uncarried mutant is far away, not adjacent
    assert (idx.prev_value > 10 * (raw.mean(axis=0).min())).any()


def test_gaussian_draws_pass_through_on_a_cumulative_triangle(contract, heldout):
    """Model 1 draws cumulative amounts and the triangle is cumulative:
    predict_at must be a pass-through, anchor NOT added."""
    post = fake_posterior_gaussian(contract, n_draws=64)
    entry = _StubCompartmental(contract, post, "gaussian")
    got = entry.predict_at(heldout, field="paid_loss", seed=3)
    idx = index_into(heldout, contract, field="paid_loss")
    raw = scorer.draw_cells(contract, post, idx, rng=np.random.default_rng(3), variant="gaussian")
    np.testing.assert_array_equal(got, raw)


def test_gaussian_draws_have_the_mixture_moments_and_no_zero_variance_anchor(contract):
    """One draw per posterior draw: mean tracks E[mu] and variance obeys the
    law of total variance E[sigma^2] + Var(mu) - a plug-in at the posterior
    mean would miss the Var(mu) term. And EVERY cell has spread: the training
    index includes the fully developed origin's final cell, exactly where
    predict()'s zero-variance anchoring would leak in if _draws_native reused
    it (CohortForecast rejects zero-variance draws outright)."""
    post = fake_posterior_gaussian(contract, n_draws=4000)
    cells = training_index(contract)
    draws = scorer.draw_cells(
        contract, post, cells, rng=np.random.default_rng(5), variant="gaussian"
    )
    mu = scorer.mu_cells(contract, post, cells, variant="gaussian")
    sig = scorer.sigma_cells(post, cells)

    assert np.allclose(draws.mean(axis=0), mu.mean(axis=0), atol=3.0)
    want = (sig**2).mean(axis=0) + mu.var(axis=0)
    assert np.allclose(draws.var(axis=0), want, rtol=0.15)
    assert (mu.var(axis=0) > 0).any()

    # the anchor check needs the anchored cell to be present
    final = (cells.delta == 1) & (cells.w == 1) & (cells.d == contract["n_d"])
    assert final.any(), "fixture lost the fully developed origin's final cell"
    assert (draws.var(axis=0) > 0).all()


def test_lognormal_draws_are_incremental_amounts_with_mixture_moments(contract):
    """The ratio draw must be scaled by premium (draws carry no measure
    declaration, so they must arrive as amounts): log(draw / premium) is the
    N(log mu, sigma) mixture, checked by both moments. A scorer returning raw
    ratio draws is off by log(1000) in the mean - far away, not adjacent."""
    post = fake_posterior_lognormal(contract, n_draws=4000)
    cells = training_index(contract)
    draws = scorer.draw_cells(
        contract, post, cells, rng=np.random.default_rng(17), variant="lognormal"
    )
    assert (draws > 0).all()

    prem = np.asarray(contract["premium"], dtype=float)[cells.w - 1]
    logratio = np.log(draws / prem[None, :])
    mu = scorer.mu_cells(contract, post, cells, variant="lognormal")
    sig = scorer.sigma_cells(post, cells)

    assert np.allclose(logratio.mean(axis=0), np.log(mu).mean(axis=0), atol=0.03)
    want = (sig**2).mean(axis=0) + np.log(mu).var(axis=0)
    assert np.allclose(logratio.var(axis=0), want, rtol=0.2)


def test_gaussian_density_is_normalized_on_the_amount_scale(contract):
    """One paid cell, one fixed posterior draw: the carried density must
    integrate to 1 over the amount. The only check that catches a wrong
    change of variable - the mutant that applies a premium carry to a density
    already on amounts integrates to premium^-1 times the truth."""
    post = fake_posterior_gaussian(contract, n_draws=1, seed=3)
    cells = training_index(contract)
    i = int(np.flatnonzero(cells.delta == 1)[-1])
    entry = _StubCompartmental(contract, post, "gaussian")
    mu = float(scorer.mu_cells(contract, post, _subset(cells, [i]), variant="gaussian")[0, 0])
    sig = float(post["sigma_paid"][0])

    def logpdf(xs):
        return entry.log_lik_at(_subset(cells, [i], value=np.asarray(xs, dtype=float)))[0]

    check_normalization(logpdf, lo=mu - 12 * sig, hi=mu + 12 * sig)

    def wrongly_carried(xs):
        return logpdf(xs) - np.log(PREMIUM)

    with pytest.raises(AssertionError, match="integrates to"):
        check_normalization(wrongly_carried, lo=mu - 12 * sig, hi=mu + 12 * sig)


def test_lognormal_density_is_normalized_on_the_amount_scale(contract):
    """The whole path for one paid increment: ratio density from the scorer,
    -log(premium) carry from the base class, integrated over the INCREMENT
    amount - which is the observation space the cell lives in, because the
    increment->cumulative anchor shift has Jacobian 1. The un-carried mutant
    (the ratio density read as an amount density) integrates to premium, not
    1: exactly the silent scale mixing the carry exists to prevent."""
    post = fake_posterior_lognormal(contract, n_draws=1, seed=5)
    cells = training_index(contract)
    paid_deep = np.flatnonzero((cells.delta == 1) & (cells.d > 1))
    i = int(paid_deep[-1])
    entry = _StubCompartmental(contract, post, "lognormal")

    prev = float(cells.prev_value[i])
    mu_ratio = float(
        scorer.mu_cells(contract, post, _subset(cells, [i]), variant="lognormal")[0, 0]
    )
    sig = float(post["sigma_paid"][0])
    median = PREMIUM * mu_ratio  # the increment amount's median
    lo, hi = median * np.exp(-12 * sig), median * np.exp(12 * sig)

    def logpdf(xs):
        one = _subset(cells, [i], value=prev + np.asarray(xs, dtype=float))
        return entry.log_lik_at(one)[0]

    check_normalization(logpdf, lo=lo, hi=hi)

    def uncarried(xs):
        one = _subset(cells, [i], value=prev + np.asarray(xs, dtype=float))
        return entry._log_lik_native(one)[0]

    with pytest.raises(AssertionError, match="integrates to"):
        check_normalization(uncarried, lo=lo, hi=hi)


def test_a_scorer_needs_a_fit_first(heldout):
    with pytest.raises(RuntimeError, match=r"call fit\(\) first"):
        Compartmental()._log_lik_native(None)
    with pytest.raises(RuntimeError, match=r"call fit\(\) first"):
        Compartmental()._draws_native(None, rng=np.random.default_rng(0))


# -- layer 2: the agreement gate ---------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["gaussian", "lognormal"])
def test_scorer_reproduces_the_fits_own_log_lik(variant):
    """THE gate, per variant. Score the training cells through the held-out
    code path and reproduce the fit's ``log_likelihood`` group elementwise.

    For the lognormal variant the reference vector covers only the rows that
    survived the keep mask, in contract order - which is precisely what
    ``_kept_rows_`` stores, and the noisy fixture guarantees the mask has
    teeth (negative deep-dev increments are dropped, so a scorer that forgot
    the subset would fail on shape before it could fail on values).
    """
    pytest.importorskip("cmdstanpy")

    tri = _joint_triangle(through=N_W, noise=0.05, seed=42)
    entry = Compartmental().fit(
        tri,
        variant=variant,
        backend="stan",
        chains=1,
        iter_warmup=300,
        iter_sampling=300,
        seed=11,
        # harness stage-1 settings: this is a wiring gate, not the retro
        target_accept=0.9,
        max_treedepth=12,
    )

    raw = np.asarray(entry.idata_.log_likelihood["log_lik"].values)
    reference = raw.reshape((raw.shape[0] * raw.shape[1], *raw.shape[2:]))

    cells = entry.training_cells()
    assert isinstance(cells, DeltaCellIndex)
    if variant == "lognormal":
        kept = entry._kept_rows_
        assert len(kept) < cells.n_cells, "the noisy fixture must drop some increments"
        cells = _subset(cells, kept)
    got = entry._log_lik_native(cells)

    assert got.shape == reference.shape
    agreement = np.abs(got - reference).max()

    # Tolerance: cmdstan writes its CSV at sig_figs=6, so the posterior draws
    # AND the reference come back rounded. CSR gets away with 1e-5 because its
    # parameters are all O(1) on the log scale; here the gaussian variant's
    # amount-scale mu (hundreds) and the lognormal's curve DIFFERENCE (which
    # amplifies relative error when increments are small) push the floor
    # higher. An index error is O(0.1) or worse, so 1e-3 still catches it.
    assert agreement < 1e-3

    # the gate must be able to fail: nudge ONE parameter and the disagreement
    # must sit orders of magnitude above the rounding floor
    perturbed = dict(entry._posterior())
    if variant == "gaussian":
        perturbed["ker"] = perturbed["ker"] * 1.01
    else:
        perturbed["b_oker"] = perturbed["b_oker"] + 0.01
    off = np.abs(
        scorer.log_lik_cells(entry.contract_, perturbed, cells, variant=variant) - reference
    ).max()
    assert off > 1e-2
    assert off > 100 * agreement
