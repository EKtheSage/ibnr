"""CCL can score cells it was not trained on, and the answer is checkable.

The same three layers as ``test_heldout_scorer_csr.py`` - closed form, the
agreement gate, the measure carry - plus the surface that is new with CCL: the
**AR term**. A cell's location reads the previous origin's residual at the same
dev lag, ``rho * (logloss[w-1, d] - mu[w-1, d])``, resolved against the
TRAINING rows. That term can be dropped, read at the wrong row, or scaled by
the wrong parameter, and in every case the result is finite, smoothly varying
and plausible - so the tests here make each of those mutations fail:

* the closed-form scalar loop includes the AR term, over a fake posterior whose
  ``mu`` is deliberately inconsistent with the recurrence, so the residuals are
  large and a dropped or mis-indexed term cannot hide;
* a ``rho = 0`` posterior must move the answer on every cell that has a
  predecessor, and must NOT move it on cells that do not (the mirror of
  ``model.stan``'s ``prev_idx == 0`` branch);
* the slow agreement gate fits real data WITH noise, because on the noiseless
  staircase the model fits exactly, the residuals collapse, and a zeroed
  ``rho`` would pass the gate - a negative control that cannot fail is not a
  control.

Draws are covered here too (CSR splits them into ``test_predicts_heldout.py``,
but the base-class scale-carry tests there are not CCL's to repeat - only the
scorer-level closed forms are).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr.gallery.bayesian.meyers_ccl import scorer
from ibnr.gallery.bayesian.meyers_ccl.model import MeyersCCL
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout
from ibnr.kernels.contract import stan_data
from ibnr.kernels.densities import normal_lpdf
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal, training_index
from ibnr.triangle.core import Triangle

N_W = N_D = 6
PREMIUM = 1000.0


def _triangle(*, through: int, noise: float = 0.0, seed: int = 29) -> Triangle:
    """The fitted cohort: ``FIT_CO``, reported loss (CCL's default field), with
    premium.

    ``noise`` multiplies each cell by ``exp(noise * eps)``. The noiseless
    staircase is log-additive in ``w`` and ``d``, which CCL fits EXACTLY -
    residuals collapse toward zero and the AR term becomes invisible. The slow
    agreement gate needs ``noise > 0`` so that a zeroed ``rho`` is detectable;
    the fast layers use fake posteriors and do not care.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, N_W + 1):
        for d in range(1, N_D + 1):
            if w + d - 1 > through:
                continue
            cum = PREMIUM * 0.65 * (1.0 - np.exp(-0.6 * d)) * (1.0 + 0.03 * w)
            if noise:
                cum *= float(np.exp(noise * rng.standard_normal()))
            eval_date = dt.date(2010 + w - 1 + d - 1, 12, 31)
            for f, v in (("reported_loss", float(cum)), ("earned_premium", PREMIUM)):
                rows.append(
                    {
                        "lob": "FIT_CO",
                        "origin_period": dt.date(2010 + w - 1, 1, 1),
                        "dev_lag": 12 * d,
                        "eval_date": eval_date,
                        "field": f,
                        "value": v,
                    }
                )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


@pytest.fixture(scope="module")
def contract() -> dict:
    return stan_data(
        _triangle(through=N_W), loss_field="reported_loss", premium_field="earned_premium"
    )


def fake_posterior(contract: dict, n_draws: int = 7, seed: int = 0) -> dict:
    """Draws with DISTINCT, non-symmetric values per index.

    ``mu`` is deliberately NOT consistent with the recurrence over the other
    parameters: the scorer must READ the fit's own ``mu`` at the predecessor
    row, and a version that recomputed it from ``logelr``/``alpha``/``beta``
    would disagree with the scalar loop here. The inconsistency also makes the
    residuals O(0.5), so the AR term is large enough that dropping it fails
    every comparison.
    """
    rng = np.random.default_rng(seed)
    n_w, n_d, len_data = contract["n_w"], contract["n_d"], contract["len_data"]
    return {
        "logelr": rng.normal(-0.4, 0.05, size=n_draws),
        "alpha": np.tile(np.linspace(0.0, 0.5, n_w), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_w)),
        "beta": np.tile(np.linspace(-1.7, 0.0, n_d), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_d)),
        "rho": rng.uniform(0.25, 0.65, size=n_draws),
        "sig": np.tile(np.linspace(0.30, 0.05, n_d), (n_draws, 1)),
        "mu": np.tile(np.linspace(5.2, 7.4, len_data), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, len_data)),
    }


def _prev_row_of(contract: dict) -> dict[tuple[int, int], int]:
    """0-based row of each training ``(w, d)``, for the scalar loops."""
    return {
        (int(a), int(b)): i
        for i, (a, b) in enumerate(zip(contract["w"], contract["d"], strict=True))
    }


# -- layer 1: closed form -----------------------------------------------------


def test_matches_a_scalar_loop_over_the_stan_formula(contract):
    """``mu[i] = logprem[i] + logelr + alpha[w[i]] + beta[d[i]]`` plus, when the
    previous origin's cell is in training, ``rho * (logloss[prev] - mu[prev])``
    - written out one cell at a time so the vectorized version has something
    independent to disagree with."""
    post = fake_posterior(contract)
    cells = training_index(contract)
    got = scorer.log_lik_cells(contract, post, cells)

    prem = np.asarray(contract["premium"], dtype=float)
    logloss = np.asarray(contract["logloss"], dtype=float)
    row_of = _prev_row_of(contract)
    expected = np.empty((len(post["logelr"]), cells.n_cells))
    for s in range(expected.shape[0]):
        for i in range(cells.n_cells):
            w, d = int(cells.w[i]) - 1, int(cells.d[i]) - 1
            mu = np.log(prem[w]) + post["logelr"][s] + post["alpha"][s, w] + post["beta"][s, d]
            prev = row_of.get((w, d + 1))  # (w-1, d) in 1-based terms
            if prev is not None:
                mu += post["rho"][s] * (logloss[prev] - post["mu"][s, prev])
            expected[s, i] = normal_lpdf(np.log(cells.value[i]), mu, post["sig"][s, d])

    assert got.shape == expected.shape
    assert np.allclose(got, expected)


def test_the_ar_term_is_alive_on_this_fixture(contract):
    """The guard on the closed form: zeroing ``rho`` must move every cell that
    has a predecessor and no cell that does not. If it moved nothing, the AR
    branch of the scalar loop above would be dead code and the test vacuous."""
    post = fake_posterior(contract)
    cells = training_index(contract)
    no_rho = {**post, "rho": np.zeros_like(post["rho"])}

    with_term = scorer.mu_cells(contract, post, cells)
    without = scorer.mu_cells(contract, no_rho, cells)

    row_of = _prev_row_of(contract)
    has_prev = np.array(
        [(int(a) - 1, int(b)) in row_of for a, b in zip(cells.w, cells.d, strict=True)]
    )
    assert has_prev.any() and (~has_prev).any(), "fixture must have both kinds of cell"
    assert (np.abs(with_term - without)[:, has_prev] > 1e-6).all()
    assert np.array_equal(with_term[:, ~has_prev], without[:, ~has_prev])


def test_a_missing_predecessor_drops_the_ar_term_like_prev_idx_zero(contract):
    """``kernels.contract`` sets ``prev_idx = 0`` for a ``(w-1, d)`` hole, and
    ``model.stan`` then skips the adjustment - the model's own definition of a
    cell in that position. The scorer must mirror it, not refuse or invent."""
    post = fake_posterior(contract)
    # (w=5, d=6): its predecessor (4, 6) is beyond the training staircase
    orphan = CellIndex(
        w=np.array([5]),
        d=np.array([6]),
        value=np.array([700.0]),
        prev_value=np.array([650.0]),
        premium=np.array([PREMIUM]),
    )
    assert (4, 6) not in _prev_row_of(contract)

    no_rho = {**post, "rho": np.zeros_like(post["rho"])}
    assert np.array_equal(
        scorer.mu_cells(contract, post, orphan), scorer.mu_cells(contract, no_rho, orphan)
    )
    # and it scores rather than raising
    assert np.isfinite(scorer.log_lik_cells(contract, post, orphan)).all()


def test_indices_are_not_interchangeable(contract):
    """A guard on the guard: if swapping w for d changed nothing, the closed-form
    test above would be vacuous. It has to matter."""
    post = fake_posterior(contract)
    cells = training_index(contract)
    swapped = CellIndex(
        w=cells.d, d=cells.w, value=cells.value, prev_value=cells.prev_value, premium=cells.premium
    )
    assert (cells.w != cells.d).any(), "fixture cannot distinguish w from d"

    a = scorer.log_lik_cells(contract, post, cells)
    b = scorer.log_lik_cells(contract, post, swapped)
    assert not np.allclose(a, b)


def test_missing_posterior_variable_names_what_is_needed(contract):
    """``rho`` and ``mu`` are the variables CSR does not need - a scorer copied
    from CSR's would silently not read them, so their absence must be loud."""
    for name in ("rho", "mu"):
        post = fake_posterior(contract)
        del post[name]
        with pytest.raises(KeyError, match=name):
            scorer.log_lik_cells(contract, post, training_index(contract))


def test_nonpositive_loss_is_refused_not_nan(contract):
    """CCL is lognormal. A zero cell would give ``-inf`` silently, and the mart
    has 316k zero cells, so it has to be an error - the same rule the contract
    already applies at fit time."""
    cells = training_index(contract)
    zeroed = CellIndex(
        w=cells.w,
        d=cells.d,
        value=np.where(np.arange(cells.n_cells) == 0, 0.0, cells.value),
        prev_value=cells.prev_value,
        premium=cells.premium,
    )
    with pytest.raises(ValueError, match="non-positive loss"):
        scorer.log_lik_cells(contract, fake_posterior(contract), zeroed)


def test_a_posterior_mu_of_the_wrong_width_is_refused(contract):
    """``mu`` is gathered by training-row index; a wrong width would gather from
    the wrong rows (or wrap) and still return plausible numbers."""
    post = fake_posterior(contract)
    post["mu"] = post["mu"][:, :-1]
    with pytest.raises(ValueError, match="training rows"):
        scorer.log_lik_cells(contract, post, training_index(contract))


# -- draws --------------------------------------------------------------------


def test_draw_cells_is_exp_of_a_normal_at_mu_and_sig(contract):
    """``model.stan``'s ``logloss[i] ~ normal(mu[i], sig[d[i]])`` read forwards.
    Reproduced exactly by seeding the same generator - which pins the draws to
    the SAME ``mu`` the density uses, AR term included."""
    post = fake_posterior(contract, n_draws=50)
    cells = training_index(contract)

    got = scorer.draw_cells(contract, post, cells, rng=np.random.default_rng(3))

    mu = scorer.mu_cells(contract, post, cells)
    sig = np.asarray(post["sig"])[:, cells.d - 1]
    want = np.exp(np.random.default_rng(3).normal(mu, sig))
    assert np.allclose(got, want)
    assert got.shape == (50, cells.n_cells)


def test_the_index_matters_for_draws_too(contract):
    post = fake_posterior(contract, n_draws=30)
    cells = training_index(contract)
    swapped = CellIndex(
        w=cells.d,
        d=cells.w,
        value=cells.value,
        prev_value=cells.prev_value,
        premium=cells.premium,
    )
    a = scorer.draw_cells(contract, post, cells, rng=np.random.default_rng(0))
    b = scorer.draw_cells(contract, post, swapped, rng=np.random.default_rng(0))
    assert not np.allclose(a, b)


def test_draws_do_not_inherit_the_density_refusal_of_a_zero_cell(contract):
    """A zero held-out cell has no lognormal DENSITY at the outcome but a
    perfectly well-defined lognormal PREDICTIVE - why the mixins are separate."""
    post = fake_posterior(contract, n_draws=20)
    cells = training_index(contract)
    zeroed = CellIndex(
        w=cells.w,
        d=cells.d,
        value=np.where(np.arange(cells.n_cells) == 0, 0.0, cells.value),
        prev_value=cells.prev_value,
        premium=cells.premium,
    )
    with pytest.raises(ValueError, match="non-positive loss"):
        scorer.log_lik_cells(contract, post, zeroed)
    got = scorer.draw_cells(contract, post, zeroed, rng=np.random.default_rng(1))
    assert np.isfinite(got).all() and (got > 0).all()


# -- layer 3: the measure carry and the entry wiring ---------------------------


class _StubCCL(MeyersCCL):
    """A fitted CCL without a sampler: the scorer only needs contract + draws."""

    def __init__(self, contract, post):
        super().__init__()
        self.contract_ = contract
        self._post = post

    def _log_lik_native(self, cells):
        return scorer.log_lik_cells(self.contract_, self._post, cells)

    def _draws_native(self, cells, *, rng):
        return scorer.draw_cells(self.contract_, self._post, cells, rng=rng)


def test_log_lik_at_carries_the_density_to_the_amount_scale(contract):
    """CCL reports a density on ``log C``; the leaderboard sums densities on
    ``C``. The carry is exactly ``- log C``, applied by the base class."""
    post = fake_posterior(contract)
    entry = _StubCCL(contract, post)
    cells = training_index(contract)

    native = entry._log_lik_native(cells)
    carried = entry.log_lik_at(cells)

    assert np.allclose(carried, native - np.log(cells.value)[None, :])
    assert entry.heldout_measure == "log_amount"
    assert isinstance(entry, ScoresHeldout)


def test_held_out_cells_index_into_the_fitted_contract(contract):
    """The end-to-end shape: cells from ``next_diagonal`` are indexed against
    the fit and scored - and on the next diagonal every scorable cell has its
    predecessor on the last training diagonal, so the AR term fires on ALL of
    them. Zeroing ``rho`` must therefore move every single column."""
    full = _triangle(through=N_W + 1)
    cells = next_diagonal(
        full, as_of="2015-12-31", fields="reported_loss", premium_field="earned_premium"
    )
    idx = index_into(cells, contract, field="reported_loss")

    assert idx.n_cells == cells.n_cells
    assert idx.w.min() >= 2, "next-diagonal cells all have a previous origin"

    post = fake_posterior(contract)
    entry = _StubCCL(contract, post)
    out = entry.log_lik_at(cells, field="reported_loss")
    assert out.shape == (7, cells.n_cells)
    assert np.isfinite(out).all()

    no_rho = _StubCCL(contract, {**post, "rho": np.zeros_like(post["rho"])})
    assert (np.abs(out - no_rho.log_lik_at(cells, field="reported_loss")) > 1e-9).all()


def test_ccl_declares_both_capabilities():
    entry = MeyersCCL()
    assert isinstance(entry, ScoresHeldout)
    assert isinstance(entry, PredictsHeldout)
    assert entry.heldout_measure == "log_amount"
    assert entry.heldout_draw_scale == "cumulative"


def test_ccl_needs_a_fit_before_scoring():
    with pytest.raises(RuntimeError, match="call fit\\(\\) first"):
        MeyersCCL()._log_lik_native(None)
    with pytest.raises(RuntimeError, match="call fit\\(\\) first"):
        MeyersCCL()._draws_native(None, rng=np.random.default_rng(0))


# -- layer 2: the agreement gate ---------------------------------------------


@pytest.mark.slow
def test_scorer_reproduces_the_fits_own_log_lik():
    """THE gate. Score the training cells through the held-out code path and the
    result must equal the fit's ``log_likelihood`` group elementwise.

    Fitted on NOISY data, deliberately: the noiseless staircase is log-additive
    in ``w`` and ``d``, CCL fits it exactly, the residuals collapse, and the
    ``rho`` negative control below could not fail. Noise keeps the residuals -
    and with them the AR term - large enough to see.
    """
    pytest.importorskip("cmdstanpy")
    from ibnr.gallery.bayesian.meyers_ccl.model import pooled

    tri = _triangle(through=N_W, noise=0.05)
    entry = MeyersCCL().fit(
        tri,
        loss_field="reported_loss",
        premium_field="earned_premium",
        backend="stan",
        chains=1,
        iter_warmup=300,
        iter_sampling=300,
        seed=11,
    )

    raw = np.asarray(entry.idata_.log_likelihood["log_lik"].values)
    reference = raw.reshape((raw.shape[0] * raw.shape[1], *raw.shape[2:]))
    got = entry._log_lik_native(entry.training_cells())

    assert got.shape == reference.shape
    agreement = np.abs(got - reference).max()

    # Tolerance is set by cmdstan's OUTPUT PRECISION (CSV at sig_figs=6), same
    # as the CSR gate; measured there at ~3e-7.
    assert agreement < 1e-5

    post = {name: pooled(entry.idata_, name) for name in scorer.REQUIRED_DRAWS}

    # Negative control 1, the CCL-specific one: a scorer that dropped the AR
    # term entirely would produce exactly this. It must be visibly wrong.
    no_rho = {**post, "rho": np.zeros_like(post["rho"])}
    off_rho = np.abs(
        scorer.log_lik_cells(entry.contract_, no_rho, entry.training_cells()) - reference
    ).max()
    assert off_rho > 1e-3
    assert off_rho > 100 * agreement

    # Negative control 2, same as CSR's: a 1% shift in one parameter separates
    # from the agreement above by orders of magnitude.
    perturbed = {**post, "beta": post["beta"] + 0.01}
    off = np.abs(
        scorer.log_lik_cells(entry.contract_, perturbed, entry.training_cells()) - reference
    ).max()
    assert off > 1e-3
    assert off > 100 * agreement
