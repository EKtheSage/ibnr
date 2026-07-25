"""CSR can score cells it was not trained on, and the answer is checkable.

The scorer is the first piece of milestone 6 that can be wrong in a way nothing
else notices. It reproduces the model's likelihood at new ``(w, d)`` positions,
so an index slip - 0-based where the contract is 1-based, ``w`` where ``d``
belongs, the wrong ``speedup`` row - yields finite, smoothly-varying, entirely
plausible log densities. Every downstream number would still compute.

So the tests come in three layers:

1. **Closed form.** ``log_lik_cells`` against a hand-written scalar loop over
   ``model.stan``'s formula, at draws chosen so a transposed index cannot
   coincide with the right answer.
2. **The agreement gate.** Score the TRAINING cells through the held-out code
   path; the result must reproduce the fit's own ``log_lik`` elementwise. This
   is the test with teeth, and it is paired with a negative control - perturb
   one parameter and assert the gate fails - because a gate nobody has watched
   fail is not a gate.
3. **The measure carry.** ``log_lik_at`` must differ from ``_log_lik_native``
   by exactly ``-log(value)``, since CSR is a density on log loss.

Layers 1 and 3 need no sampler: the scorer takes plain arrays, which is why it
is a free function rather than a method.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr.gallery.bayesian.meyers_csr import scorer
from ibnr.gallery.bayesian.meyers_csr.model import MeyersCSR
from ibnr.gallery.entry import ScoresHeldout
from ibnr.kernels.contract import stan_data
from ibnr.kernels.densities import normal_lpdf
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal, training_index
from ibnr.triangle.core import Triangle

N_W = N_D = 6
PREMIUM = 1000.0


def _triangle(*, through: int, backend=None) -> Triangle:
    """The fitted cohort: ``FIT_CO``, paid loss, with premium.

    It carries a segment column deliberately. A fixture with no cohort label
    cannot exercise the identity checks at all - and an unlabelled triangle is
    not what the mart looks like.
    """
    return _segmented_triangle("FIT_CO", through=through, with_premium=True, backend=backend)


def _segmented_triangle(
    lob,
    *,
    through: int,
    field: str = "paid_loss",
    with_premium: bool = False,
    backend=None,
) -> Triangle:
    """Same staircase, optionally carrying an ``lob`` column and/or a different
    loss field - for the identity checks, where the numbers are irrelevant and
    only the labels matter."""
    rows = []
    for w in range(1, N_W + 1):
        for d in range(1, N_D + 1):
            if w + d - 1 > through:
                continue
            cum = PREMIUM * 0.65 * (1.0 - np.exp(-0.6 * d)) * (1.0 + 0.03 * w)
            eval_date = dt.date(2010 + w - 1 + d - 1, 12, 31)
            entries = [(field, float(cum))]
            if with_premium:
                entries.append(("earned_premium", PREMIUM))
            for f, v in entries:
                row = {
                    "origin_period": dt.date(2010 + w - 1, 1, 1),
                    "dev_lag": 12 * d,
                    "eval_date": eval_date,
                    "field": f,
                    "value": v,
                }
                rows.append({"lob": lob, **row} if lob is not None else row)
    kwargs = {} if backend is None else {"backend": backend}
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative", **kwargs)


@pytest.fixture(scope="module")
def contract() -> dict:
    """A fit on cohort ``FIT_CO``, paid loss. The segment column is present so
    the identity checks have something real to compare against."""
    return stan_data(_triangle(through=N_W), loss_field="paid_loss", premium_field="earned_premium")


def fake_posterior(contract: dict, n_draws: int = 7, seed: int = 0) -> dict:
    """Draws with DISTINCT, non-symmetric values per index.

    Deliberately not random-and-similar: ``alpha`` and ``beta`` have different
    lengths and clearly different magnitudes, and ``sig`` decreases in dev as
    CSR's construction forces. A transposed or off-by-one index therefore lands
    on a visibly wrong number rather than a plausible neighbouring one.
    """
    rng = np.random.default_rng(seed)
    n_w, n_d = contract["n_w"], contract["n_d"]
    gamma = rng.uniform(0.01, 0.05, size=n_draws)
    return {
        "logelr": rng.normal(-0.4, 0.05, size=n_draws),
        "alpha": np.tile(np.linspace(0.0, 0.5, n_w), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_w)),
        "beta": np.tile(np.linspace(-1.7, 0.0, n_d), (n_draws, 1))
        + rng.normal(0, 1e-3, (n_draws, n_d)),
        "speedup": np.cumprod(
            np.column_stack([np.ones(n_draws), np.tile(1.0 - gamma, (n_w - 1, 1)).T]), axis=1
        ),
        "sig": np.tile(np.linspace(0.30, 0.05, n_d), (n_draws, 1)),
    }


# -- layer 1: closed form -----------------------------------------------------


def test_matches_a_scalar_loop_over_the_stan_formula(contract):
    """``mu[i] = logprem[i] + logelr + alpha[w[i]] + beta[d[i]] * speedup[w[i]]``
    and ``log_lik[i] = normal_lpdf(logloss[i] | mu[i], sig[d[i]])``, written out
    one cell at a time so the vectorized version has something independent to
    disagree with."""
    post = fake_posterior(contract)
    cells = training_index(contract)
    got = scorer.log_lik_cells(contract, post, cells)

    prem = np.asarray(contract["premium"], dtype=float)
    expected = np.empty((len(post["logelr"]), cells.n_cells))
    for s in range(expected.shape[0]):
        for i in range(cells.n_cells):
            w, d = int(cells.w[i]) - 1, int(cells.d[i]) - 1
            mu = (
                np.log(prem[w])
                + post["logelr"][s]
                + post["alpha"][s, w]
                + post["beta"][s, d] * post["speedup"][s, w]
            )
            expected[s, i] = normal_lpdf(np.log(cells.value[i]), mu, post["sig"][s, d])

    assert got.shape == expected.shape
    assert np.allclose(got, expected)


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
    post = fake_posterior(contract)
    del post["speedup"]
    with pytest.raises(KeyError, match="speedup"):
        scorer.log_lik_cells(contract, post, training_index(contract))


def test_nonpositive_loss_is_refused_not_nan(contract):
    """CSR is lognormal. A zero cell would give ``-inf`` silently, and the mart
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


# -- layer 3: the measure carry ----------------------------------------------


class _StubCSR(MeyersCSR):
    """A fitted CSR without a sampler: the scorer only needs contract + draws."""

    def __init__(self, contract, post):
        super().__init__()
        self.contract_ = contract
        self._post = post

    def _log_lik_native(self, cells):
        return scorer.log_lik_cells(self.contract_, self._post, cells)


def test_log_lik_at_carries_the_density_to_the_amount_scale(contract):
    """CSR reports a density on ``log C``; the leaderboard sums densities on
    ``C``. The carry is exactly ``- log C``, and it must be applied by the base
    class rather than by the entry."""
    post = fake_posterior(contract)
    entry = _StubCSR(contract, post)
    cells = training_index(contract)

    native = entry._log_lik_native(cells)
    carried = entry.log_lik_at(cells)

    assert np.allclose(carried, native - np.log(cells.value)[None, :])
    assert entry.heldout_measure == "log_amount"
    assert isinstance(entry, ScoresHeldout)


def test_the_carry_cannot_be_skipped_by_an_entry(contract):
    """``log_lik_at`` is concrete on the mixin precisely so an entry cannot
    return an unconverted density. If a subclass could override the measure to
    'amount' and have it accepted silently, the global ELPD would mix scales."""
    post = fake_posterior(contract)
    entry = _StubCSR(contract, post)
    entry.heldout_measure = "not_a_measure"
    with pytest.raises(ValueError, match="heldout_measure must be one of"):
        entry.log_lik_at(training_index(contract))


def test_held_out_cells_index_into_the_fitted_contract(contract):
    """The end-to-end shape: cells from ``next_diagonal`` are indexed against
    the fit and scored, with one column per scorable cell."""
    full = _triangle(through=N_W + 1)
    cells = next_diagonal(
        full, as_of="2015-12-31", fields="paid_loss", premium_field="earned_premium"
    )
    idx = index_into(cells, contract, field="paid_loss")

    assert idx.n_cells == cells.n_cells
    assert idx.w.min() >= 1 and idx.d.max() <= contract["n_d"]

    entry = _StubCSR(contract, fake_posterior(contract))
    out = entry.log_lik_at(cells, field="paid_loss")
    assert out.shape == (7, cells.n_cells)
    assert np.isfinite(out).all()


def test_training_prev_value_is_the_previous_DEVELOPMENT_not_the_previous_ORIGIN(contract):
    """The contract carries two different "previous cell" notions and they must
    not be confused.

    ``contract["prev_idx"]`` points at ``(w-1, d)`` - the previous ORIGIN at the
    same development - because that is CCL's accident-year AR(1) link.
    ``prev_value`` means ``(w, d-1)``, the previous DEVELOPMENT, because that is
    what differencing a cumulative triangle needs. An earlier version of
    ``training_index`` reused ``prev_idx`` for ``prev_value``.

    ``meyers_csr`` never reads ``prev_value``, so no CSR test could see it; the
    first scorer to difference cumulatives - ODP, Clark, compartmental - would
    have silently subtracted a neighbouring accident year instead of the cell's
    own history, on every cell but the first origin.
    """
    idx = training_index(contract)
    loss = np.asarray(contract["loss"], dtype=float)
    row_of = {
        (int(w), int(d)): i
        for i, (w, d) in enumerate(zip(contract["w"], contract["d"], strict=True))
    }

    for i in range(idx.n_cells):
        w, d = int(idx.w[i]), int(idx.d[i])
        if d == 1:
            assert idx.prev_value[i] == 0.0
            continue
        assert idx.prev_value[i] == loss[row_of[(w, d - 1)]]

    # and the two notions genuinely differ on this fixture, or the test is empty
    differing = [
        i
        for i in range(idx.n_cells)
        if int(idx.w[i]) > 1
        and int(idx.d[i]) > 1
        and (int(idx.w[i]) - 1, int(idx.d[i])) in row_of
        and loss[row_of[(int(idx.w[i]) - 1, int(idx.d[i]))]] != idx.prev_value[i]
    ]
    assert differing, "fixture cannot distinguish previous-origin from previous-dev"


def test_index_into_refuses_a_single_WRONG_cohort(contract):
    """The sharp version of the cohort check, and the one that matters.

    Requiring the cells to agree with *each other* is not enough: one entirely
    wrong company agrees with itself perfectly, its origin dates and dev lags
    match exactly, it indexes cleanly, and it produces a complete, plausible
    ELPD for a cohort the fit never saw. Nothing downstream could tell.

    Only comparing against the contract's own recorded identity catches it,
    which is why ``kernels.contract`` stores ``segment``/``fields`` at all.
    """
    other = _segmented_triangle("OTHER_CO", through=N_W + 1)
    cells = next_diagonal(other, as_of="2015-12-31", fields="paid_loss")

    assert set(cells.frame["lob"]) == {"OTHER_CO"}  # internally consistent...
    with pytest.raises(ValueError, match="was trained on"):  # ... and still refused
        index_into(cells, contract)


def test_index_into_refuses_a_field_the_fit_was_not_trained_on(contract):
    """A paid-loss fit handed reported-loss cells would evaluate one quantity's
    likelihood against another's data - same shape, same indices, wrong data."""
    reported = _segmented_triangle(None, through=N_W + 1, field="reported_loss")
    cells = next_diagonal(reported, as_of="2015-12-31", fields="reported_loss")
    with pytest.raises(ValueError, match="not what this fit was trained on"):
        index_into(cells, contract, field="reported_loss")


def test_field_defaults_to_the_one_the_fit_was_trained_on(contract):
    """The caller should not have to repeat what the contract already knows -
    and a caller who can pass the field is a caller who can pass the wrong one."""
    full = _triangle(through=N_W + 1)
    cells = next_diagonal(full, as_of="2015-12-31", fields="paid_loss")
    assert (
        index_into(cells, contract).n_cells
        == index_into(cells, contract, field="paid_loss").n_cells
    )


def test_index_into_refuses_cells_from_more_than_one_cohort(contract):
    """``(w, d)`` does not identify a cell - two companies or two lines share
    origin dates and development lags exactly.

    A contract is single-cohort by construction, so multi-cohort held-out cells
    scored against one fit would produce a complete, plausible ELPD for a
    mixture of cohorts. Nothing downstream could detect it.
    """
    rows = []
    for lob in ("wc", "ca"):
        for w in range(1, N_W + 1):
            for d in range(1, N_D + 1):
                if w + d - 1 > N_W + 1:
                    continue
                rows.append(
                    {
                        "lob": lob,
                        "origin_period": dt.date(2010 + w - 1, 1, 1),
                        "dev_lag": 12 * d,
                        "eval_date": dt.date(2010 + w - 1 + d - 1, 12, 31),
                        "field": "paid_loss",
                        "value": float(PREMIUM * 0.6 * (1 - np.exp(-0.6 * d))),
                    }
                )
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative")
    cells = next_diagonal(tri, as_of="2015-12-31", fields="paid_loss")

    assert set(cells.frame["lob"]) == {"wc", "ca"}
    # subsumed by the identity check: neither cohort is the fitted one, so the
    # error names what the fit actually expects rather than just "too many"
    with pytest.raises(ValueError, match="was trained on"):
        index_into(cells, contract, field="paid_loss")


def test_cells_carrying_extra_fields_are_narrowed_not_refused(contract):
    """A single-field fit handed a multi-field panel scores its own field only.

    This is the safe half of the field contract, and worth pinning next to the
    refusals: the panel legitimately carries premium (and, for a compartmental
    run, reported loss) alongside paid. Narrowing is right; what must never
    happen is scoring the other field's numbers, which
    ``test_index_into_refuses_a_field_the_fit_was_not_trained_on`` covers.
    """
    full = _triangle(through=N_W + 1)
    cells = next_diagonal(full, as_of="2015-12-31", fields=["paid_loss", "earned_premium"])
    assert set(cells.frame["field"]) == {"paid_loss", "earned_premium"}

    idx = index_into(cells, contract)
    paid_only = cells.frame[cells.frame["field"] == "paid_loss"]
    assert idx.n_cells == len(paid_only)
    assert np.allclose(idx.value, paid_only["value"].to_numpy())


def test_a_cell_outside_the_fitted_cohort_is_an_error(contract):
    """Scoring a fit against another cohort's cells would silently produce
    numbers, since the indices exist either way."""
    full = _triangle(through=N_W + 1)
    cells = next_diagonal(full, as_of="2015-12-31", fields="paid_loss")
    shifted = {**contract, "origin_periods": [dt.date(1990 + i, 1, 1) for i in range(N_W)]}
    with pytest.raises(ValueError, match="not in the fitted contract"):
        index_into(cells, shifted, field="paid_loss")


# -- layer 2: the agreement gate ---------------------------------------------


@pytest.mark.slow
def test_scorer_reproduces_the_fits_own_log_lik(tmp_path):
    """THE gate. Score the training cells through the held-out code path and the
    result must equal the fit's ``log_likelihood`` group elementwise.

    This is what makes a held-out number trustworthy: the scorer is a second,
    independent implementation of the same density, and the fit's own value is
    the reference. A scorer that quietly indexes wrong cannot pass it.
    """
    pytest.importorskip("cmdstanpy")
    from ibnr.gallery.bayesian.meyers_csr.model import pooled

    tri = _triangle(through=N_W)
    entry = MeyersCSR().fit(
        tri,
        loss_field="paid_loss",
        premium_field="earned_premium",
        backend="stan",
        chains=1,
        iter_warmup=300,
        iter_sampling=300,
        seed=11,
    )

    # Stan names the group's variable `log_lik`; both ports name it `obs`. That
    # non-uniformity is exactly why the scorer reads `posterior` instead - here
    # we read the group directly, because it is the reference being checked.
    raw = np.asarray(entry.idata_.log_likelihood["log_lik"].values)
    reference = raw.reshape((raw.shape[0] * raw.shape[1], *raw.shape[2:]))
    got = entry._log_lik_native(entry.training_cells())

    assert got.shape == reference.shape
    agreement = np.abs(got - reference).max()

    # Tolerance is set by cmdstan's OUTPUT PRECISION, not by the scorer. Stan
    # writes its CSV at sig_figs=6 by default, and both the posterior draws and
    # the reference log_lik come back through it, so recomputing from rounded
    # draws cannot agree better than ~1e-6 relative. Measured here: ~3e-7.
    # Raising cmdstan's sig_figs would tighten it; 6 figures on a log density is
    # far more than ELPD needs, so it is documented rather than chased.
    assert agreement < 1e-5

    # The gate is only worth having if it can fail, and the margin is the point:
    # a 1% shift in ONE parameter separates from the agreement above by orders
    # of magnitude, so the tolerance is nowhere near wide enough to hide a real
    # index error (which would be O(0.1) or worse).
    perturbed = {name: pooled(entry.idata_, name) for name in scorer.REQUIRED_DRAWS}
    perturbed["beta"] = perturbed["beta"] + 0.01
    off = np.abs(
        scorer.log_lik_cells(entry.contract_, perturbed, entry.training_cells()) - reference
    ).max()
    assert off > 1e-3
    assert off > 100 * agreement
