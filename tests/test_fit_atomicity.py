"""fit() is atomic on EVERY gallery entry: a failed refit changes nothing.

The bug class (first confirmed on ``deterministic/mack``, whose fix and
end-to-end test live in ``test_mack_heldout``): every entry's ``fit()`` builds
its contract before the fallible step - the sampler, the FGLS/MLE estimator,
the ensemble trainer - and any of those legitimately refuses real cohorts.
Assigning ``contract_`` (or ``variant_``, the held-out declarations, the
normalization stats, ...) before that raise leaves a TORN entry: the NEW
cohort's contract over the OLD cohort's fit. ``index_into`` checks identity
against the contract, so ``predict_at``/``log_lik_at`` on the new cohort's
cells then pass every guard and return plausible numbers built from the wrong
cohort's fit - exactly the cross-cohort error the identity checks exist to
prevent.

Pattern per entry: fit cohort A (samplers/trainers stubbed where the entry has
one, so the fast suite runs no MCMC and no torch training), then ``fit`` a
cohort B engineered to fail AFTER B's contract is built - inside the window
where the pre-fix ordering had already mutated the entry. Where the family has
a real post-contract refusal (ODP's dispersion dof check, SUR's positivity
guard, the copula's increment guard, Clark's parameter-count check) the test
uses it rather than a stub, because that is the failure a mechanical
retrospective actually hits.

The assertion is deliberately blunt: NOTHING on the entry may change - every
instance attribute must still be the very same object. That is the strongest
form of "the failed refit's identity cannot leak into the surviving fit", it
needs no per-entry knowledge of which attribute pairs must stay coherent, and
it is what keeps ``predict``/``predict_at``/``log_lik_at`` answering for
cohort A. Mutation-verified: every test here fails on the pre-fix assignment
order (contract assigned before the fallible step) and passes after it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.triangle import Triangle

from .conftest import make_multiline_triangle
from .test_copula_glm import simulate_lognormal_square
from .test_sur import simulate_cl_square

# -- the shared assertion ------------------------------------------------------


def snapshot(entry) -> dict:
    """The entry's instance state, attribute name -> object."""
    return dict(vars(entry))


def assert_untouched(entry, before: dict) -> None:
    """Every attribute is still the very same object as before the failed fit.

    Identity (``is``), not equality: a failed refit must not rebuild anything,
    and identity sidesteps ambiguous ``==`` on arrays/idata. New attributes are
    as much a leak as replaced ones (e.g. the compartmental variant's held-out
    declarations appearing from a failed lognormal refit)."""
    after = vars(entry)
    assert set(after) == set(before), (
        f"failed fit() added/removed attributes: {set(after) ^ set(before)}"
    )
    changed = [k for k in before if after[k] is not before[k]]
    assert not changed, f"failed fit() mutated {changed}"


class FitThenBoom:
    """Sampler/trainer stub: first call returns ``result``, later calls raise.

    Signature-agnostic on purpose - it stands in for ``_sample_stan`` (which
    ``fit`` calls with backend kwargs) and for ``train_ensemble`` alike."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("boom: the refit's fallible step failed")
        return self.result


# -- fixtures ------------------------------------------------------------------

#: strictly increasing paid share of ultimate by dev year: positive increments
#: everywhere, which the ODP/lognormal preps require.
_G = (0.30, 0.55, 0.75, 0.88, 0.95)


def bayes_triangle(start_year: int, n: int = 5) -> Triangle:
    """Single-cohort run-off staircase carrying paid/reported/premium.

    The ``test_nuts_sampler`` fixture, parameterized by ``start_year`` (so two
    calls yield two distinct cohorts) and by size (``n=2`` yields a triangle
    whose 3 cells are exactly the dof-check refusals of the ODP and Clark
    families). Survives every Bayesian entry's data prep, including
    compartmental's joint paid+outstanding contract (reported > paid)."""
    premium, loss_ratio, paid_share = 1000.0, 0.70, 0.80
    rows = []
    for i in range(n):
        for j in range(n - i):
            paid = premium * loss_ratio * _G[j]
            for field, value in (
                ("paid_loss", paid),
                ("reported_loss", paid / paid_share),
                ("earned_premium", premium),
            ):
                rows.append(
                    {
                        "origin_period": dt.date(start_year + i, 1, 1),
                        "dev_lag": 12 * (j + 1),
                        "eval_date": dt.date(start_year + i + j, 12, 31),
                        "field": field,
                        "value": value,
                    }
                )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


def nn_triangle(start_year: int) -> Triangle:
    """Two-LOB company square with premium, for the pooled NN contracts."""
    rng = np.random.default_rng(start_year)
    n_w = 6
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_w))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_w))
    cum = np.cumsum(incr, axis=2)
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    return make_multiline_triangle("duckdb", lobs, premium_by_lob=prem, start_year=start_year)


# -- bayesian: the sampler is the fallible step ---------------------------------


@pytest.mark.parametrize("name", ["meyers_ccl", "meyers_csr", "clark_growth_curve"])
def test_bayesian_failed_sampler_leaves_the_previous_fit_intact(name):
    entry = gallery.get(name)()
    entry._sample_stan = FitThenBoom(("idata-A", None))
    entry.fit(bayes_triangle(2010))
    before = snapshot(entry)

    with pytest.raises(RuntimeError, match="boom"):
        entry.fit(bayes_triangle(2000))  # valid cohort B; its sampler fails

    assert_untouched(entry, before)


def test_odp_failed_dispersion_check_leaves_the_previous_fit_intact():
    """The REAL post-contract refusal: ``pearson_phi`` needs more informative
    cells than the model has parameters, and that runs after ``odp_stan_data``
    accepted the cohort - no stub required for the failing leg."""
    entry = gallery.get("england_verrall_odp")()
    entry._sample_stan = FitThenBoom(("idata-A", None))
    entry.fit(bayes_triangle(2010))
    before = snapshot(entry)

    with pytest.raises(ValueError, match="informative cells"):
        entry.fit(bayes_triangle(2000, n=2))  # 3 cells vs 3 parameters

    assert_untouched(entry, before)


def test_compartmental_failed_variant_switch_leaves_the_previous_fit_intact():
    """The richest torn state in the gallery: a failed gaussian -> lognormal
    refit must not leave the lognormal declarations (``heldout_measure``,
    ``heldout_draw_scale``), ``variant_`` or the keep-mask bookkeeping stamped
    over the surviving gaussian posterior - a density would then be carried on
    the WRONG measure while every identity check still passes."""
    entry = gallery.get("compartmental")()
    entry._sample_stan = FitThenBoom(("idata-A", None))
    entry.fit(bayes_triangle(2010), variant="gaussian")
    before = snapshot(entry)
    assert before["heldout_measure"] == "amount"

    with pytest.raises(RuntimeError, match="boom"):
        entry.fit(bayes_triangle(2000), variant="lognormal")

    assert_untouched(entry, before)
    assert entry.heldout_measure == "amount"
    assert entry.variant_ == "gaussian"


# -- statistical: the estimator refuses real cohorts after the contract builds --


def test_sur_failed_positivity_guard_leaves_the_previous_fit_intact():
    cum = simulate_cl_square(np.random.default_rng(0), n_w=12, rho=0.0)
    lobs = {f"lob_{k}": cum[k] for k in range(cum.shape[0])}
    tri_a = make_multiline_triangle("duckdb", lobs, start_year=2000)
    entry = gallery.get("sur")().fit(tri_a)
    before = snapshot(entry)

    bad = cum.copy()
    bad[0, 0, 0] = 0.0  # 1/sqrt(C) whitening cannot take a zero cumulative
    lobs_b = {f"lob_{k}": bad[k] for k in range(bad.shape[0])}
    tri_b = make_multiline_triangle("duckdb", lobs_b, start_year=1990)
    with pytest.raises(ValueError, match="non-positive cumulative"):
        entry.fit(tri_b)

    assert_untouched(entry, before)


def test_copula_glm_failed_increment_guard_leaves_the_previous_fit_intact():
    cum = simulate_lognormal_square(np.random.default_rng(0), n_w=8)
    prem = {f"lob_{k}": np.full(cum.shape[1], 1000.0) for k in range(cum.shape[0])}
    tri_a = make_multiline_triangle(
        "duckdb",
        {f"lob_{k}": cum[k] for k in range(cum.shape[0])},
        premium_by_lob=prem,
        start_year=2000,
    )
    entry = gallery.get("copula_glm")().fit(tri_a)
    before = snapshot(entry)

    bad = cum.copy()
    bad[0, 0, 1] = bad[0, 0, 0] * 0.5  # a negative increment; lognormal refuses
    tri_b = make_multiline_triangle(
        "duckdb",
        {f"lob_{k}": bad[k] for k in range(bad.shape[0])},
        premium_by_lob=prem,
        start_year=1990,
    )
    with pytest.raises(ValueError, match="non-positive increment"):
        entry.fit(tri_b)

    assert_untouched(entry, before)


def test_clark_mle_failed_dof_check_leaves_the_previous_fit_intact():
    entry = gallery.get("clark")().fit(bayes_triangle(2010))
    before = snapshot(entry)

    # 3 cells vs the Cape Cod model's 3 parameters (1 ELR + 2 curve): the dof
    # check runs after the contract is built and after the curve MLE.
    with pytest.raises((ValueError, RuntimeError)):
        entry.fit(bayes_triangle(2000, n=2))

    assert_untouched(entry, before)


# -- nn: the ensemble trainer is the fallible step ------------------------------


def test_nn_transformer_failed_training_leaves_the_previous_fit_intact(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(
        "ibnr.gallery.nn.transformer.model.train_ensemble",
        FitThenBoom(("models-A", "history-A")),
    )
    entry = gallery.get("nn_transformer")()
    entry.fit(nn_triangle(2000), loss_field="paid_loss")
    before = snapshot(entry)

    with pytest.raises(RuntimeError, match="boom"):
        entry.fit(nn_triangle(1990), loss_field="paid_loss")

    assert_untouched(entry, before)


def test_nn_transformer_ml_failed_training_leaves_the_previous_fit_intact(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(
        "ibnr.gallery.nn.transformer_ml.model.train_ensemble",
        FitThenBoom(("models-A", "history-A")),
    )
    entry = gallery.get("nn_transformer_ml")()
    entry.fit(nn_triangle(2000), loss_field="paid_loss")
    before = snapshot(entry)

    with pytest.raises(RuntimeError, match="boom"):
        entry.fit(nn_triangle(1990), loss_field="paid_loss")

    assert_untouched(entry, before)
