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
    """Two-LOB company square with premium AND a case reserve, for the pooled
    NN contracts.

    The case channel is here for ``nn_paid_case``, whose contract names both
    fields: ``nn_data`` refuses an absent field by name, so without it that
    entry's fit would fail on the DATA, before the stubbed trainer runs, and the
    atomicity claim would never be exercised. The level is a share of the paid
    cumulative that shrinks with development - a reserve running down as payments
    replace it, never negative, and read off the cell it sits on rather than off
    any later one. The other five entries are fitted with ``feature_fields=()``
    and never read the channel.
    """
    rng = np.random.default_rng(start_year)
    n_w = 6
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_w))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_w))
    cum = np.cumsum(incr, axis=2)
    case = cum * np.linspace(0.6, 0.05, n_w)[None, None, :]
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    paid = make_multiline_triangle(
        "duckdb", {f"lob_{k}": cum[k] for k in range(2)}, premium_by_lob=prem, start_year=start_year
    )
    reserves = make_multiline_triangle(
        "duckdb",
        {f"lob_{k}": case[k] for k in range(2)},
        loss_field="case_reserve",
        start_year=start_year,
    )
    df = pd.concat([paid.execute(), reserves.execute()], ignore_index=True)
    return Triangle.from_long(df, measure="cumulative", backend="duckdb")


# -- bayesian: the sampler is the fallible step ---------------------------------


#: Bayesian entries whose fallible step is the sampler, stubbed via
#: ``entry._sample_stan``. ``england_verrall_odp`` and ``compartmental`` have
#: their own tests below - the first because it has a REAL post-contract
#: refusal worth using instead of a stub, the second because its torn state is
#: bigger than the contract.
BAYES_SAMPLER_ENTRIES = [
    "meyers_ccl",
    "meyers_csr",
    "clark_growth_curve",
    "guszcza_growth_curve",
]


@pytest.mark.parametrize("name", BAYES_SAMPLER_ENTRIES)
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

#: every NN entry, with the module whose ``train_ensemble`` name it calls. All
#: five share the pooled fit shape, so one parametrized test covers them - and
#: adding a row is what an entry needs to be covered at all, which is the gap
#: that let deeptriangle/mdn/resnet ship the bug after it was fixed everywhere
#: else (they were cloned from the pre-fix transformer). See
#: ``test_every_registered_entry_is_covered``.
NN_ENTRIES = [
    ("nn_transformer", "ibnr.gallery.nn.transformer.model.train_ensemble"),
    ("nn_transformer_ml", "ibnr.gallery.nn.transformer_ml.model.train_ensemble"),
    ("deeptriangle", "ibnr.gallery.nn.deeptriangle.model.train_ensemble"),
    ("mdn", "ibnr.gallery.nn.mdn.model.train_ensemble"),
    ("resnet", "ibnr.gallery.nn.resnet.model.train_ensemble"),
    ("nn_paid_case", "ibnr.gallery.nn.nn_paid_case.model.train_ensemble"),
]

#: the pooled fit call five of the six share. ``feature_fields=()``: the
#: fixture's second field is a case reserve, and deeptriangle's default names
#: reported_loss, which nn_data refuses by name. Atomicity is a property of the
#: fit lifecycle, not of the channel count.
DEFAULT_FIT_KWARGS = dict(loss_field="paid_loss", feature_fields=())

#: entry name -> the fit arguments it takes instead. ``nn_paid_case`` models two
#: fields, so it spells them ``paid_field``/``case_field`` and takes no
#: ``feature_fields`` at all - passing the shared call would be a TypeError,
#: which is a failure of the test rather than of the entry. A row here is keyed
#: to ``NN_ENTRIES``, which ``test_every_registered_entry_is_covered`` holds to
#: the registry.
FIT_KWARGS: dict[str, dict] = {
    "nn_paid_case": dict(paid_field="paid_loss", case_field="case_reserve"),
}


@pytest.mark.parametrize("name,trainer", NN_ENTRIES, ids=[n for n, _ in NN_ENTRIES])
def test_nn_failed_training_leaves_the_previous_fit_intact(monkeypatch, name, trainer):
    """A failed ensemble fit must not leave the new pool's contract and
    normalizer over the old pool's networks.

    The NN entries are the most exposed of all: ``at_cohort(segment)`` builds
    its per-cohort view straight off ``entry.contract_`` (gallery/nn/
    _heldout.py::cohort_contract), so a torn entry hands ``index_into`` the new
    pool's cohort identity while ``_heldout_draws`` runs the old pool's
    networks - every guard passes and the draws are plausible."""
    pytest.importorskip("torch")
    monkeypatch.setattr(trainer, FitThenBoom(("models-A", "history-A")))
    entry = gallery.get(name)()
    kwargs = FIT_KWARGS.get(name, DEFAULT_FIT_KWARGS)
    entry.fit(nn_triangle(2000), **kwargs)
    before = snapshot(entry)

    with pytest.raises(RuntimeError, match="boom"):
        entry.fit(nn_triangle(1990), **kwargs)

    assert_untouched(entry, before)


# -- the coverage gate ---------------------------------------------------------

#: entries pinned by a test of their own above, each because it has a REAL
#: post-contract refusal worth exercising in place of a stub
NAMED_TEST_ENTRIES = {
    "england_verrall_odp",  # pearson_phi's informative-cells dof check
    "compartmental",  # variant switch + the lognormal keep-mask bookkeeping
    "sur",  # the 1/sqrt(C) whitening's positivity guard
    "copula_glm",  # the lognormal marginal's positive-increment guard
    "clark",  # parameter count vs cells, after the curve MLE
}

#: covered end to end elsewhere rather than here: mack's failed refit is
#: asserted straight through ``predict_at`` (cohort A's draws stay
#: bit-identical, cohort B's cells are refused as the wrong cohort)
COVERED_ELSEWHERE = {"mack": "tests/test_mack_heldout.py"}


def test_every_registered_entry_is_covered():
    """Every entry in the registry must have an atomicity test - this gate is
    the only thing that catches a NEW one that does not.

    Not paranoia. ``fit()`` was made atomic across all 11 entries that existed
    on 2026-07-27 (PR #47), and within hours ``deeptriangle``, ``mdn`` and
    ``resnet`` landed from parallel sessions carrying the identical pre-fix
    ordering - they had been cloned from the transformer's source *before* the
    fix. Nothing failed: no merge conflict (the changes were concurrent, not
    overlapping), and CI runs lint and docs only. The hand-written entry lists
    in this file were themselves the gap, since a new entry simply never
    appeared in them.

    So the lists above are asserted to BE the registry. Adding an entry without
    an atomicity test now fails here, naming the entry - which is the one
    moment the author is looking at exactly this concern.
    """
    covered = (
        set(BAYES_SAMPLER_ENTRIES)
        | {name for name, _ in NN_ENTRIES}
        | NAMED_TEST_ENTRIES
        | set(COVERED_ELSEWHERE)
    )
    registered = set(gallery.list())

    missing = sorted(registered - covered)
    assert not missing, (
        f"gallery entries with no fit() atomicity test: {missing}. Every entry must "
        "build its contract (and every other piece of fitted state) into LOCALS and "
        "assign self.* only after the fallible step - sampler, estimator or trainer - "
        "has returned; see gallery/nn/transformer/model.py or deterministic/mack. Add "
        "the entry to BAYES_SAMPLER_ENTRIES or NN_ENTRIES if it fits that shape, else "
        "give it a test of its own and list it in NAMED_TEST_ENTRIES."
    )
    stale = sorted(covered - registered)
    assert not stale, (
        f"these names are listed here but are not registered gallery entries: {stale}. "
        "A renamed or removed entry leaves a row that silently tests nothing."
    )
    # and the per-entry fit arguments describe entries this file actually fits:
    # a row keyed to a name NN_ENTRIES does not carry governs nothing, and the
    # entry it was written for silently gets the shared call instead.
    orphan = sorted(set(FIT_KWARGS) - {name for name, _ in NN_ENTRIES})
    assert not orphan, f"FIT_KWARGS rows naming entries this file does not fit: {orphan}"
