"""The GalleryEntry cohort vocabulary, checked against the registry itself.

Before 0.5.0 a caller could not ask a fitted entry which cohorts it answers for,
so every consumer rebuilt that fact by hand: ``predict``/``realized_ultimates``
took a segment dict on the NN entries and none anywhere else, and
``realized_ultimates`` was not on the ABC at all. This file is the guard on the
single vocabulary that replaced it - ``cohorts()``, ``cohort_index()`` and one
``segment`` parameter meaning one thing everywhere.

**Parametrized over ``gallery.list()``, never over a hardcoded name list**, per
the entries-cloned-from-stale-templates rule: a new entry has to satisfy the
signature facts here on the day it registers, not on the day somebody remembers
to add it. The bucket census below is what keeps that honest - an entry in
neither bucket fails HERE rather than silently skipping the behaviour tests.

Fast by construction: the signature half needs no fit at all, the behaviour half
uses the four entries that need neither cmdstan nor torch, and the six that do
need cmdstan are fitted here with their sampler stubbed (last section). The NN
entries' behaviour lives in ``test_nn_segment_contract.py`` (torch).
"""

from __future__ import annotations

import datetime as dt
import inspect

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.gallery.entry import GalleryEntry
from ibnr.triangle import Triangle

from .conftest import make_cohort_triangle, make_multiline_triangle

#: one fitted cohort: `segment=None` and the fitted key mean the same thing
SINGLE_COHORT = {
    "clark",
    "clark_growth_curve",
    "compartmental",
    "copula_glm",
    "england_verrall_odp",
    "guszcza_growth_curve",
    "mack",
    "meyers_ccl",
    "meyers_csr",
    "sur",
}
#: a pooled fit: `segment=None` is the whole panel, a segment is one cohort
POOLED = {"deeptriangle", "mdn", "nn_transformer", "nn_transformer_ml", "resnet"}
#: of the single-cohort entries, the ones whose fit needs cmdstan
NEEDS_STAN = {
    "clark_growth_curve",
    "compartmental",
    "england_verrall_odp",
    "guszcza_growth_curve",
    "meyers_ccl",
    "meyers_csr",
}


def test_every_registered_entry_is_in_exactly_one_bucket():
    """The census that makes the parametrization honest.

    Without it a new entry would join ``gallery.list()``, pass the signature
    tests below, and silently skip every behaviour test in the repo - which is
    the failure mode this repo has already paid for once.
    """
    assert set(gallery.list()) == SINGLE_COHORT | POOLED
    assert not SINGLE_COHORT & POOLED
    assert NEEDS_STAN <= SINGLE_COHORT


@pytest.mark.parametrize("name", gallery.list())
def test_predict_takes_a_leading_segment(name):
    """``predict``'s first parameter is ``segment``, defaulting to None.

    Leading and defaulted rather than keyword-only because the NN entries
    already took it positionally, and defaulted so no 0.4.0 caller moves.
    """
    # unbound, so parameter 0 is `self`
    params = list(inspect.signature(gallery.get(name).predict).parameters.values())[1:]
    assert params[0].name == "segment"
    assert params[0].default is None
    assert params[0].kind is not inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("name", gallery.list())
def test_realized_ultimates_is_on_the_abc_with_one_signature(name):
    """``(full_triangle, segment=None)`` on every entry, and inherited from the ABC.

    The signature that diverged: NN entries took ``segment``, the other ten did
    not, and a cross-model outcome table had to branch on ``family``.
    """
    method = gallery.get(name).realized_ultimates
    params = list(inspect.signature(method).parameters.values())[1:]  # drop `self`
    assert [p.name for p in params[:2]] == ["full_triangle", "segment"]
    assert params[1].default is None
    assert "realized_ultimates" in GalleryEntry.__abstractmethods__


@pytest.mark.parametrize("name", gallery.list())
def test_evaluate_takes_the_same_segment(name):
    """``evaluate`` scores against ``predict``, so it selects the same way."""
    params = inspect.signature(gallery.get(name).evaluate).parameters
    assert "segment" in params
    assert params["segment"].default is None


@pytest.mark.parametrize("name", gallery.list())
def test_cohorts_is_implemented(name):
    """Every entry defines ``cohorts()`` itself - the ABC has no default.

    A defaulted ``contract_["segment"]`` would hand a pooled entry that forgot to
    override a plausible answer from a key that happens to exist.
    """
    cls = gallery.get(name)
    assert "cohorts" not in cls.__abstractmethods__
    assert cls.cohorts is not GalleryEntry.cohorts


def _looks_like_config(param: str) -> bool:
    """Is ``param`` a near-miss spelling of ``config=``?

    ``registry.register`` keys on the exact string ``"config"``, so a config
    object arriving under any other name is invisible to it: the entry keeps
    ``config_class = None``, registration is happy, and a caller who found the
    entry through ``gallery.get(name)`` has no route to the type ``fit`` wants.
    Case and underscores are stripped, then ``cfg`` or ``conf`` anywhere in the
    name is enough - ``config`` itself starts with ``conf``, so the two stems
    cover every spelling worth worrying about (``cfg``, ``_cfg``, ``Config``,
    ``model_conf``, ``configuration``). Deliberately a substring test rather
    than a prefix one: ``model_conf`` normalizes to ``modelconf``, which starts
    with neither stem.
    """
    p = param.replace("_", "").lower()
    return "cfg" in p or "conf" in p


def test_the_near_miss_heuristic_catches_the_spellings_it_claims_to():
    """The heuristic above is itself the check, so it gets its own test.

    Its first version was ``p.startswith("c") and "config" in p``, which cannot
    fire on ``cfg`` at all - ``"config" in "cfg"`` is False - so the check the
    docstring advertised was vacuous for the exact spelling it named.
    """
    assert all(
        _looks_like_config(p)
        for p in ("config", "cfg", "_cfg", "Config", "model_conf", "configuration")
    )
    # and fires on none of the parameters the 15 entries' fit signatures really take
    real = {
        p
        for name in gallery.list()
        for p in inspect.signature(gallery.get(name).fit).parameters
        if p != "config"
    }
    assert real and not any(_looks_like_config(p) for p in real)


@pytest.mark.parametrize("name", gallery.list())
def test_config_class_is_declared_exactly_when_fit_takes_one(name):
    """The route from ``gallery.get(name)`` to the type ``fit(config=)`` wants.

    Also pins the parameter's SPELLING: the registration check keys on
    ``"config"``, so an entry naming it ``cfg=`` would slip past it silently -
    which is what :func:`_looks_like_config` is for.
    """
    cls = gallery.get(name)
    params = inspect.signature(cls.fit).parameters
    takes_config = "config" in params
    if takes_config:
        assert inspect.isclass(cls.config_class)
        # constructible with no arguments - it is what fit() falls back to
        assert cls.config_class() is not None
    else:
        assert cls.config_class is None
        assert not any(_looks_like_config(p) for p in params)


# -- behaviour of `segment=`, on the entries that need no extra ------------------

CUM = np.array(
    [
        [100.0, 150.0, 175.0, 180.0],
        [110.0, 165.0, 190.0, np.nan],
        [120.0, 180.0, np.nan, np.nan],
        [130.0, np.nan, np.nan, np.nan],
    ]
)
PREMIUM = {"lob_a": np.array([200.0] * 4), "lob_b": np.array([400.0] * 4)}
SEGMENT = {"company_code": "0001"}


def _fit(name, backend_name):
    """One fitted entry per family that needs neither cmdstan nor torch."""
    if name in ("sur", "copula_glm"):
        tri = make_multiline_triangle(
            backend_name, {"lob_a": CUM, "lob_b": CUM * 2.0}, premium_by_lob=PREMIUM
        )
        return gallery.fit(name, tri, loss_field="paid_loss")
    tri = make_cohort_triangle(backend_name, CUM, segment=SEGMENT)
    if name == "clark":
        return gallery.fit(name, tri, loss_field="paid_loss", method="ldf")
    return gallery.fit(name, tri, loss_field="paid_loss")


FAST = ["mack", "clark", "sur", "copula_glm"]


@pytest.mark.parametrize("name", FAST)
def test_cohorts_names_the_fitted_cohort(name, backend_name):
    """One cohort, and it is the segment identity the fit was built on."""
    entry = _fit(name, backend_name)
    cohorts = entry.cohorts()
    assert len(cohorts) == 1
    assert cohorts[0] == SEGMENT


@pytest.mark.parametrize("name", FAST)
def test_segment_none_and_the_fitted_key_agree(name, backend_name):
    """``predict()`` and ``predict(segment=<the fitted cohort>)`` are the same call.

    Not a tautology on an "accept and discard" implementation - that passes here
    too, which is what the refusal tests below are for.
    """
    entry = _fit(name, backend_name)
    a = entry.predict(seed=0)
    b = entry.predict(segment=entry.cohorts()[0], seed=0)
    assert a.samples.shape == b.samples.shape
    assert list(a.targets["label"]) == list(b.targets["label"])
    assert np.allclose(a.samples, b.samples)


@pytest.mark.parametrize("name", FAST)
def test_a_segment_naming_no_cohort_is_refused_by_predict(name, backend_name):
    """A typo'd cohort key must RAISE, not score the fitted cohort.

    Accepting and discarding is the inert-parameter bug class: the number that
    comes back is entirely plausible and belongs to a different company.
    """
    entry = _fit(name, backend_name)
    with pytest.raises(ValueError) as excinfo:
        entry.predict(segment={"company_code": "9999"})
    message = str(excinfo.value)
    assert "9999" in message  # the supplied dict
    assert "company_code" in message  # the fit's own cohort key


@pytest.mark.parametrize("name", FAST)
def test_a_segment_naming_no_cohort_is_refused_by_realized_ultimates(name, backend_name):
    entry = _fit(name, backend_name)
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.realized_ultimates(_triangle(backend_name, name), segment={"company_code": "9999"})


@pytest.mark.parametrize("name", FAST)
def test_a_segment_naming_no_cohort_is_refused_by_evaluate(name, backend_name):
    entry = _fit(name, backend_name)
    observed = np.asarray(entry.realized_ultimates(_triangle(backend_name, name)), dtype=float)
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.evaluate(observed, segment={"company_code": "9999"})


def _triangle(backend_name, name):
    if name in ("sur", "copula_glm"):
        return make_multiline_triangle(
            backend_name, {"lob_a": CUM, "lob_b": CUM * 2.0}, premium_by_lob=PREMIUM
        )
    return make_cohort_triangle(backend_name, CUM, segment=SEGMENT)


@pytest.mark.parametrize("name", FAST)
def test_realized_ultimates_agrees_with_and_without_the_key(name, backend_name):
    entry = _fit(name, backend_name)
    tri = _triangle(backend_name, name)
    a = np.asarray(entry.realized_ultimates(tri), dtype=float)
    b = np.asarray(entry.realized_ultimates(tri, segment=entry.cohorts()[0]), dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_cohort_index_none_in_none_out(backend_name):
    """None means "whatever segment=None means for this entry" - never cohort 0."""
    entry = _fit("mack", backend_name)
    assert entry.cohort_index(None) is None


def test_cohort_index_rejects_an_unknown_column(backend_name):
    """An unknown column names the FIT's schema, not the triangle's.

    The gap-1 message did the opposite - it named ``company_name``, a column the
    caller had just passed, without saying what the fit was keyed on.
    """
    entry = _fit("mack", backend_name)
    with pytest.raises(KeyError) as excinfo:
        entry.cohort_index({"not_a_column": 1})
    assert "company_code" in str(excinfo.value)


def test_cohort_index_rejects_a_non_mapping(backend_name):
    entry = _fit("mack", backend_name)
    with pytest.raises(TypeError, match="must be a mapping"):
        entry.cohort_index(("company_code", "0001"))


def test_cohort_index_before_fit_says_to_fit_first():
    """The order matters: unfitted is a lifecycle error, not a segment error."""
    with pytest.raises(RuntimeError, match="fit"):
        gallery.get("mack")().cohort_index({"company_code": "0001"})


# -- delivery of `segment=` on the Bayesian entries, sampler stubbed ------------
#
# The six ``NEEDS_STAN`` entries had every SIGNATURE fact above checked and no
# BEHAVIOURAL one: their only `segment=` test (``test_meyers_csr.py``) needs
# cmdstan and the Schedule P mart, so it runs in no leg of a pull request.
# Measured: deleting ``self.cohort_index(segment)`` from ``meyers_ccl.predict``
# left the whole 1239-test suite green - the inert-parameter bug class, where a
# typo'd cohort key is accepted, discarded, and answered with the fitted
# cohort's perfectly plausible numbers.
#
# Stubbing the sampler is what puts these on every PR: it is the same lever
# ``test_fit_atomicity.py`` pulls, and it works here because the refusal is
# raised by ``cohort_index`` BEFORE ``predict`` reads the posterior. That
# ordering is itself the thing under test - a resolver called after the
# ``idata_ is None`` check would report "call fit() first" for a typo.

STAN_ENTRIES = sorted(NEEDS_STAN)

#: strictly increasing paid share of ultimate by dev year: positive increments
#: everywhere, which the ODP and growth-curve preps require.
_GROWTH = (0.30, 0.55, 0.75, 0.88, 0.95)
STAN_COHORT = {"company_code": "0001"}


def _stan_triangle(start_year: int = 2000, n: int = 5) -> Triangle:
    """Single-cohort run-off staircase carrying paid/reported/premium AND a segment.

    Shaped like ``test_fit_atomicity.bayes_triangle`` - reported > paid and every
    increment positive, so all six data preps accept it, including
    compartmental's joint paid+outstanding contract. The one addition is the
    ``company_code`` column: without a segment column ``cohorts()`` answers
    ``[{}]``, and a test built on that cannot tell a delivered segment from a
    discarded one.
    """
    premium, loss_ratio, paid_share = 1000.0, 0.70, 0.80
    rows = []
    for i in range(n):
        for j in range(n - i):
            paid = premium * loss_ratio * _GROWTH[j]
            for field, value in (
                ("paid_loss", paid),
                ("reported_loss", paid / paid_share),
                ("earned_premium", premium),
            ):
                rows.append(
                    {
                        **STAN_COHORT,
                        "origin_period": dt.date(start_year + i, 1, 1),
                        "dev_lag": 12 * (j + 1),
                        "eval_date": dt.date(start_year + i + j, 12, 31),
                        "field": field,
                        "value": value,
                    }
                )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


def _stub_fitted(name: str):
    """A Bayesian entry fitted with ``_sample_stan`` replaced by a placeholder.

    No cmdstan, no compile, no MCMC. The placeholder posterior is never read:
    every assertion below is about a refusal raised ahead of it.
    """
    entry = gallery.get(name)()
    entry._sample_stan = lambda *args, **kwargs: ("stub-posterior", None)
    return entry.fit(_stan_triangle())


@pytest.mark.parametrize("name", STAN_ENTRIES)
def test_stan_entry_cohorts_names_the_fitted_cohort(name):
    """``cohorts()`` answers with the identity the triangle carried."""
    assert _stub_fitted(name).cohorts() == [STAN_COHORT]


@pytest.mark.parametrize("name", STAN_ENTRIES)
def test_stan_entry_predict_refuses_a_segment_naming_no_cohort(name):
    """Both refusals, because they come from different branches of the resolver:
    an unknown COLUMN names the fit's schema, a known column with an unmatched
    VALUE reports how many cohorts it hit."""
    entry = _stub_fitted(name)
    with pytest.raises(KeyError, match="unknown segment column"):
        entry.predict(segment={"wrong": "x"})
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.predict(segment={"company_code": "9999"})


@pytest.mark.parametrize("name", STAN_ENTRIES)
def test_stan_entry_realized_ultimates_refuses_a_segment_naming_no_cohort(name):
    """The one that would otherwise return a NUMBER for the wrong company.

    ``realized_ultimates`` reads the full triangle and never touches the
    posterior, so an unwired ``segment=`` here is silent in the strongest sense:
    the array comes back the right shape, finite, and belonging to whoever the
    fit was actually built on.
    """
    entry = _stub_fitted(name)
    tri = _stan_triangle()
    with pytest.raises(KeyError, match="unknown segment column"):
        entry.realized_ultimates(tri, segment={"wrong": "x"})
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.realized_ultimates(tri, segment={"company_code": "9999"})


@pytest.mark.parametrize("name", STAN_ENTRIES)
def test_stan_entry_evaluate_refuses_a_segment_naming_no_cohort(name):
    """``evaluate`` selects through ``predict``, so it must refuse identically."""
    entry = _stub_fitted(name)
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.evaluate(np.zeros(6), segment={"company_code": "9999"})


@pytest.mark.parametrize("name", STAN_ENTRIES)
def test_stan_entry_accepts_its_own_key(name):
    """The complement: the resolver is selective, not merely hostile.

    Without this a ``cohort_index`` that refused every segment would pass all
    three refusal tests above.
    """
    entry = _stub_fitted(name)
    assert entry.cohort_index(STAN_COHORT) == 0
    assert entry.cohort_index(None) is None
    # and the outcome path runs end to end on the fitted key
    np.testing.assert_allclose(
        np.asarray(entry.realized_ultimates(_stan_triangle()), dtype=float),
        np.asarray(entry.realized_ultimates(_stan_triangle(), segment=STAN_COHORT), dtype=float),
        equal_nan=True,
    )
