"""An entry that needs premium must REFUSE a missing one by name, before it works.

Two spellings of the same mistake, one case per premium-requiring entry each:
``premium_field=None`` (the caller switched exposure off) and a
``premium_field`` naming a column the triangle does not carry (the caller left
the Schedule P default in place on their own data). Both used to surface as
whatever the code happened to hit once the missing premium reached it.

The contract this file pins:

1. the failure is a ``ValueError``, not whatever the contract builder happened
   to raise when a ``None`` reached it;
2. its message contains "premium", names the entry (or the contract function
   that needs it) and says what to pass instead;
3. it is raised from argument validation - before any compiler, sampler,
   network or optimizer runs.

Point 3 is what lets one file cover every Bayesian entry in the FAST suite
with no cmdstan and no ``[bayesian]`` extra installed: a refusal that fires
before ``import cmdstanpy`` needs neither. It is also the user-facing point.
``meyers_ccl``/``meyers_csr``/``england_verrall_odp`` used to spend a full
Stan compile to reach a ``KeyError('logprem')``, or - with the extra absent -
died at ``ModuleNotFoundError`` for arviz/cmdstanpy, which tells a caller
nothing about the argument they got wrong.

What the twelve entries did before (2026-07-28), all of them raw:

    copula_glm            KeyError 'premium', from ``c["premium"]`` well after
                          multiline_data legitimately omitted it (``sur``
                          shares that contract and is premium-free)
    compartmental         TypeError inside compartmental_stan_data
    deeptriangle, mdn, nn_transformer, nn_transformer_ml, resnet
                          TypeError "'NoneType' object is not iterable" inside
                          nn_data (nn_transformer_ml reaches it through
                          nn_company_data, which delegates there first)
    meyers_ccl, meyers_csr, england_verrall_odp
                          KeyError 'logprem' after the Stan compile, because
                          stan_data/odp_stan_data omit it when premium_field
                          is None (guszcza shares stan_data and reads it that
                          way deliberately)
    clark_growth_curve, guszcza_growth_curve
                          already refused by name; they are the voice the
                          other ten now match

The requiring set is DERIVED, from the same table that governs the
annotations: an entry takes ``premium_field`` and is not listed in
test_field_annotations.ACCEPTS_NONE as accepting ``None`` there. So a new
entry joins these cases by joining the registry, and ``clark`` - where
``None`` is a documented, supported call - stays out by staying in the table.
"""

from __future__ import annotations

import datetime as dt
import inspect

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.triangle.core import Triangle
from tests.test_field_annotations import ACCEPTS_NONE


def _premium_requiring() -> list[str]:
    names = []
    for name in gallery.list():
        fit = gallery.get(name).fit
        if "premium_field" not in inspect.signature(fit).parameters:
            continue  # mack, sur: no premium anywhere in the API
        if "premium_field" in ACCEPTS_NONE.get(name, frozenset()):
            continue  # clark: None is a documented, supported call
        names.append(name)
    return sorted(names)


#: entries whose contract needs a MULTI-LINE triangle. This is what keeps
#: ``copula_glm``'s case able to REPRODUCE the defect it pins: with the fix
#: reverted, the two-line fixture gives the documented pre-fix KeyError
#: 'premium', while the one-line fixture never reaches premium at all -
#: ``multiline_data`` refuses the shape first ("multiline models need >= 2
#: lines of business, got ['CA']", measured). Both fixtures fail pre-fix, but
#: only this one fails for the reason under test. The default stays one line
#: because the single-cohort Stan contracts refuse a multi-line triangle
#: outright; the NN entries would take either.
NEEDS_TWO_LINES = frozenset({"copula_glm"})


def _staircase(lines: tuple[str, ...] = ("CA",)) -> Triangle:
    """A 6x6 run-off triangle per line, carrying both loss fields, no premium.

    Rich enough that every entry's contract builder would BUILD if premium
    were supplied - the refusals under test are about the argument, not about
    the data being unusable. Lines share one observed-cell pattern and develop
    strictly upward, which is what the cell-wise dependence and lognormal
    marginal contracts require.
    """
    n = 6
    rows = [
        {
            "line_of_business": line,
            "origin_period": dt.date(2010 + w - 1, 1, 1),
            "dev_lag": 12 * d,
            "eval_date": dt.date(2010 + w + d - 2, 12, 31),
            "field": field,
            "value": lscale * scale * (1000.0 + 150.0 * w) * (1.0 - np.exp(-0.55 * d)),
        }
        for lscale, line in enumerate(lines, start=1)
        for w in range(1, n + 1)
        for d in range(1, n + 1)
        for field, scale in (("paid_loss", 1.0), ("reported_loss", 1.25))
        if w + d - 1 <= n
    ]
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


def test_the_derivation_found_the_premium_requiring_entries():
    """Tripwire against a silently-empty derivation.

    Parametrizing over ``[]`` collects nothing and passes green, so the set
    the cases run over is asserted here by name. Membership, not equality: a
    new premium-requiring entry joins the cases without editing this list,
    while dropping one of these twelve breaks by name.
    """
    assert {
        "clark_growth_curve",
        "compartmental",
        "copula_glm",
        "deeptriangle",
        "england_verrall_odp",
        "guszcza_growth_curve",
        "mdn",
        "meyers_ccl",
        "meyers_csr",
        "nn_transformer",
        "nn_transformer_ml",
        "resnet",
    } <= set(_premium_requiring())


#: who a refusal is allowed to name as its source. The five NN entries share
#: one check in ``nn_data``, so naming that function is naming the thing that
#: needs premium; every other entry refuses under its own name.
RAISER = {name: "nn_data" for name in ("deeptriangle", "mdn", "resnet")}
RAISER["nn_transformer"] = "nn_data"
RAISER["nn_transformer_ml"] = "nn_data"


def _assert_names_the_requirement(message: str, name: str) -> None:
    """Contract point 2: the message identifies the requirement, not just "premium".

    Asserted separately from the ``match=`` pattern because "premium" alone is
    satisfied by any message with the word in it - including ``copula_glm``'s
    pre-fix ``KeyError('premium')``, whose string contains it. What makes the
    new messages useful is the pair: the PARAMETER the caller got wrong, and
    the entry (or shared contract function) that wants it.
    """
    assert "premium_field" in message, (
        f"{name}'s refusal must name the parameter to pass, got: {message!r}"
    )
    raiser = RAISER.get(name, name)
    assert raiser in message, (
        f"{name}'s refusal must name {raiser!r} as what needs the premium, got: {message!r}"
    )


@pytest.mark.parametrize("name", _premium_requiring())
def test_premium_requiring_entry_refuses_none_by_name(name):
    if gallery.get(name).family == "nn":
        pytest.importorskip("torch")  # fit() imports torch before nn_data's refusal
    lines = ("CA", "WC") if name in NEEDS_TWO_LINES else ("CA",)
    with pytest.raises(ValueError, match="premium") as excinfo:
        gallery.fit(name, _staircase(lines), premium_field=None)
    _assert_names_the_requirement(str(excinfo.value), name)


@pytest.mark.parametrize("name", _premium_requiring())
def test_premium_requiring_entry_names_a_premium_field_it_cannot_find(name):
    """The other spelling of the same mistake: a field name, but the wrong one.

    ``premium_field`` DEFAULTS to ``"earned_premium"`` on all twelve entries -
    right for the Schedule P mart, wrong for anyone else's column names - so
    this arrives without the caller typing the argument at all, which makes it
    likelier than the ``None`` case above.

    ``Triangle.select_fields`` is a filter, so a name the triangle does not
    carry yields an empty frame rather than an error, and each contract has to
    refuse it explicitly. The single-cohort ones always did; ``nn_data`` did
    so for its LOSS fields only, and answered an absent premium field with
    "no usable cohorts after screening" - every cohort dropped one at a time
    by the premium screen, naming neither premium nor the field, and pointing
    at a ``dropped`` frame that the raise path never builds.
    """
    if gallery.get(name).family == "nn":
        pytest.importorskip("torch")
    lines = ("CA", "WC") if name in NEEDS_TWO_LINES else ("CA",)
    with pytest.raises(ValueError, match="premium") as excinfo:
        gallery.fit(name, _staircase(lines), premium_field="erned_prem")
    assert "erned_prem" in str(excinfo.value), (
        f"{name} must name the premium field it could not find, got: {excinfo.value!r}"
    )
