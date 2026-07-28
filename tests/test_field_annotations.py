"""A field parameter that accepts ``None`` must SAY so in its annotation.

``py.typed`` ships from 0.5.0, so every ``fit()`` annotation is now a promise a
type checker enforces against user code. ``clark.fit`` broke that promise: it
declared ``premium_field: str`` while accepting, testing and DOCUMENTING
``premium_field=None`` (README, the entry's card, and tests/test_clark_premium.py,
which is where the workaround was pinned in #63). A type-checked caller writing
the documented call got an error on correct code.

Only ``clark`` had the mismatch - the other twelve entries carrying a
``premium_field`` genuinely refuse ``None``, each verified by running it:

    clark_growth_curve    ValueError, "needs a premium_field (Cape Cod ultimates)"
    guszcza_growth_curve  ValueError, "cannot fit without a premium_field"
    meyers_ccl/_csr, england_verrall_odp
                          'logprem' is in STAN_DATA_KEYS and stan_data/
                          odp_stan_data omit it when premium_field is None
    copula_glm            KeyError 'premium' - the lognormal marginal divides
                          every increment by its origin's exposure
    compartmental         TypeError inside compartmental_stan_data, whose
                          premium_field is required
    deeptriangle, mdn, nn_transformer, nn_transformer_ml, resnet
                          TypeError inside nn_data, which selects the premium
                          field unconditionally

``loss_field`` is ``str`` everywhere and correctly so: every entry passes it
straight to ``select_fields``, so ``None`` names no column. Same for
``compartmental``'s ``reported_field``.

Why a hand-written table rather than probing each entry: proving a refusal means
running ``fit()``, which for the Bayesian entries compiles Stan and for the NN
entries needs ``[nn]``. The table is the cheap gate, and
``test_every_entry_classifies_its_field_parameters`` is what gives it teeth - it
asserts the table IS the registry, so an entry cloned from a stale template
cannot join the gallery without its author classifying it here. That failure
mode is not hypothetical: see test_fit_atomicity.py's own registry gate, added
after three NN entries landed carrying a bug that had already been fixed.
"""

from __future__ import annotations

import datetime as dt
import inspect
import types
import typing

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.triangle.core import Triangle

#: the field-naming parameters of ``GalleryEntry.fit`` this file governs
SWEPT_PARAMS = ("loss_field", "premium_field", "reported_field")

#: entry name -> the swept parameters whose runtime ACCEPTS ``None``. An empty
#: set means every field parameter the entry takes is required, which is the
#: normal case; the entry must still appear, because the point of this table is
#: that a new entry cannot skip the question.
ACCEPTS_NONE: dict[str, frozenset[str]] = {
    # None switches exposure off. ``method="ldf"`` anchors each origin on its
    # own paid-to-date and fits straight through; ``cape_cod`` refuses by name.
    "clark": frozenset({"premium_field"}),
    "clark_growth_curve": frozenset(),
    "compartmental": frozenset(),
    "copula_glm": frozenset(),
    "deeptriangle": frozenset(),
    "england_verrall_odp": frozenset(),
    "guszcza_growth_curve": frozenset(),
    "mack": frozenset(),
    "mdn": frozenset(),
    "meyers_ccl": frozenset(),
    "meyers_csr": frozenset(),
    "nn_transformer": frozenset(),
    "nn_transformer_ml": frozenset(),
    "resnet": frozenset(),
    "sur": frozenset(),
}


def _admits_none(hint: object) -> bool:
    """Does this resolved annotation include ``None``?

    Resolved, not spelled: ``get_type_hints`` collapses ``str | None``,
    ``Optional[str]`` and ``Union[str, None]`` to the same object, so the check
    cannot be defeated by choosing a different spelling.
    """
    origin = typing.get_origin(hint)
    if origin not in (typing.Union, types.UnionType):
        return False
    return type(None) in typing.get_args(hint)


def test_every_entry_classifies_its_field_parameters():
    """ACCEPTS_NONE must BE the registry.

    This is the gate that survives a new entry. A model cloned from another's
    source inherits its annotations verbatim, so an entry that relaxes premium
    the way ``clark`` does - or one cloned FROM clark that does not - lands with
    an annotation nobody re-derived. Failing here names the entry at the one
    moment its author is looking at exactly this question.
    """
    registered = set(gallery.list())
    classified = set(ACCEPTS_NONE)

    missing = sorted(registered - classified)
    assert not missing, (
        f"gallery entries not classified in ACCEPTS_NONE: {missing}. For each swept "
        f"parameter {SWEPT_PARAMS} the entry's fit() takes, decide whether its RUNTIME "
        "accepts None (run it - do not read the annotation, which is what this file "
        "exists to check) and add the entry with the set of parameters that do."
    )
    stale = sorted(classified - registered)
    assert not stale, (
        f"these names are classified here but are not registered gallery entries: {stale}. "
        "A renamed or removed entry leaves a row that silently governs nothing."
    )


@pytest.mark.parametrize("name", sorted(ACCEPTS_NONE))
def test_field_annotations_match_runtime_acceptance(name):
    """``str | None`` exactly where None is accepted, ``str`` everywhere else.

    Both directions are defects. An annotation that hides None (the ``clark``
    bug) errors a type-checked caller on a documented call; one that offers None
    where the runtime refuses it invites a call that dies at the contract
    builder with a KeyError or a TypeError about NoneType.
    """
    accepts = ACCEPTS_NONE[name]
    fit = gallery.get(name).fit
    hints = typing.get_type_hints(fit)
    params = inspect.signature(fit).parameters

    present = [p for p in SWEPT_PARAMS if p in params]
    assert present, f"{name}.fit takes none of {SWEPT_PARAMS}"

    unknown = sorted(accepts - set(present))
    assert not unknown, f"ACCEPTS_NONE[{name!r}] lists {unknown}, which {name}.fit does not take"

    for param in present:
        hint = hints[param]
        if param in accepts:
            assert _admits_none(hint), (
                f"{name}.fit({param}=...) accepts None at runtime but is annotated "
                f"{hint!r}. Under py.typed a type checker rejects the documented call; "
                "annotate it `str | None`."
            )
            assert typing.get_args(hint) == (str, type(None)), (
                f"{name}.fit({param}=...) should be exactly `str | None`, got {hint!r}"
            )
        else:
            assert hint is str, (
                f"{name}.fit({param}=...) refuses None at runtime, so it must stay "
                f"annotated `str`, got {hint!r}. If the runtime changed, move the "
                "parameter into ACCEPTS_NONE."
            )


def test_clark_really_does_accept_a_none_premium_field():
    """The table's one positive row, anchored to behaviour.

    Without this the file only asserts that annotations agree with a table
    somebody typed. Cheap on purpose - the full premium contract (ldf vs
    cape_cod, the inert argument, bit-identity with the old workaround) is
    tests/test_clark_premium.py.
    """
    n = 6
    rows = [
        {
            "lob": "CO_A",
            "origin_period": dt.date(2010 + w - 1, 1, 1),
            "dev_lag": 12 * d,
            "eval_date": dt.date(2010 + w + d - 2, 12, 31),
            "field": "paid_loss",
            "value": (1000.0 + 150.0 * w) * (1.0 - np.exp(-0.55 * d)),
        }
        for w in range(1, n + 1)
        for d in range(1, n + 1)
        if w + d - 1 <= n
    ]
    tri = Triangle.from_long(pd.DataFrame(rows), measure="cumulative")

    entry = gallery.fit("clark", tri, premium_field=None, method="ldf")

    assert "premium" not in entry.contract_
    assert entry.params_["omega"] > 0 and entry.params_["theta"] > 0
