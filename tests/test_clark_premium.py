"""The premium requirement of the ``clark`` entry follows the METHOD.

``cape_cod`` is exposure-based (``U[w] = ELR * premium[w]``) and cannot fit
without premium. ``ldf`` anchors each origin on its own paid-to-date and never
reads exposure - so it must fit a premium-free triangle straight through the
public entry point, with no extra arguments.

Before this file, ``gallery.fit("clark", tri, method="ldf")`` raised
``no rows for premium field 'earned_premium'``: ``fit()`` built the data
contract with the signature's premium default BEFORE the branch that decides
whether premium matters, so a method with no premium in it failed on premium.
The workaround (``premium_field=None``) was undocumented.

Fast and extra-free on purpose: no chainladder, no cmdstan. The tieouts against
``ClarkLDF`` live in ``test_clark.py``, which skips wholesale without the
``[interop]`` extra - a premium contract that only core CI can check does not
belong there.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.triangle.core import Triangle

N_W = N_D = 7


def _cumulative(w: int, d: int) -> float:
    """Smooth growth-curve-ish emergence, per-origin scale varying with w so a
    mixed-up origin index cannot pass unnoticed."""
    return (1000.0 + 150.0 * w) * (1.0 - np.exp(-0.55 * d))


def _triangle(*, with_premium: bool) -> Triangle:
    """A square-staircase cumulative paid triangle, optionally carrying an
    ``earned_premium`` field alongside every loss cell."""
    rows = []
    for w in range(1, N_W + 1):
        for d in range(1, N_D + 1):
            if w + d - 1 > N_W:  # upper triangle only
                continue
            cell = {
                "lob": "CO_A",
                "origin_period": dt.date(2010 + w - 1, 1, 1),
                "dev_lag": 12 * d,
                "eval_date": dt.date(2010 + w - 1 + d - 1, 12, 31),
            }
            rows.append({**cell, "field": "paid_loss", "value": _cumulative(w, d)})
            if with_premium:
                # deliberately not constant: cape_cod scales by each origin's
                # own exposure, so equal premium would hide a wrong lookup
                rows.append({**cell, "field": "earned_premium", "value": 4000.0 + 500.0 * w})
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


def test_ldf_fits_a_premium_free_triangle_through_gallery_fit():
    """The defect, at the public entry point: no premium field, no premium
    argument, method that needs no premium - must fit.

    Verified red on the unfixed entry: ``ValueError: no rows for premium field
    'earned_premium'``.
    """
    tri = _triangle(with_premium=False)
    assert tri.fields == ["paid_loss"]  # the premise: nothing exposure-like here

    entry = gallery.fit("clark", tri, method="ldf")

    # the fit is real, not merely non-raising: a curve was found and the
    # per-origin ultimates exceed the paid-to-date they are anchored on
    prm = entry.params_
    assert prm["method"] == "ldf"
    assert prm["omega"] > 0 and prm["theta"] > 0
    assert np.isfinite(prm["level"]).all()
    assert (prm["level"] > 0).all()
    assert "premium" not in entry.contract_  # exposure was never requested

    ults = entry.predict(n_draws=500, seed=7).mean()
    assert ults.shape == (N_W + 1,)  # per origin + total
    assert np.isfinite(ults).all()
    np.testing.assert_array_less(entry.contract_["paid_to_date"] - 1e-9, ults[:-1])


def test_cape_cod_without_premium_names_premium_and_the_method():
    """The other half of the criterion: the method that DOES need premium still
    refuses, and its message says which method and what to do about it."""
    tri = _triangle(with_premium=False)
    with pytest.raises(ValueError) as excinfo:
        gallery.fit("clark", tri, method="cape_cod")
    msg = str(excinfo.value)
    assert "premium" in msg
    assert "cape_cod" in msg
    assert "ldf" in msg  # the escape hatch, named


def test_cape_cod_still_needs_premium_when_the_field_is_switched_off():
    """``premium_field=None`` under cape_cod is the same refusal - unchanged."""
    with pytest.raises(ValueError, match="cape_cod needs a premium_field"):
        gallery.fit("clark", _triangle(with_premium=True), premium_field=None, method="cape_cod")


def test_ldf_on_a_premium_triangle_matches_the_old_workaround():
    """Chosen semantics, pinned: ``ldf`` does not request premium EVEN WHEN the
    triangle has it, so the default path is identical to ``premium_field=None``
    (the pre-fix workaround) rather than merely close to it.

    Ignoring the column is what keeps a premium defect - a broken exposure row,
    another cohort's premium - out of a method that has no premium in it.
    """
    tri = _triangle(with_premium=True)
    assert "earned_premium" in tri.fields

    default = gallery.fit("clark", tri, method="ldf")
    workaround = gallery.fit("clark", tri, premium_field=None, method="ldf")

    for key in ("omega", "theta", "phi"):
        assert default.params_[key] == workaround.params_[key]
    np.testing.assert_array_equal(default.params_["level"], workaround.params_["level"])
    np.testing.assert_array_equal(default.params_["log_params"], workaround.params_["log_params"])
    # neither contract carries exposure, so neither can be biased by it
    assert "premium" not in default.contract_
    assert "premium" not in workaround.contract_
    # and the simulated ultimates agree draw for draw at a shared seed
    np.testing.assert_array_equal(
        default.predict(n_draws=200, seed=3).samples,
        workaround.predict(n_draws=200, seed=3).samples,
    )


def test_cape_cod_with_premium_is_unchanged():
    """The fix must not touch the method that legitimately consumes exposure:
    cape_cod still reads premium into the contract and scales the ELR by it."""
    tri = _triangle(with_premium=True)
    entry = gallery.fit("clark", tri, method="cape_cod")
    premium = entry.contract_["premium"]
    assert premium.shape == (N_W,)
    # level = ELR * premium, one profiled ELR shared by every origin
    elr = entry.params_["level"] / premium
    np.testing.assert_allclose(elr, elr[0], rtol=1e-12)
    assert 0.0 < elr[0] < 10.0
