"""fit()'s premium contract, per method: ``ldf`` never consults premium, so a
losses-only triangle fits with the ``premium_field`` default in place, while
``cape_cod`` keeps its hard error. Regression tests for the bug where ``ldf``
resolved ``premium_field="earned_premium"`` while building the data contract -
ahead of the branch that would have ignored it - so a triangle carrying no
premium raised ``ValueError: no rows for premium field 'earned_premium'``
unless the caller knew to pass ``premium_field=None``.

All calls go through ``gallery.fit`` (the public entry point), not ``Clark()``
directly: the defect lives in how the default argument travels, and a test
that bypasses the entry point can pass while the entry point still fails.

Deliberately chainladder-free, unlike ``test_clark.py``: the tieout suite
skips wherever the ``[interop]`` extra is absent, and these tests must run on
the core-only CI leg (the statistical clark entry needs nothing beyond scipy).
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle, gallery

N_W = N_D = 6


def _triangle(*, with_premium: bool) -> Triangle:
    """Full 6x6 cumulative paid triangle, optionally carrying earned_premium."""
    rows = []
    for w in range(1, N_W + 1):
        premium = 1000.0 + 200.0 * w
        for d in range(1, N_D + 1):
            cell = {
                "lob": "PREMIUM_CONTRACT_CO",
                "origin_period": dt.date(2010 + w - 1, 1, 1),
                "dev_lag": 12 * d,
                "eval_date": dt.date(2010 + w - 1 + d - 1, 12, 31),
            }
            rows.append(
                {**cell, "field": "paid_loss", "value": premium * 0.65 * (1.0 - np.exp(-0.6 * d))}
            )
            if with_premium:
                rows.append({**cell, "field": "earned_premium", "value": premium})
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


def _losses_only() -> Triangle:
    return _triangle(with_premium=False)


def test_ldf_fits_a_losses_only_triangle_with_default_arguments():
    """No ``premium_field`` argument at all - the default stays in place.

    FAILED before the fix: ``ValueError: no rows for premium field
    'earned_premium'``, raised while building the data contract, before the
    method branch that never reads premium."""
    entry = gallery.fit("clark", _losses_only(), method="ldf")
    assert entry.params_["method"] == "ldf"
    # ldf resolves no premium, so none may be attached for downstream code
    # (predict targets, the scorer) to accidentally depend on.
    assert "premium" not in entry.contract_


def test_ldf_ignores_an_explicit_premium_field():
    """Premium is inert under ldf even when named explicitly: the fit succeeds
    on a triangle without the field, and the MLE is identical to the
    ``premium_field=None`` fit (same data, same deterministic optimizer)."""
    named = gallery.fit("clark", _losses_only(), method="ldf", premium_field="earned_premium")
    disabled = gallery.fit("clark", _losses_only(), method="ldf", premium_field=None)
    assert named.params_["omega"] == disabled.params_["omega"]
    assert named.params_["theta"] == disabled.params_["theta"]
    np.testing.assert_array_equal(named.params_["level"], disabled.params_["level"])


def test_ldf_does_not_resolve_premium_even_when_the_field_exists():
    """ldf must not attach premium just because the triangle carries the field:
    on a losses-only fixture, "never resolves" and "resolves when present" are
    indistinguishable, so this is the test that tells them apart.

    Mutation (verified by review): resolving premium whenever the field exists
    passes every other test in this file and the rest of the suite, while
    attaching `premium`/`logprem` to ldf contracts and populating predict()'s
    targets premium - the exact behavior the fix declares ignored."""
    entry = gallery.fit("clark", _triangle(with_premium=True), method="ldf")
    assert "premium" not in entry.contract_
    assert "logprem" not in entry.contract_
    pred = entry.predict(n_draws=20, seed=0)
    assert pred.targets["premium"].isna().all()


def test_cape_cod_still_resolves_premium_and_fails_loudly_without_it():
    """The fix must not loosen cape_cod: its default ``premium_field`` is still
    resolved against the triangle, and a missing field is still a fit-time
    error rather than a silent fallback."""
    with pytest.raises(ValueError, match="no rows for premium field"):
        gallery.fit("clark", _losses_only(), method="cape_cod")


def test_cape_cod_with_premium_field_none_keeps_its_hard_error():
    with pytest.raises(ValueError, match="cape_cod needs a premium_field"):
        gallery.fit("clark", _losses_only(), method="cape_cod", premium_field=None)
