"""A pooled NN fit scored on cells the mart's THREE segments carry. Needs torch.

This is the gap the ``analysis/03`` notebook hit and could only work around. The
Schedule P mart carries ``(company_code, company_name, line_of_business)``, but
``kernels.nn_contract`` keeps ``company_name`` out of the cohort key so that two
spellings of one company do not become two cohorts. A pooled fit was therefore
keyed on two columns while ``next_diagonal`` built cells on three, and BOTH
routes into held-out scoring failed:

* ``entry.log_lik_at(cells)`` raised ``KeyError: unknown segment column
  'company_name'`` - naming a column the caller had just passed, and saying
  nothing about what the fit was keyed on;
* ``entry.at_cohort({...}).log_lik_at(cells)`` got one level deeper and failed
  inside ``index_into`` on a 3-vs-2 schema mismatch.

The only route that worked was dropping the display column from the triangle
before BOTH ``fit()`` and ``next_diagonal()``, which is what the notebook does.

The fix narrows the cells onto the fit's own key inside ``log_lik_at`` /
``predict_at``, and the two facts that make that legitimate rather than merely
convenient are both tested here: the dropped value is **verified** against the
fitted cohort (never discarded), and the narrowed answer is elementwise
IDENTICAL to the pre-dropped route.

The fixture mirrors ``test_nn_heldout.py``: a 6x6 full square per LOB with
``as_of`` at diagonal 6. Its first origin's next-diagonal cell sits at a PINNED
dev, which this entry documents as CRPS-only, so the origins are restricted -
that is an unrelated, pre-existing asymmetry and not what these tests are about.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr import gallery  # noqa: E402
from ibnr.gallery.nn.mdn.config import MDNConfig  # noqa: E402
from ibnr.gallery.nn.mdn.model import MDN  # noqa: E402
from ibnr.kernels.forecast import CohortForecast  # noqa: E402
from ibnr.kernels.holdout import next_diagonal  # noqa: E402

from .conftest import make_multiline_triangle  # noqa: E402

START = 2000
N = 6
AS_OF = "2005-12-31"  # diagonal 6 of a 6x6 square
FIELD = "paid_loss"
COMPANY_NAME = "Acme Insurance Co"
#: skip the first origin - its next-diagonal cell is at a pinned dev (CRPS-only)
ORIGINS = [dt.date(START + w, 1, 1) for w in range(1, N)]
FULL_KEY = {
    "company_code": "0001",
    "company_name": COMPANY_NAME,
    "line_of_business": "lob_0",
}
NARROW_KEY = {"company_code": "0001", "line_of_business": "lob_0"}

TINY = MDNConfig(
    hidden_dim=16,
    n_layers=1,
    dropout=0.0,
    n_components=2,
    lob_embedding_dim=4,
    batch_size=8,
    max_epochs=3,
    patience=5,
    ensemble_size=2,
    n_draws=50,  # rollout draws (predict)
    heldout_n_draws=50,  # held-out diagonal draws (predict_at); 10,000 by default
)


def _grids():
    """Smooth, strictly increasing cumulatives - the model is not under test."""
    rng = np.random.default_rng(0)
    out = {}
    for i, lob in enumerate(("lob_0", "lob_1")):
        base = 100.0 * (1.0 + 0.4 * i)
        cum = np.empty((N, N))
        for w in range(N):
            total = 0.0
            for dev in range(N):
                total += base * (0.9**dev) * (1.0 + 0.02 * rng.standard_normal())
                cum[w, dev] = total
        out[lob] = cum
    return out


PREMIUM = {lob: np.full(N, 1000.0) for lob in ("lob_0", "lob_1")}


@pytest.fixture(scope="module")
def wide_triangle():
    """The mart's shape: three segments, one of them display-only."""
    return make_multiline_triangle(
        "duckdb", _grids(), start_year=START, premium_by_lob=PREMIUM, company_name=COMPANY_NAME
    )


@pytest.fixture(scope="module")
def narrow_triangle():
    """The notebook's workaround: the same data with the display column dropped."""
    return make_multiline_triangle("duckdb", _grids(), start_year=START, premium_by_lob=PREMIUM)


def _cells(triangle):
    one = triangle.filter(
        (triangle.expr.company_code == "0001") & (triangle.expr.line_of_business == "lob_0")
    )
    return next_diagonal(
        one,
        as_of=AS_OF,
        fields=FIELD,
        premium_field="earned_premium",
        origins=ORIGINS,
    )


@pytest.fixture(scope="module")
def wide(wide_triangle):
    entry = MDN().fit(wide_triangle, loss_field=FIELD, as_of=AS_OF, config=TINY, seed=0)
    return entry, _cells(wide_triangle)


@pytest.fixture(scope="module")
def narrow(narrow_triangle):
    entry = MDN().fit(narrow_triangle, loss_field=FIELD, as_of=AS_OF, config=TINY, seed=0)
    return entry, _cells(narrow_triangle)


def test_the_fit_is_keyed_more_narrowly_than_the_cells(wide):
    """The premise: this is a real 3-vs-2 mismatch, not a contrived one."""
    entry, cells = wide
    assert cells.segments == ("company_code", "company_name", "line_of_business")
    assert entry.contract_["segment_columns"] == ("company_code", "line_of_business")


def test_cohorts_carries_the_display_column(wide):
    """``cohorts()`` hands back the identity the TRIANGLE carried, not the key.

    A caller must not be told a cohort is unidentifiable when the fit knows
    exactly which one it is.
    """
    entry, _ = wide
    identities = entry.cohorts()
    assert len(identities) == 2
    assert identities[0] == FULL_KEY
    assert all(c["company_name"] == COMPANY_NAME for c in identities)


def test_log_lik_at_accepts_the_full_key(wide):
    """The notebook's first failure, now working: raised KeyError before."""
    entry, cells = wide
    out = entry.log_lik_at(cells, field=FIELD)
    assert out.shape == (len(entry.models_), cells.n_cells)
    assert np.isfinite(out).all()


def test_at_cohort_then_log_lik_at_accepts_the_full_key(wide):
    """The notebook's second failure: it died one level deeper, in ``index_into``."""
    entry, cells = wide
    out = entry.at_cohort(FULL_KEY).log_lik_at(cells, field=FIELD)
    assert out.shape == (len(entry.models_), cells.n_cells)


def test_at_cohort_accepts_a_subset_of_the_identity(wide):
    """A segment is a FILTER on the fit's cohorts, so a subset that names one works.

    This is what lets one loop pass the same dict to a fit keyed on all three
    columns and to a pooled fit keyed on two.
    """
    entry, cells = wide
    a = entry.at_cohort(FULL_KEY).log_lik_at(cells, field=FIELD)
    b = entry.at_cohort(NARROW_KEY).log_lik_at(cells, field=FIELD)
    np.testing.assert_array_equal(a, b)


def test_predict_at_accepts_the_full_key(wide):
    entry, cells = wide
    draws = entry.predict_at(cells, field=FIELD, seed=1)
    assert draws.shape == (TINY.heldout_n_draws, cells.n_cells)
    assert np.isfinite(draws).all()


def test_narrowing_changed_the_key_and_nothing_else(wide, narrow):
    """The numerical identity check: same densities as the pre-dropped route.

    Both fits see identical numbers - the display column is not a feature - so
    any difference here would mean the narrowing moved a cell, not just a column.
    """
    wide_entry, wide_cells = wide
    narrow_entry, narrow_cells = narrow
    assert narrow_cells.segments == ("company_code", "line_of_business")

    np.testing.assert_allclose(
        wide_entry.log_lik_at(wide_cells, field=FIELD),
        narrow_entry.log_lik_at(narrow_cells, field=FIELD),
    )
    np.testing.assert_allclose(
        wide_entry.predict_at(wide_cells, field=FIELD, seed=1),
        narrow_entry.predict_at(narrow_cells, field=FIELD, seed=1),
    )


def test_the_dropped_value_is_verified_not_discarded(wide):
    """A cells' display value that disagrees with the fitted cohort must RAISE.

    An implementation that simply dropped the extra column passes every test
    above and fails only this one - which is why it is here. The message names
    both schemas AND both values, because "these cells are not this cohort's" is
    the only thing the caller can act on.

    ``excluded`` is re-spelled alongside ``frame``, and that is what makes this
    test discriminate rather than merely pass. Mutating ``frame`` alone leaves
    the one excluded cell on the original spelling, so the two spellings become
    two distinct cohorts and ``narrowed_to``'s COLLAPSE guard raises first - with
    a message that happens to quote both values and the column name, satisfying
    every assertion below while the value check is never reached. Measured: with
    ``_keyed_to_fit``'s value check disabled the file still passed 14/14. Both
    frames re-spelled, there is exactly one cohort to narrow onto, so the value
    check is the only thing left that can refuse.
    """
    entry, cells = wide
    import dataclasses

    wrong = dataclasses.replace(
        cells,
        frame=cells.frame.assign(company_name="ACME"),
        excluded=cells.excluded.assign(company_name="ACME"),
    )
    with pytest.raises(ValueError) as excinfo:
        entry.at_cohort(FULL_KEY).log_lik_at(wrong, field=FIELD)
    message = str(excinfo.value)
    assert "ACME" in message
    assert COMPANY_NAME in message
    assert "company_name" in message
    # the refusal is the VALUE check, not the collapse guard standing in for it
    assert "agree with the fitted cohort" in message
    assert "collapse" not in message


def test_the_narrowing_never_escapes_to_the_panel(wide):
    """``CohortForecast`` still takes the ORIGINAL wide cells.

    The narrowing is internal to ``log_lik_at``/``predict_at``, so every model on
    a shared board keeps one segment schema and ``align_panel``'s mixed-schema
    refusal is untouched.
    """
    entry, cells = wide
    forecast = CohortForecast(
        model="mdn",
        task="paid@2005",
        cells=cells,
        field=FIELD,
        log_density=entry.log_lik_at(cells, field=FIELD),
        draws=entry.predict_at(cells, field=FIELD, seed=1),
    )
    assert forecast.cells.segments == ("company_code", "company_name", "line_of_business")


# -- the pooled `segment=` contract ---------------------------------------------


def test_predict_per_cohort_covers_the_panel(wide):
    """``for seg in entry.cohorts(): entry.predict(segment=seg)`` is the loop.

    Per cohort: that cohort's per-origin targets plus its total. Without a
    segment: the whole panel and no grand total.
    """
    entry, _ = wide
    panel = entry.predict(seed=0)
    per_cohort = [entry.predict(segment=seg, seed=0) for seg in entry.cohorts()]
    assert len(panel.targets) == sum(len(p.targets) - 1 for p in per_cohort)
    assert "total" in [str(x) for x in per_cohort[0].targets["label"]]


def test_a_subset_segment_resolves_on_a_pooled_fit(wide):
    """The mixed-loop case: two columns naming one cohort of a three-column key."""
    entry, _ = wide
    a = entry.predict(segment=FULL_KEY, seed=0)
    b = entry.predict(segment=NARROW_KEY, seed=0)
    np.testing.assert_allclose(a.samples, b.samples)


def test_an_ambiguous_subset_is_refused(wide):
    """Both cohorts share ``company_code``, so that alone names neither."""
    entry, _ = wide
    with pytest.raises(ValueError) as excinfo:
        entry.predict(segment={"company_code": "0001"})
    assert "matches 2 cohorts, need exactly 1" in str(excinfo.value)


def test_realized_ultimates_takes_the_same_segment(wide, wide_triangle):
    entry, _ = wide
    per_cohort = entry.realized_ultimates(wide_triangle, segment=FULL_KEY)
    subset = entry.realized_ultimates(wide_triangle, segment=NARROW_KEY)
    np.testing.assert_allclose(per_cohort, subset, equal_nan=True)
    # per origin plus the total
    assert len(per_cohort) == N + 1


def test_config_class_is_reachable_from_the_registry():
    """Gap 4: a caller who found the entry by name can build its config.

    Before this, ``MDNConfig`` was public but reachable only by module path -
    notebook 03 imports four of them that way.
    """
    cls = gallery.get("mdn")
    assert cls.config_class is MDNConfig
    assert cls.config_class(ensemble_size=1).ensemble_size == 1
