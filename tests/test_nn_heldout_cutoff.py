"""Where an NN entry thinks as_of is, when the cohort has a predecessor hole.
Skips without torch.

Every NN entry whose network reads a relative calendar position gets it from
one scalar, ``_heldout_inputs()["cutoff"]``, and the network turns it into
``dist = cal_idx - cutoff``. That scalar is the as_of boundary: the held-out
diagonal must land at distance 1, the position the cutoff augmentation
supervises most and the one ``_rollout`` steps through.

The bug this file pins: the cutoff was read off ``obs_mask``, which marks
usable INCREMENTS, not cells held. A cell whose predecessor is missing has no
usable increment and is absent from ``obs_mask`` even though the cohort
plainly had it - the fact ``_heldout.cohort_contract`` already unions
``latest_dev`` back in for when it declares training cells. So a cohort whose
latest diagonals are all hole-anchored had its as_of parked at the last
hole-free diagonal, and the held-out cells arrived several diagonals out
instead of one: a different ``dist_emb`` row, hence a different predictive.
Nothing about the number looked wrong - the entry still scored every cell, at
a plausible magnitude, through a network conditioned on the right context.

The fixture is the reproduction. ``lob_0`` is a 5-origin staircase with
calendar diagonal 5 punched out entirely, so at as_of (diagonal 6) EVERY
origin's latest cell is hole-anchored: origin 2002 holds cumulative devs
{1, 2, 4} exactly, the review's cohort. ``obs_mask`` then stops at diagonal 4
while the anchors all sit on diagonal 6, and the held-out diagonal 7 is 1 step
past the truth and 3 past the bug. Two healthy cohorts ride along because the
per-dev normalizer pools across cohorts and a lone punched cohort would pin
half the dev steps, leaving no live cell to score.

The entry list is derived from ``gallery.list()`` rather than written out, so
an entry cloned from one of these is covered the day it registers - the
stale-template bug class this package has already been bitten by twice.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ibnr import Triangle, gallery  # noqa: E402
from ibnr.gallery.nn._heldout import PooledMDNHeldout, heldout_cutoff  # noqa: E402
from ibnr.gallery.nn.deeptriangle.config import DeepTriangleConfig  # noqa: E402
from ibnr.gallery.nn.mdn.config import MDNConfig  # noqa: E402
from ibnr.gallery.nn.resnet.config import ResNetConfig  # noqa: E402
from ibnr.gallery.nn.transformer.config import TransformerConfig  # noqa: E402
from ibnr.kernels.holdout import index_into, next_diagonal  # noqa: E402

START = 2000
N = 6
AS_OF = "2005-12-31"  # calendar diagonal 6
HOLE_SEG = {"company_code": "0001", "line_of_business": "lob_0"}
#: the diagonal punched out of lob_0, which is what strands its anchors
PUNCHED_DIAGONAL = 5
#: lob_0's calendar geometry at AS_OF, all three pinned by the sanity test:
#: where its anchors sit (the truth), where obs_mask stops (the pre-fix answer),
#: and where the held-out cells land - 1 step past the truth, 3 past the bug.
AS_OF_DIAGONAL = 6
STALE_DIAGONAL = 4
HELDOUT_DIAGONAL = 7
#: dev step of the one held-out cell whose per-dev normalizer is pinned (its
#: only training-context values sit past the validation cutoff), matching the
#: other NN test fixtures
PINNED_DEVS = (6,)

#: under-powered on purpose: nothing here reads a trained prediction for
#: accuracy, only which cutoff the entry hands its network. ``heldout_n_draws``
#: is the held-out diagonal's budget (10,000 by default) and is cut to 50 for
#: the same reason as everything else here.
_TINY = dict(
    dropout=0.0,
    n_components=2,
    batch_size=8,
    max_epochs=3,
    patience=5,
    ensemble_size=2,
    heldout_n_draws=50,
)

#: name -> (entry class, fit config). Keyed by gallery name so the coverage
#: test below can compare it against the registry directly.
CONFIGS = {
    "nn_transformer": TransformerConfig(
        d_model=16, n_layers=1, n_heads=2, ffn_dim=32, lob_embedding_dim=4, n_draws=50, **_TINY
    ),
    "mdn": MDNConfig(
        hidden_dim=32, n_layers=1, embedding_dim=4, lob_embedding_dim=4, n_draws=50, **_TINY
    ),
    "resnet": ResNetConfig(
        channels=16, n_blocks=2, n_groups=4, lob_embedding_dim=4, n_draws=50, **_TINY
    ),
    "deeptriangle": DeepTriangleConfig(
        hidden_dim=16, lob_embedding_dim=4, company_embedding_dim=4, n_draws=50, **_TINY
    ),
}


def heldout_entry_names() -> list[str]:
    """Registered entries that assemble their own held-out forward inputs.

    Derived from ``gallery.list()``, not written out: the defect lives in
    ``_heldout_inputs``, so membership is "has that hook", and a new entry
    inheriting the mixin joins this test automatically instead of shipping the
    bug again under a new name.
    """
    return sorted(
        name
        for name, cls in ((n, gallery.get(n)) for n in gallery.list())
        if issubclass(cls, PooledMDNHeldout)
    )


ENTRY_NAMES = heldout_entry_names()


# -- fixture -------------------------------------------------------------------


def _square(seed: int) -> np.ndarray:
    """(N, N) cumulative square with a decaying incremental pattern."""
    rng = np.random.default_rng(seed)
    dev_level = np.exp(np.linspace(-0.8, -3.0, N))
    incr = 1000.0 * dev_level[None, :] * rng.lognormal(0.0, 0.1, size=(N, N))
    return np.cumsum(incr, axis=1)


def _punched() -> np.ndarray:
    """``lob_0``: 5 origins, calendar diagonal ``PUNCHED_DIAGONAL`` removed.

    Removing a whole diagonal rather than one cell is what makes the bug
    reproduce at all: the buggy cutoff is a max over every origin, so a single
    hole would be masked by any other origin still reporting on the latest
    diagonal. Dropping the 6th origin keeps the cohort from re-supplying a
    first-dev cell (always a usable increment) on that diagonal.
    """
    cum = _square(0)
    cum[N - 1, :] = np.nan
    for w in range(N - 1):
        d = PUNCHED_DIAGONAL - 1 - w  # cal_idx = w + d + 1
        if 0 <= d < N:
            cum[w, d] = np.nan
    return cum


def _rows(lob: str, cum: np.ndarray, premium: float = 1000.0) -> list[tuple]:
    """Long rows for one cohort; NaN is unobserved and simply not emitted."""
    out = []
    for w in range(cum.shape[0]):
        for d in range(cum.shape[1]):
            if np.isnan(cum[w, d]):
                continue
            key = (
                "0001",
                lob,
                dt.date(START + w, 1, 1),
                12 * (d + 1),
                dt.date(START + w + d, 12, 31),
            )
            out.append((*key, "paid_loss", float(cum[w, d])))
            out.append((*key, "earned_premium", premium))
    return out


_COLUMNS = [
    "company_code",
    "line_of_business",
    "origin_period",
    "dev_lag",
    "eval_date",
    "field",
    "value",
]


def _triangle(rows: list[tuple]) -> Triangle:
    return Triangle.from_long(pd.DataFrame(rows, columns=_COLUMNS), measure="cumulative")


@pytest.fixture(scope="module")
def fitted():
    """One fit per NN entry on the shared pooled triangle, plus lob_0's cells.

    Module-scoped and single-backend: the defect is pure numpy over the fitted
    contract, identical on duckdb and polars, so paying for a second backend
    would buy nothing and four more ensemble trainings.
    """
    hole = _punched()
    rows = _rows("lob_0", hole)
    for k in (1, 2):
        rows += _rows(f"lob_{k}", _square(k))
    pooled = _triangle(rows)

    entries = {
        name: gallery.get(name)().fit(
            pooled, loss_field="paid_loss", as_of=AS_OF, config=CONFIGS[name], seed=0
        )
        for name in ENTRY_NAMES
    }
    cells = next_diagonal(
        _triangle(_rows("lob_0", hole)),
        as_of=AS_OF,
        fields="paid_loss",
        premium_field="earned_premium",
    )
    return SimpleNamespace(entries=entries, cells=cells, hole=hole)


def _live(cells):
    """The held-out cells at unpinned dev steps - the ELPD-scorable ones."""
    mask = cells.frame["dev_lag"].isin([12 * d for d in PINNED_DEVS])
    return replace(cells, frame=cells.frame[~mask].reset_index(drop=True))


# -- fixture sanity ------------------------------------------------------------


def test_every_registered_nn_entry_is_covered():
    """The parametrization IS the registry's held-out NN entries.

    Guards the stale-template bug class from the other side: a hand-written
    list silently stops covering the entry added after it was written, which is
    exactly how this defect reached three entries in the first place.
    """
    assert ENTRY_NAMES, "no entry inherits PooledMDNHeldout; this file tests nothing"
    assert set(CONFIGS) == set(ENTRY_NAMES), (
        f"CONFIGS and the registry disagree: {set(CONFIGS) ^ set(ENTRY_NAMES)}. A new NN "
        "held-out entry needs a config row here, not its own copy of this file"
    )


def test_the_fixture_actually_strands_the_anchors(fitted):
    """Pin the reproduction itself, so fixture rot cannot make the real test
    below pass vacuously (obs cutoff == anchor cutoff would satisfy it).

    Three claims: origin 2002 holds cumulative devs {1, 2, 4} - the review's
    cohort, hole at dev 3; every origin's anchor sits on diagonal 6 while
    ``obs_mask`` stops at diagonal 4; and the held-out diagonal 7 is therefore
    1 step past the truth and 3 past the bug.
    """
    entry = fitted.entries[ENTRY_NAMES[0]]
    c = entry.contract_
    ci = entry.cohort_index(HOLE_SEG)
    obs = c["obs_mask"][ci]

    # origin 2002 (0-based origin index 2) in the training slice: the cells it
    # holds at or before AS_OF_DIAGONAL are cumulative devs {1, 2, 4}
    w0 = 2
    trained_devs = [
        d + 1 for d in range(N) if not np.isnan(fitted.hole[w0, d]) and w0 + d + 1 <= AS_OF_DIAGONAL
    ]
    assert trained_devs == [1, 2, 4]

    np.testing.assert_array_equal(c["latest_dev"][ci], [6, 5, 4, 3, 2, 0])
    anchored = np.nonzero(c["latest_dev"][ci] > 0)[0]
    anchor_cal = c["cal_idx"][anchored, c["latest_dev"][ci][anchored] - 1]
    assert set(anchor_cal.tolist()) == {AS_OF_DIAGONAL}, "anchors are not all on the as_of diagonal"
    assert int(c["cal_idx"][obs].max()) == STALE_DIAGONAL, "obs_mask no longer stops short of as_of"

    assert fitted.cells.n_cells == 4
    assert fitted.cells.exclusion_counts() == {
        "new_origin": 0,
        "dev_beyond_trained": 0,
        "no_predecessor": 0,
    }
    # every held-out cell is on diagonal 7: distance 1 from 6, distance 3 from 4.
    # w and d are 1-based here (index_into's space), so cal_idx is w + d - 1.
    w = np.array([o.year - START + 1 for o in fitted.cells.frame["origin_period"]])
    d = fitted.cells.frame["dev_lag"].to_numpy() // 12
    np.testing.assert_array_equal(w + d - 1, np.full(4, HELDOUT_DIAGONAL))
    assert _live(fitted.cells).n_cells == 3, "no unpinned cell left to score a density at"


# -- the fix -------------------------------------------------------------------


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_heldout_cutoff_is_the_as_of_diagonal_not_the_last_usable_increment(name, fitted):
    """The held-out diagonal sits at distance 1 from the entry's cutoff.

    This is the docstring's own claim - "so the held-out diagonal sits at
    distance 1" - asserted rather than asserted-by-comment. On this cohort the
    obs-derived answer is 4 and the truth is 6, so the pre-fix code puts the
    held-out cells 3 diagonals out.

    An entry carrying no cutoff at all (deeptriangle: relative position is
    structural in the recurrence) has nothing to misplace, but it must be shown
    to carry none under ANY spelling - otherwise a future entry could
    reintroduce the boundary under a new key and pass here by omission.
    """
    entry = fitted.entries[name]
    ci = entry.cohort_index(HOLE_SEG)
    inputs = entry._heldout_inputs(ci)

    if "cutoff" not in inputs:
        stray = [k for k in inputs if "cut" in k.lower() or "cal" in k.lower()]
        assert not stray, (
            f"{name} carries calendar input(s) {stray} under a key this test does not "
            "check; derive them from _heldout.heldout_cutoff and add the key here"
        )
        return

    got = int(inputs["cutoff"].item())
    assert HELDOUT_DIAGONAL - got == 1, (
        f"{name} conditions at calendar diagonal {got}, putting the held-out diagonal "
        f"{HELDOUT_DIAGONAL - got} steps out instead of 1 - the reading that takes the "
        f"cutoff off obs_mask (deepest USABLE INCREMENT, {STALE_DIAGONAL}) rather than "
        f"off the cells the cohort held at as_of ({AS_OF_DIAGONAL})"
    )
    # and it is the shared derivation, not a coincidence of this fixture
    assert got == AS_OF_DIAGONAL == int(heldout_cutoff(entry.contract_, ci))


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_the_cutoff_is_not_inert(name, fitted):
    """Feeding the old obs-derived cutoff changes the density it produces.

    Without this the test above proves only that a scalar in a dict has a
    value. Substituting the pre-fix cutoff through the same public call path
    must move ``log_lik_at`` - if it does not, the entry's network is ignoring
    the boundary and the assertion above is grading a number nothing reads.
    Entries with no cutoff are exempt: they genuinely have no such wire.
    """
    entry = fitted.entries[name]
    ci = entry.cohort_index(HOLE_SEG)
    if "cutoff" not in entry._heldout_inputs(ci):
        return

    cells = _live(fitted.cells)
    real = entry._heldout_inputs
    stale = int(entry.contract_["cal_idx"][entry.contract_["obs_mask"][ci]].max())
    assert int(real(ci)["cutoff"].item()) != stale, (
        f"{name} already conditions at the obs_mask boundary {stale}, so there is no "
        "substitution to make and this test cannot say whether the cutoff is live"
    )
    good = entry.log_lik_at(cells, field="paid_loss")

    def with_stale_cutoff(cohort: int) -> dict:
        inputs = real(cohort)
        inputs["cutoff"] = torch.full_like(inputs["cutoff"], stale)
        return inputs

    entry._heldout_inputs = with_stale_cutoff
    try:
        bad = entry.log_lik_at(cells, field="paid_loss")
    finally:
        del entry._heldout_inputs  # drop the instance attribute, restore the bound method
    assert not np.allclose(good, bad), (
        f"{name}'s density is unchanged by moving the conditioning boundary from {stale} "
        f"to {heldout_cutoff(entry.contract_, ci)}; the cutoff is not reaching the network"
    )


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_heldout_cells_still_score_on_the_hole_cohort(name, fitted):
    """The hole cohort remains scorable on both axes after the fix.

    A cutoff moved deeper indexes a different embedding row, and clamping or a
    grid bound could turn that into a crash or a NaN rather than a number. Both
    capabilities are exercised on the cells the fix changes, and the draws are
    checked against their anchors so a broken conditioning cannot hide behind
    finite-but-absurd output.
    """
    entry = fitted.entries[name]
    cells = fitted.cells
    live = _live(cells)

    ll = entry.log_lik_at(live, field="paid_loss")
    assert ll.shape == (CONFIGS[name].ensemble_size, live.n_cells)
    assert np.isfinite(ll).all()

    draws = entry.predict_at(cells, field="paid_loss", seed=0)
    assert draws.shape == (CONFIGS[name].heldout_n_draws, cells.n_cells)
    assert np.isfinite(draws).all()
    # cumulative basis: the base class anchored the entry's increments onto each
    # cell's training predecessor, so the draws sit in the anchors' neighborhood
    idx = index_into(cells, entry.at_cohort(HOLE_SEG).contract_, field="paid_loss")
    assert (np.abs(draws.mean(axis=0) - idx.prev_value) < 10.0 * idx.prev_value).all()
