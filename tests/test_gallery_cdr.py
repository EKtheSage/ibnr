"""``gallery.GalleryDiagonal``: a one-year CDR off any entry that can draw the
next diagonal.

The feature is wiring, so the tests are about wiring, and they come in two
layers.

1. **The equivalence test, which is the whole thing.** A stub entry whose
   ``_draws_native`` returns Mack's OWN conditional draws must produce a CDR
   *bit-identical* to ``generator="mack"``. Nothing else can pin this path:
   every intermediate array is plausible, the CDR of a wrong-but-aligned
   diagonal is finite and smooth, and a column permutation across accident years
   produces numbers of the right magnitude and sign. Byte equality against a
   route that is itself tied out to R is the only assertion that cannot be
   passed by accident.

2. **The refusals**, one per way the fit and the cells can disagree. All of them
   share the property that makes them worth testing individually: the wrong
   answer they prevent is a complete, finite, plausible page of dollars.
   ``rereserve`` cannot tell that the diagonal it was handed belongs to another
   cohort, and a missing origin re-reserves to a zero ultimate whose CDR is that
   origin's entire estimated ultimate - the largest wrong number available.

Mutation-verified: twelve deliberate breakages, all caught - dropping the
``prev_value`` comparison, dropping the ``dev_lag`` comparison, reading a
missing open origin as a closed one instead of raising, taking the cells' rows
in frame order rather than in the fit's origin order, letting an explicit
``n_draws`` sit inert, scoring every field's rows rather than the resolved one,
skipping the field-versus-fit agreement check, seeding the entry from a constant
rather than from the caller's ``rng``, skipping the training-history check,
accepting a training-value disagreement, waving through a contract with no
comparable values, and letting ``Mack.cdr_distribution`` hardcode a draw budget.

**Two of these came from review rather than from this file**, and both are worth
the note because they are the same shape - a check whose absence is invisible in
the output:

- *the seed*. The two stubs that make byte equality possible both ignore the
  injected ``rng`` on purpose, so neither could see that ``seed=`` had stopped
  arriving. Every draw is legitimate either way. ``_NoisyEntry`` exists for it.
- *the training history*. The other alignment checks tie the CELLS to the fit;
  nothing tied the ENTRY to either, so an entry refitted on restated interior
  cells - latest diagonal untouched - passed everything and moved the total CDR
  mean from 0.27 to -367.69. Measured, on this fixture, before the fix.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ibnr.gallery.cdr import GalleryDiagonal
from ibnr.gallery.entry import PredictsHeldout
from ibnr.gallery.registry import get
from ibnr.kernels.cdr import (
    CDR_METHODS,
    DEFAULT_N_DRAWS,
    MackDiagonal,
    cdr_methods,
    get_cdr_method,
    simulate_one_year_cdr,
)
from ibnr.kernels.contract import cohort_grid, stan_data
from ibnr.kernels.holdout import next_diagonal
from ibnr.kernels.mack import MackFit, _next_step_draws, fit_mack
from ibnr.triangle.core import Triangle

from .conftest import make_cohort_triangle

# A run-off staircase with one diagonal beyond the cutoff, so next_diagonal has
# something to hold out. Values are deliberately irregular: on a smooth triangle
# an off-by-one origin lands on a plausible neighbour.
FULL = np.array(
    [
        [1010.0, 1620.0, 1930.0, 2065.0, 2110.0, 2125.0],
        [1180.0, 1815.0, 2190.0, 2360.0, 2420.0, 2441.0],
        [ 940.0, 1560.0, 1885.0, 2035.0, 2088.0, np.nan],
        [1305.0, 2010.0, 2405.0, 2590.0, np.nan, np.nan],
        [1120.0, 1755.0, 2115.0, np.nan, np.nan, np.nan],
        [1240.0, 1930.0, np.nan, np.nan, np.nan, np.nan],
    ]
)  # fmt: skip
START_YEAR = 2010
#: the training cutoff - the calendar diagonal that leaves FULL one short
AS_OF = dt.date(START_YEAR + 4, 12, 31)


def _triangle(*, extra_field: bool = False) -> Triangle:
    """FULL as a cumulative Triangle, optionally with a second loss field."""
    tri = make_cohort_triangle(None, FULL, start_year=START_YEAR)
    if not extra_field:
        return tri
    df = tri.execute()
    other = df.copy()
    other["field"] = "reported_loss"
    other["value"] = other["value"] * 1.25
    return Triangle.from_long(pd.concat([df, other], ignore_index=True), measure="cumulative")


def _fit(**kwargs) -> MackFit:
    return fit_mack(_triangle(), loss_field="paid_loss", as_of=AS_OF, **kwargs)


def _cells(**kwargs):
    return next_diagonal(_triangle(**kwargs), as_of=AS_OF, fields="paid_loss")


def _contract() -> dict:
    return stan_data(_triangle().as_of(AS_OF), loss_field="paid_loss")


class _MackEchoEntry(PredictsHeldout):
    """A "fitted entry" whose posterior predictive IS Mack's conditional law.

    Not a model - a control. It draws next year's diagonal from exactly the
    moments :class:`MackDiagonal` uses, off a generator seeded the same way, so
    any difference between its CDR and the ``mack`` route's is this module's
    doing and nothing else. ``contract_`` is the single-cohort identity that
    ``_keyed_to_fit`` / ``index_into`` check the cells against.
    """

    heldout_draw_scale = "cumulative"

    def __init__(self, fit: MackFit, *, n_draws: int, seed: int, contract: dict | None = None):
        self.fit = fit
        # the contract must describe the SAME triangle as the fit, or the
        # training-history check refuses it - which is the check working. The
        # positivity test below passes a zeroed fit and must pass its contract
        # with it.
        self.contract_ = _contract() if contract is None else contract
        self._n_draws = n_draws
        self._seed = seed

    def _draws_native(self, cells, *, rng):
        # indexed off the CELLS' own 1-based (w, d); the step INTO dev index
        # d - 1 is carried by f[d - 2]
        i = np.asarray(cells.w, dtype=int) - 1
        step0 = np.asarray(cells.d, dtype=int) - 2
        return _next_step_draws(
            self.fit,
            step0,
            self.fit.cum[i, step0],
            n_draws=self._n_draws,
            rng=np.random.default_rng(self._seed),
            process="gamma",
            parameter_risk=True,
        )


class _LabelledEntry(PredictsHeldout):
    """Deterministic draws that say which origin they came from.

    For the ordering test only. Every column is a function of that cell's own
    ``w``, and of nothing positional, so permuting the cells' rows must leave
    the CDR untouched - which a positional shortcut in the alignment would not.
    """

    heldout_draw_scale = "cumulative"

    def __init__(self, fit: MackFit, *, n_draws: int = 16):
        self.fit = fit
        self.contract_ = _contract()
        self._n_draws = n_draws

    def _draws_native(self, cells, *, rng):
        i = np.asarray(cells.w, dtype=int) - 1
        step0 = np.asarray(cells.d, dtype=int) - 2
        projected = self.fit.cum[i, step0] * self.fit.f[step0]
        # a per-origin offset large enough that a swapped pair is unmistakable,
        # and a per-draw ramp so the column is a distribution and not a constant
        labelled = projected + 137.0 * (i + 1)
        return labelled[None, :] + np.arange(self._n_draws, dtype=float)[:, None]


class _NoisyEntry(PredictsHeldout):
    """Draws that actually consume the ``rng`` handed to ``_draws_native``.

    The other two stubs do not - ``_MackEchoEntry`` re-seeds its own generator
    so byte equality is reachable, and ``_LabelledEntry`` is deterministic - so
    neither can see whether the caller's seed ever arrives. This one can.
    """

    heldout_draw_scale = "cumulative"

    def __init__(self, fit: MackFit, *, n_draws: int = 32):
        self.fit = fit
        self.contract_ = _contract()
        self._n_draws = n_draws

    def _draws_native(self, cells, *, rng):
        i = np.asarray(cells.w, dtype=int) - 1
        step0 = np.asarray(cells.d, dtype=int) - 2
        projected = self.fit.cum[i, step0] * self.fit.f[step0]
        return projected[None, :] * rng.lognormal(0.0, 0.05, size=(self._n_draws, i.size))


def _echo(fit: MackFit, cells, *, n_draws: int, seed: int) -> GalleryDiagonal:
    return GalleryDiagonal(_MackEchoEntry(fit, n_draws=n_draws, seed=seed), cells)


# -- layer 1: the equivalence -------------------------------------------------


def test_a_mack_echo_entry_reproduces_the_mack_route_exactly():
    """THE test of this module.

    Two routes to one number: ``generator="mack"``, tied out elsewhere in this
    suite to R's published ``CDR(MackChainLadder(MW2014))``, and the same
    conditional draws arriving through an entry, a HoldoutCells, an
    ``index_into``, a scale carry and a cell-to-origin alignment. They must
    agree to the BIT. Not ``allclose``: an ordering slip across accident years,
    or a cell taken one dev step off, both survive a tolerance test on a
    triangle this smooth.

    Both generators consume a freshly seeded RNG in the same order - the open
    origins ascending - which is what makes byte equality reachable at all.
    """
    fit, cells = _fit(), _cells()
    n_draws, seed = 512, 20260729

    reference = simulate_one_year_cdr(
        fit,
        n_draws=n_draws,
        seed=seed,
        generator=MackDiagonal(process="gamma", parameter_risk=True),
    )
    through_gallery = simulate_one_year_cdr(
        fit, n_draws=n_draws, generator=_echo(fit, cells, n_draws=n_draws, seed=seed)
    )

    assert through_gallery.samples.shape == reference.samples.shape
    assert (
        through_gallery.samples.view(np.uint8).tobytes()
        == reference.samples.view(np.uint8).tobytes()
    )
    assert list(through_gallery.targets["label"]) == list(reference.targets["label"])


def test_the_alignment_is_by_origin_and_not_by_row_order():
    """A permutation of the held-out rows must not move the answer.

    ``predict_at``'s columns follow the cells' frame order and ``rereserve``'s
    follow the fit's origin order. Those two agree on any real HoldoutCells,
    which is exactly why a positional shortcut passes every other test here.
    """
    fit = _fit()
    cells = _cells()
    flipped_cells = replace(cells, frame=cells.frame.iloc[::-1].reset_index(drop=True))
    assert list(flipped_cells.frame["origin_period"]) != list(cells.frame["origin_period"])

    entry = _LabelledEntry(fit)
    straight = simulate_one_year_cdr(fit, generator=GalleryDiagonal(entry, cells))
    flipped = simulate_one_year_cdr(fit, generator=GalleryDiagonal(entry, flipped_cells))
    np.testing.assert_array_equal(flipped.samples, straight.samples)

    # and the labelling really does move the answer, so equality above is not
    # two identical constants agreeing
    assert np.ptp(straight.samples[0, : fit.n_w]) > 100.0


def test_the_callers_seed_reaches_the_entry():
    """``simulate_one_year_cdr(seed=...)`` is the single control on
    reproducibility on this route, and the entry is seeded from that rng rather
    than from a second knob of its own.

    A constant in its place leaves ``seed=`` inert - two runs asking for
    different seeds return the same draws - and NOTHING in the answer shows it:
    every number is a legitimate draw from the right distribution. Found by the
    mutation pass, not by the first round of tests, because the two stubs that
    make byte equality possible both ignore the ``rng`` on purpose.
    """
    fit, cells = _fit(), _cells()
    generator = GalleryDiagonal(_NoisyEntry(fit), cells)
    one = simulate_one_year_cdr(fit, seed=1, generator=generator)
    two = simulate_one_year_cdr(fit, seed=2, generator=generator)
    again = simulate_one_year_cdr(fit, seed=1, generator=generator)

    np.testing.assert_array_equal(again.samples, one.samples)
    assert not np.array_equal(one.samples, two.samples)


def test_closed_origins_get_a_zero_cdr_and_open_ones_do_not():
    """An origin already at the last dev column has no next cell, so its CDR is
    identically zero - and every open origin's must not be, or the alignment
    quietly dropped it."""
    fit = _fit()
    pred = simulate_one_year_cdr(fit, generator=_echo(fit, _cells(), n_draws=128, seed=3))
    closed = fit.latest_dev == fit.n_d - 1
    per_origin = pred.samples[:, : fit.n_w]
    assert closed.any() and (~closed).any(), "the fixture must have both kinds of origin"
    assert np.all(per_origin[:, closed] == 0.0)
    assert np.all(per_origin[:, ~closed].std(axis=0) > 0)


# -- layer 2: the refusals ----------------------------------------------------


def test_a_missing_open_origin_is_refused_rather_than_read_as_zero():
    """The refusal that matters most, and the one a leaderboard does NOT make.

    ``align_panel`` prices a dropped cell into ``panel.dropped`` and carries on,
    because a smaller CRPS panel is still a panel. A CDR has no such reading: a
    missing diagonal cell re-reserves to a zero ultimate, so the origin's whole
    estimated ultimate is reported as one year's development result.
    """
    fit, cells = _fit(), _cells()
    short = replace(cells, frame=cells.frame.iloc[1:].reset_index(drop=True))
    with pytest.raises(ValueError, match="no scorable held-out cell"):
        GalleryDiagonal(_MackEchoEntry(fit, n_draws=8, seed=1), short).check(fit)


def test_a_mismatched_predecessor_is_refused():
    """The load-bearing check: the held-out cell's ``prev_value`` and the fit's
    latest diagonal are the same cell read by two code paths. Perturbing one
    stands in for every way the two objects can come from different cutoffs,
    cohorts or fields - each of which re-reserves cleanly and means nothing."""
    fit, cells = _fit(), _cells()
    frame = cells.frame.copy()
    frame.loc[0, "prev_value"] = float(frame.loc[0, "prev_value"]) * 1.001
    with pytest.raises(ValueError, match="the fit's latest diagonal there is"):
        GalleryDiagonal(_MackEchoEntry(fit, n_draws=8, seed=1), replace(cells, frame=frame)).check(
            fit
        )


def test_a_cell_on_the_wrong_development_step_is_refused():
    fit, cells = _fit(), _cells()
    frame = cells.frame.copy()
    frame.loc[0, "dev_lag"] = int(frame.loc[0, "dev_lag"]) + 12
    with pytest.raises(ValueError, match="different diagonals or different grains"):
        GalleryDiagonal(_MackEchoEntry(fit, n_draws=8, seed=1), replace(cells, frame=frame)).check(
            fit
        )


def test_a_field_mismatch_between_the_fit_and_the_cells_is_refused():
    """Re-reserving reported-loss draws through paid-loss factors is arithmetic
    that works and means nothing."""
    fit = _fit()
    both = next_diagonal(
        _triangle(extra_field=True), as_of=AS_OF, fields=["paid_loss", "reported_loss"]
    )
    entry = _MackEchoEntry(fit, n_draws=8, seed=1)

    # two fields and no field=: refused for ambiguity rather than sorted
    with pytest.raises(ValueError, match="pass field= to name the one"):
        GalleryDiagonal(entry, both).check(fit)
    # the wrong one, named explicitly: refused against the fit
    with pytest.raises(ValueError, match="two different loss fields"):
        GalleryDiagonal(entry, both, field="reported_loss").check(fit)
    # the right one is accepted
    GalleryDiagonal(entry, both, field="paid_loss").check(fit)


def test_an_entry_trained_on_a_different_history_is_refused():
    """The third leg of the alignment, and the one review had to find.

    The other two checks tie the CELLS to the fit. Nothing tied the ENTRY to
    either, so a real entry refitted on RESTATED history - two interior cells
    moved, latest diagonal untouched, so ``prev_value`` still matched perfectly
    - sailed through. Measured before the fix on this fixture: total CDR mean
    0.27 with the matching history, -367.69 with the restated one, seven times
    the spread, every number finite and plausible.

    Uses the real ``mack`` gallery entry rather than a stub, because the point
    is that a legitimately-fitted entry was accepted against the wrong fit.
    """
    restated = FULL.copy()
    restated[0, 1] = 1200.0  # interior cells: off every diagonal at AS_OF,
    restated[1, 1] = 1400.0  # so the latest-diagonal check cannot see them
    tri, tri_restated = _triangle(), make_cohort_triangle(None, restated, start_year=START_YEAR)

    fit = fit_mack(tri, loss_field="paid_loss", as_of=AS_OF)
    cells = next_diagonal(tri, as_of=AS_OF, fields="paid_loss")
    matching = get("mack")().fit(tri.as_of(AS_OF), loss_field="paid_loss")
    different = get("mack")().fit(tri_restated.as_of(AS_OF), loss_field="paid_loss")

    # the restated fit still agrees on the latest diagonal, so the older checks
    # genuinely cannot tell the two apart
    other_fit = fit_mack(tri_restated, loss_field="paid_loss", as_of=AS_OF)
    np.testing.assert_allclose(other_fit.latest, fit.latest)

    GalleryDiagonal(matching, cells).check(fit)
    with pytest.raises(ValueError, match="fitted on a different triangle"):
        GalleryDiagonal(different, cells).check(fit)


def test_a_contract_with_no_raw_cumulatives_is_refused_by_name():
    """The NN case, and the honest limit of this route in 0.5.1.

    ``nn_contract`` keeps its losses NORMALIZED inside ``x`` and holds only the
    anchor diagonal in amounts, so there is no way to show an NN entry was
    fitted on the same triangle as the MackFit. Refused by name rather than
    assumed, because the assumption is the one measured above to move the answer
    by orders of magnitude. Simulated with a contract stripped of both value
    shapes, so the refusal is pinned without a torch fit.
    """
    fit, cells = _fit(), _cells()
    entry = _MackEchoEntry(fit, n_draws=8, seed=1)
    entry.contract_ = {
        k: v for k, v in entry.contract_.items() if k not in ("loss", "cum", "obs_mask")
    }
    with pytest.raises(ValueError, match="states no training losses on the cumulative grid"):
        GalleryDiagonal(entry, cells).check(fit)


def test_a_delta_stacked_contract_is_refused_by_name():
    """Compartmental's shape: two rows per ``(w, d)`` - outstanding and paid -
    so a value comparison would be against whichever came first. The uniqueness
    filter rejects every cell, and an entry with nothing left to compare is
    refused rather than passed for free."""
    fit, cells = _fit(), _cells()
    entry = _MackEchoEntry(fit, n_draws=8, seed=1)
    stacked = dict(entry.contract_)
    n = len(stacked["w"])
    for key in ("w", "d", "loss"):
        stacked[key] = np.concatenate([stacked[key], stacked[key]])
    stacked["delta"] = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    entry.contract_ = stacked

    with pytest.raises(ValueError, match="unambiguously"):
        GalleryDiagonal(entry, cells).check(fit)


def test_an_entry_without_the_capability_is_refused_at_construction():
    class _NotAPredictor:
        pass

    with pytest.raises(TypeError, match="must declare PredictsHeldout"):
        GalleryDiagonal(_NotAPredictor(), _cells())


def test_incremental_cells_are_refused():
    """Re-reserving is a ratio of cumulatives; increments index at the same
    (w, d) and would re-reserve into nonsense factors without erroring."""
    fit, cells = _fit(), _cells()
    with pytest.raises(ValueError, match="needs cumulative values"):
        GalleryDiagonal(
            _MackEchoEntry(fit, n_draws=8, seed=1), replace(cells, measure="incremental")
        )


def test_the_mack_positivity_precondition_is_not_applied_here():
    """``require_positive_open_diagonals`` is Mack's - his conditional variance
    is proportional to the diagonal cell. This generator's draws come from a fit
    that already happened, so it must not inherit a guard belonging to a model
    it is replacing. The same call on ``MackDiagonal`` refuses."""
    # the youngest trained origin's diagonal cell, which at this cutoff is its
    # first dev step: the "zero paid at 12 months" accident year Mack refuses
    # and the chain-ladder point estimate does not
    zeroed = FULL.copy()
    zeroed[4, 0] = 0.0
    tri = make_cohort_triangle(None, zeroed, start_year=START_YEAR)
    fit = fit_mack(tri, loss_field="paid_loss", as_of=AS_OF)
    cells = next_diagonal(tri, as_of=AS_OF, fields="paid_loss")

    with pytest.raises(ValueError, match="non-positive cumulative on the latest diagonal"):
        MackDiagonal().check(fit)
    # cohort_grid, not stan_data: the zeroed cell is exactly what stan_data's
    # lognormal positivity guard refuses, and the grid contract is the shape the
    # deterministic entries carry anyway
    entry = _MackEchoEntry(
        fit, n_draws=8, seed=1, contract=cohort_grid(tri.as_of(AS_OF), loss_field="paid_loss")
    )
    GalleryDiagonal(entry, cells).check(fit)


# -- the draw-count negotiation -----------------------------------------------


def test_an_explicit_draw_count_that_is_not_the_posteriors_is_refused():
    """The inert-parameter refusal, at this module's one place to make it.

    A posterior has the size it was fitted at. Silently returning 400 draws to a
    caller who asked for 20,000 would give them a 99.5th percentile that looks
    like it rests on 100 observations and rests on 2 - and resampling to their
    number manufactures exactly that illusion.
    """
    fit = _fit()
    generator = _echo(fit, _cells(), n_draws=400, seed=5)
    with pytest.raises(ValueError, match="no way to ask for a different number"):
        simulate_one_year_cdr(fit, n_draws=20_000, generator=generator)

    # ... and asking for nothing in particular takes the posterior's own size
    assert simulate_one_year_cdr(fit, generator=generator).samples.shape[0] == 400

    # the refusal must not tell a caller to do something they cannot: an entry
    # whose draws are a simulation (clark, the NN heads) has no draw-count knob
    # to refit with, because _draws_native takes no count at all
    with pytest.raises(ValueError) as exc:
        simulate_one_year_cdr(fit, n_draws=20_000, generator=generator)
    assert "no way to ask for a different number" in str(exc.value)
    assert "refit" not in str(exc.value)


def test_the_mack_wrappers_default_does_not_defeat_the_negotiation():
    """``Mack.cdr_distribution`` must default ``n_draws`` to None too.

    A hardcoded 20_000 on the wrapper reaches the generator as an EXPLICIT
    request, so the default call would refuse every GalleryDiagonal source -
    the negotiation works on the kernel function and is defeated one layer up,
    which is the inert-parameter shape with the polarity reversed.
    """
    import inspect

    from ibnr.gallery.deterministic.mack.model import Mack

    assert inspect.signature(Mack.cdr_distribution).parameters["n_draws"].default is None

    tri = _triangle()
    entry = get("mack")().fit(tri.as_of(AS_OF), loss_field="paid_loss")
    source = _MackEchoEntry(_fit(), n_draws=64, seed=2)
    pred = entry.cdr_distribution(generator=GalleryDiagonal(source, _cells()))
    assert pred.samples.shape[0] == 64


def test_the_none_default_leaves_the_simulating_generators_where_they_were():
    """``n_draws`` became ``None``-defaulted in 0.5.1. It must resolve to the
    literal that used to be the default, or every published Mack CDR silently
    changes its Monte Carlo error."""
    assert DEFAULT_N_DRAWS == 20_000
    for generator in (MackDiagonal(), get_cdr_method("odp_bootstrap").generator()):
        assert generator.resolve_n_draws(None) == 20_000
        assert generator.resolve_n_draws(37) == 37
    gallery_gen = _echo(_fit(), _cells(), n_draws=8, seed=1)
    assert gallery_gen.resolve_n_draws(None) is None
    assert gallery_gen.resolve_n_draws(8) == 8


# -- the option surface -------------------------------------------------------


def test_the_gallery_route_is_listed_and_refused_by_name():
    """It belongs in ``cdr_methods()`` - a caller asking what their options are
    must be told it exists - and it cannot be reached through
    ``generator="gallery"``, because a string cannot carry a fitted entry. Two
    different reasons to be nameless now live in that table, and ``route`` is
    what tells them apart."""
    table = cdr_methods()
    row = table.set_index("name").loc["gallery"]
    assert row["route"] == "simulation"
    assert "GalleryDiagonal" in row["entry_point"]
    assert "generator" not in table.columns  # the class never rides in the frame

    method = get_cdr_method("gallery")
    assert method.generator is None and method.why_not_by_name
    assert get_cdr_method("merz_wuthrich").route == "analytic"

    with pytest.raises(ValueError, match="cannot be named as a generator"):
        simulate_one_year_cdr(_fit(), n_draws=8, generator="gallery")


def test_every_registry_row_is_either_nameable_or_says_why_not():
    """The invariant the third route made non-obvious: ``generator`` and
    ``why_not_by_name`` are two states of one fact, never both and never
    neither."""
    for key, method in CDR_METHODS.items():
        assert key == method.name
        assert (method.generator is None) == bool(method.why_not_by_name)
        if method.generator is not None:
            assert method.generator.name == key


def test_the_export_is_reachable_from_the_gallery_namespace():
    """Decision 8's export set. A caller cannot write this generator down
    without the name, and no string in ``cdr_methods()`` can substitute."""
    from ibnr import gallery

    assert gallery.GalleryDiagonal is GalleryDiagonal
    assert "GalleryDiagonal" in gallery.__all__
