"""One-year CDR from a gallery model's view of next year's diagonal.

``kernels/cdr.py`` split the simulated one-year claims development result into
two axes: what generates next year's diagonal, and how the reserve is
re-estimated once it exists. The second is fixed (the volume-weighted chain
ladder, ``kernels.cdr.rereserve``). The first was Mack's conditional moments or
an ODP residual bootstrap. This module adds the third source, and it is the one
the gallery already had lying around: an entry that can draw the outcome at
held-out cells can draw next year's diagonal, because that is what those cells
are.

WHICH ENTRIES, EXACTLY - narrower than this module's first draft claimed. Every
entry whose contract keeps its training losses as raw cumulatives: the
deterministic and statistical entries (``cohort_grid``) and the stan-based
Bayesian ones (``stan_data`` / ``odp_stan_data``). NOT the NN entries, whose
contracts hold only normalized values plus an anchor diagonal, and NOT
``compartmental``, whose contract delta-stacks two quantities onto one grid.
Both are refused by name, and :func:`_training_values` says why: without raw
cumulatives there is no way to show the entry and the ``MackFit`` were fitted on
the same triangle, and the section below is what that assumption is worth.

    from ibnr import gallery
    from ibnr.kernels import fit_mack, simulate_one_year_cdr
    from ibnr.kernels.cdr import cdr_risk_measures

    cells = gallery.next_diagonal(full, as_of="1997-12-31", fields="paid_loss")
    entry = gallery.get("meyers_csr")().fit(full.as_of("1997-12-31"))
    fit = fit_mack(full, as_of="1997-12-31", loss_field="paid_loss")

    pred = simulate_one_year_cdr(fit, generator=gallery.GalleryDiagonal(entry, cells))
    cdr_risk_measures(pred, levels=(0.995,))

BACKTEST ONLY, AS SHIPPED IN 0.5.1. READ THIS BEFORE PLANNING A VALUATION ON IT.

``GalleryDiagonal`` takes a :class:`~ibnr.kernels.holdout.HoldoutCells`, and
``next_diagonal`` builds those only from cells that are ALREADY OBSERVED after
the cutoff - it reads ``D_next`` out of the data and raises "no eval_date after
... introduces a new cell" when there is none. So this route answers "what would
next year's re-reserve have looked like, from where we stood at a past cutoff",
and it cannot yet answer the same question at a CURRENT valuation, which is the
Solvency II use the one-year CDR exists for.

The two kernel generators have no such limit - ``mack`` and ``odp_bootstrap``
manufacture the diagonal from the fit alone, so ``simulate_one_year_cdr(fit)``
is prospective today. Only the gallery route needs cells, and only because
``PredictsHeldout.predict_at`` is keyed on them.

Nothing about the information is missing: a prospective cell is a location plus
a training-diagonal predecessor, both known at the cutoff, and the outcome is
the one thing a prediction does not need. What is missing is a constructor -
``HoldoutCells`` requires a ``value`` column, and deciding what that means with
no outcome is a design question, not a patch. Until then, the honest reading of
a number from this route is retrospective.

WHAT THIS QUANTITY IS, STATED PLAINLY, BECAUSE IT WILL BE MISQUOTED OTHERWISE.

It is **the chain ladder's one-year CDR under model M's view of next year**, not
"model M's one-year CDR". Both ultimates being differenced are chain-ladder
ultimates - the one estimated today, and the one that will be estimated next
year off the extended triangle - and the only thing the entry supplies is the
diagonal in between. That is deliberate and it is the same structure R's
``CDR.BootChainLadder`` has: the bootstrap does not re-reserve with the
bootstrap, it re-reserves with the chain ladder.

The honest version of "model M's own one-year CDR" would refit M on the extended
triangle, once per draw. That is thousands of MCMC fits per cohort, and it is
not what any published method does. It is not offered rather than approximated.

WHY THE MEAN IS NOT ZERO HERE, AND WHY THAT IS THE INTERESTING PART.
``E[CDR | D_I] = 0`` is a Mack result: it holds because Mack's conditional mean
for the next cell IS the chain-ladder projection, so the two ultimates agree in
expectation. Nothing makes CSR or a transformer agree with the chain ladder, so
their CDR draws are centred wherever their diagonal disagrees. That drift is a
real disagreement between two reserving methods measured in dollars of next
year's balance sheet - report it, do not square it away. Read ``mean_cdr`` and
``sd_cdr`` from ``kernels.cdr.cdr_risk_measures``; ``simulated_msep`` mixes the
two into one number and is a Mack-route measure.

WHAT IS VALIDATED. The wiring, and only the wiring:
``tests/test_gallery_cdr.py`` drives this generator with a stub entry whose
draws ARE Mack's conditional draws, and requires the CDR to come out
bit-identical to ``generator="mack"``. So the entry-to-diagonal path - field
resolution, cell-to-origin alignment, the cumulative carry, the draw count -
contributes nothing of its own. Whether a given entry's diagonal is any GOOD is
the leaderboard's CRPS question (milestone 6), not this one, and no amount of
CDR machinery answers it.

Lives in the gallery, not in ``kernels``, because it imports
``gallery.entry.PredictsHeldout`` and the re-export direction is gallery ->
kernels only (design decision 8; ``tests/test_import_purity.py`` enforces it in
a subprocess). ``kernels.cdr.CDR_METHODS`` still lists the route, naming this
class as a string - a table entry, not an import.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ibnr.gallery.entry import PredictsHeldout
from ibnr.kernels.cdr import DiagonalGenerator, _require_zero_cells_unused
from ibnr.kernels.holdout import HoldoutCells, training_index
from ibnr.kernels.mack import MackFit


def _as_date(value) -> dt.date:
    """One date type on both sides of the comparison.

    ``HoldoutCells.frame`` comes back through pandas and ``MackFit.origin_periods``
    is a list of ``datetime.date``; a ``Timestamp`` and a ``date`` for the same
    day are unequal as dict keys, which would report every origin as missing.
    """
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def _training_values(contract: dict, *, who: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(w, d, value)`` 0-based training cells of a contract, or a refusal.

    The gallery has three contract shapes and only two of them state their
    training losses as raw amounts on the cumulative grid:

    * ``kernels.contract.cohort_grid`` - a dense ``cum`` matrix plus
      ``obs_mask``. The deterministic and statistical entries.
    * ``stan_data`` / ``odp_stan_data`` / ``compartmental_stan_data`` - flat
      ``w``/``d``/``loss``, 1-based. Compartmental's is delta-STACKED, two rows
      per position, which the caller's uniqueness filter then rejects.
    * ``nn_contract`` - no raw values at all. The losses live inside ``x`` as
      **normalized** feature channels, with only ``latest_cum`` (the anchor
      diagonal) kept in amounts. The diagonal is already checked elsewhere and
      the interior simply is not recoverable here.

    So the NN entries cannot use the gallery CDR route today, and are told so by
    name. That is narrower than 0.5.1's first draft claimed, and it is the
    honest position: the alternative is a number this repo has measured going
    wrong by orders of magnitude with nothing visible in the output.
    """
    if "cum" in contract and "obs_mask" in contract:
        cum = np.asarray(contract["cum"], dtype=float)
        mask = np.asarray(contract["obs_mask"], dtype=bool)
        w, d = np.nonzero(mask)
        return w, d, cum[w, d]
    if all(k in contract for k in ("w", "d", "loss")):
        idx = training_index(contract)
        return (
            np.asarray(idx.w, dtype=int) - 1,
            np.asarray(idx.d, dtype=int) - 1,
            np.asarray(idx.value, dtype=float),
        )
    raise ValueError(
        f"{who}'s contract states no training losses on the cumulative grid, so it cannot "
        "be shown to have been fitted on the same triangle as this MackFit. The NN "
        "contracts are the case that reaches this: their values are normalized inside "
        "`x` and only the anchor diagonal survives in amounts. Until that changes, the "
        "gallery CDR route serves the entries whose contracts keep raw cumulatives - use "
        "the mack or odp_bootstrap generator, or score this entry on the leaderboard "
        "instead"
    )


@dataclass(frozen=True)
class GalleryDiagonal(DiagonalGenerator):
    """Next year's diagonal from a fitted gallery entry's posterior predictive.

    ``entry`` is anything with :class:`~ibnr.gallery.entry.PredictsHeldout` -
    a fitted single-cohort entry, or the per-cohort view a pooled NN entry
    returns from ``at_cohort(segment)``. ``cells`` is the
    :class:`~ibnr.kernels.holdout.HoldoutCells` from
    ``gallery.next_diagonal(...)``, the same object the leaderboard scores at.
    ``field`` names which loss field to re-reserve when the cells carry more
    than one (``compartmental`` scores paid and reported together); with a
    single field it is inferred and must not be repeated.

    **The alignment is checked against the fit, not assumed.** A ``MackFit``, a
    ``HoldoutCells`` and a fitted entry are three objects built by three calls,
    and nothing about their types says they describe the same cohort at the same
    cutoff on the same field off the same history. Handed a mismatched set,
    ``rereserve`` would happily return a complete page of plausible dollars. So
    :meth:`check` requires, for every OPEN origin of the fit, exactly one
    scorable cell, at the development lag one step past that origin's diagonal,
    whose ``prev_value`` equals the fit's own latest-diagonal cell. That
    comparison is load-bearing: the predecessor of the held-out cell and the
    fit's diagonal are the same number read by two different code paths, so
    requiring them equal catches a wrong cutoff, a wrong field, a wrong cohort
    and a wrong grain in one check.

    It does **not** catch a wrong history, which is why
    :meth:`_require_same_training_history` exists as a separate third leg - the
    first two tie the CELLS to the fit and leave the ENTRY unbound to either.

    **An excluded cell is fatal here, and is not on a leaderboard.** A cohort
    whose next diagonal is short a cell simply scores on fewer cells for CRPS -
    ``align_panel`` prices that in ``panel.dropped`` and moves on. A CDR cannot:
    a missing diagonal cell is not a smaller diagonal, it is an origin whose
    re-reserved ultimate would silently be taken as zero and whose whole
    estimated ultimate would then be reported as one year's development result.
    Hence a refusal naming the origin and the exclusion counts.

    **Nothing here constrains the sign of a draw.** A distributional head with
    unbounded support can draw a negative cumulative, which re-reserves into a
    meaningless factor. That is the entry's property, not this class's, and it
    is left visible rather than clipped - the same stance
    ``kernels.mack.PROCESS_LAWS`` takes when it documents that only ``gamma``
    and ``lognormal`` guarantee a positive diagonal. Check ``mean_cdr`` against
    the fit's reserve before believing a tail.
    """

    entry: PredictsHeldout
    cells: HoldoutCells
    field: str | None = None

    name = "gallery"

    def __post_init__(self) -> None:
        if not isinstance(self.entry, PredictsHeldout):
            raise TypeError(
                f"entry must declare PredictsHeldout, and {type(self.entry).__name__} does "
                "not. Drawing the outcome at a cell the fit never saw is an opt-in "
                "capability: an entry claims it by subclassing the mixin, and one that "
                "cannot is not made able to by being asked. For a pooled NN entry, pass "
                "entry.at_cohort(segment) rather than the entry"
            )
        if not isinstance(self.cells, HoldoutCells):
            raise TypeError(
                "cells must be a HoldoutCells from gallery.next_diagonal(...); got "
                f"{type(self.cells).__name__}"
            )
        if self.cells.measure != "cumulative":
            raise ValueError(
                f"these cells are {self.cells.measure!r}, and re-reserving needs cumulative "
                "values: the chain ladder is refitted on the extended triangle, and its "
                "factors are ratios of cumulatives. An incremental triangle has no "
                "MackFit to pair this with either"
            )

    # -- alignment -------------------------------------------------------------

    def _resolve_field(self) -> str:
        """Which loss field this CDR re-reserves.

        Resolved from the CELLS rather than from the entry's contract, because
        it also has to be compared with the fit's, and a three-way agreement
        checked at one place beats two pairwise checks in two.
        """
        present = sorted({str(f) for f in self.cells.frame["field"].unique()})
        if self.field is not None:
            if self.field not in present:
                raise ValueError(
                    f"field={self.field!r} is not among the held-out cells' fields {present}"
                )
            return self.field
        if len(present) != 1:
            raise ValueError(
                f"these cells carry fields {present}; pass field= to name the one to "
                "re-reserve. A CDR is a statement about one loss field, and taking the "
                "first would pick it by sort order"
            )
        return present[0]

    def _require_same_training_history(self, fit: MackFit) -> int:
        """The entry and the fit must have been trained on the SAME triangle.

        Returns how many cells were compared. The other two alignment checks tie
        the CELLS to the fit; nothing tied the ENTRY to either, and that gap is
        not theoretical. Measured on the test fixture: a ``mack`` entry refitted
        on a triangle with two restated INTERIOR cells - same latest diagonal,
        so ``prev_value`` still matched perfectly - was accepted, and the total
        CDR mean moved from 0.27 to -367.69 with seven times the spread. Every
        number finite, every number plausible, no warning anywhere.

        It matters because the two halves of the difference come from different
        places. ``fit.ultimate`` and the updated factors are the chain ladder's,
        off the fit's triangle; the diagonal is the entry's, off the entry's. If
        those triangles differ the subtraction is between two estimates of
        different things, and what comes out is not a development result.

        Compares the entry's own training values against ``fit.cum``, at every
        cell the entry indexes **unambiguously** - exactly one training row per
        ``(w, d)``. That qualifier is load-bearing rather than cautious:
        ``compartmental_stan_data`` delta-stacks two quantities onto one grid, so
        each ``(w, d)`` carries an outstanding row AND a paid row and a value
        comparison would be against whichever came first. Where nothing is
        unambiguous the entry is refused by name rather than waved through - the
        finding above is what an unverified pass is worth.
        """
        contract = getattr(self.entry, "contract_", None)
        if not isinstance(contract, dict):
            raise ValueError(
                f"{type(self.entry).__name__} carries no contract_, so its training data "
                "cannot be compared with the fit's. Both halves of a CDR must come from "
                "one triangle and there is no way to check that here"
            )
        w, d, value = _training_values(contract, who=type(self.entry).__name__)
        positions = list(zip(w.tolist(), d.tolist(), strict=True))
        seen = Counter(positions)
        usable = (
            np.array([seen[p] == 1 for p in positions], dtype=bool) & (w < fit.n_w) & (d < fit.n_d)
        )
        if not usable.any():
            raise ValueError(
                f"{type(self.entry).__name__}'s contract indexes no cell of this fit "
                f"unambiguously ({len(w)} training rows over {len(seen)} distinct (w, d) "
                "positions), so its training history cannot be checked against the fit's. "
                "A stacked contract - compartmental puts paid and outstanding on one grid - "
                "reaches this. Re-reserving anyway would rest on an assumption measured to "
                "be worth orders of magnitude on the answer"
            )

        got = value[usable]
        want = fit.cum[w[usable], d[usable]]
        bad = ~(np.isclose(got, want, rtol=1e-9, atol=0.0) | (np.isnan(got) & np.isnan(want)))
        if bad.any():
            first = int(np.nonzero(bad)[0][0])
            cells_w, cells_d = w[usable][first], d[usable][first]
            raise ValueError(
                f"{type(self.entry).__name__} was fitted on a different triangle from this "
                f"MackFit: {int(bad.sum())} of {int(usable.sum())} compared training cells "
                f"disagree, the first at origin {fit.origin_periods[cells_w]} dev index "
                f"{cells_d}, where the entry has {got[first]:.10g} and the fit has "
                f"{want[first]:.10g}. The diagonal and the factors would come from different "
                "loss histories - a restatement that leaves the latest diagonal untouched "
                "passes every other check here and still moves the answer by orders of "
                "magnitude"
            )
        return int(usable.sum())

    def _align(self, fit: MackFit) -> tuple[str, np.ndarray, np.ndarray]:
        """``(field, open_origin_indices, columns)`` tying cells to the fit.

        ``columns[k]`` is the position, in the entry's ``(n_draws, n_cells)``
        output, of the cell belonging to origin ``open_origin_indices[k]``.
        Pure and cheap - a few numpy ops over one small frame - so
        :meth:`check` and :meth:`draw` each call it rather than sharing state.
        """
        resolved = self._resolve_field()
        if fit.loss_field is not None and fit.loss_field != resolved:
            raise ValueError(
                f"the fit was built on {fit.loss_field!r} and these cells re-reserve "
                f"{resolved!r}. The chain-ladder factors and the drawn diagonal would be "
                "two different loss fields, which re-reserves cleanly and means nothing"
            )
        self._require_same_training_history(fit)

        # predict_at's columns are the frame's rows for this field, in frame
        # order - index_into filters the same way and neither narrowing nor
        # re-keying reorders rows. Positions are taken here the same way, so the
        # two orderings cannot drift.
        sub = self.cells.frame[self.cells.frame["field"] == resolved]
        origins = [_as_date(o) for o in sub["origin_period"]]
        dev_lags = sub["dev_lag"].to_numpy(dtype=np.int64)
        prev = sub["prev_value"].to_numpy(dtype=float)

        position: dict[dt.date, int] = {}
        for pos, origin in enumerate(origins):
            if origin in position:
                raise ValueError(
                    f"two held-out cells for origin {origin} on field {resolved!r}; the "
                    "next diagonal has one cell per origin, so this is not one diagonal"
                )
            position[origin] = pos

        open_ = np.nonzero(fit.latest_dev < fit.n_d - 1)[0]
        columns = np.empty(open_.size, dtype=np.int64)
        for k, i in enumerate(open_):
            origin = _as_date(fit.origin_periods[i])
            if origin not in position:
                counts = {r: n for r, n in self.cells.exclusion_counts().items() if n}
                raise ValueError(
                    f"open origin {origin} has no scorable held-out cell, so next year's "
                    "diagonal is incomplete and there is no CDR for it. A short diagonal "
                    "is survivable on a leaderboard - it scores fewer cells - but not "
                    "here: re-reserving would read the missing cell as a zero ultimate "
                    f"and report this origin's entire estimated ultimate as one year's "
                    f"development. Excluded by next_diagonal: {counts or 'nothing'}"
                )
            pos = position.pop(origin)
            columns[k] = pos

            expected_lag = (int(fit.latest_dev[i]) + 2) * fit.dev_grain_months
            if int(dev_lags[pos]) != expected_lag:
                raise ValueError(
                    f"origin {origin} sits at dev_lag {int(dev_lags[pos])} in the held-out "
                    f"cells but the fit's diagonal there is at dev index "
                    f"{int(fit.latest_dev[i])}, so its next cell is dev_lag {expected_lag}. "
                    "The cells and the fit are on different diagonals or different grains"
                )
            if not np.isclose(prev[pos], fit.latest[i], rtol=1e-9, atol=0.0):
                raise ValueError(
                    f"origin {origin}: the held-out cell's predecessor is {prev[pos]:.10g} "
                    f"but the fit's latest diagonal there is {fit.latest[i]:.10g}. These "
                    "are the same cell read twice, so a difference means the fit and the "
                    "cells were built from different cutoffs, cohorts or fields"
                )

        if position:
            stray = sorted(position)
            raise ValueError(
                f"held-out cells for origin(s) {stray[:5]} that are not open origins of "
                f"this fit, whose origins are {fit.origin_periods[0]} .. "
                f"{fit.origin_periods[-1]} with {open_.size} open. A cell past the fit's "
                "development grid has no factor to be re-reserved through; one at an "
                "origin the fit never had is a different cohort"
            )
        return resolved, open_, columns

    # -- the DiagonalGenerator contract ----------------------------------------

    def check(self, fit: MackFit) -> None:
        """Every alignment refusal, before a single posterior draw is touched.

        Deliberately does NOT call ``fit.require_positive_open_diagonals()``.
        That is Mack's precondition - his conditional variance is proportional
        to the diagonal cell - and it is no more this generator's than it is the
        bootstrap's. What an entry needs of the diagonal is whatever it needed
        to fit; the fit already happened.
        """
        _require_zero_cells_unused(fit)
        self._align(fit)

    def resolve_n_draws(self, requested: int | None) -> int | None:
        """``None`` - the posterior's own size, whatever it is.

        The base class resolves ``None`` to a Monte Carlo budget because for
        ``mack`` and ``odp_bootstrap`` the draw count is one: ask for more
        diagonals and you get more. Here the draws come from a posterior that
        was sized at fit time, and asking for 20,000 could only mean resampling
        them with replacement - Monte Carlo noise added on top of the posterior,
        and a 99.5th percentile that appears to rest on 100 observations when it
        rests on however many distinct draws there really were.

        The same applies, for a different reason, to an entry whose draws are a
        SIMULATION rather than a posterior - ``clark``'s parametric bootstrap,
        an NN mixture head. Those could in principle produce any count, but
        ``PredictsHeldout._draws_native(cells, rng)`` takes no count argument, so
        there is no way to ask; the number is fixed by the entry's own
        configuration either way.

        An explicit count is not ignored - :meth:`draw` refuses one it cannot
        honour, rather than letting it sit there inert.
        """
        return requested

    def draw(self, fit: MackFit, *, n_draws: int | None, rng: np.random.Generator) -> np.ndarray:
        """``(n_draws, n_w)`` cumulative loss on next year's diagonal, zero at
        closed origins.

        The entry is seeded FROM ``rng`` rather than from a seed of its own, so
        ``simulate_one_year_cdr(..., seed=s)`` remains the single control on
        reproducibility - a second seed on this class would be a second thing to
        set and a silent way for two "identical" runs to differ.
        """
        resolved, open_, columns = self._align(fit)
        seed = int(rng.integers(np.iinfo(np.int64).max))
        draws = np.asarray(
            self.entry.predict_at(self.cells, field=resolved, seed=seed), dtype=float
        )
        if n_draws is not None and draws.shape[0] != n_draws:
            raise ValueError(
                f"n_draws={n_draws} was requested but {type(self.entry).__name__} produced "
                f"{draws.shape[0]} draws, and PredictsHeldout has no way to ask for a "
                "different number: `_draws_native(cells, rng)` takes no count. That is a "
                "limit of the capability, not of this entry - for an MCMC entry the count "
                "IS the posterior's, and for a simulator-backed one (clark, the NN heads) "
                "it is whatever the entry's own configuration produces. Pass n_draws=None "
                "to accept it. Resampling to the requested size is deliberately not done: "
                "it would add Monte Carlo noise and no information while making the answer "
                "look as precise as the larger number"
            )

        x = np.zeros((draws.shape[0], fit.n_w))
        x[:, open_] = draws[:, columns]
        return x
