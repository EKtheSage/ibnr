"""One-year CDR from a gallery model's view of next year's diagonal.

``kernels/cdr.py`` split the simulated one-year claims development result into
two axes: what generates next year's diagonal, and how the reserve is
re-estimated once it exists. The second is fixed (the volume-weighted chain
ladder, ``kernels.cdr.rereserve``). The first was Mack's conditional moments or
an ODP residual bootstrap. This module adds the third source, and it is the one
the gallery already had lying around: **eleven entries can draw the outcome at
held-out cells**, and the next diagonal is exactly the cells they are drawn at.

    from ibnr import gallery
    from ibnr.kernels import fit_mack, simulate_one_year_cdr
    from ibnr.kernels.cdr import cdr_risk_measures

    cells = gallery.next_diagonal(full, as_of="1997-12-31", fields="paid_loss")
    entry = gallery.get("meyers_csr")().fit(full.as_of("1997-12-31"))
    fit = fit_mack(full, as_of="1997-12-31", loss_field="paid_loss")

    pred = simulate_one_year_cdr(fit, generator=gallery.GalleryDiagonal(entry, cells))
    cdr_risk_measures(pred, levels=(0.995,))

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
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ibnr.gallery.entry import PredictsHeldout
from ibnr.kernels.cdr import DiagonalGenerator
from ibnr.kernels.holdout import HoldoutCells
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

    **The alignment is checked against the fit, not assumed.** A ``MackFit`` and
    a ``HoldoutCells`` are two objects built by two calls, and nothing about
    their types says they describe the same cohort at the same cutoff on the
    same field. Handed a mismatched pair, ``rereserve`` would happily return a
    complete page of plausible dollars. So :meth:`check` requires, for every
    OPEN origin of the fit, exactly one scorable cell, at the development lag
    one step past that origin's diagonal, whose ``prev_value`` equals the fit's
    own latest-diagonal cell. That last comparison is the load-bearing one: the
    predecessor of the held-out cell and the fit's diagonal are the same number
    read by two different code paths, so requiring them equal catches a wrong
    cutoff, a wrong field, a wrong cohort and a wrong grain in one check.

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

        An explicit count is not ignored - :meth:`draw` refuses one that is not
        the posterior's, rather than letting it sit there inert.
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
                f"n_draws={n_draws} was requested but {type(self.entry).__name__} has "
                f"{draws.shape[0]} posterior draws. A fitted posterior is not a Monte "
                "Carlo budget: resampling it to the requested size would add noise and "
                "no information, and refitting to it is not something this call can do. "
                "Pass n_draws=None to take the posterior's own size, or refit the entry "
                "with the draw count you want"
            )

        x = np.zeros((draws.shape[0], fit.n_w))
        x[:, open_] = draws[:, columns]
        return x
