"""Mack's distribution-free chain ladder as a gallery entry, with the one-year
claims development result attached.

The deterministic family's reference point: no distributional assumption, no
priors, no sampler - the classical volume-weighted chain ladder plus Mack's
(1993) analytic second moments, wrapped in the simulation CLAUDE.md decision 4
requires before anything may enter the gallery (``predict()`` returns a
``PredictiveDistribution`` of ultimates like every other entry, so the shared
evaluation harness scores it unchanged).

What makes this entry more than a baseline is the one-year view. Reserve risk
in a solvency balance sheet is not "how wrong is the ultimate" but "how far can
next year's re-estimate move", the claims development result of Merz & Wuthrich
(2008). Both routes to it are exposed and are meant to be compared:

    entry.one_year_cdr()        analytic msep, per accident year and in total
    entry.cdr_distribution()    the full CDR distribution by re-reserving

``cdr_distribution()`` takes a ``generator=``, because "the one-year CDR" is
not one method: what emerges next year can be drawn from Mack's conditional
moments or from an England-Verrall ODP residual bootstrap, and the reserve is
then re-estimated the same way either side. ``kernels.cdr.cdr_methods()`` is
the list. ``one_year_cdr()`` deliberately takes no such argument - the
Merz-Wuthrich closed form linearizes the factor update around Mack's moments
and has no counterpart for another model.

The algorithms themselves live in ``kernels/mack.py``, ``kernels/cdr.py`` and
``kernels/odp_bootstrap.py`` - this entry only wires them to the gallery
contract.

On the milestone-6 board this entry is a CRPS member and a permanent ELPD N/A:
it subclasses ``PredictsHeldout`` (its one-step-ahead draws are
``kernels.mack.draw_next_cells``, the same core the CDR simulation uses) but
deliberately NOT ``ScoresHeldout``. Mack states two conditional moments, not an
observation model, so there is no predictive density to claim - the gamma step
law is an assumption of the simulation, and claiming it as a density is an
explicitly reserved decision (``forecast.MODEL_ABSENCE_REASONS
["no_predictive_density"]`` names this entry).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import numpy as np

from ibnr.gallery.entry import GalleryEntry, PredictsHeldout
from ibnr.gallery.registry import register
from ibnr.kernels.cdr import CDRResult, DiagonalGenerator, one_year_cdr, simulate_one_year_cdr
from ibnr.kernels.contract import cohort_grid, realized_values
from ibnr.kernels.holdout import CellIndex
from ibnr.kernels.mack import MackFit, draw_next_cells, fit_mack_grid, simulate_ultimates
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.kernels.rng import cohort_stream
from ibnr.triangle.core import Triangle


@register
class Mack(GalleryEntry, PredictsHeldout):
    """Distribution-free chain ladder with run-off and one-year uncertainty.

    Fitted state is ``fit_`` (development factors, Mack's ``sigma^2`` per step,
    the volumes behind each factor, and the observed triangle they came from)
    plus ``contract_``, the ``cohort_grid`` dict the fit was built from - kept
    so held-out cells can be indexed against this fit's own cohort identity.
    Everything else - ultimates, reserves, run-off MSEP, the one-year CDR, the
    held-out draws - is derived on demand.
    """

    name = "mack"
    family = "deterministic"
    #: the fit's ``cum`` is cumulative loss, so the one-step draws are too; on a
    #: cumulative triangle ``predict_at`` passes them through unchanged.
    heldout_draw_scale = "cumulative"

    def __init__(self) -> None:
        self.fit_: MackFit | None = None
        self.contract_: dict | None = None
        self._loss_field: str | None = None
        self.heldout_n_draws: int = 10_000
        self.heldout_process: str = "gamma"
        self.heldout_parameter_risk: bool = True

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        as_of: dt.date | str | None = None,
        sigma_rule: str = "mack",
        heldout_n_draws: int = 10_000,
        heldout_process: str = "gamma",
        heldout_parameter_risk: bool = True,
    ) -> Mack:
        """Estimate the chain ladder on one cohort's run-off triangle.

        ``sigma_rule`` picks how the last development step's variance is
        estimated (``"mack"``, Mack's shrinking rule, or ``"log_linear"``, the
        extrapolation chainladder-python and R's ``est.sigma="log-linear"``
        default to). It is the only modelling choice here; the factors
        themselves are volume-weighted, which is what Mack's and
        Merz-Wuthrich's variance formulas assume.

        The ``heldout_*`` knobs configure :meth:`predict_at`'s one-step draws
        (``_draws_native``'s signature is fixed by the ABC, so they ride on the
        entry): draw count, step law among ``kernels.mack.PROCESS_LAWS``, and
        whether the factors' estimation error is drawn (shared across cells,
        which is what correlates the held-out diagonal).
        """
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # Same two steps as kernels.mack.fit_mack, but the grid survives as the
        # contract: index_into needs its identity to refuse cells that are not
        # this fit's, and its w/d to refuse cells the fit was trained on.
        #
        # BUILD FIRST, ASSIGN AFTER EVERYTHING SUCCEEDED - fit() must be atomic.
        # cohort_grid accepts cohorts that fit_mack_grid then refuses (e.g. a
        # zero-volume step), and assigning contract_ before that raise leaves a
        # TORN entry: the NEW cohort's contract over the OLD cohort's factors.
        # index_into checks identity against the contract, so predict_at on the
        # new cohort's cells would pass every check and return plausible draws
        # from the wrong company's factors. A failed refit instead leaves the
        # previous fitted state fully intact and consistent.
        contract = cohort_grid(train, loss_field=loss_field)
        fitted = fit_mack_grid(contract, sigma_rule=sigma_rule)
        self._loss_field = loss_field
        self.heldout_n_draws = heldout_n_draws
        self.heldout_process = heldout_process
        self.heldout_parameter_risk = heldout_parameter_risk
        self.contract_ = contract
        self.fit_ = fitted
        return self

    def _fitted(self) -> MackFit:
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        return self.fit_

    def cohorts(self) -> list[dict]:
        """This fit's one cohort - the segment identity its contract was built
        from (see :meth:`GalleryEntry.cohorts`)."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        return [dict(self.contract_["segment"])]

    def predict(
        self,
        segment: Mapping | None = None,
        *,
        n_draws: int = 10_000,
        seed: int | None = None,
        process: str = "gamma",
        parameter_risk: bool = True,
    ) -> PredictiveDistribution:
        """Simulated FULL run-off ultimates, per origin plus a total column.

        Mack's model fixes two conditional moments and nothing else, so the
        ``process`` law is an assumption of this wrapper rather than of the
        model - see the card. ``parameter_risk`` draws the factors once per
        draw, which is what correlates the accident years.
        """
        # a single-cohort fit: accepts None or its own key, refuses anything else
        self.cohort_index(segment)
        return simulate_ultimates(
            self._fitted(),
            n_draws=n_draws,
            seed=cohort_stream(
                seed, label="predict", cohorts=self.cohorts(), field=self._loss_field
            ),
            process=process,
            parameter_risk=parameter_risk,
        )

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        """``(n_draws, n_cells)`` one-step-ahead cumulative draws at the cells.

        Glue over :func:`ibnr.kernels.mack.draw_next_cells`, which shares its
        core with the CDR simulation's next diagonal - so the board's CRPS and
        ``cdr_distribution()`` carry the identical noise assumption. The knobs
        are the ``heldout_*`` fit parameters; ``gamma`` (the default) needs a
        positive open diagonal, and ``require_positive_open_diagonals`` inside
        the kernel is what refuses the fits where it is not well-posed.
        """
        return draw_next_cells(
            self._fitted(),
            cells,
            rng=rng,
            n_draws=self.heldout_n_draws,
            process=self.heldout_process,
            parameter_risk=self.heldout_parameter_risk,
        )

    def one_year_cdr(self) -> CDRResult:
        """Merz-Wuthrich (2008) analytic one-year CDR msep, with Mack's run-off
        msep alongside for the ratio that matters."""
        return one_year_cdr(self._fitted())

    def cdr_distribution(
        self,
        *,
        n_draws: int | None = None,
        seed: int | None = None,
        generator: DiagonalGenerator | str | None = None,
        process: str | None = None,
        parameter_risk: bool | None = None,
    ) -> PredictiveDistribution:
        """The one-year CDR distribution by re-reserving ("actuary in the box").

        Same question as ``one_year_cdr()``, answered by simulation instead of
        a closed form, so it also yields quantiles - which is what a one-year
        risk capital figure actually needs. A positive draw is a release.

        ``generator`` picks what produces next year's diagonal;
        ``kernels.cdr.cdr_methods()`` lists the options and what each requires::

            entry.cdr_distribution()                            # Mack moments
            entry.cdr_distribution(generator="odp_bootstrap")   # England-Verrall

        Whichever generator is used, the reserve is re-estimated with the same
        volume-weighted chain ladder, so the two answers are comparable. Every
        argument but ``seed`` is passed straight through to
        :func:`ibnr.kernels.cdr.simulate_one_year_cdr`, including its refusal of
        ``process``/``parameter_risk`` beside an explicit ``generator`` - they
        are ``MackDiagonal``'s knobs and would otherwise be inert. ``seed`` is
        first turned into this cohort's own stream
        (:func:`ibnr.kernels.rng.cohort_stream`), so a study that fits many
        cohorts and passes one seed does not have every cohort's claims
        development result reading the same random numbers. The kernel
        function's own behavior for a plain integer seed is unchanged, for
        anyone calling it directly.

        ``n_draws=None`` means "this generator's own count", exactly as on the
        kernel function, and it is the DEFAULT here rather than a literal
        20_000. A hardcoded default would have defeated the negotiation from
        this side: it reaches the generator as an explicit request, so
        ``entry.cdr_distribution(generator=GalleryDiagonal(...))`` would have
        refused every entry whose draw count is not 20_000 - which is all of
        them. The two simulating generators still resolve None to 20_000, so no
        existing call changes.
        """
        return simulate_one_year_cdr(
            self._fitted(),
            n_draws=n_draws,
            # A distinct label keeps this cohort's CDR draws on their own
            # stream, separate from the same fit's run-off and held-out draws.
            seed=cohort_stream(
                seed, label="cdr_distribution", cohorts=self.cohorts(), field=self._loss_field
            ),
            generator=generator,
            process=process,
            parameter_risk=parameter_risk,
        )

    def summary(self):
        """Mack's per-origin table (latest, ultimate, IBNR, run-off S.E.)."""
        return self._fitted().summary()

    def realized_ultimates(
        self, full_triangle: Triangle, segment: Mapping | None = None
    ) -> np.ndarray:
        """Outcomes aligned to ``predict()``'s targets - per origin, then the
        total - read off the FULL (unsliced) triangle at its final dev lag."""
        # a single-cohort fit: accepts None or its own key, refuses anything else
        self.cohort_index(segment)
        fit = self._fitted()
        realized = realized_values(
            full_triangle,
            loss_field=self._loss_field,
            dev_lag=fit.n_d * fit.dev_grain_months,
            origins=fit.origin_periods,
        )
        # NaN propagates into the total on purpose: a partially unemerged
        # cohort has no observed total to score against.
        return np.append(realized, realized.sum())
