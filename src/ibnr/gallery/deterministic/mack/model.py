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

The algorithms themselves live in ``kernels/mack.py`` and ``kernels/cdr.py`` -
this entry only wires them to the gallery contract.

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

import numpy as np

from ibnr.gallery.entry import GalleryEntry, PredictsHeldout
from ibnr.gallery.registry import register
from ibnr.kernels.cdr import CDRResult, one_year_cdr, simulate_one_year_cdr
from ibnr.kernels.contract import cohort_grid, realized_values
from ibnr.kernels.holdout import CellIndex
from ibnr.kernels.mack import MackFit, draw_next_cells, fit_mack_grid, simulate_ultimates
from ibnr.kernels.predictive import PredictiveDistribution
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
        self._loss_field = loss_field
        self.heldout_n_draws = heldout_n_draws
        self.heldout_process = heldout_process
        self.heldout_parameter_risk = heldout_parameter_risk
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # same two steps as kernels.mack.fit_mack, but the grid survives as the
        # contract: index_into needs its identity to refuse cells that are not
        # this fit's, and its w/d to refuse cells the fit was trained on
        self.contract_ = cohort_grid(train, loss_field=loss_field)
        self.fit_ = fit_mack_grid(self.contract_, sigma_rule=sigma_rule)
        return self

    def _fitted(self) -> MackFit:
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        return self.fit_

    def predict(
        self,
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
        return simulate_ultimates(
            self._fitted(),
            n_draws=n_draws,
            seed=seed,
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
        n_draws: int = 20_000,
        seed: int | None = None,
        process: str = "gamma",
        parameter_risk: bool = True,
    ) -> PredictiveDistribution:
        """The one-year CDR distribution by re-reserving ("actuary in the box").

        Same question as ``one_year_cdr()``, answered by simulation instead of
        a closed form, so it also yields quantiles - which is what a one-year
        risk capital figure actually needs. A positive draw is a release."""
        return simulate_one_year_cdr(
            self._fitted(),
            n_draws=n_draws,
            seed=seed,
            process=process,
            parameter_risk=parameter_risk,
        )

    def summary(self):
        """Mack's per-origin table (latest, ultimate, IBNR, run-off S.E.)."""
        return self._fitted().summary()

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to ``predict()``'s targets - per origin, then the
        total - read off the FULL (unsliced) triangle at its final dev lag."""
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
