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
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.cdr import CDRResult, one_year_cdr, simulate_one_year_cdr
from ibnr.kernels.contract import realized_values
from ibnr.kernels.mack import MackFit, fit_mack, simulate_ultimates
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle


@register
class Mack(GalleryEntry):
    """Distribution-free chain ladder with run-off and one-year uncertainty.

    Fitted state is the whole of ``fit_``: development factors, Mack's
    ``sigma^2`` per step, the volumes behind each factor, and the observed
    triangle they came from. Everything else - ultimates, reserves, run-off
    MSEP, the one-year CDR - is derived from it on demand.
    """

    name = "mack"
    family = "deterministic"

    def __init__(self) -> None:
        self.fit_: MackFit | None = None
        self._loss_field: str | None = None

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        as_of: dt.date | str | None = None,
        sigma_rule: str = "mack",
    ) -> Mack:
        """Estimate the chain ladder on one cohort's run-off triangle.

        ``sigma_rule`` picks how the last development step's variance is
        estimated (``"mack"``, Mack's shrinking rule, or ``"log_linear"``, the
        extrapolation chainladder-python and R's ``est.sigma="log-linear"``
        default to). It is the only modelling choice here; the factors
        themselves are volume-weighted, which is what Mack's and
        Merz-Wuthrich's variance formulas assume.
        """
        self._loss_field = loss_field
        self.fit_ = fit_mack(triangle, loss_field=loss_field, as_of=as_of, sigma_rule=sigma_rule)
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
