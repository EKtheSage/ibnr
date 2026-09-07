"""Bayesian Clark growth-curve gallery entry (Cape Cod form).

Clark's (2003) over-dispersed Poisson growth-curve likelihood with the
Cape Cod ultimate structure, sampled with Stan. The dispersion ``phi`` is
the Pearson scale from the statistical ``clark`` entry's MLE fit of the
identical model - plug-in, mirroring ``england_verrall_odp``. The LDF
(free-ultimates) variant lives in the statistical entry; scaffold this model
if you want a Bayesian LDF version. NumPyro/PyMC ports arrive with milestone
5; the ``backend`` argument already reserves the seam."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr.gallery.bayesian._toolchain import ensure_stan_toolchain
from ibnr.gallery.bayesian.clark_growth_curve import scorer
from ibnr.gallery.entry import GalleryEntry, PredictsHeldout
from ibnr.gallery.registry import register
from ibnr.gallery.statistical.clark.model import Clark, age_interval, growth
from ibnr.kernels.contract import odp_stan_data, realized_values
from ibnr.kernels.densities import odp_draw
from ibnr.kernels.holdout import CellIndex
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.kernels.rng import cohort_stream
from ibnr.triangle.core import Triangle

STAN_FILE = Path(__file__).parent / "model.stan"

#: Clark's two growth curves G(x) = fraction of ultimate emerged by age x
#: months. loglogistic: G = 1 / (1 + (theta/x)^omega); weibull:
#: G = 1 - exp(-(x/theta)^omega). Integer codes are passed to Stan as data so
#: the port cannot drift on which curve it fits (Clark 2003, section 2).
CURVE_CODES = {"loglogistic": 1, "weibull": 2}

#: posterior backends this entry can dispatch to; all three target the same
#: posterior, which ``kernels.parity`` gates before any convergence claim
BACKENDS = ("stan", "numpyro", "pymc")


def pooled(idata, name: str) -> np.ndarray:
    """Draws for a posterior variable pooled across chains: an idata
    ``posterior[name]`` of shape (chain, draw, *dims) -> (chain*draw, *dims)."""
    arr = np.asarray(idata.posterior[name].values)
    return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


@register
class ClarkGrowthCurve(GalleryEntry, PredictsHeldout):
    name = "clark_growth_curve"
    family = "bayesian"

    #: Clark models INCREMENTAL emergence while the Schedule P triangles are
    #: cumulative, so ``predict_at`` adds each cell's training-diagonal anchor.
    #: Declared rather than assumed: an undeclared increment draw is wrong by
    #: that whole anchor while staying finite and plausible.
    #:
    #: Draws only - no ``ScoresHeldout`` and no ``heldout_measure``, on
    #: principle: the ODP quasi-likelihood is not a normalized density on any
    #: scale (``kernels/densities.py``, :ref:`odp-not-a-density`), so this
    #: entry is CRPS-scored and permanently ELPD-ineligible.
    heldout_draw_scale = "incremental"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.idata_ = None
        self.fit_ = None  # cmdstan fit object
        self.backend_: str | None = None
        self.mle_: Clark | None = None
        self._loss_field: str | None = None
        self._curve: str | None = None

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        growth_curve: str = "loglogistic",
        backend: str = "stan",
        chains: int = 4,
        iter_warmup: int = 1000,
        iter_sampling: int = 2500,
        seed: int | None = None,
        target_accept: float = 0.8,
        parallel_chains: int = 1,
        max_treedepth: int | None = None,
        nuts_sampler: str = "pymc",
        show_progress: bool = False,
    ) -> ClarkGrowthCurve:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        # nuts_sampler selects the NUTS IMPLEMENTATION over the PyMC graph; it
        # is meaningless for the other backends, so asking for one there is an
        # error rather than a silently ignored argument.
        if backend != "pymc" and nuts_sampler != "pymc":
            raise ValueError(
                f"nuts_sampler={nuts_sampler!r} is a pymc-backend control; "
                f"the {backend!r} backend does not take it"
            )
        # parallel_chains / max_treedepth are cmdstan-level controls the retro
        # harness escalates on. Validated BEFORE any data prep or MLE fit: a
        # port that accepted them silently would report an escalated fit that
        # never ran.
        if backend != "stan" and (parallel_chains != 1 or max_treedepth is not None):
            raise ValueError(
                "parallel_chains / max_treedepth are stan-backend controls; "
                f"the {backend!r} port does not take them"
            )
        if growth_curve not in CURVE_CODES:
            raise ValueError(f"growth_curve must be one of {tuple(CURVE_CODES)}")
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # odp_stan_data builds the INCREMENTAL-cell contract (Clark models
        # incremental emergence, not cumulative); premium is required for the
        # Cape Cod ELR and negative increments are rejected upstream.
        #
        # BUILD FIRST, ASSIGN AFTER THE SAMPLER SUCCEEDED - fit() must be
        # atomic. The MLE twin and the sampler both legitimately refuse
        # cohorts the contract accepted, and assigning contract_/mle_ before
        # those steps leaves a failed refit TORN: the new cohort's contract
        # over the old cohort's posterior, which index_into then accepts. See
        # meyers_ccl/mack.
        contract = odp_stan_data(train, loss_field=loss_field, premium_field=premium_field)
        if "premium" not in contract:
            raise ValueError("clark_growth_curve needs a premium_field (Cape Cod ultimates)")

        # plug-in dispersion: the Pearson scale from the MLE of the identical
        # Cape Cod model (the statistical clark entry)
        mle = Clark().fit(
            train,
            loss_field=loss_field,
            premium_field=premium_field,
            growth_curve=growth_curve,
            method="cape_cod",
        )
        c = contract
        c["phi"] = mle.params_["phi"]

        # Assemble the Stan `data` block. Ages are DATA, not model logic (so a
        # port cannot drift on the convention): each incremental cell at dev d
        # spans the mid-period age interval (age_lo, age_hi], from the one
        # shared `age_interval` (Clark 2003 measures ages from the origin's
        # *average* accident date - see its docstring).
        step = c["dev_grain_months"]
        age_lo, age_hi = age_interval(c["d"], step)
        stan_data = {
            "len_data": c["len_data"],
            "n_w": c["n_w"],
            "w": c["w"],
            "age_lo": age_lo,
            "age_hi": age_hi,
            "inc_loss": c["inc_loss"],
            "logprem_w": np.log(c["premium"]),  # (n_w,): log net earned premium
            "phi": c["phi"],  # plug-in Pearson dispersion from the MLE twin
            "curve": CURVE_CODES[growth_curve],
            # theta prior median tracks the grain (4 dev periods): keeps the
            # scale prior comparable across annual/quarterly triangles.
            "theta_prior_median": 4.0 * step,
        }
        sampler = {
            "stan": self._sample_stan,
            "numpyro": self._sample_numpyro,
            "pymc": self._sample_pymc,
        }[backend]
        extra = (
            {"parallel_chains": parallel_chains, "max_treedepth": max_treedepth}
            if backend == "stan"
            else {}
        )
        # nuts_sampler is the pymc-side analogue and has to reach _sample_pymc:
        # accepting it here and dropping it would sample with that method's own
        # "pymc" default while the caller believed a foreign NUTS ran.
        if backend == "pymc":
            extra["nuts_sampler"] = nuts_sampler
        # All three return (comparable arviz.InferenceData, raw cmdstan fit or
        # None), so predict(), convergence() and parity are backend-agnostic -
        # and fit() stays the single writer of entry state.
        idata, raw_fit = sampler(
            stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            show_progress=show_progress,
            **extra,
        )
        self.contract_ = contract
        self._loss_field = loss_field
        self._curve = growth_curve
        self.backend_ = backend
        self.mle_ = mle
        self.stan_data_ = stan_data  # the single data block every backend consumes
        self.idata_ = idata
        # None for the ports, which also clears a stale cmdstan object left by
        # an earlier stan-backend fit of a different cohort
        self.fit_ = raw_fit
        return self

    @classmethod
    def precompile(cls) -> None:
        """Compile the Stan program ahead of use (container build, or before a
        worker pool spawns, so concurrent first-fits never race the compiler)."""
        from cmdstanpy import CmdStanModel

        ensure_stan_toolchain()
        CmdStanModel(stan_file=str(STAN_FILE))

    def _sample_stan(
        self,
        stan_data,
        *,
        chains,
        iter_warmup,
        iter_sampling,
        seed,
        target_accept,
        show_progress,
        parallel_chains=1,
        max_treedepth=None,
    ):
        """Reference backend: cmdstanpy NUTS on ``model.stan``.

        Ground truth for parity - the ports are checked against this posterior
        before any convergence or runtime claim is made.
        """
        import time

        import arviz as az
        from cmdstanpy import CmdStanModel

        ensure_stan_toolchain()
        model = CmdStanModel(stan_file=str(STAN_FILE))
        t0 = time.perf_counter()
        fit = model.sample(
            data=stan_data,
            chains=chains,
            # 1 = sequential, the fair single-core runtime convention vs future
            # ports; the retro harness raises it in late escalation stages
            parallel_chains=parallel_chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            adapt_delta=target_accept,
            max_treedepth=max_treedepth,
            show_progress=show_progress,
        )
        runtime_s = time.perf_counter() - t0
        idata = az.from_cmdstanpy(fit, log_likelihood="log_lik")
        idata.attrs["runtime_s"] = runtime_s
        idata.attrs["backend"] = "stan"
        return idata, fit

    def _sample_numpyro(
        self, stan_data, *, chains, iter_warmup, iter_sampling, seed, target_accept, show_progress
    ):
        """NumPyro (JAX) port. Same parameterization, same data block.

        The growth curve needs a gradient-safe zero-age branch; see
        ``model_numpyro`` for why a bare ``where`` silently NaNs the gradient
        on the loglogistic curve.
        """
        from ibnr.gallery.bayesian.clark_growth_curve import model_numpyro

        return model_numpyro.sample(
            stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            progress_bar=show_progress,
        ), None

    def _sample_pymc(
        self,
        stan_data,
        *,
        chains,
        iter_warmup,
        iter_sampling,
        seed,
        target_accept,
        show_progress,
        nuts_sampler="pymc",
    ):
        """PyMC port. Same parameterization, same data block."""
        from ibnr.gallery.bayesian.clark_growth_curve import model_pymc

        return model_pymc.sample(
            stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            progressbar=show_progress,
        ), None

    def cohorts(self) -> list[dict]:
        """This fit's one cohort - the segment identity its contract was built
        from (see :meth:`GalleryEntry.cohorts`)."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        return [dict(self.contract_["segment"])]

    def predict(
        self, segment: Mapping | None = None, seed: int | None = None
    ) -> PredictiveDistribution:
        """Ultimates per origin + total: paid-to-date plus simulated future
        increments through the final age. Parameter risk from the posterior
        draws of (logelr, omega, theta); process risk as scaled-Poisson ODP
        draws - the same decomposition as the MLE entry, with the posterior
        replacing the MVN delta method."""
        # a single-cohort fit: accepts None or its own key, refuses anything else
        self.cohort_index(segment)
        if self.idata_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_w, n_d, step, phi = c["n_w"], c["n_d"], c["dev_grain_months"], c["phi"]
        curve = self._curve

        # Posterior draws: logelr (log ELR), omega (curve shape), theta (curve
        # scale in months). Parameter risk comes from these draws.
        logelr = pooled(self.idata_, "logelr")  # (draws,)
        om = pooled(self.idata_, "omega")  # (draws,)
        th = pooled(self.idata_, "theta")  # (draws,)
        n_draws = logelr.shape[0]
        # Cape Cod expected ultimate per origin: elr * premium. One ELR shared
        # across origins (Clark's recommendation for triangle-sized data).
        elr_prem = np.exp(logelr[:, None] + np.log(c["premium"])[None, :])  # (draws, n_w)

        rng = np.random.default_rng(
            cohort_stream(seed, label="predict", cohorts=self.cohorts(), field=self._loss_field)
        )
        # Anchor each origin at its observed paid-to-date, then add simulated
        # future increments cell-by-cell (fully-developed origins get no cells).
        ults = np.tile(c["paid_to_date"], (n_draws, 1)).astype(float)  # (draws, n_w)
        for j in range(n_w):
            # Only unobserved future dev lags (beyond the latest seen for this
            # origin) up to the triangle's final age n_d - no tail extrapolation.
            for dev in range(int(c["latest_d"][j]) + 1, n_d + 1):
                lo, hi = age_interval(dev, step)
                # Expected fraction emerging in (lo, hi]: G(hi) - G(lo).
                ginc = growth(hi, om, th, curve) - growth(lo, om, th, curve)  # (draws,)
                # Expected increment E[X] = elr*premium * (G(hi) - G(lo)).
                mu = np.maximum(elr_prem[:, j] * ginc, 1e-12)  # (draws,)
                # Process risk as scaled-Poisson ODP: Var[X] = phi * E[X], drawn
                # as phi * Poisson(mu/phi) (Clark 2003 ODP; same as the MLE twin,
                # with the posterior replacing its delta-method MVN).
                ults[:, j] += odp_draw(rng, mu, phi)

        targets = pd.DataFrame(
            {
                "label": [str(o.year) for o in c["origin_periods"]],
                "origin_period": c["origin_periods"],
                "premium": c["premium"],
            }
        )
        pred = PredictiveDistribution(samples=ults, targets=targets)
        return pred.with_total()

    def realized_ultimates(
        self, full_triangle: Triangle, segment: Mapping | None = None
    ) -> np.ndarray:
        """Outcomes aligned to predict()'s targets (per origin + total)."""
        # a single-cohort fit: accepts None or its own key, refuses anything else
        self.cohort_index(segment)
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        per_origin = realized_values(
            full_triangle,
            loss_field=self._loss_field,
            dev_lag=c["n_d"] * c["dev_grain_months"],
            origins=c["origin_periods"],
        )
        return np.append(per_origin, per_origin.sum())

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        """``(n_draws, n_cells)`` incremental draws. See ``scorer.draw_cells``.

        Three lines of glue: the arithmetic lives beside ``model.stan`` where
        it can be read against it. The growth-curve name is fitted state
        (``_curve``), not part of the contract, so it is threaded here.
        """
        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        return scorer.draw_cells(
            self.contract_, self._posterior(), cells, curve=self._curve, rng=rng
        )

    def _posterior(self) -> dict[str, np.ndarray]:
        """Pooled draws of the variables the cell-level scorer reads.

        ``posterior``, never ``log_likelihood``: Stan names that group
        ``log_lik`` and this entry's PyMC port attaches via ``pm.Potential``
        so it has no such group at all. The three sampled parameters carry the
        same names in every backend.
        """
        return {name: pooled(self.idata_, name) for name in scorer.REQUIRED_DRAWS}

    def convergence(self, var_names: list[str] | None = None) -> dict:
        """Convergence diagnostics from the fitted posterior."""
        import arviz as az

        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        if var_names is None:
            # The three sampled parameters: logelr (log ELR), omega (growth-curve
            # shape), theta (growth-curve scale). phi is plug-in data, not sampled.
            var_names = ["logelr", "omega", "theta"]
        var_names = [v for v in var_names if v in self.idata_.posterior]
        summ = az.summary(self.idata_, var_names=var_names)
        post = self.idata_.posterior
        n_draws = int(post.sizes["chain"] * post.sizes["draw"])
        diverging = None
        if "sample_stats" in self.idata_ and "diverging" in self.idata_.sample_stats:
            diverging = int(np.asarray(self.idata_.sample_stats["diverging"].values).sum())
        return {
            "backend": self.backend_,
            "runtime_s": float(self.idata_.attrs.get("runtime_s", np.nan)),
            "n_draws": n_draws,
            "max_rhat": float(summ["r_hat"].max()),
            "min_ess_bulk": float(summ["ess_bulk"].min()),
            "min_ess_tail": float(summ["ess_tail"].min()),
            "divergences": diverging,
            "divergence_frac": (None if diverging is None else diverging / n_draws),
        }
