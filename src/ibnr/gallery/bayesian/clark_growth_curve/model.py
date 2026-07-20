"""Bayesian Clark growth-curve gallery entry (Cape Cod form).

Clark's (2003) over-dispersed Poisson growth-curve likelihood with the
Cape Cod ultimate structure, sampled with Stan. The dispersion ``phi`` is
the Pearson scale from the statistical ``clark`` entry's MLE fit of the
identical model — plug-in, mirroring ``england_verrall_odp``. The LDF
(free-ultimates) variant lives in the statistical entry; scaffold this model
if you want a Bayesian LDF version. NumPyro/PyMC ports arrive with milestone
5; the ``backend`` argument already reserves the seam."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr.gallery.bayesian._toolchain import ensure_stan_toolchain
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.gallery.statistical.clark.model import Clark, growth
from ibnr.kernels.contract import odp_stan_data, realized_values
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

STAN_FILE = Path(__file__).parent / "model.stan"

CURVE_CODES = {"loglogistic": 1, "weibull": 2}

#: posterior backends this entry can dispatch to (ports land in milestone 5)
BACKENDS = ("stan",)


def pooled(idata, name: str) -> np.ndarray:
    """Draws for a posterior variable pooled across chains: an idata
    ``posterior[name]`` of shape (chain, draw, *dims) -> (chain*draw, *dims)."""
    arr = np.asarray(idata.posterior[name].values)
    return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


@register
class ClarkGrowthCurve(GalleryEntry):
    name = "clark_growth_curve"
    family = "bayesian"

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
        show_progress: bool = False,
    ) -> ClarkGrowthCurve:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if growth_curve not in CURVE_CODES:
            raise ValueError(f"growth_curve must be one of {tuple(CURVE_CODES)}")
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = odp_stan_data(train, loss_field=loss_field, premium_field=premium_field)
        if "premium" not in self.contract_:
            raise ValueError("clark_growth_curve needs a premium_field (Cape Cod ultimates)")
        self._loss_field = loss_field
        self._curve = growth_curve
        self.backend_ = backend

        # plug-in dispersion: the Pearson scale from the MLE of the identical
        # Cape Cod model (the statistical clark entry)
        self.mle_ = Clark().fit(
            train,
            loss_field=loss_field,
            premium_field=premium_field,
            growth_curve=growth_curve,
            method="cape_cod",
        )
        c = self.contract_
        c["phi"] = self.mle_.params_["phi"]

        step = c["dev_grain_months"]
        stan_data = {
            "len_data": c["len_data"],
            "n_w": c["n_w"],
            "w": c["w"],
            "age_lo": np.maximum(step * (c["d"] - 1) - step / 2, 0.0),
            "age_hi": step * c["d"] - step / 2,
            "inc_loss": c["inc_loss"],
            "logprem_w": np.log(c["premium"]),
            "phi": c["phi"],
            "curve": CURVE_CODES[growth_curve],
            "theta_prior_median": 4.0 * step,
        }
        self.idata_ = self._sample_stan(
            stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            show_progress=show_progress,
        )
        return self

    def _sample_stan(
        self, stan_data, *, chains, iter_warmup, iter_sampling, seed, target_accept, show_progress
    ):
        import time

        import arviz as az
        from cmdstanpy import CmdStanModel

        ensure_stan_toolchain()
        model = CmdStanModel(stan_file=str(STAN_FILE))
        t0 = time.perf_counter()
        self.fit_ = model.sample(
            data=stan_data,
            chains=chains,
            parallel_chains=1,  # sequential: fair single-core runtime vs future ports
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            adapt_delta=target_accept,
            show_progress=show_progress,
        )
        runtime_s = time.perf_counter() - t0
        idata = az.from_cmdstanpy(self.fit_, log_likelihood="log_lik")
        idata.attrs["runtime_s"] = runtime_s
        idata.attrs["backend"] = "stan"
        return idata

    def predict(self, seed: int | None = None) -> PredictiveDistribution:
        """Ultimates per origin + total: paid-to-date plus simulated future
        increments through the final age. Parameter risk from the posterior
        draws of (logelr, omega, theta); process risk as scaled-Poisson ODP
        draws — the same decomposition as the MLE entry, with the posterior
        replacing the MVN delta method."""
        if self.idata_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_w, n_d, step, phi = c["n_w"], c["n_d"], c["dev_grain_months"], c["phi"]
        curve = self._curve

        logelr = pooled(self.idata_, "logelr")
        om = pooled(self.idata_, "omega")
        th = pooled(self.idata_, "theta")
        n_draws = logelr.shape[0]
        elr_prem = np.exp(logelr[:, None] + np.log(c["premium"])[None, :])

        rng = np.random.default_rng(seed)
        ults = np.tile(c["paid_to_date"], (n_draws, 1)).astype(float)
        for j in range(n_w):
            for dev in range(int(c["latest_d"][j]) + 1, n_d + 1):
                lo = max(step * (dev - 1) - step / 2, 0.0)
                hi = step * dev - step / 2
                ginc = growth(hi, om, th, curve) - growth(lo, om, th, curve)
                mu = np.maximum(elr_prem[:, j] * ginc, 1e-12)
                ults[:, j] += phi * rng.poisson(mu / phi)

        targets = pd.DataFrame(
            {
                "label": [str(o.year) for o in c["origin_periods"]],
                "origin_period": c["origin_periods"],
                "premium": c["premium"],
            }
        )
        pred = PredictiveDistribution(samples=ults, targets=targets)
        return pred.with_total()

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to predict()'s targets (per origin + total)."""
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

    def convergence(self, var_names: list[str] | None = None) -> dict:
        """Convergence diagnostics from the fitted posterior."""
        import arviz as az

        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        if var_names is None:
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
