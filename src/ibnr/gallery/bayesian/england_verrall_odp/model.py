"""England & Verrall Bayesian over-dispersed Poisson (ODP) gallery entry.

The Bayesian form of the ODP cross-classified model whose MLE reproduces
chain-ladder reserves (England & Verrall 2002, sections 3.2 and 7.11), fit
with the Stan reference sampler (``model.stan``). Consumes the incremental
``kernels.contract.odp_stan_data`` dict. The dispersion ``phi`` is a plug-in
nuisance estimated from the GLM Pearson chi-square, exactly as England &
Verrall treat the scale parameter. NumPyro/PyMC ports arrive with milestone
5; the ``backend`` argument already reserves the seam."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr.gallery.bayesian._toolchain import ensure_stan_toolchain
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.contract import odp_stan_data, realized_values
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

STAN_FILE = Path(__file__).parent / "model.stan"

#: keys of the standardized contract dict that form the Stan data block
STAN_DATA_KEYS = ("len_data", "n_w", "n_d", "w", "d", "inc_loss", "logprem", "phi")

#: posterior backends this entry can dispatch to (ports land in milestone 5)
BACKENDS = ("stan",)


def pooled(idata, name: str) -> np.ndarray:
    """Draws for a posterior variable pooled across chains: an idata
    ``posterior[name]`` of shape (chain, draw, *dims) -> (chain*draw, *dims)."""
    arr = np.asarray(idata.posterior[name].values)
    return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


def odp_mle_fitted(w: np.ndarray, d: np.ndarray, inc: np.ndarray, n_w: int, n_d: int):
    """Poisson cross-classified MLE fitted means via iterative proportional
    fitting: m[w,d] = a[w] * b[d] matching the observed row and column totals
    on the triangle's support. These are exactly the chain-ladder fitted
    incrementals (Hachemeister-Stanard / Renshaw-Verrall). Returns the full
    (n_w, n_d) rectangle — future cells hold the ODP/chain-ladder point
    forecasts of the unobserved increments."""
    obs = np.zeros((n_w, n_d))
    mask = np.zeros((n_w, n_d))
    obs[w - 1, d - 1] = inc
    mask[w - 1, d - 1] = 1.0
    row_tot, col_tot = obs.sum(axis=1), obs.sum(axis=0)
    a, b = np.maximum(row_tot, 1e-12), np.ones(n_d)
    for _ in range(500):
        fit_col = (a[:, None] * b[None, :] * mask).sum(axis=0)
        b *= np.divide(col_tot, fit_col, out=np.ones(n_d), where=fit_col > 0)
        fit_row = (a[:, None] * b[None, :] * mask).sum(axis=1)
        a_new = a * np.divide(row_tot, fit_row, out=np.ones(n_w), where=fit_row > 0)
        done = np.allclose(a_new, a, rtol=1e-12)
        a = a_new
        if done:
            break
    return a[:, None] * b[None, :]


def pearson_phi(w: np.ndarray, d: np.ndarray, inc: np.ndarray, n_w: int, n_d: int) -> float:
    """Plug-in ODP dispersion: Pearson chi-square over residual dof,
    phi = sum((x - m)^2 / m) / (n - p) with p = n_w + n_d - 1."""
    m = odp_mle_fitted(w, d, inc, n_w, n_d)[w - 1, d - 1]
    if (m <= 0).any():
        raise ValueError("ODP MLE produced non-positive fitted means; degenerate triangle")
    n, p = len(inc), n_w + n_d - 1
    if n <= p:
        raise ValueError(f"triangle has {n} cells but the ODP model has {p} parameters")
    return float(((inc - m) ** 2 / m).sum() / (n - p))


@register
class EnglandVerrallODP(GalleryEntry):
    name = "england_verrall_odp"
    family = "bayesian"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.idata_ = None
        self.fit_ = None  # cmdstan fit object
        self.backend_: str | None = None
        self._loss_field: str | None = None

    def fit(
        self,
        triangle: Triangle,
        *,
        # the ODP/chain-ladder equivalence is a paid-loss result
        loss_field: str = "paid_loss",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        backend: str = "stan",
        chains: int = 4,
        iter_warmup: int = 1000,
        iter_sampling: int = 2500,
        seed: int | None = None,
        target_accept: float = 0.8,
        show_progress: bool = False,
    ) -> EnglandVerrallODP:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = odp_stan_data(train, loss_field=loss_field, premium_field=premium_field)
        c = self.contract_
        c["phi"] = pearson_phi(c["w"], c["d"], c["inc_loss"], c["n_w"], c["n_d"])
        self._loss_field = loss_field
        self.backend_ = backend
        self.idata_ = self._sample_stan(
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            show_progress=show_progress,
        )
        return self

    def _sample_stan(
        self, *, chains, iter_warmup, iter_sampling, seed, target_accept, show_progress
    ):
        import time

        import arviz as az
        from cmdstanpy import CmdStanModel

        ensure_stan_toolchain()
        model = CmdStanModel(stan_file=str(STAN_FILE))
        t0 = time.perf_counter()
        self.fit_ = model.sample(
            data={k: self.contract_[k] for k in STAN_DATA_KEYS},
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
        """Predictive distribution of ultimates (losses at the last dev period)
        by origin year plus their total: each origin's observed paid-to-date
        plus simulated future increments, X[w,d] ~ phi * Poisson(m[w,d] / phi)
        for d beyond the origin's latest observed lag — the od-Poisson process
        draw England & Verrall obtain by imputing future cells (7.11.6), and
        the same process distribution as the ODP bootstrap baselines."""
        if self.idata_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_w, n_d, phi = c["n_w"], c["n_d"], c["phi"]

        alpha = pooled(self.idata_, "alpha")  # (draws, n_w)
        beta = pooled(self.idata_, "beta")  # (draws, n_d)
        const = pooled(self.idata_, "c")  # (draws,)
        n_draws = const.shape[0]
        logprem_origin = np.log(c["premium"])  # per origin

        rng = np.random.default_rng(seed)
        ults = np.tile(c["paid_to_date"], (n_draws, 1)).astype(float)
        for j in range(n_w):
            for dev in range(int(c["latest_d"][j]) + 1, n_d + 1):
                mu = np.exp(logprem_origin[j] + const + alpha[:, j] + beta[:, dev - 1])
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
        """Outcomes aligned to predict()'s targets (per origin + total), taken
        from the full triangle at the final development lag."""
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
        """Convergence diagnostics from the fitted posterior: max R-hat, min
        bulk/tail ESS, divergence count/fraction, and wall-clock sampling
        runtime."""
        import arviz as az

        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        if var_names is None:
            var_names = ["c", "r_alpha", "r_beta"]
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
