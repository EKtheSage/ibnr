"""Meyers Changing Settlement Rate (CSR) gallery entry.

The paid-loss counterpart of ``meyers_ccl``: a cross-classified lognormal
with a settlement-rate trend term, fit with the Stan reference sampler
(``model.stan``). Consumes the identical ``kernels.contract`` data dict as
the rest of the Meyers family (the ``prev_idx`` entry is simply unused —
CSR has no across-origin correlation term). NumPyro/PyMC ports arrive with
milestone 5; the ``backend`` argument already reserves the seam."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr.gallery.bayesian._toolchain import ensure_stan_toolchain
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.contract import realized_values, stan_data
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

STAN_FILE = Path(__file__).parent / "model.stan"

#: keys of the standardized contract dict that form the Stan data block
STAN_DATA_KEYS = ("len_data", "n_w", "n_d", "w", "d", "logprem", "logloss")

#: posterior backends this entry can dispatch to (ports land in milestone 5)
BACKENDS = ("stan",)


def pooled(idata, name: str) -> np.ndarray:
    """Draws for a posterior variable pooled across chains: an idata
    ``posterior[name]`` of shape (chain, draw, *dims) -> (chain*draw, *dims)."""
    arr = np.asarray(idata.posterior[name].values)
    return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


@register
class MeyersCSR(GalleryEntry):
    name = "meyers_csr"
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
        # CSR is the monograph's PAID model
        loss_field: str = "paid_loss",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        backend: str = "stan",
        chains: int = 4,
        iter_warmup: int = 1000,
        iter_sampling: int = 2500,
        seed: int | None = None,
        # 0.9 default: the gamma x beta interaction makes CSR's centered
        # geometry harder than CCL's (Meyers ran adapt_delta = 0.9999)
        target_accept: float = 0.9,
        parallel_chains: int = 1,
        max_treedepth: int | None = None,
        show_progress: bool = False,
    ) -> MeyersCSR:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        # as_of slices the triangle to the training diagonal (backtest cutoff);
        # None trains on the whole triangle as given.
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # stan_data builds the shared Meyers contract dict (the Stan `data`
        # block per CLAUDE.md decision 3). CSR consumes logloss = log paid,
        # logprem = log net earned premium, and the (w, d) index arrays.
        self.contract_ = stan_data(train, loss_field=loss_field, premium_field=premium_field)
        self._loss_field = loss_field
        self.backend_ = backend
        self.idata_ = self._sample_stan(
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            parallel_chains=parallel_chains,
            max_treedepth=max_treedepth,
            show_progress=show_progress,
        )
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
        *,
        chains,
        iter_warmup,
        iter_sampling,
        seed,
        target_accept,
        parallel_chains,
        max_treedepth,
        show_progress,
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
        idata = az.from_cmdstanpy(self.fit_, log_likelihood="log_lik")
        idata.attrs["runtime_s"] = runtime_s
        idata.attrs["backend"] = "stan"
        return idata

    def predict(self, seed: int | None = None) -> PredictiveDistribution:
        """Predictive distribution of ultimates (losses at the last dev period)
        by origin year plus their total, via the monograph's simulation: origin 1
        is its observed C[1, n_d]; for w >= 2 the origins are independent,
        C[w, n_d] ~ lognormal(logprem[w] + logelr + alpha[w], sig[n_d])
        (beta[n_d] = 0, so the speedup term drops out of the ultimate).
        """
        if self.idata_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_w, n_d = c["n_w"], c["n_d"]

        # Posterior draws of the cross-classified parameters (see model.stan):
        #   alpha[w]  accident-year level offset (alpha[1] = 0 anchors the level)
        #   logelr    log expected loss ratio; e^logelr is the Cape Cod ELR
        #   sig[d]    lognormal scale per dev lag, decreasing in d (more settled
        #             claims -> less process variance)
        alpha = pooled(self.idata_, "alpha")  # (draws, n_w)
        logelr = pooled(self.idata_, "logelr")  # (draws,)
        sig = pooled(self.idata_, "sig")  # (draws, n_d)
        n_draws = logelr.shape[0]
        # Ultimate = last dev lag n_d; use sig at n_d. The settlement-rate term
        # beta[d] * (1 - gamma)^(w-1) vanishes because beta[n_d] = 0, so the
        # speedup/slowdown drift does NOT enter the ultimate (why CSR ultimates
        # coincide with the plain CRC model's — card.md "Predictive distribution").
        sig_last = sig[:, n_d - 1]
        logprem = np.log(c["premium"])  # per origin, shape (n_w,)

        # Origin 1 is fully developed at the cutoff: its observed C[1, n_d] is
        # the deterministic anchor (zero predictive variance), per CSR.R step 1.
        first_mask = (c["w"] == 1) & (c["d"] == n_d)
        if not first_mask.any():
            raise ValueError(
                "the first origin year is not fully developed in the training data; "
                "the CSR simulation needs the observed C[1, n_d] anchor"
            )
        c1_ult = float(c["loss"][first_mask][0])

        rng = np.random.default_rng(seed)
        ults = np.empty((n_draws, n_w))  # (draws, n_w): ultimate per origin
        ults[:, 0] = c1_ult
        # Origins w >= 2 are conditionally independent (CSR has no across-origin
        # rho, unlike CCL): each C[w, n_d] ~ lognormal(mu, sig[n_d]) with
        # mu = logprem[w] + logelr + alpha[w]. This is CSR.R step 2.
        for j in range(1, n_w):
            mu = logprem[j] + logelr + alpha[:, j]  # (draws,)
            ults[:, j] = np.exp(rng.normal(mu, sig_last))

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
        runtime. ``var_names`` defaults to the sampled (non-deterministic)
        core parameters."""
        import arviz as az

        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        if var_names is None:
            # Sampled (non-deterministic) core parameters: logelr (log ELR),
            # r_alpha (free AY offsets), r_beta (dev profile), a_ig (the
            # inverse-gamma variance components building sig2), and gamma (the
            # settlement-rate trend — CSR's signature parameter).
            var_names = ["logelr", "r_alpha", "r_beta", "a_ig", "gamma"]
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
