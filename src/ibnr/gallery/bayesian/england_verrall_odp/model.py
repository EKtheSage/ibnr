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

#: posterior backends this entry can dispatch to; all three target the same
#: posterior, which ``kernels.parity`` gates before any convergence claim
BACKENDS = ("stan", "numpyro", "pymc")


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
    (n_w, n_d) rectangle - future cells hold the ODP/chain-ladder point
    forecasts of the unobserved increments.

    The multiplicative Poisson MLE m = a[w] * b[d] is the same fit as the
    log-link linear predictor (a = exp(alpha), b = exp(c + beta)); IPF is only
    used here to get the *frequentist* fitted means for the plug-in dispersion
    below, not to fit the Bayesian model. w, d, inc are 1-based-lag ragged
    vectors over the observed cells (len_data,)."""
    # Scatter the ragged observed increments back onto a dense (n_w, n_d) grid;
    # `mask` marks which cells are in-support (the upper-left triangle).
    obs = np.zeros((n_w, n_d))
    mask = np.zeros((n_w, n_d))
    obs[w - 1, d - 1] = inc  # w, d are 1-based lags -> 0-based indices
    mask[w - 1, d - 1] = 1.0
    # Margins the fit must reproduce: origin-row totals and dev-column totals.
    row_tot, col_tot = obs.sum(axis=1), obs.sum(axis=0)  # (n_w,), (n_d,)
    # Seed row effects at the row totals, column effects at 1; alternate scaling
    # a and b until both sets of margins are matched (Poisson MLE fixed point).
    a, b = np.maximum(row_tot, 1e-12), np.ones(n_d)  # a: (n_w,), b: (n_d,)
    for _ in range(500):
        # Rescale column effects so fitted column margins hit col_tot ...
        fit_col = (a[:, None] * b[None, :] * mask).sum(axis=0)  # (n_d,)
        b *= np.divide(col_tot, fit_col, out=np.ones(n_d), where=fit_col > 0)
        # ... then rescale row effects so fitted row margins hit row_tot.
        fit_row = (a[:, None] * b[None, :] * mask).sum(axis=1)  # (n_w,)
        a_new = a * np.divide(row_tot, fit_row, out=np.ones(n_w), where=fit_row > 0)
        done = np.allclose(a_new, a, rtol=1e-12)
        a = a_new
        if done:
            break
    return a[:, None] * b[None, :]  # (n_w, n_d) outer product


def pearson_phi(w: np.ndarray, d: np.ndarray, inc: np.ndarray, n_w: int, n_d: int) -> float:
    """Plug-in ODP dispersion: Pearson chi-square over residual dof,
    phi = sum((x - m)^2 / m) / (n - p) with p = n_w + n_d - 1.

    England & Verrall's quasi-likelihood scale estimate (2002, sec. 3.2): the
    ODP scale is estimated *outside* the model from the GLM Pearson residuals,
    not sampled. p = n_w + n_d - 1 counts the free row/column effects (one
    corner constraint), matching a chain-ladder GLM's parameter count."""
    # Fitted means only at the observed cells (index the dense grid back to the
    # ragged support): m aligned to inc, both (len_data,).
    m = odp_mle_fitted(w, d, inc, n_w, n_d)[w - 1, d - 1]
    # An all-zero dev column (books fully paid before the last lag) fits m = 0
    # exactly at cells where x = 0 - those cells carry no Pearson information
    # and are excluded. m = 0 against x > 0 cannot happen (margins are matched).
    live = m > 0
    if not live.any() or (inc[~live] != 0).any():
        raise ValueError("ODP MLE produced non-positive fitted means; degenerate triangle")
    n, p = int(live.sum()), n_w + n_d - 1  # informative cells vs free parameters
    if n <= p:
        raise ValueError(f"triangle has {n} informative cells but the ODP model has {p} parameters")
    return float(((inc[live] - m[live]) ** 2 / m[live]).sum() / (n - p))


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
        parallel_chains: int = 1,
        max_treedepth: int | None = None,
        nuts_sampler: str = "pymc",
        show_progress: bool = False,
    ) -> EnglandVerrallODP:
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
        # harness escalates on. Validated BEFORE any data prep: a port that
        # accepted them silently would report an escalated fit that never ran.
        if backend != "stan" and (parallel_chains != 1 or max_treedepth is not None):
            raise ValueError(
                "parallel_chains / max_treedepth are stan-backend controls; "
                f"the {backend!r} port does not take them"
            )
        # Backtest slice: keep only cells reported on/before the cutoff diagonal.
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # Standardized incremental ODP contract (w/d lags, inc_loss, logprem,
        # paid_to_date/latest_d anchors) - the Stan `data` block is the contract.
        self.contract_ = odp_stan_data(train, loss_field=loss_field, premium_field=premium_field)
        c = self.contract_
        # Estimate the dispersion once, up front, and inject it into the data
        # dict; Stan consumes phi as data (quasi-likelihood), never samples it.
        c["phi"] = pearson_phi(c["w"], c["d"], c["inc_loss"], c["n_w"], c["n_d"])
        self._loss_field = loss_field
        self.backend_ = backend
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
        # All three return a comparable arviz.InferenceData, so predict(),
        # convergence() and parity are backend-agnostic.
        self.idata_ = sampler(
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            show_progress=show_progress,
            **extra,
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

    def _sample_numpyro(
        self, *, chains, iter_warmup, iter_sampling, seed, target_accept, show_progress
    ):
        """NumPyro (JAX) port. Same parameterization, same contract.

        The custom od-Poisson quasi-likelihood is a real ``dist.Distribution``
        so the fit carries a per-observation ``log_likelihood`` group; see
        ``model_numpyro`` for why a ``numpyro.factor`` would not.
        """
        from ibnr.gallery.bayesian.england_verrall_odp import model_numpyro

        return model_numpyro.sample(
            self.contract_,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            progress_bar=show_progress,
        )

    def _sample_pymc(
        self, *, chains, iter_warmup, iter_sampling, seed, target_accept, show_progress,
        nuts_sampler="pymc",
    ):
        """PyMC port. Same parameterization, same contract."""
        from ibnr.gallery.bayesian.england_verrall_odp import model_pymc

        return model_pymc.sample(
            self.contract_,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            progressbar=show_progress,
        )

    def predict(self, seed: int | None = None) -> PredictiveDistribution:
        """Predictive distribution of ultimates (losses at the last dev period)
        by origin year plus their total: each origin's observed paid-to-date
        plus simulated future increments, X[w,d] ~ phi * Poisson(m[w,d] / phi)
        for d beyond the origin's latest observed lag - the od-Poisson process
        draw England & Verrall obtain by imputing future cells (7.11.6), and
        the same process distribution as the ODP bootstrap baselines."""
        if self.idata_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_w, n_d, phi = c["n_w"], c["n_d"], c["phi"]

        # Posterior draws of the log-link effects (see card.md "Model"):
        #   log m[w,d] = logprem[w] + c + alpha[w] + beta[d]
        # alpha = origin (row) effect, beta = dev-lag (column) effect,
        # c = intercept; corner constraints alpha[1] = beta[1] = 0.
        alpha = pooled(self.idata_, "alpha")  # (draws, n_w) origin effects
        beta = pooled(self.idata_, "beta")  # (draws, n_d) dev-lag effects
        const = pooled(self.idata_, "c")  # (draws,) intercept
        n_draws = const.shape[0]
        logprem_origin = np.log(c["premium"])  # (n_w,) premium offset per origin

        # Ultimate = observed paid-to-date + simulated future increments.
        # Start every draw at the origin's latest cumulative paid (constant),
        # then add process draws for each still-unobserved dev lag.
        rng = np.random.default_rng(seed)
        ults = np.tile(c["paid_to_date"], (n_draws, 1)).astype(float)  # (draws, n_w)
        for j in range(n_w):  # per origin
            # Only lags strictly beyond this origin's latest observed lag are
            # unobserved; earlier lags are already in paid_to_date.
            for dev in range(int(c["latest_d"][j]) + 1, n_d + 1):
                # Posterior mean of the future increment for this (origin, lag),
                # one value per draw: mu (draws,).
                mu = np.exp(logprem_origin[j] + const + alpha[:, j] + beta[:, dev - 1])
                # Process draw: X ~ phi * Poisson(mu/phi) reproduces mean mu and
                # variance phi*mu (od-Poisson), the BootChainLadder od.pois twin.
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
