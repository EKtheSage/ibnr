"""Meyers Correlated Chain Ladder (CCL) gallery entry.

Three interchangeable posterior backends behind one entry: the Stan reference
(``model.stan``), the NumPyro port (``model_numpyro.py``) and the PyMC port
(``model_pymc.py``). All three consume the identical ``kernels.contract``
data dict and hold the same centered parameterization, so ``predict()`` reads
off a common ``arviz.InferenceData`` regardless of which sampler ran. See
card.md for the model card and the cross-backend convergence comparison, and
``kernels.parity`` for the posterior-agreement check.

The model itself (Meyers, *Stochastic Loss Reserving Using Bayesian MCMC
Models*, CAS Monograph 1 (2015), CCL; the CAY model of Monograph 8 (2019) §8),
on cumulative losses C[w, d] for origin year w and dev year d::

    log C[w, d] ~ normal(mu[w, d], sig[d])
    mu[w, d] = logprem[w] + logelr + alpha[w] + beta[d]
               + rho * (log C[w-1, d] - mu[w-1, d])       for w > 1

``rho`` is the cross-accident-year correlation: each origin's log loss is pulled
toward the *previous* origin's residual at the same dev lag, which is what makes
this "correlated" chain ladder and what the ``predict()`` recursion below
reproduces draw by draw."""

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
STAN_DATA_KEYS = ("len_data", "n_w", "n_d", "w", "d", "prev_idx", "logprem", "logloss")

#: posterior backends this entry can dispatch to
BACKENDS = ("stan", "numpyro", "pymc")


def pooled(idata, name: str) -> np.ndarray:
    """Draws for a posterior variable pooled across chains: an idata
    ``posterior[name]`` of shape (chain, draw, *dims) -> (chain*draw, *dims)."""
    arr = np.asarray(idata.posterior[name].values)
    return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


@register
class MeyersCCL(GalleryEntry):
    """Correlated Chain Ladder on a single company x line cumulative triangle.

    Meyers' best-performing model on *incurred* (net-of-bulk) triangles; the
    200-company Schedule P retrospective in card.md passes the KS uniformity
    test on every line and combined. Fitted state: ``contract_`` (the
    ``kernels.contract`` data dict, which also carries the metadata predict()
    needs), ``idata_`` (posterior, whichever backend produced it) and ``fit_``
    (the raw cmdstan object, Stan backend only).
    """

    name = "meyers_ccl"
    family = "bayesian"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.idata_ = None
        self.fit_ = None  # cmdstan fit object when backend == "stan", else None
        self.backend_: str | None = None
        self._loss_field: str | None = None

    def fit(
        self,
        triangle: Triangle,
        *,
        # Meyers' "incurred" is net of bulk+IBNR (paid + case) = reported_loss
        loss_field: str = "reported_loss",
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
    ) -> MeyersCCL:
        """Sample the CCL posterior for one cohort's training triangle.

        ``as_of`` is the backtest lever: it slices the triangle to a past
        evaluation date so the model sees only the upper triangle available
        then, and :meth:`realized_ultimates` can score against what actually
        developed afterwards (Meyers' retrospective protocol, card.md).

        Defaults are the monograph's: incurred net of bulk+IBNR against earned
        premium, 4 chains x 2500 draws after 1000 warmup. ``target_accept``
        maps to Stan's ``adapt_delta``; the centered parameterization is kept
        as published, so hard companies may need 0.9+ (see card.md).
        """
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
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # One contract dict for all three backends (CLAUDE.md #3): no backend
        # is allowed to grow its own data prep. stan_data() also validates the
        # single-cohort / positive-loss requirements of the lognormal model.
        self.contract_ = stan_data(train, loss_field=loss_field, premium_field=premium_field)
        self._loss_field = loss_field  # remembered so realized_ultimates() scores the same field
        self.backend_ = backend
        sampler = {
            "stan": self._sample_stan,
            "numpyro": self._sample_numpyro,
            "pymc": self._sample_pymc,
        }[backend]
        # parallel_chains / max_treedepth are cmdstan-level controls the
        # retro harness escalates on; the ports keep their own defaults
        extra = {}
        if backend == "stan":
            extra = {"parallel_chains": parallel_chains, "max_treedepth": max_treedepth}
        elif parallel_chains != 1 or max_treedepth is not None:
            raise ValueError(
                "parallel_chains / max_treedepth are stan-backend controls; "
                f"the {backend!r} port does not take them"
            )
        # All three return a comparable arviz.InferenceData, so everything
        # downstream (predict, convergence, parity) is backend-agnostic.
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

    # -- backends ------------------------------------------------------------

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
        # Imported lazily: the [bayesian] extra is optional, and ibnr.gallery
        # must import (and this entry must register) without cmdstanpy present.
        import time

        import arviz as az
        from cmdstanpy import CmdStanModel

        ensure_stan_toolchain()  # Windows: point cmdstan at the RTools make/gcc
        model = CmdStanModel(stan_file=str(STAN_FILE))  # compiles on first use, then cached
        t0 = time.perf_counter()
        self.fit_ = model.sample(
            # only the Stan `data` block keys; the contract's metadata entries
            # (premium, origin_periods, ...) would be rejected by cmdstan
            data={k: self.contract_[k] for k in STAN_DATA_KEYS},
            chains=chains,
            # 1 = sequential, the fair single-core runtime convention vs the
            # ports; the retro harness raises it in late escalation stages
            parallel_chains=parallel_chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            adapt_delta=target_accept,
            max_treedepth=max_treedepth,
            show_progress=show_progress,
        )
        # Sampling wall-clock only (compile excluded - it is cached after the
        # first fit of a given shape); convergence() reports it per backend.
        runtime_s = time.perf_counter() - t0
        # log_lik is generated per observation for ELPD/LOO in kernels/scores
        idata = az.from_cmdstanpy(self.fit_, log_likelihood="log_lik")
        idata.attrs["runtime_s"] = runtime_s
        idata.attrs["backend"] = "stan"
        return idata

    def _sample_numpyro(
        self, *, chains, iter_warmup, iter_sampling, seed, target_accept, show_progress
    ):
        """NumPyro (JAX) port. Same centered parameterization, same contract.

        The port replaces Stan's ``prev_idx`` forward recurrence for ``mu`` with
        its algebraically identical closed form (``kernels.contract.ccl_mu_index``)
        so the autodiff graph stays a small matmul; the posterior is unchanged
        and ``kernels.parity`` gates that claim.
        """
        from . import model_numpyro

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
        """PyMC (PyTensor) port - the readable reference implementation.

        Slowest of the three here (~7x Stan; PyTensor evaluates the gradient as
        many small ops and links no BLAS on this box), and the most sensitive to
        ``target_accept`` under the centered parameterization: at 0.8 it
        under-adapts on some companies. First fit of a given triangle shape pays
        a one-time C compile, then caches by graph shape. See card.md.
        """
        from . import model_pymc

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
        by origin year plus their total, via the monograph's simulation:
        sequentially over origins, mu[w] = logprem[w] + logelr + alpha[w]
        + rho * (log(C[w-1, n_d]) - mu[w-1]), C[w, n_d] ~ lognormal(mu[w], sig[n_d]).

        Why a hand-rolled recursion rather than Stan ``generated quantities``:
        the same simulation must run identically off any of the three backends'
        posteriors, so it lives here, on the pooled draws, and reads only
        parameters all three expose (Monograph 8, p. 38).

        Because ``mu[w]`` depends on the *simulated* ``C[w-1, n_d]`` of the
        origin above it, the recursion is inherently sequential over origins -
        this is exactly the cross-accident-year dependence ``rho`` encodes, and
        it is why the per-origin ultimates are correlated within a draw (which
        in turn widens the total's predictive distribution relative to summing
        independent margins).
        """
        if self.idata_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_w, n_d = c["n_w"], c["n_d"]  # origin years, dev years

        # Pooled posterior draws; shapes annotated per array.
        alpha = pooled(self.idata_, "alpha")  # (draws, n_w)
        logelr = pooled(self.idata_, "logelr")  # (draws,)
        rho = pooled(self.idata_, "rho")  # (draws,)
        sig = pooled(self.idata_, "sig")  # (draws, n_d)
        n_draws = logelr.shape[0]
        # Ultimate = the cell at the final dev year, so only sig[n_d] is used
        sig_last = sig[:, n_d - 1]  # (draws,)
        # Per-origin log premium, (n_w,) - note this is NOT the contract's
        # `logprem`, which is per observed row.
        logprem = np.log(c["premium"])

        # The recursion needs a starting residual, and the monograph takes it
        # from the oldest origin's OBSERVED ultimate rather than simulating it:
        # origin 1 is fully developed in a square training triangle, so C[1, n_d]
        # is data. Without that cell there is nothing to anchor `rho` on.
        first_mask = (c["w"] == 1) & (c["d"] == n_d)
        if not first_mask.any():
            raise ValueError(
                "the first origin year is not fully developed in the training data; "
                "the CCL simulation needs the observed C[1, n_d] anchor"
            )
        c1_ult = float(c["loss"][first_mask][0])

        rng = np.random.default_rng(seed)
        ults = np.empty((n_draws, n_w))  # (draws, n_w) cumulative ultimates
        # Origin 1's ultimate is observed, hence degenerate across draws.
        ults[:, 0] = c1_ult
        # Identifiability pinning from the Stan model carried through here:
        # alpha[1] = 0 (origin 1 is the reference level, absorbed into logelr)
        # and beta[n_d] = 0 (the final dev year is the reference level, so no
        # beta term appears anywhere in this ultimate-only simulation).
        mu_prev = logprem[0] + logelr  # (draws,) mu[1, n_d]
        log_c_prev = np.full(n_draws, np.log(c1_ult))  # (draws,) log C[1, n_d]
        for j in range(1, n_w):
            # (log_c_prev - mu_prev) is the previous origin's residual at dev
            # n_d; rho carries a fraction of it into this origin. All terms are
            # (draws,), so the whole cohort of draws advances one origin at a time.
            mu = logprem[j] + logelr + alpha[:, j] + rho * (log_c_prev - mu_prev)
            log_c = rng.normal(mu, sig_last)  # log C[w, n_d] ~ normal(mu, sig[n_d])
            ults[:, j] = np.exp(log_c)  # lognormal on the loss scale
            # Feed this origin's mu and simulated log loss forward: the residual
            # chain, not just the level, is what propagates the correlation.
            mu_prev, log_c_prev = mu, log_c

        targets = pd.DataFrame(
            {
                "label": [str(o.year) for o in c["origin_periods"]],
                "origin_period": c["origin_periods"],
                "premium": c["premium"],
            }
        )
        pred = PredictiveDistribution(samples=ults, targets=targets)
        # with_total() appends the per-draw sum as an extra target - summing
        # within a draw preserves the rho-induced cross-origin dependence, which
        # is the quantity the Meyers retrospective actually tests.
        return pred.with_total()

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to predict()'s targets (per origin + total), taken
        from the full triangle at the final development lag.

        ``full_triangle`` is the *unsliced* triangle - the later statements that
        reveal how the training diagonal actually developed. Restricting to
        ``c["origin_periods"]`` is load-bearing: the Schedule P mart carries
        accident years beyond the training slice, and aggregating without that
        filter silently inflates the outcome (CLAUDE.md gotcha).
        """
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        per_origin = realized_values(
            full_triangle,
            loss_field=self._loss_field,
            # contract dev index -> months, since dev_lag is stored in months
            dev_lag=c["n_d"] * c["dev_grain_months"],
            origins=c["origin_periods"],
        )
        # trailing element mirrors predict()'s with_total() target
        return np.append(per_origin, per_origin.sum())

    def convergence(self, var_names: list[str] | None = None) -> dict:
        """Cross-backend convergence diagnostics from the fitted posterior:
        max R-hat, min bulk/tail ESS, divergence count/fraction, and wall-clock
        sampling runtime. Computed once via arviz so every backend reports the
        same numbers. ``var_names`` defaults to the sampled (non-deterministic)
        core parameters."""
        import arviz as az

        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        if var_names is None:
            # The sampled parameters, not their transforms (alpha/beta/rho/sig):
            # R-hat and ESS on a deterministic function of the sampled space
            # would double-count, and the pinned entries alpha[1]/beta[n_d] are
            # constants that arviz cannot summarize.
            var_names = ["logelr", "r_alpha", "r_beta", "a_ig", "r_rho"]
        # ports may name things slightly differently; keep only what exists
        var_names = [v for v in var_names if v in self.idata_.posterior]
        summ = az.summary(self.idata_, var_names=var_names)
        post = self.idata_.posterior
        n_draws = int(post.sizes["chain"] * post.sizes["draw"])
        # Divergences are the diagnostic that matters for this centered
        # parameterization; not every backend/idata carries sample_stats.
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
