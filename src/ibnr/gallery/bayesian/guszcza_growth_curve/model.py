"""Hierarchical growth-curve loss reserving gallery entry (Guszcza / Gesmann).

References: Guszcza, *Hierarchical Growth Curve Models for Loss Reserving*
(CAS Forum, 2008) for the model structure; the implementation reference - and
the source of the likelihood and priors, held verbatim - is Gesmann,
"Hierarchical loss reserving with growth curves using brms" (magesblog.com,
2018-07-15), whose ``brm()`` call is the specification. See ``card.md``.

The model: cumulative paid **loss ratios** follow a lognormal around a
parametric growth-curve share of a per-accident-year ultimate::

    y[w,d] = C[w,d] / premium[w]
    y[w,d] ~ lognormal(log(ulr[w] * G(t; omega, theta)), sigma)
    ulr[w] = ulr_pop + sd_ulr * z_ulr[w]        (non-centered AY effect)

with ``t = d * dev_grain_months / 12`` the development age in years and ``G``
the loglogistic (the post's curve) or weibull growth curve, **reused** from
``gallery/statistical/clark/model.py::growth``.

Relation to the rest of the gallery:

- ``compartmental`` (Gesmann & Morris) is this model's successor - same
  author, same hierarchical philosophy, an ODE system where this has a curve.
  This entry is the simpler ancestor.
- ``clark_growth_curve`` shares the growth curves but is a different model:
  fixed effects and an ODP *quasi*-likelihood, hence CRPS-only. This entry's
  per-AY random ultimates and PROPER lognormal density are exactly what make
  it ELPD-eligible - a new density member for the milestone-6 board.

**Hierarchical across ACCIDENT YEARS, not across companies.** The post fits
ten companies jointly with correlated per-company effects on (ulr, omega,
theta); this entry fits ONE cohort at a time (the package-wide contract),
which collapses the company level into the population intercepts - one company
cannot identify a between-company sd or an LKJ correlation - leaving Guszcza's
single-company model. Note that this narrows the MARGINAL prior on the curve
parameters (the entity-level ``student_t(3, 0, 1)`` sd goes with the level),
which is the over-confident direction; the card states the size and the
ablation that would answer it.

Stan-only in this branch; the NumPyro/PyMC ports and their parity gates are a
follow-up task, matching how milestones 4 -> 5 sequenced the rest of the
family. The ``backend`` argument already reserves the seam.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr.gallery.bayesian._toolchain import ensure_stan_toolchain
from ibnr.gallery.bayesian.guszcza_growth_curve import scorer
from ibnr.gallery.entry import GalleryEntry, PredictsHeldout, ScoresHeldout
from ibnr.gallery.registry import register
from ibnr.gallery.statistical.clark.model import growth
from ibnr.kernels.contract import realized_values, stan_data
from ibnr.kernels.holdout import CellIndex
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

STAN_FILE = Path(__file__).parent / "model.stan"

#: integer curve codes passed to Stan as data, so a port cannot drift on which
#: curve it fits (same convention as clark_growth_curve)
CURVE_CODES = {"loglogistic": 1, "weibull": 2}

#: posterior backends this entry can dispatch to. Stan-only for now: the
#: NumPyro/PyMC ports (and their parity gates) are a follow-up task, and
#: accepting a backend that does not exist yet would be a wire without a
#: connection.
BACKENDS = ("stan",)


def pooled(idata, name: str) -> np.ndarray:
    """Draws for a posterior variable pooled across chains: an idata
    ``posterior[name]`` of shape (chain, draw, *dims) -> (chain*draw, *dims)."""
    arr = np.asarray(idata.posterior[name].values)
    return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


@register
class GuszczaGrowthCurve(GalleryEntry, ScoresHeldout, PredictsHeldout):
    name = "guszcza_growth_curve"
    family = "bayesian"

    #: the likelihood is a lognormal on cumulative paid loss RATIOS (the post's
    #: ``loss_ratio := cumulative_paid / premium``), so the density needs the
    #: ``- log premium`` carry before it can be summed with an entry that
    #: models amounts. ``ScoresHeldout.log_lik_at`` applies it. Class-level,
    #: unlike compartmental's per-variant declarations: both curves sit on the
    #: same measure.
    heldout_measure = "loss_ratio"

    #: ``y`` is the CUMULATIVE paid loss ratio, so a draw (scaled by premium in
    #: the scorer) is a cumulative amount. On a cumulative triangle this makes
    #: ``predict_at`` a pass-through; declared rather than assumed, because an
    #: undeclared scale mismatch is off by the whole training-diagonal anchor
    #: while staying finite and plausible.
    heldout_draw_scale = "cumulative"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.stan_data_: dict | None = None  # the single data block Stan consumed
        self.idata_ = None
        self.fit_ = None  # cmdstan fit object
        self.backend_: str | None = None
        self.curve_: str | None = None
        self._loss_field: str | None = None

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
        # the post's own control list: adapt_delta = 0.999, max_treedepth = 15
        # (brms defaults diverge on this nonlinear hierarchy). The model is
        # small - n_w + 5 parameters - so the settings are cheap here.
        target_accept: float = 0.999,
        max_treedepth: int = 15,
        parallel_chains: int = 1,
        show_progress: bool = False,
    ) -> GuszczaGrowthCurve:
        """Fit one cohort (single company x line) as of a training diagonal.

        ``as_of`` slices the triangle to the training cutoff; ``growth_curve``
        selects the post's loglogistic (default) or the weibull - the
        ablatable dimension of this entry, everything else shared.

        Atomic: all state is built into locals and assigned only after the
        sampler returns, so a failed re-fit cannot leave the entry torn
        between an old posterior and a new contract.
        """
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if growth_curve not in CURVE_CODES:
            raise ValueError(
                f"growth_curve must be one of {tuple(CURVE_CODES)}, got {growth_curve!r}"
            )
        if premium_field is None:
            raise ValueError(
                "guszcza_growth_curve models loss RATIOS, so it cannot fit without a premium_field"
            )
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # the shared cumulative-loss contract (CLAUDE.md decision 3); it
        # refuses non-positive losses, which is exactly the lognormal's support
        contract = stan_data(train, loss_field=loss_field, premium_field=premium_field)
        # ages in YEARS (the post's dev_year = 1..10 on the annual grain), so
        # theta's normal(4, 1) prior keeps its meaning on any grain; and the
        # loss RATIO exactly as the post forms it: cumulative_paid / premium
        t = contract["d"].astype(float) * contract["dev_grain_months"] / 12.0
        y = contract["loss"] / contract["premium"][contract["w"] - 1]
        stan_block = {
            "len_data": contract["len_data"],
            "n_w": contract["n_w"],
            "w": contract["w"],
            "t": t,
            "y": y,
            "curve": CURVE_CODES[growth_curve],
        }
        idata, cmdstan_fit = self._sample_stan(
            stan_block,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            parallel_chains=parallel_chains,
            show_progress=show_progress,
        )
        # atomic assignment: nothing above touched self
        self.contract_ = contract
        self.stan_data_ = stan_block
        self.idata_ = idata
        self.fit_ = cmdstan_fit
        self.backend_ = backend
        self.curve_ = growth_curve
        self._loss_field = loss_field
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
        stan_block,
        *,
        chains,
        iter_warmup,
        iter_sampling,
        seed,
        target_accept,
        max_treedepth,
        parallel_chains,
        show_progress,
    ):
        """cmdstanpy NUTS on ``model.stan``; returns ``(idata, fit)`` and
        touches no instance state (see ``fit()``'s atomicity note).

        Init strategy (documented per CLAUDE.md decision 7): ``z_ulr`` starts
        at 0, everything else at Stan's random default. ``ulr = ulr_pop`` at
        the start is then positive for every chain, so the initial ``mu`` is
        always finite; a fully random ``z_ulr`` can put ``ulr_pop + sd * z``
        below zero, where ``log()`` is NaN and initialization has to retry.
        A PARTIAL init on purpose - the other parameters keep their random
        starts, so the chains stay dispersed and R-hat keeps its meaning.
        """
        import time

        import arviz as az
        from cmdstanpy import CmdStanModel

        ensure_stan_toolchain()
        model = CmdStanModel(stan_file=str(STAN_FILE))
        t0 = time.perf_counter()
        fit = model.sample(
            data=stan_block,
            chains=chains,
            # 1 = sequential, the fair single-core runtime convention vs the
            # future ports; the retro harness raises it in escalation stages
            parallel_chains=parallel_chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            adapt_delta=target_accept,
            max_treedepth=max_treedepth,
            inits={"z_ulr": [0.0] * int(stan_block["n_w"])},
            show_progress=show_progress,
        )
        runtime_s = time.perf_counter() - t0
        idata = az.from_cmdstanpy(fit, log_likelihood="log_lik")
        idata.attrs["runtime_s"] = runtime_s
        idata.attrs["backend"] = "stan"
        return idata, fit

    def predict(self, seed: int | None = None) -> PredictiveDistribution:
        """Predictive distribution of cumulative paid at the triangle's final
        development age per origin, plus the total.

        Each not-fully-developed origin draws
        ``premium[w] * lognormal(log(ulr[w] * G(t_final)), sigma)`` per
        posterior draw - the model's own predictive, with the accident-year
        effect carrying what the origin's observed cells taught the posterior.
        Fully developed origins anchor at the observed value (zero variance),
        the Meyers-family convention, so the retro percentiles are comparable.
        No tail beyond the triangle's final age: the retrospective scores the
        realized paid at that age, and extrapolating past it (``G < 1`` there)
        would score a different quantity.

        ``ulr`` draws must be strictly positive, and that is CHECKED here
        through the same ``scorer.check_ulr_positive`` the density path uses -
        not assumed. The sampler rejects such draws (a non-positive ``ulr[w]``
        makes that origin's training ``mu`` NaN), so none survives a Stan fit;
        but a guard on only one of the two paths is not a guard, and the
        unguarded version emitted NaN here exactly where the scorer refused,
        with the NaN reaching ``PredictiveDistribution``, ``evaluate()`` and
        any retro CSV.
        """
        if self.idata_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_w, n_d = c["n_w"], c["n_d"]
        t_final = n_d * c["dev_grain_months"] / 12.0

        ulr = scorer.check_ulr_positive(pooled(self.idata_, "ulr"))  # (draws, n_w)
        omega = pooled(self.idata_, "omega")  # (draws,)
        theta = pooled(self.idata_, "theta")
        sigma = pooled(self.idata_, "sigma")
        g_final = growth(t_final, omega, theta, self.curve_)  # (draws,)

        w_arr = np.asarray(c["w"], dtype=int)
        d_arr = np.asarray(c["d"], dtype=int)
        rng = np.random.default_rng(seed)
        ults = np.empty((ulr.shape[0], n_w))
        for j in range(n_w):
            mine = w_arr == j + 1
            if int(d_arr[mine].max()) >= n_d:
                # fully developed at the cutoff: the observed C[w, n_d] is the
                # deterministic anchor, as in the other Bayesian entries
                ults[:, j] = float(c["loss"][mine & (d_arr == n_d)][0])
                continue
            mu = np.log(ulr[:, j] * g_final)  # (draws,)
            ults[:, j] = c["premium"][j] * rng.lognormal(mu, sigma)

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
        from the full triangle at the final development lag, restricted to the
        training origins (the CLAUDE.md post-study-origin gotcha)."""
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

    def _log_lik_native(self, cells: CellIndex) -> np.ndarray:
        """``(n_draws, n_cells)`` on this entry's own measure, the loss-ratio
        scale. Glue over ``scorer.log_lik_cells``: the arithmetic lives beside
        ``model.stan`` where it can be read against it."""
        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        return scorer.log_lik_cells(self.contract_, self._posterior(), cells, curve=self.curve_)

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        """``(n_draws, n_cells)`` cumulative-AMOUNT draws. Same glue over the
        same posterior as :meth:`_log_lik_native`, so the density and the draws
        cannot describe different distributions - both read ``mu_cells`` and
        ``sigma_cells``."""
        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        return scorer.draw_cells(
            self.contract_, self._posterior(), cells, rng=rng, curve=self.curve_
        )

    def _posterior(self) -> dict[str, np.ndarray]:
        """Pooled draws of the variables both cell-level scorers read.

        ``posterior``, never ``log_likelihood``: that group's naming is not
        uniform across backends, and it holds the TRAINING cells in any case.
        """
        return {name: pooled(self.idata_, name) for name in scorer.REQUIRED_DRAWS}

    def convergence(self, var_names: list[str] | None = None) -> dict:
        """Convergence diagnostics from the fitted posterior: max R-hat, min
        bulk/tail ESS, divergence count/fraction, and wall-clock sampling
        runtime. ``var_names`` defaults to the sampled scalar parameters (the
        non-centered ``z_ulr`` and the transformed ``ulr`` are excluded so one
        weakly identified AY effect does not dominate max_rhat)."""
        import arviz as az

        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        if var_names is None:
            var_names = ["ulr_pop", "omega", "theta", "sd_ulr", "sigma"]
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
