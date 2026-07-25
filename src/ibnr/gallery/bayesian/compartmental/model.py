"""Hierarchical compartmental reserving gallery entry (Gesmann & Morris).

Reference: Gesmann & Morris, *Hierarchical Compartmental Reserving Models*
(CAS Research Paper, 2020); the published brms code of the Section 5 case
study (appendix 7.2) is ground truth for both parameterization and priors.
See ``card.md`` for the model card and the 200-company Schedule P
retrospective, ``kernels.contract.compartmental_stan_data`` for the data
contract, and ``model.stan`` / ``model_lognormal.stan`` for the two fits.

The mechanism: premium flows through three compartments, exposure (EX) ->
case outstanding (OS) -> paid (PD), driven by two per-year rates and two
loss ratios::

    EX' = -ker * EX                    ker = earning/reporting rate
    OS' =  ker * RLR * EX - kp * OS    RLR = reported loss ratio
    PD' =  kp  * RRF * OS              kp  = payment (settlement) rate,
                                       RRF = reserve robustness factor

With EX(0) = 1 per unit of premium this linear system has the closed form
implemented in :func:`os_curve` / :func:`paid_curve` below (and, identically,
in the Stan ``functions`` blocks) - no ODE solver anywhere. The ultimate
loss ratio is RLR * RRF: RLR sets how much of premium is ever *reported* as
case reserves, RRF the factor by which the case estimates are ultimately
redundant (RRF < 1) or deficient (RRF > 1).

This is the ONLY gallery entry that fits case outstanding and paid JOINTLY:
the contract stacks two blocks of cells (delta = 0 outstanding, computed as
``reported_loss - paid_loss``; delta = 1 paid) that share one set of
compartmental parameters, and ``t`` is the development age in YEARS at the
cell's period end because ker/kp are per-year rates.

Two ablatable variants, both from the monograph's case study with its
published brms priors held verbatim:

- ``variant="gaussian"`` (default): case-study Model 1 - Gaussian on OS +
  cumulative paid amounts, correlated (RLR, RRF) accident-year effects,
  ker/kp fixed across accident years. Takes zero/negative cells natively,
  which is what a 200-company mechanical retrospective needs.
- ``variant="lognormal"``: case-study Model 2 - lognormal on OS +
  incremental paid loss ratios, accident- AND development-year varying
  effects on all four compartmental parameters. Non-positive cells cannot
  enter the likelihood; they are dropped and counted in ``dropped_cells_``.

NumPyro/PyMC ports arrive with milestone 5; the ``backend`` argument
already reserves the seam."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr.gallery.bayesian._toolchain import ensure_stan_toolchain
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.contract import compartmental_stan_data, realized_values
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

#: one Stan program per variant - the eject pattern (CLAUDE.md design note 6):
#: literal, readable source the user can scaffold out and edit, not codegen
STAN_FILES = {
    "gaussian": Path(__file__).parent / "model.stan",
    "lognormal": Path(__file__).parent / "model_lognormal.stan",
}

#: posterior backends this entry can dispatch to (ports land in milestone 5)
#: posterior backends this entry can dispatch to; all three target the same
#: posterior, which ``kernels.parity`` gates before any convergence claim
BACKENDS = ("stan", "numpyro", "pymc")


def os_curve(t, ker, kp, rlr):
    """Outstanding loss ratio at age t years: OS(t) of the EX->OS->PD system
    with EX(0) = 1 (the same closed form as the Stan functions block).

    A hump: case reserves build at rate ``ker`` as exposure is reported and
    drain at rate ``kp`` as claims settle, so OS -> 0 as t -> inf. Note the
    ``ker - kp`` denominator is singular at ker == kp; the lognormal priors
    (medians 3 and 1) keep the sampler far from that ridge in practice.
    Broadcasts over numpy arrays of posterior draws.
    """
    return rlr * ker / (ker - kp) * (np.exp(-kp * t) - np.exp(-ker * t))


def paid_curve(t, ker, kp, rlr, rrf):
    """Cumulative paid loss ratio at age t years; -> RLR * RRF as t -> inf.

    The integral of ``kp * RRF * OS(t)``, i.e. the growth curve this entry
    predicts on. Multiply by the origin's premium for an amount.
    """
    return rlr * rrf / (ker - kp) * (ker * (1 - np.exp(-kp * t)) - kp * (1 - np.exp(-ker * t)))


def pooled(idata, name: str) -> np.ndarray:
    """Draws for a posterior variable pooled across chains: an idata
    ``posterior[name]`` of shape (chain, draw, *dims) -> (chain*draw, *dims)."""
    arr = np.asarray(idata.posterior[name].values)
    return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


@register
class Compartmental(GalleryEntry):
    name = "compartmental"
    family = "bayesian"

    def __init__(self) -> None:
        # the standardized joint paid+outstanding dict (contract.py); also
        # carries the metadata predict() needs: origin_periods, premium,
        # paid_to_date, latest_d, dev_grain_months
        self.contract_: dict | None = None
        self.idata_ = None
        self.fit_ = None  # cmdstan fit object
        self.backend_: str | None = None
        self.variant_: str | None = None
        # {"outstanding": n, "paid_incremental": n} - cells the lognormal
        # likelihood could not take; a mechanical study must report them
        self.dropped_cells_: dict | None = None  # lognormal only
        self._loss_field: str | None = None

    def fit(
        self,
        triangle: Triangle,
        *,
        # the retrospective scores paid; reported (net of bulk) supplies the
        # outstanding block, OS = reported - paid
        loss_field: str = "paid_loss",
        reported_field: str = "reported_loss",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        variant: str = "gaussian",
        backend: str = "stan",
        chains: int = 4,
        iter_warmup: int = 1000,
        iter_sampling: int = 2500,
        seed: int | None = None,
        # the monograph ran both case-study models at adapt_delta = 0.99,
        # max_treedepth = 15 - the hierarchy is genuinely hard geometry
        target_accept: float = 0.99,
        max_treedepth: int = 15,
        parallel_chains: int = 1,
        show_progress: bool = False,
    ) -> Compartmental:
        """Fit one cohort (single company x line) as of a training diagonal.

        ``as_of`` is how the retrospective trains on an early diagonal; the
        contract requires paid and reported on identical cells and contiguous
        dev lags per origin. ``variant`` selects the Stan program (and with it
        the likelihood, the observation scale and how many varying effects the
        parameters carry) - everything else about the two arms is shared, so
        the variants stay directly ablatable.

        The default sampler settings are the monograph's own (adapt_delta
        0.99, max_treedepth 15): the four-parameter nonlinear hierarchy has a
        funnel-shaped geometry that diverges at Stan's defaults. They are the
        reason a company costs ~160 s (gaussian) / ~250 s (lognormal) of
        sequential-chain wall clock, ~20x the Meyers-family entries.
        """
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if variant not in STAN_FILES:
            raise ValueError(f"variant must be one of {tuple(STAN_FILES)}, got {variant!r}")
        # parallel_chains is cmdstan-only (chain-level parallelism), so a port
        # must reject it rather than silently ignore an escalation request.
        # max_treedepth is NOT cmdstan-only - NUTS in all three backends takes
        # it - so it is passed through, which is what lets a run be made
        # affordable for PyMC (see the card's runtime note).
        if backend != "stan" and parallel_chains != 1:
            raise ValueError(
                "parallel_chains is a stan-backend control; "
                f"the {backend!r} port does not take it"
            )
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = compartmental_stan_data(
            train,
            paid_field=loss_field,
            reported_field=reported_field,
            premium_field=premium_field,
        )
        self._loss_field = loss_field
        self.backend_ = backend
        self.variant_ = variant
        stan_data = (
            self._gaussian_stan_data() if variant == "gaussian" else self._lognormal_stan_data()
        )
        sampler = {
            "stan": self._sample_stan,
            "numpyro": self._sample_numpyro,
            "pymc": self._sample_pymc,
        }[backend]
        self.idata_ = sampler(
            stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            parallel_chains=parallel_chains,
            show_progress=show_progress,
        )
        return self

    @classmethod
    def precompile(cls) -> None:
        """Compile both variants' Stan programs ahead of use (container build,
        or before a worker pool spawns, so concurrent first-fits never race
        the compiler)."""
        from cmdstanpy import CmdStanModel

        ensure_stan_toolchain()
        for stan_file in STAN_FILES.values():
            CmdStanModel(stan_file=str(stan_file))

    def _gaussian_stan_data(self) -> dict:
        """Model 1 rows: the contract as-is - OS levels and cumulative paid as
        AMOUNTS, every cell kept. The Gaussian likelihood takes zero and
        negative outstanding natively (redundant case reserves, or a fully
        run-off origin), which is exactly why this arm survives a mechanical
        200-company retrospective without a clamp."""
        c = self.contract_
        return {
            "len_data": c["len_data"],
            "n_w": c["n_w"],
            "w": c["w"],
            "t": c["t"],
            "delta": c["delta"],
            "loss": c["loss"],
            "premium": c["premium"],
        }

    def _lognormal_stan_data(self) -> dict:
        """Model 2 rows: OS levels and incremental paid, as LOSS RATIOS,
        non-positive cells dropped (the lognormal cannot take them). The
        drop counts land in ``dropped_cells_`` - a mechanical study must
        report them, they are the variant's analogue of the ODP entries'
        negative-increment failures."""
        c = self.contract_
        paid_blk = c["delta"] == 1
        w = c["w"]
        value = c["loss"].copy()
        # difference the paid block within each origin (rows are sorted by
        # (delta, w, d) and contiguous per origin, so a shifted diff works).
        # prepend=0 makes dev 1 its own increment, matching the Stan branch
        # that uses paid_curve(t) undifferenced when t <= devfreq.
        for j in range(1, c["n_w"] + 1):
            sel = paid_blk & (w == j)
            value[sel] = np.diff(c["loss"][sel], prepend=0.0)
        # to loss ratios: the compartmental curves ARE loss ratios (EX(0) = 1
        # per unit premium), so Model 2 models them on that scale directly
        ratio = value / c["premium"][w - 1]
        # the lognormal has support (0, inf): a redundant case-reserve level or
        # a negative paid increment (salvage/subrogation, a reopened-then-
        # closed year) has to leave the likelihood entirely
        keep = ratio > 0
        self.dropped_cells_ = {
            "outstanding": int((~keep & (c["delta"] == 0)).sum()),
            "paid_incremental": int((~keep & paid_blk).sum()),
        }
        if not keep.any():
            raise ValueError("no positive cells left for the lognormal variant")
        # ker/kp are per-year rates, so the dev period length goes to Stan in
        # years too (1.0 on the annual Schedule P grain)
        step_years = c["dev_grain_months"] / 12.0
        return {
            "len_data": int(keep.sum()),
            "n_w": c["n_w"],
            "n_d": c["n_d"],
            "w": c["w"][keep],
            "d": c["d"][keep],
            "t": c["t"][keep],
            "delta": c["delta"][keep],
            "y": ratio[keep],
            "devfreq": step_years,
        }

    def _sample_stan(
        self,
        stan_data,
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
        import time

        import arviz as az
        from cmdstanpy import CmdStanModel

        # Windows RTools PATH/MAKE fixup; no-op elsewhere (see CLAUDE.md)
        ensure_stan_toolchain()
        model = CmdStanModel(stan_file=str(STAN_FILES[self.variant_]))
        t0 = time.perf_counter()
        self.fit_ = model.sample(
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
        # log_lik is per stacked cell, so ELPD/LOO in kernels/ scores the
        # joint paid+outstanding fit (not paid alone) - noted in the card
        idata = az.from_cmdstanpy(self.fit_, log_likelihood="log_lik")
        idata.attrs["runtime_s"] = runtime_s
        idata.attrs["backend"] = "stan"
        return idata

    def _sample_numpyro(
        self,
        stan_data,
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
        """NumPyro (JAX) port of whichever variant is selected.

        The half-Student-t scales use Stan's own constrained-parameter +
        factor construction, because NumPyro's TruncatedDistribution needs a
        Student-t CDF it cannot compute here; see ``model_numpyro``.
        """
        from ibnr.gallery.bayesian.compartmental import model_numpyro

        return model_numpyro.sample(
            stan_data,
            variant=self.variant_,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            progress_bar=show_progress,
        )

    def _sample_pymc(
        self,
        stan_data,
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
        """PyMC port of whichever variant is selected."""
        from ibnr.gallery.bayesian.compartmental import model_pymc

        return model_pymc.sample(
            stan_data,
            variant=self.variant_,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            progressbar=show_progress,
        )

    def predict(self, seed: int | None = None) -> PredictiveDistribution:
        """Predictive distribution of cumulative paid at the triangle's final
        development age per origin, plus the total.

        - gaussian: each not-fully-developed origin draws
          Normal(premium * paid_curve(t_final), sigma_paid) per posterior
          draw - the model's own (unconditional-given-parameters) predictive;
          the accident-year effects carry what the origin's observed cells
          taught the posterior. Fully developed origins anchor at the
          observed value (zero variance), as in the Meyers-family entries.
        - lognormal: anchored at paid-to-date, adding lognormal incremental
          draws cell-by-cell through the final age with the per-(origin, dev)
          compartmental parameters.
        No tail beyond the triangle's final age in either variant: the
        retrospective scores the realized paid at the triangle's last dev age,
        so extrapolating past it would score a different quantity.

        Both arms return a (draws, n_w) sample matrix that ``with_total()``
        widens to (draws, n_w + 1) - the total column is the draw-wise sum, so
        it inherits the parameter correlation across origins rather than
        assuming independence.
        """
        if self.idata_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        rng = np.random.default_rng(seed)
        ults = (
            self._predict_gaussian(rng)
            if self.variant_ == "gaussian"
            else self._predict_lognormal(rng)
        )
        targets = pd.DataFrame(
            {
                "label": [str(o.year) for o in c["origin_periods"]],
                "origin_period": c["origin_periods"],
                "premium": c["premium"],
            }
        )
        pred = PredictiveDistribution(samples=ults, targets=targets)
        return pred.with_total()

    def _predict_gaussian(self, rng) -> np.ndarray:
        """Model 1 predictive: one Normal draw on the AMOUNT scale per
        posterior draw. Returns (draws, n_w).

        Note the sharpness cost the card documents: sigma_paid is a single
        amount-scale constant shared by every origin, so a mature origin's
        predictive band is as wide (in dollars) as a green one's, and the
        total's CV comes out around 2.5% - too sharp, which is what drives the
        gaussian arm's KS failure.
        """
        c = self.contract_
        n_w, n_d = c["n_w"], c["n_d"]
        # the triangle's last dev age, in years - the scoring age
        t_final = n_d * c["dev_grain_months"] / 12.0
        rlr = pooled(self.idata_, "RLR")  # (draws, n_w)
        rrf = pooled(self.idata_, "RRF")  # (draws, n_w)
        ker = pooled(self.idata_, "ker")  # (draws,)
        kp = pooled(self.idata_, "kp")  # (draws,)
        sigma_paid = pooled(self.idata_, "sigma_paid")  # (draws,)
        ults = np.empty((rlr.shape[0], n_w))
        for j in range(n_w):
            # the oldest origin(s) already reached the scoring age in the
            # training slice: its outcome is observed, so it enters at zero
            # variance rather than being re-simulated. Same convention as the
            # Meyers-family entries, so the retro percentiles are comparable.
            if int(c["latest_d"][j]) >= n_d:  # fully developed: observed anchor
                ults[:, j] = c["paid_to_date"][j]
                continue
            # per-origin RLR/RRF carry what this origin's own cells taught the
            # posterior; ker/kp are shared across accident years in Model 1
            mu = c["premium"][j] * paid_curve(t_final, ker, kp, rlr[:, j], rrf[:, j])
            ults[:, j] = rng.normal(mu, sigma_paid)
        return ults

    def _predict_lognormal(self, rng) -> np.ndarray:
        """Model 2 predictive: anchored at paid-to-date, then one lognormal
        INCREMENT drawn per future (origin, dev) cell and accumulated. Returns
        (draws, n_w).

        Because Model 2 puts varying effects on the parameters by both
        accident year and development period, the compartmental parameters
        have to be rebuilt cell by cell (the brms nlf transforms, mirrored
        from ``model_lognormal.stan``) - there is no single curve per origin.
        Anchoring on observed paid rather than re-simulating the whole curve
        is what removes the gaussian arm's level bias (median estimate /
        outcome 0.99-1.02 per line; see card.md).
        """
        c = self.contract_
        n_w, n_d = c["n_w"], c["n_d"]
        step_years = c["dev_grain_months"] / 12.0
        # population-level ("intercept") coefficients on the unconstrained
        # scale; each is (draws,)
        b_oker = pooled(self.idata_, "b_oker")
        b_okp = pooled(self.idata_, "b_okp")
        b_orlr = pooled(self.idata_, "b_oRLR")
        b_orrf = pooled(self.idata_, "b_oRRF")
        # row 0 = oRLR, row 1 = oRRF: the CORRELATED accident-year effects
        u_ay = pooled(self.idata_, "u_ay")  # (draws, 2, n_w)
        u_rlr_dev = pooled(self.idata_, "u_RLR_dev")  # (draws, n_d)
        u_rrf_dev = pooled(self.idata_, "u_RRF_dev")
        u_ker_ay = pooled(self.idata_, "u_ker_ay")  # (draws, n_w)
        u_ker_dev = pooled(self.idata_, "u_ker_dev")
        u_kp_ay = pooled(self.idata_, "u_kp_ay")
        u_kp_dev = pooled(self.idata_, "u_kp_dev")
        sigma_paid = pooled(self.idata_, "sigma_paid")

        # start every draw at the origin's observed paid-to-date, then add
        # simulated increments for the unobserved cells only
        ults = np.tile(c["paid_to_date"], (b_oker.shape[0], 1)).astype(float)
        for j in range(n_w):
            # cells strictly past this origin's latest observed dev, out to
            # the scoring age; empty for fully developed origins
            for dev in range(int(c["latest_d"][j]) + 1, n_d + 1):
                di = dev - 1  # Stan is 1-indexed, the pooled arrays are 0-
                # the brms nlf transforms, identical to model_lognormal.stan:
                # lognormal medians (3, 1, 0.7, 0.8) with CoVs (10, 10, 20,
                # 10)%. Each line is (draws,).
                ker = 3.0 * np.exp(0.1 * (b_oker + u_ker_ay[:, j] + u_ker_dev[:, di]))
                kp = 1.0 * np.exp(0.1 * (b_okp + u_kp_ay[:, j] + u_kp_dev[:, di]))
                rlr = 0.7 * np.exp(0.2 * (b_orlr + u_ay[:, 0, j] + u_rlr_dev[:, di]))
                rrf = 0.8 * np.exp(0.1 * (b_orrf + u_ay[:, 1, j] + u_rrf_dev[:, di]))
                t_hi = dev * step_years
                mu = paid_curve(t_hi, ker, kp, rlr, rrf)
                # the cell's expected INCREMENT: the curve differenced over
                # (t - devfreq, t] with the SAME cell parameters. dev == 1 is
                # its own increment (paid_curve(0) = 0), matching the Stan
                # t <= devfreq branch.
                if dev > 1:
                    mu = mu - paid_curve(t_hi - step_years, ker, kp, rlr, rrf)
                # mu is a loss ratio, so scale by premium after drawing; the
                # lognormal is parameterized by log-mean, hence log(mu)
                ults[:, j] += c["premium"][j] * rng.lognormal(np.log(mu), sigma_paid)
        return ults

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to predict()'s targets (per origin + total).

        Read off the FULL (post-training) triangle at the same dev age
        predict() targets, restricted to the origins that were in the training
        slice - the CLAUDE.md gotcha: taking every origin present in the full
        triangle silently scores post-study accident years.
        """
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
        """Convergence diagnostics from the fitted posterior.

        Summarized over the SAMPLED parameters only - the non-centered ``z_*``
        and the transformed per-origin RLR/RRF are excluded so a single badly
        identified varying effect does not dominate max_rhat. The retro harness
        writes this dict per company; see card.md for the R-hat > 1.05 counts.
        """
        import arviz as az

        if self.idata_ is None:
            raise RuntimeError("call fit() first")
        if var_names is None:
            # sampled core parameters shared by both variants; the extra
            # lognormal scales are filtered in when present
            var_names = [
                "b_oRLR",
                "b_oRRF",
                "b_oker",
                "b_okp",
                "sd_ay",
                "sd_dev",
                "sd_ker",
                "sd_kp",
                "log_sigma_os",
                "log_sigma_paid",
            ]
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
