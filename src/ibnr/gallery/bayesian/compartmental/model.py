"""Hierarchical compartmental reserving gallery entry (Gesmann & Morris).

Compartments EX -> OS -> PD (closed-form ODE solution) fit JOINTLY to case
outstanding and paid losses — the only entry in the family that uses both
triangles at once. Two ablatable variants, both from the monograph's case
study with its published brms priors held verbatim:

- ``variant="gaussian"`` (default): case-study Model 1 — Gaussian on OS +
  cumulative paid amounts, correlated (RLR, RRF) accident-year effects,
  ker/kp fixed across accident years. Takes zero/negative cells natively,
  which is what a 200-company mechanical retrospective needs.
- ``variant="lognormal"``: case-study Model 2 — lognormal on OS +
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

STAN_FILES = {
    "gaussian": Path(__file__).parent / "model.stan",
    "lognormal": Path(__file__).parent / "model_lognormal.stan",
}

#: posterior backends this entry can dispatch to (ports land in milestone 5)
BACKENDS = ("stan",)


def os_curve(t, ker, kp, rlr):
    """Outstanding loss ratio at age t years: OS(t) of the EX->OS->PD system
    with EX(0) = 1 (the same closed form as the Stan functions block)."""
    return rlr * ker / (ker - kp) * (np.exp(-kp * t) - np.exp(-ker * t))


def paid_curve(t, ker, kp, rlr, rrf):
    """Cumulative paid loss ratio at age t years; -> RLR * RRF as t -> inf."""
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
        self.contract_: dict | None = None
        self.idata_ = None
        self.fit_ = None  # cmdstan fit object
        self.backend_: str | None = None
        self.variant_: str | None = None
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
        # max_treedepth = 15 — the hierarchy is genuinely hard geometry
        target_accept: float = 0.99,
        max_treedepth: int = 15,
        show_progress: bool = False,
    ) -> Compartmental:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if variant not in STAN_FILES:
            raise ValueError(f"variant must be one of {tuple(STAN_FILES)}, got {variant!r}")
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
        self.idata_ = self._sample_stan(
            stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            show_progress=show_progress,
        )
        return self

    def _gaussian_stan_data(self) -> dict:
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
        drop counts land in ``dropped_cells_`` — a mechanical study must
        report them, they are the variant's analogue of the ODP entries'
        negative-increment failures."""
        c = self.contract_
        paid_blk = c["delta"] == 1
        w = c["w"]
        value = c["loss"].copy()
        # difference the paid block within each origin (rows are sorted by
        # (delta, w, d) and contiguous per origin, so a shifted diff works)
        for j in range(1, c["n_w"] + 1):
            sel = paid_blk & (w == j)
            value[sel] = np.diff(c["loss"][sel], prepend=0.0)
        ratio = value / c["premium"][w - 1]
        keep = ratio > 0
        self.dropped_cells_ = {
            "outstanding": int((~keep & (c["delta"] == 0)).sum()),
            "paid_incremental": int((~keep & paid_blk).sum()),
        }
        if not keep.any():
            raise ValueError("no positive cells left for the lognormal variant")
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
        show_progress,
    ):
        import time

        import arviz as az
        from cmdstanpy import CmdStanModel

        ensure_stan_toolchain()
        model = CmdStanModel(stan_file=str(STAN_FILES[self.variant_]))
        t0 = time.perf_counter()
        self.fit_ = model.sample(
            data=stan_data,
            chains=chains,
            parallel_chains=1,  # sequential: fair single-core runtime vs future ports
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
        """Predictive distribution of cumulative paid at the triangle's final
        development age per origin, plus the total.

        - gaussian: each not-fully-developed origin draws
          Normal(premium * paid_curve(t_final), sigma_paid) per posterior
          draw — the model's own (unconditional-given-parameters) predictive;
          the accident-year effects carry what the origin's observed cells
          taught the posterior. Fully developed origins anchor at the
          observed value (zero variance), as in the Meyers-family entries.
        - lognormal: anchored at paid-to-date, adding lognormal incremental
          draws cell-by-cell through the final age with the per-(origin, dev)
          compartmental parameters.
        No tail beyond the triangle's final age in either variant.
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
        c = self.contract_
        n_w, n_d = c["n_w"], c["n_d"]
        t_final = n_d * c["dev_grain_months"] / 12.0
        rlr = pooled(self.idata_, "RLR")  # (draws, n_w)
        rrf = pooled(self.idata_, "RRF")  # (draws, n_w)
        ker = pooled(self.idata_, "ker")  # (draws,)
        kp = pooled(self.idata_, "kp")  # (draws,)
        sigma_paid = pooled(self.idata_, "sigma_paid")  # (draws,)
        ults = np.empty((rlr.shape[0], n_w))
        for j in range(n_w):
            if int(c["latest_d"][j]) >= n_d:  # fully developed: observed anchor
                ults[:, j] = c["paid_to_date"][j]
                continue
            mu = c["premium"][j] * paid_curve(t_final, ker, kp, rlr[:, j], rrf[:, j])
            ults[:, j] = rng.normal(mu, sigma_paid)
        return ults

    def _predict_lognormal(self, rng) -> np.ndarray:
        c = self.contract_
        n_w, n_d = c["n_w"], c["n_d"]
        step_years = c["dev_grain_months"] / 12.0
        b_oker = pooled(self.idata_, "b_oker")
        b_okp = pooled(self.idata_, "b_okp")
        b_orlr = pooled(self.idata_, "b_oRLR")
        b_orrf = pooled(self.idata_, "b_oRRF")
        u_ay = pooled(self.idata_, "u_ay")  # (draws, 2, n_w)
        u_rlr_dev = pooled(self.idata_, "u_RLR_dev")  # (draws, n_d)
        u_rrf_dev = pooled(self.idata_, "u_RRF_dev")
        u_ker_ay = pooled(self.idata_, "u_ker_ay")  # (draws, n_w)
        u_ker_dev = pooled(self.idata_, "u_ker_dev")
        u_kp_ay = pooled(self.idata_, "u_kp_ay")
        u_kp_dev = pooled(self.idata_, "u_kp_dev")
        sigma_paid = pooled(self.idata_, "sigma_paid")

        ults = np.tile(c["paid_to_date"], (b_oker.shape[0], 1)).astype(float)
        for j in range(n_w):
            for dev in range(int(c["latest_d"][j]) + 1, n_d + 1):
                di = dev - 1
                ker = 3.0 * np.exp(0.1 * (b_oker + u_ker_ay[:, j] + u_ker_dev[:, di]))
                kp = 1.0 * np.exp(0.1 * (b_okp + u_kp_ay[:, j] + u_kp_dev[:, di]))
                rlr = 0.7 * np.exp(0.2 * (b_orlr + u_ay[:, 0, j] + u_rlr_dev[:, di]))
                rrf = 0.8 * np.exp(0.1 * (b_orrf + u_ay[:, 1, j] + u_rrf_dev[:, di]))
                t_hi = dev * step_years
                mu = paid_curve(t_hi, ker, kp, rlr, rrf)
                if dev > 1:
                    mu = mu - paid_curve(t_hi - step_years, ker, kp, rlr, rrf)
                ults[:, j] += c["premium"][j] * rng.lognormal(np.log(mu), sigma_paid)
        return ults

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
