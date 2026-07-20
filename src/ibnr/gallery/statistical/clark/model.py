"""Clark (2003) growth-curve reserving — maximum likelihood entry.

Clark, "LDF Curve-Fitting and Stochastic Reserving: A Maximum Likelihood
Approach" (CAS Forum 2003). Incremental losses follow an over-dispersed
Poisson whose mean is an ultimate times the increment of a parametric growth
curve G(x | omega, theta) evaluated at ages measured from the origin period's
average accident date (x = 12d - 6 for annual grains — the convention
chainladder-python's ClarkLDF also uses; the tieout tests pin this).

Methods, per the paper:
- ``ldf``: each origin carries its own ultimate U_w (profile MLE
  U_w = paid-to-date / G(age_w), so the point ultimate is the truncated-LDF
  answer paid * G(x_max)/G(age_w)).
- ``cape_cod``: U_w = ELR * premium_w with a single profiled ELR — Clark's
  recommendation for thin triangles.

Predictive distribution = Clark's own variance decomposition, simulated:
parameter risk from the MVN with covariance phi * inverse observed Fisher
information (log-parameter space), process risk as scaled-Poisson ODP draws,
truncated at the triangle's final age (no tail extrapolation — the backtest
scores C[w, n_d], and chainladder's ClarkLDF truncates identically)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.contract import odp_stan_data, realized_values
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

GROWTH_CURVES = ("loglogistic", "weibull")
METHODS = ("ldf", "cape_cod")


def growth(x: np.ndarray, omega: float, theta: float, curve: str) -> np.ndarray:
    """Clark's growth functions: expected fraction of ultimate paid by age x
    (months). G(0) = 0, G -> 1 as x -> inf."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    if curve == "loglogistic":
        # overflow in (theta/x)^omega during optimizer exploration is benign:
        # 1/(1+inf) -> 0 is the correct limit
        with np.errstate(divide="ignore", over="ignore"):
            return np.where(x > 0, 1.0 / (1.0 + (theta / np.maximum(x, 1e-300)) ** omega), 0.0)
    if curve == "weibull":
        return 1.0 - np.exp(-((x / theta) ** omega))
    raise ValueError(f"growth must be one of {GROWTH_CURVES}, got {curve!r}")


def _hessian(f, x0: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of a scalar function; dims here are <= n_w + 2."""
    k = len(x0)
    h = rel_step * np.maximum(np.abs(x0), 1.0)
    hess = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            hess[i, j] = hess[j, i] = (
                f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)
            ) / (4 * h[i] * h[j])
    return hess


@register
class Clark(GalleryEntry):
    name = "clark"
    family = "statistical"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.params_: dict | None = None
        self._loss_field: str | None = None

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        growth_curve: str = "loglogistic",
        method: str = "cape_cod",
    ) -> Clark:
        if growth_curve not in GROWTH_CURVES:
            raise ValueError(f"growth_curve must be one of {GROWTH_CURVES}")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = odp_stan_data(train, loss_field=loss_field, premium_field=premium_field)
        self._loss_field = loss_field
        c = self.contract_
        w, d, inc = c["w"], c["d"], c["inc_loss"]
        step = c["dev_grain_months"]
        # ages from the average accident date: [max(step(d-1) - step/2, 0), step*d - step/2]
        age_lo = np.maximum(step * (d - 1) - step / 2, 0.0)
        age_hi = step * d - step / 2

        if method == "ldf":
            row_tot = np.array([inc[w == wi].sum() for wi in range(1, c["n_w"] + 1)])
            if (row_tot <= 0).any():
                raise ValueError("ldf method needs positive paid-to-date in every origin")
        premium = c.get("premium")
        if method == "cape_cod" and premium is None:
            raise ValueError("cape_cod needs a premium_field (U[w] = ELR * premium[w])")

        def profiled_level(om: float, th: float) -> np.ndarray:
            """MLE of per-origin ultimates given the curve: U_w (ldf) or
            ELR * premium (cape_cod), both closed-form Poisson MLEs."""
            ginc = growth(age_hi, om, th, growth_curve) - growth(age_lo, om, th, growth_curve)
            if method == "ldf":
                gsum = np.array([ginc[w == wi].sum() for wi in range(1, c["n_w"] + 1)])
                return np.array([inc[w == wi].sum() for wi in range(1, c["n_w"] + 1)]) / gsum
            elr = inc.sum() / (premium[w - 1] * ginc).sum()
            return elr * premium

        def negll_curve(logparams: np.ndarray) -> float:
            om, th = np.exp(logparams)
            ginc = growth(age_hi, om, th, growth_curve) - growth(age_lo, om, th, growth_curve)
            if (ginc <= 0).any() or not np.isfinite(ginc).all():
                return 1e12
            u = profiled_level(om, th)
            mu = u[w - 1] * ginc
            return float((mu - inc * np.log(mu)).sum())

        res = minimize(
            negll_curve,
            [np.log(1.5), np.log(4 * step)],
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
        )
        if not res.success:
            raise RuntimeError(f"Clark MLE did not converge: {res.message}")
        omega, theta = np.exp(res.x)
        level = profiled_level(omega, theta)  # per-origin ultimates at the MLE

        # full-parameter quasi-Poisson information for Clark's parameter risk;
        # everything in log space (levels are positive by construction)
        if method == "ldf":
            full0 = np.concatenate([np.log(level), res.x])
        else:
            elr = level[0] / premium[0]
            full0 = np.concatenate([[np.log(elr)], res.x])

        def negll_full(fp: np.ndarray) -> float:
            om, th = np.exp(fp[-2:])
            ginc = growth(age_hi, om, th, growth_curve) - growth(age_lo, om, th, growth_curve)
            if (ginc <= 0).any() or not np.isfinite(ginc).all():
                return 1e12
            u = np.exp(fp[: c["n_w"]]) if method == "ldf" else np.exp(fp[0]) * premium
            mu = u[w - 1] * ginc
            return float((mu - inc * np.log(mu)).sum())

        ginc = growth(age_hi, omega, theta, growth_curve) - growth(
            age_lo, omega, theta, growth_curve
        )
        mu = level[w - 1] * ginc
        n, p = len(inc), (c["n_w"] if method == "ldf" else 1) + 2
        if n <= p:
            raise ValueError(f"triangle has {n} cells but the Clark model has {p} parameters")
        phi = float(((inc - mu) ** 2 / mu).sum() / (n - p))
        hess = _hessian(negll_full, full0)
        # quasi-likelihood covariance: phi * inverse Poisson information
        cov = phi * np.linalg.pinv(hess)

        self.params_ = {
            "growth_curve": growth_curve,
            "method": method,
            "omega": float(omega),
            "theta": float(theta),
            "level": level,
            "phi": phi,
            "log_params": full0,
            "log_cov": cov,
            "age_lo": age_lo,
            "age_hi": age_hi,
        }
        return self

    def predict(self, *, n_draws: int = 10_000, seed: int | None = None) -> PredictiveDistribution:
        """Ultimates per origin + total: paid-to-date plus simulated future
        increments through the final age, with Clark's parameter risk (MVN on
        the log-parameters) and ODP process risk (scaled Poisson)."""
        if self.contract_ is None or self.params_ is None:
            raise RuntimeError("call fit() first")
        c, prm = self.contract_, self.params_
        n_w, n_d, step = c["n_w"], c["n_d"], c["dev_grain_months"]
        curve, method, phi = prm["growth_curve"], prm["method"], prm["phi"]
        premium = c.get("premium")

        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(prm["log_params"], prm["log_cov"], size=n_draws)
        om = np.exp(draws[:, -2])
        th = np.exp(draws[:, -1])
        if method == "ldf":
            levels = np.exp(draws[:, :n_w])  # (n_draws, n_w)
        else:
            levels = np.exp(draws[:, [0]]) * premium[None, :]

        ults = np.tile(c["paid_to_date"], (n_draws, 1)).astype(float)
        for j in range(n_w):
            for dev in range(int(c["latest_d"][j]) + 1, n_d + 1):
                lo = max(step * (dev - 1) - step / 2, 0.0)
                hi = step * dev - step / 2
                ginc = growth(hi, om, th, curve) - growth(lo, om, th, curve)
                mu = np.maximum(levels[:, j] * ginc, 1e-12)
                ults[:, j] += phi * rng.poisson(mu / phi)

        targets = pd.DataFrame(
            {
                "label": [str(o.year) for o in c["origin_periods"]],
                "origin_period": c["origin_periods"],
                "premium": premium if premium is not None else np.nan,
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
