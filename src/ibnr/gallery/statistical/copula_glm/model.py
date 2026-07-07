"""Copula-linked lognormal regressions across lines of business, after
Shi & Frees (2011). See card.md for the model card."""

from __future__ import annotations

import datetime as dt

import numpy as np

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.multiline import (
    assemble_predictive,
    flatten_with_totals,
    multiline_data,
    multiline_targets,
    realized_multiline,
)
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

EIG_FLOOR = 1e-10


@register
class CopulaGLM(GalleryEntry):
    name = "copula_glm"
    family = "statistical"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.beta_: np.ndarray | None = None  # (n_lob, p)
        self.sigma_: np.ndarray | None = None  # (n_lob,)
        self.corr_: np.ndarray | None = None  # (n_lob, n_lob)
        self._loss_field: str | None = None
        self._dev_effect: str = "factor"
        self._obs_w: np.ndarray | None = None  # 0-based origin index per obs cell
        self._obs_d: np.ndarray | None = None  # 0-based dev index per obs cell
        self._x: np.ndarray | None = None  # (n_obs, p) shared design
        self._pinv: np.ndarray | None = None  # (p, n_obs) precomputed lstsq
        self._latest_cum: np.ndarray | None = None  # (n_lob, n_w)
        self._latest_dev: np.ndarray | None = None  # (n_w,), 1-based

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        dev_effect: str = "factor",  # "factor" | "hoerl"
        nonpositive: str = "error",  # "error" | "drop"
    ) -> CopulaGLM:
        """Fit per-line lognormal regressions on incremental loss ratios plus
        a Gaussian copula across lines, on the as_of-sliced upper triangle."""
        if dev_effect not in ("factor", "hoerl"):
            raise ValueError(f"dev_effect must be 'factor' or 'hoerl', got {dev_effect!r}")
        if nonpositive not in ("error", "drop"):
            raise ValueError(f"nonpositive must be 'error' or 'drop', got {nonpositive!r}")
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = multiline_data(train, loss_field=loss_field, premium_field=premium_field)
        self._loss_field = loss_field
        self._dev_effect = dev_effect
        c = self.contract_
        n_lob, n_w, n_d = c["n_lob"], c["n_w"], c["n_d"]
        cum, mask = c["cum"], c["obs_mask"]

        incr = np.full_like(cum, np.nan)
        incr[:, :, 0] = cum[:, :, 0]
        incr[:, :, 1:] = cum[:, :, 1:] - cum[:, :, :-1]
        usable = mask & ~np.isnan(incr)  # aligned masks minus gap predecessors
        usable = np.broadcast_to(usable.all(axis=0), usable.shape).copy()

        nonpos = usable & ~(incr > 0)
        if nonpos.any():
            cells = int(nonpos.any(axis=0).sum())
            if nonpositive == "error":
                raise ValueError(
                    f"{cells} cells have a non-positive increment in some line; "
                    "lognormal marginals need positive increments — use paid losses, "
                    'or nonpositive="drop" (biased: it censors the left tail)'
                )
            usable &= ~nonpos.any(axis=0)  # drop the cell in every line, keep alignment

        cell = usable[0]
        obs_w, obs_d = np.nonzero(cell)
        n_obs = len(obs_w)
        ratios = incr[:, obs_w, obs_d] / c["premium"][:, obs_w]
        y = np.log(ratios)  # (n_lob, n_obs)

        x = self._design(obs_w, obs_d, n_w, n_d, dev_effect)
        p = x.shape[1]
        if n_obs <= p:
            raise ValueError(
                f"{n_obs} usable cells for {p} marginal parameters; "
                'not enough data — try dev_effect="hoerl" or a coarser model'
            )
        self._check_identified(obs_w, obs_d, n_w, n_d, dev_effect)

        pinv = np.linalg.pinv(x)
        beta = (pinv @ y.T).T  # (n_lob, p)
        resid = y - beta @ x.T
        sigma = np.sqrt((resid**2).sum(axis=1) / (n_obs - p))
        std_resid = resid / sigma[:, None]
        self.corr_ = _nearest_pd(np.corrcoef(std_resid)) if n_lob > 1 else np.ones((1, 1))

        latest_dev = np.zeros(n_w, dtype=int)
        latest_cum = np.zeros((n_lob, n_w))
        for w in range(n_w):
            devs = np.nonzero(mask[0, w])[0]
            if devs.size == 0:
                raise ValueError(f"origin {c['origin_periods'][w]} has no observations")
            latest_dev[w] = int(devs[-1]) + 1
            latest_cum[:, w] = cum[:, w, devs[-1]]

        self.beta_, self.sigma_ = beta, sigma
        self._obs_w, self._obs_d, self._x, self._pinv = obs_w, obs_d, x, pinv
        self._latest_cum, self._latest_dev = latest_cum, latest_dev
        return self

    def predict(
        self,
        *,
        n_draws: int = 10_000,
        seed: int | None = None,
        param_uncertainty: str = "bootstrap",  # "bootstrap" | "plugin"
        n_boot: int = 200,
    ) -> PredictiveDistribution:
        """Simulate ultimates: correlated lognormal future increments cell by
        cell (dependence across lines within a cell, independence across
        cells, as in Shi & Frees), anchored on the latest observed diagonal.

        ``bootstrap`` refits the marginals+copula on ``n_boot`` triangles
        simulated from the fitted model and spreads the draws over the
        replicates — parameter risk included. ``plugin`` uses point estimates.
        """
        if self.beta_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        if param_uncertainty not in ("bootstrap", "plugin"):
            raise ValueError(
                f"param_uncertainty must be 'bootstrap' or 'plugin', got {param_uncertainty!r}"
            )
        c = self.contract_
        n_lob, n_w, n_d = c["n_lob"], c["n_w"], c["n_d"]
        rng = np.random.default_rng(seed)

        if param_uncertainty == "bootstrap":
            beta_r, sigma_r, chol_r = self._bootstrap(rng, n_boot)
            rep = rng.integers(0, n_boot, size=n_draws)
            beta_d, sigma_d, chol_d = beta_r[rep], sigma_r[rep], chol_r[rep]
        else:
            beta_d = np.broadcast_to(self.beta_, (n_draws, *self.beta_.shape))
            sigma_d = np.broadcast_to(self.sigma_, (n_draws, n_lob))
            chol = np.linalg.cholesky(self.corr_)
            chol_d = np.broadcast_to(chol, (n_draws, n_lob, n_lob))

        ults = np.tile(self._latest_cum, (n_draws, 1, 1))  # (n_draws, n_lob, n_w)
        future_w, future_d = [], []
        for w in range(n_w):
            for d in range(self._latest_dev[w], n_d):
                future_w.append(w)
                future_d.append(d)
        if future_w:
            xf = self._design(np.array(future_w), np.array(future_d), n_w, n_d, self._dev_effect)
            mu = np.einsum("nkp,cp->nkc", beta_d, xf)  # (n_draws, n_lob, n_cells)
            z = rng.standard_normal((n_draws, len(future_w), n_lob))
            eta = np.einsum("nkl,ncl->nkc", chol_d, z)
            incr = c["premium"][None, :, future_w] * np.exp(mu + sigma_d[:, :, None] * eta)
            np.add.at(ults, (slice(None), slice(None), np.array(future_w)), incr)

        targets = multiline_targets(c["lobs"], c["origin_periods"], premium=c["premium"])
        return assemble_predictive(ults, targets, units=c["units"])

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to predict()'s targets, from the full triangle at
        the final dev lag."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        realized = realized_multiline(
            full_triangle,
            loss_field=self._loss_field,
            dev_lag=c["n_d"] * c["dev_grain_months"],
            lobs=c["lobs"],
            origins=c["origin_periods"],
        )
        return flatten_with_totals(realized)

    # -- internals ---------------------------------------------------------------

    def _bootstrap(self, rng, n_boot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parametric bootstrap: simulate the observed cells from the fitted
        model, refit marginals and copula per replicate."""
        n_lob = self.beta_.shape[0]
        n_obs, p = self._x.shape
        chol = np.linalg.cholesky(self.corr_)
        mu = self.beta_ @ self._x.T  # (n_lob, n_obs)

        beta_r = np.empty((n_boot, n_lob, p))
        sigma_r = np.empty((n_boot, n_lob))
        chol_r = np.empty((n_boot, n_lob, n_lob))
        for b in range(n_boot):
            eta = chol @ rng.standard_normal((n_lob, n_obs))
            y_star = mu + self.sigma_[:, None] * eta
            beta_b = (self._pinv @ y_star.T).T
            resid = y_star - beta_b @ self._x.T
            sigma_b = np.sqrt((resid**2).sum(axis=1) / (n_obs - p))
            std = resid / sigma_b[:, None]
            corr_b = _nearest_pd(np.corrcoef(std)) if n_lob > 1 else np.ones((1, 1))
            beta_r[b], sigma_r[b] = beta_b, sigma_b
            chol_r[b] = np.linalg.cholesky(corr_b)
        return beta_r, sigma_r, chol_r

    @staticmethod
    def _design(
        obs_w: np.ndarray, obs_d: np.ndarray, n_w: int, n_d: int, dev_effect: str
    ) -> np.ndarray:
        """Shared marginal design: intercept + origin dummies (w >= 2) + either
        dev dummies (d >= 2) or the 2-parameter Hoerl curve [ln(d), d]."""
        n = len(obs_w)
        cols = [np.ones(n)]
        for w in range(1, n_w):
            cols.append((obs_w == w).astype(float))
        if dev_effect == "factor":
            for d in range(1, n_d):
                cols.append((obs_d == d).astype(float))
        else:
            dev = obs_d + 1.0
            cols.append(np.log(dev))
            cols.append(dev)
        return np.column_stack(cols)

    @staticmethod
    def _check_identified(obs_w, obs_d, n_w: int, n_d: int, dev_effect: str) -> None:
        missing_w = [w + 1 for w in range(n_w) if not (obs_w == w).any()]
        if missing_w:
            raise ValueError(f"origins {missing_w} have no usable cells; effects unidentified")
        if dev_effect == "factor":
            missing_d = [d + 1 for d in range(n_d) if not (obs_d == d).any()]
            if missing_d:
                raise ValueError(
                    f"dev steps {missing_d} have no usable cells; "
                    'factor effects unidentified — try dev_effect="hoerl"'
                )


def _nearest_pd(mat: np.ndarray) -> np.ndarray:
    """Symmetrize and floor eigenvalues so the Cholesky factor exists."""
    sym = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= EIG_FLOOR:
        return sym
    scale = max(np.abs(vals).max(), 1.0)
    return (vecs * np.maximum(vals, EIG_FLOOR * scale)) @ vecs.T
