"""Seemingly unrelated regression across lines of business — the multivariate
chain ladder of Zhang (2010), estimated by hand-rolled feasible GLS. See
card.md for the model card."""

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

#: smallest eigenvalue allowed in any estimated covariance (PD floor)
EIG_FLOOR = 1e-10


@register
class SUR(GalleryEntry):
    name = "sur"
    family = "statistical"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.transitions_: list[dict] | None = None
        self.pooled_corr_: np.ndarray | None = None
        self._loss_field: str | None = None
        self._intercept: bool = False

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        as_of: dt.date | str | None = None,
        intercept: bool = False,
        min_pts_full_cov: int | None = None,
        max_iter: int = 25,
        tol: float = 1e-8,
    ) -> SUR:
        """Fit one FGLS system per development transition, jointly over the
        company's lines of business, on the as_of-sliced upper triangle."""
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = multiline_data(train, loss_field=loss_field)
        self._loss_field = loss_field
        self._intercept = intercept
        c = self.contract_
        n_lob, n_d = c["n_lob"], c["n_d"]
        cum, mask = c["cum"], c["obs_mask"]
        if (cum[mask] <= 0).any():
            raise ValueError(
                f"non-positive cumulative {loss_field!r} cells; "
                "the 1/C variance weighting needs positive cumulatives"
            )
        if min_pts_full_cov is None:
            min_pts_full_cov = n_lob + 2
        p = 2 if intercept else 1

        # pass 1: whitened data per transition + standardized OLS residuals
        # for the pooled cross-line correlation used by thin transitions
        prepared: list[dict] = []
        pooled_resid: list[np.ndarray] = []
        for d in range(n_d - 1):
            pair = mask[0, :, d] & mask[0, :, d + 1]  # masks are lob-aligned
            w_idx = np.nonzero(pair)[0]
            n = len(w_idx)
            if n == 0:
                raise ValueError(
                    f"no origin observes both dev steps {d + 1} and {d + 2}; "
                    "cannot estimate this development transition"
                )
            x0 = cum[:, w_idx, d]  # (n_lob, n)
            y0 = cum[:, w_idx, d + 1]
            root = np.sqrt(x0)
            yw = y0 / root
            xw = np.stack([1.0 / root, root], axis=-1) if intercept else root[..., None]
            tr = {"n": n, "xw": xw, "yw": yw, "x0": x0}
            if n >= 3:
                resid = np.stack(
                    [
                        yw[k] - xw[k] @ np.linalg.lstsq(xw[k], yw[k], rcond=None)[0]
                        for k in range(n_lob)
                    ]
                )
                sd = resid.std(axis=1, ddof=p)
                if (sd > 0).all():
                    pooled_resid.append(resid / sd[:, None])
            prepared.append(tr)

        if pooled_resid:
            stacked = np.hstack(pooled_resid)
            self.pooled_corr_ = _nearest_pd(np.corrcoef(stacked))
        else:
            self.pooled_corr_ = np.eye(n_lob)

        # pass 2: FGLS where the transition supports it, guarded fallbacks below
        transitions: list[dict] = []
        for tr in prepared:
            n, xw, yw = tr["n"], tr["xw"], tr["yw"]
            if n >= max(min_pts_full_cov, p + 1):
                beta, sigma, coef_cov = _fgls(xw, yw, max_iter=max_iter, tol=tol)
                method = "fgls"
            else:
                beta, sigma, coef_cov, method = self._fallback(tr, transitions, p)
            transitions.append(
                {
                    "n": n,
                    "beta": beta,
                    "sigma": _nearest_pd(sigma),
                    "coef_cov": _nearest_pd(coef_cov),
                    "method": method,
                }
            )
        self.transitions_ = transitions
        return self

    def _fallback(
        self, tr: dict, done: list[dict], p: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Per-line WLS slopes with a pooled correlation structure, and the
        Mack tail rule for the variance when the residual df run out."""
        n, xw, yw = tr["n"], tr["xw"], tr["yw"]
        n_lob = yw.shape[0]
        beta = np.stack([np.linalg.lstsq(xw[k], yw[k], rcond=None)[0] for k in range(n_lob)])
        if n - p >= 2:  # n >= 3 without intercept: residual variance is usable
            resid = yw - np.einsum("knp,kp->kn", xw, beta)
            var = (resid**2).sum(axis=1) / (n - p)
            method = "pooled_corr"
        else:
            var = _mack_tail_variance(done, n_lob)
            method = "tail"
        sd = np.sqrt(var)
        sigma = self.pooled_corr_ * np.outer(sd, sd)
        # per-equation WLS coefficient covariance, block-diagonal
        coef_cov = np.zeros((n_lob * p, n_lob * p))
        for k in range(n_lob):
            xtx_inv = np.linalg.inv(xw[k].T @ xw[k])
            coef_cov[k * p : (k + 1) * p, k * p : (k + 1) * p] = var[k] * xtx_inv
        return beta, sigma, coef_cov, method

    def predict(
        self,
        *,
        n_draws: int = 10_000,
        seed: int | None = None,
        param_uncertainty: bool = True,
    ) -> PredictiveDistribution:
        """Simulate ultimates by propagating each origin's latest diagonal
        through the fitted transitions with cross-line correlated errors.

        Per draw, one coefficient vector per transition (parameter risk is
        common across origins); process noise is independent across origins.
        Simulated cumulatives are floored at zero (see card.md)."""
        if self.transitions_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_lob, n_w, n_d = c["n_lob"], c["n_w"], c["n_d"]
        cum, mask = c["cum"], c["obs_mask"]
        p = 2 if self._intercept else 1
        rng = np.random.default_rng(seed)

        beta_draws, noise_chol = [], []
        for tr in self.transitions_:
            flat = tr["beta"].reshape(-1)
            if param_uncertainty:
                z = rng.standard_normal((n_draws, flat.size))
                draws = flat + z @ np.linalg.cholesky(tr["coef_cov"]).T
            else:
                draws = np.tile(flat, (n_draws, 1))
            beta_draws.append(draws.reshape(n_draws, n_lob, p))
            noise_chol.append(np.linalg.cholesky(tr["sigma"]))

        ults = np.empty((n_draws, n_lob, n_w))
        for w in range(n_w):
            devs = np.nonzero(mask[0, w])[0]
            if devs.size == 0:
                raise ValueError(f"origin {c['origin_periods'][w]} has no observations")
            d0 = int(devs[-1])
            state = np.tile(cum[:, w, d0], (n_draws, 1))  # (n_draws, n_lob)
            for d in range(d0, n_d - 1):
                b = beta_draws[d]
                mean = b[:, :, 0] + b[:, :, 1] * state if self._intercept else b[:, :, 0] * state
                eta = rng.standard_normal((n_draws, n_lob)) @ noise_chol[d].T
                state = np.maximum(mean + np.sqrt(np.maximum(state, 0.0)) * eta, 0.0)
            ults[:, :, w] = state

        targets = multiline_targets(c["lobs"], c["origin_periods"])
        return assemble_predictive(ults, targets, units=c["units"])

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to predict()'s targets (per lob x origin, per-lob
        totals, grand total), from the full triangle at the final dev lag."""
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


def _fgls(
    xw: np.ndarray, yw: np.ndarray, *, max_iter: int, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterated feasible GLS on a K-equation SUR system with whitened data.

    xw: (K, n, p) per-equation designs; yw: (K, n) responses.
    Returns (beta (K, p), sigma (K, K) residual covariance, coef_cov (Kp, Kp)).
    """
    n_lob, n, p = xw.shape
    beta = np.stack([np.linalg.lstsq(xw[k], yw[k], rcond=None)[0] for k in range(n_lob)])
    a = np.zeros((n_lob * p, n_lob * p))
    for _ in range(max_iter):
        resid = yw - np.einsum("knp,kp->kn", xw, beta)
        sigma = _nearest_pd((resid @ resid.T) / max(n - p, 1))
        sig_inv = np.linalg.inv(sigma)
        a[:] = 0.0
        rhs = np.zeros(n_lob * p)
        for k in range(n_lob):
            for m in range(n_lob):
                a[k * p : (k + 1) * p, m * p : (m + 1) * p] = sig_inv[k, m] * (xw[k].T @ xw[m])
                rhs[k * p : (k + 1) * p] += sig_inv[k, m] * (xw[k].T @ yw[m])
        new = np.linalg.solve(a, rhs).reshape(n_lob, p)
        done = np.max(np.abs(new - beta)) < tol * (1.0 + np.max(np.abs(beta)))
        beta = new
        if done:
            break
    resid = yw - np.einsum("knp,kp->kn", xw, beta)
    sigma = _nearest_pd((resid @ resid.T) / max(n - p, 1))
    return beta, sigma, np.linalg.inv(a)


def _mack_tail_variance(done: list[dict], n_lob: int) -> np.ndarray:
    """Mack's tail rule per line: sigma^2_d = min(sigma^4_{d-1}/sigma^2_{d-2},
    sigma^2_{d-1}, sigma^2_{d-2}), from the two most recent fitted transitions."""
    variances = [np.diag(tr["sigma"]) for tr in done]
    if not variances:
        raise ValueError("cannot apply the variance tail rule with no earlier transitions")
    if len(variances) == 1:
        return variances[-1]
    v1, v2 = variances[-1], variances[-2]  # d-1, d-2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(v2 > 0, v1**2 / v2, v1)
    return np.minimum(np.minimum(ratio, v1), v2)


def _nearest_pd(mat: np.ndarray) -> np.ndarray:
    """Symmetrize and floor the eigenvalues so downstream Cholesky factors
    exist. Floored dimensions bias correlations toward zero — documented."""
    sym = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= EIG_FLOOR:
        return sym
    scale = max(np.abs(vals).max(), 1.0)
    return (vecs * np.maximum(vals, EIG_FLOOR * scale)) @ vecs.T
