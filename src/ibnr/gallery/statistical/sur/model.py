"""Seemingly unrelated regression across lines of business - the multivariate
chain ladder of Zhang (2010), estimated by hand-rolled feasible GLS.

Role in the gallery: the frequentist stochastic-dependence baseline against
which the NN and Bayesian multiline entries are judged. One SUR *system* is fit
per development transition ``d -> d+1``, jointly over the ``K`` lines of business
of a single company. Zhang's regression, per line ``k`` and origin ``w`` (the
no-intercept default this entry ships):

    C_{k,w,d+1} = b_{k,d} * C_{k,w,d} + e_{k,w,d}
    Var(e_{k,w,d})            = sigma^2_{k,d} * C_{k,w,d}   (Mack's 1/C weighting)
    Corr(e_{k,w,d}, e_{l,w,d}) = R_d[k,l]                    (same origin, cross-line)

At the no-intercept default the per-line slope ``b_{k,d}`` is exactly the
volume-weighted chain-ladder development factor, so this is "chain ladder + a
contemporaneous cross-line error correlation", estimated jointly rather than
line by line. ``intercept=True`` switches the design to Zhang's general form
``[1, C_d]`` at one extra parameter per line per transition.

Cross-refs: card.md (model card, small-sample ladder, limitations);
``kernels.multiline`` (the shared one-company/many-LOB data contract, target
layout, and PredictiveDistribution assembly - this entry never grows its own
data prep); Zhang (2010), *A general multivariate framework for predicting
reserves*, and Prohl & Schmidt on the multivariate chain ladder.
"""

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
    """Zhang (2010) multivariate chain ladder as a SUR system per dev transition.

    Fitted state: ``transitions_`` holds one dict per ``d -> d+1`` step (the
    per-line slopes ``beta``, the whitened residual covariance ``sigma`` that
    carries the cross-line dependence, and the coefficient covariance
    ``coef_cov`` for parameter risk); ``pooled_corr_`` is the fallback cross-line
    correlation ``R_bar`` used by transitions too thin to estimate their own.
    See card.md for the estimation ladder and its documented limitations.
    """

    name = "sur"
    family = "statistical"

    def __init__(self) -> None:
        self.contract_: dict | None = None  # multiline_data() dict; None until fit()
        self.transitions_: list[dict] | None = None  # one fitted dict per dev transition
        self.pooled_corr_: np.ndarray | None = None  # (K, K) fallback cross-line corr R_bar
        self._loss_field: str | None = None
        self._intercept: bool = False  # False -> chain-ladder slope; True -> Zhang [1, C_d]

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
        company's lines of business, on the as_of-sliced upper triangle.

        Args mirror the small-sample ladder in card.md: ``min_pts_full_cov``
        (default ``K + 2``) is the origin-pair count below which a transition
        falls back from full FGLS to the pooled-correlation / Mack-tail rules;
        ``max_iter``/``tol`` govern the FGLS fixed-point iteration.
        """
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # Single canonical data prep (one company, K aligned LOBs, cumulative).
        #
        # BUILD FIRST, ASSIGN AFTER THE ESTIMATOR SUCCEEDED - fit() must be
        # atomic. The positivity guard and the per-transition checks below
        # legitimately refuse real cohorts AFTER the contract is built, and
        # assigning contract_ before them leaves a failed refit TORN: the new
        # cohort's contract over the old cohort's transitions, which predict()
        # then happily rolls forward under the wrong identity. See mack.
        contract = multiline_data(train, loss_field=loss_field)
        c = contract
        n_lob, n_d = c["n_lob"], c["n_d"]
        cum, mask = c["cum"], c["obs_mask"]  # cum: (K, n_w, n_d); mask: same shape
        # Mack's variance is proportional to C, so whitening divides by sqrt(C);
        # a zero/negative cumulative would blow up 1/sqrt(C). Guard up front.
        if (cum[mask] <= 0).any():
            raise ValueError(
                f"non-positive cumulative {loss_field!r} cells; "
                "the 1/C variance weighting needs positive cumulatives"
            )
        if min_pts_full_cov is None:
            min_pts_full_cov = n_lob + 2
        p = 2 if intercept else 1  # params per line per transition: [C_d] or [1, C_d]

        # pass 1: whitened data per transition + standardized OLS residuals
        # for the pooled cross-line correlation R_bar used by thin transitions
        prepared: list[dict] = []
        pooled_resid: list[np.ndarray] = []
        for d in range(n_d - 1):
            # Origins observing BOTH ends of the step contribute a regression
            # pair; the mask is lob-aligned (multiline_data enforces it), so
            # line 0's mask stands for all lines.
            pair = mask[0, :, d] & mask[0, :, d + 1]
            w_idx = np.nonzero(pair)[0]
            n = len(w_idx)  # usable origin pairs for this transition
            if n == 0:
                raise ValueError(
                    f"no origin observes both dev steps {d + 1} and {d + 2}; "
                    "cannot estimate this development transition"
                )
            x0 = cum[:, w_idx, d]  # (K, n) cumulative at dev d (regressor)
            y0 = cum[:, w_idx, d + 1]  # (K, n) cumulative at dev d+1 (response)
            # Whiten by 1/sqrt(C_d): under Var(e) = sigma^2 * C_d this makes each
            # equation homoskedastic, turning weighted CL into ordinary OLS.
            root = np.sqrt(x0)  # (K, n)
            yw = y0 / root  # (K, n) whitened response
            # Whitened design per line: (K, n, p). No-intercept -> column sqrt(C)
            # (slope only); intercept -> [1/sqrt(C), sqrt(C)] = whitened [1, C_d].
            xw = np.stack([1.0 / root, root], axis=-1) if intercept else root[..., None]
            tr = {"n": n, "xw": xw, "yw": yw, "x0": x0}
            # Standardized whitened OLS residuals feed the pooled correlation.
            # Only trustworthy with >= 3 points, so thin steps are skipped here.
            if n >= 3:
                resid = np.stack(
                    [
                        yw[k] - xw[k] @ np.linalg.lstsq(xw[k], yw[k], rcond=None)[0]
                        for k in range(n_lob)
                    ]
                )  # (K, n) per-line whitened residuals
                sd = resid.std(axis=1, ddof=p)  # (K,) residual sd, df = n - p
                if (sd > 0).all():
                    pooled_resid.append(resid / sd[:, None])  # (K, n) standardized
            prepared.append(tr)

        # R_bar: correlation of standardized residuals pooled across all
        # estimable transitions -> the assumed cross-line structure for thin
        # steps. Falls back to independence (identity) if nothing was poolable.
        if pooled_resid:
            stacked = np.hstack(pooled_resid)  # (K, sum_n) columns across transitions
            pooled_corr = _nearest_pd(np.corrcoef(stacked))  # (K, K)
        else:
            pooled_corr = np.eye(n_lob)

        # pass 2: FGLS where the transition supports it, guarded fallbacks below
        transitions: list[dict] = []
        for tr in prepared:
            n, xw, yw = tr["n"], tr["xw"], tr["yw"]
            # Full FGLS needs enough origin pairs to estimate the K x K residual
            # covariance; otherwise drop to the pooled-corr / tail fallback.
            if n >= max(min_pts_full_cov, p + 1):
                beta, sigma, coef_cov = _fgls(xw, yw, max_iter=max_iter, tol=tol)
                method = "fgls"
            else:
                beta, sigma, coef_cov, method = self._fallback(tr, transitions, p, pooled_corr)
            transitions.append(
                {
                    "n": n,
                    "beta": beta,  # (K, p) per-line slopes
                    # PD-repair both covariances so predict()'s Cholesky factors exist
                    "sigma": _nearest_pd(sigma),  # (K, K) whitened residual cov
                    "coef_cov": _nearest_pd(coef_cov),  # (Kp, Kp) coefficient cov
                    "method": method,  # "fgls" | "pooled_corr" | "tail"
                }
            )
        self.contract_ = contract
        self._loss_field = loss_field
        self._intercept = intercept
        self.pooled_corr_ = pooled_corr
        self.transitions_ = transitions
        return self

    def _fallback(
        self, tr: dict, done: list[dict], p: int, pooled_corr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Per-line WLS slopes with a pooled correlation structure, and the
        Mack tail rule for the variance when the residual df run out.

        Thin transitions cannot support a jointly estimated K x K covariance, so
        the slopes are estimated per line (whitened OLS = weighted chain ladder)
        and the cross-line dependence is imported from the pooled ``R_bar``;
        variances are the transition's own if any residual df remain, else Mack's
        tail extrapolation. Returns (beta, sigma, coef_cov, method), same shapes
        as ``_fgls``.
        """
        n, xw, yw = tr["n"], tr["xw"], tr["yw"]
        n_lob = yw.shape[0]
        # Per-line whitened OLS: (K, p) slopes, equivalent to CL factors per line.
        beta = np.stack([np.linalg.lstsq(xw[k], yw[k], rcond=None)[0] for k in range(n_lob)])
        if n - p >= 2:  # n >= 3 without intercept: residual variance is usable
            resid = yw - np.einsum("knp,kp->kn", xw, beta)  # (K, n)
            var = (resid**2).sum(axis=1) / (n - p)  # (K,) own whitened residual var
            method = "pooled_corr"
        else:
            # No residual df left -> extrapolate the last two transitions' vars.
            var = _mack_tail_variance(done)  # (K,)
            method = "tail"
        sd = np.sqrt(var)  # (K,)
        # Rebuild the K x K covariance as R_bar scaled by own sds: keep each
        # line's marginal variance, borrow only the correlation from the pool.
        # (R_bar arrives as an argument: during fit() it is still a local -
        # nothing is stamped on the entry until the whole estimation succeeds.)
        sigma = pooled_corr * np.outer(sd, sd)  # (K, K)
        # Coefficient covariance is block-diagonal here: each line's WLS slope
        # var(k) * (X'X)^-1 sits on its own block, no cross-line coupling (unlike
        # FGLS, where the off-diagonals are non-zero).
        coef_cov = np.zeros((n_lob * p, n_lob * p))  # (Kp, Kp)
        for k in range(n_lob):
            xtx_inv = np.linalg.inv(xw[k].T @ xw[k])  # (p, p)
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

        # Pre-draw, per transition: a stack of coefficient vectors (parameter
        # risk) and the Cholesky factor of the whitened residual covariance
        # (process risk). Both are reused across every origin.
        beta_draws, noise_chol = [], []
        for tr in self.transitions_:
            flat = tr["beta"].reshape(-1)  # (Kp,) row-major over (line, param)
            if param_uncertainty:
                # Sample beta ~ N(beta_hat, coef_cov): flat + z L' with L = chol(cov).
                z = rng.standard_normal((n_draws, flat.size))  # (n_draws, Kp)
                draws = flat + z @ np.linalg.cholesky(tr["coef_cov"]).T  # (n_draws, Kp)
            else:
                draws = np.tile(flat, (n_draws, 1))  # process-only: reuse point est
            beta_draws.append(draws.reshape(n_draws, n_lob, p))  # (n_draws, K, p)
            noise_chol.append(np.linalg.cholesky(tr["sigma"]))  # (K, K) lower-triangular

        ults = np.empty((n_draws, n_lob, n_w))
        for w in range(n_w):
            # Start each origin at its latest observed diagonal and roll forward.
            devs = np.nonzero(mask[0, w])[0]
            if devs.size == 0:
                raise ValueError(f"origin {c['origin_periods'][w]} has no observations")
            d0 = int(devs[-1])  # index of the latest observed dev step
            state = np.tile(cum[:, w, d0], (n_draws, 1))  # (n_draws, n_lob)
            for d in range(d0, n_d - 1):
                b = beta_draws[d]  # (n_draws, K, p)
                # Conditional mean C_{d+1} = b0 + b1*C_d (intercept) or b*C_d.
                mean = b[:, :, 0] + b[:, :, 1] * state if self._intercept else b[:, :, 0] * state
                # Cross-line correlated shock on the whitened scale, (n_draws, K).
                eta = rng.standard_normal((n_draws, n_lob)) @ noise_chol[d].T
                # Un-whiten by sqrt(C_d) (Mack's variance is proportional to C);
                # floor at 0 since additive noise can push a small book negative.
                state = np.maximum(mean + np.sqrt(np.maximum(state, 0.0)) * eta, 0.0)
            ults[:, :, w] = state  # simulated ultimate = cumulative at last dev

        targets = multiline_targets(c["lobs"], c["origin_periods"])
        # assemble_predictive derives per-lob and grand totals as row-sums of the
        # same draws, so diversification stays coherent.
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

    Implements Zhang (2010)'s estimator directly. The stacked SUR system has
    covariance ``Omega = Sigma (x) I_n`` (Kronecker), so the GLS normal
    equations block-decompose: block (k, m) of ``X' Omega^-1 X`` is
    ``Sigma^-1[k,m] * (X_k' X_m)`` and block k of the rhs is
    ``sum_m Sigma^-1[k,m] * (X_k' y_m)``. We alternate: estimate beta given
    Sigma, re-estimate Sigma from the residuals, until beta stops moving.
    (At Sigma = I this reduces to per-line OLS, i.e. the chain ladder.)

    xw: (K, n, p) per-equation designs; yw: (K, n) responses.
    Returns (beta (K, p), sigma (K, K) residual covariance, coef_cov (Kp, Kp)).
    """
    n_lob, n, p = xw.shape
    # Seed with per-line OLS (the Sigma = I solution).
    beta = np.stack([np.linalg.lstsq(xw[k], yw[k], rcond=None)[0] for k in range(n_lob)])
    a = np.zeros((n_lob * p, n_lob * p))  # (Kp, Kp) GLS normal matrix X' Omega^-1 X
    for _ in range(max_iter):
        resid = yw - np.einsum("knp,kp->kn", xw, beta)  # (K, n) whitened residuals
        sigma = _nearest_pd((resid @ resid.T) / max(n - p, 1))  # (K, K) residual cov
        sig_inv = np.linalg.inv(sigma)  # (K, K) = Sigma^-1
        a[:] = 0.0
        rhs = np.zeros(n_lob * p)  # (Kp,) X' Omega^-1 y
        # Assemble the block GLS normal equations from the Kronecker structure.
        for k in range(n_lob):
            for m in range(n_lob):
                a[k * p : (k + 1) * p, m * p : (m + 1) * p] = sig_inv[k, m] * (xw[k].T @ xw[m])
                rhs[k * p : (k + 1) * p] += sig_inv[k, m] * (xw[k].T @ yw[m])
        new = np.linalg.solve(a, rhs).reshape(n_lob, p)  # (K, p) updated GLS beta
        # Relative convergence: scale tol by the current coefficient magnitude.
        done = np.max(np.abs(new - beta)) < tol * (1.0 + np.max(np.abs(beta)))
        beta = new
        if done:
            break
    # Final Sigma at the converged beta; coef_cov = (X' Omega^-1 X)^-1 is the
    # FGLS asymptotic coefficient covariance (Zhang's parameter risk).
    resid = yw - np.einsum("knp,kp->kn", xw, beta)
    sigma = _nearest_pd((resid @ resid.T) / max(n - p, 1))
    return beta, sigma, np.linalg.inv(a)


def _mack_tail_variance(done: list[dict]) -> np.ndarray:
    """Mack's tail rule per line: sigma^2_d = min(sigma^4_{d-1}/sigma^2_{d-2},
    sigma^2_{d-1}, sigma^2_{d-2}), from the two most recent fitted transitions.

    Returns a (K,) variance vector; used only when a transition has no residual
    df of its own (the ``tail`` branch of ``_fallback``).
    """
    variances = [np.diag(tr["sigma"]) for tr in done]  # each (K,) diag of a fitted cov
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
    exist. Floored dimensions bias correlations toward zero - documented."""
    sym = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= EIG_FLOOR:
        return sym
    scale = max(np.abs(vals).max(), 1.0)
    return (vecs * np.maximum(vals, EIG_FLOOR * scale)) @ vecs.T
