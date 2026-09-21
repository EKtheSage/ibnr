"""The full-matrix multivariate chain ladder - Zhang (2010)'s general form,
estimated with the R ``systemfit`` SUR conventions.

Role in the gallery: ``sur``'s sibling. Where ``sur`` regresses each line's
next cumulative on its OWN current cumulative and carries the cross-line
dependence only in the error covariance, this entry regresses each line's next
cumulative on EVERY line's current cumulative, so a line's development can be
predicted by its neighbours' emergence and not only correlated with it. Per
line ``k``, origin ``w`` and development transition ``d -> d+1``:

    C_{k,w,d+1} = sum_l B_d[k,l] * C_{l,w,d} + e_{k,w,d}
    Var(e_{k,w,d})             = sigma^2_{k,d} * C_{k,w,d}   (Mack's 1/C weighting)
    Corr(e_{k,w,d}, e_{l,w,d}) = R_d[k,l]                     (same origin, cross-line)

``sur`` is the special case where ``B_d`` is diagonal, and a diagonal ``B_d``
holds exactly the volume-weighted chain-ladder factors. Mack's weighting is
applied per equation, so equation ``k`` is divided through by
``sqrt(C_{k,w,d})`` - its OWN line's current cumulative - which leaves the
regressors of the other lines on that same scale.

A full ``B_d`` costs ``K^2`` coefficients per transition where the diagonal form
costs ``K``, and a Schedule P triangle is small. So a transition is estimated as
a system only where it has more origin pairs than twice the line count, and
otherwise falls back to the diagonal volume-weighted chain ladder. On a 10 by 10
square with four lines that means the first transition only. This is the truth
of the method on this data rather than a limitation of the port, and card.md
says so.

Cross-refs: card.md; ``kernels.multiline`` (the shared one-company/many-LOB data
contract, target layout and PredictiveDistribution assembly - this entry never
grows its own data prep); ``gallery/statistical/sur`` (the diagonal sibling);
Zhang (2010), *A general multivariate framework for predicting reserves*;
Henningsen and Hamann (2007), *systemfit: A Package for Estimating Systems of
Simultaneous Equations in R*, Journal of Statistical Software 23(4), whose
defaults the estimator below reproduces.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.multiline import (
    assemble_predictive,
    flatten_with_totals,
    mack_tail_variance,
    multiline_data,
    multiline_targets,
    nearest_pd,
    realized_multiline,
)
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.kernels.rng import cohort_stream
from ibnr.triangle.core import Triangle

#: Smallest reciprocal condition number a matrix may have and still be solved.
#: R's ``solve()`` refuses anything below its ``tol``, and ``systemfit`` leaves
#: that tolerance at machine epsilon; the estimator below uses the same number
#: so that the two implementations fall back on the same transitions.
SOLVE_TOLERANCE = float(np.finfo(float).eps)


def _refuse_singular(mat: np.ndarray, what: str) -> None:
    """Raise ``LinAlgError`` when ``mat`` is singular to working precision.

    ``numpy.linalg.inv`` and ``numpy.linalg.solve`` only raise when a pivot is
    exactly zero, so a matrix that is singular in every meaningful sense - a
    reciprocal condition number of 1e-30 - is inverted quite happily, and the
    answer is rounding error multiplied by 1e30. R refuses it instead, and this
    check is what makes the two agree. The comparison is written so that a NaN
    condition number is refused too.

    Both of the estimator's two solves are checked, as R checks both, and on
    the Schedule P companies either one alone would catch the same transitions:
    a singular residual covariance makes the normal matrix built from its
    inverse ill-conditioned as well. Neither is therefore dead code - they are
    two chances to catch the same thing, and which one fires depends on the
    data.
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        rcond = 1.0 / np.linalg.cond(mat, 1)
    if not rcond > SOLVE_TOLERANCE:
        raise np.linalg.LinAlgError(
            f"{what} is computationally singular "
            f"(reciprocal condition number {rcond:.3g}, tolerance {SOLVE_TOLERANCE:.3g})"
        )


@register
class MCL(GalleryEntry):
    """Zhang (2010)'s full-matrix multivariate chain ladder, one system per
    development transition.

    Fitted state: ``transitions_`` holds one dict per ``d -> d+1`` step (the
    ``K x K`` coefficient matrix ``B``, the whitened residual covariance
    ``sigma`` carrying the cross-line dependence, the coefficient covariance
    ``coef_cov`` for parameter risk, the origin-pair count ``n``, the
    ``method`` that produced it and, when it fell back, ``fallback_reason``);
    ``pooled_corr_`` is the cross-line correlation thin transitions borrow.
    See card.md for the estimation conventions and the limitations.
    """

    name = "mcl"
    family = "statistical"

    def __init__(self) -> None:
        self.contract_: dict | None = None  # multiline_data() dict; None until fit()
        self.transitions_: list[dict] | None = None  # one fitted dict per dev transition
        self.pooled_corr_: np.ndarray | None = None  # (K, K) fallback cross-line corr R_bar
        self._loss_field: str | None = None

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        as_of: dt.date | str | None = None,
        min_obs_mult: int = 2,
    ) -> MCL:
        """Estimate one coefficient matrix per development transition on the
        as_of-sliced upper triangle, jointly over the company's lines.

        ``min_obs_mult`` sets how many origin pairs a transition needs before
        its full ``K x K`` matrix is estimated as a system: strictly more than
        ``min_obs_mult * K``. Below that the transition is the diagonal
        volume-weighted chain ladder. The default of 2 is the reference R
        implementation's.
        """
        if min_obs_mult < 1:
            raise ValueError(
                f"min_obs_mult must be at least 1, got {min_obs_mult}. Each equation of "
                "the system has one column per line, so a transition needs more origin "
                "pairs than lines before its coefficients are identified at all"
            )
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # Single canonical data prep (one company, K aligned LOBs, cumulative).
        #
        # BUILD FIRST, ASSIGN AFTER THE ESTIMATOR SUCCEEDED - fit() must be
        # atomic. The positivity guard below legitimately refuses real cohorts
        # after the contract is built, and assigning contract_ before it leaves
        # a failed refit torn: the new cohort's contract over the old cohort's
        # transitions, which point() and predict() then roll forward under the
        # wrong identity. Same rule as sur and mack.
        contract = multiline_data(train, loss_field=loss_field)
        c = contract
        n_lob, n_d = c["n_lob"], c["n_d"]
        cum, mask = c["cum"], c["obs_mask"]  # cum: (K, n_w, n_d); mask: same shape
        _require_positive_regressors(cum, mask, loss_field, c["lobs"], c["origin_periods"])

        # pass 1: whitened data per transition, plus the standardized own-line
        # OLS residuals that make the pooled cross-line correlation R_bar
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
            x0 = cum[:, w_idx, d]  # (K, n) cumulative at dev d (regressors)
            y0 = cum[:, w_idx, d + 1]  # (K, n) cumulative at dev d+1 (responses)
            prepared.append({"n": n, "x0": x0, "y0": y0})
            # Standardized own-line whitened OLS residuals feed R_bar. Only
            # trustworthy with >= 3 points, so thin steps are skipped here.
            if n >= 3:
                root = np.sqrt(x0)  # (K, n)
                yw = y0 / root  # (K, n) whitened response
                factor = (y0.sum(axis=1) / x0.sum(axis=1))[:, None]  # (K, 1)
                resid = yw - root * factor  # (K, n) whitened residuals
                sd = resid.std(axis=1, ddof=1)  # (K,) residual sd, df = n - 1
                if (sd > 0).all():
                    pooled_resid.append(resid / sd[:, None])  # (K, n) standardized

        # R_bar: correlation of standardized residuals pooled across all
        # estimable transitions -> the assumed cross-line structure for thin
        # steps. Falls back to independence (identity) if nothing was poolable.
        if pooled_resid:
            stacked = np.hstack(pooled_resid)  # (K, sum_n) columns across transitions
            pooled_corr = nearest_pd(np.corrcoef(stacked))  # (K, K)
        else:
            pooled_corr = np.eye(n_lob)

        # pass 2: the system where the transition affords it, the diagonal
        # volume-weighted chain ladder everywhere else
        transitions: list[dict] = []
        for tr in prepared:
            n, x0, y0 = tr["n"], tr["x0"], tr["y0"]
            fitted = None
            if n > min_obs_mult * n_lob:
                try:
                    b, sigma, coef_cov = system_estimate(x0, y0)
                except np.linalg.LinAlgError as exc:
                    reason = f"the system could not be solved: {exc}"
                else:
                    if np.isfinite(b).all():
                        fitted = {
                            "B": b,
                            "sigma": nearest_pd(sigma),
                            "coef_cov": nearest_pd(coef_cov),
                            "method": "system",
                            "fallback_reason": None,
                        }
                    else:
                        reason = "the system returned a non-finite coefficient"
            else:
                reason = (
                    f"{n} origin pair(s), which is not more than "
                    f"min_obs_mult * {n_lob} lines = {min_obs_mult * n_lob}"
                )
            if fitted is None:
                fitted = _fallback(x0, y0, transitions, pooled_corr, reason)
            transitions.append({"n": n, **fitted})

        self.contract_ = contract
        self._loss_field = loss_field
        self.pooled_corr_ = pooled_corr
        self.transitions_ = transitions
        return self

    def cohorts(self) -> list[dict]:
        """This fit's one cohort - the segment identity its contract was built
        from (see :meth:`GalleryEntry.cohorts`)."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        return [dict(self.contract_["segment"])]

    def point(self, segment: Mapping | None = None) -> pd.DataFrame:
        """The point ultimates: each origin's latest observed diagonal rolled
        forward by the fitted matrices, ``C_{d+1} = B_d C_d``.

        Returns the ``kernels.multiline`` target frame with one extra column,
        ``point`` - per-(lob, origin) ultimates, then per-lob totals, then the
        grand total, in the same order ``predict()`` uses. Reserves are the
        caller's subtraction: ultimate minus the latest observed cumulative.
        """
        # a single-cohort fit: accepts None or its own key, refuses anything else
        self.cohort_index(segment)
        if self.transitions_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        ults = self._roll_forward()
        frame = multiline_targets(c["lobs"], c["origin_periods"])
        frame["point"] = flatten_with_totals(ults)
        return frame

    def _roll_forward(self) -> np.ndarray:
        """(n_lob, n_w) point ultimates. One recursion, shared by point()."""
        c = self.contract_
        n_lob, n_w, n_d = c["n_lob"], c["n_w"], c["n_d"]
        cum = c["cum"]
        ults = np.empty((n_lob, n_w))
        for w in range(n_w):
            state = cum[:, w, self._latest_dev(w)]  # (K,) all lines at that origin's latest dev
            for d in range(self._latest_dev(w), n_d - 1):
                state = self.transitions_[d]["B"] @ state
            ults[:, w] = state
        return ults

    def _latest_dev(self, w: int) -> int:
        """Index of origin ``w``'s latest observed development step."""
        c = self.contract_
        devs = np.nonzero(c["obs_mask"][0, w])[0]
        if devs.size == 0:
            raise ValueError(f"origin {c['origin_periods'][w]} has no observations")
        return int(devs[-1])

    def predict(
        self,
        segment: Mapping | None = None,
        *,
        n_draws: int = 10_000,
        seed: int | None = None,
        param_uncertainty: bool = True,
    ) -> PredictiveDistribution:
        """Simulate ultimates by propagating each origin's latest diagonal
        through the fitted matrices with cross-line correlated errors.

        Per draw, one coefficient matrix per transition (parameter risk is
        common across origins); process noise is independent across origins.
        Simulated cumulatives are floored at zero (see card.md).
        """
        # a single-cohort fit: accepts None or its own key, refuses anything else
        self.cohort_index(segment)
        if self.transitions_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_lob, n_w, n_d = c["n_lob"], c["n_w"], c["n_d"]
        cum = c["cum"]
        rng = np.random.default_rng(
            cohort_stream(seed, label="predict", cohorts=self.cohorts(), field=self._loss_field)
        )

        # Pre-draw, per transition: a stack of coefficient matrices (parameter
        # risk) and the Cholesky factor of the whitened residual covariance
        # (process risk). Both are reused across every origin.
        b_draws, noise_chol = [], []
        for tr in self.transitions_:
            b_draws.append(_draw_matrices(tr, n_draws, rng, param_uncertainty))
            noise_chol.append(np.linalg.cholesky(tr["sigma"]))  # (K, K) lower-triangular

        ults = np.empty((n_draws, n_lob, n_w))
        for w in range(n_w):
            # Start each origin at its latest observed diagonal and roll forward.
            d0 = self._latest_dev(w)
            state = np.tile(cum[:, w, d0], (n_draws, 1))  # (n_draws, n_lob)
            for d in range(d0, n_d - 1):
                # Conditional mean C_{d+1} = B_d C_d, per draw.
                mean = np.einsum("nkl,nl->nk", b_draws[d], state)
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

    def realized_ultimates(
        self, full_triangle: Triangle, segment: Mapping | None = None
    ) -> np.ndarray:
        """Outcomes aligned to predict()'s targets (per lob x origin, per-lob
        totals, grand total), from the full triangle at the final dev lag."""
        # a single-cohort fit: accepts None or its own key, refuses anything else
        self.cohort_index(segment)
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


def system_estimate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One feasible-GLS step on the K-equation full-matrix system.

    ``x`` and ``y`` are ``(K, n)``: each line's cumulative at the current and
    the next development step, over the ``n`` origins that observe both.
    Equation ``k`` regresses ``y_k`` on all of ``x_1 .. x_K`` and is whitened by
    ``1 / sqrt(x_k)``, its own line's current cumulative, which is what Mack's
    variance assumption asks for.

    The three steps are the R ``systemfit`` defaults for ``method = "SUR"``,
    and the entry ties out to the reference R implementation because of them:

    1. OLS per equation.
    2. Residual covariance from those OLS residuals, uncentred, with the
       ``geomean`` denominator ``sqrt((n - p_k)(n - p_m))``. Every equation here
       has the same ``K`` columns, so that denominator is ``n - K``.
    3. ONE GLS solve - no iteration to a fixed point. The stacked system has
       covariance ``Omega = Sigma (x) I_n``, so the normal equations
       block-decompose: block ``(k, m)`` of ``X' Omega^-1 X`` is
       ``Sigma^-1[k,m] * (X_k' X_m)`` and block ``k`` of the right-hand side is
       ``sum_m Sigma^-1[k,m] * (X_k' y_m)``.

    Both matrices this inverts are refused when they are singular to working
    precision, by the same rule R applies (see :func:`_refuse_singular`), and
    the caller then falls back to the diagonal chain ladder. That is not a
    detail: a line whose paid loss has stopped developing has an exactly zero
    residual in every origin, which makes the residual covariance singular, and
    it happens on 6 of the 82 Schedule P companies this entry ties out against.
    Inverting such a matrix anyway succeeds numerically and returns coefficients
    of the wrong order - on one of those companies it turned a reserve of 2600
    into 558.

    Returns ``(B (K, K), sigma (K, K), coef_cov (K*K, K*K))``. ``B[k, l]`` is
    equation ``k``'s coefficient on line ``l``; ``coef_cov`` is row-major over
    ``(k, l)``, matching ``B.reshape(-1)``. ``sigma`` is the covariance of the
    GLS residuals under the same ``n - K`` denominator - it is what the draws
    use and it does not move the point estimate.
    """
    n_lob, n = x.shape
    root = np.sqrt(x)  # (K, n)
    # xw[k][i, l] = x[l, i] / sqrt(x[k, i]): equation k's design, (n, K)
    xw = np.transpose(x[None, :, :] / root[:, None, :], (0, 2, 1))  # (K, n, K)
    yw = y / root  # (K, n) whitened responses

    # step 1: per-equation OLS
    beta_ols = np.stack([np.linalg.lstsq(xw[k], yw[k], rcond=None)[0] for k in range(n_lob)])
    resid = yw - np.einsum("knl,kl->kn", xw, beta_ols)  # (K, n)
    # step 2: uncentred residual covariance, geomean denominator = n - K here
    sigma_ols = (resid @ resid.T) / (n - n_lob)  # (K, K)
    _refuse_singular(sigma_ols, "the residual covariance")
    sig_inv = np.linalg.inv(sigma_ols)  # (K, K)

    # step 3: one GLS solve, assembled block by block from the Kronecker structure
    size = n_lob * n_lob
    a = np.zeros((size, size))  # (K^2, K^2) = X' Omega^-1 X
    rhs = np.zeros(size)  # (K^2,) = X' Omega^-1 y
    for k in range(n_lob):
        for m in range(n_lob):
            a[k * n_lob : (k + 1) * n_lob, m * n_lob : (m + 1) * n_lob] = sig_inv[k, m] * (
                xw[k].T @ xw[m]
            )
            rhs[k * n_lob : (k + 1) * n_lob] += sig_inv[k, m] * (xw[k].T @ yw[m])
    _refuse_singular(a, "the system's normal matrix")
    b = np.linalg.solve(a, rhs).reshape(n_lob, n_lob)  # (K, K)
    coef_cov = np.linalg.inv(a)  # (K^2, K^2)

    resid_gls = yw - np.einsum("knl,kl->kn", xw, b)  # (K, n)
    sigma = (resid_gls @ resid_gls.T) / (n - n_lob)  # (K, K)
    return b, sigma, coef_cov


def volume_weighted(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The diagonal fallback: ``diag(sum(y_k) / sum(x_k))``, the per-line
    volume-weighted chain-ladder factors, with every cross-line coefficient
    fixed at zero. ``x`` and ``y`` are ``(K, n)`` as in :func:`system_estimate`.
    """
    return np.diag(y.sum(axis=1) / x.sum(axis=1))


def _fallback(
    x: np.ndarray,
    y: np.ndarray,
    done: list[dict],
    pooled_corr: np.ndarray,
    reason: str,
) -> dict:
    """The diagonal volume-weighted chain ladder, with a covariance for the
    draws borrowed from the pool where the transition is too thin to estimate
    its own.

    The point estimate here is the reference implementation's fallback exactly.
    What the reference has no need of, and this entry does, is a covariance for
    ``predict()``: each line keeps its own whitened residual variance while any
    residual degrees of freedom remain, and the cross-line correlation is
    imported from ``R_bar``. With no degrees of freedom left the variances come
    from Mack's tail rule over the transitions already fitted.

    ``coef_cov`` is ``(K, K)`` here, not ``(K^2, K^2)``: the off-diagonal
    coefficients are fixed at zero rather than estimated, so they carry no
    estimation error and there is nothing to draw for them. ``method`` is what
    tells a reader which of the two shapes is in the dict.
    """
    n_lob, n = x.shape
    b = volume_weighted(x, y)
    root = np.sqrt(x)  # (K, n)
    resid = y / root - root * np.diag(b)[:, None]  # (K, n) own-line whitened residuals
    # n >= 3 leaves residual degrees of freedom, so the transition's own
    # variance is usable; below that, extrapolate with Mack's tail rule over
    # the transitions already fitted.
    usable_df = n - 1 >= 2
    var = (resid**2).sum(axis=1) / (n - 1) if usable_df else mack_tail_variance(done)  # (K,)
    sd = np.sqrt(var)  # (K,)
    # Keep each line's own marginal variance, borrow only the correlation.
    sigma = pooled_corr * np.outer(sd, sd)  # (K, K)
    # Each factor's estimation variance is var_k * (X_k' X_k)^-1 on the whitened
    # scale, and the whitened own-line design has X_k' X_k = sum(x_k).
    coef_cov = np.diag(var / x.sum(axis=1))  # (K, K)
    return {
        "B": b,
        "sigma": nearest_pd(sigma),
        "coef_cov": nearest_pd(coef_cov),
        "method": "volume_weighted",
        "fallback_reason": reason,
    }


def _draw_matrices(
    tr: dict, n_draws: int, rng: np.random.Generator, param_uncertainty: bool
) -> np.ndarray:
    """(n_draws, K, K) coefficient matrices for one transition.

    Without parameter risk every draw is the point estimate. With it, a system
    transition draws the whole flattened matrix from ``N(vec(B), coef_cov)``,
    which keeps the estimated cross-line coefficients' own correlation; a
    fallback transition draws only the K diagonal factors and leaves the
    off-diagonal zeros alone, because those coefficients were never estimated.
    """
    b = tr["B"]
    n_lob = b.shape[0]
    if not param_uncertainty:
        return np.tile(b, (n_draws, 1, 1))
    if tr["method"] == "system":
        flat = b.reshape(-1)  # (K^2,) row-major over (equation, line)
        z = rng.standard_normal((n_draws, flat.size))
        draws = flat + z @ np.linalg.cholesky(tr["coef_cov"]).T
        return draws.reshape(n_draws, n_lob, n_lob)
    diag = np.diag(b)  # (K,)
    z = rng.standard_normal((n_draws, n_lob))
    factors = diag + z @ np.linalg.cholesky(tr["coef_cov"]).T  # (n_draws, K)
    out = np.zeros((n_draws, n_lob, n_lob))
    idx = np.arange(n_lob)
    out[:, idx, idx] = factors
    return out


def _require_positive_regressors(
    cum: np.ndarray,
    mask: np.ndarray,
    loss_field: str,
    lobs: list,
    origins: list[dt.date],
) -> None:
    """Refuse a non-positive cumulative that a development transition divides by.

    Mack's variance is proportional to C, so each equation is whitened by
    ``1 / sqrt(C)`` of its own line's current cumulative, and the diagonal
    fallback divides by the column total of the same cells. Those cells are
    exactly the ones with an observed successor.

    A cell WITHOUT an observed successor - each origin's latest diagonal - is
    never divided by. It only starts that origin's recursion, and a zero there
    is carried forward by the other lines' coefficients, which is what the
    reference R implementation does too. Refusing it would throw away real
    cohorts: two of the 82 Schedule P companies this entry ties out against
    report zero paid at 12 months on their newest accident year. ``sur``, which
    has no cross-line coefficient to carry such a cell, refuses every
    non-positive cumulative instead; card.md records the difference.
    """
    # a regressor is a cell whose origin also observes the next dev step
    pair = mask[0, :, :-1] & mask[0, :, 1:]  # (n_w, n_d - 1)
    regressors = cum[:, :, :-1][:, pair]  # (K, n_pairs)
    if (regressors > 0).all():
        return
    k_bad, w_bad, d_bad = np.nonzero((cum[:, :, :-1] <= 0) & pair[None, :, :])
    cells = [
        f"({lobs[k]}, {origins[w].isoformat()}, dev step {d + 1})"
        for k, w, d in list(zip(k_bad, w_bad, d_bad, strict=True))[:5]
    ]
    more = "" if len(k_bad) <= 5 else f" and {len(k_bad) - 5} more"
    raise ValueError(
        f"non-positive cumulative {loss_field!r} at {', '.join(cells)}{more}. "
        "Each development transition divides by its own line's current cumulative, "
        "so every cell with an observed successor must be positive"
    )
