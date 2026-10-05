"""The multivariate chain ladder's estimator, shared by every entry that needs one.

Moved out of ``gallery/statistical/mcl`` unchanged so that an entry in another
family can use the same estimate without importing a sibling entry: decision 5
puts every algorithm in ``kernels/`` once, and ``mcl`` and ``tlrn`` (whose
``member="mcl_blend"`` blends the network with this forecast) now both call it.
``mcl/model.py`` imports :func:`system_estimate` and :func:`volume_weighted` from
here, so the names stay importable from where they were.

:func:`point_grid` is the one new piece: the point forecast of a whole company
from the cells visible at a date, which is what a blend needs and the entry's
``predict`` does not (it simulates).
"""

from __future__ import annotations

import numpy as np

__all__ = ["SOLVE_TOLERANCE", "point_grid", "system_estimate", "volume_weighted"]

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


def point_grid(
    cum: np.ndarray, observed: np.ndarray, *, min_obs_mult: int = 2
) -> tuple[np.ndarray, list[str]]:
    """Roll every origin's latest observed cumulative forward to the last lag.

    ``cum`` is ``(K, n_w, n_d)`` cumulative amounts of the ``K`` lines a company
    writes, and ``observed`` the ``(n_w, n_d)`` cells visible at the date, the
    same for every line. Returns ``(grid, methods)``: the grid carries the
    observed cumulatives where they exist and the forecast elsewhere, and
    ``methods`` names how each development step was estimated, ``"system"`` (the
    full-matrix estimate), ``"volume_weighted"`` (the diagonal chain ladder) or
    ``"flat"`` (no origin pair reaches the step, so its factor matrix is the
    identity and the forecast carries the balance across it unchanged).

    The regressors of a step are its origin pairs whose every line has a positive
    cumulative at the earlier end, because the estimate whitens by that amount; a
    pair that fails is left out of that step rather than refused, which is the
    difference from the ``mcl`` entry, whose ``fit`` refuses such a cohort. A
    step is estimated as a system only with more pairs than ``min_obs_mult`` times
    the number of lines, the entry's own rule, and falls back to the diagonal
    chain ladder when it has fewer or the system cannot be solved.
    """
    n_lob, n_w, n_d = cum.shape
    factors: list[np.ndarray] = []
    methods: list[str] = []
    for d in range(n_d - 1):
        pair = observed[:, d] & observed[:, d + 1]
        x, y = cum[:, pair, d], cum[:, pair, d + 1]
        usable = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0) & (x > 0).all(axis=0)
        x, y = x[:, usable], y[:, usable]
        b, method = None, "flat"
        if x.shape[1] > min_obs_mult * n_lob:
            try:
                solved = system_estimate(x, y)[0]
            except np.linalg.LinAlgError:
                solved = None
            if solved is not None and np.isfinite(solved).all():
                b, method = solved, "system"
        if b is None and x.shape[1] > 0:
            b, method = volume_weighted(x, y), "volume_weighted"
        factors.append(np.eye(n_lob) if b is None else b)
        methods.append(method)

    grid = np.where(observed[None], cum, np.nan)
    for w in range(n_w):
        seen = np.nonzero(observed[w])[0]
        if seen.size == 0:
            continue
        state = cum[:, w, seen[-1]]
        for d in range(int(seen[-1]), n_d - 1):
            state = factors[d] @ state
            grid[:, w, d + 1] = state
    return grid, methods
