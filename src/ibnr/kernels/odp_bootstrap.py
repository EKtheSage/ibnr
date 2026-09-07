"""England & Verrall's over-dispersed Poisson bootstrap, over a dense grid.

The residual bootstrap behind ``ChainLadder::BootChainLadder`` in R and behind
England & Verrall (2002, section 8): fit the chain ladder, take Pearson
residuals against its fitted incrementals, resample them into pseudo-triangles,
refit, and add over-dispersed Poisson process noise to the projection. It is a
*diagonal generator* in ``kernels/cdr.py``'s vocabulary - it says what the
payments over the next development step might be - and it knows nothing about
what is done with them afterwards.

Free functions over plain arrays, deliberately: nothing here takes a
``Triangle`` or a ``MackFit``, so the arithmetic can be read against R's source
and tested without either. ``kernels/cdr.py`` is the only caller today.

R's source is the reference implementation (``R/BootstrapReserve.R``, Nigel de
Silva / Giuseppe Crupi / Markus Gesmann) and this file mirrors it step for step.
The four places it deviates, all documented at the point they happen:

1. **Negative observed increments are refused**, where R takes ``sign(x)`` and
   ``sqrt(abs(m))`` and carries on. The ODP family is defined on non-negative
   increments (``kernels.contract.odp_stan_data`` refuses them too, and
   ``england_verrall_odp`` inherits that limit), so this is the family's own
   boundary rather than an implementation shortcut. About half the Schedule P
   mart's paid cohorts fall outside it.
2. **The residual pool and the parameter count are general**, ``n_w x n_d``
   run-off staircase rather than R's square block: R restricts to the last
   ``n`` origins so its ``nobs = n(n+1)/2`` and ``p = 2n - 1`` formulas hold.
   Ours count the observed cells and use ``p = n_w + n_d - 1``, which is the
   same number on a square triangle and the ODP GLM's actual parameter count
   otherwise.
3. **A pseudo development step with non-positive volume takes factor 1.0.** R
   does this only for the exact ``0/0`` case (``out[is.nan(out)] <- 1``) and
   lets a negative volume through to a finite, wrong-signed factor. Both are
   unreachable on data that passes (1); the wider guard is the safer default.
4. **``od_poisson`` is ``phi * Poisson(mu / phi)``**, the England & Verrall
   construction and the one ``england_verrall_odp.predict`` already draws, where
   R's ``rpois.od`` uses a negative binomial with the same first two moments
   (mean ``mu``, variance ``phi * mu``) and integer support. Same moments, so
   the CDR standard error agrees to Monte Carlo error either way; the two
   differ in the tail, and ours cannot drift from the gallery entry's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ibnr.kernels.densities import odp_draw

#: process laws for the simulated increment. Both match the over-dispersed
#: Poisson's first two moments (mean ``mu``, variance ``phi * mu``) and differ
#: in support and tail, exactly as ``kernels.mack.PROCESS_LAWS`` do for Mack.
ODP_PROCESS_LAWS = ("od_poisson", "gamma")


@dataclass(frozen=True)
class ODPBootstrapFit:
    """The deterministic half of the bootstrap: what every pseudo-triangle is
    resampled *from*.

    ``fitted`` are the chain-ladder fitted incrementals - equivalently the
    cross-classified Poisson MLE's fitted means (Renshaw & Verrall 1998), which
    is why the ODP GLM and the chain ladder give the same reserve and why
    ``tests/test_odp_bootstrap.py`` checks these against
    ``england_verrall_odp.odp_mle_fitted``'s iterative proportional fit - two
    independent routes to one quantity.
    """

    inc: np.ndarray  # (n_w, n_d) observed increments, NaN outside the triangle
    fitted: np.ndarray  # (n_w, n_d) chain-ladder fitted increments, NaN outside
    residuals: np.ndarray  # (n_w, n_d) DoF-adjusted Pearson residuals, NaN off the pool
    obs_mask: np.ndarray  # (n_w, n_d) bool
    pool_mask: np.ndarray  # (n_w, n_d) bool - observed AND fitted > 0
    latest_dev: np.ndarray  # (n_w,) 0-based dev index of each origin's diagonal
    phi: float  # Pearson scale, sum(r^2) / (n_cells - n_params)
    n_cells: int  # observed cells (R's `nobs`)
    n_params: int  # n_w + n_d - 1 (R's `2n - 1` on a square triangle)

    @property
    def n_w(self) -> int:
        return self.inc.shape[0]

    @property
    def n_d(self) -> int:
        return self.inc.shape[1]


def fit_odp_bootstrap(
    cum: np.ndarray,
    obs_mask: np.ndarray,
    latest_dev: np.ndarray,
    f: np.ndarray,
) -> ODPBootstrapFit:
    """Fitted incrementals, Pearson scale and adjusted residuals of one cohort.

    ``f`` is supplied rather than re-estimated so the bootstrap and everything
    else built on the same cohort share one set of volume-weighted development
    factors; passing ``MackFit.f`` is what the CDR does.

    Fitted values come from R's backwards recursion (``getExpected``): project
    each origin's ultimate off its own diagonal, divide back down the ultimate
    factors, difference. With ``ultdf[j] = prod_{t>=j} f[t]``,

        ult[i]         = C[i, k_i] * ultdf[k_i]
        Chat[i, j]     = ult[i] / ultdf[j]
        mhat[i, j]     = Chat[i, j] - Chat[i, j-1]

    Non-negative observed increments make every ``f[j] >= 1`` (each pair
    origin's cumulative is non-decreasing, so the column totals are), hence
    every ``mhat >= 0`` - which is why R's ``abs()`` guards are absent here
    rather than merely omitted.

    A cell with ``mhat == 0`` carries no Pearson information and is dropped from
    the residual pool. It can only arise from an identically-zero development
    column, in which case the observed increment there is zero too (the fitted
    column margins match the observed ones); an observed non-zero against a zero
    fit would mean the fit is degenerate and is refused by name.
    """
    cum = np.asarray(cum, dtype=float)
    obs_mask = np.asarray(obs_mask, dtype=bool)
    latest_dev = np.asarray(latest_dev, dtype=int)
    f = np.asarray(f, dtype=float)
    n_w, n_d = cum.shape
    if n_d < 2:
        raise ValueError("the ODP bootstrap needs at least two development steps")

    inc = np.full((n_w, n_d), np.nan)
    inc[:, 0] = np.where(obs_mask[:, 0], cum[:, 0], np.nan)
    prev = np.where(obs_mask[:, :-1], cum[:, :-1], np.nan)
    inc[:, 1:] = np.where(obs_mask[:, 1:], cum[:, 1:] - prev, np.nan)
    negative = obs_mask & (inc < 0)
    if negative.any():
        where = ", ".join(
            f"(origin index {i}, dev step {j + 1})"
            for i, j in zip(*np.nonzero(negative), strict=True)
        )
        raise ValueError(
            f"{int(negative.sum())} negative incremental cell(s) at {where}. The "
            "over-dispersed Poisson family is defined on non-negative increments - its "
            "Pearson residual divides by a fitted mean that a negative increment can drive "
            "below zero, and its process draw has no negative mean - so the ODP bootstrap "
            "refuses this cohort outright rather than reflecting it through sign(). Mack's "
            "conditional moments carry no such restriction: the mack diagonal generator "
            "(and one_year_cdr) still answer here"
        )

    # backwards recursion from each origin's own diagonal (R: getExpected)
    ultdf = np.ones(n_d)
    for j in range(n_d - 2, -1, -1):
        ultdf[j] = ultdf[j + 1] * f[j]
    ult = cum[np.arange(n_w), latest_dev] * ultdf[latest_dev]
    fitted_cum = ult[:, None] / ultdf[None, :]
    fitted = np.full((n_w, n_d), np.nan)
    fitted[:, 0] = np.where(obs_mask[:, 0], fitted_cum[:, 0], np.nan)
    fitted[:, 1:] = np.where(obs_mask[:, 1:], fitted_cum[:, 1:] - fitted_cum[:, :-1], np.nan)

    pool_mask = obs_mask & (fitted > 0)
    dead = obs_mask & ~pool_mask
    if (inc[dead] != 0).any():
        raise ValueError(
            f"{int((inc[dead] != 0).sum())} cell(s) have a non-positive chain-ladder fitted "
            "mean against a non-zero observed increment, so their Pearson residual is "
            "undefined; the cohort's chain-ladder fit is degenerate"
        )
    if not pool_mask.any():
        raise ValueError("no cell has a positive fitted mean; nothing to resample residuals from")

    n_cells = int(obs_mask.sum())
    n_params = n_w + n_d - 1
    if n_cells <= n_params:
        raise ValueError(
            f"triangle has {n_cells} observed cells but the ODP model has {n_params} "
            "parameters, so the Pearson scale has no residual degrees of freedom"
        )
    unscaled = np.full((n_w, n_d), np.nan)
    unscaled[pool_mask] = (inc[pool_mask] - fitted[pool_mask]) / np.sqrt(fitted[pool_mask])
    dof = n_cells - n_params
    phi = float(np.nansum(unscaled**2) / dof)
    residuals = unscaled * np.sqrt(n_cells / dof)
    return ODPBootstrapFit(
        inc=inc,
        fitted=fitted,
        residuals=residuals,
        obs_mask=obs_mask,
        pool_mask=pool_mask,
        latest_dev=latest_dev,
        phi=phi,
        n_cells=n_cells,
        n_params=n_params,
    )


def draw_next_increments(
    boot: ODPBootstrapFit,
    *,
    n_draws: int,
    rng: np.random.Generator,
    process: str = "od_poisson",
    process_noise: bool = True,
    resample_residuals: bool = True,
) -> np.ndarray:
    """``(n_draws, n_w)`` simulated payments over the next development step.

    One draw is R's ``getNYCost`` input for one simulation, and the steps are
    its steps:

    1. resample the adjusted residuals with replacement over the observed cells
       and rebuild a pseudo-triangle, ``X*[i,j] = r*[i,j] sqrt(mhat[i,j]) +
       mhat[i,j]`` (R: ``randomClaims``);
    2. cumulate it and **refit the volume-weighted chain ladder** on it, giving
       ``f*`` (R: ``getIndivDFs`` + ``getAvDFs`` with the cumulative triangle as
       weights, which is the volume-weighted factor written out);
    3. the expected next increment is that refit projected off the pseudo
       diagonal, ``mu[i] = C*[i, k_i] * (f*[k_i] - 1)`` - R computes it as
       ``getIncremental(getExpected(simUlts, 1/simUltDFs))`` restricted to the
       next diagonal, which reduces to exactly this;
    4. add over-dispersed process noise about ``mu`` with variance ``phi*|mu|``.

    Only the NEXT diagonal is drawn. R builds the whole lower triangle because
    ``BootChainLadder`` also reports the full run-off reserve; the one-year CDR
    reads a single diagonal off it (``getDiagonalIndexes(t.ibnr, m+1)``), so
    the rest is never used here.

    The two risk-source switches are R's two arms plus the one it derives by
    subtraction. ``resample_residuals=False`` makes the pseudo-triangle the
    fitted triangle itself, which refits to ``f*`` exactly ``f`` (the fitted
    cumulatives are ``ult[i]/ultdf[j]``, so their column ratio is
    ``ultdf[j]/ultdf[j+1] = f[j]``) and therefore isolates process risk;
    ``process_noise=False`` is R's ``NYParamDist`` arm and isolates estimation
    risk. R reports ``CDR.Process.S.E`` as ``sqrt(total^2 - param^2)`` rather
    than simulating it, and the three arms here reproduce that identity to
    Monte Carlo error.

    Memory: one ``(n_draws, n_w, n_d)`` float64 pseudo-triangle is materialized
    (46 MB at 20,000 draws on a 17x17 triangle), then reduced away.
    """
    if process not in ODP_PROCESS_LAWS:
        raise ValueError(f"process must be one of {ODP_PROCESS_LAWS}, got {process!r}")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    n_w, n_d = boot.n_w, boot.n_d
    mask, k = boot.obs_mask, boot.latest_dev

    # 1. pseudo increments: the fitted triangle, plus resampled residual noise
    pseudo = np.zeros((n_draws, n_w, n_d))
    pseudo[:, mask] = boot.fitted[mask]
    if resample_residuals:
        pool = boot.residuals[boot.pool_mask]
        sampled = rng.choice(pool, size=(n_draws, int(mask.sum())), replace=True)
        pseudo[:, mask] += sampled * np.sqrt(boot.fitted[mask])

    # 2. cumulate in place and re-estimate the volume-weighted factors. The
    #    unobserved cells were left at 0, so the cumulative sum past an origin's
    #    diagonal is meaningless - `pair` is what keeps it out of every sum.
    np.cumsum(pseudo, axis=2, out=pseudo)
    pair = (mask[:, :-1] & mask[:, 1:]).astype(float)  # (n_w, n_d - 1)
    denom = np.einsum("dij,ij->dj", pseudo[:, :, :-1], pair)  # (n_draws, n_d - 1)
    numer = np.einsum("dij,ij->dj", pseudo[:, :, 1:], pair)
    live = denom > 0
    f_star = np.where(live, numer / np.where(live, denom, 1.0), 1.0)

    # 3. project the next increment off the PSEUDO diagonal
    diag_star = pseudo[:, np.arange(n_w), k]  # (n_draws, n_w)
    del pseudo
    mu = np.zeros((n_draws, n_w))
    open_ = np.nonzero(k < n_d - 1)[0]
    mu[:, open_] = diag_star[:, open_] * (f_star[:, k[open_]] - 1.0)

    # 4. process noise. A zero scale is the data saying the development is
    #    noiseless (every Pearson residual vanished); mirroring
    #    kernels.mack.draw_step, the draw is then its mean exactly.
    if not process_noise or boot.phi <= 0:
        return mu
    return _od_process_noise(rng, mu, boot.phi, law=process)


def _od_process_noise(
    rng: np.random.Generator, mu: np.ndarray, phi: float, *, law: str
) -> np.ndarray:
    """Over-dispersed noise about ``mu`` with variance ``phi * |mu|``.

    The ``sign(mu) * f(|mu|)`` reflection is R's (``processTriangle``): a
    resampled pseudo-triangle can refit to ``f* < 1`` and hand back a negative
    expected increment even when every observed increment was non-negative, and
    reflecting keeps the draw's mean at ``mu`` rather than discarding it. Note
    the asymmetry with the observed data, which is refused outright when
    negative - that refusal is about the family's support, this is about a
    bootstrap artifact of it.

    ``od_poisson`` is the shared :func:`~ibnr.kernels.densities.odp_draw`, so
    the rate numpy can no longer represent is handled here exactly as it is in
    the gallery's ODP entries - by returning the mean, and in one place.

    One consequence, worth writing down because the two laws are otherwise
    meant to differ only in support and tail: ``odp_draw`` refuses a mean that
    is not finite, so ``od_poisson`` names a NaN mean by count, while ``gamma``
    still hands it to ``rng.gamma`` and returns NaN without a word, which is
    what both laws did before. Neither is reachable from
    :func:`draw_next_increments` on a triangle ``fit_odp_bootstrap`` accepted:
    every mean there is a finite pseudo-diagonal cell times a finite factor.

    The ``live`` mask below is an optimization and nothing more, so do not
    write a test that expects it to change an answer. numpy returns 0 for a
    Poisson rate of 0 and for a gamma shape of 0 *without* taking a random
    number (measured, numpy 2.4.6), so passing the zero cells through would
    give the same values off the same generator state. It is kept because
    ``sign(0) * draw`` reading as 0 is an accident of ``sign``, and because the
    mask says out loud that a cell with nothing to develop has nothing to draw.
    """
    out = np.array(mu, dtype=float, copy=True)
    live = mu != 0
    if not live.any():
        return out
    magnitude = np.abs(mu[live])
    sign = np.sign(mu[live])
    if law == "od_poisson":
        out[live] = sign * odp_draw(rng, magnitude, phi)
    else:  # gamma, moment-matched: shape * scale = |mu|, shape * scale^2 = phi|mu|
        out[live] = sign * rng.gamma(shape=magnitude / phi, scale=phi)
    return out
