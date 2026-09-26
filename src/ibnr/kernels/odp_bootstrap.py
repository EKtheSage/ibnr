"""England & Verrall's over-dispersed Poisson bootstrap, over a dense grid.

The residual bootstrap behind ``ChainLadder::BootChainLadder`` in R and behind
England & Verrall (2002, section 8): fit the chain ladder, take Pearson
residuals against its fitted incrementals, resample them into pseudo-triangles,
refit, and add over-dispersed Poisson process noise to the projection.

Two uses share the deterministic half, :func:`fit_odp_bootstrap`:

- the one-year CDR's ``odp_bootstrap`` route (``kernels/cdr.py``), which draws
  the NEXT diagonal only, with :func:`draw_next_increments`;
- the full run-off bootstrap behind ``ibnr.methods.odp_bootstrap``, which
  simulates every future cell and refits the chain ladder,
  Bornhuetter-Ferguson, Benktander or Cape Cod on every simulated triangle,
  with :func:`prepare_runoff` and :func:`draw_runoff`.

Free functions over plain arrays, deliberately: nothing here takes a
``Triangle`` or a ``MackFit``, so the arithmetic can be read against R's source
and tested without either. This module imports numpy and ibnr's numpy-only
kernels, so ``ibnr.methods`` can use it without loading ibis, pandas or scipy.

R's source is the reference implementation (``R/BootstrapReserve.R``, Nigel de
Silva / Giuseppe Crupi / Markus Gesmann) and :func:`fit_odp_bootstrap`'s
defaults mirror it step for step. The places the defaults deviate, all
documented at the point they happen:

1. **Negative observed increments are refused** by default, where R takes
   ``sign(x)`` and ``sqrt(abs(m))`` and carries on. The ODP family is defined
   on non-negative increments (``kernels.contract.odp_stan_data`` refuses them
   too, and ``england_verrall_odp`` inherits that limit). About half the
   Schedule P mart's paid cohorts fall outside it.
   ``negative_increments="reflect"`` does what R and chainladder-python do.
2. **The residual pool and the parameter count are general**, ``n_w x n_d``
   run-off staircase rather than R's square block: R restricts to the last
   ``n`` origins so its ``nobs = n(n+1)/2`` and ``p = 2n - 1`` formulas hold.
   Ours count the observed cells and use ``p = n_w + n_d - 1``, which is the
   same number on a square triangle and the ODP GLM's actual parameter count
   otherwise.
3. **A pseudo development step with non-positive volume takes factor 1.0.** R
   does this only for the exact ``0/0`` case (``out[is.nan(out)] <- 1``) and
   lets a negative volume through to a finite, wrong-signed factor.
4. **``od_poisson`` is ``phi * Poisson(mu / phi)``**, the England & Verrall
   construction and the one ``england_verrall_odp.predict`` already draws, where
   R's ``rpois.od`` uses a negative binomial with the same first two moments
   (mean ``mu``, variance ``phi * mu``) and integer support.

The residual options (:data:`RESIDUAL_ADJUSTMENTS`, :data:`RESIDUAL_POOLS`)
default to R's conventions, which is what the one-year CDR route uses and what
it keeps, bit for bit. ``adjustment="hat"`` with ``pool="centred"`` is
chainladder-python's ``BootstrapODPSample`` convention, without its defects
(``docs/coming-from-chainladder.md`` lists them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ibnr.errors import Refusal, RefusedCell
from ibnr.kernels.conventional import (
    ConventionalCandidate,
    _Estimate,
    _estimate_grid,
    cape_cod_weights,
    pattern_beta,
    project_ultimates,
)
from ibnr.kernels.densities import odp_draw
from ibnr.kernels.links import REASONS as LINK_REASONS
from ibnr.kernels.links import link_factors_many, position_rules, select_links
from ibnr.kernels.tail import TailFit, TailSpec, apply_tail

#: process laws for the simulated increment. Both match the over-dispersed
#: Poisson's first two moments (mean ``mu``, variance ``phi * mu``) and differ
#: in support and tail, exactly as ``kernels.mack.PROCESS_LAWS`` do for Mack.
ODP_PROCESS_LAWS = ("od_poisson", "gamma")

#: How a Pearson residual is scaled before it is resampled. ``"dof"`` (R's)
#: multiplies every residual by ``sqrt(n / (n - p))``; ``"hat"`` divides each by
#: ``sqrt(1 - h)``, ``h`` its leverage in the ODP GLM (chainladder-python's
#: ``hat_adj=True``); ``"none"`` leaves it as it is (chainladder-python's
#: ``hat_adj=False``, whose docstring says degrees of freedom but whose code
#: adjusts nothing).
RESIDUAL_ADJUSTMENTS = ("dof", "hat", "none")

#: Which residuals are resampled. ``"all"`` (R's) is every observed cell with a
#: non-zero fitted mean that no development option left out, the cells fitted
#: exactly by construction (leverage one, residual 0) included, not centred.
#: ``"centred"`` (chainladder-python's rule) also leaves out the cells of
#: leverage one, then subtracts the pool's mean.
RESIDUAL_POOLS = ("all", "centred")

#: What a negative observed increment, or a negative fitted mean, does.
NEGATIVE_INCREMENTS = ("refuse", "reflect")

#: Why an observed cell is or is not in the residual pool: ``pool_reason``
#: indexes this tuple, -1 outside the triangle.
POOL_REASONS = ("pooled", "leverage_one", "zero_fitted_mean", "excluded_link")

#: A cell whose leverage is above this is fitted exactly: its residual is 0 by
#: construction, not by chance (the first origin's last cell and the last
#: origin's first cell, always).
LEVERAGE_ONE = 1.0 - 1e-9

#: How far outside [0, 1] a computed leverage may fall, to rounding, before the
#: hat adjustment refuses the triangle.
_LEVERAGE_SLACK = 1e-9

#: The working memory :func:`draw_runoff` aims a chunk of draws at, by default.
_CHUNK_BYTES = 64 * 2**20


@dataclass(frozen=True)
class ODPBootstrapFit:
    """The deterministic half of the bootstrap: what every pseudo-triangle is
    resampled *from*.

    ``fitted`` are the chain-ladder fitted incrementals - equivalently the
    cross-classified Poisson MLE's fitted means (Renshaw & Verrall 1998), which
    is why the ODP GLM and the chain ladder give the same reserve and why
    ``tests/test_odp_bootstrap.py`` checks these against
    ``england_verrall_odp.odp_mle_fitted``'s iterative proportional fit - two
    independent routes to one quantity. That equivalence holds for the
    volume-weighted factors over every link ratio; with other factors (another
    average, development options) the fitted values are still the backward
    recursion from those factors, and the leverage is still the ODP GLM's at
    those fitted means, which is then an approximation (see :func:`fit_odp_bootstrap`).
    """

    inc: np.ndarray  # (n_w, n_d) observed increments, NaN outside the triangle
    fitted: np.ndarray  # (n_w, n_d) chain-ladder fitted increments, NaN outside
    residuals: np.ndarray  # (n_w, n_d) adjusted Pearson residuals, NaN off the pool
    obs_mask: np.ndarray  # (n_w, n_d) bool
    pool_mask: np.ndarray  # (n_w, n_d) bool - the cells whose residuals are resampled
    latest_dev: np.ndarray  # (n_w,) 0-based dev index of each origin's diagonal
    phi: float  # Pearson scale, sum(r^2) / (n_cells - n_params)
    n_cells: int  # observed cells (R's `nobs`)
    n_params: int  # n_w + n_d - 1 (R's `2n - 1` on a square triangle)
    #: (n_w, n_d) unscaled Pearson residuals, (inc - m) / sqrt(|m|); NaN where m == 0
    unscaled: np.ndarray | None = None
    #: (n_w, n_d) leverage in the ODP GLM; NaN where m == 0, and everywhere when
    #: it could not be computed
    leverage: np.ndarray | None = None
    #: 1-D, the values resampled, in row-major order over (origin, dev) of
    #: ``pool_mask``, centred under ``pool_rule="centred"``
    pool: np.ndarray | None = None
    #: (n_w, n_d) int8 index into POOL_REASONS, -1 outside the triangle
    pool_reason: np.ndarray | None = None
    adjustment: str = "dof"
    pool_rule: str = "all"
    negative_increments: str = "refuse"
    n_negative_fitted: int = 0  # observed cells with a negative fitted mean

    @property
    def n_w(self) -> int:
        return self.inc.shape[0]

    @property
    def n_d(self) -> int:
        return self.inc.shape[1]

    @property
    def degrees_of_freedom(self) -> int:
        return self.n_cells - self.n_params


def _choose(name: str, value, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        quoted = [repr(choice) for choice in choices]
        listed = " or ".join(quoted) if len(quoted) == 2 else ", ".join(quoted)
        raise Refusal(
            "invalid_option",
            f"{name} must be one of {listed}, got {{given}}",
            option=name,
            given=value,
        )


def fit_odp_bootstrap(
    cum: np.ndarray,
    obs_mask: np.ndarray,
    latest_dev: np.ndarray,
    f: np.ndarray,
    *,
    origins: list | None = None,
    dev_grain_months: int | None = None,
    adjustment: str = "dof",
    pool: str = "all",
    negative_increments: str = "refuse",
    excluded: np.ndarray | None = None,
    exact: np.ndarray | None = None,
) -> ODPBootstrapFit:
    """Fitted incrementals, Pearson scale and adjusted residuals of one cohort.

    ``f`` is supplied rather than re-estimated so the bootstrap and everything
    else built on the same cohort share one set of development factors;
    passing ``MackFit.f`` is what the CDR does, and the run-off bootstrap passes
    the central fit's factors, after its development options and before any
    tail.

    ``origins`` (the origin periods) and ``dev_grain_months`` only let a
    refusal name its cells; every refusal here is an ``ibnr.errors.Refusal``.

    Fitted values come from R's backwards recursion (``getExpected``): project
    each origin's ultimate off its own diagonal, divide back down the ultimate
    factors, difference. With ``ultdf[j] = prod_{t>=j} f[t]``,

        ult[i]         = C[i, k_i] * ultdf[k_i]
        Chat[i, j]     = ult[i] / ultdf[j]
        mhat[i, j]     = Chat[i, j] - Chat[i, j-1]

    This is the ODP GLM's fitted mean when ``f`` is the volume-weighted factor
    over every link ratio. For any other ``f`` it is still the recursion from
    ``f`` (refitting the fitted triangle then gives ``f`` back, whatever the
    average, because every link ratio of the fitted triangle is ``f``), and the
    leverage below is the ODP GLM's at those means, which is exact only in the
    volume case.

    The options, whose defaults are R's conventions and the one-year CDR's:

    - ``adjustment``: one of :data:`RESIDUAL_ADJUSTMENTS`. Under ``"hat"`` a
      cell of leverage one gets the adjusted residual 0; under ``"dof"`` and
      ``"none"`` it keeps the unscaled residual (times the factor), which is 0
      up to rounding.
    - ``pool``: one of :data:`RESIDUAL_POOLS`. Cells of leverage one are found
      by their leverage (above :data:`LEVERAGE_ONE`), never by testing a
      residual for zero, which rounding defeats; ``exact`` replaces that rule
      (below). An observed increment of zero
      is data and stays in the pool (its residual is ``-sqrt(m)``).
    - ``negative_increments``: ``"refuse"`` refuses a negative observed
      increment (``negative_increment``) and a negative fitted mean, which only
      a factor below 1 makes (``negative_fitted_mean``); ``"reflect"`` accepts
      both, uses ``sqrt(|m|)`` and ``|m|`` wherever a fitted mean is a scale or
      a weight, and counts the negative fitted means in ``n_negative_fitted``.
    - ``excluded``: ``(n_w, n_d)`` bool, cells whose residual a development
      option left out; they leave the pool and the scale's numerator but still
      count as observed cells in its degrees of freedom (as in R and
      chainladder-python).
    - ``exact``: ``(n_w, n_d)`` bool, the cells the factors ``f`` reproduce
      exactly, from how they were estimated; ``None`` finds them by leverage.
      Without development options the two agree (the first origin's last
      cell and the last origin's first cell). With them, a cell whose link
      ratios are each the only one left at their age is reproduced exactly
      too, and :func:`prepare_runoff` passes those; chainladder-python finds
      the same cells by their residual of zero.

    The scale is ``phi = sum(r^2) / (n_cells - n_params)`` over the unscaled
    residuals of every observed cell with a non-zero fitted mean that is not
    excluded; ``n_cells`` counts every observed cell.

    Leverage is ``h = diag(X (X' W X)^+ X' W)`` with ``W = |m|``, over the
    cells with a non-zero fitted mean, computed from the singular value
    decomposition of ``W^(1/2) X``. The pseudo-inverse makes a development
    column whose fitted increments are all zero (a factor of exactly 1.0) simply
    absent instead of singular. It is computed whatever the options; under
    ``"hat"`` or ``"centred"`` a leverage that could not be computed, or falls
    outside [0, 1] by more than rounding, is refused (``degenerate_fit``).
    """
    _choose("adjustment", adjustment, RESIDUAL_ADJUSTMENTS)
    _choose("pool", pool, RESIDUAL_POOLS)
    _choose("negative_increments", negative_increments, NEGATIVE_INCREMENTS)
    cum = np.asarray(cum, dtype=float)
    obs_mask = np.asarray(obs_mask, dtype=bool)
    latest_dev = np.asarray(latest_dev, dtype=int)
    f = np.asarray(f, dtype=float)
    n_w, n_d = cum.shape
    months = 1 if dev_grain_months is None else int(dev_grain_months)
    reflect = negative_increments == "reflect"
    if excluded is None:
        excluded = np.zeros((n_w, n_d), dtype=bool)
    else:
        excluded = np.asarray(excluded, dtype=bool) & obs_mask
        if excluded.shape != (n_w, n_d):
            raise ValueError(f"excluded must be ({n_w}, {n_d}), got {excluded.shape}")

    def cells(where: np.ndarray, amounts: np.ndarray) -> list[RefusedCell]:
        return [
            RefusedCell(
                None,
                None if origins is None else origins[i],
                (int(j) + 1) * months,
                float(amounts[i, j]),
            )
            for i, j in zip(*np.nonzero(where), strict=True)
        ]

    if n_d < 2:
        raise Refusal(
            "not_identified",
            "the ODP bootstrap needs at least two development steps",
            option="cells",
        )

    inc = np.full((n_w, n_d), np.nan)
    inc[:, 0] = np.where(obs_mask[:, 0], cum[:, 0], np.nan)
    prev = np.where(obs_mask[:, :-1], cum[:, :-1], np.nan)
    inc[:, 1:] = np.where(obs_mask[:, 1:], cum[:, 1:] - prev, np.nan)
    negative = obs_mask & (inc < 0)
    if negative.any() and not reflect:
        raise Refusal(
            "negative_increment",
            f"{int(negative.sum())} negative incremental cell(s) at {{cells}}. The "
            "over-dispersed Poisson family is defined on non-negative increments - its "
            "Pearson residual divides by a fitted mean that a negative increment can drive "
            "below zero, and its process draw has no negative mean - so the ODP bootstrap "
            "refuses this cohort outright rather than reflecting it through sign(). Mack's "
            "conditional moments carry no such restriction: the mack diagonal generator "
            "(and one_year_cdr) still answer here",
            option="cells",
            cells=cells(negative, inc),
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

    below = obs_mask & (fitted < 0)
    if below.any() and not reflect:
        falling = [j for j in range(min(n_d - 1, f.size)) if f[j] < 1]
        raise Refusal(
            "negative_fitted_mean",
            f"{int(below.sum())} cell(s) have a negative fitted mean, at {{cells}}: a "
            "development factor below 1 {links} makes the fitted increments after it "
            "negative, and the over-dispersed Poisson bootstrap needs a positive mean to "
            "scale a residual by. Pass negative_increments='reflect' to bootstrap it "
            "through sign() and |m|, as chainladder-python and R do",
            option="negative_increments",
            options=("negative_increments", "cells"),
            cells=cells(below, fitted),
            links=[((j + 1) * months, (j + 2) * months) for j in falling],
        )
    # Under "refuse" no fitted mean is negative here, so != 0 is > 0, R's rule.
    nonzero = obs_mask & (fitted != 0)
    # An excluded cell's residual enters nothing, so a zero fitted mean there
    # (a link left with no ratio and a factor of 1.0) is not a problem.
    dead = obs_mask & ~nonzero & ~excluded
    if (inc[dead] != 0).any():
        raise Refusal(
            "degenerate_fit",
            f"{int((inc[dead] != 0).sum())} cell(s) have a zero fitted "
            "mean against a non-zero observed increment, so their Pearson residual is "
            "undefined and the fit is degenerate: {cells}",
            option="cells",
            cells=cells(dead & (np.nan_to_num(inc) != 0), inc),
        )
    candidates = nonzero & ~excluded
    if not candidates.any():
        raise Refusal(
            "empty_residual_pool",
            "no residual is left to resample: every cell has a zero fitted mean or is left "
            "out by the development options",
            option="cells",
        )

    n_cells = int(obs_mask.sum())
    n_params = n_w + n_d - 1
    if n_cells <= n_params:
        raise Refusal(
            "not_identified",
            f"triangle has {n_cells} observed cells but the ODP model has {n_params} "
            "parameters (one per origin and one per development age after the first), so "
            "the Pearson scale has no residual degrees of freedom; it needs more cells than "
            "parameters",
            option="cells",
        )
    unscaled = np.full((n_w, n_d), np.nan)
    unscaled[nonzero] = (inc[nonzero] - fitted[nonzero]) / np.sqrt(np.abs(fitted[nonzero]))
    in_scale = np.where(candidates, unscaled, np.nan)
    dof = n_cells - n_params
    phi = float(np.nansum(in_scale**2) / dof)

    leverage = _leverage(nonzero, np.abs(np.nan_to_num(fitted)))
    needs_leverage = adjustment == "hat" or (pool == "centred" and exact is None)
    if needs_leverage:
        h = leverage[nonzero]
        if not (np.isfinite(h).all() and (h >= -_LEVERAGE_SLACK).all()):
            bad = True
        else:
            bad = bool((h > 1 + _LEVERAGE_SLACK).any())
        if bad:
            raise Refusal(
                "degenerate_fit",
                "the hat-matrix leverage of the fitted cells could not be computed for this "
                "triangle, or is not between 0 and 1; pass residual_adjustment='dof' or "
                "'none' with residual_pool='all' to bootstrap it without the leverage",
                option="residual_adjustment",
                options=("residual_adjustment", "residual_pool"),
                given=adjustment,
            )
    leverage_one = np.nan_to_num(leverage, nan=0.0) > LEVERAGE_ONE
    if exact is None:
        exact = leverage_one
    exact = candidates & np.asarray(exact, dtype=bool)
    # A cell of leverage one has no hat factor (1 / sqrt(0)). With the GLM's own
    # fitted values its residual is 0 anyway; with factors from elsewhere (a
    # caller's own ``f``) it need not be, and it is then treated as
    # chainladder-python treats it: adjusted to 0, and out of the centred pool.
    undefined = candidates & leverage_one if adjustment == "hat" else np.zeros_like(candidates)

    if adjustment == "dof":
        adjusted = in_scale * np.sqrt(n_cells / dof)
    elif adjustment == "none":
        adjusted = in_scale.copy()
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            adjusted = in_scale / np.sqrt(1.0 - leverage)
        adjusted[exact | undefined] = 0.0
    in_pool = candidates & ~(exact | undefined) if pool == "centred" else candidates
    residuals = np.where(in_pool, adjusted, np.nan)
    values = residuals[in_pool]
    if pool == "centred":
        if not values.size:
            raise Refusal(
                "empty_residual_pool",
                "no residual is left to resample: every cell with a non-zero fitted mean is "
                "fitted exactly or left out by the development options",
                option="cells",
            )
        values = values - values.mean()

    reason = np.full((n_w, n_d), -1, dtype=np.int8)
    reason[obs_mask] = POOL_REASONS.index("pooled")
    reason[obs_mask & ~nonzero] = POOL_REASONS.index("zero_fitted_mean")
    reason[candidates & ~in_pool] = POOL_REASONS.index("leverage_one")
    reason[obs_mask & excluded] = POOL_REASONS.index("excluded_link")
    return ODPBootstrapFit(
        inc=inc,
        fitted=fitted,
        residuals=residuals,
        obs_mask=obs_mask,
        pool_mask=in_pool,
        latest_dev=latest_dev,
        phi=phi,
        n_cells=n_cells,
        n_params=n_params,
        unscaled=unscaled,
        leverage=leverage,
        pool=values,
        pool_reason=reason,
        adjustment=adjustment,
        pool_rule=pool,
        negative_increments=negative_increments,
        n_negative_fitted=int(below.sum()),
    )


def _leverage(cells: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``(n_w, n_d)`` leverage of each cell in ``cells`` in the ODP GLM, NaN elsewhere.

    The GLM has a log link, one parameter per origin and one per development
    age after the first, and working weights equal to the fitted means. Its hat
    matrix is ``W^(1/2) X (X' W X)^+ X' W^(1/2)``; its diagonal is the sum of
    squares of each row of ``U``, the left singular vectors of ``W^(1/2) X``
    with a singular value above numpy's rank tolerance. A column of ``X`` with
    no cell (a development age whose fitted increments are all zero) only
    lowers the rank. NaN everywhere if the decomposition fails.
    """
    n_w, n_d = cells.shape
    out = np.full((n_w, n_d), np.nan)
    where = np.argwhere(cells)
    if not where.size:
        return out
    rows = np.arange(len(where))
    design = np.zeros((len(where), n_w + n_d - 1))
    design[rows, where[:, 0]] = 1.0
    later = where[:, 1] > 0
    design[rows[later], n_w + where[later, 1] - 1] = 1.0
    scaled = design * np.sqrt(weight[cells])[:, None]
    try:
        left, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    except np.linalg.LinAlgError:
        return out
    if not np.isfinite(singular).all() or not singular.size:
        return out
    tolerance = singular.max() * max(scaled.shape) * np.finfo(float).eps
    rank = int((singular > tolerance).sum())
    out[cells] = (left[:, :rank] ** 2).sum(axis=1)
    return out


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

    Only the NEXT diagonal is drawn: the one-year CDR reads a single diagonal
    (R: ``getDiagonalIndexes(t.ibnr, m+1)``). :func:`draw_runoff` draws the
    whole run-off.

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
        raise Refusal(
            "invalid_option",
            f"process must be one of {ODP_PROCESS_LAWS}, got {{given}}",
            option="process",
            given=process,
        )
    if n_draws < 1:
        raise Refusal(
            "invalid_option",
            "n_draws must be positive, got {given}",
            option="n_draws",
            given=n_draws,
        )
    n_w, n_d = boot.n_w, boot.n_d
    mask, k = boot.obs_mask, boot.latest_dev

    # 1. pseudo increments: the fitted triangle, plus resampled residual noise
    pseudo = np.zeros((n_draws, n_w, n_d))
    pseudo[:, mask] = boot.fitted[mask]
    if resample_residuals:
        sampled = rng.choice(boot.pool, size=(n_draws, int(mask.sum())), replace=True)
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
    the asymmetry with the observed data, which is refused by default when
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
    :func:`draw_runoff` hands no draw with a mean that is not finite to this
    function.

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


# -- the full run-off --------------------------------------------------------------


@dataclass(frozen=True)
class RunoffProjection:
    """How every simulated triangle is refitted and projected.

    One specification with the central fit: the same method, average, a priori
    loss ratio, Cape Cod weights and trend, and tail. The link ratios each
    refit uses are ``keep``: the pairs the POSITION rules keep on the real
    triangle (the zero rule, the history window, explicit and valuation
    exclusions), decided once and reused on every simulated triangle. The
    rules that pick a ratio by its size (``drop_high``/``drop_low``,
    ``drop_above``/``drop_below``) have already acted once, on the central
    factors that make the fitted values and on the residual pool, and do not
    act again: trimming each simulated triangle again would remove ordinary
    draws a second time and pull every factor down (measured on
    chainladder-python itself: the mean falls 3.5% to 22% below the central
    estimate).

    A tail acts the same way: once. The fitted values and the residual pool
    come from the central factors BEFORE the tail, and the tail (a curve
    refitted to each draw's own factors, or a constant) is applied only in
    each refit, as chainladder-python does. With a curve attached before the
    last age in the fitted values too, the fitted triangle's ratios past the
    attachment were the curve's, and fitting the curve to each simulated
    triangle then pulled the simulated mean off the central estimate (measured
    at 20,000 draws, as a share of the central IBNR: -6.2% on genins with an
    exponential curve at 72 months, -7.0% with a Weibull, -3.4% and -4.4% on
    abc; chainladder-python stays within 0.9% on all four, and so does this
    kernel now). A curve attached at the last age reads the same factors
    either way.

    Attributes
    ----------
    method : str
        ``"cl"``, ``"bf"`` (Benktander with ``n_iters`` above 1) or ``"gcc"``.
    keep : numpy.ndarray
        ``(n_w, n_links)`` bool, the pairs every refit uses.
    unit : numpy.ndarray
        ``(n_links,)`` bool, the links ``keep`` leaves with no pair: factor
        1.0 in every draw, as in the central fit (``unsupported_factor="unity"``).
    average : str
        As ``ConventionalCandidate.average``.
    latest_dev : numpy.ndarray
        ``(n_w,)`` each origin's latest development index.
    dev_grain_months : int
    premium : numpy.ndarray or None
        ``(n_w,)``, for the a priori methods.
    expected_loss_ratio : float or None
        Bornhuetter-Ferguson and Benktander.
    n_iters : int
    weights : numpy.ndarray or None
        Cape Cod's ``(n_w, n_w)`` weights.
    trend_factor : numpy.ndarray
        ``(n_w,)`` Cape Cod's trend factor to the valuation date (1.0 at trend 0).
    tail : TailSpec or None
    central_tail : TailFit or None
        The central fit's tail, which a draw whose curve fails its checks uses.
    """

    method: str
    keep: np.ndarray
    unit: np.ndarray
    average: str
    latest_dev: np.ndarray
    dev_grain_months: int
    premium: np.ndarray | None = None
    expected_loss_ratio: float | None = None
    n_iters: int = 1
    weights: np.ndarray | None = None
    trend_factor: np.ndarray | None = None
    tail: TailSpec | None = None
    central_tail: TailFit | None = None

    @property
    def has_prior(self) -> bool:
        """Whether the method has an a priori loss ratio a multiplier can vary."""
        return self.method != "cl"


@dataclass(frozen=True)
class RunoffSetup:
    """Everything the run-off bootstrap needs before its first draw.

    ``estimate`` is the central fit (the same one ``fit_conventional_grid``
    gives), ``selection`` its link selection, ``excluded`` the cells whose
    residual a development option left out, and ``excluded_by`` the reason of
    the link that took each out (an index into ``kernels.links.REASONS``, -1
    where none did).
    """

    estimate: _Estimate
    boot: ODPBootstrapFit
    projection: RunoffProjection
    excluded: np.ndarray
    excluded_by: np.ndarray


#: The reasons a link ratio is left out that do NOT take its residual out of
#: the pool: the zero rule is about what a zero cumulative is, and a zero
#: increment is data to the bootstrap.
_KEEPS_RESIDUAL = frozenset(
    {LINK_REASONS.index(name) for name in ("included", "zero_cell", "undefined_ratio")}
)


def prepare_runoff(
    grid: dict[str, Any],
    candidate: ConventionalCandidate,
    *,
    premium=None,
    adjustment: str = "dof",
    pool: str = "all",
    negative_increments: str = "refuse",
) -> RunoffSetup:
    """The central fit, the residuals and the refit's specification, from one set of options.

    ``grid`` and ``candidate`` are as in ``kernels.fit_conventional_grid``
    (``premium`` too), without ``horizon``. The central fit's factors, after
    every development option and before any tail, make the fitted values; the cells
    whose link ratio a development option left out leave the residual pool
    (chainladder-python's rule: the later cell of the link, and for a link from
    the first age the first cell too); and the position rules make every
    refit's link ratios (:class:`RunoffProjection`). The residual options are
    :func:`fit_odp_bootstrap`'s.
    """
    if candidate.horizon is not None:
        raise Refusal(
            "not_supported",
            "the run-off bootstrap projects to the last observed age (and a tail); a "
            "fixed horizon is a replay setting",
            option="horizon",
            given=candidate.horizon,
        )
    estimate = _estimate_grid(grid, candidate, premium=premium)
    fitted = estimate.grid
    cum, mask = fitted["cum"], fitted["obs_mask"]
    periods = fitted["origin_periods"]
    step = int(fitted["dev_grain_months"])
    n_w, n_d = cum.shape
    if n_d < 2:
        # checked here, before the link selection, which has no link to index
        raise Refusal(
            "not_identified",
            "the ODP bootstrap needs at least two development ages; this triangle has "
            "one. chain_ladder gives the latest amounts as the ultimates",
            option="cells",
        )
    n_links = n_d - 1
    rules = candidate.link_rules
    selection = select_links(cum, mask, periods, step, rules, n_links, raise_exhausted=False)

    # the residual cells a development option left out
    reason = selection.reason
    out_link = selection.observed & ~np.isin(reason, list(_KEEPS_RESIDUAL))
    excluded = np.zeros((n_w, n_d), dtype=bool)
    excluded_by = np.full((n_w, n_d), -1, dtype=np.int8)
    excluded[:, 1:] = out_link
    excluded_by[:, 1:] = np.where(out_link, reason, -1)
    first = out_link[:, 0]
    excluded[first, 0] = True
    excluded_by[first, 0] = reason[first, 0]

    boot = fit_odp_bootstrap(
        cum,
        mask,
        fitted["latest_dev"],
        # the factors before any tail: a curve attached before the last age
        # enters each refit, not the fitted values (see RunoffProjection)
        estimate.untailed_factors[:n_links],
        origins=periods,
        dev_grain_months=step,
        adjustment=adjustment,
        pool=pool,
        negative_increments=negative_increments,
        excluded=excluded,
        exact=_reproduced(selection.used, fitted["latest_dev"], mask),
    )
    if negative_increments != "reflect":
        # after the data's own refusals, so a negative observed increment is named first
        _refuse_a_falling_tail(estimate, periods, step, fitted["latest_dev"], mask)
    keep = select_links(
        cum, mask, periods, step, position_rules(rules), n_links, raise_exhausted=False
    ).used
    weights = None
    if candidate.method == "gcc":
        weights = cape_cod_weights(periods, step, candidate.decay)
    projection = RunoffProjection(
        method=candidate.method,
        keep=keep,
        unit=~keep.any(axis=0),
        average=candidate.average,
        latest_dev=np.asarray(fitted["latest_dev"], dtype=int),
        dev_grain_months=step,
        premium=None if candidate.method == "cl" else np.asarray(fitted["premium"], float),
        expected_loss_ratio=candidate.expected_loss_ratio,
        n_iters=candidate.n_iters,
        weights=weights,
        trend_factor=np.asarray(estimate.origins["trend_factor"], dtype=float),
        tail=candidate.tail,
        central_tail=estimate.tail,
    )
    return RunoffSetup(estimate, boot, projection, excluded, excluded_by)


def _refuse_a_falling_tail(estimate: _Estimate, periods, step: int, latest_dev, mask) -> None:
    """Refuse a tail that puts link factors below 1, under ``negative_increments="refuse"``.

    The tail stays out of the fitted values, so a tail attached before the
    last age with steps below 1 no longer makes a negative fitted mean; it
    makes the expected payments of the future cells after its attachment
    negative in every refit instead, and those are the means the process noise
    is drawn about. This keeps the refusal such a tail met when it was in the
    fitted values, with the same reason and the same way out. Only the links
    from the attachment on are the tail's: a factor below 1 before it is the
    data's, which only a negative observed increment makes, and
    :func:`fit_odp_bootstrap` has already refused that by its own name (this
    runs after it). A tail factor below 1 beyond the last age only, which never
    entered the fitted values, is reflected as before.
    """
    tail = estimate.tail
    if tail is None:
        return
    factors = np.asarray(estimate.factors, dtype=float)
    falling = [j for j in range(tail.attach_index, factors.size) if factors[j] < 1]
    if not falling:
        return
    beta = np.asarray(estimate.beta, dtype=float)
    latest = np.asarray(latest_dev, dtype=int)
    cells = [
        RefusedCell(None, periods[i], (j + 1) * step, None)
        for i in range(mask.shape[0])
        for j in range(int(latest[i]) + 1, beta.size)
        if j - 1 in falling
    ]
    raise Refusal(
        "negative_fitted_mean",
        "the tail puts link factors below 1 ({links}), so the expected payment of the "
        "future cells after them is negative in every refit ({cells}), and the "
        "over-dispersed Poisson bootstrap needs a positive mean to draw its noise "
        "about. Pass negative_increments='reflect' to draw it through sign() and |m|, "
        "as chainladder-python and R do",
        option="negative_increments",
        options=("negative_increments", "cells"),
        cells=cells,
        links=[((j + 1) * step, (j + 2) * step) for j in falling],
    )


def _reproduced(used: np.ndarray, latest_dev, mask: np.ndarray) -> np.ndarray:
    """``(n_w, n_d)`` bool: the observed cells the central factors reproduce exactly.

    The fitted cumulative at age ``j`` is the latest amount divided back by the
    factors from ``j`` to the latest age, so it equals the observed cumulative
    whenever each of those factors is the origin's own link ratio: the only
    ratio left at that age (an average of one ratio is that ratio, for every
    average). A fitted increment is exact when the fitted cumulatives at both
    of its ends are. The factors are the ones before any tail, so none of them
    came from a curve.
    """
    n_w, n_d = mask.shape
    sole = used & (used.sum(axis=0) == 1)[None, :]
    reproduced = np.zeros((n_w, n_d), dtype=bool)
    for i, k in enumerate(np.asarray(latest_dev, dtype=int)):
        reproduced[i, k] = True
        for j in range(k - 1, -1, -1):
            reproduced[i, j] = reproduced[i, j + 1] and sole[i, j]
    exact = np.zeros((n_w, n_d), dtype=bool)
    exact[:, 0] = reproduced[:, 0]
    exact[:, 1:] = reproduced[:, :-1]
    return exact & mask


@dataclass(frozen=True)
class RunoffMeans:
    """What :func:`future_cell_means` finds for a set of simulated triangles.

    ``means`` is ``(S, n_w, n_cols)``: the noiseless payment expected in each
    future cell of each draw, 0 on the observed cells; the last column is the
    tail's development when there is a tail. ``ultimate`` and ``latest`` are
    ``(S, n_w)``, the refit's ultimates and the simulated latest amounts.
    ``factors`` is ``(S, n_links)``, each draw's link factors after the tail's
    attachment, and ``tail_factor`` ``(S,)`` each draw's tail factor (``None``
    without a tail).
    ``unit`` (``(S,)``) marks a draw in which some link had no positive volume
    (or no usable ratio) and took factor 1.0; ``tail_fallback`` (``(S,)``) a
    draw whose tail curve failed its checks and used the central fit's.
    """

    means: np.ndarray
    ultimate: np.ndarray
    latest: np.ndarray
    factors: np.ndarray
    tail_factor: np.ndarray | None
    unit: np.ndarray
    tail_fallback: np.ndarray


def future_cell_means(
    projection: RunoffProjection,
    pseudo: np.ndarray,
    *,
    prior_multiplier: np.ndarray | None = None,
) -> RunoffMeans:
    """Refit and project a stack of simulated cumulative triangles.

    ``pseudo`` is ``(S, n_w, n_d)`` cumulatives; only the observed cells are
    read. Per draw: the link factors from the pairs ``projection.keep`` holds
    (``kernels.links.link_factors_many``); 1.0 at a link with no positive
    volume or no usable ratio (counted in ``unit``) and at the links with no
    pair at all (every draw, as in the central fit, not counted); the tail
    applied to those factors, a curve refitted to each draw's own factors
    (``kernels.tail.apply_tail``), falling back to the central fit's tail where
    the curve fails its checks; the share reported at each age
    (``kernels.conventional.pattern_beta``); and the ultimates of the method
    (``kernels.conventional.project_ultimates``) from the SIMULATED latest
    amounts, as in R and chainladder-python. The central fit runs the same two
    ``conventional`` functions.

    Each draw's reserve ``U - L`` is split over its future cells in the
    pattern's proportions, ``(p_j - p_(j-1)) / (1 - p_k)`` for an origin at age
    ``k``, with ``p`` the share reported and a final ``p = 1`` for the tail
    column. For Bornhuetter-Ferguson that is ``E (p_j - p_(j-1))``, the a priori
    ultimate times the share reported in the cell.
    """
    n_sims, n_w, n_d = pseudo.shape
    n_links = n_d - 1
    step = projection.dev_grain_months
    factors = link_factors_many(
        pseudo[:, :, :-1], pseudo[:, :, 1:], projection.keep, projection.average
    )
    random_unit = np.isnan(factors) & ~projection.unit[None, :]
    unit = random_unit.any(axis=1)
    factors[np.isnan(factors)] = 1.0
    tail_factor = None
    fallback = np.zeros(n_sims, dtype=bool)
    if projection.tail is not None:
        fit = apply_tail(factors, step, projection.tail, on_error="flag")
        fallback = ~np.asarray(fit.ok, dtype=bool)
        tailed = np.array(fit.factors, dtype=float)
        tail_factor = np.array(fit.tail_factor, dtype=float)
        if fallback.any():
            central = projection.central_tail
            k = central.attach_index
            tailed[fallback] = factors[fallback]
            tailed[np.ix_(fallback, np.arange(k, n_links))] = central.factors[k:n_links]
            tail_factor[fallback] = float(central.tail_factor)
        factors = tailed
    with np.errstate(all="ignore"):
        beta = pattern_beta(factors, tail_factor)
        k = projection.latest_dev
        latest = pseudo[:, np.arange(n_w), k]
        developed = beta[:, k]
        projected = project_ultimates(
            projection.method,
            latest,
            developed,
            premium=projection.premium,
            expected_loss_ratio=projection.expected_loss_ratio,
            weights=projection.weights,
            trend_factor=projection.trend_factor,
            n_iters=projection.n_iters,
            prior_multiplier=prior_multiplier,
        )
        ultimate = projected["ultimate"]
        # the share reported at each age, and 1 at ultimate after a tail
        shares = (
            beta if tail_factor is None else np.concatenate([beta, np.ones((n_sims, 1))], axis=1)
        )
        n_cols = shares.shape[1]
        step_share = np.zeros((n_sims, n_cols))
        step_share[:, 1:] = shares[:, 1:] - shares[:, :-1]
        unreported = 1.0 - shares[:, k]  # (S, n_w)
        reserve = ultimate - latest
        spread = np.where(unreported != 0, reserve / unreported, 0.0)
        future = np.arange(n_cols)[None, :] > k[:, None]  # (n_w, n_cols)
        means = np.where(future[None], spread[:, :, None] * step_share[:, None, :], 0.0)
    return RunoffMeans(means, ultimate, latest, factors, tail_factor, unit, fallback)


@dataclass(frozen=True)
class RunoffDraws:
    """What :func:`draw_runoff` returns.

    Attributes
    ----------
    ibnr : numpy.ndarray
        ``(n_draws, n_w)`` future payments to ultimate per origin, each finite.
    unit : numpy.ndarray
        ``(n_draws,)`` bool: some refit link had no positive volume and took
        factor 1.0.
    tail_fallback : numpy.ndarray
        ``(n_draws,)`` bool: the draw's tail curve failed its checks and the
        central fit's curve was used.
    prior_multiplier : numpy.ndarray or None
        ``(n_draws,)`` the a priori multipliers, ``None`` without them.
    """

    ibnr: np.ndarray
    unit: np.ndarray
    tail_fallback: np.ndarray
    prior_multiplier: np.ndarray | None


def draw_runoff(
    boot: ODPBootstrapFit,
    projection: RunoffProjection,
    *,
    n_draws: int,
    seed: np.random.SeedSequence | int | None,
    process: str = "gamma",
    process_noise: bool = True,
    prior_cv: float = 0.0,
    chunk_draws: int | None = None,
    residual_index: np.ndarray | None = None,
    prior_multiplier: np.ndarray | None = None,
) -> RunoffDraws:
    """Simulate the whole run-off ``n_draws`` times.

    Per draw: resample the pool into every observed cell (``r* sqrt(|m|) +
    m``, the cells past each origin's diagonal are drawn too and ignored, as in
    chainladder-python), cumulate, refit and project with
    :func:`future_cell_means`, add process noise to every future cell
    (``process``, one of :data:`ODP_PROCESS_LAWS`, reflected through the sign
    of a negative mean), and add up each origin's future cells.

    Randomness: ``seed`` (a ``SeedSequence``, or an int or ``None`` to make
    one) is split with ``spawn(3)`` into three independent generators, one for
    the residual draws, one for the process noise and one for the a priori
    multipliers. Each is read chunk after chunk, so the draws are the same bit
    for bit whatever ``chunk_draws`` is, the first ``k`` draws of a run are a
    ``k``-draw run, and changing ``process`` (or turning the noise off) leaves
    the simulated triangles unchanged. A residual is picked with one uniform
    number per cell, ``floor(u * n_pool)``, so each cell takes exactly one
    64-bit number from its stream whatever the pool's size.

    ``prior_cv`` (Bornhuetter-Ferguson, Benktander and Cape Cod only) draws one
    a priori multiplier per draw, shared by every origin of the draw: lognormal
    with mean 1 and coefficient of variation ``prior_cv``, ``exp(s Z - s^2/2)``
    with ``s^2 = log(1 + prior_cv^2)``. ``prior_cv=0`` draws none and reads
    nothing from the multiplier stream.

    ``chunk_draws`` is how many draws are held in memory at once; ``None`` sizes
    a chunk to about 64 MB of working arrays. ``residual_index``
    (``(n_draws, n_w, n_d)`` indices into ``boot.pool``) and
    ``prior_multiplier`` (``(n_draws,)``) replace the draws from the first and
    the third stream, so a test can feed another library's random numbers in.

    A draw that is not finite is never turned into a number: if any is left,
    the run is refused with ``result_not_finite`` naming how many.
    """
    _choose("process", process, ODP_PROCESS_LAWS)
    count = _whole(n_draws, "n_draws", 1)
    number = isinstance(prior_cv, int | float | np.integer | np.floating)
    cv = float(prior_cv) if number and not isinstance(prior_cv, bool | np.bool_) else math.nan
    if not np.isfinite(cv) or cv < 0:
        raise Refusal(
            "invalid_option",
            "prior_cv must be a finite number of 0 or more, got {given}",
            option="prior_cv",
            given=prior_cv,
        )
    if cv > 0 and not projection.has_prior:
        raise Refusal(
            "invalid_option",
            "prior_cv varies the a priori loss ratio, and the chain ladder has none",
            option="prior_cv",
            given=prior_cv,
        )
    if prior_multiplier is not None and not projection.has_prior:
        raise ValueError("prior_multiplier needs a method with an a priori loss ratio")
    if not np.isfinite(boot.phi) or not np.isfinite(boot.pool).all():
        raise Refusal(
            "result_not_finite",
            "the Pearson scale or a residual is not a finite number: the amounts are too "
            "large, or too far apart, for their squares to stay finite. Scale them (work in "
            "thousands, say) and scale the answer back",
            option="cells",
        )
    pool = boot.pool
    n_pool = pool.size
    n_w, n_d = boot.n_w, boot.n_d
    if residual_index is not None:
        residual_index = np.asarray(residual_index)
        if residual_index.shape != (count, n_w, n_d):
            raise ValueError(f"residual_index must be {(count, n_w, n_d)}")
        if residual_index.min() < 0 or residual_index.max() >= n_pool:
            raise ValueError(f"residual_index must index a pool of {n_pool}")
    if prior_multiplier is not None:
        prior_multiplier = np.asarray(prior_multiplier, dtype=float)
        if prior_multiplier.shape != (count,):
            raise ValueError(f"prior_multiplier must be ({count},)")
    if chunk_draws is None:
        per_draw = 8 * 8 * n_w * (n_d + 1)
        chunk = max(1, _CHUNK_BYTES // per_draw)
    else:
        chunk = _whole(chunk_draws, "chunk_draws", 1)
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    residual_stream, process_stream, prior_stream = (
        np.random.default_rng(child) for child in root.spawn(3)
    )
    mask = boot.obs_mask
    mean = np.where(mask, boot.fitted, 0.0)
    scale = np.sqrt(np.abs(mean))
    ibnr = np.empty((count, n_w))
    unit = np.zeros(count, dtype=bool)
    fallback = np.zeros(count, dtype=bool)
    multipliers = None
    if prior_multiplier is not None or cv > 0:
        multipliers = np.empty(count)
    spread = np.log1p(cv * cv)
    not_finite = 0
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        if residual_index is None:
            u = residual_stream.random((stop - start, n_w, n_d))
            index = np.minimum((u * n_pool).astype(np.intp), n_pool - 1)
        else:
            index = residual_index[start:stop]
        mult = None
        if prior_multiplier is not None:
            mult = prior_multiplier[start:stop]
        elif cv > 0:
            z = prior_stream.standard_normal(stop - start)
            mult = np.exp(np.sqrt(spread) * z - spread / 2)
        if mult is not None:
            multipliers[start:stop] = mult
        # Amounts near the largest double can overflow anywhere below; such a draw
        # is not finite, and is counted and refused, so numpy's warnings say
        # nothing more.
        with np.errstate(all="ignore"):
            pseudo = np.cumsum(np.where(mask, pool[index] * scale + mean, 0.0), axis=2)
            found = future_cell_means(projection, pseudo, prior_multiplier=mult)
            means = found.means
            # A draw whose means are not finite is counted and refused below; its
            # means are set to 0 first only so the noise law is never handed them.
            bad = ~np.isfinite(means).all(axis=(1, 2))
            if bad.any():
                means = np.where(bad[:, None, None], 0.0, means)
            if process_noise and boot.phi > 0:
                means = _od_process_noise(process_stream, means, boot.phi, law=process)
            chunk_ibnr = means.sum(axis=2)
        bad |= ~np.isfinite(chunk_ibnr).all(axis=1)
        not_finite += int(bad.sum())
        ibnr[start:stop] = chunk_ibnr
        unit[start:stop] = found.unit
        fallback[start:stop] = found.tail_fallback
    if not_finite:
        raise Refusal(
            "result_not_finite",
            f"{not_finite} of the {count} simulated draws are not finite numbers: the "
            "simulated triangles refit to factors too large for their products to stay "
            "finite. A tail curve fitted to each draw is the usual cause; a constant tail, "
            "or fewer development options, avoids it",
            option="n_draws",
            given=count,
        )
    return RunoffDraws(ibnr, unit, fallback, multipliers)


def _whole(value, name: str, least: int) -> int:
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or value < least
    ):
        raise Refusal(
            "invalid_option",
            f"{name} must be a whole number of {least} or more, got {{given}}",
            option=name,
            given=value,
        )
    return int(value)
