"""Engineered example tensors for the transformer loss reserving network.

A transcription of the feature builder in the companion study's R
implementation, written against ``kernels.nn_contract``'s company contract and
free of torch: the ``tlrn`` gallery entry converts what comes out of here.

WHAT AN EXAMPLE IS. One (company, accident year). Its tokens are the company's
(line, development lag) cells, line-major with the lag varying fastest, so token
``t`` is ``line * n_d + lag`` and example ``e`` is ``company * n_w + origin``.
Every company gets the same full-width line axis, and a line it does not write
is carried and marked rather than dropped, because the network attends across
lines and needs the axis to mean the same thing everywhere.

WHAT A CUTOFF IS. A 1-based calendar diagonal ``K``. A cell is visible when
``origin + lag + 1 <= K`` - the contract's ``cal_idx``. Every statistic below is
estimated from visible cells alone, which is what lets one fit be trained at
many cutoffs on one triangle: each cutoff is a complete "forecast the next
diagonals" task built out of the same data.

That rule is the one thing in this module worth checking by hand, because
breaking it costs nothing visible. A per-lag mean taken over the whole grid,
or a chain ladder factor built from a pair of cells one of which has not
happened yet, produces a finite, smooth, plausible feature that has seen the
answer. ``tests/test_nn_features.py`` perturbs every hidden cell and requires
every returned array except the targets to come back unchanged.

THE THREE FAMILIES OF FEATURES, in the order the input projection reads them:

1. cell-level: the standardised incremental paid loss ratio at each visible
   cell, the visibility flag itself, and whether the line is written;
2. company-level, repeated over the lags of a line: that company's own
   development ratios, the cumulative loss ratio reached at its latest visible
   lag, and - in the 13-feature form - the paid-to-incurred ratio, the incurred
   loss ratio and the case reserve loss ratio there;
3. position: how much of the development is already observed, and how many
   steps past it each token sits.

Two standardisations, deliberately different. A cell-level quantity is
standardised per (line, lag), because a first-year increment and a tenth-year
increment are not on the same scale. A company-level quantity is standardised
over the whole array of such quantities, because the comparison a network needs
there is "is this company far along for a company", not "for a lag".

EIGHT FEATURES OR THIRTEEN. Paid alone gives eight. Naming an incurred channel
and a case reserve channel adds five. They are named by ROLE rather than swept
as a list of feature fields, because each of the five is a different formula:
the builder needs to know which channel is the incurred amount and which is the
outstanding balance, and a generic channel list cannot say.

WHY THE RAW GRIDS. The cumulative amount and the cumulative loss ratio at an
arbitrary visible cell are inputs here, and the contract's ``x`` is an increment
over premium whose unusable cells are padding zeros. Summing those back up would
invent an amount at every padded cell, so this module reads ``values``, the raw
grid the contract carries beside the ratios.

CLIP LIMITS. Every one is a constant below with the reference implementation's
value. They are wide enough to be inactive on ordinary data and are there for
the cohort whose early cumulative is a rounding error, where an unclipped ratio
would be the largest number the network ever sees.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "mcl_cell_forecast",
    "origin_cl_log_factors",
    "pooled_cl_factors",
    "pooled_incremental_lr",
    "tlrn_features",
]

#: standard deviations a standardised feature is clipped at (reference: 5)
CLAMP_Z = 5.0
#: a spread at or below this counts as none, and the divisor falls back to 1
SD_FLOOR = 1e-8
#: own incremental-to-cumulative development ratio (reference: -0.5, 3)
OWN_RATIO_LIMITS = (-0.5, 3.0)
#: cumulative paid loss ratio at the latest visible lag (reference: 0, 5)
CUM_LOSS_RATIO_LIMITS = (0.0, 5.0)
#: paid over incurred at the latest visible lag (reference: 0, 2)
PAID_TO_INCURRED_LIMITS = (0.0, 2.0)
#: cumulative incurred loss ratio at the latest visible lag (reference: 0, 5)
INCURRED_LOSS_RATIO_LIMITS = (0.0, 5.0)
#: case reserve over premium at the latest visible lag (reference: -1, 3)
CASE_LOSS_RATIO_LIMITS = (-1.0, 3.0)
#: development factors are floored here so their log is strictly positive and
#: the head's softplus inverse is finite (reference: 1 + 1e-6)
FACTOR_FLOOR = 1.0 + 1e-6
#: the cumulative a projection starts from, floored in amount (reference: 1)
ANCHOR_FLOOR = 1.0
#: the premium a predicted increment is divided by (reference: 1e-8)
PREMIUM_FLOOR = 1e-8
#: pooled incremental loss ratios are floored here so their log is finite
#: (the companion study's value)
LR_FLOOR = 1e-4
#: what a line with no visible cell at any lag starts at
LR_NO_CELLS = 0.05

#: channel order of ``feat``'s last axis. The first eight are the paid-only
#: form; naming an incurred and a case channel appends the other five. The
#: order is part of the contract with the network, whose input projection is one
#: linear layer: two features swapped is a different model that still trains.
FEATURE_NAMES: tuple[str, ...] = (
    "observed incremental paid loss ratio",
    "observed flag",
    "line written",
    "context incremental paid loss ratio",
    "own incremental to cumulative ratio",
    "cumulative paid loss ratio at the latest visible lag",
    "fraction of lags observed",
    "steps ahead",
    "context incremental incurred loss ratio",
    "observed incremental incurred loss ratio",
    "paid to incurred",
    "incurred loss ratio",
    "case reserve loss ratio",
)
#: how many of :data:`FEATURE_NAMES` each form uses
N_FEATURES_PAID_ONLY = 8


def _finite_or(values: np.ndarray, fill: float) -> np.ndarray:
    """``values`` with every non-finite entry replaced, the reference's idiom."""
    return np.where(np.isfinite(values), values, fill)


def _sample_stats(values: np.ndarray) -> tuple[float, float]:
    """Mean and sample spread over every finite entry.

    Fewer than two finite values gives (0, 1), and a spread at or below
    :data:`SD_FLOOR` falls back to 1 while the mean is kept - so a quantity that
    is constant across the whole cohort standardises to zero rather than
    dividing by nothing. The spread is the sample one (divisor ``n - 1``), which
    is what the reference implementation's ``sd`` computes.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0, 1.0
    spread = float(v.std(ddof=1))
    return float(v.mean()), (spread if np.isfinite(spread) and spread > SD_FLOOR else 1.0)


def _lag_stats(arr: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    """Per (line, lag) mean and spread over every company's VISIBLE cells.

    ``arr`` is (n_c, n_l, n_w, n_d) with NaN where the value is unusable. At lag
    ``d`` the visible origins are ``w < cutoff - d``, which is exactly
    ``w + d + 1 <= cutoff``. A lag no origin has reached keeps (0, 1), so its
    feature is the raw value rather than a value standardised by numbers from a
    different lag.
    """
    _, n_l, n_w, n_d = arr.shape
    mu = np.zeros((n_l, n_d))
    sd = np.ones((n_l, n_d))
    for li in range(n_l):
        for d in range(n_d):
            visible_origins = min(cutoff - d, n_w)
            if visible_origins < 1:
                continue
            mu[li, d], sd[li, d] = _sample_stats(arr[:, li, :visible_origins, d])
    return mu, sd


def _lag_means(arr: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Each company's own mean over visible origins, per (company, line, lag).

    ``visible`` is the (n_w, n_d) cutoff mask. A (company, line, lag) with no
    finite visible value gets 0, which is the reference's non-finite rule.
    """
    n_c, n_l, _, n_d = arr.shape
    out = np.zeros((n_c, n_l, n_d))
    for d in range(n_d):
        rows = np.nonzero(visible[:, d])[0]
        if rows.size == 0:
            continue
        block = arr[:, :, rows, d]
        ok = np.isfinite(block)
        count = ok.sum(axis=2)
        total = np.where(ok, block, 0.0).sum(axis=2)
        out[:, :, d] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return out


def _own_ratios(cp: np.ndarray, cutoff: int) -> np.ndarray:
    """Each company's own incremental-to-cumulative ratio per (line, lag).

    At lag ``d`` the visible origins are ``w < cutoff - d``, the increment is
    ``cp[w, d] - cp[w, d - 1]`` and the base is ``cp[w, d - 1]``; both are summed
    over those origins before dividing, so the ratio is volume weighted. An
    origin whose base is not positive contributes to neither sum. Lag 0 has no
    predecessor and is 0 by definition.
    """
    n_c, n_l, n_w, n_d = cp.shape
    out = np.zeros((n_c, n_l, n_d))
    for d in range(1, n_d):
        visible_origins = min(cutoff - d, n_w)
        if visible_origins < 1:
            continue
        base = cp[:, :, :visible_origins, d - 1]
        later = cp[:, :, :visible_origins, d]
        ok = np.isfinite(base) & np.isfinite(later) & (base > 0)
        numerator = np.where(ok, later - base, 0.0).sum(axis=2)
        denominator = np.where(ok, base, 0.0).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = numerator / denominator
        out[:, :, d] = np.clip(_finite_or(ratio, 0.0), *OWN_RATIO_LIMITS)
    return out


def pooled_cl_factors(values_paid: np.ndarray, cutoff: int) -> np.ndarray:
    """Volume-weighted chain ladder factors pooled over companies, per line.

    ``values_paid`` is the contract's (n_c, n_l, n_w, n_d) grid of cumulative
    paid amounts, NaN where absent. Returns (n_l, n_d - 1): step ``d`` carries
    the factor from lag ``d`` to lag ``d + 1``.

    A step needs BOTH of its cells to have elapsed, so it pools the origins
    ``w < cutoff - d - 1`` - one fewer than the cells at lag ``d`` alone. A step
    no origin has reached, or whose base sums to nothing, is flat.

    Factors are floored just above 1. The network's head is a softplus over a
    log factor, so a factor at or below 1 has no finite parameter that produces
    it; the floor is what lets a fitted head be initialised at these values.
    Paid recoveries therefore cannot be represented here, which is a limit of
    the head rather than of the data.
    """
    cp = np.asarray(values_paid, dtype=float)
    _, n_l, n_w, n_d = cp.shape
    factors = np.ones((n_l, n_d - 1))
    for li in range(n_l):
        for d in range(n_d - 1):
            visible_origins = min(cutoff - d - 1, n_w)
            if visible_origins < 1:
                continue
            base = np.nansum(cp[:, li, :visible_origins, d])
            later = np.nansum(cp[:, li, :visible_origins, d + 1])
            if base > 0:
                factors[li, d] = later / base
    return np.maximum(factors, FACTOR_FLOOR)


def origin_cl_log_factors(
    values_paid: np.ndarray,
    written: np.ndarray,
    cutoff: int,
    pooled_tail_one: np.ndarray,
) -> np.ndarray:
    """Each company's own log chain ladder factors, per (company, line, step).

    The same volume-weighted ratio as :func:`pooled_cl_factors`, computed from
    one company's own origins, with two different answers when it cannot be:

    * a line the company does not write stays flat, factor 1. It has no
      development to describe and borrowing the market's would put a projection
      on a line that does not exist;
    * a written line whose step has no usable pair borrows ``pooled_tail_one``,
      the pooled factors with the steps no visible pair reaches set to 1. That
      step will still be projected across, so the pooled estimate is the only
      honest number available for it.

    Unlike the pooled version, a company's own ratio pairs its cells: an origin
    contributes to both sums or to neither. Pooling over many companies makes
    the difference immaterial; within one company it is the difference between
    a factor and an artefact.
    """
    cp = np.asarray(values_paid, dtype=float)
    n_c, n_l, n_w, n_d = cp.shape
    factors = np.ones((n_c, n_l, n_d - 1))
    for d in range(n_d - 1):
        visible_origins = min(cutoff - d - 1, n_w)
        if visible_origins < 1:
            continue
        base = cp[:, :, :visible_origins, d]
        later = cp[:, :, :visible_origins, d + 1]
        ok = np.isfinite(base) & np.isfinite(later)
        denominator = np.where(ok, base, 0.0).sum(axis=2)
        numerator = np.where(ok, later, 0.0).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(denominator > 0, numerator / denominator, np.nan)
        usable = np.isfinite(ratio) & (ratio > 0)
        factors[:, :, d] = np.where(usable, ratio, pooled_tail_one[None, :, d])
    factors = np.where(np.asarray(written, dtype=bool)[:, :, None], factors, 1.0)
    return np.log(factors)


def pooled_incremental_lr(contract: dict[str, Any], cutoff: int) -> np.ndarray:
    """Mean incremental paid loss ratio over every cell known at ``cutoff``, per line and lag.

    Returns (n_l, n_d): at lag ``d`` the mean of the paid increment over premium of
    every visible cell of that line, over companies and accident years, floored at
    :data:`LR_FLOOR`. A cell is visible when its calendar diagonal is at or before
    ``cutoff`` and the company writes the line; an unusable increment (the
    contract's ``x_obs`` is False) is left out. A lag no visible cell reaches takes
    the value at the line's last lag that one does, and a line none reaches takes
    :data:`LR_NO_CELLS`.

    This is the companion study's definition, a plain mean and not a premium-weighted
    one. It is what the premium head starts from and what it falls back to at a lag no
    training target supervised, so it reads visible cells only, exactly as
    :func:`pooled_cl_factors` does: a ratio that saw a hidden cell would be a fallback
    that had seen the answer.
    """
    x = np.asarray(contract["x"], dtype=float)[:, :, 0]
    x_obs = np.asarray(contract["x_obs"], dtype=bool)[:, :, 0]
    written = np.asarray(contract["line_mask"], dtype=bool)
    visible = np.asarray(contract["cal_idx"], dtype=int) <= int(cutoff)
    ok = x_obs & written[:, :, None, None] & visible[None, None] & np.isfinite(x)
    count = ok.sum(axis=(0, 2))  # (n_l, n_d)
    total = np.where(ok, x, 0.0).sum(axis=(0, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    out = np.where(np.isfinite(mean), np.maximum(mean, LR_FLOOR), np.nan)
    for li in range(out.shape[0]):
        known = np.flatnonzero(np.isfinite(out[li]))
        out[li, np.isnan(out[li])] = out[li, known[-1]] if known.size else LR_NO_CELLS
    return out


def mcl_cell_forecast(
    contract: dict[str, Any],
    cutoff: int,
    *,
    lk: np.ndarray,
    c_lk: np.ndarray,
    p_lk: np.ndarray,
    has_history: np.ndarray,
) -> dict[str, np.ndarray]:
    """The multivariate chain ladder's cell forecast, in the layout of the head's.

    One company at a time: the lines it writes are fitted jointly on the paid
    cumulatives visible at ``cutoff`` (``kernels.multivariate_cl.point_grid``) and
    every origin is rolled forward from its latest visible lag. Returns
    ``mcl_pred`` (n_ex, n_tok), the predicted incremental loss ratios, and
    ``mcl_C`` (n_ex, n_l, n_d), the cumulative grid, in exactly the layouts and
    with exactly the convention of the network's own output: cells at or before
    an origin's latest visible lag are held at the starting balance, so a blend of
    the two is the same kind of object whatever its weights, and the projected
    cells add the forecast increments to the same floored ``c_lk``.

    Reads visible cells only, like every other statistic in this module. A
    forecast increment that is not finite is taken as zero, and a line the
    company does not write has no forecast.
    """
    from ibnr.kernels.multivariate_cl import point_grid

    cp = np.asarray(contract["values"], dtype=float)[:, :, 0]  # (n_c, n_l, n_w, n_d)
    written = np.asarray(contract["line_mask"], dtype=bool)
    n_c, n_l, n_w, n_d = cp.shape
    visible = np.asarray(contract["cal_idx"], dtype=int) <= int(cutoff)
    held_c = np.broadcast_to(c_lk.reshape(n_c, n_w, n_l)[..., None], (n_c, n_w, n_l, n_d)).copy()
    pred = np.zeros((n_c, n_w, n_l, n_d))
    pred[..., 0] = c_lk.reshape(n_c, n_w, n_l) / p_lk.reshape(n_c, n_w, n_l)
    for ci in range(n_c):
        lines = np.nonzero(written[ci])[0]
        if lines.size == 0:
            continue
        grid, _ = point_grid(cp[ci, lines], visible)
        for ki, li in enumerate(lines):
            for w in range(n_w):
                start = int(lk[ci * n_w + w])
                if not has_history[w] or start >= n_d:
                    continue
                path = np.nan_to_num(
                    np.diff(grid[ki, w, start - 1 :]), nan=0.0, posinf=0.0, neginf=0.0
                )
                row = ci * n_w + w
                pred[ci, w, li, start:] = path / p_lk[row, li]
                held_c[ci, w, li, start:] = c_lk[row, li] + np.cumsum(path)
    n_ex = n_c * n_w
    return {
        "mcl_pred": pred.reshape(n_ex, n_l * n_d),
        "mcl_C": held_c.reshape(n_ex, n_l, n_d),
    }


def _resolve_channels(
    contract: dict[str, Any], incurred_field: str | None, case_field: str | None
) -> tuple[int | None, int | None]:
    """Which contract channels the incurred and case roles name, or (None, None)."""
    if (incurred_field is None) != (case_field is None):
        raise ValueError(
            "incurred_field and case_field are given together or both left out: the "
            "13-feature form reads the incurred increment, the paid-to-incurred ratio, "
            "the incurred loss ratio and the case reserve loss ratio, and there is no "
            f"form carrying one of them without the other. Got incurred_field="
            f"{incurred_field!r}, case_field={case_field!r}"
        )
    if incurred_field is None:
        return None, None
    fields = list(contract["fields"])
    kinds = tuple(contract["field_kinds"])
    for role, name in (("incurred_field", incurred_field), ("case_field", case_field)):
        if name not in fields:
            raise ValueError(
                f"{role}={name!r} is not a channel of this contract, which carries "
                f"{fields}. Name the field in nn_company_data's feature_fields first"
            )
    i_inc, i_case = fields.index(incurred_field), fields.index(case_field)
    if kinds[i_case] != "level":
        raise ValueError(
            f"case_field={case_field!r} is channel {i_case} of this contract and is "
            f"carried as an {kinds[i_case]}. The case reserve is an evaluation-date "
            "balance and its difference is the case movement, a different quantity, so "
            "name it in nn_company_data's level_fields as well"
        )
    if kinds[i_inc] != "increment":
        raise ValueError(
            f"incurred_field={incurred_field!r} is channel {i_inc} of this contract and "
            f"is carried as a {kinds[i_inc]}. The incurred feature is the incurred "
            "EMERGENCE, so leave the field out of nn_company_data's level_fields"
        )
    return i_inc, i_case


def _refuse_holes(
    contract: dict[str, Any], cp: np.ndarray, written: np.ndarray, visible: np.ndarray, cutoff: int
) -> None:
    """Refuse a written line missing a cumulative the cutoff does not hide.

    Every projection here starts from the cumulative paid at an origin's latest
    visible lag. A missing one reads as a zero balance, which the anchor floor
    turns into one dollar, and that company's whole reserve is then built on a
    one-dollar starting point - finite, plausible and wrong. The reference
    implementation never meets this because its cohort selection requires
    complete grids; a triangle that is not complete is refused here by name.
    """
    missing = written[:, :, None, None] & visible[None, None] & ~np.isfinite(cp)
    if not missing.any():
        return
    ci, li, w, d = (int(a[0]) for a in np.nonzero(missing))
    company = contract["companies"].iloc[ci].to_dict()
    lob = contract["lob_levels"][li]
    origin = contract["origin_periods"][w]
    raise ValueError(
        f"company {company} writes {lob!r} but has a hole in its cumulative paid loss at "
        f"origin {origin} development step {d + 1}, on calendar diagonal {w + d + 1}, "
        f"which the cutoff {cutoff} does not hide ({int(missing.sum())} such cell(s)). "
        "Every projection starts from the cumulative at an origin's latest visible lag, "
        "so a missing one would silently anchor that origin at the floor. Fill the cell, "
        "drop the cohort, or build the features at an earlier cutoff"
    )


def tlrn_features(
    contract: dict[str, Any],
    *,
    cutoff: int,
    target_lo: int,
    target_hi: int,
    incurred_field: str | None = None,
    case_field: str | None = None,
    clamp: float = CLAMP_Z,
    with_mcl: bool = False,
    max_lag: int | None = None,
) -> dict[str, Any]:
    """Build one training example per (company, accident year) at ``cutoff``.

    ``contract`` is a ``kernels.nn_contract.nn_company_data`` dict whose channel
    0 is the paid target. ``cutoff`` is a 1-based calendar diagonal; every input
    is estimated from cells on or before it. ``target_lo`` and ``target_hi``
    bound the diagonals that are scored, and ``target_lo`` must be past the
    cutoff.

    Returns numpy arrays only - the entry converts them to tensors:

    ``feat`` (n_ex, n_tok, n_feat), ``target`` and ``target_mask`` and
    ``premium`` (n_ex, n_tok), ``c_lk`` and ``anchor_start`` and ``p_lk``
    (n_ex, n_l), ``lk`` and ``has_history`` (n_ex,), ``written`` (n_ex, n_l),
    ``line_ix`` and ``lag_ix`` (n_tok,) 1-based, ``fallback_logf``
    (n_l, n_d - 1), ``anchor_logf`` (n_ex, n_l, n_d - 1), ``example_company``
    and ``example_origin`` (n_ex,), plus ``n_dropped``, ``feature_names`` and
    the sizes.

    An origin whose first development period has not elapsed at the cutoff is
    KEPT in the tensors and dropped from the scoring mask: it has no cumulative
    to project from, so its cells carry nothing a development model can learn
    from, and deleting the row instead would make the example axis depend on the
    cutoff. ``n_dropped`` counts the scoring cells that costs.

    ``max_lag`` keeps only the scored cells at development lag ``max_lag`` or earlier
    (1-based): the cells whose steps the cutoff has already seen when it is the cutoff.

    ``with_mcl=True`` adds ``mcl_pred`` and ``mcl_C`` (see :func:`mcl_cell_forecast`),
    the multivariate chain ladder's forecast of the same cells from the same
    visible data, which the ``mcl_blend`` member mixes the network's forecast with.
    """
    x = np.asarray(contract["x"], dtype=float)
    if x.ndim != 5:
        raise ValueError(
            "tlrn_features needs a company contract from nn_company_data: its 'x' is "
            f"(n_c, n_lines, n_fields, n_w, n_d) and this one has {x.ndim} dimensions. "
            "The features are engineered across all of a company's lines at once, so the "
            "flat (company, line) contract cannot serve them"
        )
    i_inc, i_case = _resolve_channels(contract, incurred_field, case_field)
    has_incurred = i_inc is not None

    n_c, n_l, _, n_w, n_d = x.shape
    cutoff, target_lo, target_hi = int(cutoff), int(target_lo), int(target_hi)
    last = n_w + n_d - 1
    if not 1 <= cutoff <= last:
        raise ValueError(
            f"cutoff {cutoff} is outside this triangle's calendar diagonals, which run 1 to {last}"
        )
    if target_lo <= cutoff:
        raise ValueError(
            f"target_lo {target_lo} must be past the cutoff {cutoff}: a cell on or before "
            "the cutoff is an input, so scoring it would score the network on what it was "
            "shown"
        )
    # target_hi == target_lo - 1 is the EMPTY window, and it is a real state
    # rather than a mistake: a triangle with every cell observed has nothing past
    # its cutoff to forecast, so the scoring mask is all zero and the reserve is
    # zero. Anything more inverted than that is a swapped pair of arguments.
    if not target_lo - 1 <= target_hi <= last:
        raise ValueError(
            f"target_hi {target_hi} must be at least target_lo - 1 = {target_lo - 1} and at "
            f"most {last}, this triangle's last calendar diagonal. target_lo - 1 exactly "
            "means an empty scoring window, which is what a fully developed triangle has"
        )

    x_obs = np.asarray(contract["x_obs"], dtype=bool)
    values = np.asarray(contract["values"], dtype=float)
    written = np.asarray(contract["line_mask"], dtype=bool)  # (n_c, n_l)
    premium = np.asarray(contract["premium"], dtype=float)  # (n_c, n_l, n_w)
    cal = np.asarray(contract["cal_idx"], dtype=int)  # (n_w, n_d)
    visible = cal <= cutoff
    scored = (cal >= target_lo) & (cal <= target_hi)
    if max_lag is not None:
        scored = scored & (np.arange(1, n_d + 1) <= int(max_lag))[None, :]

    # the paid target, in both forms the reference keeps: NaN where the
    # increment is unusable, and the contract's padded zeros for the tensors
    paid_ratio_padded = x[:, :, 0]
    paid_ratio = np.where(x_obs[:, :, 0], paid_ratio_padded, np.nan)
    cp = values[:, :, 0]  # cumulative paid amounts, NaN where absent
    _refuse_holes(contract, cp, written, visible, cutoff)
    with np.errstate(invalid="ignore", divide="ignore"):
        cum_paid_ratio = cp / premium[..., None]

    mu_p, sd_p = _lag_stats(paid_ratio, cutoff)
    context_paid = _lag_means(paid_ratio, visible)
    own = _own_ratios(cp, cutoff)

    if has_incurred:
        incurred_ratio_padded = x[:, :, i_inc]
        incurred_ratio = np.where(x_obs[:, :, i_inc], incurred_ratio_padded, np.nan)
        # a non-positive incurred amount is no denominator for a paid-to-incurred
        # ratio, so it is dropped rather than clipped
        ci_amount = values[:, :, i_inc]
        ci_amount = np.where(np.isfinite(ci_amount) & (ci_amount <= 0), np.nan, ci_amount)
        with np.errstate(invalid="ignore", divide="ignore"):
            cum_incurred_ratio = values[:, :, i_inc] / premium[..., None]
            cum_case_ratio = values[:, :, i_case] / premium[..., None]
        mu_i, sd_i = _lag_stats(incurred_ratio, cutoff)
        context_incurred = _lag_means(incurred_ratio, visible)

    # the latest visible lag of each origin, 1-based and clamped into the grid.
    # An origin with no elapsed development keeps lk = 1 so every index below is
    # in range; its values are replaced by the no-history defaults straight after
    lk = np.clip(np.minimum(cutoff - np.arange(n_w), n_d), 1, n_d)  # (n_w,)
    has_history = (cutoff - np.arange(n_w)) >= 1  # (n_w,)
    at_latest = np.broadcast_to((lk - 1)[None, None, :, None], (n_c, n_l, n_w, 1))

    def latest(grid: np.ndarray) -> np.ndarray:
        """``grid`` at each origin's latest visible lag: (n_c, n_l, n_w)."""
        return np.take_along_axis(grid, at_latest, axis=3)[..., 0]

    hist = has_history[None, None, :]
    cum_amount = np.where(hist, np.maximum(_finite_or(latest(cp), 0.0), 0.0), 0.0)
    cum_lr = np.where(
        hist, np.clip(_finite_or(latest(cum_paid_ratio), 0.0), *CUM_LOSS_RATIO_LIMITS), 0.0
    )
    if has_incurred:
        with np.errstate(invalid="ignore", divide="ignore"):
            paid_over_incurred = latest(cp) / latest(ci_amount)
        paid_over_incurred = np.where(
            hist, np.clip(_finite_or(paid_over_incurred, 1.0), *PAID_TO_INCURRED_LIMITS), 1.0
        )
        incurred_lr = np.where(
            hist,
            np.clip(_finite_or(latest(cum_incurred_ratio), 0.0), *INCURRED_LOSS_RATIO_LIMITS),
            0.0,
        )
        case_lr = np.where(
            hist,
            np.clip(_finite_or(latest(cum_case_ratio), 0.0), *CASE_LOSS_RATIO_LIMITS),
            0.0,
        )

    def z(value: np.ndarray, mean, spread) -> np.ndarray:
        return np.clip((value - mean) / spread, -clamp, clamp)

    def by_company(quantity: np.ndarray) -> np.ndarray:
        """(n_c, n_l, n_w) standardised over the whole array, as (n_c, n_w, n_l, 1)."""
        return np.moveaxis(z(quantity, *_sample_stats(quantity)), 2, 1)[..., None]

    # every channel is built as (company, origin, line, lag) and flattened to
    # (example, token) at the end: example = company * n_w + origin, token =
    # line * n_d + lag, which is the layout the head and the scorer both index
    shape = (n_c, n_w, n_l, n_d)
    visible_e = visible[None, :, None, :]
    written_e = written[:, None, :, None]
    paid_e = np.moveaxis(paid_ratio_padded, 2, 1)
    steps_ahead = np.maximum(
        np.arange(1, n_d + 1)[None, None, None, :] - lk[None, :, None, None], 0
    )
    channels = [
        z(paid_e, mu_p, sd_p) * visible_e,
        visible_e.astype(float),
        written_e.astype(float),
        z(context_paid, mu_p, sd_p)[:, None],
        z(own, *_sample_stats(own))[:, None],
        by_company(cum_lr),
        (lk / n_d)[None, :, None, None],
        steps_ahead / n_d,
    ]
    if has_incurred:
        channels += [
            z(context_incurred, mu_i, sd_i)[:, None],
            z(np.moveaxis(incurred_ratio_padded, 2, 1), mu_i, sd_i) * visible_e,
            by_company(paid_over_incurred),
            by_company(incurred_lr),
            by_company(case_lr),
        ]
    feat = np.stack([np.broadcast_to(c, shape) for c in channels], axis=-1)
    feat = _finite_or(feat, 0.0)

    n_ex, n_tok = n_c * n_w, n_l * n_d
    scored_e = scored[None, :, None, :] & written_e
    premium_e = np.broadcast_to(np.moveaxis(_finite_or(premium, 0.0), 2, 1)[..., None], shape)

    def per_line(quantity: np.ndarray) -> np.ndarray:
        """(n_c, n_l, n_w) -> (n_ex, n_l), one row per example."""
        return np.moveaxis(quantity, 2, 1).reshape(n_ex, n_l)

    pooled = pooled_cl_factors(cp, cutoff)
    step_index = np.arange(1, n_d)  # 1-based development step
    # a step no visible pair reaches is not an estimated tail: it is flat, and a
    # flat factor is a log factor of zero
    pooled_tail_one = np.where(step_index[None, :] >= cutoff, 1.0, pooled)
    anchor_logf = origin_cl_log_factors(cp, written, cutoff, pooled_tail_one)

    out = {
        "feat": feat.reshape(n_ex, n_tok, len(channels)),
        "target": np.moveaxis(paid_ratio_padded, 2, 1).reshape(n_ex, n_tok),
        "target_mask": (scored_e & has_history[None, :, None, None])
        .astype(float)
        .reshape(n_ex, n_tok),
        "premium": premium_e.reshape(n_ex, n_tok),
        "c_lk": per_line(np.where(hist, np.maximum(cum_amount, ANCHOR_FLOOR), ANCHOR_FLOOR)),
        "anchor_start": per_line(np.where(hist, _finite_or(latest(cp), 0.0), 0.0)),
        "p_lk": per_line(
            np.where(hist, np.maximum(_finite_or(premium, 0.0), PREMIUM_FLOOR), PREMIUM_FLOOR)
        ),
        "lk": np.tile(lk, n_c),
        "has_history": np.tile(has_history, n_c),
        "written": np.repeat(written, n_w, axis=0),
        "line_ix": np.repeat(np.arange(1, n_l + 1), n_d),
        "lag_ix": np.tile(np.arange(1, n_d + 1), n_l),
        "fallback_logf": np.log(pooled_tail_one),
        # the premium head's counterpart of the fallback: the pooled incremental
        # loss ratio per (line, lag) as known at this cutoff
        "fallback_lr": pooled_incremental_lr(contract, cutoff),
        # the tokens this cutoff has revealed, which a masked attention may read
        "visible": np.broadcast_to(visible_e & written_e, shape).astype(float).reshape(n_ex, n_tok),
        "anchor_logf": np.repeat(anchor_logf, n_w, axis=0),
        "example_company": np.repeat(np.arange(n_c), n_w),
        "example_origin": np.tile(np.arange(n_w), n_c),
        # scoring cells lost to origins with no elapsed development
        "n_dropped": int(np.broadcast_to(scored_e, shape)[:, ~has_history].sum()),
        "feature_names": FEATURE_NAMES if has_incurred else FEATURE_NAMES[:N_FEATURES_PAID_ONLY],
        "n_l": n_l,
        "n_d": n_d,
        "n_ex": n_ex,
        "n_tok": n_tok,
        "n_feat": len(channels),
        "cutoff": cutoff,
        "target_lo": target_lo,
        "target_hi": target_hi,
    }
    if with_mcl:
        out.update(
            mcl_cell_forecast(
                contract,
                cutoff,
                lk=out["lk"],
                c_lk=out["c_lk"],
                p_lk=out["p_lk"],
                has_history=has_history,
            )
        )
    return out
