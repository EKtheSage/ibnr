"""Triangle -> standardized multi-line model data. THE multi-LOB data contract.

The multiline analogue of ``contract.py``: one canonical mapping from a
one-company, many-lines-of-business cumulative Triangle to dense arrays,
consumed by the statistical dependence models (SUR, copula) and by the
multi-line transformer (``gallery/nn/transformer_ml``, for the target layout
and predictive assembly). Per CLAUDE.md decision 3 the contract is defined
once here and every consumer takes the identical dict - no entry grows its
own data prep.

Cross-refs: ``kernels/contract.py`` (the single-cohort Stan contract this
mirrors), ``kernels/predictive.py`` (``PredictiveDistribution``),
``gallery/statistical/sur/card.md`` and ``gallery/statistical/copula_glm/card.md``.

THE CONTRACT - keys returned by ``multiline_data`` (shape; dtype; meaning):

- ``n_lob``, ``n_w``, ``n_d`` : int - sizes of the LOB, origin and dev axes.
  ``n_d`` is the LARGEST observed dev step, so the (n_w, n_d) grid is a
  rectangle whose unobserved corner is NaN, not a ragged triangle.
- ``lobs``          : list, len n_lob; the sorted distinct LOB labels. Index
  ``k`` into every LOB axis is defined by this list and nothing else.
- ``origin_periods``: list[dt.date], len n_w; sorted ascending. Index ``w``.
- ``cum``           : (n_lob, n_w, n_d) float64 - CUMULATIVE loss in triangle
  units. Absent cells are NaN (never 0: a 0 in the Schedule P mart is a real
  reported zero, see CLAUDE.md "absent = unobserved, zero = explicit").
  Dev step ``d`` (1-based, as in ``stan_data``) lives at index ``d - 1``,
  where ``d = dev_lag // dev_grain_months``.
- ``obs_mask``      : (n_lob, n_w, n_d) bool - ``~isnan(cum)``. True = observed
  (a training cell); False = to be predicted (or outside the study). Invariant
  enforced below: ``obs_mask[k] == obs_mask[0]`` for every k.
- ``dev_grain_months``: int - months per dev step (12 for annual triangles).
- ``lob_column``    : str - which segment column defined the LOB axis, so a
  consumer can label its outputs without re-deriving it.
- ``units``         : str | None - carried through from Triangle metadata onto
  the eventual ``PredictiveDistribution``.
- ``premium``       : (n_lob, n_w) float64, ONLY when ``premium_field`` is
  given - booked earned premium per (lob, origin) at its latest evaluation.
  Guaranteed complete and strictly positive (exposure-scaled models divide
  by it), so no NaN handling is needed downstream.

Alignment invariants a reviewer should check consumers rely on:

1. One company per call. Any non-LOB segment column carrying >1 distinct
   value is a hard error - these models fit one company at a time.
2. Every line shares the SAME origin axis and the SAME observed-cell mask.
   Cross-line dependence is estimated cell-by-cell (contemporaneous residual
   correlation), so a misaligned pair of triangles would silently pair up
   residuals from different (origin, dev) cells. That is an error here, not
   something to quietly intersect away.
3. Cumulative, not incremental: ``triangle.meta.measure`` must be
   "cumulative". Slice the training window with ``triangle.as_of(...)``
   BEFORE calling - this module consumes every row it is handed.

Predictive target layout (fixed, and shared by every multiline entry):

    [ (lob_0, origin_0) ... (lob_0, origin_{n_w-1}),      lob-major
      (lob_1, origin_0) ... ,                             n_lob*n_w cells
      lob_0 total ... lob_{n_lob-1} total,                n_lob rows
      grand total ]                                       1 row

Totals are always derived from the same draws as the cells they sum
(``flatten_with_totals``), so cross-line diversification is visible in the
samples rather than being an independent-sum approximation bolted on later.
``multiline_targets`` produces the matching metadata frame; the two must be
kept in lockstep.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from ibnr.kernels.contract import _as_date
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import GRAIN_MONTHS, Triangle

#: default segment column that defines the LOB axis (the Schedule P mart's name)
LOB_COLUMN = "line_of_business"


def multiline_data(
    triangle: Triangle,
    *,
    loss_field: str,
    premium_field: str | None = None,
    lob_column: str = LOB_COLUMN,
) -> dict[str, Any]:
    """Map a one-company multi-LOB cumulative Triangle to dense arrays.

    Returns the dict documented in the module docstring - the single data
    contract for every multiline entry. Slice training data with
    ``triangle.as_of(...)`` before calling: this function uses every row it
    sees, so an unsliced triangle silently trains on the future.

    Every check below is deliberately a hard error rather than a repair.
    These models estimate cross-line dependence cell by cell, and any silent
    fix-up (intersecting origins, dropping a line, densifying a hole) would
    change which residuals get paired without the caller noticing.
    """
    # incremental input would make `cum` a lie for every downstream consumer
    if triangle.meta.measure != "cumulative":
        raise ValueError("multiline_data requires a cumulative triangle")
    df = triangle.select_fields(loss_field).execute()
    if df.empty:
        raise ValueError(f"no rows for loss field {loss_field!r}")
    segs = triangle.segments
    if lob_column not in segs:
        raise ValueError(f"triangle has no {lob_column!r} segment column; segments: {segs}")
    # every OTHER segment column (company, state, ...) must be constant:
    # a multiline fit is one company's set of lines, nothing wider
    for col in (c for c in segs if c != lob_column):
        values = df[col].unique()
        if len(values) > 1:
            raise ValueError(
                f"multiple {col!r} values {sorted(map(str, values))}; "
                "multiline models fit one company at a time - filter first"
            )

    df = df.copy()
    df["origin_period"] = _as_date(df["origin_period"])
    # dev_lag is always MONTHS from origin start (CLAUDE.md milestone 1); the
    # grain converts it to the 1-based dev step d used by every contract here
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    if (df["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")
    df["d"] = (df["dev_lag"] // step).astype(int)
    if (df["d"] < 1).any():
        raise ValueError("dev_lag must be positive")
    # >1 row per cell means several evaluation dates survived: the caller
    # forgot as_of()/latest_diagonal() and the grid write below would keep
    # whichever row happened to come last
    if df.duplicated([lob_column, "origin_period", "d"]).any():
        raise ValueError(
            "multiple rows per (lob, origin, dev) cell; slice with as_of()/latest_diagonal() first"
        )

    # sorted label order IS the k index (and the sample-column order); origins
    # are the shared w axis across all lines
    lobs = sorted(df[lob_column].unique())
    if len(lobs) < 2:
        raise ValueError(f"multiline models need >= 2 lines of business, got {lobs}")
    origins = sorted(df["origin_period"].unique())
    # n_d = deepest observed dev step, so the grid is rectangular per line
    n_lob, n_w, n_d = len(lobs), len(origins), int(df["d"].max())

    k_of = {lob: k for k, lob in enumerate(lobs)}
    w_of = {o: w for w, o in enumerate(origins)}
    # scatter the long rows into the dense grid; NaN prefill = "unobserved",
    # which is what the mask below reads back out
    cum = np.full((n_lob, n_w, n_d), np.nan)
    for row in df.itertuples():
        lob = getattr(row, lob_column)
        cum[k_of[lob], w_of[row.origin_period], int(row.d) - 1] = float(row.value)
    obs_mask = ~np.isnan(cum)

    # ALIGNMENT GUARANTEE: line 0's mask is the reference. Contemporaneous
    # cross-line correlation is estimated by pairing residuals at the SAME
    # (origin, dev) cell, so a line observed on a different set of cells would
    # corrupt the pairing rather than merely lose data.
    misaligned = [lobs[k] for k in range(1, n_lob) if not (obs_mask[k] == obs_mask[0]).all()]
    if misaligned:
        raise ValueError(
            f"lines {misaligned} have a different observed-cell pattern than {lobs[0]!r}; "
            "cell-wise dependence models need aligned triangles"
        )

    # see the module docstring for the per-key contract
    data: dict[str, Any] = {
        "n_lob": n_lob,
        "n_w": n_w,
        "n_d": n_d,
        "lobs": lobs,
        "cum": cum,
        "obs_mask": obs_mask,
        "origin_periods": origins,
        "dev_grain_months": step,
        "lob_column": lob_column,
        "units": triangle.meta.units,
    }
    if premium_field is not None:
        data["premium"] = _premium_by_lob_origin(triangle, premium_field, lobs, origins, lob_column)
    return data


def _premium_by_lob_origin(
    triangle: Triangle,
    premium_field: str,
    lobs: list,
    origins: list,
    lob_column: str,
) -> np.ndarray:
    """Booked earned premium as a dense (n_lob, n_w) array, aligned to the
    ``lobs``/``origins`` axes of ``cum``.

    Premium is an origin-level exposure measure, not a triangle cell, so the
    LATEST evaluation of each (lob, origin) is the booked value - the same
    convention as ``contract._premium_by_origin``. Completeness and strict
    positivity are contract guarantees (models divide losses by premium to
    get loss ratios), so gaps raise instead of becoming NaN downstream.
    """
    pdf = triangle.select_fields(premium_field).latest_diagonal().execute()
    if pdf.empty:
        raise ValueError(f"no rows for premium field {premium_field!r}")
    pdf = pdf.copy()
    pdf["origin_period"] = _as_date(pdf["origin_period"])
    by_key = pdf.set_index([lob_column, "origin_period"])["value"]
    premium = np.full((len(lobs), len(origins)), np.nan)
    for k, lob in enumerate(lobs):
        for w, origin in enumerate(origins):
            if (lob, origin) in by_key.index:
                premium[k, w] = float(by_key.loc[(lob, origin)])
    # a hole here is usually a screening bug upstream (a line with losses but
    # no premium row), so name the offending pairs rather than imputing
    if np.isnan(premium).any():
        missing = [
            (lobs[k], origins[w].isoformat())
            for k, w in zip(*np.nonzero(np.isnan(premium)), strict=True)
        ]
        raise ValueError(f"premium missing for (lob, origin) pairs {missing}")
    if (premium <= 0).any():
        raise ValueError("non-positive premium; exposure-scaled models need positive premium")
    return premium


def realized_multiline(
    full_triangle: Triangle,
    *,
    loss_field: str,
    dev_lag: int,
    lobs: list,
    origins: list[dt.date],
    lob_column: str = LOB_COLUMN,
) -> np.ndarray:
    """Realized cumulative losses at ``dev_lag`` months, (n_lob, n_w), from the
    FULL (unsliced) triangle. NaN where the outcome is not (yet) observed.

    The backtest scoring targets: the model is fit on an ``as_of`` slice and
    then judged against what actually developed by ``dev_lag`` (the Meyers
    retrospective design). Pass ``lobs``/``origins`` straight from the fitted
    contract dict - the axes must match the prediction's, and origins present
    in the full triangle but absent from the training slice are deliberately
    NOT added here (see CLAUDE.md's post-study accident-year gotcha).
    The multiline analogue of ``contract.realized_values``; feed the result to
    ``flatten_with_totals`` to line it up with the predictive samples.
    """
    df = full_triangle.select_fields(loss_field).execute()
    df = df[df["dev_lag"] == dev_lag].copy()
    df["origin_period"] = _as_date(df["origin_period"])
    by_key = df.set_index([lob_column, "origin_period"])["value"]
    out = np.full((len(lobs), len(origins)), np.nan)
    for k, lob in enumerate(lobs):
        for w, origin in enumerate(origins):
            if (lob, origin) in by_key.index:
                out[k, w] = float(by_key.loc[(lob, origin)])
    return out


def multiline_targets(
    lobs: list,
    origins: list[dt.date],
    premium: np.ndarray | None = None,
    lob_column: str = LOB_COLUMN,
) -> pd.DataFrame:
    """Target metadata matching ``flatten_with_totals`` column order:
    per-(lob, origin) rows lob-major, then per-lob totals, then grand total.

    One row per predictive sample column, in the same order - this frame is
    what makes a ``PredictiveDistribution`` self-describing for scoring. Total
    rows carry ``origin_period=None`` (and ``line_of_business=None`` on the
    grand total) and NaN premium: premium is only defined per (lob, origin),
    and summing it would invite premium-normalized metrics on aggregate rows
    that mix exposure bases.
    """
    origin_labels = _origin_labels(origins)
    rows = []
    for k, lob in enumerate(lobs):
        for w, origin in enumerate(origins):
            rows.append(
                {
                    "label": f"{lob}/{origin_labels[w]}",
                    lob_column: lob,
                    "origin_period": origin,
                    "premium": float(premium[k, w]) if premium is not None else np.nan,
                }
            )
    for lob in lobs:
        rows.append(
            {"label": f"{lob}/total", lob_column: lob, "origin_period": None, "premium": np.nan}
        )
    rows.append({"label": "total", lob_column: None, "origin_period": None, "premium": np.nan})
    return pd.DataFrame(rows)


def flatten_with_totals(arr: np.ndarray) -> np.ndarray:
    """(..., n_lob, n_w) -> (..., n_lob*n_w + n_lob + 1): per-cell values
    lob-major, per-lob sums, grand sum. Works for realized values (2-D) and
    sample stacks (3-D) alike; NaN outcomes propagate into their sums.

    Applying the identical layout to draws AND to outcomes is what keeps
    predictions and realizations index-comparable. Summing WITHIN each draw
    (rather than summing marginal quantiles) is the actuarial point: the total
    reserve's distribution then reflects the model's own cross-line dependence,
    so diversification shows up as a narrower total than the sum of the
    per-line intervals. NaN propagation is intended - an unobserved outcome
    must poison its totals rather than be silently treated as zero.
    """
    # last two axes are (lob, origin); leading axes (draws, ...) pass through
    per_cell = arr.reshape(*arr.shape[:-2], -1)  # (..., n_lob*n_w) lob-major
    lob_totals = arr.sum(axis=-1)  # (..., n_lob) sum over origins
    grand = lob_totals.sum(axis=-1, keepdims=True)  # (..., 1)
    return np.concatenate([per_cell, lob_totals, grand], axis=-1)


def assemble_predictive(
    ults: np.ndarray,
    targets: pd.DataFrame,
    units: str | None = None,
) -> PredictiveDistribution:
    """Build the PredictiveDistribution from per-(lob, origin) ultimate draws.

    ults: (n_draws, n_lob, n_w) - ULTIMATE losses, not reserves (reserve =
    ultimate - paid to date; scoring is done on ultimates so every entry is
    comparable). Totals are derived here rather than accepted from the caller
    so they are always row-sums of the same draws, which is the only way
    cross-line diversification stays coherent.

    The width check is the contract handshake between the samples and the
    ``multiline_targets`` frame: a mismatch means the caller built one of the
    two from a different lobs/origins axis.
    """
    if ults.ndim != 3:
        raise ValueError(f"ults must be (n_draws, n_lob, n_w), got {ults.shape}")
    samples = flatten_with_totals(ults)
    if samples.shape[1] != len(targets):
        raise ValueError(
            f"{samples.shape[1]} sample columns but {len(targets)} target rows; "
            "build targets with multiline_targets(lobs, origins)"
        )
    return PredictiveDistribution(samples=samples, targets=targets, units=units)


def _origin_labels(origins: list[dt.date]) -> list[str]:
    """Short human labels for target rows: bare accident year when the origins
    are annual (the Schedule P case), full ISO dates otherwise."""
    years = [o.year for o in origins]
    if len(set(years)) == len(years):
        return [str(y) for y in years]
    return [o.isoformat() for o in origins]
