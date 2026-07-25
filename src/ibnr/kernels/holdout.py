"""The held-out diagonal: which cells a model trained at ``as_of`` is scored on.

Every retrospective in this package so far trains on a diagonal and scores
*realized ultimates* ten years later. That answers "was the reserve right", which
is the actuarial question, but it gives one outcome per cohort and it cannot be
computed until the run-off is complete. Milestone 6 adds the complementary view:
score the cells the model did **not** see on the very next diagonal, which is
available immediately and gives one outcome per cell.

The definition, and it is the whole module::

    D       = as_of                                   training cutoff
    D_next  = min{ e in triangle.eval_dates : e > D }  READ FROM THE DATA
    before  = triangle.as_of(D)
    after   = triangle.as_of(D_next)
    heldout = after ANTI-JOIN before on (segments..., field, origin_period, dev_lag)

Three things in that are load-bearing, and each is a way to get this quietly
wrong:

1. **``D_next`` is read from the data, never computed as ``D + 12 months``.**
   Schedule P is annual, so the arithmetic version passes every test written
   against it and then breaks on a quarterly triangle or a mart with a gap.

2. **The anti-join is on the full cell key, including ``dev_lag``.** A cell that
   was merely *restated* between D and D_next lands in ``after`` under the same
   key (``Triangle.as_of`` already collapses restatements to the latest
   surviving one), so it is joined away and is not scored. It is not new
   information about an unobserved cell; it is a correction to an observed one,
   and scoring it would credit the model for predicting something it was
   trained on.

3. **Exposure is attached from ``before``.** Premium is usually restated
   alongside losses; reading it from ``after`` would leak a value the model
   could not have had.

Cells on the next diagonal that the model structurally cannot score are
**excluded and counted, never dropped silently** - they land in ``excluded``
with a ``reason``. There are three:

``new_origin``
    The origin period appears for the first time at ``D_next``. Four of the five
    Bayesian entries carry per-origin parameters (``alpha[w]``, ``RLR[w]``,
    ``RRF[w]``) that simply have no draw at index ``n_w + 1``.
``dev_beyond_trained``
    The development lag is deeper than anything in training, so there is no
    ``beta[d]`` / ``sig[d]`` either.
``no_predecessor``
    The cell's immediate predecessor is missing from training, so no increment
    can be formed for it. On a run-off staircase this never fires; on a ragged
    real triangle it does.

On a square 8x8 staircase whose source also carries a 9th origin, the next
diagonal holds 9 cells: 7 scored, plus origin 1 at dev 9 (``dev_beyond_trained``)
and origin 9 at dev 1 (``new_origin``).

Scope note: only the *next* diagonal. Scoring two ahead means marginalizing over
the unobserved diagonal between, which is a nested integral for the Bayesian
entries and an autoregressive rollout for the transformer - a different
computation, not a parameter. There is deliberately no ``horizon`` argument
until that is built.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ibnr.kernels.contract import _as_date as _dates_to_python
from ibnr.triangle.core import GRAIN_MONTHS, Triangle

__all__ = ["HoldoutCells", "next_diagonal"]

#: reasons a next-diagonal cell is not scorable, in the order they are tested
EXCLUSION_REASONS: tuple[str, ...] = ("new_origin", "dev_beyond_trained", "no_predecessor")


@dataclass(frozen=True)
class HoldoutCells:
    """The scorable cells of one held-out diagonal, plus what was excluded.

    ``frame`` carries one row per scorable cell: the segment columns, then
    ``field``, ``origin_period``, ``dev_lag``, ``eval_date``, ``value``,
    ``prev_value`` and (when asked for) ``premium``.

    ``value`` is on the triangle's own measure - ``measure`` records which.
    ``prev_value`` is the same cell key one development step earlier, taken from
    the **training** slice, which is what lets a consumer form the increment
    ``value - prev_value`` without leaking: on a cumulative triangle the
    predecessor sits on the training diagonal and is data, not a prediction.
    That is also why the increment/cumulative change of variable has Jacobian 1.
    """

    frame: pd.DataFrame
    as_of: dt.date
    eval_date: dt.date
    excluded: pd.DataFrame
    train_origins: tuple[dt.date, ...]
    segments: tuple[str, ...]
    measure: str
    premium_field: str | None = None

    @property
    def n_cells(self) -> int:
        return len(self.frame)

    @property
    def values(self) -> np.ndarray:
        """Realized outcomes at the held-out cells, in frame order."""
        return self.frame["value"].to_numpy(dtype=float)

    @property
    def increments(self) -> np.ndarray:
        """``value - prev_value``, in frame order.

        Meaningful on a cumulative triangle; on an incremental one ``value`` is
        already the increment and this is not what you want.
        """
        return self.values - self.frame["prev_value"].to_numpy(dtype=float)

    @property
    def key_columns(self) -> list[str]:
        return [*self.segments, "field", "origin_period", "dev_lag"]

    def key(self) -> pd.MultiIndex:
        """Cell identity, for joining a model's scores back onto the outcomes."""
        return pd.MultiIndex.from_frame(self.frame[self.key_columns])

    def exclusion_counts(self) -> dict[str, int]:
        """How many next-diagonal cells each reason removed. Always reported -
        a model that scores fewer cells must not look better for it."""
        if self.excluded.empty:
            return dict.fromkeys(EXCLUSION_REASONS, 0)
        counts = self.excluded["reason"].value_counts().to_dict()
        return {r: int(counts.get(r, 0)) for r in EXCLUSION_REASONS}

    def __repr__(self) -> str:
        ex = ", ".join(f"{r}={n}" for r, n in self.exclusion_counts().items() if n)
        return (
            f"HoldoutCells({self.n_cells} cells, as_of={self.as_of}, "
            f"eval_date={self.eval_date}"
            f"{', excluded: ' + ex if ex else ''})"
        )


def next_diagonal(
    triangle: Triangle,
    *,
    as_of: dt.date | str,
    fields: str | Sequence[str],
    premium_field: str | None = None,
    origins: Sequence[dt.date] | None = None,
) -> HoldoutCells:
    """Cells that first become observable on the diagonal after ``as_of``.

    triangle:      the FULL triangle, not a training slice - this function does
                   its own ``as_of`` slicing on both sides of the cutoff.
    as_of:         the training cutoff D.
    fields:        loss field(s) to score. ``compartmental`` needs two (paid and
                   reported); everything else needs one.
    premium_field: exposure field to attach per cell, read from the training
                   slice. Not scored, never part of ``fields``.
    origins:       restrict to these origin periods (the Meyers study window is
                   1988-1997 while the mart runs to 2007, and an aggregate over
                   origins the training slice never had is silently wrong).
    """
    cutoff = _as_date(as_of)
    wanted = [fields] if isinstance(fields, str) else list(fields)
    if not wanted:
        raise ValueError("fields must name at least one field to score")
    if premium_field is not None and premium_field in wanted:
        raise ValueError(
            f"premium_field={premium_field!r} is also in fields; exposure is attached "
            "to cells, not scored as an outcome"
        )
    missing = sorted(set(wanted) - set(triangle.fields))
    if missing:
        raise ValueError(f"triangle has no field(s) {missing}; has {sorted(triangle.fields)}")

    later = sorted(e for e in triangle.eval_dates if e > cutoff)
    if not later:
        raise ValueError(
            f"no eval_date after {cutoff}: the triangle ends at {max(triangle.eval_dates)}, "
            "so there is no next diagonal to hold out"
        )
    d_next = later[0]  # READ from the data; never cutoff + a grain step

    segments = tuple(triangle.segments)
    keys = [*segments, "field", "origin_period", "dev_lag"]
    step = GRAIN_MONTHS[triangle.meta.dev_grain]

    # Both slices go through Triangle.as_of, so restatement collapsing is the
    # single implementation in transforms.py rather than a second one here.
    before = _normalize_dates(triangle.as_of(cutoff).execute())
    after = _normalize_dates(triangle.as_of(d_next).execute())
    if origins is not None:
        keep = {_as_date(o) for o in origins}
        before = before[before["origin_period"].isin(keep)]
        after = after[after["origin_period"].isin(keep)]
        if before.empty:
            raise ValueError("origins selected no training observations")

    train = before[before["field"].isin(wanted)]
    scored = after[after["field"].isin(wanted)]

    # anti-join on the FULL key: a restated training cell reappears here under
    # the same key and is removed, because it is a correction to an observed
    # cell rather than news about an unobserved one.
    fresh = scored.merge(train[keys], on=keys, how="left", indicator=True)
    fresh = fresh[fresh["_merge"] == "left_only"].drop(columns="_merge")
    fresh = fresh.sort_values(keys, kind="stable").reset_index(drop=True)

    train_origins = sorted(train["origin_period"].unique())
    # Per (segment, field): how deep did training go? Beyond that there is no
    # per-dev parameter to evaluate.
    max_dev = train.groupby([*segments, "field"], dropna=False)["dev_lag"].max()

    fresh["prev_value"] = _predecessor(fresh, train, keys, step)
    fresh["reason"] = _exclusion_reason(fresh, set(train_origins), max_dev, segments)

    if premium_field is not None:
        fresh["premium"] = _premium(fresh, before, segments, premium_field)

    keep_cols = [
        *segments,
        "field",
        "origin_period",
        "dev_lag",
        "eval_date",
        "value",
        "prev_value",
        *(["premium"] if premium_field is not None else []),
    ]
    ok = fresh["reason"].isna()
    return HoldoutCells(
        frame=fresh.loc[ok, keep_cols].reset_index(drop=True),
        as_of=cutoff,
        eval_date=d_next,
        excluded=fresh.loc[~ok, [*keep_cols, "reason"]].reset_index(drop=True),
        train_origins=tuple(train_origins),
        segments=segments,
        measure=triangle.meta.measure,
        premium_field=premium_field,
    )


def _as_date(value: dt.date | str) -> dt.date:
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.datetime):
        return value.date()
    return value


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Materialized date columns to ``dt.date``, matching ``contract.py``.

    ``Triangle.origins`` / ``.eval_dates`` hand back ``dt.date`` while
    ``.execute()`` hands back ``datetime64``, so without this the frame's
    ``eval_date`` column and ``HoldoutCells.eval_date`` are different types for
    the same concept - which reads fine until something joins on it.
    """
    out = df.copy()
    for column in ("origin_period", "eval_date"):
        if column in out.columns:
            out[column] = _dates_to_python(out[column])
    return out


def _predecessor(fresh: pd.DataFrame, train: pd.DataFrame, keys: list[str], step: int):
    """Value one development step back, from TRAINING data only."""
    prev = train[[*keys, "value"]].rename(columns={"value": "prev_value"})
    lookup = fresh[keys].copy()
    lookup["dev_lag"] = lookup["dev_lag"] - step
    merged = lookup.merge(prev, on=keys, how="left")
    # A cell at the first development lag has no predecessor by construction and
    # its cumulative value IS its increment, so 0 is correct rather than missing.
    return merged["prev_value"].where(lookup["dev_lag"] > 0, 0.0).to_numpy()


def _exclusion_reason(
    fresh: pd.DataFrame,
    train_origins: set,
    max_dev: pd.Series,
    segments: tuple[str, ...],
) -> pd.Series:
    """First applicable reason per row, or NaN when the cell is scorable."""
    reason = pd.Series(pd.NA, index=fresh.index, dtype="object")

    new_origin = ~fresh["origin_period"].isin(train_origins)
    reason = reason.mask(reason.isna() & new_origin, "new_origin")

    group_key = [*segments, "field"]
    trained_depth = fresh[group_key].merge(
        max_dev.rename("max_dev").reset_index(), on=group_key, how="left"
    )["max_dev"]
    trained_depth.index = fresh.index
    too_deep = trained_depth.isna() | (fresh["dev_lag"] > trained_depth)
    reason = reason.mask(reason.isna() & too_deep, "dev_beyond_trained")

    no_prev = fresh["prev_value"].isna()
    reason = reason.mask(reason.isna() & no_prev, "no_predecessor")
    return reason


def _premium(
    fresh: pd.DataFrame, before: pd.DataFrame, segments: tuple[str, ...], premium_field: str
):
    """Exposure per origin, from the TRAINING slice.

    Read from ``before`` on purpose: premium is restated alongside losses, and
    taking it from the post-cutoff slice would hand the model an exposure it
    could not have known.
    """
    prem = before[before["field"] == premium_field]
    if prem.empty:
        raise ValueError(
            f"premium_field={premium_field!r} has no observations in the training slice"
        )
    group_key = [*segments, "origin_period"]
    latest = prem.sort_values("dev_lag").groupby(group_key, dropna=False, as_index=False).last()
    merged = fresh[group_key].merge(
        latest[[*group_key, "value"]].rename(columns={"value": "premium"}),
        on=group_key,
        how="left",
    )
    merged.index = fresh.index
    return merged["premium"]
