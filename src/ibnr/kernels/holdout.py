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

__all__ = ["CellIndex", "HoldoutCells", "index_into", "next_diagonal", "training_index"]

#: reasons a next-diagonal cell is not scorable, in the order they are tested
EXCLUSION_REASONS: tuple[str, ...] = ("new_origin", "dev_beyond_trained", "no_predecessor")


@dataclass(frozen=True)
class CellIndex:
    """Cells expressed in a fitted model's own index space.

    A scorer needs ``(w, d)`` positions into the contract's ``alpha``/``beta``/
    ``sig`` vectors, not dates. Both indices are **1-based**, matching the Stan
    data block so the numpy scorers read like the ``.stan`` file beside them -
    the conversion to 0-based happens once, inside each scorer.

    The same type carries training cells and held-out cells, which is what makes
    the agreement gate possible: score the training cells with the held-out code
    path and the answer must reproduce the fit's own ``log_lik``.
    """

    w: np.ndarray  # 1-based origin index
    d: np.ndarray  # 1-based development index
    value: np.ndarray  # observed loss amount at the cell
    prev_value: np.ndarray  # same cell one dev step back, from TRAINING data
    premium: np.ndarray  # exposure at the cell's origin, from TRAINING data

    def __post_init__(self) -> None:
        n = len(self.w)
        for name in ("d", "value", "prev_value", "premium"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} entries, expected {n}")
        if n and (self.w.min() < 1 or self.d.min() < 1):
            raise ValueError("w and d are 1-based indices; got a value below 1")

    @property
    def n_cells(self) -> int:
        return len(self.w)


def training_index(contract: dict) -> CellIndex:
    """The cells a fit was trained on, in the same shape a scorer takes.

    Exists for the agreement gate: a scorer handed these must reproduce the
    fit's own ``log_lik`` elementwise. Without that, a scorer can be wrong in
    its index arithmetic and produce entirely plausible held-out numbers.

    .. warning::

       ``prev_value`` here is the previous **development** cell ``(w, d-1)``,
       matching :class:`HoldoutCells`. It is emphatically **not** the contract's
       ``prev_idx``, which points at the previous **origin** ``(w-1, d)`` -
       that is CCL's accident-year AR(1) link (``contract.py``, ``row_of.get((w
       - 1, d))``), a different quantity entirely. An earlier version of this
       function reused ``prev_idx`` and was wrong; ``meyers_csr`` does not read
       ``prev_value``, so its tests could not see it, and the first scorer that
       differenced cumulatives would have silently subtracted the wrong cell.
    """
    w = np.asarray(contract["w"], dtype=int)
    d = np.asarray(contract["d"], dtype=int)
    loss = np.asarray(contract["loss"], dtype=float)
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(w, d, strict=True))}

    def predecessor(origin: int, dev: int) -> float:
        back = row_of.get((origin, dev - 1))
        if back is not None:
            return float(loss[back])
        # a cell at the first dev has no predecessor and its cumulative value IS
        # its increment; a hole anywhere else is genuinely unknown
        return 0.0 if dev == 1 else float("nan")

    prev = np.array([predecessor(int(a), int(b)) for a, b in zip(w, d, strict=True)], dtype=float)
    premium = _contract_premium(contract, w)
    return CellIndex(w=w, d=d, value=loss, prev_value=prev, premium=premium)


def index_into(cells: HoldoutCells, contract: dict, *, field: str | None = None) -> CellIndex:
    """Map held-out cells onto a fitted contract's ``(w, d)`` index space.

    Origins come from ``contract["origin_periods"]`` and dev indices from the
    contract's own grain, so a cell can only be scored at the position the model
    actually fitted. An origin or dev the contract does not have is an error
    here rather than a silent misindex - :func:`next_diagonal` already excludes
    those cases, so reaching one means the cells and the fit disagree about
    which cohort they describe.

    **Identity is checked against the CONTRACT, not merely for internal
    consistency.** ``(w, d)`` alone does not identify a cell: two companies, or
    two lines of business, share origin dates and development lags exactly. It
    is not enough to require that the cells agree with each other - a single,
    entirely wrong cohort agrees with itself perfectly, indexes cleanly, and
    yields a complete, plausible ELPD for a company the fit never saw. So the
    cells' cohort and field must equal the ones the contract was built from,
    which is why ``kernels.contract`` records ``segment`` and ``fields``.
    """
    if "segment" not in contract or "fields" not in contract:
        raise ValueError(
            "contract carries no cohort identity ('segment'/'fields'), so held-out cells "
            "cannot be shown to belong to it. Rebuild it with stan_data / odp_stan_data / "
            "compartmental_stan_data"
        )

    frame = cells.frame
    fitted_fields = tuple(contract["fields"])
    if field is None and len(fitted_fields) == 1:
        field = fitted_fields[0]  # the fit knows its own field; do not make the caller repeat it
    if field is not None:
        if field not in fitted_fields:
            raise ValueError(
                f"field={field!r} is not what this fit was trained on ({list(fitted_fields)}); "
                "scoring it would evaluate one quantity's likelihood against another's data"
            )
        frame = frame[frame["field"] == field]
    if frame.empty:
        raise ValueError(f"no held-out cells for field={field!r}")

    present = sorted(frame["field"].unique())
    stray = [f for f in present if f not in fitted_fields]
    if stray:
        raise ValueError(f"held-out cells carry field(s) {stray}, not in {list(fitted_fields)}")
    if len(present) > 1:
        raise ValueError(
            f"held-out cells span fields {present}; pass field= to choose one of "
            f"{list(fitted_fields)}"
        )

    fitted_segment = dict(contract["segment"])
    if fitted_segment:
        missing = [s for s in fitted_segment if s not in frame.columns]
        if missing:
            raise ValueError(
                f"held-out cells have no {missing} column(s), so they cannot be shown to "
                f"belong to the fitted cohort {fitted_segment}"
            )
        actual = frame[list(fitted_segment)].drop_duplicates()
        matches = len(actual) == 1 and all(
            actual.iloc[0][k] == v for k, v in fitted_segment.items()
        )
        if not matches:
            raise ValueError(
                f"held-out cells belong to cohort(s) {actual.to_dict('records')}, but this fit "
                f"was trained on {fitted_segment}. Origin dates and dev lags are shared across "
                "cohorts, so these would index cleanly and score the wrong one"
            )
    elif cells.segments:
        # the fit has no segment columns but the cells do - the same mistake one
        # level up, and there is nothing on the contract to check against
        combos = frame[list(cells.segments)].drop_duplicates()
        if len(combos) > 1:
            raise ValueError(
                f"held-out cells span {len(combos)} cohorts on {list(cells.segments)} but the "
                "contract carries no segment identity to check them against"
            )

    origins = list(contract["origin_periods"])
    w_of = {o: i + 1 for i, o in enumerate(origins)}
    step = int(contract["dev_grain_months"])

    unknown = sorted(set(frame["origin_period"]) - set(w_of))
    if unknown:
        raise ValueError(
            f"origin(s) {unknown} are not in the fitted contract; the fit covers "
            f"{origins[0]}..{origins[-1]}"
        )
    if (frame["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the fit's {step}-month grain")

    w = frame["origin_period"].map(w_of).to_numpy(dtype=int)
    d = (frame["dev_lag"] // step).to_numpy(dtype=int)
    if d.max() > int(contract["n_d"]):
        raise ValueError(
            f"dev index {d.max()} exceeds the fit's n_d={contract['n_d']}; "
            "next_diagonal() should have excluded this as dev_beyond_trained"
        )
    premium = (
        frame["premium"].to_numpy(dtype=float)
        if "premium" in frame.columns
        else _contract_premium(contract, w)
    )
    return CellIndex(
        w=w,
        d=d,
        value=frame["value"].to_numpy(dtype=float),
        prev_value=frame["prev_value"].to_numpy(dtype=float),
        premium=premium,
    )


def _contract_premium(contract: dict, w: np.ndarray) -> np.ndarray:
    """Exposure per cell from the fitted contract, or NaN when it carries none.

    NaN rather than 1.0: an entry whose measure needs premium must fail loudly,
    and a silent 1.0 would leave a loss-ratio density unconverted while still
    returning plausible numbers.
    """
    if "premium" not in contract:
        return np.full(len(w), np.nan)
    return np.asarray(contract["premium"], dtype=float)[w - 1]


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
        """``value - prev_value``, in frame order. Cumulative triangles only.

        On an incremental triangle ``value`` is already the increment, so
        differencing it again is simply wrong - and it would be wrong quietly,
        producing a smaller number of the right sign on most cells. Hence the
        error rather than a docstring caveat.
        """
        if self.measure != "cumulative":
            raise ValueError(
                f"increments is for cumulative triangles; this one is {self.measure!r}, "
                "where `values` already holds the increments"
            )
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

    segments = tuple(triangle.segments)
    keys = [*segments, "field", "origin_period", "dev_lag"]
    step = GRAIN_MONTHS[triangle.meta.dev_grain]

    # D_next is read from the eval dates of the LOSS ROWS BEING SCORED, after
    # the origins restriction - not from the triangle's dates as a whole.
    # Premium is typically restated on its own schedule, and an exposure-only
    # update carries an eval_date with no loss cells behind it: taking the
    # triangle-wide minimum would select that date, hold out an empty diagonal,
    # and skip the real one entirely. Same for a date that only exists outside
    # the requested origin window.
    full = _normalize_dates(triangle.execute())
    scope = full[full["field"].isin(wanted)]
    if origins is not None:
        keep = {_as_date(o) for o in origins}
        scope = scope[scope["origin_period"].isin(keep)]
    if scope.empty:
        raise ValueError(f"no observations of {wanted} in the requested origins")

    later = sorted(e for e in scope["eval_date"].unique() if e > cutoff)
    if not later:
        raise ValueError(
            f"no eval_date after {cutoff} carrying any of {wanted}: they end at "
            f"{max(scope['eval_date'])}, so there is no next diagonal to hold out"
        )
    d_next = later[0]  # READ from the data; never cutoff + a grain step

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
    # Per segment: which (cohort, origin) pairs did training actually contain? A
    # company that writes a line from 1995 has no alpha[w] for 1994 even though
    # another company in the same triangle does, so a triangle-wide origin set
    # would wave that cell through to be scored against a parameter the fit
    # never estimated. A merge rather than a groupby: it is vectorized and it
    # degenerates correctly when the triangle has no segment columns at all.
    trained_pairs = train[[*segments, "origin_period"]].drop_duplicates()

    cumulative = triangle.meta.measure == "cumulative"
    fresh["prev_value"] = _predecessor(fresh, train, keys, step) if cumulative else np.nan
    fresh["reason"] = _exclusion_reason(
        fresh, trained_pairs, max_dev, segments, check_predecessor=cumulative
    )

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
    trained_pairs: pd.DataFrame,
    max_dev: pd.Series,
    segments: tuple[str, ...],
    *,
    check_predecessor: bool,
) -> pd.Series:
    """First applicable reason per row, or NaN when the cell is scorable."""
    reason = pd.Series(pd.NA, index=fresh.index, dtype="object")

    # PER (SEGMENT, ORIGIN). A triangle-wide origin set would treat a cohort's
    # very first accident year as known merely because some other cohort in the
    # same triangle wrote that year, and the cell would then be scored against
    # an alpha[w] the fit never estimated.
    origin_key = [*segments, "origin_period"]
    marked = trained_pairs.assign(_trained=True)
    seen = fresh[origin_key].merge(marked, on=origin_key, how="left")["_trained"].to_numpy()
    seen = np.where(pd.isna(seen), False, seen).astype(bool)
    reason = reason.mask(reason.isna() & ~pd.Series(seen, index=fresh.index), "new_origin")

    group_key = [*segments, "field"]
    trained_depth = fresh[group_key].merge(
        max_dev.rename("max_dev").reset_index(), on=group_key, how="left"
    )["max_dev"]
    trained_depth.index = fresh.index
    too_deep = trained_depth.isna() | (fresh["dev_lag"] > trained_depth)
    reason = reason.mask(reason.isna() & too_deep, "dev_beyond_trained")

    # Only cumulative triangles difference against a predecessor; on an
    # incremental one `value` is already the increment and there is nothing to
    # look back at, so the reason would fire on every single cell.
    if check_predecessor:
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
