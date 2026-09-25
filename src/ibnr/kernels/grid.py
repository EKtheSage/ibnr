"""The one-cohort grid of the chain-ladder fits, built and checked with numpy alone.

These helpers used to live in ``kernels/contract.py``, which imports pandas and
the Triangle layer (and so ibis). ``ibnr.methods`` needs only the grid, so they
live here, where nothing but numpy and the standard library is imported, and
``from ibnr import methods`` does not load ibis, pandas or scipy.
``kernels.contract`` re-exports every public name that moved, so code that
imports them from there keeps working.

- :func:`grid_from_columns` builds the grid dict from three columns.
- :func:`check_grid` checks a grid a caller hands in.
- :func:`require_run_off`, :func:`dev_step_index`, :func:`as_date` and
  :func:`month_end` are the pieces both use.
- ``ZERO_CELLS`` and ``TRIANGLE_MEASURES`` list the values two settings take.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from ibnr.errors import Refusal, RefusedCell, _literal

#: A triangle's measure. The same two values as ``ibnr.triangle.core.Measure``,
#: which this module cannot import without loading ibis;
#: ``tests/test_import_purity.py`` checks that the two agree.
TRIANGLE_MEASURES = ("cumulative", "incremental")


def dev_step_index(dev_lag, *, step: int) -> np.ndarray:
    """The 1-based dev step index ``d = dev_lag // step``, or a refusal by name.

    Every contract in this package - the three Stan ones, the dense cohort grid,
    the multi-LOB grid and the neural grids - stores a cell at dev index
    ``dev_lag // step`` and reads dev step 1 as the first development period.
    Two things break that division and both used to be checked separately in six
    places, in six copies of two lines:

    * an age that is not a whole number of dev steps. Floor division does not
      refuse it, it moves it: on a 12-month grain, ages 3, 15, 27 land on steps 0,
      1, 2 rather than 1, 2, 3, so every origin reads one development period
      younger than it is and the first cell falls off the grid entirely. Those
      exact ages are what chainladder's latest-diagonal anchoring produces, and
      what our own ``with_dev_grain`` produces to match it, whenever the latest
      valuation is a March 31 - so this is a shape the triangle layer emits, not
      one only bad input can reach. It is a coherent triangle (``validate``
      accepts it: every age shares one offset) and it is not a grid these
      contracts can index, which is why the refusal is here and not there.
    * a non-positive age. ``dev_lag`` counts months from the origin period start,
      so the first cell of a 12-month grain is at 12; a zero or negative age would
      index step 0 or below.

    The sign is tested first, because ``%`` here follows Python's sign rule: an
    age of -3 leaves a remainder of 9 against a 12-month grain, so testing the
    offset first would report a negative age as an anchoring problem and the
    positivity message could never be reached for it.

    Returns an ``int64`` array aligned with the input, so a caller can assign it
    straight into its frame. Both refusals are ``ibnr.errors.Refusal``s, with
    the reasons ``invalid_age`` and ``grain_mismatch``.
    """
    months = np.asarray(dev_lag, dtype=np.int64)
    non_positive = months <= 0
    if non_positive.any():
        ages = sorted({int(a) for a in np.unique(months[non_positive])})
        raise Refusal(
            "invalid_age",
            f"dev_lag must be positive, got {ages[:5]}; dev_lag is months from the origin "
            f"period start, so the first cell of a {step}-month dev grain is at {step}",
            column="dev_lag",
            count=int(non_positive.sum()),
        )
    offsets = months % step
    off_grain = offsets != 0
    if off_grain.any():
        ages = sorted({int(a) for a in np.unique(months[off_grain])})
        found = sorted({int(o) for o in np.unique(offsets)})
        raise Refusal(
            "grain_mismatch",
            f"{int(off_grain.sum())} dev_lag value(s) are not on a {step}-month grain "
            f"boundary: ages {ages[:5]} leave offsets {found} against the declared dev "
            f"grain. Every contract indexes a cell by d = dev_lag // {step}, so an age off "
            "the boundary is floored onto the step below it and the whole triangle reads "
            "one development period younger. This is what chainladder's latest-diagonal "
            "anchoring produces: a March 31 valuation regrained to an annual dev grain "
            "gives ages 3, 15, 27 rather than 12, 24, 36. Slice with as_of() to a "
            "valuation on a grain boundary before with_dev_grain(), or keep the finer dev "
            "grain.",
            column="dev_lag",
            count=int(off_grain.sum()),
        )
    return months // step


def grid_from_columns(
    origin_period,
    dev_lag,
    value,
    *,
    dev_grain_months: int,
    measure: str,
    units: str | None = None,
    loss_field: str | None = None,
    segment: dict | None = None,
) -> dict[str, Any]:
    """One cohort's cells, given as three equal-length columns, turned into the grid dict.

    The same grid, with the same checks and messages, as
    ``kernels.contract.cohort_grid_frame``, which is a thin wrapper over this function. The
    columns are plain sequences or numpy arrays and nothing here uses pandas:

    - ``origin_period``: the first day of each cell's origin period, as
      ``datetime.date`` or ``datetime`` values, numpy ``datetime64`` values or
      ISO strings such as ``"2010-01-01"``.
    - ``dev_lag``: months from the start of the origin period, a whole multiple
      of ``dev_grain_months``.
    - ``value``: the amount in the cell, as numbers.

    ``ibnr.methods`` builds its grids here.
    """
    # Vectorized throughout: this runs once per cohort in fit_mack_many's batch
    # loop, so per-row work here would put the loop's cost right back after the
    # engine round-trips were removed.
    if measure not in TRIANGLE_MEASURES:
        raise Refusal(
            "invalid_option",
            f"measure must be one of {TRIANGLE_MEASURES}, got {{given}}",
            option="measure",
            given=measure,
        )
    step = dev_grain_months
    d = dev_step_index(dev_lag, step=step)
    # Factorize the raw values, then read each DISTINCT value as a date: two
    # spellings of one origin ("2010-01-01" and a date) become one row, and the
    # conversion costs one call per origin rather than one per cell.
    codes, uniques = _factorize(origin_period)
    if len(codes) != len(d):
        raise ValueError(
            f"origin_period has {len(codes)} values and dev_lag has {len(d)}; "
            "the columns must have one value per cell"
        )
    as_dates = []
    for unique in uniques:
        try:
            as_dates.append(as_date(unique))
        except (TypeError, ValueError) as exc:
            raise Refusal(
                "unreadable_label",
                f"origin_period values must be dates: {_literal(exc)}",
                column="origin_period",
                given=unique,
            ) from exc
    origins = sorted(set(as_dates))
    position = {origin: i for i, origin in enumerate(origins)}
    w_idx = np.array([position[o] for o in as_dates], dtype=np.int64)[codes]
    n_w, n_d = len(origins), int(d.max())
    flat = w_idx.astype(np.int64) * n_d + (d - 1)
    _, first, counts = np.unique(flat, return_index=True, return_counts=True)
    if (counts > 1).any():
        repeated = np.isin(flat, flat[first[counts > 1]])
        raise Refusal(
            "duplicate",
            "multiple rows per (origin, dev) cell; slice with as_of()/latest_diagonal() first. "
            "Repeated: {cells}",
            column="origin_period",
            cells=[
                RefusedCell(None, origins[w_idx[k]], int(d[k]) * step, None)
                for k in np.flatnonzero(repeated)
            ],
        )
    amounts = np.asarray(value, dtype=float)
    if amounts.shape != d.shape:
        raise ValueError(
            f"value has {amounts.size} values and dev_lag has {d.size}; "
            "the columns must have one value per cell"
        )
    cum = np.full((n_w, n_d), np.nan)
    cum[w_idx, d - 1] = amounts
    obs_mask = ~np.isnan(cum)
    latest_dev = require_run_off(obs_mask, origins, step=step, cum=cum)

    # 1-based cell indices of the observed cells, row-major (sorted by (w, d)),
    # the same convention as stan_data's w/d. Derived from the mask rather than
    # the input rows so they cannot disagree with the grid they index.
    w_obs, d_obs = np.nonzero(obs_mask)
    scored = (loss_field,) if loss_field is not None else ()
    return {
        "n_w": n_w,
        "n_d": n_d,
        "cum": cum,
        "obs_mask": obs_mask,
        "latest_dev": latest_dev.astype(int),
        "origin_periods": origins,
        "dev_grain_months": step,
        "units": units,
        "loss_field": loss_field,
        "segment": dict(segment or {}),
        "fields": scored,
        "models": scored,
        "measure": measure,
        "w": (w_obs + 1).astype(int),
        "d": (d_obs + 1).astype(int),
    }


_MISSING_ORIGIN = "origin_period has missing values; every row needs its origin period"


def _missing_origin() -> Refusal:
    return Refusal("missing_value", _MISSING_ORIGIN, column="origin_period")


def _factorize(values) -> tuple[np.ndarray, list]:
    """Codes into the distinct values, and the distinct values, or a refusal of a missing one.

    A typed array (numpy datetime64, strings, numbers) is factorized by
    ``np.unique``. Anything else, such as an object array that mixes dates and
    ISO strings, is factorized by hashing, one dictionary lookup per value, and
    never sorted, because values of different types cannot be ordered.

    Either way the distinct values come back in the order they first appear,
    and numbers come back as plain Python numbers, so a refusal of a value that
    is not a date names the same value, spelled the same way, as it always has.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    kind = arr.dtype.kind
    if kind == "M" and np.isnat(arr).any():
        raise _missing_origin()
    if kind == "f" and np.isnan(arr).any():
        raise _missing_origin()
    if kind != "O":
        uniques, first, inverse = np.unique(arr, return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")  # first appearance, not sorted order
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        uniques = uniques[order]
        # datetime64 stays as it is: tolist() would turn nanoseconds into an int
        distinct = list(uniques) if kind == "M" else uniques.tolist()
        return rank[inverse.reshape(-1)].astype(np.int64), distinct
    if any(_is_missing(v) for v in arr):
        raise _missing_origin()
    index: dict = {}
    codes = np.fromiter(
        (index.setdefault(v, len(index)) for v in arr), dtype=np.int64, count=arr.size
    )
    return codes, list(index)


def _is_missing(value) -> bool:
    """None, a NaN or a NaT. A value that refuses to be compared with itself, as
    pandas' missing-value marker does, counts as missing too."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except TypeError:
        return True


_RUN_OFF = (
    "Every origin needs cells from its first development age up to one common "
    "evaluation date, or up to the last development age once it has run off"
)


def require_run_off(
    obs_mask: np.ndarray, origins: list, *, step: int | None = None, cum: np.ndarray | None = None
) -> np.ndarray:
    """Each origin's last observed dev index, or a refusal if the grid is not a run-off.

    The observed cells must form the staircase ``kernels.contract.cohort_grid`` describes:
    every origin observed from dev step 1 up to one common diagonal, or up to the
    last dev step once it has run off. Shared by :func:`grid_from_columns` and by
    :func:`check_grid`, which re-checks a grid it is handed because a grid is a
    plain dict a caller can build or change by hand.

    The refusal is an ``ibnr.errors.Refusal`` with the reason ``not_run_off``,
    naming the cells that are missing, or else the cells past the latest
    diagonal. ``step`` (months per development step) puts the cells' ages in
    months, and ``cum`` gives a cell past the diagonal its amount; without
    ``step`` an age is its 1-based step index.

    The diagonal the cells are compared with is the youngest origin's, as it
    always was, so the check accepts and refuses the same grids. Only the
    message looks further: it names the cells against whichever origin's
    diagonal leaves the fewest cells wrong, so one cell too many on the
    youngest origin is named as that cell, not as every older origin one cell
    short.
    """
    n_w, n_d = obs_mask.shape
    months = 1 if step is None else int(step)

    def cell(i: int, j: int, present: bool) -> RefusedCell:
        value = float(cum[i, j]) if present and cum is not None else None
        return RefusedCell(None, origins[i], (j + 1) * months, value)

    if not obs_mask[:, 0].all():
        missing = np.nonzero(~obs_mask[:, 0])[0]
        raise Refusal(
            "not_run_off",
            "the cells are not a run-off triangle: {cells} "
            + ("is" if missing.size == 1 else "are")
            + f" missing. {_RUN_OFF}",
            option="cells",
            cells=[cell(int(i), 0, False) for i in missing],
        )
    latest_dev = n_d - 1 - np.argmax(obs_mask[:, ::-1], axis=1)  # (n_w,)
    # K = the calendar diagonal, in (origin + dev) units, implied by the youngest
    # origin; every other origin must sit on the same diagonal (or be capped by
    # n_d, having already run off).
    diagonal = int(latest_dev[-1]) + (n_w - 1)
    if (obs_mask == _staircase(diagonal, n_w, n_d)).all():
        return latest_dev
    # the diagonal that leaves the fewest cells wrong, the youngest origin's on a tie
    candidates = [diagonal, *sorted({int(k) + i for i, k in enumerate(latest_dev)} - {diagonal})]
    wrong = [int((obs_mask != _staircase(k, n_w, n_d)).sum()) for k in candidates]
    reference = _staircase(candidates[int(np.argmin(wrong))], n_w, n_d)
    missing = np.argwhere(reference & ~obs_mask)
    extra = np.argwhere(obs_mask & ~reference)
    if missing.size:
        cells = [cell(int(i), int(j), False) for i, j in missing]
        verb = "is" if len(cells) == 1 else "are"
        also = f" ({len(extra)} more cell(s) are past the latest diagonal)" if len(extra) else ""
        template = f"the cells are not a run-off triangle: {{cells}} {verb} missing{also}. "
    else:
        cells = [cell(int(i), int(j), True) for i, j in extra]
        verb = "is" if len(cells) == 1 else "are"
        template = f"the cells are not a run-off triangle: {{cells}} {verb} past the latest "
        template += "diagonal. "
    raise Refusal("not_run_off", template + _RUN_OFF, option="cells", cells=cells)


def _staircase(diagonal: int, n_w: int, n_d: int) -> np.ndarray:
    expected = np.minimum(diagonal - np.arange(n_w), n_d - 1)  # (n_w,) staircase
    return np.arange(n_d)[None, :] <= expected[:, None]


#: What the chain-ladder fits (``kernels.fit_conventional`` and
#: ``kernels.fit_mack``) take a cumulative of exactly zero to be. ``"observed"``
#: keeps it as data, each kernel as it always has; ``"missing"`` follows
#: chainladder-python, which stores a zero cell as missing, so a link ratio is
#: used only when neither of its two cells is zero.
ZERO_CELLS = ("observed", "missing")


#: The grid keys the array fits read. ``premium`` is read by BF and GCC only.
_GRID_KEYS = (
    "n_w",
    "n_d",
    "cum",
    "obs_mask",
    "latest_dev",
    "origin_periods",
    "dev_grain_months",
    "measure",
)


def check_grid(grid: dict[str, Any]) -> tuple[list[dt.date], dt.date]:
    """The grid's origins as dates and its information date, or a refusal by name.

    The shared check of the two array fits, ``kernels.fit_conventional_grid``
    and ``kernels.fit_mack_grid``. A grid is a plain dict that a caller can
    build or change by hand, so neither fit trusts that it came from
    ``kernels.contract.cohort_grid_frame`` unchanged. Each check costs microseconds.

    Refused: a missing key; a measure other than ``"cumulative"``; ``n_w`` and
    ``n_d`` that are not whole numbers, or a grid with no cells; ``cum`` that is
    not a float array of shape ``(n_w, n_d)``; ``obs_mask`` that is not a
    boolean array equal to ``~isnan(cum)``; a development step that is not a
    positive whole number of months; origin periods of the wrong count, that are
    not dates, that are not the first day of a month, or that are not in
    increasing order; cells that are not a run-off triangle; ``latest_dev`` that
    is not an integer array of each origin's last observed column; origin
    periods not one development step apart (the smallest gap between
    neighbouring origins must equal the step and every gap must be a whole
    number of steps); and still-developing origins whose latest cells are on
    different dates.

    The checks of the data (origin periods that are not first days, cells that
    are not a run-off triangle, origins not one step apart, a stale diagonal)
    raise ``ibnr.errors.Refusal``. The checks of the dict's own structure (keys,
    array types and shapes, ``obs_mask``, ``latest_dev``, the measure, the
    counts) raise a plain ``ValueError``: only a grid built or changed by hand
    can fail them, so ``ibnr.methods``, which builds its own, never meets one.

    The information date is the evaluation date of the latest observed cell:
    the day before ``origin_period`` plus ``dev_lag`` months, which is a month
    end because origin periods start on the first. Origin 1988-01-01 at 12
    months is 1988-12-31.
    """
    missing = [key for key in _GRID_KEYS if key not in grid]
    if missing:
        raise ValueError(
            f"grid is missing {missing}; build it with kernels.cohort_grid_frame(), "
            "or check the keys it documents"
        )
    if grid["measure"] != "cumulative":
        raise ValueError(
            f"grid measure is {grid['measure']!r}; the chain-ladder fits need cumulative "
            "losses, so accumulate the increments before building the grid"
        )
    n_w, n_d, step = grid["n_w"], grid["n_d"], grid["dev_grain_months"]
    if not (_is_whole_number(n_w) and _is_whole_number(n_d)):
        raise ValueError(f"grid n_w and n_d must be whole numbers, got {n_w!r} and {n_d!r}")
    if n_w < 1 or n_d < 1:
        raise ValueError("grid has no cells")
    cum, mask, latest_dev = grid["cum"], grid["obs_mask"], grid["latest_dev"]
    if not isinstance(cum, np.ndarray) or cum.dtype.kind != "f" or cum.shape != (n_w, n_d):
        raise ValueError(f"grid cum must be a float array of shape (n_w, n_d) = {(n_w, n_d)}")
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != bool
        or not np.array_equal(mask, ~np.isnan(cum))
    ):
        raise ValueError("grid obs_mask must be a boolean array equal to ~isnan(cum)")
    if not _is_whole_number(step) or step < 1:
        raise ValueError(
            f"grid dev_grain_months must be a positive whole number of months, got {step!r}"
        )
    if len(grid["origin_periods"]) != n_w:
        raise ValueError(f"grid has {len(grid['origin_periods'])} origin periods for {n_w} rows")
    try:
        origins = [as_date(origin) for origin in grid["origin_periods"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grid origin_periods must be dates: {exc}") from exc
    not_starts = [o for o in origins if o.day != 1]
    if not_starts:
        raise Refusal(
            "unreadable_label",
            f"origin periods must be the first day of their period; {not_starts[:3]} are not. "
            "An origin labelled by its period's end (2010-12-31 for accident year 2010) "
            "would put every evaluation date in the wrong month",
            column="origin_period",
            cells=[RefusedCell(None, o) for o in not_starts],
        )
    expected_latest = require_run_off(mask, origins, step=int(step), cum=cum)
    if (
        not isinstance(latest_dev, np.ndarray)
        or latest_dev.dtype.kind not in "iu"
        or not np.array_equal(latest_dev, expected_latest)
    ):
        raise ValueError("grid latest_dev must be each origin's last observed dev index")
    _require_matching_grains(origins, int(step))
    ends = [
        month_end(o, (int(j) + 1) * int(step)) for o, j in zip(origins, latest_dev, strict=True)
    ]
    valuation = max(ends)
    stale = [
        o
        for o, j, end in zip(origins, latest_dev, ends, strict=True)
        if j < n_d - 1 and end != valuation
    ]
    if stale:
        raise Refusal(
            "not_run_off",
            f"origins {stale[:5]} are still developing but their latest cell is dated before "
            f"{valuation}, the latest cell of the grid; every origin that has not reached the "
            "last dev step must be observed up to the same date",
            option="cells",
            cells=[RefusedCell(None, o) for o in stale],
        )
    return origins, valuation


def _is_whole_number(value) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _require_matching_grains(origins: list[dt.date], step: int) -> None:
    """Origins one development step apart, allowing gaps of whole steps."""
    if len(origins) < 2:
        return
    months = np.array([o.year * 12 + o.month for o in origins])
    gaps = np.diff(months)
    if (gaps <= 0).any():
        raise ValueError("grid origin_periods must be in increasing order, at most one per month")
    if int(gaps.min()) != step or (gaps % step).any():
        found = sorted({int(g) for g in gaps})
        raise Refusal(
            "grain_mismatch",
            f"origin periods are {found} months apart but the development step is {step} "
            "months; the chain-ladder fits need matching origin and development grains "
            f"(neighbouring origins {step} months apart, with any gap a whole number of steps)",
            column="origin_period",
        )


def month_end(origin: dt.date, months: int) -> dt.date:
    """The last day of the month before the one that ``origin + months`` months reaches.

    For an origin on the first of a month this is the day before ``origin +
    months`` months, the package's evaluation-date convention: origin
    1988-01-01 at 12 months is 1988-12-31.
    """
    year, month = divmod(origin.year * 12 + origin.month - 1 + months, 12)
    return dt.date(year, month + 1, 1) - dt.timedelta(days=1)


def as_date(value) -> dt.date:
    """An ISO string, date, timestamp or numpy datetime64 as a plain date, or a refusal.

    Shared by the conventional kernels (a caller's information date, premium
    keys) and :func:`check_grid` (origin periods), so two dates that were typed
    differently compare equal. A missing value (``NaT``) is refused.
    """
    # No pandas here: grid_from_columns reads origin periods through this. A
    # pandas Timestamp is a datetime and is read as one; its missing value, NaT,
    # is also a datetime but is not equal to itself, which is how it is refused.
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    if isinstance(value, np.datetime64):
        # NaT comes back as None, and a year outside 1 to 9999 as a plain number
        day = value.astype("datetime64[D]").item()
        if not isinstance(day, dt.date):
            raise ValueError(f"expected an ISO date or date object, got {value!r}")
        return day
    if isinstance(value, dt.datetime) and value == value:
        return value.date()
    if isinstance(value, dt.date) and value == value:
        return value
    raise ValueError(f"expected an ISO date or date object, got {value!r}")
