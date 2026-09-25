"""Conventional point forecasts for fixed-setting, through-time research.

CL, Bornhuetter--Ferguson and Gluck's generalized Cape Cod share a development
pattern. These are point estimators, separate from Mack's stochastic model and
the distributional gallery. No Mack uncertainty formula is applied to a changed
factor estimator. See ``docs/conventional.md`` for the selection conventions.
"""

from __future__ import annotations

import datetime as dt
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

# as_date lives in contract so the grid check can read origin periods with it;
# it stays importable from here, where replay and selection read it.
from ibnr.kernels.contract import ZERO_CELLS, as_date, check_grid, cohort_grid
from ibnr.triangle import Triangle

#: One row per observed link ratio, whether or not it was used.
SELECTION_COLUMNS = [
    "from_dev_lag",
    "origin_period",
    "previous",
    "following",
    "ratio",
    "included",
    "reason",
]

#: One row per estimated development factor.
SUMMARY_COLUMNS = [
    "from_dev_lag",
    "factor",
    "n_selected",
    "unity_fallback",
    "extreme_trimming_skipped",
]


@dataclass(frozen=True)
class ConventionalCandidate:
    """Settings held fixed when a candidate is refitted at successive dates.

    ``history_periods`` selects the most recent usable paired origins at EACH age
    (under ``zero_cells="missing"``, the most recent paired origins, a pair left
    out for a zero cell keeping its place, as chainladder-python counts them).
    ``exclude`` identifies link ratios by (origin, FROM development lag in
    months). ``horizon`` is the final development lag in months, with no tail
    beyond it; omitted, it is the deepest observed age of this particular fit.
    Replays require an explicit horizon to keep the target fixed through time.

    ``zero_cells`` says what a cumulative of exactly zero is. ``"observed"``
    (the default) keeps it as data: the link ratio INTO a zero is 0 and is
    used, and the link ratio OUT of a zero is undefined and left out with
    reason ``"undefined_ratio"``. ``"missing"`` follows chainladder-python,
    which stores a zero cell as missing: a link ratio is used only when neither
    of its two cells is zero, and both the link into a zero and the link out of
    it are left out with reason ``"zero_cell"``. A zero on an origin's latest
    diagonal is still its latest amount under either rule, so its chain-ladder
    ultimate is 0 (chainladder-python leaves that ultimate missing instead).
    """

    method: str = "cl"
    history_periods: int | None = None
    average: str = "volume"
    drop_high: bool = False
    drop_low: bool = False
    exclude: tuple[tuple[dt.date, int], ...] = ()
    expected_loss_ratio: float | None = None
    decay: float | None = None
    horizon: int | None = None
    unsupported_factor: str = "raise"
    exhausted_exclusions: str = "raise"
    zero_cells: str = "observed"

    def __post_init__(self) -> None:
        if self.method not in ("cl", "bf", "gcc"):
            raise ValueError("method must be 'cl', 'bf', or 'gcc'")
        if self.average not in ("volume", "simple", "median"):
            raise ValueError("average must be 'volume', 'simple', or 'median'")
        # numpy scalars are accepted and converted to the builtin type, the same
        # way the float settings below already accept a numpy float: a grid
        # built from numpy ranges hands these in, and two candidates that differ
        # only in how a count was spelled must stay equal and hash alike.
        for name in ("history_periods", "horizon"):
            value = getattr(self, name)
            if value is None:
                continue
            counts = isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)
            if not counts or value < 1:
                raise ValueError(f"{name} must be a positive integer or None")
            object.__setattr__(self, name, int(value))
        for name in ("drop_high", "drop_low"):
            flag = getattr(self, name)
            if not isinstance(flag, bool | np.bool_):
                raise ValueError("drop_high and drop_low must be booleans")
            object.__setattr__(self, name, bool(flag))
        if self.unsupported_factor not in ("raise", "unity"):
            raise ValueError("unsupported_factor must be 'raise' or 'unity'")
        if self.exhausted_exclusions not in ("raise", "keep"):
            raise ValueError("exhausted_exclusions must be 'raise' or 'keep'")
        if self.zero_cells not in ZERO_CELLS:
            raise ValueError(
                f"zero_cells must be 'observed' or 'missing', got {self.zero_cells!r}. "
                "'observed' keeps a zero cumulative as data; 'missing' leaves out every link "
                "ratio with a zero at either end, as chainladder-python does"
            )
        if self.method == "bf":
            if self.expected_loss_ratio is None or not np.isfinite(self.expected_loss_ratio):
                raise ValueError("BF needs a finite expected_loss_ratio")
            if self.expected_loss_ratio < 0:
                raise ValueError("expected_loss_ratio must be non-negative")
        elif self.expected_loss_ratio is not None:
            raise ValueError("expected_loss_ratio is a BF setting only")
        if self.method == "gcc":
            if self.decay is None or not np.isfinite(self.decay) or not 0 <= self.decay <= 1:
                raise ValueError("GCC needs decay between 0 and 1")
        elif self.decay is not None:
            raise ValueError("decay is a GCC setting only")
        exclusions = tuple((as_date(origin), lag) for origin, lag in self.exclude)
        if any(type(lag) is not int or lag < 1 for _, lag in exclusions):
            raise ValueError("excluded development lags must be positive integer months")
        if len(set(exclusions)) != len(exclusions):
            raise ValueError("duplicate explicit exclusions")
        if self.horizon is not None and any(lag >= self.horizon for _, lag in exclusions):
            raise ValueError("excluded FROM lags must be strictly before the fixed horizon")
        object.__setattr__(self, "exclude", tuple(sorted(exclusions)))


@dataclass(frozen=True)
class ConventionalFit:
    """A point forecast and the observations actually used to estimate it.

    ``origins`` has one row per training origin. ``factor_selection`` records
    each observed pair's inclusion or exclusion; ``factor_summary`` records
    any unity fallback or skipped extreme trimming. Amounts use input units.
    """

    candidate: ConventionalCandidate
    as_of: dt.date
    grid: dict[str, Any]
    factors: np.ndarray
    beta: np.ndarray
    origins: pd.DataFrame
    factor_selection: pd.DataFrame
    factor_summary: pd.DataFrame

    def predict_cumulative(self, origin_period: dt.date | str, dev_lag: int) -> float:
        """Forecast an existing origin at an age after its last observation."""
        origin = as_date(origin_period)
        step = self.grid["dev_grain_months"]
        if (
            type(dev_lag) is not int
            or dev_lag % step
            or not step <= dev_lag <= len(self.beta) * step
        ):
            raise ValueError("dev_lag must be a grain multiple within the fitted horizon")
        rows = self.origins.loc[self.origins["origin_period"] == origin]
        if rows.empty:
            raise ValueError(f"origin {origin} was not present at the fit date")
        row = rows.iloc[0]
        if dev_lag < row["latest_dev_lag"]:
            raise ValueError("prediction age precedes the latest observation")
        return float(
            row["latest"] + row["prior_ultimate"] * (self.beta[dev_lag // step - 1] - row["beta"])
        )


def fit_conventional(
    triangle: Triangle,
    candidate: ConventionalCandidate,
    *,
    as_of: dt.date | str,
    loss_field: str = "paid_loss",
    premium_field: str = "earned_premium",
) -> ConventionalFit:
    """Fit one cohort using only information stored on or before ``as_of``.

    Premium is read from the same historical slice. BF's a priori loss ratio
    is supplied by the caller: it must have been available then too. Negative
    increments are permitted; cumulative amounts must be finite/non-negative.
    """
    cutoff = as_date(as_of)
    train = triangle.as_of(cutoff)
    if triangle.meta.origin_grain != triangle.meta.dev_grain:
        raise ValueError("conventional candidates require matching origin and development grains")
    grid = cohort_grid(
        train,
        loss_field=loss_field,
        premium_field=premium_field if candidate.method != "cl" else None,
    )
    return _fit_grid(grid, candidate, cutoff)


def fit_conventional_grid(
    grid: dict[str, Any],
    candidate: ConventionalCandidate,
    *,
    premium: Mapping | pd.Series | None = None,
) -> ConventionalFit:
    """Fit one cohort from a grid of plain arrays, with no Triangle and no database queries.

    The same estimator as :func:`fit_conventional`, giving the same factors,
    development proportions and ultimates for the same cells, but starting from
    arrays rather than a ``Triangle``. It is meant for a service that fits one
    small triangle per request, where the database queries inside
    :func:`fit_conventional` take longer than the fit itself. Importing it still
    imports ibis, because the kernels modules import the Triangle layer; only
    the queries are gone.

    ``grid`` is the one-cohort dict that ``kernels.cohort_grid_frame`` builds
    from a pandas frame with columns ``origin_period``, ``dev_lag`` (months) and
    ``value``; its page lists the keys this function reads. It must hold
    cumulative losses on a run-off triangle. The caller's dict is never changed.

    ``premium`` is needed for ``"bf"`` and ``"gcc"`` candidates only. It is keyed
    by origin period, never by position, so it cannot be lined up against the
    wrong origin: a dict, or a pandas Series indexed by origin period. Keys may be
    dates, timestamps, numpy datetime64 values or ISO strings, and amounts must
    be numbers. A grid can also carry premium itself, as a float array under the
    key ``"premium"`` in origin order; then this argument must be left out.

    The information date is not an argument; it is read from the grid. It is
    the evaluation date of the latest observed cell, which by this package's
    convention is the day before the origin period plus ``dev_lag`` months
    (origin 1988-01-01 at 12 months is 1988-12-31). It is returned as
    ``ConventionalFit.as_of``. Origin periods must therefore be the first day
    of their period.

    Refused, by name:

    - a grid missing one of the keys it reads, or whose arrays disagree with
      each other or have the wrong type: ``cum`` not a float array of shape
      ``(n_w, n_d)``, ``obs_mask`` not a boolean array equal to ``~isnan(cum)``,
      ``latest_dev`` not an integer array of each origin's last observed
      column, ``n_w``, ``n_d`` or ``dev_grain_months`` not whole numbers,
      ``origin_periods`` of the wrong length or not in increasing order;
    - origin periods that are not dates or not the first day of a month;
    - cells that are not a run-off triangle, or still-developing origins whose
      latest cells are on different dates;
    - a grid measure other than ``"cumulative"``;
    - origin periods not one development step apart. The smallest gap between
      neighbouring origins must equal the development step and every gap must be
      a whole number of steps. A missing origin period is allowed only where
      every origin before it has already reached the last development step (the
      run-off check counts origins by position), and GCC then measures its
      distances in real origin periods across the gap. Quarterly origins on an
      annual step are refused, and so is a grid whose origins are all two or
      more steps apart, because a grid cannot tell an annual axis with gaps from
      a two-year one;
    - premium passed with a ``"cl"`` candidate (it would never be read), given
      both here and on the grid, missing for a ``"bf"`` or ``"gcc"`` candidate,
      not keyed by origin, with a key that is not a date, with two keys that are
      the same origin once read as dates, with an amount that is not a number,
      missing an origin of the grid, or carrying an origin the grid does not
      have (filter it first). Premium on the grid must be a float array with one
      value per origin. Premium must also be finite and positive, which the
      estimator checks.

    Everything else (the horizon, exclusions, factor fallbacks) is refused
    exactly as :func:`fit_conventional` refuses it.
    """
    origins, valuation = check_grid(grid)
    n_w = grid["n_w"]
    fitted = dict(grid, origin_periods=origins)
    if candidate.method == "cl":
        if premium is not None:
            raise ValueError(
                "premium was passed for a 'cl' candidate, which never reads it; pass premium "
                "only for 'bf' or 'gcc'"
            )
    elif premium is not None:
        if grid.get("premium") is not None:
            raise ValueError(
                "premium was given twice: as premium= and already on the grid (from "
                "cohort_grid(premium_field=...)); pass it one way only"
            )
        fitted["premium"] = _premium_by_origin(premium, origins)
    elif grid.get("premium") is None:
        raise ValueError(
            f"a {candidate.method!r} candidate needs premium; pass premium= keyed by origin "
            "period, or build the grid with cohort_grid(premium_field=...)"
        )
    else:
        carried = grid["premium"]
        if (
            not isinstance(carried, np.ndarray)
            or carried.dtype.kind != "f"
            or carried.shape != (n_w,)
        ):
            raise ValueError(
                f"grid premium must be a float array with one value per origin, {n_w} in all"
            )
    return _fit_grid(fitted, candidate, valuation)


def _premium_by_origin(premium, origins: list[dt.date]) -> np.ndarray:
    """Premium keyed by origin period as an array in grid order, or a refusal."""
    if not isinstance(premium, Mapping | pd.Series):
        raise ValueError(
            "premium must be keyed by origin period (a dict, or a pandas Series indexed by "
            f"origin period), not {type(premium).__name__}; a plain sequence is refused "
            "because nothing in it says which origin each amount belongs to"
        )
    by_origin: dict[dt.date, Any] = {}
    repeated = set()
    for key, amount in premium.items():
        try:
            origin = as_date(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"premium key {key!r} is not an origin period: {exc}") from exc
        # A number, not something float() happens to accept: the string "500"
        # would convert, and pd.NA would raise a TypeError that names no premium.
        if isinstance(amount, bool) or not isinstance(amount, numbers.Real):
            raise ValueError(f"premium for origin {origin} is {amount!r}, which is not a number")
        if origin in by_origin:
            repeated.add(origin)
        by_origin[origin] = amount
    if repeated:
        raise ValueError(
            f"premium has more than one amount for origin(s) {sorted(repeated)} once its keys "
            "are read as dates"
        )
    absent = [o for o in origins if o not in by_origin]
    if absent:
        raise ValueError(f"premium has no amount for origin(s) {absent}")
    extra = sorted(set(by_origin) - set(origins))
    if extra:
        raise ValueError(
            f"premium has amounts for origin(s) {extra} that are not in the grid; "
            "pass premium for the grid's origins only"
        )
    return np.array([by_origin[o] for o in origins], dtype=float)


def _fit_grid(
    grid: dict[str, Any], candidate: ConventionalCandidate, cutoff: dt.date
) -> ConventionalFit:
    """Shared estimator for already sliced/validated grids (also used by replay)."""
    step = grid["dev_grain_months"]
    horizon = candidate.horizon or grid["n_d"] * step
    if horizon % step or horizon < grid["n_d"] * step:
        raise ValueError("horizon must be a grain multiple covering all observed development")
    if any(lag % step for _, lag in candidate.exclude):
        raise ValueError("excluded development lags must be grain multiples")
    cum, mask = grid["cum"], grid["obs_mask"]
    if not np.isfinite(cum[mask]).all() or (cum[mask] < 0).any():
        raise ValueError("cumulative losses must be finite and non-negative")
    factors, selection, summary = _factors(grid, candidate, horizon // step)
    beta = np.r_[1 / np.cumprod(factors[::-1])[::-1], 1.0]
    if not np.isfinite(beta).all() or (beta <= 0).any():
        raise ValueError("development pattern is not finite and positive")
    latest = cum[np.arange(grid["n_w"]), grid["latest_dev"]]
    developed = beta[grid["latest_dev"]]
    elr = np.full(grid["n_w"], np.nan)
    if candidate.method == "cl":
        prior = latest / developed
    else:
        premium = grid["premium"]
        if not np.isfinite(premium).all() or (premium <= 0).any():
            raise ValueError("premium must be finite and positive")
        if candidate.method == "bf":
            elr[:] = candidate.expected_loss_ratio
        else:
            # Distances are in origin periods, including any calendar gaps.
            periods = np.array([o.year * 12 + o.month for o in grid["origin_periods"]])
            distance = np.abs(periods[:, None] - periods[None, :]) / step
            weights = candidate.decay**distance  # 0**0 = 1 gives GCC(0) = CL.
            elr = (weights @ latest) / (weights @ (premium * developed))
        prior = premium * elr
    reserve = prior * (1 - developed)
    origins = pd.DataFrame(
        {
            "origin_period": grid["origin_periods"],
            "latest_dev_lag": (grid["latest_dev"] + 1) * step,
            "latest": latest,
            "beta": developed,
            "expected_loss_ratio": elr,
            "prior_ultimate": prior,
            "ultimate": latest + reserve,
            "reserve": reserve,
        }
    )
    if not np.isfinite(origins[["prior_ultimate", "ultimate", "reserve"]]).all().all():
        raise ValueError("conventional forecast is not finite")
    return ConventionalFit(candidate, cutoff, grid, factors, beta, origins, selection, summary)


def _factors(grid, candidate, n_dev):
    cum, mask = grid["cum"], grid["obs_mask"]
    step = grid["dev_grain_months"]
    factors = np.ones(n_dev - 1)
    selection, summary = [], []
    explicit = set(candidate.exclude)
    for j in range(n_dev - 1):
        pairs = np.flatnonzero(mask[:, j] & mask[:, j + 1]) if j + 1 < grid["n_d"] else []
        rows = []
        for i in pairs:
            c0, c1 = cum[i, j], cum[i, j + 1]
            rows.append(
                {
                    "from_dev_lag": (j + 1) * step,
                    "origin_period": grid["origin_periods"][i],
                    "previous": c0,
                    "following": c1,
                    "ratio": c1 / c0 if c0 > 0 else np.nan,
                    "included": True,
                    "reason": "included",
                }
            )
        for row in rows:
            if candidate.zero_cells == "missing":
                # chainladder's rule: a zero cumulative is a missing cell, so the
                # link into it and the link out of it both go
                if row["previous"] == 0 or row["following"] == 0:
                    row.update(included=False, reason="zero_cell")
            elif not np.isfinite(row["ratio"]):
                row.update(included=False, reason="undefined_ratio")
        # The history window counts the most recent pairs. Under "observed" an
        # undefined ratio gives up its place, so an older pair moves into the
        # window. Under "missing" a zero cell is a missing cell and keeps its
        # place, as chainladder-python's n_periods counts diagonals whether or not
        # their link ratio is missing, so the window simply holds fewer ratios.
        counted = rows if candidate.zero_cells == "missing" else [r for r in rows if r["included"]]
        if candidate.history_periods is not None:
            for row in counted[: max(0, len(counted) - candidate.history_periods)]:
                if row["included"]:
                    row.update(included=False, reason="history_window")
        for row in rows:
            if row["included"] and (row["origin_period"], row["from_dev_lag"]) in explicit:
                row.update(included=False, reason="explicit_exclusion")
        kept = [r for r in rows if r["included"]]
        # Stable ordering breaks low ties by oldest, high ties by newest. Both
        # refer to distinct observations, even when every ratio is tied.
        ordered = sorted(kept, key=lambda r: (r["ratio"], r["origin_period"]))
        drop = int(candidate.drop_low) + int(candidate.drop_high)
        skipped = False
        # An age with no usable pair at all is not an exhausted trim. It is an
        # age the data never supported, which is what ``unsupported_factor``
        # below decides. Trimming first answered that question with a refusal
        # about exclusions, and only when a trim happened to be requested: an
        # age beyond the observed development answered under 'unity' on its own
        # and refused as soon as drop_high was added, although there was nothing
        # for drop_high to remove either way.
        if drop and ordered:
            if len(ordered) <= drop:
                if candidate.exhausted_exclusions == "raise":
                    raise ValueError(
                        f"extreme exclusions exhaust paired origins at dev lag {(j + 1) * step}"
                    )
                skipped = True
            else:
                if candidate.drop_low:
                    ordered.pop(0).update(included=False, reason="drop_low")
                if candidate.drop_high:
                    ordered.pop().update(included=False, reason="drop_high")
        kept = [r for r in rows if r["included"]]
        fallback = not kept
        if kept:
            if candidate.average == "volume":
                denominator = sum(r["previous"] for r in kept)
                factor = (
                    sum(r["following"] for r in kept) / denominator if denominator > 0 else np.nan
                )
            else:
                reducer = np.mean if candidate.average == "simple" else np.median
                factor = float(reducer([r["ratio"] for r in kept]))
            fallback = not np.isfinite(factor) or factor <= 0
            if not fallback:
                factors[j] = factor
        if fallback and candidate.unsupported_factor == "raise":
            message = f"no estimable positive factor at dev lag {(j + 1) * step}"
            if not kept and any(r["reason"] == "zero_cell" for r in rows):
                message += (
                    ": no link ratio is left there once zero_cells='missing' leaves out those "
                    "with a zero cell. "
                    "unsupported_factor='unity' uses a factor of 1.0 at that age, as "
                    "chainladder-python does; zero_cells='observed' keeps the zeros as data"
                )
            raise ValueError(message)
        selection.extend(rows)
        summary.append(
            {
                "from_dev_lag": (j + 1) * step,
                "factor": factors[j],
                "n_selected": len(kept),
                "unity_fallback": fallback,
                "extreme_trimming_skipped": skipped,
            }
        )
    # Explicit columns: an age range with no observed pair at all leaves these
    # lists empty, and a frame built from an empty list has no columns, so a
    # caller reading factor_selection['ratio'] met a KeyError rather than an
    # empty column.
    return (
        factors,
        pd.DataFrame(selection, columns=SELECTION_COLUMNS),
        pd.DataFrame(summary, columns=SUMMARY_COLUMNS),
    )


def conventional_grid(
    *,
    history_periods: tuple[int | None, ...] = (None,),
    drop_high: tuple[bool, ...] = (False,),
    drop_low: tuple[bool, ...] = (False,),
    expected_loss_ratios: tuple[float, ...] = (),
    decays: tuple[float, ...] = (),
    **settings,
) -> tuple[ConventionalCandidate, ...]:
    """Explicit CL/BF/GCC Cartesian grid; no implicit paper-specific defaults.

    CL is always included. Empty BF ratios or GCC decays omit that family.
    Pass fixed settings such as ``horizon``, ``average`` and fallback policies.
    """
    candidates = []
    for history, high, low in product(history_periods, drop_high, drop_low):
        common = dict(history_periods=history, drop_high=high, drop_low=low, **settings)
        candidates.append(ConventionalCandidate("cl", **common))
        candidates.extend(
            ConventionalCandidate("bf", expected_loss_ratio=lr, **common)
            for lr in expected_loss_ratios
        )
        candidates.extend(ConventionalCandidate("gcc", decay=g, **common) for g in decays)
    return tuple(dict.fromkeys(candidates))
