"""Traditional reserving methods, one function per method, on one triangle at a time.

``from ibnr import methods``, then call the method you want by its name:

- :func:`chain_ladder`
- :func:`bornhuetter_ferguson`
- :func:`cape_cod` (Gluck's generalized Cape Cod; ``decay=1`` is the classic one)
- :func:`mack` (the chain ladder with Mack's standard errors)

Each takes the cells of ONE triangle as a table with three columns, one row per
observed cell:

- ``origin_period``: the origin period, written the way you write it: an
  integer year (``2020``), a period label (``"2020"``, ``"2020Q3"``,
  ``"2020-03"``), or a date, timestamp or ISO date string that is the period's
  first day or its last day (``2020-01-01`` or ``2020-12-31`` for the accident
  year 2020);
- ``dev_lag``: months from the start of the origin period, so the first cell of
  an annual triangle is at 12;
- ``value``: the CUMULATIVE loss in that cell.

The results echo each origin's label back in a column ``origin``, with the
value the caller wrote, next to ``origin_period``, which is always the first
day of the period. ``dev_lag`` still counts from the period's first day, so
the accident year written 2020-12-31 has its first cell at ``dev_lag`` 12.

Any table Arrow can read is accepted: a polars DataFrame, a pyarrow Table or
RecordBatch, or anything else that offers the Arrow stream interface. Other
columns are ignored. Each function returns a :class:`ReserveResult`, whose
tables are pyarrow Tables, so a service needs no DataFrame library at all; for
analysis, ``result.to_polars()`` turns any of them into a polars DataFrame
(``pip install "ibnr[polars]"``).

This module is the front door. The functions here are thin wrappers over
``ibnr.kernels``, which is where the research tools live: refitting a fixed set
of options at successive dates (``kernels.replay_conventional``), choosing
among candidates on their history (``kernels.select_conventional``), the
one-year claims development result (``kernels.one_year_cdr``), and the Triangle
path to all of these (``kernels.fit_conventional``, ``kernels.fit_mack``).
"""

from __future__ import annotations

import datetime as dt
import numbers
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ibnr.kernels.contract import as_date, check_grid, grid_from_columns
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional_grid
from ibnr.kernels.mack import fit_mack_grid

__all__ = [
    "ReserveResult",
    "bornhuetter_ferguson",
    "cape_cod",
    "chain_ladder",
    "mack",
]

#: The tables a result carries, in the order ``to_polars`` lists them.
TABLES = ("origins", "development", "link_ratios", "totals")

_CELL_COLUMNS = ("origin_period", "dev_lag", "value")


@dataclass(frozen=True)
class ReserveResult:
    """What a reserving method returns: pyarrow Tables with fixed column types.

    A missing number is an Arrow null, never NaN. ``to_polars(name)`` gives any
    of the tables as a polars DataFrame.

    Attributes
    ----------
    method : str
        The function that made it, such as ``"chain_ladder"``.
    as_of : datetime.date
        The information date, the evaluation date of the latest cell, which is
        the last day of a period (the accident year 1988 at 12 months is
        1988-12-31).
    dev_grain_months : int
        Months per development step.
    origins : pyarrow.Table
        One row per origin period: ``origin`` (the caller's label for the
        period, the value written in the cells' ``origin_period`` column, as
        an Arrow int64 for integer years of any width, string for text labels
        of any string or dictionary type, date32 for dates, or the input's
        own timestamp type, time zone included, for timestamps),
        ``origin_period`` (date32, always the first day of the period),
        ``latest_dev_lag`` (int64, months), ``latest``, ``ultimate`` and
        ``ibnr`` (float64; ``ibnr`` is ``ultimate - latest``).
        Bornhuetter-Ferguson and Cape Cod add ``expected_loss_ratio``; Mack adds
        ``mack_se`` and its two parts, ``parameter_se`` and ``process_se``.
    development : pyarrow.Table
        One row per observed development age: ``dev_lag`` (int64), ``factor``
        (the link factor from this age to the next, null at the last age),
        ``cdf`` (the factor to the last observed age, 1.0 there) and
        ``pct_reported`` (``1 / cdf``). There is no tail factor, so
        ``pct_reported`` is 1.0 at the last observed age by construction rather
        than by measurement. The chain ladder, Bornhuetter-Ferguson and Cape Cod
        add ``n_selected`` (int64, the link ratios behind the factor),
        ``unity_fallback`` and ``extreme_trimming_skipped`` (bool); Mack adds
        ``sigma`` and ``std_err`` (the factor's standard error), null at the
        last age.
    link_ratios : pyarrow.Table or None
        Every observed link ratio, one row each: ``origin`` and
        ``origin_period`` (as in ``origins``), ``from_dev_lag`` (int64, the age
        the ratio develops from), ``previous`` and ``following`` (float64, the
        cumulatives at that age and the next), ``ratio`` (float64,
        ``following / previous``, null when ``previous`` is 0), ``included``
        (bool, whether the factor used it) and ``reason`` (string:
        ``included``, ``undefined_ratio``, ``history_window``,
        ``explicit_exclusion``, ``drop_low`` or ``drop_high``). ``None`` for
        Mack, whose factors always use every ratio.
    totals : pyarrow.Table
        One row with ``latest``, ``ultimate`` and ``ibnr`` summed over the
        origins. Mack adds ``mack_se``, ``parameter_se`` and ``process_se`` for
        the total, which is not the sum of the origins' standard errors: the
        origins share the estimated factors.
    """

    method: str
    as_of: dt.date
    dev_grain_months: int
    origins: pa.Table
    development: pa.Table
    link_ratios: pa.Table | None
    totals: pa.Table

    def to_polars(self, table: str = "origins"):
        """One of the result's tables as a polars DataFrame.

        ``table`` is ``"origins"`` (the default), ``"development"``,
        ``"link_ratios"`` or ``"totals"``. Needs polars, which the ``polars``
        extra installs: ``pip install "ibnr[polars]"``.
        """
        if table not in TABLES:
            raise ValueError(f"table must be one of {TABLES}, got {table!r}")
        data = getattr(self, table)
        if data is None:
            raise ValueError(
                f"a {self.method} result has no link_ratios table: Mack's factors use every "
                "observed link ratio, so there is no selection to report"
            )
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                'to_polars needs polars; install it with pip install "ibnr[polars]"'
            ) from exc
        return pl.from_arrow(data)


def chain_ladder(
    cells,
    *,
    dev_grain_months: int = 12,
    average: str = "volume",
    history_periods: int | None = None,
    drop_high: bool = False,
    drop_low: bool = False,
    exclude=(),
    unsupported_factor: str = "raise",
    exhausted_exclusions: str = "keep",
) -> ReserveResult:
    """The chain ladder: project each origin's latest cumulative loss to ultimate.

    ``cells`` is one triangle's cells: columns ``origin_period``, ``dev_lag``
    (months from the start of the origin period) and ``value`` (cumulative
    loss), one row per observed cell, in any table Arrow can read (a polars
    DataFrame or a pyarrow Table, for example). An unobserved cell is left out,
    not given a null value. One triangle per call: two companies' rows passed
    together are refused. Every origin period from the first to the last needs
    cells, one development step apart, and every value must be zero or more.

    ``origin_period`` names each origin period in any of these ways, and may be
    dictionary-encoded (a polars Categorical or Enum):

    - an integer year with its century: ``2020`` is the period starting
      2020-01-01 (a two-digit year such as ``97`` is refused);
    - a text label: a year ``"2020"``, a quarter ``"2020Q3"`` (starting
      2020-07-01) or a month ``"2020-03"`` (starting 2020-03-01);
    - a date, a timestamp (read as its date, in its own time zone when it has
      one) or an ISO date string such as ``"2020-12-31"``. A date is read
      against ``dev_grain_months``, which is also the length of an origin
      period: the first day of a month is the period's first day, and the last
      day of a month is the period's last day. So with ``dev_grain_months=12``
      both 2020-01-01 and 2020-12-31 are the accident year 2020, and 2021-06-30
      is the year from July 2020 to June 2021; with ``dev_grain_months=3``
      2020-12-31 is the fourth quarter of 2020. Any other day is refused.

    A year, quarter or month label must name a period ``dev_grain_months``
    long. Each period must be written one way throughout the column (2020-01-01
    and 2020-12-31 together are refused), because the results echo the label
    back: ``origins`` and ``link_ratios`` carry it in a column ``origin``, the
    value written (as int64, string, date32 or the input's timestamp type, as
    :class:`ReserveResult` lists), beside ``origin_period``, which is always
    the first day of the period. ``dev_lag`` still counts from the period's
    first day, whichever day names it.

    The development options, shared with :func:`bornhuetter_ferguson` and
    :func:`cape_cod`:

    - ``dev_grain_months``: months per development step, 12 for an annual
      triangle, 3 for a quarterly one. Origin periods must be the same length.
    - ``average``: how link ratios become a factor. ``"volume"`` (the default)
      divides the sum of the later cumulatives by the sum of the earlier ones;
      ``"simple"`` is the plain mean of the ratios and ``"median"`` their median.
    - ``history_periods``: use only the latest this many link ratios at each age
      (``None``, the default, uses them all).
    - ``drop_high`` / ``drop_low``: leave out the highest and/or lowest link
      ratio at each age.
    - ``exclude``: link ratios to leave out, as ``(origin_period, dev_lag)``
      pairs, where ``dev_lag`` is the age the ratio develops FROM in whole
      months; ``(2010, 12)`` leaves out the 2010 ratio from 12 to 24 months.
      The origin may be written in any of the ways ``origin_period`` may, not
      necessarily the way the cells write it. A pair that names no link ratio
      of the triangle is refused.
    - ``unsupported_factor``: what to do at an age where no link ratio is left
      to average: ``"raise"`` (the default) refuses; ``"unity"`` uses a factor
      of 1.0 and marks the age in ``development.unity_fallback``.
    - ``exhausted_exclusions``: what to do when ``drop_high``/``drop_low``
      would leave no ratio at an age. ``"keep"`` (the default here) keeps the
      ratios untrimmed at that age and marks it in
      ``development.extreme_trimming_skipped``; ``"raise"`` refuses. The
      default differs from ``kernels.ConventionalCandidate``'s ``"raise"``
      because on a complete triangle the last age has a single link ratio, so
      ``drop_high=True`` would always be refused; chainladder-python users
      expect it to work, and the result records the skip.

    Returns a :class:`ReserveResult`. There is no tail factor: each origin is
    projected to the last observed development age.
    """
    grid, origins = _grid(cells, dev_grain_months)
    candidate = _candidate(
        "cl",
        origins,
        average=average,
        history_periods=history_periods,
        drop_high=drop_high,
        drop_low=drop_low,
        exclude=exclude,
        unsupported_factor=unsupported_factor,
        exhausted_exclusions=exhausted_exclusions,
    )
    return _conventional_result("chain_ladder", grid, origins, candidate, premium=None)


def bornhuetter_ferguson(
    cells,
    *,
    premium,
    expected_loss_ratio: float,
    dev_grain_months: int = 12,
    average: str = "volume",
    history_periods: int | None = None,
    drop_high: bool = False,
    drop_low: bool = False,
    exclude=(),
    unsupported_factor: str = "raise",
    exhausted_exclusions: str = "keep",
) -> ReserveResult:
    """Bornhuetter-Ferguson: the unreported share of an a priori ultimate.

    Each origin's ultimate is its latest cumulative loss plus
    ``premium * expected_loss_ratio * (1 - pct_reported)``, where
    ``pct_reported`` comes from the chain-ladder development pattern.

    ``cells`` and the development options are as in :func:`chain_ladder`.

    ``premium`` is keyed by origin period, never by position: either a table
    with columns ``origin_period`` and ``premium`` (a polars DataFrame or a
    pyarrow Table, for example), or a dict from origin period to amount. The
    origin periods may be written in any of the ways the cells' may, and need
    not be written the same way as the cells (integer years in the cells and
    year-end dates in premium, for example). It needs exactly one positive
    amount for each origin of the triangle and no others.

    ``expected_loss_ratio`` is the a priori loss ratio, one number for every
    origin, applied to premium.
    """
    grid, origins = _grid(cells, dev_grain_months)
    candidate = _candidate(
        "bf",
        origins,
        expected_loss_ratio=expected_loss_ratio,
        average=average,
        history_periods=history_periods,
        drop_high=drop_high,
        drop_low=drop_low,
        exclude=exclude,
        unsupported_factor=unsupported_factor,
        exhausted_exclusions=exhausted_exclusions,
    )
    return _conventional_result("bornhuetter_ferguson", grid, origins, candidate, premium=premium)


def cape_cod(
    cells,
    *,
    premium,
    decay: float = 1.0,
    dev_grain_months: int = 12,
    average: str = "volume",
    history_periods: int | None = None,
    drop_high: bool = False,
    drop_low: bool = False,
    exclude=(),
    unsupported_factor: str = "raise",
    exhausted_exclusions: str = "keep",
) -> ReserveResult:
    """Cape Cod: Bornhuetter-Ferguson with the loss ratio estimated from the triangle.

    This is Gluck's generalized Cape Cod. Each origin's expected loss ratio is
    a weighted ratio of reported losses to "used-up" premium (premium times
    ``pct_reported``) over all origins, where an origin ``k`` periods away is
    weighted by ``decay ** k``. ``decay=1`` (the default) weights every origin
    equally, which is the classic Cape Cod and one loss ratio for the whole
    triangle; ``decay=0`` uses each origin's own experience only, which is the
    chain ladder. There is no trend: amounts are compared as they are.

    ``cells`` and the development options are as in :func:`chain_ladder`, and
    ``premium`` as in :func:`bornhuetter_ferguson`. The estimated loss ratios
    are in ``origins.expected_loss_ratio``.
    """
    grid, origins = _grid(cells, dev_grain_months)
    candidate = _candidate(
        "gcc",
        origins,
        decay=decay,
        average=average,
        history_periods=history_periods,
        drop_high=drop_high,
        drop_low=drop_low,
        exclude=exclude,
        unsupported_factor=unsupported_factor,
        exhausted_exclusions=exhausted_exclusions,
    )
    return _conventional_result("cape_cod", grid, origins, candidate, premium=premium)


def mack(cells, *, dev_grain_months: int = 12, sigma_rule: str = "log_linear") -> ReserveResult:
    """Mack's chain ladder: the chain-ladder ultimate and its standard error.

    Mack (1993), distribution-free. ``cells`` is as in :func:`chain_ladder`.
    The factors are volume-weighted over every link ratio in the triangle.
    Mack's standard-error formulas are derived for exactly that estimator, and
    ibnr's Mack fit has no development options yet (no history window, no
    trimming, no exclusions), so this function does not accept them.

    ``sigma_rule`` picks how the variance is filled in at a development age
    with too few link ratios to estimate it, usually the last one:
    ``"log_linear"`` (the default, as in chainladder-python) extends a straight
    line through the logarithms of the earlier sigmas; ``"mack"`` is Mack's own
    1993 rule. The ultimates do not depend on it; the standard errors do.
    (``kernels.fit_mack`` keeps ``"mack"`` as its default, so published numbers
    made with it do not move.)

    ``origins`` and ``totals`` carry ``mack_se`` and its two parts:
    ``parameter_se``, from estimating the factors, and ``process_se``, from the
    randomness of future development, where ``mack_se ** 2 = parameter_se ** 2
    + process_se ** 2``. The standard errors need every still-developing
    origin's latest cumulative loss to be positive, and are refused otherwise.
    They also need at least one development age with two or more link ratios,
    since a sigma is estimated from the spread of link ratios: a triangle with
    at most one at every age (two origins, for example) is refused rather than
    given standard errors of 0.
    """
    grid, labels = _grid(cells, dev_grain_months)
    _, as_of = check_grid(grid)
    fit = fit_mack_grid(grid, sigma_rule=sigma_rule)
    if (fit.n_pos < 2).all():
        raise ValueError(
            "mack needs at least one development age with two or more link ratios to estimate "
            "Mack's sigma; this triangle has at most one at every age, so every sigma would be "
            "set to 0 and the standard errors would read as no uncertainty at all. "
            "chain_ladder gives the same ultimates without standard errors"
        )
    risk = fit.msep_runoff()
    step = fit.dev_grain_months
    latest, ultimate = fit.latest, fit.ultimate
    origins = pa.table(
        {
            "origin": labels.labels_for(fit.origin_periods),
            "origin_period": pa.array(fit.origin_periods, pa.date32()),
            "latest_dev_lag": pa.array((fit.latest_dev + 1) * step, pa.int64()),
            "latest": pa.array(latest, pa.float64()),
            "ultimate": pa.array(ultimate, pa.float64()),
            "ibnr": pa.array(ultimate - latest, pa.float64()),
            "mack_se": pa.array(np.sqrt(risk["msep"]), pa.float64()),
            "parameter_se": pa.array(np.sqrt(risk["parameter"]), pa.float64()),
            "process_se": pa.array(np.sqrt(risk["process"]), pa.float64()),
        }
    )
    development = pa.table(
        {
            **_pattern(fit.f, fit.n_d, step),
            "sigma": _with_last_null(np.sqrt(fit.sigma2), pa.float64()),
            "std_err": _with_last_null(np.sqrt(fit.sigma2 / fit.s), pa.float64()),
        }
    )
    totals = pa.table(
        {
            **_sums(latest, ultimate),
            "mack_se": pa.array([np.sqrt(risk["msep_total"])], pa.float64()),
            "parameter_se": pa.array([np.sqrt(risk["parameter_total"])], pa.float64()),
            "process_se": pa.array([np.sqrt(risk["process_total"])], pa.float64()),
        }
    )
    return ReserveResult("mack", as_of, step, origins, development, None, totals)


# -- the conventional methods' result ---------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """A kernel candidate, and the exclusions as the caller wrote them."""

    kernel: ConventionalCandidate
    #: each exclusion as passed, with the (period start, dev_lag) it names
    exclusions: tuple[tuple[Any, tuple[dt.date, Any]], ...]


def _candidate(method: str, origins: _Origins, *, exclude, **settings) -> _Candidate:
    if isinstance(exclude, str | bytes) or not hasattr(exclude, "__iter__"):
        raise ValueError(
            "exclude must be a sequence of (origin_period, dev_lag) pairs, such as [(2010, 12)]"
        )
    exclusions = []
    for pair in exclude:
        if not isinstance(pair, tuple | list) or len(pair) != 2:
            raise ValueError(
                f"each exclusion must be an (origin_period, dev_lag) pair, got {pair!r}"
            )
        origin, lag = pair
        start = _scalar_start(origin, "exclude origin", origins.step)
        # a numpy integer, as iterating a numpy or polars column gives, is a whole number
        if isinstance(lag, numbers.Integral) and not isinstance(lag, bool):
            lag = int(lag)
        exclusions.append((pair, (start, lag)))
    # Compared by position, not by identity: Python can hand two equal literal
    # pairs such as [(1982, 12), (1982, 12)] over as one and the same tuple.
    first_written: dict[tuple[dt.date, Any], int] = {}
    for index, (pair, key) in enumerate(exclusions):
        try:
            first = first_written.setdefault(key, index)
        except TypeError:
            continue  # an age that is not a number: the kernel refuses it by name
        if first != index:
            raise ValueError(
                f"exclude names one link ratio twice, as {_pair_shown(exclusions[first][0])} "
                f"and {_pair_shown(pair)}; list each (origin_period, dev_lag) pair once"
            )
    kernel = ConventionalCandidate(method, exclude=tuple(key for _, key in exclusions), **settings)
    return _Candidate(kernel, tuple(exclusions))


def _pair_shown(pair) -> str:
    """An exclusion as a message prints it, the origin as the caller wrote it."""
    origin, lag = pair
    return f"({_show(origin)}, {_show(lag)})"


def _conventional_result(
    name: str, grid, origins: _Origins, wrapped: _Candidate, *, premium
) -> ReserveResult:
    candidate = wrapped.kernel
    keyed = None if premium is None else _premium(premium, origins)
    fit = fit_conventional_grid(grid, candidate, premium=keyed)
    selection = fit.factor_selection
    seen = set(zip(selection["origin_period"], selection["from_dev_lag"], strict=True))
    unknown = [_pair_shown(pair) for pair, key in wrapped.exclusions if key not in seen]
    if unknown:
        raise ValueError(
            f"exclude names link ratio(s) [{', '.join(unknown)}] that the triangle does not "
            "have; each "
            "exclusion is (origin_period, dev_lag) with dev_lag the age the ratio develops "
            "FROM, and the origin must have cells at that age and the next"
        )
    step = grid["dev_grain_months"]
    table = fit.origins
    latest = table["latest"].to_numpy(dtype=float)
    ultimate = table["ultimate"].to_numpy(dtype=float)
    columns = {
        "origin": origins.labels_for(list(table["origin_period"])),
        "origin_period": pa.array(list(table["origin_period"]), pa.date32()),
        "latest_dev_lag": pa.array(table["latest_dev_lag"].to_numpy(dtype=np.int64), pa.int64()),
        "latest": pa.array(latest, pa.float64()),
        "ultimate": pa.array(ultimate, pa.float64()),
        "ibnr": pa.array(ultimate - latest, pa.float64()),
    }
    if candidate.method != "cl":
        columns["expected_loss_ratio"] = pa.array(
            table["expected_loss_ratio"].to_numpy(dtype=float), pa.float64()
        )
    summary = fit.factor_summary
    development = pa.table(
        {
            **_pattern(fit.factors, grid["n_d"], step),
            "n_selected": _with_last_null(summary["n_selected"].to_numpy(np.int64), pa.int64()),
            "unity_fallback": _with_last_null(summary["unity_fallback"].to_numpy(bool), pa.bool_()),
            "extreme_trimming_skipped": _with_last_null(
                summary["extreme_trimming_skipped"].to_numpy(bool), pa.bool_()
            ),
        }
    )
    ratio = selection["ratio"].to_numpy(dtype=float)
    link_ratios = pa.table(
        {
            "origin": origins.labels_for(list(selection["origin_period"])),
            "origin_period": pa.array(list(selection["origin_period"]), pa.date32()),
            "from_dev_lag": pa.array(selection["from_dev_lag"].to_numpy(np.int64), pa.int64()),
            "previous": pa.array(selection["previous"].to_numpy(dtype=float), pa.float64()),
            "following": pa.array(selection["following"].to_numpy(dtype=float), pa.float64()),
            # an undefined ratio (from a zero cumulative) is missing, not a number
            "ratio": pa.array(ratio, pa.float64(), mask=np.isnan(ratio)),
            "included": pa.array(selection["included"].to_numpy(bool), pa.bool_()),
            "reason": pa.array(list(selection["reason"]), pa.string()),
        }
    )
    totals = pa.table(_sums(latest, ultimate))
    return ReserveResult(name, fit.as_of, step, pa.table(columns), development, link_ratios, totals)


def _pattern(factors: np.ndarray, n_d: int, step: int) -> dict[str, pa.Array]:
    """dev_lag, factor, cdf and pct_reported, one row per observed age.

    ``factors`` has one entry per link, ``n_d - 1`` of them; there is no tail.
    """
    factors = np.asarray(factors, dtype=float)[: n_d - 1]
    cdf = np.r_[np.cumprod(factors[::-1])[::-1], 1.0]
    return {
        "dev_lag": pa.array(np.arange(1, n_d + 1, dtype=np.int64) * step, pa.int64()),
        "factor": _with_last_null(factors, pa.float64()),
        "cdf": pa.array(cdf, pa.float64()),
        "pct_reported": pa.array(1.0 / cdf, pa.float64()),
    }


def _with_last_null(values: np.ndarray, kind: pa.DataType) -> pa.Array:
    """One value per link, plus a null for the last age, which has no next age."""
    return pa.array([*values.tolist(), None], kind)


def _sums(latest: np.ndarray, ultimate: np.ndarray) -> dict[str, pa.Array]:
    total_latest, total_ultimate = float(latest.sum()), float(ultimate.sum())
    return {
        "latest": pa.array([total_latest], pa.float64()),
        "ultimate": pa.array([total_ultimate], pa.float64()),
        "ibnr": pa.array([total_ultimate - total_latest], pa.float64()),
    }


# -- origin labels -----------------------------------------------------------------

#: How an origin period may be written, for the messages.
_FORMS = (
    "write it as a year (2020), a quarter (2020Q3), a month (2020-03), or a date that is "
    "the period's first day (2020-01-01) or last day (2020-12-31)"
)
_YEAR = re.compile(r"([0-9]{4})")
_QUARTER = re.compile(r"([0-9]{4})Q([1-4])")
_MONTH = re.compile(r"([0-9]{4})-([0-9]{2})")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
#: What a year, quarter or month label is called, by its length in months.
_PERIOD_WORD = {12: "a year", 3: "a quarter", 1: "a month"}


def _label_period(text: str, name: str) -> tuple[dt.date, int | None]:
    """A text label as (the date it names, the months its period lasts).

    The months are ``None`` for a date, whose period length is read from
    ``dev_grain_months`` instead.
    """
    try:
        if match := _YEAR.fullmatch(text):
            return _year(int(match[1]), name), 12
        if match := _QUARTER.fullmatch(text):
            return dt.date(int(match[1]), 3 * int(match[2]) - 2, 1), 3
        if match := _MONTH.fullmatch(text):
            return dt.date(int(match[1]), int(match[2]), 1), 1
        if _ISO_DATE.fullmatch(text):
            return dt.date.fromisoformat(text), None
    except ValueError:
        pass  # a year 0, a month 13 or a 30 February: refused below, by the same message
    raise ValueError(f"{name} {text!r} is not an origin period: {_FORMS}")


def _period_start(
    day: dt.date, months: int | None, step: int, name: str, shown: str, *, cells: bool
) -> dt.date:
    """The first day of the period a label names, or a refusal.

    ``months`` is the length the label itself implies (12 for a year label),
    or ``None`` for a date, which is read against ``step``: the first day of a
    month starts a period and the last day of a month ends one. ``cells`` says
    whether the label is in the cells, whose origins set the period length, or
    in premium or ``exclude``, which have to follow the cells.
    """
    if months is not None:
        if months != step:
            length = "1 month" if months == 1 else f"{months} months"
            if not cells:
                raise ValueError(
                    f"{name} {shown} is {_PERIOD_WORD[months]}, {length} long, but the "
                    f"triangle's origin periods are {step} months long "
                    f"(dev_grain_months={step}). Write it as one of the triangle's origin "
                    "periods, as a label of that length or as the period's first or last day"
                )
            raise ValueError(
                f"{name} {shown} is {_PERIOD_WORD[months]}, {length} long, but "
                f"dev_grain_months={step}. The methods need origin periods one development "
                f"step long: pass dev_grain_months={months} if the triangle develops "
                f"{length} at a time. Origin periods longer than a development step "
                "(annual origins developed quarterly, for example) are not supported yet."
            )
        return day
    if day.day == 1:
        return day
    try:
        following = day + dt.timedelta(days=1)
        # the last day of a month ends a period that began step months earlier
        start = _add_months(following, -step) if following.day == 1 else None
    except (OverflowError, ValueError):
        raise ValueError(
            f"{name} {shown} would end an origin period of {step} months that starts or "
            "ends outside the years 1 to 9999"
        ) from None
    if start is not None:
        return start
    # a timestamp's label shows its time of day too, so the date read is named
    on = "" if shown in (day.isoformat(), repr(day.isoformat())) else f" falls on {day}, which"
    raise ValueError(
        f"{name} {shown}{on} is neither the first nor the last day of a month. "
        "A date must be an origin period's first day (2020-01-01) or its last day "
        f"(2020-12-31), read with dev_grain_months={step} as the period's length"
    )


def _show(value) -> str:
    """A label as a message prints it: text quoted, a year or a date as written."""
    if isinstance(value, dt.date):
        return value.isoformat()
    return repr(value) if isinstance(value, str) else str(value)


def _year(year: int, name: str) -> dt.date:
    # A year needs its century: 97 read as the year 97 would be accepted without a word.
    if not 1000 <= year <= 9999:
        raise ValueError(
            f"{name} {year} is not a four-digit year: write a year with its century "
            f"(1997, not 97). {_FORMS[0].upper()}{_FORMS[1:]}"
        )
    return dt.date(year, 1, 1)


def _scalar_start(value, name: str, step: int) -> dt.date:
    """The first day of the origin period one Python value names, or a refusal.

    Used for premium dict keys and the origins of ``exclude`` pairs, which take
    the same forms as a cell's ``origin_period`` and have to name one of the
    cells' periods.
    """
    shown = _show(value)
    if isinstance(value, bool):
        raise ValueError(f"{name} {shown} is not an origin period: {_FORMS}")
    if isinstance(value, numbers.Integral):
        return _period_start(_year(int(value), name), 12, step, name, shown, cells=False)
    if isinstance(value, str):
        day, months = _label_period(value, name)
        return _period_start(day, months, step, name, shown, cells=False)
    try:
        day = as_date(value)  # a date, a datetime (its date part) or a numpy datetime64
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} {shown} is not an origin period: {_FORMS}") from exc
    return _period_start(day, None, step, name, shown, cells=False)


@dataclass(frozen=True)
class _Origins:
    """An origin column as read: each row's period, and the caller's labels.

    ``labels`` holds each distinct label once, in the caller's Arrow type
    (int64, string, date32 or the input's timestamp type), ``shown`` each as a
    message prints it, ``label_starts`` the first day of the period each names,
    and ``row_label`` each row's position in ``labels``.
    """

    name: str
    step: int
    labels: pa.Array
    shown: list[str]
    label_starts: list[dt.date]
    row_label: np.ndarray

    @property
    def starts(self) -> np.ndarray:
        """Each row's period start, as numpy ``datetime64[D]``."""
        return np.array(self.label_starts, dtype="datetime64[D]")[self.row_label]

    def require_one_label_per_period(self) -> None:
        """Refuse a period written two ways: the results echo one label per period."""
        by_start: dict[dt.date, list[str]] = {}
        for start, shown in zip(self.label_starts, self.shown, strict=True):
            by_start.setdefault(start, []).append(shown)
        doubled = sorted((start, shown) for start, shown in by_start.items() if len(shown) > 1)
        if doubled:
            start, shown = doubled[0]
            raise ValueError(
                f"{self.name} writes the origin period starting {start} in {len(shown)} ways: "
                f"{', '.join(shown)}. Write each period one way throughout the column, "
                "because the results echo its label back"
            )

    def labels_for(self, starts) -> pa.Array:
        """The caller's label for each period start, in the caller's Arrow type."""
        position = {start: i for i, start in enumerate(self.label_starts)}
        return self.labels.take(pa.array([position[start] for start in starts], pa.int64()))

    def shown_for(self, starts) -> list[str]:
        """The caller's label for each period start, as a message prints it."""
        position = {start: i for i, start in enumerate(self.label_starts)}
        return [self.shown[position[start]] for start in starts]


def _read_origins(column: pa.ChunkedArray, name: str, step: int, *, cells: bool) -> _Origins:
    """An origin column read into periods, keeping the caller's labels, or a refusal.

    A dictionary-encoded column (a polars Categorical or Enum, for example) is
    decoded first. A timestamp is read as its date, in its own time zone when it
    has one, so a time of day is dropped, as the grid builder does for a
    ``datetime``. ``cells`` is as in :func:`_period_start`.
    """
    kind = column.type
    if column.null_count:
        raise ValueError(
            f"{name} has {column.null_count} missing value(s); every row needs its origin period"
        )
    if pa.types.is_dictionary(kind):
        # Decoded by hand, chunk by chunk: pyarrow can neither cast nor take from the
        # string_view dictionary a polars Categorical arrives as, so its values are
        # made plain strings first.
        kind = pa.string() if _string_kind(kind.value_type) else kind.value_type
        column = pa.chunked_array(
            [chunk.dictionary.cast(kind).take(chunk.indices) for chunk in column.chunks], kind
        )
    if pa.types.is_integer(kind):
        # checked before the cast, which a uint64 above the int64 range would fail
        bounds = pc.min_max(column)
        for year in (bounds["min"].as_py(), bounds["max"].as_py()):
            _year(year, name)
        column = column.cast(pa.int64())
    elif _string_kind(kind):
        column = column.cast(pa.string())
    elif pa.types.is_date(kind):
        column = column.cast(pa.date32())
    elif not pa.types.is_timestamp(kind):
        raise ValueError(
            f"{name} must be a column of integer years, text labels, dates or timestamps, "
            f"got {kind}; {_FORMS}"
        )
    labels = pc.unique(column)
    row_label = pc.index_in(column, value_set=labels).to_numpy().astype(np.int64)
    if pa.types.is_timestamp(labels.type):
        shown = labels.cast(pa.string()).to_pylist()
        # a timestamp with a time zone is read as the date in that zone
        periods = [(day, None) for day in labels.cast(pa.date32(), safe=False).to_pylist()]
    else:
        values = labels.to_pylist()
        shown = [_show(value) for value in values]
        if pa.types.is_integer(labels.type):
            periods = [(_year(value, name), 12) for value in values]
        elif pa.types.is_string(labels.type):
            periods = [_label_period(value, name) for value in values]
        else:
            periods = [(value, None) for value in values]
    starts = [
        _period_start(day, months, step, name, text, cells=cells)
        for (day, months), text in zip(periods, shown, strict=True)
    ]
    return _Origins(name, step, labels, shown, starts, row_label)


def _string_kind(kind: pa.DataType) -> bool:
    is_view = getattr(pa.types, "is_string_view", lambda _: False)
    return pa.types.is_string(kind) or pa.types.is_large_string(kind) or is_view(kind)


# -- reading the inputs ------------------------------------------------------------


def _table(data, name: str) -> pa.Table:
    """Any Arrow-readable table as a pyarrow Table; polars is never imported here."""
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    try:
        return pa.table(data)
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise ValueError(
            f"{name} must be a table Arrow can read, such as a polars DataFrame or a pyarrow "
            f"Table, not {type(data).__name__}: {exc}"
        ) from exc


def _require_columns(table: pa.Table, needed: tuple[str, ...], name: str, what: str) -> None:
    missing = [column for column in needed if column not in table.column_names]
    if missing:
        raise ValueError(
            f"{name} is missing column(s) {missing}; it has {table.column_names}. {what}"
        )


def _grid(cells, dev_grain_months) -> tuple[dict[str, Any], _Origins]:
    """The kernel grid from the cells, and the caller's origin labels."""
    if (
        not isinstance(dev_grain_months, int | np.integer)
        or isinstance(dev_grain_months, bool)
        or dev_grain_months < 1
    ):
        raise ValueError(
            "dev_grain_months must be a positive whole number of months (12 for an annual "
            f"triangle, 3 for a quarterly one), got {dev_grain_months!r}"
        )
    step = int(dev_grain_months)
    table = _table(cells, "cells")
    _require_columns(
        table,
        _CELL_COLUMNS,
        "cells",
        "It needs origin_period, dev_lag (months from the start of the origin period) and "
        "value (cumulative loss), one row per observed cell.",
    )
    if table.num_rows == 0:
        raise ValueError("cells has no rows")
    labels = _read_origins(table.column("origin_period"), "origin_period", step, cells=True)
    labels.require_one_label_per_period()
    origins = labels.starts
    lags = _whole_months(table.column("dev_lag"))
    values = _numbers(
        table.column("value"),
        "value",
        "an unobserved cell is left out of cells, not given a null value",
    )
    # A non-positive age is left to the grid builder, which names it as such.
    off_step = (lags > 0) & (lags % step != 0)
    if off_step.any():
        ages = sorted(set(lags[off_step].tolist()))
        hint = (
            "; for a quarterly triangle pass dev_grain_months=3, for a monthly one "
            "dev_grain_months=1"
            if step == 12
            else ""
        )
        raise ValueError(
            f"dev_lag values {ages[:5]} are not multiples of dev_grain_months={step}. "
            f"dev_grain_months is the months per development step{hint}"
        )
    days = origins.astype(np.int64)
    keys = np.stack([days, lags], axis=1)
    distinct, counts = np.unique(keys, axis=0, return_counts=True)
    if (counts > 1).any():
        repeated = _cells_shown(labels, distinct[counts > 1][:5])
        raise ValueError(
            f"cells has more than one row for the (origin_period, dev_lag) cell(s) {repeated}. "
            "The methods fit one cohort at a time, so filter to one company and line first; "
            "rows for the same cell are refused rather than added together"
        )
    negative = values < 0
    if negative.any():
        where = _cells_shown(labels, keys[negative][:5])
        raise ValueError(
            f"value is negative in {int(negative.sum())} cell(s), (origin_period, dev_lag) "
            f"{where}. The methods need cumulative losses of zero or more: a chain-ladder "
            "factor is a ratio of cumulatives and Mack's variance is weighted by them"
        )
    _require_consecutive_origins(labels, step)
    grid = grid_from_columns(
        origins,
        lags,
        values,
        dev_grain_months=step,
        measure="cumulative",
    )
    return grid, labels


def _day(days_since_epoch) -> dt.date:
    return dt.date(1970, 1, 1) + dt.timedelta(days=int(days_since_epoch))


def _cells_shown(labels: _Origins, keys: np.ndarray) -> str:
    """(origin, dev_lag) cells as a message prints them, each origin as the caller wrote it."""
    origins = labels.shown_for([_day(day) for day, _ in keys])
    shown = [f"({o}, {int(lag)})" for o, (_, lag) in zip(origins, keys, strict=True)]
    return f"[{', '.join(shown)}]"


def _require_consecutive_origins(labels: _Origins, step: int) -> None:
    """Refuse origin periods that are not one development step apart, naming the gap.

    The grid's run-off check counts origins by position, so with a period
    missing it accepts an older origin that stops a diagonal short of the rest
    and refuses a triangle that is a correct staircase by the calendar, with a
    message about depths rather than about the missing period.
    """
    origins = sorted(set(labels.label_starts))
    for earlier, later in zip(origins[:-1], origins[1:], strict=True):
        gap = (later.year - earlier.year) * 12 + later.month - earlier.month
        if gap == step:
            continue
        first, second = labels.shown_for([earlier, later])
        message = (
            f"origin periods {first} and {second} are {gap} months apart, but "
            f"dev_grain_months={step}. The methods need every origin period from the first "
            "to the last, one development step apart."
        )
        if gap > step and gap % step == 0:
            # the data cannot say whether a period is missing or the periods are longer;
            # a missing period has no label, so it is named by its first day
            missing = [_add_months(earlier, step * k).isoformat() for k in range(1, gap // step)]
            message += (
                f" If the origin periods are {step} months long, cells has no rows for the "
                f"period(s) starting {', '.join(missing[:5])}: give a period with no business "
                "its cells as zeros, or fit the origins on each side of the gap separately."
            )
        raise ValueError(
            message + " Origin periods longer than a development step (annual origins "
            "developed quarterly, for example) are not supported yet."
        )


def _add_months(day: dt.date, months: int) -> dt.date:
    index = day.year * 12 + day.month - 1 + months
    # origin periods start on the first; a later check refuses any that do not
    return dt.date(index // 12, index % 12 + 1, min(day.day, 28))


def _whole_months(column: pa.ChunkedArray) -> np.ndarray:
    """dev_lag as int64, or a refusal of missing, fractional or non-numeric ages."""
    kind = column.type
    if column.null_count:
        raise ValueError(f"dev_lag has {column.null_count} missing value(s)")
    if pa.types.is_integer(kind):
        return column.to_numpy().astype(np.int64)
    if pa.types.is_floating(kind):
        months = column.to_numpy().astype(float)
        fractional = ~np.isfinite(months) | (months != np.round(months))
        if fractional.any():
            raise ValueError(
                "dev_lag must be whole numbers of months, got "
                f"{sorted(set(months[fractional].tolist()))[:5]}"
            )
        return months.astype(np.int64)
    raise ValueError(f"dev_lag must be a column of whole numbers of months, got {kind}")


def _numbers(column: pa.ChunkedArray, name: str, why: str) -> np.ndarray:
    """A numeric column as float64 with nothing missing, or a refusal."""
    kind = column.type
    if not (pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind)):
        raise ValueError(f"{name} must be a numeric column, got {kind}")
    if column.null_count:
        raise ValueError(f"{name} has {column.null_count} null value(s); {why}")
    amounts = column.cast(pa.float64()).to_numpy()
    if np.isnan(amounts).any():
        raise ValueError(f"{name} has {int(np.isnan(amounts).sum())} NaN value(s); {why}")
    return amounts


def _premium(premium, origins: _Origins) -> dict[dt.date, Any]:
    """Premium keyed by period start, for ``kernels.fit_conventional_grid``.

    Its origins are read with the cells' rules and ``dev_grain_months``, so they
    may be written differently from the cells'. Repeated, missing and extra
    origins are refused here, in the caller's own labels.
    """
    step = origins.step
    if isinstance(premium, Mapping):
        keys = list(premium)
        starts = [_scalar_start(key, "premium origin", step) for key in keys]
        shown = [_show(key) for key in keys]
        amounts = list(premium.values())
        repeat_what = "amount"
    else:
        table = _table(premium, "premium")
        _require_columns(
            table,
            ("origin_period", "premium"),
            "premium",
            "It needs origin_period and premium, one row per origin period.",
        )
        read = _read_origins(
            table.column("origin_period"), "premium origin_period", step, cells=False
        )
        starts = [read.label_starts[i] for i in read.row_label]
        shown = [read.shown[i] for i in read.row_label]
        amounts = _numbers(
            table.column("premium"), "premium", "every origin needs an amount"
        ).tolist()
        repeat_what = "row"
    by_start: dict[dt.date, Any] = {}
    written: dict[dt.date, list[str]] = {}
    for start, text, amount in zip(starts, shown, amounts, strict=True):
        by_start[start] = amount
        written.setdefault(start, []).append(text)
    # each repeated period, named by the distinct ways premium writes it
    repeated = [" and ".join(dict.fromkeys(texts)) for texts in written.values() if len(texts) > 1]
    if repeated:
        raise ValueError(
            f"premium has more than one {repeat_what} for origin period(s) "
            f"{', '.join(repeated[:5])}"
        )
    periods = sorted(set(origins.label_starts))
    absent = [start for start in periods if start not in by_start]
    if absent:
        raise ValueError(
            f"premium has no amount for origin(s) {', '.join(origins.shown_for(absent[:5]))}"
        )
    extra = sorted(set(by_start) - set(periods))
    if extra:
        raise ValueError(
            f"premium has amounts for origin(s) {', '.join(written[s][0] for s in extra[:5])} "
            "that are not in the triangle; pass premium for the triangle's origins only"
        )
    return by_start
