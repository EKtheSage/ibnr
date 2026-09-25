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

A cumulative of exactly zero is read as a missing cell by default, as
chainladder-python reads it (``zero_cells="missing"``): every link ratio with a
zero at either end is left out of the factors. Pass ``zero_cells="observed"``
to keep zeros as data, which is the default of the kernels underneath.

The results echo each origin's label back in a column ``origin``, with the
value the caller wrote, next to ``origin_period``, which is always the first
day of the period. ``dev_lag`` still counts from the period's first day, so
the accident year written 2020-12-31 has its first cell at ``dev_lag`` 12.

Any table Arrow can read is accepted: a polars DataFrame, a pyarrow Table or
RecordBatch, anything else that offers the Arrow stream interface, or a dict
of columns (Python lists or numpy arrays, as a service reading JSON has them).
Other columns are ignored. Each function returns a :class:`ReserveResult`,
whose tables are pyarrow Tables, so a service needs no DataFrame library at
all; for analysis, ``result.to_polars()`` turns any of them into a polars
DataFrame (``pip install "ibnr[polars]"``).

Importing this module loads numpy and pyarrow, and not ibis, pandas or scipy,
and none of the four methods loads them when it runs, so a service that starts
a new process for a request does not pay for them (the CHANGELOG has the
times). That holds for every input above except two, which make pyarrow load
pandas: a pandas DataFrame, and a dict with a list that is not all strings,
all bools, all dates or all numbers (one holding a null or a datetime, for
example), which is left to ``pa.table``.

Every input a function in this module will not answer is refused with
:class:`Refusal`, a ``ValueError`` from ``ibnr.errors``. Its ``reason`` is a
code from a closed list (``ibnr.errors.REASONS``), such as
``"negative_cumulative"`` or ``"no_link_ratio"``, and its ``kind`` says whether
the input must change (``"input"``) or another option or method can answer
(``"model"``). It names the argument (``option``, and ``column`` for a table),
the cells, development ages or rows at fault, and each origin exactly as the
caller wrote it, the same value the result's ``origin`` column would show.
``refusal.to_dict()`` gives all of it as JSON. Any other exception from these
functions, including a plain ``ValueError``, is a defect in ibnr and should be
reported. A ``TypeError`` from a missing or misspelled keyword is Python's, and
means the calling code is wrong.

This module is the front door. The functions here are thin wrappers over
``ibnr.kernels``, which is where the research tools live: refitting a fixed set
of options at successive dates (``kernels.replay_conventional``), choosing
among candidates on their history (``kernels.select_conventional``), the
one-year claims development result (``kernels.one_year_cdr``), and the Triangle
path to all of these (``kernels.fit_conventional``, ``kernels.fit_mack``).
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ibnr import _arrow
from ibnr.errors import Refusal, RefusedCell
from ibnr.kernels.conventional import ConventionalCandidate, _estimate_grid
from ibnr.kernels.grid import as_date, check_grid, grid_from_columns
from ibnr.kernels.mack import fit_mack_grid

__all__ = [
    "Refusal",
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
        An origin whose latest cumulative is zero keeps 0 as its latest amount,
        so its chain-ladder ultimate is 0 and, under Mack with
        ``zero_cells="missing"``, its ``mack_se`` is 0 too; chainladder-python
        leaves both missing.
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
        ``included``, ``zero_cell`` (a zero at either end, left out under
        ``zero_cells="missing"``), ``undefined_ratio`` (``previous`` is 0, under
        ``zero_cells="observed"``), ``history_window``, ``explicit_exclusion``,
        ``drop_low`` or ``drop_high``). ``None`` for Mack, whose factors use
        every ratio except, under ``zero_cells="missing"``, those with a zero
        at either end.
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
            raise Refusal(
                "invalid_option",
                f"table must be one of {TABLES}, got {{given}}",
                option="table",
                given=table,
                method=self.method,
            )
        data = getattr(self, table)
        if data is None:
            raise Refusal(
                "invalid_option",
                f"a {self.method} result has no link_ratios table. Under "
                "zero_cells='missing' (the default) Mack's factors use every link ratio "
                "except those with a zero cell at either end, and "
                "methods.chain_ladder(cells).link_ratios lists the same ones with reason "
                "'zero_cell'; under zero_cells='observed' they use every observed link ratio",
                option="table",
                given=table,
                method=self.method,
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
    zero_cells: str = "missing",
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
      (``None``, the default, uses them all). Under ``zero_cells="missing"`` a
      ratio left out for a zero cell still counts as one of them, as in
      chainladder-python's ``n_periods``, so the window then holds fewer.
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
    - ``zero_cells``: what a cumulative of exactly zero is. ``"missing"`` (the
      default here) follows chainladder-python, which stores a zero cell as
      missing: a link ratio is used only when neither of its two cells is zero,
      so the ratio into a zero and the ratio out of it are both left out, shown
      in ``link_ratios`` with reason ``zero_cell``. ``"observed"`` keeps the
      zero as data: the ratio into it (0) is used, and only the ratio out of it,
      which is undefined, is left out, with reason ``undefined_ratio``. The
      default differs from ``kernels.ConventionalCandidate``'s ``"observed"``,
      so that these functions give chainladder-python's factors on triangles
      with zeros. Either way a zero on an origin's latest diagonal is its latest
      amount, so its chain-ladder ultimate is 0, where chainladder-python
      leaves that ultimate missing.

    Returns a :class:`ReserveResult`. There is no tail factor: each origin is
    projected to the last observed development age. Input it will not answer is
    refused with :class:`Refusal`, whose reason codes the module docstring
    describes.
    """
    with _CallersTerms("chain_ladder") as terms:
        grid, origins = terms.read(cells, dev_grain_months)
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
            zero_cells=zero_cells,
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
    zero_cells: str = "missing",
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
    with _CallersTerms("bornhuetter_ferguson") as terms:
        grid, origins = terms.read(cells, dev_grain_months)
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
            zero_cells=zero_cells,
        )
        return _conventional_result(
            "bornhuetter_ferguson", grid, origins, candidate, premium=premium
        )


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
    zero_cells: str = "missing",
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
    with _CallersTerms("cape_cod") as terms:
        grid, origins = terms.read(cells, dev_grain_months)
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
            zero_cells=zero_cells,
        )
        return _conventional_result("cape_cod", grid, origins, candidate, premium=premium)


def mack(
    cells,
    *,
    dev_grain_months: int = 12,
    sigma_rule: str = "log_linear",
    zero_cells: str = "missing",
) -> ReserveResult:
    """Mack's chain ladder: the chain-ladder ultimate and its standard error.

    Mack (1993), distribution-free. ``cells`` is as in :func:`chain_ladder`.
    The factors are volume-weighted over every link ratio in the triangle.
    Mack's standard-error formulas are derived for exactly that estimator, and
    ibnr's Mack fit has no development options yet (no history window, no
    trimming, no exclusions), so this function does not accept them.

    ``zero_cells`` is what a cumulative of exactly zero is. ``"missing"`` (the
    default here, as in chainladder-python) leaves out every link ratio with a
    zero at either end, from the factors and the sigmas alike. An origin whose
    latest cumulative is zero keeps 0 as its latest amount, so its ultimate is
    0 and its standard errors are 0, the limit of Mack's formula as that amount
    goes to zero; chainladder-python leaves that origin's ultimate and standard
    error missing, and its total standard error equals the one here.
    ``"observed"`` keeps zeros as data, as R's ``MackChainLadder`` and
    ``kernels.fit_mack`` (whose default it is) do: the factor uses every link
    ratio, and a zero latest cumulative on a still-developing origin is refused,
    because Mack's variance divides by it.

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
    + process_se ** 2``. Under ``zero_cells="observed"`` the standard errors
    need every still-developing origin's latest cumulative loss to be positive,
    and are refused otherwise. They also need at least one development age with
    two or more link ratios, since a sigma is estimated from the spread of link
    ratios: a triangle with at most one at every age (two origins, for example)
    is refused rather than given standard errors of 0. Input it will not answer
    is refused with :class:`Refusal`, and so is a triangle whose amounts are
    so large that a standard error is not a finite number.
    """
    with _CallersTerms("mack") as terms:
        grid, labels = terms.read(cells, dev_grain_months)
        return _mack(grid, labels, sigma_rule=sigma_rule, zero_cells=zero_cells)


def _mack(grid, labels: _Origins, *, sigma_rule: str, zero_cells: str) -> ReserveResult:
    _, as_of = check_grid(grid)
    step = grid["dev_grain_months"]
    if grid["n_d"] < 2:
        raise Refusal(
            "variance_not_estimable",
            "mack needs at least two development ages; this triangle has one. chain_ladder "
            "gives the latest amounts as the ultimates",
            option="cells",
        )
    # Amounts near the largest double are finite but Mack's sums of squares are
    # not. numpy says so with a warning, which is silenced here and below because
    # _require_finite refuses the answer by name before anything is returned.
    with np.errstate(over="ignore"):
        fit = fit_mack_grid(grid, sigma_rule=sigma_rule, zero_cells=zero_cells)
    if (fit.n_pos < 2).all():
        raise Refusal(
            "variance_not_estimable",
            "mack needs at least one development age with two or more link ratios to estimate "
            "Mack's sigma; this triangle has at most one at every age ({links}), so every "
            "sigma would be set to 0 and the standard errors would read as no uncertainty at "
            "all. chain_ladder gives the same ultimates without standard errors",
            option="cells",
            links=[((j + 1) * step, (j + 2) * step) for j in range(fit.n_d - 1)],
        )
    # Negative cells never get here (the cells are checked first), so the one
    # latest amount msep_runoff refuses is a zero under "observed". Its own
    # message names MackFit attributes a ReserveResult does not have.
    zero_latest = np.flatnonzero((fit.latest_dev < fit.n_d - 1) & (fit.latest == 0))
    if zero_cells == "observed" and zero_latest.size:
        raise Refusal(
            "variance_not_estimable",
            "mack cannot give standard errors under zero_cells='observed' while a "
            "still-developing origin's latest cumulative is zero: {origins}. Mack's variance "
            "divides by that amount. zero_cells='missing' (this function's default) gives "
            "such an origin an ultimate and a standard error of 0, and "
            "methods.chain_ladder(cells, zero_cells='observed') gives the ultimates without "
            "standard errors",
            option="zero_cells",
            cells=[
                RefusedCell(None, fit.origin_periods[i], (int(fit.latest_dev[i]) + 1) * step, 0.0)
                for i in zero_latest
            ],
        )
    # a factor so small its square is 0, or a sum of squares past the largest double:
    # amounts too large (or small) to multiply out, refused by _require_finite
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        risk = fit.msep_runoff()
        latest, ultimate = fit.latest, fit.ultimate
        per_origin = {
            "ultimate": ultimate,
            "ibnr": ultimate - latest,
            "mack_se": np.sqrt(risk["msep"]),
            "parameter_se": np.sqrt(risk["parameter"]),
            "process_se": np.sqrt(risk["process"]),
        }
        sigma = np.sqrt(fit.sigma2)
        std_err = np.sqrt(fit.sigma2 / fit.s)
        total = {
            **_sums(latest, ultimate),
            "mack_se": _arrow.float64([np.sqrt(risk["msep_total"])]),
            "parameter_se": _arrow.float64([np.sqrt(risk["parameter_total"])]),
            "process_se": _arrow.float64([np.sqrt(risk["process_total"])]),
        }
    _require_finite(fit.origin_periods, per_origin, np.r_[sigma, std_err], total)
    origins = pa.table(
        {
            "origin": labels.labels_for(fit.origin_periods),
            "origin_period": _arrow.date32(fit.origin_periods),
            "latest_dev_lag": _arrow.int64((fit.latest_dev + 1) * step),
            "latest": _arrow.float64(latest),
            **{name: _arrow.float64(values) for name, values in per_origin.items()},
        }
    )
    development = pa.table(
        {
            **_pattern(fit.f, fit.n_d, step),
            "sigma": _with_last_null(sigma, pa.float64()),
            "std_err": _with_last_null(std_err, pa.float64()),
        }
    )
    return ReserveResult("mack", as_of, step, origins, development, None, pa.table(total))


def _require_finite(periods, per_origin: dict, others, total: dict) -> None:
    """Refuse an answer with a number that is not finite, naming the origins.

    The inputs were checked finite, so such a number comes from amounts so large
    that a sum, a square or a product passes the largest double (or so far
    apart that a factor's square is 0). A result never carries NaN or infinity,
    which Arrow would store as numbers, not nulls. ``per_origin`` holds the
    per-origin columns, ``others`` any other numbers (Mack's sigmas), ``total``
    the one-row totals.
    """
    bad = np.zeros(len(periods), dtype=bool)
    for values in per_origin.values():
        bad |= ~np.isfinite(np.asarray(values, dtype=float))
    if bad.any():
        raise Refusal(
            "result_not_finite",
            "the ultimate or the standard error for {origins} is not a finite number: the "
            "amounts are too large (or too far apart) for their squares and products to "
            "stay finite. Scale them (work in thousands, say) and scale the answer back",
            option="cells",
            cells=[RefusedCell(None, periods[i]) for i in np.flatnonzero(bad)],
        )
    totals = [column[0].as_py() for column in total.values()]
    if not (np.isfinite(others).all() and all(math.isfinite(t) for t in totals)):
        raise Refusal(
            "result_not_finite",
            "a total or a development age's sigma is not a finite number: the amounts are "
            "too large (or too far apart) for their sums and squares to stay finite. Scale "
            "them (work in thousands, say) and scale the answer back",
            option="cells",
        )


# -- refusals in the caller's terms ------------------------------------------------


class _CallersTerms:
    """Re-raise a kernel's :class:`Refusal` as the caller's.

    The kernels name an origin by the first day of its period, because they
    never see the caller's labels. Inside this block a refusal comes out with
    the method named and each cell labelled as the caller wrote its origin
    (found by the period's first day, never by position), and the message is
    rendered again from its template. Only a ``Refusal`` is touched: any other
    exception from underneath is an ibnr defect and passes through as it was
    raised. The kernel's line stays the last frame of the traceback.
    """

    def __init__(self, method: str) -> None:
        self.method = method
        self.labels: _Origins | None = None

    def __enter__(self) -> _CallersTerms:
        return self

    def read(self, cells, dev_grain_months) -> tuple[dict[str, Any], _Origins]:
        """The grid and the labels, which later refusals are labelled with."""
        grid, labels = _grid(cells, dev_grain_months, terms=self)
        self.labels = labels
        return grid, labels

    def __exit__(self, kind, error, trace) -> bool:
        if not isinstance(error, Refusal):
            return False
        labels = self.labels
        label_of = labels.label_for_start if labels is not None else _no_label
        raise error.relabeled(method=self.method, label_of=label_of).with_traceback(trace) from None


def _no_label(_start: dt.date) -> None:
    return None


# -- the conventional methods' result ---------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """A kernel candidate, and the exclusions as the caller wrote them."""

    kernel: ConventionalCandidate
    #: each exclusion as passed, with the (period start, dev_lag) it names
    exclusions: tuple[tuple[Any, tuple[dt.date, Any]], ...]


def _candidate(method: str, origins: _Origins, *, exclude, **settings) -> _Candidate:
    if isinstance(exclude, str | bytes) or not hasattr(exclude, "__iter__"):
        raise Refusal(
            "invalid_option",
            "exclude must be a sequence of (origin_period, dev_lag) pairs, such as "
            "[(2010, 12)], got {given}",
            option="exclude",
            given=exclude,
        )
    step = origins.step
    exclusions = []
    for pair in exclude:
        if not isinstance(pair, tuple | list) or len(pair) != 2:
            raise Refusal(
                "invalid_option",
                "each exclusion must be an (origin_period, dev_lag) pair, got {given}",
                option="exclude",
                given=tuple(pair) if isinstance(pair, list) else pair,
            )
        origin, lag = pair
        try:
            start = _scalar_start(origin, "exclude origin", step)
        except Refusal as refusal:
            raise refusal._replace(option="exclude") from None
        # a numpy integer, as iterating a numpy or polars column gives, is a whole number
        if not isinstance(lag, numbers.Integral) or isinstance(lag, bool) or lag < 1:
            raise Refusal(
                "invalid_option",
                f"exclude {{given}} names development age {_show(lag)}, which is not a "
                "positive whole number of months",
                option="exclude",
                given=(origin, lag),
            )
        lag = int(lag)
        if lag % step:
            raise Refusal(
                "grain_mismatch",
                f"exclude names {{cells}}, but {lag} months is not a development age of this "
                f"triangle (dev_grain_months={step})",
                option="exclude",
                cells=[RefusedCell(origin, start, lag)],
            )
        exclusions.append((pair, (start, lag)))
    # Compared by position, not by identity: Python can hand two equal literal
    # pairs such as [(1982, 12), (1982, 12)] over as one and the same tuple.
    first_written: dict[tuple[dt.date, Any], int] = {}
    for index, (pair, key) in enumerate(exclusions):
        first = first_written.setdefault(key, index)
        if first != index:
            (start, lag), earlier = key, exclusions[first][0]
            raise Refusal(
                "duplicate",
                "exclude names one link ratio twice, as {cells}; list each "
                "(origin_period, dev_lag) pair once",
                option="exclude",
                cells=[RefusedCell(earlier[0], start, lag), RefusedCell(pair[0], start, lag)],
                quoted=True,
            )
    kernel = ConventionalCandidate(method, exclude=tuple(key for _, key in exclusions), **settings)
    return _Candidate(kernel, tuple(exclusions))


def _conventional_result(
    name: str, grid, origins: _Origins, wrapped: _Candidate, *, premium
) -> ReserveResult:
    candidate = wrapped.kernel
    keyed = None if candidate.method == "cl" else _premium(premium, origins, name)
    # fit_conventional_grid without its three pandas tables: the same checks and
    # numbers, held in numpy arrays and lists, so a fit here never loads pandas.
    # An overflow (amounts near the largest double) is not warned about, because
    # the estimator refuses a pattern or an ultimate that is not finite by name.
    with np.errstate(over="ignore"):
        fit = _estimate_grid(grid, candidate, premium=keyed)
    selection = fit.selection
    seen = {(row["origin_period"], row["from_dev_lag"]) for row in selection}
    unknown = [
        RefusedCell(pair[0], start, lag)
        for pair, (start, lag) in wrapped.exclusions
        if (start, lag) not in seen
    ]
    if unknown:
        raise Refusal(
            "not_in_triangle",
            "exclude names link ratio(s) {cells} that the triangle does not have; each "
            "exclusion is (origin_period, dev_lag) with dev_lag the age the ratio develops "
            "FROM, and the origin must have cells at that age and the next",
            option="exclude",
            cells=unknown,
        )
    step = grid["dev_grain_months"]
    table = fit.origins
    latest = np.asarray(table["latest"], dtype=float)
    ultimate = np.asarray(table["ultimate"], dtype=float)
    columns = {
        "origin": origins.labels_for(table["origin_period"]),
        "origin_period": _arrow.date32(table["origin_period"]),
        "latest_dev_lag": _arrow.int64(table["latest_dev_lag"]),
        "latest": _arrow.float64(latest),
        "ultimate": _arrow.float64(ultimate),
        "ibnr": _arrow.float64(ultimate - latest),
    }
    if candidate.method != "cl":
        columns["expected_loss_ratio"] = _arrow.float64(table["expected_loss_ratio"])

    def summary(key: str) -> list:
        return [row[key] for row in fit.summary]

    development = pa.table(
        {
            **_pattern(fit.factors, grid["n_d"], step),
            "n_selected": _with_last_null(summary("n_selected"), pa.int64()),
            "unity_fallback": _with_last_null(summary("unity_fallback"), pa.bool_()),
            "extreme_trimming_skipped": _with_last_null(
                summary("extreme_trimming_skipped"), pa.bool_()
            ),
        }
    )

    def link(key: str) -> list:
        return [row[key] for row in selection]

    ratio = np.array(link("ratio"), dtype=float)
    link_ratios = pa.table(
        {
            "origin": origins.labels_for(link("origin_period")),
            "origin_period": _arrow.date32(link("origin_period")),
            "from_dev_lag": _arrow.int64(link("from_dev_lag")),
            "previous": _arrow.float64(link("previous")),
            "following": _arrow.float64(link("following")),
            # an undefined ratio (from a zero cumulative) is missing, not a number
            "ratio": _arrow.float64(ratio, mask=np.isnan(ratio)),
            "included": _arrow.bool_(link("included")),
            "reason": _arrow.string(link("reason")),
        }
    )
    # the kernel checks each origin; a sum of finite ultimates can still overflow
    with np.errstate(over="ignore"):
        sums = _sums(latest, ultimate)
    _require_finite(table["origin_period"], {}, np.r_[fit.factors, ratio[~np.isnan(ratio)]], sums)
    totals = pa.table(sums)
    return ReserveResult(name, fit.as_of, step, pa.table(columns), development, link_ratios, totals)


def _pattern(factors: np.ndarray, n_d: int, step: int) -> dict[str, pa.Array]:
    """dev_lag, factor, cdf and pct_reported, one row per observed age.

    ``factors`` has one entry per link, ``n_d - 1`` of them; there is no tail.
    """
    factors = np.asarray(factors, dtype=float)[: n_d - 1]
    cdf = np.r_[np.cumprod(factors[::-1])[::-1], 1.0]
    return {
        "dev_lag": _arrow.int64(np.arange(1, n_d + 1, dtype=np.int64) * step),
        "factor": _with_last_null(factors, pa.float64()),
        "cdf": _arrow.float64(cdf),
        "pct_reported": _arrow.float64(1.0 / cdf),
    }


def _with_last_null(values, kind: pa.DataType) -> pa.Array:
    """One value per link, plus a null for the last age, which has no next age."""
    return _arrow.with_last_null(values, kind)


def _sums(latest: np.ndarray, ultimate: np.ndarray) -> dict[str, pa.Array]:
    total_latest, total_ultimate = float(latest.sum()), float(ultimate.sum())
    return {
        "latest": _arrow.float64([total_latest]),
        "ultimate": _arrow.float64([total_ultimate]),
        "ibnr": _arrow.float64([total_ultimate - total_latest]),
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
            return _year(int(match[1]), name, given=text), 12
        if match := _QUARTER.fullmatch(text):
            return dt.date(int(match[1]), 3 * int(match[2]) - 2, 1), 3
        if match := _MONTH.fullmatch(text):
            return dt.date(int(match[1]), int(match[2]), 1), 1
        if _ISO_DATE.fullmatch(text):
            return dt.date.fromisoformat(text), None
    except ValueError:
        pass  # a year 0, a month 13 or a 30 February: refused below, by the same message
    raise Refusal(
        "unreadable_label", f"{name} {{given}} is not an origin period: {_FORMS}", given=text
    )


def _period_start(
    day: dt.date,
    months: int | None,
    step: int,
    name: str,
    shown: str,
    *,
    cells: bool,
    given: Any,
) -> dt.date:
    """The first day of the period a label names, or a refusal.

    ``months`` is the length the label itself implies (12 for a year label),
    or ``None`` for a date, which is read against ``step``: the first day of a
    month starts a period and the last day of a month ends one. ``cells`` says
    whether the label is in the cells, whose origins set the period length, or
    in premium or ``exclude``, which have to follow the cells. ``given`` is the
    label as the caller wrote it, for the refusal.
    """
    if months is not None:
        if months != step:
            length = "1 month" if months == 1 else f"{months} months"
            if not cells:
                raise Refusal(
                    "grain_mismatch",
                    f"{name} {shown} is {_PERIOD_WORD[months]}, {length} long, but the "
                    f"triangle's origin periods are {step} months long "
                    f"(dev_grain_months={step}). Write it as one of the triangle's origin "
                    "periods, as a label of that length or as the period's first or last day",
                    given=given,
                )
            raise Refusal(
                "grain_mismatch",
                f"{name} {shown} is {_PERIOD_WORD[months]}, {length} long, but "
                f"dev_grain_months={step}. The methods need origin periods one development "
                f"step long: pass dev_grain_months={months} if the triangle develops "
                f"{length} at a time. Origin periods longer than a development step "
                "(annual origins developed quarterly, for example) are not supported yet.",
                given=given,
            )
        return day
    if day.day == 1:
        return day
    try:
        following = day + dt.timedelta(days=1)
        # the last day of a month ends a period that began step months earlier
        start = _add_months(following, -step) if following.day == 1 else None
    except (OverflowError, ValueError):
        raise Refusal(
            "unreadable_label",
            f"{name} {shown} would end an origin period of {step} months that starts or "
            "ends outside the years 1 to 9999",
            given=given,
        ) from None
    if start is not None:
        return start
    # a timestamp's label shows its time of day too, so the date read is named
    on = "" if shown in (day.isoformat(), repr(day.isoformat())) else f" falls on {day}, which"
    raise Refusal(
        "unreadable_label",
        f"{name} {shown}{on} is neither the first nor the last day of a month. "
        "A date must be an origin period's first day (2020-01-01) or its last day "
        f"(2020-12-31), read with dev_grain_months={step} as the period's length",
        given=given,
    )


def _show(value) -> str:
    """A label as a message prints it: text quoted, a year or a date as written."""
    if isinstance(value, dt.date):
        return value.isoformat()
    return repr(value) if isinstance(value, str) else str(value)


def _year(year: int, name: str, *, given: Any = None) -> dt.date:
    # A year needs its century: 97 read as the year 97 would be accepted without a word.
    if not 1000 <= year <= 9999:
        raise Refusal(
            "unreadable_label",
            f"{name} {year} is not a four-digit year: write a year with its century "
            f"(1997, not 97). {_FORMS[0].upper()}{_FORMS[1:]}",
            given=year if given is None else given,
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
        raise Refusal(
            "unreadable_label", f"{name} {shown} is not an origin period: {_FORMS}", given=value
        )
    if isinstance(value, numbers.Integral):
        day = _year(int(value), name, given=value)
        return _period_start(day, 12, step, name, shown, cells=False, given=value)
    if isinstance(value, str):
        day, months = _label_period(value, name)
        return _period_start(day, months, step, name, shown, cells=False, given=value)
    try:
        day = as_date(value)  # a date, a datetime (its date part) or a numpy datetime64
    except (TypeError, ValueError) as exc:
        raise Refusal(
            "unreadable_label", f"{name} {shown} is not an origin period: {_FORMS}", given=value
        ) from exc
    return _period_start(day, None, step, name, shown, cells=False, given=value)


@dataclass(frozen=True)
class _Origins:
    """An origin column as read: each row's period, and the caller's labels.

    ``labels`` holds each distinct label once, in the caller's Arrow type
    (int64, string, date32 or the input's timestamp type), ``shown`` each as a
    message prints it, ``label_starts`` the first day of the period each names,
    and ``row_label`` each row's position in ``labels``. ``lengths_known`` is
    whether every label names its period's length itself (a year, quarter or
    month label, not a date).
    """

    name: str
    step: int
    labels: pa.Array
    shown: list[str]
    label_starts: list[dt.date]
    row_label: np.ndarray
    lengths_known: bool = False

    @property
    def starts(self) -> np.ndarray:
        """Each row's period start, as numpy ``datetime64[D]``."""
        return np.array(self.label_starts, dtype="datetime64[D]")[self.row_label]

    @cached_property
    def values(self) -> list:
        """Each distinct label as a Python value (int, str, date or datetime)."""
        return self.labels.to_pylist()

    def require_one_label_per_period(self) -> None:
        """Refuse a period written two ways: the results echo one label per period."""
        by_start: dict[dt.date, list[int]] = {}
        for index, start in enumerate(self.label_starts):
            by_start.setdefault(start, []).append(index)
        doubled = sorted((start, ways) for start, ways in by_start.items() if len(ways) > 1)
        if doubled:
            values = self.values
            raise Refusal(
                "duplicate",
                f"{self.name} writes one origin period in more than one way: {{cells}}. Write "
                "each period one way throughout the column, because the results echo its "
                "label back",
                option="cells",
                column="origin_period",
                cells=[
                    RefusedCell(values[index], start) for start, ways in doubled for index in ways
                ],
                quoted=True,
            )

    def label_for_start(self, start: dt.date):
        """The caller's label for the period starting ``start``, or ``None``."""
        for index, label_start in enumerate(self.label_starts):
            if label_start == start:
                return self.values[index]
        return None

    def cell(self, row: int, dev_lag=None, value=None) -> RefusedCell:
        """Row ``row`` of the column as a refused cell, its origin as the caller wrote it."""
        index = int(self.row_label[row])
        return RefusedCell(self.values[index], self.label_starts[index], dev_lag, value)

    def labels_for(self, starts) -> pa.Array:
        """The caller's label for each period start, in the caller's Arrow type."""
        position = {start: i for i, start in enumerate(self.label_starts)}
        return self.labels.take(_arrow.int64([position[start] for start in starts]))

    def shown_for(self, starts) -> list[str]:
        """The caller's label for each period start, as a message prints it."""
        position = {start: i for i, start in enumerate(self.label_starts)}
        return [self.shown[position[start]] for start in starts]


def _read_origins(
    column: pa.ChunkedArray, name: str, step: int, *, cells: bool, option: str
) -> _Origins:
    """An origin column read into periods, keeping the caller's labels, or a refusal.

    A dictionary-encoded column (a polars Categorical or Enum, for example) is
    decoded first. A timestamp is read as its date, in its own time zone when it
    has one, so a time of day is dropped, as the grid builder does for a
    ``datetime``. ``cells`` is as in :func:`_period_start`; ``option`` is the
    argument the column belongs to, for a refusal, which names the rows of a
    label it cannot read.
    """
    kind = column.type
    where = {"option": option, "column": "origin_period"}
    if column.null_count:
        raise Refusal(
            "missing_value",
            f"{name} has {column.null_count} missing value(s); every row needs its origin period",
            rows=_positions(column.is_null()),
            **where,
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
            try:
                _year(year, name)
            except Refusal as refusal:
                rows = np.flatnonzero(_arrow.to_numpy(column) == year)
                raise refusal._replace(rows=rows, **where) from None
        column = column.cast(pa.int64())
    elif _string_kind(kind):
        column = column.cast(pa.string())
    elif pa.types.is_date(kind):
        column = column.cast(pa.date32())
    elif not pa.types.is_timestamp(kind):
        raise Refusal(
            "invalid_table",
            f"{name} must be a column of integer years, text labels, dates or timestamps, "
            f"got {kind}; {_FORMS}",
            **where,
        )
    labels = pc.unique(column)
    row_label = _arrow.to_numpy(pc.index_in(column, value_set=labels)).astype(np.int64)
    values = labels.to_pylist()
    if pa.types.is_timestamp(labels.type):
        shown = labels.cast(pa.string()).to_pylist()
        # a timestamp with a time zone is read as the date in that zone
        days = labels.cast(pa.date32(), safe=False).to_pylist()
    else:
        shown = [_show(value) for value in values]
    starts = []
    lengths_known = True
    for index, value in enumerate(values):
        try:
            if pa.types.is_timestamp(labels.type):
                day, months = days[index], None
            elif pa.types.is_integer(labels.type):
                day, months = _year(value, name), 12
            elif pa.types.is_string(labels.type):
                day, months = _label_period(value, name)
            else:
                day, months = value, None
            starts.append(
                _period_start(day, months, step, name, shown[index], cells=cells, given=value)
            )
        except Refusal as refusal:
            rows = np.flatnonzero(row_label == index)
            raise refusal._replace(rows=rows, **where) from None
        lengths_known = lengths_known and months is not None
    return _Origins(name, step, labels, shown, starts, row_label, lengths_known)


def _positions(mask) -> list[int]:
    """The row positions where a boolean Arrow column is true."""
    if isinstance(mask, pa.ChunkedArray):
        mask = mask.combine_chunks()
    return pc.indices_nonzero(mask).to_pylist()


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
    if isinstance(data, Mapping) and data and all(isinstance(key, str) for key in data):
        # pa.table on a dict of lists calls pa.array, which imports pandas, and a
        # service reading JSON has exactly that. So the columns pa.table would build
        # from lists and numpy arrays are built here without it; any other dict
        # goes to pa.table below.
        columns = [_arrow.column(values) for values in data.values()]
        if all(column is not None for column in columns):
            try:
                return pa.Table.from_arrays(columns, names=list(data))
            except (TypeError, ValueError, pa.ArrowException) as exc:
                raise _unreadable_table(data, name, exc) from exc
    pandas = sys.modules.get("pandas")
    is_pandas = pandas is not None and isinstance(data, pandas.DataFrame)
    if hasattr(data, "__arrow_c_stream__") and not is_pandas:
        # What pa.table does with such an object, without its first step: pa.table
        # asks whether the object is a pandas DataFrame, which imports pandas.
        # A pandas DataFrame still goes through pa.table, as it always has.
        try:
            return pa.RecordBatchReader.from_stream(data).read_all()
        except (TypeError, ValueError, pa.ArrowException) as exc:
            raise _unreadable_table(data, name, exc) from exc
    try:
        return pa.table(data)
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise _unreadable_table(data, name, exc) from exc


def _unreadable_table(data, name: str, exc: Exception) -> Refusal:
    also = " A dict from origin period to amount is accepted too." if name == "premium" else ""
    return Refusal(
        "invalid_table",
        f"{name} must be a table Arrow can read, such as a polars DataFrame or a pyarrow "
        f"Table, not {type(data).__name__}: {exc}.{also}",
        option=name,
    )


def _require_columns(table: pa.Table, needed: tuple[str, ...], name: str, what: str) -> None:
    missing = [column for column in needed if column not in table.column_names]
    if missing:
        raise Refusal(
            "invalid_table",
            f"{name} is missing column(s) {missing}; it has {table.column_names}. {what}",
            option=name,
            column=missing[0],
        )


def _grid(
    cells, dev_grain_months, terms: _CallersTerms | None = None
) -> tuple[dict[str, Any], _Origins]:
    """The kernel grid from the cells, and the caller's origin labels."""
    if (
        not isinstance(dev_grain_months, int | np.integer)
        or isinstance(dev_grain_months, bool)
        or dev_grain_months < 1
    ):
        raise Refusal(
            "invalid_option",
            "dev_grain_months must be a positive whole number of months (12 for an annual "
            "triangle, 3 for a quarterly one), got {given}",
            option="dev_grain_months",
            given=dev_grain_months,
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
        raise Refusal("invalid_table", "cells has no rows", option="cells")
    labels = _read_origins(
        table.column("origin_period"), "origin_period", step, cells=True, option="cells"
    )
    if terms is not None:
        terms.labels = labels  # a refusal from here on names the caller's labels
    labels.require_one_label_per_period()
    origins = labels.starts
    lags = _whole_months(table.column("dev_lag"))

    def cells_at(rows: np.ndarray, amounts=None) -> list[RefusedCell]:
        return [
            labels.cell(int(r), int(lags[r]), None if amounts is None else _amount(amounts[r]))
            for r in rows
        ]

    values = _numbers(
        table.column("value"),
        "value",
        "an unobserved cell is left out of cells, not given a null value",
        option="cells",
        cell_at=lambda rows: cells_at(rows),
    )
    where = {"option": "cells", "column": "dev_lag"}
    non_positive = np.flatnonzero(lags <= 0)
    if non_positive.size:
        ages = sorted(set(lags[non_positive].tolist()))
        raise Refusal(
            "invalid_age",
            f"dev_lag must be positive, got {ages[:5]}, in {{cells}}; dev_lag is months from "
            f"the start of the origin period, so the first cell of a {step}-month step is at "
            f"{step}",
            cells=cells_at(non_positive, values),
            **where,
        )
    off_step = np.flatnonzero(lags % step != 0)
    if off_step.size:
        ages = sorted(set(lags[off_step].tolist()))
        hint = (
            "; for a quarterly triangle pass dev_grain_months=3, for a monthly one "
            "dev_grain_months=1"
            if step == 12
            else ""
        )
        raise Refusal(
            "grain_mismatch",
            f"dev_lag values {ages[:5]} are not multiples of dev_grain_months={step}, in "
            f"{{cells}}. dev_grain_months is the months per development step{hint}",
            cells=cells_at(off_step, values),
            **where,
        )
    days = origins.astype(np.int64)
    keys = np.stack([days, lags], axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    repeated = np.flatnonzero(counts.reshape(-1)[inverse.reshape(-1)] > 1)
    if repeated.size:
        raise Refusal(
            "duplicate",
            "cells has more than one row for the same cell: {cells}. The methods fit one "
            "triangle at a time, so filter to one company and line first; rows for the same "
            "cell are refused rather than added together",
            option="cells",
            cells=cells_at(repeated, values),
        )
    negative = np.flatnonzero(values < 0)
    if negative.size:
        raise Refusal(
            "negative_cumulative",
            f"value is negative in {negative.size} cell(s): {{cells}}. The methods need "
            "cumulative losses of zero or more: a chain-ladder factor is a ratio of "
            "cumulatives and Mack's variance is weighted by them",
            option="cells",
            column="value",
            cells=cells_at(negative, values),
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


def _amount(value) -> float | None:
    """An amount for a refusal: the number, or None when it is not a finite number."""
    number = float(value)
    return number if math.isfinite(number) else None


def _require_consecutive_origins(labels: _Origins, step: int) -> None:
    """Refuse origin periods that are not one development step apart, naming the gap.

    The grid's run-off check counts origins by position, so with a period
    missing it accepts an older origin that stops a diagonal short of the rest
    and refuses a triangle that is a correct staircase by the calendar, with a
    message about depths rather than about the missing period.
    """
    origins = sorted(set(labels.label_starts))
    gaps = [
        (later.year - earlier.year) * 12 + later.month - earlier.month
        for earlier, later in zip(origins[:-1], origins[1:], strict=True)
    ]
    # Periods missing from the middle (origin_gap) where the labels say their
    # periods are one step long (a year label on an annual step: the labels were
    # checked against the step), or where some neighbours are one step apart.
    # Dates never one step apart are periods of another length (grain_mismatch),
    # such as annual origins on a quarterly step.
    gapped = bool(gaps) and (labels.lengths_known or min(gaps) == step)
    for earlier, later, gap in zip(origins[:-1], origins[1:], gaps, strict=True):
        if gap == step:
            continue
        first, second = labels.shown_for([earlier, later])
        message = (
            f"origin periods {first} and {second} are {gap} months apart, but "
            f"dev_grain_months={step}. The methods need every origin period from the first "
            "to the last, one development step apart."
        )
        # the two neighbours, as the caller wrote them
        cells = [RefusedCell(labels.label_for_start(start), start) for start in (earlier, later)]
        reason = "grain_mismatch"
        if gap > step and gap % step == 0:
            # the data cannot say whether a period is missing or the periods are longer;
            # a missing period has no label, so it is named by its first day
            missing = [_add_months(earlier, step * k) for k in range(1, gap // step)]
            message += (
                f" If the origin periods are {step} months long, cells has no rows for the "
                f"period(s) starting {', '.join(day.isoformat() for day in missing[:5])}: "
                "give a period with no business its cells as zeros, or fit the origins on "
                "each side of the gap separately."
            )
            if gapped:
                cells += [RefusedCell(None, day) for day in missing]
                reason = "origin_gap"
        raise Refusal(
            reason,
            message + " Origin periods longer than a development step (annual origins "
            "developed quarterly, for example) are not supported yet.",
            option="cells",
            column="origin_period",
            cells=cells,
        )


def _add_months(day: dt.date, months: int) -> dt.date:
    index = day.year * 12 + day.month - 1 + months
    # origin periods start on the first; a later check refuses any that do not
    return dt.date(index // 12, index % 12 + 1, min(day.day, 28))


def _whole_months(column: pa.ChunkedArray) -> np.ndarray:
    """dev_lag as int64, or a refusal of missing, fractional or non-numeric ages."""
    kind = column.type
    where = {"option": "cells", "column": "dev_lag"}
    if column.null_count:
        raise Refusal(
            "missing_value",
            f"dev_lag has {column.null_count} missing value(s)",
            rows=_positions(column.is_null()),
            **where,
        )
    if pa.types.is_integer(kind):
        return _arrow.to_numpy(column).astype(np.int64)
    if pa.types.is_floating(kind):
        months = _arrow.to_numpy(column).astype(float)
        fractional = ~np.isfinite(months) | (months != np.round(months))
        if fractional.any():
            raise Refusal(
                "invalid_age",
                "dev_lag must be whole numbers of months, got "
                f"{sorted(set(months[fractional].tolist()))[:5]}",
                rows=np.flatnonzero(fractional),
                **where,
            )
        return months.astype(np.int64)
    raise Refusal(
        "invalid_table", f"dev_lag must be a column of whole numbers of months, got {kind}", **where
    )


def _numbers(column: pa.ChunkedArray, name: str, why: str, *, option: str, cell_at) -> np.ndarray:
    """A numeric column as float64 with every value a finite number, or a refusal.

    ``cell_at(rows)`` turns row positions into the refused cells, so a missing
    or infinite amount is named by its cell.
    """
    kind = column.type
    where = {"option": option, "column": name}
    if not (pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind)):
        raise Refusal("invalid_table", f"{name} must be a numeric column, got {kind}", **where)
    if column.null_count:
        rows = _positions(column.is_null())
        raise Refusal(
            "missing_value",
            f"{name} has {column.null_count} null value(s), in {{cells}}; {why}",
            cells=cell_at(rows),
            **where,
        )
    amounts = _arrow.to_numpy(column.cast(pa.float64()))
    nan = np.flatnonzero(np.isnan(amounts))
    if nan.size:
        raise Refusal(
            "missing_value",
            f"{name} has {nan.size} NaN value(s), in {{cells}}; {why}",
            cells=cell_at(nan),
            **where,
        )
    # before any arithmetic: an infinite amount made the chain ladder refuse
    # without naming a cell, and Mack answer NaN or refuse for the wrong reason
    infinite = np.flatnonzero(np.isinf(amounts))
    if infinite.size:
        raise Refusal(
            "not_finite",
            f"{name} is infinite in {infinite.size} cell(s): {{cells}}. The methods need finite "
            "amounts",
            cells=cell_at(infinite),
            **where,
        )
    return amounts


def _premium(premium, origins: _Origins, method: str) -> dict[dt.date, Any]:
    """Premium keyed by period start, for ``kernels.fit_conventional_grid``.

    Its origins are read with the cells' rules and ``dev_grain_months``, so they
    may be written differently from the cells'. Every refusal of premium is
    made here, in the caller's own labels: repeated, missing and extra origins,
    and an amount that is not a positive finite number. An origin premium wrote
    is named as premium wrote it; one it lacks, as the cells wrote it.
    """
    if premium is None:
        raise Refusal(
            "invalid_option",
            f"{method} needs premium: a table with origin_period and premium, or a dict from "
            "origin period to amount",
            option="premium",
        )
    step = origins.step
    if isinstance(premium, Mapping):
        keys = list(premium)
        starts = []
        for key in keys:
            try:
                starts.append(_scalar_start(key, "premium origin", step))
            except Refusal as refusal:
                raise refusal._replace(option="premium") from None
        labels = keys
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
            table.column("origin_period"),
            "premium origin_period",
            step,
            cells=False,
            option="premium",
        )
        starts = [read.label_starts[i] for i in read.row_label]
        labels = [read.values[i] for i in read.row_label]
        amounts = _numbers(
            table.column("premium"),
            "premium",
            "every origin needs an amount",
            option="premium",
            cell_at=lambda rows: [read.cell(int(r)) for r in rows],
        ).tolist()
        repeat_what = "row"
    by_start: dict[dt.date, Any] = {}
    label_of: dict[dt.date, Any] = {}
    written: dict[dt.date, list[int]] = {}
    for index, (start, amount) in enumerate(zip(starts, amounts, strict=True)):
        by_start[start] = amount
        label_of.setdefault(start, labels[index])
        written.setdefault(start, []).append(index)
    repeated = [rows for rows in written.values() if len(rows) > 1]
    if repeated:
        raise Refusal(
            "duplicate",
            f"premium has more than one {repeat_what} for origin period(s) {{origins}}",
            option="premium",
            cells=[
                RefusedCell(labels[i], starts[i], None, _premium_amount(amounts[i]))
                for rows in repeated
                for i in rows
            ],
            quoted=True,
        )
    periods = sorted(set(origins.label_starts))
    absent = [start for start in periods if start not in by_start]
    if absent:
        raise Refusal(
            "origin_not_covered",
            "premium has no amount for origin(s) {origins}",
            option="premium",
            cells=[RefusedCell(origins.label_for_start(start), start) for start in absent],
        )
    extra = sorted(set(by_start) - set(periods))
    if extra:
        raise Refusal(
            "not_in_triangle",
            "premium has amounts for origin(s) {origins} that are not in the triangle; pass "
            "premium for the triangle's origins only",
            option="premium",
            cells=[RefusedCell(label_of[s], s, None, _premium_amount(by_start[s])) for s in extra],
        )
    for start in periods:
        _require_premium_amount(label_of[start], start, by_start[start])
    return by_start


def _premium_amount(amount) -> float | None:
    """A premium amount for a refusal: the number when it is a finite one."""
    if isinstance(amount, bool) or not isinstance(amount, numbers.Real):
        return None
    return _amount(amount)


def _require_premium_amount(label, start: dt.date, amount) -> None:
    """Refuse one origin's premium unless it is a positive finite number, naming it."""
    cell = [RefusedCell(label, start, None, _premium_amount(amount))]
    where = {"option": "premium", "cells": cell}
    # a number, not something float() happens to accept: the text "220" would convert
    if isinstance(amount, bool) or not isinstance(amount, numbers.Real) and amount is not None:
        raise Refusal(
            "invalid_option",
            "premium for origin {origins} is {given}, which is not a number",
            given=amount,
            **where,
        )
    if amount is None or math.isnan(amount):
        raise Refusal(
            "missing_value", "premium has no amount for origin {origins}; it is missing", **where
        )
    if math.isinf(amount):
        raise Refusal(
            "not_finite",
            "premium must be a positive finite number for every origin; it is infinite for "
            "{origins}",
            **where,
        )
    if amount <= 0:
        raise Refusal(
            "invalid_option",
            f"premium must be a positive finite number for every origin; it is {amount!r} for "
            "{origins}",
            **where,
        )
