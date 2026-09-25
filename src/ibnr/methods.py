"""Traditional reserving methods, one function per method, on one triangle at a time.

``from ibnr import methods``, then call the method you want by its name:

- :func:`chain_ladder`
- :func:`bornhuetter_ferguson`
- :func:`benktander` (Bornhuetter-Ferguson iterated; ``n_iters=1`` is Bornhuetter-Ferguson)
- :func:`cape_cod` (Gluck's generalized Cape Cod, with a trend; ``decay=1`` is the
  classic one)
- :func:`mack` (the chain ladder with Mack's standard errors; it takes the same
  development options, and ``average`` is Mack's alpha)

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

The ``development`` table has one row per observed development age, then,
with a tail, ``tail_rows`` rows for the ages after the last observed one. It
has the same columns for every method that carries them, in this order:

======================== ======== ======================================= =====================
column                   type     meaning                                 methods
======================== ======== ======================================= =====================
dev_lag                  int64    the age, in months                      all
factor                   float64  the factor to the next row; null on the all
                                  final row
cdf                      float64  the factor to ultimate, the tail's      all
                                  development included (without a tail,
                                  to the last observed age); on a tail's
                                  final row, the rest of the tail
pct_reported             float64  ``1 / cdf``                             all
source                   string   where the factor came from:             all
                                  ``"link_ratios"`` or ``"tail"``; null
                                  on the last age without a tail
curve_factor             float64  a fitted curve's factor at this row,    all
                                  for plotting fitted against selected;
                                  null on the final row and without a
                                  curve
in_tail_fit              bool     the curve was fitted through this       all
                                  age's factor; null at the last
                                  observed age, on the rows beyond it,
                                  and without a curve
n_selected               int64    link ratios selected at this age        all
unity_fallback           bool     no ratio was left and 1.0 was used      chain_ladder,
                                                                          bornhuetter_ferguson,
                                                                          benktander, cape_cod
extreme_trimming_skipped bool     ``preserve`` stopped ``drop_high`` and  all
                                  ``drop_low`` at this age
bounds_skipped           bool     ``preserve`` stopped ``drop_above`` and all
                                  ``drop_below`` at this age
sigma                    float64  Mack's sigma                            mack
std_err                  float64  the factor's standard error             mack
sigma_extrapolated       bool     sigma came from ``sigma_rule``: the     mack
                                  age kept fewer than two link ratios
======================== ======== ======================================= =====================

``n_selected`` and every column after it is null at the last observed age,
which has no link ratios after it, and on every row beyond it. On an
observed age whose ``source`` is ``"tail"`` (a tail attached before the last
age), ``n_selected`` and the three flags still describe the link ratios at
that age, but the factor is the tail's, not their average. A method carries
exactly the columns listed for it, whatever options it is given, tail or
none, so a service can read each table by name.

Every method takes a tail, the development still to come after the last
observed age: ``tail="constant"`` with ``tail_factor`` (such as 1.05), or a
curve fitted to the link factors, ``"exponential"``, ``"inverse_power"`` or
``"weibull"``. :func:`chain_ladder` describes the tail options. With a tail
every origin's ultimate includes it, the oldest origin's too, ``totals`` has
the tail factor in ``tail_factor`` (1.0 without a tail), and :func:`mack`'s
standard errors carry one more development step for it.

Importing this module loads numpy and pyarrow, and not ibis, pandas or scipy,
and none of the methods loads them when it runs, so a service that starts
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
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ibnr import _arrow
from ibnr.errors import Refusal, RefusedCell, _literal
from ibnr.kernels.conventional import ConventionalCandidate, _estimate_grid
from ibnr.kernels.grid import as_date, check_grid, grid_from_columns, month_end
from ibnr.kernels.links import REASONS as LINK_REASONS
from ibnr.kernels.links import is_all_history
from ibnr.kernels.mack import _require_mack_average, fit_mack_grid
from ibnr.kernels.odp_bootstrap import (
    NEGATIVE_INCREMENTS,
    POOL_REASONS,
    RESIDUAL_ADJUSTMENTS,
    RESIDUAL_POOLS,
    draw_runoff,
    prepare_runoff,
)
from ibnr.kernels.tail import TailFit, TailSpec, apply_tail

__all__ = [
    "BootstrapResult",
    "Refusal",
    "ReserveResult",
    "benktander",
    "bornhuetter_ferguson",
    "cape_cod",
    "chain_ladder",
    "mack",
    "odp_bootstrap",
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
        Bornhuetter-Ferguson, Benktander and Cape Cod add
        ``expected_loss_ratio``, the loss ratio applied to the origin's premium;
        Cape Cod also adds ``trended_loss_ratio`` (the loss ratio at the
        valuation date's level, chainladder-python's ``apriori_``) and
        ``trend_factor`` (1.0 at ``trend=0``); Mack adds ``mack_se`` and its
        two parts, ``parameter_se`` and ``process_se``. With a tail the
        ultimates, reserves and standard errors include it, the oldest
        origin's too.
        An origin whose latest cumulative is zero keeps 0 as its latest amount,
        so its chain-ladder ultimate is 0 and, under Mack with
        ``zero_cells="missing"``, its ``mack_se`` is 0 too; chainladder-python
        leaves both missing.
    development : pyarrow.Table
        One row per observed development age, and with a tail ``tail_rows``
        more for the ages after the last observed one: ``dev_lag`` (int64),
        ``factor`` (the factor from this row to the next, null on the final
        row), ``cdf`` (the factor to ultimate, float64) and ``pct_reported``
        (``1 / cdf``). Without a tail ``cdf`` runs to the last observed age, so
        ``pct_reported`` is 1.0 there by construction rather than by
        measurement. With one it includes the tail, and on the final row it is
        what is left of the tail after the rows shown, so the factors from any
        row times the final row's ``cdf`` are that row's ``cdf``. ``source``
        (string) is ``"link_ratios"`` for a factor averaged from link ratios
        and ``"tail"`` for one from the tail (an attached age, or a row beyond
        the triangle), null at the last age without a tail; ``curve_factor``
        (float64) is a fitted curve's factor at each row but the final one;
        ``in_tail_fit`` (bool) says whether the curve was fitted through that
        age's factor, null from the last observed age on. Every method adds
        ``n_selected`` (int64, the link ratios selected at that age; on an age
        whose ``source`` is ``"tail"`` the factor is the tail's, not their
        average), ``extreme_trimming_skipped`` and
        ``bounds_skipped`` (bool); the chain ladder, Bornhuetter-Ferguson,
        Benktander and Cape Cod add ``unity_fallback`` (bool); Mack adds
        ``sigma``, ``std_err`` (the factor's standard error) and
        ``sigma_extrapolated`` (bool), all null at the last observed age and
        beyond it. The module docstring has the table of every column and the
        methods that carry it.
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
        ``valuation_exclusion``, ``drop_above``, ``drop_below``, ``drop_low``
        or ``drop_high``: the first rule that left it out, in the order the
        rules run). Mack's is the chain ladder's for the same options; under
        ``zero_cells="observed"`` with no development option every observed
        link ratio is ``included``, a ratio out of a zero too (null, with its
        following amount in the volume sum), as R's ``MackChainLadder`` reads
        it. Every method returns this table; ``None`` is only for a result
        built by hand.
    totals : pyarrow.Table
        One row with ``latest``, ``ultimate`` and ``ibnr`` summed over the
        origins, and ``tail_factor`` (float64), the development beyond the
        last observed age: 1.0 without a tail, the constant itself for a
        constant tail attached at the last age, and what is left of it after
        the attached ages for one attached earlier (chainladder-python's
        ``tail_``). Mack adds ``mack_se``, ``parameter_se`` and ``process_se``
        for the total, which is not the sum of the origins' standard errors:
        the origins share the estimated factors; and ``tail_sigma``,
        ``tail_std_err`` (the tail step's sigma and the tail factor's standard
        error) and ``tail_position`` (where the tail sits on the link axis,
        link 1 developing from the first age), all float64 and null without a
        tail, ``tail_position`` also when ``tail_sigma`` and ``tail_std_err``
        were both given or the tail factor is exactly 1.
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
            # every method carries all four tables now; a result built by hand may not
            raise Refusal(
                "invalid_option",
                f"a {self.method} result has no {table} table",
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
    drop_high: bool | int = False,
    drop_low: bool | int = False,
    preserve: int = 1,
    drop_above: float | None = None,
    drop_below: float | None = None,
    exclude=(),
    exclude_valuations=(),
    trim_ties: str = "volume",
    unsupported_factor: str = "raise",
    exhausted_exclusions: str = "keep",
    zero_cells: str = "missing",
    tail: str | None = None,
    tail_factor: float | None = None,
    tail_decay: float | None = None,
    tail_attach_lag: int | None = None,
    tail_fit_lags: tuple | None = None,
    tail_steps: int | None = None,
    tail_rows: int | None = None,
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

    The development options, shared with :func:`bornhuetter_ferguson`,
    :func:`benktander` and :func:`cape_cod`, and with :func:`mack` except for
    ``unsupported_factor`` and the median. The rules that leave link ratios
    out run in this order, each on the ratios the ones before it left:
    ``zero_cells``, ``history_periods``, ``exclude``, ``exclude_valuations``,
    ``drop_above``/``drop_below``, then ``drop_high``/``drop_low``. Each ratio
    left out is listed in ``link_ratios`` with the first rule that removed it.

    - ``dev_grain_months``: months per development step, 12 for an annual
      triangle, 3 for a quarterly one. Origin periods must be the same length.
    - ``average``: how link ratios become a factor. ``"volume"`` (the default)
      divides the sum of the later cumulatives by the sum of the earlier ones;
      ``"simple"`` is the plain mean of the ratios; ``"regression"`` is least
      squares through the origin, ``sum(previous * following) /
      sum(previous ** 2)``; ``"median"`` is their median. (In Mack's terms each
      ratio is weighted by ``previous ** alpha``, with alpha 1 for volume, 0 for
      simple and 2 for regression.)
    - ``history_periods``: use only the latest this many link ratios at each age
      (``None``, the default, uses them all). Under ``zero_cells="missing"`` a
      ratio left out for a zero cell still counts as one of them, as in
      chainladder-python's ``n_periods``, so the window then holds fewer.
    - ``exclude``: link ratios to leave out, as ``(origin_period, dev_lag)``
      pairs, where ``dev_lag`` is the age the ratio develops FROM in whole
      months; ``(2010, 12)`` leaves out the 2010 ratio from 12 to 24 months.
      The origin may be written in any of the ways ``origin_period`` may, not
      necessarily the way the cells write it. A pair that names no link ratio
      of the triangle is refused.
    - ``exclude_valuations``: whole diagonals to leave out. Each valuation
      leaves out every link ratio that develops INTO it, so excluding 2020
      removes the development that happened during 2020 (a year distorted by
      COVID, say), and the latest valuation may be excluded too. Write each as
      the evaluation date, the last day of a development period
      (``"2020-12-31"`` or a date), or as the one period it ends: a year
      (``2020`` or ``"2020"``) on an annual triangle, a quarter (``"2020Q4"``)
      on a quarterly one, a month (``"2020-12"``) on a monthly one. A valuation
      no link ratio develops into (the first diagonal, or a date after the
      latest one) is refused. chainladder-python's ``drop_valuation`` names
      the EARLIER end of the link ratio instead, so its ``drop_valuation=2019``
      is ``exclude_valuations=[2020]`` here.
    - ``drop_above`` / ``drop_below``: leave out every link ratio strictly
      above ``drop_above`` or strictly below ``drop_below``; a ratio equal to a
      bound is kept (chainladder-python leaves it out).
    - ``drop_high`` / ``drop_low``: leave out this many of the highest and the
      lowest link ratios at each age. ``True`` means 1 and ``False`` 0.
    - ``trim_ties``: which of two equal link ratios ``drop_high``/``drop_low``
      leave out. ``"volume"`` (the default here, chainladder-python's rule)
      ranks equal ratios by the earlier cumulative: ``drop_high`` leaves out the
      one with the larger amount and ``drop_low`` the smaller; equal amounts
      too go by origin, ``drop_high`` the newer and ``drop_low`` the older.
      ``"origin"`` ranks equal ratios by origin alone, the rule of ibnr 0.7.2
      and the default of ``kernels.ConventionalCandidate``.
    - ``preserve``: the fewest link ratios the bounds may leave at an age, and
      the fewest the trims may leave (1 by default). Each rule is all or
      nothing at an age: if it would leave fewer, ``exhausted_exclusions``
      decides.
    - ``unsupported_factor``: what to do at an age where no link ratio is left
      to average: ``"raise"`` (the default) refuses; ``"unity"`` uses a factor
      of 1.0 and marks the age in ``development.unity_fallback``.
    - ``exhausted_exclusions``: what to do when the bounds, or the trims, would
      leave fewer than ``preserve`` ratios at an age. ``"keep"`` (the default
      here) does not apply that rule at that age and marks it in
      ``development.bounds_skipped`` or ``development.extreme_trimming_skipped``;
      ``"raise"`` refuses. The default differs from
      ``kernels.ConventionalCandidate``'s ``"raise"`` because on a complete
      triangle the last age has a single link ratio, so ``drop_high=True``
      would always be refused; chainladder-python users expect it to work, and
      the result records the skip.
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

    The tail options, shared with :func:`bornhuetter_ferguson`,
    :func:`benktander`, :func:`cape_cod` and :func:`mack`. A tail is the
    development still to come after the last observed age. It is applied to
    the factors the development options produced, and every origin's
    development to ultimate then includes it, the oldest origin's too, so
    Bornhuetter-Ferguson, Benktander and Cape Cod read it through
    ``pct_reported``. Every option but ``tail`` defaults to ``None``, and one
    given without ``tail`` is refused rather than ignored.

    - ``tail``: ``None`` (the default, no tail), ``"constant"``, or a curve
      fitted to the link factors by least squares: ``"exponential"`` (``f - 1 =
      exp(a + b t)`` at link ``t``, link 1 developing from the first age),
      ``"inverse_power"`` (``f - 1 = exp(a) t ** b``) or ``"weibull"`` (``f =
      1 / (1 - exp(-exp(a) t ** b))``), as chainladder-python's ``TailCurve``.
      A curve must decay with age, and an inverse power curve's slope must be
      below -1, since between -1 and 0 the product of its factors never
      converges; otherwise it is refused.
    - ``tail_factor``: a constant tail's development from ``tail_attach_lag``
      to ultimate, such as 1.05; needed for ``"constant"``. Below 1 is allowed
      and lowers the ultimates.
    - ``tail_decay``: from 0 to 1 (0.5 by default), how a constant tail is
      spread over the steps it covers: each step's development above 1 is
      ``tail_decay`` times the step's before it, and the last step holds what
      is left, so the steps multiply to ``tail_factor``. With the tail attached at the
      last observed age it changes only the rows shown, never an ultimate.
    - ``tail_attach_lag``: the age, in months, of the first link the tail
      replaces. The default is the last observed age, where no link is
      replaced; an earlier age replaces the factors from that age on with the
      curve's (or the constant's steps), and ``development.source`` says which.
    - ``tail_fit_lags``: curves only, ``(first, last)``: the ages, in months,
      that the first and the last link fitted develop from, both included;
      ``None`` at either end is the edge of the triangle. Only factors above
      1.00001 go through the line, and at least two must, or the tail is
      refused (``in_tail_fit`` shows which did).
    - ``tail_steps``: curves only, how many development steps past the last
      observed age the curve is extrapolated, 1 to 10,000 (100 by default, as
      chainladder-python's ``extrap_periods``). It is a count of steps of the
      triangle's own grain, not months. An inverse power tail still grows
      noticeably at 100 steps.
    - ``tail_rows``: how many rows beyond the last observed age ``development``
      shows, one step each (one year's by default: 1 on an annual triangle, 4
      on a quarterly one); the final row's ``cdf`` holds the rest of the tail.
      It never changes an ultimate. A curve cannot show more rows than
      ``tail_steps``.

    ``docs/coming-from-chainladder.md`` maps chainladder-python's
    ``TailConstant`` and ``TailCurve`` options onto these, with the cases
    chainladder-python rounds, ignores or answers as no tail that are refused
    here.

    Returns a :class:`ReserveResult`. Without a tail each origin is projected
    to the last observed development age. Input it will not answer is refused
    with :class:`Refusal`, whose reason codes the module docstring describes.
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
            preserve=preserve,
            drop_above=drop_above,
            drop_below=drop_below,
            exclude=exclude,
            exclude_valuations=exclude_valuations,
            trim_ties=trim_ties,
            unsupported_factor=unsupported_factor,
            exhausted_exclusions=exhausted_exclusions,
            zero_cells=zero_cells,
            tail=_tail_spec(
                tail,
                tail_factor=tail_factor,
                tail_decay=tail_decay,
                tail_attach_lag=tail_attach_lag,
                tail_fit_lags=tail_fit_lags,
                tail_steps=tail_steps,
                tail_rows=tail_rows,
            ),
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
    drop_high: bool | int = False,
    drop_low: bool | int = False,
    preserve: int = 1,
    drop_above: float | None = None,
    drop_below: float | None = None,
    exclude=(),
    exclude_valuations=(),
    trim_ties: str = "volume",
    unsupported_factor: str = "raise",
    exhausted_exclusions: str = "keep",
    zero_cells: str = "missing",
    tail: str | None = None,
    tail_factor: float | None = None,
    tail_decay: float | None = None,
    tail_attach_lag: int | None = None,
    tail_fit_lags: tuple | None = None,
    tail_steps: int | None = None,
    tail_rows: int | None = None,
) -> ReserveResult:
    """Bornhuetter-Ferguson: the unreported share of an a priori ultimate.

    Each origin's ultimate is its latest cumulative loss plus
    ``premium * expected_loss_ratio * (1 - pct_reported)``, where
    ``pct_reported`` comes from the chain-ladder development pattern.

    ``cells``, the development options and the tail options are as in
    :func:`chain_ladder`; with a tail, ``pct_reported`` includes it.

    ``premium`` is keyed by origin period, never by position: either a table
    with columns ``origin_period`` and ``premium`` (a polars DataFrame or a
    pyarrow Table, for example), or a dict from origin period to amount. The
    origin periods may be written in any of the ways the cells' may, and need
    not be written the same way as the cells (integer years in the cells and
    year-end dates in premium, for example). It needs exactly one positive
    amount for each origin of the triangle and no others.

    ``expected_loss_ratio`` is the a priori loss ratio, one number for every
    origin, applied to premium. :func:`benktander` iterates this method.
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
            preserve=preserve,
            drop_above=drop_above,
            drop_below=drop_below,
            exclude=exclude,
            exclude_valuations=exclude_valuations,
            trim_ties=trim_ties,
            unsupported_factor=unsupported_factor,
            exhausted_exclusions=exhausted_exclusions,
            zero_cells=zero_cells,
            tail=_tail_spec(
                tail,
                tail_factor=tail_factor,
                tail_decay=tail_decay,
                tail_attach_lag=tail_attach_lag,
                tail_fit_lags=tail_fit_lags,
                tail_steps=tail_steps,
                tail_rows=tail_rows,
            ),
        )
        return _conventional_result(
            "bornhuetter_ferguson", grid, origins, candidate, premium=premium
        )


def benktander(
    cells,
    *,
    premium,
    expected_loss_ratio: float,
    n_iters: int = 1,
    dev_grain_months: int = 12,
    average: str = "volume",
    history_periods: int | None = None,
    drop_high: bool | int = False,
    drop_low: bool | int = False,
    preserve: int = 1,
    drop_above: float | None = None,
    drop_below: float | None = None,
    exclude=(),
    exclude_valuations=(),
    trim_ties: str = "volume",
    unsupported_factor: str = "raise",
    exhausted_exclusions: str = "keep",
    zero_cells: str = "missing",
    tail: str | None = None,
    tail_factor: float | None = None,
    tail_decay: float | None = None,
    tail_attach_lag: int | None = None,
    tail_fit_lags: tuple | None = None,
    tail_steps: int | None = None,
    tail_rows: int | None = None,
) -> ReserveResult:
    """Benktander: Bornhuetter-Ferguson repeated, each time from the last ultimate.

    Mack (2000), "Credible claims reserves: the Benktander method". With
    ``q = 1 - pct_reported`` at each origin's latest age, ``L`` its latest
    cumulative loss and ``E = premium * expected_loss_ratio``, the ultimate
    after ``n`` iterations is ``U_n = L + q * U_(n-1)``, starting from
    ``U_0 = E``. ``n_iters=1`` is :func:`bornhuetter_ferguson` exactly, and as
    ``n_iters`` grows the ultimate moves to the chain ladder's. ``n_iters`` must
    be a whole number from 1 to 10,000; 0 would be the expected loss method,
    which ignores the reported losses, and each iteration is one more pass of a
    loop, so the upper limit bounds the time one call can take.

    ``cells``, the development options and the tail options are as in
    :func:`chain_ladder`, and ``premium`` and ``expected_loss_ratio`` as in
    :func:`bornhuetter_ferguson`. The result has the same tables and columns
    as Bornhuetter-Ferguson's.
    """
    with _CallersTerms("benktander") as terms:
        grid, origins = terms.read(cells, dev_grain_months)
        candidate = _candidate(
            "bf",
            origins,
            expected_loss_ratio=expected_loss_ratio,
            n_iters=n_iters,
            average=average,
            history_periods=history_periods,
            drop_high=drop_high,
            drop_low=drop_low,
            preserve=preserve,
            drop_above=drop_above,
            drop_below=drop_below,
            exclude=exclude,
            exclude_valuations=exclude_valuations,
            trim_ties=trim_ties,
            unsupported_factor=unsupported_factor,
            exhausted_exclusions=exhausted_exclusions,
            zero_cells=zero_cells,
            tail=_tail_spec(
                tail,
                tail_factor=tail_factor,
                tail_decay=tail_decay,
                tail_attach_lag=tail_attach_lag,
                tail_fit_lags=tail_fit_lags,
                tail_steps=tail_steps,
                tail_rows=tail_rows,
            ),
        )
        return _conventional_result("benktander", grid, origins, candidate, premium=premium)


def cape_cod(
    cells,
    *,
    premium,
    decay: float = 1.0,
    trend: float = 0.0,
    n_iters: int = 1,
    dev_grain_months: int = 12,
    average: str = "volume",
    history_periods: int | None = None,
    drop_high: bool | int = False,
    drop_low: bool | int = False,
    preserve: int = 1,
    drop_above: float | None = None,
    drop_below: float | None = None,
    exclude=(),
    exclude_valuations=(),
    trim_ties: str = "volume",
    unsupported_factor: str = "raise",
    exhausted_exclusions: str = "keep",
    zero_cells: str = "missing",
    tail: str | None = None,
    tail_factor: float | None = None,
    tail_decay: float | None = None,
    tail_attach_lag: int | None = None,
    tail_fit_lags: tuple | None = None,
    tail_steps: int | None = None,
    tail_rows: int | None = None,
) -> ReserveResult:
    """Cape Cod: Bornhuetter-Ferguson with the loss ratio estimated from the triangle.

    This is Gluck's generalized Cape Cod. Each origin's expected loss ratio is
    a weighted ratio of reported losses to "used-up" premium (premium times
    ``pct_reported``) over all origins, where an origin ``k`` periods away is
    weighted by ``decay ** k``. ``decay=1`` (the default) weights every origin
    equally, which is the classic Cape Cod and one loss ratio for the whole
    triangle; ``decay=0`` uses each origin's own experience only, which is the
    chain ladder.

    ``trend`` is an annual rate, such as 0.05 for 5% a year (0, the default,
    compares amounts as they are). Each origin's losses are first brought to
    the valuation date's level, multiplied by ``trend_factor = (1 + trend) **
    (m / 12)``, where ``m`` is the whole months from the end of the origin
    period to ``as_of``. The weighted ratio of those to used-up premium is the
    ``trended_loss_ratio``, the loss ratio at the valuation date's level
    (chainladder-python's ``apriori_``). Each origin's ``expected_loss_ratio``,
    applied to its own premium, is that divided by its own ``trend_factor``
    (chainladder-python's ``detrended_apriori_``). With ``decay=1`` the
    trended loss ratio is one number for the whole triangle; with ``decay=0``
    the trend cancels and the answer is the chain ladder's.

    ``n_iters`` iterates the result as :func:`benktander` does, with each
    origin's ``premium * expected_loss_ratio`` as the first a priori ultimate;
    1 (the default) is Cape Cod itself, and 0 and counts above 10,000 are
    refused.

    ``cells``, the development options and the tail options are as in
    :func:`chain_ladder` (with a tail, the used-up premium includes it), and
    ``premium`` as in :func:`bornhuetter_ferguson`. The loss ratios are in
    ``origins.expected_loss_ratio`` and ``origins.trended_loss_ratio``, and
    ``origins.trend_factor`` is each origin's factor (1.0 at trend 0).
    """
    with _CallersTerms("cape_cod") as terms:
        grid, origins = terms.read(cells, dev_grain_months)
        candidate = _candidate(
            "gcc",
            origins,
            decay=decay,
            trend=trend,
            n_iters=n_iters,
            average=average,
            history_periods=history_periods,
            drop_high=drop_high,
            drop_low=drop_low,
            preserve=preserve,
            drop_above=drop_above,
            drop_below=drop_below,
            exclude=exclude,
            exclude_valuations=exclude_valuations,
            trim_ties=trim_ties,
            unsupported_factor=unsupported_factor,
            exhausted_exclusions=exhausted_exclusions,
            zero_cells=zero_cells,
            tail=_tail_spec(
                tail,
                tail_factor=tail_factor,
                tail_decay=tail_decay,
                tail_attach_lag=tail_attach_lag,
                tail_fit_lags=tail_fit_lags,
                tail_steps=tail_steps,
                tail_rows=tail_rows,
            ),
        )
        return _conventional_result("cape_cod", grid, origins, candidate, premium=premium)


def mack(
    cells,
    *,
    dev_grain_months: int = 12,
    sigma_rule: str = "log_linear",
    zero_cells: str = "missing",
    average: str = "volume",
    history_periods: int | None = None,
    drop_high: bool | int = False,
    drop_low: bool | int = False,
    preserve: int = 1,
    drop_above: float | None = None,
    drop_below: float | None = None,
    exclude=(),
    exclude_valuations=(),
    trim_ties: str = "volume",
    exhausted_exclusions: str = "keep",
    tail: str | None = None,
    tail_factor: float | None = None,
    tail_decay: float | None = None,
    tail_attach_lag: int | None = None,
    tail_fit_lags: tuple | None = None,
    tail_steps: int | None = None,
    tail_rows: int | None = None,
    tail_sigma: float | None = None,
    tail_std_err: float | None = None,
) -> ReserveResult:
    """Mack's chain ladder: the chain-ladder ultimate and its standard error.

    Mack (1993), distribution-free, and Mack (1999) for the development
    options. ``cells`` is as in :func:`chain_ladder`, and so are the development
    options, which choose the link ratios exactly as they do there: the same
    options give the same factors, bit for bit, and the same ``link_ratios``
    table. There is no ``unsupported_factor``: an age the options leave with no
    link ratio is refused, because a factor of 1.0 there would be chosen, not
    estimated, and Mack's formulas give it no variance.

    ``average`` is ``"volume"`` (the default), ``"simple"`` or
    ``"regression"``. In Mack's terms each link ratio is weighted by the amount
    it starts from to the power alpha, 1, 0 or 2, and the variance of one
    development step is ``sigma ** 2 * amount ** (2 - alpha)``. ``"median"`` is
    refused: a median is not a weighted mean of the ratios, so Mack's variance
    does not exist for it (``"geometric"`` is refused for the same reason).

    The options choose which link ratios estimate the factors and the sigmas.
    The development still to come from each origin's latest amount keeps its
    full variance whatever was left out; chainladder-python drops the first
    year of it whenever a drop or a bound is set, and this function does not
    (``docs/coming-from-chainladder.md`` has the numbers). ``drop_high``,
    ``drop_low``, ``drop_above`` and ``drop_below`` choose ratios after looking
    at them, which Mack's formulas do not allow for, so with them the standard
    errors are approximate and tend to be low; R's ``MackChainLadder`` and
    chainladder-python apply the formulas the same way. ``history_periods=1``
    is refused (one ratio at every age leaves no sigma to estimate), and so are
    any options that leave at most one link ratio at every age. The one-year
    claims development result in ``ibnr.kernels`` needs a fit with no
    development options and no tail.

    ``zero_cells`` is what a cumulative of exactly zero is. ``"missing"`` (the
    default here, as in chainladder-python) leaves out every link ratio with a
    zero at either end, from the factors and the sigmas alike. An origin whose
    latest cumulative is zero keeps 0 as its latest amount, so its ultimate is
    0 and its standard errors are 0, the limit of Mack's formula as that amount
    goes to zero; chainladder-python leaves that origin's ultimate and standard
    error missing, and its total standard error equals the one here. Under
    ``average="regression"`` such an origin is refused: the variance of a step
    does not shrink with the amount, so it would get a mean of 0 and a positive
    standard error. ``"observed"`` keeps zeros as data, as R's
    ``MackChainLadder`` and ``kernels.fit_mack`` (whose default it is) do: the
    factor uses every link ratio, and a zero latest cumulative on a
    still-developing origin is refused, because Mack's variance divides by it.
    Under ``"observed"`` a link ratio out of a zero has no value, so any
    development option on a triangle with one is refused.

    ``sigma_rule`` picks how the variance is filled in at a development age
    with too few link ratios to estimate it, usually the last one:
    ``"log_linear"`` (the default, as in chainladder-python) extends a straight
    line through the logarithms of the other sigmas; ``"mack"`` is Mack's own
    1993 rule, from the two ages before. The ultimates do not depend on it; the
    standard errors do. (``kernels.fit_mack`` keeps ``"mack"`` as its default,
    so published numbers made with it do not move.)

    The tail options are as in :func:`chain_ladder`, and the tail is one more
    development step, taken by every origin, the fully developed one too (Mack
    1999; R's ``MackChainLadder(tail=, tail.se=, tail.sigma=)``). Its sigma
    and the tail factor's standard error are read off straight lines through
    the logarithms of the sigmas and of the factors' standard errors, at the
    age where a straight line through ``log(factor - 1)`` reaches the tail
    (``totals.tail_position``); the lines go through the factors above 1 and
    the positive values only, as R's do. ``tail_sigma`` and ``tail_std_err``
    (R's ``tail.sigma`` and ``tail.se``, zero or more) give either instead.
    The tail step's process variance is divided by the origin's amount at the
    last observed age to the power alpha, as every other step's is. Refused: a
    tail attached before the last observed age (no formula gives a curve
    factor's standard error where a link ratio's was); and, unless both
    ``tail_sigma`` and ``tail_std_err`` are given, a tail factor below 1, too
    few factors above 1 or positive sigmas to draw the lines through, and a
    tail larger than the line through the factors gives even at the first
    link, where the sigma would be read backwards past the data.
    chainladder-python's lines keep a left-out point's age in their sums, so
    its tail variance differs from this one whenever a factor is at or below 1
    or a sigma is 0 (``docs/coming-from-chainladder.md`` has the numbers).

    ``origins`` and ``totals`` carry ``mack_se`` and its two parts:
    ``parameter_se``, from estimating the factors, and ``process_se``, from the
    randomness of future development, where ``mack_se ** 2 = parameter_se ** 2
    + process_se ** 2``. ``development`` adds ``sigma``, ``std_err`` (the
    factor's standard error) and ``sigma_extrapolated`` (the sigma came from
    ``sigma_rule`` because the age kept fewer than two link ratios) to the
    chain ladder's columns, and ``link_ratios`` lists every observed link ratio
    and whether it was used, as for the chain ladder. Under
    ``zero_cells="observed"`` the standard errors need every still-developing
    origin's latest cumulative loss to be positive, and are refused otherwise.
    They also need at least one development age with two or more link ratios,
    since a sigma is estimated from the spread of link ratios: a triangle with
    at most one at every age (two origins, for example) is refused rather than
    given standard errors of 0. Input it will not answer is refused with
    :class:`Refusal`, and so is a triangle whose amounts are so large that a
    standard error is not a finite number, or so small that its square reads as
    0, and one where a factor is 0 (every origin at an age closing at zero),
    since every ultimate after it would be 0.
    """
    with _CallersTerms("mack") as terms:
        grid, labels = terms.read(cells, dev_grain_months)
        _require_mack_average(average)
        wrapped = _candidate(
            "cl",
            labels,
            average=average,
            history_periods=history_periods,
            drop_high=drop_high,
            drop_low=drop_low,
            preserve=preserve,
            drop_above=drop_above,
            drop_below=drop_below,
            exclude=exclude,
            exclude_valuations=exclude_valuations,
            trim_ties=trim_ties,
            exhausted_exclusions=exhausted_exclusions,
            zero_cells=zero_cells,
        )
        spec = _tail_spec(
            tail,
            tail_factor=tail_factor,
            tail_decay=tail_decay,
            tail_attach_lag=tail_attach_lag,
            tail_fit_lags=tail_fit_lags,
            tail_steps=tail_steps,
            tail_rows=tail_rows,
            tail_sigma=tail_sigma,
            tail_std_err=tail_std_err,
        )
        return _mack(grid, labels, wrapped, sigma_rule=sigma_rule, tail=spec)


def _mack(
    grid, labels: _Origins, wrapped: _Candidate, *, sigma_rule: str, tail: TailSpec | None
) -> ReserveResult:
    _, as_of = check_grid(grid)
    step = grid["dev_grain_months"]
    if grid["n_d"] < 2:
        raise Refusal(
            "variance_not_estimable",
            "mack needs at least two development ages; this triangle has one. chain_ladder "
            "gives the latest amounts as the ultimates",
            option="cells",
        )
    _require_valuations_in_triangle(grid, wrapped.valuations)
    _require_exclusions_in_triangle(grid, wrapped.exclusions)
    candidate = wrapped.kernel
    rules = candidate.link_rules
    zero_cells, average = rules.zero_cells, candidate.average
    # Under "missing" every fit reads its link ratios through kernels.links, as the
    # chain ladder does. Under "observed" with no option, 0.7.2's estimator keeps
    # R's reading of a link ratio out of a zero (it enters the volume factor).
    plain = average == "volume" and is_all_history(rules)
    links = None if zero_cells == "observed" and plain else rules
    if links is not None:
        # the triangle itself has too few link ratios for any sigma: refused here in
        # the same words as below, before the kernel would blame the options
        cum, mask = grid["cum"], grid["obs_mask"]
        pair = mask[:, :-1] & mask[:, 1:] & (cum[:, :-1] > 0)
        if zero_cells == "missing":
            pair &= cum[:, 1:] != 0
        if (pair.sum(axis=0) < 2).all():
            raise _no_sigma_anywhere(grid["n_d"], step)
    # Amounts near the largest or the smallest double are finite but Mack's sums,
    # ratios and squares are not. numpy says so with a warning, which is silenced
    # here and below because the kernel and _require_finite refuse such an answer
    # by name before anything is returned.
    with np.errstate(all="ignore"):
        fit = fit_mack_grid(
            grid,
            sigma_rule=sigma_rule,
            zero_cells=zero_cells,
            average=average,
            links=links,
            tail=tail,
        )
    zero_factor = np.flatnonzero(fit.f == 0)
    if zero_factor.size:
        # Only the last age can get here with every link ratio 0 (the age after
        # would have nothing to start from); any age can when a ratio is below
        # the smallest double.
        raise Refusal(
            "no_link_ratio",
            "the link ratios give a factor of 0.0 {links}, so every ultimate after that age "
            "would be 0, and Mack's standard errors divide by the factor. Every origin with a "
            "link ratio there closes at zero, or its amount is too small against the one before "
            "for the ratio to be a number above 0. chain_ladder with unsupported_factor='unity' "
            "uses a factor of 1.0 at that age instead",
            option="cells",
            links=[((j + 1) * step, (j + 2) * step) for j in zero_factor],
        )
    if (fit.n_pos < 2).all():
        raise _no_sigma_anywhere(fit.n_d, step)
    # Negative cells never get here (the cells are checked first), so the one
    # latest amount msep_runoff refuses is a zero under "observed". Its own
    # message names MackFit attributes a ReserveResult does not have. With a tail
    # every origin is still developing: the tail step divides by its amount too.
    zero_latest = np.flatnonzero(fit._developing & (fit.latest == 0))
    if zero_cells == "observed" and zero_latest.size:
        developing = " (with a tail, every origin)" if tail is not None else ""
        raise Refusal(
            "variance_not_estimable",
            "mack cannot give standard errors under zero_cells='observed' while a "
            f"still-developing origin's{developing} latest cumulative is zero: {{origins}}. "
            "Mack's variance divides by that amount. zero_cells='missing' (this function's "
            "default) gives "
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
    with np.errstate(all="ignore"):
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
        # Mack refuses a tail attached before the last age, so the tail's factors
        # are fit.f and only the rows beyond the triangle are the tail's
        shown = None if tail is None else apply_tail(fit.f, step, tail)
        pattern = _pattern_numbers(fit.f, fit.n_d, shown)
        tailed = tail is not None
        total = {
            **_sums(latest, ultimate),
            "mack_se": _arrow.float64([np.sqrt(risk["msep_total"])]),
            "parameter_se": _arrow.float64([np.sqrt(risk["parameter_total"])]),
            "process_se": _arrow.float64([np.sqrt(risk["process_total"])]),
            "tail_factor": _arrow.float64([fit.tail_factor]),
            "tail_sigma": _arrow.float64([np.sqrt(fit.tail_sigma2)], mask=[not tailed]),
            "tail_std_err": _arrow.float64([np.sqrt(fit.tail_se2)], mask=[not tailed]),
            "tail_position": _arrow.float64(
                [fit.tail_position or 0.0], mask=[fit.tail_position is None]
            ),
        }
    _require_finite(
        fit.origin_periods, per_origin, np.concatenate([sigma, std_err, *pattern]), total
    )
    _require_no_underflow(fit, risk["msep"])
    origins = pa.table(
        {
            "origin": labels.labels_for(fit.origin_periods),
            "origin_period": _arrow.date32(fit.origin_periods),
            "latest_dev_lag": _arrow.int64((fit.latest_dev + 1) * step),
            "latest": _arrow.float64(latest),
            **{name: _arrow.float64(values) for name, values in per_origin.items()},
        }
    )
    selection = fit.selection
    if selection is None:
        # 0.7.2's estimator: no rule was there to skip
        trimming_skipped = bounds_skipped = np.zeros(fit.n_d - 1, dtype=bool)
    else:
        trimming_skipped, bounds_skipped = selection.trimming_skipped, selection.bounds_skipped
    rows = 0 if shown is None else shown.rows
    development = pa.table(
        {
            **_pattern(pattern, step),
            **_tail_columns(fit.n_d, shown),
            "n_selected": _per_link(fit.n_obs.tolist(), pa.int64(), rows),
            "extreme_trimming_skipped": _per_link(trimming_skipped.tolist(), pa.bool_(), rows),
            "bounds_skipped": _per_link(bounds_skipped.tolist(), pa.bool_(), rows),
            "sigma": _per_link(sigma, pa.float64(), rows),
            "std_err": _per_link(std_err, pa.float64(), rows),
            "sigma_extrapolated": _per_link((fit.n_pos < 2).tolist(), pa.bool_(), rows),
        }
    )
    link_ratios = _link_ratios(labels, _mack_link_rows(fit))
    return ReserveResult("mack", as_of, step, origins, development, link_ratios, pa.table(total))


def _no_sigma_anywhere(n_d: int, step: int) -> Refusal:
    return Refusal(
        "variance_not_estimable",
        "mack needs at least one development age with two or more link ratios to estimate "
        "Mack's sigma; this triangle has at most one at every age ({links}), so every "
        "sigma would be set to 0 and the standard errors would read as no uncertainty at "
        "all. chain_ladder gives the same ultimates without standard errors",
        option="cells",
        links=[((j + 1) * step, (j + 2) * step) for j in range(n_d - 1)],
    )


def _mack_link_rows(fit) -> dict[str, list]:
    """Every observed link ratio of a Mack fit, in the chain ladder's row order:
    by development age, then by origin."""
    rows: dict[str, list] = {name: [] for name in _LINK_COLUMNS}
    selection, cum, step = fit.selection, fit.cum, fit.dev_grain_months
    observed = fit.obs_mask[:, :-1] & fit.obs_mask[:, 1:]
    for j in range(fit.n_d - 1):
        for i in np.flatnonzero(observed[:, j]):
            previous, following = cum[i, j], cum[i, j + 1]
            rows["origin_period"].append(fit.origin_periods[i])
            rows["from_dev_lag"].append((j + 1) * step)
            rows["previous"].append(previous)
            rows["following"].append(following)
            if selection is None:
                # 0.7.2's estimator: every observed link ratio enters the factor, a
                # ratio out of a zero included (its following amount adds to the sum)
                rows["ratio"].append(following / previous if previous > 0 else np.nan)
                rows["included"].append(True)
                rows["reason"].append("included")
            else:
                rows["ratio"].append(selection.ratio[i, j])
                rows["included"].append(bool(selection.used[i, j]))
                rows["reason"].append(LINK_REASONS[selection.reason[i, j]])
    return rows


#: The columns of a link_ratios table, before the caller's labels are added.
_LINK_COLUMNS = (
    "origin_period",
    "from_dev_lag",
    "previous",
    "following",
    "ratio",
    "included",
    "reason",
)


def _link_ratios(origins: _Origins, rows: dict[str, list]) -> pa.Table:
    """The link_ratios table from its columns, one row per observed link ratio.

    The one builder of the table, for the chain ladder's family and for Mack,
    so the same selection gives equal tables."""
    ratio = np.array(rows["ratio"], dtype=float)
    return pa.table(
        {
            "origin": origins.labels_for(rows["origin_period"]),
            "origin_period": _arrow.date32(rows["origin_period"]),
            "from_dev_lag": _arrow.int64(rows["from_dev_lag"]),
            "previous": _arrow.float64(rows["previous"]),
            "following": _arrow.float64(rows["following"]),
            # an undefined ratio (from a zero cumulative) is missing, not a number
            "ratio": _arrow.float64(ratio, mask=np.isnan(ratio)),
            "included": _arrow.bool_(rows["included"]),
            "reason": _arrow.string(rows["reason"]),
        }
    )


def _require_no_underflow(fit, msep: np.ndarray) -> None:
    """Refuse Mack's standard errors when a square fell below the smallest double.

    Amounts near the smallest double (about 1e-308) have squares of exactly 0.
    Mack's sigma and msep are sums of such squares, so they come out 0 and the
    standard errors would read as no uncertainty at all. An msep or a sigma is 0
    honestly only when every link ratio at the ages involved is the same (or the
    origin's latest amount is 0 under ``zero_cells="missing"``); any other 0 is
    an underflow.
    """
    cum, mask = fit.cum, fit.obs_mask
    underflowed = []
    for j in range(fit.n_d - 1):
        # the link ratios behind sigma at this age, as the kernel picks them
        if fit.selection is not None:
            pair = fit.selection.used[:, j]
        else:
            pair = mask[:, j] & mask[:, j + 1] & (cum[:, j] > 0)
            if fit.zero_cells == "missing":
                pair &= cum[:, j + 1] != 0
        with np.errstate(all="ignore"):
            ratios = cum[pair, j + 1] / cum[pair, j]
        if fit.sigma2[j] == 0 and ratios.size > 1 and np.ptp(ratios) > 0:
            underflowed.append(j)
    open_ = fit._developing & (fit.latest > 0)
    tail_noise = fit.tail_sigma2 > 0 or fit.tail_se2 > 0
    noisy = np.array(
        [(fit.sigma2[int(k) :] > 0).any() or tail_noise for k in fit.latest_dev], dtype=bool
    )
    zero_msep = np.flatnonzero(open_ & noisy & (msep == 0))
    if underflowed or zero_msep.size:
        raise Refusal(
            "result_not_finite",
            "the amounts are too small for Mack's standard errors: their squares fall below "
            "the smallest double and read as 0, so the standard errors would say there is no "
            "uncertainty at all. Scale them up (multiply by a power of ten) and scale the "
            "answer back",
            option="cells",
            cells=[RefusedCell(None, fit.origin_periods[i]) for i in zero_msep],
            links=[
                ((j + 1) * fit.dev_grain_months, (j + 2) * fit.dev_grain_months)
                for j in underflowed
            ],
        )


def _require_finite(periods, per_origin: dict, others, total: dict) -> None:
    """Refuse an answer with a number that is not finite, naming the origins.

    The inputs were checked finite, so such a number comes from amounts so large
    that a sum, a square or a product passes the largest double (or so far
    apart, or so small, that a factor's product is 0). A result never carries
    NaN or infinity, which Arrow would store as numbers, not nulls.
    ``per_origin`` holds the per-origin columns, ``others`` any other numbers
    (the factors, cdf and pct_reported, the link ratios, Mack's sigmas),
    ``total`` the one-row totals.
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
    if not (np.isfinite(others).all() and all(t is None or math.isfinite(t) for t in totals)):
        raise Refusal(
            "result_not_finite",
            "a total, a link ratio, or a development age's factor, cdf, pct_reported or "
            "sigma is not a finite number: the amounts are too large, or too far apart, for "
            "their sums, ratios and squares to stay finite. Check them for a unit error, or "
            "scale them (work in thousands, say) and scale the answer back",
            option="cells",
        )


# -- the ODP bootstrap -------------------------------------------------------------

#: The methods odp_bootstrap refits, each with the kernel candidate's method.
_BOOTSTRAP_METHODS = {
    "chain_ladder": "cl",
    "bornhuetter_ferguson": "bf",
    "benktander": "bf",
    "cape_cod": "gcc",
}

#: The process noise odp_bootstrap offers; ``"none"`` leaves the cells at their means.
BOOTSTRAP_PROCESSES = ("gamma", "od_poisson", "none")

#: The quantile levels odp_bootstrap reports by default.
_BOOTSTRAP_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)

#: The most numbers the ``draws`` table may hold (draws times origins): about
#: 7 GB at the peak, at about 70 bytes a number while they are summarised.
_MAX_DRAWN_NUMBERS = 100_000_000


@dataclass(frozen=True)
class BootstrapResult:
    """What :func:`odp_bootstrap` returns: pyarrow Tables with fixed column types.

    A missing number is an Arrow null, never NaN. ``to_polars(name)`` gives any
    of the tables as a polars DataFrame.

    Attributes
    ----------
    method : str
        The method refitted on every simulated triangle: ``"chain_ladder"``,
        ``"bornhuetter_ferguson"``, ``"benktander"`` or ``"cape_cod"``.
    as_of : datetime.date
        The information date, the evaluation date of the latest cell.
    dev_grain_months : int
        Months per development step.
    n_draws : int
        The number of simulated run-offs.
    seed : int
        The seed the draws came from: the one passed, or, when ``seed=None``
        was passed, the fresh entropy drawn for it, so passing it back as
        ``seed`` gives the same draws again.
    central : ReserveResult
        The point fit: exactly what ``methods.<method>`` returns for the same
        cells and options.
    origins : pyarrow.Table
        One row per origin period: ``origin`` and ``origin_period`` (as in
        :class:`ReserveResult`), ``latest_dev_lag`` (int64), ``latest`` (the
        actual latest cumulative), ``central_ultimate`` and ``central_ibnr``
        (from ``central``), ``mean_ibnr`` and ``sd_ibnr`` (the draws' mean, and
        their standard deviation with ``ddof=1``, null when ``n_draws`` is 1)
        and ``mean_ultimate`` (``latest + mean_ibnr``), all float64.
    totals : pyarrow.Table
        One row: ``latest``, ``central_ultimate``, ``central_ibnr``,
        ``mean_ibnr``, ``sd_ibnr`` (null at one draw) and ``mean_ultimate``
        for the sum over the origins (float64); ``n_draws``; ``n_residuals``
        (the residuals resampled); ``degrees_of_freedom`` (observed cells minus
        the ODP model's parameters, one per origin and one per development age
        after the first); ``n_negative_fitted`` (observed cells whose fitted
        mean is negative, 0 unless ``negative_increments="reflect"``);
        ``n_draws_unit_factor`` (draws in which some refitted factor had no
        positive volume and took 1.0); ``n_draws_negative_ibnr`` (draws whose
        total is below 0); ``n_draws_tail_fallback`` (draws whose tail curve
        failed its checks and used the central fit's curve), all int64;
        ``phi`` (float64, the Pearson scale); and the options
        ``residual_adjustment``, ``residual_pool``, ``negative_increments`` and
        ``process`` (string).
    quantiles : pyarrow.Table
        One row per origin, or the total, and level: ``origin`` and
        ``origin_period`` (null for the total), ``level`` (the probability, as
        passed), ``ibnr`` (the quantile of the draws, numpy's linear rule,
        which is ``np.percentile`` with the level times 100 and R's type 7) and
        ``tvar`` (the mean of the draws at or above that quantile). The total's
        rows come first, then each origin's in origin order, each in the order
        of the levels.
    draws : pyarrow.Table
        Every draw of every origin, ``n_draws`` times the number of origins
        rows, draw by draw: ``draw`` (int64, from 0), ``origin``,
        ``origin_period`` and ``ibnr`` (float64). A draw's total is the sum of
        its rows.
    residuals : pyarrow.Table
        One row per observed cell, origin by origin: ``origin``,
        ``origin_period``, ``dev_lag`` (int64), ``increment`` and ``fitted``
        (float64, the observed and the fitted incremental amount),
        ``residual`` (float64, the unscaled Pearson residual ``(increment -
        fitted) / sqrt(|fitted|)``, null where ``fitted`` is 0), ``leverage``
        (float64, the cell's leverage in the ODP GLM, null where ``fitted`` is
        0), ``adjusted`` (float64, the value resampled from this cell: adjusted,
        and centred under ``residual_pool="centred"``; null off the pool),
        ``in_pool`` (bool), ``reason`` (string: ``pooled``, ``leverage_one`` (a
        cell fitted exactly, left out under ``residual_pool="centred"``),
        ``zero_fitted_mean`` or ``excluded_link``) and ``link_reason`` (string,
        the ``link_ratios.reason`` of the link ratio whose development option
        took the cell out, null otherwise).
    """

    TABLES: ClassVar[tuple[str, ...]] = ("origins", "totals", "quantiles", "draws", "residuals")

    method: str
    as_of: dt.date
    dev_grain_months: int
    n_draws: int
    seed: int
    central: ReserveResult
    origins: pa.Table
    totals: pa.Table
    quantiles: pa.Table
    draws: pa.Table
    residuals: pa.Table

    def to_polars(self, table: str = "origins"):
        """One of the result's tables as a polars DataFrame.

        ``table`` is ``"origins"`` (the default), ``"totals"``, ``"quantiles"``,
        ``"draws"`` or ``"residuals"``; ``central.to_polars`` gives the point
        fit's. Needs polars: ``pip install "ibnr[polars]"``.
        """
        if table not in self.TABLES:
            raise Refusal(
                "invalid_option",
                f"table must be one of {self.TABLES}, got {{given}}",
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
        return pl.from_arrow(getattr(self, table))


def odp_bootstrap(
    cells,
    *,
    method: str = "chain_ladder",
    premium=None,
    expected_loss_ratio: float | None = None,
    n_iters: int = 1,
    decay: float | None = None,
    trend: float | None = None,
    prior_cv: float = 0.0,
    n_draws: int = 1000,
    seed: int | None = None,
    quantiles=_BOOTSTRAP_QUANTILES,
    residual_adjustment: str = "hat",
    residual_pool: str = "centred",
    negative_increments: str = "refuse",
    process: str = "gamma",
    dev_grain_months: int = 12,
    average: str = "volume",
    history_periods: int | None = None,
    drop_high: bool | int = False,
    drop_low: bool | int = False,
    preserve: int = 1,
    drop_above: float | None = None,
    drop_below: float | None = None,
    exclude=(),
    exclude_valuations=(),
    trim_ties: str = "volume",
    unsupported_factor: str = "raise",
    exhausted_exclusions: str = "keep",
    zero_cells: str = "missing",
    tail: str | None = None,
    tail_factor: float | None = None,
    tail_decay: float | None = None,
    tail_attach_lag: int | None = None,
    tail_fit_lags: tuple | None = None,
    tail_steps: int | None = None,
    tail_rows: int | None = None,
) -> BootstrapResult:
    """England and Verrall's over-dispersed Poisson bootstrap of the whole run-off.

    The residual bootstrap of R's ``BootChainLadder`` and chainladder-python's
    ``BootstrapODPSample``: fit ``method``, take the Pearson residuals of the
    observed increments against the fitted ones, resample them into
    ``n_draws`` simulated triangles, refit ``method`` on each, and add process
    noise to every future cell. Each draw is a possible run-off of every
    origin; ``draws`` has them all, and ``origins``, ``totals`` and
    ``quantiles`` summarise them beside the point fit, ``central``.

    ``cells``, the development options and the tail options are as in
    :func:`chain_ladder`. ``method`` is ``"chain_ladder"`` (the default),
    ``"bornhuetter_ferguson"``, ``"benktander"`` or ``"cape_cod"``; ``premium``,
    ``expected_loss_ratio``, ``n_iters``, ``decay`` and ``trend`` are as in
    those functions, each given only to the methods that read it (``decay``
    and ``trend``, ``None`` by default, are Cape Cod's 1.0 and 0.0). Mack's
    standard errors are :func:`mack`'s.

    **One set of development options.** They make the central fit, whose
    factors (after the tail) make the fitted values, and they decide which
    residuals are resampled: the cell a link ratio develops into leaves the
    pool when a development option leaves that ratio out, and so does the first
    cell of an origin whose ratio from the first age is left out
    (chainladder-python's rule; the zero rule leaves every residual in, since a
    zero increment is data). Every simulated triangle is refitted with the
    options that pick a link ratio by where it is (``history_periods``,
    ``exclude``, ``exclude_valuations``, the zero rule), decided once on the
    real triangle. The options that pick a ratio by its size (``drop_high``,
    ``drop_low``, ``drop_above``, ``drop_below``) act once, on the central
    factors and the pool, and not again on each simulated triangle: trimming
    those again removes ordinary draws a second time and pulls the simulated
    mean below the central estimate (3.5% on the Reserving app's workbook and
    22% on raa with ``drop_high``, measured on chainladder-python itself).
    ``average`` applies in both places. The fitted values are the backward
    recursion from the central factors whatever the average, so the hat
    adjustment's leverage is exactly the ODP GLM's only for the volume average
    over every link ratio; refitting the fitted triangle gives the central
    factors back for every average.

    **Tails** follow the same one specification: the attachment age and the
    curve apply to the central factors that make the fitted values and to
    every refit. A curve is refitted to each simulated triangle's own factors;
    a draw whose curve fails the tail's checks uses the central fit's curve
    and is counted in ``totals.n_draws_tail_fallback``. With a tail every
    origin's run-off includes it, as one more future cell per origin.

    The bootstrap options:

    - ``residual_adjustment``: ``"hat"`` (the default) divides each Pearson
      residual by ``sqrt(1 - h)``, ``h`` its leverage in the ODP GLM,
      chainladder-python's ``hat_adj=True``; ``"dof"`` multiplies every
      residual by ``sqrt(n / (n - p))``, R's rule; ``"none"`` leaves them as
      they are, which is what chainladder-python's ``hat_adj=False`` does
      (its docstring says degrees of freedom). The leverage is computed with a
      pseudo-inverse, so a development factor of exactly 1.0 does not switch
      the adjustment off, as it does in chainladder-python.
    - ``residual_pool``: ``"centred"`` (the default, chainladder-python's rule)
      resamples every residual except the cells fitted exactly (leverage one:
      the first origin's last cell and the last origin's first cell), after
      subtracting their mean; ``"all"`` (R's rule) resamples every residual, the
      exact cells' zeros included, as they are.
    - ``negative_increments``: ``"refuse"`` (the default) refuses a negative
      observed increment, and a negative fitted mean (only a factor below 1
      makes one), because the over-dispersed Poisson model is defined on
      non-negative increments. ``"reflect"`` bootstraps them as R and
      chainladder-python do: ``sqrt(|m|)`` scales a residual and the noise of
      a negative mean is reflected through its sign. ``totals.n_negative_fitted``
      counts the negative fitted means, whose results are rarely usable.
    - ``process``: the noise added to each future cell, with mean ``m`` and
      variance ``phi * |m|``: ``"gamma"`` (the default, as in R and
      chainladder-python), ``"od_poisson"`` (``phi`` times a Poisson count) or
      ``"none"`` (no noise, so the draws are the refitted means).
    - ``prior_cv``: Bornhuetter-Ferguson, Benktander and Cape Cod only. Above
      0 it varies the a priori loss ratio from draw to draw: each draw's a
      priori ultimates are multiplied by one lognormal number with mean 1 and
      coefficient of variation ``prior_cv``, shared by every origin of the
      draw (for Cape Cod, the loss ratio estimated from that draw's triangle).
      The central fit never draws. chainladder-python's ``apriori_sigma`` is a
      normal standard deviation instead: ``prior_cv = apriori_sigma /
      apriori`` for Bornhuetter-Ferguson and Benktander, ``prior_cv =
      apriori_sigma`` for Cape Cod, keeps its first two moments.
    - ``n_draws``: how many run-offs to simulate, a whole number of 1 or more
      (1,000 by default). ``draws`` holds ``n_draws`` times the number of
      origins rows, and that product is refused above 100,000,000.
    - ``seed``: a whole number of 0 or more makes the draws repeatable, 0
      included; ``None`` (the default) draws fresh entropy and returns it as
      ``BootstrapResult.seed``. The residuals, the process noise and the a
      priori multipliers each have a stream of their own, so ``process`` and
      ``prior_cv`` never change the simulated triangles, and the first ``k``
      draws of a run are the draws of a ``k``-draw run. The draws do not
      reproduce chainladder-python's or R's for the same seed.
    - ``quantiles``: the levels of the ``quantiles`` table, probabilities
      strictly between 0 and 1 (divide a percentile by 100).

    Every draw is a finite number: a simulated triangle whose refitted factors
    have no positive volume takes a factor of 1.0 at that age (counted in
    ``totals.n_draws_unit_factor``), and a run with a draw that is still not
    finite is refused (``result_not_finite``) rather than answered with 0 in
    its place. The simulated mean usually sits a little above the central
    estimate (3% on raa, 1% on genins); that is the refit's non-linearity, and
    R shows it too.

    Returns a :class:`BootstrapResult`. Input it will not answer is refused
    with :class:`Refusal`: besides the refusals of the point method,
    ``negative_increment`` and ``negative_fitted_mean`` (under
    ``negative_increments="refuse"``), ``degenerate_fit`` (a non-zero
    increment against a zero fitted mean, or a leverage that cannot be
    computed under the hat adjustment or the centred pool),
    ``not_identified`` (no more cells than the ODP model has parameters),
    ``empty_residual_pool`` and ``result_not_finite``. The run ties out to
    chainladder-python 0.9.2 draw for draw when both are fed the same
    residual choices, and to R's ``BootChainLadder`` within Monte Carlo error
    (``tests/test_odp_runoff.py``).
    """
    with _CallersTerms("odp_bootstrap") as terms:
        if not isinstance(method, str) or method not in _BOOTSTRAP_METHODS:
            raise Refusal(
                "invalid_option",
                "method must be 'chain_ladder', 'bornhuetter_ferguson', 'benktander' or "
                "'cape_cod', got {given}; Mack has its own standard errors in methods.mack",
                option="method",
                given=method,
            )
        count = _draw_count(n_draws)
        _require_seed(seed)
        levels = _levels(quantiles)
        for option, value, choices in (
            ("residual_adjustment", residual_adjustment, RESIDUAL_ADJUSTMENTS),
            ("residual_pool", residual_pool, RESIDUAL_POOLS),
            ("negative_increments", negative_increments, NEGATIVE_INCREMENTS),
            ("process", process, BOOTSTRAP_PROCESSES),
        ):
            if not isinstance(value, str) or value not in choices:
                listed = ", ".join(repr(choice) for choice in choices)
                raise Refusal(
                    "invalid_option",
                    f"{option} must be one of {listed}, got {{given}}",
                    option=option,
                    given=value,
                )
        cv = _prior_cv(prior_cv, method)
        settings = _bootstrap_settings(
            method,
            premium=premium,
            expected_loss_ratio=expected_loss_ratio,
            n_iters=n_iters,
            decay=decay,
            trend=trend,
        )
        grid, origins = terms.read(cells, dev_grain_months)
        if count * grid["n_w"] > _MAX_DRAWN_NUMBERS:
            raise Refusal(
                "invalid_option",
                f"n_draws times the number of origins must be at most {_MAX_DRAWN_NUMBERS:,}, "
                f"and {{given}} draws of {grid['n_w']} origins is more: each number takes "
                "about 70 bytes while the draws are summarised, so the limit is already about "
                "7 GB. Ask for fewer draws",
                option="n_draws",
                given=count,
            )
        wrapped = _candidate(
            _BOOTSTRAP_METHODS[method],
            origins,
            **settings,
            average=average,
            history_periods=history_periods,
            drop_high=drop_high,
            drop_low=drop_low,
            preserve=preserve,
            drop_above=drop_above,
            drop_below=drop_below,
            exclude=exclude,
            exclude_valuations=exclude_valuations,
            trim_ties=trim_ties,
            unsupported_factor=unsupported_factor,
            exhausted_exclusions=exhausted_exclusions,
            zero_cells=zero_cells,
            tail=_tail_spec(
                tail,
                tail_factor=tail_factor,
                tail_decay=tail_decay,
                tail_attach_lag=tail_attach_lag,
                tail_fit_lags=tail_fit_lags,
                tail_steps=tail_steps,
                tail_rows=tail_rows,
            ),
        )
        central = _conventional_result(method, grid, origins, wrapped, premium=premium)
        keyed = None if method == "chain_ladder" else _premium(premium, origins, method)
        try:
            with np.errstate(all="ignore"):
                setup = prepare_runoff(
                    grid,
                    wrapped.kernel,
                    premium=keyed,
                    adjustment=residual_adjustment,
                    pool=residual_pool,
                    negative_increments=negative_increments,
                )
        except Refusal as refusal:
            if refusal.reason != "negative_increment":
                raise
            raise refusal._replace(
                template=(
                    f"{refusal.count} negative increment(s): the cumulative falls into "
                    "{cells}. The over-dispersed Poisson bootstrap is defined on non-negative "
                    "increments; pass negative_increments='reflect' to bootstrap them through "
                    "sign(), as chainladder-python and R do, or use methods.mack"
                ),
                option="negative_increments",
                options=("negative_increments", "cells"),
            ) from None
        root = np.random.SeedSequence(seed)
        drawn = draw_runoff(
            setup.boot,
            setup.projection,
            n_draws=count,
            seed=root,
            process="gamma" if process == "none" else process,
            process_noise=process != "none",
            prior_cv=cv,
        )
        return _bootstrap_result(
            method,
            origins,
            central,
            setup,
            drawn.ibnr,
            unit=drawn.unit,
            tail_fallback=drawn.tail_fallback,
            seed=int(root.entropy),
            levels=levels,
            options={
                "residual_adjustment": residual_adjustment,
                "residual_pool": residual_pool,
                "negative_increments": negative_increments,
                "process": process,
            },
        )


def _prior_cv(prior_cv, method: str) -> float:
    if (
        not isinstance(prior_cv, numbers.Real)
        or isinstance(prior_cv, bool | np.bool_)
        or not math.isfinite(prior_cv)
        or prior_cv < 0
    ):
        raise Refusal(
            "invalid_option",
            "prior_cv must be a finite number of 0 or more, got {given}",
            option="prior_cv",
            given=prior_cv,
        )
    if prior_cv > 0 and method == "chain_ladder":
        raise Refusal(
            "invalid_option",
            "prior_cv varies the a priori loss ratio, and the chain ladder has none; use it "
            "with bornhuetter_ferguson, benktander or cape_cod",
            option="prior_cv",
            options=("prior_cv", "method"),
            given=prior_cv,
        )
    return float(prior_cv)


def _bootstrap_settings(
    method: str, *, premium, expected_loss_ratio, n_iters, decay, trend
) -> dict[str, Any]:
    """The kernel candidate's method settings, each refused where ``method`` does not read it."""

    def not_read(option: str, given, why: str) -> Refusal:
        return Refusal(
            "invalid_option",
            f"{option} was given for method={method!r}, which {why}",
            option=option,
            options=(option, "method"),
            given=given,
        )

    if method == "chain_ladder" and premium is not None:
        raise not_read("premium", None, "never reads it; premium is for the a priori methods")
    if method in ("chain_ladder", "cape_cod") and expected_loss_ratio is not None:
        why = (
            "estimates its loss ratio from the triangle"
            if method == "cape_cod"
            else "has no a priori loss ratio"
        )
        raise not_read("expected_loss_ratio", expected_loss_ratio, why)
    if isinstance(n_iters, bool | np.bool_) or (
        method in ("chain_ladder", "bornhuetter_ferguson") and n_iters != 1
    ):
        if isinstance(n_iters, bool | np.bool_):
            raise Refusal(
                "invalid_option",
                "n_iters must be a whole number of 1 or more, got {given}",
                option="n_iters",
                given=n_iters,
            )
        raise not_read(
            "n_iters",
            n_iters,
            "is not iterated; method='benktander' iterates Bornhuetter-Ferguson, and Cape Cod "
            "takes n_iters too",
        )
    for option, value in (("decay", decay), ("trend", trend)):
        if value is not None and method != "cape_cod":
            raise not_read(option, value, "does not read it; it is a Cape Cod setting")
    settings: dict[str, Any] = {}
    if method in ("bornhuetter_ferguson", "benktander"):
        settings["expected_loss_ratio"] = expected_loss_ratio
    if method in ("benktander", "cape_cod"):
        settings["n_iters"] = n_iters
    if method == "cape_cod":
        settings["decay"] = 1.0 if decay is None else decay
        settings["trend"] = 0.0 if trend is None else trend
    return settings


def _bootstrap_result(
    method: str,
    origins: _Origins,
    central: ReserveResult,
    setup,
    ibnr: np.ndarray,
    *,
    unit: np.ndarray,
    tail_fallback: np.ndarray,
    seed: int,
    levels: tuple[float, ...],
    options: dict[str, str],
) -> BootstrapResult:
    estimate, boot = setup.estimate, setup.boot
    step = central.dev_grain_months
    periods = list(estimate.origins["origin_period"])
    n_draws, n_w = ibnr.shape
    latest = np.asarray(estimate.origins["latest"], dtype=float)
    ultimate = np.asarray(estimate.origins["ultimate"], dtype=float)
    no_sd = n_draws == 1
    with np.errstate(all="ignore"):
        total = ibnr.sum(axis=1)
        mean = ibnr.mean(axis=0)
        sd = ibnr.std(axis=0, ddof=1) if not no_sd else np.zeros(n_w)
        total_mean = float(total.mean())
        total_sd = float(total.std(ddof=1)) if not no_sd else 0.0
        mean_ultimate = latest + mean
        total_latest = float(latest.sum())
        tails = [_tail(total, level) for level in levels] + [
            _tail(ibnr[:, i], level) for i in range(n_w) for level in levels
        ]
    _require_finite(
        periods,
        {"mean_ibnr": mean, "sd_ibnr": sd, "mean_ultimate": mean_ultimate},
        np.array(tails, dtype=float).ravel(),
        {
            "mean": _arrow.float64([total_mean]),
            "sd": _arrow.float64([total_sd]),
            "ultimate": _arrow.float64([total_latest + total_mean]),
        },
    )
    shown = origins.labels_for(periods)
    dates = _arrow.date32(periods)
    origin_table = pa.table(
        {
            "origin": shown,
            "origin_period": dates,
            "latest_dev_lag": _arrow.int64(estimate.origins["latest_dev_lag"]),
            "latest": _arrow.float64(latest),
            "central_ultimate": _arrow.float64(ultimate),
            "central_ibnr": _arrow.float64(ultimate - latest),
            "mean_ibnr": _arrow.float64(mean),
            "sd_ibnr": _arrow.float64(sd, mask=np.full(n_w, no_sd)),
            "mean_ultimate": _arrow.float64(mean_ultimate),
        }
    )
    totals = pa.table(
        {
            "latest": central.totals.column("latest"),
            "central_ultimate": central.totals.column("ultimate"),
            "central_ibnr": central.totals.column("ibnr"),
            "mean_ibnr": _arrow.float64([total_mean]),
            "sd_ibnr": _arrow.float64([total_sd], mask=np.array([no_sd])),
            "mean_ultimate": _arrow.float64([total_latest + total_mean]),
            "n_draws": _arrow.int64([n_draws]),
            "n_residuals": _arrow.int64([boot.pool.size]),
            "degrees_of_freedom": _arrow.int64([boot.degrees_of_freedom]),
            "n_negative_fitted": _arrow.int64([boot.n_negative_fitted]),
            "n_draws_unit_factor": _arrow.int64([int(unit.sum())]),
            "n_draws_negative_ibnr": _arrow.int64([int((total < 0).sum())]),
            "n_draws_tail_fallback": _arrow.int64([int(tail_fallback.sum())]),
            "phi": _arrow.float64([boot.phi]),
            **{name: _arrow.string([value]) for name, value in options.items()},
        }
    )
    # quantile rows: the total's (a null origin) first, then each origin's
    n_levels = len(levels)
    row_origin = np.r_[np.zeros(n_levels, dtype=np.int64), np.repeat(np.arange(n_w), n_levels)]
    is_total = np.r_[np.ones(n_levels, dtype=bool), np.zeros(n_w * n_levels, dtype=bool)]
    at = _arrow.int64(row_origin, mask=is_total)
    values = np.array(tails, dtype=float).reshape(-1, 2)
    quantile_table = pa.table(
        {
            "origin": shown.take(at),
            "origin_period": dates.take(at),
            "level": _arrow.float64(np.tile(levels, n_w + 1)),
            "ibnr": _arrow.float64(values[:, 0]),
            "tvar": _arrow.float64(values[:, 1]),
        }
    )
    each = _arrow.int64(np.tile(np.arange(n_w), n_draws))
    draws = pa.table(
        {
            "draw": _arrow.int64(np.repeat(np.arange(n_draws), n_w)),
            "origin": shown.take(each),
            "origin_period": dates.take(each),
            "ibnr": _arrow.float64(ibnr.ravel()),
        }
    )
    return BootstrapResult(
        method,
        central.as_of,
        step,
        n_draws,
        seed,
        central,
        origin_table,
        totals,
        quantile_table,
        draws,
        _residual_table(shown, dates, setup, step),
    )


def _residual_table(shown: pa.Array, dates: pa.Array, setup, step: int) -> pa.Table:
    """One row per observed cell, origin by origin: the residuals and which were resampled."""
    boot = setup.boot
    observed = boot.obs_mask
    rows, cols = np.nonzero(observed)
    resampled = np.full(observed.shape, np.nan)
    resampled[boot.pool_mask] = boot.pool
    at = _arrow.int64(rows)

    def nullable(values: np.ndarray) -> pa.Array:
        values = np.asarray(values, dtype=float)
        missing = np.isnan(values)
        return _arrow.float64(np.where(missing, 0.0, values), mask=missing)

    link_reason = setup.excluded_by[observed]
    return pa.table(
        {
            "origin": shown.take(at),
            "origin_period": dates.take(at),
            "dev_lag": _arrow.int64((cols + 1) * step),
            "increment": _arrow.float64(boot.inc[observed]),
            "fitted": _arrow.float64(boot.fitted[observed]),
            "residual": nullable(boot.unscaled[observed]),
            "leverage": nullable(
                np.where(boot.fitted[observed] != 0, boot.leverage[observed], np.nan)
            ),
            "adjusted": nullable(resampled[observed]),
            "in_pool": _arrow.bool_(boot.pool_mask[observed]),
            "reason": _arrow.string([POOL_REASONS[code] for code in boot.pool_reason[observed]]),
            "link_reason": _arrow.strings_or_nulls(
                [None if code < 0 else LINK_REASONS[code] for code in link_reason]
            ),
        }
    )


def _tail(sample: np.ndarray, level: float) -> tuple[float, float]:
    """The quantile at ``level`` and the mean of the draws at or above it.

    ``np.quantile`` at ``level`` is ``np.percentile`` at ``100 * level``, and the
    tail mean is taken over the draws themselves, as the Reserving app's
    ``/cdr`` takes it, so the two give the same numbers.
    """
    threshold = np.quantile(sample, level)
    return float(threshold), float(sample[sample >= threshold].mean())


def _draw_count(n_draws) -> int:
    if (
        not isinstance(n_draws, numbers.Integral)
        or isinstance(n_draws, bool | np.bool_)
        or n_draws < 1
    ):
        raise Refusal(
            "invalid_option",
            "n_draws must be a whole number of 1 or more, got {given}",
            option="n_draws",
            given=n_draws,
        )
    return int(n_draws)


def _require_seed(seed) -> None:
    if seed is None:
        return
    if not isinstance(seed, numbers.Integral) or isinstance(seed, bool | np.bool_) or seed < 0:
        raise Refusal(
            "invalid_option",
            "seed must be None or a whole number of 0 or more, got {given}",
            option="seed",
            given=seed,
        )


def _levels(quantiles) -> tuple[float, ...]:
    """The quantile levels as floats, or a refusal naming what was passed."""
    listed = isinstance(quantiles, list | tuple) or (
        isinstance(quantiles, np.ndarray) and quantiles.ndim == 1
    )
    refusal = Refusal(
        "invalid_option",
        "quantiles must be a list of probabilities strictly between 0 and 1, such as 0.995 "
        "(divide a percentile by 100), got {given}",
        option="quantiles",
        given=tuple(quantiles) if listed else quantiles,
    )
    if not listed:
        raise refusal
    levels = []
    for level in quantiles:
        if not isinstance(level, numbers.Real) or isinstance(level, bool | np.bool_):
            raise refusal
        value = float(level)
        if not 0.0 < value < 1.0:  # also false for NaN
            raise refusal
        levels.append(value)
    return tuple(levels)


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
        raise error._relabeled(method=self.method, label_of=label_of).with_traceback(
            trace
        ) from None


def _no_label(_start: dt.date) -> None:
    return None


# -- the conventional methods' result ---------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """A kernel candidate, and the exclusions as the caller wrote them."""

    kernel: ConventionalCandidate
    #: each exclusion as passed, with the (period start, dev_lag) it names
    exclusions: tuple[tuple[Any, tuple[dt.date, Any]], ...]
    #: each excluded valuation as passed, with the evaluation date it names
    valuations: tuple[tuple[Any, dt.date], ...] = ()


def _candidate(
    method: str, origins: _Origins, *, exclude, exclude_valuations=(), **settings
) -> _Candidate:
    valuations = _valuations(exclude_valuations, origins.step)
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
    kernel = ConventionalCandidate(
        method,
        exclude=tuple(key for _, key in exclusions),
        exclude_valuations=tuple(day for _, day in valuations),
        **settings,
    )
    return _Candidate(kernel, tuple(exclusions), valuations)


#: How an excluded valuation may be written, for the messages.
_VALUATION_FORMS = (
    "write it as the evaluation date, the last day of a development period (2020-12-31), or "
    "as the one development period it ends: a year (2020) on an annual triangle, a quarter "
    "(2020Q4) on a quarterly one, a month (2020-12) on a monthly one"
)


def _valuations(values, step: int) -> tuple[tuple[Any, dt.date], ...]:
    """Each excluded valuation as written, with the evaluation date it names, or a refusal.

    A date is taken as it is; a year, quarter or month label names the last day
    of that period, and must be one development period long. A valuation named
    twice is refused here; one that is not a diagonal of the triangle, or that
    no link ratio develops into, is refused once the grid is known.
    """
    if isinstance(values, str | bytes) or not hasattr(values, "__iter__"):
        raise Refusal(
            "invalid_option",
            "exclude_valuations must be a sequence of valuations, such as [2020] or "
            "['2020-12-31'], got {given}",
            option="exclude_valuations",
            given=values,
        )
    read = []
    for value in values:
        read.append((value, _valuation(value, step)))
    first_written: dict[dt.date, Any] = {}
    for value, day in read:
        if day in first_written:
            earlier = first_written[day]
            raise Refusal(
                "duplicate",
                f"exclude_valuations names {day.isoformat()} twice, as {_show(earlier)} and "
                f"{_show(value)}; list each valuation once",
                option="exclude_valuations",
                given=value,
            )
        first_written[day] = value
    return tuple(read)


def _valuation(value, step: int) -> dt.date:
    """The evaluation date one excluded valuation names, or a refusal."""
    shown = _show(value)
    where = {"option": "exclude_valuations", "given": value}
    unreadable = Refusal(
        "unreadable_label",
        f"exclude_valuations {shown} is not a valuation: {_VALUATION_FORMS}",
        **where,
    )
    start, months = None, None
    if isinstance(value, numbers.Integral) and not isinstance(value, bool | np.bool_):
        if not 1000 <= value <= 9999:
            raise unreadable
        start, months = dt.date(int(value), 1, 1), 12
    elif isinstance(value, str) and not _ISO_DATE.fullmatch(value):
        try:
            start, months = _label_period(value, "exclude_valuations")
        except Refusal:
            raise unreadable from None
    if months is not None:
        if months != step:
            length = "1 month" if months == 1 else f"{months} months"
            raise Refusal(
                "grain_mismatch",
                f"exclude_valuations {shown} is {_PERIOD_WORD[months]}, {length} long, but the "
                f"development periods are {step} months long (dev_grain_months={step}). "
                f"{_VALUATION_FORMS[0].upper()}{_VALUATION_FORMS[1:]}",
                **where,
            )
        return _add_months(start, months) - dt.timedelta(days=1)
    if isinstance(value, bool | np.bool_ | float | np.floating):
        raise unreadable  # as_date would not read these either, but say so in these terms
    try:
        day = as_date(value)
    except (TypeError, ValueError):
        raise unreadable from None
    if (day + dt.timedelta(days=1)).day != 1:
        raise Refusal(
            "unreadable_label",
            f"exclude_valuations {shown} is not the last day of a month, so it ends no "
            f"development period: {_VALUATION_FORMS}",
            **where,
        )
    return day


def _require_exclusions_in_triangle(grid, exclusions) -> None:
    """Refuse an exclusion that names a link ratio the triangle does not have.

    ``exclusions`` is :attr:`_Candidate.exclusions`: each pair as written, with
    the (period start, dev_lag) it names.
    """
    step = grid["dev_grain_months"]
    origins, mask = grid["origin_periods"], grid["obs_mask"]
    seen = {
        (origins[i], (j + 1) * step)
        for i, j in zip(*np.nonzero(mask[:, :-1] & mask[:, 1:]), strict=True)
    }
    unknown = [
        RefusedCell(pair[0], start, lag)
        for pair, (start, lag) in exclusions
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


def _require_valuations_in_triangle(grid, valuations) -> None:
    """Refuse an excluded valuation that no link ratio of the triangle develops into."""
    if not valuations:
        return
    step = grid["dev_grain_months"]
    origins, mask = grid["origin_periods"], grid["obs_mask"]
    into = sorted(
        {
            month_end(origins[i], (j + 2) * step)
            for i, j in zip(*np.nonzero(mask[:, :-1] & mask[:, 1:]), strict=True)
        }
    )
    have = (
        f"the link ratios develop into {into[0]} to {into[-1]}, every {step} months"
        if into
        else "the triangle has no link ratio"
    )
    first = origins[0]
    for value, day in valuations:
        if day in into:
            continue
        following = day + dt.timedelta(days=1)
        months = (following.year - first.year) * 12 + following.month - first.month
        if months % step:
            raise Refusal(
                "grain_mismatch",
                f"exclude_valuations {_show(value)} is not the last day of a development period "
                f"of this triangle; {have}",
                option="exclude_valuations",
                given=value,
            )
        raise Refusal(
            "not_in_triangle",
            f"exclude_valuations names {day}, but no link ratio develops into that date; {have}",
            option="exclude_valuations",
            given=value,
        )


def _conventional_result(
    name: str, grid, origins: _Origins, wrapped: _Candidate, *, premium
) -> ReserveResult:
    candidate = wrapped.kernel
    _require_valuations_in_triangle(grid, wrapped.valuations)
    keyed = None if candidate.method == "cl" else _premium(premium, origins, name)
    # fit_conventional_grid without its three pandas tables: the same checks and
    # numbers, held in numpy arrays and lists, so a fit here never loads pandas.
    # An overflow or an underflow (amounts near the largest or the smallest
    # double) is not warned about, because the estimator refuses a factor, a
    # pattern or an ultimate that is not finite by name, and _require_finite
    # below refuses the rest.
    with np.errstate(all="ignore"):
        fit = _estimate_grid(grid, candidate, premium=keyed)
    selection = fit.selection
    _require_exclusions_in_triangle(grid, wrapped.exclusions)
    step = grid["dev_grain_months"]
    table = fit.origins
    latest = np.asarray(table["latest"], dtype=float)
    ultimate = np.asarray(table["ultimate"], dtype=float)
    tail = fit.tail
    with np.errstate(all="ignore"):
        ibnr = ultimate - latest
        pattern = _pattern_numbers(fit.factors, grid["n_d"], tail)
        # the kernel checks each origin; a sum of finite ultimates can still overflow
        sums = _sums(latest, ultimate)
        sums["tail_factor"] = _arrow.float64([1.0 if tail is None else tail.tail_factor])
    columns = {
        "origin": origins.labels_for(table["origin_period"]),
        "origin_period": _arrow.date32(table["origin_period"]),
        "latest_dev_lag": _arrow.int64(table["latest_dev_lag"]),
        "latest": _arrow.float64(latest),
        "ultimate": _arrow.float64(ultimate),
        "ibnr": _arrow.float64(ibnr),
    }
    if candidate.method != "cl":
        columns["expected_loss_ratio"] = _arrow.float64(table["expected_loss_ratio"])
    if candidate.method == "gcc":
        columns["trended_loss_ratio"] = _arrow.float64(table["trended_loss_ratio"])
        columns["trend_factor"] = _arrow.float64(table["trend_factor"])

    def summary(key: str) -> list:
        return [row[key] for row in fit.summary]

    rows = 0 if tail is None else tail.rows
    development = pa.table(
        {
            **_pattern(pattern, step),
            **_tail_columns(grid["n_d"], tail),
            "n_selected": _per_link(summary("n_selected"), pa.int64(), rows),
            "unity_fallback": _per_link(summary("unity_fallback"), pa.bool_(), rows),
            "extreme_trimming_skipped": _per_link(
                summary("extreme_trimming_skipped"), pa.bool_(), rows
            ),
            "bounds_skipped": _per_link(summary("bounds_skipped"), pa.bool_(), rows),
        }
    )

    rows = {key: [row[key] for row in selection] for key in _LINK_COLUMNS}
    link_ratios = _link_ratios(origins, rows)
    ratio = np.array(rows["ratio"], dtype=float)
    per_origin = {"ultimate": ultimate, "ibnr": ibnr}
    if candidate.method == "gcc":
        # a trend near -1, or far above it, can take these past a double while the
        # ultimates stay finite
        per_origin["trended_loss_ratio"] = table["trended_loss_ratio"]
        per_origin["trend_factor"] = table["trend_factor"]
    _require_finite(
        table["origin_period"],
        per_origin,
        np.concatenate([*pattern, ratio[~np.isnan(ratio)], _curve_numbers(tail)]),
        sums,
    )
    totals = pa.table(sums)
    return ReserveResult(name, fit.as_of, step, pa.table(columns), development, link_ratios, totals)


def _pattern_numbers(
    factors: np.ndarray, n_d: int, tail: TailFit | None = None
) -> tuple[np.ndarray, ...]:
    """(factor, cdf, pct_reported), one row per development age.

    ``factors`` has one entry per link, ``n_d - 1`` of them. Without a tail the
    rows are the observed ages; with one they go on for ``tail.rows`` rows
    beyond the last observed age, each row's factor is the step to the next
    row, and each cdf includes the tail. Computed without warnings; the caller
    refuses a number that is not finite.
    """
    factors = np.asarray(factors, dtype=float)[: n_d - 1]
    with np.errstate(all="ignore"):
        cdf = np.r_[np.cumprod(factors[::-1])[::-1], 1.0]
        if tail is None:
            return factors, cdf, 1.0 / cdf
        factors = np.r_[factors, tail.shown]
        cdf = np.r_[cdf * tail.tail_factor, tail.beyond_cdf[1:]]
        return factors, cdf, 1.0 / cdf


def _tail_columns(n_d: int, tail: TailFit | None) -> dict[str, pa.Array]:
    """source, curve_factor and in_tail_fit, one row per development age.

    ``source`` says where each row's factor came from: ``"link_ratios"`` for a
    factor averaged from link ratios, ``"tail"`` for a factor (or, on the last
    row, the rest of the development) that came from the tail. Without a tail
    the last observed age has no factor and no source.
    """
    links = n_d - 1
    if tail is None:
        return {
            "source": _arrow.strings_or_nulls(["link_ratios"] * links + [None]),
            "curve_factor": _arrow.with_nulls([], pa.float64(), n_d),
            "in_tail_fit": _arrow.with_nulls([], pa.bool_(), n_d),
        }
    rows = tail.rows
    k = tail.attach_index
    source = ["link_ratios"] * k + ["tail"] * (links - k + 1 + rows)
    if tail.curve is None:
        curve = _arrow.with_nulls([], pa.float64(), n_d + rows)
        in_fit = _arrow.with_nulls([], pa.bool_(), n_d + rows)
    else:
        curve = _arrow.with_last_null(tail.curve, pa.float64())
        in_fit = _arrow.with_nulls(tail.in_fit, pa.bool_(), 1 + rows)
    return {"source": _arrow.strings_or_nulls(source), "curve_factor": curve, "in_tail_fit": in_fit}


def _curve_numbers(tail: TailFit | None) -> np.ndarray:
    """A curve tail's factors at every row, for the finiteness check."""
    if tail is None or tail.curve is None:
        return np.empty(0)
    return np.asarray(tail.curve, dtype=float)


def _per_link(values, kind: pa.DataType, rows: int) -> pa.Array:
    """One value per link, then a null for the last observed age and each tail row."""
    return _arrow.with_nulls(values, kind, 1 + rows)


#: The tail options every method takes, in the order they are listed.
_TAIL_OPTIONS = (
    "tail_factor",
    "tail_decay",
    "tail_attach_lag",
    "tail_fit_lags",
    "tail_steps",
    "tail_rows",
    "tail_sigma",
    "tail_std_err",
)


def _tail_spec(tail, **options) -> TailSpec | None:
    """The kernel's TailSpec from the methods' tail options, or a refusal.

    Every option other than ``tail`` defaults to None, so one passed without a
    tail is refused rather than ignored.
    """
    given = [name for name in _TAIL_OPTIONS if options.get(name) is not None]
    if tail is None:
        if given:
            name = given[0]
            raise Refusal(
                "invalid_option",
                f"{name} was given but tail is None; pass tail='constant' (with tail_factor) or "
                "a curve, 'exponential', 'inverse_power' or 'weibull', to have a tail",
                option=name,
                options=(name, "tail"),
                given=options[name],
            )
        return None
    fit_lags = options.get("tail_fit_lags")
    if tail == "constant" and fit_lags is not None:
        # The kernel reads (None, None) as not given, so a caller's (None, None)
        # would pass there unnoticed; here any value was given.
        raise Refusal(
            "invalid_option",
            "tail_fit_lags is a curve setting; a constant tail takes tail_factor and tail_decay",
            option="tail_fit_lags",
            options=("tail_fit_lags", "tail"),
            given=fit_lags,
        )
    return TailSpec(
        tail,
        factor=options.get("tail_factor"),
        decay=options.get("tail_decay"),
        attach_lag=options.get("tail_attach_lag"),
        fit_lags=(None, None) if fit_lags is None else fit_lags,
        steps=options.get("tail_steps"),
        rows=options.get("tail_rows"),
        sigma=options.get("tail_sigma"),
        std_err=options.get("tail_std_err"),
        option_prefix="tail_",
    )


def _pattern(numbers: tuple[np.ndarray, ...], step: int) -> dict[str, pa.Array]:
    """dev_lag, factor, cdf and pct_reported, one row per development age."""
    factors, cdf, pct_reported = numbers
    return {
        "dev_lag": _arrow.int64(np.arange(1, cdf.size + 1, dtype=np.int64) * step),
        "factor": _with_last_null(factors, pa.float64()),
        "cdf": _arrow.float64(cdf),
        "pct_reported": _arrow.float64(pct_reported),
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
    """A label as a message template holds it: text quoted, a year or a date as
    written, a numpy scalar as the value it holds, and braces never read as a
    placeholder."""
    if isinstance(value, np.number | np.bool_ | np.str_):
        value = value.item()
    if isinstance(value, dt.date):
        return value.isoformat()
    return _literal(repr(value) if isinstance(value, str) else str(value))


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
            f"got {_literal(kind)}; {_FORMS}",
            **where,
        )
    labels = pc.unique(column)
    row_label = _arrow.to_numpy(pc.index_in(column, value_set=labels)).astype(np.int64)
    if pa.types.is_timestamp(labels.type):
        shown = labels.cast(pa.string()).to_pylist()
        # a timestamp with a time zone is read as the date in that zone
        days = labels.cast(pa.date32(), safe=False).to_pylist()
        # never as Python datetimes: from pyarrow 25, to_pylist on a timestamp
        # with a time zone imports pandas
        values = shown
    else:
        values = labels.to_pylist()
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
        f"Table, not {type(data).__name__}: {_literal(exc)}.{also}",
        option=name,
    )


def _require_columns(table: pa.Table, needed: tuple[str, ...], name: str, what: str) -> None:
    missing = [column for column in needed if column not in table.column_names]
    if missing:
        raise Refusal(
            "invalid_table",
            f"{name} is missing column(s) {missing}; it has {_literal(table.column_names)}. {what}",
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
        # an age of 2**63 months or more has no whole-number type to go in, and
        # numpy would turn it into -9223372036854775808 with only a warning
        huge = np.flatnonzero(np.abs(months) >= 2.0**63)
        if huge.size:
            shown = ", ".join(repr(value) for value in sorted(set(months[huge].tolist()))[:5])
            raise Refusal(
                "invalid_age",
                f"dev_lag {shown} is too large to be a number of months",
                rows=huge,
                **where,
            )
        return months.astype(np.int64)
    raise Refusal(
        "invalid_table",
        f"dev_lag must be a column of whole numbers of months, got {_literal(kind)}",
        **where,
    )


def _numbers(column: pa.ChunkedArray, name: str, why: str, *, option: str, cell_at) -> np.ndarray:
    """A numeric column as float64 with every value a finite number, or a refusal.

    ``cell_at(rows)`` turns row positions into the refused cells, so a missing
    or infinite amount is named by its cell.
    """
    kind = column.type
    where = {"option": option, "column": name}
    if not (pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind)):
        raise Refusal(
            "invalid_table", f"{name} must be a numeric column, got {_literal(kind)}", **where
        )
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
            "premium must be a positive finite number for every origin; it is {given} for "
            "{origins}",
            given=amount,
            **where,
        )
