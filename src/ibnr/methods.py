"""Traditional reserving methods, one function per method, on one triangle at a time.

``from ibnr import methods``, then call the method you want by its name:

- :func:`chain_ladder`
- :func:`bornhuetter_ferguson`
- :func:`benktander` (Bornhuetter-Ferguson iterated; ``n_iters=1`` is Bornhuetter-Ferguson)
- :func:`cape_cod` (Gluck's generalized Cape Cod, with a trend; ``decay=1`` is the
  classic one)
- :func:`mack` (the chain ladder with Mack's standard errors; it takes the same
  development options, and ``average`` is Mack's alpha)
- :func:`tweedie_glm` (a Tweedie GLM fitted to the increments; power 1 with its
  defaults is the over-dispersed Poisson model, which reproduces the chain
  ladder)
- :func:`ml_development` (a random forest or gradient boosting fitted to the
  cells, chainladder-python's ``DevelopmentML``; needs the ``ml`` extra, and
  usually overshoots the chain ladder by the method's nature)
- :func:`one_year_cdr` (the one-year claims development result: how far next
  year's re-estimate of Mack's chain-ladder ultimate can move, as the
  Merz-Wuthrich standard error beside a simulation of next year)

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
to keep zeros as data, which is the default of the kernels underneath and of
:func:`one_year_cdr`.

The results echo each origin's label back in a column ``origin``, with the
value the caller wrote, next to ``origin_period``, which is always the first
day of the period. ``dev_lag`` still counts from the period's first day, so
the accident year written 2020-12-31 has its first cell at ``dev_lag`` 12.

Any table Arrow can read is accepted: a polars DataFrame, a pyarrow Table or
RecordBatch, anything else that offers the Arrow stream interface, or a dict
of columns (Python lists or numpy arrays, as a service reading JSON has them).
Other columns are ignored. Each function returns a :class:`ReserveResult`
(:func:`one_year_cdr` a :class:`OneYearCDRResult`), whose tables are pyarrow
Tables, so a service needs no DataFrame library at all; for analysis,
``result.to_polars()`` turns any of them into a polars DataFrame
(``pip install "ibnr[polars]"``).

The ``development`` table has one row per observed development age, and the
same columns for every method that carries them, in this order:

======================== ======== ======================================= =====================
column                   type     meaning                                 methods
======================== ======== ======================================= =====================
dev_lag                  int64    the age, in months                      all
factor                   float64  the link factor to the next age; null   all
                                  at the last age
cdf                      float64  the factor to the last observed age     all
pct_reported             float64  ``1 / cdf``                             all
n_selected               int64    link ratios behind the factor           link-ratio methods
unity_fallback           bool     no ratio was left and 1.0 was used      chain_ladder,
                                  (for ml_development, no cell was left   bornhuetter_ferguson,
                                  to fit)                                 benktander, cape_cod,
                                                                          ml_development
extreme_trimming_skipped bool     ``preserve`` stopped ``drop_high`` and  link-ratio methods
                                  ``drop_low`` at this age
bounds_skipped           bool     ``preserve`` stopped ``drop_above`` and link-ratio methods
                                  ``drop_below`` at this age
sigma                    float64  Mack's sigma                            mack
std_err                  float64  the factor's standard error             mack
sigma_extrapolated       bool     sigma came from ``sigma_rule``: the     mack
                                  age kept fewer than two link ratios
n_observed               int64    observed increments at the age          tweedie_glm
n_trained                int64    training rows at the age                ml_development
======================== ======== ======================================= =====================

The link-ratio methods are every method but :func:`tweedie_glm` and
:func:`ml_development`, which fit the cells; their ``factor``, ``cdf`` and
``pct_reported`` are null where the fitted pattern differs by origin (each
origin's is then in ``cells``). Every column but ``dev_lag``, ``cdf``,
``pct_reported``, ``n_observed`` and ``n_trained`` is null at the last age,
which has no next age. A method carries exactly the columns listed for it,
whatever options it is given, so a service can read each table by name.

Importing this module loads numpy and pyarrow, and not ibis, pandas, scipy or
scikit-learn, and seven of the eight functions do not load them when they run,
so a service that starts a new process for a request does not pay for them
(the CHANGELOG has the times). The eighth, :func:`ml_development`, fits a
scikit-learn model: its first call imports scikit-learn, which loads scipy and
pandas, never ibis. That holds for every input above except two, which make
pyarrow load pandas: a pandas DataFrame, and a dict with a list that is not
all strings, all bools, all dates or all numbers (one holding a null or a
datetime, for example), which is left to ``pa.table``.

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
among candidates on their history (``kernels.select_conventional``), other
ways to draw next year's diagonal for the one-year claims development result
(``kernels.simulate_one_year_cdr``), and the Triangle path to all of these
(``kernels.fit_conventional``, ``kernels.fit_mack``).
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
from ibnr.kernels.cdr import _one_year_cdr_draws
from ibnr.kernels.cdr import one_year_cdr as _merz_wuthrich
from ibnr.kernels.conventional import ConventionalCandidate, _estimate_grid
from ibnr.kernels.glm import TweedieFit, TweedieSpec, fit_tweedie_grid
from ibnr.kernels.grid import as_date, check_grid, grid_from_columns, month_end
from ibnr.kernels.links import REASONS as LINK_REASONS
from ibnr.kernels.links import is_all_history
from ibnr.kernels.mack import PROCESS_LAWS, MackFit, _require_mack_average, fit_mack_grid
from ibnr.kernels.ml_development import (
    MLDevelopmentFit,
    MLDevelopmentSpec,
    fit_ml_development_grid,
)
from ibnr.kernels.rng import cohort_stream

__all__ = [
    "OneYearCDRResult",
    "Refusal",
    "ReserveResult",
    "benktander",
    "bornhuetter_ferguson",
    "cape_cod",
    "chain_ladder",
    "mack",
    "ml_development",
    "one_year_cdr",
    "tweedie_glm",
]

#: The tables a result carries, in the order ``to_polars`` lists them.
TABLES = ("origins", "development", "link_ratios", "totals", "cells", "coefficients")

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
        two parts, ``parameter_se`` and ``process_se``.
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
        than by measurement. Every method but :func:`tweedie_glm` and
        :func:`ml_development` adds ``n_selected`` (int64, the link ratios
        behind the factor), ``extreme_trimming_skipped`` and ``bounds_skipped``
        (bool); the chain ladder, Bornhuetter-Ferguson, Benktander and Cape Cod
        add ``unity_fallback`` (bool); Mack adds ``sigma``, ``std_err`` (the
        factor's standard error) and ``sigma_extrapolated`` (bool), all null at
        the last age. :func:`tweedie_glm` adds ``n_observed`` (int64) and
        :func:`ml_development` ``n_trained`` (int64) and ``unity_fallback``;
        both fit a pattern that can differ by origin, and where it does,
        ``factor``, ``cdf`` and ``pct_reported`` are null and each origin's are
        in ``cells``. The module docstring has the table of every column and
        the methods that carry it.
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
        it. Every link-ratio method returns this table; ``None`` for
        :func:`tweedie_glm` and :func:`ml_development`, which fit the cells
        rather than link ratios, and for a result built by hand.
    totals : pyarrow.Table
        One row with ``latest``, ``ultimate`` and ``ibnr`` summed over the
        origins. Mack adds ``mack_se``, ``parameter_se`` and ``process_se`` for
        the total, which is not the sum of the origins' standard errors: the
        origins share the estimated factors. :func:`tweedie_glm` and
        :func:`ml_development` add the columns their docstrings list.
    cells : pyarrow.Table or None
        One row per cell of the full rectangle, observed or not, from a method
        that fits a model to the cells (:func:`tweedie_glm`,
        :func:`ml_development`); ``None`` for the others. Its columns are listed
        in each of those functions.
    coefficients : pyarrow.Table or None
        One row per term of a fitted model (:func:`tweedie_glm`); ``None`` for
        the others.
    """

    method: str
    as_of: dt.date
    dev_grain_months: int
    origins: pa.Table
    development: pa.Table
    link_ratios: pa.Table | None
    totals: pa.Table
    cells: pa.Table | None = None
    coefficients: pa.Table | None = None

    def to_polars(self, table: str = "origins"):
        """One of the result's tables as a polars DataFrame.

        ``table`` is ``"origins"`` (the default), ``"development"``,
        ``"link_ratios"``, ``"totals"``, ``"cells"`` or ``"coefficients"``; a
        table the result does not carry is refused. Needs polars, which the
        ``polars`` extra installs: ``pip install "ibnr[polars]"``.
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
            # tweedie_glm and ml_development have no link_ratios, only they have
            # cells, only tweedie_glm has coefficients, and a result built by
            # hand may lack any table
            raise Refusal(
                "invalid_option",
                f"a {self.method} result has no {table} table. {_absent(self.method, table)}",
                option="table",
                given=table,
                method=self.method,
            )
        return _to_polars(data)


def _to_polars(data: pa.Table):
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            'to_polars needs polars; install it with pip install "ibnr[polars]"'
        ) from exc
    return pl.from_arrow(data)


def _absent(method: str, table: str) -> str:
    """Why a result has no such table, and where its numbers are instead."""
    if table == "link_ratios" and method == "ml_development":
        return (
            "The model is fitted to the cells, not to link ratios; each origin's fitted factors "
            "are in cells, and in development when every origin shares one pattern"
        )
    if table == "link_ratios" and method == "tweedie_glm":
        return (
            "The GLM is fitted to the increments, not to link ratios; its fitted factors are "
            "in development (under the log link) and, origin by origin, in cells"
        )
    if table == "coefficients" and method == "ml_development":
        return "A tree model has no coefficients; its fitted cells are in the cells table"
    if table in ("cells", "coefficients"):
        return (
            "Only tweedie_glm and ml_development fit a model to the cells, and their results "
            "carry the cells table (tweedie_glm's the coefficients table too)"
        )
    return "This result was built by hand without it"


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
            preserve=preserve,
            drop_above=drop_above,
            drop_below=drop_below,
            exclude=exclude,
            exclude_valuations=exclude_valuations,
            trim_ties=trim_ties,
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

    ``cells`` and the development options are as in :func:`chain_ladder`, and
    ``premium`` and ``expected_loss_ratio`` as in :func:`bornhuetter_ferguson`.
    The result has the same tables and columns as Bornhuetter-Ferguson's.
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

    ``cells`` and the development options are as in :func:`chain_ladder`, and
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
    development options.

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
        return _mack(grid, labels, wrapped, sigma_rule=sigma_rule)


def _mack(grid, labels: _Origins, wrapped: _Candidate, *, sigma_rule: str) -> ReserveResult:
    _, as_of = check_grid(grid)
    step = grid["dev_grain_months"]
    _require_two_ages(grid, "mack")
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
    fit = _checked_mack_fit(
        grid,
        sigma_rule=sigma_rule,
        zero_cells=zero_cells,
        method="mack",
        average=average,
        links=links,
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
        pattern = _pattern_numbers(fit.f, fit.n_d)
        total = {
            **_sums(latest, ultimate),
            "mack_se": _arrow.float64([np.sqrt(risk["msep_total"])]),
            "parameter_se": _arrow.float64([np.sqrt(risk["parameter_total"])]),
            "process_se": _arrow.float64([np.sqrt(risk["process_total"])]),
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
    development = pa.table(
        {
            **_pattern(pattern, step),
            "n_selected": _with_last_null(fit.n_obs.tolist(), pa.int64()),
            "extreme_trimming_skipped": _with_last_null(trimming_skipped.tolist(), pa.bool_()),
            "bounds_skipped": _with_last_null(bounds_skipped.tolist(), pa.bool_()),
            "sigma": _with_last_null(sigma, pa.float64()),
            "std_err": _with_last_null(std_err, pa.float64()),
            "sigma_extrapolated": _with_last_null((fit.n_pos < 2).tolist(), pa.bool_()),
        }
    )
    link_ratios = _link_ratios(labels, _mack_link_rows(fit))
    return ReserveResult("mack", as_of, step, origins, development, link_ratios, pa.table(total))


def _no_sigma_anywhere(n_d: int, step: int, method: str) -> Refusal:
    return Refusal(
        "variance_not_estimable",
        f"{method} needs at least one development age with two or more link ratios to "
        "estimate Mack's sigma; this triangle has at most one at every age ({links}), so every "
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


def _require_two_ages(grid, method: str) -> None:
    if grid["n_d"] < 2:
        raise Refusal(
            "variance_not_estimable",
            f"{method} needs at least two development ages; this triangle has one. "
            "chain_ladder gives the latest amounts as the ultimates",
            option="cells",
        )


def _checked_mack_fit(
    grid,
    *,
    sigma_rule: str,
    zero_cells: str,
    method: str,
    average: str = "volume",
    links=None,
) -> MackFit:
    """Mack's fit, refused where its standard errors cannot be estimated.

    Shared by :func:`mack` and :func:`one_year_cdr`; ``method`` is the calling
    function's name, which the messages use. ``average`` and ``links`` are
    passed to ``fit_mack_grid``; :func:`one_year_cdr` takes no development
    option, so it leaves both at their defaults.
    """
    step = grid["dev_grain_months"]
    _require_two_ages(grid, method)
    if links is not None:
        # the triangle itself has too few link ratios for any sigma: refused here in
        # the same words as below, before the kernel would blame the options
        cum, mask = grid["cum"], grid["obs_mask"]
        pair = mask[:, :-1] & mask[:, 1:] & (cum[:, :-1] > 0)
        if zero_cells == "missing":
            pair &= cum[:, 1:] != 0
        if (pair.sum(axis=0) < 2).all():
            raise _no_sigma_anywhere(grid["n_d"], step, method)
    # Amounts near the largest or the smallest double are finite but Mack's sums,
    # ratios and squares are not. numpy says so with a warning, which is silenced
    # here and below because the kernel and _require_finite refuse such an answer
    # by name before anything is returned.
    with np.errstate(all="ignore"):
        fit = fit_mack_grid(
            grid, sigma_rule=sigma_rule, zero_cells=zero_cells, average=average, links=links
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
        raise _no_sigma_anywhere(fit.n_d, step, method)
    return fit


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
    open_ = (fit.latest_dev < fit.n_d - 1) & (fit.latest > 0)
    noisy = np.array([(fit.sigma2[int(k) :] > 0).any() for k in fit.latest_dev], dtype=bool)
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
    if not (np.isfinite(others).all() and all(math.isfinite(t) for t in totals)):
        raise Refusal(
            "result_not_finite",
            "a total, a link ratio, or a development age's factor, cdf, pct_reported or "
            "sigma is not a finite number: the amounts are too large, or too far apart, for "
            "their sums, ratios and squares to stay finite. Check them for a unit error, or "
            "scale them (work in thousands, say) and scale the answer back",
            option="cells",
        )


def tweedie_glm(
    cells,
    *,
    power: float = 1.0,
    link: str = "log",
    origin: str = "factor",
    calendar: str = "none",
    projection: str = "pattern",
    dev_grain_months: int = 12,
    max_iter: int = 100,
    tail=None,
) -> ReserveResult:
    """A Tweedie GLM fitted to the triangle's increments, and each origin's ultimate.

    ``cells`` is as in :func:`chain_ladder`: CUMULATIVE losses, one row per
    observed cell, zero or more. The function takes the increments of those
    cumulatives and fits, by iteratively reweighted least squares, a GLM in
    which each increment has mean ``mu`` and variance ``dispersion * mu **
    power``:

    - ``power``: 0 is the normal distribution, 1 the over-dispersed Poisson (the
      default), between 1 and 2 the compound Poisson-gamma, 2 the gamma, and
      above 2 the other Tweedie distributions. No Tweedie distribution has a
      power between 0 and 1, and those powers are refused. At power 1 and above
      every increment must be zero or more, and at power 2 and above more than
      zero; power 0 accepts negative increments, but under the log link, which
      keeps every fitted increment above zero, an age or origin whose
      increments sum to zero or less can leave no finite fit (the refusal names
      it; ``link="identity"`` fits it).
    - ``link``: ``"log"`` (the default), ``log(mu)`` is the sum of the terms, or
      ``"identity"``, ``mu`` is.
    - ``origin``: ``"factor"`` (the default) gives each origin its own level;
      ``"none"`` gives every origin the same expected increments.
    - ``calendar``: ``"none"`` (the default) or ``"trend"``, a straight line in
      the calendar period on the link scale. Only with ``origin="none"``: beside
      origin and development factors a calendar trend cannot be estimated,
      because the calendar period is the origin index plus the development
      index. Under the log link a calendar trend with development factors fits
      the same means as a straight line across origins, so it is not an
      inflation estimate.
    - ``projection``: ``"pattern"`` (the default) takes each origin's latest
      cumulative times its fitted development from the latest age to the last,
      as chainladder-python's ``TweedieGLM`` with ``Chainladder`` does;
      ``"increments"`` adds the fitted future increments to the latest
      cumulative, as R's ``glmReserve`` does. The two agree at power 1 with a
      log link and origin factors. ``"increments"`` needs origin factors.
    - ``max_iter``: the most iterations (100 by default). A fit that has not
      settled by then is refused, never returned. The fit stops when no fitted
      increment moves by more than 1e-10 times the largest increment, a rule
      that does not depend on the units, and there is no penalty, so the
      answer scales exactly with the amounts.
    - ``dev_grain_months``: as in :func:`chain_ladder`.
    - ``tail``: not supported yet; anything but ``None`` is refused. Each origin
      is projected to the last observed development age.

    Power 1 with a log link and origin and development factors (the defaults)
    is the over-dispersed Poisson model, whose ultimates equal the
    volume-weighted chain ladder's, ``chain_ladder(cells, zero_cells="observed")``,
    by either projection. The exception is an origin whose losses start from
    zero (a cumulative of 0 followed by a positive one): the chain ladder keeps
    it at 0 times its factors, and the GLM gives it a level from its later cells.
    Under the log link an origin (with origin factors) or a development age
    whose observed increments are all zero is fitted at exactly zero, the limit
    of the fit: its coefficient is null in ``coefficients`` with
    ``fitted_zero`` true, the zero origin's ultimate is 0 and the factor into a
    zero age is 1. Where the losses leave the fit with no finite answer (an
    origin whose only losses sit where the model needs a positive level from
    cells that are all zero), it is refused as ``degenerate_fit``.

    Unlike chainladder-python's ``TweedieGLM``, there is no penalty and no
    ``alpha``: chainladder always applies scikit-learn's ridge penalty of 1.0,
    whatever ``alpha`` it is given, so its answers depend on the units of the
    amounts. ``docs/coming-from-chainladder.md`` lists the differences.

    Returns a :class:`ReserveResult` whose ``link_ratios`` is ``None``:

    - ``origins``: ``origin``, ``origin_period``, ``latest_dev_lag``,
      ``latest``, ``ultimate`` and ``ibnr`` (by the chosen projection) and
      ``model_ibnr`` (float64, the sum of the fitted future increments, which
      is ``ibnr`` under ``projection="increments"``).
    - ``development``: ``dev_lag``, ``factor``, ``cdf`` and ``pct_reported``,
      the one fitted pattern every origin shares under the log link; null
      under the identity link, whose pattern differs by origin (it is in
      ``cells``), and null where the fitted cumulative is 0 (a first age with no
      losses). ``n_observed`` (int64) counts the observed increments at the age.
    - ``cells``: one row per cell of the full rectangle, observed or not:
      ``origin``, ``origin_period``, ``dev_lag``, ``observed`` (bool),
      ``increment`` (null when not observed), ``fitted_increment``,
      ``fitted_cumulative``, ``factor`` (the origin's fitted factor to the next
      age; null at the last age or where the fitted cumulative is 0), ``cdf``
      (to the last age; null where the fitted cumulative is 0) and
      ``pearson_residual`` (``(increment - fitted) / sqrt(fitted ** power)``,
      not divided by the dispersion; null when not observed or the fitted
      increment is 0).
    - ``coefficients``: one row per term, with the first origin and the first
      age with losses as the reference levels, as in R's ``glm``: ``term``
      (``intercept``, ``origin``, ``development`` or ``calendar``),
      ``origin`` and ``origin_period`` (null unless an origin term),
      ``dev_lag`` (int64, null unless a development term), ``estimate`` (on
      the link scale) and ``std_error`` (null when ``fitted_zero`` or when the
      dispersion is undefined), and ``fitted_zero`` (bool).
    - ``totals``: ``latest``, ``ultimate``, ``ibnr``, ``model_ibnr``, ``power``,
      ``link`` (string), ``deviance``, ``pearson_chi2``, ``dispersion``
      (``pearson_chi2 / (n_observed - n_parameters)``, null when they are
      equal), ``n_observed``, ``n_parameters`` and ``iterations`` (int64).
      ``n_observed`` and ``n_parameters`` count only the cells and terms in
      the regression, not those fitted at zero.

    Refused, besides the refusals of :func:`chain_ladder`'s ``cells``: a
    negative increment at power 1 or above (``negative_increment``), a zero
    increment at power 2 or above (``zero_increment``), no losses at all or
    more terms than the cells can tell apart (``not_identified``), a fit that
    did not settle (``did_not_converge``), a fit with no finite answer
    (``degenerate_fit``, including a fitted mean that falls below 1e-8 of the
    largest increment on a cell whose increment is zero or less), an
    identity-link fitted increment of zero or less at a power above 0, or one
    that falls to 1e-8 of the largest increment where the increment is 0
    (``negative_fitted_mean``), and an ultimate below zero
    (``negative_projection``). Whether a triangle is refused does not depend
    on its units.
    """
    with _CallersTerms("tweedie_glm") as terms:
        grid, labels = terms.read(cells, dev_grain_months)
        spec = TweedieSpec(
            power=power,
            link=link,
            origin=origin,
            calendar=calendar,
            projection=projection,
            max_iter=max_iter,
        )
        if tail is not None:
            raise Refusal(
                "not_supported",
                "tweedie_glm takes no tail yet: each origin is projected to the last observed "
                "development age. Pass tail=None",
                option="tail",
                given=tail,
            )
        # Amounts near the largest or the smallest double: numpy's warnings are
        # silenced because a number that is not finite is refused by name below.
        with np.errstate(all="ignore"):
            fit = fit_tweedie_grid(grid, spec)
        return _tweedie_result(fit, labels)


def _nullable(values, missing) -> pa.Array:
    """A float64 array with a null where ``missing`` is true (never NaN)."""
    values = np.asarray(values, dtype=float)
    missing = np.asarray(missing, dtype=bool)
    return _arrow.float64(np.where(missing, 0.0, values), mask=missing)


def _tweedie_result(fit: TweedieFit, labels: _Origins) -> ReserveResult:
    step = fit.dev_grain_months
    n_w, n_d = fit.cumulative.shape
    periods = fit.origin_periods
    observed = fit.observed
    with np.errstate(all="ignore"):
        latest, ultimate, model_ibnr = fit.latest, fit.ultimate, fit.model_ibnr
        ibnr = ultimate - latest
        fitted, fitted_cum = fit.fitted, fit.fitted_cumulative
        factors, cdf, residual = fit.factors, fit.cdf, fit.pearson_residuals
        sums = {
            **_sums(latest, ultimate),
            "model_ibnr": _arrow.float64([float(model_ibnr.sum())]),
        }
    ages = np.arange(1, n_d + 1, dtype=np.int64) * step

    # the development table: the shared pattern under the log link
    if fit.spec.link == "log":
        live = np.flatnonzero(fitted_cum[:, -1] > 0)
        reference = fitted_cum[live[0]] if live.size else np.zeros(n_d)
        with np.errstate(all="ignore"):
            pattern_cdf = reference[-1] / reference
            pattern_factor = np.r_[reference[1:] / reference[:-1], 0.0]
        no_cdf = reference == 0
        no_factor = np.r_[reference[:-1] == 0, True]
    else:
        pattern_cdf = pattern_factor = np.zeros(n_d)
        no_cdf = no_factor = np.ones(n_d, dtype=bool)
    with np.errstate(all="ignore"):
        pattern_pct = 1.0 / pattern_cdf
    development = pa.table(
        {
            "dev_lag": _arrow.int64(ages),
            "factor": _nullable(pattern_factor, no_factor),
            "cdf": _nullable(pattern_cdf, no_cdf),
            "pct_reported": _nullable(pattern_pct, no_cdf),
            "n_observed": _arrow.int64(observed.sum(axis=0)),
        }
    )

    # the cells table: every cell of the rectangle, origin by origin
    cell_periods = [periods[i] for i in range(n_w) for _ in range(n_d)]
    cell_factor = np.c_[factors, np.zeros(n_w)]
    no_cell_factor = np.c_[fitted_cum[:, :-1] == 0, np.ones(n_w, dtype=bool)]
    no_residual = ~observed | (fitted == 0)
    cells = pa.table(
        {
            "origin": labels.labels_for(cell_periods),
            "origin_period": _arrow.date32(cell_periods),
            "dev_lag": _arrow.int64(np.tile(ages, n_w)),
            "observed": _arrow.bool_(observed.ravel()),
            "increment": _nullable(fit.increments.ravel(), ~observed.ravel()),
            "fitted_increment": _arrow.float64(fitted.ravel()),
            "fitted_cumulative": _arrow.float64(fitted_cum.ravel()),
            "factor": _nullable(cell_factor.ravel(), no_cell_factor.ravel()),
            "cdf": _nullable(cdf.ravel(), (fitted_cum == 0).ravel()),
            "pearson_residual": _nullable(residual.ravel(), no_residual.ravel()),
        }
    )

    # the coefficients table
    kinds = [kind for kind, _ in fit.terms]
    term_period = [periods[i] if kind == "origin" else None for kind, i in fit.terms]
    term_age = np.array([(j + 1) * step if kind == "development" else 0 for kind, j in fit.terms])
    no_se = fit.fitted_zero | (fit.n_obs == fit.n_params)
    coefficients = pa.table(
        {
            "term": _arrow.string(kinds),
            "origin": labels.labels_or_null(term_period),
            "origin_period": _arrow.date32(term_period),
            "dev_lag": _arrow.int64(term_age, mask=np.array(kinds) != "development"),
            "estimate": _nullable(fit.coef, fit.fitted_zero),
            "std_error": _nullable(fit.coef_se, no_se),
            "fitted_zero": _arrow.bool_(fit.fitted_zero),
        }
    )

    no_dispersion = fit.n_obs == fit.n_params
    totals = pa.table(
        {
            **sums,
            "power": _arrow.float64([fit.spec.power]),
            "link": _arrow.string([fit.spec.link]),
            "deviance": _arrow.float64([fit.deviance]),
            "pearson_chi2": _arrow.float64([fit.pearson_chi2]),
            "dispersion": _nullable([fit.dispersion], [no_dispersion]),
            "n_observed": _arrow.int64([fit.n_obs]),
            "n_parameters": _arrow.int64([fit.n_params]),
            "iterations": _arrow.int64([fit.iterations]),
        }
    )
    origins = pa.table(
        {
            "origin": labels.labels_for(periods),
            "origin_period": _arrow.date32(periods),
            "latest_dev_lag": _arrow.int64((fit.latest_dev + 1) * step),
            "latest": _arrow.float64(latest),
            "ultimate": _arrow.float64(ultimate),
            "ibnr": _arrow.float64(ibnr),
            "model_ibnr": _arrow.float64(model_ibnr),
        }
    )
    _require_finite(periods, {"ultimate": ultimate, "ibnr": ibnr, "model_ibnr": model_ibnr}, [], {})
    _require_finite_tables(development, cells, coefficients, totals)
    return ReserveResult(
        "tweedie_glm",
        fit.as_of,
        step,
        origins,
        development,
        None,
        totals,
        cells=cells,
        coefficients=coefficients,
    )


def _require_finite_tables(*tables: pa.Table) -> None:
    """Refuse a result whose float columns hold a number that is not finite.

    Every missing number is a null by the time this runs, so a NaN or an
    infinity here comes from amounts too large (or too small) for the fit's
    squares and sums, such as the deviance of amounts near the largest double.
    """
    for table in tables:
        for name, column in zip(table.column_names, table.columns, strict=True):
            if not pa.types.is_floating(column.type):
                continue
            present = column.drop_null()
            if len(present) and not pc.all(pc.is_finite(present)).as_py():
                raise Refusal(
                    "result_not_finite",
                    f"the fit's {name} is not a finite number: the amounts are too large, or "
                    "too far apart, for its squares and sums to stay finite. Check them for a "
                    "unit error, or scale them (work in thousands, say) and scale the answer "
                    "back",
                    option="cells",
                )


# -- machine-learning development ----------------------------------------------------


def ml_development(
    cells,
    *,
    estimator: str,
    seed: int = 0,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    learning_rate: float | None = None,
    response: str = "incremental",
    origin: str = "factor",
    calendar: str = "none",
    dev_grain_months: int = 12,
    zero_cells: str = "observed",
    unsupported_factor: str = "raise",
    tail=None,
) -> ReserveResult:
    """A random forest or gradient boosting fitted to the cells, and each origin's ultimate.

    This is chainladder-python's ``DevelopmentML`` followed by its
    ``Chainladder``, without pandas or patsy. ``cells`` is as in
    :func:`chain_ladder`: CUMULATIVE losses, one row per observed cell, zero or
    more. The function fits a scikit-learn tree ensemble to one row per cell
    (its increment by default) with an indicator for each development age and
    each origin, predicts every cell of the rectangle, and projects each
    origin's latest cumulative by its own fitted development from its latest
    age to the last: ``ultimate = latest * fitted_cumulative[last age] /
    fitted_cumulative[latest age]``. It needs scikit-learn, which the ``ml``
    extra installs (``pip install "ibnr[ml]"``); importing ``ibnr.methods``
    does not load it, and the first call does, with the scipy and pandas it
    loads.

    Read these before using the answer:

    - It is a point estimate, with no standard error or distribution.
    - **Tree models projected this way usually overshoot the chain ladder**,
      often by a lot, and that comes from the method, not from a defect: the
      youngest origin has one training row, the trees split it off by its
      origin indicator, and its predicted increment stays near that one value
      at every later age. On a 10 x 10 Schedule P paid triangle the forest
      (seed 42) gives 2.7 times the chain ladder's IBNR and boosting 1.8
      times; on MW2014 the forest gives 40 times. It is not a rule: on GenIns
      the forest gives 0.93 times and boosting 0.87 times.
    - A random forest's answer depends on ``seed`` (from 1.5% to 3.6% of the
      total, as the standard deviation over 20 seeds on six public
      triangles). Gradient boosting's does not with these settings: it looks
      at every feature at every split and uses every row.
    - ``response="cumulative"`` gave a negative total IBNR with the forest on
      all six test triangles, because the fitted cumulatives fall with age.

    Options:

    - ``estimator``: ``"random_forest"`` or ``"gradient_boosting"``, required
      (the two differ by 52% on the Schedule P triangle above). For a Tweedie
      GLM, chainladder-python's third estimator, use :func:`tweedie_glm`.
    - ``seed``: passed to scikit-learn as ``random_state``, unchanged, so the
      same integer gives chainladder-python's numbers. A whole number from 0 to
      4294967295; 0 by default. ``None`` is refused, because a forest with no
      seed gives a different answer on every call.
    - ``n_estimators`` (100), ``max_depth`` (``None``: unlimited for the
      forest, 3 for boosting), ``min_samples_leaf`` (the forest only;
      ``None`` is 1) and ``learning_rate`` (boosting only; ``None`` is 0.1):
      scikit-learn's settings of the same names, the others at scikit-learn's
      defaults. A setting that does not apply to the estimator is refused
      rather than ignored.
    - ``response``: ``"incremental"`` (the default) fits each cell's
      increment, the first age's being its cumulative; ``"cumulative"`` fits
      the cumulative.
    - ``origin``: ``"factor"`` (the default) gives the model an indicator per
      origin; ``"none"`` leaves origins out, so every origin shares one
      fitted pattern.
    - ``calendar``: ``"none"`` (the default) or ``"trend"``, the calendar
      period (in development steps) as one numeric column. These two are
      :func:`tweedie_glm`'s design words; the four designs are
      chainladder-python's ``C(development) + C(origin)`` (the default),
      ``C(development)``, ``C(development) + valuation`` and
      ``C(development) + C(origin) + valuation``.
    - ``zero_cells``: ``"observed"`` (the default here, unlike the link-ratio
      methods) trains on every observed cell, a zero cumulative included.
      ``"missing"`` leaves out every cell whose cumulative is zero, as
      chainladder-python does. The default differs because under
      ``"missing"`` an age whose cells are all zero leaves the model nothing
      to fit there, which happens on 208 of the 722 clrd paid triangles with
      losses; under ``"observed"`` it never happens.
    - ``unsupported_factor``: under ``zero_cells="missing"``, what to do at an
      age whose cells are all zero: ``"raise"`` (the default) refuses;
      ``"unity"`` develops by a factor of 1 into it (a fitted increment of 0).
      ``development.unity_fallback`` marks the factor INTO such an age, on
      the row of the age before it, so an all-zero first age has no row to
      show on and shows only as ``n_trained`` 0.
    - ``dev_grain_months``: as in :func:`chain_ladder`.
    - ``tail``: not supported yet; anything but ``None`` is refused. Each
      origin is projected to the last observed development age.

    Returns a :class:`ReserveResult` whose ``link_ratios`` and
    ``coefficients`` are ``None``:

    - ``origins``: ``origin``, ``origin_period``, ``latest_dev_lag``,
      ``latest``, ``ultimate`` and ``ibnr`` (the projection above),
      ``fitted_latest`` and ``fitted_ultimate`` (the fitted cumulative at the
      latest age and at the last), ``cdf`` (their ratio; 1.0 at the last age)
      and ``model_ibnr`` (the sum of the fitted future increments, which
      differs from ``ibnr`` wherever the fitted cumulative at the latest age
      differs from the actual one, for a tree model nearly always). These four
      are null for an origin the model cannot place: with ``origin="factor"``
      and ``zero_cells="missing"``, an origin whose cells are all zero, which
      has no training row (its ultimate is 0). ``cdf`` is null too where the
      fitted cumulative at the latest age is zero or less and the latest is 0.
    - ``development``: ``dev_lag``; ``factor``, ``cdf`` and ``pct_reported``
      when every origin shares one fitted pattern (as with ``origin="none"``
      and ``calendar="none"``), null otherwise, when each origin's own are in
      ``cells``; ``n_trained`` (int64, the training rows at the age); and
      ``unity_fallback`` (bool, whether the factor from this age to the next
      is the 1 of ``unsupported_factor="unity"``; null at the last age).
    - ``cells``: one row per cell of the full rectangle, observed or not:
      ``origin``, ``origin_period``, ``dev_lag``, ``observed`` and ``trained``
      (bool), ``increment`` (null when not observed), ``fitted_increment``,
      ``fitted_cumulative``, ``factor`` (the origin's fitted factor to the
      next age; null at the last age or where the fitted cumulative is zero or
      less) and ``cdf`` (to the last age; null where the fitted cumulative is
      zero or less). The fitted columns are null for an origin the model
      cannot place.
    - ``totals``: ``latest``, ``ultimate``, ``ibnr``, ``model_ibnr``, and the
      settings that produced the answer: ``estimator``, ``seed``,
      ``n_estimators``, ``max_depth`` (null for an unlimited forest),
      ``min_samples_leaf`` (null for boosting), ``learning_rate`` (null for the
      forest), ``response``, ``origin_term``, ``calendar_term`` and
      ``zero_cells`` (strings), ``n_training_rows`` and
      ``scikit_learn_version``, because boosting's last digits changed between
      scikit-learn 1.6.1 and 1.9.0.

    Refused, besides the refusals of :func:`chain_ladder`'s ``cells``: a
    setting outside its values or one that does not apply to the estimator
    (``invalid_option``); a triangle whose cells are all zero, as
    :func:`tweedie_glm` refuses it, and fewer than 2 cells to fit (both
    ``not_identified``); under ``zero_cells="missing"`` with
    ``unsupported_factor="raise"``, an age whose cells are all zero
    (``no_link_ratio``); fitted values that are not finite numbers, from
    amounts near the largest double (``result_not_finite``); a
    still-developing origin with losses whose fitted cumulative at its latest
    age is zero or less, so its factor to ultimate is undefined, and an
    ultimate below zero (both ``negative_projection``), in that order; and a
    tail (``not_supported``). Negative increments are fitted like any others.
    ``docs/coming-from-chainladder.md`` lists where the answers differ from
    chainladder-python's.
    """
    with _CallersTerms("ml_development") as terms:
        grid, labels = terms.read(cells, dev_grain_months)
        spec = MLDevelopmentSpec(
            estimator=estimator,
            seed=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            response=response,
            origin=origin,
            calendar=calendar,
            zero_cells=zero_cells,
            unsupported_factor=unsupported_factor,
        )
        if tail is not None:
            raise Refusal(
                "not_supported",
                "ml_development takes no tail yet: each origin is projected to the last observed "
                "development age. Pass tail=None",
                option="tail",
                given=tail,
            )
        # Amounts near the largest double: a sum of fitted values can overflow,
        # and a number that is not finite is refused by name below.
        with np.errstate(all="ignore"):
            fit = fit_ml_development_grid(grid, spec)
        return _ml_result(fit, labels)


def _ml_result(fit: MLDevelopmentFit, labels: _Origins) -> ReserveResult:
    step = fit.dev_grain_months
    n_w, n_d = fit.cumulative.shape
    periods = fit.origin_periods
    observed, placed = fit.observed, fit.placed
    unplaced = ~placed
    with np.errstate(all="ignore"):
        latest, ultimate = fit.latest, fit.ultimate
        ibnr = ultimate - latest
        model_ibnr = fit.model_ibnr
        fitted, fitted_cum = fit.fitted, fit.fitted_cumulative
        factors, cdf = fit.factors, fit.cdf
        sums = {
            **_sums(latest, ultimate),
            "model_ibnr": _arrow.float64([float(model_ibnr[placed].sum())]),
        }
    ages = np.arange(1, n_d + 1, dtype=np.int64) * step

    # the development table: one pattern for every origin, when there is one
    reference = fit.pattern_cumulative
    if reference is not None:
        live = reference > 0
        with np.errstate(all="ignore"):
            pattern_cdf = np.where(live, reference[-1] / reference, 0.0)
            pattern_factor = np.r_[np.where(live[:-1], reference[1:] / reference[:-1], 0.0), 0.0]
            pattern_pct = np.where(live, 1.0 / pattern_cdf, 0.0)
        no_cdf = ~live
        no_factor = np.r_[~live[:-1], True]
    else:
        pattern_cdf = pattern_factor = pattern_pct = np.zeros(n_d)
        no_cdf = no_factor = np.ones(n_d, dtype=bool)
    development = pa.table(
        {
            "dev_lag": _arrow.int64(ages),
            "factor": _nullable(pattern_factor, no_factor),
            "cdf": _nullable(pattern_cdf, no_cdf),
            "pct_reported": _nullable(pattern_pct, no_cdf),
            "n_trained": _arrow.int64(fit.trained.sum(axis=0)),
            "unity_fallback": _with_last_null(fit.unity_ages[1:], pa.bool_()),
        }
    )

    # the cells table: every cell of the rectangle, origin by origin
    cell_periods = [periods[i] for i in range(n_w) for _ in range(n_d)]
    no_fit = np.repeat(unplaced, n_d)
    cell_factor = np.c_[factors, np.zeros(n_w)]
    no_cell_factor = np.c_[np.isnan(factors), np.ones(n_w, dtype=bool)]
    cells = pa.table(
        {
            "origin": labels.labels_for(cell_periods),
            "origin_period": _arrow.date32(cell_periods),
            "dev_lag": _arrow.int64(np.tile(ages, n_w)),
            "observed": _arrow.bool_(observed.ravel()),
            "trained": _arrow.bool_(fit.trained.ravel()),
            "increment": _nullable(fit.increments.ravel(), ~observed.ravel()),
            "fitted_increment": _nullable(fitted.ravel(), no_fit),
            "fitted_cumulative": _nullable(fitted_cum.ravel(), no_fit),
            "factor": _nullable(cell_factor.ravel(), no_cell_factor.ravel()),
            "cdf": _nullable(cdf.ravel(), np.isnan(cdf).ravel()),
        }
    )

    spec = fit.spec
    settings = spec.settings
    forest = spec.estimator == "random_forest"
    depth = settings["max_depth"]
    totals = pa.table(
        {
            **sums,
            "estimator": _arrow.string([spec.estimator]),
            "seed": _arrow.int64([spec.seed]),
            "n_estimators": _arrow.int64([settings["n_estimators"]]),
            "max_depth": _arrow.int64([depth or 0], mask=np.array([depth is None])),
            "min_samples_leaf": _arrow.int64(
                [settings["min_samples_leaf"] or 0], mask=np.array([not forest])
            ),
            "learning_rate": _nullable([settings["learning_rate"] or 0.0], [forest]),
            "response": _arrow.string([spec.response]),
            "origin_term": _arrow.string([spec.origin]),
            "calendar_term": _arrow.string([spec.calendar]),
            "zero_cells": _arrow.string([spec.zero_cells]),
            "n_training_rows": _arrow.int64([fit.n_training_rows]),
            "scikit_learn_version": _arrow.string([fit.sklearn_version]),
        }
    )
    origin_cdf = fit.origin_cdf
    origins = pa.table(
        {
            "origin": labels.labels_for(periods),
            "origin_period": _arrow.date32(periods),
            "latest_dev_lag": _arrow.int64((fit.latest_dev + 1) * step),
            "latest": _arrow.float64(latest),
            "ultimate": _arrow.float64(ultimate),
            "ibnr": _arrow.float64(ibnr),
            "fitted_latest": _nullable(fit.fitted_latest, unplaced),
            "fitted_ultimate": _nullable(fit.fitted_ultimate, unplaced),
            "cdf": _nullable(origin_cdf, np.isnan(origin_cdf)),
            "model_ibnr": _nullable(model_ibnr, unplaced),
        }
    )
    _require_finite(
        periods,
        {"ultimate": ultimate, "ibnr": ibnr, "model_ibnr": np.where(placed, model_ibnr, 0.0)},
        [],
        {},
    )
    _require_finite_tables(origins, development, cells, totals)
    return ReserveResult(
        "ml_development", fit.as_of, step, origins, development, None, totals, cells=cells
    )


# -- the one-year claims development result ---------------------------------------

#: The quantile levels :func:`one_year_cdr` reports when given none: the Reserving
#: app's default percentiles, 50 to 99.9, divided by 100.
_CDR_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999)

#: What the seed is combined with to make the random stream. They are the ones the
#: gallery's mack entry uses for a Triangle with one segment ``Total`` and the
#: field ``values`` (what ``Triangle.from_chainladder`` gives a one-column
#: chainladder Triangle named ``values``), so a seed gives the draws the
#: Reserving app's ``/cdr`` gave before it moved to this function.
_CDR_STREAM = {"label": "cdr_distribution", "cohorts": [{"Total": "Total"}], "field": "values"}

#: The most numbers :func:`one_year_cdr` draws: ``n_draws`` times the number of
#: origins. Measured on raa, each takes about 70 bytes at the peak (the draws,
#: their copies and the Arrow tables), so this is about 7 GB. The Reserving app
#: caps its own requests at 50,000 draws.
_MAX_DRAWN_NUMBERS = 100_000_000


@dataclass(frozen=True)
class OneYearCDRResult:
    """What :func:`one_year_cdr` returns: pyarrow Tables with fixed column types.

    A missing number is an Arrow null, never NaN. ``to_polars(name)`` gives any
    of the tables as a polars DataFrame.

    Every simulated number is about ``ultimate_change``, next year's
    re-estimated ultimate minus today's: positive when the reserve is
    strengthened, negative when it is released. That is the opposite sign of
    the claims development result in ``kernels.simulate_one_year_cdr``, which
    is positive for a release.

    Attributes
    ----------
    method : str
        ``"one_year_cdr"``.
    as_of : datetime.date
        The information date, the evaluation date of the latest cell.
    dev_grain_months : int
        Months per development step, always 12.
    n_draws : int
        The number of simulated years.
    seed : int or None
        The seed as passed; ``None`` means the draws came from fresh entropy
        and cannot be made again.
    origins : pyarrow.Table
        One row per origin period: ``origin`` and ``origin_period`` (as in
        :class:`ReserveResult`), ``latest_dev_lag`` (int64), ``latest``,
        ``ultimate`` and ``ibnr`` (today's chain-ladder figures),
        ``mean_ultimate_change`` and ``sd_ultimate_change`` (the draws' mean,
        and their standard deviation with ``ddof=1``, null when ``n_draws`` is
        1), ``cdr_se`` (the Merz-Wuthrich standard error of the one-year
        claims development result) and ``runoff_se`` (Mack's standard error of
        the whole run-off, what :func:`mack` reports as ``mack_se`` under the
        same ``sigma_rule`` and ``zero_cells``), all float64. An origin at its
        last development age has a change of 0 in every draw.
    totals : pyarrow.Table
        One row: ``latest``, ``ultimate``, ``ibnr``, ``mean_ultimate_change``,
        ``sd_ultimate_change``, ``cdr_se``, ``runoff_se`` (float64) and
        ``n_draws`` (int64), for the sum over the origins. The standard errors
        are not the sums of the origins': the origins share the factors.
    quantiles : pyarrow.Table
        One row per origin, or the total, and level: ``origin`` and
        ``origin_period`` (null for the total), ``level`` (the probability, as
        passed), ``ultimate_change`` (the quantile of the draws, numpy's
        linear rule, which is ``np.percentile`` with the level times 100) and
        ``tvar`` (the mean of the draws at or above that quantile). The total's
        rows come first, then each origin's in origin order, each in the order
        of the levels.
    draws : pyarrow.Table
        Every draw of every origin, ``n_draws`` times the number of origins
        rows, draw by draw: ``draw`` (int64, from 0), ``origin``,
        ``origin_period`` and ``ultimate_change`` (float64). A draw's total is
        the sum of its rows.
    """

    TABLES: ClassVar[tuple[str, ...]] = ("origins", "totals", "quantiles", "draws")

    method: str
    as_of: dt.date
    dev_grain_months: int
    n_draws: int
    seed: int | None
    origins: pa.Table
    totals: pa.Table
    quantiles: pa.Table
    draws: pa.Table

    def to_polars(self, table: str = "origins"):
        """One of the result's tables as a polars DataFrame.

        ``table`` is ``"origins"`` (the default), ``"totals"``, ``"quantiles"``
        or ``"draws"``. Needs polars: ``pip install "ibnr[polars]"``.
        """
        if table not in self.TABLES:
            raise Refusal(
                "invalid_option",
                f"table must be one of {self.TABLES}, got {{given}}",
                option="table",
                given=table,
                method=self.method,
            )
        return _to_polars(getattr(self, table))


def one_year_cdr(
    cells,
    *,
    dev_grain_months: int = 12,
    sigma_rule: str = "log_linear",
    zero_cells: str = "observed",
    n_draws: int = 20_000,
    seed: int | None = None,
    process: str = "gamma",
    parameter_risk: bool = True,
    quantiles=_CDR_QUANTILES,
) -> OneYearCDRResult:
    """The one-year claims development result: how far next year's re-estimate
    of the chain-ladder ultimate can move.

    ``cells`` is as in :func:`chain_ladder`. The chain ladder is Mack's:
    volume-weighted factors over every link ratio, with no development options
    and no tail, because the formulas below are derived for exactly that.

    Two answers come back, and they are meant to be compared:

    - the closed form of Merz and Wuthrich (2008), ``cdr_se``, per origin and
      in total, with Mack's run-off standard error ``runoff_se`` beside it;
    - a simulation of ``n_draws`` possible next years. Each draws next year's
      diagonal from Mack's conditional moments (with ``process`` as the
      shape of the noise, and, when ``parameter_risk`` is true, the factors
      drawn from their estimation error too), re-estimates the volume-weighted
      factors with that diagonal added, and reports the change in each
      origin's ultimate. Its standard deviation agrees with ``cdr_se`` to
      Monte Carlo error when ``parameter_risk`` is true, and it also gives
      quantiles and tail means, which a capital figure needs.

    ``ultimate_change`` is next year's ultimate minus today's, so positive is
    adverse (a strengthening); the kernels' claims development result has the
    opposite sign.

    Options:

    - ``dev_grain_months``: months per development step. The result is a
      one-year figure only on an annual triangle, so any other step is read and
      then refused: aggregate a quarterly or monthly triangle to annual first,
      or use :func:`mack` for the run-off standard error.
    - ``sigma_rule``: as in :func:`mack`; ``"log_linear"`` (the default, as in
      chainladder-python) or ``"mack"``. It moves the standard errors and the
      draws, not the ultimates.
    - ``zero_cells``: ``"observed"`` (the default here, unlike the other
      methods) keeps a zero cumulative as data. ``"missing"``, chainladder's
      rule, is accepted only where it changes nothing, on a triangle with no
      zero: the one-year formulas have not been checked with link ratios left
      out for a zero. Under either rule every still-developing origin needs a
      positive latest cumulative, because Mack's variance divides by it.
    - ``n_draws``: how many next years to simulate, a whole number of 1 or
      more (20,000 by default; the 99.5% quantile then rests on the top 100
      draws). ``n_draws`` times the number of origins numbers are held in
      memory and returned in ``draws``, about 70 bytes each at the peak, and
      that product is refused above 100,000,000 (about 7 GB).
    - ``seed``: a whole number of 0 or more makes the draws repeatable;
      ``None`` (the default) draws from fresh entropy. The seed is turned into
      a stream the same way the gallery's ``mack`` entry does for a triangle
      with one segment ``Total`` and the field ``values``, so for the same
      seed, draw count and options the draws equal, bit for bit, the ones the
      Reserving app's ``/cdr`` route gave through that entry, except that an
      origin at its last age has 0.0 where the app's negated draws have -0.0.
      A seed passed straight to ``kernels.simulate_one_year_cdr`` is a
      different stream.
    - ``process``: the shape of next year's noise, ``"gamma"`` (the default),
      ``"lognormal"`` or ``"normal"``. All three have Mack's mean and
      variance; only the first two keep the cumulative positive.
    - ``parameter_risk``: ``True`` (the default) also draws the factors from
      their estimation error, which is the half of the one-year risk that
      comes from next year's factors moving; ``False`` leaves only the noise
      of next year's cells.
    - ``quantiles``: the levels of the ``quantiles`` table, probabilities
      strictly between 0 and 1 (divide a percentile by 100). The default is
      0.5, 0.75, 0.9, 0.95, 0.99, 0.995 and 0.999.

    Returns a :class:`OneYearCDRResult`. Input it will not answer is refused
    with :class:`Refusal`, as the module docstring describes; besides the
    refusals of :func:`mack`, a step that is not twelve months and a zero
    under ``zero_cells="missing"`` are refused with ``not_supported``, and a
    zero latest cumulative on a still-developing origin with
    ``variance_not_estimable``. The Merz-Wuthrich figures tie out to R's
    ``ChainLadder`` (``CDR(MackChainLadder(MW2014, est.sigma="Mack"))``) to 6
    decimals, per origin and in total.
    """
    with _CallersTerms("one_year_cdr") as terms:
        count = _draw_count(n_draws)
        _require_seed(seed)
        levels = _levels(quantiles)
        if not isinstance(process, str) or process not in PROCESS_LAWS:
            raise Refusal(
                "invalid_option",
                f"process must be one of {PROCESS_LAWS}, got {{given}}",
                option="process",
                given=process,
            )
        if not isinstance(parameter_risk, bool | np.bool_):
            raise Refusal(
                "invalid_option",
                "parameter_risk must be True or False, got {given}",
                option="parameter_risk",
                given=parameter_risk,
            )
        grid, labels = terms.read(cells, dev_grain_months)
        return _one_year_cdr(
            grid,
            labels,
            sigma_rule=sigma_rule,
            zero_cells=zero_cells,
            n_draws=count,
            seed=seed,
            process=str(process),
            parameter_risk=bool(parameter_risk),
            levels=levels,
        )


def _one_year_cdr(
    grid,
    labels: _Origins,
    *,
    sigma_rule: str,
    zero_cells: str,
    n_draws: int,
    seed: int | None,
    process: str,
    parameter_risk: bool,
    levels: tuple[float, ...],
) -> OneYearCDRResult:
    _, as_of = check_grid(grid)
    if n_draws * grid["n_w"] > _MAX_DRAWN_NUMBERS:
        raise Refusal(
            "invalid_option",
            f"n_draws times the number of origins must be at most {_MAX_DRAWN_NUMBERS:,}, and "
            f"{{given}} draws of {grid['n_w']} origins is more: each number takes about 70 bytes "
            "while the draws are summarised, so the limit is already about 7 GB. Ask for fewer "
            "draws",
            option="n_draws",
            given=n_draws,
        )
    step = grid["dev_grain_months"]
    if step != 12:
        # the kernels refuse this too, in words about Triangle methods
        raise Refusal(
            "not_supported",
            f"the one-year claims development result needs a 12-month development step, and "
            f"this triangle's is {step} months, so one step forward would be a {step}-month "
            "result reported as a year. Aggregate the cells to annual origins and ages (from "
            "a year-end valuation) first, or use methods.mack for the standard error of the "
            "whole run-off, which does not depend on the step",
            option="dev_grain_months",
            given=step,
        )
    zero_at = [
        RefusedCell(None, grid["origin_periods"][i], (int(j) + 1) * step, 0.0)
        for i, j in zip(*np.nonzero(grid["obs_mask"] & (grid["cum"] == 0)), strict=True)
    ]
    if zero_cells == "missing" and zero_at:
        # Checked before the fit, whose own refusals under this rule (a sigma left
        # with nothing to fill it from) would be about a rule this function does
        # not take on such a triangle. The kernels refuse this too, in words about
        # MackFit. With no zero, "missing" is the same fit as "observed".
        raise Refusal(
            "not_supported",
            "one_year_cdr takes zero_cells='missing' only when no cumulative is zero, and "
            "{cells} are zero: the one-year formulas and the re-estimate of next year's factors "
            "have not been checked with the link ratios that rule leaves out. zero_cells="
            "'observed' (the default here) keeps the zeros as data; methods.mack gives the "
            "run-off standard error under either rule",
            option="zero_cells",
            given=zero_cells,
            cells=zero_at,
        )
    fit = _checked_mack_fit(
        grid, sigma_rule=sigma_rule, zero_cells=zero_cells, method="one_year_cdr"
    )
    zero_latest = np.flatnonzero((fit.latest_dev < fit.n_d - 1) & (fit.latest == 0))
    if zero_latest.size:
        raise Refusal(
            "variance_not_estimable",
            "one_year_cdr needs every still-developing origin's latest cumulative to be "
            "positive, and it is zero for {cells}: Mack's variance and the one-year formula "
            "divide by that amount. methods.mack with zero_cells='missing' gives such an origin "
            "a run-off standard error of 0, and methods.chain_ladder gives the ultimates",
            option="cells",
            cells=[
                RefusedCell(None, fit.origin_periods[i], (int(fit.latest_dev[i]) + 1) * step, 0.0)
                for i in zero_latest
            ],
        )
    # Amounts too large or too small for the squares: numpy warns, and the checks
    # below refuse such an answer by name before any draw is made.
    with np.errstate(all="ignore"):
        analytic = _merz_wuthrich(fit)
        latest, ultimate = fit.latest, fit.ultimate
        per_origin = {
            "ultimate": ultimate,
            "ibnr": ultimate - latest,
            "cdr_se": np.sqrt(analytic.msep),
            "runoff_se": np.sqrt(analytic.runoff_msep),
        }
        sums = _sums(latest, ultimate)
        spread = {
            "cdr_se": _arrow.float64([np.sqrt(analytic.msep_total)]),
            "runoff_se": _arrow.float64([np.sqrt(analytic.runoff_msep_total)]),
        }
    # Underflow first: amounts near the smallest double also leave a one-year msep
    # NaN (0 / 0), which the finite check would call too large.
    _require_no_underflow(fit, analytic.runoff_msep)
    _require_finite(fit.origin_periods, per_origin, [], {**sums, **spread})

    stream = cohort_stream(seed, **_CDR_STREAM)
    with np.errstate(all="ignore"):
        cdr = _one_year_cdr_draws(
            fit,
            n_draws=n_draws,
            seed=stream,
            generator=None,
            process=process,
            parameter_risk=parameter_risk,
        )
        # Laid out as simulate_one_year_cdr's samples, the total last, and negated
        # once, so that every summary below is the float the gallery entry's draws
        # give when summarised the same way. Adding 0.0 turns the -0.0 that negating
        # a fully developed origin's 0.0 gives into 0.0 and leaves every other
        # number as it is.
        delta = -np.hstack([cdr, cdr.sum(axis=1, keepdims=True)]) + 0.0
    n_w = fit.n_w
    changes, total = delta[:, :n_w], delta[:, -1]
    # a draw that is not finite makes its origin's mean not finite, which
    # _require_finite refuses by name
    with np.errstate(all="ignore"):
        mean = [float(changes[:, i].mean()) for i in range(n_w)]
        sd = [float(changes[:, i].std(ddof=1)) if n_draws > 1 else 0.0 for i in range(n_w)]
        total_mean = float(total.mean())
        total_sd = float(total.std(ddof=1)) if n_draws > 1 else 0.0
        tails = [_tail(total, level) for level in levels] + [
            _tail(changes[:, i], level) for i in range(n_w) for level in levels
        ]
    no_sd = n_draws == 1
    _require_finite(
        fit.origin_periods,
        {"mean_ultimate_change": mean, "sd_ultimate_change": sd},
        np.array(tails, dtype=float).ravel(),
        {"mean": _arrow.float64([total_mean]), "sd": _arrow.float64([total_sd])},
    )

    periods = fit.origin_periods
    shown = labels.labels_for(periods)
    dates = _arrow.date32(periods)
    origins = pa.table(
        {
            "origin": shown,
            "origin_period": dates,
            "latest_dev_lag": _arrow.int64((fit.latest_dev + 1) * step),
            "latest": _arrow.float64(latest),
            "ultimate": _arrow.float64(ultimate),
            "ibnr": _arrow.float64(per_origin["ibnr"]),
            "mean_ultimate_change": _arrow.float64(mean),
            "sd_ultimate_change": _nullable(sd, np.full(n_w, no_sd)),
            "cdr_se": _arrow.float64(per_origin["cdr_se"]),
            "runoff_se": _arrow.float64(per_origin["runoff_se"]),
        }
    )
    totals = pa.table(
        {
            **sums,
            "mean_ultimate_change": _arrow.float64([total_mean]),
            "sd_ultimate_change": _nullable([total_sd], [no_sd]),
            **spread,
            "n_draws": _arrow.int64([n_draws]),
        }
    )
    # quantile rows: the total's (a null origin) first, then each origin's
    n_levels = len(levels)
    row_origin = np.r_[np.zeros(n_levels, dtype=np.int64), np.repeat(np.arange(n_w), n_levels)]
    is_total = np.r_[np.ones(n_levels, dtype=bool), np.zeros(n_w * n_levels, dtype=bool)]
    at = _arrow.int64(row_origin, mask=is_total)
    quantile_values = np.array(tails, dtype=float).reshape(-1, 2)
    quantile_table = pa.table(
        {
            "origin": shown.take(at),
            "origin_period": dates.take(at),
            "level": _arrow.float64(np.tile(levels, n_w + 1)),
            "ultimate_change": _arrow.float64(quantile_values[:, 0]),
            "tvar": _arrow.float64(quantile_values[:, 1]),
        }
    )
    each = _arrow.int64(np.tile(np.arange(n_w), n_draws))
    draws = pa.table(
        {
            "draw": _arrow.int64(np.repeat(np.arange(n_draws), n_w)),
            "origin": shown.take(each),
            "origin_period": dates.take(each),
            "ultimate_change": _arrow.float64(changes.ravel()),
        }
    )
    return OneYearCDRResult(
        "one_year_cdr", as_of, step, n_draws, seed, origins, totals, quantile_table, draws
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
    with np.errstate(all="ignore"):
        ibnr = ultimate - latest
        pattern = _pattern_numbers(fit.factors, grid["n_d"])
        # the kernel checks each origin; a sum of finite ultimates can still overflow
        sums = _sums(latest, ultimate)
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

    development = pa.table(
        {
            **_pattern(pattern, step),
            "n_selected": _with_last_null(summary("n_selected"), pa.int64()),
            "unity_fallback": _with_last_null(summary("unity_fallback"), pa.bool_()),
            "extreme_trimming_skipped": _with_last_null(
                summary("extreme_trimming_skipped"), pa.bool_()
            ),
            "bounds_skipped": _with_last_null(summary("bounds_skipped"), pa.bool_()),
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
        np.concatenate([*pattern, ratio[~np.isnan(ratio)]]),
        sums,
    )
    totals = pa.table(sums)
    return ReserveResult(name, fit.as_of, step, pa.table(columns), development, link_ratios, totals)


def _pattern_numbers(factors: np.ndarray, n_d: int) -> tuple[np.ndarray, ...]:
    """(factor, cdf, pct_reported): a factor per link, the others per observed age.

    ``factors`` has one entry per link, ``n_d - 1`` of them; there is no tail.
    Computed without warnings; the caller refuses a number that is not finite.
    """
    factors = np.asarray(factors, dtype=float)[: n_d - 1]
    with np.errstate(all="ignore"):
        cdf = np.r_[np.cumprod(factors[::-1])[::-1], 1.0]
        return factors, cdf, 1.0 / cdf


def _pattern(numbers: tuple[np.ndarray, ...], step: int) -> dict[str, pa.Array]:
    """dev_lag, factor, cdf and pct_reported, one row per observed age."""
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

    def labels_or_null(self, starts) -> pa.Array:
        """As :meth:`labels_for`, with a null where ``starts`` holds ``None``."""
        position = {start: i for i, start in enumerate(self.label_starts)}
        index = [0 if start is None else position[start] for start in starts]
        missing = np.array([start is None for start in starts], dtype=bool)
        return self.labels.take(_arrow.int64(index, mask=missing))

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
