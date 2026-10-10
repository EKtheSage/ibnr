"""A Tweedie GLM fitted to a triangle's increments by iteratively reweighted least squares.

Each observed incremental amount ``y[i, j]`` (origin ``i``, development step
``j``) has mean ``mu[i, j]`` and variance ``phi * mu[i, j] ** power``, and

- under the log link, ``log(mu[i, j]) = c + a[i] + b[j] + g * t[i, j]``;
- under the identity link, ``mu[i, j]`` is that same sum.

``a`` (origin factors) is in the model when ``origin="factor"``, ``b``
(development factors) always, and ``g`` (a linear calendar trend, ``t`` the
calendar period counted in development steps from the first origin's first
cell) when ``calendar="trend"``. The first origin and the first development
age with losses are the reference levels (``a = 0``, ``b = 0`` there), as in
R's ``glm`` with treatment contrasts.

``power`` names the distribution: 0 is the normal, 1 the over-dispersed
Poisson, between 1 and 2 the compound Poisson-gamma, 2 the gamma, and above 2
the other Tweedie distributions with a positive mean. No Tweedie distribution
has a power between 0 and 1. Power 1 with a log link and origin and
development factors reproduces the volume-weighted chain ladder, except where
an origin's losses start from zero (a cumulative of 0 followed by a positive
one): there the chain ladder keeps the origin at 0, and the GLM gives it a
level from its later cells.

The fit is Fisher scoring, the iteration R's ``glm.fit`` runs: each step solves
a weighted least-squares problem by QR. There is no penalty, so the answer
does not depend on the units of the amounts: the triangle is divided by its
largest increment before the fit and the fitted means multiplied back after.
The fit stops when no fitted mean on an observed cell moved by more than
``TOLERANCE`` (1e-10) times the largest increment, and a fit that has not
stopped within ``max_iter`` steps is refused, never returned.

Under the log link an origin or a development age whose observed increments
are all zero has a maximum-likelihood mean of exactly 0, which the log link
can only approach. Such rows (with origin factors) and columns are taken out
of the regression, fitted at exactly 0, and their coefficients reported as
missing with ``fitted_zero`` set. That is the limit of the fit, and matches the
chain ladder: a zero origin's ultimate is 0, and a zero column's factor is 1.

This module imports numpy and ``ibnr.errors`` only (and the grid check), so
``ibnr.methods`` can use it without loading pandas, scipy or ibis.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from dataclasses import dataclass

import numpy as np

from ibnr.errors import Refusal, RefusedCell
from ibnr.kernels.grid import check_grid

__all__ = [
    "CALENDAR_TERMS",
    "LINKS",
    "ORIGIN_TERMS",
    "PROJECTIONS",
    "TweedieFit",
    "TweedieSpec",
    "fit_tweedie_grid",
]

LINKS = ("log", "identity")
ORIGIN_TERMS = ("factor", "none")
CALENDAR_TERMS = ("none", "trend")
PROJECTIONS = ("pattern", "increments")

#: The fit stops when no fitted mean on an observed cell moved by more than
#: this share of the largest absolute increment in one step.
TOLERANCE = 1e-10
#: A fitted mean below this share of the largest absolute increment, on an
#: observed cell whose increment is zero or less, means the fit exists only in
#: the limit (under the log link) or has no answer (under the identity link at
#: a power above 0). A small mean on a small positive increment is an answer.
BOUNDARY = 1e-8
#: How many times a step is halved back towards the last valid one before the
#: fit gives up, as R's glm.fit does when a step leaves the valid means.
_HALVINGS = 50

_POWER_TEXT = (
    "power must be 0 (normal), 1 (over-dispersed Poisson), between 1 and 2 (compound "
    "Poisson-gamma), 2 (gamma) or above; no Tweedie distribution has a power between 0 and 1"
)


def _choose(name: str, value, choices: tuple[str, ...]) -> None:
    """Refuse a setting that is not one of its choices, naming it and its value."""
    if not isinstance(value, str) or value not in choices:
        listed = " or ".join(repr(choice) for choice in choices)
        raise Refusal(
            "invalid_option", f"{name} must be {listed}, got {{given}}", option=name, given=value
        )


@dataclass(frozen=True)
class TweedieSpec:
    """What to fit: the distribution, the link, the design and the projection.

    Attributes
    ----------
    power : float
        The Tweedie power: 0, or 1 and above (see the module docstring).
    link : str
        ``"log"`` (the default) or ``"identity"``.
    origin : str
        ``"factor"`` (the default): one level per origin. ``"none"``: no origin
        term, so every origin shares one set of expected increments.
    calendar : str
        ``"none"`` (the default) or ``"trend"``: a linear trend in the calendar
        period. Only with ``origin="none"``: with origin and development
        factors the trend is a sum of their columns and cannot be estimated.
        Under the log link with development factors a calendar trend fits the
        same means as a linear trend across origins, so do not read it as an
        inflation rate.
    projection : str
        How an origin's ultimate is made. ``"pattern"`` (the default) is the
        latest cumulative times the fitted pattern's development factor from
        the latest age to the last, as chainladder-python reports it.
        ``"increments"`` is the latest cumulative plus the fitted future
        increments, as R's ``glmReserve`` reports it. The two agree at power 1
        with a log link and origin factors. ``"increments"`` needs origin
        factors: without them every origin would get the same future amounts
        whatever its size.
    max_iter : int
        The most Fisher-scoring steps to take (100 by default).
    """

    power: float = 1.0
    link: str = "log"
    origin: str = "factor"
    calendar: str = "none"
    projection: str = "pattern"
    max_iter: int = 100

    def __post_init__(self) -> None:
        power = self.power
        if (
            not isinstance(power, numbers.Real)
            or isinstance(power, bool | np.bool_)
            or not math.isfinite(float(power))
            or (float(power) != 0 and float(power) < 1)
        ):
            raise Refusal(
                "invalid_option", _POWER_TEXT + ", got {given}", option="power", given=power
            )
        object.__setattr__(self, "power", float(power))
        _choose("link", self.link, LINKS)
        _choose("origin", self.origin, ORIGIN_TERMS)
        _choose("calendar", self.calendar, CALENDAR_TERMS)
        _choose("projection", self.projection, PROJECTIONS)
        steps = self.max_iter
        if (
            not isinstance(steps, int | np.integer)
            or isinstance(steps, bool | np.bool_)
            or steps < 1
        ):
            raise Refusal(
                "invalid_option",
                "max_iter must be a positive whole number of iterations, got {given}",
                option="max_iter",
                given=steps,
            )
        object.__setattr__(self, "max_iter", int(steps))
        if self.origin == "factor" and self.calendar == "trend":
            raise Refusal(
                "invalid_option",
                "a calendar trend cannot be told apart from origin and development factors: a "
                "cell's calendar period is its origin's position plus its age's position, so "
                "the trend's column is a sum of theirs. Pass origin='none' to fit a calendar "
                "trend",
                option="calendar",
                options=("calendar", "origin"),
                given=self.calendar,
            )
        if self.origin == "none" and self.projection == "increments":
            raise Refusal(
                "invalid_option",
                "with no origin term every origin gets the same future amounts whatever its "
                "size, so projection='increments' would not depend on the origin; use "
                "projection='pattern'",
                option="projection",
                options=("projection", "origin"),
                given=self.projection,
            )


@dataclass(frozen=True)
class TweedieFit:
    """A fitted Tweedie GLM on one triangle's increments.

    Arrays are ``(n_w, n_d)`` (origins by development ages) unless noted.
    Amounts are in the units of the input.

    Attributes
    ----------
    spec : TweedieSpec
    origin_periods : list of datetime.date
        The first day of each origin period.
    as_of : datetime.date
        The evaluation date of the latest cell.
    dev_grain_months : int
    cumulative : numpy.ndarray
        The cumulative amounts, NaN where unobserved.
    increments : numpy.ndarray
        The incremental amounts, NaN where unobserved.
    fitted : numpy.ndarray
        The fitted increment in every cell, observed or not; exactly 0.0 in an
        origin or age fitted at zero.
    latest : numpy.ndarray
        ``(n_w,)`` each origin's latest cumulative.
    latest_dev : numpy.ndarray
        ``(n_w,)`` the 0-based development index of each origin's latest cell.
    terms : tuple
        ``(kind, index)`` per coefficient: ``("intercept", None)``,
        ``("origin", i)``, ``("development", j)`` or ``("calendar", None)``.
        The reference origin and age have no term.
    coef : numpy.ndarray
        The estimates on the link scale; NaN for a term fitted at zero.
    coef_se : numpy.ndarray
        ``sqrt(dispersion * diag((X'WX)^-1))``; NaN for a term fitted at zero
        or when the dispersion is undefined.
    fitted_zero : numpy.ndarray
        Bool per term: the origin or age was fitted at exactly zero.
    deviance, pearson_chi2 : float
        Summed over the observed cells in the regression.
    dispersion : float
        ``pearson_chi2 / (n_obs - n_params)``; NaN when ``n_obs == n_params``.
    n_obs : int
        Observed cells in the regression (cells of an origin or age fitted at
        zero are not counted).
    n_params : int
        Estimated coefficients.
    iterations : int
        Fisher-scoring steps taken.
    """

    spec: TweedieSpec
    origin_periods: list[dt.date]
    as_of: dt.date
    dev_grain_months: int
    cumulative: np.ndarray
    increments: np.ndarray
    fitted: np.ndarray
    latest: np.ndarray
    latest_dev: np.ndarray
    terms: tuple[tuple[str, int | None], ...]
    coef: np.ndarray
    coef_se: np.ndarray
    fitted_zero: np.ndarray
    deviance: float
    pearson_chi2: float
    dispersion: float
    n_obs: int
    n_params: int
    iterations: int

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.cumulative)

    @property
    def fitted_cumulative(self) -> np.ndarray:
        return np.cumsum(self.fitted, axis=1)

    @property
    def factors(self) -> np.ndarray:
        """``(n_w, n_d - 1)`` each origin's fitted factor from one age to the next.

        NaN where the fitted cumulative at the earlier age is 0.
        """
        cum = self.fitted_cumulative
        with np.errstate(all="ignore"):
            return np.where(cum[:, :-1] != 0, cum[:, 1:] / cum[:, :-1], np.nan)

    @property
    def cdf(self) -> np.ndarray:
        """Each origin's fitted factor from each age to the last; NaN where the
        fitted cumulative is 0."""
        cum = self.fitted_cumulative
        with np.errstate(all="ignore"):
            return np.where(cum != 0, cum[:, -1:] / cum, np.nan)

    @property
    def common_pattern(self) -> bool:
        """Whether every origin's fitted factors agree to 1e-12 relative.

        Always true under the log link, where ``exp(a[i])`` separates from the
        development terms; usually false under the identity link.
        """
        factors = self.factors
        defined = ~np.isnan(factors).any(axis=0)
        if factors.size == 0 or not defined.any():
            return True
        rows = factors[:, defined]
        rows = rows[~np.isnan(rows).any(axis=1)]
        return bool(np.all(np.abs(rows - rows[0]) <= 1e-12 * np.abs(rows[0])))

    @property
    def pattern(self) -> np.ndarray | None:
        """``(n_d - 1,)`` the common fitted factors under the log link (NaN from
        an age whose fitted cumulative is 0), or None under the identity link."""
        if self.spec.link != "log":
            return None
        cum = self.fitted_cumulative
        live = np.flatnonzero(cum[:, -1] > 0)
        if not live.size:
            return np.full(cum.shape[1] - 1, np.nan)
        return self.factors[live[0]]

    @property
    def future(self) -> np.ndarray:
        """Bool, the cells past each origin's latest age."""
        n_d = self.cumulative.shape[1]
        return np.arange(n_d)[None, :] > self.latest_dev[:, None]

    @property
    def model_ibnr(self) -> np.ndarray:
        """``(n_w,)`` the sum of each origin's fitted future increments."""
        return np.where(self.future, self.fitted, 0.0).sum(axis=1)

    @property
    def ultimate(self) -> np.ndarray:
        """``(n_w,)`` each origin's ultimate under ``spec.projection``."""
        if self.spec.projection == "increments":
            return self.latest + self.model_ibnr
        return _pattern_ultimate(self.fitted_cumulative, self.latest, self.latest_dev)

    @property
    def pearson_residuals(self) -> np.ndarray:
        """``(y - mu) / sqrt(mu ** power)``, unscaled by the dispersion; NaN where
        unobserved or where the fitted mean is 0. Written ``mu ** (power / 2)``,
        so a mean near the smallest double does not underflow to 0 when raised
        to the full power."""
        mu, y = self.fitted, self.increments
        with np.errstate(all="ignore"):
            out = (y - mu) / np.abs(mu) ** (self.spec.power / 2.0)
        return np.where(np.isnan(y) | (mu == 0), np.nan, out)


def _pattern_ultimate(fitted_cum: np.ndarray, latest: np.ndarray, latest_dev: np.ndarray):
    """The latest cumulative times the fitted pattern from the latest age to the last.

    An origin whose fitted cumulative at its latest age is 0 has nothing to
    scale; its latest amount is 0 there (see :func:`fit_tweedie_grid`), and so
    is its ultimate.
    """
    rows = np.arange(latest.size)
    at_latest = fitted_cum[rows, latest_dev]
    with np.errstate(all="ignore"):
        ratio = np.where(at_latest != 0, fitted_cum[:, -1] / at_latest, 1.0)
    return latest * ratio


def fit_tweedie_grid(grid: dict, spec: TweedieSpec | None = None) -> TweedieFit:
    """Fit a Tweedie GLM to the increments of a cumulative grid.

    ``grid`` is the dict ``kernels.cohort_grid_frame`` builds (cumulative
    amounts, one run-off triangle), checked by ``kernels.grid.check_grid``.
    ``spec`` is a :class:`TweedieSpec` (its defaults: power 1, log link,
    origin and development factors, the pattern projection).

    Refused, each with ``ibnr.errors.Refusal``: a negative increment at power 1
    or above (``negative_increment``); a zero increment at power 2 or above
    (``zero_increment``); no losses at all, or a design whose terms the
    observed cells cannot tell apart (``not_identified``); a fit that did not
    settle within ``spec.max_iter`` steps (``did_not_converge``); a fit that
    exists only in the limit, where a log-link fitted mean on an observed cell
    whose increment is zero or less falls to ``BOUNDARY`` times the largest
    increment or the weighted design loses rank while iterating
    (``degenerate_fit``; at power 0 the message names the ages and origins whose
    increments sum to zero or less, which the log link cannot fit); an
    identity-link fitted mean of zero or below, observed or future, at a power
    above 0, including one that falls to ``BOUNDARY`` times the largest
    increment on an observed zero increment, so the verdict does not depend on
    the units (``negative_fitted_mean``); and an ultimate below zero for an
    origin whose cumulatives are zero or more (``negative_projection``).
    """
    spec = TweedieSpec() if spec is None else spec
    if not isinstance(spec, TweedieSpec):
        raise TypeError(f"spec must be a TweedieSpec, got {type(spec).__name__}")
    origins, as_of = check_grid(grid)
    step = int(grid["dev_grain_months"])
    cum = np.asarray(grid["cum"], dtype=float)
    observed = np.asarray(grid["obs_mask"], dtype=bool)
    latest_dev = np.asarray(grid["latest_dev"], dtype=np.int64)
    n_w, n_d = cum.shape
    inc = cum.copy()
    inc[:, 1:] = cum[:, 1:] - cum[:, :-1]
    rows = np.arange(n_w)
    latest = cum[rows, latest_dev]

    def cells_at(where: np.ndarray) -> list[RefusedCell]:
        return [
            RefusedCell(None, origins[i], int(j + 1) * step, float(cum[i, j]))
            for i, j in np.argwhere(where)
        ]

    power = spec.power
    if not (observed & (inc != 0)).any():
        raise Refusal(
            "not_identified",
            "cells has no losses to fit: every observed increment is zero",
            option="cells",
        )
    negative = observed & (inc < 0)
    if power >= 1 and negative.any():
        raise Refusal(
            "negative_increment",
            "the cumulative falls at {cells}, so the increment there is negative; power "
            f"{_show_power(power)} needs increments of zero or more. power=0 (normal) accepts "
            "them, or use chain_ladder",
            option="cells",
            cells=cells_at(negative),
        )
    zero = observed & (inc == 0)
    if power >= 2 and zero.any():
        raise Refusal(
            "zero_increment",
            "the increment is zero at {cells}; power "
            f"{_show_power(power)} needs every increment above zero. A power between 1 and 2 "
            "(compound Poisson-gamma) accepts zeros",
            option="cells",
            cells=cells_at(zero),
        )

    # -- the design ---------------------------------------------------------------
    log = spec.link == "log"
    # Under the log link a row (with origin factors) or a column whose observed
    # increments are all zero is fitted at exactly zero, outside the regression.
    losses = observed & (inc != 0)
    zero_row = ~losses.any(axis=1) if log and spec.origin == "factor" else np.zeros(n_w, bool)
    zero_col = ~losses.any(axis=0) if log else np.zeros(n_d, bool)
    live_rows = np.flatnonzero(~zero_row)
    live_cols = np.flatnonzero(~zero_col)
    origin_step = np.array(
        [((o.year - origins[0].year) * 12 + o.month - origins[0].month) // step for o in origins],
        dtype=float,
    )

    terms: list[tuple[str, int | None]] = [("intercept", None)]
    term_zero: list[bool] = [False]
    if spec.origin == "factor":
        for i in range(n_w):
            if i != live_rows[0]:
                terms.append(("origin", i))
                term_zero.append(bool(zero_row[i]))
    for j in range(n_d):
        if j != live_cols[0]:
            terms.append(("development", j))
            term_zero.append(bool(zero_col[j]))
    if spec.calendar == "trend":
        terms.append(("calendar", None))
        term_zero.append(False)
    fitted_zero = np.array(term_zero, dtype=bool)
    live_terms = [t for t, z in zip(terms, term_zero, strict=True) if not z]

    def design(cell_rows: np.ndarray, cell_cols: np.ndarray) -> np.ndarray:
        columns = []
        for kind, index in live_terms:
            if kind == "intercept":
                columns.append(np.ones(cell_rows.size))
            elif kind == "origin":
                columns.append((cell_rows == index).astype(float))
            elif kind == "development":
                columns.append((cell_cols == index).astype(float))
            else:
                columns.append(origin_step[cell_rows] + cell_cols)
        return np.column_stack(columns)

    in_fit = observed & ~zero_row[:, None] & ~zero_col[None, :]
    fit_i, fit_j = np.nonzero(in_fit)
    x = design(fit_i, fit_j)
    n_obs, n_params = x.shape
    if np.linalg.matrix_rank(x) < n_params:
        _refuse_unidentified(x, live_terms, origins, step)

    # -- Fisher scoring, on the amounts divided by the largest increment -------------
    scale = float(np.max(np.abs(inc[observed])))
    y = inc[fit_i, fit_j] / scale
    stop = _Stopped(spec, y, (fit_i, fit_j), cells_at, origins, step, scale, (n_w, n_d))
    try:
        beta, mu, iterations = _irls(x, y, spec)
    except _Stalled as stalled:
        raise stop.refusal(stalled) from None
    # A fitted mean falling to zero where the increment is zero or less is the
    # sign that the fit exists only in the limit; a small mean fitted to a
    # small positive increment is not. The test is relative to the largest
    # increment, so the verdict does not depend on the units.
    limit = stop.falling(mu)
    if limit.any() and (log or power > 0):
        raise stop.limit_refusal(mu, limit)

    # -- the fitted rectangle -----------------------------------------------------------
    all_i, all_j = np.nonzero(~zero_row[:, None] & ~zero_col[None, :])
    eta_all = design(all_i, all_j) @ beta
    fitted = np.zeros((n_w, n_d))
    with np.errstate(over="ignore"):
        fitted[all_i, all_j] = (np.exp(eta_all) if log else eta_all) * scale
    if not log and power > 0:
        bad = fitted <= 0
        if bad.any():
            raise Refusal(
                "negative_fitted_mean",
                f"the identity link gave a fitted increment of zero or less at {{cells}}, which "
                f"power {_show_power(power)} cannot have; use link='log'",
                option="link",
                given=spec.link,
                cells=[
                    RefusedCell(None, origins[i], int(j + 1) * step) for i, j in np.argwhere(bad)
                ],
            )

    # -- standard errors, deviance, dispersion ------------------------------------------
    weights = _weights(mu, power, log)
    a = x * np.sqrt(weights)[:, None]
    unit = _unit_deviance(y, mu, power)
    with np.errstate(all="ignore"):
        pearson = float(np.sum((y - mu) ** 2 / np.abs(mu) ** power))
        # numpy's power, which gives infinity where Python's raises OverflowError;
        # a deviance past the largest double is refused by the front door by name
        units = np.float64(scale) ** (2.0 - power)
        deviance = float(np.sum(unit) * units)
        pearson_chi2 = float(pearson * units)
    dof = n_obs - n_params
    dispersion_scaled = pearson / dof if dof > 0 else math.nan
    dispersion = pearson_chi2 / dof if dof > 0 else math.nan
    if not np.isfinite(a).all() or np.linalg.matrix_rank(a) < n_params:
        raise stop.refusal(_Stalled("rank", mu))
    r_inv = np.linalg.inv(np.linalg.qr(a, mode="r"))
    variance = np.sum(r_inv**2, axis=1) * dispersion_scaled
    se_live = np.sqrt(variance)
    beta_out = beta.copy()
    if log:
        beta_out[0] += math.log(scale)  # the intercept carries the units
    else:
        beta_out *= scale
        se_live *= scale
    coef = np.full(len(terms), np.nan)
    coef_se = np.full(len(terms), np.nan)
    coef[~fitted_zero] = beta_out
    coef_se[~fitted_zero] = se_live

    fit = TweedieFit(
        spec=spec,
        origin_periods=origins,
        as_of=as_of,
        dev_grain_months=step,
        cumulative=cum,
        increments=np.where(observed, inc, np.nan),
        fitted=fitted,
        latest=latest,
        latest_dev=latest_dev,
        terms=tuple(terms),
        coef=coef,
        coef_se=coef_se,
        fitted_zero=fitted_zero,
        deviance=deviance,
        pearson_chi2=pearson_chi2,
        dispersion=dispersion,
        n_obs=int(n_obs),
        n_params=int(n_params),
        iterations=iterations,
    )
    _require_non_negative_ultimates(fit, cum, observed)
    return fit


def _show_power(power: float) -> str:
    return f"{power:g}"


def _weights(mu: np.ndarray, power: float, log: bool) -> np.ndarray:
    """Fisher-scoring weights ``(dmu/deta) ** 2 / mu ** power``."""
    with np.errstate(all="ignore"):
        if log:
            return mu ** (2.0 - power)
        return 1.0 / np.abs(mu) ** power if power else np.ones_like(mu)


def _valid(mu: np.ndarray, power: float, log: bool) -> np.ndarray:
    """Bool per cell: a mean the distribution can have."""
    ok = np.isfinite(mu)
    if log or power > 0:
        ok &= mu > 0
    return ok


class _Stalled(Exception):
    """Fisher scoring could not go on. Private: :func:`fit_tweedie_grid` turns it
    into a :class:`Refusal` once it can say why in the triangle's terms.

    ``why`` is ``"rank"`` (the weighted design lost rank or stopped being
    finite), ``"no_step"`` (no step, halved or not, keeps every mean valid;
    ``bad`` marks the means that were not) or ``"max_iter"`` (``change`` is the
    last step's largest move). ``mu`` is the fitted means when it stopped, on
    the scale of ``y``.
    """

    def __init__(self, why: str, mu: np.ndarray, *, bad=None, change: float = math.nan):
        super().__init__(why)
        self.why = why
        self.mu = mu
        self.bad = np.zeros(mu.shape, dtype=bool) if bad is None else bad
        self.change = change


def _irls(x, y, spec: TweedieSpec):
    """Fisher scoring; (coefficients, fitted means, steps), or :class:`_Stalled`.

    ``y`` is the increments divided by the largest absolute one. The start is
    halfway between each increment (a zero or negative one replaced by the mean
    of the positive ones) and their mean, which is positive for every triangle
    this was tried on, where R's own start fails on a negative increment under
    the log link. A step that leaves the valid means is halved back towards the
    last one, up to ``_HALVINGS`` times, as R's ``glm.fit`` does.
    """
    power, log = spec.power, spec.link == "log"
    positive = y[y > 0]
    filler = positive.mean() if positive.size else 1.0
    start = np.where(y > 0, y, filler)
    centre = y.mean() if y.mean() > 0 else start.mean()
    mu = (start + centre) / 2.0
    eta = np.log(mu) if log else mu.copy()
    beta = None
    change = math.inf
    for iteration in range(1, spec.max_iter + 1):
        derivative = mu if log else np.ones_like(mu)
        weights = _weights(mu, power, log)
        z = eta + (y - mu) / derivative
        root = np.sqrt(weights)
        a = x * root[:, None]
        if not np.isfinite(a).all() or np.linalg.matrix_rank(a) < x.shape[1]:
            raise _Stalled("rank", mu)
        q, r = np.linalg.qr(a)
        proposal = np.linalg.solve(r, q.T @ (root * z))
        new_eta, new_mu = _means(x, proposal, log)
        halvings = 0
        while not _valid(new_mu, power, log).all():
            if beta is None or halvings == _HALVINGS:
                raise _Stalled("no_step", mu, bad=~_valid(new_mu, power, log))
            proposal = (proposal + beta) / 2.0
            new_eta, new_mu = _means(x, proposal, log)
            halvings += 1
        change = float(np.max(np.abs(new_mu - mu)))
        beta, eta, mu = proposal, new_eta, new_mu
        if change <= TOLERANCE:
            return beta, mu, iteration
    raise _Stalled("max_iter", mu, change=change)


def _means(x, beta, log: bool):
    eta = x @ beta
    with np.errstate(over="ignore"):
        return eta, (np.exp(eta) if log else eta)


_RANK_LOST = (
    "while fitting, fitted means fell so close to zero, or grew so large, that the terms could "
    "no longer be estimated"
)

_ZEROS = "An origin or an age has only zeros where the model needs a positive level. "
_CHAIN_LADDER = "chain_ladder(cells, unsupported_factor='unity') answers it"


def _boundary(detail: str, cells: list[RefusedCell] = (), *, why: str = _ZEROS) -> Refusal:
    return Refusal(
        "degenerate_fit",
        f"this triangle's losses leave the model without a finite fit: {detail}. {why}"
        + _CHAIN_LADDER,
        option="cells",
        cells=cells,
    )


class _Stopped:
    """Why a fit stopped short, or reached a limit, as a :class:`Refusal` in the
    triangle's terms: cells, ages and origins, never regression rows."""

    def __init__(self, spec: TweedieSpec, y, cells, cells_at, origins, step, scale, shape):
        self.spec = spec
        self.log = spec.link == "log"
        self.y = y
        self.fit_i, self.fit_j = cells
        self.cells_at = cells_at
        self.origins = origins
        self.step = step
        self.scale = scale
        self.shape = shape

    def falling(self, mu: np.ndarray) -> np.ndarray:
        """Bool per regression cell: a mean below ``BOUNDARY`` where the increment
        is zero or less, the sign that the fit exists only in the limit."""
        return (self.y <= 0) & (mu < BOUNDARY)

    def _grid(self, flags: np.ndarray) -> np.ndarray:
        where = np.zeros(self.shape, dtype=bool)
        where[self.fit_i[flags], self.fit_j[flags]] = True
        return where

    def refusal(self, stalled: _Stalled) -> Refusal:
        power = self.spec.power
        if not self.log and power > 0:
            # the identity link reaches the boundary mu = 0 exactly; whether
            # rounding leaves a mean just above or just below it depends on the
            # units, so every route there gets the same refusal
            limit = self.falling(stalled.mu) | (stalled.bad & (self.y <= 0))
            if limit.any():
                return self.limit_refusal(stalled.mu, limit)
            if stalled.why == "no_step":
                return Refusal(
                    "negative_fitted_mean",
                    "the identity link gives a fitted increment of zero or less at {cells}, "
                    f"which power {_show_power(power)} cannot have, and no step of the fit "
                    "avoids it; use link='log'",
                    option="link",
                    given=self.spec.link,
                    cells=self._named(stalled.bad),
                )
        if stalled.why == "max_iter":
            return self._not_settled(stalled.change)
        if stalled.why == "no_step":
            return self._degenerate(
                "no step of the fit keeps the fitted mean at {cells} a finite number", stalled.bad
            )
        return self._degenerate(_RANK_LOST)

    def limit_refusal(self, mu: np.ndarray, limit: np.ndarray) -> Refusal:
        if self.log:
            return self._degenerate(
                f"the fitted mean at {{cells}} falls to {float(mu[limit].min()):.3g} times the "
                "largest increment",
                limit,
            )
        return Refusal(
            "negative_fitted_mean",
            "the identity link's fitted increment falls to zero at {cells}, where the increment "
            f"is 0, and power {_show_power(self.spec.power)} needs every fitted increment above "
            "zero, so the fit has no answer. link='log' fits an origin or an age whose "
            "increments are all 0 at exactly 0",
            option="link",
            given=self.spec.link,
            cells=self.cells_at(self._grid(limit)),
        )

    def _named(self, flags: np.ndarray) -> list[RefusedCell]:
        return [
            RefusedCell(None, self.origins[i], int(j + 1) * self.step)
            for i, j in zip(self.fit_i[flags], self.fit_j[flags], strict=True)
        ]

    def _falling_groups(self) -> tuple[list[int], list[int]]:
        """The ages, and (with origin factors) the origins, whose increments in the
        regression sum to zero or less. Only power 0 takes negative increments,
        and at a higher power an origin or age of zeros is out of the regression."""
        if not self.log or self.spec.power != 0:
            return [], []
        ages = [int(j) for j in np.unique(self.fit_j) if self.y[self.fit_j == j].sum() <= 0]
        rows = []
        if self.spec.origin == "factor":
            rows = [int(i) for i in np.unique(self.fit_i) if self.y[self.fit_i == i].sum() <= 0]
        return ages, rows

    def _groups_text(self, ages: list[int], rows: list[int]) -> str:
        parts = []
        if ages:
            parts.append(f"at {_and([str((j + 1) * self.step) for j in ages])} months")
        if rows:
            parts.append("of {origins}")
        return (
            f"The observed increments {' and those '.join(parts)} sum to zero or less, and "
            "under the log link every fitted increment is above zero"
        )

    def _degenerate(self, detail: str, flags: np.ndarray | None = None) -> Refusal:
        ages, rows = self._falling_groups()
        if not ages and not rows:
            named = [] if flags is None else self.cells_at(self._grid(flags))
            return _boundary(detail, named)
        # name the ages and origins that fall on balance; the cells the detail
        # would name are among theirs
        return _boundary(
            detail.replace(" at {cells}", ""),
            [RefusedCell(None, self.origins[i]) for i in rows],
            why=f"{self._groups_text(ages, rows)}. link='identity' fits them at power 0, or ",
        )

    def _not_settled(self, change: float) -> Refusal:
        text = (
            f"the fit did not settle within max_iter={self.spec.max_iter} iterations: the last "
            f"step moved a fitted increment by {change * self.scale:.6g}, {change * 100:.3g}% of "
            "the largest increment. Pass a larger max_iter"
        )
        ages, rows = self._falling_groups()
        if ages or rows:
            text += (
                f". {self._groups_text(ages, rows)}, which can leave the model without a finite "
                "fit; link='identity' fits them at power 0"
            )
        return Refusal(
            "did_not_converge",
            text,
            option="max_iter",
            given=self.spec.max_iter,
            cells=[RefusedCell(None, self.origins[i]) for i in rows],
        )


def _refuse_unidentified(x: np.ndarray, live_terms, origins, step) -> None:
    """Refuse a design whose columns are not independent, naming the terms involved."""
    _, singular, vt = np.linalg.svd(x)
    tol = singular.max() * max(x.shape) * np.finfo(float).eps
    rank = int((singular > tol).sum())
    null = vt[rank:]
    involved = [k for k in range(len(live_terms)) if np.any(np.abs(null[:, k]) > 1e-8)]
    named = [live_terms[k] for k in involved]
    kinds = [kind for kind, _ in named]
    ages = [str((index + 1) * step) for kind, index in named if kind == "development"]
    cells = [RefusedCell(None, origins[index]) for kind, index in named if kind == "origin"]
    parts = []
    if "intercept" in kinds:
        parts.append("the intercept")
    if cells:
        parts.append("the origin factors of {origins}")
    if ages:
        parts.append(f"the development factors at {_and(ages)} months")
    if "calendar" in kinds:
        parts.append("the calendar trend")
    raise Refusal(
        "not_identified",
        f"the observed cells leave {_and(parts)} undetermined: {x.shape[0]} cell(s) for "
        f"{x.shape[1]} terms, of which {rank} can be estimated. Fit fewer terms, or give the "
        "fit more cells",
        option="cells",
        cells=cells,
    )


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _unit_deviance(y: np.ndarray, mu: np.ndarray, power: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        if power == 0:
            return (y - mu) ** 2
        if power == 1:
            safe = np.where(y > 0, y, 1.0)
            return 2.0 * (np.where(y > 0, y * np.log(safe / mu), 0.0) - (y - mu))
        if power == 2:
            return 2.0 * (-np.log(y / mu) + (y - mu) / mu)
        return 2.0 * (
            np.maximum(y, 0.0) ** (2.0 - power) / ((1.0 - power) * (2.0 - power))
            - y * mu ** (1.0 - power) / (1.0 - power)
            + mu ** (2.0 - power) / (2.0 - power)
        )


def _require_non_negative_ultimates(fit: TweedieFit, cum: np.ndarray, observed) -> None:
    """Refuse an ultimate below zero for an origin whose cumulatives are zero or more."""
    ultimate = fit.ultimate
    non_negative = np.array(
        [bool((cum[i][observed[i]] >= 0).all()) for i in range(cum.shape[0])], dtype=bool
    )
    bad = np.flatnonzero(non_negative & (ultimate < 0))
    rows = np.arange(cum.shape[0])
    at_latest = fit.fitted_cumulative[rows, fit.latest_dev]
    # the pattern route has nothing to scale where the fitted cumulative is 0
    # or below and the latest amount is not
    no_pattern = np.flatnonzero(
        (fit.spec.projection == "pattern") & (at_latest <= 0) & (fit.latest != 0)
    )
    bad = np.union1d(bad, no_pattern)
    if bad.size:
        other = "increments" if fit.spec.projection == "pattern" else "pattern"
        raise Refusal(
            "negative_projection",
            "the fitted model projects {origins} below zero (or the fitted cumulative at the "
            f"latest age is zero or less, so the pattern has nothing to scale) with "
            f"projection={fit.spec.projection!r}, although the cumulatives are zero or more. "
            f"projection={other!r} or link='log' may answer",
            option="projection",
            given=fit.spec.projection,
            cells=[RefusedCell(None, fit.origin_periods[i]) for i in bad],
        )
