"""A random forest or gradient boosting fitted to a triangle's cells, and each origin's ultimate.

This is chainladder-python's ``DevelopmentML`` followed by its ``Chainladder``,
written in numpy around scikit-learn's two tree ensembles:

1. **Rows.** One training row per observed cell, origin by origin and, within
   an origin, by development age. Under ``zero_cells="missing"`` a cell whose
   cumulative is zero is left out, as chainladder-python leaves it out.
2. **Response.** The cell's increment (``response="incremental"``, the first
   age's increment being its cumulative) or its cumulative.
3. **Design.** A column of ones, then one indicator per development age after
   the first that has training rows, then (``origin="factor"``) one indicator
   per origin after the first that has training rows, then
   (``calendar="trend"``) the calendar period counted in development steps.
   This is patsy's treatment coding of ``C(development) + C(origin)`` (plus
   ``valuation``), column for column. A tree does not need the column of ones,
   but a random forest draws its features by position, so leaving it out or
   putting the rows in another order changes the forest's answer.
4. **Fitted rectangle.** The model predicts every cell of the rectangle,
   observed or not. Under the incremental response the fitted cumulative is
   the running sum of the fitted increments; under the cumulative response it
   is the prediction itself.
5. **Ultimate.** Each origin's latest cumulative times its own fitted
   development from its latest age to the last,
   ``latest * fitted_cum[last] / fitted_cum[latest age]``.

The seed goes to scikit-learn as ``random_state``, unchanged, so the same
integer gives chainladder-python's numbers. A random forest's answer depends
on it; gradient boosting's does not with these settings, because it looks at
every feature at every split and uses every row.

Tree models projected this way usually overshoot the chain ladder, often by
a lot: the youngest origin has one training row, trees split it off by its
origin indicator, and its predicted increment stays near that one value at
every later age. With seed 42 the forest's IBNR is 1.4 to 40 times the chain
ladder's on raa, UKMotor, ABC, MW2014 and a Schedule P triangle, and 0.93
times on GenIns (boosting 0.87 times there). That comes from the method, not
from a defect of this code.

This module imports numpy and ``ibnr.errors`` only (and the grid check), so
``ibnr.methods`` can import it without loading scikit-learn. scikit-learn,
which loads scipy and pandas, is imported inside :func:`fit_ml_development_grid`.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from dataclasses import dataclass

import numpy as np

from ibnr.errors import Refusal, RefusedCell
from ibnr.kernels.grid import ZERO_CELLS, check_grid

__all__ = [
    "CALENDAR_TERMS",
    "ESTIMATORS",
    "ORIGIN_TERMS",
    "RESPONSES",
    "SEED_MAX",
    "UNSUPPORTED",
    "MLDevelopmentFit",
    "MLDevelopmentSpec",
    "fit_ml_development_grid",
]

ESTIMATORS = ("random_forest", "gradient_boosting")
RESPONSES = ("incremental", "cumulative")
ORIGIN_TERMS = ("factor", "none")
CALENDAR_TERMS = ("none", "trend")
UNSUPPORTED = ("raise", "unity")
#: The largest integer scikit-learn takes as a ``random_state``.
SEED_MAX = 2**32 - 1

#: scikit-learn's own defaults, used when a setting is left as ``None``.
_FOREST_MIN_SAMPLES_LEAF = 1
_BOOSTING_MAX_DEPTH = 3
_BOOSTING_LEARNING_RATE = 0.1


def _choose(name: str, value, choices: tuple[str, ...]) -> None:
    """Refuse a setting that is not one of its choices, naming it and its value."""
    if not isinstance(value, str) or value not in choices:
        listed = " or ".join(repr(choice) for choice in choices)
        raise Refusal(
            "invalid_option", f"{name} must be {listed}, got {{given}}", option=name, given=value
        )


def _whole(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool | np.bool_)


def _positive_whole(name: str, value, *, none_means: str | None = None) -> int | None:
    """A positive whole number, or ``None`` where ``none_means`` says what that is."""
    if value is None and none_means is not None:
        return None
    if not _whole(value) or value < 1:
        after = f", or None ({none_means})" if none_means is not None else ""
        raise Refusal(
            "invalid_option",
            f"{name} must be a positive whole number{after}, got {{given}}",
            option=name,
            given=value,
        )
    return int(value)


@dataclass(frozen=True)
class MLDevelopmentSpec:
    """What to fit: the estimator, its settings, the response and the design.

    Attributes
    ----------
    estimator : str
        ``"random_forest"`` (scikit-learn's ``RandomForestRegressor``) or
        ``"gradient_boosting"`` (``GradientBoostingRegressor``). No default:
        the two answers are far apart (52% on one Schedule P triangle).
    seed : int
        Passed to scikit-learn as ``random_state``, unchanged: a whole number
        from 0 to ``SEED_MAX``. ``None`` is refused, because a forest with no
        seed gives a different answer on every call.
    n_estimators : int
        Trees in the forest, or boosting stages (100).
    max_depth : int or None
        The deepest a tree may grow; ``None`` is unlimited for the forest and
        3 for boosting, scikit-learn's defaults.
    min_samples_leaf : int or None
        The forest only: the fewest rows in a leaf; ``None`` is 1. Refused
        with ``gradient_boosting``, where it would change nothing here.
    learning_rate : float or None
        Boosting only: the shrinkage of each stage, above 0; ``None`` is 0.1.
        Refused with ``random_forest``, which has no such setting.
    response : str
        ``"incremental"`` (the default) models each cell's increment;
        ``"cumulative"`` models its cumulative.
    origin : str
        ``"factor"`` (the default): an indicator per origin. ``"none"``: no
        origin term, so every origin shares one fitted pattern.
    calendar : str
        ``"none"`` (the default) or ``"trend"``: the calendar period, counted
        in development steps from the first origin's first cell, as one
        numeric column (chainladder-python's ``valuation``).
    zero_cells : str
        ``"observed"`` (the default) trains on every observed cell, a zero
        cumulative included. ``"missing"`` leaves out every cell whose
        cumulative is zero, as chainladder-python does.
    unsupported_factor : str
        Under ``zero_cells="missing"``, what to do at a development age whose
        cells are all zero, which leaves the model nothing to fit there:
        ``"raise"`` (the default) refuses; ``"unity"`` develops by a factor of
        1 into that age (a fitted increment of 0).
    """

    estimator: str
    seed: int = 0
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_leaf: int | None = None
    learning_rate: float | None = None
    response: str = "incremental"
    origin: str = "factor"
    calendar: str = "none"
    zero_cells: str = "observed"
    unsupported_factor: str = "raise"

    def __post_init__(self) -> None:
        if not isinstance(self.estimator, str) or self.estimator not in ESTIMATORS:
            raise Refusal(
                "invalid_option",
                "estimator must be 'random_forest' or 'gradient_boosting', got {given}. For a "
                "Tweedie GLM use tweedie_glm",
                option="estimator",
                given=self.estimator,
            )
        seed = self.seed
        if not _whole(seed) or not 0 <= seed <= SEED_MAX:
            raise Refusal(
                "invalid_option",
                f"seed must be a whole number from 0 to {SEED_MAX}, got {{given}}; a random "
                "forest with no seed gives a different answer on every call",
                option="seed",
                given=seed,
            )
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "n_estimators", _positive_whole("n_estimators", self.n_estimators))
        unlimited = "unlimited" if self.estimator == "random_forest" else "3"
        object.__setattr__(
            self, "max_depth", _positive_whole("max_depth", self.max_depth, none_means=unlimited)
        )
        if self.min_samples_leaf is not None and self.estimator != "random_forest":
            raise Refusal(
                "invalid_option",
                "min_samples_leaf applies to random_forest only; leave it out for "
                "gradient_boosting, where it would change nothing",
                option="min_samples_leaf",
                options=("min_samples_leaf", "estimator"),
                given=self.min_samples_leaf,
            )
        object.__setattr__(
            self,
            "min_samples_leaf",
            _positive_whole("min_samples_leaf", self.min_samples_leaf, none_means="1"),
        )
        rate = self.learning_rate
        if rate is not None:
            if self.estimator != "gradient_boosting":
                raise Refusal(
                    "invalid_option",
                    "learning_rate applies to gradient_boosting only; leave it out for "
                    "random_forest, which has no such setting",
                    option="learning_rate",
                    options=("learning_rate", "estimator"),
                    given=rate,
                )
            if (
                not isinstance(rate, numbers.Real)
                or isinstance(rate, bool | np.bool_)
                or not math.isfinite(float(rate))
                or float(rate) <= 0
            ):
                raise Refusal(
                    "invalid_option",
                    "learning_rate must be a finite number above 0, or None (0.1), got {given}",
                    option="learning_rate",
                    given=rate,
                )
            object.__setattr__(self, "learning_rate", float(rate))
        _choose("response", self.response, RESPONSES)
        _choose("origin", self.origin, ORIGIN_TERMS)
        _choose("calendar", self.calendar, CALENDAR_TERMS)
        _choose("zero_cells", self.zero_cells, ZERO_CELLS)
        _choose("unsupported_factor", self.unsupported_factor, UNSUPPORTED)

    @property
    def settings(self) -> dict:
        """The estimator's settings as scikit-learn gets them, ``None`` filled in.

        ``max_depth`` stays ``None`` for an unlimited forest; ``min_samples_leaf``
        is ``None`` for boosting and ``learning_rate`` for the forest.
        """
        forest = self.estimator == "random_forest"
        return {
            "n_estimators": self.n_estimators,
            "max_depth": (
                self.max_depth if forest or self.max_depth is not None else _BOOSTING_MAX_DEPTH
            ),
            "min_samples_leaf": (
                (self.min_samples_leaf or _FOREST_MIN_SAMPLES_LEAF) if forest else None
            ),
            "learning_rate": None if forest else (self.learning_rate or _BOOSTING_LEARNING_RATE),
        }


@dataclass(frozen=True)
class MLDevelopmentFit:
    """A fitted tree model on one triangle, and each origin's projection.

    Arrays are ``(n_w, n_d)`` (origins by development ages) unless noted.

    Attributes
    ----------
    spec : MLDevelopmentSpec
    origin_periods : list of datetime.date
        The first day of each origin period.
    as_of : datetime.date
        The evaluation date of the latest cell.
    dev_grain_months : int
    cumulative : numpy.ndarray
        The cumulative amounts, NaN where unobserved.
    trained : numpy.ndarray
        Bool: the cells that were training rows.
    fitted : numpy.ndarray
        The fitted increment in every cell (the difference of the fitted
        cumulative under ``response="cumulative"``); NaN for every cell of an
        origin the model cannot place (``origin="factor"`` and no training
        row), and 0.0 at an age in ``unity_ages``.
    fitted_cumulative : numpy.ndarray
        The fitted cumulative in every cell; NaN where ``fitted`` is.
    latest : numpy.ndarray
        ``(n_w,)`` each origin's latest cumulative.
    latest_dev : numpy.ndarray
        ``(n_w,)`` the 0-based development index of each origin's latest cell.
    unity_ages : numpy.ndarray
        ``(n_d,)`` bool: ages with no training row that
        ``unsupported_factor="unity"`` develops by a factor of 1.
    n_training_rows : int
    feature_names : tuple of str
        The design's columns in order: ``"intercept"``, ``"development:24"``
        (months), ``"origin:1989-01-01"`` (the period's first day),
        ``"calendar"``.
    sklearn_version : str
        The scikit-learn version that fitted it.
    """

    spec: MLDevelopmentSpec
    origin_periods: list[dt.date]
    as_of: dt.date
    dev_grain_months: int
    cumulative: np.ndarray
    trained: np.ndarray
    fitted: np.ndarray
    fitted_cumulative: np.ndarray
    latest: np.ndarray
    latest_dev: np.ndarray
    unity_ages: np.ndarray
    n_training_rows: int
    feature_names: tuple[str, ...]
    sklearn_version: str

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.cumulative)

    @property
    def increments(self) -> np.ndarray:
        """The observed increments (the first age's is its cumulative), NaN elsewhere."""
        cum = self.cumulative
        out = cum.copy()
        out[:, 1:] = cum[:, 1:] - cum[:, :-1]
        return out

    @property
    def placed(self) -> np.ndarray:
        """``(n_w,)`` bool: origins the model gives fitted values."""
        return ~np.isnan(self.fitted_cumulative).any(axis=1)

    @property
    def factors(self) -> np.ndarray:
        """``(n_w, n_d - 1)`` each origin's fitted factor from one age to the next;
        NaN where the fitted cumulative at the earlier age is zero or less."""
        cum = self.fitted_cumulative
        with np.errstate(all="ignore"):
            return np.where(cum[:, :-1] > 0, cum[:, 1:] / cum[:, :-1], np.nan)

    @property
    def cdf(self) -> np.ndarray:
        """Each origin's fitted factor from each age to the last; NaN where the
        fitted cumulative is zero or less."""
        cum = self.fitted_cumulative
        with np.errstate(all="ignore"):
            return np.where(cum > 0, cum[:, -1:] / cum, np.nan)

    @property
    def fitted_latest(self) -> np.ndarray:
        """``(n_w,)`` the fitted cumulative at each origin's latest age."""
        return self.fitted_cumulative[np.arange(self.latest.size), self.latest_dev]

    @property
    def fitted_ultimate(self) -> np.ndarray:
        """``(n_w,)`` the fitted cumulative at the last age."""
        return self.fitted_cumulative[:, -1]

    @property
    def origin_cdf(self) -> np.ndarray:
        """``(n_w,)`` each origin's factor from its latest age to the last.

        1.0 for an origin at the last age; NaN where the fitted cumulative at
        the latest age is zero or less, or the origin has no fitted values.
        """
        at_last = self.latest_dev == self.cumulative.shape[1] - 1
        with np.errstate(all="ignore"):
            ratio = np.where(
                self.fitted_latest > 0, self.fitted_ultimate / self.fitted_latest, np.nan
            )
        return np.where(at_last & self.placed, 1.0, ratio)

    @property
    def ultimate(self) -> np.ndarray:
        """``(n_w,)`` the latest cumulative times ``origin_cdf``; 0 where the latest is 0."""
        with np.errstate(all="ignore"):
            return np.where(self.latest == 0, 0.0, self.latest * self.origin_cdf)

    @property
    def future(self) -> np.ndarray:
        """Bool, the cells past each origin's latest age."""
        n_d = self.cumulative.shape[1]
        return np.arange(n_d)[None, :] > self.latest_dev[:, None]

    @property
    def model_ibnr(self) -> np.ndarray:
        """``(n_w,)`` the sum of each origin's fitted future increments; NaN for an
        origin with no fitted values."""
        return np.where(self.future, self.fitted, 0.0).sum(axis=1)

    @property
    def common_pattern(self) -> bool:
        """Whether every fitted origin's factors agree to 1e-12 relative.

        True with ``origin="none"`` and ``calendar="none"``, where every origin
        gets the same predictions; for a tree model with origin indicators it
        is almost never true.
        """
        factors = self.factors[self.placed]
        if factors.size == 0:
            return True
        defined = ~np.isnan(factors)
        if not (defined == defined[0]).all():
            return False
        rows = np.where(defined, factors, 0.0)
        return bool(np.all(np.abs(rows - rows[0]) <= 1e-12 * np.abs(rows[0])))

    @property
    def pattern_cumulative(self) -> np.ndarray | None:
        """``(n_d,)`` the first fitted origin's fitted cumulative when every
        origin shares one pattern (``common_pattern``), else None."""
        placed = np.flatnonzero(self.placed)
        if not self.common_pattern or not placed.size:
            return None
        return self.fitted_cumulative[placed[0]]


def fit_ml_development_grid(grid: dict, spec: MLDevelopmentSpec) -> MLDevelopmentFit:
    """Fit a random forest or gradient boosting to a cumulative grid's cells.

    ``grid`` is the dict ``kernels.cohort_grid_frame`` builds (cumulative
    amounts, one run-off triangle), checked by ``kernels.grid.check_grid``.
    ``spec`` is an :class:`MLDevelopmentSpec`.

    Refused, each with ``ibnr.errors.Refusal``: fewer than 2 training rows
    (``not_identified``); under ``zero_cells="missing"`` with
    ``unsupported_factor="raise"``, an age whose cells are all zero
    (``no_link_ratio``); a fitted value that is not a finite number, from
    amounts near the largest double (``result_not_finite``); a
    still-developing origin with losses whose fitted cumulative at its latest
    age is zero or less, so its factor to ultimate is undefined, and a
    negative ultimate (both ``negative_projection``). Those are checked in
    that order. Before them all, a triangle whose cells are all zero is
    refused as ``tweedie_glm`` refuses it (``not_identified``, "cells has no
    losses to fit"). scikit-learn not installed raises ``ImportError``.
    """
    if not isinstance(spec, MLDevelopmentSpec):
        raise TypeError(f"spec must be an MLDevelopmentSpec, got {type(spec).__name__}")
    origins, as_of = check_grid(grid)
    step = int(grid["dev_grain_months"])
    cum = np.asarray(grid["cum"], dtype=float)
    observed = np.asarray(grid["obs_mask"], dtype=bool)
    latest_dev = np.asarray(grid["latest_dev"], dtype=np.int64)
    n_w, n_d = cum.shape
    latest = cum[np.arange(n_w), latest_dev]
    inc = cum.copy()
    inc[:, 1:] = cum[:, 1:] - cum[:, :-1]

    if not (observed & (cum != 0)).any():
        # the same refusal, in the same words, as tweedie_glm's
        raise Refusal(
            "not_identified",
            "cells has no losses to fit: every observed increment is zero",
            option="cells",
        )
    trained = observed & (cum != 0) if spec.zero_cells == "missing" else observed.copy()
    n_rows = int(trained.sum())
    if n_rows < 2:
        missing = spec.zero_cells == "missing"
        raise Refusal(
            "not_identified",
            f"cells has {n_rows} cell(s) to fit"
            + (" (a zero cumulative is not fitted under zero_cells='missing')" if missing else "")
            + "; a model needs at least 2",
            option="zero_cells" if missing and observed.sum() >= 2 else "cells",
        )
    by_age = trained.any(axis=0)
    empty_ages = np.flatnonzero(~by_age)
    if empty_ages.size and spec.unsupported_factor == "raise":
        ages = [str(int(j + 1) * step) for j in empty_ages]
        raise Refusal(
            "no_link_ratio",
            f"every cell at dev_lag {_and(ages)} is zero, so zero_cells='missing' leaves the "
            "model nothing to fit there. Pass unsupported_factor='unity' to develop by a factor "
            "of 1 into that age, or zero_cells='observed' to fit the zeros",
            option="unsupported_factor",
            options=("unsupported_factor", "zero_cells"),
            links=[(int(j) * step, int(j + 1) * step) for j in empty_ages if j > 0],
            cells=[
                RefusedCell(None, origins[i], int(j + 1) * step, 0.0)
                for i, j in np.argwhere(observed & ~trained & ~by_age[None, :])
            ],
        )

    # -- the design: patsy's treatment coding, column for column ------------------
    dev_levels = [int(j) for j in np.flatnonzero(by_age)]
    origin_levels = (
        [int(i) for i in np.flatnonzero(trained.any(axis=1))] if spec.origin == "factor" else []
    )
    origin_step = np.array(
        [((o.year - origins[0].year) * 12 + o.month - origins[0].month) // step for o in origins],
        dtype=float,
    )
    names = ["intercept"]
    names += [f"development:{(j + 1) * step}" for j in dev_levels[1:]]
    names += [f"origin:{origins[i].isoformat()}" for i in origin_levels[1:]]
    if spec.calendar == "trend":
        names.append("calendar")

    def design(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        columns = [np.ones(rows.size)]
        columns += [(cols == j).astype(float) for j in dev_levels[1:]]
        columns += [(rows == i).astype(float) for i in origin_levels[1:]]
        if spec.calendar == "trend":
            columns.append(origin_step[rows] + cols)
        return np.column_stack(columns)

    # origin by origin and, within an origin, by age: np.nonzero's row-major order
    fit_i, fit_j = np.nonzero(trained)
    response = inc if spec.response == "incremental" else cum
    y = response[fit_i, fit_j]

    model, version = _estimator(spec)
    model.fit(design(fit_i, fit_j), y)

    # -- the fitted rectangle ----------------------------------------------------------
    placed = np.ones(n_w, dtype=bool)
    if spec.origin == "factor":
        placed[:] = False
        placed[origin_levels] = True
    all_i, all_j = np.nonzero(placed[:, None] & by_age[None, :])
    predicted = np.full((n_w, n_d), np.nan)
    predicted[all_i, all_j] = model.predict(design(all_i, all_j))
    unity_ages = ~by_age
    fitted_cum = np.full((n_w, n_d), np.nan)
    with np.errstate(all="ignore"):
        if spec.response == "incremental":
            fitted = np.where(unity_ages[None, :], 0.0, predicted)
            fitted_cum[placed] = np.cumsum(fitted[placed], axis=1)
        else:
            for j in range(n_d):
                if not unity_ages[j]:
                    fitted_cum[placed, j] = predicted[placed, j]
                else:
                    fitted_cum[placed, j] = fitted_cum[placed, j - 1] if j else 0.0
            fitted = fitted_cum.copy()
            fitted[:, 1:] = fitted_cum[:, 1:] - fitted_cum[:, :-1]
    fitted[~placed] = np.nan
    if not (np.isfinite(fitted[placed]).all() and np.isfinite(fitted_cum[placed]).all()):
        # Amounts near the largest double: scikit-learn's own squares overflow
        # and it predicts NaN, or the running sum passes the largest double.
        # Either is refused here, before a NaN could read as a fitted
        # cumulative of zero or less.
        raise Refusal(
            "result_not_finite",
            "the model's fitted values are not all finite numbers: the amounts are too large "
            "for the fit's squares and sums to stay finite. Scale them (work in thousands, say) "
            "and scale the answer back",
            option="cells",
        )

    fit = MLDevelopmentFit(
        spec=spec,
        origin_periods=origins,
        as_of=as_of,
        dev_grain_months=step,
        cumulative=cum,
        trained=trained,
        fitted=fitted,
        fitted_cumulative=fitted_cum,
        latest=latest,
        latest_dev=latest_dev,
        unity_ages=unity_ages,
        n_training_rows=n_rows,
        feature_names=tuple(names),
        sklearn_version=version,
    )
    _require_projection(fit)
    return fit


def _estimator(spec: MLDevelopmentSpec):
    """The scikit-learn estimator for ``spec``, and scikit-learn's version.

    Imported here, never at module level, so that ``ibnr.methods`` loads
    scikit-learn (and the scipy and pandas it imports) only when this runs.
    """
    try:
        import sklearn
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    except ImportError as exc:
        raise ImportError(
            'ml_development needs scikit-learn; install it with pip install "ibnr[ml]"'
        ) from exc
    settings = spec.settings
    if spec.estimator == "random_forest":
        model = RandomForestRegressor(
            n_estimators=settings["n_estimators"],
            max_depth=settings["max_depth"],
            min_samples_leaf=settings["min_samples_leaf"],
            random_state=spec.seed,
            n_jobs=1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=settings["n_estimators"],
            learning_rate=settings["learning_rate"],
            max_depth=settings["max_depth"],
            random_state=spec.seed,
        )
    return model, sklearn.__version__


def _require_projection(fit: MLDevelopmentFit) -> None:
    """Refuse an origin the fitted pattern cannot project, or projects below zero."""
    n_d = fit.cumulative.shape[1]
    step = fit.dev_grain_months
    developing = (fit.latest != 0) & (fit.latest_dev < n_d - 1)
    fitted_latest = fit.fitted_latest
    with np.errstate(invalid="ignore"):
        no_factor = np.flatnonzero(developing & ~(fitted_latest > 0))
    if no_factor.size:
        first = int(no_factor[0])
        raise Refusal(
            "negative_projection",
            "the model's fitted cumulative at the latest age of {cells} is zero or less "
            f"({_amount(fitted_latest[first])} for the first), so the factor from there to "
            "ultimate is undefined. The other estimator, the incremental response or a method "
            "with one pattern for every origin (tweedie_glm, chain_ladder) may answer",
            option="estimator",
            given=fit.spec.estimator,
            cells=[
                RefusedCell(
                    None,
                    fit.origin_periods[i],
                    int(fit.latest_dev[i] + 1) * step,
                    float(fitted_latest[i]),
                )
                for i in no_factor
            ],
        )
    ultimate = fit.ultimate
    negative = np.flatnonzero(ultimate < 0)
    if negative.size:
        first = int(negative[0])
        raise Refusal(
            "negative_projection",
            "the model projects a negative ultimate for {origins} "
            f"({_amount(ultimate[first])} for the first) from losses that are zero or more: its "
            "fitted cumulative falls between the latest age and the last. The other estimator, "
            "the incremental response or a method with one pattern for every origin "
            "(tweedie_glm, chain_ladder) may answer",
            option="estimator",
            given=fit.spec.estimator,
            cells=[
                RefusedCell(None, fit.origin_periods[i], None, float(ultimate[i])) for i in negative
            ],
        )


def _amount(value: float) -> str:
    return f"{float(value):.6g}"


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"
