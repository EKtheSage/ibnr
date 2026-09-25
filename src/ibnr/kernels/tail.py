"""Tail factors: the development still to come after the last observed age.

A tail is a change to the development pattern, made after the link factors are
estimated and before anything is projected with them. :class:`TailSpec` holds
one, fixed like a ``ConventionalCandidate``; :func:`apply_tail` applies it to
the link factors of a fit (or to one row of factors per simulation, for a
bootstrap); :func:`tail_variance` gives the tail step's sigma and the tail
factor's standard error for Mack's standard errors.

Four kinds of tail:

- ``"constant"``: a factor the caller chooses, such as 1.05, which is the
  development from the attachment age to ultimate. With the default attachment
  (the last observed age) it is the tail factor itself. It is spread over the
  rows shown beyond the triangle by ``decay``: the steps are ``1 + x0 *
  decay ** k``, and the last shown step holds what is left, so the steps
  multiply to exactly the factor. With an earlier attachment the same steps
  replace the link factors from that age on, and the tail factor beyond the
  last observed age is what is left of the factor after them.
- ``"exponential"``, ``"inverse_power"`` and ``"weibull"``: a curve fitted to
  the link factors by least squares, extrapolated ``steps`` development steps
  past the last observed age. Link ``t`` (``t = 1`` develops from the first
  age) has factor ``c(t) = 1 + g(t)``:

  ============= ================================== =============================
  kind          ``g(t)``                           the straight line fitted
  ============= ================================== =============================
  exponential   ``exp(a + b t)``                   ``log(f - 1)`` on ``t``
  inverse_power ``exp(a) t ** b``                  ``log(f - 1)`` on ``log t``
  weibull       ``1 / (1 - exp(-exp(a) t ** b))``  ``log(log(f / (f - 1)))`` on
                ``- 1``                            ``log t``
  ============= ================================== =============================

  Only factors above :data:`MIN_FIT_FACTOR` (1.00001) enter the line, and only
  those developing from an age inside ``fit_lags`` when it is given. The tail
  factor is ``c(n + 1) * ... * c(n + steps)`` for ``n`` link factors. The curve
  must decay: an exponential or inverse power curve needs a negative slope,
  a Weibull curve a positive one, and an inverse power curve a slope below -1,
  since between -1 and 0 the product of its factors never converges and the
  tail would be set by ``steps`` alone.

This is chainladder-python's ``TailConstant`` and ``TailCurve`` arithmetic
(chainladder 0.9.2), with these differences: ``fit_lags`` includes both ends
(chainladder's ``fit_period`` leaves out its end), an attachment at the first
age is honoured for a constant tail (chainladder ignores it), and every case
chainladder rounds, crashes on or silently answers as no tail is refused here
by name. ``docs/coming-from-chainladder.md`` lists them.

This module imports numpy and ``ibnr.errors`` only, so ``ibnr.methods`` can use
it without loading ibis, pandas or scipy.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from ibnr.errors import Refusal

__all__ = [
    "CURVES",
    "MIN_FIT_FACTOR",
    "TAIL_KINDS",
    "TailFit",
    "TailSpec",
    "TailVariance",
    "apply_tail",
    "tail_variance",
]

#: The kinds of tail.
TAIL_KINDS = ("constant", "exponential", "inverse_power", "weibull")

#: The fitted curves.
CURVES = TAIL_KINDS[1:]

#: A link factor at or below this never enters a curve fit (chainladder's
#: ``reg_threshold[0]``): ``log(f - 1)`` is not a number at 1 and below.
MIN_FIT_FACTOR = 1.00001

#: The share of a constant tail's step that the next step keeps, by default.
DEFAULT_DECAY = 0.5

#: How many development steps a curve is extrapolated past the last age, by
#: default (chainladder's ``extrap_periods``).
DEFAULT_STEPS = 100

#: The most steps a curve is extrapolated, which bounds the time and memory one
#: fit takes: an inverse power tail is still growing at 10,000 steps (genins
#: 1.29243 at 100, 1.32104 at 10,000), so a larger count buys little.
MAX_STEPS = 10_000

#: How many terms of ``decay ** k`` the constant tail's first step is solved
#: over, as in chainladder.
_DECAY_TERMS = 1000


@dataclass(frozen=True)
class TailSpec:
    """A tail, held fixed like a ``ConventionalCandidate``.

    Ages are months from the start of the origin period; counts are
    development steps. Every field is checked when the spec is made; the
    checks that need the triangle (an age on its grid, two factors to fit a
    curve to) are made by :func:`apply_tail`.

    Attributes
    ----------
    kind : str
        One of :data:`TAIL_KINDS`.
    factor : float or None
        ``"constant"`` only, and needed there: the development from
        ``attach_lag`` to ultimate, such as 1.05. A positive finite number;
        below 1 is allowed (a triangle whose losses fall after the last age).
    decay : float or None
        ``"constant"`` only: from 0 to 1, the share of each tail step's
        development that the next step keeps. ``None`` is 0.5.
    attach_lag : int or None
        The age the ratio of the first link the tail replaces develops FROM,
        a development age of the triangle. ``None`` is the last observed age,
        where nothing is replaced.
    fit_lags : tuple of (int or None, int or None)
        Curves only: the first and the last link to fit the curve to, each
        named by the age it develops FROM, both included. ``None`` at either
        end is the edge of the triangle.
    steps : int or None
        Curves only: development steps extrapolated beyond the last observed
        age, 1 to :data:`MAX_STEPS`. ``None`` is 100.
    rows : int or None
        How many development steps beyond the last observed age are shown one
        by one; the rest of the tail is carried by the last of them. It never
        moves an ultimate. ``None`` is ``12 // dev_grain_months`` (at least 1,
        and for a curve at most ``steps``).
    sigma, std_err : float or None
        Mack only: the tail step's sigma and the tail factor's standard error,
        each used instead of the value read off the other sigmas. Zero or a
        positive finite number.
    option_prefix : str
        How refusals name the fields: ``""`` names them as here, ``"tail_"``
        as ``ibnr.methods`` does (``tail_decay``, and ``tail`` for ``kind``).
        It changes no number, and specs that differ only in it are equal.
    """

    kind: str
    factor: float | None = None
    decay: float | None = None
    attach_lag: int | None = None
    fit_lags: tuple[int | None, int | None] = (None, None)
    steps: int | None = None
    rows: int | None = None
    sigma: float | None = None
    std_err: float | None = None
    option_prefix: str = field(default="", compare=False, repr=False)

    def __post_init__(self) -> None:
        name = self.name
        if self.option_prefix not in ("", "tail_"):
            raise ValueError("option_prefix is '' or 'tail_'")
        if not isinstance(self.kind, str) or self.kind not in TAIL_KINDS:
            factor = (
                f"; a constant tail's factor goes in {name('factor')}" if self.option_prefix else ""
            )
            raise Refusal(
                "invalid_option",
                f"{name('kind')} is the kind of tail: 'constant', 'exponential', 'inverse_power' "
                f"or 'weibull', got {{given}}{factor}",
                option=name("kind"),
                given=self.kind,
            )
        constant = self.kind == "constant"
        if constant:
            for other in ("fit_lags", "steps"):
                if getattr(self, other) != (None, None) and getattr(self, other) is not None:
                    raise Refusal(
                        "invalid_option",
                        f"{name(other)} is a curve setting; a constant tail takes "
                        f"{name('factor')} and {name('decay')}",
                        option=name(other),
                        options=(name(other), name("kind")),
                        given=getattr(self, other),
                    )
            value = self.factor
            if not _is_number(value) or not math.isfinite(value) or value <= 0:
                raise Refusal(
                    "invalid_option",
                    f"a constant tail needs {name('factor')}, the development from the "
                    "attachment age to ultimate, such as 1.05: a positive finite number, got "
                    "{given}",
                    option=name("factor"),
                    given=value,
                )
            object.__setattr__(self, "factor", float(value))
            decay = self.decay
            if decay is not None:
                if not _is_number(decay) or not 0 <= decay <= 1:
                    raise Refusal(
                        "invalid_option",
                        f"{name('decay')} must be between 0 and 1, got {{given}}: it is the "
                        "share of each tail step's development that the next step keeps",
                        option=name("decay"),
                        given=decay,
                    )
                object.__setattr__(self, "decay", float(decay))
        else:
            for other in ("factor", "decay"):
                if getattr(self, other) is not None:
                    raise Refusal(
                        "invalid_option",
                        f"{name(other)} is a constant-tail setting; a {self.kind} tail is fitted "
                        f"to the link factors, and takes {name('fit_lags')} and {name('steps')}",
                        option=name(other),
                        options=(name(other), name("kind")),
                        given=getattr(self, other),
                    )
            self._check_fit_lags()
            steps = self.steps
            if steps is not None:
                if not _is_count(steps) or not 1 <= steps <= MAX_STEPS:
                    raise Refusal(
                        "invalid_option",
                        f"{name('steps')} is how many development steps the curve is "
                        f"extrapolated past the last observed age, a whole number from 1 to "
                        f"{MAX_STEPS}, got {{given}}",
                        option=name("steps"),
                        given=steps,
                    )
                object.__setattr__(self, "steps", int(steps))
        if self.attach_lag is not None:
            lag = self.attach_lag
            if not _is_count(lag) or lag < 1:
                raise Refusal(
                    "invalid_option",
                    f"{name('attach_lag')} is the age, in whole months, of the first link the "
                    "tail replaces, got {given}",
                    option=name("attach_lag"),
                    given=lag,
                )
            object.__setattr__(self, "attach_lag", int(lag))
        rows = self.rows
        if rows is not None:
            if not _is_count(rows) or not 0 <= rows <= MAX_STEPS:
                raise Refusal(
                    "invalid_option",
                    f"{name('rows')} is how many development steps beyond the last observed age "
                    f"are shown one by one, a whole number from 0 to {MAX_STEPS}, got {{given}}",
                    option=name("rows"),
                    given=rows,
                )
            object.__setattr__(self, "rows", int(rows))
            if not constant and rows > self.n_steps:
                raise Refusal(
                    "invalid_option",
                    f"{name('rows')}={rows} asks for {rows} rows beyond the triangle, but "
                    f"{name('steps')}={self.n_steps} extrapolates {self.n_steps} step(s)",
                    option=name("rows"),
                    options=(name("rows"), name("steps")),
                    given=rows,
                )
        for other in ("sigma", "std_err"):
            value = getattr(self, other)
            if value is None:
                continue
            if not _is_number(value) or not math.isfinite(value) or value < 0:
                raise Refusal(
                    "invalid_option",
                    f"{name(other)} must be zero or a positive finite number, got {{given}}",
                    option=name(other),
                    given=value,
                )
            object.__setattr__(self, other, float(value))

    def _check_fit_lags(self) -> None:
        name = self.name
        lags = self.fit_lags
        if isinstance(lags, list):
            lags = tuple(lags)
        if not isinstance(lags, tuple) or len(lags) != 2:
            raise Refusal(
                "invalid_option",
                f"{name('fit_lags')} is a pair (first, last) of the ages, in months, the first "
                "and the last link fitted develop from, either of them None, got {given}",
                option=name("fit_lags"),
                given=self.fit_lags,
            )
        for lag in lags:
            if lag is not None and (not _is_count(lag) or lag < 1):
                raise Refusal(
                    "invalid_option",
                    f"{name('fit_lags')} {{given}} holds {_shown(lag)}, which is not a positive "
                    "whole number of months",
                    option=name("fit_lags"),
                    given=lags,
                )
        lags = tuple(None if lag is None else int(lag) for lag in lags)
        if None not in lags and lags[0] > lags[1]:
            raise Refusal(
                "invalid_option",
                f"{name('fit_lags')} {{given}} starts after it ends; it is (first, last)",
                option=name("fit_lags"),
                given=lags,
            )
        object.__setattr__(self, "fit_lags", lags)

    def name(self, field_name: str) -> str:
        """What a refusal calls a field: itself, or ``tail_`` + it at the front door."""
        if not self.option_prefix:
            return field_name
        return "tail" if field_name == "kind" else self.option_prefix + field_name

    @property
    def is_curve(self) -> bool:
        return self.kind in CURVES

    @property
    def n_steps(self) -> int:
        """Development steps a curve is extrapolated: ``steps``, or 100."""
        return DEFAULT_STEPS if self.steps is None else self.steps

    @property
    def n_decay(self) -> float:
        """A constant tail's decay: ``decay``, or 0.5."""
        return DEFAULT_DECAY if self.decay is None else self.decay

    def n_rows(self, dev_grain_months: int) -> int:
        """Rows shown beyond the last observed age: ``rows``, or one year of steps
        (``12 // dev_grain_months``, at least 1, and at most ``steps`` for a curve)."""
        if self.rows is not None:
            return self.rows
        rows = max(1, 12 // dev_grain_months)
        return min(rows, self.n_steps) if self.is_curve else rows


@dataclass(frozen=True)
class TailFit:
    """A tail applied to link factors.

    Every array has a leading axis of simulations when :func:`apply_tail` was
    given one row of factors per simulation, and none otherwise.

    Attributes
    ----------
    spec : TailSpec
    factors : numpy.ndarray
        ``(..., n_links)`` the link factors after the attachment: the input's,
        with the links from ``attach_lag`` on replaced by the tail's steps.
    tail_factor : numpy.ndarray
        ``(...)`` the development beyond the last observed age.
    shown : numpy.ndarray
        ``(..., rows)`` the step factors shown beyond the last observed age,
        from it to the next row.
    beyond_cdf : numpy.ndarray
        ``(..., rows + 1)`` the development to ultimate from the last observed
        age (``tail_factor``) and from each row beyond it; the last is the rest
        of the tail, which the last row carries.
    curve : numpy.ndarray or None
        Curves: ``(..., n_links + rows)`` the curve's factor ``1 + g(t)`` at
        every link of the triangle and every row shown beyond it, for comparing
        fitted with selected. ``None`` for a constant tail.
    in_fit : numpy.ndarray or None
        Curves: ``(..., n_links)`` bool, the factors the line went through.
    intercept, slope : numpy.ndarray or None
        Curves: ``(...)`` the fitted line.
    attach_index : int
        The index of the first link replaced; ``n_links`` when none is.
    ok : numpy.ndarray
        ``(...)`` bool: the fit passed every check. Always all true unless
        :func:`apply_tail` was called with ``on_error="flag"``; the factors of a
        row that failed hold NaN, and its ``intercept`` and ``slope`` are the
        line that failed (NaN with fewer than two points).
    """

    spec: TailSpec
    factors: np.ndarray
    tail_factor: np.ndarray
    shown: np.ndarray
    beyond_cdf: np.ndarray
    curve: np.ndarray | None
    in_fit: np.ndarray | None
    intercept: np.ndarray | None
    slope: np.ndarray | None
    attach_index: int
    ok: np.ndarray

    @property
    def rows(self) -> int:
        return self.shown.shape[-1]


def apply_tail(
    factors, dev_grain_months: int, spec: TailSpec, *, on_error: str = "raise"
) -> TailFit:
    """Apply a tail to link factors.

    ``factors`` holds one factor per link of the triangle, ``(n_links,)``, or
    one row of them per simulation, ``(n_sims, n_links)``; link ``j`` develops
    from age ``(j + 1) * dev_grain_months``. The curve is fitted to each row
    on its own, with the closed-form least-squares line, so a bootstrap fits
    every simulated triangle's curve in one call.

    Refused, by name (with ``on_error="raise"``, the default): an
    ``attach_lag`` or a ``fit_lags`` end that is not a development age of the
    triangle, or after its last age that can carry one; a curve with fewer than
    two factors above :data:`MIN_FIT_FACTOR` to fit (``not_identified``); a
    curve that does not decay (``tail_not_decaying``); and a tail that is not a
    finite positive number (``result_not_finite``), never answered as no tail.
    With ``on_error="flag"`` a row of factors whose curve fails one of the last
    three is not refused: its arrays hold NaN and ``ok`` is false for it, so a
    bootstrap can count those draws and handle them.
    """
    if on_error not in ("raise", "flag"):
        raise Refusal(
            "invalid_option",
            "on_error must be 'raise' or 'flag', got {given}",
            option="on_error",
            given=on_error,
        )
    step = int(dev_grain_months)
    given = np.asarray(factors, dtype=float)
    if given.ndim not in (1, 2):
        raise ValueError(f"factors must be (n_links,) or (n_sims, n_links), got {given.shape}")
    f = np.atleast_2d(given)
    n = f.shape[1]
    k = _attach_index(spec, n, step)
    rows = spec.n_rows(step)
    if spec.kind == "constant":
        fit = _constant(f, spec, k, rows)
    else:
        fit = _curve(f, spec, k, rows, step, on_error)
    if given.ndim == 1:
        fit = _squeeze(fit)
    return fit


def _attach_index(spec: TailSpec, n: int, step: int) -> int:
    """The index of the first link the tail replaces; ``n`` for none."""
    if spec.attach_lag is None:
        return n
    lag = spec.attach_lag
    last = (n + 1) * step
    ages = _ages(step, n + 1)
    if lag % step:
        raise Refusal(
            "grain_mismatch",
            f"{spec.name('attach_lag')} {lag} is not a development age of this triangle ({ages})",
            option=spec.name("attach_lag"),
            given=lag,
        )
    if lag > last:
        raise Refusal(
            "not_in_triangle",
            f"{spec.name('attach_lag')} {lag} is after the last observed age {last}; the tail "
            f"attaches at a development age of this triangle ({ages})",
            option=spec.name("attach_lag"),
            given=lag,
        )
    return lag // step - 1


def _fit_range(spec: TailSpec, n: int, step: int) -> tuple[int, int]:
    """The 0-based indices of the first and the last link a curve is fitted to."""
    first, last = spec.fit_lags
    ages = _ages(step, n)
    bounds = []
    for lag, edge in ((first, 0), (last, n - 1)):
        if lag is None:
            bounds.append(edge)
            continue
        if lag % step:
            raise Refusal(
                "grain_mismatch",
                f"{spec.name('fit_lags')} {spec.fit_lags!r} names {lag} months, which is not a "
                f"development age of this triangle; the links develop from {ages}",
                option=spec.name("fit_lags"),
                given=spec.fit_lags,
            )
        if lag > n * step:
            raise Refusal(
                "not_in_triangle",
                f"{spec.name('fit_lags')} {spec.fit_lags!r} names {lag} months, but no link "
                f"develops from that age; the links develop from {ages}",
                option=spec.name("fit_lags"),
                given=spec.fit_lags,
            )
        bounds.append(lag // step - 1)
    return bounds[0], bounds[1]


def _ages(step: int, count: int) -> str:
    """The first ``count`` development ages, as a message lists them."""
    if count <= 0:
        return "none"
    if count <= 3:
        return ", ".join(str(step * (i + 1)) for i in range(count))
    return f"{step}, {2 * step}, ... {count * step}"


def _constant(f: np.ndarray, spec: TailSpec, k: int, rows: int) -> TailFit:
    """chainladder's ``TailConstant`` arithmetic (``_apply_decay``)."""
    m, n = f.shape
    tail = spec.factor
    width = (n - k) + rows + 1  # every step from the attachment to the end
    decay = spec.n_decay
    arr = decay ** np.arange(max(_DECAY_TERMS, width), dtype=float)
    if width < 2 or tail == 1.0:
        x0 = 0.0  # one step holds the whole tail, or there is no tail to spread
    else:
        head = arr[:_DECAY_TERMS]
        qa, qb, qc = np.sum(head**2), np.sum(head), -np.log(tail)
        with np.errstate(invalid="ignore"):
            x0 = float((-qb + np.sqrt(qb**2 - 4 * qa * qc)) / (2 * qa))
        if not math.isfinite(x0):
            raise Refusal(
                "result_not_finite",
                f"a constant tail of {tail!r} cannot be spread over its steps with "
                f"{spec.name('decay')}={decay!r}: the first step would have to fall further "
                f"than the decay allows. Raise {spec.name('decay')}, or show no rows beyond the "
                f"triangle ({spec.name('rows')}=0) with the tail attached at the last age",
                option=spec.name("decay"),
                options=(spec.name("decay"), spec.name("factor")),
                given=decay,
            )
    steps = (1 + x0 * arr)[:width]
    replaced, beyond = steps[: n - k], steps[n - k : -1]
    # the development from the last observed age: what is left of the factor after
    # the replaced links, so the steps from the attachment multiply to it
    tail_factor = tail / np.prod(replaced) if replaced.size else tail
    rest = tail / np.prod(steps[:-1])
    shown = beyond
    # the development to ultimate from the last observed age and from each row beyond
    # it: the tail factor, then the steps still to come, the last row holding the rest
    beyond_cdf = np.r_[tail_factor, _reverse_cumprod(np.r_[shown[1:], rest])[:rows]]
    factors = f.copy()
    factors[:, k:] = replaced
    return TailFit(
        spec=spec,
        factors=factors,
        tail_factor=np.full(m, tail_factor),
        shown=np.tile(shown, (m, 1)),
        beyond_cdf=np.tile(beyond_cdf, (m, 1)),
        curve=None,
        in_fit=None,
        intercept=None,
        slope=None,
        attach_index=k,
        ok=np.ones(m, dtype=bool),
    )


def _reverse_cumprod(values: np.ndarray) -> np.ndarray:
    """``out[i] = values[i] * values[i + 1] * ...``."""
    return np.cumprod(values[::-1])[::-1]


def _curve(f: np.ndarray, spec: TailSpec, k: int, rows: int, step: int, on_error: str) -> TailFit:
    """chainladder's ``TailCurve`` arithmetic, one least-squares line per row of ``f``."""
    m, n = f.shape
    first, last = _fit_range(spec, n, step) if n else (0, -1)
    t = np.arange(1, n + 1, dtype=float)
    x = t if spec.kind == "exponential" else np.log(t)
    in_range = (np.arange(n) >= first) & (np.arange(n) <= last)
    in_fit = in_range & np.isfinite(f) & (f > MIN_FIT_FACTOR)
    count = in_fit.sum(axis=1)
    with np.errstate(all="ignore"):
        safe = np.where(in_fit, f, 2.0)
        y = np.log(np.log(safe / (safe - 1))) if spec.kind == "weibull" else np.log(safe - 1)
        y = np.where(in_fit, y, 0.0)
        xs = np.where(in_fit, x, 0.0)
        x_mean = xs.sum(axis=1) / count
        y_mean = y.sum(axis=1) / count
        dx = np.where(in_fit, x - x_mean[:, None], 0.0)
        slope = (dx * (y - y_mean[:, None])).sum(axis=1) / (dx**2).sum(axis=1)
        intercept = y_mean - slope * x_mean
        big_t = np.arange(1, n + spec.n_steps + 1, dtype=float)
        curve = 1.0 + _g(spec.kind, intercept[:, None], slope[:, None], big_t[None, :])
        beyond = curve[:, n:]
        beyond_all = _reverse_cumprod_rows(beyond)
        tail_factor = beyond_all[:, 0]
        # the development from each row beyond the last age: rows + 1 values, the last
        # holding what is left after the rows shown (1.0 when every step is shown)
        beyond_cdf = np.concatenate([beyond_all, np.ones((m, 1))], axis=1)[:, : rows + 1]
    reason = np.full(m, "", dtype=object)
    reason[count < 2] = "not_identified"
    decays = {
        "exponential": slope < 0,
        "inverse_power": slope < -1,
        "weibull": slope > 0,
    }[spec.kind]
    reason[(reason == "") & ~decays] = "tail_not_decaying"
    finite = (
        np.isfinite(tail_factor)
        & (tail_factor > 0)
        & np.isfinite(curve[:, : n + rows]).all(axis=1)
        & np.isfinite(beyond_cdf).all(axis=1)
    )
    reason[(reason == "") & ~finite] = "result_not_finite"
    ok = reason == ""
    if not ok.all() and on_error == "raise":
        bad = int(np.flatnonzero(~ok)[0])
        raise _curve_refusal(
            spec, str(reason[bad]), in_range, int(count[bad]), float(slope[bad]), step, (bad, m)
        )
    factors = f.copy()
    factors[:, k:] = curve[:, k:n]
    shown = beyond[:, :rows]
    out = TailFit(
        spec=spec,
        factors=factors,
        tail_factor=tail_factor,
        shown=shown,
        beyond_cdf=beyond_cdf,
        curve=curve[:, : n + rows],
        in_fit=in_fit,
        intercept=intercept,
        slope=slope,
        attach_index=k,
        ok=ok,
    )
    if not ok.all():
        out = _blank(out, ~ok)
    return out


def _g(kind: str, a, b, t):
    """The curve's development beyond 1 at link ``t``, as chainladder computes it."""
    if kind == "exponential":
        return np.exp(b * t + a)
    if kind == "inverse_power":
        return np.exp(a) * t**b
    return 1 / (1 - np.exp(-np.exp(a) * t**b)) - 1


def _reverse_cumprod_rows(values: np.ndarray) -> np.ndarray:
    return np.cumprod(values[:, ::-1], axis=1)[:, ::-1]


def _curve_refusal(
    spec: TailSpec,
    reason: str,
    in_range: np.ndarray,
    count: int,
    slope: float,
    step: int,
    position: tuple[int, int],
) -> Refusal:
    """The refusal for one row of factors whose curve failed a check.

    ``position`` is (the row, how many rows there are), so a bootstrap's
    refusal names the simulation."""
    where = np.flatnonzero(in_range)
    if where.size > 1:
        span = f"from {(where[0] + 1) * step} to {(where[-1] + 1) * step} months"
    elif where.size:
        span = f"at {(where[0] + 1) * step} months"
    else:
        span = "in the fit range"
    row, m = position
    row = f" (in simulation row {row + 1} of {m})" if m > 1 else ""
    kind = spec.kind.replace("_", " ")
    links = [((j + 1) * step, (j + 2) * step) for j in where]
    if reason == "not_identified":
        article = "a" if spec.kind == "weibull" else "an"
        factors = "factor" if where.size == 1 else "factors"
        return Refusal(
            "not_identified",
            f"{article} {kind} tail needs a straight line through at least two link factors "
            f"above {MIN_FIT_FACTOR}; {span} only {count} of the {where.size} {factors} "
            f"{'is' if count == 1 else 'are'} above {MIN_FIT_FACTOR}{row}. Widen "
            f"{spec.name('fit_lags')} or use a constant tail",
            option=spec.name("kind"),
            options=(spec.name("kind"), spec.name("fit_lags")),
            given=spec.kind,
            links=links,
        )
    if reason == "tail_not_decaying":
        if spec.kind == "inverse_power" and slope < 0:
            why = (
                f"its slope is {slope:.4g}, between -1 and 0, so the product of its factors "
                f"never converges and the tail would be set by {spec.name('steps')} alone"
            )
        else:
            why = f"it grows with age (slope {slope:.4g}), so it has no tail"
        return Refusal(
            "tail_not_decaying",
            f"the {kind} curve fitted to the factors {span} does not decay: {why}{row}. Choose "
            f"other links with {spec.name('fit_lags')}, another curve, or a constant tail",
            option=spec.name("kind"),
            options=(spec.name("kind"), spec.name("fit_lags")),
            given=spec.kind,
            links=links,
        )
    return Refusal(
        "result_not_finite",
        f"the {kind} tail fitted to the factors {span} is not a finite number: the fitted "
        f"curve is too steep for its factors to multiply out{row}. Choose other links with "
        f"{spec.name('fit_lags')}, fewer {spec.name('steps')}, or a constant tail",
        option=spec.name("kind"),
        given=spec.kind,
        links=links,
    )


def _blank(fit: TailFit, bad: np.ndarray) -> TailFit:
    """The fit with NaN in every factor of the rows ``bad``; their line is kept."""
    out = {}
    for name in ("factors", "tail_factor", "shown", "beyond_cdf", "curve"):
        values = getattr(fit, name).copy()
        values[bad] = np.nan
        out[name] = values
    return TailFit(**{**fit.__dict__, **out})


def _squeeze(fit: TailFit) -> TailFit:
    """A one-row fit without its leading axis."""
    out = {}
    for name in ("factors", "tail_factor", "shown", "beyond_cdf", "curve", "in_fit"):
        value = getattr(fit, name)
        out[name] = None if value is None else value[0]
    for name in ("intercept", "slope"):
        value = getattr(fit, name)
        out[name] = None if value is None else value[0]
    out["ok"] = fit.ok[0]
    return TailFit(**{**fit.__dict__, **out})


# -- Mack's tail variance ------------------------------------------------------


@dataclass(frozen=True)
class TailVariance:
    """The tail step's variance terms for Mack's standard errors.

    Attributes
    ----------
    sigma2 : float
        The tail step's sigma squared.
    se2 : float
        The tail factor's standard error squared.
    position : float or None
        Where the tail sits on the link axis (``t = 1`` is the first link): the
        ``t`` at which the straight line through ``log(f_t - 1)`` reaches
        ``log(tail_factor - 1)``. ``None`` when it was not needed: both values
        given, or a tail factor of exactly 1.
    """

    sigma2: float
    se2: float
    position: float | None


def tail_variance(
    f,
    sigma2,
    s,
    tail_factor: float,
    *,
    sigma: float | None = None,
    std_err: float | None = None,
    dev_grain_months: int = 12,
    option_prefix: str = "",
) -> TailVariance:
    """The tail step's sigma and the tail factor's standard error, for Mack.

    Mack (1999) and R's ``MackChainLadder(tail=)``: the tail is one more
    development step with factor ``tail_factor``. Its position ``p`` on the
    link axis is where a straight line through ``(t, log(f_t - 1))``, over the
    factors above 1, reaches ``log(tail_factor - 1)``. The tail's sigma is read
    at ``p`` off a straight line through ``(t, log sigma_t)`` over the positive
    sigmas, and its standard error off one through ``(t, log se_t)``, with
    ``se_t = sqrt(sigma2_t / s_t)``, over the positive ones.

    These are R's lines, with the points that have no logarithm left out.
    chainladder-python leaves such a point's ``t`` in its sums (and fills a
    zero sigma with 1e-320), so it agrees only where no factor is at or below 1
    and no sigma is zero or missing.

    ``sigma`` and ``std_err`` replace their own read; the position is found
    only when one of them is still needed. A tail factor of exactly 1 gives
    zero for both reads. Refused as ``variance_not_estimable``: a tail factor
    below 1 without both values given, fewer than two factors above 1 or two
    positive sigmas (or standard errors) when a line is needed, factors that do
    not decay (a line that does not fall), and a position below 1, before the
    first link, where the sigma would be extrapolated backwards.
    """
    names = TailSpec("constant", factor=1.0, option_prefix=option_prefix).name
    f = np.asarray(f, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    s = np.asarray(s, dtype=float)
    tail_factor = float(tail_factor)
    need_sigma, need_se = sigma is None, std_err is None
    if not (need_sigma or need_se):
        return TailVariance(float(sigma) ** 2, float(std_err) ** 2, None)
    if tail_factor == 1.0:
        return TailVariance(
            0.0 if need_sigma else float(sigma) ** 2,
            0.0 if need_se else float(std_err) ** 2,
            None,
        )
    both = f"{names('sigma')} and {names('std_err')}"
    if tail_factor < 1:
        raise Refusal(
            "variance_not_estimable",
            f"the tail factor is {tail_factor!r}, below 1. The tail's sigma is read off the "
            "factors' decay at the age where the curve through them reaches the tail, which "
            f"needs a tail above 1; pass {both}",
            option=names("sigma") if need_sigma else names("std_err"),
            options=(names("sigma"), names("std_err")),
        )
    t = np.arange(1, f.size + 1, dtype=float)
    above = np.isfinite(f) & (f > 1)
    step = dev_grain_months
    if above.sum() < 2:
        raise Refusal(
            "variance_not_estimable",
            f"the tail's sigma is read off a straight line through the logarithms of the link "
            f"factors less 1, which needs two factors above 1; {int(above.sum())} of the "
            f"{f.size} are. Pass {both}",
            option=names("sigma") if need_sigma else names("std_err"),
            options=(names("sigma"), names("std_err")),
        )
    slope, intercept = _line(t[above], np.log(f[above] - 1))
    if not slope < 0:
        raise Refusal(
            "variance_not_estimable",
            "the link factors do not fall toward 1 with age (the straight line through the "
            f"logarithms of the factors less 1 has slope {slope:.4g}), so there is no age at "
            f"which the tail sits to read its sigma at. Pass {both}",
            option=names("sigma") if need_sigma else names("std_err"),
            options=(names("sigma"), names("std_err")),
        )
    position = float((math.log(tail_factor - 1) - intercept) / slope)
    if not position >= 1:
        raise Refusal(
            "variance_not_estimable",
            f"the tail factor {tail_factor!r} is larger than the straight line through the link "
            f"factors gives even at the first link (it sits at link {position:.4g}, before the "
            f"link from {step} months), so its sigma would be extrapolated backwards past the "
            f"data. Pass {both}",
            option=names("sigma") if need_sigma else names("std_err"),
            options=(names("sigma"), names("std_err")),
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigmas = np.sqrt(sigma2)
        errors = np.sqrt(sigma2 / s)
    out = []
    for need, values, given, what in (
        (need_sigma, sigmas, sigma, "sigmas"),
        (need_se, errors, std_err, "factor standard errors"),
    ):
        if not need:
            out.append(float(given) ** 2)
            continue
        positive = np.isfinite(values) & (values > 0)
        if positive.sum() < 2:
            raise Refusal(
                "variance_not_estimable",
                f"the tail's {'sigma' if what == 'sigmas' else 'standard error'} is read off a "
                f"straight line through the logarithms of the {what}, which needs two positive "
                f"ones; {int(positive.sum())} of the {values.size} are. Pass {both}",
                option=names("sigma") if what == "sigmas" else names("std_err"),
                options=(names("sigma"), names("std_err")),
            )
        b, a = _line(t[positive], np.log(values[positive]))
        out.append(float(np.exp(a + b * position)) ** 2)
    return TailVariance(out[0], out[1], position)


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of the least-squares line through the points."""
    x_mean, y_mean = x.mean(), y.mean()
    dx = x - x_mean
    slope = float((dx * (y - y_mean)).sum() / (dx**2).sum())
    return slope, float(y_mean - slope * x_mean)


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _is_count(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool | np.bool_)


def _shown(value) -> str:
    return repr(value)
