"""Which link ratios a development factor is estimated from, and how they are averaged.

One selection, written once: the conventional point fits use it now
(``kernels/conventional.py``), and it is the piece the Mack fit and the
bootstrap refit are to share, so that one set of development options means the
same thing in all three. It imports numpy and the standard library only, so
``ibnr.methods`` can use it without loading ibis, pandas or scipy.

:class:`LinkRules` holds the options and checks them. :func:`select_links`
applies them to a grid, one link (development age to the next) at a time, in
this order, each rule acting on the link ratios the rules before it left:

1. the observed pairs: both cells present;
2. the zero rule (``zero_cells``): under ``"missing"`` a pair with a zero at
   either end is left out as ``zero_cell``; under ``"observed"`` a ratio that is
   not a finite number (a pair starting from zero) is left out as
   ``undefined_ratio``;
3. the history window (``history_periods``): only the latest this many pairs by
   origin are kept. Under ``"missing"`` every observed pair counts towards the
   window, as chainladder-python's ``n_periods`` counts diagonals, so a pair left
   out for a zero cell keeps its place; under ``"observed"`` only the pairs still
   used count, so an older pair moves into the window;
4. explicit exclusions (``exclude``): ``(origin period, FROM age in months)``;
5. excluded valuations (``exclude_valuations``): every link ratio whose LATER
   cell is valued on one of these dates, so excluding a year removes the
   development that happened during it;
6. bounds (``drop_above``, ``drop_below``): a ratio strictly above
   ``drop_above`` or strictly below ``drop_below`` is left out; a ratio equal to
   a bound is kept;
7. trims (``drop_high``, ``drop_low``): the ``drop_high`` highest and the
   ``drop_low`` lowest of the ratios still used are left out.

Steps 6 and 7 each obey ``preserve``, the fewest ratios they may leave at a
link, all or nothing: if a rule would leave fewer, ``exhausted_exclusions``
decides. ``"raise"`` refuses; ``"keep"`` skips that rule at that link, and the
selection records it (``bounds_skipped``, ``trimming_skipped``). A link that has
no ratio left before step 6 or 7 is not an exhausted rule; the caller's own
fallback decides what an empty link means.

Ties in step 7 follow ``trim_ties``. ``"origin"`` ranks by ratio, then by
origin: among equal ratios ``drop_low`` removes the oldest origin and
``drop_high`` the newest. ``"volume"`` ranks by ratio, then by the earlier
cumulative, then by origin: among equal ratios ``drop_high`` removes the one
with the largest earlier cumulative and ``drop_low`` the smallest, which is
chainladder-python's rule.

:func:`link_factors` averages what is left: ``"volume"``, ``"simple"``,
``"regression"`` or ``"median"``. In Mack's notation each ratio is weighted by
``previous ** alpha``, with ``alpha`` in :data:`ALPHA`; the median has no
alpha.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from dataclasses import dataclass

import numpy as np

from ibnr.errors import Refusal, RefusedCell, _literal
from ibnr.kernels.grid import ZERO_CELLS, as_date, month_end

#: How link ratios become a factor.
AVERAGES = ("volume", "simple", "regression", "median")

#: Mack's exponent for each average: a ratio is weighted by ``previous ** alpha``.
#: The median has none.
ALPHA = {"simple": 0, "volume": 1, "regression": 2}

#: How ``drop_high`` and ``drop_low`` break ties between equal ratios.
TRIM_TIES = ("origin", "volume")

#: What ``exhausted_exclusions`` may be.
EXHAUSTED = ("raise", "keep")

#: Why a link ratio is or is not used, in the order the rules run. A ratio gets
#: the reason of the first rule that removed it.
REASONS = (
    "included",
    "zero_cell",
    "undefined_ratio",
    "history_window",
    "explicit_exclusion",
    "valuation_exclusion",
    "drop_above",
    "drop_below",
    "drop_low",
    "drop_high",
)

_CODE = {reason: index for index, reason in enumerate(REASONS)}


@dataclass(frozen=True)
class LinkRules:
    """The development options that decide which link ratios make a factor.

    Every field is checked when the rules are made, and numbers are stored as
    builtin types: a numpy integer as ``int``, ``True`` as ``1`` for
    ``drop_high`` and ``drop_low``, a numpy float as ``float``, and the dates in
    ``exclude`` and ``exclude_valuations`` as ``datetime.date``, sorted.

    Attributes
    ----------
    history_periods : int or None
        Use only the latest this many pairs at each link; ``None`` uses all.
    exclude : tuple of (date, int)
        Link ratios to leave out, as (origin period's first day, the age in
        months the ratio develops FROM).
    exclude_valuations : tuple of date
        Evaluation dates (each the last day of a month); every link ratio that
        develops INTO one of them is left out.
    drop_above, drop_below : float or None
        Leave out ratios strictly above ``drop_above`` or strictly below
        ``drop_below``.
    drop_high, drop_low : int
        How many of the highest and lowest ratios to leave out at each link.
    preserve : int
        The fewest ratios the bounds, and the trims, may leave at a link.
    trim_ties : str
        ``"origin"`` or ``"volume"``: how the trims break ties (module docstring).
    exhausted_exclusions : str
        ``"raise"`` or ``"keep"``: what happens when a rule would leave fewer
        than ``preserve`` ratios.
    zero_cells : str
        ``"observed"`` or ``"missing"``: what a cumulative of exactly zero is.
    """

    history_periods: int | None = None
    exclude: tuple[tuple[dt.date, int], ...] = ()
    exclude_valuations: tuple[dt.date, ...] = ()
    drop_above: float | None = None
    drop_below: float | None = None
    drop_high: int = 0
    drop_low: int = 0
    preserve: int = 1
    trim_ties: str = "origin"
    exhausted_exclusions: str = "raise"
    zero_cells: str = "observed"

    def __post_init__(self) -> None:
        history = self.history_periods
        if history is not None:
            if not _is_count(history) or history < 1:
                raise Refusal(
                    "invalid_option",
                    "history_periods must be a positive integer or None, got {given}",
                    option="history_periods",
                    given=history,
                )
            object.__setattr__(self, "history_periods", int(history))
        for name in ("drop_high", "drop_low"):
            value = getattr(self, name)
            if isinstance(value, bool | np.bool_):
                value = int(bool(value))
            if not _is_count(value) or value < 0:
                raise Refusal(
                    "invalid_option",
                    f"{name} must be a whole number of link ratios (0 or more), or True or "
                    "False, got {given}; one number applies at every development age",
                    option=name,
                    given=getattr(self, name),
                )
            object.__setattr__(self, name, int(value))
        if not _is_count(self.preserve) or self.preserve < 1:
            raise Refusal(
                "invalid_option",
                "preserve is the fewest link ratios the drop rules may leave at an age; it must "
                "be a whole number of 1 or more, got {given}",
                option="preserve",
                given=self.preserve,
            )
        object.__setattr__(self, "preserve", int(self.preserve))
        _choose("trim_ties", self.trim_ties, TRIM_TIES)
        _choose("exhausted_exclusions", self.exhausted_exclusions, EXHAUSTED)
        if self.zero_cells not in ZERO_CELLS:
            raise Refusal(
                "invalid_option",
                "zero_cells must be 'observed' or 'missing', got {given}. "
                "'observed' keeps a zero cumulative as data; 'missing' leaves out every link "
                "ratio with a zero at either end, as chainladder-python does",
                option="zero_cells",
                given=self.zero_cells,
            )
        for name in ("drop_above", "drop_below"):
            bound = getattr(self, name)
            if bound is None:
                continue
            if not _is_number(bound) or not math.isfinite(bound):
                raise Refusal(
                    "invalid_option",
                    f"{name} must be a finite number or None, got {{given}}; one number "
                    "applies at every development age",
                    option=name,
                    given=bound,
                )
            object.__setattr__(self, name, float(bound))
        if (
            self.drop_above is not None
            and self.drop_below is not None
            and not self.drop_below < self.drop_above
        ):
            raise Refusal(
                "invalid_option",
                f"drop_below must be below drop_above, got drop_below={self.drop_below!r} and "
                f"drop_above={self.drop_above!r}",
                option="drop_below",
                options=("drop_below", "drop_above"),
            )
        object.__setattr__(self, "exclude", _read_exclusions(self.exclude))
        object.__setattr__(self, "exclude_valuations", _read_valuations(self.exclude_valuations))

    @property
    def trims(self) -> bool:
        """Whether any trim is asked for."""
        return bool(self.drop_high or self.drop_low)


def is_all_history(rules: LinkRules) -> bool:
    """True when the rules keep every observed ratio the zero rule allows.

    That is: no history window, no explicit or valuation exclusion, no bound and
    no trim. ``preserve``, ``trim_ties`` and ``exhausted_exclusions`` only say
    how the bounds and trims act, so they do not count. This reads the SETTINGS,
    not what they removed from a given triangle: a window of 9 removes nothing
    from a 9 x 9 triangle today and one ratio from next year's.
    """
    return not settings_named(rules)


def settings_named(rules: LinkRules) -> list[tuple[str, str]]:
    """Each rule that can leave a link ratio out, as (option, phrase for a message).

    For example ``[("history_periods", "history_periods=5"), ("exclude",
    "exclude (2 link ratios)")]``; an empty list for rules that keep every
    observed ratio. The exclusions are counted, not listed, because a kernel
    names an origin by its period's first day and a caller may have written it
    another way.
    """
    named = []
    if rules.history_periods is not None:
        named.append(("history_periods", f"history_periods={rules.history_periods}"))
    if rules.exclude:
        count = len(rules.exclude)
        named.append(("exclude", f"exclude ({count} link ratio{'s' if count > 1 else ''})"))
    if rules.exclude_valuations:
        count = len(rules.exclude_valuations)
        phrase = f"exclude_valuations ({count} valuation{'s' if count > 1 else ''})"
        named.append(("exclude_valuations", phrase))
    for name in ("drop_above", "drop_below"):
        if getattr(rules, name) is not None:
            named.append((name, f"{name}={getattr(rules, name)!r}"))
    for name in ("drop_high", "drop_low"):
        if getattr(rules, name):
            named.append((name, f"{name}={getattr(rules, name)}"))
    return named


#: The option behind each reason a link ratio is left out.
OPTION_OF_REASON = {
    "zero_cell": "zero_cells",
    "undefined_ratio": "zero_cells",
    "history_window": "history_periods",
    "explicit_exclusion": "exclude",
    "valuation_exclusion": "exclude_valuations",
    "drop_above": "drop_above",
    "drop_below": "drop_below",
    "drop_low": "drop_low",
    "drop_high": "drop_high",
}


def _is_count(value) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _choose(name: str, value, choices: tuple[str, ...]) -> None:
    """Refuse a setting that is not one of its choices, naming it and its value."""
    if value not in choices:
        quoted = [repr(choice) for choice in choices]
        listed = (
            " or ".join(quoted)
            if len(quoted) == 2
            else f"{', '.join(quoted[:-1])}, or {quoted[-1]}"
        )
        raise Refusal(
            "invalid_option",
            f"{name} must be {listed}, got {{given}}",
            option=name,
            given=value,
        )


def _read_exclusions(exclude) -> tuple[tuple[dt.date, int], ...]:
    exclusions = []
    for origin, lag in exclude:
        try:
            exclusions.append((as_date(origin), lag))
        except (TypeError, ValueError) as exc:
            raise Refusal(
                "unreadable_label",
                f"exclude origin {{given}} is not a date: {_literal(exc)}",
                option="exclude",
                given=origin,
            ) from exc
    for _, lag in exclusions:
        if type(lag) is not int or lag < 1:
            raise Refusal(
                "invalid_option",
                "excluded development lags must be positive integer months, got {given}",
                option="exclude",
                given=lag,
            )
    if len(set(exclusions)) != len(exclusions):
        twice = [key for key in dict.fromkeys(exclusions) if exclusions.count(key) > 1]
        raise Refusal(
            "duplicate",
            "duplicate explicit exclusions: {cells}",
            option="exclude",
            cells=[RefusedCell(None, origin, lag) for origin, lag in twice],
        )
    return tuple(sorted(exclusions))


def _read_valuations(valuations) -> tuple[dt.date, ...]:
    if isinstance(valuations, str | bytes) or not hasattr(valuations, "__iter__"):
        raise Refusal(
            "invalid_option",
            "exclude_valuations must be a sequence of evaluation dates, got {given}",
            option="exclude_valuations",
            given=valuations,
        )
    days = []
    for value in valuations:
        try:
            day = as_date(value)
        except (TypeError, ValueError) as exc:
            raise Refusal(
                "unreadable_label",
                f"exclude_valuations {{given}} is not a date: {_literal(exc)}",
                option="exclude_valuations",
                given=value,
            ) from exc
        if (day + dt.timedelta(days=1)).day != 1:
            raise Refusal(
                "unreadable_label",
                "exclude_valuations {given} is not the last day of a month; an evaluation date "
                "is the last day of a development period",
                option="exclude_valuations",
                given=value,
            )
        days.append(day)
    twice = sorted({day for day in days if days.count(day) > 1})
    if twice:
        raise Refusal(
            "duplicate",
            f"exclude_valuations names {', '.join(day.isoformat() for day in twice)} more than "
            "once; list each evaluation date once",
            option="exclude_valuations",
        )
    return tuple(sorted(days))


@dataclass(frozen=True)
class LinkSelection:
    """Which link ratios :func:`select_links` kept, and why the others went.

    Each array has one row per origin and one column per link (from each
    development age to the next), ``n_links`` in all; the two flags have one
    value per link.

    Attributes
    ----------
    previous, following : numpy.ndarray
        The cumulatives at the two ends of each pair, NaN where the pair is not
        observed.
    ratio : numpy.ndarray
        ``following / previous``, NaN where the pair is not observed or
        ``previous`` is 0.
    observed : numpy.ndarray
        Whether both cells of the pair are observed.
    used : numpy.ndarray
        Whether the ratio enters the factor.
    reason : numpy.ndarray
        int8 index into :data:`REASONS`, -1 where the pair is not observed.
    bounds_skipped : numpy.ndarray
        Per link: ``preserve`` stopped ``drop_above``/``drop_below`` there.
    trimming_skipped : numpy.ndarray
        Per link: ``preserve`` stopped ``drop_high``/``drop_low`` there.
    """

    previous: np.ndarray
    following: np.ndarray
    ratio: np.ndarray
    observed: np.ndarray
    used: np.ndarray
    reason: np.ndarray
    bounds_skipped: np.ndarray
    trimming_skipped: np.ndarray

    @property
    def n_used(self) -> np.ndarray:
        """How many ratios each link's factor is estimated from."""
        return self.used.sum(axis=0)


def select_links(
    cum: np.ndarray,
    mask: np.ndarray,
    origin_periods,
    dev_grain_months: int,
    rules: LinkRules,
    n_links: int,
    *,
    raise_exhausted: bool = True,
) -> LinkSelection:
    """Apply ``rules`` to a cumulative grid, one link at a time.

    ``cum`` and ``mask`` are the grid's ``(n_w, n_d)`` cumulatives and observed
    cells; ``origin_periods`` each origin's first day, in order;
    ``dev_grain_months`` the months per development step. ``n_links`` may be
    more than ``n_d - 1`` (a fixed horizon beyond the observed development);
    those links have no pairs. The cumulatives must be finite and zero or more
    where observed, which the callers check first.

    A rule that would leave fewer than ``rules.preserve`` ratios at a link is
    skipped there and recorded. Under ``exhausted_exclusions="raise"`` the
    first such link is then refused with :class:`ibnr.errors.Refusal`, reason
    ``exclusions_exhausted``, unless ``raise_exhausted=False``, which leaves
    that to the caller (:func:`exhausted_refusal`).
    """
    n_w, n_d = cum.shape
    step = int(dev_grain_months)
    origins = [as_date(origin) for origin in origin_periods]
    shape = (n_w, n_links)
    previous = np.full(shape, np.nan)
    following = np.full(shape, np.nan)
    ratio = np.full(shape, np.nan)
    observed = np.zeros(shape, dtype=bool)
    reason = np.full(shape, -1, dtype=np.int8)
    bounds_skipped = np.zeros(n_links, dtype=bool)
    trimming_skipped = np.zeros(n_links, dtype=bool)
    explicit = set(rules.exclude)
    valuations = set(rules.exclude_valuations)
    missing = rules.zero_cells == "missing"
    for j in range(min(n_links, n_d - 1)):
        rows = np.flatnonzero(mask[:, j] & mask[:, j + 1])
        if not rows.size:
            continue
        c0, c1 = cum[rows, j], cum[rows, j + 1]
        # c0 is never negative here, so the ratio is defined exactly when c0 > 0
        r = np.full(rows.size, np.nan)
        positive = c0 > 0
        r[positive] = c1[positive] / c0[positive]
        previous[rows, j], following[rows, j], ratio[rows, j] = c0, c1, r
        observed[rows, j] = True
        why = np.zeros(rows.size, dtype=np.int8)
        # 2. the zero rule
        if missing:
            why[(c0 == 0) | (c1 == 0)] = _CODE["zero_cell"]
        else:
            why[~np.isfinite(r)] = _CODE["undefined_ratio"]
        # 3. the history window, counted by origin: every observed pair under
        # "missing", only the pairs still used under "observed"
        if rules.history_periods is not None:
            counted = np.arange(rows.size) if missing else np.flatnonzero(why == 0)
            older = counted[: max(0, counted.size - rules.history_periods)]
            why[older[why[older] == 0]] = _CODE["history_window"]
        # 4. explicit exclusions, by (origin, FROM age)
        if explicit:
            named = np.array([(origins[i], (j + 1) * step) in explicit for i in rows])
            why[named & (why == 0)] = _CODE["explicit_exclusion"]
        # 5. excluded valuations, by the LATER cell's evaluation date
        if valuations:
            named = np.array([month_end(origins[i], (j + 2) * step) in valuations for i in rows])
            why[named & (why == 0)] = _CODE["valuation_exclusion"]
        # 6. the bounds
        if rules.drop_above is not None or rules.drop_below is not None:
            above, below = _outside(r, why == 0, rules)
            gone = int(above.sum() + below.sum())
            if gone:
                if int((why == 0).sum()) - gone < rules.preserve:
                    bounds_skipped[j] = True
                else:
                    why[above] = _CODE["drop_above"]
                    why[below] = _CODE["drop_below"]
        # 7. the trims
        if rules.trims:
            live = np.flatnonzero(why == 0)
            if live.size:
                if live.size - rules.drop_high - rules.drop_low < rules.preserve:
                    trimming_skipped[j] = True
                else:
                    if rules.trim_ties == "volume":
                        order = np.lexsort((live, c0[live], r[live]))
                    else:
                        order = np.lexsort((live, r[live]))
                    ranked = live[order]
                    why[ranked[: rules.drop_low]] = _CODE["drop_low"]
                    if rules.drop_high:
                        why[ranked[ranked.size - rules.drop_high :]] = _CODE["drop_high"]
        reason[rows, j] = why
    used = reason == _CODE["included"]
    selection = LinkSelection(
        previous, following, ratio, observed, used, reason, bounds_skipped, trimming_skipped
    )
    if raise_exhausted and rules.exhausted_exclusions == "raise":
        for j in range(n_links):
            refusal = exhausted_refusal(selection, rules, j, step)
            if refusal is not None:
                raise refusal
    return selection


def _outside(r: np.ndarray, live: np.ndarray, rules: LinkRules) -> tuple[np.ndarray, np.ndarray]:
    """Which of the ``live`` ratios lie strictly above ``drop_above``, and strictly below
    ``drop_below``; a ratio equal to a bound is inside."""
    none = np.zeros_like(live)
    above = live & (r > rules.drop_above) if rules.drop_above is not None else none
    below = live & (r < rules.drop_below) if rules.drop_below is not None else none
    return above, below


def exhausted_refusal(
    selection: LinkSelection, rules: LinkRules, j: int, dev_grain_months: int
) -> Refusal | None:
    """The refusal for link ``j`` if a rule there would leave fewer than ``preserve``
    ratios, or ``None``.

    :func:`select_links` raises it itself under ``exhausted_exclusions="raise"``,
    for the first such link. A caller that runs its own checks link by link (the
    conventional fit does, so that the first link at fault is the one named,
    whichever check finds it) selects with ``raise_exhausted=False`` and asks
    here at each link. The bounds are named before the trims.
    """
    step = int(dev_grain_months)
    reasons = selection.reason[:, j]
    if selection.bounds_skipped[j]:
        # the bounds did not act, so every ratio they saw is included or trimmed now
        seen = np.isin(reasons, [_CODE[name] for name in ("included", "drop_low", "drop_high")])
        above, below = _outside(selection.ratio[:, j], seen, rules)
        n = int(seen.sum())
        return _bounds_exhausted(rules, j, step, n, n - int(above.sum() + below.sum()))
    if selection.trimming_skipped[j]:
        n = int(selection.used[:, j].sum())
        return _trims_exhausted(rules, j, step, n, n - rules.drop_high - rules.drop_low)
    return None


def _link(j: int, step: int) -> tuple[int, int]:
    return ((j + 1) * step, (j + 2) * step)


def _counted(name: str, count: int) -> str:
    """``drop_high`` for one ratio, ``drop_high=2`` for more."""
    return name if count == 1 else f"{name}={count}"


def _trims_exhausted(rules: LinkRules, j: int, step: int, n: int, left: int) -> Refusal:
    asked = [(name, getattr(rules, name)) for name in ("drop_high", "drop_low")]
    flags = [name for name, count in asked if count]
    rule = " and ".join(_counted(name, count) for name, count in asked if count)
    if left <= 0 and rules.preserve == 1:
        return Refusal(
            "exclusions_exhausted",
            f"{rule} would leave no link ratio {{links}}; pass exhausted_exclusions='keep' to "
            "keep that age's ratios untrimmed",
            option="exhausted_exclusions",
            options=("exhausted_exclusions", *flags),
            links=[_link(j, step)],
        )
    return Refusal(
        "exclusions_exhausted",
        f"{rule} would leave {max(left, 0)} of the {n} link ratio(s) {{links}}, fewer than "
        f"preserve={rules.preserve}; pass exhausted_exclusions='keep' to keep that age's "
        "ratios untrimmed, or lower preserve",
        option="exhausted_exclusions",
        options=("exhausted_exclusions", *flags, "preserve"),
        links=[_link(j, step)],
    )


def _bounds_exhausted(rules: LinkRules, j: int, step: int, n: int, left: int) -> Refusal:
    asked = [
        (name, getattr(rules, name))
        for name in ("drop_above", "drop_below")
        if getattr(rules, name) is not None
    ]
    rule = " and ".join(f"{name}={value!r}" for name, value in asked)
    return Refusal(
        "exclusions_exhausted",
        f"{rule} would leave {left} of the {n} link ratio(s) {{links}}, fewer than "
        f"preserve={rules.preserve}; pass exhausted_exclusions='keep' to apply neither bound "
        "at that age, or lower preserve",
        option="exhausted_exclusions",
        options=("exhausted_exclusions", *(name for name, _ in asked), "preserve"),
        links=[_link(j, step)],
    )


def link_factors(selection: LinkSelection, average: str) -> tuple[np.ndarray, np.ndarray]:
    """Each link's factor from the ratios ``selection`` uses, and how many there are.

    ``average`` is one of :data:`AVERAGES`:

    - ``"volume"``: ``sum(following) / sum(previous)``;
    - ``"simple"``: the mean of the ratios;
    - ``"regression"``: ``sum(previous * following) / sum(previous ** 2)``,
      least squares through the origin;
    - ``"median"``: the median of the ratios.

    The factor is NaN where no ratio is used, and where the volume average's
    denominator is not positive. No fallback is applied: the caller decides
    what an empty link, or a factor that is not a positive finite number,
    means. The sums run in origin order.
    """
    if average not in AVERAGES:
        _choose("average", average, AVERAGES)
    used = selection.used
    n_links = used.shape[1]
    factor = np.full(n_links, np.nan)
    for j in range(n_links):
        keep = used[:, j]
        if not keep.any():
            continue
        x = selection.previous[keep, j]
        y = selection.following[keep, j]
        if average == "volume":
            # Python's sum over numpy floats: plain additions in origin order, which is
            # what 0.7.2 computed, bit for bit
            denominator = sum(x)
            factor[j] = sum(y) / denominator if denominator > 0 else np.nan
        elif average == "regression":
            denominator = sum(x * x)
            factor[j] = sum(x * y) / denominator if denominator > 0 else np.nan
        else:
            reducer = np.mean if average == "simple" else np.median
            factor[j] = float(reducer(selection.ratio[keep, j]))
    return factor, used.sum(axis=0)
