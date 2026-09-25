"""Refusal: the one exception ibnr raises for input it will not answer.

``Refusal`` is a ``ValueError``, so code that catches ``ValueError`` keeps
working. It carries a reason code from a closed list, :data:`REASONS`, so a
caller can tell one refusal from another without reading the message, and the
cells, development ages and argument involved, with each origin written the
way the caller wrote it.

Every code has a kind (:data:`KIND`):

- ``"input"``: the request itself must change (a negative cumulative, an age
  that is not a whole number of months, an unknown option value);
- ``"model"``: the request is well formed, but the method cannot answer it with
  these options; another option or another method can.

Codes are never renamed or removed. A new code may be added in a patch release
and is named in the CHANGELOG, so a caller should handle a code it does not
know by its kind.

Kind ``"input"``:

- ``invalid_option``: an argument has the wrong type, is out of range, is not
  one of its choices, does not apply to this method, or conflicts with another.
- ``invalid_table``: a table argument is not a table Arrow can read, lacks a
  column, has no rows, or has a column of the wrong type.
- ``missing_value``: a null or NaN where a value is needed.
- ``not_finite``: an infinite amount.
- ``unreadable_label``: a period or date label that cannot be read (an unknown
  form, a two-digit year, a date that is neither a first nor a last day).
- ``invalid_age``: a ``dev_lag`` that is not a positive whole number of months.
- ``grain_mismatch``: ages or labels that do not match the development step.
- ``duplicate``: the same cell, origin or exclusion given twice, including one
  period written two ways.
- ``negative_cumulative``: a cumulative amount below zero.
- ``origin_gap``: an origin period missing between the first and the last.
- ``not_run_off``: cells that are not a run-off triangle.
- ``not_in_triangle``: a reference to something the triangle does not have.
- ``origin_not_covered``: a per-origin input that lacks one of the origins.

Kind ``"model"``:

- ``no_link_ratio``: no link ratio is left at an age to estimate a factor from,
  or the ones left give a factor of 0 (every origin closing at zero), and the
  options ask to refuse rather than use 1.0.
- ``exclusions_exhausted``: the trims (``drop_high``/``drop_low``) or the bounds
  (``drop_above``/``drop_below``) would leave fewer link ratios at an age than
  ``preserve`` allows.
- ``variance_not_estimable``: Mack's sigma (or a tail's) cannot be estimated.
- ``negative_increment``: a negative incremental amount where the model needs
  zero or more.
- ``zero_increment``: a zero incremental amount where the model needs more.
- ``negative_fitted_mean``: a fitted mean of zero or below where the model's
  distribution needs it positive.
- ``not_identified``: too few cells or points to estimate what was asked.
- ``degenerate_fit``: the fit exists only on a boundary, or gives observed data
  zero probability.
- ``did_not_converge``: an iterative fit did not settle within its limit.
- ``tail_not_decaying``: a fitted tail curve does not decay, so its product
  does not converge.
- ``empty_residual_pool``: a bootstrap has no residual left to resample.
- ``result_not_finite``: the inputs are finite and the answer is not, such as a
  link ratio, factor or sum past the largest double, or a standard error whose
  square falls below the smallest double and reads as 0.
- ``not_supported``: a combination ibnr deliberately does not answer.
- ``negative_projection``: a projected cumulative or ultimate below zero from
  data that is zero or more.

This module imports the standard library only, so importing it costs nothing
at start-up, and both ``ibnr.kernels`` and ``ibnr.methods`` can import it.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
import re
from dataclasses import dataclass
from typing import Any

__all__ = ["KIND", "MAX_CELLS", "REASONS", "Refusal", "RefusedCell"]

#: Every reason code and its kind, in the order the docs list them.
_CODES = (
    # the request itself must change
    ("invalid_option", "input"),
    ("invalid_table", "input"),
    ("missing_value", "input"),
    ("not_finite", "input"),
    ("unreadable_label", "input"),
    ("invalid_age", "input"),
    ("grain_mismatch", "input"),
    ("duplicate", "input"),
    ("negative_cumulative", "input"),
    ("origin_gap", "input"),
    ("not_run_off", "input"),
    ("not_in_triangle", "input"),
    ("origin_not_covered", "input"),
    # the request is well formed; these options or this method cannot answer it
    ("no_link_ratio", "model"),
    ("exclusions_exhausted", "model"),
    ("variance_not_estimable", "model"),
    ("negative_increment", "model"),
    ("zero_increment", "model"),
    ("negative_fitted_mean", "model"),
    ("not_identified", "model"),
    ("degenerate_fit", "model"),
    ("did_not_converge", "model"),
    ("tail_not_decaying", "model"),
    ("empty_residual_pool", "model"),
    ("result_not_finite", "model"),
    ("not_supported", "model"),
    ("negative_projection", "model"),
)

#: The closed list of reason codes.
REASONS: tuple[str, ...] = tuple(code for code, _ in _CODES)

#: Each reason code's kind: ``"input"`` or ``"model"``.
KIND: dict[str, str] = dict(_CODES)

#: The most cells, and the most rows, a refusal keeps; ``count`` holds the total.
MAX_CELLS = 100

#: How many cells, origins or links a message names before "and N more".
_SHOWN = 5

#: The placeholders a message template may use.
_PLACEHOLDERS = re.compile(r"\{(cells|origins|links|given)\}")

#: What a brace in the caller's own text is held as inside a template, so that a
#: column name or a label spelling ``{given}`` is printed as written. These are
#: two characters from Unicode's private-use area, which no text of ibnr's uses.
_OPEN, _CLOSE = chr(0xE000), chr(0xE001)
_HOLD = str.maketrans({"{": _OPEN, "}": _CLOSE})
_RELEASE = str.maketrans({_OPEN: "{", _CLOSE: "}"})


def _literal(text: Any) -> str:
    """The caller's text for a message template, with its braces never read as placeholders."""
    return str(text).translate(_HOLD)


@dataclass(frozen=True)
class RefusedCell:
    """One cell, or one origin, a refusal is about.

    Attributes
    ----------
    origin : object
        The origin as the caller wrote it (an int, a str, a date or a
        datetime). ``None`` when the caller never wrote this period, such as a
        period missing from the middle of the triangle, and in a refusal raised
        by a kernel, which does not know the caller's labels.
    origin_period : datetime.date or None
        The first day of the origin period; ``None`` if the label could not be
        read.
    dev_lag : int or None
        Months from the start of the origin period; ``None`` when the refusal is
        about the origin as a whole (its premium, for example).
    value : float or None
        The amount sent for this cell, or the premium for this origin; ``None``
        when there is none (a missing cell).
    """

    origin: object = None
    origin_period: dt.date | None = None
    dev_lag: int | None = None
    value: float | None = None


class Refusal(ValueError):
    """ibnr will not answer this input, and says why with a reason code.

    ``str(refusal)`` is the message, as for any ``ValueError``. The fields say
    the same thing in a form a program can use. (ibnr builds one as
    ``Refusal(reason, template, **fields)``: ``template`` is the message with the
    placeholders ``{cells}``, ``{origins}``, ``{links}`` and ``{given}``, filled
    in from the fields, and ``quoted=True`` puts text labels in quotes. A caller
    only reads refusals; the fields below and :meth:`to_dict` are the contract.)

    Attributes
    ----------
    reason : str
        One of :data:`REASONS`.
    kind : str
        ``KIND[reason]``: ``"input"`` or ``"model"``.
    method : str or None
        The ``ibnr.methods`` function that refused, such as ``"mack"``; ``None``
        when a kernel was called directly.
    option : str or None
        The argument at fault, named as the called function names it
        (``"cells"``, ``"premium"``, ``"average"``).
    column : str or None
        The column of a table argument (``"origin_period"``, ``"dev_lag"``,
        ``"value"``, ``"premium"``).
    options : tuple of str
        Every argument involved, ``option`` first, for a conflict between two.
    given : object
        The argument's value as passed, for a refused option or label;
        ``None`` otherwise.
    cells : tuple of RefusedCell
        The cells or origins at fault, sorted by origin period and age, at most
        :data:`MAX_CELLS` of them.
    links : tuple of (int, int)
        ``(from_dev_lag, to_dev_lag)`` in months, for a refusal about a
        development age.
    rows : tuple of int
        0-based row positions in the table argument, when a cell cannot be
        named (a missing origin, for example); at most :data:`MAX_CELLS`.
    count : int
        How many cells, rows or items are at fault in all, which can be more
        than ``cells`` keeps.
    """

    def __init__(
        self,
        reason: str,
        template: str,
        *,
        option: str | None = None,
        column: str | None = None,
        options: tuple[str, ...] = (),
        given: Any = None,
        cells=(),
        links=(),
        rows=(),
        count: int | None = None,
        method: str | None = None,
        quoted: bool = False,
    ) -> None:
        if reason not in KIND:
            raise ValueError(f"unknown refusal reason {reason!r}; the codes are {REASONS}")
        cells = tuple(sorted(cells, key=_cell_order))
        rows = tuple(int(row) for row in rows)
        links = tuple((int(a), int(b)) for a, b in links)
        found = max(len(cells), len(rows), len(links))
        if count is None:
            count = found
        elif count < found:
            raise ValueError(f"count {count} is less than the {found} items the refusal names")
        if option is not None and not options:
            options = (option,)
        self.reason = reason
        self.kind = KIND[reason]
        self.method = method
        self.option = option
        self.column = column
        self.options = tuple(options)
        self.given = given
        self.cells = cells[:MAX_CELLS]
        self.links = links
        self.rows = rows[:MAX_CELLS]
        self.count = int(count)
        self._template = template
        self._quoted = quoted
        super().__init__(self._render())

    # -- the message -----------------------------------------------------------

    def _render(self) -> str:
        values = {
            "cells": lambda: _cells_text(self.cells, self.count, self._quoted),
            "origins": lambda: _origins_text(self.cells, self._quoted),
            "links": lambda: "; ".join(f"from {a} to {b} months" for a, b in self.links),
            "given": lambda: _given_text(self.given),
        }
        # one pass: text put in for a placeholder is never read for another one
        message = _PLACEHOLDERS.sub(lambda match: values[match.group(1)](), self._template)
        return message.translate(_RELEASE)

    # -- copies ----------------------------------------------------------------

    def _fields(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "column": self.column,
            "options": self.options,
            "given": self.given,
            "cells": self.cells,
            "links": self.links,
            "rows": self.rows,
            "count": self.count,
            "method": self.method,
            "quoted": self._quoted,
        }

    def __reduce__(self):
        # Without this, pickling calls Refusal(message), which fails: a process
        # pool sends exceptions between processes by pickling them.
        return (_rebuild, (self.reason, self._template, self._fields()))

    def _replace(self, **changes: Any) -> Refusal:
        """A copy with some fields (or the template) changed and the message rendered again."""
        template = changes.pop("template", self._template)
        fields = self._fields()
        if self.count == max(len(self.cells), len(self.rows), len(self.links)):
            # the count was the number of items named, so it follows the new ones;
            # a count above that (items cut at the cap) is kept as it is
            fields["count"] = None
        fields.update(changes)
        return Refusal(self.reason, template, **fields)

    def _relabeled(self, *, method: str, label_of) -> Refusal:
        """The same refusal, from ``method``, with the caller's origin labels.

        ``label_of(period_start)`` returns the caller's label for a period, or
        ``None`` for a period the caller never wrote. Every cell with no label
        gets the one ``label_of`` gives, and the message is rendered again from
        its template.
        """
        cells = []
        for cell in self.cells:
            if cell.origin is None and cell.origin_period is not None:
                label = label_of(cell.origin_period)
                if label is not None:
                    cell = RefusedCell(label, cell.origin_period, cell.dev_lag, cell.value)
            cells.append(cell)
        return self._replace(cells=tuple(cells), method=method)

    # -- the payload -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The refusal as plain JSON types: no NaN or infinity, no dates.

        ``json.dumps(refusal.to_dict(), allow_nan=False)`` always succeeds. A
        date or datetime becomes its ISO string; NaN and infinity become
        ``None`` (the message still says which); a ``given`` that is not a
        JSON scalar or a list of them becomes its ``repr``.
        """
        return {
            "reason": self.reason,
            "kind": self.kind,
            "method": self.method,
            "option": self.option,
            "column": self.column,
            "options": list(self.options),
            "given": _json_given(self.given),
            "cells": [
                {
                    "origin": _json_scalar(cell.origin),
                    "origin_period": _json_scalar(cell.origin_period),
                    "dev_lag": _json_scalar(cell.dev_lag),
                    "value": _json_scalar(cell.value),
                }
                for cell in self.cells
            ],
            "links": [[a, b] for a, b in self.links],
            "rows": list(self.rows),
            "count": self.count,
            "message": str(self),
        }


def _rebuild(reason: str, template: str, fields: dict[str, Any]) -> Refusal:
    return Refusal(reason, template, **fields)


# -- rendering ------------------------------------------------------------------


def _cell_order(cell: RefusedCell) -> tuple:
    start = cell.origin_period
    lag = cell.dev_lag
    return (start is not None, start or dt.date.min, lag is not None, lag or 0)


def _label(cell: RefusedCell, quoted: bool) -> str:
    """An origin as a message prints it: the caller's label, with no quotes."""
    origin = cell.origin
    if origin is None:
        # a kernel's refusal, or a period nobody wrote: named by its first day
        return cell.origin_period.isoformat() if cell.origin_period else "an origin"
    if isinstance(origin, dt.datetime):
        return origin.isoformat(sep=" ")
    if isinstance(origin, dt.date):
        return origin.isoformat()
    if isinstance(origin, str):
        return repr(origin) if quoted else origin
    return str(origin)


def _plain(value: Any) -> Any:
    """A numpy scalar as the Python value it holds (``np.float64(-0.1)`` as ``-0.1``).

    Found by its module rather than imported, so this module keeps to the
    standard library.
    """
    kind = type(value)
    # a datetime64 is left as it is: .item() gives nanoseconds as a bare int
    if kind.__module__ == "numpy" and kind.__name__[:10] not in ("datetime64", "timedelta6"):
        try:
            return value.item()
        except (TypeError, ValueError):  # an array, which has no one value
            return value
    return value


def _given_text(value: Any) -> str:
    """A passed value as a message prints it: text quoted, a date as ISO, else its repr."""
    value = _plain(value)
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, tuple):
        inner = ", ".join(_given_text(item) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    return repr(value)


def _listed(items: list[str], total: int) -> str:
    """``a, b and c``, or ``a, b, c, d, e and 12 more`` when ``total`` is larger."""
    if total > len(items):
        return f"{', '.join(items)} and {total - len(items)} more"
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _cells_text(cells: tuple[RefusedCell, ...], count: int, quoted: bool) -> str:
    shown = []
    for cell in cells[:_SHOWN]:
        label = _label(cell, quoted)
        shown.append(label if cell.dev_lag is None else f"({label}, {cell.dev_lag} months)")
    # count says how many cells there were only when the list was cut at the cap;
    # otherwise it may count rows or links as well
    total = max(count, len(cells)) if len(cells) >= MAX_CELLS else len(cells)
    return _listed(shown, total)


def _origins_text(cells: tuple[RefusedCell, ...], quoted: bool) -> str:
    labels = list(dict.fromkeys(_label(cell, quoted) for cell in cells))
    return _listed(labels[:_SHOWN], len(labels))


# -- JSON -----------------------------------------------------------------------


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, dt.date):  # a datetime is a date too
        return value.isoformat()
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return repr(value)


def _json_given(value: Any) -> Any:
    if isinstance(value, list | tuple):
        if all(_is_json_scalar(item) for item in value):
            return [_json_scalar(item) for item in value]
        return repr(value)
    if _is_json_scalar(value):
        return _json_scalar(value)
    return repr(value)


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, bool | str | numbers.Real | dt.date)
