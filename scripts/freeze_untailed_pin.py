"""Write ``tests/data/untailed_pin.json``: every untailed answer before tails.

``tests/test_tail.py`` refits every case with the current code and requires the
same answer, so adding tails moved no untailed number:

- ``conventional``: ``kernels.fit_conventional_grid`` for chain ladder,
  Bornhuetter-Ferguson and Cape Cod under the option sets of
  ``scripts/freeze_conventional_pin.py`` and the development options added
  since, digested as that script digests (factors, pattern, origin columns,
  link-ratio rows, summary);
- ``mack``: ``kernels.fit_mack_grid``'s arrays, ``msep_runoff`` and the
  ``to_arrow()`` payload under Mack's sigma rule, with no options and with each
  average and a window; ``mack_log_linear``: the same fits under the log-linear
  sigma rule, as numbers (the sigmas and ``msep_runoff``, the only answers the
  sigma rule reaches);
- ``methods``: every column of every table of ``chain_ladder``,
  ``bornhuetter_ferguson``, ``benktander``, ``cape_cod`` and ``mack``, under
  their defaults and a few options, each column digested from its values
  (floats by their exact bits, nulls as nulls). A column added later is not in
  the pin and is not compared; every column that was there must be the same.

Most of the pin is digests of exact bits, which the test requires to be the same
bit for bit. The cases that take ``log``, ``exp`` or a power with an inexact
result are stored as numbers instead, and the test compares them to 1e-14: the
pin is frozen on Windows, and the math libraries of Windows and Linux differ in
the last bits of those functions. Those cases are

- the log-linear sigma rule, which fits a line through ``log(sigma)`` and takes
  ``exp`` of it (``mack_log_linear``, and ``methods.mack``'s ``mack`` and
  ``mack_observed`` cases, since the log-linear rule is its default);
- Cape Cod's trend, ``(1 + trend) ** years`` (the ``gcc_trend`` conventional
  cases and the ``methods`` ``cape_cod`` case).

Cape Cod's decay of 0.75 takes powers too, but ``0.75 ** k`` is exact
(``3 ** k / 4 ** k``), so those cases stay digests. In a case stored as
numbers, the link-ratio rows and every column that is not a float stay
digests: no log, exp or trend reaches them.

The file was frozen from the branch ``feat/generalized-mack`` (the code before
tails) by running this script against that branch's source:

    git archive feat/generalized-mack src | tar -x -C <somewhere>
    uv run python scripts/freeze_untailed_pin.py <somewhere>/src

The triangles are the five public ones, a 30 x 30 one built from a formula,
the five public ones with one zero cell, and every fourteenth clrd paid-loss
cohort (needs chainladder-python).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "untailed_pin.json"
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_conventional_pin as conventional  # noqa: E402
from freeze_mack_pin import with_zeros  # noqa: E402

#: Option sets added to the conventional fit since freeze_conventional_pin.
CONVENTIONAL_EXTRA = {
    "regression": {"average": "regression"},
    "bounds": {"drop_above": 3.0, "drop_below": 1.001, "exhausted_exclusions": "keep"},
    "volume_ties": {"drop_high": 1, "trim_ties": "volume", "exhausted_exclusions": "keep"},
    "gcc_trend": {"trend": 0.04},
}
#: fit_mack_grid settings: (average, a history window or None).
MACK_SETTINGS = [
    ("volume", None),
    ("simple", None),
    ("regression", None),
    ("volume", 3),
]
#: methods calls: (function, options).
METHOD_CALLS = {
    "chain_ladder": ("chain_ladder", {}),
    "chain_ladder_options": (
        "chain_ladder",
        {"history_periods": 5, "drop_high": True, "average": "simple"},
    ),
    "bornhuetter_ferguson": ("bornhuetter_ferguson", {"expected_loss_ratio": 0.65}),
    "benktander": ("benktander", {"expected_loss_ratio": 0.65, "n_iters": 3}),
    "cape_cod": ("cape_cod", {"decay": 0.75, "trend": 0.02}),
    "mack": ("mack", {}),
    "mack_regression": ("mack", {"average": "regression", "sigma_rule": "mack"}),
    "mack_observed": ("mack", {"zero_cells": "observed"}),
}
#: The cases stored as numbers: their answers take log, exp or (1 + trend) ** t.
NUMBER_CONVENTIONAL = frozenset({"gcc_trend"})
NUMBER_METHODS = frozenset({"cape_cod", "mack", "mack_observed"})
#: The methods tables whose float columns are stored as numbers in those cases.
NUMBER_TABLES = frozenset({"origins", "development", "totals"})


def _floats(h, values) -> None:
    h.update(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())


def _flags(h, values) -> None:
    h.update(np.ascontiguousarray(np.asarray(values, dtype=bool)).tobytes())


def _refused(h, refusal) -> str:
    h.update(f"refused|{getattr(refusal, 'reason', '')}|{refusal}".encode())
    return h.hexdigest()[:24]


def _numbers(values) -> list[float]:
    """An array as a list of floats, which JSON writes and reads back exactly."""
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def conventional_numbers(fit) -> dict | str:
    """A conventional fit's factors, pattern and origin columns as numbers.

    The link-ratio rows and the factor summary take no trend, so they stay one
    digest of their bits.
    """
    h = hashlib.sha256()
    if isinstance(fit, ValueError):
        return _refused(h, fit)
    selection = fit.factor_selection
    for column in ("from_dev_lag", "previous", "following", "ratio"):
        _floats(h, selection[column])
    _flags(h, selection["included"])
    h.update("|".join(selection["reason"]).encode())
    h.update("|".join(str(origin) for origin in selection["origin_period"]).encode())
    summary = fit.factor_summary
    for column in ("from_dev_lag", "factor", "n_selected"):
        _floats(h, summary[column])
    for column in ("unity_fallback", "extreme_trimming_skipped"):
        _flags(h, summary[column])
    out: dict = {"link_ratios": h.hexdigest()[:24]}
    out["factors"] = _numbers(fit.factors)
    out["beta"] = _numbers(fit.beta)
    for column in conventional.ORIGIN_COLUMNS:
        out[f"origins.{column}"] = _numbers(fit.origins[column])
    return out


def conventional_pin(triangles: dict[str, list]) -> dict[str, str | dict]:
    from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional_grid

    out: dict[str, str | dict] = dict(conventional.pin(triangles))
    for name, rows in triangles.items():
        try:
            grid = conventional.grid_of(rows)
        except ValueError:
            continue
        starts = sorted({dt.date(y, 1, 1) for y, _, _ in rows})
        top = max(v for _, _, v in rows)
        premium = dict(zip(starts, np.linspace(1.0, 2.0, len(starts)) * top, strict=True))
        for label, options in CONVENTIONAL_EXTRA.items():
            for method, extra in conventional.METHOD_EXTRA.items():
                if "trend" in options and method != "gcc":
                    continue
                try:
                    fit = fit_conventional_grid(
                        grid,
                        ConventionalCandidate(method, **options, **extra),
                        premium=None if method == "cl" else premium,
                    )
                except ValueError as refusal:
                    fit = refusal
                key = f"{name}|{method}|{label}"
                if label in NUMBER_CONVENTIONAL:
                    out[key] = conventional_numbers(fit)
                else:
                    out[key] = conventional.digest(fit)
    return out


def _mack_fit(grid, sigma_rule: str, average: str, history: int | None):
    from ibnr.kernels.links import LinkRules
    from ibnr.kernels.mack import fit_mack_grid

    links = None if history is None else LinkRules(history_periods=history)
    return fit_mack_grid(grid, sigma_rule=sigma_rule, average=average, links=links)


def mack_digest(grid, average: str, history: int | None) -> str:
    """A Mack-rule fit's arrays, ``msep_runoff`` and ``to_arrow()``, by their bits."""
    h = hashlib.sha256()
    try:
        fit = _mack_fit(grid, "mack", average, history)
        h.update(fit.to_arrow())
        risk = fit.msep_runoff()
    except ValueError as refusal:
        return _refused(h, refusal)
    for name in ("f", "sigma2", "s", "n_obs", "n_pos", "full", "ultimate", "reserve"):
        _floats(h, getattr(fit, name))
    for name in ("msep", "process", "parameter", "msep_total", "process_total", "parameter_total"):
        _floats(h, risk[name])
    return h.hexdigest()[:24]


def mack_numbers(grid, average: str, history: int | None) -> dict | str:
    """A log-linear fit's sigmas and ``msep_runoff``, as numbers.

    The sigma rule reaches nothing else: the factors, the counts and the
    completed triangle are the Mack-rule fit's, which ``mack_digest`` pins by
    their bits.
    """
    try:
        fit = _mack_fit(grid, "log_linear", average, history)
        risk = fit.msep_runoff()
    except ValueError as refusal:
        return _refused(hashlib.sha256(), refusal)
    out = {"sigma2": _numbers(fit.sigma2)}
    for name in ("msep", "process", "parameter", "msep_total", "process_total", "parameter_total"):
        out[name] = _numbers(risk[name])
    return out


def column_digest(column) -> str:
    """A column's values: floats by their exact bits, nulls as nulls."""
    values = []
    for value in column.to_pylist():
        if isinstance(value, float):
            values.append(value.hex())
        elif isinstance(value, dt.date):
            values.append(value.isoformat())
        else:
            values.append(value)
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()[:24]


def method_digests(rows, function: str, options: dict, numbers: bool = False) -> dict | str:
    """Every column of a methods result, digested; with ``numbers``, the float
    columns of ``origins``, ``development`` and ``totals`` as their values."""
    import pyarrow as pa

    from ibnr import methods

    years, lags, values = zip(*rows, strict=True)
    cells = pa.table(
        {
            "origin_period": pa.array(years, pa.int64()),
            "dev_lag": pa.array(lags, pa.int64()),
            "value": pa.array(values, pa.float64()),
        }
    )
    options = dict(options)
    if function != "chain_ladder" and function != "mack":
        top = max(v for _, _, v in rows)
        years = sorted(set(years))
        options["premium"] = dict(
            zip(years, (np.linspace(1.0, 2.0, len(years)) * top).tolist(), strict=True)
        )
    try:
        result = getattr(methods, function)(cells, **options)
    except ValueError as refusal:
        return f"refused|{getattr(refusal, 'reason', '')}|{refusal}"
    out = {}
    for table in ("origins", "development", "link_ratios", "totals"):
        data = getattr(result, table)
        for name in data.column_names:
            floating = pa.types.is_floating(data[name].type)
            if numbers and table in NUMBER_TABLES and floating:
                out[f"{table}.{name}"] = data[name].to_pylist()
            else:
                out[f"{table}.{name}"] = column_digest(data[name])
    return out


def pin(triangles: dict[str, list]) -> dict:
    mack, log_linear, numbers = {}, {}, {}
    for name, rows in triangles.items():
        try:
            grid = conventional.grid_of(rows)
        except ValueError:
            continue  # not a run-off triangle: nothing to fit
        for average, history in MACK_SETTINGS:
            key = f"{name}|{average}|{history}"
            mack[key] = mack_digest(grid, average, history)
            log_linear[key] = mack_numbers(grid, average, history)
        for label, (function, options) in METHOD_CALLS.items():
            numbers[f"{name}|{label}"] = method_digests(
                rows, function, options, numbers=label in NUMBER_METHODS
            )
    return {
        "conventional": conventional_pin(triangles),
        "mack": mack,
        "mack_log_linear": log_linear,
        "methods": numbers,
    }


def triangles() -> dict[str, list]:
    return {**conventional.public_triangles(), **with_zeros(), **conventional.clrd_triangles()}


def main() -> None:
    src = sys.argv[1]
    sys.path.insert(0, src)
    import ibnr

    assert Path(ibnr.__file__).resolve().is_relative_to(Path(src).resolve()), ibnr.__file__
    frozen = pin(triangles())
    OUT.write_text(json.dumps(frozen, sort_keys=True) + "\n", "utf-8", newline="\n")
    print(
        f"wrote {OUT}: {len(frozen['conventional'])} conventional, {len(frozen['mack'])} mack, "
        f"{len(frozen['mack_log_linear'])} mack log-linear, {len(frozen['methods'])} methods cases"
    )


if __name__ == "__main__":
    main()
