"""Write ``tests/data/untailed_pin.json``: every untailed answer before tails, as digests.

``tests/test_tail.py`` refits every case with the current code and requires the
same digest, so adding tails moved no untailed number by a single bit:

- ``conventional``: ``kernels.fit_conventional_grid`` for chain ladder,
  Bornhuetter-Ferguson and Cape Cod under the option sets of
  ``scripts/freeze_conventional_pin.py`` and the development options added
  since, digested as that script digests (factors, pattern, origin columns,
  link-ratio rows, summary);
- ``mack``: ``kernels.fit_mack_grid``'s arrays, ``msep_runoff`` and the
  ``to_arrow()`` payload, with no options and with each average and a window;
- ``methods``: every column of every table of ``chain_ladder``,
  ``bornhuetter_ferguson``, ``benktander``, ``cape_cod`` and ``mack``, under
  their defaults and a few options, each column digested from its values
  (floats by their exact bits, nulls as nulls). A column added later is not in
  the pin and is not compared; every column that was there must be the same.

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


def _floats(h, values) -> None:
    h.update(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())


def _refused(h, refusal) -> str:
    h.update(f"refused|{getattr(refusal, 'reason', '')}|{refusal}".encode())
    return h.hexdigest()[:24]


def conventional_pin(triangles: dict[str, list]) -> dict[str, str]:
    from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional_grid

    out = conventional.pin(triangles)
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
                out[f"{name}|{method}|{label}"] = conventional.digest(fit)
    return out


def mack_digest(grid, average: str, history: int | None) -> str:
    from ibnr.kernels.links import LinkRules
    from ibnr.kernels.mack import fit_mack_grid

    h = hashlib.sha256()
    links = None if history is None else LinkRules(history_periods=history)
    try:
        fit = fit_mack_grid(grid, sigma_rule="log_linear", average=average, links=links)
        h.update(fit.to_arrow())
        risk = fit.msep_runoff()
    except ValueError as refusal:
        return _refused(h, refusal)
    for name in ("f", "sigma2", "s", "n_obs", "n_pos", "full", "ultimate", "reserve"):
        _floats(h, getattr(fit, name))
    for name in ("msep", "process", "parameter", "msep_total", "process_total", "parameter_total"):
        _floats(h, risk[name])
    return h.hexdigest()[:24]


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


def method_digests(rows, function: str, options: dict) -> dict[str, str] | str:
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
            out[f"{table}.{name}"] = column_digest(data[name])
    return out


def pin(triangles: dict[str, list]) -> dict:
    mack, numbers = {}, {}
    for name, rows in triangles.items():
        try:
            grid = conventional.grid_of(rows)
        except ValueError:
            continue  # not a run-off triangle: nothing to fit
        for average, history in MACK_SETTINGS:
            mack[f"{name}|{average}|{history}"] = mack_digest(grid, average, history)
        for label, (function, options) in METHOD_CALLS.items():
            numbers[f"{name}|{label}"] = method_digests(rows, function, options)
    return {"conventional": conventional_pin(triangles), "mack": mack, "methods": numbers}


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
        f"{len(frozen['methods'])} methods cases"
    )


if __name__ == "__main__":
    main()
