"""Write ``tests/data/mack_default_pin.json``: Mack's answers before development options.

``tests/test_generalized_mack.py`` refits every case with the current code and
compares. Two things are pinned:

- ``kernel``: ``kernels.fit_mack_grid`` with no ``average`` and no ``links``,
  under both sigma rules and both zero rules, digested down to the raw bytes
  of every array on the fit (``f``, ``sigma2``, ``s``, ``n_obs``, ``n_pos``,
  the completed triangle), of ``msep_runoff``'s arrays and totals, and of the
  fit's ``to_arrow()`` payload; or to a refusal's reason and message. These
  must not move by a single bit.
- ``methods``: ``methods.mack``'s numbers under its defaults, and with
  ``sigma_rule="mack"`` and ``zero_cells="observed"``, stored as numbers (or a
  refusal's reason), because ``methods.mack`` now reads its factors through
  the shared link selection and may move in the last place.

The file was frozen from the branch ``feat/development-options`` (the code
before this change) by running this script against that branch's source:

    git archive feat/development-options src | tar -x -C <somewhere>
    uv run python scripts/freeze_mack_pin.py <somewhere>/src

The triangles are the six in ``scripts/freeze_conventional_pin.py`` (the five
public ones and a 30 x 30 one), the five public ones with one interior
cumulative set to 0, and every tenth clrd paid-loss cohort (needs
chainladder-python).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "mack_default_pin.json"
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_conventional_pin import grid_of, public_triangles  # noqa: E402

#: (sigma_rule, zero_cells) for the kernel's own defaults and their alternatives.
KERNEL_SETTINGS = [
    (rule, zeros) for rule in ("mack", "log_linear") for zeros in ("observed", "missing")
]
#: methods.mack's option sets that existed before.
METHOD_SETTINGS = {
    "default": {},
    "mack_rule": {"sigma_rule": "mack"},
    "observed": {"zero_cells": "observed"},
}


def with_zeros() -> dict[str, list]:
    """The five public triangles with the second origin's first cumulative set to 0."""
    out = {}
    for name, rows in json.loads(
        (ROOT / "tests" / "data" / "refusal_triangles.json").read_text("utf-8")
    ).items():
        second = sorted({y for y, _, _ in rows})[1]
        out[f"{name}_zero"] = [[y, d, 0.0 if (y, d) == (second, 12) else v] for y, d, v in rows]
    return out


def clrd_triangles() -> dict[str, list]:
    """Every tenth clrd paid-loss cohort, as [year, dev_lag, value] cells."""
    import chainladder as cl
    import pandas as pd

    frame = cl.load_sample("clrd")["CumPaidLoss"].to_frame(keepdims=True).reset_index()
    frame["year"] = pd.to_datetime(frame["origin"]).dt.year
    out = {}
    for k, ((company, line), group) in enumerate(frame.groupby(["GRNAME", "LOB"], sort=True)):
        if k % 10:
            continue
        out[f"clrd|{company}|{line}"] = [
            [int(y), int(d), float(v)]
            for y, d, v in zip(
                group["year"], group["development"], group["CumPaidLoss"], strict=True
            )
        ]
    return out


def kernel_digest(grid, sigma_rule: str, zero_cells: str) -> str:
    from ibnr.kernels.mack import fit_mack_grid

    h = hashlib.sha256()
    try:
        fit = fit_mack_grid(grid, sigma_rule=sigma_rule, zero_cells=zero_cells)
        h.update(fit.to_arrow())
        risk = fit.msep_runoff()
    except ValueError as refusal:
        h.update(f"refused|{getattr(refusal, 'reason', '')}|{refusal}".encode())
        return h.hexdigest()[:24]

    def floats(values) -> None:
        h.update(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())

    for name in ("f", "sigma2", "s", "n_obs", "n_pos", "full", "ultimate"):
        floats(getattr(fit, name))
    for name in ("msep", "process", "parameter", "msep_total", "process_total", "parameter_total"):
        floats(risk[name])
    return h.hexdigest()[:24]


def method_numbers(rows, options: dict) -> dict:
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
    try:
        result = methods.mack(cells, **options)
    except ValueError as refusal:
        return {"refused": getattr(refusal, "reason", "")}
    out = {}
    for table, columns in (
        ("origins", ("ultimate", "ibnr", "mack_se", "parameter_se", "process_se")),
        ("totals", ("ultimate", "ibnr", "mack_se", "parameter_se", "process_se")),
        ("development", ("factor", "cdf", "sigma", "std_err")),
    ):
        for column in columns:
            out[f"{table}.{column}"] = getattr(result, table)[column].to_pylist()
    return out


def pin(triangles: dict[str, list]) -> dict[str, dict]:
    kernel, numbers = {}, {}
    for name, rows in triangles.items():
        try:
            grid = grid_of(rows)
        except ValueError:
            continue  # not a run-off triangle: nothing to fit
        for rule, zeros in KERNEL_SETTINGS:
            kernel[f"{name}|{rule}|{zeros}"] = kernel_digest(grid, rule, zeros)
        for label, options in METHOD_SETTINGS.items():
            numbers[f"{name}|{label}"] = method_numbers(rows, options)
    return {"kernel": kernel, "methods": numbers}


def main() -> None:
    src = sys.argv[1]
    sys.path.insert(0, src)
    import ibnr

    assert Path(ibnr.__file__).resolve().is_relative_to(Path(src).resolve()), ibnr.__file__
    frozen = pin({**public_triangles(), **with_zeros(), **clrd_triangles()})
    OUT.write_text(json.dumps(frozen, sort_keys=True) + "\n", "utf-8", newline="\n")
    print(
        f"wrote {OUT}: {len(frozen['kernel'])} kernel cases, {len(frozen['methods'])} method cases"
    )


if __name__ == "__main__":
    main()
