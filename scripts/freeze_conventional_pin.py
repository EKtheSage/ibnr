"""Write ``tests/data/conventional_selection_pin.json``: the conventional fit's answers
before ``kernels/links.py``, as digests of their raw bytes.

``tests/test_development_options.py`` refits every case with the current code
and requires the same digest, so no option set that existed before the shared
link selection moves a single bit: the factors, the development pattern, the
origin table's columns, every link ratio's row and reason, the per-age summary,
and each refusal's reason and message.

The file was frozen from commit 01e5c2f (the last commit before
``kernels/links.py``) by running this version of the script against that
commit's source:

    git archive 01e5c2f src | tar -x -C <somewhere>
    uv run python scripts/freeze_conventional_pin.py <somewhere>/src

Rerunning it against the current source must write the same file. The
triangles are the five public ones in ``tests/data/refusal_triangles.json``, a
30 x 30 one built from a formula, and every fourteenth clrd paid-loss cohort
that is a run-off triangle (needs chainladder-python). The test imports
``digest`` and ``pin`` from here.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "conventional_selection_pin.json"

#: The option sets that existed before kernels/links.py, by name.
OPTION_SETS = {
    "plain": {},
    "simple": {"average": "simple"},
    "median": {"average": "median"},
    "history_3": {"history_periods": 3},
    "history_5_both": {"history_periods": 5, "drop_high": True, "drop_low": True},
    "high": {"drop_high": True},
    "low": {"drop_low": True},
    "both_keep": {"drop_high": True, "drop_low": True, "exhausted_exclusions": "keep"},
    "both_raise": {"drop_high": True, "drop_low": True},
    "unity_low": {"unsupported_factor": "unity", "exhausted_exclusions": "keep", "drop_low": True},
    "missing": {"zero_cells": "missing", "unsupported_factor": "unity"},
    "missing_trim": {
        "zero_cells": "missing",
        "unsupported_factor": "unity",
        "drop_high": True,
        "exhausted_exclusions": "keep",
        "history_periods": 4,
    },
    "missing_simple": {"zero_cells": "missing", "average": "simple", "unsupported_factor": "unity"},
    "exclude_first": {"exclude": ((dt.date(1990, 1, 1), 12),), "unsupported_factor": "unity"},
}
METHOD_EXTRA = {"cl": {}, "bf": {"expected_loss_ratio": 0.65}, "gcc": {"decay": 0.75}}
ORIGIN_COLUMNS = (
    "latest_dev_lag",
    "latest",
    "beta",
    "expected_loss_ratio",
    "prior_ultimate",
    "ultimate",
    "reserve",
)


def digest(fit) -> str:
    """A short digest of a fit's bytes, or of a refusal's reason and message."""
    h = hashlib.sha256()
    if isinstance(fit, ValueError):
        h.update(f"refused|{getattr(fit, 'reason', '')}|{fit}".encode())
        return h.hexdigest()[:24]

    def floats(values) -> None:
        h.update(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())

    def flags(values) -> None:
        h.update(np.ascontiguousarray(np.asarray(values, dtype=bool)).tobytes())

    floats(fit.factors)
    floats(fit.beta)
    for column in ORIGIN_COLUMNS:
        floats(fit.origins[column])
    selection = fit.factor_selection
    for column in ("from_dev_lag", "previous", "following", "ratio"):
        floats(selection[column])
    flags(selection["included"])
    h.update("|".join(selection["reason"]).encode())
    h.update("|".join(str(origin) for origin in selection["origin_period"]).encode())
    summary = fit.factor_summary
    for column in ("from_dev_lag", "factor", "n_selected"):
        floats(summary[column])
    for column in ("unity_fallback", "extreme_trimming_skipped"):
        flags(summary[column])
    return h.hexdigest()[:24]


def public_triangles() -> dict[str, list]:
    """The five public triangles, and one 30 x 30 built from a formula.

    The large one has up to 29 link ratios at an age, where numpy's own sums add
    in a different order from a loop over the origins, so a change of summation
    order shows in its bits; the public triangles have at most 16.
    """
    triangles = json.loads((ROOT / "tests" / "data" / "refusal_triangles.json").read_text("utf-8"))
    rows = []
    for i in range(30):
        amount = 1000.0 + 37.0 * i
        for j in range(30 - i):
            rows.append([1990 + i, 12 * (j + 1), amount])
            amount *= 1.0 + 1.0 / (j + 1) ** 1.5 + 0.013 * ((7 * i + 3 * j) % 11)
    triangles["formula_30"] = rows
    return triangles


def clrd_triangles() -> dict[str, list]:
    """Every fourteenth clrd paid-loss cohort, as [year, dev_lag, value] cells."""
    import chainladder as cl
    import pandas as pd

    frame = cl.load_sample("clrd")["CumPaidLoss"].to_frame(keepdims=True).reset_index()
    frame["year"] = pd.to_datetime(frame["origin"]).dt.year
    out = {}
    for k, ((company, line), group) in enumerate(frame.groupby(["GRNAME", "LOB"], sort=True)):
        if k % 14:
            continue
        out[f"clrd|{company}|{line}"] = [
            [int(y), int(d), float(v)]
            for y, d, v in zip(
                group["year"], group["development"], group["CumPaidLoss"], strict=True
            )
        ]
    return out


def grid_of(rows):
    from ibnr.kernels.grid import grid_from_columns

    years, lags, values = zip(*rows, strict=True)
    return grid_from_columns(
        np.array([dt.date(y, 1, 1) for y in years], dtype="datetime64[D]"),
        np.array(lags),
        np.array(values, dtype=float),
        dev_grain_months=12,
        measure="cumulative",
    )


def pin(triangles: dict[str, list]) -> dict[str, str]:
    from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional_grid

    out = {}
    for name, rows in triangles.items():
        try:
            grid = grid_of(rows)
        except ValueError:
            continue  # not a run-off triangle: nothing to fit
        starts = sorted({dt.date(y, 1, 1) for y, _, _ in rows})
        top = max(v for _, _, v in rows)
        premium = dict(zip(starts, np.linspace(1.0, 2.0, len(starts)) * top, strict=True))
        for label, options in OPTION_SETS.items():
            for method, extra in METHOD_EXTRA.items():
                try:
                    fit = fit_conventional_grid(
                        grid,
                        ConventionalCandidate(method, **options, **extra),
                        premium=None if method == "cl" else premium,
                    )
                except ValueError as refusal:
                    fit = refusal
                out[f"{name}|{method}|{label}"] = digest(fit)
    return out


def main() -> None:
    src = sys.argv[1]
    sys.path.insert(0, src)
    import ibnr

    assert Path(ibnr.__file__).resolve().is_relative_to(Path(src).resolve()), ibnr.__file__
    cases = pin({**public_triangles(), **clrd_triangles()})
    OUT.write_text(json.dumps(cases, indent=0, sort_keys=True) + "\n", "utf-8", newline="\n")
    print(f"wrote {OUT}: {len(cases)} cases")


if __name__ == "__main__":
    main()
