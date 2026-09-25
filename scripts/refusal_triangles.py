"""Write ``tests/data/refusal_triangles.json``: five public triangles as plain cells.

``tests/test_refusal.py`` starts its seeded fuzz from raa, genins, ukmotor, abc
and mw2014. They are vendored so the fuzz runs on every CI leg, including those
without chainladder-python, which is where the samples come from.

Run: ``uv run python scripts/refusal_triangles.py`` (needs chainladder).
Each triangle is a list of ``[accident year, dev_lag in months, cumulative]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import chainladder as cl
import numpy as np
import pandas as pd

NAMES = ("raa", "genins", "ukmotor", "abc", "mw2014")
OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "refusal_triangles.json"


def main() -> None:
    triangles = {}
    for name in NAMES:
        long = cl.load_sample(name).to_frame(keepdims=True).reset_index()
        years = pd.to_datetime(long["origin"]).dt.year.to_numpy(dtype=np.int64)
        lags = long["development"].to_numpy(dtype=np.int64)
        values = long["values"].to_numpy(dtype=float)
        triangles[name] = [
            [int(y), int(d), float(v)] for y, d, v in zip(years, lags, values, strict=True)
        ]
    OUT.write_text(json.dumps(triangles) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} with {sum(len(t) for t in triangles.values())} cells")


if __name__ == "__main__":
    main()
