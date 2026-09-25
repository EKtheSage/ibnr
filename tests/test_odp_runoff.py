"""The full run-off ODP bootstrap's kernel: residual options, the draws, the refit, tails.

Sections:

1. the one-year CDR's ODP route did not move: every digest of
   ``tests/data/odp_cdr_pin.json`` (frozen from ``feat/tails`` by
   ``scripts/freeze_odp_cdr_pin.py``) is recomputed and must match bit for bit;
2. the residual options of ``fit_odp_bootstrap``: the adjustment, the pool,
   leverage, excluded cells, the scale, negative increments;
3. the draws: seeds, chunks, streams, the prior multiplier, finiteness;
4. the refit: one specification with the central fit, the position rules,
   the unit factor, tails;
5. chainladder-python 0.9.2 on shared random numbers (marker ``tieout``);
6. Monte Carlo agreement with chainladder-python and with R's
   ``BootChainLadder`` (frozen in ``tests/data/r_bootchainladder.json`` by
   ``scripts/r_bootchainladder.R``).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ibnr.kernels.grid import grid_from_columns

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import freeze_odp_cdr_pin as frozen  # noqa: E402

PUBLIC = json.loads((DATA / "refusal_triangles.json").read_text("utf-8"))
PIN = json.loads((DATA / "odp_cdr_pin.json").read_text("utf-8"))


def grid_of(rows, step: int = 12) -> dict:
    years, lags, values = zip(*rows, strict=True)
    return grid_from_columns(
        np.array([dt.date(y, 1, 1) for y in years], dtype="datetime64[D]"),
        np.array(lags),
        np.array(values, dtype=float),
        dev_grain_months=step,
        measure="cumulative",
    )


def rows_of(matrix, first_year: int = 2001, step: int = 12) -> list[list]:
    return [
        [first_year + i, step * (j + 1), float(v)]
        for i, row in enumerate(matrix)
        for j, v in enumerate(row)
        if v is not None and not np.isnan(v)
    ]


def premium_of(rows) -> dict:
    years = sorted({y for y, _, _ in rows})
    top = max(v for _, _, v in rows)
    amounts = np.linspace(1.0, 2.0, len(years)) * top
    return {dt.date(y, 1, 1): float(a) for y, a in zip(years, amounts, strict=True)}


def pseudo_of(boot, index) -> np.ndarray:
    """The simulated cumulative triangles for residual indices, as draw_runoff builds them."""
    mask = boot.obs_mask
    mean = np.where(mask, boot.fitted, 0.0)
    return np.cumsum(np.where(mask, boot.pool[index] * np.sqrt(np.abs(mean)) + mean, 0.0), axis=2)


RAA = grid_of(PUBLIC["raa"])
GENINS = grid_of(PUBLIC["genins"])


# -- 1. the one-year CDR's ODP route did not move ---------------------------------


def _moved(now: dict) -> list[str]:
    return [
        f"{name}|{key}"
        for name, digests in now.items()
        for key, value in digests.items()
        if PIN[name].get(key) != value
    ]


def test_the_one_year_cdr_odp_route_did_not_move_on_the_public_triangles():
    """The defaults of fit_odp_bootstrap, draw_next_increments and the CDR's
    odp_bootstrap generator give the bits feat/tails gave, refusals included.
    Mutation: make the kernel's default pool centred, or its default
    adjustment 'none'; the digests move."""
    import freeze_conventional_pin

    now = frozen.pin(freeze_conventional_pin.public_triangles())
    assert set(now) == {"raa", "genins", "ukmotor", "abc", "mw2014", "formula_30"}
    assert _moved(now) == []


@pytest.mark.tieout
def test_the_one_year_cdr_odp_route_did_not_move_on_clrd():
    pytest.importorskip("chainladder")
    import freeze_conventional_pin

    now = frozen.pin(freeze_conventional_pin.clrd_triangles())
    assert len(now) == 36
    assert _moved(now) == []
