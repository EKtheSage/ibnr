"""Shared fixtures. Every transformation test runs against BOTH ibis backends
(duckdb and polars) via the ``backend_name`` fixture; divergences are bugs or
must be documented in transforms.py."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle

BACKENDS = ["duckdb", "polars"]


@pytest.fixture(params=BACKENDS)
def backend_name(request) -> str:
    return request.param


def d(iso: str) -> dt.date:
    return dt.date.fromisoformat(iso)


@pytest.fixture
def small_cumulative(backend_name) -> Triangle:
    """3 origins x up-to-3 yearly devs, one segment, cumulative paid."""
    rows = [
        ("auto", "2020-01-01", 12, "2020-12-31", 100.0),
        ("auto", "2020-01-01", 24, "2021-12-31", 150.0),
        ("auto", "2020-01-01", 36, "2022-12-31", 175.0),
        ("auto", "2021-01-01", 12, "2021-12-31", 110.0),
        ("auto", "2021-01-01", 24, "2022-12-31", 165.0),
        ("auto", "2022-01-01", 12, "2022-12-31", 120.0),
    ]
    df = pd.DataFrame(rows, columns=["lob", "origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    return Triangle.from_long(df, measure="cumulative", backend=backend_name)


def make_multiline_triangle(
    backend_name: str,
    cum_by_lob: dict[str, np.ndarray],
    *,
    start_year: int = 2010,
    premium_by_lob: dict[str, np.ndarray] | None = None,
    company: str = "0001",
    loss_field: str = "paid_loss",
    premium_field: str = "earned_premium",
) -> Triangle:
    """One-company multi-LOB cumulative triangle from (n_w, n_d) arrays.

    ``cum_by_lob`` maps line_of_business -> cumulative matrix, NaN = unobserved.
    Yearly grain: origin w = Jan 1 of start_year + w, dev step d (0-based) =
    dev_lag 12*(d+1) months, eval_date = Dec 31 of start_year + w + d.
    Premium (per lob, per origin) is emitted alongside each observed loss cell;
    latest_diagonal() recovers the booked value.
    """
    rows = []
    for lob, cum in cum_by_lob.items():
        n_w, n_d = cum.shape
        for w in range(n_w):
            for dev in range(n_d):
                if np.isnan(cum[w, dev]):
                    continue
                origin = dt.date(start_year + w, 1, 1)
                eval_date = dt.date(start_year + w + dev, 12, 31)
                rows.append(
                    (company, lob, origin, 12 * (dev + 1), eval_date, loss_field, cum[w, dev])
                )
                if premium_by_lob is not None:
                    rows.append(
                        (
                            company,
                            lob,
                            origin,
                            12 * (dev + 1),
                            eval_date,
                            premium_field,
                            float(premium_by_lob[lob][w]),
                        )
                    )
    df = pd.DataFrame(
        rows,
        columns=[
            "company_code",
            "line_of_business",
            "origin_period",
            "dev_lag",
            "eval_date",
            "field",
            "value",
        ],
    )
    return Triangle.from_long(df, measure="cumulative", backend=backend_name)


def upper_mask(n_w: int, n_d: int) -> np.ndarray:
    """Boolean (n_w, n_d): True where origin w has dev step d observed in a
    square upper triangle (w + d < max(n_w, n_d))."""
    w, d = np.meshgrid(np.arange(n_w), np.arange(n_d), indexing="ij")
    return (w + d) < max(n_w, n_d)


def sorted_long(t: Triangle) -> pd.DataFrame:
    """Canonical materialized form for cross-backend / cross-library comparison."""
    df = t.execute()
    df["origin_period"] = pd.to_datetime(df["origin_period"])
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    df["dev_lag"] = df["dev_lag"].astype("int64")
    df["value"] = df["value"].astype("float64")
    keys = [c for c in df.columns if c != "value"]
    return df.sort_values(keys).reset_index(drop=True)[sorted(df.columns)]


def assert_triangles_equal(a: Triangle, b: Triangle, atol: float = 1e-8) -> None:
    fa, fb = sorted_long(a), sorted_long(b)
    assert list(fa.columns) == list(fb.columns)
    assert len(fa) == len(fb), f"row counts differ: {len(fa)} vs {len(fb)}"
    key_cols = [c for c in fa.columns if c != "value"]
    pd.testing.assert_frame_equal(fa[key_cols], fb[key_cols], check_dtype=False)
    pd.testing.assert_series_equal(
        fa["value"], fb["value"], check_exact=False, atol=atol, check_dtype=False
    )
