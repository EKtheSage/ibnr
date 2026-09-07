"""Shared fixtures. Every transformation test runs against BOTH ibis backends
(duckdb and polars) via the ``backend_name`` fixture; divergences are bugs or
must be documented in transforms.py.

Why both backends: ibis is the dataframe frontend (CLAUDE.md design decision 2)
and duckdb/polars are equally supported, but the ibis polars backend (10.x) has
NO window-function support at all - every ``WindowFunction`` raises
``OperationNotDefinedError``, and there is no ``ScalarSubquery`` either. That
forces ``triangle/transforms.py`` to express running sums / lags as equi-join +
group-by. Parameterizing the whole suite on ``backend_name`` is what keeps those
rewrites honest: a formulation that silently works only on duckdb fails here.

Fixtures here are deliberately hand-built (no mart, no sample loader) so the
package's core invariants are testable without the Schedule P gold mart or any
optional extra.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.triangle.core import GRAIN_MONTHS


def _polars_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("polars") is not None


#: polars is an optional extra for users, but the dev group installs it, so a
#: skip here means a broken dev env rather than an expected configuration.
BACKENDS = [
    "duckdb",
    pytest.param(
        "polars",
        marks=pytest.mark.skipif(
            not _polars_available(),
            reason="polars backend not installed; `uv sync` installs it via the dev group",
        ),
    ),
]


@pytest.fixture(params=BACKENDS)
def backend_name(request) -> str:
    """Parameterizes any test that touches ibis so it runs on both backends.

    Test ids gain a ``[duckdb]`` / ``[polars]`` suffix, which is how a single
    backend can be targeted: ``pytest "tests/test_transforms.py::test_as_of[polars]"``.
    """
    return request.param


def d(iso: str) -> dt.date:
    """Terse ``date`` literal - triangle keys are dates, not timestamps."""
    return dt.date.fromisoformat(iso)


@pytest.fixture
def small_cumulative(backend_name) -> Triangle:
    """3 origins x up-to-3 yearly devs, one segment, cumulative paid.

    A minimal square upper triangle: origin 2020 seen at devs 12/24/36, 2021 at
    12/24, 2022 at 12 - i.e. exactly the cells a reserving actuary would have at
    12/31/2022. Values are chosen so every increment is distinct and hand-checkable
    (2020: 100 -> +50 -> +25).
    """
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


def _add_months(day: dt.date, months: int) -> dt.date:
    """First day of the month ``months`` after the month ``day`` sits in."""
    total = day.month - 1 + months
    return dt.date(day.year + total // 12, total % 12 + 1, 1)


def make_cohort_triangle(
    backend_name: str | None,
    cum: np.ndarray,
    *,
    start_year: int = 2010,
    loss_field: str = "paid_loss",
    segment: dict[str, str] | None = None,
    dev_grain: str = "Y",
) -> Triangle:
    """Single-cohort cumulative triangle from one ``(n_w, n_d)`` matrix.

    The one-LOB counterpart of ``make_multiline_triangle``, for the contracts
    that take a single triangle at a time (``kernels/contract.py::cohort_grid``,
    ``kernels/mack.py``, ``kernels/cdr.py``). Same conventions: NaN = unobserved
    and simply not emitted, yearly grain, origin ``i`` = Jan 1 of
    ``start_year + i``, dev index ``j`` = dev_lag ``12*(j+1)`` months, eval date
    = Dec 31 of ``start_year + i + j``. ``segment`` adds constant segment
    columns, which is how the "one cohort only" guards get exercised.

    ``dev_grain`` is ``"Y"``, ``"Q"`` or ``"M"`` and moves BOTH grains together,
    so the triangle stays a run-off staircase. With a step of ``n`` months,
    origin ``i`` starts ``n*i`` months after Jan 1 of ``start_year``, dev index
    ``j`` is dev_lag ``n*(j+1)`` months, and the eval date is the last day of
    the month that lands in. The default reproduces the annual rows exactly, a
    row-by-row comparison ``tests/test_cdr.py`` asserts; the other two grains
    exist so a test can build the triangle the one-year CDR now refuses.
    """
    step = GRAIN_MONTHS[dev_grain]
    rows = []
    n_w, n_d = cum.shape
    for i in range(n_w):
        origin = _add_months(dt.date(start_year, 1, 1), step * i)
        for j in range(n_d):
            if np.isnan(cum[i, j]):
                continue
            lag = step * (j + 1)
            row = {
                "origin_period": origin,
                "dev_lag": lag,
                "eval_date": _add_months(origin, lag) - dt.timedelta(days=1),
                "field": loss_field,
                "value": float(cum[i, j]),
            }
            rows.append({**(segment or {}), **row})
    df = pd.DataFrame(rows)
    kwargs = {} if backend_name is None else {"backend": backend_name}
    return Triangle.from_long(
        df, measure="cumulative", origin_grain=dev_grain, dev_grain=dev_grain, **kwargs
    )


def make_multiline_triangle(
    backend_name: str,
    cum_by_lob: dict[str, np.ndarray],
    *,
    start_year: int = 2010,
    premium_by_lob: dict[str, np.ndarray] | None = None,
    company: str = "0001",
    company_name: str | dict[str, str] | None = None,
    loss_field: str = "paid_loss",
    premium_field: str = "earned_premium",
) -> Triangle:
    """One-company multi-LOB cumulative triangle from (n_w, n_d) arrays.

    ``cum_by_lob`` maps line_of_business -> cumulative matrix, NaN = unobserved.
    Yearly grain: origin w = Jan 1 of start_year + w, dev step d (0-based) =
    dev_lag 12*(d+1) months, eval_date = Dec 31 of start_year + w + d.
    Premium (per lob, per origin) is emitted alongside each observed loss cell;
    latest_diagonal() recovers the booked value.

    Builder for the multi-LOB contracts (kernels/multiline.py, kernels/nn_contract.py),
    which consume one company at a time with >= 2 lines. Matrix layout mirrors the
    contracts' own ``(n_w, n_d)`` grids so expected values can be written as arrays.
    NaN means unobserved and is simply not emitted - the long format never densifies
    (absent = unobserved, zero = an explicit observation).

    ``company_name`` adds the mart's THIRD segment column, the display-only one
    ``kernels.nn_contract`` keeps out of the cohort key. Pass a string for one
    spelling, or ``{lob: name}`` to give a single company two spellings - which
    is the ambiguity ``nn_data`` must refuse rather than silently pool.
    """
    rows = []
    for lob, cum in cum_by_lob.items():
        name = company_name[lob] if isinstance(company_name, dict) else company_name
        prefix = (company,) if name is None else (company, name)
        n_w, n_d = cum.shape
        for w in range(n_w):
            for dev in range(n_d):
                if np.isnan(cum[w, dev]):
                    continue
                origin = dt.date(start_year + w, 1, 1)
                eval_date = dt.date(start_year + w + dev, 12, 31)
                rows.append(
                    (*prefix, lob, origin, 12 * (dev + 1), eval_date, loss_field, cum[w, dev])
                )
                if premium_by_lob is not None:
                    rows.append(
                        (
                            *prefix,
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
            *([] if company_name is None else ["company_name"]),
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
    square upper triangle (w + d < max(n_w, n_d)).

    The reference answer for ``obs_mask`` in the contract tests: a cell is
    observed iff its calendar diagonal has already elapsed.
    """
    w, d = np.meshgrid(np.arange(n_w), np.arange(n_d), indexing="ij")
    return (w + d) < max(n_w, n_d)


def sorted_long(t: Triangle) -> pd.DataFrame:
    """Canonical materialized form for cross-backend / cross-library comparison.

    duckdb and polars (and chainladder) disagree on row order and on the physical
    dtypes of dates/ints, none of which is semantically meaningful for a long-format
    triangle. Normalizing dtypes, then sorting on every key column and on the column
    names themselves, reduces "are these the same triangle?" to frame equality.
    """
    df = t.execute()
    df["origin_period"] = pd.to_datetime(df["origin_period"])
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    df["dev_lag"] = df["dev_lag"].astype("int64")
    df["value"] = df["value"].astype("float64")
    keys = [c for c in df.columns if c != "value"]
    return df.sort_values(keys).reset_index(drop=True)[sorted(df.columns)]


def assert_triangles_equal(a: Triangle, b: Triangle, atol: float = 1e-8) -> None:
    """Two triangles hold the same cells with the same values.

    Keys are compared exactly (a fabricated or dropped cell is a hard failure - that
    is the whole point of the sparsity conventions), values only to ``atol``. The
    1e-8 default is absolute rather than relative because Schedule P values are
    dollars in the 1e3-1e7 range where float64 round-trips through SQL engines are
    exact well below a cent; anything larger would be a real arithmetic difference.
    """
    fa, fb = sorted_long(a), sorted_long(b)
    assert list(fa.columns) == list(fb.columns)
    assert len(fa) == len(fb), f"row counts differ: {len(fa)} vs {len(fb)}"
    key_cols = [c for c in fa.columns if c != "value"]
    # check_dtype=False: backends return int32/int64 and date/datetime variants
    # interchangeably; only the values carry meaning.
    pd.testing.assert_frame_equal(fa[key_cols], fb[key_cols], check_dtype=False)
    pd.testing.assert_series_equal(
        fa["value"], fb["value"], check_exact=False, atol=atol, check_dtype=False
    )
