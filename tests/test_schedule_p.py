"""Adapter tests against the CAS Schedule P gold mart - resolved exactly like
production code: env var, else the GitHub release default (with local cache).
Skipped when no data source is reachable, so the package works without the
mart.

What this file protects: ``data/schedule_p.py`` is the only bridge between the
external CAS Schedule P warehouse and the Triangle layer. These tests pin the
field/segment mapping, the grain and dev-lag convention, and - most
importantly - that our long-format pivot reproduces the *published* wide paid
triangle mart cell for cell. That tie-out is what lets the Meyers-style
retrospectives be trusted: a silent column remap would shift every backtest.

Marker rationale (``mart``): these need real warehouse data, which is not
vendored. They auto-skip when neither ``IBNR_SCHEDULE_P_WAREHOUSE`` nor the
GitHub release is reachable, so ``uv run pytest`` is green on a clean checkout.
The package's core invariants stay covered by the hand-built public-sample
fixtures in ``conftest.py`` - the mart is an optional data source, never a
build dependency. Resolution deliberately goes through the same
``active_mart_path`` production uses rather than a test-only path, so a broken
resolver shows up as a skip-with-reason rather than a false pass.
"""

import datetime as dt
import os

import pytest

from ibnr.data.schedule_p import DEFAULT_SOURCE, active_mart_path, load_schedule_p

WAREHOUSE = os.environ.get("IBNR_SCHEDULE_P_WAREHOUSE") or DEFAULT_SOURCE


def mart_available() -> bool:
    """True when the configured data source (local warehouse or GitHub
    release) is reachable; the release path downloads once into the cache.

    Broad ``except`` on purpose: any failure mode (no gh CLI, no auth, offline,
    bad spec) should degrade to a clean skip, never a collection error.
    """
    try:
        return active_mart_path(WAREHOUSE).exists()
    except Exception:
        return False


# Resolved once at import: the release path may download, and the flag is also
# re-exported to ``test_statistical_mart.py`` so both files gate identically.
MART_AVAILABLE = mart_available()

pytestmark = [
    pytest.mark.mart,
    pytest.mark.skipif(
        not MART_AVAILABLE,
        reason="CAS Schedule P gold mart not reachable (no local warehouse, no gh release access)",
    ),
]


def test_active_mart_path_resolves():
    """Source resolution lands on a real parquet asset, whichever branch (local
    warehouse or cached GitHub release) supplied it."""
    p = active_mart_path(WAREHOUSE)
    assert p.exists()
    assert p.suffix == ".parquet"


@pytest.fixture(scope="module")
def wkcomp():
    """Whole workers-comp slice, loaded once per module (the parquet scan is
    the expensive part; all downstream filtering is lazy ibis)."""
    return load_schedule_p(WAREHOUSE, lines=["workers_compensation"])


def test_mapping_and_validity(wkcomp):
    """The adapter's contract with the mart: yearly-origin/yearly-dev grain,
    cumulative measure, the exact field set (note ``reported_loss`` is the
    derived net-of-bulk incurred that milestone 2 showed to be load-bearing),
    the segment columns, and dev lags as months 12..120. Also asserts a single
    company's triangle passes strict Triangle validation."""
    assert wkcomp.meta.grain == "OYDY"
    assert wkcomp.meta.measure == "cumulative"
    assert set(wkcomp.fields) == {
        "paid_loss",
        "incurred_loss",
        "bulk_loss",
        "case_reserve",
        "earned_premium",
        "earned_premium_direct",
        "reported_loss",
    }
    assert set(wkcomp.segments) == {"company_code", "company_name", "line_of_business"}
    # dev_lag is always months from origin START, first diagonal = 12 (CLAUDE.md
    # convention); Schedule P gives 10 development years.
    assert wkcomp.dev_lags == [12 * i for i in range(1, 11)]
    import ibis

    # 26956 is a large, complete-square workers-comp filer - a stable spot
    # check that exists in every publish of the mart.
    one = wkcomp.filter(ibis._.company_code == "26956")
    assert one.validate(strict=True) == []


def test_paid_triangle_ties_to_wide_mart():
    """Our pivot of the training mart must equal the published wide paid mart.

    The gold publish ships both a long training mart and a pre-pivoted wide
    paid triangle. Reading the long one and pivoting it ourselves must
    reproduce the wide one exactly - an independent check that the
    origin/dev/eval mapping is right, since the two are produced by different
    code paths. Exact equality (approx only for float repr) is expected: this
    is a reshape, not a computation.
    """
    import duckdb

    wide_path = active_mart_path(WAREHOUSE, mart="mart_paid_loss_triangle")
    ref = (
        duckdb.sql(
            f"select * from read_parquet('{wide_path.as_posix()}') "
            "where company_code = '26956' and line_of_business = 'workers_compensation'"
        )
        .df()
        .set_index("accident_year")
        .sort_index()
    )
    t = load_schedule_p(WAREHOUSE, lines=["workers_compensation"], companies=["26956"])
    ours = t.select_fields("paid_loss").to_wide()
    # Wide mart columns are development *years* ("1".."10"); ours are months.
    for ay, row in ours.iterrows():
        for dev_months, val in row.items():
            assert val == pytest.approx(ref.loc[ay.year, str(dev_months // 12)]), (
                f"mismatch at AY {ay.year} dev {dev_months}"
            )


def test_as_of_gives_upper_triangle():
    """``as_of()`` on real mart data yields exactly the classic upper triangle.

    This is the backtesting primitive: train on the diagonal as at a past
    evaluation date, score the realized ultimates that arrive later. Anything
    leaking past the cutoff would silently make every retrospective optimistic.
    """
    import pandas as pd

    t = load_schedule_p(WAREHOUSE, lines=["workers_compensation"], companies=["26956"])
    train = t.as_of(dt.date(1997, 12, 31)).select_fields("paid_loss")
    df = train.execute()
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    df["origin_period"] = pd.to_datetime(df["origin_period"])
    assert (df["eval_date"] <= pd.Timestamp("1997-12-31")).all()
    # Restricted to 1988-1997 on purpose: the mart carries accident years past
    # the Meyers study window (through 2007), and counting the whole slice
    # would mix in post-window origins (CLAUDE.md gotcha).
    # 10 accident years 1988-1997 -> triangular cell count for the upper triangle
    ay = df["origin_period"].dt.year
    upper = df[(ay >= 1988) & (ay <= 1997)]
    assert len(upper) == 55  # 10 * 11 / 2
