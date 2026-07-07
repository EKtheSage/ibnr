"""Adapter tests against the CAS Schedule P gold mart — resolved exactly like
production code: env var, else the GitHub release default (with local cache).
Skipped when no data source is reachable, so the package works without the
mart."""

import datetime as dt
import os

import pytest

from ibnr.data.schedule_p import DEFAULT_SOURCE, active_mart_path, load_schedule_p

WAREHOUSE = os.environ.get("IBNR_SCHEDULE_P_WAREHOUSE") or DEFAULT_SOURCE


def mart_available() -> bool:
    """True when the configured data source (local warehouse or GitHub
    release) is reachable; the release path downloads once into the cache."""
    try:
        return active_mart_path(WAREHOUSE).exists()
    except Exception:
        return False


MART_AVAILABLE = mart_available()

pytestmark = [
    pytest.mark.mart,
    pytest.mark.skipif(
        not MART_AVAILABLE,
        reason="CAS Schedule P gold mart not reachable (no local warehouse, no gh release access)",
    ),
]


def test_active_mart_path_resolves():
    p = active_mart_path(WAREHOUSE)
    assert p.exists()
    assert p.suffix == ".parquet"


@pytest.fixture(scope="module")
def wkcomp():
    return load_schedule_p(WAREHOUSE, lines=["workers_compensation"])


def test_mapping_and_validity(wkcomp):
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
    assert wkcomp.dev_lags == [12 * i for i in range(1, 11)]
    import ibis

    one = wkcomp.filter(ibis._.company_code == "26956")
    assert one.validate(strict=True) == []


def test_paid_triangle_ties_to_wide_mart():
    """Our pivot of the training mart must equal the published wide paid mart."""
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
    for ay, row in ours.iterrows():
        for dev_months, val in row.items():
            assert val == pytest.approx(ref.loc[ay.year, str(dev_months // 12)]), (
                f"mismatch at AY {ay.year} dev {dev_months}"
            )


def test_as_of_gives_upper_triangle():

    import pandas as pd

    t = load_schedule_p(WAREHOUSE, lines=["workers_compensation"], companies=["26956"])
    train = t.as_of(dt.date(1997, 12, 31)).select_fields("paid_loss")
    df = train.execute()
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    df["origin_period"] = pd.to_datetime(df["origin_period"])
    assert (df["eval_date"] <= pd.Timestamp("1997-12-31")).all()
    # 10 accident years 1988-1997 -> triangular cell count for the upper triangle
    ay = df["origin_period"].dt.year
    upper = df[(ay >= 1988) & (ay <= 1997)]
    assert len(upper) == 55
