"""End-to-end smokes for the new gallery entries against the Schedule P gold
mart. Auto-skips when the mart is absent. No cmdstan involved, so these run
in the fast suite (mart-gated only)."""

from __future__ import annotations

import duckdb
import ibis
import numpy as np
import pytest

from ibnr.data.schedule_p import active_mart_path, load_schedule_p

from .test_schedule_p import WAREHOUSE

pytestmark = [
    pytest.mark.mart,
    pytest.mark.skipif(
        not (WAREHOUSE / "_active_manifest.json").exists(),
        reason="CAS Schedule P gold mart not available",
    ),
]

LINES = ["commercial_auto", "workers_compensation"]


def candidate_companies(n: int = 5) -> list[str]:
    """Companies with complete squares in both LINES and strictly positive
    paid increments on the training window (copula-compatible)."""
    mart = active_mart_path(WAREHOUSE)
    q = f"""
    with cells as (
        select company_code, line_of_business, accident_year, development_age,
               statement_year, cum_paid_loss, earned_prem_net,
               cum_paid_loss - lag(cum_paid_loss) over (
                   partition by company_code, line_of_business, accident_year
                   order by development_age) as incr
        from read_parquet('{mart.as_posix()}')
        where line_of_business in ('{LINES[0]}', '{LINES[1]}')
          and accident_year between 1988 and 1997
    ),
    train as (
        select company_code,
               count(*) as n_train,
               min(coalesce(incr, cum_paid_loss)) as min_incr,
               min(earned_prem_net) as min_prem
        from cells where statement_year <= 1997 group by 1
    ),
    full_square as (
        select company_code, count(*) as n_all, sum(cum_paid_loss) as volume
        from cells group by 1
    )
    select t.company_code
    from train t join full_square f using (company_code)
    where t.n_train = 110 and f.n_all = 200
      and t.min_incr > 0 and t.min_prem > 0
    order by f.volume desc
    limit {n}
    """
    return duckdb.sql(q).df()["company_code"].tolist()


@pytest.fixture(scope="module")
def two_lines():
    return load_schedule_p(WAREHOUSE, lines=LINES)


@pytest.fixture(scope="module")
def company(two_lines):
    codes = candidate_companies(1)
    if not codes:
        pytest.skip("no copula-compatible company found in the mart")
    return codes[0]


def _check_total_row(pred, realized):
    table = pred.summary(observed=realized)
    total = table.iloc[-1]
    assert np.isfinite(total["estimate"]) and total["estimate"] > 0
    assert 0 < total["cv"] < 5
    if np.isfinite(total["outcome"]):
        assert 0 <= total["percentile"] <= 100
    return table


def test_sur_mart_smoke(two_lines, company):
    from ibnr import gallery

    tri = two_lines.filter(ibis._.company_code == company)
    entry = gallery.fit("sur", tri, loss_field="paid_loss", as_of="1997-12-31")
    pred = entry.predict(n_draws=2000, seed=0)
    realized = entry.realized_ultimates(tri)
    table = _check_total_row(pred, realized)
    assert table["label"].str.endswith("/total").sum() == len(LINES)


def test_copula_mart_smoke(two_lines, company):
    from ibnr import gallery

    tri = two_lines.filter(ibis._.company_code == company)
    entry = gallery.fit("copula_glm", tri, loss_field="paid_loss", as_of="1997-12-31")
    pred = entry.predict(n_draws=2000, seed=0, n_boot=50)
    realized = entry.realized_ultimates(tri)
    _check_total_row(pred, realized)
    assert -1.0 <= entry.corr_[0, 1] <= 1.0


def test_nn_transformer_mart_smoke(two_lines):
    torch = pytest.importorskip("torch")  # noqa: F841
    import ibis as _ibis

    from ibnr import gallery
    from ibnr.gallery.nn.transformer.config import TransformerConfig

    codes = candidate_companies(5)
    if not codes:
        pytest.skip("no candidate companies found in the mart")
    tri = two_lines.filter(_ibis._.company_code.isin(codes))
    cfg = TransformerConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        n_components=2,
        lob_embedding_dim=4,
        batch_size=8,
        max_epochs=10,
        patience=5,
        ensemble_size=2,
        n_draws=100,
    )
    entry = gallery.fit(
        "nn_transformer",
        tri,
        loss_field="paid_loss",
        as_of="1997-12-31",
        config=cfg,
        seed=0,
    )
    seg = {"company_code": codes[0], "line_of_business": LINES[0]}
    pred = entry.predict(segment=seg, seed=0)
    realized = entry.realized_ultimates(tri, segment=seg)
    assert np.isfinite(pred.samples).all()
    _check_total_row(pred, realized)
