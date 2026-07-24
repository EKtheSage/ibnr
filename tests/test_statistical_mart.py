"""End-to-end smokes for the new gallery entries against the Schedule P gold
mart. Auto-skips when the mart is absent. No cmdstan involved, so these run
in the fast suite (mart-gated only).

What this file protects: the synthetic-data tests in ``test_sur.py`` /
``test_copula_glm.py`` / ``test_nn_transformer.py`` simulate from each model's
own generative assumptions, so they cannot catch failures that only real data
produces - ragged triangles, wildly unequal exposure, near-degenerate
covariances, companies whose paid increments go negative. These smokes run the
full public path (``gallery.fit`` -> ``predict`` -> ``realized_ultimates`` ->
``summary``) on genuine CAS Schedule P triangles and assert the output is
sane rather than exact.

Marker rationale (``mart``): the warehouse is not vendored, so these skip
cleanly on a clean checkout - the package must be usable and testable without
it (CLAUDE.md). They are NOT ``slow``: no Stan compile, and the NN config below
is deliberately tiny.

Assertions are intentionally weak (finite, positive, plausible CV, percentile
in range). Accuracy is not tested here - that is the job of the 200-company
retrospectives in ``scripts/`` - this only guarantees the entries do not fall
over on real inputs.
"""

from __future__ import annotations

import duckdb
import ibis
import numpy as np
import pytest

from ibnr.data.schedule_p import active_mart_path, load_schedule_p

from .test_schedule_p import MART_AVAILABLE, WAREHOUSE

pytestmark = [
    pytest.mark.mart,
    pytest.mark.skipif(
        not MART_AVAILABLE,
        reason="CAS Schedule P gold mart not reachable",
    ),
]

# Two lines, because these are multi-line entries: SUR/copula estimate a
# cross-line covariance, which needs at least two.
LINES = ["commercial_auto", "workers_compensation"]


def candidate_companies(n: int = 5) -> list[str]:
    """Companies with complete squares in both LINES and strictly positive
    paid increments on the training window (copula-compatible).

    The screen exists because the copula entry's lognormal marginals have no
    density at non-positive increments, so a company with a single negative
    paid increment would make the smoke fail for a documented, uninteresting
    reason. Selecting the same cohort for every entry also keeps the models
    comparable (cohort-data parity).

    Magic numbers: 110 = the 1988-1997 upper triangle (55 cells) on each of two
    lines; 200 = the full 10x10 square on each of two lines, i.e. the realized
    ultimates needed for scoring. Ordering by volume prefers large, stable
    filers whose triangles are least likely to be numerically pathological.

    The ``accident_year between 1988 and 1997`` filter is mandatory, not
    cosmetic: the mart carries accident years through 2007, and counting cells
    or ultimates over the whole slice would silently include origins that are
    not in the as-of training window (CLAUDE.md gotcha).
    """
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
    """Both lines loaded once per module; per-test company filtering is lazy."""
    return load_schedule_p(WAREHOUSE, lines=LINES)


@pytest.fixture(scope="module")
def company(two_lines):
    """The single largest copula-compatible filer, shared by the SUR and copula
    smokes so the two are fit on identical data."""
    codes = candidate_companies(1)
    if not codes:
        pytest.skip("no copula-compatible company found in the mart")
    return codes[0]


def _check_total_row(pred, realized):
    """Sanity-check the grand-total row of the scored summary.

    Bounds are loose on purpose - this is a smoke, not a calibration test.
    CV < 5 (500%) merely rules out a blown-up predictive distribution, and the
    percentile check confirms the PIT machinery placed the realized ultimate
    somewhere on [0, 100] rather than emitting NaN.
    """
    table = pred.summary(observed=realized)
    total = table.iloc[-1]  # target layout puts the grand total last
    assert np.isfinite(total["estimate"]) and total["estimate"] > 0
    assert 0 < total["cv"] < 5
    if np.isfinite(total["outcome"]):
        assert 0 <= total["percentile"] <= 100
    return table


def test_sur_mart_smoke(two_lines, company):
    """SUR survives a real two-line company end to end via the public
    ``gallery.fit`` API and emits one subtotal row per line."""
    from ibnr import gallery

    tri = two_lines.filter(ibis._.company_code == company)
    entry = gallery.fit("sur", tri, loss_field="paid_loss", as_of="1997-12-31")
    pred = entry.predict(n_draws=2000, seed=0)
    realized = entry.realized_ultimates(tri)
    table = _check_total_row(pred, realized)
    assert table["label"].str.endswith("/total").sum() == len(LINES)


def test_copula_mart_smoke(two_lines, company):
    """Same for the copula entry, plus the estimated cross-line correlation is
    a valid correlation on real (not simulated-from-the-model) residuals - the
    case most likely to produce a degenerate or out-of-range estimate."""
    from ibnr import gallery

    tri = two_lines.filter(ibis._.company_code == company)
    entry = gallery.fit("copula_glm", tri, loss_field="paid_loss", as_of="1997-12-31")
    # n_boot kept small: the parametric bootstrap refits per replicate and this
    # is a smoke, not an interval-width measurement.
    pred = entry.predict(n_draws=2000, seed=0, n_boot=50)
    realized = entry.realized_ultimates(tri)
    _check_total_row(pred, realized)
    assert -1.0 <= entry.corr_[0, 1] <= 1.0


def test_nn_transformer_mart_smoke(two_lines):
    """The transformer trains globally across MANY company x LOB triangles and
    then predicts one segment. Fitting on five companies (ten triangles) rather
    than one is the point: Schedule P triangles are ~55 cells, so overfitting is
    the central NN risk and the entry is only ever validated pooled.

    ``importorskip`` keeps torch optional - ``ibnr.gallery`` must import and
    register NN entries without the ``[nn]`` extra installed.
    """
    torch = pytest.importorskip("torch")  # noqa: F841
    import ibis as _ibis

    from ibnr import gallery
    from ibnr.gallery.nn.transformer.config import TransformerConfig

    codes = candidate_companies(5)
    if not codes:
        pytest.skip("no candidate companies found in the mart")
    tri = two_lines.filter(_ibis._.company_code.isin(codes))
    # Minimal architecture / epochs / ensemble: this asserts the wiring runs on
    # real data, not that the model is well trained. Production configs live in
    # the entry defaults and the comparison scripts.
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
    # global fit -> per-segment predict: one of the five companies is scored
    seg = {"company_code": codes[0], "line_of_business": LINES[0]}
    pred = entry.predict(segment=seg, seed=0)
    realized = entry.realized_ultimates(tri, segment=seg)
    assert np.isfinite(pred.samples).all()
    _check_total_row(pred, realized)
