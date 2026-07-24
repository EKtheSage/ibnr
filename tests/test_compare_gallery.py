"""scripts/compare_gallery.py::point_context - the chain_ladder skill benchmark.

Why a test for a script: `point_context` produces the `chain_ladder` rows every
other model in the study is measured against, and it used to compute the
volume-weighted age-to-age factors inline. It now calls `kernels.mack.fit_mack`
instead. `test_matches_legacy_inline_chain_ladder` keeps the retired inline
formula as an executable reference so the de-duplication stays a
de-duplication: if the kernel and the old benchmark ever disagree on a clean
run-off cohort, that is a regression in the published leaderboard, not a
refactor.

Two things beyond the agreement are pinned here: the outcome query must keep
excluding origins that post-date the training slice (the 2.4x gotcha in
CLAUDE.md), and a cohort `fit_mack` genuinely refuses - one with a negative
cumulative - must land in the results as a failure row rather than aborting the
study, with its anchor and premium still recorded for every other model.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from compare_gallery import point_context  # noqa: E402

AS_OF = "2013-12-31"

#: 4 origins x 4 yearly devs, a full square (both halves). Sliced at AS_OF it
#: becomes the run-off staircase [3, 2, 1, 0]; the dev-48 column is what the
#: outcome is read from. Per-cohort scaling below keeps every cell distinct.
SQUARE = np.array(
    [
        [100.0, 150.0, 165.0, 170.0],
        [120.0, 180.0, 200.0, 206.0],
        [110.0, 165.0, 181.0, 187.0],
        [130.0, 195.0, 214.0, 220.0],
    ]
)

#: cohort -> (scale, premium per origin). Two companies x two lines, so the
#: company "ALL" rows are real sums rather than a single line renamed.
COHORTS = {
    ("0001", "auto"): (1.0, 400.0),
    ("0001", "gl"): (2.5, 900.0),
    ("0002", "auto"): (0.7, 300.0),
    ("0002", "gl"): (1.9, 750.0),
}
SCORED = {"0001": ["auto", "gl"], "0002": ["auto", "gl"]}


def _rows(code: str, line: str, cum: np.ndarray, premium: float, start_year: int = 2010):
    """Long-format cells for one cohort: yearly grain, origin i = Jan 1 of
    ``start_year + i``, dev step j = dev_lag 12*(j+1), eval = Dec 31 of
    ``start_year + i + j``. Premium rides along on every loss cell so
    ``latest_diagonal()`` recovers the booked value (as the mart does)."""
    out = []
    n_w, n_d = cum.shape
    for i in range(n_w):
        for j in range(n_d):
            if np.isnan(cum[i, j]):
                continue
            key = (
                code,
                line,
                dt.date(start_year + i, 1, 1),
                12 * (j + 1),
                dt.date(start_year + i + j, 12, 31),
            )
            out.append((*key, "paid_loss", float(cum[i, j])))
            out.append((*key, "earned_premium", premium))
    return out


def _triangle(backend_name: str, overrides: dict | None = None, post_study: bool = False):
    """The study fixture: every cohort in COHORTS as a full square.

    ``overrides`` replaces a cohort's matrix outright (used to plant the
    non-positive cell). ``post_study`` adds a 2014 accident year, which is
    invisible at AS_OF but sits in the full triangle at dev 48 - exactly the
    shape that once inflated outcomes 2.4x.
    """
    rows = []
    for (code, line), (scale, premium) in COHORTS.items():
        cum = (overrides or {}).get((code, line), SQUARE * scale)
        rows += _rows(code, line, cum, premium)
        if post_study:
            # one extra origin, whose earliest eval_date (2014-12-31) already
            # post-dates AS_OF, so it is absent from the training slice
            rows += _rows(code, line, cum[:1] * 3.0, premium, start_year=2014)
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


def _args(loss_field: str = "paid_loss"):
    return SimpleNamespace(as_of=AS_OF, loss_field=loss_field)


def legacy_point_context(tri_all, scored, args):
    """The inline implementation `point_context` replaced, verbatim.

    Kept here and nowhere else: it is the reference the refactor is checked
    against, not code anyone should call. Volume-weighted age-to-age factors
    over overlapping origins, each origin's diagonal cell developed to the last
    observed dev column.
    """
    train = tri_all.as_of(args.as_of)
    cum = train.select_fields(args.loss_field).execute()
    prem = train.select_fields("earned_premium").latest_diagonal().execute()
    full = tri_all.select_fields(args.loss_field).execute()
    n_d_months = int(full["dev_lag"].max())

    anchors, premiums, rows = {}, {}, []
    for code, lines_c in scored.items():
        cl_total, anchor_total, outcome_total, prem_total = 0.0, 0.0, 0.0, 0.0
        for line in lines_c:
            sub = cum[(cum["company_code"] == code) & (cum["line_of_business"] == line)]
            grid = sub.pivot_table(
                index="origin_period", columns="dev_lag", values="value"
            ).sort_index()
            devs = sorted(grid.columns)
            factors = {}
            for a, b in zip(devs[:-1], devs[1:], strict=True):
                both = grid[[a, b]].dropna()
                factors[a] = float(both[b].sum() / both[a].sum()) if len(both) else 1.0
            est = 0.0
            anchor = 0.0
            for _, r in grid.iterrows():
                obs = r.dropna()
                latest_dev, latest = obs.index[-1], float(obs.iloc[-1])
                anchor += latest
                for d in devs[devs.index(latest_dev) : -1]:
                    latest *= factors[d]
                est += latest
            f_sub = full[
                (full["company_code"] == code)
                & (full["line_of_business"] == line)
                & (full["dev_lag"] == n_d_months)
                & full["origin_period"].isin(grid.index)
            ]
            outcome = float(f_sub["value"].sum())
            p_sub = prem[(prem["company_code"] == code) & (prem["line_of_business"] == line)]
            line_prem = float(p_sub["value"].sum())
            anchors[(code, line)] = anchor
            premiums[(code, line)] = line_prem
            rows.append(
                {
                    "model": "chain_ladder",
                    "line": line,
                    "company_code": code,
                    "estimate": est,
                    "outcome": outcome,
                }
            )
            cl_total += est
            anchor_total += anchor
            outcome_total += outcome
            prem_total += line_prem
        anchors[(code, "ALL")] = anchor_total
        premiums[(code, "ALL")] = prem_total
        rows.append(
            {
                "model": "chain_ladder",
                "line": "ALL",
                "company_code": code,
                "estimate": cl_total,
                "outcome": outcome_total,
            }
        )
    return rows, anchors, premiums


def _frame(rows):
    return (
        pd.DataFrame(rows)
        .sort_values(["company_code", "line"])
        .reset_index(drop=True)[["model", "company_code", "line", "estimate", "outcome"]]
    )


@pytest.mark.parametrize("post_study", [False, True])
@pytest.mark.parametrize("zero_cell", [False, True])
def test_matches_legacy_inline_chain_ladder(backend_name, post_study, zero_cell):
    """fit_mack reproduces the retired inline benchmark cell for cell.

    Estimates, outcomes, anchors and premiums must all agree - on the real
    1997-12-31 paid_loss panel the two agree to 3e-16 on all 152 cells, so the
    tolerance here is float noise, not slack.

    ``zero_cell`` plants an accident year at zero paid at 12 months, the shape
    of the two real Schedule P cohorts (29440 and 42439, other_liability). It
    used to be the one place the kernel and the inline formula diverged; since
    the factor stopped demanding strictly positive cumulatives it is just
    another cell, and pinning the agreement is what keeps it that way.
    """
    overrides = None
    if zero_cell:
        holed = SQUARE.copy()
        holed[2, 0] = 0.0  # 2012 origin: above the diagonal, in step 0's pair set
        overrides = {("0001", "gl"): holed}
    tri = _triangle(backend_name, overrides=overrides, post_study=post_study)
    rows, anchors, premiums = point_context(tri, SCORED, _args())
    want_rows, want_anchors, want_premiums = legacy_point_context(tri, SCORED, _args())

    got, want = _frame(rows), _frame(want_rows)
    pd.testing.assert_frame_equal(
        got[["model", "company_code", "line"]], want[["model", "company_code", "line"]]
    )
    np.testing.assert_allclose(got["estimate"], want["estimate"], rtol=1e-12)
    np.testing.assert_allclose(got["outcome"], want["outcome"], rtol=1e-12)
    assert anchors.keys() == want_anchors.keys()
    for key, value in want_anchors.items():
        assert anchors[key] == pytest.approx(value, rel=1e-12)
    for key, value in want_premiums.items():
        assert premiums[key] == pytest.approx(value, rel=1e-12)
    # the benchmark is a real projection, not a copy of the anchor
    assert all(r["estimate"] > anchors[(r["company_code"], r["line"])] for r in rows)


def test_outcome_excludes_post_study_origins(backend_name):
    """The load-bearing guard: a 2014 accident year sits in the full triangle
    but not in the training slice, so it must not enter the outcome.

    Its dev-48 cell is 3x the 2010 origin's, so forgetting the restriction is
    a large, obvious error rather than a rounding one - which is exactly how it
    slipped through the first time (CLAUDE.md's 2.4x gotcha)."""
    clean = point_context(_triangle(backend_name), SCORED, _args())[0]
    with_post = point_context(_triangle(backend_name, post_study=True), SCORED, _args())[0]
    assert _frame(clean)["outcome"].tolist() == _frame(with_post)["outcome"].tolist()


def test_records_mack_rejection_as_a_failure_row(backend_name):
    """A cohort Mack's model rejects becomes a failure row; the study goes on.

    The rejections that remain are the ones no reading of Mack's model survives
    - here a NEGATIVE cumulative, which would drive sigma_j^2 itself negative
    and hand back a negative msep with a NaN standard error. (Schedule P
    incurred net of bulk can produce these, which is why the study needs a
    recorded failure rather than a crash.) A cohort with a zero cell is NOT one
    of them any more; that case is pinned as an agreement in
    `test_matches_legacy_inline_chain_ladder[zero_cell=True]`.
    """
    holed = SQUARE.copy()
    holed[2, 0] = -40.0  # 2012 origin, negative paid at 12 months
    tri = _triangle(backend_name, overrides={("0001", "gl"): holed})
    rows, anchors, premiums = point_context(tri, SCORED, _args())
    by_key = {(r["company_code"], r["line"]): r for r in rows}

    bad = by_key[("0001", "gl")]
    assert np.isnan(bad["estimate"])
    assert "negative cumulative" in bad["error"]
    # a partial sum is not a benchmark: the company total is withheld too
    assert np.isnan(by_key[("0001", "ALL")]["estimate"])
    assert "gl" in by_key[("0001", "ALL")]["error"]

    # everything the OTHER models need for this cell survives the rejection -
    # anchor/premium/outcome are read off the triangle, not off the MackFit
    assert anchors[("0001", "gl")] == pytest.approx(holed[[0, 1, 2, 3], [3, 2, 1, 0]].sum())
    assert premiums[("0001", "gl")] == pytest.approx(900.0 * 4)
    assert bad["outcome"] == pytest.approx(holed[:, 3].sum())
    assert not np.isnan(anchors[("0001", "ALL")])

    # unaffected cohorts are untouched, and the failure did not abort the run
    assert not np.isnan(by_key[("0002", "ALL")]["estimate"])
    assert by_key[("0001", "auto")]["estimate"] == pytest.approx(
        legacy_point_context(tri, SCORED, _args())[0][0]["estimate"]
    )
