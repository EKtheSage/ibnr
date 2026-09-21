"""mcl and mack against the companion study's R implementation on the Schedule
P mart.

This is the test the mcl entry exists to pass. The companion study replays four
classical reserving methods in R over 93 multi-line companies at 31 December
2007; ``tests/data/`` vendors the company reserves that replay produced, and
the two tests below refit the same companies through this package and compare.

Skips unless the pinned gold publish is cached. The 82 companies marked
``point_ok`` are the comparison set: on the other 11 the R chain ladder failed
on a line with a zero or negative paid cell, which this package's contracts
refuse by name, and those refusals are asserted rather than skipped.

Marker rationale (``mart``): the warehouse is not vendored, so this skips
cleanly on a clean checkout.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.data.schedule_p import active_mart_path, load_schedule_p

#: The publish the R replay read. Written out rather than resolved through
#: ``pinned_source``, which would reach the release before the availability
#: check below could turn a failure into a skip; the tag is already concrete,
#: so there is nothing for it to pin.
PUBLISH = "20260613_041006"
SOURCE = f"github://EKtheSage/cas-schedule-p-data-model@{PUBLISH}"
DATA = Path(__file__).parent / "data"
AS_OF = dt.date(2007, 12, 31)
ORIGINS = [dt.date(y, 1, 1) for y in range(1998, 2008)]
LINES = [
    "commercial_auto",
    "other_liability",
    "private_passenger_auto",
    "workers_compensation",
]


def _cached() -> bool:
    """True when the pinned publish is reachable; this may download it once.

    Broad ``except`` on purpose, as in ``tests/test_schedule_p.py``: offline,
    no access, a moved release - every failure has to become a clean skip
    rather than an error while the file is being imported.
    """
    try:
        return active_mart_path(SOURCE).exists()
    except Exception:
        return False


pytestmark = [
    pytest.mark.mart,
    pytest.mark.skipif(not _cached(), reason=f"Schedule P publish {PUBLISH} is not reachable"),
]


@pytest.fixture(scope="module")
def study():
    """The vendored R company table and the line-of-business pairs it covers."""
    companies = pd.read_csv(DATA / "tlrn_study_company_reserves.csv", dtype={"company_code": str})
    pairs = pd.read_csv(DATA / "tlrn_study_pairs.csv", dtype={"company_code": str})
    assert len(companies) == 93
    assert len(pairs) == 243
    assert int(companies["point_ok"].sum()) == 82
    return companies, pairs


@pytest.fixture(scope="module")
def mart(study):
    """One triangle over every company in the study, read from the mart once.

    ``company_name`` is dropped: it is a display-only segment column, and the
    multi-line contract asks every non-line segment column to be constant.
    """
    companies, _ = study
    tri = load_schedule_p(SOURCE, companies=list(companies["company_code"]), lines=LINES)
    tri = tri.filter(tri.expr.origin_period.isin(ORIGINS))
    return tri.with_expr(tri.expr.drop("company_name"))


def _company_triangle(mart, code: str, lines: list[str]):
    """One company on exactly the lines the study selected.

    Filtering on the company alone is not enough. The mart carries every line
    a company reported, and the study kept only those with a complete positive
    history over its first five diagonals - so a company selected on two lines
    still has four in the mart, and the two that were dropped are usually the
    ones with a zero paid cumulative. Fitting the wider set refuses the
    company by name rather than tying out.
    """
    return mart.filter(mart.expr.company_code == code, mart.expr.line_of_business.isin(lines))


def _latest_sum(entry) -> float:
    """Sum over lines and origins of the latest observed cumulative.

    The reserve the R replay reports is ultimate minus this, so it is what
    turns ``point()``'s ultimates into a comparable number.
    """
    c = entry.contract_
    cum, mask = c["cum"], c["obs_mask"]
    total = 0.0
    for k in range(c["n_lob"]):
        for w in range(c["n_w"]):
            devs = np.nonzero(mask[k, w])[0]
            total += float(cum[k, w, devs[-1]])
    return total


def _lines_of(pairs, code: str) -> list[str]:
    return sorted(pairs.loc[pairs["company_code"] == code, "line_of_business"])


def test_mcl_and_mack_reserves_tie_out_on_the_82_companies(study, mart):
    """Both point reserves reproduce the R replay to 1e-6 relative.

    ``mack`` is here as well as ``mcl`` because it is the control: the two
    implementations have to agree on the data and on the reserve definition
    before agreeing on the harder estimator means anything. A difference in
    which cells the triangle holds, or in whether the oldest accident year
    counts, would move both columns.
    """
    companies, pairs = study
    scored = companies[companies["point_ok"]]
    rows = []
    for code in scored["company_code"]:
        lines = _lines_of(pairs, code)
        tri = _company_triangle(mart, code, lines)
        fitted = gallery.fit("mcl", tri, loss_field="paid_loss", as_of=AS_OF)
        # the fit covers exactly the study's lines and its ten accident years
        assert fitted.contract_["lobs"] == lines
        assert fitted.contract_["n_w"] == 10 and fitted.contract_["n_d"] == 10
        mcl_reserve = float(fitted.point()["point"].iloc[-1]) - _latest_sum(fitted)
        mack_reserve = 0.0
        for line in lines:
            one_line = tri.filter(tri.expr.line_of_business == line)
            mack = gallery.fit("mack", one_line, loss_field="paid_loss", as_of=AS_OF)
            mack_reserve += float(mack.fit_.reserve.sum())
        rows.append({"company_code": code, "mcl": mcl_reserve, "mack": mack_reserve})
    got = pd.DataFrame(rows).merge(scored, on="company_code")
    assert len(got) == 82
    np.testing.assert_allclose(got["mack"], got["r_chain_ladder_reserve"], rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(got["mcl"], got["r_mcl_reserve"], rtol=1e-6, atol=1e-2)


def test_mcl_differs_from_the_chain_ladder_on_most_of_the_82(study, mart):
    """The tie-out above would also pass if mcl quietly WERE the chain ladder.

    It is not: the cross-line coefficients move the company reserve on the
    great majority of these companies, and by more than a rounding error. This
    is the assertion that makes the other test about the estimator rather than
    about the plumbing, and it reads the two vendored R columns only, so it
    costs nothing.
    """
    companies, _ = study
    scored = companies[companies["point_ok"]]
    relative = (scored["r_mcl_reserve"] - scored["r_chain_ladder_reserve"]).abs() / scored[
        "r_chain_ladder_reserve"
    ].abs()
    assert (relative > 1e-3).sum() >= 70


def test_the_eleven_excluded_companies_are_refused_by_name(study, mart):
    """Every company the R chain ladder could not score is refused here, and
    for the reason the R failure had: a paid cumulative at or below zero on a
    cell some development transition divides by."""
    companies, pairs = study
    excluded = companies.loc[~companies["point_ok"], "company_code"]
    assert len(excluded) == 11
    for code in excluded:
        tri = _company_triangle(mart, code, _lines_of(pairs, code))
        with pytest.raises(ValueError, match="non-positive cumulative"):
            gallery.fit("mcl", tri, loss_field="paid_loss", as_of=AS_OF)


def test_every_vendored_company_and_line_is_in_the_mart(study, mart):
    """The vendored tables name cohorts this publish actually carries.

    A company code the mart spells differently, or a line renamed since the
    replay, would leave the tie-out silently scoring fewer companies than it
    claims - or fitting a company on three lines where the study used four,
    which still produces a plausible reserve.
    """
    companies, pairs = study
    frame = mart.select_fields("paid_loss").execute()
    present = set(zip(frame["company_code"], frame["line_of_business"], strict=True))
    vendored = set(zip(pairs["company_code"], pairs["line_of_business"], strict=True))
    assert not vendored - present
    assert set(companies["company_code"]) == set(pairs["company_code"])
