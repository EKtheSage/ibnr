"""The tlrn robustness study's runner, for a Colab GPU (or TPU) runtime.

The question is whether the companion study's accident-year model is effective against
the classical benchmarks, or whether one lucky run of it looks that way. This script
trains it many times: for each of two cohorts, each of two rows and each of ``--draws``
independent fit seeds, it calls ``gallery.fit("tlrn", ..., backend="jax")`` in this
process and saves the fit as soon as it finishes. ``scripts/tlrn_study_eval.py`` reads
what is saved, on a laptop without jax. The questions, the rules and the verdict are
fixed in ``knowledge/decisions/tlrn-robustness-study.md``; the Colab steps are in
``docs/tlrn-study.md``.

* **Cohort A** is notebook 04's: 93 companies, 243 company-line pairs, 82 of them scored.
  It is built by ``tlrn_colab.build_triangle`` from the cas-schedule-p package, so it is
  the very triangle the earlier Colab fits used.
* **Cohort B** is the companion study's own: its selection rule with the cutoff at the
  last known diagonal, which gives 84 companies and 212 pairs, all scored. By default it
  is built from the cas-schedule-p mart, whose cells for these lines and years are
  identical to the four cached CAS files'; ``--cas-dir`` builds it from those files
  instead, and the counts and the cohort hash are asserted either way.
* **Rows** are ``tlrn_8_ay`` (paid only) and ``tlrn_13_ay`` (paid, incurred and case
  reserve), exactly as ``tlrn_colab.py --config accident_year`` configures them.
* **Draw** ``k`` is one fit with seed ``11 + 100003 k``.

Every fit is saved with three files that share a stem (cohort, row, draw, ibnr version):
the settings record ``.json`` (written last, and the only thing that marks a fit as
finished), the company reserves ``_reserves.csv`` that the evaluator reads, and the fitted
entry ``.pkl`` (skipped by ``--no-entries``). A fit is reused only when its record matches
this run's settings exactly; a record that differs stops the run before any training, and
leaves the saved fit alone. A disconnect loses at most the fit that was running.

Usage, in a Colab cell (not ``!python``: a TPU opens in one process only)::

    import tlrn_study_colab

    tlrn_study_colab.main(["--out", OUT, "--smoke"])          # a few epochs, to check the setup
    tlrn_study_colab.main(["--out", OUT])                     # the full study, 40 fits
    tlrn_study_colab.main(["--out", OUT, "--cohort", "B", "--draws", "3"])   # a piece

**The second study** (``--study factor_head``) runs the companion study's newest model, the
factor head, in the same way: rows ``tlrn_13_fh`` and ``tlrn_13_fh_factors`` (the same model
with the network switched off), cohorts A, B and a new cohort C (accident years 1988 to 1997,
valued at 1997-12-31, outcomes from the later calendar years the mart carries), 10 draws,
60 fits. Its rules are in ``knowledge/decisions/tlrn-factor-head-study.md``. Its records say
``"study": 2`` and ``"study_name": "factor_head"``, so they can never be mistaken for the first
study's, and a fit of it writes, besides the three files above, three tables the evaluator
reads (valuation dates 6, 7, 8 and 10 are the model's own: 10 is the last known diagonal):

* ``_valuations.csv``: one row per (valuation, company): ``valuation``, ``company_code``,
  ``n_cells`` (the cells scored: diagonals after the valuation date, development lag at most
  the valuation date, and at valuation 10 every cell past the last diagonal), ``predicted``
  (the mean of the 20 blended members), ``predicted_net`` (the mean of the 20 raw networks,
  before the blend) and ``actual`` (the known total, empty at valuation 10);
* ``_backtest.csv``: the fitted entry's own ``backtest_`` table, unchanged, with its
  company columns: ``unit``, ``cutoff``, ``horizon``, ``predicted``, ``actual``, ``size``,
  ``residual_scale``, ``standardised_error``, ``company_code``;
* ``_members.csv``: one row per (valuation, member): ``best_epoch``, ``epochs_run``, the two
  validation errors, the blend weight ``alpha`` and, from the third study's table on,
  ``non_finite_loss_epochs`` (epochs of the first run whose training loss was not finite).

The valuation tables need each retrained valuation date's members, which the fitted entry does
not keep (only the final fit's). The runner takes them by wrapping the entry's own
``_finish_protocol`` while ``fit`` runs (:func:`capture_protocols`), so no library code changes;
:func:`valuation_tables` checks them against the entry before anything is saved.

**The third study** (``--study companion_final``) runs the companion's final model (premium loss
ratio head, attention across accident years, company batches, a refit on everything known, alpha
fitted on one diagonal) on cohorts B and C. Its rows are ``tlrn_cf_refit`` (one run that gives the
refit forecasts and, from the same members before the refit, the no-refit ones) and
``tlrn_cf_factors`` (the network switched off). Each valuation date 6, 7, 8 and 10 is a separate
protocol on its OWN company set, ``select_pairs(cutoff=v)`` (:func:`valuation_pairs`), so a fit is
not one ``gallery.fit`` call but :func:`fit_valuations`; date ``v`` uses seed ``seed + 20000 v``
and member ``m`` of it ``seed + 20000 v + 1000 m``. It saves ``_valuations.csv`` (as the second
study's, plus ``predicted_selected`` and ``predicted_net_selected``, the forecasts of the networks
as they stood at their chosen epoch) and ``_members.csv``. Its records say ``"study": 3``. Rules:
``knowledge/decisions/tlrn-study3-preregistration.md``.

**The fourth study** (``--study companion_revised``) runs the companion's revised model (the third
study's without the per-count embedding, ``maturity_embedding=False``) on cohorts B and C in the
same per-date way, with the third study's seeds, so draw ``k`` is paired with the third study's
draw ``k``. Its six rows, in the pre-registered order, are ``tlrn_cr_refit`` (the main row),
``tlrn_cr_factors`` (the network off), the three single-draw variant rows ``tlrn_cr_noline``,
``tlrn_cr_noay`` and ``tlrn_cr_paid`` (cohort B, draw 0 only) and the rerun control
``tlrn_cf_rerun`` (the third study's model fitted again, cohort B, draw 0 only). A single-draw
row asked for with more than one draw, or on cohort C, is refused by name. Its records say
``"study": 4`` and carry ``paid_only`` per row; its run records add the versions of pandas,
numpy and duckdb (the third study's data hashes differed between Colab and the laptop for a
reason nobody could check). The pre-registered stops are in the runner: after the main row's
draw 0 on cohort C takes over 25 minutes, or when the saved fits add up to 8 GPU hours, the
run stops and asks (:func:`check_budget`, :func:`check_first_fit`); a fit with a non-finite
forecast or a member with a non-finite training loss is rerun once with the same seed and, if it
recurs, reported as failed and left out;
a single-draw row that fails is reported as failed and delays nothing. The network-off row
may be taken from the third study's fits (``--reuse-factors-from``), which the pre-registration
allows only if a small identity check passes (see its amendment). Rules:
``knowledge/decisions/tlrn-study4-preregistration.md``.

Nothing here has been run on a GPU: the real results do not exist yet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import platform
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tlrn_colab

#: the ibnr release the CURRENT study (the third) is run with on Colab. The install cells of its
#: Colab kit and its how-to pin exactly this version, and a test holds them equal to it. The
#: earlier studies were run with, and their how-to documents still say, 0.7.4: each ``Study``
#: carries its own pin (``Study.pinned_ibnr``), so moving this one never rewrites their history
PINNED_IBNR = "0.7.6"
#: the release the FOURTH study is run with on Colab (its notebook installs exactly this and
#: fetches the runner scripts at the tag ``v0.7.7``). The third study keeps ``PINNED_IBNR``:
#: its notebook, how-to and tests are history. The package in this checkout is not bumped
#: (the release is a separate step), so no test holds this equal to ``ibnr.__version__``
COMPANION_REVISED_PINNED_IBNR = "0.7.7"
#: the release the first two studies were run with
EARLIER_PINNED_IBNR = "0.7.4"
#: bump when the study's definition changes, so old fits are never reused under a new one
STUDY_VERSION = 1
COHORTS = ("A", "B")
ROWS = ("tlrn_8_ay", "tlrn_13_ay")
#: the second study (the factor head). Its records carry study 2 and the name, so the first
#: study's records, which carry neither the name nor a 2, are never read as its own
FACTOR_HEAD_VERSION = 2
FACTOR_HEAD_COHORTS = ("A", "B", "C")
FACTOR_HEAD_ROWS = ("tlrn_13_fh", "tlrn_13_fh_factors")
#: the third study (the companion's final model, refit, per-date company sets)
COMPANION_FINAL_VERSION = 3
COMPANION_FINAL_COHORTS = ("B", "C")
COMPANION_FINAL_ROWS = ("tlrn_cf_refit", "tlrn_cf_factors")
COMPANION_FINAL_TABLES = ("valuations", "members")
#: the fourth study (the companion's revised model: no per-count embedding)
COMPANION_REVISED_VERSION = 4
COMPANION_REVISED_COHORTS = ("B", "C")
COMPANION_REVISED_ROWS = (
    "tlrn_cr_refit",
    "tlrn_cr_factors",
    "tlrn_cr_noline",
    "tlrn_cr_noay",
    "tlrn_cr_paid",
    "tlrn_cf_rerun",
)
COMPANION_REVISED_TABLES = COMPANION_FINAL_TABLES
#: the main row and the network-off row of the fourth study
REVISED_MAIN_ROW = "tlrn_cr_refit"
REVISED_FACTORS_ROW = "tlrn_cr_factors"
#: the pre-registered order of the cohorts: C4's draw 0 is the pipeline check, so C comes first
COMPANION_REVISED_FIT_ORDER = ("C", "B")
#: rows that are fitted once, on cohort B, at draw 0 (descriptive, no verdict rule uses them)
SINGLE_DRAW_ROWS = ("tlrn_cr_noline", "tlrn_cr_noay", "tlrn_cr_paid", "tlrn_cf_rerun")
SINGLE_DRAW_COHORTS = ("B",)
#: the stops that are pre-registered for the fourth study
STOP_FIRST_FIT_SECONDS = 25 * 60
BUDGET_HOURS = 8.0
#: the members table's count of a member's epochs whose training loss was not finite
NON_FINITE_LOSS = "non_finite_loss_epochs"
#: the parameters per network the pre-registration expects, by whether the per-count table is in
EXPECTED_PARAMETERS = {"revised": 18_153, "final": 18_537}
#: pairs per valuation date the pre-registration expects, by cohort
EXPECTED_PAIRS_BY_DATE = {
    "B": {6: 230, 7: 226, 8: 219, 10: 212},
    "C": {6: 237, 7: 236, 8: 232, 10: 232},
}
#: the valuation dates the third study trains and scores; the last is the last known diagonal
VALUATIONS = (6, 7, 8, 10)
#: every cohort a runner can load (the first study uses A and B only)
ALL_COHORTS = ("A", "B", "C")
DEFAULT_DRAWS = 10
DEFAULT_MEMBERS = 20
#: smoke runs keep the ensemble this small unless --members says otherwise
SMOKE_MEMBERS = 2
#: draw k trains with seed SEED_FIT + SEED_STRIDE * k. Members are seeded seed + 1000 m
#: and the retrained valuation dates seed + DATE_SEED_STRIDE v (study 3), all below 100,000
#: apart, so no two draws share a stream.
SEED_STRIDE = 100_003
#: the third study's valuation date ``v`` seeds its protocol ``seed + DATE_SEED_STRIDE * v``
#: and its member ``m`` ``+ 1000 m``. With 20 members a date spans 19,000, so the stride of
#: 20,000 keeps dates 6, 7, 8 and 10 on disjoint streams (a stride of 10,000 let member
#: ``m + 10`` of date ``v`` repeat member ``m`` of date ``v + 1``).
DATE_SEED_STRIDE = 20_000

#: the four cached CAS Schedule P files and the line each one is
CAS_FILES = {
    "ppauto": "private_passenger_auto",
    "wkcomp": "workers_compensation",
    "comauto": "commercial_auto",
    "othliab": "other_liability",
}
CAS_COLUMNS = [
    "GRCODE",
    "AccidentYear",
    "DevelopmentYear",
    "DevelopmentLag",
    "IncurredLosses",
    "CumPaidLoss",
    "BulkLoss",
    "EarnedPremNet",
]
CELL_COLUMNS = [
    "company_code",
    "line_of_business",
    "statement_year",
    "accident_year",
    "dev_age",
    "paid",
    "incurred",
    "bulk",
    "case",
    "premium",
]
KEY = ["company_code", "line_of_business"]

#: the counts each cohort's selection must reproduce: input pairs, complete squares,
#: pairs with usable known cells, pairs of companies with at least two such lines
EXPECTED = {
    "A": {"audit": [668, 414, 330, 243], "companies": 93, "board": 82},
    "B": {"audit": [668, 414, 297, 212], "companies": 84, "board": 84},
    # the Meyers window, accident years 1988 to 1997 (the second study's out-of-sample cohort)
    "C": {"audit": [675, 405, 322, 232], "companies": 83, "board": 83},
}
COHORT_HASH = {
    "A": tlrn_colab.COHORT_HASH,
    "B": "de51bae56cca563c",
    "C": "4a613cc4cb5f0f94",
}
#: the accident years of each cohort's triangle, first and last
YEARS = {"A": (1998, 2007), "B": (1998, 2007), "C": (1988, 1997)}
#: the valuation date each cohort's fits train on (the last known diagonal)
AS_OF = {
    "A": tlrn_colab.AS_OF,
    "B": tlrn_colab.AS_OF,
    "C": dt.date(1997, 12, 31),
}
BOARD_HASH_A = "b0654c51cf80ac0e"
#: the selection's calendar-index cutoff: notebook 04 uses the early diagonal 5, the
#: companion study the last known diagonal
CUTOFF = {"A": tlrn_colab.CUTOFF, "B": 10, "C": 10}
MAX_LOSS_RATIO = 3.0


# -- the cells -------------------------------------------------------------------------


def cells_from_mart(
    source: str | None = None, years: tuple[int, int] | None = None
) -> pd.DataFrame:
    """The four lines, accident years 1998 to 2007 (or ``years``), from the cas-schedule-p mart.

    One row per (company, line, accident year, development year), in ``CELL_COLUMNS``.
    ``years`` is the first and last accident year: cohort C uses ``YEARS["C"]``, 1988 to 1997.
    """
    import duckdb

    from ibnr.data.schedule_p import active_mart_path, pinned_source

    first, last = years or (tlrn_colab.ORIGINS[0].year, tlrn_colab.ORIGINS[-1].year)
    mart = active_mart_path(pinned_source(source or tlrn_colab.SOURCE_URL))
    lines = ", ".join(f"'{ln}'" for ln in tlrn_colab.LINES)
    frame = (
        duckdb.connect()
        .execute(f"""
        select company_code, line_of_business, statement_year, accident_year,
               development_age as dev_age, cum_paid_loss as paid, incurred_loss as incurred,
               bulk_loss as bulk, case_reserve as "case", earned_prem_net as premium
        from read_parquet('{mart.as_posix()}')
        where accident_year between {first} and {last}
          and line_of_business in ({lines})
    """)
        .df()
    )
    return _tidy(frame)


def cells_from_cas(cas_dir: str | Path) -> pd.DataFrame:
    """The same cells from the four cached CAS Schedule P files in ``cas_dir``.

    The case reserve is incurred minus paid minus bulk, which is how the mart's own
    ``case_reserve`` column is defined (checked cell for cell on the published mart).
    """
    folder = Path(cas_dir)
    missing = [f"{stem}.csv" for stem in CAS_FILES if not (folder / f"{stem}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"{folder} is missing {', '.join(missing)}. Cohort B needs the four CAS Schedule "
            f"P files {', '.join(f'{s}.csv' for s in CAS_FILES)} in one folder"
        )
    parts = []
    for stem, line in CAS_FILES.items():
        raw = pd.read_csv(folder / f"{stem}.csv", float_precision="round_trip")
        absent = [c for c in CAS_COLUMNS if c not in raw.columns]
        if absent:
            raise ValueError(f"{stem}.csv has no column(s) {absent}; found {list(raw.columns)}")
        part = pd.DataFrame(
            {
                "company_code": raw["GRCODE"].astype(str),
                "line_of_business": line,
                "statement_year": raw["DevelopmentYear"].astype(int),
                "accident_year": raw["AccidentYear"].astype(int),
                "dev_age": raw["DevelopmentLag"].astype(int),
                "paid": raw["CumPaidLoss"].astype(float),
                "incurred": raw["IncurredLosses"].astype(float),
                "bulk": raw["BulkLoss"].astype(float),
                "premium": raw["EarnedPremNet"].astype(float),
            }
        )
        part["case"] = part["incurred"] - part["paid"] - part["bulk"]
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    lo, hi = tlrn_colab.ORIGINS[0].year, tlrn_colab.ORIGINS[-1].year
    frame = frame[frame["accident_year"].between(lo, hi)]
    return _tidy(frame)


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """The cells in ``CELL_COLUMNS`` order and a fixed row order, so two routes compare equal."""
    frame = frame[CELL_COLUMNS].copy()
    for col in ("statement_year", "accident_year", "dev_age"):
        frame[col] = frame[col].astype(int)
    frame["company_code"] = frame["company_code"].astype(str)
    return frame.sort_values([*KEY, "accident_year", "dev_age"]).reset_index(drop=True)


# -- the selection ---------------------------------------------------------------------


def select_pairs(
    cells: pd.DataFrame, *, cutoff: int, max_loss_ratio: float = MAX_LOSS_RATIO
) -> tuple[list[tuple[str, str]], list[int]]:
    """The companion study's selection rule, with the calendar-index ``cutoff`` as a parameter.

    1. a complete square (every accident year at every development age) with a finite,
       strictly positive net earned premium on every cell;
    2. a finite, strictly positive cumulative paid and premium, and a cumulative paid loss
       ratio below ``max_loss_ratio``, on every cell with calendar index
       ``accident year - first accident year + development age`` at most ``cutoff``;
    3. at least two such lines in the company.

    Notebook 04 runs it with ``cutoff=5`` (cohort A) and the companion study with the
    cutoff at the last known diagonal (cohort B, 10 here). Returns the sorted pairs and the
    count after the input and after each of the three steps.
    """
    n_ay = cells["accident_year"].nunique()
    n_dev = cells["dev_age"].nunique()
    ay0 = int(cells["accident_year"].min())
    work = cells.assign(
        _bad_premium=~(np.isfinite(cells["premium"]) & (cells["premium"] > 0)),
        _calendar=cells["accident_year"] - ay0 + cells["dev_age"],
    )
    groups = work.groupby(KEY)
    audit = [groups.ngroups]
    grid = groups.agg(
        n_cells=("paid", "size"),
        n_ay=("accident_year", "nunique"),
        n_dev=("dev_age", "nunique"),
        n_bad_premium=("_bad_premium", "sum"),
    )
    complete = grid.index[
        (grid["n_cells"] == n_ay * n_dev)
        & (grid["n_ay"] == n_ay)
        & (grid["n_dev"] == n_dev)
        & (grid["n_bad_premium"] == 0)
    ]
    audit.append(len(complete))

    in_square = pd.MultiIndex.from_frame(work[KEY]).isin(complete)
    visible = work[in_square & (work["_calendar"] <= cutoff)]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = visible["paid"] / visible["premium"]
    unusable = (
        ~np.isfinite(visible["paid"])
        | (visible["paid"] <= 0)
        | visible["_bad_premium"]
        | ~np.isfinite(ratio)
        | (ratio >= max_loss_ratio)
    )
    bad = unusable.groupby([visible["company_code"], visible["line_of_business"]]).sum()
    eligible = [(str(c), str(ln)) for c, ln in bad.index[bad == 0]]
    audit.append(len(eligible))

    per_company: dict[str, int] = {}
    for company, _ in eligible:
        per_company[company] = per_company.get(company, 0) + 1
    pairs = sorted((c, ln) for c, ln in eligible if per_company[c] >= 2)
    audit.append(len(pairs))
    return pairs, audit


def pairs_hash(pairs) -> str:
    """A short hash of the sorted pairs, the same recipe as notebook 04's cohort hash."""
    return hashlib.sha256(repr(sorted(pairs)).encode()).hexdigest()[:16]


def cells_hash(cells: pd.DataFrame) -> str:
    """A short hash of the cell values, so two sources of one cohort can be shown equal."""
    ordered = _tidy(cells)
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()[:16]


def board_companies(cells: pd.DataFrame, pairs) -> list[str]:
    """The companies `mack` and `mcl` can both fit, as notebook 04 selects them.

    A development transition divides by the cumulative paid it starts from, so a
    non-positive (or missing) cumulative on a cell that has an observed successor, at
    calendar index at most ``n - 1``, rules a company out. A zero on an origin's latest
    diagonal is never divided by and is allowed.
    """
    chosen = cells[pd.MultiIndex.from_frame(cells[KEY]).isin(pairs)]
    n_ay = cells["accident_year"].nunique()
    ay0 = int(cells["accident_year"].min())
    regressors = chosen[(chosen["accident_year"] - ay0 + chosen["dev_age"]) <= n_ay - 1]
    nonpositive = regressors[~(regressors["paid"] > 0) | ~np.isfinite(regressors["paid"])]
    excluded = set(nonpositive["company_code"])
    return sorted({c for c, _ in pairs} - excluded)


def actual_reserves(cells: pd.DataFrame, pairs, *, as_of_year: int) -> pd.DataFrame:
    """Per company: anchor, realized ultimate and actual reserve, summed over its lines.

    The anchor is the cumulative paid on the valuation diagonal (statement year
    ``as_of_year``) added over accident years and lines; the ultimate is the cumulative
    paid at the deepest development age. ``premium`` is the net earned premium once per
    accident year and line.
    """
    chosen = cells[pd.MultiIndex.from_frame(cells[KEY]).isin(pairs)]
    deepest = chosen["dev_age"].max()
    anchor = chosen[chosen["statement_year"] == as_of_year].groupby("company_code")["paid"].sum()
    ultimate = chosen[chosen["dev_age"] == deepest].groupby("company_code")["paid"].sum()
    once = chosen.groupby([*KEY, "accident_year"])["premium"].first().reset_index()
    premium = once.groupby("company_code")["premium"].sum()
    out = pd.DataFrame({"anchor": anchor, "ultimate": ultimate, "premium": premium})
    out["actual_reserve"] = out["ultimate"] - out["anchor"]
    out.index.name = "company_code"
    return out.sort_index()


def triangle_from_cells(cells: pd.DataFrame, pairs):
    """The study triangle for ``pairs``, in the layout `load_schedule_p` gives.

    Segments ``company_code`` and ``line_of_business``; fields ``paid_loss``,
    ``incurred_loss`` (gross of bulk), ``case_reserve`` and ``earned_premium``; origin
    January 1 of the accident year, ``dev_lag`` in months, ``eval_date`` December 31 of the
    statement year.
    """
    from ibnr.triangle.io import from_long

    chosen = cells[pd.MultiIndex.from_frame(cells[KEY]).isin(pairs)]
    wide = pd.DataFrame(
        {
            "company_code": chosen["company_code"].to_numpy(),
            "line_of_business": chosen["line_of_business"].to_numpy(),
            "origin_period": [dt.date(int(y), 1, 1) for y in chosen["accident_year"]],
            "dev_lag": (chosen["dev_age"].to_numpy() * 12).astype("int64"),
            "eval_date": [dt.date(int(y), 12, 31) for y in chosen["statement_year"]],
            "paid_loss": chosen["paid"].to_numpy(),
            "incurred_loss": chosen["incurred"].to_numpy(),
            "case_reserve": chosen["case"].to_numpy(),
            "earned_premium": chosen["premium"].to_numpy(),
        }
    )
    return from_long(
        wide,
        fields=["paid_loss", "incurred_loss", "case_reserve", "earned_premium"],
        segments=KEY,
        measure="cumulative",
        origin_grain="Y",
        dev_grain="Y",
        units="USD thousands",
    )


@dataclass
class Cohort:
    """One cohort: who is in it, what it is worth, and the triangle the fits train on."""

    name: str
    pairs: list
    audit: list
    companies: list
    board: list
    pairs_hash: str
    data_hash: str
    actual: pd.DataFrame
    source: str
    #: the cells of the selected pairs, so a triangle can be built without the source again
    cells: pd.DataFrame | None = field(default=None, repr=False)
    triangle: object = field(default=None, repr=False)
    #: the valuation date the fits train on: the last known diagonal of this cohort
    as_of: dt.date = tlrn_colab.AS_OF
    #: every cell the source carries for this cohort's accident years (not only the selected
    #: pairs'), so a company set can be selected again at an earlier valuation date
    all_cells: pd.DataFrame | None = field(default=None, repr=False)
    #: ``{valuation: pairs}`` of the third study, filled by :func:`valuation_pairs`
    by_valuation: dict | None = field(default=None, repr=False)

    @property
    def description(self) -> str:
        return (
            f"cohort {self.name}: {len(self.companies)} companies, {len(self.pairs)} pairs, "
            f"{len(self.board)} scored; selection {self.audit}; hash {self.pairs_hash}; "
            f"data {self.data_hash}; from {self.source}"
        )


def select_cohort(name: str, cells: pd.DataFrame, *, source: str) -> Cohort:
    """Cohort ``name`` from ``cells``, with its published counts and hash asserted.

    No triangle is built here; see :func:`build_cohort`.
    """
    pairs, audit = select_pairs(cells, cutoff=CUTOFF[name])
    expected = EXPECTED[name]
    companies = sorted({c for c, _ in pairs})
    board = board_companies(cells, pairs)
    digest = pairs_hash(pairs)
    if (
        audit != expected["audit"]
        or len(companies) != expected["companies"]
        or len(board) != expected["board"]
        or digest != COHORT_HASH[name]
    ):
        raise AssertionError(
            f"cohort {name} from {source} does not match its published selection: counts "
            f"{audit} (expected {expected['audit']}), {len(companies)} companies (expected "
            f"{expected['companies']}), {len(board)} scored (expected {expected['board']}), "
            f"hash {digest} (expected {COHORT_HASH[name]})"
        )
    chosen = cells[pd.MultiIndex.from_frame(cells[KEY]).isin(pairs)]
    return Cohort(
        name=name,
        pairs=pairs,
        audit=audit,
        companies=companies,
        board=board,
        pairs_hash=digest,
        data_hash=cells_hash(chosen),
        actual=actual_reserves(cells, pairs, as_of_year=AS_OF[name].year),
        source=source,
        cells=chosen.reset_index(drop=True),
        as_of=AS_OF[name],
        all_cells=cells,
    )


def load_cohort(name: str, *, cas_dir: str | Path | None = None, cells=None) -> Cohort:
    """Cohort ``name`` selected and checked, without a triangle.

    Cohort B is selected from the mart, or from the four CAS files in ``cas_dir``; cohorts A
    and C only from the mart. ``cells`` lets a caller that already loaded the mart's cells
    for this cohort's accident years share them.
    """
    if name not in ALL_COHORTS:
        raise ValueError(f"cohort must be one of {ALL_COHORTS}, got {name!r}")
    if name != "B" and cas_dir is not None:
        raise ValueError(
            "cohort A is built from the cas-schedule-p package exactly as scripts/tlrn_colab.py "
            "builds it, and cohort C from the same package; --cas-dir is for cohort B"
        )
    if name == "B" and cas_dir is not None:
        return select_cohort(name, cells_from_cas(cas_dir), source=f"CAS files in {cas_dir}")
    source = f"cas-schedule-p publish {tlrn_colab.PUBLISH}"
    if cells is None:
        cells = cells_from_mart(years=YEARS[name])
    return select_cohort(name, cells, source=source)


def build_cohort(name: str, *, cas_dir: str | Path | None = None, cells=None) -> Cohort:
    """Cohort ``name`` with the triangle the fits train on.

    Cohort A's is ``tlrn_colab.build_triangle`` (notebook 04's own assertions), so it is
    the very triangle the earlier Colab fits used. Cohort B's is built from its cells.
    """
    cohort = load_cohort(name, cas_dir=cas_dir, cells=cells)
    if name == "A":
        triangle, n_pairs, n_companies = tlrn_colab.build_triangle()
        assert (n_pairs, n_companies) == (len(cohort.pairs), len(cohort.companies))
        cohort.triangle = triangle
    else:
        cohort.triangle = triangle_from_cells(cohort.cells, cohort.pairs)
    return cohort


# -- the fits --------------------------------------------------------------------------


@dataclass(frozen=True)
class Study:
    """What distinguishes one study from the other: its cohorts, rows, records and tables."""

    name: str
    #: ``settings["study"]``: the number a record carries, so a folder can hold both studies
    version: int
    cohorts: tuple
    rows: tuple
    #: the ``tlrn_colab.configs`` name that builds the rows
    config: str
    #: the extra per-fit tables (``_<name>.csv``) the evaluator reads
    tables: tuple = ()
    #: each valuation date is its own protocol on its own company set (:func:`fit_valuations`)
    per_valuation: bool = False
    #: the ibnr release this study's Colab kit installs and its fits are named with
    pinned_ibnr: str = EARLIER_PINNED_IBNR
    #: rows fitted once (draw 0) and only on ``single_draw_cohorts``; empty for the earlier studies
    single_draw_rows: tuple = ()
    single_draw_cohorts: tuple = ()
    #: the pre-registered stops (the first-fit time and the GPU-hour budget) and the retry of a
    #: non-finite fit are the runner's job for this study
    stop_rules: bool = False


FACTOR_HEAD_TABLES = ("valuations", "backtest", "members")
STUDIES = {
    "accident_year": Study("accident_year", STUDY_VERSION, COHORTS, ROWS, "accident_year"),
    "factor_head": Study(
        "factor_head",
        FACTOR_HEAD_VERSION,
        FACTOR_HEAD_COHORTS,
        FACTOR_HEAD_ROWS,
        "factor_head",
        FACTOR_HEAD_TABLES,
    ),
    "companion_final": Study(
        "companion_final",
        COMPANION_FINAL_VERSION,
        COMPANION_FINAL_COHORTS,
        COMPANION_FINAL_ROWS,
        "companion_final",
        COMPANION_FINAL_TABLES,
        per_valuation=True,
        pinned_ibnr=PINNED_IBNR,
    ),
    "companion_revised": Study(
        "companion_revised",
        COMPANION_REVISED_VERSION,
        COMPANION_REVISED_COHORTS,
        COMPANION_REVISED_ROWS,
        "companion_revised",
        COMPANION_REVISED_TABLES,
        per_valuation=True,
        pinned_ibnr=COMPANION_REVISED_PINNED_IBNR,
        single_draw_rows=SINGLE_DRAW_ROWS,
        single_draw_cohorts=SINGLE_DRAW_COHORTS,
        stop_rules=True,
    ),
}


def get_study(name: str) -> Study:
    if name not in STUDIES:
        raise ValueError(f"study must be one of {sorted(STUDIES)}, got {name!r}")
    return STUDIES[name]


def draw_seed(draw: int) -> int:
    """The fit seed of draw ``draw``: draw 0 is the earlier single fit's seed."""
    return tlrn_colab.SEED_FIT + SEED_STRIDE * draw


def row_configs(members: int, smoke: bool, study: str = "accident_year"):
    """``(row, fit keyword arguments, TLRNConfig)`` for the study's two rows."""
    return tlrn_colab.configs(get_study(study).config, members, smoke)


def fit_settings(
    cohort, row, roles, config, *, draw: int, smoke: bool, study: str = "accident_year"
) -> dict:
    """Everything that decides what a saved fit IS, as plain JSON values.

    The first study's settings are exactly what they were before the second existed, so its
    saved fits are still reused; the second study's carry ``study`` 2 and its name besides.
    """
    spec = get_study(study)
    config_fields = asdict(config)
    if (
        spec.version != COMPANION_REVISED_VERSION
        and config_fields.get("maturity_embedding") is True
    ):
        # The field is newer than the first three studies. With the table in (its only value
        # then) a record of theirs must read exactly as it did before the field existed, so a
        # fit saved by an earlier ibnr is still recognised as made with these settings.
        del config_fields["maturity_embedding"]
    settings = {
        "study": spec.version,
        "cohort": cohort.name,
        "row": row,
        "draw": draw,
        "seed": draw_seed(draw),
        "smoke": smoke,
        "config": config_fields,
        "roles": roles,
        "backend": "jax",
        "as_of": getattr(cohort, "as_of", tlrn_colab.AS_OF).isoformat(),
        "loss_field": tlrn_colab.FIELD,
        "publish_id": tlrn_colab.PUBLISH,
        "cohort_hash": cohort.pairs_hash,
        "data_hash": cohort.data_hash,
    }
    if spec.name != "accident_year":
        settings["study_name"] = spec.name
    if spec.version == COMPANION_REVISED_VERSION:
        # the fitted entry does not record it, so the settings and the run record do
        settings["paid_only"] = bool(roles.get("paid_only", False))
    if spec.per_valuation:
        # who is in each date's company set decides what the fit is, as the cohort hash does
        settings["valuations"] = list(VALUATIONS)
        settings["valuation_hashes"] = {
            str(v): pairs_hash(p) for v, p in valuation_pairs(cohort).items()
        }
    # through JSON, so a tuple in the config compares equal to the list read back
    return json.loads(json.dumps(settings))


def fit_paths(out: Path, cohort: str, row: str, draw: int, *, smoke: bool, version: str):
    """``(fitted entry, settings record, company reserves)`` of one fit.

    For example ``A_tlrn_8_ay_d03-ibnr0.7.4.pkl``, ``.json`` and ``_reserves.csv``.
    """
    stem = f"{cohort}_{row}_d{draw:02d}{'_smoke' if smoke else ''}-ibnr{version}"
    return out / f"{stem}.pkl", out / f"{stem}.json", out / f"{stem}_reserves.csv"


def table_paths(record: Path, names) -> dict[str, Path]:
    """The extra tables of a fit, next to its record: ``<stem>_<name>.csv``."""
    return {name: record.with_name(f"{record.stem}_{name}.csv") for name in names}


def settings_differences(mine: dict, theirs: dict) -> list[str]:
    """The names of the settings that differ, with the config's own fields spelled out."""
    differ = [k for k in sorted(mine.keys() | theirs.keys()) if mine.get(k) != theirs.get(k)]
    if "config" in differ:
        mine_c, theirs_c = mine.get("config") or {}, theirs.get("config") or {}
        differ.remove("config")
        differ += [
            f"config.{k}"
            for k in sorted(mine_c.keys() | theirs_c.keys())
            if mine_c.get(k) != theirs_c.get(k)
        ]
    return differ


def saved_fit(
    pkl: Path,
    record: Path,
    reserves: Path,
    settings: dict,
    tables: dict | None = None,
    *,
    tolerate: tuple = (),
) -> dict | None:
    """The record of a finished fit made with exactly ``settings``, or None if there is none.

    A fit is finished when its record exists, because the record is written last. A saved
    fit made with OTHER settings is refused, not reused and not overwritten; so is a
    record whose reserves file (or any of the study's extra ``tables``) is gone, and files
    left with no record (an interruption between the writes), because nothing then says
    what they were fitted with. Settings named in ``tolerate`` may differ.
    """
    tables = tables or {}
    if not record.exists():
        leftovers = [p.name for p in (pkl, reserves, *tables.values()) if p.exists()]
        if leftovers:
            raise SystemExit(
                f"{', '.join(leftovers)} exist with no settings record ({record.name}), so "
                "nothing says what they were fitted with. Delete them to refit, or write "
                "this run to a different --out folder"
            )
        return None
    saved = json.loads(record.read_text(encoding="utf-8"))
    differ = [
        name
        for name in settings_differences(settings, saved.get("settings", {}))
        if name not in tolerate
    ]
    if differ:
        raise SystemExit(
            f"{record.name} was fitted with different settings from this run's (they differ "
            f"in {', '.join(differ)}). Write this run to a different --out folder, or delete "
            "its .pkl, .json and .csv files to refit it here"
        )
    for path in (reserves, *tables.values()):
        if not path.exists():
            raise SystemExit(
                f"{record.name} is a finished fit but {path.name} is missing. Delete the "
                "record (and the .pkl) to refit it, or restore the file"
            )
    return saved


def reserves_frame(entry) -> pd.DataFrame:
    """One row per company: anchor, premium, lines, the ensemble reserve and each member's.

    Rows follow ``entry.cohorts()``. The ensemble reserve is the mean of the member
    columns when every member is kept, which is the accident-year variant. The third
    study's :class:`ValuationFits` carries the frame of its last known diagonal itself.
    """
    if isinstance(entry, ValuationFits):
        return entry.reserves
    codes = [cohort["company_code"] for cohort in entry.cohorts()]
    members = np.atleast_2d(np.asarray(entry.member_company_reserves(), dtype=float))
    frame = pd.DataFrame(
        {
            "company_code": codes,
            "anchor": np.asarray(entry.company_anchors(), dtype=float),
            "premium": np.asarray(entry.company_size_, dtype=float),
            "n_lines": np.asarray(entry.contract_["line_mask"]).sum(axis=1).astype(int),
            "reserve": np.asarray(entry.company_reserves(), dtype=float),
        }
    )
    for m in range(members.shape[0]):
        frame[f"member_{m:02d}"] = members[m]
    return frame


# -- the third study: one protocol per valuation date, each on its own companies ----------


def valuation_pairs(cohort, valuations=VALUATIONS) -> dict:
    """``{v: pairs}``: the company-line pairs selected again at each valuation date.

    ``select_pairs(cells, cutoff=v)`` on all the cohort's cells, which is the companion's
    ``select_cohort(cutoff=v)``. The last valuation date is the cohort's own cutoff, so its
    set must be the cohort's pairs; the sets are nested (an earlier date selects every pair a
    later one does). Both are checked, and the result is kept on the cohort.
    """
    if cohort.by_valuation is not None and tuple(cohort.by_valuation) == tuple(valuations):
        return cohort.by_valuation
    if cohort.all_cells is None:
        raise ValueError(
            f"cohort {cohort.name} carries no cells to select again at an earlier date"
        )
    sets = {v: select_pairs(cohort.all_cells, cutoff=v)[0] for v in valuations}
    last = max(valuations)
    if last == CUTOFF[cohort.name] and sets[last] != cohort.pairs:
        raise AssertionError(
            f"cohort {cohort.name}: the pairs selected at valuation {last} are not the cohort's"
        )
    ordered = sorted(valuations)
    for early, late in zip(ordered, ordered[1:], strict=False):
        lost = set(sets[late]) - set(sets[early])
        if lost:
            raise AssertionError(
                f"cohort {cohort.name}: {len(lost)} pair(s) selected at valuation {late} are "
                f"not selected at the earlier valuation {early}, so the sets are not nested"
            )
    cohort.by_valuation = sets
    return sets


@dataclass
class ValuationFits:
    """What a third-study fit keeps: the tables of every date and the last date's reserves.

    A plain container (no torch objects, no closures), so it pickles. ``reserves`` is the
    frame :func:`reserves_frame` returns for the last known diagonal: the refit ensemble's
    reserve and each member's. ``reserves_selected`` is the same ensemble before the refit.
    """

    valuations: pd.DataFrame
    members: pd.DataFrame
    reserves: pd.DataFrame
    reserves_selected: pd.DataFrame
    pairs: dict
    seeds: dict
    config: object = field(repr=False, default=None)
    #: the fourth study's paid-only row (the fitted object itself does not record it)
    paid_only: bool = False

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return {"valuations": self.valuations, "members": self.members}


def date_seed(seed: int, v: int) -> int:
    """The seed of valuation ``v``'s protocol inside a fit with seed ``seed``."""
    return seed + DATE_SEED_STRIDE * v


def study_roles() -> dict:
    """The third study always reads the incurred and case channels (the 13 features)."""
    return {"incurred_field": tlrn_colab.INCURRED_FIELD, "case_field": tlrn_colab.CASE_FIELD}


def fit_valuation(
    triangle,
    v: int,
    config,
    seed: int,
    *,
    as_of,
    backend: str = "jax",
    show_progress: bool = False,
    paid_only: bool = False,
) -> dict:
    """Train, refit and score ONE valuation date ``v`` on the companies of ``triangle``.

    ``paid_only`` is the fourth study's paid-only row: the five incurred and case reserve input
    channels are set to zero in every feature set while all 13 inputs stay (the same option
    ``TLRN.fit(paid_only=True)`` has; this function builds its own features, so the option has
    to be passed on here and is not inherited).

    The protocol is the entry's own (``TLRN._run_protocols`` with one spec) on the full
    10-lag contract of ``triangle`` at ``as_of``, with the network trained on the data known at
    ``v`` and the forecasts of the cells after ``v`` scored. Slicing the triangle at ``v`` is
    not used: it would cut the lags the network is built for. At the last known diagonal the
    spec is the entry's final protocol (forecast every later diagonal); at an earlier date it
    is a calibration protocol (forecast up to the last known diagonal, lags up to ``v``).
    ``seed`` is the date's own seed (:func:`date_seed`). Returns the run, the contract and the
    entry that ran it, for :func:`date_tables`.
    """
    if not config.refit:
        raise ValueError("the third study's tables compare the refit with the run before it")
    if not isinstance(paid_only, bool):
        raise ValueError(f"paid_only must be True or False, got {paid_only!r}")
    import torch

    from ibnr.gallery.nn._scheme import splits
    from ibnr.gallery.nn.tlrn.model import TLRN, _load_jax_backend
    from ibnr.kernels.nn_contract import nn_company_data
    from ibnr.kernels.nn_features import tlrn_features

    roles = study_roles()
    incurred, case = roles["incurred_field"], roles["case_field"]
    contract = nn_company_data(
        triangle.as_of(as_of),
        loss_field=tlrn_colab.FIELD,
        feature_fields=(incurred, case),
        level_fields=(case,),
        premium_field="earned_premium",
    )
    n_c, n_l, _, n_w, n_d = contract["x"].shape
    last_diagonal = n_w + n_d - 1
    obs_any = contract["obs_mask"].any(axis=1)
    _, _, train_end = splits(obs_any, contract["cal_idx"], config.val_diagonals)
    last_known = train_end + config.val_diagonals
    if not 1 <= v <= last_known:
        raise ValueError(f"valuation {v} is outside 1 to the last known diagonal {last_known}")

    def features(cutoff, target_lo, target_hi, max_lag=None, with_mcl=None):
        return tlrn_features(
            contract,
            cutoff=cutoff,
            target_lo=target_lo,
            target_hi=target_hi,
            incurred_field=incurred,
            case_field=case,
            with_mcl=config.member == "mcl_blend" if with_mcl is None else with_mcl,
            max_lag=max_lag,
            paid_only=paid_only,
        )

    if v == last_known:
        spec = {"c_max": v, "target_hi": last_diagonal}
    else:
        reached = config.scoring == "reached_cells"
        spec = {"c_max": v, "target_hi": last_known, "final_max_lag": v if reached else None}
    spec |= {"seed": seed, "show_progress": show_progress}
    entry = TLRN()
    (run,) = entry._run_protocols(
        contract,
        config,
        [spec],
        features=features,
        dev=torch.device("cpu"),
        processes=1,
        torch=torch,
        jax_backend=_load_jax_backend() if backend == "jax" else None,
    )
    return {"run": run, "contract": contract, "entry": entry, "last_known": last_known, "v": v}


def date_tables(fitted: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(valuations rows, members rows)`` of one date, from :func:`fit_valuation`'s result.

    Per company: the cells scored, the mean over the members of the refit blend
    (``predicted``), of the same members before the refit (``predicted_selected``), of the raw
    refit network (``predicted_net``) and of the raw network before the refit
    (``predicted_net_selected``), each added over the cells scored; and the actual over them
    where every scored diagonal is already observed.
    """
    import torch

    from ibnr.gallery.nn.tlrn.model import TLRN
    from ibnr.kernels.nn_contract import cohort_identities

    run, contract, v = fitted["run"], fitted["contract"], fitted["v"]
    codes = [c["company_code"] for c in cohort_identities(contract, key="companies")]
    n_c = len(codes)
    final, tensors = run["final_set"], run["final_tensors"]
    after, before = run, run["selected"]

    def mean_of(arm, blend: bool) -> np.ndarray:
        with torch.no_grad():
            stack = [
                arm["forward"](m, tensors, blend=blend)["pred"].cpu().numpy() for m in arm["kept"]
            ]
        return TLRN._company_totals(np.mean(stack, axis=0), final, n_c)

    n_cells = np.zeros(n_c)
    np.add.at(n_cells, final["example_company"], final["target_mask"].sum(axis=1))
    known = int(final["target_hi"]) <= fitted["last_known"]
    actual = TLRN._company_totals(final["target"], final, n_c) if known else np.full(n_c, np.nan)
    valuations = pd.DataFrame(
        {
            "valuation": v,
            "company_code": codes,
            "n_cells": n_cells.astype(int),
            "predicted": mean_of(after, True),
            "predicted_selected": mean_of(before, True),
            "predicted_net": mean_of(after, False),
            "predicted_net_selected": mean_of(before, False),
            "actual": actual,
        }
    )
    members = run["selection"].copy()
    members.insert(0, "valuation", v)
    # the weights kept are from a check before any bad step, so nothing in the forecasts shows
    # a member whose loss went non-finite later; the refit's own loss needs no count, because
    # the refit keeps its last weights and a non-finite step would reach its forecasts
    members[NON_FINITE_LOSS] = [
        sum(not math.isfinite(r["train"]) for r in records) for records in run["history"]
    ]
    return valuations, members


def date_reserves(fitted: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The reserves frame of a fit at the last known diagonal: refit, and before the refit."""
    from ibnr.gallery.nn.tlrn.model import TLRN
    from ibnr.kernels.nn_contract import cohort_identities

    run, contract, entry = fitted["run"], fitted["contract"], fitted["entry"]
    final, tensors = run["final_set"], run["final_tensors"]
    codes = [c["company_code"] for c in cohort_identities(contract, key="companies")]
    mask = contract["line_mask"]
    base = pd.DataFrame(
        {
            "company_code": codes,
            "anchor": np.where(mask[:, :, None], contract["latest_cum"], 0.0).sum(axis=(1, 2)),
            "premium": [
                float(np.nansum(np.where(mask[c][:, None], contract["premium"][c], np.nan)))
                for c in range(len(codes))
            ],
            "n_lines": mask.sum(axis=1).astype(int),
        }
    )

    def frame(arm) -> pd.DataFrame:
        def reserves(models) -> np.ndarray:
            point = entry._ensemble_pred(models, tensors, arm["forward"])
            _, per, _ = TLRN._assemble_point(
                point, final, contract, models, tensors, arm["forward"]
            )
            return np.nansum(per, axis=(1, 2))

        out = base.copy()
        out["reserve"] = reserves(arm["kept"])
        for m, model in enumerate(arm["models"]):
            out[f"member_{m:02d}"] = reserves([model])
        return out

    return frame(run), frame(run["selected"])


def fit_valuations(
    cohort,
    config,
    seed: int,
    *,
    valuations=VALUATIONS,
    backend: str = "jax",
    show_progress: bool = False,
    log=print,
    paid_only: bool = False,
) -> ValuationFits:
    """The third study's fit: every valuation date trained on its own company set.

    For each date ``v`` the companies are those of ``select_pairs(cutoff=v)`` (the pairs of
    :func:`valuation_pairs`), a triangle is built from them, and :func:`fit_valuation` trains,
    refits and scores the date. Date ``v`` has seed ``seed + 20000 v``, its members
    ``+ 1000 m``. The last known diagonal's reserves become ``reserves``.
    """
    sets = valuation_pairs(cohort, valuations)
    last = max(valuations)
    tables, members, reserves, selected, seeds = [], [], None, None, {}
    for v in valuations:
        started = time.perf_counter()
        triangle = triangle_from_cells(cohort.all_cells, sets[v])
        fitted = fit_valuation(
            triangle,
            v,
            config,
            date_seed(seed, v),
            as_of=cohort.as_of,
            backend=backend,
            show_progress=show_progress and v == last,
            paid_only=paid_only,
        )
        table, member_rows = date_tables(fitted)
        tables.append(table)
        members.append(member_rows)
        seeds[v] = date_seed(seed, v)
        if v == last:
            reserves, selected = date_reserves(fitted)
        log(
            f"    valuation {v}: {len(sets[v])} pairs, {len(table)} companies, "
            f"{time.perf_counter() - started:.0f}s"
        )
    return ValuationFits(
        valuations=pd.concat(tables, ignore_index=True),
        members=pd.concat(members, ignore_index=True),
        reserves=reserves,
        reserves_selected=selected,
        pairs=sets,
        seeds=seeds,
        config=config,
        paid_only=paid_only,
    )


# -- what the second study saves besides the reserves ------------------------------------


@contextmanager
def capture_protocols():
    """Collect the finished protocol of every valuation date while ``fit`` runs.

    A fit trains the method at the last known diagonal and, under
    ``calibration="retrain_per_valuation"``, again at each earlier valuation date; the
    entry keeps only the final one's members and, from the others, the company totals that
    became ``backtest_``. The second study scores every valuation date, and the raw network
    before the blend, so it needs the others' members. This wraps the entry's own
    ``_finish_protocol`` (the function that turns trained members into a protocol's result)
    and records what it returns, changing nothing the fit computes. It is a private name:
    :func:`valuation_tables` checks what it collects against the entry, and the runner's tests
    fit a tiny model through it.
    """
    from ibnr.gallery.nn.tlrn.model import TLRN

    original = getattr(TLRN, "_finish_protocol", None)
    if original is None:
        raise RuntimeError(
            "this ibnr has no TLRN._finish_protocol, which the second study's runner wraps to "
            "collect the retrained valuation dates; it was written for ibnr 0.7.4"
        )
    runs: list[dict] = []

    def finish(self, prep, cfg, models, history):
        run = original(self, prep, cfg, models, history)
        runs.append(run)
        return run

    TLRN._finish_protocol = finish
    try:
        yield runs
    finally:
        TLRN._finish_protocol = original


def _close(a, b, what: str) -> None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = max(1.0, float(np.nanmax(np.abs(b))) if b.size else 1.0)
    if not np.allclose(a, b, rtol=1e-4, atol=1e-5 * scale, equal_nan=True):
        gap = float(np.nanmax(np.abs(a - b)))
        raise RuntimeError(
            f"{what}: the valuation tables differ from the fitted entry by up to {gap:,.4f} "
            f"(scale {scale:,.0f}); the collected protocols are not the entry's own, so "
            "nothing is saved"
        )


def valuation_tables(entry, runs) -> dict[str, pd.DataFrame]:
    """The three tables of the second study's fit: valuations, the entry's backtest, members.

    ``runs`` is what :func:`capture_protocols` collected, in the order ``fit`` finished
    them: the last known diagonal first, then each calibration valuation date. For each, the
    mean over its members of the blended forecast (``predicted``) and of the raw network
    (``predicted_net``), added over the cells scored, per company; the actual over the same
    cells where the diagonals are known; and the number of cells. Before it returns, the
    last known diagonal's ``predicted`` must equal ``entry.company_reserves()`` and each
    retrained date's must equal the entry's own ``backtest_`` row, so a collection that is
    not the entry's own stops the run before anything is written.
    """
    import torch

    from ibnr.gallery.nn.tlrn.model import TLRN

    cfg = entry.config_
    codes = [c["company_code"] for c in entry.cohorts()]
    n_c = len(codes)
    last = int(entry.feature_stats_["final_cutoff"])
    expected = [last, *[int(k) for k in cfg.calibration_cutoffs]]
    got = [int(run["final_set"]["cutoff"]) for run in runs]
    if cfg.calibration != "retrain_per_valuation" or got != expected:
        raise ValueError(
            f"the valuation tables need one protocol per valuation date {expected} "
            f"(calibration='retrain_per_valuation'); the fit finished protocols at {got} "
            f"with calibration={cfg.calibration!r}"
        )

    valuations, members = [], []
    for run in runs:
        final, tensors, forward = run["final_set"], run["final_tensors"], run["forward"]
        v = int(final["cutoff"])
        with torch.no_grad():
            blended = np.mean(
                [forward(m, tensors)["pred"].cpu().numpy() for m in run["kept"]], axis=0
            )
            network = np.mean(
                [forward(m, tensors, blend=False)["pred"].cpu().numpy() for m in run["kept"]],
                axis=0,
            )
        n_cells = np.zeros(n_c)
        np.add.at(n_cells, final["example_company"], final["target_mask"].sum(axis=1))
        # the actual is known only where every scored diagonal is already observed
        known = int(final["target_hi"]) <= last
        actual = (
            TLRN._company_totals(final["target"], final, n_c) if known else np.full(n_c, np.nan)
        )
        valuations.append(
            pd.DataFrame(
                {
                    "valuation": v,
                    "company_code": codes,
                    "n_cells": n_cells.astype(int),
                    "predicted": TLRN._company_totals(blended, final, n_c),
                    "predicted_net": TLRN._company_totals(network, final, n_c),
                    "actual": actual,
                }
            )
        )
        selection = run["selection"].copy()
        selection.insert(0, "valuation", v)
        members.append(selection)
    valuations = pd.concat(valuations, ignore_index=True)
    members = pd.concat(members, ignore_index=True)

    now = valuations[valuations["valuation"] == last]
    _close(now["predicted"], entry.company_reserves(), f"valuation {last} reserves")
    back = entry.backtest_.copy()  # saved as the entry has it, company columns included
    if back["company_code"].tolist() != [codes[u] for u in back["unit"]]:
        raise RuntimeError("the entry's backtest rows do not name the companies of its cohorts")
    joined = back.merge(
        valuations.rename(columns={"predicted": "table_predicted", "actual": "table_actual"}),
        left_on=["company_code", "cutoff"],
        right_on=["company_code", "valuation"],
        how="left",
    )
    _close(joined["predicted"], joined["table_predicted"], "the entry's backtest predictions")
    _close(joined["actual"], joined["table_actual"], "the entry's backtest actuals")
    return {"valuations": valuations, "backtest": back, "members": members}


def write_atomic(path: Path, write) -> None:
    """Write through a temporary name, so an interrupted write never looks like a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    write(partial)
    os.replace(partial, path)


def save_entry(entry, path: Path) -> None:
    def write(partial: Path) -> None:
        with open(partial, "wb") as handle:
            pickle.dump(entry, handle, protocol=pickle.HIGHEST_PROTOCOL)

    write_atomic(path, write)


def save_reserves(frame: pd.DataFrame, path: Path) -> None:
    write_atomic(path, lambda partial: frame.to_csv(partial, index=False))


def save_record(path: Path, settings: dict, run: dict) -> None:
    write_atomic(
        path,
        lambda partial: partial.write_text(
            json.dumps({"settings": settings, "run": run}, indent=2), encoding="utf-8"
        ),
    )


def collect_records(out: Path, *, smoke: bool, study: str = "accident_year") -> dict[str, dict]:
    """Every finished fit's record in ``out`` of this study and kind (smoke or full), by stem."""
    spec = get_study(study)
    records = {}
    for path in sorted(out.glob("*-ibnr*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        settings = record.get("settings", {})
        if settings.get("study") != spec.version or bool(settings.get("smoke")) != smoke:
            continue
        if spec.name != "accident_year" and settings.get("study_name") != spec.name:
            continue
        records[path.stem] = record
    return records


def write_manifest(out: Path, *, smoke: bool, study: str = "accident_year") -> Path:
    """The folder's manifest, built from the records of the fits actually in it."""
    spec = get_study(study)
    stem = "manifest_study" if spec.name == "accident_year" else f"manifest_{spec.name}"
    path = out / f"{stem}{'_smoke' if smoke else ''}.json"
    body = {
        "study": spec.version,
        "smoke": smoke,
        "publish_id": tlrn_colab.PUBLISH,
        "fits": collect_records(out, smoke=smoke, study=study),
    }
    if spec.name != "accident_year":
        body["study_name"] = spec.name
    write_atomic(path, lambda partial: partial.write_text(json.dumps(body, indent=2), "utf-8"))
    return path


def row_applies(spec: Study, row: str, cohort_name: str, draw: int) -> bool:
    """Whether ``row`` is fitted on this cohort at this draw: single-draw rows only at draw 0."""
    if row in spec.single_draw_rows:
        return draw == 0 and cohort_name in spec.single_draw_cohorts
    return True


def check_row_request(spec: Study, rows, cohort_names, draws: int) -> None:
    """Refuse, by name, a single-draw row asked for with several draws or on another cohort.

    Raises ``ValueError``. The runner's default (no ``--rows``) is every row of the study and
    plans each single-draw row where it applies; only an explicit request is refused.
    """
    for row in rows:
        if row not in spec.single_draw_rows:
            continue
        if draws > 1:
            raise ValueError(
                f"{row} is fitted once, at draw 0 (the pre-registration's single-draw rows carry "
                f"no verdict and are not repeated), so --draws {draws} is refused; use --draws 1"
            )
        outside = [n for n in cohort_names if n not in spec.single_draw_cohorts]
        if outside:
            raise ValueError(
                f"{row} is fitted on cohort {', '.join(spec.single_draw_cohorts)} only, so cohort "
                f"{', '.join(outside)} is refused"
            )


def plan_study(
    out: Path,
    cohorts: dict,
    rows,
    draws: int,
    *,
    members: int,
    smoke: bool,
    version: str,
    study: str = "accident_year",
    reuse_dir: Path | None = None,
):
    """What to fit and what is already done, checked before any training.

    Returns ``(todo, done)``. Both are lists of dicts in draw-major order (all rows and
    cohorts of draw 0, then of draw 1, ...), so a run stopped at any point holds an
    even set of fits. The fourth study's ``todo`` is in its pre-registered order instead
    (:func:`registered_order`). A saved fit with other settings raises ``SystemExit`` here. Rows
    that apply only at draw 0 on some cohorts (the fourth study's) are left out elsewhere. With
    ``reuse_dir``, an item of the fourth study's network-off row has ``reuse`` set: it is taken
    from the third study's fits in that folder instead of being trained. For a study with
    ``stop_rules`` a saved fit may differ in ``data_hash``, which is the text of a table and so
    moves with the pandas and numpy of the session; the pre-registration says that alone does
    not stop the study, and the item carries the saved value as ``saved_data_hash``.
    """
    spec = get_study(study)
    tolerate = ("data_hash",) if spec.stop_rules else ()
    configs = {row: (roles, cfg) for row, roles, cfg in row_configs(members, smoke, study)}
    rows = list(dict.fromkeys(rows))  # a repeated row would train the same fit twice
    todo, done = [], []
    for draw in range(draws):
        for name, cohort in cohorts.items():
            for row in rows:
                if not row_applies(spec, row, name, draw):
                    continue
                roles, cfg = configs[row]
                settings = fit_settings(
                    cohort, row, roles, cfg, draw=draw, smoke=smoke, study=study
                )
                pkl, record, reserves = fit_paths(
                    out, name, row, draw, smoke=smoke, version=version
                )
                tables = table_paths(record, spec.tables)
                saved = saved_fit(pkl, record, reserves, settings, tables, tolerate=tolerate)
                item = {
                    "cohort": name,
                    "row": row,
                    "draw": draw,
                    "roles": roles,
                    "config": cfg,
                    "settings": settings,
                    "pkl": pkl,
                    "record": record,
                    "reserves": reserves,
                    "tables": tables,
                    "reuse": reuse_dir is not None and row == REVISED_FACTORS_ROW and not smoke,
                    "saved_data_hash": (saved or {}).get("settings", {}).get("data_hash"),
                }
                (done if saved is not None else todo).append(item)
    if spec.stop_rules:
        todo.sort(key=lambda item: registered_order(spec, item))
    return todo, done


def registered_order(spec: Study, item: dict) -> tuple:
    """Where a fit of the fourth study comes in the pre-registered order.

    The main row on C4 (its draw 0 is the pipeline check, and the 25-minute stop reads it), the
    main row on B4, the network-off row, then the single-draw rows.
    """
    return (
        spec.rows.index(item["row"]),
        COMPANION_REVISED_FIT_ORDER.index(item["cohort"]),
        item["draw"],
    )


# -- the fourth study: the stops, the retry, and the reuse of the third study's factors fits ----


def failed_seconds(out: Path, *, smoke: bool = False) -> float:
    """Seconds spent on the fits reported as failed, read from their ``FAILED_*.txt`` files."""
    total = 0.0
    for path in out.glob("FAILED_*.txt"):
        if ("_smoke" in path.name) == smoke:
            found = re.search(r"^seconds: ([0-9.]+)$", path.read_text("utf-8"), re.MULTILINE)
            total += float(found.group(1)) if found else 0.0
    return total


def fit_seconds_used(out: Path, *, study: str) -> float:
    """Seconds the study's full fits took, not counting fits taken from another study.

    The budget is GPU time, so an attempt that gave no usable fit counts too: the first attempt
    of a fit that was rerun (``failed_attempt_seconds``) and every attempt of a reported failure.
    """
    records = collect_records(out, smoke=False, study=study)
    saved = sum(
        float(r.get("run", {}).get("fit_seconds", 0.0))
        + float(r.get("run", {}).get("failed_attempt_seconds", 0.0))
        for r in records.values()
        if not r.get("run", {}).get("reused_from")
    )
    return saved + failed_seconds(out)


def check_budget(used_seconds: float, limit_hours: float = BUDGET_HOURS) -> float:
    """Stop (pre-registered) when the fits already add up to the budget; hours used.

    ``limit_hours`` is larger than the registered 8 only when Ethan has answered this stop.
    """
    used = used_seconds / 3600.0
    if used >= limit_hours:
        raise SystemExit(
            f"STOP: the saved fits add up to {used:.2f} GPU hours, at or over the pre-registered "
            f"{limit_hours:g} hours. Ask Ethan before running more. If he agrees to more, pass "
            "the new total with --answered-budget-hours (BUDGET_ANSWER_HOURS in the notebook)."
        )
    return used


def check_first_fit(
    seconds: float, limit: float = STOP_FIRST_FIT_SECONDS, *, answer: str | None = None
) -> float:
    """Stop (pre-registered) when draw 0 of the main row on cohort C took over 25 minutes.

    ``answer`` is Ethan's answer to this stop; with it the run goes on.
    """
    if seconds > limit and not answer:
        raise SystemExit(
            f"STOP: draw 0 on C4 took {seconds / 60:.1f} minutes, over the pre-registered "
            f"{limit / 60:.0f}. Ask Ethan before running more. If he says go on, pass his answer "
            "with --answered-first-fit (FIRST_FIT_ANSWER in the notebook)."
        )
    return seconds


def is_first_fit(item: dict) -> bool:
    """The fit the 25-minute rule is about: the main row, cohort C, draw 0."""
    return (item["cohort"], item["row"], item["draw"]) == ("C", REVISED_MAIN_ROW, 0)


def non_finite(frame: pd.DataFrame, tables: dict) -> list[str]:
    """Names of the saved quantities that hold a missing or infinite value where one is a defect.

    The forecast columns of the valuations table, the reserve column, and the members table's
    stored epochs and blend weight, and any member with a non-finite training loss, whose kept
    weights can be finite. (``actual`` is empty at the last valuation by design.)
    """
    bad = []
    checks = [("reserves.reserve", frame["reserve"])]
    valuations = tables.get("valuations")
    if valuations is not None:
        for col in ("predicted", "predicted_selected", "predicted_net", "predicted_net_selected"):
            if col in valuations:
                checks.append((f"valuations.{col}", valuations[col]))
    members = tables.get("members")
    if members is not None:
        for col in ("alpha", "best_epoch", "refit_epochs"):
            if col in members:
                checks.append((f"members.{col}", members[col]))
    for name, values in checks:
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            bad.append(name)
    if members is not None and members.get(NON_FINITE_LOSS, pd.Series(dtype=int)).any():
        bad.append(f"members.{NON_FINITE_LOSS}")
    return bad


def failure_path(out: Path, item: dict) -> Path:
    """Where a failed fit is reported: a text file that no reader of records can mistake for one."""
    return item["record"].with_name(f"FAILED_{item['record'].stem}.txt")


def fit_or_report(item: dict, fit_fn, cohort, spec: Study, *, log=print):
    """Run ``fit_fn`` for one item; ``(entry, run, tables)`` or ``None`` for a reported failure.

    A fit whose forecasts or stored epochs are not finite, with a member whose training loss was
    not finite, or with a member that never had a finite check (the entry's
    ``NoFiniteValidation``, raised before the refit) is rerun ONCE with the same seed; if that
    recurs, it is reported as failed and left out (its cohort's verdict is then provisional,
    which the evaluator prints). A single-draw row that raises is reported as failed too and
    delays nothing. Any other exception from a row that carries a verdict is a defect and
    propagates. The seconds of the attempts that gave no usable fit go into the failure file, or
    into the run record as ``failed_attempt_seconds``, because they are GPU time like any other.
    """
    from ibnr.gallery.nn.tlrn.model import NoFiniteValidation

    seed = item["settings"]["seed"]
    single = item["row"] in spec.single_draw_rows
    reason = ""
    lost = 0.0
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            entry, run, *rest = fit_fn(cohort, item["row"], item["roles"], item["config"], seed)
        except NoFiniteValidation as error:
            lost += time.perf_counter() - started
            reason = f"non-finite training: {error}"
        except Exception as error:
            if not single:
                raise
            lost += time.perf_counter() - started
            reason = f"{type(error).__name__}: {error}"
            log(f"    {item['row']} failed ({reason}); reported, nothing waits for it")
            break
        else:
            tables = rest[0] if rest else {}
            bad = non_finite(reserves_frame(entry), tables)
            if not bad:
                if lost:
                    run = {**run, "failed_attempt_seconds": round(lost, 1)}
                return entry, run, tables
            lost += float(run.get("fit_seconds", time.perf_counter() - started))
            reason = f"non-finite {', '.join(bad)}"
        if attempt == 1:
            log(f"    {reason}; rerunning once with the same seed {seed}")
        else:
            log(f"    {reason} again; reported as failed and left out")
    failure_path(item["record"].parent, item).write_text(
        f"{item['cohort']} {item['row']} draw {item['draw']} (seed {seed}) failed: {reason}\n"
        f"seconds: {lost:.1f}\n",
        encoding="utf-8",
    )
    return None


def reuse_factors_fit(item: dict, reuse_dir: Path) -> tuple:
    """The fourth study's network-off row taken from the third study's fit of the same draw.

    Allowed only after the pre-registered identity check passed (the amendment in the
    pre-registration says so). The third study's ``tlrn_cf_factors`` record for this cohort and
    draw must be that study's (study 3, its name, the seed its draw implies, not a smoke fit),
    for this cohort's pairs, the same publish, the same valuation date and the same company
    set at every valuation date, and its configuration must be this row's except for the one
    field the identity check covers, ``maturity_embedding``. The data hash may differ (the
    third study's fits ran on another machine); both are kept in the run record. The tables are
    copied unchanged. Returns ``(reserves frame, run, tables)`` where ``run`` carries
    ``reused_from``, the source fit's own environment and seconds, and no new seconds.
    """
    source_study = get_study("companion_final")
    matches = sorted(
        Path(reuse_dir).glob(f"{item['cohort']}_tlrn_cf_factors_d{item['draw']:02d}-ibnr*.json")
    )
    matches = [p for p in matches if "_smoke" not in p.name]
    if len(matches) != 1:
        raise SystemExit(
            f"cohort {item['cohort']} draw {item['draw']}: expected one finished third-study "
            f"tlrn_cf_factors fit in {reuse_dir}, found {len(matches)}"
        )
    path = matches[0]
    record = json.loads(path.read_text(encoding="utf-8"))
    theirs, mine = record.get("settings", {}), item["settings"]
    expected = {
        "study": source_study.version,
        "study_name": source_study.name,
        "row": "tlrn_cf_factors",
        "cohort": item["cohort"],
        "draw": item["draw"],
        "seed": draw_seed(item["draw"]),
        "smoke": False,
    }
    for key, want in expected.items():
        if theirs.get(key) != want:
            raise SystemExit(f"{path.name}: {key} is {theirs.get(key)!r}, expected {want!r}")
    same = ("cohort_hash", "publish_id", "as_of", "loss_field", "valuations", "valuation_hashes")
    differ = [k for k in same if theirs.get(k) != mine.get(k)]
    if differ:
        raise SystemExit(
            f"{path.name} was fitted on a different cohort than this run's ({', '.join(differ)} "
            "differ), so it cannot stand in for this study's network-off row"
        )
    mine_cfg = {k: v for k, v in mine["config"].items() if k != "maturity_embedding"}
    their_cfg = {k: v for k, v in theirs["config"].items() if k != "maturity_embedding"}
    config_differ = [
        k for k in sorted(mine_cfg.keys() | their_cfg.keys()) if mine_cfg.get(k) != their_cfg.get(k)
    ]
    if config_differ or theirs.get("roles") != {
        k: v for k, v in mine["roles"].items() if k != "paid_only"
    }:
        raise SystemExit(
            f"{path.name} was fitted with other settings than this row's "
            f"({', '.join(config_differ) or 'roles'} differ), so it is not reused"
        )
    stem = path.stem
    reserves = pd.read_csv(path.with_name(f"{stem}_reserves.csv"), dtype={"company_code": str})
    tables = {
        name: pd.read_csv(path.with_name(f"{stem}_{name}.csv"), dtype={"company_code": str})
        for name in COMPANION_REVISED_TABLES
    }
    # the source fit's own environment stays under ``reused_from``; the record names no other
    run = {
        "fit_seconds": 0.0,
        "reused_from": {
            "study": "companion_final",
            "row": "tlrn_cf_factors",
            "record": path.name,
            "ibnr": stem.rsplit("-ibnr", 1)[-1],
            "fit_seconds": record.get("run", {}).get("fit_seconds"),
            "data_hash": theirs.get("data_hash"),
            "run": record.get("run", {}),
        },
        "paid_only": False,
    }
    if theirs.get("data_hash") != mine.get("data_hash"):
        run["reused_from"]["data_hash_differs"] = True
    return reserves, run, tables


def run_study(
    out: Path,
    cohorts: dict,
    rows,
    draws: int,
    *,
    members: int,
    smoke: bool,
    version: str,
    fit_fn,
    save_entries: bool = True,
    log=print,
    study: str = "accident_year",
    reuse_dir: Path | None = None,
    first_fit_answer: str | None = None,
    budget_hours: float = BUDGET_HOURS,
) -> dict:
    """Fit what is missing, saving each fit the moment it finishes.

    ``fit_fn(cohort, row, roles, config, seed)`` returns ``(entry, run)`` where ``run``
    is a dict of what the fit ran on; for a study with extra tables it returns
    ``(entry, run, tables)`` with a frame for each table the study names. The record is
    written last. Returns ``{"fitted": n, "skipped": n, "failed": n, "reused": n}``.

    For a study with ``stop_rules`` (the fourth) and outside a smoke run: the saved seconds are
    checked against the 8 GPU hours before every fit, the main row's draw 0 on cohort C is
    checked against 25 minutes the moment it is saved (and again at the start of every later
    run in the folder, whichever cohorts that run asks for),
    a fit with a non-finite forecast is rerun once and then reported (:func:`fit_or_report`),
    and with ``reuse_dir`` the network-off row is taken from the third study's fits
    (:func:`reuse_factors_fit`). A fit with a ``FAILED_*.txt`` file has had its one rerun, so a
    later pass counts it as failed and does not fit it; deleting that file asks for a new try.
    Both stops ask Ethan, so both can take his answer: ``first_fit_answer`` (his words) lets the
    run go past the 25 minutes and ``budget_hours`` is the larger total he agreed to. Each is
    written into the record of every fit saved under it, as ``run["stop_answers"]``.
    """
    spec = get_study(study)
    todo, done = plan_study(
        out,
        cohorts,
        rows,
        draws,
        members=members,
        smoke=smoke,
        version=version,
        study=study,
        reuse_dir=reuse_dir,
    )
    for item in done:
        log(f"cohort {item['cohort']} {item['row']} draw {item['draw']}: already saved, skipped")
        if item["saved_data_hash"] != item["settings"]["data_hash"]:
            log(
                f"    its data hash is {item['saved_data_hash']}, this session's is "
                f"{item['settings']['data_hash']}; the evaluator reports the difference"
            )
    earlier_failures = 0
    if spec.stop_rules:
        to_fit = []
        for item in todo:
            path = failure_path(out, item)
            if path.exists():
                earlier_failures += 1
                log(
                    f"cohort {item['cohort']} {item['row']} draw {item['draw']}: reported as "
                    f"failed already, not fitted again (delete {path.name} to try again)"
                )
            else:
                to_fit.append(item)
        todo = to_fit
    log(f"{len(todo)} fit(s) to run, {len(done)} already saved")
    stops = spec.stop_rules and not smoke
    answers = {}
    if stops and first_fit_answer:
        answers["first_fit"] = first_fit_answer
    if stops and budget_hours != BUDGET_HOURS:
        answers["budget_hours"] = budget_hours
    if stops:
        # from the folder, not from this run's plan: a run of cohort B alone is still stopped
        for record in collect_records(out, smoke=False, study=study).values():
            if is_first_fit(record["settings"]):
                seconds = float(record.get("run", {}).get("fit_seconds", 0.0))
                check_first_fit(seconds, answer=first_fit_answer)
    counts = {"failed": 0, "reused": 0}
    for number, item in enumerate(todo, 1):
        cohort = cohorts[item["cohort"]]
        seed = item["settings"]["seed"]
        if item["reuse"]:
            log(
                f"[{number}/{len(todo)}] cohort {item['cohort']} {item['row']} draw "
                f"{item['draw']}: taken from the third study's fit in {reuse_dir}"
            )
            frame, run, tables = reuse_factors_fit(item, reuse_dir)
            entry = None
            counts["reused"] += 1
        else:
            if stops:
                used = check_budget(fit_seconds_used(out, study=study), budget_hours)
                log(f"    GPU hours already used: {used:.2f} of {budget_hours:g}")
            log(
                f"[{number}/{len(todo)}] cohort {item['cohort']} {item['row']} draw "
                f"{item['draw']} (seed {seed}): {item['config'].ensemble_size} members, "
                f"{item['config'].max_epochs} epochs",
            )
            if spec.stop_rules:
                got = fit_or_report(item, fit_fn, cohort, spec, log=log)
                if got is None:
                    counts["failed"] += 1
                    continue
                entry, run, tables = got
            else:
                entry, run, *rest = fit_fn(cohort, item["row"], item["roles"], item["config"], seed)
                tables = rest[0] if rest else {}
            frame = reserves_frame(entry)
        if set(tables) != set(spec.tables):
            raise ValueError(
                f"the {spec.name} study saves the tables {list(spec.tables)} with each fit but "
                f"fit_fn returned {sorted(tables)}"
            )
        if answers:
            run = {**run, "stop_answers": answers}
        if save_entries and entry is not None:
            save_entry(entry, item["pkl"])
        save_reserves(frame, item["reserves"])
        for name, table in tables.items():
            save_reserves(table, item["tables"][name])
        save_record(item["record"], item["settings"], run)  # last: it marks the fit finished
        write_manifest(out, smoke=smoke, study=study)
        log(f"    saved {item['record'].stem} ({run.get('fit_seconds', '?')} s)")
        board = frame[frame["company_code"].isin(cohort.board)].set_index("company_code")
        actual = cohort.actual.loc[board.index, "actual_reserve"]
        pool = float(np.abs(board["reserve"] - actual).sum() / actual.sum())
        log(f"    this draw's Pool_APE on the {len(board)} scored companies: {pool:.4f}")
        if stops and is_first_fit(item):
            check_first_fit(float(run.get("fit_seconds", 0.0)), answer=first_fit_answer)
    summary = {"fitted": len(todo) - counts["failed"] - counts["reused"], "skipped": len(done)}
    counts["failed"] += earlier_failures
    # the earlier studies' summary keeps exactly its two keys
    return {**summary, **counts} if spec.stop_rules else summary


def library_versions() -> dict:
    """The versions of the libraries whose differences could change a data hash or a number.

    The third study's data hashes differed between Colab and the laptop and the record held no
    version of pandas, numpy or duckdb to say why; the fourth study's records name them.
    """
    import duckdb

    return {"pandas": pd.__version__, "numpy": np.__version__, "duckdb": duckdb.__version__}


def prefit_checks(cohorts: dict, *, log=print) -> dict:
    """The pre-registered checks that fit nothing and can be made on the runtime itself.

    * the pairs selected at each valuation date equal the pre-registration's counts, for every
      cohort asked for;
    * the revised network has 18,153 parameters and no per-count table, the third study's
      18,537 with it, and the same seed gives both the same starting value for every
      parameter they share (the table's draws are still consumed);
    * the five channels the paid-only row zeroes are the incurred and case reserve ones.

    Raises ``SystemExit`` naming what differs; returns what was measured.
    """
    import torch

    from ibnr.gallery.nn.tlrn.network import TLRNNetwork
    from ibnr.kernels.nn_features import FEATURE_NAMES, N_FEATURES_PAID_ONLY

    found: dict = {"pairs": {}}
    for name, cohort in cohorts.items():
        sets = valuation_pairs(cohort)
        got = {v: len(sets[v]) for v in VALUATIONS}
        found["pairs"][name] = got
        log(f"prefit check: cohort {name} pairs by valuation date {got}")
        if got != EXPECTED_PAIRS_BY_DATE[name]:
            raise SystemExit(
                f"cohort {name}: pairs by valuation date {got} differ from the pre-registration's "
                f"{EXPECTED_PAIRS_BY_DATE[name]}"
            )

    def build(config):
        torch.manual_seed(11)
        return TLRNNetwork(
            config, n_lines=4, n_lag=10, n_feat=13, n_origin=10, head_init=torch.zeros(4, 10)
        )

    revised = build(tlrn_colab.companion_revised_config())
    final = build(tlrn_colab.companion_final_config())
    counts = {
        "revised": sum(p.numel() for p in revised.parameters()),
        "final": sum(p.numel() for p in final.parameters()),
    }
    found["parameters"] = counts
    log(f"prefit check: parameters per network {counts}")
    if counts != EXPECTED_PARAMETERS:
        raise SystemExit(f"parameters per network {counts}, expected {EXPECTED_PARAMETERS}")
    if "nobs_emb.weight" in revised.state_dict() or "nobs_emb.weight" not in final.state_dict():
        raise SystemExit("the per-count table is in the revised network or missing from the final")
    final_state = final.state_dict()
    unequal = [k for k, v in revised.state_dict().items() if not torch.equal(v, final_state[k])]
    if unequal:
        raise SystemExit(f"shared parameters start at other values for one seed: {unequal[:3]}")
    zeroed = FEATURE_NAMES[N_FEATURES_PAID_ONLY:]
    found["paid_only_channels"] = list(zeroed)
    log(f"prefit check: the paid-only row zeroes {list(zeroed)}")
    if len(zeroed) != 5 or not all(("incurred" in n or "case" in n) for n in zeroed):
        raise SystemExit(
            f"the paid-only channels are not the five incurred and case ones: {zeroed}"
        )
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="folder for the saved fits")
    parser.add_argument(
        "--study",
        choices=sorted(STUDIES),
        default="accident_year",
        help="accident_year: the first study; factor_head: the second (cohorts A, B and C); "
        "companion_final: the third (cohorts B and C); companion_revised: the fourth "
        "(cohorts B and C)",
    )
    parser.add_argument(
        "--cohort",
        choices=(*ALL_COHORTS, "both", "all"),
        default=None,
        help="A, B or C; both is A and B; all (the default) is every cohort of the study",
    )
    parser.add_argument(
        "--rows", default=None, help="comma list of the study's rows (default: both of them)"
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="draws 0 to N-1")
    parser.add_argument("--members", type=int, default=None, help="20 full, 2 with --smoke")
    parser.add_argument("--cas-dir", type=Path, default=None, help="the four CAS files, cohort B")
    parser.add_argument("--smoke", action="store_true", help="ten epochs, to check the setup")
    parser.add_argument(
        "--no-entries", action="store_true", help="do not pickle the fitted entries"
    )
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0: leave as is)")
    parser.add_argument("--quiet", action="store_true", help="no progress line every 100 epochs")
    parser.add_argument(
        "--reuse-factors-from",
        type=Path,
        default=None,
        help="the fourth study only: take the network-off row from the third study's fits in "
        "this folder instead of training it (allowed by the identity check in the "
        "pre-registration's amendment); ignored by --smoke, which trains every row",
    )
    parser.add_argument(
        "--answered-first-fit",
        default=None,
        metavar="ANSWER",
        help="the fourth study only: Ethan's answer to the 25-minute stop, in his words; the run "
        "goes past that stop and the answer is written into every fit record saved after it",
    )
    parser.add_argument(
        "--answered-budget-hours",
        type=float,
        default=None,
        metavar="HOURS",
        help="the fourth study only: the total GPU hours Ethan agreed to after the budget stop; "
        "the run stops there instead, and the number goes into every fit record saved after it",
    )
    args = parser.parse_args(argv)

    spec = get_study(args.study)
    wanted = args.rows or ",".join(spec.rows)
    rows = list(dict.fromkeys(r.strip() for r in wanted.split(",") if r.strip()))
    unknown = [r for r in rows if r not in spec.rows]
    if unknown or not rows:
        parser.error(f"--rows must be a comma list from {list(spec.rows)}, got {args.rows!r}")
    if args.draws < 1:
        parser.error("--draws must be at least 1")
    if args.reuse_factors_from is not None and spec.version != COMPANION_REVISED_VERSION:
        parser.error("--reuse-factors-from is for --study companion_revised")
    answered = args.answered_first_fit is not None or args.answered_budget_hours is not None
    if answered and not spec.stop_rules:
        parser.error("--answered-first-fit and --answered-budget-hours are for a study with stops")
    if args.answered_first_fit is not None and not args.answered_first_fit.strip():
        parser.error("--answered-first-fit needs Ethan's answer as its text")
    if args.answered_budget_hours is not None and args.answered_budget_hours <= BUDGET_HOURS:
        parser.error(f"--answered-budget-hours must be above the registered {BUDGET_HOURS:g}")
    members = args.members or (SMOKE_MEMBERS if args.smoke else DEFAULT_MEMBERS)
    choice = args.cohort or "all"
    names = {"both": ["A", "B"], "all": list(spec.cohorts)}.get(choice, [choice])
    outside = [n for n in names if n not in spec.cohorts]
    if outside:
        parser.error(
            f"cohort {', '.join(outside)} is not part of the {spec.name} study, whose cohorts "
            f"are {list(spec.cohorts)}"
        )
    if args.cas_dir is not None and "B" not in names:
        parser.error("--cas-dir is for cohort B; add --cohort B or --cohort both")
    if args.rows:
        try:
            check_row_request(spec, rows, names, args.draws)
        except ValueError as error:
            parser.error(str(error))

    import jax
    import torch

    import ibnr
    from ibnr import gallery

    if args.threads:
        torch.set_num_threads(args.threads)
    print("ibnr", ibnr.__version__, "| jax", jax.__version__, "| backend", jax.default_backend())
    print("devices", jax.devices(), flush=True)
    if jax.default_backend() == "cpu" and not args.smoke:
        print("warning: jax sees no accelerator; on a CPU the torch backend is faster")

    t0 = time.perf_counter()
    # accident years 1998 to 2007 serve A and B; cohort C has its own cells (1988 to 1997)
    recent = any(n in ("A", "B") for n in names)
    cells = cells_from_mart() if (recent and ("A" in names or args.cas_dir is None)) else None
    cohorts = {}
    for name in names:
        shared = cells_from_mart(years=YEARS["C"]) if name == "C" else cells
        cohorts[name] = build_cohort(
            name, cas_dir=args.cas_dir if name == "B" else None, cells=shared
        )
        print(cohorts[name].description, flush=True)
    print(f"cohorts built in {time.perf_counter() - t0:.0f}s")
    if spec.stop_rules:
        prefit_checks(cohorts)

    device = jax.devices()[0]

    def fit_fn(cohort, row, roles, config, seed):
        started = time.perf_counter()
        runs = []
        if spec.per_valuation:
            # the third study: a protocol per valuation date, each on its own companies
            entry = fit_valuations(
                cohort,
                config,
                seed,
                backend="jax",
                show_progress=not args.quiet,
                paid_only=bool(roles.get("paid_only", False)),
            )
        else:
            # the second study keeps every valuation date's members while the fit runs
            collecting = capture_protocols() if spec.tables else nullcontext([])
            with collecting as runs:
                entry = gallery.fit(
                    "tlrn",
                    cohort.triangle,
                    loss_field=tlrn_colab.FIELD,
                    as_of=cohort.as_of,
                    seed=seed,
                    config=config,
                    backend="jax",
                    show_progress=not args.quiet,
                    **roles,
                )
        run = {
            "ibnr": ibnr.__version__,
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "device_kind": getattr(device, "device_kind", None),
            "fit_seconds": round(time.perf_counter() - started, 1),
            "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "cohort_source": cohort.source,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **library_versions(),
            "paid_only": bool(roles.get("paid_only", False)),
        }
        if spec.per_valuation:
            run["pairs_by_valuation"] = {str(v): len(p) for v, p in entry.pairs.items()}
            return entry, run, entry.tables
        if spec.tables:
            return entry, run, valuation_tables(entry, runs)
        return entry, run

    summary = run_study(
        args.out,
        cohorts,
        rows,
        args.draws,
        members=members,
        smoke=args.smoke,
        version=ibnr.__version__,
        fit_fn=fit_fn,
        save_entries=not args.no_entries,
        study=args.study,
        reuse_dir=None if args.smoke else args.reuse_factors_from,
        first_fit_answer=args.answered_first_fit,
        budget_hours=args.answered_budget_hours or BUDGET_HOURS,
    )
    write_manifest(args.out, smoke=args.smoke, study=args.study)
    extra = ""
    if spec.stop_rules:
        extra = f", {summary['reused']} taken from the third study, {summary['failed']} failed"
    print(
        f"done: {summary['fitted']} fitted, {summary['skipped']} already saved{extra}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
