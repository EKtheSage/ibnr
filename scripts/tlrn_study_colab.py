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
  validation errors and the blend weight ``alpha``.

The valuation tables need each retrained valuation date's members, which the fitted entry does
not keep (only the final fit's). The runner takes them by wrapping the entry's own
``_finish_protocol`` while ``fit`` runs (:func:`capture_protocols`), so no library code changes;
:func:`valuation_tables` checks them against the entry before anything is saved.

Nothing here has been run on a GPU: the real results do not exist yet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tlrn_colab

#: the ibnr release both studies are run with on Colab. The install cells of the Colab kits and
#: the how-to documents pin exactly this version, and a test holds them equal to it
PINNED_IBNR = "0.7.4"
#: bump when the study's definition changes, so old fits are never reused under a new one
STUDY_VERSION = 1
COHORTS = ("A", "B")
ROWS = ("tlrn_8_ay", "tlrn_13_ay")
#: the second study (the factor head). Its records carry study 2 and the name, so the first
#: study's records, which carry neither the name nor a 2, are never read as its own
FACTOR_HEAD_VERSION = 2
FACTOR_HEAD_COHORTS = ("A", "B", "C")
FACTOR_HEAD_ROWS = ("tlrn_13_fh", "tlrn_13_fh_factors")
#: every cohort a runner can load (the first study uses A and B only)
ALL_COHORTS = ("A", "B", "C")
DEFAULT_DRAWS = 10
DEFAULT_MEMBERS = 20
#: smoke runs keep the ensemble this small unless --members says otherwise
SMOKE_MEMBERS = 2
#: draw k trains with seed SEED_FIT + SEED_STRIDE * k. Members are seeded seed + 1000 m
#: and the retrained valuation dates seed + 10000 v, all below 100,000 apart, so no two
#: draws share a stream.
SEED_STRIDE = 100_003

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
    settings = {
        "study": spec.version,
        "cohort": cohort.name,
        "row": row,
        "draw": draw,
        "seed": draw_seed(draw),
        "smoke": smoke,
        "config": asdict(config),
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
    pkl: Path, record: Path, reserves: Path, settings: dict, tables: dict | None = None
) -> dict | None:
    """The record of a finished fit made with exactly ``settings``, or None if there is none.

    A fit is finished when its record exists, because the record is written last. A saved
    fit made with OTHER settings is refused, not reused and not overwritten; so is a
    record whose reserves file (or any of the study's extra ``tables``) is gone, and files
    left with no record (an interruption between the writes), because nothing then says
    what they were fitted with.
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
    differ = settings_differences(settings, saved.get("settings", {}))
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
    columns when every member is kept, which is the accident-year variant.
    """
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
):
    """What to fit and what is already done, checked before any training.

    Returns ``(todo, done)``. Both are lists of dicts in draw-major order (all rows and
    cohorts of draw 0, then of draw 1, ...), so a run stopped at any point holds an
    even set of fits. A saved fit with other settings raises ``SystemExit`` here.
    """
    spec = get_study(study)
    configs = {row: (roles, cfg) for row, roles, cfg in row_configs(members, smoke, study)}
    rows = list(dict.fromkeys(rows))  # a repeated row would train the same fit twice
    todo, done = [], []
    for draw in range(draws):
        for name, cohort in cohorts.items():
            for row in rows:
                roles, cfg = configs[row]
                settings = fit_settings(
                    cohort, row, roles, cfg, draw=draw, smoke=smoke, study=study
                )
                pkl, record, reserves = fit_paths(
                    out, name, row, draw, smoke=smoke, version=version
                )
                tables = table_paths(record, spec.tables)
                saved = saved_fit(pkl, record, reserves, settings, tables)
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
                }
                (done if saved is not None else todo).append(item)
    return todo, done


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
) -> dict:
    """Fit what is missing, saving each fit the moment it finishes.

    ``fit_fn(cohort, row, roles, config, seed)`` returns ``(entry, run)`` where ``run``
    is a dict of what the fit ran on; for a study with extra tables it returns
    ``(entry, run, tables)`` with a frame for each table the study names. The record is
    written last. Returns ``{"fitted": n, "skipped": n}``.
    """
    spec = get_study(study)
    todo, done = plan_study(
        out, cohorts, rows, draws, members=members, smoke=smoke, version=version, study=study
    )
    for item in done:
        log(f"cohort {item['cohort']} {item['row']} draw {item['draw']}: already saved, skipped")
    log(f"{len(todo)} fit(s) to run, {len(done)} already saved")
    for number, item in enumerate(todo, 1):
        cohort = cohorts[item["cohort"]]
        seed = item["settings"]["seed"]
        log(
            f"[{number}/{len(todo)}] cohort {item['cohort']} {item['row']} draw {item['draw']} "
            f"(seed {seed}): {item['config'].ensemble_size} members, "
            f"{item['config'].max_epochs} epochs",
        )
        entry, run, *rest = fit_fn(cohort, item["row"], item["roles"], item["config"], seed)
        tables = rest[0] if rest else {}
        if set(tables) != set(spec.tables):
            raise ValueError(
                f"the {spec.name} study saves the tables {list(spec.tables)} with each fit but "
                f"fit_fn returned {sorted(tables)}"
            )
        frame = reserves_frame(entry)
        if save_entries:
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
    return {"fitted": len(todo), "skipped": len(done)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="folder for the saved fits")
    parser.add_argument(
        "--study",
        choices=sorted(STUDIES),
        default="accident_year",
        help="accident_year: the first study; factor_head: the second (cohorts A, B and C)",
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
    args = parser.parse_args(argv)

    spec = get_study(args.study)
    wanted = args.rows or ",".join(spec.rows)
    rows = list(dict.fromkeys(r.strip() for r in wanted.split(",") if r.strip()))
    unknown = [r for r in rows if r not in spec.rows]
    if unknown or not rows:
        parser.error(f"--rows must be a comma list from {list(spec.rows)}, got {args.rows!r}")
    if args.draws < 1:
        parser.error("--draws must be at least 1")
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

    device = jax.devices()[0]

    def fit_fn(cohort, row, roles, config, seed):
        started = time.perf_counter()
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
        }
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
    )
    write_manifest(args.out, smoke=args.smoke, study=args.study)
    print(f"done: {summary['fitted']} fitted, {summary['skipped']} already saved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
