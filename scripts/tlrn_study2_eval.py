"""The second tlrn study's evaluator: does the factor-head model beat the benchmarks?

Reads the fits that ``scripts/tlrn_study_colab.py --study factor_head`` saved and scores
them at four valuation dates against the chain ladder (``mack``) and the multivariate chain
ladder (``mcl``), both fitted by ibnr's own entries on exactly the data known at each date
and scored on exactly the cells the transformer is scored on. It needs no jax and no
accelerator, only ibnr, the cas-schedule-p package and the saved folder. The design and its
reasons are in ``knowledge/decisions/tlrn-factor-head-study.md``, which this docstring
repeats; the first study's evaluator is ``scripts/tlrn_study_eval.py`` and its statistics
are reused here.

PRE-REGISTERED, written before any evaluator code and before any real fit existed
=================================================================================

The question. Is the companion study's newest model (13 features, line and lag attention,
a factor head, blended with the multivariate chain ladder, 20 members averaged) more
accurate than ``mack`` and ``mcl``? The model is the row ``tlrn_13_fh``. Its comparison
rows are ``tlrn_13_fh_factors`` (the same model with the network switched off, so the head's
pooled factors and the alpha blend only) and the raw network (``predicted_net`` of the main
fits, the forecast before the blend, which needs no extra fit). The first study's
accident-year rows are compared when their fits are supplied.

Draws and cohorts. Each row has 10 independent fits (draws 0 to 9, fit seed
``11 + 100003 k``) on each cohort. Cohort A is notebook 04's (93 companies, 243 pairs, 82
scored), cohort B the companion's (84 companies, inside A), cohort C new (accident years
1988 to 1997, valued at 1997-12-31, real later development from the mart). A draw set that
is not exactly draws 0 to 9 is labelled provisional.

What is scored. Four valuation dates, called 6, 7, 8 and 10 as the companion calls them:
a valuation ``v`` is the calendar diagonal ``v`` of the 10-by-10 square (10 is the last
known one, 2007-12-31 on A and B, 1997-12-31 on C). At ``v`` the cells scored are those
on diagonals ``v + 1`` to 10 at development lag ``v`` or less, for origins that have at
least one elapsed development period; at ``v = 10`` they are every cell past diagonal 10
(45 per pair). A company's predicted and actual values at ``v`` are the sums of the
incremental paid over those cells, over its lines, so at ``v = 10`` the actual is the
cohort's realised reserve. Actuals at ``v < 10`` come from the known upper triangle.
Cell counts per company are checked against the saved tables' ``n_cells`` column, and the
saved tables' predictions and actuals against each other (the saved ``actual`` against the
evaluator's own, which is built from the cells) before any score is computed. The scored
companies are the cohort's board (those ``mack`` and ``mcl`` can fit in notebook 04's
rule) for which both benchmarks answered at all four valuations; a company a benchmark
refuses at any valuation is left out of every valuation and every row and named in the
output, so every comparison is on one set of companies.

Statistics. A company's error at ``v`` is predicted minus actual. **Pool_APE** is the sum of
absolute errors over the sum of actuals (``ibnr.kernels.point_metrics``'s ``pool_ape``).
The **draw average** at ``v`` is the mean over draws of each company's prediction. The
**rolling Pool_APE** is the plain mean of Pool_APE over the four valuations. The
**dominant company** is the company with the largest actual reserve at valuation 10, found
in code. The **bootstrap** resamples the scored companies 10,000 times (seed 2008), takes
the same resampled companies for the model and the benchmark and for every valuation (a
company's four valuation dates stay together), and reports the 2.5th and 97.5th percentiles
of the difference model minus benchmark. A resample whose actuals add to zero has no
Pool_APE and is left out.

The verdict, for the main row ``tlrn_13_fh`` only, on each cohort, from the draw average.

* ``effective`` only if ALL hold. (i) At valuation 10 the Pool_APE difference (model minus
  benchmark) has a 95% interval wholly below zero against both ``mack`` and ``mcl``.
  (ii) The mean over valuations 6, 7, 8 and 10 of the Pool_APE difference has an interval
  wholly below zero against both. (iii) The median over the draws of the valuation-10
  Pool_APE is below both benchmarks' valuation-10 Pool_APE. (iv) With the dominant company
  left out, the valuation-10 difference interval is not wholly above zero against either
  benchmark.
* ``worse`` if the interval in (i) or (ii) is wholly above zero against either benchmark.
* ``not distinguishable`` otherwise.

"Wholly below zero" means the upper end is below zero; "wholly above" means the lower end is
above zero. No multiplicity correction is made over three cohorts. A and B are not
independent (B lies inside A), and C shares most of its companies with both.

What a verdict means. The companion chose its design and hyperparameters with the A and B
test data in view, so the verdict on A or B is labelled **confirmatory of a fixed
specification, not out-of-sample**. The verdict on C is the only out-of-sample claim, with
two caveats beside it: C is the older accident-year window, and it shares most companies
with A and B. Nothing here shows what happens in other years.

Reported, not in the verdict (fixed in advance). Per valuation and for the rolling mean:
Pool_APE and the Pool_APE of each draw (min, median, max, sd); Pool_PE (bias: the sum of
errors over the sum of actuals); the median absolute percentage error over companies with a
nonzero actual (MedAPE); the mean absolute error (MAE); the mean absolute log of predicted
over actual reserve over companies where both are positive (the count is shown); the share
of companies where the model's absolute error is smaller than each benchmark's; Pool_APE
within quartiles of the actual reserve (quartiles of the valuation's own actuals, equal
counts, the dominant company in the top one); and the valuation-10 difference with the
dominant company left out, with its interval. Descriptive only, because the companion's
claimed gains are largest on them and they are driven by tiny reserves and by one company:
the mean absolute percentage error (MAPE) and the reserve-weighted root mean squared error
(wRMSE, the square root of the actual-weighted mean squared error, the companion's
definition). Ablations are reported as the difference of each comparison row's Pool_APE
from the main row's, with the same bootstrap, outside the verdict.

Tie-outs. The benchmarks at valuation 10 must equal the first study's benchmark reserves
(``tlrn_study_eval.benchmark_reserves``, run when ``--tie-out-first-study`` is given) and,
on cohort A, notebook 04's published Pool_APE to 1e-6. With ``--companion-rolling`` the
companion's published rolling table is read through its path: its cells and companies
columns must equal this evaluator's, and its chain ladder and chain ladder + multivariate
chain ladder Pool_APE at each valuation are printed beside ``mack`` and ``mcl``. That
comparison is a tie-out of the benchmarks, and a difference is reported, not hidden: his
chain ladder is his own implementation.

Usage::

    python scripts/tlrn_study2_eval.py --fits DIR [--cohort A|B|C|all] [--out DIR]
        [--cache DIR] [--cas-dir DIR] [--companion-rolling FILE]
        [--first-study-fits DIR] [--tie-out-first-study] [--smoke]

A smoke folder (``--smoke``) is scored too, so the evaluator can be checked, and its output
carries a banner saying the numbers must never be quoted. Nothing has run on a GPU when this
file is written, and no result exists.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tlrn_colab
import tlrn_study_colab as study
import tlrn_study_eval as first

AMENDMENT_1 = """Amendment 1 (2026-10-15, before any fit of this study had been run). It adds a
second question and changes nothing in the rules above; it is repeated from
``knowledge/decisions/tlrn-factor-head-study.md``.

The question. Does the network add anything beyond shrinking toward the multivariate chain
ladder? The main row ``tlrn_13_fh`` is compared with the ablation row ``tlrn_13_fh_factors``
(same cohort, same scored companies, same draws, both blended with the multivariate chain
ladder). The statistic is the paired company-level bootstrap of the draw-average difference in
Pool_APE, model minus ablation, with the primary verdict's resample (10,000, seed 2008, 2.5th
and 97.5th percentiles, a company's valuation dates kept together), at valuation 10 and as the
mean over valuations 6, 7, 8 and 10, and at valuation 10 with the dominant company left out.

``network adds value`` only if the valuation-10 interval AND the mean-over-valuations interval
are both wholly below zero AND the valuation-10 interval without the dominant company is not
wholly above zero. ``network hurts`` if either of the first two intervals is wholly above
zero. ``not distinguishable`` otherwise. The raw network against the ablation is reported,
descriptively. The verdict is reported beside the primary one and is not part of the primary
``effective`` rule.
"""

#: the four valuation dates, as the companion calls them (10 is the last known diagonal)
VALUATIONS = (6, 7, 8, 10)
LAST = max(VALUATIONS)
MAIN_ROW = "tlrn_13_fh"
FACTORS_ROW = "tlrn_13_fh_factors"
ROWS = (MAIN_ROW, FACTORS_ROW)
#: the first study's rows, scored at valuation 10 only when their fits are supplied
FIRST_STUDY_ROWS = ("tlrn_8_ay", "tlrn_13_ay")
BENCHMARKS = ("mack", "mcl")
RAW_NETWORK = "raw_network"
N_BOOT = first.N_BOOT
BOOT_SEED = first.BOOT_SEED
LEVEL = first.LEVEL
VERDICTS = ("effective", "not distinguishable", "worse")
#: Amendment 1 (A2): the paired question 'does the network add anything beyond the blend?'
NETWORK_VERDICTS = ("network adds value", "network hurts", "not distinguishable")
#: how far a saved table may differ from the evaluator's own cell sums (the tables come
#: from float32 tensors)
TABLE_RTOL = 1e-4
TABLE_ATOL = 1e-3
#: what each cohort's verdict means
CONFIRMATORY = "confirmatory of a fixed specification, not out-of-sample"
OUT_OF_SAMPLE = (
    "out-of-sample in accident years only (not tuned on as far as we know, assumed not known; "
    "its companies and calendar years overlap A and B)"
)
STATUS = {"A": CONFIRMATORY, "B": CONFIRMATORY, "C": OUT_OF_SAMPLE}


# -- which cells are scored and what they are worth ------------------------------------


def scored_mask(v: int, n_w: int, n_d: int, last_known: int | None = None) -> np.ndarray:
    """``(n_w, n_d)`` booleans: the cells valuation ``v`` is scored on.

    A cell is scored when its origin had at least one elapsed development period at ``v``
    and its calendar diagonal (``origin + lag``, 1-based) is past ``v``. At the last known
    diagonal that is every later cell; earlier, the cells up to the last known diagonal
    and at development lag ``v`` or less (the lags the valuation has already seen the
    steps of). Indices are 0-based; this is the companion's scoring window, which the saved
    tables' ``n_cells`` column is checked against.
    """
    last_known = n_w if last_known is None else last_known
    w = np.arange(n_w)[:, None]
    d = np.arange(n_d)[None, :]
    calendar = w + d + 1
    has_history = (v - w) >= 1
    if v >= last_known:
        low, high, max_lag = v + 1, n_w + n_d - 1, n_d
    else:
        low, high, max_lag = v + 1, last_known, v
    return has_history & (calendar >= low) & (calendar <= high) & (d + 1 <= max_lag)


def increments(cum: np.ndarray) -> np.ndarray:
    """Incremental amounts from a cumulative grid along its last axis (NaN stays NaN)."""
    return np.diff(cum, axis=-1, prepend=0.0)


def paid_squares(cells: pd.DataFrame, pairs) -> tuple[dict, int, int, int]:
    """``({pair: (n_w, n_d) cumulative paid}, n_w, n_d, first accident year)``.

    A cell the source does not carry is NaN. Accident years and development ages are the
    cohort's own (10 by 10 here), indexed from 0.
    """
    chosen = cells[pd.MultiIndex.from_frame(cells[study.KEY]).isin(pairs)]
    ay0 = int(chosen["accident_year"].min())
    n_w = int(chosen["accident_year"].max()) - ay0 + 1
    n_d = int(chosen["dev_age"].max())
    squares = {}
    for pair, part in chosen.groupby(study.KEY):
        grid = np.full((n_w, n_d), np.nan)
        rows = part["accident_year"].to_numpy() - ay0
        grid[rows, part["dev_age"].to_numpy() - 1] = part["paid"].to_numpy()
        squares[pair] = grid
    return squares, n_w, n_d, ay0


def truth_table(cohort, valuations=VALUATIONS) -> pd.DataFrame:
    """Per (valuation, company): the cells scored and the actual paid over them.

    Built from the cohort's cells, which hold the later development, so the actual at the
    last known diagonal is the cohort's realised reserve; that equality is checked.
    """
    squares, n_w, n_d, _ = paid_squares(cohort.cells, cohort.pairs)
    rows = []
    for v in valuations:
        mask = scored_mask(v, n_w, n_d)
        per_company: dict[str, list] = {}
        for (company, _), cum in squares.items():
            inc = increments(cum)[mask]
            if not np.isfinite(inc).all():
                raise SystemExit(f"cohort {cohort.name}: a scored cell of {company} is missing")
            tally = per_company.setdefault(company, [0, 0.0])
            tally[0] += int(mask.sum())
            tally[1] += float(inc.sum())
        rows += [
            {"valuation": v, "company_code": c, "n_cells": n, "actual": a}
            for c, (n, a) in sorted(per_company.items())
        ]
    out = pd.DataFrame(rows)
    last = out[out["valuation"] == LAST].set_index("company_code")["actual"]
    want = cohort.actual["actual_reserve"].reindex(last.index)
    gap = float((last - want).abs().max())
    if gap > 1e-6 * max(1.0, float(want.abs().max())):
        raise SystemExit(
            f"cohort {cohort.name}: the actual over the cells scored at valuation {LAST} differs "
            f"from the cohort's realised reserve by up to {gap}"
        )
    return out


# -- the benchmarks at every valuation date --------------------------------------------


def valuation_date(ay0: int, v: int) -> dt.date:
    """The last day of the statement year of calendar diagonal ``v``."""
    return dt.date(ay0 + v - 1, 12, 31)


def mack_cell_grid(entry, n_w: int, n_d: int, ay0: int) -> np.ndarray:
    """The fitted chain ladder's completed cumulative grid in the cohort's (n_w, n_d) layout."""
    full = np.asarray(entry.fit_.full, dtype=float)
    origins = [p.year - ay0 for p in entry.fit_.origin_periods]
    out = np.full((n_w, n_d), np.nan)
    out[np.ix_(origins, range(full.shape[1]))] = full
    return out


def mcl_cell_grids(entry, n_w: int, n_d: int, ay0: int) -> dict[str, np.ndarray]:
    """The fitted multivariate chain ladder's completed cumulative grid for each line.

    Each origin's latest observed amount is rolled forward through the fitted matrices,
    keeping every step, which is what the entry's ``point`` does for the last one only; the
    last column is checked against it.
    """
    c = entry.contract_
    cum, seen = np.asarray(c["cum"], dtype=float), np.asarray(c["obs_mask"], dtype=bool)
    n_lob, n_o, n_t = cum.shape
    grid = np.where(seen, cum, np.nan)
    for w in range(n_o):
        d0 = int(np.nonzero(seen[0, w])[0][-1])
        state = cum[:, w, d0]
        for d in range(d0, n_t - 1):
            state = entry.transitions_[d]["B"] @ state
            grid[:, w, d + 1] = state
    ultimate = float(entry.point()["point"].iloc[-1])
    if not np.isclose(grid[:, :, -1].sum(), ultimate, rtol=1e-9, atol=1e-9):
        raise RuntimeError("the multivariate chain ladder's grid does not end at its own point")
    origins = [p.year - ay0 for p in c["origin_periods"]]
    out = {}
    for k, line in enumerate(c["lobs"]):
        full = np.full((n_w, n_d), np.nan)
        full[np.ix_(origins, range(n_t))] = grid[k]
        out[str(line)] = full
    return out


def benchmark_cells(cohort, valuations=VALUATIONS, *, log=print) -> pd.DataFrame:
    """``mack`` and ``mcl`` per (valuation, company) on exactly the scored cells.

    Fitted by ibnr's gallery entries as the first study fits them (``mack`` per
    company-line pair, ``mcl`` per company) but on the cells known at each valuation date
    (``as_of`` the end of that statement year), and read per cell instead of per reserve.
    A company either benchmark refuses, or that ``mack`` fits on only some of its lines, is
    NaN for it at that valuation. The frame also carries the evaluator's own ``actual`` and
    ``n_cells``.
    """
    from ibnr import gallery

    squares, n_w, n_d, ay0 = paid_squares(cohort.cells, cohort.pairs)
    tri = study.triangle_from_cells(cohort.cells, cohort.pairs)
    e = tri.expr
    truth = truth_table(cohort, valuations)
    lines_of: dict[str, list] = {}
    for company, line in cohort.pairs:
        lines_of.setdefault(company, []).append(line)
    fit_kwargs = dict(loss_field=tlrn_colab.FIELD)
    t0 = time.perf_counter()
    refused = {b: 0 for b in BENCHMARKS}
    rows = []
    board = set(cohort.board)
    for v in valuations:
        mask = scored_mask(v, n_w, n_d)
        as_of = valuation_date(ay0, v)
        mack = {}
        for company, line in cohort.pairs:
            if company not in board:
                continue
            sub = tri.filter((e.company_code == company) & (e.line_of_business == line))
            try:
                entry = gallery.fit(
                    "mack", sub, **fit_kwargs, as_of=as_of, sigma_rule="mack", heldout_n_draws=10
                )
                inc = increments(mack_cell_grid(entry, n_w, n_d, ay0))[mask]
                mack[(company, line)] = float(inc.sum()) if np.isfinite(inc).all() else np.nan
            except Exception:
                refused["mack"] += 1
        for company in cohort.board:
            pairs = [(company, ln) for ln in lines_of[company]]
            got = [mack.get(p, np.nan) for p in pairs]
            mack_total = float(sum(got)) if np.isfinite(got).all() else np.nan
            mcl_total = np.nan
            try:
                entry = gallery.fit(
                    "mcl", tri.filter(e.company_code == company), **fit_kwargs, as_of=as_of
                )
                grids = mcl_cell_grids(entry, n_w, n_d, ay0)
                if set(grids) == set(lines_of[company]):
                    inc = [increments(g)[mask] for g in grids.values()]
                    mcl_total = float(sum(i.sum() for i in inc))
            except Exception:
                refused["mcl"] += 1
            rows.append(
                {"valuation": v, "company_code": company, "mack": mack_total, "mcl": mcl_total}
            )
    log(f"  benchmarks fitted in {time.perf_counter() - t0:.0f}s; refusals {refused}")
    return pd.DataFrame(rows).merge(truth, on=["valuation", "company_code"], how="left")


def cached_benchmark_cells(cohort, cache_dir: Path | None, *, log=print) -> pd.DataFrame:
    """:func:`benchmark_cells`, kept in ``cache_dir`` so a rerun does not refit them.

    The file name carries the cohort and its data hash, so a cache made from other cells is
    never read.
    """
    if cache_dir is None:
        return benchmark_cells(cohort, log=log)
    path = Path(cache_dir) / f"benchmarks2_{cohort.name}_{cohort.data_hash}.csv"
    if path.exists():
        frame = pd.read_csv(path, dtype={"company_code": str})
        columns = {"valuation", "company_code", "mack", "mcl", "n_cells", "actual"}
        if (
            set(frame.columns) == columns
            and len(frame) == len(VALUATIONS) * len(cohort.board)
            and set(frame["company_code"]) == set(cohort.board)
            and set(frame["valuation"]) == set(VALUATIONS)
        ):
            log(f"  benchmarks read from {path}")
            return frame
    frame = benchmark_cells(cohort, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


# -- the saved fits --------------------------------------------------------------------

RECORD_NAME = re.compile(
    r"^(?P<cohort>[ABC])_(?P<row>tlrn_\w+?)_d(?P<draw>\d+)(?P<smoke>_smoke)?-ibnr(?P<version>.+)\.json$"
)


@dataclass
class Fit:
    """One saved fit of the second study: its settings and its four tables."""

    draw: int
    seed: int
    settings: dict
    run: dict
    reserves: pd.DataFrame
    valuations: pd.DataFrame
    backtest: pd.DataFrame
    members: pd.DataFrame


def load_fits(folder: Path, cohort: str, row: str, *, smoke: bool) -> list[Fit]:
    """Every finished fit of one cohort and row, in draw order, checked to be one study.

    A record counts only if it says it is the second study's (``study`` 2 and the name
    ``factor_head``), so a folder holding the first study's fits as well is read correctly.
    All fits must share every setting except the draw and its seed, each seed must be the
    one its draw implies, and every table must be there. Two ibnr versions of one draw are
    refused rather than one of them picked.
    """
    spec = study.get_study("factor_head")
    found: dict[int, tuple[Path, str]] = {}
    for path in sorted(Path(folder).glob(f"{cohort}_{row}_d*.json")):
        m = RECORD_NAME.match(path.name)
        if m is None or m["cohort"] != cohort or m["row"] != row or bool(m["smoke"]) != smoke:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        settings = record.get("settings", {})
        if settings.get("study") != spec.version or settings.get("study_name") != spec.name:
            continue
        draw = int(m["draw"])
        if draw in found:
            raise SystemExit(
                f"draw {draw} of cohort {cohort} {row} was saved by two ibnr versions "
                f"({found[draw][1]} and {m['version']}) in {folder}; keep one"
            )
        found[draw] = (path, m["version"])
    fits = []
    for draw in sorted(found):
        path, _ = found[draw]
        record = json.loads(path.read_text(encoding="utf-8"))
        settings = record["settings"]
        if settings["seed"] != study.draw_seed(settings["draw"]) or settings["draw"] != draw:
            raise SystemExit(f"{path.name}: the seed or draw in the record is not draw {draw}'s")
        reserves = path.with_name(path.stem + "_reserves.csv")
        tables = study.table_paths(path, spec.tables)
        for needed in (reserves, *tables.values()):
            if not needed.exists():
                raise SystemExit(f"{path.name} has no {needed.name}")
        read = {name: pd.read_csv(p, dtype={"company_code": str}) for name, p in tables.items()}
        frame = pd.read_csv(reserves, dtype={"company_code": str}).set_index("company_code")
        fits.append(
            Fit(
                draw,
                settings["seed"],
                settings,
                record.get("run", {}),
                frame,
                read["valuations"],
                read["backtest"],
                read["members"],
            )
        )
    if fits:
        reference = fits[0].settings
        for fit in fits[1:]:
            differ = study.settings_differences(
                {k: v for k, v in reference.items() if k not in ("draw", "seed")},
                {k: v for k, v in fit.settings.items() if k not in ("draw", "seed")},
            )
            if differ:
                raise SystemExit(
                    f"cohort {cohort} {row}: draw {fit.draw} was fitted with different settings "
                    f"from draw {fits[0].draw} (they differ in {', '.join(differ)}); a folder "
                    "must hold one study"
                )
    return fits


#: relative tolerance of the check that a fit saw the cohort's own anchors and premiums
PLUMBING_TOL = 1e-6


def check_fits_against_cohort(cohort, row: str, fits: list[Fit], *, strict: bool = False) -> str:
    """Refuse a fit made for another cohort; return a note when only the cell hash differs.

    The cohort hash (the pairs), the publish id, the study number and name and the
    valuation date must equal the evaluator's. The cell hash is the sha256 of the cells
    printed as text, so another machine can print the same numbers differently (last-bit
    float differences) and give another hash. When only it differs, every fit's anchor,
    premium, number of lines and company set are compared with the cohort's own, and any
    gap is refused by name; if all agree the fits are scored and the returned note says so.
    ``strict`` refuses instead.
    """
    from collections import Counter

    where = f"cohort {cohort.name} {row}"
    spec = study.get_study("factor_head")
    for fit in fits:
        s = fit.settings
        for key, want in (
            ("cohort_hash", cohort.pairs_hash),
            ("publish_id", tlrn_colab.PUBLISH),
            ("study", spec.version),
            ("study_name", spec.name),
            ("cohort", cohort.name),
            ("as_of", cohort.as_of.isoformat()),
        ):
            if s.get(key) != want:
                raise SystemExit(
                    f"{where} draw {fit.draw} was fitted on a different cohort or data from the "
                    f"one this evaluator built ({key} {s.get(key)} against {want})"
                )
    recorded = {f.settings["data_hash"] for f in fits}
    if recorded == {cohort.data_hash}:
        return ""
    shown = ", ".join(sorted(recorded))
    if strict:
        raise SystemExit(
            f"{where} was fitted on data with hash {shown}, but this evaluator built "
            f"{cohort.data_hash} (--strict-data-hash)"
        )
    want = cohort.actual
    lines = pd.Series(Counter(c for c, _ in cohort.pairs))
    for fit in fits:
        frame = fit.reserves
        if set(frame.index) != set(cohort.companies):
            extra = sorted(set(frame.index) ^ set(cohort.companies))
            raise SystemExit(
                f"{where} draw {fit.draw}: the fit's companies differ from the cohort's "
                f"({len(extra)} in only one of them, e.g. {extra[:5]})"
            )
        for col, ours in (("anchor", want["anchor"]), ("premium", want["premium"])):
            mine = ours.reindex(frame.index)
            keep = mine.notna()
            diff = (frame[col] - mine)[keep].abs()
            scale = PLUMBING_TOL * max(1.0, float(mine[keep].abs().max()))
            if len(diff) and float(diff.max()) > scale:
                bad = diff.idxmax()
                raise SystemExit(
                    f"{where} draw {fit.draw}: the data hash differs and the fit's {col} for "
                    f"company {bad} differs from the cohort's by {float(diff.max()):,.4f} "
                    f"(tolerance {scale:,.4f}); the fit was made on other data"
                )
        n_lines = lines.reindex(frame.index).fillna(0).astype(int)
        if not (frame["n_lines"].astype(int) == n_lines).all():
            bad = (frame["n_lines"].astype(int) != n_lines).idxmax()
            raise SystemExit(
                f"{where} draw {fit.draw}: the data hash differs and the fit has "
                f"{frame.loc[bad, 'n_lines']} lines for company {bad} against the cohort's "
                f"{n_lines[bad]}; the fit was made on other data"
            )
    return (
        f"{where}: data hash differs (fits recorded {shown}, evaluator built "
        f"{cohort.data_hash}); anchors, premiums, lines and companies of every fit equal the "
        "cohort's within tolerance, so the fits are scored"
    )


def check_tables_against_truth(cohort, row: str, fits: list[Fit], truth: pd.DataFrame) -> None:
    """Refuse a fit whose saved tables disagree with the cells this evaluator scores.

    Per company and valuation, the saved ``n_cells`` must equal the evaluator's count, the
    saved ``actual`` (known only before the last diagonal) must equal the evaluator's own
    sum of the observed increments, and the saved prediction at the last diagonal must equal
    the reserve the fit stored. A fit that fails was scored on other cells than these.
    """
    own = truth.set_index(["valuation", "company_code"])
    where = f"cohort {cohort.name} {row}"
    for fit in fits:
        tab = fit.valuations.set_index(["valuation", "company_code"])
        if set(tab.index.get_level_values("valuation")) != set(VALUATIONS):
            raise SystemExit(
                f"{where} draw {fit.draw}: the valuations table covers "
                f"{sorted(set(tab.index.get_level_values('valuation')))}, not {list(VALUATIONS)}"
            )
        mine = own.reindex(tab.index)
        if mine["n_cells"].isna().any():
            gone = tab.index[mine["n_cells"].isna()][:3].tolist()
            raise SystemExit(f"{where} draw {fit.draw}: companies the cohort lacks, e.g. {gone}")
        bad = tab["n_cells"].to_numpy() != mine["n_cells"].to_numpy()
        if bad.any():
            where_bad = tab.index[bad][0]
            raise SystemExit(
                f"{where} draw {fit.draw}: the fit scored {tab.loc[where_bad, 'n_cells']} cells "
                f"for (valuation, company) {where_bad} and this evaluator scores "
                f"{mine.loc[where_bad, 'n_cells']}; the scoring windows differ"
            )
        known = tab.index.get_level_values("valuation") != LAST
        saved, ours = tab["actual"].to_numpy()[known], mine["actual"].to_numpy()[known]
        off = ~np.isclose(saved, ours, rtol=TABLE_RTOL, atol=TABLE_ATOL)
        if off.any():
            at = tab.index[known][off][0]
            raise SystemExit(
                f"{where} draw {fit.draw}: the saved actual for {at} is "
                f"{tab.loc[at, 'actual']} and the cells give {mine.loc[at, 'actual']}"
            )
        last = tab.xs(LAST, level="valuation")["predicted"]
        stored = fit.reserves["reserve"].reindex(last.index)
        if not np.allclose(last, stored, rtol=TABLE_RTOL, atol=TABLE_ATOL):
            raise SystemExit(
                f"{where} draw {fit.draw}: the valuation-{LAST} prediction differs from the "
                "reserve the fit stored"
            )


def draw_arrays(fits: list[Fit], companies: list[str], column: str) -> np.ndarray:
    """``(draws, valuations, companies)`` of one column of the saved valuations tables."""
    out = np.full((len(fits), len(VALUATIONS), len(companies)), np.nan)
    for k, fit in enumerate(fits):
        tab = fit.valuations.set_index(["valuation", "company_code"])[column]
        for vi, v in enumerate(VALUATIONS):
            out[k, vi] = tab.xs(v, level="valuation").reindex(companies).to_numpy()
    if not np.isfinite(out).all():
        raise SystemExit(f"a fit's valuations table has no finite {column} for some company")
    return out


# -- statistics ------------------------------------------------------------------------

pool_ape = first.pool_ape


def pool_ape_by_valuation(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Pool_APE at each valuation: ``pred`` and ``actual`` are ``(valuations, companies)``."""
    return np.array([pool_ape(pred[i], actual[i]) for i in range(actual.shape[0])])


def rolling(values: np.ndarray, axis: int | None = None):
    """The plain mean over valuations of a per-valuation statistic.

    ``axis`` is the valuation axis of an array of resampled statistics; without it the
    mean of all the values is returned as a float.
    """
    mean = np.mean(values, axis=axis)
    return float(mean) if axis is None else mean


def resampled_pool_ape(pred: np.ndarray, actual: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``(valuations, resamples)``: Pool_APE of each resample at each valuation.

    One ``idx`` for every valuation, so a company's valuation dates stay together.
    """
    return np.vstack(
        [first.pool_ape_resampled(pred[i], actual[i], idx) for i in range(actual.shape[0])]
    )


def interval(diff: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    """The percentile interval of the bootstrap differences (NaN resamples left out)."""
    tail = (1.0 - level) / 2.0
    return float(np.nanquantile(diff, tail)), float(np.nanquantile(diff, 1.0 - tail))


def difference_summary(
    pred: np.ndarray, bench: np.ndarray, actual: np.ndarray, idx: np.ndarray
) -> dict:
    """Pool_APE of ``pred`` minus ``bench`` at valuation 10 and rolling, with intervals.

    ``pred``, ``bench`` and ``actual`` are ``(valuations, companies)``. Negative favours
    ``pred``. ``share_favour_*`` is the share of resamples where ``pred`` is better.
    """
    last = VALUATIONS.index(LAST)
    boot_p = resampled_pool_ape(pred, actual, idx)
    boot_b = resampled_pool_ape(bench, actual, idx)
    diff_v = boot_p - boot_b
    diff_roll = rolling(boot_p, axis=0) - rolling(boot_b, axis=0)
    point_v = pool_ape_by_valuation(pred, actual) - pool_ape_by_valuation(bench, actual)
    out = {}
    for name, point, boot in (
        ("last", point_v[last], diff_v[last]),
        ("rolling", rolling(point_v), diff_roll),
    ):
        lo, hi = interval(boot)
        finite = boot[np.isfinite(boot)]
        out[name] = {
            "diff": float(point),
            "low": lo,
            "high": hi,
            "share_favour": float(np.mean(finite < 0)) if finite.size else float("nan"),
        }
    out["by_valuation"] = {
        v: dict(zip(("low", "high"), interval(diff_v[i]), strict=True))
        | {"diff": float(point_v[i])}
        for i, v in enumerate(VALUATIONS)
    }
    return out


def decide(stats: dict) -> tuple[str, dict, str]:
    """The pre-registered verdict, the four conditions and the sentence that justifies it.

    ``stats`` has, for each benchmark ``b`` in ``BENCHMARKS``: ``last[b]`` and
    ``rolling[b]`` (``low``, ``high`` of the difference interval at valuation 10 and over
    the mean of the valuations), ``excluding[b]`` (the valuation-10 interval without the
    dominant company); ``median_draw_last`` (the median over draws of the valuation-10
    Pool_APE) and ``pool_last[b]`` (the benchmarks' valuation-10 Pool_APE).
    """

    def below(i):
        return bool(np.isfinite(i["high"]) and i["high"] < 0)

    def above(i):
        return bool(np.isfinite(i["low"]) and i["low"] > 0)

    conditions = {
        "i_last_below_both": all(below(stats["last"][b]) for b in BENCHMARKS),
        "ii_rolling_below_both": all(below(stats["rolling"][b]) for b in BENCHMARKS),
        "iii_median_draw_below_both": all(
            stats["median_draw_last"] < stats["pool_last"][b] for b in BENCHMARKS
        ),
        "iv_excluding_dominant_not_above": all(
            np.isfinite(stats["excluding"][b]["low"]) and not above(stats["excluding"][b])
            for b in BENCHMARKS
        ),
    }
    worse = any(above(stats["last"][b]) or above(stats["rolling"][b]) for b in BENCHMARKS)
    if all(conditions.values()):
        word = "effective"
    elif worse:
        word = "worse"
    else:
        word = "not distinguishable"
    parts = []
    for b in BENCHMARKS:
        a, r, x = stats["last"][b], stats["rolling"][b], stats["excluding"][b]
        parts.append(
            f"vs {b}: valuation 10 [{a['low']:+.4%}, {a['high']:+.4%}], rolling "
            f"[{r['low']:+.4%}, {r['high']:+.4%}], valuation 10 without the dominant company "
            f"[{x['low']:+.4%}, {x['high']:+.4%}]"
        )
    reason = (
        "; ".join(parts)
        + f"; median-draw valuation-10 Pool_APE {stats['median_draw_last']:.4%} against "
        + " and ".join(f"{b} {stats['pool_last'][b]:.4%}" for b in BENCHMARKS)
        + "; conditions "
        + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in conditions.items())
    )
    return word, conditions, reason


def point_metrics(pred, actual, *, bench=None) -> dict:
    """The reported metrics of one set of company reserves against the actuals.

    ``pool_ape`` and ``pool_pe`` (bias) are sums over sums; ``medape`` and ``mape`` use
    the companies with a nonzero actual; ``mae`` is the mean absolute error; ``log_ratio``
    is the mean absolute log of predicted over actual over companies where both are
    positive (``n_log`` of them); ``wrmse`` is the square root of the actual-weighted mean
    squared error. ``mape`` and ``wrmse`` are descriptive only. With ``bench`` the share of
    companies where the absolute error is smaller than the benchmark's is added.
    """
    p, a = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    err = p - a
    nonzero = a != 0
    ape = np.abs(err[nonzero]) / np.abs(a[nonzero])
    both = (a > 0) & (p > 0)
    out = {
        "n": int(a.size),
        "pool_ape": float(np.abs(err).sum() / a.sum()),
        "pool_pe": float(err.sum() / a.sum()),
        "medape": float(np.median(ape)) if ape.size else float("nan"),
        "mae": float(np.abs(err).mean()),
        "log_ratio": float(np.abs(np.log(p[both] / a[both])).mean()) if both.any() else np.nan,
        "n_log": int(both.sum()),
        "mape": float(ape.mean()) if ape.size else float("nan"),
        "wrmse": float(np.sqrt((a * err**2).sum() / a.sum())),
    }
    if bench is not None:
        b = np.asarray(bench, dtype=float)
        out["share_better_than_benchmark"] = float(np.mean(np.abs(err) < np.abs(b - a)))
    return out


def size_quartiles(actual: np.ndarray) -> np.ndarray:
    """Quartile number 1 to 4 of each company's actual reserve (equal counts, ties by order)."""
    order = np.argsort(np.argsort(np.asarray(actual, dtype=float), kind="stable"), kind="stable")
    return (order * 4 // len(order)) + 1


# -- one cohort ------------------------------------------------------------------------


def scored_companies(bench: pd.DataFrame, board: list[str]) -> tuple[list[str], dict]:
    """The board companies both benchmarks answered for at every valuation, and who is out.

    ``bench`` is :func:`benchmark_cells`' frame. A company a benchmark refuses at any
    valuation is left out of every valuation and every row, so one set of companies is
    compared throughout. ``left_out`` maps each to the benchmark and valuations that failed.
    """
    left_out = {}
    keep = []
    for company in board:
        part = bench[bench["company_code"] == company]
        failed = {
            b: sorted(part.loc[part[b].isna(), "valuation"].astype(int).tolist())
            for b in BENCHMARKS
            if part[b].isna().any()
        }
        if failed or len(part) != len(VALUATIONS):
            left_out[company] = failed or {"rows": len(part)}
        else:
            keep.append(company)
    return keep, left_out


def wide(bench: pd.DataFrame, companies: list[str], column: str) -> np.ndarray:
    """``(valuations, companies)`` of one column of the benchmark frame."""
    table = bench.pivot(index="valuation", columns="company_code", values=column)
    return table.loc[list(VALUATIONS), companies].to_numpy(dtype=float)


def draw_pool_ape(P: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """``(draws, valuations)``: Pool_APE of each draw at each valuation."""
    return np.array([pool_ape_by_valuation(P[k], actual) for k in range(P.shape[0])])


def last_difference(pred_last, ref_last, actual_last, idx: np.ndarray) -> dict:
    """Valuation-10 Pool_APE of ``pred_last`` minus ``ref_last`` with its bootstrap interval."""
    boot = first.pool_ape_resampled(pred_last, actual_last, idx) - first.pool_ape_resampled(
        ref_last, actual_last, idx
    )
    lo, hi = interval(boot)
    finite = boot[np.isfinite(boot)]
    return {
        "diff": pool_ape(pred_last, actual_last) - pool_ape(ref_last, actual_last),
        "low": lo,
        "high": hi,
        "share_favour": float(np.mean(finite < 0)) if finite.size else float("nan"),
    }


def decide_network(stats: dict) -> tuple[str, dict, str]:
    """Amendment 1's verdict on whether the network adds anything beyond the blend.

    ``stats`` has ``last``, ``rolling`` and ``excluding``, each ``{"low", "high"}``: the
    interval of the Pool_APE difference ``main row minus ablation row`` at valuation 10, as
    the mean over the valuations, and at valuation 10 without the dominant company.
    """

    def below(i):
        return bool(np.isfinite(i["high"]) and i["high"] < 0)

    def above(i):
        return bool(np.isfinite(i["low"]) and i["low"] > 0)

    conditions = {
        "a_last_below": below(stats["last"]),
        "b_rolling_below": below(stats["rolling"]),
        "c_excluding_not_above": bool(
            np.isfinite(stats["excluding"]["low"]) and not above(stats["excluding"])
        ),
    }
    if all(conditions.values()):
        word = "network adds value"
    elif above(stats["last"]) or above(stats["rolling"]):
        word = "network hurts"
    else:
        word = "not distinguishable"
    reason = (
        "; ".join(
            f"{name} [{stats[key]['low']:+.4%}, {stats[key]['high']:+.4%}]"
            for name, key in (
                ("valuation 10", "last"),
                ("rolling", "rolling"),
                ("valuation 10 without the dominant company", "excluding"),
            )
        )
        + "; conditions "
        + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in conditions.items())
    )
    return word, conditions, reason


def metric_table(series: dict, actual: np.ndarray, bench: dict) -> pd.DataFrame:
    """Every reported metric, per method and valuation, and their mean over valuations.

    ``series`` maps a method to its ``(valuations, companies)`` predictions (a draw average
    for a transformer row). The rolling row is the plain mean of the valuations' metrics.
    """
    rows = []
    for method, pred in series.items():
        per = []
        for i, v in enumerate(VALUATIONS):
            row = point_metrics(pred[i], actual[i])
            for b in BENCHMARKS:
                if method != b:
                    row[f"share_better_than_{b}"] = point_metrics(
                        pred[i], actual[i], bench=bench[b][i]
                    )["share_better_than_benchmark"]
            per.append({"method": method, "valuation": str(v), **row})
        rows += per
        keys = [k for k in per[0] if k not in ("method", "valuation", "n", "n_log")]
        mean = {k: float(np.mean([r[k] for r in per])) for k in keys}
        rows.append({"method": method, "valuation": "rolling", **mean})
    return pd.DataFrame(rows)


def quartile_table(series: dict, actual: np.ndarray) -> pd.DataFrame:
    """Pool_APE within quartiles of the actual reserve, per method and valuation."""
    rows = []
    for i, v in enumerate(VALUATIONS):
        q = size_quartiles(actual[i])
        for k in (1, 2, 3, 4):
            inside = q == k
            for method, pred in series.items():
                a = actual[i][inside]
                rows.append(
                    {
                        "valuation": v,
                        "quartile": k,
                        "n": int(inside.sum()),
                        "actual_reserve": float(a.sum()),
                        "method": method,
                        "pool_ape": float(np.abs(pred[i][inside] - a).sum() / a.sum())
                        if a.sum() != 0
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def alpha_summary(fits: list[Fit]) -> pd.DataFrame:
    """Median blend weight (alpha) over all members of all draws, per valuation."""
    members = pd.concat([f.members for f in fits], ignore_index=True)
    return members.groupby("valuation")["alpha"].median().rename("alpha_median").reset_index()


def network_verdict_fields(network_stats: dict | None, draw_sets: dict) -> dict:
    """Amendment 1's verdict, labelled like the primary one, as fields of a cohort's result.

    The draw sets of the two rows must both be exactly draws 0 to 9 and equal to each other,
    or the verdict is labelled provisional with what is missing.
    """
    if network_stats is None:
        return {
            "network_verdict": None,
            "network_label": f"not scored (needs fits of {MAIN_ROW} and {FACTORS_ROW})",
            "network_reason": f"no fits of both {MAIN_ROW} and {FACTORS_ROW} for this cohort",
            "network_provisional": True,
            "network_conditions": {},
            "network_statistics": None,
        }
    word, conditions, reason = decide_network(network_stats)
    main_draws, other_draws = draw_sets[MAIN_ROW], draw_sets[FACTORS_ROW]
    notes = []
    for row, draws in ((MAIN_ROW, main_draws), (FACTORS_ROW, other_draws)):
        provisional, missing = first.draw_completeness(draws)
        if provisional:
            notes.append(
                f"{row} has {len(draws)} of {study.DEFAULT_DRAWS} draws"
                + (f" (missing {', '.join(str(d) for d in missing)})" if missing else "")
            )
    if sorted(main_draws) != sorted(other_draws):
        notes.append("the two rows' draw sets differ")
    label = word
    if notes:
        note = "provisional: " + "; ".join(notes)
        reason = f"{note}; {reason}"
        label = f"{word} ({note})"
    return {
        "network_verdict": word,
        "network_label": label,
        "network_reason": reason,
        "network_provisional": bool(notes),
        "network_conditions": conditions,
        "network_statistics": network_stats,
    }


def evaluate_cohort(
    cohort,
    fits_by_row: dict[str, list[Fit]],
    bench: pd.DataFrame,
    *,
    first_study: dict[str, np.ndarray] | None = None,
) -> dict:
    """Everything the second study reports for one cohort, from fits and benchmarks.

    ``fits_by_row`` has the study's rows; ``bench`` is :func:`benchmark_cells`' frame (its
    ``actual`` and ``n_cells`` are the evaluator's own). ``first_study`` maps one of the
    first study's rows to its ``(companies,)`` valuation-10 reserves (the draw average) on
    the scored companies. The verdict is made for the main row alone, and only when it has
    fits.
    """
    companies, left_out = scored_companies(bench, list(cohort.board))
    if len(companies) < 3:
        raise SystemExit(f"cohort {cohort.name}: only {len(companies)} companies scored")
    actual = wide(bench, companies, "actual")
    bench_arrays = {b: wide(bench, companies, b) for b in BENCHMARKS}
    last = VALUATIONS.index(LAST)
    top, share = first.dominant_company(pd.Series(actual[last], index=companies))
    keep_ex = np.array([c != top for c in companies])
    idx = first.bootstrap_indices(len(companies))
    idx_ex = first.bootstrap_indices(int(keep_ex.sum()))

    draws, series, draw_sets = {}, {b: bench_arrays[b] for b in BENCHMARKS}, {}
    main_fits = fits_by_row.get(MAIN_ROW) or []
    if main_fits:
        draws[RAW_NETWORK] = draw_arrays(main_fits, companies, "predicted_net")
        series[RAW_NETWORK] = draws[RAW_NETWORK].mean(axis=0)
    for row in ROWS:
        fits = fits_by_row.get(row) or []
        if fits:
            draws[row] = draw_arrays(fits, companies, "predicted")
            series[row] = draws[row].mean(axis=0)
            draw_sets[row] = [f.draw for f in fits]
    if main_fits:
        draw_sets[RAW_NETWORK] = draw_sets[MAIN_ROW]

    result: dict = {
        "cohort": cohort.name,
        "status": STATUS.get(cohort.name, ""),
        "companies": companies,
        "left_out": left_out,
        "dominant_company": top,
        "dominant_share": share,
        "n_cells_last": int(bench.loc[bench["valuation"] == LAST, "n_cells"].sum()),
        "draw_sets": draw_sets,
        "actual": actual,
        "series": series,
        "draws": draws,
    }

    differences, rows_out = {}, []
    for method in (r for r in (MAIN_ROW, FACTORS_ROW, RAW_NETWORK) if r in series):
        for b in BENCHMARKS:
            d = difference_summary(series[method], bench_arrays[b], actual, idx)
            differences[(method, b)] = d
            for scope in ("last", "rolling"):
                rows_out.append({"method": method, "benchmark": b, "statistic": scope, **d[scope]})
            for v, cell in d["by_valuation"].items():
                rows_out.append(
                    {
                        "method": method,
                        "benchmark": b,
                        "statistic": f"valuation {v}",
                        **cell,
                        "share_favour": np.nan,
                    }
                )
    result["differences"] = pd.DataFrame(rows_out)

    excluding = {}
    for method in (r for r in (MAIN_ROW, FACTORS_ROW, RAW_NETWORK) if r in series):
        for b in BENCHMARKS:
            excluding[(method, b)] = last_difference(
                series[method][last][keep_ex],
                bench_arrays[b][last][keep_ex],
                actual[last][keep_ex],
                idx_ex,
            )
    result["excluding_dominant"] = pd.DataFrame(
        [{"method": m, "benchmark": b, **v} for (m, b), v in excluding.items()]
    )

    result["metrics"] = metric_table(series, actual, bench_arrays)
    result["quartiles"] = quartile_table(series, actual)
    pools = {b: pool_ape_by_valuation(bench_arrays[b], actual) for b in BENCHMARKS}
    result["benchmark_pool_ape"] = pools

    # the ablations: every comparison row against the main row, same bootstrap
    ablations = []
    if MAIN_ROW in series:
        for other in (r for r in (RAW_NETWORK, FACTORS_ROW) if r in series):
            d = difference_summary(series[other], series[MAIN_ROW], actual, idx)
            for scope in ("last", "rolling"):
                ablations.append(
                    {"row": other, "against": MAIN_ROW, "statistic": scope, **d[scope]}
                )
        for other, reserve in (first_study or {}).items():
            d = last_difference(reserve, series[MAIN_ROW][last], actual[last], idx)
            ablations.append({"row": other, "against": MAIN_ROW, "statistic": "last", **d})
    result["ablations"] = pd.DataFrame(ablations)

    # Amendment 1: the paired question, main row minus the factors-only row (not the verdict)
    network_rows, network_stats = [], None
    if MAIN_ROW in series and FACTORS_ROW in series:
        d = difference_summary(series[MAIN_ROW], series[FACTORS_ROW], actual, idx)
        ex = last_difference(
            series[MAIN_ROW][last][keep_ex],
            series[FACTORS_ROW][last][keep_ex],
            actual[last][keep_ex],
            idx_ex,
        )
        network_stats = {"last": d["last"], "rolling": d["rolling"], "excluding": ex}
        for scope, cell in (("last", d["last"]), ("rolling", d["rolling"]), ("excluding", ex)):
            network_rows.append(
                {"model": MAIN_ROW, "ablation": FACTORS_ROW, "statistic": scope, **cell}
            )
        if RAW_NETWORK in series:
            raw = difference_summary(series[RAW_NETWORK], series[FACTORS_ROW], actual, idx)
            raw_ex = last_difference(
                series[RAW_NETWORK][last][keep_ex],
                series[FACTORS_ROW][last][keep_ex],
                actual[last][keep_ex],
                idx_ex,
            )
            for scope, cell in (
                ("last", raw["last"]),
                ("rolling", raw["rolling"]),
                ("excluding", raw_ex),
            ):
                network_rows.append(
                    {"model": RAW_NETWORK, "ablation": FACTORS_ROW, "statistic": scope, **cell}
                )
    result["network_check"] = pd.DataFrame(network_rows)

    # the draws
    draw_rows = []
    summaries = {}
    for row, P in draws.items():
        per = draw_pool_ape(P, actual)
        summaries[row] = {
            **{str(v): first.draw_summary(per[:, i]) for i, v in enumerate(VALUATIONS)},
            "rolling": first.draw_summary(per.mean(axis=1)),
        }
        for k in range(per.shape[0]):
            draw_rows.append(
                {
                    "row": row,
                    "draw": draw_sets[row][k],
                    **{f"valuation_{v}": per[k, i] for i, v in enumerate(VALUATIONS)},
                    "rolling": per[k].mean(),
                }
            )
    result["draw_pool_ape"] = pd.DataFrame(draw_rows)
    result["draw_summary"] = summaries
    if main_fits:
        result["alpha"] = alpha_summary(main_fits)

    # the verdict
    if main_fits:
        per_main = draw_pool_ape(draws[MAIN_ROW], actual)
        stats = {
            "last": {b: differences[(MAIN_ROW, b)]["last"] for b in BENCHMARKS},
            "rolling": {b: differences[(MAIN_ROW, b)]["rolling"] for b in BENCHMARKS},
            "excluding": {b: excluding[(MAIN_ROW, b)] for b in BENCHMARKS},
            "median_draw_last": float(np.median(per_main[:, last])),
            "pool_last": {b: float(pools[b][last]) for b in BENCHMARKS},
        }
        word, conditions, reason = decide(stats)
        provisional, missing = first.draw_completeness(draw_sets[MAIN_ROW])
        label = word
        if provisional:
            note = f"provisional: {len(draw_sets[MAIN_ROW])} of {study.DEFAULT_DRAWS} draws"
            if missing:
                note += " (missing draws " + ", ".join(str(d) for d in missing) + ")"
            reason = f"{note}; {reason}"
            label = f"{word} ({note})"
        result.update(
            verdict=word,
            conditions=conditions,
            verdict_reason=reason,
            verdict_label=label,
            provisional=provisional,
            missing_draws=missing,
            n_draws=len(draw_sets[MAIN_ROW]),
            verdict_statistics=stats,
        )
    else:
        result.update(
            verdict=None,
            verdict_label="not scored (no finished fit of the main row)",
            verdict_reason=f"no fits of {MAIN_ROW} for this cohort",
            provisional=True,
            missing_draws=list(range(study.DEFAULT_DRAWS)),
            n_draws=0,
        )
    result.update(network_verdict_fields(network_stats, draw_sets))
    return result


def company_table(result: dict) -> pd.DataFrame:
    """One row per (valuation, company): the actual, both benchmarks and the draw averages."""
    rows = []
    for i, v in enumerate(VALUATIONS):
        frame = pd.DataFrame(
            {
                "valuation": v,
                "company_code": result["companies"],
                "actual": result["actual"][i],
            }
        )
        for method, pred in result["series"].items():
            frame[method] = pred[i]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


# -- tie-outs --------------------------------------------------------------------------


def companion_rolling_tie_out(path: Path, result: dict) -> pd.DataFrame:
    """The companion's published rolling table against this evaluator's benchmarks.

    His ``cells`` and ``companies`` columns must equal this evaluator's counts (cohort B
    only: it is his cohort), and his chain ladder and chain ladder + multivariate chain
    ladder Pool_APE are shown beside ``mack`` and ``mcl`` with the difference, which is
    reported, not required to vanish: his chain ladder is his own implementation. His
    transformer rows are shown beside the draws' averages for reading only.
    """
    table = pd.read_csv(path).set_index("valuation")
    rows = []
    for i, v in enumerate(VALUATIONS):
        if v not in table.index:
            raise SystemExit(f"{path} has no row for valuation {v}")
        theirs = table.loc[v]
        n_cells = int(result["n_cells_by_valuation"][v])
        if n_cells != int(theirs["cells"]) or len(result["companies"]) != int(theirs["companies"]):
            raise SystemExit(
                f"valuation {v}: this evaluator scores {n_cells} cells over "
                f"{len(result['companies'])} companies and the companion's table has "
                f"{int(theirs['cells'])} over {int(theirs['companies'])}; the scoring windows "
                "or the cohorts differ"
            )
        row = {
            "valuation": v,
            "companies_ours": len(result["companies"]),
            "companies_theirs": int(theirs["companies"]),
            "cells_ours": n_cells,
            "cells_theirs": int(theirs["cells"]),
        }
        for ours, name in (("mack", "Chain ladder"), ("mcl", "Chain ladder + MCL")):
            mine = pool_ape(result["series"][ours][i], result["actual"][i])
            row[f"{ours}_ours"] = mine
            row[f"{ours}_theirs"] = float(theirs[name])
            row[f"{ours}_difference"] = mine - float(theirs[name])
        for method, name in ((MAIN_ROW, "TLRN"), (RAW_NETWORK, "TLRN network only")):
            if method in result["series"]:
                row[f"{method}_draw_average"] = pool_ape(
                    result["series"][method][i], result["actual"][i]
                )
                row[f"{method}_theirs"] = float(theirs[name])
        if "alpha" in result and "alpha_median" in table.columns:
            ours_alpha = result["alpha"].set_index("valuation")["alpha_median"]
            row["alpha_median_ours"] = float(ours_alpha.get(v, np.nan))
            row["alpha_median_theirs"] = float(theirs["alpha_median"])
        rows.append(row)
    return pd.DataFrame(rows)


def reference_tie_out(cohort, result: dict) -> dict:
    """Notebook 04's Pool_APE of the two classical rows on cohort A, at valuation 10."""
    out = {}
    if cohort.name != "A" or cohort.pairs_hash != study.COHORT_HASH["A"]:
        return out
    last = VALUATIONS.index(LAST)
    if len(result["companies"]) != len(cohort.board):
        return {"note": "not compared: some board companies are left out"}
    for b, ref in first.REFERENCE_A.items():
        mine = pool_ape(result["series"][b][last], result["actual"][last])
        rel = abs(mine / ref - 1.0)
        out[b] = {"ours": mine, "notebook_04": ref, "relative": rel}
        if rel > first.REFERENCE_TOL:
            raise SystemExit(
                f"{b} does not reproduce notebook 04's Pool_APE on cohort A at valuation "
                f"{LAST}: {mine} against {ref}"
            )
    return out


def first_study_tie_out(cohort, result: dict, *, log=print) -> dict:
    """The valuation-10 benchmarks against the first study's own reserves (cohorts A and B).

    Runs the first evaluator's benchmark fits (about a minute) and compares ``mack`` and
    ``mcl`` company by company; the two evaluators must agree to rounding.
    """
    if cohort.as_of != tlrn_colab.AS_OF:
        return {"note": f"not applicable: cohort {cohort.name} is valued at {cohort.as_of}"}
    old = first.benchmark_reserves(cohort, log=log)
    last = VALUATIONS.index(LAST)
    companies = result["companies"]
    out = {}
    for b in BENCHMARKS:
        mine = result["series"][b][last]
        theirs = old[b].reindex(companies).to_numpy(dtype=float)
        out[b] = {"max_relative_difference": first.max_relative_difference(mine, theirs)}
    out["actual"] = {
        "max_relative_difference": first.max_relative_difference(
            result["actual"][last], old["actual"].reindex(companies).to_numpy(dtype=float)
        )
    }
    return out


# -- the report ------------------------------------------------------------------------


def fmt_pct(x: float) -> str:
    return f"{100 * x:.3f}%"


def fmt_extra(value) -> str:
    """A tie-out or note as plain text: nested dictionaries as ``key value`` lists."""
    if isinstance(value, dict):
        parts = [f"{key} {fmt_extra(item)}" for key, item in value.items()]
        if any(isinstance(item, dict) for item in value.values()):
            return "\n   " + "\n   ".join(parts)
        return "; ".join(parts)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render(cohort, result: dict, extras: dict, banner: str) -> str:
    """The text report of one cohort."""
    lines = []
    if banner:
        lines += [banner, ""]
    lines.append(f"=== {cohort.description}")
    lines.append(
        f"status of any verdict on this cohort: {result['status']}"
        f"; valuation {LAST} is {cohort.as_of}"
    )
    lines.append(
        f"scored companies: {len(result['companies'])} of the cohort's board "
        f"{len(cohort.board)}; cells scored at valuation {LAST}: {result['n_cells_last']}"
    )
    if result["left_out"]:
        lines.append(f"left out because a benchmark failed: {result['left_out']}")
    lines.append(
        f"dominant company {result['dominant_company']} holds {result['dominant_share']:.1%} of "
        f"the actual valuation-{LAST} reserve"
    )
    for key, value in extras.items():
        lines.append(f"{key}: {fmt_extra(value)}")
    pools = result["benchmark_pool_ape"]
    lines.append("benchmark Pool_APE by valuation " + str(list(VALUATIONS)) + ":")
    for b in BENCHMARKS:
        lines.append(
            f"   {b}: "
            + "  ".join(fmt_pct(x) for x in pools[b])
            + f"  | rolling {fmt_pct(pools[b].mean())}"
        )
    for row, summary in result["draw_summary"].items():
        provisional = ""
        d = result["draw_sets"][row]
        if first.draw_completeness(d)[0]:
            provisional = " (PROVISIONAL, not the registered 10 draws)"
        lines += ["", f"--- {row}: {len(d)} draws{provisional}"]
        for key, s in summary.items():
            lines.append(
                f"   Pool_APE per draw at valuation {key}: min {fmt_pct(s['min'])}  median "
                f"{fmt_pct(s['median'])}  max {fmt_pct(s['max'])}  sd "
                f"{fmt_pct(s['sd']) if s['n'] > 1 else 'n/a'}"
            )
    lines += ["", "Pool_APE of the draw average, and the other reported metrics:"]
    metrics = result["metrics"].round(5)
    lines.append(metrics.to_string(index=False))
    lines += [
        "",
        "Difference in Pool_APE (row minus benchmark; negative favours the row), 95% interval:",
    ]
    lines.append(result["differences"].round(5).to_string(index=False))
    lines += ["", "Valuation-10 difference without the dominant company:"]
    lines.append(result["excluding_dominant"].round(5).to_string(index=False))
    lines += ["", "Pool_APE within quartiles of the actual reserve:"]
    wide_q = result["quartiles"].pivot_table(
        index=["valuation", "quartile", "n"], columns="method", values="pool_ape"
    )
    lines.append(wide_q.round(5).to_string())
    if len(result["ablations"]):
        lines += ["", "Ablations (row minus the main row; positive means the main row is better):"]
        lines.append(result["ablations"].round(5).to_string(index=False))
    if "alpha" in result:
        lines += ["", "Median blend weight alpha over members and draws:"]
        lines.append(result["alpha"].round(4).to_string(index=False))
    lines += [
        "",
        f"VERDICT for {MAIN_ROW} on cohort {cohort.name} ({result['status']}): "
        f"{(result['verdict'] or 'not scored').upper()}",
        f"   {result['verdict_reason']}",
        "",
        "Amendment 1, a separate question (not part of the verdict above): does the network add "
        "anything beyond the blend? Pool_APE difference, model minus the factors-only row, "
        "negative favours the model:",
    ]
    if len(result["network_check"]):
        lines.append(result["network_check"].round(5).to_string(index=False))
    lines += [
        f"NETWORK QUESTION for {MAIN_ROW} on cohort {cohort.name}: "
        f"{(result['network_verdict'] or 'not scored').upper()}",
        f"   {result['network_reason']}",
    ]
    return "\n".join(lines)


def run_cohort(
    name: str,
    fits_dir: Path,
    *,
    smoke: bool = False,
    cells=None,
    cas_dir: Path | None = None,
    cache_dir: Path | None = None,
    strict: bool = False,
    first_study_dir: Path | None = None,
    companion_rolling: Path | None = None,
    tie_out_first: bool = False,
    rows=ROWS,
    log=print,
) -> tuple[object, dict] | None:
    """Load cohort ``name``, its fits and its benchmarks, and evaluate. None when no fit exists.

    ``cells`` are the mart's cells for this cohort's accident years (loaded when not given).
    The returned result carries ``extras`` (tie-outs and notes) for the report.
    """
    if cells is None:
        cells = study.cells_from_mart(years=study.YEARS[name])
    cohort = study.load_cohort(name, cas_dir=cas_dir if name == "B" else None, cells=cells)
    log(cohort.description)
    fits_by_row = {row: load_fits(fits_dir, name, row, smoke=smoke) for row in rows}
    if not any(fits_by_row.values()):
        log(f"cohort {name}: no finished fits in {fits_dir}, benchmarks not fitted, skipped")
        return None
    bench = cached_benchmark_cells(cohort, cache_dir, log=log)
    truth = truth_table(cohort)
    extras: dict = {}
    for row, fits in fits_by_row.items():
        if not fits:
            log(f"cohort {name} {row}: no finished fits in {fits_dir}")
            continue
        note = check_fits_against_cohort(cohort, row, fits, strict=strict)
        if note:
            log(note)
            extras[f"data hash, {row}"] = note
        check_tables_against_truth(cohort, row, fits, truth)
    first_rows = {}
    if first_study_dir is not None and name in study.COHORTS:
        companies, _ = scored_companies(bench, list(cohort.board))
        for row in FIRST_STUDY_ROWS:
            old = first.load_fits(first_study_dir, name, row, smoke=smoke)
            if old:
                matrix = first.reserve_matrix(old, list(cohort.board), cohort.actual)
                reserve = pd.Series(matrix.mean(axis=0), index=list(cohort.board))
                first_rows[f"first_study:{row}"] = reserve.reindex(companies).to_numpy()
                extras[f"first study {row}"] = f"{len(old)} draws read from {first_study_dir}"
    result = evaluate_cohort(cohort, fits_by_row, bench, first_study=first_rows)
    result["n_cells_by_valuation"] = {
        v: int(
            bench[(bench["valuation"] == v) & bench["company_code"].isin(result["companies"])][
                "n_cells"
            ].sum()
        )
        for v in VALUATIONS
    }
    tie = reference_tie_out(cohort, result)
    if tie:
        extras["notebook 04 Pool_APE at valuation 10"] = tie
    if tie_out_first:
        extras["first study benchmarks at valuation 10"] = first_study_tie_out(
            cohort, result, log=log
        )
    if companion_rolling is not None and name == "B":
        table = companion_rolling_tie_out(companion_rolling, result)
        result["companion_rolling"] = table
        extras["companion's rolling table"] = "\n" + table.round(6).to_string(index=False)
    result["extras"] = extras
    return cohort, result


def save_tables(out: Path, cohort, result: dict) -> None:
    """Write the cohort's tables as CSV files in ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    name = cohort.name
    result["metrics"].to_csv(out / f"{name}_metrics_by_valuation.csv", index=False)
    result["differences"].to_csv(out / f"{name}_differences.csv", index=False)
    result["excluding_dominant"].to_csv(out / f"{name}_excluding_dominant.csv", index=False)
    result["quartiles"].to_csv(out / f"{name}_size_quartiles.csv", index=False)
    result["ablations"].to_csv(out / f"{name}_ablations.csv", index=False)
    result["network_check"].to_csv(out / f"{name}_network_check.csv", index=False)
    result["draw_pool_ape"].to_csv(out / f"{name}_draw_pool_ape.csv", index=False)
    company_table(result).to_csv(out / f"{name}_company_predictions.csv", index=False)
    if "companion_rolling" in result:
        result["companion_rolling"].to_csv(
            out / f"{name}_companion_rolling_tie_out.csv", index=False
        )


def verdict_record(result: dict) -> dict:
    """The JSON-able verdict of one cohort."""
    record = {
        "verdict": result["verdict"],
        "label": result["verdict_label"],
        "reason": result["verdict_reason"],
        "status": result["status"],
        "provisional": result["provisional"],
        "missing_draws": result["missing_draws"],
        "n_draws": result["n_draws"],
        "scored_companies": len(result["companies"]),
        "left_out": result["left_out"],
        "dominant_company": result["dominant_company"],
        "dominant_share": result["dominant_share"],
    }
    if result["verdict"] is not None:
        record["conditions"] = result["conditions"]
    record["network_question"] = {
        "verdict": result["network_verdict"],
        "label": result["network_label"],
        "reason": result["network_reason"],
        "provisional": result["network_provisional"],
        "conditions": result["network_conditions"],
    }
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fits", type=Path, required=True, help="the runner's --out folder")
    parser.add_argument("--cohort", choices=(*study.ALL_COHORTS, "all"), default="all")
    parser.add_argument("--out", type=Path, default=None, help="write the report and tables here")
    parser.add_argument(
        "--cache", type=Path, default=None, help="keep the benchmark fits here (default: --out)"
    )
    parser.add_argument("--cas-dir", type=Path, default=None, help="the four CAS files, cohort B")
    parser.add_argument(
        "--companion-rolling", type=Path, default=None, help="the companion's rolling table"
    )
    parser.add_argument(
        "--first-study-fits", type=Path, default=None, help="the first study's folder (A and B)"
    )
    parser.add_argument("--tie-out-first-study", action="store_true")
    parser.add_argument("--strict-data-hash", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="score a smoke folder")
    args = parser.parse_args(argv)

    names = list(study.ALL_COHORTS) if args.cohort == "all" else [args.cohort]
    banner = (
        "SMOKE RUN. The fits were a few epochs and a few members. Nothing below may be quoted."
        if args.smoke
        else ""
    )
    if banner:
        print(banner)
    reports, verdicts = [], {}
    cells_ab = None
    for name in names:
        t0 = time.perf_counter()
        if name in ("A", "B"):
            cells_ab = cells_ab if cells_ab is not None else study.cells_from_mart()
            cells = cells_ab
        else:
            cells = study.cells_from_mart(years=study.YEARS[name])
        got = run_cohort(
            name,
            args.fits,
            smoke=args.smoke,
            cells=cells,
            cas_dir=args.cas_dir,
            cache_dir=args.cache or args.out,
            strict=args.strict_data_hash,
            first_study_dir=args.first_study_fits,
            companion_rolling=args.companion_rolling,
            tie_out_first=args.tie_out_first_study,
        )
        if got is None:
            continue
        cohort, result = got
        report = render(cohort, result, result["extras"], banner)
        print(report)
        reports.append(report)
        verdicts[name] = verdict_record(result)
        print(f"(cohort {name} evaluated in {time.perf_counter() - t0:.0f}s)")
        if args.out is not None:
            save_tables(args.out, cohort, result)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.txt").write_text("\n\n".join(reports), encoding="utf-8")
        (args.out / "verdicts.json").write_text(
            json.dumps({"smoke": args.smoke, "verdicts": verdicts}, indent=2, default=str), "utf-8"
        )
    print("\nVERDICTS" + (" (SMOKE, not a result)" if args.smoke else ""))
    for key, value in verdicts.items():
        print(f"  {key}: {value['label']} [{value['status']}] ({value['n_draws']} draws)")
        print(f"     network question: {value['network_question']['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
