"""The tlrn robustness study's evaluator: does the accident-year model beat the benchmarks?

Reads the fits that ``scripts/tlrn_study_colab.py`` saved (one small company-reserve CSV and
one settings record per fit) and scores them against benchmarks it computes itself with
ibnr's own entries. It needs no jax and no accelerator, only ibnr, the cas-schedule-p
package and the saved folder. The study's design and its reasons are in
``knowledge/decisions/tlrn-robustness-study.md``.

PRE-REGISTERED, written before any real fit existed
===================================================

What is scored. Two cohorts, each valued at 2007-12-31 on accident years 1998 to 2007:
cohort A (notebook 04's, 93 companies, 82 scored) and cohort B (the companion study's, 84
companies, all scored; a subset of A). Two rows, ``tlrn_8_ay`` and ``tlrn_13_ay``, each
with 10 independent draws (one draw is one fit of 20 members at 4 valuation dates, fit seed
``11 + 100003 k``). The benchmarks are ``mack`` (volume-weighted chain ladder, one fit per
company-line pair), ``mcl`` (multivariate chain ladder, one fit per company) and ``sur``,
from ibnr's gallery. A company's reserve is its 120-month cumulative paid minus its
cumulative paid at the valuation date, added over its lines. **Pool_APE** is the sum of
absolute reserve errors over the scored companies divided by the sum of actual reserves
(notebook 04's definition, ``ibnr.kernels.point_metrics``'s ``pool_ape``). The **draw
average** is the mean over draws of a company's reserve. The **dominant company** is the
company with the largest share of the cohort's actual reserve, found in code.

The six questions, answered with numbers for every cohort and row.

1. Draw to draw: Pool_APE of each draw as min / median / max / sd, and Pool_APE of the
   draw average, beside each benchmark.
2. Win rate: the fraction of draws whose Pool_APE is below ``mack``'s and below ``mcl``'s.
3. Paired company comparison of the draw average against ``mack`` and ``mcl`` (and ``sur``
   where it fit): the share of companies where the model's absolute error is smaller; the
   median over companies of the paired difference in absolute percentage error (model
   minus benchmark; negative favours the model); a company-level bootstrap of the pooled
   Pool_APE difference (model minus benchmark; 10,000 resamples of companies, seed 2008,
   the same resampled companies for model and benchmark, 95% percentile interval); and a
   two-sided Wilcoxon signed-rank p value on the paired absolute percentage errors.
   Companies with a zero actual reserve have no percentage error and are left out of the
   median and the Wilcoxon only. Each is given with the dominant company included and
   with it excluded.
4. The companion's bootstrap claim ("48.8% of bootstrap resamples favour TLRN") replicated:
   the share of the 10,000 resamples whose model Pool_APE is below the benchmark's, for
   each draw (min / median / max over draws) and for the draw average.
5. Variance decomposition: (a) the share of a single draw's squared error that is
   variance between draws (exact: mean squared error of a draw equals the squared error of
   the draw average plus the between-draw variance), on dollars and on each company's
   relative error; (b) the intraclass correlation of the log company reserve across draws
   (a company-size effect, near 1 by construction, so draw_noise_share leads the output);
   (c) Pool_APE of the average of every subset of k draws, k = 1 to the number of draws,
   with the share of those averages below ``mack`` and ``mcl``: does a bigger ensemble
   change the answer?
6. The verdict, per cohort and row, from the rule below.

The verdict rule. ``D(b)`` is the 95% bootstrap interval of Pool_APE(draw average) minus
Pool_APE(b), on the cohort with the dominant company included.

* ``effective``: the median-draw Pool_APE is below both ``mack``'s and ``mcl``'s AND
  ``D(mack)`` and ``D(mcl)`` both lie wholly below zero.
* ``worse``: not effective, and ``D(c)`` lies wholly above zero for ``c`` the benchmark with
  the lower Pool_APE of ``mack`` and ``mcl``.
* ``not distinguishable``: everything else.

No multiplicity correction is made over the eight comparisons, and cohort B lies inside
cohort A, so two cohort verdicts are not independent evidence. Nothing here shows what
happens at other valuation dates or in other years.

Usage::

    python scripts/tlrn_study_eval.py --fits DIR [--cohort A|B|both] [--rows ROWS]
        [--out DIR] [--cache DIR] [--cas-dir DIR] [--companion-reserves FILE] [--smoke]

``--cache`` (default: ``--out``) keeps the benchmark fits, which take minutes, so a rerun
while more draws arrive is quick. ``--companion-reserves`` is the companion study's
published company-reserve table; with it cohort B's ``mack`` and ``mcl`` are tied out to
its chain ladder and chain ladder + MCL columns, and its published transformer is shown
against the draws.

A smoke folder (``--smoke``) is scored too, so the evaluator can be checked, and its output
carries a banner saying the numbers must never be quoted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import tlrn_colab
import tlrn_study_colab as study
from scipy import stats

N_BOOT = 10_000
BOOT_SEED = 2008
LEVEL = 0.95
BENCHMARKS = ("mack", "mcl", "sur")
#: the two the verdict rule uses
VERDICT_BENCHMARKS = ("mack", "mcl")
VERDICTS = ("effective", "not distinguishable", "worse")
#: notebook 04's own Pool_APE of the two classical rows on cohort A, reproduced to this
#: relative tolerance (it asserts the reserves to 1e-6)
REFERENCE_A = {"mack": 0.05773223310028683, "mcl": 0.05660614217208907}
REFERENCE_TOL = 1e-6
#: how many subsets of k draws to enumerate exactly; beyond it a fixed random sample is used
MAX_SUBSETS = 5000
#: notebook 04's mack settings; only the point is read here
MACK_KWARGS = {"sigma_rule": "mack", "heldout_n_draws": 10_000}


# -- statistics ------------------------------------------------------------------------


def pool_ape(pred, actual) -> float:
    """Sum of absolute reserve errors over the sum of actual reserves."""
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    return float(np.abs(p - a).sum() / a.sum())


def draw_summary(values) -> dict:
    """Min, median, max and sample standard deviation of one value per draw."""
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
    }


def win_rate(draw_values, benchmark: float) -> float:
    """The fraction of draws whose value is strictly below the benchmark's."""
    v = np.asarray(draw_values, dtype=float)
    return float(np.mean(v < benchmark))


def bootstrap_indices(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """``(n_boot, n)`` company indices drawn with replacement, reproducible from ``seed``."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def pool_ape_resampled(pred, actual, idx: np.ndarray) -> np.ndarray:
    """Pool_APE of each resample of companies in ``idx``."""
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    total = a[idx].sum(axis=1)
    # a resample whose actual reserves add to zero has no Pool_APE; it is left out below
    return np.abs(p - a)[idx].sum(axis=1) / np.where(total != 0, total, np.nan)


def paired_comparison(pred, bench, actual, idx: np.ndarray, *, level: float = LEVEL) -> dict:
    """The model against one benchmark on the same companies (question 3 and 4).

    Differences are model minus benchmark, so a negative number favours the model.
    """
    p, b, a = (np.asarray(x, dtype=float) for x in (pred, bench, actual))
    err_p, err_b = np.abs(p - a), np.abs(b - a)
    nonzero = a != 0
    ape_diff = (err_p[nonzero] - err_b[nonzero]) / np.abs(a[nonzero])
    diff = pool_ape_resampled(p, a, idx) - pool_ape_resampled(b, a, idx)
    tail = (1.0 - level) / 2.0
    if ape_diff.size and np.any(ape_diff != 0):
        wilcoxon_p = float(stats.wilcoxon(ape_diff, zero_method="wilcox").pvalue)
    else:
        wilcoxon_p = 1.0 if ape_diff.size else float("nan")
    return {
        "n": int(a.size),
        "pool_ape_model": pool_ape(p, a),
        "pool_ape_benchmark": pool_ape(b, a),
        "pool_ape_diff": pool_ape(p, a) - pool_ape(b, a),
        "share_companies_won": float(np.mean(err_p < err_b)),
        "median_ape_diff": float(np.median(ape_diff)) if ape_diff.size else float("nan"),
        "ci_low": float(np.nanquantile(diff, tail)),
        "ci_high": float(np.nanquantile(diff, 1.0 - tail)),
        "share_resamples_favour_model": float(np.mean(diff[np.isfinite(diff)] < 0)),
        "wilcoxon_p": wilcoxon_p,
    }


def dominant_company(actual: pd.Series) -> tuple[str, float]:
    """The company with the largest share of the total actual reserve, and that share."""
    top = actual.idxmax()
    return str(top), float(actual.loc[top] / actual.sum())


def variance_decomposition(R, actual) -> dict:
    """Draw noise against company differences (question 5, parts a and b).

    ``R`` is ``(draws, companies)`` of company reserves. For every company the mean over
    draws of the squared error equals the squared error of the draw average plus the
    between-draw (population) variance, so ``draw_noise_share`` is how much of a single
    draw's squared error disappears when draws are averaged.
    """
    R = np.asarray(R, dtype=float)
    a = np.asarray(actual, dtype=float)
    mean = R.mean(axis=0)
    mse = ((R - a) ** 2).mean(axis=0)
    var = R.var(axis=0)
    bias2 = (mean - a) ** 2
    nonzero = a != 0
    rel = (1.0 / a[nonzero] ** 2) if nonzero.any() else np.array([])
    out = {
        "n_draws": int(R.shape[0]),
        "n_companies": int(R.shape[1]),
        "mse_single_draw": float(mse.sum()),
        "mse_draw_average": float(bias2.sum()),
        "between_draw_variance": float(var.sum()),
        "draw_noise_share": float(var.sum() / mse.sum()) if mse.sum() > 0 else float("nan"),
        "draw_noise_share_relative": (
            float((var[nonzero] * rel).sum() / (mse[nonzero] * rel).sum())
            if nonzero.any() and (mse[nonzero] * rel).sum() > 0
            else float("nan")
        ),
        "median_cv_across_draws": float(np.median(R.std(axis=0, ddof=1) / np.abs(mean)))
        if R.shape[0] > 1 and np.all(mean != 0)
        else float("nan"),
    }
    # the log scale needs a positive reserve in every draw; a company without one is left out
    positive = (R > 0).all(axis=0) if R.shape[0] > 1 else np.zeros(R.shape[1], dtype=bool)
    out["icc_companies"] = int(positive.sum())
    out["icc_log_reserve"] = _icc(np.log(R[:, positive])) if positive.sum() >= 2 else float("nan")
    return out


def _icc(y: np.ndarray) -> float:
    """One-way intraclass correlation of ``y`` (draws, companies): between companies."""
    d, c = y.shape
    grand = y.mean()
    msb = d * ((y.mean(axis=0) - grand) ** 2).sum() / (c - 1)
    msw = ((y - y.mean(axis=0)) ** 2).sum() / (c * (d - 1))
    denominator = msb + (d - 1) * msw
    return float((msb - msw) / denominator) if denominator > 0 else float("nan")


def subset_curve(R, actual, benchmarks: dict, *, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Pool_APE of the average of k draws, over subsets of k draws (question 5c).

    Every subset is used when there are at most ``MAX_SUBSETS`` of that size; otherwise a
    fixed random sample of that many. ``benchmarks`` maps a name to its Pool_APE, and the
    ``below_<name>`` columns are the share of subsets whose average is strictly below it.
    """
    R = np.asarray(R, dtype=float)
    a = np.asarray(actual, dtype=float)
    n = R.shape[0]
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(1, n + 1):
        total = comb(n, k)
        if total <= MAX_SUBSETS:
            subsets = [list(s) for s in itertools.combinations(range(n), k)]
        else:
            subsets = [sorted(rng.choice(n, k, replace=False).tolist()) for _ in range(MAX_SUBSETS)]
        values = np.array([pool_ape(R[s].mean(axis=0), a) for s in subsets])
        row = {
            "k": k,
            "subsets": len(subsets),
            "exhaustive": total <= MAX_SUBSETS,
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
        for name, value in benchmarks.items():
            row[f"below_{name}"] = float(np.mean(values < value))
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(median_draw: float, pool: dict, interval: dict) -> tuple[str, str]:
    """The pre-registered verdict and the sentence that justifies it.

    ``pool`` maps ``mack`` and ``mcl`` to their Pool_APE; ``interval`` maps each to the
    ``(low, high)`` bootstrap interval of Pool_APE(draw average) minus Pool_APE(benchmark).
    """
    below_both = all(median_draw < pool[b] for b in VERDICT_BENCHMARKS)
    wholly_below = all(interval[b][1] < 0 for b in VERDICT_BENCHMARKS)
    strongest = min(VERDICT_BENCHMARKS, key=lambda b: pool[b])
    wholly_above = interval[strongest][0] > 0
    if below_both and wholly_below:
        word = "effective"
    elif wholly_above:
        word = "worse"
    else:
        word = "not distinguishable"
    parts = [
        f"median-draw Pool_APE {median_draw:.4%} against mack {pool['mack']:.4%} and "
        f"mcl {pool['mcl']:.4%}",
        "draw-average difference interval "
        + ", ".join(
            f"vs {b} [{interval[b][0]:+.4%}, {interval[b][1]:+.4%}]" for b in VERDICT_BENCHMARKS
        ),
    ]
    return word, "; ".join(parts)


def missing_benchmark_text(bench: pd.DataFrame, names=VERDICT_BENCHMARKS) -> str:
    """Which benchmark lacks a reserve for which companies, empty when none does."""
    parts = []
    for b in names:
        gone = [str(c) for c in bench.index[~bench[b].notna().to_numpy()]]
        if gone:
            shown = ", ".join(gone[:8]) + (f" and {len(gone) - 8} more" if len(gone) > 8 else "")
            noun = "company" if len(gone) == 1 else "companies"
            parts.append(f"{b} has no reserve for {len(gone)} {noun} ({shown})")
    return "; ".join(parts)


def draw_completeness(draws: list[int]) -> tuple[bool, list[int]]:
    """``(provisional, missing)``: the pre-registered rule needs draws 0 to 9, all present."""
    wanted = list(range(study.DEFAULT_DRAWS))
    missing = sorted(set(wanted) - set(draws))
    return sorted(draws) != wanted, missing


# -- the saved fits --------------------------------------------------------------------

RECORD_NAME = re.compile(
    r"^(?P<cohort>[AB])_(?P<row>tlrn_\w+?)_d(?P<draw>\d+)(?P<smoke>_smoke)?-ibnr(?P<version>.+)\.json$"
)


@dataclass
class Fit:
    draw: int
    seed: int
    frame: pd.DataFrame
    settings: dict
    run: dict


def load_fits(folder: Path, cohort: str, row: str, *, smoke: bool) -> list[Fit]:
    """Every finished fit of one cohort and row, in draw order, checked to be one study.

    All fits must share every setting except the draw and its seed, and each seed must be
    the one its draw implies. Two ibnr versions of one draw are refused rather than one
    of them picked.
    """
    found: dict[int, tuple[Path, str]] = {}
    for path in sorted(Path(folder).glob(f"{cohort}_{row}_d*.json")):
        m = RECORD_NAME.match(path.name)
        if m is None or m["cohort"] != cohort or m["row"] != row or bool(m["smoke"]) != smoke:
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
        reserves = path.with_name(path.stem + "_reserves.csv")
        if not reserves.exists():
            raise SystemExit(f"{path.name} has no {reserves.name}")
        if settings["seed"] != study.draw_seed(settings["draw"]) or settings["draw"] != draw:
            raise SystemExit(f"{path.name}: the seed or draw in the record is not draw {draw}'s")
        frame = pd.read_csv(reserves, dtype={"company_code": str}).set_index("company_code")
        fits.append(Fit(draw, settings["seed"], frame, settings, record.get("run", {})))
    if fits:
        reference = fits[0].settings
        for fit in fits[1:]:
            differ = [
                k
                for k in sorted(reference.keys() | fit.settings.keys())
                if k not in ("draw", "seed") and reference.get(k) != fit.settings.get(k)
            ]
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

    The cohort hash (the pairs), the publish id and the study version must equal the
    evaluator's. The cell hash is the sha256 of the cells printed as text, so another
    machine can print the same numbers differently (last-bit float differences) and give
    another hash. When only it differs, every fit's anchor, premium, number of lines and
    company set are compared with the cohort's own, and any gap is refused by name; if all
    agree the fits are scored and the returned note says so. ``strict`` refuses instead.
    """
    where = f"cohort {cohort.name} {row}"
    for fit in fits:
        s = fit.settings
        for key, want in (
            ("cohort_hash", cohort.pairs_hash),
            ("publish_id", tlrn_colab.PUBLISH),
            ("study", study.STUDY_VERSION),
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
        frame = fit.frame
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


def reserve_matrix(fits: list[Fit], board: list[str], actual: pd.DataFrame) -> np.ndarray:
    """``(draws, board companies)`` of the draw-level reserves, checked against the cohort.

    The anchor each fit stored must equal the cohort's own, which is the plumbing check
    that the fit and the evaluator see the same cumulative paid.
    """
    rows = []
    for fit in fits:
        missing = [c for c in board if c not in fit.frame.index]
        if missing:
            raise SystemExit(
                f"draw {fit.draw} has no reserve for {len(missing)} companies: {missing[:5]}"
            )
        sub = fit.frame.loc[board]
        gap = float(np.abs(sub["anchor"].to_numpy() - actual.loc[board, "anchor"].to_numpy()).max())
        if gap > 1e-6 * max(1.0, float(actual["anchor"].abs().max())):
            raise SystemExit(
                f"draw {fit.draw}: the fit's anchors differ from the cohort's by up to {gap:,.0f}; "
                "the fit was made on other data"
            )
        rows.append(sub["reserve"].to_numpy(dtype=float))
    return np.vstack(rows)


# -- the benchmarks --------------------------------------------------------------------


def benchmark_reserves(cohort, *, log=print) -> pd.DataFrame:
    """Company reserves of mack, mcl and sur, with the actual reserve, from ibnr's entries.

    Fitted as notebook 04 fits them (``mack`` per company-line pair, ``mcl`` and ``sur``
    per company) and read with ``reserve_rows(point="native")``. A company a benchmark
    refuses, or that ``mack`` fits on only some of its lines, is NaN for that benchmark
    (a reserve short of a line is plausible and wrong). The triangle's anchors and
    outcomes are checked against the cohort's own.
    """
    from ibnr import gallery
    from ibnr.gallery import reserve_rows

    # An in-memory triangle of the cohort's cells: the mart's own triangle filters 243 pairs
    # with an OR on every query, which made the fits several times slower for the same cells.
    tri = study.triangle_from_cells(cohort.cells, cohort.pairs)
    e = tri.expr
    # the cohort's own valuation date: cohort C is valued at 1997-12-31, not 2007-12-31
    as_of = getattr(cohort, "as_of", tlrn_colab.AS_OF)
    kwargs = dict(
        as_of=as_of,
        loss_field=tlrn_colab.FIELD,
        premium_field="earned_premium",
        point="native",
    )
    fit_kwargs = dict(loss_field=tlrn_colab.FIELD, as_of=as_of)
    board = list(cohort.board)
    expected_lines = {c: sum(1 for cc, _ in cohort.pairs if cc == c) for c in board}
    parts, refused = [], {b: 0 for b in BENCHMARKS}
    t0 = time.perf_counter()
    for company, line in cohort.pairs:
        if company not in expected_lines:
            continue
        sub = tri.filter((e.company_code == company) & (e.line_of_business == line))
        try:
            entry = gallery.fit("mack", sub, **fit_kwargs, **MACK_KWARGS)
            parts.append(reserve_rows(entry, tri, **kwargs).assign(model="mack"))
        except Exception:
            refused["mack"] += 1
    for name in ("mcl", "sur"):
        for company in board:
            try:
                entry = gallery.fit(name, tri.filter(e.company_code == company), **fit_kwargs)
                parts.append(reserve_rows(entry, tri, **kwargs).assign(model=name))
            except Exception:
                refused[name] += 1
    log(f"  benchmarks fitted in {time.perf_counter() - t0:.0f}s; refusals {refused}")
    rows = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame(index=pd.Index(board, name="company_code"))
    for name in BENCHMARKS:
        got = rows[rows["model"] == name].groupby("company_code")
        reserve = got["predicted_reserve"].sum().reindex(out.index)
        if name == "mack":
            reserve[got.size().reindex(out.index).fillna(0) != pd.Series(expected_lines)] = np.nan
        out[name] = reserve
        done = reserve.dropna().index
        if len(done):
            want = cohort.actual.loc[done]
            gap = max(
                float(np.abs(got["actual_reserve"].sum().loc[done] - want["actual_reserve"]).max()),
                float(np.abs(got["anchor"].sum().loc[done] - want["anchor"]).max()),
            )
            if gap > 1e-6 * float(want["anchor"].abs().max()):
                raise SystemExit(
                    f"{name}: the triangle's anchor or actual reserve differs from the "
                    f"cohort's by {gap}"
                )
    out["actual"] = cohort.actual.loc[out.index, "actual_reserve"]
    return out


def cached_benchmarks(cohort, cache_dir: Path | None, *, log=print) -> pd.DataFrame:
    """:func:`benchmark_reserves`, kept in ``cache_dir`` so a rerun does not refit them.

    The file name carries the cohort and its data hash, so a cache made from other cells is
    never read. ``cache_dir=None`` always computes.
    """
    if cache_dir is None:
        return benchmark_reserves(cohort, log=log)
    path = Path(cache_dir) / f"benchmarks_{cohort.name}_{cohort.data_hash}.csv"
    if path.exists():
        frame = pd.read_csv(path, dtype={"company_code": str}).set_index("company_code")
        if list(frame.index) == list(cohort.board) and set(frame.columns) == {
            *BENCHMARKS,
            "actual",
        }:
            log(f"  benchmarks read from {path}")
            return frame[[*BENCHMARKS, "actual"]]
    frame = benchmark_reserves(cohort, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def companion_reserves(path: Path, board: list[str]) -> pd.DataFrame:
    """The companion's published company reserves from its table, for the cohort's companies."""
    table = pd.read_csv(path, dtype={"GRCODE": str}).set_index("GRCODE")
    missing = [c for c in board if c not in table.index]
    if missing:
        raise SystemExit(
            f"{path} has no row for {len(missing)} companies of the cohort: {missing[:5]}"
        )
    return table.loc[board]


# -- the evaluation --------------------------------------------------------------------


def evaluate_row(
    row: str,
    R: np.ndarray,
    bench: pd.DataFrame,
    board: list[str],
    *,
    draws: list[int] | None = None,
) -> dict:
    """Questions 1 to 6 for one row of one cohort, from the draw reserves ``R``.

    ``draws`` are the draw indices of the rows of ``R`` (default 0 to n-1). The verdict
    rule is the pre-registered one whatever is passed, but unless the set is exactly draws
    0 to 9 the verdict is reported as provisional: ``verdict`` keeps the rule's word and
    ``verdict_label`` and ``verdict_reason`` say how many of the 10 draws were present.
    """
    draws = list(range(R.shape[0])) if draws is None else [int(d) for d in draws]
    if len(draws) != R.shape[0]:
        raise ValueError(f"{len(draws)} draw indices for {R.shape[0]} rows of reserves")
    actual = bench["actual"].to_numpy(dtype=float)
    average = R.mean(axis=0)
    top, share = dominant_company(pd.Series(actual, index=board))
    keep_all = np.ones(len(board), dtype=bool)
    keep_ex = np.array([c != top for c in board])

    per_draw = np.array([pool_ape(R[d], actual) for d in range(R.shape[0])])
    per_draw_ex = np.array([pool_ape(R[d][keep_ex], actual[keep_ex]) for d in range(R.shape[0])])
    pools = {
        b: pool_ape(bench[b].to_numpy(dtype=float), actual)
        for b in BENCHMARKS
        if bench[b].notna().all()
    }
    result = {
        "row": row,
        "n_draws": int(R.shape[0]),
        "dominant_company": top,
        "dominant_share": share,
        "draw_pool_ape": per_draw,
        "draw_pool_ape_excluding_dominant": per_draw_ex,
        "summary": draw_summary(per_draw),
        "summary_excluding_dominant": draw_summary(per_draw_ex),
        "average_pool_ape": pool_ape(average, actual),
        "average_pool_ape_excluding_dominant": pool_ape(average[keep_ex], actual[keep_ex]),
        "benchmark_pool_ape": pools,
        "benchmark_pool_ape_excluding_dominant": {
            b: pool_ape(bench[b].to_numpy(dtype=float)[keep_ex], actual[keep_ex]) for b in pools
        },
        "win_rate": {b: win_rate(per_draw, pools[b]) for b in pools},
        "win_rate_excluding_dominant": {
            b: win_rate(
                per_draw_ex, pool_ape(bench[b].to_numpy(dtype=float)[keep_ex], actual[keep_ex])
            )
            for b in pools
        },
    }
    comparisons, claim = [], []
    for scope, keep in (("including dominant", keep_all), ("excluding dominant", keep_ex)):
        idx = bootstrap_indices(int(keep.sum()))
        for b in BENCHMARKS:
            col = bench[b].to_numpy(dtype=float)
            ok = keep & np.isfinite(col)
            if not ok.any():
                continue
            idx_b = idx if ok.sum() == keep.sum() else bootstrap_indices(int(ok.sum()))
            cmp = paired_comparison(average[ok], col[ok], actual[ok], idx_b)
            comparisons.append({"scope": scope, "benchmark": b, **cmp})
            shares = np.array(
                [
                    paired_comparison(R[d][ok], col[ok], actual[ok], idx_b)[
                        "share_resamples_favour_model"
                    ]
                    for d in range(R.shape[0])
                ]
            )
            claim.append(
                {
                    "scope": scope,
                    "benchmark": b,
                    "n": int(ok.sum()),
                    "draw_min": float(shares.min()),
                    "draw_median": float(np.median(shares)),
                    "draw_max": float(shares.max()),
                    "draw_average": cmp["share_resamples_favour_model"],
                }
            )
    result["comparisons"] = pd.DataFrame(comparisons)
    result["bootstrap_claim"] = pd.DataFrame(claim)
    result["variance"] = variance_decomposition(R, actual)
    result["variance_excluding_dominant"] = variance_decomposition(R[:, keep_ex], actual[keep_ex])
    result["subset_curve"] = subset_curve(R, actual, pools)
    inc = result["comparisons"]
    inc = inc[inc["scope"] == "including dominant"].set_index("benchmark")
    if all(b in pools and b in inc.index for b in VERDICT_BENCHMARKS):
        word, why = verdict(
            result["summary"]["median"],
            {b: pools[b] for b in VERDICT_BENCHMARKS},
            {b: (inc.loc[b, "ci_low"], inc.loc[b, "ci_high"]) for b in VERDICT_BENCHMARKS},
        )
    else:
        word = "not distinguishable"
        why = "the rule needs mack and mcl for every company, and " + (
            missing_benchmark_text(bench) or "a benchmark is missing"
        )
    provisional, missing = draw_completeness(draws)
    label = word
    if provisional:
        note = f"provisional: {len(draws)} of {study.DEFAULT_DRAWS} draws"
        if missing:
            note += " (missing draws " + ", ".join(str(d) for d in missing) + ")"
        why = f"{note}; {why}"
        label = f"{word} ({note})"
    result.update(
        verdict=word,
        verdict_reason=why,
        verdict_label=label,
        provisional=provisional,
        missing_draws=missing,
        draws=draws,
    )
    return result


def fmt_pct(x: float) -> str:
    return f"{100 * x:.3f}%"


VARIANCE_LEAD = ("draw_noise_share", "draw_noise_share_relative")


def fmt_variance(v: dict) -> str:
    """The variance decomposition with the draw-noise shares first and the ICC labelled."""
    order = [*VARIANCE_LEAD, *(k for k in v if k not in VARIANCE_LEAD)]
    parts = []
    for k in order:
        value = v[k]
        text = f"{value:.4g}" if isinstance(value, float) else f"{value}"
        if k == "icc_log_reserve":
            text += " (company-size effect, not draw noise)"
        parts.append(f"{k} {text}")
    return ", ".join(parts)


def render(cohort, row_results: dict[str, dict], extras: dict, banner: str) -> str:
    """The text report of one cohort."""
    lines = []
    if banner:
        lines += [banner, ""]
    lines.append(f"=== {cohort.description}")
    for key, value in extras.items():
        lines.append(f"{key}: {value}")
    for row, res in row_results.items():
        s, sx = res["summary"], res["summary_excluding_dominant"]
        lines += [
            "",
            f"--- {row}: {res['n_draws']} draws"
            + (" (PROVISIONAL, not the registered 10)" if res["provisional"] else "")
            + f"; dominant company {res['dominant_company']} "
            f"holds {res['dominant_share']:.1%} of the actual reserve",
            "Q1 Pool_APE per draw: " + " ".join(f"{fmt_pct(v)}" for v in res["draw_pool_ape"]),
            f"   min {fmt_pct(s['min'])}  median {fmt_pct(s['median'])}  max {fmt_pct(s['max'])}  "
            f"sd {fmt_pct(s['sd']) if s['n'] > 1 else 'n/a'}  | draw average "
            f"{fmt_pct(res['average_pool_ape'])}",
            f"   excluding the dominant company: median {fmt_pct(sx['median'])}, draw average "
            f"{fmt_pct(res['average_pool_ape_excluding_dominant'])}",
            "   benchmarks: "
            + "  ".join(f"{b} {fmt_pct(v)}" for b, v in res["benchmark_pool_ape"].items())
            + "  | excluding dominant: "
            + "  ".join(
                f"{b} {fmt_pct(v)}" for b, v in res["benchmark_pool_ape_excluding_dominant"].items()
            ),
            "Q2 win rate (draws below the benchmark): "
            + "  ".join(f"{b} {v:.0%}" for b, v in res["win_rate"].items())
            + "  | excluding dominant: "
            + "  ".join(f"{b} {v:.0%}" for b, v in res["win_rate_excluding_dominant"].items()),
            "Q3 paired company comparison of the draw average (negative favours the model):",
            res["comparisons"].round(5).to_string(index=False),
            "Q4 share of 10,000 company resamples whose model Pool_APE is below the benchmark's:",
            res["bootstrap_claim"].round(4).to_string(index=False),
            "Q5a/b variance decomposition (including dominant company). Read "
            "draw_noise_share first: the part of a single draw's squared error that "
            "averaging draws removes. icc_log_reserve is a company-size effect (reserves "
            "differ by orders of magnitude between companies), near 1 by construction, "
            "and does not measure draw noise:",
            "   " + fmt_variance(res["variance"]),
            "   excluding dominant: " + fmt_variance(res["variance_excluding_dominant"]),
            "Q5c Pool_APE of the average of k draws:",
            res["subset_curve"].round(5).to_string(index=False),
            f"Q6 VERDICT: {row} on cohort {cohort.name} is {res['verdict'].upper()} "
            f"({res['verdict_reason']})",
        ]
    return "\n".join(lines)


def max_relative_difference(a, b) -> float:
    """Largest ``|a - b| / |b|`` over the companies, the tie-out measure used for reserves.

    A company whose reference reserve is exactly zero has no relative difference and
    contributes its absolute difference instead.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.where(b != 0, np.abs(b), 1.0)
    return float(np.max(np.abs(a - b) / scale))


def tie_out(cohort, bench: pd.DataFrame, companion: pd.DataFrame | None) -> dict:
    """How the benchmark rows reproduce their references."""
    out = {}
    actual = bench["actual"].to_numpy(dtype=float)
    pools = {
        b: pool_ape(bench[b].to_numpy(dtype=float), actual)
        for b in BENCHMARKS
        if bench[b].notna().all()
    }
    if cohort.name == "A":
        absent = missing_benchmark_text(bench, tuple(REFERENCE_A))
        if absent:
            raise SystemExit(f"cannot tie out to notebook 04's Pool_APE on cohort A: {absent}")
        for b, ref in REFERENCE_A.items():
            rel = abs(pools[b] / ref - 1.0)
            out[f"{b} Pool_APE against notebook 04's"] = (
                f"{pools[b]:.10f} vs {ref:.10f} (relative {rel:.2e})"
            )
            if rel > REFERENCE_TOL:
                raise SystemExit(
                    f"{b} does not reproduce notebook 04's Pool_APE on cohort A: "
                    f"{pools[b]} vs {ref}"
                )
    if companion is not None:
        # ibnr's `sur` is a different estimator from the companion's SUR (other defaults),
        # so it is not expected to tie out and is not compared
        for ours, theirs in (("mack", "IBNR_CL"), ("mcl", "IBNR_CL_MCL")):
            if theirs in companion.columns and bench[ours].notna().all():
                a = bench[ours].to_numpy(dtype=float)
                b = companion[theirs].to_numpy(dtype=float)
                out[f"{ours} reserves against the companion's {theirs}"] = (
                    f"max relative difference {max_relative_difference(a, b):.2e}"
                )
        if "IBNR_obs" in companion.columns:
            a = bench["actual"].to_numpy(dtype=float)
            b = companion["IBNR_obs"].to_numpy(dtype=float)
            out["actual reserves against the companion's IBNR_obs"] = (
                f"max relative difference {max_relative_difference(a, b):.2e}"
            )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fits", type=Path, required=True, help="the runner's --out folder")
    parser.add_argument("--cohort", choices=("A", "B", "both"), default="both")
    parser.add_argument("--rows", default=",".join(study.ROWS))
    parser.add_argument("--out", type=Path, default=None, help="write the report and tables here")
    parser.add_argument(
        "--cache", type=Path, default=None, help="keep the benchmark fits here (default: --out)"
    )
    parser.add_argument("--cas-dir", type=Path, default=None, help="the four CAS files, cohort B")
    parser.add_argument(
        "--companion-reserves", type=Path, default=None, help="cohort B's published table"
    )
    parser.add_argument("--smoke", action="store_true", help="score a smoke folder")
    parser.add_argument(
        "--strict-data-hash",
        action="store_true",
        help="refuse a fit whose recorded cell hash differs from the one this evaluator builds "
        "(by default such a fit is scored when its anchors, premiums, lines and companies "
        "equal the cohort's)",
    )
    args = parser.parse_args(argv)

    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    names = list(study.COHORTS) if args.cohort == "both" else [args.cohort]
    banner = (
        "SMOKE RUN. The fits were a few epochs and a few members. Nothing below may be quoted."
        if args.smoke
        else ""
    )
    if banner:
        print(banner)
    cells = study.cells_from_mart()
    reports, verdicts, data_hash_notes = [], {}, {}
    for name in names:
        t0 = time.perf_counter()
        cohort = study.load_cohort(name, cas_dir=args.cas_dir if name == "B" else None, cells=cells)
        print(cohort.description, flush=True)
        # Benchmarks cost about 80 s per cohort: fit them only for a cohort with a finished fit.
        fits_by_row = {row: load_fits(args.fits, name, row, smoke=args.smoke) for row in rows}
        if not any(fits_by_row.values()):
            print(f"cohort {name}: no finished fits in {args.fits}, benchmarks not fitted, skipped")
            continue
        bench = cached_benchmarks(cohort, args.cache or args.out)
        companion = (
            companion_reserves(args.companion_reserves, cohort.board)
            if name == "B" and args.companion_reserves
            else None
        )
        extras = tie_out(cohort, bench, companion)
        results = {}
        for row in rows:
            fits = fits_by_row[row]
            if not fits:
                print(f"cohort {name} {row}: no finished fits in {args.fits}, skipped")
                continue
            note = check_fits_against_cohort(cohort, row, fits, strict=args.strict_data_hash)
            if note:
                print(note, flush=True)
                data_hash_notes[f"{name}/{row}"] = {
                    "note": note,
                    "fits_data_hash": fits[0].settings["data_hash"],
                    "evaluator_data_hash": cohort.data_hash,
                }
            R = reserve_matrix(fits, cohort.board, cohort.actual)
            results[row] = evaluate_row(row, R, bench, cohort.board, draws=[f.draw for f in fits])
            if companion is not None and "IBNR_TLRN" in companion.columns:
                ref = pool_ape(
                    companion["IBNR_TLRN"].to_numpy(dtype=float),
                    bench["actual"].to_numpy(dtype=float),
                )
                share_below = float(np.mean(results[row]["draw_pool_ape"] < ref))
                extras[f"{row}: companion's published transformer Pool_APE"] = (
                    f"{ref:.4%}; {share_below:.0%} of these draws are below it"
                )
            verdicts[f"{name}/{row}"] = {
                "verdict": results[row]["verdict"],
                "reason": results[row]["verdict_reason"],
                "label": results[row]["verdict_label"],
                "provisional": results[row]["provisional"],
                "missing_draws": results[row]["missing_draws"],
                "n_draws": results[row]["n_draws"],
            }
            if f"{name}/{row}" in data_hash_notes:
                verdicts[f"{name}/{row}"]["data_hash"] = data_hash_notes[f"{name}/{row}"]
        report = render(cohort, results, extras, banner)
        print(report)
        reports.append(report)
        print(f"(cohort {name} evaluated in {time.perf_counter() - t0:.0f}s)")
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            for row, res in results.items():
                pd.DataFrame(
                    {
                        "pool_ape": res["draw_pool_ape"],
                        "pool_ape_excluding_dominant": res["draw_pool_ape_excluding_dominant"],
                    }
                ).rename_axis("draw").to_csv(args.out / f"{name}_{row}_pool_ape_by_draw.csv")
                res["comparisons"].to_csv(
                    args.out / f"{name}_{row}_paired_comparison.csv", index=False
                )
                res["bootstrap_claim"].to_csv(
                    args.out / f"{name}_{row}_bootstrap_claim.csv", index=False
                )
                res["subset_curve"].to_csv(args.out / f"{name}_{row}_subset_curve.csv", index=False)
            bench.to_csv(args.out / f"{name}_benchmark_reserves.csv")
    if args.out is not None:
        (args.out / "report.txt").write_text("\n\n".join(reports), encoding="utf-8")
        (args.out / "verdicts.json").write_text(
            json.dumps({"smoke": args.smoke, "verdicts": verdicts}, indent=2), "utf-8"
        )
    print("\nVERDICTS" + (" (SMOKE, not a result)" if args.smoke else ""))
    for key, value in verdicts.items():
        print(f"  {key}: {value['label']} ({value['n_draws']} draws)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
