"""Milestone 9: speed benchmark vs chainladder-python.

Times equivalent operations on ibnr's two ibis backends (duckdb, polars)
against chainladder's numpy 4D backend, on three chainladder sample datasets
(raa, quarterly, clrd) and the Schedule P gold mart when it is available.

Methodology
-----------
* Each library is timed on its own canonical ingestion path from the SAME
  in-memory pandas data: ibnr from the stacked-long frame (its native layout),
  chainladder from the field-wide frame (its native layout). Both frames are
  prepared once, outside the timed region.
* ibis is lazy, so every ibnr operation is timed through to a materialized
  result (``execute()``, or ``count()`` for construction, which forces the
  engine-side pipeline without adding a pandas transfer that construction of
  the chainladder array does not pay either). chainladder's numpy backend is
  eager, so its calls are timed as-is.
* ``schedule_p / read_parquet`` is the one end-to-end file op: ibnr's
  ``load_schedule_p`` (parquet scan + unpivot in-engine) vs the equivalent
  chainladder user path (``pd.read_parquet`` + derived column + ``cl.Triangle``).
* Mack, two shapes: ``mack_fit_loop_*`` loops filter + ``fit_mack`` per cohort
  against a chainladder loop (its ``MackChainladder`` cannot fit multi-index
  triangles at all in 0.8.x); ``mack_fit_batch_*`` times ``fit_mack_many``
  (one materialization, cohorts gridded from the shared frame) against
  chainladder's only vectorized batch, the point-ultimate ``Chainladder`` -
  which computes strictly less (no sigma^2). Cohorts are pre-screened once
  (outside timing) so every implementation fits the identical set.
* Timing: ``time.perf_counter``, 1 warmup + ``--repeat`` runs (repeats shrink
  automatically for ops whose warmup exceeds ``--budget`` seconds); the table
  reports the median, the CSV keeps min/median/mean.
* Light result verification (cell counts across implementations, Mack ultimate
  tie-outs) runs outside the timed region; the full semantic guarantees live
  in the ``tieout`` test suite, not here.

Usage
-----
    uv run python scripts/benchmark_speed.py
    uv run python scripts/benchmark_speed.py --datasets raa clrd --repeat 3
    uv run python scripts/benchmark_speed.py --skip-mart

Outputs ``analysis/results/benchmark_speed.csv`` (+ ``_meta.json``) and prints
a markdown comparison table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import time
import warnings
from pathlib import Path

import chainladder as cl
import numpy as np
import pandas as pd
from ibis import _ as ix

from ibnr import Triangle
from ibnr.kernels.mack import fit_mack, fit_mack_many

CORE = ("origin_period", "dev_lag", "eval_date", "field", "value")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "analysis" / "results"

# ibnr backends to benchmark; chainladder is always included as the reference.
IBNR_BACKENDS = ("duckdb", "polars")


# -- data preparation (untimed) -------------------------------------------------


def cl_long_frame(tri: cl.Triangle) -> pd.DataFrame:
    """chainladder Triangle -> stacked-long pandas frame (ibnr's canonical input).

    Same reshape as ``ibnr.triangle.io.from_chainladder``, kept local because the
    benchmark needs the intermediate pandas frame itself, which io deliberately
    does not expose.
    """
    df = tri.to_frame(keepdims=True, implicit_axis=True, origin_as_datetime=True).reset_index()
    field_cols = [str(c) for c in tri.columns]
    id_cols = [c for c in df.columns if c not in field_cols]
    long = df.melt(id_vars=id_cols, value_vars=field_cols, var_name="field", value_name="value")
    long = long.dropna(subset=["value"])
    long = long.rename(
        columns={"origin": "origin_period", "development": "dev_lag", "valuation": "eval_date"}
    )
    long["origin_period"] = long["origin_period"].dt.date
    long["eval_date"] = long["eval_date"].dt.date
    long["dev_lag"] = long["dev_lag"].astype("int64")
    return long.reset_index(drop=True)


def wide_frame(long: pd.DataFrame, segments: list[str]) -> pd.DataFrame:
    """Stacked-long -> field-wide pandas frame (chainladder's canonical input)."""
    wide = long.pivot_table(
        index=[*segments, "origin_period", "eval_date"],
        columns="field",
        values="value",
        aggfunc="sum",
    ).reset_index()
    wide.columns.name = None
    wide["origin_period"] = pd.to_datetime(wide["origin_period"])
    wide["eval_date"] = pd.to_datetime(wide["eval_date"])
    return wide


def cl_construct(
    wide: pd.DataFrame, fields: list[str], segments: list[str], cumulative: bool
) -> cl.Triangle:
    return cl.Triangle(
        wide,
        origin="origin_period",
        development="eval_date",
        columns=fields,
        index=segments or None,
        cumulative=cumulative,
    )


def ibnr_construct(
    long: pd.DataFrame, backend: str, origin_grain: str, dev_grain: str, cumulative: bool
) -> Triangle:
    t = Triangle.from_long(
        long,
        measure="cumulative" if cumulative else "incremental",
        origin_grain=origin_grain,
        dev_grain=dev_grain,
        backend=backend,
    )
    t.count()  # force the engine-side cast/scan pipeline; ibis is lazy
    return t


def next_day(d: dt.date) -> pd.Timestamp:
    """Exclusive-upper-bound timestamp for chainladder's end-of-day valuations.

    ``tri[tri.valuation < next_day(cutoff)]`` keeps the ``cutoff`` diagonal,
    matching ibnr's inclusive ``as_of(cutoff)`` (see CLAUDE.md gotcha: cl
    valuations are end-of-day, so ``<= cutoff`` would drop it).
    """
    return pd.Timestamp(d) + pd.Timedelta(days=1)


# -- timing harness -------------------------------------------------------------


class Bench:
    def __init__(self, repeat: int, budget: float):
        self.repeat = repeat
        self.budget = budget
        self.rows: list[dict] = []

    def run(self, dataset: str, op: str, impl: str, fn, rows_in: int) -> None:
        gc.collect()
        t0 = time.perf_counter()
        fn()  # warmup (also surfaces errors before the timed loop)
        warmup = time.perf_counter() - t0
        if warmup > self.budget:
            repeat = 1
        elif warmup > self.budget / 5:
            repeat = min(self.repeat, 3)
        else:
            repeat = self.repeat
        times = []
        for _ in range(repeat):
            gc.collect()
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        self.rows.append(
            {
                "dataset": dataset,
                "op": op,
                "impl": impl,
                "rows_in": rows_in,
                "repeat": repeat,
                "warmup_s": round(warmup, 6),
                "min_s": round(min(times), 6),
                "median_s": round(statistics.median(times), 6),
                "mean_s": round(statistics.fmean(times), 6),
            }
        )
        print(
            f"  {dataset:11s} {op:18s} {impl:14s} "
            f"median {statistics.median(times) * 1e3:10.2f} ms  (n={repeat})"
        )


# -- per-dataset suites ----------------------------------------------------------


def screen_mack_cohorts(
    t: Triangle, seg_col: str, names: list[str], loss_field: str, as_of: str | None = None
) -> list[str]:
    """Cohorts (segment values) on which ``fit_mack`` succeeds; screened once on
    duckdb so every implementation later fits the identical set."""
    ok = []
    for n in names:
        try:
            fit_mack(t.filter(ix[seg_col] == n), loss_field=loss_field, as_of=as_of)
            ok.append(n)
        except Exception:
            pass
    return ok


def bench_dataset(
    bench: Bench,
    name: str,
    long: pd.DataFrame,
    *,
    origin_grain: str,
    dev_grain: str,
    cumulative: bool,
    cutoff: dt.date,
    backends: tuple[str, ...],
    grain_target: str | None = None,
    aggregate_over: tuple[str, str] | None = None,  # (segment kept, description)
    mack: dict | None = None,  # {loss_field, seg_col?, cohorts?, as_of?}
) -> None:
    segments = [c for c in long.columns if c not in CORE]
    fields = sorted(long["field"].unique())
    n = len(long)
    print(f"\n== {name}: {n:,} rows, {len(fields)} fields, segments={segments} ==")

    wide = wide_frame(long, segments)

    # construction (from each library's canonical pandas frame)
    for b in backends:
        bench.run(
            name,
            "construct",
            f"ibnr[{b}]",
            lambda b=b: ibnr_construct(long, b, origin_grain, dev_grain, cumulative),
            n,
        )
    bench.run(
        name,
        "construct",
        "chainladder",
        lambda: cl_construct(wide, fields, segments, cumulative),
        n,
    )

    # prebuilt triangles for the transform ops (construction cost excluded)
    itri = {b: ibnr_construct(long, b, origin_grain, dev_grain, cumulative) for b in backends}
    ctri = cl_construct(wide, fields, segments, cumulative)
    iincr = {b: t.to_incremental() for b, t in itri.items()}
    for t in iincr.values():
        t.count()  # materialize once so to_cumulative below times only itself
    cincr = ctri.cum_to_incr()

    ops = [
        ("to_incremental", lambda t: t.to_incremental().execute(), lambda: ctri.cum_to_incr()),
        (
            "to_cumulative",
            lambda t: t.to_cumulative().execute(),
            lambda: cincr.incr_to_cum(),
            iincr,
        ),
        (
            "as_of_slice",
            lambda t: t.as_of(cutoff).execute(),
            lambda: ctri[ctri.valuation < next_day(cutoff)],
        ),
        ("latest_diagonal", lambda t: t.latest_diagonal().execute(), lambda: ctri.latest_diagonal),
    ]
    if grain_target:
        ops.append(
            (
                f"dev_grain_{dev_grain}_to_{grain_target}",
                lambda t: t.with_dev_grain(grain_target).execute(),
                lambda: ctri.grain(f"O{origin_grain}D{grain_target}"),
            )
        )
    if aggregate_over:
        keep, _desc = aggregate_over

        def ibnr_agg(t):
            return (
                t.expr.group_by([keep, "origin_period", "dev_lag", "eval_date", "field"])
                .agg(value=ix.value.sum())
                .execute()
            )

        ops.append((f"aggregate_by_{keep}", ibnr_agg, lambda: ctri.groupby(keep).sum()))

    for entry in ops:
        op, ifn, cfn = entry[0], entry[1], entry[2]
        source = entry[3] if len(entry) > 3 else itri
        for b in backends:
            bench.run(name, op, f"ibnr[{b}]", lambda ifn=ifn, t=source[b]: ifn(t), n)
        bench.run(name, op, "chainladder", cfn, n)

    # light verification (warn-only; the tieout test suite is the enforcement
    # point): same cells survive the slice ops on every implementation
    ref = itri[backends[0]].as_of(cutoff).execute()
    n_ref = len(ref)
    for b in backends[1:]:
        n_b = len(itri[b].as_of(cutoff).execute())
        if n_b != n_ref:
            print(f"  WARN {name}: as_of cells differ across ibnr backends ({n_b} vs {n_ref})")
    # chainladder stores explicit zeros as NaN (the zero-vs-missing conflation
    # documented in CLAUDE.md), so its finite-cell count is compared against
    # ibnr's NONZERO rows - e.g. Schedule P's 316k explicit zero cells (bulk
    # loss, case reserves) are real observations to us and invisible to cl.
    n_cl_asof = int(np.isfinite(ctri[ctri.valuation < next_day(cutoff)].values).sum())
    n_ref_nonzero = int((ref["value"] != 0).sum())
    if n_cl_asof != n_ref_nonzero:
        print(f"  WARN {name}: as_of cells differ, cl {n_cl_asof} vs ibnr nonzero {n_ref_nonzero}")

    # Mack
    if mack:
        loss_field = mack["loss_field"]
        as_of = mack.get("as_of")
        seg_col, cohorts = mack.get("seg_col"), mack.get("cohorts")
        if cohorts is None:
            # single-cohort fit (the triangle IS one cohort, e.g. raa)
            for b in backends:
                bench.run(
                    name,
                    "mack_fit",
                    f"ibnr[{b}]",
                    lambda t=itri[b]: fit_mack(t, loss_field=loss_field, as_of=as_of),
                    n,
                )
            csub = ctri[loss_field]
            if as_of is not None:
                bench.run(
                    name,
                    "mack_fit",
                    "chainladder",
                    lambda: cl.MackChainladder().fit(
                        csub[csub.valuation < next_day(dt.date.fromisoformat(as_of))]
                    ),
                    n,
                )
            else:
                bench.run(
                    name, "mack_fit", "chainladder", lambda: cl.MackChainladder().fit(csub), n
                )
            # tie the ultimates out so the timed work is provably the same fit
            ifit = fit_mack(itri[backends[0]], loss_field=loss_field, as_of=as_of)
            cfit = cl.MackChainladder().fit(
                csub
                if as_of is None
                else csub[csub.valuation < next_day(dt.date.fromisoformat(as_of))]
            )
            if not np.allclose(ifit.full[:, -1].sum(), np.nansum(cfit.ultimate_.values), rtol=1e-6):
                print(f"  WARN {name}: Mack ultimates diverge between ibnr and chainladder")
        else:
            # Many cohorts, three rows. chainladder 0.8.x cannot fit
            # MackChainladder on a multi-index triangle at all (ValueError in
            # _get_full_std_err_, even on its own clrd sample), so:
            #   mack_fit_loop_*  - loop vs loop through each public API
            #   mack_fit_batch_* - ibnr's fit_mack_many (one materialization,
            #                      grids from the shared frame) vs chainladder's
            #                      only vectorized batch, the point-ultimate
            #                      Chainladder. Work asymmetry, in ibnr's favor
            #                      to lose: fit_mack_many also estimates the
            #                      sigma^2 variance parameters; Chainladder
            #                      produces point ultimates only.
            sub_long = mack["sub_long"]
            sub_long = sub_long[sub_long[seg_col].isin(cohorts)].reset_index(drop=True)
            nsub = len(sub_long)
            isub = {
                b: ibnr_construct(sub_long, b, origin_grain, dev_grain, cumulative)
                for b in backends
            }
            csub = cl_construct(
                wide_frame(sub_long[sub_long["field"] == loss_field], segments),
                [loss_field],
                segments,
                cumulative,
            )

            def ibnr_loop(t):
                for c in cohorts:
                    fit_mack(t.filter(ix[seg_col] == c), loss_field=loss_field, as_of=as_of)

            op = f"mack_fit_loop_{len(cohorts)}_cohorts"
            for b in backends:
                bench.run(name, op, f"ibnr[{b}]", lambda t=isub[b]: ibnr_loop(t), nsub)

            def cl_loop():
                for c in cohorts:
                    cl.MackChainladder().fit(csub[csub[seg_col] == c])

            bench.run(name, op, "chainladder", cl_loop, nsub)

            op = f"mack_fit_batch_{len(cohorts)}_cohorts"
            for b in backends:
                bench.run(
                    name,
                    op,
                    f"ibnr[{b}]",
                    lambda t=isub[b]: fit_mack_many(t, loss_field=loss_field, as_of=as_of),
                    nsub,
                )
            bench.run(name, op, "chainladder", lambda: cl.Chainladder().fit(csub), nsub)

            # batch == loop == chainladder's point ultimates, on the same cohorts
            panel = fit_mack_many(isub[backends[0]], loss_field=loss_field, as_of=as_of)
            total = sum(f.ultimate.sum() for f in panel.fits.values())
            cl_total = float(np.nansum(cl.Chainladder().fit(csub).ultimate_.values))
            if not np.allclose(total, cl_total, rtol=1e-6):
                print(f"  WARN {name}: batch ultimates diverge, ibnr {total} vs cl {cl_total}")


# -- schedule_p extras -----------------------------------------------------------


def cl_mart_from_parquet(mart_path: Path) -> cl.Triangle:
    """The chainladder-user counterpart of ``load_schedule_p``: same parquet,
    same field mapping and derived reported_loss, chainladder's own ingestion."""
    df = pd.read_parquet(mart_path)
    rename = {
        "cum_paid_loss": "paid_loss",
        "incurred_loss": "incurred_loss",
        "bulk_loss": "bulk_loss",
        "case_reserve": "case_reserve",
        "earned_prem_net": "earned_premium",
        "earned_prem_direct": "earned_premium_direct",
    }
    df = df.rename(columns=rename)
    df["reported_loss"] = df["incurred_loss"] - df["bulk_loss"]
    df["origin_period"] = pd.to_datetime(df["accident_year"], format="%Y")
    df["eval_date"] = pd.to_datetime(df["statement_year"].astype(str) + "-12-31")
    fields = [*rename.values(), "reported_loss"]
    return cl.Triangle(
        df,
        origin="origin_period",
        development="eval_date",
        columns=fields,
        index=["company_code", "company_name", "line_of_business"],
        cumulative=True,
    )


# -- main ------------------------------------------------------------------------


def main() -> None:
    # Degenerate cohorts (zero cells, short histories) make chainladder emit
    # sqrt/overflow RuntimeWarnings on every fit; printing them would distort
    # the timings, and correctness is the tieout suite's job, not this script's.
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore", category=UserWarning)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["raa", "quarterly", "clrd", "schedule_p"],
        choices=["raa", "quarterly", "clrd", "schedule_p"],
    )
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--budget", type=float, default=10.0, help="warmup seconds above which repeats shrink to 1"
    )
    ap.add_argument(
        "--skip-mart", action="store_true", help="drop schedule_p even if the mart resolves"
    )
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    backends = tuple(b for b in IBNR_BACKENDS)
    bench = Bench(args.repeat, args.budget)
    meta: dict = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "ibnr",
                "ibis-framework",
                "duckdb",
                "polars",
                "chainladder",
                "numpy",
                "pandas",
            )
        },
        "args": {"repeat": args.repeat, "budget": args.budget, "datasets": args.datasets},
    }

    if "raa" in args.datasets:
        raa = cl.load_sample("raa")
        long = cl_long_frame(raa)
        bench_dataset(
            bench,
            "raa",
            long,
            origin_grain="Y",
            dev_grain="Y",
            cumulative=raa.is_cumulative,
            cutoff=dt.date(1985, 12, 31),
            backends=backends,
            mack={"loss_field": str(raa.columns[0])},
        )

    if "quarterly" in args.datasets:
        q = cl.load_sample("quarterly")
        long = cl_long_frame(q)
        evals = sorted(long["eval_date"].unique())
        bench_dataset(
            bench,
            "quarterly",
            long,
            origin_grain="Y",
            dev_grain="Q",
            cumulative=q.is_cumulative,
            cutoff=evals[len(evals) // 2],
            backends=backends,
            grain_target="Y",
        )

    if "clrd" in args.datasets:
        clrd = cl.load_sample("clrd")
        long = cl_long_frame(clrd)
        # Mack cohort set: wkcomp companies that pass the staircase screen
        wk = long[long["LOB"] == "wkcomp"].reset_index(drop=True)
        t_screen = ibnr_construct(wk, "duckdb", "Y", "Y", clrd.is_cumulative)
        names = sorted(wk["GRNAME"].unique())
        cohorts = screen_mack_cohorts(t_screen, "GRNAME", names, "CumPaidLoss")
        print(f"clrd wkcomp Mack screen: {len(cohorts)}/{len(names)} cohorts fit cleanly")
        bench_dataset(
            bench,
            "clrd",
            long,
            origin_grain="Y",
            dev_grain="Y",
            cumulative=clrd.is_cumulative,
            cutoff=dt.date(1993, 12, 31),
            backends=backends,
            aggregate_over=("LOB", "industry aggregate"),
            mack={
                "loss_field": "CumPaidLoss",
                "seg_col": "GRNAME",
                "cohorts": cohorts,
                "sub_long": wk,
            },
        )

    if "schedule_p" in args.datasets and not args.skip_mart:
        try:
            from ibnr.data.schedule_p import active_mart_path, active_publish_id, load_schedule_p

            mart_path = active_mart_path()
            meta["schedule_p_publish_id"] = active_publish_id()
        except Exception as exc:
            print(f"\nschedule_p skipped: mart unavailable ({exc})")
            mart_path = None
        if mart_path is not None:
            t0 = load_schedule_p()
            long = t0.execute()
            for c in ("origin_period", "eval_date"):
                if str(long[c].dtype).startswith("datetime64"):
                    long[c] = long[c].dt.date
            n = len(long)
            # end-to-end from parquet (each library's realistic user path)
            for b in backends:
                bench.run(
                    "schedule_p",
                    "read_parquet",
                    f"ibnr[{b}]",
                    lambda b=b: load_schedule_p(backend=b).count(),
                    n,
                )
            bench.run(
                "schedule_p",
                "read_parquet",
                "chainladder",
                lambda: cl_mart_from_parquet(mart_path),
                n,
            )
            # single-company backtest fit: first WC company passing the screen
            wc = long[long["line_of_business"] == "workers_compensation"]
            t_screen = ibnr_construct(wc.reset_index(drop=True), "duckdb", "Y", "Y", True)
            codes = sorted(wc["company_code"].unique())
            passing = []
            for code in codes:
                passing = screen_mack_cohorts(
                    t_screen, "company_code", [code], "paid_loss", as_of="1997-12-31"
                )
                if passing:
                    break
            one = long[
                (long["line_of_business"] == "workers_compensation")
                & (long["company_code"] == passing[0])
            ].reset_index(drop=True)
            bench_dataset(
                bench,
                "schedule_p",
                long,
                origin_grain="Y",
                dev_grain="Y",
                cumulative=True,
                cutoff=dt.date(1997, 12, 31),
                backends=backends,
                aggregate_over=("line_of_business", "industry aggregate"),
                mack=None,
            )
            # Mack on the one screened company: slice-at-1997 + fit, both libraries
            for b in backends:
                t_one = ibnr_construct(one, b, "Y", "Y", True)
                bench.run(
                    "schedule_p",
                    "mack_fit_backtest",
                    f"ibnr[{b}]",
                    lambda t=t_one: fit_mack(t, loss_field="paid_loss", as_of="1997-12-31"),
                    len(one),
                )
            cone = cl_construct(
                wide_frame(one, ["company_code", "company_name", "line_of_business"]),
                ["paid_loss"],
                ["company_code", "company_name", "line_of_business"],
                True,
            )
            bench.run(
                "schedule_p",
                "mack_fit_backtest",
                "chainladder",
                lambda: cl.MackChainladder().fit(cone[cone.valuation < pd.Timestamp("1998-01-01")]),
                len(one),
            )
            # batch backtest fit across every screened WC company: fit_mack_many
            # doubles as its own screen (on_error="skip"), run once untimed; the
            # timed panel then holds only clean cohorts, identically for both
            # libraries. chainladder's row is its point-ultimate Chainladder on
            # the 1997 slice (its only multi-cohort batch; no sigma^2 - see the
            # clrd note).
            t_wc = ibnr_construct(wc.reset_index(drop=True), "duckdb", "Y", "Y", True)
            screen = fit_mack_many(
                t_wc, loss_field="paid_loss", as_of="1997-12-31", on_error="skip"
            )
            good_codes = sorted(k[0] for k in screen.fits)
            print(
                f"schedule_p WC batch screen: {len(screen.fits)} companies fit cleanly, "
                f"{len(screen.errors)} rejected"
            )
            wc_good = wc[wc["company_code"].isin(good_codes)].reset_index(drop=True)
            op = f"mack_fit_batch_{len(good_codes)}_wc_companies"
            for b in backends:
                t_b = ibnr_construct(wc_good, b, "Y", "Y", True)
                bench.run(
                    "schedule_p",
                    op,
                    f"ibnr[{b}]",
                    lambda t=t_b: fit_mack_many(t, loss_field="paid_loss", as_of="1997-12-31"),
                    len(wc_good),
                )
            cwc = cl_construct(
                wide_frame(
                    wc_good[wc_good["field"] == "paid_loss"],
                    ["company_code", "company_name", "line_of_business"],
                ),
                ["paid_loss"],
                ["company_code", "company_name", "line_of_business"],
                True,
            )
            bench.run(
                "schedule_p",
                op,
                "chainladder",
                lambda: cl.Chainladder().fit(cwc[cwc.valuation < pd.Timestamp("1998-01-01")]),
                len(wc_good),
            )

    # -- outputs ----------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(bench.rows)
    csv_path = args.out_dir / "benchmark_speed.csv"
    df.to_csv(csv_path, index=False)
    (args.out_dir / "benchmark_speed_meta.json").write_text(json.dumps(meta, indent=2))

    table = df.pivot_table(index=["dataset", "op"], columns="impl", values="median_s", sort=False)
    order = [c for c in ("ibnr[duckdb]", "ibnr[polars]", "chainladder") if c in table.columns]
    table = table[order]
    if "chainladder" in table.columns and "ibnr[duckdb]" in table.columns:
        table["cl / duckdb"] = table["chainladder"] / table["ibnr[duckdb]"]
    print("\n### Median seconds per operation\n")
    print(markdown_table(table))
    print(f"\nwrote {csv_path}")


def markdown_table(table: pd.DataFrame) -> str:
    """Minimal markdown renderer (pandas' to_markdown needs tabulate, not a dep)."""
    head = ["dataset", "op", *table.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for (ds, op), row in table.iterrows():
        cells = ["" if pd.isna(v) else f"{v:.4f}" for v in row]
        lines.append("| " + " | ".join([ds, op, *cells]) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
