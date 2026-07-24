"""Scaling curves: where does the ibnr/chainladder crossover happen?

The fixed-dataset benchmark (``benchmark_speed.py``) showed scale deciding the
winner - chainladder's numpy wins small in-memory transforms, ibnr's engines
win the 1M-row mart - but four dataset sizes cannot say WHERE each operation
crosses over. This script sweeps synthetic multi-company Schedule-P-shaped
triangles from ~1e5 to ~1e7 long rows and times the same transform set on all
three implementations, so each operation gets a curve instead of a point.

Synthetic data: annual-grain run-off staircases (20 origins x up to 10 devs,
155 cells per cohort), 3 lines of business x 3 measure fields per company;
size is scaled by the company count. Values are lognormal draws - the
transforms never look at the values, only move them.

Methodology is inherited from benchmark_speed (same Bench harness, same
canonical-ingestion rule, same materialization rule); see that module's
docstring. Sizes above ``--budget`` warmup seconds drop to a single repeat, so
the 1e7 chainladder points stay affordable.

Usage
-----
    uv run python scripts/benchmark_scaling.py
    uv run python scripts/benchmark_scaling.py --sizes 100000 1000000 --repeat 5

Outputs ``analysis/results/benchmark_scaling.csv`` and a log-log crossover
chart ``analysis/results/benchmark_scaling.png``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ibis import _ as ix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_speed import (  # noqa: E402
    RESULTS_DIR,
    Bench,
    cl_construct,
    ibnr_construct,
    markdown_table,
    next_day,
)

N_ORIGINS = 20
N_DEVS = 10
N_LOBS = 3
FIELDS = ["paid_loss", "incurred_loss", "case_reserve"]
START_YEAR = 1988
#: staircase cells per cohort: 20 origins, dev depth min(10, 20 - w)
CELLS_PER_COHORT = sum(min(N_DEVS, N_ORIGINS - w) for w in range(N_ORIGINS))


def synth(target_rows: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Generate ~``target_rows`` long rows of Schedule-P-shaped triangles.

    Returns ``(long, wide, n_companies)``: the stacked-long frame (ibnr's
    canonical input) and the field-wide frame (chainladder's), built from one
    pass so the two are the same cells by construction rather than by pivot.
    """
    rows_per_company = CELLS_PER_COHORT * N_LOBS * len(FIELDS)
    n_companies = max(1, round(target_rows / rows_per_company))
    rng = np.random.default_rng(seed)

    # one cohort's staircase template, then tile it across company x lob
    w = np.repeat(np.arange(N_ORIGINS), [min(N_DEVS, N_ORIGINS - i) for i in range(N_ORIGINS)])
    d = np.concatenate([np.arange(min(N_DEVS, N_ORIGINS - i)) for i in range(N_ORIGINS)])
    n_cells = w.size  # == CELLS_PER_COHORT
    n_cohorts = n_companies * N_LOBS

    company = np.repeat([f"C{i:05d}" for i in range(n_companies)], N_LOBS * n_cells)
    lob = np.tile(np.repeat([f"lob{j}" for j in range(N_LOBS)], n_cells), n_companies)
    w_all = np.tile(w, n_cohorts)
    d_all = np.tile(d, n_cohorts)
    origin = pd.to_datetime(pd.DataFrame({"year": START_YEAR + w_all, "month": 1, "day": 1}))
    eval_date = pd.to_datetime(
        pd.DataFrame({"year": START_YEAR + w_all + d_all, "month": 12, "day": 31})
    )
    wide = pd.DataFrame(
        {
            "company": company,
            "lob": lob,
            "origin_period": origin,
            "dev_lag": (d_all + 1) * 12,
            "eval_date": eval_date,
        }
    )
    for f in FIELDS:
        wide[f] = rng.lognormal(mean=8.0, sigma=1.0, size=len(wide)).round(2)

    long = wide.melt(
        id_vars=["company", "lob", "origin_period", "dev_lag", "eval_date"],
        value_vars=FIELDS,
        var_name="field",
        value_name="value",
    )
    long["origin_period"] = long["origin_period"].dt.date
    long["eval_date"] = long["eval_date"].dt.date
    return long, wide, n_companies


def bench_size(bench: Bench, target: int, cutoff: dt.date, backends: tuple[str, ...]) -> None:
    """Generate one synthetic size and time the transform set on all three
    implementations. A function (not a loop body) so the timed lambdas bind
    this size's frames, and everything is freed before the next size."""
    long, wide, n_companies = synth(target)
    n = len(long)
    print(f"\n== target {target:,} -> {n:,} rows, {n_companies:,} companies ==")
    ds = f"{target:.0e}".replace("e+0", "e")

    for b in backends:
        bench.run(
            ds, "construct", f"ibnr[{b}]", lambda b=b: ibnr_construct(long, b, "Y", "Y", True), n
        )
    bench.run(
        ds,
        "construct",
        "chainladder",
        lambda: cl_construct(wide, FIELDS, ["company", "lob"], True),
        n,
    )

    itri = {b: ibnr_construct(long, b, "Y", "Y", True) for b in backends}
    ctri = cl_construct(wide, FIELDS, ["company", "lob"], True)

    ops = [
        ("to_incremental", lambda t: t.to_incremental().execute(), lambda: ctri.cum_to_incr()),
        (
            "as_of_slice",
            lambda t: t.as_of(cutoff).execute(),
            lambda: ctri[ctri.valuation < next_day(cutoff)],
        ),
        (
            "latest_diagonal",
            lambda t: t.latest_diagonal().execute(),
            lambda: ctri.latest_diagonal,
        ),
        (
            "aggregate_by_lob",
            lambda t: (
                t.expr.group_by(["lob", "origin_period", "dev_lag", "eval_date", "field"])
                .agg(value=ix.value.sum())
                .execute()
            ),
            lambda: ctri.groupby("lob").sum(),
        ),
    ]
    for op, ifn, cfn in ops:
        for b in backends:
            bench.run(ds, op, f"ibnr[{b}]", lambda ifn=ifn, t=itri[b]: ifn(t), n)
        bench.run(ds, op, "chainladder", cfn, n)


def main() -> None:
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore", category=UserWarning)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100_000, 300_000, 1_000_000, 3_000_000, 10_000_000],
        help="target long-row counts",
    )
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    backends = ("duckdb", "polars")
    bench = Bench(args.repeat, args.budget)
    cutoff = dt.date(START_YEAR + N_ORIGINS // 2 - 1, 12, 31)  # mid-history diagonal

    for target in args.sizes:
        bench_size(bench, target, cutoff, backends)

    # -- outputs ----------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(bench.rows)
    csv_path = args.out_dir / "benchmark_scaling.csv"
    df.to_csv(csv_path, index=False)
    meta = {
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
        "args": {"sizes": args.sizes, "repeat": args.repeat, "budget": args.budget},
        "shape": {
            "origins": N_ORIGINS,
            "devs": N_DEVS,
            "lobs": N_LOBS,
            "fields": FIELDS,
            "cells_per_cohort": CELLS_PER_COHORT,
        },
    }
    (args.out_dir / "benchmark_scaling_meta.json").write_text(json.dumps(meta, indent=2))

    table = df.pivot_table(index=["dataset", "op"], columns="impl", values="median_s", sort=False)
    order = [c for c in ("ibnr[duckdb]", "ibnr[polars]", "chainladder") if c in table.columns]
    table = table[order]
    if "chainladder" in table.columns and "ibnr[duckdb]" in table.columns:
        table["cl / duckdb"] = table["chainladder"] / table["ibnr[duckdb]"]
    print("\n### Median seconds per operation\n")
    print(markdown_table(table))

    plot(df, args.out_dir / "benchmark_scaling.png")
    print(f"\nwrote {csv_path}")


def plot(df: pd.DataFrame, path: Path) -> None:
    """Log-log median-seconds-vs-rows curves, one panel per operation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = list(dict.fromkeys(df["op"]))
    styles = {"ibnr[duckdb]": "tab:blue", "ibnr[polars]": "tab:green", "chainladder": "tab:red"}
    ncols = 3
    nrows = -(-len(ops) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), squeeze=False)
    for ax, op in zip(axes.flat, ops, strict=False):
        sub = df[df["op"] == op]
        for impl, color in styles.items():
            line = sub[sub["impl"] == impl].sort_values("rows_in")
            ax.loglog(line["rows_in"], line["median_s"], "o-", color=color, label=impl)
        ax.set_title(op)
        ax.set_xlabel("long rows")
        ax.set_ylabel("median seconds")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes.flat[len(ops) :]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("ibnr vs chainladder: scaling on synthetic Schedule-P-shaped triangles")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
