"""Head-to-head gallery backtest on the Schedule P gold mart.

Compares the transformer (nn_transformer) against the classical multivariate
baselines (sur, copula_glm) — and optionally meyers_ccl — on the Meyers
retrospective protocol: train as of 1997-12-31, score realized ultimates,
report outcome percentiles (KS uniformity) and CRPS per model.

Company set: companies passing the monograph's Table A.1 screens on EVERY
requested line (the multiline models need all lines at once), so every model
is scored on the identical (company, line) pairs. The headline field is
paid_loss — the copula's lognormal marginals cannot take the negative late
increments of reported losses.

Fit shapes per model:
  sur, copula_glm   one fit per company (all lines jointly)
  nn_transformer    ONE pooled fit on every company x line in the mart,
                    then per-(company, line) segment predicts
  meyers_ccl        one cmdstan fit per company x line (slow; opt-in)

Usage:
    uv run python scripts/compare_gallery.py --per-line 5           # wiring check
    uv run python scripts/compare_gallery.py                        # full fast trio
    uv run python scripts/compare_gallery.py --models sur copula_glm nn_transformer meyers_ccl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ibis
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from meyers_validation import DEFAULT_WAREHOUSE, MEYERS_LINES, select_companies  # noqa: E402

from ibnr import gallery  # noqa: E402
from ibnr.data.schedule_p import active_mart_path, active_publish_id, load_schedule_p  # noqa: E402
from ibnr.kernels.calibration import ks_uniformity  # noqa: E402
from ibnr.kernels.scores import crps  # noqa: E402

FAST_MODELS = ["sur", "copula_glm", "nn_transformer"]


def multiline_company_set(mart: Path, lines: list[str], per_line: int) -> list[str]:
    """Companies passing the Table A.1 screens on every requested line."""
    sets = []
    for line in lines:
        selected = select_companies(mart, line, per_line=10_000)  # screen, don't cap yet
        sets.append(set(selected["company_code"]))
        print(f"  {line}: {len(selected)} companies pass the screens", flush=True)
    common = sorted(set.intersection(*sets))
    print(f"  intersection across {len(lines)} lines: {len(common)} companies", flush=True)
    return common[:per_line] if per_line else common


def line_total_rows(pred, realized, label_map: dict[str, str]) -> dict[str, dict]:
    """Extract per-label rows (estimate/se/cv/outcome/percentile/crps) from a
    multiline predictive distribution."""
    table = pred.summary(observed=realized)
    scores = crps(pred.samples, realized)
    out = {}
    for key, label in label_map.items():
        i = table.index[table["label"] == label][0]
        row = table.loc[i]
        out[key] = {
            "estimate": row["estimate"],
            "se": row["se"],
            "cv": row["cv"],
            "outcome": row["outcome"],
            "percentile": row["percentile"],
            "crps": scores[i],
        }
    return out


def run_multiline_model(model: str, tri_all, companies, lines, args) -> list[dict]:
    rows = []
    for i, code in enumerate(companies, 1):
        tri = tri_all.filter(ibis._.company_code == code)
        t0 = time.perf_counter()
        try:
            extra = {"nonpositive": args.copula_nonpositive} if model == "copula_glm" else {}
            try:
                entry = gallery.fit(
                    model, tri, loss_field=args.loss_field, as_of=args.as_of, **extra
                )
            except ValueError as e:
                # dropped cells can deplete late dev steps; the Hoerl curve is
                # the documented reduced marginal for exactly this case
                if model != "copula_glm" or "hoerl" not in str(e):
                    raise
                entry = gallery.fit(
                    model,
                    tri,
                    loss_field=args.loss_field,
                    as_of=args.as_of,
                    dev_effect="hoerl",
                    **extra,
                )
                print(f"  {model} {code}: factor marginal unidentified, using hoerl", flush=True)
            pred = entry.predict(n_draws=args.draws, seed=args.seed)
            realized = entry.realized_ultimates(tri)
            labels = {line: f"{line}/total" for line in lines} | {"ALL": "total"}
            per_label = line_total_rows(pred, realized, labels)
            secs = time.perf_counter() - t0
            for key, vals in per_label.items():
                rows.append(
                    {"model": model, "line": key, "company_code": code, **vals, "seconds": secs}
                )
            print(
                f"  [{i}/{len(companies)}] {model} {code}: "
                f"total pct={per_label['ALL']['percentile']:.1f} ({secs:.1f}s)",
                flush=True,
            )
        except Exception as e:  # keep the study going; record the failure
            rows.append({"model": model, "line": "ALL", "company_code": code, "error": str(e)})
            print(f"  [{i}/{len(companies)}] {model} {code}: FAILED {e}", flush=True)
    return rows


def run_transformer(tri_market, companies, lines, args) -> list[dict]:
    t0 = time.perf_counter()
    entry = gallery.fit(
        "nn_transformer",
        tri_market,
        loss_field=args.loss_field,
        feature_fields=tuple(args.nn_features),
        as_of=args.as_of,
        seed=args.seed,
    )
    fit_secs = time.perf_counter() - t0
    n_cohorts = len(entry.contract_["cohorts"])
    print(f"  nn_transformer: pooled fit on {n_cohorts} cohorts ({fit_secs:.0f}s)", flush=True)

    rows = []
    for line in lines:
        for code in companies:
            seg = {"company_code": code, "line_of_business": line}
            t1 = time.perf_counter()
            try:
                pred = entry.predict(segment=seg, n_draws=args.nn_draws, seed=args.seed)
                realized = entry.realized_ultimates(tri_market, segment=seg)
                per_label = line_total_rows(pred, realized, {line: "total"})
                rows.append(
                    {
                        "model": "nn_transformer",
                        "line": line,
                        "company_code": code,
                        **per_label[line],
                        "seconds": time.perf_counter() - t1,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "model": "nn_transformer",
                        "line": line,
                        "company_code": code,
                        "error": str(e),
                    }
                )
                print(f"  nn_transformer {line} {code}: FAILED {e}", flush=True)
    return rows


def run_meyers(tri_by_line, companies, lines, args) -> list[dict]:
    rows = []
    for line in lines:
        for i, code in enumerate(companies, 1):
            tri = tri_by_line[line].filter(ibis._.company_code == code)
            t0 = time.perf_counter()
            try:
                entry = gallery.fit(
                    "meyers_ccl",
                    tri,
                    loss_field=args.loss_field,
                    as_of=args.as_of,
                    chains=args.chains,
                    seed=args.seed,
                    iter_sampling=args.draws // args.chains,
                )
                pred = entry.predict(seed=args.seed)
                realized = entry.realized_ultimates(tri)
                per_label = line_total_rows(pred, realized, {line: "total"})
                rows.append(
                    {
                        "model": "meyers_ccl",
                        "line": line,
                        "company_code": code,
                        **per_label[line],
                        "seconds": time.perf_counter() - t0,
                    }
                )
                print(
                    f"  [{i}/{len(companies)}] meyers_ccl {line} {code}: "
                    f"pct={per_label[line]['percentile']:.1f}",
                    flush=True,
                )
            except Exception as e:
                rows.append(
                    {"model": "meyers_ccl", "line": line, "company_code": code, "error": str(e)}
                )
                print(f"  meyers_ccl {line} {code}: FAILED {e}", flush=True)
    return rows


def summarize(df: pd.DataFrame) -> None:
    ok = df[df["percentile"].notna() & (df["line"] != "ALL")]
    print("\n== Calibration: KS uniformity of line-total outcome percentiles ==")
    for model, grp in ok.groupby("model"):
        print(f"\n{model}:")
        for line, sub in grp.groupby("line"):
            print(f"  {line:<24} n={len(sub):>3}  {ks_uniformity(sub['percentile'] / 100)}")
        print(f"  {'COMBINED':<24} n={len(grp):>3}  {ks_uniformity(grp['percentile'] / 100)}")

    print("\n== Sharpness: mean CRPS / outcome (lower is better) ==")
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = ok.assign(rel_crps=ok["crps"] / ok["outcome"])
    pivot = ok.pivot_table(index="line", columns="model", values="rel_crps", aggfunc="mean")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    grand = df[(df["line"] == "ALL") & df["percentile"].notna()]
    if len(grand):
        print("\n== Diversified grand totals (multiline models) ==")
        for model, grp in grand.groupby("model"):
            print(f"  {model:<12} n={len(grp):>3}  {ks_uniformity(grp['percentile'] / 100)}")

    failed = df[df.get("error").notna()] if "error" in df else pd.DataFrame()
    if len(failed):
        print(f"\n{len(failed)} (model, company) fits failed:")
        print(failed.groupby(["model"])["error"].count().to_string())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    ap.add_argument("--lines", nargs="+", default=MEYERS_LINES, choices=MEYERS_LINES)
    ap.add_argument("--per-line", type=int, default=0, help="cap companies (0 = all passing)")
    ap.add_argument(
        "--models",
        nargs="+",
        default=FAST_MODELS,
        choices=[*FAST_MODELS, "meyers_ccl"],
    )
    ap.add_argument("--loss-field", default="paid_loss")
    ap.add_argument(
        "--copula-nonpositive",
        default="drop",
        choices=["drop", "error"],
        help="Schedule P paid still dips at late lags (salvage); drop keeps the "
        "copula in the study at a documented left-tail bias",
    )
    ap.add_argument("--as-of", default="1997-12-31")
    ap.add_argument("--draws", type=int, default=10_000, help="statistical model draws")
    ap.add_argument("--nn-draws", type=int, default=1000)
    ap.add_argument("--nn-features", nargs="*", default=["reported_loss"])
    ap.add_argument("--chains", type=int, default=4, help="meyers_ccl chains")
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parents[1] / "analysis" / "results" / "compare_gallery.csv",
    )
    args = ap.parse_args()

    mart = active_mart_path(args.warehouse)
    print("selecting companies (screens on every line):", flush=True)
    companies = multiline_company_set(mart, args.lines, args.per_line)
    if not companies:
        print("no companies pass the screens on every line")
        return 1

    all_rows: list[dict] = []
    tri_market = load_schedule_p(args.warehouse, lines=args.lines)

    for model in ("sur", "copula_glm"):
        if model in args.models:
            print(f"\n{model}: one fit per company, {len(args.lines)} lines jointly", flush=True)
            all_rows += run_multiline_model(model, tri_market, companies, args.lines, args)

    if "nn_transformer" in args.models:
        print("\nnn_transformer: pooled market fit", flush=True)
        all_rows += run_transformer(tri_market, companies, args.lines, args)

    if "meyers_ccl" in args.models:
        print("\nmeyers_ccl: one cmdstan fit per company x line", flush=True)
        tri_by_line = {line: load_schedule_p(args.warehouse, lines=[line]) for line in args.lines}
        all_rows += run_meyers(tri_by_line, companies, args.lines, args)

    df = pd.DataFrame(all_rows)
    # provenance: every row traces to an exact gold publish and loss field
    df["mart_publish_id"] = active_publish_id(args.warehouse)
    df["loss_field"] = args.loss_field
    args.out.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.loss_field}" if args.loss_field != "paid_loss" else ""
    out = args.out.with_stem(args.out.stem + suffix)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    summarize(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
