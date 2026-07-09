"""Head-to-head gallery backtest on the Schedule P gold mart.

Compares the transformer (nn_transformer) against the classical multivariate
baselines (sur, copula_glm) — and optionally meyers_ccl — on the Meyers
retrospective protocol: train as of 1997-12-31, score realized ultimates,
report outcome percentiles (KS uniformity) and CRPS per model.

Company set: companies passing the monograph's Table A.1 screens on AT
LEAST TWO requested lines, each scored on exactly its passing lines — the
same (company, line) pairs for every model, including the transformer's
training pool (no monoline data anywhere; Ethan's comparability rule). The
headline field is paid_loss — the copula's lognormal marginals cannot take
the negative late increments of reported losses.

Fit shapes per model:
  sur, copula_glm   one fit per company (its screened lines jointly)
  nn_transformer    ONE pooled fit on the screened (company, line) pairs,
                    then per-(company, line) segment predicts
  meyers_ccl        one cmdstan fit per company x line (slow; opt-in)
  chain_ladder      volume-weighted CL point estimate per (company, line) —
                    the distribution-free skill benchmark (always included)

Every row carries `anchor` (loss-to-date at the as_of date) and `premium`
(line premium), so downstream analysis can score on the RESERVE basis:
ultimate errors flatter models for data they merely copied forward.

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
ML_MODELS = ["nn_ml_ar", "nn_ml_joint"]


def screened_company_lines(
    mart: Path, lines: list[str], per_line: int
) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str]]]]:
    """(scored, pools).

    scored: company -> its screen-passing lines, for companies passing the
    Table A.1 screens on >=2 requested lines; every model scores exactly
    these (company, line) pairs. ``per_line`` caps the number of companies
    (wiring checks), 0 = all.

    pools: (company, line) training-pair pools for the transformer —
    "multiline" (the scored pairs; the default and the comparability rule),
    "screened" (>=1 passing line, i.e. + monoline companies; for comparison runs only).
    """
    sets: dict[str, set[str]] = {}
    for line in lines:
        selected = select_companies(mart, line, per_line=10_000)  # screen, don't cap yet
        sets[line] = set(selected["company_code"])
        print(f"  {line}: {len(selected)} companies pass the screens", flush=True)
    by_company: dict[str, list[str]] = {}
    for line in lines:  # keep the requested line order within each company
        for code in sets[line]:
            by_company.setdefault(code, []).append(line)
    scored = {c: ls for c, ls in sorted(by_company.items()) if len(ls) >= 2}
    if per_line:
        scored = dict(list(scored.items())[:per_line])
    multiline_pairs = [(c, ln) for c, ls in scored.items() for ln in ls]
    screened_pairs = [(c, ln) for c, ls in sorted(by_company.items()) for ln in ls]
    print(
        f"  scored: {len(scored)} companies with >=2 passing lines "
        f"({len(multiline_pairs)} pairs); screened pool adds "
        f"{len(screened_pairs) - len(multiline_pairs)} monoline pairs",
        flush=True,
    )
    return scored, {"multiline": multiline_pairs, "screened": screened_pairs}


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


def point_context(tri_all, scored, args) -> tuple[list[dict], dict, dict]:
    """(chain_ladder rows, anchors, premiums) for the scored pairs.

    anchor = loss-to-date at the as_of date (sum of each origin's latest
    observed cumulative in the training slice); premium = line premium.
    chain_ladder = volume-weighted CL point ultimate — the distribution-free
    skill benchmark every model is measured against. Keys are
    (company, line) plus (company, "ALL") sums.
    """
    train = tri_all.as_of(args.as_of)
    cum = train.select_fields(args.loss_field).execute()
    prem = train.select_fields("earned_premium").latest_diagonal().execute()
    full = tri_all.select_fields(args.loss_field).execute()
    n_d_months = int(full["dev_lag"].max())

    anchors: dict[tuple[str, str], float] = {}
    premiums: dict[tuple[str, str], float] = {}
    rows: list[dict] = []
    for code, lines_c in scored.items():
        cl_total, anchor_total, outcome_total, prem_total = 0.0, 0.0, 0.0, 0.0
        for line in lines_c:
            sub = cum[(cum["company_code"] == code) & (cum["line_of_business"] == line)]
            grid = sub.pivot_table(
                index="origin_period", columns="dev_lag", values="value"
            ).sort_index()
            devs = sorted(grid.columns)
            # volume-weighted development factors over overlapping origins
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
                # the mart carries origins beyond the study window; outcomes
                # are only the training slice's origins (same as the models)
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


def run_multiline_model(model: str, tri_all, scored, args) -> list[dict]:
    rows = []
    for i, (code, lines_c) in enumerate(scored.items(), 1):
        tri = tri_all.filter((ibis._.company_code == code) & ibis._.line_of_business.isin(lines_c))
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
            labels = {line: f"{line}/total" for line in lines_c} | {"ALL": "total"}
            per_label = line_total_rows(pred, realized, labels)
            secs = time.perf_counter() - t0
            for key, vals in per_label.items():
                rows.append(
                    {"model": model, "line": key, "company_code": code, **vals, "seconds": secs}
                )
            print(
                f"  [{i}/{len(scored)}] {model} {code} ({len(lines_c)} lines): "
                f"total pct={per_label['ALL']['percentile']:.1f} ({secs:.1f}s)",
                flush=True,
            )
        except Exception as e:  # keep the study going; record the failure
            rows.append({"model": model, "line": "ALL", "company_code": code, "error": str(e)})
            print(f"  [{i}/{len(scored)}] {model} {code}: FAILED {e}", flush=True)
    return rows


def pair_filter(tri, pairs: list[tuple[str, str]]):
    """Filter a triangle to exact (company, line) pairs via a concat key."""
    keys = [f"{c}|{ln}" for c, ln in pairs]
    return tri.filter((ibis._.company_code + "|" + ibis._.line_of_business).isin(keys))


def run_transformer(tri_market, scored, pools, args) -> list[dict]:
    # default pool: exactly the scored (company, line) pairs — the models see
    # identical cohort data; "screened" (+monoline pairs) and "market" (every
    # cohort) are alternate pools, run only to measure how much the pool
    # choice matters. The pinned-dev mean is a pooled statistic, so pool
    # hygiene is load-bearing.
    from ibnr.gallery.nn.transformer import TransformerConfig

    tri_pool = (
        tri_market if args.nn_pool == "market" else pair_filter(tri_market, pools[args.nn_pool])
    )
    t0 = time.perf_counter()
    entry = gallery.fit(
        "nn_transformer",
        tri_pool,
        loss_field=args.loss_field,
        feature_fields=tuple(args.nn_features),
        as_of=args.as_of,
        seed=args.seed,
        config=TransformerConfig(exposure_sigma=args.nn_exposure_sigma),
    )
    fit_secs = time.perf_counter() - t0
    n_cohorts = len(entry.contract_["cohorts"])
    print(f"  nn_transformer: pooled fit on {n_cohorts} cohorts ({fit_secs:.0f}s)", flush=True)
    if args.nn_exposure_sigma:
        import torch

        ps = [float(torch.nn.functional.softplus(m.raw_p)) for m in entry.models_]
        print(
            f"  exposure power p per member: {[round(p, 3) for p in ps]} "
            f"(mean {sum(ps) / len(ps):.3f}; 1.0 = flat constant-CV baseline, "
            "<1 tightens large books & widens small)",
            flush=True,
        )

    rows = []
    for code, lines_c in scored.items():
        for line in lines_c:
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


def run_transformer_ml(tri_market, scored, pools, args, dependence: str) -> list[dict]:
    """The multi-line transformer: one pooled company-level fit (attention
    across each company's screened lines), then per-company predicts in the
    SUR layout — per-line totals AND the diversified grand total."""
    from ibnr.gallery.nn.transformer_ml import TransformerMLConfig

    name = f"nn_ml_{dependence}"
    tri_pool = (
        tri_market if args.nn_pool == "market" else pair_filter(tri_market, pools[args.nn_pool])
    )
    t0 = time.perf_counter()
    entry = gallery.fit(
        "nn_transformer_ml",
        tri_pool,
        loss_field=args.loss_field,
        feature_fields=tuple(args.nn_features),
        as_of=args.as_of,
        seed=args.seed,
        config=TransformerMLConfig(dependence=dependence),
    )
    n_companies = len(entry.contract_["companies"])
    print(
        f"  {name}: pooled fit on {n_companies} companies ({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

    rows = []
    for i, (code, lines_c) in enumerate(scored.items(), 1):
        t1 = time.perf_counter()
        try:
            pred = entry.predict(
                segment={"company_code": code}, n_draws=args.nn_draws, seed=args.seed
            )
            realized = entry.realized_ultimates(tri_market, segment={"company_code": code})
            labels = {line: f"{line}/total" for line in lines_c} | {"ALL": "total"}
            per_label = line_total_rows(pred, realized, labels)
            secs = time.perf_counter() - t1
            for key, vals in per_label.items():
                rows.append(
                    {"model": name, "line": key, "company_code": code, **vals, "seconds": secs}
                )
            if i % 10 == 0 or i == len(scored):
                print(f"  [{i}/{len(scored)}] {name} {code}", flush=True)
        except Exception as e:
            rows.append({"model": name, "line": "ALL", "company_code": code, "error": str(e)})
            print(f"  {name} {code}: FAILED {e}", flush=True)
    return rows


def run_meyers(tri_by_line, scored, lines, args) -> list[dict]:
    rows = []
    for line in lines:
        codes = [c for c, ls in scored.items() if line in ls]
        for i, code in enumerate(codes, 1):
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
                    f"  [{i}/{len(codes)}] meyers_ccl {line} {code}: "
                    f"pct={per_label[line]['percentile']:.1f}",
                    flush=True,
                )
            except Exception as e:
                rows.append(
                    {"model": "meyers_ccl", "line": line, "company_code": code, "error": str(e)}
                )
                print(f"  meyers_ccl {line} {code}: FAILED {e}", flush=True)
    return rows


def point_summary(df: pd.DataFrame) -> None:
    """Distribution-free comparison on the RESERVE basis: est/actual reserve
    = (estimate|outcome) - anchor. Reports error levels, the chain-ladder
    skill ratio, and paired Wilcoxon tests (all models score identical
    cells). Premium-normalized MAE is the robust headline — actual reserves
    can sit near zero (late favorable development), which blows up APEs."""
    from scipy import stats

    ok = df[df["estimate"].notna() & (df["line"] != "ALL")].copy()
    ok["res_est"] = ok["estimate"] - ok["anchor"]
    ok["res_act"] = ok["outcome"] - ok["anchor"]
    ok["abs_err"] = (ok["res_est"] - ok["res_act"]).abs()
    ok["err_premium_pct"] = (ok["res_est"] - ok["res_act"]) / ok["premium"] * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        ok["res_ape"] = ok["abs_err"] / ok["res_act"].abs()

    print("\n== Point prediction: reserve-basis errors ==")
    agg = ok.groupby("model").agg(
        n=("abs_err", "size"),
        mae_prem_pct=("err_premium_pct", lambda s: s.abs().mean()),
        bias_prem_pct=("err_premium_pct", "mean"),
        mdape_reserve=("res_ape", "median"),
    )
    # CL skill on the common cells: MAE_model / MAE_chain_ladder
    cells = ok.pivot_table(index=["line", "company_code"], columns="model", values="abs_err")
    if "chain_ladder" in cells.columns:
        agg["cl_skill"] = cells.mean() / cells["chain_ladder"].mean()
    print(agg.round(3).to_string())

    models = [m for m in cells.columns if m != "chain_ladder"]
    print("\n== Paired Wilcoxon on |reserve error| (p-values; < means row beats col) ==")
    order = [*models, *(["chain_ladder"] if "chain_ladder" in cells.columns else [])]
    for a in order:
        parts = []
        for b in order:
            if a == b:
                parts.append("      -")
                continue
            pair = cells[[a, b]].dropna()
            diff = pair[a] - pair[b]
            if len(diff) < 6 or (diff == 0).all():
                parts.append("     na")
                continue
            p = stats.wilcoxon(diff).pvalue
            marker = "<" if diff.median() < 0 else ">"
            parts.append(f"{marker}{p:6.3f}")
        print(f"  {a:<16}" + " ".join(parts))
    print(f"  {'':16}" + " ".join(f"{m[:7]:>7}" for m in order))


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
    ap.add_argument(
        "--warehouse",
        default=DEFAULT_WAREHOUSE,
        help="local warehouse path or github://owner/repo@publish_id "
        "(default: the latest GitHub release)",
    )
    ap.add_argument("--lines", nargs="+", default=MEYERS_LINES, choices=MEYERS_LINES)
    ap.add_argument("--per-line", type=int, default=0, help="cap companies (0 = all passing)")
    ap.add_argument(
        "--models",
        nargs="+",
        default=FAST_MODELS,
        choices=[*FAST_MODELS, *ML_MODELS, "meyers_ccl"],
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
    ap.add_argument(
        "--nn-pool",
        default="multiline",
        choices=["multiline", "screened", "market"],
        help="transformer training pool: companies passing screens on >=2 "
        "lines (default), >=1 line, or every company in the mart",
    )
    ap.add_argument(
        "--nn-exposure-sigma",
        action="store_true",
        help="single-line transformer only: learn a premium power p so the "
        "predictive dollar sd scales as premium**p (p=1 is the flat-sigma "
        "baseline). Off = baseline; on/off arms are directly comparable.",
    )
    ap.add_argument("--chains", type=int, default=4, help="meyers_ccl chains")
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parents[1] / "analysis" / "results" / "compare_gallery.csv",
    )
    args = ap.parse_args()

    mart = active_mart_path(args.warehouse)
    print("selecting companies (screens per line):", flush=True)
    scored, pools = screened_company_lines(mart, args.lines, args.per_line)
    if not scored:
        print("no companies pass the screens on >=2 lines")
        return 1

    tri_market = load_schedule_p(args.warehouse, lines=args.lines)
    print("\nchain_ladder point benchmark + anchors/premiums", flush=True)
    all_rows, anchors, premiums = point_context(tri_market, scored, args)

    for model in ("sur", "copula_glm"):
        if model in args.models:
            print(f"\n{model}: one fit per company, its screened lines jointly", flush=True)
            all_rows += run_multiline_model(model, tri_market, scored, args)

    if "nn_transformer" in args.models:
        print(f"\nnn_transformer: pooled fit, --nn-pool {args.nn_pool}", flush=True)
        all_rows += run_transformer(tri_market, scored, pools, args)

    for dep in ("ar", "joint"):
        if f"nn_ml_{dep}" in args.models:
            print(f"\nnn_ml_{dep}: pooled company fit ({dep} dependence head)", flush=True)
            all_rows += run_transformer_ml(tri_market, scored, pools, args, dep)

    if "meyers_ccl" in args.models:
        print("\nmeyers_ccl: one cmdstan fit per company x line", flush=True)
        tri_by_line = {line: load_schedule_p(args.warehouse, lines=[line]) for line in args.lines}
        all_rows += run_meyers(tri_by_line, scored, args.lines, args)

    df = pd.DataFrame(all_rows)
    keys = list(zip(df["company_code"], df["line"], strict=True))
    df["anchor"] = [anchors.get(k, np.nan) for k in keys]
    df["premium"] = [premiums.get(k, np.nan) for k in keys]
    # provenance: every row traces to an exact gold publish and loss field
    df["mart_publish_id"] = active_publish_id(args.warehouse)
    df["loss_field"] = args.loss_field
    args.out.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.loss_field}" if args.loss_field != "paid_loss" else ""
    out = args.out.with_stem(args.out.stem + suffix)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    summarize(df)
    point_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
