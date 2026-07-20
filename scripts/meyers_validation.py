"""Meyers retrospective validation of the Meyers-family entries on the
Schedule P gold mart.

Reproduces the monograph's protocol: per line of business, mechanically select
up to 50 stable insurers (appendix criteria), fit the model to the upper
triangle as of 1997-12-31, record the predictive percentile of the realized
total ultimate, and test the percentiles for uniformity (KS / p-p).

Models: ``meyers_ccl`` fits Meyers' "incurred" = reported_loss (incurred net
of bulk+IBNR); ``meyers_csr`` fits paid_loss with Meyers' floor-of-1 clamp
(his ``pmax(cum_pdloss, 1)``) so the paid study keeps the identical company
cohort as the incurred one. Company selection applies both Table A.1 screens
(CV1 net premium, CV2 net/direct premium ratio) for every model.

Usage:
    uv run python scripts/meyers_validation.py --per-line 50
    uv run python scripts/meyers_validation.py --model meyers_csr --per-line 50
    uv run python scripts/meyers_validation.py --lines workers_compensation --per-line 5 --chains 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import ibis
import pandas as pd

from ibnr import gallery
from ibnr.data.schedule_p import active_mart_path, load_schedule_p
from ibnr.kernels.calibration import ks_uniformity

# None falls through to ibnr's resolution: IBNR_SCHEDULE_P_WAREHOUSE env var,
# else the GitHub release default (github://...@latest, cached locally).
DEFAULT_WAREHOUSE = None

#: literature pairing: CCL scores incurred (net of bulk); CSR, the ODP
#: (whose chain-ladder equivalence is a paid result) and Clark score paid
MODEL_LOSS_FIELDS = {
    "meyers_ccl": "reported_loss",
    "meyers_csr": "paid_loss",
    "england_verrall_odp": "paid_loss",
    "clark": "paid_loss",
    "clark_growth_curve": "paid_loss",
}

#: models fit by MCMC — they take the chains/warmup/draws arguments and
#: report R-hat; the likelihood-based entries (clark) take neither
MCMC_MODELS = {"meyers_ccl", "meyers_csr", "england_verrall_odp", "clark_growth_curve"}

#: models needing Meyers' pmax(cum_pdloss, 1) floor — the lognormal cannot
#: take non-positive cells. The ODP takes zeros natively and must see the
#: unclamped data (a clamp would silently alter increments); its negative-
#: increment failures are recorded, exactly like the bootstrap ODP's.
MODELS_WITH_PAID_CLAMP = {"meyers_csr"}

MEYERS_LINES = [
    "commercial_auto",
    "private_passenger_auto",
    "workers_compensation",
    "other_liability",
]

#: Table A.1 limits: CV1 on net earned premium, CV2 on the net/direct premium ratio
CV1_LIMITS = {
    "commercial_auto": 0.795,
    "private_passenger_auto": 1.003,
    "workers_compensation": 0.772,
    "other_liability": 0.628,
}
CV2_LIMITS = {
    "commercial_auto": 0.125,
    "private_passenger_auto": 0.125,
    "workers_compensation": 0.300,
    "other_liability": 0.15,
}

EXCLUDED_GROUPS = {"38997"}  # excluded by Meyers after provisional testing

TRAIN_AYS = (1988, 1997)
# Monograph appendix: "minimum annual premium of greater than $20,000 and
# minimum annual incurred loss of greater than $4,000", with Schedule P
# entries in $1,000s — i.e. 20 and 4 in data units. (Reading them as $20M/$4M
# leaves only ~10 WC companies, far short of Meyers' 50; this reading leaves
# 72 passing the WC CV screen, of which he took the top 50.)
MIN_PREMIUM = 20.0
MIN_LOSS = 4.0


def select_companies(mart_path: Path, line: str, per_line: int) -> pd.DataFrame:
    """Mechanical selection per the monograph appendix (incl. the CV2 screen)."""
    q = f"""
    with base as (
        select company_code, accident_year, development_age, statement_year,
               incurred_loss - bulk_loss as reported_loss,
               earned_prem_net, earned_prem_direct
        from read_parquet('{mart_path.as_posix()}')
        where line_of_business = '{line}'
          and accident_year between {TRAIN_AYS[0]} and {TRAIN_AYS[1]}
    ),
    square as (
        select company_code,
               count(*) as n_cells,
               -- positivity needed only where the lognormal model trains
               min(case when statement_year <= 1997 then reported_loss end) as min_train_cell
        from base group by 1
    ),
    prem as (  -- premium and latest-diagonal loss per AY as booked in the 1997 statement
        select company_code,
               min(earned_prem_net) as min_prem,
               min(reported_loss) as min_ay_loss,
               stddev_samp(earned_prem_net) / avg(earned_prem_net) as cv1,
               stddev_samp(earned_prem_net / nullif(earned_prem_direct, 0))
                   / avg(earned_prem_net / nullif(earned_prem_direct, 0)) as cv2
        from base where statement_year = 1997 group by 1
    )
    select s.company_code, p.cv1, p.cv2
    from square s join prem p using (company_code)
    where s.n_cells = 100            -- complete 10x10 square incl. outcomes
      and s.min_train_cell > 0       -- lognormal needs positive training cells
      and p.min_prem > {MIN_PREMIUM}
      and p.min_ay_loss > {MIN_LOSS}
      and p.cv1 < {CV1_LIMITS[line]}
      and p.cv2 < {CV2_LIMITS[line]}
    order by p.cv1 asc
    """
    out = duckdb.sql(q).df()
    out = out[~out["company_code"].isin(EXCLUDED_GROUPS)]
    return out.head(per_line).reset_index(drop=True)


def run_line(
    warehouse: Path, line: str, companies: pd.DataFrame, args: argparse.Namespace
) -> list[dict]:
    tri_line = load_schedule_p(warehouse, lines=[line])
    loss_field = args.loss_field or MODEL_LOSS_FIELDS[args.model]
    if loss_field == "paid_loss" and args.model in MODELS_WITH_PAID_CLAMP:
        # Meyers' pmax(cum_pdloss, 1): floor paid cells at 1 (in $000s) so the
        # lognormal accepts every cohort the incurred screens admit
        e = tri_line.expr
        tri_line = tri_line.with_expr(
            e.mutate(value=ibis.ifelse(e.field == "paid_loss", ibis.greatest(e.value, 1), e.value))
        )
    rows = []
    for i, code in enumerate(companies["company_code"], 1):
        tri = tri_line.filter(ibis._.company_code == code)
        t0 = time.perf_counter()
        try:
            fit_kwargs = (
                {
                    "chains": args.chains,
                    "iter_warmup": args.warmup,
                    "iter_sampling": args.draws,
                    "seed": args.seed,
                }
                if args.model in MCMC_MODELS
                else {}
            )
            entry = gallery.fit(
                args.model,
                tri,
                loss_field=loss_field,
                as_of="1997-12-31",
                **fit_kwargs,
            )
            pred = entry.predict(seed=args.seed)
            realized = entry.realized_ultimates(tri)
            table = pred.summary(observed=realized)
            total = table.iloc[-1]
            rhat = (
                entry.fit_.summary()["R_hat"].max() if args.model in MCMC_MODELS else float("nan")
            )
            rows.append(
                {
                    "line": line,
                    "company_code": code,
                    "estimate": total["estimate"],
                    "se": total["se"],
                    "cv": total["cv"],
                    "outcome": total["outcome"],
                    "percentile": total["percentile"],
                    "max_rhat": rhat,
                    "seconds": time.perf_counter() - t0,
                }
            )
            print(
                f"  [{i}/{len(companies)}] {line} {code}: pct={total['percentile']:.1f} "
                f"rhat={rhat:.3f} ({rows[-1]['seconds']:.1f}s)",
                flush=True,
            )
        except Exception as e:  # keep the study going; record the failure
            rows.append({"line": line, "company_code": code, "error": str(e)})
            print(f"  [{i}/{len(companies)}] {line} {code}: FAILED {e}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--warehouse",
        default=DEFAULT_WAREHOUSE,
        help="local warehouse path or github://owner/repo@publish_id "
        "(default: the latest GitHub release)",
    )
    ap.add_argument("--model", default="meyers_ccl", choices=sorted(MODEL_LOSS_FIELDS))
    ap.add_argument(
        "--loss-field",
        default=None,
        help=f"override the model's monograph loss field (defaults: {MODEL_LOSS_FIELDS})",
    )
    ap.add_argument("--lines", nargs="+", default=MEYERS_LINES, choices=MEYERS_LINES)
    ap.add_argument("--per-line", type=int, default=50)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = (
            Path(__file__).parents[1] / "analysis" / "results" / f"{args.model}_validation.csv"
        )

    mart = active_mart_path(args.warehouse)
    all_rows: list[dict] = []
    for line in args.lines:
        companies = select_companies(mart, line, args.per_line)
        print(f"{line}: {len(companies)} companies selected", flush=True)
        all_rows += run_line(args.warehouse, line, companies, args)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    ok = df[df.get("percentile").notna()] if "percentile" in df else pd.DataFrame()
    if len(ok) >= 5:
        print("\nUniformity of total-outcome percentiles (Meyers p-p test):")
        for line, grp in ok.groupby("line"):
            print(f"  {line:<24} n={len(grp):>3}  {ks_uniformity(grp['percentile'] / 100)}")
        print(f"  {'ALL':<24} n={len(ok):>3}  {ks_uniformity(ok['percentile'] / 100)}")
        report = {
            "n": len(ok),
            "ks_all": ks_uniformity(ok["percentile"] / 100).statistic,
            "failed": int(df["percentile"].isna().sum()) if "percentile" in df else 0,
        }
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
