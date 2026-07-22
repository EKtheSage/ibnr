"""Meyers retrospective validation of the Meyers-family entries on the
Schedule P gold mart.

Reproduces the monograph's protocol: per line of business, mechanically select
up to 50 stable insurers (appendix criteria), fit the model to the upper
triangle as of 1997-12-31, record the predictive percentile of the realized
total ultimate, and test the percentiles for uniformity (KS / p-p).

Models: ``meyers_ccl`` fits Meyers' "incurred" = reported_loss (incurred net
of bulk+IBNR); ``meyers_csr`` fits paid_loss with Meyers' floor-of-1 clamp
(his ``pmax(cum_pdloss, 1)``) so the paid study keeps the identical company
cohort as the incurred one. ``compartmental`` scores paid but fits paid and
outstanding (= reported - paid) jointly, unclamped — its gaussian default
takes zero cells natively and the lognormal variant drops and counts its
own non-positive cells. Company selection applies both Table A.1 screens
(CV1 net premium, CV2 net/direct premium ratio) for every model.

METHODOLOGY (this script produces published results — read before changing)

1. Cohort-data parity. `select_companies` runs the identical mechanical
   screen for every model, so any two entries validated here are scored on
   the same insurers. That is why `meyers_csr` carries Meyers' floor-of-1
   paid clamp: without it the lognormal would reject cohorts the incurred
   study kept, and the two studies would no longer be comparable. Model
   differences must come from the models, not from who they were allowed
   to see. `compare_gallery.py` reuses `select_companies` for the same
   reason.

2. Training slice vs realized outcomes. Each entry fits
   `as_of="1997-12-31"` — the upper triangle a reserving actuary could have
   seen at year-end 1997 — while the outcome is the ultimate realized in
   later statement years, read from the FULL triangle by
   `entry.realized_ultimates(tri)`. CRITICAL GOTCHA (CLAUDE.md): the mart
   carries accident years past the 1988-1997 study window, so any aggregate
   taken from the full triangle MUST be restricted to the origins present in
   the training slice, or it silently sums post-study accident years (this
   bit `compare_gallery.point_context` once and produced 2.4x inflated
   outcomes). Here the restriction is enforced twice over: the selection SQL
   only admits companies with a complete 10x10 square over
   `TRAIN_AYS` = 1988-1997, and `realized_ultimates()` scores exactly the
   origins the entry trained on.

3. Scoring. Only the TOTAL row of the summary table is kept (`table.iloc[-1]`
   = the sum over accident years), matching the monograph, which validates
   the distribution of total unpaid loss rather than individual cells. Its
   `percentile` is the PIT value of the realized total; uniformity of those
   percentiles across companies is tested with KS at 5% (critical value
   1.36/sqrt(n) — 19.2 at n=50, 9.6 at n=200; `kernels.calibration` prints
   `D x 100` and flags rejection with `*`, Meyers' own convention).

4. Failure handling. A fit that raises records an `error` row and the study
   continues; the JSON report prints a `failed` count. Failures are visible
   in the CSV but drop out of the KS panel, so a model that fit fewer
   companies is calibrated on a smaller (and probably easier) panel — always
   read `n` and `failed` together. The one silent data alteration is the
   paid clamp, restricted to `MODELS_WITH_PAID_CLAMP` and documented there.

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
    # compartmental scores paid but also consumes reported (outstanding =
    # reported - paid) through its own joint contract
    "compartmental": "paid_loss",
}

#: models fit by MCMC — they take the chains/warmup/draws arguments and
#: report R-hat; the likelihood-based entries (clark) take neither
MCMC_MODELS = {
    "meyers_ccl",
    "meyers_csr",
    "england_verrall_odp",
    "clark_growth_curve",
    "compartmental",
}

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

#: the monograph's study window: accident years 1988-1997, so the as_of
#: 1997-12-31 slice is a full 10x10 upper triangle and the realized lower half
#: is available in later statement years. Any origin outside this range is
#: post-study and must never enter an outcome aggregate (see module docstring).
TRAIN_AYS = (1988, 1997)
# Monograph appendix: "minimum annual premium of greater than $20,000 and
# minimum annual incurred loss of greater than $4,000", with Schedule P
# entries in $1,000s — i.e. 20 and 4 in data units. (Reading them as $20M/$4M
# leaves only ~10 WC companies, far short of Meyers' 50; this reading leaves
# 72 passing the WC CV screen, of which he took the top 50.)
MIN_PREMIUM = 20.0
MIN_LOSS = 4.0


def select_companies(mart_path: Path, line: str, per_line: int) -> pd.DataFrame:
    """Mechanical selection per the monograph appendix (incl. the CV2 screen).

    Deliberately mechanical: no judgement, no per-model tuning, so the cohort
    is reproducible and identical across every entry validated here (the
    cohort-data parity rule). The screens select insurers whose book was
    STABLE over the study window — Meyers' point is that a reserving model
    should be tested where the data is well behaved, not where growth or
    reinsurance churn confounds development.

    - CV1: coefficient of variation of net earned premium across accident
      years — rejects rapidly growing/shrinking books.
    - CV2: CV of the net/direct premium ratio — rejects books whose
      reinsurance program changed materially over the window.
    - complete 10x10 square: the company must have all 100 cells, so the
      realized outcome exists for every training origin.

    Rows come back ordered by cv1 ascending; ``per_line`` then takes the top
    n (Meyers' "top 50"), so the cap is deterministic, not a random sample.
    """
    # Runs against the mart parquet directly rather than through the Triangle
    # layer: this is cohort selection, not modelling, and it needs raw
    # Schedule P columns (bulk_loss, direct premium) the triangle does not carry.
    #
    # Notes on the query below:
    #   * `incurred_loss - bulk_loss` is Meyers' "incurred", NET of bulk+IBNR.
    #     That definition is load-bearing, not cosmetic: gross-of-bulk incurred
    #     fails the WC KS test badly (D=36.7). See CLAUDE.md milestone 2.
    #   * `n_cells = 100` is 10 accident years x 10 development ages — anything
    #     less means some training origin has no realized outcome to score.
    #   * the `prem` CTE reads statement_year = 1997 only: premium and the
    #     latest-diagonal loss as BOOKED at the as_of date, so selection uses
    #     no information from after the training cutoff.
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
    """Fit + score one line of business, one company at a time.

    Returns one row per company: the predictive total (estimate/se/cv), the
    realized outcome, its percentile (the PIT value fed to the KS test), the
    worst R-hat of the fit, and wall-clock seconds. Companies whose fit raises
    yield an ``error`` row instead — the study runs to completion either way.
    """
    # the FULL triangle for this line; each entry slices to as_of internally
    # and scores only the origins it trained on
    tri_line = load_schedule_p(warehouse, lines=[line])
    loss_field = args.loss_field or MODEL_LOSS_FIELDS[args.model]
    if loss_field == "paid_loss" and args.model in MODELS_WITH_PAID_CLAMP:
        # Meyers' pmax(cum_pdloss, 1): floor paid cells at 1 (in $000s) so the
        # lognormal accepts every cohort the incurred screens admit.
        # This is the ONE place the harness alters the data. It is confined to
        # MODELS_WITH_PAID_CLAMP because a clamp shifts increments; the ODP and
        # compartmental entries must see the unclamped series.
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
            if args.growth_curve is not None:
                if args.model not in ("clark", "clark_growth_curve"):
                    raise SystemExit(f"--growth-curve does not apply to {args.model}")
                fit_kwargs["growth_curve"] = args.growth_curve
            if args.variant is not None:
                if args.model != "compartmental":
                    raise SystemExit(f"--variant does not apply to {args.model}")
                fit_kwargs["variant"] = args.variant
            # the monograph's cutoff: train on the diagonal visible at
            # year-end 1997, score against what later statements realized
            entry = gallery.fit(
                args.model,
                tri,
                loss_field=loss_field,
                as_of="1997-12-31",
                **fit_kwargs,
            )
            pred = entry.predict(seed=args.seed)
            # realized_ultimates() restricts to the origins the entry trained
            # on, keeping post-study accident years out of the outcome
            realized = entry.realized_ultimates(tri)
            table = pred.summary(observed=realized)
            # last row = the sum over accident years. Meyers validates the
            # TOTAL unpaid distribution, not per-origin cells, so this is the
            # only row scored.
            total = table.iloc[-1]
            # worst R-hat across all parameters: a convergence red flag stored
            # alongside the result so a bad fit can be spotted post hoc rather
            # than quietly polluting the KS panel
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
    """Select cohorts per line, fit + score each, write the CSV, test uniformity."""
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
    ap.add_argument(
        "--growth-curve",
        default=None,
        choices=["loglogistic", "weibull"],
        help="clark / clark_growth_curve only: override the growth curve "
        "(entry default: loglogistic)",
    )
    ap.add_argument(
        "--variant",
        default=None,
        choices=["gaussian", "lognormal"],
        help="compartmental only: case-study Model 1 (gaussian, entry default) "
        "or Model 2 (lognormal)",
    )
    ap.add_argument("--lines", nargs="+", default=MEYERS_LINES, choices=MEYERS_LINES)
    # 50 per line x 4 lines = the monograph's 200-company retrospective; the
    # combined KS critical value at n=200 is 9.6 (x100)
    ap.add_argument("--per-line", type=int, default=50)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=2500)
    # fixes both the selection order and the sampler streams, so a published
    # run reproduces cell for cell
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        # one CSV per model so entries never overwrite each other's results;
        # non-default ablations get a suffix (precedent: clark_validation_weibull)
        suffix = f"_{args.variant}" if args.variant not in (None, "gaussian") else ""
        args.out = (
            Path(__file__).parents[1]
            / "analysis"
            / "results"
            / f"{args.model}_validation{suffix}.csv"
        )

    # Resolves --warehouse (local path, github://...@publish_id, or the
    # @latest release) to a concrete cached parquet. Provenance: published
    # runs should pass an explicit @publish_id so the cohort is pinned to an
    # immutable gold publish rather than whatever "latest" happened to be.
    mart = active_mart_path(args.warehouse)
    all_rows: list[dict] = []
    # one cohort per line; the same cohort every model sees (see docstring)
    for line in args.lines:
        companies = select_companies(mart, line, args.per_line)
        print(f"{line}: {len(companies)} companies selected", flush=True)
        all_rows += run_line(args.warehouse, line, companies, args)

    # written before scoring so a long MCMC run's results survive a crash in
    # the summary code
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # THE headline test: are the predictive distributions honest? Percentiles
    # are PIT values in 0-100, so divide by 100 before the uniformity test.
    # Failure rows have a null percentile and drop out here — read `failed`
    # in the JSON report alongside `n`.
    ok = df[df.get("percentile").notna()] if "percentile" in df else pd.DataFrame()
    if len(ok) >= 5:  # KS on a handful of points says nothing; don't print it
        print("\nUniformity of total-outcome percentiles (Meyers p-p test):")
        # per line (n<=50, crit 19.2) then pooled ALL (n<=200, crit 9.6): the
        # pooled test is the demanding one and is the published headline
        for line, grp in ok.groupby("line"):
            print(f"  {line:<24} n={len(grp):>3}  {ks_uniformity(grp['percentile'] / 100)}")
        print(f"  {'ALL':<24} n={len(ok):>3}  {ks_uniformity(ok['percentile'] / 100)}")
        # machine-readable one-liner for the calling harness/notebook
        report = {
            "n": len(ok),
            "ks_all": ks_uniformity(ok["percentile"] / 100).statistic,
            "failed": int(df["percentile"].isna().sum()) if "percentile" in df else 0,
        }
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
