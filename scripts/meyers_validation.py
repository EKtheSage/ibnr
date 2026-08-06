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
outstanding (= reported - paid) jointly, unclamped - its gaussian default
takes zero cells natively and the lognormal variant drops and counts its
own non-positive cells. Company selection applies both Table A.1 screens
(CV1 net premium, CV2 net/direct premium ratio) for every model.

METHODOLOGY (this script produces published results - read before changing)

1. Cohort-data parity, and where the screen lives. `select_companies` runs
   the identical mechanical screen for every model, so any two entries
   validated here are scored on the same insurers. That is why `meyers_csr`
   carries Meyers' floor-of-1 paid clamp: without it the lognormal would
   reject cohorts the incurred study kept, and the two studies would no
   longer be comparable. Model differences must come from the models, not
   from who they were allowed to see. `compare_gallery.py` and
   `parity_gallery.py` reuse `select_companies` for the same reason.

   The screen is no longer DEFINED here. It moved into
   `cas_schedule_p.screens` - the data's own PyPI package, built from the
   same repository as the gold mart and released alongside it - and this
   module re-exports its nine names so every existing
   `from meyers_validation import ...` keeps working unchanged.

   That is a stronger no-drift guarantee than the old one, not just a
   tidier one. The old argument was "there is exactly one copy of the
   screen, in this script, so import the script"; but that copy lived in a
   different repository from the mart it queries, so nothing tied the two
   together - a mart column could be renamed, or a screen constant edited,
   and only a run would notice. Now the constants, the SQL and the parquet
   are one release artifact with one version number, and this repo pins it
   (`cas-schedule-p>=2026.6.13`, dev group). A published run's cohort is
   therefore reproducible from two pins - the package version and the
   mart's publish_id - rather than from a git commit of this file.

2. Training slice vs realized outcomes. Each entry fits
   `as_of="1997-12-31"` - the upper triangle a reserving actuary could have
   seen at year-end 1997 - while the outcome is the ultimate realized in
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
   1.36/sqrt(n) - 19.2 at n=50, 9.6 at n=200; `kernels.calibration` prints
   `D x 100` and flags rejection with `*`, Meyers' own convention).

4. Failure handling. A fit that raises records an `error` row and the study
   continues; the JSON report prints a `failed` count. Failures are visible
   in the CSV but drop out of the KS panel, so a model that fit fewer
   companies is calibrated on a smaller (and probably easier) panel - always
   read `n` and `failed` together. The one silent data alteration is the
   paid clamp, restricted to `MODELS_WITH_PAID_CLAMP` and documented there.

5. Provenance. Every results row is stamped with the gold-mart `publish_id`
   resolved at run time (`mart_publish_id`), so a CSV traces back to an exact
   immutable publish even when `--warehouse` was left at "@latest" - same
   convention as `compare_gallery.py` (CLAUDE.md: experiment runs should pin
   a concrete @publish_id).

6. Parallel execution + staged escalation (`kernels.harness`). Companies fit
   across a process pool (all cores by default; `--workers`, or the
   IBNR_MAX_WORKERS env var in containers). Each model runs a two-stage
   sampler policy (STAGE_POLICIES): a cheap first pass, then a re-fit at the
   expensive settings ONLY for companies failing the convergence gates
   (R-hat/divergences/ESS - read `stage` in the CSV to see who escalated).
   For the Meyers-family entries stage 1 IS the published entry default, so
   escalation can only improve on the published runs; for `compartmental`
   stage 1 relaxes the monograph's adapt_delta 0.99 / treedepth 15 (that is
   the point - those settings cost 160-250 s/company and most companies do
   not need them). `--no-escalate` restores the single-stage entry-default
   behavior of the published CSVs. Note `max_rhat` is now
   `entry.convergence()`'s (core sampled parameters, matching the cards'
   convention) rather than the worst over all transformed quantities.

Usage:
    uv run python scripts/meyers_validation.py --per-line 50
    uv run python scripts/meyers_validation.py --model meyers_csr --per-line 50
    uv run python scripts/meyers_validation.py --lines workers_compensation --per-line 5 --chains 2
    uv run python scripts/meyers_validation.py --model compartmental --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

# The Meyers company screen, in the data's own package (see the METHODOLOGY
# note above). Six of these nine are unused in this module and imported purely
# so that `from meyers_validation import <name>` keeps resolving: notebook 03
# and `compare_gallery.py` / `parity_gallery.py` / `heldout_leaderboard.py`
# all reach for them by that path, and moving the definition must not move
# where callers find it.
from cas_schedule_p.screens import (
    AS_OF,
    CV1_LIMITS,  # noqa: F401 - re-exported for callers of this module
    CV2_LIMITS,  # noqa: F401 - re-exported
    EXCLUDED_GROUPS,  # noqa: F401 - re-exported
    MEYERS_LINES,
    MIN_LOSS,  # noqa: F401 - re-exported
    MIN_PREMIUM,  # noqa: F401 - re-exported
    TRAIN_AYS,  # noqa: F401 - re-exported (heldout_leaderboard.py reads it)
    select_companies,
)

from ibnr.data.schedule_p import active_mart_path, active_publish_id, pinned_source
from ibnr.kernels.calibration import ks_uniformity
from ibnr.kernels.harness import RetroTask, SamplerSettings, run_retro

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

#: models needing Meyers' pmax(cum_pdloss, 1) floor - the lognormal cannot
#: take non-positive cells. The ODP takes zeros natively and must see the
#: unclamped data (a clamp would silently alter increments); its negative-
#: increment failures are recorded, exactly like the bootstrap ODP's.
MODELS_WITH_PAID_CLAMP = {"meyers_csr"}

#: two-stage sampler escalation per model (kernels.harness). Stage 1 is the
#: cheap first pass, stage 2 re-fits only the companies that fail the
#: convergence gates - at the expensive settings, with chains in parallel
#: (few tasks remain by then, so cores are otherwise idle). target_accept /
#: max_treedepth of None mean "the entry's own default", so the Meyers-family
#: default stage 1 reproduces the published single-stage runs exactly.
DEFAULT_STAGES = (
    SamplerSettings(),
    SamplerSettings(target_accept=0.99, max_treedepth=15, parallel_chains=4),
)
STAGE_POLICIES = {
    # the compartmental entry's OWN default is the monograph's expensive
    # adapt_delta 0.99 / treedepth 15 (160-250 s/company); stage 1 relaxes it
    # and lets the gates decide who really needs the monograph settings
    "compartmental": (
        SamplerSettings(target_accept=0.9, max_treedepth=12),
        SamplerSettings(target_accept=0.99, max_treedepth=15, parallel_chains=4),
    ),
}


def stages_for(model: str, args: argparse.Namespace) -> tuple[SamplerSettings, ...]:
    """The escalation ladder for one model, with the CLI's MCMC budget applied.

    ``--no-escalate`` collapses to a single entry-default stage - the exact
    behavior (and cost) of the published pre-harness runs.
    """
    policy = (SamplerSettings(),) if args.no_escalate else STAGE_POLICIES.get(model, DEFAULT_STAGES)
    return tuple(
        replace(
            s,
            chains=args.chains,
            iter_warmup=args.warmup,
            iter_sampling=args.draws,
            parallel_chains=min(s.parallel_chains, args.chains),
        )
        for s in policy
    )


def print_progress(row: dict, stage: int, done: int, total: int) -> None:
    """Per-fit progress line, called from the parent as pool results land."""
    where = f"  [{done}/{total} stage{stage}] {row['line']} {row['company_code']}"
    if row.get("error") is not None:
        print(f"{where}: FAILED {row['error']}", flush=True)
        return
    rhat = row.get("max_rhat")
    rhat = float("nan") if rhat is None else float(rhat)
    print(
        f"{where}: pct={row['percentile']:.1f} rhat={rhat:.3f} ({row['seconds']:.1f}s)",
        flush=True,
    )


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
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="process-pool size (default: IBNR_MAX_WORKERS env var, else all cores minus one)",
    )
    ap.add_argument(
        "--serial",
        action="store_true",
        help="run everything in this process (debugging; real tracebacks)",
    )
    ap.add_argument(
        "--no-escalate",
        action="store_true",
        help="single stage at the entry's default sampler settings - the exact "
        "behavior of the published pre-harness runs (compartmental included)",
    )
    args = ap.parse_args()
    if args.out is None:
        # one CSV per model so entries never overwrite each other's results;
        # non-default variants get a suffix (precedent: clark_validation_weibull)
        suffix = f"_{args.variant}" if args.variant not in (None, "gaussian") else ""
        args.out = (
            Path(__file__).parents[1]
            / "analysis"
            / "results"
            / f"{args.model}_validation{suffix}.csv"
        )
    # fail on a bad --out NOW, not after hours of MCMC when the CSV writes
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # entry-specific fit arguments, passed through the harness verbatim
    fit_kwargs: dict = {}
    if args.growth_curve is not None:
        if args.model not in ("clark", "clark_growth_curve"):
            raise SystemExit(f"--growth-curve does not apply to {args.model}")
        fit_kwargs["growth_curve"] = args.growth_curve
    if args.variant is not None:
        if args.model != "compartmental":
            raise SystemExit(f"--variant does not apply to {args.model}")
        fit_kwargs["variant"] = args.variant
    loss_field = args.loss_field or MODEL_LOSS_FIELDS[args.model]
    # Meyers' pmax(cum_pdloss, 1): floor paid cells at 1 (in $000s) so the
    # lognormal accepts every cohort the incurred screens admit. The ONE data
    # alteration, confined to MODELS_WITH_PAID_CLAMP (a clamp shifts
    # increments; ODP and compartmental must see the unclamped series).
    clamp_paid = loss_field == "paid_loss" and args.model in MODELS_WITH_PAID_CLAMP

    # Resolves --warehouse (local path, github://...@publish_id, or the
    # @latest release) to a concrete cached parquet. Provenance: published
    # runs should pass an explicit @publish_id so the cohort is pinned to an
    # immutable gold publish rather than whatever "latest" happened to be.
    mart = active_mart_path(args.warehouse)
    # pinned + cache-warmed so pool workers never resolve @latest or call gh
    source = pinned_source(args.warehouse)

    # one cohort per line; the same cohort every model sees (see docstring)
    #
    # `mart` is passed EXPLICITLY and must stay that way. `select_companies`
    # now lives in cas_schedule_p and defaults to the mart bundled in that
    # wheel, which is a fixed publish - convenient for a caller who wants
    # exactly that vintage, and wrong here. A retrospective has to screen on
    # the publish it FITS, or the cohort comes from one dataset and the
    # triangles from another; --warehouse would silently stop reaching the
    # screen, and a run pinned to an old @publish_id would be selected by a
    # newer one. Same rule in compare_gallery.py and parity_gallery.py.
    tasks: list[RetroTask] = []
    for line in args.lines:
        companies = select_companies(mart, line, args.per_line)
        print(f"{line}: {len(companies)} companies selected", flush=True)
        tasks += [
            RetroTask(
                model=args.model,
                warehouse=source,
                line=line,
                company_code=code,
                as_of=AS_OF,
                loss_field=loss_field,
                clamp_paid=clamp_paid,
                seed=args.seed,
                fit_kwargs=dict(fit_kwargs),
            )
            for code in companies["company_code"]
        ]

    rows = run_retro(
        tasks,
        stages=stages_for(args.model, args),
        max_workers=args.workers,
        executor="serial" if args.serial else "process",
        progress=print_progress,
    )

    # written before scoring so a long MCMC run's results survive a crash in
    # the summary code
    df = pd.DataFrame(rows)
    # Provenance stamping (CLAUDE.md): the gold mart is published as immutable
    # GitHub releases, so recording the resolved publish_id pins these results
    # to an exact dataset version even when --warehouse was "@latest".
    df["mart_publish_id"] = active_publish_id(args.warehouse)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # THE headline test: are the predictive distributions honest? Percentiles
    # are PIT values in 0-100, so divide by 100 before the uniformity test.
    # Failure rows have a null percentile and drop out here - read `failed`
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
