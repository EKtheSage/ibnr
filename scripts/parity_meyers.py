"""Cross-backend parity + convergence comparison for meyers_ccl (milestone 4).

Fits the identical CCL model — same centered parameterization, same data
contract, same sampler settings — with all three backends (Stan reference,
NumPyro, PyMC) on the same Schedule P company triangle(s) as of 1997-12-31,
then:

1. **Parity** (correctness gate): each port's posterior is compared to the Stan
   reference via ``kernels.parity`` (mean agreement in MCSE units, SD ratio,
   marginal KS). Must pass before any convergence claim is meaningful.
2. **Convergence** (the comparison): per backend, max R-hat, min bulk/tail ESS,
   divergence count/fraction, and wall-clock sampling runtime.

Writes ``analysis/results/parity_meyers.csv`` and
``analysis/results/convergence_meyers.csv``. With ``--synthetic`` it runs on a
CCL-simulated triangle and skips the mart (useful without the warehouse; Stan
still needs a cmdstan toolchain).

METHODOLOGY (this script produces published results — read before changing)

* Cohort-data parity. All three backends are fit on the SAME triangle object,
  sliced at the same as_of, with the same chains/warmup/draws/target_accept
  and the same seed. `_fit_backends` is the single place those settings are
  applied, precisely so no backend can quietly get a different problem. Only
  then does a runtime or ESS comparison measure the sampler rather than the
  setup (CLAUDE.md: "same model across PPLs is only same if parameterization
  is held constant").

* Parity gates, convergence does not. `kernels.parity.compare_posteriors`
  compares every marginal's mean and SD to the Stan reference in MCSE units
  (z-scores, ~N(0,1) under the null of identical posteriors); `passed` is
  True only when every parameter clears `z_tol`. KS on the pooled draws is
  recorded for context but deliberately does NOT gate — MCMC autocorrelation
  inflates it. A FAIL means the port is a different model, and any
  convergence number from that run is meaningless.

* Unlike the retrospective harnesses this script scores no outcomes, so the
  post-study-origin trap does not arise: nothing here reads the realized
  lower half of the triangle. It only compares posteriors of the fit.

* Failure handling. There is no try/except here, by design — a backend that
  errors should stop the run loudly rather than leave a hole in a parity
  table. Partial results ARE preserved: both CSVs are rewritten after every
  cohort (see `main`).

* Provenance. `--warehouse` accepts `github://owner/repo@publish_id`; the
  seed fixes the sampler streams. `--synthetic` results are self-contained
  (the triangle is generated from `--seed`) and are NOT comparable to
  mart-based rows — keep the two kinds of run in separate files.

Usage:
    uv run python scripts/parity_meyers.py --line workers_compensation --n-companies 5
    uv run python scripts/parity_meyers.py --backends numpyro pymc --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parents[1] / "analysis" / "results"


def _load_company_contracts(warehouse, line: str, n_companies: int, seed: int):
    """Return a list of (label, training-triangle) for the top-n selected
    companies on ``line`` as of 1997-12-31 (reuses the monograph selection).

    Reuses ``meyers_validation.select_companies`` rather than picking
    triangles ad hoc, so parity is demonstrated on exactly the kind of data
    the published retrospective runs on — real Schedule P cohorts, including
    their awkward late-development cells, not a curated easy case.

    The triangle is returned whole; ``_fit_backends`` passes as_of to
    ``MeyersCCL.fit``, which does the slicing.
    """
    import sys

    import ibis

    from ibnr.data.schedule_p import active_mart_path, load_schedule_p

    sys.path.insert(0, str(Path(__file__).parent))
    from meyers_validation import select_companies

    mart = active_mart_path(warehouse)
    picks = select_companies(mart, line, n_companies)
    tri_line = load_schedule_p(warehouse, lines=[line])
    return [
        (f"{line}:{code}", tri_line.filter(ibis._.company_code == code))
        for code in picks["company_code"]
    ]


def _fit_backends(triangle, backends, args):
    """Fit each backend on one triangle; return {backend: fitted MeyersCCL}.

    THE parity invariant lives here: every backend gets byte-identical
    arguments — same triangle, same loss field, same as_of, same chains /
    warmup / draws / target_accept / seed. Only ``backend`` varies. Anything
    that differs between backends must therefore be an implementation
    difference, which is exactly what the comparison is supposed to isolate.
    """
    from ibnr.gallery.bayesian.meyers_ccl.model import MeyersCCL

    fitted = {}
    for backend in backends:
        # reported_loss = Meyers' incurred net of bulk — the CCL literature
        # pairing; the Stan reference posterior is defined on this field
        entry = MeyersCCL().fit(
            triangle,
            loss_field="reported_loss",
            as_of="1997-12-31",
            backend=backend,
            chains=args.chains,
            iter_warmup=args.warmup,
            iter_sampling=args.draws,
            seed=args.seed,
            target_accept=args.target_accept,
        )
        fitted[backend] = entry
    return fitted


def _synthetic_triangle(seed: int):
    """A CCL-simulated single-cohort triangle (paid_loss + premium) as a Triangle.

    Draws from the Correlated Chain Ladder generative model itself, so the
    fitted model is correctly specified and the posterior is well behaved —
    the point is to exercise the three samplers on a known-good problem where
    a parity failure can only be an implementation bug, never model
    misspecification. Used when the Schedule P mart is unavailable (CI).
    """
    import datetime as dt

    from ibnr import Triangle

    rng = np.random.default_rng(seed)
    n_w = n_d = 10  # 10 accident years x 10 development ages
    prem = rng.uniform(8000, 20000, n_w)  # (n_w,) earned premium per AY
    logelr, rho = -0.5, 0.3  # log expected loss ratio; AY-to-AY correlation
    # alpha[0] pinned to 0 and beta[-1] pinned to 0: the CCL identifiability
    # constraints (level absorbed by logelr, tail absorbed by the last dev)
    alpha = np.concatenate([[0.0], rng.normal(0, 0.15, n_w - 1)])  # (n_w,) AY effects
    # sorted increasing to 0 so development is monotone toward ultimate
    beta = np.concatenate([np.sort(rng.uniform(-1.5, 0.0, n_d - 1)), [0.0]])  # (n_d,)
    a = rng.uniform(0.2, 0.6, n_d)
    # Meyers' sig construction: variance accumulates from the tail inward, so
    # sig is DECREASING in d — early, immature cells are the noisy ones
    sig = np.sqrt(np.cumsum(a[::-1])[::-1] * 0.01)  # (n_d,)
    logprem = np.log(prem)
    mu = np.zeros((n_w, n_d))  # (n_w, n_d) conditional means on the log scale
    logc = np.zeros((n_w, n_d))  # (n_w, n_d) simulated log cumulative loss
    for w in range(n_w):
        for d in range(n_d):
            m = logprem[w] + logelr + alpha[w] + beta[d]
            # the "correlated" in CCL: each AY's log loss at dev d is pulled
            # toward the PREVIOUS AY's residual at the same d
            if w > 0:
                m += rho * (logc[w - 1, d] - mu[w - 1, d])
            mu[w, d] = m
            logc[w, d] = rng.normal(m, sig[d])
    rows = []
    for w in range(n_w):
        for d in range(n_d):
            # emit the FULL square: the upper triangle is what fit() will
            # slice out via as_of, the lower half is the realized outcome a
            # scoring harness would use. (This branch is a deliberate no-op —
            # nothing is filtered; parity itself never reads the lower half.)
            if w + d >= n_d:  # upper triangle + realized lower for scoring
                pass
            origin = dt.date(1988 + w, 1, 1)
            # dev_lag is months from origin START, so the first diagonal is 12;
            # eval_date is the year-end that (origin + dev_lag) lands in
            # (CLAUDE.md conventions — bermuda's dev_lag differs, do not copy it)
            eval_date = dt.date(1988 + w + d, 12, 31)
            rows.append(
                (
                    "synthetic",
                    origin,
                    12 * (d + 1),
                    eval_date,
                    "reported_loss",
                    float(np.exp(logc[w, d])),
                )
            )
            rows.append(
                ("synthetic", origin, 12 * (d + 1), eval_date, "earned_premium", float(prem[w]))
            )
    df = pd.DataFrame(
        rows, columns=["company_code", "origin_period", "dev_lag", "eval_date", "field", "value"]
    )
    return Triangle.from_long(df, measure="cumulative")


def run(label, triangle, backends, args):
    """Fit every backend on one cohort, then report convergence and parity.

    Returns (convergence rows, parity rows). Read them in that order but
    TRUST them in the reverse order: convergence numbers describe a sampler
    only if parity says all three backends were sampling the same posterior.
    """
    from ibnr.kernels.parity import compare_posteriors

    fitted = _fit_backends(triangle, backends, args)
    conv_rows = []
    for backend, entry in fitted.items():
        # max R-hat, min bulk/tail ESS, divergence count, wall-clock sampling
        # time — the standard HMC diagnostic set, per backend
        c = entry.convergence()
        c["label"] = label
        conv_rows.append(c)
        print(
            f"  {label} {backend:<8} runtime={c['runtime_s']:6.1f}s "
            f"max_rhat={c['max_rhat']:.3f} min_ess_bulk={c['min_ess_bulk']:.0f} "
            f"div={c['divergences']}",
            flush=True,
        )

    parity_rows = []
    # Stan is ground truth (design decision 7): the literature's published
    # implementation. Without it — e.g. CI with no cmdstan toolchain — the
    # first requested backend stands in, which still catches numpyro-vs-pymc
    # divergence but is NOT evidence of matching the literature.
    reference = "stan" if "stan" in fitted else backends[0]
    idatas = {b: e.idata_ for b, e in fitted.items()}
    if len(idatas) > 1:  # nothing to compare a single backend against
        report = compare_posteriors(idatas, reference=reference)
        # one row per backend: the WORST mean/sd z-score and KS over all
        # compared parameter elements, plus the overall pass flag
        summ = report.summary()
        for backend, row in summ.iterrows():
            parity_rows.append(
                {
                    "label": label,
                    "reference": reference,
                    "backend": backend,
                    "max_abs_z_mean": row["max_abs_z_mean"],
                    "max_abs_z_sd": row["max_abs_z_sd"],
                    "max_ks": row["max_ks"],
                    "n_params": int(row["n_params"]),
                    "passed": bool(row["passed"]),
                }
            )
            print(
                f"    parity {backend} vs {reference}: max|z_mean|={row['max_abs_z_mean']:.2f} "
                f"max|z_sd|={row['max_abs_z_sd']:.2f} max_ks={row['max_ks']:.3f} "
                f"-> {'PASS' if row['passed'] else 'FAIL'}",
                flush=True,
            )
    return conv_rows, parity_rows


def main() -> int:
    """Resolve cohorts, run every backend on each, write parity + convergence CSVs."""
    ap = argparse.ArgumentParser(description=__doc__)
    # None falls through to ibnr's resolution (env var, else the @latest
    # GitHub release); pass github://owner/repo@publish_id to pin a publish
    ap.add_argument("--warehouse", default=None)
    ap.add_argument("--line", default="workers_compensation")
    ap.add_argument("--n-companies", type=int, default=5)
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["stan", "numpyro", "pymc"],
        choices=["stan", "numpyro", "pymc"],
    )
    # Sampler settings deliberately shared by every backend. Parity compares
    # in MCSE units, so short chains widen the tolerance rather than causing
    # false failures — but changing these changes the MCSE and hence how
    # sensitive the gate is. 4x2500 matches the retrospective harness.
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=2500)
    # 0.8 is Stan's default; raising it hides divergences that the
    # convergence comparison is meant to expose
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--synthetic", action="store_true", help="simulate a triangle; skip the mart")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    if args.synthetic:
        # distinct seeds -> distinct simulated triangles, so a parity pass is
        # not an accident of one lucky dataset
        cohorts = [
            (f"synthetic:{s}", _synthetic_triangle(seed=args.seed + s))
            for s in range(max(1, args.n_companies))
        ]
    else:
        cohorts = _load_company_contracts(args.warehouse, args.line, args.n_companies, args.seed)
    print(f"{len(cohorts)} cohort(s); backends={args.backends}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conv_path = args.out_dir / "convergence_meyers.csv"
    parity_path = args.out_dir / "parity_meyers.csv"

    conv_all, parity_all = [], []
    for label, tri in cohorts:
        conv, parity = run(label, tri, args.backends, args)
        conv_all += conv
        parity_all += parity
        # rewrite after each cohort so a long run's partial results survive
        pd.DataFrame(conv_all).to_csv(conv_path, index=False)
        pd.DataFrame(parity_all).to_csv(parity_path, index=False)

    conv_df = pd.DataFrame(conv_all)
    parity_df = pd.DataFrame(parity_all)
    print(f"\nwrote {conv_path}\nwrote {parity_path}")

    if not conv_df.empty:
        # median runtime (typical cost) but WORST r-hat, WORST ess and TOTAL
        # divergences: a backend is only as trustworthy as its worst cohort,
        # so those are aggregated pessimistically on purpose
        print("\nConvergence by backend (median over cohorts):")
        agg = conv_df.groupby("backend").agg(
            runtime_s=("runtime_s", "median"),
            max_rhat=("max_rhat", "max"),
            min_ess_bulk=("min_ess_bulk", "min"),
            divergences=("divergences", "sum"),
        )
        print(agg.to_string())
    if not parity_df.empty:
        # anything below 1.0 means at least one cohort's port did not match
        # the Stan reference — inspect that cohort before quoting any
        # convergence or runtime figure for this backend
        print("\nParity pass rate by backend:")
        print(parity_df.groupby("backend")["passed"].mean().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
