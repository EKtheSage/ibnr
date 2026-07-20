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
    companies on ``line`` as of 1997-12-31 (reuses the monograph selection)."""
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
    """Fit each backend on one triangle; return {backend: fitted MeyersCCL}."""
    from ibnr.gallery.bayesian.meyers_ccl.model import MeyersCCL

    fitted = {}
    for backend in backends:
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
    """A CCL-simulated single-cohort triangle (paid_loss + premium) as a Triangle."""
    import datetime as dt

    from ibnr import Triangle

    rng = np.random.default_rng(seed)
    n_w = n_d = 10
    prem = rng.uniform(8000, 20000, n_w)
    logelr, rho = -0.5, 0.3
    alpha = np.concatenate([[0.0], rng.normal(0, 0.15, n_w - 1)])
    beta = np.concatenate([np.sort(rng.uniform(-1.5, 0.0, n_d - 1)), [0.0]])
    a = rng.uniform(0.2, 0.6, n_d)
    sig = np.sqrt(np.cumsum(a[::-1])[::-1] * 0.01)
    logprem = np.log(prem)
    mu = np.zeros((n_w, n_d))
    logc = np.zeros((n_w, n_d))
    for w in range(n_w):
        for d in range(n_d):
            m = logprem[w] + logelr + alpha[w] + beta[d]
            if w > 0:
                m += rho * (logc[w - 1, d] - mu[w - 1, d])
            mu[w, d] = m
            logc[w, d] = rng.normal(m, sig[d])
    rows = []
    for w in range(n_w):
        for d in range(n_d):
            if w + d >= n_d:  # upper triangle + realized lower for scoring
                pass
            origin = dt.date(1988 + w, 1, 1)
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
    from ibnr.kernels.parity import compare_posteriors

    fitted = _fit_backends(triangle, backends, args)
    conv_rows = []
    for backend, entry in fitted.items():
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
    reference = "stan" if "stan" in fitted else backends[0]
    idatas = {b: e.idata_ for b, e in fitted.items()}
    if len(idatas) > 1:
        report = compare_posteriors(idatas, reference=reference)
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warehouse", default=None)
    ap.add_argument("--line", default="workers_compensation")
    ap.add_argument("--n-companies", type=int, default=5)
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["stan", "numpyro", "pymc"],
        choices=["stan", "numpyro", "pymc"],
    )
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=2500)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--synthetic", action="store_true", help="simulate a triangle; skip the mart")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    if args.synthetic:
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
        print("\nConvergence by backend (median over cohorts):")
        agg = conv_df.groupby("backend").agg(
            runtime_s=("runtime_s", "median"),
            max_rhat=("max_rhat", "max"),
            min_ess_bulk=("min_ess_bulk", "min"),
            divergences=("divergences", "sum"),
        )
        print(agg.to_string())
    if not parity_df.empty:
        print("\nParity pass rate by backend:")
        print(parity_df.groupby("backend")["passed"].mean().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
