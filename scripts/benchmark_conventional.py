"""Reproducible conventional replay/selection benchmark, with no optional ML deps.

Run from the repository root:
  python scripts/benchmark_conventional.py --published --synthetic-seeds 30

Published grids use the literal table ranges, whose counts conflict with the
paper's prose. Synthetic scenarios use one prespecified smaller grid. Every
seed is retained; no tuning follows inspection of terminal evaluation losses.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr import __version__
from ibnr.kernels.conventional import ConventionalCandidate, conventional_grid, fit_conventional
from ibnr.kernels.replay import replay_conventional
from ibnr.kernels.selection import (
    ConventionalSelection,
    evaluate_conventional,
    score_replay,
    select_conventional,
)

ROOT = Path(__file__).resolve().parents[1]


def named_grid(**kwargs):
    """Names are deterministic and document lexical tie-breaking."""
    return {
        f"{c.method}_w{c.history_periods or 'all'}_h{int(c.drop_high)}_l{int(c.drop_low)}"
        + (f"_lr{c.expected_loss_ratio:.2f}" if c.method == "bf" else "")
        + (f"_g{c.decay:.2f}" if c.method == "gcc" else ""): c
        for c in conventional_grid(**kwargs)
    }


def select_batched(triangle, candidates, dates, *, loss_field, batch_size=64):
    """Bound memory, applying the SAME library selection to every grid chunk.

    A global minimum is the minimum of chunk minima. Full ranking and scoring
    rows are retained; only non-winning bulky historical fit objects are freed.
    """
    winners, ranks, scores = {m: [] for m in ("ave", "cdr")}, {}, {}
    for m in winners:
        ranks[m], scores[m] = [], []
    items = list(candidates.items())
    for start in range(0, len(items), batch_size):
        chunk = dict(items[start : start + batch_size])
        replay = replay_conventional(
            triangle, chunk, dates, loss_field=loss_field, on_error="record"
        )
        for metric in winners:
            scored = score_replay(replay, metric=metric)
            scores[metric].append(scored)
            if not (scored.groupby("candidate")["status"].agg(lambda s: s.eq("ok").all())).any():
                # Nothing in this chunk is eligible; retain failures rather than
                # pretending the missing candidates were never in the grid.
                ranks[metric].append(
                    pd.DataFrame(
                        [
                            {
                                "candidate": n,
                                "eligible": False,
                                "n_scored": int(g["status"].eq("ok").sum()),
                                "n_required": len(g),
                                "mean_rmse": np.nan,
                                "reason": "; ".join(
                                    dict.fromkeys(g.loc[g["status"] != "ok", "reason"])
                                )
                                + (
                                    "; no fit at the selection date"
                                    if (n, dates[-1]) not in replay.fits
                                    else ""
                                ),
                            }
                            for n, g in scored.groupby("candidate")
                        ]
                    )
                )
                continue
            decision = select_conventional(replay, selection_as_of=dates[-1], metric=metric)
            winners[metric].append(decision)
            ranks[metric].append(decision.ranking)
    result = {}
    for metric, choices in winners.items():
        if not choices:
            raise ValueError(
                f"no eligible candidate for {metric} across {len(candidates)} candidates"
            )
        winner = min(choices, key=lambda d: (d.ranking.iloc[0]["mean_rmse"], d.name))
        ranking = (
            pd.concat(ranks[metric], ignore_index=True)
            .sort_values(
                ["eligible", "mean_rmse", "candidate"], ascending=[False, True, True], kind="stable"
            )
            .reset_index(drop=True)
        )
        result[metric] = replace(
            winner, ranking=ranking, scores=pd.concat(scores[metric], ignore_index=True)
        )
    return result


def fixed_decision(triangle, candidate, date, loss_field, name):
    """A prespecified comparison baseline, with no historical selection score."""
    fit = fit_conventional(triangle, candidate, as_of=date, loss_field=loss_field)
    return ConventionalSelection(
        name,
        candidate,
        date,
        "fixed",
        fit,
        pd.DataFrame(),
        pd.DataFrame(),
        loss_field,
        "earned_premium",
        triangle.meta.units,
        fit.grid["segment"],
    )


def result_row(decision, triangle, end, **labels):
    evaluation = evaluate_conventional(decision, triangle, as_of=end)
    return {
        **labels,
        "candidate": decision.name,
        "method": decision.candidate.method,
        "settings": json.dumps(asdict(decision.candidate), default=str, sort_keys=True),
        "selection_as_of": decision.as_of,
        "evaluation_as_of": end,
        "history_mean_rmse": decision.ranking.iloc[0]["mean_rmse"]
        if not decision.ranking.empty
        else np.nan,
        **evaluation.summary,
    }


def published_benchmark():
    from conventional_examples import EXAMPLE_METADATA, load_published_examples

    examples = load_published_examples()
    rows, rankings = [], []
    refs = {
        ("swiss", "cl"): (669.69, 675.38, 617.81),
        ("swiss", "bf"): (576.38, 537.27, 527.90),
        ("swiss", "gcc"): (604.27, 670.69, 580.21),
        ("liability", "all"): (3170.88, 2552.39, 2893.23),
        ("property", "all"): (638.38, 626.73, 794.65),
    }
    for dataset, triangle in examples.items():
        meta = EXAMPLE_METADATA[dataset]
        if dataset == "swiss":
            dates = tuple(dt.date(y, 12, 31) for y in range(1984, 1998))
            windows, ratios, end = (
                tuple(range(10, 20)),
                tuple(np.arange(50, 71) / 100),
                dt.date(2016, 12, 31),
            )
            families = ("cl", "bf", "gcc")
        else:
            dates = tuple(pd.date_range("2012-03-31", "2014-12-31", freq="QE").date)
            windows, ratios, end = (
                tuple(range(5, 22)),
                tuple(np.arange(40, 61) / 100),
                dt.date(2019, 12, 31),
            )
            families = ("all",)
        common = dict(
            horizon=meta.horizon_months, unsupported_factor="unity", exhausted_exclusions="keep"
        )
        grid = named_grid(
            history_periods=windows,
            drop_high=(False, True),
            drop_low=(False, True),
            expected_loss_ratios=ratios,
            decays=tuple(np.arange(0, 21) / 20),
            **common,
        )
        for family in families:
            subset = {n: c for n, c in grid.items() if family == "all" or c.method == family}
            print(
                f"Published {dataset}/{family}: {len(subset)} candidates, "
                f"{len(dates) - 1} intervals",
                flush=True,
            )
            selected = select_batched(triangle, subset, dates, loss_field=meta.loss_field)
            method = "gcc" if family == "all" else family
            baseline = ConventionalCandidate(
                method,
                expected_loss_ratio=0.60 if method == "bf" else None,
                decay=0.75 if method == "gcc" else None,
                **common,
            )
            decisions = {
                "baseline": fixed_decision(
                    triangle, baseline, dates[-1], meta.loss_field, f"basic_{method}"
                ),
                **selected,
            }
            for rule, reference in zip(
                ("baseline", "ave", "cdr"), refs[dataset, family], strict=True
            ):
                row = result_row(
                    decisions[rule],
                    triangle,
                    end,
                    study="published",
                    dataset=dataset,
                    family=family,
                    rule=rule,
                    grid_size=len(subset),
                    seed=None,
                )
                row["paper_rmse"] = reference
                row["difference_from_paper"] = row["rmse"] - reference
                rows.append(row)
            for metric, decision in selected.items():
                rankings.append(
                    decision.ranking.assign(dataset=dataset, family=family, metric=metric)
                )
    return rows, pd.concat(rankings, ignore_index=True)


def synthetic_benchmark(n_seeds):
    from conventional_synthetic import (
        EVALUATION_DATE,
        HORIZON,
        REPLAY_DATES,
        SCENARIOS,
        SELECTION_DATE,
        synthetic_portfolio,
    )

    common = dict(horizon=HORIZON, unsupported_factor="unity", exhausted_exclusions="keep")
    grid = named_grid(
        history_periods=(None, 3, 5),
        drop_high=(False, True),
        expected_loss_ratios=(0.4, 0.5, 0.6),
        decays=(0.25, 0.75, 1.0),
        **common,
    )
    rows = []
    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            triangle = synthetic_portfolio(seed, scenario)
            selected = select_batched(triangle, grid, REPLAY_DATES, loss_field="paid_loss")
            for method in ("cl", "bf", "gcc"):
                candidate = ConventionalCandidate(
                    method,
                    expected_loss_ratio=0.6 if method == "bf" else None,
                    decay=0.75 if method == "gcc" else None,
                    **common,
                )
                selected[f"basic_{method}"] = fixed_decision(
                    triangle, candidate, SELECTION_DATE, "paid_loss", f"basic_{method}"
                )
            rows.extend(
                result_row(
                    d,
                    triangle,
                    EVALUATION_DATE,
                    study="synthetic",
                    dataset=scenario,
                    family="all",
                    rule=rule,
                    seed=seed,
                    grid_size=len(grid),
                )
                for rule, d in selected.items()
            )
        print(f"Synthetic {scenario}: {n_seeds} seeds, {len(grid)} candidates each", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", action="store_true")
    parser.add_argument("--synthetic-seeds", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/results/conventional")
    args = parser.parse_args()
    if args.synthetic_seeds < 0 or not (args.published or args.synthetic_seeds):
        parser.error("choose --published and/or a positive --synthetic-seeds count")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.published:
        published, rankings = published_benchmark()
        rows.extend(published)
        rankings.to_csv(args.output / "published_rankings.csv", index=False)
    if args.synthetic_seeds:
        rows.extend(synthetic_benchmark(args.synthetic_seeds))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "results.csv", index=False)
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *sorted((ROOT / "src/ibnr/kernels").glob("conventional.py")),
            ROOT / "src/ibnr/kernels/replay.py",
            ROOT / "src/ibnr/kernels/selection.py",
            *sorted((ROOT / "scripts").glob("*conventional*.py")),
        ]
    }
    manifest = {
        "ibnr_version": __version__,
        "published": args.published,
        "synthetic_seeds": list(range(args.synthetic_seeds)),
        "source_hashes": hashes,
        "protocol": "docs/conventional-benchmark.md",
    }
    if args.published:
        from conventional_examples import SOURCE_SHA256, SOURCE_URL

        manifest.update(published_source=SOURCE_URL, published_sha256=SOURCE_SHA256)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(frame.groupby(["study", "dataset", "rule"])["rmse"].agg(["count", "mean"]).to_string())
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
