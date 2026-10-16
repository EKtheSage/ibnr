"""Notebook 04's tlrn fits on the JAX backend, for a Colab TPU (or GPU) runtime.

The heavy cell of ``analysis/04_nn_architectures_vs_classical.ipynb`` is "The two tlrn
fits": ``tlrn_8`` (paid only) and ``tlrn_13`` (paid, incurred and case), each one pooled
fit over the notebook's 243 company-line pairs. On the dev laptop's CPU that cell takes
hours. This script rebuilds the same triangle the notebook fits on - the same publish,
the same selection rule, with the notebook's own counts and cohort hash asserted - runs
the same fits with ``backend="jax"``, and writes each fitted entry to ``--out`` as soon as
it is done. A run that is interrupted (a Colab disconnect) keeps every entry already
written, and the next run skips them, so at most one fit is ever lost. Each entry is
written with a settings record beside it (the same name ending ``.json``), and a saved
entry is reused only when its record matches this run's settings exactly: a rerun with
a different ``--members``, ``--config`` or ``--smoke`` is refused rather than handed the
old fits, and the manifest describes the fits in the folder from their own records.

How the saved entries are used is in ``docs/tlrn-on-colab.md``: copy the folder back,
``pickle.load`` each file in place of the ``gallery.fit("tlrn", ...)`` call, and run the
rest of the notebook unchanged. An entry is loaded with the ibnr version that saved it.

Usage, on the runtime::

    python scripts/tlrn_colab.py --out /content/drive/MyDrive/ibnr/tlrn_fits
    python scripts/tlrn_colab.py --out ... --config accident_year --members 20
    python scripts/tlrn_colab.py --out ... --config factor_head    # the second study's rows
    python scripts/tlrn_colab.py --out ... --smoke      # a few epochs, to check the setup

The TPU speed is unmeasured: this script was checked on a CPU only (in ``--smoke`` form).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

#: notebook 04's study, value for value (its cells 4, 7, 9 and 13)
PUBLISH = "20260613_041006"
SOURCE_URL = f"github://EKtheSage/cas-schedule-p-data-model@{PUBLISH}"
AS_OF = dt.date(2007, 12, 31)
FIELD = "paid_loss"
INCURRED_FIELD = "incurred_loss"
CASE_FIELD = "case_reserve"
ORIGINS = [dt.date(y, 1, 1) for y in range(1998, 2008)]
LINES = ["commercial_auto", "other_liability", "private_passenger_auto", "workers_compensation"]
CUTOFF = 5
SEED_FIT = 11
AUDIT = [668, 414, 330, 243]
COHORT_HASH = "08605b3c80513bca"


def build_triangle():
    """The notebook's ``tri``: 243 selected pairs, the ten study accident years."""
    import duckdb
    import ibis
    import numpy as np

    from ibnr.data.schedule_p import active_mart_path, load_schedule_p, pinned_source

    source = pinned_source(SOURCE_URL)
    mart = active_mart_path(source)
    con = duckdb.connect()
    lines_sql = ", ".join(f"'{ln}'" for ln in LINES)
    cells = con.execute(f"""
        select company_code, line_of_business, accident_year,
               development_age as dev_lag, cum_paid_loss, earned_prem_net
        from read_parquet('{mart.as_posix()}')
        where accident_year between {ORIGINS[0].year} and {ORIGINS[-1].year}
          and line_of_business in ({lines_sql})
    """).df()

    audit = []
    groups = cells.groupby(["company_code", "line_of_business"])
    audit.append(groups.ngroups)
    grid = groups.agg(
        n_cells=("cum_paid_loss", "size"),
        n_ay=("accident_year", "nunique"),
        n_dev=("dev_lag", "nunique"),
        min_prem=("earned_prem_net", "min"),
        n_prem_missing=("earned_prem_net", lambda s: int(s.isna().sum())),
    ).reset_index()
    complete = grid[
        (grid["n_cells"] == 100)
        & (grid["n_ay"] == len(ORIGINS))
        & (grid["n_dev"] == len(ORIGINS))
        & (grid["min_prem"] > 0)
        & (grid["n_prem_missing"] == 0)
    ][["company_code", "line_of_business"]]
    audit.append(len(complete))

    vis = cells.merge(complete, on=["company_code", "line_of_business"])
    vis = vis[(vis["accident_year"] - ORIGINS[0].year + vis["dev_lag"]) <= CUTOFF].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vis["loss_ratio"] = vis["cum_paid_loss"] / vis["earned_prem_net"]
    vis["unusable"] = (
        ~np.isfinite(vis["cum_paid_loss"])
        | (vis["cum_paid_loss"] <= 0)
        | ~np.isfinite(vis["earned_prem_net"])
        | (vis["earned_prem_net"] <= 0)
        | ~np.isfinite(vis["loss_ratio"])
        | (vis["loss_ratio"] >= 3)
    )
    bad = vis.groupby(["company_code", "line_of_business"])["unusable"].sum()
    eligible = list(bad[bad == 0].index)
    audit.append(len(eligible))
    per_company: dict[str, int] = {}
    for company, _ in eligible:
        per_company[company] = per_company.get(company, 0) + 1
    pairs = sorted((c, ln) for c, ln in eligible if per_company[c] >= 2)
    audit.append(len(pairs))
    assert audit == AUDIT, (audit, AUDIT)
    cohort_hash = hashlib.sha256(repr(sorted(pairs)).encode()).hexdigest()[:16]
    assert cohort_hash == COHORT_HASH, cohort_hash

    companies = sorted({c for c, _ in pairs})
    tri_all = load_schedule_p(source, companies=companies, lines=LINES)
    e = tri_all.expr
    keep = ibis.literal(False)
    for c, ln in pairs:
        keep = keep | ((e.company_code == c) & (e.line_of_business == ln))
    selected = tri_all.filter(keep)
    selected = selected.with_expr(selected.expr.drop("company_name"))
    tri = selected.filter(selected.expr.origin_period.isin(ORIGINS))
    assert sorted(tri.origins) == ORIGINS
    return tri, len(pairs), len(companies)


def factor_head_config(**overrides):
    """The companion study's newest model (the factor head), as one ``TLRNConfig``.

    Every choice that defines the model is written out here and not left to the library's
    defaults, so a change of default in a later ibnr cannot change the model unnoticed. The
    network reads each cell's features, attends across the company's lines and across
    lags (one block, no accident-year attention), and corrects a learned development factor
    per line and step; each member is blended with the multivariate chain ladder by a fitted
    weight, twenty members are averaged, and the whole method is retrained at valuation
    dates 6, 7 and 8. ``overrides`` replace any choice, which is how ``factors_only`` and
    the smoke budget are set. The choices and the evidence for them are in
    ``knowledge/decisions/tlrn-factor-head-study.md``.
    """
    from ibnr.gallery.nn.tlrn.config import TLRNConfig

    choices = {
        # the head: softplus(phi + eps * net), phi starting at init_step / lag
        "head": "ldf",
        "eps": 0.5,
        "init_step": 0.57,
        "cl_anchor": False,
        "factors_only": False,
        "tail_policy": "observed_cl",
        # one block, line and lag attention only
        "attention": None,
        "cross_line": True,
        "n_layers": 1,
        "d_model": 32,
        "n_heads": 2,
        "dropout": 0.3,
        "mask": "unwritten_lines",
        # the member and the ensemble
        "member": "mcl_blend",
        "ensemble_size": 20,
        "keep": None,
        # batches, cutoffs and validation
        "batch_unit": "example",
        "batch_size": 64,
        "cutoff_sampling": "per_epoch",
        "min_cutoff": 2,
        "val_diagonals": 2,
        "scoring": "reached_cells",
        # the loss and the optimiser
        "w_pe": 0.5,
        "w_mse": 0.1,
        "mse_scale": 0.005,
        "lr": 3e-3,
        "lr_phi": 3e-2,
        "weight_decay": 0.0,
        "warmup": 20,
        "grad_clip": 1.0,
        "max_epochs": 3000,
        "min_epochs": 500,
        "check_every": 5,
        "patience": 600,
        # the earlier valuation dates the whole method is retrained at
        "calibration": "retrain_per_valuation",
        "calibration_cutoffs": (6, 7, 8),
        "calibration_horizons": (2, 3, 4),
    }
    return TLRNConfig(**{**choices, **overrides})


def configs(name: str, members: int, smoke: bool):
    """``(row, fit keyword arguments, TLRNConfig)`` for each fit of the cell."""
    from ibnr.gallery.nn.tlrn.config import TLRNConfig

    if name == "factor_head":
        # the second study's two rows: the full model, and the same model with the network
        # switched off (the head's pooled factors blended with the multivariate chain ladder)
        budget = {"ensemble_size": members}
        if smoke:
            budget |= {"max_epochs": 10, "min_epochs": 0, "check_every": 5, "patience": 100}
        roles = {"incurred_field": INCURRED_FIELD, "case_field": CASE_FIELD}
        return [
            ("tlrn_13_fh", roles, factor_head_config(**budget)),
            ("tlrn_13_fh_factors", roles, factor_head_config(factors_only=True, **budget)),
        ]
    if name == "published":
        # notebook 04's own budget: the published protocol, forty members all kept
        budget = {"ensemble_size": members, "keep": members}
        make = TLRNConfig
    else:
        # the companion study's accident-year variant, every member averaged
        budget = {"ensemble_size": members}
        make = TLRNConfig.accident_year_variant
    if smoke:
        budget |= {"max_epochs": 10, "min_epochs": 0, "check_every": 5, "patience": 100}
    roles = {"incurred_field": INCURRED_FIELD, "case_field": CASE_FIELD}
    suffix = "" if name == "published" else "_ay"
    return [
        (f"tlrn_8{suffix}", {}, make(**budget)),
        (f"tlrn_13{suffix}", roles, make(**budget)),
    ]


def fit_settings(row: str, roles: dict, config, *, config_name: str, smoke: bool) -> dict:
    """Everything that decides what a saved fit IS, as plain JSON values.

    Two runs with equal settings fit the same members, so a saved fit is reused only
    under equal settings. The ibnr version is in the file name instead (a fitted entry is
    loaded with the ibnr that wrote it), and what a run ran on (jax, the device) is
    recorded beside the settings without deciding anything.
    """
    from dataclasses import asdict

    settings = {
        "row": row,
        "config_name": config_name,
        "smoke": smoke,
        "config": asdict(config),
        "roles": roles,
        "backend": "jax",
        "seed": SEED_FIT,
        "as_of": AS_OF.isoformat(),
        "loss_field": FIELD,
        "publish_id": PUBLISH,
        "cohort_hash": COHORT_HASH,
    }
    # through JSON, so a tuple in the config compares equal to the list read back
    return json.loads(json.dumps(settings))


def fit_paths(out: Path, row: str, *, smoke: bool, version: str) -> tuple[Path, Path]:
    """``(fitted entry, its settings record)``, e.g. ``tlrn_8-ibnr0.7.4.pkl`` and ``.json``."""
    stem = f"{row}{'_smoke' if smoke else ''}-ibnr{version}"
    return out / f"{stem}.pkl", out / f"{stem}.json"


def saved_fit(pkl: Path, record: Path, settings: dict) -> dict | None:
    """The record of a saved fit made with exactly ``settings``, or None if there is none.

    A saved fit made with OTHER settings is refused, not reused and not overwritten:
    reusing it would put a fit of another budget under this run's name, and overwriting
    it would destroy a fit someone may still want. The way out is a different ``--out``
    folder, or deleting the two files.
    """
    if not pkl.exists():
        return None
    if not record.exists():
        raise SystemExit(
            f"{pkl} exists with no settings record ({record.name}), so nothing says what "
            "it was fitted with. Write this run to a different --out folder, or delete "
            "the .pkl to refit it"
        )
    saved = json.loads(record.read_text(encoding="utf-8"))
    theirs = saved.get("settings", {})
    differ = [
        k for k in sorted(settings.keys() | theirs.keys()) if settings.get(k) != theirs.get(k)
    ]
    if differ:
        if "config" in differ:
            mine_c, theirs_c = settings.get("config", {}), theirs.get("config") or {}
            differ.remove("config")
            differ += [
                f"config.{k}"
                for k in sorted(mine_c.keys() | theirs_c.keys())
                if mine_c.get(k) != theirs_c.get(k)
            ]
        raise SystemExit(
            f"{pkl} was fitted with different settings from this run's (they differ in "
            f"{', '.join(differ)}). Write this run to a different --out folder, or delete "
            "the .pkl and its .json to refit it here"
        )
    return saved


def write_record(record: Path, settings: dict, run: dict) -> None:
    """The settings a fit was made with and what it ran on, beside the fitted entry.

    Written AFTER the entry, so a record always describes a complete fit; an entry left
    with no record (an interruption between the two writes) is refused, not trusted.
    """
    partial = record.with_name(record.name + ".part")
    partial.write_text(json.dumps({"settings": settings, "run": run}, indent=2), "utf-8")
    os.replace(partial, record)


def manifest(records: dict[str, dict], *, config_name: str, smoke: bool) -> dict:
    """The run's manifest, built from the records of the fits actually in the folder.

    Each row's settings and timing come from its own record, so a fit an earlier run
    saved is described as that run made it, not as this run would have.
    """
    return {
        "config": config_name,
        "smoke": smoke,
        "publish_id": PUBLISH,
        "fits": {row: records[row] for row in sorted(records)},
    }


def save(entry, path: Path) -> None:
    """Write through a temporary name, so an interrupted write never looks like a fit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as handle:
        pickle.dump(entry, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(partial, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="folder for the fitted entries")
    parser.add_argument(
        "--config", choices=("published", "accident_year", "factor_head"), default="published"
    )
    parser.add_argument("--members", type=int, default=None, help="40 published, 20 variant")
    parser.add_argument("--smoke", action="store_true", help="ten epochs, to check the setup")
    args = parser.parse_args(argv)
    members = args.members or (40 if args.config == "published" else 20)
    if args.smoke:
        members = min(members, 4)

    import jax

    import ibnr
    from ibnr import gallery

    # A fitted entry is loaded with the ibnr that wrote it, so the version goes in the name.
    print("ibnr", ibnr.__version__, "| jax", jax.__version__, "| backend", jax.default_backend())
    print("devices", jax.devices(), flush=True)
    if jax.default_backend() == "cpu" and not args.smoke:
        print("warning: jax sees no accelerator; on a CPU the torch backend is faster")

    # check every saved fit before the slow work, so a mismatch stops the run at once
    plan, records = [], {}
    for row, roles, config in configs(args.config, members, args.smoke):
        settings = fit_settings(row, roles, config, config_name=args.config, smoke=args.smoke)
        pkl, record = fit_paths(args.out, row, smoke=args.smoke, version=ibnr.__version__)
        saved = saved_fit(pkl, record, settings)
        if saved is not None:
            print(f"{row}: already saved at {pkl} with these settings, skipped")
            records[row] = saved
        else:
            plan.append((row, roles, config, settings, pkl, record))

    if plan:
        t0 = time.perf_counter()
        tri, n_pairs, n_companies = build_triangle()
        built = time.perf_counter() - t0
        print(f"{n_pairs} pairs over {n_companies} companies, built in {built:.0f}s")

    for row, roles, config, settings, pkl, record in plan:
        print(f"{row}: {config.ensemble_size} members, {config.max_epochs} epochs", flush=True)
        t0 = time.perf_counter()
        entry = gallery.fit(
            "tlrn",
            tri,
            loss_field=FIELD,
            as_of=AS_OF,
            seed=SEED_FIT,
            config=config,
            backend="jax",
            show_progress=True,
            **roles,
        )
        seconds = round(time.perf_counter() - t0, 1)
        save(entry, pkl)
        run = {
            "ibnr": ibnr.__version__,
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "fit_seconds": seconds,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        write_record(record, settings, run)
        records[row] = {"settings": settings, "run": run}
        print(f"{row}: fitted in {seconds:.0f}s, saved to {pkl}", flush=True)
        print(entry.selection_.to_string(), flush=True)

    stem = "manifest_smoke" if args.smoke else f"manifest_{args.config}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}.json").write_text(
        json.dumps(manifest(records, config_name=args.config, smoke=args.smoke), indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
