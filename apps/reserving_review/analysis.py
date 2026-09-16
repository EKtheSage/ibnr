"""Build a reviewable evidence snapshot from a historical cumulative CSV."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr import Triangle
from ibnr.kernels.conventional import _date, conventional_grid
from ibnr.kernels.replay import replay_conventional
from ibnr.kernels.selection import select_conventional
from ibnr.triangle.core import CORE_COLUMNS, GRAIN_MONTHS

ROOT = Path(__file__).resolve().parents[2]
MAX_ROWS = 5000
MAX_CUTOFFS = 40


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _cutoffs(start, end, step):
    dates = [start]
    while dates[-1] < end and len(dates) < MAX_CUTOFFS:
        old = pd.Timestamp(dates[-1])
        new = old + pd.DateOffset(months=step)
        if old.is_month_end:
            new += pd.offsets.MonthEnd(0)
        dates.append(new.date())
    if len(dates) < 2 or dates[-1] != end:
        raise ValueError(
            f"history dates must end exactly at the cutoff with 2-{MAX_CUTOFFS} "
            "consecutive grain-aligned dates"
        )
    return dates


def analyze_request(payload: dict) -> dict:
    """Analyze through the chosen cutoff and return a JSON-safe immutable input.

    The caller authenticates/authorizes before invoking this function. It does
    not accept precomputed reserves or scores from the browser. Later source
    rows are removed before both the computation and the saved evidence.
    """
    if not isinstance(payload, dict):
        raise ValueError("analysis request must be a JSON object")
    text = payload.get("csv")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("upload a long-format cumulative CSV or load the demo")
    if len(text.encode("utf-8")) > 1_500_000:
        raise ValueError("CSV is too large for this local reference application")
    title = payload.get("title", "Reserving review")
    if not isinstance(title, str) or not title.strip() or len(title) > 160:
        raise ValueError("title must contain 1-160 characters")
    grain = payload.get("grain", "Y")
    if not isinstance(grain, str) or grain not in GRAIN_MONTHS:
        raise ValueError("grain must be Y, Q or M")
    horizon = payload.get("horizon")
    if type(horizon) is not int or not 1 <= horizon <= 720 or horizon % GRAIN_MONTHS[grain]:
        raise ValueError("horizon must be a grain multiple of 1-720 months")
    allow_unity = payload.get("allow_unity", False)
    if type(allow_unity) is not bool:
        raise ValueError("allow_unity must be a boolean")
    cutoff, first = _date(payload.get("as_of")), _date(payload.get("history_start"))
    dates = _cutoffs(first, cutoff, GRAIN_MONTHS[grain])
    metric = payload.get("metric", "ave")
    if metric not in ("ave", "cdr"):
        raise ValueError("metric must be ave or cdr")
    loss_field, premium_field = (
        payload.get("loss_field", "paid_loss"),
        payload.get("premium_field", "earned_premium"),
    )
    if any(not isinstance(f, str) or not f.strip() for f in (loss_field, premium_field)):
        raise ValueError("loss and premium field names must be nonempty strings")
    if loss_field == premium_field:
        raise ValueError("loss and premium must be different fields")
    units = payload.get("units", "source units")
    if not isinstance(units, str) or not units.strip() or len(units) > 100:
        raise ValueError("units must be a short nonempty label")
    records = csv.reader(io.StringIO(text.lstrip("\ufeff")), strict=True)
    try:
        header = next(records)
        for number, record in enumerate(records, start=2):
            if len(record) != len(header):
                raise ValueError(f"CSV record {number} has a different width from its header")
    except (csv.Error, StopIteration) as exc:
        raise ValueError("CSV is malformed") from exc
    if len(set(header)) != len(header):
        raise ValueError("CSV has duplicate column names")
    frame = pd.read_csv(io.StringIO(text))
    if not set(CORE_COLUMNS).issubset(frame.columns):
        raise ValueError(f"CSV must contain {', '.join(CORE_COLUMNS)}")
    if frame.empty or len(frame) > MAX_ROWS:
        raise ValueError(f"CSV must contain 1-{MAX_ROWS} rows")
    if frame.isna().any().any():
        raise ValueError("CSV has missing values; no rows may be silently dropped")
    for name in ("origin_period", "eval_date"):
        frame[name] = pd.to_datetime(frame[name], format="ISO8601", errors="raise").dt.date
    for name in ("value", "dev_lag"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name]).all():
            raise ValueError(f"{name} must be finite")
    if (frame["dev_lag"] <= 0).any() or (frame["dev_lag"] % GRAIN_MONTHS[grain]).any():
        raise ValueError("dev_lag must contain positive grain multiples in months")
    triangle = Triangle.from_long(
        frame, origin_grain=grain, dev_grain=grain, measure="cumulative", units=units.strip()
    )
    # Retain historical versions through the cutoff. Collapsing with as_of
    # here would erase older values needed by the earlier replay fits.
    triangle = triangle.filter(triangle.expr.eval_date <= cutoff)
    available = triangle.execute().sort_values(
        [*triangle.segments, "field", "origin_period", "dev_lag", "eval_date"]
    )
    if available.empty:
        raise ValueError("no observations are available at the chosen cutoff")
    # Refuse a several-cohort CSV here by name. Further down, every candidate at
    # every date fails on it, and the combined reasons become an error too long
    # to read and too long to show in the browser.
    if triangle.segments:
        combinations = available[triangle.segments].drop_duplicates()
        if len(combinations) != 1:
            raise ValueError(
                f"the CSV must contain one cohort; found {len(combinations)} combinations "
                f"of {', '.join(triangle.segments)}. Filter the CSV to one cohort, or drop "
                "the columns that separate them."
            )
    # Use the same frozen small grid throughout the application's review run.
    candidates = {}
    for c in conventional_grid(
        history_periods=(None, 3, 5),
        drop_high=(False, True),
        expected_loss_ratios=(0.4, 0.5, 0.6),
        decays=(0.25, 0.75, 1.0),
        horizon=horizon,
        unsupported_factor="unity" if allow_unity else "raise",
        exhausted_exclusions="keep",
    ):
        name = f"{c.method}_w{c.history_periods or 'all'}_h{int(c.drop_high)}"
        if c.method == "bf":
            name += f"_lr{c.expected_loss_ratio:.2f}"
        if c.method == "gcc":
            name += f"_g{c.decay:.2f}"
        candidates[name] = c
    replay = replay_conventional(
        triangle,
        candidates,
        dates,
        loss_field=loss_field,
        premium_field=premium_field,
        on_error="record",
    )
    selection = select_conventional(replay, selection_as_of=cutoff, metric=metric)
    source_csv = available.to_csv(index=False, lineterminator="\n")
    warnings = []
    fallback_dates, trimming_dates = [], []
    for date in dates:
        fit = replay.fits[selection.name, date]
        if fit.factor_summary.empty:
            continue
        if fit.factor_summary["unity_fallback"].any():
            fallback_dates.append(date.isoformat())
        if fit.factor_summary["extreme_trimming_skipped"].any():
            trimming_dates.append(date.isoformat())
    if fallback_dates:
        warnings.append(
            "Factor 1 substituted at unsupported ages in fits dated " + ", ".join(fallback_dates)
        )
    if trimming_dates:
        warnings.append(
            "Extreme trimming skipped to retain sparse observations in fits dated "
            + ", ".join(trimming_dates)
        )
    unavailable = int((~selection.ranking["eligible"]).sum())
    if unavailable:
        warnings.append(
            f"{unavailable} candidates lack a complete scoring history; "
            "reasons are retained in the ranking."
        )
    if len(available) < len(frame):
        warnings.append(
            "Later observations were removed from the saved information snapshot; "
            "historical versions were retained."
        )
    source_paths = [
        ROOT / "src/ibnr/kernels" / name
        for name in ("conventional.py", "replay.py", "selection.py", "contract.py")
    ]
    source_paths.append(Path(__file__))
    engine = {
        "python": platform.python_version(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("ibnr", "numpy", "pandas", "ibis-framework", "duckdb")
        },
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
    }
    return _json_ready(
        {
            "title": title.strip(),
            "as_of": cutoff,
            "history_start": first,
            "metric": metric,
            "loss_field": loss_field,
            "premium_field": premium_field,
            "units": units.strip(),
            "horizon": horizon,
            "grain": grain,
            "segment": replay.segment,
            "source_hash": hashlib.sha256(source_csv.encode("utf-8")).hexdigest(),
            "source_csv": source_csv,
            "source_rows": len(available),
            "engine": engine,
            "selected": {
                "name": selection.name,
                "settings": asdict(selection.candidate),
                "mean_rmse": selection.ranking.iloc[0]["mean_rmse"],
            },
            "ranking": selection.ranking.to_dict("records"),
            "origins": selection.fit.origins.to_dict("records"),
            "factor_summary": selection.fit.factor_summary.to_dict("records"),
            "factor_selection": selection.fit.factor_selection.to_dict("records"),
            "history_scores": selection.scores.to_dict("records"),
            "warnings": warnings,
        }
    )


def demo_request() -> dict:
    """Synthetic history only; no future outcomes are exposed in the demo CSV."""
    from scripts.conventional_synthetic import synthetic_portfolio

    frame = synthetic_portfolio(0, "stable").as_of("2020-12-31").execute()
    return {
        "title": "Synthetic liability - 2020 review",
        "csv": frame.to_csv(index=False),
        "as_of": "2020-12-31",
        "history_start": "2011-12-31",
        "grain": "Y",
        "horizon": 96,
        "loss_field": "paid_loss",
        "premium_field": "earned_premium",
        "units": "synthetic USD",
        "metric": "ave",
        "allow_unity": False,
    }
