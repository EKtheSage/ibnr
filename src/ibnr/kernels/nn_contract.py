"""Triangle -> standardized NN training data.

The NN analogue of ``contract.py``: one canonical mapping from a
multi-segment cumulative Triangle to dense numpy grids + masks, consumed by
the neural gallery entries. Torch conversion happens inside each entry —
this module never imports torch.

Conventions:
- A *cohort* is one distinct segment combination (company x lob, ...), the
  unit the NN pools over. Axis 0 of every per-cohort array.
- Grids are (n_w, n_d) with 1-based dev step d stored at index d-1 and a
  shared origin axis across cohorts (missing origins are just masked out).
- ``x`` holds *incremental loss ratios*: incremental loss / origin premium —
  the cross-cohort normalizer, same role ``logprem`` plays in ``stan_data``.
  Increments exist only against the immediate predecessor dev (first dev =
  cumulative), mirroring ``transforms.to_incremental``; a cell whose
  predecessor is missing is treated as unobserved.
- ``obs_mask`` marks cells with a *usable increment*, not raw presence.
- ``latest_cum``/``latest_dev`` anchor ultimates: predicted future increments
  (x premium) are added onto ``latest_cum``.
- Calendar index ``cal_idx[w, d] = w + d + 1`` (1-based diagonal number) —
  used for cutoff augmentation and eval_date-style validation splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ibnr.kernels.contract import _as_date
from ibnr.triangle.core import GRAIN_MONTHS, Triangle

#: segment columns that are display-only and never define a cohort
DISPLAY_COLUMNS = ("company_name",)


def nn_data(
    triangle: Triangle,
    *,
    loss_field: str = "reported_loss",
    feature_fields: tuple[str, ...] = (),
    premium_field: str = "earned_premium",
    segment_columns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Map a multi-segment cumulative Triangle to dense grids and masks.

    Slice training data with ``triangle.as_of(...)`` before calling — this
    function uses every row. Cohorts that cannot be used (missing/non-positive
    premium, no usable increments) are dropped and reported in ``dropped``.
    """
    if triangle.meta.measure != "cumulative":
        raise ValueError("nn_data requires a cumulative triangle")
    if segment_columns is None:
        segment_columns = tuple(c for c in triangle.segments if c not in DISPLAY_COLUMNS)
    seg_cols = list(segment_columns)
    if not seg_cols:
        raise ValueError("nn_data needs at least one segment column to define cohorts")

    fields = [loss_field, *feature_fields]
    if len(set(fields)) != len(fields):
        raise ValueError(f"duplicate fields in loss_field + feature_fields: {fields}")
    df = triangle.select_fields(fields).execute()
    if df.empty:
        raise ValueError(f"no rows for fields {fields}")
    df = df.copy()
    df["origin_period"] = _as_date(df["origin_period"])
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    if (df["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")
    df["d"] = (df["dev_lag"] // step).astype(int)
    if (df["d"] < 1).any():
        raise ValueError("dev_lag must be positive")
    if df.duplicated([*seg_cols, "field", "origin_period", "d"]).any():
        raise ValueError(
            "multiple rows per (cohort, field, origin, dev) cell; "
            "slice with as_of()/latest_diagonal() first"
        )

    origins = sorted(df["origin_period"].unique())
    n_w, n_d = len(origins), int(df["d"].max())
    w_of = {o: w for w, o in enumerate(origins)}
    n_f = len(fields)

    premium_df = triangle.select_fields(premium_field).latest_diagonal().execute()
    premium_df = premium_df.copy()
    premium_df["origin_period"] = _as_date(premium_df["origin_period"])

    def norm(key) -> tuple:
        return key if isinstance(key, tuple) else (key,)

    cohort_keys = sorted(map(tuple, df[seg_cols].drop_duplicates().itertuples(index=False)))
    grouped = {norm(k): g for k, g in df.groupby(seg_cols)}
    prem_grouped = (
        {norm(k): g for k, g in premium_df.groupby(seg_cols)} if not premium_df.empty else {}
    )

    kept: list[dict] = []
    dropped: list[dict] = []
    x_list, obs_list, prem_list, anchor_cum_list, anchor_dev_list = [], [], [], [], []

    for key in cohort_keys:
        sub = grouped[key]
        row = dict(zip(seg_cols, key, strict=True))

        prem_sub = prem_grouped.get(key)
        premium = np.full(n_w, np.nan)
        if prem_sub is not None:
            for r in prem_sub.itertuples():
                premium[w_of[r.origin_period]] = float(r.value)

        cum = np.full((n_f, n_w, n_d), np.nan)
        for r in sub.itertuples():
            cum[fields.index(r.field), w_of[r.origin_period], int(r.d) - 1] = float(r.value)

        # increments against the immediate predecessor; first dev = cumulative
        incr = np.full_like(cum, np.nan)
        incr[:, :, 0] = cum[:, :, 0]
        incr[:, :, 1:] = cum[:, :, 1:] - cum[:, :, :-1]
        obs = ~np.isnan(incr[0])

        origin_has_loss = ~np.all(np.isnan(cum[0]), axis=1)
        bad_premium = origin_has_loss & ~(premium > 0)
        if bad_premium.any():
            dropped.append({**row, "reason": "missing or non-positive premium"})
            continue
        if not obs.any():
            dropped.append({**row, "reason": f"no usable {loss_field!r} increments"})
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = incr / premium[None, :, None]
        x = np.where(np.isnan(ratios), 0.0, ratios)
        # non-target channels may be missing where the target is observed; zero-fill
        x[:, ~obs] = 0.0

        latest_dev = np.zeros(n_w, dtype=int)
        latest_cum = np.zeros(n_w)
        cum_obs = ~np.isnan(cum[0])
        for w in range(n_w):
            devs = np.nonzero(cum_obs[w])[0]
            if devs.size:
                latest_dev[w] = int(devs[-1]) + 1
                latest_cum[w] = cum[0, w, devs[-1]]

        kept.append(row)
        x_list.append(x)
        obs_list.append(obs)
        prem_list.append(np.where(premium > 0, premium, np.nan))
        anchor_cum_list.append(latest_cum)
        anchor_dev_list.append(latest_dev)

    if not kept:
        raise ValueError("no usable cohorts after screening; see the dropped reasons")

    cohorts = pd.DataFrame(kept)
    premium = np.stack(prem_list)

    if "line_of_business" in cohorts.columns:
        lob_levels = sorted(cohorts["line_of_business"].unique())
        lob_idx = cohorts["line_of_business"].map(lob_levels.index).to_numpy(dtype=int)
    else:
        lob_levels = ["all"]
        lob_idx = np.zeros(len(cohorts), dtype=int)
    if "company_code" in cohorts.columns:
        company_levels = sorted(cohorts["company_code"].unique())
        company_idx = cohorts["company_code"].map(company_levels.index).to_numpy(dtype=int)
    else:
        company_levels = ["all"]
        company_idx = np.zeros(len(cohorts), dtype=int)

    w_grid, d_grid = np.meshgrid(np.arange(n_w), np.arange(n_d), indexing="ij")
    return {
        "x": np.stack(x_list),
        "obs_mask": np.stack(obs_list),
        "cal_idx": w_grid + d_grid + 1,
        "premium": premium,
        "log_premium": np.log(np.nanmean(premium, axis=1)),
        "lob_idx": lob_idx,
        "lob_levels": lob_levels,
        "company_idx": company_idx,
        "company_levels": company_levels,
        "latest_cum": np.stack(anchor_cum_list),
        "latest_dev": np.stack(anchor_dev_list),
        "cohorts": cohorts,
        "dropped": pd.DataFrame(dropped, columns=[*seg_cols, "reason"]),
        "origin_periods": origins,
        "n_w": n_w,
        "n_d": n_d,
        "fields": fields,
        "dev_grain_months": step,
    }


def nn_company_data(
    triangle: Triangle,
    *,
    loss_field: str = "reported_loss",
    feature_fields: tuple[str, ...] = (),
    premium_field: str = "earned_premium",
    segment_columns: tuple[str, ...] | None = None,
    lob_column: str = "line_of_business",
) -> dict[str, Any]:
    """Company-cohort variant of ``nn_data`` for the multi-line transformer.

    A cohort here is one COMPANY; its lines of business become an explicit
    axis so a model can attend across them (the learned analogue of SUR's
    contemporaneous correlation / the copula's cell dependence). Built by
    regrouping ``nn_data``'s (company, line) cohorts — screening, increment
    and premium rules are identical by construction. Lines a company does
    not write (or that were dropped) are all-zero and ``line_mask``-ed out.

    Adds over the flat contract:
    - arrays gain a line axis: ``x`` (n_c, L, F, W, D), ``obs_mask``
      (n_c, L, W, D), ``premium``/``latest_cum`` (n_c, L, W),
      ``latest_dev`` (n_c, L, W), ``log_premium`` (n_c, L);
    - ``line_mask`` (n_c, L) — lines actually present per company;
    - ``companies`` — one row per company (segment columns minus the LOB).
    """
    flat = nn_data(
        triangle,
        loss_field=loss_field,
        feature_fields=feature_fields,
        premium_field=premium_field,
        segment_columns=segment_columns,
    )
    cohorts = flat["cohorts"]
    if lob_column not in cohorts.columns:
        raise ValueError(f"no {lob_column!r} segment column; cannot form a line axis")
    company_cols = [c for c in cohorts.columns if c != lob_column]
    if not company_cols:
        raise ValueError("need at least one company-identifying segment column")

    companies = (
        cohorts[company_cols].drop_duplicates().sort_values(company_cols).reset_index(drop=True)
    )
    row_of = {tuple(r): i for i, r in enumerate(companies.itertuples(index=False))}
    n_c, n_l = len(companies), len(flat["lob_levels"])
    _, n_f, n_w, n_d = flat["x"].shape

    x = np.zeros((n_c, n_l, n_f, n_w, n_d))
    obs = np.zeros((n_c, n_l, n_w, n_d), dtype=bool)
    premium = np.full((n_c, n_l, n_w), np.nan)
    log_premium = np.zeros((n_c, n_l))
    latest_cum = np.zeros((n_c, n_l, n_w))
    latest_dev = np.zeros((n_c, n_l, n_w), dtype=int)
    line_mask = np.zeros((n_c, n_l), dtype=bool)
    for k in range(len(cohorts)):
        ci = row_of[tuple(cohorts.iloc[k][company_cols])]
        li = int(flat["lob_idx"][k])
        x[ci, li] = flat["x"][k]
        obs[ci, li] = flat["obs_mask"][k]
        premium[ci, li] = flat["premium"][k]
        log_premium[ci, li] = flat["log_premium"][k]
        latest_cum[ci, li] = flat["latest_cum"][k]
        latest_dev[ci, li] = flat["latest_dev"][k]
        line_mask[ci, li] = True

    return {
        "x": x,
        "obs_mask": obs,
        "line_mask": line_mask,
        "cal_idx": flat["cal_idx"],
        "premium": premium,
        "log_premium": log_premium,
        "lob_levels": flat["lob_levels"],
        "latest_cum": latest_cum,
        "latest_dev": latest_dev,
        "companies": companies,
        "dropped": flat["dropped"],
        "origin_periods": flat["origin_periods"],
        "n_w": n_w,
        "n_d": n_d,
        "fields": flat["fields"],
        "dev_grain_months": flat["dev_grain_months"],
    }


def cutoff_masks(
    obs_mask: np.ndarray, cal_idx: np.ndarray, cutoff: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split observed cells at a calendar cutoff: (context, target).

    context = observed and on/before the cutoff diagonal; target = observed
    and after it. Shapes broadcast over any leading cohort axes.
    """
    on_or_before = cal_idx <= cutoff
    return obs_mask & on_or_before, obs_mask & ~on_or_before
