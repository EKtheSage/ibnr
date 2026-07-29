"""Triangle -> standardized NN training data. THE neural data contract.

The NN analogue of ``contract.py``: one canonical mapping from a
multi-segment cumulative Triangle to dense numpy grids + masks, consumed by
the neural gallery entries (``gallery/nn/transformer``,
``gallery/nn/transformer_ml``). Per CLAUDE.md decision 3 the contract lives
here once and every entry consumes the identical dict; per the [nn]-extra
rule, torch conversion happens inside each entry - this module never imports
torch (``tests/test_gallery.py`` enforces the import boundary).

Cross-refs: ``kernels/contract.py`` (Stan contract; ``logprem`` there plays
the role ``premium`` plays here), ``kernels/multiline.py`` (the multi-LOB
statistical contract and the shared target layout),
``gallery/nn/transformer/card.md`` (why increments-over-premium, why the
per-dev standardization is pinned).

CORE CONVENTIONS

- A *cohort* is one distinct segment combination (company x lob, ...): the
  unit the NN pools over, and axis 0 of every per-cohort array. Order is
  ``sorted()`` over the segment tuples and is pinned by the ``cohorts`` frame.
- Grids are (n_w, n_d): origin index ``w`` (sorted origins, SHARED across all
  cohorts so a cohort missing an origin is simply masked out) x 1-based dev
  step ``d = dev_lag // dev_grain_months`` stored at index ``d - 1``.
- ``x`` holds INCREMENTAL LOSS RATIOS - incremental loss / that origin's
  premium. Dividing by premium is the cross-cohort normalizer that lets one
  network pool a $2M company with a $2B one; incrementals (rather than
  cumulatives) keep successive cells from being near-perfectly autocorrelated.
  The first dev's increment IS its cumulative, and later increments are taken
  against the IMMEDIATE predecessor dev only (mirroring
  ``transforms.to_incremental``), so a cell whose predecessor is missing has
  no usable increment and counts as unobserved.
- ``obs_mask`` therefore marks cells with a *usable increment*, NOT raw cell
  presence. Reviewers should not read it as "the triangle has a value here".
- Absent cells are represented as x = 0.0 AND obs_mask = False. The zero is
  padding for the tensor only - it is never a claim that the increment was
  zero, and every consumer must gate on the mask, never on ``x != 0``.
- ``latest_cum``/``latest_dev`` are the per-origin ANCHORS: ultimate =
  latest_cum + premium * (sum of predicted future incremental loss ratios).
  Predicting ratios and re-anchoring on the observed cumulative is what keeps
  the network from having to reproduce the level of each triangle.
- Calendar index ``cal_idx[w, d] = w + d + 1`` is the 1-based diagonal number
  (cell (0,0) sits on diagonal 1). Constant calendar time = constant
  cal_idx, which is what ``as_of``/eval_date means on a dense grid; it drives
  cutoff augmentation and the eval_date-style validation split.

THE CONTRACT - keys returned by ``nn_data`` (shape; dtype; meaning). n_c =
kept cohorts, n_f = channels (= 1 + len(feature_fields)), n_w origins,
n_d dev steps:

- ``x``              : (n_c, n_f, n_w, n_d) float64 - incremental loss ratios.
  Channel 0 is ALWAYS ``loss_field`` (the prediction target); channels 1..
  follow ``feature_fields`` order. NaN-free by construction: unusable cells
  are zero-filled (see the mask rule above).
- ``obs_mask``       : (n_c, n_w, n_d) bool - usable channel-0 increments;
  True = observed/trainable, False = to be predicted or absent.
- ``cal_idx``        : (n_w, n_d) int - 1-based diagonal number; shared by all
  cohorts (calendar time is a property of the grid, not of a cohort).
- ``premium``        : (n_c, n_w) float64 - booked earned premium per origin,
  NaN where the cohort has no positive premium for that origin (kept cohorts
  only have such NaNs on origins with no losses at all).
- ``log_premium``    : (n_c,) float64 - log of the cohort's mean origin
  premium; a size feature, the NN analogue of Stan's ``logprem`` offset.
- ``lob_idx``/``lob_levels``, ``company_idx``/``company_levels`` : (n_c,) int
  categorical codes plus their level lists (embedding vocabularies). Degrade
  to a single "all" level when the column is absent, so a network always has
  a valid embedding table.
- ``latest_cum``     : (n_c, n_w) float64 - latest OBSERVED cumulative loss
  per origin (0.0 when the origin is entirely unobserved for this cohort).
- ``latest_dev``     : (n_c, n_w) int - 1-based dev index of that anchor
  (0 = nothing observed). Future cells are ``d_index >= latest_dev``.
- ``cohorts``        : DataFrame, n_c rows - the segment values per cohort,
  row order == axis 0 of every array above. The join key back to the triangle.
- ``segment_columns``: tuple[str, ...] - ``cohorts``' columns, i.e. the cohort
  KEY schema. Named explicitly because it can be narrower than the triangle's
  segment schema (see ``display``), and a consumer comparing the two needs to
  read the fit's own answer rather than re-derive it.
- ``display``        : DataFrame, n_c rows - the segment columns that were
  DROPPED from the cohort key, one row per kept cohort, row-aligned with
  ``cohorts``. Empty-column frame when nothing was dropped. Together
  ``cohorts`` and ``display`` are a cohort's full segment identity as the
  triangle carried it, which is what ``GalleryEntry.cohorts()`` hands back and
  what lets a held-out scorer VERIFY a dropped value instead of discarding it.
  Every dropped column is guaranteed to be a function of the key (below), so
  the row is well defined.
- ``dropped``        : DataFrame - screened-out cohorts + reason; a failed
  cohort is reported, never silently vanished (retro scripts log these).
- ``origin_periods`` : list[dt.date], len n_w, ascending. ``n_w``/``n_d``:
  int sizes. ``fields``: list[str], the channel order of axis 1.
- ``dev_grain_months``: int - months per dev step.

INVARIANTS consumers rely on: the triangle must be cumulative and pre-sliced
with ``as_of(...)`` (this module uses every row it is given); one row per
(cohort, field, origin, dev) - duplicates mean several evaluation dates
survived; and axis 0 alignment between ``x``, ``obs_mask``, ``premium``,
``latest_cum``, ``latest_dev``, ``lob_idx``, ``company_idx`` and ``cohorts``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ibnr.kernels.contract import _as_date
from ibnr.triangle.core import GRAIN_MONTHS, Triangle

#: segment columns that are display-only and never define a cohort
DISPLAY_COLUMNS = ("company_name",)


def _refuse_undetermined(df: pd.DataFrame, key_cols: list[str], dropped: list[str]) -> None:
    """Every dropped segment column must be a FUNCTION of the kept ones.

    A display column is dropped from the cohort key so that two spellings of one
    company do not split it in two. That is only safe while the spelling is
    determined by the key: if ``company_code`` 0086 carries two
    ``company_name``s, the two cohorts collapse onto one key and their cells are
    pooled into one grid - silently, whenever their (origin, dev) cells happen to
    be disjoint, since the duplicate-cell guard cannot see it. Silent ambiguity
    is worse than the failure the narrowing exists to remove.

    Applies to the ``DISPLAY_COLUMNS`` default AND to an explicit
    ``segment_columns=``; the explicit path merged cohorts silently too.

    **Measured against the real mart before shipping** (publish_id
    ``20260613_041006``), because the data model derives ``company_name`` via a
    LEFT JOIN onto ``sat_company_details`` and a null or second spelling would
    fire this on every pooled NN fit. It does not: across all four Meyers lines
    the mart carries 353 distinct ``company_code``s, zero null names and zero
    codes with more than one name, so both narrowings pass - the study's pooled
    panel (60 companies / 152 cohorts) and the full ``--nn-pool market`` pool
    (221 companies / 405 cohorts) alike. If a later publish breaks that, this
    refusal is still the correct answer and the remedy is an explicit
    ``segment_columns=`` in the affected caller, not a relaxed guard.
    """
    if not dropped:
        return
    for col in dropped:
        spread = df.groupby(key_cols, dropna=False)[col].nunique()
        bad = spread[spread > 1]
        if not len(bad):
            continue
        key = bad.index[0]
        key = key if isinstance(key, tuple) else (key,)
        rows = df.loc[(df[key_cols] == list(key)).all(axis=1), col]
        values = sorted(map(str, rows.unique()))
        raise ValueError(
            f"segment column {col!r} is not determined by the cohort key {key_cols}: "
            f"key {key} carries {len(values)} values {values[:3]}. Dropping {col!r} from "
            "the key would collapse those cohorts onto one, pooling their cells into a "
            "single grid. Reconcile the values, or pass segment_columns= to keep the "
            "column in the key"
        )


def cohort_identities(contract: dict[str, Any], *, key: str = "cohorts") -> list[dict]:
    """Each cohort's FULL segment identity, in contract row order.

    The cohort KEY (``cohorts``, or ``companies`` for the company-cohort
    contract) merged with ``display``, the segment columns the key does not
    carry. That merge is what ``GalleryEntry.cohorts()`` returns for every NN
    entry, so a caller loops over the identities the triangle actually carried
    rather than the narrower key the pooling required.
    """
    frame, display = contract[key], contract["display"]
    if len(display) != len(frame):
        raise ValueError(
            f"contract['display'] has {len(display)} rows but contract[{key!r}] has "
            f"{len(frame)}; they must be row-aligned"
        )
    return [{**frame.iloc[i].to_dict(), **display.iloc[i].to_dict()} for i in range(len(frame))]


def nn_data(
    triangle: Triangle,
    *,
    loss_field: str = "reported_loss",
    feature_fields: tuple[str, ...] = (),
    premium_field: str = "earned_premium",
    segment_columns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Map a multi-segment cumulative Triangle to dense grids and masks.

    Returns the dict documented in the module docstring - the single data
    contract for every NN entry. Slice training data with
    ``triangle.as_of(...)`` before calling: this function uses every row it
    sees, so an unsliced triangle silently trains on the future.

    Unlike the Stan/multiline contracts, unusable COHORTS are dropped rather
    than raised on (missing/non-positive premium, no usable increments) and
    reported in ``dropped``: an NN fit spans hundreds of company x line
    triangles from the Schedule P mart, so one bad triangle must not kill the
    run - but it must still be visible, since a shrinking pool changes what
    the pooled model learned.
    """
    # argument validation, so it precedes every read of the triangle. This one
    # check covers nn_company_data - and so every NN entry - because that
    # function delegates here as its first statement.
    if premium_field is None:
        raise ValueError(
            "nn_data requires a premium_field: every NN entry models incremental loss "
            "RATIOS (increment / premium), so there is no premium-free form. Pass "
            "premium_field naming a per-origin premium field (default 'earned_premium')."
        )
    if triangle.meta.measure != "cumulative":
        raise ValueError("nn_data requires a cumulative triangle")
    if segment_columns is None:
        # default cohort key = every segment column except display-only ones;
        # including e.g. company_name would split a company in two whenever
        # the mart carries two spellings of its name
        segment_columns = tuple(c for c in triangle.segments if c not in DISPLAY_COLUMNS)
    seg_cols = list(segment_columns)
    if not seg_cols:
        raise ValueError("nn_data needs at least one segment column to define cohorts")
    unknown = [c for c in seg_cols if c not in triangle.segments]
    if unknown:
        raise ValueError(
            f"segment_columns {unknown} are not segment columns of this triangle "
            f"({list(triangle.segments)})"
        )
    # segment columns the cohort key does NOT carry; they ride along as `display`
    display_cols = [c for c in triangle.segments if c not in seg_cols]

    # channel order of axis 1: target first, then features (fixed, relied on
    # downstream as `x[:, 0]` = the prediction target)
    fields = [loss_field, *feature_fields]
    if len(set(fields)) != len(fields):
        raise ValueError(f"duplicate fields in loss_field + feature_fields: {fields}")
    df = triangle.select_fields(fields).execute()
    if df.empty:
        raise ValueError(f"no rows for fields {fields}")
    # BEFORE the duplicate-cell guard below, deliberately. Both can fire on the
    # same triangle, and "this column is not determined by the key" is the cause
    # while "multiple rows per cell" is only the symptom - and when the merged
    # cohorts' cells happen to be disjoint the duplicate guard does not fire at
    # all, which is the case this exists for.
    _refuse_undetermined(df, seg_cols, display_cols)
    df = df.copy()
    df["origin_period"] = _as_date(df["origin_period"])
    # dev_lag is months from origin start (CLAUDE.md milestone 1); d is the
    # 1-based dev step, stored at index d-1 in every grid below
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    if (df["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")
    df["d"] = (df["dev_lag"] // step).astype(int)
    if (df["d"] < 1).any():
        raise ValueError("dev_lag must be positive")
    # duplicates = several evaluation dates survived; the grid write below
    # would keep whichever row landed last, so refuse instead
    if df.duplicated([*seg_cols, "field", "origin_period", "d"]).any():
        raise ValueError(
            "multiple rows per (cohort, field, origin, dev) cell; "
            "slice with as_of()/latest_diagonal() first"
        )

    # ONE origin/dev grid shared by all cohorts: pooling requires every cohort
    # to sit in the same (w, d) coordinate system, and a cohort that lacks an
    # origin is just masked out rather than shifted
    origins = sorted(df["origin_period"].unique())
    n_w, n_d = len(origins), int(df["d"].max())
    w_of = {o: w for w, o in enumerate(origins)}
    n_f = len(fields)

    # premium is an origin-level exposure measure: take its latest evaluation
    # (the booked value), same convention as contract._premium_by_origin
    premium_df = triangle.select_fields(premium_field).latest_diagonal().execute()
    # select_fields is a filter, so a field the triangle does not carry yields
    # an empty frame rather than an error - and every cohort would then fail the
    # premium screen below one at a time, leaving "no usable cohorts" as the
    # only message. Refuse by name, wording matched to contract._premium_by_origin
    # and multiline._premium_by_lob_origin so all three contracts answer alike.
    if premium_df.empty:
        raise ValueError(f"no rows for premium field {premium_field!r}")
    premium_df = premium_df.copy()
    premium_df["origin_period"] = _as_date(premium_df["origin_period"])

    def norm(key) -> tuple:
        # pandas groupby yields a scalar key for a single group column
        return key if isinstance(key, tuple) else (key,)

    # sorted tuple order defines cohort axis 0 (and the `cohorts` row order)
    cohort_keys = sorted(map(tuple, df[seg_cols].drop_duplicates().itertuples(index=False)))
    grouped = {norm(k): g for k, g in df.groupby(seg_cols)}
    prem_grouped = (
        {norm(k): g for k, g in premium_df.groupby(seg_cols)} if not premium_df.empty else {}
    )

    kept: list[dict] = []
    kept_display: list[dict] = []
    dropped: list[dict] = []
    x_list, obs_list, prem_list, anchor_cum_list, anchor_dev_list = [], [], [], [], []

    for key in cohort_keys:
        sub = grouped[key]
        row = dict(zip(seg_cols, key, strict=True))

        prem_sub = prem_grouped.get(key)
        premium = np.full(n_w, np.nan)  # (n_w,) NaN = no premium for that origin
        if prem_sub is not None:
            for r in prem_sub.itertuples():
                premium[w_of[r.origin_period]] = float(r.value)

        # scatter long rows into the dense cumulative grid; NaN = unobserved
        cum = np.full((n_f, n_w, n_d), np.nan)  # (n_f, n_w, n_d)
        for r in sub.itertuples():
            cum[fields.index(r.field), w_of[r.origin_period], int(r.d) - 1] = float(r.value)

        # increments against the immediate predecessor; first dev = cumulative.
        # Subtracting NaN propagates NaN, which is exactly the rule we want: a
        # cell whose predecessor is missing has NO usable increment (we refuse
        # to bridge a gap, which would fabricate development).
        incr = np.full_like(cum, np.nan)  # (n_f, n_w, n_d)
        incr[:, :, 0] = cum[:, :, 0]
        incr[:, :, 1:] = cum[:, :, 1:] - cum[:, :, :-1]
        obs = ~np.isnan(incr[0])  # (n_w, n_d) usable TARGET-channel increments

        # screen 1: every origin that has losses must have positive premium -
        # x is loss/premium, so a zero/missing denominator is unusable, and
        # dropping the whole cohort keeps its origin axis interpretable
        origin_has_loss = ~np.all(np.isnan(cum[0]), axis=1)  # (n_w,)
        bad_premium = origin_has_loss & ~(premium > 0)  # NaN > 0 is False
        if bad_premium.any():
            dropped.append({**row, "reason": "missing or non-positive premium"})
            continue
        # screen 2: nothing to train on
        if not obs.any():
            dropped.append({**row, "reason": f"no usable {loss_field!r} increments"})
            continue

        # incremental LOSS RATIOS: divide by the origin's premium so cohorts of
        # wildly different size share one scale (the NN's cross-cohort
        # normalizer, cf. logprem in stan_data). NaN/inf here only arise on
        # cells the mask already excludes.
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = incr / premium[None, :, None]  # (n_f, n_w, n_d)
        x = np.where(np.isnan(ratios), 0.0, ratios)
        # non-target channels may be missing where the target is observed; zero-fill
        # so `x` is NaN-free. This zero is PADDING, not a zero increment -
        # consumers must gate on obs_mask, never on `x != 0`.
        x[:, ~obs] = 0.0

        # per-origin anchors from the CUMULATIVE grid (not the increments): the
        # deepest observed dev and its cumulative loss. Ultimate is rebuilt as
        # latest_cum + premium * sum(predicted future ratios), so the network
        # never has to reproduce the level of the triangle.
        latest_dev = np.zeros(n_w, dtype=int)  # (n_w,) 1-based; 0 = nothing observed
        latest_cum = np.zeros(n_w)  # (n_w,) dollars at that anchor
        cum_obs = ~np.isnan(cum[0])
        for w in range(n_w):
            devs = np.nonzero(cum_obs[w])[0]
            if devs.size:
                latest_dev[w] = int(devs[-1]) + 1  # 0-based index -> 1-based dev step
                latest_cum[w] = cum[0, w, devs[-1]]

        kept.append(row)
        # _refuse_undetermined above guarantees one value per key, so .iloc[0]
        # is exact rather than a sample
        kept_display.append({col: sub[col].iloc[0] for col in display_cols})
        x_list.append(x)
        obs_list.append(obs)
        # normalize "no premium" to NaN (a stray 0 would divide-by-zero later)
        prem_list.append(np.where(premium > 0, premium, np.nan))
        anchor_cum_list.append(latest_cum)
        anchor_dev_list.append(latest_dev)

    if not kept:
        raise ValueError("no usable cohorts after screening; see the dropped reasons")

    cohorts = pd.DataFrame(kept)
    # reindex, not construct-and-hope: with no display columns the list is all
    # empty dicts and pandas hands back a 0-row frame
    display = pd.DataFrame(kept_display, columns=display_cols).reindex(range(len(kept)))
    premium = np.stack(prem_list)

    # categorical codes for embedding tables. When the column is absent the
    # level list degrades to a single "all" so a network always has a valid
    # (size >= 1) vocabulary instead of a special case.
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

    # calendar/diagonal index: cells with equal w + d share an evaluation date,
    # so cal_idx is the dense-grid stand-in for eval_date. +1 makes it the
    # 1-based diagonal number (the first diagonal, dev 12 months, is 1).
    w_grid, d_grid = np.meshgrid(np.arange(n_w), np.arange(n_d), indexing="ij")
    # see the module docstring for the per-key contract
    return {
        "x": np.stack(x_list),  # (n_c, n_f, n_w, n_d)
        "obs_mask": np.stack(obs_list),  # (n_c, n_w, n_d)
        "cal_idx": w_grid + d_grid + 1,  # (n_w, n_d), shared by all cohorts
        "premium": premium,  # (n_c, n_w)
        # cohort size feature: log of the MEAN origin premium (nanmean skips
        # origins this cohort does not write). The NN analogue of stan_data's
        # logprem offset.
        "log_premium": np.log(np.nanmean(premium, axis=1)),  # (n_c,)
        "lob_idx": lob_idx,
        "lob_levels": lob_levels,
        "company_idx": company_idx,
        "company_levels": company_levels,
        "latest_cum": np.stack(anchor_cum_list),  # (n_c, n_w)
        "latest_dev": np.stack(anchor_dev_list),  # (n_c, n_w), 1-based
        "cohorts": cohorts,  # n_c rows, row order == axis 0 of every array
        "segment_columns": tuple(seg_cols),  # the cohort KEY schema
        # segment columns the key does not carry, row-aligned with `cohorts`;
        # together they are a cohort's full identity
        "display": display,
        # screened-out cohorts stay visible: a shrinking pool changes the fit
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
    regrouping ``nn_data``'s (company, line) cohorts - screening, increment
    and premium rules are identical by construction. Lines a company does
    not write (or that were dropped) are all-zero and ``line_mask``-ed out.

    Adds over the flat contract:
    - arrays gain a line axis: ``x`` (n_c, L, F, W, D), ``obs_mask``
      (n_c, L, W, D), ``premium``/``latest_cum`` (n_c, L, W),
      ``latest_dev`` (n_c, L, W), ``log_premium`` (n_c, L);
    - ``line_mask`` (n_c, L) - lines actually present per company;
    - ``companies`` - one row per company (segment columns minus the LOB), with
      ``segment_columns`` naming that schema and ``display`` carrying the
      dropped segment columns row-aligned to it.

    Alignment guarantees (what makes cross-line attention meaningful):
    - The line axis is ``lob_levels`` - a GLOBAL vocabulary shared by every
      company, so index ``li`` means the same line everywhere and an embedding
      or an attention head can be compared across companies.
    - The (w, d) grid and ``cal_idx`` are shared across companies AND lines:
      calendar time is a property of the grid, so one cutoff diagonal applies
      to all of a company's lines at once (which is what makes a company a
      single joint training example).
    - Absent lines are all-zero across ``x``/``premium``/anchors and False in
      ``line_mask``; as in the flat contract those zeros are padding, so a
      consumer MUST gate on ``line_mask`` (and ``obs_mask``) rather than on
      the values. ``log_premium`` is 0 there too - a padding value, not a
      $1 premium.
    - Unlike ``kernels.multiline``, lines here are NOT required to share an
      observed-cell pattern: the mask is what the attention consumes, so
      ragged lines are representable rather than an error.
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
    # a company is a cohort key with the LOB dimension removed
    company_cols = [c for c in cohorts.columns if c != lob_column]
    if not company_cols:
        raise ValueError("need at least one company-identifying segment column")

    # THE KEY NARROWS AGAIN HERE (cohort minus the LOB), so the same rule has to
    # hold a second time: a display column must be determined by the COMPANY
    # columns, or two companies would merge onto one row of the line axis.
    display_cols = list(flat["display"].columns)
    identity = pd.concat([cohorts, flat["display"]], axis=1)
    _refuse_undetermined(identity, company_cols, display_cols)

    companies = (
        cohorts[company_cols].drop_duplicates().sort_values(company_cols).reset_index(drop=True)
    )
    row_of = {tuple(r): i for i, r in enumerate(companies.itertuples(index=False))}
    # one display row per COMPANY, row-aligned with `companies` (a left merge, so
    # the row order above is what survives)
    company_display = companies.merge(
        identity[[*company_cols, *display_cols]].drop_duplicates(subset=company_cols),
        on=company_cols,
        how="left",
    ).drop(columns=company_cols)
    # L is the GLOBAL number of lines, not this company's count - every company
    # gets a full-width line axis and unwritten lines are masked off
    n_c, n_l = len(companies), len(flat["lob_levels"])
    _, n_f, n_w, n_d = flat["x"].shape

    # zero/False prefill IS the "line absent" representation (see docstring)
    x = np.zeros((n_c, n_l, n_f, n_w, n_d))
    obs = np.zeros((n_c, n_l, n_w, n_d), dtype=bool)
    premium = np.full((n_c, n_l, n_w), np.nan)
    log_premium = np.zeros((n_c, n_l))
    latest_cum = np.zeros((n_c, n_l, n_w))
    latest_dev = np.zeros((n_c, n_l, n_w), dtype=int)
    line_mask = np.zeros((n_c, n_l), dtype=bool)
    # scatter each flat (company, line) cohort into its (ci, li) slot; because
    # this is a pure regrouping of nn_data's output, the screening, increment
    # and premium rules are identical by construction
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
        "x": x,  # (n_c, L, n_f, n_w, n_d)
        "obs_mask": obs,  # (n_c, L, n_w, n_d) usable target increments
        "line_mask": line_mask,  # (n_c, L) lines this company writes
        "cal_idx": flat["cal_idx"],  # (n_w, n_d), shared by companies AND lines
        "premium": premium,  # (n_c, L, n_w)
        "log_premium": log_premium,  # (n_c, L), 0 on absent lines (padding)
        "lob_levels": flat["lob_levels"],  # global line vocabulary; index = L axis
        "latest_cum": latest_cum,  # (n_c, L, n_w) anchors
        "latest_dev": latest_dev,  # (n_c, L, n_w), 1-based
        "companies": companies,  # n_c rows, row order == axis 0
        "segment_columns": tuple(company_cols),  # the COMPANY key schema
        "display": company_display,  # dropped segment columns, aligned to `companies`
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
    and after it. This is a *simulated* ``as_of``: pretending the valuation
    date was ``cutoff`` turns one historical triangle into a supervised
    "predict the next diagonals" task, which is both the training-time
    augmentation that multiplies tiny Schedule P triangles and the shape of
    the eval_date validation split (see ``gallery/nn/transformer/model.py``).
    Conditioning is inclusive of the cutoff diagonal because a real valuation
    at that date can see it.

    ``cal_idx`` is (n_w, n_d) while ``obs_mask`` carries leading cohort/line
    axes; numpy broadcasting aligns them, so both returned masks have
    ``obs_mask``'s shape. Both are subsets of ``obs_mask``, so unobserved and
    absent cells appear in neither - future cells to be PREDICTED are exactly
    the ones in neither mask.
    """
    on_or_before = cal_idx <= cutoff
    return obs_mask & on_or_before, obs_mask & ~on_or_before
