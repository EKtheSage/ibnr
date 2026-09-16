"""Triangle -> standardized model data. The Stan ``data`` block is the contract.

``stan_data`` produces the dict consumed by the cross-classified lognormal
family (Meyers CRC/CCL/CSR). NumPyro and PyMC implementations consume the
IDENTICAL dict - no backend grows its own data prep.

``cohort_grid`` is the array-shaped sibling for the models that are not
likelihood-in-Stan at all (the distribution-free chain ladder of
``kernels/mack.py`` and the one-year CDR of ``kernels/cdr.py``): same single
cohort, same validation, but a dense ``(n_w, n_d)`` matrix rather than the
ragged Stan vectors. It is the single-cohort analogue of
``multiline.multiline_data`` - see that module for the multi-LOB grid.

Conventions:
- ``w``/``d`` are 1-based origin/dev indices (Stan style), sorted by (w, d).
- ``prev_idx[i]`` is the 1-based row index of the observation at
  (w[i]-1, d[i]) - the **previous ORIGIN at the same development**, which is
  CCL's accident-year AR(1) link - or 0 when w[i] == 1. Because rows are sorted
  by (w, d), prev_idx[i] < i+1 always, so mu can be built in one forward pass.
  It is emphatically **not** the previous development cell (w[i], d[i]-1); that
  is what differencing a cumulative triangle needs, and
  ``kernels.holdout.training_index`` computes it separately. Reading one for the
  other is a silent, plausible-looking error, and it has happened once. The name
  is Stan's - ``model.stan`` declares ``prev_idx`` in its data block - so it
  stays, and the ambiguity is answered here instead.
- ``segment`` / ``fields`` record WHICH cohort and which field(s) the contract
  was built from. Nothing in the Stan block uses them; they exist so that a
  scorer handed cells cannot evaluate another company's, or another field's,
  data against this fit. ``(w, d)`` alone does not identify a cell - two
  companies share origin dates and development lags exactly.
- ``logprem`` is per-observation; ``premium`` is per-origin (each origin's
  premium at its latest eval in the triangle, i.e. the booked value).
- ``cohort_grid``'s matrix is 0-based on both axes: ``cum[i, j]`` is origin
  ``i`` at dev step ``j + 1`` (dev_lag ``(j + 1) * dev_grain_months``).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from ibnr.triangle.core import GRAIN_MONTHS, Triangle


def _cohort_identity(
    triangle: Triangle,
    df: pd.DataFrame,
    fields: tuple[str, ...],
    *,
    models: tuple[str, ...] | None = None,
) -> dict:
    """Which cohort, which field(s) and which measure this contract describes.

    Carried on every contract so a scorer can refuse cells that are not this
    fit's. The single-cohort check above guarantees one combination, so taking
    the first row is exact rather than a sample.

    ``fields`` are the SOURCE fields read from the triangle. ``models`` is what
    the entry actually puts a likelihood on, when that differs - compartmental
    reads ``paid`` and ``reported`` but models paid and *outstanding*
    (reported - paid), so raw reported-loss cells are not something it can
    score even though ``reported_loss`` appears in ``fields``.

    ``measure`` matters because a cumulative fit handed incremental cells scores
    them at the identical ``(w, d)`` positions and returns finite, plausible,
    entirely wrong log densities - measured at a total of -3383 against a
    correct -35.
    """
    segs = triangle.segments
    segment = {s: df[s].iloc[0] for s in segs} if segs and len(df) else {}
    return {
        "segment": segment,
        "fields": tuple(fields),
        "models": tuple(models if models is not None else fields),
        "measure": triangle.meta.measure,
    }


def stan_data(
    triangle: Triangle,
    *,
    loss_field: str,
    premium_field: str | None = None,
) -> dict[str, Any]:
    """Map a single-cohort cumulative Triangle to the standardized data dict.

    The triangle must contain exactly one segment combination (one company x
    line); slice with ``triangle.filter`` first. Slice training data with
    ``triangle.as_of(...)`` before calling - this function uses every row.

    Consecutive origins must also be one dev step apart, because ``prev_idx``
    reads the step from one origin to the next as one elapsed development period
    (see :func:`require_origin_axis_step`, which says what that costs the one
    entry here that does not read ``w`` that way).
    """
    if triangle.meta.measure != "cumulative":
        raise ValueError("stan_data requires a cumulative triangle")
    df = triangle.select_fields(loss_field).execute()
    if df.empty:
        raise ValueError(f"no rows for loss field {loss_field!r}")
    segs = triangle.segments
    if segs and len(df.drop_duplicates(segs)) > 1:
        raise ValueError(
            f"triangle has multiple segment combinations on {segs}; filter to one cohort first"
        )

    df = df.copy()
    df["origin_period"] = _as_date(df["origin_period"])
    df["eval_date"] = _as_date(df["eval_date"])
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    df["d"] = dev_step_index(df["dev_lag"], step=step)

    origins = sorted(df["origin_period"].unique())
    # w and d are one shared calendar clock here - prev_idx links origin w-1 to
    # origin w as one elapsed development period - so the origin axis has to step
    # by exactly one dev step
    require_origin_axis_step(origins, step=step)
    n_w, n_d = len(origins), int(df["d"].max())
    w_of = {o: i + 1 for i, o in enumerate(origins)}

    df["w"] = df["origin_period"].map(w_of)
    if df.duplicated(["w", "d"]).any():
        raise ValueError(
            "multiple rows per (origin, dev) cell; slice with as_of()/latest_diagonal() first"
        )
    nonpos = df["value"] <= 0
    if nonpos.any():
        raise ValueError(
            f"{int(nonpos.sum())} cells have non-positive {loss_field!r}; "
            "lognormal models need positive losses (Meyers' selection excludes these)"
        )
    df = df.sort_values(["w", "d"]).reset_index(drop=True)

    row_of = {(int(r.w), int(r.d)): i + 1 for i, r in df.iterrows()}
    prev_idx = np.array(
        [row_of.get((int(r.w) - 1, int(r.d)), 0) if r.w > 1 else 0 for r in df.itertuples()],
        dtype=int,
    )

    data: dict[str, Any] = {
        "len_data": len(df),
        "n_w": n_w,
        "n_d": n_d,
        "w": df["w"].to_numpy(dtype=int),
        "d": df["d"].to_numpy(dtype=int),
        "prev_idx": prev_idx,
        "logloss": np.log(df["value"].to_numpy(dtype=float)),
        # metadata (not part of the Stan data block proper)
        **_cohort_identity(triangle, df, (loss_field,)),
        "origin_periods": origins,
        "dev_grain_months": step,
        "loss": df["value"].to_numpy(dtype=float),
    }

    if premium_field is not None:
        premium = premium_by_origin(
            triangle, premium_field, origins, segment=_cohort_identity(triangle, df, ())["segment"]
        )
        data["premium"] = premium
        data["logprem"] = np.log(premium)[data["w"] - 1]
    return data


def premium_by_origin(
    triangle: Triangle,
    premium_field: str,
    origins: list,
    *,
    segment: dict | None = None,
) -> np.ndarray:
    """One exposure per origin, from THIS cohort.

    Public, and called from outside this module: ``kernels.replay`` grids one
    historical slice per replay date and attaches that slice's own premium to
    it, so it needs the same exposure rule the Stan contracts here read by.

    The single-cohort guards above are applied to the LOSS rows, because that is
    the frame they build from. Premium is a different field, so it needs its own
    check - and the check has to be that it MATCHES the loss cohort, not merely
    that it is internally consistent.

    Both halves matter, and only the first was obvious. A triangle with one
    company's losses and two companies' premium picked whichever premium row
    sorted first: half the origins on another line's exposure, 7.8x out. But a
    triangle whose premium rows are *entirely* another company's is perfectly
    consistent with itself, passes any uniqueness test, and is completely wrong -
    measured, ``segment={'company': 'co1'}`` recorded on a contract carrying
    co2's premium throughout.

    Wrong premium is wrong ``mu`` for every entry in the lognormal family, and a
    wrong Jacobian for anything on a loss-ratio measure.
    """
    pdf = triangle.select_fields(premium_field).latest_diagonal().execute()
    if pdf.empty:
        raise ValueError(f"no rows for premium field {premium_field!r}")
    segs = triangle.segments
    if segs and len(pdf.drop_duplicates(segs)) > 1:
        raise ValueError(
            f"premium field {premium_field!r} spans multiple segment combinations on "
            f"{segs}; filter to one cohort first (the loss field is already single-cohort, "
            "so this is exposure from another cohort)"
        )
    if segment:
        actual = {k: pdf[k].iloc[0] for k in segment if k in pdf.columns}
        if actual != segment:
            raise ValueError(
                f"premium field {premium_field!r} belongs to cohort {actual}, but the loss "
                f"field belongs to {segment}. A consistent set of premium rows for the WRONG "
                "cohort passes every uniqueness check and is entirely wrong"
            )
    pdf = pdf.copy()
    pdf["origin_period"] = _as_date(pdf["origin_period"])
    if pdf["origin_period"].duplicated().any():
        dupes = sorted(pdf.loc[pdf["origin_period"].duplicated(), "origin_period"].unique())
        raise ValueError(
            f"premium field {premium_field!r} has multiple rows for origin(s) {dupes} after "
            "collapsing to the latest diagonal; exposure must be one value per origin"
        )
    by_origin = pdf.set_index("origin_period")["value"]
    missing = [o for o in origins if o not in by_origin.index]
    if missing:
        raise ValueError(f"premium missing for origins {missing}")
    premium = by_origin.loc[origins].to_numpy(dtype=float)
    if len(premium) != len(origins):
        raise ValueError(f"premium has {len(premium)} values for {len(origins)} origins")
    if (premium <= 0).any():
        raise ValueError("non-positive premium; lognormal exposure models need positive premium")
    return premium


def odp_stan_data(
    triangle: Triangle,
    *,
    loss_field: str,
    premium_field: str | None = None,
) -> dict[str, Any]:
    """Map a single-cohort cumulative Triangle to the incremental ODP dict.

    Same conventions as ``stan_data`` (1-based ``w``/``d`` sorted by (w, d)),
    but the observations are *incremental* losses ``inc_loss`` differenced
    within each origin - the over-dispersed Poisson family models increments,
    not cumulatives. Zero increments are legitimate; negative increments are
    rejected (the ODP quasi-likelihood is undefined there, exactly the
    limitation of the bootstrap ODP baselines). Also carries ``paid_to_date``
    (the latest observed cumulative per origin) and ``latest_d`` (its dev
    index) - the anchors the predictive simulation completes from.
    """
    if triangle.meta.measure != "cumulative":
        raise ValueError("odp_stan_data requires a cumulative triangle")
    df = triangle.select_fields(loss_field).execute()
    if df.empty:
        raise ValueError(f"no rows for loss field {loss_field!r}")
    segs = triangle.segments
    if segs and len(df.drop_duplicates(segs)) > 1:
        raise ValueError(
            f"triangle has multiple segment combinations on {segs}; filter to one cohort first"
        )

    df = df.copy()
    df["origin_period"] = _as_date(df["origin_period"])
    df["eval_date"] = _as_date(df["eval_date"])
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    df["d"] = dev_step_index(df["dev_lag"], step=step)

    origins = sorted(df["origin_period"].unique())
    n_w, n_d = len(origins), int(df["d"].max())
    w_of = {o: i + 1 for i, o in enumerate(origins)}

    df["w"] = df["origin_period"].map(w_of)
    if df.duplicated(["w", "d"]).any():
        raise ValueError(
            "multiple rows per (origin, dev) cell; slice with as_of()/latest_diagonal() first"
        )
    df = df.sort_values(["w", "d"]).reset_index(drop=True)

    # incremental differencing needs contiguous dev cells within each origin
    for w_idx, grp in df.groupby("w"):
        devs = grp["d"].tolist()
        if devs != list(range(1, len(devs) + 1)):
            raise ValueError(
                f"origin index {w_idx} has non-contiguous dev lags {devs}; "
                "incremental differencing would fabricate increments"
            )
    inc = df.groupby("w")["value"].diff()
    inc = inc.fillna(df["value"]).to_numpy(dtype=float)  # first dev = cumulative
    if (inc < 0).any():
        raise ValueError(
            f"{int((inc < 0).sum())} negative incremental {loss_field!r} cells; "
            "the over-dispersed Poisson likelihood requires non-negative increments"
        )

    latest = df.loc[df.groupby("w")["d"].idxmax()].sort_values("w")
    data: dict[str, Any] = {
        "len_data": len(df),
        "n_w": n_w,
        "n_d": n_d,
        "w": df["w"].to_numpy(dtype=int),
        "d": df["d"].to_numpy(dtype=int),
        "inc_loss": inc,
        # metadata (not part of the Stan data block proper)
        **_cohort_identity(triangle, df, (loss_field,)),
        "origin_periods": origins,
        "dev_grain_months": step,
        "loss": df["value"].to_numpy(dtype=float),
        "paid_to_date": latest["value"].to_numpy(dtype=float),
        "latest_d": latest["d"].to_numpy(dtype=int),
    }

    if premium_field is not None:
        premium = premium_by_origin(
            triangle, premium_field, origins, segment=_cohort_identity(triangle, df, ())["segment"]
        )
        data["premium"] = premium
        data["logprem"] = np.log(premium)[data["w"] - 1]
    return data


def compartmental_stan_data(
    triangle: Triangle,
    *,
    paid_field: str,
    reported_field: str,
    premium_field: str,
) -> dict[str, Any]:
    """Map a single-cohort cumulative Triangle to the compartmental (joint
    paid + outstanding) dict.

    The Gesmann & Morris compartmental family fits BOTH processes at once:
    cumulative paid (``delta = 1``) and case outstanding (``delta = 0``,
    computed here as ``reported_field - paid_field`` - a level, not a
    cumulative). Rows are the stacked cells sorted by (delta, w, d), with
    ``t`` the development age in YEARS at the cell's period end (t = d for
    annual grains) - the monograph's ODE rate parameters are per-year, and
    its wkcomp case study measures t exactly this way (Lag = 1..10).

    Both fields must be present on the same (w, d) cells (inner-join
    semantics would silently drop data - mismatches raise instead). No
    positivity is enforced: the Gaussian variant takes any value; the
    lognormal variant drops its own non-positive cells and must document
    the count. Carries ``paid_to_date``/``latest_d`` (per-origin anchors)
    and per-origin ``premium`` like the other contracts.
    """
    if premium_field is None:
        raise ValueError(
            "compartmental_stan_data requires a premium_field: the compartmental family "
            "models losses per unit of premium (premium is the exposure the ODE flows "
            "through), so pass premium_field naming a per-origin premium field - the "
            "compartmental entry's own default is 'earned_premium'."
        )
    if triangle.meta.measure != "cumulative":
        raise ValueError("compartmental_stan_data requires a cumulative triangle")
    df = triangle.select_fields([paid_field, reported_field]).execute()
    if df.empty:
        raise ValueError(f"no rows for fields {paid_field!r}/{reported_field!r}")
    segs = triangle.segments
    if segs and len(df.drop_duplicates(segs)) > 1:
        raise ValueError(
            f"triangle has multiple segment combinations on {segs}; filter to one cohort first"
        )

    df = df.copy()
    df["origin_period"] = _as_date(df["origin_period"])
    df["eval_date"] = _as_date(df["eval_date"])
    step = GRAIN_MONTHS[triangle.meta.dev_grain]
    # the same check runs on `wide` below, where d is actually computed; this one
    # only moves the refusal earlier, so a triangle whose ages are off the grain
    # boundary is named by its geometry rather than by whichever field the pivot
    # then reports as missing on some cells
    dev_step_index(df["dev_lag"], step=step)

    wide = df.pivot_table(
        index=["origin_period", "dev_lag"], columns="field", values="value", aggfunc="first"
    )
    counts = df.groupby(["origin_period", "dev_lag", "field"]).size()
    if (counts > 1).any():
        raise ValueError(
            "multiple rows per (origin, dev) cell; slice with as_of()/latest_diagonal() first"
        )
    for field in (paid_field, reported_field):
        if field not in wide.columns or wide[field].isna().any():
            raise ValueError(
                f"{field!r} is missing on some (origin, dev) cells; the compartmental "
                "model needs paid and reported on the identical cells"
            )
    wide = wide.reset_index()

    origins = sorted(wide["origin_period"].unique())
    wide["d"] = dev_step_index(wide["dev_lag"], step=step)
    n_w, n_d = len(origins), int(wide["d"].max())
    w_of = {o: i + 1 for i, o in enumerate(origins)}
    wide["w"] = wide["origin_period"].map(w_of)
    wide = wide.sort_values(["w", "d"]).reset_index(drop=True)

    # the paid anchors (and the lognormal variant's incremental differencing)
    # need contiguous dev cells within each origin
    for w_idx, grp in wide.groupby("w"):
        devs = grp["d"].tolist()
        if devs != list(range(1, len(devs) + 1)):
            raise ValueError(
                f"origin index {w_idx} has non-contiguous dev lags {devs}; "
                "paid-to-date anchoring would fabricate cells"
            )

    paid = wide[paid_field].to_numpy(dtype=float)
    outstanding = (wide[reported_field] - wide[paid_field]).to_numpy(dtype=float)
    w_arr = wide["w"].to_numpy(dtype=int)
    d_arr = wide["d"].to_numpy(dtype=int)

    # stacked rows: the outstanding block (delta = 0) then the paid block
    # (delta = 1), each sorted by (w, d)
    latest = wide.loc[wide.groupby("w")["d"].idxmax()].sort_values("w")
    data: dict[str, Any] = {
        "len_data": 2 * len(wide),
        "n_w": n_w,
        "n_d": n_d,
        "w": np.concatenate([w_arr, w_arr]),
        "d": np.concatenate([d_arr, d_arr]),
        "t": np.concatenate([d_arr, d_arr]).astype(float) * (step / 12.0),
        "delta": np.concatenate([np.zeros(len(wide), dtype=int), np.ones(len(wide), dtype=int)]),
        "loss": np.concatenate([outstanding, paid]),
        # metadata (not part of the Stan data block proper)
        # `df`, not `wide`: the pivot indexes on (origin_period, dev_lag) only,
        # so it has already dropped the segment columns identity is read from.
        # `models` is paid + OUTSTANDING: reported_loss is read but never
        # modelled directly, so a raw reported-loss cell is not scorable here.
        **_cohort_identity(
            triangle,
            df,
            (paid_field, reported_field),
            models=(paid_field, "outstanding"),
        ),
        "origin_periods": origins,
        "dev_grain_months": step,
        "paid_to_date": latest[paid_field].to_numpy(dtype=float),
        "latest_d": latest["d"].to_numpy(dtype=int),
    }
    data["premium"] = premium_by_origin(
        triangle,
        premium_field,
        origins,
        segment=_cohort_identity(triangle, df, ())["segment"],
    )
    return data


def ccl_mu_index(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Static arrays that turn the CCL ``mu`` recurrence into one matmul.

    ``model.stan`` builds mu with a forward recurrence over ``prev_idx``:

        mu[w, d] = base[w, d] + rho * (logloss[w-1, d] - mu[w-1, d])   (w > 1)

    where base[w, d] = logprem + logelr + alpha[w] + beta[d]. Within a fixed
    dev column this linear recurrence has the exact closed form

        mu[w, d] = sum_{k=1..w} (-rho)^(w-k) * B[k, d],
        B[k, d]  = base[k, d] + rho * logloss[k-1, d]   (B[1, d] = base[1, d])

    so mu = P(rho) @ B with P[i, j] = (-rho)^(w_i - w_j) for cells j in the same
    column as i with w_j <= w_i (else 0). ``P`` is the only rho-dependent piece;
    everything here is data, precomputed once. This form keeps the NumPyro/PyMC
    autodiff graphs tiny (an N x N matmul) instead of an N-deep scalar chain -
    the PyTensor C-compile of the unrolled chain is the dominant cost otherwise.
    Returns ``expo`` (N x N, the exponents w_i - w_j), ``colmask`` (N x N, 1.0
    where j contributes to i), and ``logloss_prev`` (N, the observed previous-
    origin loss, 0 where absent).
    """
    w = np.asarray(data["w"], dtype=int)
    d = np.asarray(data["d"], dtype=int)
    prev0 = np.asarray(data["prev_idx"], dtype=int) - 1  # -1 = no previous cell
    logloss = np.asarray(data["logloss"], dtype=float)

    same_col = (d[:, None] == d[None, :]) & (w[None, :] <= w[:, None])
    expo = np.where(same_col, w[:, None] - w[None, :], 0).astype(int)
    colmask = same_col.astype(float)
    logloss_prev = np.where(prev0 >= 0, logloss[np.where(prev0 >= 0, prev0, 0)], 0.0)
    return {"expo": expo, "colmask": colmask, "logloss_prev": logloss_prev}


def cohort_grid(
    triangle: Triangle,
    *,
    loss_field: str,
    premium_field: str | None = None,
) -> dict[str, Any]:
    """Map a single-cohort cumulative Triangle to a dense ``(n_w, n_d)`` matrix.

    THE CONTRACT (keys; shape; meaning):

    - ``n_w``, ``n_d``       : int - origin count and deepest observed dev step.
    - ``cum``                : (n_w, n_d) float64 - cumulative loss, NaN where
      unobserved. ``cum[i, j]`` is origin ``i`` at dev step ``j + 1``.
    - ``obs_mask``           : (n_w, n_d) bool - ``~isnan(cum)``.
    - ``latest_dev``         : (n_w,) int - 0-based dev index of each origin's
      last observed cell, i.e. its position on the latest diagonal.
    - ``origin_periods``     : list[dt.date], len n_w, sorted ascending.
    - ``dev_grain_months``   : int - months per dev step.
    - ``units``, ``loss_field`` : carried through for labelling.
    - ``premium``            : (n_w,) float64, only when ``premium_field`` is given.
    - ``segment``, ``fields``, ``models``, ``measure`` : the cohort identity, same
      as the Stan contracts (see :func:`_cohort_identity`) - what lets
      ``kernels.holdout.index_into`` refuse cells that are not this fit's.
    - ``w``, ``d``           : (n_cells,) int, **1-based** origin/dev index of every
      observed cell (Stan style, like ``stan_data``), row-major over the grid.
      Exists for the training-overlap refusal in ``index_into``: a held-out score
      computed on training cells is not wrong-looking, it is flattering.

    Unlike the Stan contracts above, the consumers here (Mack's distribution-free
    chain ladder and the Merz-Wuthrich one-year CDR) are *recursive over the
    diagonal*: they estimate one development factor per dev step from the cells
    above the latest diagonal, then roll each origin forward from that diagonal.
    That only has meaning on a genuine run-off shape, so the observed set must be

        observed(i, j)  <=>  j <= min(K - i, n_d - 1),   K = latest_dev[-1] + n_w - 1

    (a staircase whose steps fall by one origin per dev step, flattening once an
    origin is fully developed). Anything else - an interior hole, a ragged
    diagonal from mixed evaluation dates - is a hard error rather than a repair:
    silently filling it would fabricate the very cells the factors are estimated
    from. Slice with ``as_of()`` first if the triangle carries several diagonals.
    """
    if triangle.meta.measure != "cumulative":
        raise ValueError("cohort_grid requires a cumulative triangle")
    df = triangle.select_fields(loss_field).execute()
    if df.empty:
        raise ValueError(f"no rows for loss field {loss_field!r}")
    segs = triangle.segments
    if segs and len(df.drop_duplicates(segs)) > 1:
        raise ValueError(
            f"triangle has multiple segment combinations on {segs}; filter to one cohort first"
        )
    identity = _cohort_identity(triangle, df, (loss_field,))
    data = cohort_grid_frame(
        df,
        dev_grain_months=GRAIN_MONTHS[triangle.meta.dev_grain],
        units=triangle.meta.units,
        loss_field=loss_field,
        segment=identity["segment"],
        measure=identity["measure"],
    )
    # splat the full identity so this entry point and stan_data cannot drift on
    # what "identity" means (fields/models are (loss_field,) either way today)
    data.update(identity)
    if premium_field is not None:
        data["premium"] = premium_by_origin(
            triangle,
            premium_field,
            data["origin_periods"],
            segment=identity["segment"],
        )
    return data


def cohort_grid_frame(
    df,
    *,
    dev_grain_months: int,
    units: str | None = None,
    loss_field: str | None = None,
    segment: dict | None = None,
    measure: str,
) -> dict[str, Any]:
    """:func:`cohort_grid`'s frame half: one cohort's already-materialized rows
    (``origin_period``, ``dev_lag``, ``value``) to the dense contract dict.

    Split out so a batch caller (``kernels.mack.fit_mack_many``) can materialize
    a multi-cohort triangle ONCE and grid each cohort from the shared frame;
    the per-cohort engine round-trip is what dominates a filter+fit loop. All
    contract guarantees (dev-grain multiples, one row per cell, run-off
    staircase) are enforced here, identically for both entry points.

    ``segment``/``measure`` are the cohort-identity half a bare frame cannot
    derive for itself (no Triangle here): the batch caller supplies them from
    its own group key and metadata, :func:`cohort_grid` from
    :func:`_cohort_identity`. ``measure`` is REQUIRED, no default: it stamps an
    identity fact this function cannot verify, and ``index_into``'s measure
    refusal would be keyed off a defaulted lie the first time an incremental
    frame reached it. Like ``segment``, it is consciously supplied or the call
    is refused. ``fields``/``models`` are ``(loss_field,)`` - the deterministic
    entries model exactly the field they read.
    """
    # Vectorized throughout: this runs once per cohort in fit_mack_many's batch
    # loop, so per-row pandas iteration here would put the loop's cost right
    # back after the engine round-trips were removed.
    step = dev_grain_months
    d = dev_step_index(df["dev_lag"], step=step)
    w_idx, origin_arr = pd.factorize(_as_date(df["origin_period"]), sort=True)
    origins = list(origin_arr)
    n_w, n_d = len(origins), int(d.max())
    flat = w_idx.astype(np.int64) * n_d + (d - 1)
    if np.unique(flat).size != flat.size:
        raise ValueError(
            "multiple rows per (origin, dev) cell; slice with as_of()/latest_diagonal() first"
        )
    cum = np.full((n_w, n_d), np.nan)
    cum[w_idx, d - 1] = df["value"].to_numpy(dtype=float)
    obs_mask = ~np.isnan(cum)

    if not obs_mask[:, 0].all():
        missing = [origins[i] for i in np.nonzero(~obs_mask[:, 0])[0]]
        raise ValueError(f"origins {missing} have no observation at the first dev step")
    latest_dev = obs_mask.shape[1] - 1 - np.argmax(obs_mask[:, ::-1], axis=1)  # (n_w,)
    # K = the calendar diagonal, in (origin + dev) units, implied by the youngest
    # origin; every other origin must sit on the same diagonal (or be capped by
    # n_d, having already run off).
    diagonal = int(latest_dev[-1]) + (n_w - 1)
    expected = np.minimum(diagonal - np.arange(n_w), n_d - 1)  # (n_w,) staircase
    reference = np.arange(n_d)[None, :] <= expected[:, None]
    if not (obs_mask == reference).all():
        rows = sorted(set(np.nonzero(obs_mask != reference)[0].tolist()))[:5]
        bad = [(origins[i], int(obs_mask[i].sum()), int(expected[i]) + 1) for i in rows]
        raise ValueError(
            "observed cells are not a run-off triangle; the chain-ladder kernels need "
            "each origin observed from dev step 1 up to one common calendar diagonal. "
            f"(origin, observed cells, expected depth) mismatches: {bad}"
        )

    # 1-based cell indices of the observed cells, row-major (sorted by (w, d)),
    # the same convention as stan_data's w/d. Derived from the mask rather than
    # the input rows so they cannot disagree with the grid they index.
    w_obs, d_obs = np.nonzero(obs_mask)
    scored = (loss_field,) if loss_field is not None else ()
    return {
        "n_w": n_w,
        "n_d": n_d,
        "cum": cum,
        "obs_mask": obs_mask,
        "latest_dev": latest_dev.astype(int),
        "origin_periods": origins,
        "dev_grain_months": step,
        "units": units,
        "loss_field": loss_field,
        "segment": dict(segment or {}),
        "fields": scored,
        "models": scored,
        "measure": measure,
        "w": (w_obs + 1).astype(int),
        "d": (d_obs + 1).astype(int),
    }


def realized_values(
    triangle: Triangle,
    *,
    loss_field: str,
    dev_lag: int,
    origins: list[dt.date],
) -> np.ndarray:
    """Realized cumulative losses at ``dev_lag`` months for the given origins,
    taken from the FULL (unsliced) triangle - the scoring targets for
    backtests. NaN where the outcome is not (yet) observed."""
    df = triangle.select_fields(loss_field).execute()
    df = df[df["dev_lag"] == dev_lag].copy()
    df["origin_period"] = _as_date(df["origin_period"])
    by_origin = df.set_index("origin_period")["value"]
    return np.array([float(by_origin.get(o, np.nan)) for o in origins])


def _as_date(series):
    if str(series.dtype).startswith("datetime64"):
        return series.dt.date
    return series


def dev_step_index(dev_lag, *, step: int) -> np.ndarray:
    """The 1-based dev step index ``d = dev_lag // step``, or a refusal by name.

    Every contract in this package - the three Stan ones, the dense cohort grid,
    the multi-LOB grid and the neural grids - stores a cell at dev index
    ``dev_lag // step`` and reads dev step 1 as the first development period.
    Two things break that division and both used to be checked separately in six
    places, in six copies of two lines:

    * an age that is not a whole number of dev steps. Floor division does not
      refuse it, it moves it: on a 12-month grain, ages 3, 15, 27 land on steps 0,
      1, 2 rather than 1, 2, 3, so every origin reads one development period
      younger than it is and the first cell falls off the grid entirely. Those
      exact ages are what chainladder's latest-diagonal anchoring produces, and
      what our own ``with_dev_grain`` produces to match it, whenever the latest
      valuation is a March 31 - so this is a shape the triangle layer emits, not
      one only bad input can reach. It is a coherent triangle (``validate``
      accepts it: every age shares one offset) and it is not a grid these
      contracts can index, which is why the refusal is here and not there.
    * a non-positive age. ``dev_lag`` counts months from the origin period start,
      so the first cell of a 12-month grain is at 12; a zero or negative age would
      index step 0 or below.

    The sign is tested first, because ``%`` here follows Python's sign rule: an
    age of -3 leaves a remainder of 9 against a 12-month grain, so testing the
    offset first would report a negative age as an anchoring problem and the
    positivity message could never be reached for it.

    Returns an ``int64`` array aligned with the input, so a caller can assign it
    straight into its frame.
    """
    months = np.asarray(dev_lag, dtype=np.int64)
    non_positive = months <= 0
    if non_positive.any():
        ages = sorted({int(a) for a in np.unique(months[non_positive])})
        raise ValueError(
            f"dev_lag must be positive, got {ages[:5]}; dev_lag is months from the origin "
            f"period start, so the first cell of a {step}-month dev grain is at {step}"
        )
    offsets = months % step
    off_grain = offsets != 0
    if off_grain.any():
        ages = sorted({int(a) for a in np.unique(months[off_grain])})
        found = sorted({int(o) for o in np.unique(offsets)})
        raise ValueError(
            f"{int(off_grain.sum())} dev_lag value(s) are not on a {step}-month grain "
            f"boundary: ages {ages[:5]} leave offsets {found} against the declared dev "
            f"grain. Every contract indexes a cell by d = dev_lag // {step}, so an age off "
            "the boundary is floored onto the step below it and the whole triangle reads "
            "one development period younger. This is what chainladder's latest-diagonal "
            "anchoring produces: a March 31 valuation regrained to an annual dev grain "
            "gives ages 3, 15, 27 rather than 12, 24, 36. Slice with as_of() to a "
            "valuation on a grain boundary before with_dev_grain(), or keep the finer dev "
            "grain."
        )
    return months // step


def _months_between(a: dt.date, b: dt.date) -> int:
    return (b.year - a.year) * 12 + b.month - a.month


def require_origin_axis_step(origins: list[dt.date], *, step: int) -> None:
    """Refuse an origin axis whose consecutive periods are not one dev step apart.

    ``stan_data`` and ``nn_data`` both index a cell by a pair of integers, origin
    index ``w`` and dev step ``d``, and both then read those two as one shared
    calendar clock. In ``nn_data`` it is explicit: ``cal_idx = w + d + 1`` is the
    diagonal number that the validation split, the cutoff augmentation and the
    held-out cutoff all slice on, standing in for the evaluation date. In
    ``stan_data`` it is the step from one origin to the next, which ``prev_idx``
    links as one elapsed development period.

    Both readings hold only while one origin step equals one dev step. Two
    geometries break it, and neither one is malformed data:

    * an origin axis with a hole in it (accident years 2010, 2012, 2013), where the
      index advances one step over two calendar years;
    * annual origins on a quarterly dev grain, where each row of the grid sits four
      diagonals below the row above it but one ``cal_idx`` apart.

    Measured on the first: three cells whose real evaluation date is 2013-12-31 are
    given cal_idx 4, 3 and 3, so the validation split holds one of them out and
    trains on the other two - it trains on the diagonal it is scored on. Nothing
    raises, and nothing about the result looks wrong.

    What is asked for is one origin step per dev step, not an annual grain: a
    quarterly origin axis on a quarterly dev grain is accepted, and so is a
    monthly one on a monthly grain.

    The axis checked here is the POOLED one, the union of origins over every
    cohort, so one cohort that skips an accident year its neighbours carry is
    unaffected: it keeps its row on the shared axis and is simply masked out.

    One consumer is caught by sharing a door rather than by its own reading of
    ``w``. ``stan_data`` is used by three gallery entries: ``meyers_ccl`` and
    ``meyers_csr`` read ``w`` as a clock, and ``guszcza_growth_curve`` does not,
    using it only to index ``ulr[w]`` and ``premium[w-1]``, exactly as
    ``odp_stan_data`` and ``compartmental_stan_data`` use theirs (which is why
    those two are NOT checked). So Guszcza will refuse a gapped origin axis it
    could in principle fit. That is a deliberate cost of putting the check on the
    shared contract rather than in two model files, and it is written down here so
    it can be revisited rather than discovered.
    """
    pairs = list(zip(origins[:-1], origins[1:], strict=True))
    bad = [(a, b, _months_between(a, b)) for a, b in pairs if _months_between(a, b) != step]
    if not bad:
        return
    a, b, gap = bad[0]
    raise ValueError(
        f"the origin axis is not spaced one dev step apart: {a} to {b} is {gap} months "
        f"against a {step}-month dev grain ({len(bad)} of {len(pairs)} origin steps). The "
        "origin index w and the dev index d are read as one shared calendar clock: the "
        "neural contract's cal_idx = w + d + 1 is the evaluation date that the validation "
        "split, the cutoff augmentation and the held-out cutoff all slice on, and the "
        "cross-classified models step from one origin to the next as one elapsed "
        "development period. Neither is calendar time when one origin step is not one dev "
        "step. Restrict the triangle to a contiguous run of origins, or bring the dev "
        "grain to the origin step with with_dev_grain()."
    )
