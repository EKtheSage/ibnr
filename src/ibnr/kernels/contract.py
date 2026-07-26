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
    if (df["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")

    origins = sorted(df["origin_period"].unique())
    dev_steps = sorted((df["dev_lag"] // step).unique())
    if dev_steps[0] < 1:
        raise ValueError("dev_lag must be positive")
    n_w, n_d = len(origins), int(dev_steps[-1])
    w_of = {o: i + 1 for i, o in enumerate(origins)}

    df["w"] = df["origin_period"].map(w_of)
    df["d"] = (df["dev_lag"] // step).astype(int)
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
        premium = _premium_by_origin(
            triangle, premium_field, origins, segment=_cohort_identity(triangle, df, ())["segment"]
        )
        data["premium"] = premium
        data["logprem"] = np.log(premium)[data["w"] - 1]
    return data


def _premium_by_origin(
    triangle: Triangle,
    premium_field: str,
    origins: list,
    *,
    segment: dict | None = None,
) -> np.ndarray:
    """One exposure per origin, from THIS cohort.

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
    if (df["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")

    origins = sorted(df["origin_period"].unique())
    dev_steps = sorted((df["dev_lag"] // step).unique())
    if dev_steps[0] < 1:
        raise ValueError("dev_lag must be positive")
    n_w, n_d = len(origins), int(dev_steps[-1])
    w_of = {o: i + 1 for i, o in enumerate(origins)}

    df["w"] = df["origin_period"].map(w_of)
    df["d"] = (df["dev_lag"] // step).astype(int)
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
        premium = _premium_by_origin(
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
    if (df["dev_lag"] % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")

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
    dev_steps = sorted((wide["dev_lag"] // step).unique())
    if dev_steps[0] < 1:
        raise ValueError("dev_lag must be positive")
    n_w, n_d = len(origins), int(dev_steps[-1])
    w_of = {o: i + 1 for i, o in enumerate(origins)}
    wide["w"] = wide["origin_period"].map(w_of)
    wide["d"] = (wide["dev_lag"] // step).astype(int)
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
    data["premium"] = _premium_by_origin(
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
    data = cohort_grid_frame(
        df,
        dev_grain_months=GRAIN_MONTHS[triangle.meta.dev_grain],
        units=triangle.meta.units,
        loss_field=loss_field,
    )
    if premium_field is not None:
        data["premium"] = _premium_by_origin(
            triangle,
            premium_field,
            data["origin_periods"],
            segment=_cohort_identity(triangle, df, ())["segment"],
        )
    return data


def cohort_grid_frame(
    df,
    *,
    dev_grain_months: int,
    units: str | None = None,
    loss_field: str | None = None,
) -> dict[str, Any]:
    """:func:`cohort_grid`'s frame half: one cohort's already-materialized rows
    (``origin_period``, ``dev_lag``, ``value``) to the dense contract dict.

    Split out so a batch caller (``kernels.mack.fit_mack_many``) can materialize
    a multi-cohort triangle ONCE and grid each cohort from the shared frame;
    the per-cohort engine round-trip is what dominates a filter+fit loop. All
    contract guarantees (dev-grain multiples, one row per cell, run-off
    staircase) are enforced here, identically for both entry points.
    """
    # Vectorized throughout: this runs once per cohort in fit_mack_many's batch
    # loop, so per-row pandas iteration here would put the loop's cost right
    # back after the engine round-trips were removed.
    step = dev_grain_months
    dev = df["dev_lag"].to_numpy(dtype=np.int64)
    if (dev % step != 0).any():
        raise ValueError(f"dev_lag values are not multiples of the {step}-month dev grain")
    d = dev // step
    if (d < 1).any():
        raise ValueError("dev_lag must be positive")
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
