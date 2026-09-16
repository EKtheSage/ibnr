"""Compartmental log density and draws at arbitrary cells, in plain numpy.

The counterpart of the two Stan programs' ``generated quantities`` blocks,
except that it evaluates at cells the fit never saw. Free functions over plain
arrays, per CLAUDE.md decision 6: backend-blind, unit-testable without a
sampler, living beside the Stan sources they mirror. The curves themselves are
imported from ``model.py`` (``os_curve`` / ``paid_curve``), the same closed
forms the Stan ``functions`` blocks implement - one algebra, three readers.

One likelihood switch per variant (``model.stan:144-146`` /
``model_lognormal.stan:167-169``)::

    gaussian:   loss[i] ~ normal(mu[i], delta[i] == 0 ? sigma_os : sigma_paid)
    lognormal:  y[i]    ~ lognormal(log(mu[i]), delta[i] == 0 ? sigma_os : sigma_paid)

where the gaussian ``mu`` is an AMOUNT (premium times the curve; outstanding a
level, paid cumulative) and the lognormal ``mu`` is a loss RATIO (outstanding a
level, paid the increment of ``paid_curve`` over ``(t - devfreq, t]``).

**Delta dispatch is explicit.** Training cells arrive as a
:class:`~ibnr.kernels.holdout.DeltaCellIndex` carrying both stacked blocks;
held-out cells arrive as a plain :class:`~ibnr.kernels.holdout.CellIndex` from
``index_into``, which can only ever resolve the PAID field, because
'outstanding' is derived (reported - paid), not a raw field the triangle
stores. :func:`cell_deltas` encodes that rule in one place.

**The lognormal per-cell parameters are reconstructed from sampled sites**
(``u = sd * z``, ``model_lognormal.stan:92-97`` verbatim) rather than read
from Stan's ``u_*`` transformed parameters, which the ports do not expose -
reading them would silently make this module Stan-only. The one exception is
``u_ay``, which IS read directly: all three backends expose it as a
deterministic, so it is safe, and its Cholesky construction is not worth
re-implementing here.

**``t`` comes from the CONTRACT's grain** (``d * dev_grain_months / 12``),
never from the cells: ker/kp are per-year rates, and on the annual Schedule P
grain ``t == d`` numerically, so a units bug here passes every mart-shaped
test and only a non-annual triangle can see it.

The two variants' densities are on DIFFERENT measures (amounts vs loss
ratios), so their values are not comparable with each other here - the entry
declares ``heldout_measure`` per variant and ``ScoresHeldout.log_lik_at``
applies the one carry to Lebesgue-on-amount.
"""

from __future__ import annotations

import numpy as np

from ibnr.gallery.bayesian.compartmental.model import os_curve, paid_curve
from ibnr.kernels.densities import lognormal_lpdf, normal_lpdf
from ibnr.kernels.holdout import CellIndex, DeltaCellIndex

__all__ = ["cell_deltas", "draw_cells", "log_lik_cells", "mu_cells", "sigma_cells"]

#: posterior variables each variant's scorer needs, in every backend.
#: The gaussian set exists under identical names in all three backends (Stan
#: transformed parameters / generated quantities; deterministics in both
#: ports). The lognormal set is the SAMPLED sd_*/z_* sites plus the ``u_ay``
#: deterministic, from which the per-cell parameters are reconstructed -
#: Stan's ``u_*_dev`` / ``u_*_ay`` transformed parameters are deliberately
#: not on the list because the ports do not expose them.
REQUIRED_DRAWS: dict[str, tuple[str, ...]] = {
    "gaussian": ("RLR", "RRF", "ker", "kp", "sigma_os", "sigma_paid"),
    "lognormal": (
        "b_oRLR",
        "b_oRRF",
        "b_oker",
        "b_okp",
        "u_ay",
        "sd_dev",
        "z_RLR_dev",
        "z_RRF_dev",
        "sd_ker",
        "z_ker_ay",
        "z_ker_dev",
        "sd_kp",
        "z_kp_ay",
        "z_kp_dev",
        "sigma_os",
        "sigma_paid",
    ),
}

VARIANTS: tuple[str, ...] = tuple(REQUIRED_DRAWS)


def cell_deltas(cells: CellIndex) -> np.ndarray:
    """Which observation block each cell belongs to: 0 = outstanding, 1 = paid.

    A :class:`~ibnr.kernels.holdout.DeltaCellIndex` says so itself. A plain
    ``CellIndex`` can only have come through ``index_into``, whose one
    scorable raw field on the compartmental contract is the paid one -
    'outstanding' is derived, not a field - so a plain index IS the paid
    block. This function is the single place that rule is encoded; the
    density, the draws and the sigma switch all read it from here.
    """
    if isinstance(cells, DeltaCellIndex):
        return np.asarray(cells.delta, dtype=int)
    return np.ones(cells.n_cells, dtype=int)


def mu_cells(
    contract: dict, post: dict[str, np.ndarray], cells: CellIndex, *, variant: str
) -> np.ndarray:
    """``(n_draws, n_cells)`` likelihood location, per variant.

    gaussian (``model.stan:102-110``): AMOUNTS -
    ``premium[w] * (delta == 0 ? os_curve(t) : paid_curve(t))`` with
    per-origin RLR/RRF and ker/kp shared across accident years.

    lognormal (``model_lognormal.stan:104-122``): loss RATIOS - per-CELL
    compartmental parameters rebuilt from base + sd * z, outstanding read off
    the curve, paid differenced over ``(t - devfreq, t]`` with the SAME cell
    parameters at both ends. ``max(t - devfreq, 0)`` is exactly Stan's
    ``t > devfreq`` branch, because ``paid_curve(0) == 0`` - the same clamp
    both ports use.
    """
    _require(post, variant)
    delta = cell_deltas(cells)
    t, devfreq = _ages(contract, cells)
    w0 = np.asarray(cells.w, dtype=int) - 1
    d0 = np.asarray(cells.d, dtype=int) - 1

    if variant == "gaussian":
        ker = np.asarray(post["ker"], dtype=float).reshape(-1, 1)
        kp = np.asarray(post["kp"], dtype=float).reshape(-1, 1)
        rlr = np.asarray(post["RLR"], dtype=float)[:, w0]
        rrf = np.asarray(post["RRF"], dtype=float)[:, w0]
        premium = np.asarray(contract["premium"], dtype=float)[w0]
        lr = np.where(
            delta[None, :] == 0,
            os_curve(t[None, :], ker, kp, rlr),
            paid_curve(t[None, :], ker, kp, rlr, rrf),
        )
        return premium[None, :] * lr

    ker, kp, rlr, rrf = _cell_parameters(post, w0, d0)
    prev_age = np.maximum(t - devfreq, 0.0)
    paid_incr = paid_curve(t[None, :], ker, kp, rlr, rrf) - paid_curve(
        prev_age[None, :], ker, kp, rlr, rrf
    )
    return np.where(delta[None, :] == 0, os_curve(t[None, :], ker, kp, rlr), paid_incr)


def log_lik_cells(
    contract: dict, post: dict[str, np.ndarray], cells: CellIndex, *, variant: str
) -> np.ndarray:
    """``(n_draws, n_cells)`` log density on the variant's own measure.

    gaussian: ``normal_lpdf(loss | mu, sigma)`` on AMOUNTS, per
    ``model.stan:156-159``. No positivity anywhere - the Gaussian takes zero
    and negative outstanding natively, which is why this arm survives a
    mechanical retrospective, and the same must hold here.

    lognormal: ``lognormal_lpdf(y | log(mu), sigma)`` on loss RATIOS, per
    ``model_lognormal.stan:179-183``. The observed ratio is formed exactly as
    ``_lognormal_stan_data`` forms Stan's ``y``: the paid increment is
    ``value - prev_value`` (the predecessor is training data) and the divisor
    is the CONTRACT's per-origin premium, the same number Stan divided by. A
    non-positive outstanding level or paid increment is refused - the family
    limit the entry applies at fit time (``dropped_cells_``), and at held-out
    cells it means the cohort has no density (``scoring_refused``), not that
    the number should be clamped.
    """
    mu = mu_cells(contract, post, cells, variant=variant)
    sigma = sigma_cells(post, cells)
    if variant == "gaussian":
        return normal_lpdf(np.asarray(cells.value, dtype=float)[None, :], mu, sigma)
    ratio = _observed_ratios(contract, cells)
    return lognormal_lpdf(ratio[None, :], np.log(mu), sigma)


def draw_cells(
    contract: dict,
    post: dict[str, np.ndarray],
    cells: CellIndex,
    *,
    rng: np.random.Generator,
    variant: str,
) -> np.ndarray:
    """``(n_draws, n_cells)`` predictive draws, one per posterior draw.

    gaussian: ``rng.normal(mu, sigma)`` - CUMULATIVE paid amounts at paid
    cells (OS-level amounts at outstanding ones), hence the entry's
    ``heldout_draw_scale = "cumulative"`` for this variant. Deliberately NOT
    ``predict()``'s simulation: that one anchors fully developed origins at
    the observed value with zero variance, and a cell-level forecast must
    never emit a point mass (``CohortForecast`` rejects zero-variance draws).
    Every cell here is ``mu + sigma * eps``, per posterior draw.

    lognormal: ``premium * rng.lognormal(log(mu), sigma)`` - INCREMENTAL paid
    amounts (OS-level amounts at outstanding cells), hence
    ``heldout_draw_scale = "incremental"``. The ratio draw is scaled to the
    amount here because draws, unlike densities, carry no measure
    declaration: ``predict_at`` adds the cumulative anchor (``prev_value``,
    an amount) and the panel scores against the triangle's amounts, so the
    draws must arrive in those units.

    Neither branch reads ``cells.value``: this is a forecast, not an
    evaluation, so it does not inherit ``log_lik_cells``'s lognormal
    refusal - a cohort with a negative held-out paid increment can still be
    CRPS-scored where it cannot be ELPD-scored, which is exactly why the two
    capabilities are separate mixins.
    """
    mu = mu_cells(contract, post, cells, variant=variant)
    sigma = sigma_cells(post, cells)
    if variant == "gaussian":
        return rng.normal(mu, sigma)
    premium = np.asarray(contract["premium"], dtype=float)[np.asarray(cells.w, dtype=int) - 1]
    return premium[None, :] * rng.lognormal(np.log(mu), sigma)


def sigma_cells(post: dict[str, np.ndarray], cells: CellIndex) -> np.ndarray:
    """``(n_draws, n_cells)`` residual scale: ``sigma_os`` or ``sigma_paid``
    by delta.

    One expression, shared by the density and the draws, so the two cannot
    disagree about which block a cell observes. The names are identical in
    every backend and both variants; only the scale's MEANING differs (an
    amount for gaussian, a log-ratio CV for lognormal), which
    :func:`mu_cells`'s variant switch carries alongside.
    """
    delta = cell_deltas(cells)
    sigma_os = np.asarray(post["sigma_os"], dtype=float).reshape(-1, 1)
    sigma_paid = np.asarray(post["sigma_paid"], dtype=float).reshape(-1, 1)
    return np.where(delta[None, :] == 0, sigma_os, sigma_paid)


def _require(post: dict[str, np.ndarray], variant: str) -> None:
    if variant not in REQUIRED_DRAWS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    missing = [name for name in REQUIRED_DRAWS[variant] if name not in post]
    if missing:
        raise KeyError(f"posterior is missing {missing}; have {sorted(post)}")


def _ages(contract: dict, cells: CellIndex) -> tuple[np.ndarray, float]:
    """``(t, devfreq)`` in YEARS, from the contract's grain.

    Identical arithmetic to ``contract.py``'s ``t = d * step / 12`` and the
    entry's ``devfreq = step / 12``. Derived from the contract and never read
    off the cells: on the annual grain ``t == d`` numerically, so a scorer
    that used the dev index as an age would pass every mart-shaped test and
    evaluate a quarterly triangle at 4x the true age.
    """
    step_years = float(contract["dev_grain_months"]) / 12.0
    return np.asarray(cells.d, dtype=float) * step_years, step_years


def _cell_parameters(
    post: dict[str, np.ndarray], w0: np.ndarray, d0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell ``(ker, kp, RLR, RRF)`` for the lognormal variant, each
    ``(n_draws, n_cells)`` - ``model_lognormal.stan:92-97`` and ``104-107``.

    The varying effects are reconstructed from the sampled sites by the Stan
    file's own identities (``u_RLR_dev = sd_dev[1] * z_RLR_dev`` and so on;
    1-based there, 0-based here), because the ``u_*`` transformed parameters
    exist only in the Stan fit. ``u_ay`` is the exception: all three backends
    expose it as a deterministic, and its Cholesky construction is not worth
    re-implementing here.
    """
    b_orlr = np.asarray(post["b_oRLR"], dtype=float).reshape(-1, 1)
    b_orrf = np.asarray(post["b_oRRF"], dtype=float).reshape(-1, 1)
    b_oker = np.asarray(post["b_oker"], dtype=float).reshape(-1, 1)
    b_okp = np.asarray(post["b_okp"], dtype=float).reshape(-1, 1)
    u_ay = np.asarray(post["u_ay"], dtype=float)  # (n_draws, 2, n_w)
    sd_dev = np.asarray(post["sd_dev"], dtype=float)
    sd_ker = np.asarray(post["sd_ker"], dtype=float)
    sd_kp = np.asarray(post["sd_kp"], dtype=float)

    u_rlr_dev = sd_dev[:, [0]] * np.asarray(post["z_RLR_dev"], dtype=float)
    u_rrf_dev = sd_dev[:, [1]] * np.asarray(post["z_RRF_dev"], dtype=float)
    u_ker_ay = sd_ker[:, [0]] * np.asarray(post["z_ker_ay"], dtype=float)
    u_ker_dev = sd_ker[:, [1]] * np.asarray(post["z_ker_dev"], dtype=float)
    u_kp_ay = sd_kp[:, [0]] * np.asarray(post["z_kp_ay"], dtype=float)
    u_kp_dev = sd_kp[:, [1]] * np.asarray(post["z_kp_dev"], dtype=float)

    ker = 3.0 * np.exp(0.1 * (b_oker + u_ker_ay[:, w0] + u_ker_dev[:, d0]))
    kp = 1.0 * np.exp(0.1 * (b_okp + u_kp_ay[:, w0] + u_kp_dev[:, d0]))
    rlr = 0.7 * np.exp(0.2 * (b_orlr + u_ay[:, 0, w0] + u_rlr_dev[:, d0]))
    rrf = 0.8 * np.exp(0.1 * (b_orrf + u_ay[:, 1, w0] + u_rrf_dev[:, d0]))
    return ker, kp, rlr, rrf


def _observed_ratios(contract: dict, cells: CellIndex) -> np.ndarray:
    """The lognormal variant's observed ``y`` per cell: OS level (delta = 0)
    or paid increment (delta = 1), over the contract's per-origin premium -
    exactly ``_lognormal_stan_data``'s assembly, cell by cell.

    The increment differences against ``prev_value``, which sits on the
    training diagonal (data, not a prediction - the same fact that makes the
    increment/cumulative Jacobian 1). The premium is the contract's, not the
    cells', because Stan's ``y`` was divided by that exact number and the
    agreement gate compares elementwise.

    **The two premium sources must agree, and that is checked here.** The
    ratio divides by the CONTRACT's premium while the base class's measure
    carry (``to_amount_scale``) divides by the CELLS' premium - the holdout
    frame's, when ``next_diagonal`` attached one. Both are the training
    slice's booked value, so today they cannot differ; but if they ever did,
    the carried density would silently stop integrating to 1 - a wrong
    Jacobian, the bug class nothing downstream can see. Cells carrying no
    premium (NaN) are exempt from the comparison: the carry has its own
    refusal for those.
    """
    delta = cell_deltas(cells)
    value = np.asarray(cells.value, dtype=float)
    prev = np.asarray(cells.prev_value, dtype=float)
    paid = delta == 1
    missing = paid & np.isnan(prev)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} paid cell(s) have no predecessor, so no increment can "
            "be formed for the lognormal likelihood; next_diagonal excludes such cells "
            "upstream (no_predecessor)"
        )
    amount = np.where(paid, value - prev, value)
    premium = np.asarray(contract["premium"], dtype=float)[np.asarray(cells.w, dtype=int) - 1]
    cell_premium = np.asarray(cells.premium, dtype=float)
    mismatched = ~np.isnan(cell_premium) & ~np.isclose(cell_premium, premium)
    if mismatched.any():
        raise ValueError(
            f"{int(mismatched.sum())} cell(s) carry a premium that disagrees with the "
            "fitted contract's per-origin premium (the cells' comes from the holdout "
            "frame, attached by next_diagonal from the training slice; the contract's "
            "is premium_by_origin's booked value). The observed ratio divides by the "
            "contract's number while the measure carry divides by the cells', so a "
            "mismatch would produce a density that silently no longer integrates to 1"
        )
    ratio = amount / premium
    bad = ratio <= 0
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} cell(s) have a non-positive outstanding level or paid "
            "increment; the lognormal variant has no density there - the same family "
            "limit _lognormal_stan_data applies at fit time and counts in dropped_cells_"
        )
    return ratio
