"""Mack's distribution-free chain ladder, implemented natively over a Triangle.

Mack (1993), *Distribution-free calculation of the standard error of chain
ladder reserve estimates* (ASTIN Bulletin 23/2). The model makes no
distributional assumption; it fixes the first two conditional moments of the
development, which is exactly what the one-year claims development result of
``kernels/cdr.py`` needs on top:

    E[C_{i,j+1} | C_{i,0}, ..., C_{i,j}]   = f_j * C_{i,j}
    Var(C_{i,j+1} | C_{i,0}, ..., C_{i,j}) = sigma_j^2 * C_{i,j}

with accident years independent. Estimated on the observed run-off triangle by

    f_j-hat      = sum_i C_{i,j+1} / S_j,        S_j = sum_i C_{i,j}
    sigma_j-hat^2 = 1/(n_j - 1) * sum_i C_{i,j} * (C_{i,j+1}/C_{i,j} - f_j-hat)^2

The volume-weighted (alpha = 1) factor is Mack's estimator and the one
Merz-Wuthrich assume; no other averaging is offered here, because the CDR
formulas downstream are only valid for this one.

THE TWO SUMS DO NOT ALWAYS RUN OVER THE SAME ORIGINS. Both range over the
origins observing both ends of the step, but sigma's summand carries a 1/C_{i,j}
(the weighted residual is (C_{i,j+1} - f_j C_{i,j})^2 / C_{i,j}), so it is
estimable only where C_{i,j} > 0:

    f_j-hat        all pair origins; needs only S_j > 0
    sigma_j-hat^2  the pair origins with C_{i,j} > 0, count ``n_j``, df n_j - 1

They coincide on any triangle with strictly positive cumulatives, which is most
of them. They part on a real and unexceptional cohort - an accident year with
zero paid at 12 months - whose chain-ladder ultimate is perfectly well defined
and whose sigma simply has one fewer observation behind it. Refusing the whole
cohort would throw away a usable reserve estimate; quietly summing 0 * inf into
the variance would be worse. ``MackFit.n_obs`` and ``MackFit.n_pos`` record the
two counts separately so the divergence is visible rather than inferred.

The one thing this file cannot check on the factor path is the LATEST DIAGONAL:
those cells have no observed successor and so enter no step's estimator, yet
every variance formula divides by them. That guard therefore lives on the
variance path - see ``MackFit.require_positive_open_diagonals``.

Why this exists rather than a call into chainladder-python: chainladder is an
optional interop extra (``ibnr[interop]``), and the CDR is a *core* deliverable
that has to run in the duckdb-only install and inside the compute image. The
tie-out tests (``tests/test_mack.py``, marked ``tieout``) pin every quantity
here against ``cl.MackChainladder`` on raa, so "native" never means "different".

Cross-refs: ``kernels/contract.py::cohort_grid`` (the data contract),
``kernels/cdr.py`` (one-year CDR built on ``MackFit``),
``gallery/deterministic/mack`` (the gallery entry that wraps this in a
``PredictiveDistribution``).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ibnr.kernels.contract import cohort_grid
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

#: how the variance of the LAST development step is estimated. That step has a
#: single observation, so it has no residual degrees of freedom of its own.
SIGMA_RULES = ("mack", "log_linear")

#: shapes for a simulated development step. Mack's model fixes only the first
#: two conditional moments, so a simulation must add one assumption; all three
#: match those moments and differ in tail shape and support.
PROCESS_LAWS = ("gamma", "normal", "lognormal")


@dataclass(frozen=True)
class MackFit:
    """A fitted distribution-free chain ladder on one cohort.

    Arrays are 0-based on both axes: dev index ``j`` spans ``0 .. n_d - 1`` and
    the development step ``j -> j + 1`` carries ``f[j]``, ``sigma2[j]``,
    ``s[j]`` (its volume denominator), ``n_obs[j]`` (the origins behind the
    FACTOR) and ``n_pos[j]`` (the origins behind the SIGMA), each of length
    ``n_d - 1``. The last two differ only where a pair origin has a zero
    cumulative; see the module docstring.

    ``cum`` keeps the observed triangle (NaN outside it); ``full`` is the same
    matrix with the lower triangle filled by the chain-ladder projection, so
    ``full[:, -1]`` is the ultimate and ``full[i, j]`` for ``j > latest_dev[i]``
    is the ``C-hat_{i,j}`` that Mack's and Merz-Wuthrich's variance formulas
    both evaluate at.
    """

    cum: np.ndarray  # (n_w, n_d) observed cumulative, NaN outside the triangle
    obs_mask: np.ndarray  # (n_w, n_d) bool
    latest_dev: np.ndarray  # (n_w,) 0-based dev index of each origin's diagonal cell
    f: np.ndarray  # (n_d - 1,) volume-weighted development factors
    sigma2: np.ndarray  # (n_d - 1,) Mack process variance parameters
    s: np.ndarray  # (n_d - 1,) S_j = sum of C_{i,j} over the origins used for f[j]
    n_obs: np.ndarray  # (n_d - 1,) origins behind f[j] (the full pair set)
    n_pos: np.ndarray  # (n_d - 1,) origins behind sigma2[j] (those with C_{i,j} > 0)
    origin_periods: list[dt.date]
    dev_grain_months: int
    sigma_rule: str
    units: str | None = None
    loss_field: str | None = None

    # -- point estimates -------------------------------------------------------

    @property
    def n_w(self) -> int:
        return self.cum.shape[0]

    @property
    def n_d(self) -> int:
        return self.cum.shape[1]

    @property
    def latest(self) -> np.ndarray:
        """(n_w,) each origin's cumulative loss on the latest diagonal."""
        return self.cum[np.arange(self.n_w), self.latest_dev]

    @property
    def full(self) -> np.ndarray:
        """(n_w, n_d) observed triangle completed by the chain-ladder projection."""
        out = self.cum.copy()
        for i in range(self.n_w):
            for j in range(int(self.latest_dev[i]), self.n_d - 1):
                out[i, j + 1] = out[i, j] * self.f[j]
        return out

    @property
    def ultimate(self) -> np.ndarray:
        """(n_w,) projected ultimate = the completed triangle's last column."""
        return self.full[:, -1]

    @property
    def reserve(self) -> np.ndarray:
        """(n_w,) IBNR = ultimate - latest. Zero for a fully developed origin."""
        return self.ultimate - self.latest

    # -- preconditions ---------------------------------------------------------

    def require_positive_open_diagonals(self) -> None:
        """Every OPEN origin's latest-diagonal cell must be strictly positive.

        The factor estimator cannot enforce this and never could. Its ``c0``
        cells are exactly the cells with an observed successor, and on a run-off
        staircase an open origin's diagonal cell has none - so the diagonal is
        the one cell class no factor-side guard ever sees. Every variance formula
        downstream then divides by it: ``msep_runoff``'s process term
        (``ratio_j / C-hat_{i,j}`` starting at ``j = latest_dev[i]``) and
        Merz-Wuthrich's ``Phi_i = ratio_k / C_{i,k} + ...``.

        numpy divides silently, so without this the failure is invisible: a zero
        diagonal returns NaN msep for that origin AND a NaN total, a negative one
        returns a finite NEGATIVE msep whose square root is then NaN, and
        ``simulate_ultimates`` returns an exactly degenerate zero column because
        ``var = sigma2 * state`` is non-positive and every draw comes back at its
        mean. Nothing raises; the numbers are just wrong.

        Deliberately NOT called on the point path. The chain-ladder ultimate is
        a product of factors off that cell and needs no positivity at all, and
        the gallery's skill benchmark (``scripts/compare_gallery.py``) wants the
        ultimate even for a cohort whose variance is undefined. ``fit_mack``
        therefore still succeeds; only ``msep_runoff`` / ``simulate_ultimates`` /
        the CDR refuse.

        A CLOSED origin (already at the last dev column) is exempt: it has no
        remaining step, so nothing divides by its diagonal.
        """
        open_ = self.latest_dev < self.n_d - 1
        diag = self.cum[np.arange(self.n_w), self.latest_dev]
        bad = np.nonzero(open_ & ~(diag > 0))[0]
        if bad.size:
            cells = ", ".join(f"{self.origin_periods[i]}={diag[i]:g}" for i in bad)
            raise ValueError(
                f"non-positive cumulative on the latest diagonal of open origin(s) {cells}. "
                "Mack's conditional variance is proportional to that cell, so every msep "
                "rolling forward from it is undefined. The point estimate does not depend "
                "on it and is still available as .ultimate / .reserve"
            )

    # -- run-off (total) uncertainty -------------------------------------------

    def msep_runoff(self) -> dict[str, np.ndarray | float]:
        """Mack's conditional MSEP of the FULL run-off reserve.

        Mack (1993) formula (3), per accident year ``i``:

            msep_i = C-hat_{i,J}^2 * sum_{j=k_i}^{J-1} (sigma_j^2 / f_j^2)
                                     * (1 / C-hat_{i,j} + 1 / S_j)

        where ``k_i = latest_dev[i]`` is the dev index of ``i``'s diagonal cell
        and ``J = n_d - 1``. The ``1/C-hat`` term is process risk, the ``1/S_j``
        term estimation risk; both are returned separately because
        ``cl.MackChainladder`` exposes them separately and the tie-out checks
        each. For the aggregate, Mack's second formula adds the estimation-risk
        covariance between accident years, which share the same estimated
        factors:

            msep_total = sum_i msep_i
                       + 2 * sum_{i<k} C-hat_{i,J} C-hat_{k,J}
                             * sum_{j=k_i}^{J-1} (sigma_j^2 / f_j^2) / S_j

        (the inner sum runs over the OLDER year's dev range, which is the
        intersection of the two ranges). Process risk carries no cross term:
        accident years are independent under Mack's assumptions.

        Returns ``msep`` / ``process`` / ``parameter`` per origin (variances,
        not standard errors) plus the scalars ``msep_total``,
        ``process_total``, ``parameter_total``.
        """
        self.require_positive_open_diagonals()  # 1/C-hat_{i,j} below starts there
        full = self.full
        ratio = np.divide(  # (n_d - 1,) sigma_j^2 / f_j^2, the recurring weight
            self.sigma2, self.f**2, out=np.zeros_like(self.sigma2), where=self.f != 0
        )
        process = np.zeros(self.n_w)
        parameter = np.zeros(self.n_w)
        for i in range(self.n_w):
            for j in range(int(self.latest_dev[i]), self.n_d - 1):
                process[i] += ratio[j] / full[i, j]
                parameter[i] += ratio[j] / self.s[j]
        ult2 = self.ultimate**2
        process *= ult2
        parameter *= ult2
        msep = process + parameter

        # aggregate estimation risk: every pair of accident years shares the
        # factors estimated on the dev steps they both still have to run through
        cross = 0.0
        for i in range(self.n_w):
            tail_ult = self.ultimate[i + 1 :].sum()
            if tail_ult == 0.0:
                continue
            shared = sum(ratio[j] / self.s[j] for j in range(int(self.latest_dev[i]), self.n_d - 1))
            cross += 2.0 * self.ultimate[i] * tail_ult * shared
        return {
            "msep": msep,
            "process": process,
            "parameter": parameter,
            "msep_total": float(msep.sum() + cross),
            "process_total": float(process.sum()),
            "parameter_total": float(parameter.sum() + cross),
        }

    def summary(self) -> pd.DataFrame:
        """One row per origin: latest, ultimate, IBNR and the run-off standard
        error, plus a ``total`` row. Mirrors ``cl.MackChainladder.summary_``."""
        risk = self.msep_runoff()
        out = pd.DataFrame(
            {
                "origin": self.origin_periods,
                "latest": self.latest,
                "ultimate": self.ultimate,
                "ibnr": self.reserve,
                "runoff_se": np.sqrt(risk["msep"]),
            }
        )
        total = {
            "origin": "total",
            "latest": self.latest.sum(),
            "ultimate": self.ultimate.sum(),
            "ibnr": self.reserve.sum(),
            "runoff_se": np.sqrt(risk["msep_total"]),
        }
        return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def fit_mack(
    triangle: Triangle,
    *,
    loss_field: str = "paid_loss",
    as_of: dt.date | str | None = None,
    sigma_rule: str = "mack",
) -> MackFit:
    """Fit the distribution-free chain ladder on a single-cohort Triangle.

    ``as_of`` slices the backtest diagonal first (the training window); the
    triangle must hold exactly one segment combination. ``sigma_rule`` selects
    how the last development step's variance is estimated - see
    ``_estimate_factors``.
    """
    train = triangle.as_of(as_of) if as_of is not None else triangle
    return fit_mack_grid(cohort_grid(train, loss_field=loss_field), sigma_rule=sigma_rule)


def fit_mack_grid(grid: dict[str, Any], *, sigma_rule: str = "mack") -> MackFit:
    """Fit from an already-built ``cohort_grid`` dict (the array entry point)."""
    if sigma_rule not in SIGMA_RULES:
        raise ValueError(f"sigma_rule must be one of {SIGMA_RULES}, got {sigma_rule!r}")
    cum, mask = grid["cum"], grid["obs_mask"]
    n_d = grid["n_d"]
    if n_d < 2:
        raise ValueError("a chain ladder needs at least two development steps")
    f, sigma2, s, n_obs, n_pos = _estimate_factors(cum, mask, sigma_rule=sigma_rule)
    return MackFit(
        cum=cum,
        obs_mask=mask,
        latest_dev=grid["latest_dev"],
        f=f,
        sigma2=sigma2,
        s=s,
        n_obs=n_obs,
        n_pos=n_pos,
        origin_periods=grid["origin_periods"],
        dev_grain_months=grid["dev_grain_months"],
        sigma_rule=sigma_rule,
        units=grid.get("units"),
        loss_field=grid.get("loss_field"),
    )


def _estimate_factors(
    cum: np.ndarray, mask: np.ndarray, *, sigma_rule: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Volume-weighted factors and Mack's variance parameters, step by step.

    WHICH CELLS MUST BE POSITIVE, and why it is not "all of them". The factor
    ``f_j = sum(C_{i,j+1}) / S_j`` divides once, by the column total, so it needs
    only ``S_j > 0``; an origin sitting at zero contributes nothing to the
    denominator and its successor to the numerator, which is exactly the
    column-total estimator every textbook writes down. Mack's sigma divides per
    origin, so it is estimable only on the pair origins with ``C_{i,j} > 0``.
    Hence three separate, separately-named errors rather than one blanket check:

    - ``C_{i,j} < 0``  -> hard error. A negative weight makes sigma_j^2 itself
      negative, which makes the msep negative and its square root NaN, all
      silently. There is no reading of Mack's model under which this is data.
    - ``S_j <= 0``     -> hard error. 0/0 factor.
    - fewer than 2 positive origins at a step that HAS several pairs -> hard
      error. This is not the last step's missing-degrees-of-freedom case and
      must not be extrapolated into: ``_tail_sigma2`` would happily return 0.0
      at ``j = 0`` and declare the most volatile development step noiseless.

    The last step (and any step with a single observation) genuinely has no
    residual degrees of freedom. Two conventions are offered, because the two
    reference implementations disagree and the difference is visible in the CDR:

    ``mack``        Mack's own rule from section 3 of the paper,
                    sigma_J^2 = min(sigma_{J-1}^4 / sigma_{J-2}^2,
                                    min(sigma_{J-2}^2, sigma_{J-1}^2)) -
                    what ``cl.Development(sigma_interpolation='mack')`` does.
    ``log_linear``  regress log(sigma_j) on j over the estimable steps and
                    extrapolate - the default of chainladder-python and of R's
                    ``MackChainLadder(est.sigma = "log-linear")``, and therefore
                    the convention behind most published raa numbers. Falls back
                    to Mack's rule when the regression is not identified (fewer
                    than two positive sigmas) or would give a non-positive value.

    Returns (f, sigma2, s, n_obs, n_pos), each of length ``n_d - 1``.
    """
    n_d = cum.shape[1]
    f = np.zeros(n_d - 1)
    sigma2 = np.full(n_d - 1, np.nan)
    s = np.zeros(n_d - 1)
    n_obs = np.zeros(n_d - 1, dtype=int)
    n_pos = np.zeros(n_d - 1, dtype=int)
    for j in range(n_d - 1):
        pair = mask[:, j] & mask[:, j + 1]
        c0, c1 = cum[pair, j], cum[pair, j + 1]
        if c0.size == 0:
            raise ValueError(
                f"no origin observes both dev steps {j + 1} and {j + 2}; "
                "the triangle cannot support a chain-ladder factor there"
            )
        if (c0 < 0).any():
            bad = [f"{o}" for o in np.nonzero(pair)[0][c0 < 0]]
            raise ValueError(
                f"negative cumulative loss at dev step {j + 1} (origin index {', '.join(bad)}); "
                "Mack's variance is proportional to C_{i,j}, so a negative cell drives "
                "sigma_j^2 itself negative and every msep built on it with it"
            )
        s[j] = c0.sum()
        if s[j] <= 0:
            raise ValueError(
                f"zero volume at dev step {j + 1}: every origin observing both ends of "
                "the step sits at zero, so the volume-weighted factor is 0/0"
            )
        n_obs[j] = c0.size
        f[j] = c1.sum() / s[j]
        # sigma's weighted residual is (c1 - f*c0)^2 / c0, so only the strictly
        # positive origins can carry it - a subset of the factor's. df is that
        # subset's count minus one, NOT n_obs - 1: dividing a shorter sum by a
        # longer df would shrink sigma for exactly the cohorts this admits.
        pos = c0 > 0
        n_pos[j] = int(pos.sum())
        if n_obs[j] > 1 and n_pos[j] < 2:
            raise ValueError(
                f"dev step {j + 1} has {n_obs[j]} origins but only {n_pos[j]} with a "
                "positive cumulative, so sigma_j^2 has nothing to be estimated from. "
                "This is not the last step's missing-degrees-of-freedom case and is not "
                "extrapolated into - at an early step that would silently declare the "
                "most volatile part of the development noiseless"
            )
        if n_pos[j] > 1:
            p0, p1 = c0[pos], c1[pos]
            sigma2[j] = float((p0 * (p1 / p0 - f[j]) ** 2).sum() / (n_pos[j] - 1))
    missing = np.nonzero(np.isnan(sigma2))[0]
    for j in missing:
        sigma2[j] = _tail_sigma2(sigma2, j, rule=sigma_rule)
    return f, sigma2, s, n_obs, n_pos


def simulate_ultimates(
    fit: MackFit,
    *,
    n_draws: int = 10_000,
    seed: int | None = None,
    process: str = "gamma",
    parameter_risk: bool = True,
) -> PredictiveDistribution:
    """Simulate FULL run-off ultimates from a fitted Mack model.

    Mack's model is distribution-free, so a predictive *distribution* needs one
    assumption beyond it: the shape of the step-to-step shock. ``process``
    picks it from ``PROCESS_LAWS``; every choice matches Mack's two conditional
    moments and they differ only in tail shape and support. This is the
    bootstrap wrapper CLAUDE.md decision 4 requires of a deterministic baseline
    before it may enter the gallery, and it is the run-off counterpart of the
    one-year re-reserving in ``kernels/cdr.py``.

    Parameter risk is drawn ONCE PER DRAW and shared across accident years -
    that shared factor draw is what makes the accident years correlated, and
    dropping it (``parameter_risk=False``) leaves pure, independent process
    risk. The ``total`` column is the row-sum of the same draws, so the
    diversification is in the samples rather than assumed.
    """
    # a non-positive diagonal would give var = sigma2 * state <= 0, which
    # draw_step returns at its mean - a silent point mass, not an error
    fit.require_positive_open_diagonals()
    rng = np.random.default_rng(seed)
    n_w, n_d = fit.n_w, fit.n_d
    f_true = np.tile(fit.f, (n_draws, 1))
    if parameter_risk:
        se = np.sqrt(np.where(fit.s > 0, fit.sigma2 / fit.s, 0.0))
        f_true = np.maximum(f_true + rng.standard_normal((n_draws, n_d - 1)) * se, 1e-12)

    ult = np.empty((n_draws, n_w))
    for i in range(n_w):
        state = np.full(n_draws, fit.cum[i, fit.latest_dev[i]])
        for j in range(int(fit.latest_dev[i]), n_d - 1):
            mean = f_true[:, j] * state
            var = fit.sigma2[j] * state
            state = draw_step(rng, mean, np.maximum(var, 0.0), law=process)
        ult[:, i] = state

    targets = pd.DataFrame(
        {
            "label": [str(o) for o in fit.origin_periods],
            "origin_period": fit.origin_periods,
        }
    )
    return PredictiveDistribution(samples=ult, targets=targets, units=fit.units).with_total("total")


def draw_step(
    rng: np.random.Generator, mean: np.ndarray, var: np.ndarray, *, law: str
) -> np.ndarray:
    """One development step's shock, matching ``mean`` and ``var`` elementwise.

    Shared by the run-off simulation above and the one-year re-reserving in
    ``kernels/cdr.py`` so both carry the identical noise assumption. Degenerate
    cells (zero variance - e.g. a step whose sigma was estimated as zero) come
    back at their mean exactly, under every law.

    ``normal`` is unfloored on purpose: Mack's model constrains two moments and
    nothing else, so flooring would bias the comparison against the (also
    linear, also unfloored) analytic CDR formula. ``gamma`` and ``lognormal``
    keep the simulated cumulative positive and need a positive mean.
    """
    if law not in PROCESS_LAWS:
        raise ValueError(f"process must be one of {PROCESS_LAWS}, got {law!r}")
    out = np.array(mean, dtype=float, copy=True)
    live = var > 0
    if not live.any():
        return out
    m, v = np.asarray(mean)[live], np.asarray(var)[live]
    if law == "normal":
        out[live] = m + np.sqrt(v) * rng.standard_normal(m.shape)
        return out
    if (m <= 0).any():
        raise ValueError(f"{law} process noise needs positive conditional means")
    if law == "gamma":
        out[live] = rng.gamma(shape=m**2 / v, scale=v / m)
    else:  # lognormal, moment-matched
        s2 = np.log1p(v / m**2)
        out[live] = rng.lognormal(mean=np.log(m) - 0.5 * s2, sigma=np.sqrt(s2))
    return out


def _tail_sigma2(sigma2: np.ndarray, j: int, *, rule: str) -> float:
    """Variance for a development step with no residual degrees of freedom."""
    known = np.array([v for v in sigma2[:j] if np.isfinite(v)])
    if known.size == 0:
        # Nothing to extrapolate from (a 2-column triangle): the step is
        # deterministic as far as the data can tell.
        return 0.0
    if rule == "log_linear" and known.size >= 2 and (known > 0).all():
        idx = np.array([k for k in range(j) if np.isfinite(sigma2[k])], dtype=float)
        slope, intercept = np.polyfit(idx, np.log(np.sqrt(known)), 1)
        # Deliberately uncapped: chainladder's loglinear_interpolation and R's
        # est.sigma="log-linear" both take the raw extrapolation, and the
        # tie-out tests pin us to it. Rising sigmas therefore extrapolate UP -
        # which is why Mack's own (always shrinking) rule is the default here.
        extrapolated = float(np.exp(intercept + slope * j) ** 2)
        if np.isfinite(extrapolated) and extrapolated > 0:
            return extrapolated
    if known.size == 1:
        return float(known[-1])
    last, prev = float(known[-1]), float(known[-2])
    ratio = last**2 / prev if prev > 0 else last
    return float(min(ratio, last, prev))
