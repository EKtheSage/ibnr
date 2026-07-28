"""Clark (2003) growth-curve reserving - maximum likelihood entry.

Clark, "LDF Curve-Fitting and Stochastic Reserving: A Maximum Likelihood
Approach" (CAS Forum 2003). Incremental losses follow an over-dispersed
Poisson whose mean is an ultimate times the increment of a parametric growth
curve G(x | omega, theta) evaluated at ages measured from the origin period's
average accident date (x = 12d - 6 for annual grains - the convention
chainladder-python's ClarkLDF also uses; the tieout tests pin this).

Methods, per the paper:
- ``ldf``: each origin carries its own ultimate U_w (profile MLE
  U_w = paid-to-date / G(age_w), so the point ultimate is the truncated-LDF
  answer paid * G(x_max)/G(age_w)).
- ``cape_cod``: U_w = ELR * premium_w with a single profiled ELR - Clark's
  recommendation for thin triangles.

``premium_field`` is read by ``cape_cod`` ONLY. An ``ldf`` fit never requests
the column, so it runs on a premium-free triangle with no extra arguments and
gives the same answer on a triangle that has one.

Predictive distribution = Clark's own variance decomposition, simulated:
parameter risk from the MVN with covariance phi * inverse observed Fisher
information (log-parameter space), process risk as scaled-Poisson ODP draws,
truncated at the triangle's final age (no tail extrapolation - the backtest
scores C[w, n_d], and chainladder's ClarkLDF truncates identically).

Relation to ``bayesian/clark_growth_curve``: identical model and growth-curve
parameterization, different inference. This statistical-family entry is
frequentist - a single MLE point estimate whose uncertainty is *simulated*
(parameter risk from an asymptotic MVN via the delta method, process risk from
scaled-Poisson draws), so no MCMC and no priors. The Bayesian twin instead
samples the joint posterior of the same parameters. Both emit the same
``PredictiveDistribution`` and feed the shared evaluation harness."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ibnr.gallery.entry import GalleryEntry, PredictsHeldout
from ibnr.gallery.registry import register
from ibnr.kernels.contract import odp_stan_data, realized_values
from ibnr.kernels.holdout import CellIndex
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

GROWTH_CURVES = ("loglogistic", "weibull")
METHODS = ("ldf", "cape_cod")

#: Objective value standing in for "this curve is infeasible" (some cell gets a
#: non-positive expected increment). A rejection sentinel, not a deviance -
#: :func:`_check_converged` refuses an optimum that never escaped it.
_REJECT_PENALTY = 1e12

#: Nelder-Mead simplex tolerance, on ``(log omega, log theta)``. Those stay
#: O(1) whatever the triangle's currency and size, so an ABSOLUTE tolerance is
#: meaningful here - and this is the criterion that actually pins the fit:
#: 1e-8 in log space is 1e-8 relative on omega and theta.
_XATOL = 1e-8

#: ``fatol`` as a FRACTION of the objective's magnitude, never an absolute
#: number. The concentrated Poisson deviance is ``sum(mu - inc*log(mu))`` over
#: loss AMOUNTS, so its magnitude is set by the data: about -4.3e8 on genins and
#: larger on a real Schedule P cohort. One ULP there is 6e-8, so the old
#: absolute ``fatol=1e-10`` demanded agreement ~600 million times finer than the
#: gap between adjacent representable float64s. Nelder-Mead terminates only when
#: BOTH the simplex spread <= xatol AND max|f_i - f_0| <= fatol, so that
#: criterion was satisfiable ONLY when every simplex vertex evaluated
#: BIT-IDENTICALLY (measured on genins: final f-spread exactly 0.0). It usually
#: did, by luck - until the ``pytest (all)`` CI leg, whose torch/jax/pymc
#: install changes BLAS/OMP thread counts and therefore the summation order
#: inside the objective, stalled to ``maxiter`` and raised on a weibull fit that
#: had converged in 90 of 2000 iterations.
#:
#: Shifting or rescaling the objective does NOT fix this. The resolution floor
#: comes from summing ~1e8-magnitude terms; subtracting ``f(x0)`` afterwards
#: moves the value without buying back a single bit. 1e-12 of the magnitude is
#: ~7e3 ULPs - comfortably reachable, and not the binding criterion anyway.
_FATOL_REL = 1e-12

#: Absolute floor under the scale-relative ``fatol``, for the degenerate case of
#: an objective whose magnitude is ~0 (there is nothing to be relative to).
_FATOL_MIN = 1e-10

#: Iteration cap. The genins fits use 59 (loglogistic) and 58 (weibull), so a
#: run anywhere near this cap is stalling rather than working. (Pre-fix they
#: took 65 and 90 - the extra iterations were spent chasing a ``fatol`` that
#: could not be met, not improving the answer: the fitted omega/theta/phi are
#: unchanged and the objective at the optimum is bit-identical.)
_MAXITER = 2000

#: Coordinate half of the ``maxiter`` fallback gate. A simplex collapsed to this
#: spread in log space has pinned omega and theta to 1e-6 relative - three orders
#: tighter than the 2e-3 the chainladder tieout asks of them - so the PARAMETERS
#: have stopped moving, whatever ``res.success`` says about the criterion it
#: could not tick. Necessary, NOT sufficient: see :data:`_FSPREAD_SLACK`.
_SIMPLEX_XTOL = 1e-6

#: Objective half of the ``maxiter`` fallback gate, as a multiple of the
#: ``fatol`` the fit asked for at this objective's scale. Coordinate collapse
#: alone is not convergence: Nelder-Mead's shrink steps can contract the simplex
#: to a point while its vertices still evaluate orders of magnitude apart -
#: classically against the :data:`_REJECT_PENALTY` plateau, where the objective
#: is flat because every move is infeasible and the optimizer is giving up
#: rather than finishing. So the objective must have collapsed too, and on ITS
#: OWN SCALE for the reason in :data:`_FATOL_REL`. The band is deliberately the
#: requested ``fatol`` times this slack rather than an independent number, so
#: the two cannot drift apart: 1e3 * 1e-12 = 1e-9 relative, still nine matching
#: significant figures across the simplex, while leaving three orders of room
#: over the tolerance a fit stalling on rounding noise missed by (summation-order
#: differences across a few dozen ~1e8-magnitude terms run to ~1e-14 relative,
#: five orders inside this band).
_FSPREAD_SLACK = 1e3


def _check_converged(res) -> float:
    """Refuse a curve MLE that did not converge; return the final simplex spread.

    Acceptance, in the order the checks run:

    1. EVERY vertex of the final simplex must be a real objective value - finite
       and off :data:`_REJECT_PENALTY`. Checked whatever ``res.success`` says,
       because a simplex can collapse INSIDE the rejection region and report
       success (a converged answer to the wrong question). Reading ``res.fun``
       instead is not enough: it is the BEST vertex, so a simplex straddling the
       feasibility boundary - one healthy vertex, the rest parked on the
       sentinel - looks perfectly fine through it.
    2. ``res.success`` is then HONORED. With a reachable ``fatol``
       (:data:`_FATOL_REL`) it is reachable again, and scipy's own pair of
       criteria (coordinate spread <= ``xatol`` AND ``max|f_i - f_0| <= fatol``)
       is strictly tighter than anything below, so there is nothing to re-derive.
    3. Only in the ``maxiter`` case do the relaxed criteria apply - and BOTH
       halves are required. This is the CI failure the fallback exists for:
       Nelder-Mead reports ``maxiter`` when its function-value criterion stays
       unmet for purely numerical reasons while the fit has long since arrived.
       Accepting that needs the simplex collapsed in COORDINATES
       (:data:`_SIMPLEX_XTOL` - the parameters have stopped moving) AND in
       OBJECTIVE VALUE, scale-relatively (:data:`_FSPREAD_SLACK` - the vertices
       agree about what they found). Coordinates alone is the hole: it accepts
       a simplex sitting on the penalty plateau, which is the optimizer
       surrendering, not converging.
    """
    sim, fvals = res.final_simplex
    fvals = np.asarray(fvals, dtype=float)
    spread = float(np.max(np.abs(sim[1:] - sim[0])))

    unusable = ~np.isfinite(fvals) | (fvals >= _REJECT_PENALTY)
    if unusable.any():
        raise RuntimeError(
            f"Clark MLE did not converge: {int(unusable.sum())} of {fvals.size} vertices of "
            "the final simplex give a non-positive expected increment in some cell (the "
            "objective's infeasibility sentinel), so the fit is anchored against the "
            "rejection region rather than resting on a feasible curve"
        )
    if res.success:
        return spread
    if spread > _SIMPLEX_XTOL:
        raise RuntimeError(
            f"Clark MLE did not converge: {res.message} (simplex spread {spread:.2e} "
            f"after {res.nit} iterations, needs <= {_SIMPLEX_XTOL:g})"
        )
    fspread = float(fvals.max() - fvals.min())
    # The fatol this objective's magnitude warrants, times the fallback slack.
    band = _FSPREAD_SLACK * max(_FATOL_MIN, _FATOL_REL * float(np.abs(fvals).max()))
    if fspread > band:
        raise RuntimeError(
            f"Clark MLE did not converge: {res.message} (the parameters stopped moving but "
            f"the objective did not - objective spread {fspread:.2e} across the final "
            f"simplex after {res.nit} iterations, needs <= {band:.2e})"
        )
    return spread


def growth(x: np.ndarray, omega: float, theta: float, curve: str) -> np.ndarray:
    """Clark's growth functions: expected fraction of ultimate paid by age x.
    G(0) = 0, G -> 1 as x -> inf.

    Clark (2003), the two curves he fits:
      loglogistic  G = x^omega / (x^omega + theta^omega)
      weibull      G = 1 - exp(-(x/theta)^omega)
    omega = shape (steepness), theta = scale (age at which half of ultimate is
    reported). x is a scalar age or a vector of ages, returned elementwise.

    **Unit-agnostic in x and theta**, and deliberately so - both appear only as
    the ratio ``theta/x``, so the function is correct for any age unit provided
    ``x`` and ``theta`` are on the SAME one. The Clark entries here use MONTHS
    (Clark's own convention, and what ``age_interval`` returns);
    ``bayesian/guszcza_growth_curve`` uses YEARS, so that its ``theta ~
    normal(4, 1)`` prior keeps the blog post's meaning on any dev grain. A
    fitted ``theta`` is therefore NOT comparable across those cards without
    converting - and the age conventions differ too (Clark measures from the
    origin's average accident date, ``12d - 6``; the Guszcza entry from the
    period end, ``12d``)."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    if curve == "loglogistic":
        # Computed as 1/(1 + (theta/x)^omega), algebraically identical to the
        # x^omega/(x^omega + theta^omega) card form but numerically stable.
        # overflow in (theta/x)^omega during optimizer exploration is benign:
        # 1/(1+inf) -> 0 is the correct limit
        with np.errstate(divide="ignore", over="ignore"):
            return np.where(x > 0, 1.0 / (1.0 + (theta / np.maximum(x, 1e-300)) ** omega), 0.0)
    if curve == "weibull":
        return 1.0 - np.exp(-((x / theta) ** omega))
    raise ValueError(f"growth must be one of {GROWTH_CURVES}, got {curve!r}")


def age_interval(d, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Clark's mid-period age interval for a cell at dev index ``d``:
    ``(max(step*(d-1) - step/2, 0), step*d - step/2]`` months.

    The single home of the convention - it used to be spelled out at four call
    sites (both Clark entries' fit and predict), and a fifth copy drifting by
    ``step/2`` would produce perfectly plausible mis-scaled draws. Ages are
    measured from the origin period's AVERAGE accident date, so the raw dev
    window ``((d-1)*step, d*step]`` is shifted back half a period (Clark 2003;
    the same ``x = 12d - 6`` convention as chainladder's ClarkLDF, pinned by
    the tieout tests). The first period's lower edge clamps to exactly ``0.0``,
    where :func:`growth` returns exactly ``0.0`` - both facts are load-bearing
    (see the zero-age gotcha in ``bayesian/clark_growth_curve/card.md``).

    ``d`` may be a scalar or an array of dev indices; ``step`` is the dev grain
    in months (``contract["dev_grain_months"]``).
    """
    d = np.asarray(d, dtype=float)
    lo = np.maximum(step * (d - 1) - step / 2, 0.0)
    hi = step * d - step / 2
    return lo, hi


def _hessian(f, x0: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of a scalar function; dims here are <= n_w + 2.

    Numerical observed information for Clark's parameter risk: the Hessian of
    the negative quasi-log-likelihood at the MLE. Returns a (k, k) matrix for
    k = len(x0) parameters (log-levels + log-omega + log-theta)."""
    k = len(x0)
    h = rel_step * np.maximum(np.abs(x0), 1.0)
    hess = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            hess[i, j] = hess[j, i] = (
                f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)
            ) / (4 * h[i] * h[j])
    return hess


@register
class Clark(GalleryEntry, PredictsHeldout):
    name = "clark"
    family = "statistical"

    #: Clark models INCREMENTAL emergence while the Schedule P triangles are
    #: cumulative, so ``predict_at`` must add each cell's training-diagonal
    #: anchor. Declared rather than assumed: an undeclared increment draw is
    #: wrong by that whole anchor while staying finite and plausible.
    heldout_draw_scale = "incremental"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.params_: dict | None = None
        self._loss_field: str | None = None
        #: held-out draw count. An attribute rather than an argument because
        #: ``PredictsHeldout`` fixes ``_draws_native``'s signature; unlike the
        #: Bayesian entries this one has no posterior whose size decides it.
        #: Defaults to ``predict()``'s 10_000.
        self.n_heldout_draws: int = 10_000

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        # ``str | None``, not ``str``: ``None`` is a SUPPORTED value here (it
        # switches exposure off, which ``ldf`` fits happily on and ``cape_cod``
        # refuses by name below) and it is spelled that way in the README, the
        # card and tests/test_clark_premium.py. Every other entry's
        # premium_field is genuinely ``str`` - see tests/test_field_annotations.
        premium_field: str | None = "earned_premium",
        as_of: dt.date | str | None = None,
        growth_curve: str = "loglogistic",
        method: str = "cape_cod",
    ) -> Clark:
        if growth_curve not in GROWTH_CURVES:
            raise ValueError(f"growth_curve must be one of {GROWTH_CURVES}")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        # Backtest slice, then the same incremental ODP contract the E&V entry
        # uses (Clark shares the od-Poisson likelihood, only the mean differs).
        #
        # BUILD FIRST, ASSIGN AFTER THE MLE SUCCEEDED - fit() must be atomic.
        # The optimizer and the dof/positivity checks below legitimately refuse
        # cohorts the contract accepted, and assigning contract_ before them
        # leaves a failed refit TORN: the new cohort's contract over the old
        # cohort's MLE, which index_into then accepts - predict_at would draw
        # one cohort's cells from another cohort's curve. See mack.
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # PREMIUM FOLLOWS THE METHOD, not the signature default. ``ldf`` anchors
        # each origin on its own paid-to-date and never reads exposure, so its
        # contract is built WITHOUT premium even when the triangle carries one:
        # requesting a column the fitted model never uses turns every premium
        # defect (no such field, a cohort mismatch, a non-positive value) into a
        # failure of a method that has no premium in it. An ldf fit therefore
        # carries no ``contract_["premium"]`` and ``predict()`` reports NaN
        # exposure in its targets - identical to passing ``premium_field=None``,
        # which was the workaround before this was the default.
        #
        # ``cape_cod`` is the opposite (U[w] = ELR * premium[w]), so it demands
        # exposure UP FRONT and names itself in the error: the contract's own
        # "no rows for premium field" answer does not tell a caller that the
        # cheapest fix may be ``method="ldf"``.
        needs_premium = method == "cape_cod"
        if needs_premium:
            if premium_field is None:
                raise ValueError("cape_cod needs a premium_field (U[w] = ELR * premium[w])")
            if premium_field not in train.fields:
                raise ValueError(
                    f"cape_cod needs a premium_field (U[w] = ELR * premium[w]) but the "
                    f"triangle carries no {premium_field!r} field (it has "
                    f"{train.fields}); name the exposure field, or use method='ldf', "
                    "which anchors on paid-to-date and needs no premium"
                )
        contract = odp_stan_data(
            train,
            loss_field=loss_field,
            premium_field=premium_field if needs_premium else None,
        )
        c = contract
        w, d, inc = c["w"], c["d"], c["inc_loss"]  # ragged (len_data,) per observed cell
        step = c["dev_grain_months"]  # 12 for annual grain
        # Each cell spans an age interval measured from the origin's average
        # accident date; the cell's expected increment is
        # U * (G(age_hi) - G(age_lo)). One shared home for the convention.
        age_lo, age_hi = age_interval(d, step)  # each (len_data,)

        if method == "ldf":
            row_tot = np.array([inc[w == wi].sum() for wi in range(1, c["n_w"] + 1)])
            if (row_tot <= 0).any():
                raise ValueError("ldf method needs positive paid-to-date in every origin")
        # None for ldf by construction (the contract was built without it); the
        # cape_cod gate above is what guarantees this is an array when it is used.
        premium = c.get("premium")

        def profiled_level(om: float, th: float) -> np.ndarray:
            """MLE of per-origin ultimates given the curve: U_w (ldf) or
            ELR * premium (cape_cod), both closed-form Poisson MLEs.

            Given (omega, theta), the level parameters maximize the Poisson
            likelihood in closed form, so the optimizer only searches the 2-D
            curve. ldf: U_w = (row paid) / (row sum of G-increments). cape_cod:
            a single ELR = total paid / sum(premium * G-increment)."""
            # Per-cell share of ultimate falling in this cell's age interval.
            ginc = growth(age_hi, om, th, growth_curve) - growth(age_lo, om, th, growth_curve)
            if method == "ldf":
                gsum = np.array([ginc[w == wi].sum() for wi in range(1, c["n_w"] + 1)])  # (n_w,)
                return np.array([inc[w == wi].sum() for wi in range(1, c["n_w"] + 1)]) / gsum
            elr = inc.sum() / (premium[w - 1] * ginc).sum()  # scalar Cape Cod ELR
            return elr * premium  # (n_w,)

        def negll_curve(logparams: np.ndarray) -> float:
            # logparams = (log omega, log theta); optimize in log space to keep
            # both curve parameters positive. Returns the concentrated Poisson
            # deviance (up to a data-only constant) with levels profiled out.
            om, th = np.exp(logparams)
            ginc = growth(age_hi, om, th, growth_curve) - growth(age_lo, om, th, growth_curve)
            if (ginc <= 0).any() or not np.isfinite(ginc).all():
                return _REJECT_PENALTY  # reject curves giving non-positive increments
            u = profiled_level(om, th)  # (n_w,)
            mu = u[w - 1] * ginc  # (len_data,) fitted cell means
            # -loglik for Poisson dropping the x-only terms: sum(mu - x*log mu).
            return float((mu - inc * np.log(mu)).sum())

        # Imported here, not at module scope: registering the gallery imports
        # every entry's module, so a top-level ``scipy.optimize`` - which pulls
        # ``scipy.sparse.linalg`` and ``scipy.linalg`` behind it - charges its
        # import to every caller, including the ones that never fit a Clark
        # curve. This one MLE call is the only use in the package.
        from scipy.optimize import minimize

        # 2-D Nelder-Mead over the curve; init omega=1.5, theta=4 periods.
        x0 = np.array([np.log(1.5), np.log(4 * step)])
        # SCALE-RELATIVE function tolerance (see _FATOL_REL): the deviance's
        # magnitude is the data's, so an absolute fatol is a lottery on float64
        # rounding. Costs one extra objective evaluation; should x0 itself be
        # infeasible the scale is the sentinel, which only makes fatol looser
        # than needed, and _XATOL binds regardless.
        fatol = max(_FATOL_MIN, _FATOL_REL * abs(negll_curve(x0)))
        res = minimize(
            negll_curve,
            x0,
            method="Nelder-Mead",
            options={"xatol": _XATOL, "fatol": fatol, "maxiter": _MAXITER},
        )
        simplex_spread = _check_converged(res)
        omega, theta = np.exp(res.x)
        level = profiled_level(omega, theta)  # (n_w,) per-origin ultimates at the MLE

        # For parameter risk Clark needs the FULL joint information over levels
        # and curve together (profiling hides the level<->curve correlation).
        # Assemble the full log-parameter vector at the MLE, everything in log
        # space so simulated draws stay positive by construction.
        if method == "ldf":
            full0 = np.concatenate([np.log(level), res.x])  # (n_w + 2,): log U_w, log om, log th
        else:
            elr = level[0] / premium[0]  # recover the scalar ELR from U = ELR*prem
            full0 = np.concatenate([[np.log(elr)], res.x])  # (3,): log ELR, log om, log th

        def negll_full(fp: np.ndarray) -> float:
            # Same Poisson deviance as negll_curve but with the levels held as
            # free parameters (not profiled) so the Hessian sees all dimensions.
            om, th = np.exp(fp[-2:])
            ginc = growth(age_hi, om, th, growth_curve) - growth(age_lo, om, th, growth_curve)
            if (ginc <= 0).any() or not np.isfinite(ginc).all():
                return _REJECT_PENALTY
            u = np.exp(fp[: c["n_w"]]) if method == "ldf" else np.exp(fp[0]) * premium
            mu = u[w - 1] * ginc  # (len_data,)
            return float((mu - inc * np.log(mu)).sum())

        # Fitted means at the MLE, aligned to the observed cells.
        ginc = growth(age_hi, omega, theta, growth_curve) - growth(
            age_lo, omega, theta, growth_curve
        )
        mu = level[w - 1] * ginc  # (len_data,)
        # Scale (dispersion) phi = Pearson chi-square / residual dof, Clark's
        # sigma^2 estimate; p = level params (n_w or 1) + the two curve params.
        n, p = len(inc), (c["n_w"] if method == "ldf" else 1) + 2
        if n <= p:
            raise ValueError(f"triangle has {n} cells but the Clark model has {p} parameters")
        phi = float(((inc - mu) ** 2 / mu).sum() / (n - p))
        # Observed information at the MLE (numerical Hessian of -loglik) ...
        hess = _hessian(negll_full, full0)  # (p, p)
        # ... inverted and scaled by phi = the quasi-likelihood delta-method
        # covariance of the log-parameters (Clark's parameter-risk covariance).
        cov = phi * np.linalg.pinv(hess)  # (p, p)

        self.contract_ = contract
        self._loss_field = loss_field
        self.params_ = {
            "growth_curve": growth_curve,
            "method": method,
            "omega": float(omega),
            "theta": float(theta),
            "level": level,
            "phi": phi,
            "log_params": full0,
            "log_cov": cov,
            "age_lo": age_lo,
            "age_hi": age_hi,
            # The optimizer's own report, recorded rather than discarded: with
            # the convergence gate now reading the simplex instead of trusting
            # res.success, these are the numbers that say whether a cohort's fit
            # sailed in or limped. `iterations` vs `max_iterations` is the stall
            # detector across a 200-company retrospective, and it is what
            # tests/test_clark.py pins on genins.
            "optimizer": {
                "objective": float(res.fun),
                "fatol": float(fatol),
                "xatol": _XATOL,
                "iterations": int(res.nit),
                "max_iterations": _MAXITER,
                "simplex_spread": simplex_spread,
                "success": bool(res.success),
            },
        }
        return self

    def predict(self, *, n_draws: int = 10_000, seed: int | None = None) -> PredictiveDistribution:
        """Ultimates per origin + total: paid-to-date plus simulated future
        increments through the final age, with Clark's parameter risk (MVN on
        the log-parameters) and ODP process risk (scaled Poisson)."""
        if self.contract_ is None or self.params_ is None:
            raise RuntimeError("call fit() first")
        c, prm = self.contract_, self.params_
        n_w, n_d, step = c["n_w"], c["n_d"], c["dev_grain_months"]
        curve, phi = prm["growth_curve"], prm["phi"]
        premium = c.get("premium")

        # PARAMETER RISK: draw the whole log-parameter vector from its
        # asymptotic MVN (mean = MLE, cov = the delta-method covariance above),
        # then exponentiate back to the natural scale. Shared with the held-out
        # scorer so predict() and predict_at() cannot drift apart.
        from ibnr.gallery.statistical.clark import scorer  # local: scorer imports this module

        rng = np.random.default_rng(seed)
        post = scorer.param_draws(c, prm, n_draws=n_draws, rng=rng)
        om = post["omega"]  # (n_draws,) curve shape per draw
        th = post["theta"]  # (n_draws,) curve scale per draw
        levels = post["level"]  # (n_draws, n_w) ultimate per origin per draw

        # Ultimate = observed paid-to-date + simulated future increments; start
        # every draw at the origin's latest cumulative paid (constant).
        ults = np.tile(c["paid_to_date"], (n_draws, 1)).astype(float)  # (n_draws, n_w)
        for j in range(n_w):  # per origin
            for dev in range(int(c["latest_d"][j]) + 1, n_d + 1):  # unobserved lags only
                # Age interval for this future cell (same convention as fit()).
                lo, hi = age_interval(dev, step)
                # Per-draw share of ultimate in the interval, then the cell mean.
                ginc = growth(hi, om, th, curve) - growth(lo, om, th, curve)  # (n_draws,)
                mu = np.maximum(levels[:, j] * ginc, 1e-12)  # (n_draws,)
                # PROCESS RISK: od-Poisson draw X ~ phi * Poisson(mu/phi),
                # mean mu, variance phi*mu - same process law as the ODP entries.
                ults[:, j] += phi * rng.poisson(mu / phi)

        targets = pd.DataFrame(
            {
                "label": [str(o.year) for o in c["origin_periods"]],
                "origin_period": c["origin_periods"],
                "premium": premium if premium is not None else np.nan,
            }
        )
        pred = PredictiveDistribution(samples=ults, targets=targets)
        return pred.with_total()

    def _draws_native(self, cells: CellIndex, *, rng: np.random.Generator) -> np.ndarray:
        """``(n_heldout_draws, n_cells)`` incremental draws. See ``scorer.draw_cells``.

        Three lines of glue over the same recipe ``predict()`` runs - MVN
        parameter sample plus od-Poisson process noise - factored into
        ``scorer.py`` so the two cannot drift. This entry's "posterior" is that
        MVN sample, drawn here from ``params_`` with the caller's ``rng``.
        """
        if self.params_ is None:
            raise RuntimeError("call fit() first")
        from ibnr.gallery.statistical.clark import scorer  # local: scorer imports this module

        return scorer.draw_cells(
            self.contract_, self.params_, cells, n_draws=self.n_heldout_draws, rng=rng
        )

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to predict()'s targets (per origin + total)."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        per_origin = realized_values(
            full_triangle,
            loss_field=self._loss_field,
            dev_lag=c["n_d"] * c["dev_grain_months"],
            origins=c["origin_periods"],
        )
        return np.append(per_origin, per_origin.sum())
