"""One-year claims development result (CDR): how much the reserve estimate can
move over the NEXT twelve months, rather than over the whole run-off.

This is the Solvency II reserve-risk view. Mack's classical MSEP
(``kernels/mack.py``) answers "how wrong can the ultimate be?"; the one-year
CDR answers "how much can next year's balance sheet re-estimate move?", which
is what a one-year risk capital figure is calibrated to.

Definition. At time ``I`` the reserve estimate for accident year ``i`` is built
from the chain-ladder factors known then; one year later the new diagonal has
emerged and every factor is re-estimated. The (observable) claims development
result is the change in the estimated ultimate:

    CDR_i(I+1) = C-hat_{i,J}^{I} - C-hat_{i,J}^{I+1}
               = R_i^{I} - ( paid during the year + R_i^{I+1} )

so a positive CDR is a release and a negative one a strengthening. Under the
model E[CDR | D_I] = 0, and the risk measure is the conditional MSEP about
zero, ``msep = E[CDR^2 | D_I]``.

Two independent routes are implemented, deliberately ablatable against each
other (they answer the same question and must agree):

``merz_wuthrich``  the closed form of Merz & Wuthrich (2008), *Modelling the
                   claims development result for solvency purposes* (CAS
                   E-Forum, Fall 2008) - the market-standard analytic answer,
                   per accident year and aggregated.
``simulate``       "actuary in the box": simulate only the next diagonal from
                   the fitted Mack model, append it, re-run the chain ladder on
                   the extended triangle and take the difference. Gives the full
                   one-year CDR *distribution* (a ``PredictiveDistribution``,
                   per CLAUDE.md decision 4), not just its second moment - and
                   therefore the tail quantiles (``cdr_risk_measures``) that a
                   one-year capital figure is actually read off.

The analytic and simulated second moments agree to Monte Carlo error when the
simulation is run with the matching risk components - see
``tests/test_cdr.py::test_analytic_matches_simulation``.

Cross-refs: ``kernels/mack.py`` (the fit this consumes), ``kernels/contract.py``
(``cohort_grid``), ``gallery/deterministic/mack/card.md`` (the model card, which
documents the estimator and its limits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ibnr.kernels.mack import PROCESS_LAWS, MackFit, _next_step_draws
from ibnr.kernels.predictive import PredictiveDistribution


@dataclass(frozen=True)
class CDRResult:
    """One-year CDR uncertainty for one cohort, per accident year and in total.

    All ``msep`` fields are MEAN SQUARED ERRORS (variances about a zero CDR);
    take a square root for the standard error the summary table reports. The
    run-off figures come from the same ``MackFit`` and are carried alongside
    because the interesting number is usually the ratio of the two.
    """

    origin_periods: list
    ibnr: np.ndarray  # (n_w,) reserve at time I
    msep: np.ndarray  # (n_w,) one-year CDR msep
    msep_total: float
    runoff_msep: np.ndarray  # (n_w,) Mack full run-off msep
    runoff_msep_total: float
    method: str
    units: str | None = None

    def summary(self) -> pd.DataFrame:
        """One row per origin plus a ``total`` row: reserve, one-year standard
        error, run-off standard error, and the share of run-off risk that
        emerges in the first year (``one_year_share`` = cdr_se / runoff_se)."""
        cdr_se = np.sqrt(self.msep)
        runoff_se = np.sqrt(self.runoff_msep)
        out = pd.DataFrame(
            {
                "origin": self.origin_periods,
                "ibnr": self.ibnr,
                "cdr_se": cdr_se,
                "runoff_se": runoff_se,
            }
        )
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "origin": "total",
                            "ibnr": self.ibnr.sum(),
                            "cdr_se": np.sqrt(self.msep_total),
                            "runoff_se": np.sqrt(self.runoff_msep_total),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out["one_year_share"] = np.where(
                out["runoff_se"] > 0, out["cdr_se"] / out["runoff_se"], np.nan
            )
        return out


def _open_years(fit: MackFit) -> np.ndarray:
    """(n_w,) bool: origins that are not yet fully developed, i.e. the ones that
    still have a next diagonal cell to observe and therefore a CDR."""
    return fit.latest_dev < fit.n_d - 1


def _new_observation_origin(fit: MackFit) -> dict[int, int]:
    """dev step ``j`` -> the origin whose next diagonal cell joins that step's
    factor next year.

    On a run-off staircase exactly one origin sits at each dev index, so the
    update of ``f_j`` between time I and I+1 is driven by exactly one new
    observation: the one from the origin with ``latest_dev == j``. Dev steps
    deeper than the current diagonal get no new observation at all (only
    possible when the triangle has fewer origins than development steps), and
    are simply absent from the mapping.
    """
    return {int(j): int(i) for i, j in enumerate(fit.latest_dev) if j < fit.n_d - 1}


def one_year_cdr(fit: MackFit) -> CDRResult:
    """Merz-Wuthrich (2008) closed-form msep of the one-year CDR.

    Write ``ratio_j = sigma_j^2 / f_j^2``, ``S_j`` for the volume behind
    ``f_j`` today and ``S_j^{I+1} = S_j + C[i_j][j]`` for the same volume once
    the new diagonal has arrived, where ``i_j`` is the single accident year
    joining dev step ``j`` next year. The leverage of that one new observation
    on the re-estimated factor is

        a_j = C[i_j][j] / S_j^{I+1}

    and it is the whole story of the one-year view: within twelve months an
    accident year learns (i) its own next cell in full, and (ii) about every
    later factor ONLY through that single new observation. So for accident year
    ``i`` with its diagonal at dev ``k = latest_dev[i]``,

        Phi_i   = ratio_k / C[i][k]
                + sum_{j>k} ratio_j * C[i_j][j] / (S_j^{I+1})^2      (process)
        Delta_i = ratio_k / S_k
                + sum_{j>k} a_j^2 * ratio_j / S_j                    (estimation)
        msep_i  = Chat[i][J]^2 * (Phi_i + Delta_i)

    Compare Mack's run-off msep, ``Chat^2 * sum_{j>=k} ratio_j (1/Chat[i][j] +
    1/S_j)``: the one-year formula keeps the ``j = k`` terms in full and damps
    every later one by the leverage ``a_j`` (squared, for the estimation part).
    An accident year one step from ultimate therefore has one-year msep exactly
    equal to its run-off msep, which is the sharpest test of the formula and is
    asserted in ``tests/test_cdr.py``.

    Aggregation carries a cross term, because accident years share the factors
    they still have to run through. With
    ``V_i = ratio_k/S_k + sum_{j>k} a_j * ratio_j / S_j``,

        msep_total = sum_i Chat_i^2 ratio_{k_i} / C[i][k_i]
                   + sum_i sum_{i'} V_{min(i,i')} * Chat_i * Chat_{i'}

    - the min picks the OLDER year of each pair, whose (shorter) list of
    remaining steps is the set the two share. This is the grouping of Wuthrich's
    own reference implementation (R ChainLadder, ``CL_MSEPs``); it is
    algebraically the same total as the Phi/Delta split above, which
    ``tests/test_cdr.py`` pins. Note only the TOTAL is convention-free: R calls
    just ``ratio_k/C[i][k]`` process variance and lumps the rest into parameter
    uncertainty, where the split here follows the paper - ``Phi`` is the part a
    re-reserving simulation with ``parameter_risk=False`` reproduces.

    Valid only for the volume-weighted (alpha = 1) chain ladder and without a
    tail factor - the same two restrictions R's ``CDR.MackChainLadder``
    enforces, and both are structural here: ``fit_mack`` estimates nothing else.
    """
    # Phi_i divides by C_{i,k_i} - the latest diagonal, the one cell class no
    # factor-side guard can see. Checked here as well as inside msep_runoff()
    # below so the failure is named before the loop builds a page of NaN.
    fit.require_positive_open_diagonals()
    n_w, n_d = fit.n_w, fit.n_d
    last = n_d - 1
    k = fit.latest_dev
    open_ = _open_years(fit)
    ult = fit.ultimate
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(fit.f != 0, fit.sigma2 / fit.f**2, 0.0)  # (n_d - 1,)

    # the one new observation each dev step gains next year, and its leverage
    new_of = _new_observation_origin(fit)
    c_new = np.zeros(n_d - 1)  # C[i_j][j], 0 where no origin joins step j
    for j, i in new_of.items():
        c_new[j] = fit.cum[i, j]
    s_next = fit.s + c_new  # S_j^{I+1}
    alpha = np.divide(c_new, s_next, out=np.zeros_like(c_new), where=s_next > 0)

    phi = np.zeros(n_w)
    delta = np.zeros(n_w)
    v = np.zeros(n_w)  # the shared term that drives the aggregation cross sum
    for i in np.nonzero(open_)[0]:
        j0 = int(k[i])
        later = slice(j0 + 1, last)
        phi[i] = ratio[j0] / fit.cum[i, j0] + float(
            np.sum(ratio[later] * c_new[later] / np.where(s_next[later] > 0, s_next[later] ** 2, 1))
        )
        tail_s = np.where(fit.s[later] > 0, fit.s[later], np.inf)
        delta[i] = ratio[j0] / fit.s[j0] + float(np.sum(alpha[later] ** 2 * ratio[later] / tail_s))
        v[i] = ratio[j0] / fit.s[j0] + float(np.sum(alpha[later] * ratio[later] / tail_s))

    msep = ult**2 * (phi + delta)
    # aggregate: own-process terms, then every pair's shared estimation error
    own_process = np.zeros(n_w)
    own_process[open_] = ult[open_] ** 2 * ratio[k[open_]] / fit.cum[open_, k[open_]]
    older = np.minimum(np.arange(n_w)[:, None], np.arange(n_w)[None, :])  # (n_w, n_w)
    cross = float((v[older] * np.outer(ult, ult)).sum())

    runoff = fit.msep_runoff()
    return CDRResult(
        origin_periods=fit.origin_periods,
        ibnr=fit.reserve,
        msep=msep,
        msep_total=float(own_process.sum() + cross),
        runoff_msep=np.asarray(runoff["msep"]),
        runoff_msep_total=float(runoff["msep_total"]),
        method="merz_wuthrich",
        units=fit.units,
    )


def simulate_one_year_cdr(
    fit: MackFit,
    *,
    n_draws: int = 20_000,
    seed: int | None = None,
    process: str = "gamma",
    parameter_risk: bool = True,
) -> PredictiveDistribution:
    """Actuary in the box: the one-year CDR distribution by re-reserving.

    One draw is one possible next year:

    1. (``parameter_risk``) draw the *true* development factors from Mack's
       estimation-error distribution, ``f_j ~ N(f_j-hat, sigma_j^2 / S_j)``;
    2. draw the next diagonal cell of every open origin with Mack's conditional
       moments, ``E = f_{k_i} C_{i,k_i}``, ``Var = sigma_{k_i}^2 C_{i,k_i}``,
       independently across accident years;
    3. append that diagonal and RE-RUN the volume-weighted chain ladder. Only
       the factors move, and each moves by exactly one new observation:
       ``f_j^{I+1} = (S_j f_j-hat + X_{i_j}) / (S_j + C_{i_j,j})``;
    4. re-project each origin's ultimate off its new diagonal cell and take
       ``CDR_i = C-hat_{i,J}^{I} - C-hat_{i,J}^{I+1}``.

    ``parameter_risk`` is the ablation switch: off, the draws contain only the
    process risk of the next diagonal (the ``Phi`` half of the Merz-Wuthrich
    formula); on, they also carry the estimation error of the factors (its
    ``Delta`` half). ``process`` chooses the shape of the step-2 shock among
    ``PROCESS_LAWS`` - all three match Mack's first two moments, and only
    ``gamma``/``lognormal`` guarantee a positive diagonal. Mack's model itself
    fixes nothing beyond those two moments, so this choice is an assumption of
    the simulation, not of the model; it is the reason ``normal`` is offered
    (it is the shape the analytic linearization implicitly compares against).

    Returns per-origin CDR draws plus a ``total`` column derived from the same
    draws, so the diversification between accident years is in the samples.
    A positive draw is a reserve release.
    """
    if process not in PROCESS_LAWS:
        raise ValueError(f"process must be one of {PROCESS_LAWS}, got {process!r}")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    # step 2's Var = sigma_{k_i}^2 * C_{i,k_i} is non-positive off a non-positive
    # diagonal, and draw_step then returns the mean exactly - an invisible point
    # mass rather than an error, which is the one failure a simulation cannot
    # surface on its own
    fit.require_positive_open_diagonals()
    rng = np.random.default_rng(seed)
    n_w, n_d = fit.n_w, fit.n_d
    k = fit.latest_dev
    diag = fit.latest  # (n_w,) each origin's cumulative on the current diagonal
    open_ = _open_years(fit)

    # 1.-2. true factors per draw (parameter risk; Var(f_j-hat) = sigma_j^2/S_j
    # is Mack's estimation-error variance for the volume-weighted factor), then
    # the next diagonal cell of every open origin with Mack's conditional
    # moments. These two steps ARE the held-out one-step cell draw, so they
    # live once in ``kernels.mack._next_step_draws``, shared with
    # ``draw_next_cells`` - the CRPS board and the CDR cannot drift apart.
    x = np.zeros((n_draws, n_w))
    idx = np.nonzero(open_)[0]
    x[:, idx] = _next_step_draws(
        fit,
        k[idx],
        diag[idx],
        n_draws=n_draws,
        rng=rng,
        process=process,
        parameter_risk=parameter_risk,
    )

    # 3. re-estimate every factor on the extended triangle
    f_new = np.tile(fit.f, (n_draws, 1))
    for j, i in _new_observation_origin(fit).items():
        f_new[:, j] = (fit.s[j] * fit.f[j] + x[:, i]) / (fit.s[j] + diag[i])

    # 4. re-project: suffix products of the UPDATED factors, P[:, j] = prod_{t>=j} f_new[t]
    suffix = np.ones((n_draws, n_d))
    for j in range(n_d - 2, -1, -1):
        suffix[:, j] = suffix[:, j + 1] * f_new[:, j]
    ult_new = np.zeros((n_draws, n_w))
    ult_new[:, idx] = x[:, idx] * suffix[:, k[idx] + 1]
    # closed origins never move: their CDR is identically zero
    ult_new[:, ~open_] = fit.ultimate[~open_]

    cdr = fit.ultimate[None, :] - ult_new
    targets = pd.DataFrame(
        {
            "label": [str(o) for o in fit.origin_periods],
            "origin_period": fit.origin_periods,
        }
    )
    return PredictiveDistribution(samples=cdr, targets=targets, units=fit.units).with_total("total")


def simulated_msep(pred: PredictiveDistribution) -> np.ndarray:
    """MSEP about zero from simulated CDR draws, ``E[CDR^2]``.

    Not the sample variance: the CDR risk measure is the mean square about the
    zero the model predicts, so a simulation whose mean drifts off zero is
    penalised for it rather than being silently re-centred."""
    return (pred.samples**2).mean(axis=0)


def cdr_risk_measures(
    pred: PredictiveDistribution, levels: tuple[float, ...] = (0.995,)
) -> pd.DataFrame:
    """VaR and TVaR of the one-year LOSS, from simulated CDR draws.

    The capital question is asked on the adverse side, so everything here is
    stated on the loss ``-CDR`` (the reserve strengthening): ``VaR_0.995`` is
    the 99.5th percentile of that loss, the Solvency II reserve-risk basis, and
    ``TVaR_0.995`` its mean beyond that point. A negative VaR means even the
    adverse tail at that level is still a release.

    Quantiles are exact empirical order statistics of the draws, so the tail
    knots are as good as the draw count and no better - at 20k draws the 99.5th
    percentile rests on 100 observations. Raise ``n_draws`` before reading much
    into 99.9.

    Needs the simulated distribution rather than the analytic msep: a closed
    form gives a second moment, and no second moment implies a quantile.
    """
    if not all(0.0 < level < 1.0 for level in levels):
        raise ValueError(f"levels must lie strictly inside (0, 1), got {levels}")
    loss = -pred.samples  # adverse = the ultimate revised UP
    out = pred.targets.copy()
    out["mean_cdr"] = pred.samples.mean(axis=0)
    out["sd_cdr"] = pred.samples.std(axis=0, ddof=1)
    for level in levels:
        var = np.quantile(loss, level, axis=0)
        # TVaR over the draws at or beyond VaR; with few draws in the tail this
        # is a small-sample mean, which is exactly what the caller should see.
        tail = np.where(loss >= var, loss, np.nan)
        with np.errstate(invalid="ignore"):
            tvar = np.nanmean(tail, axis=0)
        out[f"var_{level:g}"] = var
        out[f"tvar_{level:g}"] = tvar
    return out
