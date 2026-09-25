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

One other quantity in this package is also called a CDR and is not this one:
``kernels.replay``'s ``cdr`` column is the OBSERVED change in one conventional
candidate's fitted ultimate over one development period, one number per origin,
written the opposite way round so that positive is adverse, and carrying no
distribution at all.

ANNUAL DEVELOPMENT GRAIN ONLY. "One year" here means one development step, and
the two are the same span only when the triangle develops in twelve-month
steps. The three functions that name a year, :func:`one_year_cdr`,
:func:`simulate_one_year_cdr` and :func:`rereserve`, therefore refuse a fit
whose development grain is quarterly or monthly, naming the grain they
measured. The two generators do not, because a generator only draws the next
step and never calls it a year. The arithmetic would be right for one step, but
on a quarterly or a monthly fit that step is three months or one rather than
the twelve the name promises, and nothing in the answer would show it. The
remedy is to aggregate the triangle to an annual grain first, from a year-end
valuation, or, when the whole run-off is the question rather than one year of
it, to read ``MackFit.msep_runoff()``, which does not depend on the grain.

TWO INDEPENDENT AXES, because that is what the question has.

Every simulated one-year CDR is the same two-step recipe, and the two steps are
separate choices:

1. **What generates next year's diagonal?** A :class:`DiagonalGenerator`.
   ``mack`` draws it from Mack's conditional moments; ``odp_bootstrap`` draws it
   from an England-Verrall Pearson-residual bootstrap with over-dispersed
   Poisson process noise. Anything that can draw a next-diagonal cell qualifies,
   and as of 0.5.1 that includes every gallery entry with ``PredictsHeldout`` -
   ``ibnr.gallery.GalleryDiagonal``, which lives in the gallery because
   ``kernels`` never imports it (design decision 8).
2. **How is the reserve re-estimated once that diagonal exists?** The
   volume-weighted chain ladder - :func:`rereserve`. This is the market
   convention and it is what R's ``ChainLadder`` uses for *both* of its CDR
   methods, so it is one implementation shared by every generator rather than
   an option.

Before this release the two were fused to Mack inside ``simulate_one_year_cdr``.
:func:`cdr_methods` is the discoverable list of what is available, what each
route requires and returns, and how each one is validated.

THE CLOSED FORM IS NOT ONE OPTION AMONG MANY. :func:`one_year_cdr` implements
Merz & Wuthrich (2008), *Modelling the claims development result for solvency
purposes* (CAS E-Forum, Fall 2008) - a linearization of the chain-ladder factor
update around Mack's conditional moments. It is Mack-specific by construction:
there is no ODP bootstrap version of it, and it is deliberately NOT reachable
through a ``generator=`` argument. Asking for it there is refused by name, in
the same spirit as the milestone-6 capability mixins - a method that cannot
answer says so instead of returning a plausible number.

The analytic and simulated second moments agree to Monte Carlo error when the
simulation is run on the ``mack`` generator with the matching risk components -
see ``tests/test_cdr.py::test_analytic_matches_simulation``.

Cross-refs: ``kernels/mack.py`` (the fit this consumes),
``kernels/odp_bootstrap.py`` (the ODP generator's engine),
``kernels/contract.py`` (``cohort_grid``),
``gallery/deterministic/mack/card.md`` (the model card, which documents the
estimators and their limits).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from ibnr.errors import Refusal, RefusedCell, _literal
from ibnr.kernels.links import settings_named
from ibnr.kernels.mack import PROCESS_LAWS, MackFit, _next_step_draws, _refuse_n_draws
from ibnr.kernels.odp_bootstrap import ODP_PROCESS_LAWS, draw_next_increments, fit_odp_bootstrap
from ibnr.kernels.predictive import PredictiveDistribution

#: draws a generator produces when the caller names no count. A Monte Carlo
#: budget, so it belongs to the generators that HAVE one - a generator wrapping
#: a fitted posterior does not, and says so by returning None from
#: :meth:`DiagonalGenerator.resolve_n_draws`.
DEFAULT_N_DRAWS = 20_000


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
        emerges in the first year (``one_year_share`` = cdr_se / runoff_se).

        ``cdr_se`` is a one-year figure when this result came from
        :func:`one_year_cdr`, which refuses any development grain other than
        twelve months, so there the column heading and the span agree. A
        :class:`CDRResult` decoded from Arrow bytes carries no grain and is not
        checked, so a payload written by an older version can hold a figure for
        a shorter span."""
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

    # -- serialization ---------------------------------------------------------

    def to_arrow(self, *, compression: str | None = None) -> bytes:
        """Arrow IPC bytes. The two totals ride as shape-() float arrays rather
        than header scalars, because a degenerate cohort's msep is legitimately
        NaN and strict JSON cannot represent it."""
        from ibnr.kernels import codec

        return codec.to_arrow(self, compression=compression)

    @classmethod
    def from_arrow(cls, data: bytes) -> CDRResult:
        """Decode a CDR written by :meth:`to_arrow`, refusing any other kind."""
        from ibnr.kernels import codec

        return codec.from_arrow(data, expect="CDRResult")


def _require_annual_step(fit: MackFit) -> None:
    """Refuse a fit whose development step is not twelve months.

    Everything in this module advances the triangle by exactly ONE development
    step and calls the answer a one-year claims development result. The two are
    the same thing only on an annual triangle. On a quarterly or a monthly fit
    the step is three months or one, so the number would be a three-month or a
    one-month development result reported under a one-year name: finite,
    plausible, and covering a shorter span than the name says, with nothing in
    the output to show it.

    Called inside :func:`one_year_cdr`, :func:`simulate_one_year_cdr` and
    :func:`rereserve` rather than on :meth:`DiagonalGenerator.check`, because a
    third-party diagonal re-reserved through :func:`rereserve` reaches no
    generator at all, and because the gallery's ``mack`` entry binds these
    functions by name when it is imported.

    The first remedy the message names is qualified, because the aggregation on
    its own is not enough: ``with_dev_grain("Y")`` anchors the annual buckets to
    the latest diagonal, so a triangle whose latest valuation is not a year end
    aggregates to development lags such as 9, 21 and 33, which ``validate``
    flags and ``fit_mack`` then refuses. Slicing to a year end with ``as_of``
    first is what makes the route work.
    """
    step = fit.dev_grain_months
    if step == 12:
        return
    raise Refusal(
        "not_supported",
        "the one-year claims development result needs an annual development grain, and "
        f"this fit has a {step}-month development grain. Every route here advances the "
        "triangle by exactly one development step (the Merz-Wuthrich closed form, every "
        "DiagonalGenerator and rereserve()), and here that step is not twelve months, so "
        f"the answer would be a {step}-month development result reported under a one-year "
        "name. Either aggregate the triangle to an annual grain first, "
        "which costs development resolution: slice to a year-end valuation with as_of() "
        'and then call Triangle.with_origin_grain("Y").with_dev_grain("Y"), since the '
        "annual buckets are anchored to the latest diagonal and a mid-year one gives "
        "development lags the annual grain rejects. Or read the run-off uncertainty from "
        "MackFit.msep_runoff(), which does not depend on the development grain",
        option="dev_grain_months",
        given=step,
    )


def _require_zero_cells_unused(fit: MackFit) -> None:
    """Refuse a fit on which ``zero_cells="missing"`` changed anything.

    The Merz-Wuthrich formulas, and the re-reserving in :func:`rereserve`, are
    written for the volume-weighted chain ladder over every observed pair, and
    nobody has checked them under chainladder-python's rule that a zero cell is
    missing. Two things change under that rule: link ratios with a zero at
    either end leave the factors, and a still-developing origin can sit at a
    latest amount of zero, which every formula here divides by. A ``"missing"``
    fit with neither is the identical fit to an ``"observed"`` one and is
    accepted.

    Called inside :func:`one_year_cdr`, :func:`simulate_one_year_cdr` and
    :func:`rereserve`, for the same reason as :func:`_require_annual_step`:
    every route to a one-year result, including a third-party diagonal and
    ``ibnr.gallery.GalleryDiagonal``, passes through one of the three.
    """
    if fit.zero_cells != "missing":
        return
    open_ = fit.latest_dev < fit.n_d - 1
    zero_latest = [
        RefusedCell(None, o, (int(k) + 1) * fit.dev_grain_months, 0.0)
        for o, k, z in zip(
            fit.origin_periods, fit.latest_dev, open_ & (fit.latest == 0), strict=True
        )
        if z
    ]
    if not fit.zero_links and not zero_latest:
        return
    found = []
    if fit.zero_links:
        found.append(f"left out {fit.zero_links} link ratio(s) with a zero cell at either end")
    if zero_latest:
        found.append("kept a latest amount of zero for open origin(s) {origins}")
    if zero_latest:
        # "observed" refuses a zero latest amount too (every formula here divides
        # by it), so refitting that way would only move the refusal
        way_out = (
            "A one-year result needs every open origin's latest amount to be positive under "
            "either setting, so read the run-off uncertainty from MackFit.msep_runoff(), "
            "which covers this rule"
        )
    else:
        way_out = (
            "Refit with zero_cells='observed' (fit_mack's default), which keeps zeros as data, "
            "or read the run-off uncertainty from MackFit.msep_runoff(), which covers this rule"
        )
    raise Refusal(
        "not_supported",
        "the one-year claims development result is not available for a fit made with "
        f"zero_cells='missing' that {' and '.join(found)}: the Merz-Wuthrich formulas, and "
        "the re-reserving that simulate_one_year_cdr and rereserve do, have not been checked "
        f"under that rule. {way_out}",
        option="zero_cells",
        cells=zero_latest,
    )


def _require_all_history_volume(fit: MackFit) -> None:
    """Refuse a fit made with development options: another average, a window,
    exclusions, bounds or drops.

    The Merz-Wuthrich formulas are derived for the volume-weighted chain ladder
    over every link ratio (every term is ``sigma_j^2 / f_j^2`` with ``S_j`` the
    column total), and :func:`rereserve` re-runs exactly that estimator on next
    year's triangle. The check reads the SETTINGS, not what they removed:
    ``history_periods=9`` removes nothing from a 9 x 9 triangle and a ratio
    from next year's, which is the triangle ``rereserve`` fits.

    Called beside :func:`_require_zero_cells_unused`, in the same three places.
    """
    if fit.all_history_volume:
        return
    named = [("average", f"average={fit.average!r}")] if fit.average != "volume" else []
    if fit.links is not None:
        named.extend(settings_named(fit.links))
    options = [option for option, _ in named]
    phrases = [phrase for _, phrase in named]
    listed = phrases[0] if len(phrases) == 1 else f"{', '.join(phrases[:-1])} and {phrases[-1]}"
    raise Refusal(
        "not_supported",
        "the one-year claims development result needs the volume-weighted chain ladder over "
        "every link ratio: the Merz-Wuthrich formulas are derived for that estimator and "
        f"rereserve() re-runs it on next year's triangle. This fit used {listed}. Refit "
        "without development options for a one-year result, or read the run-off uncertainty "
        "of this fit from MackFit.msep_runoff()",
        option=options[0],
        options=tuple(options),
    )


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

    Valid only for the volume-weighted (alpha = 1) chain ladder, without a tail
    factor, and on an annual development grain. The first two are the same
    restrictions R's ``CDR.MackChainLadder`` enforces, and both are structural
    here: ``fit_mack`` estimates nothing else. The third is what makes the
    single development step above a year, and a fit on any other grain is
    refused by name. So is a fit made with ``zero_cells="missing"`` that left
    out a link ratio or kept a zero latest amount (see
    ``_require_zero_cells_unused``), because nobody has checked the
    formula under that rule.

    MACK-SPECIFIC BY CONSTRUCTION, and deliberately not offered as a
    ``generator=`` option on :func:`simulate_one_year_cdr`. Every term above is
    ``sigma_j^2 / f_j^2``: the formula is a linearization of the chain-ladder
    factor update *around Mack's conditional moments*, so there is no ODP
    bootstrap version of it and no version for any other model. The routes that
    do generalize are the simulated ones - :func:`cdr_methods` lists them.
    """
    # One development step is one year only on an annual triangle, and every
    # term below is one step wide.
    _require_annual_step(fit)
    _require_zero_cells_unused(fit)
    _require_all_history_volume(fit)
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


# ---------------------------------------------------------------------------
# axis 2: how the reserve is re-estimated once next year's diagonal exists
# ---------------------------------------------------------------------------


def rereserve(fit: MackFit, next_diagonal: np.ndarray) -> np.ndarray:
    """``(n_draws, n_w)`` one-year CDR draws from simulated next-diagonal values.

    The second axis, and it is not a choice: the reserve is re-estimated with
    the **volume-weighted chain ladder**, which is the market convention and
    what R's ``ChainLadder`` uses for its Mack CDR and its bootstrap CDR alike
    (``getAvDFs(dfs, wghts)`` with the cumulative triangle as weights *is* the
    volume-weighted factor written out). Every generator therefore shares this
    one implementation, and a third-party diagonal - draws from any model that
    can predict next year's cells - can be re-reserved by calling this directly.

    ``next_diagonal[:, i]`` is origin ``i``'s **cumulative** loss one year on,
    on the same basis as ``fit.cum``. Closed origins are ignored (they have no
    next cell, so their CDR is identically zero); pass zeros there.

    What happens, per draw:

    1. append the diagonal and re-run the chain ladder. Only the factors move,
       and each moves by exactly one new observation, because on a run-off
       staircase exactly one origin joins each development step next year::

           f_j^{I+1} = (S_j f_j-hat + X_{i_j}) / (S_j + C_{i_j,j})

       which is algebraically the volume-weighted factor recomputed on the
       extended triangle - the numerator ``S_j f_j-hat`` is ``sum_i C_{i,j+1}``
       and the denominator is ``sum_i C_{i,j}``, so adding the new pair to each
       is the refit;
    2. re-project each origin's ultimate off its new diagonal cell with the
       suffix product of the UPDATED factors;
    3. ``CDR_i = C-hat_{i,J}^{I} - C-hat_{i,J}^{I+1}``, so a positive draw is a
       reserve release.

    Only ``f``, ``s``, ``latest``, ``latest_dev`` and ``ultimate`` are read off
    the fit - the volume-weighted chain ladder and the triangle it came from.
    Mack's ``sigma2`` is not touched, which is why the ODP bootstrap can use
    the identical function.

    ONE MACK ASSUMPTION SURVIVES HERE AND IT IS WORTH NAMING. Step 3 differences
    against ``fit.ultimate``, the *deterministic* chain-ladder ultimate at time
    I. Under Mack that is exactly right and it is what makes ``E[CDR | D_I] =
    0`` - the draws are centred on zero, so ``E[CDR^2]`` (:func:`simulated_msep`)
    is the risk measure. Under a residual bootstrap the simulated diagonal's
    mean is only *approximately* the chain-ladder projection, so the CDR draws
    carry a small bootstrap bias and the mean square about zero is not quite the
    variance. R makes the same distinction from the other side: its Mack CDR
    reports the analytic msep (about zero) while ``CDR.BootChainLadder`` reports
    ``sd()`` of the re-reserved amount (about its own mean). Report both when
    the generator is not ``mack``.

    Under a GALLERY generator the drift is not a nuisance at all - it is the
    model saying next year's diagonal will land somewhere other than where the
    chain ladder puts it, which is most of what a separate model is FOR. It is
    still not part of a variance, so the same rule applies with more force:
    ``mean`` and ``sd`` separately, never ``E[CDR^2]`` alone.
    """
    x = np.asarray(next_diagonal, dtype=float)
    if x.ndim != 2 or x.shape[1] != fit.n_w:
        raise Refusal(
            "invalid_option",
            f"next_diagonal must be (n_draws, n_w={fit.n_w}) cumulative values, got "
            f"{x.shape}. One column per origin of the fit, in its origin order",
            option="next_diagonal",
        )
    # the shape says how many origins the diagonal covers; the grain says how
    # far forward it is, and one step forward is one year only on an annual fit.
    _require_annual_step(fit)
    _require_zero_cells_unused(fit)
    _require_all_history_volume(fit)
    n_draws = x.shape[0]
    n_w, n_d = fit.n_w, fit.n_d
    k = fit.latest_dev
    diag = fit.latest  # (n_w,) each origin's cumulative on the current diagonal
    open_ = _open_years(fit)
    idx = np.nonzero(open_)[0]

    # 1. re-estimate every factor on the extended triangle
    f_new = np.tile(fit.f, (n_draws, 1))
    for j, i in _new_observation_origin(fit).items():
        f_new[:, j] = (fit.s[j] * fit.f[j] + x[:, i]) / (fit.s[j] + diag[i])

    # 2. re-project: suffix products of the UPDATED factors, P[:, j] = prod_{t>=j} f_new[t]
    suffix = np.ones((n_draws, n_d))
    for j in range(n_d - 2, -1, -1):
        suffix[:, j] = suffix[:, j + 1] * f_new[:, j]
    ult_new = np.zeros((n_draws, n_w))
    ult_new[:, idx] = x[:, idx] * suffix[:, k[idx] + 1]
    # closed origins never move: their CDR is identically zero
    ult_new[:, ~open_] = fit.ultimate[~open_]

    # 3. the observable claims development result
    return fit.ultimate[None, :] - ult_new


# ---------------------------------------------------------------------------
# axis 1: what generates next year's diagonal
# ---------------------------------------------------------------------------


class DiagonalGenerator(ABC):
    """What next year's diagonal might be. The first axis of a simulated CDR.

    Two methods, and the split is the point. :meth:`check` asks whether this
    generator can answer for this cohort AT ALL and raises naming the cause if
    not - the honest-refusal idiom the held-out capability mixins use, moved to
    a place where the refusals genuinely differ: ``mack`` needs a strictly
    positive open diagonal (its conditional variance is proportional to it) and
    tolerates negative increments; ``odp_bootstrap`` is the exact reverse. That
    asymmetry is why the guard could not stay in ``simulate_one_year_cdr``,
    where it applied Mack's precondition to everything.

    :meth:`draw` returns ``(n_draws, n_w)`` **cumulative** losses one year on,
    zero at closed origins, ready for :func:`rereserve`.

    Each generator carries its own knobs as dataclass fields rather than taking
    them from a shared ``simulate_one_year_cdr`` signature. That is deliberate:
    ``process``/``parameter_risk`` are Mack's vocabulary and
    ``process_noise``/``resample_residuals`` are the bootstrap's, and a shared
    signature would let a caller pass one generator's knob and have it silently
    ignored by another - this repo's named inert-parameter bug class, where the
    answer looks perfect because the argument never reached anything.

    Not exported from ``ibnr.kernels``, by the same rule that keeps
    ``ScoresHeldout``/``PredictsHeldout`` at ``ibnr.gallery.entry``: subclassing
    it is how a generator DECLARES itself, and a caller only ever constructs the
    concrete ones.
    """

    #: registry key, and what ``generator="..."`` accepts
    name: ClassVar[str]

    @abstractmethod
    def check(self, fit: MackFit) -> None:
        """Raise, naming the cause, if this generator cannot serve this cohort."""

    @abstractmethod
    def draw(self, fit: MackFit, *, n_draws: int | None, rng: np.random.Generator) -> np.ndarray:
        """``(n_draws, n_w)`` cumulative loss on next year's diagonal.

        ``n_draws`` is whatever :meth:`resolve_n_draws` returned, so a generator
        that never returns None from there can read it as an ``int``.
        """

    def resolve_n_draws(self, requested: int | None) -> int | None:
        """How many draws to ask :meth:`draw` for, given what the caller asked.

        Default: the caller's number, or :data:`DEFAULT_N_DRAWS` when they named
        none. That is right for any generator whose draw count is a **Monte
        Carlo budget** - ``mack`` and ``odp_bootstrap`` both simulate as many
        diagonals as they are asked for, and more of them only buys precision.

        It is wrong for a generator whose draws are a **fitted posterior**,
        where the count is a property of the fit and not of this call. Such a
        generator returns ``None`` for "however many the source has" and refuses
        a mismatched explicit count, rather than resampling posterior draws with
        replacement - which would add Monte Carlo noise and no information while
        making the answer look like it had the precision of the larger number.
        ``ibnr.gallery.GalleryDiagonal`` is the case this exists for.
        """
        return DEFAULT_N_DRAWS if requested is None else requested


@dataclass(frozen=True)
class MackDiagonal(DiagonalGenerator):
    """Next year's diagonal from Mack's conditional moments.

    Per draw: optionally draw the *true* development factors from Mack's
    estimation-error distribution ``f_j ~ N(f_j-hat, sigma_j^2 / S_j)``, then
    draw each open origin's next cell with ``E = f_{k_i} C_{i,k_i}``,
    ``Var = sigma_{k_i}^2 C_{i,k_i}``, independently across accident years.

    ``parameter_risk`` is the risk-source switch: off, the draws contain only the
    process risk of the next diagonal (the ``Phi`` half of the Merz-Wuthrich
    formula); on, they also carry the estimation error of the factors (its
    ``Delta`` half). ``process`` chooses the shape of the shock among
    ``kernels.mack.PROCESS_LAWS`` - all three match Mack's first two moments,
    and only ``gamma``/``lognormal`` guarantee a positive diagonal. Mack's model
    fixes nothing beyond those two moments, so this choice is an assumption of
    the simulation, not of the model; it is the reason ``normal`` is offered
    (it is the shape the analytic linearization implicitly compares against).

    The draw itself is ``kernels.mack._next_step_draws``, shared with
    ``draw_next_cells`` - the leaderboard's CRPS and this CDR cannot drift apart.
    """

    process: str = "gamma"
    parameter_risk: bool = True

    name = "mack"

    def __post_init__(self) -> None:
        if self.process not in PROCESS_LAWS:
            raise Refusal(
                "invalid_option",
                f"process must be one of {PROCESS_LAWS}, got {{given}}",
                option="process",
                given=self.process,
            )

    def check(self, fit: MackFit) -> None:
        """``Var = sigma_{k_i}^2 C_{i,k_i}`` is non-positive off a non-positive
        diagonal, and ``draw_step`` then returns the mean exactly - an invisible
        point mass rather than an error, which is the one failure a simulation
        cannot surface on its own."""
        _require_zero_cells_unused(fit)
        _require_all_history_volume(fit)
        fit.require_positive_open_diagonals()

    def draw(self, fit: MackFit, *, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        k = fit.latest_dev
        x = np.zeros((n_draws, fit.n_w))
        idx = np.nonzero(_open_years(fit))[0]
        x[:, idx] = _next_step_draws(
            fit,
            k[idx],
            fit.latest[idx],
            n_draws=n_draws,
            rng=rng,
            process=self.process,
            parameter_risk=self.parameter_risk,
        )
        return x


@dataclass(frozen=True)
class ODPBootstrapDiagonal(DiagonalGenerator):
    """Next year's diagonal from an England-Verrall ODP residual bootstrap.

    The generator behind R's ``CDR.BootChainLadder``: resample DoF-adjusted
    Pearson residuals into a pseudo-triangle, refit the chain ladder on it,
    project the next diagonal off that refit and add over-dispersed Poisson
    process noise. The mechanics are ``kernels/odp_bootstrap.py``, where they
    can be read against R's source; this class is the CDR-side wiring.

    ``process`` is the noise law (``od_poisson`` = ``phi * Poisson(mu/phi)``,
    the England-Verrall construction ``england_verrall_odp.predict`` also draws,
    or ``gamma``; both have mean ``mu`` and variance ``phi * mu``). Note R's
    ``BootChainLadder`` defaults to ``gamma``, so a like-for-like comparison
    with R needs ``process="gamma"`` set explicitly.

    ``resample_residuals`` and ``process_noise`` are the two risk-source switches -
    R's ``NYCost`` arm is both on, its ``NYParamDist`` arm is
    ``process_noise=False``, and the process-only arm R derives by subtraction
    is ``resample_residuals=False``. Turning both off is refused: every draw
    would be identical, which is a degenerate answer rather than a meaningful
    with-and-without comparison.

    **The family limit is real and is refused by name.** The ODP quasi-likelihood
    is defined on non-negative increments, so a cohort with a negative paid
    increment cannot be bootstrapped - about half the Schedule P mart, the same
    limitation ``england_verrall_odp`` carries. :meth:`check` raises there and
    says so, and points at the ``mack`` generator, which has no such restriction.
    """

    process: str = "od_poisson"
    process_noise: bool = True
    resample_residuals: bool = True

    name = "odp_bootstrap"

    def __post_init__(self) -> None:
        if self.process not in ODP_PROCESS_LAWS:
            raise Refusal(
                "invalid_option",
                f"process must be one of {ODP_PROCESS_LAWS}, got {{given}}",
                option="process",
                given=self.process,
            )
        if not (self.process_noise or self.resample_residuals):
            raise Refusal(
                "invalid_option",
                "process_noise and resample_residuals are both off, so the bootstrap has no "
                "risk source left and every draw would be identical. Turn one back on, or "
                "read the point estimate off MackFit.reserve",
                option="process_noise",
                options=("process_noise", "resample_residuals"),
            )

    def check(self, fit: MackFit) -> None:
        """Build the deterministic half of the bootstrap and discard it: it is
        where the negative-increment refusal and the degrees-of-freedom check
        live, and both are cheap enough to pay twice."""
        _require_zero_cells_unused(fit)
        _require_all_history_volume(fit)
        fit_odp_bootstrap(
            fit.cum,
            fit.obs_mask,
            fit.latest_dev,
            fit.f,
            origins=fit.origin_periods,
            dev_grain_months=fit.dev_grain_months,
        )

    def draw(self, fit: MackFit, *, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        boot = fit_odp_bootstrap(
            fit.cum,
            fit.obs_mask,
            fit.latest_dev,
            fit.f,
            origins=fit.origin_periods,
            dev_grain_months=fit.dev_grain_months,
        )
        payments = draw_next_increments(
            boot,
            n_draws=n_draws,
            rng=rng,
            process=self.process,
            process_noise=self.process_noise,
            resample_residuals=self.resample_residuals,
        )
        # R's getTriangleNextYear: the simulated payments join the ORIGINAL
        # triangle's diagonal, not the pseudo-triangle's. The pseudo-triangle
        # only ever sets the payments' scale.
        return np.where(_open_years(fit)[None, :], fit.latest[None, :] + payments, 0.0)


# ---------------------------------------------------------------------------
# the option surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CDRMethod:
    """One row of :func:`cdr_methods` - a route to a one-year CDR and its terms.

    ``generator`` is the :class:`DiagonalGenerator` **class** (not an instance),
    mirroring ``gallery.get(name)``: it is what a caller constructs to configure
    the method, ``get_cdr_method("odp_bootstrap").generator(process="gamma")``.

    ``generator is None`` and ``why_not_by_name`` go together, and exactly one
    of the two states is legal for any row: either the class is here and
    ``generator="<name>"`` builds it, or it is absent and ``why_not_by_name``
    says why a name cannot. **The two reasons a row is nameless are different
    and ``route`` tells them apart.** ``merz_wuthrich`` is ``route="analytic"``:
    it generates no diagonal at all, so there is nothing to name. ``gallery`` is
    ``route="simulation"``: it is a perfectly good generator, but it wraps a
    FITTED entry and the cells it predicts at, and no string can carry those.
    """

    name: str
    route: str  # "analytic" | "simulation"
    generates: str  # what produces next year's diagonal
    re_estimates: str  # the second axis - the same for every simulation route
    returns: str
    entry_point: str
    requires: str
    validated: str
    reference: str
    generator: type[DiagonalGenerator] | None = None
    #: why ``generator="<this name>"`` is refused, or None when it is accepted.
    #: Reads as the middle of a sentence - see :func:`_resolve_generator`.
    why_not_by_name: str | None = None


#: every route to a one-year CDR, keyed by name. The two routes ``generator=``
#: cannot take are listed anyway: a caller asking "what are my options" must be
#: told they exist, and told in the same breath how each is actually reached.
#:
#: The ``gallery`` row names a class in ``ibnr.gallery`` as a STRING and imports
#: nothing - decision 8's re-export direction is gallery -> kernels only, and
#: this table is documentation, not a registry that constructs anything.
#: the zero_cells precondition every route shares; see _require_zero_cells_unused
_ZERO_CELLS_REQUIREMENT = (
    "; and, for a fit made with zero_cells='missing', no link ratio left out for a zero "
    "cell and no open origin whose latest amount is zero"
)
#: the estimator every route assumes; see _require_all_history_volume
_SETTINGS_REQUIREMENT = (
    "; and a fit with no development options: the volume average over every link ratio"
)

CDR_METHODS: dict[str, CDRMethod] = {
    "merz_wuthrich": CDRMethod(
        name="merz_wuthrich",
        route="analytic",
        generates="nothing - the factor update is linearized, not simulated",
        re_estimates="volume-weighted chain ladder (linearized)",
        returns="CDRResult (msep per origin and in total, no quantiles)",
        entry_point="one_year_cdr(fit)",
        requires=(
            "a MackFit on an annual development grain, with a strictly positive open diagonal"
            + _ZERO_CELLS_REQUIREMENT
            + _SETTINGS_REQUIREMENT
        ),
        validated=(
            "golden tie-out: R ChainLadder CDR(MackChainLadder(MW2014, "
            'est.sigma="Mack")) to 7 decimals, per origin and in total'
        ),
        reference="Merz & Wuthrich (2008), CAS E-Forum Fall 2008",
        generator=None,
        why_not_by_name=(
            "it is an analytic method, not a diagonal generator: it linearizes the "
            "chain-ladder factor update around Mack's conditional moments rather than "
            "simulating anything, so there is no version of it for another model"
        ),
    ),
    "mack": CDRMethod(
        name="mack",
        route="simulation",
        generates="Mack's conditional moments, E = f*C and Var = sigma^2*C",
        re_estimates="volume-weighted chain ladder (re-run on the extended triangle)",
        returns="PredictiveDistribution of the CDR (quantiles, VaR/TVaR)",
        entry_point='simulate_one_year_cdr(fit, generator="mack")',
        requires=(
            "a MackFit on an annual development grain, with a strictly positive open diagonal"
            + _ZERO_CELLS_REQUIREMENT
            + _SETTINGS_REQUIREMENT
        ),
        validated=(
            "agrees with the merz_wuthrich closed form to Monte Carlo error, "
            "which is itself tied out to R to 7 decimals"
        ),
        reference="Mack (1993) moments; 'actuary in the box' re-reserving",
        generator=MackDiagonal,
    ),
    "odp_bootstrap": CDRMethod(
        name="odp_bootstrap",
        route="simulation",
        generates="ODP Pearson-residual bootstrap + over-dispersed Poisson noise",
        re_estimates="volume-weighted chain ladder (re-run on the extended triangle)",
        returns="PredictiveDistribution of the CDR (quantiles, VaR/TVaR)",
        entry_point='simulate_one_year_cdr(fit, generator="odp_bootstrap")',
        requires=(
            "a MackFit on an annual development grain, whose observed increments "
            "are all non-negative" + _ZERO_CELLS_REQUIREMENT + _SETTINGS_REQUIREMENT
        ),
        validated=(
            "NO published digits exist - R's CDR.BootChainLadder example prints "
            "no output and a bootstrap is stochastic. Validated instead against "
            "R's ALGORITHM (a literal transcription of getNYCost reproduces the "
            "draws to 1e-10) and, for the standard error, against a delta-method "
            "reference computed off the re-reserving Jacobian, to Monte Carlo error"
        ),
        reference=(
            "England & Verrall (2002) sec. 8; R ChainLadder BootChainLadder / "
            "CDR.BootChainLadder (Crupi, Gesmann)"
        ),
        generator=ODPBootstrapDiagonal,
    ),
    "gallery": CDRMethod(
        name="gallery",
        route="simulation",
        generates=(
            "the posterior predictive of any fitted gallery entry with "
            "PredictsHeldout, at the next diagonal's cells"
        ),
        re_estimates="volume-weighted chain ladder (re-run on the extended triangle)",
        returns="PredictiveDistribution of the CDR (quantiles, VaR/TVaR)",
        entry_point=(
            "from ibnr.gallery import GalleryDiagonal; simulate_one_year_cdr("
            "fit, generator=GalleryDiagonal(entry, cells))"
        ),
        requires=(
            "a MackFit on an annual development grain, and a HoldoutCells "
            "describing the SAME cohort, field and cutoff, with one scorable cell "
            "for every open origin of the fit" + _ZERO_CELLS_REQUIREMENT + _SETTINGS_REQUIREMENT
        ),
        validated=(
            "the wiring, not the model: a generator that reproduces Mack's own "
            "conditional draws is shown to reproduce the mack route's CDR exactly, "
            "so the entry-to-diagonal path adds nothing of its own. What a given "
            "ENTRY's diagonal is worth is the leaderboard's CRPS question, not this one"
        ),
        reference="R ChainLadder's getNYCost structure with a third diagonal source",
        generator=None,
        why_not_by_name=(
            "it wraps a FITTED entry and the held-out cells it predicts at, and a name "
            "carries neither. Build it and pass the instance"
        ),
    ),
}


def cdr_methods() -> pd.DataFrame:
    """Every way to get a one-year CDR, and the terms of each.

    The answer to "which method is the one-year CDR, and what else could it
    be?": one row per route, with what generates next year's diagonal, how the
    reserve is re-estimated afterwards, what the call returns, what it requires
    of the cohort, and - honestly, per route - what it has actually been
    validated against::

        from ibnr.kernels.cdr import cdr_methods

        cdr_methods()[["name", "route", "generates", "returns"]]

    Names in the ``name`` column are what ``generator=`` accepts, except
    ``merz_wuthrich``, whose ``entry_point`` says where it lives instead.
    """
    return pd.DataFrame(
        [{k: v for k, v in vars(m).items() if k != "generator"} for m in CDR_METHODS.values()]
    )


def get_cdr_method(name: str) -> CDRMethod:
    """The :class:`CDRMethod` descriptor for one route, or a KeyError naming the
    known ones. Mirrors ``gallery.get``: what comes back describes the method
    and carries its generator **class**, so the caller constructs and configures
    it - ``get_cdr_method("odp_bootstrap").generator(process="gamma")``."""
    try:
        return CDR_METHODS[name]
    except KeyError:
        raise KeyError(
            f"no CDR method named {name!r}; known: {sorted(CDR_METHODS)}. "
            "cdr_methods() lists what each one requires and returns"
        ) from None


def _resolve_generator(
    generator: DiagonalGenerator | str | None,
    *,
    process: str | None,
    parameter_risk: bool | None,
) -> DiagonalGenerator:
    """``generator=`` plus the two legacy Mack knobs -> one generator instance.

    ``generator=None`` reproduces the 0.5.0 signature exactly, defaults and
    all, so every 0.5.0 call site keeps its numbers bit for bit. Supplying a
    generator AND a Mack knob is refused rather than silently ignored: the knob
    would otherwise be inert, which is the failure mode this repo has already
    shipped once and now tests for by name.
    """
    if generator is None:
        return MackDiagonal(
            process="gamma" if process is None else process,
            parameter_risk=True if parameter_risk is None else parameter_risk,
        )
    inert = [
        name
        for name, value in (("process", process), ("parameter_risk", parameter_risk))
        if value is not None
    ]
    if inert:
        given = getattr(generator, "name", generator)
        raise Refusal(
            "invalid_option",
            f"{', '.join(inert)} is a MackDiagonal setting and cannot be combined with "
            f"generator={_literal(repr(given))}; every generator carries its own knobs, so "
            "pass them to the generator itself (e.g. MackDiagonal(process=...)). Accepting them "
            "here would leave the argument inert whenever the generator is not mack",
            option="generator",
            options=("generator", *inert),
        )
    if isinstance(generator, DiagonalGenerator):
        return generator
    if isinstance(generator, str):
        if generator not in CDR_METHODS:
            # get_cdr_method answers a KeyError, as a lookup does; an argument
            # that names no method is a refused option
            raise Refusal(
                "invalid_option",
                f"no CDR method named {{given}}; known: {sorted(CDR_METHODS)}. "
                "cdr_methods() lists what each one requires and returns",
                option="generator",
                given=generator,
            )
        method = get_cdr_method(generator)
        if method.generator is None:
            raise Refusal(
                "invalid_option",
                f"{generator!r} cannot be named as a generator: {method.why_not_by_name}. "
                f"Use: {method.entry_point}",
                option="generator",
                given=generator,
            )
        return method.generator()
    raise Refusal(
        "invalid_option",
        "generator must be a DiagonalGenerator, a method name from cdr_methods(), or None; "
        f"got {type(generator).__name__}",
        option="generator",
    )


def simulate_one_year_cdr(
    fit: MackFit,
    *,
    n_draws: int | None = None,
    seed: int | np.random.SeedSequence | None = None,
    generator: DiagonalGenerator | str | None = None,
    process: str | None = None,
    parameter_risk: bool | None = None,
) -> PredictiveDistribution:
    """Actuary in the box: the one-year CDR distribution by re-reserving.

    One draw is one possible next year - :meth:`DiagonalGenerator.draw` says
    what emerged, :func:`rereserve` re-runs the volume-weighted chain ladder on
    the extended triangle and differences the ultimates. Returns per-origin CDR
    draws plus a ``total`` column derived from the same draws, so the
    diversification between accident years is in the samples; a positive draw is
    a reserve release.

    ``generator`` picks the first axis: a method name from :func:`cdr_methods`,
    or a configured instance::

        simulate_one_year_cdr(fit)                                  # Mack, defaults
        simulate_one_year_cdr(fit, generator="odp_bootstrap")
        simulate_one_year_cdr(fit, generator=ODPBootstrapDiagonal(process="gamma"))

    ``n_draws=None`` means "this generator's own count", which is
    :data:`DEFAULT_N_DRAWS` for the two simulating generators and the fitted
    posterior's own size for ``ibnr.gallery.GalleryDiagonal`` - see
    :meth:`DiagonalGenerator.resolve_n_draws`. It replaced a literal ``20_000``
    default in 0.5.1 and changes no number: both simulating generators resolve
    ``None`` to exactly that.

    ``process`` and ``parameter_risk`` are :class:`MackDiagonal`'s knobs, kept
    on this signature because they are the published 0.5.0 API - ``None`` means
    "that generator's default", so the defaults and every explicit call are
    unchanged. They cannot be combined with ``generator=``; pass them to the
    generator instead, and see :func:`_resolve_generator` for why.

    ``seed`` is anything ``np.random.default_rng`` accepts. The ``mack`` gallery
    entry hands a per-cohort ``SeedSequence`` through here; a plain integer keeps
    the byte-exact meaning it has always had.

    ``generator="merz_wuthrich"`` is refused by name: the closed form is a
    linearization of the chain-ladder factor update around Mack's moments, not a
    way of generating a diagonal, and :func:`one_year_cdr` is where it lives.
    ``generator="gallery"`` is refused too, for the opposite reason - it IS a
    generator, but it wraps a fitted entry that no string can carry.
    """
    gen = _resolve_generator(generator, process=process, parameter_risk=parameter_risk)
    if n_draws is not None and n_draws < 1:
        raise _refuse_n_draws(n_draws)
    # before the generator is consulted and before any draw: the grain is a
    # property of the fit, so no generator can make a non-annual step a year.
    _require_annual_step(fit)
    _require_zero_cells_unused(fit)
    _require_all_history_volume(fit)
    gen.check(fit)
    rng = np.random.default_rng(seed)
    cdr = rereserve(fit, gen.draw(fit, n_draws=gen.resolve_n_draws(n_draws), rng=rng))
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
    penalised for it rather than being silently re-centred.

    **That zero is Mack's.** ``E[CDR | D_I] = 0`` holds exactly under Mack's
    conditional moments, so on the ``mack`` generator this is the msep the
    Merz-Wuthrich closed form estimates. A residual bootstrap centres its
    diagonal on the pseudo-triangle's refit rather than on the chain-ladder
    projection, so its draws carry a small bias and this differs from their
    variance by its square. R draws the same distinction from the other side:
    ``CDR.MackChainLadder`` reports the analytic msep (about zero) and
    ``CDR.BootChainLadder`` reports ``sd()`` of the re-reserved amount (about
    its own mean). On a non-Mack generator, report both.

    On ``ibnr.gallery.GalleryDiagonal`` the gap can be large and is not noise:
    it is the entry's own view of next year's diagonal disagreeing with the
    chain ladder's. Quoting this number alone there reports a model's
    disagreement as though it were its volatility. ``cdr_risk_measures`` returns
    ``mean_cdr`` and ``sd_cdr`` side by side for exactly that reason."""
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
        raise Refusal(
            "invalid_option",
            "levels must lie strictly inside (0, 1), got {given}",
            option="levels",
            given=tuple(levels),
        )
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
