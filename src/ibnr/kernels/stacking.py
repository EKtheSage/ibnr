"""Model stacking over held-out forecast panels, via bayesblend.

The two-cutoff design (``forecast.py``'s module docstring states it; this module
implements it): weights are FITTED on the pointwise ELPD of an earlier panel and
APPLIED to forecasts at a later cutoff. A panel pins one ``as_of``, so
:func:`stack` reads two - the weights panel and the evaluation forecasts - and
refuses ``weights_panel.as_of >= evaluation.as_of``. It also requires every
outcome used to fit weights to have been observed by ``evaluation.as_of``:
different forecast cutoffs can still target the same unseen diagonal. Dates
are checked by cell key on the ELPD fitting population; later CRPS-only outcomes
do not influence weights. Outcomes observed exactly on the evaluation cutoff
are available, consistent with ``Triangle.as_of``.

What bayesblend gets, and what it never sees
--------------------------------------------

bayesblend is fed one number per (model, cell): the pointwise ELPD from
``panel.pointwise``, already reduced by :func:`forecast.logmeanexp`, wrapped via
``Draws.from_lpd`` so bayesblend never re-reduces raw draws. The matrix is
pivoted by the panel KEY in one canonical sorted order shared by every model -
never positionally. Two equal-length unlabelled vectors can agree with each
other perfectly and describe different cells; the sum is permutation-invariant,
so the fitted weights would be garbage while every total stayed plausible.

**Relative, then floored.** bayesblend never sees the pointwise ELPD itself.
Each cell's values are passed RELATIVE to that cell's best finite member and
then floored at :data:`LPD_FLOOR` = -700.0, so every number bayesblend receives
lies in ``[-700, 0]`` with each cell's best member at exactly 0. There are two
reasons for that, and only the second one is about ``-inf``.

``MleStacking`` works in linear space: it exponentiates, minimizes
``-sum(log(Y @ w))`` and hands SLSQP the Jacobian ``1 / (Y @ w)``. The boundary
is NOT ``exp`` underflowing to zero, which takes about -746; it is that
reciprocal, which overflows to infinity as soon as ``Y @ w`` falls below
``1 / DBL_MAX``, about 5.6e-309, whose log is about -709.78. Where that lands in
the log densities themselves depends on the weights and on how far apart the
members are, because ``Y @ w`` is a weighted mixture and so sits below its
largest term: on the 16 board-like cells measured here (two members three nats
apart, at the uniform starting weights) it is reached between -709.0 and -709.4.
SLSQP then stops at iteration 1 with ``success = False``, and the vector it
stopped at is the uniform one it started from: a valid simplex and no fit at
all. Measured on those cells: one cell at -710 shared by every member turned a
1.0/0.0 fit into 0.5/0.5, with the other fifteen cells at the usual -9, and
0.5/0.5 scores 10.3 nats below the answer the fit should have given. A common
per-cell offset cannot move the optimum (it scales the mixture density at that
cell by the same factor for every weight vector), so subtracting each cell's
best finite member changes no correct answer and takes the matrix out of that
range whatever the absolute level of the densities.

A pointwise ELPD of ``-inf`` is a legitimate verdict on this board ("the model
gave the outcome zero density"), but bayesblend's own lpd reduction is the
textbook max shift, which turns ``-inf`` into NaN - measured in this
environment: ``Draws.from_lpd([-10, -inf, -9]).lpd`` comes back as
``[-10, nan, -9]``. That NaN would flow into the SLSQP objective and the weights
would come back NaN. The same floor answers it: ``exp(-700)`` is ~1e-304, a
weight contribution indistinguishable from zero, while staying a normal float -
no NaN, and no exact 0.0 row to make ``log(Y @ w)`` blow up when every member
missed the same cell.
The count of ``-inf`` entries is carried on the result (``n_floored_neg_inf``)
so a run where that verdict was reached is visible; a finite member clipped by
the same floor is NOT counted, because 700 nats behind the cell's best is
already zero at double precision and clipping it changes no weight.

The order is what makes the ranking right: each cell's maximum over its FINITE
members first, the floor second. Flooring first puts a member that gave zero
density (floored to -700) ABOVE a finite member 800 nats back, which is the two
verdicts the wrong way round. A cell no member covered has no finite maximum and
uses 0.0 as its reference, so it floors to one constant across members and adds
the same number to the objective for every weight vector - which is what "every
member missed it" has to mean for weight fitting. Do not write the subtraction
as ``matrix - matrix.max(axis=0)``: on such a cell that is ``-inf - -inf``,
i.e. NaN, which is the one thing the floor exists to prevent.

**The optimizer's own verdict is read.** A failed SLSQP solve is not an
exception in bayesblend: it stores the failed ``OptimizeResult`` and hands back
whatever vector SLSQP stopped at, which is finite, non-negative and sums to 1
whatever the status, so no check on the vector itself can tell it from a fitted
answer. At the boundary above that vector is the uniform starting point, because
SLSQP stops at its first iteration; a solve that ran out of iterations instead
would hand back a partly fitted vector, equally unmarked. ``_fit_weights``
refuses either by name, carrying scipy's status and message.

Methods: ``"mle"`` (default, ``MleStacking`` - pure scipy, no cmdstan),
``"pseudo_bma"`` (``PseudoBma``, numpy/scipy), ``"bayes"`` (``BayesStacking``)
and ``"hierarchical"`` (``HierarchicalBayesStacking``) - the last two compile
Stan, so their tests live behind ``-m slow``. Hierarchical stacking fits
per-cell weights against a ``dev_lag`` covariate; this module applies the
cell-AVERAGED posterior-mean weight, so the covariate structure informs the fit
but the applied weight is global - one weight per model is what the stacked
pseudo-model below is defined over. Note WHAT is averaged over: the WEIGHTS
panel's cells, whose development mix is systematically shallower than the
evaluation panel's (the earlier cutoff never contains a dev-120 cell - see
``forecast.py``'s panel geometry), so the deepest evaluation cells influence
the applied weight only through the fitted covariate slope, never the average
itself. bayesblend is imported lazily inside the
fitting function: its import pulls cmdstanpy and arviz at module scope, and the
core install has neither.

The stacked pseudo-model
------------------------

One model set, one weight vector, both arms. The weights are fitted over the
ELPD members of the weights panel, and the stacked model's CRPS arm pools the
SAME members with the SAME weights - draws-only entries (england_verrall_odp,
mack) are NOT in the stack, because there is no density-fitted weight for them
(design choice (a); the alternative, a second draws-only pseudo-model with
CRPS-fitted weights, is a different estimator and a separate decision).

**ELPD arm.** The mixture's cell density is ``sum_m w_m * p_m(y)``, whose log
under the existing draw reduction is exactly a row concatenation: with member
arrays ``ll_m`` of shape ``(S_m, n_cells)`` and ``S = sum(S_m)``,

    ``logmeanexp(vstack([ll_m + log(w_m * S / S_m)])) ==
    logsumexp(log(w_m) + logmeanexp(ll_m), over m)``

verified to 1.8e-15. So the stacked forecast is a plain
:class:`~ibnr.kernels.forecast.CohortForecast` whose ``log_density`` is the
members' arrays stacked with an additive offset, and ``pointwise_elpd()``
returns the correct stacked score with no new reduction primitive. Zero-weight
members are DROPPED from the stack, not offset by ``log(0)``: a ``-inf`` block
contributes nothing to the mixture but poisons the constructor's zero-variance
check with NaN.

**CRPS arm.** ``scores.crps`` has no weight argument - the estimator is
unweighted over rows - so mixture weights enter through ROW COUNTS. The pooled
draw matrix has ``target = min_m(S_m) * n_members`` rows, apportioned to members
by largest remainder (ties broken by model name), each member contributing
evenly spaced rows of its own array. Deliberately NOT bayesblend's ``_blend``,
which resamples members stochastically per datapoint: an unseeded blend makes
the stacked CRPS irreproducible run to run, and a seeded one still cannot be
reproduced from the weights alone. Largest remainder + even spacing is
deterministic twice over - same inputs, same rows, same order. Weight
resolution in this arm is ``1/target``: a member whose weight is positive but
small enough to win no seat contributes ZERO rows to the stacked draws while
still contributing its (tiny) share to the stacked density.

The stacked forecasts are ordinary ``CohortForecast`` objects named
``stacked_<method>``. They do not get a private scoring path: pass them INTO
:func:`~ibnr.kernels.forecast.align_panel` beside the base models and the
leaderboard produces their row under exactly the guards everyone else passes. A
member that refused a cohort at evaluation makes the stacked cohort an
``Absence`` (``scoring_refused``) on that arm - the mixture is defined over all
its members or not at all.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from ibnr.kernels.forecast import Absence, CohortForecast, ForecastPanel

__all__ = [
    "LPD_FLOOR",
    "METHODS",
    "StackingResult",
    "apply_weights",
    "stack",
]

#: How far below its cell's best finite member a pointwise ELPD may reach before
#: bayesblend sees it. ``exp(-700)`` is ~1e-304: still a normal positive float
#: (no exact-zero row in MLE's ``log(Y @ w)``), yet a weight contribution
#: indistinguishable from 0. It has to stay above ``log`` of the smallest normal
#: double, -708.40, so that ``exp(LPD_FLOOR)`` is a normal float and the
#: Jacobian's ``1 / (Y @ w)`` stays finite; the overflow itself starts once
#: ``Y @ w`` falls below ``1 / DBL_MAX``, whose log is about -709.78. ``exp``
#: does not reach zero until about -746, which is the wrong boundary and about
#: 37 nats too late. The floor catches ``-inf`` too, which bayesblend's own
#: max-shift lpd reduction would otherwise turn into NaN.
LPD_FLOOR: float = -700.0

#: method name -> the bayesblend fitter it resolves to (resolved lazily inside
#: ``_fit_weights``; importing bayesblend pulls cmdstanpy + arviz, which the
#: core install does not have).
METHODS: dict[str, str] = {
    "mle": "MleStacking",
    "pseudo_bma": "PseudoBma",
    "bayes": "BayesStacking",
    "hierarchical": "HierarchicalBayesStacking",
}

#: Weights must sum to 1 within this before they are applied. Looser than float
#: noise would need, tighter than any real fitting failure.
_WEIGHT_TOL = 1e-6


@dataclass(frozen=True)
class StackingResult:
    """Fitted weights, their provenance, and the stacked forecasts.

    ``weights``            model -> weight, summing to 1 (checked at
                           construction). The full fitted vector, INCLUDING any
                           zero weights - a dropped member is still a verdict.
    ``method``             key into :data:`METHODS`.
    ``n_cells_weight_fit`` cells the weights were fitted on (the weight panel's
                           ELPD panel size).
    ``n_floored_neg_inf``  (model, cell) entries whose pointwise ELPD was
                           ``-inf``. Nonzero means some member gave some
                           outcome the weights were fitted on zero density. A
                           finite member clipped by :data:`LPD_FLOOR` is not
                           counted here: 700 nats behind its cell's best is
                           already zero at double precision, so the clip is
                           numerics rather than a verdict.
    ``weights_as_of``      the weight panel's cutoff - strictly earlier than
                           the evaluation cutoff, by construction.
    ``weights_fingerprint`` the weight panel's ``elpd_fingerprint``, so the cell
                           set behind the weights is checkable mechanically.
    ``forecasts``          the stacked pseudo-model's ``CohortForecast`` per
                           evaluation cohort, ready for ``align_panel``.
    """

    method: str
    weights: dict[str, float]
    n_cells_weight_fit: int
    n_floored_neg_inf: int
    weights_as_of: dt.date
    weights_fingerprint: str
    forecasts: tuple[CohortForecast, ...]

    def __post_init__(self) -> None:
        _check_weights(self.weights)
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}, got {self.method!r}")

    @property
    def model(self) -> str:
        """The stacked pseudo-model's name on the board."""
        return f"stacked_{self.method}"


def _check_weights(weights: Mapping[str, float]) -> None:
    if not weights:
        raise ValueError("weights are empty; a stack over no models is not a model")
    for name, w in weights.items():
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(
                f"weight for {name!r} is {w!r}; stacking weights are finite and non-negative. "
                "NaN here usually means a -inf pointwise ELPD reached bayesblend unfloored"
            )
    total = float(sum(weights.values()))
    if abs(total - 1.0) > _WEIGHT_TOL:
        raise ValueError(
            f"weights sum to {total!r}, not 1. A non-simplex weight vector silently rescales "
            "every stacked density by the same factor, which shifts the stacked ELPD by "
            "log(sum w) per cell and still looks entirely plausible"
        )


def stack(
    weights_panel: ForecastPanel,
    evaluation: Iterable[CohortForecast],
    *,
    method: str = "mle",
    seed: int | None = None,
) -> StackingResult:
    """Fit stacking weights on one panel, build stacked forecasts at a later one.

    ``weights_panel`` is the ALIGNED earlier panel (its ``pointwise`` frame is
    the fitting data); ``evaluation`` is the later cutoff's forecasts as the raw
    ``CohortForecast`` objects - a ``ForecastPanel`` will not do here, because
    the stacked pseudo-model is built from the members' ``(n_draws, n_cells)``
    arrays and a panel only retains their reductions.

    Refused, each with the reason it must be:

    * ``weights_panel.as_of >= evaluation as_of`` - weights graded on the cells
      that chose them measure selection, not skill;
    * any weight-fitting outcome observed after the evaluation cutoff, or
      missing/ambiguous availability metadata for a used cell. Outcome dates
      equal to the evaluation cutoff are allowed. CRPS-only cells are not used
      to fit weights and do not enter this check;
    * mixed ``as_of``/``task``/segment schema/measure among the evaluation
      forecasts, and any disagreement of those with the weights panel. What is
      checked WHERE: those four panel-identity checks happen here;
      :func:`apply_weights` checks per-cohort member agreement (task, field,
      cell keys, observed values, ``eval_date``, ``train_origins``); the
      remaining panel checks - premium agreement, the upstream exclusion
      censuses, cross-cohort duplicates - happen when the stacked forecasts are
      aligned WITH their members, which is the only supported way to score
      them;
    * a member-set mismatch: the models offering a density at evaluation must
      be EXACTLY the weight panel's ELPD members. A weight vector fitted over
      one member set cannot be applied to another silently - a missing member
      leaves its weight stranded, an extra one has no weight at all. Evaluation
      forecasts from draws-only models are ignored (they are not in the stack;
      see the module docstring);
    * fewer than two ELPD members - there is nothing to weight;
    * an ``mle`` solve whose scipy result reports ``success = False``
      (``RuntimeError``, carrying scipy's status and message): bayesblend
      returns the vector SLSQP stopped at rather than raising, and that vector
      passes every check on the weights themselves.

    ``seed`` reaches the fitters that take one (``pseudo_bma``, ``bayes``,
    ``hierarchical``); ``mle`` is deterministic and ignores it.
    """
    if not isinstance(weights_panel, ForecastPanel):
        raise TypeError(
            f"weights_panel must be a ForecastPanel built by align_panel, got "
            f"{type(weights_panel)}. The aligned panel is what guarantees the pointwise "
            "ELPDs are on one identical cell set before anything is fitted to them"
        )
    if isinstance(evaluation, ForecastPanel):
        raise TypeError(
            "evaluation must be the CohortForecast objects themselves, not a ForecastPanel: "
            "a panel carries reduced pointwise scores, and the stacked pseudo-model is built "
            "from the members' (n_draws, n_cells) arrays, which only the forecasts retain"
        )
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    items = list(evaluation)
    if not items:
        raise ValueError("stack needs at least one evaluation CohortForecast")

    eval_as_of = _one("as_of", {f.as_of for f in items})
    eval_task = _one("task", {f.task for f in items})
    eval_segments = _one("segment schema", {tuple(f.cells.segments) for f in items})
    eval_measure = _one("triangle measure", {f.measure for f in items})

    if not weights_panel.as_of < eval_as_of:
        raise ValueError(
            f"weights_panel.as_of ({weights_panel.as_of}) must precede the evaluation as_of "
            f"({eval_as_of}). Weights fitted and applied at the same cutoff are graded on "
            "the cells that chose them; two cutoffs is the design, in that order"
        )
    if weights_panel.task != eval_task:
        raise ValueError(
            f"task mismatch: weights fitted on {weights_panel.task!r}, evaluation is "
            f"{eval_task!r}. The paid and reported boards stay separate, and so do their "
            "stacks"
        )
    if tuple(weights_panel.segments) != eval_segments:
        raise ValueError(
            f"segment schema mismatch: weights panel keys cohorts by "
            f"{tuple(weights_panel.segments)}, evaluation by {eval_segments}. A weight "
            "fitted over cohorts keyed one way cannot be applied over cohorts keyed another"
        )
    panel_measure = _one("triangle measure", set(weights_panel.cells["measure"]))
    if panel_measure != eval_measure:
        raise ValueError(
            f"triangle measure mismatch: weights panel is {panel_measure!r}, evaluation is "
            f"{eval_measure!r}. The observed values are different numbers on the two bases"
        )

    members = tuple(weights_panel.elpd_members)
    if len(members) < 2:
        raise ValueError(
            f"the weights panel has {len(members)} ELPD member(s) ({list(members)}); "
            "stacking needs at least two. A one-model stack is that model with extra steps"
        )
    eval_density_models = {f.model for f in items if f.has_density}
    eval_density_models |= {
        f.model
        for f in items
        if f.density_absence is not None and not f.density_absence.is_model_level
    }
    if eval_density_models != set(members):
        raise ValueError(
            f"member-set mismatch: the weights were fitted over {sorted(members)} but the "
            f"evaluation forecasts offer a density for {sorted(eval_density_models)}. A "
            "weight vector fitted over one member set cannot be applied to another - a "
            "missing member strands its weight, an extra one has none. Refit the weights "
            "over the set you mean to stack"
        )

    lpd, dev_lag, n_floored = _lpd_matrix(weights_panel, evaluation_as_of=eval_as_of)
    weights = _fit_weights(lpd, method=method, seed=seed, dev_lag=dev_lag)
    _check_weights(weights)
    # Renormalize the float residue so the applied vector is exactly a simplex;
    # _check_weights above has already refused anything actually broken.
    total = float(sum(weights.values()))
    weights = {m: w / total for m, w in weights.items()}

    stacked = apply_weights(
        weights,
        [f for f in items if f.model in weights],
        model=f"stacked_{method}",
    )
    return StackingResult(
        method=method,
        weights=weights,
        n_cells_weight_fit=weights_panel.n_cells_for("elpd"),
        n_floored_neg_inf=n_floored,
        weights_as_of=weights_panel.as_of,
        weights_fingerprint=weights_panel.elpd_fingerprint,
        forecasts=tuple(stacked),
    )


def _one(name: str, values: set):
    """The single value, or a refusal naming the mix."""
    if len(values) > 1:
        raise ValueError(
            f"evaluation forecasts disagree on {name}: {sorted(values, key=repr)}. "
            "One stack reads one panel's worth of forecasts; align_panel would refuse "
            "this mix too"
        )
    return next(iter(values))


def _lpd_matrix(
    panel: ForecastPanel, *, evaluation_as_of: dt.date
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Per member, the pointwise ELPD over the ELPD cells, relative and floored.

    Every model's vector is sorted by the SAME canonical key order (the panel's
    key columns), and the key sequences are checked identical across members -
    the pivot is by key, never by position. The values are then handed to
    :func:`_relative_lpd`, which is what bayesblend actually receives. Returns
    ``(lpd_by_model, dev_lag, n_floored)`` with ``dev_lag`` in the same
    canonical order (the hierarchical method's covariate).
    """
    pointwise = panel.pointwise
    live = pointwise[pointwise["on_elpd_panel"] & pointwise["model"].isin(panel.elpd_members)]
    # Match availability to the exact rows about to enter the optimizer. The
    # cell table also carries CRPS-only outcomes, which need not be available.
    # Check keys and missing dates explicitly: max() would skip unknown dates,
    # and a stored panel's mutable frames may have lost some of their metadata.
    used_keys = live["key"].drop_duplicates()
    metadata = panel.cells.loc[panel.cells["key"].isin(used_keys), ["key", "eval_date"]]
    if metadata["key"].duplicated().any():
        raise ValueError(
            "weight-fitting outcomes have ambiguous eval_date metadata: duplicate keys"
        )
    dates = metadata.set_index("key")["eval_date"].reindex(used_keys)
    if dates.isna().any():
        raise ValueError(
            "weight-fitting outcomes have missing eval_date metadata; every used cell "
            "must have a known observation date before weights can be fitted"
        )
    later = dates[dates > evaluation_as_of]
    if not later.empty:
        raise ValueError(
            f"{len(later)} weight-fitting outcomes were observed as late as {later.max()}, "
            f"after the evaluation as_of ({evaluation_as_of}). Fit weights only on "
            "outcomes available on or before the evaluation cutoff"
        )
    out: dict[str, np.ndarray] = {}
    reference_keys: list | None = None
    dev_lag = np.empty(0)
    for model in panel.elpd_members:
        mine = live[live["model"] == model].sort_values(panel.key_columns, kind="stable")
        keys = mine["key"].tolist()
        if reference_keys is None:
            reference_keys = keys
            dev_lag = mine["dev_lag"].to_numpy(dtype=float)
        elif keys != reference_keys:
            raise ValueError(
                f"{model} covers different ELPD-panel keys than {panel.elpd_members[0]}; "
                "the panel invariant (one identical cell set per score) does not hold, so "
                "an lpd matrix built from it would align rows across different cells"
            )
        values = mine["elpd"].to_numpy(dtype=float)
        # -inf is the one non-finite verdict this board has; NaN and +inf are
        # both bugs upstream, and letting either through hands the failure to
        # SLSQP, which then reports a convergence problem instead of the input.
        bad = ~(np.isfinite(values) | np.isneginf(values))
        if bad.any():
            raise ValueError(
                f"{model} has {int(bad.sum())} NaN or +inf pointwise ELPD value(s) among "
                "the cells the weights are fitted on; both are bugs upstream (-inf is the "
                "legitimate zero-density verdict), find it rather than fit weights around it"
            )
        out[model] = values
    relative, n_floored = _relative_lpd(out)
    return relative, dev_lag, n_floored


def _relative_lpd(lpd: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    """Each cell's pointwise ELPD relative to that cell's best finite member.

    The returned values are in ``[LPD_FLOOR, 0]``, with each cell's best member
    at exactly 0, whatever the absolute level of the densities was - which holds
    because every value reaching here is finite or ``-inf``, refused by name in
    :func:`_lpd_matrix` otherwise. That is what keeps ``MleStacking``'s
    linear-space arithmetic away from the level where its Jacobian overflows and
    SLSQP hands back a vector it never fitted; the
    optimum itself is unchanged, because a common per-cell offset scales the
    mixture density at that cell by the same factor for every weight vector. The
    module docstring has the boundary and the reason the order matters.

    Also returns the count of ``-inf`` entries, which is the zero-density
    verdict and goes on the record. Finite entries clipped by the same floor are
    not counted: they are already zero at double precision.
    """
    names = list(lpd)
    matrix = np.vstack([lpd[name] for name in names])
    finite = np.isfinite(matrix)
    n_neg_inf = int(np.isneginf(matrix).sum())
    # A cell with no finite member has no reference to be relative to, so it
    # takes 0.0 and floors to one constant across members: uninformative, which
    # is what "every member missed it" means here. Do NOT let -inf into the
    # maximum, in either direction: including it makes an all-missed cell NaN,
    # and flooring before the maximum ranks a zero-density member above a finite
    # one further back.
    best = np.where(
        finite.any(axis=0),
        np.max(np.where(finite, matrix, -np.inf), axis=0),
        0.0,
    )
    relative = np.maximum(matrix - best, LPD_FLOOR)
    return {name: relative[i] for i, name in enumerate(names)}, n_neg_inf


def _fit_weights(
    lpd: dict[str, np.ndarray],
    *,
    method: str,
    seed: int | None,
    dev_lag: np.ndarray,
) -> dict[str, float]:
    """One scalar weight per model, from the chosen bayesblend fitter.

    bayesblend is imported HERE, not at module scope: it pulls cmdstanpy and
    arviz on import, and the core install has neither.
    """
    import bayesblend

    draws = {model: bayesblend.Draws.from_lpd(values) for model, values in lpd.items()}
    if method == "mle":
        fitted = bayesblend.MleStacking(draws).fit()
        # bayesblend does not raise on a failed solve: it keeps the failed
        # OptimizeResult and hands back whatever vector SLSQP stopped at, which
        # is finite, non-negative and sums to 1 whatever the status, so no check
        # on the vector can tell it from a fitted answer.
        solve = fitted.model_info
        if not solve.success:
            n_cells = next(iter(lpd.values())).size
            raise RuntimeError(
                f"stacking {len(lpd)} members over {n_cells} cells: MleStacking's SLSQP "
                f"solve did not converge (status {solve.status}: {solve.message!r}, "
                f"objective {float(solve.fun):g} after {solve.nit} iteration(s)). The "
                "weights bayesblend hands back are the vector SLSQP stopped at - its "
                "uniform starting point when it stops at the first iteration - which is a "
                "valid simplex and no fit at all. Every value it was given lies in "
                f"[{LPD_FLOOR}, 0], so the numerical range that used to cause this is "
                "ruled out; read those members' pointwise ELPD over the cells the weights "
                "are fitted on before trying another method"
            )
    elif method == "pseudo_bma":
        fitted = bayesblend.PseudoBma(draws, seed=seed).fit()
    elif method == "bayes":
        fitted = bayesblend.BayesStacking(draws, seed=seed).fit()
    elif method == "hierarchical":
        fitted = bayesblend.HierarchicalBayesStacking(
            draws,
            continuous_covariates={"dev_lag": [float(x) for x in dev_lag]},
            seed=seed,
        ).fit()
    else:  # pragma: no cover - stack() has already validated
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    # ``weights`` is the posterior-mean weight per model: (1, 1) for the global
    # methods, (1, n_cells) for hierarchical - the mean over cells makes the
    # applied weight global either way (documented in the module docstring).
    return {model: float(np.asarray(w).mean()) for model, w in fitted.weights.items()}


def apply_weights(
    weights: Mapping[str, float],
    evaluation: Iterable[CohortForecast],
    *,
    model: str,
) -> list[CohortForecast]:
    """Build the stacked pseudo-model's forecast for every evaluation cohort.

    The convenience half of :func:`stack`, public so a weight vector can be
    re-applied without refitting. ``weights`` must be a simplex over the member
    models; ``evaluation`` supplies each member's ``CohortForecast`` per cohort
    (forecasts from models outside ``weights`` are ignored). Returns one
    ``CohortForecast`` named ``model`` per cohort, to be passed into
    ``align_panel`` beside the members - there is no private scoring path.

    Zero-weight members are dropped from BOTH arms before building: a
    ``log(0)`` offset row-block is ``-inf`` everywhere, which contributes
    nothing to the mixture but poisons the constructor's zero-variance check;
    and a zero weight apportions zero draw rows anyway. A KEPT member missing a
    cohort (or refusing an arm there) makes the stacked cohort an ``Absence``
    on that arm - the mixture is defined over all its members or not at all.

    Per cohort, the member forecasts must AGREE on task, field, cell keys,
    observed values, ``eval_date`` and ``train_origins``, and a disagreement
    raises. The stacked forecast is anchored on one member's ``cells``, so
    without these checks two members disagreeing on an observed value would be
    silently resolved in favour of whichever came FIRST in the list - an
    order-dependent answer, invisible to an ``align_panel`` over the stacked
    forecasts alone, because the losing member's value never reaches the panel. What is
    NOT checked here: premium agreement and the upstream exclusion censuses,
    which ``align_panel`` checks when the stacked forecasts are aligned WITH
    their members (the only supported way to score them), and the triangle
    measure, which :func:`stack` checks against the weights panel.
    """
    _check_weights(weights)
    kept = [m for m in sorted(weights) if weights[m] > 0.0]
    items = list(evaluation)

    by_cohort: dict[tuple, dict[str, CohortForecast]] = {}
    order: list[tuple] = []
    for f in items:
        if f.model not in weights:
            continue
        mine = by_cohort.setdefault(f.cohort, {})
        if f.model in mine:
            raise ValueError(
                f"{f.model} appears twice for cohort {f.cohort}; a duplicated forecast "
                "would enter the mixture twice. align_panel refuses the same thing"
            )
        if f.cohort not in order:
            order.append(f.cohort)
        mine[f.model] = f
    if not by_cohort:
        raise ValueError(
            f"no evaluation forecasts from the weighted models {sorted(weights)}; nothing to stack"
        )

    out: list[CohortForecast] = []
    for cohort in order:
        present = by_cohort[cohort]
        _agree(cohort, present, "task", lambda f: f.task)
        _agree(cohort, present, "field", lambda f: f.field)
        _agree(cohort, present, "cell keys", lambda f: tuple(f.keys))
        # The three checks the anchor would otherwise launder: the stacked
        # forecast carries ONE member's cells, so a disagreement here would be
        # resolved in favour of whichever member came first in the list, and an
        # align_panel over the stacked forecasts alone would never see it.
        _agree(cohort, present, "observed value", lambda f: tuple(f.key_frame["value"].tolist()))
        _agree(cohort, present, "eval_date", lambda f: f.eval_date)
        _agree(cohort, present, "train_origins", lambda f: f.cells.train_origins)
        anchor = next(iter(present.values()))

        missing = [m for m in kept if m not in present]
        if missing:
            absence = Absence(
                "scoring_refused",
                f"stacked member(s) {missing} have no forecast for this cohort",
            )
            out.append(
                CohortForecast(
                    model=model,
                    task=anchor.task,
                    cells=anchor.cells,
                    field=anchor.field,
                    density_absence=absence,
                    draws_absence=absence,
                )
            )
            continue

        no_density = [m for m in kept if not present[m].has_density]
        if no_density:
            log_density = None
            density_absence = Absence(
                "scoring_refused",
                f"stacked member(s) {no_density} offer no density for this cohort: "
                + "; ".join(f"{m}: {present[m].density_absence}" for m in no_density),
            )
        else:
            log_density = _stacked_log_density(weights, {m: present[m] for m in kept})
            density_absence = None

        no_draws = [m for m in kept if not present[m].has_draws]
        if no_draws:
            draws = None
            draws_absence = Absence(
                "scoring_refused",
                f"stacked member(s) {no_draws} offer no draws for this cohort: "
                + "; ".join(f"{m}: {present[m].draws_absence}" for m in no_draws),
            )
        else:
            draws = _pooled_draws(weights, {m: present[m] for m in kept})
            draws_absence = None

        out.append(
            CohortForecast(
                model=model,
                task=anchor.task,
                cells=anchor.cells,
                field=anchor.field,
                log_density=log_density,
                draws=draws,
                density_absence=density_absence,
                draws_absence=draws_absence,
            )
        )
    return out


def _agree(cohort: tuple, present: Mapping[str, CohortForecast], name: str, get) -> None:
    values = {get(f) for f in present.values()}
    if len(values) > 1:
        raise ValueError(
            f"the member forecasts for cohort {cohort} disagree on {name}; they do not "
            "describe the same cells, so no mixture over them is defined. align_panel "
            "would refuse them too"
        )


def _stacked_log_density(
    weights: Mapping[str, float], members: Mapping[str, CohortForecast]
) -> np.ndarray:
    """Row-concatenate the members' arrays with the ``log(w * S / S_m)`` offset.

    ``logmeanexp`` of the result equals ``logsumexp(log(w_m) + lpd_m)`` - the
    stacked mixture's pointwise log density - verified to 1.8e-15, so no new
    reduction primitive exists for a stacked forecast to disagree with.
    """
    names = sorted(members)
    sizes = {m: members[m].log_density.shape[0] for m in names}
    total = sum(sizes.values())
    blocks = [members[m].log_density + np.log(weights[m] * total / sizes[m]) for m in names]
    return np.vstack(blocks)


def _pooled_draws(
    weights: Mapping[str, float], members: Mapping[str, CohortForecast]
) -> np.ndarray:
    """Weight-proportional deterministic pooling of the members' draws.

    Target row count = smallest member draw count x member count; rows are
    apportioned by largest remainder and each member contributes evenly spaced
    rows of its own array (cycling, deterministically, if allotted more rows
    than it has). Never bayesblend's ``_blend``, which resamples stochastically
    per datapoint.
    """
    names = sorted(members)
    target = min(members[m].draws.shape[0] for m in names) * len(names)
    counts = _largest_remainder({m: weights[m] for m in names}, target)
    blocks = []
    for m in names:
        n = counts[m]
        if n == 0:
            continue
        rows = members[m].draws
        idx = np.floor(np.arange(n) * (rows.shape[0] / n)).astype(int)
        blocks.append(rows[idx])
    return np.vstack(blocks)


def _largest_remainder(weights: Mapping[str, float], target: int) -> dict[str, int]:
    """Apportion ``target`` seats to ``weights`` (a simplex), deterministically.

    Floor of each quota first, then one extra seat per largest fractional
    remainder; remainder ties break by model name so two runs cannot disagree.
    Exactly ``target`` seats come back - the property ``round()`` per member
    does not have.
    """
    total = float(sum(weights.values()))
    quotas = {m: target * w / total for m, w in weights.items()}
    seats = {m: int(math.floor(q)) for m, q in quotas.items()}
    leftover = target - sum(seats.values())
    for m in sorted(quotas, key=lambda m: (-(quotas[m] - seats[m]), m))[:leftover]:
        seats[m] += 1
    return seats
