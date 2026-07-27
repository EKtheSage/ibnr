"""Model stacking over held-out forecast panels, via bayesblend.

The two-cutoff design (``forecast.py``'s module docstring states it; this module
implements it): weights are FITTED on the pointwise ELPD of an earlier panel and
APPLIED to forecasts at a later cutoff. A panel pins one ``as_of``, so
:func:`stack` reads two - the weights panel and the evaluation forecasts - and
refuses ``weights_panel.as_of >= evaluation.as_of``, the guard the
``ForecastPanel`` docstring promises. Fitting and scoring the same cutoff would
grade the weights on the cells that chose them.

What bayesblend gets, and what it never sees
--------------------------------------------

bayesblend is fed one number per (model, cell): the pointwise ELPD from
``panel.pointwise``, already reduced by :func:`forecast.logmeanexp`, wrapped via
``Draws.from_lpd`` so bayesblend never re-reduces raw draws. The matrix is
pivoted by the panel KEY in one canonical sorted order shared by every model -
never positionally. Two equal-length unlabelled vectors can agree with each
other perfectly and describe different cells; the sum is permutation-invariant,
so the fitted weights would be garbage while every total stayed plausible.

**The ``-inf`` floor.** A pointwise ELPD of ``-inf`` is a legitimate verdict on
this board ("the model gave the outcome zero density"), but bayesblend's own lpd
reduction is the textbook max shift, which turns ``-inf`` into NaN - measured in
this environment: ``Draws.from_lpd([-10, -inf, -9]).lpd`` -> ``[-10, nan, -9]``.
That NaN then flows into the SLSQP objective and the weights come back NaN. So
``-inf`` is floored at :data:`LPD_FLOOR` = -700.0 BEFORE bayesblend:
``exp(-700)`` is ~1e-304, a weight contribution indistinguishable from zero,
while staying a normal float - no NaN, and no exact 0.0 row to make MLE's
``log(Y @ w)`` blow up when every member missed the same cell. The floored cell
count is carried on the result (``n_floored_neg_inf``) so a run where the floor
did real work is visible.

Methods: ``"mle"`` (default, ``MleStacking`` - pure scipy, no cmdstan),
``"pseudo_bma"`` (``PseudoBma``, numpy/scipy), ``"bayes"`` (``BayesStacking``)
and ``"hierarchical"`` (``HierarchicalBayesStacking``) - the last two compile
Stan, so their tests live behind ``-m slow``. Hierarchical stacking fits
per-cell weights against a ``dev_lag`` covariate; this module applies the
cell-AVERAGED posterior-mean weight, so the covariate structure informs the fit
but the applied weight is global - one weight per model is what the stacked
pseudo-model below is defined over. bayesblend is imported lazily inside the
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
deterministic twice over - same inputs, same rows, same order.

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

#: Where a pointwise ELPD of ``-inf`` is floored before bayesblend sees it.
#: ``exp(-700)`` is ~1e-304: still a normal positive float (no exact-zero row in
#: MLE's ``log(Y @ w)``), yet a weight contribution indistinguishable from 0.
#: bayesblend's own max-shift lpd reduction turns a raw ``-inf`` into NaN.
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
    ``n_floored_neg_inf``  (model, cell) lpd entries floored at
                           :data:`LPD_FLOOR`. Nonzero means some member gave
                           some weight-panel outcome zero density.
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
    * mixed ``as_of``/``task``/segment schema/measure among the evaluation
      forecasts, and any disagreement of those with the weights panel - the
      same identity checks ``align_panel`` applies, one cutoff earlier;
    * a member-set mismatch: the models offering a density at evaluation must
      be EXACTLY the weight panel's ELPD members. A weight vector fitted over
      one member set cannot be applied to another silently - a missing member
      leaves its weight stranded, an extra one has no weight at all. Evaluation
      forecasts from draws-only models are ignored (they are not in the stack;
      see the module docstring);
    * fewer than two ELPD members - there is nothing to weight.

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

    lpd, dev_lag, n_floored = _lpd_matrix(weights_panel)
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


def _lpd_matrix(panel: ForecastPanel) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Per member, the pointwise ELPD over the panel's ELPD cells, floored.

    Every model's vector is sorted by the SAME canonical key order (the panel's
    key columns), and the key sequences are checked identical across members -
    the pivot is by key, never by position. Returns ``(lpd_by_model, dev_lag,
    n_floored)`` with ``dev_lag`` in the same canonical order (the hierarchical
    method's covariate).
    """
    pointwise = panel.pointwise
    live = pointwise[pointwise["on_elpd_panel"] & pointwise["model"].isin(panel.elpd_members)]
    out: dict[str, np.ndarray] = {}
    reference_keys: list | None = None
    dev_lag = np.empty(0)
    n_floored = 0
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
        if np.isnan(values).any():
            raise ValueError(
                f"{model} has NaN pointwise ELPD on the panel; NaN is a bug upstream "
                "(-inf is the legitimate zero-density verdict), find it rather than fit "
                "weights around it"
            )
        floored = np.isneginf(values)
        n_floored += int(floored.sum())
        values = np.where(floored, LPD_FLOOR, values)
        out[model] = values
    return out, dev_lag, n_floored


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
