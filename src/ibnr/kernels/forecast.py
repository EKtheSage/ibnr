"""Held-out forecasts, aligned across models, reduced to a leaderboard.

``kernels/holdout.py`` decides which cells are held out. Two opt-in mixins turn
a fit plus those cells into an ``(n_draws, n_cells)`` array: ``ScoresHeldout``
gives log densities, already carried to Lebesgue-on-the-loss-amount by
``densities.to_amount_scale``, and ``PredictsHeldout`` gives predictive draws,
already carried to the triangle's own basis. This module is everything between
those arrays and a board, and it is exactly three jobs:

1. reduce the draw axis - with :func:`logmeanexp` for a density, never ``mean``;
   with ``scores.crps`` for draws;
2. put every model on ONE identical set of cells per score, joined by key, never
   by position;
3. report a capability a model does not have as **N/A** - not 0, not dropped.

Nothing here has an external reference to check itself against. Upstream, every
number is pinned by something: ``holdout.py`` checks cells against the fit's own
contract, ``densities.py`` integrates each measure to 1, and
``test_heldout_scorer_csr.py`` checks the scorer against the fit's own
``log_lik``. This layer takes correct arrays and reduces them, so every wrong
answer it can produce is a finite float of the right sign and the right order of
magnitude. The guards below exist because of that, not in spite of it.

What goes silently wrong
------------------------

**Mean of the logs instead of log of the mean.** ``ll.mean(axis=0)`` is one
character from :func:`logmeanexp` and estimates a different thing:
``E_theta[log p(y|theta)]`` rather than ``log E_theta[p(y|theta)]``, which is
what a predictive score is. Jensen makes the naive version always smaller, by
roughly ``Var_draws(ll)/2`` per cell. That gap is not a constant, so it does not
cancel in a comparison - it grows with how much the pointwise log density moves
across draws, which is the ratio of parameter uncertainty to the model's own
process noise. A model with a large fitted ``sig`` is *favoured* by it. Every
number it produces is negative, finite, smooth in the data, and ranks a bad fit
below a good one, so no value-only test sees it.

**The naive ``log(mean(exp(ll)))``, and its max-shifted hand-rolled cousin.**
CSR's ``sig[d]`` shrinks with development, so a badly missed deep-dev cell
reaches log densities of several hundred negative nats and ``exp`` underflows to
0.0, making the whole column ``-inf``. Worse, the textbook shift
``m + log(sum(exp(ll - m)))`` returns **NaN** when ``m`` is ``-inf``, with only
a RuntimeWarning - measured in this environment. That NaN is then skipped by
pandas (below) and the model silently scores on fewer cells. ``scipy`` is a core
dependency; :func:`logmeanexp` uses ``scipy.special.logsumexp``, which returns
``-inf`` correctly.

**A missing ELPD becomes 0.0, and 0.0 is the best score on the board.**
Measured here, pandas 2.3.3: ``df.groupby("model")["elpd"].sum()`` over an
all-missing group returns ``0.0``, not missing, and ``idxmax()`` returns that
group. **A nullable ``Float64`` dtype does not fix it** - only ``min_count=1``
does. The same 0.0 arrives through ``np.nansum([nan, nan])``,
``pivot_table(fill_value=0)``, and a ``fillna(0)`` in a formatting step. On this
board a pointwise ELPD is ``-log(y_i)`` plus the model's own term, so with the
mart in USD thousands a typical cell scores around -10 nats: imputing 0 credits
that cell with about +10, more than the whole plausible spread between two real
models. Ten of eleven registered entries have no held-out density today, so a
single ``fillna(0)`` puts all ten above the one model that was actually scored.
This module therefore never aggregates a possibly-empty column with pandas: each
model's total is reduced in Python from its own array, and an absent total is
``pd.NA`` with a reason recorded beside it.

**Positional alignment.** ``ScoresHeldout.log_lik_at`` returns
``(n_draws, n_cells)`` with no labels; ``index_into`` narrows the frame with
``frame[frame["field"] == field]`` and returns a ``CellIndex`` that carries no
key at all. So a caller holding two models' arrays has two equal-length vectors
that agree with each other perfectly and may describe different cells. A sum is
invariant under a permutation, so the model totals stay exactly right while
every paired difference, per-cohort breakdown and stacking weight underneath is
garbage. This is the same error ``index_into`` already refuses one level down -
"a single, entirely wrong cohort agrees with itself perfectly, indexes cleanly,
and yields a complete, plausible ELPD" - committed one level up.

The mitigation here is construction, not assertion: :class:`CohortForecast`
narrows ``cells.frame`` with the *same single expression* ``index_into`` uses,
and the caller must pass ``field=`` explicitly to both, so the two cannot select
different rows. The width check in ``__post_init__`` is what pins them together.
The durable fix is a ``key`` field on ``CellIndex``, populated inside
``index_into`` where the column order is actually decided; that is a change to
``holdout.py`` and does not belong in this PR.

**Summing over different cells.** ELPD is extensive, so a model that scores
fewer cells scores higher, and the cells models refuse are not a random sample -
they are the volatile ones. Dividing by ``n`` does not fix this: on the amount
scale a cell's log density is ``-log(y_i)`` plus a model term of order 1, so the
per-cell mean is dominated by which *cohorts* survived, not by how many cells
did. A cohort a hundred times larger contributes about 4.6 nats less per cell
however good the model is. The only honest cross-model ELPD is a sum over an
identical cell set, which is what :func:`align_panel` builds.

Two capabilities, two panels
---------------------------

A forecast offers a **density** (which gives ELPD), **draws** (which give CRPS),
either, or neither. The two are independent and each gets its OWN membership,
its own intersection and its own fingerprint.

That separation is not tidiness, it is the only way both columns can be honest
at once. ``england_verrall_odp`` has perfectly good draws and no usable density
(its quasi-likelihood is not normalized - see :ref:`odp-not-a-density`), and it
refuses roughly half the mart outright. On one shared panel those refusals would
delete cells from ``meyers_csr``'s **ELPD**, a column ODP does not even appear
in. Measured on a two-cohort fixture: with separate panels, ODP refusing a cohort
took the CRPS panel from 8 cells to 4 and left the ELPD panel at 8, unchanged.

Each panel is the **intersection over the models eligible for that score**, and
each column carries its own member list, ``n_cells`` and fingerprint. Adding an
eligible model can therefore change another model's published number in that
column - deliberately. The alternative (each model on its own coverage) makes the
column not a comparison at all, which is worse; and the usual objection, that a
number depending on a set is not reproducible, is answered the same way
``mart_publish_id`` answers it: the number carries the set.

The consequence a reader must not miss is that **``elpd`` and ``crps`` on one
board row can rest on different cells**. That is why neither is ever labelled
with a bare ``n_cells``: the board has ``n_cells_elpd`` and ``n_cells_crps``, and
``panel.cells`` flags each cell with which panels it is on.

Two vocabularies both spelled "measure"
---------------------------------------

``HoldoutCells.measure`` is the TRIANGLE's basis, ``cumulative`` or
``incremental``. ``densities.MEASURES`` is the DENSITY's scale, ``amount`` /
``log_amount`` / ``loss_ratio``. A :class:`CohortForecast`'s ``log_density`` is
always on Lebesgue-on-amount, because ``log_lik_at`` is the only way to produce
one and it always carries - so there is deliberately no measure field here, a
field with exactly one legal value being a knob that cannot be turned. The
triangle basis is carried, through ``cells.measure``, and is checked for
agreement across models: ``X = C - C_prev`` has Jacobian 1 so the *densities* of
a cumulative and an incremental fit are comparable, but the *observed values*
are different numbers and anything scored against them would not be.

Panel geometry, measured
------------------------

On the milestone-3 screened panel (152 ``(company, line)`` cohorts, complete
10x10 squares over accident years 1988-1997), built with ``origins`` restricted
to the study window:

===================  =======  ==========  =====================================
``as_of``            cells    dev lags    exclusions
===================  =======  ==========  =====================================
1997-12-31           9        24..120     none
1996-12-31           8        24..108     1 new_origin, 1 dev_beyond_trained
===================  =======  ==========  =====================================

So the evaluation panel is 152 x 9 = 1368 cells, and the panel the two-cutoff
stacking design would fit weights on is 152 x 8 = 1216 cells with a
systematically shallower development mix - it never contains a dev-120 cell,
which is exactly where CSR's ``sig[d]`` is smallest and the log density most
extreme. The two panels are not interchangeable and this module refuses to hold
both at once: a panel pins one ``as_of``, and stacking reads two panels.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ibnr.kernels.holdout import EXCLUSION_REASONS, HoldoutCells
from ibnr.kernels.scores import crps

__all__ = [
    "ABSENCE_REASONS",
    "CAPABILITIES",
    "COHORT_ABSENCE_REASONS",
    "MODEL_ABSENCE_REASONS",
    "REASON_AXIS",
    "SCORE_DIRECTION",
    "Absence",
    "CohortForecast",
    "ForecastPanel",
    "align_panel",
    "leaderboard",
    "logmeanexp",
    "resolve_field",
]


#: Why a whole MODEL has no held-out density. Uniform across every cohort of
#: that model, and checked to be so - a reason that applies to some cohorts and
#: not others is a cohort-level reason, and mixing the two is how "this entry
#: cannot do it" gets confused with "this run did not do it".
#:
#: The two are kept apart because they libel different entries. Nine of the
#: eleven registered entries do not subclass ``ScoresHeldout`` today, but only
#: four of them are ineligible on principle (the ones ``no_predictive_density``
#: names below); saying "no normalized predictive density" about ``sur``
#: (multivariate normal), ``copula_glm`` (lognormal marginals) or the
#: MDN-headed NN entries would be false, and the board prints these.
MODEL_ABSENCE_REASONS: dict[str, str] = {
    "no_predictive_density": (
        "the entry has no normalized predictive density on any scale, so no change of "
        "variable can give it an ELPD. england_verrall_odp and clark_growth_curve and "
        "statistical/clark (ODP quasi-likelihood - Poisson only up to proportionality, "
        "see densities.py :ref:`odp-not-a-density`) and deterministic/mack (bootstrap, "
        "no stated observation model) are the entries this covers. Permanent N/A, and "
        "the entry still occupies a board row. DENSITY axis only - most of these entries "
        "can still draw, so they belong on the CRPS board"
    ),
    "no_cell_sampler": (
        "the entry cannot draw an outcome at an individual held-out cell. A point or "
        "quantile predictor is the clear case. DRAWS axis only. Note this is a much "
        "narrower claim than 'has no predictive distribution': deterministic/mack's "
        "bootstrap has no stated observation model, so no density, but it can certainly "
        "draw - it belongs on the CRPS board and not the ELPD one"
    ),
    "scorer_not_implemented": (
        "the entry states an observation model but has no held-out scorer yet - it does "
        "not subclass ScoresHeldout (density axis) or PredictsHeldout (draws axis). "
        "Temporary, and NOT a statement that the entry lacks the capability. Use "
        "Absence.detail to say what is missing (compartmental's blocker, for instance, is "
        "that training_index refuses its delta-stacked contract, so it has no in-sample "
        "agreement gate behind any held-out number it would produce)"
    ),
}

#: Which axis a model-level reason may be declared on. A reason that names the
#: wrong capability is refused, because the board PRINTS these: labelling
#: ``deterministic/mack`` ``no_predictive_density`` on the CRPS column would say
#: it cannot draw, which is false - its bootstrap draws perfectly well, it just
#: has no observation model to give an ELPD.
REASON_AXIS: dict[str, str] = {
    "no_predictive_density": "density",
    "no_cell_sampler": "draws",
}

#: Why one (model, cohort) has no density, for a model that otherwise does.
#: Cohort-atomic on purpose: no shipped scorer refuses individual cells.
#: ``meyers_csr``'s raises for the WHOLE array on any non-positive loss, and
#: ``to_amount_scale`` raises for the whole array on a non-positive covariate,
#: so one zero-paid cell costs the cohort's nine cells, not one.
COHORT_ABSENCE_REASONS: dict[str, str] = {
    "fit_failed": "the fit raised, or failed its convergence gates, so there are no draws",
    "scoring_refused": (
        "the fit exists but the scorer or the measure carry refused this cohort - a "
        "documented family limit such as a non-positive loss under a lognormal. Put the "
        "exception text in Absence.detail"
    ),
    "not_offered": (
        "this run did not ask the model about this cohort. Distinct from a refusal: a "
        "refusal is evidence about the model, this is a gap in the run"
    ),
}

ABSENCE_REASONS: dict[str, str] = {**MODEL_ABSENCE_REASONS, **COHORT_ABSENCE_REASONS}

#: The two capabilities a forecast can offer, and the board column each feeds.
#: They are INDEPENDENT: an entry may have either, both, or neither. Keeping them
#: apart is what lets ``england_verrall_odp`` be scored on CRPS while staying off
#: the ELPD panel entirely - and, crucially, what stops its refusing half the
#: mart from shrinking anybody's ELPD.
CAPABILITIES: dict[str, str] = {"density": "elpd", "draws": "crps"}

#: Which way is good, per score column. The leaderboard has no default sort and
#: no sort argument (decided 2026-07-25), which moves the risk onto whatever
#: renders it - and a board carrying two columns that point in opposite
#: directions, with no machine-readable direction, invites an average-rank
#: column that is nonsense.
SCORE_DIRECTION: dict[str, str] = {
    "elpd": "higher_is_better",
    "elpd_per_cell": "higher_is_better",
    "crps": "lower_is_better",
    "crps_per_cell": "lower_is_better",
}

#: Cell identity. Segment columns are spliced in per panel - which ones exist is
#: a property of the triangle, not of this module. ``as_of`` leads because the
#: same cohort's same cell predicted from a 1996 fit and from a 1997 fit are two
#: different predictions; it is NOT there for collision safety (the two cutoffs'
#: cells are disjoint in ``(origin_period, dev_lag)`` anyway), it is there so a
#: long frame concatenating two panels is unambiguous and so a fingerprint
#: cannot match across cutoffs.
#:
#: ``eval_date`` is deliberately NOT in the key. It is one value per
#: (cohort, cutoff) and is checked for agreement instead: two models that
#: disagree about the next diagonal were handed different training slices, and
#: that is an error to raise rather than a distinction to join on.
KEY_LEAD: tuple[str, ...] = ("task", "as_of")
KEY_TAIL: tuple[str, ...] = ("field", "origin_period", "dev_lag")


@dataclass(frozen=True)
class Absence:
    """Why an array is missing, from a closed vocabulary.

    A bare ``None`` cannot distinguish "this entry has no density" from "this
    run did not compute one", and the board must: the first is a permanent N/A
    that belongs on every board forever, the second is a gap. So ``None`` never
    stands alone - :class:`CohortForecast` refuses an absent array with no
    ``Absence`` and refuses a present array that has one.

    The vocabulary is closed so the N/A column is a census that can be counted
    across runs, rather than free text that can only be read.
    """

    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in ABSENCE_REASONS:
            raise ValueError(
                f"reason must be one of {sorted(ABSENCE_REASONS)}, got {self.reason!r}. "
                "Add a named reason rather than a free-text one: the board prints these "
                "and an open vocabulary cannot be read as a census"
            )

    @property
    def is_model_level(self) -> bool:
        return self.reason in MODEL_ABSENCE_REASONS

    def check_axis(self, axis: str) -> None:
        """Refuse a reason that names the other capability.

        ``no_predictive_density`` on the draws axis would print "has no
        normalized predictive density" beside an empty CRPS cell, which reads as
        "cannot draw" and is false for every ODP and bootstrap entry in the
        gallery. The two axes are independent, so their vocabularies have to be.
        """
        wants = REASON_AXIS.get(self.reason)
        if wants is not None and wants != axis:
            raise ValueError(
                f"reason {self.reason!r} describes the {wants!r} capability but was "
                f"declared on the {axis!r} one. {ABSENCE_REASONS[self.reason]}"
            )

    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


def logmeanexp(log_values, *, axis: int = 0) -> np.ndarray:
    """``log(mean(exp(x)))`` along ``axis``, stably.

    This is the whole of the ELPD reduction over draws and the single easiest
    place in the milestone to be quietly wrong, because the wrong answer is
    smooth, finite, correctly signed and correctly ordered nearly always.

    It estimates ``log E_theta[p(y | theta)]``, the log posterior predictive
    density. ``x.mean(axis=0)`` estimates ``E_theta[log p(y | theta)]``, which is
    the average log density of a *randomly drawn* parameter and is not a
    predictive score. By Jensen the second is always smaller, by about
    ``Var_draws(x)/2`` per cell - a gap that varies by model, so it does not
    cancel in a comparison.

    ``-inf`` is preserved, in both forms. A single ``-inf`` draw is fine and is
    absorbed correctly: part of the posterior gave the outcome zero density. A
    column of all ``-inf`` returns ``-inf``, which is a verdict, not a bug - the
    model assigned zero probability to something that happened. It is
    ``scipy.special.logsumexp`` that gets this right; the textbook max-shift
    ``m + log(sum(exp(x - m)))`` returns NaN there, because ``-inf - (-inf)`` is
    NaN, and only warns.

    NaN is refused rather than propagated: it is never a verdict here, and
    pandas skips it by default two layers away from wherever it came from.
    """
    a = np.asarray(log_values, dtype=float)
    if a.size and np.isnan(a).any():
        raise ValueError(
            f"{int(np.isnan(a).sum())} NaN value(s) in the log densities. NaN is not a "
            "score - -inf is the way a model says 'zero density here'. A NaN usually "
            "means a hand-rolled logmeanexp hit an all -inf column, or an inf - inf "
            "upstream; find it rather than letting an aggregation skip it"
        )
    n = a.shape[axis]
    if n < 2:
        raise ValueError(
            f"logmeanexp needs at least 2 draws along axis {axis}, got {n}. At one draw "
            "it is arithmetically identical to the mean of the logs - the exact estimand "
            "this function exists to avoid - and it is a plug-in density at a single "
            "parameter value rather than a posterior predictive one"
        )
    return logsumexp(a, axis=axis) - np.log(n)


def _ess_kish(log_values: np.ndarray) -> np.ndarray:
    """Per cell, the effective number of draws behind its ``logmeanexp``.

    ``(sum w)^2 / sum w^2`` on ``w_s = exp(ll_s)``, which is ``S / (1 + cv2)``.
    There are no importance weights anywhere in this computation - the draws are
    exact posterior draws and nothing is reweighted - so Pareto-k in the PSIS
    sense does not apply. What can still go wrong is that the *integrand* is
    heavy-tailed across draws, so one draw out of ten thousand carries the
    answer; then the estimate is very noisy and, because of the outer log,
    biased downward. This is the readout for that, and it is why the board
    reports a panel minimum rather than an average.

    Ignores MCMC autocorrelation, so it is an upper bound on the true effective
    count. An all ``-inf`` column returns 0.0 rather than NaN.
    """
    a = np.asarray(log_values, dtype=float)
    out = np.zeros(a.shape[1], dtype=float)
    top = a.max(axis=0)
    live = np.isfinite(top)
    if live.any():
        w = np.exp(a[:, live] - top[live])
        s1 = w.sum(axis=0)
        out[live] = (s1 * s1) / (w * w).sum(axis=0)
    return out


def resolve_field(cells: HoldoutCells) -> str:
    """The single field a ``HoldoutCells`` covers, or an error.

    Resolved from the CELLS, not from a contract. ``index_into`` can default the
    field from ``contract["models"]``, but that route is unavailable twice over
    here: the returned ``CellIndex`` does not say which field was chosen, and an
    ELPD-ineligible entry may have no contract at all - which is exactly the
    case the N/A machinery exists for, so its row must still be keyable.
    """
    present = sorted(cells.frame["field"].unique())
    if not present:
        raise ValueError("these HoldoutCells carry no scorable rows")
    if len(present) > 1:
        raise ValueError(
            f"these HoldoutCells span fields {present}; pass field= explicitly. A forecast "
            "covers one field, the same narrowing index_into applies"
        )
    return str(present[0])


def _narrowed(cells: HoldoutCells, field: str) -> pd.DataFrame:
    """``cells.frame`` restricted to one field.

    Deliberately the same single expression ``index_into`` uses (holdout.py, the
    ``frame = frame[frame["field"] == field]`` line), on the same frame, so the
    row order behind an ``(n_draws, n_cells)`` array and the row order of the
    keys attached to it cannot diverge. ``next_diagonal`` sorts and resets the
    index before returning, so a boolean mask preserves the relative order in
    both places.
    """
    out = cells.frame[cells.frame["field"] == field]
    if out.empty:
        raise ValueError(
            f"no held-out cells for field={field!r}; this HoldoutCells covers "
            f"{sorted(cells.frame['field'].unique())}"
        )
    return out


@dataclass(frozen=True, eq=False)
class CohortForecast:
    """One model, one cohort, one cutoff, one field, at the held-out cells.

    This granularity is forced from both sides. Below it, every layer is
    single-cohort and refuses to be otherwise: ``next_diagonal`` raises on more
    than one segment combination (measured before the guard: 20 January-origin
    cohorts plus one July-origin cohort returned 2 cells instead of 62), and all
    three contract builders raise on a multi-cohort triangle. Above it, every
    layer loops per cohort, and ``kernels.harness`` runs those loops in **spawned
    processes** - so whatever a worker returns must be picklable, which a live
    entry or an ``InferenceData`` is not. Plain numpy plus a ``HoldoutCells`` is.

    ``cells`` is carried whole rather than copied piecemeal. That is provenance:
    only ``next_diagonal`` can build one, so a ``CohortForecast`` cannot describe
    cells that were never shown to be held out, and ``as_of``, ``eval_date``, the
    segment schema, the triangle basis, ``train_origins`` and the upstream
    exclusion census all arrive consistent with each other by construction rather
    than by five parallel arguments that can drift.

    ``log_density`` is ``(n_draws, n_cells)`` on Lebesgue-on-the-loss-amount -
    the scale ``ScoresHeldout.log_lik_at`` guarantees and the only scale this
    module accepts. Either it or ``absence`` is set, never both and never
    neither.

    A pooled fit (the NN entries fit once globally, then predict per cohort)
    produces one ``CohortForecast`` per scored cohort from a single fit. That is
    still held out - the pooled model trained on the ``as_of`` slice of every
    cohort - but ``model`` names the entry, not the fit, and any future
    clustered standard error must cluster on the FIT, which for those entries is
    the whole panel and not the cohort.

    Refused at construction, each because the alternative computes:

    * an array with no ``Absence``, or an ``Absence`` with an array;
    * a width that disagrees with the field-narrowed cell count - the check that
      pins the arrays to their keys;
    * fewer than 2 draws (see :func:`logmeanexp`);
    * zero cells: a cohort whose whole diagonal was excluded upstream has
      nothing to score, and a zero-width forecast would pass alignment
      invisibly while making the model look like it covered the cohort. Read
      ``cells.exclusion_counts()`` instead;
    * any NaN;
    * zero variance across draws in every cell, which means someone handed a
      repeated point estimate rather than posterior draws. ``logmeanexp`` then
      degenerates to the plug-in log density, which is systematically
      overconfident and looks completely normal.
    """

    model: str
    task: str
    cells: HoldoutCells
    field: str
    log_density: np.ndarray | None = None
    draws: np.ndarray | None = None
    density_absence: Absence | None = None
    draws_absence: Absence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cells, HoldoutCells):
            raise TypeError(
                "cells must be a HoldoutCells built by next_diagonal, so the forecast "
                f"cannot describe cells nothing showed to be held out; got {type(self.cells)}"
            )
        frame = _narrowed(self.cells, self.field)
        if len(frame) == 0:
            raise ValueError(
                f"{self.model}: zero cells. A cohort whose diagonal was entirely excluded "
                "upstream has nothing to score; a zero-width forecast would align "
                "invisibly and make the model look like it covered this cohort. "
                f"Exclusions here: {self.cells.exclusion_counts()}"
            )

        for axis, array_name, absence_name in (
            ("density", "log_density", "density_absence"),
            ("draws", "draws", "draws_absence"),
        ):
            array = getattr(self, array_name)
            absence = getattr(self, absence_name)
            if (array is None) == (absence is None):
                raise ValueError(
                    f"{self.model}: exactly one of {array_name} and {absence_name} must be "
                    f"set. An array with no reason attached and a reason with an array both "
                    "mean the board cannot say whether a missing number is permanent or a "
                    "gap in a run. A model offering neither capability sets BOTH absences"
                )
            if absence is not None:
                absence.check_axis(axis)
                continue
            object.__setattr__(self, array_name, self._validated(array, array_name, len(frame)))

    def _validated(self, array, name: str, n_cells: int) -> np.ndarray:
        """Shared shape and sanity checks for both capability arrays."""
        a = np.asarray(array, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"{self.model}: {name} must be 2-D (draws x cells), got {a.shape}")
        if a.shape[1] != n_cells:
            raise ValueError(
                f"{self.model}: {name} has {a.shape[1]} columns but field="
                f"{self.field!r} narrows these cells to {n_cells} rows. The columns and "
                "the keys are matched by POSITION at this one seam, so a width mismatch is "
                "the only signal that they came from different row selections - pass the "
                "same explicit field= to log_lik_at() / predict_at() that you passed here, "
                "and never let index_into default it"
            )
        if a.shape[0] < 2:
            raise ValueError(
                f"{self.model}: {name} has {a.shape[0]} draw(s). Two is the minimum for "
                "both reductions: at one draw logmeanexp is the mean of the logs and CRPS "
                "has an identically-zero spread term, so each degenerates to a plug-in "
                "estimate that is systematically overconfident and looks entirely normal"
            )
        if np.isnan(a).any():
            raise ValueError(
                f"{self.model}: {int(np.isnan(a).sum())} NaN value(s) in {name}. NaN is a "
                "bug that turns a board total into NaN two layers from its cause"
            )
        if name == "draws" and not np.isfinite(a).all():
            raise ValueError(
                f"{self.model}: {int((~np.isfinite(a)).sum())} non-finite draw(s). An "
                "infinite loss is not a forecast - unlike a log density, where -inf is the "
                "legitimate verdict 'this outcome had zero probability'"
            )
        # Variance per column, over its FINITE entries only. A -inf row in a
        # log-density column is a verdict ("this draw gave the outcome zero
        # density"), not a draw of the same quantity, so it carries no variance
        # information - and a column mixing finite and -inf rows would otherwise
        # put a NaN into ``var`` (with a RuntimeWarning), and ``NaN == 0.0`` is
        # False, silently disarming this guard for the WHOLE forecast: a
        # repeated point estimate in the other columns then sails through. A
        # column needs at least 2 finite entries to have a variance; columns
        # with fewer (all -inf is the common case) are excluded, and the guard
        # fires when every column that has one shows exactly 0.0. Draws are
        # fully finite (checked above), so there this is the plain per-column
        # variance, unchanged.
        finite = np.isfinite(a)
        checkable = np.flatnonzero(finite.sum(axis=0) >= 2)
        variances = [float(a[finite[:, j], j].var()) for j in checkable]
        if variances and max(variances) == 0.0:
            raise ValueError(
                f"{self.model}: {name} has zero variance across draws in every cell, so "
                "these are not posterior draws - a point estimate was repeated. The "
                "reduction then returns a plug-in value, which is systematically "
                "overconfident and entirely plausible-looking"
            )
        return a

    # -- identity --------------------------------------------------------------

    @property
    def as_of(self) -> dt.date:
        return self.cells.as_of

    @property
    def eval_date(self) -> dt.date:
        return self.cells.eval_date

    @property
    def measure(self) -> str:
        """The TRIANGLE's basis, cumulative or incremental - not the density's."""
        return self.cells.measure

    @property
    def key_frame(self) -> pd.DataFrame:
        """One row per covered cell: the panel key, then the payload.

        In the same order as ``log_density``'s columns, which is what makes
        :meth:`pointwise_elpd` joinable.
        """
        frame = _narrowed(self.cells, self.field)
        out = pd.DataFrame(index=range(len(frame)))
        out["task"] = self.task
        out["as_of"] = self.as_of
        for column in [*self.cells.segments, *KEY_TAIL]:
            out[column] = frame[column].to_numpy()
        out["eval_date"] = self.eval_date
        out["measure"] = self.measure
        out["value"] = frame["value"].to_numpy(dtype=float)
        out["premium"] = (
            frame["premium"].to_numpy(dtype=float)
            if "premium" in frame.columns
            else np.full(len(frame), np.nan)
        )
        return out

    @property
    def keys(self) -> list[tuple]:
        """Panel keys, in array column order. Tuples, so they set-operate."""
        columns = [*KEY_LEAD, *self.cells.segments, *KEY_TAIL]
        return [tuple(row) for row in self.key_frame[columns].itertuples(index=False, name=None)]

    @property
    def cohort(self) -> tuple:
        """The cohort's segment VALUES. ``()`` for an unsegmented triangle."""
        if not self.cells.segments:
            return ()
        frame = _narrowed(self.cells, self.field)
        return tuple(frame.iloc[0][s] for s in self.cells.segments)

    @property
    def n_cells(self) -> int:
        return len(_narrowed(self.cells, self.field))

    @property
    def n_draws(self) -> int | None:
        """Draw count behind whichever arrays are present, or ``None``.

        The two capabilities may carry different draw counts - a density is one
        value per posterior draw while a sampler may thin - so this reports the
        density's when there is one and the draws' otherwise, and
        :meth:`n_draws_for` answers per axis.
        """
        for array in (self.log_density, self.draws):
            if array is not None:
                return int(array.shape[0])
        return None

    def n_draws_for(self, axis: str) -> int | None:
        array = self.log_density if axis == "density" else self.draws
        return None if array is None else int(array.shape[0])

    @property
    def has_density(self) -> bool:
        return self.log_density is not None

    @property
    def has_draws(self) -> bool:
        return self.draws is not None

    def offers(self, axis: str) -> bool:
        if axis not in CAPABILITIES:
            raise ValueError(f"axis must be one of {sorted(CAPABILITIES)}, got {axis!r}")
        return self.has_density if axis == "density" else self.has_draws

    def absence_for(self, axis: str) -> Absence | None:
        return self.density_absence if axis == "density" else self.draws_absence

    # -- reductions ------------------------------------------------------------

    def pointwise_elpd(self) -> np.ndarray:
        """``(n_cells,)`` log predictive density per cell, ``logmeanexp`` over draws.

        Per cell, THEN summed - never summed inside the exponential. Summing the
        log densities of a cohort's cells before the reduction gives the log
        density of the whole diagonal *jointly* under each draw, which is a
        legitimate but different quantity: the cells share one posterior, so the
        joint is not the product of the marginals and it is not additive across
        cohorts in the way a leaderboard and a stacking objective both need.
        """
        if self.log_density is None:
            raise ValueError(f"{self.model} has no density here: {self.density_absence}")
        return logmeanexp(self.log_density, axis=0)

    def pointwise_crps(self) -> np.ndarray:
        """``(n_cells,)`` CRPS per cell, against the realized outcomes. Lower is better.

        The draws arrive on the TRIANGLE's basis - that is
        ``PredictsHeldout.predict_at``'s guarantee - so they are scored against
        ``cells.values`` directly and no scale question survives to this layer.

        CRPS is translation-equivariant, so an entry that models increments and
        one that models cumulatives give comparable numbers once each is on its
        triangle's basis; it is scale-*dependent* in the other sense, though, so
        a CRPS in dollars and one in thousands are not comparable. That is the
        same units question the panel's ``value`` agreement check already
        catches, one layer up.
        """
        if self.draws is None:
            raise ValueError(f"{self.model} has no draws here: {self.draws_absence}")
        return crps(self.draws, _narrowed(self.cells, self.field)["value"].to_numpy(dtype=float))

    def ess_kish(self) -> np.ndarray:
        """``(n_cells,)`` effective draws behind each cell. See :func:`_ess_kish`."""
        if self.log_density is None:
            raise ValueError(f"{self.model} has no density here: {self.density_absence}")
        return _ess_kish(self.log_density)

    @classmethod
    def unavailable(
        cls,
        *,
        model: str,
        task: str,
        cells: HoldoutCells,
        density_reason: str,
        draws_reason: str,
        detail: str = "",
        field: str | None = None,
    ) -> CohortForecast:
        """A forecast that offers NEITHER capability, naming a reason for each.

        Both reasons are REQUIRED and neither has a default. "Does not subclass
        ``ScoresHeldout``" is not by itself evidence of which reason applies, and
        a default would put the wrong one on a board row: only four registered
        entries have no density on principle while five more simply have no
        scorer yet, and printing ``no_predictive_density`` next to ``sur``
        or ``copula_glm`` would be a false statement about the model.

        Two reasons rather than one because the axes are independent and usually
        differ. ``deterministic/mack`` is the clearest case: ``no_predictive_density``
        on the density axis is permanent (its bootstrap states no observation
        model), while its draws axis is real - it subclasses ``PredictsHeldout``
        and belongs on the CRPS board. A single shared reason would libel an
        entry shaped like that on one axis or the other.
        """
        return cls(
            model=model,
            task=task,
            cells=cells,
            field=field if field is not None else resolve_field(cells),
            density_absence=Absence(reason=density_reason, detail=detail),
            draws_absence=Absence(reason=draws_reason, detail=detail),
        )

    def __repr__(self) -> str:
        parts = [
            f"elpd={self.n_cells}x{self.n_draws_for('density')}"
            if self.has_density
            else f"elpd=N/A({self.density_absence.reason})",
            f"crps={self.n_cells}x{self.n_draws_for('draws')}"
            if self.has_draws
            else f"crps=N/A({self.draws_absence.reason})",
        ]
        return (
            f"CohortForecast({self.model}, {self.task} @ {self.as_of}, "
            f"cohort={self.cohort}, field={self.field}, {', '.join(parts)})"
        )


@dataclass(frozen=True, eq=False)
class ForecastPanel:
    """Many models on ONE identical set of held-out cells, at one cutoff.

    ``cells``      one row per panel cell: the key, then ``eval_date``,
                   ``measure``, ``value``, ``premium``. The outcomes, once.
    ``pointwise``  one row per (model, cell): ``elpd``, ``ess_kish``,
                   ``n_draws``. Long format - it is what any future paired test
                   between two models reads, and it is the same shape as
                   everything else in this package.
    ``by_cohort``  one row per (model, cohort): the cohort's ELPD sum and cell
                   count. The unit any clustered standard error will need.
    ``coverage``   one row per model: what it offered, what survived, and its
                   membership status.
    ``absences``   one row per (model, cohort) with no density: reason, detail.
                   The census behind every N/A on the board.
    ``dropped``    one row per cell a member offered and the panel could not
                   use, naming the members that lacked it. Never empty silently.
    ``excluded``   the upstream ``HoldoutCells.exclusion_counts()`` per cohort,
                   carried through so a smaller diagonal is visible on the board
                   rather than being mistaken for an easier one.
    ``members``    the ELPD-eligible models the intersection was taken over.
                   Stamped, because the ELPD column is a function of this set.
    ``fingerprint``sha256 of the sorted panel keys, so two runs can be checked
                   for panel identity mechanically rather than by eye.

    ``as_of`` is a panel-level scalar. It is part of cell identity, but a panel
    enforces one value by REFUSING a mix rather than by keying on it: a
    mixed-cutoff intersection would silently delete every cohort fitted at only
    one of the two cutoffs, and the cutoffs' panels are not even the same shape
    (9 cells per cohort at 1997, 8 at 1996 - see the module docstring). Two
    cutoffs is two panels, which is also what lets a later ``stack()`` assert
    ``weights.as_of < evaluation.as_of`` instead of trusting its caller.
    """

    task: str
    as_of: dt.date
    segments: tuple[str, ...]
    units: str | None
    elpd_members: tuple[str, ...]
    crps_members: tuple[str, ...]
    elpd_fingerprint: str
    crps_fingerprint: str
    cells: pd.DataFrame
    pointwise: pd.DataFrame
    by_cohort: pd.DataFrame
    coverage: pd.DataFrame
    absences: pd.DataFrame
    dropped: pd.DataFrame
    excluded: pd.DataFrame

    @property
    def key_columns(self) -> list[str]:
        return [*KEY_LEAD, *self.segments, *KEY_TAIL]

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(self.coverage["model"])

    @property
    def n_cells(self) -> int:
        """Cells in the UNION of the two panels. Neither column's n - use those."""
        return len(self.cells)

    def n_cells_for(self, score: str) -> int:
        """Cells on one score's own panel. The denominator that column is on."""
        if score not in set(CAPABILITIES.values()):
            raise ValueError(f"score must be one of {sorted(CAPABILITIES.values())}, got {score!r}")
        return int(self.cells[f"on_{score}_panel"].sum())

    @property
    def n_cohorts(self) -> int:
        return int(self.cells.groupby(list(self.segments)).ngroups) if self.segments else 1

    def n_cohorts_for(self, score: str) -> int:
        on = self.cells[self.cells[f"on_{score}_panel"]]
        if not self.segments:
            return 1 if len(on) else 0
        return int(on.groupby(list(self.segments)).ngroups)

    def __repr__(self) -> str:
        return (
            f"ForecastPanel({self.task} @ {self.as_of}, {len(self.coverage)} models, "
            f"{self.n_cohorts} cohorts | elpd: {self.n_cells_for('elpd')} cells over "
            f"{list(self.elpd_members)} | crps: {self.n_cells_for('crps')} cells over "
            f"{list(self.crps_members)} | dropped={len(self.dropped)})"
        )


def _fingerprint(keys: list[tuple]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys, key=repr):
        digest.update(repr(key).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def _refuse_mixed(name: str, values: set, extra: str) -> None:
    if len(values) > 1:
        raise ValueError(f"forecasts disagree on {name}: {sorted(values, key=repr)}. {extra}")


def align_panel(forecasts: Iterable[CohortForecast], *, units: str | None = None) -> ForecastPanel:
    """Put every model on the SAME cells - per score - and report what fell out.

    Each score's panel is the **intersection over the models eligible for that
    score**: a cell survives on the ELPD panel only if every density-member
    offered a density for it, and on the CRPS panel only if every draws-member
    offered draws for it. That is not a convenience, it is what makes either
    column a comparison at all - two totals over different cell sets differ by
    the cells, not by the models, and that difference is invisible on the board
    because both numbers are finite, the right sign and the right order of
    magnitude. Normalizing per cell does not repair it either: a cell's log
    density is ``-log(y_i)`` plus a model term of order 1, so a per-cell mean
    over a different set of cohorts is dominated by cohort SIZE.

    **The two panels are separate**, and that is load-bearing rather than
    fastidious. ``england_verrall_odp`` is a draws-member and not a
    density-member; if there were one shared panel, its refusing about half the
    mart would delete cells from ``meyers_csr``'s ELPD - a column ODP does not
    appear in at all.

    **Membership, and what it costs.** A model is a member for a score if it has
    no model-level absence on that axis and offered that capability for at least
    one cohort. So:

    * ``deterministic/mack`` is a CRPS member and never an ELPD one: its
      bootstrap draws fine but states no observation model. ``england_verrall_odp``,
      ``clark_growth_curve`` and ``statistical/clark`` are the same shape.
    * A member that refuses one cohort removes that cohort **from that score's
      panel** for every other member of that score. The price is real and it is
      reported cell by cell in ``dropped``, which names both the score and the
      members that lacked the cell; the alternative is an incomparable column.
    * A model that offered nothing on an axis is excluded from that membership
      rather than allowed to empty the panel, and its row says
      ``na: no_scored_cohorts``. Including it would collapse that column to zero
      cells and tell the reader nothing about anyone.

    ``task``, ``as_of``, the segment schema and the triangle basis are DERIVED
    from the forecasts and a mix is refused, rather than being parameters. The
    precedent is ``index_into``, where ``field`` was made to default to the
    contract's own: a caller who can pass the task is a caller who can pass the
    wrong one, and the forecasts already know.

    Raises, each because the alternative computes:

    ``mixed task``
        The paid and reported boards stay separate. Merging them sums densities
        of two different quantities into one number that still prints.
    ``mixed as_of``
        Two cutoffs is two panels. See :class:`ForecastPanel`.
    ``mixed segment schema``
        A cohort keyed on ``(company, line)`` and one keyed on ``line`` alone
        cannot be told apart along the missing dimension - the same reason
        ``index_into`` compares schemas and not just values.
    ``duplicate (model, cell)``
        Two forecasts of one model covering one cell, e.g. a re-run appended to
        a results list. The cell would count twice in that model's sum and once
        in everyone else's.
    ``disagreeing observed value``
        Two models report different outcomes at the same key. Reachable, and the
        likeliest cause is not exotic: Meyers' ``pmax(paid, 1)`` clamp, which
        the retro harness applies before fitting for the lognormal entries only.
        A clamped run and an unclamped one produce identical keys and different
        values at exactly the cells that matter. It is also what catches a units
        mismatch, which would otherwise shift every log density by a constant
        large enough to decide the ranking.
    ``disagreeing eval_date``
        Same cohort and cutoff, different next diagonal: the models were handed
        different training slices, so nothing below is a comparison.
    ``disagreeing measure``
        One model's cells are cumulative and another's incremental. The
        densities would be comparable (Jacobian 1) but the observed values are
        different numbers.
    ``disagreeing train_origins``
        The models were fitted over different origin windows. This passes every
        other check here - same keys, same values, same eval_date - and CLAUDE.md
        already records this exact bug class inflating an outcome aggregate 2.4x.
    ``disagreeing exclusion counts``
        The upstream held-out definitions differ, i.e. the models were not asked
        the same question.
    ``empty intersection``
        Raised WITH the per-model coverage counts, so the caller can see which
        member emptied the board and drop it deliberately. Returning an empty
        panel would let a leaderboard of zeros be published.

    An entry whose contract carries ``delta`` (compartmental) cannot reach here:
    ``training_index`` refuses it, so it has no in-sample agreement gate, and the
    agreement gate is what makes any held-out number trustworthy. Record it as
    ``scorer_not_implemented`` with that blocker in the detail.
    """
    items = list(forecasts)
    if not items:
        raise ValueError("align_panel needs at least one CohortForecast")

    _refuse_mixed(
        "task",
        {f.task for f in items},
        "The paid and reported boards stay separate; merging them sums densities of two "
        "different quantities into one number that still prints.",
    )
    _refuse_mixed(
        "as_of",
        {f.as_of for f in items},
        "Two cutoffs is two panels: a mixed intersection silently deletes every cohort "
        "fitted at only one of them, and the cutoffs do not even produce the same number "
        "of cells per cohort.",
    )
    _refuse_mixed(
        "segment schema",
        {tuple(f.cells.segments) for f in items},
        "A fit keyed on fewer dimensions cannot tell two cohorts apart along the missing "
        "one, so its cells would index cleanly and score the wrong cohort.",
    )
    _refuse_mixed(
        "triangle measure",
        {f.measure for f in items},
        "Cumulative and incremental cells sit at the same key with different observed "
        "values, so the outcomes are not the same numbers even though the densities are "
        "comparable (Jacobian 1).",
    )

    task = items[0].task
    as_of = items[0].as_of
    segments = tuple(items[0].cells.segments)
    key_columns = [*KEY_LEAD, *segments, *KEY_TAIL]

    # -- per-forecast records, keyed ------------------------------------------
    records: list[pd.DataFrame] = []
    offered: dict[str, set[tuple]] = {}
    for f in items:
        frame = f.key_frame
        frame.insert(0, "model", f.model)
        frame["cohort"] = [f.cohort] * len(frame)
        if f.has_density:
            frame["elpd"] = f.pointwise_elpd()
            frame["ess_kish"] = f.ess_kish()
            frame["n_draws_density"] = f.n_draws_for("density")
        else:
            frame["elpd"] = np.nan
            frame["ess_kish"] = np.nan
            frame["n_draws_density"] = pd.NA
        if f.has_draws:
            frame["crps"] = f.pointwise_crps()
            frame["n_draws_sample"] = f.n_draws_for("draws")
        else:
            frame["crps"] = np.nan
            frame["n_draws_sample"] = pd.NA
        keys = f.keys
        frame["key"] = keys
        seen = offered.setdefault(f.model, set())
        clash = seen & set(keys)
        if clash:
            raise ValueError(
                f"{f.model} covers {len(clash)} cell(s) twice, e.g. {sorted(clash, key=repr)[0]}. "
                "A duplicated forecast (a re-run appended to a results list is the usual "
                "way) counts those cells twice in this model's sum and once in everyone "
                "else's, which reads as a model difference"
            )
        seen.update(keys)
        records.append(frame)
    long = pd.concat(records, ignore_index=True)

    # -- cross-model agreement on the things a key does not carry -------------
    #
    # ``dropna`` is per column and it is not a detail. An absent OUTCOME would be
    # a broken cell, so a null there is itself a disagreement. An absent PREMIUM
    # just means that model never asked for exposure, which is legitimate - only
    # entries whose density is on loss ratios need it - so nulls are skipped and
    # only the models that actually carry a premium have to agree with each other.
    for column, dropna, hint in (
        (
            "value",
            False,
            "the outcome itself differs at the same key - check the paid clamp "
            "(harness RetroTask.clamp_paid applies pmax(paid, 1) for the lognormal "
            "entries only) and the triangle's units",
        ),
        (
            "eval_date",
            False,
            "the models were handed different training slices, so their 'next diagonal' "
            "is not the same diagonal",
        ),
        (
            "premium",
            True,
            "the models were handed different EXPOSURE for the same cell. A loss-ratio "
            "density is carried to the amount scale by subtracting log(premium), so a "
            "premium of 1000 against 2000 shifts that entry's ELPD by log(2) per cell "
            "with nothing else on the board moving - and the panel would keep whichever "
            "premium it saw first. Reachable through a restated exposure or a "
            "direct-versus-net premium field",
        ),
    ):
        spread = long.groupby("key", sort=False)[column].nunique(dropna=dropna)
        bad = spread[spread > 1]
        if len(bad):
            example = long[long["key"] == bad.index[0]][["model", column]]
            raise ValueError(
                f"{len(bad)} cell(s) have a disagreeing {column} across models: {hint}. "
                f"First: {bad.index[0]} -> {example.to_dict('records')}"
            )

    by_cohort_cells = {f.cohort: f for f in items}
    for cohort in sorted({f.cohort for f in items}, key=repr):
        same = [f for f in items if f.cohort == cohort]
        _refuse_mixed(
            f"train_origins for cohort {cohort}",
            {f.cells.train_origins for f in same},
            "The fits cover different origin windows. Every other check here passes - "
            "same keys, same outcomes, same eval_date - so nothing else would catch it.",
        )
        _refuse_mixed(
            f"exclusion counts for cohort {cohort}",
            {tuple(sorted(f.cells.exclusion_counts().items())) for f in same},
            "The upstream held-out definitions differ, i.e. the models were not asked the "
            "same question about this cohort.",
        )

    # -- membership, PER CAPABILITY -------------------------------------------
    #
    # The two axes get their own membership and their own intersection, and that
    # separation is the whole reason ODP can be on the board at all. ODP has
    # draws but no usable density: on one shared panel its refusing about half
    # the mart would shrink CSR's ELPD, which is precisely the cost the
    # eligibility rule was meant to avoid. Two panels, two fingerprints, each
    # column labelled with its own n.
    all_models = sorted({f.model for f in items})
    model_level: dict[str, dict[str, Absence]] = {}
    members: dict[str, tuple[str, ...]] = {}
    offering: dict[str, dict[str, set[tuple]]] = {}
    panel_keys: dict[str, set[tuple]] = {}

    for axis, column in CAPABILITIES.items():
        level: dict[str, Absence] = {}
        for model in all_models:
            mine = [f for f in items if f.model == model]
            declared = [
                f.absence_for(axis)
                for f in mine
                if f.absence_for(axis) is not None and f.absence_for(axis).is_model_level
            ]
            if not declared:
                continue
            reasons = {a.reason for a in declared}
            if len(reasons) > 1:
                raise ValueError(
                    f"{model} declares more than one model-level {axis} absence "
                    f"{sorted(reasons)}; a model-level reason describes the ENTRY and must "
                    "be uniform across its cohorts. A reason that applies to some cohorts "
                    f"and not others is a cohort-level one ({sorted(COHORT_ABSENCE_REASONS)})"
                )
            if any(f.offers(axis) for f in mine):
                raise ValueError(
                    f"{model} declares the model-level {axis} absence "
                    f"{sorted(reasons)[0]!r} on some cohorts and offers {axis} on others. "
                    "Either the entry has that capability or it does not"
                )
            # Uniform means uniform: declared on EVERY cohort, not merely never
            # contradicted by a scored one. Mixing a model-level reason with a
            # cohort-level one on the same axis reads as "this entry has no
            # density" for the whole model, while `panel.absences` simultaneously
            # records a transient refusal for one cohort - the board and its own
            # census then disagree, and the board is the thing people read.
            if len(declared) != len(mine):
                others = sorted(
                    {
                        f.absence_for(axis).reason
                        for f in mine
                        if f.absence_for(axis) is not None
                        and not f.absence_for(axis).is_model_level
                    }
                )
                raise ValueError(
                    f"{model} declares the model-level {axis} absence "
                    f"{sorted(reasons)[0]!r} on {len(declared)} of its {len(mine)} cohorts, "
                    f"and the cohort-level reason(s) {others} on the rest. A model-level "
                    "reason is a claim about the ENTRY, so it holds for every cohort or for "
                    "none - otherwise the board reports a permanent N/A that its own absence "
                    "census contradicts"
                )
            level[model] = declared[0]
        model_level[axis] = level

        keys_by_model = {
            model: set(long.loc[(long["model"] == model) & long[column].notna(), "key"])
            for model in all_models
        }
        offering[axis] = keys_by_model
        eligible = tuple(m for m in all_models if m not in level and keys_by_model[m])
        members[axis] = eligible
        if eligible:
            common = set.intersection(*(keys_by_model[m] for m in eligible))
            if not common:
                census = {m: len(keys_by_model[m]) for m in eligible}
                raise ValueError(
                    f"the intersection over {column.upper()} members is empty, so there is "
                    f"no {column} panel. Cells offered per member: {census}. Drop the "
                    "member that empties it and rerun rather than publishing a board over "
                    "zero cells"
                )
            panel_keys[axis] = common
        else:
            panel_keys[axis] = set()

    # The cell table is the UNION of the two panels: a cell used by either column
    # belongs in it, flagged with which panels it is on. When neither axis has a
    # member nothing was comparable at all, and the union of everything offered
    # is what a reader needs to see to understand an all-N/A board.
    union_keys = panel_keys["density"] | panel_keys["draws"]
    if not union_keys:
        union_keys = set().union(*(offered[m] for m in all_models))

    cells = (
        long.drop_duplicates("key")
        .loc[
            lambda d: d["key"].isin(union_keys),
            [*key_columns, "eval_date", "measure", "value", "premium", "cohort", "key"],
        ]
        .sort_values(key_columns, kind="stable")
        .reset_index(drop=True)
    )
    cells["on_elpd_panel"] = cells["key"].isin(panel_keys["density"])
    cells["on_crps_panel"] = cells["key"].isin(panel_keys["draws"])

    pointwise = (
        long.loc[
            long["key"].isin(union_keys),
            [
                "model",
                *key_columns,
                "cohort",
                "key",
                "elpd",
                "crps",
                "ess_kish",
                "n_draws_density",
                "n_draws_sample",
            ],
        ]
        .sort_values(["model", *key_columns], kind="stable")
        .reset_index(drop=True)
    )
    pointwise["on_elpd_panel"] = pointwise["key"].isin(panel_keys["density"])
    pointwise["on_crps_panel"] = pointwise["key"].isin(panel_keys["draws"])

    # -- what fell out ---------------------------------------------------------
    dropped_rows = []
    for axis, column in CAPABILITIES.items():
        eligible = members[axis]
        if not eligible:
            continue
        reach: set[tuple] = set().union(*(offering[axis][m] for m in eligible))
        for key in sorted(reach - panel_keys[axis], key=repr):
            lacking = [m for m in eligible if key not in offering[axis][m]]
            dropped_rows.append(
                {
                    "score": column,
                    **dict(zip(key_columns, key, strict=True)),
                    "missing_from": tuple(lacking),
                }
            )
    dropped = pd.DataFrame(dropped_rows, columns=["score", *key_columns, "missing_from"])

    absences = pd.DataFrame(
        [
            {
                "model": f.model,
                "cohort": f.cohort,
                "field": f.field,
                "axis": axis,
                "score": CAPABILITIES[axis],
                "reason": f.absence_for(axis).reason,
                "scope": "model" if f.absence_for(axis).is_model_level else "cohort",
                "detail": f.absence_for(axis).detail,
                "n_cells": f.n_cells,
            }
            for f in items
            for axis in CAPABILITIES
            if f.absence_for(axis) is not None
        ],
        columns=[
            "model",
            "cohort",
            "field",
            "axis",
            "score",
            "reason",
            "scope",
            "detail",
            "n_cells",
        ],
    )

    excluded = pd.DataFrame(
        [
            {"cohort": cohort, **f.cells.exclusion_counts(), "n_scorable": f.n_cells}
            for cohort, f in sorted(by_cohort_cells.items(), key=repr)
        ],
        columns=["cohort", *EXCLUSION_REASONS, "n_scorable"],
    )

    by_cohort = _by_cohort(pointwise)
    coverage = pd.DataFrame(
        [
            _coverage_row(model, items, members, model_level, offering, panel_keys)
            for model in all_models
        ]
    )

    return ForecastPanel(
        task=task,
        as_of=as_of,
        segments=segments,
        units=units,
        elpd_members=members["density"],
        crps_members=members["draws"],
        elpd_fingerprint=_fingerprint(sorted(panel_keys["density"], key=repr)),
        crps_fingerprint=_fingerprint(sorted(panel_keys["draws"], key=repr)),
        cells=cells,
        pointwise=pointwise,
        by_cohort=by_cohort,
        coverage=coverage,
        absences=absences,
        dropped=dropped,
        excluded=excluded,
    )


def _by_cohort(pointwise: pd.DataFrame) -> pd.DataFrame:
    """Per (model, cohort) totals, each score over its OWN panel.

    The unit any future clustered standard error or stacking objective reads.
    Each score is summed only over the cells on its own panel, so the two columns
    of a row can legitimately rest on different cell counts - which is why both
    counts are carried rather than one shared ``n_cells``.
    """
    out = []
    for (model, cohort), group in pointwise.groupby(["model", "cohort"], sort=True):
        row = {"model": model, "cohort": cohort}
        for column, flag in (("elpd", "on_elpd_panel"), ("crps", "on_crps_panel")):
            live = group.loc[group[flag] & group[column].notna(), column]
            row[column] = float(live.sum()) if len(live) else pd.NA
            row[f"n_cells_{column}"] = int(len(live))
        out.append(row)
    return pd.DataFrame(
        out, columns=["model", "cohort", "elpd", "n_cells_elpd", "crps", "n_cells_crps"]
    )


def _coverage_row(
    model: str,
    items: list[CohortForecast],
    members: dict[str, tuple[str, ...]],
    model_level: dict[str, dict[str, Absence]],
    offering: dict[str, dict[str, set[tuple]]],
    panel_keys: dict[str, set[tuple]],
) -> dict:
    """One model's census, per capability."""
    mine = [f for f in items if f.model == model]
    row: dict = {"model": model, "n_cohorts_offered": len(mine)}
    for axis, column in CAPABILITIES.items():
        own = offering[axis][model]
        kept = own & panel_keys[axis]
        if model in model_level[axis]:
            status = f"na: {model_level[axis][model].reason}"
        elif not own:
            status = "na: no_scored_cohorts"
        elif model not in members[axis]:
            status = "na: not_a_member"
        else:
            status = "scored"
        row[f"is_{column}_member"] = model in members[axis]
        row[f"{column}_status"] = status
        row[f"n_cohorts_{column}"] = sum(1 for f in mine if f.offers(axis))
        row[f"n_cells_{column}_own"] = len(own)
        row[f"n_cells_{column}_on_panel"] = len(kept)
        row[f"n_cells_{column}_dropped"] = len(own) - len(kept)
    return row


def leaderboard(panel: ForecastPanel) -> pd.DataFrame:
    """The board. One row per model, ordered by model name. No sort key.

    **There is no ``sort_by=`` argument and no default sort.** Held-out ELPD and
    held-out CRPS are published side by side and the reader picks (decided
    2026-07-25). A sort parameter would move that choice from the reader into
    this function, and whichever value people typed first would become the house
    ranking by habit. Rows come back in model-name order so a published CSV is
    byte-reproducible; ``DataFrame.sort_values`` is one call away, and
    :data:`SCORE_DIRECTION` says which way each column runs so a renderer cannot
    guess wrong.

    Columns:

    **The two score columns can rest on different cells**, because a density and
    draws are independent capabilities with independent panels. Each therefore
    carries its own ``n_cells``, member list and fingerprint, and there is
    deliberately no bare ``n_cells`` column for a reader to attach to the wrong
    one.

    ``task``, ``as_of``           the protocol and the training cutoff
    ``model``                     entry name
    ``n_cohorts``                 cohorts in the union of the two panels
    ``elpd_members``              the density-eligible models the ELPD panel was
                                  intersected over. **The ELPD column is a
                                  function of this set**, so it travels with it,
                                  the same discipline ``mart_publish_id`` gets
    ``elpd_fingerprint``          sha256 of the sorted ELPD panel keys
    ``n_cells_elpd``              the ELPD panel - identical on every row
    ``elpd``                      SUM of the pointwise log predictive density
                                  over the ELPD panel. ``pd.NA``, never 0, when
                                  unavailable
    ``elpd_per_cell``             ``elpd / n_cells_elpd``, for reading. It cannot
                                  change a ranking (``n`` is the same for every
                                  row) and it is NOT a repair for a coverage
                                  difference - it is here so a reader can tell
                                  0.05 from 0.5 on a total near -14,000
    ``elpd_status``               ``scored``, or ``na: <reason>``
    ``n_cells_zero_density``      cells whose pointwise ELPD is ``-inf``. A model
                                  with ``elpd = -inf`` must show WHY on the same
                                  row, or the total reads as a bug rather than a
                                  verdict
    ``min_ess_kish``              smallest effective draw count behind any cell
                                  **on the ELPD panel**. Every per-score readout
                                  is taken from that score's own slice, never
                                  from the union - a cell on the CRPS panel only
                                  would otherwise describe a column it is not in
    ``crps_members``,
    ``crps_fingerprint``,
    ``n_cells_crps``              the same three for the CRPS panel, which is a
                                  DIFFERENT set of cells
    ``crps``                      SUM of the pointwise CRPS over the CRPS panel.
                                  Lower is better - the opposite direction to
                                  ``elpd``, which is why :data:`SCORE_DIRECTION`
                                  is machine-readable
    ``crps_per_cell``, ``crps_status``   as above
    ``n_draws_density_min``,
    ``n_draws_sample_min``        smallest draw count behind any of this model's
                                  cohorts, per capability. ``logmeanexp`` is
                                  biased downward by about ``cv2/(2S)``, always
                                  in the same direction, and sample CRPS is
                                  biased by ``O(1/S)`` too, so a model that never
                                  escalated through the harness's sampler stages
                                  scores worse for a reason that is not the model
    ``n_cells_elpd_own``,
    ``n_cells_elpd_dropped``,
    ``n_cells_crps_own``,
    ``n_cells_crps_dropped``      this model's own coverage per score, and what
                                  the intersection cost it

    **The totals are reduced in Python from each model's own array, not with a
    pandas aggregation.** ``groupby().sum()`` over an all-missing group returns
    ``0.0`` - measured here on pandas 2.3.3, and a nullable ``Float64`` dtype
    does not change it, only ``min_count=1`` does. On this board 0.0 is the best
    ELPD there is and the best CRPS there is, simultaneously. Making the
    aggregation unreachable beats guarding it.

    **There is no standard-error column, and no pairwise ``elpd_diff``.** The SE
    needs a decision this module must not make silently: the honest clustering
    unit is the FIT, and the fit is not the cohort for every entry - ``sur`` and
    ``copula_glm`` fit once per company across its lines, and the NN entries fit
    once over the whole panel, so a cohort-clustered SE would be wrong for four
    of the candidate entries in the same direction (too small).
    ``panel.by_cohort`` is the input any such estimator will take.

    **There is no KS / PIT column.** The critical value ``1.36/sqrt(n)`` assumes
    independence: nine cells of one cohort share one posterior and one calendar
    year, so at cell level that value is about three times too tight and models
    would be flagged as miscalibrated when they are not. It is also a different
    estimand from the published milestone-3 KS, which is one PIT per cohort on
    the run-off-to-ultimate predictive.
    """
    n_elpd = panel.n_cells_for("elpd")
    n_crps = panel.n_cells_for("crps")
    board = []
    for row in panel.coverage.to_dict("records"):
        model = row["model"]
        mine = panel.pointwise[panel.pointwise["model"] == model]
        # ``mine`` spans the UNION of the two panels, so every per-score readout
        # below - the totals AND the diagnostics - is taken from that score's own
        # slice. Reading a diagnostic off the union describes cells the column is
        # not computed over: a cell on the CRPS panel only, thinned to 2 draws,
        # would report n_draws_min = 2 beside an ELPD that rests on 100.
        on_elpd = mine[mine["on_elpd_panel"]]
        on_crps = mine[mine["on_crps_panel"]]

        elpd_vals = on_elpd.loc[on_elpd["elpd"].notna(), "elpd"].to_numpy(dtype=float)
        has_elpd = row["is_elpd_member"] and len(elpd_vals) == n_elpd and n_elpd > 0

        crps_vals = on_crps.loc[on_crps["crps"].notna(), "crps"].to_numpy(dtype=float)
        has_crps = row["is_crps_member"] and len(crps_vals) == n_crps and n_crps > 0

        board.append(
            {
                "task": panel.task,
                "as_of": panel.as_of,
                "model": model,
                "n_cohorts": panel.n_cohorts,
                # -- ELPD, on its own panel --------------------------------------
                "elpd_members": ",".join(panel.elpd_members),
                "elpd_fingerprint": panel.elpd_fingerprint,
                "n_cells_elpd": n_elpd,
                "elpd": float(elpd_vals.sum()) if has_elpd else pd.NA,
                "elpd_per_cell": float(elpd_vals.sum() / n_elpd) if has_elpd else pd.NA,
                "elpd_status": row["elpd_status"],
                "n_cells_zero_density": int(np.isneginf(elpd_vals).sum()) if has_elpd else pd.NA,
                "min_ess_kish": (
                    float(on_elpd["ess_kish"].dropna().min())
                    if has_elpd and on_elpd["ess_kish"].notna().any()
                    else pd.NA
                ),
                # -- CRPS, on ITS own panel, which is not the same one -----------
                "crps_members": ",".join(panel.crps_members),
                "crps_fingerprint": panel.crps_fingerprint,
                "n_cells_crps": n_crps,
                "crps": float(crps_vals.sum()) if has_crps else pd.NA,
                "crps_per_cell": float(crps_vals.sum() / n_crps) if has_crps else pd.NA,
                "crps_status": row["crps_status"],
                # -- draw counts and this model's own coverage -------------------
                "n_draws_density_min": _least(on_elpd["n_draws_density"], has_elpd),
                "n_draws_sample_min": _least(on_crps["n_draws_sample"], has_crps),
                "n_cells_elpd_own": row["n_cells_elpd_own"],
                "n_cells_elpd_dropped": row["n_cells_elpd_dropped"],
                "n_cells_crps_own": row["n_cells_crps_own"],
                "n_cells_crps_dropped": row["n_cells_crps_dropped"],
            }
        )
    out = pd.DataFrame(board)
    for column in ("elpd", "elpd_per_cell", "crps", "crps_per_cell", "min_ess_kish"):
        out[column] = out[column].astype("Float64")
    for column in ("n_cells_zero_density", "n_draws_density_min", "n_draws_sample_min"):
        out[column] = out[column].astype("Int64")
    return out.sort_values("model", kind="stable").reset_index(drop=True)


def _least(column: pd.Series, live: bool):
    """Smallest non-missing value, or ``pd.NA``. Never 0 from an empty column."""
    if not live:
        return pd.NA
    present = column.dropna()
    return int(present.min()) if len(present) else pd.NA
