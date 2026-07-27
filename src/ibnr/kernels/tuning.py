"""Random-search hyperparameter optimization for the NN gallery entries.

Per decision 5 the search machinery lives ONCE in kernels and the entries call
it; per the NN overfitting gotcha the objective is the entry's own eval_date
validation NLL - the trailing-diagonal holdout ``fit()`` already early-stops
on - never a test-period quantity. :func:`validation_score` reads it straight
off the fitted entry's ``history_``: the per-member minimum finite ``val``
over epochs (the early-stopping best, whose weights ``fit()`` restores),
averaged over ensemble members. Lower is better.

The search itself is plain random search over a small vocabulary of parameter
distributions (:class:`Choice`, :class:`Uniform`, :class:`LogUniform`,
:class:`IntChoice`, :class:`IntLogUniform`), dependency-free and torch-free at
module level. Trials run SEQUENTIALLY on purpose: one NN fit already saturates
the machine (data-parallel batches, all-core BLAS), so a trial pool would just
thrash - parallelism belongs inside the fit, not across trials. Smarter
schedules (successive halving / hyperband) are a deliberate extension point:
they slot in as an alternative to :func:`tune`'s loop over the same
``fit_trial`` callable and :class:`TuningResult`, with a budget argument added
to the callable's signature - nothing in the distributions or the result type
assumes the trial count is fixed up front.

Reproducibility contract: ``tune(seed=s)`` draws every trial's parameters from
one ``np.random.default_rng(s)`` stream BEFORE running it, so the parameter
sequence is a function of (seed, space insertion order) alone - a failing
trial cannot perturb its successors' parameters. Trial ``i`` fits with
``trial_seed = s + 10_000 * (i + 1)`` - offset so no trial fits with the bare
``s``, which is the very seed the tuner's own parameter stream is built from
(and the seed ``fit()``'s member 0 would reuse as ``default_rng(s)``) -
echoing the NN family's widely-spaced member seeds (``seed + 1000 * member``
inside ``fit()``), so no two trials' member streams collide for ensembles up
to 10 members.
"""

from __future__ import annotations

import abc
import dataclasses
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Choice",
    "Dist",
    "IntChoice",
    "IntLogUniform",
    "LogUniform",
    "TuningResult",
    "Uniform",
    "sample",
    "tune",
    "tune_entry",
    "validation_score",
]

# -- parameter distributions ---------------------------------------------------


class Dist(abc.ABC):
    """One hyperparameter's search distribution."""

    @abc.abstractmethod
    def sample(self, rng: np.random.Generator):
        """Draw one value using ``rng`` (and nothing else)."""


@dataclass(frozen=True)
class Choice(Dist):
    """Uniform over an explicit set of values (categorical or mixed-type).

    Values are returned as the exact Python objects handed in - never routed
    through a numpy array, which would silently coerce a mixed-type set.
    """

    values: tuple

    def __post_init__(self) -> None:
        if isinstance(self.values, str) or not isinstance(self.values, Sequence):
            raise TypeError(f"Choice takes a sequence of values, got {self.values!r}")
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError("Choice needs at least one value")

    def sample(self, rng: np.random.Generator):
        return self.values[int(rng.integers(len(self.values)))]


@dataclass(frozen=True)
class IntChoice(Choice):
    """A :class:`Choice` whose values must all be Python ints (no bools)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for v in self.values:
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"IntChoice values must be ints, got {v!r}")

    def sample(self, rng: np.random.Generator) -> int:
        return int(super().sample(rng))


@dataclass(frozen=True)
class Uniform(Dist):
    """Uniform float on ``[lo, hi)``."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not self.lo < self.hi:
            raise ValueError(f"Uniform needs lo < hi, got [{self.lo}, {self.hi})")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))


@dataclass(frozen=True)
class LogUniform(Dist):
    """Log-uniform float on ``[lo, hi)`` - uniform in ``log(x)``, the right
    scale for learning rates and weight decays."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo <= 0:
            raise ValueError(f"LogUniform needs lo > 0, got {self.lo}")
        if not self.lo < self.hi:
            raise ValueError(f"LogUniform needs lo < hi, got [{self.lo}, {self.hi})")

    def sample(self, rng: np.random.Generator) -> float:
        return float(math.exp(rng.uniform(math.log(self.lo), math.log(self.hi))))


@dataclass(frozen=True)
class IntLogUniform(Dist):
    """Log-uniform rounded to int on ``[lo, hi]`` - for sizes spanning orders
    of magnitude (d_model, ffn_dim)."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        for name, v in (("lo", self.lo), ("hi", self.hi)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"IntLogUniform {name} must be an int, got {v!r}")
        if self.lo < 1:
            raise ValueError(f"IntLogUniform needs lo >= 1, got {self.lo}")
        if not self.lo < self.hi:
            raise ValueError(f"IntLogUniform needs lo < hi, got [{self.lo}, {self.hi}]")

    def sample(self, rng: np.random.Generator) -> int:
        raw = math.exp(rng.uniform(math.log(self.lo), math.log(self.hi)))
        return int(min(max(round(raw), self.lo), self.hi))


def sample(space: Mapping[str, Dist], rng: np.random.Generator) -> dict:
    """Draw one parameter dict from ``space``, consuming ``rng`` once per
    entry in insertion order (so the draw sequence is a pure function of the
    rng state and the space's key order)."""
    out = {}
    for name, dist in space.items():
        if not isinstance(dist, Dist):
            raise TypeError(
                f"space[{name!r}] is {type(dist).__name__}, not a Dist "
                "(Choice/IntChoice/Uniform/LogUniform/IntLogUniform)"
            )
        out[name] = dist.sample(rng)
    return out


# -- random search -------------------------------------------------------------

#: trial-bookkeeping columns of ``TuningResult.trials``; a space key may not
#: shadow one, or its sampled values would silently overwrite the bookkeeping
RESERVED_COLUMNS = ("trial", "score", "seed", "seconds", "error")


@dataclass(frozen=True, eq=False)
class TuningResult:
    """Everything one :func:`tune` run learned.

    ``trials`` has one row per trial in execution order: ``trial``, one column
    per space key (the sampled params, flattened), ``score``, ``seed``,
    ``seconds``, ``error`` (None for a successful trial, the exception text
    for a failed one; failed trials carry ``score`` NaN). ``best_params`` /
    ``best_score`` are the lowest-scoring successful trial - the objective is
    a validation NLL, so lower is better.
    """

    trials: pd.DataFrame
    best_params: dict
    best_score: float
    #: default report size for :meth:`top`
    keep_top: int = 5

    def top(self, k: int | None = None) -> pd.DataFrame:
        """The ``k`` (default ``keep_top``) best successful trials, ascending
        by score; fewer rows if fewer trials succeeded."""
        k = self.keep_top if k is None else k
        ok = self.trials[self.trials["error"].isna()]
        return ok.sort_values("score", kind="mergesort").head(k).reset_index(drop=True)


def tune(
    fit_trial: Callable[[dict, int], float],
    space: Mapping[str, Dist],
    *,
    n_trials: int,
    seed: int,
    keep_top: int = 5,
) -> TuningResult:
    """Random search: minimize ``fit_trial(params, trial_seed)`` over
    ``n_trials`` draws from ``space``.

    Trials run sequentially (see the module docstring for why) and a trial
    that raises is recorded as failed - the NN family legitimately fails on
    some configs (e.g. n_heads not dividing d_model) and one such region must
    not kill the run. A non-finite return (NaN/inf) is also recorded as a
    failure: NaN poisons ranking silently and ``-inf`` would be an
    unbeatable score, so neither may ever become ``best_score``. Raises
    ``RuntimeError`` only when every trial failed.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    for name in space:
        if name in RESERVED_COLUMNS:
            raise ValueError(
                f"space key {name!r} collides with a trials-frame bookkeeping "
                f"column; reserved: {RESERVED_COLUMNS}"
            )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    best_score, best_params = math.inf, None
    for i in range(n_trials):
        # sampled BEFORE the fit, from the one stream: the parameter sequence
        # depends only on (seed, space order), never on trial outcomes
        params = sample(space, rng)
        # (i + 1): trial 0 must not fit with the bare seed, which already
        # names the tuner's own parameter stream (module docstring)
        trial_seed = seed + 10_000 * (i + 1)
        start = time.perf_counter()
        error: str | None = None
        try:
            score = float(fit_trial(params, trial_seed))
            if not math.isfinite(score):
                error = f"fit_trial returned a non-finite score: {score}"
                score = math.nan
        except Exception as exc:
            score = math.nan
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start
        rows.append(
            {
                "trial": i,
                **params,
                "score": score,
                "seed": trial_seed,
                "seconds": seconds,
                "error": error,
            }
        )
        if error is None and score < best_score:
            best_score, best_params = score, dict(params)

    if best_params is None:
        first = rows[0]["error"]
        raise RuntimeError(f"all {n_trials} trials failed; first error: {first}")
    columns = ["trial", *space, "score", "seed", "seconds", "error"]
    trials = pd.DataFrame(rows, columns=columns)
    return TuningResult(
        trials=trials, best_params=best_params, best_score=best_score, keep_top=keep_top
    )


# -- gallery glue --------------------------------------------------------------


def validation_score(entry) -> float:
    """The fitted entry's eval_date validation NLL: per ensemble member the
    minimum FINITE ``val`` over epochs (the early-stopping best, whose weights
    ``fit()`` restores), averaged over members. Lower is better.

    Reads the ``history_`` every NN entry records (one list of
    ``{"epoch", "train", "val"}`` dicts per member); never a test-period
    quantity, so tuning on it cannot leak the backtest.

    NaN epochs are skipped, and order-independently so - a plain ``min`` would
    score ``[nan, 3.0]`` and ``[3.0, nan]`` differently, the same multiset
    with opposite verdicts. Skipping matches what ``fit()`` restored: its
    early stopping never snapshots a NaN val, so the best finite epoch is the
    one the weights correspond to. A member whose history is ALL NaN scores
    NaN (its all-NaN RuntimeWarning suppressed here as noise), which
    ``tune()``'s non-finite quarantine then records as a failed trial.
    """
    if not hasattr(entry, "history_"):
        raise TypeError(
            f"{type(entry).__name__} records no history_; validation_score "
            "reads the per-epoch validation NLL that only the NN entries track"
        )
    history = entry.history_
    if not history:
        raise ValueError(f"{type(entry).__name__}.history_ is empty or None; fit the entry first")
    per_member = []
    for m, member_history in enumerate(history):
        if not member_history:
            raise ValueError(f"member {m} has an empty epoch history")
        vals = np.asarray([rec["val"] for rec in member_history], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN member -> NaN, quarantined
            per_member.append(float(np.nanmin(vals)))
    return float(np.mean(per_member))


def tune_entry(
    entry_name: str,
    triangle,
    space: Mapping[str, Dist],
    *,
    base_config,
    n_trials: int,
    seed: int,
    keep_top: int = 5,
    fit_kwargs: Mapping | None = None,
) -> TuningResult:
    """Random search over an NN gallery entry's config dataclass.

    Each trial fits ``entry_name`` on ``triangle`` with
    ``dataclasses.replace(base_config, **params)`` and ``seed=trial_seed``,
    scored by :func:`validation_score`. ``fit_kwargs`` (loss_field, as_of,
    device, ...) pass through to ``gallery.fit`` verbatim on every trial;
    ``config`` and ``seed`` are tune_entry's own to pass and may not appear
    there. Space keys must name fields of ``base_config`` - checked up front,
    since a typo would otherwise fail every trial identically.
    """
    if not dataclasses.is_dataclass(base_config) or isinstance(base_config, type):
        raise TypeError(f"base_config must be a config dataclass instance, got {base_config!r}")
    known = {f.name for f in dataclasses.fields(base_config)}
    unknown = sorted(set(space) - known)
    if unknown:
        raise ValueError(f"space keys {unknown} are not fields of {type(base_config).__name__}")
    kwargs = dict(fit_kwargs or {})
    if overlap := ({"config", "seed"} & set(kwargs)):
        raise ValueError(
            f"fit_kwargs may not carry {sorted(overlap)}: tune_entry builds config "
            "from base_config and passes each trial's own seed"
        )

    def fit_trial(params: dict, trial_seed: int) -> float:
        # lazy on purpose: kernels never imports gallery at module scope
        # (gallery imports kernels; the graph must stay acyclic)
        from ibnr import gallery

        config = dataclasses.replace(base_config, **params)
        entry = gallery.fit(entry_name, triangle, config=config, seed=trial_seed, **kwargs)
        return validation_score(entry)

    return tune(fit_trial, space, n_trials=n_trials, seed=seed, keep_top=keep_top)
