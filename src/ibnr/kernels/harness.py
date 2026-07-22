"""Parallel retrospective harness: fit + score many cohorts across a worker
pool, with staged sampler escalation.

This is the compute layer between a study script (``scripts/meyers_validation
.py``, ``scripts/compare_gallery.py``) and the gallery entries. A study
describes WHAT to run — a list of :class:`RetroTask` (model x line x company,
training cutoff, loss field) and a per-model escalation policy — and the
harness decides HOW: it fans the tasks over a process pool, gates every fit on
its convergence diagnostics, and re-fits only the failures at the next stage's
more expensive sampler settings. It is also the seam a future hosted API calls
(one request = one task list), which is why nothing in here knows about cohort
selection, CSV layout, or the KS test.

Parallelism model
-----------------
Cross-COMPANY process parallelism is the primary axis: a retro has far more
tasks than cores, and one cmdstan chain saturates one core, so ``chains``
running sequentially inside each of N workers beats N/4 workers running 4
parallel chains (same core-seconds, better packing, and it preserves the
entries' fair single-core runtime convention). Within-fit ``parallel_chains``
becomes worth it only in a LATE escalation stage, when the few surviving
tasks would otherwise leave cores idle — which is why it is a per-stage
:class:`SamplerSettings` field and the pool shrinks by that factor.

Workers are separate processes (spawn, so behavior is identical on Windows
and Linux): cmdstanpy shells out per fit and duckdb/ibis state must not be
shared across forks. Each worker re-reads the (locally cached) gold mart for
just its company — cheap next to any MCMC fit. Two parent-side preconditions
keep the pool boring: the warehouse spec handed to tasks must be CONCRETE
(``data.schedule_p.pinned_source``, so workers never resolve ``@latest`` or
call gh), and :func:`precompile` builds every Stan executable up front so
workers never race the compiler.
"""

from __future__ import annotations

import inspect
import math
import multiprocessing
import os
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

ENV_MAX_WORKERS = "IBNR_MAX_WORKERS"


def default_max_workers() -> int:
    """Worker-pool size when the caller does not choose one: the
    ``IBNR_MAX_WORKERS`` env var (the container-level control — inside a
    cpu-limited container ``os.cpu_count()`` still reports the HOST's cores,
    so an orchestrator should set this alongside ``--cpus``), else all
    schedulable CPUs, leaving one for the parent and the OS."""
    env = os.environ.get(ENV_MAX_WORKERS)
    if env:
        return max(1, int(env))
    count = getattr(os, "process_cpu_count", os.cpu_count)() or 2
    return max(1, count - 1)


@dataclass(frozen=True)
class SamplerSettings:
    """One escalation stage's MCMC budget.

    ``None`` means "the entry's own default" and is not passed at all, so a
    single default-constructed stage reproduces a plain ``gallery.fit()``.
    ``fit_kwargs`` filters against the entry's signature, so the same stage
    list can drive a study that mixes MCMC and likelihood-based entries (the
    latter simply ignore all of this).
    """

    chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 2500
    #: within-fit chain parallelism; the pool shrinks by this factor, so it
    #: only helps once a stage has fewer tasks than workers (see module doc)
    parallel_chains: int = 1
    target_accept: float | None = None
    max_treedepth: int | None = None

    def fit_kwargs(self, fit_fn: Callable) -> dict:
        """The subset of these settings that ``fit_fn`` actually accepts."""
        kwargs: dict = {
            "chains": self.chains,
            "iter_warmup": self.iter_warmup,
            "iter_sampling": self.iter_sampling,
            "parallel_chains": self.parallel_chains,
        }
        if self.target_accept is not None:
            kwargs["target_accept"] = self.target_accept
        if self.max_treedepth is not None:
            kwargs["max_treedepth"] = self.max_treedepth
        params = inspect.signature(fit_fn).parameters
        return {k: v for k, v in kwargs.items() if k in params}


@dataclass(frozen=True)
class ConvergenceGates:
    """Pass/fail thresholds on an entry's ``convergence()`` diagnostics.

    A fit failing any gate is re-run at the next stage's settings. Missing
    diagnostics (None/NaN — e.g. a likelihood-based entry with no sampler)
    pass by construction: there is nothing an escalated sampler would fix.
    Defaults: the model cards' R-hat 1.05 reporting convention, ~0 tolerance
    for divergences (0.002 of draws), and a min bulk ESS low enough to flag
    only genuinely stuck chains.
    """

    max_rhat: float = 1.05
    max_divergence_frac: float = 0.002
    min_ess_bulk: float = 100.0

    def passes(self, row: Mapping) -> bool:
        if row.get("error") is not None:
            return False  # a raise might be a sampler failure; retry escalated
        return not (
            _exceeds(row.get("max_rhat"), self.max_rhat)
            or _exceeds(row.get("divergence_frac"), self.max_divergence_frac)
            or _below(row.get("min_ess_bulk"), self.min_ess_bulk)
        )


def _known(value) -> float:
    """None/NaN -> NaN (an unknown diagnostic never trips a gate)."""
    return float("nan") if value is None else float(value)


def _exceeds(value, limit: float) -> bool:
    v = _known(value)
    return not math.isnan(v) and v > limit


def _below(value, limit: float) -> bool:
    v = _known(value)
    return not math.isnan(v) and v < limit


@dataclass(frozen=True)
class RetroTask:
    """One unit of work: fit one model to one company x line cohort as of a
    training cutoff, score it against the realized outcome.

    ``warehouse`` must already be concrete (a local path or a pinned
    ``github://owner/repo@publish_id`` with the mart cached) — see
    ``data.schedule_p.pinned_source``. ``fit_kwargs`` carries entry-specific
    arguments (``variant``, ``growth_curve``) verbatim: unlike sampler
    settings they are NOT signature-filtered, so a typo fails loudly as an
    error row instead of silently fitting the default.
    """

    model: str
    warehouse: str
    line: str
    company_code: str
    as_of: str
    loss_field: str | None = None  # None: the entry's own default
    #: apply Meyers' pmax(paid, 1) floor before fitting (lognormal-on-paid
    #: entries only — the study script owns the policy of who gets this)
    clamp_paid: bool = False
    seed: int | None = None
    fit_kwargs: dict = field(default_factory=dict)


#: diagnostics copied from ``entry.convergence()`` into every result row —
#: the gate inputs, plus tail ESS for post-hoc reading
_DIAG_KEYS = ("max_rhat", "min_ess_bulk", "min_ess_tail", "divergence_frac")


def run_task(task: RetroTask, settings: SamplerSettings) -> dict:
    """Fit + score one task: the worker function. Never raises — a failure
    comes back as an ``error`` row so the surrounding study always completes
    (the sequential harness's contract, kept)."""
    row: dict = {"model": task.model, "line": task.line, "company_code": task.company_code}
    t0 = time.perf_counter()
    try:
        import ibis

        from ibnr import gallery
        from ibnr.data.schedule_p import load_schedule_p

        tri = load_schedule_p(task.warehouse, lines=[task.line], companies=[task.company_code])
        if task.clamp_paid:
            e = tri.expr
            tri = tri.with_expr(
                e.mutate(
                    value=ibis.ifelse(e.field == "paid_loss", ibis.greatest(e.value, 1), e.value)
                )
            )
        cls = gallery.get(task.model)
        kwargs = settings.fit_kwargs(cls.fit)
        if "seed" in inspect.signature(cls.fit).parameters:
            kwargs["seed"] = task.seed
        if task.loss_field is not None:
            kwargs["loss_field"] = task.loss_field
        kwargs.update(task.fit_kwargs)
        entry = cls().fit(tri, as_of=task.as_of, **kwargs)
        pred = entry.predict(seed=task.seed)
        # outcomes come from the FULL triangle; realized_ultimates restricts to
        # the training origins (the post-study accident-year gotcha, CLAUDE.md)
        realized = entry.realized_ultimates(tri)
        total = pred.summary(observed=realized).iloc[-1]
        row.update(
            estimate=total["estimate"],
            se=total["se"],
            cv=total["cv"],
            outcome=total["outcome"],
            percentile=total["percentile"],
        )
        if hasattr(entry, "convergence"):
            diag = entry.convergence()
            row.update({k: diag.get(k) for k in _DIAG_KEYS})
    except Exception as e:  # noqa: BLE001 - the study must run to completion
        row["error"] = f"{type(e).__name__}: {e}"
    row["seconds"] = time.perf_counter() - t0
    return row


def precompile(models: Iterable[str] | None = None) -> None:
    """Compile the Stan program of every (given, else registered) entry that
    has one. Called before the pool spawns — concurrent first-fits would race
    the compiler over the same executable — and at container build time, so a
    container start never pays the compile."""
    from ibnr import gallery

    for name in sorted(models) if models is not None else gallery.list():
        pc = getattr(gallery.get(name), "precompile", None)
        if pc is not None:
            pc()


def run_retro(
    tasks: Iterable[RetroTask],
    *,
    stages: tuple[SamplerSettings, ...] = (SamplerSettings(),),
    gates: ConvergenceGates | None = None,
    max_workers: int | None = None,
    executor: str = "process",
    runner: Callable[[RetroTask, SamplerSettings], dict] = run_task,
    progress: Callable[[dict, int, int, int], None] | None = None,
) -> list[dict]:
    """Run every task through the staged escalation, in parallel.

    Stage 1 fits all tasks at ``stages[0]``; each later stage re-fits only the
    tasks whose latest row fails ``gates``. A task's final row is its LAST
    attempt (stamped ``row["stage"]``), so escalation replaces, never
    duplicates. Rows come back in task order regardless of completion order.

    ``executor="serial"`` runs everything inline in this process — for tests,
    debugging (real tracebacks), and platforms where spawning is unwanted.
    ``runner`` is injectable for the same reason. ``progress`` (if given) is
    called as ``progress(row, stage, done_in_stage, total_in_stage)`` from the
    parent as each fit lands.
    """
    tasks = list(tasks)
    if gates is None:
        gates = ConvergenceGates()
    if not stages:
        raise ValueError("need at least one SamplerSettings stage")
    if max_workers is None:
        max_workers = default_max_workers()
    if executor == "process" and runner is run_task:
        precompile({t.model for t in tasks})

    rows: list[dict | None] = [None] * len(tasks)
    pending = list(range(len(tasks)))
    for stage_no, settings in enumerate(stages, start=1):
        if not pending:
            break
        # a stage running k parallel chains per fit gets max_workers // k
        # workers, so total chain processes never oversubscribe the budget
        stage_workers = max(1, max_workers // max(1, settings.parallel_chains))
        stage_rows = _run_stage(
            [(i, tasks[i]) for i in pending],
            settings,
            stage_no=stage_no,
            max_workers=stage_workers,
            executor=executor,
            runner=runner,
            progress=progress,
        )
        for i, row in stage_rows.items():
            row["stage"] = stage_no
            rows[i] = row
        pending = [i for i in pending if not gates.passes(rows[i])]
    return rows  # type: ignore[return-value]  # every slot filled in stage 1


def _run_stage(
    items: list[tuple[int, RetroTask]],
    settings: SamplerSettings,
    *,
    stage_no: int,
    max_workers: int,
    executor: str,
    runner: Callable,
    progress: Callable | None,
) -> dict[int, dict]:
    out: dict[int, dict] = {}

    def record(i: int, row: dict) -> None:
        out[i] = row
        if progress is not None:
            progress(row, stage_no, len(out), len(items))

    if executor == "serial" or max_workers == 1 or len(items) == 1:
        for i, task in items:
            record(i, runner(task, settings))
        return out
    if executor != "process":
        raise ValueError(f"executor must be 'process' or 'serial', got {executor!r}")
    # spawn (not fork): identical semantics on Windows/Linux, and no inherited
    # duckdb/ibis/cmdstanpy state in the children
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(max_workers, len(items)), mp_context=ctx) as pool:
        futures = {pool.submit(runner, task, settings): (i, task) for i, task in items}
        for fut in as_completed(futures):
            i, task = futures[fut]
            exc = fut.exception()
            if exc is not None:
                # runner catches its own exceptions, so this is pool machinery
                # failing (unpicklable result, worker killed); still a row
                row = {
                    "model": task.model,
                    "line": task.line,
                    "company_code": task.company_code,
                    "error": f"worker: {type(exc).__name__}: {exc}",
                }
            else:
                row = fut.result()
            record(i, row)
    return out
