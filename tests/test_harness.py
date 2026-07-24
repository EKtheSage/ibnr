"""kernels.harness: convergence gates, staged escalation, settings filtering.

The orchestration is exercised with an injected stub runner and the serial
executor, so none of this needs cmdstan (or the mart). Real parallel
end-to-end fits are covered by the validation scripts (`-m slow` territory).
"""

from __future__ import annotations

import inspect

import pytest

from ibnr.kernels.harness import (
    ConvergenceGates,
    RetroTask,
    SamplerSettings,
    run_retro,
)

AS_OF = "1997-12-31"


def make_task(code: str) -> RetroTask:
    return RetroTask(model="stub", warehouse="unused", line="wc", company_code=code, as_of=AS_OF)


class ScriptedRunner:
    """Stub runner returning scripted diagnostics per (company, attempt)."""

    def __init__(self, script: dict[str, list[dict]]):
        self.script = script
        self.calls: list[tuple[str, SamplerSettings]] = []

    def __call__(self, task: RetroTask, settings: SamplerSettings) -> dict:
        attempt = sum(1 for code, _ in self.calls if code == task.company_code)
        self.calls.append((task.company_code, settings))
        row = {"model": task.model, "line": task.line, "company_code": task.company_code}
        row.update(self.script[task.company_code][attempt])
        return row


GOOD = {"max_rhat": 1.001, "divergence_frac": 0.0, "min_ess_bulk": 900.0, "percentile": 50.0}
BAD_RHAT = {"max_rhat": 1.2, "divergence_frac": 0.0, "min_ess_bulk": 900.0, "percentile": 50.0}


# -- gates ---------------------------------------------------------------------


def test_gates_fail_on_each_diagnostic():
    gates = ConvergenceGates(max_rhat=1.05, max_divergence_frac=0.002, min_ess_bulk=100.0)
    assert gates.passes(GOOD)
    assert not gates.passes({**GOOD, "max_rhat": 1.06})
    assert not gates.passes({**GOOD, "divergence_frac": 0.01})
    assert not gates.passes({**GOOD, "min_ess_bulk": 50.0})
    assert not gates.passes({"error": "boom"})


def test_gates_pass_on_unknown_diagnostics():
    """None/NaN/absent diagnostics never trip a gate: a likelihood-based entry
    with no sampler has nothing an escalated sampler could fix."""
    gates = ConvergenceGates()
    assert gates.passes({})
    assert gates.passes({"max_rhat": None, "divergence_frac": None, "min_ess_bulk": None})
    assert gates.passes({"max_rhat": float("nan"), "min_ess_bulk": float("nan")})


# -- SamplerSettings.fit_kwargs ------------------------------------------------


def test_fit_kwargs_filters_to_signature():
    def mcmc_fit(self, triangle, *, chains, iter_warmup, iter_sampling, target_accept=0.8):
        pass

    s = SamplerSettings(chains=2, iter_warmup=500, iter_sampling=100, parallel_chains=4)
    kwargs = s.fit_kwargs(mcmc_fit)
    # parallel_chains not in the signature -> dropped; target_accept None ->
    # "use the entry default" -> not passed even though accepted
    assert kwargs == {"chains": 2, "iter_warmup": 500, "iter_sampling": 100}

    s2 = SamplerSettings(target_accept=0.99, max_treedepth=15)
    assert s2.fit_kwargs(mcmc_fit)["target_accept"] == 0.99
    assert "max_treedepth" not in s2.fit_kwargs(mcmc_fit)

    def mle_fit(self, triangle, *, loss_field="paid_loss"):
        pass

    assert s2.fit_kwargs(mle_fit) == {}  # non-MCMC entry ignores all of it


def test_stan_entries_accept_every_sampler_setting():
    """Every registered MCMC entry's fit() must accept the full settings
    vocabulary, or escalation silently degrades to a partial re-run."""
    gallery = pytest.importorskip("ibnr.gallery")
    for name in (
        "meyers_ccl",
        "meyers_csr",
        "england_verrall_odp",
        "clark_growth_curve",
        "compartmental",
    ):
        params = inspect.signature(gallery.get(name).fit).parameters
        for key in (
            "chains",
            "iter_warmup",
            "iter_sampling",
            "parallel_chains",
            "target_accept",
            "max_treedepth",
            "seed",
        ):
            assert key in params, f"{name}.fit() lacks {key}"
        assert hasattr(gallery.get(name), "precompile"), f"{name} lacks precompile()"


# -- run_retro escalation ------------------------------------------------------

STAGE1 = SamplerSettings(target_accept=0.9)
STAGE2 = SamplerSettings(target_accept=0.99, parallel_chains=4)


def test_escalation_refits_only_failures_and_keeps_order():
    runner = ScriptedRunner(
        {
            "A": [GOOD],
            "B": [BAD_RHAT, GOOD],  # fails stage 1, clean at stage 2
            "C": [{"error": "RuntimeError: sampler blew up"}, {"error": "again"}],
        }
    )
    rows = run_retro(
        [make_task("A"), make_task("B"), make_task("C")],
        stages=(STAGE1, STAGE2),
        executor="serial",
        runner=runner,
    )
    assert [r["company_code"] for r in rows] == ["A", "B", "C"]  # task order kept
    assert rows[0]["stage"] == 1 and rows[0]["max_rhat"] == 1.001
    # B's final row is the ESCALATED attempt - replaced, not duplicated
    assert rows[1]["stage"] == 2 and rows[1]["max_rhat"] == 1.001
    # errors escalate too (the raise may have been a sampler failure)
    assert rows[2]["stage"] == 2 and rows[2]["error"] == "again"
    # A ran once; B and C ran at stage-1 then stage-2 settings
    assert [c for c, _ in runner.calls] == ["A", "B", "C", "B", "C"]
    assert [s.target_accept for _, s in runner.calls] == [0.9, 0.9, 0.9, 0.99, 0.99]


def test_single_stage_never_refits():
    runner = ScriptedRunner({"A": [BAD_RHAT]})
    rows = run_retro([make_task("A")], stages=(STAGE1,), executor="serial", runner=runner)
    assert len(runner.calls) == 1
    assert rows[0]["stage"] == 1


def test_progress_callback_sees_every_attempt():
    seen = []
    runner = ScriptedRunner({"A": [GOOD], "B": [BAD_RHAT, GOOD]})

    def progress(row, stage, done, total):
        seen.append((row["company_code"], stage, done, total))

    run_retro(
        [make_task("A"), make_task("B")],
        stages=(STAGE1, STAGE2),
        executor="serial",
        runner=runner,
        progress=progress,
    )
    assert seen == [("A", 1, 1, 2), ("B", 1, 2, 2), ("B", 2, 1, 1)]


def test_empty_stages_rejected():
    with pytest.raises(ValueError, match="at least one"):
        run_retro([make_task("A")], stages=(), executor="serial", runner=ScriptedRunner({}))
