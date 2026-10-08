# Knowledge Update Log

## 2026-10-08
* **Creation**: Recorded [tlrn gets an optional JAX training backend](/decisions/tlrn-jax-training-backend.md) and its evidence, [the JAX training backend matches torch](/findings/tlrn-jax-backend-parity.md), including why Adam lets the two backends part where a true gradient is zero.
* **Update**: [JAX versus torch on a CPU](/findings/jax-versus-torch-on-cpu.md) no longer says no backend was built; it links the accelerator-only one.

## 2026-10-05
* **Initialization**: Created this bundle at Ethan's request ([Knowledge is kept in OKF format](/decisions/knowledge-in-okf-format.md)).
* **Creation**: Recorded the tlrn work of 0.7.3: [design choices as components](/decisions/tlrn-choices-as-named-components.md), [CPU training throughput](/findings/tlrn-cpu-training-throughput.md), [JAX versus torch on a CPU](/findings/jax-versus-torch-on-cpu.md) and [reproducing the companion notebook](/findings/companion-notebook-reproduction.md).
* **Creation**: Recorded two gotchas found while releasing 0.7.3 and re-running notebook 04: [spawned workers re-run the script](/gotchas/spawn-workers-rerun-the-script.md) and [the release gate](/gotchas/release-gate-wants-green-runs-on-the-tagged-commit.md).
* **Creation**: Added the [notebook 04 playbook](/playbooks/rerun-notebook-04.md) and two references.
* **Creation**: Added the gotcha [a new file in scripts/ needs a compute-image decision](/gotchas/new-scripts-need-a-compute-image-decision.md), found when this bundle's own check script failed the lean CI legs.
* **Update**: Settled the [release gate gotcha](/gotchas/release-gate-wants-green-runs-on-the-tagged-commit.md): the cancelled test runs were a hosted-runner outage, and v0.7.3 published to PyPI after the failed jobs were re-run and the tag pushed again.
* **Migration**: Moved the project knowledge from the agent's private memory notes into this bundle at Ethan's request: 40 concepts across findings, decisions, gotchas and playbooks, listed in the directory indexes. The long status log became concepts by subject; pull request and temporary-folder bookkeeping, session mechanics and facts that CLAUDE.md already records in full were left out. Each concept was checked against its source note and one added claim was removed. Notes about how Ethan wants the agent to work stay in private memory.
