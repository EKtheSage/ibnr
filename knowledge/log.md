# Knowledge Update Log

## 2026-10-12
* **Update**: [tlrn on a Colab TPU](/findings/tlrn-jax-tpu-run.md) now records the accident-year fits on a Colab GPU (451 s and 434 s for 80 members each, about 70 times faster than the laptop's one fit), replacing the cut-off TPU estimate, and adds the GPU-or-TPU runtime lesson.
* **Update**: [Notebook 04 results](/findings/notebook-04-results.md) gained the 0.7.3 run (1,438 minutes) with the accident-year variant: `tlrn_ay` scores Pool_APE 0.0622, behind mack (0.0577) and MCL (0.0566), while `tlrn_13_blend` stays best at 0.0516. [How long the long runs take](/findings/how-long-runs-take.md) gained the run and its per-fit times.
* **Creation**: Recorded [tlrn on a Colab TPU](/findings/tlrn-jax-tpu-run.md): the first TPU timing of the JAX backend (618 s for 40 members, 32 to 48 times faster than the laptop), how its members compare with torch's, and the one-process-per-TPU and Python 3.13 lessons.

## 2026-10-08
* **Creation**: Recorded [tlrn gets an optional JAX training backend](/decisions/tlrn-jax-training-backend.md) and its evidence, [the JAX training backend matches torch](/findings/tlrn-jax-backend-parity.md), including why Adam lets the two backends part where a true gradient is zero.
* **Update**: [JAX versus torch on a CPU](/findings/jax-versus-torch-on-cpu.md) no longer says no backend was built; it links the accelerator-only one.
* **Update**: After review, [the JAX training backend matches torch](/findings/tlrn-jax-backend-parity.md) records the Adam bias-correction fix (each update had been 6.5e-6 off torch's) and the weight-decay tests, and [the backend decision](/decisions/tlrn-jax-training-backend.md) notes that the Colab runner reuses a saved fit only under the same settings.

## 2026-10-05
* **Initialization**: Created this bundle at Ethan's request ([Knowledge is kept in OKF format](/decisions/knowledge-in-okf-format.md)).
* **Creation**: Recorded the tlrn work of 0.7.3: [design choices as components](/decisions/tlrn-choices-as-named-components.md), [CPU training throughput](/findings/tlrn-cpu-training-throughput.md), [JAX versus torch on a CPU](/findings/jax-versus-torch-on-cpu.md) and [reproducing the companion notebook](/findings/companion-notebook-reproduction.md).
* **Creation**: Recorded two gotchas found while releasing 0.7.3 and re-running notebook 04: [spawned workers re-run the script](/gotchas/spawn-workers-rerun-the-script.md) and [the release gate](/gotchas/release-gate-wants-green-runs-on-the-tagged-commit.md).
* **Creation**: Added the [notebook 04 playbook](/playbooks/rerun-notebook-04.md) and two references.
* **Creation**: Added the gotcha [a new file in scripts/ needs a compute-image decision](/gotchas/new-scripts-need-a-compute-image-decision.md), found when this bundle's own check script failed the lean CI legs.
* **Update**: Settled the [release gate gotcha](/gotchas/release-gate-wants-green-runs-on-the-tagged-commit.md): the cancelled test runs were a hosted-runner outage, and v0.7.3 published to PyPI after the failed jobs were re-run and the tag pushed again.
* **Migration**: Moved the project knowledge from the agent's private memory notes into this bundle at Ethan's request: 40 concepts across findings, decisions, gotchas and playbooks, listed in the directory indexes. The long status log became concepts by subject; pull request and temporary-folder bookkeeping, session mechanics and facts that CLAUDE.md already records in full were left out. Each concept was checked against its source note and one added claim was removed. Notes about how Ethan wants the agent to work stay in private memory.
