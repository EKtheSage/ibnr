---
okf_version: "0.2"
---

# Findings

* [tlrn CPU training throughput](findings/tlrn-cpu-training-throughput.md) - what one tlrn step costs, how the cost scales with threads and processes, and what a full run takes on the laptop.
* [JAX versus torch on a CPU](findings/jax-versus-torch-on-cpu.md) - the companion notebook's JAX code run on this CPU against ibnr's torch loop.
* [Reproducing the companion notebook](findings/marco-notebook-reproduction.md) - what ibnr's accident-year variant matched, what it did not, and the reduced-scale result.

# Decisions

* [tlrn design choices as named components](decisions/tlrn-choices-as-named-components.md) - why a network variant is a config and not a copy of the entry.
* [Knowledge is kept in OKF format](decisions/knowledge-in-okf-format.md) - where this bundle lives and the conventions it follows.

# Gotchas

* [Spawned workers re-run the script](gotchas/spawn-workers-rerun-the-script.md) - the main guard, and the name collision that broke a notebook-to-script run.
* [The release gate wants green runs on the tagged commit](gotchas/release-gate-wants-green-runs-on-the-tagged-commit.md) - why the first v0.7.3 tag published nothing.

# Playbooks

* [Re-run notebook 04](playbooks/rerun-notebook-04.md) - how long each stage takes and how to run it so a failure is cheap.

# References

* [The companion study's repository](references/transformers-reserving-repo.md) - where the notebook, its cached data and its published tables live.
* [Open Knowledge Format](references/okf-spec.md) - the format this bundle is written in.
