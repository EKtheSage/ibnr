---
type: Decision
title: tlrn entry scope and attribution
description: Ethan's decisions of 2026-09-20 on how the tlrn entry behaves by default, what it reports, how MCL joins, and why the reference study is not named in the repository.
tags: [tlrn, mcl, design, attribution, companion-study]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries dated 2026-09-20 to 2026-09-23 (lines 100-117 and 203-256)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: spec
    resource: docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md
    title: The approved design (with plans A, B, M and C under docs/superpowers/plans/)
---

# Decisions (Ethan, 2026-09-20)

* **Attribution stays neutral until the colleague's paper is out.** The colleague and the
  colleague's repository are never named in the ibnr repository. Vendored reference tables
  carry neutral names (`tests/data/tlrn_study_*.csv`).[^note]
* **The entry's default is the reference's behaviour.** Its early-stopping patience of 600
  checks never fires in 3,000 epochs. Notebook 04 reports both the default and a patience
  of 120 (600 epochs).[^note]
* **No per-cell draws for tlrn.** It reports company-level CRPS and PIT only and has no
  held-out mixins. `predict(segment)` has one target, `"total"`, with calibrated draws, so
  `evaluate()["point"]["metrics"]` is `None` for tlrn and its point board goes through
  `reserve_rows(point="native")`.[^note]
* **MCL is its own gallery entry** (`mcl`), not a part of tlrn.[^note]

# How the work was ordered

The approved design[^spec] split the work into four plans: A (point scores,
`kernels/point_scores.py`) and B (the training scheme and `kernels/residual_calibration.py`)
built in parallel, then C (the tlrn entry, `kernels/nn_features.py`) and M (the `mcl`
entry), then the 0.7.0 release, then a new notebook 04 on the reference study's cohort and
protocol, leaving notebooks 03b and 03c untouched.[^note]

# Later decisions (2026-09-23)

For 0.7.1, Ethan agreed to 40 members per variant, dropping `tlrn_13_early`, and parallel
training. The parallel training is `fit(processes=n)`, a spawn pool that sends its payload
through a temporary file; see [spawned workers re-run the
script](/gotchas/spawn-workers-rerun-the-script.md). The reasons for averaging all members
are in [which tlrn members are kept](/findings/tlrn-member-choice.md).[^note]

How tlrn variants are configured now is in [tlrn design choices as named
components](/decisions/tlrn-choices-as-named-components.md).

[^note]: Project status log, entries dated 2026-09-20 to 2026-09-23 (lines 100-117 and 203-256)
[^spec]: The approved design (with plans A, B, M and C under docs/superpowers/plans/)
