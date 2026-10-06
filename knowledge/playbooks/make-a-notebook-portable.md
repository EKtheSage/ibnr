---
type: Playbook
title: Making an analysis notebook portable
description: "How to make an analysis notebook run from any directory on a plain pip install: vendor repo scripts, pin the data publish explicitly (a bare pinned_source() does call gh), assert the package version, and prove it from a neutral directory and a fresh PyPI venv."
tags: [notebook, reproducibility, schedule-p, data]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note notebook-portability-pattern.md (private, outside the repository)
    title: Notebook portability pattern
    last_modified: 2026-08-05T04:54:59.510Z
  - id: pr-84
    resource: https://github.com/EKtheSage/ibnr/pull/84
    title: Notebook 3b (where the recipe was established)
---

Established while building notebook 3b (2026-08-03, PR #84).[^pr-84] A notebook is
self-contained when it runs from any directory, in any repository, on a plain
`pip install "ibnr[nn]==<version>" matplotlib` plus a warm data cache.[^note]

# The recipe

1. **Vendor repository scripts; never import them.** `scripts/` is not packaged, so
   `sys.path` edits plus `from meyers_validation import ...` are the one hard tie to a
   checkout. Copy the code verbatim with a provenance comment naming the source file and
   commit. The future home for the Schedule P company selection is the companion data
   package (Ethan's direction), not `ibnr.data`.[^note]
2. **Pin the data publish explicitly:**
   `pinned_source("github://EKtheSage/cas-schedule-p-data-model@20260613_041006")`.
   Pass the pinned source positionally to all of `load_schedule_p`, `active_mart_path`
   and `active_publish_id`.[^note]
3. **Assert the package version** (`ibnr.__version__`) whenever the notebook touches
   private internals (`gallery.nn` modules, file:line anchors).[^note]
4. **Prove it twice, both runs from a neutral working directory outside any
   repository:** once with the dev venv, and once in a venv built **only** from PyPI
   (`uv venv && uv pip install "ibnr[nn]==X" matplotlib nbclient ipykernel`). The PyPI run
   catches drift in a consumer's environment: it resolved torch 2.13 against the lock's
   2.12, and the seeded NN fits still reproduced the leaderboard identically to printed
   precision.[^note]

# Traps

* **A bare `pinned_source()` does call gh.** Notebook 03's bare `pinned_source()` carries
  a comment saying it "never resolves @latest, never calls gh". That describes the
  **returned** string being safe for worker processes, not the call. With no argument it
  falls through to `@latest` and does call gh. With an explicit tag and a warm cache
  there is no gh call at all, and `active_publish_id(SOURCE)` is safe to assert (a
  mismatch between the manifest and the tag raises).[^note] The comment is the [inert
  parameter bug class](/gotchas/inert-parameter-bug-class.md) in prose.
* **`IBNR_SCHEDULE_P_WAREHOUSE` is inert with explicit sources.** The environment
  variable is consulted only when the source argument is None. Do not claim otherwise in
  prose; that claim was a review finding.[^note]

# Facts about the notebook 03 evaluation set that recur

* Dropping Stan cuts the run from about 19 minutes to 2.5 minutes.
* The held-out board is CRPS-only: every NN held-out cell sits past the pinned
  standardization frontier (so the held-out density refuses), and Mack has draws but no
  density. An empty ELPD column with a per-model `elpd_status` reason is the correct
  output, not a bug.[^note]

# Related

* How the notebook's prose should read: [Notebook prose register](/decisions/notebook-prose-register.md).
* Running notebook 04 in pieces: [Re-run notebook 04](/playbooks/rerun-notebook-04.md).

[^note]: Notebook portability pattern
[^pr-84]: Notebook 3b (where the recipe was established)
