---
type: Gotcha
title: Checking that a notebook run really happened
description: Ways a long notebook execution can look successful when it was not, met between July and September 2026, and the checks that caught them.
tags: [notebook, process, verification, windows]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries of 2026-07-30 (lines 1479-1483), 2026-08-05 and 2026-08-06 (lines 1579-1623) and 2026-09-22 (lines 163-174)"
    last_modified: 2026-09-25T08:31:20.962Z
---

How to run notebook 04 in pieces is in [re-run notebook 04](/playbooks/rerun-notebook-04.md).
These are the traps that apply to any long notebook run.

# The traps

* **A wrapper's exit code is not nbconvert's.** A background bash wrapper reported exit
  code 0 for a run that had failed. Read the run's own log line for its exit status
  (2026-07-30). That cost a wrong success report.[^note]
* **`nbconvert --inplace` writes the notebook only at the end.** A failed run leaves the
  file byte-identical, which looked like a perfectly reproducible re-run in a spot check
  of one cell. Diff the outputs against the previous run before calling a re-execution
  real.[^note]
* **A Windows restart kills a run and saves nothing.** On 2026-09-22 a restart chosen
  from the Start menu killed nbconvert 113 minutes in, with exit code 1073807364
  (0x40010004, the code Windows gives processes it kills at logoff or restart). The output
  folder was empty and the run started again from zero. The laptop also sleeps after 5
  idle minutes on AC power unless the wrapper keeps it awake (`SetThreadExecutionState`).[^note]
* **A patch script that stops early silently loses edits.** A script applying review
  fixes to notebook 3b failed its last assert, so `nbformat.write` never ran, two of three
  "applied" fixes were lost, and the merged notebook kept an unused import that broke
  `main`'s lint job for about two hours (2026-08-05). Verify a multi-edit script by running
  the real check (ruff or the checker) on the final file, not by the script's own success
  message, and watch the checks after merging.[^note]
* **One training seed is not enough for a comparison.** A single-seed draft of notebook 3b
  claimed the case channel helped deeptriangle; refits at a second seed showed it inside
  the seed's own spread. A cell that refits at a second seed is cheap (about 2.5 minutes
  there) and turns a hedge into a measurement (2026-08-06).[^note]
* **Diagnose only the origins a rollout simulates.** `case_paths()` covers fully
  developed origins too, whose "terminal" level is the observed value repeated
  (2026-08-06).[^note]
* **For float32 draws, test for a constant with `np.ptp() == 0`, not `std() == 0`**
  (2026-08-06).[^note]

[^note]: Project status log, entries of 2026-07-30 (lines 1479-1483), 2026-08-05 and 2026-08-06 (lines 1579-1623) and 2026-09-22 (lines 163-174)
