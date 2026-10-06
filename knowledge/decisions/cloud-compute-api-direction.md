---
type: Decision
title: ibnr's end state is a hosted compute API
description: "Ethan's stated direction (2026-07-21) is for ibnr to be hosted as a cloud compute API called from apps, Excel and notebooks, with the parallel retro harness as the seam it wraps."
tags: [architecture, api, harness, roadmap]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note cloud-api-vision.md (private, outside the repository)
    title: Cloud API vision
    last_modified: 2026-07-22T03:21:25.720Z
  - id: harness
    resource: src/ibnr/kernels/harness.py
    title: The parallel retro harness (run_retro, RetroTask)
  - id: dockerfile
    resource: Dockerfile
    title: The compute image
---

# Decision

Ethan's stated direction (2026-07-21): `ibnr` will eventually be hosted in the cloud as
a compute API that other apps, Excel and Python notebooks call.[^note]

# What already serves it

* The parallel retro harness (`kernels/harness.py`) was deliberately built as a
  library-level component, not as script logic, so that the future API wraps the same
  `run_retro` / `RetroTask` entry points.[^harness]
* The `Dockerfile` (the package, cmdstan 2.39.0, and all gallery Stan models
  pre-compiled; parallel by default through `IBNR_MAX_WORKERS`) is the base image for
  that service.[^dockerfile]

# What to do

No HTTP layer existed when this was recorded. When it is built, wrap the harness; do not
fork its logic.[^note]

[^note]: Cloud API vision
[^harness]: The parallel retro harness (run_retro, RetroTask)
[^dockerfile]: The compute image
