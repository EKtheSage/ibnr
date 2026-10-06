---
type: Measurement
title: Install size and deployment measurements
description: How big an ibnr install is and why polars became an extra, what fits in an Azure Function, and which public installs worked on which platforms (July to September 2026).
tags: [packaging, deployment, azure, install, size]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries of 2026-07-23 (lines 812-820 and 886-898), 2026-07-28 (lines 1308-1312) and 2026-09-07 (lines 286-291)"
    last_modified: 2026-09-25T08:31:20.962Z
---

# Install size (2026-07-23)

Measured on a clean environment: ibnr 0.3.0 took 511 MB of site-packages, of which the
polars runtime was 176 MB. Moving polars to the `ibnr[polars]` extra in 0.4.0 took the
core install to **331 MB, a saving of 180 MB (35%)**. Bare ibis is only 17 MB; the weight
is in the backends. duckdb stays in the core because it is the default backend; dropping
it would break `pip install ibnr` and save only about another 84 MB, since pyarrow comes
along anyway. The change was driven by an Azure Function in another repository that
imports ibnr.[^note]

Also established that day: the package was already installable with uv before the first
publish. `uv build` gives a valid wheel, the `.stan` and `card.md` files are bundled
(hatchling picks them up because they are tracked by git under `src/ibnr/`), and a clean
core-only install imports `ibnr` and `ibnr.gallery` and registers all 10 entries it had then, because the
extras are imported only inside `.fit()`.[^note]

# Azure (2026-07-28)

Core ibnr fits an Azure Function: a 129 MB zip and about 6.6 s to import, against the
30 s start-up limit. The cmdstan and neural tier needs the container route; Azure
Container Apps Jobs was recommended, at about $10 to $60 a month. The harness's `run_task`
already has the shape of a queue worker; `RetroTask` is the last API tied to Schedule
P.[^note]

# Public installs (2026-09-07, pip 26.2, PyPI 0.5.9)

* Windows, Python 3.11 and 3.12: install and import work, on the numpy-1 stack (numpy
  1.26.4, arviz 0.18, pymc 5.25.1, jax 0.7). On 3.12, matplotlib 3.7.2 builds from source.
* Linux, Python 3.11: resolves from wheels only.
* Linux, Python 3.12: `ResolutionImpossible`, both with wheels only and with source
  distributions allowed.

The note says the README's install section carries these numbers. Which extras each line
was measured with is not recorded in the note.[^note]

[^note]: Project status log, entries of 2026-07-23 (lines 812-820 and 886-898), 2026-07-28 (lines 1308-1312) and 2026-09-07 (lines 286-291)
