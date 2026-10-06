---
type: Measurement
title: Speed benchmark against chainladder-python, scaling
description: What decides whether ibnr or chainladder-python is faster (dense regular triangles against ragged real ones, not row count), measured in July 2026, with the one optimization lead it left.
tags: [performance, benchmark, chainladder, duckdb, polars, milestone-9]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries of 2026-07-24 and 2026-07-25 (lines 987-1030) and 2026-07-28 (lines 1263-1264)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: pr-20
    resource: https://github.com/EKtheSage/ibnr/pull/20
    title: scripts/benchmark_scaling.py
  - id: pr-19
    resource: https://github.com/EKtheSage/ibnr/pull/19
    title: fit_mack_many, the batch Mack fit
---

`CLAUDE.md` (milestone 9) has the headline numbers of `scripts/benchmark_speed.py`. This
concept keeps the scaling result and the facts that are not there.

# Result: shape decides, not size

`scripts/benchmark_scaling.py` builds synthetic staircase triangles of 1e5 to 1e7 rows.
On perfectly dense, regular triangles chainladder-python's 4D arrays keep winning cell
transforms at **every** size: `latest_diagonal` is 26 times faster there at 1e7 rows.
ibnr's wins that grow with scale are construction (4.5 to 6 times) and group-by
aggregation (6.3 times). So the boundary is dense and regular against ragged and real (the
Schedule P mart), not the row count.[^pr-20]

# Optimization lead (not tried)

duckdb's `latest_diagonal`, written as a join, degrades worst: 13.4 s at 1e7 rows against
polars' 3.0 s. Try a window-function formulation on the duckdb path only; polars has no
window support in ibis and keeps the join.[^note]

# Facts worth not re-deriving

* chainladder 0.8.x `MackChainladder` raises `ValueError` on any triangle with more than
  one index, even its own clrd sample. Only its point-ultimate `Chainladder` is
  vectorized: 32 ms against ibnr's 0.6 to 1.0 s per-cohort loop over 92 workers'
  compensation cohorts. `fit_mack_many` (PR #19, 2026-07-25) closed that gap: 92 clrd
  cohorts in 43 to 47 ms, 16 to 20 times faster than the loop.[^pr-19]
* chainladder stores explicit zeros as missing at construction, so the mart's 316,000
  zero cells vanish and cell counts tie out across the libraries only on non-zero cells.[^note]
* The mart has no restated history: every cell has one view, with evaluation at origin
  plus lag minus 1.[^note]
* Making the scipy imports lazy took about 1.26 s off importing `ibnr.gallery` (15 of 15
  interleaved pairs, 2026-07-28).[^note]

Ethan said on 2026-07-24 the benchmark work was not finished; the pending operations are
listed in [open questions](/findings/open-questions-2026-09.md).

[^note]: Project status log, entries of 2026-07-24 and 2026-07-25 (lines 987-1030) and 2026-07-28 (lines 1263-1264)
[^pr-20]: scripts/benchmark_scaling.py
[^pr-19]: fit_mack_many, the batch Mack fit
