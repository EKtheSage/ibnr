---
type: Decision
title: The Reserving app moves from chainladder to ibnr
description: "Ethan's decisions (2026-09-23/24) for moving his Reserving app's Azure Function off chainladder-python onto ibnr: zero cells as missing, the log-linear sigma rule, the ibnr.methods front door, Arrow at the boundary and pandas out of ibnr's code."
tags: [reserving-app, chainladder, methods, arrow, polars, pandas]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note reserving-app-ibnr-migration.md (private, outside the repository)
    title: Reserving app ibnr migration
    last_modified: 2026-09-25T02:47:02.301Z
  - id: issue-151
    resource: https://github.com/EKtheSage/ibnr/issues/151
    title: The parked metric-layer redesign
---

# Why

Started 2026-09-23/24 at Ethan's request: "i strongly believe we can reach feature parity
and more performant ... with ibnr". The app is his Reserving app, whose backend is an
Azure Function that fits one triangle per HTTP request, until then through
chainladder-python.[^note] The state of the work is in [Reserving app migration
status](/findings/reserving-app-migration-status.md).

A gap analysis on 2026-09-23 (19 agents, each area checked by a second reviewer)
found:[^note]

* ibnr already matches chainladder on chain ladder, Mack (log-linear rule), BF, Cape Cod
  decay and the basic factor options.
* Still missing in ibnr: tails, a full run-off ODP bootstrap, Benktander / Cape Cod
  trend, Mack under non-default development options, GLM, and the remaining factor
  options.
* The request path must skip ibis: through `Triangle` it is no faster than chainladder.

# Decisions (Ethan, 2026-09-24)

* **D1: a zero cell counts as missing, as chainladder does**, for point estimates and
  for Mack. A link is used only when neither of its cells is zero. ibnr gets this as an
  **option** on both kernels; ibnr's own defaults stay. (Before it, the conventional
  kernel dropped the link out of a zero but kept the link into it, as a ratio of 0, and
  Mack kept both.) A zero on the latest diagonal gives 0, not chainladder's null.
* **D2: the log-linear sigma rule** on the app's `/reserve` and `/cdr` endpoints, always
  passed explicitly. The `/cdr` numbers move.
* **The front door is `ibnr.methods`**, imported as `from ibnr import methods`. It has
  `chain_ladder`, `mack`, `bornhuetter_ferguson` and `cape_cod`, uses ibnr's own option
  names (`history_periods`), and `methods.mack` defaults to the log-linear sigma rule.
* **Arrow at the boundary; no pandas.** Inputs are polars or pyarrow tables, read through
  the Arrow interface without importing polars. Outputs are `pyarrow.Table`s. Every
  result has `.to_polars()`, docs examples are polars-first, and
  `pip install "ibnr[polars]"` is recommended for analysis. Ethan wants people steered to
  polars and offered no pandas helpers.
* **Polars in core was rejected.** It would add about 185 MB and about 0.9 s of cold
  start to the Azure Function.
* **pandas is to be removed from ibnr's own code, module by module, after `methods`
  ships.** pandas stays installed because `ibis[duckdb]` requires it. The first step was a
  numpy-only grid builder inside the methods pull request.[^note]

# Origin labels (Ethan, 2026-09-24)

* Accident years may be written as either the year's start or its end, and Ethan thinks
  of 12/31 as the reserving date. `origin_period` stays the period **start** everywhere in
  ibnr.
* `ibnr.methods` accepts integer years, "2020" / "2020Q3" / "2020-03" labels, and dates
  on the first or last day of a month. A last-day date is read as the period end, using
  `dev_grain_months` as the period length, so a 2021-06-30 year ends in June.
* Results echo the caller's own label in an `origin` column. Ethan chose "show your own
  label".[^note]

# What to do

* Do not add public APIs that return pandas. New tables are Arrow, with `.to_polars()`
  where people look at them.
* The metric-layer redesign (issue #151: metric objects in the style of R's yardstick,
  `level` separate from `by`, a long Arrow result with `.to_polars()`) is **parked** until
  the pandas removal reaches `kernels/point_scores.py`. Ethan did not want it built in
  parallel and then rewritten.[^issue-151]
* Notebook 04 stays pinned to ibnr 0.7.1, so the removal does not touch it until someone
  re-pins it.
* The analysis prototypes (tail, full run-off bootstrap, GLM by IRLS, a copy of
  DevelopmentML) existed only in that session's scratchpad. Rebuild the reference targets
  from chainladder 0.9.2 inside ibnr tests; do not assume they exist.[^note]

Related: [ibnr versions move slowly](/decisions/slow-versioning.md).

[^note]: Reserving app ibnr migration
[^issue-151]: The parked metric-layer redesign
