---
type: Finding
title: Reserving app migration status
description: "Where moving the Reserving app's Azure Function from chainladder-python to ibnr stood on 2026-09-25: the app's location and plan document, ibnr steps 1 to 1b merged, zero_cells built, and ibnr 0.7.2 released."
tags: [reserving-app, chainladder, release, status]
status: deprecated
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2026-10-12T00:00:00Z
sources:
  - id: note
    resource: agent memory note reserving-app-ibnr-migration.md (private, outside the repository)
    title: Reserving app ibnr migration
    last_modified: 2026-09-25T02:47:02.301Z
  - id: pr-149
    resource: https://github.com/EKtheSage/ibnr/pull/149
    title: fit_conventional_grid (5eca4e1)
  - id: pr-152
    resource: https://github.com/EKtheSage/ibnr/pull/152
    title: ibnr.methods (4a63608)
  - id: pr-153
    resource: https://github.com/EKtheSage/ibnr/pull/153
    title: zero_cells option (65a95f1)
  - id: pr-154
    resource: https://github.com/EKtheSage/ibnr/pull/154
    title: Release 0.7.2 (eaf34cd)
---

**This is a snapshot from 2026-09-25 and has surely moved since.** The decisions behind
it are in [The Reserving app moves from chainladder to
ibnr](/decisions/reserving-app-moves-to-ibnr.md).

# The app

`C:\Users\EthanKang\Projects\Reserving\chainladder-mvp`. It has no git remote; local
branches are merged into main with merge commits, and its commits do carry the Claude
Co-Authored-By trailer, unlike ibnr's. The backend is `azure-function/function_app.py`, a
uv project since commit ec70715 (2026-09-23), whose pyproject pinned
`chainladder==0.9.2` and `ibnr==0.6.0`. Another agent session was committing to its main
the same night, so work there went in a worktree under `.claude/worktrees/`.[^note]

The plan, `docs/PLAN-IBNR-MIGRATION.md`, is on the app's branch
`docs/ibnr-migration-plan` (worktree `.claude/worktrees/ibnr-migration-plan`) and was
**not** merged to the app's main. It holds seven steps and fourteen decisions (D1 to
D14), each with a recommendation.[^note]

# ibnr steps

* **Step 1**, PR #149 (`feat/conventional-grid-fit`): adds `kernels.fit_conventional_grid`
  (the array path, about 2 ms against 28 ms on RAA), exports `cohort_grid_frame` and
  `fit_mack_grid`, and adds the shared `contract.check_grid`, which `fit_mack` /
  `fit_mack_many` now also run. Merged 2026-09-24 as 5eca4e1.[^pr-149]
* **Step 1b**, PR #152, `ibnr.methods`: merged 2026-09-24 as 4a63608.[^pr-152] Branches and
  worktrees of both were cleaned up.
* **D1 built**, PR #153 (`feat/zero-cells`): the kernels default to
  `zero_cells="observed"`, byte-identical to before; `"missing"` is chainladder's rule, and
  `ibnr.methods` defaults to it. Under `"missing"` the one-year CDR refuses any fit the
  rule changed. The codec version was **not** bumped; that was an open point for
  Ethan.[^pr-153]

# Released as ibnr 0.7.2 (2026-09-24)

#153 merged (65a95f1), then release PR #154 (eaf34cd), tag v0.7.2, Release run
36086976027 green.[^pr-154]

* PyPI serves the wheel (sha256 114799d7...) and the sdist.
* A clean PyPI install ran `methods.chain_ladder` on integer years and `methods.mack`
  without importing polars.
* `uv pip` needed `--refresh` to see the new version.[^note]

# Next step at the time

App step 2 in chainladder-mvp, pinned to `ibnr==0.7.2`.[^note]

[^note]: Reserving app ibnr migration
[^pr-149]: fit_conventional_grid (5eca4e1)
[^pr-152]: ibnr.methods (4a63608)
[^pr-153]: zero_cells option (65a95f1)
[^pr-154]: Release 0.7.2 (eaf34cd)
