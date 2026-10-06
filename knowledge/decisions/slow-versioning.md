---
type: Decision
title: ibnr versions move slowly
description: "Additions and fixes ship as patch releases (0.7.x), even when they add a public module; a minor bump (0.8) is kept for when the roadmap's goals are done (Ethan, 2026-09-24)."
tags: [release, versioning, process]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note versioning-slow.md (private, outside the repository)
    title: Slow versioning
    last_modified: 2026-09-25T02:21:32.963Z
  - id: claude-md
    resource: https://github.com/EKtheSage/ibnr/blob/main/CLAUDE.md
    title: "CLAUDE.md, Tooling & conventions: Versioning is slow"
---

# Decision

Ethan, 2026-09-24, when the agent proposed releasing the chainladder-migration work
(#149 `fit_conventional_grid`, #152 `ibnr.methods`, #153 `zero_cells`) as 0.8.0: "we
should slow down the versioning - maybe 0.7.2, i feel 0.8 is quite major, something that
happens when everything you set out to do gets done."[^note]

# Why

To him a minor bump is a milestone marker, the roadmap's goals reached, not a signal that
the public surface grew. That holds even though the CHANGELOG header's loose-semver note
allows a minor bump to change kernel signatures.[^note]

# How to apply

* Propose the next **patch** version for feature and fix releases, even when they add a
  public module or new refusals on existing functions.
* Keep a minor bump for when Ethan says a body of work is complete, and ask before
  proposing one.
* Name any change that could break existing code in the release notes, rather than using
  it to justify a bigger bump.[^claude-md]

Related: [The Reserving app moves from chainladder to ibnr](/decisions/reserving-app-moves-to-ibnr.md).

[^note]: Slow versioning
[^claude-md]: "CLAUDE.md, Tooling & conventions: Versioning is slow"
