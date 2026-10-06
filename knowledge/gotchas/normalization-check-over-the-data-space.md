---
type: Gotcha
title: A normalization check must integrate over the data's space
description: "A density normalization test that sums over a grid chosen to make the arithmetic work (ODP's lattice) can pass for any parameters; integrate over the space the observations actually live in."
tags: [evaluation, elpd, densities, testing]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note milestone6-eval-decisions.md (private, outside the repository)
    title: Milestone 6 eval decisions
    last_modified: 2026-07-26T20:22:06.320Z
  - id: pr-28
    resource: https://github.com/EKtheSage/ibnr/pull/28
    title: The external review that caught it
---

# What happened

In milestone 6 the ODP density was briefly treated as ELPD-eligible after a `-log(phi)`
conversion. The normalization test that let it through summed `exp(.) * phi` over the
lattice `{0, phi, 2phi, ...}` and got 1.000000.[^note]

That is true and irrelevant. It recovers the Poisson probability sum for **any**
parameters, and observed losses are arbitrary real numbers that never land on that
lattice. In fact `exp(odp_lpdf)/phi` does not integrate to 1, by an amount that varies
with `mu/phi`. External review caught it (PR #28) and the decision was reversed; see
[Milestone 6 evaluation design](/decisions/milestone-6-evaluation-design.md).[^pr-28]

# The lesson

A check that cannot fail is worse than none, because it gets quoted as evidence.
**Integrate over the observation space the data actually lives in, never over a grid
chosen to make the arithmetic work.**[^note]

The sibling failure, a check that is consistent with itself rather than right, is in
[the inert parameter bug class](/gotchas/inert-parameter-bug-class.md).

[^note]: Milestone 6 eval decisions
[^pr-28]: The external review that caught it
