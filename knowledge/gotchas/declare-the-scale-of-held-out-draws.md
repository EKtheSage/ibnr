---
type: Gotcha
title: Held-out draws must declare their scale
description: "Draws for held-out cells need a declared scale (cumulative or incremental), exactly as a density needs a declared measure; undeclared draws are wrong by the whole training-diagonal anchor and still look plausible."
tags: [evaluation, crps, holdout, leaderboard]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note declare-the-scale-not-just-the-measure.md (private, outside the repository)
    title: Declare the scale, not just the measure
    last_modified: 2026-07-26T20:22:20.603Z
  - id: pr-33
    resource: https://github.com/EKtheSage/ibnr/pull/33
    title: PredictsHeldout (where this was established)
---

# The rule

Established while building `PredictsHeldout` (PR #33, 2026-07-26).[^pr-33]

A held-out **density** declares `heldout_measure`, and the base class carries it to one
common scale. A held-out **draw** needs exactly the same treatment: `heldout_draw_scale`
is `cumulative` or `incremental`, and `PredictsHeldout.predict_at` carries the draws to
the triangle's own basis.[^note]

# Why it is not bookkeeping

Three of the five Bayesian entries model increments, while the Schedule P triangles are
cumulative. Draws scored against `HoldoutCells.values` without a declared scale are wrong
by the whole training-diagonal anchor (the cell's previous cumulative value). Measured on
a cell with anchor 1000: **CRPS 996 where the truth is 3.4**. Both numbers are finite,
both smooth, both entirely plausible on a leaderboard. It is the same bug class as a
missing Jacobian.[^note]

# Why the conversion is safe

`C = X + C_prev`, with `C_prev` on the *training* diagonal, so it is data the model
already had. That is the same fact that makes the increment-to-cumulative Jacobian equal
to 1 for a density.[^note]

# Why comparing across scales is legitimate at all

CRPS is translation-equivariant: `CRPS(F+c, y+c) == CRPS(F, y)`, verified to 5.5e-12. So
once each entry is on its triangle's basis the numbers are directly comparable. This
holds only for the *shift*: CRPS still depends on units, so dollars against thousands
remains a real error (caught one layer up, by the check that observed values agree).[^note]

# What to do

* When adding a second way to produce the same quantity, make each producer **declare**
  its convention and convert in **one** shared place.
* Never accept a `target_scale=` argument from the caller: a setting whose wrong value is
  exactly the bug.
* `predict_at` takes a `HoldoutCells`, not a bare `CellIndex`, precisely because only the
  former records what the target basis is.[^note]

Watch for the vocabulary collision: `HoldoutCells.measure` is the **triangle's** basis,
while `densities.MEASURES` is the **density's** scale. Two different things are spelled
"measure".[^note]

Related: [Milestone 6 evaluation design](/decisions/milestone-6-evaluation-design.md) and
[the inert parameter bug class](/gotchas/inert-parameter-bug-class.md).

[^note]: Declare the scale, not just the measure
[^pr-33]: PredictsHeldout (where this was established)
