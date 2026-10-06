---
type: Finding
title: Which tlrn members are kept decides the result
description: Keeping the best two of ten tlrn members by validation score is close to a random pick; averaging all members is steadier and better, and the gap to the reference study came from which pair was kept.
tags: [tlrn, ensembles, model-selection, companion-study]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries dated 2026-09-23 (lines 100-141) and 2026-09-24 (lines 67-81)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: pr-150
    resource: https://github.com/EKtheSage/ibnr/pull/150
    title: Notebook 04 on 0.7.1, with the section "How much of the result is the choice of members"
---

# Result

The reference study trains ten members (copies of the network from different random
starts) and keeps the best two by validation score. Measured on notebook 04's 40-member
fits in the 2026-09-24 run, that choice is **effectively random**: the rank correlation
between a member's validation score and its score on the real outcome is **0.14** for the
13-feature fit and **-0.06** for the paid-only fit.[^note]

Drawing 5,000 random sets of 10 members from the 40 (13 features), as the notebook's
section "How much of the result is the choice of members" does:[^note][^pr-150]

| rule | median Pool_APE, raw | median Pool_APE, blended with MCL | runs that beat MCL |
|---|---|---|---|
| keep the best two | 6.40% | 5.61% | 60% |
| average all ten | 5.81% | 5.20% | 97% |

Averaging all 40 gave the notebook's best row, 5.16% blended (see [notebook 04
results](/findings/notebook-04-results.md)).

# What this explained

* 0.7.0's finding that the paid-only fit beats the 13-feature fit came from which two
  members were kept.[^note]
* The 1.14-point gap between ibnr's `tlrn_13` and the reference (2026-09-23): swapping one
  kept checkpoint (`tlrn_13` against `tlrn_13_early`) moved raw Pool_APE by 0.14 points.
  Retraining the ten members the same day showed that ibnr and the reference produce the
  same spread of members, and the gap is purely which pair was kept. No code change and no
  release followed from it.[^note] The detailed evidence, including all 45 pairs of the
  reference's own ten members, is in [ibnr's tlrn matches the R
  reference](/findings/tlrn-port-matches-r-reference.md).
* Ethan asked whether the reference's seeds would give the reference's numbers. Answer
  given: no, because thousands of random calls happen in order, floating-point sums depend
  on order, and the best two of ten members are selected. Measuring the spread
  over seeds was proposed instead.[^note]

# What was decided

For 0.7.1 (2026-09-23, Ethan's "ok"): 40 members per variant, the variant
`tlrn_13_early` dropped, members trained in parallel, and `keep=40` (average them all) in
the notebook, with `_keep2` rows (best two of members 0 to 9) kept for comparison.[^note]

[^note]: Project status log, entries dated 2026-09-23 (lines 100-141) and 2026-09-24 (lines 67-81)
[^pr-150]: Notebook 04 on 0.7.1, with the section "How much of the result is the choice of members"
