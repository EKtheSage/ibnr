---
type: Decision
title: Rules for a fair model comparison
description: "Ethan's standing rules for any model comparison or backtest in ibnr: identical training data for every model, point metrics beside distributional ones, build both plausible designs, and keep each change separately switchable."
tags: [evaluation, backtest, comparison, methodology]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note comparison-design-principles.md (private, outside the repository)
    title: Comparison design principles
    last_modified: 2026-08-06T07:39:45.643Z
---

# The rules

Stated by Ethan on 2026-07-08 during the multi-line transformer work.[^note]

1. **Every model sees identical training data.** Giving a pooled model extra cohorts
   that the per-company baselines cannot use (for example companies writing only one
   line) "seems like cheating", even if it might help. Enforce this at the level of the
   (company, line) pair, not the company.
2. **Always evaluate point predictions as well as distributions.** CRPS and the KS test
   alone are not enough; he wants a comparison that uses the predicted values only and
   assumes no distribution. The agreed design: reserve basis (subtract the `as_of`
   anchor), premium-normalized mean absolute error as the robust headline, median
   absolute percentage error, a skill ratio against the volume-weighted chain ladder,
   and a paired Wilcoxon test.
3. **When two designs are plausible, build both and let the data decide** (for example
   the "ar" against "joint" dependence heads), rather than arguing in advance.
4. **Simple fixes first, and keep each change separately switchable.** Simple fixes a
   colleague suggests are tried before clever ones. Ethan asks which change actually
   drove an improvement, so each change gets its own switch: one CSV per experiment
   arm, with bit-identical baseline rows across runs through fixed seeds, so a change's
   effect can be measured on its own. (Describe this as an "on/off comparison" or
   "measuring the effect on its own"; Ethan banned a family of jargon words for it.)

# Why

The package's credibility, and the CAS manuscript's, rests on fair comparisons that a
reviewer cannot pick apart. Changes mixed together, and data advantages for one model,
undermine both.[^note]

# What to do

Before running any new backtest: check that every entrant has the same training data,
include the chain-ladder benchmark and the reserve-basis point table, and structure the
experiment so each design change lands in its own output file.[^note]

[^note]: Comparison design principles
