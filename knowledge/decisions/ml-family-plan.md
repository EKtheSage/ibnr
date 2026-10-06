---
type: Decision
title: The machine-learning family plan
description: The tabular-learner milestone Ethan agreed on 2026-07-25 (GLM, then random forest, then three gradient-boosting libraries), its hard constraint, and the open question of conformal prediction.
tags: [roadmap, ml, glm, conformal]
status: draft
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entry of 2026-07-25 (lines 1099-1125)"
    last_modified: 2026-09-25T08:31:20.962Z
---

# Decision

On 2026-07-25 Ethan agreed a new milestone to follow milestone 5: a machine-learning, or
tabular-learner, family. Ordered by value for the effort, with his sign-off on the GLM
going first:[^note]

1. **GLM.** A frequentist over-dispersed Poisson GLM, whose maximum-likelihood fit
   reproduces the chain ladder, wrapped in the England-Verrall bootstrap: the frequentist
   twin of the existing `england_verrall_odp`. The cheapest of the three, and the only one
   with a tie-out reference, since chainladder-python ships `TweedieGLM` and
   `DevelopmentML`.
2. **Random forest**, as a quantile regression forest (Meinshausen 2006). Its leaves
   already hold the conditional sample, so the distribution comes nearly free.
3. **Gradient boosting, all three of LightGBM, NGBoost and CatBoost.** Ethan's call: these
   are three genuinely different designs, not three spellings of one. NGBoost is
   distributional by construction, CatBoost has `RMSEWithUncertainty`, and LightGBM needs a
   quantile loss (watch for quantiles crossing).

The family string is likely a new `"ml"`, as `deterministic` was opened for mack; the GLM
would sit in `statistical`.[^note]

# Hard constraint

Design decision 4 bars point estimators from the gallery, so every one of these needs a
real distributional head.[^note]

# Conformal prediction (open)

Ethan also asked, on 2026-07-25, for conformal prediction to be investigated. The concern
raised before any research: conformal coverage guarantees need the calibration and test
cases to be exchangeable, and reserving breaks that by construction. The test set is the
lower triangle (future calendar periods), residuals depend on the lag, calendar effects
hit whole diagonals, and the scored quantity is an origin's ultimate, a sum over cells,
not one cell. A calibration set across companies (about 200 in the mart) is the one unit
that might be defensibly exchangeable.[^note]

Candidate literature named: weighted conformal under covariate shift (Tibshirani et al.
2019), adaptive conformal inference (Gibbs and Candes 2021), EnbPI (Xu and Xie 2021),
conformal prediction beyond exchangeability (Barber et al. 2023) and conformalized
quantile regression (Romano et al. 2019). A research run was made on 2026-07-25, but its
conclusion was never written into the note, so the outcome is unknown here.[^note]

# Status

The note records no work on this family after 2026-07-25 (its last entry is
2026-09-25), and does not say whether the milestone was given a number in `CLAUDE.md`.

[^note]: Project status log, entry of 2026-07-25 (lines 1099-1125)
