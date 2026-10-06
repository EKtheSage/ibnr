---
type: Finding
title: Where the milestones stood in September 2026
description: The state of each roadmap milestone as the agent's status note last recorded it (2026-09-25), with completion dates and the measured results not kept in CLAUDE.md.
tags: [roadmap, milestones, status]
status: draft
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2026-12-31T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, milestone entries of 2026-06-11 to 2026-09-22 (lines 484-506, 672-811, 907-986, 1075-1209, 1314-1335, 1470-1477, 262)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: claude-md
    resource: https://github.com/EKtheSage/ibnr/blob/main/CLAUDE.md
    title: The milestone list (the authority on scope)
---

`CLAUDE.md` holds the milestone list and most results.[^claude-md] This is the state as of
the note's last entry, 2026-09-25, and the results the note has that `CLAUDE.md` does not.

# State

| milestone | state | date |
|---|---|---|
| 1. Triangle layer, tie-outs, interop, mart adapter | done | 2026-06-11 |
| 2. `meyers_ccl` and the Meyers retrospective | done | 2026-06-13 |
| 3. Neural family opens, statistical dependence baselines | done (moved ahead of the ports at Ethan's request) | 2026-07-06 |
| 4. All Bayesian entries in Stan first, with 200-company retrospectives | done | 2026-07-21 |
| 5. NumPyro and PyMC ports pass parity | done | 2026-07-25 |
| 6. Held-out ELPD, PIT, stacking, leaderboard | started 2026-07-25; still open, the live leaderboard rerun not done | |
| 9. Speed benchmark | done; Ethan said more was to come | 2026-07-24 |
| 11. The chainladder 0.10 estimator surface | triaged, not designed | 2026-07-30 |

Outside the numbered list: the one-year CDR (2026-07-24, made multi-method 2026-07-29),
the conventional point candidates with replay and selection (0.6.0, 2026-09-16), and tlrn
with mcl (0.7.0, 2026-09-22). A further milestone, the [machine-learning
family](/decisions/ml-family-plan.md), was agreed on 2026-07-25.[^note]

# Results not in CLAUDE.md

* **Milestone 4.** The parallel harness reran both compartmental retrospectives (Gaussian
  D = 40.0*, was 39.9*; lognormal D = 16.2*, was 16.1*, with commercial auto passing at
  12.3, was 12.5; a star marks a KS rejection at 5%). Median shift of a company's percentile 0.06 and 0.14 points
  (maximum 3.8); the rerun files replace the published sequential ones. In the
  sequential run, the Gaussian variant had a median coefficient of variation of 2.5% with
  110 of 200 outcomes outside the 5 to 95 band, and other liability's median
  estimate-over-outcome of 0.84 had 35 of 50 above the 95th percentile, because priors
  taken from one workers' compensation company bind there (ULR prior median 0.56, and
  `kp ~ LN(0, 0.1)` is too fast for other liability). The lognormal variant has 63 of 200
  outside the band and no bias (estimate over outcome 0.99 to 1.02 on all lines).[^note]
* **Milestone 4, CSR.** Meyers' official appendix `CSR.R` is the ground truth:
  `gamma ~ normal(0, 0.05)` is a standard deviation (his prose quotes variances
  elsewhere). Private passenger auto fails alone at D = 25.3* (Meyers' was 18.5,
  marginal).[^note]
* **Milestone 4, ODP family.** 104 of 200 companies were refused for negative paid
  increments (private passenger auto worst); Meyers' bootstrap tolerated them, so his n is
  200. A Pearson-phi bug once cost 34 companies: lag columns that are zero all the way down
  fit m = 0 at x = 0 cells and must be left out of the Pearson sum, not rejected.[^note]
* **Milestone 4, Clark.** Ages shifted to the middle of the period (x = 12d - 6) reproduce
  chainladder's `ClarkLDF` exactly (omega 1.4355, theta 48.51 on genins); plain ages give
  omega 2.04.[^note]
* **Milestone 6.** After PR #34 (2026-07-27), four entries are permanently without a
  density for ELPD (`england_verrall_odp`, `clark_growth_curve`, `statistical/clark`,
  `deterministic/mack`). The `meyers_ccl` scorer was checked to agree with the fit's own
  log-likelihood within 1e-5, with deliberately broken versions off by more than 100
  times.[^note]
* **One-year CDR.** On MW2014 the bootstrap route's total CDR standard error is 2,507
  against Mack's 1,843 (+36%): a different model, not a discrepancy (2026-07-29). The
  Mack route matches R's published `MW2014` output to 7 decimal places (total
  1842.8507073), and chainladder-python has no CDR at all (checked in 0.9.2 and upstream
  main).[^note]
* **Milestone 3 to 5 neural parameter counts.** The transformer builds 70,121 parameters
  (8 x 8) or 70,505 (10 x 10), 66,944 of them in the two encoder layers, not the "about
  120,000" its card once claimed; mdn's "about 30,000" was 26,353 (2026-07-28 and
  2026-07-29).[^note]

[^note]: Project status log, milestone entries of 2026-06-11 to 2026-09-22 (lines 484-506, 672-811, 907-986, 1075-1209, 1314-1335, 1470-1477, 262)
[^claude-md]: The milestone list (the authority on scope)
