---
type: Finding
title: Held-out results of notebooks 03, 03b and 03c
description: What the gallery comparison notebooks 03, 03b and 03c measured on held-out cells (July to August 2026), including the defects their runs exposed and which earlier numbers those defects invalidate.
tags: [notebook, leaderboard, held-out, crps, elpd, calibration]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-04-05T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries of 2026-07-28 (lines 1269-1291), 2026-08-03 to 2026-08-06 (lines 1495-1626) and 2026-08-25 (lines 387-482)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: pr-69
    resource: https://github.com/EKtheSage/ibnr/pull/69
    title: Notebook 03's corrected boards
  - id: pr-98
    resource: https://github.com/EKtheSage/ibnr/pull/98
    title: Notebook 03b on 0.5.5
  - id: pr-104
    resource: https://github.com/EKtheSage/ibnr/pull/104
    title: Per-cohort draw streams
  - id: pr-106
    resource: https://github.com/EKtheSage/ibnr/pull/106
    title: Notebook 03c rerun on 0.5.8
---

"CRPS" is the continuous ranked probability score of the predictive draws (lower is
better); "ELPD" is the expected log predictive density (higher is better); "PIT" is the
outcome's percentile in the model's predictive distribution, which should look uniform.

# Notebook 03: the whole gallery, 25 cohorts (2026-07-28)

The first version (PR #66) was superseded by PR #69 after two problems were found in
review.[^pr-69]

* **Provenance.** The fits trained on 1988 to 1997 while `next_diagonal` claimed 1990 to
  1997. Fixed by cutting the window in the data. Consequence: on a provenance-consistent
  set of cells, **the neural entries have no ELPD** (see [open
  questions](/findings/open-questions-2026-09.md)). Their earlier ELPD rows existed only
  because the scored diagonal stopped two development steps short of the training.
* **Calibration test.** A KS test on 168 dependent cells was demoted to descriptive. The
  calibrated test is the cohort-level PIT of the summed next diagonal (n = 24): **only
  `meyers_csr` survives at 5%**, and every model's mean cohort PIT is 0.13 to 0.40, so
  totals are biased high across the gallery.[^note]

Corrected boards: CRPS, mack 3,336 best, then `meyers_csr` 3,815. ELPD covers four MCMC
entries only; compartmental is top at -8.94 but its minimum Kish effective sample size is
2.0, so it is unranked in practice. Off-label, reported-loss CSR beats CCL. ODP's rule of
non-negative paid increments does not imply positive outstanding: compartmental-lognormal
refused 7 of 25 cohorts.[^note]

Defects that invalidate earlier numbers: any neural held-out number produced before
PRs #56 and #58 is wrong. #56: the cutoff came from usable increments instead of the
latest-lag anchors, so a cohort with lags {1, 2, 4} was conditioned three diagonals out
instead of one. #58: draws were scaled by the supplied premium while the network trained
on the contract's premium, and changing only the supplied premium moved densities by about
90 nats per cell without raising.[^note]

# Notebook 03b: neural entries against mack (2026-08-03 to 2026-08-06)

Notebook 03's exact cohorts with the Stan family dropped: four neural entries against
deterministic mack on held-out cells, SUR at the ultimate level. A full run takes 2.5
minutes.[^note]

* **The ELPD column is empty by construction**: mack has no density, and every neural
  held-out cell sits past the pinned standardization frontier.
* **Mack beats every neural entry on CRPS**: 3,204 per cell, against deeptriangle 4,233,
  resnet 4,390, mdn 5,266 and `nn_transformer` 10,844. Mack had 10,000 draws and the neural
  entries 500, and the CRPS estimate's bias favours mack slightly.[^note]

The 0.5.5 update (PR #98) added three paid rows using the case-reserve head:[^pr-98]

* The GRU backbone beats the transformer backbone at both training seeds: CRPS about 2.7
  against about 3.7 to 4.0 percent of premium. A second-seed refit cell is the evidence.
* Deeptriangle's case channel is a wash on CRPS and PIT (within what the seed alone moves)
  and about 1 point of premium worse on signed error at both seeds. A single-seed draft
  claimed otherwise; review refits caught it.
* Restricted to the origins the rollout actually simulates, 55.2% and 45.0% of terminal
  case levels fell below zero, against 0 of 900 observed. That is the disclosed overshoot
  of the head, measured. (In 0.5.6 `nn_paid_case` floors the case level at zero; the 3b
  rerun then showed 0.0% below zero and 52.7% and 42.8% exactly at zero.)

# Notebook 03c: the multi-line transformer on the newer data (2026-08-24 and 2026-08-25)

The first run over many companies of `nn_transformer_ml`: accident years 1998 to 2007,
cutoff 2007-12-31, 91 companies, 243 cohorts, six lines, using the reference study's
selection rule rebuilt from the mart. Requiring positive premium drops 4 cohorts. 57
cells, 9.8 hours on 0.5.6.[^note]

**A defect found by this run, fixed in PR #104 (0.5.7):** every entry's `predict()` and
the shared `predict_at` restarted `default_rng(seed)` identically for each fitted cohort,
so draws under a shared study seed were comonotone across cohorts. Mack showed an implied
correlation of 0.255; correlations measured before the fix ranged 0.49 to 0.98 across six
code paths. The fix derives a stream from (seed, method label, cohort identity, field,
cutoff) in `kernels/rng.py`. Notebook 3b's claim of independence "for every method here
except sur inside a company" was false for mack and SUR on the code 3b ran.[^pr-104]

The rerun on 0.5.8 (PR #106, 10.3 hours):[^pr-106]

* **Cohort-level calibration.** `nn_ml_ar` has the lowest held-out KS D on the board, 6.3,
  and passes. Deeptriangle 6.9 passes, mack 8.71 passes at p = 0.050, `nn_ml_joint` 8.75
  misses narrowly.
* **Premium-normalized held-out CRPS.** Deeptriangle 2.12, then mack 2.18; the multi-line
  arms come last among the neural entries (2.48 and 2.60).
* **Dollar CRPS.** The note says `nn_ml_joint` is second on the dollar board and gives
  "992 vs deeptriangle 953, mack 350" (see the caveat at the end). In the 0.5.6 run mack
  kept the lead in the dollar view, because one company is 63% of premium.
* The joint arm has the least biased neural ultimate total (+2.35% of premium).
* **Implied cross-line correlation, lowest to highest:** mack 0.000 (it was 0.255 before
  the fix), single-line entries sharing members 0.01 to 0.15 (resnet's 0.15 is
  disagreement between members, not dependence), SUR 0.05, joint 0.12. Dependence lives in
  the joint head, not in attention. Joint beats AR at both seeds on every column, but in
  the 0.5.6 run the seed alone moved CRPS by 0.26 to 0.36, as much as the gap between the
  arms.
* Mack's ultimate refusals depend on the random stream: 2 cohorts on the 0.5.7 streams,
  1 before, because the Gamma law checks simulated paths.
* Runtime in the 0.5.6 run: joint about 1 hour, AR about 3.2 hours (2.7 of them the
  rollout).

Caveat: if lower is better, the three dollar numbers (992, 953, 350) do not put the joint
arm second. The note does not explain the ordering or the units; read the notebook's own
board before quoting them.

[^note]: Project status log, entries of 2026-07-28 (lines 1269-1291), 2026-08-03 to 2026-08-06 (lines 1495-1626) and 2026-08-25 (lines 387-482)
[^pr-69]: Notebook 03's corrected boards
[^pr-98]: Notebook 03b on 0.5.5
[^pr-104]: Per-cohort draw streams
[^pr-106]: Notebook 03c rerun on 0.5.8
