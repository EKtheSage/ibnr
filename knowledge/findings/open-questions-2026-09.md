---
type: Finding
title: Open questions as of late September 2026
description: The design calls, reruns and loose ends the agent's status note still listed as open, each with the date it was raised; check each against the repository before acting, since some may have been settled since.
tags: [roadmap, open-questions]
status: draft
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2026-12-31T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, open items across entries dated 2026-07-20 to 2026-09-25"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: issue-151
    resource: https://github.com/EKtheSage/ibnr/issues/151
    title: Metric-layer redesign, parked
  - id: issue-129
    resource: https://github.com/EKtheSage/ibnr/issues/129
    title: Leaderboard and compartmental reruns
---

Each item below was open on the date given. None is stated as open today; the note was
last written on 2026-09-25.

# Design calls waiting on Ethan

* **ELPD for the neural entries at the development frontier** (raised 2026-07-28, PR #69).
  On a set of held-out cells that is consistent with what the fits trained on, the deepest
  scored cell always sits at the fit's development frontier, where the validation split
  leaves the per-lag normalizer pinned, so `log_lik_at` refuses all 100 of 100 cells.
  Neural entries keep their CRPS cells. Giving them ELPD needs a frontier-aware validation
  split or a tail treatment.[^note]
* **A feature-past-cutoff clamp, and a practical-equivalence band to replace the fixed
  `z_tol = 4` in parity checks.** Listed as the two open design questions from 2026-08-06
  on. The note does not describe the clamp further.[^note]
* **"Arm A" comparison**, queued. The note never says what Arm A is.[^note]
* **A prospective one-year CDR through the gallery route** (2026-07-29). `next_diagonal`
  builds cells only from observations that exist after the cutoff, so the gallery route is
  backtest only. The missing piece is a constructor for cells with no outcome
  (`HoldoutCells` requires a `value`), which is a design question.[^note]
* **A `hierarchical` compartmental variant** with per-line prior medians from the mart and
  wider coefficients of variation, proposed 2026-07-21 and not built.[^note]

# Work waiting on something

* **The live held-out leaderboard rerun over the mart** (milestone 6's real remaining work,
  2026-07-29; filed as issue #129 on 2026-09-07 together with a compartmental rerun on
  0.5.9). No pilot results exist; the stash holding the old ones was dropped on
  2026-08-03 at Ethan's request.[^note][^issue-129]
* **Metric-layer redesign** (in the style of R's yardstick, Arrow in and Arrow out), parked
  as issue #151 on 2026-09-24. It waits for the pandas removal to reach
  `kernels/point_scores.py`; do not build it before then.[^issue-151]
* **The Balona and Richman (2020) reproduction gap** (2026-09-16): the paper's selected
  winners do not reproduce. The Swiss baselines match; the quarterly basic generalized Cape
  Cod is off by about 0.5. Causes not established. Leads named: the Cape Cod decay and
  premium convention, the RMSE weighting, the date axis, and applying exclusions before
  the history window.[^note]
* **Speed benchmark, round 2** (2026-07-24): more operations still pending (MSEP and CDR
  timings, interop round-trip cost, memory footprint). See [speed benchmark
  scaling](/findings/speed-benchmark-scaling.md) for the duckdb `latest_diagonal`
  lead.[^note]

# Smaller items flagged, not done

* `CLAUDE.md` and `sur/card.md` cite Zhang (2010) under the wrong title. The published
  title is "A general multivariate chain ladder model", Insurance: Mathematics and
  Economics 46(3) (flagged 2026-09-20 and 2026-09-22).[^note]
* `copula_glm` keeps its own `_nearest_pd`; `mcl` is not wired into
  `scripts/compare_gallery.py`; the tlrn head test could add a low-cutoff case
  (2026-09-22).[^note]
* `analysis/README.md` lacks rows for notebooks 03b and 03c (2026-09-22).[^note]
* Compartmental ignores the `idata.attrs` backend label under a foreign NUTS sampler
  (filed as a follow-up task 2026-09-07).[^note]
* PyMC's blow-up at `target_accept` 0.8 on company 11347 is explained only as
  under-adaptation of the centered parameterization (2026-07-20). See [PyMC speed on
  Windows](/findings/pymc-speed-on-windows.md).[^note]
* Yank 0.2.0 on PyPI (it misreports its version); open on 2026-07-28. See [cut a
  release](/playbooks/cut-a-release.md).[^note]
* The conformal-prediction research run on 2026-07-25 has no recorded conclusion; see
  [the machine-learning family plan](/decisions/ml-family-plan.md).[^note]

[^note]: Project status log, open items across entries dated 2026-07-20 to 2026-09-25
[^issue-151]: Metric-layer redesign, parked
[^issue-129]: Leaderboard and compartmental reruns
