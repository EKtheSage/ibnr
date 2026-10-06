---
type: Decision
title: Milestone 6 evaluation design
description: "The held-out leaderboard publishes CRPS and ELPD side by side with no default sort; ODP and Clark are not ELPD-eligible because a quasi-likelihood is not a density; density and draws are two independent capabilities; plus the scope Ethan set and what was settled while building."
tags: [evaluation, leaderboard, elpd, crps, milestone-6]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note milestone6-eval-decisions.md (private, outside the repository)
    title: Milestone 6 eval decisions
    last_modified: 2026-07-26T20:22:06.320Z
  - id: pr-25
    resource: https://github.com/EKtheSage/ibnr/pull/25
    title: m6-holdout-slicing (kernels/holdout.py)
  - id: pr-26
    resource: https://github.com/EKtheSage/ibnr/pull/26
    title: m6-densities (kernels/densities.py)
  - id: pr-27
    resource: https://github.com/EKtheSage/ibnr/pull/27
    title: m6-csr-scorer (ScoresHeldout and the CSR scorer)
  - id: pr-28
    resource: https://github.com/EKtheSage/ibnr/pull/28
    title: The review that reversed the ODP decision and fixed six bugs
  - id: pr-33
    resource: https://github.com/EKtheSage/ibnr/pull/33
    title: m6-forecast-object (kernels/forecast.py, PredictsHeldout)
---

Milestone 6 is the held-out evaluation of the gallery: score each model on the next
diagonal of a triangle after a cutoff date, by ELPD (expected log predictive density,
which needs a proper density) and by CRPS (continuous ranked probability score, which
needs draws). These are Ethan's calls; do not reopen them without asking.[^note]

# Decision 1: no single ranking currency

Decided 2026-07-25, against the design workflow's recommendation. CRPS on held-out cells
and held-out ELPD are published side by side with **no default sort key**; the reader
picks. Rejected: making CRPS the sort key with ELPD as a second, non-ranking tier. So
`leaderboard()` must not silently pick a sort; both columns are first-class, and
"leaderboard works" means both are populated and correct.[^note]

# Decision 2: ODP and Clark are not ELPD-eligible

First decided 2026-07-25 the other way (convert ODP's lattice density by applying
`-log(phi)` and rank ELPD globally), then **reversed the same day by external review (PR
#28)**.[^pr-28] The premise given to Ethan was wrong: the `-log(phi)` carry had been
described as "a convention, not a theorem", and it is neither.[^note]

* `exp(odp_lpdf)/phi` **does not integrate to 1**, and the shortfall **varies with
  `mu/phi`**: 0.688 at 0.5, 0.834 at 1.0, 0.947 at 2.0, approaching 1 only for
  `mu/phi >= 20`. So it is not even a constant that cancels between models on the same
  cells: it tilts a ranking toward whichever model puts more mass in low-`mu` cells, that
  is, the tail.
* ODP is a *quasi*-likelihood. For non-integer `x/phi` it is Poisson only up to
  proportionality (England & Verrall 2.3.5), so it never was a normalized law and no
  change of variable makes it one.

So `england_verrall_odp` and `clark_growth_curve` are **not** ELPD-eligible. They get CRPS
and PIT until given a real predictive distribution (negative binomial or Tweedie, a
modelling decision). `MEASURES` holds only `amount` / `log_amount` / `loss_ratio`, and
`REJECTED_MEASURES` records the reason.[^note]

How the wrong version survived review the first time is its own gotcha: [A normalization
check must integrate over the data's space](/gotchas/normalization-check-over-the-data-space.md).

**ELPD is not Bayesian-only** (the review's framing). Any model with a normalized
predictive density qualifies: distributional NN heads, a GLM with a declared observation
model. Being fitted by MCMC does not qualify a model; having a proper density does.
Point or quantile predictors, ODP, and bootstrap-wrapped deterministic entries do not.[^note]

# Why one global ELPD ranking is hard

It requires every entry on **one** measure and **one** set of observations. Worked out
from the audit of the entries:[^note]

* CCL and CSR are densities on **log** cumulative loss, so they need `- log C` to reach
  the amount scale.
* The increment-to-cumulative Jacobian is **1**, because `C_prev` sits on the training
  diagonal and is data. So amount-scale cumulative and amount-scale incremental are
  already the same measure.
* ODP and Clark are a **lattice** density. The plan when this list was written was
  `-log(phi)`; that is the step the reversal in Decision 2 withdrew.
* compartmental/lognormal is on **loss ratios**, so it needs `- log(premium)`.
* compartmental scores **two blocks** (paid and outstanding) where every other entry
  scores one. Observation sets must match to rank globally, so the global ELPD should be
  restricted to the common field (paid) with the outstanding block reported separately.
  This one is not a Jacobian and cannot be fixed by one.

**How to apply:** every density carries its measure explicitly, and the conversion is
applied in **one** place in `kernels`, never per entry. A test must assert that the
converted densities integrate to 1 on the amount scale: a wrong Jacobian does not look
like a shift, it looks like a differently shaped model, so a value-only comparison will
not catch it.[^note] See [the inert parameter bug class](/gotchas/inert-parameter-bug-class.md)
on why the test must be shown to fail first.

# Decision 3: density and draws are two independent capabilities (PR #33)

A forecast has a density (`ScoresHeldout`, giving ELPD) and/or draws (`PredictsHeldout`,
giving CRPS), and **each gets its own set of member models, its own intersection of cells
and its own fingerprint**. This is not tidiness. ODP has good draws and no usable density,
so one shared set of cells would let ODP's refusal of about half the Schedule P mart
delete cells from CSR's ELPD, a column ODP is not even on. Measured: ODP refusing one
cohort took the CRPS cells from 8 to 4 and left the ELPD cells at 8.[^pr-33]

Consequence: `elpd` and `crps` on one leaderboard row can rest on different cells, so
nothing is ever labelled with a bare `n_cells`.[^note]

# Ethan's three calls, 2026-07-26

Asked because each changed the code:[^note]

1. **Grow the pull request** to add `PredictsHeldout` and CSR draws rather than ship an
   ELPD-only board; his "both columns populated" bar from 2026-07-25 held.
2. **A model refusing a cohort shrinks that score's cells for everyone**, itemised in
   `panel.dropped`. Rejected: a minimum-coverage rule, and separate cells per model.
3. **`-inf` propagates** (with `n_cells_zero_density` explaining it) rather than the cell
   being refused. A model that gave the outcome zero density should rank last, not escape
   and rank on its easy cells. Note for later: a paired `elpd_diff` then gives
   `-inf - (-inf) = nan`, so the pairwise layer needs its own rule.

**Declined in PR #33, with the reason:** a standard-error column. The honest clustering
unit is the **fit**, and the fit is not the cohort for four candidate entries (`sur` and
`copula_glm` fit once per company across its lines; both NN entries fit once over the
whole set of companies), so a cohort-clustered standard error is wrong for all four in the
same direction: too small. `panel.by_cohort` is the input when this is built.[^note]

# Milestone 6 scope (Ethan, 2026-07-25, after the reviews)

1. **Full cross-family board now**: `predict_cells` for `nn_transformer`,
   `nn_transformer_ml`, `sur`, `copula_glm` and `deterministic/mack` as well as the
   Bayesian entries. This absorbs most of milestone 7. The transformer is the cheap one:
   `_rollout` already builds per-cell draws and then sums them into ultimates, so it is a
   change of what is exposed, not new modelling.
2. **ODP and Clark get a Tweedie variant** (1 < p < 2: continuous with a point mass at
   zero, which fits incremental paid and the mart's many zero cells better than negative
   binomial). That makes them ELPD-eligible through a real predictive law rather than the
   rejected lattice carry. It is a switchable variant, so the England-Verrall reference
   arm stays intact.
3. **Stacking uses two cutoffs**: weights fit on the diagonal after `as_of=1996`,
   evaluated on the one after 1997. About twice the retrospective wall-clock time through
   the parallel harness; the only version whose blend score is genuinely out-of-sample.
4. **The evaluation cells are the 152-cell set** (60 companies passing the entry criteria
   on 2 or more lines), the `compare_gallery.csv` set from milestone 3, so the new numbers
   sit beside the published point and CRPS results. Task name `paid_next_diagonal_v1`;
   paid and reported boards stay separate.[^note]

# What was built (as of 2026-07-26)

* **PR #25 `m6-holdout-slicing`**, `kernels/holdout.py`: `next_diagonal()` /
  `HoldoutCells`. `D_next` is **read** from `triangle.eval_dates`, never computed as
  `D + one grain` (annual Schedule P hides that bug). The anti-join is on the **full**
  cell key, so a merely restated training cell is not scored as new. Premium is attached
  from the training slice. Unscorable cells are excluded with a counted reason:
  `new_origin`, `dev_beyond_trained`, `no_predecessor`. An 8x8 triangle with a 9th origin
  gives 7 scored and 2 excluded. There is deliberately **no** `horizon` argument: two
  diagonals ahead is a nested integral or an autoregressive rollout, a different
  computation, and adding the setting first is the inert parameter bug class.[^pr-25]
* **PR #26 `m6-densities`**, `kernels/densities.py`: the normal / lognormal / ODP
  catalogue plus `to_amount_scale()`, the single place where measure changes happen.
  `check_normalization()` is the guard rail.[^pr-26]
* **PR #27 `m6-csr-scorer`**: the `ScoresHeldout` mixin (`gallery/entry.py`) plus
  `CellIndex` / `training_index` / `index_into` and `meyers_csr/scorer.py`. **The pattern
  every other entry follows**: a free function over plain arrays beside `model.stan`,
  taking `(contract, post, cells)` and returning the density on the entry's **own**
  measure; the mixin's concrete `log_lik_at()` applies the conversion so an entry cannot
  skip it. **Scorers read `idata.posterior`, never `idata.log_likelihood`**: that group is
  not uniform (Stan's `log_lik` against the ports' `obs`, NumPyro adds scalar `*_prior`
  factor sites, Clark's PyMC port has no group at all), while `alpha` / `beta` /
  `speedup` / `sig` are transformed parameters both ports re-expose as
  deterministics.[^pr-27]
* **PR #33 `m6-forecast-object`**, `kernels/forecast.py`: `logmeanexp`,
  `CohortForecast`, `align_panel`, `leaderboard`, plus the `PredictsHeldout` mixin and
  CSR's `draw_cells`. Still open after it: standard errors, pairwise `elpd_diff`,
  stacking, evaluation over several cutoffs, and the per-entry wiring for the other ten
  entries.[^pr-33]

# The agreement check

This is the milestone's key correctness device: score the **training** cells through the
held-out code path and reproduce the fit's own `log_lik` element by element. Its
tolerance is **1e-5, set by cmdstan's `sig_figs=6` CSV output**, not by the scorer. Draws
and reference both round-trip through that file, so recomputation cannot agree better
than about 1e-6 relative (measured 3e-7). Do not chase this to 1e-10. Instead assert that
the negative control (perturb one parameter by 1%) lands more than 100 times further away.
A real index slip is of order 0.1.[^note]

# Six bugs PR #28 fixed that the shipped tests could not see

All mutation-verified. Four from the review of #25 and #26:[^pr-28]

* `prev_value` used the contract's `prev_idx`, which is the previous **origin**
  `(w-1, d)` (CCL's AR(1) link), not the previous **development** `(w, d-1)`. CSR never
  reads it, so no CSR test could catch it.
* `D_next` was read before filtering to the scored fields, so a premium-only restatement
  became "the next diagonal": an empty hold-out, with the real one skipped.
* `new_origin` was decided triangle-wide, so one cohort's history vouched for another's,
  at the oldest and largest-reserve origin.
* Incremental triangles got cumulative differencing.

Two from the review of PR #27, the first being the subtle one:

* **`index_into` checked cells against each other, not against the fit.** The first fix
  only rejected cells spanning several cohorts. A single, entirely **wrong** cohort agrees
  with itself perfectly, indexes cleanly, and returns a complete plausible ELPD for a
  company the fit never saw. Reproduced: a `FIT_CO` contract accepted `OTHER_CO` cells,
  and a paid-loss contract accepted `field="reported_loss"`.
* Root cause: the contract recorded **no** identity. `kernels/contract.py` now stores
  `segment` (dict) and `fields` (tuple) on all three `*_stan_data` builders through
  `_cohort_identity`, and `index_into` requires them and compares. `field` now
  **defaults** to the contract's own: a caller who can pass the field is a caller who can
  pass the wrong one.

**Declined from the #27 review:** renaming `prev_idx` to `prev_origin_idx`. `prev_idx` is
Stan's own name, declared in `model.stan`'s data block, so renaming it forks the published
data contract. The ambiguity is answered in `contract.py`'s conventions docstring instead,
pointing both ways.[^note]

# Settled while building, worth not working out again

* Restricting `origins` also shrinks the trained development depth, so the window's
  oldest origin legitimately loses its deepest held-out cell. Correct, not a bug.
* `Triangle.origins` / `.eval_dates` return `dt.date`, but `.execute()` returns
  `datetime64`. `contract._as_date` is the established normalizer; reuse it rather than
  inventing a second convention.
* Integrating an increment-space density from 0 gives about 0.73, because the support is
  `(-C_prev, inf)`. That is a bound error, **not** a Jacobian error; do not "fix" it with
  a constant.[^note]

Related: [Held-out draws must declare their scale](/gotchas/declare-the-scale-of-held-out-draws.md).

[^note]: Milestone 6 eval decisions
[^pr-25]: m6-holdout-slicing (kernels/holdout.py)
[^pr-26]: m6-densities (kernels/densities.py)
[^pr-27]: m6-csr-scorer (ScoresHeldout and the CSR scorer)
[^pr-28]: The review that reversed the ODP decision and fixed six bugs
[^pr-33]: m6-forecast-object (kernels/forecast.py, PredictsHeldout)
