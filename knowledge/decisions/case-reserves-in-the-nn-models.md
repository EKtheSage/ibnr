---
type: Decision
title: Case reserves in the NN models (0.5.4 to 0.5.5)
description: "The closed 0.5.4-0.5.5 build arc that brought case reserves and incurred loss into the NN models (the case-reserve head, the joint paid-and-case entry nn_paid_case, live fits in notebook 3b), with the package facts verified for it and the two questions it left open."
tags: [nn, case-reserves, incurred, notebook-3b, release]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2026-11-05T00:00:00Z
sources:
  - id: note
    resource: agent memory note next-build-054-plan.md (private, outside the repository)
    title: The 0.5.4 to 0.5.5 build arc and the notebook 3b update
    last_modified: 2026-08-06T10:20:06.269Z
  - id: pr-91
    resource: https://github.com/EKtheSage/ibnr/pull/91
    title: 0.5.4 feature work (squash bd69c82)
  - id: pr-92
    resource: https://github.com/EKtheSage/ibnr/pull/92
    title: Release 0.5.4 (release commit 53468dc, tag v0.5.4)
  - id: pr-93
    resource: https://github.com/EKtheSage/ibnr/pull/93
    title: Notebook 3b ultimates section (e810dfd)
  - id: pr-94
    resource: https://github.com/EKtheSage/ibnr/pull/94
    title: deeptriangle case-reserve head
  - id: pr-95
    resource: https://github.com/EKtheSage/ibnr/pull/95
    title: nn_paid_case, the joint paid and case entry (main af42363)
  - id: pr-97
    resource: https://github.com/EKtheSage/ibnr/pull/97
    title: Release 0.5.5 (main c07d811, tag v0.5.5)
  - id: pr-98
    resource: https://github.com/EKtheSage/ibnr/pull/98
    title: Notebook 3b methods x fields update (main 7ee4350)
---

# Status

**Closed.** Everything in this build arc shipped. It stays here for its decisions, the
facts verified while planning it, and two questions that were still open on 2026-08-06
(the `stale_after` date is for those open items, which may have moved since).[^note]

* 0.5.4 on PyPI: tag v0.5.4, release commit 53468dc, feature squash bd69c82 (PR #91),
  release PR #92.[^pr-92]
* Notebook 3b's ultimates section landed (e810dfd, PR #93), executed from a cold start
  on 0.5.4. Its verdict: all six methods over-state paid ultimates and fail the paid PIT
  test (96-month paid is cash to date); all six pass on incurred; deeptriangle has the
  best CRPS on paid, mdn on incurred, nn_transformer is last on both. PR #93 also fixed
  the notebook's `PREMIUM_TOTAL`, which was 10 times too large (premium summed over all
  ten booking ages), so the multi-line section's error-as-a-percentage-of-premium
  numbers changed tenfold.[^pr-93]
* The case-reserve head (PR #94), the joint entry `nn_paid_case` (PR #95), the 0.5.5
  release (PR #97) and the notebook 3b methods x fields update (PR #98, executed from a
  cold start against PyPI 0.5.5) all merged by 2026-08-06.[^note]

# What Ethan asked for (2026-08-05)

1. Case reserves / incurred loss in the NN models, shipped as ibnr 0.5.4.
2. Notebook 3b gains a comparison of **ultimates** across methods and fields: NN paid and
   incurred, Mack paid and incurred, SUR paid. Point estimates, errors and distributions.
3. A CRPS definition at its first mention in the notebook.[^note]

The four questions put to him were answered on 2026-08-04: all four NN entries on
incurred, yes; SUR on incurred, add it now (against the paid-only recommendation); 0.5.4
released before the notebook update, yes; an ultimate-level CRPS column beside PIT,
yes.[^note]

# Decisions

* **The case-reserve head** (decided 2026-08-05, shipped in PR #94): deeptriangle's
  auxiliary head trains on the case-reserve **level** when channel 1 is a level, which is
  faithful to Kuo, replacing 0.5.4's refusal. The increment path is byte-identical. Ethan
  confirmed; the name "aux-on-level" was rejected. It is "the case-reserve head", with the
  target inferred from `field_kinds`.[^pr-94]
* **Arm C built straight away** (decided 2026-08-05). Ethan removed the "Arm A first"
  condition ("we should model this"): case reserves run down toward 0 as payments replace
  them, and also jump up on new information (a late hospital bill), so the rundown is
  non-linear and not monotone. It is a state with its own dynamics, not a static
  covariate. The joint entry simulates **both** channels forward, which also closes the
  disclosed limitation of features frozen during the rollout.[^note]
  * The head predicts (paid increment, case **movement**) jointly, with the case
    **level** reconstructed as state and fed back as an input channel at each rollout
    step.
  * The backbone is **both** a transformer and a GRU, as a config switch (his standing
    rule to build both variants).
  * Proposed by the agent and not objected to: the name `nn_paid_case`; the held-out
    board scores the **paid margin** of the joint density (the margin of a bivariate
    Gaussian mixture is a univariate mixture, so `PooledMDNHeldout` is reusable through
    `_forward_mixture`); realized case values are diagnostics, not board columns.
  * Shipped as PR #95 (merged 2026-08-05, main af42363), the sixth NN entry. Review left
    one disclosed approximation (a stale start when the case cell at the paid anchor is
    missing; listed as a limitation in the card) and one recorded non-defect (negative
    terminal case levels are legitimate: the mart books negative case reserves net of
    salvage).[^pr-95]
* **Ethan's directives, evening of 2026-08-05:**[^note]
  * **No notebook 3c, ever.** The comparison lives in notebook 3b, with every model
    trained and fitted **live** in the notebook: no CSV imports, no prior-run results.
  * Release 0.5.5 first, so 3b can pin a published wheel, after the mdn-card fix from a
    separate session landed (it did, as #96).
  * The 0.5.5 release swept a banned family of jargon words out of shipped prose
    (wording only; the sweep ended up covering all of `src`, `tests`, `scripts` and
    CLAUDE.md).
* **Notebook 3b design.** The methods x fields section gained `nn_paid_case` (both
  backbones, live pooled fits) and deeptriangle in its case-reserve-head configuration
  (feature = the case reserve level) on the **paid** arm. `nn_paid_case` sits out the
  incurred arm with a stated sentence: reported loss already contains case reserves, so
  targeting reported and case jointly would put the case movement inside both targets.
  That was the agent's call; Ethan can override it.[^note]

# Package facts verified for this work (2026-08-04)

Re-verified by five scout agents; two earlier claims were corrected.[^note]

* **Corrected:** `fit(feature_fields=...)` has existed on all five NN entries since the
  initial import, so there was no gap in what was exposed. Adding it to the configs as
  well was **rejected**: two competing sources for one setting is the inert parameter bug
  class (see [the inert parameter bug class](/gotchas/inert-parameter-bug-class.md)). The
  real 0.5.4 work was per-channel observedness `x_obs` (rollout cells promoted to
  observed were fabricating feature zeros, so missing and zero were ambiguous),
  `level_fields` for `case_reserve` (a level: differencing gives the movement, and the
  level itself could not be represented), and `heldout_n_draws`.
* **Corrected:** the NN held-out draw count was 1000 (the config's `n_draws` default
  through `_heldout.py`), not 500. Every other CRPS entry delivers 10,000. 0.5.4 added
  `heldout_n_draws` for the NN entries, separate from the rollout budget, so both sides
  of the CRPS board score on 10,000 draws (Ethan had asked why the counts differed).
* Mack's field is a fit argument (`loss_field="paid_loss"`), so Mack on incurred needs no
  package change. NN on incurred as a **target** is also config-only (the market study
  before milestone 6 scored transformers on `reported_loss`).
* No bootstrap is used anywhere in the ultimate paths. Mack ultimates come from
  `kernels.mack.simulate_ultimates` (parametric Monte Carlo from Mack's own conditional
  moments, process and parameter risk, 10k draws); NN ultimates from the autoregressive
  rollout (500 draws pooled over 5 members); SUR from the FGLS joint normal with
  cross-line covariance. The England-Verrall bootstrap in the package serves only the
  one-year CDR's diagonal generator.
* The mart carries `case_reserve` as its own field. It is a **level**, not a cumulative:
  its "increment" is the case movement.

# The CRPS definition placed at first mention

"The continuous ranked probability score (CRPS) generalizes absolute error to a
distributional forecast: it is the average absolute difference between a draw from the
forecast and the realized value, less half the average absolute difference between two
independent draws - the second term charging the forecast for its own spread, so neither
a wide hedge nor a false certainty scores well. For a point forecast it reduces to
absolute error; lower is better."[^note]

# Open as of 2026-08-06

1. **Features past the cutoff at prediction time.** Training masks every channel at the
   drawn cutoff, but the rollout and held-out paths mask features by observedness alone.
   On a ragged cohort, a feature booked deeper than every target cell reaches the network
   at a distance training never showed. Kept as is (the contemporaneous cell is
   informative) and disclosed in all five cards. The alternative is a one-line calendar
   clamp in `_heldout_inputs` and the rollouts.[^note]
2. **Arm A**, the case-reserve input channel compared with and without, on paired
   per-cohort CRPS with fixed seeds. Planned as a script and CSV first, and a notebook
   only if interesting. Arm C no longer depends on it; it remains available as an on/off
   comparison. (Note: the later "no notebook 3c, ever" directive above post-dates this
   plan's "notebook (3c) if interesting".)[^note]

The note also listed a second design question under "two design questions"; it was the
case-reserve head, decided on 2026-08-05 as recorded above.

Related: [Rules for a fair model comparison](/decisions/model-comparison-rules.md) and
[Making an analysis notebook portable](/playbooks/make-a-notebook-portable.md).

[^note]: The 0.5.4 to 0.5.5 build arc and the notebook 3b update
[^pr-92]: Release 0.5.4 (release commit 53468dc, tag v0.5.4)
[^pr-93]: Notebook 3b ultimates section (e810dfd)
[^pr-94]: deeptriangle case-reserve head
[^pr-95]: nn_paid_case, the joint paid and case entry (main af42363)
