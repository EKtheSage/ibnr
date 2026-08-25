# Changelog

This file starts at 0.5.0. For 0.1.0 through 0.4.0, read `git log v0.1.0..v0.4.0` -
those releases predate the file and reconstructing them now would be a summary of
a summary.

Versions follow [semantic versioning](https://semver.org/), loosely: while the
package is `Development Status :: 3 - Alpha`, a minor bump is free to change a
kernel signature. The public surface named in CLAUDE.md decision 8 (`Triangle`,
`gallery.list/fit/evaluate/stack/scaffold/leaderboard`) is the part treated as
stable. Nothing in it was removed or renamed in this release; it GREW (the
held-out evaluation pipeline, and a `segment` argument on three entry methods).
(As built that surface is `Triangle` plus
`gallery.list/get/fit/stack/leaderboard/next_diagonal/CohortForecast/Absence/align_panel/SCORE_DIRECTION`
- `evaluate` is a method on a fitted entry and `scaffold` is planned, per the
corrected decision 8.)

## 0.5.8 - 2026-08-24

A one-feature release: `nn_transformer_ml` gains held-out scoring (#105), the
adapter notebook 3c's first execution had to work without. Purely additive -
no existing entry's draws or densities change on this version.

### The multi-line transformer scores held-out cells (#105)

The entry's fitted cohort is a company - all of its lines are one training
example - while a held-out cohort built by `next_diagonal` is a single
(company, line) pair. The new `gallery/nn/_heldout_ml.py` bridges the two: it
slices a per-(company, line) contract out of the company-shaped one (putting
`line_of_business` back into the segment key, with the same training-closure
and identity guards every other entry answers to), and the entry now
subclasses `ScoresHeldout` and `PredictsHeldout`, so `log_lik_at` and
`predict_at` work through the ordinary calls. The forward pass conditions on
everything the company observed across all its lines; the scored line's
predictive is read off the head - under the `"joint"` head as that line's
marginal of the multivariate mixture, whose weights are unchanged and whose
per-component scale is the matching row of the Cholesky factor. The density
algebra and draw loop were extracted from `gallery/nn/_heldout.py` and are
shared, not copied; the single-line entries' behavior was verified unchanged
byte for byte. `TransformerMLConfig` gains `heldout_n_draws = 10_000` to match
its siblings, and per-cohort draw streams (0.5.7) apply to the new path
automatically.

## 0.5.7 - 2026-08-24

A one-fix release: per-cohort draw streams (#104), found by notebook 3c's
multi-company study (#103). Draws from entry-level `predict`, `predict_at` and
`cdr_distribution` change for a given integer seed - a different sample from
the same distribution - so any pipeline pinning those draws byte-for-byte
re-pins on this version. The kernel functions' own seed behavior is untouched.

### Gallery draws stop sharing one noise stream across cohorts (#104)

Notebook 3c measured a defect in how the gallery turned a seed into random
numbers. A gallery entry is fitted to one cohort, so a study over twenty-five
companies is twenty-five separate fits, and a script that wants reproducible
results passes all of them the same seed - which is what notebooks 3b and 3c
do. Every `predict` and the shared `predict_at` answered that seed with
`np.random.default_rng(seed)`, starting the same generator from scratch each
time, so draw `i` of every company read the same underlying random numbers and
the companies' simulated ultimates rose and fell together. For `mack` the mean
implied correlation between cohorts came out at 0.255, where independent draws
at 10,000 draws would sit near 0.01 of sampling noise.

Each cohort's own distribution was never affected. What was affected is every
quantity read across cohorts *within a draw*: a company total, a panel total,
the spread of either, and any calibration statistic computed from those sums.

* **The fix.** Entries now derive a stream from the caller's seed before drawing
  anything. The new `kernels/rng.py` builds a short text naming what is being
  drawn - the method (`predict`, `predict_at`, `cdr_distribution`), the cohort's
  segment identity, the loss field, the training cutoff - hashes it with sha256,
  and folds eight 32-bit numbers from that digest into a
  `numpy.random.SeedSequence` behind the seed. sha256 rather than Python's own
  `hash()`, which is salted differently in every process and would make a rerun
  of the same script produce different numbers. Wired into
  `PredictsHeldout.predict_at` (the one shared held-out path, so it covers the
  pooled NN entries too), the nine single-cohort `predict` methods, and both of
  the `mack` entry's kernel calls.
* **What changes for users.** Draws from an entry's `predict`, `predict_at` and
  `cdr_distribution` differ from 0.5.6 for the same seed. Each cohort's own
  distribution is statistically unchanged - the numbers are a different sample
  from the same distribution, not a different distribution. The same seed and
  the same cohort still reproduce bit for bit, in any process and on any
  machine. A `Generator` or `SeedSequence` passed as `seed` is used exactly as
  given: a caller who built their own stream gets that stream, not one derived
  from it, which is also how two entries can still be held on common random
  numbers when a comparison wants that.
* **What does not change.** The kernel functions are untouched:
  `kernels.mack.simulate_ultimates`, `kernels.mack.draw_next_cells` and
  `kernels.cdr.simulate_one_year_cdr` called with a plain integer seed are
  byte-identical to 0.5.6, so the CDR byte pins and the R `ChainLadder` tie-outs
  still hold. Their `seed` annotation widened to
  `int | np.random.SeedSequence | None`, which is documentation of what
  `np.random.default_rng` already accepted, not a behavior change. The NN
  entries' `predict()` is deliberately untouched as well: one pooled fit runs a
  single cached rollout and slices it per cohort, so its cohorts already read
  different positions of one stream and the defect never arose there.

## 0.5.6 - 2026-08-12

A one-fix release: the case-level floor in `nn_paid_case` (#100), on `main`
since 2026-08-06 but not in any published wheel until now. Also ships the
`schedule_p` docstring correction (#99): `case_reserve` is reported minus paid
(net of bulk), not incurred minus paid - the stored values were always right,
the description of them was not.

### `nn_paid_case` floors the simulated case level at zero (#100)

A case reserve is booked down TO zero and never past it, and 0.5.5's rollout did
not say so: it advanced the state as `level += movement` with nothing stopping
the walk crossing zero, and roughly half the simulated terminal levels on the
Schedule P panel landed below it (55% with the transformer body, 45% with the
GRU, against zero of that panel's 900 observed case cells). The state update is
now `level = max(level + movement, 0)`.

* **New config field** `NNPaidCaseConfig.floor_case_at_zero`, default `True`. It
  is a rollout knob shared by both backbones, so it is not in `BACKBONE_KNOBS`
  and neither body refuses it. It is also in the rollout cache key, unlike every
  other config field, because it is the one knob a caller is meant to flip on a
  FITTED entry - keyed on `(n_draws, seed)` alone, that flip returned the other
  arm's cached array, byte-identical and with no error.
* **The draw is untouched.** The clamp is arithmetic on the case STATE, applied
  after the joint (paid increment, case movement) sample; the paid coordinate is
  written unchanged and the generator has already advanced. The two arms
  therefore consume the same random stream - same generator, same call order, a
  common-random-numbers pairing - and are byte-identical **until the first
  bind**, which is why `floor_case_at_zero=False` reproduces the 0.5.5 walk
  exactly (verified against the published 0.5.5 wheel in a clean venv, on
  `case_paths()` and `predict().samples`, on both backbones). Past the first
  bind the floored arm feeds the network a different case level, so its later
  samples legitimately differ.
* **The floored level is the only copy of the state**: it is what feeds channel 1
  forward on the next step AND what `case_paths()` reports, terminal and
  per-diagonal alike. A rollout that clamped the read-out while handing the
  network the unfloored level would pass every diagnostic; it is caught by test.
  One exception, pre-existing and unchanged by the floor: at a PINNED dev the
  channel-1 input is masked to the pin (standardized 0, the pooled level mean),
  so the network sees the pin there and not the level, floored or not.
* **The starting level is not floored.** `_initial_case_level` carries the
  deepest observed case level as the triangle reported it, negative included; a
  recovery can outrun the case estimate, and restating an observation is not
  constraining a simulation. The consequence is narrow: every origin the rollout
  projects has a cell on the first future diagonal, so a negative start is
  clamped there and never reaches `case_paths()`. Only an origin with no future
  cell at all can show one.
* **What moves and what does not.** `predict()`'s rollout ultimates CAN move
  with the floor on - the floored level feeds back and changes the next
  diagonal's mixture - and on the small test fixtures that movement is nonzero
  but small. The held-out board columns do NOT move: `predict_at` /
  `log_lik_at` are a single forward pass at observed features with no level walk
  in them, so they are byte-identical between the two arms (asserted, which
  doubles as proof that the two fits trained identically).

`floor_case_at_zero=False` is kept so the change can be measured with and
without it, not as a fallback. Measure it on `predict()`'s ultimates and on a
Meyers-style retrospective - **not** on the held-out board, whose rows are
identical between the arms by construction.

## 0.5.5 - 2026-08-06

The case-reserve arc built on 0.5.4's channel machinery lands: deeptriangle
learns from the case-reserve level it previously refused, and `nn_paid_case` -
the sixth NN entry - models paid development and case-reserve dynamics jointly.
Three PRs (#94, #95, #96) plus a repo-wide wording pass.

### The case-reserve head (#94)

0.5.4 made deeptriangle refuse a level at channel 1 of its auxiliary task by
name, because level-minus-increment is neither the outstanding increment nor
the outstanding level. That refusal is now replaced by the case-reserve head:
when channel 1 carries a level (inferred from `field_kinds`, no new knob), the
auxiliary MDN trains on the case-reserve level itself, masked to cells where
both channels are real. The increment path is byte-identical to 0.5.4 for the
same seed (measured), so existing fits are unchanged.

### `nn_paid_case` (#95)

A joint model of paid development and case-reserve dynamics, motivated by what
a case reserve is: a state with dynamics, not a static covariate - it runs down
toward zero as payments replace it and jumps when new information arrives. Per
cell, a K-component bivariate Gaussian mixture predicts (paid increment, case
movement) with full per-component covariance, so the correlation between payment
and case run-off is a learned per-cell quantity. The case LEVEL is an input
channel the rollout advances (`level += movement`, in ratio space) and feeds
back - both channels write back, both flags promote, so the frozen-feature
rollout limitation the other five NN entries disclose does not apply here. Two
switchable backbones (`config.backbone = "transformer" | "gru"`) share one head
module; foreign knobs are refused by name. Training is mixed-observedness: the
joint density where both targets are real, the closed-form margin where one is -
no cell discarded, no target fabricated. Held-out scoring takes the paid margin
of the joint density (a bivariate mixture's margin is a univariate mixture,
test-pinned against the raw head output) through the shared pooled-MDN path, so
the entry joins the board column-comparable at 10,000 draws. `case_paths()` is
the case run-off diagnostic: terminal simulated case levels per draw, or the
full walk over projected diagonals with `per_diagonal=True`. The case path's
calibration is unvalidated in this release (no realized-case board column) and
the card says so.

### Card fixes (#96)

The mdn card's evaluate example used a one-key `segment` that raises whenever
the pooled fit carries more than one line for a company (the normal case); it
now uses the two-key form, matching the `predict` call above it and the other
NN cards. The transformer card's placeholder set literal in the same slot is
spelled out the same way. `transformer_ml`'s one-key example is correct as it
stands - that entry's cohort unit is the company - and is deliberately
unchanged.

### Wording pass

Prose that described with-and-without comparisons through a lab-jargon term
now says what it means: "switchable", "comparison arm", "variant". Cards,
docstrings, comments and test prose across the package; one test function in
`tests/test_odp_bootstrap.py` renamed to `test_both_switches_off_is_refused`.
No behavior change.

## 0.5.4 - 2026-08-05

Case reserves (and any eval-date snapshot) become usable NN input channels
with honest semantics, and the NN entries' held-out CRPS draws stop being
rationed by the rollout budget. One PR (#91), adversarially reviewed.

### Per-channel observedness: `x_obs`

`kernels.nn_contract.nn_data` now returns `x_obs`, a per-channel usable-value
mask whose channel 0 equals `obs_mask` exactly, and every NN entry conditions
per channel: tokens are `[values * chan_flags, chan_flags]`, element-identical
to the old per-cell form on a single-channel fit (seeded draws are
byte-identical). Two defects the cards used to disclose as limitations are
closed. A rollout cell promoted to context no longer presents the contract's
padding zero as an observed zero feature increment - promotion raises only the
target channel's flag, so next year's features stay what they are, unobserved.
And deeptriangle's auxiliary outstanding head now trains only where BOTH its
channels are real; the fabricated pre-fix target at a punched fixture cell was
measurably negative outstanding, paid exceeding reported.

### `level_fields`: snapshots carried undifferenced

`fit(feature_fields=("case_reserve",), level_fields=("case_reserve",))` is the
new consumer spelling. A field named in `level_fields` skips differencing -
`case_reserve` is a snapshot whose difference is the case movement while the
informative quantity is the level - and `field_kinds` records each channel's
kind. The target cannot be a level, a level must also be a feature, and an
absent field is refused by name (`select_fields` is a filter, so the old
behaviour fit a dead all-masked channel silently - the refusal immediately
caught two test fixtures that had been doing exactly that since the
deeptriangle entry landed). deeptriangle refuses a LEVEL at channel 1 for its
auxiliary task by name - level-minus-increment is neither the outstanding
increment nor the outstanding level - with `config(aux_weight=0.0)` as the
single-task escape.

### `heldout_n_draws`

The four held-out-capable NN configs gain `heldout_n_draws = 10_000`, read by
the held-out draw path instead of the rollout's `n_draws` (a held-out diagonal
is one forward pass per ensemble member regardless of draw count, so 10,000 is
cheap there and was not in a rollout). The NN rows of a CRPS board now rest on
the same draw count as every other entry; they carried 1,000 before, the
rollout default inherited by coincidence. `transformer_ml`'s config does not
gain the field - it has no held-out surface, and an unread field would be an
inert parameter.

### One disclosed edge, kept deliberately

Training gates every channel at the drawn augmentation cutoff; the rollout and
held-out paths gate features by observedness alone. On a ragged cohort whose
feature is booked on a deeper calendar diagonal than every target cell, the
network therefore conditions on a feature cell at a distance training never
showed. Kept, because that contemporaneous cell is genuinely informative; all
five cards state it.

## 0.5.3 - 2026-08-04

Released without a changelog section at the time; backfilled here from the
commit body. One consumer-visible change: Schedule P gold-mart downloads go
over anonymous HTTPS (stdlib urllib) with the `gh` CLI demoted to a fallback,
so a fresh clone with no GitHub tooling works.

## 0.5.2 - 2026-07-29

Two review findings against 0.5.1's own work, no public surface change and no
shipped behaviour change - both defects are in a study script and a test guard.
Released rather than held because one of them would have put unreliable MCMC
draws on a published leaderboard.

### The parameter-count guard could not see an unpinned count

0.5.1 pinned every parameter count the NN cards disclose. The regex that read
those values was also answering "does this card disclose a count?", and in that
role it could not see `~30k parameters` - the exact prose format the guard was
written to stamp out, and the one `mdn/card.md` had carried alongside a
transformer figure the transformer's own card had already retracted. A card
could revert to prose, sit in the "discloses nothing" list, and pass.

Detection is now a separate, wider pattern, used both ways: a card listed as
disclosing nothing must contain no count-shaped phrase at all, and a card that
DOES disclose must have every such phrase inside a bolded claim that a builder
rebuilds. It requires the literal word "parameters" after the number, which is
what keeps it off the historical mentions the cards legitimately carry ("quoted
~120k, which was never the number").

### Four defects between the worker pool and the published board

All in `scripts/heldout_leaderboard.py`, found by review of the script that
lands milestone 6's study. Two would have corrupted its output and two would
have aborted or misreported it.

* **A fit that failed its convergence gates at the final escalation stage was
  scored onto the board.** `run_retro` uses the gates to decide what to
  *re-run*; it does not discard a fit that still fails at the last stage,
  because that is the caller's decision - and this script was not making it. A
  fit with R-hat 1.4 contributed draws to a published board with nothing saying
  so. It is now downgraded to a `fit_failed` absence rather than dropped:
  `align_panel` intersects, so a silent drop would delete those cells from
  every OTHER model's column, making a badly-converged fit look good while
  costing everyone coverage. One `ConvergenceGates` instance is shared with
  `run_retro` so escalation and judgement cannot drift.
* **`heldout_stacking.json` was written only on success**, leaving a previous
  run's weights beside freshly-overwritten CSVs - an output directory that
  looked complete and self-consistent and was not. Written either way now, with
  the failure payload naming the reason.
* **A pool-machinery failure crashed the progress callback.** Those rows carry
  no `as_of` (the task never got far enough to have one), so indexing it raised
  `KeyError` inside the callback, aborting the study and hiding the real
  failure behind a missing-key traceback.
* **The documented default invocation could not run.** It selects every board
  entry, needing `[bayesian]` and `[nn]`, and a plain `uv sync` installs
  neither; it failed late inside a spawned worker as an `ImportError` about
  cmdstanpy. A preflight now names the extra and the models that want it and
  exits before the mart is touched, checking with `find_spec` so torch is never
  imported. The entry-to-extra map is derived from the registry.

## 0.5.1 - 2026-07-29

A patch in version number only where the CDR is concerned: the one-year CDR
became multi-method and then gallery-wide, `guszcza_growth_curve` closed the
last parity gap, and two drift guards landed. Nothing was removed or renamed and
no published number moved; `ibnr.gallery.__all__` grew by one name.

### Multi-method one-year CDR (out of band)

"The one-year CDR" was one method - Mack's - because the two things it does were
fused. They are now two axes, and only one of them is a choice.

* **Axis 1, what generates next year's diagonal, is now selectable.**
  `kernels.cdr.DiagonalGenerator` with two implementations: `MackDiagonal`
  (Mack's conditional moments, the previous behaviour) and
  `ODPBootstrapDiagonal` (England & Verrall's Pearson-residual bootstrap with
  over-dispersed Poisson process noise - the generator behind R's
  `CDR.BootChainLadder`). `simulate_one_year_cdr(fit, generator=...)` takes a
  method name or a configured instance.
* **Axis 2, how the reserve is re-estimated afterwards, is not a choice.**
  `kernels.cdr.rereserve(fit, next_diagonal)` re-runs the volume-weighted chain
  ladder on the extended triangle and differences the ultimates. One
  implementation shared by every generator - it is the market convention and
  what R uses for *both* of its CDR methods. It is public, so draws from any
  model that predicts next year's cells can be re-reserved into a directly
  comparable CDR.
* **`kernels/odp_bootstrap.py` (new)**: the bootstrap engine as free functions
  over plain arrays, written to be read against R's `BootstrapReserve.R` and
  documenting each of the four places it deviates.
* **The option surface**: `cdr_methods()` lists every route with what it
  generates, how it re-estimates, what it returns, what it requires of the
  cohort and - route by route - what it has actually been validated against.
  `get_cdr_method(name)` returns the descriptor, carrying the generator
  **class**, mirroring `gallery.get`.
* **`merz_wuthrich` is listed but is not a generator, and asking for it as one
  is refused by name.** The closed form linearizes the chain-ladder factor
  update around Mack's conditional moments; there is no version of it for
  another model. `one_year_cdr(fit)` is unchanged and remains the only way to
  reach it.
* **Mack's precondition moved off the shared path onto its own generator.**
  `require_positive_open_diagonals` was applied to every simulated CDR; it is
  Mack's (his conditional variance is proportional to the diagonal cell) and the
  bootstrap has the opposite requirement (non-negative increments, and it
  answers happily for an accident year with zero paid at 12 months). Each
  generator now states and enforces its own.
* Honest limits, stated in the card and the docstrings: the bootstrap route is
  validated **to Monte Carlo error against R's algorithm**, not to published
  digits - R's `CDR.BootChainLadder` example prints none, and a bootstrap is
  stochastic. The test suite transcribes `getNYCost` literally and requires
  agreement to 1e-10 on a shared diagonal, checks the process-only standard
  error against a delta-method reference computed off the re-reserving Jacobian,
  and cross-checks the fitted values against `england_verrall_odp`'s iterative
  proportional fit. Also documented: `E[CDR] = 0` is Mack's, so
  `simulated_msep`'s mean square about zero is the variance only on the `mack`
  generator - R reports `sd()` for its bootstrap route for the same reason.

**No published number moved.** `simulate_one_year_cdr`'s Mack path is
bit-identical to 0.5.0's for the same seed - verified over 49 arrays spanning
two triangles, both sigma rules, all three process laws, both `parameter_risk`
settings and two draw budgets, compared as raw bytes - and the R MW2014 golden
tie-out is untouched. `process`/`parameter_risk` default to `None` on the
signature instead of `"gamma"`/`True` so that "not supplied" is distinguishable
from "supplied"; the applied defaults are unchanged, and combining either with
an explicit `generator=` is refused rather than left inert.

### The one-year CDR opens to the gallery

`rereserve` being public was the door. **`ibnr.gallery.GalleryDiagonal`** is a
third `DiagonalGenerator` that takes a fitted entry with `PredictsHeldout` and
re-reserves the draws it already produces for the leaderboard's CRPS column, so
CCL, CSR, ODP, Clark and Mack reach a one-year CDR with no new theory. It lives
in the gallery rather than `kernels`, since it imports `PredictsHeldout` and the
re-export direction is gallery -> kernels only; `cdr_methods()` still lists the
route, naming the class as a string.

* **What the number is, because it will be misquoted.** It is the *chain
  ladder's* one-year CDR under model M's view of next year, not "model M's
  one-year CDR". Both differenced ultimates are chain-ladder ultimates and only
  the diagonal between them is the model's - the structure R's
  `CDR.BootChainLadder` already has. The honest alternative refits M on the
  extended triangle once per draw; it is not offered rather than approximated.
  Consequently `E[CDR|D_I] = 0` does **not** hold here (it is Mack's result), so
  read `mean_cdr` and `sd_cdr` from `cdr_risk_measures` and not
  `simulated_msep`, which folds a disagreement between two methods into
  something that reads as volatility.
* **Two limits, both refused by name rather than assumed.** The route is
  **backtest only** - `next_diagonal` builds cells only from observations that
  already exist after the cutoff, so a *current* valuation is not reachable this
  way (the `mack` and `odp_bootstrap` generators are unaffected and remain
  prospective). And it excludes the **NN entries and `compartmental`**, whose
  contracts keep no raw cumulatives for the training-history check below.
* **All three objects are bound to each other, not two of them.** A `MackFit`, a
  `HoldoutCells` and a fitted entry come from three calls. Two checks tie the
  cells to the fit; the third ties the entry to both, by comparing its own
  contract values against `fit.cum`. Without it, an entry refitted on restated
  *interior* history - latest diagonal untouched, so every other check passed -
  was accepted, and the total CDR mean moved from 0.27 to -367.69 with seven
  times the spread, every number finite. Found by review.
* `simulate_one_year_cdr(n_draws=...)` and `Mack.cdr_distribution(n_draws=...)`
  now default to `None`, meaning "this generator's own count".
  `DiagonalGenerator.resolve_n_draws` is the seam: a Monte Carlo budget for the
  two simulating generators (both resolve `None` to the previous literal 20,000,
  so no existing call changes) and the source's own size for a fitted posterior,
  which refuses a mismatched explicit count instead of resampling to it.

### Documentation and drift guards

* **A "Chain ladder & reserve risk" reference section** on the docs site. The
  CDR and Mack kernels had no reference page at all, so `cdr_methods()` - the
  discoverability feature - was not discoverable. Thirteen entries, all
  resolving statically under `dynamic: false`.
* **Every parameter count an NN card discloses is pinned to the network that
  builds it.** `transformer/card.md` had retracted a `~120k` figure and the
  retraction never reached `mdn/card.md`, which had copied the comparison; mdn's
  own "~30k" was 26,353. Cross-card references get a builder each, every `nn`
  entry must be classified as disclosing a count or not, and the "does not
  disclose" list is verified against the cards rather than trusted.

### Milestone 5 - `guszcza_growth_curve` joins the parity gate

The entry landed 2026-07-27, two days after milestone 5 was declared complete, so
it was the only Bayesian entry without NumPyro and PyMC ports and the only one
absent from `scripts/parity_gallery.py`. Its card had recorded the ports as a
follow-up task, so this was a deferral rather than a deliberate exclusion.

* **Both ports added** - `model_numpyro.py` and `model_pymc.py`, plus `_shared.py`
  for the pieces that are not PPL-specific (curve codes, tree depth, the data
  guard) so the two cannot drift apart. `BACKENDS` is now
  `("stan", "numpyro", "pymc")`.
* **4 of 4 parity against the Stan reference**, two WC companies as of
  1997-12-31, 4 chains x 2500 draws: every z-score below 2.4 against a tolerance
  of 4, zero divergences in every backend on every cohort, R-hat 1.00. Published
  in `analysis/results/{parity,convergence}_guszcza.csv` and tabulated in the
  card. New `kernels.parity.GUSZCZA_PARITY_VARS`.
* **`scripts/parity_gallery.py` gained `--nuts-sampler`**, which reaches the
  `pymc` leg only. This entry needs it: at the `adapt_delta = 0.999` its source
  specifies, PyMC's native PyTensor NUTS runs ~130x slower than the identical
  graph through JAX (measured 860 s against 6.7 s), the same situation
  `compartmental` documents. `convergence()` now reports the sampler's own label
  (`pymc:numpyro`) rather than the backend argument, so a published row says
  what produced it.
* **`max_treedepth` is a shared control on this entry**, not a cmdstan-only one:
  its default of 15 differs from both PPLs' default of 10, and holding it
  constant is part of what parity means. Only `parallel_chains` is refused by
  the ports.
* A standing requirement is now recorded in CLAUDE.md: a new `bayesian` entry is
  not done until it has both ports and a published parity row. What expired with
  this gap was the milestone number, not decision 7.

## 0.5.0 - 2026-07-27

The 0.4.0 wheel on PyPI was 49 commits behind `main`, so this release is mostly a
catch-up: milestone 5 finished, milestones 6 and 7 opened, the one-year CDR landed
out of band, and the package moved to numpy 2 and grew a test CI.

### Public API - the cohort vocabulary

Five gaps found by building `analysis/03` through the public API alone, and they
were one gap seen five times: **a fitted entry could not say which cohorts it
answers for**, so every caller reconstructed that fact by hand.

* **`GalleryEntry.cohorts()` (new, abstract)** returns every cohort the fit
  answers for, in `predict()`'s target order, as the cohort's FULL segment
  identity - including a column the fit's own key does not carry.
  **`cohort_index(segment)`** is the one resolver behind it: `segment` is a
  *filter* on this fit's cohorts, so any subset naming exactly one is accepted
  and one naming none raises (naming the fit's key and the supplied dict) rather
  than quietly scoring the fitted cohort.
* **`predict`, `realized_ultimates` and `evaluate` now take the identical
  leading `segment: Mapping | None = None`** on all 15 entries.
  `realized_ultimates` moved onto the ABC. Previously the NN entries took a
  segment dict and the other ten took none, so a cross-model outcome table
  needed a `family == "nn"` branch. Every existing call site passes its
  arguments by keyword, so no 0.4.0 caller moves.
* **A pooled NN fit can be scored on the mart's own cells.**
  `kernels.nn_contract` keeps display-only segments (`company_name`) out of the
  cohort key, so a pooled fit was keyed on two columns while `next_diagonal`
  built cells on three - and both `entry.log_lik_at(cells)` and
  `entry.at_cohort({...}).log_lik_at(cells)` failed, the first naming a column
  the caller had just passed. The cells are now re-keyed onto the fit's own
  schema at the mixin boundary (`HoldoutCells.narrowed_to`), with each dropped
  value **verified** against the fitted cohort first; `index_into`'s schema
  equality is left exact, and the narrowing never escapes, so a shared board
  still sees one segment schema. Backed by a new refusal in `nn_data`: a segment
  column dropped from the key must be a *function* of the key, or two cohorts
  would collapse onto one grid - silently, whenever their cells are disjoint.
  `nn_data`/`nn_company_data` gained `segment_columns` and `display` keys. That
  refusal was measured against the real mart before shipping, since the data
  model derives `company_name` through a LEFT JOIN and a null or second spelling
  would fire it on every pooled fit: on publish `20260613_041006` all four
  Meyers lines carry 353 company codes with zero null names and zero codes
  spelled two ways, and both the study's pooled panel (60 companies /
  152 cohorts) and the full `--nn-pool market` pool (221 / 405) build clean.
* **`gallery.get(name).config_class`** is the dataclass an entry's
  `fit(config=...)` takes, or `None`. Registration checks the declaration both
  ways - missing when `fit` takes a config, and stale when it does not.
* **`ibnr.gallery.__all__` gained `next_diagonal`, `CohortForecast`, `Absence`,
  `align_panel` and `SCORE_DIRECTION`**, the four steps that BUILD the panel
  `leaderboard()` consumes plus the direction the board has no default sort for.
  The rule, now written in the module docstring and CLAUDE.md decision 8: a name
  is exported if a caller must construct or call it to get from a fitted entry
  to a board row. `ibnr.kernels` re-exports the same names plus `ForecastPanel`;
  `kernels` still never imports the gallery, and there is a subprocess test for it.
* **For anyone subclassing `GalleryEntry` out of tree:** `cohorts()` and
  `realized_ultimates()` are now abstract, so a subclass that implements neither
  will not register. There are no such subclasses (`gallery.scaffold()` is
  unbuilt), and the CALL surface CLAUDE.md protects is unchanged.
* `kernels.multiline.multiline_data` now stamps `segment` and `measure`, the
  cohort identity SUR and the copula GLM answer for.
* Note on `analysis/03`: its committed run predates all of this. Its
  `drop("company_name")` workaround and `family == "nn"` branch still run
  correctly - they are simply no longer necessary.

### Packaging

* **Python 3.11 and 3.12.** `requires-python` is now `>=3.11,<3.13`, and the
  classifiers say the same. The ceiling is a deliberate choice rather than a
  limit of the code: the core install, `[polars]`, `[nn]` and `[viz]` were all
  measured resolving wheels-only on 3.13 and 3.14. `[bayesian]` and
  `[interop]` cannot follow, because both transitively pin numpy below 2 -
  through `arviz` 0.18 (pulled by `bayesblend` 0.0.8) and through
  `bermuda-ledger` 2.3.0 - and the newest numpy under 2 is 1.26.4, which does not
  support Python 3.13 at all. Since packaging metadata cannot say "3.11 to 3.14
  unless you asked for `[interop]`", the package claims one range for everything
  it ships and a 3.13 user gets the standard "requires a different Python"
  refusal (`pip` on a clean 3.13 venv: `Package 'ibnr' requires a different
  Python: 3.13.13 not in '<3.13,>=3.11'`). Revisit the cap when those two
  upstreams move. (An alternative
  that gated the two extras on a deliberately unregistered package name was
  built and then rejected: it only holds while nobody uploads that name.)
  Caveat worth knowing on the versions that are supported: on 3.12 `[bayesian]`
  installs but not wheels-only, because `bayesblend` pins `matplotlib==3.7.2`
  whose newest wheel is cp311, so 3.12 builds it from source.
* **numpy 2.** Development and CI now run numpy 2.4.6; a plain `pip install ibnr`
  resolves numpy 2.5.1 and pandas 3.0.5. The core floor stays `numpy>=1.26`
  because raising it to 2 would make `ibnr[interop]` and `ibnr[bayesian]`
  unsatisfiable on PyPI - the blockers are upstream metadata on code that works,
  and this repo bridges them with `[tool.uv] override-dependencies` rather than
  shipping metadata nobody can install. Measured: the core suite gives the same
  557 passed / 0 failed on the locked numpy 2.4.6 + pandas 2.3.3 and on numpy
  2.5.1 + pandas 3.0.5, and all extras together give 0 failed.
* **pymc 5.28.5 / pytensor 2.38.3** were not an optional upgrade. pytensor
  2.31.7 unpacks numpy's `einsum_path` result as a five-tuple and numpy 2.4
  returns three, so every LKJ-based compartmental test died on "not enough
  values to unpack" the moment numpy moved. Worth knowing before anyone pins
  pymc back.
* **`ibis-framework` is capped below 13.** 12.0.0 is what is locked and what every
  CI leg runs. Transforms are written around backend-specific ibis behaviour, so a
  major bump has to be a deliberate change with the dual-backend suite re-run.
* **`arviz<1` and `pymc<6`** in the `[bayesian]` extra. Both are measured breaks:
  arviz 1.x drops the `az.from_dict(posterior=...)` signature the parity tests
  use, and pymc 6.x changed `LKJCorrRV.rv_op`.
* **`chainladder>=0.9.2`** in `[interop]`, raised from 0.8.18 - the old floor let a
  fresh install resolve 0.8.26, which is not the version the tie-outs were
  validated against.
* **`py.typed`.** The package now ships the PEP 561 marker, so type checkers use
  the annotations instead of treating `ibnr` as untyped.

### Testing and CI

* **A pytest workflow, and gates that make a green run mean something.** Nine
  legs: core, core on 3.11, one per extra, everything at once, plus two that
  install without the lockfile and force numpy and pandas to their newest
  releases - the only legs that grade the resolution a downstream consumer
  actually gets. Each leg declares what it installed and which files it exists to
  exercise, and fails if a test was skipped because a package that leg installed
  could not be imported, if a named file contributed zero executed tests, or if
  the total falls below a floor.
* **`tests/test_import_purity.py`.** Walks every submodule under a blocker that
  refuses the optional extras, and checks the four public import paths pull in
  none of them. This is what makes "the core install stays light" a checked claim
  rather than a convention.
* A `test` dependency group carved out of `dev`, so a genuinely core-only
  environment can be built at all.

### Milestone 5 - cross-backend parity (complete)

* NumPyro and PyMC ports for `meyers_csr`, `england_verrall_odp`,
  `clark_growth_curve` and `compartmental`, each gated against the Stan reference
  posterior.
* **The parity gate was silently lenient.** It read `az.summary`, which rounds to
  three decimals, so any parameter whose MCSE rounded to zero scored a perfect
  z - precisely for the best-identified parameters. The published milestone-4 CCL
  figures came through this bug and are superseded.
* **`meyers_ccl`'s `a_ig` bound is load-bearing.** Stan declares
  `<lower=0, upper=1e5>` and both ports had left it unbounded, because the
  truncated *prior* mass is negligible - but the *posterior* piles into that
  corner at deep development lags and pulled `sig` 5-12% low. Restored in all four
  ports.
* `nuts_sampler` is exposed through every Bayesian entry's `fit()`.

### Milestone 6 - held-out scoring, stacking, leaderboard (in progress)

* `kernels/holdout.py` (which cells a fit at a given `as_of` is scored on),
  `kernels/densities.py` (the one place a density changes measure),
  `kernels/forecast.py` (the forecast object and the leaderboard) and
  `kernels/stacking.py` (bayesblend stacking over two panels).
* A forecast carries two independent capabilities - a density, giving ELPD, and
  draws, giving CRPS - each with its own panel membership, so one model's refusal
  cannot delete cells from a column it does not appear in.
* Held-out scorers and predictors for CSR, CCL, compartmental, guszcza, ODP,
  Clark, Mack and the NN family.
* `kernels/codec.py` - the wire format. `to_arrow`/`from_arrow` round-trip
  `PredictiveDistribution`, `Triangle`, `MackFit`/`MackFitPanel`, the CDR
  result, forecast panels and plain frames losslessly over Arrow IPC, with a
  summary-only JSON mode for callers that cannot take megabytes of draws;
  `peek_kind` routes a payload without decoding it. This promoted `pyarrow>=15`
  from the dev group to a core dependency - no new weight, since
  `ibis-framework[duckdb]` has always pulled it transitively, but the codec
  imports it directly and an inherited requirement can vanish in an upstream
  release.

### Milestone 7 - the rest of the NN family (in progress)

* New entries: `mdn`, `deeptriangle`, `resnet`.
* A shared NN training scheme and one shared held-out mixin across all four NN
  entries.
* `kernels/tuning.py`: random-search hyper-parameter search over NN configs.

### New gallery entries

* `guszcza_growth_curve` - hierarchical growth curve reserving (Gesmann/Guszcza).
* `mdn`, `deeptriangle`, `resnet` (see above).

### One-year CDR (out of band)

* `kernels/cdr.py`: the Merz-Wuthrich 2008 analytic one-year claims development
  result per accident year and in total, plus an "actuary in the box" re-reserving
  simulation, plus VaR/TVaR of the CDR loss. Ties out to R ChainLadder's published
  `CDR()` output to seven decimal places.
* `kernels/mack.py` gained a native distribution-free chain ladder and
  `fit_mack_many`, a batch fit that closes the multi-cohort gap against
  chainladder-python's vectorized point estimate.

### Milestone 9 - speed benchmark (complete)

* `scripts/benchmark_speed.py` times construction, cumulative/incremental
  conversion, `as_of`, grain changes, aggregation, parquet ingestion and Mack fits
  against chainladder-python on both ibis backends. Headline: scale decides -
  chainladder wins small in-memory transforms, ibnr wins at mart scale and on
  every Mack fit.

### Fixes

* **`fit()` is atomic across every gallery entry.** A failed refit used to leave a
  half-updated entry behind; it now leaves the previous state untouched.
* **Null segment keys are rejected at ingestion.** Every join in
  `triangle/transforms.py` is a plain equi-join, and SQL join equality is false
  for `NULL = NULL`, so a single null segment value silently deleted a whole
  cohort from `as_of`, `latest_diagonal` and `to_incremental` - identically on
  both backends, and reachable from the real mart.
* Premium must match the loss cohort, and fields must share a diagonal.
* The zero-variance guard survives density columns that mix finite and `-inf`.
* Two inert parameters removed (`line_embedding_dim`, and `n_lob` in
  `_mack_tail_variance`) - both accepted, neither ever read.

### Tooling

* ruff floor raised to 0.16 across the repo, and CI checks formatting again.
* Python code blocks inside markdown are linted and formatted like source
  (`scripts/lint_md_snippets.py`), which also rejects doc samples that do not
  parse.
