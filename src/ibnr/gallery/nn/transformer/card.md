# nn_transformer - masked-cell triangle transformer with a mixture density head

**Family:** nn (PyTorch; requires the `[nn]` extra at fit time - registration
does not)
**Lineage:** Kuo's *DeepTriangle* (pooled NN reserving across many
company x LOB triangles) reimagined as an attention model: instead of GRU
sequences per origin, every (origin, dev) cell of the triangle is a token and
a small encoder attends over the whole grid - development, accident-year, and
calendar structure are all learned positional embeddings.

## Data

`kernels.nn_contract.nn_data`: a *cohort* is one company x line of business.
Targets are **incremental loss ratios** (incremental loss / origin premium) -
the cross-cohort normalizer. Inputs may carry extra channels
(`feature_fields`, e.g. paid alongside reported). Increments require an
immediate-predecessor observation; the anchor for ultimates is each origin's
latest observed *cumulative* value. Cohorts with unusable premium are dropped
and reported.

**Observedness is per channel** (`x_obs`), and so is the context the network
conditions on. A feature can be missing where the target is observed, and
observed where the target is missing; the contract's zero at an unusable cell
is padding, and the channel's own flag is what says so. Before this the single
per-cell flag was the target's, so a feature hole was read as an observed zero
increment - and every cell the rollout promoted claimed observed feature values
it never had (see "Prediction"). A field named in `level_fields` is carried
UNDIFFERENCED - an eval-date snapshot such as `case_reserve`, whose difference
is the case movement while the informative quantity is the outstanding level;
it is a subset of `feature_fields` (it declares a channel's kind, it does not
add one) and `nn_data` owns the semantics and the refusals.

Per-(channel, dev) standardization statistics are computed from that channel's
own training-context cells only (never the validation diagonals). Devs with
fewer than two context values - in practice the deepest dev, observed only
on the held-out diagonal - are **pinned**: their standardized value is 0 by
definition (mean from all observed cells at that dev, std 1), and rollout
draws there are forced to 0, i.e. the pooled dev mean. Pinning replaced the
original inherit-earlier-dev-stats scheme, which denormalized tail cells at
mid-development magnitudes and systematically overstated fast-decaying
tails (worst for PPA in the first backtest).

## Network

Token = `Linear([channel values * channel flags, channel flags])` + origin
embedding + dev embedding + **relative calendar embedding** (distance past the
conditioning cutoff, clamped to [0, n_d]) + broadcast conditioning (LOB
embedding + normalized log premium). One flag per channel, so a value is never
read without the flag that says whether it is real; with a single channel (the
default, no `feature_fields`) that is the same projection over the same numbers
as the earlier per-cell flag, which is why the sizes below are unchanged.
Encoder: 2 pre-LN transformer layers, d_model 64,
4 heads, FFN 128, dropout 0.15, GELU. Head: mixture density network, K=3
Gaussians per cell on the normalized incremental loss-ratio scale.

**Size: about 70k.** Measured, not estimated: **70,121 parameters** on an 8x8
grid with one channel and 4 LOB levels, **70,505 parameters** on a full 10x10
Schedule P triangle (`analysis/03_gallery_api_comparison.ipynb` prints the
per-module table). 66,944 of those are the two encoder layers and do not depend
on the triangle's size at all; only the three position embeddings do, and they
are about 2% of the count. An earlier revision of this card quoted ~120k, which
was never the number this configuration builds - and the stale figure outlived
the correction by being copied into `mdn/card.md`, which is why both numbers
above are now pinned by `tests/test_nn_parameter_counts.py`.

**Why the calendar embedding is relative (v3):** forecasts live on calendar
diagonals beyond the training window, where v1/v2's absolute learned
calendar embedding never received a gradient - untrained parameters injected
exactly at prediction cells. Distance-past-cutoff is supervised directly by
the cutoff augmentation, and because the rollout re-encodes after each
sampled diagonal, inference only ever consumes distance-1 predictions - the
most supervised case. The trade-off, disclosed: absolute calendar-year
(inflation) effects are no longer explicitly encoded and must be inferred
from context values.

**Why an MDN head:**
- lognormal heads fail outright - net-of-bulk incremental losses go negative
  (case releases; the milestone-2 lesson);
- quantile heads cannot produce coherent *sums* - ultimates add many future
  cells, and marginal quantiles don't add;
- the MDN gives a proper sampleable per-cell density; a **deep ensemble
  (5 seeds)** supplies epistemic spread on top of the mixture's aleatoric
  spread. Both mechanisms of the gallery's distributional-head requirement.

**No company embedding in v1** - ~600 companies with ~55 cells each is a
memorization vector, not a feature. Company identity enters through its own
observed cells, LOB, and size (log premium).

**Exposure-aware sigma (optional, `config.exposure_sigma`, off by default).**
Targets are loss *ratios* to premium, so a flat MDN sigma on the ratio makes
the predictive **dollar** sd scale as premium¹ - a constant coefficient of
variation across company size. Actuarially, larger books are relatively less
volatile per dollar (CV shrinks with size), so this misfits the size
extremes: too-narrow small books (the PPA-on-paid calibration miss) and
mis-sized large ones. With the flag on, the sigma head carries a single
learnable log-dollar power `p = softplus(raw_p)`; sigma is multiplied by
`premium^(p-1)` about the pooled mean premium, so the effective dollar sd
scales as `premium^p`. `p` is initialized at **1.0**, which reproduces the
flat-sigma baseline exactly - the two arms are a clean on/off comparison
(`compare_gallery.py --nn-exposure-sigma`). `p < 1` widens small books and
tightens large ones; `p` is a single scalar (per-line is a future extension)
to stay conservative on ~150-cell backtests.

*Result (paid, 152-cell backtest, 2026-07-08).* The data learns
`p ≈ 0.965` (5 members 0.952-0.982) - below 1, the actuarially-expected
direction, but only slightly. Net effect is a small, consistent win:
combined KS 18.3 → 16.6, combined median CRPS/outcome 0.0267 → 0.0252,
reserve MAE 3.00 → 2.95 pts of premium, chain-ladder skill 1.94 → 1.60;
mixed per-line (commercial auto and PPA improve on KS, other liability
worsens). PPA still rejects - its miss is outcomes piled low (the post-1997
regime), a location bias no width lever can fix. Kept **off by default**: a
documented opt-in lever, not the headline. Compare arm:
`analysis/results/experiment_exposure_sigma.csv` vs `compare_gallery.csv`.

## Training

- **Pooled** over every cohort; a batch is 64 cohort-triangles.
- **Calendar-cutoff augmentation:** per cohort per epoch, draw a fake as_of
  cutoff uniformly in `[min_cutoff, val_cutoff - 1]`; condition on the
  sub-triangle, score NLL on later observed training cells. Every triangle
  yields ~5 distinct "predict the future diagonals" tasks per epoch - the
  main small-data multiplier. The cutoff gates every channel, so a feature
  past the drawn cutoff is masked, not read. One edge is deliberately NOT
  matched at prediction time: the rollout and the held-out path gate features
  by observedness alone, so on a ragged cohort whose feature is booked on a
  deeper calendar diagonal than every target cell and anchor, they present a
  feature cell at a distance training never showed - kept because that
  contemporaneous cell is genuinely informative.
- **Validation by eval_date:** the trailing `val_diagonals=1` observed
  calendar diagonal of the *training window* is excluded from all training
  contexts and targets; early stopping (patience 25) on its NLL, best
  weights restored. Test-period diagonals never enter fit at all.
- AdamW (lr 3e-4, weight decay 1e-2), gradient clip 1.0, max 400 epochs.
- All defaults are hand-chosen for the Schedule P regime and disclosed here;
  systematic HPO is deferred to `kernels/tuning.py`.

## Prediction

Autoregressive **diagonal-by-diagonal rollout** - the transformer analogue of
the chain-ladder recursion: sample every future cell on the next calendar
diagonal from the MDN, insert the samples as context, re-encode, continue.
Draws are therefore jointly coherent across cells (dependence via the shared
encoded context), and ultimates = anchor cumulative + premium x summed
sampled future increments. Draws are pooled over the ensemble members
(default 1000 total). `predict(segment=...)` slices the cached global
rollout - fit once, score every company.

Promotion raises **the target channel's flag only**. The sampled value is a
target increment; the feature channels at that cell are genuinely unobserved,
so their flags stay off and the network sees a future cell the way it was
trained to - target known, features absent. Raising the whole cell (the
pre-0.5.4 per-cell flag) handed the network the contract's padding zero as an
observed feature increment at every cell the rollout filled.

## Held-out scoring (milestone 6)

The entry mixes in `PooledMDNHeldout` (`gallery/nn/_heldout.py`), which holds
the held-out scoring for every NN entry in the gallery - this one, `mdn`,
`deeptriangle` and `resnet` share one implementation, not four copies of it.
The entry itself supplies only two methods: `_heldout_inputs` (assemble one
cohort's forward inputs) and `_forward_mixture` (call the network, return
`(log_pi, mu, sigma)`). Both are abstract, so an entry cannot forget one.

Because the fit is pooled across cohorts while `kernels.holdout` scores one
cohort at a time, capability is served **per cohort**:
`entry.at_cohort(segment)` returns a light scorer view whose
`log_lik_at`/`predict_at` are the unmodified base-class implementations over a
single-cohort adapter contract, so `index_into`'s cohort-identity and
training-overlap guards apply unchanged. The entry-level
`log_lik_at`/`predict_at` resolve the cohort from the cells' own segment
values and delegate.

- **Conditioning: everything the cohort held at as_of, per channel.**
  `_heldout_inputs` hands the network the contract's `x_obs` for that cohort,
  so each channel conditions on its own observed cells. Training additionally
  gates every channel at the drawn cutoff, so a feature observed past the
  cohort's as_of diagonal (ragged booking) reaches the network here in a
  configuration training never showed - see "Training" for why it is kept.
- **Draw scale: `incremental`.** A draw is one forward pass per ensemble
  member at cutoff = the cohort's as_of diagonal (the held-out diagonal sits
  at distance 1, the most-supervised relative-calendar position - no rollout),
  sampled from the MDN, un-standardized (`z * std0[d] + mean0[d]`) and scaled
  by premium: an incremental dollar amount. The base class anchors it onto
  the cell's training-diagonal predecessor to reach the cumulative triangle
  basis. Draws are split across members exactly as the rollout splits them;
  per-member torch seeds derive from `predict_at(seed=...)`.
- **Density measure: `loss_ratio`.** Per member, the mixture log density of
  the standardized ratio with the standardization Jacobian folded in
  (`logsumexp_K(log_pi + log N(z; mu, sigma)) - log std0[d]`), i.e. a density
  of the incremental loss ratio; the base class subtracts `log premium` to
  reach Lebesgue-on-amount (the increment/cumulative step has Jacobian 1).
  The **draw axis is the ensemble members** (>= 2 required), so `logmeanexp`
  over it is the ensemble-average predictive density. Normalization over the
  amount space is pinned by test (`densities.check_normalization`).
- **Pinned-dev asymmetry.** At a pinned dev (fewer than two training-context
  values reached the per-dev normalizer - in practice the deepest dev) there
  is no trained head. Draws keep rollout semantics: the sampled value is
  forced to the pooled dev mean, a point-mass column (legal for CRPS; a
  request where EVERY cell is pinned is refused). The density is REFUSED
  outright - an untrained head on a degenerate scale is not an honest
  predictive law. The entry is therefore **CRPS-scorable at cells where it is
  not ELPD-scorable**; a retro harness should map the pinned-dev refusal to a
  cohort-level `scoring_refused` absence on the density axis only.

`nn_transformer_ml` has none of this wiring yet: its multiline layout needs
its own per-(company, line) adapter design.

- **Display segments are narrowed away, and verified on the way.** `nn_data`
  keeps display-only columns (`company_name`) out of the cohort key so two
  spellings of one company cannot become two cohorts, which makes this fit's key
  narrower than the triangle's segment schema. `log_lik_at` / `predict_at`
  therefore re-key the supplied `HoldoutCells` onto the fit's own schema before
  `index_into` sees them - checking each dropped value against `cohorts()[i]`
  first, so a cell belonging to a different spelling is refused rather than
  quietly scored here. `index_into`'s schema equality is left exact, and the
  narrowing never escapes: the caller keeps handing the original wide cells to
  `CohortForecast`, so a shared board still has one segment schema. Before
  0.5.0 there was no working route at all - `log_lik_at` raised "unknown segment
  column 'company_name'" and `at_cohort(...)` failed one level deeper.

## Evaluate flow

Global fit / per-cohort predict inverts the meyers_ccl loop:

```python
entry = gallery.fit("nn_transformer", tri, as_of="1997-12-31")
# or with case reserves as an undifferenced level channel beside paid loss:
entry = gallery.fit(
    "nn_transformer",
    tri,
    loss_field="paid_loss",
    feature_fields=("case_reserve",),
    level_fields=("case_reserve",),
    as_of="1997-12-31",
)
pred = entry.predict(segment={"company_code": code, "line_of_business": line})
realized = entry.realized_ultimates(tri, segment={"company_code": code, "line_of_business": line})
pred.summary(observed=realized)  # same Meyers-style table as every entry
```

## Limitations

- Small-data regime is the central risk; mitigations: tiny network, dropout,
  weight decay, cutoff augmentation, eval_date early stopping, ensembling,
  no company embedding. First lever if validation NLL diverges: d_model 32.
- **Feature channels are not simulated during rollout, and the rollout now
  says so rather than faking it.** Only the target channel is sampled and fed
  back, so at each successive future diagonal the model conditions on one more
  observed target cell and on no new feature values at all: the deeper the
  rollout runs, the fewer of its inputs are features. That is a real
  limitation - a joint model that simulates every channel forward is a
  different entry - but it is now the honest version of one. Before, the
  promoted cell claimed observed features whose value was contract padding, so
  the deep-lag cells were conditioned on fabricated zeros rather than on
  nothing.
- `level_fields` changes what a channel MEANS, not how it is scored: a level
  channel is still divided by premium and standardized per (channel, dev), and
  channel 0 - the emergence being predicted - cannot be one.
- Origins with no observed cells get pure-extrapolation ultimates (anchor 0);
  origins without premium produce NaN ultimates.
- Total-ultimate calibration may still be too narrow (per-cell MDN +
  diagonal AR); the Meyers PIT harness is the arbiter - a failed KS is a
  documented finding, not a hidden one. `config.exposure_sigma` is the first
  lever aimed squarely at this, reshaping predictive width by company size.
