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
and reported. Per-(channel, dev) standardization statistics are computed from
training-context cells only (never the validation diagonals). Devs with
fewer than two context values - in practice the deepest dev, observed only
on the held-out diagonal - are **pinned**: their standardized value is 0 by
definition (mean from all observed cells at that dev, std 1), and rollout
draws there are forced to 0, i.e. the pooled dev mean. Pinning replaced the
original inherit-earlier-dev-stats scheme, which denormalized tail cells at
mid-development magnitudes and systematically overstated fast-decaying
tails (worst for PPA in the first backtest).

## Network

Token = `Linear([channel values * flag, flag])` + origin embedding + dev
embedding + **relative calendar embedding** (distance past the conditioning
cutoff, clamped to [0, n_d]) + broadcast conditioning (LOB embedding +
normalized log premium). Encoder: 2 pre-LN transformer layers, d_model 64,
4 heads, FFN 128, dropout 0.15, GELU (~120k parameters). Head: mixture
density network, K=3 Gaussians per cell on the normalized incremental
loss-ratio scale.

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
  main small-data multiplier.
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

## Held-out scoring (milestone 6)

The entry subclasses both held-out mixins; because the fit is pooled across
cohorts while `kernels.holdout` scores one cohort at a time, capability is
served **per cohort**: `entry.at_cohort(segment)` returns a light scorer view
whose `log_lik_at`/`predict_at` are the unmodified base-class implementations
over a single-cohort adapter contract (`gallery/nn/_heldout.py`), so
`index_into`'s cohort-identity and training-overlap guards apply unchanged.
The entry-level `log_lik_at`/`predict_at` resolve the cohort from the cells'
own segment values and delegate.

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

## Evaluate flow

Global fit / per-cohort predict inverts the meyers_ccl loop:

```python
entry = gallery.fit("nn_transformer", tri, as_of="1997-12-31")
pred = entry.predict(segment={"company_code": code, "line_of_business": line})
realized = entry.realized_ultimates(tri, segment={...})
pred.summary(observed=realized)  # same Meyers-style table as every entry
```

## Limitations

- Small-data regime is the central risk; mitigations: tiny network, dropout,
  weight decay, cutoff augmentation, eval_date early stopping, ensembling,
  no company embedding. First lever if validation NLL diverges: d_model 32.
- Feature channels are not simulated during rollout - future cells feed back
  the target channel only.
- Origins with no observed cells get pure-extrapolation ultimates (anchor 0);
  origins without premium produce NaN ultimates.
- Total-ultimate calibration may still be too narrow (per-cell MDN +
  diagonal AR); the Meyers PIT harness is the arbiter - a failed KS is a
  documented finding, not a hidden one. `config.exposure_sigma` is the first
  lever aimed squarely at this, reshaping predictive width by company size.
