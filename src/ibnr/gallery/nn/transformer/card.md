# nn_transformer — masked-cell triangle transformer with a mixture density head

**Family:** nn (PyTorch; requires the `[nn]` extra at fit time — registration
does not)
**Lineage:** Kuo's *DeepTriangle* (pooled NN reserving across many
company x LOB triangles) reimagined as an attention model: instead of GRU
sequences per origin, every (origin, dev) cell of the triangle is a token and
a small encoder attends over the whole grid — development, accident-year, and
calendar structure are all learned positional embeddings.

## Data

`kernels.nn_contract.nn_data`: a *cohort* is one company x line of business.
Targets are **incremental loss ratios** (incremental loss / origin premium) —
the cross-cohort normalizer. Inputs may carry extra channels
(`feature_fields`, e.g. paid alongside reported). Increments require an
immediate-predecessor observation; the anchor for ultimates is each origin's
latest observed *cumulative* value. Cohorts with unusable premium are dropped
and reported. Per-(channel, dev) standardization statistics are computed from
training-context cells only (never the validation diagonals). Devs with
fewer than two context values — in practice the deepest dev, observed only
on the held-out diagonal — are **pinned**: their standardized value is 0 by
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
calendar embedding never received a gradient — untrained parameters injected
exactly at prediction cells. Distance-past-cutoff is supervised directly by
the cutoff augmentation, and because the rollout re-encodes after each
sampled diagonal, inference only ever consumes distance-1 predictions — the
most supervised case. The trade-off, disclosed: absolute calendar-year
(inflation) effects are no longer explicitly encoded and must be inferred
from context values.

**Why an MDN head:**
- lognormal heads fail outright — net-of-bulk incremental losses go negative
  (case releases; the milestone-2 lesson);
- quantile heads cannot produce coherent *sums* — ultimates add many future
  cells, and marginal quantiles don't add;
- the MDN gives a proper sampleable per-cell density; a **deep ensemble
  (5 seeds)** supplies epistemic spread on top of the mixture's aleatoric
  spread. Both mechanisms of the gallery's distributional-head requirement.

**No company embedding in v1** — ~600 companies with ~55 cells each is a
memorization vector, not a feature. Company identity enters through its own
observed cells, LOB, and size (log premium).

## Training

- **Pooled** over every cohort; a batch is 64 cohort-triangles.
- **Calendar-cutoff augmentation:** per cohort per epoch, draw a fake as_of
  cutoff uniformly in `[min_cutoff, val_cutoff - 1]`; condition on the
  sub-triangle, score NLL on later observed training cells. Every triangle
  yields ~5 distinct "predict the future diagonals" tasks per epoch — the
  main small-data multiplier.
- **Validation by eval_date:** the trailing `val_diagonals=1` observed
  calendar diagonal of the *training window* is excluded from all training
  contexts and targets; early stopping (patience 25) on its NLL, best
  weights restored. Test-period diagonals never enter fit at all.
- AdamW (lr 3e-4, weight decay 1e-2), gradient clip 1.0, max 400 epochs.
- All defaults are hand-chosen for the Schedule P regime and disclosed here;
  systematic HPO is deferred to `kernels/tuning.py`.

## Prediction

Autoregressive **diagonal-by-diagonal rollout** — the transformer analogue of
the chain-ladder recursion: sample every future cell on the next calendar
diagonal from the MDN, insert the samples as context, re-encode, continue.
Draws are therefore jointly coherent across cells (dependence via the shared
encoded context), and ultimates = anchor cumulative + premium x summed
sampled future increments. Draws are pooled over the ensemble members
(default 1000 total). `predict(segment=...)` slices the cached global
rollout — fit once, score every company.

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
- Feature channels are not simulated during rollout — future cells feed back
  the target channel only.
- Origins with no observed cells get pure-extrapolation ultimates (anchor 0);
  origins without premium produce NaN ultimates.
- Total-ultimate calibration may still be too narrow (per-cell MDN +
  diagonal AR); the Meyers PIT harness is the arbiter — a failed KS is a
  documented finding, not a hidden one.
