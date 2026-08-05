# mdn - per-cell mixture density network (the no-attention ablation)

**Family:** nn (PyTorch; requires the `[nn]` extra at fit time - registration
does not)
**Lineage:** Bishop's mixture density network, applied per triangle cell. Not
a new modeling idea: this entry exists as the architecture ablation of
`nn_transformer` - same data contract, same masked-cell objective, same
training scheme, same MDN head, same rollout and held-out wiring, and NO
sequence model, NO attention. A leaderboard gap between the two entries is
attributable to the encoder body, because the encoder body is the only thing
that differs.

## Why this entry exists

The transformer's results bundle two claims: (a) the rig works (incremental
loss ratios, pinned per-dev standardization, calendar-cutoff augmentation,
eval_date validation, deep ensembling), and (b) attention over the triangle
grid adds predictive value on top of it. `mdn` isolates (b) by holding (a)
fixed. Everything that is not the encoder is imported from the shared
modules rather than re-implemented, so the two entries cannot drift apart:
`gallery/nn/_scheme.py` (splits + pinned norm stats), `gallery/nn/_training.py`
(deep-ensemble loop, member seeds `seed + 1000 * member`),
`gallery/nn/_heldout.py` (the per-cohort adapter AND the whole held-out
scoring implementation, via `PooledMDNHeldout`), and the transformer's own
`mdn_nll`/`mdn_sample` (head loss and sampler, imported - not copied).

## Data

`kernels.nn_contract.nn_data`, identical to the transformer: a *cohort* is
one company x line of business; targets are **incremental loss ratios**
(incremental loss / origin premium); per-(channel, dev) standardization from
training-context cells only, with devs holding fewer than two context values
**pinned** (standardized value 0, mean from all observed cells at that dev,
std 1; rollout draws there forced to the pooled dev mean). The pinning is
the v2 fix recorded in CLAUDE.md and is shared source, not a copy.

Observedness is **per channel** (`x_obs`), not per cell. A triangle can report
the target at a cell and not a feature there, or the other way round, so each
channel is conditioned on its own usable cells: a missing feature is masked
out rather than read as a reported zero, and each channel's standardization
statistics come from the cells where that channel has a value. Channel 0's
mask is the target's and is exactly `obs_mask`, so a fit with no feature
fields is the same fit it always was.

`fit(level_fields=...)` names feature channels carried **undifferenced** - for
fields that are eval-date snapshots rather than amounts that accumulate, such
as `case_reserve`, whose difference is the case movement while the informative
quantity is the outstanding level. It must be a subset of `feature_fields`
(it declares a channel's kind, it does not add a channel) and the target
cannot be one. See `kernels.nn_contract.nn_data` for the semantics; the entry
only threads the argument through.

## Network

Per target cell, an MLP over a fixed-size feature vector - no tokens, no
attention. The vector concatenates:

- **cohort summary**: the masked mean of each (channel, dev)'s standardized
  context values, pooled over origins - each channel averaged over its own
  context cells - plus each dev's context-cell fraction (channel 0's context
  count / n_w). This is the entry's whole cross-origin view: where the
  transformer learns which cells to look at, the MLP gets one fixed per-dev
  average.
- **the target origin's own masked row**: its standardized context values
  across devs (zeroed off-context) plus per-dev context flags - the
  chain-ladder-natural conditioning on the origin's own history. The flag
  vector is channel 0's: it is the structural "does this cell exist" signal,
  and that is the target's question.
- **origin embedding + dev embedding + relative calendar embedding**
  (distance past the conditioning cutoff, clamped to [0, n_d]). The calendar
  encoding is RELATIVE for the same reason the transformer's is: forecast
  diagonals lie past the training window, where an absolute calendar
  embedding never received a gradient (the documented v1/v2 defect).
- **LOB embedding + normalized log premium** - cohort conditioning.

Non-context values are zeroed *before* any summary is taken, so a cell past
the cutoff cannot influence any prediction - its own included. The gate is per
channel: a value never enters a summary without its own channel's flag. With
one channel the two forms are the same function, which is why this change left
every single-channel result untouched. Body: 2
hidden layers of 128, GELU, dropout 0.1 - **26,353 parameters** on an 8x8 grid,
comfortably under the transformer's **70,121 parameters** at the same shape.
Both figures are pinned by `tests/test_nn_parameter_counts.py`; an earlier
revision of this card put the transformer at ~120k, a figure `transformer/card.md`
had already corrected and which survived here only because nothing checked it.
Head: the same K=3 Gaussian MDN per cell on the
normalized incremental loss-ratio scale, sigma = softplus + 1e-3.

## Training

Identical to the transformer by construction (shared `train_ensemble`):
pooled over cohorts, calendar-cutoff augmentation in
`[min_cutoff, val_cutoff)`, validation on the trailing eval_date diagonal,
AdamW (lr 3e-4, weight decay 1e-2), gradient clip 1.0, early stopping
(patience 25), deep ensemble of 5 members seeded `seed + 1000 * member`.
The optimization defaults are deliberately the transformer's - tuning one
arm and not the other would turn the architecture ablation into a tuning
comparison.

## Prediction

Autoregressive **diagonal-by-diagonal rollout**, mirroring the transformer's
`_rollout` structure exactly: sample every future cell on the next calendar
diagonal from the MDN, promote the samples to context, recompute the context
summary, continue. Promotion sets **channel 0's flag only** - the sampled
target value now exists, next year's feature values do not - so the summary
never counts a feature the rollout invented. The dependence between a cohort's
cells flows through the shared summary rather than through attention - a
strictly cruder channel, which is part of what the ablation measures.
Ultimates = anchor cumulative + premium x summed sampled future increments;
draws pooled over the ensemble.

## Held-out scoring (milestones 6/7)

Shared code, via `PooledMDNHeldout` (`gallery/nn/_heldout.py`):
`at_cohort(segment)` returns a scorer view whose `log_lik_at`/`predict_at` are
the unmodified base-class implementations, so `index_into`'s cohort-identity
and training-overlap guards apply unchanged. `mdn` adds only the two abstract
hooks, `_heldout_inputs` and `_forward_mixture`, and both are identical in
substance to the transformer's - this entry differs from it in the encoder,
nowhere else.

- **Draw scale: `incremental`.** One forward pass per ensemble member at
  cutoff = the cohort's as_of diagonal, `mdn_sample` at the requested cells,
  un-standardized (`z * std0[d] + mean0[d]`) and scaled by premium; the base
  class anchors onto each cell's training-diagonal predecessor.
- **Density measure: `loss_ratio`.** Per member,
  `logsumexp_K(log_pi + log N(z; mu, sigma)) - log std0[d]` (the
  standardization Jacobian folded in), then the base class subtracts
  `log premium` to reach Lebesgue-on-amount. The draw axis is the ensemble
  members (>= 2 required); normalization over the amount space is pinned by
  test (`densities.check_normalization`).
- **Pinned-dev asymmetry.** Draws at a pinned dev are the point mass at the
  pooled dev mean (legal for CRPS; an all-pinned request is refused); the
  density there is REFUSED outright. CRPS-scorable where not ELPD-scorable,
  exactly like the transformer.

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

```python
entry = gallery.fit("mdn", tri, as_of="1997-12-31")
pred = entry.predict(segment={"company_code": code, "line_of_business": line})
realized = entry.realized_ultimates(tri, segment={"company_code": code})
pred.summary(observed=realized)  # same Meyers-style table as every entry
```

## Limitations

- The cohort summary is a per-dev mean: cross-origin structure beyond "the
  average development at this dev" is invisible, and the model cannot weight
  a similar origin above a dissimilar one. That is the point - it is the
  capability being ablated - but it is a real predictive handicap.
- Like the single-line transformer, per-line draws are independent: no
  cross-line dependence, no diversified company total.
- **Feature channels are not simulated during rollout**, and the rollout is
  honest about it rather than papering over it: a promoted cell carries the
  sampled target and no feature flag, so the deeper the rollout goes, the
  fewer observed feature values the cohort summary is built from. A fit with
  feature channels therefore conditions on progressively less at deep lags -
  a real limitation, and one whose cost is now visible in the summary's
  per-channel counts instead of hidden inside a fabricated zero. Simulating
  the features forward needs a joint head over all channels, which is a
  different entry.
- All the transformer's small-data caveats apply; the mitigations (tiny
  network, dropout, weight decay, augmentation, eval_date early stopping,
  ensembling, no company embedding) are inherited unchanged.
