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

## Network

Per target cell, an MLP over a fixed-size feature vector - no tokens, no
attention. The vector concatenates:

- **cohort summary**: the masked mean of each (channel, dev)'s standardized
  context values, pooled over origins, plus each dev's context-cell fraction
  (context count / n_w). This is the entry's whole cross-origin view: where
  the transformer learns which cells to look at, the MLP gets one fixed
  per-dev average.
- **the target origin's own masked row**: its standardized context values
  across devs (zeroed off-context) plus per-dev context flags - the
  chain-ladder-natural conditioning on the origin's own history.
- **origin embedding + dev embedding + relative calendar embedding**
  (distance past the conditioning cutoff, clamped to [0, n_d]). The calendar
  encoding is RELATIVE for the same reason the transformer's is: forecast
  diagonals lie past the training window, where an absolute calendar
  embedding never received a gradient (the documented v1/v2 defect).
- **LOB embedding + normalized log premium** - cohort conditioning.

Non-context values are zeroed *before* any summary is taken, so a cell past
the cutoff cannot influence any prediction - its own included. Body: 2
hidden layers of 128, GELU, dropout 0.1 (~30k parameters, even smaller than
the transformer's ~120k). Head: the same K=3 Gaussian MDN per cell on the
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
summary, continue. The dependence between a cohort's cells flows through the
shared summary rather than through attention - a strictly cruder channel,
which is part of what the ablation measures. Ultimates = anchor cumulative +
premium x summed sampled future increments; draws pooled over the ensemble.

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
- Feature channels are not simulated during rollout - future cells feed back
  the target channel only.
- All the transformer's small-data caveats apply; the mitigations (tiny
  network, dropout, weight decay, augmentation, eval_date early stopping,
  ensembling, no company embedding) are inherited unchanged.
