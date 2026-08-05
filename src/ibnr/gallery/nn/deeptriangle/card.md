# deeptriangle - Kuo's DeepTriangle: GRU encoder/decoder with a mixture density head

**Family:** nn (PyTorch; requires the `[nn]` extra at fit time - registration
does not)
**Lineage:** Kuo, *DeepTriangle: A Deep Learning Approach to Loss Reserving*
(original in Keras). Reimplemented in pytorch over this package's NN data
contract, with three disclosed adaptations: the point heads are replaced by
mixture density heads (a point estimator cannot enter the gallery - decision
4), the auxiliary claims-outstanding target is produced inside the entry
(OS = reported - paid, or a case-reserve level read as it stands; the contract
carries no OS field), and the paper's clean
encode-then-decode split is generalized to ragged conditioning boundaries via
per-step masked dispatch between the encoder and decoder GRU cells.

## Data

`kernels.nn_contract.nn_data`, exactly as `nn_transformer`: a *cohort* is one
company x line of business; the target is an **incremental loss ratio**
(incremental loss / origin premium), the cross-cohort normalizer. Default
channels follow Kuo's two tasks: `loss_field="paid_loss"` (the target) plus
`feature_fields=("reported_loss",)`, from which the auxiliary outstanding
increment is derived on the ratio scale as `reported - paid`. Per-(channel,
dev) standardization statistics come from training-context cells only via the
shared `gallery/nn/_scheme.py::norm_stats`, with the **pinned-dev rule
preserved**: devs with fewer than two context values (in practice the
deepest) have standardized value 0 by definition, and rollout draws there are
forced to 0, i.e. the pooled dev mean. The auxiliary channel gets its own
per-dev stats under the same rule.

**Conditioning is per channel.** The context mask handed to the network is the
contract's `x_obs` - per-channel usable values, whose channel 0 IS `obs_mask` -
so a channel's value is read only where that channel has a value of its own,
and each channel's per-dev statistics are estimated on its own cells. A feature
missing at a cell the target holds is masked rather than read as a zero
increment, and the auxiliary OS target is trained only where BOTH channels are
real (`x_obs[1] & x_obs[0]`) - or, when channel 1 is a level, where that one
channel is real (`x_obs[1]`), since a level needs neither a predecessor dev nor
a second channel to exist.

That fixes a distortion this card used to disclose. Before 0.5.4 the mask was
one flag per CELL, the target's, so a hole in reported at an observed paid cell
corrupted TWO auxiliary targets rather than one: increments are differenced
against the immediate predecessor dev, so a hole at dev d wipes out both the
dev-d and dev-d+1 reported increments, and the contract's padding zero then
stood in for each - the derived outstanding at both cells read as MINUS the
paid increment, an outstanding that shrinks by exactly what was just paid.
Those cells are now dropped from the auxiliary loss instead of fabricated. On the Schedule P
mart paid and reported are booked on the same cells, so the two masks coincide
there and the fix bites on ragged data - and on the rollout (below), where it
bites always.

**`fit(level_fields=...)`** names feature channels carried UNDIFFERENCED - an
eval-date snapshot such as `case_reserve`, whose difference is the case movement
while the informative quantity is the outstanding level. It declares the KIND of
a channel already named in `feature_fields` rather than adding one;
`kernels/nn_contract.py` owns the semantics and the refusals. The default
channel pair does not use it: reported loss genuinely accumulates, so
differencing it is right. A LEVEL at channel 1 switches the auxiliary head onto
that level itself - the **case-reserve head** below - because the difference
channel 1 minus channel 0 only means something between two increments, and a
level-minus-increment hybrid is neither the outstanding increment nor the
outstanding level. Which form runs is read off `field_kinds`; there is no config
knob for it, since a knob whose only legal value the contract determines is a
knob that cannot be turned.

## Network

Per origin, the dev sequence is processed by a **GRU encoder/decoder pair**
(hidden 64, shared learned initial state): per-step input =
`Linear([channel values * channel flags, channel flags])` - 2F wide, one flag
per channel - + dev-lag embedding + broadcast cohort conditioning (LOB
embedding, optional company embedding, normalized log premium). The
encoder/decoder dispatch reads CHANNEL 0's flag, since a step is a step of the
target: steps inside the target's conditioning context run the ENCODER cell on
the input as built; steps outside run the DECODER cell on it with the target's
value zeroed - position + conditioning dominating, so the state rolls forward
open-loop, which is Kuo's decoder emitting the remaining dev steps. The
masked per-step dispatch (rather than pack/pad split sequences) is what lets
one batch mix cutoffs and predecessor holes. Heads read the state AFTER each
step; only non-context cells are ever scored, so a head never reads a state
that consumed the cell's own value.

**Each origin is an INDEPENDENT sequence at inference, and this is the sharp
difference from `nn_transformer`.** The recurrence runs along the development
axis only: origins are folded into the batch, so nothing flows between them
inside a forward pass. Change origin 0's context values and every other
origin's prediction is bitwise unchanged. Origins still share information,
but only through the trained weights - the dev-lag embeddings, the GRU cells
and the heads were fit on all origins of all cohorts at once. That is
learned-at-training-time sharing, not conditioning at prediction time. The
transformer is the opposite: its self-attention spans the whole origin x dev
grid, so one cell's value moves every other cell's prediction in the same
forward pass. Read every phrase below of the form "conditioned on everything
the cohort had at as_of" with that in mind: for this entry it means each
origin is conditioned on its OWN observed development, and each origin's
prediction can be computed alone.

**No calendar input at all.** The transformer needed a *relative* calendar
embedding to avoid the v1/v2 absolute-embedding defect (untrained parameters
injected exactly at forecast cells). The GRU does not even need that:
relative position arises structurally from recurrence - the decoder knows how
far past the boundary it is by how many steps it has run since the flag
dropped - so absolute calendar position is unrepresentable rather than merely
avoided.

**MDN heads, not Kuo's point outputs.** Both tasks get a K=3 Gaussian mixture
head over the normalized incremental ratio (same rationale as the
transformer: lognormal heads fail on negative increments, quantile heads
cannot produce coherent sums). The mixture loss and sampler are **imported
from `gallery/nn/transformer/network.py`** (`mdn_nll`/`mdn_sample`) - one
implementation of the mixture math per package. A deep ensemble (5 seeds)
supplies epistemic spread on top of the mixture's aleatoric spread.

**Company embedding (`config.company_embedding`, ON by default).** Kuo's
actual design, and the deliberate departure from the transformer's
no-embedding choice: `contract["company_idx"]` feeds an 8-dim embedding in
the broadcast conditioning. ~600 companies x ~55 cells is a memorization
vector, so the flag exists precisely to run the ablation - `False` removes
the embedding table entirely (the parameter count changes; pinned by test).
Whether the embedding helps or hurts on the Schedule P backtest is an
empirical question the compare harness answers, not a claim this card makes.

**Auxiliary task (`config.aux_weight`, default 1.0).** Kuo trains paid and
claims outstanding jointly; here the second head is an MDN trained with plain
`mdn_nll` at weight `aux_weight` (a second MDN rather than Kuo's MSE, so
there is exactly one loss family in the entry; `aux_weight=0.0` is the
single-task ablation arm). It has **two forms, chosen by `field_kinds[1]`, not
by a knob**:

- **Outstanding increment** (channel 1 is an increment, the default pair): the
  target is `x[1] - x[0]` on the ratio scale - incremental reported minus
  incremental paid - trained where both channels are real.
- **Case-reserve head** (channel 1 is a level, e.g.
  `level_fields=("case_reserve",)`): the target is that level ratio itself,
  trained where that one channel is real. This is Kuo's own second task read
  literally, and the level is the harder and more informative thing to learn: a
  case reserve does not accumulate, it drains toward zero as payments replace it
  and jumps upward when new information arrives - a hospital bill reported
  months late - so its rundown is neither linear nor monotone.

Standardization is the same either way: per dev, from training-context cells,
under the target's pinning rule. Early stopping tracks the TARGET head's
validation NLL only, so model selection is not coupled to `aux_weight`. Rollout
and held-out scoring only ever consume the target head - the auxiliary head is a
training-time regularizer, in both forms.

## Training

Identical scheme to the transformer, via the shared machinery:

- **Pooled** over every cohort; a batch is 64 cohort-triangles.
- **Calendar-cutoff augmentation** (`gallery/nn/_training.py`): per cohort
  per epoch, draw a fake as_of cutoff; the encoder consumes cells on/before
  it, the loss scores observed training cells strictly after it. The decoder
  is trained the way it is used - conditioned on a sub-triangle, never on
  absolute calendar position - with one ragged-data edge: prediction gates
  features by observedness alone, so a feature booked past every target cell
  reaches the network at a distance training never showed (kept because the
  contemporaneous cell is genuinely informative; the transformer card's
  "Training" section states the same edge).
- **Validation by eval_date** (`gallery/nn/_scheme.py::splits`): the trailing
  observed calendar diagonal of the training window is excluded from all
  training contexts and targets; early stopping (patience 25) on its
  target-head NLL, best weights restored.
- Deep ensemble via `train_ensemble`: member seeds `seed + 1000 * member`.
- AdamW (lr 3e-4, weight decay 1e-2), gradient clip 1.0, max 400 epochs.

## Prediction

Autoregressive **diagonal-by-diagonal rollout**, mirroring the transformer's:
sample every future cell on the next calendar diagonal from the target head,
promote the samples to context on the TARGET channel only (nothing was sampled
for a feature, so its flag stays off), re-encode, continue - so the decoder only
ever runs one step past genuine-or-sampled context, the distance the
augmentation supervises most. Ultimates = anchor
cumulative + premium x summed sampled future increments; draws are pooled
over the ensemble members and cached per (n_draws, seed).
`predict(segment=...)` slices the cached global rollout.

## Held-out scoring (milestone 6 wiring)

Shared code, not a copy of the transformer's: the entry mixes in
`PooledMDNHeldout` (`gallery/nn/_heldout.py`), the one implementation every NN
entry uses. Because the fit is pooled while `kernels.holdout` scores one
cohort at a time, capability is served per cohort through
`entry.at_cohort(segment)` -> a `CohortHeldout` view whose
`log_lik_at`/`predict_at` are the unmodified base-class implementations over
a single-cohort adapter contract, so `index_into`'s cohort-identity and
training-overlap guards apply unchanged. The entry-level methods resolve the
cohort from the cells' own segment values and delegate.

DeepTriangle supplies the two per-entry hooks: `_heldout_inputs` (its
conditioning takes a company embedding index rather than a calendar cutoff, and
per-channel flags straight off `x_obs`, the same mask fit trains on) and
`_forward_mixture`, which is where the **auxiliary head is dropped** - the
network returns `(mixture, aux)` and only the mixture is a predictive density.
The auxiliary claims-outstanding task is a training-time regularizer and is
never consulted at scoring time.

- **Draw scale: `incremental`.** One forward pass per ensemble member
  conditioned on everything the cohort had at as_of - the held-out diagonal
  is the decoder's next step, so one decoder step suffices and no rollout is
  run. Sampled from the target head, un-standardized
  (`z * std0[d] + mean0[d]`) and scaled by premium: an incremental dollar
  amount, anchored onto the training-diagonal predecessor by the base class.
- **Density measure: `loss_ratio`.** Per member, the target head's mixture
  log density of the standardized ratio with the standardization Jacobian
  folded in (`logsumexp_K(log_pi + log N(z; mu, sigma)) - log std0[d]`); the
  base class subtracts `log premium` to reach Lebesgue-on-amount (the
  increment/cumulative step has Jacobian 1). The draw axis is the ensemble
  members (>= 2 required), so `logmeanexp` over it is the ensemble-average
  predictive density. Normalization over the amount space is pinned by test.
- **Pinned-dev asymmetry.** Draws keep rollout semantics at a pinned dev
  (point mass at the pooled dev mean; a request where EVERY cell is pinned is
  refused); the density is REFUSED outright, naming the pinned dev steps. The
  entry is CRPS-scorable at cells where it is not ELPD-scorable.

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

Global fit / per-cohort predict, same as the transformer:

```python
entry = gallery.fit("deeptriangle", tri, as_of="1997-12-31")
pred = entry.predict(segment={"company_code": code, "line_of_business": line})
realized = entry.realized_ultimates(tri, segment={"company_code": code, "line_of_business": line})
pred.summary(observed=realized)
```

## Limitations

- Small-data regime is the central risk; mitigations: tiny network, dropout,
  weight decay, cutoff augmentation, eval_date early stopping, ensembling.
  The company embedding is a known memorization vector kept because it is the
  paper's design - flag it off if validation NLL diverges.
- **Feature channels are still not simulated during rollout, and the
  disclosure is now sharper rather than smaller.** A promoted cell turns on the
  target channel's flag alone, so a future cell no longer presents the
  contract's padding zero as an observed reported increment - which it did at
  every forecast cell before 0.5.4, and "stale" was the wrong word for it: the
  value was fabricated, not old. What replaces the fabrication is an honest
  absence, so the rollout conditions on FEWER observed features the deeper it
  goes, and past as_of the reported channel is unobserved everywhere.
  Simulating the features forward is a different model, not a patch to this
  one.
- Per-line draws are independent: no cross-line dependence, no diversified
  company total (see `nn_ml_*`).
- Origins with no observed cells get pure-extrapolation ultimates (anchor 0);
  origins without premium produce NaN ultimates.
- The auxiliary outstanding-increment target exists only where BOTH channels are
  observed, so a cohort whose reported development is ragged trains that head on
  fewer cells than its paid head; see "Data". The case-reserve form needs one
  channel and so loses fewer cells, but it is scored on nothing else: no test
  here says a level auxiliary task predicts paid loss better than the derived
  increment does, and the compare harness is what would.
