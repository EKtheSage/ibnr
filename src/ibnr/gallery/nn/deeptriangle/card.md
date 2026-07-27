# deeptriangle - Kuo's DeepTriangle: GRU encoder/decoder with a mixture density head

**Family:** nn (PyTorch; requires the `[nn]` extra at fit time - registration
does not)
**Lineage:** Kuo, *DeepTriangle: A Deep Learning Approach to Loss Reserving*
(original in Keras). Reimplemented in pytorch over this package's NN data
contract, with three disclosed adaptations: the point heads are replaced by
mixture density heads (a point estimator cannot enter the gallery - decision
4), the auxiliary claims-outstanding target is derived inside the entry
(OS = reported - paid; the contract does not carry OS), and the paper's clean
encode-then-decode split is generalized to ragged conditioning boundaries via
per-step masked dispatch between the encoder and decoder GRU cells.

## Data

`kernels.nn_contract.nn_data`, exactly as `nn_transformer`: a *cohort* is one
company x line of business; targets are **incremental loss ratios**
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

One derived-target caveat, disclosed: the contract masks usable increments
for the TARGET channel only, so the auxiliary OS increment is trained on
`obs_mask` cells and is distorted wherever the reported channel is padding at
an observed paid cell. On the Schedule P mart paid and reported are booked on
the same cells, so the masks coincide in practice.

## Network

Per origin, the dev sequence is processed by a **GRU encoder/decoder pair**
(hidden 64, shared learned initial state): per-step input =
`Linear([channel values * flag, flag])` + dev-lag embedding + broadcast
cohort conditioning (LOB embedding, optional company embedding, normalized
log premium). Steps inside the conditioning context run the ENCODER cell on
the true values; steps outside run the DECODER cell on the same input with
values zeroed - position + conditioning only, so the state rolls forward
open-loop, which is Kuo's decoder emitting the remaining dev steps. The
masked per-step dispatch (rather than pack/pad split sequences) is what lets
one batch mix cutoffs and predecessor holes. Heads read the state AFTER each
step; only non-context cells are ever scored, so a head never reads a state
that consumed the cell's own value.

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
claims outstanding jointly; here the OS head is a second MDN over the
standardized incremental outstanding ratio, trained with plain
`mdn_nll` at weight `aux_weight` (a second MDN rather than Kuo's MSE, so
there is exactly one loss family in the entry; `aux_weight=0.0` is the
single-task ablation arm). Early stopping tracks the TARGET head's validation
NLL only, so model selection is not coupled to `aux_weight`. Rollout and
held-out scoring only ever consume the target head - the auxiliary head is a
training-time regularizer.

## Training

Identical scheme to the transformer, via the shared machinery:

- **Pooled** over every cohort; a batch is 64 cohort-triangles.
- **Calendar-cutoff augmentation** (`gallery/nn/_training.py`): per cohort
  per epoch, draw a fake as_of cutoff; the encoder consumes cells on/before
  it, the loss scores observed training cells strictly after it. The decoder
  is therefore trained exactly the way it is used: conditioned only on data
  at or before a cutoff, never on absolute calendar position.
- **Validation by eval_date** (`gallery/nn/_scheme.py::splits`): the trailing
  observed calendar diagonal of the training window is excluded from all
  training contexts and targets; early stopping (patience 25) on its
  target-head NLL, best weights restored.
- Deep ensemble via `train_ensemble`: member seeds `seed + 1000 * member`.
- AdamW (lr 3e-4, weight decay 1e-2), gradient clip 1.0, max 400 epochs.

## Prediction

Autoregressive **diagonal-by-diagonal rollout**, mirroring the transformer's:
sample every future cell on the next calendar diagonal from the target head,
promote the samples to context (they become encoder steps), re-encode,
continue - so the decoder only ever runs one step past genuine-or-sampled
context, the distance the augmentation supervises most. Ultimates = anchor
cumulative + premium x summed sampled future increments; draws are pooled
over the ensemble members and cached per (n_draws, seed).
`predict(segment=...)` slices the cached global rollout.

## Held-out scoring (milestone 6 wiring)

Exactly the transformer's pattern: the entry subclasses both mixins, and
because the fit is pooled while `kernels.holdout` scores one cohort at a
time, capability is served per cohort through `entry.at_cohort(segment)` -> a
`CohortHeldout` view (`gallery/nn/_heldout.py`) whose
`log_lik_at`/`predict_at` are the unmodified base-class implementations over
a single-cohort adapter contract, so `index_into`'s cohort-identity and
training-overlap guards apply unchanged. The entry-level methods resolve the
cohort from the cells' own segment values and delegate.

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
- Feature channels are not simulated during rollout - future cells feed back
  the target channel only, so deep-lag predictions condition on an
  increasingly stale reported channel (inherited from the transformer, same
  disclosure).
- Per-line draws are independent: no cross-line dependence, no diversified
  company total (see `nn_ml_*`).
- Origins with no observed cells get pure-extrapolation ultimates (anchor 0);
  origins without premium produce NaN ultimates.
- The auxiliary OS target inherits the target channel's mask; see "Data".
