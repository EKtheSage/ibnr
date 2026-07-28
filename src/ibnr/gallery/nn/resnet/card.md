# resnet - residual convolutional triangle network with a mixture density head

**Family:** nn (PyTorch; requires the `[nn]` extra at fit time - registration
does not)
**Lineage:** the third encoder-body ablation in the NN family. The transformer
(`nn_transformer`) reads the triangle with global self-attention; the per-cell
MLP (`mdn`) reads each cell with no spatial structure at all; this entry reads
it with LOCAL 2-D structure - residual 3x3 convolutions over the
(origin x dev) grid, the ResNet recipe (He et al. 2016) at triangle scale.
The actuarial prior it encodes: development is dominated by a cell's
neighborhood - the same origin's adjacent devs, adjacent origins at the same
dev, the calendar diagonal through the cell - and longer-range structure
should have to propagate through depth rather than being free at layer one.

Everything except the encoder body is deliberately identical to the
transformer - contract, training scheme, MDN head (imported from the
transformer's network module, not copied), rollout, held-out wiring - so a
leaderboard difference between the three entries measures the body, not an
accident of plumbing.

## Data

`kernels.nn_contract.nn_data`, unchanged from the transformer: a *cohort* is
one company x line of business; targets are **incremental loss ratios**
(incremental loss / origin premium); extra channels via `feature_fields`.
Per-(channel, dev) standardization statistics are computed from
training-context cells only, with the **pinned** rule for devs having fewer
than two context values (standardized value defined as 0, mean from all
observed cells at that dev, std 1; rollout draws forced to the pooled dev
mean). The pinning is the shared `gallery/nn/_scheme.py` implementation - the
v2 fix - not a re-derivation.

## Network

Input is an image-like stack of per-cell channels over the (n_w, n_d) grid:

- the F standardized value channels, **zeroed outside the context mask**;
- the **context mask itself as an explicit input channel**. The contract's
  zeros are padding, never data (every consumer gates on the mask), and for
  a conv body the mask-as-channel is how the network learns that: it is the
  only signal separating "zero because unobserved/future" from "zero because
  a zero increment was observed";
- a **relative calendar channel**: distance past the conditioning cutoff,
  clamped to [0, n_d] exactly like the transformer's `dist_emb`, scaled by
  1/n_d. Relative, never absolute - forecast diagonals lie past the training
  window, where an absolute calendar encoding never received a gradient (the
  documented v1/v2 transformer defect);
- broadcast cohort conditioning: LOB embedding (dim 8) + normalized log
  premium, constant over the grid. No company embedding, for the same
  memorization reason as the transformer.

Body: a 3x3 conv stem to 32 channels, 2-D channel dropout (0.15), then 3
pre-activation residual blocks (GroupNorm -> GELU -> 3x3 conv, twice, plus
the identity skip). The default network has **59,753 parameters** at the
reference shape (`n_lob=4`, `n_features=1`); unlike an attention body the
count does not depend on n_w/n_d, since a conv has no positional embedding
tables. The 7 convs of the default stack (stem + 3 blocks x 2) each add 1
cell of radius, so a prediction sees a **radius-7** patch - most of a
Schedule P 10x10 triangle, but reached through locality rather than granted
by attention. Head: a final GroupNorm + GELU and a 1x1 conv reading the same
K=3 Gaussian MDN per cell as the transformer; `mdn_nll` and `mdn_sample` are
imported from `gallery/nn/transformer/network.py` so the three bodies share
one head implementation.

**Why 32 channels and not 64.** Width was cut from the first draft's 64
after measuring: at 64 the trunk carries 230,057 parameters (each residual
block holds two 64x64x3x3 convs, ~37k each), roughly 3x the 70,505 the
transformer's default network carries at the same shape. That made the
"deliberately tiny" claim above false and turned a body ablation into a
capacity comparison. Depth was kept at 3 blocks because depth IS the
architectural claim here - it sets the receptive field - and only width,
which is pure capacity, was reduced. The disclosed count is pinned by
`tests/test_resnet.py::test_disclosed_parameter_count`, which reads the
number out of this card, so the two cannot drift apart again.

**Why GroupNorm, never BatchNorm.** Under calendar-cutoff augmentation every
cohort in a batch is conditioned at its OWN drawn cutoff. BatchNorm's batch
statistics would mix activations across those different conditioning states -
information flowing between augmentation tasks within a batch at train time -
and its eval-time running stats would be an average over cutoffs that matches
no actual conditioning. GroupNorm normalizes within a single sample, so
nothing crosses the batch axis. (BatchNorm is forbidden in this entry; a
variant that wants it must first answer the leakage argument above.)

**Leakage and the receptive field.** A convolution sees the whole grid -
unlike attention there is no mask argument to hide a cell architecturally.
What protects against conditioning on the future is the input construction:
value channels are multiplied by the context flag before the first
convolution, so a beyond-cutoff cell contributes only its (mask=0, distance)
position, never its value. This is pinned by the receptive-field no-leak
test in `tests/test_resnet.py`: poison a beyond-cutoff cell's value and the
network's output must be bit-identical everywhere. That test is the
load-bearing one for a conv body.

## Training

Identical to the transformer, via the shared `gallery/nn/_training.py` loop:
pooled over every cohort (batch = 64 cohort-triangles), calendar-cutoff
augmentation (per cohort per epoch a fake as_of drawn in
`[min_cutoff, val_cutoff - 1]`), validation by eval_date (the trailing
observed calendar diagonal of the training window, early stopping on its NLL
with patience 25, best weights restored), AdamW (lr 3e-4, weight decay 1e-2),
gradient clip 1.0, max 400 epochs, deep ensemble of 5 members seeded
`seed + 1000 * member`. All defaults are hand-chosen for the Schedule P
regime and disclosed here; systematic HPO is deferred to `kernels/tuning.py`.

## Prediction

Autoregressive **diagonal-by-diagonal rollout**, the same scheme as the
transformer: sample every future cell on the next calendar diagonal from the
MDN, insert the samples as context (the context flag advances with them),
re-encode, continue; ultimates = anchor cumulative + premium x summed sampled
future increments; draws pooled over the ensemble members (default 1000).
`predict(segment=...)` slices the cached global rollout - fit once, score
every company. Per-line draws are INDEPENDENT: this entry models no
cross-line dependence and emits no diversified total.

## Held-out scoring (milestone 6 wiring)

Shared code, not a copy: the entry mixes in `PooledMDNHeldout`
(`gallery/nn/_heldout.py`), and capability is served **per cohort** via
`entry.at_cohort(segment)` -> a scorer view over the single-cohort adapter
contract, so `index_into`'s cohort-identity and training-overlap guards apply
unchanged. `resnet` adds only the two abstract hooks, `_heldout_inputs` and
`_forward_mixture`; both match the transformer's, since this entry differs
from it in the encoder alone.

- **Draw scale: `incremental`.** One forward pass per ensemble member at
  cutoff = the cohort's as_of diagonal, `mdn_sample` at the requested cells,
  un-standardized (`z * std0[d] + mean0[d]`) and scaled by premium; the base
  class anchors onto the training-diagonal predecessor.
- **Density measure: `loss_ratio`.** Per member, the mixture log density of
  the standardized ratio with the standardization Jacobian folded in
  (`logsumexp_K(log_pi + log N(z; mu, sigma)) - log std0[d]`); the base class
  subtracts `log premium` to reach Lebesgue-on-amount. The draw axis is the
  ensemble members (>= 2 required). Normalization over the amount space is
  pinned by test (`densities.check_normalization`).
- **Pinned-dev asymmetry.** Draws at a pinned dev keep rollout semantics (a
  point mass at the pooled dev mean; a request where EVERY cell is pinned is
  refused); the density there is REFUSED outright - an untrained head on a
  degenerate scale is not an honest predictive law. The entry is
  CRPS-scorable at cells where it is not ELPD-scorable.

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

Global fit / per-cohort predict, identical to the transformer:

```python
entry = gallery.fit("resnet", tri, as_of="1997-12-31")
pred = entry.predict(segment={"company_code": code, "line_of_business": line})
realized = entry.realized_ultimates(tri, segment={"company_code": code, "line_of_business": line})
pred.summary(observed=realized)  # same Meyers-style table as every entry
```

## Limitations

- Small-data regime is the central risk; mitigations: tiny network, channel
  dropout, weight decay, cutoff augmentation, eval_date early stopping,
  ensembling, no company embedding. First lever if validation NLL diverges:
  channels 24 or n_blocks 2.
- The receptive field is finite (grows with depth): structure farther than
  7 cells needs more blocks to influence a prediction. That is the point of
  the ablation - if the transformer beats this entry, global attention earns
  its keep; if not, locality was enough.
- Feature channels are not simulated during rollout - future cells feed back
  the target channel only.
- Origins with no observed cells get pure-extrapolation ultimates (anchor 0);
  origins without premium produce NaN ultimates.
- Per-line draws are independent; only the `nn_ml_*` entries model cross-line
  dependence.
