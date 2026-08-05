# nn_paid_case - joint paid + case-reserve NN with a simulated case state

**Family:** nn (PyTorch; requires the `[nn]` extra at fit time - registration
does not)
**Lineage:** the pooled-NN reserving line of this gallery (`nn_transformer`,
`deeptriangle`) crossed with the compartmental entry's structural idea - that
outstanding claims DRAIN into paid - but with no functional form imposed on the
rundown. Where `deeptriangle`'s case-reserve head predicts the case level as a
training-time auxiliary task nothing downstream reads, this entry predicts the
paid increment and the case MOVEMENT jointly and consumes both.

## Why this entry exists: drain and shock

A case reserve is not a covariate that sits still. It is a **state with
dynamics**, and it moves two ways at once:

- it **drains** toward zero as payments replace it - the claim department's
  estimate is converted into cash, so paid up and case down is the normal
  quarter;
- it **jumps** upward when new information arrives - a hospital bill reported
  months after the accident, a reserve strengthening, a suit filed.

So the rundown is neither linear nor monotone, and a model that reads today's
case reserve as a frozen input cannot carry it into a multi-year projection.
Every 0.5.4 NN entry discloses exactly that limitation ("feature channels are
not simulated during rollout"): the deeper the projection runs, the fewer of its
inputs are features, until a five-year forecast is conditioned on nothing but
its own simulated paid history.

This entry removes the limitation rather than disclosing it. Per cell it
predicts **(paid increment, case movement) jointly**, samples both from the same
mixture component and the same normal draw, writes the sampled paid increment
into channel 0, integrates the sampled movement into the case LEVEL and writes
that into channel 1, promotes both channels' flags and re-encodes. The paid
projection therefore conditions on a live case position at every step, and the
learned payment-drain correlation - paid up, case down - is what keeps the two
simulated paths consistent with each other.

## Data

`kernels.nn_contract.nn_data`, with the channels fixed by the entry rather than
chosen by the caller:

- **channel 0** = incremental **paid** loss ratio (paid increment / origin
  premium) - the emergence being predicted, and the column the leaderboard
  scores;
- **channel 1** = the **case reserve LEVEL** ratio, carried UNDIFFERENCED
  (`level_fields=(case_field,)`), because a case reserve is an eval-date
  snapshot rather than an amount that accumulates. This is the state the network
  conditions on.

`fit` takes `paid_field` / `case_field` / `premium_field` and deliberately no
`feature_fields` or `level_fields`: the two channels ARE the design, and a
caller who wants a free choice of channels wants `nn_transformer` or
`deeptriangle`. A triangle that does not carry the case field is refused by
name, by `nn_data`, rather than fitted with one all-masked channel.

**The MOVEMENT target is derived in-entry from channel 1** (`model.case_movement`):

```python
move[d] = level[d] - level[d - 1]  # usable where BOTH cells are present
move[1] = level[1]  # dev 1: the case position was zero before the year opened
```

Input is the level (the state), target is the movement (the dynamics), both from
one stored channel - no duplicated field, no second contract function. The
movement's observedness is its OWN: a hole in the case field at dev `d` costs
the movement target at `d` and at `d + 1`, while the paid increment at both
cells is perfectly observable. That asymmetry is why the loss is
mixed-observedness (below) rather than a joint density over the intersection.

Standardization follows the family: per (channel, dev) statistics from that
channel's own training-context cells, never the validation diagonals, with the
**pinned-dev rule** (fewer than two context values -> standardized value 0 by
definition, mean from all observed cells at that dev, std 1; sampled draws there
are forced to the pooled dev mean). The movement target gets its **own** per-dev
statistics under the same rule, stored beside the input stats. It has to: a
level decays across development while a movement is centred near zero and
changes sign, so standardizing one with the other's mean and spread would put
every target at a location the head then has to undo.

## The head

One **bivariate Gaussian mixture per cell** (`head.py`, shared by both
backbones): mixture weights, a mean vector, and a full 2x2 covariance per
component via a Cholesky factor `L = [[softplus(a) + floor, 0], [b, softplus(c)
+ floor]]` - positive diagonal by construction, unconstrained off-diagonal, so
the correlation is free over (-1, 1) and no matrix is ever inverted (the
Mahalanobis term is a triangular solve).

**Full covariance is the point.** The payment-drain correlation is a per-cell
quantity - strong late in development, weak at dev 1 - and it is what this entry
exists to learn. Two independent marginal heads would give the same means and
the wrong joint, so a sampled diagonal fed back into the rollout would carry
paid and case movements that do not offset each other.

**The loss is MIXED-OBSERVEDNESS** (`head.nll_mixed`): one scalar, three
disjoint masks, no data discarded.

- both targets observed -> the joint bivariate density;
- paid observed, case movement not -> the closed-form **paid margin**;
- case movement observed, paid not -> the **case margin**.

All three are the same fitted head marginalized - a Gaussian mixture's margin is
the mixture of its components' margins, exactly - not a second model. The
alternative, dropping every cell without both targets, would throw away the
majority of the signal on the field the board actually scores. The masks must
partition the scored cells and that is CHECKED rather than documented: an
overlap would score a cell under two densities, which no output would reveal.

The head is gradient-tested against finite differences (including near-singular
components and masked cells), and each margin is checked against numerical
quadrature of the joint - the case margin's `sqrt(L10**2 + L11**2)` degrades to
`L11` if the off-diagonal is dropped, which is a perfectly valid-looking density
that understates every case scale by the size of the correlation.

## Backbones

`config.backbone` selects the encoder body; everything else - contract, head,
loss, training scheme, rollout, held-out wiring - is literally the same code, so
the two are an ablation of the encoder alone.

- **`"transformer"` (default)**, `network_transformer.py`: the masked-cell
  attention encoder of `nn_transformer`. Every cell is a token
  (`[channel values * channel flags, channel flags]`, width 2F = 4), plus origin
  and dev embeddings, a RELATIVE calendar embedding (distance past the
  conditioning cutoff) and broadcast LOB + log-premium conditioning. One cell's
  value moves every other cell's prediction in the same forward pass.
- **`"gru"`**, `network_gru.py`: the per-origin GRU encoder/decoder of
  `deeptriangle`, dispatched per dev step on channel 0's flag. Each origin is an
  INDEPENDENT sequence at inference - the recurrence runs along development only
  - and relative calendar position arises structurally from the recurrence, so
  this body needs no calendar input at all.

**Which to run.** The transformer is the default because attention across the
whole grid is what carries a case shock at one origin into another origin's
prediction, and cross-origin propagation of information is the reason to
condition on case reserves at all. The GRU is the honest control: it gets the
same joint head and the same simulated state, so a win for the transformer that
survives it is about attention, and one that does not is about the head. Run
both on any study that claims either.

**The config refuses a foreign knob.** A transformer-only knob set on a `gru`
fit (or the reverse) raises by name in `__post_init__` rather than sitting
inert - a config that describes a network nobody built is the repo's named
inert-parameter bug class.

Not carried over from `nn_transformer`: its optional exposure-aware sigma. That
lever rescales a univariate sigma, and the scale here is a Cholesky factor whose
off-diagonal is the quantity of interest; widening it by a premium power is a
design question of its own rather than a flag to copy.

## Training

Identical calendar machinery to the rest of the family (`gallery/nn/_scheme.py`,
`gallery/nn/_training.py`):

- **pooled** over every cohort (company x line of business); a batch is 64
  cohort-triangles;
- **calendar-cutoff augmentation**: per cohort per batch, draw a fake as_of
  diagonal, condition on cells on/before it (per channel, so a case level past
  the cutoff is masked rather than read) and score the observed training cells
  strictly after it;
- **validation by eval_date**: the trailing `val_diagonals=1` observed diagonal
  of the training window is excluded from every context and target; early
  stopping on its NLL, best weights restored;
- AdamW (lr 3e-4, weight decay 1e-2), gradient clip 1.0, deep ensemble of 5
  independently seeded members.

**Early stopping tracks the MIXED loss, not the paid margin alone**, and that is
the one deliberate departure from `deeptriangle` (whose early stopping watches
the target head only, so model selection is not coupled to `aux_weight`). Here
there is no auxiliary weight and no second loss to be coupled to: one density
covers both coordinates, and the case coordinate is consumed at prediction time,
so a member that fits paid well and case badly degrades the paid projection it
would be selected for.

## Prediction: the state-update rollout

Autoregressive, diagonal by diagonal, over BOTH channels. Per future calendar
diagonal, per draw:

1. one forward pass, one **joint** sample per future cell on that diagonal -
   same component, same normal draw for both coordinates, which is what carries
   the learned correlation into the simulated diagonal;
2. the paid increment ratio is un-standardized and written into channel 0;
3. the case **level state** is advanced, `level += movement`, in RATIO space,
   and the new level is re-standardized with the LEVEL channel's own per-dev
   statistics and written into channel 1;
4. **both** channels' flags are promoted at those cells;
5. re-encode, next diagonal.

Step 3 is where an implementation can be plausibly wrong: standardized values
cannot be added. Each dev has its own mean and spread, so `z_level[d] +
z_move[d]` is not the standardized new level, and a rollout built that way
produces smooth, finite, entirely believable numbers. The movement is integrated
on the ratio scale - the only scale on which a level and a movement are the same
quantity - and re-standardized afterwards.

Step 4 is the mirror image of every other NN entry. They promote channel 0's
flag ALONE, precisely because they simulated nothing else and raising a feature
flag would present the contract's padding zero as an observed value. This entry
simulated both channels, so both flags rise - which is what per-channel
promotion was built for in 0.5.4.

Ultimates are rebuilt exactly as the family does: `latest_cum + premium x sum of
future paid increment ratios`, pooled over ensemble members (1000 draws by
default). `predict(segment=...)` slices the cached global rollout - fit once,
score every cohort. **`predict` is paid-only** for board comparability.

## The drain diagnostic

`entry.case_paths(per_diagonal=True)` returns the full walk,
`(n_draws, n_levels, n_c, n_w)` over the projected calendar diagonals - where
the level steps down and where a shock lands, not only where it ends.
`entry.case_paths()` returns `(n_draws, n_c, n_w)` - the simulated **terminal
case level ratio** per (draw, cohort, origin), off the same cached rollout as
`predict`, so they are the same draws. It is a **diagnostic accessor, not a
`PredictiveDistribution`**, and it is deliberately not one: there is no
realized-case column on any board to score it against.

What to read from it:

- the terminal level should **concentrate near zero**. A case reserve that has
  done its job is nearly exhausted by the end of the projection;
- a **fat positive tail** is the model saying development continues past the
  triangle's window - real information about the tail, and a reason to distrust
  the ultimate at face value rather than a bug;
- a mass of **negative** terminal levels IS a defect: the movement head is
  unconstrained, so nothing stops a drain from overshooting into a case reserve
  below zero. Nothing in v1 prevents it (see "Limitations").

## Held-out scoring

The entry mixes in `PooledMDNHeldout` (`gallery/nn/_heldout.py`) UNCHANGED, like
the four univariate NN entries, and supplies only the two hooks:

- `_forward_mixture` returns the head's **paid margin** `(log_pi, mu_p,
  sigma_p)` over the grid - a univariate mixture, which is exactly the shape the
  mixin scores and samples. It is the fitted joint head marginalized exactly,
  not a second model, so the board row is a number from the model that was
  actually fitted.
- `_heldout_inputs` builds one cohort's forward inputs from the contract's
  `x_obs` (per-channel conditioning), plus the as_of calendar cutoff for the
  transformer backbone. The GRU arm carries no cutoff key at all, because that
  body has no calendar boundary to place.

Everything else is the shared implementation: `heldout_measure = "loss_ratio"`
(the standardization Jacobian is folded in, the base class subtracts
`log premium` to reach Lebesgue-on-amount), `heldout_draw_scale = "incremental"`
(the base class anchors each draw onto its cell's training-diagonal
predecessor), the ensemble members as the density's draw axis, the CONTRACT's
premium on both hooks with the caller's verified against it, and the pinned-dev
asymmetry (draws survive as a point mass at the pooled dev mean; the density is
refused). Held-out draws use `heldout_n_draws = 10_000`, the count every other
CRPS-capable entry puts on the board, which is affordable because a held-out
diagonal is one forward pass per member rather than a rollout.

**The case margin is NOT board-scored in v1** - see "Limitations".

## Evaluate flow

```python
from ibnr import gallery

entry = gallery.fit(
    "nn_paid_case",
    tri,
    paid_field="paid_loss",
    case_field="case_reserve",
    as_of="1997-12-31",
)
pred = entry.predict(segment={"company_code": code, "line_of_business": line})
realized = entry.realized_ultimates(tri, segment={"company_code": code, "line_of_business": line})
pred.summary(observed=realized)  # the same Meyers-style table as every entry

# the drain diagnostic: terminal case level ratio per (draw, cohort, origin)
terminal = entry.case_paths()

# the GRU control, same head, same loss, same rollout
gru = gallery.get("nn_paid_case").config_class(backbone="gru", hidden_dim=64)
control = gallery.fit("nn_paid_case", tri, config=gru, as_of="1997-12-31")
```

## Limitations

- **No realized-case board column, so the case path is unvalidated in v1.** The
  entry is scored on its paid margin alone, and nothing in this release checks
  whether the simulated case levels are calibrated - only that the paid
  projection they feed is. A case-side holdout (score the case level at the next
  diagonal the way `next_diagonal` scores paid) is the obvious follow-up and is
  not built.
- **The movement head is unconstrained**, so a simulated case level can go
  negative, and a long projection can drift there cell by cell. A non-negative
  parameterization (predict a log-drain, or floor the level at zero) would fix
  the sign at the cost of the upward shocks that motivate the entry, so v1
  keeps the honest unconstrained version and reports the terminal level as a
  diagnostic instead.
- **Single line.** Each cohort is encoded independently, so a company's per-line
  draws carry no cross-line dependence (`nn_transformer_ml`'s job).
- **Small data is still the central risk.** Two channels and a wider head mean
  more parameters per training cell than the univariate entries have; the
  mitigations are the family's (tiny networks, dropout, weight decay, cutoff
  augmentation, eval_date early stopping, ensembling, no company embedding by
  default) and the first lever if validation NLL diverges is a narrower body.
- **The case field must exist and must be a level.** Schedule P carries
  `case_reserve` directly; a mart that carries only paid and reported would have
  to derive it (reported - paid) before fitting, and a derived case reserve
  inherits both fields' holes.
- Origins with no observed cells get pure-extrapolation ultimates (anchor 0) and
  a case state starting from zero; origins without premium produce NaN
  ultimates.
- When the case cell at the paid anchor is missing while an earlier one exists,
  the rollout starts the state from that stale earlier level and the movements
  over the skipped devs are never sampled - on a draining reserve the start is
  overstated by the skipped drain, silently, and the paid projection conditions
  on it. Rare on the mart (the case reserve sits on the same statement rows as
  paid) but reachable through the hole-inheritance path above.
