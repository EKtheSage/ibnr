# tlrn - the transformer loss reserving network

Family: `nn`. Cohort: a company. Config: `TLRNConfig`. Held-out mixins: none.

## Lineage

A reproduction of the Transformer Loss Reserving Network of a companion
manuscript in preparation, built here from its R implementation onto ibnr's data
contract.

Reproduced: the engineered features and their two forms, the axial attention
body, the positive development-factor head with its cumulative projection and
its fallback for unsupervised steps, the point objective with its three terms
and its minibatch denominators, the checkpoint protocol, the best-two-of-ten
seed selection, and the size-stratified historical residual calibration.

Not reproduced: the shrinkage of this model's reserve toward a classical one.
The study supplies that weight as a constant selected on historical origins and
ships no search for it, so forming the blend is the analysis notebook's job and
`predict_reserve_draws` exists to put a calibrated spread around whatever point
the blend produces.

Seeds in torch and in R torch are different streams, so a rerun reproduces the
protocol and the magnitude of the result, never the digits.

## Data

`nn_company_data` on company cohorts. One training example is one company at one
accident year, and its tokens are that year's (line, development lag) cells,
line-major with the lag varying fastest. Lines the company does not write are
carried on the axis and marked, so the line index means the same thing for every
company.

A cutoff is a 1-based calendar diagonal `K`; a cell is visible when
`origin + lag + 1 <= K`. Every feature is estimated from visible cells alone,
which is what lets one fit be trained at many cutoffs on one triangle.

Eight features from paid alone, in this order:

1. the standardised incremental paid loss ratio at each visible cell;
2. the visibility flag itself;
3. whether the company writes the line;
4. that company's own mean incremental paid loss ratio at the token's lag;
5. that company's own volume-weighted incremental-to-cumulative ratio at the lag;
6. the cumulative paid loss ratio the accident year has reached;
7. how much of the development is already observed;
8. how many development steps past that the token sits.

Naming `incurred_field` and `case_field` adds five, each reading its channel a
different way: the company's mean incurred emergence at the lag, the
standardised incurred emergence at each visible cell, the paid-to-incurred ratio
reached, the incurred loss ratio reached and the case reserve loss ratio
reached. They are named by ROLE rather than swept as a channel list because no
channel list can say which is the emergence and which the outstanding balance.

Cell-level features are standardised per (line, lag); company-level ones over
the whole array of that quantity. The clip limits are named constants in
`kernels/nn_features.py` and are wide enough to be inactive on ordinary data.

A written line missing a cumulative that the cutoff does not hide is refused by
name: the projection starts from the cumulative at each origin's latest visible
lag, and a missing one would silently anchor that origin at one dollar.

## Network

Input projection, plus three embedding tables - the line, the lag, and how many
lags of the accident year are observed - then `n_layers` axial blocks, a final
layer norm, and a one-number-per-token output layer. A block attends across the
LINES within a lag, then across the LAGS within a line, each pre-norm with a
residual path, then a GELU feed-forward of twice the token width.

The cross-line attention carries a key padding mask, so a written line's output
cannot depend on a line the company does not write. An example with no written
line is refused rather than answered, because every key would be masked and the
output would be NaN.

At `d_model = 32`, two heads, one layer, 13 features, 4 lines and 10 lags the
network has **14,309 parameters**; the paid-only form has 160 fewer. All three
embedding tables carry one unused row, because the indices are 1-based as the
reference implementation's are.

`cross_line = False` skips the line attention while still building it, so the
comparison between the two arms is a comparison of the attention rather than of
the model size.

## Head

The network does not predict a cell. It predicts a log development factor per
(line, step), `softplus(phi + eps * net)`, where `phi` is a learned per (line,
step) parameter and `net` the network's correction to it. The softplus keeps
every factor above one, so a cumulative can only grow - paid recoveries are
outside what this head can express.

The cells follow by projecting each origin's cumulative forward from its latest
visible lag and differencing back, so the model cannot disagree with itself
about two cells of one origin. Set the factors to the pooled chain ladder's and
switch the network off and the head reproduces the chain ladder exactly; that is
the reference implementation's first check and it is a test here.

`tail_policy = "observed_cl"` substitutes, at every step no training target was
projected across, the chain ladder factors observable at the example's own
cutoff. Such a step has never received a gradient, so reporting its parameter as
a learned tail would be reporting the initialisation.

`cl_anchor` is the switchable variant: start at each company's own chain ladder
factors and learn a correction bounded by `anchor_width`. Its output layer is
zeroed at construction, so it starts exactly at the chain ladder rather than
arriving near it, and it projects from the unfloored starting balance so an
origin that has paid nothing stays at nothing.

## Training

The objective is the accident-year/line absolute percentage error, plus `w_pe`
times a pooled bias penalty, plus `w_mse` times a masked squared error over
`mse_scale`. The first term sums signed dollar errors WITHIN an (accident year,
line) before taking the absolute value, so a development error inside one line
costs nothing while being wrong about the year costs the difference. The second
sums everything before the absolute value, so it sees only the bias. All three
use the minibatch's own denominators, which is what the protocol does.

The schedule is a linear warmup over `warmup` epochs then a cosine to zero. The
log factors train at `lr_phi`, ten times the network's `lr`, because they carry
the level of the forecast while the network only corrects it.

One cutoff is drawn per EPOCH and shared by every batch of that epoch
(`cutoff_sampling = "per_epoch"`), and a batch is 64 examples. The trailing
`val_diagonals` calendar diagonals validate; training cutoffs run from
`min_cutoff` to `c_max - val_diagonals - 1`, and the two validation sets score
every diagonal still held out at their own cutoff rather than one step only.

`patience = 600` counts validation checks, so at `check_every = 5` it is 3000
epochs - the whole budget, and it never fires. That is the protocol: every
member runs the full schedule and the retained state is its best validation
checkpoint, with no refit afterwards. `patience = 120` is the 600-epoch
early-stopping variant, which is a different run and has to be asked for.

Ten members are trained under independent seeds and the two with the lowest
held-out accident-year/line error are kept. `selection_` reports every member -
its best epoch, its epochs run, both validation errors and whether it was kept -
because a selection over ten optimiser outcomes is part of the result, and a
table showing only the survivors would hide how much of the reported score is
the selection rather than the model.

What that costs. Measured on the study's own company set - 93 companies, 243
company-line pairs, four lines, accident years 1998 to 2007, valuation
2007-12-31, from the Schedule P publish `20260613_041006` - on a Windows laptop
with 16 torch threads, one member, no parallelism: about **1.1 seconds per
epoch** paid-only and about **1.2 seconds per epoch** with the incurred and
case features. So one member's full 3000-epoch schedule is roughly an hour, and
the ten-member protocol run one after another is roughly 9 to 10 hours.

Treat those as one significant figure. Eleven runs of the same two
configurations on an otherwise idle machine spread from 0.8 to 1.5 seconds per
epoch, which is wider than the gap between the two forms, so the ratio between
them is not something this measurement establishes. Each timing also includes
the one-off setup - reading the mart and building fourteen feature sets for 93
companies - so the per-epoch figure is an upper bound and the projection is
conservative.

## Prediction

`point(segment)` gives one company's deterministic ultimates in the multi-line
layout - per (line, origin), per-line totals, grand total - as the mean over the
kept checkpoints. `point_cumulative_` carries the whole projected cumulative
grid with the observed part held at the triangle's own values, so a caller can
read the next calendar diagonal without a second forward pass.

`predict(segment)` gives ONE target, that company's total ultimate. Its draws
are NOT a native predictive distribution: the kept checkpoints are applied at
`calibration_cutoffs`, their company-level errors are standardised by
`max(|predicted|, 0.01 * premium, 1)`, pooled by premium stratum, centred on each
stratum's median and resampled around the final point. So the spread says how
wrong this model has been historically on companies of that size.

Two consequences worth stating plainly. The calibration cutoffs overlap the
training targets at the earlier ones - a forecast made at cutoff 5 and scored to
diagonal 10 covers cells the network trained on - which is the study's own
choice and makes the earlier residuals optimistic. And the calibration is at
company level only, so this entry joins neither held-out mixin and produces no
per-cell draws; spreading a company total over cells would invent cell
uncertainty the method never claimed.

`predict_reserve_draws` puts the same calibrated spread around any company
reserve vector, which is what the study's blended point needs.

One thing to know before reading `evaluate()`'s point block. Its only target is
labelled `"total"`, and `kernels.point_scores.point_summary` drops a target with
that label, because on a multi-line layout the total is the sum of the others
and scoring it too would count every error twice. So `point["errors"]` carries
the company's one row while `point["metrics"]` is `None`, with
`point["excluded"]["total"]` saying which exclusion emptied it. That is the
right answer for a per-target table; the cross-model point board reaches this
entry through `reserve_rows(point="native")`, which reads the deterministic
point rather than this table.

## Evaluate flow

```python
from ibnr import gallery

entry = gallery.get("tlrn")().fit(triangle, loss_field="paid_loss", as_of="2007-12-31")
company = entry.cohorts()[0]

point = entry.point(company)  # (line, origin) rows, per-line totals, grand total
draws = entry.predict(company, seed=11)
scores = entry.evaluate(entry.realized_ultimates(triangle, company), company)

blended = 0.658 * entry.company_reserves() + 0.342 * classical_reserves
blended_draws = entry.predict_reserve_draws(blended, seed=11)
```

## Limitations

- No per-cell draws and no per-cell density, so neither held-out mixin and no
  place on the per-cell board. The point reserves join the point board.
- The head cannot produce a development factor below one, so a line with genuine
  paid recoveries is misfitted rather than refused.
- A written line missing a visible cumulative is refused, which means a company
  with an incomplete grid has to be excluded from the cohort rather than fitted
  around.
- Loss aggregation by company, the study's other training objective, is not
  built; only the accident-year/line form is.
- The feature channels are frozen inputs: nothing simulates incurred or case
  forward, so a deep projection conditions on progressively less than a shallow
  one.
