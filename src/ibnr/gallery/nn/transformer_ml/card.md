# nn_transformer_ml - multi-line triangle transformer with cross-line dependence

**Family:** nn (PyTorch; requires the `[nn]` extra at fit time - registration
does not)
**Lineage:** the single-line `nn_transformer` promoted to the COMPANY level.
A cohort is one company; every (line, origin, dev) cell is a token and one
encoder attends across the company's lines - the learned analogue of SUR's
contemporaneous correlation (Zhang 2010) and the copula's cell-level
dependence (Shi & Frees 2011), which are this entry's explicit-baseline
comparators.

## Data

`kernels.nn_contract.nn_company_data`: `nn_data`'s (company, line) cohorts
regrouped along a line axis, so screening/increment/premium rules are
identical by construction. Targets are incremental loss ratios per line.
Lines a company does not write are `line_mask`-ed out: excluded from
attention (padding mask), losses, and rollout. Pinned per-(line, channel,
dev) standardization as in the single-line model v2 - stats via the same
helper, one implementation, each channel standardized on the cells where
that channel itself has a value.

Conditioning is PER CHANNEL, on the contract's `x_obs`: a feature that is
missing where the target is observed is masked out rather than read as a
zero increment. In training every channel's flag is additionally gated at
the drawn cutoff; the rollout gates by observedness alone, so a feature
booked on a deeper calendar diagonal than every target cell reaches it in a
configuration training never showed - kept, because that contemporaneous
cell is genuinely informative. `fit(feature_fields=...)` names the extra input channels;
`fit(level_fields=...)` declares which of them are eval-date snapshots
carried undifferenced - `case_reserve` is the motivating one - with the
semantics and refusals in `kernels.nn_contract.nn_data`.

## Network

Token = `Linear([channel values * channel flags, channel flags])` + line
embedding + origin embedding + dev embedding + relative calendar embedding
(distance past the conditioning cutoff) + per-line normalized log premium.
One flag per channel, so no value is consumed without its own channel's
flag; with a single target channel (the default) that is the same two inputs
as the per-cell flag it replaced.

Encoder: 2 pre-LN transformer layers, d_model 64, 4 heads, FFN 128, dropout
0.15, GELU. Sequence length L*W*D (400 for four Schedule P lines).

The line embedding is ADDED to the token, so its width is necessarily
`d_model` and there is deliberately no `line_embedding_dim` config knob. This
differs from the single-line entry, which CONCATENATES its LOB embedding with
the conditioning scalar and projects the result, and so can size that
embedding independently (`TransformerConfig.lob_embedding_dim`).

**Two dependence heads** (`config.dependence`) - the research question is
which mechanism carries cross-line dependence better:

- **"ar"** - univariate K=3 MDN per cell (as the single-line model). In the
  rollout, each calendar diagonal is sampled one LINE at a time in a seeded
  random order, each line's samples fed back as context before the next -
  dependence purely via conditioning. Line-order asymmetry is the known
  cost; the order is re-drawn per diagonal.
- **"joint"** - per (origin, dev) cell-group, a K=3 mixture of multivariate
  Gaussians over the line vector (Cholesky-parameterized scale). Absent or
  non-target lines are marginalized (a marginal of a Gaussian mixture is
  the mixture of marginals - covariance submatrix per line-pattern). All
  lines of a diagonal are drawn jointly in one shot.

## Training & prediction

Identical scheme to the single-line model: calendar-cutoff augmentation per
company (a cutoff applies to all lines - calendar is shared), trailing
`val_diagonals=1` validation with early stopping, AdamW, 5-seed deep
ensemble, pinned draws forced to the pooled dev mean. Rollout is diagonal
by diagonal; `predict(segment={"company_code": ...})` returns the SUR
layout - per-(lob, origin) ultimates, per-lob totals, grand total - so
cross-line diversification is visible in the draws and directly comparable
against `sur` / `copula_glm`.

## Held-out scoring

The entry mixes in `ScoresHeldout` and `PredictsHeldout`, so it reaches the
ELPD and CRPS boards like the single-line NN entries. What it needed that they
did not is an adapter: their cohort is already a (company, line) pair, while
this fit's cohort is a **company** and a held-out cohort built by
`next_diagonal` is one (company, line) pair. `gallery/nn/_heldout_ml.py` is
that adapter - `company_line_contract` slices one pair out of the company
contract and puts `line_of_business` back into the segment key, so
`index_into`'s cohort-identity, measure and training-overlap guards apply here
exactly as they do to a Stan fit. The density algebra, the refusals, the
exposure check and the draw loop come from `gallery/nn/_heldout.py`, unchanged
and shared with the four single-line entries.

Capability is therefore served **per pair**: `entry.at_cohort(segment)` returns
a scorer view whose `log_lik_at`/`predict_at` are the unmodified base-class
implementations. The segment must name the line - `at_cohort` refuses one that
does not, because the company alone does not identify a held-out cohort. The
entry-level `log_lik_at`/`predict_at` read the pair off the cells' own segment
values and delegate.

- **Conditioning and cutoff are read differently, on purpose.** The forward
  pass conditions on everything the COMPANY observed at as_of, per channel and
  across all its lines - that cross-line context is the entry's reason to
  exist. The relative-calendar cutoff, by contrast, is the **scored line's own**
  as_of diagonal, so its held-out cell sits at distance 1: the most-supervised
  position and the rollout's first step. Reading the cutoff off the company's
  deepest line instead would push a slower-reporting line's held-out diagonal
  two or three steps out, into a different embedding row and a different
  predictive distribution. On complete squares every line of a company reaches
  the same diagonal and the two readings coincide.
- **Draw scale: `incremental`.** One forward pass per ensemble member, mixture
  sampling at the requested cells, un-standardized on the scored line's own
  per-dev statistics (`z * std0[li, d] + mean0[li, d]`) and scaled by premium:
  an incremental dollar amount. The base class anchors it onto the cell's
  training-diagonal predecessor to reach the cumulative triangle basis, and
  derives the draw stream from the study seed together with the cells' cohort
  identity - so two lines of one company under one seed draw different random
  numbers.
- **Density measure: `loss_ratio`.** Per member, the mixture log density of the
  standardized ratio with the standardization Jacobian folded in; the base
  class subtracts `log premium` to reach Lebesgue-on-amount. The **draw axis is
  the ensemble members** (>= 2 required).
- **The `"joint"` head is marginalized to the scored line.** Its output is a
  mixture of L-variate Gaussians over the whole line vector, and a one-line
  score needs its marginal. The marginal of a mixture of multivariate Gaussians
  is the mixture of the components' marginals with the **weights unchanged**,
  so component *k* contributes weight `pi_k`, mean `mu_k[li]`, and standard
  deviation `sqrt(Cov_k[li, li])`. With `Cov = L L'` that variance is
  `sum_j L[li, j]^2`, i.e. the squared length of row `li` of the Cholesky
  factor, which is read directly rather than by forming the covariance matrix.
  The `"ar"` head is already univariate per cell, so there the line is just an
  axis to index. Both heads then run through the same density and sampling
  code.
- **Pinned-dev asymmetry**, per LINE (the normalizer is per (line, channel,
  dev)). At a pinned dev - fewer than two training-context values reached that
  line's per-dev normalizer, in practice the deepest dev - there is no trained
  head. Draws keep rollout semantics: the sampled value is forced to the pooled
  dev mean, a point-mass column, legal for CRPS; a request where EVERY cell is
  pinned is refused. The density is REFUSED outright, naming the dev steps. The
  entry is therefore CRPS-scorable at cells where it is not ELPD-scorable.
- **Display segments are narrowed away, and verified on the way.**
  `nn_company_data` keeps display-only columns (`company_name`) out of the
  company key, so this fit's key is narrower than the triangle's segment
  schema. `log_lik_at`/`predict_at` re-key the supplied `HoldoutCells` onto the
  fit's own schema first, checking each dropped value against the pair's full
  identity, so cells belonging to a different spelling are refused rather than
  quietly scored here.

## Dependence diagnostics (what to check first)

- diversification ratio: grand-total sd / sum of per-line sds, vs SUR and
  copula on the same company;
- implied correlation of line-total draws vs `sur.pooled_corr_` /
  `copula_glm.corr_`. If it is ~0 for the "ar" head, attention is not
  carrying dependence and the "joint" head is the fallback mechanism.

## Limitations

- Everything the single-line card lists (small data, unsupervised deepest
  devs behind the pin).
- Feature channels are NOT simulated forward. The rollout feeds back the
  target channel only, so a sampled cell carries a target value and no
  feature values, and the deeper the rollout goes the fewer observed feature
  channels it conditions on. That is now the honest representation - the
  feature flags stay off, where before promotion set one per-cell flag and
  the network read the contract's padding zero as an observed zero
  increment - but it still means a feature informs the near diagonals far
  more than the far ones.
- "ar": within-diagonal dependence is directional; randomized order makes
  the marginal draw exchangeable across draws, not within one draw.
- "joint": L*(L+1)/2 covariance parameters per component per cell-group is
  the overfitting frontier; watch validation NLL vs "ar" closely.
- Companies with different line sets share one encoder; the padding mask
  removes absent lines, but the pooled normalization means thin lines lean
  on the market's scale.
