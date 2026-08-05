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
