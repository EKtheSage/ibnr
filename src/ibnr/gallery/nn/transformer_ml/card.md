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
helper, one implementation.

## Network

Token = `Linear([channel values * flag, flag])` + line embedding + origin
embedding + dev embedding + relative calendar embedding (distance past the
conditioning cutoff) + per-line normalized log premium. Encoder: 2 pre-LN
transformer layers, d_model 64, 4 heads, FFN 128, dropout 0.15, GELU.
Sequence length L*W*D (400 for four Schedule P lines).

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

- Everything the single-line card lists (small data, feature channels not
  simulated in rollout, unsupervised deepest devs behind the pin).
- "ar": within-diagonal dependence is directional; randomized order makes
  the marginal draw exchangeable across draws, not within one draw.
- "joint": L*(L+1)/2 covariance parameters per component per cell-group is
  the overfitting frontier; watch validation NLL vs "ar" closely.
- Companies with different line sets share one encoder; the padding mask
  removes absent lines, but the pooled normalization means thin lines lean
  on the market's scale.
