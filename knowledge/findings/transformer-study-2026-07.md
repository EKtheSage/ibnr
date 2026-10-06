---
type: Finding
title: The transformer study of July 2026
description: Measured results from the milestone-3 work on the single-line and multi-line triangle transformers (v1 to v3, training pools, exposure-aware sigma), including what the numbers do and do not support.
tags: [nn, transformer, nn_transformer, nn_transformer_ml, results, milestone-3]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, milestone-3 entries of 2026-07-06 to 2026-07-08 (lines 494-592)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: results
    resource: analysis/results/ (compare_gallery.csv, experiment_exposure_sigma.csv, experiment_v3_market.csv, *_v1_inherit_stats.csv, *_v2_sixcompany.csv)
    title: The study's results files
---

`CLAUDE.md` (milestone 3) already records the headline numbers. This concept keeps the
measurements and reasons it does not.

# v1 to v2: never inherit normalization across lags

With a validation split on the trailing diagonal, the deepest development lag never
appears as a training target. v1 borrowed the nearest earlier lag's normalization
statistics there, which put tail cells back on a mid-development scale: paid private
passenger auto error +20%, reported +148%, coefficients of variation 5 to 10 times too
wide. v2 pins such a lag instead (standardized value 0, mean from the observed cells at
that lag, standard deviation 1, rollout draws forced to the pooled lag mean). The fix was
the colleague's suggestion (2026-07-07). Never reintroduce statistic inheritance across
lags.[^note]

First backtests (6 companies passing the selection rule on all four lines, 24 line
totals; v1 files kept as `*_v1_inherit_stats.csv`):[^note]

* **Paid.** All three models failed the combined KS test with percentiles clustered low.
  This is the post-1997 favourable-development regime (Meyers saw the same), a regime
  finding and not a bug. SUR sharpest (relative CRPS 0.034); transformer v2 about equal to
  the copula (0.050).
* **Reported.** Transformer v2 was the only model passing the combined KS test at 5%
  (D = 26.7 against a critical value of 27.8; SUR 27.8 with p = 0.049; `meyers_ccl` 31.3).
  Error +1.7%, relative CRPS 0.030, about equal to `meyers_ccl`; SUR still sharpest at
  0.022.

# v3: relative calendar encoding

Encoding calendar time as distance past the conditioning cutoff replaced a learned
absolute calendar embedding that was untrained at the forecast diagonals. Paid mean
relative CRPS fell from 0.050 to 0.030, and the single-line transformer then beat SUR
(0.034) on the 6-company set.[^note]

# Training pool

Ordered from best to worst: the multiline pairs that are scored, then the scored pairs
plus single-line companies, then the full mart. Each wider pool was worse. The pool
interacts with the pinned normalization, because pinned means are pooled statistics.
Ethan's comparability rule (train only on the scored (company, line) pairs) is also the
best-performing pool, so fairness and performance do not pull apart.[^note]

# Why reserve-basis point errors

The widened study (60 companies, 152 pairs) reports point errors on the reserve, not the
ultimate: an ultimate-basis error flatters a model for the paid-to-date it merely copies.
Ethan wanted a point comparison free of distributional assumptions beside CRPS and
KS.[^note]

Dependence can be read from the results files alone. The implied average correlation
across lines is[^note]

```text
(se_ALL^2 - sum_i se_i^2) / (2 * sum_{i<j} se_i * se_j)
```

# Exposure-aware sigma: a small win, off by default (2026-07-08)

`nn_transformer`'s `exposure_sigma` flag multiplies the mixture head's sigma by
premium^(p-1) about the pooled mean premium, with one learned power p = softplus(raw_p),
so a dollar standard deviation scales as premium^p. It starts at p = 1, which equals the
flat-sigma model exactly, so with and without is a clean comparison
(`compare_gallery.py --nn-exposure-sigma`). The data learns p of about 0.965 (5 members,
0.952 to 0.982).[^note]

| measure | flat sigma | exposure sigma |
|---|---|---|
| combined paid KS D | 18.3 | 16.6 |
| median relative CRPS | 0.0267 | 0.0252 |
| reserve MAE, points of premium | 3.00 | 2.95 |
| chain-ladder skill | 1.94 | 1.60 |

It cannot fix private passenger auto, where outcomes pile low: that is a location or
regime bias, not a width problem. Kept as an opt-in lever; results in
`experiment_exposure_sigma.csv`. The market-pool run's file was renamed to
`experiment_v3_market.csv`. Remaining levers named: a per-line p, then hyperparameter
search in `kernels/tuning.py`.[^note]

# What the single-line transformer cannot do

Its attention stays inside one company-and-line triangle. It has no mechanism for
dependence across lines (only shared weights and a line embedding), and its per-line
draws are independent. SUR and the copula do model it (checked in the code). Never claim
otherwise in a manuscript.[^note]

[^note]: Project status log, milestone-3 entries of 2026-07-06 to 2026-07-08 (lines 494-592)
