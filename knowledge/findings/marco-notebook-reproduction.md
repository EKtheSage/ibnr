---
type: Finding
title: Reproducing the companion notebook
description: What ibnr's accident-year variant matched in the companion notebook, the four differences found and fixed, and the reduced-scale result on the notebook's own cohort.
tags: [tlrn, reproduction, companion-study]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:20:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: notebook
    resource: transformers_reserving/Code/tlrn_pipeline_best.ipynb (sibling repository checkout)
    title: The companion notebook (JAX)
  - id: tables
    resource: transformers_reserving/Exhibits/tab (M_T0_company_reserves.csv, R_T1_rolling_origin.csv, T_T1_members.csv)
    title: The notebook's published tables
  - id: pr-164
    resource: https://github.com/EKtheSage/ibnr/pull/164
    title: The design choices and the fidelity fixes
---

# Result

`TLRNConfig.accident_year_variant()` reproduces the notebook's method closely enough
that, at about a tenth of its training, the company-reserve Pool_APE is **0.0557
against the notebook's 0.0555 at full scale**. The published `tlrn` at the same scale is
0.0579, the multivariate chain ladder 0.0541 and the chain ladder 0.0548.[^tables] The
variant is level with chain ladder, not clearly better: that is the notebook's own
finding too (48.8% of its bootstrap resamples favour TLRN).

# What was checked

* **Cohort.** Applying the notebook's selection rule to its cached CAS files gives 84
  companies and 212 company-line pairs, as the notebook reports.
* **Classical forecast.** ibnr's multivariate chain ladder reserves match the notebook's
  `IBNR_CL_MCL` for all 84 companies to a maximum relative difference of 3.7e-13.
* **Head and training protocol.** The exact weighted-median alpha, the loss, the two
  learning rates (0.03 for the per-lag ratios, 0.003 for the network), gradient clipping,
  one training cutoff per epoch, batches of 8 whole companies, the observed-cell mask on
  all three attentions, flat multivariate chain ladder steps beyond the cutoff, the cap of
  5 on the log ratio, and the calibration scaling all matched.

# Four differences found by reading the notebook, and fixed

1. The pooled incremental loss ratio is a **plain mean** over visible cells, floored at
   1e-4, with a lag no visible cell reaches taking the line's last known lag (0.05 if
   none). The first port used a premium-weighted ratio floored at 1e-6, which would have
   forecast about zero at the late lags of an early valuation.
2. The premium head uses the same **factor-support rule** as the factor head, applied to
   lags 1 onward, not a separate per-lag rule.
3. Validation and the retrained backtests **score only cells whose development steps the
   cutoff has seen** (lag at most the cutoff). Now `scoring="reached_cells"`.
4. Alpha is **0, not 1**, when the network and the chain ladder never differ.

# Still different

Embedding tables are 0-based in the notebook and 1-based in ibnr, so parameter counts
differ (the notebook's is 18,537). The two use different random streams, so the same
seed does not give the same numbers.

# Reduced-scale numbers

8 members, 600 epochs, same cohort. Backtest Pool_APE at valuations 6, 7 and 8: 0.0375,
0.0350, 0.0431 (the notebook at full scale: 0.034, 0.029, 0.038). Median alpha 0.37
(0.51 at full scale).

# Where the members' best checkpoints fall

In the notebook's own member table the best epoch is early for the earlier valuations:
median 98 (maximum 900) at valuation 6, 628 (1,275) at 7, 715 (1,820) at 8 and 1,598
(2,435) at 10. Training all 3,000 epochs of the earlier valuations is therefore mostly
unused, and early stopping with a patience of about 600 epochs would cut roughly 30% of a
full run, at the price of changing the protocol.

[^tables]: The notebook's published tables
