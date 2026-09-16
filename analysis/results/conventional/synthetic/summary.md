# Synthetic conventional-procedure results

Run: 2026-09-15. Thirty prespecified seeds per scenario, 120 portfolios total;
42 candidates per portfolio; 600 evaluated forecasts including three fixed
baselines. Every forecast has all seven unknown-at-selection terminal targets
observed. See [the protocol](../../../../docs/conventional-benchmark.md),
[individual results](results.csv) and [source manifest](manifest.json).
The same 30 seeds are paired across scenarios; these are 120 scenario/seed
cases, not 120 independent replications. See the shared
[environment record](../environment.json) for runtime provenance. Two recorded
values are to be read literally: `manifest.json` says `ibnr_version` 0.5.8
because the editable install's metadata was stale when the run executed, not
because the code came from that release, and `environment.json` says `git_head`
is the base commit because these runs predate this branch's first commit. The
six `source_hashes` in the manifest are the authoritative pin on what ran; they
matched the committed source files when this was checked.

## Mean terminal RMSE

Amounts below are **thousands of synthetic USD**. Each number is the mean of
30 portfolio RMSEs; lower is better. These are point-forecast errors, not
calibration scores or confidence intervals.

| Scenario | Basic CL | Basic BF(.60) | Basic GCC(.75) | AvE-selected | CDR-selected |
|---|---:|---:|---:|---:|---:|
| Stable | 109.23 | **73.22** | 76.26 | 77.53 | 82.64 |
| Noisy | 214.62 | **118.42** | 136.43 | 140.42 | 125.15 |
| Drift | 214.47 | 493.83 | 220.14 | **175.83** | 275.19 |
| Unseen shock | 191.06 | 214.35 | **185.18** | 186.66 | 206.65 |

## Paired comparison with basic CL

Percentage change is `(mean selected RMSE / mean basic CL RMSE - 1) × 100`.
Negative is an improvement. Win counts compare the two forecasts on the same
portfolio seed and exclude ties.

| Scenario | AvE mean change | AvE wins / 30 | CDR mean change | CDR wins / 30 |
|---|---:|---:|---:|---:|
| Stable | -29.0% | 23 | -24.3% | 20 |
| Noisy | -34.6% | 22 | -41.7% | 23 |
| Drift | -18.0% | 19 | +28.3% | 14 |
| Unseen shock | -2.3% | 20 | +8.2% | 16 |

## Interpretation

- AvE selection improves on basic CL on average in each of these four scenarios.
  It does not beat the best fixed baseline in every scenario: fixed BF wins
  stable/noisy, and fixed GCC narrowly wins the shock scenario.
- CDR selection improves on CL for stable/noisy portfolios but worsens mean
  error under drift and the unseen shock. A win rate above one half can coexist
  with worse mean RMSE when the losing portfolios have larger errors.
- The stable and shock scenarios use identical history through selection. Their
  same-seed selected settings and historical scores are identical. Their later
  losses differ, demonstrating a limit of using past performance to select a
  forecasting procedure.
- This is an exploratory synthetic stress test with 30 seeds per scenario.
  The generator, parameter ranges and comparison baselines were fixed before
  inspecting these results. The simulation is not evidence that one selection
  metric is universally superior or that the library matches ReserveAI.
