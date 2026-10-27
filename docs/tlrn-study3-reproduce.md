# Reproducing the third tlrn study's B3 result

This guide is for someone who has the ibnr repository and a Google account, and wants to get the
B3 result of [the third study](tlrn-study3.md) (the companion's own cohort, accident years 1998
to 2007) from scratch, or to check one table of it. The result itself is written up in
[the finding](../knowledge/findings/tlrn-study3-results.md). The saved evaluator output is in
[`analysis/results/tlrn_study3/`](../analysis/results/tlrn_study3/README.md).

Commands below were run in the session that wrote this guide unless the text says "not run".
Where a command is quoted from the notebook or from `docs/tlrn-study3.md` and was not run again,
that is said too.

## 1. What you need

| piece | what the study used |
|---|---|
| data | cas-schedule-p publish `20260613_041006`, table `mart_reserving_model_training` |
| ibnr | `ibnr[jax]==0.7.6` from PyPI |
| Python on Colab | 3.13.16 (Linux), with `--ignore-requires-python` (see below) |
| jax | 0.11.1 (the Colab runtime's own; do not replace it) |
| torch | the Colab runtime's own; the version was not recorded in the fit records |
| accelerator | one NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab's G4) |
| Python on the laptop (scoring) | 3.12.13 with `uv sync`; no jax or torch needed |

**The data.** Nothing to download by hand. `tlrn_study_colab.cells_from_mart` resolves
`github://EKtheSage/cas-schedule-p-data-model@20260613_041006`, downloads the mart from that
GitHub release over anonymous HTTPS (the data repository is public), checks its sha256 against
the release manifest and caches it under `~/.cache/ibnr` (or the folder named by
`IBNR_CACHE_DIR`). The Colab notebook also installs `cas-schedule-p==2026.6.13`.

**Python 3.13.** ibnr declares `requires-python = ">=3.11,<3.13"` because two extras this study
does not use (`[bayesian]`, `[interop]`) cannot install on 3.13. Colab now runs 3.13, so the
notebook adds `--ignore-requires-python` to its two `pip install` calls when the interpreter is
3.13 or later. Outside Colab, use Python 3.12 and none of this applies.

## 2. Run the fits on Colab

1. Open the notebook in Colab (the link is the usual Colab address of a GitHub file; not opened
   by me, since the fits were made from a copy of the same notebook):
   `https://colab.research.google.com/github/EKtheSage/ibnr/blob/main/analysis/05_tlrn_study3_on_colab.ipynb`
2. Runtime, Change runtime type, a GPU. A G4 (Blackwell) is what the times below are for. An
   A100 should be about twice as slow and a T4 about 13 times slower (the planning figures in
   the pre-registration; only the G4 was measured).
3. Leave the settings cell alone except `SOURCE` and `REF`. The values that matter, as the
   notebook holds them: `EXPECTED_VERSION = "0.7.6"`, `STUDY = "companion_final"`,
   `COHORT_ORDER = ("C", "B")`, `MAIN_ROW = "tlrn_cf_refit"`, `FACTORS_ROW = "tlrn_cf_factors"`,
   `DRAWS = 10`, `MEMBERS = 20`, `EXPECTED_PARAMETERS = 18537`. The draw seeds are
   `11 + 100003 k` for draw `k`, the date `v` adds `DATE_SEED_STRIDE * v` with
   `DATE_SEED_STRIDE = 20_000` and member `m` adds `1000 m`.
4. Run the cells from the top, one at a time. The checks that fit nothing (section 6 of the
   notebook) must pass: they recompute the cohort selection and compare it with the published
   counts (B3: 230, 226, 219, 212 pairs and 90, 89, 87, 84 companies at dates 6, 7, 8, 10),
   and count the network's parameters.
5. For B3 alone, the notebook cell "B3, draws 0 to 9" calls
   `run_piece(tlrn_study_colab.main, OUT, "B", MAIN_ROW, first=1)`, and the notebook's draw 0
   call on C3 shows the pattern for one fit:
   `tlrn_study_colab.main(["--study", STUDY, "--out", OUT, "--cohort", "C", "--rows", MAIN_ROW, "--draws", "1"])`.
   Replacing `"C"` by `"B"` gives B3 draw 0 (not run in that form; the notebook's own B3 cell
   was what ran). The full notebook runs C3 first, then B3, then the network-off rows; B3 only
   needs the B3 cells and, for the comparison with the network switched off, the B3 part of
   section 11.

**What it costs, measured on 2026-10-26.** One main fit (`tlrn_cf_refit`, 20 members at each of 4
dates, 3,000 epochs) took 501 to 535 seconds on B3 (median 521). A network-off fit
(`tlrn_cf_factors`) took 31 to 33 seconds. B3 alone is therefore about 10 x 521 s = 1.45 GPU hours
for the main row plus 0.09 hours for the network-off row. The whole study (40 fits, both
cohorts) took 2.94 GPU hours of fit time and 2.9 hours from the first fit to the last. What this
costs in Colab compute units depends on your plan and was not measured. The notebook stops
by itself if draw 0 on C3 takes more than 25 minutes or the saved fits pass 8 GPU hours.

**Where the output lands.** `My Drive/ibnr_colab_cf/study3`, a flat folder (the evaluator does
not read subfolders). A disconnect loses at most the running fit; run the cell again and saved
fits with matching settings are skipped.

## 3. Score the fits

On a laptop with the repository, `uv sync` and Drive for desktop (or `rclone`) showing the
folder. The command is the one in [the study 3 how-to](tlrn-study3.md) (not run again with a
Drive path; run below with the folder of the saved fits):

```text
uv run python scripts/tlrn_study3_eval.py --fits "G:\My Drive\ibnr_colab_cf\study3" --cohort B --out C:\Temp\study3_report
```

`--cohort B` scores only B3; leave it out (or write `all`) for both cohorts. `--cache DIR` keeps
the benchmark fits so a second run skips them. The first run fits `mack` and `mcl` at the four
dates, which took about 45 s for B3 and 52 s for C3 in the run that produced the saved output;
a cached run of B3 took 24 s. In this session the same script was run through the project's
interpreter instead of `uv run` (the shared uv cache was locked by other processes) with the
fits folder, `--cohort B` and a copy of the saved benchmark cache. It printed the same verdicts
and reproduced the saved B tables to 1e-10 (relative) in every column except one:
`share_better_than_mcl` for `mack` at date 6 was 0.300 instead of 0.311, one company of 90
flipping on a near tie between two errors that differ at the 1e-12 level. So a rerun of the
evaluator is exact for Pool_APE and all verdicts, and a count of "companies better than" can
move by one company when two errors are tied to rounding.

The evaluator accepts Colab fits although the data hash text differs from the one it builds
(see "Data hash" in the finding). Add `--strict-data-hash` to make that an error instead.

## 4. What to expect for B3

Pool_APE (the sum of absolute reserve errors over the sum of actual reserves) of the draw
average of 10 draws. Date 10 is the last known diagonal (2007-12-31); "rolling" is the plain
mean over the four dates. The per-draw range is the spread over the ten draws.

| row | date 10 | per-draw range at date 10 | rolling |
|---|---|---|---|
| `tlrn_cf_refit` blended (the model) | 4.02% | 3.73 to 4.26 | 3.86% |
| `tlrn_cf_norefit` blended without refit | 4.70% | 4.51 to 5.02 | 4.05% |
| raw network (refit) | 3.19% | 2.93 to 3.69 | 3.26% |
| `tlrn_cf_factors` (network off) | 5.64% | 5.53 to 5.74 | 5.01% |
| `mack` | 5.48% | one value | 4.90% |
| `mcl` | 5.41% | one value | 4.93% |

**Tolerances.** `mack` and `mcl` are deterministic: a rerun should match to the printed digits
(5.481 and 5.410 at date 10; 4.896 and 4.926 rolling). A new set of GPU fits is not bit for bit
repeatable and the draws are random, so the three network rows move. One draw's date-10 Pool_APE
has a standard deviation of 0.18 points (refit), 0.16 (no refit), 0.29 (raw network) and 0.07
(network off) in this run, so the mean of ten draws should land within about 0.2 points of the
table for the refit and no-refit rows, 0.3 for the raw network and 0.1 for the network-off row.
The rolling means move more (draw sd 0.34 for the refit): allow 0.35. These are rules of thumb
from one set of ten draws, not a fitted interval. A single draw can sit anywhere in the range
(the refit's ranged 3.73 to 4.26).

**The verdicts to expect on B3**, from the same rules: P1 (the refit helps) not distinguishable,
because the pooled interval includes zero although 10 of 10 draws favour the refit; P2 (better
than `mack` and `mcl`) not distinguishable; the network adds value; the blend not
distinguishable. The companion's published 3.93% (blended) and 2.96% (raw network) are
replicated in the evaluator's sense (`REPLICATES`: ours 4.02, range 3.73 to 4.26; raw network
3.29 mean over draws).

## 5. Check the benchmarks

The benchmark rows are fitted by the evaluator with ibnr's gallery entries `mack` (per
company-line pair) and `mcl` (per company), on exactly each date's company set. Its output has
three checks, all in `evaluation/report.txt`:

* `mack` against an independent chain ladder written in plain numpy: the largest relative
  difference over the four dates is 2.2e-16 (B3) in the saved run, 2.2e-16 to 4.4e-16 in the
  rerun.
* The date-10 benchmarks against the second study's saved values
  (`analysis/results/tlrn_study2/evaluation`, the default of `--study2-evaluation`): `mack`
  0.0548098 and `mcl` 0.0540981 on B3, relative differences of 9e-16 and 1e-15.
* The companion's published figures: his chain ladder is 5.48% and his chain ladder with the
  multivariate chain ladder is 5.41% at date 10, which are ibnr's `mack` and `mcl` figures
  above. `mcl` has no independent recomputation in the evaluator.

## 6. Cheap checks

**Network-off rows only.** The `tlrn_cf_factors` fits took about 31 s each on the GPU. They need
the same install and runtime as the main row, so they are the quick way to test a setup before
spending 9 minutes a fit. They cannot reproduce the headline (the network is off); they should
land at 5.5 to 5.8% on B3 at date 10 (all ten draws did, 5.53 to 5.74). Not run on a CPU.

**The smoke check** (notebook section 7): ten epochs, two members, one draw. Its numbers must
never be quoted.

**Recompute a table from the saved evaluator output, no GPU, no mart.** From the repository
root, with the saved `B_company_predictions.csv` (the draw-average forecast of every row for
every company and date, and the actual reserve). This was run and prints the Pool_APE by row
and date in percent (for example `tlrn_cf_refit` 4.63, 3.72, 3.07 and 4.02 at dates 6, 7, 8
and 10):

```python
import pandas as pd

table = pd.read_csv("analysis/results/tlrn_study3/evaluation/B_company_predictions.csv")
rows = []
for valuation, group in table.groupby("valuation"):
    actual = group["actual"]
    for method in group.columns[3:]:
        error = (group[method] - actual).abs().sum() / actual.sum()
        rows.append((valuation, method, error))
pool_ape = pd.DataFrame(rows, columns=["valuation", "method", "pool_ape"])
print(pool_ape.pivot(index="method", columns="valuation", values="pool_ape").mul(100).round(2))
```

**Recompute the refit forecast from the saved fits.** This needs the ten
`B_tlrn_cf_refit_dNN-ibnr0.7.6_valuations.csv` files (the Colab output; they are not in the
repository) and gives the draw average that `B_company_predictions.csv` holds. It was run on
the study's fits and the largest difference from the saved column was 3e-11:

```python
import pathlib

import pandas as pd

folder = pathlib.Path("study3")  # the folder of saved fits
saved = pd.read_csv(
    "analysis/results/tlrn_study3/evaluation/B_company_predictions.csv",
    dtype={"company_code": str},
)
saved = saved[saved["valuation"] == 10].set_index("company_code")

draws = []
for path in sorted(folder.glob("B_tlrn_cf_refit_d*-ibnr0.7.6_valuations.csv")):
    table = pd.read_csv(path, dtype={"company_code": str})
    draws.append(table[table["valuation"] == 10].set_index("company_code")["predicted"])
mean = sum(draws) / len(draws)
print(float((mean - saved["tlrn_cf_refit"]).abs().max()))
print(float((mean - saved["actual"]).abs().sum() / saved["actual"].sum()))
```

In `_valuations.csv` the columns `predicted`, `predicted_selected`, `predicted_net` and
`predicted_net_selected` are the refit blend, the same members' blend without the refit, the
raw network with the refit and the raw network without it. The `actual` column in those files is
not the reserve outcome at date 10 (it sums to zero there), so use the
evaluator's table for the actuals, as above.

## 7. What each file is

In the Colab output folder, one set per fit, named `B_tlrn_cf_refit_d03-ibnr0.7.6` (cohort, row,
draw, ibnr version):

| file | what |
|---|---|
| `.json` | settings and run record (device, seconds, versions, the data hash and the hash of each date's pairs); written last, so it marks the fit as finished |
| `_reserves.csv` | the company reserve of the fit: anchor, premium, number of lines, the blended reserve and each member's |
| `_valuations.csv` | one row per date and company: cells scored, the four forecasts above and the actual |
| `_members.csv` | one row per trained member: validation errors, blend weight alpha, the best epoch, epochs run (3,000) and the epoch the refit ran to |
| `.pkl` | the fitted result object (about 75 KB each); the evaluator does not read it |
| `manifest_companion_final.json` | the runner's list of fits with their settings |

The evaluator writes (and the repository keeps, see
[the folder's README](../analysis/results/tlrn_study3/README.md)): `report.txt` (everything it
prints), `evaluator_log.txt`, `verdicts.json`, `benchmarks3_*.csv` (the `mack` and `mcl` cell
forecasts, reusable as `--cache`) and one table per cohort and question (`B_*`, `C_*`).
