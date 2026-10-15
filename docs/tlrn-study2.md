# Testing the companion's final model on a Colab GPU

The companion study's newest model is not the accident-year model that the
[first study](tlrn-study.md) tested. Its repository moved, on 2026-10-08, to a model that
reads a log development factor per line and step through a learned starting point, with
attention across lines and across lags only and no accident-year attention. This second
study trains that model many times with ibnr (`backend="jax"`) and asks one question: is
it more accurate than the chain ladder (`mack`) and the multivariate chain ladder (`mcl`)?
The rules were fixed before any fit and before any evaluator code, in
[the decision note](../knowledge/decisions/tlrn-factor-head-study.md), and the evaluator's
module docstring repeats them. Read them before the results, and do not change a rule after
seeing them.

**Nothing in this study has been run on a GPU yet, and no result exists.** The runner and
the evaluator were checked on a laptop CPU with tiny budgets only (two members, ten epochs),
so any number from those smoke runs must never be quoted.

## What is run

| piece | what |
|---|---|
| rows | `tlrn_13_fh` (the model: 13 features, factor head, blended with `mcl`, 20 members averaged) and `tlrn_13_fh_factors` (the same with the network switched off) |
| raw network | read from the main fits' `predicted_net` column; no extra fit |
| draws | 10 independent fits per row and cohort, seeds `11 + 100003 k` |
| fit | 20 members at each of 4 valuation dates, 3,000 epochs, `backend="jax"` |
| cohort A | notebook 04's: 93 companies, 243 pairs, 82 scored, valued at 2007-12-31 |
| cohort B | the companion's: 84 companies, 212 pairs, inside A, valued at 2007-12-31 |
| cohort C | new: accident years 1988 to 1997, valued at 1997-12-31, the same lines and selection rule, 83 companies, 232 pairs |
| valuation dates | 6, 7, 8 and 10 (the companion's indices: calendar diagonal 10 is the last known one) |

That is 10 x 2 x 3 = 60 fits (40 on A and B alone). **The verdicts on A and B are labelled
confirmatory of a fixed specification, not out-of-sample**: the companion chose its design
and its hyperparameters with their test data in view. Only cohort C is a test the
specification was not tuned against, and it shares most of its companies with A and B.

## Running on Colab

The fits run in Colab; the scoring runs on your laptop. The Colab kit is a folder of three
files (`tlrn_colab.py`, `tlrn_study_colab.py` and a notebook, `ibnr_study2_colab.ipynb`)
with its own plain `README.txt`; it installs the published ibnr 0.7.4 from PyPI, so no
wheel and no GitHub access are needed. Every runner call is in-process and carries
`--study factor_head`; without it the first study runs instead.

```python
import tlrn_study_colab

tlrn_study_colab.main(["--study", "factor_head", "--out", OUT, "--smoke", "--draws", "1"])
tlrn_study_colab.main(["--study", "factor_head", "--out", OUT, "--cohort", "A", "--draws", "10"])
```

Each fit saves, under a shared stem such as `A_tlrn_13_fh_d03-ibnr0.7.4`: the settings
record `.json` (written last, marks the fit finished), `_reserves.csv`, `_valuations.csv`
(one row per valuation date and company: cells scored, the blended forecast, the raw
network and the actual), `_backtest.csv`, `_members.csv` (each member's blend weight) and the
fitted entry `.pkl`. Everything saved is skipped on a rerun, a changed setting stops the
run before any training, and a disconnect loses at most the fit that was running.

The earlier accident-year model took about 250 seconds a fit on a Blackwell GPU (Colab G4),
460 on an A100 and 3,200 on a T4. This model has no accident-year attention, so those are
estimates, not measurements: plan on about 4 hours for all 60 fits on a G4 or an A100.

## Scoring on the laptop

Copy the folder back, for example to `analysis/results/tlrn_study2/fits`, and run the
evaluator from a checkout. It needs `uv sync` and the cas-schedule-p data, and neither jax
nor torch.

```text
uv run python scripts/tlrn_study2_eval.py --fits analysis/results/tlrn_study2/fits \
    --out analysis/results/tlrn_study2/report
uv run python scripts/tlrn_study2_eval.py --fits analysis/results/tlrn_study2/fits \
    --cohort B --companion-rolling path/to/the/companion/R_T1_rolling_origin.csv \
    --tie-out-first-study --out analysis/results/tlrn_study2/report
```

For each valuation date the evaluator fits `mack` (per company-line pair) and `mcl` (per
company) with ibnr's gallery entries on the cells known at that date and reads their
forecasts cell by cell, so the benchmarks are scored on exactly the cells the transformer
is scored on. The fits are kept in a cache beside the results, and a rerun while more draws
arrive is quick (about 45 seconds a cohort the first time). Before it scores anything it
checks that the saved tables agree with its own cell sums: the number of cells per company
and valuation, the known actuals, and the valuation-10 forecast against the stored reserve.

Three tie-outs back the benchmarks. With `--companion-rolling` (the companion's rolling
table, passed as a path; it is another person's private data and is never copied here) the
evaluator checks his cell and company counts and shows `mack` and `mcl` beside his chain
ladder and chain ladder + MCL at each valuation. With `--tie-out-first-study` the
valuation-10 benchmarks are compared company by company with the first study's evaluator.
On cohort A the valuation-10 Pool_APE of `mack` and `mcl` must reproduce notebook 04's to
1e-6, or the evaluator stops.

It prints, for each cohort, Pool_APE per draw and for the draw average at every valuation
and rolling, the pre-registered differences from `mack` and `mcl` with 95% bootstrap
intervals, the dominant company left out, the secondary and descriptive metrics, size
quartiles, the ablations, and one verdict, and it writes the tables and `verdicts.json`
to `--out`. Add `--smoke` to score a smoke folder; its output is marked and must never
be quoted.

Reading rules. The verdict is **provisional** unless a cohort holds exactly draws 0 to 9 of
the main row: it is printed as `effective (provisional: 3 of 10 draws ...)`, never as a bare
word. A company a benchmark refuses at any valuation is left out of every row and named in
the output. The word `effective` needs four things at once (a valuation-10 interval and a
rolling interval wholly below zero against both benchmarks, a median-draw Pool_APE below
both, and no interval wholly above zero once the dominant company is left out); the
decision note has the exact rule. MAPE and the reserve-weighted RMSE are shown as
descriptive only.

## Notebook 05

`analysis/05_tlrn_factor_head_study.ipynb` loads the saved fits from a folder (the path is
a variable at the top, with `TLRN_STUDY2_FITS` as an override), runs the evaluator's
functions and shows the verdict tables and a few plain figures: the spread of the draws at
each valuation, the differences from the benchmarks with their intervals, the results with
and without the dominant company, and the ablations. Every sentence about results is
written from the computed values or phrased conditionally, and it says nothing about
results before they exist. The committed notebook has no outputs: run it after the Colab
fits are in the folder.

## What this cannot show

The companion's design was tuned on A and B, so only C is a test the specification could fail
for a reason it was not tuned against, and C overlaps A and B in companies. The bootstrap
resamples companies, so it measures company-to-company sampling and says nothing about
other years beyond what C adds. Ten draws sample training randomness thinly. The model blends
the multivariate chain ladder into every member (the blend weight is fitted), so it is
partly the benchmark. GPU fits are not bit-for-bit repeatable, so a rerun gives different
draws.
