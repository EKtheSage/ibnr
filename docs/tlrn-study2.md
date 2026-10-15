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
seeing them. The note carries a dated **Amendment 1** (2026-10-15, made before any fit
existed) that adds a second question, fixes the run order and states what the primary verdict
can detect; it changes none of the original rules.

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
and its hyperparameters with their test data in view. Cohort C is out-of-sample in accident
years only. As far as we know the companion did not tune on it (assumed, not known). It is
not independent in calendar time (its outcomes sit in statement years 1998 to 2006, which
overlap A and B and the companion's valuations 6, 7 and 8) nor in companies (67 of its 83
companies are in A, 60 in B, and 155 of its 232 pairs are B's pairs), so it does not test a
new economic environment or other companies. Every cohort is selected on future cells
existing (a complete 10-by-10 square, outcome cells included), a survivorship shared by the
model and the benchmarks.

**Run cohort C first, then A, then B.** If the primary verdict on C is `not distinguishable`
or `worse`, no further GPU time is needed to decide the main question: A and B are
confirmation runs on data the specification was tuned on, with a precision too low to show a
half-point gain. Whether to run them is Ethan's decision.

**What the primary verdict can detect.** In the first study the paired 95% intervals of the
Pool_APE difference on A and B were 5.4 to 11.0 points wide (9.3 to 11.0 against `mack`, 5.4
to 7.8 against `mcl`), so a gain has to be about three points or more to clear both benchmarks. A gain of about
half a point cannot be detected on a single cohort, and `not distinguishable` on A or B is
therefore the expected result, not news.

**A second question, beside the verdict.** Does the network add anything beyond shrinking
toward the multivariate chain ladder? The evaluator compares the main row with the
factors-only row (same cohort, companies and draws, both blended) by the same paired
bootstrap of Pool_APE, model minus factors-only: `network adds value` only if the
valuation-10 and the mean-over-dates intervals are both wholly below zero and the
valuation-10 interval without the dominant company is not wholly above zero; `network hurts`
if either of the first two is wholly above zero; `not distinguishable` otherwise. It is
printed after the verdict, written to `verdicts.json` (`network_question`) and
`<cohort>_network_check.csv`, and never changes the verdict.

## Running on Colab

The fits run in Colab; the scoring runs on your laptop. The Colab kit is a folder of three
files (`tlrn_colab.py`, `tlrn_study_colab.py` and a notebook, `ibnr_study2_colab.ipynb`)
with its own plain `README.txt`; it installs the published ibnr 0.7.4 from PyPI, so no
wheel and no GitHub access are needed. Every runner call is in-process and carries
`--study factor_head`; without it the first study runs instead.

```python
import tlrn_study_colab

tlrn_study_colab.main(["--study", "factor_head", "--out", OUT, "--smoke", "--draws", "1"])
tlrn_study_colab.main(["--study", "factor_head", "--out", OUT, "--cohort", "C", "--draws", "10"])
# then the same with "A", then "B", if the owner decides to run them
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

Download the `study2` folder from Drive (`My Drive/ibnr_colab_fh/study2`) to
`C:\Temp\study2`, and run the evaluator from a checkout. This is the one folder name used
everywhere: the kit's README, notebook 05 (`TLRN_STUDY2_FITS` defaults to it) and the commands
below. It needs `uv sync` and the cas-schedule-p data, and neither jax nor torch.

```text
uv run python scripts/tlrn_study2_eval.py --fits C:\Temp\study2 --out C:\Temp\study2_report
uv run python scripts/tlrn_study2_eval.py --fits C:\Temp\study2 --cohort B ^
    --companion-rolling path\to\the\companion\R_T1_rolling_origin.csv ^
    --tie-out-first-study --out C:\Temp\study2_report
```

(The `^` continues a line in `cmd`; in PowerShell use a backtick, in a POSIX shell a
backslash.)

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
quartiles, the ablations, one verdict and, separately, the paired network question, and it
writes the tables and `verdicts.json` to `--out`. Add `--smoke` to score a smoke folder; its output is marked and must never
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

`analysis/05_tlrn_factor_head_study.ipynb` loads the saved fits from `C:\Temp\study2` (the
path is a variable at the top, with `TLRN_STUDY2_FITS` as an override), cohort C first, runs the evaluator's
functions and shows the verdict tables and a few plain figures: the spread of the draws at
each valuation, the differences from the benchmarks with their intervals, the results with
and without the dominant company, and the ablations. Every sentence about results is
written from the computed values or phrased conditionally, and it says nothing about
results before they exist. The committed notebook has no outputs: run it after the Colab
fits are in the folder.

## What this cannot show

The companion's design was tuned on A and B. Cohort C is independent in accident years only
(not tuned on as far as we know, assumed, not known): it overlaps A and B in calendar time
and in companies, so it tests neither a new economic environment nor other companies. Every
cohort is selected on future cells existing, a survivorship shared by the benchmarks. On a
single cohort the primary verdict can detect a gain of about three points or more of Pool_APE and
not a half-point gain. The bootstrap resamples companies, so it measures company-to-company
sampling and says nothing about other years beyond what C adds. Ten draws sample training randomness thinly. The model blends
the multivariate chain ladder into every member (the blend weight is fitted), so it is
partly the benchmark. GPU fits are not bit-for-bit repeatable, so a rerun gives different
draws.
