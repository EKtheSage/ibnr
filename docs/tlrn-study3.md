# Testing the companion's latest model on a Colab GPU

The [second study](tlrn-study2.md) tested the companion's model as it stood on 2026-10-08
(a factor head, no accident-year attention, no refit). The companion's repository has since moved
to a different model, and this third study tests that one. It trains the model many times with
ibnr (`backend="jax"`) and asks two questions:

* **P1, the refit.** Each network is first trained to the epoch where it does best on the newest
  known diagonal. The companion then trains it again on everything known up to the cutoff and
  stops at that same epoch (the "refit"). Does the refit lower the error of the company
  reserves? One training run gives both forecasts, with and without the refit, so the comparison
  is paired.
* **P2, effectiveness.** Is the model with the refit more accurate than the chain ladder (`mack`)
  and the multivariate chain ladder (`mcl`)? This is the second study's question and rule, asked
  again of the new model.

The rules, the stopping rules and the thresholds were fixed before any fit and any evaluator code,
in [the decision note](../knowledge/decisions/tlrn-study3-preregistration.md), and the
evaluator's module docstring repeats them. Read them before the results, and do not change a
rule after seeing a result. A rule that turns out to be wrong is changed by a dated amendment at
the end of the note, written before the fit it would affect.

**Nothing in this study has been run on a GPU yet, and no result exists.** The runner and the
evaluator were checked on a laptop CPU with tiny budgets only (two members, ten epochs), so any
number from those smoke runs must never be quoted.

## What is run

| piece | what |
|---|---|
| rows | `tlrn_cf_refit` (the model; one run gives the refit forecast, the forecast without the refit from the same members, and the raw network) and `tlrn_cf_factors` (the same model with the network switched off) |
| draws | 10 independent fits per row and cohort, seeds `11 + 100003 k` |
| fit | 20 members at each of 4 valuation dates, 3,000 epochs, `backend="jax"` |
| cohort C3 | the fresh cohort: accident years 1988 to 1997, valued at 1997-12-31 |
| cohort B3 | the companion's: accident years 1998 to 2007, valued at 2007-12-31 |
| valuation dates | 6, 7, 8 and 10 (calendar diagonal 10 is the last known one) |
| companies | chosen again at each valuation date, so each date trains and is scored on its own set |

That is 10 x 2 x 2 = 40 fits. The verdict on B3 is **confirmatory of a fixed specification, not
out of sample**: the companion chose this design with that cohort in view. C3 is out of sample in
accident years only, and weaker than the second study's C, because that study's results existed
before this model's rules were written. C3 overlaps B3 in calendar time and in most companies.

## Running on Colab

The notebook is `analysis/05_tlrn_study3_on_colab.ipynb`. Open it in Colab and set the runtime
first: Runtime, Change runtime type, a GPU (a G4 or an A100; a T4 is about 13 times slower than
the G4 and not a reasonable target). Then run the cells from the top, one at a time.

1. **Settings.** Leave them alone, except `SOURCE` and `REF`. The Colab runtime installs ibnr
   0.7.6. That release may not be on PyPI yet: with `SOURCE = "git"` the notebook installs the
   branch `REF` (default `tlrn-study3`) from GitHub, and with `"pypi"` it installs
   `ibnr[jax]==0.7.6`. Either way the next cell stops if the installed version is not 0.7.6.
2. **Install and fetch the scripts.** The notebook installs ibnr in two steps (so pip cannot
   compile anything), fetches `tlrn_colab.py` and `tlrn_study_colab.py` from the same ref and
   checks that the runner pins the version the notebook expects.
3. **Mount Drive.** Fits are saved to `My Drive/ibnr_colab_cf/study3`. A disconnect loses at most
   the fit that was running; run the same cell again and saved fits are skipped. A saved fit
   made with other settings stops the run before any training and is left alone.
4. **Checks that fit nothing.** The notebook selects the companies again at each valuation date
   and compares the counts with the pre-registration (230, 226, 219 and 212 pairs on B3; 237,
   236, 232 and 232 on C3), and builds the network once to count its parameters (18,601
   expected). If anything differs, stop and send the printout to Ethan.
5. **Smoke check.** Ten epochs, two members, one draw, in its own folder. If it fails, stop.
6. **C3, draw 0.** One full main fit. It is the pipeline check and the first timing.
7. **C3 draws 1 to 9, then B3 draws 0 to 9, then the network-off fits** on both cohorts, in that
   order. The notebook runs one draw at a time and checks the budget before each.

The runner is called inside the notebook's own Python process (not with `!python`), because an
accelerator can be opened by only one process. If the runner stops with an `AssertionError`
saying a cohort does not match its published selection (counts or hash), it has stopped the whole
call: it does not skip that cohort. Do not edit the check. Send the message, because the mart or
the code has changed.

**Stopping rules (pre-registered).** Stop and ask Ethan if draw 0 of the main row on C3 takes
longer than 25 minutes, or if the saved fits add up to 8 GPU hours before the study is finished.
The helper cells raise `SystemExit` with a message that begins `STOP` in both cases. A member
with a non-finite loss stops its fit; rerun that fit once, and if it happens again report it.

**How long.** The pre-registered planning figure, never measured, is 400 to 500 seconds per main
fit and 3 to 6 GPU hours for everything on a Blackwell-class card (Colab G4); an A100 is about
twice as slow. Every fit's record holds the seconds it took, and the notebook prints them after
each piece. **Send those numbers back with the results.**

Each fit saves, under a shared stem such as `C_tlrn_cf_refit_d03-ibnr0.7.6`: the settings record
`.json` (written last, marks the fit finished), `_reserves.csv`, `_valuations.csv` (one row per
valuation date and company: cells scored, the refit blend, the same members before the refit, the
raw network with and without the refit, and the actual), `_members.csv` (each member's blend
weight, chosen epoch and the epoch the refit ran to) and the fitted result `.pkl`.

## Getting the results back

The last cell of the notebook writes `study3_results.zip` (records, tables, manifest and the
timings; the `.pkl` files are left out) to `My Drive/ibnr_colab_cf/download`. Download it and unzip
it to `C:\Temp\study3` on the laptop.

## Scoring on the laptop

The evaluator needs `uv sync` and the cas-schedule-p data, and neither jax nor torch:

```text
uv run python scripts/tlrn_study3_eval.py --fits C:\Temp\study3 --out C:\Temp\study3_report
uv run python scripts/tlrn_study3_eval.py --fits C:\Temp\study3 --cohort C --out C:\Temp\study3_report
```

For each valuation date the evaluator fits `mack` (per company-line pair) and `mcl` (per company)
with ibnr's gallery entries on the cells known at that date, for exactly that date's companies,
and reads their forecasts cell by cell, so the benchmarks are scored on exactly the cells the
network is scored on. Before it scores anything it checks that the saved tables agree with its
own cell sums, that every fit was trained on the companies it selects at each date, that every
member ran 3,000 epochs and that every refit ran to the epoch its member chose. It ties the
valuation-10 benchmarks to the second study's saved values, recomputes `mack` at every date with
a separate chain ladder written in plain numpy, and checks that the dominant company is in every
date's set.

It prints, for each cohort, Pool_APE per draw and for the draw average, the P1 and P2 verdicts
with their conditions and 95% intervals, the same rules on a stratified bootstrap, the network
and blend questions, the replication of the companion's reported figures (3.93% at valuation 10
and the refit gain), the spread of the blend weight, the precision of every interval and the
overall statements across the two cohorts. It writes the tables and `verdicts.json` to `--out`.
Add `--smoke` to score a smoke folder; its output is marked and must never be quoted.

Reading rules. A verdict is **provisional** unless a cohort holds exactly draws 0 to 9 of every
row it uses, and is then printed as `helps (provisional: 3 of 10 draws ...)`, never as a bare
word. A company a benchmark cannot answer for at any date it belongs to is left out of every date
and named in the output.

## What this cannot show

The companion's design was chosen with B3 in view. C3 is independent in accident years only: it
overlaps B3 in calendar time and in companies, so it tests neither a new economic environment nor
other companies. Every cohort is selected on future cells existing, a survivorship shared by the
benchmarks. The bootstrap resamples companies, so it measures company-to-company sampling and says
nothing about other years. One company holds about two thirds of the reserve at valuation 10, so
every dollar-based statistic depends on it. Ten draws sample training randomness thinly, and GPU
fits are not bit-for-bit repeatable, so a rerun gives different draws. The refit stops at an epoch
that was chosen with the same cells in view that the refit then learns from; a gain from the
refit is an upper bound on what an independent choice would give.
