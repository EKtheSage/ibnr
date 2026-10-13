# Testing the accident-year model on a Colab GPU

The companion study's accident-year model (`TLRNConfig.accident_year_variant()`) is one
neural network trained once, and one training run is one draw of its randomness. This
study trains it ten times on each of two cohorts and compares the whole spread of
outcomes with the chain ladder (`mack`), the multivariate chain ladder (`mcl`) and `sur`,
so that "effective" can be answered with more than one lucky or unlucky run. The
questions and the verdict rule were fixed before any fit was made, in
[the decision note](../knowledge/decisions/tlrn-robustness-study.md). Read it before the
results, and do not change a rule after seeing them.

**Nothing in this study has been run on a GPU yet.** The runner and the evaluator were
checked on a laptop CPU with tiny budgets only (two members, ten epochs; 52 to 122 seconds
per fit, almost all of it compilation), so there are no results to quote.

## What is run

| piece | what |
|---|---|
| rows | `tlrn_8_ay` (paid only) and `tlrn_13_ay` (paid, incurred and case reserve) |
| draws | 10 independent fits per row and cohort, seeds `11 + 100003 k` |
| fit | 20 members at each of 4 valuation dates, 3,000 epochs, `backend="jax"` |
| cohort A | notebook 04's: 93 companies, 243 pairs, 82 scored |
| cohort B | the companion study's: 84 companies, 212 pairs, all scored (inside A) |
| valuation | 2007-12-31, accident years 1998 to 2007 |

That is 10 x 2 x 2 = 40 fits. A Colab GPU took about 7.5 minutes per fit in the earlier
run, so the whole study is about 5 hours. It runs in pieces and resumes, as described
below.

## Cohort B and the data

Cohort B is the companion study's selection rule applied to its four cached CAS Schedule P
files (`ppauto`, `wkcomp`, `comauto`, `othliab`), with the calendar-index cutoff at the
last known diagonal. **The same cohort can be built from the cas-schedule-p package.** For
these four lines and accident years 1998 to 2007 the mart holds 62,010 cells, and every
one of them (paid, incurred, bulk, premium, and the case reserve derived from them) equals
the cell in the CAS files. So the runner builds cohort B from the package by default, with
the selection counts (668, 414, 297, 212 pairs; 84 companies) and the cohort hash
asserted. `--cas-dir` builds it from the CAS files instead and asserts the same, which
is the check that the two routes agree. You do not have to upload the CAS files.

## What to upload

The repository is private, so Colab cannot fetch it. Put these in one Drive folder, for
example `MyDrive/ibnr/study`:

* the ibnr wheel, built on your machine from a checkout of `main`
  (the PyPI release does not carry the JAX backend yet):

  ```text
  uv build --wheel
  ```

  which writes `dist/ibnr-0.7.3-py3-none-any.whl`;
* `scripts/tlrn_colab.py` and `scripts/tlrn_study_colab.py` (the second imports the first);
* optionally the four CAS files in a subfolder `cas`, if you want `--cas-dir`.

The evaluator (`scripts/tlrn_study_eval.py`) runs on your machine afterwards and needs no
upload.

## The Colab cells

Set the runtime first: Runtime, Change runtime type, GPU. Run these in order, each in its
own cell. Run the runner inside the notebook, not with `!python`: a TPU can be opened by
one process only, and the first cell already holds it.

1. Check the accelerator. You want `gpu` (or `tpu`); `cpu` means the runtime type was not
   changed.

   ```python
   import jax

   print(jax.__version__, jax.default_backend(), jax.devices())
   ```

2. Mount Drive and name the folders.

   ```python
   from google.colab import drive

   drive.mount("/content/drive")
   STUDY = "/content/drive/MyDrive/ibnr/study"
   OUT = "/content/drive/MyDrive/ibnr/study_fits"
   ```

3. Install ibnr from the wheel, in two steps so that pip cannot compile anything (Colab runs
   Python 3.13 and ibnr declares 3.11 to 3.12, hence `--ignore-requires-python`):

   ```text
   !pip install -q --no-deps --ignore-requires-python {STUDY}/ibnr-0.7.3-py3-none-any.whl
   !pip install -q --only-binary=:all: --ignore-requires-python "{STUDY}/ibnr-0.7.3-py3-none-any.whl[jax]" cas-schedule-p==2026.6.13
   ```

4. Check the versions, then make the scripts importable.

   ```python
   import sys

   import duckdb
   import ibis

   import ibnr

   print("ibnr", ibnr.__version__, "| ibis", ibis.__version__, "| duckdb", duckdb.__version__)
   assert ibis.__version__.startswith("12."), "pip chose an old ibis"
   sys.path.insert(0, STUDY)
   ```

5. Smoke check: two draws, two members, ten epochs, into its own folder. It builds both
   cohorts, asserts their counts and hashes, and trains 8 tiny fits. Stop here if it
   fails.

   ```python
   import tlrn_study_colab

   tlrn_study_colab.main(["--out", OUT + "_smoke", "--smoke", "--draws", "2"])
   ```

6. The study. Everything saved is skipped, so this cell is also the "resume" cell.

   ```python
   tlrn_study_colab.main(["--out", OUT])
   ```

   Draws go in order (draw 0 for every cohort and row, then draw 1, ...), so a run stopped
   at any point holds an even set. To run in pieces, ask for fewer draws or one cohort or
   row, and later run the full cell again:

   ```python
   tlrn_study_colab.main(["--out", OUT, "--draws", "3"])  # draws 0 to 2, both cohorts
   tlrn_study_colab.main(["--out", OUT, "--cohort", "B", "--rows", "tlrn_13_ay"])
   tlrn_study_colab.main(["--out", OUT, "--cas-dir", STUDY + "/cas", "--cohort", "B"])
   ```

   Progress prints every hundred epochs (`--quiet` stops that), and after each fit the
   runner prints that draw's Pool_APE on the scored companies.

7. See what is saved. Read `device_kind` (the GPU model) before quoting a time.

   ```python
   import json
   from pathlib import Path

   for record in sorted(Path(OUT).glob("*-ibnr*.json")):
       info = json.loads(record.read_text(encoding="utf-8"))
       print(record.name, info["run"]["fit_seconds"], info["run"]["device_kind"])
   ```

## What is saved

For each fit, three files that share a stem such as `A_tlrn_8_ay_d03-ibnr0.7.3`:

* `.json`, the settings record (configuration, members, epochs, seed, cohort hash, data
  hash) and what it ran on (device, jax version, seconds). It is written last, and it is
  what marks a fit as finished.
* `_reserves.csv`, one row per company: the anchor, the premium, the number of lines, the
  ensemble reserve and each member's reserve. This is all the evaluator reads.
* `.pkl`, the fitted entry (skipped by `--no-entries`, which saves Drive space; the
  evaluator does not need it).

`manifest_study.json` lists the records in the folder. **A disconnect loses at most the
fit that was running**: run step 6 again. A saved fit is skipped only when its settings
match this run's exactly; a different `--members`, for example, stops the run before any
training and names what differs. Give such a run its own `--out` folder. A smoke run and a
full run never mix (the smoke files carry `_smoke` in their names).

## Evaluating, on your machine

Copy the folder back and run the evaluator from a checkout. It needs `uv sync` and the
cas-schedule-p data; it needs neither jax nor torch.

```text
uv run python scripts/tlrn_study_eval.py --fits study_fits --out study_results
uv run python scripts/tlrn_study_eval.py --fits study_fits --cohort B --cas-dir cas \
    --companion-reserves path/to/the/companion/M_T0_company_reserves.csv
```

The evaluator fits the benchmarks itself (about 90 seconds per cohort on a 16-core laptop,
kept in a cache beside the results so a rerun is quick), checks that
`mack` and `mcl` reproduce notebook 04's Pool_APE on cohort A, and, when given the
companion's published company reserves, that they reproduce its chain ladder and
multivariate chain ladder reserves on cohort B. It prints the six answers for every cohort
and row and one verdict each, and writes the tables and `verdicts.json` to `--out`.
Add `--smoke` to score a smoke folder; its output is marked and must never be quoted.

Two reading rules. First, the verdict is **provisional** unless the folder holds exactly
draws 0 to 9 for that cohort and row: with fewer draws, or a gap, the evaluator still
applies the pre-registered rule but prints (and writes to `verdicts.json`) the word with
`provisional: N of 10 draws` and the missing draw numbers, never a bare `effective`.
Second, in the variance decomposition read `draw_noise_share` and
`draw_noise_share_relative` first (the part of one draw's squared error that averaging
draws removes). `icc_log_reserve` is a company-size effect: reserves differ by orders of
magnitude between companies, so it sits near 1 by construction and does not measure draw
noise. A cohort with no saved fits is skipped before its benchmarks are fitted.

## Expected cost

| step | time |
|---|---|
| install and smoke check | about 10 minutes |
| one fit at the full budget | about 7.5 minutes on a Colab GPU (the earlier run: 434 to 451 s) |
| the full study, 40 fits | about 5 hours |
| the evaluator, on a laptop | about 3 minutes the first time, seconds after (the benchmark fits are cached) |
| disk | the reserves files are tiny; each fitted entry is about 1.4 to 2.7 MB at 2 members, so expect tens of MB at 20 (not measured), and `--no-entries` skips them |
