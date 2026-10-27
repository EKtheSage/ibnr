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

**The study was run on 2026-10-26 on a Colab GPU, and its results are in
[the finding](../knowledge/findings/tlrn-study3-results.md).** The saved evaluator output is in
`analysis/results/tlrn_study3/`. To repeat the study or to check one table, follow
[the reproduction guide](tlrn-study3-reproduce.md). Smoke runs (two members, ten epochs) were
used to check the runner and the evaluator only, and their numbers must never be quoted.

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

## What a cohort, a draw and a member are

* **A cohort** is a set of (company, line) pairs from the Schedule P mart publish
  `20260613_041006`. **B3** is accident years 1998 to 2007, valued at the end of 2007 (the
  companion's own). **C3** is accident years 1988 to 1997, valued at the end of 1997.
* **Which pairs.** A pair is in if it has a complete 10 by 10 square with positive premium on
  every cell; if, on the cells known at the valuation date, paid is positive and cumulative
  paid over premium is below 3; and if its company has at least two such lines.
* **Four dates.** The valuation dates are 6, 7, 8 and 10 (calendar diagonals). The companies are
  picked again at each date, from the cells known then, so each date trains and is scored on
  its own set, and the sets are nested.
* **A member** is one network trained from one seed. **A draw** is one complete fit: 20 members
  at each of the four dates, averaged after each is blended with the multivariate chain
  ladder. That is 80 networks per draw. There are 10 draws per cohort, so 800 networks per
  cohort and 1,600 in all in the main row (the network-off row is extra).
* **Seeds.** Draw `k` has the fit seed `11 + 100003 k`. Date `v` adds `20000 v` and member `m`
  adds `1000 m`, so every network has its own stream.
* **The final number** for a company is the mean of the 10 draws' forecasts. The error is
  scored on that draw average, and the bootstrap resamples companies, not draws.

How the pieces nest (seeds in brackets, counts for one row):

```text
study (companion_final, ibnr 0.7.6)
 +-- cohort  B3 (AY 1998-2007)        or      C3 (AY 1988-1997)
      +-- row  tlrn_cf_refit  (the model)   or   tlrn_cf_factors  (network off)
           +-- draw k = 0..9   [fit seed 11 + 100003 k]            10 draws
                +-- valuation date v = 6, 7, 8, 10                  4 dates
                |     [seed + 20000 v]   its own company set S_v
                |     (nested: S_10 inside S_8 inside S_7 inside S_6)
                     +-- member m = 0..19  [seed + 20000 v + 1000 m]  20 networks
                           trained to its best epoch, then refit
                           to that epoch on everything known at v
```

One draw is therefore 80 networks per row, a cohort's main row is 800, and the two cohorts
together 1,600. The company reserve of a draw is the mean of its 20 members' blended forecasts;
the reported number is the mean of the 10 draws.

## The cohorts in detail

Both cohorts are read from the Schedule P mart of cas-schedule-p publish `20260613_041006`
(table `mart_reserving_model_training`), four lines only: `commercial_auto`,
`other_liability`, `private_passenger_auto` and `workers_compensation`. Each (company, line)
pair has a 10 by 10 square of cumulative paid loss (accident year by development year), a net
earned premium per accident year and, in the cell tables, incurred loss, bulk loss and case
reserve. Calendar index `c` of a cell is `accident year - first accident year + development
age`, so the diagonal `c` is the end of calendar year `first accident year + c - 1`.

| | B3 | C3 |
|---|---|---|
| accident years | 1998 to 2007 | 1988 to 1997 |
| mart rows read | 62,010 | 67,500 |
| valuation dates 6, 7, 8, 10 (year end) | 2003, 2004, 2005, 2007 | 1993, 1994, 1995, 1997 |
| last known diagonal (date 10) | 2007-12-31 | 1997-12-31 |
| pairs at dates 6, 7, 8, 10 | 230, 226, 219, 212 | 237, 236, 232, 232 |
| companies at dates 6, 7, 8, 10 | 90, 89, 87, 84 | 84, 84, 83, 83 |
| pairs at date 10 by line (commercial auto, other liability, private passenger auto, workers' compensation) | 74, 62, 59, 17 | 70, 64, 58, 40 |
| cells scored per pair at dates 6, 7, 8, 10 | 14, 15, 13, 45 | 14, 15, 13, 45 |
| cells scored at date 10 | 9,540 (212 pairs x 45) | 10,440 (232 x 45) |
| selection funnel at date 10 (groups, complete squares, usable paid and premium, two lines) | 668, 414, 297, 212 | 675, 405, 322, 232 |
| company 1767's share of the actual date-10 reserve | 67.3% | 66.7% |
| status of any verdict | confirmatory of a fixed specification, not out of sample | out of sample in accident years only |

**How pairs are chosen** (`select_pairs` in `scripts/tlrn_study_colab.py`, the companion's rule
with the cutoff as a parameter). At valuation date `v` a (company, line) pair is in if (1) it
has a complete 10 by 10 square, every cell with a finite positive net earned premium, the cells
after `v` included; (2) on the cells with calendar index at most `v`, cumulative paid loss and
premium are finite and positive and the paid loss ratio is below 3 on every cell; and (3) its
company has at least two such lines. The rule is applied again at each date to the cells known
then, so a company can enter at an early date and is then in every later set. The scoring unit
is the company.

**Overlap.** At date 10, 155 of C3's 232 pairs are also B3 pairs (recomputed for this note with
`select_pairs`: B3 has 212 pairs, C3 232, and 60 companies appear in both). Company 1767, which
holds about two thirds of the reserve, is in both. The accident years do not overlap, but a
B3 network at date 10 is trained on cells of calendar years 1998 to 2007, and C3's companies
mostly report in those years too, so the two cohorts are far from independent samples of
companies.

**Why B3 is not out of sample, and C3 only partly.** B3 is the companion's own cohort: the
model's architecture, hyperparameters and the 3,000-epoch budget were chosen by the companion
with this cohort and its date-10 score in view, so a good B3 score confirms that a fixed
specification behaves as reported and says nothing about unseen data. C3 uses different
accident years and so is new in that respect only: it shares most of its companies with B3,
the model was not tuned on it, but the second study's results on the same cohort were known
before this model's rules were written, so it is weaker than a fresh cohort would be.

What this design does and does not do. C3 overlaps B3: at date 10, 155 of C3's 232 pairs are
also B3 pairs (measured in the second study), so C3 is out of sample in accident years only. The
companion reports one fit with one seed, and ibnr's random streams cannot reproduce his, so the
study asks whether the model he built is plausible against a spread of ibnr's own fits. It does
not reproduce his numbers, and the section of the report that sets his published figures beside
ours is descriptive. His own bootstrap is 90% on the date-averaged error.

## Running on Colab

The notebook is `analysis/05_tlrn_study3_on_colab.ipynb`. Open it in Colab and set the runtime
first: Runtime, Change runtime type, a GPU (a G4 or an A100; a T4 is about 13 times slower than
the G4 and not a reasonable target). Then run the cells from the top, one at a time.

0. **One-time setup, only while the repository is private.** While it is public the notebook
   downloads the two runner scripts without a token and you can skip this step. If the
   repository is private again, the notebook stops with a message that says so, and needs a
   GitHub token that can only read. ibnr itself comes from PyPI and needs no token. A token in
   the Colab secret is used when it exists. Create the token at <https://github.com/settings/personal-access-tokens/new>: expiry
   30 days, "Only select repositories" and pick `ibnr`, Repository permissions, Contents,
   Read-only. In Colab open the key icon (Secrets), add a secret named `GITHUB_TOKEN` with the
   token and switch "Notebook access" on. Delete the token on GitHub when the study is done. The
   notebook never prints the token and never writes it to Drive.
1. **Settings.** Leave them alone, except `SOURCE` and `REF`. The Colab runtime installs ibnr
   0.7.6. With the default `SOURCE = "pypi"` it installs `ibnr[jax]==0.7.6` from PyPI (public,
   no token). `SOURCE = "git"` is only for an unreleased branch: it installs the branch `REF`
   (default `tlrn-study3`) from the private repository with the token. Either way the next
   cell stops if the installed version is not 0.7.6.
2. **Install and fetch the scripts.** The notebook installs ibnr in two steps (so pip cannot
   compile anything), downloads `tlrn_colab.py` and `tlrn_study_colab.py` through the GitHub
   contents API at the release tag `v0.7.6` (`SCRIPTS_REF`; the earlier releases are tagged
   `v0.7.5` and so on; to test an unreleased change set `SCRIPTS_BRANCH` to a branch name),
   saves each only if it parses as Python, and checks that the runner pins the version the
   notebook expects. A refused token (HTTP 401 or 403) is retried once without it; with no
   token, a 401, 403 or 404 says the repository may be private. On Python 3.13 (what Colab
   runs now) the install adds `--ignore-requires-python` and prints why, because ibnr declares
   Python below 3.13 for extras this notebook does not use.
3. **Mount Drive.** Fits are saved to `My Drive/ibnr_colab_cf/study3`. A disconnect loses at most
   the fit that was running; run the same cell again and saved fits are skipped. A saved fit
   made with other settings stops the run before any training and is left alone.
4. **Checks that fit nothing.** The notebook selects the companies again at each valuation date
   and compares the counts with the pre-registration (230, 226, 219 and 212 pairs on B3; 237,
   236, 232 and 232 on C3), and builds the network once to count its parameters (18,537
   expected, the companion's own count). If anything differs, stop and send the printout to Ethan.
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

**How long.** The pre-registered planning figure was 400 to 500 seconds per main fit and 3 to 6
GPU hours for everything on a Blackwell-class card (Colab G4); an A100 is about twice as slow.
Measured on 2026-10-26 on an NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab's G4): a main
fit took 501 to 535 seconds on B3 (median 521) and 461 to 480 on C3 (median 478), a
network-off fit 31 to 33 seconds, and all 40 fits 2.94 GPU hours. Every fit's record holds
the seconds it took, and the notebook prints them after each piece.

Each fit saves, under a shared stem such as `C_tlrn_cf_refit_d03-ibnr0.7.6`: the settings record
`.json` (written last, marks the fit finished), `_reserves.csv`, `_valuations.csv` (one row per
valuation date and company: cells scored, the refit blend, the same members before the refit, the
raw network with and without the refit, and the actual), `_members.csv` (each member's blend
weight, chosen epoch and the epoch the refit ran to) and the fitted result `.pkl`.

## Getting the results back

Nothing is zipped or downloaded. Every fit is saved in the Drive folder
`My Drive/ibnr_colab_cf/study3`, which is flat on purpose: the evaluator reads that folder and
does not look into subfolders. Make the folder available on the laptop with Drive for desktop
(it shows up as a drive such as `G:\My Drive`) or with `rclone`. The last cell of the notebook
prints the number of finished fits, the GPU hours, any fit missing a table, and the exact
command for the laptop, built from the notebook's own `DRIVE_FOLDER` and `LAPTOP_DRIVE_ROOT`
settings.

## Scoring on the laptop

The evaluator needs `uv sync` and the cas-schedule-p data, and neither jax nor torch. The
[reproduction guide](tlrn-study3-reproduce.md) has the steps in order, the numbers to expect and
how to check the benchmarks.

```text
uv run python scripts/tlrn_study3_eval.py --fits "G:\My Drive\ibnr_colab_cf\study3" --out C:\Temp\study3_report
uv run python scripts/tlrn_study3_eval.py --fits "G:\My Drive\ibnr_colab_cf\study3" --cohort C --out C:\Temp\study3_report
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
and the refit gain), the spread of the blend weight, RMSE, SMAPE and the number of
over-estimated companies beside the other point metrics, a table of the companion's published
Table 10 beside ibnr's ten draws (descriptive, not a decision rule), the precision of every interval and the
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
