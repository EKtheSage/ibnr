Results of the second tlrn study (see knowledge/findings/tlrn-factor-head-study-results.md and
the rules in knowledge/decisions/tlrn-factor-head-study.md).

fits/                 For each of the 60 fits (cohorts A, B and C, rows tlrn_13_fh and
                      tlrn_13_fh_factors, draws 0 to 9, ibnr 0.7.4, jax 0.11.1, cas-schedule-p
                      publish 20260613_041006, all on one NVIDIA RTX PRO 6000 Blackwell GPU):
                        .json          settings and run record (device, seconds, versions)
                        _reserves.csv  the company reserves of that fit
                        _valuations.csv, _backtest.csv, _members.csv
                                       forecasts by valuation date, the backtest cells, and
                                       one row per trained member (blend weight, best epoch)
                      manifest_factor_head.json is the runner's list of the fits.
evaluation/           What scripts/tlrn_study2_eval.py wrote from those fits: verdicts.json,
                      report.txt, the benchmark reserves (benchmarks2_*.csv), and one table per
                      cohort and question (A_*, B_*, C_*). evaluator_log.txt is the full text it
                      printed. B_companion_rolling_tie_out.csv and B_members_fidelity.csv put the
                      companion study's published figures beside ours (report only).

The model files (.pkl, about 425 MB in all) are not kept here. They are in the owner's Google Drive.

To rerun the evaluator from the repository root (the three runs of the benchmarks take a few
minutes):

  uv run --extra nn --extra jax python scripts/tlrn_study2_eval.py --fits analysis/results/tlrn_study2/fits --out <new folder> --cohort all --first-study-fits analysis/results/tlrn_study/fits

Add --companion-rolling and --companion-members, each pointing to the companion study's published
table of that name, to repeat the cohort B tie-out; those tables are not in this repository.
The fits were made on Colab, where the cell hash text differs from the one the evaluator builds
on a laptop (the log shows "data hash differs"). The evaluator accepts such fits only with the
change in pull request 177 (it accepts fits whose hash text differs when the anchors, premiums,
lines and companies agree). Without it the run stops at the hash check.
