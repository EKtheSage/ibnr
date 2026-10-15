Results of the tlrn robustness study (see knowledge/findings/tlrn-robustness-study-results.md).

fits/                 One .json (settings and run record: device, seconds, versions) and one
                      _reserves.csv (the company reserves of that fit) for each of the 40
                      fits: cohorts A and B, rows tlrn_8_ay and tlrn_13_ay, draws 0 to 9,
                      ibnr 0.7.4, cas-schedule-p publish 20260613_041006.
fits/manifest_study.json  The runner's list of the fits.
evaluation/           What scripts/tlrn_study_eval.py wrote from those fits: verdicts.json,
                      report.txt, the benchmark reserves, and one table per question, cohort
                      and row. evaluator_log.txt is the full text it printed.

The model files (.pkl, about 270 MB in all) are not kept here. They are in the owner's Google Drive.

To rerun the evaluator from the repository root:

  uv run python scripts/tlrn_study_eval.py --fits analysis/results/tlrn_study/fits --out <new folder> --cohort both

Add --companion-reserves <the companion study's company reserves table> to repeat the cohort B
tie-out; that table is not in this repository. The fits were made on Colab, where the cell
hash text differs from the one the evaluator builds. The evaluator only accepts such fits
with the change in pull request 177 (accepts fits whose hash text differs when the anchors,
premiums, lines and companies agree). Without it the run stops at the hash check.
