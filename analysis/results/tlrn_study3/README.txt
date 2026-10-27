Results of the third tlrn study (see knowledge/findings/tlrn-study3-results.md, the rules in
knowledge/decisions/tlrn-study3-preregistration.md and the steps in docs/tlrn-study3-reproduce.md).

evaluation/           What scripts/tlrn_study3_eval.py wrote from the 40 fits made on 2026-10-26
                      (cohorts B3 and C3, rows tlrn_cf_refit and tlrn_cf_factors, draws 0 to 9,
                      ibnr 0.7.6, jax 0.11.1, Python 3.13.16, cas-schedule-p publish
                      20260613_041006, one NVIDIA RTX PRO 6000 Blackwell Server Edition). The
                      numbers are as the evaluator printed them; nothing was edited.
  report.txt          everything the evaluator prints, both cohorts
  evaluator_log.txt   the same text with the evaluator's progress lines (the "data hash
                      differs" notes, the seconds the benchmarks took)
  verdicts.json       the verdicts, conditions and intervals for P1, P2, the network and the
                      blend question, and the replication of the companion's figures
  benchmarks3_B_*.csv, benchmarks3_C_*.csv
                      the forecasts of mack and mcl, cell by cell, at the four dates; the
                      file name carries the cohort, its data hash and the hash of all the dates'
                      pairs. Pass the folder as --cache to skip fitting them again.
  B_*, C_*            one table per cohort and question:
    metrics_by_valuation       the seven point metrics and the share of companies better than
                               mack and mcl, by arm and date (6, 7, 8, 10, rolling, pooled)
    company_predictions        the draw-average forecast of every arm for every company and
                               date, and the actual reserve (USD thousands)
    differences                Pool_APE differences with 95% company-bootstrap intervals
    precision                  the width of each interval and the gain it could just detect
    size_quartiles             Pool_APE within quartiles of the actual reserve
    draw_pool_ape_refit_norefit  each draw's Pool_APE with and without the refit
    alpha_spread               the spread of the blend weight over members and draws
    companion_beside_ours      B only: the companion's published figures beside the ten draws
                               (descriptive, not a decision rule)

The fits themselves (the 40 settings and run records, company reserve tables, valuation
forecasts, member tables and the fitted .pkl results, 3.3 MB without the .pkl files and 3.0 MB of them) are not
kept here. They are in the owner's Google Drive folder ibnr_colab_cf/study3.

To rescore from the repository root, with the fits in <fits folder>:

  uv run python scripts/tlrn_study3_eval.py --fits <fits folder> --out <new folder> --cache analysis/results/tlrn_study3/evaluation

(Not run in exactly that form: --cache wants a folder it may write to, so copy the two
benchmarks3_*.csv files into a scratch folder first. A rescore with the cache of this folder
gave the same verdicts and the same Pool_APE to rounding.)

The fits were made on Colab, where the cell hash text differs from the one the evaluator builds
on a laptop (the log shows "data hash differs"); see "Data hash" in the finding.
