Results of the fourth tlrn study (see knowledge/findings/tlrn-study4-results.md, the rules in
knowledge/decisions/tlrn-study4-preregistration.md and the notebook analysis/06_tlrn_study4_on_colab.ipynb).

evaluation/           What scripts/tlrn_study4_eval.py wrote from the 44 fits made on 2026-11-25
                      (cohorts B4 and C4; rows tlrn_cr_refit, 10 draws per cohort, and on B4 at
                      draw 0 the single-draw rows tlrn_cr_noline, tlrn_cr_noay, tlrn_cr_paid and
                      tlrn_cf_rerun; ibnr 0.7.7, jax 0.11.1, Python 3.13.16, pandas 2.2.3,
                      numpy 2.4.6, duckdb 1.3.2, cas-schedule-p publish 20260613_041006). The
                      numbers are as the evaluator printed them; nothing was edited. The 20
                      tlrn_cr_factors rows (network switched off) are the third study's
                      tlrn_cf_factors fits, taken from its folder and not refitted, as the
                      pre-registration allows.
  report.txt          everything the evaluator writes to its report, both cohorts
  evaluator_log.txt   the evaluator's whole console output: the progress lines (the "data hash
                      differs" notes, the seconds the benchmarks took) and, at the end, the
                      VERDICTS and OVERALL lines
  verdicts.json       the verdicts, conditions and intervals for P1, P2, the network and the
                      blend question, and the replication of the companion's figures
  benchmarks3_B_*.csv, benchmarks3_C_*.csv
                      the forecasts of mack and mcl, cell by cell, at the four dates; the
                      file name carries the cohort, its data hash and the hash of all the dates'
                      pairs (the same two files as the third study's, by name). Pass the folder
                      as --cache to skip fitting them again.
  B_*, C_*            one table per cohort and question:
    metrics_by_valuation       the point metrics and the share of companies better than
                               mack and mcl, by arm and date (6, 7, 8, 10, rolling, pooled)
    company_predictions        the draw-average forecast of every arm for every company and
                               date, and the actual reserve (USD thousands)
    differences                Pool_APE differences with 95% company-bootstrap intervals
    precision                  the width of each interval and the gain it could just detect
    size_quartiles             Pool_APE within quartiles of the actual reserve
    draw_pool_ape_refit_norefit  each draw's Pool_APE with and without the refit
    alpha_spread               the spread of the blend weight over members and draws
    p0_per_draw                P0, descriptive: the fourth study's draws against the third
                               study's, draw by draw
    companion_beside_ours      the companion's published figures beside the ten draws
                               (descriptive, not a decision rule)
    single_draw_rows           B only: the four single-draw rows at draw 0 (descriptive; no
                               verdict)

What ran. Colab, 2026-11-25, 44 of 44 fits saved, none failed, 6.67 of the 8 GPU hours
budgeted. The cells of notebook 06 were taken byte for byte from the tag v0.7.7 (notebook
sha256 e1b7d2d10d4a648be7b9764696dc8da347aff952edd7113850133000039e6950) and run in the
notebook's own kernel through a small driver cell that ran the long cells on a background
thread. The runner is scripts/tlrn_study_colab.py.

Devices. Cohort C draw 0 of tlrn_cr_refit ran on an NVIDIA L4 and took 1711.2 seconds (28.5
minutes), over the pre-registered 25 minute stop. Ethan was asked and answered on 2026-11-25:
"switch to a faster GPU and go on (draw 0 took 28.5 minutes on an L4)". The answer is recorded
in every later fit (--answered-first-fit). All the other fits ran on an NVIDIA A100-SXM4-80GB:
about 950 seconds per main fit on C and about 1015 seconds on B. The 8 hour budget stop was
not reached. The third study's factors rows (reused) ran on an NVIDIA RTX PRO 6000 Blackwell
Server Edition with ibnr 0.7.6.

The fits themselves (settings and run records, company reserve tables, valuation forecasts,
member tables and the fitted .pkl results) are not kept here. They are in the owner's Google
Drive folder ibnr_colab_cf/study4 (the third study's, which the reused rows came from, is
ibnr_colab_cf/study3).

To rescore, from a checkout at the v0.7.7 code and with the fits of both studies on disk:

  uv run python scripts/tlrn_study4_eval.py --fits <study4 fits folder> --study3-fits <study3 fits folder> --tie-out analysis/results/tlrn_study3/evaluation --cache <cache folder> --out <new folder>

(--cache wants a folder the evaluator may write to, so copy the two benchmarks3_*.csv files of
this folder into a scratch folder first. The run recorded here used that form, with the
scratch folder holding those two files; the evaluator exited with code 0.)

The fits were made on Colab, where the cell hash text differs from the one the evaluator builds
on a laptop (the log shows "data hash differs"); see "Data hash" in the finding.
