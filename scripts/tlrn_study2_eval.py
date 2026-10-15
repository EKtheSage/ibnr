"""The second tlrn study's evaluator: does the factor-head model beat the benchmarks?

Reads the fits that ``scripts/tlrn_study_colab.py --study factor_head`` saved and scores
them at four valuation dates against the chain ladder (``mack``) and the multivariate chain
ladder (``mcl``), both fitted by ibnr's own entries on exactly the data known at each date
and scored on exactly the cells the transformer is scored on. It needs no jax and no
accelerator, only ibnr, the cas-schedule-p package and the saved folder. The design and its
reasons are in ``knowledge/decisions/tlrn-factor-head-study.md``, which this docstring
repeats; the first study's evaluator is ``scripts/tlrn_study_eval.py`` and its statistics
are reused here.

PRE-REGISTERED, written before any evaluator code and before any real fit existed
=================================================================================

The question. Is the companion study's newest model (13 features, line and lag attention,
a factor head, blended with the multivariate chain ladder, 20 members averaged) more
accurate than ``mack`` and ``mcl``? The model is the row ``tlrn_13_fh``. Its comparison
rows are ``tlrn_13_fh_factors`` (the same model with the network switched off, so the head's
pooled factors and the alpha blend only) and the raw network (``predicted_net`` of the main
fits, the forecast before the blend, which needs no extra fit). The first study's
accident-year rows are compared when their fits are supplied.

Draws and cohorts. Each row has 10 independent fits (draws 0 to 9, fit seed
``11 + 100003 k``) on each cohort. Cohort A is notebook 04's (93 companies, 243 pairs, 82
scored), cohort B the companion's (84 companies, inside A), cohort C new (accident years
1988 to 1997, valued at 1997-12-31, real later development from the mart). A draw set that
is not exactly draws 0 to 9 is labelled provisional.

What is scored. Four valuation dates, called 6, 7, 8 and 10 as the companion calls them:
a valuation ``v`` is the calendar diagonal ``v`` of the 10-by-10 square (10 is the last
known one, 2007-12-31 on A and B, 1997-12-31 on C). At ``v`` the cells scored are those
on diagonals ``v + 1`` to 10 at development lag ``v`` or less, for origins that have at
least one elapsed development period; at ``v = 10`` they are every cell past diagonal 10
(45 per pair). A company's predicted and actual values at ``v`` are the sums of the
incremental paid over those cells, over its lines, so at ``v = 10`` the actual is the
cohort's realised reserve. Actuals at ``v < 10`` come from the known upper triangle.
Cell counts per company are checked against the saved tables' ``n_cells`` column, and the
saved tables' predictions and actuals against each other (the saved ``actual`` against the
evaluator's own, which is built from the cells) before any score is computed. The scored
companies are the cohort's board (those ``mack`` and ``mcl`` can fit in notebook 04's
rule) for which both benchmarks answered at all four valuations; a company a benchmark
refuses at any valuation is left out of every valuation and every row and named in the
output, so every comparison is on one set of companies.

Statistics. A company's error at ``v`` is predicted minus actual. **Pool_APE** is the sum of
absolute errors over the sum of actuals (``ibnr.kernels.point_metrics``'s ``pool_ape``).
The **draw average** at ``v`` is the mean over draws of each company's prediction. The
**rolling Pool_APE** is the plain mean of Pool_APE over the four valuations. The
**dominant company** is the company with the largest actual reserve at valuation 10, found
in code. The **bootstrap** resamples the scored companies 10,000 times (seed 2008), takes
the same resampled companies for the model and the benchmark and for every valuation (a
company's four valuation dates stay together), and reports the 2.5th and 97.5th percentiles
of the difference model minus benchmark. A resample whose actuals add to zero has no
Pool_APE and is left out.

The verdict, for the main row ``tlrn_13_fh`` only, on each cohort, from the draw average.

* ``effective`` only if ALL hold. (i) At valuation 10 the Pool_APE difference (model minus
  benchmark) has a 95% interval wholly below zero against both ``mack`` and ``mcl``.
  (ii) The mean over valuations 6, 7, 8 and 10 of the Pool_APE difference has an interval
  wholly below zero against both. (iii) The median over the draws of the valuation-10
  Pool_APE is below both benchmarks' valuation-10 Pool_APE. (iv) With the dominant company
  left out, the valuation-10 difference interval is not wholly above zero against either
  benchmark.
* ``worse`` if the interval in (i) or (ii) is wholly above zero against either benchmark.
* ``not distinguishable`` otherwise.

"Wholly below zero" means the upper end is below zero; "wholly above" means the lower end is
above zero. No multiplicity correction is made over three cohorts. A and B are not
independent (B lies inside A), and C shares most of its companies with both.

What a verdict means. The companion chose its design and hyperparameters with the A and B
test data in view, so the verdict on A or B is labelled **confirmatory of a fixed
specification, not out-of-sample**. The verdict on C is the only out-of-sample claim, with
two caveats beside it: C is the older accident-year window, and it shares most companies
with A and B. Nothing here shows what happens in other years.

Reported, not in the verdict (fixed in advance). Per valuation and for the rolling mean:
Pool_APE and the Pool_APE of each draw (min, median, max, sd); Pool_PE (bias: the sum of
errors over the sum of actuals); the median absolute percentage error over companies with a
nonzero actual (MedAPE); the mean absolute error (MAE); the mean absolute log of predicted
over actual reserve over companies where both are positive (the count is shown); the share
of companies where the model's absolute error is smaller than each benchmark's; Pool_APE
within quartiles of the actual reserve (quartiles of the valuation's own actuals, equal
counts, the dominant company in the top one); and the valuation-10 difference with the
dominant company left out, with its interval. Descriptive only, because the companion's
claimed gains are largest on them and they are driven by tiny reserves and by one company:
the mean absolute percentage error (MAPE) and the reserve-weighted root mean squared error
(wRMSE, the square root of the actual-weighted mean squared error, the companion's
definition). Ablations are reported as the difference of each comparison row's Pool_APE
from the main row's, with the same bootstrap, outside the verdict.

Tie-outs. The benchmarks at valuation 10 must equal the first study's benchmark reserves
(``tlrn_study_eval.benchmark_reserves``, run when ``--tie-out-first-study`` is given) and,
on cohort A, notebook 04's published Pool_APE to 1e-6. With ``--companion-rolling`` the
companion's published rolling table is read through its path: its cells and companies
columns must equal this evaluator's, and its chain ladder and chain ladder + multivariate
chain ladder Pool_APE at each valuation are printed beside ``mack`` and ``mcl``. That
comparison is a tie-out of the benchmarks, and a difference is reported, not hidden: his
chain ladder is his own implementation.

Usage::

    python scripts/tlrn_study2_eval.py --fits DIR [--cohort A|B|C|all] [--out DIR]
        [--cache DIR] [--cas-dir DIR] [--companion-rolling FILE]
        [--first-study-fits DIR] [--tie-out-first-study] [--smoke]

A smoke folder (``--smoke``) is scored too, so the evaluator can be checked, and its output
carries a banner saying the numbers must never be quoted. Nothing has run on a GPU when this
file is written, and no result exists.
"""

from __future__ import annotations
