---
type: Finding
title: ibnr's tlrn matches the R reference
description: "ibnr's tlrn reproduces the R reference's inference (loading R's checkpoints) and its training loop; notebook 04's gap to the published result is entirely which pair of members was kept, and the published 10% gain over CL+MCL is a favourable draw."
tags: [tlrn, reproduction, ensemble, nn, companion-study]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note tlrn-port-vs-r-reference.md (private, outside the repository)
    title: tlrn port vs R reference
    last_modified: 2026-09-25T08:31:01.592Z
---

# The question

Established 2026-09-23 while diagnosing why notebook 04's `tlrn_13` (raw 6.77%, blend
5.77% company Pool_APE) missed the reference result (5.63% / 5.09%). The reference here is
an R implementation whose ten saved checkpoints sit in
`Exhibits/models` in the companion study's repository; for that repository see [The companion study's
repository](/references/companion-study-repository.md).[^note]

# Result

**Same distribution; the gap is entirely which pair was kept.** ibnr's members and R's
members are draws from the same distribution of results. The published 10% gain over
CL+MCL (the blend of chain ladder and the multivariate chain ladder) is a favourable
draw: the typical gain is about 3%, and averaging all ten members gives about 4%.[^note]

# Inference is identical (proven)

The reference's ten checkpoints (`Exhibits/models/tlrn_seed_*.pt` in the companion study's repository,
safetensors from R torch 0.17) were exported with R and loaded into ibnr 0.7.0 by
replacing `ibnr.gallery.nn.tlrn.model.train_ensemble`. ibnr reproduced all ten seeds'
validation and test Pool_APE to about 1e-6, kept the same pair (31415, 42), and gave a raw
ensemble of 5.625621% against R's 5.625620%.[^note]

R torch embeddings are 1-based, so R's line, lag and visible-lag tables load after
`np.roll(w, 1, axis=0)`. Ethan ruled on 2026-09-23 **not** to align ibnr to R's 1-based
rows ("ibnr is python").[^note]

# The training loop matches too

The agent first claimed it did not, and was wrong. The R file has **two** trainers.
`train_tlrn` (8-company batches, fixed full-cutoff denominators) is called by nothing in
the pipeline. The saved seeds come from `fit_checkpoint_seed` -> `train_checkpoint_tlrn`:
64-example batches, per-batch denominators, a loss of AY/line APE + 0.5 PE +
0.1 MSE/0.005, Adam with learning rate .03 for phi and .003 for the rest, a 20-step
warmup then cosine schedule, gradient clip 1, one training cutoff per epoch drawn
uniformly over 2..7, and patience of 600 checks (never fires). That is the same as ibnr's
port.[^note]

What misled: the checkpoint file says loss_normalization "full-origin denominators", but
`save_point_fit` writes that string as a constant whatever trained the model. **Check
which function a saved artifact came from before trusting its metadata.**[^note]

Two tiny remaining differences, not fixed, effect probably negligible: ibnr's threshold
for a new best is 1e-6 (shared `_training.py`), R's is 1e-12; and a batch with no scored
cells is skipped by ibnr while R still takes an Adam step with zero gradient.[^note]

# All 45 pairs of R's own ten models

Scored through ibnr's code. The blend via `ibnr.kernels.shrink_toward(raw, mcl, 0.658)`
reproduces R's 5.0880% exactly.[^note]

* Raw 4.99-7.57% (median 6.29%); blend 4.70-6.29% (median 5.50%).
* R's chosen pair 31415+42 beats 82% (raw) / 84% (blend) of its own pairs. Notebook 04's
  `tlrn_13` (6.77% / 5.77%) would beat only 20% / 24%: inside R's range, in its worse
  quarter.
* 15 of 45 of R's own pairs blend **worse** than CL+MCL (5.66%). The median pair beats it
  by about 3%, not the reported 10%.
* Ethan had read T1 ("held-out AY/line APE") as the test score. It is the **validation**
  score, on which ibnr's members match (7.73-8.36% against R's 7.68-8.60%).

# Settled by retraining ibnr's ten members (2026-09-23)

ibnr's ten notebook-04 members were retrained in four parallel processes (seed
11 + 1000*m, 4 threads; 88 minutes). Validation scores match notebook 04 to 5e-7, and the
kept pair 1+5 reproduces 6.7703% / 5.7685% exactly. Scored like R's:[^note]

| | R | ibnr |
|---|---|---|
| single members, mean (range) | 6.47% (5.08-8.14) | 6.50% (5.08-8.21) |
| pairs, raw median | 6.29% | 6.22% |
| pairs, blended median | 5.50% | 5.44% |
| average of all ten, raw / blend | 6.20% / 5.44% | 6.23% / 5.42% |

30 of 45 pairs beat CL+MCL in both. ibnr's member 4 (test 5.08%) lost its place in the
kept pair to member 5 (7.26%) by 0.00006 validation APE; pair 1+4 would have scored
5.70% / 5.06%, which is R's published result.[^note]

Before the retrain, the evidence was weaker: R's ten seeds span 5.08-8.14% test Pool_APE
(mean 6.47%, sd 0.88), its kept pair includes its best test seed (42, 5.08%), and the
validation score barely predicts the test score (Spearman about 0.4).[^note]

# Pool study (20 members: R's 10 and ibnr's 10)

An ensemble's reserve is **exactly** the mean of its members' reserves (`_ensemble_pred`
averages pred, and reserves are a sum of pred*mask*premium), so any group can be scored
from per-member company reserves.[^note]

Average of N members, median raw / blend:

| N | raw | blend |
|---|---|---|
| 1 | 6.48% | 5.53% |
| 2 | 6.25% | 5.47% |
| 5 | 6.21% | 5.44% |
| 10 | 6.21% | 5.43% |
| 20 | 6.22% | 5.42% |

More members buy steadiness, not a better typical result. The blend beats CL+MCL 94% of
the time at N=10 and always at N>=15; raw never beats plain chain ladder (5.77%).[^note]

The keep-two rule on 2000 random 10-member runs: raw 6.12% (10th to 90th percentile
5.63-6.77), blend 5.41% (5.06-5.77), beats MCL 82% of the time. Averaging all ten: 6.21%
(5.91-6.52), 5.43% (5.24-5.63), 94%. R's published run and notebook 04's run are exactly
the 10th and 90th percentiles of keep-two.[^note]

Averaging all members already works through `TLRNConfig(keep=ensemble_size)`. Ethan
(2026-09-23) wanted notebook 04 to show the keep-two spread across versions and
average-all against CL/MCL. The plan proposed: per-member reserves kept on the fitted
entry, parallel member training in the library, 40-member pools for `tlrn_8` / `tlrn_13`,
and dropping `tlrn_13_early`.[^note]

# Where the working files went

Everything was done in `C:/Temp/tlrncheck` (the R export script
`export_r_weights.R` run with R 4.5.3 and an R torch library under
the reconciliation outputs folder's `R-library`, `transplant.py` and its venv, `pairs.py`,
`train_member.py`, `score_ours.py`, `pool_study.py`, and `out/*.csv`). That folder was
moved to the Recycle Bin on 2026-09-25 at Ethan's request to clean up the temp folders,
and is gone once the bin is emptied. The findings stand on their own; to redo any of it,
rebuild the scripts from the descriptions here.[^note]

Related: [Reproducing the companion notebook](/findings/companion-notebook-reproduction.md)
and [tlrn CPU training throughput](/findings/tlrn-cpu-training-throughput.md).

[^note]: tlrn port vs R reference
