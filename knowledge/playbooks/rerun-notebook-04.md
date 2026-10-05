---
type: Playbook
title: Re-run notebook 04
description: How long each stage of analysis/04 takes, and how to run it in pieces so a failure costs minutes and not hours.
tags: [notebook, tlrn, runtime]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:45:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: nb04
    resource: analysis/04_nn_architectures_vs_classical.ipynb
    title: The notebook, with the saved outputs of the 0.7.1 run
  - id: session
    resource: the 2026-10-05 session's smoke run of the edited notebook (67 minutes)
    title: Smoke run timings
---

# How long it takes

From the saved outputs of the 0.7.1 published run (618 minutes in all):[^nb04]

| stage | time |
|---|---|
| `next_diagonal` for 243 pairs | 240 s |
| `mack`, 243 pair fits | 424 s |
| `mcl` and `sur` | 144 s each |
| `nn_transformer_ml` (fit 1,468 s plus forecasting 5,496 s) | 6,963 s |
| the other five pooled neural entries | 400 to 990 s each |
| `tlrn_8` (40 members) | 12,111 s |
| `tlrn_13` (40 members) | 13,224 s |

The accident-year variant adds a fit of 20 members at four valuation dates, 240,000
member-epochs, estimated at 4.4 to 6.7 hours on this laptop (see [tlrn CPU training
throughput](/findings/tlrn-cpu-training-throughput.md)).

A smoke run (`IBNR_NB04_PROTOCOL=smoke`) still takes about 67 minutes, because the neural
entries' forecasting and the classical fits do not shrink with the training budget.[^session]

# How to run it

1. **Do not use `jupyter nbconvert --execute` for a first run.** It prints nothing until the
   end, and a stalled cell is indistinguishable from a slow one.
2. Convert the code cells to one script that prints a line before each cell, and put the
   whole body under `if __name__ == "__main__":` (see [spawned workers re-run the
   script](/gotchas/spawn-workers-rerun-the-script.md)). Use a timing alias a notebook
   will not rebind.
3. Run it detached, with output to a log file. A tool-managed background command stops
   after two hours at most, far short of the full run.
4. Run the smoke protocol first and let it finish; it exercises every cell that the
   published run will.
5. For the published run install the exact pinned wheel (`ibnr[nn]==<version>`), since the
   notebook asserts the version.
6. Set `torch.set_num_threads(1)` and `processes` to the core count for the variant's
   fit; the notebook's other tlrn fits use four processes of four threads.

[^nb04]: The notebook, with the saved outputs of the 0.7.1 run
[^session]: Smoke run timings
