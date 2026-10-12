---
type: Measurement
title: tlrn on a Colab TPU
description: The first TPU run of the JAX backend, 40 members of tlrn_8 and tlrn_13 at the published budget, against the same fits in notebook 04 on the dev laptop's CPU.
tags: [tlrn, jax, tpu, performance, colab]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-12T12:00:00Z }
stale_after: 2027-01-10T00:00:00Z
sources:
  - id: colab
    resource: the Colab run of analysis/04_tlrn_fits_on_colab.ipynb pasted into the 2026-10-12 working session (ibnr 0.7.3 from main, jax 0.7.2, Python 3.13, one TPU device)
    title: Colab TPU output
  - id: laptop
    resource: the cache timestamps and tlrn_members.csv of the notebook 04 run on 0.7.3, 2026-10-07 to 2026-10-08 (dev laptop, CPU, torch)
    title: Notebook 04 run on the laptop
  - id: decision
    resource: /decisions/tlrn-jax-training-backend.md
    title: The decision that built the backend
---

# Result

On one Colab TPU device the JAX backend trained **40 members for 3,000 epochs in 618 s**
for each of `tlrn_8` and `tlrn_13`. The same two fits took **8 h 20 min** and **5 h 35 min**
in notebook 04 on the laptop's CPU with torch, so the TPU was about **48 times** and **32
times** faster.[^colab][^laptop] The first 100 epochs took 52 s (compilation included) and
each later 100 took about 19 s, which is about 207 member-epochs per second.

# Are the members the same kind of member?

Close, not identical. Both runs train 40 members per row, with different random streams
(dropout masks differ, as the decision note says), so only the spread can be compared.

| Row | Backend | Median ay-line APE | Range | Median company APE |
|---|---|---|---|---|
| tlrn_8 | torch, laptop | 0.0840 | 0.0813 to 0.0905 | 0.0460 |
| tlrn_8 | JAX, TPU | 0.0854 | 0.0809 to 0.0909 | 0.0462 |
| tlrn_13 | torch, laptop | 0.0796 | 0.0725 to 0.0861 | 0.0459 |
| tlrn_13 | JAX, TPU | 0.0801 | 0.0743 to 0.0864 | 0.0467 |

The medians differ by 0.0014 (`tlrn_8`) and 0.0005 (`tlrn_13`) on member spreads of about
0.002, which is a small gap but not a tested equality. No statistical test was run, and the
two runs used the same fit seed with different random streams.

# The accident-year variant

The 80-member accident-year fit (20 members at each of four valuation dates) ran at about
0.6 s per epoch, so a full 3,000 epochs should take about 30 minutes. The run reached epoch
1,700 (1,085 s) and the output stopped there with no error. Why is unknown at the time of
writing: a Colab disconnect is the usual cause, and a fit that is cut off must start over,
because the checkpoint is one saved file per finished fit.

# Lessons from getting it to run

* A TPU can be opened by one process only. The notebook's first cell holds it, so the
  runner must be called inside the notebook (`tlrn_colab.main([...])`), not started with
  `!python`, which fails with "The TPU is already in use by process with pid ...".
* Colab's Python is 3.13 and ibnr declares 3.11 to 3.12, so the install needs
  `--ignore-requires-python`. The cap comes from the `bayesian` and `interop` extras.
* The repository had to be public for Colab to fetch it.

[^colab]: Colab TPU output
[^laptop]: Notebook 04 run on the laptop
