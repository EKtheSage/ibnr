---
type: Measurement
title: tlrn on a Colab TPU
description: The first TPU run of the JAX backend, 40 members of tlrn_8 and tlrn_13 at the published budget, against the same fits in notebook 04 on the dev laptop's CPU, plus the accident-year fits (80 members per row) on a Colab GPU.
tags: [tlrn, jax, tpu, performance, colab]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-12T12:00:00Z }
stale_after: 2027-01-10T00:00:00Z
sources:
  - id: colab
    resource: the Colab run of analysis/04_tlrn_fits_on_colab.ipynb pasted into the 2026-10-12 working session (ibnr 0.7.3 from main, jax 0.7.2, Python 3.13, one TPU device)
    title: Colab TPU output
  - id: gpu
    resource: the Colab run of the accident-year fits pasted into the 2026-10-12 working session (ibnr 0.7.3, jax 0.11.1, Python 3.13.15, backend gpu, CudaDevice(id=0)), saved to Drive as tlrn_8_ay-ibnr0.7.3.pkl and tlrn_13_ay-ibnr0.7.3.pkl in tlrn_fits_ay
    title: Colab GPU output
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

The accident-year config (`accident_year`: premium_lr head, attention over line, lag and
accident year, mcl_blend member, retrain_per_valuation calibration) trains 20 members at
each of 4 valuation dates, so 80 members per row, for 3,000 epochs with batches of 8
companies. The finished fits ran on a Colab **GPU**, not the TPU (jax 0.11.1, backend gpu,
device CudaDevice(id=0), Python 3.13.15, ibnr 0.7.3; the GPU model was not printed).[^gpu]

| Row | Fit time | First 100 epochs | Each later 100 epochs |
|---|---|---|---|
| tlrn_8_ay | 451.4 s | 75 s (compilation included) | about 12.3 s |
| tlrn_13_ay | 434.0 s | 63 s | about 12.3 s |

All 80 members of each row trained all 3,000 epochs (no early stop). Both fits were saved
to Drive as `tlrn_8_ay-ibnr0.7.3.pkl` and `tlrn_13_ay-ibnr0.7.3.pkl` in the folder
`tlrn_fits_ay`. The member validation scores (20 rows printed per fit, one per member of
the final protocol) were:

| Row | validation_ay_line_ape | validation_company_ape | alpha |
|---|---|---|---|
| tlrn_8_ay | about 0.071 to 0.080 | about 0.032 to 0.046 | 0.27 to 0.69 |
| tlrn_13_ay | about 0.067 to 0.074 | about 0.033 to 0.045 | 0.44 to 0.77 |

**Against the laptop.** Notebook 04 on the laptop CPU with torch took **8 h 51 min** for
its single accident-year fit (cache timestamps 2026-10-08 09:59 to 18:50).[^laptop] One GPU
fit of about 7.5 minutes is roughly **70 times** faster. This compares one laptop fit with
one GPU fit of a similar configuration. They are not identical fits: the laptop fit is the
single row notebook 04 calls `tlrn_ay`, and these are two different rows. No comparison of
the members' scores with torch's was made.

**Against the TPU.** Earlier the same day the TPU ran an accident-year fit of the same size
at about 0.6 s per epoch. The first attempt reached epoch 1,700 at 1,085 s and its output
stopped (the run was cut off, and a fit that is cut off must start over, because the
checkpoint is one saved file per finished fit). A second attempt left a Drive record of
1,879.9 s on the TPU; Ethan removed that folder and reran, so it is superseded and not a
result. Those TPU runs are not complete records. Taking them as they are, the GPU was
about 5 times faster per epoch on this variant. That is a one-off observation, not a
ranking of GPUs and TPUs: the JAX versions differ (0.11.1 against 0.7.2) and the GPU model
is unknown. The fits of the published protocol (40 members of tlrn_8 and tlrn_13) are the
ones that ran on the TPU, in 618 s each.

# Lessons from getting it to run

* A TPU can be opened by one process only. The notebook's first cell holds it, so the
  runner must be called inside the notebook (`tlrn_colab.main([...])`), not started with
  `!python`, which fails with "The TPU is already in use by process with pid ...".
* Colab's Python is 3.13 and ibnr declares 3.11 to 3.12, so the install needs
  `--ignore-requires-python`. The cap comes from the `bayesian` and `interop` extras.
* The repository had to be public for Colab to fetch it.
* A Colab runtime can be a GPU or a TPU, depending on the runtime type chosen. The runner
  records which in each fit's `.json` (`jax_backend` and `devices`), so check there before
  quoting a time.

[^colab]: Colab TPU output
[^gpu]: Colab GPU output
[^laptop]: Notebook 04 run on the laptop
