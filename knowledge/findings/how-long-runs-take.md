---
type: Measurement
title: How long the long runs take
description: Wall-clock times recorded for retrospectives, notebooks, tlrn training and builds on the dev laptop between July and September 2026, with the places where measurements disagree.
tags: [performance, runtime, notebook, tlrn, harness]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-01-03T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, timing figures across entries of 2026-07-08 to 2026-09-25"
    last_modified: 2026-09-25T08:31:20.962Z
---

All on the 16-core Windows dev laptop unless said otherwise. For the tlrn throughput
profile see [tlrn CPU training throughput](/findings/tlrn-cpu-training-throughput.md); for
notebook 04's stage times see [re-run notebook 04](/playbooks/rerun-notebook-04.md).

# Bayesian retrospectives

* Compartmental at the monograph's sampler settings (`adapt_delta` 0.99, tree depth 15):
  about 160 to 250 s per company, about 190 s typical; workers' compensation lognormal
  median 488 s. Other Bayesian entries: about 10 s per company (2026-07-21).[^note]
* Both 200-company compartmental retrospectives through the parallel harness with staged
  escalation: **1 h 52 min for both variants**, 16 fit-hours in total, about 10 times
  faster than one after another. 19 and 21 of 200 companies escalated to the second stage
  (2026-07-22).[^note]
* Harness stage 1 for compartmental (0.9 / 12): about 37 to 39 s per company against
  about 160 s at the monograph settings (about 4 times); `meyers_ccl` on 4 workers'
  compensation companies about 16 s wall clock (2026-07-21).[^note]
* PyTensor's first C compile: about 4 to 5 minutes, then cached (2026-07-08). See [PyMC
  speed on Windows](/findings/pymc-speed-on-windows.md).[^note]

# Notebooks

| run | wall clock | date |
|---|---|---|
| notebook 03b | 2.5 min | 2026-08-03 |
| notebook 03c on 0.5.6 | 9.8 h | 2026-08-24 |
| notebook 03c on 0.5.8 | 10.3 h | 2026-08-25 |
| notebook 04 smoke runs | 59 to 62 min | 2026-09-22 to 2026-09-24 |
| notebook 04 on 0.7.0 | 599 min | 2026-09-22 |
| notebook 04 on 0.7.1 | 614 min | 2026-09-24 |
| notebook 04 on 0.7.1, second round | 618 min | 2026-09-25 |

In notebook 03c on 0.5.6 the joint multi-line arm took about 1 hour and the AR arm about
3.2 hours, 2.7 of them the rollout.[^note]

# tlrn training

The numbers moved as the training setup changed; each is given with its date.

* 2026-09-21, 16 torch threads, one seed: about 1.1 s per epoch paid-only and about 1.2 s
  with 13 features, so about 1 hour per 3,000-epoch seed, 9 to 10 hours per 10-seed fit,
  and about 30 hours for notebook 04's three fits in a row.[^note]
* 2026-09-22, 4 torch threads: 0.25 s per epoch against 0.8 to 1.0 s at 16 threads (see
  [torch thread count for small models](/findings/torch-threads-for-small-models.md)). The
  0.7.0 notebook's three tlrn fits took 158, 148 and 86 minutes.[^note]
* 2026-09-23: 4 processes of 4 threads each gave 0.19 s per epoch for a round of 4
  members (40 members estimated at about 95 minutes); 8 processes of 2 threads gave
  0.69 s per epoch for a round of 8, which the note calls slower.[^note]
* 2026-09-24: the 40-member fits actually took 224 and 233 minutes at 4 x 4. The short
  120-epoch measurement had underestimated by about 2.3 times. The note adds "a real
  10-member round of 4 takes ~22 min per member"; what that phrase measures exactly is
  unclear.[^note]

**Where this disagrees with the newer profile.** [tlrn CPU training
throughput](/findings/tlrn-cpu-training-throughput.md) (2026-10-05) measured 11.3
member-epochs per second at 4 x 4 and 12.3 at 8 x 2, so 8 x 2 was slightly *faster*
there, against this note's finding (2026-09-23) that 8 x 2 was slower. The 2026-09-23
figures work out to about 21 member-epochs per second at 4 x 4 (4 members / 0.19 s) and
about 11.6 at 8 x 2 (8 / 0.69 s); the 4 x 4 figure is above the 10 to 15 ceiling the newer
profile reports. The two used different networks and run lengths (the newer one profiles
the accident-year variant), and the 2026-09-24 entry already says the short measurement
underestimated real fit times by about 2.3 times. Which comparison holds for a full fit is
not settled.

# Speed-up ideas proposed on 2026-09-23, none built or measured then

1. Train the ten members as one batched network (torch.func vmap or a member dimension),
   estimated 3 to 10 times faster; the digits would change.
2. Members in parallel processes. Each member seeds its own torch and numpy streams, so
   results should match at the same thread count. (Built later as `fit(processes=n)` in
   0.7.1.)
3. Derive the patience-120 fit from the full fit by saving each member's weights at every
   validation improvement: the trajectories are identical until the stop (members 3, 5, 6,
   7 and 8 share best epoch and score in both runs). That would save the 86-minute fit
   exactly.
4. `torch.compile` (needs MSVC on Windows; unchecked then).
5. A Linux machine with many cores.

The newer measurements rule out the first and fourth on this laptop: see [JAX versus torch
on a CPU](/findings/jax-versus-torch-on-cpu.md) and [tlrn CPU training
throughput](/findings/tlrn-cpu-training-throughput.md).[^note]

# Builds and releases

* great-docs site build: about 2.5 minutes, 40 pages (2026-07-23).[^note]
* Trusted publishing workflow: 45 s for 0.5.9, 52 s for 0.5.7.[^note]

[^note]: Project status log, timing figures across entries of 2026-07-08 to 2026-09-25
