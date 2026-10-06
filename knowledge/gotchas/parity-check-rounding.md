---
type: Gotcha
title: The parity check rounded its z-scores away
description: "Until 2026-07-24 kernels/parity.py scored z = 0 for well-identified parameters because az.summary rounds to 3 decimals; the milestone-4 CCL parity figures were measured through it, and the draw budget moves the verdict both ways."
tags: [bayesian, parity, arviz, testing]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note parity-gate-rounding-bug.md (private, outside the repository)
    title: Parity check rounding bug
    last_modified: 2026-07-24T17:11:14.836Z
  - id: test
    resource: tests/test_parity_meyers_csr.py::test_parity_summaries_are_not_rounded
    title: The regression test
---

# The bug

`kernels.parity.compare_posteriors` computed its z-scores from `az.summary(...)`, whose
**default rounds every value to 3 decimals** (`round_to=None` means 3 decimal places;
`round_to="none"`, the string, disables it). Since
z = (mean_ref - mean_port) / hypot(mcse_ref, mcse_port), any parameter whose MCSE rounded
to `0.000` hit the `if mcse > 0 else 0.0` guard and was scored **z = 0, an automatic
pass**. The best-identified parameters, where a real discrepancy shows first, were exactly
the ones exempted.[^note]

The symptom that exposed it: reported z-scores landing on exact multiples of sqrt(2)
(1.000, 1.414, 2.121).[^note]

Fixed 2026-07-24 (`round_to="none"` in `_summ`), guarded by
`test_parity_summaries_are_not_rounded`.[^test]

# Consequence for published numbers

The milestone-4 CCL parity claim was measured through the broken check. The card's
"max|z_mean| <= 2.8, max|z_sd| <= 3.6" is not reproducible. Re-run at CCL's own published
protocol (2 workers' compensation companies, 4 chains x 500 draws, warmup 1000, target
acceptance 0.9): numpyro 3.99/3.83 pass and pymc 2.41/4.05 **fail** on company 11347; both
pass on 38687. That is 3 of 4, sitting right on `z_tol = 4` rather than comfortably
inside it. CSR at 4 x 2500 failed 4 of 4 with z_mean 6.8-8.1.[^note]

The root cause of those failures was **not** the check: it was the missing `a_ig` bound
(see [The Meyers a_ig bound is load-bearing](/gotchas/meyers-a-ig-bound.md)). After fixing
that, CSR is 3 of 4 at 2500 draws and CCL is 4 of 4 at 2500 draws.[^note]

# The draw budget cuts both ways (measured)

A *systematic* difference grows with draws (MCSE shrinks like 1/sqrt(draws)), so a fixed
`z_tol` gets stricter the longer you sample. But at **low** draws `mcse_sd` is itself
poorly estimated, which destabilizes the ratio and can inflate z.[^note]

Evidence: at CCL's published 500-draw protocol the SD check straddled the tolerance with
the failing backend alternating by cohort (numpyro failed on 38687, pymc on 11347), the
signature of noise. Re-running the same fits at 2500 draws dropped pymc's max|z_sd| from
**5.83 to 3.24** and numpyro's from 3.28 to 2.45, all passing. Had the gap been
systematic it would have grown by about sqrt(5).[^note]

# What to do

* Run parity at 2500 draws or more. Below about 1000 the SD check is dominated by noise
  and produces alternating spurious failures.
* Still open (Ethan's call; do not change it alone): whether to replace the fixed `z_tol`
  with a practical-equivalence band (TOST-style,
  `|diff| < max(k*mcse, rel_tol*|mean_ref|)`) so the verdict cannot be flipped by the draw
  budget in either direction.[^note]

[^note]: Parity check rounding bug
[^test]: The regression test
