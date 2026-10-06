---
type: Gotcha
title: An optimizer tolerance must scale with the objective
description: "An absolute optimizer tolerance on an objective denominated in loss amounts can be smaller than one float spacing, so convergence depends on the machine's floating point rather than the fit; scale it with the objective."
tags: [optimization, numerics, clark, ci]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note tolerance-must-scale-with-the-objective.md (private, outside the repository)
    title: Tolerance must scale with the objective
    last_modified: 2026-07-28T07:31:58.318Z
  - id: pr-55
    resource: https://github.com/EKtheSage/ibnr/pull/55
    title: Scale-relative convergence for the Clark curve MLE (b13fbc1)
  - id: model
    resource: src/ibnr/gallery/statistical/clark/model.py
    title: _FATOL_REL, _FATOL_MIN, _XATOL and _check_converged
---

# What happened

Fixed on main by PR #55 (commit `b13fbc1`, 2026-07-27) in
`gallery/statistical/clark/model.py`: the constants `_FATOL_REL` (1e-12), `_FATOL_MIN`
and `_XATOL`, plus a `_check_converged` check.[^pr-55] It landed after
`test_ldf_ties_to_chainladder[weibull]` failed one CI leg while three sibling runs on the
same base passed minutes apart (see [What CI actually checks](/findings/ci-test-matrix.md)).[^note]

# Why it failed

scipy's Nelder-Mead stops only when the simplex is small in **both** coordinates and
value (`max|x_i - x_0| <= xatol` **and** `max|f_i - f_0| <= fatol`). Clark's concentrated
Poisson deviance is denominated in the triangle's own currency: -4.3e8 on the genins
sample, where one ULP (the spacing between adjacent floats) is 6e-8. The shipped
`fatol=1e-10` was **1/1678th of a single ULP**, so its half of the test could only be met
when every simplex vertex happened to evaluate to a bit-identical float: decided by the
runner's math library, not by the fit.[^note]

# Why it matters

The estimate was never wrong. Across 200 starts perturbed by 1e-4, converged or not,
omega spanned 2.4e-7 relative, against the 2e-3 the tie-out test asks for. The optimizer
found the answer every time and threw it away 1% of the time. So the thing to fix, and to
test, is the **criterion**: not the answer, and not the iteration budget. Raising
`maxiter` would have fixed nothing, because the criterion was unreachable rather than slow
to reach.[^note]

# What to do

* Any tolerance compared against a quantity denominated in loss amounts must scale with
  that quantity: `fatol = _FATOL_REL * |objective|`, floored at `_FATOL_MIN`. That gives
  about 5e3-7e3 ULPs of headroom at every deviance magnitude from 1e3 to 1e13 (verified
  independently: 1200 perturbed-start fits across five orders of magnitude, zero
  stalls).[^model]
* Tolerances on **log** parameters can stay absolute, because those are of order 1 in any
  currency.
* Before trusting any absolute tolerance, compare it with `np.spacing(|f|)` at the values
  it will actually see.
* Two diagnostics settle it fast: re-sum the objective's terms in a different order
  (mathematically neutral, different rounding; it flipped this fit from 65 iterations to
  the 2000 cap), and sweep tiny start perturbations to get a **failure rate** rather than
  one anecdote.[^note]

# Related

* PR #55 records what the fit actually used in `params_["optimizer"]` and asserts on
  that, rather than re-calling the tolerance helper from the test, which would pass just
  as happily with the helper disconnected. See [the inert parameter bug
  class](/gotchas/inert-parameter-bug-class.md).
* A later session re-derived this entire fix from scratch because it never checked
  whether main already carried it. It did. See [Main moves during a
  session](/gotchas/main-moves-during-a-session.md).[^note]

[^note]: Tolerance must scale with the objective
[^pr-55]: Scale-relative convergence for the Clark curve MLE (b13fbc1)
[^model]: _FATOL_REL, _FATOL_MIN, _XATOL and _check_converged
