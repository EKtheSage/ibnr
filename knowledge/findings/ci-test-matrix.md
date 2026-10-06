---
type: Finding
title: What CI actually checks
description: "Since PR #53 (2026-07-27) CI runs a seven-leg pytest matrix whose post-run step fails a leg that silently ran nothing; before that, CI ran only lint and the docs build."
tags: [ci, testing, pytest, extras]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-01-05T00:00:00Z
sources:
  - id: note
    resource: agent memory note ci-has-no-test-job.md (private, outside the repository)
    title: What ibnr's CI actually checks
    last_modified: 2026-07-28T00:18:52.846Z
  - id: pr-53
    resource: https://github.com/EKtheSage/ibnr/pull/53
    title: Added .github/workflows/test.yml
  - id: workflow
    resource: .github/workflows/test.yml
    title: The test workflow
---

# Before 2026-07-27

For most of the repository's life `.github/workflows/` held only `docs.yml`, `lint.yml`
and `release.yml`. No workflow ran pytest, so a green pull request proved only that ruff
and the docs build passed. CLAUDE.md described a CI test matrix that did not exist.[^note]

# Since PR #53

PR #53 added `.github/workflows/test.yml`: seven ubuntu legs (a "leg" is one job of the
matrix, with its own installed set of extras):[^pr-53]

* `core`, `core-py311`,
* one per extra: `polars`, `nn`, `bayesian`, `interop`,
* `all`.

About 18 billed runner-minutes per pull request, about 6 minutes wall clock. The GitHub
runners are **faster** than the dev laptop for this workload, not slower.[^note]

Each leg declares what it installed and what it must run. A step after pytest reads the
JUnit XML and fails the leg on:[^workflow]

* a skip caused by a package that leg installed;
* a named file contributing zero executed tests (`required_files`);
* a test count below a floor.

The floors only detect a collapse; they are not checks on small changes in the count.
`required_files` is the check with teeth.

# Why it is needed

The failure it guards against is specific to this repository and invisible to pytest.
Six test files call `importorskip("torch")` at module level, and a module-level skip
collapses a whole file into one `s`. A core-only environment reports 540 passed / 197
skipped and exits 0: fully green with 155 tests silently absent.[^note] See
[Missing extras skip tests silently](/gotchas/extras-skip-tests-silently.md).

# A known flaky test (as of 2026-07-27)

`pytest (all)` intermittently failed
`tests/test_clark.py::test_ldf_ties_to_chainladder[weibull]` with "Clark MLE did not
converge: Maximum number of iterations has been exceeded". Measured on one commit: fail,
fail, pass on three re-runs of the identical commit, while main passed twice. So a red
`all` leg beside six green legs is not evidence that a change broke anything: re-run
before investigating, and never conclude from a 2-against-2 split.[^note]

The cause: `Clark.fit`'s Nelder-Mead optimizer was given an absolute `fatol=1e-10` on an
objective of magnitude 4.3e8, whose spacing between adjacent floats (ULP) is 6e-8, so the
stopping test could only be met when the simplex vertices were bit-identical. Windows
landed there every time; the Linux `all` environment sometimes did not. It was also a
production hazard: a real cohort could get a spurious `RuntimeError` from a fit sitting
at its optimum. Only this one leg saw it, which was itself the tell: the same test passed
in `interop`.[^note] The fix and its lesson are in [An optimizer tolerance must scale
with the objective](/gotchas/tolerance-must-scale-with-the-objective.md).

# What to do

* CI is a real check now, but read *which* leg failed. A red isolated leg usually means
  an extra's tests stopped running, not that they broke.
* Still run locally with the extras a change touches before claiming it is verified; a
  leg only covers what its `required_files` and floor name.
* When adding a gallery entry or test file that needs an extra, add it to that leg's
  `required_files`, or nothing notices when it stops running.
* Do not add `GH_TOKEN` to that workflow: `gh auth status` failing is what
  `test_schedule_p_github.py` asserts on.[^note]

[^note]: What ibnr's CI actually checks
[^pr-53]: Added .github/workflows/test.yml
[^workflow]: The test workflow
