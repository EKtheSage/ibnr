---
type: Gotcha
title: A new file in scripts/ needs a compute-image decision
description: tests/test_compute_image.py fails the lean CI legs when a script is neither promised by the Docker image nor listed as excluded with a reason; running only the tests near the change misses it.
tags: [ci, scripts, docker, process]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-06T04:30:00Z }
sources:
  - id: test
    resource: https://github.com/EKtheSage/ibnr/blob/main/tests/test_compute_image.py
    title: The test and its NOT_PROMISED table
  - id: pr-166
    resource: https://github.com/EKtheSage/ibnr/pull/166
    title: The pull request whose first CI run failed on it
---

# The rule

`test_every_script_is_either_promised_or_excluded_with_a_reason` compares the files in
`scripts/` with the scripts the Dockerfile promises to run and the `NOT_PROMISED` table
in `tests/test_compute_image.py`. A script in neither fails the test with "present but
unaccounted for". The point is that a new study script cannot join the repository
without someone deciding whether the compute image can run it.[^test]

# How it showed up

Adding `scripts/check_okf.py` (the knowledge-bundle check) made the test fail on the
five lean CI legs (core, core-py311, polars, core-latest, polars-latest). The legs that
install extras were still running when the failure was read. The fix is an entry in
`NOT_PROMISED` with the reason: the script checks this repository's own markdown, needs
PyYAML (which the image does not install) and computes no result.[^pr-166]

# Why it was missed

Before pushing, only the tests near the change had been run (the knowledge, markdown and
lint tests). The failing test lives in an unrelated file. Run the whole fast suite before
pushing a change that adds a file to `scripts/`, `tests/` or the repository root.

[^test]: The test and its NOT_PROMISED table
[^pr-166]: The pull request whose first CI run failed on it
