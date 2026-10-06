---
type: Gotcha
title: The release gate wants green runs on the tagged commit
description: The release workflow refuses to publish unless the Tests and Lint runs on the exact tagged commit finished green; runs cancelled because GitHub never gave them a runner count as not green, and the fix is to re-run the failed jobs and tag again.
tags: [release, ci, pypi]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:40:00Z }
sources:
  - id: workflow
    resource: https://github.com/EKtheSage/ibnr/blob/main/.github/workflows/release.yml
    title: The release workflow and its test gate
  - id: run
    resource: https://github.com/EKtheSage/ibnr/actions/runs/37369930638
    title: The 0.7.3 release run that refused to publish
  - id: tests-run
    resource: https://github.com/EKtheSage/ibnr/actions/runs/37369902815
    title: The Tests run on the merge commit (attempts 1 and 2 cancelled, attempt 3 green)
---

# What happened

Release 0.7.3: the release PR (#165) merged to main as `ebd8ebe`, and `v0.7.3` was
tagged and pushed straight away. The Release workflow's first job, "Tests green on this
commit", waited five minutes, then failed with "nothing was published": six pytest jobs of
the push-to-main Tests run on that commit, and the `ruff` job of the Lint run, had ended
in the state *cancelled*. The build job was cancelled and the publish job skipped.[^run]

The gate looks for pytest and ruff check runs *on the tagged commit*. The PR's own green
checks do not count, because they ran on the branch, not on the merge commit. Tests and
Lint run on pushes to main and on pull requests, never on a tag.[^workflow]

# Why the runs were cancelled

GitHub did not give the jobs a runner. The annotation on a cancelled job of the second
attempt reads "The job was not acquired by Runner of type hosted even after multiple
attempts": the six jobs started at 20:51:28 UTC and were all cancelled together at
21:06:29, fifteen minutes later. It was an outage of the hosted runners, not a problem in
the code and not caused by the tag push. (The annotation was read on the second attempt;
the first attempt's jobs were cancelled the same way but their annotation was not
read.)[^tests-run]

# What to do

1. Tag only after Tests and Lint have finished green on the merge commit on main.
2. If jobs were cancelled, read a cancelled job's annotation. If it says the job was not
   acquired by a runner, re-run only the failed jobs (`gh run rerun <run id> --failed`)
   and wait for green. Here the second re-run (attempt 3) went green.
3. Delete the tag and push it again (`git push origin :refs/tags/v0.7.3`, then tag the same
   commit and push). Re-running the release run itself cannot help, because it creates no
   test runs; only a new tag push starts the release workflow again.

[^run]: The 0.7.3 release run that refused to publish
[^workflow]: The release workflow and its test gate
[^tests-run]: The Tests run on the merge commit (attempts 1 and 2 cancelled, attempt 3 green)
