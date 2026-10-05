---
type: Gotcha
title: The release gate wants green runs on the tagged commit
description: The release workflow refuses to publish unless the Tests and Lint runs on the exact tagged commit finished green; cancelled runs count as not green.
tags: [release, ci, pypi]
status: draft
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:40:00Z }
sources:
  - id: workflow
    resource: https://github.com/EKtheSage/ibnr/blob/main/.github/workflows/release.yml
    title: The release workflow and its test gate
  - id: run
    resource: https://github.com/EKtheSage/ibnr/actions/runs/37369930638
    title: The 0.7.3 release run that refused to publish
---

# What happened

Release 0.7.3: the release PR (#165) merged to main as `ebd8ebe`, and `v0.7.3` was
tagged and pushed straight away. The Release workflow's first job, "Tests green on this
commit", waited five minutes, then failed with "nothing was published": the push-to-main
Tests run on that commit had six pytest jobs and the Lint run had `ruff`, all in the
state *cancelled*. The build job was cancelled and the publish job skipped.[^run]

The gate looks for pytest and ruff check runs *on the tagged commit*. The PR's own green
checks do not count, because they ran on the branch, not on the merge commit. Tests and
Lint run on pushes to main and on pull requests, never on a tag.[^workflow]

# What to do

Tag only after Tests and Lint have finished green on the merge commit on main. If runs
were cancelled, re-run those workflows, wait for green, delete the tag and push it
again. Re-running the release run itself cannot help, because it creates no test runs.

# Not established

Why the push-to-main runs were cancelled. They were still in progress when the tag was
pushed minutes after the merge; whether the tag push or a concurrency setting cancelled
them was not checked. Status stays `draft` until the retag is done and this is settled.

[^run]: The 0.7.3 release run that refused to publish
[^workflow]: The release workflow and its test gate
