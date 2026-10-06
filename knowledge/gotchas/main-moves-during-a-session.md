---
type: Gotcha
title: Main moves during a session
description: "Parallel sessions land pull requests on main while a task is in progress, so a branch can silently revert merged work or redo a fix that is already on main; check main and the worktree list at the start, and rebase before committing."
tags: [git, process, worktrees, parallel-sessions]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note rebase-before-pr-main-moves.md (private, outside the repository)
    title: Rebase before the PR, main moves
    last_modified: 2026-07-28T07:31:51.326Z
  - id: pr-55
    resource: https://github.com/EKtheSage/ibnr/pull/55
    title: Scale-relative convergence for the Clark curve MLE (b13fbc1)
  - id: pr-60
    resource: https://github.com/EKtheSage/ibnr/pull/60
    title: Defer scipy.integrate and scipy.optimize off the import path
---

# The trap

`main` in `ibnr` advances **during** a single agent session. Parallel sessions run in
their own worktrees and land pull requests continuously, so a branch cut at the start of a
task can be two or three merged pull requests behind by the time the work is
committed.[^note]

# How it showed up

**Silently reverting someone's fix.** 2026-07-27: a branch cut from `41e4334` finished
about 40 minutes later with `main` at `32cf8af`, two pull requests ahead. One of them
(#60, "defer scipy.integrate and scipy.optimize off the import path") had moved
`from scipy.optimize import minimize` from module scope into `Clark.fit()`, the exact
function being edited.[^pr-60] Committing on the stale base and opening a pull request
would have moved that import straight back to module scope and re-broken a deliberate
cold-start optimization, while the diff looked innocent because the change was never made
on purpose.[^note]

**The expensive version: the fix was already on main.** Also 2026-07-27, the Clark MLE
convergence bug: researched, implemented, tested and committed over about an hour, while
PR #55 (`b13fbc1`), fixing the identical bug by the identical rule
(`fatol = 1e-12 x the objective's magnitude`), was already in main's history the whole
time.[^pr-55] The worktree was checked out at an older commit (`41e4334`), and the task
brief described the failure as live. Everything built was redundant on arrival. (The fix
itself: [An optimizer tolerance must scale with the
objective](/gotchas/tolerance-must-scale-with-the-objective.md).)[^note]

Two signals were present in the first command of that session and both were missed: the
session-start git snapshot listed `b13fbc1 fix: scale-relative convergence for the Clark
curve MLE (#55)` among main's recent commits, and `git worktree list` showed sibling
branches `fix/clark-ldf-premium` and `fix/clark-ldf-optional-premium` on the same gallery
entry. A worktree's HEAD is not main: `git log --oneline -5 main` is a different question
from `git log --oneline -5`, and only the first answers "has this already been
fixed".[^note]

**Work only in a worktree branch is not durable here.** That session's worktree was
deleted mid-flight by another session's cleanup, and the branch and its commits were later
removed entirely (not in any reflog).[^note]

# Why it is hard to see

The damage is invisible in a normal review of your own diff. You read your change, it
looks right, and the reversal of someone else's work rides along as context you never
typed. `git diff main --stat` showing files you never touched is the tell.[^note]

# What to do

1. At the **start** of a task, run `git log --oneline -15 main` and `git worktree list`.
   If a sibling branch name overlaps the target, say so before building anything.
2. Before committing, and again before opening a pull request, run
   `git log --oneline HEAD..main`. If it is not empty, `git rebase main` and resolve
   deliberately: read each conflict for intent, not just text, because the conflicting
   hunk is usually someone's recent fix.
3. Re-run the tests afterwards, since a clean textual rebase can still be semantically
   wrong.[^note]

Related: [What CI actually checks](/findings/ci-test-matrix.md), and from the same
session [write_text rewrites line endings on Windows](/gotchas/write-text-rewrites-line-endings.md).

[^note]: Rebase before the PR, main moves
[^pr-55]: Scale-relative convergence for the Clark curve MLE (b13fbc1)
[^pr-60]: Defer scipy.integrate and scipy.optimize off the import path
