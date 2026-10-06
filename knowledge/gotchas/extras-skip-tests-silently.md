---
type: Gotcha
title: Missing extras skip tests silently
description: "A green `uv run pytest` in a fresh worktree does not cover the nn or bayesian gallery entries, because their tests skip when the extra is not installed."
tags: [testing, pytest, extras, nn, bayesian]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note extras-skip-tests-silently.md (private, outside the repository)
    title: Extras skip tests silently
    last_modified: 2026-07-25T10:12:07.650Z
---

# The trap

`uv sync` installs the core package and the dev group only. In a fresh worktree the
tests for the `[nn]` and `[bayesian]` extras therefore **skip rather than fail**, and a
green `uv run pytest` proves nothing about those entries.[^note]

Measured 2026-07-25: 387 passed / 41 skipped without torch, against 422 passed / 38
skipped after `uv sync --extra nn`. The whole multi-line transformer test file was one
silent skip (`could not import 'torch'`).[^note]

# Why it happens

The skip is by design: CLAUDE.md says torch must never be imported at module level, and
CI runs each extra in isolation. But it means the default local command silently
under-tests. It is the same failure shape as [the inert parameter bug
class](/gotchas/inert-parameter-bug-class.md): the check ran and said nothing, so it
looked like coverage.[^note]

When the note was written, CI had no pytest workflow at all (only lint, docs and
release), so nothing downstream caught what the local skip hid. That changed on
2026-07-27; see [What CI actually checks](/findings/ci-test-matrix.md).

# What to do

After editing anything under `gallery/nn/` or `gallery/bayesian/`, re-run the touched
test file with the matching extra installed and confirm it actually executed: `-rs`
prints the skip reasons, and pytest's exit code 5 means nothing was collected.[^note]

Budget for it: `--extra nn` pulls torch, a multi-GB install into that worktree's
`.venv`, which is also what makes removing the worktree slow.

[^note]: Extras skip tests silently
