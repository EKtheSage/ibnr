---
type: Gotcha
title: Removing git worktrees on Windows
description: Why git worktree removal half-fails on this Windows machine, which tools lie about what is on disk, and what actually deletes a leftover tree.
tags: [git, worktrees, windows, cleanup]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, cleanup entries of 2026-07-25 (lines 1049-1073), 2026-07-26 (lines 1211-1216), 2026-08-03 (lines 1485-1493), 2026-08-06 (lines 476-481) and 2026-09-23 (lines 119-123)"
    last_modified: 2026-09-25T08:31:20.962Z
---

Each task here gets its own branch in its own worktree, so worktrees are removed often.
These traps cost real time.

# The traps

* **`git worktree remove` fails with "Filename too long" when the worktree holds a
  `.venv`, and fails only half way.** It unregisters the worktree but leaves the
  directory. That is how three orphan directories were created, and once their `.git` link
  file was gone git could neither remove nor inspect them. Delete the `.venv` first, then
  remove the worktree.[^note]
* **Two tools lie about long-path trees.** `Get-ChildItem "\\?\..."` reported 0 items on
  trees holding more than 20,000 files, and `Remove-Item "\\?\..."` reported success while
  deleting nothing. Use `[System.IO.Directory]::GetFileSystemEntries` or `EnumerateFiles`
  with the `\\?\` prefix to see what is really there.[^note]
* **Empty orphan directories are left behind** in `.claude/worktrees/`. Check that
  directory on disk, not only `git worktree list` (2026-07-26).[^note]
* **An empty directory that looks unlocked can be another session's shell working
  directory.** Nothing shows it in the process command lines; retry after the owning
  session closes (2026-08-06).[^note]
* **`gh pr merge --delete-branch` from a worktree** fails its local step when `main` is
  checked out elsewhere, and that failure also skips the remote branch delete. Check
  `git ls-remote --heads` after merging (2026-08-03).[^note]

# What works

`[System.IO.Directory]::Delete("\\?\<path>", $true)` from PowerShell deletes such a tree.
PowerShell `Remove-Item` and `cmd rmdir` both failed (the latter is also refused by the
agent's permission check). Then `git worktree prune`. This worked first time on
2026-09-07 and 2026-09-23. On 2026-07-25 it reclaimed 2.5 GB, taking the repository
directory from 3.8 GB to 1.29 GB.[^note]

[^note]: Project status log, cleanup entries of 2026-07-25 (lines 1049-1073), 2026-07-26 (lines 1211-1216), 2026-08-03 (lines 1485-1493), 2026-08-06 (lines 476-481) and 2026-09-23 (lines 119-123)
