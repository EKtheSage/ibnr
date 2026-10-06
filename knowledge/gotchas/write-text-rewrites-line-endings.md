---
type: Gotcha
title: write_text rewrites line endings on Windows
description: "On the Windows dev box, round-tripping a source file through pathlib's write_text for a temporary edit rewrites the whole file with CRLF line endings; use an editor tool instead."
tags: [windows, git, line-endings]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note rebase-before-pr-main-moves.md (private, outside the repository)
    title: Rebase before the PR, main moves (closing paragraph)
    last_modified: 2026-07-28T07:31:51.326Z
  - id: gitattributes
    resource: .gitattributes
    title: "* text=auto eol=lf"
---

# The trap

Never round-trip a source file through `pathlib.write_text` on this Windows machine to
make a temporary edit. It rewrites the whole file with CRLF line endings: measured, 401
CRLFs written into a file that had none.[^note]

`.gitattributes` carries `* text=auto eol=lf`, so git normalizes the endings on commit,
but the working-tree diff is unreadable until you `git checkout --` the file.[^gitattributes]

# What to do

Make the edit with the editor tool (the agent's Edit tool) rather than a read-modify-write
in Python.[^note]

Found in the same session as [Main moves during a session](/gotchas/main-moves-during-a-session.md).

[^note]: Rebase before the PR, main moves (closing paragraph)
[^gitattributes]: "* text=auto eol=lf"
