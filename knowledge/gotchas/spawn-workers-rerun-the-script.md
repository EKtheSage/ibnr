---
type: Gotcha
title: Spawned workers re-run the script
description: A script that trains with processes>1 must sit under a main guard, because every spawned worker re-imports it; a notebook kernel does not need one.
tags: [tlrn, multiprocessing, windows, notebooks]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:35:00Z }
sources:
  - id: claude-md
    resource: https://github.com/EKtheSage/ibnr/blob/main/CLAUDE.md
    title: "CLAUDE.md gotcha on spawn workers (0.7.1)"
  - id: session
    resource: the 2026-10-05 session, converting analysis/04 to a script to log per-cell progress
    title: Where it was met again
---

# The rule

On Windows the only multiprocessing start method is `spawn`, and a spawned worker
re-imports the module that started it before it does anything else. A script that calls
`fit(processes=...)` at top level therefore makes every worker run the script again. Put
the work under `if __name__ == "__main__":`. A notebook kernel has no file to re-import,
so a notebook needs no guard.[^claude-md]

# How it showed up

Converting notebook 04 into a plain script to get per-cell timing, without a guard: each
`tlrn` worker started re-running the whole notebook from the top. The log showed "cell 2
start" again and again, one Python process held 4.7 GB, and the run was stopped by hand.
Wrapping the converted body in the guard fixed it.[^session]

# A second trap in the same harness

The timing helper used the name `_t`, and notebook 04 assigns a DataFrame to `_t` in a
later cell, so the helper's own next call failed with `'DataFrame' object has no
attribute 'time'` an hour into a run. Give harness names something a notebook will not
use, for example `__nbtime`.

[^claude-md]: CLAUDE.md gotcha on spawn workers (0.7.1)
[^session]: Where it was met again
