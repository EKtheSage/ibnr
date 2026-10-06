---
type: Gotcha
title: A scratch venv under a deep path breaks scipy on Windows
description: "A venv built under the session scratchpad directory cannot import scipy.stats or scipy.sparse.linalg because a compiled file's path exceeds Windows' 260-character limit; build scratch venvs at a short path such as a subdirectory of %TEMP%."
tags: [windows, venv, scipy, environment]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-04-05T00:00:00Z
sources:
  - id: note
    resource: agent memory note scratchpad-venv-path-too-long.md (private, outside the repository)
    title: Scratch venv path too long
    last_modified: 2026-09-25T08:31:03.503Z
---

# The symptom

A scratch venv created inside the agent's session scratchpad directory
(`%TEMP%\claude\<project>\<session>\scratchpad\venv-x`) installs fine, but:[^note]

* `from scipy import stats` fails with
  `ModuleNotFoundError: No module named 'scipy.spatial.transform._rigid_transform_cy'`;
* `import scipy.sparse.linalg` fails with
  `ImportError: cannot import name '_arpacklib' from partially initialized module 'scipy.sparse.linalg._eigen.arpack' (most likely due to a circular import)`.

Loading the `.pyd` directly gives the real reason:
`ImportError: DLL load failed while importing _arpacklib: The specified module could not be found`.

# The cause

Measured 2026-09-05: the `.pyd` path was 265 characters, over Windows' 260-character
MAX_PATH, and `LongPathsEnabled` is 0 in the registry on this machine, so the import
system cannot even read the file. The identical install at
`C:\Users\EthanKang\AppData\Local\Temp\ibnr-fl` imports cleanly, and so does CI on
Linux.[^note]

The scratchpad path alone is about 180 characters, so any compiled package with a deep
module tree (scipy, torch) can cross MAX_PATH. Reinstalling with `--no-cache --reinstall`
does not help; the wheel is fine.[^note]

# What to do

* Put throwaway venvs at a short path under `%TEMP%`, in their own subdirectory, delete
  them when done, and mention the location in the summary. Plain scripts and outputs can
  stay in the scratchpad.
* Never use `%TEMP%` itself: a stray script there named like a standard-library module
  (once a `timeit.py`) is imported in place of the standard library by any Python process
  started with that working directory.
* If scipy "circular import" errors appear in a scratch venv, check the path length
  before suspecting the dependency set.
* In Git Bash, prepend a venv's `Scripts` directory to PATH in POSIX form
  (`/c/Users/...`), or a subprocess call to `ruff` inside a test fails with
  `FileNotFoundError` for no real reason.[^note]

[^note]: Scratch venv path too long
