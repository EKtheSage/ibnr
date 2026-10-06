---
type: Playbook
title: Cut a release
description: The pre-flight checks run before tagging an ibnr release, the traps hit while publishing to PyPI, and the release history from 0.2.0 to 0.7.1 as the agent's status note recorded it.
tags: [release, pypi, ci, process]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-04-05T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, release entries from 2026-07-23 to 2026-09-23 (lines 82-98, 176-180, 262-308, 821-873, 1336-1345, 1411-1455, 1533-1541, 1562-1602)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: publishing
    resource: docs/publishing.md
    title: Why the first release is 0.2.0
  - id: pr-133
    resource: https://github.com/EKtheSage/ibnr/pull/133
    title: The release workflow's test check
---

# Pre-flight before tagging

Run before the 0.5.1 tag (2026-07-29) and recorded as worth repeating as is:[^note]

1. `main` equals `origin/main` equals the release commit, with a clean working tree.
2. No tag of that version exists locally or on the remote (`git tag -l`).
3. `uv build`.
4. `twine check --strict` on both artifacts.
5. Read the wheel's own `METADATA`.
6. Install the built wheel into a clean virtual environment and import it.
7. Tag the release commit by its hash, not a moving `main`, because follow-up PRs may
   already be open.

Since PR #133 (2026-09-07) the release workflow publishes only when the pytest jobs and
`ruff` on the tagged commit finished green. It polls in-progress runs every 60 seconds for
up to 30 minutes, and treats a skipped or neutral run as a failure.[^pr-133] See [the release
check wants green runs on the tagged commit](/gotchas/release-gate-wants-green-runs-on-the-tagged-commit.md)
for how that went wrong on 0.7.3.

Version numbers move slowly: fixes and additions are patch releases (see [ibnr versions
move slowly](/decisions/slow-versioning.md)). 0.5.9 rather than
0.6.0 was Ethan's call, because its new refusals replace answers that were already
wrong.[^note]

# Traps met while publishing

* **PyPI lags the upload by 30 to 60 seconds.** Right after a green release run,
  `pypi.org/pypi/ibnr/json` still showed the previous version and `uv pip install
  ibnr==<new>` failed as unsatisfiable. Neither is a failed publish. `pypi.org/simple/ibnr/`
  updated first; re-query the JSON with `Cache-Control: no-cache` and retry the install.
  All three agreed within a minute (2026-07-29).[^note]
* **There will never be a 0.1.0.** A stale `v0.1.0` tag already sat on the 2026-07-06
  "Initial import" commit, so the first release was 0.2.0.[^publishing]
* **0.2.0 misreports its version** as "0.1.0", from a second hard-coded version literal.
  `__version__` is now read from `importlib.metadata` and `tests/test_version.py` guards it.
  Yanking 0.2.0 on PyPI needs the web interface and was still not done on 2026-07-28.[^note]
* **`invalid-publisher` on the first publish** (2026-07-23): PyPI's trusted publisher was
  keyed to repository `ibnr` while GitHub sent `probabilistic-ml-reserving`. Fixed by
  renaming the GitHub repository to `EKtheSage/ibnr`. Renaming a repository invalidates a
  trusted publisher, which is cheap to redo only while it is still pending.[^note]
* **`400 Non-user identities cannot create new projects`** (the `cas-schedule-p` data
  package, 2026-08-04): the pending publisher carried the wrong project name while the
  OIDC identity matched. A different error from `invalid-publisher`, which means the
  identity itself did not match. After the fix on the account side, `gh run rerun --failed`
  republished from the run's existing artifacts with no new tag.[^note]
* **With `license = "MPL-2.0"` (an SPDX expression), hatchling errors if a `License ::`
  classifier is also present.** Do not add one.[^note]
* **GitHub billing can stop every job.** On 2026-09-23 the repository was private, so
  Actions minutes were billed, and every job of release PR #148 reported "not started
  because recent account payments have failed or your spending limit needs to be
  increased". Only Ethan could fix it; it also blocks the trusted-publishing release.[^note]
* **Merging from this agent's shells** (2026-09-07): the PowerShell tool's permission
  check refused `gh pr merge` twice while the Bash tool ran it. `--delete-branch` then
  failed locally with "main is checked out elsewhere", and that failure also skipped the
  remote branch delete; delete it by hand and check `git ls-remote --heads`.[^note]
* 0.5.7 was cut as a version-bump commit straight to `main` without a PR, which the note
  calls "0.5.6 precedent"; the same note also records a release PR (#101) for 0.5.6, so
  which releases skipped a PR is unclear. Later releases went through a release PR.[^note]

# Release history

As recorded in the note. The CHANGELOG is the authority on contents.

| version | date | what the note says about it |
|---|---|---|
| 0.2.0 | 2026-07-23 | first PyPI release, trusted publishing on a tag push |
| 0.3.0 | 2026-07-23 | the `interop` extra, so `to_chainladder`/`to_bermuda` name their missing dependency |
| 0.4.0 | 2026-07-23 | polars moved to an extra; duckdb-only core |
| 0.5.0 | 2026-07-28 | tagged from `864ea9f` |
| 0.5.1 | 2026-07-29 | tagged at `972b4a9`; `GalleryDiagonal` |
| 0.5.2 | 2026-07-29 | tagged at `87f7d72`; no public change, cut so the leaderboard script fix was on record |
| 0.5.3 | 2026-08-05 | tagged at `b62979e`; anonymous HTTPS transport for the Schedule P download |
| 0.5.4 | not dated in the note | |
| 0.5.5 | 2026-08-06 | case-reserve head, `nn_paid_case`, wording pass |
| 0.5.6 | recorded in the 2026-08-25 entry | version bump only, shipping two fixes already on `main` |
| 0.5.7 | recorded in the 2026-08-25 entry | per-cohort draw streams; entry-level draws for a given integer seed changed |
| 0.5.8 | recorded in the 2026-08-25 entry | additive only (the multiline held-out adapter) |
| 0.5.9 | 2026-09-07 | the `sqlglot<30.18` cap reaches users |
| 0.6.0 | 2026-09-16 | conventional candidates, replay and selection |
| 0.7.0 | 2026-09-22 | tlrn, mcl, point scores, residual calibration |
| 0.7.1 | 2026-09-23 | member reserves and `processes=` for tlrn |

[^note]: Project status log, release entries from 2026-07-23 to 2026-09-23 (lines 82-98, 176-180, 262-308, 821-873, 1336-1345, 1411-1455, 1533-1541, 1562-1602)
[^publishing]: Why the first release is 0.2.0
[^pr-133]: The release workflow's test check
