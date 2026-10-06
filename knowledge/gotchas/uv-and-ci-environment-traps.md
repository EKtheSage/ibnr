---
type: Gotcha
title: uv, CI and packaging traps
description: Environment and CI traps hit while developing ibnr and its data package, each with the fix that worked.
tags: [uv, ci, packaging, windows, dev-environment]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
stale_after: 2027-04-05T00:00:00Z
sources:
  - id: note
    resource: agent memory note project-status.md (private, outside the repository)
    title: "Project status log, entries of 2026-07-20 to 2026-09-20 (lines 226-227, 278-291, 716-720, 826-837, 1032-1047, 1218-1223, 1253-1255, 1520-1561)"
    last_modified: 2026-09-25T08:31:20.962Z
  - id: pr-131
    resource: https://github.com/EKtheSage/ibnr/pull/131
    title: The first green Docker image build
  - id: pr-85
    resource: https://github.com/EKtheSage/ibnr/pull/85
    title: The test dependency group rule
---

# Local environment

* **"uv trampoline failed to canonicalize script path"** (2026-07-25). The
  `.venv/Scripts/pytest.exe` shim still pointed at the checkout's old path from before the
  repository was renamed. Plain `uv sync` does not rebuild console shims for packages that
  are already installed. Fix: `uv sync --extra bayesian --reinstall-package pytest`.
  Workaround: `uv run python -m pytest` skips the shim.[^note]
* **Plain `uv sync` removes extras** that an earlier sync installed (arviz, cmdstanpy and
  xarray disappeared). In a checkout that needs them, always sync with the extra.[^note]
* **Another session may change the shared checkout under a running test suite.** PRs
  merged and pulled into `main` mid-run produced one failure that did not reproduce
  (2026-07-25). Re-run before believing a failure in the main checkout. While someone
  else is editing, stage files by name rather than `git add -A`.[^note] See also [main
  moves during a session](/gotchas/main-moves-during-a-session.md).
* **Python's `write_text` on Windows turns line endings into CRLF.** It changed 520 line
  endings in a design spec (2026-09-20); fixed by rewriting the bytes.[^note] See [write_text
  rewrites line endings on Windows](/gotchas/write-text-rewrites-line-endings.md).
* **great-docs on Windows needs `PYTHONIOENCODING=utf-8`**, or it dies on a cp1252
  encoding error. The build takes about 2.5 minutes and needs the `quarto` CLI.[^note]

# CI

* **A package the test suite imports belongs in the `test` dependency group, not `dev`.**
  Every CI pytest job syncs `--no-default-groups --group test`, so a `dev` dependency that
  test modules reach (here through `scripts/` imports) broke collection on all eight jobs
  at once (PR #85, 2026-08-04).[^pr-85]
* **`ruff format` formats notebooks**, and CI checks it, so run `uv run ruff format` on a
  new `.ipynb` before pushing (2026-08-25).[^note]
* **A newer ruff can change formatting.** When ruff moved to 0.16 (2026-07-26) the lint
  job gained `ruff format --check` over python blocks in markdown and the snippet linter;
  rebase and re-run all three before merging.[^note]
* **`uv venv` refuses a runner's existing `.venv`**; the unlocked CI jobs needed
  `--clear` (2026-07-28).[^note]
* **The `setup-uv` action has no floating major tag past v7**, so `@v9` does not resolve;
  pin a full version such as v9.0.0 (2026-08-04, data repository).[^note]

# Docker image (PR #131, 2026-09-07)

* pip cannot resolve `.[bayesian]` at all, and the numpy overrides exist only in uv, so
  the image installs the locked stack: pinned uv 0.11.7 and `uv sync --frozen
  --no-default-groups --extra bayesian`, with `cas-schedule-p` added by
  `uv pip install --no-deps`.[^pr-131]
* `UV_PYTHON` set globally outranks `VIRTUAL_ENV` for `uv pip install`, which put the data
  wheel into the system site-packages and cost one red build. Set it per `uv sync`
  step.[^pr-131]
* The check run in the container needed `docker run -i`; without it the heredoc check
  passed without ever executing.[^pr-131]

# Data package (cas-schedule-p, 2026-08-04)

The data is gitignored, and hatchling reads only the project-root `.gitignore`, so
`[tool.hatch.build] artifacts` is what puts the parquet files in the wheel. Without it
both artifacts build cleanly, pass `twine check`, and ship no data at all (measured). Its
CI counts the parquet files in the wheel and tests the installed wheel before publishing.
Package tags must never create GitHub releases, because a release would become the
consumers' `@latest` data.[^note]

[^note]: Project status log, entries of 2026-07-20 to 2026-09-20 (lines 226-227, 278-291, 716-720, 826-837, 1032-1047, 1218-1223, 1253-1255, 1520-1561)
[^pr-131]: The first green Docker image build
[^pr-85]: The test dependency group rule
