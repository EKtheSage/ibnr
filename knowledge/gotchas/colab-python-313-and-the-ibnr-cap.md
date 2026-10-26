---
type: Gotcha
title: Colab runs Python 3.13 and ibnr caps Python below 3.13
description: How the Colab notebook installs ibnr on a newer Python than the package declares, why that is safe for the extras it uses, and what a billing lock on GitHub Actions looks like.
tags: [colab, notebooks, python, packaging, github-actions]
status: stable
stale_after: 2027-04-26T00:00:00Z
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-26T12:00:00Z }
sources:
  - id: nb
    resource: analysis/05_tlrn_study3_on_colab.ipynb
    title: "The install and fetch cells"
  - id: cap
    resource: CLAUDE.md
    title: "Tooling and conventions, the Python cap (decision of 2026-07-28)"
---

# Colab is on Python 3.13

The Colab log for the third tlrn study showed `/usr/local/lib/python3.13/dist-packages`, so
Colab now runs Python 3.13. ibnr 0.7.6 declares `requires-python = ">=3.11,<3.13"`, so a plain
`pip install ibnr==0.7.6` there is refused. Measured in a clean 3.13 venv with pip 26.2:
`Could not find a version that satisfies the requirement ibnr==0.7.6 (from versions: 0.3.0, 0.4.0)`.[^cap]

The cap exists for the `[bayesian]` and `[interop]` extras, which pull numpy below 2 and
have no wheel for 3.13. The notebook installs neither. With `--ignore-requires-python` the
notebook's two steps worked on 3.13: `--no-deps ibnr==0.7.6`, then
`--only-binary=:all: ibnr[jax]==0.7.6 cas-schedule-p==2026.6.13`. That gave ibnr 0.7.6, torch
2.14.1, jax 0.12.0.1, numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.0, duckdb 2.0.0, ibis 12.0.0 and
sqlglot 30.17.0. `import ibnr` worked and `TLRNConfig.companion_final_variant()` built a
network of 18,537 parameters, the count the notebook checks. No fit was run.[^nb]

So the notebook passes the flag only when `sys.version_info >= (3, 13)` and prints why. Do not
lift the cap in the package for this: the extras really cannot install there.

* `uv pip install` has no `--ignore-requires-python`; use pip, which is also what Colab runs.
* A fresh pip install of the `[jax]` extra on Windows took about 8.5 minutes, mostly torch.

# The scripts need no token while the repository is public

The notebook fetches two runner scripts through the GitHub contents API. The first version
stopped with "GITHUB_TOKEN is not set" because the repository had been private. It now uses a
token when the Colab secret exists, retries once without it if GitHub refuses it, and with no
token raises a message naming a possibly private repository only on a 401, 403 or 404.[^nb]

# A billing lock fails every job in seconds

When the account is locked for a billing problem, every GitHub Actions job fails within about
two seconds with the annotation "The job was not started because your account is locked due to
a billing issue". The logs are empty. A public repository does not exempt the account from the
lock. The fix is on the billing page, which only the account owner can reach.

[^cap]: CLAUDE.md, the Python cap (decision of 2026-07-28).
[^nb]: analysis/05_tlrn_study3_on_colab.ipynb, install and fetch cells.
