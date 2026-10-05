---
type: Decision
title: Knowledge is kept in OKF format
description: Knowledge collected while working on ibnr is written into this bundle in the Open Knowledge Format v0.2, with the conventions below.
tags: [process, knowledge, okf]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:30:00Z }
sources:
  - id: ethan
    resource: Ethan's instruction in the 2026-10-05 session, pointing at https://github.com/GoogleCloudPlatform/open-knowledge-format
    title: "From now on manage all knowledge you collect in ibnr with this format"
  - id: spec
    resource: /references/okf-spec.md
    title: The format
---

# Decision

Ethan asked (2026-10-05) that all knowledge collected in ibnr be managed in the [Open
Knowledge Format](/references/okf-spec.md).[^ethan] This directory, `knowledge/` at the
repository root, is the bundle. It is shared and version-controlled with the code, which
is where durable project knowledge belongs; it is not an agent-private store.

# What goes in

A durable fact learned while working here that the code and the git history do not already
say: a measurement, a finding, a decision with its reasons, a gotcha, a playbook, a
reference to something outside the repository. Not what the code already records, and not
what only matters to one conversation.

# Conventions

* **One concept per file**, `type` in the frontmatter. Types in use: `Measurement`,
  `Finding`, `Decision`, `Gotcha`, `Playbook`, `Reference`. A new type is fine; consumers
  tolerate unknown ones.
* **Directories by kind**: `findings/`, `decisions/`, `gotchas/`, `playbooks/`,
  `references/`.
* **Frontmatter**: `type`, `title`, `description`, `tags`, `status`, `generated`, and
  `sources` whenever the concept rests on something a reader could follow. Add
  `stale_after` to anything that depends on hardware, a library version or a price.
* **Who wrote it**: `generated.by` is `claude-code/<model>` for a concept written by the
  agent. `verified` is added only when a person has confirmed it, as `human:<id>`; the
  agent never writes it for itself.
* **Links** between concepts are absolute bundle paths, such as `/findings/x.md`.
  Per-claim attribution uses a footnote whose label is a `sources[].id`.
* **Every change** updates the directory's `index.md` and adds a line to `log.md`, newest
  date first.
* **Checked** by `scripts/check_okf.py`, which `tests/test_knowledge_bundle.py` runs.

# Not migrated

The agent's earlier private memory notes (outside the repository) were not moved here.
Moving them is a separate decision.

[^ethan]: "From now on manage all knowledge you collect in ibnr with this format"
