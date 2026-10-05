---
type: Reference
title: Open Knowledge Format
description: The format this bundle is written in, with the rules a writer here needs.
tags: [okf, process]
status: stable
generated: { by: claude-code/claude-sonnet-5-5, at: 2026-10-05T22:55:00Z }
stale_after: 2027-04-05T00:00:00Z
sources:
  - id: spec
    resource: https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md
    title: OKF v0.2 specification
    author: team:GoogleCloudPlatform
---

# What it is

A bundle is a directory of markdown files with YAML frontmatter; no registry and no
tooling are required.[^spec] This repository's bundle is `knowledge/`; see [Knowledge is
kept in OKF format](/decisions/knowledge-in-okf-format.md) for the conventions.

# The rules a writer needs (v0.2)

* Every concept file has frontmatter with a non-empty `type`. Everything else is optional.
* Recommended keys: `title`, `description`, `resource`, `tags`.
* Provenance, trust and lifecycle keys: `sources` (each with a `resource`, and optionally
  `id`, `title`, `author`, `usage_count`, `last_modified`), `generated: { by, at }`,
  `verified: [{ by, at }]`, `status` (`draft`, `stable`, `deprecated`; absent means
  stable), `stale_after` (an absolute instant).
* Every timestamp is ISO 8601 with an explicit UTC offset, for example
  `2026-06-30T14:00:00Z`.
* Actors: `<producer>/<version>` for agents, `human:<id>` for people, `process:<id>` for
  automation. A trust tier comes from `verified`: none is unverified, non-human only is
  machine-confirmed, any `human:` actor is human-reviewed.
* `index.md` and `log.md` are reserved names. An `index.md` has no frontmatter, except
  that the bundle root's may carry `okf_version`. Index entries are
  `* [Title](path) - description` under headings. A `log.md` lists date headings
  (`## YYYY-MM-DD`), newest first.
* Links between concepts are standard markdown links; the bundle-relative form
  (`/dir/file.md`) is recommended. A broken link is tolerated by a consumer.
* Per-claim attribution is a footnote whose label is a `sources[].id`.
* Unknown types and unknown frontmatter keys must be tolerated and preserved.

# Left out

Version 0.2 also defines an `Attested Computation` concept type (a sanctioned computation
with an executor and an attester). Nothing here uses it.

[^spec]: OKF v0.2 specification
