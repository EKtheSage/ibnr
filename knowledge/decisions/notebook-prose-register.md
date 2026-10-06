---
type: Decision
title: Notebook prose register
description: "Ethan's standing preference (2026-08-04, extended 2026-08-05) for analysis-notebook narrative: it reads like a CAS E-Forum or Variance paper, defines its terms, uses the profession's words, and a term ban covers outputs as well as sources."
tags: [notebook, writing, style]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note notebook-portability-pattern.md (private, outside the repository)
    title: Notebook portability pattern (item 6, prose register)
    last_modified: 2026-08-05T04:54:59.510Z
  - id: pr-86
    resource: https://github.com/EKtheSage/ibnr/pull/86
    title: Where the prose register was set
---

# Decision

Set by Ethan on 2026-08-04 (PR #86): notebook narrative reads like a CAS E-Forum or
*Variance* paper.[^pr-86]

* A paper-style opening: what is compared, on what data, scored how.
* Plain nominal headings and a Limitations section.
* Formal citations (the Meyers monograph, Mack 1993, Zhang 2010).
* No references to other notebooks, and no naming of scores or model families the
  notebook does not use.
* No chatbot cadence: no "deliberately / honest / the whole point", no dash used as a
  punchline, no cute section titles, no second-person imperatives.[^note]

# Further rulings (2026-08-05)

* No "amortized" (rare in actuarial usage).
* Define terms at first use: "panel" in the econometric sense; "screen" as a fixed
  mechanical set of entry criteria.
* Use the profession's terms, not coinages: "upper triangle", never "training staircase".
* Sort contrasts: keep "X rather than Y" only when Y is the default a knowledgeable reader
  would assume; flatten strawmen.
* **The general absence rule** (it replaces an earlier special case about credentials):
  a statement about what is NOT done, NOT needed, or NEVER engages earns its place only
  when the reader would otherwise assume the opposite. "Nothing is tuned per cohort"
  stays. Introducing a mechanism only to dismiss it, reassurances that an assert "fails
  loudly", and defensive implementation trivia go.[^note]

# How to apply it

* **A term ban must cover outputs too.** Trim the display cells (library reprs and
  full-board frames leak column names) and re-execute, then sweep the final file's
  sources **and** outputs. A sweep of sources only passed while five stale outputs still
  showed the banned material.
* Comment-only code edits are proven safe by an AST-identity check against HEAD, which is
  what lets prose-only pull requests skip re-execution.
* Figures are **derived** from the data objects, never placed by hand: a hand-drawn
  triangle figure shipped with four missing cells.[^note]

Related: [Making an analysis notebook portable](/playbooks/make-a-notebook-portable.md).

[^note]: Notebook portability pattern (item 6, prose register)
[^pr-86]: Where the prose register was set
