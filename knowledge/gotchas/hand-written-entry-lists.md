---
type: Gotcha
title: Hand-written entry lists in tests
description: "Gallery entries are cloned from older entries, so a template bug spreads, and a hand-written list of entries that have a capability tests a newly capable entry with nothing at all; derive the list from the registry."
tags: [testing, gallery, registry]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note entries-cloned-from-stale-templates.md (private, outside the repository)
    title: Entries cloned from stale templates
    last_modified: 2026-07-30T00:51:08.405Z
  - id: pr-47
    resource: https://github.com/EKtheSage/ibnr/pull/47
    title: The fit() torn-state fix
  - id: pr-52
    resource: https://github.com/EKtheSage/ibnr/pull/52
    title: test_every_registered_entry_is_covered
---

# First half: a template bug spreads

Gallery entries are written by copying an existing entry's `model.py`, so a defect fixed
in the original silently reappears in every entry added after the fix.[^note]

Measured 2026-07-27: the `fit()` torn-state fix (PR #47) covered the 11 entries that
existed at the time.[^pr-47] `nn/deeptriangle`, `nn/mdn` and `nn/resnet` landed from
parallel sessions within hours carrying the identical pre-fix ordering
(`self.contract_ = nn_data(...)` before the fallible `train_ensemble(...)`). All three
expose `log_lik_at` / `predict_at`, which is what makes it a correctness bug rather than
a tidiness one. `bayesian/guszcza_growth_curve`, landed the same day, was clean: it was
written against the newer template.[^note]

The fix and the clones happen at the same time, so no rebase or merge conflict ever shows
the divergence. CI's pytest matrix ([What CI actually checks](/findings/ci-test-matrix.md))
catches the half that turns tests red, but not the silent half below.

# Second half, and it is silent: a list of entries that have a capability

Measured 2026-07-29: `tests/test_nuts_sampler.py` opened with `ENTRIES = [...]`, "every
entry with a PyMC port", written out by hand. When `guszcza_growth_curve` gained its
ports, its new `nuts_sampler` support was tested by nothing at all. No test went red,
because the list simply did not name it.[^note]

Fixed by deriving the list (`family == "bayesian"`), plus a guard asserting that the
derived set is the whole family and cannot collapse to empty. That also turns CLAUDE.md's
standing rule "a bayesian entry is not done until it has both ports" into something a
test enforces rather than something a document asks for.[^note]

So a hand-written list of entries that *have* a property is worse than one of entries
that *lack* it: its failure is a green run on zero cases.

# What to do

* When a test pins a per-entry rule, assert that its entry list **is** `gallery.list()`
  rather than trusting a hand-written list.
  `tests/test_fit_atomicity.py::test_every_registered_entry_is_covered` does this as of
  2026-07-27 (PR #52) and is the pattern to copy: it fails naming the uncovered entry, at
  the moment the author is looking at that concern.[^pr-52]
* A per-entry fixture table is fine and often unavoidable (the families need different
  cohort recipes and different ways to force a failure). What must not be hand-maintained
  is *which entries are checked at all*.
* To force a failure, prefer a real guard the entry's own estimator raises after its
  contract is built (ODP's informative-cells degrees-of-freedom check, SUR's positivity
  guard, the copula's increment guard) over a stubbed sampler.
* When adding a row for an entry that is already correct, verify the row by injecting the
  bug: a green row on a correct entry proves nothing.[^note]

Same reasoning as [the inert parameter bug class](/gotchas/inert-parameter-bug-class.md):
test the property through the public entry point, over everything that claims it.

# When widening a shared value, search for the old value first

Widening `BACKENDS = ("stan",)` to all three backends broke two tests in
`tests/test_heldout_scorer_guszcza.py`. One asserted the tuple literally. The other used
`backend="numpyro"` to prove the validator fires; once numpyro became real, it sampled
instead and died on a missing `arviz` in a core environment.[^note]

A `grep -rn BACKENDS tests/` costs seconds and finds both; running the suite costs many
minutes and, if the wrong files are running, finds neither. Then verify against a CI leg
**without** the extras, because that is where a newly reachable code path fails on an
optional import.[^note]

[^note]: Entries cloned from stale templates
[^pr-47]: The fit() torn-state fix
[^pr-52]: test_every_registered_entry_is_covered
