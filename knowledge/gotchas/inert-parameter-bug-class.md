---
type: Gotcha
title: The inert parameter bug class
description: "A parameter that is accepted and validated but never forwarded, and its siblings (a check that compares a set with itself, a check that binds two of three objects), recur in ibnr; test delivery through the public entry point and check against the authority."
tags: [testing, bug-class, review, mutation-testing]
status: stable
generated: { by: claude-code/claude-opus-5-5, at: 2026-10-05T23:00:00Z }
sources:
  - id: note
    resource: agent memory note inert-parameter-bug-class.md (private, outside the repository)
    title: Inert parameter bug class
    last_modified: 2026-07-29T23:36:39.999Z
  - id: pr-23
    resource: https://github.com/EKtheSage/ibnr/pull/23
    title: nuts_sampler dropped by the Bayesian entries' fit()
  - id: pr-74
    resource: https://github.com/EKtheSage/ibnr/pull/74
    title: GalleryDiagonal (the third variant and the test-stub finding)
---

# The bug class

A parameter that a public method accepts, validates, and then never forwards is a
recurring bug class in this repository. It was found in PR #23 (`nuts_sampler` dropped by
all five Bayesian entries' `fit()`).[^pr-23] An audit of the rest of the gallery
immediately turned up two more: `line_embedding_dim` in `nn/transformer_ml/config.py` (a
user-facing config setting the network never reads) and `n_lob` in
`sur/model.py::_mack_tail_variance`.[^note]

# Why it matters

It has the same shape as the four milestone-5 defects (the `a_ig` bound, the parity check
rounding, Clark's NaN gradient, `sd_ay` positivity): the code runs, produces plausible
output, and is labelled as if the requested thing happened. Nothing downstream can tell.
Tests written with `inspect.signature` are actively misleading here: they assert the far
end of a wire that may never have been connected, and they pass on the broken
implementation.[^note]

See [The Meyers a_ig bound is load-bearing](/gotchas/meyers-a-ig-bound.md) and [The
parity check rounded its z-scores away](/gotchas/parity-check-rounding.md) for two of
those milestone-5 defects.

# Sibling 1: a check that is internally consistent instead of right

This is the more expensive one, and it shipped three times in milestone 6 before the
lesson stuck:[^note]

* `index_into` rejected held-out cells that spanned cohorts. A single, entirely **wrong**
  cohort agrees with itself perfectly and was accepted.
* `_premium_by_origin` rejected premium spanning cohorts. Premium that was entirely
  another company's was accepted, with the contract still recording the right cohort.
* `next_diagonal` picked one diagonal date for several fields. Each field was
  self-consistent; the earliest won and the others vanished silently.

Every time, the fix is the same: compare against the **authority** (what the fit was
actually built on), never against the other members of the same set. Ask "consistent with
what?". If the answer is "itself", it is not a check.[^note]

# Sibling 2: a check that binds two of three objects

Found by review on 2026-07-29 (PR #74).[^pr-74] `GalleryDiagonal` takes a `MackFit`, a
`HoldoutCells` and a fitted entry. Two careful checks tied the cells to the fit, including
the good one, `prev_value == fit.latest` (the same number by two code paths). Nothing tied
the **entry** to either. An entry refitted on restated *interior* history, with the latest
diagonal untouched, passed everything: the total CDR mean moved from 0.27 to -367.69,
with seven times the spread, and every number finite.[^note]

Count the objects a call combines and check that every one of them is bound to something,
not just that some pair agrees. The pair that agrees is the pair you thought about.

Following that fix honestly may **narrow** a feature. Here it excluded the NN entries and
compartmental, whose contracts keep no raw cumulatives to compare, so they are refused by
name rather than assumed. A smaller verified surface beats a larger one resting on an
assumption already measured to be worth orders of magnitude.[^note]

# A test stub can hide an inert parameter

Found 2026-07-29, PR #74. `GalleryDiagonal.draw` derives the entry's seed from the
caller's `rng`, so `simulate_one_year_cdr(seed=)` stays the one control on
reproducibility. Replacing that with a constant makes `seed=` inert, and the mutation pass
missed it, because the two stubs written to make byte-equality possible both ignore the
injected `rng` on purpose. No output can distinguish the two cases: every draw is a
legitimate draw from the right distribution either way.[^pr-74]

A stub built for determinism cannot test that randomness is wired. A seed-delivery test
needs its own stub that actually consumes the `rng`, and the mutation pass (deliberately
breaking the code to see if a test fails) is what reveals the gap. Run it even when the
tests already look thorough.[^note]

# What to do

Test **delivery** through the real public entry point, not the private method and not the
signature. The cheap pattern that works:[^note]

1. Monkeypatch the private sink with a recorder.
2. Call the public method over a small synthetic fixture.
3. Assert the recorder saw the value.

Real data preparation and real dispatch run; nothing expensive does. Then confirm the test
has teeth by stashing the fix and watching the test fail. A test that passes both before
and after the fix is worse than no test, because it certifies the bug.

Related: [Rules for a fair model comparison](/decisions/model-comparison-rules.md).

[^note]: Inert parameter bug class
[^pr-23]: nuts_sampler dropped by the Bayesian entries' fit()
[^pr-74]: GalleryDiagonal (the third variant and the test-stub finding)
