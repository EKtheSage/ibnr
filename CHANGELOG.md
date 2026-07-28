# Changelog

This file starts at 0.5.0. For 0.1.0 through 0.4.0, read `git log v0.1.0..v0.4.0` -
those releases predate the file and reconstructing them now would be a summary of
a summary.

Versions follow [semantic versioning](https://semver.org/), loosely: while the
package is `Development Status :: 3 - Alpha`, a minor bump is free to change a
kernel signature. The public surface named in CLAUDE.md decision 8 (`Triangle`,
`gallery.list/fit/evaluate/stack/scaffold/leaderboard`) is the part treated as
stable, and nothing in it changed in this release.

## 0.5.0 - 2026-07-27

The 0.4.0 wheel on PyPI was 49 commits behind `main`, so this release is mostly a
catch-up: milestone 5 finished, milestones 6 and 7 opened, the one-year CDR landed
out of band, and the package moved to numpy 2 and grew a test CI.

### Packaging

* **Python 3.11 through 3.14.** `requires-python` is unchanged at `>=3.11` - it
  governs the core install, and the core resolves wheels-only on every one of
  3.11, 3.12, 3.13 and 3.14. The classifiers now say so.
* **Two extras carry a Python 3.12 ceiling, and it is declared rather than
  discovered at runtime.** `[bayesian]` and `[interop]` both transitively pin
  numpy below 2 - through `arviz` 0.18 (pulled by `bayesblend` 0.0.8) and through
  `bermuda-ledger` 2.3.0 - and the newest numpy under 2 is 1.26.4, which does not
  support Python 3.13 at all. Above 3.12 those extras now fail during dependency
  resolution with a message that names the reason, instead of installing
  something broken or silently omitting a package and crashing at import. Caveat
  worth knowing: on 3.12 `[bayesian]` installs but not wheels-only, because
  `bayesblend` pins `matplotlib==3.7.2` whose newest wheel is cp311, so 3.12
  builds it from source.
* **numpy 2.** Development and CI now run numpy 2.4.6; a plain `pip install ibnr`
  resolves numpy 2.5.1 and pandas 3.0.5. The core floor stays `numpy>=1.26`
  because raising it to 2 would make `ibnr[interop]` and `ibnr[bayesian]`
  unsatisfiable on PyPI - the blockers are upstream metadata on code that works,
  and this repo bridges them with `[tool.uv] override-dependencies` rather than
  shipping metadata nobody can install. Measured: the core suite gives the same
  548 passed / 0 failed on numpy 2.4.6 and on 2.5.1, both with pandas 3.0.5, and
  all extras together give 932 passed / 0 failed.
* **pymc 5.28.5 / pytensor 2.38.3** were not an optional upgrade. pytensor
  2.31.7 unpacks numpy's `einsum_path` result as a five-tuple and numpy 2.4
  returns three, so every LKJ-based compartmental test died on "not enough
  values to unpack" the moment numpy moved. Worth knowing before anyone pins
  pymc back.
* **`ibis-framework` is capped below 13.** 12.0.0 is what is locked and what every
  CI leg runs. Transforms are written around backend-specific ibis behaviour, so a
  major bump has to be a deliberate change with the dual-backend suite re-run.
* **`arviz<1` and `pymc<6`** in the `[bayesian]` extra. Both are measured breaks:
  arviz 1.x drops the `az.from_dict(posterior=...)` signature the parity tests
  use, and pymc 6.x changed `LKJCorrRV.rv_op`.
* **`chainladder>=0.9.2`** in `[interop]`, raised from 0.8.18 - the old floor let a
  fresh install resolve 0.8.26, which is not the version the tie-outs were
  validated against.
* **`py.typed`.** The package now ships the PEP 561 marker, so type checkers use
  the annotations instead of treating `ibnr` as untyped.

### Testing and CI

* **A pytest workflow, and gates that make a green run mean something.** Nine
  legs: core, core on 3.11, one per extra, everything at once, plus two that
  install without the lockfile and force numpy and pandas to their newest
  releases - the only legs that grade the resolution a downstream consumer
  actually gets. Each leg declares what it installed and which files it exists to
  exercise, and fails if a test was skipped because a package that leg installed
  could not be imported, if a named file contributed zero executed tests, or if
  the total falls below a floor.
* **`tests/test_import_purity.py`.** Walks every submodule under a blocker that
  refuses the optional extras, and checks the four public import paths pull in
  none of them. This is what makes "the core install stays light" a checked claim
  rather than a convention.
* A `test` dependency group carved out of `dev`, so a genuinely core-only
  environment can be built at all.

### Milestone 5 - cross-backend parity (complete)

* NumPyro and PyMC ports for `meyers_csr`, `england_verrall_odp`,
  `clark_growth_curve` and `compartmental`, each gated against the Stan reference
  posterior.
* **The parity gate was silently lenient.** It read `az.summary`, which rounds to
  three decimals, so any parameter whose MCSE rounded to zero scored a perfect
  z - precisely for the best-identified parameters. The published milestone-4 CCL
  figures came through this bug and are superseded.
* **`meyers_ccl`'s `a_ig` bound is load-bearing.** Stan declares
  `<lower=0, upper=1e5>` and both ports had left it unbounded, because the
  truncated *prior* mass is negligible - but the *posterior* piles into that
  corner at deep development lags and pulled `sig` 5-12% low. Restored in all four
  ports.
* `nuts_sampler` is exposed through every Bayesian entry's `fit()`.

### Milestone 6 - held-out scoring, stacking, leaderboard (in progress)

* `kernels/holdout.py` (which cells a fit at a given `as_of` is scored on),
  `kernels/densities.py` (the one place a density changes measure),
  `kernels/forecast.py` (the forecast object and the leaderboard) and
  `kernels/stacking.py` (bayesblend stacking over two panels).
* A forecast carries two independent capabilities - a density, giving ELPD, and
  draws, giving CRPS - each with its own panel membership, so one model's refusal
  cannot delete cells from a column it does not appear in.
* Held-out scorers and predictors for CSR, CCL, compartmental, guszcza, ODP,
  Clark, Mack and the NN family.

### Milestone 7 - the rest of the NN family (in progress)

* New entries: `mdn`, `deeptriangle`, `resnet`.
* A shared NN training scheme and one shared held-out mixin across all four NN
  entries.
* `kernels/tuning.py`: random-search hyper-parameter search over NN configs.

### New gallery entries

* `guszcza_growth_curve` - hierarchical growth curve reserving (Gesmann/Guszcza).
* `mdn`, `deeptriangle`, `resnet` (see above).

### One-year CDR (out of band)

* `kernels/cdr.py`: the Merz-Wuthrich 2008 analytic one-year claims development
  result per accident year and in total, plus an "actuary in the box" re-reserving
  simulation, plus VaR/TVaR of the CDR loss. Ties out to R ChainLadder's published
  `CDR()` output to seven decimal places.
* `kernels/mack.py` gained a native distribution-free chain ladder and
  `fit_mack_many`, a batch fit that closes the multi-cohort gap against
  chainladder-python's vectorized point estimate.

### Milestone 9 - speed benchmark (complete)

* `scripts/benchmark_speed.py` times construction, cumulative/incremental
  conversion, `as_of`, grain changes, aggregation, parquet ingestion and Mack fits
  against chainladder-python on both ibis backends. Headline: scale decides -
  chainladder wins small in-memory transforms, ibnr wins at mart scale and on
  every Mack fit.

### Fixes

* **`fit()` is atomic across every gallery entry.** A failed refit used to leave a
  half-updated entry behind; it now leaves the previous state untouched.
* **Null segment keys are rejected at ingestion.** Every join in
  `triangle/transforms.py` is a plain equi-join, and SQL join equality is false
  for `NULL = NULL`, so a single null segment value silently deleted a whole
  cohort from `as_of`, `latest_diagonal` and `to_incremental` - identically on
  both backends, and reachable from the real mart.
* Premium must match the loss cohort, and fields must share a diagonal.
* The zero-variance guard survives density columns that mix finite and `-inf`.
* Two inert parameters removed (`line_embedding_dim`, and `n_lob` in
  `_mack_tail_variance`) - both accepted, neither ever read.

### Tooling

* ruff floor raised to 0.16 across the repo, and CI checks formatting again.
* Python code blocks inside markdown are linted and formatted like source
  (`scripts/lint_md_snippets.py`), which also rejects doc samples that do not
  parse.
