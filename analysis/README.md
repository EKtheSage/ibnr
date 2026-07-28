# analysis

Exploratory notebooks. Not part of the package or its test suite - they read
the package and its artifacts (the validation CSVs in `analysis/results/`, the
local CAS Schedule P gold mart) and tell the story.

`results/` holds generated outputs (e.g. `scripts/meyers_validation.py` writes
its per-insurer CSVs here by default). Regenerable - safe to delete and rebuild.

## Running

```sh
uv sync --extra bayesian --extra nn   # BOTH: 03 fits the NN family as well as Stan
uv run jupyter lab analysis/
```

`--extra bayesian` alone is not enough for `03`, which fits `nn_transformer`,
`mdn`, `resnet` and `deeptriangle` and so needs torch from `--extra nn`
(`uv sync --all-extras` also works and additionally brings polars, chainladder
and altair, none of which the notebooks use). A notebook that fits `meyers_ccl`
also needs a working cmdstan install - see the repo CLAUDE.md.

**The gold mart is required, not optional, for `01` and `03`.** `01`'s
mart-dependent cells are guarded by `HAS_MART` and self-skip when it is not
reachable; `03` has no such guard - its panel, its fits and its board all come
from the mart, and it asserts the exact 10-company panel it expects, so without
the mart it fails at the third cell. `load_schedule_p` resolves and caches the
pinned GitHub release by default; point at a local warehouse with
`IBNR_SCHEDULE_P_WAREHOUSE` if it lives elsewhere.

## Notebooks

| notebook | what it covers |
|---|---|
| `01_triangle_and_meyers_ccl.ipynb` | Milestones 1-2: the Triangle layer & chainladder tie-out (both backends), the Schedule P adapter, a live CCL fit, and the 200-insurer Meyers validation (gross vs net of bulk). |
| `02_transformer_vs_statistical.ipynb` | Milestone 3: the NN family against the statistical dependence baselines, read off the published `compare_gallery*.csv` retrospectives over 152 cohorts. |
| `03_gallery_api_comparison.ipynb` | Milestone 6, live: every number from a real fit through `gallery.fit` -> `log_lik_at`/`predict_at` -> `align_panel` -> `gallery.leaderboard`, on a 25-cohort Schedule P panel at the 1997 cutoff, accident years 1990-1997. Reads no CSVs. Shows the two-capability split (ELPD for density-eligible entries, CRPS for all), what the intersection costs, PIT calibration (descriptive at cell level, tested at cohort level) and the multiline entries' ultimate-level appendix. |

To re-execute a notebook headless (e.g. after a data refresh):

```sh
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 analysis/01_triangle_and_meyers_ccl.ipynb
```

`03` fits the whole gallery live and takes about 20 minutes on the dev box (the
committed run's own timing table is the measurement, and `compartmental` alone is
a third of it), so it needs a much larger timeout than the others:

```sh
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 analysis/03_gallery_api_comparison.ipynb
```

**`03`'s committed run predates the 0.5.0 API fixes it motivated.** Building it
through the public API alone surfaced five gaps, all closed after it was
committed, so three things in its cells document friction that no longer exists:

* the `tri.with_expr(tri.expr.drop("company_name"))` cell, labelled "API
  FRICTION" - a pooled NN fit can now be scored on cells carrying all three of
  the mart's segments, so the drop is unnecessary;
* `point_row`'s `is_nn = gallery.get(name).family == "nn"` branch - `predict`
  and `realized_ultimates` take the same `segment` argument on every entry now,
  so the branch collapses to two unconditional lines;
* the `from ibnr.kernels.forecast import ...` / `from ibnr.kernels.holdout
  import next_diagonal` block - all five names are on `ibnr.gallery`.

All three still RUN correctly against the current code (checked by extracting
and executing exactly those calls), which is why the notebook is not re-executed:
its outputs are a 30-minute live gallery fit and re-running it to delete
three lines of workaround would change every timing in it.
