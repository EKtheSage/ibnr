# analysis

Exploratory notebooks. Not part of the package or its test suite - they read
the package and its artifacts (the validation CSVs in `analysis/results/`, the
local CAS Schedule P gold mart) and tell the story.

`results/` holds generated outputs (e.g. `scripts/meyers_validation.py` writes
its per-insurer CSVs here by default). Regenerable - safe to delete and rebuild.

## Running

```sh
uv sync --extra bayesian          # notebooks need the gallery + jupyter tooling
uv run jupyter lab analysis/
```

A notebook that fits `meyers_ccl` needs a working cmdstan install (see the repo
CLAUDE.md). Cells guarded by `HAS_MART` self-skip when the gold mart isn't
reachable; point at it with `IBNR_SCHEDULE_P_WAREHOUSE` if it lives elsewhere.

## Notebooks

| notebook | what it covers |
|---|---|
| `01_triangle_and_meyers_ccl.ipynb` | Milestones 1-2: the Triangle layer & chainladder tie-out (both backends), the Schedule P adapter, a live CCL fit, and the 200-insurer Meyers validation (gross vs net of bulk). |
| `02_transformer_vs_statistical.ipynb` | Milestone 3: the NN family against the statistical dependence baselines, read off the published `compare_gallery*.csv` retrospectives over 152 cohorts. |
| `03_gallery_api_comparison.ipynb` | Milestone 6, live: every number from a real fit through `gallery.fit` -> `log_lik_at`/`predict_at` -> `align_panel` -> `gallery.leaderboard`, on a 25-cohort Schedule P panel at the 1997 cutoff. Reads no CSVs. Shows the two-capability split (ELPD for density-eligible entries, CRPS for all), what the intersection costs, PIT calibration and the multiline entries' ultimate-level appendix. |

To re-execute a notebook headless (e.g. after a data refresh):

```sh
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 analysis/01_triangle_and_meyers_ccl.ipynb
```

`03` fits the whole gallery live and takes about 30 minutes on the dev box (the
committed run's own timing table is the measurement), so it needs a much larger
timeout than the others:

```sh
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 analysis/03_gallery_api_comparison.ipynb
```
