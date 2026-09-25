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
| `03_gallery_api_comparison.ipynb` | Two things at once. (a) Milestone 6, live: every number from a real fit through `gallery.fit` -> `log_lik_at`/`predict_at` -> `align_panel` -> `gallery.leaderboard`, on a 25-cohort Schedule P panel at the 1997 cutoff, accident years 1990-1997. Reads no CSVs. Shows the two-capability split (ELPD for density-eligible entries, CRPS for all), what the intersection costs, PIT calibration (descriptive at cell level, tested at cohort level) and the multiline entries' ultimate-level appendix. (b) The gallery's **introduction to the neural family**: a section placed immediately before the NN fits opens `nn_transformer` up - the `nn_data` contract on the panel's own grid, `TriangleTransformer.__init__` and `forward` with the tensor shapes captured from a live forward pass, the measured parameter count, the MDN head and `mdn_nll`, the calendar-date validation split and the pinned per-dev standardization, and the cutoff augmentation. The source it quotes - the contract builder, `forward`, the MDN head, `mdn_nll`, `norm_stats`, `splits` and the augmentation - is printed with `inspect.getsource` from `src/ibnr/gallery/nn/` rather than retyped, so those seven blocks cannot drift from the code; the section's other eleven code cells are the notebook's own analysis of them, and drift there is caught only by re-executing. A follow-up after the fits shows the same cell's learned mixture (checked against `log_lik_at`) and the autoregressive rollout. |
| `04_nn_architectures_vs_classical.ipynb` | Two reserving architectures against classical benchmarks on the companion study's cohort set: 93 companies and 243 company-line pairs, accident years 1998-2007 valued at 2007-12-31, with calendar 2008 held out. `tlrn` is fitted twice (paid only, thirteen features) with forty members each, trained in four worker processes and averaged, beside the pooled neural entries and `mack`/`mcl`/`sur`; the study's own rule, the best two of ten members, is read off the stored member reserves, a section measures how much the choice and the number of members moves the result, and another breaks the company board down by all seven of the study's point metrics, by line, by accident year, by premium quartile and company by company. Every model is scored at three levels: the next calendar diagonal, the 120-month endpoint per company-line pair, and the company reserve (Pool_APE over 82 companies, printed beside the reference replay's four constants). Asserts that `mcl` and `mack` reproduce the reference company reserves to 1e-6 relative, and the reference implementation's own tables of the seven metrics, of Pool_APE by line and (for `mcl`) of the errors by accident year to 1e-9. Pinned to ibnr 0.7.1 from PyPI, `cas-schedule-p` 2026.6.13 and gold publish `20260613_041006`. The committed run took 618 minutes. |

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

`04` is written against the published package, not this checkout. Install
`"ibnr[nn]==0.7.1" cas-schedule-p==2026.6.13 matplotlib` into a fresh Python 3.11 or
3.12 environment and run it from any directory; after the install it needs no network.
The committed run used `IBNR_NB04_PROTOCOL=published`, the default, and took 618 minutes
on the dev laptop, so give it no timeout. nbconvert writes the notebook only when it
finishes, so a restart part way through loses the whole run:

```sh
IBNR_NB04_PROTOCOL=published jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 04_nn_architectures_vs_classical.ipynb
```

`IBNR_NB04_PROTOCOL=smoke` runs the same cells on a few-seed, few-epoch budget in about
an hour. It checks that the notebook runs end to end, and none of its numbers may be
quoted.

**`03` motivated the 0.5.0 API fixes, and as of 0.5.2 it is written against
them.** Building it through the public API alone surfaced five gaps; three
showed up as workarounds in its own cells. Two are gone. **The third is still
load-bearing, and that was established by removing it and watching the
re-execution fail** - see the long comment on the `drop("company_name")` cell,
which now says exactly which two call paths still need one schema and why
0.5.0's narrowing does not reach them (`index_into` demands exact schema
equality by design, and the pooled contract carries no single-cohort identity
to verify a dropped value against). The two that are gone:

* `point_row`'s `is_nn = gallery.get(name).family == "nn"` branch - `predict`
  and `realized_ultimates` take the same `segment` argument on every entry now,
  so the branch collapsed to two unconditional lines;
* the `from ibnr.kernels.forecast import ...` / `from ibnr.kernels.holdout
  import next_diagonal` block - all five names are on `ibnr.gallery`, which is
  the export set this notebook is the motivating caller for.

The notebook was re-executed end to end on 0.5.2 to land those changes, so its
outputs, its timings and the `ibnr` version it prints are all from one run
against current code. Every timing in it therefore differs from the 0.5.0 run;
the numbers that are not timings should not, and a diff that shows an estimate
moving is worth reading rather than waving through.
