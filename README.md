# ibnr

Gallery-centric probabilistic loss reserving: Bayesian MCMC and neural network
reserving methods with mandatory evaluation, model stacking, and a long-format
triangle data layer backed by duckdb and polars (via ibis). A companion to
[chainladder-python](https://github.com/casact/chainladder-python), not a fork.

## Status

Early development.

**Triangle layer**: a `Triangle` is a tidy long table — `origin_period,
dev_lag, eval_date, field, value` plus arbitrary segment columns — with
transformations (cumulative/incremental, grain changes, `as_of()` backtesting
slices) written once in ibis and tested against both the duckdb and polars
backends, tying out to chainladder-python on the public raa/clrd samples.

```python
import chainladder as cl
from ibnr import Triangle

t = Triangle.from_chainladder(cl.load_sample("raa"))
t.to_incremental().to_wide()
t.as_of("1985-12-31")          # the triangle as known at year-end 1985
t.to_chainladder()             # lossless round-trip
```

**Gallery (Bayesian)**: `meyers_ccl` — Meyers' Correlated Chain Ladder in
Stan, fit/predict/evaluate through the mandatory `GalleryEntry` contract,
producing a `PredictiveDistribution` of ultimates. Requires the `[bayesian]`
extra and a cmdstan installation.

```python
from ibnr import gallery

entry = gallery.fit("meyers_ccl", triangle, as_of="1997-12-31")
pred = entry.predict()                       # ultimates by origin + total
pred.summary(observed=entry.realized_ultimates(triangle))  # Meyers-style table
print(gallery.get("meyers_ccl").card())     # the model card

# the monograph's retrospective validation (PIT uniformity across insurers)
# uv run python scripts/meyers_validation.py --per-line 50
```

**Gallery (NN + statistical)**: `nn_transformer` — a PyTorch masked-cell
triangle transformer with a mixture density head and deep ensembling, trained
pooled across every company × line of business (`[nn]` extra) — alongside two
classical multivariate dependence baselines: `sur` (Zhang's multivariate chain
ladder via feasible GLS) and `copula_glm` (Shi & Frees' copula-linked
lognormal regressions). All three produce the same `PredictiveDistribution`
and are compared head-to-head by `scripts/compare_gallery.py` (KS/PIT
calibration + CRPS on the Meyers retrospective protocol). See
`analysis/02_transformer_vs_statistical.ipynb` for the comparison analysis.

## Data: the CAS Schedule P gold mart

Real-data fitting and the `-m mart` tests read the **gold mart published by
[`cas-schedule-p-data-model`](https://github.com/EKtheSage/cas-schedule-p-data-model)**
(Ethan's CAS Schedule P database; Data Vault warehouse with versioned gold
publishes). This package never touches raw Schedule P — it consumes only the
published mart, from either source:

```python
# GitHub release (recommended for consumers; needs `gh auth login` once —
# the repo is private). Downloads ~5 MB to ~/.cache/ibnr, sha256-verified,
# then reads locally forever after:
tri = load_schedule_p("github://EKtheSage/cas-schedule-p-data-model@20260613_041006")

# local warehouse checkout (producer-side dev):
tri = load_schedule_p("../cas-schedule-p-data-model/warehouse")
```

Both forms also work through the `IBNR_SCHEDULE_P_WAREHOUSE` environment
variable; `IBNR_CACHE_DIR` relocates the release cache. Each data-repo gold
promote is published as an **immutable release tagged with its `publish_id`**
carrying every gold table plus a `manifest.json` (asset, sha256, bytes) — the
same `publish_id` the harness scripts stamp into every results CSV, so any
figure traces to an exact publish.

The mart of record is `mart_reserving_model_training`: 150+ companies × 4
Schedule P lines, accident years 1988–1997, dev ages 1–10, USD thousands;
fields `cum_paid_loss`, `incurred_loss`, `bulk_loss`,
`earned_prem_net/direct`, with `reported_loss = incurred − bulk` derived by
the adapter (`src/ibnr/data/schedule_p.py`). Everything mart-dependent
auto-skips when no data source is available — the package and its test suite
work standalone on the public raa/clrd samples.

## Related repositories

Three-repo research setup (see `docs/three-repo-workflow.md` for the full
integration proposal):

| repo | role |
|---|---|
| `cas-schedule-p-data-model` | data: Data Vault warehouse → versioned gold mart publishes |
| `probabilistic-ml-reserving` (this repo, package `ibnr`) | modeling: gallery entries, eval kernels, backtest harnesses |
| [`transformers_reserving`](https://github.com/marcopark90/transformers_reserving) | research manuscript (CAS grant, Quarto): consumes experiment artifacts produced here |

## Development

```sh
uv sync                                  # core + dev deps
uv run pytest                            # full suite (both ibis backends)
uv run pytest -m "not tieout and not mart"   # fast unit tests only
uv run ruff check . && uv run ruff format --check .
```

Optional extras: `[bayesian]` (cmdstanpy, numpyro, pymc, arviz, bayesblend),
`[nn]` (torch), `[viz]` (altair). The core depends only on
`ibis-framework[duckdb,polars]`.

License: MPL-2.0.
