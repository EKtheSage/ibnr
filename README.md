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
# default — no argument needed: the newest GitHub release of the data repo
# (needs `gh auth login` once; the repo is private). @latest resolves to a
# concrete publish_id, downloads ~5 MB to ~/.cache/ibnr, sha256-verified,
# then reads locally forever after:
tri = load_schedule_p()

# pin an exact publish (what experiment runs should do):
tri = load_schedule_p("github://EKtheSage/cas-schedule-p-data-model@20260613_041006")

# local warehouse checkout (producer-side dev override):
tri = load_schedule_p("../cas-schedule-p-data-model/warehouse")
```

Resolution order: explicit argument > `IBNR_SCHEDULE_P_WAREHOUSE` environment
variable (either form) > the `@latest` GitHub release. `IBNR_CACHE_DIR`
relocates the release cache. Each data-repo gold
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

## Parallel retrospectives & the compute container

`ibnr.kernels.harness` is the compute layer for every study script (and the
seam a future hosted scoring API will call): it fans company×line fits across
a process pool — **all visible cores by default**, `IBNR_MAX_WORKERS` or
`--workers` to override — and runs a staged sampler-escalation policy: a cheap
first pass, then a re-fit at expensive settings (monograph `adapt_delta`,
parallel chains) *only* for companies failing the convergence gates
(R-hat / divergences / bulk ESS). Every results row records which `stage` it
came from. `--serial` and `--no-escalate` reproduce the sequential
single-stage behavior of the published runs.

```sh
uv run python scripts/meyers_validation.py --model compartmental --per-line 50   # parallel + escalation, by default
```

The `Dockerfile` packages all of this as a self-contained compute image —
package, cmdstan and **every gallery Stan model pre-compiled** — so other
services can call it with zero startup cost:

```sh
docker build -t ibnr .
docker run --rm -e GH_TOKEN=<token> -e IBNR_MAX_WORKERS=8 --cpus 8 \
  -v ibnr-cache:/data/ibnr-cache -v "$PWD/results:/app/analysis/results" \
  ibnr python scripts/meyers_validation.py --model compartmental --per-line 50
```

`GH_TOKEN` authenticates the gold-mart release download (private data repo);
set `IBNR_MAX_WORKERS` to match `--cpus`, since a cpu-limited container still
reports the host's core count to Python.

## Related repositories

Three-repo research setup (see `docs/three-repo-workflow.md` for the full
integration proposal):

| repo | role |
|---|---|
| `cas-schedule-p-data-model` | data: Data Vault warehouse → versioned gold mart publishes |
| `probabilistic-ml-reserving` (this repo, package `ibnr`) | modeling: gallery entries, eval kernels, backtest harnesses |
| [`transformers_reserving`](https://github.com/marcopark90/transformers_reserving) | research manuscript (CAS grant, Quarto): consumes experiment artifacts produced here |

## Documentation

The API documentation site is generated with
[great-docs](https://github.com/posit-dev/great-docs) (Quarto-based) from
`great-docs.yml` plus the package docstrings. Requires the `quarto` CLI on your
PATH.

```sh
uv run --no-default-groups --group docs great-docs build      # -> great-docs/_site
uv run --no-default-groups --group docs great-docs preview    # serve locally
```

The `great-docs/` build directory is gitignored; only `great-docs.yml` is
tracked. `.github/workflows/docs.yml` rebuilds the site on every push and
deploys `main` to GitHub Pages.

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
