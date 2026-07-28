# ibnr

Gallery-centric probabilistic loss reserving: Bayesian MCMC and neural network
reserving methods with mandatory evaluation, model stacking, and a long-format
triangle data layer backed by duckdb and polars (via ibis). A companion to
[chainladder-python](https://github.com/casact/chainladder-python), not a fork.

## Status

Early development.

**Triangle layer**: a `Triangle` is a tidy long table - `origin_period,
dev_lag, eval_date, field, value` plus arbitrary segment columns - with
transformations (cumulative/incremental, grain changes, `as_of()` backtesting
slices) written once in ibis and tested against both the duckdb and polars
backends, tying out to chainladder-python on the public raa/clrd samples.

```python
import chainladder as cl
from ibnr import Triangle

t = Triangle.from_chainladder(cl.load_sample("raa"))  # needs the [interop] extra
t.to_incremental().to_wide()
t.as_of("1985-12-31")  # the triangle as known at year-end 1985
t.to_chainladder()  # lossless round-trip
```

To build one from your own data instead, see [Building a Triangle from your own
data](#building-a-triangle-from-your-own-data) - `Triangle.from_long` needs no
extras and is what every other constructor funnels into.

**Gallery (Bayesian)**: `meyers_ccl` - Meyers' Correlated Chain Ladder in
Stan, fit/predict/evaluate through the mandatory `GalleryEntry` contract,
producing a `PredictiveDistribution` of ultimates. Requires the `[bayesian]`
extra and a cmdstan installation.

```python
from ibnr import gallery

entry = gallery.fit("meyers_ccl", triangle, as_of="1997-12-31")
pred = entry.predict()  # ultimates by origin + total
pred.summary(observed=entry.realized_ultimates(triangle))  # Meyers-style table
print(gallery.get("meyers_ccl").card())  # the model card

# the monograph's retrospective validation (PIT uniformity across insurers)
# uv run python scripts/meyers_validation.py --per-line 50
```

**Gallery (NN + statistical)**: `nn_transformer` - a PyTorch masked-cell
triangle transformer with a mixture density head and deep ensembling, trained
pooled across every company × line of business (`[nn]` extra) - alongside two
classical multivariate dependence baselines: `sur` (Zhang's multivariate chain
ladder via feasible GLS) and `copula_glm` (Shi & Frees' copula-linked
lognormal regressions). All three produce the same `PredictiveDistribution`
and are compared head-to-head by `scripts/compare_gallery.py` (KS/PIT
calibration + CRPS on the Meyers retrospective protocol). See
`analysis/02_transformer_vs_statistical.ipynb` for the comparison analysis.

**Gallery (deterministic) + one-year CDR**: `mack` - the distribution-free
chain ladder (Mack 1993) computed natively over `Triangle`, with the
**one-year claims development result** of Merz & Wuthrich (2008) on top: how
far next year's re-estimate can move, which is the Solvency II reserve-risk
view rather than the full run-off one. Core install, no extras.

```python
entry = gallery.fit("mack", triangle, loss_field="paid_loss")
entry.summary()  # latest, ultimate, IBNR, Mack run-off S.E.
entry.one_year_cdr().summary()  # + one-year S.E. and its share of run-off risk
entry.cdr_distribution(n_draws=20_000)  # the same by re-reserving, with quantiles

from ibnr.kernels.cdr import cdr_risk_measures

cdr_risk_measures(entry.cdr_distribution(n_draws=100_000, seed=1))  # VaR/TVaR 99.5
```

The analytic msep ties out to R ChainLadder's published `CDR()` output on the
`MW2014` triangle to seven decimals, and the closed form and the re-reserving
simulation agree to Monte Carlo error. chainladder-python has no CDR at all,
so nothing here is delegated to it.

## Installation

```sh
uv add ibnr            # or: pip install ibnr
```

The core install is **`ibis-framework[duckdb]` + numpy + pandas + scipy** and
covers the triangle layer, the `statistical` and
`deterministic` gallery entries, and the evaluation kernels. polars is *not* in
it: the second ibis backend is fully supported but its ~176 MB runtime is 35% of
the install that a duckdb-only caller never executes, so since 0.4.0 it sits
behind an extra - which matters when you are sizing a serverless deployment. The
heavier methods sit behind optional extras too:

```sh
uv add "ibnr[bayesian]"   # cmdstanpy, numpyro, pymc, arviz, bayesblend
uv add "ibnr[nn]"         # torch
uv add "ibnr[viz]"        # altair
uv add "ibnr[interop]"    # chainladder + bermuda, for to_chainladder()/to_bermuda()
uv add "ibnr[polars]"     # the second ibis backend (duckdb is the default)
```

> **`[bayesian]` installs cmdstanpy, not CmdStan itself.** The Stan entries
> compile their `model.stan` at runtime, so a CmdStan toolchain must be present.
> Install it once with `python -m cmdstanpy.install_cmdstan` - this needs a C++
> toolchain (RTools on Windows, `build-essential`/Xcode command-line tools on
> Linux/macOS). The bundled `Dockerfile` ships CmdStan with every gallery Stan
> model pre-compiled if you would rather not set this up locally.

## Building a Triangle from your own data

`Triangle.from_long` is the single ingestion path: `from_chainladder`,
`from_bermuda` and `load_schedule_p` all reshape their input into a long frame
and hand it here, so whatever they can express your own data can too. It is part
of the core install and needs no extras.

A **cell** is `(segments..., origin_period, dev_lag, eval_date, field)` carrying
one `value`:

| column | type | meaning |
|---|---|---|
| `origin_period` | date | first day of the accident/underwriting period |
| `dev_lag` | int | **months from the origin period's start**, counting the valuation month itself - so the first annual diagonal is 12, not 0 |
| `eval_date` | date | last day of the month the cell was valued at; a stored column, not derived, because `as_of()` backtesting slices on it |
| `field` | str | which measure the row carries: `paid_loss`, `reported_loss`, `earned_premium`, ... |
| `value` | float | the number |
| anything else | | a **segment**: the cohort key (`lob`, `company`, ...) |

Absent means unobserved: rows with a null `value` are dropped rather than
stored, and nothing is densified, so the unobserved half of the square simply is
not there. Zero, by contrast, is an explicit observation and is kept.

```python
import datetime as dt

import pandas as pd

from ibnr import Triangle

# cumulative paid loss by accident year, at development ages 12, 24, 36... months
paid = {
    2018: [400, 660, 790, 870, 922],
    2019: [830, 1290, 1560, 1750],
    2020: [1190, 1930, 2380],
    2021: [1620, 2510],
    2022: [2050],
}
premium = {2018: 2000, 2019: 4100, 2020: 5900, 2021: 8000, 2022: 10200}


def year_end(origin_year: int, dev_lag: int) -> dt.date:
    """dev_lag counts the valuation month, so age 12 on a 2018 origin is 2018-12-31."""
    return dt.date(origin_year + dev_lag // 12 - 1, 12, 31)


rows = [
    {
        "lob": "auto",  # a segment column: the cohort key
        "origin_period": dt.date(year, 1, 1),
        "dev_lag": 12 * (d + 1),  # MONTHS from the origin period's start
        "eval_date": year_end(year, 12 * (d + 1)),
        "field": "paid_loss",
        "value": float(value),
    }
    for year, values in paid.items()
    for d, value in enumerate(values)
]
# premium is just another field, booked once per origin at its first evaluation
rows += [
    {
        "lob": "auto",
        "origin_period": dt.date(year, 1, 1),
        "dev_lag": 12,
        "eval_date": year_end(year, 12),
        "field": "earned_premium",
        "value": float(value),
    }
    for year, value in premium.items()
]

tri = Triangle.from_long(pd.DataFrame(rows), segments=["lob"], measure="cumulative")
print(tri)
print(tri.select_fields("paid_loss").to_wide())
```

```
Triangle(grain=OYDY, measure=cumulative, units=None, segments=[lob])
dev_lag            12      24      36      48     60
origin_period
2018-01-01      400.0   660.0   790.0   870.0  922.0
2019-01-01      830.0  1290.0  1560.0  1750.0    NaN
2020-01-01     1190.0  1930.0  2380.0     NaN    NaN
2021-01-01     1620.0  2510.0     NaN     NaN    NaN
2022-01-01     2050.0     NaN     NaN     NaN    NaN
```

`measure` (`"cumulative"` or `"incremental"`), `origin_grain` and `dev_grain`
(`"Y"`, `"Q"`, `"M"`) and `units` are metadata the transforms and every gallery
entry trust; they default to cumulative annual/annual. Pass
`dev_lag_unit="periods"` if your dev column counts development *years* (1, 2,
3...) rather than months - it is multiplied by the dev grain on the way in.

**Accepted inputs.** Anything ibis can register: a pandas DataFrame, a polars
DataFrame, a pyarrow Table, an ibis table expression, or a `str`/`Path` to a
parquet file (read straight by the backend, never through pandas). An ibis
expression keeps its own backend and is not re-registered.

```python
tri = Triangle.from_long("losses.parquet", segments=["lob"])
```

**Your own column names.** The five core names are keyword arguments, so nothing
has to be renamed upstream:

```python
tri = Triangle.from_long(
    my_frame,
    origin="accident_year",
    dev="age_months",
    eval_date="valued_at",
    field="measure",
    value="amount",
    segments=["lob"],
)
```

**Wide input.** If your measures are columns rather than a `field`/`value` pair,
name them with `fields=[...]` and they are unpivoted for you. Every column that
is not a measure and not one of the three key columns is treated as a segment:

```python
# columns: lob | origin_period | dev_lag | eval_date | paid_loss | earned_premium
tri = Triangle.from_long(wide_frame, fields=["paid_loss", "earned_premium"], segments=["lob"])
```

`segments=[...]` restricts which extra columns are kept (default: all of them).
Drop the ones that do not identify a cohort - a stray column splits cells that
should have been one.

**Segment values must be non-null, and ingestion refuses them.** Every transform
(`as_of`, `latest_diagonal`, `to_cumulative`, `to_incremental`) equi-joins on the
segment columns, and SQL join equality is false for `NULL = NULL`, so one null
segment value silently *deletes* that cohort - no error, no warning, and a clean
`validate()`. Give those rows an explicit value (`"unknown"`), drop them, or
leave the column out with `segments=[...]`:

```
ValueError: null segment key in lob (1 rows). Segment columns identify the cohort, so a
null in one names no cohort and is silently dropped by every transform that joins on it
(as_of, latest_diagonal, to_cumulative, to_incremental) - the cohort would disappear from
results with no error. ...
```

**Which field is the loss, which is the premium.** Gallery entries do not guess:
each `fit()` takes `loss_field=` and (where the model has an exposure term)
`premium_field=`. The defaults are the Schedule P mart's names, and
`premium_field` is `"earned_premium"` on every entry that has it - but
`loss_field` is **not** one value across the gallery, so omitting it quietly
picks a basis for you:

| `loss_field` default | entries |
|---|---|
| `"reported_loss"` | `meyers_ccl`, `mdn`, `nn_transformer`, `nn_transformer_ml`, `resnet` |
| `"paid_loss"` | the other ten: `clark`, `clark_growth_curve`, `compartmental`, `copula_glm`, `deeptriangle`, `england_verrall_odp`, `guszcza_growth_curve`, `mack`, `meyers_csr`, `sur` |

`deeptriangle` is the one to watch: it is an NN entry but keeps the paper's
paid-loss basis, so "the NN entries are reported-basis" is true of four of the
five and wrong for that one. Pass `loss_field=` explicitly whenever the basis
matters - which is always, if you are comparing entries to each other.

Premium is genuinely required by every entry that models a loss *ratio* or
carries a log-premium offset: `meyers_ccl`, `meyers_csr`,
`guszcza_growth_curve`, `clark_growth_curve`, `compartmental`,
`england_verrall_odp`, `copula_glm` and all five NN entries. Only `mack` and
`sur` have no `premium_field` argument at all. A premium field that is missing,
duplicated per origin, non-positive, or belongs to a different cohort than the
losses is an error at fit time rather than a silent zero (`ValueError: no rows
for premium field 'earned_premium'`) - except in the NN entries, which train
pooled across many cohorts and so *drop* an unusable cohort from the pool
instead of refusing the whole fit.

`clark` needs premium under its default `method="cape_cod"` (`U[w] = ELR *
premium[w]`), not under `method="ldf"`, which estimates a free ultimate per
origin and neither reads nor resolves premium - on a triangle of losses only,
`gallery.fit("clark", losses, method="ldf")` just works.

**Bringing your own connection.** `backend=` takes `"duckdb"` (the default),
`"polars"` (needs the `[polars]` extra), or an already-connected ibis backend -
which is how you point ingestion at a persistent database, a tuned duckdb, or a
connection shared with the rest of your application:

```python
import ibis

con = ibis.duckdb.connect("warehouse.ddb")
tri = Triangle.from_long("losses.parquet", segments=["lob"], backend=con)
```

**Getting the frame back.** There is no `to_long`. The triangle *is* the long
frame, so `tri.to_pandas()` / `tri.to_polars()` materialize it in the schema
above, `tri.expr` hands you the underlying ibis expression to push further work
into the engine, and `tri.to_wide(field)` pivots one field to an origin x dev
matrix for display. `to_polars()` needs the `[polars]` extra even on a duckdb
triangle: the call goes straight through to ibis, so on the core install it
raises a bare `ModuleNotFoundError: No module named 'polars'` rather than the
install hint you get from `backend="polars"`. `to_pandas()` is always
available.

## Using the gallery

Two things surprise every first caller. The first is that `import ibnr` does not
give you `ibnr.gallery`:

```
>>> import ibnr
>>> ibnr.gallery
AttributeError: module 'ibnr' has no attribute 'gallery'
```

The second is that `gallery.get` hands back a class, not a fitted model:

```python
from ibnr import gallery  # the import that works

gallery.list()  # names of every registered entry
entry_cls = gallery.get("mack")  # a CLASS, not an instance or a fitted model
fitted = entry_cls().fit(tri, loss_field="paid_loss")  # so: instantiate, then fit
fitted = gallery.fit("mack", tri, loss_field="paid_loss")  # the same, in one call
print(fitted.summary())
```

```
       origin  latest      ultimate         ibnr   runoff_se
0  2018-01-01   922.0    922.000000     0.000000    0.000000
1  2019-01-01  1750.0   1854.597701   104.597701   24.926810
2  2020-01-01  2380.0   2812.043629   432.043629   49.558833
3  2021-01-01  2510.0   3615.332407  1105.332407   78.222432
4  2022-01-01  2050.0   4670.333208  2620.333208  149.225144
5       total  9612.0  13874.306946  4262.306946  233.743645
```

`ibnr/__init__.py` exports only `Triangle`, `TriangleMeta` and `__version__`;
`gallery` is a submodule, and a submodule is an attribute of its package only
once something has imported it. `gallery.get` mirrors that literalness: it
returns the registered class so you can read `.card()` or `.family` without
constructing anything, which is why the call is `gallery.get(name)()`.

Evaluation is per entry, not a module-level call: `fitted.evaluate(observed)`
scores realized outcomes against the predictive distribution, and
`gallery.leaderboard(...)` / `gallery.stack(...)` combine several fitted entries
over a shared panel.

## Data: the CAS Schedule P gold mart

Real-data fitting and the `-m mart` tests read the **gold mart published by
[`cas-schedule-p-data-model`](https://github.com/EKtheSage/cas-schedule-p-data-model)**
(Ethan's CAS Schedule P database; Data Vault warehouse with versioned gold
publishes). This package never touches raw Schedule P - it consumes only the
published mart, from either source:

```python
# default - no argument needed: the newest GitHub release of the data repo
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
carrying every gold table plus a `manifest.json` (asset, sha256, bytes) - the
same `publish_id` the harness scripts stamp into every results CSV, so any
figure traces to an exact publish.

The mart of record is `mart_reserving_model_training`: 150+ companies × 4
Schedule P lines, accident years 1988-1997, dev ages 1-10, USD thousands;
fields `cum_paid_loss`, `incurred_loss`, `bulk_loss`,
`earned_prem_net/direct`, with `reported_loss = incurred - bulk` derived by
the adapter (`src/ibnr/data/schedule_p.py`). Everything mart-dependent
auto-skips when no data source is available - the package and its test suite
work standalone on the public raa/clrd samples.

## Parallel retrospectives & the compute container

`ibnr.kernels.harness` is the compute layer for every study script (and the
seam a future hosted scoring API will call): it fans company×line fits across
a process pool - **all visible cores by default**, `IBNR_MAX_WORKERS` or
`--workers` to override - and runs a staged sampler-escalation policy: a cheap
first pass, then a re-fit at expensive settings (monograph `adapt_delta`,
parallel chains) *only* for companies failing the convergence gates
(R-hat / divergences / bulk ESS). Every results row records which `stage` it
came from. `--serial` and `--no-escalate` reproduce the sequential
single-stage behavior of the published runs.

```sh
uv run python scripts/meyers_validation.py --model compartmental --per-line 50   # parallel + escalation, by default
```

The `Dockerfile` packages all of this as a self-contained compute image -
package, cmdstan and **every gallery Stan model pre-compiled** - so other
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
| `ibnr` (this repo) | modeling: gallery entries, eval kernels, backtest harnesses |
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
`[interop]` (chainladder, bermuda-ledger), `[polars]` (the second ibis backend),
`[nn]` (torch), `[viz]` (altair). The core depends only on
`ibis-framework[duckdb]` + scipy; duckdb is the default backend and needs nothing
extra.

License: MPL-2.0 - see [LICENSE](LICENSE).
