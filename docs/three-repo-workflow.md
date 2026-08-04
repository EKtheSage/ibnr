# Three-repo workflow: data → modeling → manuscript

**Status: proposal (2026-07-06).** How the CAS Schedule P database, this
package, and the `transformers_reserving` manuscript repo fit together for
the dependency-modeling-with-transformers research project (CAS individual
grant, Kang & De Virgilis).

## The three repos and their single responsibilities

```
cas-schedule-p-data-model          ibnr                                   transformers_reserving
─────────────────────────          ────                                   ──────────────────────
Data Vault warehouse               models + evaluation + harnesses        Quarto manuscript
raw Schedule P → gold mart         gallery: nn_transformer, sur,          experiments/ (thin runner,
versioned parquet publishes        copula_glm, meyers_ccl, ...            pins ibnr) → results/*.csv
_active_manifest.json              kernels: CRPS, PIT/KS, contracts       R/ggplot figures read the
                                   scripts/compare_gallery.py             CSVs; .qmd weaves the paper
        │                                   │                                      ▲
        └── parquet + manifest ────────────►└── tidy CSV artifacts ────────────────┘
            (path/env contract)                 (artifact contract)
```

- **Data** (`cas-schedule-p-data-model`): owns ingestion and the Data Vault
  model; its only public surface is the **gold mart publish** - a versioned
  parquet file named by `warehouse/_active_manifest.json`. Nothing downstream
  reaches past the mart.
- **Modeling** (this repo, package `ibnr`): owns model implementations, the
  evaluation kernels (implemented once), and the backtest harnesses. It is
  paper-agnostic: nothing in here should know about the manuscript.
- **Manuscript** (`transformers_reserving`): owns the research narrative,
  figures, and paper-specific experiment configuration. It runs experiments
  *through* ibnr and typesets *from* the artifacts.

## Contract 1 - data → modeling (in place, both transports)

`ibnr.data.schedule_p.load_schedule_p()` accepts (directly or via
`IBNR_SCHEDULE_P_WAREHOUSE`):

- a **local warehouse path** (producer-side dev: the sibling checkout, active
  publish resolved through `_active_manifest.json`);
- a **GitHub release spec** `github://EKtheSage/cas-schedule-p-data-model@<publish_id>`
  (consumer side: the data repo publishes each gold promote as an immutable
  release tagged with its publish_id via `pipeline/release.py`; the data repo
  is public, so ibnr fetches over anonymous HTTPS into `~/.cache/ibnr` - the
  `gh` CLI is a fallback, not a requirement - verifies sha256 against the
  release `manifest.json`, and reads locally thereafter).

The harness scripts stamp `mart_publish_id` into every results CSV, so any
figure in the paper traces to an exact data publish. Marco needs neither the
vault pipeline nor a warehouse clone: the ~5 MB release download is the entire
data dependency, and it needs no account, no token and no tooling. (The
`cas-schedule-p` PyPI package is the other zero-setup route - `pip install
cas-schedule-p` carries one gold publish inside the wheel, plus the Meyers
company screen in `cas_schedule_p.screens`.)

## Contract 2 - modeling → manuscript (the proposal)

### Step 0: put ibnr on GitHub

This repo is not under version control yet. `git init`, push to GitHub
(private is fine), tag releases. Without this there is nothing for the
manuscript repo to pin.

### Step 1: the manuscript repo grows an `experiments/` directory

A small uv-managed Python project inside `transformers_reserving`:

```
transformers_reserving/
  Code/            # existing R prototype (torch-for-R, ggplot adapters) - kept
  Manuscript/      # manuscript.qmd + references.bib (existing)
  experiments/     # NEW - thin Python runner
    pyproject.toml #   depends on: ibnr @ git+https://github.com/<org>/ibnr
    uv.lock        #   pins the exact ibnr commit → reproducible experiments
    configs/       #   paper-specific choices: lines, screens, seeds, model configs
    run_backtest.py  # calls ibnr's gallery/harness, writes ../results/*.csv
  results/         # NEW - committed CSV artifacts (small, versioned, diffable)
  References/
```

The runner is intentionally thin - it configures and calls
`ibnr.gallery.fit(...)` / the `compare_gallery` machinery; **no modeling code
lives in the manuscript repo**. When an experiment needs a model variant,
the variant is added to ibnr's gallery (with a card and tests) and the
manuscript repo bumps its pinned commit. `uv.lock` + the mart version stamp
make every figure reproducible from two pins.

### Step 2: figures stay in R, fed by the artifact contract

Marco's plotting investment (`plot_model_comparison.R`,
`model_plot_adapters.R`, `plotting_core.R`) survives untouched in spirit: the
adapters re-point from the R prototype's outputs to the tidy CSVs in
`results/`. The CSV layout is already tidy one-row-per-(model, line,
company) with estimate/se/cv/outcome/percentile/crps - designed to be read
by R as easily as Python. `manuscript.qmd` (Quarto handles R and Python
chunks side by side) reads `results/*.csv` in R chunks and renders ggplot
figures; no figure ever computes a model.

### Step 3: the R torch prototype's role

`transformer_torch.R` / `transformer_cumulative_torch.R` /
`multivariate.R` / `traditional.R` were the exploration that shaped the
research questions. Under this proposal the Python gallery entries
(`nn_transformer`, `sur`, `copula_glm`, `meyers_ccl`) become the canonical
experiment implementations - one engine, tested, with cards documenting every
modeling choice. The R files stay in `Code/` as the historical prototype (or
move to an `archive/`), and the manuscript's methods section cites the ibnr
cards. If an R-native artifact is ever required (e.g. a journal demands R),
`gallery.scaffold()`-style ejection of a single model is the path - not a
parallel implementation maintained by hand.

### Why artifact handoff instead of calling ibnr from the .qmd

- **Cost separation**: the pooled transformer fit is ~20 min CPU; you do not
  want that inside a Quarto render. Experiments run once, artifacts commit,
  the paper renders in seconds forever after.
- **Language neutrality**: CSVs keep the R/ggplot pipeline and any future
  Python figures equally happy.
- **Reviewability**: committed CSVs diff cleanly when a model or the data
  publish changes - a rerun that moves a number is visible in the PR.

## Division-of-labor summary

| concern | lives in |
|---|---|
| Schedule P ingestion, Data Vault, mart publishes | `cas-schedule-p-data-model` |
| model code, priors/architectures, eval metrics, harnesses, tests | `ibnr` |
| which experiments the paper runs (configs, seeds, company screens) | `transformers_reserving/experiments` |
| results artifacts (CSV), figures, prose | `transformers_reserving` |

## Concrete next actions

1. ~~`git init` + GitHub push of this repo; tag `v0.1.0`.~~ **Done 2026-07-07**
   (https://github.com/EKtheSage/ibnr).
2. ~~Add mart-version stamping to the harness CSVs.~~ **Done.**
3. ~~Data releases + `github://` consumption.~~ **Done**: data repo publishes
   gold promotes as releases (`pipeline/release.py`); ibnr consumes them via
   `load_schedule_p("github://EKtheSage/cas-schedule-p-data-model@<publish_id>")`.
4. Scaffold `experiments/` + `results/` in `transformers_reserving`; port the
   first experiment (the four-way paid-loss backtest) as `run_backtest.py`
   with a config file pinning the ibnr commit + mart publish_id; commit its CSV.
5. Re-point one existing ggplot figure (model comparison) at the CSV as the
   proof of the pipeline; then migrate the rest.
