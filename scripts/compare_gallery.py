"""Head-to-head gallery backtest on the Schedule P gold mart.

Compares the transformer (nn_transformer) against the classical multivariate
baselines (sur, copula_glm) - and optionally meyers_ccl - on the Meyers
retrospective protocol: train as of 1997-12-31, score realized ultimates,
report outcome percentiles (KS uniformity) and CRPS per model.

Company set: companies passing the monograph's Table A.1 screens on AT
LEAST TWO requested lines, each scored on exactly its passing lines - the
same (company, line) pairs for every model, including the transformer's
training pool (no monoline data anywhere; Ethan's comparability rule). The
headline field is paid_loss - the copula's lognormal marginals cannot take
the negative late increments of reported losses.

Fit shapes per model:
  sur, copula_glm   one fit per company (its screened lines jointly)
  nn_transformer    ONE pooled fit on the screened (company, line) pairs,
                    then per-(company, line) segment predicts
  meyers_ccl        one cmdstan fit per company x line (slow; opt-in)
  chain_ladder      volume-weighted CL point estimate per (company, line),
                    from kernels.mack.fit_mack - the distribution-free skill
                    benchmark (always included)

Every row carries `anchor` (loss-to-date at the as_of date) and `premium`
(line premium), so downstream analysis can score on the RESERVE basis:
ultimate errors flatter models for data they merely copied forward.

METHODOLOGY (this script produces published results - read before changing)

1. Cohort-data parity. Every model is scored on the SAME (company, line)
   cells, and the pooled NN entries train on exactly those scored pairs
   (`--nn-pool multiline`, the default). Letting one model see a wider or
   narrower pool turns a calibration comparison into a data comparison;
   the alternate pools exist only to *measure* that effect, never as the
   headline. See `screened_company_lines` and `pair_filter`.

2. Training slice vs realized outcomes. Models fit `tri_all.as_of(args.as_of)`
   - the upper triangle visible at 1997-12-31. Outcomes come from the FULL
   triangle (the realized bottom-right). CRITICAL GOTCHA (CLAUDE.md): the
   Schedule P mart carries accident years past the study window (1988-2007,
   not 1988-1997), so any outcome or aggregate read off the full triangle
   MUST be restricted to the origins present in the training slice. Omitting
   that restriction silently sums post-study accident years - it bit
   `point_context` once and inflated "outcomes" by 2.4x. The guard lives in
   `point_context` (the `origin_period.isin(grid.index)` filter); entries'
   own `realized_ultimates()` apply the equivalent restriction internally.

3. Point scoring is RESERVE basis, not ultimate basis. ultimate = anchor +
   reserve, and the anchor is data every model merely copied forward, so
   ultimate-basis errors credit all models for the same free information and
   compress the spread between them. `point_summary` subtracts the anchor
   first. Errors are premium-normalized (a distribution-free scale that stays
   finite when the actual reserve sits near zero after late favorable
   development) and compared with paired Wilcoxon tests, which are valid
   precisely because of the cohort-data parity in (1).

4. Distribution scoring: KS/PIT and CRPS. `summarize` takes the predictive
   percentile of each realized outcome (its PIT value) and tests uniformity
   with the Kolmogorov-Smirnov statistic against the 5% critical value
   1.36/sqrt(n) (`kernels.calibration`); `*` in the printed result flags
   rejection. CRPS (`kernels.scores`) measures sharpness-given-calibration
   but carries dollar units, so it is reported as CRPS/outcome to make
   companies of wildly different size comparable.

5. Provenance. Each results row is stamped with the gold-mart `publish_id`
   and the `loss_field`, so a CSV always traces back to an exact immutable
   publish (CLAUDE.md: experiment runs should pin a concrete @publish_id).

6. Failure handling. Fits that raise are recorded as an `error` row rather
   than aborting the study, and `summarize` prints the per-model failure
   count. Failures are NOT silently dropped from the record, but they do
   drop out of the scored panels (rows with a null percentile/estimate), so
   compare failure counts across models before reading any leaderboard.
   This applies to the chain_ladder benchmark too - see `point_context` for
   the cohorts Mack's model rejects, and note that a benchmark computed on a
   smaller panel than the models it scores makes `cl_skill` an unpaired
   comparison. Documented soft filters: `--copula-nonpositive drop` (known
   left-tail bias) and the copula's Hoerl fallback in `run_multiline_model`.

Usage:
    uv run python scripts/compare_gallery.py --per-line 5           # wiring check
    uv run python scripts/compare_gallery.py                        # full fast trio
    uv run python scripts/compare_gallery.py --models sur copula_glm nn_transformer meyers_ccl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ibis
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from meyers_validation import DEFAULT_WAREHOUSE, MEYERS_LINES, select_companies  # noqa: E402

from ibnr import gallery  # noqa: E402
from ibnr.data.schedule_p import active_mart_path, active_publish_id, load_schedule_p  # noqa: E402
from ibnr.kernels.calibration import ks_uniformity  # noqa: E402
from ibnr.kernels.mack import fit_mack  # noqa: E402
from ibnr.kernels.scores import crps  # noqa: E402

#: models cheap enough to run as the default set (seconds-to-minutes per
#: company); meyers_ccl is opt-in because it compiles and samples in cmdstan.
#: "mdn" is the transformer's no-attention variant - same pool, seeds and
#: augmentation, so running both in one study isolates the encoder body.
FAST_MODELS = ["sur", "copula_glm", "nn_transformer", "deeptriangle", "mdn", "resnet"]
#: multi-line transformer arms - the only entries that model cross-line
#: dependence, so the only ones producing a meaningful diversified total.
ML_MODELS = ["nn_ml_ar", "nn_ml_joint"]


def screened_company_lines(
    mart: Path, lines: list[str], per_line: int
) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str]]]]:
    """(scored, pools) - the cohort definition the whole study rests on.

    scored: company -> its screen-passing lines, for companies passing the
    Table A.1 screens on >=2 requested lines; every model scores exactly
    these (company, line) pairs. ``per_line`` caps the number of companies
    (wiring checks), 0 = all.

    The >=2-line requirement is what makes the multi-line entries (sur,
    copula_glm, nn_ml_*) meaningful at all - a monoline company has no
    cross-line dependence to estimate - and holding the cohort fixed across
    ALL models is Ethan's comparability rule: differences in the leaderboard
    must come from the models, not from who they were allowed to look at.

    pools: (company, line) training-pair pools for the transformer -
    "multiline" (the scored pairs; the default and the comparability rule),
    "screened" (>=1 passing line, i.e. + monoline companies; for comparison runs only).
    """
    sets: dict[str, set[str]] = {}
    for line in lines:
        selected = select_companies(mart, line, per_line=10_000)  # screen, don't cap yet
        sets[line] = set(selected["company_code"])
        print(f"  {line}: {len(selected)} companies pass the screens", flush=True)
    by_company: dict[str, list[str]] = {}
    for line in lines:  # keep the requested line order within each company
        for code in sets[line]:
            by_company.setdefault(code, []).append(line)
    # sorted() so the cohort (and any --per-line truncation of it) is
    # deterministic across runs - results CSVs must be reproducible
    scored = {c: ls for c, ls in sorted(by_company.items()) if len(ls) >= 2}
    if per_line:
        scored = dict(list(scored.items())[:per_line])
    multiline_pairs = [(c, ln) for c, ls in scored.items() for ln in ls]
    screened_pairs = [(c, ln) for c, ls in sorted(by_company.items()) for ln in ls]
    print(
        f"  scored: {len(scored)} companies with >=2 passing lines "
        f"({len(multiline_pairs)} pairs); screened pool adds "
        f"{len(screened_pairs) - len(multiline_pairs)} monoline pairs",
        flush=True,
    )
    return scored, {"multiline": multiline_pairs, "screened": screened_pairs}


def line_total_rows(pred, realized, label_map: dict[str, str]) -> dict[str, dict]:
    """Extract per-label rows (estimate/se/cv/outcome/percentile/crps) from a
    multiline predictive distribution.

    ``label_map`` maps an output key (a line name, or "ALL") to the target
    label the entry used, so every model's rows land in one common schema.
    ``percentile`` is the PIT value of the realized outcome under the model's
    own predictive draws - the input to the KS uniformity test. ``crps`` is
    in dollars here; `summarize` normalizes it by the outcome.
    """
    table = pred.summary(observed=realized)
    # scores[i] aligns with table row i: crps() returns one value per target,
    # in the same target order the summary table is built from
    scores = crps(pred.samples, realized)
    out = {}
    for key, label in label_map.items():
        i = table.index[table["label"] == label][0]
        row = table.loc[i]
        out[key] = {
            "estimate": row["estimate"],
            "se": row["se"],
            "cv": row["cv"],
            "outcome": row["outcome"],
            "percentile": row["percentile"],
            "crps": scores[i],
        }
    return out


def point_context(tri_all, scored, args) -> tuple[list[dict], dict, dict]:
    """(chain_ladder rows, anchors, premiums) for the scored pairs.

    anchor = loss-to-date at the as_of date (sum of each origin's latest
    observed cumulative in the training slice); premium = line premium.
    chain_ladder = volume-weighted CL point ultimate - the distribution-free
    skill benchmark every model is measured against. Keys are
    (company, line) plus (company, "ALL") sums.

    The estimate is ``kernels.mack.fit_mack(...).ultimate.sum()``: the same
    volume-weighted age-to-age factors this function used to compute inline,
    from the one implementation the gallery's Mack entry and the one-year CDR
    also use. A benchmark that drifts from the kernel it claims to be would
    quietly mis-score every model in the study, so it is not reimplemented here.

    The anchor is what makes reserve-basis scoring possible downstream:
    reserve = ultimate - anchor, so subtracting it strips out the loss
    dollars every model simply carried forward from the training slice.
    Premium is the normalizer for scale-free point errors. Both, and the
    outcome, are read straight off the triangle rather than off the ``MackFit``
    - a cohort Mack's model rejects (below) must still contribute its
    anchor/premium to EVERY OTHER model's reserve-basis scoring.

    !! THE POST-STUDY-ORIGIN TRAP !! This function reads BOTH the as_of
    training slice (`train`, for anchors/premium/origins) and the full
    triangle (`full`, for realized outcomes). The mart's accident years run
    past the 1988-1997 study window, so the outcome query MUST be filtered to
    the origins present in the training slice. This function once omitted that
    filter and reported outcomes 2.4x too large (CLAUDE.md gotcha). The guard
    is the `origin_period.isin(...)` predicate below - do not remove it, and
    replicate it in any new full-triangle aggregate.

    COHORTS MACK REJECTS. ``cohort_grid`` demands a clean run-off staircase,
    and ``fit_mack`` rejects a negative cumulative, a dev step of zero volume,
    and a step with too few positive origins to estimate a sigma from. It does
    NOT reject a mere zero: the volume-weighted factor needs only S_j > 0, so an
    accident year with zero paid at 12 months keeps its ultimate and only loses
    one observation from that step's sigma. That distinction is load-bearing
    here - 2 of the 152 scored cells (companies 29440 and 42439,
    other_liability) are exactly that shape, and under a blanket positivity rule
    the benchmark would score 150 cells against models scoring 152, which is
    precisely what makes `cl_skill` an unpaired comparison. A cohort that IS
    rejected is recorded as a failure row rather than aborting the study,
    exactly like a model fit that raises, so it lands in the failure census
    instead of silently reporting a number Mack's model does not admit. On the
    published panel nothing is rejected.
    """
    train = tri_all.as_of(args.as_of)  # upper triangle visible at the as_of date
    # each origin's latest observed cumulative in the training slice; summed
    # over a cohort's origins this is the anchor (loss-to-date), and the
    # origins themselves are the guard the outcome query needs
    diag = train.select_fields(args.loss_field).latest_diagonal().execute()
    prem = train.select_fields("earned_premium").latest_diagonal().execute()
    full = tri_all.select_fields(args.loss_field).execute()  # incl. realized lower half
    # deepest dev lag in the mart = the column that realizes each origin's ultimate
    n_d_months = int(full["dev_lag"].max())

    anchors: dict[tuple[str, str], float] = {}
    premiums: dict[tuple[str, str], float] = {}
    rows: list[dict] = []
    for code, lines_c in scored.items():
        cl_total, anchor_total, outcome_total, prem_total = 0.0, 0.0, 0.0, 0.0
        rejected: list[str] = []
        for line in lines_c:
            d_sub = diag[(diag["company_code"] == code) & (diag["line_of_business"] == line)]
            anchor = float(d_sub["value"].sum())
            f_sub = full[
                (full["company_code"] == code)
                & (full["line_of_business"] == line)
                & (full["dev_lag"] == n_d_months)
                # LOAD-BEARING GUARD (see docstring): the mart carries origins
                # beyond the study window, so outcomes are restricted to the
                # training slice's origins - exactly the origins the models
                # were asked to predict. Dropping this inflated outcomes 2.4x.
                & full["origin_period"].isin(d_sub["origin_period"])
            ]
            outcome = float(f_sub["value"].sum())
            p_sub = prem[(prem["company_code"] == code) & (prem["line_of_business"] == line)]
            line_prem = float(p_sub["value"].sum())
            anchors[(code, line)] = anchor
            premiums[(code, line)] = line_prem

            cohort = tri_all.filter(
                (ibis._.company_code == code) & (ibis._.line_of_business == line)
            )
            try:
                fit = fit_mack(cohort, loss_field=args.loss_field, as_of=args.as_of)
                est = float(fit.ultimate.sum())
            except ValueError as e:  # keep the study going; record the failure
                est, err = np.nan, str(e)
                rejected.append(line)
                print(f"  chain_ladder {line} {code}: FAILED {e}", flush=True)
            else:
                err = None
                cl_total += est
            row: dict = {
                "model": "chain_ladder",
                "line": line,
                "company_code": code,
                "estimate": est,
                "outcome": outcome,
            }
            if err is not None:
                row["error"] = err
            rows.append(row)
            anchor_total += anchor
            outcome_total += outcome
            prem_total += line_prem
        # company "ALL" row: a plain sum across the company's scored lines.
        # For a point estimate that is exactly right - diversification only
        # affects the spread, which chain_ladder does not produce. A partial
        # sum would not be, so a company with any rejected line gets no
        # estimate; its anchor/premium/outcome are unaffected.
        anchors[(code, "ALL")] = anchor_total
        premiums[(code, "ALL")] = prem_total
        all_row: dict = {
            "model": "chain_ladder",
            "line": "ALL",
            "company_code": code,
            "estimate": np.nan if rejected else cl_total,
            "outcome": outcome_total,
        }
        if rejected:
            all_row["error"] = f"lines rejected by fit_mack: {', '.join(rejected)}"
        rows.append(all_row)
    return rows, anchors, premiums


def run_multiline_model(model: str, tri_all, scored, args) -> list[dict]:
    """Per-company fits for the frequentist dependence models (sur, copula_glm).

    Each company is fit ONCE on its screened lines jointly, so the model can
    estimate cross-line correlation (Zhang 2010 SUR / Shi & Frees 2011 copula);
    scoring then reads both the per-line totals and the diversified company
    total out of the same predictive distribution. The as_of slicing happens
    inside ``gallery.fit`` - the triangle handed in here is the full one.

    A fit that raises is recorded as an ``error`` row and the study continues;
    silently skipping would make the failure invisible in the results CSV.
    """
    rows = []
    for i, (code, lines_c) in enumerate(scored.items(), 1):
        # exactly this company's SCORED lines - cohort-data parity
        tri = tri_all.filter((ibis._.company_code == code) & ibis._.line_of_business.isin(lines_c))
        t0 = time.perf_counter()
        try:
            extra = {"nonpositive": args.copula_nonpositive} if model == "copula_glm" else {}
            try:
                entry = gallery.fit(
                    model, tri, loss_field=args.loss_field, as_of=args.as_of, **extra
                )
            except ValueError as e:
                # dropped cells can deplete late dev steps; the Hoerl curve is
                # the documented reduced marginal for exactly this case
                if model != "copula_glm" or "hoerl" not in str(e):
                    raise
                entry = gallery.fit(
                    model,
                    tri,
                    loss_field=args.loss_field,
                    as_of=args.as_of,
                    dev_effect="hoerl",
                    **extra,
                )
                print(f"  {model} {code}: factor marginal unidentified, using hoerl", flush=True)
            pred = entry.predict(n_draws=args.draws, seed=args.seed)
            # realized_ultimates() reads the FULL triangle but restricts to the
            # origins the entry trained on - the same post-study-origin guard
            # point_context applies by hand
            realized = entry.realized_ultimates(tri)
            # per-line totals plus the diversified company total ("total")
            labels = {line: f"{line}/total" for line in lines_c} | {"ALL": "total"}
            per_label = line_total_rows(pred, realized, labels)
            secs = time.perf_counter() - t0
            for key, vals in per_label.items():
                rows.append(
                    {"model": model, "line": key, "company_code": code, **vals, "seconds": secs}
                )
            print(
                f"  [{i}/{len(scored)}] {model} {code} ({len(lines_c)} lines): "
                f"total pct={per_label['ALL']['percentile']:.1f} ({secs:.1f}s)",
                flush=True,
            )
        except Exception as e:  # keep the study going; record the failure
            rows.append({"model": model, "line": "ALL", "company_code": code, "error": str(e)})
            print(f"  [{i}/{len(scored)}] {model} {code}: FAILED {e}", flush=True)
    return rows


def pair_filter(tri, pairs: list[tuple[str, str]]):
    """Filter a triangle to exact (company, line) pairs via a concat key."""
    keys = [f"{c}|{ln}" for c, ln in pairs]
    return tri.filter((ibis._.company_code + "|" + ibis._.line_of_business).isin(keys))


def run_transformer(tri_market, scored, pools, args) -> list[dict]:
    """The single-line transformer: ONE pooled fit, then per-(company, line) predicts.

    Unlike the per-company statistical models, the NN is trained globally
    across cohorts and only specialized at predict time via ``segment=``. Its
    per-line draws are INDEPENDENT - this entry models no cross-line
    dependence, so it emits no diversified company total (see nn_ml_* for that).
    """
    # default pool: exactly the scored (company, line) pairs - the models see
    # identical cohort data; "screened" (+monoline pairs) and "market" (every
    # cohort) are alternate pools, run only to measure how much the pool
    # choice matters. The pinned-dev mean is a pooled statistic, so pool
    # hygiene is load-bearing.
    from ibnr.gallery.nn.transformer import TransformerConfig

    tri_pool = (
        tri_market if args.nn_pool == "market" else pair_filter(tri_market, pools[args.nn_pool])
    )
    t0 = time.perf_counter()
    entry = gallery.fit(
        "nn_transformer",
        tri_pool,
        loss_field=args.loss_field,
        feature_fields=tuple(args.nn_features),
        as_of=args.as_of,
        seed=args.seed,
        config=TransformerConfig(exposure_sigma=args.nn_exposure_sigma),
    )
    fit_secs = time.perf_counter() - t0
    n_cohorts = len(entry.contract_["cohorts"])
    print(f"  nn_transformer: pooled fit on {n_cohorts} cohorts ({fit_secs:.0f}s)", flush=True)
    if args.nn_exposure_sigma:
        import torch

        ps = [float(torch.nn.functional.softplus(m.raw_p)) for m in entry.models_]
        print(
            f"  exposure power p per member: {[round(p, 3) for p in ps]} "
            f"(mean {sum(ps) / len(ps):.3f}; 1.0 = flat constant-CV baseline, "
            "<1 tightens large books & widens small)",
            flush=True,
        )

    rows = []
    # predict the SCORED pairs only, whatever the training pool was - the
    # scored panel is fixed across models by construction
    for code, lines_c in scored.items():
        for line in lines_c:
            seg = {"company_code": code, "line_of_business": line}
            t1 = time.perf_counter()
            try:
                pred = entry.predict(segment=seg, n_draws=args.nn_draws, seed=args.seed)
                realized = entry.realized_ultimates(tri_market, segment=seg)
                per_label = line_total_rows(pred, realized, {line: "total"})
                rows.append(
                    {
                        "model": "nn_transformer",
                        "line": line,
                        "company_code": code,
                        **per_label[line],
                        "seconds": time.perf_counter() - t1,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "model": "nn_transformer",
                        "line": line,
                        "company_code": code,
                        "error": str(e),
                    }
                )
                print(f"  nn_transformer {line} {code}: FAILED {e}", flush=True)
    return rows


def run_deeptriangle(tri_market, scored, pools, args) -> list[dict]:
    """Kuo's DeepTriangle (GRU encoder/decoder, MDN head): ONE pooled fit,
    then per-(company, line) predicts - same shape as run_transformer.

    Per-line draws are INDEPENDENT (no diversified company total); the entry
    derives its auxiliary claims-outstanding target from the first feature
    field, so ``--nn-features reported_loss`` (the default) is load-bearing.
    """
    from ibnr.gallery.nn.deeptriangle import DeepTriangleConfig

    tri_pool = (
        tri_market if args.nn_pool == "market" else pair_filter(tri_market, pools[args.nn_pool])
    )
    t0 = time.perf_counter()
    entry = gallery.fit(
        "deeptriangle",
        tri_pool,
        loss_field=args.loss_field,
        feature_fields=tuple(args.nn_features),
        as_of=args.as_of,
        seed=args.seed,
        config=DeepTriangleConfig(),
    )
    fit_secs = time.perf_counter() - t0
    n_cohorts = len(entry.contract_["cohorts"])
    print(f"  deeptriangle: pooled fit on {n_cohorts} cohorts ({fit_secs:.0f}s)", flush=True)

    rows = []
    # predict the SCORED pairs only, whatever the training pool was - the
    # scored panel is fixed across models by construction
    for code, lines_c in scored.items():
        for line in lines_c:
            seg = {"company_code": code, "line_of_business": line}
            t1 = time.perf_counter()
            try:
                pred = entry.predict(segment=seg, n_draws=args.nn_draws, seed=args.seed)
                realized = entry.realized_ultimates(tri_market, segment=seg)
                per_label = line_total_rows(pred, realized, {line: "total"})
                rows.append(
                    {
                        "model": "deeptriangle",
                        "line": line,
                        "company_code": code,
                        **per_label[line],
                        "seconds": time.perf_counter() - t1,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "model": "deeptriangle",
                        "line": line,
                        "company_code": code,
                        "error": str(e),
                    }
                )
                print(f"  deeptriangle {line} {code}: FAILED {e}", flush=True)
    return rows


def run_mdn(tri_market, scored, pools, args) -> list[dict]:
    """The no-attention variant: run_transformer's exact shape with the MLP
    entry. Same pool, seed, features and as_of, so any gap between the mdn
    and nn_transformer rows is attributable to the encoder body."""
    from ibnr.gallery.nn.mdn import MDNConfig

    tri_pool = (
        tri_market if args.nn_pool == "market" else pair_filter(tri_market, pools[args.nn_pool])
    )
    t0 = time.perf_counter()
    entry = gallery.fit(
        "mdn",
        tri_pool,
        loss_field=args.loss_field,
        feature_fields=tuple(args.nn_features),
        as_of=args.as_of,
        seed=args.seed,
        config=MDNConfig(),
    )
    n_cohorts = len(entry.contract_["cohorts"])
    print(f"  mdn: pooled fit on {n_cohorts} cohorts ({time.perf_counter() - t0:.0f}s)", flush=True)

    rows = []
    # predict the SCORED pairs only, whatever the training pool was - the
    # scored panel is fixed across models by construction
    for code, lines_c in scored.items():
        for line in lines_c:
            seg = {"company_code": code, "line_of_business": line}
            t1 = time.perf_counter()
            try:
                pred = entry.predict(segment=seg, n_draws=args.nn_draws, seed=args.seed)
                realized = entry.realized_ultimates(tri_market, segment=seg)
                per_label = line_total_rows(pred, realized, {line: "total"})
                rows.append(
                    {
                        "model": "mdn",
                        "line": line,
                        "company_code": code,
                        **per_label[line],
                        "seconds": time.perf_counter() - t1,
                    }
                )
            except Exception as e:
                rows.append({"model": "mdn", "line": line, "company_code": code, "error": str(e)})
                print(f"  mdn {line} {code}: FAILED {e}", flush=True)
    return rows


def run_resnet(tri_market, scored, pools, args) -> list[dict]:
    """The residual conv net: run_transformer's exact shape - ONE pooled fit,
    then per-(company, line) predicts on the fixed scored panel. Same pool,
    seed, draws, features and loss field, so any gap between the resnet and
    nn_transformer rows is attributable to the encoder body (local 3x3
    convolutions vs global attention). Per-line draws are independent; no
    diversified total.

    ONE exception to "differs only by the encoder body", and it is announced
    rather than silent: ``--nn-exposure-sigma`` is a single-line-transformer
    knob (``TransformerConfig.exposure_sigma``); ``ResNetConfig`` has no
    equivalent, so with the flag on the transformer's sigma head carries a
    learned premium power and resnet's does not. The run is still meaningful -
    resnet is simply the flat-sigma baseline - but the two arms then differ in
    the head as well as the body, so the mismatch is printed on every run.
    """
    from ibnr.gallery.nn.resnet import ResNetConfig

    if args.nn_exposure_sigma:
        print(
            "  WARNING: --nn-exposure-sigma has no resnet equivalent "
            "(ResNetConfig has no exposure_sigma); resnet runs the flat-sigma "
            "head while nn_transformer runs the exposure-scaled one, so these "
            "rows differ in the HEAD as well as the encoder body",
            flush=True,
        )
    tri_pool = (
        tri_market if args.nn_pool == "market" else pair_filter(tri_market, pools[args.nn_pool])
    )
    t0 = time.perf_counter()
    entry = gallery.fit(
        "resnet",
        tri_pool,
        loss_field=args.loss_field,
        feature_fields=tuple(args.nn_features),
        as_of=args.as_of,
        seed=args.seed,
        config=ResNetConfig(),
    )
    n_cohorts = len(entry.contract_["cohorts"])
    print(
        f"  resnet: pooled fit on {n_cohorts} cohorts ({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

    rows = []
    # predict the SCORED pairs only, whatever the training pool was - the
    # scored panel is fixed across models by construction
    for code, lines_c in scored.items():
        for line in lines_c:
            seg = {"company_code": code, "line_of_business": line}
            t1 = time.perf_counter()
            try:
                pred = entry.predict(segment=seg, n_draws=args.nn_draws, seed=args.seed)
                realized = entry.realized_ultimates(tri_market, segment=seg)
                per_label = line_total_rows(pred, realized, {line: "total"})
                rows.append(
                    {
                        "model": "resnet",
                        "line": line,
                        "company_code": code,
                        **per_label[line],
                        "seconds": time.perf_counter() - t1,
                    }
                )
            except Exception as e:
                rows.append(
                    {"model": "resnet", "line": line, "company_code": code, "error": str(e)}
                )
                print(f"  resnet {line} {code}: FAILED {e}", flush=True)
    return rows


def run_transformer_ml(tri_market, scored, pools, args, dependence: str) -> list[dict]:
    """The multi-line transformer: one pooled company-level fit (attention
    across each company's screened lines), then per-company predicts in the
    SUR layout - per-line totals AND the diversified grand total.

    ``dependence`` selects the head: "ar" samples lines sequentially with an
    autoregressive link, "joint" uses a multivariate Gaussian-mixture head
    with absent lines marginalized out. Both arms share every other setting,
    so the comparison isolates the dependence structure (switchable by design).
    Emitting the SUR layout is what makes these rows directly comparable to
    the sur/copula_glm rows, grand total included.
    """
    from ibnr.gallery.nn.transformer_ml import TransformerMLConfig

    name = f"nn_ml_{dependence}"
    tri_pool = (
        tri_market if args.nn_pool == "market" else pair_filter(tri_market, pools[args.nn_pool])
    )
    t0 = time.perf_counter()
    entry = gallery.fit(
        "nn_transformer_ml",
        tri_pool,
        loss_field=args.loss_field,
        feature_fields=tuple(args.nn_features),
        as_of=args.as_of,
        seed=args.seed,
        config=TransformerMLConfig(dependence=dependence),
    )
    n_companies = len(entry.contract_["companies"])
    print(
        f"  {name}: pooled fit on {n_companies} companies ({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

    rows = []
    for i, (code, lines_c) in enumerate(scored.items(), 1):
        t1 = time.perf_counter()
        try:
            pred = entry.predict(
                segment={"company_code": code}, n_draws=args.nn_draws, seed=args.seed
            )
            realized = entry.realized_ultimates(tri_market, segment={"company_code": code})
            labels = {line: f"{line}/total" for line in lines_c} | {"ALL": "total"}
            per_label = line_total_rows(pred, realized, labels)
            secs = time.perf_counter() - t1
            for key, vals in per_label.items():
                rows.append(
                    {"model": name, "line": key, "company_code": code, **vals, "seconds": secs}
                )
            if i % 10 == 0 or i == len(scored):
                print(f"  [{i}/{len(scored)}] {name} {code}", flush=True)
        except Exception as e:
            rows.append({"model": name, "line": "ALL", "company_code": code, "error": str(e)})
            print(f"  {name} {code}: FAILED {e}", flush=True)
    return rows


def run_meyers(tri_by_line, scored, lines, args) -> list[dict]:
    """meyers_ccl: an independent cmdstan fit per (company, line).

    The Bayesian reference point. CCL is a single-line model, so it gets one
    fit per scored pair and produces no diversified total. Opt-in because each
    fit samples in cmdstan (minutes, not seconds).
    """
    rows = []
    for line in lines:
        # restrict to companies whose SCORED lines include this line, so CCL
        # is evaluated on the identical panel as every other model
        codes = [c for c, ls in scored.items() if line in ls]
        for i, code in enumerate(codes, 1):
            tri = tri_by_line[line].filter(ibis._.company_code == code)
            t0 = time.perf_counter()
            try:
                entry = gallery.fit(
                    "meyers_ccl",
                    tri,
                    loss_field=args.loss_field,
                    as_of=args.as_of,
                    chains=args.chains,
                    seed=args.seed,
                    iter_sampling=args.draws // args.chains,
                )
                pred = entry.predict(seed=args.seed)
                realized = entry.realized_ultimates(tri)
                per_label = line_total_rows(pred, realized, {line: "total"})
                rows.append(
                    {
                        "model": "meyers_ccl",
                        "line": line,
                        "company_code": code,
                        **per_label[line],
                        "seconds": time.perf_counter() - t0,
                    }
                )
                print(
                    f"  [{i}/{len(codes)}] meyers_ccl {line} {code}: "
                    f"pct={per_label[line]['percentile']:.1f}",
                    flush=True,
                )
            except Exception as e:
                rows.append(
                    {"model": "meyers_ccl", "line": line, "company_code": code, "error": str(e)}
                )
                print(f"  meyers_ccl {line} {code}: FAILED {e}", flush=True)
    return rows


def point_summary(df: pd.DataFrame) -> None:
    """Distribution-free comparison on the RESERVE basis: est/actual reserve
    = (estimate|outcome) - anchor. Reports error levels, the chain-ladder
    skill ratio, and paired Wilcoxon tests (all models score identical
    cells). Premium-normalized MAE is the robust headline - actual reserves
    can sit near zero (late favorable development), which blows up APEs.

    Why reserve and not ultimate basis: ultimate = anchor + reserve, and the
    anchor is loss already paid/reported at the as_of date - data every model
    copied forward unchanged. Scoring ultimates therefore credits every model
    with the same large, free, identical quantity, shrinking the apparent
    difference between a good and a bad reserve estimate. Subtracting the
    anchor scores only the part the model actually predicted.
    """
    from scipy import stats

    # Scored panel: per-line cells only. "ALL" rows are dropped because only
    # the multiline entries produce them - keeping them would score models on
    # different cells and break the paired design below. Rows with a null
    # estimate are the recorded failures and fall out here.
    ok = df[df["estimate"].notna() & (df["line"] != "ALL")].copy()
    ok["res_est"] = ok["estimate"] - ok["anchor"]  # predicted reserve
    ok["res_act"] = ok["outcome"] - ok["anchor"]  # realized reserve
    ok["abs_err"] = (ok["res_est"] - ok["res_act"]).abs()
    # signed, premium-normalized error: scale-free across companies of wildly
    # different size, and finite even when the realized reserve is near zero
    ok["err_premium_pct"] = (ok["res_est"] - ok["res_act"]) / ok["premium"] * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        # reserve APE - reported only as a MEDIAN below; its denominator can
        # be ~0, so the mean of this column would be dominated by a few cells
        ok["res_ape"] = ok["abs_err"] / ok["res_act"].abs()

    print("\n== Point prediction: reserve-basis errors ==")
    agg = ok.groupby("model").agg(
        n=("abs_err", "size"),
        mae_prem_pct=("err_premium_pct", lambda s: s.abs().mean()),
        bias_prem_pct=("err_premium_pct", "mean"),
        mdape_reserve=("res_ape", "median"),
    )
    # CL skill on the common cells: MAE_model / MAE_chain_ladder. <1 means the
    # model beats the distribution-free volume-weighted chain ladder; a
    # probabilistic model that cannot clear 1.0 buys its intervals with worse
    # central estimates.
    # cells: (line, company) x model matrix of |reserve error| - the paired
    # layout the Wilcoxon tests below need
    cells = ok.pivot_table(index=["line", "company_code"], columns="model", values="abs_err")
    if "chain_ladder" in cells.columns:
        agg["cl_skill"] = cells.mean() / cells["chain_ladder"].mean()
    print(agg.round(3).to_string())

    models = [m for m in cells.columns if m != "chain_ladder"]
    # Paired (not two-sample) Wilcoxon: models are compared cell by cell on the
    # same (line, company), so the enormous between-company variance in reserve
    # size cancels out of the test. This pairing is only legitimate because of
    # the cohort-data parity enforced upstream.
    print("\n== Paired Wilcoxon on |reserve error| (p-values; < means row beats col) ==")
    order = [*models, *(["chain_ladder"] if "chain_ladder" in cells.columns else [])]
    for a in order:
        parts = []
        for b in order:
            if a == b:
                parts.append("      -")
                continue
            # dropna(): a cell where either model failed is excluded from THIS
            # pair only, so each comparison uses the largest valid paired set
            pair = cells[[a, b]].dropna()
            diff = pair[a] - pair[b]
            # too few pairs (or exact ties everywhere) -> the test is degenerate
            if len(diff) < 6 or (diff == 0).all():
                parts.append("     na")
                continue
            p = stats.wilcoxon(diff).pvalue
            # direction is read off the median difference; the p-value itself
            # is two-sided, so "<0.020" means "a beats b, significantly"
            marker = "<" if diff.median() < 0 else ">"
            parts.append(f"{marker}{p:6.3f}")
        print(f"  {a:<16}" + " ".join(parts))
    print(f"  {'':16}" + " ".join(f"{m[:7]:>7}" for m in order))


def summarize(df: pd.DataFrame) -> None:
    """Distributional scoring: PIT/KS calibration, then CRPS sharpness.

    Calibration first, sharpness second, deliberately. A model whose PIT
    values are uniform is making honest probability statements; only among
    honest models does a lower CRPS mean anything. A very sharp but badly
    calibrated model (see the compartmental gaussian arm, CV ~2.5%) would win
    on CRPS while being wrong about its own uncertainty.

    ks_uniformity prints ``D x 100`` with the 5% critical value 1.36/sqrt(n)
    x 100 and marks rejection with ``*`` - Meyers' own convention, so results
    here are directly comparable to the monograph's tables.
    """
    # per-line cells only; "ALL" (diversified totals) get their own panel
    # below because only the multiline entries produce them
    ok = df[df["percentile"].notna() & (df["line"] != "ALL")]
    print("\n== Calibration: KS uniformity of line-total outcome percentiles ==")
    for model, grp in ok.groupby("model"):
        print(f"\n{model}:")
        # per line, then pooled: COMBINED has ~4x the n, so its critical value
        # is ~half as wide - a model can pass every line yet fail COMBINED
        for line, sub in grp.groupby("line"):
            print(f"  {line:<24} n={len(sub):>3}  {ks_uniformity(sub['percentile'] / 100)}")
        print(f"  {'COMBINED':<24} n={len(grp):>3}  {ks_uniformity(grp['percentile'] / 100)}")

    print("\n== Sharpness: mean CRPS / outcome (lower is better) ==")
    # CRPS is in dollars and therefore scales with company size; dividing by
    # the realized outcome makes it comparable across cells and lines. It is
    # only interpretable alongside the KS panel above (see docstring).
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = ok.assign(rel_crps=ok["crps"] / ok["outcome"])
    pivot = ok.pivot_table(index="line", columns="model", values="rel_crps", aggfunc="mean")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    # Company-level totals: only sur, copula_glm and nn_ml_* model cross-line
    # dependence, so only their totals test whether the claimed diversification
    # benefit is real. The single-line transformer's per-line draws are
    # independent, so it emits no "ALL" row at all.
    grand = df[(df["line"] == "ALL") & df["percentile"].notna()]
    if len(grand):
        print("\n== Diversified grand totals (multiline models) ==")
        for model, grp in grand.groupby("model"):
            print(f"  {model:<12} n={len(grp):>3}  {ks_uniformity(grp['percentile'] / 100)}")

    # Failure census. Report it next to the leaderboard: a model that fit only
    # the easy cells can look well calibrated on a quietly smaller panel.
    failed = df[df.get("error").notna()] if "error" in df else pd.DataFrame()
    if len(failed):
        print(f"\n{len(failed)} (model, company) fits failed:")
        print(failed.groupby(["model"])["error"].count().to_string())


def main() -> int:
    """Run the study: fix the cohort, run each requested model on it, score, write.

    Order matters - the cohort is resolved ONCE up front and every model
    runner is handed the same ``scored`` dict, which is the mechanism behind
    the cohort-data parity rule.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--warehouse",
        default=DEFAULT_WAREHOUSE,
        help="local warehouse path or github://owner/repo@publish_id "
        "(default: the latest GitHub release)",
    )
    ap.add_argument("--lines", nargs="+", default=MEYERS_LINES, choices=MEYERS_LINES)
    ap.add_argument("--per-line", type=int, default=0, help="cap companies (0 = all passing)")
    ap.add_argument(
        "--models",
        nargs="+",
        default=FAST_MODELS,
        choices=[*FAST_MODELS, *ML_MODELS, "meyers_ccl"],
    )
    # paid_loss is the headline field: copula_glm's lognormal marginals cannot
    # represent the negative late increments that reported_loss develops
    ap.add_argument("--loss-field", default="paid_loss")
    ap.add_argument(
        "--copula-nonpositive",
        default="drop",
        choices=["drop", "error"],
        help="Schedule P paid still dips at late lags (salvage); drop keeps the "
        "copula in the study at a documented left-tail bias",
    )
    # the monograph's training cutoff: fit the 1988-1997 upper triangle,
    # score the ultimates realized in the later statement years
    ap.add_argument("--as-of", default="1997-12-31")
    ap.add_argument("--draws", type=int, default=10_000, help="statistical model draws")
    ap.add_argument("--nn-draws", type=int, default=1000)
    # the NN may condition on reported_loss as a feature while predicting paid;
    # the statistical entries see only the loss field itself
    ap.add_argument("--nn-features", nargs="*", default=["reported_loss"])
    ap.add_argument(
        "--nn-pool",
        default="multiline",
        choices=["multiline", "screened", "market"],
        help="transformer training pool: companies passing screens on >=2 "
        "lines (default), >=1 line, or every company in the mart",
    )
    ap.add_argument(
        "--nn-exposure-sigma",
        action="store_true",
        help="single-line transformer only: learn a premium power p so the "
        "predictive dollar sd scales as premium**p (p=1 is the flat-sigma "
        "baseline). Off = baseline; on/off arms are directly comparable.",
    )
    ap.add_argument("--chains", type=int, default=4, help="meyers_ccl chains")
    # one seed drives selection order, sampler streams and NN init, so a rerun
    # of a published configuration reproduces the CSV exactly
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parents[1] / "analysis" / "results" / "compare_gallery.csv",
    )
    args = ap.parse_args()

    mart = active_mart_path(args.warehouse)
    # STEP 1 - fix the cohort once. Everything below is scored on exactly this
    # set of (company, line) pairs.
    print("selecting companies (screens per line):", flush=True)
    scored, pools = screened_company_lines(mart, args.lines, args.per_line)
    if not scored:
        print("no companies pass the screens on >=2 lines")
        return 1

    # tri_market is the FULL triangle (both halves). Every entry does its own
    # as_of slicing internally; nothing downstream may aggregate the realized
    # half without restricting to the training slice's origins.
    tri_market = load_schedule_p(args.warehouse, lines=args.lines)
    # STEP 2 - benchmark + the anchors/premiums that make reserve-basis and
    # premium-normalized scoring possible later
    print("\nchain_ladder point benchmark + anchors/premiums", flush=True)
    all_rows, anchors, premiums = point_context(tri_market, scored, args)

    for model in ("sur", "copula_glm"):
        if model in args.models:
            print(f"\n{model}: one fit per company, its screened lines jointly", flush=True)
            all_rows += run_multiline_model(model, tri_market, scored, args)

    if "nn_transformer" in args.models:
        print(f"\nnn_transformer: pooled fit, --nn-pool {args.nn_pool}", flush=True)
        all_rows += run_transformer(tri_market, scored, pools, args)

    if "deeptriangle" in args.models:
        print(f"\ndeeptriangle: pooled fit, --nn-pool {args.nn_pool}", flush=True)
        all_rows += run_deeptriangle(tri_market, scored, pools, args)

    if "mdn" in args.models:
        print(f"\nmdn: pooled fit (no-attention variant), --nn-pool {args.nn_pool}", flush=True)
        all_rows += run_mdn(tri_market, scored, pools, args)

    if "resnet" in args.models:
        print(f"\nresnet: pooled fit (conv-body variant), --nn-pool {args.nn_pool}", flush=True)
        all_rows += run_resnet(tri_market, scored, pools, args)

    for dep in ("ar", "joint"):
        if f"nn_ml_{dep}" in args.models:
            print(f"\nnn_ml_{dep}: pooled company fit ({dep} dependence head)", flush=True)
            all_rows += run_transformer_ml(tri_market, scored, pools, args, dep)

    if "meyers_ccl" in args.models:
        print("\nmeyers_ccl: one cmdstan fit per company x line", flush=True)
        tri_by_line = {line: load_schedule_p(args.warehouse, lines=[line]) for line in args.lines}
        all_rows += run_meyers(tri_by_line, scored, args.lines, args)

    df = pd.DataFrame(all_rows)
    # join anchors/premiums onto every row by (company, line) - including the
    # "ALL" rows, whose anchor/premium are the company-level sums. Failure rows
    # get them too, so a failed cell is still identifiable in the CSV.
    keys = list(zip(df["company_code"], df["line"], strict=True))
    df["anchor"] = [anchors.get(k, np.nan) for k in keys]
    df["premium"] = [premiums.get(k, np.nan) for k in keys]
    # Provenance stamping (CLAUDE.md): the gold mart is published as immutable
    # GitHub releases, so recording the resolved publish_id pins these results
    # to an exact dataset version even when --warehouse was "@latest".
    df["mart_publish_id"] = active_publish_id(args.warehouse)
    df["loss_field"] = args.loss_field
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # non-headline loss fields get their own file so a reported_loss run never
    # overwrites the published paid_loss results
    suffix = f"_{args.loss_field}" if args.loss_field != "paid_loss" else ""
    out = args.out.with_stem(args.out.stem + suffix)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    summarize(df)
    point_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
