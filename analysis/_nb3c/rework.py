"""Rework the merged 3c cells for the 0.5.8 rerun.

Three change sets, applied to the extracted cells:
1. wording sweep: "fingerprint" -> "hash", "membership" -> model-list wording,
   in prose AND printed labels (API attribute names stay as code);
2. version re-pin 0.5.6 -> 0.5.8;
3. the multi-line arms join the held-out board through the new adapter, so the
   ml fitting moves ahead of the board, the board machinery gains their rows,
   and the endpoint section reuses the fits.

Every targeted replacement is asserted to hit exactly once (or the stated
count), so a pattern that stopped matching fails the build instead of
silently leaving the old text in place.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cells import CELLS, C, MD  # noqa: E402

cells = [[kind, src] for kind, src in CELLS]
assert len(cells) == 57


def replace_in(idx, old, new, count=1):
    kind, src = cells[idx]
    n = src.count(old)
    assert n == count, f"cell {idx}: pattern found {n}x, expected {count}: {old[:70]!r}"
    cells[idx][1] = src.replace(old, new)


def replace_everywhere(old, new, expected_total):
    total = 0
    for cell in cells:
        n = cell[1].count(old)
        if n:
            cell[1] = cell[1].replace(old, new)
            total += n
    assert total == expected_total, f"{old!r}: replaced {total}, expected {expected_total}"


# --- change set 1+2: wording sweep and version pin --------------------------------
replace_everywhere("0.5.6", "0.5.8", 9)
replace_everywhere("PANEL_FINGERPRINT", "PANEL_HASH", 5)
replace_everywhere('"panel fingerprint"', '"panel hash"', 1)
replace_everywhere("panel fingerprint", "panel hash", 2)  # cell-5 prose + limitations
replace_everywhere('print("            fingerprint", panel.crps_fingerprint)', 'print("            hash       ", panel.crps_fingerprint)', 1)
replace_everywhere('print("CRPS fingerprint", panel.crps_fingerprint', 'print("CRPS hash       ", panel.crps_fingerprint', 1)
replace_in(
    21,
    "each score gets\nits own panel: its own membership, its own intersection of cells, and its own\nfingerprint, stamped on every board row, so that a number always carries the set of\nmodels that produced it.",
    "each score gets\nits own panel: its own list of models, its own intersection of cells, and its own\nhash, stamped on every board row, so that a number always carries the set of\nmodels that produced it.",
)
replace_in(
    6,
    "the lock is the cohort-list fingerprint plus the counts",
    "the lock is the cohort-list hash plus the counts",
)

# --- change set 3a: intro records what changed on the rerun -----------------------
replace_in(
    0,
    "runs it, times it, and reports the times rather than capping them.",
    "runs it, times it, and reports the times rather than capping them.\n\nThis is the second execution of the study. The first, on ibnr 0.5.6, compared the\nmulti-line arms at the endpoint level only and found a defect in the package while\ndoing it: entries restarted one random stream per fitted cohort under a shared seed,\nso summed draws across cohorts were correlated by construction. ibnr 0.5.7 fixed the\nstreams, and 0.5.8 adds the adapter that lets the multi-line transformer score\nheld-out cells at all. This run therefore differs from the first in three ways: the\ntwo multi-line arms appear on the held-out board, every company- and panel-total\nspread rests on genuinely independent streams, and the cross-model correlation table\nnow measures model structure rather than a shared-noise artifact.",
)
replace_in(
    0,
    "* the four single-line neural entries - `nn_transformer`, `mdn`, `resnet` and\n  `deeptriangle` - against Mack's distribution-free chain ladder on the next calendar\n  diagonal,",
    "* the four single-line neural entries - `nn_transformer`, `mdn`, `resnet` and\n  `deeptriangle` - and both multi-line transformer arms against Mack's\n  distribution-free chain ladder on the next calendar diagonal,",
)

# --- change set 3b: capabilities section --------------------------------------------
cells[13][1] = '''## Entry capabilities

A forecast carries two independent capabilities, and the separation is what lets one
board hold entries of very different shapes:

* `ScoresHeldout` -> `log_lik_at()` -> a normalized predictive density;
* `PredictsHeldout` -> `predict_at()` -> predictive draws, and hence CRPS.

The continuous ranked probability score (CRPS) generalizes absolute error to a
distributional forecast: it is the average absolute difference between a draw from the
forecast and the realized value, less half the average absolute difference between two
independent draws - the second term charging the forecast for its own spread, so neither
a wide hedge nor a false certainty scores well. For a point forecast it reduces to
absolute error; lower is better.

An entry may have either, both, or neither. `mack` draws from a bootstrap and states no
observation model, so it has draws and no density. `nn_transformer_ml` carries both
capabilities as of ibnr 0.5.8: its fitted cohort is a company while a held-out cohort is
a company-line pair, and the release adds the adapter that bridges the two - each
held-out call names the line, the network encodes the whole company as context, and the
scored line's predictive is read off the head (for the `"joint"` head, as the line's
marginal of the multivariate mixture). The first execution of this notebook could not
score those rows; this one does. `sur` still subclasses neither mixin, so it has no
board row and is compared at the endpoint level, where every entry produces a number.

The roster below is built from the live registry. `config_class` is the same idea one
level down: it is how an entry names the dataclass its `fit(config=...)` takes.'''

replace_in(
    14,
    '# The entries this notebook uses, in the order they appear: the five on the held-out\n# board, then the two that join at the endpoint level.\nON_THIS_BOARD = ["mack", "nn_transformer", "mdn", "resnet", "deeptriangle"]\nENDPOINT_ONLY = ["sur", "nn_transformer_ml"]',
    '# The entries this notebook uses: the five single-line rows of the held-out board, the\n# multi-line entry whose two arms join them as nn_ml_ar / nn_ml_joint, and sur, which\n# joins at the endpoint level only.\nON_THIS_BOARD = ["mack", "nn_transformer", "mdn", "resnet", "deeptriangle"]\nML_ENTRY = "nn_transformer_ml"\nENDPOINT_ONLY = ["sur"]',
)
replace_in(
    14,
    'display(roster[roster["entry"].isin(ON_THIS_BOARD + ENDPOINT_ONLY)].reset_index(drop=True))',
    'display(\n    roster[roster["entry"].isin([*ON_THIS_BOARD, ML_ENTRY, *ENDPOINT_ONLY])].reset_index(\n        drop=True\n    )\n)',
)

# --- change set 3c: models-and-fitting prose ----------------------------------------
replace_in(
    15,
    "`nn_transformer_ml` is fitted once per dependence\nmode, on all 91 companies together, each company's lines forming one joint training\nexample.",
    "`nn_transformer_ml` is fitted once per dependence\nmode, on all 91 companies together, each company's lines forming one joint training\nexample - and, new on this run, each arm's fit also supplies a held-out board row per\ncohort. A board row needs no rollout: the held-out diagonal sits one step past the\ncompany's observed context, so scoring it is one forward pass per ensemble member,\nwhile the rollout bill arrives only at the endpoint section.",
)

# --- change set 3d: the ml section moves ahead of the board -------------------------
ml_intro = cells[34][1]
assert ml_intro.startswith("## The multi-line transformer")
ml_intro = ml_intro.replace(
    "**The entry has no held-out board row**, for the reason the capabilities section gave:\nits cohort is a company while a held-out cohort is a company-line pair, and the adapter\nthat bridges them is deliberately deferred in the source. It joins at the 120-month\nendpoint through the ordinary gallery calls - `predict(segment=company)` returns the\nsame target layout as `sur` (per-line-and-origin endpoints, per-line totals, grand\ntotal), and `realized_ultimates` aligns the outcomes to it.",
    "**The entry joins the held-out board through the 0.5.8 adapter.** Its cohort is a\ncompany while a held-out cohort is a company-line pair; the adapter names the scored\nline, conditions on everything the company observed, and reads that line's predictive\noff the head - for the `\\\"joint\\\"` head as the line's marginal of the multivariate\nmixture, whose weights are unchanged and whose per-component scale is the matching row\nof the Cholesky factor. At the endpoint it still answers through the ordinary gallery\ncalls - `predict(segment=company)` returns the same target layout as `sur`\n(per-line-and-origin endpoints, per-line totals, grand total), and\n`realized_ultimates` aligns the outcomes to it.",
)
assert "0.5.8 adapter" in ml_intro

ml_board_fit = '''# Two pooled fits, one per dependence mode - the same budget as every other neural
# entry - then one held-out board row per cohort off each. No rollout runs here: a
# held-out diagonal is one forward pass per ensemble member, so the board's 10,000
# draws per cohort cost what a rollout could not. The endpoint section below reuses
# these same fits and pays the rollout bill there.
MLConfig = gallery.get("nn_transformer_ml").config_class

ml_fits: dict[str, object] = {}
ml_seconds: dict[str, dict] = {}

for row_name, dep in ML_MODES.items():
    t0 = time.perf_counter()
    entry = gallery.fit(
        "nn_transformer_ml",
        tri,
        loss_field=FIELD,
        as_of=AS_OF,
        seed=SEED_FIT,
        config=MLConfig(**ML_CONFIG, dependence=dep),
    )
    t_fit = time.perf_counter() - t0
    for key in COHORTS:
        forecasts.append(forecast_for(row_name, entry, key))
    ml_fits[row_name] = entry
    ml_seconds[row_name] = {"fit + heldout scoring": round(time.perf_counter() - t0, 1)}
    print(
        f"{row_name:12s} fit {t_fit:7.1f}s + held-out scoring "
        f"{time.perf_counter() - t0 - t_fit:6.1f}s over {len(COHORTS)} cohorts",
        flush=True,
    )'''

ml_endpoint = '''# The endpoint reads reuse the board section's fits; what is new here is the rollout -
# the first predict() call triggers the one shared simulation over every company
# (cached on (n_draws, seed)), so the remaining ninety are reads.
ml_ult_rows: list[dict] = []
ml_company_rows: list[dict] = []
ml_refusals: list[dict] = []
ml_line_draws: dict[tuple[str, tuple[str, str]], np.ndarray] = {}
ml_company_draws: dict[tuple[str, str], np.ndarray] = {}

for row_name, entry in ml_fits.items():
    t0 = time.perf_counter()
    _first = entry.predict(segment={"company_code": next(iter(PANEL))}, seed=SEED_DRAW)
    t_roll = time.perf_counter() - t0

    t0 = time.perf_counter()
    for company, lines in PANEL.items():
        ctri = tri.filter(tri.expr.company_code == company)
        try:
            pred = entry.predict(segment={"company_code": company}, seed=SEED_DRAW)
            outcome = np.asarray(
                entry.realized_ultimates(ctri, segment={"company_code": company}), dtype=float
            )
        except Exception as exc:
            ml_refusals.append(dict(model=row_name, company_code=company, error=repr(exc)[:180]))
            continue
        labels = pred.targets["label"].astype(str).tolist()
        cdf_at = pred.cdf(outcome)
        for ln in lines:
            i = labels.index(f"{ln}/total")
            ml_line_draws[(row_name, (company, ln))] = np.asarray(pred.samples[:, i], dtype=float)
            ml_ult_rows.append(
                dict(
                    model=row_name,
                    company_code=company,
                    line_of_business=ln,
                    predicted_ultimate=float(pred.mean()[i]),
                    realized_ultimate=float(outcome[i]),
                    premium=float(PREMIUM_TOTAL[(company, ln)]),
                    outcome_pct=100.0 * float(cdf_at[i]),
                )
            )
        i = labels.index("total")
        ml_company_draws[(row_name, company)] = np.asarray(pred.samples[:, i], dtype=float)
        ml_company_rows.append(
            dict(
                model=row_name,
                company_code=company,
                n_lines=len(lines),
                predicted_total=float(pred.mean()[i]),
                sd=float(pred.std()[i]),
                realized_total=float(outcome[i]),
                pct_error=100.0 * (float(pred.mean()[i]) - outcome[i]) / outcome[i],
                outcome_pct=100.0 * float(cdf_at[i]),
            )
        )
    t_score = time.perf_counter() - t0
    ml_seconds[row_name]["rollout"] = round(t_roll, 1)
    ml_seconds[row_name]["company_reads"] = round(t_score, 1)
    print(
        f"{row_name:12s} rollout (500 draws, all {len(PANEL)} companies) {t_roll:7.1f}s | "
        f"company reads {t_score:5.1f}s",
        flush=True,
    )

print(f"\\nrefusals: {len(ml_refusals)}")
for r in ml_refusals:
    print("  ", r)'''

# the endpoint ml cell (35) is replaced by the reuse version; the intro (34) and the
# new board-fit cell move to sit right after the single-line pooled fits (index 20)
cells[35][1] = ml_endpoint
moved = [[MD, ml_intro], [C, ml_board_fit]]
del cells[34]
for offset, cell in enumerate(moved):
    cells.insert(21 + offset, cell)
# after the move: indices 21..34 shift by +1 (one net insertion before them)
assert cells[21][1].startswith("## The multi-line transformer")
assert cells[23][1].startswith("## Results")
assert cells[36][1].startswith("# The endpoint reads reuse")

# --- change set 3e: board notes, refusal framing, calibration prose -----------------
replace_in(
    23,
    "No held-out density is reported on this roster. `mack` has none to give, and the four\nneural entries decline one on every cohort whose deepest scored cell falls on a pinned\ndevelopment step",
    "No held-out density is reported on this roster. `mack` has none to give, and all six\nneural rows - the four single-line entries and both multi-line arms - decline one on\nevery cohort whose deepest scored cell falls on a pinned development step",
)
replace_in(
    31,
    "3. the four neural rows are one pooled fit each, so all of their cells come from a\n   single estimation.",
    "3. the six neural rows are one pooled fit each (the two multi-line arms included), so\n   all of their cells come from a single estimation.",
)
replace_in(
    32,
    "the four NN rows one pooled fit each",
    "the six neural rows one pooled fit each",
)

# --- change set 3f: the pinned-cell split learns per-line pinning -------------------
replace_in(
    26,
    '''# the pinned development steps, read off one fitted entry's own normalizer
# (norm_["pinned"] is (n_channels, n_devs); channel 0 is the target)
_pinned_devs = sorted(
    int(d) for d in (np.nonzero(nn_fits["nn_transformer"].norm_["pinned"][0])[0] + 1) * 12
)''',
    '''# pinned development steps, read off each fit's own normalizer. The single-line
# entries pin per development step (norm_["pinned"] is (n_channels, n_devs), channel 0
# the target); the multi-line arms pin per (line, step), so medical malpractice - one
# company - pins a step the other lines do not.
_pinned_single = {
    int(d) for d in (np.nonzero(nn_fits["nn_transformer"].norm_["pinned"][0])[0] + 1) * 12
}
_pinned_ml = {
    (row, lob): {int(d) for d in (np.nonzero(fit.norm_["pinned"][li, 0])[0] + 1) * 12}
    for row, fit in ml_fits.items()
    for li, lob in enumerate(fit.contract_["lob_levels"])
}


def _is_pinned(model, lob, dev):
    if model in ml_fits:
        return dev in _pinned_ml[(model, lob)]
    return dev in _pinned_single''',
)
replace_in(
    26,
    'cells_df["cells"] = np.where(\n    cells_df["dev_lag"].isin(_pinned_devs), "pinned fallback", "trained head"\n)',
    'cells_df["cells"] = [\n    "pinned fallback" if _is_pinned(m, lob, d) else "trained head"\n    for m, lob, d in zip(\n        cells_df["model"], cells_df["line_of_business"], cells_df["dev_lag"], strict=True\n    )\n]',
)
replace_in(
    26,
    'print(f"pinned development steps on this window: {_pinned_devs} months")',
    'print(f"pinned steps, single-line entries: {sorted(_pinned_single)} months; "\n      f"multi-line arms pin per line (medical malpractice adds 108)")',
)

# --- change set 3g: correlation prose gains the 0.5.7 provenance --------------------
replace_in(
    45,
    "* `mack` fits every cohort independently with distinct RNG streams, so its implied\n  correlation is a sampling-noise zero - the negative control;",
    "* `mack` fits every cohort independently with distinct random streams, so its implied\n  correlation is a sampling-noise zero - the negative control. That sentence is TRUE on\n  this run and was false on the first: the first execution of this table measured 0.255\n  here, which turned out to be the package restarting one stream per cohort under the\n  shared seed. ibnr 0.5.7 derives a distinct stream per cohort, and this row is the\n  check that the fix behaves;",
)

# --- change set 3h: interpretation cells reopen -------------------------------------
# post-move layout: 45 correlation framing (MD), 46 correlation (C), 47 main
# interpretation (MD), 48 seed framing (MD), 49 seed check (C), 50 seed
# interpretation (MD)
main_interp, seed_frame, seed_check, seed_interp = 47, 48, 49, 50
assert cells[main_interp][1].startswith("### Reading the boards")
assert cells[seed_frame][1].startswith("### How much of a gap is the training seed")
assert "SEED_ALT = 12" in cells[seed_check][1]
assert "survives the seed" in cells[seed_interp][1]
cells[main_interp][1] = "{{FILL_MAIN_INTERPRETATION}}"
cells[seed_interp][1] = "{{FILL_SEED_INTERPRETATION}}"

# --- change set 3i: the seed check also scores the board's statistic ----------------
replace_in(
    seed_check,
    "    _per_cohort = np.array(\n        [\n            float(crps_fn(r[\"draws\"][:, None], np.asarray([r[\"realized\"]], dtype=float))[0])\n            / r[\"premium\"]\n            for r in _rows\n        ]\n    )",
    "    _per_cohort = np.array(\n        [\n            float(crps_fn(r[\"draws\"][:, None], np.asarray([r[\"realized\"]], dtype=float))[0])\n            / r[\"premium\"]\n            for r in _rows\n        ]\n    )\n    # the board's own statistic at the alternate seed: held-out cell CRPS as % of\n    # origin premium, equal cohort weight, off predict_at's 10,000 draws per cohort\n    _ho = []\n    for key in COHORTS:\n        _draws = entry.predict_at(cells[key], field=FIELD, seed=SEED_DRAW)\n        _vals = cells[key].frame\n        _kf = _vals[_vals[\"field\"] == FIELD]\n        _cell_crps = crps_fn(_draws, _kf[\"value\"].to_numpy(dtype=float))\n        _ho.append(float(np.mean(100.0 * _cell_crps / _kf[\"premium\"].to_numpy(dtype=float))))\n    _heldout_crps = float(np.mean(_ho))",
)
replace_in(
    seed_check,
    "    seed_rows.append(\n        dict(\n            model=row_name,\n            seed=SEED_ALT,\n            crps_pct_of_premium=float(np.mean(100.0 * _per_cohort)),",
    "    seed_rows.append(\n        dict(\n            model=row_name,\n            seed=SEED_ALT,\n            heldout_crps_pct_premium=_heldout_crps,\n            crps_pct_of_premium=float(np.mean(100.0 * _per_cohort)),",
)
replace_in(
    seed_check,
    "    _b = ult_board[ult_board[\"model\"] == row_name].iloc[0]\n    _k = ks_ult.set_index(\"model\").loc[row_name]\n    seed_rows.append(\n        dict(\n            model=row_name,\n            seed=SEED_FIT,\n            crps_pct_of_premium=float(_b[\"crps_pct_of_premium\"]),",
    "    _b = ult_board[ult_board[\"model\"] == row_name].iloc[0]\n    _k = ks_ult.set_index(\"model\").loc[row_name]\n    _nv = norm_view.loc[row_name, \"equal_cohort\"]\n    seed_rows.append(\n        dict(\n            model=row_name,\n            seed=SEED_FIT,\n            heldout_crps_pct_premium=float(_nv),\n            crps_pct_of_premium=float(_b[\"crps_pct_of_premium\"]),",
)
replace_in(
    seed_frame,
    "The two multi-line fits are repeated below at one alternate seed and scored\nthrough the same per-company loop on the same panel.",
    "The two multi-line fits are repeated below at one alternate seed and scored\nthrough the same per-company loop on the same panel - and, new on this run, through\nthe held-out board's own statistic as well, off each refit's 10,000 predict_at draws\nper cohort.",
)

# --- change set 3j: limitations -----------------------------------------------------
lim = next(i for i, c in enumerate(cells) if c[0] == MD and c[1].startswith("## Limitations"))
cells[lim][1] = '''## Limitations

1. **One cutoff, one calendar year.** 91 companies and 243 cohorts are a far wider
   panel than notebook 3b's, but every held-out cell is realized in calendar year 2008
   and every endpoint in the statements of 2008-2016 from one valuation date. A
   rejection on any calibration table can mean miscalibration or one shared calendar
   shock; a decision-grade study needs rolling cutoffs, which the mart's 1988-2007
   accident-year record supports.
2. **The panel is retrospectively selected, and thin where it is thin.** Complete
   all-positive 10x10 squares, at least two qualifying lines, positive premium
   throughout - a survivor population, not a random sample. Medical malpractice enters
   with one company and products liability with seven, so line-level statements about
   those lines rest on almost nothing.
3. **The selection rule lives in this notebook.** It mirrors
   `transformers_reserving/Code/data_prep.R` and runs on the pinned publish, but it is
   not yet a versioned artifact of the data package the way the Meyers company criteria
   are. The panel hash asserted above is the interim lock.
4. **The neural budget is reduced and mostly single-seed.** 120 epochs and five
   ensemble members against the entries' 400-epoch defaults; the two-seed check covers
   the multi-line arms only. Every held-out board row rests on 10,000 draws, the
   multi-line arms' included; endpoint draws remain 500 for every neural row against
   10,000 for `mack` and `sur`, and sample CRPS pays a small bias at 500.
5. **Information is asymmetric three ways.** `mack` sees one cohort per fit; the
   single-line neural entries see all 243 cohorts; the multi-line arms see all 243
   cohorts and, within a company, every other line's history; `deeptriangle`
   additionally consumes reported loss as an auxiliary channel. These are the models'
   intended designs, but the boards are not like-for-like information comparisons.
6. **`ultimate` means the 120-month grid endpoint.** No tail is fitted, and paid at 120
   months is not a settled cost. Paid is the only field scored; the incurred side of
   the new vintage is untouched here.
7. **The KS statistics are weak or descriptive throughout.** Cell-level statistics have
   no valid critical value; cohort- and company-level tests have real n but clustered
   units in one calendar year. All detect gross miscalibration and nothing finer.'''

# --- write the reworked builder -----------------------------------------------------
out = HERE / "cells_v2.py"
lines = ['"""Reworked 3c cells for the 0.5.8 rerun (generated by rework.py)."""', "", 'MD = "markdown"', 'C = "code"', "", "CELLS = ["]
for kind, src in cells:
    tag = "MD" if kind == MD else "C"
    lines.append(f"    ({tag}, {src!r}),")
lines.append("]")
out.write_text("\n".join(lines), encoding="utf-8")

joined = "\n\n".join(src for _, src in cells)
for banned in ("fingerprint", "membership", "screen"):
    hits = [
        line for line in joined.splitlines() if banned in line.lower() and "crps_fingerprint" not in line
    ]
    assert not hits, (banned, hits[:4])
assert joined.count("{{FILL") == 2
print(f"wrote {out} with {len(cells)} cells; wording and version checks pass")
