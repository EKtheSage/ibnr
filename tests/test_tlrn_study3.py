"""The third tlrn study's runner: the companion's final model, a refit, per-date companies.

``scripts/tlrn_colab.py --config companion_final`` and ``scripts/tlrn_study_colab.py --study
companion_final`` fit the companion's final model (the premium loss ratio head, attention
across accident years, company batches, a refit on everything known, alpha fitted on one
diagonal). Nothing here trains a real network. The tests drive the things whose mistakes would
cost a Colab afternoon or quietly change an answer:

* the model's configuration, written out value by value, and the two rows' difference;
* the study registry and the settings (a record of this study is never read as another's);
* the company set selected again at each valuation date, and the cells and benchmarks it feeds;
* parameter delivery through ``main`` (the cohort's own date, seed, row and budget);
* a real tiny fit, checked against ``gallery.fit`` (the same protocol, called a different way)
  and against the entry's own read-outs.
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import tlrn_colab  # noqa: E402
import tlrn_study2_eval as ev  # noqa: E402
import tlrn_study_colab as study  # noqa: E402

from .test_nn_features import START, study_arrays  # noqa: E402

# -- the model's configuration ---------------------------------------------------------------

#: the companion's final model, as the pre-registration lists it (every one a choice)
COMPANION = {
    "head": "premium_lr",
    "eps": 0.5,
    "lr_cap": 5.0,
    "attention": ("line", "lag", "ay"),
    "mask": "observed_cells",
    "cross_line": True,
    "n_layers": 1,
    "d_model": 32,
    "n_heads": 2,
    "dropout": 0.3,
    "factors_only": False,
    "member": "mcl_blend",
    "ensemble_size": 20,
    "keep": None,
    "batch_unit": "company",
    "batch_size": 8,
    "cutoff_sampling": "per_epoch",
    "min_cutoff": 2,
    "val_diagonals": 2,
    "scoring": "reached_cells",
    "w_pe": 0.5,
    "w_mse": 0.1,
    "mse_scale": 0.005,
    "lr": 0.003,
    "lr_phi": 0.03,
    "weight_decay": 0.0,
    "warmup": 20,
    "grad_clip": 1.0,
    "max_epochs": 3000,
    "min_epochs": 500,
    "check_every": 5,
    "patience": 600,
    "refit": True,
    "alpha_on": "first_diagonal",
}


def rows(members=20, smoke=False):
    return {
        r: (roles, cfg) for r, roles, cfg in tlrn_colab.configs("companion_final", members, smoke)
    }


def test_the_main_row_is_the_companions_final_model_value_for_value():
    roles, cfg = rows()["tlrn_cf_refit"]
    got = asdict(cfg)
    for name, value in COMPANION.items():
        assert got[name] == value, name
    assert roles == {"incurred_field": "incurred_loss", "case_field": "case_reserve"}


def test_the_main_row_is_the_library_preset_with_its_choices_written_out():
    from ibnr.gallery.nn.tlrn.config import TLRNConfig

    assert rows()["tlrn_cf_refit"][1] == TLRNConfig.companion_final_variant()


def test_the_comparison_row_differs_only_by_switching_the_network_off():
    main, factors = rows()["tlrn_cf_refit"], rows()["tlrn_cf_factors"]
    assert main[0] == factors[0]
    a, b = asdict(main[1]), asdict(factors[1])
    assert {k for k in a if a[k] != b[k]} == {"factors_only"}
    assert b["factors_only"] is True and b["refit"] is True and b["head"] == "premium_lr"


def test_the_row_order_is_the_registry_order():
    assert list(rows()) == ["tlrn_cf_refit", "tlrn_cf_factors"] == list(study.COMPANION_FINAL_ROWS)


def test_the_smoke_budget_reaches_both_rows_and_the_members_argument_too():
    for _, cfg in rows(members=3, smoke=True).values():
        assert (cfg.max_epochs, cfg.min_epochs, cfg.check_every, cfg.patience) == (10, 0, 5, 100)
        assert cfg.ensemble_size == 3 and cfg.refit and cfg.alpha_on == "first_diagonal"
    for _, cfg in rows(members=7).values():
        assert cfg.max_epochs == 3000 and cfg.ensemble_size == 7


def test_the_earlier_studies_rows_are_untouched():
    assert [r for r, _, _ in tlrn_colab.configs("accident_year", 20, False)] == [
        "tlrn_8_ay",
        "tlrn_13_ay",
    ]
    assert [r for r, _, _ in tlrn_colab.configs("factor_head", 20, False)] == [
        "tlrn_13_fh",
        "tlrn_13_fh_factors",
    ]
    for config in ("accident_year", "factor_head", "published"):
        for _, _, cfg in tlrn_colab.configs(config, 20, False):
            assert cfg.refit is False and cfg.alpha_on == "all_validation"


def test_the_colab_runner_accepts_the_new_config_name():
    import argparse

    seen = []
    original = argparse.ArgumentParser.parse_args

    def spy(self, argv=None, namespace=None):
        try:
            return original(self, argv, namespace)
        finally:
            seen.append(argv)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(argparse.ArgumentParser, "parse_args", spy)
        with pytest.raises(ModuleNotFoundError):
            # jax (or ibnr) is imported only after the arguments are accepted
            mp.setitem(sys.modules, "jax", None)
            tlrn_colab.main(["--out", "unused", "--config", "companion_final", "--smoke"])
    assert seen


# -- the study registry and the settings -----------------------------------------------------


def synthetic_cells(*, spike: bool) -> pd.DataFrame:
    """Five companies of two lines, accident years 2000 to 2009, in the runner's cell layout.

    With ``spike`` the second line of the fourth company has a loss ratio above 3 at calendar
    index 8, so that pair (and with it the company, left with one line) is selected at the
    valuation dates 6 and 7 and not at 8 and 10: the company sets differ by date.
    """
    a = study_arrays(n_companies=5, n_w=10, seed=3)
    n_c, n_l, n_w, n_d = a["paid"].shape
    rows = []
    for c in range(n_c):
        for li in range(n_l):
            for w in range(n_w):
                for d in range(n_d):
                    paid = float(a["paid"][c, li, w, d])
                    if spike and (c, li, w, d) == (3, 1, 1, 6):
                        paid *= 8.0
                    rows.append(
                        {
                            "company_code": f"{c + 1:04d}",
                            "line_of_business": f"lob_{'ab'[li]}",
                            "statement_year": START + w + d,
                            "accident_year": START + w,
                            "dev_age": d + 1,
                            "paid": paid,
                            "incurred": float(a["incurred"][c, li, w, d]),
                            "bulk": 0.0,
                            "case": float(a["case"][c, li, w, d]),
                            "premium": float(a["premium"][c, li, w]),
                        }
                    )
    return study._tidy(pd.DataFrame(rows))


def cohort_from(cells: pd.DataFrame, name="B") -> SimpleNamespace:
    """A cohort the way ``select_cohort`` builds one, without its published-count checks."""
    pairs, audit = study.select_pairs(cells, cutoff=10)
    chosen = cells[pd.MultiIndex.from_frame(cells[study.KEY]).isin(pairs)].reset_index(drop=True)
    return study.Cohort(
        name=name,
        pairs=pairs,
        audit=audit,
        companies=sorted({c for c, _ in pairs}),
        board=study.board_companies(cells, pairs),
        pairs_hash=study.pairs_hash(pairs),
        data_hash=study.cells_hash(chosen),
        actual=study.actual_reserves(cells, pairs, as_of_year=START + 9),
        source="synthetic",
        cells=chosen,
        as_of=dt.date(START + 9, 12, 31),
        all_cells=cells,
    )


def test_the_registry_names_the_third_study():
    third = study.get_study("companion_final")
    assert third.cohorts == ("B", "C") and third.per_valuation is True
    assert third.rows == ("tlrn_cf_refit", "tlrn_cf_factors")
    assert third.tables == ("valuations", "members")
    assert third.version == 3
    assert len({study.get_study(n).version for n in study.STUDIES}) == 3
    assert study.VALUATIONS == (6, 7, 8, 10)
    # the earlier studies are not per-valuation and keep their constants
    assert not study.get_study("accident_year").per_valuation
    assert not study.get_study("factor_head").per_valuation
    assert study.COHORTS == ("A", "B") and study.FACTOR_HEAD_COHORTS == ("A", "B", "C")


def test_the_settings_carry_the_study_and_the_company_set_of_every_date():
    cohort = cohort_from(synthetic_cells(spike=True))
    roles, cfg = rows()["tlrn_cf_refit"]
    got = study.fit_settings(
        cohort, "tlrn_cf_refit", roles, cfg, draw=2, smoke=False, study="companion_final"
    )
    assert got["study"] == 3 and got["study_name"] == "companion_final"
    assert got["seed"] == 11 + 100_003 * 2 and got["valuations"] == [6, 7, 8, 10]
    assert set(got["valuation_hashes"]) == {"6", "7", "8", "10"}
    assert got["valuation_hashes"]["10"] == cohort.pairs_hash
    assert got["valuation_hashes"]["6"] != got["valuation_hashes"]["10"]
    assert got["config"]["refit"] is True and got["config"]["alpha_on"] == "first_diagonal"
    # the second study's settings gained nothing
    second = study.fit_settings(
        cohort_from(synthetic_cells(spike=False), "B"),
        "tlrn_13_fh",
        *tlrn_colab.configs("factor_head", 20, False)[0][1:],
        draw=0,
        smoke=False,
        study="factor_head",
    )
    assert "valuations" not in second and "valuation_hashes" not in second


def test_the_third_studys_records_are_never_read_as_another_studys(tmp_path):
    import json

    cohort = cohort_from(synthetic_cells(spike=False))
    for name in ("companion_final", "factor_head"):
        roles, cfg = {r: (ro, c) for r, ro, c in study.row_configs(20, False, name)}[
            study.get_study(name).rows[0]
        ]
        settings = study.fit_settings(
            cohort, study.get_study(name).rows[0], roles, cfg, draw=0, smoke=False, study=name
        )
        (tmp_path / f"{name}-ibnr9.9.9.json").write_text(
            json.dumps({"settings": settings, "run": {}}), encoding="utf-8"
        )
    assert len(study.collect_records(tmp_path, smoke=False, study="companion_final")) == 1
    assert len(study.collect_records(tmp_path, smoke=False, study="factor_head")) == 1
    assert len(study.collect_records(tmp_path, smoke=False, study="accident_year")) == 0


# -- the company set at each valuation date --------------------------------------------------


def test_the_company_set_is_selected_again_at_each_date_and_nests():
    cohort = cohort_from(synthetic_cells(spike=True))
    sets = study.valuation_pairs(cohort)
    assert list(sets) == [6, 7, 8, 10]
    counts = {v: len(p) for v, p in sets.items()}
    assert counts == {6: 10, 7: 10, 8: 8, 10: 8}
    assert sets[10] == cohort.pairs
    for early, late in ((6, 7), (7, 8), (8, 10)):
        assert set(sets[late]) <= set(sets[early])
    # the dropped company is the spiked one, and it left because only one of its lines did
    assert {c for c, _ in sets[6]} - {c for c, _ in sets[10]} == {"0004"}
    assert ("0004", "lob_b") not in sets[8] and ("0004", "lob_a") not in sets[8]
    # it is kept on the cohort, not selected twice
    assert study.valuation_pairs(cohort) is sets


def test_a_set_that_does_not_end_at_the_cohorts_pairs_is_refused():
    cohort = cohort_from(synthetic_cells(spike=False))
    cohort.pairs = cohort.pairs[:-2]
    with pytest.raises(AssertionError, match="not the cohort's"):
        study.valuation_pairs(cohort)


def test_a_cohort_without_its_cells_cannot_be_selected_again():
    cohort = cohort_from(synthetic_cells(spike=False))
    cohort.all_cells = None
    with pytest.raises(ValueError, match="no cells"):
        study.valuation_pairs(cohort)


def test_select_cohort_keeps_all_the_cells_for_the_later_selection():
    # the real constructor: a cohort built from cells keeps every cell, not only the chosen pairs
    cells = synthetic_cells(spike=True)
    assert cohort_from(cells).all_cells is cells
    import inspect

    assert "all_cells=cells" in inspect.getsource(study.select_cohort)


# -- the truth and the benchmarks at each date ------------------------------------------------


def test_the_truth_table_scores_each_date_on_its_own_companies():
    cohort = cohort_from(synthetic_cells(spike=True))
    sets = study.valuation_pairs(cohort)
    per_date = ev.truth_table(cohort, pairs_by_valuation=sets)
    for v in study.VALUATIONS:
        got = per_date[per_date["valuation"] == v]
        assert sorted(got["company_code"]) == sorted({c for c, _ in sets[v]})
    # the dropped company is scored at the early dates only
    assert "0004" in set(per_date.loc[per_date["valuation"] == 7, "company_code"])
    assert "0004" not in set(per_date.loc[per_date["valuation"] == 10, "company_code"])
    # the cells are the evaluator's scoring window, whatever the company set
    n_w = n_d = 10
    for v in study.VALUATIONS:
        per_line = int(ev.scored_mask(v, n_w, n_d).sum())
        rows = per_date[per_date["valuation"] == v]
        lines = {c: sum(1 for cc, _ in sets[v] if cc == c) for c in rows["company_code"]}
        assert rows.set_index("company_code")["n_cells"].to_dict() == {
            c: per_line * n for c, n in lines.items()
        }
    # the last date is the cohort's realised reserve (the table checks it itself)
    last = per_date[per_date["valuation"] == 10].set_index("company_code")["actual"]
    want = cohort.actual["actual_reserve"].reindex(last.index)
    assert np.allclose(last, want)


def test_the_truth_table_is_what_it_was_when_no_sets_are_given():
    cohort = cohort_from(synthetic_cells(spike=False))
    default = ev.truth_table(cohort)
    explicit = ev.truth_table(cohort, pairs_by_valuation={v: cohort.pairs for v in ev.VALUATIONS})
    pd.testing.assert_frame_equal(default, explicit)


def test_the_benchmarks_are_fitted_and_scored_on_each_dates_own_companies():
    cohort = cohort_from(synthetic_cells(spike=True))
    sets = study.valuation_pairs(cohort)
    got = ev.benchmark_cells(cohort, pairs_by_valuation=sets, log=lambda *_: None)
    for v in study.VALUATIONS:
        part = got[got["valuation"] == v]
        assert sorted(part["company_code"]) == sorted({c for c, _ in sets[v]})
    # a company the early dates select is benchmarked there and not at the last date
    assert "0004" in set(got.loc[got["valuation"] == 6, "company_code"])
    assert "0004" not in set(got.loc[got["valuation"] == 10, "company_code"])
    # where the set is the cohort's, the per-date route is the old route
    old = ev.benchmark_cells(cohort, log=lambda *_: None)
    old10 = old[old["valuation"] == 10].set_index("company_code")
    new10 = got[got["valuation"] == 10].set_index("company_code")
    cols = ["mack", "mcl", "n_cells", "actual"]
    pd.testing.assert_frame_equal(new10.loc[old10.index, cols], old10[cols], rtol=1e-9)
    # the benchmarks are real numbers on this fixture
    assert np.isfinite(got[["mack", "mcl"]].to_numpy()).all()


# -- parameter delivery through main ---------------------------------------------------------


def fake_fits(seed: int) -> study.ValuationFits:
    """What ``fit_valuations`` returns, with the fields ``run_study`` and ``main`` read."""
    codes = ["1", "2", "3"]
    reserves = pd.DataFrame(
        {
            "company_code": codes,
            "anchor": 1.0,
            "premium": 10.0,
            "n_lines": 2,
            "reserve": [100.0, 200.0, 300.0],
            "member_00": [100.0, 200.0, 300.0],
        }
    )
    return study.ValuationFits(
        valuations=pd.DataFrame(
            {"valuation": 10, "company_code": codes, "n_cells": 9, "predicted": seed * 1.0}
        ),
        members=pd.DataFrame({"valuation": 10, "member": [0], "alpha": [0.5]}),
        reserves=reserves,
        reserves_selected=reserves,
        pairs={6: [("1", "a")] * 4, 10: [("1", "a")] * 3},
        seeds={6: seed, 10: seed},
    )


def test_main_delivers_the_study_the_cohorts_date_the_row_and_the_budget(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    sent, built = [], []

    def fake_fit_valuations(cohort, config, seed, **kw):
        sent.append({"cohort": cohort.name, "config": config, "seed": seed, **kw})
        return fake_fits(seed)

    def build(name, **kw):
        built.append((name, kw))
        return SimpleNamespace(
            name=name,
            pairs_hash="p" * 16,
            data_hash="d" * 16,
            board=["1", "2", "3"],
            actual=pd.DataFrame(
                {"actual_reserve": [100.0, 200.0, 300.0]},
                index=pd.Index(["1", "2", "3"], name="company_code"),
            ),
            source="stub",
            description="stub",
            as_of=study.AS_OF[name],
            by_valuation={v: [(name, str(v))] for v in study.VALUATIONS},
        )

    monkeypatch.setattr(study, "fit_valuations", fake_fit_valuations)
    monkeypatch.setattr(study, "cells_from_mart", lambda years=None: ("cells", years))
    monkeypatch.setattr(study, "build_cohort", build)
    study.main(
        [
            "--out",
            str(tmp_path),
            "--study",
            "companion_final",
            "--cohort",
            "C",
            "--smoke",
            "--draws",
            "2",
            "--members",
            "3",
            "--rows",
            "tlrn_cf_factors",
        ]
    )
    assert [b[0] for b in built] == ["C"]
    assert built[0][1]["cells"] == ("cells", (1988, 1997))
    assert [c["seed"] for c in sent] == [11, 11 + 100_003]
    for call in sent:
        assert call["cohort"] == "C" and call["backend"] == "jax"
        cfg = call["config"]
        assert cfg.factors_only is True and cfg.refit is True and cfg.ensemble_size == 3
        assert cfg.max_epochs == 10 and cfg.alpha_on == "first_diagonal"
    stem = "C_tlrn_cf_factors_d01_smoke-ibnr"
    assert any(
        p.name.startswith(stem) and p.name.endswith("_members.csv") for p in tmp_path.iterdir()
    )
    assert (tmp_path / "manifest_companion_final_smoke.json").exists()
    # cohort A is not part of the third study
    with pytest.raises(SystemExit):
        study.main(["--out", str(tmp_path / "x"), "--study", "companion_final", "--cohort", "A"])


# -- a real tiny fit -------------------------------------------------------------------------

BUDGET = {
    "d_model": 8,
    "dropout": 0.0,
    "max_epochs": 4,
    "min_epochs": 0,
    "check_every": 2,
    "patience": 5,
    "batch_size": 8,
    "n_strata": 1,
    "min_per_stratum": 3,
    "n_draws": 20,
    "warmup": 1,
}
SEED = 7


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    cohort = cohort_from(synthetic_cells(spike=True))
    config = tlrn_colab.companion_final_config(ensemble_size=2, **BUDGET)
    fits = study.fit_valuations(cohort, config, SEED, backend="torch", log=lambda *_: None)
    return cohort, config, fits


def test_every_date_is_trained_on_its_own_companies(tiny):
    cohort, _, fits = tiny
    sets = study.valuation_pairs(cohort)
    table = fits.valuations
    assert table["valuation"].drop_duplicates().tolist() == [6, 7, 8, 10]
    for v in study.VALUATIONS:
        got = sorted(table.loc[table["valuation"] == v, "company_code"])
        assert got == sorted({c for c, _ in sets[v]})
    assert fits.seeds == {v: SEED + 20_000 * v for v in study.VALUATIONS}
    assert set(fits.pairs) == set(study.VALUATIONS)


def test_the_cells_scored_are_the_evaluators_window_for_each_dates_companies(tiny):
    cohort, _, fits = tiny
    truth = ev.truth_table(cohort, pairs_by_valuation=study.valuation_pairs(cohort))
    got = fits.valuations.merge(truth, on=["valuation", "company_code"], suffixes=("", "_truth"))
    assert len(got) == len(fits.valuations) == len(truth)
    assert (got["n_cells"] == got["n_cells_truth"]).all()
    known = got["valuation"] < 10
    assert np.allclose(got.loc[known, "actual"], got.loc[known, "actual_truth"], rtol=1e-6)
    assert got.loc[~known, "actual"].isna().all()


def test_the_refit_and_no_refit_forecasts_are_both_there_and_differ(tiny):
    _, _, fits = tiny
    table = fits.valuations
    for column in ("predicted", "predicted_selected", "predicted_net", "predicted_net_selected"):
        assert np.isfinite(table[column]).all(), column
    assert not np.allclose(table["predicted"], table["predicted_selected"])
    assert not np.allclose(table["predicted_net"], table["predicted_net_selected"])
    members = fits.members
    assert {"valuation", "alpha", "best_epoch", "refit_epochs"} <= set(members.columns)
    assert len(members) == 4 * 2
    assert (members["refit_epochs"] == members["best_epoch"]).all()


def test_the_last_date_matches_the_entry_run_by_gallery_fit(tiny):
    from ibnr import gallery

    cohort, config, fits = tiny
    sets = study.valuation_pairs(cohort)
    triangle = study.triangle_from_cells(cohort.all_cells, sets[10])
    entry = gallery.fit(
        "tlrn",
        triangle,
        loss_field="paid_loss",
        as_of=cohort.as_of,
        seed=study.date_seed(SEED, 10),
        config=config,
        backend="torch",
        **study.study_roles(),
    )
    codes = [c["company_code"] for c in entry.cohorts()]
    last = fits.valuations[fits.valuations["valuation"] == 10].set_index("company_code")
    assert list(last.index) == codes
    # the table, the reserves frame and the entry's own read-outs are one number three ways
    assert np.allclose(last["predicted"], entry.company_reserves(), rtol=1e-4)
    assert np.allclose(last["predicted_selected"], entry.company_reserves(selected=True), rtol=1e-4)
    assert np.allclose(fits.reserves["reserve"], entry.company_reserves(), rtol=1e-4)
    assert np.allclose(
        fits.reserves_selected["reserve"], entry.company_reserves(selected=True), rtol=1e-4
    )
    members = entry.member_company_reserves()
    member_cols = [c for c in fits.reserves.columns if c.startswith("member_")]
    assert np.allclose(fits.reserves[member_cols].to_numpy().T, members, rtol=1e-4)
    assert np.allclose(fits.reserves["anchor"], entry.company_anchors())
    assert np.allclose(fits.reserves["premium"], entry.company_size_)
    assert fits.reserves["n_lines"].tolist() == entry.contract_["line_mask"].sum(axis=1).tolist()
    # the paired arm reads the very members the refit started from
    assert not np.allclose(entry.company_reserves(), entry.company_reserves(selected=True))


@pytest.mark.parametrize("v", [6, 7, 8])
def test_an_earlier_date_matches_the_entrys_own_retrained_protocol(tiny, v):
    from ibnr import gallery

    cohort, config, fits = tiny
    sets = study.valuation_pairs(cohort)
    triangle = study.triangle_from_cells(cohort.all_cells, sets[v])
    # the entry retrains the method at ``v`` from the companies of its triangle (the cell set
    # of S_v) with the seed that date_seed gives: the same protocol, called as gallery.fit calls it
    cfg = replace(config, calibration_cutoffs=(v,), calibration_horizons=(10 - v,))
    with study.capture_protocols() as runs:
        entry = gallery.fit(
            "tlrn",
            triangle,
            loss_field="paid_loss",
            as_of=cohort.as_of,
            # gallery.fit seeds its retrained date seed + 10000 v; the study's date_seed is the
            # same stream, so hand the entry the seed that lands on it
            seed=study.date_seed(SEED, v) - 10_000 * v,
            config=cfg,
            backend="torch",
            **study.study_roles(),
        )
    via_entry = study.date_tables(
        {"run": runs[1], "contract": entry.contract_, "v": v, "last_known": 10}
    )[0]
    mine = fits.valuations[fits.valuations["valuation"] == v].reset_index(drop=True)
    assert mine["company_code"].tolist() == via_entry["company_code"].tolist()
    for column in ("predicted", "predicted_selected", "predicted_net", "actual", "n_cells"):
        assert np.allclose(mine[column], via_entry[column], rtol=1e-4), column
    # and the entry's own calibration totals are those forecasts
    back = entry.backtest_
    back = back[back["cutoff"] == v].set_index("company_code")
    assert np.allclose(
        back["predicted"].reindex(mine["company_code"]), mine["predicted"], rtol=1e-4
    )


def test_a_fit_without_a_refit_is_refused_by_name():
    cohort = cohort_from(synthetic_cells(spike=False))
    triangle = study.triangle_from_cells(cohort.all_cells, cohort.pairs)
    config = tlrn_colab.companion_final_config(ensemble_size=1, refit=False, **BUDGET)
    with pytest.raises(ValueError, match="refit"):
        study.fit_valuation(triangle, 6, config, 1, as_of=cohort.as_of, backend="torch")


def test_a_valuation_past_the_last_known_diagonal_is_refused():
    pytest.importorskip("torch")
    cohort = cohort_from(synthetic_cells(spike=False))
    triangle = study.triangle_from_cells(cohort.all_cells, cohort.pairs)
    config = tlrn_colab.companion_final_config(ensemble_size=1, **BUDGET)
    with pytest.raises(ValueError, match="outside 1 to the last known diagonal 10"):
        study.fit_valuation(triangle, 11, config, 1, as_of=cohort.as_of, backend="torch")


def test_the_fit_is_pickled_and_its_reserves_frame_is_the_last_dates(tiny, tmp_path):
    import pickle

    _, _, fits = tiny
    path = tmp_path / "fits.pkl"
    study.save_entry(fits, path)
    again = pickle.loads(path.read_bytes())
    assert study.reserves_frame(again).equals(fits.reserves)
    assert set(fits.tables) == set(study.COMPANION_FINAL_TABLES)


# -- the jax path and the network-off row ----------------------------------------------------


@pytest.mark.parametrize("v", [6, 10])
def test_the_jax_path_gives_the_torch_path_numbers(v):
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    cohort = cohort_from(synthetic_cells(spike=False))
    triangle = study.triangle_from_cells(cohort.all_cells, cohort.pairs)
    config = tlrn_colab.companion_final_config(ensemble_size=2, **BUDGET)
    tables = {}
    for backend in ("torch", "jax"):
        fitted = study.fit_valuation(triangle, v, config, 5, as_of=cohort.as_of, backend=backend)
        tables[backend] = study.date_tables(fitted)
    for column in ("predicted", "predicted_selected", "predicted_net"):
        assert np.allclose(tables["jax"][0][column], tables["torch"][0][column], rtol=2e-3), column
    assert np.allclose(tables["jax"][1]["alpha"], tables["torch"][1]["alpha"], atol=2e-3)
    assert (tables["jax"][1]["refit_epochs"] == tables["torch"][1]["refit_epochs"]).all()


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_the_network_off_row_fits_and_refits_with_the_premium_head(backend):
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    cohort = cohort_from(synthetic_cells(spike=False))
    triangle = study.triangle_from_cells(cohort.all_cells, cohort.pairs)
    config = tlrn_colab.companion_final_config(ensemble_size=2, factors_only=True, **BUDGET)
    fitted = study.fit_valuation(triangle, 10, config, 5, as_of=cohort.as_of, backend=backend)
    table, members = study.date_tables(fitted)
    assert config.head == "premium_lr" and config.factors_only
    assert np.isfinite(table[["predicted", "predicted_selected", "predicted_net"]]).all().all()
    assert (table["predicted"] > 0).all()
    assert len(members) == 2 and (members["refit_epochs"] == members["best_epoch"]).all()
