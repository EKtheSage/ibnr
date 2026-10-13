"""The tlrn robustness study: its cohorts, its resumable runner and its evaluator.

``scripts/tlrn_study_colab.py`` trains the accident-year model many times and
``scripts/tlrn_study_eval.py`` scores what it saved. Neither is tested by fitting a network
here (a fit is minutes on a GPU); these tests drive the pieces whose mistakes would cost
a Colab afternoon or quietly change an answer:

* the selection rule, with its cutoff as the one thing that tells cohort A from cohort B;
* the two sources of cohort B (the mart and the cached CAS files) giving identical cells;
* the runner skipping what is saved, refusing what was saved under other settings, losing
  at most the fit that was running, and writing a record last;
* the statistics, each against a quantity worked out another way;
* the verdict rule, one case for each of its three words.

Where real data is needed (the published counts of the two cohorts) the test skips when
the Schedule P mart is not in the local cache.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import tlrn_colab  # noqa: E402
import tlrn_study_colab as study  # noqa: E402
import tlrn_study_eval as ev  # noqa: E402

# -- synthetic cells -----------------------------------------------------------------------

N = 4  # accident years and development ages of the synthetic squares
AY0 = 2000


def square(company, line, *, premium=1000.0, drop=None, edit=None, ay0=AY0):
    """A complete N x N square of cumulative paid, incurred and case, in the cell layout.

    ``drop`` removes one (accident year index, development age) cell; ``edit`` maps
    (accident year index, development age) to a dict of column overrides.
    """
    rows = []
    for w in range(N):
        for d in range(1, N + 1):
            if drop == (w, d):
                continue
            paid = 100.0 * (w + 1) * (1.0 - 0.5**d)
            row = {
                "company_code": company,
                "line_of_business": line,
                "statement_year": ay0 + w + d - 1,
                "accident_year": ay0 + w,
                "dev_age": d,
                "paid": paid,
                "incurred": paid * 1.1,
                "bulk": 0.0,
                "premium": premium,
            }
            row.update((edit or {}).get((w, d), {}))
            row["case"] = row["incurred"] - row["paid"] - row["bulk"]
            rows.append(row)
    return pd.DataFrame(rows)


def cells_of(*squares):
    return study._tidy(pd.concat(squares, ignore_index=True))


def six_companies():
    """Eleven pairs that fall out of the rule at different steps (see the test below)."""
    return cells_of(
        square("c1", "L1"),
        square("c1", "L2"),
        square("c2", "L1"),
        square("c2", "L2", edit={(0, 4): {"premium": 0.0}}),  # a bad premium on a future cell
        square("c3", "L1"),
        square("c3", "L2", edit={(3, 1): {"paid": 0.0}}),  # zero paid on the latest diagonal
        square("c4", "L1"),
        square("c4", "L2", edit={(0, 1): {"paid": 3500.0}}),  # loss ratio 3.5 on the first cell
        square("c5", "L1", drop=(1, 2)),  # an incomplete square
        square("c5", "L2"),
        square("c6", "L1"),
    )


# -- the selection --------------------------------------------------------------------------


def test_selection_counts_each_step_and_the_cutoff_makes_it_stricter():
    cells = six_companies()
    # the early diagonal 2 (notebook 04's cutoff, scaled to this square): c3's zero paid sits
    # at calendar index 4, beyond it, so c3 still has two usable lines
    early_pairs, early_audit = study.select_pairs(cells, cutoff=2)
    assert early_audit == [11, 9, 8, 4]
    assert early_pairs == [("c1", "L1"), ("c1", "L2"), ("c3", "L1"), ("c3", "L2")]
    # the last known diagonal (the companion study's cutoff): c3's zero is now visible
    late_pairs, late_audit = study.select_pairs(cells, cutoff=N)
    assert late_audit == [11, 9, 7, 2]
    assert late_pairs == [("c1", "L1"), ("c1", "L2")]
    assert set(late_pairs) <= set(early_pairs)


def test_selection_refuses_a_loss_ratio_of_three_and_a_nonpositive_premium():
    ok = study.select_pairs(cells_of(square("a", "L1"), square("a", "L2")), cutoff=N)[0]
    assert ok == [("a", "L1"), ("a", "L2")]
    high = square("a", "L2", edit={(0, 1): {"paid": 3000.0}})  # ratio exactly 3.0 is refused
    assert study.select_pairs(cells_of(square("a", "L1"), high), cutoff=N)[0] == []
    negative = square("a", "L2", edit={(1, 3): {"premium": -1.0}})
    assert study.select_pairs(cells_of(square("a", "L1"), negative), cutoff=N)[0] == []


def test_board_companies_leave_out_a_zero_that_a_transition_divides_by():
    only = cells_of(
        square("c1", "L1"),
        square("c1", "L2"),
        # calendar index 3 is the last that has an observed successor (N - 1), so a zero
        # there is divided by
        square("c2", "L1"),
        square("c2", "L2", edit={(2, 1): {"paid": 0.0}}),
        # a zero on the latest diagonal (index N) is never divided by
        square("c3", "L1"),
        square("c3", "L2", edit={(3, 1): {"paid": 0.0}}),
    )
    pairs = [(c, ln) for c in ("c1", "c2", "c3") for ln in ("L1", "L2")]
    assert study.board_companies(only, pairs) == ["c1", "c3"]


def test_actual_reserve_is_the_deepest_paid_minus_the_valuation_diagonal():
    cells = cells_of(square("c1", "L1"), square("c1", "L2"))
    got = study.actual_reserves(cells, [("c1", "L1"), ("c1", "L2")], as_of_year=AY0 + N - 1)
    paid = lambda w, d: 100.0 * (w + 1) * (1.0 - 0.5**d)  # noqa: E731
    anchor = sum(paid(w, N - w) for w in range(N))  # the cell on the diagonal of each year
    ultimate = sum(paid(w, N) for w in range(N))
    assert got.loc["c1", "anchor"] == pytest.approx(2 * anchor)
    assert got.loc["c1", "ultimate"] == pytest.approx(2 * ultimate)
    assert got.loc["c1", "actual_reserve"] == pytest.approx(2 * (ultimate - anchor))
    assert got.loc["c1", "premium"] == pytest.approx(2 * N * 1000.0)


def test_pairs_hash_is_notebook_04s_recipe_and_cells_hash_ignores_row_order():
    import hashlib

    pairs = [("2", "b"), ("1", "a")]
    assert study.pairs_hash(pairs) == hashlib.sha256(repr(sorted(pairs)).encode()).hexdigest()[:16]
    cells = six_companies()
    assert study.cells_hash(cells) == study.cells_hash(cells.sample(frac=1.0, random_state=1))
    changed = cells.copy()
    changed.loc[3, "paid"] += 1.0
    assert study.cells_hash(changed) != study.cells_hash(cells)


# -- the two sources of cohort B ------------------------------------------------------------

LINE_STEM = {line: stem for stem, line in study.CAS_FILES.items()}


def write_cas_files(folder: Path, cells: pd.DataFrame, *, extra_rows: pd.DataFrame | None = None):
    """The cells as four CAS Schedule P files, with the columns those files carry."""
    for line, stem in LINE_STEM.items():
        part = cells[cells["line_of_business"] == line]
        frame = pd.DataFrame(
            {
                "GRCODE": part["company_code"].astype(int),
                "GRNAME": "Some Ins Co",
                "AccidentYear": part["accident_year"],
                "DevelopmentYear": part["statement_year"],
                "DevelopmentLag": part["dev_age"],
                "IncurredLosses": part["incurred"],
                "CumPaidLoss": part["paid"],
                "BulkLoss": part["bulk"],
                "EarnedPremDIR": 1,
                "EarnedPremCeded": 0,
                "EarnedPremNet": part["premium"],
                "Single": 0,
                "PostedReserves2007": 1.5,
            }
        )
        if extra_rows is not None:
            frame = pd.concat([frame, extra_rows.assign(_line=line)[frame.columns]])
        frame.to_csv(folder / f"{stem}.csv", index=False)


def study_years_cells():
    """Synthetic cells in the accident years the loaders keep (1998 to 2000)."""
    lines = list(study.CAS_FILES.values())
    return cells_of(*(square(str(code), ln, ay0=1998) for code in (101, 202) for ln in lines))


def test_the_cas_files_give_the_cells_the_mart_gives(tmp_path):
    cells = study_years_cells()
    write_cas_files(tmp_path, cells)
    from_files = study.cells_from_cas(tmp_path)
    pd.testing.assert_frame_equal(from_files, cells)
    assert study.cells_hash(from_files) == study.cells_hash(cells)
    # the case reserve is incurred minus paid minus bulk, as the mart's column is defined
    np.testing.assert_allclose(
        from_files["case"], from_files["incurred"] - from_files["paid"] - from_files["bulk"]
    )


def test_cas_files_outside_the_study_years_are_dropped(tmp_path):
    cells = study_years_cells()
    old = pd.DataFrame(
        {
            "GRCODE": 101,
            "GRNAME": "x",
            "AccidentYear": 1997,
            "DevelopmentYear": 1997,
            "DevelopmentLag": 1,
            "IncurredLosses": 9.0,
            "CumPaidLoss": 9.0,
            "BulkLoss": 0.0,
            "EarnedPremDIR": 1,
            "EarnedPremCeded": 0,
            "EarnedPremNet": 9.0,
            "Single": 0,
            "PostedReserves2007": 1.0,
        },
        index=[0],
    )
    write_cas_files(tmp_path, cells, extra_rows=old)
    assert study.cells_from_cas(tmp_path)["accident_year"].min() == 1998


def test_a_missing_cas_file_is_named(tmp_path):
    write_cas_files(tmp_path, study_years_cells())
    (tmp_path / "wkcomp.csv").unlink()
    with pytest.raises(FileNotFoundError, match="wkcomp.csv"):
        study.cells_from_cas(tmp_path)


def test_the_study_triangle_has_the_layout_the_mart_loader_gives():
    cells = study_years_cells()
    pairs = [("101", "commercial_auto"), ("101", "workers_compensation")]
    tri = study.triangle_from_cells(cells, pairs)
    assert sorted(tri.fields) == ["case_reserve", "earned_premium", "incurred_loss", "paid_loss"]
    assert sorted(tri.segments) == ["company_code", "line_of_business"]
    frame = tri.expr.execute()
    frame["origin_period"] = pd.to_datetime(frame["origin_period"])
    frame["eval_date"] = pd.to_datetime(frame["eval_date"])
    assert set(zip(frame["company_code"], frame["line_of_business"], strict=True)) == set(pairs)
    paid = frame[frame["field"] == "paid_loss"].set_index(
        ["line_of_business", "origin_period", "dev_lag"]
    )
    # the 12-month cell of accident year 1999: dev_lag 12, evaluated 31 December 1999
    row = paid.loc[("commercial_auto", pd.Timestamp("1999-01-01"), 12)]
    assert row["eval_date"] == pd.Timestamp("1999-12-31")
    assert row["value"] == pytest.approx(100.0 * 2 * 0.5)
    # a pair outside the list is not in the triangle
    assert len(tri.as_of("2001-12-31").filter(tri.expr.company_code == "101").execute()) > 0


# -- the published cohorts (needs the Schedule P mart in the local cache) ------------------


def _mart_cached() -> bool:
    from ibnr.data import schedule_p

    repo = "EKtheSage/cas-schedule-p-data-model"
    try:
        return (schedule_p._cache_dir(repo, tlrn_colab.PUBLISH) / "manifest.json").exists()
    except Exception:
        return False


@pytest.mark.mart
@pytest.mark.skipif(not _mart_cached(), reason="the pinned Schedule P publish is not in the cache")
def test_the_two_cohorts_reproduce_their_published_counts_and_b_lies_inside_a():
    cells = study.cells_from_mart()
    a = study.select_cohort("A", cells, source="test")
    b = study.select_cohort("B", cells, source="test")
    assert (len(a.companies), len(a.pairs), len(a.board)) == (93, 243, 82)
    assert (len(b.companies), len(b.pairs), len(b.board)) == (84, 212, 84)
    assert a.pairs_hash == tlrn_colab.COHORT_HASH
    # the cohort carries exactly the cells of its pairs, and its data hash is theirs
    assert set(zip(a.cells["company_code"], a.cells["line_of_business"], strict=True)) == set(
        a.pairs
    )
    assert study.cells_hash(a.cells) == a.data_hash
    assert len(a.cells) == len(a.pairs) * 100
    assert set(b.pairs) <= set(a.pairs)
    # the dominant company is found, not named: 66% of cohort A's scored reserve
    top, share = ev.dominant_company(a.actual.loc[a.board, "actual_reserve"])
    assert top == "1767" and share == pytest.approx(0.663, abs=0.001)


@pytest.mark.mart
@pytest.mark.skipif(not _mart_cached(), reason="the pinned Schedule P publish is not in the cache")
def test_a_changed_cell_breaks_the_published_counts_loudly():
    cells = study.cells_from_mart()
    broken = cells.copy()
    broken.loc[broken["premium"] > 0, "premium"] = broken["premium"] * 1.0
    broken.loc[broken.index[:200], "premium"] = -1.0
    with pytest.raises(AssertionError, match="does not match its published selection"):
        study.select_cohort("B", broken, source="test")


# -- seeds, settings and the files of a fit --------------------------------------------------


def test_draw_seeds_start_at_the_earlier_fit_and_no_two_streams_collide():
    assert study.draw_seed(0) == tlrn_colab.SEED_FIT == 11
    # a fit seeds its members seed + 1000 m and its retrained valuation dates seed + 10000 v
    offsets = {10_000 * v + 1_000 * m for v in (0, 6, 7, 8) for m in range(20)}
    streams = [{study.draw_seed(k) + offset for offset in offsets} for k in range(10)]
    for a, b in itertools.combinations(range(10), 2):
        assert streams[a].isdisjoint(streams[b]), (a, b)


def cohort_stub(name="A", *, data_hash="d" * 16, board=("1", "2", "3")):
    actual = pd.DataFrame(
        {"actual_reserve": [100.0, 200.0, 300.0], "anchor": [1000.0, 2000.0, 3000.0]},
        index=pd.Index(list(board), name="company_code"),
    )
    return SimpleNamespace(
        name=name,
        pairs_hash="p" * 16,
        data_hash=data_hash,
        board=list(board),
        actual=actual,
        triangle=None,
        source="stub",
    )


def settings_for(cohort, row="tlrn_8_ay", draw=0, members=20, smoke=False):
    roles, cfg = {r: (ro, c) for r, ro, c in study.row_configs(members, smoke)}[row]
    return study.fit_settings(cohort, row, roles, cfg, draw=draw, smoke=smoke)


def test_two_rows_use_the_accident_year_variant_and_only_the_second_has_the_roles():
    configs = {r: (roles, cfg) for r, roles, cfg in study.row_configs(20, False)}
    assert list(configs) == ["tlrn_8_ay", "tlrn_13_ay"]
    assert configs["tlrn_8_ay"][0] == {}
    assert configs["tlrn_13_ay"][0] == {
        "incurred_field": "incurred_loss",
        "case_field": "case_reserve",
    }
    for _, cfg in configs.values():
        assert cfg.head == "premium_lr" and cfg.keep is None and cfg.ensemble_size == 20
        assert cfg.max_epochs == 3000 and cfg.calibration == "retrain_per_valuation"


def test_settings_decide_reuse_by_everything_but_the_ibnr_version():
    a, b = cohort_stub("A"), cohort_stub("A", data_hash="e" * 16)
    base = settings_for(a)
    assert settings_for(a) == base
    for changed in (
        settings_for(a, draw=1),
        settings_for(a, members=19),
        settings_for(a, row="tlrn_13_ay"),
        settings_for(a, smoke=True),
        settings_for(b),
    ):
        assert changed != base
    assert base["seed"] == 11 and settings_for(a, draw=3)["seed"] == 11 + 3 * 100_003


def test_file_names_carry_cohort_row_draw_smoke_and_version():
    pkl, record, reserves = study.fit_paths(
        Path("f"), "B", "tlrn_13_ay", 7, smoke=False, version="0.7.3"
    )
    assert pkl.name == "B_tlrn_13_ay_d07-ibnr0.7.3.pkl"
    assert record.name == "B_tlrn_13_ay_d07-ibnr0.7.3.json"
    assert reserves.name == "B_tlrn_13_ay_d07-ibnr0.7.3_reserves.csv"
    assert study.fit_paths(Path("f"), "A", "tlrn_8_ay", 0, smoke=True, version="x")[1].name == (
        "A_tlrn_8_ay_d00_smoke-ibnrx.json"
    )


# -- the runner ------------------------------------------------------------------------------


class FakeEntry:
    """Just enough of a fitted TLRN for the runner: a few attributes, picklable."""

    def __init__(self, seed: int, codes=("1", "2", "3")):
        rng = np.random.default_rng(seed)
        self._codes = list(codes)
        self.members = rng.uniform(80.0, 320.0, size=(4, len(codes)))
        self.company_size_ = np.array([10.0, 20.0, 30.0])
        self.contract_ = {"line_mask": np.array([[True, True], [True, False], [True, True]])}

    def cohorts(self):
        return [{"company_code": c} for c in self._codes]

    def member_company_reserves(self):
        return self.members

    def company_reserves(self):
        return self.members.mean(axis=0)

    def company_anchors(self):
        return np.array([1000.0, 2000.0, 3000.0])


class Recorder:
    """A fit_fn that records its calls, and can fail on a chosen call."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, cohort, row, roles, config, seed):
        if self.fail_on == len(self.calls):
            raise KeyboardInterrupt("the runtime disconnected")
        self.calls.append((cohort.name, row, seed, config.ensemble_size))
        return FakeEntry(seed), {"fit_seconds": 1.5, "jax_backend": "gpu"}


def run(out, fit_fn, *, draws=2, members=20, cohorts=None, rows=study.ROWS, **kw):
    cohorts = cohorts or {"A": cohort_stub("A"), "B": cohort_stub("B")}
    return study.run_study(
        out,
        cohorts,
        rows,
        draws,
        members=members,
        smoke=kw.pop("smoke", False),
        version="9.9.9",
        fit_fn=fit_fn,
        log=lambda *_: None,
        **kw,
    )


def test_a_run_fits_everything_in_draw_major_order_and_a_second_run_fits_nothing(tmp_path):
    first = Recorder()
    assert run(tmp_path, first) == {"fitted": 8, "skipped": 0}
    # draw 0 for every cohort and row before draw 1 starts
    assert [(c, r) for c, r, *_ in first.calls[:4]] == [
        ("A", "tlrn_8_ay"),
        ("A", "tlrn_13_ay"),
        ("B", "tlrn_8_ay"),
        ("B", "tlrn_13_ay"),
    ]
    assert [seed for *_, seed, _ in first.calls] == [11] * 4 + [11 + 100_003] * 4
    second = Recorder()
    assert run(tmp_path, second) == {"fitted": 0, "skipped": 8}
    assert second.calls == []
    manifest = json.loads((tmp_path / "manifest_study.json").read_text(encoding="utf-8"))
    assert len(manifest["fits"]) == 8 and manifest["smoke"] is False


def test_every_fit_is_saved_as_a_record_a_reserves_file_and_an_entry(tmp_path):
    run(tmp_path, Recorder(), draws=1, cohorts={"A": cohort_stub("A")}, rows=("tlrn_8_ay",))
    pkl, record, reserves = study.fit_paths(
        tmp_path, "A", "tlrn_8_ay", 0, smoke=False, version="9.9.9"
    )
    assert pkl.exists() and record.exists() and reserves.exists()
    body = json.loads(record.read_text(encoding="utf-8"))
    assert body["run"]["jax_backend"] == "gpu"
    assert body["settings"]["cohort"] == "A" and body["settings"]["seed"] == 11
    frame = pd.read_csv(reserves, dtype={"company_code": str})
    assert list(frame.columns)[:5] == ["company_code", "anchor", "premium", "n_lines", "reserve"]
    assert list(frame.columns)[5:] == [f"member_{m:02d}" for m in range(4)]
    entry = FakeEntry(11)
    np.testing.assert_allclose(frame["reserve"], entry.company_reserves())
    np.testing.assert_allclose(frame[[f"member_{m:02d}" for m in range(4)]].T, entry.members)
    assert list(frame["n_lines"]) == [2, 1, 2]


def test_an_interrupted_run_keeps_what_finished_and_resumes_with_the_rest(tmp_path):
    flaky = Recorder(fail_on=3)  # the fourth fit dies
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, flaky)
    assert len(flaky.calls) == 3
    assert len(list(tmp_path.glob("*.json"))) == 4  # three records and the manifest
    resumed = Recorder()
    assert run(tmp_path, resumed) == {"fitted": 5, "skipped": 3}
    assert len(resumed.calls) == 5
    # nothing is left half written under a final name
    assert not list(tmp_path.glob("*.part"))


def test_a_saved_fit_with_other_settings_stops_the_run_before_any_training(tmp_path):
    run(tmp_path, Recorder(), draws=1)
    untouched = {p.name: p.read_bytes() for p in tmp_path.glob("A_*")}
    for kwargs, named in (
        ({"members": 19}, "config.ensemble_size"),
        (
            {"cohorts": {"A": cohort_stub("A", data_hash="e" * 16), "B": cohort_stub("B")}},
            "data_hash",
        ),
    ):
        again = Recorder()
        with pytest.raises(SystemExit, match=named):
            run(tmp_path, again, draws=1, **kwargs)
        assert again.calls == []
    assert {p.name: p.read_bytes() for p in tmp_path.glob("A_*")} == untouched


def test_smoke_and_full_fits_do_not_mix(tmp_path):
    run(tmp_path, Recorder(), draws=1, smoke=True, cohorts={"A": cohort_stub("A")})
    full = Recorder()
    assert run(tmp_path, full, draws=1, cohorts={"A": cohort_stub("A")})["fitted"] == 2
    assert (tmp_path / "manifest_study_smoke.json").exists()
    assert len(json.loads((tmp_path / "manifest_study.json").read_text())["fits"]) == 2


def test_files_with_no_record_or_a_record_with_no_reserves_are_refused(tmp_path):
    run(tmp_path, Recorder(), draws=1, cohorts={"A": cohort_stub("A")}, rows=("tlrn_8_ay",))
    pkl, record, reserves = study.fit_paths(
        tmp_path, "A", "tlrn_8_ay", 0, smoke=False, version="9.9.9"
    )
    reserves.unlink()
    with pytest.raises(SystemExit, match="_reserves.csv is missing"):
        run(tmp_path, Recorder(), draws=1, cohorts={"A": cohort_stub("A")}, rows=("tlrn_8_ay",))
    record.unlink()
    pkl.write_bytes(b"x")
    with pytest.raises(SystemExit, match="no settings record"):
        run(tmp_path, Recorder(), draws=1, cohorts={"A": cohort_stub("A")}, rows=("tlrn_8_ay",))


def test_without_entries_only_the_record_and_the_reserves_are_written_and_resume_still_works(
    tmp_path,
):
    run(tmp_path, Recorder(), draws=1, save_entries=False)
    assert not list(tmp_path.glob("*.pkl"))
    assert len(list(tmp_path.glob("*_reserves.csv"))) == 4
    again = Recorder()
    assert run(tmp_path, again, draws=1, save_entries=False)["fitted"] == 0


def test_reserves_frame_of_a_real_fit_matches_the_entry():
    pytest.importorskip("torch")
    from ibnr import gallery
    from ibnr.gallery.nn.tlrn.config import TLRNConfig

    from .test_nn_features import study_arrays, study_triangle

    config = TLRNConfig.accident_year_variant(
        d_model=8,
        n_heads=2,
        dropout=0.0,
        max_epochs=3,
        min_epochs=0,
        check_every=2,
        patience=5,
        batch_size=3,
        ensemble_size=2,
        calibration_cutoffs=(5, 6, 7),
        calibration_horizons=(1, 2, 3),
        n_strata=1,
        min_per_stratum=3,
        n_draws=20,
        warmup=1,
    )
    entry = gallery.fit(
        "tlrn",
        study_triangle("duckdb", study_arrays(n_companies=4, n_w=8, seed=3)),
        loss_field="paid_loss",
        as_of="2007-12-31",
        config=config,
        seed=7,
    )
    frame = study.reserves_frame(entry)
    assert len(frame) == 4 and frame["company_code"].tolist() == [
        c["company_code"] for c in entry.cohorts()
    ]
    np.testing.assert_allclose(frame["reserve"], entry.company_reserves())
    np.testing.assert_allclose(frame["anchor"], entry.company_anchors())
    members = frame[[c for c in frame.columns if c.startswith("member_")]].to_numpy().T
    assert members.shape == (2, 4)
    # every member is kept, so the ensemble is the mean of the members
    np.testing.assert_allclose(frame["reserve"], members.mean(axis=0))


# -- the statistics --------------------------------------------------------------------------


def test_pool_ape_is_the_sum_of_absolute_errors_over_the_sum_of_actuals():
    from ibnr.kernels import point_metrics

    pred, actual = np.array([90.0, 260.0, 280.0]), np.array([100.0, 200.0, 300.0])
    assert ev.pool_ape(pred, actual) == pytest.approx((10 + 60 + 20) / 600)
    assert ev.pool_ape(pred, actual) == pytest.approx(point_metrics(pred, actual)["pool_ape"])


def test_draw_summary_and_win_rate():
    s = ev.draw_summary([0.05, 0.07, 0.06, 0.09])
    assert (s["min"], s["median"], s["max"], s["n"]) == (0.05, 0.065, 0.09, 4)
    assert s["sd"] == pytest.approx(np.std([0.05, 0.07, 0.06, 0.09], ddof=1))
    assert np.isnan(ev.draw_summary([0.05])["sd"])
    # strictly below: a tie is not a win
    assert ev.win_rate([0.05, 0.06, 0.06, 0.09], 0.06) == 0.25


def test_bootstrap_indices_are_reproducible_and_in_range():
    a = ev.bootstrap_indices(20, 500, 3)
    assert a.shape == (500, 20) and a.min() >= 0 and a.max() < 20
    np.testing.assert_array_equal(a, ev.bootstrap_indices(20, 500, 3))
    assert not np.array_equal(a, ev.bootstrap_indices(20, 500, 4))


def test_resampled_pool_ape_with_the_identity_resample_is_the_plain_one():
    pred, actual = np.array([90.0, 260.0, 280.0]), np.array([100.0, 200.0, 300.0])
    identity = np.arange(3)[None, :]
    assert ev.pool_ape_resampled(pred, actual, identity)[0] == pytest.approx(
        ev.pool_ape(pred, actual)
    )


def test_a_model_that_is_better_everywhere_has_an_interval_below_zero():
    rng = np.random.default_rng(0)
    actual = rng.uniform(100.0, 1000.0, size=40)
    bench = actual * 1.20
    model = actual * 1.05
    idx = ev.bootstrap_indices(40, 2000, 1)
    got = ev.paired_comparison(model, bench, actual, idx)
    assert got["ci_high"] < 0 and got["share_resamples_favour_model"] == 1.0
    assert got["share_companies_won"] == 1.0
    assert got["median_ape_diff"] == pytest.approx(0.05 - 0.20)
    assert got["pool_ape_diff"] == pytest.approx(0.05 - 0.20)
    assert got["wilcoxon_p"] < 0.001
    worse = ev.paired_comparison(bench, model, actual, idx)
    assert worse["ci_low"] > 0 and worse["share_resamples_favour_model"] == 0.0


def test_a_model_equal_to_its_benchmark_is_not_distinguishable_from_it():
    actual = np.array([100.0, 200.0, 300.0, 400.0])
    pred = np.array([110.0, 190.0, 330.0, 360.0])
    got = ev.paired_comparison(pred, pred.copy(), actual, ev.bootstrap_indices(4, 500, 1))
    assert got["pool_ape_diff"] == 0 and got["ci_low"] == 0 == got["ci_high"]
    assert got["wilcoxon_p"] == 1.0 and got["share_companies_won"] == 0.0
    assert got["share_resamples_favour_model"] == 0.0


def test_the_bootstrap_interval_brackets_the_full_sample_difference_when_it_is_mixed():
    rng = np.random.default_rng(5)
    actual = rng.uniform(100.0, 1000.0, size=60)
    model = actual * rng.uniform(0.8, 1.2, size=60)
    bench = actual * rng.uniform(0.8, 1.2, size=60)
    got = ev.paired_comparison(model, bench, actual, ev.bootstrap_indices(60, 4000, 2))
    assert got["ci_low"] < got["pool_ape_diff"] < got["ci_high"]
    assert 0.0 < got["share_resamples_favour_model"] < 1.0


def test_a_zero_actual_reserve_has_no_percentage_error_but_still_counts_in_the_pool():
    actual = np.array([0.0, 100.0, 200.0])
    model, bench = np.array([5.0, 90.0, 210.0]), np.array([20.0, 120.0, 190.0])
    got = ev.paired_comparison(model, bench, actual, ev.bootstrap_indices(3, 200, 1))
    assert got["pool_ape_model"] == pytest.approx(25 / 300)
    assert got["median_ape_diff"] == pytest.approx(np.median([(10 - 20) / 100, (10 - 10) / 200]))


def test_the_dominant_company_is_the_largest_actual_reserve_and_nobody_is_named():
    top, share = ev.dominant_company(pd.Series({"x": 10.0, "y": 70.0, "z": 20.0}))
    assert (top, share) == ("y", 0.7)


def test_the_squared_error_of_a_draw_is_the_squared_error_of_the_average_plus_the_variance():
    rng = np.random.default_rng(7)
    actual = rng.uniform(100.0, 1000.0, size=12)
    R = actual * rng.uniform(0.8, 1.3, size=(10, 12))
    got = ev.variance_decomposition(R, actual)
    assert got["mse_single_draw"] == pytest.approx(
        got["mse_draw_average"] + got["between_draw_variance"]
    )
    assert got["mse_single_draw"] == pytest.approx(((R - actual) ** 2).mean(axis=0).sum())
    assert 0.0 < got["draw_noise_share"] < 1.0 and 0.0 < got["draw_noise_share_relative"] < 1.0


def test_draws_that_agree_have_no_draw_noise_and_distinct_companies_give_an_icc_near_one():
    actual = np.array([100.0, 1000.0, 10_000.0])
    R = np.tile(actual * 1.1, (6, 1)) * (1 + 1e-6 * np.arange(6)[:, None])
    got = ev.variance_decomposition(R, actual)
    assert got["draw_noise_share"] < 1e-9
    assert got["icc_log_reserve"] > 0.999
    pure_noise = np.random.default_rng(1).normal(100.0, 5.0, size=(50, 30))
    assert abs(ev.variance_decomposition(pure_noise, np.full(30, 100.0))["icc_log_reserve"]) < 0.3


def test_a_company_without_a_positive_reserve_is_left_out_of_the_log_scale_icc_only():
    rng = np.random.default_rng(8)
    actual = rng.uniform(100.0, 1000.0, size=10)
    R = actual * rng.uniform(0.9, 1.1, size=(5, 10))
    R[2, 4] = -5.0  # one draw says company 4 has a negative reserve
    got = ev.variance_decomposition(R, actual)
    assert got["icc_companies"] == 9 and np.isfinite(got["icc_log_reserve"])
    clean = ev.variance_decomposition(np.delete(R, 4, axis=1), np.delete(actual, 4))
    assert got["icc_log_reserve"] == pytest.approx(clean["icc_log_reserve"])
    # the squared-error decomposition still uses every company
    assert got["n_companies"] == 10 and got["mse_single_draw"] > clean["mse_single_draw"]


def test_the_subset_curve_enumerates_every_subset_and_ends_at_the_full_average():
    rng = np.random.default_rng(2)
    actual = rng.uniform(100.0, 1000.0, size=15)
    R = actual * rng.uniform(0.7, 1.4, size=(5, 15))
    bench_value = ev.pool_ape(R[0], actual)
    curve = ev.subset_curve(R, actual, {"mack": bench_value})
    assert list(curve["k"]) == [1, 2, 3, 4, 5]
    assert list(curve["subsets"]) == [5, 10, 10, 5, 1] and curve["exhaustive"].all()
    # k = 1: the five single draws
    singles = [ev.pool_ape(R[i], actual) for i in range(5)]
    assert curve.loc[0, "min"] == pytest.approx(min(singles))
    assert curve.loc[0, "max"] == pytest.approx(max(singles))
    assert curve.loc[0, "below_mack"] == pytest.approx(np.mean(np.array(singles) < bench_value))
    # k = 5: the one average of everything
    assert curve.loc[4, "median"] == pytest.approx(ev.pool_ape(R.mean(axis=0), actual))
    # averaging cannot be worse than the worst single draw
    assert curve["max"].is_monotonic_decreasing


def test_a_curve_with_too_many_subsets_samples_and_says_so():
    R = np.random.default_rng(3).uniform(90.0, 110.0, size=(16, 6))
    curve = ev.subset_curve(R, np.full(6, 100.0), {})
    middle = curve[curve["k"] == 8].iloc[0]
    assert not middle["exhaustive"] and middle["subsets"] == ev.MAX_SUBSETS
    assert curve[curve["k"] == 16].iloc[0]["exhaustive"]


# -- the verdict -----------------------------------------------------------------------------

POOL = {"mack": 0.0577, "mcl": 0.0566}


def test_the_verdict_rule_has_one_case_for_each_word():
    clear = {"mack": (-0.012, -0.002), "mcl": (-0.010, -0.001)}
    word, why = ev.verdict(0.050, POOL, clear)
    assert word == "effective"
    assert "5.0000%" in why and "mack 5.7700%" in why and "mcl 5.6600%" in why
    # better on the median but one interval reaches zero
    assert ev.verdict(0.050, POOL, {"mack": (-0.012, -0.002), "mcl": (-0.010, 0.001)})[0] == (
        "not distinguishable"
    )
    # intervals wholly below zero but the median draw is not below mcl
    assert ev.verdict(0.0570, POOL, clear)[0] == "not distinguishable"
    # wholly above zero against the stronger benchmark (mcl)
    above = {"mack": (-0.001, 0.010), "mcl": (0.001, 0.012)}
    assert ev.verdict(0.062, POOL, above)[0] == "worse"
    # wholly above zero against the weaker benchmark only is not worse
    weaker_only = {"mack": (0.001, 0.010), "mcl": (-0.004, 0.012)}
    assert ev.verdict(0.062, POOL, weaker_only)[0] == "not distinguishable"
    # the stronger benchmark flips with the pooled errors
    assert ev.verdict(0.062, {"mack": 0.050, "mcl": 0.0566}, weaker_only)[0] == "worse"


# -- loading fits and the whole row ----------------------------------------------------------


def write_fit(folder: Path, cohort: str, row: str, draw: int, reserves, *, anchor, mutate=None):
    cohort_obj = cohort_stub(cohort)
    settings = settings_for(cohort_obj, row=row, draw=draw)
    if mutate:
        settings = {**settings, **mutate}
    pkl, record, csv = study.fit_paths(folder, cohort, row, draw, smoke=False, version="9.9.9")
    frame = pd.DataFrame(
        {"company_code": cohort_obj.board, "anchor": anchor, "premium": 1.0, "n_lines": 1}
    )
    frame["reserve"] = reserves
    frame.to_csv(csv, index=False)
    study.save_record(record, settings, {"fit_seconds": 1.0})


def test_load_fits_returns_draws_in_order_and_refuses_a_folder_of_two_studies(tmp_path):
    anchor = [1000.0, 2000.0, 3000.0]
    for draw in (2, 0, 1):
        write_fit(tmp_path, "A", "tlrn_8_ay", draw, [100.0 + draw, 200.0, 300.0], anchor=anchor)
    fits = ev.load_fits(tmp_path, "A", "tlrn_8_ay", smoke=False)
    assert [f.draw for f in fits] == [0, 1, 2]
    assert [f.seed for f in fits] == [11, 11 + 100_003, 11 + 2 * 100_003]
    assert ev.load_fits(tmp_path, "B", "tlrn_8_ay", smoke=False) == []
    write_fit(
        tmp_path,
        "A",
        "tlrn_8_ay",
        3,
        [1.0, 2.0, 3.0],
        anchor=anchor,
        mutate={"data_hash": "z" * 16},
    )
    with pytest.raises(SystemExit, match="data_hash"):
        ev.load_fits(tmp_path, "A", "tlrn_8_ay", smoke=False)


def test_a_fit_made_on_other_anchors_is_refused_by_the_reserve_matrix(tmp_path):
    cohort = cohort_stub("A")
    write_fit(tmp_path, "A", "tlrn_8_ay", 0, [100.0, 200.0, 300.0], anchor=[1000.0, 2000.0, 3000.0])
    write_fit(tmp_path, "A", "tlrn_8_ay", 1, [100.0, 200.0, 300.0], anchor=[1000.0, 2100.0, 3000.0])
    fits = ev.load_fits(tmp_path, "A", "tlrn_8_ay", smoke=False)
    with pytest.raises(SystemExit, match="anchors differ"):
        ev.reserve_matrix(fits, cohort.board, cohort.actual)
    good = ev.reserve_matrix(fits[:1], cohort.board, cohort.actual)
    np.testing.assert_allclose(good, [[100.0, 200.0, 300.0]])


def test_evaluate_row_ties_its_answers_together():
    rng = np.random.default_rng(11)
    n = 30
    actual = rng.uniform(100.0, 1000.0, size=n)
    actual[0] = 20_000.0  # one dominant company
    board = [f"c{i}" for i in range(n)]
    bench = pd.DataFrame(
        {
            "mack": actual * rng.uniform(0.85, 1.15, size=n),
            "mcl": actual * rng.uniform(0.85, 1.15, size=n),
            "sur": actual * rng.uniform(0.85, 1.15, size=n),
            "actual": actual,
        },
        index=pd.Index(board, name="company_code"),
    )
    R = actual * rng.uniform(0.75, 1.25, size=(10, n))
    res = ev.evaluate_row("tlrn_8_ay", R, bench, board)
    assert res["dominant_company"] == "c0" and res["dominant_share"] > 0.5
    # question 1: the summary is the per-draw Pool_APE, and the average is a separate number
    per_draw = [ev.pool_ape(R[d], actual) for d in range(10)]
    assert res["summary"]["median"] == pytest.approx(np.median(per_draw))
    assert res["average_pool_ape"] == pytest.approx(ev.pool_ape(R.mean(axis=0), actual))
    # question 2: win rate against the benchmark's own Pool_APE
    assert res["win_rate"]["mack"] == pytest.approx(
        np.mean(np.array(per_draw) < ev.pool_ape(bench["mack"], actual))
    )
    # question 3 and 4 have a row for each benchmark and scope, and the draw average's share
    # in question 4 is the same number as in question 3
    comparisons, claim = res["comparisons"], res["bootstrap_claim"]
    assert set(comparisons["benchmark"]) == {"mack", "mcl", "sur"}
    assert set(comparisons["scope"]) == {"including dominant", "excluding dominant"}
    merged = comparisons.merge(claim, on=["scope", "benchmark"])
    np.testing.assert_allclose(
        merged["share_resamples_favour_model"], merged["draw_average"], rtol=0, atol=0
    )
    assert (claim["draw_min"] <= claim["draw_median"]).all()
    assert (claim["draw_median"] <= claim["draw_max"]).all()
    # excluding the dominant company really leaves one company out
    assert comparisons.groupby("scope")["n"].first().to_dict() == {
        "excluding dominant": n - 1,
        "including dominant": n,
    }
    assert res["verdict"] in ev.VERDICTS
    # the excluded-scope Pool_APE differs from the full one
    assert res["average_pool_ape"] != res["average_pool_ape_excluding_dominant"]


def test_a_benchmark_missing_for_some_company_is_compared_on_the_rest():
    rng = np.random.default_rng(12)
    n = 20
    actual = rng.uniform(100.0, 1000.0, size=n)
    board = [f"c{i}" for i in range(n)]
    sur = actual * rng.uniform(0.9, 1.1, size=n)
    sur[[3, 4]] = np.nan
    bench = pd.DataFrame(
        {
            "mack": actual * 1.1,
            "mcl": actual * 0.9,
            "sur": sur,
            "actual": actual,
        },
        index=pd.Index(board, name="company_code"),
    )
    res = ev.evaluate_row("tlrn_8_ay", actual * rng.uniform(0.9, 1.1, size=(4, n)), bench, board)
    sur_rows = res["comparisons"][res["comparisons"]["benchmark"] == "sur"]
    assert set(sur_rows["n"]) <= {18, 17}
    assert "sur" not in res["benchmark_pool_ape"]  # no pooled number on an incomplete cover


def test_the_wire_up_from_a_saved_folder_to_a_verdict(tmp_path):
    """Fits on disk, the reserve matrix and the row evaluation, end to end on stubs."""
    cohort = cohort_stub("A")
    actual = cohort.actual["actual_reserve"].to_numpy()
    anchor = cohort.actual["anchor"].to_list()
    rng = np.random.default_rng(4)
    for draw in range(3):
        write_fit(
            tmp_path, "A", "tlrn_8_ay", draw, actual * rng.uniform(0.9, 1.1, 3), anchor=anchor
        )
    fits = ev.load_fits(tmp_path, "A", "tlrn_8_ay", smoke=False)
    R = ev.reserve_matrix(fits, cohort.board, cohort.actual)
    bench = pd.DataFrame(
        {"mack": actual * 1.2, "mcl": actual * 0.8, "sur": actual * 1.1, "actual": actual},
        index=pd.Index(cohort.board, name="company_code"),
    )
    res = ev.evaluate_row("tlrn_8_ay", R, bench, cohort.board)
    assert res["n_draws"] == 3 and res["verdict"] in ev.VERDICTS
    text = ev.render(cohort_with_description(cohort), {"tlrn_8_ay": res}, {}, "")
    assert "Q6 VERDICT" in text and "Q1" in text and "Q5c" in text


def cohort_with_description(cohort):
    return SimpleNamespace(name=cohort.name, description="a stub cohort")


# -- the benchmarks and their tie-outs -------------------------------------------------------


def bench_stub(board, scale):
    actual = np.linspace(100.0, 400.0, len(board))
    return pd.DataFrame(
        {
            "mack": actual * scale,
            "mcl": actual * (2 - scale),
            "sur": actual * scale,
            "actual": actual,
        },
        index=pd.Index(board, name="company_code"),
    )


def test_the_benchmark_fits_are_cached_by_cohort_and_data_hash_and_not_refitted(
    tmp_path, monkeypatch
):
    cohort = cohort_stub("A")
    calls = []

    def fake(c, *, log=print):
        calls.append(c.name)
        return bench_stub(c.board, 1.1)

    monkeypatch.setattr(ev, "benchmark_reserves", fake)
    first = ev.cached_benchmarks(cohort, tmp_path, log=lambda *_: None)
    again = ev.cached_benchmarks(cohort, tmp_path, log=lambda *_: None)
    assert calls == ["A"]
    pd.testing.assert_frame_equal(first, again)
    # other cells are another cache entry, and no cache directory means always compute
    ev.cached_benchmarks(cohort_stub("A", data_hash="e" * 16), tmp_path, log=lambda *_: None)
    ev.cached_benchmarks(cohort, None, log=lambda *_: None)
    assert calls == ["A", "A", "A"]


def test_a_cache_for_other_companies_is_not_trusted(tmp_path, monkeypatch):
    cohort = cohort_stub("A")
    stale = bench_stub(["1", "2"], 1.1)  # the cohort has three scored companies
    stale.to_csv(tmp_path / f"benchmarks_A_{cohort.data_hash}.csv")
    monkeypatch.setattr(ev, "benchmark_reserves", lambda c, *, log=print: bench_stub(c.board, 1.1))
    got = ev.cached_benchmarks(cohort, tmp_path, log=lambda *_: None)
    assert list(got.index) == cohort.board


def test_the_tie_out_to_notebook_04_refuses_a_pool_ape_that_moved():
    board = [f"c{i}" for i in range(5)]
    bench = bench_stub(board, 1.0)
    bench["mack"] = bench["actual"] * (1 + ev.REFERENCE_A["mack"])  # Pool_APE exactly the reference
    bench["mcl"] = bench["actual"] * (1 + ev.REFERENCE_A["mcl"])
    cohort = SimpleNamespace(name="A")
    got = ev.tie_out(cohort, bench, None)
    assert any("mack" in key for key in got) and any("mcl" in key for key in got)
    bench["mack"] = bench["actual"] * 1.07
    with pytest.raises(SystemExit, match="does not reproduce notebook 04"):
        ev.tie_out(cohort, bench, None)


def test_the_tie_out_to_the_companions_table_reports_the_largest_relative_difference():
    board = [f"c{i}" for i in range(4)]
    bench = bench_stub(board, 1.0)
    companion = pd.DataFrame(
        {
            "IBNR_CL": bench["mack"].to_numpy() * (1 + np.array([0, 1e-9, 0, 0])),
            "IBNR_CL_MCL": bench["mcl"].to_numpy(),
            "IBNR_CL_SUR": bench["sur"].to_numpy() * 2,  # a different estimator: never compared
            "IBNR_obs": bench["actual"].to_numpy(),
        },
        index=pd.Index(board, name="GRCODE"),
    )
    got = ev.tie_out(SimpleNamespace(name="B"), bench, companion)
    assert (
        got["mack reserves against the companion's IBNR_CL"] == "max relative difference 1.00e-09"
    )
    assert (
        got["mcl reserves against the companion's IBNR_CL_MCL"]
        == "max relative difference 0.00e+00"
    )
    assert not any("SUR" in key or "sur" in key for key in got)
    assert ev.max_relative_difference([1.0, 2.0], [1.0, 4.0]) == 0.5
    assert ev.max_relative_difference([3.0], [0.0]) == 3.0  # a zero reference: absolute


# -- the pre-registered constants are the ones the decision note states ------------------------


def test_the_bootstrap_settings_are_the_pre_registered_ones():
    assert (ev.N_BOOT, ev.BOOT_SEED, ev.LEVEL) == (10_000, 2008, 0.95)
    assert ev.VERDICT_BENCHMARKS == ("mack", "mcl")
    assert ev.VERDICTS == ("effective", "not distinguishable", "worse")


def test_the_interval_is_the_2_5_to_97_5_percentile_of_the_resampled_differences():
    rng = np.random.default_rng(21)
    actual = rng.uniform(100.0, 1000.0, size=40)
    model = actual * rng.uniform(0.8, 1.2, size=40)
    bench = actual * rng.uniform(0.8, 1.2, size=40)
    idx = ev.bootstrap_indices(40, 3000, 9)
    got = ev.paired_comparison(model, bench, actual, idx)
    diff = ev.pool_ape_resampled(model, actual, idx) - ev.pool_ape_resampled(bench, actual, idx)
    low, high = np.quantile(diff, [0.025, 0.975])
    assert (got["ci_low"], got["ci_high"]) == pytest.approx((low, high), rel=1e-12)


def test_a_row_is_evaluated_with_the_default_resamples_of_the_right_size():
    rng = np.random.default_rng(22)
    n = 25
    actual = rng.uniform(100.0, 1000.0, size=n)
    board = [f"c{i}" for i in range(n)]
    bench = pd.DataFrame(
        {
            "mack": actual * rng.uniform(0.9, 1.1, size=n),
            "mcl": actual * rng.uniform(0.9, 1.1, size=n),
            "sur": actual * rng.uniform(0.9, 1.1, size=n),
            "actual": actual,
        },
        index=pd.Index(board, name="company_code"),
    )
    R = actual * rng.uniform(0.8, 1.2, size=(4, n))
    res = ev.evaluate_row("tlrn_8_ay", R, bench, board)
    row = res["comparisons"].query("scope == 'including dominant' and benchmark == 'mack'").iloc[0]
    # the 10,000 resamples of the 25 companies, seed 2008, for the draw average
    direct = ev.paired_comparison(
        R.mean(axis=0), bench["mack"].to_numpy(), actual, ev.bootstrap_indices(n)
    )
    assert (row["ci_low"], row["ci_high"]) == (direct["ci_low"], direct["ci_high"])
    assert ev.bootstrap_indices(n).shape == (10_000, n)
