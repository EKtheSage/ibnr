"""The second study's runner through the JAX backend, which is what Colab uses.

The other runner tests train with torch or stub the fit out, so nothing ran
``gallery.fit(..., backend="jax")`` through the runner's own path. This one does, on a tiny
synthetic triangle on the CPU: ``study.main`` builds the (stubbed) cohort, fits one member for
ten epochs with the JAX backend, collects the valuation tables, and saves the files the
evaluator reads. It skips cleanly when jax or torch is not installed.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import tlrn_colab  # noqa: E402
import tlrn_study_colab as study  # noqa: E402

#: a budget small enough for a CPU test (the real smoke budget still has 4 strata of 40)
TINY = {
    "d_model": 8,
    "dropout": 0.0,
    "min_epochs": 0,
    "check_every": 2,
    "patience": 5,
    "batch_size": 8,
    "n_strata": 1,
    "min_per_stratum": 3,
    "n_draws": 20,
    "warmup": 1,
}


@pytest.fixture(scope="module")
def jax_run(tmp_path_factory):
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from ibnr.gallery.nn.tlrn import jax_backend

    from .test_nn_features import START, study_arrays, study_triangle

    triangle = study_triangle("duckdb", study_arrays(n_companies=5, n_w=10, seed=3))
    codes = [f"{i + 1:04d}" for i in range(5)]
    cohort = SimpleNamespace(
        name="A",
        pairs_hash="p" * 16,
        data_hash="d" * 16,
        board=codes,
        actual=pd.DataFrame(
            {"actual_reserve": [1.0] * 5, "anchor": [1.0] * 5},
            index=pd.Index(codes, name="company_code"),
        ),
        triangle=triangle,
        source="synthetic",
        description="a tiny synthetic cohort",
        as_of=dt.date(START + 9, 12, 31),
    )
    calls = []
    real = jax_backend.train_protocols

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    out = tmp_path_factory.mktemp("jax_run")
    patched = pytest.MonkeyPatch()
    patched.setattr(jax_backend, "train_protocols", spy)
    patched.setattr(study, "cells_from_mart", lambda years=None: None)
    patched.setattr(study, "build_cohort", lambda name, **kw: cohort)
    original = tlrn_colab.factor_head_config
    patched.setattr(tlrn_colab, "factor_head_config", lambda **o: original(**{**TINY, **o}))
    try:
        study.main(
            [
                "--out",
                str(out),
                "--study",
                "factor_head",
                "--cohort",
                "A",
                "--smoke",
                "--draws",
                "1",
                "--members",
                "1",
                "--rows",
                "tlrn_13_fh",
                "--quiet",
            ]
        )
    finally:
        patched.undo()
    return out, calls


def test_the_runner_trains_with_jax_and_writes_every_file_of_a_fit(jax_run):
    out, calls = jax_run
    assert calls, "backend='jax' did not reach the JAX training function"
    stem = "A_tlrn_13_fh_d00_smoke-ibnr"
    names = {p.name for p in out.iterdir()}
    suffixes = (
        ".json",
        ".pkl",
        "_reserves.csv",
        "_valuations.csv",
        "_backtest.csv",
        "_members.csv",
    )
    for suffix in suffixes:
        assert any(n.startswith(stem) and n.endswith(suffix) for n in names), suffix
    assert "manifest_factor_head_smoke.json" in names


def test_the_record_says_jax_and_the_tables_cover_the_four_valuation_dates(jax_run):
    out, _ = jax_run
    record = next(out.glob("A_tlrn_13_fh_d00_smoke-ibnr*.json"))
    info = json.loads(record.read_text(encoding="utf-8"))
    assert info["settings"]["backend"] == "jax" and info["settings"]["study"] == 2
    assert info["run"]["jax_backend"] == "cpu" and info["run"]["fit_seconds"] > 0
    valuations = pd.read_csv(next(out.glob("A_tlrn_13_fh_d00_smoke-ibnr*_valuations.csv")))
    assert sorted(valuations["valuation"].unique()) == [6, 7, 8, 10]
    assert valuations["predicted"].notna().all() and valuations["predicted_net"].notna().all()
    members = pd.read_csv(next(out.glob("A_tlrn_13_fh_d00_smoke-ibnr*_members.csv")))
    assert set(members["valuation"]) == {6, 7, 8, 10}
    assert members["alpha"].between(0, 1).all()
