"""The application computes its evidence on real historical data paths."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest
from apps.reserving_review.analysis import analyze_request, demo_request

from ibnr import Triangle
from ibnr.kernels.conventional import ConventionalCandidate
from ibnr.kernels.replay import replay_conventional
from ibnr.kernels.selection import score_replay

from .test_conventional_replay import full_triangle


@pytest.fixture(scope="module")
def demo_payload():
    return demo_request()


@pytest.fixture(scope="module")
def demo_snapshot(demo_payload):
    return analyze_request(demo_payload)


def test_demo_snapshot_contains_real_reproducible_evidence(demo_snapshot):
    result = demo_snapshot
    assert len(result["ranking"]) == 42
    assert len(result["origins"]) == 21
    assert result["selected"]["name"] == result["ranking"][0]["candidate"]
    assert all(r["eligible"] for r in result["ranking"])
    assert len(result["factor_summary"]) == 7
    assert len(result["history_scores"]) == 42 * 9
    assert result["source_hash"] == hashlib.sha256(result["source_csv"].encode()).hexdigest()
    assert result["engine"]["source_hashes"]
    frame = pd.read_csv(io.StringIO(result["source_csv"]))
    assert frame.eval_date.max() <= "2020-12-31"
    json.dumps(result, allow_nan=False)


def test_future_input_cannot_enter_saved_evidence_or_selection(demo_payload, demo_snapshot):
    frame = pd.read_csv(io.StringIO(demo_payload["csv"]))
    future = frame.iloc[[0]].assign(eval_date="2030-12-31", value=9e9)
    # Append without rewriting the historical decimal values through another
    # parser/formatter roundtrip: those original input bytes must stay fixed.
    payload = {
        **demo_payload,
        "csv": demo_payload["csv"] + future.to_csv(index=False, header=False),
    }
    result = analyze_request(payload)
    assert result["selected"] == demo_snapshot["selected"]
    assert result["source_hash"] == demo_snapshot["source_hash"]
    assert result["origins"] == demo_snapshot["origins"]
    assert result["warnings"]


def test_saved_source_preserves_earlier_versions_for_replay():
    tri = full_triangle("duckdb")
    frame = tri.execute()
    original = frame[(frame.field == "paid_loss") & (frame.dev_lag == 12)].iloc[[0]]
    revised = original.assign(eval_date=dt.date(2014, 12, 31), value=130.0)
    frame = pd.concat([frame, revised], ignore_index=True)
    payload = {
        "title": "Restatement test",
        "csv": frame.to_csv(index=False),
        "as_of": "2014-12-31",
        "history_start": "2013-12-31",
        "grain": "Y",
        "horizon": 48,
        "loss_field": "paid_loss",
        "premium_field": "earned_premium",
        "metric": "ave",
        "units": "test",
    }
    result = analyze_request(payload)
    saved = pd.read_csv(io.StringIO(result["source_csv"]))
    versions = saved[
        (
            pd.to_datetime(saved.origin_period).dt.date
            == pd.Timestamp(original.iloc[0].origin_period).date()
        )
        & (saved.dev_lag == 12)
        & (saved.field == "paid_loss")
    ]
    assert len(versions) == 2
    spec = ConventionalCandidate(**result["selected"]["settings"])
    for column in ("origin_period", "eval_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.date
    reference = replay_conventional(
        Triangle.from_long(frame, units="test"),
        {result["selected"]["name"]: spec},
        ["2013-12-31", "2014-12-31"],
    )
    scores = score_replay(reference)
    np.testing.assert_allclose(scores.rmse.iloc[0], result["selected"]["mean_rmse"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("metric", "invented"),
        ("grain", "W"),
        ("grain", []),
        ("horizon", 95),
        ("horizon", "96"),
        ("allow_unity", "yes"),
        ("title", ""),
        ("units", ""),
        ("history_start", "2020-12-31"),
        ("as_of", "2020-12-01"),
    ],
)
def test_invalid_form_values_are_refused(demo_payload, field, value):
    with pytest.raises(ValueError):
        analyze_request({**demo_payload, field: value})


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_header",
        "extra_field",
        "missing_field",
        "missing_value",
        "nonfinite",
        "fractional_lag",
    ],
)
def test_invalid_source_is_not_silently_repaired(demo_payload, damage):
    frame = pd.read_csv(io.StringIO(demo_payload["csv"]))
    if damage == "duplicate_header":
        csv_text = demo_payload["csv"].replace("origin_period", "eval_date", 1)
    elif damage in ("extra_field", "missing_field"):
        records = list(csv.reader(io.StringIO(demo_payload["csv"])))
        records[1:] = [
            ["unheaded_segment", *row] if damage == "extra_field" else row[:-1]
            for row in records[1:]
        ]
        output = io.StringIO()
        csv.writer(output).writerows(records)
        csv_text = output.getvalue()
    else:
        if damage == "missing_value":
            frame.loc[0, "value"] = np.nan
        elif damage == "nonfinite":
            frame.loc[0, "value"] = np.inf
        else:
            frame["dev_lag"] = frame.dev_lag.astype(float)
            frame.loc[0, "dev_lag"] = 12.5
        csv_text = frame.to_csv(index=False)
    with pytest.raises(ValueError):
        analyze_request({**demo_payload, "csv": csv_text})


def test_one_age_horizon_retains_mature_origin_restatement():
    result = analyze_request(
        {
            "csv": (
                "origin_period,dev_lag,eval_date,field,value\n"
                "2019-01-01,12,2019-12-31,paid_loss,100\n"
                "2019-01-01,12,2020-12-31,paid_loss,110\n"
                "2019-01-01,12,2019-12-31,earned_premium,200\n"
                "2020-01-01,12,2020-12-31,paid_loss,50\n"
                "2020-01-01,12,2020-12-31,earned_premium,100\n"
            ),
            "as_of": "2020-12-31",
            "history_start": "2019-12-31",
            "horizon": 12,
        }
    )
    assert result["factor_summary"] == []
    assert result["factor_selection"] == []
    assert result["selected"]["mean_rmse"] == 10.0
    assert len(result["origins"]) == 2
