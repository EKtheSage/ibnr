"""Calendar, basis and provenance checks for the published-example adapter.

Offline tests generate their own data. The separately marked network test reads
the genuine appendix and checks published cells/totals, without redistributing
that dataset in the test suite.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import sys
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import conventional_examples as examples  # noqa: E402


def synthetic_source():
    triangles = []
    for name, count, depth, grain, start, basis in (
        ("swiss", 19, 20, "Y", "1979", "paid"),
        ("liability", 20, 21, "Q", "2010Q1", "incurred"),
        ("property", 20, 21, "Q", "2010Q1", "incurred"),
    ):
        step = 12 if grain == "Y" else 3
        values = [[10000 + 100 * i + j for j in range(depth)] for i in range(count)]
        if name == "property":
            values[0][1] = 9000  # A negative increment must survive ingestion.
        triangles.append(
            {
                "id": name,
                "basis": basis,
                "valueType": "cumulative",
                "originPeriods": pd.period_range(start, periods=count, freq=grain)
                .astype(str)
                .tolist(),
                "developmentMonths": list(range(step, step * depth + 1, step)),
                "values": values,
                "earnedPremium": [50000 + i for i in range(count)],
                # Deliberately misleading: shading must not govern data dates.
                "regionsAsPrinted": [["initial"] * depth for _ in range(count)],
            }
        )
    return {"triangles": triangles}


def write_source(monkeypatch, tmp_path, data=None):
    raw = json.dumps(synthetic_source() if data is None else data).encode()
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(examples, "SOURCE_SHA256", digest)
    path = tmp_path / f"{digest}.json"
    path.write_bytes(raw)
    return path


def test_loaded_cells_keep_basis_dates_premiums_and_fixed_horizon(
    monkeypatch, tmp_path, backend_name
):
    write_source(monkeypatch, tmp_path)
    triangles = examples.load_published_examples(backend_name, cache_dir=tmp_path)
    assert set(triangles) == {"swiss", "liability", "property"}
    for name, count, depth, grain, loss, first_end, last_end in (
        ("swiss", 19, 20, "Y", "paid_loss", dt.date(1979, 12, 31), dt.date(2016, 12, 31)),
        ("liability", 20, 21, "Q", "incurred_loss", dt.date(2010, 3, 31), dt.date(2019, 12, 31)),
        ("property", 20, 21, "Q", "incurred_loss", dt.date(2010, 3, 31), dt.date(2019, 12, 31)),
    ):
        triangle = triangles[name]
        frame = triangle.to_pandas()
        assert triangle.count() == count * (depth + 1)
        assert triangle.segments == []
        assert triangle.fields == sorted([loss, "earned_premium"])
        assert triangle.meta.origin_grain == triangle.meta.dev_grain == grain
        assert triangle.meta.measure == "cumulative"
        assert triangle.meta.units == examples.UNITS
        assert triangle.eval_dates[0] == first_end
        assert triangle.eval_dates[-1] == last_end
        assert triangle.dev_lags[-1] == (240 if grain == "Y" else 63)
        premium = frame.loc[frame.field == "earned_premium"]
        assert len(premium) == count
        assert premium.dev_lag.unique().tolist() == [12 if grain == "Y" else 3]
        # Every observation must be hidden until its assigned calendar date,
        # regardless of the deliberately incorrect region labels above.
        first = triangle.as_of(first_end).to_pandas()
        assert len(first) == 2
        assert sorted(first.value) == [10000, 50000]
    property_frame = triangles["property"].to_pandas()
    values = (
        property_frame.loc[
            (property_frame.field == "incurred_loss")
            & (pd.to_datetime(property_frame.origin_period).dt.year == 2010)
            & (pd.to_datetime(property_frame.origin_period).dt.month == 1)
        ]
        .sort_values("dev_lag")
        .value
    )
    assert values.iloc[1] - values.iloc[0] == -1000


def test_download_is_verified_then_reused_offline(monkeypatch, tmp_path):
    raw = json.dumps(synthetic_source()).encode()
    monkeypatch.setattr(examples, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    calls = []

    def download(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(raw)

    monkeypatch.setattr(examples, "urlopen", download)
    assert examples._source_bytes(tmp_path) == raw
    assert examples._source_bytes(tmp_path) == raw
    assert calls == [(examples.SOURCE_URL, 30)]
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]


def test_corrupt_cache_is_refused_before_parsing(monkeypatch, tmp_path):
    path = write_source(monkeypatch, tmp_path)
    path.write_bytes(b"not the published appendix")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        examples.load_published_examples(cache_dir=tmp_path)


def test_changed_download_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(examples, "urlopen", lambda *args, **kwargs: io.BytesIO(b"changed"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        examples._source_bytes(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_failure_names_source_and_offline_cache(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(examples, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="verified copy may be placed") as error:
        examples._source_bytes(tmp_path)
    assert examples.SOURCE_URL in str(error.value)
    assert str(tmp_path) in str(error.value)


@pytest.mark.parametrize("mutation", ["duplicate", "basis", "origin", "horizon", "row"])
def test_source_axes_cannot_silently_change(monkeypatch, tmp_path, mutation):
    data = synthetic_source()
    swiss = data["triangles"][0]
    if mutation == "duplicate":
        data["triangles"][-1] = swiss
    elif mutation == "basis":
        swiss["basis"] = "incurred"
    elif mutation == "origin":
        swiss["originPeriods"][0] = "1978"
    elif mutation == "horizon":
        swiss["developmentMonths"][-1] = 252
    else:
        swiss["values"][0].pop()
    write_source(monkeypatch, tmp_path, data)
    with pytest.raises(ValueError, match="schema|exactly once"):
        examples.load_published_examples(cache_dir=tmp_path)


@pytest.mark.network
def test_published_appendix_cells_and_totals(tmp_path, backend_name):
    triangles = examples.load_published_examples(backend_name, cache_dir=tmp_path)
    for name, first_claim, last_claim, all_claims, premiums, terminal in (
        ("swiss", 3670, 31819, 7527375, 671862, 399434),
        ("liability", 2014, 16406, 3034613, 375362, 180407),
        ("property", 15886, 36437, 12713884, 1212009, 610466),
    ):
        frame = triangles[name].to_pandas()
        claims = frame.loc[frame.field != "earned_premium"].sort_values(
            ["origin_period", "dev_lag"]
        )
        assert claims.value.iloc[0] == first_claim
        assert claims.value.iloc[-1] == last_claim
        assert claims.value.sum() == all_claims
        assert frame.loc[frame.field == "earned_premium", "value"].sum() == premiums
        assert claims.loc[claims.dev_lag == claims.dev_lag.max(), "value"].sum() == terminal
