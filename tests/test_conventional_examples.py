"""Calendar, basis and provenance checks for the transcribed-appendix adapter.

Most tests build their own data and point the loader at it, so a schema or axis
change is caught without depending on the real numbers. Two tests read the
transcribed file that ships with this repository: one shows that loading it
needs no network access at all, the other checks the published cells and totals
it has to reproduce.
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import sys
from pathlib import Path

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
    tables = {name: meta.declared_tables for name, meta in examples.EXAMPLE_METADATA.items()}
    return {"source": {"tables": tables}, "triangles": triangles}


def write_source(monkeypatch, tmp_path, data=None):
    path = tmp_path / "balona_richman_2020_appendix.json"
    path.write_text(json.dumps(synthetic_source() if data is None else data), encoding="utf-8")
    monkeypatch.setattr(examples, "DATA_PATH", path)
    return path


def test_loaded_cells_keep_basis_dates_premiums_and_fixed_horizon(
    monkeypatch, tmp_path, backend_name
):
    write_source(monkeypatch, tmp_path)
    triangles = examples.load_published_examples(backend_name)
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


def test_loading_the_repository_file_needs_no_network_access(monkeypatch):
    """Opening any socket during the load is a failure, not a slow path."""

    def refuse(*args, **kwargs):
        raise AssertionError("the loader opened a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    triangles = examples.load_published_examples()
    assert sorted(triangles) == ["liability", "property", "swiss"]
    assert examples.DATA_PATH.is_file()
    assert not [name for name in vars(examples) if "SOURCE" in name or name == "urlopen"]


def test_transcribed_appendix_reproduces_the_published_cells_and_totals(backend_name):
    triangles = examples.load_published_examples(backend_name)
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
        examples.load_published_examples()


def test_a_truncated_triangle_names_the_fields_it_lost(monkeypatch, tmp_path):
    data = synthetic_source()
    del data["triangles"][0]["values"]
    del data["triangles"][0]["earnedPremium"]
    write_source(monkeypatch, tmp_path, data)
    with pytest.raises(ValueError, match="missing values, earnedPremium"):
        examples.load_published_examples()


def test_a_damaged_file_is_refused_before_parsing(monkeypatch, tmp_path):
    path = write_source(monkeypatch, tmp_path)
    path.write_text('{"source": {"tables": {}}, "triangles": [', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        examples.load_published_examples()


def test_a_file_without_a_list_of_triangles_is_refused(monkeypatch, tmp_path):
    write_source(monkeypatch, tmp_path, {"source": {"tables": {}}})
    with pytest.raises(ValueError, match="source record and a list of triangles"):
        examples.load_published_examples()


def test_a_missing_file_names_the_path_it_expected(monkeypatch, tmp_path):
    monkeypatch.setattr(examples, "DATA_PATH", tmp_path / "absent.json")
    with pytest.raises(ValueError, match="missing from"):
        examples.load_published_examples()


@pytest.mark.parametrize("mutation", ["edition", "dropped", "absent"])
def test_table_and_page_numbers_must_match_the_declared_edition(monkeypatch, tmp_path, mutation):
    data = synthetic_source()
    if mutation == "edition":
        # The 23 April 2021 revision prints the Swiss triangle as Table 28 on
        # page 52; this module declares the 14 August 2020 numbering.
        data["source"]["tables"]["swiss"] = {
            "claimsTable": 28,
            "claimsPage": 52,
            "premiumTable": 27,
            "premiumPage": 51,
        }
    elif mutation == "dropped":
        del data["source"]["tables"]["liability"]
    else:
        del data["source"]["tables"]
    write_source(monkeypatch, tmp_path, data)
    with pytest.raises(ValueError, match="different editions|record the table and page"):
        examples.load_published_examples()


def test_every_example_declares_where_it_was_transcribed_from():
    declared = json.loads(examples.DATA_PATH.read_text(encoding="utf-8"))["source"]
    assert declared["date"] == "2020-08-14"
    assert declared["authors"] == ["Caesar Balona", "Ronald Richman"]
    for name, metadata in examples.EXAMPLE_METADATA.items():
        assert declared["tables"][name] == metadata.declared_tables
