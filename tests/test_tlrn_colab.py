"""scripts/tlrn_colab.py reuses a saved fit only when it was made with the same settings.

The script resumes an interrupted Colab run by skipping fits already in its folder. It
used to recognise a fit by its row name alone, so a rerun with another ``--members`` or
``--config`` silently kept the old fits and wrote a manifest describing the new run. These
tests drive the pure pieces - the settings, the paths, the reuse decision and the manifest
- without fitting anything, so they need neither jax nor torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import tlrn_colab  # noqa: E402


def _settings(config_name="published", members=40, smoke=False):
    return {
        row: tlrn_colab.fit_settings(row, roles, cfg, config_name=config_name, smoke=smoke)
        for row, roles, cfg in tlrn_colab.configs(config_name, members, smoke)
    }


def _save(out: Path, row: str, settings: dict, *, version="9.9.9", smoke=False):
    """Stand in for a finished fit: an entry file and its settings record."""
    pkl, record = tlrn_colab.fit_paths(out, row, smoke=smoke, version=version)
    out.mkdir(parents=True, exist_ok=True)
    pkl.write_bytes(b"a fitted entry")
    tlrn_colab.write_record(record, settings, {"fit_seconds": 12.5})
    return pkl, record


def test_the_file_names_are_the_ones_the_docs_load():
    pkl, record = tlrn_colab.fit_paths(Path("fits"), "tlrn_13_ay", smoke=False, version="0.7.3")
    assert (pkl.name, record.name) == ("tlrn_13_ay-ibnr0.7.3.pkl", "tlrn_13_ay-ibnr0.7.3.json")
    assert [row for row, _, _ in tlrn_colab.configs("accident_year", 20, False)] == [
        "tlrn_8_ay",
        "tlrn_13_ay",
    ]


def test_a_fit_made_with_the_same_settings_is_reused(tmp_path):
    settings = _settings()["tlrn_8"]
    pkl, record = tlrn_colab.fit_paths(tmp_path, "tlrn_8", smoke=False, version="9.9.9")
    assert tlrn_colab.saved_fit(pkl, record, settings) is None  # nothing saved yet
    _save(tmp_path, "tlrn_8", settings)
    saved = tlrn_colab.saved_fit(pkl, record, _settings()["tlrn_8"])
    assert saved == {"settings": settings, "run": {"fit_seconds": 12.5}}


def test_a_fit_made_with_other_members_is_refused_not_reused(tmp_path):
    """The bug: ``--members 20`` after a 40-member run skipped both fits."""
    _save(tmp_path, "tlrn_8", _settings(members=40)["tlrn_8"])
    pkl, record = tlrn_colab.fit_paths(tmp_path, "tlrn_8", smoke=False, version="9.9.9")
    with pytest.raises(SystemExit, match=r"config\.ensemble_size, config\.keep"):
        tlrn_colab.saved_fit(pkl, record, _settings(members=20)["tlrn_8"])
    assert pkl.read_bytes() == b"a fitted entry"  # and left alone, not overwritten


def test_other_fit_settings_are_refused_too(tmp_path):
    _save(tmp_path, "tlrn_8", _settings()["tlrn_8"])
    pkl, record = tlrn_colab.fit_paths(tmp_path, "tlrn_8", smoke=False, version="9.9.9")
    other = _settings()["tlrn_8"] | {"seed": 12}
    with pytest.raises(SystemExit, match="seed"):
        tlrn_colab.saved_fit(pkl, record, other)
    epochs = _settings()["tlrn_8"]
    epochs["config"] = epochs["config"] | {"max_epochs": 7}
    with pytest.raises(SystemExit, match=r"config\.max_epochs"):
        tlrn_colab.saved_fit(pkl, record, epochs)


def test_a_fit_with_no_settings_record_is_refused(tmp_path):
    """An older run's file, or one interrupted between writing the entry and its record."""
    pkl, record = _save(tmp_path, "tlrn_8", _settings()["tlrn_8"])
    record.unlink()
    with pytest.raises(SystemExit, match="no settings record"):
        tlrn_colab.saved_fit(pkl, record, _settings()["tlrn_8"])


def test_the_manifest_describes_the_fits_from_their_own_records(tmp_path):
    settings = _settings()
    for row, s in settings.items():
        _save(tmp_path, row, s)
    records = {}
    for row, s in settings.items():
        pkl, record = tlrn_colab.fit_paths(tmp_path, row, smoke=False, version="9.9.9")
        records[row] = tlrn_colab.saved_fit(pkl, record, s)
    written = json.loads(
        json.dumps(tlrn_colab.manifest(records, config_name="published", smoke=False))
    )
    assert sorted(written["fits"]) == ["tlrn_13", "tlrn_8"]
    for row in settings:
        assert written["fits"][row]["settings"]["config"]["ensemble_size"] == 40
        assert written["fits"][row]["run"]["fit_seconds"] == 12.5
    assert written["fits"]["tlrn_13"]["settings"]["roles"] == {
        "incurred_field": "incurred_loss",
        "case_field": "case_reserve",
    }
