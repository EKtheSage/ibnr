import pandas as pd
import pytest

from ibnr import Triangle


def _tri(backend_name, rows):
    df = pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "value"])
    df["origin_period"] = pd.to_datetime(df["origin_period"]).dt.date
    df["eval_date"] = pd.to_datetime(df["eval_date"]).dt.date
    df["field"] = "paid_loss"
    return Triangle.from_long(df, backend=backend_name)


def test_clean_triangle_passes(small_cumulative):
    assert small_cumulative.validate(strict=True) == []


def test_duplicate_cells(backend_name):
    t = _tri(
        backend_name,
        [("2020-01-01", 12, "2020-12-31", 1.0), ("2020-01-01", 12, "2020-12-31", 2.0)],
    )
    issues = t.validate(strict=False)
    assert any("duplicated" in i for i in issues)
    with pytest.raises(ValueError, match="validation failed"):
        t.validate(strict=True)


def test_restated_cells_flagged(backend_name):
    t = _tri(
        backend_name,
        [("2020-01-01", 12, "2020-12-31", 1.0), ("2020-01-01", 12, "2021-12-31", 2.0)],
    )
    issues = t.validate(strict=False)
    assert any("multiple eval_dates" in i for i in issues)
    # eval alignment also fires for the restated row, by design
    assert any("does not align" in i for i in issues)


def test_eval_misalignment(backend_name):
    t = _tri(backend_name, [("2020-01-01", 12, "2021-06-30", 1.0)])
    assert any("does not align" in i for i in t.validate(strict=False))


def test_dev_lag_off_grain(backend_name):
    t = _tri(backend_name, [("2020-01-01", 9, "2020-09-30", 1.0)])
    assert any("multiple of 12 months" in i for i in t.validate(strict=False))
