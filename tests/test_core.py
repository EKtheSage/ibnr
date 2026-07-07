import datetime as dt

import pytest

from ibnr import Triangle, TriangleMeta


def test_meta_validation():
    assert TriangleMeta().grain == "OYDY"
    assert TriangleMeta(origin_grain="Y", dev_grain="Q").grain == "OYDQ"
    with pytest.raises(ValueError):
        TriangleMeta(origin_grain="W")
    with pytest.raises(ValueError):
        TriangleMeta(measure="paid")


def test_construction_and_introspection(small_cumulative):
    t = small_cumulative
    assert t.segments == ["lob"]
    assert t.fields == ["paid_loss"]
    assert t.origins == [dt.date(2020, 1, 1), dt.date(2021, 1, 1), dt.date(2022, 1, 1)]
    assert t.dev_lags == [12, 24, 36]
    assert t.eval_dates[-1] == dt.date(2022, 12, 31)
    assert t.count() == 6
    assert "OYDY" in repr(t) and "cumulative" in repr(t)


def test_missing_columns_rejected(small_cumulative):
    with pytest.raises(ValueError, match="missing core columns"):
        Triangle(small_cumulative.expr.drop("eval_date"))


def test_select_fields_and_filter(small_cumulative):
    import ibis

    assert small_cumulative.select_fields("paid_loss").count() == 6
    assert small_cumulative.filter(ibis._.dev_lag == 12).count() == 3


def test_to_wide(small_cumulative):
    wide = small_cumulative.to_wide()
    assert wide.shape == (3, 3)
    assert wide.loc[dt.date(2020, 1, 1), 36] == 175.0
    assert wide.loc[dt.date(2022, 1, 1), 12] == 120.0
