# Point scores (PR A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Point-error metrics (Pool_APE, Pool_PE, MAE, RMSE and the secondary ones) implemented once in `kernels/point_scores.py`, with the aggregation level explicit, reachable from `GalleryEntry.evaluate()` and from a cross-model reserve table.

**Architecture:** One torch-free kernel module holding four pure functions (`point_metrics`, `level_errors`, `shrink_toward`, `reserve_rows`) plus a `point_summary` helper that `evaluate()` calls. `mack` gains a deterministic `point()` so the reserve table can read a native point. Exports on `ibnr.kernels` and (for `reserve_rows` only) on `ibnr.gallery`.

**Tech Stack:** Python 3.11/3.12, numpy, pandas, pytest. No torch anywhere in this PR.

**Spec:** `docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md` (section 5.1). Read it first; the plan argues from it.

## Global Constraints

- `kernels/` never imports `ibnr.gallery` (subprocess-tested by `tests/test_gallery.py::test_kernels_never_imports_the_gallery`). `reserve_rows` takes a fitted entry as an argument and only calls its public methods; it imports nothing from the gallery.
- Torch must not be imported at module level anywhere in this PR (`tests/test_import_purity.py`).
- Lint: `uv run ruff check . && uv run ruff format .` and `uv run python scripts/lint_md_snippets.py` must pass.
- Commit messages: conventional prefix (`feat:`, `test:`, `docs:`), subject plus body, NO `Co-Authored-By` or "generated with" lines, in commits or in the PR body.
- Prose rules for docstrings, comments, CHANGELOG and cards: plain sentences. Never use the words "screen", "panel", "gate", "fingerprint", "membership", "ablation" (or any form of it), "seed noise", "chip", "drain". Existing API names such as `align_panel` may be written in backticks when naming the API.
- A test for a refusal must use `pytest.raises(..., match=...)`, never a bare `pytest.raises(ValueError)`.
- Every claimed refusal or identity in a new test file is checked by breaking it once on purpose (a mutation check) and watching the test go red; the PR body lists the mutations tried.
- Work on branch `feat/point-scores` in worktree `C:\Users\EthanKang\Projects\ibnr-wt\point-scores`, created from `origin/main`. Before committing and again before opening the PR run `git log --oneline HEAD..origin/main`; if non-empty, `git rebase origin/main` and rerun the tests.
- Dependencies are unchanged. `uv sync` in the worktree is enough for this PR.

---

### Task 1: `point_metrics`, `level_errors`, `shrink_toward`

**Files:**
- Create: `src/ibnr/kernels/point_scores.py`
- Test: `tests/test_point_scores.py`

**Interfaces:**
- Produces:
  - `point_metrics(predicted, actual) -> dict[str, float]` with keys `n`, `mae`, `rmse`, `wrmse`, `mape`, `medape`, `pool_ape`, `pool_pe`, `prop_over`.
  - `level_errors(frame, *, predicted, actual, level) -> pd.DataFrame` with the `level` columns, `n_rows`, `predicted`, `actual`, `error`.
  - `shrink_toward(point, baseline, alpha) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
"""kernels.point_scores: point-error metrics with the aggregation level explicit.

Errors are summed WITHIN a level before the absolute value is taken. That is
where cancellation between lines happens, and the level is named on every call
so the reader can tell a company-level Pool_APE from a line-level one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.point_scores import level_errors, point_metrics, shrink_toward

# Company 671 from the 2026-09-20 reconciliation, USD thousands. The three
# overestimates almost offset the personal-auto underestimate.
COMPANY_671 = pd.DataFrame(
    {
        "company_code": ["671"] * 4,
        "line_of_business": [
            "commercial_auto",
            "other_liability",
            "private_passenger_auto",
            "workers_compensation",
        ],
        "predicted": [9761.03, 2336.71, 92807.64, 27312.97],
        "actual": [9746.0, 2009.0, 93662.0, 26811.0],
    }
)


def test_company_level_sums_before_the_absolute_value():
    """Company error is the sum of signed line errors (-9.65); the sum of absolute
    line errors is 1699.07. Swapping the order of sum and abs must fail this."""
    company = level_errors(COMPANY_671, predicted="predicted", actual="actual", level=["company_code"])
    assert list(company.columns) == ["company_code", "n_rows", "predicted", "actual", "error"]
    assert len(company) == 1
    assert company["n_rows"].iloc[0] == 4
    assert company["error"].iloc[0] == pytest.approx(-9.65, abs=0.01)
    pair = level_errors(
        COMPANY_671,
        predicted="predicted",
        actual="actual",
        level=["company_code", "line_of_business"],
    )
    assert len(pair) == 4
    assert pair["error"].abs().sum() == pytest.approx(1699.07, abs=0.01)


def test_level_errors_refuses_a_missing_level_column_and_a_missing_forecast():
    with pytest.raises(KeyError, match="level column"):
        level_errors(COMPANY_671, predicted="predicted", actual="actual", level=["state"])
    holed = COMPANY_671.assign(predicted=[9761.03, np.nan, 92807.64, 27312.97])
    with pytest.raises(ValueError, match="non-finite"):
        level_errors(holed, predicted="predicted", actual="actual", level=["company_code"])


def test_point_metrics_closed_forms():
    m = point_metrics([110.0, 90.0], [100.0, 100.0])
    assert m["n"] == 2
    assert m["mae"] == pytest.approx(10.0)
    assert m["rmse"] == pytest.approx(10.0)
    assert m["wrmse"] == pytest.approx(10.0)
    assert m["mape"] == pytest.approx(0.1)
    assert m["medape"] == pytest.approx(0.1)
    assert m["pool_ape"] == pytest.approx(0.10)
    assert m["pool_pe"] == pytest.approx(0.0)
    assert m["prop_over"] == pytest.approx(0.5)


def test_pool_ape_weights_by_reserve_size_and_mape_does_not():
    """Pool_APE is a dollar-weighted ratio: a 1 unit miss on a 1 unit reserve
    beside a perfect 99 unit reserve is 1 percent, while MAPE calls it 50 percent."""
    m = point_metrics([2.0, 99.0], [1.0, 99.0])
    assert m["pool_ape"] == pytest.approx(0.01)
    assert m["mape"] == pytest.approx(0.5)


def test_mape_and_medape_skip_zero_actuals_and_pool_ape_does_not():
    m = point_metrics([5.0, 3.0], [0.0, 2.0])
    assert m["mape"] == pytest.approx(0.5)
    assert m["medape"] == pytest.approx(0.5)
    assert m["pool_ape"] == pytest.approx(3.0)


def test_wrmse_weights_squared_errors_by_actual():
    """wRMSE = sqrt(sum(actual * e^2) / sum(actual)), the R study's definition."""
    predicted, actual = np.array([12.0, 100.0]), np.array([10.0, 90.0])
    m = point_metrics(predicted, actual)
    want = np.sqrt((10.0 * 4.0 + 90.0 * 100.0) / 100.0)
    assert m["wrmse"] == pytest.approx(want)


def test_point_metrics_refusals():
    with pytest.raises(ValueError, match="non-finite"):
        point_metrics([1.0, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        point_metrics([1.0, 1.0], [1.0, np.inf])
    with pytest.raises(ValueError, match="non-positive"):
        point_metrics([1.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="non-positive"):
        point_metrics([1.0, 1.0], [-3.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        point_metrics([1.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="at least one"):
        point_metrics([], [])


def test_shrink_toward_endpoints_and_the_published_weight():
    point, baseline = np.array([120.0, 80.0]), np.array([100.0, 100.0])
    np.testing.assert_allclose(shrink_toward(point, baseline, 0.0), baseline)
    np.testing.assert_allclose(shrink_toward(point, baseline, 1.0), point)
    np.testing.assert_allclose(shrink_toward(point, baseline, 0.658), [113.16, 86.84])


def test_shrink_toward_refusals():
    with pytest.raises(ValueError, match="alpha"):
        shrink_toward([1.0], [1.0], 1.5)
    with pytest.raises(ValueError, match="same shape"):
        shrink_toward([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(ValueError, match="non-finite"):
        shrink_toward([np.nan], [1.0], 0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_point_scores.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ibnr.kernels.point_scores'`.

- [ ] **Step 3: Write the implementation**

```python
"""Point-error metrics, implemented once, with the aggregation level explicit.

Every function here scores a POINT forecast against a realized value. Nothing
here touches draws; CRPS and PIT live in ``kernels.scores`` and
``kernels.predictive``.

The one idea to carry away: errors are summed WITHIN a level before the
absolute value is taken. On company 671 of the 2026-09-20 reconciliation the
four line errors were +15.03, +327.71, -854.36 and +501.97 (USD thousands): the
company error is their signed sum, -9.65, while the sum of their absolute
values is 1699.07. A company-level Pool_APE therefore credits a model for
overestimating one line and underestimating another, and a line-level one does
not. Neither is wrong; they answer different questions, which is why
:func:`level_errors` takes the level as an argument and names it in its output.

Definitions (``e = predicted - actual`` per unit of the chosen level):

- ``mae``       mean |e|
- ``rmse``      sqrt(mean e^2)
- ``wrmse``     sqrt(sum(actual * e^2) / sum(actual)) - the actual-weighted form
- ``mape``      mean |e / actual| over units with a nonzero actual
- ``medape``    median |e / actual| over the same units
- ``pool_ape``  sum |e| / sum actual - the R study's headline
- ``pool_pe``   sum e / sum actual - the signed version, a bias measure
- ``prop_over`` share of units with e > 0

A missing forecast is refused, never dropped: on a fixed cohort set a model that
cannot forecast a unit must not score on a smaller set than its neighbours.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["level_errors", "point_metrics", "point_summary", "reserve_rows", "shrink_toward"]


def point_metrics(predicted, actual) -> dict[str, float]:
    """Point-error metrics over aligned units of one level. See the module docstring."""
    p = np.asarray(predicted, dtype=float).reshape(-1)
    a = np.asarray(actual, dtype=float).reshape(-1)
    if p.shape != a.shape:
        raise ValueError(
            f"predicted and actual must have the same length, got {p.size} and {a.size}"
        )
    if p.size == 0:
        raise ValueError("point_metrics needs at least one unit")
    bad_p, bad_a = ~np.isfinite(p), ~np.isfinite(a)
    if bad_p.any() or bad_a.any():
        raise ValueError(
            f"{int(bad_p.sum())} non-finite predicted and {int(bad_a.sum())} non-finite actual "
            "value(s); a missing forecast on a fixed cohort set is an error, not a smaller set"
        )
    total = float(a.sum())
    if total <= 0:
        raise ValueError(
            f"sum of actual is non-positive ({total}); pool_ape and pool_pe divide by it"
        )
    e = p - a
    nonzero = a != 0
    ape = np.abs(e[nonzero] / a[nonzero])
    return {
        "n": int(p.size),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "wrmse": float(np.sqrt(np.sum(a * e**2) / total)),
        "mape": float(np.mean(ape)) if ape.size else float("nan"),
        "medape": float(np.median(ape)) if ape.size else float("nan"),
        "pool_ape": float(np.sum(np.abs(e)) / total),
        "pool_pe": float(np.sum(e) / total),
        "prop_over": float(np.mean(e > 0)),
    }


def level_errors(
    frame: pd.DataFrame, *, predicted: str, actual: str, level: Sequence[str]
) -> pd.DataFrame:
    """Sum ``predicted`` and ``actual`` within ``level`` and return one row per unit.

    Columns: the ``level`` columns, ``n_rows`` (rows summed into the unit),
    ``predicted``, ``actual``, ``error``. Feed the result to :func:`point_metrics`.
    """
    level = list(level)
    missing = [c for c in level if c not in frame.columns]
    if missing:
        raise KeyError(f"level column(s) {missing} not in the frame; it has {list(frame.columns)}")
    for name in (predicted, actual):
        if name not in frame.columns:
            raise KeyError(f"column {name!r} not in the frame; it has {list(frame.columns)}")
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{int((~np.isfinite(values)).sum())} non-finite value(s) in {name!r}; a "
                "missing forecast on a fixed cohort set is an error, not a smaller set"
            )
    grouped = frame.groupby(level, sort=True, dropna=False)
    out = grouped.agg(
        n_rows=(predicted, "size"), predicted=(predicted, "sum"), actual=(actual, "sum")
    ).reset_index()
    out["error"] = out["predicted"] - out["actual"]
    return out[[*level, "n_rows", "predicted", "actual", "error"]]


def shrink_toward(point, baseline, alpha: float) -> np.ndarray:
    """``baseline + alpha * (point - baseline)``: a point pulled toward a baseline.

    The R study's final estimator is ``shrink_toward(raw TLRN, MCL, 0.658)``.
    ``alpha = 0`` is the baseline, ``alpha = 1`` the point itself.
    """
    if not (isinstance(alpha, (int, float)) and np.isfinite(alpha) and 0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be a finite number in [0, 1], got {alpha!r}")
    p = np.asarray(point, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if p.shape != b.shape:
        raise ValueError(f"point and baseline must have the same shape, got {p.shape} and {b.shape}")
    if not (np.isfinite(p).all() and np.isfinite(b).all()):
        raise ValueError("point and baseline must be finite; a non-finite entry cannot be blended")
    return b + alpha * (p - b)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_point_scores.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation check, then commit**

Swap `out["error"] = out["predicted"] - out["actual"]` for a per-row absolute error summed within the level; `test_company_level_sums_before_the_absolute_value` must fail. Remove the `nonzero` mask; `test_mape_and_medape_skip_zero_actuals_and_pool_ape_does_not` must fail. Restore both.

```bash
git add src/ibnr/kernels/point_scores.py tests/test_point_scores.py
git commit -m "feat: point-error metrics with the aggregation level explicit

point_metrics, level_errors and shrink_toward in kernels/point_scores.py.
Errors are summed within the named level before the absolute value, which
is the cancellation the company-level Pool_APE carries by design."
```

---

### Task 2: `mack.point()` and `reserve_rows`

**Files:**
- Modify: `src/ibnr/gallery/deterministic/mack/model.py` (add `point`)
- Modify: `src/ibnr/kernels/point_scores.py` (add `reserve_rows`)
- Test: `tests/test_point_scores.py` (append), `tests/test_mack.py` (append one test)

**Interfaces:**
- Consumes: `entry.cohorts()`, `entry.cohort_index(segment)`, `entry.predict(segment=..., **kw)`, `entry.realized_ultimates(full_triangle, segment=...)`, `Triangle.as_of`, `Triangle.select_fields`, `Triangle.latest_diagonal`, `Triangle.execute`, `Triangle.segments`.
- Produces:
  - `Mack.point(segment=None) -> pd.DataFrame` with columns `label`, `origin_period`, `point` (the deterministic ultimate per origin plus a `total` row), in the same row order as `predict().targets`.
  - `reserve_rows(entry, full_triangle, *, as_of, loss_field, premium_field="earned_premium", point="draw_mean", segment=None, predict_kwargs=None) -> pd.DataFrame` with columns: the entry's cohort key columns, `predicted_ultimate`, `realized_ultimate`, `anchor`, `predicted_reserve`, `actual_reserve`, `premium`, `n_draws`, `point_source`.

Read first: `src/ibnr/gallery/deterministic/mack/model.py` (how `predict` builds its targets through `kernels.mack.simulate_ultimates`; the `MackFit.ultimate`, `.reserve`, `.latest` properties in `src/ibnr/kernels/mack.py`), `src/ibnr/kernels/multiline.py::multiline_targets` (the multi-line target layout: per-(lob, origin) rows, per-lob `"<lob>/total"` rows, one `"total"` row), `tests/conftest.py::make_cohort_triangle` and `make_multiline_triangle`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_point_scores.py`:

```python
import datetime as dt

from ibnr import gallery
from ibnr.kernels.point_scores import reserve_rows

from .conftest import make_cohort_triangle, make_multiline_triangle

# a full 4 x 4 square: as_of at the third diagonal leaves realized values to score
SQUARE = np.array(
    [
        [100.0, 150.0, 175.0, 185.0],
        [110.0, 165.0, 190.0, 200.0],
        [120.0, 180.0, 210.0, 222.0],
        [130.0, 195.0, 230.0, 240.0],
    ]
)
AS_OF = dt.date(2012, 12, 31)  # origins 2010..2013; the 2012 valuation sees a 3-deep triangle


def test_mack_point_is_the_deterministic_ultimate(backend_name):
    tri = make_cohort_triangle(backend_name, SQUARE, start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of=AS_OF)
    frame = entry.point()
    assert list(frame.columns) == ["label", "origin_period", "point"]
    labels = frame["label"].astype(str).tolist()
    assert labels == entry.predict(n_draws=10).targets["label"].astype(str).tolist()
    assert labels[-1] == "total"
    np.testing.assert_allclose(frame["point"].to_numpy()[:-1], entry.fit_.ultimate)
    assert frame["point"].iloc[-1] == pytest.approx(entry.fit_.ultimate.sum())


def test_reserve_rows_for_a_single_cohort_entry(backend_name):
    tri = make_cohort_triangle(backend_name, SQUARE, start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of=AS_OF)
    rows = reserve_rows(
        entry, tri, as_of=AS_OF, loss_field="paid_loss", premium_field=None, point="native"
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    # anchor = latest observed cumulative at the valuation, summed over origins:
    # 2010 at dev 3, 2011 at dev 2, 2012 at dev 1 (2013 is not yet written at as_of)
    assert row["anchor"] == pytest.approx(175.0 + 165.0 + 120.0)
    assert row["realized_ultimate"] == pytest.approx(185.0 + 200.0 + 222.0)
    assert row["actual_reserve"] == pytest.approx(row["realized_ultimate"] - row["anchor"])
    assert row["predicted_ultimate"] == pytest.approx(entry.fit_.ultimate.sum())
    assert row["predicted_reserve"] == pytest.approx(row["predicted_ultimate"] - row["anchor"])
    assert row["point_source"] == "native"
    assert np.isnan(row["premium"])

    draws = reserve_rows(
        entry,
        tri,
        as_of=AS_OF,
        loss_field="paid_loss",
        premium_field=None,
        predict_kwargs={"seed": 3, "n_draws": 2000},
    )
    assert draws["point_source"].iloc[0] == "draw_mean"
    assert draws["n_draws"].iloc[0] == 2000
    # the draw mean sits near the deterministic point but is not it
    assert draws["predicted_ultimate"].iloc[0] == pytest.approx(row["predicted_ultimate"], rel=0.05)


def test_reserve_rows_for_a_multi_line_entry_sums_the_company(backend_name):
    lobs = {"auto": SQUARE, "liab": SQUARE * 2.0}
    prem = {"auto": np.full(4, 1000.0), "liab": np.full(4, 3000.0)}
    tri = make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=2010)
    entry = gallery.fit("sur", tri, loss_field="paid_loss", as_of=AS_OF)
    rows = reserve_rows(
        entry, tri, as_of=AS_OF, loss_field="paid_loss", predict_kwargs={"seed": 5, "n_draws": 500}
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["anchor"] == pytest.approx(3 * (175.0 + 165.0 + 120.0))
    assert row["realized_ultimate"] == pytest.approx(3 * (185.0 + 200.0 + 222.0))
    # premium is summed over the origins the valuation has written, on both lines
    assert row["premium"] == pytest.approx(3 * 1000.0 + 3 * 3000.0)
    assert row["point_source"] == "draw_mean"


def test_reserve_rows_refusals(backend_name):
    tri = make_cohort_triangle(backend_name, SQUARE, start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of=AS_OF)
    with pytest.raises(ValueError, match="no field named 'earned_premium'"):
        reserve_rows(entry, tri, as_of=AS_OF, loss_field="paid_loss")
    with pytest.raises(ValueError, match="point must be"):
        reserve_rows(entry, tri, as_of=AS_OF, loss_field="paid_loss", premium_field=None, point="median")

    class NoPoint:
        """An entry-shaped object without point(); the native route must refuse it by name."""

        name = "stub"

        def cohorts(self):
            return entry.cohorts()

        def cohort_index(self, segment):
            return entry.cohort_index(segment)

        def predict(self, segment=None, **kw):
            return entry.predict(segment=segment, **kw)

        def realized_ultimates(self, full, segment=None):
            return entry.realized_ultimates(full, segment=segment)

    with pytest.raises(TypeError, match="does not implement point"):
        reserve_rows(
            NoPoint(), tri, as_of=AS_OF, loss_field="paid_loss", premium_field=None, point="native"
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_point_scores.py -q -k "point_is or reserve_rows"`
Expected: FAIL (`AttributeError: 'Mack' object has no attribute 'point'`, `ImportError` for `reserve_rows`).

- [ ] **Step 3: Implement `Mack.point`**

In `src/ibnr/gallery/deterministic/mack/model.py`, beside `predict`:

```python
    def point(self, segment: Mapping | None = None) -> pd.DataFrame:
        """The deterministic chain-ladder ultimates, one row per origin plus ``total``.

        Same rows, in the same order, as ``predict().targets``, so a caller can
        read the point and the draws off the same labels. Read from ``fit_``
        directly: no simulation, so ``seed`` has nothing to do here.
        """
        self.cohort_index(segment)
        fit = self._fitted()
        ultimates = np.asarray(fit.ultimate, dtype=float)
        labels = [*(str(o.year) for o in fit.origin_periods), "total"]
        return pd.DataFrame(
            {
                "label": labels,
                "origin_period": [*fit.origin_periods, None],
                "point": [*ultimates.tolist(), float(ultimates.sum())],
            }
        )
```

Check against `predict().targets` while writing it: if the targets' labels are not the bare accident years, use whatever `simulate_ultimates` builds (the test compares the two label lists, so a mismatch fails loudly). Import `pandas as pd` at the top of the module if it is not there.

- [ ] **Step 4: Implement `reserve_rows` and the shared anchor helper**

Append to `src/ibnr/kernels/point_scores.py`:

```python
POINT_SOURCES: tuple[str, ...] = ("draw_mean", "native")


def _cohort_frame(triangle, field: str, cohort: dict) -> pd.DataFrame:
    """Latest-diagonal rows of ``field`` for one cohort of ``triangle``.

    ``cohort`` may carry columns the triangle does not (a display column a pooled
    fit kept beside its key); only the shared columns filter, and every shared
    column must match.
    """
    frame = triangle.select_fields(field).latest_diagonal().execute()
    keep = np.ones(len(frame), dtype=bool)
    for column, value in cohort.items():
        if column in frame.columns:
            keep &= frame[column].astype(str).to_numpy() == str(value)
    return frame[keep]


def _total_index(labels: list[str], what: str) -> int:
    if "total" not in labels:
        raise ValueError(
            f"{what} carries no 'total' row (labels: {labels[:6]}...); reserve_rows reads the "
            "cohort's total ultimate off that row"
        )
    return labels.index("total")


def reserve_rows(
    entry,
    full_triangle,
    *,
    as_of,
    loss_field: str,
    premium_field: str | None = "earned_premium",
    point: str = "draw_mean",
    segment=None,
    predict_kwargs: dict | None = None,
) -> pd.DataFrame:
    """One row per cohort: predicted and actual reserve at the grid endpoint.

    ``reserve = ultimate - anchor``, the anchor being the cumulative observed at
    ``as_of`` summed over the cohort's origins, read from
    ``full_triangle.as_of(as_of)`` - the same slice the entry was fitted on.
    ``realized_ultimate`` comes from ``entry.realized_ultimates``, which restricts
    itself to the training origins, so both sides of the subtraction cover the
    same accident years.

    ``point="draw_mean"`` reads the mean of ``entry.predict()``'s ``total`` row;
    ``point="native"`` reads the ``total`` row of ``entry.point()`` and refuses an
    entry that has none. ``point_source`` records which. ``predict_kwargs`` are
    handed to ``predict`` unchanged (``seed``, ``n_draws``), so an entry that does
    not accept one raises rather than silently ignoring it.

    ``premium`` is the premium field's latest value per origin, summed over the
    origins written at ``as_of``; ``premium_field=None`` records NaN. Feed the
    rows to :func:`level_errors` with ``level=["company_code"]`` for a company
    board, or with the full cohort key for a pair board.
    """
    if point not in POINT_SOURCES:
        raise ValueError(f"point must be one of {POINT_SOURCES}, got {point!r}")
    if point == "native" and not callable(getattr(entry, "point", None)):
        name = getattr(entry, "name", type(entry).__name__)
        raise TypeError(
            f"{name} does not implement point(); it has draws only, so use point='draw_mean'"
        )
    training = full_triangle.as_of(as_of)
    if loss_field not in training.fields:
        raise ValueError(f"no field named {loss_field!r}; the triangle carries {sorted(training.fields)}")
    if premium_field is not None and premium_field not in training.fields:
        raise ValueError(
            f"no field named {premium_field!r}; the triangle carries {sorted(training.fields)}. "
            "Pass premium_field=None to record no premium"
        )
    kwargs = dict(predict_kwargs or {})
    cohorts = entry.cohorts()
    ci = entry.cohort_index(segment)
    chosen = cohorts if ci is None else [cohorts[ci]]
    rows = []
    for cohort in chosen:
        pred = entry.predict(segment=cohort, **kwargs)
        labels = pred.targets["label"].astype(str).tolist()
        i = _total_index(labels, f"{getattr(entry, 'name', type(entry).__name__)}.predict targets")
        realized = np.asarray(entry.realized_ultimates(full_triangle, segment=cohort), dtype=float)
        if point == "native":
            frame = entry.point(segment=cohort)
            j = _total_index(frame["label"].astype(str).tolist(), "point() frame")
            predicted = float(frame["point"].iloc[j])
        else:
            predicted = float(pred.mean()[i])
        anchor = float(_cohort_frame(training, loss_field, cohort)["value"].sum())
        premium = (
            float(_cohort_frame(training, premium_field, cohort)["value"].sum())
            if premium_field is not None
            else float("nan")
        )
        rows.append(
            {
                **cohort,
                "predicted_ultimate": predicted,
                "realized_ultimate": float(realized[i]),
                "anchor": anchor,
                "predicted_reserve": predicted - anchor,
                "actual_reserve": float(realized[i]) - anchor,
                "premium": premium,
                "n_draws": int(pred.n_draws),
                "point_source": point,
            }
        )
    return pd.DataFrame(rows)
```

Two things to check while implementing, both against the code rather than this plan: whether `Triangle.fields` is the attribute name for the set of fields (it is used as `triangle.fields` in `kernels/nn_contract.py`; if it is a method, call it); and whether `latest_diagonal()` on a sliced triangle returns one row per (cohort, origin) for the field (it should; `kernels/nn_contract.py` uses the same call for premium). If `make_cohort_triangle` rows carry no segment column, `_cohort_frame` filters on nothing and returns every row, which is the single cohort.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_point_scores.py tests/test_mack.py -q`
Expected: all PASS on both backends.

- [ ] **Step 6: Mutation check, then commit**

Make `reserve_rows` ignore `point` and always use the draw mean; `test_reserve_rows_for_a_single_cohort_entry` must fail on `point_source` and on the exact `predicted_ultimate`. Make `_cohort_frame` skip the cohort filter; the multi-line test must still pass (one company) but add, and keep, a second company to `test_reserve_rows_for_a_multi_line_entry_sums_the_company` (`company="0002"` with `SQUARE * 5`) fitted separately, and assert its anchor is `3 * 5 * (175 + 165 + 120)`: with the filter removed the anchor doubles and the test fails. Restore.

```bash
git add src/ibnr/kernels/point_scores.py src/ibnr/gallery/deterministic/mack/model.py tests/test_point_scores.py
git commit -m "feat: reserve_rows builds the predicted-versus-actual reserve table for any entry

mack gains point(), the deterministic ultimates in predict()'s target
order. reserve_rows reads the total row of predict() or point(), the
anchor and premium from the as_of slice, and records which point it used."
```

---

### Task 3: `evaluate()` reports point errors

**Files:**
- Modify: `src/ibnr/kernels/point_scores.py` (add `point_summary`)
- Modify: `src/ibnr/gallery/entry.py:GalleryEntry.evaluate`
- Test: `tests/test_point_scores.py` (append)

**Interfaces:**
- Produces: `point_summary(pred, observed) -> dict` with keys `errors` (DataFrame: the targets plus `estimate`, `outcome`, `error`, `pct_error`), `metrics` (a `point_metrics` dict or `None`), `excluded` (dict with `total` and `missing_outcome` counts).
- `GalleryEntry.evaluate()` returns `{"summary", "percentiles", "crps", "point"}`.

- [ ] **Step 1: Write the failing test**

```python
from ibnr.kernels.point_scores import point_summary
from ibnr.kernels.predictive import PredictiveDistribution


def test_point_summary_excludes_the_total_row_and_missing_outcomes():
    samples = np.array([[10.0, 20.0, 30.0], [12.0, 22.0, 34.0]])  # third column = the total
    targets = pd.DataFrame({"label": ["2010", "2011", "total"]})
    pred = PredictiveDistribution(samples=samples, targets=targets)
    out = point_summary(pred, [11.0, np.nan, 32.0])
    errors = out["errors"]
    assert list(errors["label"]) == ["2010", "2011", "total"]
    np.testing.assert_allclose(errors["estimate"], [11.0, 21.0, 32.0])
    np.testing.assert_allclose(errors["error"], [0.0, np.nan, 0.0])
    np.testing.assert_allclose(errors["pct_error"], [0.0, np.nan, 0.0])
    assert out["excluded"] == {"total": 1, "missing_outcome": 1}
    assert out["metrics"]["n"] == 1
    assert out["metrics"]["pool_ape"] == pytest.approx(0.0)


def test_point_summary_with_nothing_scorable_reports_none():
    pred = PredictiveDistribution(
        samples=np.array([[1.0], [3.0]]), targets=pd.DataFrame({"label": ["total"]})
    )
    out = point_summary(pred, [2.0])
    assert out["metrics"] is None
    assert out["excluded"] == {"total": 1, "missing_outcome": 0}


def test_evaluate_carries_the_point_block(backend_name):
    tri = make_cohort_triangle(backend_name, SQUARE, start_year=2010)
    entry = gallery.fit("mack", tri, loss_field="paid_loss", as_of=AS_OF)
    outcome = entry.realized_ultimates(tri)
    scored = entry.evaluate(outcome)
    assert set(scored) == {"summary", "percentiles", "crps", "point"}
    point = scored["point"]
    assert len(point["errors"]) == len(scored["summary"])
    assert point["metrics"]["n"] == 3  # three origins written at as_of; the total row is excluded
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_point_scores.py -q -k "point_summary or carries_the_point"`
Expected: FAIL (`ImportError` for `point_summary`; `evaluate` has no `point` key).

- [ ] **Step 3: Implement**

Append to `src/ibnr/kernels/point_scores.py`:

```python
def point_summary(pred, observed) -> dict:
    """Point errors of a ``PredictiveDistribution``'s draw means against outcomes.

    ``errors`` has one row per target: the target metadata plus ``estimate``
    (draw mean), ``outcome``, ``error`` and ``pct_error`` (NaN where the outcome
    is missing or zero). ``metrics`` is :func:`point_metrics` over the targets
    that are scorable: rows whose ``label`` is not ``"total"`` (the total is the
    sum of the others and would count every error twice) and whose outcome is
    finite. ``None`` when nothing is scorable. ``excluded`` counts both reasons.
    """
    obs = np.asarray(observed, dtype=float).reshape(-1)
    if obs.shape != (pred.n_targets,):
        raise ValueError(f"observed must have shape ({pred.n_targets},), got {obs.shape}")
    estimate = pred.mean()
    errors = pred.targets.copy()
    errors["estimate"] = estimate
    errors["outcome"] = obs
    errors["error"] = estimate - obs
    with np.errstate(divide="ignore", invalid="ignore"):
        errors["pct_error"] = np.where(obs != 0, (estimate - obs) / obs, np.nan)
    is_total = (
        errors["label"].astype(str).to_numpy() == "total"
        if "label" in errors.columns
        else np.zeros(len(errors), dtype=bool)
    )
    missing = ~np.isfinite(obs)
    scorable = ~is_total & ~missing
    metrics = point_metrics(estimate[scorable], obs[scorable]) if scorable.any() else None
    return {
        "errors": errors,
        "metrics": metrics,
        "excluded": {"total": int(is_total.sum()), "missing_outcome": int(missing.sum())},
    }
```

`point_metrics` refuses a non-positive actual total; a cohort whose scorable outcomes sum to zero or less would raise here. Catch that one `ValueError` and set `metrics = None`, recording `"non_positive_actual": 1` in `excluded`, so `evaluate()` never dies on a pathological cohort; add a test row for it (`observed = [0.0, 0.0, 0.0]` on the two-target-plus-total example must give `metrics is None`).

In `src/ibnr/gallery/entry.py`, add the import `from ibnr.kernels.point_scores import point_summary` beside the `crps` import and extend `evaluate`:

```python
        return {
            "summary": table,
            "percentiles": pred.cdf(obs) * 100.0,
            "crps": crps(pred.samples, obs),
            "point": point_summary(pred, obs),
        }
```

Update the docstring's first paragraph: "the Meyers-style summary table, the outcome percentile of each target, the CRPS of each target, and the point errors of the draw means (``point_summary``: per-target error and the level metrics over the non-total targets)".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_point_scores.py tests/test_mack.py tests/test_entry_contract.py tests/test_meyers_ccl.py tests/test_meyers_csr.py -q`
Expected: PASS (the Stan-backed tests in the last two files skip or stub; nothing there asserts the exact key set of `evaluate()`. If one does, extend its expected set.)

- [ ] **Step 5: Commit**

```bash
git add src/ibnr/kernels/point_scores.py src/ibnr/gallery/entry.py tests/test_point_scores.py
git commit -m "feat: evaluate() reports point errors beside CRPS and percentiles

point_summary scores the draw means per target and the level metrics over
the non-total targets with a finite outcome; the total row is excluded
because it would count every error twice."
```

---

### Task 4: exports, docs, changelog

**Files:**
- Modify: `src/ibnr/kernels/__init__.py` (import and `__all__`)
- Modify: `src/ibnr/gallery/__init__.py` (import `reserve_rows` from `ibnr.kernels.point_scores`, add to `__all__`, extend the module docstring's rule paragraph)
- Modify: `tests/test_gallery.py` (`EXPECTED_EXPORTS` gains `"reserve_rows"`; both module lists in `test_every_export_is_the_kernels_object_itself` and `test_the_gallerys_own_exports_are_not_kernels_re_exports` gain `point_scores`)
- Modify: `great-docs.yml` (under `Evaluation kernels`, add `kernels.point_metrics`, `kernels.level_errors`, `kernels.shrink_toward`, `kernels.point_summary`; under `Held-out evaluation`, add `gallery.reserve_rows` after `gallery.leaderboard`)
- Modify: `CHANGELOG.md` (`## Unreleased`)
- Modify: `CLAUDE.md` decision 8 (one added sentence)

- [ ] **Step 1: Make the export test fail first**

Add `"reserve_rows"` to `EXPECTED_EXPORTS` in `tests/test_gallery.py` and `point_scores` to both module tuples, then run:

Run: `uv run pytest tests/test_gallery.py -q -k "exports"`
Expected: FAIL (`gallery.__all__` lacks `reserve_rows`).

- [ ] **Step 2: Wire the exports**

`src/ibnr/kernels/__init__.py`: add

```python
from ibnr.kernels.point_scores import (
    level_errors,
    point_metrics,
    point_summary,
    reserve_rows,
    shrink_toward,
)
```

and the five names to `__all__` in alphabetical position.

`src/ibnr/gallery/__init__.py`: add `from ibnr.kernels.point_scores import reserve_rows`, add `"reserve_rows"` to `__all__`, and append to the docstring after the `GalleryDiagonal` paragraph:

```
``reserve_rows`` (0.7.0) joins by the first clause: it is the call a caller makes
to get from fitted entries to a published POINT board - the company reserve table
scored by ``kernels.point_metrics``. ``point_metrics`` and ``level_errors`` stay on
``ibnr.kernels``: they consume that table and never touch an entry.
```

- [ ] **Step 3: Docs and changelog**

`CHANGELOG.md`, under `## Unreleased`, add a paragraph:

```
New: point-error metrics. `kernels/point_scores.py` implements `point_metrics`
(Pool_APE, Pool_PE, MAE, RMSE, wRMSE, MAPE, MedAPE, prop_over), `level_errors`
(sum within a named level before the absolute value), `shrink_toward` (a point
pulled toward a baseline) and `reserve_rows` (predicted-versus-actual reserve
per cohort for any fitted entry, from the draw mean or a native point).
`GalleryEntry.evaluate()` gains a `point` key. `mack` gains `point()`.
`reserve_rows` is exported from `ibnr.gallery`.
```

`CLAUDE.md`, in decision 8, after the `GalleryDiagonal` sentence, add: "**`reserve_rows` (0.7.0) joined by the first clause**: it is the call from fitted entries to a published point board (the company reserve table `kernels.point_metrics` scores); `point_metrics`/`level_errors`/`shrink_toward` stay on `ibnr.kernels` because they consume that table and never touch an entry."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gallery.py tests/test_import_purity.py tests/test_point_scores.py -q`
Expected: PASS, including `test_kernels_never_imports_the_gallery` and the torch-free import subprocess test.

- [ ] **Step 5: Commit**

```bash
git add src/ibnr/kernels/__init__.py src/ibnr/gallery/__init__.py tests/test_gallery.py great-docs.yml CHANGELOG.md CLAUDE.md
git commit -m "feat: export the point scores; reserve_rows joins the gallery surface

reserve_rows is the call from fitted entries to a published point board,
which is decision 8's first clause. The metrics stay on ibnr.kernels."
```

---

### Task 5: lint, full fast suite, rebase, pull request

- [ ] **Step 1: Lint and format**

Run: `uv run ruff check . && uv run ruff format . && uv run python scripts/lint_md_snippets.py`
Expected: clean. If `ruff format` rewrites a file, re-run the tests and amend nothing; make a new commit `style: ruff format`.

- [ ] **Step 2: Full fast suite**

Run: `uv run pytest -q -rs`
Expected: all pass; skips only for torch, cmdstan, the mart and the `[interop]` extra. Record the pass/skip counts for the PR body.

- [ ] **Step 3: Rebase if main moved**

Run: `git fetch origin && git log --oneline HEAD..origin/main`
If non-empty: `git rebase origin/main`, resolve reading each conflict for intent, rerun `uv run pytest -q`.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/point-scores
gh pr create --repo EKtheSage/ibnr --base main --title "feat: point-error metrics, reserve_rows and evaluate()'s point block" --body-file <(cat <<'EOF'
Point-error metrics implemented once in `kernels/point_scores.py`: `point_metrics` (Pool_APE, Pool_PE, MAE, RMSE, wRMSE, MAPE, MedAPE, prop_over), `level_errors` (sum within a named level before the absolute value - the cancellation a company-level Pool_APE carries by design, with company 671 from the reconciliation as the worked example), `shrink_toward` and `reserve_rows`. `GalleryEntry.evaluate()` gains a `point` key; `mack` gains `point()`; `reserve_rows` is exported from `ibnr.gallery` under decision 8's first clause.

Spec: `docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md`, section 5.1 (on branch `design/tlrn-mcl-point-scores`).

Verification: `uv run pytest -q` (<counts>), ruff check/format and `scripts/lint_md_snippets.py` clean. Mutation checks tried: <list>.
EOF
)
```

Fill in the counts and the mutation list. Do not merge. Report back: the PR URL, changed files, the verification output, and anything unresolved.
