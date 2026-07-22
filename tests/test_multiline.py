"""kernels.multiline: the shared multi-LOB data contract.

The multi-line entries (sur, copula_glm, nn_transformer_ml) all model dependence
ACROSS lines of business for one company, so they need a rectangular
(lob, origin, dev) grid with a common origin axis and an identical observed-cell
pattern per line — that alignment is the whole precondition of a cross-line
covariance being meaningful. This file pins the grid construction, the target
layout, and the guards that refuse data the models cannot honestly consume.

Target layout is a fixed convention shared by every multi-line entry:
per-(lob, origin) ultimates first in lob-major order, then one total per lob, then
the diversified grand total last. Totals are summed draw by draw, so the grand
total's spread carries the modelled cross-line dependence rather than assuming
independence — that number is the diversification benefit these models exist to
quantify.

Runs on both ibis backends via ``backend_name``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ibnr import Triangle
from ibnr.kernels.multiline import (
    assemble_predictive,
    flatten_with_totals,
    multiline_data,
    multiline_targets,
    realized_multiline,
)

from .conftest import make_multiline_triangle, upper_mask

# Standard 3x3 upper triangle, NaN = unobserved. lob_b is exactly 2x lob_a in both
# loss and premium, so the two lines share loss ratios — any per-line asymmetry in
# the emitted arrays is then a contract bug rather than a data artefact.
CUM_A = np.array(
    [
        [100.0, 150.0, 175.0],
        [110.0, 165.0, np.nan],
        [120.0, np.nan, np.nan],
    ]
)
CUM_B = CUM_A * 2.0
PREMIUM = {"lob_a": np.array([200.0, 210.0, 220.0]), "lob_b": np.array([400.0, 420.0, 440.0])}


@pytest.fixture
def two_lob(backend_name) -> Triangle:
    return make_multiline_triangle(
        backend_name,
        {"lob_a": CUM_A, "lob_b": CUM_B},
        premium_by_lob=PREMIUM,
    )


def test_multiline_data_shapes_and_values(two_lob):
    """Full spec of the emitted grid: ``cum``/``obs_mask`` are (n_lob, n_w, n_d) with
    lines in stable sorted order, ``premium`` is (n_lob, n_w), and the origin axis is
    dates rather than positions so outcomes can be joined back unambiguously."""
    data = multiline_data(two_lob, loss_field="paid_loss", premium_field="earned_premium")
    assert data["n_lob"] == 2 and data["n_w"] == 3 and data["n_d"] == 3
    assert data["lobs"] == ["lob_a", "lob_b"]
    assert data["origin_periods"] == [dt.date(2010 + w, 1, 1) for w in range(3)]
    assert data["dev_grain_months"] == 12
    np.testing.assert_allclose(data["cum"][0], CUM_A)
    np.testing.assert_allclose(data["cum"][1], CUM_B)
    np.testing.assert_array_equal(data["obs_mask"][0], upper_mask(3, 3))
    np.testing.assert_allclose(data["premium"][0], PREMIUM["lob_a"])
    np.testing.assert_allclose(data["premium"][1], PREMIUM["lob_b"])


def test_multiline_data_rejects_multiple_companies(backend_name):
    """Two companies must raise: the cross-line correlation these models estimate is
    a within-company quantity, and pooling companies would silently confound it with
    between-company variation."""
    a = make_multiline_triangle(backend_name, {"lob_a": CUM_A, "lob_b": CUM_B}, company="0001")
    b = make_multiline_triangle(backend_name, {"lob_a": CUM_A, "lob_b": CUM_B}, company="0002")
    df = pd.concat([a.execute(), b.execute()], ignore_index=True)
    both = Triangle.from_long(df, measure="cumulative", backend=backend_name)
    with pytest.raises(ValueError, match="one company at a time"):
        multiline_data(both, loss_field="paid_loss")


def test_multiline_data_rejects_single_lob(backend_name):
    """One line must raise rather than degenerate: there is no dependence to model,
    and a 1x1 correlation matrix would make the entry look like it had succeeded."""
    t = make_multiline_triangle(backend_name, {"lob_a": CUM_A})
    with pytest.raises(ValueError, match=">= 2 lines"):
        multiline_data(t, loss_field="paid_loss")


def test_multiline_data_rejects_misaligned_masks(backend_name):
    """Lines with different observed-cell patterns must raise. Residuals from
    different sets of cells cannot be paired, so a covariance estimated across
    ragged lines would be computed on mismatched calendar periods."""
    ragged = CUM_B.copy()
    ragged[0, 2] = np.nan  # lob_b missing a cell lob_a has
    t = make_multiline_triangle(backend_name, {"lob_a": CUM_A, "lob_b": ragged})
    with pytest.raises(ValueError, match="different observed-cell pattern"):
        multiline_data(t, loss_field="paid_loss")


def test_multiline_data_rejects_incremental(backend_name):
    """The contract is defined on cumulative losses; an incremental triangle raises
    instead of being reinterpreted as cumulative."""
    t = make_multiline_triangle(backend_name, {"lob_a": CUM_A, "lob_b": CUM_B})
    incr = t.to_incremental()
    with pytest.raises(ValueError, match="cumulative"):
        multiline_data(incr, loss_field="paid_loss")


def test_multiline_data_rejects_restated_cells(backend_name):
    """A triangle still carrying restatement history (same cell at two eval_dates)
    must raise, not pick arbitrarily. The caller has to decide the vintage via
    ``as_of()`` / ``latest_diagonal()`` — silently choosing would leak the future."""
    t = make_multiline_triangle(backend_name, {"lob_a": CUM_A, "lob_b": CUM_B})
    df = t.execute()
    df = df.copy()
    for col in ("origin_period", "eval_date"):
        df[col] = pd.to_datetime(df[col]).dt.date
    restated = df.iloc[[0]].assign(eval_date=dt.date(2023, 12, 31), value=999.0)
    hist = Triangle.from_long(
        pd.concat([df, restated], ignore_index=True), measure="cumulative", backend=backend_name
    )
    with pytest.raises(ValueError, match="multiple rows per"):
        multiline_data(hist, loss_field="paid_loss")


def test_realized_multiline_alignment(two_lob):
    """Outcomes come back on the same (lob, origin) grid the model was fit on, with
    NaN where the outcome has not emerged — positional alignment is what stops a
    retrospective scoring lob_a's draws against lob_b's realized ultimate."""
    data = multiline_data(two_lob, loss_field="paid_loss")
    realized = realized_multiline(
        two_lob,
        loss_field="paid_loss",
        dev_lag=36,
        lobs=data["lobs"],
        origins=data["origin_periods"],
    )
    np.testing.assert_allclose(realized[0], [175.0, np.nan, np.nan])
    np.testing.assert_allclose(realized[1], [350.0, np.nan, np.nan])


def test_targets_and_flatten_layout():
    """Pins the target layout end to end: label order, the flattening of a
    (draws, lob, origin) cube into leaderboard columns, and that both totals levels
    are appended as draw-wise sums (so dependence survives into the grand total)."""
    lobs = ["lob_a", "lob_b"]
    origins = [dt.date(2010 + w, 1, 1) for w in range(3)]
    targets = multiline_targets(
        lobs, origins, premium=np.vstack([PREMIUM["lob_a"], PREMIUM["lob_b"]])
    )
    # 2 lobs x 3 origins, + one total per lob, + the diversified grand total
    assert len(targets) == 2 * 3 + 2 + 1
    assert targets["label"].tolist() == [
        "lob_a/2010",
        "lob_a/2011",
        "lob_a/2012",
        "lob_b/2010",
        "lob_b/2011",
        "lob_b/2012",
        "lob_a/total",
        "lob_b/total",
        "total",
    ]

    # sequential values (0..11) so every flattened position is individually
    # identifiable — a transposed lob/origin axis would be visible immediately
    ults = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)  # (draws, lob, origin)
    flat = flatten_with_totals(ults)
    assert flat.shape == (2, 9)  # (draws, targets), matching len(targets) above
    np.testing.assert_allclose(flat[0, :6], np.arange(6.0))
    np.testing.assert_allclose(flat[0, 6:8], [0 + 1 + 2, 3 + 4 + 5])
    assert flat[0, 8] == flat[0, 6] + flat[0, 7]

    pred = assemble_predictive(ults, targets)
    assert pred.n_targets == 9
    # grand total is the sum of per-lob totals draw by draw
    np.testing.assert_allclose(pred.samples[:, 8], pred.samples[:, 6] + pred.samples[:, 7])


def test_assemble_predictive_validates_layout():
    """Assembly rejects a cube of the wrong rank or the wrong origin count instead of
    broadcasting it into a plausible-looking PredictiveDistribution."""
    lobs = ["a", "b"]
    origins = [dt.date(2010, 1, 1)]
    targets = multiline_targets(lobs, origins)
    with pytest.raises(ValueError, match="n_draws, n_lob, n_w"):
        assemble_predictive(np.zeros((4, 2)), targets)
    with pytest.raises(ValueError, match="target rows"):
        assemble_predictive(np.zeros((4, 2, 5)), targets)
