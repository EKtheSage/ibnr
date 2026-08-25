"""nn_transformer_ml held-out wiring: the per-(company, line) adapter, draws
(CRPS axis) and the mixture density (ELPD axis), under BOTH dependence heads.
Skips without torch.

What this file protects, in four layers:

1. **The adapter.** The fit's cohort is a COMPANY while a held-out cohort built
   by ``next_diagonal`` is one (company, line) pair, so
   ``gallery/nn/_heldout_ml.py`` slices a pair out of the company contract and
   puts ``line_of_business`` back into the segment key. Every guard in
   ``kernels.holdout.index_into`` - cohort identity, segment schema, training
   overlap - then applies here exactly as it does to a Stan fit, and the tests
   drive those guards through real refusals rather than the happy path.
2. **Draws.** ``predict_at`` returns incremental draws anchored to each cell's
   training-diagonal predecessor BY THE BASE CLASS (``heldout_draw_scale =
   "incremental"``); the anchor is roughly two orders of magnitude above the
   increment, so a dropped conversion is unmistakable. Two lines of one company
   under one study seed must also draw different random numbers.
3. **Density.** ``log_lik_at`` is checked against an independent recomputation -
   the test runs its own forward pass and writes both Jacobians out by hand -
   for the ``"ar"`` head and for the ``"joint"`` head, whose scored line has to
   be marginalized out of a multivariate mixture. A negative control shows that
   recomputation can fail: reading the wrong line of the joint head's mean or
   Cholesky factor moves the answer.
4. **The pinned-dev asymmetry**, which here is per LINE: the density is refused
   where the draws still work.

The fixture is three companies on 6x6 full squares with ``as_of`` at diagonal 6,
so each (company, line) pair's next diagonal holds 5 cells at dev steps 6, 5, 4,
3, 2 - of which dev 6 is pinned (no training-context value reaches that dev's
normalizer, since its only observation sits on the validation diagonal). The
third company writes one line only, which is what gives the "line this company
does not write" refusal something real to refuse.
"""

from __future__ import annotations

import datetime as dtm
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ibnr import Triangle, gallery  # noqa: E402
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout  # noqa: E402
from ibnr.gallery.nn._heldout_ml import (  # noqa: E402
    MLCohortHeldout,
    company_line_contract,
    company_line_cutoff,
)
from ibnr.gallery.nn.transformer_ml.config import TransformerMLConfig  # noqa: E402
from ibnr.gallery.nn.transformer_ml.model import NNTransformerML  # noqa: E402
from ibnr.kernels.densities import MEASURES  # noqa: E402
from ibnr.kernels.holdout import HoldoutCells, index_into, next_diagonal  # noqa: E402
from ibnr.kernels.rng import heldout_stream  # noqa: E402

from .conftest import BACKENDS, make_multiline_triangle  # noqa: E402


def _tiny(dependence: str) -> TransformerMLConfig:
    return TransformerMLConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0.0,
        n_components=2,
        dependence=dependence,
        batch_size=8,
        max_epochs=2,
        patience=5,
        ensemble_size=2,
        # the two draw budgets differ on purpose: they are independent knobs
        # (the rollout's and the held-out diagonal's), and equal values would
        # let either one stand in for the other everywhere below
        n_draws=50,
        heldout_n_draws=40,
    )


START = 2000
AS_OF = "2005-12-31"  # diagonal 6 of a 6x6 square starting in 2000
LOBS = ("lob_0", "lob_1")
#: company code -> the lines it writes. "0003" writes one line, which is what
#: the "does not write that line" refusal needs.
WRITES = {"0001": ("lob_0", "lob_1"), "0002": ("lob_0", "lob_1"), "0003": ("lob_0",)}
NAMES = {"0001": "Alpha Mutual", "0002": "Beta Casualty", "0003": "Gamma Indemnity"}
PREMIUM = 1000.0
#: dev steps whose per-(line, dev) normalizer is pinned at as_of = diagonal 6:
#: no training-context value reaches dev 6, while dev 5 keeps one per company.
PINNED_DEVS = (6,)
#: every (company, line) pair the fixture fits, as a held-out cohort segment
PAIRS = [
    {"company_code": code, "line_of_business": lob} for code, lobs in WRITES.items() for lob in lobs
]
SEG_A0 = PAIRS[0]  # company 0001, lob_0
SEG_A1 = PAIRS[1]  # company 0001, lob_1


def _matrix(seed: int) -> np.ndarray:
    """One (6, 6) cumulative square with a decaying incremental pattern."""
    rng = np.random.default_rng(seed)
    dev_level = np.exp(np.linspace(-0.8, -3.0, 6))
    incr = 1000.0 * dev_level[None, :] * rng.lognormal(0.0, 0.1, size=(6, 6))
    return np.cumsum(incr, axis=1)


def _matrices() -> dict[tuple[str, str], np.ndarray]:
    """One square per (company, line) pair, each with its own draw."""
    out = {}
    for i, (code, lobs) in enumerate(WRITES.items()):
        for j, lob in enumerate(lobs):
            out[(code, lob)] = _matrix(10 * i + j)
    return out


def _company_triangle(backend: str, code: str, cum: dict) -> Triangle:
    """One company's multi-line triangle, with the display-only name column."""
    return make_multiline_triangle(
        backend,
        {lob: cum[(code, lob)] for lob in WRITES[code]},
        premium_by_lob={lob: np.full(6, PREMIUM) for lob in WRITES[code]},
        start_year=START,
        company=code,
        company_name=NAMES[code],
    )


def _pooled_triangle(backend: str, cum: dict) -> Triangle:
    """Every company in one triangle - the pool the entry is fitted on.

    Built by materializing each company's triangle and concatenating the long
    frames, so the fixture and the per-pair cells below come from exactly the
    same conventions helper and cannot drift apart.
    """
    frames = []
    for code in WRITES:
        df = _company_triangle(backend, code, cum).execute()
        for column in ("origin_period", "eval_date"):
            df[column] = pd.to_datetime(df[column]).dt.date
        frames.append(df)
    return Triangle.from_long(
        pd.concat(frames, ignore_index=True), measure="cumulative", backend=backend
    )


def _cells_for(backend: str, code: str, lob: str, cum: dict) -> HoldoutCells:
    """One (company, line) pair's next-diagonal cells."""
    one = make_multiline_triangle(
        backend,
        {lob: cum[(code, lob)]},
        premium_by_lob={lob: np.full(6, PREMIUM)},
        start_year=START,
        company=code,
        company_name=NAMES[code],
    )
    return next_diagonal(one, as_of=AS_OF, fields="paid_loss", premium_field="earned_premium")


def _fit_params() -> list:
    """One parameter per (backend, dependence head), carrying the backend's own
    marks - which is how the polars leg keeps its skip when polars is absent."""
    out = []
    for backend in BACKENDS:
        name = backend if isinstance(backend, str) else backend.values[0]
        marks = () if isinstance(backend, str) else backend.marks
        for dependence in ("ar", "joint"):
            out.append(pytest.param((name, dependence), marks=marks, id=f"{name}-{dependence}"))
    return out


@pytest.fixture(scope="module", params=_fit_params())
def fitted(request):
    """One tiny pooled fit per (backend, dependence head), plus each pair's cells.

    Module-scoped: the fit is the expensive part, and every test here leaves it
    as it found it (the two that mutate ``models_`` / ``config_`` restore them
    in a ``finally``; the one that calls ``predict`` fills the rollout cache,
    which is derived state nothing else here reads).
    """
    backend, dependence = request.param
    cum = _matrices()
    pooled = _pooled_triangle(backend, cum)
    entry = NNTransformerML().fit(
        pooled, loss_field="paid_loss", as_of=AS_OF, config=_tiny(dependence), seed=0
    )
    cells = {
        (code, lob): _cells_for(backend, code, lob, cum)
        for code, lobs in WRITES.items()
        for lob in lobs
    }
    return SimpleNamespace(entry=entry, cells=cells, cum=cum, dependence=dependence)


def _cells(fitted, segment: dict) -> HoldoutCells:
    return fitted.cells[(segment["company_code"], segment["line_of_business"])]


def _indexed(view, cells: HoldoutCells):
    """``cells`` in the fit's index space, through the same narrowing
    ``log_lik_at``/``predict_at`` apply.

    The fixture's cells carry ``company_name``, which the fit's key does not, so
    a bare ``index_into`` refuses them on schema equality - the narrowing (and
    the check of the dropped value) is what the base class does first.
    """
    return index_into(view._keyed_to_fit(cells), view.contract_, field="paid_loss")


def _split(cells: HoldoutCells, *, pinned: bool) -> HoldoutCells:
    """The next-diagonal cells at pinned (or unpinned) dev steps only."""
    dev_lags = [12 * d for d in PINNED_DEVS]
    mask = cells.frame["dev_lag"].isin(dev_lags)
    frame = cells.frame[mask if pinned else ~mask].reset_index(drop=True)
    return replace(cells, frame=frame)


def _hand_built_contract() -> dict:
    """A deliberately ASYMMETRIC two-company, two-line company contract.

    The fitted fixture's staircase is symmetric in (w, d) and complete in every
    line, so a w/d swap or a line/company swap in the adapter would slip through
    it. Nothing here is a square: company 0 writes both lines with different
    depths, company 1 writes only line 0.
    """
    obs = np.zeros((2, 2, 2, 3), dtype=bool)  # (n_c, L, n_w, n_d)
    obs[0, 0, 0, :] = True  # company 0, line 0, origin 1: devs 1..3 usable
    obs[0, 0, 1, 0] = True  # company 0, line 0, origin 2: dev 1 usable...
    obs[0, 1, 0, :2] = True  # company 0, line 1, origin 1: devs 1..2
    obs[1, 0, 0, 0] = True  # company 1, line 0, origin 1: dev 1
    latest = np.zeros((2, 2, 2), dtype=int)
    latest[0, 0] = [3, 3]  # ...but ANCHORED at dev 3 (hole at dev 2)
    latest[0, 1] = [2, 0]
    latest[1, 0] = [1, 0]
    premium = np.full((2, 2, 2), np.nan)
    premium[0, 0] = [100.0, 200.0]
    premium[0, 1] = [300.0, 400.0]
    premium[1, 0] = [500.0, 600.0]
    line_mask = np.array([[True, True], [True, False]])
    w_grid, d_grid = np.meshgrid(np.arange(2), np.arange(3), indexing="ij")
    return {
        "companies": pd.DataFrame({"company_code": ["0001", "0002"]}),
        "lob_levels": ["lob_0", "lob_1"],
        "line_mask": line_mask,
        "obs_mask": obs,
        "latest_dev": latest,
        "premium": premium,
        "cal_idx": w_grid + d_grid + 1,
        "fields": ["paid_loss"],
        "origin_periods": [dtm.date(2000, 1, 1), dtm.date(2001, 1, 1)],
        "dev_grain_months": 12,
        "n_w": 2,
        "n_d": 3,
        "segment_columns": ("company_code",),
    }


# -- fixture sanity ------------------------------------------------------------


def test_fixture_shape_is_what_the_file_claims(fitted):
    """Pin the structure every other test relies on: 5 scorable cells per pair
    at dev steps {2..6}, nothing excluded, exactly dev 6 pinned on every line,
    and a third company that writes one line. If the triangle or the as_of
    drifts, this fails first and names the rot."""
    entry = fitted.entry
    c = entry.contract_
    assert list(c["companies"]["company_code"]) == ["0001", "0002", "0003"]
    assert list(c["lob_levels"]) == list(LOBS)
    np.testing.assert_array_equal(c["line_mask"], [[True, True], [True, True], [True, False]])
    for segment in PAIRS:
        cells = _cells(fitted, segment)
        assert cells.n_cells == 5, segment
        assert cells.exclusion_counts() == {
            "new_origin": 0,
            "dev_beyond_trained": 0,
            "no_predecessor": 0,
        }
        assert sorted(cells.frame["dev_lag"] // 12) == [2, 3, 4, 5, 6]
    for li in range(len(LOBS)):
        pinned = entry.norm_["pinned"][li, 0]  # target channel, (n_d,)
        np.testing.assert_array_equal(pinned, [False, False, False, False, False, True])
    # both sides of the pinned/unpinned split are non-empty, or the asymmetry
    # tests below would silently test nothing
    assert _split(_cells(fitted, SEG_A0), pinned=True).n_cells == 1
    assert _split(_cells(fitted, SEG_A0), pinned=False).n_cells == 4


def test_entry_declares_both_capabilities(fitted):
    """The scale declarations the base classes act on, and the capability claims
    themselves - read off the REGISTRY, so an entry that stopped declaring them
    fails here rather than at the first call."""
    cls = gallery.get("nn_transformer_ml")
    assert issubclass(cls, ScoresHeldout) and issubclass(cls, PredictsHeldout)
    entry = fitted.entry
    assert entry.heldout_measure == "loss_ratio" and entry.heldout_measure in MEASURES
    assert entry.heldout_draw_scale == "incremental"
    view = entry.at_cohort(SEG_A0)
    assert isinstance(view, MLCohortHeldout)
    assert isinstance(view, ScoresHeldout) and isinstance(view, PredictsHeldout)
    assert view.heldout_measure == "loss_ratio"
    assert view.heldout_draw_scale == "incremental"
    assert view.cohort == (0, 0)


def test_the_views_identity_names_the_company_and_the_line(fitted):
    """``_cell_identity`` is the cohort's FULL identity, and it must carry every
    column: the company key, the display-only column the key drops, and the line.

    The base class checks each column it drops from the supplied cells against
    this dict, which today is only ``company_name`` - the line is never a
    dropped column, because it is part of the fit's own key. So the line entry
    is asserted here directly rather than through the narrowing path: the method
    answers "which cohort is this scorer for", and a company without a line is
    not an answer to that question.
    """
    view = fitted.entry.at_cohort(SEG_A1)
    assert view._cell_identity() == {
        "company_code": "0001",
        "company_name": NAMES["0001"],
        "line_of_business": "lob_1",
    }


# -- the adapter ---------------------------------------------------------------


def test_company_line_contract_on_a_hand_built_contract():
    """The adapter mapping itself, on the asymmetric hand-built contract.

    Pins four things at once: the line rejoins the segment key; the (company,
    line) slice is taken on the right two axes; premium is that pair's row and
    1-D; and the anchor rule holds - a per-origin anchor whose own increment was
    unusable (predecessor hole) is absent from ``obs_mask`` but WAS training
    data, and must be declared trained or the overlap guard would let it be
    scored as held out.
    """
    contract = _hand_built_contract()
    got = company_line_contract(contract, 0, 0, models=("paid_loss",))
    declared = set(zip(got["w"].tolist(), got["d"].tolist(), strict=True))
    assert declared == {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)}
    # guard on the guard: the expected set is not w/d symmetric
    assert declared != {(d, w) for w, d in declared}
    assert got["segment"] == {"company_code": "0001", "line_of_business": "lob_0"}
    assert got["models"] == ["paid_loss"] and got["measure"] == "cumulative"
    np.testing.assert_allclose(got["premium"], [100.0, 200.0])
    assert got["premium"].shape == (2,)

    # the OTHER line of the same company is a different cohort with different
    # cells and a different exposure - a line/company axis swap shows up here
    other = company_line_contract(contract, 0, 1, models=("paid_loss",))
    assert other["segment"]["line_of_business"] == "lob_1"
    assert set(zip(other["w"].tolist(), other["d"].tolist(), strict=True)) == {
        (1, 1),
        (1, 2),
    }
    np.testing.assert_allclose(other["premium"], [300.0, 400.0])

    with pytest.raises(IndexError, match="company 5 out of range"):
        company_line_contract(contract, 5, 0, models=("paid_loss",))
    with pytest.raises(IndexError, match="line 9 out of range"):
        company_line_contract(contract, 0, 9, models=("paid_loss",))
    with pytest.raises(ValueError, match="does not write"):
        company_line_contract(contract, 1, 1, models=("paid_loss",))


def test_adapter_declares_the_predecessor_of_every_obs_cell():
    """Cumulative values at devs {1, 2, 4, 5} on one line: the dev-3 hole makes
    the dev-4 increment unusable, so obs = {1, 2, 5} and the anchor is dev 5 -
    dev 4 is neither, yet its VALUE fed the dev-5 increment during training.

    The declared set must be the honest closure rather than relying on
    ``next_diagonal`` excluding such cells as ``no_predecessor`` downstream:
    ``index_into`` must refuse the dev-4 cell as training data. Dev 3, the
    genuine hole, stays undeclared - the closure is minimal, not a blanket fill.

    Mutation this must catch: drop the predecessor-closure union in
    ``company_line_contract`` - ``index_into`` then accepts the dev-4 cell and
    the ``raises`` below fails.
    """
    obs = np.zeros((1, 1, 1, 5), dtype=bool)
    obs[0, 0, 0, [0, 1, 4]] = True  # usable increments at devs 1, 2, 5
    w_grid, d_grid = np.meshgrid(np.arange(1), np.arange(5), indexing="ij")
    contract = {
        "companies": pd.DataFrame({"company_code": ["0001"]}),
        "lob_levels": ["lob_0"],
        "line_mask": np.array([[True]]),
        "obs_mask": obs,
        "latest_dev": np.array([[[5]]]),
        "premium": np.array([[[100.0]]]),
        "cal_idx": w_grid + d_grid + 1,
        "fields": ["paid_loss"],
        "origin_periods": [dtm.date(2000, 1, 1)],
        "dev_grain_months": 12,
        "n_w": 1,
        "n_d": 5,
        "segment_columns": ("company_code",),
    }
    adapter = company_line_contract(contract, 0, 0, models=("paid_loss",))
    declared = set(zip(adapter["w"].tolist(), adapter["d"].tolist(), strict=True))
    # obs {1, 2, 5} + anchor {5} + predecessors {1, 4}; dev 3 undeclared
    assert declared == {(1, 1), (1, 2), (1, 4), (1, 5)}

    frame = pd.DataFrame(
        [
            {
                "company_code": "0001",
                "line_of_business": "lob_0",
                "field": "paid_loss",
                "origin_period": dtm.date(2000, 1, 1),
                "dev_lag": 48,  # dev 4: the probed cell
                "eval_date": dtm.date(2003, 12, 31),
                "value": 123.0,
                "prev_value": np.nan,
            }
        ]
    )
    cells = HoldoutCells(
        frame=frame,
        as_of=dtm.date(2004, 12, 31),
        eval_date=dtm.date(2003, 12, 31),
        excluded=frame.iloc[0:0],
        train_origins=(dtm.date(2000, 1, 1),),
        segments=("company_code", "line_of_business"),
        measure="cumulative",
    )
    with pytest.raises(ValueError, match="TRAINING data"):
        index_into(cells, adapter, field="paid_loss")


def test_company_line_cutoff_is_the_scored_lines_own_diagonal():
    """The cutoff convention: each line's as_of diagonal, not the company's.

    On the hand-built contract company 0's line 0 reaches diagonal 4 while its
    line 1 reaches diagonal 2, so a company-wide reading would put line 1's
    held-out cell two steps past the cutoff instead of one - a different row of
    the relative calendar embedding and therefore a different predictive
    distribution.

    The union with the per-origin anchors is what makes line 0's answer 4: its
    origin-2 cell at dev 3 is an anchor whose own increment is unusable (the
    dev-2 hole), so it is absent from ``obs_mask``, and reading ``obs_mask``
    alone answers 3 for a line that plainly held a cell on diagonal 4.
    """
    contract = _hand_built_contract()
    assert company_line_cutoff(contract, 0, 0) == 4
    obs_only = int(contract["cal_idx"][contract["obs_mask"][0, 0]].max())
    assert obs_only == 3  # what dropping the anchor union would answer
    assert company_line_cutoff(contract, 0, 1) == 2
    assert company_line_cutoff(contract, 1, 0) == 1
    with pytest.raises(ValueError, match="does not write"):
        company_line_cutoff(contract, 1, 1)


def test_cutoffs_coincide_on_complete_squares(fitted):
    """Every line of every company in the fixture reaches the same diagonal, so
    the per-line cutoff equals the company's - the case the docstring claims
    coincides."""
    c = fitted.entry.contract_
    for ci in range(len(c["companies"])):
        cutoffs = {company_line_cutoff(c, ci, li) for li in np.nonzero(c["line_mask"][ci])[0]}
        assert cutoffs == {6}


def test_adapter_indexes_cells_correctly(fitted):
    """``index_into`` accepts the per-pair adapter and produces the right
    (w, d)/value/premium mapping. The declared training cells are exactly the
    as_of staircase, so the overlap guard has the full training set to check
    against - a short list would wave training cells through."""
    entry = fitted.entry
    view = entry.at_cohort(SEG_A0)
    adapter = view.contract_
    assert adapter["segment"] == SEG_A0
    assert adapter["models"] == ["paid_loss"]
    assert adapter["measure"] == "cumulative"
    # training cells = the 21-cell staircase w + d <= 7 (diagonals 1..6)
    declared = set(zip(adapter["w"].tolist(), adapter["d"].tolist(), strict=True))
    assert declared == {(w, d) for w in range(1, 7) for d in range(1, 7) if w + d <= 7}
    assert adapter["premium"].shape == (6,)
    np.testing.assert_allclose(adapter["premium"], PREMIUM)

    cells = _cells(fitted, SEG_A0)
    idx = _indexed(view, cells)
    assert idx.n_cells == 5
    # the next diagonal: w + d == 8, in frame (origin, dev) sort order
    np.testing.assert_array_equal(idx.w + idx.d, np.full(5, 8))
    np.testing.assert_allclose(idx.value, cells.values)
    np.testing.assert_allclose(idx.premium, PREMIUM)
    # the standalone helper builds the same dict the view carries
    direct = company_line_contract(entry.contract_, 0, 0, models=("paid_loss",))
    np.testing.assert_array_equal(direct["w"], adapter["w"])
    np.testing.assert_array_equal(direct["d"], adapter["d"])
    assert direct["segment"] == adapter["segment"]


def test_at_cohort_needs_the_line_named(fitted):
    """The fit's cohort is a company; a held-out cohort is a (company, line)
    pair. A segment naming only the company cannot be resolved and must say so
    by name rather than pick a line."""
    with pytest.raises(ValueError, match="line_of_business"):
        fitted.entry.at_cohort({"company_code": "0001"})


def test_at_cohort_refuses_an_unknown_line(fitted):
    """A line that is not on the fit's line axis was never fitted for anybody."""
    with pytest.raises(ValueError, match="unknown line_of_business"):
        fitted.entry.at_cohort({"company_code": "0001", "line_of_business": "lob_9"})


def test_at_cohort_refuses_a_line_the_company_does_not_write(fitted):
    """Company 0003 writes one line. The other line's arrays are padding zeros,
    so scoring it would return numbers for a triangle the fit never saw - and
    every one of them would be finite and plausible."""
    with pytest.raises(ValueError, match="does not write"):
        fitted.entry.at_cohort({"company_code": "0003", "line_of_business": "lob_1"})


def test_at_cohort_refuses_an_unknown_company(fitted):
    """The company half of the pair goes through the shared ``cohort_index``,
    so a typo'd company is refused with the fit's own cohort key named."""
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        fitted.entry.at_cohort({"company_code": "9999", "line_of_business": "lob_0"})


def test_adapter_refuses_another_line_of_the_same_company(fitted):
    """(w, d) alone cannot identify a cell: lob_1's cells index cleanly into
    lob_0's adapter and would score the wrong line of the right company -
    exactly the mistake a company-level cohort invites. Driven through the
    view's public methods, so the guard is on the path callers actually use."""
    view = fitted.entry.at_cohort(SEG_A0)
    with pytest.raises(ValueError, match="was trained on"):
        view.log_lik_at(_cells(fitted, SEG_A1), field="paid_loss")
    with pytest.raises(ValueError, match="was trained on"):
        view.predict_at(_cells(fitted, SEG_A1), field="paid_loss", seed=0)


def test_adapter_refuses_another_company_on_the_same_line(fitted):
    """The other half of the same guard: company 0002's lob_0 cells against
    company 0001's lob_0 view.

    Two guards stand in the way here and both are checked, because they fire at
    different depths. The display-only ``company_name`` is checked while the
    cells are narrowed onto the fit's key, so it catches the wrong company
    first; a caller who already narrowed the cells themselves meets
    ``index_into``'s cohort-identity guard instead. Neither may let the cells
    through - they index cleanly and would score the fitted company.
    """
    view = fitted.entry.at_cohort(SEG_A0)
    other = _cells(fitted, {"company_code": "0002", "line_of_business": "lob_0"})
    with pytest.raises(ValueError, match="company_name"):
        view.log_lik_at(other, field="paid_loss")
    narrowed = other.narrowed_to(["company_code", "line_of_business"])
    with pytest.raises(ValueError, match="was trained on"):
        view.log_lik_at(narrowed, field="paid_loss")


def test_adapter_refuses_training_cells(fitted):
    """A held-out score computed on training data is systematically too good.
    Rewrite one cell key to a trained (w, d) and the overlap guard must fire."""
    cells = _cells(fitted, SEG_A0)
    frame = cells.frame.copy()
    frame.loc[0, "dev_lag"] = 12  # (origin 2000, dev 1) is training data
    doctored = replace(cells, frame=frame)
    with pytest.raises(ValueError, match="TRAINING data"):
        fitted.entry.predict_at(doctored, field="paid_loss", seed=0)


def test_adapter_refuses_a_premium_mismatch(fitted):
    """The exposure the fit standardized against is the only one either hook may
    divide or multiply by, and the base class's measure carry divides by the
    CELLS' premium - so a disagreement leaves a density that no longer
    integrates to 1 and rescales every draw, silently and plausibly."""
    cells = _cells(fitted, SEG_A0)
    frame = cells.frame.copy()
    frame["premium"] = frame["premium"] * 1.5
    doctored = replace(cells, frame=frame)
    with pytest.raises(ValueError, match="disagrees with the fitted contract"):
        fitted.entry.log_lik_at(_split(doctored, pinned=False), field="paid_loss")
    with pytest.raises(ValueError, match="disagrees with the fitted contract"):
        fitted.entry.predict_at(doctored, field="paid_loss", seed=0)


def test_adapter_refuses_a_display_column_mismatch(fitted):
    """``company_name`` is not part of the fit's key, so the cells are narrowed
    onto the key before ``index_into`` sees them - and each dropped value is
    CHECKED against the pair's full identity on the way. A cell carrying another
    spelling of the company must be refused, not quietly scored here."""
    cells = _cells(fitted, SEG_A0)
    assert "company_name" in cells.segments  # or the narrowing never runs
    frame = cells.frame.copy()
    frame["company_name"] = "Not Alpha Mutual"
    doctored = replace(cells, frame=frame)
    with pytest.raises(ValueError, match="company_name"):
        fitted.entry.log_lik_at(_split(doctored, pinned=False), field="paid_loss")


def test_entry_needs_cells_that_name_a_cohort(fitted):
    """The entry-level methods resolve the pair from the cells, so a bare
    CellIndex (which carries no segment values) is refused with a pointer at
    at_cohort - not mis-scored against an arbitrary pair."""
    view = fitted.entry.at_cohort(SEG_A0)
    idx = _indexed(view, _cells(fitted, SEG_A0))
    with pytest.raises(TypeError, match="at_cohort"):
        fitted.entry.log_lik_at(idx)
    with pytest.raises(TypeError, match="at_cohort"):
        fitted.entry.predict_at(idx)


def test_the_pooled_entry_has_no_single_identity(fitted):
    """A pooled fit spans many pairs, so its own ``_cell_identity`` must refuse
    rather than hand back a plausible key that happens to exist."""
    with pytest.raises(RuntimeError, match="at_cohort"):
        fitted.entry._cell_identity()
    with pytest.raises(NotImplementedError, match="loss ratios"):
        fitted.entry.training_cells()


# -- draws (PredictsHeldout) ---------------------------------------------------


def test_predict_at_shape_seed_and_variance(fitted):
    """Draw contract: (config.heldout_n_draws, n_cells); reproducible per seed;
    live cells have genuine spread; pinned cells are the rollout-semantics point
    mass at anchor + premium * pooled dev mean, exactly."""
    entry = fitted.entry
    cells = _cells(fitted, SEG_A0)
    a = entry.predict_at(cells, field="paid_loss", seed=11)
    assert a.shape == (entry.config_.heldout_n_draws, 5)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, entry.predict_at(cells, field="paid_loss", seed=11))
    assert not np.allclose(a, entry.predict_at(cells, field="paid_loss", seed=12))

    frame = cells.frame
    pinned_col = frame["dev_lag"].isin([12 * d for d in PINNED_DEVS]).to_numpy()
    assert (a[:, ~pinned_col].std(axis=0) > 0).all()
    # pinned columns: zero spread, value = anchor + premium * pooled dev mean,
    # read at THIS LINE's row of the normalizer
    mean0 = entry.norm_["mean"][0, 0]
    d0 = (frame["dev_lag"] // 12).to_numpy() - 1
    expected = frame["prev_value"].to_numpy() + PREMIUM * mean0[d0]
    # exact point mass: max == min per column (std would show a ~1e-13 numpy
    # mean-rounding artifact even on bit-identical values)
    assert (a[:, pinned_col].max(axis=0) == a[:, pinned_col].min(axis=0)).all()
    np.testing.assert_allclose(a[0, pinned_col], expected[pinned_col], rtol=1e-6)


def test_predict_at_anchors_increments_through_the_base_class(fitted):
    """The anchor conversion is the base class's job and it must actually run.

    The entry draws INCREMENTS while the triangle is cumulative, so draws scored
    without the declared-scale conversion are wrong by the whole anchor - the
    CRPS-996-where-the-truth-is-3.4 bug class. ``predict_at`` derives its
    generator from the seed together with the cells' cohort identity, the field
    and the cutoff (``kernels.rng``), so rebuilding that stream here makes the
    native draws reproducible and the assertion exact equality.

    Mutation this must catch: ``heldout_draw_scale = "cumulative"`` (or any
    skipped conversion) makes ``got == native``, off by every anchor.
    """
    entry = fitted.entry
    cells = _cells(fitted, SEG_A0)
    view = entry.at_cohort(SEG_A0)
    idx = _indexed(view, cells)
    stream = heldout_stream(11, cells, field="paid_loss")
    native = view._draws_native(idx, rng=np.random.default_rng(stream))
    got = entry.predict_at(cells, field="paid_loss", seed=11)
    anchor = idx.prev_value
    np.testing.assert_allclose(got, native + anchor[None, :], rtol=1e-12)
    # the shift is large relative to the draws, so the mutant is unmistakable
    assert anchor.min() > 100.0
    assert np.abs(native).mean() < anchor.min()


def test_predict_at_refuses_all_pinned_cells(fitted):
    """A request whose every cell is pinned would return a point mass in every
    column - not a predictive distribution - and must be refused loudly."""
    only_pinned = _split(_cells(fitted, SEG_A0), pinned=True)
    with pytest.raises(ValueError, match="pinned"):
        fitted.entry.predict_at(only_pinned, field="paid_loss", seed=0)


def test_two_lines_of_one_company_do_not_share_one_stream(fitted):
    """One study seed, two lines, two different streams of random numbers.

    The stream is derived in ``PredictsHeldout.predict_at`` from the seed
    together with the cells' cohort identity, which includes the line - so this
    entry inherits the separation rather than implementing it. Without it, draw
    ``i`` of both lines would read the same underlying numbers and any quantity
    summed across a company's lines within a draw - the company total, its
    spread, a diversification ratio - would carry noise that is not real.

    At 4,000 draws one sampling standard deviation of a sample correlation is
    about 0.016, so the 0.1 threshold is more than six of them from zero and the
    test does not flicker.
    """
    entry = fitted.entry
    keep = entry.config_
    entry.config_ = replace(keep, heldout_n_draws=4000)
    try:
        a = entry.predict_at(_cells(fitted, SEG_A0), field="paid_loss", seed=5)
        b = entry.predict_at(_cells(fitted, SEG_A1), field="paid_loss", seed=5)
        assert a.shape == b.shape == (4000, 5)
        # the sum across cells within a draw is what shared noise damages
        rho = float(np.corrcoef(a.sum(axis=1), b.sum(axis=1))[0, 1])
        assert abs(rho) < 0.1, rho
        # and the separation must not have cost reproducibility
        np.testing.assert_array_equal(
            a, entry.predict_at(_cells(fitted, SEG_A0), field="paid_loss", seed=5)
        )
    finally:
        entry.config_ = keep


def test_the_two_draw_budgets_are_delivered_independently(fitted):
    """One config, two knobs, two paths: ``predict_at`` spends
    ``heldout_n_draws`` and ``predict`` spends ``n_draws``.

    Both halves are asserted from the same fit, because a change that simply
    renamed the field would move BOTH counts and pass either half alone.
    """
    entry = fitted.entry
    cfg = entry.config_
    assert cfg.heldout_n_draws != cfg.n_draws  # or either could stand in
    draws = entry.predict_at(_cells(fitted, SEG_A0), field="paid_loss", seed=3)
    assert draws.shape == (cfg.heldout_n_draws, 5)
    # n_draws=None is the spelling that reads the config, i.e. the rollout knob
    pred = entry.predict(segment={"company_code": "0001"}, n_draws=None, seed=3)
    assert pred.samples.shape[0] == cfg.n_draws


def test_draws_reproduce_after_a_fresh_refit(fitted, request):
    """The same call on a from-scratch refit of the same data gives the same
    draws bit for bit - the fit's own seeding and the held-out stream together,
    not one of them."""
    backend, dependence = request.node.callspec.params["fitted"]
    cum = _matrices()
    again = NNTransformerML().fit(
        _pooled_triangle(backend, cum),
        loss_field="paid_loss",
        as_of=AS_OF,
        config=_tiny(dependence),
        seed=0,
    )
    cells = _cells(fitted, SEG_A0)
    first = fitted.entry.predict_at(cells, field="paid_loss", seed=17)
    second = again.predict_at(cells, field="paid_loss", seed=17)
    assert first.tobytes() == second.tobytes()


# -- the shared draw loop ------------------------------------------------------


def test_the_shared_draw_loop_seeds_every_member_including_empty_ones():
    """One torch seed is drawn from the caller's generator per ensemble member,
    for EVERY member, including ones the split leaves with no draws.

    ``sample_mixture_draws`` is the piece this change lifted out of the
    single-line mixin so both entry shapes can call it, and this is the property
    the lift could most easily have lost: skipping the seed of a zero-draw
    member leaves the numpy stream in a different place, so the SAME call would
    return different draws depending only on how the requested count divides
    across the ensemble. Nothing downstream can see that - a draw count is never
    wrong, only different.

    Driven with stand-in members and a fixed per-cell mixture, because the
    property is about the generator rather than about any network: three
    members and two draws is the smallest split with an empty member.
    """
    from ibnr.gallery.nn._heldout import sample_mixture_draws

    params = (
        torch.zeros(1, 1),  # log_pi: one component, weight 1
        torch.zeros(1, 1),  # mu
        torch.ones(1, 1),  # sigma
    )
    rng = np.random.default_rng(0)
    draws = sample_mixture_draws(
        ["member-a", "member-b", "member-c"],
        n_draws=2,
        rng=rng,
        cell_params=lambda _model: params,
        pinned_cells=np.zeros(1, dtype=bool),
        device=torch.device("cpu"),
    )
    assert draws.shape == (2, 1)  # the split is [1, 1, 0]
    reference = np.random.default_rng(0)
    for _ in range(3):  # one seed per member, empty one included
        reference.integers(0, 2**63 - 1)
    assert int(rng.integers(0, 2**63 - 1)) == int(reference.integers(0, 2**63 - 1))


# -- density (ScoresHeldout) ---------------------------------------------------


def test_log_lik_refused_at_pinned_devs_but_draws_survive(fitted):
    """The documented asymmetry, both halves on the SAME cells: a pinned dev has
    no trained head, so the density is refused (naming the dev steps and the
    entry), while the draws still work."""
    entry = fitted.entry
    cells = _cells(fitted, SEG_A0)  # includes the pinned-dev cell
    with pytest.raises(ValueError, match=r"pinned dev step\(s\) \[6\]") as excinfo:
        entry.log_lik_at(cells, field="paid_loss")
    assert "nn_transformer_ml" in str(excinfo.value)
    draws = entry.predict_at(cells, field="paid_loss", seed=0)
    assert draws.shape == (entry.config_.heldout_n_draws, 5)

    # and the unpinned subset scores cleanly on the density axis too
    ll = entry.log_lik_at(_split(cells, pinned=False), field="paid_loss")
    assert ll.shape == (entry.config_.ensemble_size, 4)
    assert np.isfinite(ll).all()
    # the rows are the ensemble members and they must genuinely differ -
    # identically-seeded members would make logmeanexp a plug-in in disguise
    assert not np.allclose(ll[0], ll[1])


def test_log_lik_needs_a_real_ensemble(fitted):
    """One member is a plug-in density, not an ensemble average; the density
    axis needs at least 2 rows, and the entry should say WHY."""
    entry = fitted.entry
    cells = _split(_cells(fitted, SEG_A0), pinned=False)
    keep = entry.models_
    entry.models_ = keep[:1]
    try:
        with pytest.raises(ValueError, match="ensemble_size"):
            entry.log_lik_at(cells, field="paid_loss")
    finally:
        entry.models_ = keep


def test_log_lik_carry_is_exactly_log_premium(fitted):
    """The measure-carry layer: ``log_lik_at == _log_lik_native - log premium``.
    Pins that the entry declares loss_ratio and the base applies exactly that
    covariate - an entry declaring "amount" would pass every shape check and
    rank, while being unconverted."""
    entry = fitted.entry
    cells = _split(_cells(fitted, SEG_A0), pinned=False)
    view = entry.at_cohort(SEG_A0)
    idx = _indexed(view, cells)
    native = view._log_lik_native(idx)
    carried = view.log_lik_at(cells, field="paid_loss")
    np.testing.assert_allclose(carried, native - np.log(idx.premium)[None, :], rtol=1e-12)


def _independent_mixture(entry, segment, idx, *, line: int | None = None):
    """Mixture parameters of one line at ``idx``'s cells, recomputed from
    scratch: the test's own standardization, its own conditioning context, its
    own cutoff, its own forward pass and - for the joint head - its own
    marginalization through the full covariance matrix.

    ``line`` overrides which line is read out of the joint head, which is what
    the negative control below turns into a failure.
    """
    c, norm = entry.contract_, entry.norm_
    ci = entry.cohort_index({"company_code": segment["company_code"]})
    li = list(c["lob_levels"]).index(segment["line_of_business"])
    read = li if line is None else line

    x_norm = (c["x"][ci] - norm["mean"][:, :, None, :]) / norm["std"][:, :, None, :]
    x_norm = np.where(norm["pinned"][:, :, None, :], 0.0, x_norm)
    obs = c["obs_mask"][ci, li]
    cutoff = int(c["cal_idx"][obs].max())  # the scored line's as_of diagonal
    prem_norm = np.where(
        c["line_mask"][ci],
        (c["log_premium"][ci] - norm["prem_mean"]) / norm["prem_std"],
        0.0,
    )
    args = (
        torch.tensor(x_norm[None], dtype=torch.float32),
        torch.tensor(c["x_obs"][ci][None]),
        torch.tensor(c["line_mask"][ci][None]),
        torch.tensor(prem_norm[None], dtype=torch.float32),
        torch.tensor([cutoff], dtype=torch.long),
    )
    w0, d0 = idx.w - 1, idx.d - 1
    out = []
    with torch.no_grad():
        for model in entry.models_:
            if entry.config_.dependence == "ar":
                log_pi, mu, sigma = model.forward_ar(*args)
                out.append(
                    (
                        log_pi[0, read].numpy()[w0, d0],
                        mu[0, read].numpy()[w0, d0],
                        sigma[0, read].numpy()[w0, d0],
                    )
                )
                continue
            log_pi, mu, scale_tril = model.forward_joint(*args)
            lp = log_pi[0].numpy()[w0, d0]  # (n_cells, K)
            mu_all = mu[0].numpy()[w0, d0]  # (n_cells, K, L)
            chol = scale_tril[0].numpy()[w0, d0]  # (n_cells, K, L, L)
            # marginal of a mixture of multivariate Gaussians = the mixture of
            # the components' marginals, weights unchanged. Formed here through
            # the FULL covariance, independently of the implementation's
            # row-norm shortcut.
            cov = np.einsum("ckab,ckdb->ckad", chol, chol)
            out.append((lp, mu_all[..., read], np.sqrt(cov[..., read, read])))
    return [np.stack(a) for a in zip(*out, strict=True)]


def _expected_log_lik(entry, segment, idx, *, line: int | None = None) -> np.ndarray:
    """Hand-written log density at ``idx``'s cells, on Lebesgue-on-amount.

    Both Jacobians are written out here rather than borrowed: the ``-log
    std0[d]`` standardization change of variable and the ``-log premium``
    measure carry.
    """
    li = list(entry.contract_["lob_levels"]).index(segment["line_of_business"])
    mean0 = entry.norm_["mean"][li, 0]
    std0 = entry.norm_["std"][li, 0]
    d0 = idx.d - 1
    z = ((idx.value - idx.prev_value) / idx.premium - mean0[d0]) / std0[d0]
    lp, mu, sigma = _independent_mixture(entry, segment, idx, line=line)
    comp = -0.5 * ((z[None, :, None] - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    dens_z = np.log(np.exp(lp + comp).sum(axis=-1))  # small K: a direct sum is fine
    return dens_z - np.log(std0[d0])[None, :] - np.log(idx.premium)[None, :]


@pytest.mark.parametrize("segment", [SEG_A0, SEG_A1])
def test_log_lik_matches_independent_recomputation(fitted, segment):
    """Recompute the density from scratch - own forward pass, standardization,
    mixture assembly, joint-head marginalization and BOTH Jacobians written out
    by hand - and require agreement with ``log_lik_at`` to float32 accuracy.

    Runs under both heads (the fixture is parametrized on ``dependence``) and on
    two lines of one company, which is what catches a line axis read at a fixed
    index. Mutations this catches: dropping either Jacobian, conditioning at the
    wrong cutoff, reading another line's normalizer, scoring the cumulative
    value instead of the increment, and - for the joint head - taking the
    covariance's diagonal from the wrong row.
    """
    entry = fitted.entry
    cells = _split(_cells(fitted, segment), pinned=False)
    view = entry.at_cohort(segment)
    idx = _indexed(view, cells)
    got = view.log_lik_at(cells, field="paid_loss")
    expected = _expected_log_lik(entry, segment, idx)
    assert got.shape == expected.shape == (entry.config_.ensemble_size, 4)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_reading_the_wrong_line_of_the_joint_head_changes_the_answer(fitted):
    """Negative control on the marginalization: the check above can fail.

    The joint head carries one mixture over the whole line vector, so "which
    line" is an index into each component's mean and into the rows of its
    Cholesky factor. Reading the other line gives a different mean and a
    different variance, and the recomputation must notice - otherwise the
    agreement test would pass on an implementation that marginalized the wrong
    line. Only meaningful for the joint head; the "ar" head is checked the same
    way, by reading the other line's own univariate parameters.
    """
    entry = fitted.entry
    cells = _split(_cells(fitted, SEG_A0), pinned=False)
    view = entry.at_cohort(SEG_A0)
    idx = _indexed(view, cells)
    got = view.log_lik_at(cells, field="paid_loss")
    wrong = _expected_log_lik(entry, SEG_A0, idx, line=1)
    assert np.isfinite(wrong).all()
    assert not np.allclose(got, wrong, rtol=1e-3), "the wrong line gave the same density"


def test_density_normalizes_over_the_amount_space(fitted):
    """Integrate the ensemble-average predictive density over the CUMULATIVE
    amount at one held-out cell and require mass 1.

    This is the only check that catches a wrong change of variable
    (kernels/densities.py): dropping ``-log std0[d]`` leaves a smooth,
    plausible, correctly-ranking density that integrates to 1/std0 - here orders
    of magnitude off. The integration runs through ``log_lik_at`` itself, so
    both Jacobians are on the path. Bounds come from the mixture parameters
    (plus or minus 12 component standard deviations), keeping the interval tight
    enough for the quadrature to see the peak.
    """
    from ibnr.kernels.densities import check_normalization
    from ibnr.kernels.forecast import logmeanexp
    from ibnr.kernels.holdout import CellIndex

    entry = fitted.entry
    view = entry.at_cohort(SEG_A0)
    cells = _cells(fitted, SEG_A0)
    one = replace(cells, frame=cells.frame[cells.frame["dev_lag"] == 36].reset_index(drop=True))
    assert one.n_cells == 1  # dev step 3: unpinned
    idx = _indexed(view, one)
    prem = float(idx.premium[0])
    prev = float(idx.prev_value[0])
    d0 = int(idx.d[0]) - 1
    mean0 = entry.norm_["mean"][0, 0][d0]
    std0 = entry.norm_["std"][0, 0][d0]

    log_pi, mu, sigma = entry._heldout_mixture((0, 0), idx)  # (n_members, 1, K)
    centers = prev + prem * (mean0 + std0 * mu[:, 0, :])
    scales = prem * std0 * sigma[:, 0, :]
    lo = float((centers - 12.0 * scales).min())
    hi = float((centers + 12.0 * scales).max())

    def logpdf(ys):
        arr = np.atleast_1d(np.asarray(ys, dtype=float))
        cix = CellIndex(
            w=np.full(arr.size, idx.w[0], dtype=int),
            d=np.full(arr.size, idx.d[0], dtype=int),
            value=arr,
            prev_value=np.full(arr.size, prev),
            premium=np.full(arr.size, prem),
        )
        return logmeanexp(view.log_lik_at(cix), axis=0)

    mass = check_normalization(logpdf, lo=lo, hi=hi, tol=1e-4)
    assert abs(mass - 1.0) <= 1e-4
