"""nn_transformer held-out wiring: the per-cohort adapter contract, incremental
draws (CRPS axis) and the MDN density (ELPD axis). Skips without torch.

What this file protects, in three layers:

1. **The adapter.** ``gallery/nn/_heldout.py::cohort_contract`` maps one cohort
   of the pooled ``nn_data`` contract into the shape ``kernels.holdout.index_into``
   demands, so every one of its guards - cohort identity, segment schema,
   training overlap - applies to the NN entry exactly as to a Stan one. The
   tests drive the guards through real refusals, not just the happy path.
2. **Draws.** ``predict_at`` returns incremental draws anchored to each cell's
   training-diagonal predecessor BY THE BASE CLASS (``heldout_draw_scale =
   "incremental"``); the anchor here is ~2 orders of magnitude above the
   increment, so a dropped conversion is unmistakable. Pinned devs keep
   rollout semantics (point mass at the pooled dev mean) and an all-pinned
   request is refused.
3. **Density.** ``_heldout_log_lik`` is checked against an independent
   recomputation (its own forward pass in the test, the standardization
   Jacobian written out by hand) and against ``densities.check_normalization``
   over the amount space - the only check that catches a wrong Jacobian. The
   pinned-dev refusal is the documented asymmetry: the same cells stay
   CRPS-scorable where they are not ELPD-scorable.

The fixture is a 6x6 full square per LOB with ``as_of`` at diagonal 6, so the
next diagonal holds 5 cells at dev steps 6, 5, 4, 3, 2 - of which dev 6 is
PINNED (zero training-context values: its only observation sits on the
validation diagonal; dev 5 keeps two context values because the normalizer
pools BOTH lob cohorts). That split is what lets one fixture exercise both
sides of the asymmetry; a sanity test pins it so fixture rot cannot silently
empty either side.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr import gallery  # noqa: E402
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout  # noqa: E402
from ibnr.gallery.nn._heldout import PooledMDNHeldout, cohort_contract  # noqa: E402
from ibnr.gallery.nn.transformer.config import TransformerConfig  # noqa: E402
from ibnr.gallery.nn.transformer.model import NNTransformer  # noqa: E402
from ibnr.kernels.densities import MEASURES, check_normalization  # noqa: E402
from ibnr.kernels.forecast import logmeanexp  # noqa: E402
from ibnr.kernels.holdout import (  # noqa: E402
    CellIndex,
    HoldoutCells,
    index_into,
    next_diagonal,
)
from ibnr.kernels.rng import heldout_stream  # noqa: E402

from .conftest import BACKENDS, make_multiline_triangle  # noqa: E402

TINY = TransformerConfig(
    d_model=16,
    n_layers=1,
    n_heads=2,
    ffn_dim=32,
    dropout=0.0,
    n_components=2,
    lob_embedding_dim=4,
    batch_size=8,
    max_epochs=3,
    patience=5,
    ensemble_size=2,
    # the two draw budgets differ on purpose: they are independent knobs (the
    # rollout's and the held-out diagonal's), and equal values would let either
    # one stand in for the other everywhere below
    n_draws=50,
    heldout_n_draws=40,
)

START = 2000
AS_OF = "2005-12-31"  # diagonal 6 of a 6x6 square starting in 2000
SEG0 = {"company_code": "0001", "line_of_business": "lob_0"}
SEG1 = {"company_code": "0001", "line_of_business": "lob_1"}
#: dev steps of the next-diagonal cells whose per-dev normalizer is pinned
#: (fewer than two training-context values at as_of = diagonal 6; dev 5 keeps
#: two because the normalizer pools both lob cohorts)
PINNED_DEVS = (6,)


def _matrices() -> np.ndarray:
    """(2, 6, 6) cumulative squares with a decaying incremental pattern -
    shared by the pooled fit triangle and the per-cohort cells triangles, so
    the cells describe exactly the data the entry trained on."""
    rng = np.random.default_rng(0)
    n_w = n_d = 6
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    return np.cumsum(incr, axis=2)


@pytest.fixture(scope="module", params=BACKENDS)
def fitted(request):
    """One TINY pooled fit per backend, plus each cohort's next-diagonal cells.

    Module-scoped: the fit is the expensive part and every test here leaves it
    as it found it (the one that mutates ``models_`` restores it in a
    ``finally``; the one that calls ``predict`` fills the rollout cache, which
    is derived state nothing else here reads).
    """
    backend = request.param
    cum = _matrices()
    prem = np.full(6, 1000.0)
    pooled = make_multiline_triangle(
        backend,
        {"lob_0": cum[0], "lob_1": cum[1]},
        premium_by_lob={"lob_0": prem, "lob_1": prem},
        start_year=START,
    )
    entry = NNTransformer().fit(pooled, loss_field="paid_loss", as_of=AS_OF, config=TINY, seed=0)

    def cells_for(lob: str, matrix: np.ndarray):
        one = make_multiline_triangle(
            backend, {lob: matrix}, premium_by_lob={lob: prem}, start_year=START
        )
        return next_diagonal(one, as_of=AS_OF, fields="paid_loss", premium_field="earned_premium")

    return SimpleNamespace(
        entry=entry,
        cells0=cells_for("lob_0", cum[0]),
        cells1=cells_for("lob_1", cum[1]),
    )


def _split(cells, *, pinned: bool):
    """The next-diagonal cells at pinned (or unpinned) dev steps only."""
    dev_lags = [12 * d for d in PINNED_DEVS]
    mask = cells.frame["dev_lag"].isin(dev_lags)
    frame = cells.frame[mask if pinned else ~mask].reset_index(drop=True)
    return replace(cells, frame=frame)


# -- fixture sanity ------------------------------------------------------------


def test_fixture_shape_is_what_the_file_claims(fitted):
    """Pin the structure every other test relies on: 5 scorable cells at dev
    steps {2..6}, nothing excluded, and exactly dev 6 pinned. If the triangle
    or the as_of drifts, this fails first and names the rot."""
    cells = fitted.cells0
    assert cells.n_cells == 5
    assert cells.exclusion_counts() == {
        "new_origin": 0,
        "dev_beyond_trained": 0,
        "no_predecessor": 0,
    }
    assert sorted(cells.frame["dev_lag"] // 12) == [2, 3, 4, 5, 6]
    pinned = fitted.entry.norm_["pinned"][0]  # target channel, (n_d,)
    np.testing.assert_array_equal(pinned, [False, False, False, False, False, True])
    # both sides of the pinned/unpinned split are non-empty, or the asymmetry
    # tests below would silently test nothing
    assert _split(cells, pinned=True).n_cells == 1
    assert _split(cells, pinned=False).n_cells == 4


def test_entry_declares_both_capabilities(fitted):
    """The scale declarations the base classes act on, and the mixin claims
    themselves. ``heldout_measure`` must be a real MEASURES key or the carry
    would fail on every call rather than here."""
    entry = fitted.entry
    assert isinstance(entry, ScoresHeldout) and isinstance(entry, PredictsHeldout)
    assert entry.heldout_measure == "loss_ratio" and entry.heldout_measure in MEASURES
    assert entry.heldout_draw_scale == "incremental"
    view = entry.at_cohort(SEG0)
    assert isinstance(view, ScoresHeldout) and isinstance(view, PredictsHeldout)
    assert view.heldout_measure == "loss_ratio"
    assert view.heldout_draw_scale == "incremental"


# -- the adapter ---------------------------------------------------------------


def test_cohort_contract_on_a_hand_built_contract():
    """The adapter mapping itself, on a deliberately ASYMMETRIC two-cohort
    dict - the fitted fixture's staircase is symmetric in (w, d), so a w/d
    swap in the adapter would slip through it. Also pins the anchor rule: a
    per-origin anchor whose own increment was unusable (predecessor hole) is
    absent from obs_mask but WAS training data, and must be declared trained
    or the overlap guard would let it be scored as held out."""
    import datetime as dtm

    import pandas as pd

    obs = np.zeros((2, 2, 3), dtype=bool)  # (n_c, n_w, n_d): 2 origins x 3 devs
    obs[0, 0, :] = True  # cohort 0, origin 1: devs 1..3 usable
    obs[0, 1, 0] = True  # cohort 0, origin 2: dev 1 usable...
    latest = np.array([[3, 3], [1, 1]])  # ...but ANCHORED at dev 3 (hole at dev 2)
    contract = {
        "cohorts": pd.DataFrame({"lob": ["a", "b"]}),
        "obs_mask": obs,
        "latest_dev": latest,
        "premium": np.array([[100.0, 200.0], [1.0, 2.0]]),
        "fields": ["paid_loss"],
        "origin_periods": [dtm.date(2000, 1, 1), dtm.date(2001, 1, 1)],
        "dev_grain_months": 12,
        "n_w": 2,
        "n_d": 3,
    }
    got = cohort_contract(contract, 0, models=("paid_loss",))
    declared = set(zip(got["w"].tolist(), got["d"].tolist(), strict=True))
    assert declared == {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)}
    # guard on the guard: the expected set is not w/d symmetric
    assert declared != {(d, w) for w, d in declared}
    assert got["segment"] == {"lob": "a"}
    assert got["models"] == ["paid_loss"] and got["measure"] == "cumulative"
    np.testing.assert_allclose(got["premium"], [100.0, 200.0])  # cohort 0's row, 1-D
    with pytest.raises(IndexError, match="out of range"):
        cohort_contract(contract, 5, models=("paid_loss",))


def test_adapter_declares_the_predecessor_of_every_obs_cell():
    """Cumulative values at devs {1, 2, 4, 5}: the dev-3 hole makes the dev-4
    increment unusable, so obs = {1, 2, 5} and the anchor is dev 5 - dev 4 is
    neither, yet its VALUE fed the dev-5 increment during training. The
    declared set must be the honest closure rather than relying on
    ``next_diagonal`` excluding such cells as ``no_predecessor`` downstream:
    ``index_into`` must refuse the dev-4 cell as training data. Dev 3, the
    genuine hole, stays undeclared - the closure is minimal, not a blanket
    fill.

    Mutation this must catch: drop the predecessor-closure union in
    ``cohort_contract`` - ``index_into`` then accepts the dev-4 cell and the
    ``raises`` below fails.
    """
    import datetime as dtm

    import pandas as pd

    obs = np.zeros((1, 1, 5), dtype=bool)
    obs[0, 0, [0, 1, 4]] = True  # usable increments at devs 1, 2, 5
    contract = {
        "cohorts": pd.DataFrame({"lob": ["a"]}),
        "obs_mask": obs,
        "latest_dev": np.array([[5]]),
        "premium": np.array([[100.0]]),
        "fields": ["paid_loss"],
        "origin_periods": [dtm.date(2000, 1, 1)],
        "dev_grain_months": 12,
        "n_w": 1,
        "n_d": 5,
    }
    adapter = cohort_contract(contract, 0, models=("paid_loss",))
    declared = set(zip(adapter["w"].tolist(), adapter["d"].tolist(), strict=True))
    # obs {1, 2, 5} + anchor {5} + predecessors {1, 4}; dev 3 undeclared
    assert declared == {(1, 1), (1, 2), (1, 4), (1, 5)}

    frame = pd.DataFrame(
        [
            {
                "lob": "a",
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
        segments=("lob",),
        measure="cumulative",
    )
    with pytest.raises(ValueError, match="TRAINING data"):
        index_into(cells, adapter, field="paid_loss")


def test_adapter_indexes_cells_correctly(fitted):
    """``index_into`` accepts the per-cohort adapter and produces the right
    (w, d)/value/premium mapping. The declared training cells are exactly the
    as_of staircase, so the overlap guard has the full training set to check
    against - a short list would wave training cells through."""
    entry = fitted.entry
    view = entry.at_cohort(SEG0)
    adapter = view.contract_
    assert adapter["segment"] == SEG0
    assert adapter["models"] == ["paid_loss"]
    assert adapter["measure"] == "cumulative"
    # training cells = the 21-cell staircase w + d <= 7 (diagonals 1..6)
    declared = set(zip(adapter["w"].tolist(), adapter["d"].tolist(), strict=True))
    assert declared == {(w, d) for w in range(1, 7) for d in range(1, 7) if w + d <= 7}
    assert adapter["premium"].shape == (6,)
    np.testing.assert_allclose(adapter["premium"], 1000.0)

    idx = index_into(fitted.cells0, adapter, field="paid_loss")
    assert idx.n_cells == 5
    # the next diagonal: w + d == 8, in frame (origin, dev) sort order
    np.testing.assert_array_equal(idx.w + idx.d, np.full(5, 8))
    np.testing.assert_allclose(idx.value, fitted.cells0.values)
    np.testing.assert_allclose(idx.premium, 1000.0)
    # the standalone helper builds the same dict the view carries
    direct = cohort_contract(entry.contract_, entry.cohort_index(SEG0), models=("paid_loss",))
    np.testing.assert_array_equal(direct["w"], adapter["w"])
    np.testing.assert_array_equal(direct["d"], adapter["d"])
    assert direct["segment"] == adapter["segment"]


def test_adapter_refuses_wrong_cohort(fitted):
    """(w, d) alone cannot identify a cell: lob_1's cells index cleanly into
    lob_0's adapter and would score the wrong cohort silently. The identity
    guard must catch it - through the view's public method, so the guard is on
    the path callers actually use."""
    view0 = fitted.entry.at_cohort(SEG0)
    with pytest.raises(ValueError, match="was trained on"):
        view0.log_lik_at(fitted.cells1, field="paid_loss")
    with pytest.raises(ValueError, match="was trained on"):
        view0.predict_at(fitted.cells1, field="paid_loss", seed=0)


def test_adapter_refuses_training_cells(fitted):
    """A held-out score computed on training data is systematically too good.
    Rewrite one cell key to a trained (w, d) and the overlap guard must fire."""
    frame = fitted.cells0.frame.copy()
    frame.loc[0, "dev_lag"] = 12  # (origin 2000, dev 1) is training data
    doctored = replace(fitted.cells0, frame=frame)
    with pytest.raises(ValueError, match="TRAINING data"):
        fitted.entry.predict_at(doctored, field="paid_loss", seed=0)


def test_entry_needs_cells_that_name_a_cohort(fitted):
    """The entry-level methods resolve the cohort from the cells, so a bare
    CellIndex (which carries no segment values) is refused with a pointer at
    at_cohort - not mis-scored against an arbitrary cohort."""
    idx = index_into(fitted.cells0, fitted.entry.at_cohort(SEG0).contract_, field="paid_loss")
    with pytest.raises(TypeError, match="at_cohort"):
        fitted.entry.log_lik_at(idx)


# -- draws (PredictsHeldout) ---------------------------------------------------


def test_predict_at_shape_seed_and_variance(fitted):
    """Draw contract: (config.heldout_n_draws, n_cells); reproducible per seed;
    live cells have genuine spread; pinned cells are the rollout-semantics
    point mass at anchor + premium * pooled dev mean, exactly."""
    entry = fitted.entry
    cells = fitted.cells0
    a = entry.predict_at(cells, field="paid_loss", seed=11)
    assert a.shape == (TINY.heldout_n_draws, 5)
    assert np.isfinite(a).all()
    b = entry.predict_at(cells, field="paid_loss", seed=11)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, entry.predict_at(cells, field="paid_loss", seed=12))

    frame = cells.frame
    pinned_col = frame["dev_lag"].isin([12 * d for d in PINNED_DEVS]).to_numpy()
    assert (a[:, ~pinned_col].std(axis=0) > 0).all()
    # pinned columns: zero spread, value = anchor + premium * pooled dev mean
    mean0 = fitted.entry.norm_["mean"][0]
    d0 = (frame["dev_lag"] // 12).to_numpy() - 1
    expected = frame["prev_value"].to_numpy() + 1000.0 * mean0[d0]
    # exact point mass: max == min per column (std would show a ~1e-13 numpy
    # mean-rounding artifact even on bit-identical values)
    assert (a[:, pinned_col].max(axis=0) == a[:, pinned_col].min(axis=0)).all()
    np.testing.assert_allclose(a[0, pinned_col], expected[pinned_col], rtol=1e-6)


def test_predict_at_anchors_increments_through_the_base_class(fitted):
    """The anchor conversion is the base class's job and it must actually run.

    The entry draws INCREMENTS (~30-500 here) while the triangle is cumulative
    (anchors ~450-2000), so draws scored without the declared-scale conversion
    are wrong by the whole anchor - the CRPS-996-where-truth-is-3.4 bug class.
    ``predict_at`` derives its generator from the seed together with the cells'
    cohort identity, the field and the cutoff (``kernels.rng``), so rebuilding
    that stream here makes the native draws reproducible and the assertion
    exact equality.

    Mutation this must catch: ``heldout_draw_scale = "cumulative"`` (or any
    skipped conversion) makes ``got == native``, off by every anchor.
    """
    entry = fitted.entry
    cells = fitted.cells0
    view = entry.at_cohort(SEG0)
    idx = index_into(cells, view.contract_, field="paid_loss")
    stream = heldout_stream(11, cells, field="paid_loss")
    native = view._draws_native(idx, rng=np.random.default_rng(stream))
    got = entry.predict_at(cells, field="paid_loss", seed=11)
    anchor = idx.prev_value
    np.testing.assert_allclose(got, native + anchor[None, :], rtol=1e-12)
    # the shift is huge relative to the draws, so the mutant is unmistakable
    assert anchor.min() > 100.0
    assert np.abs(native).mean() < anchor.min()


def test_predict_at_refuses_all_pinned_cells(fitted):
    """A request whose every cell is pinned would return a point mass in every
    column - not a predictive distribution - and must be refused loudly
    (CohortForecast would reject the array anyway, but with a message about
    variance, not about the cause)."""
    only_pinned = _split(fitted.cells0, pinned=True)
    with pytest.raises(ValueError, match="pinned"):
        fitted.entry.predict_at(only_pinned, field="paid_loss", seed=0)


# -- the two draw budgets ------------------------------------------------------


def test_the_two_draw_budgets_are_delivered_independently(fitted):
    """One config, two knobs, two paths: ``predict_at`` spends
    ``heldout_n_draws`` and ``predict`` spends ``n_draws``.

    Held-out draws used to come off the ROLLOUT's ``n_draws``, so the NN
    entries reached the CRPS board on 1,000 draws where every other
    CRPS-capable entry delivered 10,000 - and nothing could see it, because a
    draw count is never wrong, only small. The split is what makes 10,000
    affordable: a held-out diagonal is one forward pass per ensemble member
    whatever the count, while raising ``n_draws`` to match would pay for it
    once per future diagonal of the rollout.

    Both halves are asserted from the same fit, because a fix that simply
    renamed the field would move BOTH counts and pass either half alone.
    """
    entry = fitted.entry
    assert TINY.heldout_n_draws != TINY.n_draws  # or either could stand in
    draws = entry.predict_at(fitted.cells0, field="paid_loss", seed=3)
    assert draws.shape == (TINY.heldout_n_draws, 5)
    # n_draws=None is the spelling that reads the config, i.e. the rollout knob
    # on its default path - untouched by the split
    pred = entry.predict(segment=SEG0, n_draws=None, seed=3)
    assert pred.samples.shape[0] == TINY.n_draws


def test_every_heldout_nn_entry_defaults_to_ten_thousand_heldout_draws():
    """The default every held-out NN entry puts on the board, which is the
    10,000 draws the Bayesian posteriors and ``mack`` already deliver.

    Derived from the registry rather than a hand list: an entry is covered when
    it is an NN entry that can draw at held-out cells, so one cloned from any of
    these joins the pin the day it registers instead of shipping a smaller
    default under a new name.

    The rule is deliberately NOT "inherits ``PooledMDNHeldout``".
    ``nn_transformer_ml`` cannot inherit it - its cohort is a company while a
    held-out cohort is a (company, line) pair, so it carries its own adapter -
    and it would have slipped out of this pin the day it gained held-out
    scoring, which is the hand-list failure one level up.
    """
    configs = {
        name: cls.config_class
        for name, cls in ((n, gallery.get(n)) for n in gallery.list())
        if cls.family == "nn" and issubclass(cls, PredictsHeldout)
    }
    # guard on the guard: a loop over an empty registry passes for free. Six is
    # what this release ships (transformer, transformer_ml, mdn, resnet,
    # deeptriangle, nn_paid_case); a seventh joins the loop by registering,
    # without moving this floor.
    assert len(configs) >= 6
    # and every entry that CAN share one mixin still does, so the split above
    # stays a statement about the multi-line contract rather than a licence to
    # fork the scoring code once per entry
    assert sum(issubclass(gallery.get(n), PooledMDNHeldout) for n in configs) == len(configs) - 1
    for name, config_class in configs.items():
        assert config_class().heldout_n_draws == 10_000, name


# -- density (ScoresHeldout) ---------------------------------------------------


def test_log_lik_refused_at_pinned_devs_but_draws_survive(fitted):
    """The documented asymmetry, both halves on the SAME cells: a pinned dev
    has no trained head, so the density is refused (naming the dev steps),
    while the draws still work - the entry is CRPS-scorable where it is not
    ELPD-scorable."""
    cells = fitted.cells0  # includes the pinned-dev cell
    with pytest.raises(ValueError, match=r"pinned dev step\(s\) \[6\]"):
        fitted.entry.log_lik_at(cells, field="paid_loss")
    draws = fitted.entry.predict_at(cells, field="paid_loss", seed=0)
    assert draws.shape == (TINY.heldout_n_draws, 5)

    # and the unpinned subset scores cleanly on the density axis too
    ll = fitted.entry.log_lik_at(_split(cells, pinned=False), field="paid_loss")
    assert ll.shape == (TINY.ensemble_size, 4)
    assert np.isfinite(ll).all()
    # the rows are the ensemble members and they must genuinely differ -
    # identically-seeded members would make logmeanexp a plug-in in disguise
    assert not np.allclose(ll[0], ll[1])


def test_log_lik_matches_independent_recomputation(fitted):
    """Recompute the density from scratch - own forward pass, standardization,
    mixture assembly and BOTH Jacobians written out by hand - and require
    agreement with ``log_lik_at`` to float32 accuracy.

    Everything after the shared fitted state (weights, norm stats) is
    independent: the conditioning context, the as_of cutoff, ``z``, the
    ``-log std0[d]`` standardization Jacobian and the ``-log premium`` measure
    carry. Mutations this catches: dropping either Jacobian term, conditioning
    at the wrong cutoff, reading the wrong dev's normalizer, and scoring the
    cumulative value instead of the increment.
    """
    entry = fitted.entry
    cells = _split(fitted.cells0, pinned=False)
    view = entry.at_cohort(SEG0)
    idx = index_into(cells, view.contract_, field="paid_loss")
    got = view.log_lik_at(cells, field="paid_loss")  # (n_members, n_cells), amount scale

    c = entry.contract_
    ci = entry.cohort_index(SEG0)
    norm = entry.norm_
    x_norm = (c["x"][ci] - norm["mean"][:, None, :]) / norm["std"][:, None, :]
    x_norm = np.where(norm["pinned"][:, None, :], 0.0, x_norm)
    obs = c["obs_mask"][ci]
    cutoff = int(c["cal_idx"][obs].max())  # the as_of diagonal
    assert cutoff == 6
    xt = torch.tensor(x_norm[None], dtype=torch.float32)
    # the entry conditions on per-channel observedness, so the independent
    # recomputation must hand the network the same (F, W, D) mask it gets
    ctx = torch.tensor(c["x_obs"][ci][None])
    lob = torch.tensor([c["lob_idx"][ci]], dtype=torch.long)
    prem_feat = torch.tensor(
        [(c["log_premium"][ci] - norm["prem_mean"]) / norm["prem_std"]], dtype=torch.float32
    )
    cut = torch.tensor([cutoff], dtype=torch.long)

    w0, d0 = idx.w - 1, idx.d - 1
    mean0, std0 = norm["mean"][0], norm["std"][0]
    increment = idx.value - idx.prev_value
    z = (increment / idx.premium - mean0[d0]) / std0[d0]
    expected = np.empty((len(entry.models_), idx.n_cells))
    with torch.no_grad():
        for m, model in enumerate(entry.models_):
            log_pi, mu, sigma = model(xt, ctx, lob, prem_feat, cut)
            lp = log_pi[0].numpy()[w0, d0]  # (n_cells, K)
            mu_ = mu[0].numpy()[w0, d0]
            sg = sigma[0].numpy()[w0, d0]
            comp = -0.5 * ((z[:, None] - mu_) / sg) ** 2 - np.log(sg) - 0.5 * np.log(2.0 * np.pi)
            dens_z = np.log(np.exp(lp + comp).sum(axis=1))  # small K: direct sum is fine
            # z -> ratio Jacobian, then ratio -> amount measure carry
            expected[m] = dens_z - np.log(std0[d0]) - np.log(idx.premium)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_log_lik_carry_is_exactly_log_premium(fitted):
    """The measure-carry layer: ``log_lik_at == _log_lik_native - log premium``.
    Pins that the entry declares loss_ratio and the base applies exactly that
    covariate - an entry declaring "amount" would pass every shape check and
    rank, while being unconverted."""
    entry = fitted.entry
    cells = _split(fitted.cells0, pinned=False)
    view = entry.at_cohort(SEG0)
    idx = index_into(cells, view.contract_, field="paid_loss")
    native = view._log_lik_native(idx)
    carried = view.log_lik_at(cells, field="paid_loss")
    np.testing.assert_allclose(carried, native - np.log(idx.premium)[None, :], rtol=1e-12)


def test_density_normalizes_over_the_amount_space(fitted):
    """Integrate the ensemble-average predictive density over the CUMULATIVE
    amount at one held-out cell and require mass 1 to 1e-4.

    This is the only check that catches a wrong change of variable
    (kernels/densities.py): dropping ``-log std0[d]`` leaves a smooth,
    plausible, correctly-ranking density that integrates to 1/std0 - here
    orders of magnitude off. The integration runs through ``log_lik_at``
    itself, so both Jacobians (standardization and premium) are on the path.
    Bounds come from the mixture parameters (+/- 12 component sd), keeping the
    interval tight enough for quad to see the peak.
    """
    entry = fitted.entry
    view = entry.at_cohort(SEG0)
    one = replace(
        fitted.cells0,
        frame=fitted.cells0.frame[fitted.cells0.frame["dev_lag"] == 36].reset_index(drop=True),
    )
    assert one.n_cells == 1  # dev step 3: unpinned
    idx = index_into(one, view.contract_, field="paid_loss")
    ci = entry.cohort_index(SEG0)
    prem = float(idx.premium[0])
    prev = float(idx.prev_value[0])
    d0 = int(idx.d[0]) - 1
    mean0, std0 = entry.norm_["mean"][0][d0], entry.norm_["std"][0][d0]

    log_pi, mu, sigma = entry._heldout_mixture(ci, idx)  # (n_members, 1, K)
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


def test_log_lik_needs_a_real_ensemble(fitted):
    """One member is a plug-in density, not an ensemble average; the density
    axis needs >= 2 rows (logmeanexp and CohortForecast both require it, but
    the entry should say WHY, not fail downstream)."""
    entry = fitted.entry
    cells = _split(fitted.cells0, pinned=False)
    keep = entry.models_
    entry.models_ = keep[:1]
    try:
        with pytest.raises(ValueError, match="ensemble_size"):
            entry.log_lik_at(cells, field="paid_loss")
    finally:
        entry.models_ = keep
