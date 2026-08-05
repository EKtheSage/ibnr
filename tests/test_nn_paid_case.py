"""gallery.nn.nn_paid_case: the joint paid + case-reserve entry. Skips without torch.

``tests/test_paid_case_head.py`` protects the bivariate head's mathematics; this
file protects everything the entry does WITH it, and every entry-level test runs
on BOTH backbones (``transformer`` and ``gru``) because the config selects a
different network module, a different input set and a different held-out input
dict - three places a second body can diverge silently.

What is pinned here, in the order the fit visits it:

1. **The config refuses a knob the chosen backbone cannot read**, by name. One
   dataclass covers two bodies, so the alternative is an inert parameter: a
   config saying ``d_model=256`` while a 64-wide GRU trains.
2. **The movement target's derivation.** ``move[d] = level[d] - level[d-1]``
   where both cells are present, the dev-1 movement IS the level, and the
   movement gets its OWN per-dev standardization - not the level channel's,
   which is a different quantity on the same scale.
3. **The rollout promotes BOTH channels**, which is the exact mirror of the
   other four NN entries' stay-off test: they promote channel 0 alone because
   they simulated nothing else, and this entry simulated both.
4. **The state update is integration in RATIO space.** With the sampler pinned
   to a constant, the channel-1 value at every promoted cell is rebuilt in numpy
   from ``level += movement`` and compared against what the network was actually
   handed on the NEXT forward pass. The mutation this exists for is adding
   standardized values (``z_level + z_move``), which produces smooth, finite,
   entirely believable draws.
5. **Held-out delivery**: ``_forward_mixture`` IS ``head.paid_margin`` of the
   raw output (not a re-derived margin), and the two draw budgets are
   independent.

Fixtures are deliberately under-powered (6x6 squares, two cohorts, two epochs,
two ensemble members): nothing here asserts predictive quality. duckdb only -
every claim below is numpy over a fitted contract, so a second ibis backend
would re-run six ensemble trainings to re-check an engine that never sees them.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ibnr import Triangle, gallery  # noqa: E402
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout  # noqa: E402
from ibnr.gallery.nn._scheme import norm_stats, splits  # noqa: E402
from ibnr.gallery.nn.nn_paid_case import head  # noqa: E402
from ibnr.gallery.nn.nn_paid_case.config import BACKBONE_KNOBS, NNPaidCaseConfig  # noqa: E402
from ibnr.gallery.nn.nn_paid_case.model import (  # noqa: E402
    NNPaidCase,
    call_backbone,
    case_movement,
)
from ibnr.gallery.nn.nn_paid_case.network_gru import PaidCaseGRU  # noqa: E402
from ibnr.gallery.nn.nn_paid_case.network_transformer import PaidCaseTransformer  # noqa: E402
from ibnr.kernels.densities import MEASURES  # noqa: E402
from ibnr.kernels.holdout import next_diagonal  # noqa: E402

from .conftest import make_multiline_triangle  # noqa: E402

START = 2000
N = 6
AS_OF = "2005-12-31"  # calendar diagonal 6 of a 6x6 square starting in 2000
SEG0 = {"company_code": "0001", "line_of_business": "lob_0"}
SEG1 = {"company_code": "0001", "line_of_business": "lob_1"}
PREMIUM = 1000.0

#: both bodies, through every entry-level test below
BACKBONES = ("transformer", "gru")


def tiny(backbone: str) -> NNPaidCaseConfig:
    """An under-powered config for one backbone, with the other body's knobs
    left at their defaults (the config refuses anything else)."""
    common = dict(
        dropout=0.0,
        n_components=2,
        lob_embedding_dim=4,
        batch_size=8,
        max_epochs=2,
        patience=5,
        ensemble_size=2,
        n_draws=40,  # rollout draws (predict)
        heldout_n_draws=120,  # held-out diagonal draws (predict_at); 10,000 by default
    )
    if backbone == "transformer":
        return NNPaidCaseConfig(
            backbone="transformer", d_model=16, n_layers=1, n_heads=2, ffn_dim=32, **common
        )
    return NNPaidCaseConfig(backbone="gru", hidden_dim=16, **common)


def _paid() -> np.ndarray:
    """(2, N, N) cumulative paid squares with a decaying incremental pattern -
    the family's fixture shape, so the pinned-dev structure carries over."""
    rng = np.random.default_rng(0)
    dev_level = np.exp(np.linspace(-0.8, -3.0, N))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, N, N))
    return np.cumsum(incr, axis=2)


def _case(paid: np.ndarray) -> np.ndarray:
    """(2, N, N) case reserve LEVELS that drain as the paid develops.

    ``0.6 x (ultimate - paid to date)`` plus a floor, so the level falls with
    development the way a real case reserve does - which is what makes the
    movement target predominantly negative and the drain diagnostic meaningful.
    """
    ult = paid[:, :, -1][:, :, None]
    return 0.6 * (ult - paid) + 25.0


def paid_and_case_triangle() -> Triangle:
    """Two-LOB company square carrying paid_loss, case_reserve and premium."""
    paid = _paid()
    case = _case(paid)
    prem = {f"lob_{k}": np.full(N, PREMIUM) for k in range(2)}
    t_paid = make_multiline_triangle(
        "duckdb",
        {"lob_0": paid[0], "lob_1": paid[1]},
        premium_by_lob=prem,
        start_year=START,
    )
    t_case = make_multiline_triangle(
        "duckdb",
        {"lob_0": case[0], "lob_1": case[1]},
        loss_field="case_reserve",
        start_year=START,
    )
    df = pd.concat([t_paid.execute(), t_case.execute()], ignore_index=True)
    return Triangle.from_long(df, measure="cumulative", backend="duckdb")


def paid_only_triangle() -> Triangle:
    """The same company with NO case field - what the fit must refuse by name."""
    paid = _paid()
    prem = {f"lob_{k}": np.full(N, PREMIUM) for k in range(2)}
    return make_multiline_triangle(
        "duckdb", {"lob_0": paid[0], "lob_1": paid[1]}, premium_by_lob=prem, start_year=START
    )


@pytest.fixture(scope="module", params=BACKBONES)
def fitted(request):
    """One TINY pooled fit per backbone at ``AS_OF``, plus lob_0's held-out cells.

    Module-scoped: the fit is the expensive part and every test is read-only
    against it (the two that drive a rollout drop the cache afterwards).
    """
    backbone = request.param
    cfg = tiny(backbone)
    triangle = paid_and_case_triangle()
    entry = NNPaidCase().fit(triangle, as_of=AS_OF, config=cfg, seed=0)
    paid = _paid()
    one = make_multiline_triangle(
        "duckdb",
        {"lob_0": paid[0]},
        premium_by_lob={"lob_0": np.full(N, PREMIUM)},
        start_year=START,
    )
    cells = next_diagonal(one, as_of=AS_OF, fields="paid_loss", premium_field="earned_premium")
    return SimpleNamespace(
        backbone=backbone, config=cfg, entry=entry, triangle=triangle, cells=cells
    )


def _drop_rollout_cache(entry) -> None:
    """Leave no spied/stubbed rollout behind for the next test on this fixture."""
    entry._rollout_key = None
    entry._rollout_ults = None
    entry._rollout_case = None


# -- config: two bodies, one dataclass -----------------------------------------


@pytest.mark.parametrize(
    "backbone,knob,value",
    [
        ("gru", "d_model", 256),
        ("gru", "n_layers", 4),
        ("gru", "n_heads", 8),
        ("gru", "ffn_dim", 512),
        ("transformer", "hidden_dim", 128),
        ("transformer", "company_embedding", True),
        ("transformer", "company_embedding_dim", 16),
    ],
)
def test_a_knob_the_backbone_cannot_read_is_refused_by_name(backbone, knob, value):
    """The inert-parameter bug class, closed at the config.

    A union config that ignores half its knobs would train a 16-wide GRU while
    reporting ``d_model=256`` - no error, no output difference, and the run's
    recorded configuration a lie. The refusal must name the knob AND the body it
    belongs to, because "which of these did I mean" is the only thing the caller
    can act on.
    """
    with pytest.raises(ValueError) as excinfo:
        NNPaidCaseConfig(backbone=backbone, **{knob: value})
    message = str(excinfo.value)
    assert knob in message
    assert backbone in message


def test_a_foreign_knob_left_at_its_default_is_accepted():
    """The guard on the guard, and the reason the check is "moved away from the
    default" rather than "belongs to the other body": ``NNPaidCaseConfig()``
    itself carries every knob at its default, so a stricter rule would make the
    default config unbuildable."""
    default = NNPaidCaseConfig()
    for backbone, knobs in BACKBONE_KNOBS.items():
        other = {k: getattr(default, k) for k in knobs}
        NNPaidCaseConfig(backbone=("gru" if backbone == "transformer" else "transformer"), **other)


def test_an_unknown_backbone_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown backbone"):
        NNPaidCaseConfig(backbone="lstm")


def test_both_backbones_are_reachable_and_declare_their_own_inputs():
    """``INPUT_KEYS`` is the contract between the entry and the two bodies, and
    the difference is the point: only the transformer takes a calendar cutoff,
    only the GRU takes a company index. A union signature with each ignoring
    half would be the same inert-parameter defect one layer down."""
    assert "cutoff" in PaidCaseTransformer.INPUT_KEYS
    assert "comp" not in PaidCaseTransformer.INPUT_KEYS
    assert "comp" in PaidCaseGRU.INPUT_KEYS
    assert "cutoff" not in PaidCaseGRU.INPUT_KEYS


# -- networks ------------------------------------------------------------------


def _build(backbone: str, *, n_features=2, n_w=5, n_d=4):
    cfg = tiny(backbone)
    if backbone == "transformer":
        return PaidCaseTransformer(cfg, n_lob=3, n_features=n_features, n_w=n_w, n_d=n_d)
    return PaidCaseGRU(cfg, n_lob=3, n_company=2, n_features=n_features, n_w=n_w, n_d=n_d)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_forward_shapes_and_a_valid_bivariate_mixture(backbone):
    """Every cell carries a normalized mixture over a 2-vector: weights sum to
    one, the Cholesky diagonal is strictly positive (so the density's triangular
    solve can never divide by zero), and the event axis is 2."""
    torch.manual_seed(0)
    model = _build(backbone)
    b, n_w, n_d, k = 3, 5, 4, tiny(backbone).n_components
    inputs = {
        "x": torch.randn(b, 2, n_w, n_d),
        "ctx": torch.rand(b, 2, n_w, n_d) > 0.5,
        "lob": torch.randint(0, 3, (b,)),
        "comp": torch.zeros(b, dtype=torch.long),
        "prem": torch.randn(b),
        "cutoff": torch.randint(1, 8, (b,)),
    }
    log_pi, mu, chol = call_backbone(model, inputs)
    assert log_pi.shape == (b, n_w, n_d, k)
    assert mu.shape == (b, n_w, n_d, k, 2)
    assert chol.shape == (b, n_w, n_d, k, 2, 2)
    assert (chol.diagonal(dim1=-2, dim2=-1) > 0).all()
    torch.testing.assert_close(
        log_pi.logsumexp(dim=-1), torch.zeros(b, n_w, n_d), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("backbone", BACKBONES)
def test_a_per_cell_mask_is_refused_by_name(backbone):
    """A caller on the pre-0.5.4 (B, W, D) mask must be told, not served: that
    mask BROADCASTS against x whenever the batch and channel counts line up
    (here B = F = 2), so every cohort would be masked with another channel's
    flags. For this entry the rank is doubly load-bearing - the rollout's whole
    job is to move the two channels' flags together at some cells and not at
    others."""
    b = n_f = 2
    model = _build(backbone, n_features=n_f)
    x = torch.randn(b, n_f, 5, 4)
    per_cell = torch.ones(b, 5, 4, dtype=torch.bool)
    assert (x * per_cell.to(x.dtype)).shape == x.shape  # the silent path this closes
    inputs = {
        "x": x,
        "ctx": per_cell,
        "lob": torch.zeros(b, dtype=torch.long),
        "comp": torch.zeros(b, dtype=torch.long),
        "prem": torch.zeros(b),
        "cutoff": torch.full((b,), 3, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="PER CHANNEL"):
        call_backbone(model, inputs)


# -- the movement target -------------------------------------------------------


def test_case_movement_differences_the_level_where_both_cells_are_present():
    """The derivation, on values small enough to check by hand.

    Four claims in one grid: dev 1's movement is the LEVEL (the case position
    was zero before the accident year opened), an interior movement is the
    difference, a cell whose own level is missing has no movement, and neither
    does the cell AFTER it - the successor's predecessor is gone. That last one
    is the whole reason the loss is mixed-observedness: one case hole costs two
    movement targets while both paid increments stay perfectly observable.
    """
    level = np.array([[[10.0, 8.0, 0.0, 5.0]]])
    level_obs = np.array([[[True, True, False, True]]])
    move, move_obs = case_movement(level, level_obs)
    np.testing.assert_allclose(move[0, 0], [10.0, -2.0, 0.0, 0.0])
    np.testing.assert_array_equal(move_obs[0, 0], [True, True, False, False])


def test_an_unusable_movement_is_zero_filled_padding_not_a_value():
    """The contract's own convention, one level up: an unusable cell is 0.0 AND
    masked, never a claim that the movement was zero. A consumer that gated on
    ``move != 0`` would read the hole in the test above as a real zero
    movement."""
    level = np.array([[[3.0, 0.0]]])
    level_obs = np.array([[[True, False]]])
    move, move_obs = case_movement(level, level_obs)
    assert move[0, 0, 1] == 0.0 and not move_obs[0, 0, 1]


def test_the_movement_target_gets_its_own_per_dev_statistics(fitted):
    """The fit standardizes the movement with the MOVEMENT's stats.

    Rebuilt independently from the contract and compared, then required to
    DIFFER from the level channel's own statistics - which is the mutation this
    exists for. A level decays across development while a movement is centred
    near zero and changes sign, so borrowing the level's mean and spread puts
    every target at a location the head has to undo; nothing about the loss, the
    shapes or the draws would say so.
    """
    entry = fitted.entry
    c = entry.contract_
    _, _, val_cutoff = splits(c["obs_mask"], c["cal_idx"], entry.config_.val_diagonals)
    gate = c["cal_idx"] <= val_cutoff
    move, move_obs = case_movement(c["x"][:, 1], c["x_obs"][:, 1])
    mean, std, pinned = norm_stats(move[:, None], move_obs & gate, move_obs)

    np.testing.assert_array_equal(entry.norm_["move_mean"], mean[0])
    np.testing.assert_array_equal(entry.norm_["move_std"], std[0])
    np.testing.assert_array_equal(entry.norm_["move_pinned"], pinned[0])
    assert not np.allclose(entry.norm_["move_mean"], entry.norm_["mean"][1]), (
        "the movement's per-dev means equal the LEVEL channel's on this fixture, so "
        "matching them proves nothing about which grid the fit standardized"
    )


# -- the entry contract --------------------------------------------------------


def test_registered_with_the_familys_declarations():
    cls = gallery.get("nn_paid_case")
    assert cls.family == "nn"
    assert cls.config_class is NNPaidCaseConfig
    assert cls.heldout_measure == "loss_ratio" and cls.heldout_measure in MEASURES
    assert cls.heldout_draw_scale == "incremental"
    assert issubclass(cls, ScoresHeldout) and issubclass(cls, PredictsHeldout)


def test_fit_predict_contract(fitted):
    """The GalleryEntry contract for one segment: per-origin ultimates plus a
    total, finite draws, a scorable summary against realized ultimates, and a
    fully developed first origin anchored exactly at its observed cumulative."""
    entry = fitted.entry
    pred = entry.predict(segment=SEG0, seed=0)
    assert pred.n_targets == N + 1  # per origin + total
    assert pred.targets["label"].tolist()[-1] == "total"
    assert np.isfinite(pred.samples).all()
    assert (pred.samples > 0).mean() > 0.9

    realized = entry.realized_ultimates(fitted.triangle, segment=SEG0)
    assert realized.shape == (N + 1,)
    table = pred.summary(observed=realized)
    assert {"estimate", "se", "cv", "outcome", "percentile"} <= set(table.columns)

    first = entry.contract_["latest_cum"][entry.cohort_index(SEG0), 0]
    np.testing.assert_allclose(pred.samples[:, 0], first)


def test_cohorts_and_cohort_index(fitted):
    """The 0.5.0 cohort vocabulary: ``cohorts()`` in predict's target order, a
    segment as a FILTER on them, and a typo refused rather than answered with
    the other cohort's plausible numbers."""
    entry = fitted.entry
    cohorts = entry.cohorts()
    assert cohorts == [SEG0, SEG1]
    assert entry.cohort_index(SEG1) == 1
    assert entry.cohort_index(None) is None
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.predict(segment={"line_of_business": "lob_9"})


def test_seeded_determinism_and_the_shared_rollout_cache(fitted):
    """Determinism: a fresh fit with the same seed reproduces the draws exactly -
    member seeds, batch order and the joint sampling all derive from it.
    Caching: the rollout is computed for ALL cohorts at once and reused across
    segments (identity check), and the case path comes off the same simulation.
    """
    entry = fitted.entry
    a = entry.predict(segment=SEG0, seed=7)
    cached = entry._rollout_ults
    b = entry.predict(segment=SEG1, seed=7)
    assert entry._rollout_ults is cached  # one rollout, sliced per cohort
    assert entry.case_paths(seed=7, per_diagonal=True) is entry._rollout_case
    # the terminal read is a VIEW of the cached path's last step, not a copy
    assert np.shares_memory(entry.case_paths(seed=7), entry._rollout_case)
    assert not np.allclose(a.samples, b.samples)

    again = NNPaidCase().fit(fitted.triangle, as_of=AS_OF, config=fitted.config, seed=0)
    np.testing.assert_allclose(a.samples, again.predict(segment=SEG0, seed=7).samples)


def test_fit_refuses_a_triangle_without_the_case_field(fitted):
    """The 0.5.4 absent-field refusal, surfaced through THIS fit's signature.

    ``select_fields`` is a filter, so a field the triangle does not carry yields
    no rows rather than an error - and the fit would otherwise run with a case
    channel that is masked everywhere, conditioning nothing and predicting a
    movement target that does not exist, which is indistinguishable from a case
    reserve that happened to be flat.
    """
    with pytest.raises(ValueError, match="case_reserve"):
        NNPaidCase().fit(paid_only_triangle(), as_of=AS_OF, config=fitted.config, seed=0)


# -- the rollout ---------------------------------------------------------------


def _spy_forward(entry, monkeypatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """Record ``(x, context_mask)`` at every forward pass of this fit's body."""
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    network = type(entry.models_[0])
    real_forward = network.forward

    def spy(self, x, context_mask, *rest, **kwargs):
        seen.append((x.detach().cpu().numpy().copy(), context_mask.cpu().numpy().copy()))
        return real_forward(self, x, context_mask, *rest, **kwargs)

    monkeypatch.setattr(network, "forward", spy)
    return seen


def test_the_rollout_promotes_both_channels_at_every_sampled_cell(fitted, monkeypatch):
    """The mirror of the other four NN entries' stay-off test.

    They promote channel 0's flag ALONE, because they simulated nothing else and
    raising a feature flag would present the contract's padding zero as an
    observed value. This entry SAMPLES both coordinates at every future cell, so
    both flags must rise - and rise on exactly the same cells, since a level
    written without its flag is a value the network will never read.

    Recorded through ``predict()``, the public path. The last two assertions are
    what keep it from passing vacuously: promotion must actually have happened,
    and the channel-1 flags must genuinely GROW beyond the contract's own.
    """
    entry = fitted.entry
    c = entry.contract_
    n_c = c["x"].shape[0]
    future = np.arange(c["n_d"])[None, None, :] >= c["latest_dev"][:, :, None]

    seen = _spy_forward(entry, monkeypatch)
    _drop_rollout_cache(entry)
    try:
        entry.predict(segment=SEG0, seed=0)
    finally:
        _drop_rollout_cache(entry)
    masks = [m for _, m in seen]
    assert masks, "predict() made no forward pass; the rollout never ran"

    grew = False
    for mask in masks:
        chunk = mask.shape[0] // n_c
        base = np.repeat(c["x_obs"], chunk, axis=0)
        assert not (base & ~mask).any(), "the rollout dropped an observed cell"
        promoted = mask & ~base  # (rows, F, W, D)
        assert not (promoted & ~np.repeat(future, chunk, axis=0)[:, None]).any(), (
            "a promoted cell sits outside the future region"
        )
        np.testing.assert_array_equal(
            promoted[:, 0],
            promoted[:, 1],
            err_msg=(
                "the two channels' promoted cells differ - the rollout samples a paid "
                "increment AND a case movement at every future cell, so a cell promoted "
                "on one channel and not the other is a value the network cannot read "
                "(or padding it will)"
            ),
        )
        grew |= bool(promoted[:, 1].any())
    assert grew, "channel 1's flags never grew, so the claim above is vacuous"


def test_the_rollout_starts_from_the_last_observed_case_level(fitted):
    """Where the state walk BEGINS, checked independently of the walk itself.

    ``test_the_rollout_integrates_the_movement_into_the_case_level`` rebuilds the
    trajectory from ``_initial_case_level`` and so follows it wherever it goes -
    a start of 0.0 everywhere would satisfy every assertion there, while making
    each projected level wrong by the whole outstanding position (a $2M case
    reserve that the projection never sees). So the start is rebuilt here by a
    plain loop over the contract, and the vectorized "last True from the right"
    it is compared against is deliberately a different piece of code.

    "Strictly before the origin's first projected cell" is the rule, and it ties
    the state to the rollout's own notion of the past: a case cell booked on a
    deeper diagonal than the paid anchor must not count both as the starting
    level and as a cell the rollout simulates over.
    """
    entry = fitted.entry
    c = entry.contract_
    n_c, _, n_w, n_d = c["x"].shape
    got = entry._initial_case_level()
    assert got.shape == (n_c, n_w)

    expected = np.zeros((n_c, n_w))
    for ci in range(n_c):
        for w in range(n_w):
            devs = [
                d for d in range(n_d) if c["x_obs"][ci, 1, w, d] and d < int(c["latest_dev"][ci, w])
            ]
            expected[ci, w] = c["x"][ci, 1, w, devs[-1]] if devs else 0.0
    np.testing.assert_allclose(got, expected)
    assert (got > 0).any(), (
        "no origin starts from a positive case level on this fixture, so a start of "
        "zero everywhere would pass"
    )


def test_the_rollout_integrates_the_movement_into_the_case_level(fitted, monkeypatch):
    """The state update, cell by cell, against a numpy rebuild.

    With the joint sampler pinned to a constant, every future cell's movement is
    known, so the whole channel-1 trajectory can be rebuilt: start from the
    cohort's last observed case level, add the movement IN RATIO SPACE at each
    projected diagonal, re-standardize with the LEVEL channel's own per-dev
    statistics. The rebuild is compared against what the network was actually
    handed on the NEXT forward pass - the only place the update is observable -
    and the terminal state against ``case_paths()``.

    The mutation this exists for is adding standardized values
    (``z_level[d] + z_move[d]``), which is not the standardized new level
    because each dev has its own mean and spread. It produces smooth, finite,
    plausibly-sized case levels and plausibly-sized ultimates. Using the
    MOVEMENT's statistics to re-standardize the level, or the level's to
    un-standardize the movement, fails here for the same reason.
    """
    entry = fitted.entry
    c = entry.contract_
    norm = entry.norm_
    n_c, _, n_w, n_d = c["x"].shape
    paid_z, move_z = 0.3, -0.7

    def constant_sample(log_pi, mu, chol, generator=None):
        shape = log_pi.shape[:-1]
        return torch.stack(
            [
                torch.full(shape, paid_z, dtype=mu.dtype, device=mu.device),
                torch.full(shape, move_z, dtype=mu.dtype, device=mu.device),
            ],
            dim=-1,
        )

    monkeypatch.setattr(head, "sample_joint", constant_sample)
    seen = _spy_forward(entry, monkeypatch)

    _drop_rollout_cache(entry)
    try:
        entry.predict(segment=SEG0, seed=0)
        terminal = entry.case_paths(seed=0)
    finally:
        _drop_rollout_cache(entry)

    # the reference walk, in numpy, on the contract's own numbers
    x_norm = (c["x"] - norm["mean"][None, :, None, :]) / norm["std"][None, :, None, :]
    x_norm = np.where(norm["pinned"][None, :, None, :], 0.0, x_norm)
    level = entry._initial_case_level().copy()  # (n_c, n_w) ratio
    future = np.arange(n_d)[None, None, :] >= c["latest_dev"][:, :, None]
    cal = c["cal_idx"]
    # a pinned dev's sampled z is forced to 0, so its movement is the pooled mean
    move_ratio = np.where(norm["move_pinned"], 0.0, move_z) * norm["move_std"] + norm["move_mean"]
    grids = [x_norm[:, 1].copy()]  # channel 1 as the k-th forward pass sees it
    for lv in sorted(np.unique(cal[future.any(axis=0)])):
        cells = future & (cal == lv)  # (n_c, n_w, n_d)
        if not cells.any():
            continue
        stepped = level + (move_ratio[None, None, :] * cells).sum(axis=2)
        level = np.where(cells.any(axis=2), stepped, level)
        level_z = (stepped[:, :, None] - norm["mean"][1]) / norm["std"][1]
        level_z = np.where(norm["pinned"][1], 0.0, level_z)
        grid = grids[-1].copy()
        grid[cells] = level_z[cells]
        grids.append(grid)

    n_steps = len(grids) - 1
    assert n_steps >= 2, "fewer than two projected diagonals; the walk tests nothing"
    n_members = len(entry.models_)
    assert len(seen) == n_members * n_steps, (
        f"expected one forward pass per projected diagonal per member "
        f"({n_members} x {n_steps}), recorded {len(seen)}"
    )
    for k, (x, _) in enumerate(seen):
        chunk = x.shape[0] // n_c
        got = x[:, 1].reshape(n_c, chunk, n_w, n_d)
        expected = np.repeat(grids[k % n_steps][:, None], chunk, axis=1)
        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"channel 1 at forward pass {k} is not level + movement in ratio space",
        )

    # the terminal state, after the last write (which no forward pass sees)
    assert terminal.shape == (fitted.config.n_draws, n_c, n_w)
    np.testing.assert_allclose(
        terminal, np.repeat(level[None], fitted.config.n_draws, axis=0), rtol=1e-4, atol=1e-5
    )


def test_case_paths_is_a_diagnostic_over_the_same_draws(fitted):
    """Shape, cohort order and provenance: ``(n_draws, n_c, n_w)`` off the SAME
    cached rollout ``predict`` used, so the two describe one simulation rather
    than two. It is deliberately not a PredictiveDistribution - there is no
    realized-case column to score it against (card.md "Limitations")."""
    entry = fitted.entry
    n_c, n_w = len(entry.cohorts()), entry.contract_["n_w"]
    paths = entry.case_paths(seed=3)
    assert paths.shape == (fitted.config.n_draws, n_c, n_w)
    assert np.isfinite(paths).all()
    entry.predict(segment=SEG0, seed=3)
    full = entry.case_paths(seed=3, per_diagonal=True)
    assert full is entry._rollout_case  # one simulation, two read-outs
    # the walk itself: one step per future calendar diagonal, ascending; the
    # last step IS the terminal read, and the walk genuinely walks - a path
    # that repeated its endpoint L times would pass every shape check here
    assert full.ndim == 4 and full.shape[0] == fitted.config.n_draws
    assert full.shape[2:] == (n_c, n_w)
    np.testing.assert_array_equal(full[:, -1], paths)
    if full.shape[1] > 1:
        assert not np.array_equal(full[:, 0], full[:, -1]), (
            "every step of the case path equals its endpoint - the per-diagonal "
            "walk is not being recorded, only the terminal state repeated"
        )


# -- held-out wiring -----------------------------------------------------------


def test_forward_mixture_is_the_heads_paid_margin(fitted):
    """``_forward_mixture`` returns ``head.paid_margin`` of the raw output, not a
    re-derivation of it.

    Compared elementwise against the margin of the same forward pass, exactly
    (the members are in eval mode and dropout is 0), so a hand-rolled
    "``mu[..., 0]``, ``sqrt(cov_00)``" that happened to be right today could not
    drift tomorrow. The shapes are asserted too: the margin's mean must have LOST
    the event axis, which is what tells the mixin it is scoring a univariate
    mixture rather than silently indexing a bivariate one.
    """
    entry = fitted.entry
    model = entry.models_[0]
    inputs = entry._heldout_inputs(0)
    with torch.no_grad():
        raw = call_backbone(model, inputs)
        got = entry._forward_mixture(model, inputs)
    expected = head.paid_margin(*raw)
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
    log_pi, mu, sigma = got
    assert log_pi.shape == mu.shape == sigma.shape == raw[0].shape
    assert mu.shape == raw[1].shape[:-1]  # the event axis is gone


def test_heldout_inputs_carry_the_calendar_boundary_only_where_it_is_read(fitted):
    """The transformer arm places an as_of cutoff; the GRU arm carries none.

    Relative calendar position is structural in the recurrence, so a ``cutoff``
    key on that arm would be an input nothing reads - and
    ``tests/test_nn_heldout_cutoff.py`` would grade it as live. Both arms
    condition per channel, on the contract's own ``x_obs``.
    """
    entry = fitted.entry
    c = entry.contract_
    inputs = entry._heldout_inputs(0)
    assert set(inputs) == set(type(entry.models_[0]).INPUT_KEYS)
    np.testing.assert_array_equal(inputs["ctx"].cpu().numpy()[0], c["x_obs"][0])
    if fitted.backbone == "transformer":
        assert int(inputs["cutoff"].item()) == int(c["cal_idx"][c["obs_mask"][0]].max())
    else:
        assert not [k for k in inputs if "cut" in k.lower() or "cal" in k.lower()]


def test_the_two_draw_budgets_are_delivered_independently(fitted):
    """One config, two knobs, two paths: ``predict_at`` spends
    ``heldout_n_draws`` and ``predict`` spends ``n_draws``.

    Both halves from the same fit, because a fix that simply renamed the field
    would move both counts and pass either half alone. This is the entry's whole
    claim to a board row: a held-out diagonal costs one forward pass per member
    whatever the draw count, so 10,000 is affordable here where a 10,000-draw
    rollout is not.
    """
    entry = fitted.entry
    cfg = fitted.config
    assert cfg.heldout_n_draws != cfg.n_draws  # or either could stand in
    draws = entry.predict_at(fitted.cells, field="paid_loss", seed=3)
    assert draws.shape == (cfg.heldout_n_draws, fitted.cells.n_cells)
    assert np.isfinite(draws).all()
    pred = entry.predict(segment=SEG0, n_draws=None, seed=3)
    assert pred.samples.shape[0] == cfg.n_draws


def test_the_paid_margin_scores_and_draws_on_the_board(fitted):
    """End to end through the shared mixin, on the cells ``next_diagonal`` builds.

    The density is refused at pinned devs (no trained head) and delivered
    elsewhere, one row per ensemble member; the draws survive everywhere and are
    anchored onto each cell's training predecessor by the base class, so they
    sit on the triangle's cumulative basis rather than the entry's incremental
    one.
    """
    entry = fitted.entry
    cells = fitted.cells
    pinned_devs = np.nonzero(entry.norm_["pinned"][0])[0] + 1
    live_mask = ~cells.frame["dev_lag"].isin([12 * int(d) for d in pinned_devs])
    live = replace(cells, frame=cells.frame[live_mask].reset_index(drop=True))
    assert live.n_cells, "every held-out cell is pinned; the density leg tests nothing"

    ll = entry.log_lik_at(live, field="paid_loss")
    assert ll.shape == (len(entry.models_), live.n_cells)
    assert np.isfinite(ll).all()
    # identically-seeded members would make logmeanexp a plug-in in disguise
    assert not np.allclose(ll[0], ll[1])

    draws = entry.predict_at(cells, field="paid_loss", seed=5)
    anchors = cells.frame["prev_value"].to_numpy()
    assert (np.abs(draws.mean(axis=0) - anchors) < 10.0 * anchors).all()
