"""The refit and the alpha split of ``tlrn``, on both training backends.

The two options are the companion study's newest protocol changes: ``refit`` retrains each
member on every known diagonal to the epoch its first run chose, and
``alpha_on="first_diagonal"`` fits the blend weight on one validation diagonal and picks the
checkpoint on the next. Skips cleanly without torch, and the jax variants without jax.

A parameter that is accepted and then ignored is the bug this repository keeps meeting, so
every test below reaches the option through ``gallery.fit`` and compares something the option
decides with an independent computation, and the defaults are pinned to the values the
previous release gave (taken by running that release, not by reading it).

1. the defaults: a pinned default fit, and the config's new fields off;
2. ``alpha_on``: which cells the weight is fitted on and which cells score the checkpoint;
3. ``refit``: the weights kept are the weights at the member's own epoch, from the same
   seed, on the wider sets, with the same alpha, and the paired no-refit forecast is the one
   ``refit=False`` produces;
4. the preset: its fields and its parameter count.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr import gallery  # noqa: E402
from ibnr.gallery.nn._training import train_ensemble, warmup_cosine  # noqa: E402
from ibnr.gallery.nn.tlrn import model as tlrn_model  # noqa: E402
from ibnr.gallery.nn.tlrn.blend import blend_weight  # noqa: E402
from ibnr.gallery.nn.tlrn.config import TLRNConfig  # noqa: E402
from ibnr.gallery.nn.tlrn.model import (  # noqa: E402
    TLRN,
    _epoch_probe,
    _refuse_look_ahead,
    _stored_epochs,
)
from ibnr.gallery.nn.tlrn.network import TLRNNetwork  # noqa: E402
from ibnr.kernels.nn_features import pooled_incremental_lr  # noqa: E402

from .test_nn_features import study_arrays, study_triangle  # noqa: E402
from .test_tlrn import AS_OF, tiny  # noqa: E402

HAS_JAX = importlib.util.find_spec("jax") is not None
BACKENDS = [
    "torch",
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="needs the [jax] extra")),
]
#: the fixture for the accident-year variant: four companies, eight accident years
AY_AS_OF = "2007-12-31"


def blend_config(**overrides) -> TLRNConfig:
    """Two members of the blend, all kept, three validation checks of a six-epoch run."""
    base = {
        "member": "mcl_blend",
        "keep": None,
        "max_epochs": 6,
        "check_every": 2,
        "patience": 10,
    }
    return replace(tiny(), **{**base, **overrides})


def ay_config(**overrides) -> TLRNConfig:
    """The accident-year variant at test size, retrained at three earlier valuation dates."""
    return TLRNConfig.accident_year_variant(
        **{
            "d_model": 8,
            "n_heads": 2,
            "dropout": 0.0,
            "max_epochs": 4,
            "min_epochs": 0,
            "check_every": 2,
            "patience": 5,
            "batch_size": 3,
            "ensemble_size": 2,
            "calibration_cutoffs": (5, 6, 7),
            "calibration_horizons": (1, 2, 3),
            "n_strata": 1,
            "min_per_stratum": 3,
            "n_draws": 50,
            "warmup": 1,
            **overrides,
        }
    )


def fit(config, *, backend="torch", seed=5, ay=False, **kwargs):
    arrays = study_arrays(n_companies=4, n_w=8, seed=3) if ay else study_arrays()
    return gallery.fit(
        "tlrn",
        study_triangle("duckdb", arrays),
        loss_field="paid_loss",
        as_of=AY_AS_OF if ay else AS_OF,
        config=config,
        seed=seed,
        backend=backend,
        **kwargs,
    )


# -- 1. the defaults ---------------------------------------------------------------------


def test_the_new_fields_are_off_by_default():
    cfg = TLRNConfig()
    assert cfg.refit is False and cfg.alpha_on == "all_validation"
    # the accident-year variant does not turn them on: it is the variant, not the final model
    ay = TLRNConfig.accident_year_variant()
    assert ay.refit is False and ay.alpha_on == "all_validation"


#: what the previous release (0.7.5) gave for these fits: the first six entries of
#: ``member_company_reserves()`` and the epochs each member chose. A default fit must not move.
DEFAULT_PINS = {
    "default": (
        3398.992320522666,
        6025.12339130044,
        10322.520654648542,
        3405.0056375563145,
        6039.983347058296,
        10340.428121387959,
    ),
    "blend": (2989.5534, 5480.5, 9012.141, 2948.2125, 5428.0078, 8879.3451),
}


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_default_fit_is_the_previous_releases_fit(backend):
    """Both backends, both members: same reserves to float32 rounding, same chosen epochs."""
    plain = fit(tiny(), backend=backend)
    blend = fit(replace(tiny(), member="mcl_blend", keep=None), backend=backend)
    for entry, key in ((plain, "default"), (blend, "blend")):
        got = entry.member_company_reserves().ravel()[:6]
        np.testing.assert_allclose(got, DEFAULT_PINS[key], rtol=2e-5)
        assert entry.selection_["best_epoch"].tolist() == [2, 2]
        assert "refit_epochs" not in entry.selection_.columns
        assert entry.member_reserves_selected_ is None
    # the same options spelled out change nothing
    spelled = fit(replace(tiny(), refit=False, alpha_on="all_validation"), backend=backend)
    np.testing.assert_array_equal(
        spelled.member_company_reserves(), plain.member_company_reserves()
    )


@pytest.mark.skipif(not HAS_JAX, reason="needs the [jax] extra")
def test_the_accident_year_variant_is_unchanged_on_both_backends():
    """The premium head with accident-year attention, retrained at three earlier dates."""
    want = [5055.1313, 10177.7586, 15076.8214, 20515.7601, 5021.5102, 10079.399]
    for backend in ("torch", "jax"):
        entry = fit(ay_config(max_epochs=5, check_every=2), backend=backend, ay=True)
        np.testing.assert_allclose(entry.member_company_reserves().ravel()[:6], want, rtol=2e-5)
        assert entry.selection_["best_epoch"].tolist() == [2, 4]


def test_the_config_refuses_what_cannot_work():
    with pytest.raises(ValueError, match="alpha_on must be one of"):
        TLRNConfig(alpha_on="second_diagonal")
    with pytest.raises(ValueError, match="has no alpha"):
        TLRNConfig(alpha_on="first_diagonal")  # member="network"
    with pytest.raises(ValueError, match="exactly val_diagonals=2"):
        TLRNConfig(alpha_on="first_diagonal", member="mcl_blend", val_diagonals=3)
    with pytest.raises(ValueError, match="refit must be True or False"):
        TLRNConfig(refit="yes")
    # a legal one, for contrast
    assert TLRNConfig(alpha_on="first_diagonal", member="mcl_blend").alpha_on == "first_diagonal"


# -- 2. alpha_on --------------------------------------------------------------------------


def run_log(monkeypatch):
    """Record the ``(prep, models)`` of every protocol ``fit`` finishes."""
    log = []
    real = TLRN._finish_protocol

    def spy(self, prep, cfg, models, history):
        log.append((prep, models))
        return real(self, prep, cfg, models, history)

    monkeypatch.setattr(TLRN, "_finish_protocol", spy)
    return log


def by_group(cells, tset, n_l, n_d):
    weight = tset["target_mask"] * tset["premium"]
    return (cells * weight).reshape(-1, n_l, n_d).sum(2).cpu().numpy()


def independent_blend(prep, model, n_l, n_d):
    """Per validation set: the error of the chain ladder, the network's gap to it, the actual."""
    out = []
    with torch.no_grad():
        for tset in prep["val_tensors"]:
            net = prep["forward"](model, tset, blend=False)["pred"]
            out.append(
                (
                    by_group(tset["mcl_pred"] - tset["target"], tset, n_l, n_d),
                    by_group(net - tset["mcl_pred"], tset, n_l, n_d),
                    by_group(tset["target"], tset, n_l, n_d),
                )
            )
    return out


@pytest.mark.parametrize("backend", BACKENDS)
def test_alpha_is_fitted_on_the_first_diagonal_and_the_checkpoint_scored_on_the_second(
    backend, monkeypatch
):
    log = run_log(monkeypatch)
    entry = fit(
        blend_config(alpha_on="first_diagonal", lr=3e-2, lr_phi=1e-1), backend=backend, seed=6
    )
    ((prep, models),) = log
    c_max = entry.feature_stats_["final_cutoff"]
    n_l, n_d = entry.contract_["x"].shape[1], entry.contract_["x"].shape[4]

    # the first validation set stops at its own next diagonal, the second is the last one
    assert [s["target_hi"] for s in prep["val_sets"]] == [c_max - 1, c_max]
    default_log = run_log(monkeypatch)
    fit(blend_config(), backend=backend)
    assert [s["target_hi"] for s in default_log[0][0]["val_sets"]] == [c_max, c_max]

    tol = 1e-6 if backend == "torch" else 2e-3
    differs = []
    for member, model in enumerate(models):
        first, last = (independent_blend(prep, model, n_l, n_d)[i] for i in (0, -1))
        alpha = blend_weight(first[0], first[1])  # the first set alone
        assert float(model.alpha) == pytest.approx(alpha, abs=tol)
        score = np.abs(last[0] + alpha * last[1]).sum() / np.abs(last[2]).sum()
        row = entry.selection_.loc[member]
        assert row["validation_ay_line_ape"] == pytest.approx(score, rel=tol * 10)
        assert row["alpha"] == pytest.approx(alpha, abs=tol)
        both = blend_weight(
            np.concatenate([first[0], last[0]]), np.concatenate([first[1], last[1]])
        )
        differs.append(abs(both - alpha) > 1e-3)
    assert any(differs)  # fitting on both diagonals would have said something else


@pytest.mark.parametrize("backend", BACKENDS)
def test_alpha_on_changes_the_fit_and_reaches_the_selection(backend):
    default = fit(blend_config(), backend=backend)
    first = fit(blend_config(alpha_on="first_diagonal"), backend=backend)
    assert not np.allclose(
        default.selection_["validation_ay_line_ape"], first.selection_["validation_ay_line_ape"]
    )
    assert not np.allclose(default.member_company_reserves(), first.member_company_reserves())


def test_the_blend_score_unit():
    """``_blend_ay_line_ape`` on hand-made sets: alpha from set 0, the score on the last set."""
    n_l, n_d = 1, 3

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("alpha", torch.ones(()))

    def tset(target, mcl, net):
        def row(x):
            return torch.tensor([x], dtype=torch.float32)

        return {
            "target": row(target),
            "mcl_pred": row(mcl),
            "net": row(net),
            "target_mask": torch.ones(1, 3),
            "premium": torch.ones(1, 3),
        }

    sets = [
        tset([1.0] * 3, [1.5] * 3, [1.0] * 3),  # the chain ladder is 1.5 over, the network exact
        tset([2.0] * 3, [1.0] * 3, [3.0] * 3),  # the chain ladder 3 under, the network 3 over
    ]

    def forward(model, t, blend=False):
        return {"pred": t["net"]}

    # per (accident year, line): set 0 has error +1.5 and gap -1.5, set 1 has -3 and +6
    both, first = Holder(), Holder()
    score_both = TLRN._blend_ay_line_ape(both, sets, forward, n_l, n_d)
    score_first = TLRN._blend_ay_line_ape(first, sets, forward, n_l, n_d, alpha_first=True)
    assert float(both.alpha) == pytest.approx(0.5)  # |1.5 - 1.5a| + |-3 + 6a| is least at a = 1/2
    assert score_both == pytest.approx(0.75 / 9.0)
    assert float(first.alpha) == pytest.approx(1.0)  # set 0 alone: the network is exact
    assert score_first == pytest.approx(abs(-3.0 + 6.0) / 6.0)  # scored on set 1 alone


# -- 3. refit ------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_no_refit_forecast_is_what_a_fit_without_the_refit_gives(backend):
    """One training run gives both arms, and the paired arm is the refit=False fit."""
    plain = fit(blend_config(), backend=backend)
    again = fit(blend_config(refit=True), backend=backend)
    rtol = 1e-12 if backend == "torch" else 1e-6
    np.testing.assert_allclose(
        again.member_company_reserves(selected=True), plain.member_company_reserves(), rtol=rtol
    )
    np.testing.assert_allclose(
        again.company_reserves(selected=True), plain.company_reserves(), rtol=1e-6
    )
    # the same selection, alpha included, plus the epoch the refit ran to
    columns = list(plain.selection_.columns)
    np.testing.assert_allclose(
        again.selection_[columns].select_dtypes("number").to_numpy(dtype=float),
        plain.selection_[columns].select_dtypes("number").to_numpy(dtype=float),
        rtol=1e-6,
    )
    assert again.selection_["refit_epochs"].tolist() == again.selection_["best_epoch"].tolist()
    # and the refit networks forecast something else
    assert not np.allclose(again.member_company_reserves(), plain.member_company_reserves())
    assert not np.allclose(again.company_reserves(), again.company_reserves(selected=True))
    assert again.member_company_reserves().shape == plain.member_company_reserves().shape
    assert again.feature_stats_["refit_cutoffs"] == tuple(
        range(2, again.feature_stats_["final_cutoff"])
    )


def test_selected_needs_a_refit():
    plain = fit(blend_config())
    with pytest.raises(ValueError, match="has no refit"):
        plain.member_company_reserves(selected=True)
    with pytest.raises(ValueError, match="has no refit"):
        plain.company_reserves(selected=True)


def capture_preps(monkeypatch):
    """Every ``(is_refit, prep)`` ``_prepare_protocol`` builds."""
    made = []
    real = TLRN._prepare_protocol

    def spy(self, *args, **kwargs):
        prep = real(self, *args, **kwargs)
        made.append((kwargs.get("refit_of") is not None, prep))
        return prep

    monkeypatch.setattr(TLRN, "_prepare_protocol", spy)
    return made


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_refit_trains_on_everything_known_and_starts_where_the_cutoff_says(
    backend, monkeypatch
):
    made = capture_preps(monkeypatch)
    entry = fit(blend_config(head="premium_lr", refit=True), backend=backend)
    (_, first), (is_refit, second) = made
    assert is_refit
    c_max = entry.feature_stats_["final_cutoff"]
    train_end = c_max - 2
    # the first run's targets stop where validation starts; the refit's reach the last diagonal
    assert {s["target_hi"] for s in first["train_sets"].values()} == {train_end}
    assert {s["target_hi"] for s in second["train_sets"].values()} == {c_max}
    assert list(second["train_sets"]) == list(range(2, c_max))
    assert second["val_sets"] == [] and second["train_end"] == c_max
    # the validation cells are now training targets
    seen = sum(s["target_mask"].sum() for s in second["train_sets"].values())
    held = sum(s["target_mask"].sum() for s in first["train_sets"].values())
    assert seen > held
    # the premium head starts at the pooled loss ratios known at each run's own last diagonal
    contract = entry.contract_
    np.testing.assert_allclose(
        first["make_model"]().beta.detach().numpy(),
        np.log(pooled_incremental_lr(contract, train_end)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        second["make_model"]().beta.detach().numpy(),
        np.log(pooled_incremental_lr(contract, c_max)),
        rtol=1e-6,
    )
    # the factor support is read off the wider sets, and is never narrower
    assert (second["support_np"] >= first["support_np"]).all()
    np.testing.assert_array_equal(entry.factor_support_, second["support_np"])
    np.testing.assert_array_equal(entry.selected_factor_support_, first["support_np"])


def test_the_refit_keeps_the_weights_at_the_epoch_the_first_run_chose(monkeypatch):
    """Against a hand-rolled run: train exactly that many epochs and take the end state.

    The hand-rolled run does not use the refit's stopping rule. It asks ``train_ensemble`` for
    ``stop`` epochs on the schedule of the full budget, with a score that falls at every check,
    so the best checkpoint is the last one.
    """
    made = capture_preps(monkeypatch)
    cfg = blend_config(refit=True, dropout=0.3)
    entry = fit(cfg, seed=6)
    second = next(prep for is_refit, prep in made if is_refit)
    stops = entry.selection_["refit_epochs"].tolist()
    assert len(set(stops)) > 1 or stops[0] < cfg.max_epochs  # a stop short of the budget
    for member, stop in enumerate(stops):
        calls = [0]

        def falling(_model, calls=calls):
            calls[0] += 1
            return -float(calls[0])

        (hand,), _ = train_ensemble(
            second["n_units"],
            config=replace(cfg, max_epochs=stop, patience=1000, min_epochs=0),
            seed=6,
            make_model=second["make_model"],
            train_loss=second["train_loss"],
            val_loss=falling,
            min_cutoff=cfg.min_cutoff,
            val_cutoff=second["train_end"],
            device=torch.device("cpu"),
            schedule=warmup_cosine(cfg.max_epochs, cfg.warmup),
            param_groups=second["param_groups"],
            min_epochs=0,
            check_every=cfg.check_every,
            cutoff_sampling=cfg.cutoff_sampling,
            members=[member],
        )
        mine = entry.models_[member].state_dict()
        for name, value in hand.state_dict().items():
            if name == "alpha":
                continue  # the refit keeps the first run's alpha, checked below
            torch.testing.assert_close(mine[name], value, rtol=0, atol=0)
        # alpha is the first run's, untouched by the refit
        selected = entry.selection_.loc[member, "alpha"]
        assert float(mine["alpha"]) == pytest.approx(selected, abs=1e-7)


def test_the_refit_stop_must_be_an_epoch_a_check_runs_after():
    assert _epoch_probe(4, 2, 7)(None) == 2.0  # the first check is epoch 2
    probe = _epoch_probe(7, 2, 7)  # 7 is the last epoch, so it is checked
    assert [probe(None) for _ in range(4)] == [5.0, 3.0, 1.0, 0.0]
    for stop in (0, 3, 8):
        with pytest.raises(ValueError, match="not one of the epochs"):
            _epoch_probe(stop, 2, 7)


def test_the_refit_runs_in_worker_processes_and_trains_the_same_members():
    one = fit(blend_config(refit=True), processes=1)
    two = fit(blend_config(refit=True), processes=2)
    np.testing.assert_allclose(
        two.member_company_reserves(), one.member_company_reserves(), rtol=1e-6
    )
    np.testing.assert_allclose(
        two.member_company_reserves(selected=True),
        one.member_company_reserves(selected=True),
        rtol=1e-6,
    )


@pytest.mark.skipif(not HAS_JAX, reason="needs the [jax] extra")
def test_both_backends_refit_to_the_same_networks():
    torch_fit = fit(blend_config(refit=True))
    jax_fit = fit(blend_config(refit=True), backend="jax")
    np.testing.assert_allclose(
        jax_fit.member_company_reserves(), torch_fit.member_company_reserves(), rtol=2e-3
    )
    np.testing.assert_allclose(
        jax_fit.member_company_reserves(selected=True),
        torch_fit.member_company_reserves(selected=True),
        rtol=2e-3,
    )


def record_stops(monkeypatch, backend):
    """The ``stops`` each protocol's refit is handed, in protocol order, on either backend."""
    seen = []
    if backend == "jax":
        from ibnr.gallery.nn.tlrn import jax_backend

        real = jax_backend.train_protocols

        def spy(preps, cfg, torch_, **kwargs):
            if kwargs.get("stops") is not None:
                seen.append([list(map(int, row)) for row in kwargs["stops"]])
            return real(preps, cfg, torch_, **kwargs)

        monkeypatch.setattr(jax_backend, "train_protocols", spy)
    else:
        real = TLRN._refit_in_process

        def spy(prep, cfg, stops, dev):
            seen.append(list(stops))
            return real(prep, cfg, stops, dev)

        monkeypatch.setattr(TLRN, "_refit_in_process", staticmethod(spy))
    return seen


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_refit_follows_every_protocol_of_a_retrained_calibration(backend, monkeypatch):
    """The method retrained at the earlier dates is the method that was fitted: refit too.

    Each protocol gets its own stops, member by member: protocol p is told to stop at
    ``[2, 4]`` when p is even and ``[4, 2]`` when odd, and the refit must be handed exactly
    those, and record them in that protocol's selection table.
    """
    made = capture_preps(monkeypatch)
    forced = {"n": 0}

    def stored(history, *, single_precision):
        i = forced["n"]
        forced["n"] += 1
        return [2, 4] if i % 2 == 0 else [4, 2]

    monkeypatch.setattr(tlrn_model, "_stored_epochs", stored)
    seen = record_stops(monkeypatch, backend)
    runs_seen = []
    real_run = TLRN._run_protocols

    def spy_runs(self, *args, **kwargs):
        runs = real_run(self, *args, **kwargs)
        runs_seen.extend(runs)
        return runs

    monkeypatch.setattr(TLRN, "_run_protocols", spy_runs)
    entry = fit(ay_config(refit=True, alpha_on="first_diagonal"), backend=backend, ay=True)
    assert [flag for flag, _ in made] == [False] * 4 + [True] * 4
    want = [[2, 4], [4, 2], [2, 4], [4, 2]]
    assert [r["selection"]["refit_epochs"].tolist() for r in runs_seen] == want
    # the refit is handed those stops (one list per protocol under torch, one array under jax)
    flat = seen[0] if backend == "jax" else seen
    assert flat == want
    assert entry.backtest_["predicted"].notna().all()
    assert (
        entry.member_company_reserves().shape == entry.member_company_reserves(selected=True).shape
    )
    assert not np.allclose(
        entry.member_company_reserves(), entry.member_company_reserves(selected=True)
    )


def test_keeping_fewer_members_still_refits_every_trained_one():
    entry = fit(blend_config(refit=True, keep=1, ensemble_size=3))
    assert entry.member_company_reserves().shape[0] == 3
    assert entry.member_company_reserves(selected=True).shape[0] == 3
    kept = entry.selection_.index[entry.selection_["kept"]].tolist()
    np.testing.assert_allclose(
        entry.company_reserves(), entry.member_company_reserves()[kept].mean(axis=0), rtol=1e-6
    )
    assert entry.selection_["refit_epochs"].notna().all()


def test_a_member_without_a_finite_score_cannot_be_refit(monkeypatch):
    real = TLRN._selection_table

    def broken(models, history, keep, company_ape):
        table = real(models, history, keep, company_ape)
        table["best_epoch"] = table["best_epoch"].astype(float)
        table.loc[0, "best_epoch"] = np.nan
        return table

    monkeypatch.setattr(TLRN, "_selection_table", staticmethod(broken))
    with pytest.raises(ValueError, match="never produced a finite validation score"):
        fit(blend_config(refit=True))


# -- 4. the preset --------------------------------------------------------------------------


def test_the_companion_final_variant_is_the_variant_plus_the_two_protocol_changes():
    cfg = TLRNConfig.companion_final_variant()
    assert (cfg.refit, cfg.alpha_on) == (True, "first_diagonal")
    # the model: premium loss ratio head, three attention axes on observed cells, a blend
    assert (cfg.head, cfg.attention, cfg.mask) == (
        "premium_lr",
        ("line", "lag", "ay"),
        "observed_cells",
    )
    assert (cfg.eps, cfg.lr_cap, cfg.tail_policy) == (0.5, 5.0, "observed_cl")
    assert (cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout) == (32, 2, 1, 0.3)
    assert cfg.member == "mcl_blend" and cfg.scoring == "reached_cells"
    # the training: eight whole companies a batch, 3000 epochs, the two learning rates
    assert (cfg.batch_unit, cfg.batch_size) == ("company", 8)
    assert (cfg.max_epochs, cfg.warmup, cfg.check_every) == (3000, 20, 5)
    assert (cfg.lr, cfg.lr_phi, cfg.weight_decay, cfg.grad_clip) == (3e-3, 3e-2, 0.0, 1.0)
    assert (cfg.w_pe, cfg.w_mse, cfg.mse_scale) == (0.5, 0.1, 0.005)
    assert (cfg.min_cutoff, cfg.val_diagonals, cfg.cutoff_sampling) == (2, 2, "per_epoch")
    # twenty members, all averaged; patience that never fires in 3000 epochs
    assert (cfg.ensemble_size, cfg.keep, cfg.patience) == (20, None, 600)
    assert cfg.patience * cfg.check_every >= cfg.max_epochs
    # the calibration: the method retrained at valuation dates 6, 7 and 8, scored 4, 3, 2 ahead
    assert cfg.calibration == "retrain_per_valuation"
    assert (cfg.calibration_cutoffs, cfg.calibration_horizons) == ((6, 7, 8), (2, 3, 4))
    # an override replaces one choice and nothing else
    assert TLRNConfig.companion_final_variant(refit=False).refit is False
    assert TLRNConfig.companion_final_variant(ensemble_size=3).alpha_on == "first_diagonal"


def test_the_companion_final_variant_has_the_companions_18537_parameters():
    """The companion reports 18,537, and the preset builds exactly that (its embedding tables
    have no unused rows). The accident-year variant, which is not the preset, keeps 18,601.
    Counted by building the networks; the layer-by-layer table is in
    ``test_tlrn_companion_network.py``."""
    cfg = TLRNConfig.companion_final_variant()
    model = TLRNNetwork(
        cfg, n_lines=4, n_lag=10, n_feat=13, n_origin=10, head_init=torch.zeros(4, 10)
    )
    assert sum(p.numel() for p in model.parameters()) == 18_537
    older = TLRNNetwork(
        TLRNConfig.accident_year_variant(),
        n_lines=4,
        n_lag=10,
        n_feat=13,
        n_origin=10,
        head_init=torch.zeros(4, 10),
    )
    assert sum(p.numel() for p in older.parameters()) == 18_601
    assert model.beta.shape == (4, 10)
    assert "blocks.0.attn_ay.in_proj_weight" in dict(model.named_parameters())


# -- 5. the stored epoch, per-member stops, and the no-look-ahead check --------------------------


def script_validation(monkeypatch, per_member):
    """Make the first run's validation scores a script: member ``m`` sees ``per_member[m]``."""
    real = TLRN._prepare_protocol

    def spy(self, *args, **kwargs):
        prep = real(self, *args, **kwargs)
        if kwargs.get("refit_of") is not None:
            return prep
        calls = [0]
        checks = len(per_member[0])

        def scripted(_model):
            m, i = divmod(calls[0], checks)
            calls[0] += 1
            return per_member[m][i]

        return {**prep, "val_loss": scripted}

    monkeypatch.setattr(TLRN, "_prepare_protocol", spy)


def test_the_stored_epochs_are_the_last_improvement_beyond_the_threshold():
    """The weights are stored at the last check that beats the best by more than 1e-6."""
    # argmin is the third check, by 5e-7: under the threshold, so the weights stay at check 2
    history = [
        [
            {"epoch": 1, "val": 1.0},
            {"epoch": 3, "val": 0.5},
            {"epoch": 5, "val": 0.5 - 5e-7},
        ],
        [{"epoch": 1, "val": 1.0}, {"epoch": 3, "val": 0.5}, {"epoch": 5, "val": 0.4}],
        [{"epoch": 1, "val": float("nan")}, {"epoch": 3, "val": float("inf")}],
    ]
    got = _stored_epochs(history, single_precision=False)
    assert got[:2] == [4, 6] and np.isnan(got[2])
    assert _stored_epochs(history, single_precision=True)[:2] == [4, 6]


def test_the_refit_stops_where_the_weights_were_stored_not_at_the_lowest_score(monkeypatch):
    """A synthetic sequence whose lowest score comes after the last stored improvement."""
    cfg = blend_config(refit=True)
    # three checks (epochs 2, 4, 6) per member; member 0's lowest is the last by 5e-7
    script_validation(monkeypatch, [[1.0, 0.5, 0.5 - 5e-7], [1.0, 0.6, 0.4]])
    entry = fit(cfg)
    table = entry.selection_
    assert table["best_epoch"].tolist() == [6, 6]  # the lowest score: unchanged semantics
    assert table["refit_epochs"].tolist() == [4, 6]  # where the weights were stored


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_refit_epochs_replay_the_training_loops_storing_rule(backend, monkeypatch):
    """On every backend ``refit_epochs`` is the last check that beat the best by over 1e-6.

    Recomputed here with a plain loop from the history the first run recorded.
    """
    histories = []
    real = TLRN._finish_protocol

    def spy(self, prep, cfg, models, history):
        histories.append(history)
        return real(self, prep, cfg, models, history)

    monkeypatch.setattr(TLRN, "_finish_protocol", spy)
    entry = fit(blend_config(refit=True), backend=backend)
    want = []
    for records in histories[0]:
        best, stored = np.inf, None
        for r in records:
            if np.isfinite(r["val"]) and np.float32(r["val"]) < np.float32(best) - 1e-6:
                best, stored = r["val"], r["epoch"] + 1
        want.append(stored)
    assert entry.selection_["refit_epochs"].tolist() == want


def forced_stops(monkeypatch, stops):
    monkeypatch.setattr(tlrn_model, "_stored_epochs", lambda history, *, single_precision: stops)


def test_members_with_different_stops_refit_the_same_on_every_backend(monkeypatch):
    forced_stops(monkeypatch, [2, 4])
    seen = record_stops(monkeypatch, "torch")
    one = fit(blend_config(refit=True), processes=1)
    assert seen == [[2, 4]] and one.selection_["refit_epochs"].tolist() == [2, 4]
    two = fit(blend_config(refit=True), processes=2)
    np.testing.assert_allclose(
        two.member_company_reserves(), one.member_company_reserves(), rtol=1e-6
    )
    # the stops are used member by member: [4, 4] gives a different first member
    forced_stops(monkeypatch, [4, 4])
    same = fit(blend_config(refit=True), processes=1)
    mine, other = one.member_company_reserves(), same.member_company_reserves()
    assert not np.allclose(mine[0], other[0])
    np.testing.assert_allclose(mine[1], other[1], rtol=1e-6)
    if HAS_JAX:
        forced_stops(monkeypatch, [2, 4])
        jax_fit = fit(blend_config(refit=True), backend="jax")
        assert jax_fit.selection_["refit_epochs"].tolist() == [2, 4]
        np.testing.assert_allclose(
            jax_fit.member_company_reserves(), one.member_company_reserves(), rtol=2e-3
        )


def test_the_look_ahead_check_names_a_target_scored_and_trained_alike():
    trained = [np.array([[1.0, 0.0, 0.0, 0.0]])]
    _refuse_look_ahead(trained, np.array([[0.0, 0.0, 1.0, 1.0]]))  # apart: nothing to say
    _refuse_look_ahead([], np.array([[1.0, 1.0, 1.0, 1.0]]))
    _refuse_look_ahead(trained, None)
    with pytest.raises(ValueError, match="look-ahead: 1 cell"):
        _refuse_look_ahead(trained, np.array([[1.0, 0.0, 1.0, 0.0]]))


@pytest.mark.parametrize("refit", [False, True])
def test_a_fit_whose_scored_cells_overlap_its_training_targets_is_refused(refit, monkeypatch):
    """Through ``gallery.fit``: the final feature set is made to score a training cell."""
    real = TLRN._prepare_protocol

    def spy(self, contract, cfg, **kwargs):
        features = kwargs["features"]

        def greedy(*args, **kw):
            out = features(*args, **kw)
            if args[0] == kwargs["c_max"]:  # the final set: score everything
                out = {**out, "target_mask": np.ones_like(out["target_mask"])}
            return out

        return real(self, contract, cfg, **{**kwargs, "features": greedy})

    monkeypatch.setattr(TLRN, "_prepare_protocol", spy)
    with pytest.raises(ValueError, match="look-ahead"):
        fit(blend_config(refit=refit))


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(dict(keep=2, ensemble_size=3), id="keep2"),
        pytest.param(dict(head="ldf", keep=2, ensemble_size=3), id="ldf-keep2"),
        pytest.param(dict(head="premium_lr"), id="premium_lr"),
    ],
)
def test_the_mean_of_the_selected_members_is_the_mean_prediction_reserve(config):
    """``company_reserves(selected=True)`` (mean of members) equals the plain fit's reserve.

    A reserve is a sum of forecast cells and the ensemble averages cells, so the mean of the
    kept members' reserves and the reserve of the mean prediction agree. The plain
    ``refit=False`` fit reports the second; the refit fit's ``selected=True`` the first.
    """
    plain = fit(blend_config(**config))
    again = fit(blend_config(refit=True, **config))
    np.testing.assert_allclose(
        again.company_reserves(selected=True), plain.company_reserves(), rtol=1e-6
    )
    kept = again.selection_.index[again.selection_["kept"]].tolist()
    np.testing.assert_allclose(
        again.company_reserves(),
        again.member_company_reserves()[kept].mean(axis=0),
        rtol=1e-6,
    )
