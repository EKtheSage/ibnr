"""gallery.nn.tlrn's JAX training backend against the torch code it mirrors.

Skips cleanly without torch or without jax. On CI it runs on the leg that installs both.

What this file protects. ``backend="jax"`` is only worth having if it trains the SAME
model the torch path trains, so every test below compares the two rather than checking
the JAX code against itself, and they go from the smallest piece to the whole fit:

1. the forward pass: a torch-initialised network's weights loaded into the JAX forward
   give the same predictions, on every head, mask and attention layout the entry has;
2. the loss and its gradient against torch autograd on one batch, including a batch
   padded with zero-weight rows, which is how the JAX program makes a short last batch
   the same shape as the others;
3. the optimiser: gradient clipping and AdamW with two learning rates under the warmup
   cosine schedule, step by step against ``torch.optim.AdamW``;
4. the whole training loop with dropout off, where the two backends must follow the same
   trajectory to float32 rounding - same epochs, same validation scores, same checkpoint
   kept, same early stop - because they start from the same weights and see the same
   batches and cutoffs;
5. the whole fit with dropout on, where they cannot be compared number for number
   (different random streams), so the members' validation scores are compared as two
   samples of the same procedure;
6. that every argument reaches the JAX path through the public entry point, because a
   parameter that is accepted and then ignored is the bug this repository keeps meeting.
"""

from __future__ import annotations

import math
import pickle
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from ibnr import gallery  # noqa: E402
from ibnr.gallery.nn._training import warmup_cosine  # noqa: E402
from ibnr.gallery.nn.tlrn import head as tlrn_head  # noqa: E402
from ibnr.gallery.nn.tlrn import jax_backend as jb  # noqa: E402
from ibnr.gallery.nn.tlrn.blend import blend_weight  # noqa: E402
from ibnr.gallery.nn.tlrn.config import TLRNConfig  # noqa: E402
from ibnr.gallery.nn.tlrn.network import TLRNNetwork  # noqa: E402
from ibnr.kernels.nn_features import pooled_incremental_lr, tlrn_features  # noqa: E402

from .test_nn_features import company_contract, study_arrays, study_triangle  # noqa: E402
from .test_tlrn import tiny  # noqa: E402

#: the fixture for the accident-year variant: four companies, eight accident years, so
#: the final valuation is diagonal 8 and the method can be retrained at 5, 6 and 7
AY_AS_OF = "2007-12-31"


def ay_arrays():
    return study_arrays(n_companies=4, n_w=8, seed=3)


def tiny_ay(**overrides) -> TLRNConfig:
    """The accident-year variant at test size.

    Four protocols (the final fit at diagonal 8 and retrained ones at 5, 6 and 7), whose
    training cutoffs number 4, 1, 2 and 3, so the program's padding of the cutoff axis is
    exercised. Batches of 3 companies out of 4 leave a last batch of one, so the padding
    of a short batch is exercised too.
    """
    return TLRNConfig.accident_year_variant(
        **{
            "d_model": 8,
            "n_heads": 2,
            "dropout": 0.0,
            "max_epochs": 5,
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


def fit(config, *, arrays=None, as_of="2005-12-31", seed=7, incurred=False, **kwargs):
    roles = {"incurred_field": "incurred_loss", "case_field": "case_reserve"} if incurred else {}
    return gallery.fit(
        "tlrn",
        study_triangle("duckdb", arrays if arrays is not None else study_arrays()),
        loss_field="paid_loss",
        as_of=as_of,
        config=config,
        seed=seed,
        **roles,
        **kwargs,
    )


# -- 1. the forward pass -------------------------------------------------------------------


#: every head, mask, attention layout and tail rule the entry has, each against torch
FORWARD_CASES = {
    "published": {},
    "anchored": {"cl_anchor": True},
    "legacy_tail": {"tail_policy": "legacy_init"},
    "factors_only": {"factors_only": True},
    "lag_only": {"cross_line": False},
    "observed_cells": {"mask": "observed_cells"},
    "accident_year": {
        "head": "premium_lr",
        "attention": ("line", "lag", "ay"),
        "mask": "observed_cells",
        "batch_unit": "company",
        "member": "mcl_blend",
    },
    "ay_unwritten": {"attention": ("line", "lag", "ay"), "batch_unit": "company"},
    "two_layers": {"n_layers": 2},
}


def _features(contract, cutoff, *, with_mcl=False):
    return tlrn_features(
        contract,
        cutoff=cutoff,
        target_lo=cutoff + 1,
        target_hi=contract["x"].shape[3] + contract["x"].shape[4] - 1,
        incurred_field="incurred_loss",
        case_field="case_reserve",
        with_mcl=with_mcl,
    )


def _torch_net(cfg, contract, features, cutoff):
    _, n_l, _, n_w, n_d = contract["x"].shape
    head_init = None
    if cfg.head == "premium_lr":
        head_init = torch.tensor(
            np.log(pooled_incremental_lr(contract, cutoff)), dtype=torch.float32
        )
    torch.manual_seed(0)
    model = TLRNNetwork(
        cfg, n_lines=n_l, n_lag=n_d, n_feat=features["n_feat"], n_origin=n_w, head_init=head_init
    )
    # the published initialisation shrinks the output layer a hundredfold, so a fresh
    # network barely moves the forecast; a forward check through it would compare the
    # head alone. Give every parameter a visible size so the body decides the answer.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ("phi", "beta"):
                param.add_(0.1 * torch.randn_like(param))
            else:
                param.copy_(0.4 * torch.randn_like(param))
    model.eval()
    return model


def _torch_forward(model, features, support):
    t = {k: torch.tensor(np.asarray(features[k]), dtype=torch.float32) for k in jb.ROW_KEYS}
    for k in ("lk",):
        t[k] = t[k].long()
    t["written"] = t["written"].bool()
    t["visible"] = t["visible"].bool()
    with torch.no_grad():
        return model(
            t["feat"],
            t["c_lk"],
            t["p_lk"],
            t["lk"],
            torch.tensor(features["line_ix"]).long(),
            torch.tensor(features["lag_ix"]).long(),
            t["written"],
            factor_support=support,
            fallback_logf=torch.tensor(features["fallback_logf"], dtype=torch.float32),
            fallback_lr=torch.tensor(features["fallback_lr"], dtype=torch.float32),
            anchor_logf=t["anchor_logf"],
            anchor_start=t["anchor_start"],
            visible=t["visible"],
        )


@pytest.mark.parametrize("case", sorted(FORWARD_CASES))
def test_forward_matches_torch(case):
    """A torch network's weights in the JAX forward give torch's predictions.

    On the 13-feature contract with one company writing only one line, so the mask of an
    unwritten line is part of what is compared. Every parameter is drawn at a size where
    the body moves the answer, so the comparison is of the whole network and not of the
    head's starting factors.
    """
    contract = company_contract("duckdb", lines={1: [0]})
    cfg = TLRNConfig(dropout=0.3, d_model=8, n_heads=2, **FORWARD_CASES[case])
    cutoff = 4
    features = _features(contract, cutoff, with_mcl=cfg.member == "mcl_blend")
    model = _torch_net(cfg, contract, features, cutoff)
    n_l, n_d = model.n_lines, model.n_lag
    support_np = np.ones((n_l, n_d - 1), dtype=bool)
    support_np[:, -2:] = False  # the last steps fall back, so the substitution is compared
    support = torch.tensor(support_np) if cfg.tail_policy == "observed_cl" else None
    want = _torch_forward(model, features, support)

    spec = jb.spec_of(cfg, n_l=n_l, n_d=n_d, n_w=contract["x"].shape[3])
    names = [n for n, _ in model.named_parameters()]
    params = jb.params_from_state(model.state_dict(), names)
    got = jb.forward(params, jb.as_set(features), spec, support=jnp.asarray(support_np))

    for key in ("pred", "C"):
        expected = want[key].numpy()
        scale = np.abs(expected).max()
        np.testing.assert_allclose(np.asarray(got[key]), expected, rtol=2e-5, atol=2e-6 * scale)
    # and the network did move the forecast, or the comparison above says nothing
    if not cfg.factors_only:
        bare = jb.forward(
            {**params, "head.weight": jnp.zeros_like(params["head.weight"])},
            jb.as_set(features),
            spec,
            support=jnp.asarray(support_np),
        )
        assert not np.allclose(np.asarray(bare["pred"]), np.asarray(got["pred"]), rtol=1e-3)


def test_dropout_acts_in_training_only():
    """Eval mode is deterministic and training mode drops, with torch's 1 / (1 - p) scale."""
    contract = company_contract("duckdb")
    cfg = TLRNConfig(dropout=0.5, d_model=8, n_heads=2)
    features = _features(contract, 4)
    model = _torch_net(cfg, contract, features, 4)
    spec = jb.spec_of(cfg, n_l=model.n_lines, n_d=model.n_lag, n_w=contract["x"].shape[3])
    names = [n for n, _ in model.named_parameters()]
    params = jb.params_from_state(model.state_dict(), names)
    s = jb.as_set(features)
    support = jnp.ones((model.n_lines, model.n_lag - 1), bool)
    key = jax.random.key(0)
    evaluated = jb.forward(params, s, spec, train=False, key=key, support=support)["pred"]
    trained = jb.forward(params, s, spec, train=True, key=key, support=support)["pred"]
    other = jb.forward(params, s, spec, train=True, key=jax.random.key(1), support=support)["pred"]
    assert not np.allclose(np.asarray(evaluated), np.asarray(trained))
    assert not np.allclose(np.asarray(trained), np.asarray(other))

    x = jnp.ones((20000,))
    dropped = np.asarray(jb._dropout(x, jax.random.key(2), 0.3, True))
    np.testing.assert_allclose(np.unique(dropped), [0.0, 1 / 0.7], rtol=1e-6)
    assert dropped.mean() == pytest.approx(1.0, abs=0.03)


# -- 2. the loss and its gradient ------------------------------------------------------------


@pytest.mark.parametrize("case", ["published", "accident_year"])
def test_loss_and_gradient_match_torch_autograd(case):
    """``point_loss`` and its gradient with respect to every parameter, on one batch.

    The batch is then padded the way the JAX program pads a short last batch - copies of
    the first unit, weighted zero - and the loss and gradient must not move: a padded row
    that leaked into a numerator, a denominator or the attention would show here.
    """
    contract = company_contract("duckdb", lines={1: [0]})
    cfg = TLRNConfig(dropout=0.0, d_model=8, n_heads=2, **FORWARD_CASES[case])
    cutoff = 3
    features = tlrn_features(
        contract,
        cutoff=cutoff,
        target_lo=cutoff + 1,
        target_hi=5,
        incurred_field="incurred_loss",
        case_field="case_reserve",
    )
    model = _torch_net(cfg, contract, features, cutoff)
    model.train()
    n_l, n_d = model.n_lines, model.n_lag
    n_w = contract["x"].shape[3]
    support_np = np.ones((n_l, n_d - 1), dtype=bool)
    support_np[:, -1] = False
    support = torch.tensor(support_np)
    rows = np.arange(n_w, 3 * n_w) if cfg.batch_unit == "company" else np.array([1, 4, 5, 7, 9])

    t = {k: torch.tensor(np.asarray(features[k])[rows], dtype=torch.float32) for k in jb.ROW_KEYS}
    out = model(
        t["feat"],
        t["c_lk"],
        t["p_lk"],
        t["lk"].long(),
        torch.tensor(features["line_ix"]).long(),
        torch.tensor(features["lag_ix"]).long(),
        t["written"].bool(),
        factor_support=support,
        fallback_logf=torch.tensor(features["fallback_logf"], dtype=torch.float32),
        fallback_lr=torch.tensor(features["fallback_lr"], dtype=torch.float32),
        anchor_logf=t["anchor_logf"],
        anchor_start=t["anchor_start"],
        visible=t["visible"].bool(),
    )
    loss = tlrn_head.point_loss(
        out["pred"],
        t["target"],
        t["target_mask"],
        t["premium"],
        n_l,
        n_d,
        w_pe=cfg.w_pe,
        w_mse=cfg.w_mse,
        mse_scale=cfg.mse_scale,
    )
    model.zero_grad()
    loss.backward()
    want = {n: p.grad.numpy() for n, p in model.named_parameters() if p.grad is not None}
    assert float(loss.detach()) > 0

    spec = jb.spec_of(cfg, n_l=n_l, n_d=n_d, n_w=n_w)
    names = [n for n, _ in model.named_parameters()]
    params = jb.params_from_state(model.state_dict(), names)
    full = jb.as_set(features)

    def batch_loss(p, take, weight):
        s = {k: full[k][take] for k in jb.ROW_KEYS}
        s |= {k: full[k] for k in (*jb.SET_KEYS, "line_ix", "lag_ix")}
        pred = jb.forward(p, s, spec, train=True, support=jnp.asarray(support_np))["pred"]
        return jb.point_loss(
            pred, s["target"], s["target_mask"] * weight[:, None], s["premium"], spec
        )

    value, grads = jax.value_and_grad(batch_loss)(
        params, jnp.asarray(rows), jnp.ones(len(rows), jnp.float32)
    )
    assert float(value) == pytest.approx(float(loss.detach()), rel=1e-5)
    for name in names:
        expected = want.get(name, np.zeros_like(np.asarray(params[name])))
        scale = max(np.abs(expected).max(), 1e-12)
        np.testing.assert_allclose(
            np.asarray(grads[name]), expected, rtol=1e-4, atol=1e-4 * scale, err_msg=name
        )

    # padded: one more unit's worth of copies of the first unit, weighted zero
    pad = rows[:n_w] if cfg.batch_unit == "company" else rows[:1]
    padded_rows = np.concatenate([rows, pad])
    weight = np.concatenate([np.ones(len(rows)), np.zeros(len(pad))]).astype(np.float32)
    padded_value, padded_grads = jax.value_and_grad(batch_loss)(
        params, jnp.asarray(padded_rows), jnp.asarray(weight)
    )
    assert float(padded_value) == pytest.approx(float(value), rel=1e-6)
    for name in names:
        np.testing.assert_allclose(
            np.asarray(padded_grads[name]), np.asarray(grads[name]), rtol=1e-5, atol=1e-9
        )


# -- 3. the optimiser --------------------------------------------------------------------------


def test_clip_and_adamw_match_torch_step_for_step():
    """Six steps of clipping plus AdamW with two rate groups under the warmup cosine.

    The gradients are large enough that clipping binds, the decay is on, and one parameter
    is in the head's group at ten times the rate, as ``param_groups`` builds them.
    """
    rng = np.random.default_rng(0)
    shapes = {"phi": (2, 5), "inp.weight": (4, 3), "inp.bias": (4,)}
    start = {n: rng.normal(size=s).astype(np.float32) for n, s in shapes.items()}
    tensors = {n: torch.nn.Parameter(torch.tensor(v)) for n, v in start.items()}
    lr, lr_phi, wd, clip = 3e-3, 3e-2, 0.01, 0.5
    opt = torch.optim.AdamW(
        [
            {"params": [tensors["phi"]], "lr": lr_phi},
            {"params": [tensors["inp.weight"], tensors["inp.bias"]]},
        ],
        lr=lr,
        weight_decay=wd,
    )
    for group in opt.param_groups:
        group["base_lr"] = group["lr"]
    schedule = warmup_cosine(6, 2)

    params = {n: jnp.asarray(v) for n, v in start.items()}
    m = {n: jnp.zeros_like(v) for n, v in params.items()}
    v = {n: jnp.zeros_like(x) for n, x in params.items()}
    t = jnp.float32(0.0)
    base_lr = {"phi": lr_phi, "inp.weight": lr, "inp.bias": lr}
    decays = dict.fromkeys(params, True)
    for epoch in range(1, 7):
        grads = {n: (3.0 * rng.normal(size=s)).astype(np.float32) for n, s in shapes.items()}
        mult = schedule(epoch)
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * mult
        opt.zero_grad()
        for n, g in grads.items():
            tensors[n].grad = torch.tensor(g)
        norm = float(torch.nn.utils.clip_grad_norm_(list(tensors.values()), clip))
        assert norm > clip  # clipping binds on every step
        opt.step()

        clipped = jb.clip_by_global_norm({n: jnp.asarray(g) for n, g in grads.items()}, clip)
        params, m, v, t = jb.adamw_update(
            params, m, v, t, clipped, base_lr=base_lr, mult=mult, weight_decay=wd, decays=decays
        )
        for n in shapes:
            np.testing.assert_allclose(
                np.asarray(params[n]),
                tensors[n].detach().numpy(),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{n} after step {epoch}",
            )
    assert float(t) == 6.0


def test_blend_weight_matches_the_numpy_one():
    rng = np.random.default_rng(4)
    for _ in range(20):
        a = rng.normal(size=60).astype(np.float32)
        b = rng.normal(size=60).astype(np.float32)
        b[rng.random(60) < 0.2] = 0.0
        assert float(jb.blend_weight(jnp.asarray(a), jnp.asarray(b))) == pytest.approx(
            blend_weight(a, b), abs=1e-6
        )
    assert float(jb.blend_weight(jnp.ones(4), jnp.zeros(4))) == 0.0


# -- 4. the whole training loop, dropout off -------------------------------------------------


def _assert_same_training(torch_entry, jax_entry, *, rtol):
    """Two fits followed the same trajectory: histories, selection, weights, reserves.

    WHEN THIS CAN FAIL WITH NEITHER BACKEND WRONG. The agreement holds while every
    parameter's gradient over a batch is either a real number or exactly zero in both.
    It breaks when the true gradient is zero and torch's autograd leaves rounding there:
    measured on this fixture (seed 11, a batch whose scored cells all start past the
    first development step), torch gave ``phi[:, 0]`` a gradient of 5e-9 where jax gave
    exactly 0, and Adam, which divides by the gradient's own size, turned torch's 5e-9
    into a step of a third of the learning rate. The fits below use seeds where that does
    not happen; a failure after a change should be read with it in mind (see
    ``knowledge/findings/tlrn-jax-backend-parity.md``).
    """
    a, b = torch_entry.selection_, jax_entry.selection_
    pd.testing.assert_series_equal(a["kept"], b["kept"])
    pd.testing.assert_series_equal(a["epochs_run"], b["epochs_run"])
    pd.testing.assert_series_equal(a["best_epoch"], b["best_epoch"])
    for col in ("validation_ay_line_ape", "validation_company_ape"):
        np.testing.assert_allclose(b[col], a[col], rtol=rtol)
    for ha, hb in zip(torch_entry.history_, jax_entry.history_, strict=True):
        assert [(r["member"], r["epoch"]) for r in ha] == [(r["member"], r["epoch"]) for r in hb]
        np.testing.assert_allclose([r["train"] for r in hb], [r["train"] for r in ha], rtol=rtol)
        np.testing.assert_allclose(
            [r["val"] for r in hb], [r["val"] for r in ha], rtol=rtol, equal_nan=True
        )
    np.testing.assert_allclose(
        jax_entry.member_company_reserves(), torch_entry.member_company_reserves(), rtol=rtol
    )
    for ma, mb in zip(torch_entry.models_, jax_entry.models_, strict=True):
        for (na, va), (nb, vb) in zip(
            ma.state_dict().items(), mb.state_dict().items(), strict=True
        ):
            assert na == nb
            va, vb = va.numpy(), vb.numpy()
            if na.endswith("in_proj_bias"):
                # the KEY bias adds the same number to every key's score, which the
                # softmax cancels, so its true gradient is always zero and it moves on
                # rounding alone, differently in each backend; it changes no output
                d = va.shape[0] // 3
                va, vb = np.delete(va, np.s_[d : 2 * d]), np.delete(vb, np.s_[d : 2 * d])
            scale = float(np.abs(va).max()) or 1.0
            np.testing.assert_allclose(vb, va, rtol=rtol, atol=rtol * scale, err_msg=na)


def test_published_training_follows_the_torch_trajectory():
    """Dropout off, so nothing random is left but the batches and cutoffs, which are torch's.

    Clipping binds (``grad_clip`` is small), weight decay is on, and 18 examples in
    batches of 8 leave a last batch of 2, so every part of the loop is in the comparison.
    """
    cfg = replace(tiny(), max_epochs=4, grad_clip=0.05, weight_decay=0.01, ensemble_size=3)
    a = fit(cfg)
    b = fit(cfg, backend="jax")
    _assert_same_training(a, b, rtol=2e-4)


def test_accident_year_training_follows_the_torch_trajectory():
    """The accident-year variant: four valuation dates trained as one program.

    The premium head, attention across accident years on observed cells only, whole
    companies per batch (with a padded last batch), the chain-ladder blend whose weight
    is refitted at every check and kept with its checkpoint, and three retrained
    protocols with 1, 2 and 3 training cutoffs beside the final one's 4.
    """
    cfg = tiny_ay(grad_clip=0.05)
    a = fit(cfg, arrays=ay_arrays(), as_of=AY_AS_OF, incurred=True)
    b = fit(cfg, arrays=ay_arrays(), as_of=AY_AS_OF, incurred=True, backend="jax")
    _assert_same_training(a, b, rtol=2e-4)
    np.testing.assert_allclose(b.selection_["alpha"], a.selection_["alpha"], rtol=1e-3, atol=1e-5)
    # the calibration is built from the retrained protocols, so it agrees too
    np.testing.assert_allclose(b.backtest_["predicted"], a.backtest_["predicted"], rtol=1e-3)


def stopping(**overrides) -> TLRNConfig:
    """A budget where early stopping fires, at a different epoch for each member.

    A learning rate high enough that validation stops improving within a few checks:
    under torch the four members stop after 6, 8, 6 and 4 epochs at seed 2.
    """
    return replace(
        tiny(),
        **{
            "max_epochs": 12,
            "check_every": 1,
            "patience": 2,
            "min_epochs": 3,
            "warmup": 0,
            "ensemble_size": 4,
            "lr": 0.05,
            "lr_phi": 0.05,
            **overrides,
        },
    )


def _stop_by_the_rule(history: list[dict], patience: int, min_epochs: int, max_epochs: int):
    """Replay ``train_ensemble``'s stopping rule on a history: True if it ends where it should.

    The rule is read off the history's own validation scores, so it holds whatever the
    scores are - which is what makes it a check of the JAX loop's stopping logic that
    does not depend on the two backends agreeing about the numbers.
    """
    best, left = math.inf, patience
    for record in history:
        if math.isnan(record["val"]):
            continue
        if record["val"] < best - 1e-6:
            best, left = record["val"], patience
        else:
            left -= 1
            if left <= 0 and record["epoch"] + 1 >= min_epochs:
                return record["epoch"] + 1 == len(history)
    return len(history) == max_epochs


def test_early_stopping_stops_where_torch_stops():
    """``patience`` counts checks and ``min_epochs`` holds the stop back, in both backends.

    Two checks. Each backend's histories end exactly where the stopping rule, replayed on
    that backend's own validation scores, says they should; and at this seed the two
    backends stop every member at the same epoch with the same scores.
    """
    cfg = stopping(keep=4)
    a = fit(cfg, seed=2)
    b = fit(cfg, seed=2, backend="jax")
    runs = a.selection_["epochs_run"]
    assert (runs < 12).any() and runs.nunique() > 1, a.selection_  # the stop did fire
    for entry in (a, b):
        for history in entry.history_:
            assert _stop_by_the_rule(history, cfg.patience, cfg.min_epochs, cfg.max_epochs)
    _assert_same_training(a, b, rtol=5e-4)


# -- 5. the whole fit, dropout on -------------------------------------------------------------


def test_a_jax_fit_with_dropout_is_a_working_entry_of_the_same_quality():
    """With dropout the two backends draw different masks, so they cannot agree number for
    number. What can be required: the entry works end to end, the JAX members are not the
    torch members, and their validation scores look like the same procedure's.

    Eight members each at the published dropout of 0.3. The comparison is deliberately
    coarse - the medians within a factor of 1.5 of each other and the ranges overlapping -
    because eight members of a tiny fixture is a small sample, and that is what it can
    honestly support.
    """
    cfg = replace(tiny(), dropout=0.3, max_epochs=30, check_every=5, ensemble_size=8, keep=2)
    a = fit(cfg, seed=5)
    b = fit(cfg, seed=5, backend="jax")
    off = fit(replace(cfg, dropout=0.0), seed=5, backend="jax")

    sa = a.selection_["validation_ay_line_ape"].to_numpy()
    sb = b.selection_["validation_ay_line_ape"].to_numpy()
    assert np.isfinite(sb).all()
    assert 1 / 1.5 < np.median(sb) / np.median(sa) < 1.5, (sa, sb)
    assert sb.min() < sa.max() and sa.min() < sb.max(), (sa, sb)
    assert not np.allclose(sa, sb)  # different dropout draws, different members
    # and dropout did act in the JAX training: without it the members come out different
    assert not np.allclose(sb, off.selection_["validation_ay_line_ape"].to_numpy())

    company = b.cohorts()[0]
    assert len(b.point(company)) == 2 * 6 + 2 + 1
    reserves = b.company_reserves()
    assert reserves.shape == (3,) and np.isfinite(reserves).all() and (reserves > 0).all()
    pred = b.predict(company, seed=3)
    assert pred.samples.shape == (50, 1) and np.isfinite(pred.samples).all()
    assert b.member_company_reserves().shape == (8, 3)
    assert len(b.models_) == 2

    # a fitted entry survives pickling, which is how a Colab run keeps its fits
    again = pickle.loads(pickle.dumps(b))
    np.testing.assert_array_equal(again.predict(company, seed=3).samples, pred.samples)
    np.testing.assert_array_equal(again.point_ultimates_, b.point_ultimates_)


# -- 6. every argument reaches the JAX path ----------------------------------------------------


def test_backend_jax_trains_without_the_torch_loop(monkeypatch):
    """The torch loop is never entered under ``backend="jax"``, and the JAX one is."""

    def boom(*args, **kwargs):
        raise RuntimeError("the torch training loop ran")

    monkeypatch.setattr("ibnr.gallery.nn.tlrn.model.train_ensemble", boom)
    entry = fit(tiny(), backend="jax")
    assert len(entry.selection_) == 2
    with pytest.raises(RuntimeError, match="the torch training loop ran"):
        fit(tiny())

    calls = []
    real = jb.train_protocols
    monkeypatch.setattr(jb, "train_protocols", lambda *a, **k: calls.append(1) or real(*a, **k))
    fit(tiny(), backend="jax")
    assert calls == [1]


def test_the_budget_reaches_the_jax_path():
    """``max_epochs``, ``ensemble_size``, ``keep``, ``check_every`` and ``patience``."""
    cfg = replace(tiny(), max_epochs=7, check_every=3, ensemble_size=3, keep=2)
    entry = fit(cfg, backend="jax")
    table = entry.selection_
    assert len(table) == 3  # ensemble_size
    assert int(table["kept"].sum()) == 2 and len(entry.models_) == 2  # keep
    assert (table["epochs_run"] == 7).all()  # max_epochs
    for history in entry.history_:
        checked = [r["epoch"] for r in history if not math.isnan(r["val"])]
        assert checked == [2, 5, 6]  # check_every, and the last epoch
    assert entry.member_company_reserves().shape == (3, 3)

    runs = fit(stopping(), seed=2, backend="jax").selection_["epochs_run"]
    assert (runs < 12).any() and (runs >= 3).all()  # patience, held back by min_epochs
    held = fit(stopping(min_epochs=7), seed=2, backend="jax").selection_["epochs_run"]
    assert (held >= 7).all() and not held.equals(runs)  # min_epochs
    patient = fit(stopping(patience=50), seed=2, backend="jax").selection_["epochs_run"]
    assert (patient == 12).all()  # patience


def test_jax_fits_are_reproducible_under_a_seed():
    a = fit(replace(tiny(), dropout=0.3), seed=4, backend="jax")
    b = fit(replace(tiny(), dropout=0.3), seed=4, backend="jax")
    c = fit(replace(tiny(), dropout=0.3), seed=5, backend="jax")
    np.testing.assert_array_equal(a.member_company_reserves(), b.member_company_reserves())
    assert not np.array_equal(a.member_company_reserves(), c.member_company_reserves())


def test_backend_refusals():
    with pytest.raises(ValueError, match="backend must be one of"):
        fit(tiny(), backend="tpu")
    with pytest.raises(ValueError, match="processes=2"):
        fit(tiny(), backend="jax", processes=2)
    with pytest.raises(ValueError, match="cutoff_sampling='per_epoch'"):
        fit(replace(tiny(), cutoff_sampling="per_example"), backend="jax")


def test_the_entry_imports_without_jax():
    """jax is imported inside ``fit(backend="jax")`` only, never by the entry's module."""
    code = (
        "import sys; import ibnr.gallery.nn.tlrn.model; import ibnr.gallery; "
        "assert 'jax' not in sys.modules, 'importing the tlrn entry pulled in jax'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
