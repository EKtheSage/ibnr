"""gallery.nn.transformer: network, training scheme, entry contract.
Skips cleanly when torch is not installed."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.nn.transformer.config import TransformerConfig  # noqa: E402
from ibnr.gallery.nn.transformer.model import NNTransformer, _norm_stats, _splits  # noqa: E402
from ibnr.gallery.nn.transformer.network import (  # noqa: E402
    TriangleTransformer,
    mdn_nll,
    mdn_sample,
)

from .conftest import make_multiline_triangle  # noqa: E402

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
    n_draws=50,
)

START = 2000


def synthetic_triangle(backend_name, n_w=6, seed=0):
    rng = np.random.default_rng(seed)
    n_d = n_w
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    cum = np.cumsum(incr, axis=2)
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    return make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)


# -- network -------------------------------------------------------------------


def test_forward_shapes_and_validity():
    cfg = TINY
    torch.manual_seed(0)
    model = TriangleTransformer(cfg, n_lob=3, n_features=2, n_w=5, n_d=4)
    b = 7
    x = torch.randn(b, 2, 5, 4)
    ctx = torch.rand(b, 5, 4) > 0.5
    lob = torch.randint(0, 3, (b,))
    prem = torch.randn(b)
    log_pi, mu, sigma = model(x, ctx, lob, prem)
    assert log_pi.shape == mu.shape == sigma.shape == (b, 5, 4, cfg.n_components)
    assert (sigma > 0).all()
    torch.testing.assert_close(log_pi.logsumexp(dim=-1), torch.zeros(b, 5, 4), atol=1e-5, rtol=0)


def test_mdn_nll_and_sample_consistency():
    torch.manual_seed(1)
    shape = (64, 3, 3)
    log_pi = torch.log_softmax(torch.randn(*shape, 2), dim=-1)
    mu = torch.randn(*shape, 2)
    sigma = torch.rand(*shape, 2) + 0.5
    y = torch.randn(*shape)
    mask = torch.ones(shape, dtype=torch.bool)
    nll = mdn_nll(log_pi, mu, sigma, y, mask)
    assert torch.isfinite(nll)

    gen = torch.Generator().manual_seed(2)
    s1 = mdn_sample(log_pi, mu, sigma, generator=gen)
    assert s1.shape == shape
    gen2 = torch.Generator().manual_seed(2)
    s2 = mdn_sample(log_pi, mu, sigma, generator=gen2)
    torch.testing.assert_close(s1, s2)


def test_overfit_one_batch():
    torch.manual_seed(3)
    cfg = TINY
    model = TriangleTransformer(cfg, n_lob=1, n_features=1, n_w=4, n_d=4)
    x = torch.randn(8, 1, 4, 4)
    ctx = torch.zeros(8, 4, 4, dtype=torch.bool)
    ctx[:, :, :2] = True
    tgt = ~ctx
    y = x[:, 0]
    lob = torch.zeros(8, dtype=torch.long)
    prem = torch.zeros(8)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(300):
        log_pi, mu, sigma = model(x, ctx, lob, prem)
        loss = mdn_nll(log_pi, mu, sigma, y, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0] - 0.5, f"no material overfit: {losses[0]:.3f} -> {losses[-1]:.3f}"


# -- training scheme helpers -----------------------------------------------------


def test_splits_exclude_validation_diagonal():
    obs = np.zeros((2, 4, 4), dtype=bool)
    cal = np.arange(4)[:, None] + np.arange(4)[None, :] + 1
    obs[:] = cal <= 4  # upper triangle
    context, val_target, val_cutoff = _splits(obs, cal, val_diagonals=1)
    assert val_cutoff == 3
    assert not (context & (cal > 3)[None]).any()
    np.testing.assert_array_equal(val_target, obs & (cal == 4)[None])
    # partition of the observed cells
    np.testing.assert_array_equal(context | val_target, obs)
    assert not (context & val_target).any()


def test_splits_raise_on_tiny_windows():
    obs = np.ones((1, 2, 1), dtype=bool)
    cal = np.array([[1], [2]])
    with pytest.raises(ValueError, match="diagonals"):
        _splits(obs, cal, val_diagonals=1)


def test_norm_stats_ignore_validation_cells():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(5, 1, 4, 4))
    cal = np.arange(4)[:, None] + np.arange(4)[None, :] + 1
    obs = np.broadcast_to(cal <= 4, (5, 4, 4))
    context, _, _ = _splits(obs, cal, val_diagonals=1)
    x_spiked = x.copy()
    x_spiked[:, :, cal == 4] = 1e6  # poison the validation diagonal
    mean_a, std_a = _norm_stats(x, context)
    mean_b, std_b = _norm_stats(x_spiked, context)
    np.testing.assert_allclose(mean_a, mean_b)
    np.testing.assert_allclose(std_a, std_b)


# -- entry ---------------------------------------------------------------------


def test_fit_predict_contract(backend_name):
    t = synthetic_triangle(backend_name)
    entry = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    seg = {"company_code": "0001", "line_of_business": "lob_0"}
    pred = entry.predict(segment=seg, seed=0)
    assert pred.n_targets == 6 + 1  # per origin + total
    assert pred.targets["label"].tolist()[-1] == "total"
    assert np.isfinite(pred.samples).all()
    assert (pred.samples > 0).mean() > 0.95  # ultimates are overwhelmingly positive

    realized = entry.realized_ultimates(t, segment=seg)
    assert realized.shape == (7,)
    table = pred.summary(observed=realized)
    assert {"estimate", "se", "cv", "outcome", "percentile"} <= set(table.columns)

    # fully developed first origin: ultimate anchored at its observed value
    first = entry.contract_["latest_cum"][entry._cohort_index(seg), 0]
    np.testing.assert_allclose(pred.samples[:, 0], first)


def test_predict_all_cohorts_layout(backend_name):
    t = synthetic_triangle(backend_name)
    entry = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    pred = entry.predict(seed=0)
    assert pred.n_targets == 2 * 6  # cohorts x origins, no grand total
    assert {"company_code", "line_of_business", "origin_period", "premium"} <= set(
        pred.targets.columns
    )
    realized = entry.realized_ultimates(t)
    assert realized.shape == (12,)


def test_predict_caches_rollout_and_is_reproducible(backend_name):
    t = synthetic_triangle(backend_name)
    entry = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    a = entry.predict(segment={"company_code": "0001", "line_of_business": "lob_0"}, seed=7)
    cached = entry._rollout_ults
    b = entry.predict(segment={"company_code": "0001", "line_of_business": "lob_1"}, seed=7)
    assert entry._rollout_ults is cached  # same rollout reused across segments
    assert not np.allclose(a.samples, b.samples)

    entry2 = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    a2 = entry2.predict(segment={"company_code": "0001", "line_of_business": "lob_0"}, seed=7)
    np.testing.assert_allclose(a.samples, a2.samples)


def test_training_targets_never_touch_validation_diagonal(backend_name):
    # white-box: with val_diagonals=1 the training target mask is capped at
    # val_cutoff; verified via the split helper on the entry's real contract
    t = synthetic_triangle(backend_name)
    entry = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    c = entry.contract_
    context, val_target, val_cutoff = _splits(c["obs_mask"], c["cal_idx"], TINY.val_diagonals)
    assert (c["cal_idx"][context.any(axis=0)] <= val_cutoff).all()
    assert (c["cal_idx"][val_target.any(axis=0)] > val_cutoff).all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        NNTransformer().predict()


def test_unknown_segment_raises(backend_name):
    t = synthetic_triangle(backend_name)
    entry = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    with pytest.raises(KeyError, match="unknown segment column"):
        entry.predict(segment={"nope": "x"})
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.predict(segment={"company_code": "9999"})
