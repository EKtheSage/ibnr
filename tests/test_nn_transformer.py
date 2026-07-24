"""gallery.nn.transformer: network, training scheme, entry contract.
Skips cleanly when torch is not installed.

What this file protects, in three layers:

1. **Network.** The masked-cell triangle transformer emits a valid mixture
   density at every cell (normalized weights, positive sigmas), the MDN NLL is
   finite, and sampling is seed-reproducible. A point estimator cannot enter the
   gallery - the distributional head IS the contract, so its validity is pinned
   directly rather than inferred from downstream metrics.
2. **Training scheme.** The calendar split and per-dev standardization. These
   are where leakage hides: validation is a held-out *calendar diagonal* (not a
   random cell split, which would leak future development into training), and
   normalization statistics must be computed from context cells only. The v2
   fix recorded in CLAUDE.md - pinned per-dev standardization replacing the old
   inherit-earlier-dev tail stats - was the dominant defect in v1, so
   ``_norm_stats`` gets an explicit poisoning test.
3. **Entry.** PredictiveDistribution layout, ``realized_ultimates`` alignment,
   determinism, and error paths.

Two recurring themes worth knowing before reading:

*Determinism.* Every fit/predict takes an explicit ``seed`` and every sampling
helper an explicit ``torch.Generator``. Tests assert bit-comparable repeats
(same seed -> same draws) and that different segments still differ. Without
this a deep ensemble's spread is unreproducible and no calibration result is
citable.

*Small triangles.* Schedule P triangles are ~55 cells, so overfitting is the
central risk (CLAUDE.md). The entry therefore fits ONE global model across many
company x LOB triangles and predicts per segment; the tests exercise that
global-fit/per-segment-predict shape rather than a single-triangle fit. The
real many-triangle check runs on mart data in ``test_statistical_mart.py``.

``TINY`` is a deliberately under-powered config so the suite stays fast; nothing
here asserts predictive quality. Entry-level tests run on BOTH ibis backends.
"""

from __future__ import annotations

from dataclasses import replace

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
    """Two lognormal-noise LOB triangles with a decaying incremental pattern.

    A full square (not just the upper triangle) so ``realized_ultimates`` has
    something to score; small and well-behaved because these tests check
    plumbing, not fit quality.
    """
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
    """The MDN head is a valid density at every cell: one (log_pi, mu, sigma)
    triple per (batch, origin, dev, component), sigma strictly positive, and
    mixture weights normalized (logsumexp == 0)."""
    cfg = TINY
    torch.manual_seed(0)
    model = TriangleTransformer(cfg, n_lob=3, n_features=2, n_w=5, n_d=4)
    b = 7
    x = torch.randn(b, 2, 5, 4)
    ctx = torch.rand(b, 5, 4) > 0.5
    lob = torch.randint(0, 3, (b,))
    prem = torch.randn(b)
    cutoff = torch.randint(1, 8, (b,))
    log_pi, mu, sigma = model(x, ctx, lob, prem, cutoff)
    assert log_pi.shape == mu.shape == sigma.shape == (b, 5, 4, cfg.n_components)
    assert (sigma > 0).all()
    # atol 1e-5 in float32: log_softmax normalization is exact up to fp error.
    torch.testing.assert_close(log_pi.logsumexp(dim=-1), torch.zeros(b, 5, 4), atol=1e-5, rtol=0)


def test_mdn_nll_and_sample_consistency():
    """The mixture loss is finite on random inputs, and drawing from the head
    is reproducible: the same explicit ``torch.Generator`` seed yields
    bit-identical samples. Reproducible draws are a precondition for any
    calibration claim made from this model."""
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
    """The standard "can it learn at all" check: gradients flow end to end
    through masking, embeddings and the MDN head, so the loss drops materially
    on a single fixed batch. A model that cannot overfit one batch has a wiring
    bug (detached tensor, wrong mask) that no accuracy metric would localize."""
    torch.manual_seed(3)
    cfg = TINY
    model = TriangleTransformer(cfg, n_lob=1, n_features=1, n_w=4, n_d=4)
    x = torch.randn(8, 1, 4, 4)
    ctx = torch.zeros(8, 4, 4, dtype=torch.bool)
    ctx[:, :, :2] = True  # first two dev lags are context; the rest are targets
    tgt = ~ctx
    y = x[:, 0]
    lob = torch.zeros(8, dtype=torch.long)
    prem = torch.zeros(8)
    cutoff = torch.full((8,), 2, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(600):
        log_pi, mu, sigma = model(x, ctx, lob, prem, cutoff)
        loss = mdn_nll(log_pi, mu, sigma, y, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert np.isfinite(losses).all()
    # best-so-far, not final: at this lr the tail of the trajectory wobbles
    assert min(losses) < losses[0] - 0.5, (
        f"no material overfit: {losses[0]:.3f} -> best {min(losses):.3f}"
    )


def test_exposure_sigma_head_matches_baseline_at_init():
    """The optional exposure-scaled sigma head is a strict, ablatable extension.

    Actuarial motivation: process variance scales with exposure, so sigma is
    multiplied by premium**(p-1) with p learnable. Initialized at p = 1 the
    variant must be *numerically identical* to the flat-sigma baseline - that
    is what makes any measured difference attributable to the feature rather
    than to a reinitialized head. The second half checks the knob is live: a
    non-unit p changes sigma and gradient reaches ``raw_p``.
    """
    # p = softplus(raw_p) starts at 1.0, so the exposure factor is premium**0
    # everywhere and the head is identical to the flat-sigma baseline. Only
    # sigma is touched; mu is never scaled.
    torch.manual_seed(0)
    kw = dict(n_lob=2, n_features=2, n_w=5, n_d=4)
    base = TriangleTransformer(TINY, **kw)
    exp = TriangleTransformer(replace(TINY, exposure_sigma=True), **kw)
    exp.prem_log_std.fill_(1.7)  # non-trivial spread: masks nothing at p == 1
    exp.load_state_dict({**exp.state_dict(), **base.state_dict()})  # share the encoder weights
    base.eval()
    exp.eval()
    b = 6
    x = torch.randn(b, 2, 5, 4)
    ctx = torch.ones(b, 5, 4, dtype=torch.bool)
    lob = torch.zeros(b, dtype=torch.long)
    prem = torch.randn(b)  # varied premium: still no effect while p == 1
    cutoff = torch.full((b,), 2, dtype=torch.long)
    with torch.no_grad():
        _, mu0, s0 = base(x, ctx, lob, prem, cutoff)
        _, mu1, s1 = exp(x, ctx, lob, prem, cutoff)
    assert torch.nn.functional.softplus(exp.raw_p).item() == pytest.approx(1.0)
    torch.testing.assert_close(s1, s0)
    torch.testing.assert_close(mu1, mu0)

    # a non-unit power rescales sigma by premium**(p-1); gradient reaches raw_p
    exp.raw_p.data.fill_(2.0)
    s2 = exp(x, ctx, lob, prem, cutoff)[2]
    assert not torch.allclose(s2, s0)
    mdn_nll(*exp(x, ctx, lob, prem, cutoff), x[:, 0], ctx).backward()
    assert exp.raw_p.grad is not None and bool(torch.isfinite(exp.raw_p.grad))


# -- training scheme helpers -----------------------------------------------------


def test_splits_exclude_validation_diagonal():
    """Validation is a held-out calendar DIAGONAL, not a random cell split.

    A random split would let the model see a cell's own future development, so
    the split helper must produce a clean partition: context = everything at or
    before the cutoff diagonal, validation target = the observed cells beyond
    it, with no overlap and nothing dropped.
    """
    obs = np.zeros((2, 4, 4), dtype=bool)
    # cal[w, d] = calendar index (origin + dev), i.e. the diagonal a cell sits on
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
    """A triangle too short to give up a diagonal fails loudly instead of
    training with an empty context or an empty validation set."""
    obs = np.ones((1, 2, 1), dtype=bool)
    cal = np.array([[1], [2]])
    with pytest.raises(ValueError, match="diagonals"):
        _splits(obs, cal, val_diagonals=1)


def test_norm_stats_ignore_validation_cells_except_pinned():
    """No leakage through standardization - and an explicit record of the one
    documented exception.

    Per-dev mean/std are computed from CONTEXT cells only, so poisoning the
    validation diagonal with 1e6 must leave them untouched. The exception is a
    dev lag with fewer than two context values (the deepest dev on a triangle):
    there std is *pinned* to 1 and the mean falls back to observed cells at
    that dev. That pinning is the v2 fix from CLAUDE.md - v1 inherited the
    previous dev's statistics and that inheritance was the dominant defect
    (paid err +14% -> +4%, rel-CRPS 0.092 -> 0.050). The test asserts both the
    no-leak property and the exact shape of the exception so it cannot regress
    silently in either direction.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(5, 1, 4, 4))
    cal = np.arange(4)[:, None] + np.arange(4)[None, :] + 1
    obs = np.broadcast_to(cal <= 4, (5, 4, 4))
    context, _, _ = _splits(obs, cal, val_diagonals=1)
    x_spiked = x.copy()
    x_spiked[:, :, cal == 4] = 1e6  # poison the validation diagonal
    mean_a, std_a, pin_a = _norm_stats(x, context, obs)
    mean_b, std_b, pin_b = _norm_stats(x_spiked, context, obs)
    # only the deepest dev lacks context cells -> pinned, std fixed at 1
    np.testing.assert_array_equal(pin_a, [[False, False, False, True]])
    np.testing.assert_array_equal(pin_a, pin_b)
    np.testing.assert_allclose(std_a[0, 3], 1.0)
    # non-pinned devs never see validation cells
    np.testing.assert_allclose(mean_a[:, :3], mean_b[:, :3])
    np.testing.assert_allclose(std_a, std_b)
    # the pinned dev's mean is the only place obs (val) cells enter - by design
    np.testing.assert_allclose(mean_b[0, 3], 1e6)


# -- entry ---------------------------------------------------------------------


def test_fit_predict_contract(backend_name):
    """The GalleryEntry contract for a single segment: per-origin ultimates
    plus a total, positive draws, a scorable summary against realized
    ultimates, and - importantly - a fully developed first origin whose
    "predicted" ultimate is anchored at its observed latest cumulative rather
    than resampled."""
    t = synthetic_triangle(backend_name)
    entry = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    seg = {"company_code": "0001", "line_of_business": "lob_0"}
    pred = entry.predict(segment=seg, seed=0)
    assert pred.n_targets == 6 + 1  # per origin + total
    assert pred.targets["label"].tolist()[-1] == "total"
    assert np.isfinite(pred.samples).all()
    # Not all-positive: the MDN is unconstrained on the standardized scale, so
    # a small tail of negative draws is expected on an under-trained TINY fit.
    assert (pred.samples > 0).mean() > 0.95  # ultimates are overwhelmingly positive

    realized = entry.realized_ultimates(t, segment=seg)
    assert realized.shape == (7,)
    table = pred.summary(observed=realized)
    assert {"estimate", "se", "cv", "outcome", "percentile"} <= set(table.columns)

    # fully developed first origin: ultimate anchored at its observed value
    first = entry.contract_["latest_cum"][entry._cohort_index(seg), 0]
    np.testing.assert_allclose(pred.samples[:, 0], first)


def test_predict_all_cohorts_layout(backend_name):
    """Predicting with no segment returns every (company, LOB) cohort x origin
    and deliberately NO grand total: the single-line transformer's per-line
    draws are independent, so summing them would imply a cross-line
    independence assumption it has not earned (only ``nn_transformer_ml``
    models dependence). Premium travels on the targets frame for
    exposure-normalized scoring."""
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
    """Two properties at once.

    Caching: the autoregressive diagonal rollout is computed for ALL cohorts
    at once and reused, so asking for a second segment must not re-run it
    (identity check on the cached array) - while still returning that segment's
    own draws.

    Determinism: a fresh fit with the same seed reproduces the draws exactly.
    Deep-ensemble members are seeded from the entry seed, so this covers
    training init, batching order and sampling together.
    """
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
    """The no-leakage split is enforced on the entry's REAL contract, not just
    on hand-built masks - the early-stopping signal must come from data the
    training loss never saw, or validation loss is meaningless."""
    # white-box: with val_diagonals=1 the training target mask is capped at
    # val_cutoff; verified via the split helper on the entry's real contract
    t = synthetic_triangle(backend_name)
    entry = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    c = entry.contract_
    context, val_target, val_cutoff = _splits(c["obs_mask"], c["cal_idx"], TINY.val_diagonals)
    assert (c["cal_idx"][context.any(axis=0)] <= val_cutoff).all()
    assert (c["cal_idx"][val_target.any(axis=0)] > val_cutoff).all()


def test_fit_predict_exposure_sigma(backend_name):
    """End-to-end wiring of the exposure-sigma variant through the entry: the
    pooled log-premium spread computed at fit time is handed to every ensemble
    member (so the exposure factor is on a comparable scale across cohorts) and
    prediction still satisfies the usual contract."""
    # distinct per-line premium so the pooled premium spread is non-trivial and
    # the learnable exposure power is wired end to end.
    rng = np.random.default_rng(0)
    n_w = n_d = 6
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    cum = np.cumsum(incr, axis=2)
    lobs = {"lob_0": cum[0], "lob_1": cum[1]}
    prem = {"lob_0": np.full(n_w, 800.0), "lob_1": np.full(n_w, 6000.0)}
    t = make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)

    entry = NNTransformer().fit(
        t, loss_field="paid_loss", config=replace(TINY, exposure_sigma=True), seed=0
    )
    assert entry.norm_["prem_std"] > 0
    # the entry hands each ensemble member the pooled log-premium spread
    for m in entry.models_:
        assert float(m.prem_log_std) == pytest.approx(entry.norm_["prem_std"])
    pred = entry.predict(segment={"company_code": "0001", "line_of_business": "lob_0"}, seed=0)
    assert pred.n_targets == 6 + 1
    assert np.isfinite(pred.samples).all()
    assert (pred.samples > 0).mean() > 0.95


def test_predict_before_fit_raises():
    """GalleryEntry lifecycle: predict() before fit() is a clear RuntimeError."""
    with pytest.raises(RuntimeError, match="fit"):
        NNTransformer().predict()


def test_unknown_segment_raises(backend_name):
    """Segment selection fails loudly on both an unknown column and a known
    column with no matching cohort - silently returning zero cohorts would
    surface much later as an empty predictive distribution."""
    t = synthetic_triangle(backend_name)
    entry = NNTransformer().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    with pytest.raises(KeyError, match="unknown segment column"):
        entry.predict(segment={"nope": "x"})
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.predict(segment={"company_code": "9999"})
