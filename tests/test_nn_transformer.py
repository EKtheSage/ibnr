"""gallery.nn.transformer: network, training scheme, entry contract.
Skips cleanly when torch is not installed.

What this file protects, in three layers:

1. **Network.** The masked-cell triangle transformer emits a valid mixture
   density at every cell (normalized weights, positive sigmas), the MDN NLL is
   finite, and sampling is seed-reproducible. A point estimator cannot enter the
   gallery - the distributional head IS the contract, so its validity is pinned
   directly rather than inferred from downstream metrics. The context mask is
   PER CHANNEL (B, F, W, D): a channel's value is read only behind its own flag,
   which is what keeps a missing feature from arriving as an observed zero.
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

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ibnr import Triangle  # noqa: E402
from ibnr.gallery.nn.transformer import model as tf_model  # noqa: E402
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
#: as_of diagonal 6 of a 6x6 square starting at START - leaves real future cells
#: for the rollout to walk, which a full square does not (every origin is
#: anchored at its last dev and nothing is sampled at all).
AS_OF = "2005-12-31"
SEG0 = {"company_code": "0001", "line_of_business": "lob_0"}


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


#: (w, dev) of the feature hole :func:`featured_triangle` punches in lob_0. Dev
#: step 0 sits on calendar diagonal 1, so it is inside every conditioning window
#: - which is what keeps the per-channel context assertions from passing
#: vacuously at a late cutoff.
FEATURE_HOLE = (0, 0)
#: (w, dev) of the hole punched in lob_1's LEVEL field - the cell that is
#: usable undifferenced and unusable as an increment twice over (itself and its
#: successor), which is what makes the two kinds distinguishable by count
LEVEL_HOLE = (1, 1)
#: the fields :func:`featured_triangle` carries beyond the target
FEATURE = "reported_loss"
LEVEL = "case_reserve"


def featured_triangle(backend_name, n_w=6, seed=0):
    """Two LOB cohorts carrying a feature and a level field ON THEIR OWN CELLS.

    ``conftest.make_multiline_triangle`` emits every field at every observed
    cell, so a feature that is missing exactly where the target is present -
    the case per-channel observedness exists for - cannot be expressed with it.
    lob_0's ``reported_loss`` has a hole at :data:`FEATURE_HOLE`, which costs
    that channel two usable cells (the hole and the successor with nothing to
    difference against) while ``obs_mask`` is untouched.

    ``case_reserve`` is an eval-date SNAPSHOT on the same grid: the shape
    ``level_fields`` exists for. Its hole (lob_1, :data:`LEVEL_HOLE`) costs the
    undifferenced channel one cell and the differenced one two, which is how a
    test tells the two kinds apart. Conventions match the shared builder - yearly
    grain, origin w = Jan 1 of START + w, dev index dev = dev_lag 12*(dev + 1),
    NaN = not emitted. Premium is emitted wherever any field has a cell so it
    never becomes the reason a cohort is screened out.
    """
    rng = np.random.default_rng(seed)
    n_d = n_w
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    cum = np.cumsum(incr, axis=2)
    case = 300.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))

    rows: list[dict] = []
    for k in range(2):
        reported, level = cum[k] * 1.4, case[k].copy()
        if k == 0:
            reported = reported.copy()
            reported[FEATURE_HOLE] = np.nan
        else:
            level[LEVEL_HOLE] = np.nan
        by_field = {"paid_loss": cum[k], FEATURE: reported, LEVEL: level}
        present: set[tuple[int, int]] = set()
        for field, mat in by_field.items():
            for w in range(n_w):
                for d0 in range(n_d):
                    if np.isnan(mat[w, d0]):
                        continue
                    present.add((w, d0))
                    rows.append({"lob": f"lob_{k}", "w": w, "d": d0, "f": field, "v": mat[w, d0]})
        for w, d0 in sorted(present):
            rows.append({"lob": f"lob_{k}", "w": w, "d": d0, "f": "earned_premium", "v": 1000.0})

    frame = pd.DataFrame(
        [
            {
                "company_code": "0001",
                "line_of_business": r["lob"],
                "origin_period": dt.date(START + r["w"], 1, 1),
                "dev_lag": 12 * (r["d"] + 1),
                "eval_date": dt.date(START + r["w"] + r["d"], 12, 31),
                "field": r["f"],
                "value": float(r["v"]),
            }
            for r in rows
        ]
    )
    return Triangle.from_long(frame, measure="cumulative", backend=backend_name)


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
    ctx = torch.rand(b, 2, 5, 4) > 0.5  # per channel, independent across channels
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
    ctx = torch.zeros(8, 1, 4, 4, dtype=torch.bool)
    ctx[:, :, :, :2] = True  # first two dev lags are context; the rest are targets
    tgt = ~ctx[:, 0]
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
    """The optional exposure-scaled sigma head is a strict, switchable extension.

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
    ctx = torch.ones(b, 2, 5, 4, dtype=torch.bool)
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
    mdn_nll(*exp(x, ctx, lob, prem, cutoff), x[:, 0], ctx[:, 0]).backward()
    assert exp.raw_p.grad is not None and bool(torch.isfinite(exp.raw_p.grad))


def test_a_channels_value_is_read_only_behind_its_own_flag():
    """The per-channel context mask, stated as the property it buys.

    One flag per cell was the target's, so a feature value at a cell the TARGET
    was observed at reached the encoder whether or not the feature itself was
    observed there - and at a cell the contract had zero-filled, that value was
    padding presented as an observed zero increment (defect D2). The fix is
    structural: the token is ``[values * channel flags, channel flags]``, so
    channel 1's value can only enter behind channel 1's flag.

    Both directions, because only the pair is a claim: poisoning the feature
    where ITS flag is down changes nothing even with the target's flag up, and
    the same poison where the flag is up moves the output - without the second
    half a network that ignored channel 1 entirely would pass.
    """
    torch.manual_seed(0)
    model = TriangleTransformer(TINY, n_lob=1, n_features=2, n_w=5, n_d=4).eval()
    x = torch.randn(1, 2, 5, 4)
    ctx = torch.ones(1, 2, 5, 4, dtype=torch.bool)
    ctx[0, 1, 2, 2] = False  # the feature is missing at one cell the target has
    lob = torch.zeros(1, dtype=torch.long)
    prem = torch.zeros(1)
    cutoff = torch.full((1,), 4, dtype=torch.long)

    with torch.no_grad():
        base = model(x, ctx, lob, prem, cutoff)
        masked = x.clone()
        masked[0, 1, 2, 2] = 1e3  # absurd value at the masked (channel, cell)
        assert ctx[0, 0, 2, 2], "the target is observed there; that is the point"
        blind = model(masked, ctx, lob, prem, cutoff)
        # and the same poison where the feature IS flagged
        seen = x.clone()
        seen[0, 1, 3, 0] = 1e3
        assert ctx[0, 1, 3, 0]
        live = model(seen, ctx, lob, prem, cutoff)

    for got, want in zip(blind, base, strict=True):
        torch.testing.assert_close(got, want)
    assert not torch.allclose(live[1], base[1]), (
        "a feature value behind a RAISED flag did not move the head, so the test above "
        "passes only because channel 1 is ignored altogether"
    )


def test_one_channel_tokens_are_the_same_projection_as_the_per_cell_flag():
    """The regression guarantee behind every unchanged number in this suite.

    Token width is 2F, not F + 1 - which at F = 1 is the same ``Linear(2, d)``
    over the same two inputs (``[x * flag, flag]``), because the only channel's
    flag IS the cell's. That identity is why the disclosed parameter counts,
    the seeded draws and the shared training-context assertions are untouched
    by the per-channel move; F = 2 is where the width actually differs.
    """

    def width(n_features: int) -> int:
        return TriangleTransformer(
            TINY, n_lob=1, n_features=n_features, n_w=4, n_d=4
        ).value_proj.in_features

    assert width(1) == 2
    assert width(3) == 6


def test_a_per_cell_context_mask_is_refused_by_name():
    """The pre-0.5.4 (B, W, D) mask is a silent wrong answer, so it is a loud one.

    Every caller in the package now passes ``x_obs``-shaped masks; a per-cell
    one would condition each feature channel wherever the TARGET is observed,
    which is exactly the defect the shape change removes. Torch's own error
    for the shape mismatch names a concat dimension, not the mistake.
    """
    model = TriangleTransformer(TINY, n_lob=1, n_features=2, n_w=5, n_d=4)
    x = torch.randn(1, 2, 5, 4)
    with pytest.raises(ValueError, match="PER CHANNEL"):
        model(
            x,
            torch.ones(1, 5, 4, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1),
            torch.full((1,), 4, dtype=torch.long),
        )


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


def test_scheme_helpers_are_the_shared_implementations():
    """The training-scheme helpers live ONCE, in ``gallery/nn/_scheme.py``;
    this module's ``_splits``/``_norm_stats`` are compatibility aliases, not
    copies. A drifted second implementation of the pinning rule is the v2
    regression waiting to happen, so identity (``is``) rather than behavior
    is the assertion."""
    from ibnr.gallery.nn import _scheme

    assert _splits is _scheme.splits
    assert _norm_stats is _scheme.norm_stats


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
    first = entry.contract_["latest_cum"][entry.cohort_index(seg), 0]
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


def _record_forwards(monkeypatch) -> list[dict]:
    """Record every context mask the network is handed, tagged with the training
    batch in flight (``None`` outside ``train_loss`` - the validation pass, and
    the rollout when this is installed after the fit).

    The same two spies ``tests/test_nn_training_context.py`` uses to pin the
    channel-0 gate for every NN entry. That file fits with no feature fields, so
    it cannot see the FEATURE channels' gate at all - which is what the tests
    below are for, and why the spies are repeated rather than shared.
    """
    calls: list[dict] = []
    live: dict[str, dict | None] = {"batch": None}
    real_train_ensemble = tf_model.train_ensemble
    real_forward = TriangleTransformer.forward

    def spy_train_ensemble(n_cohorts, *, train_loss, **kwargs):
        def wrapped(model, idx, cutoffs):
            live["batch"] = {
                "idx": idx.detach().cpu().numpy().copy(),
                "cutoffs": cutoffs.detach().cpu().numpy().copy(),
            }
            try:
                return train_loss(model, idx, cutoffs)
            finally:
                live["batch"] = None

        return real_train_ensemble(n_cohorts, train_loss=wrapped, **kwargs)

    def spy_forward(self, x, context_mask, *rest, **kwargs):
        calls.append({"ctx": context_mask.detach().cpu().numpy().copy(), "batch": live["batch"]})
        return real_forward(self, x, context_mask, *rest, **kwargs)

    monkeypatch.setattr(tf_model, "train_ensemble", spy_train_ensemble)
    monkeypatch.setattr(TriangleTransformer, "forward", spy_forward)
    return calls


def test_training_conditions_each_channel_on_its_own_observed_cells(backend_name, monkeypatch):
    """The entry's training context is ``x_obs`` gated by the cutoff, PER CHANNEL.

    Two claims that the shared no-leak file cannot make, because it fits with no
    feature fields and so has only one channel to look at:

    1. a feature is conditioned on where the FEATURE is usable, not where the
       target is. Under the old per-cell flag the hole this fixture punches was
       invisible: the target is observed there, so the feature's padding zero
       was fed to the encoder as an observed zero increment;
    2. the calendar gate reaches every channel, so training and the rollout
       condition alike (a feature past the cutoff is masked, not read).

    Channel 0 is asserted to be exactly what it always was - obs_mask under the
    same gate - which is the regression half of the same statement.
    """
    t = featured_triangle(backend_name)
    calls = _record_forwards(monkeypatch)
    entry = NNTransformer().fit(
        t, loss_field="paid_loss", feature_fields=(FEATURE,), config=TINY, seed=0
    )
    c = entry.contract_
    cal, obs, x_obs = c["cal_idx"], c["obs_mask"], c["x_obs"]
    n_f = x_obs.shape[1]
    assert n_f == 2
    assert not np.array_equal(x_obs[:, 1], obs), (
        "the fixture's feature channel matches the target's observedness, so every "
        "assertion below would hold under the old per-cell mask too"
    )

    training = [rec for rec in calls if rec["batch"] is not None]
    assert training, "no forward pass recorded inside train_loss; the spy is misplaced"
    for rec in training:
        idx, cutoffs = rec["batch"]["idx"], rec["batch"]["cutoffs"]
        ctx = rec["ctx"]
        assert ctx.shape == (len(idx), n_f, *cal.shape), (
            f"the context mask is {ctx.shape}, not per channel {(len(idx), n_f, *cal.shape)}"
        )
        gate = cal[None, None] <= cutoffs[:, None, None, None]  # (B, 1, W, D)
        np.testing.assert_array_equal(ctx, x_obs[idx] & gate)
        np.testing.assert_array_equal(ctx[:, 0], obs[idx] & gate[:, 0])
    assert any((rec["ctx"][:, 1] != rec["ctx"][:, 0]).any() for rec in training), (
        "no training batch conditioned the feature channel differently from the target's"
    )

    _, _, val_cutoff = _splits(obs, cal, TINY.val_diagonals)
    validation = [rec for rec in calls if rec["batch"] is None]
    assert validation, "no validation pass recorded; early stopping's context is unchecked"
    for rec in validation:
        np.testing.assert_array_equal(rec["ctx"], x_obs & (cal <= val_cutoff))


def test_rollout_promotes_only_the_target_channels_flag(backend_name, monkeypatch):
    """A sampled cell is a TARGET increment; its feature channels stay unobserved.

    The rollout writes its draw into channel 0 and used to promote the whole
    cell to context, which under a per-cell flag told the encoder that every
    feature at that cell was observed - at a value that is the contract's
    padding zero, standardized. Nothing downstream can see that: the draws stay
    finite and plausible, and the deeper the rollout runs the more fabricated
    zeros it conditions on.

    Asserted diagonal by diagonal against the rollout's own future mask, so the
    promotion is pinned to the exact cells rather than to a count, and the
    feature flags are required to be the contract's ``x_obs`` throughout.
    """
    t = featured_triangle(backend_name)
    entry = NNTransformer().fit(
        t,
        loss_field="paid_loss",
        feature_fields=(FEATURE,),
        as_of=AS_OF,
        config=TINY,
        seed=0,
    )
    calls = _record_forwards(monkeypatch)  # installed AFTER the fit: rollout only
    entry.predict(segment=SEG0, seed=0)
    assert calls, "the rollout made no forward pass; there is nothing to promote"

    c = entry.contract_
    n_c, _, _, n_d = c["x"].shape
    future = np.arange(n_d)[None, None, :] >= c["latest_dev"][:, :, None]  # (n_c, n_w, n_d)
    cal_levels = sorted(np.unique(c["cal_idx"][future.any(axis=0)]))
    assert len(cal_levels) >= 2, "a single-step rollout cannot show promotion carrying forward"
    assert len(calls) == len(cal_levels) * len(entry.models_)

    chunk = calls[0]["ctx"].shape[0] // n_c
    fut_rows = np.repeat(future, chunk, axis=0)  # rows are cohort-major, as _rollout builds them
    feature_flags = np.repeat(c["x_obs"][:, 1], chunk, axis=0)
    target_flags = np.repeat(c["obs_mask"], chunk, axis=0)

    for member in range(len(entry.models_)):
        steps = calls[member * len(cal_levels) : (member + 1) * len(cal_levels)]
        np.testing.assert_array_equal(steps[0]["ctx"][:, 0], target_flags)
        for lv, rec, nxt in zip(cal_levels, steps, steps[1:], strict=False):
            promoted = fut_rows & (c["cal_idx"][None] == lv)
            assert promoted.any(), f"diagonal {lv} promoted nothing; the step is vacuous"
            np.testing.assert_array_equal(nxt["ctx"][:, 0], rec["ctx"][:, 0] | promoted)
            np.testing.assert_array_equal(nxt["ctx"][:, 1], rec["ctx"][:, 1])
    for rec in calls:
        np.testing.assert_array_equal(
            rec["ctx"][:, 1],
            feature_flags,
            err_msg="a rollout step raised a feature flag; next year's features are unobserved",
        )


def test_heldout_inputs_condition_per_channel(backend_name):
    """The held-out path conditions the way training did: ``x_obs``, per channel.

    The two must agree or the network is scored under a conditioning it never
    saw - and the cohort this asks for is the one whose feature has a hole, so
    the per-channel and per-cell answers differ.
    """
    t = featured_triangle(backend_name)
    entry = NNTransformer().fit(
        t,
        loss_field="paid_loss",
        feature_fields=(FEATURE,),
        as_of=AS_OF,
        config=TINY,
        seed=0,
    )
    c = entry.contract_
    ci = entry.cohort_index(SEG0)
    inputs = entry._heldout_inputs(ci)
    ctx = inputs["ctx"].cpu().numpy()
    assert ctx.shape == (1, 2, c["n_w"], c["n_d"])
    np.testing.assert_array_equal(ctx[0], c["x_obs"][ci])
    np.testing.assert_array_equal(ctx[0, 0], c["obs_mask"][ci])
    assert not np.array_equal(ctx[0, 1], ctx[0, 0]), (
        "this cohort's feature hole vanished from the contract; the assertion above is "
        "satisfied by a per-cell mask too"
    )
    # and the whole held-out forward still assembles at two channels
    log_pi, mu, sigma = entry._forward_mixture(entry.models_[0], inputs)
    assert log_pi.shape == mu.shape == sigma.shape == (1, c["n_w"], c["n_d"], TINY.n_components)


def test_level_fields_reaches_the_contract(backend_name):
    """``fit(level_fields=...)`` is delivered, and the channel really is a level.

    A snapshot field such as `case_reserve` has no meaningful difference - the
    difference is the case movement, while the outstanding level is the
    informative quantity - so the contract must carry it undifferenced. Checked
    against the same contract built WITHOUT the declaration, so the assertion is
    about the kwarg rather than about arithmetic that would hold either way:
    the level channel differenced along dev must reproduce the increment
    channel, at the cells the incremental form calls usable.
    """
    from ibnr.kernels.nn_contract import nn_data

    t = featured_triangle(backend_name)
    entry = NNTransformer().fit(
        t,
        loss_field="paid_loss",
        feature_fields=(LEVEL,),
        level_fields=(LEVEL,),
        config=TINY,
        seed=0,
    )
    c = entry.contract_
    assert c["fields"] == ["paid_loss", LEVEL]
    assert c["field_kinds"] == ("increment", "level")

    as_incr = nn_data(t, loss_field="paid_loss", feature_fields=(LEVEL,))
    assert as_incr["field_kinds"] == ("increment", "increment")
    usable = as_incr["x_obs"][:, 1, :, 1:]
    np.testing.assert_allclose(
        np.diff(c["x"][:, 1], axis=-1)[usable],
        as_incr["x"][:, 1, :, 1:][usable],
        rtol=1e-9,
        atol=1e-12,
    )
    # a level cell needs no predecessor, so it is usable one dev earlier
    assert c["x_obs"][:, 1].sum() > as_incr["x_obs"][:, 1].sum()
    pred = entry.predict(segment=SEG0, seed=0)
    assert np.isfinite(pred.samples).all()


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
