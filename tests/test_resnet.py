"""gallery.nn.resnet: network, no-leak input construction, entry contract,
held-out wiring. Skips cleanly when torch is not installed.

What this file protects, in three layers:

1. **Network.** The residual conv encoder emits a valid mixture density at
   every cell (normalized weights, positive sigmas) and can overfit one batch
   (gradients flow through the conv stack, the mask channels and the MDN head).
   The **receptive-field no-leak tests are the load-bearing ones for a conv
   body**: convolutions see the whole grid - there is no attention mask to
   hide a cell architecturally - so the ONLY thing standing between the model
   and the future is the input construction zeroing every non-context value.
   Poisoning a beyond-cutoff cell must leave the output bit-identical, and so
   must poisoning a channel the mask leaves off at a cell another channel is
   on at, since observedness is per channel.
2. **Entry.** PredictiveDistribution layout, anchoring of fully developed
   origins, fit/predict determinism under an explicit seed, and the
   per-channel conditioning: the context reaching the network during fit()
   never claims a feature the contract has no usable value for, a promoted
   rollout cell advances the TARGET channel's flag alone, and
   ``level_fields`` reaches the contract as an undifferenced channel.
3. **Held-out wiring.** The transformer's exact pattern over the shared
   per-cohort adapter: draw shape/seed/anchor, the pinned-dev asymmetry
   (draws point-mass, density refused), wrong-cohort refusal, and
   ``densities.check_normalization`` over the AMOUNT space through
   ``log_lik_at`` - the only check that catches a wrong standardization
   Jacobian. The draw path is additionally pinned by a moment check against
   the analytic mixture mean, which is what catches a swapped
   un-standardization order (``(z + mean) * std`` instead of
   ``z * std + mean``) that every shape/seed test would wave through.

``TINY`` is a deliberately under-powered config so the suite stays fast;
nothing here asserts predictive quality. Entry-level tests run on BOTH ibis
backends. The held-out fixture is the same 6x6 full square as the
transformer's: as_of at diagonal 6 leaves the next diagonal with 5 cells at
dev steps {2..6}, of which dev 6 is PINNED - one fixture exercises both sides
of the pinned asymmetry.
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
from ibnr.gallery.nn.resnet.config import ResNetConfig  # noqa: E402
from ibnr.gallery.nn.resnet.model import ResNet  # noqa: E402
from ibnr.gallery.nn.resnet.network import TriangleResNet, mdn_nll  # noqa: E402
from ibnr.kernels.densities import MEASURES, check_normalization  # noqa: E402
from ibnr.kernels.forecast import logmeanexp  # noqa: E402
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal  # noqa: E402

from .conftest import BACKENDS, make_multiline_triangle  # noqa: E402

TINY = ResNetConfig(
    channels=16,
    n_blocks=2,
    n_groups=4,
    dropout=0.0,
    n_components=2,
    lob_embedding_dim=4,
    batch_size=8,
    max_epochs=3,
    patience=5,
    ensemble_size=2,
    n_draws=50,  # rollout draws (predict)
    heldout_n_draws=50,  # held-out diagonal draws (predict_at); 10,000 by default
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


def synthetic_triangle(backend_name):
    """Two-LOB full squares (so ``realized_ultimates`` has outcomes)."""
    cum = _matrices()
    prem = {f"lob_{k}": np.full(6, 1000.0) for k in range(2)}
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    return make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)


def two_field_triangle(backend_name, *, feature_field="reported_loss", drop_first_dev=False):
    """The same squares plus a SECOND field, so F = 2 fits are exercised.

    ``make_multiline_triangle`` emits one loss field per call, so the two are
    built separately and concatenated (the idiom tests/test_deeptriangle.py
    uses). ``drop_first_dev`` removes the feature's first dev column, which
    makes its usable increments a STRICT subset of the target's - dev steps 1
    and 2 lose theirs, one for want of a cell and the next for want of a
    predecessor. That asymmetry is what per-channel observedness exists to
    represent, and a mask built from the target's alone would claim both cells.
    """
    cum = _matrices()
    feat = cum * 1.15  # a correlated second field; nothing here reads its level
    if drop_first_dev:
        feat = feat.copy()
        feat[:, :, 0] = np.nan
    prem = np.full(6, 1000.0)
    target = make_multiline_triangle(
        backend_name,
        {"lob_0": cum[0], "lob_1": cum[1]},
        premium_by_lob={"lob_0": prem, "lob_1": prem},
        start_year=START,
    )
    feature = make_multiline_triangle(
        backend_name,
        {"lob_0": feat[0], "lob_1": feat[1]},
        loss_field=feature_field,
        start_year=START,
    )
    df = pd.concat([target.execute(), feature.execute()], ignore_index=True)
    return Triangle.from_long(df, measure="cumulative", backend=backend_name)


# -- network -------------------------------------------------------------------


def test_registered():
    """The entry self-registers under its designed name and family."""
    assert "resnet" in gallery.list()
    assert gallery.get("resnet").family == "nn"


def test_forward_shapes_and_validity():
    """The MDN head is a valid density at every cell of a NON-square grid
    (n_w != n_d catches a W/D transposition in the head's permute/reshape):
    one (log_pi, mu, sigma) triple per (batch, origin, dev, component), sigma
    strictly positive, mixture weights normalized (logsumexp == 0). The context
    mask is PER CHANNEL - (B, F, W, D), one flag per value channel."""
    cfg = TINY
    torch.manual_seed(0)
    model = TriangleResNet(cfg, n_lob=3, n_features=2, n_w=5, n_d=4)
    b = 7
    x = torch.randn(b, 2, 5, 4)
    ctx = torch.rand(b, 2, 5, 4) > 0.5
    lob = torch.randint(0, 3, (b,))
    prem = torch.randn(b)
    cutoff = torch.randint(1, 8, (b,))
    log_pi, mu, sigma = model(x, ctx, lob, prem, cutoff)
    assert log_pi.shape == mu.shape == sigma.shape == (b, 5, 4, cfg.n_components)
    assert (sigma > 0).all()
    # atol 1e-5 in float32: log_softmax normalization is exact up to fp error.
    torch.testing.assert_close(log_pi.logsumexp(dim=-1), torch.zeros(b, 5, 4), atol=1e-5, rtol=0)


def test_config_rejects_indivisible_groups():
    """GroupNorm needs n_groups | channels; the config refuses at construction
    rather than failing inside torch with a shape error at fit time."""
    with pytest.raises(ValueError, match="n_groups"):
        ResNetConfig(channels=16, n_groups=5)


def test_the_parameter_count_is_independent_of_the_grid():
    """A conv body has no positional embedding tables, so its count does not
    depend on n_w/n_d - which is what makes a SINGLE disclosed number meaningful
    on this card at all, where the transformer's card has to quote two.

    The disclosed figure itself is pinned in
    ``tests/test_nn_parameter_counts.py``, which reads every bolded count out of
    every NN card and rebuilds it; that used to live here and covered only this
    entry, while ``transformer`` and ``mdn`` quoted numbers nothing checked (and
    ``mdn`` quoted a transformer figure the transformer card had retracted).
    What stays here is the claim that is resnet's alone.
    """
    counts = {
        (n_w, n_d): sum(
            p.numel()
            for p in TriangleResNet(
                ResNetConfig(), n_lob=4, n_features=1, n_w=n_w, n_d=n_d
            ).parameters()
        )
        for n_w, n_d in ((10, 10), (6, 6), (12, 8))
    }
    assert len(set(counts.values())) == 1, f"count varies with the grid: {counts}"


def test_overfit_one_batch():
    """The standard "can it learn at all" check: gradients flow end to end
    through the conv stack, the mask/distance channels and the MDN head, so
    the loss drops materially on a single fixed batch. A model that cannot
    overfit one batch has a wiring bug (detached tensor, wrong mask) that no
    accuracy metric would localize."""
    torch.manual_seed(3)
    cfg = TINY
    model = TriangleResNet(cfg, n_lob=1, n_features=1, n_w=4, n_d=4)
    x = torch.randn(8, 1, 4, 4)
    ctx = torch.zeros(8, 1, 4, 4, dtype=torch.bool)  # (B, F, W, D), one flag per channel
    ctx[:, :, :, :2] = True  # first two dev lags are context; the rest are targets
    tgt = ~ctx[:, 0]  # the loss scores the TARGET channel's cells
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


def test_receptive_field_no_leak_beyond_cutoff():
    """THE load-bearing test for a conv body. A 3x3 conv stack's receptive
    field covers the whole grid - unlike attention there is no mask argument
    hiding future cells architecturally - so the only protection against
    conditioning on the future is the input construction: every value channel
    must be zeroed outside the context mask before the first convolution
    (``x * flags`` in the forward), with the masks themselves supplied as input
    channels so "zeroed" stays distinguishable from "a zero increment".

    Poison EVERY beyond-cutoff cell in EVERY value channel with 1e6 and the
    output must be bit-identical everywhere - in particular at the
    pre-cutoff-conditioned target cells the training loss scores. Zero
    tolerance: 1e6 * 0.0 == 0.0 exactly, so any nonzero difference means a
    poisoned value reached a convolution.

    Mutation this must catch: drop the ``* flag`` gating (un-zero the future
    cells) - the poison then flows through the stem into every prediction.
    """
    torch.manual_seed(0)
    n_w = n_d = 6
    model = TriangleResNet(TINY, n_lob=2, n_features=2, n_w=n_w, n_d=n_d)
    model.eval()
    b = 3
    cal = torch.arange(n_w)[:, None] + torch.arange(n_d)[None, :] + 1  # (W, D) 1-based diagonal
    cut = 4
    cutoff = torch.full((b,), cut, dtype=torch.long)
    ctx = (cal <= cut).expand(b, 2, n_w, n_d)  # every channel conditions on diagonals 1..4
    x = torch.randn(b, 2, n_w, n_d)
    lob = torch.randint(0, 2, (b,))
    prem = torch.randn(b)
    with torch.no_grad():
        base = model(x, ctx, lob, prem, cutoff)
        x_poisoned = x.clone()
        x_poisoned[:, :, cal > cut] = 1e6  # all channels, every beyond-cutoff cell
        poisoned = model(x_poisoned, ctx, lob, prem, cutoff)
    assert bool((x_poisoned != x).any())  # the poison really landed
    for a, p in zip(base, poisoned, strict=True):
        torch.testing.assert_close(a, p, atol=0.0, rtol=0.0)  # bit-identical


def test_a_masked_off_channel_cannot_reach_the_output():
    """The per-channel half of the same guarantee, and the one the test above
    cannot make: there it is a whole CELL that is off, so any gating rule
    passes it.

    Observedness is per channel (``nn_contract``'s ``x_obs``) - a feature can
    be missing at a cell whose target is observed, where the contract's value
    is a padding zero. So poison every (channel, cell) the mask leaves off,
    including cells another channel is on at, and the output must be
    bit-identical. Zero tolerance for the same reason as above: 1e6 * 0.0 is
    exactly 0.0, so any difference means a poisoned value reached a
    convolution.

    Mutation this must catch: gating every channel with one flag
    (``x * flags[:, :1]``), which passes the cutoff test - where the channels
    share a mask - while reading an absent feature as an observed zero.
    """
    torch.manual_seed(0)
    n_w = n_d = 6
    model = TriangleResNet(TINY, n_lob=2, n_features=2, n_w=n_w, n_d=n_d)
    model.eval()
    b = 3
    gen = torch.Generator().manual_seed(1)
    ctx = torch.rand(b, 2, n_w, n_d, generator=gen) > 0.4
    assert bool((ctx[:, 0] != ctx[:, 1]).any())  # the channels genuinely disagree
    x = torch.randn(b, 2, n_w, n_d)
    lob = torch.randint(0, 2, (b,))
    prem = torch.randn(b)
    cutoff = torch.full((b,), 4, dtype=torch.long)
    with torch.no_grad():
        base = model(x, ctx, lob, prem, cutoff)
        x_poisoned = torch.where(ctx, x, torch.full_like(x, 1e6))
        poisoned = model(x_poisoned, ctx, lob, prem, cutoff)
    assert bool((x_poisoned != x).any())  # the poison really landed
    for a, p in zip(base, poisoned, strict=True):
        torch.testing.assert_close(a, p, atol=0.0, rtol=0.0)  # bit-identical


# -- entry ---------------------------------------------------------------------


def test_fit_predict_contract(backend_name):
    """The GalleryEntry contract for a single segment: per-origin ultimates
    plus a total, finite draws, a scorable summary against realized
    ultimates, and a fully developed first origin whose "predicted" ultimate
    is anchored at its observed latest cumulative rather than resampled."""
    t = synthetic_triangle(backend_name)
    entry = ResNet().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    pred = entry.predict(segment=SEG0, seed=0)
    assert pred.n_targets == 6 + 1  # per origin + total
    assert pred.targets["label"].tolist()[-1] == "total"
    assert np.isfinite(pred.samples).all()
    # Not all-positive: the MDN is unconstrained on the standardized scale, so
    # a small tail of negative draws is expected on an under-trained TINY fit.
    assert (pred.samples > 0).mean() > 0.95  # ultimates are overwhelmingly positive

    realized = entry.realized_ultimates(t, segment=SEG0)
    assert realized.shape == (7,)
    table = pred.summary(observed=realized)
    assert {"estimate", "se", "cv", "outcome", "percentile"} <= set(table.columns)

    # fully developed first origin: ultimate anchored at its observed value
    first = entry.contract_["latest_cum"][entry.cohort_index(SEG0), 0]
    np.testing.assert_allclose(pred.samples[:, 0], first)


def test_predict_caches_rollout_and_is_reproducible(backend_name):
    """Caching: the rollout is computed for ALL cohorts at once and a second
    segment reuses it (identity on the cached array) while returning its own
    draws. Determinism: a fresh fit with the same seed reproduces the draws
    exactly - member seeds, batching and sampling all derive from it."""
    t = synthetic_triangle(backend_name)
    entry = ResNet().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    a = entry.predict(segment=SEG0, seed=7)
    cached = entry._rollout_ults
    b = entry.predict(segment=SEG1, seed=7)
    assert entry._rollout_ults is cached  # same rollout reused across segments
    assert not np.allclose(a.samples, b.samples)

    entry2 = ResNet().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    a2 = entry2.predict(segment=SEG0, seed=7)
    np.testing.assert_allclose(a.samples, a2.samples)


# -- per-channel conditioning and level_fields ---------------------------------


def _spy_on_forward(monkeypatch) -> list[np.ndarray]:
    """Record every context mask the network is handed, in call order."""
    seen: list[np.ndarray] = []
    real_forward = TriangleResNet.forward

    def spy(self, x, context_mask, *rest, **kwargs):
        seen.append(context_mask.detach().cpu().numpy().copy())
        return real_forward(self, x, context_mask, *rest, **kwargs)

    monkeypatch.setattr(TriangleResNet, "forward", spy)
    return seen


def test_training_context_is_gated_per_channel(monkeypatch):
    """Every context mask reaching the network during fit() is (B, F, W, D),
    and the feature channel's flags never claim a cell that feature has no
    usable value at.

    The fixture drops the feature's first dev column, so its usable increments
    are a strict subset of the target's. A context built from ``obs_mask``
    broadcast over the channels - the shape this entry carried before
    per-channel observedness - would flag those cells as observed, and the
    contract's value there is a padding zero: the network would read an
    absent feature as a zero increment. Validation passes are recorded by the
    same spy and held to the same rule, one fixed cutoff later.
    """
    t = two_field_triangle("duckdb", drop_first_dev=True)
    seen = _spy_on_forward(monkeypatch)
    entry = ResNet().fit(
        t, loss_field="paid_loss", feature_fields=("reported_loss",), config=TINY, seed=0
    )
    c = entry.contract_
    feature_obs = c["x_obs"][:, 1]  # (n_c, n_w, n_d)
    assert (feature_obs == feature_obs[0]).all()  # both cohorts share the pattern
    assert not feature_obs[0][:, :2].any()  # dev steps 1-2 have no usable increment
    assert seen, "no forward pass was recorded - the spy is wired to the wrong method"

    for k, ctx in enumerate(seen):
        assert ctx.shape[1:] == (2, c["n_w"], c["n_d"])
        leaked = int((ctx[:, 1] & ~feature_obs[0][None]).sum())
        assert not leaked, (
            f"forward pass {k}: {leaked} feature-channel flag(s) sit where the feature "
            "has no usable value - the context is the target's mask over every channel, "
            "not each channel's own"
        )
    assert any(bool((ctx[:, 0] & ~ctx[:, 1]).any()) for ctx in seen), (
        "the two channels' contexts never differed, so the assertions above are vacuous"
    )


def test_rollout_promotes_only_the_target_channel(monkeypatch):
    """A promoted rollout cell raises CHANNEL 0's context flag and nothing else.

    The sampled value is a target increment and nothing simulates next year's
    features, so their flags must stay off. Promoting the whole cell - which is
    what one flag per cell forced - told the network the feature channels were
    OBSERVED there, where the contract holds padding: the rollout conditioned on
    a fabricated zero increment, and on one more of them per diagonal.

    Spied on the arguments the network actually received during ``predict()``,
    which is the only place the promotion is visible: every ultimate the rollout
    returns is finite and plausible either way.
    """
    t = two_field_triangle("duckdb")
    entry = ResNet().fit(
        t,
        loss_field="paid_loss",
        feature_fields=("reported_loss",),
        as_of=AS_OF,
        config=TINY,
        seed=0,
    )
    seen = _spy_on_forward(monkeypatch)  # after the fit: rollout passes only
    entry.predict(segment=SEG0, seed=0)
    assert len(seen) > 1, "the rollout made one pass, so no promotion happened to check"

    base = seen[0]
    for k, ctx in enumerate(seen):
        np.testing.assert_array_equal(
            ctx[:, 1],
            base[:, 1],
            err_msg=f"forward pass {k}: a feature-channel flag moved during the rollout",
        )
        assert not (base[:, 0] & ~ctx[:, 0]).any()  # the target's context only grows
    promoted = seen[-1][:, 0] & ~base[:, 0]
    assert promoted.any(), "no cell was promoted - the fixture has no future cells"
    assert not (seen[-1][:, 1] & promoted).any()


def test_heldout_inputs_carry_each_channels_own_flags():
    """The held-out forward conditions on the contract's ``x_obs`` for that
    cohort - the same per-channel mask fit() and the rollout use - rather than
    the target's mask over every channel."""
    t = two_field_triangle("duckdb", drop_first_dev=True)
    entry = ResNet().fit(
        t,
        loss_field="paid_loss",
        feature_fields=("reported_loss",),
        as_of=AS_OF,
        config=TINY,
        seed=0,
    )
    ci = entry.cohort_index(SEG0)
    c = entry.contract_
    ctx = entry._heldout_inputs(ci)["ctx"].cpu().numpy()
    assert ctx.shape == (1, 2, c["n_w"], c["n_d"])
    np.testing.assert_array_equal(ctx[0], c["x_obs"][ci])
    # non-vacuous: the feature is unusable at cells the target is observed at
    assert (ctx[0, 0] & ~ctx[0, 1]).any()


def test_level_fields_reaches_the_contract():
    """``fit(level_fields=...)`` is threaded to ``nn_data``, which carries that
    channel UNDIFFERENCED: an eval-date snapshot such as ``case_reserve`` is
    not an amount that accumulates, and its difference is the case *movement*
    rather than the outstanding level.

    Asserted on the channel's VALUES, not on the kwarg being accepted: the
    level channel must equal snapshot / premium at every cell, which a
    differenced channel cannot match.
    """
    t = two_field_triangle("duckdb", feature_field="case_reserve")
    entry = ResNet().fit(
        t,
        loss_field="paid_loss",
        feature_fields=("case_reserve",),
        level_fields=("case_reserve",),
        config=TINY,
        seed=0,
    )
    c = entry.contract_
    assert c["field_kinds"] == ("increment", "level")
    ci = entry.cohort_index(SEG0)
    np.testing.assert_allclose(c["x"][ci, 1], _matrices()[0] * 1.15 / 1000.0)
    # a level needs no predecessor, so every cell of a full square is usable
    assert c["x_obs"][ci, 1].all()


# -- held-out wiring -----------------------------------------------------------


@pytest.fixture(scope="module", params=BACKENDS)
def fitted(request):
    """One TINY pooled fit per backend, plus each cohort's next-diagonal cells.
    Module-scoped: the fit is the expensive part and every test here is
    read-only against it."""
    backend = request.param
    cum = _matrices()
    prem = np.full(6, 1000.0)
    pooled = make_multiline_triangle(
        backend,
        {"lob_0": cum[0], "lob_1": cum[1]},
        premium_by_lob={"lob_0": prem, "lob_1": prem},
        start_year=START,
    )
    entry = ResNet().fit(pooled, loss_field="paid_loss", as_of=AS_OF, config=TINY, seed=0)

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


def test_fixture_and_declarations(fitted):
    """Pin the structure every test below relies on - 5 scorable cells at dev
    steps {2..6} with exactly dev 6 pinned - and the scale declarations the
    base classes act on. If the triangle or the as_of drifts, this fails
    first and names the rot."""
    cells = fitted.cells0
    assert cells.n_cells == 5
    assert sorted(cells.frame["dev_lag"] // 12) == [2, 3, 4, 5, 6]
    pinned = fitted.entry.norm_["pinned"][0]  # target channel, (n_d,)
    np.testing.assert_array_equal(pinned, [False, False, False, False, False, True])
    assert _split(cells, pinned=True).n_cells == 1
    assert _split(cells, pinned=False).n_cells == 4

    entry = fitted.entry
    assert isinstance(entry, ScoresHeldout) and isinstance(entry, PredictsHeldout)
    assert entry.heldout_measure == "loss_ratio" and entry.heldout_measure in MEASURES
    assert entry.heldout_draw_scale == "incremental"
    view = entry.at_cohort(SEG0)
    assert view.heldout_measure == "loss_ratio"
    assert view.heldout_draw_scale == "incremental"


def test_predict_at_shape_seed_and_anchor(fitted):
    """Draw contract: (config.heldout_n_draws, n_cells); reproducible per seed;
    live cells have genuine spread; pinned cells are the exact point mass at
    anchor + premium * pooled dev mean; and the incremental-to-cumulative
    anchoring is the BASE CLASS's doing - ``predict_at`` must equal the native
    incremental draws plus each cell's training-diagonal predecessor.

    Mutation the anchor half must catch: ``heldout_draw_scale = "cumulative"``
    (or any skipped conversion) makes ``got == native``, off by every anchor -
    the anchors (~450-2000) dwarf the increments (~30-500), the
    CRPS-996-where-truth-is-3.4 bug class.
    """
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
    mean0 = entry.norm_["mean"][0]
    d0 = (frame["dev_lag"] // 12).to_numpy() - 1
    expected = frame["prev_value"].to_numpy() + 1000.0 * mean0[d0]
    assert (a[:, pinned_col].max(axis=0) == a[:, pinned_col].min(axis=0)).all()
    np.testing.assert_allclose(a[0, pinned_col], expected[pinned_col], rtol=1e-6)

    # the anchor conversion runs in the base class, exactly once
    view = entry.at_cohort(SEG0)
    idx = index_into(cells, view.contract_, field="paid_loss")
    native = view._draws_native(idx, rng=np.random.default_rng(11))
    np.testing.assert_allclose(a, native + idx.prev_value[None, :], rtol=1e-12)
    assert idx.prev_value.min() > 100.0
    assert np.abs(native).mean() < idx.prev_value.min()


def test_predict_at_draw_mean_matches_the_mixture(fitted):
    """Moment check on the draw path at one unpinned cell: the empirical mean
    of many native draws must match the analytic mean of the un-standardized
    ensemble mixture, ``premium * (mean0[d] + std0[d] * E_members[sum_k pi_k
    mu_k])``, within Monte Carlo error.

    This is what pins the UN-STANDARDIZATION ORDER: a swapped
    ``(z + mean0) * std0`` (or a dropped mean0/std0/premium factor) shifts
    the mean by orders of magnitude more than the tolerance, while every
    shape/seed/anchor test - and even the pinned-dev point-mass check, where
    std0 == 1 makes both orders agree - waves it through.
    """
    entry = fitted.entry
    view = entry.at_cohort(SEG0)
    one = replace(
        fitted.cells0,
        frame=fitted.cells0.frame[fitted.cells0.frame["dev_lag"] == 36].reset_index(drop=True),
    )
    assert one.n_cells == 1  # dev step 3: unpinned
    idx = index_into(one, view.contract_, field="paid_loss")
    d0 = int(idx.d[0]) - 1
    prem = float(idx.premium[0])
    mean0 = float(entry.norm_["mean"][0][d0])
    std0 = float(entry.norm_["std"][0][d0])

    log_pi, mu, sigma = entry._heldout_mixture(entry.cohort_index(SEG0), idx)  # (m, 1, K)
    pi = np.exp(log_pi[:, 0, :])
    m_z = (pi * mu[:, 0, :]).sum(axis=1)  # per-member mixture mean of z
    v_z = (pi * (sigma[:, 0, :] ** 2 + mu[:, 0, :] ** 2)).sum(axis=1) - m_z**2
    analytic_mean = prem * (mean0 + std0 * m_z.mean())
    # pooled-draw variance on the dollar scale: within- + between-member
    var_dollar = (prem * std0) ** 2 * (v_z.mean() + m_z.var())

    n_draws = 4000  # divisible by ensemble_size, so the member split is even
    keep = entry.config_
    # the HELD-OUT budget is what _draws_native spends; the rollout's n_draws
    # would leave this at the fixture's 50 and the moment check with 9x the
    # Monte Carlo error the tolerance below is written against
    entry.config_ = replace(keep, heldout_n_draws=n_draws)
    try:
        native = view._draws_native(idx, rng=np.random.default_rng(5))
    finally:
        entry.config_ = keep
    assert native.shape == (n_draws, 1)
    se = float(np.sqrt(var_dollar / n_draws))
    assert abs(float(native.mean()) - analytic_mean) < 6.0 * se


def test_predict_at_refuses_all_pinned_cells(fitted):
    """A request whose every cell is pinned would return a point mass in every
    column - not a predictive distribution - and must be refused loudly."""
    only_pinned = _split(fitted.cells0, pinned=True)
    with pytest.raises(ValueError, match="pinned"):
        fitted.entry.predict_at(only_pinned, field="paid_loss", seed=0)


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


def test_density_normalizes_over_the_amount_space(fitted):
    """Integrate the ensemble-average predictive density over the CUMULATIVE
    amount at one held-out cell and require mass 1 to 1e-4.

    This is the only check that catches a wrong change of variable
    (kernels/densities.py): dropping the ``-log std0[d]`` standardization
    Jacobian leaves a smooth, plausible, correctly-ranking density that
    integrates to 1/std0 - orders of magnitude off here. The integration runs
    through ``log_lik_at`` itself, so both Jacobians (standardization and
    premium) are on the path. Bounds come from the mixture parameters
    (+/- 12 component sd), keeping the interval tight enough for quad to see
    the peak.
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


def test_adapter_refuses_wrong_cohort(fitted):
    """(w, d) alone cannot identify a cell: lob_1's cells index cleanly into
    lob_0's adapter and would score the wrong cohort silently. The identity
    guard must catch it - through the view's public methods, so the guard is
    on the path callers actually use."""
    view0 = fitted.entry.at_cohort(SEG0)
    with pytest.raises(ValueError, match="was trained on"):
        view0.log_lik_at(fitted.cells1, field="paid_loss")
    with pytest.raises(ValueError, match="was trained on"):
        view0.predict_at(fitted.cells1, field="paid_loss", seed=0)
