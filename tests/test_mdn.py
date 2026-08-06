"""gallery.nn.mdn: the no-attention variant entry. Skips without torch.

What this file protects, in three layers:

1. **Network.** The per-cell MLP emits a valid mixture density at every cell
   (normalized weights, positive sigmas), gradients flow (overfit-one-batch),
   and - the entry's defining property - the context summary is MASKED: a value
   at a cell past the cutoff cannot influence any prediction, its own included.
   That no-leak gate is the analogue of the transformer's norm-stats poisoning
   test, checked here at the exact place this architecture could leak (the
   summary is computed from values, where the transformer's tokens are).
2. **Comparison integrity.** The entry only answers "what does attention buy"
   if everything except the encoder body is genuinely shared: the head loss and
   sampler must BE the transformer's (identity check, not behavioral), and the
   module must contain no attention or recurrence at all.
3. **Per-channel conditioning (0.5.4).** The mask is one flag per (channel,
   cell), not one per cell: a value is never consumed without its own
   channel's flag, each channel's summary mean divides by its own context
   count, the structural flags (the per-dev fraction, the origin row's flag
   vector) are channel 0's, the fit's context is ``x_obs`` under the calendar
   gate, and a promoted rollout cell gains channel 0's flag alone. With one
   channel every one of those reduces to the pre-0.5.4 function, so the
   fixtures here deliberately hole a feature field where the target is
   present - the only configuration in which the two differ at all.
4. **Held-out wiring.** The transformer's exact pattern over the shared
   per-cohort adapter: draw shape/seed/anchor semantics, the standardization
   Jacobian (pinned by integrating the density over the amount space - the only
   check that catches a wrong change of variable), the un-standardization order
   of draws, the pinned-dev asymmetry, and the wrong-cohort refusal.

The held-out fixture mirrors ``test_nn_heldout.py``: a 6x6 full square per LOB
with ``as_of`` at diagonal 6, so the next diagonal holds 5 cells at dev steps
{2..6}, of which dev 6 is PINNED (its only observation sits on the validation
diagonal; dev 5 keeps two context values because the normalizer pools both lob
cohorts). ``TINY`` is deliberately under-powered; nothing here asserts
predictive quality.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ibnr import Triangle  # noqa: E402
from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout  # noqa: E402
from ibnr.gallery.nn._scheme import norm_stats, splits  # noqa: E402
from ibnr.gallery.nn.mdn import model as mdn_model  # noqa: E402
from ibnr.gallery.nn.mdn import network as mdn_net  # noqa: E402
from ibnr.gallery.nn.mdn.config import MDNConfig  # noqa: E402
from ibnr.gallery.nn.mdn.model import MDN  # noqa: E402
from ibnr.gallery.nn.mdn.network import TriangleMDN  # noqa: E402
from ibnr.gallery.nn.transformer import network as transformer_net  # noqa: E402
from ibnr.kernels.densities import MEASURES, check_normalization  # noqa: E402
from ibnr.kernels.forecast import logmeanexp  # noqa: E402
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal  # noqa: E402

from .conftest import BACKENDS, make_multiline_triangle  # noqa: E402

TINY = MDNConfig(
    hidden_dim=32,
    n_layers=1,
    dropout=0.0,
    n_components=2,
    embedding_dim=4,
    lob_embedding_dim=4,
    batch_size=8,
    max_epochs=3,
    patience=5,
    ensemble_size=2,
    n_draws=50,  # rollout draws (predict)
    heldout_n_draws=400,  # cheap (one forward per member, no rollout) and large
    # enough for the draw-mean check to have negligible Monte Carlo error
)

START = 2000
AS_OF = "2005-12-31"  # diagonal 6 of a 6x6 square starting in 2000
SEG0 = {"company_code": "0001", "line_of_business": "lob_0"}
SEG1 = {"company_code": "0001", "line_of_business": "lob_1"}
#: dev steps of the next-diagonal cells whose per-dev normalizer is pinned
PINNED_DEVS = (6,)


def _matrices() -> np.ndarray:
    """(2, 6, 6) cumulative squares with a decaying incremental pattern -
    the same construction as test_nn_heldout's, so the pinned-dev structure
    the module docstring claims carries over."""
    rng = np.random.default_rng(0)
    n_w = n_d = 6
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    return np.cumsum(incr, axis=2)


def synthetic_triangle(backend_name, seed=0):
    """Two-LOB full square for the entry-contract tests (full so
    ``realized_ultimates`` has something to score)."""
    cum = _matrices()
    prem = {f"lob_{k}": np.full(6, 1000.0) for k in range(2)}
    return make_multiline_triangle(
        backend_name,
        {"lob_0": cum[0], "lob_1": cum[1]},
        premium_by_lob=prem,
        start_year=START,
    )


#: (origin, dev step) cells where lob_0 reports ``paid_loss`` and NOT
#: ``case_reserve``. Both are inside the ``AS_OF`` upper triangle, and each
#: costs the feature channel a second cell (the successor whose increment loses
#: its predecessor), so four cells separate the two channels' observedness.
FEATURE_HOLES = ((0, 0), (2, 1))


def feature_triangle(backend_name: str) -> Triangle:
    """Two-LOB triangle carrying ``paid_loss`` AND ``case_reserve`` (+ premium),
    with case_reserve HOLED on lob_0 at :data:`FEATURE_HOLES`.

    The holes are the whole point. Per-channel and per-cell conditioning are the
    same function wherever the channels share an observed pattern, so a fixture
    without them would let every assertion in the per-channel section pass on
    the pre-0.5.4 code. lob_1 is left whole, so the pool also holds a cohort
    whose channels agree.
    """
    paid = _matrices()
    reserve = 0.4 * paid
    for w, d0 in FEATURE_HOLES:
        reserve[0, w, d0] = np.nan
    prem = np.full(6, 1000.0)
    t_paid = make_multiline_triangle(
        backend_name,
        {"lob_0": paid[0], "lob_1": paid[1]},
        premium_by_lob={"lob_0": prem, "lob_1": prem},
        start_year=START,
    )
    t_reserve = make_multiline_triangle(
        backend_name,
        {"lob_0": reserve[0], "lob_1": reserve[1]},
        loss_field="case_reserve",
        start_year=START,
    )
    df = pd.concat([t_paid.execute(), t_reserve.execute()], ignore_index=True)
    return Triangle.from_long(df, measure="cumulative", backend=backend_name)


# -- network -------------------------------------------------------------------


def test_forward_shapes_and_validity():
    """The MDN head is a valid density at every cell: one (log_pi, mu, sigma)
    triple per (batch, origin, dev, component), sigma strictly positive, and
    mixture weights normalized (logsumexp == 0)."""
    cfg = TINY
    torch.manual_seed(0)
    model = TriangleMDN(cfg, n_lob=3, n_features=2, n_w=5, n_d=4)
    b = 7
    x = torch.randn(b, 2, 5, 4)
    ctx = torch.rand(b, 2, 5, 4) > 0.5
    lob = torch.randint(0, 3, (b,))
    prem = torch.randn(b)
    cutoff = torch.randint(1, 8, (b,))
    log_pi, mu, sigma = model(x, ctx, lob, prem, cutoff)
    assert log_pi.shape == mu.shape == sigma.shape == (b, 5, 4, cfg.n_components)
    assert (sigma > 0).all()
    torch.testing.assert_close(log_pi.logsumexp(dim=-1), torch.zeros(b, 5, 4), atol=1e-5, rtol=0)


def test_no_attention_no_recurrence():
    """The comparison only isolates attention if the encoder body is an MLP:
    no attention, no recurrence, no transformer layers anywhere in the module
    tree. A helpful contributor adding 'just one attention layer' would
    silently turn the architecture comparison into nothing."""
    model = TriangleMDN(TINY, n_lob=2, n_features=1, n_w=4, n_d=4)
    banned = (torch.nn.MultiheadAttention, torch.nn.TransformerEncoderLayer, torch.nn.RNNBase)
    for module in model.modules():
        assert not isinstance(module, banned), f"{type(module).__name__} in the mdn body"


def test_head_helpers_are_the_transformer_implementations():
    """``mdn_nll``/``mdn_sample`` are IMPORTED from the transformer, not
    copied: identity (``is``), not behavior, so the head loss and sampler
    cannot drift between the two arms of the comparison."""
    assert mdn_net.mdn_nll is transformer_net.mdn_nll
    assert mdn_net.mdn_sample is transformer_net.mdn_sample


def test_context_summary_no_leak():
    """A cell past the cutoff must not influence ANY prediction - its own
    included. Poison one non-context cell with 1e6 and require bit-identical
    outputs everywhere; then poison a CONTEXT cell and require a change, so
    the test is shown to be live rather than comparing two ignored inputs.

    This is the mdn analogue of the transformer's norm-stats poisoning test:
    the only place this architecture reads values is the masked summary
    (cohort per-dev means + the origin's own row), so the ``x * flag`` gate is
    the entry's entire no-leak defense - drop it and every summary carries
    future values into the cells that are supposed to predict them.
    """
    torch.manual_seed(0)
    model = TriangleMDN(TINY, n_lob=2, n_features=2, n_w=5, n_d=5)
    model.eval()
    b = 3
    x = torch.randn(b, 2, 5, 5)
    cal = torch.arange(5)[:, None] + torch.arange(5)[None, :] + 1  # (W, D) diagonals
    ctx = (cal <= 3)[None, None].expand(b, 2, 5, 5).clone()  # context = diagonals 1..3
    lob = torch.zeros(b, dtype=torch.long)
    prem = torch.zeros(b)
    cutoff = torch.full((b,), 3, dtype=torch.long)
    with torch.no_grad():
        base = model(x, ctx, lob, prem, cutoff)

        # poison a future cell (diagonal 5 > cutoff 3, not in ctx)
        x_poisoned = x.clone()
        assert not ctx[0, :, 2, 2].any()
        x_poisoned[:, :, 2, 2] = 1e6
        leaked = model(x_poisoned, ctx, lob, prem, cutoff)
        for a, c in zip(base, leaked, strict=True):
            torch.testing.assert_close(a, c, atol=0.0, rtol=0.0)

        # guard the guard: a poisoned CONTEXT cell must change the output
        x_ctx = x.clone()
        assert ctx[0, :, 1, 1].all()
        x_ctx[:, :, 1, 1] = 1e6
        changed = model(x_ctx, ctx, lob, prem, cutoff)
    assert not torch.allclose(base[1], changed[1])


def test_forward_masks_each_channel_against_its_own_flag():
    """A value is never consumed without ITS OWN channel's flag.

    Poison a feature-channel value at a cell whose TARGET flag is on and whose
    feature flag is off: nothing may move. Then turn that one feature flag on
    and require the output to change, so the first half cannot pass by the
    input being ignored altogether.

    The mutation this catches is the pre-0.5.4 mask shape - one flag per cell,
    broadcast across channels - under which channel 0's flag admitted the
    feature's contract padding as an observed value. That padding is a zero the
    triangle never reported, which is the defect per-channel masking exists to
    remove.
    """
    torch.manual_seed(0)
    model = TriangleMDN(TINY, n_lob=2, n_features=2, n_w=5, n_d=5)
    model.eval()
    b = 3
    x = torch.randn(b, 2, 5, 5)
    lob = torch.zeros(b, dtype=torch.long)
    prem = torch.zeros(b)
    cutoff = torch.full((b,), 3, dtype=torch.long)
    ctx = torch.ones(b, 2, 5, 5, dtype=torch.bool)
    ctx[:, 1, 2, 2] = False  # the feature is missing here; the target is not

    with torch.no_grad():
        base = model(x, ctx, lob, prem, cutoff)
        poisoned = x.clone()
        poisoned[:, 1, 2, 2] = 1e6
        masked = model(poisoned, ctx, lob, prem, cutoff)
        for a, c in zip(base, masked, strict=True):
            torch.testing.assert_close(a, c, atol=0.0, rtol=0.0)

        # guard the guard: flag that one feature cell and the output must move
        live = ctx.clone()
        live[:, 1, 2, 2] = True
        changed = model(poisoned, live, lob, prem, cutoff)
    assert not torch.allclose(base[1], changed[1])


def test_a_per_cell_mask_is_refused_by_name():
    """A caller left on the pre-0.5.4 (B, W, D) mask must be told, not served.

    That mask BROADCASTS against ``x`` rather than raising whenever the batch
    and channel counts line up - here B = F = 2, so ``x * flags`` succeeds and
    every cohort in the batch is masked with another channel's flags. The rank
    is therefore checked rather than trusted.
    """
    b = n_f = 2
    model = TriangleMDN(TINY, n_lob=2, n_features=n_f, n_w=5, n_d=5)
    x = torch.randn(b, n_f, 5, 5)
    per_cell = torch.ones(b, 5, 5, dtype=torch.bool)
    # the silent path this guard closes: the product is well-formed
    assert (x * per_cell.to(x.dtype)).shape == x.shape
    with pytest.raises(ValueError, match="per channel"):
        model(
            x,
            per_cell,
            torch.zeros(b, dtype=torch.long),
            torch.zeros(b),
            torch.full((b,), 3, dtype=torch.long),
        )


def test_summary_counts_are_per_channel_and_the_flag_vectors_are_the_targets():
    """Read the MLP's own input vector back and rebuild the masked blocks.

    Two claims, and neither is visible at the output: each channel's summary
    mean divides by ITS OWN context count (a count taken from channel 0 - the
    pre-0.5.4 shape - divides a feature's masked sum by the wrong number
    wherever the channels' observedness differs), and the two places a single
    structural flag is needed (the per-dev context fraction, the origin row's
    flag vector) use channel 0's, the target's.

    The rebuild is against the network's input to ``body``, captured by a
    pre-hook, because the MLP is opaque afterwards: an output-level comparison
    can only show that two inputs differ, not which arithmetic produced them.
    """
    torch.manual_seed(0)
    n_f, n_w, n_d = 2, 5, 4
    model = TriangleMDN(TINY, n_lob=2, n_features=n_f, n_w=n_w, n_d=n_d)
    model.eval()
    x = torch.randn(1, n_f, n_w, n_d)
    ctx = torch.rand(1, n_f, n_w, n_d) > 0.4
    assert (ctx[0, 0] != ctx[0, 1]).any(), "the channels agree; the rebuild would be vacuous"

    captured: list[torch.Tensor] = []

    def grab(_module, args) -> None:
        captured.append(args[0])

    handle = model.body.register_forward_pre_hook(grab)
    try:
        with torch.no_grad():
            model(
                x,
                ctx,
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1),
                torch.ones(1, dtype=torch.long),
            )
    finally:
        handle.remove()
    feat = captured[0]  # (1, W, D, feat_dim)

    flags = ctx.to(x.dtype)
    counts = flags.sum(dim=2)  # (1, F, D) - each channel's own
    means = (x * flags).sum(dim=2) / counts.clamp(min=1.0)  # (1, F, D)
    cohort = torch.cat([means.flatten(1), counts[:, 0] / float(n_w)], dim=1)  # (1, F*D + D)
    width = n_f * n_d + n_d

    torch.testing.assert_close(feat[0, 0, 0, :width], cohort[0])
    # and it really is a COHORT summary: the same block at every cell
    assert (feat[..., :width] == feat[0, 0, 0, :width]).all()

    rows = torch.cat([(x * flags).permute(0, 2, 1, 3).flatten(2), flags[:, 0]], dim=2)
    torch.testing.assert_close(feat[0, :, 0, width : 2 * width], rows[0])


def test_overfit_one_batch():
    """The standard "can it learn at all" check: gradients flow end to end
    through the masked summary, embeddings and the MDN head, so the loss
    drops materially on a single fixed batch. A model that cannot overfit one
    batch has a wiring bug (detached tensor, wrong mask) that no accuracy
    metric would localize."""
    torch.manual_seed(3)
    model = TriangleMDN(TINY, n_lob=1, n_features=1, n_w=4, n_d=4)
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
        loss = mdn_net.mdn_nll(log_pi, mu, sigma, y, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert np.isfinite(losses).all()
    # best-so-far, not final: at this lr the tail of the trajectory wobbles
    assert min(losses) < losses[0] - 0.5, (
        f"no material overfit: {losses[0]:.3f} -> best {min(losses):.3f}"
    )


# -- entry ---------------------------------------------------------------------


def test_fit_predict_contract(backend_name):
    """The GalleryEntry contract for a single segment: per-origin ultimates
    plus a total, finite draws, a scorable summary against realized ultimates,
    and a fully developed first origin anchored at its observed cumulative."""
    t = synthetic_triangle(backend_name)
    entry = MDN().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    seg = SEG0
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
    first = entry.contract_["latest_cum"][entry.cohort_index(seg), 0]
    np.testing.assert_allclose(pred.samples[:, 0], first)


def test_predict_caches_rollout_and_is_reproducible(backend_name):
    """Caching: the rollout is computed for ALL cohorts at once and reused
    across segments (identity check). Determinism: a fresh fit with the same
    seed reproduces the draws exactly - member seeds, batch order and
    sampling all derive from the entry seed."""
    t = synthetic_triangle(backend_name)
    entry = MDN().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    a = entry.predict(segment=SEG0, seed=7)
    cached = entry._rollout_ults
    b = entry.predict(segment=SEG1, seed=7)
    assert entry._rollout_ults is cached  # same rollout reused across segments
    assert not np.allclose(a.samples, b.samples)

    entry2 = MDN().fit(t, loss_field="paid_loss", config=TINY, seed=0)
    a2 = entry2.predict(segment=SEG0, seed=7)
    np.testing.assert_allclose(a.samples, a2.samples)


# -- per-channel conditioning (0.5.4) ------------------------------------------


@pytest.fixture(scope="module", params=BACKENDS)
def featured(request):
    """One TINY pooled fit over :func:`feature_triangle`, at ``AS_OF``.

    Two channels whose observedness genuinely differs, and an ``as_of`` slice so
    the rollout has future cells to promote. Module-scoped: every test against
    it is read-only.
    """
    t = feature_triangle(request.param)
    entry = MDN().fit(
        t,
        loss_field="paid_loss",
        feature_fields=("case_reserve",),
        as_of=AS_OF,
        config=TINY,
        seed=0,
    )
    assert (entry.contract_["x_obs"][:, 0] != entry.contract_["x_obs"][:, 1]).any(), (
        "the fixture's channels share an observedness, so nothing here is testing "
        "per-channel behavior"
    )
    return entry


def test_fit_conditions_every_channel_on_its_own_cells(backend_name, monkeypatch):
    """What reaches the network during fit is ``x_obs`` under the calendar gate.

    ``tests/test_nn_training_context.py`` pins the calendar half of that mask for
    the family; this pins the channel half for this entry, on both passes.
    Training: each channel's own usable cells at or before the batch's augmented
    cutoff. Validation: each channel's own usable cells at or before the
    validation cutoff.

    The mutation it must catch is the pre-0.5.4 form ``obs_mask[idx] & gate``,
    which hands every channel the TARGET's observedness - so a cell the target
    reports and the feature does not arrives flagged, and the network reads the
    contract's padding zero as a reported case reserve. The fixture holes the
    feature where the target is present, and the loop asserts the two masks
    actually diverge on some batch, so the check cannot pass vacuously.
    """
    calls: list[tuple[np.ndarray, tuple | None]] = []
    live: dict[str, tuple | None] = {"batch": None}
    real_train = mdn_model.train_ensemble
    real_forward = TriangleMDN.forward

    def spy_train(n_cohorts, *, train_loss, **kwargs):
        def wrapped(model, idx, cutoffs):
            live["batch"] = (idx.cpu().numpy().copy(), cutoffs.cpu().numpy().copy())
            try:
                return train_loss(model, idx, cutoffs)
            finally:
                live["batch"] = None

        return real_train(n_cohorts, train_loss=wrapped, **kwargs)

    def spy_forward(self, x, context_mask, *rest, **kwargs):
        calls.append((context_mask.cpu().numpy().copy(), live["batch"]))
        return real_forward(self, x, context_mask, *rest, **kwargs)

    monkeypatch.setattr(mdn_model, "train_ensemble", spy_train)
    monkeypatch.setattr(TriangleMDN, "forward", spy_forward)
    entry = MDN().fit(
        feature_triangle(backend_name),
        loss_field="paid_loss",
        feature_fields=("case_reserve",),
        as_of=AS_OF,
        config=TINY,
        seed=0,
    )

    c = entry.contract_
    x_obs, cal = c["x_obs"], c["cal_idx"]
    _, _, val_cutoff = splits(c["obs_mask"], cal, TINY.val_diagonals)

    training = [(mask, batch) for mask, batch in calls if batch is not None]
    assert training, "no forward pass ran inside train_loss; the spy is wired to nothing"
    diverged = False
    for mask, (idx, cutoffs) in training:
        gate = cal[None, None] <= cutoffs[:, None, None, None]  # (B, 1, W, D)
        np.testing.assert_array_equal(
            mask,
            x_obs[idx] & gate,
            err_msg=(
                "the training context is not each channel's own usable cells at or "
                "before the augmented cutoff"
            ),
        )
        diverged |= bool((mask[:, 0] != mask[:, 1]).any())
    assert diverged, (
        "no training batch had a cell where the two channels' context differs, so the "
        "assertion above holds for the target-only mask too"
    )

    validation = [mask for mask, batch in calls if batch is None]
    assert validation, "no validation pass was recorded; early stopping's context is unchecked"
    for mask in validation:
        np.testing.assert_array_equal(
            mask,
            x_obs & (cal <= val_cutoff),
            err_msg="the validation context is not each channel's own training-window cells",
        )


def test_normalizer_estimates_each_channel_on_its_own_cells(featured):
    """A feature channel's per-dev mean and std come from the cells where THAT
    channel has a value.

    ``_scheme.norm_stats`` still accepts the target-only masks and reproduces
    the pre-0.5.4 numbers from them bit for bit, so this mutation is one
    argument wide and changes no shape anywhere. It is caught by rebuilding
    both forms: the fit must match the per-channel one AND differ from the
    target-only one. The second half is what makes the first non-vacuous, and
    the difference is the defect itself - the target-only form averages the
    feature's padding zeros in as though they were reported values.
    """
    entry = featured
    c = entry.contract_
    _, _, val_cutoff = splits(c["obs_mask"], c["cal_idx"], TINY.val_diagonals)
    gate = c["cal_idx"] <= val_cutoff
    per_channel = norm_stats(c["x"], c["x_obs"] & gate, c["x_obs"])
    target_only = norm_stats(c["x"], c["obs_mask"] & gate, c["obs_mask"])
    for key, expected in zip(("mean", "std", "pinned"), per_channel, strict=True):
        np.testing.assert_array_equal(entry.norm_[key], expected, err_msg=key)
    assert not np.array_equal(per_channel[0], target_only[0]), (
        "the two forms agree on this fixture, so matching the per-channel one proves "
        "nothing about which masks the fit passed"
    )


def test_rollout_promotes_only_the_target_channel(featured, monkeypatch):
    """A promoted rollout cell carries the sampled TARGET and no feature flag.

    Before 0.5.4 the rollout promoted the whole cell (``ctx = ctx | cells``)
    against a per-cell mask, so every cell it sampled also declared its feature
    channels observed - at the contract's standardized padding, a value no
    triangle ever reported. The feature flags must therefore stay exactly the
    contract's ``x_obs`` for the whole rollout, while channel 0's grow by cells
    that are all in the future region.

    Recorded through ``predict()``, the public path, and the last assertion
    requires promotion to have happened at all - otherwise a rollout that never
    sampled anything would satisfy every claim above.
    """
    entry = featured
    c = entry.contract_
    n_c = c["x"].shape[0]
    future = np.arange(c["n_d"])[None, None, :] >= c["latest_dev"][:, :, None]

    masks: list[np.ndarray] = []
    real_forward = TriangleMDN.forward

    def spy_forward(self, x, context_mask, *rest, **kwargs):
        masks.append(context_mask.cpu().numpy().copy())
        return real_forward(self, x, context_mask, *rest, **kwargs)

    monkeypatch.setattr(TriangleMDN, "forward", spy_forward)
    entry._rollout_key = None  # the cached rollout would make no forward call
    entry.predict(segment=SEG0, seed=0)
    entry._rollout_key = None  # leave no mask-recording rollout behind for other tests
    entry._rollout_ults = None
    assert masks, "predict() made no forward pass; the rollout never ran"

    promoted_any = False
    for mask in masks:
        chunk = mask.shape[0] // n_c
        np.testing.assert_array_equal(
            mask[:, 1:],
            np.repeat(c["x_obs"][:, 1:], chunk, axis=0),
            err_msg=(
                "a feature channel's flags moved during the rollout - next year's "
                "features are unobserved and promoting them presents padding as data"
            ),
        )
        obs = np.repeat(c["obs_mask"], chunk, axis=0)
        assert not (obs & ~mask[:, 0]).any(), "the rollout dropped an observed target cell"
        promoted = mask[:, 0] & ~obs
        assert not (promoted & ~np.repeat(future, chunk, axis=0)).any(), (
            "a promoted cell sits outside the future region"
        )
        promoted_any |= bool(promoted.any())
    assert promoted_any, "no cell was ever promoted, so the feature-flag claim is vacuous"


def test_level_fields_reach_the_contract_undifferenced(backend_name):
    """``fit(level_fields=...)`` is delivered, not merely accepted.

    A signature test proves a wire exists; this reads the channel back off two
    fits of the SAME triangle. The target channel must be untouched and the
    feature channel must change - from the period-to-period movement to the
    snapshot itself - with ``field_kinds`` saying which is which. On the
    hole-free cohort the level channel is exactly the running sum of the
    increment channel, which is what "undifferenced" means and what an inert
    keyword could not produce.
    """
    t = feature_triangle(backend_name)
    common = dict(loss_field="paid_loss", feature_fields=("case_reserve",), config=TINY, seed=0)
    incr = MDN().fit(t, **common)
    level = MDN().fit(t, level_fields=("case_reserve",), **common)

    assert incr.contract_["field_kinds"] == ("increment", "increment")
    assert level.contract_["field_kinds"] == ("increment", "level")
    np.testing.assert_allclose(level.contract_["x"][:, 0], incr.contract_["x"][:, 0])
    assert not np.allclose(level.contract_["x"][:, 1], incr.contract_["x"][:, 1])

    ci = level.cohort_index(SEG1)  # lob_1, the cohort with no feature holes
    assert level.contract_["x_obs"][ci, 1].all()
    np.testing.assert_allclose(
        level.contract_["x"][ci, 1],
        np.cumsum(incr.contract_["x"][ci, 1], axis=-1),
        rtol=1e-10,
    )


# -- held-out wiring -----------------------------------------------------------


@pytest.fixture(scope="module", params=BACKENDS)
def fitted(request):
    """One TINY pooled fit per backend, plus each cohort's next-diagonal cells.

    Module-scoped: the fit is the expensive part and every test here is
    read-only against it.
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
    entry = MDN().fit(pooled, loss_field="paid_loss", as_of=AS_OF, config=TINY, seed=0)

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


def test_fixture_shape_is_what_the_file_claims(fitted):
    """Pin the structure every other test relies on: 5 scorable cells at dev
    steps {2..6}, nothing excluded, and exactly dev 6 pinned. If the triangle
    or the as_of drifts, this fails first and names the rot."""
    cells = fitted.cells0
    assert cells.n_cells == 5
    assert sorted(cells.frame["dev_lag"] // 12) == [2, 3, 4, 5, 6]
    pinned = fitted.entry.norm_["pinned"][0]  # target channel, (n_d,)
    np.testing.assert_array_equal(pinned, [False, False, False, False, False, True])
    assert _split(cells, pinned=True).n_cells == 1
    assert _split(cells, pinned=False).n_cells == 4


def test_entry_declares_both_capabilities(fitted):
    """The scale declarations the base classes act on. ``heldout_measure``
    must be a real MEASURES key or the carry would fail on every call rather
    than here; both declarations must match the transformer's, or the two
    entries' held-out numbers stop being term-for-term comparable."""
    entry = fitted.entry
    assert isinstance(entry, ScoresHeldout) and isinstance(entry, PredictsHeldout)
    assert entry.heldout_measure == "loss_ratio" and entry.heldout_measure in MEASURES
    assert entry.heldout_draw_scale == "incremental"
    view = entry.at_cohort(SEG0)
    assert isinstance(view, ScoresHeldout) and isinstance(view, PredictsHeldout)
    assert view.heldout_measure == "loss_ratio"
    assert view.heldout_draw_scale == "incremental"


def test_heldout_inputs_condition_each_channel_on_its_own_cells(featured):
    """The held-out forward conditions on ``x_obs``, per channel - the same mask
    fit() builds, not the target's flag reused for every channel.

    Pre-0.5.4 this dict carried ``obs_mask``, so on a cohort whose feature is
    holed the density and the draws were formed against a context claiming a
    case reserve the triangle never reported. The cohort asserted on is one
    where the two channels genuinely disagree, so the equality has teeth.
    """
    entry = featured
    ci = entry.cohort_index(SEG0)
    c = entry.contract_
    ctx = entry._heldout_inputs(ci)["ctx"]
    assert tuple(ctx.shape) == (1, c["x"].shape[1], c["n_w"], c["n_d"])
    np.testing.assert_array_equal(ctx.cpu().numpy()[0], c["x_obs"][ci])
    assert (c["x_obs"][ci, 0] != c["x_obs"][ci, 1]).any()


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
    assert (a[:, pinned_col].max(axis=0) == a[:, pinned_col].min(axis=0)).all()
    np.testing.assert_allclose(a[0, pinned_col], expected[pinned_col], rtol=1e-6)


def test_predict_at_anchors_increments_through_the_base_class(fitted):
    """The anchor conversion is the base class's job and it must actually run.

    The entry draws INCREMENTS while the triangle is cumulative (anchors ~2
    orders of magnitude larger), so draws scored without the declared-scale
    conversion are wrong by the whole anchor - the CRPS-996-where-truth-is-3.4
    bug class. ``predict_at`` seeds ``default_rng(seed)`` exactly as the base
    does, so the native draws are reproducible and the assertion is exact.

    Mutation this must catch: ``heldout_draw_scale = "cumulative"`` (or any
    skipped conversion) makes ``got == native``, off by every anchor.
    """
    entry = fitted.entry
    cells = fitted.cells0
    view = entry.at_cohort(SEG0)
    idx = index_into(cells, view.contract_, field="paid_loss")
    native = view._draws_native(idx, rng=np.random.default_rng(11))
    got = entry.predict_at(cells, field="paid_loss", seed=11)
    anchor = idx.prev_value
    np.testing.assert_allclose(got, native + anchor[None, :], rtol=1e-12)
    # the shift is huge relative to the draws, so the mutant is unmistakable
    assert anchor.min() > 100.0
    assert np.abs(native).mean() < anchor.min()


def test_predict_at_unstandardizes_in_the_right_order(fitted):
    """A draw is ``premium * (z * std0[d] + mean0[d])``, in that order.

    The empirical mean of the draws at each live cell must match the mixture's
    own mean pushed through the same map (ensemble-averaged, + anchor), within
    a CLT bound. The wrong composition ``(z + mean0) * std0`` still yields
    smooth, finite, plausibly-sized draws - but shifts every column by
    ``premium * mean0 * (std0 - 1)``, hundreds of dollars here against a
    Monte Carlo error of a few, so the mutant is unmistakable.
    """
    entry = fitted.entry
    cells = _split(fitted.cells0, pinned=False)
    view = entry.at_cohort(SEG0)
    idx = index_into(cells, view.contract_, field="paid_loss")
    draws = entry.predict_at(cells, field="paid_loss", seed=11)  # (heldout_n_draws, 4)

    d0 = idx.d - 1
    mean0, std0 = entry.norm_["mean"][0][d0], entry.norm_["std"][0][d0]
    log_pi, mu, _ = entry._heldout_mixture(entry.cohort_index(SEG0), idx)
    mix_mean_z = (np.exp(log_pi) * mu).sum(axis=-1).mean(axis=0)  # (n_cells,) over members
    expected = idx.prev_value + idx.premium * (mean0 + std0 * mix_mean_z)
    mc_err = draws.std(axis=0) / np.sqrt(draws.shape[0])
    assert (np.abs(draws.mean(axis=0) - expected) < 6.0 * mc_err).all()


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


def test_log_lik_matches_independent_recomputation(fitted):
    """Recompute the density from scratch - own forward pass, standardization,
    mixture assembly and BOTH Jacobians written out by hand - and require
    agreement with ``log_lik_at`` to float32 accuracy.

    Everything after the shared fitted state (weights, norm stats) is
    independent: the conditioning context, the as_of cutoff, ``z``, the
    ``-log std0[d]`` standardization Jacobian and the ``-log premium`` measure
    carry. This is the only test that pins the DEV INDEXING on the density
    path: a density built with the wrong dev's ``std0`` in both ``z`` and the
    Jacobian is still self-consistently normalized (the normalization check
    passes on it by construction), still finite, still ranks - and is wrong at
    every cell. Other mutations this catches: dropping either Jacobian term,
    conditioning at the wrong cutoff, and scoring the cumulative value instead
    of the increment.
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
    ctx = torch.tensor(c["x_obs"][ci][None])  # per-channel; channel 0 IS obs here
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
    Bounds come from the mixture parameters (+/- 12 component sd)."""
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
    guard must catch it - through the view's public method, so the guard is
    on the path callers actually use."""
    view0 = fitted.entry.at_cohort(SEG0)
    with pytest.raises(ValueError, match="was trained on"):
        view0.log_lik_at(fitted.cells1, field="paid_loss")
    with pytest.raises(ValueError, match="was trained on"):
        view0.predict_at(fitted.cells1, field="paid_loss", seed=0)
