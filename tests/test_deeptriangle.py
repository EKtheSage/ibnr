"""gallery.nn.deeptriangle: network, entry contract, held-out wiring.
Skips cleanly when torch is not installed.

What this file protects, in three layers:

1. **Network.** The GRU encoder/decoder emits a valid mixture density at every
   cell from BOTH heads (normalized weights, positive sigmas), origins are
   BITWISE independent sequences at inference (the card's sharp architectural
   difference from ``nn_transformer``, whose attention spans the whole grid),
   gradients flow end to end through the masked encoder/decoder dispatch
   (overfit-one-batch), and the company-embedding flag is genuinely delivered
   through the public fit path - a signature test proves a wire exists, a
   parameter-count test proves it is connected.
2. **Entry.** PredictiveDistribution layout, rollout caching, and seed
   reproducibility (fit twice same seed -> identical validation histories),
   on both ibis backends.
3. **Held-out wiring.** The transformer's at_cohort pattern applied to this
   entry: incremental draws anchored by the base class, the MDN density
   recomputed independently (both Jacobians by hand), normalization over the
   amount space through ``log_lik_at`` itself, the pinned-dev asymmetry, and
   the wrong-cohort refusal through the view. The draws' un-standardization
   ORDER gets its own moment check: ``(z + mean) * std`` and
   ``z * std + mean`` agree wherever std == 1 (every pinned dev), so only an
   unpinned-cell expectation separates them.

The fixture mirrors ``test_nn_heldout.py``: 6x6 full squares per LOB with
``as_of`` at diagonal 6, so the next diagonal holds 5 cells at dev steps
6, 5, 4, 3, 2 - of which dev 6 is PINNED. The triangles carry BOTH paid and
reported, so the auxiliary claims-outstanding task is exercised end to end.
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
from ibnr.gallery.nn.deeptriangle import model as dt_model  # noqa: E402
from ibnr.gallery.nn.deeptriangle import network as dt_net  # noqa: E402
from ibnr.gallery.nn.deeptriangle.config import DeepTriangleConfig  # noqa: E402
from ibnr.gallery.nn.deeptriangle.model import DeepTriangle  # noqa: E402
from ibnr.gallery.nn.deeptriangle.network import DeepTriangleGRU  # noqa: E402
from ibnr.gallery.nn.transformer import network as tf_net  # noqa: E402
from ibnr.gallery.nn.transformer.network import mdn_nll  # noqa: E402
from ibnr.kernels.densities import MEASURES, check_normalization  # noqa: E402
from ibnr.kernels.forecast import logmeanexp  # noqa: E402
from ibnr.kernels.holdout import CellIndex, index_into, next_diagonal  # noqa: E402

from .conftest import BACKENDS, make_multiline_triangle  # noqa: E402

TINY = DeepTriangleConfig(
    hidden_dim=16,
    dropout=0.0,
    n_components=2,
    lob_embedding_dim=4,
    company_embedding_dim=4,
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


def _matrices(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(paid, reported) cumulative (2, 6, 6) squares with a decaying
    incremental pattern; reported develops ~15% above paid so the derived
    outstanding increments are non-trivial."""
    rng = np.random.default_rng(seed)
    n_w = n_d = 6
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    rep_incr = incr * rng.lognormal(0.15, 0.05, size=(2, n_w, n_d))
    return np.cumsum(incr, axis=2), np.cumsum(rep_incr, axis=2)


def two_field_triangle(backend_name: str, paid, reported, prem, *, reported_holes=()) -> Triangle:
    """Multi-LOB triangle carrying paid_loss AND reported_loss (+ premium),
    so the entry's default channel pair (and the aux OS task) is exercised.

    ``reported_holes`` drops ``(lob, w, d)`` cells from the REPORTED channel
    alone - paid booked, reported absent. That is the only shape in which
    per-channel observedness differs from the target's, and the mart never
    shows it (paid and reported are booked on the same cells there), so the
    tests that need the distinction have to build it."""
    t_paid = make_multiline_triangle(
        backend_name,
        {"lob_0": paid[0], "lob_1": paid[1]},
        premium_by_lob={"lob_0": prem, "lob_1": prem},
        start_year=START,
    )
    t_rep = make_multiline_triangle(
        backend_name,
        {"lob_0": reported[0], "lob_1": reported[1]},
        loss_field="reported_loss",
        start_year=START,
    )
    rep = t_rep.execute()
    for lob, w, dev in reported_holes:
        drop = (
            (rep["line_of_business"] == lob)
            & (pd.to_datetime(rep["origin_period"]).dt.year == START + w)
            & (rep["dev_lag"] == 12 * (dev + 1))
        )
        assert drop.any(), f"no reported cell at {(lob, w, dev)} to punch out"
        rep = rep[~drop]
    df = pd.concat([t_paid.execute(), rep], ignore_index=True)
    return Triangle.from_long(df, measure="cumulative", backend=backend_name)


def synthetic_triangle(backend_name: str) -> Triangle:
    paid, reported = _matrices()
    return two_field_triangle(backend_name, paid, reported, np.full(6, 1000.0))


# -- network -------------------------------------------------------------------


def test_forward_shapes_and_validity():
    """Both MDN heads are valid densities at every cell: one (log_pi, mu,
    sigma) triple per (batch, origin, dev, component), sigma strictly
    positive, mixture weights normalized (logsumexp == 0)."""
    cfg = TINY
    torch.manual_seed(0)
    model = DeepTriangleGRU(cfg, n_lob=3, n_company=4, n_features=2, n_w=5, n_d=4)
    b = 7
    x = torch.randn(b, 2, 5, 4)
    ctx = torch.rand(b, 2, 5, 4) > 0.5  # per channel
    lob = torch.randint(0, 3, (b,))
    comp = torch.randint(0, 4, (b,))
    prem = torch.randn(b)
    target, aux = model(x, ctx, lob, comp, prem)
    for log_pi, mu, sigma in (target, aux):
        assert log_pi.shape == mu.shape == sigma.shape == (b, 5, 4, cfg.n_components)
        assert (sigma > 0).all()
        torch.testing.assert_close(
            log_pi.logsumexp(dim=-1), torch.zeros(b, 5, 4), atol=1e-5, rtol=0
        )


def test_origins_are_independent_sequences():
    """The card's bitwise claim, and the documented architectural difference
    from ``nn_transformer``: the recurrence runs along the DEV axis only, with
    origins folded into the batch (``tok.reshape(b * n_w, n_d, d_model)``), so
    nothing flows between origins inside a forward pass. Poison origin 0 on
    both of its input paths - channel values by 1e6 AND its context flags
    flipped - and every other origin must come back BITWISE identical
    (``torch.equal``, not allclose: the claim is bitwise or it is nothing), on
    BOTH heads, all three mixture parameters.

    Origin 0 itself must change. Without that half the test passes trivially
    on a model that ignores its inputs, which is the same verdict a genuinely
    independent architecture gives. The mutation this exists to catch: run the
    GRU over the flattened w*d sequence instead of per-origin, and origin 0's
    state leaks into every later origin.
    """
    cfg = TINY
    torch.manual_seed(0)
    model = DeepTriangleGRU(cfg, n_lob=3, n_company=4, n_features=2, n_w=5, n_d=4)
    model.eval()  # cfg.dropout is 0, but the claim is about inference
    b = 3
    x = torch.randn(b, 2, 5, 4)
    ctx = torch.rand(b, 2, 5, 4) > 0.5  # per channel
    lob = torch.randint(0, 3, (b,))
    comp = torch.randint(0, 4, (b,))
    prem = torch.randn(b)

    x_bad = x.clone()
    x_bad[:, :, 0, :] += 1e6
    ctx_bad = ctx.clone()
    ctx_bad[:, :, 0, :] = ~ctx_bad[:, :, 0, :]

    with torch.no_grad():
        base = model(x, ctx, lob, comp, prem)
        poisoned = model(x_bad, ctx_bad, lob, comp, prem)

    names = ("log_pi", "mu", "sigma")
    for head, before, after in (("target", base[0], poisoned[0]), ("aux", base[1], poisoned[1])):
        for name, was, now in zip(names, before, after, strict=True):
            assert torch.equal(was[:, 1:], now[:, 1:]), (
                f"{head}.{name}: origins 1.. moved when only origin 0's inputs changed"
            )
            assert not torch.equal(was[:, 0], now[:, 0]), (
                f"{head}.{name}: origin 0 did not move - the perturbation was inert, "
                "so the identity above proves nothing"
            )


def test_mixture_math_is_imported_not_copied():
    """The MDN loss/sampler live once, in the transformer's network module;
    deeptriangle's network must not carry a second implementation."""
    assert not hasattr(dt_net, "mdn_nll")
    assert not hasattr(dt_net, "mdn_sample")


def test_overfit_one_batch():
    """Gradients flow end to end through the masked encoder/decoder dispatch,
    the embeddings and the MDN head: the loss drops materially on one fixed
    batch, and BOTH GRU cells plus the aux head receive finite gradients. A
    model that cannot overfit one batch has a wiring bug (detached tensor,
    wrong mask) no accuracy metric would localize."""
    torch.manual_seed(3)
    cfg = TINY
    model = DeepTriangleGRU(cfg, n_lob=1, n_company=1, n_features=1, n_w=4, n_d=4)
    x = torch.randn(8, 1, 4, 4)
    ctx = torch.zeros(8, 1, 4, 4, dtype=torch.bool)
    ctx[:, :, :, :2] = True  # first two dev lags are context; the rest are targets
    tgt = ~ctx[:, 0]
    y = x[:, 0]
    lob = torch.zeros(8, dtype=torch.long)
    comp = torch.zeros(8, dtype=torch.long)
    prem = torch.zeros(8)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # one combined-loss backward first: every trainable block must be live
    (t_params, a_params) = model(x, ctx, lob, comp, prem)
    (mdn_nll(*t_params, y, tgt) + mdn_nll(*a_params, y, tgt)).backward()
    for name in ("encoder", "decoder"):
        grad = getattr(model, name).weight_ih.grad
        assert grad is not None and torch.isfinite(grad).all(), f"{name} got no gradient"
    assert model.aux_head.weight.grad is not None
    assert model.company_emb.weight.grad is not None
    opt.zero_grad()

    losses = []
    for _ in range(600):
        t_params, _ = model(x, ctx, lob, comp, prem)
        loss = mdn_nll(*t_params, y, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert np.isfinite(losses).all()
    # best-so-far, not final: at this lr the tail of the trajectory wobbles
    assert min(losses) < losses[0] - 0.5, (
        f"no material overfit: {losses[0]:.3f} -> best {min(losses):.3f}"
    )


def test_company_embedding_flag_changes_parameter_count():
    """The inert-parameter check, through the PUBLIC fit path: the flag must
    reach the constructed networks, so flag-off fits must have strictly fewer
    parameters and no embedding table. A mutation that hardcodes the flag in
    ``fit`` (or ignores it in the network) passes every signature test and
    fails here."""
    t = synthetic_triangle("duckdb")
    quick = replace(TINY, max_epochs=1, ensemble_size=1)
    on = gallery.fit("deeptriangle", t, as_of=AS_OF, config=quick, seed=0)
    off = gallery.fit(
        "deeptriangle", t, as_of=AS_OF, config=replace(quick, company_embedding=False), seed=0
    )
    n_on = sum(p.numel() for p in on.models_[0].parameters())
    n_off = sum(p.numel() for p in off.models_[0].parameters())
    assert n_on > n_off
    # the difference is exactly the embedding table + the wider cond_proj rows
    n_company = len(on.contract_["company_levels"])
    expected = quick.company_embedding_dim * (n_company + quick.hidden_dim)
    assert n_on - n_off == expected
    assert hasattr(on.models_[0], "company_emb")
    assert not hasattr(off.models_[0], "company_emb")


# -- entry ---------------------------------------------------------------------


def test_fit_predict_contract(backend_name):
    """The GalleryEntry contract for a single segment: per-origin ultimates
    plus a total, finite draws, a scorable summary against realized ultimates,
    and a fully developed first origin anchored at its observed latest
    cumulative rather than resampled."""
    t = synthetic_triangle(backend_name)
    entry = DeepTriangle().fit(t, config=TINY, seed=0)
    pred = entry.predict(segment=SEG0, seed=0)
    assert pred.n_targets == 6 + 1  # per origin + total
    assert pred.targets["label"].tolist()[-1] == "total"
    assert np.isfinite(pred.samples).all()
    # Not all-positive: the MDN is unconstrained on the standardized scale, so
    # a small tail of negative draws is expected on an under-trained TINY fit.
    assert (pred.samples > 0).mean() > 0.95

    realized = entry.realized_ultimates(t, segment=SEG0)
    assert realized.shape == (7,)
    table = pred.summary(observed=realized)
    assert {"estimate", "se", "cv", "outcome", "percentile"} <= set(table.columns)

    # fully developed first origin: ultimate anchored at its observed value
    first = entry.contract_["latest_cum"][entry.cohort_index(SEG0), 0]
    np.testing.assert_allclose(pred.samples[:, 0], first)

    # the aux task was live: OS normalization stats were derived and stored
    assert "aux_mean" in entry.norm_ and entry.norm_["aux_mean"].shape == (6,)


def test_predict_caches_rollout_and_is_reproducible(backend_name):
    """Caching: the rollout is computed for ALL cohorts at once and reused
    across segments (identity check). Determinism: a fresh fit with the same
    seed reproduces the draws exactly - training init, batching order and
    sampling are all seed-derived. The fit is sliced with ``as_of`` so future
    cells exist and the rollout genuinely samples (on the full square every
    origin is developed and the draws would be seed-independent constants)."""
    t = synthetic_triangle(backend_name)
    entry = DeepTriangle().fit(t, as_of=AS_OF, config=TINY, seed=0)
    a = entry.predict(segment=SEG0, seed=7)
    cached = entry._rollout_ults
    b = entry.predict(segment=SEG1, seed=7)
    assert entry._rollout_ults is cached  # same rollout reused across segments
    assert not np.allclose(a.samples, b.samples)

    entry2 = DeepTriangle().fit(t, as_of=AS_OF, config=TINY, seed=0)
    a2 = entry2.predict(segment=SEG0, seed=7)
    np.testing.assert_allclose(a.samples, a2.samples)
    # a different rollout seed re-rolls (cache keyed on (n_draws, seed))
    c = entry.predict(segment=SEG0, seed=8)
    assert not np.allclose(a.samples, c.samples)


def test_fit_same_seed_reproduces_val_history(backend_name):
    """Fit twice with one seed: identical per-member validation histories
    (exact float equality - same weights, same batches, same augmented
    cutoffs). A different seed must genuinely change the trajectory."""
    t = synthetic_triangle(backend_name)
    h1 = DeepTriangle().fit(t, config=TINY, seed=0).history_
    h2 = DeepTriangle().fit(t, config=TINY, seed=0).history_
    assert h1 == h2
    assert len(h1) == TINY.ensemble_size and all(len(m) > 0 for m in h1)
    h3 = DeepTriangle().fit(t, config=TINY, seed=1).history_
    assert h1 != h3


def test_registered_and_no_feature_fields_disables_aux():
    """The entry is registered under its declared name, and a fit without
    feature fields trains single-task (no aux stats) yet still predicts."""
    assert "deeptriangle" in gallery.list()
    paid, _ = _matrices()
    t = make_multiline_triangle(
        "duckdb",
        {"lob_0": paid[0], "lob_1": paid[1]},
        premium_by_lob={"lob_0": np.full(6, 1000.0), "lob_1": np.full(6, 1000.0)},
        start_year=START,
    )
    entry = DeepTriangle().fit(t, feature_fields=(), config=TINY, seed=0)
    assert "aux_mean" not in entry.norm_
    pred = entry.predict(segment=SEG0, seed=0)
    assert pred.n_targets == 7 and np.isfinite(pred.samples).all()


# -- per-channel conditioning --------------------------------------------------
#
# The entry conditions each channel on ITS OWN usable cells, not on the target's.
# Nothing downstream can see a violation: a feature read where it has no value
# contributes the contract's padding zero, which is a perfectly plausible
# standardized increment, so every shape, seed and score test stays green. These
# tests therefore watch what actually reaches the network.

#: reported punched out at origin 0, dev step 5. Its increment AND the dev-6
#: increment that differences against it both go unusable, and dev step 5 sits
#: on calendar diagonal 5 - the only diagonal the TINY fit's augmented cutoff
#: (always 4, one below the validation cutoff) ever scores. So the hole lands
#: where the auxiliary loss can see it rather than somewhere inert.
REPORTED_HOLE = ("lob_0", 0, 4)


def holed_triangle(backend_name: str = "duckdb") -> Triangle:
    paid, reported = _matrices()
    return two_field_triangle(
        backend_name, paid, reported, np.full(6, 1000.0), reported_holes=(REPORTED_HOLE,)
    )


def record_fit(monkeypatch, triangle, cfg, **fit_kwargs):
    """Run a real ``fit()`` through the public entry point and record what the
    training loop actually handed the network.

    ``.ctx`` is one record per forward pass - the context mask, tagged with the
    training batch in flight (None for the per-epoch validation pass).
    ``.batches`` is one record per ``train_loss`` call, carrying that batch's
    cohort ``idx``/``cutoffs`` and the ``mdn_nll`` masks in CALL order, which is
    how the target loss (first) is told from the auxiliary one (second).
    """
    ctx_calls: list[dict] = []
    batches: list[dict] = []
    live: dict[str, dict | None] = {"batch": None}
    real_train_ensemble = dt_model.train_ensemble
    real_forward = DeepTriangleGRU.forward
    real_nll = tf_net.mdn_nll

    def spy_train_ensemble(n_cohorts, *, train_loss, **kwargs):
        def wrapped(model, idx, cutoffs):
            batch = {
                "idx": idx.detach().cpu().numpy().copy(),
                "cutoffs": cutoffs.detach().cpu().numpy().copy(),
                "nll": [],
            }
            live["batch"] = batch
            try:
                return train_loss(model, idx, cutoffs)
            finally:
                live["batch"] = None
                batches.append(batch)

        return real_train_ensemble(n_cohorts, train_loss=wrapped, **kwargs)

    def spy_forward(self, x, context_mask, *rest, **kwargs):
        ctx_calls.append(
            {"ctx": context_mask.detach().cpu().numpy().copy(), "batch": live["batch"]}
        )
        return real_forward(self, x, context_mask, *rest, **kwargs)

    def spy_nll(log_pi, mu, sigma, y, mask):
        if live["batch"] is not None:
            live["batch"]["nll"].append(mask.detach().cpu().numpy().copy())
        return real_nll(log_pi, mu, sigma, y, mask)

    monkeypatch.setattr(dt_model, "train_ensemble", spy_train_ensemble)
    monkeypatch.setattr(DeepTriangleGRU, "forward", spy_forward)
    monkeypatch.setattr(tf_net, "mdn_nll", spy_nll)
    entry = DeepTriangle().fit(triangle, config=cfg, seed=0, **fit_kwargs)
    return SimpleNamespace(entry=entry, ctx=ctx_calls, batches=batches)


def test_training_context_is_per_channel_and_gated_by_the_cutoff(monkeypatch):
    """Every mask the network sees during fit is ``x_obs & (cal <= cutoff)``,
    channel by channel - the same conditioning the rollout and the held-out
    path present, which is the whole reason to train on it.

    Four claims: the mask carries a channel axis at all; it is exactly that
    expression on every training batch (so no channel is conditioned on a cell
    it has no value at, and none is masked that it does); the calendar gate
    excluded something on every batch, so the equality is not vacuous; and the
    per-epoch validation pass stops at the validation cutoff on every channel,
    since that is the number early stopping trusts.
    """
    cfg = replace(TINY, max_epochs=2, ensemble_size=1)
    rec = record_fit(monkeypatch, holed_triangle(), cfg, as_of=AS_OF)
    c = rec.entry.contract_
    cal, x_obs = np.asarray(c["cal_idx"]), np.asarray(c["x_obs"])
    _, _, val_cutoff = splits(c["obs_mask"], cal, cfg.val_diagonals)

    training = [r for r in rec.ctx if r["batch"] is not None]
    assert training, "no forward pass ran inside train_loss - the spy is wired wrong"
    for call in training:
        idx, cutoffs = call["batch"]["idx"], call["batch"]["cutoffs"]
        assert call["ctx"].shape == (len(idx), *x_obs.shape[1:]), (
            "the training context has no channel axis; a feature is being conditioned "
            "on wherever the TARGET has a value"
        )
        gate = cal[None, None] <= cutoffs[:, None, None, None]  # (B, 1, W, D)
        np.testing.assert_array_equal(
            call["ctx"],
            x_obs[idx] & gate,
            err_msg="the training context is not each channel's own cells at or before the cutoff",
        )
        assert (x_obs[idx] & ~gate).any(), "the gate excluded nothing; the equality is vacuous"

    validation = [r for r in rec.ctx if r["batch"] is None]
    assert validation, "no validation pass ran; early stopping gets its number there"
    for call in validation:
        np.testing.assert_array_equal(call["ctx"], x_obs & (cal <= val_cutoff)[None, None])


def test_aux_loss_is_scored_only_where_both_channels_are_real(monkeypatch):
    """The auxiliary target is a DIFFERENCE of two channels, so it exists only
    where both do. At a cell whose reported value is contract padding, the
    derived outstanding increment is the negative paid increment - a fabricated
    number, not a stale one - and training on it teaches the head a relationship
    that never happened.

    Checked on the masks ``mdn_nll`` actually received, with the non-vacuity
    half asserted too: the punched-out cell must be inside the target's mask,
    or a mutation that reuses the target mask for the aux head passes here.
    """
    cfg = replace(TINY, max_epochs=2, ensemble_size=1)
    rec = record_fit(monkeypatch, holed_triangle(), cfg, as_of=AS_OF)
    x_obs = rec.entry.contract_["x_obs"]
    both = x_obs[:, 1] & x_obs[:, 0]  # (n_c, n_w, n_d)

    scored = [b for b in rec.batches if b["nll"]]
    assert scored, "no batch reached the loss"
    witnessed = 0
    for batch in scored:
        assert len(batch["nll"]) == 2, (
            f"expected a target and an auxiliary loss per batch, got {len(batch['nll'])}"
        )
        target_mask, aux_mask = batch["nll"]
        idx = batch["idx"]
        np.testing.assert_array_equal(
            aux_mask,
            target_mask & both[idx],
            err_msg="the auxiliary loss scores cells where one of its two channels is padding",
        )
        witnessed += int((target_mask & ~both[idx]).any())
    assert witnessed, (
        "no batch scored a target cell whose reported channel is padding, so the equality "
        "above cannot tell the aux mask from the target's - the hole has stopped biting"
    )


def test_aux_normalizer_comes_from_cells_where_both_channels_are_real(monkeypatch):
    """The same rule one layer up: the outstanding channel's per-dev mean/std
    are estimated on ``x_obs[1] & x_obs[0]``, not on the target's mask.

    Verified against an independent ``norm_stats`` call, and against the
    target-masked statistics it must NOT equal - which is what makes this a
    test of the gating rather than of ``norm_stats``."""
    cfg = replace(TINY, max_epochs=1, ensemble_size=1)
    entry = DeepTriangle().fit(holed_triangle(), as_of=AS_OF, config=cfg, seed=0)
    c = entry.contract_
    cal, x_obs = np.asarray(c["cal_idx"]), np.asarray(c["x_obs"])
    _, _, val_cutoff = splits(c["obs_mask"], cal, cfg.val_diagonals)

    both = x_obs[:, 1] & x_obs[:, 0]
    assert (x_obs[:, 0] & ~both).any(), "the hole is not in the fit; nothing here is tested"
    os_ratio = (c["x"][:, 1] - c["x"][:, 0])[:, None]  # (n_c, 1, n_w, n_d)
    window = cal <= val_cutoff
    want = norm_stats(os_ratio, both & window, both)
    stale = norm_stats(os_ratio, x_obs[:, 0] & window, x_obs[:, 0])

    np.testing.assert_allclose(entry.norm_["aux_mean"], want[0][0])
    np.testing.assert_allclose(entry.norm_["aux_std"], want[1][0])
    np.testing.assert_array_equal(entry.norm_["aux_pinned"], want[2][0])
    assert not np.allclose(entry.norm_["aux_mean"], stale[0][0]), (
        "the two masks give the same statistics on this fixture, so the assertions above "
        "would pass on the target-masked version"
    )


def test_rollout_promotes_only_the_target_channels_flag(fitted, monkeypatch):
    """A sampled cell becomes context for the TARGET channel and for nothing
    else. Nothing was sampled for a feature channel there, so its value is the
    contract's padding zero; promoting the whole cell would present that zero
    as an observed increment - a fabricated observation at every forecast cell,
    which is exactly where it does most damage.

    Recorded off the network's own arguments during ``predict()``. The last two
    assertions are the ones with teeth: cells ARE promoted (or the constancy is
    vacuous), and the promoted cells are ones the feature channel does not
    hold (or whole-cell promotion would look identical here).
    """
    entry = fitted.entry
    c = entry.contract_
    n_c, n_f, n_w, n_d = c["x"].shape
    assert n_f == 2, "the default channel pair is what makes this claim observable"

    calls: list[np.ndarray] = []
    real_forward = DeepTriangleGRU.forward

    def spy_forward(self, x, context_mask, *rest, **kwargs):
        calls.append(context_mask.detach().cpu().numpy().copy())
        return real_forward(self, x, context_mask, *rest, **kwargs)

    monkeypatch.setattr(DeepTriangleGRU, "forward", spy_forward)
    cached = (entry._rollout_key, entry._rollout_ults)
    try:
        entry._rollout_key = None  # the cache would skip the rollout entirely
        entry.predict(segment=SEG0, n_draws=4, seed=5)
    finally:
        entry._rollout_key, entry._rollout_ults = cached

    assert calls, "no forward pass ran; the rollout was served from cache"
    rows = calls[0].shape[0]
    chunk = rows // n_c  # cohorts are repeat_interleave'd: row = cohort*chunk + draw
    feature_flags = np.repeat(c["x_obs"][:, 1:], chunk, axis=0)
    for step, ctx in enumerate(calls):
        assert ctx.shape == (rows, n_f, n_w, n_d), "the rollout context has no channel axis"
        np.testing.assert_array_equal(
            ctx[:, 1:],
            feature_flags,
            err_msg=f"rollout step {step} turned a feature channel's flag on at a sampled cell",
        )
    promoted = calls[-1][:, 0] & ~calls[0][:, 0]
    assert promoted.any(), "no cell was promoted, so the constancy above proves nothing"
    assert not (feature_flags[:, 0] & promoted).any(), (
        "every promoted cell is one the feature channel already holds, so whole-cell "
        "promotion would look identical here - the fixture has stopped discriminating"
    )


def test_level_fields_reaches_the_contract():
    """``fit(level_fields=...)`` must be DELIVERED, not merely accepted.

    A signature test passes on a fit that drops the argument, and so does every
    prediction test - an undifferenced channel is a plausible input either way.
    So: the contract records the kind, the channel carries the snapshot rather
    than its difference, the default is unchanged, and a level that names no
    feature channel is refused through this path (the contract's rule, reached
    only if the kwarg travels).
    """
    t = synthetic_triangle("duckdb")
    quick = replace(TINY, max_epochs=1, ensemble_size=1)
    entry = DeepTriangle().fit(
        t,
        feature_fields=("reported_loss",),
        level_fields=("reported_loss",),
        config=quick,
        seed=0,
    )
    c = entry.contract_
    assert c["field_kinds"] == ("increment", "level")
    # the level channel is the snapshot over premium, not the difference of two
    _, reported = _matrices()
    ci = entry.cohort_index(SEG0)
    np.testing.assert_allclose(c["x"][ci, 1, 0, :], reported[0][0] / 1000.0)

    plain = DeepTriangle().fit(t, config=quick, seed=0)
    assert plain.contract_["field_kinds"] == ("increment", "increment")
    assert not np.allclose(plain.contract_["x"][ci, 1], c["x"][ci, 1])

    with pytest.raises(ValueError, match="level_fields"):
        DeepTriangle().fit(t, level_fields=("earned_premium",), config=quick, seed=0)


# -- held-out wiring -----------------------------------------------------------


@pytest.fixture(scope="module", params=BACKENDS)
def fitted(request):
    """One TINY pooled fit per backend, plus each cohort's next-diagonal cells.

    Module-scoped: the fit is the expensive part and every test here is
    read-only against it."""
    backend = request.param
    paid, reported = _matrices()
    prem = np.full(6, 1000.0)
    pooled = two_field_triangle(backend, paid, reported, prem)
    entry = DeepTriangle().fit(pooled, as_of=AS_OF, config=TINY, seed=0)

    def cells_for(lob: str, matrix: np.ndarray):
        one = make_multiline_triangle(
            backend, {lob: matrix}, premium_by_lob={lob: prem}, start_year=START
        )
        return next_diagonal(one, as_of=AS_OF, fields="paid_loss", premium_field="earned_premium")

    return SimpleNamespace(
        entry=entry,
        cells0=cells_for("lob_0", paid[0]),
        cells1=cells_for("lob_1", paid[1]),
    )


def _split(cells, *, pinned: bool):
    """The next-diagonal cells at pinned (or unpinned) dev steps only."""
    dev_lags = [12 * d for d in PINNED_DEVS]
    mask = cells.frame["dev_lag"].isin(dev_lags)
    frame = cells.frame[mask if pinned else ~mask].reset_index(drop=True)
    return replace(cells, frame=frame)


def test_fixture_shape_is_what_the_file_claims(fitted):
    """Pin the structure every other test relies on: 5 scorable cells at dev
    steps {2..6}, nothing excluded, exactly dev 6 pinned, and both sides of
    the pinned/unpinned split non-empty."""
    cells = fitted.cells0
    assert cells.n_cells == 5
    assert sorted(cells.frame["dev_lag"] // 12) == [2, 3, 4, 5, 6]
    pinned = fitted.entry.norm_["pinned"][0]  # target channel, (n_d,)
    np.testing.assert_array_equal(pinned, [False, False, False, False, False, True])
    assert _split(cells, pinned=True).n_cells == 1
    assert _split(cells, pinned=False).n_cells == 4


def test_entry_declares_both_capabilities(fitted):
    """The scale declarations the base classes act on, on both the entry and
    the per-cohort view it hands out."""
    entry = fitted.entry
    assert isinstance(entry, ScoresHeldout) and isinstance(entry, PredictsHeldout)
    assert entry.heldout_measure == "loss_ratio" and entry.heldout_measure in MEASURES
    assert entry.heldout_draw_scale == "incremental"
    view = entry.at_cohort(SEG0)
    assert view.heldout_measure == "loss_ratio"
    assert view.heldout_draw_scale == "incremental"
    assert view.segment == SEG0


def test_heldout_inputs_condition_every_channel_on_its_own_cells(fitted):
    """The held-out forward pass conditions on ``x_obs``, per channel - the
    same mask fit() trains on, which is what makes the held-out diagonal the
    decoder's next step rather than a distribution shift."""
    entry = fitted.entry
    x_obs = entry.contract_["x_obs"][entry.cohort_index(SEG0)]  # (n_f, n_w, n_d)
    ctx = entry._heldout_inputs(entry.cohort_index(SEG0))["ctx"].cpu().numpy()
    assert ctx.shape == (1, *x_obs.shape)
    np.testing.assert_array_equal(ctx[0], x_obs)


def test_wrong_cohort_refused_through_the_view(fitted):
    """(w, d) alone cannot identify a cell: lob_1's cells index cleanly into
    lob_0's adapter and would score the wrong cohort silently. The identity
    guard must fire through the view's public methods."""
    view0 = fitted.entry.at_cohort(SEG0)
    with pytest.raises(ValueError, match="was trained on"):
        view0.log_lik_at(fitted.cells1, field="paid_loss")
    with pytest.raises(ValueError, match="was trained on"):
        view0.predict_at(fitted.cells1, field="paid_loss", seed=0)


def test_entry_needs_cells_that_name_a_cohort(fitted):
    """A bare CellIndex carries no segment values, so the entry-level methods
    refuse it with a pointer at at_cohort rather than guessing a cohort."""
    idx = index_into(fitted.cells0, fitted.entry.at_cohort(SEG0).contract_, field="paid_loss")
    with pytest.raises(TypeError, match="at_cohort"):
        fitted.entry.log_lik_at(idx)
    with pytest.raises(TypeError, match="at_cohort"):
        fitted.entry.predict_at(idx)


def test_predict_at_shape_seed_and_variance(fitted):
    """Draw contract: (config.heldout_n_draws, n_cells); reproducible per seed;
    live cells have genuine spread; pinned cells are the rollout-semantics point
    mass at anchor + premium * pooled dev mean, exactly."""
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


def test_predict_at_anchors_increments_through_the_base_class(fitted):
    """The anchor conversion is the base class's job and it must actually run:
    the entry draws INCREMENTS while the triangle is cumulative, so draws
    scored without the declared-scale conversion are wrong by the whole anchor
    (the CRPS-996-where-truth-is-3.4 bug class). Mutation this must catch:
    ``heldout_draw_scale = "cumulative"`` makes ``got == native``."""
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


def test_draws_unstandardize_in_the_right_order(fitted):
    """Draw moments pin the un-standardization ORDER at unpinned cells.

    ``z * std0 + mean0`` and ``(z + mean0) * std0`` agree wherever std0 == 1,
    which is every PINNED dev - so the point-mass check can never see this
    mutation and an unpinned-cell expectation is the only witness. The pooled
    draw mean must sit within Monte Carlo error of
    ``premium * (mean0 + std0 * mixture_mean)``; under the swapped order it is
    off by ``premium * mean0 * (1 - std0)``, orders of magnitude beyond the
    tolerance here (std0 ~ 0.03). Deterministic: the rng is seeded."""
    entry = fitted.entry
    view = entry.at_cohort(SEG0)
    cells = _split(fitted.cells0, pinned=False)
    idx = index_into(cells, view.contract_, field="paid_loss")
    native = view._draws_native(idx, rng=np.random.default_rng(3))  # (heldout_n_draws, 4)

    ci = entry.cohort_index(SEG0)
    log_pi, mu, sigma = entry._heldout_mixture(ci, idx)  # (n_members, n_cells, K)
    pi = np.exp(log_pi)
    m_member = (pi * mu).sum(axis=-1)  # (n_members, n_cells) mixture means
    v_member = (pi * (sigma**2 + mu**2)).sum(axis=-1) - m_member**2
    m_mix = m_member.mean(axis=0)  # draws split evenly across members
    tot_var = v_member.mean(axis=0) + m_member.var(axis=0)

    d0 = np.asarray(idx.d, dtype=int) - 1
    mean0, std0 = entry.norm_["mean"][0][d0], entry.norm_["std"][0][d0]
    assert (np.abs(std0 - 1.0) > 0.5).all()  # the mutation is visible here
    assert (np.abs(mean0) > 0.01).all()
    expected = idx.premium * (mean0 + std0 * m_mix)
    se = idx.premium * std0 * np.sqrt(tot_var / native.shape[0])
    err = np.abs(native.mean(axis=0) - expected)
    assert (err < 6.0 * se).all(), f"draw means off by {err} vs 6*se {6 * se}"


def test_predict_at_refuses_all_pinned_cells(fitted):
    """A request whose every cell is pinned would be a point mass in every
    column - not a predictive distribution - and must be refused loudly."""
    only_pinned = _split(fitted.cells0, pinned=True)
    with pytest.raises(ValueError, match="pinned"):
        fitted.entry.predict_at(only_pinned, field="paid_loss", seed=0)


def test_log_lik_refused_at_pinned_devs_but_draws_survive(fitted):
    """The documented asymmetry, both halves on the SAME cells: the density is
    refused at a pinned dev (naming the dev steps) while the draws still work,
    and the unpinned subset scores cleanly with genuinely differing members."""
    cells = fitted.cells0  # includes the pinned-dev cell
    with pytest.raises(ValueError, match=r"pinned dev step\(s\) \[6\]"):
        fitted.entry.log_lik_at(cells, field="paid_loss")
    draws = fitted.entry.predict_at(cells, field="paid_loss", seed=0)
    assert draws.shape == (TINY.heldout_n_draws, 5)

    ll = fitted.entry.log_lik_at(_split(cells, pinned=False), field="paid_loss")
    assert ll.shape == (TINY.ensemble_size, 4)
    assert np.isfinite(ll).all()
    assert not np.allclose(ll[0], ll[1])


def test_log_lik_matches_independent_recomputation(fitted):
    """Recompute the density from scratch - own forward pass, standardization,
    mixture assembly and BOTH Jacobians written out by hand - and require
    agreement with ``log_lik_at`` to float32 accuracy. Mutations this catches:
    dropping either Jacobian term (``- log std0[d]`` and ``- log premium``),
    standardizing in the wrong order, reading the wrong dev's normalizer, and
    scoring the cumulative value instead of the increment."""
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
    x_obs = c["x_obs"][ci]  # (n_f, n_w, n_d) - the entry conditions per channel
    xt = torch.tensor(x_norm[None], dtype=torch.float32)
    ctx = torch.tensor(x_obs[None])
    lob = torch.tensor([c["lob_idx"][ci]], dtype=torch.long)
    comp = torch.tensor([c["company_idx"][ci]], dtype=torch.long)
    prem_feat = torch.tensor(
        [(c["log_premium"][ci] - norm["prem_mean"]) / norm["prem_std"]], dtype=torch.float32
    )

    w0, d0 = idx.w - 1, idx.d - 1
    mean0, std0 = norm["mean"][0], norm["std"][0]
    increment = idx.value - idx.prev_value
    z = (increment / idx.premium - mean0[d0]) / std0[d0]
    expected = np.empty((len(entry.models_), idx.n_cells))
    with torch.no_grad():
        for m, model in enumerate(entry.models_):
            (log_pi, mu, sigma), _ = model(xt, ctx, lob, comp, prem_feat)
            lp = log_pi[0].numpy()[w0, d0]  # (n_cells, K)
            mu_ = mu[0].numpy()[w0, d0]
            sg = sigma[0].numpy()[w0, d0]
            comp_ll = -0.5 * ((z[:, None] - mu_) / sg) ** 2 - np.log(sg) - 0.5 * np.log(2.0 * np.pi)
            dens_z = np.log(np.exp(lp + comp_ll).sum(axis=1))  # small K: direct sum is fine
            # z -> ratio Jacobian, then ratio -> amount measure carry
            expected[m] = dens_z - np.log(std0[d0]) - np.log(idx.premium)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_density_normalizes_over_the_amount_space(fitted):
    """Integrate the ensemble-average predictive density over the CUMULATIVE
    amount at one held-out cell and require mass 1 to 1e-4. The only check
    that catches a wrong change of variable: dropping ``-log std0[d]`` leaves
    a smooth, plausible, correctly-ranking density that integrates to 1/std0.
    The integration runs through ``log_lik_at`` itself, so both Jacobians are
    on the path."""
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
    axis needs >= 2 rows, and the entry should say WHY, not fail downstream."""
    entry = fitted.entry
    cells = _split(fitted.cells0, pinned=False)
    keep = entry.models_
    entry.models_ = keep[:1]
    try:
        with pytest.raises(ValueError, match="ensemble_size"):
            entry.log_lik_at(cells, field="paid_loss")
    finally:
        entry.models_ = keep
