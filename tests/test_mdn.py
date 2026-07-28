"""gallery.nn.mdn: the no-attention ablation entry. Skips without torch.

What this file protects, in three layers:

1. **Network.** The per-cell MLP emits a valid mixture density at every cell
   (normalized weights, positive sigmas), gradients flow (overfit-one-batch),
   and - the entry's defining property - the context summary is MASKED: a value
   at a cell past the cutoff cannot influence any prediction, its own included.
   That no-leak gate is the analogue of the transformer's norm-stats poisoning
   test, checked here at the exact place this architecture could leak (the
   summary is computed from values, where the transformer's tokens are).
2. **Ablation integrity.** The entry only answers "what does attention buy" if
   everything except the encoder body is genuinely shared: the head loss and
   sampler must BE the transformer's (identity check, not behavioral), and the
   module must contain no attention/recurrence to ablate around.
3. **Held-out wiring.** The transformer's exact pattern over the shared
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
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.entry import PredictsHeldout, ScoresHeldout  # noqa: E402
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
    n_draws=400,  # cheap here (one forward per member, no rollout) and large
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
    ctx = torch.rand(b, 5, 4) > 0.5
    lob = torch.randint(0, 3, (b,))
    prem = torch.randn(b)
    cutoff = torch.randint(1, 8, (b,))
    log_pi, mu, sigma = model(x, ctx, lob, prem, cutoff)
    assert log_pi.shape == mu.shape == sigma.shape == (b, 5, 4, cfg.n_components)
    assert (sigma > 0).all()
    torch.testing.assert_close(log_pi.logsumexp(dim=-1), torch.zeros(b, 5, 4), atol=1e-5, rtol=0)


def test_no_attention_no_recurrence():
    """The ablation is only an ablation if the encoder body really is an MLP:
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
    cannot drift between the two arms of the ablation."""
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
    ctx = (cal <= 3)[None].expand(b, 5, 5).clone()  # context = diagonals 1..3
    lob = torch.zeros(b, dtype=torch.long)
    prem = torch.zeros(b)
    cutoff = torch.full((b,), 3, dtype=torch.long)
    with torch.no_grad():
        base = model(x, ctx, lob, prem, cutoff)

        # poison a future cell (diagonal 5 > cutoff 3, not in ctx)
        x_poisoned = x.clone()
        assert not ctx[0, 2, 2]
        x_poisoned[:, :, 2, 2] = 1e6
        leaked = model(x_poisoned, ctx, lob, prem, cutoff)
        for a, c in zip(base, leaked, strict=True):
            torch.testing.assert_close(a, c, atol=0.0, rtol=0.0)

        # guard the guard: a poisoned CONTEXT cell must change the output
        x_ctx = x.clone()
        assert ctx[0, 1, 1]
        x_ctx[:, :, 1, 1] = 1e6
        changed = model(x_ctx, ctx, lob, prem, cutoff)
    assert not torch.allclose(base[1], changed[1])


def test_overfit_one_batch():
    """The standard "can it learn at all" check: gradients flow end to end
    through the masked summary, embeddings and the MDN head, so the loss
    drops materially on a single fixed batch. A model that cannot overfit one
    batch has a wiring bug (detached tensor, wrong mask) that no accuracy
    metric would localize."""
    torch.manual_seed(3)
    model = TriangleMDN(TINY, n_lob=1, n_features=1, n_w=4, n_d=4)
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


def test_predict_at_shape_seed_and_variance(fitted):
    """Draw contract: (config.n_draws, n_cells); reproducible per seed;
    live cells have genuine spread; pinned cells are the rollout-semantics
    point mass at anchor + premium * pooled dev mean, exactly."""
    entry = fitted.entry
    cells = fitted.cells0
    a = entry.predict_at(cells, field="paid_loss", seed=11)
    assert a.shape == (TINY.n_draws, 5)
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
    draws = entry.predict_at(cells, field="paid_loss", seed=11)  # (n_draws, 4)

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
    assert draws.shape == (TINY.n_draws, 5)

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
    ctx = torch.tensor(obs[None])
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
