"""gallery.nn.transformer_ml: multi-line network, both dependence heads,
entry contract. Skips cleanly when torch is not installed.

What this file protects: ``nn_transformer_ml`` is the only NN entry that models
CROSS-LINE dependence. It fits on company cohorts (all lines of a company at
once, attention running across lines) and predicts the SUR target layout so it
is directly comparable to the statistical dependence baselines.

Two switchable dependence variants, both exercised here (CLAUDE.md: build both
design variants, keep changes switchable):

* ``"ar"`` - line-by-line autoregressive sampling; independent per-line heads.
* ``"joint"`` - a multivariate Gaussian-mixture head over lines, with absent
  lines marginalized out of the likelihood.

The invariants that matter are exactly the ones a multi-line model can get
wrong without any shape error: the joint likelihood must genuinely IGNORE
absent lines (companies do not write every line), the joint head must be able
to express correlation at all, and - for both variants - the grand total must
be the sum of line totals WITHIN each draw, so the diversified total carries
the dependence structure instead of being a sum of marginal means.

Conditioning is PER CHANNEL (0.5.4), and the same "no shape error" hazard
applies to it: one flag per cell instead of one per channel still runs, still
trains, and silently tells the network that a missing feature is a zero
increment - at every cell the rollout has just promoted, on every cell where
a feature is absent but the target is not. Three tests below pin it: the
network never consumes a value without that channel's own flag, fit builds
the context from the contract's ``x_obs``, and the rollout promotes the
TARGET channel alone.

Determinism and small-triangle caveats are as in ``test_nn_transformer.py``:
explicit seeds/generators throughout, and a deliberately under-powered config
(``tiny``) since nothing here asserts predictive quality. Entry tests run on
BOTH ibis backends.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ibnr import Triangle  # noqa: E402
from ibnr.gallery.nn import _scheme  # noqa: E402
from ibnr.gallery.nn.transformer_ml.config import TransformerMLConfig  # noqa: E402
from ibnr.gallery.nn.transformer_ml.model import NNTransformerML  # noqa: E402
from ibnr.gallery.nn.transformer_ml.network import (  # noqa: E402
    TriangleTransformerML,
    joint_mdn_nll,
    joint_mdn_sample,
)

from .conftest import make_multiline_triangle  # noqa: E402


def tiny(dependence: str) -> TransformerMLConfig:
    """Smallest config that still exercises every code path (a mixture needs
    >= 2 components, an ensemble >= 2 members); ``dependence`` selects the
    variant under test."""
    return TransformerMLConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0.0,
        n_components=2,
        dependence=dependence,
        batch_size=8,
        max_epochs=3,
        patience=5,
        ensemble_size=2,
        n_draws=40,
    )


START = 2000


def synthetic_triangle(backend_name, n_w=6, seed=0):
    """One company, two LOB full squares with a decaying incremental pattern.
    Full (not upper) so ``realized_ultimates`` has values to score against."""
    rng = np.random.default_rng(seed)
    n_d = n_w
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    cum = np.cumsum(incr, axis=2)
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    return make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)


def feature_triangle(
    backend_name,
    *,
    n_w=6,
    upper=False,
    feature_missing_dev=None,
    target_missing_dev=None,
    seed=0,
):
    """Same company/two-LOB shape, plus a ``case_reserve`` channel beside the
    paid target - the fixture anything per-channel needs, since
    ``make_multiline_triangle`` emits the loss and its premium only.

    ``upper`` emits the run-off staircase (w + d < n_w) instead of the full
    square, which is what leaves the rollout future cells to sample - on a
    full square every origin's anchor is already at the deepest dev and
    nothing is promoted.

    The two ``*_missing_dev`` knobs make the channels' observedness differ, in
    the two directions that matter and that a per-cell flag collapses:
    ``feature_missing_dev`` drops the feature from one dev column (target
    observed, feature not), and ``target_missing_dev`` drops the paid target
    from one dev column of the FIRST line only (feature observed, target not).
    """
    rng = np.random.default_rng(seed)
    n_d = n_w
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    # a case reserve runs off toward zero as the paid loss matures
    case_share = np.linspace(0.6, 0.05, n_d)
    rows = []
    for k in range(2):
        lob = f"lob_{k}"
        incr = 1000.0 * dev_level[None, :] * rng.lognormal(0.0, 0.1, size=(n_w, n_d))
        cum = np.cumsum(incr, axis=1)
        for w in range(n_w):
            for step in range(n_d):
                if upper and w + step >= n_w:
                    continue
                cell = (
                    "0001",
                    lob,
                    dt.date(START + w, 1, 1),
                    12 * (step + 1),
                    dt.date(START + w + step, 12, 31),
                )
                if not (k == 0 and step == target_missing_dev):
                    rows.append((*cell, "paid_loss", float(cum[w, step])))
                rows.append((*cell, "earned_premium", 1000.0))
                if step != feature_missing_dev:
                    rows.append((*cell, "case_reserve", float(cum[w, step] * case_share[step])))
    df = pd.DataFrame(
        rows,
        columns=[
            "company_code",
            "line_of_business",
            "origin_period",
            "dev_lag",
            "eval_date",
            "field",
            "value",
        ],
    )
    return Triangle.from_long(df, measure="cumulative", backend=backend_name)


def fit_with_case_reserve(triangle, dependence="ar", *, level=True, seed=0):
    """Fit the entry on the two-channel fixture: paid target + case reserve,
    the latter declared a LEVEL (an eval-date snapshot, whose difference would
    be the case movement) unless a test wants the differenced form."""
    return NNTransformerML().fit(
        triangle,
        loss_field="paid_loss",
        feature_fields=("case_reserve",),
        level_fields=("case_reserve",) if level else (),
        config=tiny(dependence),
        seed=seed,
    )


def record_context(monkeypatch):
    """Spy on every context mask handed to the network. Both heads funnel
    through ``encode``, so one patch covers ``ar`` and ``joint`` alike."""
    seen: list[np.ndarray] = []
    real = TriangleTransformerML.encode

    def spy(self, x, context_mask, *rest, **kwargs):
        seen.append(context_mask.detach().cpu().numpy().copy())
        return real(self, x, context_mask, *rest, **kwargs)

    monkeypatch.setattr(TriangleTransformerML, "encode", spy)
    return seen


# -- network -------------------------------------------------------------------


@pytest.mark.parametrize("dependence", ["ar", "joint"])
def test_forward_shapes_and_validity(dependence):
    """Both heads emit a valid density, with the shape difference that defines
    them: ``ar`` gives per-line univariate mixtures (sigma > 0, weights
    normalized); ``joint`` gives one mixture over lines whose components carry
    a mean vector and a lower-triangular Cholesky factor with positive
    diagonal - i.e. a genuine multivariate normal, not a diagonal one."""
    cfg = tiny(dependence)
    torch.manual_seed(0)
    model = TriangleTransformerML(cfg, n_lines=3, n_features=2, n_w=5, n_d=4)
    b = 6
    x = torch.randn(b, 3, 2, 5, 4)
    # one flag per (line, CHANNEL, cell) - x's own shape
    ctx = torch.rand(b, 3, 2, 5, 4) > 0.5
    lm = torch.ones(b, 3, dtype=torch.bool)
    # Real companies do not write every line; the line mask must be tolerated
    # by the forward pass, not just by the loss.
    lm[:, 2] = False  # one absent line everywhere
    prem = torch.randn(b, 3)
    cutoff = torch.randint(1, 8, (b,))
    if dependence == "ar":
        log_pi, mu, sigma = model.forward_ar(x, ctx, lm, prem, cutoff)
        assert log_pi.shape == mu.shape == sigma.shape == (b, 3, 5, 4, cfg.n_components)
        assert (sigma > 0).all()
        torch.testing.assert_close(
            log_pi.logsumexp(dim=-1), torch.zeros(b, 3, 5, 4), atol=1e-5, rtol=0
        )
    else:
        log_pi, mu, scale = model.forward_joint(x, ctx, lm, prem, cutoff)
        assert log_pi.shape == (b, 5, 4, cfg.n_components)
        assert mu.shape == (b, 5, 4, cfg.n_components, 3)
        assert scale.shape == (b, 5, 4, cfg.n_components, 3, 3)
        diag = scale.diagonal(dim1=-2, dim2=-1)
        assert (diag > 0).all()
        assert (scale.triu(1) == 0).all()  # lower-triangular


def test_a_channel_value_is_never_consumed_without_its_own_flag():
    """Poison a feature channel wherever ITS flag is off; the encoding must be
    bit-identical.

    The per-cell flag this replaced could not express "target observed here,
    feature not": it gated the whole cell, so the feature's value - which at
    such a cell is the contract's padding zero, or the standardized image of
    it - entered the token as though it had been observed. The poisoned cells
    here are exactly the ones a single flag would have let through: channel 0
    in context, channel 1 out of it.
    """
    torch.manual_seed(0)
    model = TriangleTransformerML(tiny("ar"), n_lines=2, n_features=2, n_w=4, n_d=4)
    b = 3
    x = torch.randn(b, 2, 2, 4, 4)
    ctx = torch.ones(b, 2, 2, 4, 4, dtype=torch.bool)
    ctx[:, :, 1, :, 2] = False  # the feature is unobserved at dev index 2 only
    lm = torch.ones(b, 2, dtype=torch.bool)
    prem = torch.zeros(b, 2)
    cutoff = torch.full((b,), 3)
    args = (ctx, lm, prem, cutoff)

    model.eval()
    with torch.no_grad():
        clean = model.encode(x, *args)
        poisoned = x.clone()
        poisoned[:, :, 1, :, 2] = 1e6
        assert not torch.equal(x, poisoned)  # the poison really landed
        torch.testing.assert_close(clean, model.encode(poisoned, *args), atol=0, rtol=0)


def test_a_per_cell_context_mask_is_refused_by_name():
    """The 0.5.3 spelling - one flag per cell - must not be accepted quietly.

    It would broadcast a channel's flag onto channels it does not describe (or,
    where the axis sizes happen to line up, produce a plausible wrong answer),
    so the shape is checked against ``x``'s own rather than left to torch.
    """
    torch.manual_seed(0)
    model = TriangleTransformerML(tiny("ar"), n_lines=2, n_features=2, n_w=4, n_d=4)
    x = torch.randn(2, 2, 2, 4, 4)
    per_cell = torch.ones(2, 2, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="per cell"):
        model.encode(
            x, per_cell, torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, 2), torch.full((2,), 3)
        )


def test_joint_nll_marginalizes_absent_lines():
    """The joint likelihood must integrate absent lines OUT, not impute them.

    Tested adversarially: poison the masked line's values with 1e6 and require
    the NLL to be bit-identical. Merely zeroing a masked term after computing a
    full multivariate density would still change the answer through the shared
    covariance, so this is the assertion that pins true marginalization.
    """
    torch.manual_seed(1)
    b, n_w, n_d, k, n_l = 4, 3, 3, 2, 3
    log_pi = torch.log_softmax(torch.randn(b, n_w, n_d, k), dim=-1)
    mu = torch.randn(b, n_w, n_d, k, n_l)
    # Build a valid Cholesky factor by hand: lower-triangular with a strictly
    # positive diagonal (+0.5 keeps it away from singular).
    scale = torch.randn(b, n_w, n_d, k, n_l, n_l).tril()
    idx = torch.arange(n_l)
    scale[..., idx, idx] = scale[..., idx, idx].abs() + 0.5
    y = torch.randn(b, n_l, n_w, n_d)
    mask = torch.ones(b, n_l, n_w, n_d, dtype=torch.bool)
    mask[:, 2] = False  # line 2 never counts
    nll = joint_mdn_nll(log_pi, mu, scale, y, mask)
    assert torch.isfinite(nll)
    # poisoning the excluded line's values must not change the NLL
    y_poisoned = y.clone()
    y_poisoned[:, 2] = 1e6
    torch.testing.assert_close(nll, joint_mdn_nll(log_pi, mu, scale, y_poisoned, mask))


def test_joint_sample_shapes_and_determinism():
    """Sampling the joint head returns one value per (cell, line) - component
    selection is shared across lines, which is what makes the draw coherent -
    and repeats exactly under the same explicit generator seed."""
    torch.manual_seed(2)
    b, n_w, n_d, k, n_l = 3, 2, 2, 2, 3
    log_pi = torch.log_softmax(torch.randn(b, n_w, n_d, k), dim=-1)
    mu = torch.randn(b, n_w, n_d, k, n_l)
    scale = torch.randn(b, n_w, n_d, k, n_l, n_l).tril()
    idx = torch.arange(n_l)
    scale[..., idx, idx] = scale[..., idx, idx].abs() + 0.5
    gen = torch.Generator().manual_seed(3)
    s1 = joint_mdn_sample(log_pi, mu, scale, generator=gen)
    assert s1.shape == (b, n_w, n_d, n_l)
    gen2 = torch.Generator().manual_seed(3)
    torch.testing.assert_close(s1, joint_mdn_sample(log_pi, mu, scale, generator=gen2))


def test_joint_head_correlation_is_learnable():
    """The joint head can express correlation: with an off-diagonal scale,
    sampled line pairs correlate.

    Without this the ``joint`` variant would be an expensive re-parameterization
    of ``ar``. Set up as the simplest possible case (one cell, one component,
    two lines) so the implied correlation is analytic and the check is a direct
    property of ``joint_mdn_sample``.
    """
    b, n_w, n_d, k, n_l = 1, 1, 1, 1, 2
    log_pi = torch.zeros(b, n_w, n_d, k)
    mu = torch.zeros(b, n_w, n_d, k, n_l)
    # Cholesky of a unit-variance 2x2 with rho = 0.9: row 2 is (0.9, sqrt(1-0.81)).
    scale = torch.tensor([[1.0, 0.0], [0.9, 0.4359]])[None, None, None, None]
    gen = torch.Generator().manual_seed(0)
    draws = torch.stack(
        [joint_mdn_sample(log_pi, mu, scale, generator=gen)[0, 0, 0] for _ in range(4000)]
    )
    corr = np.corrcoef(draws.numpy().T)[0, 1]
    # 0.8 not 0.9: 4000 draws leave sampling error, and the test only needs to
    # prove correlation is transmitted, not measure it precisely.
    assert corr > 0.8  # implied correlation 0.9


# -- entry ---------------------------------------------------------------------


@pytest.mark.parametrize("dependence", ["ar", "joint"])
def test_fit_predict_contract(backend_name, dependence):
    """Both variants fit and predict the SUR target layout, so this entry drops
    straight into the same comparison as ``sur``/``copula_glm``. The closing
    assertion is the important one: the grand total equals the sum of line
    totals within each DRAW, which is what makes the diversified total inherit
    the model's dependence rather than being a sum of marginals."""
    t = synthetic_triangle(backend_name)
    entry = NNTransformerML().fit(t, loss_field="paid_loss", config=tiny(dependence), seed=0)
    pred = entry.predict(segment={"company_code": "0001"}, seed=0)
    # SUR layout: 2 lobs x 6 origins + 2 lob totals + grand total
    assert pred.n_targets == 2 * 6 + 2 + 1
    assert pred.targets["label"].tolist()[-1] == "total"
    assert np.isfinite(pred.samples).all()
    assert (pred.samples[:, -1] > 0).mean() > 0.95

    realized = entry.realized_ultimates(t, segment={"company_code": "0001"})
    assert realized.shape == (15,)
    table = pred.summary(observed=realized)
    assert {"estimate", "se", "cv", "outcome", "percentile"} <= set(table.columns)

    # grand total = sum of lob totals inside the same draws (coherent draws)
    np.testing.assert_allclose(
        pred.samples[:, -1], pred.samples[:, -3] + pred.samples[:, -2], rtol=1e-6
    )


def test_predict_caches_and_reproduces(backend_name):
    """The autoregressive rollout is computed once and reused across predict
    calls (identity check), and a fresh fit at the same seed reproduces the
    draws exactly - deep-ensemble spread has to be reproducible for any
    calibration result to be citable."""
    t = synthetic_triangle(backend_name)
    entry = NNTransformerML().fit(t, loss_field="paid_loss", config=tiny("ar"), seed=0)
    a = entry.predict(segment={"company_code": "0001"}, seed=7)
    cached = entry._rollout_ults
    entry.predict(segment={"company_code": "0001"}, seed=7)
    assert entry._rollout_ults is cached

    entry2 = NNTransformerML().fit(t, loss_field="paid_loss", config=tiny("ar"), seed=0)
    a2 = entry2.predict(segment={"company_code": "0001"}, seed=7)
    np.testing.assert_allclose(a.samples, a2.samples)


def test_level_fields_reaches_the_contract(backend_name):
    """``fit(level_fields=...)`` is delivered, not merely accepted.

    Read off the fitted contract's ``field_kinds``, which is the only place the
    declaration shows up: a level channel is carried undifferenced, so an
    argument that never reached ``nn_company_data`` would leave the case
    reserve differenced into case MOVEMENT with every downstream number still
    finite and plausible. The refusal is the contract's and is exercised
    through ``fit`` for the same reason - it proves the kwarg travels.
    """
    t = feature_triangle(backend_name)
    entry = fit_with_case_reserve(t)
    assert entry.contract_["fields"] == ["paid_loss", "case_reserve"]
    assert entry.contract_["field_kinds"] == ("increment", "level")

    plain = fit_with_case_reserve(t, level=False)
    assert plain.contract_["field_kinds"] == ("increment", "increment")
    # a level that is not also a feature names no channel
    with pytest.raises(ValueError, match="level_fields"):
        NNTransformerML().fit(
            t, loss_field="paid_loss", level_fields=("case_reserve",), config=tiny("ar"), seed=0
        )


def test_training_context_is_the_channels_own_observedness(backend_name, monkeypatch):
    """Every context mask fit hands the network is the contract's ``x_obs``
    under a calendar gate - per channel, not the target's mask repeated.

    The fixture drops the case reserve from one dev column while the paid
    target stays observed there, so the two masks genuinely differ; the last
    assertion is what makes the test non-vacuous, requiring at least one cell
    conditioned on for the target and NOT for the feature. Built on a single
    company so every batch is that company and the recorded mask needs no
    index bookkeeping.
    """
    seen = record_context(monkeypatch)
    t = feature_triangle(backend_name, feature_missing_dev=2)
    entry = fit_with_case_reserve(t)
    c = entry.contract_
    x_obs = c["x_obs"]  # (1, L, F, W, D)
    assert x_obs.shape[0] == 1, "fixture must be one company for the index-free assertions"
    assert (x_obs[:, :, 0] & ~x_obs[:, :, 1]).any(), (
        "the fixture no longer has a cell where the target is observed and the feature is "
        "not, so nothing here distinguishes a per-channel mask from a per-cell one"
    )

    assert seen, "no context mask was recorded - the spy is wired to the wrong method"
    for ctx in seen:
        assert ctx.shape[1:] == x_obs.shape[1:], "the mask lost its channel axis"
        leaked = ctx & ~x_obs  # a flag on a cell whose channel has no usable value
        assert not leaked.any(), (
            f"{int(leaked.sum())} flag(s) claim a value the contract calls unusable - "
            "the padding zero would be read as an observed value"
        )
    assert any((ctx[:, :, 0] & ~ctx[:, :, 1]).any() for ctx in seen), (
        "no mask ever conditioned on the target without the feature, so a single per-cell "
        "flag would have produced the same fit"
    )


def test_feature_channel_stats_come_from_its_own_observed_cells(backend_name):
    """Per-(line, channel, dev) standardization is estimated on the cells where
    THAT channel has a value.

    Standardizing a feature on the TARGET's cells throws away every value the
    feature has where the target has none, and at a dev where the target has
    none at all it pins the channel - standardized to a constant 0, its own
    values unused. The fixture drops the paid target from one dev column of
    the first line, and the old target-gated call is recomputed and required
    to DIFFER there, so the assertion cannot pass by coincidence.
    """
    t = feature_triangle(backend_name, target_missing_dev=3)
    entry = fit_with_case_reserve(t)
    c = entry.contract_
    cal = c["cal_idx"]
    _, _, val_cutoff = _scheme.splits(c["obs_mask"].any(axis=1), cal, entry.config_.val_diagonals)
    chan_elig = c["x_obs"] & (cal[None, None, None] <= val_cutoff)

    for li in range(c["x"].shape[1]):
        mean, std, pinned = _scheme.norm_stats(c["x"][:, li], chan_elig[:, li], c["x_obs"][:, li])
        np.testing.assert_allclose(entry.norm_["mean"][li], mean)
        np.testing.assert_allclose(entry.norm_["std"][li], std)
        np.testing.assert_array_equal(entry.norm_["pinned"][li], pinned)

    # the target-gated form the entry used before x_obs existed, on the line
    # whose target column is missing
    li0 = c["lob_levels"].index("lob_0")
    old_ctx = c["obs_mask"] & (cal[None, None] <= val_cutoff)
    _, _, old_pinned = _scheme.norm_stats(c["x"][:, li0], old_ctx[:, li0], c["obs_mask"][:, li0])
    assert old_pinned[1, 3] and not entry.norm_["pinned"][li0, 1, 3], (
        "the feature's stats are the same either way; the fixture stopped exercising the "
        "cells where the feature is observed and the target is not"
    )


@pytest.mark.parametrize("dependence", ["ar", "joint"])
def test_rollout_promotes_only_the_target_channel(backend_name, monkeypatch, dependence):
    """A sampled cell re-enters the context as a target value and nothing else.

    Next year's case reserve is not simulated, so flagging it would present the
    contract's padding zero - standardized, so not even a zero on the model's
    scale - as an observed snapshot. Recorded off the real ``predict()`` path:
    the feature flags must stay exactly the contract's, while the target flags
    must grow, and both halves are asserted because a rollout that promoted
    nothing would satisfy the first one trivially.
    """
    t = feature_triangle(backend_name, upper=True)
    entry = fit_with_case_reserve(t, dependence)
    c = entry.contract_
    x_obs = c["x_obs"]  # (n_c, L, F, W, D)
    n_c = x_obs.shape[0]

    seen = record_context(monkeypatch)  # installed AFTER fit: rollout masks only
    entry.predict(segment={"company_code": "0001"}, seed=0)
    assert seen, "the rollout made no forward pass - nothing was promoted or recorded"

    promoted = 0
    for ctx in seen:
        assert ctx.shape[1:] == x_obs.shape[1:], "the mask lost its channel axis"
        # the rollout replicates each company `chunk` times, company-major
        ctx = ctx.reshape(n_c, -1, *x_obs.shape[1:])
        feature = ctx[:, :, :, 1:]
        np.testing.assert_array_equal(
            feature,
            np.broadcast_to(x_obs[:, None, :, 1:], feature.shape),
            err_msg="a feature flag moved during the rollout; only the target is fed back",
        )
        target = ctx[:, :, :, 0]
        assert (target >= x_obs[:, None, :, 0]).all(), "an observed target cell left the context"
        promoted += int((target & ~x_obs[:, None, :, 0]).sum())
    assert promoted, (
        "no sampled cell was ever promoted, so the feature-flag assertion above is vacuous - "
        "the fixture has stopped leaving future cells to roll out"
    )


def test_predict_before_fit_raises():
    """GalleryEntry lifecycle: predict() before fit() is a clear RuntimeError."""
    with pytest.raises(RuntimeError, match="fit"):
        NNTransformerML().predict()


def test_unknown_segment_raises(backend_name):
    """Segment selection fails loudly. Note the cohort unit here is the COMPANY
    (all its lines together), so the no-match message speaks of companies, not
    company x LOB cohorts as in the single-line entry."""
    t = synthetic_triangle(backend_name)
    entry = NNTransformerML().fit(t, loss_field="paid_loss", config=tiny("ar"), seed=0)
    with pytest.raises(KeyError, match="unknown segment column"):
        entry.predict(segment={"nope": "x"})
    with pytest.raises(ValueError, match="matches 0 cohorts"):
        entry.predict(segment={"company_code": "9999"})
