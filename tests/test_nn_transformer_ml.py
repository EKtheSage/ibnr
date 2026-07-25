"""gallery.nn.transformer_ml: multi-line network, both dependence heads,
entry contract. Skips cleanly when torch is not installed.

What this file protects: ``nn_transformer_ml`` is the only NN entry that models
CROSS-LINE dependence. It fits on company cohorts (all lines of a company at
once, attention running across lines) and predicts the SUR target layout so it
is directly comparable to the statistical dependence baselines.

Two ablatable dependence variants, both exercised here (CLAUDE.md: build both
design variants, keep changes ablatable):

* ``"ar"`` - line-by-line autoregressive sampling; independent per-line heads.
* ``"joint"`` - a multivariate Gaussian-mixture head over lines, with absent
  lines marginalized out of the likelihood.

The invariants that matter are exactly the ones a multi-line model can get
wrong without any shape error: the joint likelihood must genuinely IGNORE
absent lines (companies do not write every line), the joint head must be able
to express correlation at all, and - for both variants - the grand total must
be the sum of line totals WITHIN each draw, so the diversified total carries
the dependence structure instead of being a sum of marginal means.

Determinism and small-triangle caveats are as in ``test_nn_transformer.py``:
explicit seeds/generators throughout, and a deliberately under-powered config
(``tiny``) since nothing here asserts predictive quality. Entry tests run on
BOTH ibis backends.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

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
    ctx = torch.rand(b, 3, 5, 4) > 0.5
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
    with pytest.raises(ValueError, match="matches 0 companies"):
        entry.predict(segment={"company_code": "9999"})
