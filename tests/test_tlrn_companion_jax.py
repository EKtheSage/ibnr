"""``empty_query_bias`` and ``compact_embeddings`` in the JAX backend, against torch.

Skips cleanly without torch or jax. ``test_tlrn_companion_network.py`` pins what the options
do in torch; this file requires the JAX forward and the JAX training loop to do the same,
and a hand-written numpy copy of the companion's attention to agree with the JAX one.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from ibnr.gallery.nn.tlrn import jax_backend as jb  # noqa: E402
from ibnr.gallery.nn.tlrn.config import TLRNConfig  # noqa: E402

from .test_nn_features import company_contract  # noqa: E402
from .test_tlrn_companion_network import (  # noqa: E402
    D,
    companion_mha,
    np_weights,
)
from .test_tlrn_jax import (  # noqa: E402
    AY_AS_OF,
    _assert_same_training,
    _features,
    _torch_forward,
    _torch_net,
    ay_arrays,
    fit,
    tiny_ay,
)

OPTIONS = {
    "empty_bias": {"empty_query_bias": True},
    "compact": {"compact_embeddings": True},
    "both": {"empty_query_bias": True, "compact_embeddings": True},
}
LAYOUTS = {
    "lines_lags": {"mask": "observed_cells"},
    "accident_year": {
        "head": "premium_lr",
        "attention": ("line", "lag", "ay"),
        "mask": "observed_cells",
        "batch_unit": "company",
        "member": "mcl_blend",
    },
    # the published mask: only the accident-year axis can have an empty query
    "ay_unwritten": {"attention": ("line", "lag", "ay"), "batch_unit": "company"},
}


def _forward_both(cfg):
    contract = company_contract("duckdb", lines={1: [0]})
    features = _features(contract, 4, with_mcl=cfg.member == "mcl_blend")
    model = _torch_net(cfg, contract, features, 4)
    n_l, n_d = model.n_lines, model.n_lag
    support_np = np.ones((n_l, n_d - 1), dtype=bool)
    support = torch.tensor(support_np)
    want = _torch_forward(model, features, support)
    spec = jb.spec_of(cfg, n_l=n_l, n_d=n_d, n_w=contract["x"].shape[3])
    params = jb.params_from_state(model.state_dict(), [n for n, _ in model.named_parameters()])
    s = jb.as_set(features)
    got = jb.forward(params, s, spec, support=jnp.asarray(support_np))
    # the same weights through the same forward with the empty-query rule switched off
    without = jb.forward(
        params, s, spec._replace(empty_bias=False), support=jnp.asarray(support_np)
    )
    return want, got, without


@pytest.mark.parametrize("option", sorted(OPTIONS))
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_the_jax_forward_matches_torch_with_each_option(layout, option):
    cfg = TLRNConfig(dropout=0.0, d_model=8, n_heads=2, **LAYOUTS[layout], **OPTIONS[option])
    want, got, without = _forward_both(cfg)
    for key in ("pred", "C"):
        expected = want[key].numpy()
        scale = np.abs(expected).max()
        np.testing.assert_allclose(np.asarray(got[key]), expected, rtol=2e-5, atol=2e-6 * scale)
    # where the rule is on it changes the answer on this batch (which has hidden groups), so
    # the agreement above is agreement on the bias and not on nothing
    if cfg.empty_query_bias:
        assert not np.allclose(np.asarray(got["pred"]), np.asarray(without["pred"]), rtol=1e-4)


@pytest.mark.parametrize("empty_query_bias", [True, False])
def test_the_jax_attention_equals_the_companions_on_hidden_groups(empty_query_bias):
    torch.manual_seed(3)
    attn = torch.nn.MultiheadAttention(D, 2, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        attn.in_proj_bias.normal_()
        attn.out_proj.bias.normal_()
    h = torch.randn(6, 5, D)
    readable = torch.rand(6, 5) > 0.5
    readable[0] = False
    readable[3] = False
    readable[1] = True
    readable[2, 1:] = False
    readable[2, 0] = True
    readable[4:, ::2] = True
    params = {
        "a.in_proj_weight": jnp.asarray(attn.in_proj_weight.detach().numpy()),
        "a.in_proj_bias": jnp.asarray(attn.in_proj_bias.detach().numpy()),
        "a.out_proj.weight": jnp.asarray(attn.out_proj.weight.detach().numpy()),
        "a.out_proj.bias": jnp.asarray(attn.out_proj.bias.detach().numpy()),
    }
    spec = jb.spec_of(
        TLRNConfig(d_model=D, n_heads=2, dropout=0.0, mask="observed_cells", empty_query_bias=True),
        n_l=2,
        n_d=3,
        n_w=4,
    )
    spec = spec._replace(empty_bias=empty_query_bias)
    got = np.asarray(
        jb._attend(
            params,
            "a",
            jnp.asarray(h.numpy()),
            jnp.asarray(readable.numpy()),
            guard=True,
            spec=spec,
            train=False,
            key=None,
        )
    )
    companion = companion_mha(h.numpy(), pad=~readable.numpy(), heads=2, **np_weights(attn))
    hidden = ~readable.numpy().any(axis=1)
    assert hidden.sum() == 2
    # the groups that can read something are the companion's whatever the option says
    np.testing.assert_allclose(got[~hidden], companion[~hidden], rtol=1e-4, atol=1e-5)
    if empty_query_bias:
        np.testing.assert_allclose(got, companion, rtol=1e-4, atol=1e-5)
    else:
        np.testing.assert_array_equal(got[hidden], 0.0)


def test_both_options_train_along_the_torch_trajectory():
    """The accident-year variant with both options on: every part of the loop agrees, so the
    JAX program indexes the compact tables and returns the bias the way torch does."""
    cfg = tiny_ay(grad_clip=0.05, empty_query_bias=True, compact_embeddings=True)
    a = fit(cfg, arrays=ay_arrays(), as_of=AY_AS_OF, incurred=True)
    b = fit(cfg, arrays=ay_arrays(), as_of=AY_AS_OF, incurred=True, backend="jax")
    _assert_same_training(a, b, rtol=2e-4)
    assert a.models_[0].line_emb.weight.shape[0] == a.models_[0].n_lines
