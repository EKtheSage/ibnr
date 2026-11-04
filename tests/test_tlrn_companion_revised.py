"""``maturity_embedding`` and ``companion_revised_variant``, and the ``paid_only`` option.

The fourth tlrn study's model is the third study's with the per-count ("lags observed")
embedding removed, and its paid-only ablation zeroes the five incurred and case reserve inputs
while keeping all 13. What this file protects, in order:

1. the config: the default, the preset, the refusal of a non-boolean, and the preset differing
   from ``companion_final_variant`` in exactly one field;
2. the size (18,153 against 18,537) and that every shared parameter starts at the same value
   for the same seed, because the dropped table's draws are still consumed;
3. forward parity against a numpy transcription of the companion's revised forward pass, and
   the jax forward against the torch one;
4. delivery through ``gallery.fit`` on both backends (a field that is accepted and then
   ignored is the bug this repository keeps meeting);
5. ``paid_only``: the channels it zeroes are the five named ones, and it is connected.

Skips cleanly without torch (and the jax halves without jax).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.nn.tlrn import head as tlrn_head  # noqa: E402
from ibnr.gallery.nn.tlrn.config import TLRNConfig  # noqa: E402
from ibnr.gallery.nn.tlrn.network import TLRNNetwork  # noqa: E402
from ibnr.kernels.nn_features import FEATURE_NAMES, N_FEATURES_PAID_ONLY  # noqa: E402

from .test_nn_features import company_contract  # noqa: E402
from .test_tlrn_companion_network import (  # noqa: E402
    companion_block,
    count,
    features_for,
    np_layer_norm,
    run_network,
    small_preset,
)

# -- 1. the config ---------------------------------------------------------------------------


def test_maturity_embedding_is_on_by_default_and_off_in_the_revised_preset():
    assert TLRNConfig().maturity_embedding is True
    assert TLRNConfig.companion_final_variant().maturity_embedding is True
    assert TLRNConfig.companion_revised_variant().maturity_embedding is False


@pytest.mark.parametrize("bad", [0, 1, "no", None])
def test_a_non_boolean_maturity_embedding_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="maturity_embedding must be True or False"):
        TLRNConfig(maturity_embedding=bad)


def test_the_revised_preset_differs_from_the_final_one_in_exactly_one_field():
    final = dataclasses.asdict(TLRNConfig.companion_final_variant())
    revised = dataclasses.asdict(TLRNConfig.companion_revised_variant())
    assert {k for k in final if final[k] != revised[k]} == {"maturity_embedding"}
    assert set(final) == set(revised)


def test_an_override_still_wins_on_the_revised_preset():
    assert TLRNConfig.companion_revised_variant(maturity_embedding=True).maturity_embedding


# -- 2. the size and the starting weights ----------------------------------------------------


def build(cfg, seed=11) -> TLRNNetwork:
    torch.manual_seed(seed)
    return TLRNNetwork(
        cfg, n_lines=4, n_lag=10, n_feat=13, n_origin=10, head_init=torch.zeros(4, 10)
    )


def test_the_revised_network_has_18153_parameters_and_no_maturity_table():
    revised = build(TLRNConfig.companion_revised_variant())
    final = build(TLRNConfig.companion_final_variant())
    assert count(revised) == 18_153
    assert count(final) == 18_537
    assert not any(k.startswith("nobs_emb") for k in dict(revised.named_parameters()))
    assert not any(k.startswith("nobs_emb") for k in revised.state_dict())
    assert "nobs_emb.weight" in final.state_dict()
    assert count(final) - count(revised) == final.nobs_emb.weight.numel() == 12 * 32


@pytest.mark.parametrize("seed", [0, 11])
def test_every_shared_parameter_starts_at_the_same_value_for_the_same_seed(seed):
    revised = build(TLRNConfig.companion_revised_variant(), seed).state_dict()
    final = build(TLRNConfig.companion_final_variant(), seed).state_dict()
    assert set(final) - set(revised) == {"nobs_emb.weight"}
    for name, value in revised.items():
        assert torch.equal(value, final[name]), name


# -- 3. forward parity -----------------------------------------------------------------------


def revised_net_and_features(monkeypatch):
    contract = company_contract("duckdb", lines={1: [0]})
    cutoff = 4
    feats = features_for(contract, cutoff)
    _, n_l, _, n_w, n_d = contract["x"].shape
    cfg = TLRNConfig.companion_revised_variant(
        d_model=8, n_heads=2, dropout=0.0, member="network", alpha_on="all_validation"
    )
    from ibnr.kernels.nn_features import pooled_incremental_lr

    head_init = torch.tensor(np.log(pooled_incremental_lr(contract, cutoff)), dtype=torch.float32)
    torch.manual_seed(5)
    net = TLRNNetwork(
        cfg, n_lines=n_l, n_lag=n_d, n_feat=feats["n_feat"], n_origin=n_w, head_init=head_init
    ).eval()
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name != "beta":
                p.copy_(0.4 * torch.randn_like(p))
    return net, feats, (n_l, n_d, n_w)


def test_the_network_equals_the_companions_revised_forward_to_the_head(monkeypatch):
    """``h = (feat * gate) @ inp.w + inp.b + line_emb[line] + lag_emb[lag]``, no count term."""
    net, feats, (n_l, n_d, n_w) = revised_net_and_features(monkeypatch)
    captured = {}
    real = tlrn_head.log_ratios

    def spy(beta, net_out, *args, **kwargs):
        captured["net"] = net_out.detach().clone()
        return real(beta, net_out, *args, **kwargs)

    monkeypatch.setattr(tlrn_head, "log_ratios", spy)
    run_network(net, feats)
    x = (
        feats["feat"].astype(np.float32) @ net.inp.weight.detach().numpy().T
        + net.inp.bias.detach().numpy()
        + net.line_emb.weight.detach().numpy()[np.asarray(feats["line_ix"]) - 1]
        + net.lag_emb.weight.detach().numpy()[np.asarray(feats["lag_ix"]) - 1]
    )
    readable = np.asarray(feats["visible"]).astype(bool)
    h = companion_block(net.blocks[0], x, readable, n_l, n_d, n_w)
    out = (
        np_layer_norm(net.ln, h) @ net.head.weight.detach().numpy().T
        + net.head.bias.detach().numpy()
    )
    np.testing.assert_allclose(
        captured["net"].numpy(), out[..., 0].reshape(-1, n_l, n_d), rtol=2e-4, atol=2e-5
    )


def test_the_count_term_is_really_gone_from_the_forward_pass(monkeypatch):
    """The same weights with the table put back give a different answer, so the parity above
    is agreement on a forward without the term and not on a term that happens to be small."""
    net, feats, _ = revised_net_and_features(monkeypatch)
    out = run_network(net, feats)
    # put the revised weights into a network that has the table, with a random table
    # (a constant one would be removed by the layer norm)
    final = TLRNNetwork(
        TLRNConfig.companion_final_variant(
            d_model=8, n_heads=2, dropout=0.0, member="network", alpha_on="all_validation"
        ),
        n_lines=net.n_lines,
        n_lag=net.n_lag,
        n_feat=feats["n_feat"],
        n_origin=net.n_origin,
        head_init=torch.zeros(net.n_lines, net.n_lag),
    ).eval()
    final.load_state_dict(
        {**net.state_dict(), "nobs_emb.weight": torch.randn_like(final.nobs_emb.weight)}
    )
    other = run_network(final, feats)
    assert not torch.allclose(out["logr"], other["logr"], rtol=1e-4)


def test_the_jax_forward_equals_the_torch_one_for_the_revised_config():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from ibnr.gallery.nn.tlrn import jax_backend as jb

    from .test_tlrn_jax import _features, _torch_forward, _torch_net

    cfg = TLRNConfig.companion_revised_variant(d_model=8, n_heads=2, dropout=0.0)
    contract = company_contract("duckdb", lines={1: [0]})
    features = _features(contract, 4, with_mcl=True)
    model = _torch_net(cfg, contract, features, 4)
    n_l, n_d = model.n_lines, model.n_lag
    support_np = np.ones((n_l, n_d - 1), dtype=bool)
    want = _torch_forward(model, features, torch.tensor(support_np))
    spec = jb.spec_of(cfg, n_l=n_l, n_d=n_d, n_w=contract["x"].shape[3])
    assert spec.maturity is False
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith("nobs_emb") for n in names)
    params = jb.params_from_state(model.state_dict(), names)
    got = jb.forward(params, jb.as_set(features), spec, support=jnp.asarray(support_np))
    for key in ("pred", "C"):
        expected = want[key].numpy()
        np.testing.assert_allclose(
            np.asarray(got[key]), expected, rtol=2e-5, atol=2e-6 * np.abs(expected).max()
        )


# -- 4. delivery through gallery.fit ---------------------------------------------------------


@pytest.fixture(params=["torch", "jax"])
def backend(request):
    if request.param == "jax":
        pytest.importorskip("jax")
    return request.param


def _fit(config, backend, **kwargs):
    from .test_tlrn_refit import fit

    return fit(config, backend=backend, ay=True, **kwargs)


def test_maturity_embedding_reaches_the_trained_networks_and_the_forecast(backend):
    on = _fit(small_preset(), backend)
    off = _fit(small_preset(maturity_embedding=False), backend)
    for model in on.models_:
        assert hasattr(model, "nobs_emb")
    for model in off.models_:
        assert not hasattr(model, "nobs_emb")
        assert count(model) == count(on.models_[0]) - on.models_[0].nobs_emb.weight.numel()
    assert not np.allclose(on.member_company_reserves(), off.member_company_reserves(), rtol=1e-6)


# -- 5. paid_only ----------------------------------------------------------------------------

ROLES = {"incurred_field": "incurred_loss", "case_field": "case_reserve"}


def test_the_paid_only_channels_are_the_five_incurred_and_case_ones():
    assert N_FEATURES_PAID_ONLY == 8
    assert FEATURE_NAMES[N_FEATURES_PAID_ONLY:] == (
        "context incremental incurred loss ratio",
        "observed incremental incurred loss ratio",
        "paid to incurred",
        "incurred loss ratio",
        "case reserve loss ratio",
    )
    # and channel 6 (lags observed / N_J) is a paid channel that stays
    assert FEATURE_NAMES[6] == "fraction of lags observed"


def test_paid_only_features_zero_exactly_those_channels_and_keep_the_rest():
    from ibnr.kernels.nn_features import tlrn_features

    contract = company_contract("duckdb", lines={1: [0]})
    kwargs = dict(
        cutoff=4,
        target_lo=5,
        target_hi=contract["x"].shape[3] + contract["x"].shape[4] - 1,
        **ROLES,
    )
    full = tlrn_features(contract, **kwargs)
    paid = tlrn_features(contract, paid_only=True, **kwargs)
    assert paid["n_feat"] == full["n_feat"] == 13
    assert (full["feat"][..., 8:] != 0).any(axis=(0, 1)).all()  # each really carries signal
    assert (paid["feat"][..., 8:] == 0).all()
    np.testing.assert_array_equal(paid["feat"][..., :8], full["feat"][..., :8])


def test_paid_only_without_the_roles_is_refused_by_name():
    from ibnr.kernels.nn_features import tlrn_features

    contract = company_contract("duckdb", lines={1: [0]})
    with pytest.raises(ValueError, match="needs incurred_field and case_field"):
        tlrn_features(contract, cutoff=4, target_lo=5, target_hi=9, paid_only=True)
    with pytest.raises(ValueError, match="needs incurred_field and case_field"):
        _fit(small_preset(), "torch", paid_only=True)
    with pytest.raises(ValueError, match="paid_only must be True or False"):
        _fit(small_preset(), "torch", paid_only=1, **ROLES)


def test_paid_only_reaches_the_network_inputs_and_the_forecast(backend, monkeypatch):
    """Every batch the network sees has zero incurred and case channels, and not otherwise."""
    seen: list[np.ndarray] = []
    real = TLRNNetwork.forward

    def spy(self, feat, *args, **kwargs):
        seen.append(feat.detach().numpy().copy())
        return real(self, feat, *args, **kwargs)

    monkeypatch.setattr(TLRNNetwork, "forward", spy)
    cfg = small_preset(maturity_embedding=False)
    full = _fit(cfg, backend, **ROLES)
    n_full = len(seen)
    paid = _fit(cfg, backend, paid_only=True, **ROLES)
    if backend == "torch":
        # the jax backend trains in compiled code; the torch forward is still used to select
        # and read out, so something was seen either way
        assert n_full > 0
    assert len(seen) > n_full
    assert all(f.shape[-1] == 13 for f in seen)
    before, after = seen[:n_full], seen[n_full:]
    assert any((f[..., 8:] != 0).any() for f in before)
    assert all((f[..., 8:] == 0).all() for f in after)
    assert any((f[..., :8] != 0).any() for f in after)
    assert paid.models_[0].inp.weight.shape[1] == 13
    assert not np.allclose(full.member_company_reserves(), paid.member_company_reserves())
