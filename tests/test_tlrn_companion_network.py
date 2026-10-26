"""``empty_query_bias`` and ``compact_embeddings``: the network the companion's code builds.

Skips cleanly without torch (the jax half is ``test_tlrn_companion_jax.py``). What this
file protects, in the order of the file:

1. the size: the preset's network has the companion's 18,537 parameters, layer by layer
   (built, not summed from the paper), and the unused rows of the default tables are gone;
2. the rule for a query with nothing to read: the attention's output bias, not zero, on
   each of the three axes, and unchanged (zero) with the option off;
3. a hand-written numpy copy of the companion's attention and block, run on a batch with
   hidden groups, against the network;
4. delivery through ``gallery.fit``: a parameter that is accepted and then ignored is the
   bug this repository keeps meeting, so each option is compared with and without it, and
   the preset is compared with the same options spelled out.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.special import erf  # noqa: E402

from ibnr.gallery.nn.tlrn import head as tlrn_head  # noqa: E402
from ibnr.gallery.nn.tlrn.config import TLRNConfig  # noqa: E402
from ibnr.gallery.nn.tlrn.network import AxialBlock, TLRNNetwork, attend  # noqa: E402
from ibnr.kernels.nn_features import pooled_incremental_lr, tlrn_features  # noqa: E402

from .test_nn_features import company_contract  # noqa: E402

# -- 1. the size -----------------------------------------------------------------------------


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def preset_network(**overrides) -> TLRNNetwork:
    cfg = TLRNConfig.companion_final_variant(**overrides)
    return TLRNNetwork(
        cfg, n_lines=4, n_lag=10, n_feat=13, n_origin=10, head_init=torch.zeros(4, 10)
    )


def test_the_preset_network_is_the_companions_size_layer_by_layer():
    """The companion's own figures: 1,280 + 3 x 4,288 + 4,256 + 97 + 40 = 18,537."""
    net = preset_network()
    by_name = dict(net.named_parameters())

    def numel(*prefixes):
        return sum(v.numel() for k, v in by_name.items() if k.startswith(prefixes))

    assert numel("inp.", "line_emb", "lag_emb", "nobs_emb") == 1_280
    # an attention with the layer norm in front of it
    assert numel("blocks.0.attn_line.", "blocks.0.ln1.") == 4_288
    assert numel("blocks.0.attn_lag.", "blocks.0.ln2.") == 4_288
    assert numel("blocks.0.attn_ay.", "blocks.0.ln_ay.") == 4_288
    assert numel("blocks.0.ff.", "blocks.0.ln3.") == 4_256
    assert numel("ln.", "head.") == 97
    assert numel("beta") == 40
    # every parameter is in one of those groups, and they add up to the companion's count
    assert count(net) == 1_280 + 3 * 4_288 + 4_256 + 97 + 40 == 18_537


def test_compact_embeddings_drops_exactly_the_two_unused_rows_and_nothing_else():
    cfg = TLRNConfig.accident_year_variant()
    kwargs = {
        "n_lines": 4,
        "n_lag": 10,
        "n_feat": 13,
        "n_origin": 10,
        "head_init": torch.zeros(4, 10),
    }
    wide = TLRNNetwork(cfg, **kwargs)
    compact = TLRNNetwork(replace(cfg, compact_embeddings=True), **kwargs)
    assert count(wide) - count(compact) == 64 == 2 * 32
    assert (wide.line_emb.weight.shape[0], wide.lag_emb.weight.shape[0]) == (5, 11)
    assert (compact.line_emb.weight.shape[0], compact.lag_emb.weight.shape[0]) == (4, 10)
    # the visible-lag table keeps its twelve rows in both
    assert wide.nobs_emb.weight.shape[0] == compact.nobs_emb.weight.shape[0] == 12
    # the accident-year variant, which is not the preset, keeps the default size
    assert count(wide) == 18_601


def test_the_published_ldf_network_is_14309_by_default_and_14245_compact():
    """R's 14,309 (indices from 1) and the companion's JAX count for the same network."""
    kwargs = {"n_lines": 4, "n_lag": 10, "n_feat": 13}
    default = TLRNNetwork(TLRNConfig(), **kwargs)
    compact = TLRNNetwork(TLRNConfig(compact_embeddings=True), **kwargs)
    assert (count(default), count(compact)) == (14_309, 14_245)


def features_for(contract, cutoff, *, with_mcl=False):
    return tlrn_features(
        contract,
        cutoff=cutoff,
        target_lo=cutoff + 1,
        target_hi=contract["x"].shape[3] + contract["x"].shape[4] - 1,
        incurred_field="incurred_loss",
        case_field="case_reserve",
        with_mcl=with_mcl,
    )


def run_network(net, feats):
    """One forward pass over a feature set, the way the entry calls the network."""
    t = {
        k: torch.tensor(np.asarray(feats[k]), dtype=torch.float32)
        for k in ("feat", "c_lk", "p_lk", "fallback_lr", "fallback_logf")
    }
    with torch.no_grad():
        return net(
            t["feat"],
            t["c_lk"],
            t["p_lk"],
            torch.tensor(feats["lk"]).long(),
            torch.tensor(feats["line_ix"]).long(),
            torch.tensor(feats["lag_ix"]).long(),
            torch.tensor(feats["written"]).bool(),
            fallback_lr=t["fallback_lr"] if net.cfg.head == "premium_lr" else None,
            visible=torch.tensor(feats["visible"]).bool(),
        )


@pytest.mark.parametrize("compact", [True, False])
def test_the_embedding_rows_read_are_the_companions_when_compact(compact):
    """Line and lag come in 1-based; compact reads them 0-based and the visible-lag table at
    ``lk``. Every row holds its own index times 100, so the sum entering the first block
    names the rows that were read."""
    contract = company_contract("duckdb")
    feats = features_for(contract, 4)
    _, n_l, _, _, n_d = contract["x"].shape
    cfg = TLRNConfig(dropout=0.0, d_model=8, n_heads=2, compact_embeddings=compact)
    net = TLRNNetwork(cfg, n_lines=n_l, n_lag=n_d, n_feat=feats["n_feat"]).eval()
    with torch.no_grad():
        for emb in (net.line_emb, net.lag_emb, net.nobs_emb):
            rows = torch.arange(emb.weight.shape[0], dtype=torch.float32)[:, None]
            emb.weight.copy_(rows * 100.0 * torch.ones_like(emb.weight))
        net.inp.weight.zero_()
        net.inp.bias.zero_()
    seen = {}

    def record(module, args):
        seen["h"] = args[0].clone()  # returns None: the hook only looks

    net.blocks[0].register_forward_pre_hook(record)
    run_network(net, feats)
    line_ix, lag_ix = np.asarray(feats["line_ix"]), np.asarray(feats["lag_ix"])
    lk = np.asarray(feats["lk"])
    shift = 1 if compact else 0
    # compact: line/lag row = index - 1 (read 0-based), visible-lag row = lk.
    # default: line/lag row = index (1-based, row 0 unused), visible-lag row = lk + 1.
    want = (
        100.0 * ((line_ix - shift) + (lag_ix - shift))[None, :] + 100.0 * (lk + 1 - shift)[:, None]
    )
    np.testing.assert_allclose(seen["h"][..., 0].numpy(), want)
    # the same table in both: nobs_emb has twelve rows either way, and the largest row read
    # is in range (lk is at most n_lag)
    assert net.nobs_emb.weight.shape[0] == n_d + 2 and lk.max() + 1 - shift < n_d + 2


# -- 2. a query with nothing to read ----------------------------------------------------------

D = 8
N_L, N_J, N_W, N_C = 2, 3, 4, 2
AXES_CASES = [("lag",), ("line", "lag"), ("line", "lag", "ay")]


def make_block(axes, *, empty_query_bias, mask="observed_cells") -> AxialBlock:
    torch.manual_seed(0)
    block = AxialBlock(
        D,
        2,
        0.0,
        N_L,
        N_J,
        axes=axes,
        n_origin=N_W,
        mask=mask,
        empty_query_bias=empty_query_bias,
    ).eval()
    # every attention gets a different, clearly nonzero output bias, so a bias that went to
    # the wrong axis (or came back as zero) shows
    with torch.no_grad():
        for i, name in enumerate(("attn_line", "attn_lag", "attn_ay")):
            if hasattr(block, name):
                getattr(block, name).out_proj.bias.copy_(torch.full((D,), 0.5 * (i + 1)))
    return block


def hidden_everything_block_output(block, axes):
    """Run the block with nothing revealed, against the output assembled by hand.

    With every key hidden each attention returns its output bias (or zero), whatever the
    weights, so the block's output is ``x`` plus those, through the feed-forward.
    """
    torch.manual_seed(1)
    x = torch.randn(N_C * N_W, N_L * N_J, D)
    written = torch.ones(N_C * N_W, N_L, dtype=torch.bool)
    visible = torch.zeros(N_C * N_W, N_L * N_J, dtype=torch.bool)
    with torch.no_grad():
        got, *_ = block(x, written, visible=visible)
        added = torch.zeros(D)
        for name, axis in (("attn_line", "line"), ("attn_lag", "lag"), ("attn_ay", "ay")):
            if axis in axes and block.empty_query_bias:
                added = added + getattr(block, name).out_proj.bias
        mid = x + added
        want = mid + block.ff(block.ln3(mid))
    return got, want


@pytest.mark.parametrize("axes", AXES_CASES)
@pytest.mark.parametrize("empty_query_bias", [True, False])
def test_with_every_key_hidden_the_output_is_the_bias_or_zero_on_each_axis(axes, empty_query_bias):
    block = make_block(axes, empty_query_bias=empty_query_bias)
    got, want = hidden_everything_block_output(block, axes)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    # and the option decides: the other setting gives a different block output
    other = make_block(axes, empty_query_bias=not empty_query_bias)
    other_got, _ = hidden_everything_block_output(other, axes)
    assert not torch.allclose(got, other_got, atol=1e-4)


@pytest.mark.parametrize("empty_query_bias", [True, False])
def test_attend_gives_a_fully_hidden_sequence_the_bias_exactly_or_zero_exactly(empty_query_bias):
    torch.manual_seed(2)
    attn = torch.nn.MultiheadAttention(D, 2, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        attn.out_proj.bias.copy_(torch.linspace(-1, 1, D))
        h = torch.randn(3, 5, D)
        readable = torch.ones(3, 5, dtype=torch.bool)
        readable[1] = False  # sequence 1 reads nothing; the others read everything
        out, weights = attend(attn, h, readable, True, guard=True, empty_bias=empty_query_bias)
        plain, _ = attend(attn, h, torch.ones(3, 5, dtype=torch.bool), False, guard=True)
    want = attn.out_proj.bias.expand(5, D) if empty_query_bias else torch.zeros(5, D)
    assert torch.equal(out[1], want)
    # a sequence with something to read is untouched by the option
    assert torch.equal(out[0], plain[0]) and torch.equal(out[2], plain[2])
    # the companion's weights are zero there as well; the default keeps its own
    if empty_query_bias:
        assert torch.equal(weights[1], torch.zeros(5, 5))


def test_the_bias_gets_a_gradient_where_a_query_is_empty_and_the_default_gives_it_none():
    for flag, expect in ((True, True), (False, False)):
        block = make_block(("lag",), empty_query_bias=flag, mask="observed_cells")
        block.train()
        x = torch.randn(N_C * N_W, N_L * N_J, D)
        written = torch.ones(N_C * N_W, N_L, dtype=torch.bool)
        visible = torch.zeros(N_C * N_W, N_L * N_J, dtype=torch.bool)
        out, *_ = block(x, written, visible=visible)
        out.sum().backward()
        grad = block.attn_lag.out_proj.bias.grad
        assert (grad is not None and bool((grad.abs() > 0).all())) is expect


# -- 3. a hand-written copy of the companion's rule ------------------------------------------


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def companion_mha(x, in_w, in_b, out_w, out_b, pad, heads):
    """The companion's ``_mha``: weights zeroed for a query with no readable key, then
    ``o @ out_w + out_b`` as for every other query."""
    n, s, d = x.shape
    dh = d // heads
    q, k, v = np.split(x @ in_w + in_b, 3, axis=-1)
    split = lambda t: t.reshape(n, s, heads, dh).transpose(0, 2, 1, 3)  # noqa: E731
    q, k, v = split(q), split(k), split(v)
    scores = (q / math.sqrt(dh)) @ k.transpose(0, 1, 3, 2)
    scores = np.where(pad[:, None, None, :], -1e30, scores)
    w = softmax(scores)
    w = np.where((~pad).any(-1)[:, None, None, None], w, 0.0)
    o = (w @ v).transpose(0, 2, 1, 3).reshape(n, s, d)
    return o @ out_w + out_b


def np_weights(attn):
    return {
        "in_w": attn.in_proj_weight.detach().numpy().T,
        "in_b": attn.in_proj_bias.detach().numpy(),
        "out_w": attn.out_proj.weight.detach().numpy().T,
        "out_b": attn.out_proj.bias.detach().numpy(),
    }


def np_layer_norm(ln, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * ln.weight.detach().numpy() + ln.bias.detach().numpy()


def np_gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def test_attend_equals_the_companions_attention_on_partly_and_fully_hidden_groups():
    torch.manual_seed(3)
    attn = torch.nn.MultiheadAttention(D, 2, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        attn.in_proj_bias.normal_()
        attn.out_proj.bias.normal_()
    h = torch.randn(6, 5, D)
    readable = torch.rand(6, 5) > 0.5
    readable[0] = False  # fully hidden
    readable[3] = False
    readable[1] = True  # fully readable
    readable[2, 1:] = False
    readable[2, 0] = True
    readable[4:, ::2] = True  # one key
    with torch.no_grad():
        got, _ = attend(attn, h, readable, False, guard=True, empty_bias=True)
    want = companion_mha(h.numpy(), pad=~readable.numpy(), heads=2, **np_weights(attn))
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-4, atol=1e-5)
    # the fully hidden groups really are in the batch, and carry the bias
    np.testing.assert_allclose(got[0, 0].numpy(), attn.out_proj.bias.detach().numpy(), atol=1e-6)


def companion_block(block, x, readable, n_l, n_j, n_w):
    """The companion's ``_block`` in numpy (dropout off), on the weights of a torch block."""
    b, n_tok, d = x.shape
    heads = block.attn_line.num_heads
    # across lines, within each lag
    h = np_layer_norm(block.ln1, x).reshape(b, n_l, n_j, d).transpose(0, 2, 1, 3)
    h = h.reshape(b * n_j, n_l, d)
    pad = ~readable.reshape(b, n_l, n_j).transpose(0, 2, 1).reshape(b * n_j, n_l)
    a = companion_mha(h, pad=pad, heads=heads, **np_weights(block.attn_line))
    x = x + a.reshape(b, n_j, n_l, d).transpose(0, 2, 1, 3).reshape(b, n_tok, d)
    # across lags, within each line
    h = np_layer_norm(block.ln2, x).reshape(b * n_l, n_j, d)
    a = companion_mha(
        h, pad=~readable.reshape(b * n_l, n_j), heads=heads, **np_weights(block.attn_lag)
    )
    x = x + a.reshape(b, n_tok, d)
    # across accident years, within each company, line and lag
    bc = b // n_w
    to_ay = lambda t: t.reshape((bc, n_w, n_l, n_j) + t.shape[3:]).swapaxes(1, 2).swapaxes(2, 3)  # noqa: E731
    h = to_ay(np_layer_norm(block.ln_ay, x).reshape(b, n_l, n_j, d)).reshape(bc * n_l * n_j, n_w, d)
    pad = ~to_ay(readable.reshape(b, n_l, n_j)).reshape(bc * n_l * n_j, n_w)
    a = companion_mha(h, pad=pad, heads=heads, **np_weights(block.attn_ay))
    a = a.reshape(bc, n_l, n_j, n_w, d).transpose(0, 3, 1, 2, 4).reshape(b, n_tok, d)
    x = x + a
    ff = block.ff
    mid = (
        np_layer_norm(block.ln3, x) @ ff[0].weight.detach().numpy().T + ff[0].bias.detach().numpy()
    )
    return x + np_gelu(mid) @ ff[3].weight.detach().numpy().T + ff[3].bias.detach().numpy()


def test_the_block_equals_the_companions_block_on_a_batch_with_hidden_groups():
    n_c, n_w, n_l, n_j = 2, 4, 3, 5
    block = AxialBlock(
        D,
        2,
        0.0,
        n_l,
        n_j,
        axes=("line", "lag", "ay"),
        n_origin=n_w,
        mask="observed_cells",
        empty_query_bias=True,
    ).eval()
    torch.manual_seed(4)
    with torch.no_grad():
        for p in block.parameters():
            p.copy_(0.5 * torch.randn_like(p))
    x = torch.randn(n_c * n_w, n_l * n_j, D)
    # a staircase: accident year w has revealed lags 0 .. n_j - 1 - w, so late lags are hidden
    # for every line (a line group at a late lag reads nothing) and the last year is hidden
    # entirely (a lag group reads nothing), and a column of accident years can be empty
    visible = np.zeros((n_c * n_w, n_l, n_j), dtype=bool)
    for e in range(n_c * n_w):
        visible[e, :, : max(n_j - 1 - (e % n_w) * 2, 0)] = True
    visible[:, 2, :] &= np.arange(n_j) < 1  # line 2 has revealed one lag only
    flat = visible.reshape(n_c * n_w, n_l * n_j)
    written = torch.ones(n_c * n_w, n_l, dtype=torch.bool)
    with torch.no_grad():
        got, *_ = block(x, written, visible=torch.tensor(flat))
    want = companion_block(block, x.numpy(), flat, n_l, n_j, n_w)
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-4, atol=1e-5)
    # the batch has hidden groups on all three axes, or the comparison says nothing
    assert not visible[:, :, -1].any()  # a lag no line has revealed
    assert not visible[(n_w - 1) :: n_w].any()  # an accident year with nothing revealed
    # and the option matters on it: the default block disagrees with the companion's
    plain = AxialBlock(
        D, 2, 0.0, n_l, n_j, axes=("line", "lag", "ay"), n_origin=n_w, mask="observed_cells"
    ).eval()
    plain.load_state_dict(block.state_dict())
    with torch.no_grad():
        zero, *_ = plain(x, written, visible=torch.tensor(flat))
    assert not np.allclose(zero.numpy(), want, rtol=1e-3, atol=1e-3)


def test_the_network_equals_the_companions_forward_to_the_head_on_real_features(monkeypatch):
    """Embeddings read the companion's way (0-based line and lag, ``lk`` for the visible
    lags), the block as above, the final norm and the output layer: ``net`` per token."""
    contract = company_contract("duckdb", lines={1: [0]})
    cutoff = 4
    feats = features_for(contract, cutoff)
    _, n_l, _, n_w, n_d = contract["x"].shape
    cfg = TLRNConfig.accident_year_variant(
        d_model=8,
        n_heads=2,
        dropout=0.0,
        empty_query_bias=True,
        compact_embeddings=True,
        member="network",
    )
    head_init = torch.tensor(np.log(pooled_incremental_lr(contract, cutoff)), dtype=torch.float32)
    torch.manual_seed(5)
    net = TLRNNetwork(
        cfg, n_lines=n_l, n_lag=n_d, n_feat=feats["n_feat"], n_origin=n_w, head_init=head_init
    ).eval()
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name not in ("beta",):
                p.copy_(0.4 * torch.randn_like(p))

    captured = {}
    real = tlrn_head.log_ratios

    def spy(beta, net_out, *args, **kwargs):
        captured["net"] = net_out.detach().clone()
        return real(beta, net_out, *args, **kwargs)

    monkeypatch.setattr(tlrn_head, "log_ratios", spy)
    run_network(net, feats)
    # the companion's input: features through the projection, plus the three embeddings read
    # 0-based (line and lag) and at lk (visible lags)
    x = (
        feats["feat"].astype(np.float32) @ net.inp.weight.detach().numpy().T
        + net.inp.bias.detach().numpy()
        + net.line_emb.weight.detach().numpy()[np.asarray(feats["line_ix"]) - 1]
        + net.lag_emb.weight.detach().numpy()[np.asarray(feats["lag_ix"]) - 1]
        + net.nobs_emb.weight.detach().numpy()[np.asarray(feats["lk"])][:, None, :]
    )
    readable = np.asarray(feats["visible"]).astype(bool)
    assert (~readable).any(axis=1).all()  # every example has hidden tokens
    h = companion_block(net.blocks[0], x, readable, n_l, n_d, n_w)
    out = (
        np_layer_norm(net.ln, h) @ net.head.weight.detach().numpy().T
        + net.head.bias.detach().numpy()
    )
    np.testing.assert_allclose(
        captured["net"].numpy(), out[..., 0].reshape(-1, n_l, n_d), rtol=2e-4, atol=2e-5
    )


# -- 4. delivery through gallery.fit ----------------------------------------------------------


def test_nothing_changes_when_the_options_are_spelled_out_off():
    base = TLRNConfig.accident_year_variant(d_model=8, n_heads=2)
    assert replace(base, empty_query_bias=False, compact_embeddings=False) == base


def small_ay(**overrides) -> TLRNConfig:
    from .test_tlrn_refit import ay_config

    return ay_config(**overrides)


def small_preset(**overrides) -> TLRNConfig:
    """The preset at test size: the accident-year variant's small settings, same overrides."""
    base = small_ay(refit=True, alpha_on="first_diagonal")
    sizes = {
        k: getattr(base, k)
        for k in (
            "d_model",
            "n_heads",
            "dropout",
            "max_epochs",
            "min_epochs",
            "check_every",
            "patience",
            "batch_size",
            "ensemble_size",
            "calibration_cutoffs",
            "calibration_horizons",
            "n_strata",
            "min_per_stratum",
            "n_draws",
            "warmup",
        )
    }
    return TLRNConfig.companion_final_variant(**{**sizes, **overrides})


def _fit(config, backend):
    from .test_tlrn_refit import fit

    return fit(config, backend=backend, ay=True)


@pytest.fixture(params=["torch", "jax"])
def backend(request):
    if request.param == "jax":
        pytest.importorskip("jax")
    return request.param


def test_empty_query_bias_reaches_the_fit_on_both_backends(backend):
    """Same initial weights with and without it (it adds no parameter), so any difference
    in the trained forecasts is the rule itself, delivered from the config to the network."""
    off = _fit(small_ay(), backend)
    on = _fit(small_ay(empty_query_bias=True), backend)
    assert not np.allclose(on.member_company_reserves(), off.member_company_reserves(), rtol=1e-6)
    assert [p.shape for p in on.models_[0].parameters()] == [
        p.shape for p in off.models_[0].parameters()
    ]


def test_compact_embeddings_reaches_the_trained_networks_on_both_backends(backend):
    off = _fit(small_ay(), backend)
    on = _fit(small_ay(compact_embeddings=True), backend)
    for entry, extra in ((off, 1), (on, 0)):
        for model in entry.models_:
            assert model.line_emb.weight.shape[0] == model.n_lines + extra
            assert model.lag_emb.weight.shape[0] == model.n_lag + extra
            assert model.nobs_emb.weight.shape[0] == model.n_lag + 2
    assert (
        count(off.models_[0]) - count(on.models_[0]) == 2 * off.models_[0].line_emb.weight.shape[1]
    )


def test_the_preset_really_sets_both_options_for_a_fit(backend):
    """The preset fit is the explicitly spelled fit, bit for bit, and has the compact tables."""
    spelled = small_ay(
        refit=True, alpha_on="first_diagonal", empty_query_bias=True, compact_embeddings=True
    )
    a = _fit(small_preset(), backend)
    b = _fit(spelled, backend)
    np.testing.assert_array_equal(a.member_company_reserves(), b.member_company_reserves())
    assert a.models_[0].line_emb.weight.shape[0] == a.models_[0].n_lines
    # and it is not the fit with either option left out
    for left_out in ("empty_query_bias", "compact_embeddings"):
        c = _fit(small_preset(**{left_out: False}), backend)
        assert not np.array_equal(c.member_company_reserves(), a.member_company_reserves())
