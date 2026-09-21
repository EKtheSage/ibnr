"""gallery.nn.tlrn: the axial network, the development-factor head, the entry.

Skips cleanly when torch is not installed.

What this file protects. ``tlrn`` does not predict cells. It predicts a positive
log development factor per (line, development step), and the cells follow from
projecting each origin's cumulative forward with those factors. That design
makes one test decisive:
:func:`test_head_reproduces_the_pooled_chain_ladder` sets the factors to the
pooled chain ladder's, switches the network off, and requires the head's cell
predictions to sum back to the chain ladder reserve. A head that fails it is
mis-specified, and no amount of training repairs that - so it is the reference
implementation's own first check and it is the first real test here.

Everything else guards a specific way the projection can be finite and wrong:
an off-by-one in which lag a projection starts from, a factor substituted at the
wrong step, a loss that takes its absolute value before summing instead of
after, and the cross-line attention reading a line the company does not write.

The parameter count is pinned at 14,309, the reference implementation's, which
depends on all three embedding tables carrying one unused row. The count is the
only thing that can see that convention, since the unused row never changes an
output.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.nn.tlrn import head as tlrn_head  # noqa: E402
from ibnr.gallery.nn.tlrn.config import TLRNConfig  # noqa: E402
from ibnr.gallery.nn.tlrn.network import TLRNNetwork  # noqa: E402
from ibnr.kernels.nn_features import pooled_cl_factors  # noqa: E402

from .test_nn_features import company_contract, study_arrays  # noqa: E402


def cfg(**overrides) -> TLRNConfig:
    """A config with dropout off, so every test below is deterministic."""
    return TLRNConfig(**{"dropout": 0.0, **overrides})


def tensors(features: dict, device="cpu") -> dict:
    """The feature dict as the tensors ``TLRNNetwork.forward`` takes."""
    out = {
        k: torch.tensor(features[k], dtype=torch.float32, device=device)
        for k in ("feat", "target", "target_mask", "premium", "c_lk", "anchor_start", "p_lk")
    }
    out["fallback_logf"] = torch.tensor(
        features["fallback_logf"], dtype=torch.float32, device=device
    )
    out["anchor_logf"] = torch.tensor(features["anchor_logf"], dtype=torch.float32, device=device)
    for k in ("lk", "line_ix", "lag_ix"):
        out[k] = torch.tensor(features[k], dtype=torch.long, device=device)
    out["written"] = torch.tensor(features["written"], dtype=torch.bool, device=device)
    return out


def run(model: TLRNNetwork, t: dict, **kwargs) -> dict:
    return model(
        t["feat"],
        t["c_lk"],
        t["p_lk"],
        t["lk"],
        t["line_ix"],
        t["lag_ix"],
        t["written"],
        **kwargs,
    )


# -- the network ------------------------------------------------------------------


def test_parameter_count_matches_the_reference():
    """14,309 at ``d_model = 32`` with 13 features on 4 lines and 10 lags.

    Built from the per-module counts so a failure says WHICH module moved, not
    just that the total did. The three embedding tables each carry one unused
    row because the indices are 1-based, which nothing but this count can see.
    """
    model = TLRNNetwork(cfg(d_model=32, n_heads=2, n_layers=1), n_lines=4, n_lag=10, n_feat=13)
    by_module = {
        "inp": 13 * 32 + 32,
        "line_emb": (4 + 1) * 32,
        "lag_emb": (10 + 1) * 32,
        "nobs_emb": (10 + 2) * 32,
        "blocks.0.attn_line": 4 * 32 * 32 + 4 * 32,
        "blocks.0.ln1": 2 * 32,
        "blocks.0.attn_lag": 4 * 32 * 32 + 4 * 32,
        "blocks.0.ln2": 2 * 32,
        "blocks.0.ff": (32 * 64 + 64) + (64 * 32 + 32),
        "blocks.0.ln3": 2 * 32,
        "ln": 2 * 32,
        "head": 32 + 1,
        "phi": 4 * 9,
    }
    for name, want in by_module.items():
        module = model.get_submodule(name) if name != "phi" else None
        got = model.phi.numel() if module is None else sum(p.numel() for p in module.parameters())
        assert got == want, name
    assert sum(by_module.values()) == 14_309
    assert sum(p.numel() for p in model.parameters()) == 14_309

    eight = TLRNNetwork(cfg(d_model=32, n_heads=2, n_layers=1), n_lines=4, n_lag=10, n_feat=8)
    assert sum(p.numel() for p in eight.parameters()) == 14_149  # five fewer input columns

    # the output layer starts an order of magnitude below the default
    # initialisation, so a fresh network is a small correction to the learned
    # factors rather than a competing prediction. Nothing else can see this.
    assert float(model.head.weight.detach().abs().max()) < HEAD_WEIGHT_CEILING
    assert float(model.head.bias.detach().abs().max()) == 0.0
    anchored = TLRNNetwork(
        cfg(d_model=32, n_heads=2, n_layers=1, cl_anchor=True), n_lines=4, n_lag=10, n_feat=13
    )
    assert float(anchored.head.weight.detach().abs().max()) == 0.0
    assert sum(p.numel() for p in anchored.parameters()) == 14_309


#: the default initialisation of a 32-wide linear layer is uniform on
#: +/- 1/sqrt(32) = 0.177, so an unscaled head is above this with certainty
HEAD_WEIGHT_CEILING = 0.01


def test_attention_never_reads_an_unwritten_line():
    """A written line's output cannot move when an absent line's input does.

    The key padding mask is the only thing stopping it, and without the mask the
    model still runs, still trains and quietly mixes a company's real lines with
    the contract's padding for lines it does not write.
    """
    torch.manual_seed(0)
    n_l, n_d, n_feat = 3, 4, 8
    model = TLRNNetwork(cfg(d_model=16, n_heads=2), n_lines=n_l, n_lag=n_d, n_feat=n_feat).eval()
    written = torch.tensor([[True, True, False]])
    feat = torch.randn(1, n_l * n_d, n_feat)
    args = dict(
        c_lk=torch.full((1, n_l), 100.0),
        p_lk=torch.full((1, n_l), 1000.0),
        lk=torch.tensor([2]),
        line_ix=torch.repeat_interleave(torch.arange(1, n_l + 1), n_d),
        lag_ix=torch.arange(1, n_d + 1).repeat(n_l),
    )
    with torch.no_grad():
        before = model(feat, written=written, **args)
        moved = feat.clone()
        moved[:, 2 * n_d :, :] += 3.7  # the absent line's tokens
        after = model(moved, written=written, **args)
    # BIT-IDENTICAL, not merely close. A masked key's attention weight is exactly
    # zero, so the written lines' path never touches the moved numbers at all -
    # and the leak this guards against is small enough that a default float32
    # tolerance would pass straight through it (measured: 4e-6 on the
    # predictions, below assert_close's 1e-5, while the log factors move 7e-5).
    assert torch.equal(before["logf"][:, :2], after["logf"][:, :2])
    assert torch.equal(before["pred"][:, : 2 * n_d], after["pred"][:, : 2 * n_d])
    # the absent line's own output does move, which is what makes the check above
    # a statement about the mask rather than about a dead network
    assert not torch.equal(before["logf"][:, 2:], after["logf"][:, 2:])

    with pytest.raises(ValueError, match="no written line"):
        model(feat, written=torch.tensor([[False, False, False]]), **args)


# -- the head ----------------------------------------------------------------------


def test_project_matches_a_hand_chain_ladder():
    """Observed cells are held; only cells past the latest visible lag grow.

    Written out cell by cell rather than as a loop, because the whole content of
    the test is the index convention: ``lk`` is 1-BASED, so ``lk = 3`` means
    lags 1 to 3 are observed and the first projected cumulative is at lag 4,
    which is index 3.
    """
    torch.manual_seed(0)
    logf = torch.rand(1, 2, 5) * 0.3 + 0.05
    c_lk = torch.tensor([[400.0, 250.0]])
    p_lk = torch.tensor([[1000.0, 500.0]])
    lk = torch.tensor([3])
    pred, c = tlrn_head.project(logf, c_lk, p_lk, lk)
    assert c.shape == (1, 2, 6)
    assert pred.shape == (1, 12)

    for li in range(2):
        # lags 1 to 3 are observed and held at the starting balance
        torch.testing.assert_close(c[0, li, :3], c_lk[0, li].expand(3))
        # lag 4 grows by one factor, lag 6 by the last three
        torch.testing.assert_close(c[0, li, 3], c_lk[0, li] * torch.exp(logf[0, li, 2]))
        torch.testing.assert_close(c[0, li, 5], c_lk[0, li] * torch.exp(logf[0, li, 2:5].sum()))
        # the prediction is the increment over the origin's premium
        token = li * 6 + 3
        torch.testing.assert_close(pred[0, token], (c[0, li, 3] - c[0, li, 2]) / p_lk[0, li])
        # a held cell's increment is zero, except the first, which is the whole
        # starting balance
        torch.testing.assert_close(pred[0, li * 6 + 1], torch.zeros(()))
        torch.testing.assert_close(pred[0, li * 6], c_lk[0, li] / p_lk[0, li])

    # the anchored variant projects from the UNFLOORED balance, so an origin that
    # has paid nothing stays at nothing instead of starting at one dollar
    zero_start = torch.tensor([[0.0, 250.0]])
    _, anchored = tlrn_head.project(
        logf, torch.ones_like(zero_start), p_lk, lk, anchor_start=zero_start
    )
    assert float(anchored[0, 0].abs().max()) == 0.0


def test_head_reproduces_the_pooled_chain_ladder(backend_name):
    """The reference implementation's own first check.

    Put the pooled chain ladder's factors into the head, switch the network off,
    and the company-line reserves must be the chain ladder's - computed here
    from each origin's latest visible cumulative and the same factors, with no
    reference to the head at all. Float32 throughout, so the tolerance is
    relative.
    """
    cutoff, n_d = 6, 6
    arrays = study_arrays()
    contract = company_contract(backend_name, arrays)
    features = tlrn_features_at(contract, cutoff)
    paid, written = arrays["paid"], contract["line_mask"]
    factors = pooled_cl_factors(contract["values"][:, :, 0], cutoff)

    model = TLRNNetwork(
        cfg(d_model=8, n_heads=1), n_lines=2, n_lag=n_d, n_feat=features["n_feat"]
    ).eval()
    with torch.no_grad():
        # softplus inverse of the log factor, so softplus(phi) IS log f
        model.phi.copy_(torch.tensor(np.log(np.exp(np.log(factors)) - 1.0), dtype=torch.float32))
    t = tensors(features)
    with torch.no_grad():
        pred = run(model, t, use_residual=False)["pred"].numpy()

    # the head's answer: every scored cell's predicted increment, in dollars
    scored = features["target_mask"] * features["premium"]
    from_head = np.zeros((3, 2))
    for e in range(features["n_ex"]):
        ci = features["example_company"][e]
        for li in range(2):
            block = slice(li * n_d, (li + 1) * n_d)
            from_head[ci, li] += float((pred[e, block] * scored[e, block]).sum())

    # the same projection done directly, with no head involved
    direct = np.zeros((3, 2))
    for ci in range(3):
        for li in range(2):
            if not written[ci, li]:
                continue
            for w in range(6):
                lk = max(min(cutoff - w, n_d), 1)
                start = max(paid[ci, li, w, lk - 1], 1.0)
                ultimate = start * np.prod(factors[li, lk - 1 :])
                direct[ci, li] += ultimate - start

    relative = np.abs(from_head - direct) / np.maximum(np.abs(direct), 1.0)
    assert relative.max() < 1e-4, f"head vs direct chain ladder: {relative.max():.3e}"
    assert direct.sum() > 0  # the comparison is not two columns of zeros


def test_factor_support_substitutes_the_fallback(backend_name):
    """A step no training target supervised takes the observable factor instead.

    The parameter at such a step is whatever the optimiser left it at - it has
    never received a gradient - so reporting it as a learned tail would be
    reporting the initialisation.
    """
    cutoff = 6
    contract = company_contract(backend_name)
    features = tlrn_features_at(contract, cutoff)
    t = tensors(features)
    model = TLRNNetwork(
        cfg(d_model=8, n_heads=1), n_lines=2, n_lag=6, n_feat=features["n_feat"]
    ).eval()
    support = torch.ones((2, 5), dtype=torch.bool)
    support[:, -1] = False
    with torch.no_grad():
        held = run(model, t, factor_support=support, fallback_logf=t["fallback_logf"])["logf"]
        free = run(model, t)["logf"]
    fallback = t["fallback_logf"]
    # the unsupported step is the fallback for every example
    torch.testing.assert_close(held[:, :, -1], fallback[None, :, -1].expand(held.shape[0], 2))
    # the supported steps are untouched, and the two really differ at the last
    torch.testing.assert_close(held[:, :, :-1], free[:, :, :-1])
    assert not torch.allclose(free[:, :, -1], held[:, :, -1])

    with pytest.raises(ValueError, match="fallback_logf"):
        run(model, t, factor_support=support)


def test_anchor_keeps_zero_correction_at_the_chain_ladder(backend_name):
    """The anchored variant STARTS at the company's own chain ladder factors.

    Its output layer is zeroed at construction and its correction is a tanh of
    the distance from the initialisation, so at construction the correction is
    exactly zero and the model is the chain ladder - not something that arrives
    near it after training.
    """
    cutoff = 6
    contract = company_contract(backend_name)
    features = tlrn_features_at(contract, cutoff)
    t = tensors(features)
    model = TLRNNetwork(
        cfg(d_model=8, n_heads=1, cl_anchor=True), n_lines=2, n_lag=6, n_feat=features["n_feat"]
    ).eval()
    with torch.no_grad():
        out = run(model, t, anchor_logf=t["anchor_logf"], anchor_start=t["anchor_start"])
    torch.testing.assert_close(out["logf"], t["anchor_logf"])
    # and the correction really is bounded: move phi a long way and it saturates
    with torch.no_grad():
        model.phi.add_(5.0)
        moved = run(model, t, anchor_logf=t["anchor_logf"], anchor_start=t["anchor_start"])
    gap = (moved["logf"] - t["anchor_logf"]).abs()
    assert float(gap.max()) <= model.cfg.anchor_width + 1e-6
    assert float(gap.max()) > 0.5 * model.cfg.anchor_width

    with pytest.raises(ValueError, match="anchor_logf"):
        run(model, t, anchor_start=t["anchor_start"])
    with pytest.raises(ValueError, match="anchor_start"):
        run(model, t, anchor_logf=t["anchor_logf"])


def test_losses_closed_forms():
    """Two lines, three lags, hand-chosen numbers, one closed form per term.

    The distinction that matters is WHEN the absolute value is taken. The
    accident-year/line term sums signed dollar errors within a line and takes
    the absolute value after, so a development error inside one line costs
    nothing; the pooled term sums everything before the absolute value, so it
    only sees the bias; the squared-error term takes neither and sees each cell.
    """
    n_l, n_d = 2, 3
    pred = torch.tensor([[0.10, 0.16, 0.30, 0.05, 0.10, 0.25]])
    targ = torch.tensor([[0.12, 0.18, 0.30, 0.05, 0.05, 0.20]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]])
    prem = torch.tensor([[100.0, 100.0, 100.0, 200.0, 200.0, 200.0]])

    # line 0 dollar errors on its scored cells: -2 and -2, so -4 in total
    # line 1 dollar errors: 0, +10 and +10, so +20 in total
    # the two lines lean opposite ways, which is what separates the first two
    # terms: one adds 4 and 20, the other cancels them to 16
    line_errors = torch.tensor([-4.0, 20.0])
    line_actuals = torch.tensor([0.12 * 100 + 0.18 * 100, (0.05 + 0.05 + 0.20) * 200])
    want_ape = float(line_errors.abs().sum() / (line_actuals.abs().sum() + 1e-8))
    got_ape = tlrn_head.ay_line_ape_loss(pred, targ, mask, prem, n_l, n_d)
    assert float(got_ape) == pytest.approx(want_ape)

    want_pe = float(abs(line_errors.sum()) / (line_actuals.sum() + 1e-8))
    assert float(tlrn_head.pool_pe_loss(pred, targ, mask, prem)) == pytest.approx(want_pe)

    squared = ((pred - targ) ** 2 * mask).sum()
    assert float(tlrn_head.masked_mse(pred, targ, mask)) == pytest.approx(
        float(squared / mask.sum())
    )
    # an explicit denominator replaces the mask count, which is how the study's
    # fixed-denominator form is expressed
    assert float(tlrn_head.masked_mse(pred, targ, mask, 2.0)) == pytest.approx(float(squared / 2.0))
    assert float(tlrn_head.masked_mse(pred, targ, torch.zeros_like(mask))) == 0.0

    total = tlrn_head.point_loss(
        pred, targ, mask, prem, n_l, n_d, w_pe=0.5, w_mse=0.1, mse_scale=0.005
    )
    assert float(total) == pytest.approx(
        want_ape + 0.5 * want_pe + 0.1 * float(tlrn_head.masked_mse(pred, targ, mask)) / 0.005
    )
    # the two absolute-value conventions are not the same number here, which is
    # what makes the first two assertions distinguishable
    assert want_ape == pytest.approx(24.0 / 90.0)
    assert want_pe == pytest.approx(16.0 / 90.0)


def test_factor_support_is_read_off_the_masks_and_starting_lags():
    """Which factors a training set could move, from masks and lags alone."""
    n_l, n_d = 2, 4
    # one example, starting at lag 2, with a scored cell at lag 4 of line 0
    mask = np.zeros((1, n_l * n_d))
    mask[0, 3] = 1.0
    support = tlrn_head.factor_support([mask], [np.array([2])], n_l, n_d)
    # the projection from lag 2 to lag 4 crosses steps 2->3 and 3->4, which are
    # 0-based steps 1 and 2; step 0->1 is before the start and is not supervised
    np.testing.assert_array_equal(support[0], [False, True, True])
    np.testing.assert_array_equal(support[1], [False, False, False])

    # a second training set unions in, it does not replace
    other = np.zeros((1, n_l * n_d))
    other[0, n_d + 1] = 1.0
    both = tlrn_head.factor_support([mask, other], [np.array([2]), np.array([1])], n_l, n_d)
    np.testing.assert_array_equal(both[0], [False, True, True])
    np.testing.assert_array_equal(both[1], [True, False, False])


def tlrn_features_at(contract: dict, cutoff: int) -> dict:
    """The 13-feature form at ``cutoff``, scored over everything past it."""
    from ibnr.kernels.nn_features import tlrn_features

    last = contract["n_w"] + contract["n_d"] - 1
    return tlrn_features(
        contract,
        cutoff=cutoff,
        target_lo=cutoff + 1,
        target_hi=last,
        incurred_field="incurred_loss",
        case_field="case_reserve",
    )
