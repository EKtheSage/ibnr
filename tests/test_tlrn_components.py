"""tlrn's named design choices: the premium head, the accident-year attention, the masks.

Skips cleanly when torch is not installed.

What this file protects. ``tlrn`` is five choices made independently (see
``gallery/nn/tlrn/components.py``), and each new choice can be finite, trainable
and wrong in a way no shape error shows. So the checks here are the ones that
each choice has to pass, written against the CHOICE rather than against one
configuration, and ``test_every_registered_choice_fits`` runs a tiny fit over
every name in the registries so a choice cannot be added without being fitted.

* the premium head, with the network switched off, must give premium times the
  pooled incremental loss ratio - the head's own closed form, the counterpart of
  ``test_head_reproduces_the_pooled_chain_ladder``;
* the accident-year attention must read across the accident years of one company
  and never across companies: perturb one company and every other company's
  output has to come back bit for bit the same;
* ``mask="observed_cells"`` must keep a token from reading a cell the forecast
  date has not revealed, and a token with nothing to read must get nothing, not
  NaN;
* no choice may let a cell after the forecast date reach the network's output.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.nn.tlrn import components  # noqa: E402
from ibnr.gallery.nn.tlrn import head as tlrn_head  # noqa: E402
from ibnr.gallery.nn.tlrn import model as tlrn_model  # noqa: E402
from ibnr.gallery.nn.tlrn.blend import blend_weight  # noqa: E402
from ibnr.gallery.nn.tlrn.config import TLRNConfig  # noqa: E402
from ibnr.gallery.nn.tlrn.model import TLRN as TLRNEntry  # noqa: E402
from ibnr.gallery.nn.tlrn.network import AxialBlock, TLRNNetwork  # noqa: E402
from ibnr.kernels.multivariate_cl import point_grid  # noqa: E402
from ibnr.kernels.nn_features import pooled_incremental_lr, tlrn_features  # noqa: E402

from .test_nn_features import company_contract, study_triangle  # noqa: E402
from .test_tlrn import AS_OF, run, tensors, tiny  # noqa: E402


def cfg(**overrides) -> TLRNConfig:
    return TLRNConfig(**{"dropout": 0.0, "d_model": 16, "n_heads": 2, **overrides})


def features(contract, cutoff=3, hi=6):
    return tlrn_features(
        contract,
        cutoff=cutoff,
        target_lo=cutoff + 1,
        target_hi=hi,
        incurred_field="incurred_loss",
        case_field="case_reserve",
    )


def network(config, f, contract, cutoff):
    head_init = np.log(pooled_incremental_lr(contract, cutoff))
    return TLRNNetwork(
        config,
        n_lines=f["n_l"],
        n_lag=f["n_d"],
        n_feat=f["n_feat"],
        n_origin=contract["x"].shape[3],
        head_init=torch.tensor(head_init, dtype=torch.float32),
    ).eval()


def forward(model, t, **kwargs):
    return run(
        model,
        t,
        fallback_logf=t["fallback_logf"],
        fallback_lr=t["fallback_lr"],
        anchor_logf=t["anchor_logf"],
        anchor_start=t["anchor_start"],
        visible=t["visible"],
        **kwargs,
    )


# -- the config ---------------------------------------------------------------------


def test_combinations_that_cannot_work_are_refused_by_name():
    with pytest.raises(ValueError, match="batch_unit='company'"):
        TLRNConfig(attention=("line", "lag", "ay"))
    with pytest.raises(ValueError, match="head='ldf'"):
        TLRNConfig(head="premium_lr", cl_anchor=True)
    with pytest.raises(ValueError, match="once each, in that order"):
        TLRNConfig(attention=("lag", "line"), batch_unit="company")
    with pytest.raises(ValueError, match="must include 'lag'"):
        TLRNConfig(attention=("line",))
    with pytest.raises(ValueError, match="cross_line=True disagrees"):
        TLRNConfig(attention=("lag",))
    for field, bad in (("head", "x"), ("mask", "x"), ("batch_unit", "x")):
        with pytest.raises(ValueError, match=field):
            TLRNConfig(**{field: bad})
    # the defaults are the published model
    assert components.attention_axes(TLRNConfig()) == ("line", "lag")
    assert components.attention_axes(TLRNConfig(cross_line=False)) == ("lag",)


def test_the_accident_year_attention_adds_exactly_one_attention_and_one_norm():
    base = TLRNNetwork(cfg(), n_lines=4, n_lag=10, n_feat=13, n_origin=10)
    with_ay = TLRNNetwork(
        cfg(attention=("line", "lag", "ay"), batch_unit="company"),
        n_lines=4,
        n_lag=10,
        n_feat=13,
        n_origin=10,
    )
    d = 16
    extra = sum(p.numel() for p in with_ay.parameters()) - sum(p.numel() for p in base.parameters())
    assert extra == (4 * d * d + 4 * d) + 2 * d
    with pytest.raises(ValueError, match="n_origin"):
        TLRNNetwork(
            cfg(attention=("line", "lag", "ay"), batch_unit="company"),
            n_lines=4,
            n_lag=10,
            n_feat=13,
        )


# -- the premium head -----------------------------------------------------------------


def test_premium_head_with_the_network_off_is_premium_times_the_pooled_ratio(backend_name):
    cutoff = 3
    contract = company_contract(backend_name)
    f = features(contract, cutoff)
    t = tensors(f)
    t["fallback_lr"] = torch.tensor(f["fallback_lr"], dtype=torch.float32)
    t["visible"] = torch.tensor(f["visible"], dtype=torch.bool)
    model = network(cfg(head="premium_lr"), f, contract, cutoff)
    out = forward(model, t, use_residual=False)

    n_l, n_d = f["n_l"], f["n_d"]
    lr = f["fallback_lr"]  # the pooled ratio the head starts at
    pred = out["pred"].detach().reshape(-1, n_l, n_d)
    lk = f["lk"]
    for b in range(len(lk)):
        for j in range(int(lk[b]), n_d):  # the cells past the latest visible lag
            np.testing.assert_allclose(pred[b, :, j].numpy(), lr[:, j], rtol=1e-5)
    # the reserve of an origin is premium times the pooled ratios it has left. The
    # grid is single precision and starts from the balance, so its last cell is
    # compared at the balance's resolution, not the increment's
    c = out["C"].detach().numpy()
    for b in range(len(lk)):
        want = f["p_lk"][b] * lr[:, int(lk[b]) :].sum(axis=1)
        np.testing.assert_allclose(
            c[b, :, -1], f["c_lk"][b] + want, rtol=1e-5, atol=1e-6 * float(c.max())
        )


def test_premium_head_scales_with_premium_not_with_the_latest_balance(backend_name):
    """Doubling premium doubles the reserve; doubling the balance leaves it alone."""
    cutoff = 3
    contract = company_contract(backend_name)
    f = features(contract, cutoff)
    t = tensors(f)
    t["fallback_lr"] = torch.tensor(f["fallback_lr"], dtype=torch.float32)
    t["visible"] = torch.tensor(f["visible"], dtype=torch.bool)
    model = network(cfg(head="premium_lr"), f, contract, cutoff)

    def reserve(c_scale, p_scale):
        """What the head adds past the latest visible lag: premium times its ratios."""
        u = dict(t, c_lk=t["c_lk"] * c_scale, p_lk=t["p_lk"] * p_scale)
        out = forward(model, u, use_residual=False)
        n_l, n_d = f["n_l"], f["n_d"]
        ahead = torch.arange(n_d).view(1, 1, n_d) >= u["lk"].view(-1, 1, 1)
        pred = out["pred"].detach().reshape(-1, n_l, n_d)
        return (pred * ahead).sum(dim=2) * u["p_lk"]

    base = reserve(1.0, 1.0)
    torch.testing.assert_close(reserve(1.0, 2.0), 2.0 * base)
    torch.testing.assert_close(reserve(2.0, 1.0), base)


def test_premium_head_substitutes_the_pooled_ratio_where_no_target_supervised(backend_name):
    cutoff = 3
    contract = company_contract(backend_name)
    f = features(contract, cutoff)
    t = tensors(f)
    t["fallback_lr"] = torch.tensor(f["fallback_lr"], dtype=torch.float32)
    t["visible"] = torch.tensor(f["visible"], dtype=torch.bool)
    model = network(cfg(head="premium_lr"), f, contract, cutoff)
    with torch.no_grad():
        model.beta += 1.0  # every learned ratio moves away from the pooled one
    support = torch.ones(f["n_l"], f["n_d"], dtype=torch.bool)
    support[0, 4] = False
    out = forward(model, t, use_residual=False, factor_support=support)["logr"].detach()
    lr = torch.log(t["fallback_lr"])
    torch.testing.assert_close(out[:, 0, 4], lr[0, 4].expand(out.shape[0]))
    torch.testing.assert_close(out[:, 1, 4], (model.beta[1, 4]).expand(out.shape[0]).detach())


def test_premium_head_caps_the_log_ratio():
    config = cfg(head="premium_lr", lr_cap=1.0)
    n_l, n_d = 2, 4
    model = TLRNNetwork(
        config, n_lines=n_l, n_lag=n_d, n_feat=8, head_init=torch.full((n_l, n_d), 3.0)
    )
    logr = tlrn_head.log_ratios(model.beta.unsqueeze(0), None, 0.5, config.lr_cap)
    assert float(logr.detach().max()) == 1.0


# -- the accident-year attention ------------------------------------------------------


def block_inputs(n_c=3, n_w=4, n_l=2, n_j=3, d=8, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n_c * n_w, n_l * n_j, d)
    written = torch.ones(n_c * n_w, n_l, dtype=torch.bool)
    visible = torch.ones(n_c * n_w, n_l * n_j, dtype=torch.bool)
    return x, written, visible, (n_c, n_w, n_l, n_j, d)


def test_accident_year_attention_never_crosses_companies_and_does_cross_years():
    x, written, visible, (n_c, n_w, n_l, n_j, d) = block_inputs()
    block = AxialBlock(
        d, 2, 0.0, n_l, n_j, axes=("line", "lag", "ay"), n_origin=n_w, mask="unwritten_lines"
    ).eval()
    with torch.no_grad():
        before, *_ = block(x, written, visible=visible)
        moved = x.clone()
        moved[0 * n_w : 1 * n_w] += 2.5  # every accident year of company 0
        after, *_ = block(moved, written, visible=visible)
        # the other companies are BIT-IDENTICAL: a mixed attention would move them
        assert torch.equal(before[n_w:], after[n_w:])
        assert not torch.equal(before[:n_w], after[:n_w])

        # within a company the years DO read each other: move one year and a
        # different year of the same company changes, which a dead axis would not
        moved = x.clone()
        moved[0] += 2.5
        after, *_ = block(moved, written, visible=visible)
        assert not torch.equal(before[1], after[1])
        assert torch.equal(before[n_w:], after[n_w:])


def test_a_batch_that_is_not_whole_companies_is_refused():
    x, written, visible, (n_c, n_w, n_l, n_j, d) = block_inputs()
    block = AxialBlock(d, 2, 0.0, n_l, n_j, axes=("line", "lag", "ay"), n_origin=n_w).eval()
    with pytest.raises(ValueError, match="whole number of companies"):
        block(x[:-1], written[:-1], visible=visible[:-1])


# -- the masks ------------------------------------------------------------------------


@pytest.mark.parametrize("axes", [("line", "lag"), ("line", "lag", "ay")])
def test_observed_cells_mask_keeps_a_token_from_reading_the_future(axes):
    x, written, visible, (n_c, n_w, n_l, n_j, d) = block_inputs()
    # the last lag of every line is not yet revealed, on every example
    visible = visible.reshape(-1, n_l, n_j).clone()
    visible[:, :, -1] = False
    visible = visible.reshape(-1, n_l * n_j)
    block = AxialBlock(d, 2, 0.0, n_l, n_j, axes=axes, n_origin=n_w, mask="observed_cells").eval()
    with torch.no_grad():
        before, *_ = block(x, written, visible=visible)
        moved = x.clone()
        hidden = ~visible
        moved[hidden] += 4.0
        after, *_ = block(moved, written, visible=visible)
    # a visible token's output reads visible keys only, so it cannot move
    assert torch.equal(before[visible], after[visible])
    # and the hidden tokens do move, so the check above is about the mask
    assert not torch.equal(before[hidden], after[hidden])


def test_a_token_with_nothing_to_read_gets_nothing_not_nan():
    x, written, visible, (n_c, n_w, n_l, n_j, d) = block_inputs()
    visible = torch.zeros_like(visible)  # the forecast date has revealed nothing
    block = AxialBlock(
        d, 2, 0.0, n_l, n_j, axes=("line", "lag", "ay"), n_origin=n_w, mask="observed_cells"
    ).eval()
    with torch.no_grad():
        out, *_ = block(x, written, visible=visible)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="no `visible`"):
        block(x, written)


# -- no choice lets the future in ---------------------------------------------------------


@pytest.mark.parametrize("head", components.HEADS)
@pytest.mark.parametrize("mask", components.MASKS)
def test_hidden_cells_do_not_reach_the_network_output(head, mask, backend_name):
    """Replace every cell past the cutoff with noise; the output must not move."""
    cutoff = 3
    contract = company_contract(backend_name)
    hidden = contract["cal_idx"] > cutoff
    rng = np.random.default_rng(5)
    noisy = dict(contract)
    noisy["values"] = np.where(hidden, rng.uniform(1, 1e6, hidden.shape), contract["values"])
    noisy["x"] = np.where(hidden, rng.normal(size=hidden.shape), contract["x"])

    config = cfg(head=head, mask=mask, attention=("line", "lag", "ay"), batch_unit="company")
    outs = []
    for source in (contract, noisy):
        f = features(source, cutoff)
        t = tensors(f)
        t["fallback_lr"] = torch.tensor(f["fallback_lr"], dtype=torch.float32)
        t["visible"] = torch.tensor(f["visible"], dtype=torch.bool)
        torch.manual_seed(0)
        model = network(config, f, contract, cutoff)  # the start is the clean data's
        with torch.no_grad():
            outs.append(forward(model, t))
    for key in ("pred", "C"):
        assert torch.equal(outs[0][key], outs[1][key]), key


# -- fitting every registered choice --------------------------------------------------


def combos():
    for head in components.HEADS:
        for mask in components.MASKS:
            for member in components.MEMBERS:
                yield head, mask, member, ("line", "lag"), "example"
                yield head, mask, member, ("line", "lag", "ay"), "company"


@pytest.mark.parametrize(("head", "mask", "member", "attention", "unit"), list(combos()))
def test_every_registered_choice_fits(head, mask, member, attention, unit, backend_name):
    config = replace(
        tiny(),
        head=head,
        mask=mask,
        member=member,
        attention=attention,
        batch_unit=unit,
        batch_size=2 if unit == "company" else 8,
    )
    entry = TLRNEntry().fit(
        study_triangle(backend_name, None),
        loss_field="paid_loss",
        as_of=AS_OF,
        config=config,
        seed=0,
    )
    reserves = entry.company_reserves()
    assert np.isfinite(reserves).all()
    assert (reserves >= 0).all()
    assert len(entry.models_) == 1
    if member == "mcl_blend":
        alpha = entry.selection_["alpha"]
        assert alpha.between(0.0, 1.0).all()
        assert float(entry.models_[0].alpha) == float(alpha[entry.selection_["kept"]].iloc[0])
    else:
        assert "alpha" not in entry.selection_.columns
    assert entry.factor_support_.shape == (2, 6 if head == "premium_lr" else 5)
    # the reserve is the ultimate less what is already paid, whatever the head
    np.testing.assert_allclose(
        np.nansum(entry.point_ultimates_ - entry.contract_["latest_cum"], axis=(1, 2)),
        reserves,
        rtol=1e-5,
    )
    repeat = TLRNEntry().fit(
        study_triangle(backend_name, None),
        loss_field="paid_loss",
        as_of=AS_OF,
        config=config,
        seed=0,
    )
    np.testing.assert_array_equal(entry.company_reserves(), repeat.company_reserves())


def test_company_batching_batches_companies(backend_name, monkeypatch):
    seen = {}
    real = tlrn_model.train_ensemble

    def spy(n_units, **kwargs):
        seen["n_units"] = n_units
        return real(n_units, **kwargs)

    monkeypatch.setattr(tlrn_model, "train_ensemble", spy)
    triangle = study_triangle(backend_name, None)
    for unit in ("example", "company"):
        TLRNEntry().fit(
            triangle,
            loss_field="paid_loss",
            as_of=AS_OF,
            config=replace(tiny(), batch_unit=unit, batch_size=2),
            seed=0,
        )
        n_c = 3
        n_w = 6
        assert seen["n_units"] == (n_c if unit == "company" else n_c * n_w)


# -- the multivariate chain ladder member -------------------------------------------------


def test_the_blend_weight_is_the_exact_minimiser():
    rng = np.random.default_rng(3)
    grid = np.linspace(0.0, 1.0, 100_001)
    for _ in range(20):
        n = int(rng.integers(1, 12))
        a, b = rng.normal(size=n) * 10, rng.normal(size=n) * 5
        got = blend_weight(a, b)
        objective = np.abs(a[None, :] + grid[:, None] * b[None, :]).sum(axis=1)
        assert 0.0 <= got <= 1.0
        # nothing on a fine grid beats it by more than the grid's own resolution
        assert np.abs(a + got * b).sum() <= objective.min() + 1e-4
    # the weight is clipped into [0, 1]: an error the network would only worsen
    # takes alpha to 0, and one it fixes more than fully takes alpha to 1
    assert blend_weight(np.array([1.0]), np.array([1.0])) == 0.0
    assert blend_weight(np.array([-5.0]), np.array([1.0])) == 1.0
    # nothing to choose between: the network is taken whole
    assert blend_weight(np.array([1.0, 2.0]), np.zeros(2)) == 1.0


def exact_chain_ladder_square(n_lob=2, n_w=6, n_d=6, seed=0):
    rng = np.random.default_rng(seed)
    factors = 1.0 + rng.uniform(0.05, 0.6, (n_lob, n_d - 1)) / np.arange(1, n_d)
    base = rng.uniform(50, 150, (n_lob, n_w))
    cum = np.empty((n_lob, n_w, n_d))
    cum[:, :, 0] = base
    for d in range(1, n_d):
        cum[:, :, d] = cum[:, :, d - 1] * factors[:, d - 1][:, None]
    return cum


def test_point_grid_recovers_an_exact_chain_ladder_triangle():
    cum = exact_chain_ladder_square()
    n_w, n_d = cum.shape[1:]
    observed = (np.arange(n_w)[:, None] + np.arange(n_d)[None, :]) < n_w
    grid, methods = point_grid(np.where(observed[None], cum, np.nan), observed)
    np.testing.assert_allclose(grid, cum, rtol=1e-9)
    # every step an origin pair reaches got an estimate
    assert "flat" not in methods


def test_point_grid_is_flat_where_no_pair_reaches_a_step():
    cum = exact_chain_ladder_square()
    n_w, n_d = cum.shape[1:]
    observed = (np.arange(n_w)[:, None] + np.arange(n_d)[None, :]) < 2  # two diagonals
    grid, methods = point_grid(np.where(observed[None], cum, np.nan), observed)
    assert methods[0] != "flat"
    assert set(methods[1:]) == {"flat"}
    # a flat step carries the balance across unchanged
    np.testing.assert_allclose(grid[:, 0, -1], grid[:, 0, 1])


def blend_features(contract, cutoff):
    return tlrn_features(
        contract,
        cutoff=cutoff,
        target_lo=cutoff + 1,
        target_hi=6,
        incurred_field="incurred_loss",
        case_field="case_reserve",
        with_mcl=True,
    )


def test_the_blend_member_forecasts_in_the_heads_layout_from_visible_cells_only(backend_name):
    cutoff = 4
    contract = company_contract(backend_name)
    f = blend_features(contract, cutoff)
    n_l, n_d = f["n_l"], f["n_d"]
    # held at the starting balance until the latest visible lag, as the head holds it
    mcl_c = f["mcl_C"]
    for b in range(f["n_ex"]):
        start = int(f["lk"][b])
        held = np.repeat(f["c_lk"][b][:, None], start, axis=1)
        np.testing.assert_array_equal(mcl_c[b, :, :start], held)
    # past it the forecast adds mcl_pred * premium to the balance
    pred = f["mcl_pred"].reshape(-1, n_l, n_d)
    for b in range(f["n_ex"]):
        start = int(f["lk"][b])
        added = pred[b, :, start:] * f["p_lk"][b][:, None]
        grown = f["c_lk"][b][:, None] + np.cumsum(added, axis=1)
        np.testing.assert_allclose(mcl_c[b, :, start:], grown, rtol=1e-9)
    # no cell the cutoff hides reaches either array
    hidden = contract["cal_idx"] > cutoff
    perturbed = dict(contract)
    perturbed["values"] = np.where(hidden, contract["values"] * 1.37 + 0.123, contract["values"])
    perturbed["x"] = np.where(hidden, contract["x"] * 1.37 + 0.123, contract["x"])
    after = blend_features(perturbed, cutoff)
    np.testing.assert_array_equal(f["mcl_pred"], after["mcl_pred"])
    np.testing.assert_array_equal(f["mcl_C"], after["mcl_C"])


def test_keep_none_averages_every_trained_member(backend_name):
    triangle = study_triangle(backend_name, None)
    every = TLRNEntry().fit(
        triangle,
        loss_field="paid_loss",
        as_of=AS_OF,
        config=replace(tiny(), keep=None),
        seed=0,
    )
    named = TLRNEntry().fit(
        triangle,
        loss_field="paid_loss",
        as_of=AS_OF,
        config=replace(tiny(), keep=2),
        seed=0,
    )
    assert every.selection_["kept"].all()
    assert len(every.models_) == 2
    np.testing.assert_array_equal(every.company_reserves(), named.company_reserves())
    assert TLRNConfig(keep=None).n_kept == 10
