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

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.nn.tlrn import head as tlrn_head  # noqa: E402
from ibnr.gallery.nn.tlrn.config import TLRNConfig  # noqa: E402
from ibnr.gallery.nn.tlrn.model import TLRN as TLRNEntry  # noqa: E402
from ibnr.gallery.nn.tlrn.model import _refuse_overlap  # noqa: E402
from ibnr.gallery.nn.tlrn.network import TLRNNetwork  # noqa: E402
from ibnr.kernels.nn_features import pooled_cl_factors  # noqa: E402

from .test_nn_features import company_contract, study_arrays, study_triangle  # noqa: E402


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


# -- the entry ---------------------------------------------------------------------


def tiny() -> TLRNConfig:
    """The smallest config that still runs every step of the protocol.

    Three training cutoffs, two validation diagonals, two members of which one
    is kept, and a calibration over three cutoffs - the fixture's ``c_max`` is
    6, so the horizons are 3, 2 and 1. Nothing here asserts predictive quality.
    """
    return TLRNConfig(
        d_model=8,
        n_heads=1,
        n_layers=1,
        dropout=0.0,
        max_epochs=3,
        min_epochs=0,
        check_every=1,
        patience=5,
        batch_size=8,
        ensemble_size=2,
        keep=1,
        min_cutoff=2,
        val_diagonals=2,
        calibration_cutoffs=(3, 4, 5),
        calibration_horizons=(1, 2, 3),
        n_strata=1,
        min_per_stratum=3,
        n_draws=50,
        warmup=1,
    )


AS_OF = "2005-12-31"  # the fixture's full square, sliced to a run-off triangle


def fit_tiny(backend_name, arrays=None, *, seed=0, incurred=False, entry=None, **kwargs):
    """``entry`` refits an existing instance, which the atomicity test needs."""
    roles = {"incurred_field": "incurred_loss", "case_field": "case_reserve"} if incurred else {}
    return (entry or TLRNEntry()).fit(
        study_triangle(backend_name, arrays),
        loss_field="paid_loss",
        as_of=AS_OF,
        config=tiny(),
        seed=seed,
        **roles,
        **kwargs,
    )


def test_fit_selection_and_history(backend_name):
    """Every trained member is reported; only the kept ones survive as models."""
    entry = fit_tiny(backend_name)
    table = entry.selection_
    assert len(table) == 2
    assert list(table.columns) == [
        "member",
        "best_epoch",
        "epochs_run",
        "validation_ay_line_ape",
        "validation_company_ape",
        "kept",
    ]
    assert int(table["kept"].sum()) == 1
    # the kept member is the one with the lowest held-out accident-year/line
    # error, which is the score the selection is made on
    best = table.sort_values(["validation_ay_line_ape", "member"]).iloc[0]["member"]
    assert int(table.loc[table["kept"], "member"].iloc[0]) == int(best)
    assert len(entry.models_) == len(entry.history_) == 1
    assert (table["epochs_run"] == 3).all()
    assert table["validation_company_ape"].notna().all()
    # the factor support covers 2 lines x 5 development steps
    assert entry.factor_support_.shape == (2, 5)


def test_point_layout_and_reserve_identity(backend_name):
    """The multi-line layout for one company, every (company, line, origin) for none."""
    entry = fit_tiny(backend_name)
    company = entry.cohorts()[0]
    one = entry.point(company)
    assert len(one) == 2 * 6 + 2 + 1  # per (line, origin), per-line totals, grand total
    assert one["label"].iloc[-1] == "total"
    assert list(one["label"][12:14]) == ["lob_a/total", "lob_b/total"]
    per_lob = one.loc[one["label"].str.endswith("/total"), "point"].sum()
    assert one["point"].iloc[-1] == pytest.approx(per_lob)

    every = entry.point()
    assert len(every) == 3 * 2 * 6
    assert "label" not in every.columns
    assert set(every.columns) >= {"company_code", "line_of_business", "origin_period", "point"}

    # the ultimate is the cumulative paid to date plus the projected reserve
    written = entry.contract_["line_mask"][:, :, None]
    np.testing.assert_allclose(
        np.where(written, entry.point_ultimates_ - entry.contract_["latest_cum"], 0.0),
        np.where(written, entry.point_reserves_, 0.0),
    )
    assert np.isnan(entry.point_ultimates_[~entry.contract_["line_mask"]]).all()


def test_the_cumulative_grid_holds_the_observed_cells(backend_name):
    """``point_cumulative_`` is the triangle where it is observed and the
    projection where it is not, so a caller can read the next diagonal off it."""
    arrays = study_arrays()
    entry = fit_tiny(backend_name, arrays)
    c_max, n_d = 6, 6
    for ci in range(3):
        for li in range(2):
            for w in range(6):
                lk = max(min(c_max - w, n_d), 1)
                np.testing.assert_allclose(
                    entry.point_cumulative_[ci, li, w, :lk], arrays["paid"][ci, li, w, :lk]
                )
                if lk < n_d:
                    # the projection grows, so the first projected cell minus the
                    # anchor is the next calendar diagonal's point
                    assert (
                        entry.point_cumulative_[ci, li, w, lk] > arrays["paid"][ci, li, w, lk - 1]
                    )
    reserve = entry.point_cumulative_[:, :, :, -1] - entry.contract_["latest_cum"]
    written = entry.contract_["line_mask"][:, :, None]
    np.testing.assert_allclose(
        np.where(written, reserve, 0.0), np.where(written, entry.point_reserves_, 0.0), rtol=1e-5
    )


def test_predict_is_calibrated_draws_around_the_point(backend_name):
    """One target per company, drawn from the kept checkpoints' own history."""
    entry = fit_tiny(backend_name)
    company = entry.cohorts()[0]
    pred = entry.predict(company, seed=3)
    assert pred.samples.shape == (50, 1)
    assert list(pred.targets["label"]) == ["total"]

    # EVERY pool is centred on its own median, which is what makes the draws a
    # spread around the point rather than a spread around the point shifted by
    # however biased this model happened to be historically. Checked on the
    # pools themselves: on a small stratum the median of fifty resampled draws
    # is too blunt to tell a centred pool from an uncentred one.
    for pool in entry.calibration_.pools:
        assert np.median(pool) == pytest.approx(0.0, abs=1e-12)
    point = entry.company_reserves()[0] + entry.company_anchors()[0]
    assert np.median(pred.samples) == pytest.approx(point, rel=0.2)

    again = entry.predict(company, seed=3)
    np.testing.assert_array_equal(pred.samples, again.samples)
    assert not np.array_equal(entry.predict(company, seed=4).samples, pred.samples)

    every = entry.predict(seed=3)
    assert every.samples.shape == (50, 3)
    # one company drawn alone is the same company's column when every company is drawn
    np.testing.assert_array_equal(every.samples[:, [0]], pred.samples)

    # and the same spread can be put around any other reserve vector
    other = entry.predict_reserve_draws(entry.company_reserves() * 1.5, seed=3)
    assert other.shape == (50, 3)
    assert np.median(other[:, 0]) > np.median(pred.samples) - entry.company_anchors()[0]


def test_realized_ultimates_align(backend_name):
    """One number per company, read from the FULL triangle at the last lag."""
    arrays = study_arrays()
    full = study_triangle(backend_name, arrays)
    entry = fit_tiny(backend_name, arrays)
    company = entry.cohorts()[0]

    one = entry.realized_ultimates(full, company)
    assert one.shape == (1,)
    assert one[0] == pytest.approx(arrays["paid"][0, :, :, -1].sum())
    every = entry.realized_ultimates(full)
    assert every.shape == (3,)
    np.testing.assert_allclose(every, arrays["paid"][:, :, :, -1].sum(axis=(1, 2)))

    # the base class's four blocks, point errors included
    scores = entry.evaluate(one, company)
    assert set(scores) == {"summary", "percentiles", "crps", "point"}
    assert len(scores["percentiles"]) == 1
    errors = scores["point"]["errors"]
    assert len(errors) == 1
    assert float(errors["outcome"].iloc[0]) == pytest.approx(one[0])
    # the estimate is the DRAW MEAN, which is unseeded here, so the row is
    # checked for internal consistency and for sitting near the entry's own
    # deterministic point rather than against a second unseeded draw
    assert float(errors["error"].iloc[0]) == pytest.approx(
        float(errors["estimate"].iloc[0]) - float(errors["outcome"].iloc[0])
    )
    point = entry.company_reserves()[0] + entry.company_anchors()[0]
    assert float(errors["estimate"].iloc[0]) == pytest.approx(point, rel=0.2)
    # This entry's ONLY target is the company total, and the point metrics drop
    # a target labelled "total" because it is otherwise the sum of the others
    # counted twice. So there is nothing left to compute them on, and the block
    # says which exclusion emptied it rather than reporting a number built from
    # one row. The cross-model point board reaches this entry through
    # reserve_rows(point="native"), which reads the native point rather than
    # this per-target table.
    assert scores["point"]["metrics"] is None
    assert scores["point"]["excluded"]["total"] == 1


def test_seed_determinism(backend_name):
    """Two fits under one seed give the same point; a different seed does not."""
    arrays = study_arrays()
    same = [fit_tiny(backend_name, arrays, seed=5).point_ultimates_ for _ in range(2)]
    np.testing.assert_array_equal(same[0], same[1])
    other = fit_tiny(backend_name, arrays, seed=6).point_ultimates_
    assert not np.array_equal(same[0], other)


def test_thirteen_feature_form(backend_name):
    """Naming both roles widens the input projection; naming one is refused."""
    paid_only = fit_tiny(backend_name)
    assert paid_only.models_[0].inp.in_features == 8
    both = fit_tiny(backend_name, incurred=True)
    assert both.models_[0].inp.in_features == 13
    assert both.feature_stats_["n_feat"] == 13

    with pytest.raises(ValueError, match="both"):
        TLRNEntry().fit(
            study_triangle(backend_name),
            loss_field="paid_loss",
            incurred_field="incurred_loss",
            as_of=AS_OF,
            config=tiny(),
        )


def test_config_refusals():
    with pytest.raises(ValueError, match="keep"):
        TLRNConfig(keep=3, ensemble_size=2)
    with pytest.raises(ValueError, match="val_diagonals"):
        TLRNConfig(val_diagonals=0)
    with pytest.raises(ValueError, match="patience"):
        TLRNConfig(patience=0)
    with pytest.raises(ValueError, match="check_every"):
        TLRNConfig(check_every=0)
    with pytest.raises(ValueError, match="min_epochs"):
        TLRNConfig(min_epochs=10, max_epochs=5)
    with pytest.raises(ValueError, match="tail_policy"):
        TLRNConfig(tail_policy="guess")
    with pytest.raises(ValueError, match="cutoff_sampling"):
        TLRNConfig(cutoff_sampling="per_cell")
    with pytest.raises(ValueError, match="n_heads"):
        TLRNConfig(d_model=8, n_heads=3)


def test_fit_refusals(backend_name):
    """A calibration cutoff the triangle cannot score, and no room to train."""
    t = study_triangle(backend_name)
    with pytest.raises(ValueError, match="calibration cutoff"):
        TLRNEntry().fit(
            t,
            loss_field="paid_loss",
            as_of=AS_OF,
            config=replace(tiny(), calibration_cutoffs=(9,)),
        )
    with pytest.raises(ValueError, match="training cutoffs"):
        TLRNEntry().fit(
            t, loss_field="paid_loss", as_of=AS_OF, config=replace(tiny(), min_cutoff=4)
        )


def test_no_leak_through_the_entry(backend_name):
    """Nothing past the valuation date reaches the fit, and the last visible
    diagonal does.

    The first half is the leak check: move every cell the ``as_of`` slice drops
    and the point must be bit-identical. The second half is what makes the first
    half a test rather than a statement that the model ignores its data - move
    the last diagonal the fit DOES see and the point must change.
    """
    arrays = study_arrays()
    base = fit_tiny(backend_name, arrays, seed=2).point_ultimates_

    hidden = {k: v.copy() for k, v in arrays.items()}
    for w in range(6):
        for d in range(6):
            if w + d + 1 > 6:  # past the valuation date, dropped by as_of
                for key in ("paid", "incurred", "case"):
                    hidden[key][:, :, w, d] = hidden[key][:, :, w, d] * 1.37 + 0.123
    np.testing.assert_array_equal(base, fit_tiny(backend_name, hidden, seed=2).point_ultimates_)

    visible = {k: v.copy() for k, v in arrays.items()}
    for w in range(6):
        d = 5 - w  # the latest visible diagonal
        for key in ("paid", "incurred", "case"):
            visible[key][:, :, w, d] = visible[key][:, :, w, d] * 1.37 + 0.123
    assert not np.array_equal(base, fit_tiny(backend_name, visible, seed=2).point_ultimates_)


def test_training_and_validation_targets_cannot_share_a_cell():
    """The guard `fit` runs on its own windows, on hand-made masks.

    Validation picks which member is kept, so a cell scored in both windows
    makes that choice on data every member trained on. The check is on the masks
    rather than on the arithmetic that built them, because the arithmetic is
    what a change here gets wrong.
    """
    trained = np.array([[1.0, 1.0, 0.0, 0.0]])
    apart = np.array([[0.0, 0.0, 1.0, 1.0]])
    _refuse_overlap([trained], [apart])  # disjoint: nothing to say
    _refuse_overlap([], [apart])  # a fit with no validation set at all
    with pytest.raises(ValueError, match="both a training target and a validation target"):
        _refuse_overlap([trained], [np.array([[0.0, 1.0, 1.0, 0.0]])])


def test_a_fully_developed_triangle_reserves_nothing(backend_name):
    """Every cell observed means nothing left to forecast, and that is an answer.

    The fixture's full square is such a triangle: fitted without an ``as_of``
    its latest diagonal is the last one, so the final scoring window is empty,
    the reserve is zero and the ultimate is what has already been paid. Refusing
    it would make this the only entry that cannot read a run-off triangle that
    has finished running off.
    """
    arrays = study_arrays()
    entry = TLRNEntry().fit(
        study_triangle(backend_name, arrays),
        loss_field="paid_loss",
        config=replace(tiny(), calibration_cutoffs=(6, 7, 8), calibration_horizons=(3, 4, 5)),
        seed=0,
    )
    written = entry.contract_["line_mask"][:, :, None]
    np.testing.assert_allclose(np.where(written, entry.point_reserves_, 0.0), 0.0)
    np.testing.assert_allclose(
        np.where(written, entry.point_ultimates_, 0.0),
        np.where(written, arrays["paid"][:, :, :, -1], 0.0),
    )
    np.testing.assert_allclose(entry.company_reserves(), 0.0)


def test_a_failed_refit_leaves_the_previous_fit_intact(backend_name, monkeypatch):
    """``fit`` is atomic: a fit that raises must not half-replace the last one.

    The shared parametrized check in ``tests/test_fit_atomicity.py`` stubs the
    trainer with sentinel objects, which works for an entry that stores what
    training returned and stops. This one computes its point, its selection
    table and its calibration from the trained members, so the stub cannot
    stand in for them and the claim is asserted against a real fit here.
    """
    arrays = study_arrays()
    entry = fit_tiny(backend_name, arrays, seed=1)
    before = {
        "point": entry.point_ultimates_.copy(),
        "selection": entry.selection_.copy(),
        "models": entry.models_,
        "contract": entry.contract_,
        "calibration": entry.calibration_,
        "size": entry.company_size_.copy(),
        "members": entry.member_company_reserves().copy(),
    }

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("ibnr.gallery.nn.tlrn.model.train_ensemble", boom)
    moved = {k: v * 3.0 for k, v in arrays.items()}
    with pytest.raises(RuntimeError, match="boom"):
        fit_tiny(backend_name, moved, seed=1, entry=entry)

    np.testing.assert_array_equal(entry.point_ultimates_, before["point"])
    pd.testing.assert_frame_equal(entry.selection_, before["selection"])
    assert entry.models_ is before["models"]
    assert entry.contract_ is before["contract"]
    assert entry.calibration_ is before["calibration"]
    np.testing.assert_array_equal(entry.company_size_, before["size"])
    np.testing.assert_array_equal(entry.member_company_reserves(), before["members"])


# -- every member, and members trained in parallel ------------------------------------


def test_every_trained_member_keeps_its_company_reserves(backend_name):
    """The members the selection drops are scored too, and the ensemble is their mean.

    The kept ensemble averages its members' forecast cells and a reserve is a sum of
    cells, so the kept rows must average to ``company_reserves()`` exactly; any other
    combination of members can then be scored without refitting.
    """
    entry = fit_tiny(backend_name, seed=3)
    members = entry.member_company_reserves()
    assert members.shape == (2, len(entry.cohorts()))
    kept = entry.selection_["kept"].to_numpy()
    assert kept.sum() == 1
    np.testing.assert_allclose(members[kept].mean(axis=0), entry.company_reserves(), rtol=1e-6)
    assert not np.allclose(members[0], members[1])  # the dropped member is its own forecast


def test_keeping_every_member_averages_them_all(backend_name):
    """``keep = ensemble_size`` is the average of every trained member."""
    arrays = study_arrays()
    one = fit_tiny(backend_name, arrays, seed=3)
    every = TLRNEntry().fit(
        study_triangle(backend_name, arrays),
        loss_field="paid_loss",
        as_of=AS_OF,
        config=replace(tiny(), keep=2),
        seed=3,
    )
    assert every.selection_["kept"].all()
    np.testing.assert_allclose(
        every.company_reserves(), one.member_company_reserves().mean(axis=0), rtol=1e-6
    )


def test_members_trained_in_processes_are_the_members_trained_in_one():
    """``processes`` changes where the members train, never what they learn.

    Member m is seeded ``seed + 1000 * m`` wherever it runs, and each worker uses the
    calling process's torch thread count, so the two fits must agree exactly: the same
    selection, the same member forecasts and the same kept weights. The caller runs on
    one thread here, which is not torch's default, so a worker that ignored the count
    it was sent would be refused by the pool's own check.
    """
    arrays = study_arrays()
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        one = fit_tiny("duckdb", arrays, seed=7)
        two = fit_tiny("duckdb", arrays, seed=7, processes=2)
    finally:
        torch.set_num_threads(before)
    pd.testing.assert_frame_equal(one.selection_, two.selection_)
    np.testing.assert_array_equal(one.member_company_reserves(), two.member_company_reserves())
    assert one.history_ == two.history_
    for a, b in zip(one.models_, two.models_, strict=True):
        for (name_a, value_a), (name_b, value_b) in zip(
            a.state_dict().items(), b.state_dict().items(), strict=True
        ):
            assert name_a == name_b
            assert torch.equal(value_a, value_b), name_a


@pytest.mark.parametrize("processes", [0, -1, 1.5, "2"])
def test_processes_must_be_a_positive_int(processes):
    with pytest.raises(ValueError, match="processes"):
        fit_tiny("duckdb", processes=processes)


def test_an_unguarded_script_gets_an_error_not_a_hang(tmp_path):
    """A worker process re-imports the script that started it, so a script that calls
    ``fit(processes=...)`` at top level, with no ``if __name__ == "__main__":`` guard,
    starts workers that each try to start workers, and Python stops them while they
    start. The caller must be told so, promptly. Before the setup data went through a
    file, the caller instead blocked forever writing that data into a pipe the stopped
    worker never read."""
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from tests.test_tlrn import fit_tiny, study_arrays\n"
        'fit_tiny("duckdb", study_arrays(), seed=7, processes=2)\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=180
    )
    assert result.returncode != 0
    assert 'if __name__ == "__main__":' in result.stderr
    assert "processes" in result.stderr


def test_processes_train_on_the_cpu_only():
    """A worker process cannot share the caller's accelerator, so it is refused by name
    rather than silently trained on the CPU."""
    with pytest.raises(ValueError, match="processes"):
        fit_tiny("duckdb", processes=2, device="meta")


def test_backend_jax_without_jax_says_how_to_install_it_before_any_work(monkeypatch):
    """``backend="jax"`` with jax missing names the extra, and says so before the features
    are built. This lives here, not in ``test_tlrn_jax.py``, because that file skips
    without jax and this is the case where jax is absent: it runs on the [nn] leg,
    which installs torch and not jax. jax is hidden by a ``None`` in ``sys.modules``,
    which makes ``import jax`` fail as it does when jax is not installed."""
    import ibnr.gallery.nn.tlrn as tlrn_package

    monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.delitem(sys.modules, "ibnr.gallery.nn.tlrn.jax_backend", raising=False)
    monkeypatch.delattr(tlrn_package, "jax_backend", raising=False)

    def built(*args, **kwargs):
        raise AssertionError("the features were built before jax was looked for")

    monkeypatch.setattr("ibnr.gallery.nn.tlrn.model.nn_company_data", built)
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'ibnr\[jax\]'") as info:
        fit_tiny("duckdb", backend="jax")
    assert info.value.name == "jax"
