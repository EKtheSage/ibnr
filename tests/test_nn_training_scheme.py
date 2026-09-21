"""gallery.nn._training: the shared training loop's scheme extensions.

The first test is the one every other test in this file rests on: with every
new argument left at its default, the extended loop must produce the SAME
weights and the SAME history as the loop on ``origin/main`` before this change,
bit for bit, on this machine, in this process. The reference is a verbatim copy
of that loop kept here rather than a stored hash, because a hash of float32
weights is a property of the platform and its BLAS, and this test has to pass
on the Windows dev box and on the Linux CI jobs alike.

The remaining tests check each extension by its EFFECT on the trained weights
or on what reached the loss callback, never by inspecting a signature: an
argument that is accepted and never read is the repository's named
inert-parameter bug class.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.nn._training import train_ensemble, warmup_cosine  # noqa: E402


def _reference_train_ensemble(
    n_cohorts,
    *,
    config,
    seed,
    make_model,
    train_loss,
    val_loss,
    min_cutoff,
    val_cutoff,
    device,
    show_progress=False,
):
    """VERBATIM copy of train_ensemble from origin/main at 482ff9b. Do not edit."""
    import torch

    models = []
    histories = []
    for member in range(config.ensemble_size):
        member_seed = None if seed is None else seed + 1000 * member
        if member_seed is not None:
            torch.manual_seed(member_seed)
        rng = np.random.default_rng(member_seed)
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        best_val, best_state, patience_left = math.inf, None, config.patience
        history = []
        for epoch in range(config.max_epochs):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            perm = rng.permutation(n_cohorts)
            for start in range(0, n_cohorts, config.batch_size):
                idx = torch.tensor(perm[start : start + config.batch_size], device=device)
                cutoffs = torch.tensor(
                    rng.integers(min_cutoff, val_cutoff, size=len(idx)), device=device
                )
                loss = train_loss(model, idx, cutoffs)
                if loss is None:
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            model.eval()
            with torch.no_grad():
                epoch_val = float(val_loss(model))
            history.append(
                {"epoch": epoch, "train": epoch_loss / max(n_batches, 1), "val": epoch_val}
            )
            if show_progress:
                print(f"member {member} epoch {epoch}: val {epoch_val:.4f}")
            if epoch_val < best_val - 1e-6:
                best_val, patience_left = epoch_val, config.patience
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        histories.append(history)
    return models, histories


class Tiny(torch.nn.Module):
    """A 3 -> 1 linear map with dropout, so weight init, dropout and batch order
    all consume the member's random stream."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)
        self.drop = torch.nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(self.lin(x)).squeeze(-1)


def make_problem(n: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    y = x @ torch.tensor([1.0, -2.0, 0.5]) + 0.1 * torch.randn(n, generator=g)
    return x, y


def loops(x, y, *, record=None):
    """The three callbacks train_ensemble takes. The cutoffs enter the loss, so a
    change in how they are drawn changes the weights; ``record`` collects each
    batch's cutoffs for the sampling tests."""

    def make_model():
        return Tiny()

    def train_loss(model, idx, cutoffs):
        if record is not None:
            record.append(cutoffs.detach().cpu().numpy().copy())
        return ((model(x[idx]) - y[idx]) ** 2 * (1 + 0.01 * cutoffs.float())).mean()

    def val_loss(model):
        return float(((model(x) - y) ** 2).mean())

    return make_model, train_loss, val_loss


def config(**over):
    base = dict(
        ensemble_size=2,
        batch_size=4,
        max_epochs=6,
        patience=3,
        lr=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


COMMON = dict(min_cutoff=2, val_cutoff=5, device=torch.device("cpu"))


def _callbacks(x, y, **over):
    make_model, train_loss, val_loss = loops(x, y)
    kwargs = dict(make_model=make_model, train_loss=train_loss, val_loss=val_loss)
    kwargs.update(over)
    return kwargs


def _states(models):
    return [{k: v.clone() for k, v in m.state_dict().items()} for m in models]


def _assert_same_fit(a, b):
    models_a, hist_a = a
    models_b, hist_b = b
    assert len(models_a) == len(models_b)
    for sa, sb in zip(_states(models_a), _states(models_b), strict=True):
        assert sa.keys() == sb.keys()
        for k in sa:
            assert torch.equal(sa[k], sb[k]), k
    assert [[(h["epoch"], h["train"], h["val"]) for h in hs] for hs in hist_a] == [
        [(h["epoch"], h["train"], h["val"]) for h in hs] for hs in hist_b
    ]


def test_defaults_reproduce_the_reference_loop_bit_for_bit():
    x, y = make_problem()
    ref = _reference_train_ensemble(16, config=config(), seed=3, **_callbacks(x, y), **COMMON)
    new = train_ensemble(16, config=config(), seed=3, **_callbacks(x, y), **COMMON)
    _assert_same_fit(ref, new)


def test_defaults_reproduce_the_reference_loop_when_early_stopping_fires():
    """A validation loss that never improves stops after `patience` epochs; the
    break point must land on the same epoch in both loops."""
    x, y = make_problem()
    kw = dict(_callbacks(x, y, val_loss=lambda model: 1.0), **COMMON)
    cfg = dict(patience=2, max_epochs=20)
    ref = _reference_train_ensemble(16, config=config(**cfg), seed=1, **kw)
    new = train_ensemble(16, config=config(**cfg), seed=1, **kw)
    _assert_same_fit(ref, new)
    assert len(ref[1][0]) == 3  # epoch 0 improves from inf; epochs 1 and 2 exhaust patience 2


def test_warmup_cosine_matches_the_r_learning_rate_multiplier():
    """Values of learning_rate_multiplier(epoch, schedule_epochs, warmup) from the
    R study, checked at the corners: linear warmup, the peak, the midpoint, the end."""
    f = warmup_cosine(100, warmup=20)
    assert f(1) == pytest.approx(0.05)
    assert f(20) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.5)  # progress 40/80 -> cos(pi/2) = 0
    assert f(100) == pytest.approx(0.0)
    assert f(150) == pytest.approx(0.0)  # progress is capped at 1


def test_warmup_cosine_edge_cases():
    one = warmup_cosine(1)
    assert one(1) == 1.0 and one(2) == 0.0  # a one-epoch run still updates once
    short = warmup_cosine(5, warmup=20)  # warmup capped at schedule_epochs - 1 = 4
    assert short(4) == pytest.approx(1.0)
    assert short(5) == pytest.approx(0.0)
    none = warmup_cosine(10, warmup=0)
    assert none(1) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.1)))


def test_warmup_cosine_refusals():
    with pytest.raises(ValueError, match="schedule_epochs"):
        warmup_cosine(0)
    with pytest.raises(ValueError, match="warmup"):
        warmup_cosine(10, warmup=-1)
    with pytest.raises(ValueError, match="epoch"):
        warmup_cosine(10)(0)


def _run(cfg=None, seed=3, **extra):
    x, y = make_problem()
    return train_ensemble(
        16, config=cfg or config(), seed=seed, **_callbacks(x, y), **COMMON, **extra
    )


def test_a_zero_schedule_freezes_the_weights():
    """Effect, not signature: multiplier 0 at every epoch means no parameter moves,
    while the default run moves them."""
    torch.manual_seed(3)
    init = _states([Tiny()])[0]  # member 0 seeds torch with 3 + 1000*0 before make_model()
    frozen, _ = _run(schedule=lambda epoch: 0.0)
    moved, _ = _run()
    for k in init:
        assert torch.equal(frozen[0].state_dict()[k], init[k]), k
        assert not torch.equal(moved[0].state_dict()[k], init[k]), k


def test_the_schedule_is_called_once_per_epoch_with_one_based_epochs():
    seen = []

    def spy(epoch):
        seen.append(epoch)
        return 1.0

    _, hist = _run(cfg=config(ensemble_size=1, max_epochs=4, patience=10), schedule=spy)
    assert seen == [1, 2, 3, 4]
    assert len(hist[0]) == 4


def test_a_schedule_of_one_reproduces_the_default_run():
    _assert_same_fit(_run(schedule=lambda epoch: 1.0), _run())


def _weight_frozen_groups(model):
    """The weight at rate 0, the bias on config.lr."""
    return [{"params": [model.lin.weight], "lr": 0.0}, {"params": [model.lin.bias]}]


def test_param_groups_give_each_group_its_own_rate():
    """A group with lr 0 stays put while the other trains."""
    torch.manual_seed(3)
    init = _states([Tiny()])[0]
    models, _ = _run(cfg=config(ensemble_size=1), param_groups=_weight_frozen_groups)
    state = models[0].state_dict()
    assert torch.equal(state["lin.weight"], init["lin.weight"])
    assert not torch.equal(state["lin.bias"], init["lin.bias"])


def test_param_groups_and_schedule_compose():
    """The schedule scales each group's OWN base rate: a group at lr 0 stays at 0
    under any multiplier, and the other group still moves at multiplier 0.5."""
    torch.manual_seed(3)
    init = _states([Tiny()])[0]
    models, _ = _run(
        cfg=config(ensemble_size=1),
        param_groups=_weight_frozen_groups,
        schedule=lambda epoch: 0.5,
    )
    state = models[0].state_dict()
    assert torch.equal(state["lin.weight"], init["lin.weight"])
    assert not torch.equal(state["lin.bias"], init["lin.bias"])


def test_schedule_refusals():
    with pytest.raises(ValueError, match="schedule"):
        _run(schedule=lambda epoch: -0.5)
    with pytest.raises(ValueError, match="schedule"):
        _run(schedule=lambda epoch: float("nan"))
