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


def test_members_trains_each_named_member_exactly_as_the_whole_run_does():
    """``members`` lets one ensemble be spread over several processes: member m trained
    alone must be member m of the whole run, weights and history alike."""
    cfg = config(ensemble_size=3)
    whole_models, whole_history = _run(cfg)
    for m in range(3):
        models, history = _run(cfg, members=[m])
        _assert_same_fit(([whole_models[m]], [whole_history[m]]), (models, history))
        assert {h["member"] for h in history[0]} == {m}


@pytest.mark.parametrize("members", [[2], [-1], [0, 0], [], ["0"], [True], 0])
def test_members_refusals(members):
    with pytest.raises(ValueError, match="members"):
        _run(config(ensemble_size=2), members=members)


def test_members_and_keep_do_not_combine():
    """``keep`` selects across the whole ensemble, which a subset of it cannot do."""
    with pytest.raises(ValueError, match="members"):
        _run(config(ensemble_size=2), members=[0], keep=1)


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


def _flat_run(cfg, **extra):
    """A validation loss that never improves, so early stopping decides the length."""
    x, y = make_problem()
    return train_ensemble(
        16,
        config=cfg,
        seed=1,
        **_callbacks(x, y, val_loss=lambda model: 1.0),
        **COMMON,
        **extra,
    )


def test_min_epochs_holds_early_stopping_back():
    """Patience 1 with a flat validation loss would stop after epoch 2 (0-based
    epoch 1); min_epochs=5 keeps training through epoch 5."""
    _, hist = _flat_run(config(ensemble_size=1, patience=1, max_epochs=8))
    assert len(hist[0]) == 2
    _, hist = _flat_run(config(ensemble_size=1, patience=1, max_epochs=8), min_epochs=5)
    assert len(hist[0]) == 5


def test_check_every_validates_on_multiples_and_on_the_last_epoch():
    calls = []

    def counting_val(model):
        calls.append(len(calls))
        return 1.0

    x, y = make_problem()
    _, hist = train_ensemble(
        16,
        config=config(ensemble_size=1, patience=100, max_epochs=7),
        seed=1,
        **_callbacks(x, y, val_loss=counting_val),
        **COMMON,
        check_every=3,
    )
    assert len(calls) == 3  # epochs 3, 6 and the last (7), 1-based
    vals = [h["val"] for h in hist[0]]
    assert [math.isfinite(v) for v in vals] == [False, False, True, False, False, True, True]
    assert all(math.isfinite(h["train"]) for h in hist[0])  # training ran on every epoch


def test_patience_counts_checks_not_epochs():
    """check_every=2 and patience=2: the flat loss stops after the third check
    (epoch 6, 1-based) rather than after three epochs."""
    _, hist = _flat_run(config(ensemble_size=1, patience=2, max_epochs=20), check_every=2)
    assert len(hist[0]) == 6


def test_best_weights_are_restored_from_a_validated_epoch_only():
    """With check_every=2 the restored state is the best CHECKED epoch's, even if an
    unchecked epoch would have scored better."""
    x, y = make_problem()
    _, _, real_val = loops(x, y)
    seen = []

    def spy_val(model):
        seen.append({k: v.clone() for k, v in model.state_dict().items()})
        return real_val(model)

    models, hist = train_ensemble(
        16,
        config=config(ensemble_size=1, patience=100, max_epochs=6),
        seed=2,
        **_callbacks(x, y, val_loss=spy_val),
        **COMMON,
        check_every=2,
    )
    checked = [h["val"] for h in hist[0] if math.isfinite(h["val"])]
    assert len(checked) == 3 < len(hist[0])
    best = int(np.argmin(checked))
    for k, v in models[0].state_dict().items():
        assert torch.equal(v, seen[best][k]), k


def test_min_epochs_and_check_every_refusals():
    with pytest.raises(ValueError, match="min_epochs"):
        _flat_run(config(ensemble_size=1, max_epochs=3), min_epochs=4)
    with pytest.raises(ValueError, match="check_every"):
        _flat_run(config(ensemble_size=1), check_every=0)


def _recorded_run(cfg, **extra):
    """Run one member and hand back every batch's cutoffs, in order."""
    x, y = make_problem(n=16)
    record: list = []
    make_model, train_loss, val_loss = loops(x, y, record=record)
    train_ensemble(
        16,
        config=cfg,
        seed=7,
        make_model=make_model,
        train_loss=train_loss,
        val_loss=val_loss,
        **COMMON,
        **extra,
    )
    return record


def test_per_epoch_cutoffs_are_shared_across_an_epochs_batches():
    record = _recorded_run(
        config(ensemble_size=1, batch_size=4, max_epochs=12, patience=100),
        cutoff_sampling="per_epoch",
    )
    per_epoch = [record[i : i + 4] for i in range(0, len(record), 4)]  # 4 batches per epoch
    assert len(per_epoch) == 12
    for batches in per_epoch:
        values = np.concatenate(batches)
        assert values.min() == values.max()  # one cutoff for the whole epoch
        assert 2 <= values[0] < 5
    assert len({int(b[0][0]) for b in per_epoch}) > 1  # and it varies across epochs


def test_per_example_cutoffs_vary_within_a_batch_by_default():
    record = _recorded_run(config(ensemble_size=1, batch_size=8, max_epochs=6, patience=100))
    assert any(len(np.unique(b)) > 1 for b in record)


def test_cutoff_sampling_refusal():
    with pytest.raises(ValueError, match="cutoff_sampling"):
        _run(cutoff_sampling="per_batch")


def _best_val(history):
    finite = [h["val"] for h in history if math.isfinite(h["val"])]
    return min(finite) if finite else math.inf


def test_keep_selects_the_lowest_validation_members_in_member_order():
    cfg = dict(ensemble_size=4, max_epochs=5, patience=10)
    all_models, all_hist = _run(cfg=config(**cfg), seed=11)
    kept_models, kept_hist = _run(cfg=config(**cfg), seed=11, keep=2)
    scores = [_best_val(h) for h in all_hist]
    want = sorted(sorted(range(4), key=lambda m: (scores[m], m))[:2])
    assert [h[0]["member"] for h in kept_hist] == want
    assert [h[0]["member"] for h in all_hist] == [0, 1, 2, 3]
    for kept, m in zip(kept_models, want, strict=True):
        for k, v in kept.state_dict().items():
            assert torch.equal(v, all_models[m].state_dict()[k]), (m, k)


def _scored_by_member(scores, **extra):
    """Hand member m the validation score scores[m], so which members win is a
    property of the test rather than of the data. Every member runs the same
    number of epochs, so the call count says which member is validating."""
    x, y = make_problem()
    calls = [0]
    epochs = 2

    def scripted_val(model):
        member = calls[0] // epochs
        calls[0] += 1
        return scores[member]

    return train_ensemble(
        16,
        config=config(ensemble_size=len(scores), max_epochs=epochs, patience=100),
        seed=5,
        **_callbacks(x, y, val_loss=scripted_val),
        **COMMON,
        **extra,
    )


def test_keep_returns_the_kept_members_in_member_order_not_score_order():
    """Members 3 and 1 validate best, in that order; they come back as 1 then 3."""
    scores = [0.9, 0.2, 0.8, 0.1]
    _, all_hist = _scored_by_member(scores)
    assert [_best_val(h) for h in all_hist] == scores
    kept_models, kept_hist = _scored_by_member(scores, keep=2)
    assert [h[0]["member"] for h in kept_hist] == [1, 3]
    assert len(kept_models) == 2


def test_keep_breaks_ties_on_the_lower_member_index():
    _, kept_hist = _scored_by_member([0.5, 0.1, 0.1, 0.5], keep=1)
    assert [h[0]["member"] for h in kept_hist] == [1]


def test_keep_equal_to_ensemble_size_keeps_everything():
    _assert_same_fit(_run(keep=2), _run())


def test_keep_refusals():
    with pytest.raises(ValueError, match="keep"):
        _run(keep=3)  # ensemble_size is 2
    with pytest.raises(ValueError, match="keep"):
        _run(keep=0)
