# Training scheme and residual calibration (PR B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the shared NN training loop (`gallery/nn/_training.py::train_ensemble`) with a learning-rate schedule, parameter groups, minimum epochs, check-every, per-epoch cutoff sampling and best-k member selection, every default bit-identical to today; and add `kernels/residual_calibration.py`, the size-stratified rolling-origin residual calibration, generic over any point forecaster.

**Architecture:** `train_ensemble` keeps its signature and gains keyword-only defaulted arguments; a verbatim copy of the current loop lives in the test file as the reference the defaults are compared against. `warmup_cosine` lives beside it in `_training.py`. The calibration module is torch-free numpy and pandas: rolling residuals, stratified centred pools, resampled draws, and a leave-one-out coverage table.

**Tech Stack:** Python 3.11/3.12, numpy, pandas, torch (tests for the loop only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md` (sections 5.2 and 5.3). Read it first. Two deliberate departures from it, both explained in the tasks: `warmup_cosine` stays in `gallery/nn/_training.py` and is NOT exported from `ibnr.kernels` (kernels never imports the gallery); and there is no `n_workers` argument, because `make_model`/`train_loss`/`val_loss` are closures over tensors and cannot cross a spawned process.

## Global Constraints

- `kernels/` never imports `ibnr.gallery` (subprocess-tested). `ibnr.gallery` must import without torch (`tests/test_gallery.py::test_gallery_import_does_not_require_torch`). `_training.py` imports torch INSIDE `train_ensemble` only, as today.
- Every default of `train_ensemble` reproduces today's loop bit for bit. The reference copy in the test file is taken from `origin/main` BEFORE any edit to `_training.py`.
- Lint: `uv run ruff check . && uv run ruff format .` and `uv run python scripts/lint_md_snippets.py` must pass.
- Commit messages: conventional prefix, subject plus body, NO `Co-Authored-By` or "generated with" lines, in commits or the PR body.
- Prose rules for docstrings, comments and CHANGELOG: plain sentences. Never use the words "screen", "panel", "gate", "fingerprint", "membership", "ablation" (or any form), "seed noise", "chip", "drain". "seed" only in its random-number sense.
- A test for a refusal uses `pytest.raises(..., match=...)`.
- Every claimed identity or refusal in a new test file is checked by breaking it once on purpose (a mutation check) and watching the test go red; the PR body lists the mutations tried.
- Work on branch `feat/training-scheme-calibration` in worktree `C:\Users\EthanKang\Projects\ibnr-wt\training-scheme`, created from `origin/main`. Run `uv sync --extra nn` there (torch is multi-GB; that is expected). Before committing and again before opening the PR run `git log --oneline HEAD..origin/main`; if non-empty, rebase and rerun the tests.
- After editing `_training.py`, run the touched tests with torch installed and confirm they EXECUTED (`-rs` prints skip reasons; a module-level `importorskip` collapses a whole file into one skip).

---

### Task 1: the reference loop and the bit-identity test

**Files:**
- Test: `tests/test_nn_training_scheme.py` (create)

**Interfaces:**
- Produces (test-only): `_reference_train_ensemble`, a verbatim copy of `train_ensemble` from `origin/main`; `Tiny`, `make_problem`, `loops`, `CFG` fixtures the later tasks reuse.

- [ ] **Step 1: Copy the reference loop BEFORE touching `_training.py`**

Run: `git show origin/main:src/ibnr/gallery/nn/_training.py` and paste the body of `train_ensemble` into the test file as `_reference_train_ensemble`, unchanged except for the name. Then write the file:

```python
"""gallery.nn._training: the shared training loop's scheme extensions.

The first test is the one every other test in this file rests on: with every
new argument left at its default, the extended loop must produce the SAME
weights and the SAME history as the loop on ``origin/main`` before this change,
bit for bit, on this machine, in this process. The reference is a verbatim copy
of that loop kept here rather than a stored hash, because a hash of float32
weights is a property of the platform and its BLAS, and this test has to pass
on the Windows dev box and on the Linux CI jobs alike.

The remaining tests check each extension by its EFFECT on the trained weights
or on what reached the loss callback, never by inspecting a signature: a knob
that is accepted and never read is the repo's named inert-parameter bug class.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.nn import _training  # noqa: E402
from ibnr.gallery.nn._training import train_ensemble  # noqa: E402


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
    ref = _reference_train_ensemble(16, config=config(), seed=3, **dict(zip(
        ("make_model", "train_loss", "val_loss"), loops(x, y), strict=True)), **COMMON)
    new = train_ensemble(16, config=config(), seed=3, **dict(zip(
        ("make_model", "train_loss", "val_loss"), loops(x, y), strict=True)), **COMMON)
    _assert_same_fit(ref, new)


def test_defaults_reproduce_the_reference_loop_when_early_stopping_fires():
    """A validation loss that never improves stops after `patience` epochs; the
    break point must land on the same epoch in both loops."""
    x, y = make_problem()
    make_model, train_loss, _ = loops(x, y)

    def flat_val(model):
        return 1.0

    kw = dict(make_model=make_model, train_loss=train_loss, val_loss=flat_val, **COMMON)
    ref = _reference_train_ensemble(16, config=config(patience=2, max_epochs=20), seed=1, **kw)
    new = train_ensemble(16, config=config(patience=2, max_epochs=20), seed=1, **kw)
    _assert_same_fit(ref, new)
    assert len(ref[1][0]) == 3  # epoch 0 improves from inf; epochs 1 and 2 exhaust patience 2
```

- [ ] **Step 2: Run the tests to verify they pass against the UNCHANGED loop**

Run: `uv run pytest tests/test_nn_training_scheme.py -q -rs`
Expected: 2 PASS (both loops are the same code today; this establishes the reference is faithful). If either fails, the copy is not verbatim: fix the copy, not the loop.

- [ ] **Step 3: Commit**

```bash
git add tests/test_nn_training_scheme.py
git commit -m "test: pin the shared NN training loop against a verbatim reference copy

Every scheme extension that follows must leave the defaults bit-identical
to this copy of train_ensemble from origin/main."
```

---

### Task 2: `warmup_cosine`

**Files:**
- Modify: `src/ibnr/gallery/nn/_training.py` (add the function and export it in `__all__`)
- Test: `tests/test_nn_training_scheme.py` (append)

**Interfaces:**
- Produces: `warmup_cosine(schedule_epochs: int, warmup: int = 20) -> Callable[[int], float]`, epoch 1-based.

- [ ] **Step 1: Write the failing tests**

```python
from ibnr.gallery.nn._training import warmup_cosine  # noqa: E402  (add to the import block)


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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_nn_training_scheme.py -q -k warmup`
Expected: FAIL with `ImportError: cannot import name 'warmup_cosine'`.

- [ ] **Step 3: Implement**

In `src/ibnr/gallery/nn/_training.py`, after the imports:

```python
__all__ = ["train_ensemble", "warmup_cosine"]


def warmup_cosine(schedule_epochs: int, warmup: int = 20):
    """Linear warmup then cosine decay to zero, as a multiplier on the base rate.

    The R study's ``learning_rate_multiplier``, ported with its edge cases: a
    one-epoch schedule still performs one update (multiplier 1 at epoch 1, 0
    after); the warmup is capped at ``schedule_epochs - 1`` so at least one
    post-warmup epoch exists; progress past the schedule end is capped at 1, so
    the multiplier stays at zero rather than rising again. Epochs are 1-based,
    which is how :func:`train_ensemble` calls it.
    """
    if not isinstance(schedule_epochs, int) or schedule_epochs < 1:
        raise ValueError(f"schedule_epochs must be an int >= 1, got {schedule_epochs!r}")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be an int >= 0, got {warmup!r}")

    def multiplier(epoch: int) -> float:
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError(f"epoch is 1-based and must be >= 1, got {epoch!r}")
        if schedule_epochs == 1:
            return float(epoch == 1)
        w = min(warmup, schedule_epochs - 1)
        if w > 0 and epoch <= w:
            return epoch / w
        progress = min((epoch - w) / (schedule_epochs - w), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return multiplier
```

- [ ] **Step 4: Run to verify they pass, then commit**

Run: `uv run pytest tests/test_nn_training_scheme.py -q`

```bash
git add src/ibnr/gallery/nn/_training.py tests/test_nn_training_scheme.py
git commit -m "feat: warmup_cosine learning-rate multiplier for the NN training loop"
```

---

### Task 3: `schedule` and `param_groups`

**Files:**
- Modify: `src/ibnr/gallery/nn/_training.py::train_ensemble`
- Test: `tests/test_nn_training_scheme.py` (append)

**Interfaces:**
- `train_ensemble(..., schedule: Callable[[int], float] | None = None, param_groups: Callable[[Any], list[dict]] | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
def _run(cfg=None, seed=3, **extra):
    x, y = make_problem()
    make_model, train_loss, val_loss = loops(x, y)
    return train_ensemble(
        16,
        config=cfg or config(),
        seed=seed,
        make_model=make_model,
        train_loss=train_loss,
        val_loss=val_loss,
        **COMMON,
        **extra,
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


def test_param_groups_give_each_group_its_own_rate():
    """A group with lr 0 stays put while the other trains."""
    torch.manual_seed(3)
    init = _states([Tiny()])[0]

    def groups(model):
        return [
            {"params": [model.lin.weight], "lr": 0.0},
            {"params": [model.lin.bias]},  # inherits config.lr
        ]

    models, _ = _run(cfg=config(ensemble_size=1), param_groups=groups)
    state = models[0].state_dict()
    assert torch.equal(state["lin.weight"], init["lin.weight"])
    assert not torch.equal(state["lin.bias"], init["lin.bias"])


def test_param_groups_and_schedule_compose():
    """The schedule scales each group's OWN base rate: a group at lr 0 stays at 0
    under any multiplier, and the other group still moves at multiplier 1."""
    torch.manual_seed(3)
    init = _states([Tiny()])[0]

    def groups(model):
        return [{"params": [model.lin.weight], "lr": 0.0}, {"params": [model.lin.bias]}]

    models, _ = _run(cfg=config(ensemble_size=1), param_groups=groups, schedule=lambda e: 0.5)
    state = models[0].state_dict()
    assert torch.equal(state["lin.weight"], init["lin.weight"])
    assert not torch.equal(state["lin.bias"], init["lin.bias"])


def test_schedule_refusals():
    with pytest.raises(ValueError, match="schedule"):
        _run(schedule=lambda epoch: -0.5)
    with pytest.raises(ValueError, match="schedule"):
        _run(schedule=lambda epoch: float("nan"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_nn_training_scheme.py -q -k "schedule or param_groups"`
Expected: FAIL with `TypeError: train_ensemble() got an unexpected keyword argument`.

- [ ] **Step 3: Implement**

Change the signature and the two touched places inside the member loop:

```python
def train_ensemble(
    n_cohorts: int,
    *,
    config: Any,
    seed: int | None,
    make_model: Callable[[], Any],
    train_loss: Callable[[Any, Any, Any], Any],
    val_loss: Callable[[Any], float],
    min_cutoff: int,
    val_cutoff: int,
    device: Any,
    show_progress: bool = False,
    schedule: Callable[[int], float] | None = None,
    param_groups: Callable[[Any], list[dict]] | None = None,
) -> tuple[list, list[list[dict]]]:
```

Optimizer construction:

```python
        params = param_groups(model) if param_groups is not None else model.parameters()
        opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]  # what the schedule multiplies
```

At the top of each epoch, before `model.train()`:

```python
            if schedule is not None:
                mult = schedule(epoch + 1)  # 1-based, as the R study counts epochs
                if not (isinstance(mult, (int, float)) and math.isfinite(mult) and mult >= 0):
                    raise ValueError(
                        f"schedule({epoch + 1}) returned {mult!r}; a learning-rate multiplier "
                        "must be a finite number >= 0"
                    )
                for group in opt.param_groups:
                    group["lr"] = group["base_lr"] * mult
```

Nothing else changes, so `schedule=None, param_groups=None` is the reference loop exactly (`group["base_lr"]` is an extra key AdamW ignores).

- [ ] **Step 4: Run the whole file, then commit**

Run: `uv run pytest tests/test_nn_training_scheme.py -q -rs`
Expected: all PASS, including the two bit-identity tests.

```bash
git add src/ibnr/gallery/nn/_training.py tests/test_nn_training_scheme.py
git commit -m "feat: train_ensemble takes a learning-rate schedule and parameter groups

Both defaulted so the loop stays bit-identical; the schedule multiplies
each group's own base rate, which is how the R study trains phi at ten
times the network rate under one cosine decay."
```

---

### Task 4: `min_epochs`, `check_every`, patience in checks

**Files:**
- Modify: `src/ibnr/gallery/nn/_training.py::train_ensemble`
- Test: `tests/test_nn_training_scheme.py` (append)

**Interfaces:**
- `train_ensemble(..., min_epochs: int = 0, check_every: int = 1)`. `config.patience` counts validation checks. History records for un-validated epochs carry `val = nan`.

- [ ] **Step 1: Write the failing tests**

```python
def _flat_run(cfg, **extra):
    x, y = make_problem()
    make_model, train_loss, _ = loops(x, y)
    return train_ensemble(
        16,
        config=cfg,
        seed=1,
        make_model=make_model,
        train_loss=train_loss,
        val_loss=lambda model: 1.0,
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
    make_model, train_loss, _ = loops(x, y)
    _, hist = train_ensemble(
        16,
        config=config(ensemble_size=1, patience=100, max_epochs=7),
        seed=1,
        make_model=make_model,
        train_loss=train_loss,
        val_loss=counting_val,
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
    make_model, train_loss, val_loss = loops(x, y)
    seen = []

    def spy_val(model):
        seen.append({k: v.clone() for k, v in model.state_dict().items()})
        return val_loss(model)

    models, hist = train_ensemble(
        16,
        config=config(ensemble_size=1, patience=100, max_epochs=6),
        seed=2,
        make_model=make_model,
        train_loss=train_loss,
        val_loss=spy_val,
        **COMMON,
        check_every=2,
    )
    checked = [h["val"] for h in hist[0] if math.isfinite(h["val"])]
    best = int(np.argmin(checked))
    for k, v in models[0].state_dict().items():
        assert torch.equal(v, seen[best][k]), k


def test_min_epochs_and_check_every_refusals():
    with pytest.raises(ValueError, match="min_epochs"):
        _flat_run(config(ensemble_size=1, max_epochs=3), min_epochs=4)
    with pytest.raises(ValueError, match="check_every"):
        _flat_run(config(ensemble_size=1), check_every=0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_nn_training_scheme.py -q -k "min_epochs or check_every or counts_checks or validated_epoch"`

- [ ] **Step 3: Implement**

Add the arguments (`min_epochs: int = 0`, `check_every: int = 1`) and validate them once at the top:

```python
    if not isinstance(check_every, int) or check_every < 1:
        raise ValueError(f"check_every must be an int >= 1, got {check_every!r}")
    if not isinstance(min_epochs, int) or min_epochs < 0 or min_epochs > config.max_epochs:
        raise ValueError(
            f"min_epochs must be an int in [0, max_epochs={config.max_epochs}], got {min_epochs!r}"
        )
```

Replace the validation block at the end of each epoch with:

```python
            is_last = epoch + 1 == config.max_epochs
            if (epoch + 1) % check_every != 0 and not is_last:
                history.append({"epoch": epoch, "train": epoch_loss / max(n_batches, 1), "val": math.nan})
                continue
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
                if patience_left <= 0 and epoch + 1 >= min_epochs:
                    break
```

At `check_every=1, min_epochs=0` every epoch validates and the break condition is the old one, so the reference tests must still pass.

- [ ] **Step 4: Run the whole file, then commit**

Run: `uv run pytest tests/test_nn_training_scheme.py -q -rs`

```bash
git add src/ibnr/gallery/nn/_training.py tests/test_nn_training_scheme.py
git commit -m "feat: train_ensemble validates every check_every epochs and honours min_epochs

Patience now counts validation checks, which at check_every=1 is the old
count in epochs. Un-validated epochs record a NaN val so
kernels.tuning.validation_score, which skips NaN, reads them as before."
```

---

### Task 5: per-epoch cutoff sampling

**Files:**
- Modify: `src/ibnr/gallery/nn/_training.py::train_ensemble`
- Test: `tests/test_nn_training_scheme.py` (append)

**Interfaces:**
- `train_ensemble(..., cutoff_sampling: str = "per_example")`, the other value `"per_epoch"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_per_epoch_cutoffs_are_shared_across_an_epochs_batches():
    x, y = make_problem(n=16)
    record = []
    make_model, train_loss, val_loss = loops(x, y, record=record)
    train_ensemble(
        16,
        config=config(ensemble_size=1, batch_size=4, max_epochs=12, patience=100),
        seed=7,
        make_model=make_model,
        train_loss=train_loss,
        val_loss=val_loss,
        **COMMON,
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
    x, y = make_problem(n=16)
    record = []
    make_model, train_loss, val_loss = loops(x, y, record=record)
    train_ensemble(
        16,
        config=config(ensemble_size=1, batch_size=8, max_epochs=6, patience=100),
        seed=7,
        make_model=make_model,
        train_loss=train_loss,
        val_loss=val_loss,
        **COMMON,
    )
    assert any(len(np.unique(b)) > 1 for b in record)


def test_cutoff_sampling_refusal():
    with pytest.raises(ValueError, match="cutoff_sampling"):
        _run(cutoff_sampling="per_batch")
```

- [ ] **Step 2: Run to verify they fail**, then **Step 3: Implement**

```python
CUTOFF_SAMPLING: tuple[str, ...] = ("per_example", "per_epoch")
```

Validate at the top (`if cutoff_sampling not in CUTOFF_SAMPLING: raise ValueError(...)`). In the epoch loop, before the batch loop:

```python
            epoch_cutoff = (
                int(rng.integers(min_cutoff, val_cutoff)) if cutoff_sampling == "per_epoch" else None
            )
```

and in the batch loop:

```python
                if epoch_cutoff is None:
                    cutoffs = torch.tensor(
                        rng.integers(min_cutoff, val_cutoff, size=len(idx)), device=device
                    )
                else:
                    cutoffs = torch.full((len(idx),), epoch_cutoff, dtype=torch.long, device=device)
```

The per-example branch is the reference's exact call, in the same place in the stream.

- [ ] **Step 4: Run the whole file, then commit**

```bash
git add src/ibnr/gallery/nn/_training.py tests/test_nn_training_scheme.py
git commit -m "feat: train_ensemble can draw one augmented cutoff per epoch

The R study builds one example set per epoch; per_epoch reproduces that.
per_example stays the default and the reference's exact random stream."
```

---

### Task 6: `keep`, best-k member selection

**Files:**
- Modify: `src/ibnr/gallery/nn/_training.py::train_ensemble`
- Test: `tests/test_nn_training_scheme.py` (append)

**Interfaces:**
- `train_ensemble(..., keep: int | None = None)`. Returned `models`/`histories` hold only the kept members, in member order; every history record gains `"member": <original index>`.

- [ ] **Step 1: Write the failing tests**

```python
def _best_val(history):
    finite = [h["val"] for h in history if math.isfinite(h["val"])]
    return min(finite) if finite else math.inf


def test_keep_selects_the_lowest_validation_members_in_member_order():
    all_models, all_hist = _run(cfg=config(ensemble_size=4, max_epochs=5, patience=10), seed=11)
    kept_models, kept_hist = _run(
        cfg=config(ensemble_size=4, max_epochs=5, patience=10), seed=11, keep=2
    )
    scores = [_best_val(h) for h in all_hist]
    want = sorted(sorted(range(4), key=lambda m: (scores[m], m))[:2])
    assert [h[0]["member"] for h in kept_hist] == want
    assert [h[0]["member"] for h in all_hist] == [0, 1, 2, 3]
    for kept, m in zip(kept_models, want, strict=True):
        for k, v in kept.state_dict().items():
            assert torch.equal(v, all_models[m].state_dict()[k]), (m, k)


def test_keep_equal_to_ensemble_size_keeps_everything():
    _assert_same_fit(_run(keep=2), _run())


def test_keep_refusals():
    with pytest.raises(ValueError, match="keep"):
        _run(keep=3)  # ensemble_size is 2
    with pytest.raises(ValueError, match="keep"):
        _run(keep=0)
```

`_assert_same_fit` compares `(epoch, train, val)` tuples, so the added `member` key does not disturb the reference tests.

- [ ] **Step 2: Run to verify they fail**, then **Step 3: Implement**

Validate at the top:

```python
    if keep is not None and (not isinstance(keep, int) or not 1 <= keep <= config.ensemble_size):
        raise ValueError(
            f"keep must be an int in [1, ensemble_size={config.ensemble_size}] or None, got {keep!r}"
        )
```

Add `"member": member` to both history record dicts. After the member loop:

```python
    if keep is not None and keep < len(models):
        def best(history):
            finite = [h["val"] for h in history if math.isfinite(h["val"])]
            return min(finite) if finite else math.inf

        order = sorted(range(len(models)), key=lambda m: (best(histories[m]), m))
        chosen = sorted(order[:keep])
        models = [models[m] for m in chosen]
        histories = [histories[m] for m in chosen]
    return models, histories
```

- [ ] **Step 4: Run the whole file, run the entries' own tests, commit**

Run: `uv run pytest tests/test_nn_training_scheme.py tests/test_nn_transformer.py tests/test_nn_training_context.py tests/test_tuning.py -q -rs`
Expected: all executed and PASS. `test_nn_training_context.py` wraps `train_ensemble`; if its spy asserts the exact history keys, extend the expected set with `member`.

```bash
git add src/ibnr/gallery/nn/_training.py tests/test_nn_training_scheme.py
git commit -m "feat: train_ensemble keeps the best k members by validation score

The R study trains ten seeds and keeps the two with the lowest validation
error; keep=None returns every member as before. History records now say
which member they belong to."
```

Update the module docstring of `_training.py` to list the six extensions in one paragraph each, and state the two departures from the spec (no `n_workers`: the callbacks are closures over tensors and cannot cross a spawned process; `warmup_cosine` lives here because kernels never imports the gallery). Commit as `docs: describe the training scheme extensions`.

---

### Task 7: `kernels/residual_calibration.py`

**Files:**
- Create: `src/ibnr/kernels/residual_calibration.py`
- Test: `tests/test_residual_calibration.py` (core, no torch)

**Interfaces:**
- `rolling_residuals(forecast_at, actual_at, *, cutoffs, n_periods, size, scale_floor=(0.01, 1.0)) -> pd.DataFrame` with columns `unit`, `cutoff`, `horizon`, `predicted`, `actual`, `size`, `residual_scale`, `standardised_error`, and `frame.attrs["n_dropped_nonfinite"]`.
- `ntile(values, n) -> np.ndarray` (1-based stratum per element, dplyr's `ntile` rule).
- `calibrate(residuals, *, horizons, n_strata=4, min_per_stratum=40) -> Calibration`.
- `Calibration` (frozen dataclass): `horizons`, `n_strata`, `unit_stratum` (dict unit -> stratum), `size_edges` (array of upper size bounds per stratum, last is +inf), `medians` (per stratum), `pools` (tuple of centred error arrays per stratum), `residuals` (the rows used), with method `strata_for(size) -> np.ndarray`.
- `calibrated_draws(calibration, *, point, size, n_draws, rng, scale_floor=(0.01, 1.0)) -> np.ndarray` of shape `(n_draws, n_units)`.
- `leave_one_out_coverage(calibration, *, levels=(0.8, 0.95)) -> pd.DataFrame` with columns `horizon`, `nominal_coverage`, `empirical_coverage`, `forecasts`.

- [ ] **Step 1: Write the failing tests**

```python
"""kernels.residual_calibration: size-stratified rolling-origin residual pools.

The R study's uncertainty for a point forecaster, written once: apply the fixed
forecaster at earlier cutoffs, standardise the errors by a floored scale, pool
them by unit size, centre each pool on its median, and resample around the final
point. A synthetic forecaster with a KNOWN error law is the arbiter here - the
draws must recover its quantiles - and every refusal is named.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.residual_calibration import (
    Calibration,
    calibrate,
    calibrated_draws,
    leave_one_out_coverage,
    ntile,
    rolling_residuals,
)

N_UNITS = 200
SIZES = np.linspace(100.0, 20_000.0, N_UNITS)  # premium-like, strictly increasing
CUTOFFS = (5, 6, 7, 8, 9)
N_PERIODS = 10


def _synthetic(seed=0, spread=0.2):
    """Forecast = 3 x size; actual = forecast x (1 + u), u uniform on [-spread, spread]
    drawn per (cutoff, unit). The error law is known, so the pools must recover it."""
    rng = np.random.default_rng(seed)
    u = {k: rng.uniform(-spread, spread, size=N_UNITS) for k in CUTOFFS}

    def forecast_at(k):
        return 3.0 * SIZES

    def actual_at(k):
        return 3.0 * SIZES * (1.0 + u[k])

    return forecast_at, actual_at, u


def test_ntile_matches_dplyr():
    """dplyr >= 1.0: bins as equal as possible, the LARGER bins first. Five values
    in two bins are 3 + 2; ten in four are 3 + 3 + 2 + 2."""
    assert ntile(np.array([5.0, 1.0, 3.0, 2.0, 4.0]), 2).tolist() == [2, 1, 1, 1, 2]
    assert ntile(np.arange(10.0), 4).tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    assert ntile(np.array([2.0, 2.0, 1.0]), 3).tolist() == [2, 3, 1]  # ties broken by position
    assert ntile(np.arange(93.0), 4).tolist().count(1) == 24  # the R study's 93 companies: 24, 23, 23, 23


def test_rolling_residuals_columns_horizon_and_scale():
    forecast_at, actual_at, u = _synthetic()
    rows = rolling_residuals(
        forecast_at, actual_at, cutoffs=CUTOFFS, n_periods=N_PERIODS, size=SIZES
    )
    assert list(rows.columns) == [
        "unit", "cutoff", "horizon", "predicted", "actual", "size", "residual_scale",
        "standardised_error",
    ]
    assert len(rows) == N_UNITS * len(CUTOFFS)
    assert rows.attrs["n_dropped_nonfinite"] == 0
    assert set(rows["horizon"]) == {5, 4, 3, 2, 1}
    first = rows[(rows["cutoff"] == 5) & (rows["unit"] == 0)].iloc[0]
    assert first["horizon"] == 5
    # scale = max(|predicted|, 0.01 * size, 1): here |predicted| = 300 dominates
    assert first["residual_scale"] == pytest.approx(300.0)
    assert first["standardised_error"] == pytest.approx(u[5][0])


def test_rolling_residuals_floors_the_scale_and_drops_nonfinite_rows():
    size = np.array([50.0, 1e6])

    def forecast_at(k):
        return np.array([0.0, np.nan])

    def actual_at(k):
        return np.array([2.0, 1.0])

    rows = rolling_residuals(forecast_at, actual_at, cutoffs=(1,), n_periods=2, size=size)
    assert len(rows) == 1
    assert rows.attrs["n_dropped_nonfinite"] == 1
    assert rows["residual_scale"].iloc[0] == pytest.approx(1.0)  # max(0, 0.5, 1)


def test_rolling_residuals_refuses_a_forecaster_of_the_wrong_length():
    def bad(k):
        return np.ones(3)

    with pytest.raises(ValueError, match="length"):
        rolling_residuals(bad, bad, cutoffs=(1,), n_periods=2, size=SIZES)


def test_calibrate_strata_are_size_quartiles_over_units_and_pools_are_centred():
    forecast_at, actual_at, _ = _synthetic()
    rows = rolling_residuals(forecast_at, actual_at, cutoffs=CUTOFFS, n_periods=N_PERIODS, size=SIZES)
    cal = calibrate(rows, horizons=(3, 4, 5))
    assert isinstance(cal, Calibration)
    assert cal.horizons == (3, 4, 5)
    strata = np.array([cal.unit_stratum[u] for u in range(N_UNITS)])
    assert strata.tolist() == ntile(SIZES, 4).tolist()
    assert all(len(p) == 50 * 3 for p in cal.pools)  # 50 units x 3 horizons per stratum
    for p in cal.pools:
        assert np.median(p) == pytest.approx(0.0, abs=1e-12)
    assert len(cal.residuals) == N_UNITS * 3
    np.testing.assert_array_equal(cal.strata_for(SIZES), strata)
    assert cal.strata_for(np.array([0.0, 1e9])).tolist() == [1, 4]


def test_calibrate_refuses_a_thin_stratum_and_unknown_horizons():
    forecast_at, actual_at, _ = _synthetic()
    rows = rolling_residuals(forecast_at, actual_at, cutoffs=CUTOFFS, n_periods=N_PERIODS, size=SIZES)
    with pytest.raises(ValueError, match="stratum"):
        calibrate(rows, horizons=(3,), n_strata=4, min_per_stratum=51)
    with pytest.raises(ValueError, match="horizon"):
        calibrate(rows, horizons=(3, 9))


def test_calibrated_draws_recover_the_known_error_law():
    forecast_at, actual_at, _ = _synthetic(spread=0.2)
    rows = rolling_residuals(forecast_at, actual_at, cutoffs=CUTOFFS, n_periods=N_PERIODS, size=SIZES)
    cal = calibrate(rows, horizons=(3, 4, 5))
    point = 3.0 * SIZES
    draws = calibrated_draws(cal, point=point, size=SIZES, n_draws=20_000, rng=np.random.default_rng(1))
    assert draws.shape == (20_000, N_UNITS)
    # u is uniform on [-0.2, 0.2] with median 0, so the 10th and 90th percentiles of
    # draws / point - 1 sit near -0.16 and +0.16
    rel = draws / point[None, :] - 1.0
    assert np.quantile(rel, 0.10) == pytest.approx(-0.16, abs=0.02)
    assert np.quantile(rel, 0.90) == pytest.approx(0.16, abs=0.02)
    again = calibrated_draws(cal, point=point, size=SIZES, n_draws=20_000, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(draws, again)


def test_calibrated_draws_use_the_units_own_stratum():
    """Two strata with disjoint error pools: a unit's draws come from ITS pool."""
    size = np.array([1.0, 2.0, 3.0, 4.0] * 25)  # 100 units, two strata of 50 (ntile on ties by position)
    errors = np.where(size <= 2.0, -0.5, +0.5)  # small units always under, large always over

    def forecast_at(k):
        return np.full(100, 100.0)

    def actual_at(k):
        return 100.0 * (1.0 + errors)

    rows = rolling_residuals(forecast_at, actual_at, cutoffs=(1, 2), n_periods=3, size=size)
    cal = calibrate(rows, horizons=(1, 2), n_strata=2, min_per_stratum=10)
    draws = calibrated_draws(cal, point=np.full(100, 100.0), size=size, n_draws=50, rng=np.random.default_rng(0))
    # pools are centred, so each stratum's draws sit exactly at the point
    np.testing.assert_allclose(draws, 100.0)
    # and an un-centred read shows which pool each unit drew from
    assert cal.medians.tolist() == [-0.5, 0.5]


def test_calibrated_draws_refusals():
    forecast_at, actual_at, _ = _synthetic()
    rows = rolling_residuals(forecast_at, actual_at, cutoffs=CUTOFFS, n_periods=N_PERIODS, size=SIZES)
    cal = calibrate(rows, horizons=(3, 4, 5))
    with pytest.raises(ValueError, match="same length"):
        calibrated_draws(cal, point=np.ones(3), size=SIZES, n_draws=10, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="n_draws"):
        calibrated_draws(cal, point=3.0 * SIZES, size=SIZES, n_draws=0, rng=np.random.default_rng(0))


def test_leave_one_out_coverage_is_near_nominal_for_a_uniform_error_law():
    forecast_at, actual_at, _ = _synthetic(seed=3, spread=0.2)
    rows = rolling_residuals(forecast_at, actual_at, cutoffs=CUTOFFS, n_periods=N_PERIODS, size=SIZES)
    cal = calibrate(rows, horizons=(3, 4, 5))
    table = leave_one_out_coverage(cal, levels=(0.8, 0.95))
    assert list(table.columns) == ["horizon", "nominal_coverage", "empirical_coverage", "forecasts"]
    assert set(table["horizon"]) == {3, 4, 5}
    assert (table["forecasts"] == N_UNITS).all()
    for level in (0.8, 0.95):
        sub = table[table["nominal_coverage"] == level]
        # 200 forecasts per horizon: one standard deviation of the coverage is
        # about 0.03 at 0.8, so 0.08 is a little under three
        assert sub["empirical_coverage"].between(level - 0.08, level + 0.08).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_residual_calibration.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""Size-stratified rolling-origin residual calibration for a point forecaster.

The R study's uncertainty for its point model, written once and generic: any
callable that forecasts a fixed set of units at a cutoff can be calibrated.

1. :func:`rolling_residuals` applies the forecaster at each earlier cutoff and
   standardises ``actual - predicted`` by ``max(|predicted|, 0.01 * size, 1)``.
2. :func:`calibrate` keeps the named horizons, assigns every unit a size
   stratum (dplyr's ``ntile`` over the units), and centres each stratum's
   errors on its median.
3. :func:`calibrated_draws` resamples a unit's stratum pool around its final
   point, on the same floored scale.
4. :func:`leave_one_out_coverage` is the R study's cross-unit historical
   coverage table: each calibration forecast is covered by an interval built
   from the OTHER units of its stratum.

What this is not: a native predictive distribution. Whoever consumes the draws
must say they are historically calibrated. It is also blind to what the
forecaster trained on: the R study calibrates on cutoffs whose targets overlap
the network's training targets, and that choice belongs to the caller, who
picks ``cutoffs`` and ``horizons``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Calibration",
    "calibrate",
    "calibrated_draws",
    "leave_one_out_coverage",
    "ntile",
    "rolling_residuals",
]


def ntile(values, n: int) -> np.ndarray:
    """dplyr's ``ntile`` (1.0 and later): rank by value with ties broken by
    position, then bins as equal in size as possible with the LARGER bins first.
    ``len = 93, n = 4`` gives bins of 24, 23, 23, 23, which is how the R study's
    companies fall into premium quartiles. 1-based."""
    v = np.asarray(values, dtype=float).reshape(-1)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    order = np.argsort(v, kind="stable")
    rank = np.empty(v.size, dtype=int)
    rank[order] = np.arange(1, v.size + 1)
    smaller = v.size // n
    n_larger = v.size % n
    larger = smaller + 1
    threshold = larger * n_larger  # ranks up to here fall in the n_larger larger bins
    in_larger = rank <= threshold
    bins = np.where(
        in_larger,
        np.ceil(rank / max(larger, 1)),
        n_larger + np.ceil((rank - threshold) / max(smaller, 1)),
    )
    return bins.astype(int)


def _scale(predicted: np.ndarray, size: np.ndarray, floor: tuple[float, float]) -> np.ndarray:
    return np.maximum.reduce([np.abs(predicted), floor[0] * size, np.full_like(size, floor[1])])


def rolling_residuals(
    forecast_at: Callable[[int], np.ndarray],
    actual_at: Callable[[int], np.ndarray],
    *,
    cutoffs: Sequence[int],
    n_periods: int,
    size,
    scale_floor: tuple[float, float] = (0.01, 1.0),
) -> pd.DataFrame:
    """Standardised forecast errors of a fixed forecaster at earlier cutoffs.

    ``forecast_at(k)`` and ``actual_at(k)`` return one value per unit, in one
    fixed unit order, for cutoff ``k``; ``horizon = n_periods - k``. Rows whose
    standardised error is not finite are dropped and counted in
    ``frame.attrs["n_dropped_nonfinite"]``, as the R study filters them.
    """
    size = np.asarray(size, dtype=float).reshape(-1)
    rows = []
    dropped = 0
    for k in cutoffs:
        predicted = np.asarray(forecast_at(k), dtype=float).reshape(-1)
        actual = np.asarray(actual_at(k), dtype=float).reshape(-1)
        if predicted.size != size.size or actual.size != size.size:
            raise ValueError(
                f"cutoff {k}: forecast_at returned length {predicted.size} and actual_at "
                f"length {actual.size}; both must match size's length {size.size}"
            )
        scale = _scale(predicted, size, scale_floor)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (actual - predicted) / scale
        keep = np.isfinite(z)
        dropped += int((~keep).sum())
        for u in np.nonzero(keep)[0]:
            rows.append(
                {
                    "unit": int(u),
                    "cutoff": int(k),
                    "horizon": int(n_periods - k),
                    "predicted": float(predicted[u]),
                    "actual": float(actual[u]),
                    "size": float(size[u]),
                    "residual_scale": float(scale[u]),
                    "standardised_error": float(z[u]),
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "unit", "cutoff", "horizon", "predicted", "actual", "size", "residual_scale",
            "standardised_error",
        ],
    )
    frame.attrs["n_dropped_nonfinite"] = dropped
    return frame


@dataclass(frozen=True)
class Calibration:
    """Centred residual pools per size stratum; see the module docstring."""

    horizons: tuple[int, ...]
    n_strata: int
    unit_stratum: dict
    size_edges: np.ndarray  # upper size bound of each stratum; the last is +inf
    medians: np.ndarray  # (n_strata,) the median subtracted from each pool
    pools: tuple  # n_strata arrays of centred standardised errors
    residuals: pd.DataFrame  # the rows used, with a `stratum` column

    def strata_for(self, size) -> np.ndarray:
        """1-based stratum per size, by the stored upper bounds."""
        s = np.asarray(size, dtype=float).reshape(-1)
        return np.searchsorted(self.size_edges[:-1], s, side="left") + 1


def calibrate(
    residuals: pd.DataFrame,
    *,
    horizons: Sequence[int],
    n_strata: int = 4,
    min_per_stratum: int = 40,
) -> Calibration:
    horizons = tuple(int(h) for h in horizons)
    have = set(residuals["horizon"].unique())
    missing = [h for h in horizons if h not in have]
    if missing:
        raise ValueError(f"horizon(s) {missing} not in the residuals, which carry {sorted(have)}")
    units = residuals.groupby("unit")["size"].first().sort_index()
    stratum = ntile(units.to_numpy(), n_strata)
    unit_stratum = dict(zip(units.index.tolist(), stratum.tolist(), strict=True))
    sizes = units.to_numpy()
    edges = np.array(
        [sizes[stratum == s].max() if (stratum == s).any() else -np.inf for s in range(1, n_strata + 1)]
    )
    edges[-1] = np.inf
    used = residuals[residuals["horizon"].isin(horizons)].copy()
    used["stratum"] = used["unit"].map(unit_stratum).astype(int)
    medians, pools = [], []
    for s in range(1, n_strata + 1):
        errors = used.loc[used["stratum"] == s, "standardised_error"].to_numpy(dtype=float)
        if errors.size < min_per_stratum:
            raise ValueError(
                f"stratum {s} of {n_strata} has {errors.size} residual(s) over horizons "
                f"{horizons}; fewer than {min_per_stratum} is too few to resample from"
            )
        med = float(np.median(errors))
        medians.append(med)
        pools.append(errors - med)
    return Calibration(
        horizons=horizons,
        n_strata=n_strata,
        unit_stratum=unit_stratum,
        size_edges=edges,
        medians=np.array(medians),
        pools=tuple(pools),
        residuals=used.reset_index(drop=True),
    )


def calibrated_draws(
    calibration: Calibration,
    *,
    point,
    size,
    n_draws: int,
    rng: np.random.Generator,
    scale_floor: tuple[float, float] = (0.01, 1.0),
) -> np.ndarray:
    """``(n_draws, n_units)``: ``point + scale * resampled centred residual``."""
    p = np.asarray(point, dtype=float).reshape(-1)
    s = np.asarray(size, dtype=float).reshape(-1)
    if p.size != s.size:
        raise ValueError(f"point and size must have the same length, got {p.size} and {s.size}")
    if not isinstance(n_draws, int) or n_draws < 1:
        raise ValueError(f"n_draws must be an int >= 1, got {n_draws!r}")
    scale = _scale(p, s, scale_floor)
    strata = calibration.strata_for(s)
    out = np.empty((n_draws, p.size))
    for u in range(p.size):
        pool = calibration.pools[strata[u] - 1]
        out[:, u] = p[u] + scale[u] * rng.choice(pool, size=n_draws, replace=True)
    return out


def leave_one_out_coverage(
    calibration: Calibration, *, levels: Sequence[float] = (0.8, 0.95)
) -> pd.DataFrame:
    """Coverage of intervals built for each calibration forecast from the OTHER
    units of its stratum (the R study's cross-company historical coverage)."""
    used = calibration.residuals
    rows = []
    for level in levels:
        alpha = (1.0 - level) / 2.0
        covered = []
        for _, r in used.iterrows():
            pool = used.loc[
                (used["stratum"] == r["stratum"]) & (used["unit"] != r["unit"]), "standardised_error"
            ].to_numpy(dtype=float)
            centre = np.median(pool)
            lo = r["predicted"] + r["residual_scale"] * np.quantile(pool - centre, alpha, method="median_unbiased")
            hi = r["predicted"] + r["residual_scale"] * np.quantile(pool - centre, 1 - alpha, method="median_unbiased")
            covered.append(lo <= r["actual"] <= hi)
        cov = used.assign(covered=covered).groupby("horizon")["covered"].agg(["mean", "size"])
        for horizon, (mean, n) in cov.iterrows():
            rows.append(
                {"horizon": int(horizon), "nominal_coverage": float(level),
                 "empirical_coverage": float(mean), "forecasts": int(n)}
            )
    return pd.DataFrame(rows, columns=["horizon", "nominal_coverage", "empirical_coverage", "forecasts"])
```

`np.quantile(..., method="median_unbiased")` is R's type 8, the type the R study uses. If the leave-one-out loop is slow on 600 rows it is still under a second; do not vectorise it in this PR.

- [ ] **Step 4: Run the tests, adjust tolerances only if a test is wrong about the mathematics (not to make it pass), commit**

Run: `uv run pytest tests/test_residual_calibration.py -q`

```bash
git add src/ibnr/kernels/residual_calibration.py tests/test_residual_calibration.py
git commit -m "feat: size-stratified rolling-origin residual calibration for point forecasters

rolling_residuals, calibrate, calibrated_draws and leave_one_out_coverage,
the R study's uncertainty recipe written once and generic over the forecaster."
```

---

### Task 8: exports, docs, changelog

**Files:**
- Modify: `src/ibnr/kernels/__init__.py` (import `Calibration`, `calibrate`, `calibrated_draws`, `leave_one_out_coverage`, `rolling_residuals`; add to `__all__`)
- Modify: `great-docs.yml` (new section after `Evaluation kernels`: title `Residual calibration`, desc one sentence, contents the five names with `members: true` on `kernels.Calibration`)
- Modify: `CHANGELOG.md` (`## Unreleased`)
- Modify: `src/ibnr/gallery/nn/_training.py` module docstring (if not done in Task 6)

- [ ] **Step 1: Wire and document**

CHANGELOG paragraph:

```
New: the shared NN training loop (`gallery/nn/_training.py::train_ensemble`)
takes a learning-rate `schedule` (`warmup_cosine` ports the R study's warmup
plus cosine decay), `param_groups`, `min_epochs`, `check_every` (patience now
counts validation checks), `cutoff_sampling="per_epoch"` and `keep` (best k
members by validation score). Every default is bit-identical to 0.6.0,
checked against a verbatim copy of the old loop. New:
`kernels/residual_calibration.py`, the size-stratified rolling-origin
residual calibration of a point forecaster (`rolling_residuals`,
`calibrate`, `calibrated_draws`, `leave_one_out_coverage`).
```

- [ ] **Step 2: Run the import-boundary tests**

Run: `uv run pytest tests/test_gallery.py tests/test_import_purity.py tests/test_residual_calibration.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/ibnr/kernels/__init__.py great-docs.yml CHANGELOG.md src/ibnr/gallery/nn/_training.py
git commit -m "feat: export the residual calibration kernel; document the training scheme"
```

---

### Task 9: lint, suites, rebase, pull request

- [ ] **Step 1: Lint** - `uv run ruff check . && uv run ruff format . && uv run python scripts/lint_md_snippets.py`
- [ ] **Step 2: Suites** - `uv run pytest -q -rs` with torch installed. Confirm `tests/test_nn_training_scheme.py`, `tests/test_nn_transformer.py`, `tests/test_nn_transformer_ml.py`, `tests/test_deeptriangle.py`, `tests/test_mdn.py`, `tests/test_resnet.py`, `tests/test_nn_paid_case.py` and `tests/test_nn_training_context.py` EXECUTED (not skipped). Record counts.
- [ ] **Step 3: Rebase if `git log --oneline HEAD..origin/main` is non-empty**, rerun the suites.
- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/training-scheme-calibration
gh pr create --repo EKtheSage/ibnr --base main --title "feat: NN training scheme extensions and residual calibration" --body-file <(cat <<'EOF'
`train_ensemble` gains `schedule` (with `warmup_cosine`), `param_groups`, `min_epochs`, `check_every` (patience in checks), `cutoff_sampling="per_epoch"` and `keep`, every default bit-identical to 0.6.0 against a verbatim copy of the old loop kept in the test file. `kernels/residual_calibration.py` is the R study's size-stratified rolling-origin residual calibration, generic over any point forecaster.

Two departures from the spec, both explained in the module docstrings: no `n_workers` (the callbacks are closures over tensors and cannot cross a spawned process) and `warmup_cosine` lives in `gallery/nn/_training.py` rather than `ibnr.kernels` (kernels never imports the gallery).

Spec: `docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md`, sections 5.2 and 5.3 (on branch `design/tlrn-mcl-point-scores`).

Verification: `uv run pytest -q -rs` with `--extra nn` (<counts>; the NN test files executed), ruff and `scripts/lint_md_snippets.py` clean. Mutation checks tried: <list>.
EOF
)
```

Do not merge. Report back: the PR URL, changed files, the verification output, the two departures, and anything unresolved.
