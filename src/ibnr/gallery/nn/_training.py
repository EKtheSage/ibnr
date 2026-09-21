"""The deep-ensemble training loop shared by every NN gallery entry.

Extracted from ``NNTransformer.fit`` / ``NNTransformerML.fit``, which carried
~110 near-identical lines each; a third copy per new entry (deeptriangle, mdn,
resnet) would have made the training scheme un-auditable. The loop owns
everything the two copies agreed on:

- member seeding: ``member_seed = seed + 1000 * member`` feeds BOTH
  ``torch.manual_seed`` (weight init, dropout) and the numpy generator
  (batch shuffling, augmented cutoffs) - the widely-spaced per-member seeds
  from milestone 3;
- calendar-cutoff augmentation batching: per epoch a fresh cohort
  permutation, per batch one augmented cutoff per cohort drawn uniformly
  from ``[min_cutoff, val_cutoff)``;
- AdamW + gradient clipping;
- the per-epoch validation pass (``model.eval()`` under ``no_grad``);
- early stopping with best-validation state snapshot/restore.

What stays in the entry, behind two callbacks, is exactly what differs:
building the model (``make_model``) and evaluating the loss on a batch or on
the validation split (``train_loss`` / ``val_loss``) - mask shapes and head
dispatch are entry business, the loop never sees a tensor of data.

The RNG consumption order is IDENTICAL to the pre-refactor loops (seed ->
construct model -> per-epoch permutation -> per-batch cutoffs -> forward), so
a fixed seed reproduces the milestone-3 fits bit for bit; the equivalence was
verified against pre-refactor loss trajectories, final weights and predictive
draws on both entries.

Torch is imported inside :func:`train_ensemble` only - this module must be
importable without the ``[nn]`` extra (the subprocess test in
``tests/test_gallery.py`` is the honest check).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["train_ensemble", "warmup_cosine"]


def warmup_cosine(schedule_epochs: int, warmup: int = 20) -> Callable[[int], float]:
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
    min_epochs: int = 0,
    check_every: int = 1,
) -> tuple[list, list[list[dict]]]:
    """Train ``config.ensemble_size`` independently-seeded members.

    n_cohorts:   training examples (axis 0 of the entry's contract arrays).
    config:      any config carrying ``ensemble_size``, ``batch_size``,
                 ``max_epochs``, ``patience``, ``lr``, ``weight_decay`` and
                 ``grad_clip`` (both transformer configs do; duck-typed so a
                 new entry's config needs no shared base class).
    seed:        entry-level seed; member ``m`` trains under
                 ``seed + 1000 * m`` (None stays None: unseeded).
    make_model:  ``() -> nn.Module`` on ``device``, called AFTER the member's
                 ``torch.manual_seed`` so weight init consumes the member
                 stream. Any post-construction setup (e.g. filling
                 ``prem_log_std``) belongs in this closure.
    train_loss:  ``(model, idx, cutoffs) -> Tensor | None`` - the batch NLL.
                 ``idx`` is a long tensor of cohort rows, ``cutoffs`` one
                 augmented conditioning diagonal per row. Return ``None`` to
                 skip a batch whose cutoffs left nothing to score (the draw
                 still consumed the rng, keeping streams aligned).
    val_loss:    ``(model) -> float`` - validation NLL at the fixed
                 ``val_cutoff`` split. Called with the model in eval mode
                 under ``torch.no_grad()``.
    min_cutoff / val_cutoff: the augmentation window; cutoffs are drawn from
                 ``[min_cutoff, val_cutoff)`` so the fixed validation
                 diagonal(s) are never conditioned on during training.
    device:      the ``torch.device`` batches should be built on.
    schedule:    ``(epoch) -> float``, a multiplier on every parameter group's
                 own base learning rate, called once per epoch with a 1-based
                 epoch. ``None`` holds every rate at its base value, which is
                 what the loop did before. :func:`warmup_cosine` is the R
                 study's schedule.
    param_groups: ``(model) -> list[dict]``, AdamW parameter groups, each free
                 to carry its own ``lr``. ``None`` is one group over
                 ``model.parameters()`` at ``config.lr``. A group without an
                 ``lr`` inherits ``config.lr``.
    min_epochs:  early stopping cannot end a member before this many epochs
                 have run. 0 is the old behaviour.
    check_every: run the validation pass every this many epochs, and always on
                 the last epoch. ``config.patience`` counts validation checks,
                 which at ``check_every=1`` is the old count in epochs. An
                 epoch that was not validated records ``val = nan``.

    Returns ``(models, histories)``: each model in eval mode with its
    best-validation weights restored, and per-member per-epoch
    ``{"epoch", "train", "val"}`` records.
    """
    import torch

    if not isinstance(check_every, int) or check_every < 1:
        raise ValueError(f"check_every must be an int >= 1, got {check_every!r}")
    if not isinstance(min_epochs, int) or min_epochs < 0 or min_epochs > config.max_epochs:
        raise ValueError(
            f"min_epochs must be an int in [0, max_epochs={config.max_epochs}], got {min_epochs!r}"
        )

    models: list = []
    histories: list[list[dict]] = []
    # DEEP ENSEMBLE: train ensemble_size independent members, each with its
    # own seed (weights, batch order, and augmented cutoffs all differ).
    # Pooling their draws at predict time adds epistemic spread on top of the
    # head's aleatoric spread.
    for member in range(config.ensemble_size):
        # widely-spaced per-member seed so members don't share an RNG stream
        member_seed = None if seed is None else seed + 1000 * member
        if member_seed is not None:
            torch.manual_seed(member_seed)
        rng = np.random.default_rng(member_seed)
        model = make_model()
        params = param_groups(model) if param_groups is not None else model.parameters()
        opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]  # the rate the schedule multiplies

        best_val, best_state, patience_left = math.inf, None, config.patience
        history: list[dict] = []
        for epoch in range(config.max_epochs):
            if schedule is not None:
                mult = schedule(epoch + 1)  # 1-based, as the R study counts epochs
                if not (isinstance(mult, int | float) and math.isfinite(mult) and mult >= 0):
                    raise ValueError(
                        f"schedule({epoch + 1}) returned {mult!r}; a learning-rate multiplier "
                        "must be a finite number >= 0"
                    )
                for group in opt.param_groups:
                    group["lr"] = group["base_lr"] * mult
            model.train()
            epoch_loss, n_batches = 0.0, 0
            perm = rng.permutation(n_cohorts)  # shuffle cohorts into batches each epoch
            for start in range(0, n_cohorts, config.batch_size):
                idx = torch.tensor(perm[start : start + config.batch_size], device=device)
                # CALENDAR-CUTOFF AUGMENTATION: draw a fake as_of diagonal per
                # cohort; the entry conditions on cells on/before it and scores
                # the observed training cells strictly after it. Each triangle
                # yields many "predict the next diagonals" tasks per epoch -
                # the main small-data multiplier.
                cutoffs = torch.tensor(
                    rng.integers(min_cutoff, val_cutoff, size=len(idx)), device=device
                )  # (B,) 1-based conditioning diagonal per cohort
                loss = train_loss(model, idx, cutoffs)
                if loss is None:
                    continue  # this batch's cutoffs left nothing to score
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()
                epoch_loss += float(loss.detach())
                n_batches += 1

            # validation: condition on the whole training window (cutoff =
            # val_cutoff), score NLL on the held-out trailing diagonal(s).
            # This is the eval_date-style split, not a random cell holdout.
            # Epochs between two checks record a NaN val, which
            # kernels.tuning.validation_score already skips.
            is_last = epoch + 1 == config.max_epochs
            if (epoch + 1) % check_every != 0 and not is_last:
                history.append(
                    {"epoch": epoch, "train": epoch_loss / max(n_batches, 1), "val": math.nan}
                )
                continue
            model.eval()
            with torch.no_grad():
                epoch_val = float(val_loss(model))
            history.append(
                {"epoch": epoch, "train": epoch_loss / max(n_batches, 1), "val": epoch_val}
            )
            if show_progress:
                print(f"member {member} epoch {epoch}: val {epoch_val:.4f}")
            # early stopping: snapshot best-val weights, stop after
            # `patience` checks without improvement, then restore the best.
            if epoch_val < best_val - 1e-6:
                best_val, patience_left = epoch_val, config.patience
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0 and epoch + 1 >= min_epochs:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)  # restore best-val weights
        model.eval()
        models.append(model)
        histories.append(history)
    return models, histories
