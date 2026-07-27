"""Training-scheme helpers shared by every NN gallery entry. Torch-free.

Two functions, both born in ``gallery/nn/transformer/model.py`` and promoted
here once ``transformer_ml`` started importing them across entry boundaries
under their old private names. They are the *scheme*, not the network: how the
eval_date-style validation split is carved and how per-dev standardization is
pinned. Any entry that consumes ``kernels.nn_contract`` grids should use these
rather than re-derive them - the pinning rule in :func:`norm_stats` is the v2
fix recorded in CLAUDE.md (milestone 3) and re-implementing it is the single
most expensive regression available to a new NN entry.
"""

from __future__ import annotations

import numpy as np

__all__ = ["norm_stats", "splits"]


def splits(
    obs_mask: np.ndarray, cal_idx: np.ndarray, val_diagonals: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Carve the eval_date validation split out of the *training window*.

    Returns (context_eligible, val_target, val_cutoff): the trailing
    ``val_diagonals`` observed calendar diagonals become the validation
    targets for early stopping and are excluded from every training context
    and every training target - validating by calendar time (eval_date), the
    way the model is actually used at prediction, rather than by a random cell
    split. ``val_cutoff`` is the newest diagonal training may condition on."""
    c_max = int(cal_idx[obs_mask.any(axis=0)].max())
    val_cutoff = c_max - val_diagonals
    if val_cutoff < 2:
        raise ValueError(
            f"latest observed diagonal is {c_max}; need at least {2 + val_diagonals} "
            "diagonals to hold one out for validation and still train"
        )
    context_eligible = obs_mask & (cal_idx <= val_cutoff)
    val_target = obs_mask & (cal_idx > val_cutoff)
    if not val_target.any():
        raise ValueError("no cells on the validation diagonal(s)")
    return context_eligible, val_target, val_cutoff


def norm_stats(
    x: np.ndarray, cells: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-(channel, dev) mean/std over training-context ``cells`` - the
    validation diagonals must not leak through the normalizer.

    Devs with fewer than two context values (or zero spread) - in practice
    the deepest dev, whose only observations sit on the held-out validation
    diagonal - are *pinned*: their standardized values are defined as 0, so
    unstandardizing a prediction there returns the pooled dev mean rather
    than a value denormalized at an earlier dev's magnitude (the old
    inherit-earlier-stats scheme, which overstated fast-decaying tails).
    A pinned dev's mean falls back to all ``obs`` cells at that dev - the
    only data that exists there; std is fixed at 1."""
    _, n_f, _, n_d = x.shape
    mean, std = np.zeros((n_f, n_d)), np.ones((n_f, n_d))
    pinned = np.zeros((n_f, n_d), dtype=bool)
    for f in range(n_f):
        for d in range(n_d):
            vals = x[:, f, :, d][cells[:, :, d]]
            if vals.size >= 2 and float(vals.std()) > 1e-8:
                mean[f, d], std[f, d] = float(vals.mean()), float(vals.std())
            else:
                fallback = x[:, f, :, d][obs[:, :, d]]
                mean[f, d] = float(fallback.mean()) if fallback.size else 0.0
                pinned[f, d] = True
    return mean, std, pinned
