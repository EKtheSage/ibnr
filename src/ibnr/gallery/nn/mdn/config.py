"""Hyperparameters for the mdn gallery entry. Torch-free on purpose: the
entry registers (and this config is importable) without the [nn] extra.

Optimization, data-scheme and predictive defaults are IDENTICAL to
``TransformerConfig`` - the entry is the transformer's architecture ablation,
so everything that is not the encoder body is held fixed on purpose (see
card.md "Why this entry exists"). Systematic HPO lives in kernels/tuning.py."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MDNConfig:
    """All knobs for one ``mdn`` fit. The network block is the only part that
    departs from ``TransformerConfig``: there is no attention, so the sizes
    describe a per-cell MLP rather than an encoder stack."""

    # network (per-cell MLP; ~30k params at these sizes - even smaller than
    # the transformer, and the Schedule P small-data risk applies unchanged)
    hidden_dim: int = 128
    n_layers: int = 2  # hidden layers in the MLP body
    dropout: float = 0.1
    n_components: int = 3  # MDN mixture components K (Gaussians per cell)
    embedding_dim: int = 8  # origin / dev / distance-past-cutoff embeddings
    lob_embedding_dim: int = 8
    # optimization (AdamW; patience = early-stopping window on validation NLL).
    # Same values as the transformer - the ablation holds the recipe fixed.
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 64  # cohort-triangles per batch, not cells
    max_epochs: int = 400
    patience: int = 25
    grad_clip: float = 1.0
    # data scheme (calendar-cutoff augmentation + eval_date validation split)
    min_cutoff: int = 4  # smallest augmented calendar cutoff (clamped to fit)
    val_diagonals: int = 1  # trailing calendar diagonals held out of training
    # predictive
    ensemble_size: int = 5  # deep-ensemble members (distinct seeds) -> epistemic spread
    n_draws: int = 1000  # posterior-predictive draws, pooled across members
    # held-out CRPS draws (predict_at), a separate budget from the rollout
    # above: scoring one diagonal costs a single forward pass per ensemble
    # member plus mixture sampling, so 10,000 - what every other CRPS-capable
    # entry puts on the board - is cheap here where a 10,000-draw rollout is
    # not. Same value and same reasoning as ``TransformerConfig``.
    heldout_n_draws: int = 10_000
