"""Hyperparameters for the triangle transformer. Torch-free on purpose: the
entry registers (and this config is importable) without the [nn] extra.

Defaults are hand-chosen for the Schedule P regime (~600 cohorts x ~55
observed cells) and disclosed as such in card.md; systematic HPO arrives
with kernels/tuning.py (deferred)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # network
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.15
    n_components: int = 3  # MDN mixture components
    lob_embedding_dim: int = 8
    # exposure-aware sigma: when True the MDN scale carries a learnable
    # premium power p, so the predictive dollar sd scales as premium**p
    # instead of the flat premium**1 (constant coefficient of variation) the
    # ratio normalization otherwise bakes in. p = softplus(raw_p) is a single
    # trained scalar initialized at p = 1.0, which reproduces the baseline
    # exactly. Off by default so the baseline stays a clean on/off comparison.
    # See card.md "Exposure-aware sigma".
    exposure_sigma: bool = False
    # optimization
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 400
    patience: int = 25
    grad_clip: float = 1.0
    # data scheme
    min_cutoff: int = 4  # smallest augmented calendar cutoff (clamped to fit)
    val_diagonals: int = 1  # trailing calendar diagonals held out of training
    # predictive
    ensemble_size: int = 5
    n_draws: int = 1000
