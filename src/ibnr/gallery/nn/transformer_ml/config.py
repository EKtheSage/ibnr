"""Hyperparameters for the multi-line triangle transformer. Torch-free on
purpose: the entry registers (and this config is importable) without the
[nn] extra.

Network defaults mirror the single-line transformer (disclosed as
hand-chosen for the Schedule P regime); the new knob is ``dependence`` —
how cross-line dependence enters the DRAWS (the encoder always attends
across lines):

- "ar":    univariate MDN per cell; within each rollout diagonal the lines
           are sampled one at a time in a seeded random order, each fed
           back as context before the next — dependence via conditioning.
- "joint": one multivariate Gaussian mixture per (origin, dev) cell-group
           across the company's lines (Cholesky-parameterized), sampled
           jointly — dependence via an explicit correlated head.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerMLConfig:
    """Hyperparameters for one ``nn_transformer_ml`` fit.

    Network block mirrors the single-line ``TransformerConfig`` (same regime,
    same hand-tuned Schedule P defaults); the multi-line additions are the
    line embedding and, above all, ``dependence``, which selects how cross-line
    correlation enters the predictive draws. See card.md for the study these
    defaults were fixed against.
    """

    # network
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.15
    n_components: int = 3  # mixture components K (shared by both heads)
    line_embedding_dim: int = 8  # per-line identity added to every token
    # cross-line dependence mechanism (the entry's research knob):
    #   "ar"    -> univariate MDN per cell; dependence via line-by-line
    #              autoregressive sampling within each rollout diagonal.
    #   "joint" -> multivariate Gaussian-mixture head over the line vector;
    #              dependence modeled explicitly, drawn in one shot.
    dependence: str = "ar"  # "ar" | "joint"
    # optimization
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32  # companies per batch (each carries L line-triangles)
    max_epochs: int = 400
    patience: int = 25  # early-stopping patience on validation NLL
    grad_clip: float = 1.0
    # data scheme
    min_cutoff: int = 4  # earliest calendar diagonal a training cutoff may fall on
    val_diagonals: int = 1  # trailing diagonals held out for the validation split
    # predictive
    ensemble_size: int = 5  # deep-ensemble members; draws are split across them
    n_draws: int = 1000  # total predictive draws pooled over the ensemble
