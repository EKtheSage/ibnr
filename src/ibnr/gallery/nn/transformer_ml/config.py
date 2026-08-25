"""Hyperparameters for the multi-line triangle transformer. Torch-free on
purpose: the entry registers (and this config is importable) without the
[nn] extra.

Network defaults mirror the single-line transformer (disclosed as
hand-chosen for the Schedule P regime); the new knob is ``dependence`` -
how cross-line dependence enters the DRAWS (the encoder always attends
across lines):

- "ar":    univariate MDN per cell; within each rollout diagonal the lines
           are sampled one at a time in a seeded random order, each fed
           back as context before the next - dependence via conditioning.
- "joint": one multivariate Gaussian mixture per (origin, dev) cell-group
           across the company's lines (Cholesky-parameterized), sampled
           jointly - dependence via an explicit correlated head.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerMLConfig:
    """Hyperparameters for one ``nn_transformer_ml`` fit.

    Network block mirrors the single-line ``TransformerConfig`` (same regime,
    same hand-tuned Schedule P defaults); the multi-line addition is
    ``dependence``, which selects how cross-line correlation enters the
    predictive draws. See card.md for the study these defaults were fixed
    against.
    """

    # network
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.15
    n_components: int = 3  # mixture components K (shared by both heads)
    # NOTE: no line_embedding_dim knob - unlike the single-line entry (which
    # CONCATENATES its LOB embedding then projects), this network ADDS the line
    # embedding to the token, so its width is necessarily d_model.
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
    # A SEPARATE budget from n_draws, because the two cost different things:
    # n_draws pays for predict()'s rollout, which re-runs the network once per
    # future diagonal, while the held-out draws behind predict_at cover ONE
    # diagonal - a single forward pass per ensemble member and then mixture
    # sampling, so 10,000 is cheap here and the same count in a rollout is not.
    # 10,000 is what every other CRPS-capable entry puts on the board (the
    # Bayesian posteriors' 4x2500; mack's fit kwarg, whose name this borrows).
    heldout_n_draws: int = 10_000
