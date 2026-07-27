"""Hyperparameters for the residual convolutional triangle network. Torch-free
on purpose: the entry registers (and this config is importable) without the
[nn] extra.

Defaults are hand-chosen for the Schedule P regime (~600 cohorts x ~55
observed cells) and disclosed as such in card.md; systematic HPO lives in
kernels/tuning.py (random search over this dataclass)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResNetConfig:
    """All knobs for one ``resnet`` fit, grouped by role (network,
    optimization, data scheme, predictive). Held in a plain dataclass so it is
    importable without torch and so ``card.md``'s disclosed defaults live in
    exactly one place. The optimization / data-scheme / predictive blocks
    mirror ``TransformerConfig`` value for value, so a resnet-vs-transformer
    comparison isolates the encoder body (the ablation this entry exists for).
    """

    # network (~75k params at these sizes; deliberately tiny - the Schedule P
    # regime is small-data and the central risk is overfitting, see card.md)
    channels: int = 64  # conv width throughout the residual trunk
    n_blocks: int = 3  # residual blocks (two 3x3 convs each)
    n_groups: int = 8  # GroupNorm groups; must divide channels. NEVER
    # BatchNorm: batch statistics would couple cohorts that were conditioned
    # at different augmented cutoffs (see card.md "Why GroupNorm")
    dropout: float = 0.15  # 2-D channel dropout after the stem
    n_components: int = 3  # MDN mixture components K (Gaussians per cell)
    lob_embedding_dim: int = 8
    # optimization (AdamW; patience = early-stopping window on validation NLL)
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

    def __post_init__(self) -> None:
        if self.channels % self.n_groups != 0:
            raise ValueError(
                f"n_groups ({self.n_groups}) must divide channels ({self.channels}); "
                "GroupNorm partitions the channel axis into equal groups"
            )
