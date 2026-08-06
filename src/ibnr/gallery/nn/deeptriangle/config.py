"""Hyperparameters for the DeepTriangle entry. Torch-free on purpose: the
entry registers (and this config is importable) without the [nn] extra.

Defaults are hand-chosen for the Schedule P regime (~600 cohorts x ~55
observed cells) and disclosed as such in card.md; systematic HPO lives in
kernels/tuning.py (random search over this dataclass)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeepTriangleConfig:
    """All knobs for one ``deeptriangle`` fit, grouped by role (network,
    optimization, data scheme, predictive). Held in a plain dataclass so it is
    importable without torch and so ``card.md``'s disclosed defaults live in
    exactly one place. The optimization block is duck-typed by
    ``gallery/nn/_training.py::train_ensemble``."""

    # network (GRU encoder/decoder; deliberately tiny - the Schedule P regime
    # is small-data and the central risk is overfitting, see card.md)
    hidden_dim: int = 64  # GRU hidden state = token width
    dropout: float = 0.15
    n_components: int = 3  # MDN mixture components K (Gaussians per cell)
    lob_embedding_dim: int = 8
    # company embedding: Kuo's actual design, and the deliberate departure from
    # the transformer's no-embedding choice. ON by default because it IS the
    # paper's architecture; flag it off for the comparison arm (card.md
    # "Company embedding"). ~600 companies x ~55 cells is a memorization
    # vector, so the comparison is the honest check, not an afterthought.
    company_embedding: bool = True
    company_embedding_dim: int = 8
    # auxiliary task: a second MDN head on the incremental claims-outstanding
    # loss ratio (OS = reported - paid, derived inside the entry), trained
    # jointly with this weight. 0.0 disables the auxiliary loss entirely -
    # the single-task comparison arm. Rollout and held-out scoring only ever
    # consume the target head, whatever this is set to.
    aux_weight: float = 1.0
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
    # held-out CRPS draws (predict_at), a separate budget from the rollout
    # above: scoring one diagonal costs a single forward pass per ensemble
    # member plus mixture sampling, so 10,000 - what every other CRPS-capable
    # entry puts on the board - is cheap here where a 10,000-draw rollout is
    # not. The name matches ``mack``'s fit kwarg, which means the same thing.
    heldout_n_draws: int = 10_000
