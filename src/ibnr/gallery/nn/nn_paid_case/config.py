"""Hyperparameters for ``nn_paid_case``. Torch-free on purpose: the entry
registers (and this config is importable) without the [nn] extra.

ONE config for TWO backbones, which is the only interesting thing about this
file. The entry ships a transformer body and a GRU body over the same data
contract, the same loss and the same head, so the comparison between them
isolates the encoder alone - and that only holds while everything else is
literally the same object. Hence one dataclass rather than two.

The obvious way to do that is a union of both bodies' knobs, silently ignoring
whichever half the chosen backbone does not read. That is the repo's named
inert-parameter bug class: ``NNPaidCaseConfig(backbone="gru", d_model=256)``
would train a 64-wide GRU and report a config saying 256. So the union is split
into blocks and :meth:`NNPaidCaseConfig.__post_init__` REFUSES a foreign knob
set away from its default, naming the knob and the backbone that owns it. Same
idiom as ``ResNetConfig``'s divisibility check: the config refuses to describe a
fit that cannot happen.

A foreign knob left at its default is accepted, and has to be: the defaults are
what ``NNPaidCaseConfig()`` itself carries, so refusing them would mean no
default config could be built at all. The check is therefore "moved away from
the default", which is exactly the set of values a caller can have typed.

Defaults are hand-chosen for the Schedule P regime (~600 cohorts x ~55 observed
cells) and disclosed in card.md; systematic HPO lives in kernels/tuning.py.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

#: Knobs that belong to exactly ONE backbone's body, by backbone name. Every
#: other field is shared - read by both bodies, or by the training scheme and
#: the predictive paths, which are backbone-independent. This mapping is what
#: ``__post_init__`` refuses against, and it is also the complete list of what
#: the two bodies do NOT have in common.
BACKBONE_KNOBS: dict[str, tuple[str, ...]] = {
    "transformer": ("d_model", "n_layers", "n_heads", "ffn_dim"),
    "gru": ("hidden_dim", "company_embedding", "company_embedding_dim"),
}


@dataclass
class NNPaidCaseConfig:
    """All knobs for one ``nn_paid_case`` fit, grouped by role.

    Blocks: which backbone, the shared network settings, the two per-backbone
    body blocks, optimization, data scheme, predictive. The optimization block
    is duck-typed by ``gallery/nn/_training.py::train_ensemble`` and the
    data-scheme/predictive blocks mirror ``TransformerConfig`` value for value,
    so a nn_paid_case-vs-transformer comparison isolates what this entry changes
    (the joint head and the state-update rollout) rather than the schedule.
    """

    #: ``"transformer"`` (masked-cell attention over the whole grid) or
    #: ``"gru"`` (per-origin encoder/decoder recurrence). Both end in the same
    #: bivariate head and train on the same loss; see card.md "Backbones".
    backbone: str = "transformer"

    # network, shared by both bodies
    n_components: int = 3  # mixture components K (bivariate Gaussians per cell)
    dropout: float = 0.15
    lob_embedding_dim: int = 8

    # network, TRANSFORMER body only
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 128

    # network, GRU body only
    hidden_dim: int = 64  # GRU hidden state = token width
    # OFF by default, unlike ``deeptriangle`` (where it is Kuo's own design):
    # the two backbones here exist to be compared, and an embedding table one of
    # them cannot have would make the comparison partly about company identity.
    # ~600 companies x ~55 cells is a memorization vector besides. Flag it on
    # for the with-embedding arm - a GRU-only experiment by construction.
    company_embedding: bool = False
    company_embedding_dim: int = 8

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
    n_draws: int = 1000  # rollout draws (predict), pooled across members
    # ROLLOUT STATE, and a SHARED knob - both bodies roll out through the same
    # code, so this is not in BACKBONE_KNOBS and neither backbone refuses it.
    # A case reserve is booked down TO zero and never past it: when the claim
    # closes the department releases whatever is left, and an outstanding
    # position below nothing does not exist. So the simulated level walk is
    # truncated there, ``level = max(level + movement, 0)``. Only the case STATE
    # is floored - the joint (paid, movement) draw is untouched and consumes the
    # same randomness either way - so ``False`` reproduces the 0.5.5 walk
    # exactly for the same seed, and a floored and an unfloored run share every
    # draw. It is kept rather than deleted so the change can be measured with
    # and without it.
    floor_case_at_zero: bool = True
    # held-out CRPS draws (predict_at), a separate budget from the rollout
    # above: scoring one diagonal costs a single forward pass per ensemble
    # member plus mixture sampling, so 10,000 - what every other CRPS-capable
    # entry puts on the board - is cheap here where a 10,000-draw rollout is
    # not. Same value and same reasoning as ``TransformerConfig``.
    heldout_n_draws: int = 10_000

    def __post_init__(self) -> None:
        if self.backbone not in BACKBONE_KNOBS:
            raise ValueError(
                f"unknown backbone {self.backbone!r}; nn_paid_case ships "
                f"{sorted(BACKBONE_KNOBS)} and they share everything but the encoder body"
            )
        default = {f.name: f.default for f in fields(self)}
        foreign = [
            (name, owner)
            for owner, knobs in BACKBONE_KNOBS.items()
            if owner != self.backbone
            for name in knobs
            if getattr(self, name) != default[name]
        ]
        if foreign:
            listed = ", ".join(
                f"{name}={getattr(self, name)!r} (belongs to the {owner!r} backbone)"
                for name, owner in foreign
            )
            raise ValueError(
                f"backbone={self.backbone!r} does not read {listed}. A knob the chosen "
                "backbone cannot consume would be silently inert - the config would "
                "describe a network that was never built. Set backbone= to the body you "
                "meant, or leave the other body's knobs at their defaults"
            )
