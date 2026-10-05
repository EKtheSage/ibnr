"""Hyperparameters for ``tlrn``. Torch-free on purpose: the entry registers (and
this config is importable) without the [nn] extra.

The defaults ARE the companion study's published checkpoint protocol, value for
value, so ``TLRNConfig()`` reproduces the run the study reports rather than a
tuned-for-this-repo variant of it. Three of them look odd out of context and are
worth reading before they are changed:

``patience = 600`` counts validation checks, and at ``check_every = 5`` that is
3000 epochs - the whole budget. It never fires. That is deliberate: the study
runs every member for the full schedule and keeps the best validation
checkpoint, so early stopping is not part of the protocol. ``patience = 120`` is
the 600-epoch early-stopping variant, which is a different run and has to be
asked for.

``lr_phi`` is ten times ``lr``. ``phi`` is the per (line, step) log development
factor, the parameter that carries the level of the forecast, and the network
around it only corrects that level - so the two train at different rates and
:func:`~ibnr.gallery.nn._training.train_ensemble` is given two parameter groups.

``ensemble_size = 10`` with ``keep = 2`` trains ten independently seeded members
and keeps the two that validated best. It is a selection over optimiser
outcomes, not a deep ensemble whose members are all averaged, and the entry's
``selection_`` table reports every member so the ones that were dropped stay
visible.

The switches ``cross_line``, ``factors_only`` and ``cl_anchor`` exist so each
design choice can be run with and without it. Every one of them is off-by-
default in the sense that the default is what the study published.
"""

from __future__ import annotations

from dataclasses import dataclass

from ibnr.gallery.nn._training import CUTOFF_SAMPLING
from ibnr.gallery.nn.tlrn.components import check_combination

#: what happens to a development step no training target ever supervised.
#: ``observed_cl`` substitutes the chain ladder factors observable at the
#: example's own cutoff; ``legacy_init`` leaves the parameter at whatever it
#: drifted to, which is the study's earlier behaviour and is kept switchable.
TAIL_POLICY: tuple[str, ...] = ("observed_cl", "legacy_init")


@dataclass
class TLRNConfig:
    """All knobs for one ``tlrn`` fit, grouped by role.

    The optimisation block is duck-typed by
    ``gallery/nn/_training.py::train_ensemble`` (``ensemble_size``,
    ``batch_size``, ``max_epochs``, ``patience``, ``lr``, ``weight_decay``,
    ``grad_clip``), which is why those names match the other NN configs even
    where this entry's values do not.
    """

    # network
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    dropout: float = 0.3
    #: how much of the log factor the network is allowed to move, per step
    eps: float = 0.5
    #: log factor at step m is initialised to ``init_step / m``, so the factor
    #: to ultimate starts near exp(0.57 * H_9) - between a fast line and a slow
    #: one. The shape is fixed; the magnitudes are learned.
    init_step: float = 0.57
    #: attention across lines within a lag. Off leaves the lag attention alone
    #: and makes the network single-line, at the same parameter count.
    cross_line: bool = True
    #: train the log factors alone, with no network correction at all
    factors_only: bool = False
    #: start from each company's own chain ladder factors and learn a bounded
    #: correction around them, rather than from the decaying initialisation
    cl_anchor: bool = False
    #: half-width of that correction, in log factor units
    anchor_width: float = 0.1
    tail_policy: str = "observed_cl"
    #: what the network output is read as; see ``components.py`` for the choices
    #: and which of them can be combined. ``ldf`` is the published model.
    head: str = "ldf"
    #: the axes a block attends along, in order. ``None`` follows ``cross_line``:
    #: ``("line", "lag")`` or ``("lag",)``. ``("line", "lag", "ay")`` adds the
    #: accident-year attention and needs ``batch_unit="company"``.
    attention: tuple[str, ...] | None = None
    #: what an attention may read: ``unwritten_lines`` masks only absent lines,
    #: ``observed_cells`` masks every cell the forecast date has not revealed
    mask: str = "unwritten_lines"
    #: how a trained network becomes a member's forecast: ``network`` alone, or
    #: ``mcl_blend``, the multivariate chain ladder plus alpha times the
    #: network's difference from it (alpha is fitted at each validation check)
    member: str = "network"
    #: the premium head's cap on ``beta + eps * net``, so a ratio is at most e**cap
    lr_cap: float = 5.0

    # loss: the accident-year/line absolute percentage error, a pooled bias
    # penalty, and a squared-error term on the scale the ratios live on
    w_pe: float = 0.5
    w_mse: float = 0.1
    mse_scale: float = 0.005

    # optimisation
    lr: float = 3e-3
    #: the log factors train ten times faster than the network around them
    lr_phi: float = 3e-2
    weight_decay: float = 0.0
    warmup: int = 20
    #: what ``batch_size`` counts: ``example`` is one (company, accident year),
    #: ``company`` is a whole company with every accident year
    batch_unit: str = "example"
    #: batch units per batch: examples, or companies under ``batch_unit="company"``
    batch_size: int = 64
    max_epochs: int = 3000
    min_epochs: int = 500
    check_every: int = 5
    #: counted in validation checks; see the module docstring for why the
    #: default never fires
    patience: int = 600
    grad_clip: float = 1.0

    # data
    min_cutoff: int = 2
    val_diagonals: int = 2
    cutoff_sampling: str = "per_epoch"

    # ensemble: train this many seeds, keep the best few by validation score
    ensemble_size: int = 10
    #: members averaged; ``None`` averages every trained member
    keep: int | None = 2

    # uncertainty: the historical residual calibration. ``calibration`` says
    # whether the forecasts it scores come from the final models (``rescore_final``)
    # or from the method retrained at each cutoff (``retrain_per_valuation``)
    calibration: str = "rescore_final"
    calibration_cutoffs: tuple[int, ...] = (5, 6, 7, 8, 9)
    calibration_horizons: tuple[int, ...] = (3, 4, 5)
    n_strata: int = 4
    min_per_stratum: int = 40
    n_draws: int = 4000

    @property
    def n_kept(self) -> int:
        """How many members are averaged: ``keep``, or all of them when it is ``None``."""
        return self.ensemble_size if self.keep is None else self.keep

    def __post_init__(self) -> None:
        if self.keep is not None and not 1 <= self.keep <= self.ensemble_size:
            raise ValueError(
                f"keep must be in [1, ensemble_size={self.ensemble_size}], got {self.keep}. "
                "The kept members are the ones that validated best, so keeping more than "
                "were trained is not a request this entry can meet"
            )
        if self.patience < 1:
            raise ValueError(
                f"patience counts validation checks and must be >= 1, got {self.patience}"
            )
        if self.check_every < 1:
            raise ValueError(f"check_every must be >= 1, got {self.check_every}")
        if self.min_epochs > self.max_epochs:
            raise ValueError(
                f"min_epochs {self.min_epochs} is past max_epochs {self.max_epochs}, so early "
                "stopping could never be reached and the two disagree about the budget"
            )
        if self.val_diagonals < 1:
            raise ValueError(
                f"val_diagonals must be >= 1, got {self.val_diagonals}: the member kept is the "
                "best validation checkpoint, and with no held-out diagonal there is nothing to "
                "choose it on"
            )
        if self.tail_policy not in TAIL_POLICY:
            raise ValueError(
                f"tail_policy must be one of {list(TAIL_POLICY)}, got {self.tail_policy!r}"
            )
        if self.cutoff_sampling not in CUTOFF_SAMPLING:
            raise ValueError(
                f"cutoff_sampling must be one of {list(CUTOFF_SAMPLING)}, "
                f"got {self.cutoff_sampling!r}"
            )
        check_combination(self)
        if not self.lr_cap > 0:
            raise ValueError(f"lr_cap must be positive, got {self.lr_cap}")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} must be a positive multiple of n_heads {self.n_heads}: "
                "attention splits the token width evenly over the heads"
            )
