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

``refit`` and ``alpha_on`` are the two protocol changes of the companion study's newest
code, both off by default: the default config is still the published protocol, bit for bit.

The switches ``cross_line``, ``factors_only`` and ``cl_anchor`` exist so each
design choice can be run with and without it. Every one of them is off-by-
default in the sense that the default is what the study published.
"""

from __future__ import annotations

from dataclasses import dataclass

from ibnr.gallery.nn._training import CUTOFF_SAMPLING
from ibnr.gallery.nn.tlrn.components import attention_axes, check_combination

#: what happens to a development step no training target ever supervised.
#: ``observed_cl`` substitutes the chain ladder factors observable at the
#: example's own cutoff; ``legacy_init`` leaves the parameter at whatever it
#: drifted to, which is the study's earlier behaviour and is kept switchable.
TAIL_POLICY: tuple[str, ...] = ("observed_cl", "legacy_init")

#: which validation cells fit the blend weight alpha. ``all_validation`` fits it on every
#: validation diagonal and scores the blend on the same cells, which is the published
#: protocol. ``first_diagonal`` fits it on the first validation diagonal only and scores the
#: blend, which picks the checkpoint, on the second: the cells that choose the epoch are
#: then not the cells that chose alpha.
ALPHA_ON: tuple[str, ...] = ("all_validation", "first_diagonal")


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
    #: which cells validation (and a retrained calibration) scores: ``all_cells``,
    #: or ``reached_cells``, only those whose development steps the cutoff has seen
    scoring: str = "all_cells"
    #: the premium head's cap on ``beta + eps * net``, so a ratio is at most e**cap
    lr_cap: float = 5.0
    #: which validation cells fit alpha under ``member="mcl_blend"``; see :data:`ALPHA_ON`
    alpha_on: str = "all_validation"
    #: after the best checkpoint of each member is chosen, train the same member again
    #: from the same seed on every diagonal known at the cutoff (the validation diagonals
    #: included) and keep the weights at exactly the epoch the first run chose. Alpha is not
    #: fitted again. The forecasts are then the refit networks'; the networks as they
    #: stood at the chosen epoch stay readable (``selected=True`` on the read-outs)
    refit: bool = False
    #: what an attention returns for a query that can read no key at all (a hidden cell
    #: with every cell of its group hidden too). ``False`` returns zero, which also drops
    #: the attention's output bias. ``True`` returns that bias, which is what the
    #: companion's code does (its weights are zero there and the output projection still
    #: adds its bias). It only matters where such a query can occur, so it needs
    #: ``mask="observed_cells"`` or the ``"ay"`` axis
    empty_query_bias: bool = False
    #: line and lag embedding tables with exactly one row per line and per lag, indexed
    #: from zero, as the companion's are, instead of one extra unused row each (indexed
    #: from one, as the R implementation's are). 64 fewer parameters and the same function;
    #: the visible-lag table keeps its twelve rows and is read at ``lk`` rather than
    #: ``lk + 1``, as the companion reads it
    compact_embeddings: bool = False

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

    @classmethod
    def accident_year_variant(cls, **overrides) -> TLRNConfig:
        """The companion study's accident-year variant, as one config.

        Every choice that variant differs on, named: the premium loss ratio head,
        attention across lines, lags and accident years reading only the cells the
        forecast date has revealed, whole-company batches of eight, a blend with the
        multivariate chain ladder, twenty members all averaged, and a calibration
        from the method retrained at valuation dates 6, 7 and 8 (so the errors it
        pools come from forecasts 4, 3 and 2 diagonals ahead). ``overrides`` replace
        any of them, which is how one choice is switched back to the published
        model to see what it was worth.

        Written from a description of that notebook, not from its code, so a number
        that differs from the notebook's is a question about this config before it
        is a question about the model.
        """
        choices = {
            "head": "premium_lr",
            "attention": ("line", "lag", "ay"),
            "mask": "observed_cells",
            "batch_unit": "company",
            "batch_size": 8,
            "member": "mcl_blend",
            "ensemble_size": 20,
            "keep": None,
            "scoring": "reached_cells",
            "calibration": "retrain_per_valuation",
            "calibration_cutoffs": (6, 7, 8),
            "calibration_horizons": (2, 3, 4),
        }
        return cls(**{**choices, **overrides})

    @classmethod
    def companion_final_variant(cls, **overrides) -> TLRNConfig:
        """The companion study's final model: the accident-year variant, refit, and alpha
        fitted on one diagonal.

        :meth:`accident_year_variant` plus the two protocol changes of the companion's
        newest code: ``refit=True`` (each member is retrained on every known diagonal and
        stopped at the epoch it chose) and ``alpha_on="first_diagonal"`` (alpha from the
        diagonal before the last known one, the checkpoint chosen on the last), plus the
        two network details that make ibnr's network the companion's function and size:
        ``empty_query_bias=True`` and ``compact_embeddings=True`` (18,537 parameters).
        Batches of
        eight companies, 3000 epochs, twenty members all averaged, learning rates 0.003 for
        the network and 0.03 for the premium head's ``beta``, and the calibration retrained
        at valuation dates 6, 7 and 8 are the variant's own and were checked against the
        companion's training code. ``overrides`` replace any choice.
        """
        choices = {
            "refit": True,
            "alpha_on": "first_diagonal",
            "empty_query_bias": True,
            "compact_embeddings": True,
        }
        return cls.accident_year_variant(**{**choices, **overrides})

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
        if self.alpha_on not in ALPHA_ON:
            raise ValueError(f"alpha_on must be one of {list(ALPHA_ON)}, got {self.alpha_on!r}")
        if self.alpha_on == "first_diagonal":
            if self.member != "mcl_blend":
                raise ValueError(
                    "alpha_on='first_diagonal' says where the blend weight alpha is fitted, and "
                    f"member={self.member!r} has no alpha. Set member='mcl_blend' or "
                    "alpha_on='all_validation'"
                )
            if self.val_diagonals != 2:
                raise ValueError(
                    "alpha_on='first_diagonal' fits alpha on the first validation diagonal and "
                    "scores the blend on the second, so it needs exactly val_diagonals=2, got "
                    f"{self.val_diagonals}"
                )
        if not isinstance(self.refit, bool):
            raise ValueError(f"refit must be True or False, got {self.refit!r}")
        for name in ("empty_query_bias", "compact_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, got {getattr(self, name)!r}")
        if self.empty_query_bias and not (
            self.mask == "observed_cells" or "ay" in attention_axes(self)
        ):
            raise ValueError(
                "empty_query_bias says what an attention returns for a query with no key to "
                "read, and with mask='unwritten_lines' and no 'ay' axis no attention ever has "
                "such a query (an example with no written line is refused). Set "
                "mask='observed_cells' or add 'ay' to attention, or leave "
                "empty_query_bias=False"
            )
        if not self.lr_cap > 0:
            raise ValueError(f"lr_cap must be positive, got {self.lr_cap}")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} must be a positive multiple of n_heads {self.n_heads}: "
                "attention splits the token width evenly over the heads"
            )
