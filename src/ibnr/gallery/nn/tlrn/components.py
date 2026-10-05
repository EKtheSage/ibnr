"""The named design choices of ``tlrn``, and which of them can be combined.

Torch-free on purpose, like ``config.py``: the entry registers and its config
validates without the [nn] extra.

A ``tlrn`` is five choices made independently, each with a short list of names:

``head``
    What the network's output is read as, and so how a cell forecast is built.
    ``ldf`` is the published model: a log development factor per (line, step),
    projected from the latest cumulative. ``premium_lr`` reads the output as an
    incremental loss ratio per (line, lag) and multiplies by premium, so the
    forecast scales with exposure instead of with the latest paid.
``attention``
    Which axes of the (company, accident year, line, lag) grid a block attends
    along, in the order it applies them. ``("line", "lag")`` is the published
    model; adding ``"ay"`` lets an accident year read the same cells of the
    company's other accident years.
``mask``
    What an attention may read. ``unwritten_lines`` masks only the lines a
    company does not write; ``observed_cells`` masks every cell the forecast
    date has not revealed.
``batch_unit``
    What ``batch_size`` counts. ``example`` is one (company, accident year);
    ``company`` is one company with all its accident years together, which an
    ``ay`` attention needs because it attends across them.
``calibration``
    Where the historical errors that become the predictive distribution come
    from. ``rescore_final`` applies the final models at earlier cutoffs, so those
    forecasts are of cells the models trained on. ``retrain_per_valuation``
    retrains the whole method from scratch at each calibration cutoff on what was
    known then, so its errors are out of sample, at the price of a full fit each.
``member`` and ``keep``
    How a trained network becomes a member's forecast and how many members are
    averaged. ``network`` is the network's forecast alone; ``mcl_blend`` is the
    multivariate chain ladder plus a weight alpha times the network's difference
    from it, alpha fitted at every validation check and kept with the checkpoint
    that validated best. ``keep`` members are averaged, or every trained member
    when it is ``None``.

WHY THE COMBINATIONS ARE CHECKED HERE. The choices are not independent: ``ay``
attention reads across accident years so it cannot run on a batch of unrelated
examples; the anchored variant measures its correction from chain ladder factors
that only the ``ldf`` head has. A flat set of switches would let a caller build
every one of those wrong combinations and learn that from a shape error deep in
a forward pass. :func:`check_combination` refuses them at construction, naming
the two choices that disagree and the one change that resolves it.

``tests/test_tlrn_components.py`` runs its contract checks over every name in
these tuples, so a choice cannot be added here without being held to them.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ATTENTION_AXES",
    "BATCH_UNITS",
    "CALIBRATIONS",
    "HEADS",
    "MASKS",
    "MEMBERS",
    "attention_axes",
    "check_combination",
]

#: what the network output is read as
HEADS: tuple[str, ...] = ("ldf", "premium_lr")
#: the axes a block may attend along, in the order they are applied
ATTENTION_AXES: tuple[str, ...] = ("line", "lag", "ay")
#: what an attention may read
MASKS: tuple[str, ...] = ("unwritten_lines", "observed_cells")
#: how a trained network becomes a member's forecast
MEMBERS: tuple[str, ...] = ("network", "mcl_blend")
#: where the calibration errors come from
CALIBRATIONS: tuple[str, ...] = ("rescore_final", "retrain_per_valuation")
#: what ``batch_size`` counts
BATCH_UNITS: tuple[str, ...] = ("example", "company")


def attention_axes(cfg: Any) -> tuple[str, ...]:
    """The axes ``cfg`` attends along: ``attention``, else what ``cross_line`` implies."""
    if cfg.attention is not None:
        return tuple(cfg.attention)
    return ("line", "lag") if cfg.cross_line else ("lag",)


def check_combination(cfg: Any) -> None:
    """Refuse a combination of choices that cannot work, naming what disagrees."""
    for name, value, allowed in (
        ("head", cfg.head, HEADS),
        ("mask", cfg.mask, MASKS),
        ("member", cfg.member, MEMBERS),
        ("calibration", cfg.calibration, CALIBRATIONS),
        ("batch_unit", cfg.batch_unit, BATCH_UNITS),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")

    if cfg.attention is not None:
        axes = tuple(cfg.attention)
        unknown = [a for a in axes if a not in ATTENTION_AXES]
        ordered = [a for a in ATTENTION_AXES if a in axes]
        if unknown or list(axes) != ordered or len(set(axes)) != len(axes):
            raise ValueError(
                f"attention must list axes from {list(ATTENTION_AXES)} once each, in that "
                f"order, got {list(axes)}"
            )
        if "lag" not in axes:
            raise ValueError(
                "attention must include 'lag': the lag attention is the one axis every "
                f"variant of this network has, got {list(axes)}"
            )
        if cfg.cross_line != ("line" in axes):
            raise ValueError(
                f"cross_line={cfg.cross_line} disagrees with attention={list(axes)}: the two "
                "name the same thing. Set attention alone, and leave cross_line at its default"
            )
    axes = attention_axes(cfg)

    if "ay" in axes and cfg.batch_unit != "company":
        raise ValueError(
            "attention includes 'ay', which attends across a company's accident years, so a "
            "batch must hold whole companies: set batch_unit='company'. A batch of "
            "(company, accident year) examples drawn at random would put accident years of "
            "different companies in one attention sequence"
        )
    if cfg.head == "premium_lr" and cfg.cl_anchor:
        raise ValueError(
            "cl_anchor measures a bounded correction from the company's own chain ladder "
            "development factors, which only head='ldf' has; head='premium_lr' forecasts "
            "from premium. Set cl_anchor=False or head='ldf'"
        )
