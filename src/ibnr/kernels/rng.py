"""One study seed, a different stream of random numbers per cohort.

**The defect this exists to fix.** A gallery entry is fitted to one cohort, so a
study over twenty-five companies is twenty-five separate fits, and a script that
wants reproducible results passes all of them the same seed - which is exactly
what the notebooks in ``analysis/`` do. Every ``predict`` and the shared
``predict_at`` used to answer that seed with ``np.random.default_rng(seed)``,
starting the same generator from scratch each time. Draw ``i`` of every cohort
then read the same underlying random numbers, so the cohorts' simulated
ultimates rose and fell together. Notebook 3c measured the consequence for
``mack``: a mean implied cross-cohort correlation of 0.255, where independent
draws at 10,000 draws would sit near 0.01 of sampling noise.

Each cohort's own distribution was never affected - its mean, its spread and its
quantiles were all correct. What was affected is every quantity read across
cohorts *within a draw*: a company total, a panel total, the spread of either,
and any calibration statistic computed from those sums. Those are exactly the
numbers a capital figure is made of, so the artifact mattered where it was
hardest to see.

**Why the fix sits here and not in the kernel functions.** The obvious place to
put it would be ``kernels.mack.simulate_ultimates`` and friends, since they are
what actually calls ``default_rng``. That is closed off deliberately. Those
functions' behavior for a plain integer seed is pinned byte for byte by
``tests/test_cdr.py``, and the Merz-Wuthrich tie-out against R's ``ChainLadder``
package rests on the same construction. A kernel function is a mathematical
routine that takes whatever random source it is handed; deciding that one seed
means twenty-five different streams is a study-level policy, and the gallery is
where a study meets a fitted cohort. So the kernels are untouched and every
entry derives its stream before calling them.

**What the derivation is.** The caller's integer is combined with a short text
that names what is being drawn - the method's label, the cohort's segment
identity, the loss field, the training cutoff - and the pair becomes a
``numpy.random.SeedSequence``. The text is hashed with sha256 and the first
thirty-two bytes of the digest are read as eight 32-bit numbers, little-endian,
which follow the integer into the sequence's entropy.

Two properties follow, and both are the point:

* **The same cohort and the same seed always give the same draws**, in any
  process, on any machine. sha256 is fixed by its specification and the byte
  order is written out explicitly rather than inherited from the platform.
  Python's own ``hash()`` would have been shorter and is unusable here: it is
  salted differently in every process, so a rerun of the same script would
  produce different numbers.
* **Two different cohorts get streams with no relationship to each other**,
  because ``SeedSequence`` is built to turn nearby entropy into distant states.

**The passthrough rule.** Only a bare integer is treated as a study seed. A
``Generator``, ``BitGenerator`` or ``SeedSequence`` handed in as ``seed`` is
returned unchanged: the caller has already chosen a stream, and a caller who
went to that trouble means the stream they built, not one derived from it.
``seed=None`` returns ``None``, so ``default_rng(None)`` downstream still draws
fresh entropy from the operating system.

Deriving a stream cannot be skipped by an entry that forgets to: the held-out
path goes through one shared method on ``PredictsHeldout``, and the run-off path
is one line in each entry's ``predict``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ibnr.kernels.holdout import HoldoutCells

__all__ = ["cohort_stream", "heldout_stream"]

#: separates the top-level parts of the canonical text (label, field, cutoff,
#: then one piece per cohort)
_PART_SEP = "\x1e"
#: separates the ``key=value`` pieces within one cohort
_PAIR_SEP = "\x1f"
#: how many 32-bit words of the sha256 digest follow the seed into the entropy
_N_WORDS = 8


def cohort_stream(
    seed: Any,
    *,
    label: str,
    cohorts: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    field: str | None = None,
    as_of: dt.date | str | None = None,
) -> Any:
    """A stream of random numbers belonging to one cohort and one drawing method.

    ``seed`` is the study-level seed a script passes to every fit. ``label``
    says what is being drawn (``"predict"``, ``"predict_at"``,
    ``"cdr_distribution"``), so one fit answering one seed still keeps its
    run-off draws, its held-out draws and its claims development result on
    separate streams. ``cohorts`` is the cohort's segment identity - one
    mapping, or an iterable of them for a fit covering several cohorts at once.
    ``field`` and ``as_of`` are the loss field being drawn and the training
    cutoff, both optional and both left out of the text when absent.

    Returns a ``SeedSequence`` for an integer seed, ``None`` for ``None``, and
    anything else exactly as it came in - see the module docstring for why.
    A negative integer raises, from ``SeedSequence`` itself, which is the same
    refusal ``np.random.default_rng(-1)`` gives.
    """
    if seed is None:
        return None
    # bool is listed for readability only; it is a subclass of int.
    if not isinstance(seed, (bool, int, np.integer)):
        return seed
    text = _canonical_text(label=label, cohorts=cohorts, field=field, as_of=as_of)
    return np.random.SeedSequence([int(seed), *_digest_words(text)])


def heldout_stream(seed: Any, cells: HoldoutCells, *, field: str | None = None) -> Any:
    """The stream ``PredictsHeldout.predict_at`` draws its held-out cells with.

    Public and importable so a test can rebuild the exact stream an entry used
    and compare draws against a kernel function called directly - which several
    tests in this repo do, and which is how the held-out path stays pinned to
    the kernel it wraps.

    The cohort identity comes from the cells' own segment columns, the cutoff
    from ``cells.as_of``, and the field either from ``field`` or from the cells
    when they carry exactly one. Cells are what both models in a comparison
    share, so two entries handed the same cells derive the same stream and stay
    comparable on common random numbers.
    """
    if not isinstance(cells, HoldoutCells):
        raise TypeError(
            "heldout_stream needs a HoldoutCells: the stream is keyed on the cells' "
            f"cohort, field and cutoff, and a bare CellIndex carries none of them; "
            f"got {type(cells).__name__}"
        )
    frame = cells.frame
    if cells.segments:
        cohorts: list[Mapping[str, Any]] = (
            frame[list(cells.segments)].drop_duplicates().to_dict("records")
        )
    else:
        # An unsegmented triangle is one cohort whose identity is the empty
        # mapping - the same thing kernels.contract records as its segment.
        cohorts = [{}]

    resolved = field
    if resolved is None:
        # Mirrors index_into's stance: one field or say so. The two resolutions
        # provably agree wherever a call succeeds, because index_into either
        # filters the frame to a single field or refuses outright.
        present = sorted(frame["field"].unique())
        if not present:
            raise ValueError(
                "these held-out cells carry no rows, so there is no field to key the "
                "stream on; pass field= if you meant to build a stream anyway"
            )
        if len(present) > 1:
            raise ValueError(
                f"held-out cells span fields {present}; pass field= to choose one of them"
            )
        resolved = present[0]

    return cohort_stream(
        seed, label="predict_at", cohorts=cohorts, field=resolved, as_of=cells.as_of
    )


def _canonical_text(
    *,
    label: str,
    cohorts: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    field: str | None,
    as_of: dt.date | str | None,
) -> str:
    """The text whose hash separates one cohort's stream from another's.

    Everything is sorted before joining, so neither the order the cohorts were
    listed in nor the order a dictionary happens to iterate its keys can change
    the answer. Absent parts are left out rather than written as an empty
    string, which keeps ``field=None`` from colliding with ``field=""``.
    """
    parts = [str(label)]
    if field is not None:
        parts.append(str(field))
    if as_of is not None:
        parts.append(str(as_of))
    parts.extend(sorted(_cohort_text(c) for c in _as_cohort_list(cohorts)))
    return _PART_SEP.join(parts)


def _as_cohort_list(
    cohorts: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """One mapping means one cohort; anything else is iterated.

    The Mapping test has to come first. A dict is iterable, and iterating it
    yields its keys, so a single cohort passed on its own would otherwise be
    read as one cohort per segment column - silently, and with a stream that
    changes whenever a column is added.
    """
    if isinstance(cohorts, Mapping):
        return [cohorts]
    return list(cohorts)


def _cohort_text(cohort: Mapping[str, Any]) -> str:
    """One cohort's identity as ``key=value`` pieces, sorted by key."""
    if not isinstance(cohort, Mapping):
        raise TypeError(
            f"each cohort must be a mapping of segment column to value, got {type(cohort).__name__}"
        )
    pairs = sorted((str(k), str(v)) for k, v in cohort.items())
    return _PAIR_SEP.join(f"{k}={v}" for k, v in pairs)


def _digest_words(text: str) -> list[int]:
    """Eight 32-bit numbers from the sha256 digest of ``text``.

    The byte order is written out rather than left to the platform: a stream
    that differed between a laptop and a container would make a published study
    irreproducible on the machine that did not run it.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 4 * _N_WORDS, 4)]
