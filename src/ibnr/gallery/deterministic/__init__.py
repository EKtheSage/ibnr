"""Deterministic gallery entries: classical methods whose point estimate is
exact and whose uncertainty is analytic and/or simulated, not sampled."""

from ibnr.gallery.deterministic import mack  # noqa: F401
from ibnr.gallery.deterministic.mack.model import Mack

__all__ = ["Mack"]
