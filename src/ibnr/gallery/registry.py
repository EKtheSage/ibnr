"""The gallery registry. Entries self-register at import via @register."""

from __future__ import annotations

import inspect
from pathlib import Path

from ibnr.gallery.entry import GalleryEntry

_REGISTRY: dict[str, type[GalleryEntry]] = {}


def register(cls: type[GalleryEntry]) -> type[GalleryEntry]:
    """Register a gallery entry after checking it honors the contract."""
    if not (inspect.isclass(cls) and issubclass(cls, GalleryEntry)):
        raise TypeError(f"{cls!r} is not a GalleryEntry subclass")
    if inspect.isabstract(cls):
        raise TypeError(f"{cls.__name__} does not implement the full GalleryEntry interface")
    for attr in ("name", "family"):
        if not isinstance(getattr(cls, attr, None), str):
            raise TypeError(f"{cls.__name__} must define a class-level string {attr!r}")
    if not (Path(inspect.getfile(cls)).parent / "card.md").exists():
        raise TypeError(f"{cls.__name__} has no card.md; every gallery entry ships its card")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"gallery entry name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def entries() -> dict[str, type[GalleryEntry]]:
    return dict(_REGISTRY)


def get(name: str) -> type[GalleryEntry]:
    """The registered entry **class** - not an instance, not a fitted model.

    So a caller fits with ``gallery.get("mack")().fit(tri)``, or with
    :func:`fit`, which is that expression. Returning the class is what lets
    ``.card()`` and ``.family`` be read without constructing anything, which the
    leaderboard and the docs site both do.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no gallery entry named {name!r}; known: {sorted(_REGISTRY)}") from None


def fit(name: str, triangle, **kwargs) -> GalleryEntry:
    return get(name)().fit(triangle, **kwargs)
