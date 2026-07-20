"""Windows C++ toolchain wiring shared by the Bayesian backends.

Both cmdstan (via cmdstanpy) and PyMC (via PyTensor) need a C/C++ compiler on
Windows. Modern RTools installs (43/44/45) ship a plain ``make`` and a g++ under
``x86_64-w64-mingw32.static.posix``; these helpers put them on PATH so the
backends can compile. No-ops on non-Windows or when a toolchain is already set.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

# RTools roots newest-first; each carries a posix g++ and a usr/bin make.
_RTOOLS_ROOTS = ("C:/rtools45", "C:/rtools44", "C:/rtools43", "C:/rtools40")


def _rtools_bindirs() -> tuple[Path, Path] | None:
    for root in _RTOOLS_ROOTS:
        gxx = Path(root) / "x86_64-w64-mingw32.static.posix" / "bin"
        mk = Path(root) / "usr" / "bin"
        if gxx.exists() and mk.exists():
            return gxx, mk
    return None


def ensure_stan_toolchain() -> None:
    """Best-effort: put an RTools g++/make on PATH so cmdstan can compile.

    cmdstanpy assumes mingw32-make/RTools40; modern RTools ship plain ``make``,
    so we also set ``MAKE``.
    """
    if platform.system() != "Windows":
        return
    if shutil.which("g++") and (shutil.which("mingw32-make") or shutil.which("make")):
        os.environ.setdefault("MAKE", "mingw32-make" if shutil.which("mingw32-make") else "make")
        return
    dirs = _rtools_bindirs()
    if dirs is not None:
        gxx, mk = dirs
        os.environ["PATH"] = f"{gxx};{mk};{os.environ['PATH']}"
        os.environ.setdefault("MAKE", "make")


def ensure_pytensor_cxx() -> str | None:
    """Point PyTensor at an RTools g++ so it compiles C ops instead of falling
    back to the (very slow) Python evaluation path.

    Returns the g++ path configured, or None if nothing was needed/found.
    Idempotent: a no-op once ``pytensor.config.cxx`` is already set.
    """
    if platform.system() != "Windows":
        return None
    import pytensor

    if pytensor.config.cxx:
        return pytensor.config.cxx
    dirs = _rtools_bindirs()
    if dirs is None:
        return None
    gxx, mk = dirs
    gxx_exe = gxx / "g++.exe"
    if not gxx_exe.exists():
        return None
    if str(gxx) not in os.environ["PATH"]:
        os.environ["PATH"] = f"{gxx};{mk};{os.environ['PATH']}"
    pytensor.config.cxx = str(gxx_exe)
    return str(gxx_exe)
