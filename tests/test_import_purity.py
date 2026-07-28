"""Importing ibnr must not drag an optional extra in with it.

Two separate claims, and they fail for different reasons:

1. **The public import paths stay light.** ``import ibnr.gallery`` registers
   every nn and bayesian entry, so a single module-level ``import torch`` (or
   pymc, or cmdstanpy) in any entry makes the core install unusable for
   everyone. This generalises the two hand-written subprocess guards that
   already lived in ``test_gallery.py`` (torch) and ``test_stacking.py``
   (bayesblend) - the same defect in a third extra had nothing watching it.

2. **Every module in the package imports under a core-only install**, except a
   declared allowlist. The check above only sees modules that some collected
   test happens to import; this one walks the package, which is what makes
   "catch hidden imports" literally true rather than approximately true.

Both run in a **subprocess**. In an environment that has the extras installed -
the `all` CI leg, and every dev box - torch is already in this process's
``sys.modules`` from an earlier test, which would mask the violation entirely. A
clean interpreter is the only honest check, and it is also why these tests have
teeth in the leg where the extras ARE present rather than the leg where they are
absent.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Every optional dependency, direct or transitive, that must stay out of the
#: public import paths. jax/pytensor/matplotlib are here because they are what
#: numpyro/pymc/arviz cost in practice - naming only the direct dependency would
#: miss an entry that imports pytensor to build a graph at module scope.
HEAVY = (
    "torch",
    "pymc",
    "numpyro",
    "arviz",
    "bayesblend",
    "cmdstanpy",
    "chainladder",
    "bermuda",
    "altair",
    "jax",
    "pytensor",
    "matplotlib",
)

#: The modules a user reaches for. ``ibnr.kernels.stacking`` and
#: ``ibnr.kernels.harness`` are named separately from ``ibnr.gallery`` because
#: neither is imported by it - a regression in either would otherwise be
#: invisible until someone called ``gallery.stack()``.
PUBLIC_IMPORTS = (
    "ibnr",
    "ibnr.gallery",
    "ibnr.kernels.stacking",
    "ibnr.kernels.harness",
)

#: Modules that legitimately require an extra, checked as an exact set rather
#: than a prefix. These are the nn entries' pytorch ``nn.Module`` definitions,
#: which cannot be written without torch at module scope; their siblings
#: (``model.py``, ``config.py``) import torch lazily inside fit/predict, which is
#: what keeps ``ibnr.gallery`` importable. A NEW name appearing here is a design
#: decision - adding it should be a deliberate edit, not a silent one.
NEEDS_NN_EXTRA = frozenset(
    {
        "ibnr.gallery.nn.deeptriangle.network",
        "ibnr.gallery.nn.mdn.network",
        "ibnr.gallery.nn.resnet.network",
        "ibnr.gallery.nn.transformer.network",
        "ibnr.gallery.nn.transformer_ml.network",
    }
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """A clean interpreter, with the child's own assertion text surfaced.

    ``check=True`` alone reports "exit status 1" and throws the traceback away,
    which for an import-purity failure is precisely the information needed.
    """
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    return proc


@pytest.mark.parametrize("target", PUBLIC_IMPORTS)
def test_public_import_pulls_in_no_optional_extra(target):
    """Mutation: move ``import torch`` to the top of any nn ``model.py``; the
    ``ibnr.gallery`` case fails naming torch."""
    code = (
        "import sys\n"
        f"import {target}\n"
        f"leaked = [m for m in {HEAVY!r} if m in sys.modules]\n"
        f"assert not leaked, 'importing {target} pulled in ' + repr(leaked)\n"
    )
    _run(code)


def test_the_heavy_list_is_not_vacuous():
    """Guard the guard: if none of HEAVY is installed in this environment, the
    test above passes for free and proves nothing. Skipping here rather than
    failing is correct - the isolated CI legs deliberately have most of these
    absent, and the `all` leg is where the claim is actually tested."""
    import importlib.util

    present = [m for m in HEAVY if importlib.util.find_spec(m) is not None]
    if not present:
        pytest.skip("no optional extra installed; the purity check is vacuous here")
    assert present


#: Child program: make every optional extra unimportable, then import every
#: submodule of the package and report which ones could not be imported.
#:
#: The blocker is what lets this test run in EVERY CI leg instead of only the
#: core one. Skipping it wherever torch happens to be installed would disarm it
#: on the `all` leg - the leg that installs everything and therefore the one
#: where a new module-level ``import pymc`` is most likely to be written and
#: least likely to be noticed. It also keeps the test off the `all` leg's
#: strict-skip allowlist, which exists for facts about the runner and should not
#: grow entries describing our own test suite.
_WALK = """
import importlib, json, pkgutil, sys

BLOCKED = set({blocked!r})


class _NoExtras:
    "Refuses the optional extras the way a core-only install refuses them."

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name.split('.')[0]!r}}", name=name)
        return None


sys.meta_path.insert(0, _NoExtras())

import ibnr

failed = {{}}
for mod in sorted(m.name for m in pkgutil.walk_packages(ibnr.__path__, "ibnr.")):
    try:
        importlib.import_module(mod)
    except Exception as exc:
        failed[mod] = f"{{type(exc).__name__}}: {{exc}}"
print(json.dumps(failed))
"""


def test_every_submodule_imports_without_any_extra():
    """The whole package, not just what the collected tests happen to touch.

    Compares the failures to ``NEEDS_NN_EXTRA`` as a SET - a module dropping OUT
    of the allowlist is as much a finding as one joining it, because it means the
    allowlist has gone stale and is no longer describing the package.
    """
    import json

    failed = json.loads(_run(_WALK.format(blocked=sorted(HEAVY))).stdout.strip().splitlines()[-1])

    assert set(failed) == set(NEEDS_NN_EXTRA), (
        "the set of modules needing an extra changed.\n"
        f"  unexpectedly failing: {sorted(set(failed) - NEEDS_NN_EXTRA)}\n"
        f"  no longer failing:    {sorted(NEEDS_NN_EXTRA - set(failed))}\n"
        f"  reasons: {failed}"
    )
