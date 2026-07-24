"""Guards the gallery contract: registration, the GalleryEntry ABC, and the
no-torch-at-import rule.

Three invariants live here:

1. **Entries self-register on import.** ``ibnr.gallery`` must expose every
   family (bayesian / nn / statistical / deterministic) through
   ``gallery.list()`` without the caller importing model modules by hand.
2. **Torch is never imported at module level** (CLAUDE.md, "Tooling &
   conventions"). The core install has no ``[nn]`` extra, so a stray
   top-level ``import torch`` in an nn entry would break plain
   ``import ibnr.gallery`` for everyone. Only a subprocess can prove this -
   see ``test_gallery_import_does_not_require_torch``.
3. **The GalleryEntry contract is enforced at registration time, not at
   fit time** (design decision 5: "an entry that fails the eval harness does
   not register"). ``register()`` rejects non-subclasses, still-abstract
   classes, missing ``name``/``family``, and a missing ``card.md``.

All fast - no cmdstan, no mart, no sampling.
"""

import subprocess
import sys

import pytest

from ibnr import gallery
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register


def test_meyers_ccl_registered():
    """The reference Bayesian entry is reachable by name and tagged bayesian."""
    assert "meyers_ccl" in gallery.list()
    cls = gallery.get("meyers_ccl")
    assert cls.family == "bayesian"


def test_all_families_registered():
    """One representative per family registers on plain `import ibnr.gallery`.

    Subset (``<=``) rather than equality on purpose: the gallery grows, and this
    test should not need editing every time an entry lands.
    """
    assert {"meyers_ccl", "sur", "copula_glm", "nn_transformer", "mack"} <= set(gallery.list())
    assert gallery.get("nn_transformer").family == "nn"
    assert gallery.get("mack").family == "deterministic"


def test_gallery_import_does_not_require_torch():
    """Importing the gallery must not pull torch in, even though nn entries register.

    Runs in a **subprocess** because torch may already be in this process's
    ``sys.modules`` (another test, or the [nn] extra being installed here), which
    would mask the violation. A clean interpreter is the only honest check.
    ``check=True`` turns the in-child assertion failure into a test failure.
    """
    # registration must work without the [nn] extra: importing the gallery
    # may not pull torch in (lazy imports inside fit/predict only)
    code = (
        "import sys; import ibnr.gallery as g; "
        "assert 'nn_transformer' in g.list(); "
        "assert 'torch' not in sys.modules, 'gallery import pulled in torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_card_is_real():
    """`card()` returns the entry's actual card.md, not a stub.

    Checks three markers that only the real CCL card carries: the model's name,
    a parameter from the Stan model (``logelr``), and the Parameterization
    section that design decision 7 requires of every card (centered vs
    non-centered etc. must be documented before cross-backend parity means
    anything).
    """
    card = gallery.get("meyers_ccl").card()
    assert "Correlated Chain Ladder" in card
    assert "logelr" in card
    assert "Parameterization" in card  # parity demands documented parameterization


def test_unknown_entry():
    """A typo'd entry name fails loudly (and the error lists what is known)."""
    with pytest.raises(KeyError, match="no gallery entry"):
        gallery.get("nope")


def test_register_rejects_incomplete_entries():
    """register() enforces the GalleryEntry contract at import time.

    Three rejection paths, in the order registry.register() checks them.
    """

    # 1. not a GalleryEntry subclass at all
    class NotAnEntry:
        pass

    with pytest.raises(TypeError, match="not a GalleryEntry"):
        register(NotAnEntry)

    # 2. subclass but still abstract: predict() unimplemented, so the entry
    #    could never produce a PredictiveDistribution (design decision 4)
    class StillAbstract(GalleryEntry):
        name = "x"
        family = "bayesian"

        def fit(self, triangle, **kw):
            return self

        # predict missing -> abstract

    with pytest.raises(TypeError, match="full GalleryEntry interface"):
        register(StillAbstract)

    # 3. complete interface but no card.md next to the module: every entry
    #    ships its card (the gallery is documentation-first)
    class NoCard(GalleryEntry):
        name = "y"
        family = "bayesian"

        def fit(self, triangle, **kw):
            return self

        def predict(self):
            raise NotImplementedError

    with pytest.raises(TypeError, match="card.md"):
        register(NoCard)
