"""Guards the gallery contract: registration, the GalleryEntry ABC, and the
rules about what may not be imported at module level.

Four invariants live here:

1. **Entries self-register on import.** ``ibnr.gallery`` must expose every
   family (bayesian / nn / statistical / deterministic) through
   ``gallery.list()`` without the caller importing model modules by hand.
2. **Torch is never imported at module level** (CLAUDE.md, "Tooling &
   conventions"). The core install has no ``[nn]`` extra, so a stray
   top-level ``import torch`` in an nn entry would break plain
   ``import ibnr.gallery`` for everyone. Only a subprocess can prove this -
   see ``test_gallery_import_does_not_require_torch``.
3. **Neither ``import ibnr`` nor importing the gallery pulls in
   ``scipy.integrate`` or ``scipy.optimize``.** Each serves one narrow path
   and each costs about a second, which a downstream Azure Function pays on
   every cold start - see
   ``test_import_does_not_pull_heavy_scipy_subpackages``.
4. **The GalleryEntry contract is enforced at registration time, not at
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


@pytest.mark.parametrize("stmt", ["import ibnr", "from ibnr import gallery"])
def test_import_does_not_pull_heavy_scipy_subpackages(stmt):
    """Neither import may drag ``scipy.integrate`` or ``scipy.optimize`` in.

    Both serve exactly one narrow path each - ``check_normalization`` (a
    test-time guard rail) and Clark's MLE - and between them they cost about a
    second of import time, ``scipy.integrate`` largely because its ``_bvp``
    submodule imports ``scipy.optimize`` and ``scipy.sparse.linalg``. A
    downstream Azure Function pays that on every cold start against a fixed
    30 s app-init timeout, so the imports live at their point of use and this
    test is what keeps them there: the natural edit - hoisting one back to
    module scope for tidiness - is silent otherwise.

    ``scipy.special`` is deliberately NOT in the list. It is unavoidable
    (``gammaln`` for ``odp_lpdf``, ``logsumexp`` in ``forecast.py``), so
    asserting on it would fail for a reason nobody can act on.

    Subprocess for the same reason as the torch guard above: this process has
    almost certainly imported both already via pytest's own dependencies.
    """
    code = (
        f"import sys; {stmt}; "
        "leaked = [m for m in ('scipy.integrate', 'scipy.optimize') if m in sys.modules]; "
        f"assert not leaked, '{stmt} pulled in ' + repr(leaked)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


#: CLAUDE.md decision 8 pins this list verbatim; an addition is a deliberate
#: edit in both places, not something a convenience import does on its way past.
EXPECTED_EXPORTS = {
    "SCORE_DIRECTION",
    "Absence",
    "CohortForecast",
    "GalleryDiagonal",
    "GalleryEntry",
    "align_panel",
    "fit",
    "get",
    "leaderboard",
    "list",
    "next_diagonal",
    "stack",
}

#: exports the gallery DEFINES rather than re-exports from kernels, so
#: ``test_every_export_is_the_kernels_object_itself`` can insist that everything
#: else really is the kernels object and not pass vacuously on a name that
#: quietly stopped being one.
GALLERY_OWN_EXPORTS = {"GalleryDiagonal", "GalleryEntry", "fit", "get", "list"}


def test_gallery_exports_the_whole_leaderboard_pipeline():
    """``__all__`` is exactly the designed set - no more, no less.

    The four names beyond the 0.4.0 surface are the steps a caller must take to
    BUILD the panel ``leaderboard()`` consumes: which cells (``next_diagonal``),
    one model's arrays at them (``CohortForecast``), why an array is missing
    (``Absence``), and the cross-model intersection (``align_panel``).
    ``SCORE_DIRECTION`` joins them because the board has no default sort.

    ``GalleryDiagonal`` (0.5.1) is the one name admitted on decision 8's second
    clause - a fitted entry to a one-year CDR rather than to a board row. It has
    to be here: ``kernels.cdr`` cannot hand it out by name the way it hands out
    ``"mack"``, because it carries a fitted entry.
    """
    assert set(gallery.__all__) == EXPECTED_EXPORTS


@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS - GALLERY_OWN_EXPORTS))
def test_every_export_is_the_kernels_object_itself(name):
    """Identity, not equality: a re-export must not become a copy.

    ``leaderboard`` reading a different ``SCORE_DIRECTION`` than the caller
    sorted by would be invisible until the board ranked backwards.
    """
    from ibnr.kernels import forecast, holdout, stacking

    modules = [m for m in (forecast, holdout, stacking) if hasattr(m, name)]
    assert modules, (
        f"{name} is in EXPECTED_EXPORTS but lives in no kernels module; if the gallery "
        "now defines it, add it to GALLERY_OWN_EXPORTS deliberately"
    )
    assert getattr(gallery, name) is getattr(modules[0], name)


@pytest.mark.parametrize("name", sorted(GALLERY_OWN_EXPORTS))
def test_the_gallerys_own_exports_are_not_kernels_re_exports(name):
    """The other side of the split. A name listed as the gallery's own must not
    silently become a kernels re-export - that would mean ``kernels`` had grown
    an import of something in the gallery, or the two had diverged into a copy
    each."""
    from ibnr.kernels import cdr, forecast, holdout, stacking

    assert not any(hasattr(m, name) for m in (cdr, forecast, holdout, stacking))


def test_kernels_never_imports_the_gallery():
    """Re-export direction is gallery -> kernels, and only that way.

    The harness's spawn-based workers import ``kernels`` without the gallery, and
    a kernels->gallery edge would drag every entry (and every registration) into
    a process that needs one function.
    """
    code = (
        "import sys; import ibnr.kernels; "
        "assert 'ibnr.gallery' not in sys.modules, 'ibnr.kernels imported the gallery'"
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


class _Complete(GalleryEntry):
    """Everything the ABC demands, and nothing that would let it register."""

    name = "y"
    family = "bayesian"

    def fit(self, triangle, **kw):
        return self

    def cohorts(self):
        return [{}]

    def predict(self, segment=None, **kw):
        raise NotImplementedError

    def realized_ultimates(self, full_triangle, segment=None):
        raise NotImplementedError


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

    # 2b. `cohorts()` is abstract too, and deliberately not defaulted: a pooled
    #     entry that forgot to override a `contract_["segment"]` default would
    #     get a plausible answer from a key that happens to exist.
    class NoCohorts(GalleryEntry):
        name = "x2"
        family = "bayesian"

        def fit(self, triangle, **kw):
            return self

        def predict(self, segment=None, **kw):
            raise NotImplementedError

        def realized_ultimates(self, full_triangle, segment=None):
            raise NotImplementedError

    with pytest.raises(TypeError, match="full GalleryEntry interface"):
        register(NoCohorts)

    # 3. complete interface but no card.md next to the module: every entry
    #    ships its card (the gallery is documentation-first)
    with pytest.raises(TypeError, match="card.md"):
        register(_Complete)


def test_register_rejects_a_missing_config_class():
    """An entry whose fit() takes config= must say which type to build.

    Without the declaration, a caller who found the entry through
    ``gallery.get(name)`` has no route to its config class but a module path -
    which is exactly what notebook 03 had to do for four NN entries.
    """

    class TakesConfig(_Complete):
        name = "z"

        def fit(self, triangle, config=None, **kw):
            return self

    with pytest.raises(TypeError, match="declares no config_class"):
        register(TakesConfig)


def test_register_rejects_a_stale_config_class():
    """...and the other direction: a declaration nothing can reach.

    An entry that declares ``config_class`` while its ``fit`` takes no
    ``config=`` is exactly as misleading as one that declares none, so the
    check runs both ways.
    """

    class Stale(_Complete):
        name = "z2"
        config_class = dict

    with pytest.raises(TypeError, match="takes no config="):
        register(Stale)
