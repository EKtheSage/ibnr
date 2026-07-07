import subprocess
import sys

import pytest

from ibnr import gallery
from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register


def test_meyers_ccl_registered():
    assert "meyers_ccl" in gallery.list()
    cls = gallery.get("meyers_ccl")
    assert cls.family == "bayesian"


def test_all_families_registered():
    assert {"meyers_ccl", "sur", "copula_glm", "nn_transformer"} <= set(gallery.list())
    assert gallery.get("nn_transformer").family == "nn"


def test_gallery_import_does_not_require_torch():
    # registration must work without the [nn] extra: importing the gallery
    # may not pull torch in (lazy imports inside fit/predict only)
    code = (
        "import sys; import ibnr.gallery as g; "
        "assert 'nn_transformer' in g.list(); "
        "assert 'torch' not in sys.modules, 'gallery import pulled in torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_card_is_real():
    card = gallery.get("meyers_ccl").card()
    assert "Correlated Chain Ladder" in card
    assert "logelr" in card
    assert "Parameterization" in card  # parity demands documented parameterization


def test_unknown_entry():
    with pytest.raises(KeyError, match="no gallery entry"):
        gallery.get("nope")


def test_register_rejects_incomplete_entries():
    class NotAnEntry:
        pass

    with pytest.raises(TypeError, match="not a GalleryEntry"):
        register(NotAnEntry)

    class StillAbstract(GalleryEntry):
        name = "x"
        family = "bayesian"

        def fit(self, triangle, **kw):
            return self

        # predict missing -> abstract

    with pytest.raises(TypeError, match="full GalleryEntry interface"):
        register(StillAbstract)

    class NoCard(GalleryEntry):
        name = "y"
        family = "bayesian"

        def fit(self, triangle, **kw):
            return self

        def predict(self):
            raise NotImplementedError

    with pytest.raises(TypeError, match="card.md"):
        register(NoCard)
