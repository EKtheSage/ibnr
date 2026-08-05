"""``nn_paid_case``: the joint paid + case-reserve NN entry.

Per cell it predicts (paid increment, case movement) from ONE bivariate mixture
(``head.py``), samples both, and rolls both forward - so the case reserve is a
simulated state rather than a frozen input channel, and the frozen-feature
limitation every other NN entry discloses does not apply. Two ablatable
backbones (``network_transformer.py``, ``network_gru.py``) over that one head.

Torch-free at module scope, like every other entry package: ``head.py`` and the
two network modules import torch (an ``nn.Module`` cannot be written without it)
and are imported from inside fit/predict only, so ``ibnr.gallery`` registers this
entry without the ``[nn]`` extra - subprocess-tested by
``tests/test_import_purity.py`` and ``tests/test_gallery.py``.
"""

from ibnr.gallery.nn.nn_paid_case.config import NNPaidCaseConfig
from ibnr.gallery.nn.nn_paid_case.model import NNPaidCase

__all__ = ["NNPaidCase", "NNPaidCaseConfig"]
