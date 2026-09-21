"""``tlrn``: the transformer loss reserving network.

Reproduces the companion study's R model on ibnr's data contract. A company at
one accident year is one training example whose tokens are its (line,
development lag) cells; axial attention runs across lines within a lag and
across lags within a line; and the output is not a cell but a positive log
development factor per (line, step), from which the cells follow by projecting
each origin's cumulative forward. Trained under the study's checkpoint protocol
- one cutoff per epoch, validation on held-out diagonals, ten seeds of which the
best two are kept - and given a company-level predictive distribution by
historical residual calibration rather than by a distributional head.

``TLRN`` is the gallery entry and ``network.TLRNNetwork`` the module it builds,
the same split every NN entry here uses.

Torch-free at module scope, like every other entry package: ``head.py`` and
``network.py`` import torch and are imported from inside fit and predict only,
so ``ibnr.gallery`` registers this entry without the ``[nn]`` extra -
subprocess-tested by ``tests/test_import_purity.py`` and ``tests/test_gallery.py``.
"""

from ibnr.gallery.nn.tlrn.config import TLRNConfig

__all__ = ["TLRNConfig"]
