"""``nn_paid_case``: the joint paid + case-reserve entry (in construction).

Stage 1 ships the shared head only (``head.py``). The entry class, its config
and the two backbones land in stage 2, at which point this module re-exports
them the way every other nn entry does.

Deliberately EMPTY for now, and in particular free of any torch import: the
package is walked by ``tests/test_import_purity.py`` under a core-only install,
and ``ibnr.gallery`` must register every entry without the ``[nn]`` extra.
``head.py`` does import torch at module scope - like every other entry's
network module, because an ``nn.Module`` cannot be written without it - so it
is imported from inside fit/predict, never from here."""
