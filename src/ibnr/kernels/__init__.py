"""Evaluation, contract and wire-format kernels.

Every name here is imported eagerly, including the codec's - which looks like a
cold-start defect and is not. ``kernels/codec.py`` imports pyarrow at module
scope and this package sits on the import path of every gallery entry
(``ibnr.kernels.predictive`` runs this file first), so deferring the codec
behind a PEP 562 ``__getattr__`` reads as an obvious win. It was built and
measured, and it buys nothing: **pandas imports pyarrow itself** - a bare
``import pyarrow`` in ``pandas/compat/pyarrow.py``, executed whenever pyarrow is
installed, and it always is because the codec requires it - so pyarrow is in
``sys.modules`` before ibnr's first line either way. Interleaved A/B on this box
(11 alternating reps per arm, warm bytecode, fresh interpreter each rep):

    from ibnr import gallery    eager 3.05 s   lazy 3.20 s   (2.8-3.5 s spread)

The lazy arm removes exactly ONE module from a 1,098-module process and no
measurable time, and ``scipy.special`` - the other heavy import on this path -
arrives through ``kernels.densities`` regardless. So this file stays plain.
``tests/test_import_purity.py::test_the_wire_format_is_not_what_a_public_import_pays_for``
is the tripwire on the premise rather than on the fix: it goes red if pandas
ever stops importing pyarrow, which is when the deferral would start to pay.
"""

from ibnr.kernels.cdr import CDRResult, cdr_risk_measures, one_year_cdr, simulate_one_year_cdr
from ibnr.kernels.codec import (
    CODEC_VERSION,
    CONTENT_TYPE_ARROW,
    CONTENT_TYPE_JSON,
    DEFAULT_QUANTILES,
    from_arrow,
    peek_kind,
    to_arrow,
    to_summary,
)
from ibnr.kernels.densities import to_amount_scale
from ibnr.kernels.holdout import HoldoutCells, next_diagonal
from ibnr.kernels.mack import MackFit, fit_mack, simulate_ultimates
from ibnr.kernels.parity import ParityReport, compare_posteriors
from ibnr.kernels.predictive import PredictiveDistribution

__all__ = [
    "CDRResult",
    "CODEC_VERSION",
    "CONTENT_TYPE_ARROW",
    "CONTENT_TYPE_JSON",
    "DEFAULT_QUANTILES",
    "HoldoutCells",
    "MackFit",
    "ParityReport",
    "PredictiveDistribution",
    "cdr_risk_measures",
    "compare_posteriors",
    "fit_mack",
    "from_arrow",
    "next_diagonal",
    "one_year_cdr",
    "peek_kind",
    "simulate_one_year_cdr",
    "simulate_ultimates",
    "to_amount_scale",
    "to_arrow",
    "to_summary",
]
