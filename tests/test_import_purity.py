"""Importing ibnr must not drag an optional extra in with it.

Three separate claims, and they fail for different reasons:

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

3. **``ibnr.methods`` loads neither ibis, pandas, scipy nor scikit-learn**, on
   import or when any of its four methods runs. These are core dependencies,
   not extras, so claim 1 has nothing to say about them; the cost is a cold
   start, which a service calling ``ibnr.methods`` pays on every new process.
   ``ibnr/__init__.py`` and ``ibnr/kernels/__init__.py`` import their names on
   first read to make this hold, and the tests below also check that those
   lazy imports hand out the same objects the eager ones did.

All three run in a **subprocess**. In an environment that has the extras installed -
the `all` CI leg, and every dev box - torch is already in this process's
``sys.modules`` from an earlier test, which would mask the violation entirely. A
clean interpreter is the only honest check, and it is also why these tests have
teeth in the leg where the extras ARE present rather than the leg where they are
absent.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import textwrap

import numpy as np
import pyarrow as pa
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
        "ibnr.gallery.nn.nn_paid_case.head",
        "ibnr.gallery.nn.nn_paid_case.network_gru",
        "ibnr.gallery.nn.nn_paid_case.network_transformer",
        "ibnr.gallery.nn.resnet.network",
        "ibnr.gallery.nn.tlrn.head",
        "ibnr.gallery.nn.tlrn.network",
        "ibnr.gallery.nn.transformer.network",
        "ibnr.gallery.nn.transformer_ml.network",
    }
)

#: Modules that need the [jax] extra: tlrn's JAX training backend, written in jax at
#: module scope as the networks above are written in torch, and imported by
#: ``TLRN.fit`` only when ``backend="jax"`` asks for it.
NEEDS_JAX_EXTRA = frozenset({"ibnr.gallery.nn.tlrn.jax_backend"})


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


#: What ``ibnr.methods`` must not load: not when it is imported, and not when a
#: method runs. polars is checked too wherever the caller has not imported it.
NOT_FOR_METHODS = ("ibis", "pandas", "scipy", "sklearn")

#: raa, the Mack (1993) triangle, as in tests/test_methods.py. Copied rather than
#: imported, because importing that module in a child process would load pandas.
RAA = [
    [5012, 8269, 10907, 11805, 13539, 16181, 18009, 18608, 18662, 18834],
    [106, 4285, 5396, 10666, 13782, 15599, 15496, 16169, 16704],
    [3410, 8992, 13873, 16141, 18735, 22214, 22863, 23466],
    [5655, 11555, 15766, 21266, 23425, 26083, 27067],
    [1092, 9565, 15836, 22169, 25955, 26180],
    [1513, 6445, 11702, 12935, 15852],
    [557, 4020, 10946, 12314],
    [1351, 6947, 13112],
    [3133, 5395],
    [2063],
]
#: raa's chain-ladder total ultimate and Mack total standard error, as in
#: tests/test_methods.py (chainladder-python 0.9.2's numbers).
RAA_TOTAL_ULTIMATE = 213122.22826121017
RAA_TOTAL_MACK_SE = 26880.74032989


def test_importing_methods_loads_no_ibis_pandas_scipy_or_sklearn():
    """Mutation: put ``import pandas as pd`` back at the top of
    ``kernels/conventional.py``, or import ``Triangle`` eagerly in
    ``ibnr/__init__.py``; this fails naming pandas, or ibis and pandas."""
    code = (
        "import sys\n"
        "from ibnr import methods\n"
        f"loaded = [m for m in {(*NOT_FOR_METHODS, 'polars')!r} if m in sys.modules]\n"
        "assert not loaded, 'from ibnr import methods loaded ' + repr(loaded)\n"
    )
    _run(code)


#: The four methods, each called once with its defaults and once with options,
#: and two refusals. Every call reads its cells from an Arrow file, as a service
#: receiving Arrow bytes would, because building a table from Python lists with
#: ``pa.table`` or ``pa.array`` makes pyarrow import pandas itself.
_METHOD_CALLS = """
import json, sys
import pyarrow.ipc as ipc
from ibnr import methods

cells = ipc.open_file(sys.argv[1]).read_all()
premium_table = ipc.open_file(sys.argv[2]).read_all()
premium = {{1981 + i: 20000.0 + 2000.0 * i for i in range(10)}}
NOT = {not_for!r}

calls = {{
    "chain_ladder": lambda: methods.chain_ladder(cells),
    "chain_ladder_options": lambda: methods.chain_ladder(
        cells, history_periods=3, drop_high=True, exclude=[(1982, 12)], zero_cells="observed"
    ),
    "bornhuetter_ferguson": lambda: methods.bornhuetter_ferguson(
        cells, premium=premium, expected_loss_ratio=0.7
    ),
    "bornhuetter_ferguson_table": lambda: methods.bornhuetter_ferguson(
        cells, premium=premium_table, expected_loss_ratio=0.7, average="median"
    ),
    "chain_ladder_selection": lambda: methods.chain_ladder(
        cells,
        average="regression",
        drop_high=2,
        drop_low=1,
        preserve=2,
        drop_above=5.0,
        drop_below=1.001,
        exclude_valuations=[1989, "1985-12-31"],
        trim_ties="origin",
    ),
    "benktander": lambda: methods.benktander(
        cells, premium=premium, expected_loss_ratio=0.7, n_iters=3, average="simple"
    ),
    "cape_cod": lambda: methods.cape_cod(cells, premium=premium_table, decay=0.5),
    "cape_cod_trend": lambda: methods.cape_cod(
        cells, premium=premium_table, trend=0.05, n_iters=2, exclude_valuations=["1988"]
    ),
    "mack": lambda: methods.mack(cells),
    "mack_options": lambda: methods.mack(cells, sigma_rule="mack", zero_cells="observed"),
    "mack_selection": lambda: methods.mack(
        cells,
        average="regression",
        history_periods=6,
        drop_high=1,
        drop_above=5.0,
        exclude=[(1982, 12)],
        exclude_valuations=[1989],
    ),
    "tweedie_glm": lambda: methods.tweedie_glm(cells, power=0),
    "tweedie_glm_options": lambda: methods.tweedie_glm(
        cells, power=0, link="identity", origin="none", calendar="trend", max_iter=50
    ),
    "refused_tweedie_glm": lambda: methods.tweedie_glm(cells),
    "refused_grain": lambda: methods.chain_ladder(cells, dev_grain_months=5),
    "refused_exclusion": lambda: methods.chain_ladder(cells, exclude=[(1990, 12)]),
    "refused_valuation": lambda: methods.chain_ladder(cells, exclude_valuations=["1990Q4"]),
}}
out = {{}}
for name, call in calls.items():
    try:
        result = call()
        tables = ("origins", "development", "link_ratios", "totals", "cells", "coefficients")
        answer = {{
            t: getattr(result, t).to_pylist() for t in tables if getattr(result, t) is not None
        }}
    except ValueError as exc:
        answer = str(exc)
    loaded = [m for m in NOT if m in sys.modules]
    out[name] = {{"loaded": loaded, "answer": answer}}
print(json.dumps(out, default=str))
"""


def _raa_files(tmp_path) -> list[str]:
    """raa's cells and a premium table, written as Arrow files for a child to read."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    origin, lag, value = [], [], []
    for i, row in enumerate(RAA):
        for j, amount in enumerate(row):
            origin.append(1981 + i)
            lag.append(12 * (j + 1))
            value.append(float(amount))
    cells = pa.table({"origin_period": origin, "dev_lag": lag, "value": value})
    premium = pa.table(
        {
            "origin_period": list(range(1981, 1991)),
            "premium": [20000.0 + 2000.0 * i for i in range(10)],
        }
    )
    paths = []
    for name, table in (("cells", cells), ("premium", premium)):
        path = tmp_path / f"{name}.arrow"
        with ipc.new_file(str(path), table.schema) as writer:
            writer.write_table(table)
        paths.append(str(path))
    return paths


def _answers(code: str, *paths: str) -> dict:
    import json

    proc = subprocess.run([sys.executable, "-c", code, *paths], capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_every_method_runs_without_loading_ibis_pandas_scipy_or_sklearn(tmp_path):
    """Each call leaves all four unloaded, and answers as it does with them loaded.

    The second half is what stops the first from passing on a child that never
    reached the methods: the answers are compared value for value with the same
    calls made in a process that has already loaded pandas, scipy and ibis.
    Mutations: build one output column with ``pa.array`` in ``methods.py``
    (pyarrow then imports pandas), or have ``_conventional_result`` read
    ``fit_conventional_grid``'s pandas tables; each fails naming pandas.
    """
    paths = _raa_files(tmp_path)
    answers = _answers(_METHOD_CALLS.format(not_for=NOT_FOR_METHODS), *paths)
    expected = _answers(
        "import pandas, scipy.stats, ibis\n" + _METHOD_CALLS.format(not_for=()), *paths
    )

    assert set(answers) == set(expected)
    for name, got in answers.items():
        assert got["loaded"] == [], f"{name} loaded {got['loaded']}"
        assert got["answer"] == expected[name]["answer"], name
    # the refusals were refusals, and every other call answered
    refused = {name for name, got in answers.items() if isinstance(got["answer"], str)}
    assert refused == {
        "refused_grain",
        "refused_exclusion",
        "refused_valuation",
        "refused_tweedie_glm",
    }


def test_every_column_type_the_methods_read_loads_no_pandas(tmp_path):
    """Each column type the methods accept goes through its own reading code
    (integer years of another width, text and dictionary labels, dates that end
    a period, timestamps with a time zone, float ages, decimal amounts). Each is
    read in a clean process without loading pandas, and gives raa's total."""
    import datetime as dt
    import decimal

    import pyarrow as pa
    import pyarrow.ipc as ipc

    rows = [(1981 + i, 12 * (j + 1), float(v)) for i, r in enumerate(RAA) for j, v in enumerate(r)]
    years = [o for o, _, _ in rows]
    lags = [lag for _, lag, _ in rows]
    values = [v for _, _, v in rows]
    labels = [str(year) for year in years]
    forms = {
        "int32 years, int16 ages, float32 amounts": (
            pa.array(years, pa.int32()),
            pa.array(lags, pa.int16()),
            pa.array(values, pa.float32()),
        ),
        "text labels, float ages, integer amounts": (
            pa.array(labels),
            pa.array([float(lag) for lag in lags]),
            pa.array([int(v) for v in values]),
        ),
        "dictionary labels, decimal amounts": (
            pa.array(labels).dictionary_encode(),
            pa.array(lags),
            pa.array([decimal.Decimal(int(v)) for v in values], pa.decimal128(12, 2)),
        ),
        "large-string labels, uint8 ages": (
            pa.array(labels, pa.large_string()),
            pa.array(lags, pa.uint8()),
            pa.array(values),
        ),
        "float16 ages": (pa.array(years), pa.array(np.array(lags, np.float16)), pa.array(values)),
        "float32 ages": (pa.array(years), pa.array(lags, pa.float32()), pa.array(values)),
        "uint64 ages": (pa.array(years), pa.array(lags, pa.uint64()), pa.array(values)),
        "dates that end the year": (
            pa.array([dt.date(year, 12, 31) for year in years], pa.date32()),
            pa.array(lags),
            pa.array(values),
        ),
        "timestamps with a time zone": (
            pa.array([dt.datetime(year, 1, 1) for year in years], pa.timestamp("us", tz="UTC")),
            pa.array(lags),
            pa.array(values),
        ),
    }
    paths = []
    for k, (origin, lag, value) in enumerate(forms.values()):
        table = pa.table({"origin_period": origin, "dev_lag": lag, "value": value})
        path = tmp_path / f"form{k}.arrow"
        with ipc.new_file(str(path), table.schema) as writer:
            writer.write_table(table)
        paths.append(str(path))
    code = (
        "import sys\n"
        "import pyarrow.ipc as ipc\n"
        "from ibnr import methods\n"
        "for path in sys.argv[1:]:\n"
        "    cells = ipc.open_file(path).read_all()\n"
        "    total = methods.chain_ladder(cells).totals['ultimate'][0].as_py()\n"
        "    se = methods.mack(cells).totals['mack_se'][0].as_py()\n"
        f"    assert abs(total - {RAA_TOTAL_ULTIMATE!r}) < 1e-6, (path, total)\n"
        f"    assert abs(se - {RAA_TOTAL_MACK_SE!r}) < 1e-6, (path, se)\n"
        f"    loaded = [m for m in {NOT_FOR_METHODS!r} if m in sys.modules]\n"
        "    assert not loaded, (path, loaded)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, *paths], capture_output=True, text=True)
    assert proc.returncode == 0, f"{list(forms)}\n{proc.stderr}"


def _number_types():
    from ibnr import _arrow

    return list(_arrow._NUMPY_TYPES)


@pytest.mark.parametrize("kind", _number_types(), ids=str)
def test_to_numpy_reads_every_number_type_as_pyarrow_does(kind):
    """``_arrow.to_numpy`` gives the array ``to_numpy`` gives, type and values,
    for every Arrow number type it reads, whole, sliced (a nonzero offset) and
    in chunks. Mutation: map ``pa.float16()`` to ``np.float32``; the float16
    case fails."""
    from ibnr import _arrow

    values = np.array([0, 1, 7, 12, 24, 100], dtype=_arrow._NUMPY_TYPES[kind])
    whole = pa.array(values, kind)
    for array in (whole, whole.slice(2, 3), pa.chunked_array([whole.slice(0, 2), whole[2:]])):
        expected = array.to_numpy()
        got = _arrow.to_numpy(array)
        assert got.dtype == expected.dtype
        assert got.tolist() == expected.tolist()
    assert _arrow.to_numpy(whole.slice(0, 0)).dtype == values.dtype


def test_a_polars_frame_goes_in_without_loading_pandas():
    """``pa.table()`` asks whether its argument is a pandas DataFrame, which
    imports pandas; ``methods`` reads a polars frame through the Arrow stream
    interface instead. Mutation: drop that branch from ``methods._table``; this
    fails naming pandas."""
    pytest.importorskip("polars")
    code = (
        "import sys\n"
        "import polars as pl\n"
        "from ibnr import methods\n"
        f"RAA = {RAA!r}\n"
        "rows = [(1981 + i, 12 * (j + 1), float(v)) for i, r in enumerate(RAA) "
        "for j, v in enumerate(r)]\n"
        "cells = pl.DataFrame(rows, schema=['origin_period', 'dev_lag', 'value'], orient='row')\n"
        "premium = pl.DataFrame({'origin_period': list(range(1981, 1991)), "
        "'premium': [20000.0 + 2000.0 * i for i in range(10)]})\n"
        "methods.chain_ladder(cells)\n"
        "methods.bornhuetter_ferguson(cells, premium=premium, expected_loss_ratio=0.7)\n"
        "methods.cape_cod(cells, premium=premium)\n"
        "total = methods.mack(cells).totals['mack_se'][0].as_py()\n"
        f"loaded = [m for m in {NOT_FOR_METHODS!r} if m in sys.modules]\n"
        "assert not loaded, 'a polars frame loaded ' + repr(loaded)\n"
        f"assert abs(total - {RAA_TOTAL_MACK_SE!r}) < 1e-6, total\n"
    )
    _run(code)


def test_a_dict_of_lists_goes_in_without_loading_pandas():
    """A service reading JSON has the columns as Python lists; ``pa.table`` on
    such a dict calls ``pa.array``, which imports pandas (about 1 s on the dev
    box). ``methods`` builds those columns itself. Lists of strings, ints,
    floats and dates, and numpy arrays, each give raa's totals with no pandas.
    Mutation: send a dict straight to ``pa.table`` in ``methods._table``; this
    fails naming pandas."""
    code = (
        "import datetime as dt, sys\n"
        "import numpy as np\n"
        "from ibnr import methods\n"
        f"RAA = {RAA!r}\n"
        "rows = [(1981 + i, 12 * (j + 1), v) for i, r in enumerate(RAA) "
        "for j, v in enumerate(r)]\n"
        "years = [o for o, _, _ in rows]\n"
        "lags = [lag for _, lag, _ in rows]\n"
        "values = [v for _, _, v in rows]\n"
        "forms = {\n"
        "    'text labels, int ages, float amounts': {'origin_period': [str(y) for y in years],"
        " 'dev_lag': lags, 'value': [float(v) for v in values]},\n"
        "    'int years, float ages, int amounts': {'origin_period': years,"
        " 'dev_lag': [float(g) for g in lags], 'value': values},\n"
        "    'dates, mixed int and float amounts': {'origin_period':"
        " [dt.date(y, 12, 31) for y in years], 'dev_lag': lags,"
        " 'value': [float(v) if k % 2 else v for k, v in enumerate(values)]},\n"
        "    'numpy arrays': {'origin_period': np.array([str(y) for y in years]),"
        " 'dev_lag': np.array(lags, dtype=np.int32), 'value': np.array(values, dtype=float)},\n"
        "}\n"
        "premium = {'origin_period': list(range(1981, 1991)),"
        " 'premium': [20000.0 + 2000.0 * i for i in range(10)]}\n"
        "for name, cells in forms.items():\n"
        "    total = methods.chain_ladder(cells).totals['ultimate'][0].as_py()\n"
        "    se = methods.mack(cells).totals['mack_se'][0].as_py()\n"
        f"    assert abs(total - {RAA_TOTAL_ULTIMATE!r}) < 1e-6, (name, total)\n"
        f"    assert abs(se - {RAA_TOTAL_MACK_SE!r}) < 1e-6, (name, se)\n"
        f"    loaded = [m for m in {NOT_FOR_METHODS!r} if m in sys.modules]\n"
        "    assert not loaded, (name, loaded)\n"
    )
    _run(code)


#: Dicts ``methods`` reads, each compared with what ``pa.table`` makes of it.
#: The last five are ones it leaves to ``pa.table``: a null, a datetime, an
#: integer too large for int64, and two lists of mixed types.
_DICTS = [
    {"a": ["1988", "1989"], "b": [12, 24], "c": [1.5, 2.0], "d": [True, False]},
    {"a": [1988, 1989], "b": [12.0, 24.0], "c": [1, 2.5], "d": (3, 4)},
    {"a": [dt.date(1988, 12, 31), dt.date(1989, 1, 1)], "b": ["é", "x"]},
    {
        "a": np.array([1, 2], dtype=np.uint16),
        "b": np.array(["x", "yz"]),
        "c": np.array([0.5, 1.5], dtype=np.float32),
        "d": np.array([True, False]),
        "e": np.array([2**63 + 1, 5], dtype=np.uint64),
        "f": np.array(["1988-01-01", "1989-12-31"], dtype="datetime64[D]"),
    },
    {"a": pa.array([1, 2]), "b": pa.chunked_array([[1.0], [2.0]]), "c": [np.int64(1), 2]},
    {"a": [None, 1]},
    {"a": [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)]},
    {"a": [2**70, 1]},
    {"a": [1, "x"]},
    {"a": [True, 1]},
]


@pytest.mark.parametrize("data", _DICTS)
def test_a_dict_is_read_as_pa_table_reads_it(data):
    """The columns built without pandas are the ones ``pa.table`` builds, type
    and value, and a dict that cannot be built that way still reaches
    ``pa.table``, which answers or refuses as it always has."""
    from ibnr import methods

    try:
        expected = pa.table(data)
    except (ValueError, OverflowError):
        with pytest.raises((ValueError, OverflowError)):
            methods._table(data, "cells")
        return
    got = methods._table(data, "cells")
    assert got.schema == expected.schema
    assert got.equals(expected)


def test_a_dict_with_columns_of_different_lengths_is_refused():
    from ibnr import methods

    with pytest.raises(ValueError, match="must be a table Arrow can read"):
        methods._table({"a": [1, 2], "b": [1.0]}, "cells")


def test_a_nan_in_a_dict_stays_nan_as_in_pa_table():
    """``pa.table`` keeps a NaN from a list or a numpy array as NaN, not null,
    so ``methods`` refuses it by name as a NaN."""
    from ibnr import methods

    for column in ([1.0, float("nan")], np.array([1.0, np.nan])):
        got = methods._table({"a": column}, "cells").column("a")
        assert got.type == pa.float64()
        assert got.null_count == 0
        assert np.isnan(got[1].as_py())


def test_a_bare_import_of_ibnr_loads_no_triangle_layer_and_keeps_its_attributes():
    """``import ibnr`` no longer loads ibis; ``ibnr.Triangle`` and
    ``ibnr.triangle`` still work, and ``ibnr.gallery`` is still an
    AttributeError until something imports it. Mutation: import ``Triangle``
    eagerly in ``ibnr/__init__.py``; this fails naming ibis and pandas."""
    code = (
        "import sys\n"
        "import ibnr\n"
        f"loaded = [m for m in {NOT_FOR_METHODS!r} if m in sys.modules]\n"
        "assert not loaded, 'import ibnr loaded ' + repr(loaded)\n"
        "# dir() before any name is read: a read caches the name in globals()\n"
        "missing = set(ibnr.__all__) - set(dir(ibnr))\n"
        "assert not missing, 'dir(ibnr) leaves out ' + repr(missing)\n"
        "assert 'ibis' not in sys.modules, 'dir(ibnr) imported ibis'\n"
        "try:\n"
        "    ibnr.gallery\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('ibnr.gallery answered before anything imported it')\n"
        "assert 'ibnr.triangle' not in sys.modules\n"
        "layer = ibnr.triangle  # read before anything imports it\n"
        "from ibnr.triangle import Triangle, TriangleMeta\n"
        "assert layer.Triangle is Triangle\n"
        "assert ibnr.Triangle is Triangle and ibnr.TriangleMeta is TriangleMeta\n"
        "assert 'ibis' in sys.modules\n"
        "namespace = {}\n"
        "exec('from ibnr import *', namespace)\n"
        "assert {'Triangle', 'TriangleMeta', '__version__'} <= set(namespace)\n"
        "assert sorted(ibnr.__all__) == ['Triangle', 'TriangleMeta', '__version__']\n"
    )
    _run(code)


def test_every_kernels_name_is_the_object_its_module_defines():
    """The lazy ``kernels`` hands out exactly the objects the eager one did.
    Mutation: point one ``_LAZY`` entry at the wrong module, or drop one; this
    fails naming it."""
    import importlib

    from ibnr import kernels

    assert set(kernels._LAZY) == set(kernels.__all__)
    for name in kernels.__all__:
        module = importlib.import_module(f"ibnr.kernels.{kernels._LAZY[name]}")
        assert getattr(kernels, name) is getattr(module, name), name
    with pytest.raises(AttributeError, match="no attribute 'no_such_name'"):
        kernels.no_such_name  # noqa: B018


def test_a_kernels_submodule_is_still_an_attribute_and_star_import_works():
    """Importing every name eagerly made each kernels submodule an attribute, so
    ``kernels.codec`` answered; it still does, in a fresh process. ``dir()``
    lists every ``__all__`` name before any is read, without importing it.
    Mutation: have ``kernels.__dir__`` return ``sorted(globals())``; this fails
    naming the names it leaves out."""
    code = (
        "import sys\n"
        "from ibnr import kernels\n"
        f"loaded = [m for m in {NOT_FOR_METHODS!r} if m in sys.modules]\n"
        "assert not loaded, 'from ibnr import kernels loaded ' + repr(loaded)\n"
        "# dir() before any name is read: a read caches the name in globals()\n"
        "missing = set(kernels.__all__) - set(dir(kernels))\n"
        "assert not missing, 'dir(kernels) leaves out ' + repr(sorted(missing))\n"
        f"loaded = [m for m in {NOT_FOR_METHODS!r} if m in sys.modules]\n"
        "assert not loaded, 'dir(kernels) loaded ' + repr(loaded)\n"
        "assert 'ibnr.kernels.codec' not in sys.modules\n"
        "assert kernels.codec.to_arrow is kernels.to_arrow\n"
        "namespace = {}\n"
        "exec('from ibnr.kernels import *', namespace)\n"
        "missing = set(kernels.__all__) - set(namespace)\n"
        "assert not missing, missing\n"
    )
    _run(code)


def _type_checking_imports(path) -> dict[str, str]:
    """name -> module for every import under ``if TYPE_CHECKING:`` in a file."""
    import ast

    names = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom):
                    for alias in statement.names:
                        names[alias.asname or alias.name] = statement.module
    return names


def test_the_docs_build_finds_every_lazy_name_in_source():
    """The docs build reads source and runs nothing (``dynamic: false``), so each
    lazily imported name must also be imported under ``TYPE_CHECKING``, from the
    module ``__getattr__`` takes it from. Mutation: delete one name from either
    block; this fails."""
    from pathlib import Path

    import ibnr
    from ibnr import kernels

    assert _type_checking_imports(Path(kernels.__file__)) == {
        name: f"ibnr.kernels.{module}" for name, module in kernels._LAZY.items()
    }
    assert _type_checking_imports(Path(ibnr.__file__)) == dict(ibnr._LAZY)


def test_the_glm_kernel_loads_numpy_only():
    """``kernels.glm`` (``methods.tweedie_glm``'s fit) imports numpy and ibnr's own
    numpy-only modules; a fit in the same process loads nothing more."""
    script = textwrap.dedent(
        f"""
        import sys
        import numpy as np
        from ibnr.kernels.glm import fit_tweedie_grid
        from ibnr.kernels.grid import grid_from_columns
        origins = np.array(["2001-01-01"] * 3 + ["2002-01-01"] * 2 + ["2003-01-01"], "M8[D]")
        lags = np.array([12, 24, 36, 12, 24, 12])
        values = np.array([100.0, 150, 170, 110, 160, 120])
        grid = grid_from_columns(origins, lags, values, dev_grain_months=12, measure="cumulative")
        fit_tweedie_grid(grid)
        print([m for m in {NOT_FOR_METHODS!r} if m in sys.modules])
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=300
    ).stdout
    assert out.strip() == "[]"


def test_the_grid_module_and_contract_share_one_set_of_helpers():
    """``kernels.grid`` holds the helpers; ``kernels.contract`` re-exports the
    same objects, and ``grid.TRIANGLE_MEASURES`` is the Triangle layer's ``Measure``."""
    from typing import get_args

    from ibnr.kernels import contract, grid
    from ibnr.triangle.core import Measure

    for name in (
        "ZERO_CELLS",
        "as_date",
        "check_grid",
        "dev_step_index",
        "grid_from_columns",
        "month_end",
        "require_run_off",
    ):
        assert getattr(contract, name) is getattr(grid, name), name
    assert get_args(Measure) == grid.TRIANGLE_MEASURES


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

    Compares the failures to ``NEEDS_NN_EXTRA | NEEDS_JAX_EXTRA`` as a SET - a module dropping OUT
    of the allowlist is as much a finding as one joining it, because it means the
    allowlist has gone stale and is no longer describing the package.
    """
    import json

    failed = json.loads(_run(_WALK.format(blocked=sorted(HEAVY))).stdout.strip().splitlines()[-1])

    expected = NEEDS_NN_EXTRA | NEEDS_JAX_EXTRA
    assert set(failed) == set(expected), (
        "the set of modules needing an extra changed.\n"
        f"  unexpectedly failing: {sorted(set(failed) - expected)}\n"
        f"  no longer failing:    {sorted(expected - set(failed))}\n"
        f"  reasons: {failed}"
    )
