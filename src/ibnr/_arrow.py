"""Arrow arrays built from numpy, and read back into numpy, without loading pandas.

pyarrow imports pandas, when pandas is installed, the first time ``pa.array``
or ``pa.scalar`` turns Python or numpy values into Arrow, and the first time
``Array.to_numpy`` turns Arrow into numpy. pandas is a dependency of ibnr, so it
is always installed, and importing it took 1.1 to 1.5 s on the dev box on
2026-09-25, after numpy and pyarrow. ``ibnr.methods`` does not use pandas, so it
builds and reads its arrays here, straight from and to the arrays' memory
buffers, which pyarrow does without pandas.

Only what ``ibnr.methods`` uses is here: columns with no nulls read as numpy;
float64, int64, bool, date32 and string arrays built from numpy arrays or
lists; and ``column``, which builds each column of a dict of Python lists or
numpy arrays, the input a service reading JSON has, as ``pa.table`` would
(``pa.table`` on such a dict imports pandas). The values are the ones
``pa.array`` and ``to_numpy`` give;
``tests/test_methods.py`` checks the methods' results against the kernels
value for value.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pyarrow as pa

#: The numpy type of each Arrow number type ``to_numpy`` reads.
_NUMPY_TYPES = {
    pa.int8(): np.int8,
    pa.int16(): np.int16,
    pa.int32(): np.int32,
    pa.int64(): np.int64,
    pa.uint8(): np.uint8,
    pa.uint16(): np.uint16,
    pa.uint32(): np.uint32,
    pa.uint64(): np.uint64,
    pa.float16(): np.float16,
    pa.float32(): np.float32,
    pa.float64(): np.float64,
}

_EPOCH = dt.date(1970, 1, 1).toordinal()


def to_numpy(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """A number column with no nulls as a numpy array of the matching type.

    The same array ``values.to_numpy()`` gives, copied out of Arrow's memory so
    it can be written to.
    """
    array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    if array.null_count:
        raise ValueError("to_numpy reads a column with no nulls")  # callers refuse nulls first
    kind = _NUMPY_TYPES.get(array.type)
    if kind is None:
        raise TypeError(f"to_numpy reads number columns, got {array.type}")
    if len(array) == 0:
        return np.empty(0, dtype=kind)
    start = array.offset * np.dtype(kind).itemsize
    return np.frombuffer(array.buffers()[1], dtype=kind, count=len(array), offset=start).copy()


def _validity(valid: np.ndarray) -> tuple[pa.Buffer | None, int]:
    """Arrow's validity bitmap for a boolean array of which values are present."""
    missing = int((~valid).sum())
    if not missing:
        return None, 0
    return pa.py_buffer(np.packbits(valid, bitorder="little")), missing


def float64(values, *, mask: np.ndarray | None = None) -> pa.Array:
    """A float64 array; ``mask`` is True where a value is missing."""
    data = np.ascontiguousarray(values, dtype=np.float64)
    bitmap, missing = (None, 0) if mask is None else _validity(~np.asarray(mask, dtype=bool))
    return pa.Array.from_buffers(
        pa.float64(), len(data), [bitmap, pa.py_buffer(data)], null_count=missing
    )


def int64(values) -> pa.Array:
    data = np.ascontiguousarray(values, dtype=np.int64)
    return pa.Array.from_buffers(pa.int64(), len(data), [None, pa.py_buffer(data)])


def bool_(values) -> pa.Array:
    data = np.asarray(values, dtype=bool)
    bits = pa.py_buffer(np.packbits(data, bitorder="little"))
    return pa.Array.from_buffers(pa.bool_(), len(data), [None, bits])


def date32(days: list[dt.date]) -> pa.Array:
    data = np.array([day.toordinal() - _EPOCH for day in days], dtype=np.int32)
    return pa.Array.from_buffers(pa.date32(), len(data), [None, pa.py_buffer(data)])


def string(values: list[str]) -> pa.Array:
    encoded = [value.encode("utf-8") for value in values]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.array([len(value) for value in encoded], dtype=np.int32))
    return pa.Array.from_buffers(
        pa.string(),
        len(encoded),
        [None, pa.py_buffer(offsets), pa.py_buffer(b"".join(encoded))],
    )


#: The Arrow type of each numpy number type ``column`` builds from.
_ARROW_TYPES = {np.dtype(kind): arrow for arrow, kind in _NUMPY_TYPES.items()}


def column(values) -> pa.Array | pa.ChunkedArray | None:
    """One column of a dict of columns, as ``pa.table`` would build it, or None.

    Builds what a service reading JSON or numpy has: a list or tuple of only
    strings, only bools, only dates, or only ints and floats (float64 if any is
    a float, else int64), or a one-dimensional numpy array of numbers, bools,
    strings or days. An Arrow array is returned as it is. Anything else (a
    null, a datetime, an int beyond int64, mixed kinds, an empty list) gives
    None, and the caller leaves the whole dict to ``pa.table``.
    """
    if isinstance(values, pa.Array | pa.ChunkedArray):
        return values
    if isinstance(values, np.ndarray):
        kind = values.dtype
        if values.ndim != 1:
            return None
        if kind in _ARROW_TYPES:
            data = np.ascontiguousarray(values)
            return pa.Array.from_buffers(_ARROW_TYPES[kind], len(data), [None, pa.py_buffer(data)])
        if kind == np.bool_:
            return bool_(values)
        if kind.kind == "U":
            return string(values.tolist())
        if kind == np.dtype("datetime64[D]") and not np.isnat(values).any():
            return date32(values.tolist())
        if kind != np.object_:
            return None
        values = values.tolist()
    if not isinstance(values, list | tuple) or not values:
        return None
    kinds = {_kind(value) for value in values}
    try:
        if kinds == {"str"}:
            return string(list(values))
        if kinds == {"bool"}:
            return bool_(list(values))
        if kinds == {"date"}:
            return date32(list(values))
        if kinds == {"int"}:
            return int64(np.array(values, dtype=np.int64))
        if kinds == {"float"} or kinds == {"int", "float"}:
            return float64(np.array(values, dtype=np.float64))
    except OverflowError:  # an int beyond int64 or float64
        return None
    return None


def _kind(value) -> str | None:
    """Which of the kinds ``column`` builds a Python value is, if any."""
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool | np.bool_):
        return "bool"
    if isinstance(value, int | np.integer):
        return "int"
    if isinstance(value, float | np.floating):
        return "float"
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return "date"
    return None


def with_last_null(values, kind: pa.DataType) -> pa.Array:
    """The values followed by one null, as a float64, int64 or bool array."""
    values = np.asarray(values)
    valid = np.r_[np.ones(len(values), dtype=bool), False]
    if kind == pa.float64():
        data = np.r_[values.astype(np.float64), 0.0]
    elif kind == pa.int64():
        data = np.r_[values.astype(np.int64), 0]
    elif kind == pa.bool_():
        data = np.packbits(np.r_[values.astype(bool), False], bitorder="little")
    else:
        raise TypeError(f"with_last_null builds float64, int64 or bool arrays, not {kind}")
    bitmap, missing = _validity(valid)
    return pa.Array.from_buffers(kind, len(valid), [bitmap, pa.py_buffer(data)], null_count=missing)
