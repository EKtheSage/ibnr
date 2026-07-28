"""The single place an ibnr result becomes bytes.

Nothing in this package builds a wire payload anywhere else, for the same reason
``densities.py`` is the only place a density changes measure: a format that is
open-coded at three call sites is three formats within a release.

**Why Arrow IPC, uncompressed.** Measured on the payload an API actually
returns - a 10-target, 10,000-draw ``PredictiveDistribution``, 800,000 B of raw
float64:

    Arrow IPC stream, uncompressed    802,888 B    encode ~0.5 ms
    Arrow IPC stream, lz4             803,320 B    encode ~0.9 ms
    Arrow IPC stream, zstd            742,432 B    encode ~1.4 ms
    parquet, defaults                 978,363 B    encode ~7 ms
    JSON, full float repr           1,952,983 B    encode ~39 ms
    summary JSON                        3,571 B    encode ~3 ms

Method: median of 50 encodes after 5 warmups, on a 2025 laptop-class x86 box
(Intel Core Ultra 9 285H, Windows 11, pyarrow 24, python 3.12). The SIZES are
deterministic and reproduce byte for byte. The TIMES are quoted to one
significant figure because that is all that survives - re-running the same
50-encode median moved the uncompressed row between 0.38 and 0.64 ms across
sessions on this machine, a ~40% swing - so the ratios below are the finding
and the absolute milliseconds are context. Every timing ratio quoted here is
far enough from 1 that a swing that size cannot reach it.

Three of those rows decide the design. Compression is close to useless because
posterior mantissas are high-entropy noise - zstd buys 7.5% of the size for
about three times the encode cost, and lz4 comes out LARGER than both the
uncompressed envelope and the raw draws - so compression is a knob and never
the default. parquet's default is 22% bigger than Arrow because it attempts
dictionary encoding on doubles that are all distinct; it is the right codec for
a bulk triangle export and the wrong one for draws. And JSON is 2.4x the size
at roughly 90x the encode time, which is why the JSON mode here carries a
SUMMARY and not the draws.

**Why lossless, exactly.** float32 would halve the payload for a maximum
relative error of 6e-8 - that is just the float32 rounding bound, 2**-24 - and
thinning 10:1 would cut it by 90%. What thinning costs is the tail. Over 200
seeds of this payload, dropping 10,000 draws to 1,000 moves the 99.5th
percentile by a median 2.2% (10th-90th of that spread: 1.6% to 3.1%) against
0.7% for the median - so the number a reserve answer is actually read off
degrades about three times faster than the middle of the distribution. Draws
cross this boundary bit for bit or not at all; the tests compare
``samples.view(np.uint8)``, because an ``allclose`` assertion passes cleanly
against a float32 encoder and would therefore be worthless.

Envelope, version 1. One Arrow IPC **stream** per artifact:

    body table          the artifact's largest rectangular array
    schema metadata     b"ibnr.kind"            artifact type name
                        b"ibnr.version"         codec version, ASCII int
                        b"ibnr.header"          UTF-8 JSON object of scalars
                        b"ibnr.frames.<name>"   nested IPC stream, one DataFrame
                        b"ibnr.arrays.<name>"   nested IPC stream, one ndarray
                        b"ibnr.nested.<name>"   a complete nested envelope

Arrow metadata values are raw bytes, so nesting a whole IPC stream inside one is
legal and byte-exact. Every array shape and ``n_draws`` lives in the header
rather than being inferred from the table, because a zero-target distribution
encodes to a zero-column table (64 bytes) from which the draw count cannot be
recovered.

Frames go through ``pa.Table.from_pandas``, which handles every dtype in this
package except object columns holding tuples or ``pd.NA`` - see
:data:`_TAGGED_COLUMNS` for why those two matter and what happens to them.

The header is emitted with ``allow_nan=False``. That is a guard rail rather
than a formality: ``json.dumps`` will happily write the bare tokens ``NaN`` and
``Infinity``, which are not JSON and which a non-Python client will reject. Any
scalar that can legitimately be non-finite - a degenerate cohort's msep, say -
therefore travels as a shape-() entry under ``arrays``, where it keeps its
exact bit pattern.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from ibnr.kernels.cdr import CDRResult
from ibnr.kernels.forecast import ForecastPanel
from ibnr.kernels.mack import MackFit, MackFitPanel
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

#: Envelope version, written into every payload and checked on the way back in.
#: A payload from a NEWER codec is refused rather than partly decoded.
CODEC_VERSION: int = 1

#: MIME type for :func:`to_arrow` output, for an HTTP layer's ``Content-Type``
#: and ``Accept`` negotiation.
CONTENT_TYPE_ARROW: str = "application/vnd.apache.arrow.stream"

#: MIME type for :func:`to_summary` output, which is the negotiated fallback for
#: a caller that cannot take the draws.
CONTENT_TYPE_JSON: str = "application/json"

#: Quantile grid for :func:`to_summary`. Runs out to 0.5% / 99.5% because the
#: tail is the point of a reserve distribution; a summary that stopped at the
#: 95th would answer a different question from the draws it replaces.
DEFAULT_QUANTILES: tuple[float, ...] = (
    0.005,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.975,
    0.995,
)

_KIND = b"ibnr.kind"
_VERSION = b"ibnr.version"
_HEADER = b"ibnr.header"
_FRAME_PREFIX = b"ibnr.frames."
_ARRAY_PREFIX = b"ibnr.arrays."
_NESTED_PREFIX = b"ibnr.nested."

#: Object columns Arrow cannot restore on its own, marked per frame. Two of them
#: matter and both are in ``ForecastPanel``:
#:
#: * tuple columns (``key``, ``cohort``, ``missing_from``). Arrow has no tuple
#:   type, so a list column decodes to an ndarray - which is unhashable, and
#:   these columns are grouped, joined and ``isin``-ed on. That is not a lossy
#:   round trip, it is a panel that raises on first use.
#: * object columns mixing values with ``pd.NA`` (``pointwise.n_draws_density``,
#:   ``by_cohort.elpd``). Arrow types those as int64/double with nulls and pandas
#:   decodes them as float64 with NaN, which silently converts "this model has no
#:   ELPD" into "not a number" - the missing-versus-value distinction the whole
#:   board rests on (CLAUDE.md milestone 6: a missing score is pd.NA, NEVER 0.0).
_TAGGED_COLUMNS = b"ibnr.tagged_columns"


# -- envelope primitives -------------------------------------------------------


def _write_stream(table: pa.Table, compression: str | None) -> bytes:
    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(compression=compression)
    with pa.ipc.new_stream(sink, table.schema, options=options) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _read_stream(data: bytes) -> pa.Table:
    with pa.ipc.open_stream(pa.py_buffer(data)) as reader:
        return reader.read_all()


def _pack(
    kind: str,
    body: pa.Table,
    header: dict,
    *,
    frames: dict[str, pd.DataFrame] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
    nested: dict[str, bytes] | None = None,
    compression: str | None = None,
) -> bytes:
    """Assemble one envelope. The only writer in the package."""
    header = dict(header)
    if arrays:
        # Shape and dtype are recorded, not inferred: the payload is flattened
        # C-order into one column, so a (3, 4) and a (4, 3) array are the same
        # bytes and only this record tells them apart.
        header["arrays"] = {
            name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for name, arr in arrays.items()
        }
    metadata: dict[bytes, bytes] = {
        _KIND: kind.encode("utf-8"),
        _VERSION: str(CODEC_VERSION).encode("ascii"),
        _HEADER: json.dumps(header, allow_nan=False).encode("utf-8"),
    }
    for name, frame in (frames or {}).items():
        metadata[_FRAME_PREFIX + name.encode("utf-8")] = _frame_to_bytes(frame, compression)
    for name, arr in (arrays or {}).items():
        metadata[_ARRAY_PREFIX + name.encode("utf-8")] = _array_to_bytes(arr, compression)
    for name, blob in (nested or {}).items():
        metadata[_NESTED_PREFIX + name.encode("utf-8")] = blob
    return _write_stream(body.replace_schema_metadata(metadata), compression)


def _unpack(data: bytes) -> tuple[str, pa.Table, dict, dict, dict, dict]:
    """Take one envelope apart into (kind, body, header, frames, arrays, nested)."""
    table = _read_stream(data)
    metadata = table.schema.metadata or {}
    if _KIND not in metadata:
        raise ValueError("not an ibnr envelope: the Arrow schema carries no ibnr.kind")
    version = int(metadata[_VERSION])
    if version > CODEC_VERSION:
        # Forward-incompatible input must fail rather than decode the parts it
        # happens to recognize: a partially understood payload is a wrong answer
        # with no error attached to it.
        raise ValueError(
            f"envelope is codec version {version}, this build understands up to {CODEC_VERSION}"
        )
    header = json.loads(metadata[_HEADER])
    frames, arrays, nested = {}, {}, {}
    shapes = header.get("arrays", {})
    for key, blob in metadata.items():
        if key.startswith(_FRAME_PREFIX):
            frames[key[len(_FRAME_PREFIX) :].decode("utf-8")] = _frame_from_bytes(blob)
        elif key.startswith(_ARRAY_PREFIX):
            name = key[len(_ARRAY_PREFIX) :].decode("utf-8")
            arrays[name] = _array_from_bytes(blob, shapes[name])
        elif key.startswith(_NESTED_PREFIX):
            nested[key[len(_NESTED_PREFIX) :].decode("utf-8")] = blob
    return metadata[_KIND].decode("utf-8"), table, header, frames, arrays, nested


# -- DataFrames ----------------------------------------------------------------


def _frame_to_bytes(frame: pd.DataFrame, compression: str | None) -> bytes:
    """One DataFrame as a nested IPC stream.

    ``preserve_index=None`` is pyarrow's default and the right one here: a
    RangeIndex rides in the pandas metadata and a named index becomes a column,
    so both come back as they went in. Object columns Arrow cannot restore go
    through :func:`_tag` value by value; anything :func:`_tag` does not know is
    refused there by name rather than coerced to its ``repr``.
    """
    tagged = [c for c in frame.columns if frame[c].dtype == object and not _arrow_native(frame[c])]
    if tagged:
        frame = frame.copy()
        for column in tagged:
            frame[column] = [json.dumps(_tag(v)) for v in frame[column]]
    table = pa.Table.from_pandas(frame, preserve_index=None)
    metadata = dict(table.schema.metadata or {})
    metadata[_TAGGED_COLUMNS] = json.dumps(tagged).encode("utf-8")
    return _write_stream(table.replace_schema_metadata(metadata), compression)


def _frame_from_bytes(data: bytes) -> pd.DataFrame:
    table = _read_stream(data)
    metadata = table.schema.metadata or {}
    tagged = json.loads(metadata.get(_TAGGED_COLUMNS, b"[]"))
    frame = table.to_pandas()
    for column in tagged:
        frame[column] = pd.Series(
            [_untag(json.loads(v)) for v in frame[column]], index=frame.index, dtype=object
        )
    return frame


def _arrow_native(series: pd.Series) -> bool:
    """Whether Arrow gives this object column back unchanged, value for value.

    Only two shapes qualify, and both were measured rather than assumed: an
    all-string column (string -> object of str) and an all-``datetime.date``
    column (date32 -> object of date, because the pandas metadata rides in the
    Arrow schema). A ``datetime`` is deliberately excluded - it decodes as a
    ``Timestamp``, which is a different type - and any missing value disqualifies
    the column, because None, NaN and pd.NA all encode to one Arrow null and
    only one of them can come back.

    Everything else is tagged. The bias is towards tagging: a false negative
    costs a few JSON-quoted bytes in a metadata blob, a false positive is a
    silently altered value.
    """
    values = list(series)
    if any(_is_missing(v) for v in values):
        return False
    return all(isinstance(v, str) for v in values) or all(type(v) is dt.date for v in values)


# -- ndarrays ------------------------------------------------------------------


def _array_to_bytes(arr: np.ndarray, compression: str | None) -> bytes:
    return _write_stream(pa.table({"v": pa.array(arr.ravel(order="C"))}), compression)


def _array_from_bytes(data: bytes, record: dict) -> np.ndarray:
    flat = _read_stream(data)["v"].to_numpy(zero_copy_only=False)
    # astype before reshape so a bool mask comes back bool rather than float:
    # `obs_mask` indexes with it, and a float mask indexes by POSITION.
    return flat.astype(np.dtype(record["dtype"])).reshape(record["shape"])


# -- typed values --------------------------------------------------------------
#
# The escape hatch for everything JSON and Arrow both round off. Cohort keys are
# tuples of mixed types - a date beside a string beside an int - which no Arrow
# list column can hold; a board column is python floats beside pd.NA, which Arrow
# flattens to one null. Each value therefore carries its own type tag and
# reconstructs exactly, rather than as whatever the wire format rounded it to.


def _tag(value) -> list:
    if value is None:
        return ["null", None]
    if value is pd.NA:
        # pd.NA and NaN are one Arrow null but two different pandas answers, and
        # on the leaderboard the difference is "no score" versus "a score of not
        # a number". They are tagged apart here so they cannot merge.
        return ["na", None]
    if value is pd.NaT:
        return ["nat", None]
    if isinstance(value, tuple):
        return ["tuple", [_tag(part) for part in value]]
    if isinstance(value, bool | np.bool_):
        return ["bool", bool(value)]
    if isinstance(value, int | np.integer):
        return ["int", int(value)]
    if isinstance(value, float | np.floating):
        # repr, not the number: JSON has no NaN and no Infinity, and repr of a
        # float round-trips exactly through float() including those three.
        return ["float", repr(float(value))]
    if isinstance(value, pd.Timestamp):
        return ["timestamp", value.isoformat()]
    if isinstance(value, dt.datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, dt.date):
        return ["date", value.isoformat()]
    if isinstance(value, str):
        return ["str", value]
    raise TypeError(f"cannot serialize {value!r} of type {type(value).__name__}")


def _untag(tagged: Sequence) -> Any:
    kind, value = tagged
    if kind == "na":
        return pd.NA
    if kind == "nat":
        return pd.NaT
    if kind == "tuple":
        return tuple(_untag(part) for part in value)
    if kind == "float":
        return float(value)
    if kind == "date":
        return dt.date.fromisoformat(value)
    if kind == "datetime":
        return dt.datetime.fromisoformat(value)
    if kind == "timestamp":
        return pd.Timestamp(value)
    return value


def _is_missing(value) -> bool:
    """None / pd.NA / NaT / NaN.

    Spelled out rather than delegated to ``pd.isna``, which returns an ARRAY for
    a tuple - and the columns this is called on are full of tuples.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and bool(np.isnan(value))


# -- PredictiveDistribution ----------------------------------------------------


def _encode_predictive(obj: PredictiveDistribution, compression: str | None) -> bytes:
    # One column per target rather than one flat column: 802,888 B against
    # 800,272 B on the docstring's payload, 0.3%, so size does not decide it -
    # and columns are readable in any Arrow tool without knowing the layout.
    body = pa.table({f"t{i}": obj.samples[:, i] for i in range(obj.n_targets)})
    header = {"n_draws": obj.n_draws, "n_targets": obj.n_targets, "units": obj.units}
    return _pack(
        "PredictiveDistribution",
        body,
        header,
        frames={"targets": obj.targets},
        compression=compression,
    )


def _decode_predictive(body: pa.Table, header: dict, frames: dict, arrays, nested):
    n_draws, n_targets = header["n_draws"], header["n_targets"]
    samples = np.empty((n_draws, n_targets), dtype=float)
    for i in range(n_targets):
        samples[:, i] = body.column(f"t{i}").to_numpy(zero_copy_only=False)
    return PredictiveDistribution(samples=samples, targets=frames["targets"], units=header["units"])


# -- Triangle ------------------------------------------------------------------


def _encode_triangle(obj: Triangle, compression: str | None) -> bytes:
    meta = obj.meta
    header = {
        "origin_grain": meta.origin_grain,
        "dev_grain": meta.dev_grain,
        "measure": meta.measure,
        "units": meta.units,
        "segments": obj.segments,
    }
    # `backend` is deliberately absent. Which engine holds a triangle is a fact
    # about the process, not about the data, so it is a decode-time argument.
    return _pack(
        "Triangle",
        pa.Table.from_pandas(obj.to_pandas(), preserve_index=False),
        header,
        compression=compression,
    )


def _decode_triangle(body: pa.Table, header: dict, frames, arrays, nested, *, backend=None):
    from ibnr.triangle import io

    # Straight back through the one ingestion path, which is what keeps the
    # null-segment guard on the wire boundary too - a triangle arriving over
    # HTTP gets the same refusal as one built from a frame.
    return io.from_long(
        body,
        segments=header["segments"],
        measure=header["measure"],
        origin_grain=header["origin_grain"],
        dev_grain=header["dev_grain"],
        units=header["units"],
        backend=backend,
    )


# -- MackFit -------------------------------------------------------------------

#: Everything on a MackFit that is not `cum`. `n_obs` and `n_pos` are BOTH here
#: and are allowed to differ: `f[j]` is estimated over every pair origin,
#: `sigma2[j]` only over those with a positive cumulative, and dropping either
#: count loses the positivity contract that lets an origin with zero paid at 12
#: months keep its chain-ladder ultimate.
_MACK_ARRAYS = ("obs_mask", "latest_dev", "f", "sigma2", "s", "n_obs", "n_pos")


def _encode_mack_fit(obj: MackFit, compression: str | None) -> bytes:
    body = pa.table({f"d{j}": obj.cum[:, j] for j in range(obj.n_d)})
    header = {
        "origin_periods": [_tag(p) for p in obj.origin_periods],
        "dev_grain_months": obj.dev_grain_months,
        "sigma_rule": obj.sigma_rule,
        "units": obj.units,
        "loss_field": obj.loss_field,
        "n_w": obj.n_w,
        "n_d": obj.n_d,
    }
    arrays = {name: getattr(obj, name) for name in _MACK_ARRAYS}
    return _pack("MackFit", body, header, arrays=arrays, compression=compression)


def _decode_mack_fit(body: pa.Table, header: dict, frames, arrays, nested) -> MackFit:
    n_w, n_d = header["n_w"], header["n_d"]
    cum = np.empty((n_w, n_d), dtype=float)
    for j in range(n_d):
        cum[:, j] = body.column(f"d{j}").to_numpy(zero_copy_only=False)
    return MackFit(
        cum=cum,
        origin_periods=[_untag(p) for p in header["origin_periods"]],
        dev_grain_months=header["dev_grain_months"],
        sigma_rule=header["sigma_rule"],
        units=header["units"],
        loss_field=header["loss_field"],
        **{name: arrays[name] for name in _MACK_ARRAYS},
    )


# -- MackFitPanel --------------------------------------------------------------


def _encode_mack_panel(obj: MackFitPanel, compression: str | None) -> bytes:
    keys = list(obj.fits)
    header = {
        "by": list(obj.by),
        "keys": [[_tag(part) for part in key] for key in keys],
        # `errors` is payload, not an exception. `on_error="skip"` doubles as the
        # cohort screen, so WHICH cohorts failed and why is a result of the fit
        # and has to survive the wire.
        "error_keys": [[_tag(part) for part in key] for key in obj.errors],
        "error_messages": list(obj.errors.values()),
    }
    # Cohorts are addressed by position, not by a joined string: a key is a tuple
    # of arbitrary values and any separator chosen here would be a value some
    # company code eventually contains.
    nested = {str(i): _encode_mack_fit(obj.fits[key], compression) for i, key in enumerate(keys)}
    return _pack("MackFitPanel", pa.table({}), header, nested=nested, compression=compression)


def _decode_mack_panel(body, header: dict, frames, arrays, nested: dict) -> MackFitPanel:
    keys = [tuple(_untag(part) for part in key) for key in header["keys"]]
    fits = {key: from_arrow(nested[str(i)]) for i, key in enumerate(keys)}
    errors = {
        tuple(_untag(part) for part in key): message
        for key, message in zip(header["error_keys"], header["error_messages"], strict=True)
    }
    return MackFitPanel(fits=fits, errors=errors, by=tuple(header["by"]))


# -- CDRResult -----------------------------------------------------------------


def _encode_cdr(obj: CDRResult, compression: str | None) -> bytes:
    body = pa.table({"ibnr": obj.ibnr, "msep": obj.msep, "runoff_msep": obj.runoff_msep})
    header = {
        "origin_periods": [_tag(p) for p in obj.origin_periods],
        "method": obj.method,
        "units": obj.units,
    }
    # The two totals are shape-() arrays rather than header scalars because a
    # msep is a variance that a degenerate cohort can legitimately leave NaN,
    # and the header refuses non-finite floats by design.
    arrays = {
        "msep_total": np.asarray(obj.msep_total, dtype=float),
        "runoff_msep_total": np.asarray(obj.runoff_msep_total, dtype=float),
    }
    return _pack("CDRResult", body, header, arrays=arrays, compression=compression)


def _decode_cdr(body: pa.Table, header: dict, frames, arrays, nested) -> CDRResult:
    def column(name: str) -> np.ndarray:
        return body.column(name).to_numpy(zero_copy_only=False)

    return CDRResult(
        origin_periods=[_untag(p) for p in header["origin_periods"]],
        ibnr=column("ibnr"),
        msep=column("msep"),
        msep_total=float(arrays["msep_total"]),
        runoff_msep=column("runoff_msep"),
        runoff_msep_total=float(arrays["runoff_msep_total"]),
        method=header["method"],
        units=header["units"],
    )


# -- ForecastPanel -------------------------------------------------------------

#: Every frame a ``ForecastPanel`` carries. The encoder, the decoder and the
#: round-trip test all read this one tuple, so a frame added to the panel and not
#: to this list fails loudly at encode rather than going missing on the wire.
_PANEL_FRAMES = (
    "cells",
    "pointwise",
    "by_cohort",
    "coverage",
    "absences",
    "dropped",
    "excluded",
)


def _encode_forecast_panel(obj: ForecastPanel, compression: str | None) -> bytes:
    header = {
        "task": obj.task,
        "as_of": obj.as_of.isoformat(),
        "segments": list(obj.segments),
        "units": obj.units,
        # The member lists and fingerprints are not decoration: a score is only
        # meaningful with the set of models the panel was intersected over, so an
        # encoding that dropped them would ship a number nobody can reproduce.
        "elpd_members": list(obj.elpd_members),
        "crps_members": list(obj.crps_members),
        "elpd_fingerprint": obj.elpd_fingerprint,
        "crps_fingerprint": obj.crps_fingerprint,
    }
    frames = {name: getattr(obj, name) for name in _PANEL_FRAMES}
    return _pack("ForecastPanel", pa.table({}), header, frames=frames, compression=compression)


def _decode_forecast_panel(body, header: dict, frames: dict, arrays, nested) -> ForecastPanel:
    return ForecastPanel(
        task=header["task"],
        as_of=dt.date.fromisoformat(header["as_of"]),
        segments=tuple(header["segments"]),
        units=header["units"],
        elpd_members=tuple(header["elpd_members"]),
        crps_members=tuple(header["crps_members"]),
        elpd_fingerprint=header["elpd_fingerprint"],
        crps_fingerprint=header["crps_fingerprint"],
        **{name: frames[name] for name in _PANEL_FRAMES},
    )


# -- DataFrame (the leaderboard, and anything else tabular) --------------------


def _encode_frame(obj: pd.DataFrame, compression: str | None) -> bytes:
    # No coercion to float64 on the way out. The leaderboard's nullable Float64
    # and Int64 columns are the whole missing-vs-zero-density distinction: a
    # missing score is pd.NA and a model that gave the outcome zero density is
    # -inf, and float64 collapses the first into NaN and makes them one column
    # of "not a number".
    return _pack(
        "DataFrame",
        pa.table({}),
        {"name": obj.attrs.get("name")},
        frames={"frame": obj},
        compression=compression,
    )


def _decode_frame(body, header: dict, frames: dict, arrays, nested) -> pd.DataFrame:
    frame = frames["frame"]
    if header.get("name") is not None:
        frame.attrs["name"] = header["name"]
    return frame


# -- dispatch ------------------------------------------------------------------

_ENCODERS: dict[type, tuple[str, Any]] = {
    PredictiveDistribution: ("PredictiveDistribution", _encode_predictive),
    Triangle: ("Triangle", _encode_triangle),
    MackFit: ("MackFit", _encode_mack_fit),
    MackFitPanel: ("MackFitPanel", _encode_mack_panel),
    CDRResult: ("CDRResult", _encode_cdr),
    ForecastPanel: ("ForecastPanel", _encode_forecast_panel),
    pd.DataFrame: ("DataFrame", _encode_frame),
}

_DECODERS: dict[str, Any] = {
    "PredictiveDistribution": _decode_predictive,
    "Triangle": _decode_triangle,
    "MackFit": _decode_mack_fit,
    "MackFitPanel": _decode_mack_panel,
    "CDRResult": _decode_cdr,
    "ForecastPanel": _decode_forecast_panel,
    "DataFrame": _decode_frame,
}


def _encoder_for(obj):
    for cls in type(obj).__mro__:
        if cls in _ENCODERS:
            return _ENCODERS[cls]
    raise TypeError(
        f"no wire encoding for {type(obj).__name__}; encodable types are "
        f"{sorted(kind for kind, _ in _ENCODERS.values())}"
    )


def to_arrow(obj, *, compression: str | None = None) -> bytes:
    """Serialize one ibnr result to an Arrow IPC stream.

    ``compression`` is ``None``, ``"lz4"`` or ``"zstd"``; anything else is
    refused by pyarrow. Leave it alone unless the link is the bottleneck: on
    posterior draws zstd buys 7.5% of the size for about three times the encode
    cost, and lz4 comes out larger than the uncompressed envelope.
    """
    _, encode = _encoder_for(obj)
    return encode(obj, compression)


def from_arrow(data: bytes, **kwargs):
    """Reconstruct whatever :func:`to_arrow` wrote, dispatching on ``ibnr.kind``.

    ``kwargs`` reach the per-kind decoder; today only ``Triangle`` takes one
    (``backend=``).
    """
    kind, body, header, frames, arrays, nested = _unpack(data)
    if kind not in _DECODERS:
        raise ValueError(f"unknown envelope kind {kind!r}; known kinds are {sorted(_DECODERS)}")
    return _DECODERS[kind](body, header, frames, arrays, nested, **kwargs)


def peek_kind(data: bytes) -> str:
    """The artifact type, read from the stream's schema message.

    What this saves is the DECODE, not the read. A router can dispatch a payload
    it is about to hand somewhere else without ever reconstructing the pandas
    frames and numpy arrays inside it - but it is not a cheap peek at a small
    prefix, and how little it reads depends entirely on the kind. For
    ``PredictiveDistribution``, ``MackFit``, ``CDRResult`` and ``Triangle`` the
    payload sits in record batches AFTER the schema, so this touches almost none
    of it. For ``DataFrame``, ``ForecastPanel`` and ``MackFitPanel`` the body
    table is empty and the entire payload rides in the schema metadata as nested
    streams, so this reads 100% of the bytes - it simply does not unpack any of
    them.
    """
    with pa.ipc.open_stream(pa.py_buffer(data)) as reader:
        metadata = reader.schema.metadata or {}
    if _KIND not in metadata:
        raise ValueError("not an ibnr envelope: the Arrow schema carries no ibnr.kind")
    return metadata[_KIND].decode("utf-8")


# -- summary mode --------------------------------------------------------------


def to_summary(obj, *, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> dict:
    """A JSON-safe digest for callers that cannot take the draws.

    3,571 B against 802,888 B for a 10-target, 10,000-draw distribution - a 225x
    reduction, and the reason this is a negotiated mode rather than a size
    threshold: whether a caller gets a distribution or a description of one is
    the caller's decision to make explicitly.

    There is deliberately **no** ``from_summary``. A summary is not a
    distribution and must not be reconstructible into one; that is the whole
    reason ``PredictiveDistribution`` exists (CLAUDE.md decision 4). A caller who
    needs draws asks for Arrow.
    """
    if not isinstance(obj, PredictiveDistribution):
        raise TypeError(
            f"no summary form for {type(obj).__name__}; only PredictiveDistribution "
            "has a draws payload worth summarizing - everything else is already small "
            "enough to send whole"
        )
    levels = [float(q) for q in quantiles]
    if obj.n_draws == 0:
        mean = sd = [None] * obj.n_targets
        grid = [[None] * len(levels) for _ in range(obj.n_targets)]
    else:
        with np.errstate(invalid="ignore"):
            mean = [_finite(v) for v in obj.mean()]
            sd = [_finite(v) for v in obj.std()]
            # np.quantile returns (n_levels, n_targets); the summary is read per
            # target, so it is transposed here rather than by every client.
            q = np.quantile(obj.samples, levels, axis=0) if levels else np.empty((0, obj.n_targets))
        grid = [[_finite(v) for v in row] for row in q.T]
    return {
        "kind": "PredictiveDistribution",
        "version": CODEC_VERSION,
        "units": obj.units,
        "n_draws": obj.n_draws,
        "targets": [
            {name: _json_scalar(value) for name, value in row.items()}
            for row in obj.targets.to_dict("records")
        ],
        "mean": mean,
        "sd": sd,
        "quantile_levels": levels,
        "quantiles": grid,
    }


def _finite(value) -> float | None:
    """Non-finite floats become ``null``.

    Strict JSON has no NaN and no Infinity, and ``json.dumps`` emits the bare
    tokens ``NaN`` / ``Infinity`` by default - output that parses in Python and
    in almost nothing else. Every emitter here passes ``allow_nan=False`` so
    that cannot happen quietly; this is what makes that possible.
    """
    value = float(value)
    return value if np.isfinite(value) else None


def _json_scalar(value):
    if _is_missing(value):
        return None
    if isinstance(value, dt.datetime | pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, np.floating | float):
        return _finite(value)
    return str(value)
