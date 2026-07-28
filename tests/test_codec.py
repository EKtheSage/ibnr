"""kernels.codec: the wire format every ibnr result crosses a process boundary in.

The failure mode this module has to be tested against is not "the bytes are
wrong" - a broken codec here returns finite numbers of the right magnitude in a
frame of the right shape. float32 draws are correct to seven digits. Thinned
draws have the right mean. An object column of ``pd.NA`` decoded as float64 NaN
looks identical in a printed frame. So the assertions below are deliberately
sharper than "the values match":

* draws are compared through ``.view(np.uint8)``, because ``assert_allclose``
  passes cleanly against a float32 encoder and is therefore worthless here;
* ``pd.NA`` is asserted to still BE ``pd.NA``, not merely missing, because on the
  leaderboard NaN and NA are two different verdicts (CLAUDE.md milestone 6: a
  missing score is ``pd.NA``, never 0.0, which is simultaneously the best ELPD
  and the best CRPS there is);
* a tuple key is asserted to still be a ``tuple``, because a decoded ndarray has
  the same contents and is unhashable, so the panel raises on first groupby
  rather than at decode;
* the payload has a size ceiling, which is what catches an accidental fall back
  to JSON (1,952,983 B) or to dictionary-encoded parquet (978,363 B) on a body
  that should be 802,888 B.

Each of those was checked by breaking the encoder on purpose and confirming the
test goes red - a value-only version of any of them stays green.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import warnings

import ibis
import numpy as np
import pandas as pd
import pytest

from ibnr.kernels import codec
from ibnr.kernels.cdr import CDRResult, one_year_cdr
from ibnr.kernels.forecast import (
    Absence,
    CohortForecast,
    ForecastPanel,
    align_panel,
    leaderboard,
)
from ibnr.kernels.holdout import next_diagonal
from ibnr.kernels.mack import MackFit, MackFitPanel, fit_mack, fit_mack_many
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

# A run-off staircase with a ZERO first cell on the second origin. That zero is
# the positivity contract in miniature: it counts towards f[0] (which needs only
# a positive column sum) and not towards sigma2[0] (whose weighted residual
# divides by the cell), so n_obs and n_pos legitimately differ and the codec has
# to carry both.
CUM = np.array(
    [
        [1000.0, 1500.0, 1750.0, 1800.0],
        [0.0, 1600.0, 1900.0, np.nan],
        [1200.0, 1750.0, np.nan, np.nan],
        [1300.0, np.nan, np.nan, np.nan],
    ]
)

TASK = "paid_next_diagonal_v1"

#: Every kind ``to_arrow`` can write, read off the encoder registry rather than
#: listed by hand. A kind that joins the registry joins the compression
#: parametrization below in the same commit - which is precisely what did not
#: happen for ``Triangle``, whose encoder took ``compression`` and dropped it.
ENCODABLE_KINDS = sorted(kind for kind, _ in codec._ENCODERS.values())


def _rows(company: str, cum: np.ndarray = CUM, start: int = 2010) -> list[dict]:
    out = []
    n_w, n_d = cum.shape
    for i in range(n_w):
        for j in range(n_d):
            if np.isnan(cum[i, j]):
                continue
            out.append(
                {
                    "company": company,
                    "origin_period": dt.date(start + i, 1, 1),
                    "dev_lag": 12 * (j + 1),
                    "eval_date": dt.date(start + i + j, 12, 31),
                    "field": "paid_loss",
                    "value": float(cum[i, j]),
                }
            )
    return out


@pytest.fixture
def pred() -> PredictiveDistribution:
    """The realistic API payload: 10 accident years, 10,000 draws, 800,000 B raw.

    Target metadata deliberately mixes the three dtypes a real ``targets`` frame
    carries - object-dtype ``datetime.date``, float64 premium, object-dtype
    label - because each takes a different Arrow path back.
    """
    rng = np.random.default_rng(7)
    samples = rng.normal(1e6, 1e5, (10_000, 10))
    targets = pd.DataFrame(
        {
            "origin_period": [dt.date(1988 + i, 1, 1) for i in range(10)],
            "premium": rng.uniform(1e5, 1e6, 10),
            "label": [f"AY{1988 + i}" for i in range(10)],
        }
    )
    return PredictiveDistribution(samples=samples, targets=targets, units="usd")


@pytest.fixture
def triangle(backend_name) -> Triangle:
    return Triangle.from_long(
        pd.DataFrame(_rows("CO_A")), measure="cumulative", units="usd", backend=backend_name
    )


@pytest.fixture
def panel_and_board():
    """A two-model ``ForecastPanel``, one of which has draws but no density.

    That asymmetry is the point: it is what puts ``pd.NA`` in the board's ELPD
    column and in ``by_cohort``, and an object column mixing values with
    ``pd.NA`` is exactly what Arrow flattens to float64 NaN if left alone.
    """
    tri = Triangle.from_long(pd.DataFrame(_rows("CO_A")), measure="cumulative")
    cells = next_diagonal(tri, as_of="2012-12-31", fields=["paid_loss"])
    rng = np.random.default_rng(3)
    scored = CohortForecast(
        model="m_scored",
        task=TASK,
        cells=cells,
        field="paid_loss",
        log_density=rng.normal(-np.log(cells.values) - 2.0, 0.3, size=(50, cells.n_cells)),
        draws=np.exp(rng.normal(np.log(cells.values), 0.25, size=(50, cells.n_cells))),
    )
    draws_only = CohortForecast(
        model="m_draws_only",
        task=TASK,
        cells=cells,
        field="paid_loss",
        log_density=None,
        density_absence=Absence("no_predictive_density", "quasi-likelihood"),
        draws=np.exp(rng.normal(np.log(cells.values), 0.4, size=(50, cells.n_cells))),
    )
    forecast_panel = align_panel([scored, draws_only])
    return forecast_panel, leaderboard(forecast_panel)


@pytest.fixture
def encodables(pred, panel_and_board) -> dict:
    """One artifact of every registered kind, keyed by its kind string.

    Deliberately on the default backend rather than through the ``backend_name``
    parameterization: compression is a property of the bytes and nothing about it
    is backend-specific, so a cross product here would double the runtime to
    measure the same thing twice.
    """
    forecast_panel, board = panel_and_board
    tri = Triangle.from_long(pd.DataFrame(_rows("CO_A")), measure="cumulative", units="usd")
    fit = fit_mack(tri, loss_field="paid_loss")
    bad = CUM.copy()
    bad[:, 0] = 0.0  # a zero-volume first step, so the panel carries an error as well as a fit
    both = Triangle.from_long(
        pd.DataFrame(_rows("CO_A") + _rows("CO_B", cum=bad)), measure="cumulative"
    )
    return {
        "PredictiveDistribution": pred,
        "Triangle": tri,
        "MackFit": fit,
        "MackFitPanel": fit_mack_many(both, loss_field="paid_loss", on_error="skip"),
        "CDRResult": one_year_cdr(fit),
        "ForecastPanel": forecast_panel,
        "DataFrame": board,
    }


def _identity(obj) -> bytes:
    """A kind-agnostic identity for a decoded artifact: the bytes it encodes to
    uncompressed.

    Comparing encodings rather than fields is what lets one assertion serve all
    seven kinds, including the two (``MackFitPanel``, ``ForecastPanel``) whose
    entire payload rides in nested streams. A triangle is normalized to a sorted
    frame first: its rows are the artifact, its row ORDER belongs to the engine,
    and no backend promises one.
    """
    if isinstance(obj, Triangle):
        frame = obj.to_pandas()
        obj = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    return codec.to_arrow(obj)


def _retag(data: bytes, key: bytes, value: bytes) -> bytes:
    """Rewrite one envelope metadata field, to forge input decode must refuse."""
    table = codec._read_stream(data)
    metadata = dict(table.schema.metadata)
    metadata[key] = value
    return codec._write_stream(table.replace_schema_metadata(metadata), None)


# -- PredictiveDistribution ----------------------------------------------------


def test_draws_survive_bit_for_bit(pred):
    """Every draw comes back with an identical bit pattern.

    Compared as raw bytes, not as numbers. The lossy encodings that would be
    tempting here all pass an approximate check: float32 halves the payload for
    a maximum relative error of 6e-8 (the float32 rounding bound), and 10x
    thinning cuts it by 90% while moving the median only 0.7%. What they cost is
    the tail - over 200 seeds the thinned 99.5th percentile drifts a median
    2.2%, three times as far, and the 99.5th percentile is what a reserve answer
    is read off.
    """
    back = PredictiveDistribution.from_arrow(pred.to_arrow())
    assert back.samples.dtype == pred.samples.dtype
    assert back.samples.shape == pred.samples.shape
    assert np.array_equal(back.samples.view(np.uint8), pred.samples.view(np.uint8))
    assert back.units == pred.units


def test_targets_frame_survives_including_its_dtypes():
    """``targets`` is metadata, and metadata that changes type is metadata that
    lies. A date must not come back a Timestamp, a Categorical must not come back
    a string column, and a nullable Int64's NA must not become a float NaN."""
    targets = pd.DataFrame(
        {
            "origin_period": [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            "lob": pd.Categorical(["wkcomp", "othliab"]),
            "n_claims": pd.array([17, None], dtype="Int64"),
            "label": ["a", "b"],
        }
    )
    pred = PredictiveDistribution(samples=np.ones((4, 2)), targets=targets)
    back = PredictiveDistribution.from_arrow(pred.to_arrow())
    pd.testing.assert_frame_equal(back.targets, pred.targets)
    assert isinstance(back.targets["origin_period"].iloc[0], dt.date)
    assert back.targets["n_claims"].iloc[1] is pd.NA


def test_nonfinite_draws_survive(pred):
    """NaN and both infinities are legitimate draw values - a divergent chain,
    or a zero-density cell whose log density is ``-inf`` - and they are exactly
    what strict JSON cannot express at all."""
    samples = np.array([[np.nan, np.inf], [-np.inf, 1.0], [0.0, -0.0]])
    dist = PredictiveDistribution(samples=samples, targets=pd.DataFrame({"label": ["a", "b"]}))
    back = PredictiveDistribution.from_arrow(dist.to_arrow())
    assert np.array_equal(back.samples.view(np.uint8), samples.view(np.uint8))
    assert np.isnan(back.samples[0, 0])
    assert back.samples[0, 1] == np.inf
    assert back.samples[1, 0] == -np.inf


def test_degenerate_shapes_round_trip():
    """A distribution with no targets encodes to a zero-column body, from which
    the draw count cannot be read - which is why every shape is written into the
    header rather than inferred from the table."""
    empty = PredictiveDistribution(samples=np.zeros((7, 0)), targets=pd.DataFrame(index=range(0)))
    back = PredictiveDistribution.from_arrow(empty.to_arrow())
    assert (back.n_draws, back.n_targets) == (7, 0)

    single = PredictiveDistribution(samples=np.array([[42.5]]), targets=pd.DataFrame({"l": ["x"]}))
    back = PredictiveDistribution.from_arrow(single.to_arrow())
    assert back.samples.tolist() == [[42.5]]


def test_payload_stays_close_to_the_raw_bytes(pred):
    """Size guard rail. 10 targets x 10,000 draws is 800,000 B of float64 and
    Arrow IPC adds 0.4% of framing, for 802,888 B. The ceiling is set just above
    that so an accidental fall back to another codec fails here: JSON measured
    1,952,983 B and parquet at its defaults 978,363 B, because it attempts
    dictionary encoding on doubles that are all distinct."""
    assert pred.samples.nbytes == 800_000
    assert len(pred.to_arrow()) < 900_000


# -- compression ---------------------------------------------------------------


def test_every_registered_kind_has_a_fixture(encodables):
    """The parametrization below is only as complete as this mapping, so a kind
    added to the encoder registry without a fixture fails here rather than
    quietly not being tested at all."""
    assert sorted(encodables) == ENCODABLE_KINDS


@pytest.mark.parametrize("kind", ENCODABLE_KINDS)
@pytest.mark.parametrize("compression", ["lz4", "zstd"])
def test_compression_changes_the_bytes_and_not_the_answer(encodables, kind, compression):
    """Compression is a knob on EVERY kind, and a knob has to turn something.

    Parametrized over the whole registry rather than over the one kind whose
    payload is big enough to be interesting, because the defect this pins is not
    "the bytes are the wrong size" - it is an encoder that accepts ``compression``
    and never forwards it. Nothing about the decoded answer can see that: the
    round trip is perfect and the argument simply does nothing. ``Triangle`` did
    exactly this, and plain, lz4 and zstd all came out byte-identical.
    """
    obj = encodables[kind]
    plain = codec.to_arrow(obj)
    squeezed = codec.to_arrow(obj, compression=compression)
    assert plain != squeezed, f"compression={compression!r} never reached the {kind} encoder"
    assert _identity(codec.from_arrow(plain)) == _identity(codec.from_arrow(squeezed))


@pytest.mark.parametrize("kind", ENCODABLE_KINDS)
def test_an_unknown_codec_is_refused(encodables, kind):
    """The other half of the same wire. An argument that reaches pyarrow is one
    pyarrow can reject by name; an argument that is dropped on the way accepts
    every string there is, including a typo for ``zstd``."""
    with pytest.raises(ValueError, match="compression"):
        codec.to_arrow(encodables[kind], compression="not_a_codec")


def test_compression_is_a_knob_and_never_the_default(pred):
    """Why the default is ``None``, on the only payload here big enough for a
    ratio to mean anything. Posterior mantissas are high-entropy noise: zstd
    takes 7.5% off the size for about three times the encode cost, and lz4 comes
    out LARGER than both the uncompressed envelope and the raw draws."""
    plain = pred.to_arrow()
    assert len(pred.to_arrow(compression="zstd")) < len(plain)
    assert len(pred.to_arrow(compression="lz4")) > len(plain) > pred.samples.nbytes


# -- summary mode --------------------------------------------------------------


def test_summary_is_strict_json_and_far_smaller(pred):
    """The negotiated JSON mode. ~3.5 KB against ~800 KB, and it must survive
    ``allow_nan=False``: ``json.dumps`` emits the bare tokens ``NaN`` and
    ``Infinity`` by default, which are not JSON and which a non-Python client
    rejects."""
    summary = pred.to_summary()
    encoded = json.dumps(summary, allow_nan=False)  # raises if a NaN escaped
    assert len(encoded) < len(pred.to_arrow()) / 100
    assert summary["n_draws"] == 10_000
    assert len(summary["targets"]) == 10
    assert summary["targets"][0]["origin_period"] == "1988-01-01"
    assert len(summary["quantiles"]) == 10  # one row per TARGET, not per level
    assert len(summary["quantiles"][0]) == len(summary["quantile_levels"])
    assert summary["mean"][0] == pytest.approx(1e6, rel=0.01)


def test_summary_nulls_out_what_json_cannot_hold():
    """A non-finite moment becomes ``null`` rather than a token no other language
    parses. One draw makes the sample SD undefined, which is the honest way to
    reach that branch."""
    dist = PredictiveDistribution(
        samples=np.array([[np.inf, 1.0]]), targets=pd.DataFrame({"l": ["a", "b"]})
    )
    with warnings.catch_warnings():  # ddof=1 on one draw; the undefined SD is the point
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = dist.to_summary(quantiles=(0.5,))
    assert summary["mean"] == [None, 1.0]
    assert summary["sd"] == [None, None]  # ddof=1 on a single draw
    json.dumps(summary, allow_nan=False)


def test_the_requested_quantile_grid_is_the_one_reported(pred):
    """``quantiles=`` has to reach ``np.quantile``, not just be accepted.

    The default grid has eleven levels, so a method that took the argument and
    dropped it would return a perfectly valid summary with the wrong levels in
    it - and a test that only read ``mean`` and ``sd`` would never notice.
    Asserted against the default here so the two cannot be confused.
    """
    assert len(codec.DEFAULT_QUANTILES) > 2, "the default has to differ from the request"

    summary = pred.to_summary(quantiles=(0.25, 0.75))
    assert summary["quantile_levels"] == [0.25, 0.75]
    assert [len(row) for row in summary["quantiles"]] == [2] * pred.n_targets
    # and the values are the requested levels, not the first two default ones
    expected = np.quantile(pred.samples, [0.25, 0.75], axis=0).T
    np.testing.assert_allclose(summary["quantiles"], expected)

    assert pred.to_summary()["quantile_levels"] == list(codec.DEFAULT_QUANTILES)


def test_a_bad_quantile_grid_is_refused_whether_or_not_there_are_draws():
    """The zero-draw branch never calls ``np.quantile``, so it used to be the one
    place an impossible grid was accepted.

    A distribution with no draws is not a corrupt object - it is what a refused
    cohort produces - and it answers with a grid of nulls, one per level. Asking
    it for the 150th percentile therefore returned a perfectly well-formed
    summary whose ``quantile_levels`` said 1.5, while the identical call against
    a distribution WITH draws raised from inside numpy. Same request, two
    verdicts, decided by the data rather than the request.
    """
    targets = pd.DataFrame({"l": ["a", "b"]})
    empty = PredictiveDistribution(samples=np.zeros((0, 2)), targets=targets)
    filled = PredictiveDistribution(samples=np.ones((5, 2)), targets=targets)
    assert empty.n_draws == 0 and filled.n_draws == 5

    for dist in (empty, filled):
        for grid in [(0.5, 1.5), (-0.01,), (float("nan"),)]:
            with pytest.raises(ValueError, match=r"quantile levels must lie in \[0, 1\]"):
                dist.to_summary(quantiles=grid)

    # and the legitimate empty summary still answers, one null per level
    summary = empty.to_summary(quantiles=(0.05, 0.5))
    assert summary["quantile_levels"] == [0.05, 0.5]
    assert summary["quantiles"] == [[None, None], [None, None]]
    assert summary["mean"] == [None, None]
    json.dumps(summary, allow_nan=False)


def test_there_is_no_from_summary():
    """A summary is a description of a distribution, not a distribution. Making
    it reconstructible would let a caller score, stack or rank on eleven
    quantiles while believing they held draws - which is the whole reason
    ``PredictiveDistribution`` is the unifying output type (CLAUDE.md #4)."""
    assert not hasattr(PredictiveDistribution, "from_summary")
    assert not hasattr(codec, "from_summary")


def test_summary_refuses_types_that_have_no_draws(triangle):
    assert codec.to_summary.__doc__  # the refusal is documented, not incidental
    with pytest.raises(TypeError, match="no summary form"):
        codec.to_summary(triangle)


# -- routing and refusals ------------------------------------------------------


def test_peek_kind_reads_only_the_schema(pred, triangle, panel_and_board):
    """A router dispatches on the kind without paying to decode a body it is
    about to hand somewhere else."""
    forecast_panel, board = panel_and_board
    fit = fit_mack(triangle, loss_field="paid_loss")
    for obj, kind in [
        (pred, "PredictiveDistribution"),
        (triangle, "Triangle"),
        (fit, "MackFit"),
        (one_year_cdr(fit), "CDRResult"),
        (forecast_panel, "ForecastPanel"),
        (board, "DataFrame"),
    ]:
        assert codec.peek_kind(codec.to_arrow(obj)) == kind


def test_decode_refuses_an_unknown_kind(pred):
    forged = _retag(pred.to_arrow(), b"ibnr.kind", b"SomethingElse")
    assert codec.peek_kind(forged) == "SomethingElse"
    with pytest.raises(ValueError, match="unknown envelope kind"):
        codec.from_arrow(forged)


def test_decode_refuses_a_future_version(pred):
    """Forward-incompatible input fails loudly. A payload written by a newer
    codec may put different things in the header, and decoding the parts this
    build happens to recognize would return a wrong answer with no error on it."""
    forged = _retag(pred.to_arrow(), b"ibnr.version", str(codec.CODEC_VERSION + 1).encode())
    with pytest.raises(ValueError, match="codec version"):
        codec.from_arrow(forged)


def test_decode_refuses_a_foreign_arrow_stream():
    """A plain Arrow stream is not an ibnr envelope, and the failure must say so
    rather than raising a KeyError from somewhere in the middle."""
    stream = codec._write_stream(pa_table_of_ones(), None)
    with pytest.raises(ValueError, match="no ibnr.kind"):
        codec.peek_kind(stream)
    with pytest.raises(ValueError, match="no ibnr.kind"):
        codec.from_arrow(stream)


def pa_table_of_ones():
    import pyarrow as pa

    return pa.table({"x": [1.0, 2.0]})


def test_unencodable_type_is_named():
    with pytest.raises(TypeError, match="no wire encoding for"):
        codec.to_arrow(object())


#: The typed entry point for every kind that has one, read back off the classes
#: themselves. ``DataFrame`` is deliberately absent and named separately below:
#: the type belongs to pandas, so ``codec.from_arrow`` is the only door it has.
TYPED_DECODERS = {
    "PredictiveDistribution": PredictiveDistribution.from_arrow,
    "Triangle": Triangle.from_arrow,
    "MackFit": MackFit.from_arrow,
    "MackFitPanel": MackFitPanel.from_arrow,
    "CDRResult": CDRResult.from_arrow,
    "ForecastPanel": ForecastPanel.from_arrow,
}


def test_every_kind_has_a_typed_decoder_or_is_named_as_having_none():
    """Stale-duplicate guard. A kind joining the registry joins the wrong-kind
    parametrization below in the same commit, rather than being the one entry
    point nobody checked."""
    assert set(TYPED_DECODERS) | {"DataFrame"} == set(codec._DECODERS)


@pytest.mark.parametrize("kind", sorted(TYPED_DECODERS))
def test_a_typed_decoder_refuses_every_other_kind(encodables, kind):
    """``PredictiveDistribution.from_arrow(triangle_bytes)`` must not return a
    Triangle.

    The typed classmethods are thin delegations to the generic dispatcher, which
    routes on the payload's own ``ibnr.kind`` - so without a check the class the
    caller named is decoration, and the value they get back is whatever the bytes
    happened to be. Nothing downstream necessarily notices: it is a real,
    correctly decoded artifact of the wrong type, and the AttributeError lands
    somewhere else entirely.

    Both names are asserted to be in the message, because "this is not a
    MackFit" without saying what it IS sends the reader back to the wire.
    """
    decode = TYPED_DECODERS[kind]
    assert decode(codec.to_arrow(encodables[kind])) is not None

    for other in ENCODABLE_KINDS:
        if other == kind:
            continue
        with pytest.raises(ValueError, match="expected an envelope of kind") as excinfo:
            decode(codec.to_arrow(encodables[other]))
        message = str(excinfo.value)
        assert kind in message and other in message, message


def test_the_expected_kind_is_itself_checked(pred):
    """A typo in ``expect=`` refuses every payload there is, which is a confusing
    way to learn about a typo. It is refused by name instead, before the bytes
    are touched."""
    with pytest.raises(ValueError, match="unknown expected kind 'MackFitt'"):
        codec.from_arrow(pred.to_arrow(), expect="MackFitt")


# -- Triangle ------------------------------------------------------------------


def test_triangle_round_trips_on_both_backends(triangle, backend_name):
    """A triangle has to make the trip in BOTH directions or the API is
    write-only. Decode goes back through ``io.from_long``, the single ingestion
    path, so no second parser exists to drift from the first.

    ``backend`` is asserted on the RESULT, not merely passed. It is the one
    decode argument this codec takes, and a decoder that accepted it and ignored
    it would put every triangle on duckdb while this test still went green on
    both parameters - the whole point of the argument being decode-time is that
    the caller, not the payload, chooses the engine.
    """
    back = Triangle.from_arrow(triangle.to_arrow(), backend=backend_name)
    assert ibis.get_backend(back.expr).name == backend_name
    assert back.meta == triangle.meta
    assert back.segments == triangle.segments
    left = triangle.to_pandas().sort_values(["origin_period", "dev_lag"]).reset_index(drop=True)
    right = back.to_pandas().sort_values(["origin_period", "dev_lag"]).reset_index(drop=True)
    assert len(left) == len(right)
    np.testing.assert_array_equal(left["value"].to_numpy(), right["value"].to_numpy())
    assert list(left["dev_lag"]) == list(right["dev_lag"])
    assert [str(v) for v in left["origin_period"]] == [str(v) for v in right["origin_period"]]


def test_triangle_meta_is_not_defaulted_on_decode(backend_name):
    """Grain, measure and units travel with the data. Decoding them from the
    class defaults would silently relabel an incremental quarterly triangle as
    cumulative annual, and every transform downstream trusts that label."""
    rows = pd.DataFrame(_rows("CO_A"))
    tri = Triangle.from_long(
        rows,
        measure="incremental",
        origin_grain="Q",
        dev_grain="Q",
        units="thousands",
        backend=backend_name,
    )
    back = Triangle.from_arrow(tri.to_arrow(), backend=backend_name)
    assert back.meta.measure == "incremental"
    assert back.meta.grain == "OQDQ"
    assert back.meta.units == "thousands"


def test_triangle_decode_still_refuses_a_null_segment(backend_name):
    """The null-segment guard is at ingestion, and decode goes through
    ingestion - so a triangle arriving over the wire gets the same refusal as one
    built from a frame. Forged directly, because ``from_long`` will not build the
    offending Triangle in the first place."""
    import pyarrow as pa

    frame = pd.DataFrame(_rows("CO_A"))
    frame.loc[0, "company"] = None
    forged = codec._pack(
        "Triangle",
        pa.Table.from_pandas(frame, preserve_index=False),
        {
            "origin_grain": "Y",
            "dev_grain": "Y",
            "measure": "cumulative",
            "units": None,
            "segments": ["company"],
        },
    )
    with pytest.raises(ValueError, match="null segment key"):
        Triangle.from_arrow(forged, backend=backend_name)


# -- MackFit / MackFitPanel ----------------------------------------------------


def test_mack_fit_round_trip_keeps_both_origin_counts(triangle):
    """``n_obs`` and ``n_pos`` are different counts by design - the factor is
    estimated over every pair origin, the sigma only over origins with a positive
    cumulative - and the fixture has a zero cell so they actually differ here.
    Encoding one and deriving the other would lose the positivity contract."""
    fit = fit_mack(triangle, loss_field="paid_loss")
    assert fit.n_obs[0] != fit.n_pos[0], "fixture no longer exercises the positivity split"

    back = MackFit.from_arrow(fit.to_arrow())
    assert np.array_equal(back.cum.view(np.uint8), fit.cum.view(np.uint8))
    for name in ("obs_mask", "latest_dev", "f", "sigma2", "s", "n_obs", "n_pos"):
        left, right = getattr(fit, name), getattr(back, name)
        assert right.dtype == left.dtype, name
        assert np.array_equal(right, left), name
    # bool, not float: obs_mask INDEXES with this array, and a float mask indexes
    # by position instead of by predicate.
    assert back.obs_mask.dtype == np.bool_
    assert back.origin_periods == fit.origin_periods
    assert (back.sigma_rule, back.units, back.loss_field) == (
        fit.sigma_rule,
        fit.units,
        fit.loss_field,
    )
    np.testing.assert_array_equal(back.ultimate, fit.ultimate)


def test_mack_panel_carries_the_cohorts_that_failed(backend_name):
    """``errors`` is a result, not an exception: ``on_error='skip'`` doubles as
    the cohort screen, so which cohorts were screened out is part of the answer
    and has to survive the wire."""
    bad = CUM.copy()
    bad[:, 0] = 0.0  # a zero-volume first step - refused by the estimator guards
    frame = pd.DataFrame(_rows("CO_A") + _rows("CO_B", cum=bad))
    tri = Triangle.from_long(frame, measure="cumulative", backend=backend_name)
    panel = fit_mack_many(tri, loss_field="paid_loss", on_error="skip")
    assert list(panel.fits) == [("CO_A",)] and list(panel.errors) == [("CO_B",)]

    back = MackFitPanel.from_arrow(panel.to_arrow())
    assert back.by == panel.by
    assert list(back.fits) == list(panel.fits)
    assert back.errors == panel.errors
    assert np.array_equal(back[("CO_A",)].f, panel[("CO_A",)].f)


def test_mack_fit_cum_keeps_its_own_dtype(triangle):
    """``cum`` is the only array on a ``MackFit`` that used to be rebuilt at a
    hardcoded float64.

    Every other one records its dtype and comes back in it - which is what keeps
    ``obs_mask`` a bool mask rather than a float one - so a float32 triangle
    decoded into a float64 fit that is equal to it, silently 2x the memory and
    no longer the array that was sent. Bit-compared inside the dtype, because
    float32 -> float64 -> float32 is lossless and an ``allclose`` check cannot
    see the trip at all.
    """
    fit = dataclasses.replace(
        fit_mack(triangle, loss_field="paid_loss"), cum=CUM.astype(np.float32)
    )
    back = MackFit.from_arrow(fit.to_arrow())
    assert back.cum.dtype == np.float32
    assert np.array_equal(back.cum.view(np.uint8), fit.cum.view(np.uint8))


def test_a_panel_refuses_a_nested_payload_that_is_not_a_fit(encodables):
    """The panel's children get the same check its own envelope does.

    A ``MackFitPanel`` is the one kind whose payload is other envelopes, and the
    decoder used to hand each nested blob to the generic dispatcher - which
    routes on the blob's OWN kind. So a tampered child decoded into whatever it
    claimed to be and went into ``fits`` under a cohort key, where the panel
    reports it as a fitted cohort and the first ``.ultimate`` fails somewhere
    with no reference to the wire at all.
    """
    panel = encodables["MackFitPanel"]
    assert len(panel.fits) == 1, "fixture no longer has exactly one nested fit"
    forged = _retag(panel.to_arrow(), b"ibnr.nested.0", codec.to_arrow(encodables["Triangle"]))
    with pytest.raises(ValueError, match="expected an envelope of kind 'MackFit'"):
        MackFitPanel.from_arrow(forged)


# -- CDRResult -----------------------------------------------------------------


def test_cdr_result_round_trip(triangle):
    """The two totals ride as shape-() float arrays rather than header scalars,
    because an msep is a variance a degenerate cohort can leave NaN and the
    header refuses non-finite floats by design."""
    result = one_year_cdr(fit_mack(triangle, loss_field="paid_loss"))
    back = codec.from_arrow(result.to_arrow())
    for name in ("ibnr", "msep", "runoff_msep"):
        assert np.array_equal(
            getattr(back, name).view(np.uint8), getattr(result, name).view(np.uint8)
        ), name
    assert back.msep_total == result.msep_total
    assert back.runoff_msep_total == result.runoff_msep_total
    assert back.origin_periods == result.origin_periods
    assert (back.method, back.units) == (result.method, result.units)
    pd.testing.assert_frame_equal(back.summary(), result.summary())


# -- leaderboard and ForecastPanel ---------------------------------------------


def test_leaderboard_keeps_na_distinct_from_a_real_number(panel_and_board):
    """The one distinction this board is built on. A missing score is ``pd.NA``,
    never 0.0 - which on this board is simultaneously the best ELPD and the best
    CRPS there is - and the nullable dtypes are what carry it. Coercing to
    float64 on the way out collapses NA into NaN and makes the two one column of
    "not a number"."""
    _, board = panel_and_board
    missing = board["elpd"].isna().to_numpy()
    assert missing.any() and not missing.all(), "fixture no longer has a model without an ELPD"

    back = codec.from_arrow(codec.to_arrow(board))
    pd.testing.assert_frame_equal(back, board)
    assert back["elpd"].dtype == "Float64"
    assert back["n_cells_zero_density"].dtype == "Int64"
    assert back["elpd"].iloc[int(np.argmax(missing))] is pd.NA
    np.testing.assert_array_equal(back["elpd"].isna().to_numpy(), missing)


def test_neg_inf_and_na_stay_apart_in_one_column():
    """A model that gave the outcome zero density scores ``-inf`` and must rank
    LAST; a model with no density at all scores ``pd.NA`` and does not rank. One
    nullable column has to hold both without merging them."""
    board = pd.DataFrame({"model": ["a", "b", "c"], "elpd": pd.array([-1.5, None, -np.inf])})
    back = codec.from_arrow(codec.to_arrow(board))
    pd.testing.assert_frame_equal(back, board)
    assert back["elpd"].isna().tolist() == [False, True, False]
    assert back["elpd"].iloc[2] == -np.inf


def test_forecast_panel_round_trip(panel_and_board):
    """All seven frames, both member lists and both fingerprints. The membership
    is not decoration - a score is only meaningful with the set it was
    intersected over, so a panel that arrived without it carries a number nobody
    can reproduce."""
    forecast_panel, _ = panel_and_board
    back = codec.from_arrow(forecast_panel.to_arrow())
    # Read off codec._PANEL_FRAMES rather than re-listed here: a hand-copied list
    # goes stale silently, and an eighth frame added to the panel would then be
    # encoded and never checked. Pinned at seven so the constant shrinking is
    # itself a failure.
    assert len(codec._PANEL_FRAMES) == 7
    for name in codec._PANEL_FRAMES:
        # obj=name, not a trailing `, name`: the latter is a two-element tuple
        # expression, not an assert, so the label never reaches any failure.
        pd.testing.assert_frame_equal(
            getattr(back, name), getattr(forecast_panel, name), obj=f"ForecastPanel.{name}"
        )
    assert back.task == forecast_panel.task
    assert back.as_of == forecast_panel.as_of
    assert back.segments == forecast_panel.segments
    assert back.elpd_members == forecast_panel.elpd_members
    assert back.crps_members == forecast_panel.crps_members
    assert back.elpd_fingerprint == forecast_panel.elpd_fingerprint
    assert back.crps_fingerprint == forecast_panel.crps_fingerprint
    # The panel is still USABLE, not merely equal: n_cells_for groups on the
    # segment columns and leaderboard() re-reduces every score from the frames.
    assert back.n_cells_for("elpd") == forecast_panel.n_cells_for("elpd")
    pd.testing.assert_frame_equal(leaderboard(back), leaderboard(forecast_panel))


def test_tuple_keys_come_back_as_tuples(panel_and_board):
    """Arrow has no tuple type and a list column decodes to an ndarray, which has
    identical contents and is unhashable. ``key`` and ``cohort`` are grouped,
    joined and ``isin``-ed on, so a decoded array is not a lossy round trip - it
    is a panel that raises the first time it is used."""
    forecast_panel, _ = panel_and_board
    back = codec.from_arrow(forecast_panel.to_arrow())
    for column in ("key", "cohort"):
        value = back.cells[column].iloc[0]
        assert isinstance(value, tuple), column
        assert value == forecast_panel.cells[column].iloc[0]
        assert hash(value) == hash(forecast_panel.cells[column].iloc[0])
    # mixed types inside one key: task (str), as_of (date), dev_lag (int)
    assert isinstance(back.cells["key"].iloc[0][1], dt.date)


def test_object_column_of_na_does_not_become_nan(panel_and_board):
    """``pointwise.n_draws_density`` is an object column of ints beside
    ``pd.NA`` - the shape Arrow types as int64-with-nulls and pandas hands back
    as float64 NaN, quietly turning "this model has no density" into "not a
    number"."""
    forecast_panel, _ = panel_and_board
    column = forecast_panel.pointwise["n_draws_density"]
    assert column.dtype == object and column.isna().any() and column.notna().any()

    back = codec.from_arrow(forecast_panel.to_arrow())
    decoded = back.pointwise["n_draws_density"]
    assert decoded.dtype == object
    assert [v is pd.NA for v in decoded] == [v is pd.NA for v in column]
    assert [v for v in decoded if v is not pd.NA] == [v for v in column if v is not pd.NA]


# -- plain frames --------------------------------------------------------------


def test_dataframe_index_and_dtypes_survive():
    """A named string index and a nullable column, together."""
    frame = pd.DataFrame(
        {"score": [1.0, 2.0], "n": pd.array([3, None], dtype="Int64")},
        index=pd.Index(["a", "b"], name="model"),
    )
    back = codec.from_arrow(codec.to_arrow(frame))
    pd.testing.assert_frame_equal(back, frame)
    assert back.index.name == "model"


def test_a_nullable_index_keeps_na_apart_from_nan():
    """The index gets the treatment the columns get, because it is data too.

    An ``Int64`` index of ``[3, pd.NA]`` used to come back float64 ``[3.0, nan]``:
    the same missing-versus-value collapse the board's columns are guarded
    against, one axis over. Nothing about the decoded frame looks wrong - the
    values print identically - and it is a different verdict.
    """
    frame = pd.DataFrame(
        {"v": [1.0, 2.0]}, index=pd.Index(pd.array([3, None], dtype="Int64"), name="n")
    )
    back = codec.from_arrow(codec.to_arrow(frame))
    pd.testing.assert_frame_equal(back, frame)
    assert back.index.dtype == "Int64"
    assert back.index[1] is pd.NA


def test_a_tuple_valued_index_survives_as_tuples():
    """A flat object index of tuples - the shape a caller gets from
    ``set_index`` on a cohort key. ``pa.Table.from_pandas`` raises on it
    outright (``Expected bytes, got a 'int' object``), so this was not lossy, it
    was a frame that could not be sent at all."""
    keys = [("wkcomp", 1988), ("othliab", 1989)]
    frame = pd.DataFrame(
        {"v": [1.0, 2.0]}, index=pd.Index(keys, tupleize_cols=False, name="cohort")
    )
    back = codec.from_arrow(codec.to_arrow(frame))
    pd.testing.assert_frame_equal(back, frame)
    assert isinstance(back.index[0], tuple)
    assert back.loc[[("wkcomp", 1988)], "v"].tolist() == [1.0]


def test_a_named_multiindex_survives_with_its_level_types():
    """Levels, names and per-level dtypes. A MultiIndex of (str, date) is what a
    per-cohort board is indexed by, and a date level coming back as a Timestamp
    would silently change every join it takes part in."""
    index = pd.MultiIndex.from_tuples(
        [("wkcomp", dt.date(2020, 1, 1)), ("othliab", dt.date(2021, 1, 1))],
        names=["lob", "origin"],
    )
    frame = pd.DataFrame({"elpd": pd.array([-1.5, None], dtype="Float64")}, index=index)
    back = codec.from_arrow(codec.to_arrow(frame))
    pd.testing.assert_frame_equal(back, frame)
    assert back.index.names == ["lob", "origin"]
    assert isinstance(back.index[0][1], dt.date)
    assert back["elpd"].iloc[1] is pd.NA


def test_an_index_does_not_collide_with_a_column_of_the_same_name():
    """The index travels as an extra column, so its label has to be one no real
    column can hold - including a column that IS the index's name."""
    frame = pd.DataFrame({"v": [1.0, 2.0], "__ibnr_index_0__": [7, 8]}, index=pd.Index([1, 2]))
    frame.index.name = "v"
    back = codec.from_arrow(codec.to_arrow(frame))
    pd.testing.assert_frame_equal(back, frame)
    assert list(back.columns) == ["v", "__ibnr_index_0__"]


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(pd.RangeIndex(2), id="range-default"),
        pytest.param(pd.RangeIndex(5, 7), id="range-offset"),
        pytest.param(pd.RangeIndex(0, 4, 2, name="i"), id="range-step-named"),
        pytest.param(pd.Index([1.5, np.nan], name="f"), id="float-with-nan"),
        pytest.param(pd.CategoricalIndex(["x", "y"], name="c"), id="categorical"),
        pytest.param(pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]), id="multiindex-unnamed"),
        pytest.param(pd.DatetimeIndex(["2020-01-01", "2020-02-01"], name="t"), id="datetime"),
        # The unnamed cases are their own risk, not a duplicate of the named
        # ones: the index rides as a column that HAS a label, and pandas reads
        # an index name off the data when none is given - so an unnamed index
        # comes back named after the placeholder unless that is undone.
        pytest.param(pd.Index(["a", "b"]), id="str-unnamed"),
        pytest.param(pd.Index(pd.array([3, None], dtype="Int64")), id="nullable-unnamed"),
    ],
)
def test_every_index_flavour_round_trips(index):
    """The shapes that already worked, kept working. A RangeIndex is described
    rather than materialized - start/stop/step in the record, no column - so the
    empty-body kinds stay empty."""
    frame = pd.DataFrame({"v": [1.0, 2.0]}, index=index)
    back = codec.from_arrow(codec.to_arrow(frame))
    pd.testing.assert_frame_equal(back, frame, check_index_type=True)
    assert type(back.index) is type(index)
    assert back.index.names == index.names


def test_mixed_object_column_keeps_each_value_its_own_type():
    """A column of an int, a string and a date has no Arrow type at all. Tagging
    values individually is what lets it survive - and survive as three types
    rather than three strings, which is the failure a ``str`` fallback would
    produce and which reads as success."""
    frame = pd.DataFrame({"x": [1, "two", dt.date(2020, 1, 1), np.nan]})
    back = codec.from_arrow(codec.to_arrow(frame))
    pd.testing.assert_frame_equal(back, frame)
    assert [type(v).__name__ for v in back["x"][:3]] == ["int", "str", "date"]


def test_a_value_with_no_encoding_is_named_and_refused():
    """Failing at the boundary IS the contract. Coercing an unknown object to
    its ``repr`` would move the error somewhere it can no longer be attributed to
    the frame that caused it."""
    frame = pd.DataFrame({"x": [{"a", "b"}]})
    with pytest.raises(TypeError, match="cannot serialize"):
        codec.to_arrow(frame)


# -- tagged values -------------------------------------------------------------


def test_an_unknown_tagged_value_kind_is_refused_by_name(triangle):
    """The refuse-by-name contract has a decode side too, and it was open.

    ``_untag`` dispatched on the tag and ended in a bare ``return value``, which
    is right for the four kinds JSON already delivers correctly and is also what
    every OTHER kind fell into. A tag of ``["set", [...]]`` therefore came back
    as a plain list - not an error, a value - and landed in
    ``MackFit.origin_periods``, where the first thing to notice is a report
    labelled with a list. Forged through the header, because the encoder cannot
    be made to write one.
    """
    fit = fit_mack(triangle, loss_field="paid_loss")
    data = fit.to_arrow()
    header = json.loads(codec._read_stream(data).schema.metadata[codec._HEADER])
    header["origin_periods"][0] = ["set", ["1988-01-01"]]
    forged = _retag(data, codec._HEADER, json.dumps(header).encode("utf-8"))

    with pytest.raises(ValueError, match="unknown tagged value kind 'set'"):
        MackFit.from_arrow(forged)


def test_every_tag_the_encoder_writes_is_understood_by_the_decoder():
    """The two halves are two hand-written dispatch tables, and the failure when
    they drift is silent in one direction: a new ``_tag`` kind that ``_untag``
    does not know used to fall through and come back as its JSON shape. Asserted
    as a SET so a kind cannot be added to either side alone."""
    # (sent, received). The two numpy entries are the deliberate normalization:
    # JSON has one integer type and one float type, so a numpy scalar comes back
    # as the python scalar it tags as - equal, one bit for bit, and not the same
    # class. Everything else is expected back as itself.
    cases = [
        (None, None),
        (pd.NA, pd.NA),
        (pd.NaT, pd.NaT),
        (("a", 1, dt.date(2020, 1, 1)), ("a", 1, dt.date(2020, 1, 1))),
        (True, True),
        (np.int64(3), 3),
        (np.float64(1.5), 1.5),
        (float("nan"), float("nan")),
        (pd.Timestamp("2020-01-01T12:00"), pd.Timestamp("2020-01-01T12:00")),
        (dt.datetime(2020, 1, 1, 12, 0), dt.datetime(2020, 1, 1, 12, 0)),
        (dt.date(2020, 1, 1), dt.date(2020, 1, 1)),
        ("text", "text"),
    ]
    assert {codec._tag(sent)[0] for sent, _ in cases} == set(codec._TAG_KINDS)

    for sent, expected in cases:
        back = codec._untag(codec._tag(sent))
        assert type(back) is type(expected), sent
        if isinstance(expected, float) and np.isnan(expected):
            assert np.isnan(back)
        else:
            assert back is expected or back == expected, sent
