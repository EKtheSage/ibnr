"""Load the published Balona--Richman appendix for conventional-method research.

Usage: ``load_published_examples()`` returns complete published rectangles, NOT
valuation-ready upper triangles. Always choose an explicit ``as_of`` before
fitting. Development endpoints are 240 months for Swiss liability and 63 months
for the two quarterly examples; they mean final published development, not a
verified fully settled ultimate or an estimated tail beyond the appendix.

The inspected source does not state a redistribution license. The data therefore
remain a runtime download, cached outside this repository and checked against a
fixed SHA256 before parsing. No third-party data fixture is redistributed.

The appendix gives origin periods and development months, not observation dates.
This adapter assigns each cell the end of its corresponding calendar period.
Premium is stored once, at the origin period's end, as an explicit retrospective
availability convention. In particular, Swiss premiums were simulated using
ultimate losses: dating them earlier does NOT make them an independently observed
historical input. The printed initial/training/future shading is not used to
decide availability. See ``SOURCE_NOTES_URL`` and ``EXAMPLE_METADATA``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from ibnr import Triangle

SOURCE_URL = "https://ibnr.co/research/appendix-data.json"
SOURCE_SHA256 = "7333caea41fecc907dc0c98e46f38c64559cec2158cdacdb4bfc22f604dad455"
SOURCE_DATE = dt.date(2026, 9, 15)
SOURCE_NOTES_URL = "https://ibnr.co/origins#source-notes"
PAPER_URL = "https://ibnr.co/research/balona-richman-2021.pdf"
UNITS = "Source triangle units; currency and magnitude scale unspecified"
PREMIUM_FIELD = "earned_premium"


@dataclass(frozen=True)
class ExampleMetadata:
    """Source identity and fixed calendar/horizon conventions for one example."""

    title: str
    loss_field: str
    origin_grain: str
    dev_grain: str
    first_origin: dt.date
    origin_count: int
    development_periods: int
    claims_table: int
    claims_page: int
    premium_table: int
    premium_page: int
    notes: str

    @property
    def step_months(self) -> int:
        return {"Y": 12, "Q": 3}[self.dev_grain]

    @property
    def horizon_months(self) -> int:
        return self.development_periods * self.step_months


EXAMPLE_METADATA = {
    "swiss": ExampleMetadata(
        "Swiss private liability",
        "paid_loss",
        "Y",
        "Y",
        dt.date(1979, 1, 1),
        19,
        20,
        28,
        52,
        27,
        51,
        "Premiums simulated from ultimate experience and a 60% target loss ratio; "
        "the paper reports a realised weighted-average loss ratio of 59.4%. "
        "These are not independent historical premium observations.",
    ),
    "liability": ExampleMetadata(
        "Quarterly long-tail liability",
        "incurred_loss",
        "Q",
        "Q",
        dt.date(2010, 1, 1),
        20,
        21,
        30,
        54,
        29,
        53,
        "Claims scaled for confidentiality; premiums normalised around a 50% "
        "average loss ratio. Late-cell observation versus extrapolation is unspecified.",
    ),
    "property": ExampleMetadata(
        "Quarterly short-tail property",
        "incurred_loss",
        "Q",
        "Q",
        dt.date(2010, 1, 1),
        20,
        21,
        32,
        56,
        31,
        55,
        "Claims scaled for confidentiality; premiums normalised around a 50% "
        "average loss ratio. Negative increments are retained. Late-cell observation "
        "versus extrapolation is unspecified.",
    ),
}


def _month_start(origin: dt.date, months: int) -> dt.date:
    year, month = divmod(origin.year * 12 + origin.month - 1 + months, 12)
    return dt.date(year, month + 1, 1)


def _verified_bytes(raw: bytes) -> bytes:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(
            f"Published appendix SHA256 mismatch: expected {SOURCE_SHA256}, got {actual}. "
            "Do not use changed data as the pinned benchmark source."
        )
    return raw


def _source_bytes(cache_dir: str | Path | None) -> bytes:
    directory = (
        Path(tempfile.gettempdir()) / "ibnr-published-examples"
        if cache_dir is None
        else Path(cache_dir)
    )
    cached = directory / f"{SOURCE_SHA256}.json"
    if cached.exists():
        return _verified_bytes(cached.read_bytes())
    request = Request(SOURCE_URL, headers={"User-Agent": "ibnr-research-examples/1"})
    try:
        with urlopen(request, timeout=30) as response:
            raw = _verified_bytes(response.read())
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"Could not download the published appendix from {SOURCE_URL}. "
            f"A verified copy may be placed at {cached}."
        ) from exc
    directory.mkdir(parents=True, exist_ok=True)
    # Atomic replacement keeps two simultaneous research processes from seeing
    # a partial cache file. Only this newly created temporary file is cleaned up.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
        os.replace(temporary, cached)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return raw


def _triangle(source: dict, metadata: ExampleMetadata, backend: str) -> Triangle:
    origins = [
        _month_start(metadata.first_origin, i * metadata.step_months)
        for i in range(metadata.origin_count)
    ]
    labels = [
        str(origin.year)
        if metadata.origin_grain == "Y"
        else f"{origin.year}Q{(origin.month - 1) // 3 + 1}"
        for origin in origins
    ]
    lags = list(range(metadata.step_months, metadata.horizon_months + 1, metadata.step_months))
    if (
        source["originPeriods"] != labels
        or source["developmentMonths"] != lags
        or source["valueType"] != "cumulative"
        or source["basis"] != metadata.loss_field.removesuffix("_loss")
        or len(source["values"]) != len(origins)
        or any(len(row) != len(lags) for row in source["values"])
        or len(source["earnedPremium"]) != len(origins)
    ):
        raise ValueError(f"Unexpected appendix schema or axes for {source['id']}")
    rows = []
    for origin, values, premium in zip(
        origins, source["values"], source["earnedPremium"], strict=True
    ):
        if not math.isfinite(premium) or premium <= 0:
            raise ValueError("Appendix premiums must be finite and positive")
        for lag, value in zip(lags, values, strict=True):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Appendix cumulative claims must be finite and nonnegative")
            rows.append(
                (
                    origin,
                    lag,
                    _month_start(origin, lag) - dt.timedelta(days=1),
                    metadata.loss_field,
                    value,
                )
            )
        rows.append(
            (
                origin,
                lags[0],
                _month_start(origin, lags[0]) - dt.timedelta(days=1),
                PREMIUM_FIELD,
                premium,
            )
        )
    return Triangle.from_long(
        pd.DataFrame(rows, columns=["origin_period", "dev_lag", "eval_date", "field", "value"]),
        measure="cumulative",
        origin_grain=metadata.origin_grain,
        dev_grain=metadata.dev_grain,
        units=UNITS,
        backend=backend,
    )


def load_published_examples(
    backend: str = "duckdb", *, cache_dir: str | Path | None = None
) -> dict[str, Triangle]:
    """Return complete appendix data as ``swiss``, ``liability`` and ``property``.

    The first call downloads and verifies the source; later calls reuse the
    verified cache. ``cache_dir`` permits an explicit offline cache location.
    Constants record source provenance and the required fixed model horizons.
    """
    data = json.loads(_source_bytes(cache_dir))
    sources = data["triangles"]
    if len(sources) != len(EXAMPLE_METADATA) or {s["id"] for s in sources} != set(EXAMPLE_METADATA):
        raise ValueError(
            "Published appendix must contain swiss, liability and property exactly once"
        )
    return {s["id"]: _triangle(s, EXAMPLE_METADATA[s["id"]], backend) for s in sources}
