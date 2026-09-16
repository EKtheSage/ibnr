"""Load the Balona-Richman appendix triangles for conventional-method research.

The paper is Caesar Balona and Ronald Richman, "The Actuary and IBNR Techniques:
A Machine Learning Approach", 14 August 2020, SSRN abstract 3697256
(https://ssrn.com/abstract=3697256). There is no journal volume to cite. The
three appendix triangles and their earned premium tables were transcribed from
the copy the Institute and Faculty of Actuaries distributes and are stored in
this repository at ``DATA_PATH``, so loading them reads a file and never reaches
the network. That file's ``source`` record carries the table and page each
transcription came from, and ``EXAMPLE_METADATA`` declares the same numbers, so
a file describing a different edition of the paper is refused rather than read.

The Swiss triangle is not the authors' own data: the paper takes it from Gisler
(2015) and notes that the figures there have been adjusted for privacy reasons.
The two quarterly triangles were supplied to the authors by an insurer, with the
figures divided by a random number to preserve confidentiality and the earned
premium normalised to a 50% average loss ratio.

Usage: ``load_published_examples()`` returns complete published rectangles, NOT
valuation-ready upper triangles. Always choose an explicit ``as_of`` before
fitting. Development endpoints are 240 months for Swiss liability and 63 months
for the two quarterly examples; they mean final published development, not a
verified fully settled ultimate or an estimated tail beyond the appendix.

The appendix gives origin periods and development months, not observation dates.
This adapter assigns each cell the end of its corresponding calendar period.
Premium is stored once, at the origin period's end, as an explicit retrospective
availability convention. In particular, Swiss premiums were simulated from
ultimate losses: dating them earlier does NOT make them an independently observed
historical input. The printed initial/training/future shading is not used to
decide availability. See ``EXAMPLE_METADATA`` and the transcribed file's own
``source`` notes.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ibnr import Triangle

DATA_PATH = Path(__file__).resolve().parents[1] / "analysis/data/balona_richman_2020_appendix.json"
UNITS = "Source triangle units; currency and magnitude scale unspecified"
PREMIUM_FIELD = "earned_premium"
REQUIRED_KEYS = (
    "id",
    "basis",
    "valueType",
    "originPeriods",
    "developmentMonths",
    "values",
    "earnedPremium",
)


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

    @property
    def declared_tables(self) -> dict[str, int]:
        """Where in the paper this example was transcribed from."""
        return {
            "claimsTable": self.claims_table,
            "claimsPage": self.claims_page,
            "premiumTable": self.premium_table,
            "premiumPage": self.premium_page,
        }


EXAMPLE_METADATA = {
    "swiss": ExampleMetadata(
        "Swiss private liability",
        "paid_loss",
        "Y",
        "Y",
        dt.date(1979, 1, 1),
        19,
        20,
        24,
        49,
        23,
        48,
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
        26,
        51,
        25,
        50,
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
        28,
        53,
        27,
        52,
        "Claims scaled for confidentiality; premiums normalised around a 50% "
        "average loss ratio. Negative increments are retained. Late-cell observation "
        "versus extrapolation is unspecified.",
    ),
}


def _month_start(origin: dt.date, months: int) -> dt.date:
    year, month = divmod(origin.year * 12 + origin.month - 1 + months, 12)
    return dt.date(year, month + 1, 1)


def _appendix() -> dict:
    """Read the transcribed appendix, naming the file when it cannot be used."""
    if not DATA_PATH.exists():
        raise ValueError(
            f"The transcribed Balona-Richman appendix is missing from {DATA_PATH}. "
            "It is committed to this repository; restore it rather than downloading data."
        )
    try:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{DATA_PATH} is not valid JSON ({exc}); the file is damaged") from exc
    if not isinstance(data, dict) or not isinstance(data.get("triangles"), list):
        raise ValueError(f"{DATA_PATH} must hold a source record and a list of triangles")
    return data


def _check_declared_tables(record: object) -> None:
    """Refuse a file transcribed from a different edition of the paper."""
    tables = record.get("tables") if isinstance(record, dict) else None
    if not isinstance(tables, dict):
        raise ValueError(f"{DATA_PATH} must record the table and page of each transcription")
    for name, metadata in EXAMPLE_METADATA.items():
        if tables.get(name) != metadata.declared_tables:
            raise ValueError(
                f"{DATA_PATH} records {tables.get(name)} for {name}, but this module declares "
                f"{metadata.declared_tables}. The two describe different editions of the paper."
            )


def _triangle(source: dict, metadata: ExampleMetadata, backend: str) -> Triangle:
    missing = [key for key in REQUIRED_KEYS if key not in source]
    if missing:
        raise ValueError(f"Transcribed {metadata.title} triangle is missing {', '.join(missing)}")
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


def load_published_examples(backend: str = "duckdb") -> dict[str, Triangle]:
    """Return the appendix data as ``swiss``, ``liability`` and ``property``.

    The transcribed tables are read from ``DATA_PATH`` inside this repository, so
    the call needs no network access. Constants record source provenance and the
    required fixed model horizons.
    """
    data = _appendix()
    sources = data["triangles"]
    if len(sources) != len(EXAMPLE_METADATA) or {s.get("id") for s in sources} != set(
        EXAMPLE_METADATA
    ):
        raise ValueError(
            "The transcribed appendix must contain swiss, liability and property exactly once"
        )
    _check_declared_tables(data.get("source"))
    return {s["id"]: _triangle(s, EXAMPLE_METADATA[s["id"]], backend) for s in sources}
