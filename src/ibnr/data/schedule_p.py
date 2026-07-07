"""Adapter for the CAS Schedule P gold mart (cas-schedule-p-data-model repo).

The data pipeline lives in that repo; this package only consumes its published
gold mart. The mart of record is ``mart_reserving_model_training``: long-format
loss observations keyed by company x line_of_business x accident_year x
development_age x statement_year. Schema (verified 2026-06-12 rebuild):

    company_code, company_name, line_of_business, statement_year, accident_year,
    development_age (years, 1-10), calendar_age, incurred_loss, cum_paid_loss,
    bulk_loss, earned_prem_net, earned_prem_direct, case_reserve, loss_ratio,
    paid_to_incurred_ratio, ata_factor_into_this_age, publish_id

Note: the mart's ``incurred_loss`` is gross of bulk+IBNR and its
``case_reserve`` is incurred - paid (so it also contains bulk). The adapter
derives ``reported_loss`` = incurred_loss - bulk_loss (paid + true case),
which is what Meyers' monograph calls "incurred".

Physical storage is parquet under ``warehouse/`` with the active publish chosen
by ``warehouse/_active_manifest.json``. Point this adapter at the warehouse
directory (or set the IBNR_SCHEDULE_P_WAREHOUSE environment variable).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ibis

from ibnr.triangle.core import Triangle
from ibnr.triangle.io import from_long, resolve_backend

ENV_VAR = "IBNR_SCHEDULE_P_WAREHOUSE"
TRAINING_MART = "mart_reserving_model_training"

#: mart column -> triangle field name. Loss fields are cumulative; premiums,
#: reserves and bulk are eval-date snapshots carried along as fields.
DEFAULT_FIELDS = {
    "cum_paid_loss": "paid_loss",
    "incurred_loss": "incurred_loss",  # gross of bulk+IBNR
    "bulk_loss": "bulk_loss",
    "case_reserve": "case_reserve",
    "earned_prem_net": "earned_premium",
    "earned_prem_direct": "earned_premium_direct",
}

#: derived field: triangle field name -> ibis expression over the mart table
DERIVED_FIELDS = {
    # Meyers' "incurred": net of bulk+IBNR, i.e. paid + case reserves
    "reported_loss": lambda t: t.incurred_loss - t.bulk_loss,
}

SEGMENTS = ["company_code", "company_name", "line_of_business"]


def active_mart_path(warehouse: str | Path | None = None, mart: str = TRAINING_MART) -> Path:
    """Resolve a mart's parquet path from the warehouse's active-publish manifest."""
    warehouse = _resolve_warehouse(warehouse)
    manifest = json.loads((warehouse / "_active_manifest.json").read_text())
    rel = Path(manifest["tables"][mart])
    # manifest paths are relative to the data-model repo root ("warehouse\...")
    return warehouse.parent / rel


def active_publish_id(warehouse: str | Path | None = None) -> str:
    """The active gold publish's version stamp — stamp this into every
    results artifact so figures trace back to an exact data publish."""
    warehouse = _resolve_warehouse(warehouse)
    manifest = json.loads((warehouse / "_active_manifest.json").read_text())
    return str(manifest["publish_id"])


def _resolve_warehouse(warehouse: str | Path | None) -> Path:
    if warehouse is None:
        warehouse = os.environ.get(ENV_VAR)
        if warehouse is None:
            raise ValueError(
                f"no warehouse path given and {ENV_VAR} is not set; "
                "point at the cas-schedule-p-data-model warehouse directory"
            )
    return Path(warehouse)


def load_schedule_p(
    warehouse: str | Path | None = None,
    *,
    lines: list[str] | None = None,
    companies: list[str] | None = None,
    fields: dict[str, str] | None = None,
    derived: dict | None = None,
    backend: str | None = None,
) -> Triangle:
    """Load the reserving training mart as a Triangle.

    Mapping: origin_period = Jan 1 of accident_year; dev_lag = development_age
    in months; eval_date = Dec 31 of statement_year (annual statement date).
    Values are USD thousands. ``lines``/``companies`` filter line_of_business /
    company_code before materializing.
    """
    fields = fields or DEFAULT_FIELDS
    derived = DERIVED_FIELDS if derived is None else derived
    con = resolve_backend(backend)
    t = con.read_parquet(str(active_mart_path(warehouse)))
    if lines:
        t = t.filter(t.line_of_business.isin(lines))
    if companies:
        t = t.filter(t.company_code.isin(companies))
    t = t.select(
        *SEGMENTS,
        origin_period=ibis.date(t.accident_year, 1, 1),
        dev_lag=(t.development_age * 12).cast("int64"),
        eval_date=ibis.date(t.statement_year, 12, 31),
        **{new: t[old] for old, new in fields.items()},
        **{name: expr(t) for name, expr in derived.items()},
    )
    return from_long(
        t,
        fields=[*fields.values(), *derived],
        measure="cumulative",
        origin_grain="Y",
        dev_grain="Y",
        units="USD thousands",
    )
