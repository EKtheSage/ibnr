"""Durable review evidence and separation of duties, using only SQLite/stdlib.

Runs hold immutable canonical JSON snapshots. Workflow state and booked reserves
are reconstructed from append-only, hash-linked events on every read. Triggers
refuse ordinary SQL updates/deletes of either table. These checks detect accidental
or unauthorized changes; they are not protection against a database administrator
who can drop triggers and rewrite the entire database and its hashes.

Actor identities and roles must come from the authenticated server configuration,
never request JSON. This module independently enforces ownership and role checks.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import math
import sqlite3
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

UNVERIFIABLE = "UNVERIFIABLE"
"""Listing status of a run whose hash chain no longer verifies."""


class ReviewError(Exception):
    """A review request that the HTTP layer can report without a traceback."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class Conflict(ReviewError):
    def __init__(self, message: str):
        super().__init__(message, 409)


class Forbidden(ReviewError):
    def __init__(self, message: str):
        super().__init__(message, 403)


class IntegrityError(ReviewError):
    def __init__(self, message: str):
        super().__init__(message, 409)


@dataclass(frozen=True)
class Actor:
    name: str
    roles: frozenset[str]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ReviewError("An authenticated actor name is required")
        if isinstance(self.roles, str) or any(
            not isinstance(role, str) or not role for role in self.roles
        ):
            raise ReviewError("Actor roles must be a collection of role names")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "roles", frozenset(self.roles))


def _canonical(value) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ReviewError("Review evidence must be finite JSON data") from exc


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _amount(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewError("Reserve amounts must be finite numbers")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ReviewError("Reserve amounts must be finite numbers") from exc
    if not math.isfinite(result):
        raise ReviewError("Reserve amounts must be finite numbers")
    return result


def _reason(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise ReviewError("A nonempty reason is required")
    return reason.strip()


def _totals(snapshot: dict, overrides: dict) -> tuple[float, float]:
    try:
        model = math.fsum(row["reserve"] for row in snapshot["origins"])
        booked = math.fsum(
            overrides.get(row["origin_period"], {}).get("reserve", row["reserve"])
            for row in snapshot["origins"]
        )
    except OverflowError as exc:
        raise ReviewError("Total reserves must be finite") from exc
    if not math.isfinite(model) or not math.isfinite(booked):
        raise ReviewError("Total reserves must be finite")
    return model, booked


def _snapshot_json(snapshot: dict) -> str:
    if not isinstance(snapshot, dict):
        raise ReviewError("A review snapshot must be an object")
    # Serialize and parse to detach caller-owned mutable containers, rejecting
    # NaN and non-JSON objects before anything can reach the ledger.
    encoded = _canonical(snapshot)
    value = json.loads(encoded)
    if not isinstance(value.get("origins"), list) or not value["origins"]:
        raise ReviewError("A review snapshot needs origin-level reserve evidence")
    seen = set()
    for row in value["origins"]:
        if not isinstance(row, dict):
            raise ReviewError("Origin evidence must contain objects")
        origin = row.get("origin_period")
        try:
            if not isinstance(origin, str) or dt.date.fromisoformat(origin).isoformat() != origin:
                raise ValueError
        except ValueError as exc:
            raise ReviewError("Origin periods must be ISO dates") from exc
        if origin in seen:
            raise ReviewError("Origin periods must be unique")
        seen.add(origin)
        for field in ("latest", "ultimate", "reserve"):
            _amount(row.get(field))
    _totals(value, {})
    return encoded


_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_runs (
    id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES review_runs(id),
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_events (
    run_id TEXT NOT NULL REFERENCES review_runs(id),
    revision INTEGER NOT NULL CHECK(revision > 0),
    event_json TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    PRIMARY KEY(run_id, revision)
);
CREATE TRIGGER IF NOT EXISTS review_runs_no_update
BEFORE UPDATE ON review_runs BEGIN
    SELECT RAISE(ABORT, 'Review snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS review_runs_no_replace
BEFORE INSERT ON review_runs
WHEN EXISTS (SELECT 1 FROM review_runs WHERE id = NEW.id)
BEGIN
    SELECT RAISE(ABORT, 'Review snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS review_runs_no_delete
BEFORE DELETE ON review_runs BEGIN
    SELECT RAISE(ABORT, 'Review snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS review_events_no_update
BEFORE UPDATE ON review_events BEGIN
    SELECT RAISE(ABORT, 'Review events are append-only');
END;
CREATE TRIGGER IF NOT EXISTS review_events_no_delete
BEFORE DELETE ON review_events BEGIN
    SELECT RAISE(ABORT, 'Review events are append-only');
END;
CREATE TRIGGER IF NOT EXISTS review_events_sequential
BEFORE INSERT ON review_events
WHEN NEW.revision != COALESCE(
    (SELECT MAX(revision) + 1 FROM review_events WHERE run_id = NEW.run_id), 1
)
BEGIN
    SELECT RAISE(ABORT, 'Review event revisions must be consecutive');
END;
"""


class ReviewStore:
    """A file-backed ledger; each request uses its own SQLite transaction.

    Revision 1 creates a DRAFT. Editing/submitting belongs to its author with
    the analyst role. An independent reviewer can approve/reject a SUBMITTED
    run. Decided runs stay immutable; their analyst may create at most one
    linked DRAFT, and every read reports that child as ``superseded_by``.
    """

    def __init__(self, path):
        if str(path) == ":memory:":
            raise ReviewError("Use a file path for a durable review store")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _transaction(self, *, write=False):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise IntegrityError(str(exc)) from exc
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _reader(actor):
        if not isinstance(actor, Actor) or not actor.roles.intersection({"analyst", "reviewer"}):
            raise Forbidden("An analyst or reviewer role is required")

    @staticmethod
    def _visible(actor, run):
        return ("analyst" in actor.roles and actor.name == run["created_by"]) or (
            "reviewer" in actor.roles and run["status"] != "DRAFT"
        )

    @staticmethod
    def _owner(actor, run):
        if "analyst" not in actor.roles or actor.name != run["created_by"]:
            raise Forbidden("Only the author with the analyst role can change this review")

    @staticmethod
    def _revision(run, expected_revision):
        if type(expected_revision) is not int or expected_revision < 1:
            raise ReviewError("expected_revision must be a positive integer")
        if run["revision"] != expected_revision:
            raise Conflict("The review changed; reload it before applying this action")

    @staticmethod
    def _append(connection, run, actor, action, reason, payload):
        event = {
            "run_id": run["id"],
            "revision": run["revision"] + 1,
            "at": _now(),
            "actor": actor.name,
            "roles": sorted(actor.roles),
            "action": action,
            "reason": reason,
            "payload": payload,
            "previous_hash": run["events"][-1]["hash"] if run["events"] else run["snapshot_hash"],
        }
        if not run["events"]:
            event["at"] = run["created_at"]
        encoded = _canonical(event)
        connection.execute(
            "INSERT INTO review_events VALUES (?, ?, ?, ?)",
            (run["id"], event["revision"], encoded, _hash(encoded)),
        )

    def _new(self, connection, actor, encoded, *, parent=None):
        run = {
            "id": uuid.uuid4().hex,
            "parent_id": parent["id"] if parent else None,
            "created_by": actor.name,
            "created_at": _now(),
            "snapshot_hash": _hash(encoded),
            "revision": 0,
            "events": [],
        }
        connection.execute(
            "INSERT INTO review_runs VALUES (?, ?, ?, ?, ?, ?)",
            (
                run["id"],
                run["parent_id"],
                run["created_by"],
                run["created_at"],
                encoded,
                run["snapshot_hash"],
            ),
        )
        self._append(
            connection,
            run,
            actor,
            "revise" if parent else "create",
            f"Revision of {parent['id']} at revision {parent['revision']}"
            if parent
            else "Created reserving review",
            {
                "snapshot_hash": run["snapshot_hash"],
                "parent_id": run["parent_id"],
                "parent_revision": parent["revision"] if parent else None,
                "parent_event_hash": parent["events"][-1]["hash"] if parent else None,
                "overrides": parent["overrides"] if parent else {},
            },
        )
        return self._read(connection, run["id"])

    def _read(self, connection, run_id):
        record = connection.execute("SELECT * FROM review_runs WHERE id = ?", (run_id,)).fetchone()
        if record is None:
            raise ReviewError("Review not found", 404)
        try:
            return self._verified(connection, dict(record))
        except (ValueError, TypeError, KeyError, OverflowError, ReviewError) as exc:
            raise IntegrityError(f"Review evidence failed integrity verification: {exc}") from exc

    def _verified(self, connection, record):
        encoded = record.pop("snapshot_json")
        snapshot = json.loads(encoded)
        if _snapshot_json(snapshot) != encoded or not hmac.compare_digest(
            _hash(encoded), record["snapshot_hash"]
        ):
            raise ValueError("snapshot hash or canonical encoding differs")
        rows = connection.execute(
            "SELECT * FROM review_events WHERE run_id = ? ORDER BY revision", (record["id"],)
        ).fetchall()
        if not rows:
            raise ValueError("event ledger is empty")
        status, overrides, events = "DRAFT", {}, []
        previous_hash = record["snapshot_hash"]
        known_origins = {row["origin_period"] for row in snapshot["origins"]}
        for revision, row in enumerate(rows, start=1):
            event = json.loads(row["event_json"])
            if (
                row["revision"] != revision
                or event["revision"] != revision
                or event["run_id"] != record["id"]
                or event["previous_hash"] != previous_hash
                or _canonical(event) != row["event_json"]
                or not hmac.compare_digest(_hash(row["event_json"]), row["event_hash"])
            ):
                raise ValueError("event hash chain or sequence differs")
            event_actor = Actor(event["actor"], frozenset(event["roles"]))
            _reason(event["reason"])
            stamp = dt.datetime.fromisoformat(event["at"])
            if stamp.utcoffset() != dt.timedelta(0):
                raise ValueError("event timestamps must use UTC")
            action, payload = event["action"], event["payload"]
            if revision == 1:
                if (
                    action != ("revise" if record["parent_id"] else "create")
                    or event_actor.name != record["created_by"]
                    or "analyst" not in event_actor.roles
                    or event["at"] != record["created_at"]
                    or payload["snapshot_hash"] != record["snapshot_hash"]
                    or payload["parent_id"] != record["parent_id"]
                ):
                    raise ValueError("creation event differs from run metadata")
                initial_overrides = payload["overrides"]
                if not isinstance(initial_overrides, dict) or not set(initial_overrides).issubset(
                    known_origins
                ):
                    raise ValueError("initial overrides refer to unknown origins")
                overrides = {origin: dict(value) for origin, value in initial_overrides.items()}
                if not record["parent_id"] and overrides:
                    raise ValueError("a new review cannot have unrecorded overrides")
                for override in overrides.values():
                    _amount(override["reserve"])
                    _reason(override["reason"])
                    if override["actor"] != record["created_by"]:
                        raise ValueError("override author differs from run author")
            elif action in {"override", "submit"}:
                if status != "DRAFT":
                    raise ValueError("an edit or submission changed a closed draft")
                self._owner(event_actor, record)
                if action == "override":
                    if payload["origin_period"] not in known_origins:
                        raise ValueError("override refers to an unknown origin")
                    overrides[payload["origin_period"]] = {
                        "reserve": _amount(payload["reserve"]),
                        "reason": event["reason"],
                        "actor": event_actor.name,
                    }
                else:
                    status = "SUBMITTED"
            elif action in {"approve", "reject"}:
                if (
                    status != "SUBMITTED"
                    or "reviewer" not in event_actor.roles
                    or event_actor.name == record["created_by"]
                ):
                    raise ValueError("decision violates independent review or workflow order")
                status = "APPROVED" if action == "approve" else "REJECTED"
            else:
                raise ValueError("unknown or misplaced event action")
            previous_hash = row["event_hash"]
            events.append({**event, "hash": previous_hash})
        model, booked = _totals(snapshot, overrides)
        # A revision is a separate run whose ledger cannot append to this one,
        # so the link is read from the child's parent_id. ``revise`` refuses a
        # second child, and the ordering keeps older databases deterministic.
        child = connection.execute(
            "SELECT id FROM review_runs WHERE parent_id = ? ORDER BY created_at, id",
            (record["id"],),
        ).fetchone()
        return {
            **record,
            "status": status,
            "superseded_by": child["id"] if child else None,
            "revision": len(events),
            "snapshot": snapshot,
            "overrides": overrides,
            "model_reserve": model,
            "booked_reserve": booked,
            "events": events,
        }

    def create_run(self, actor: Actor, snapshot: dict) -> dict:
        self._reader(actor)
        if "analyst" not in actor.roles:
            raise Forbidden("The analyst role is required to create a review")
        encoded = _snapshot_json(snapshot)
        with self._transaction(write=True) as connection:
            return self._new(connection, actor, encoded)

    def list_runs(self, actor: Actor) -> list[dict]:
        self._reader(actor)
        summaries = []
        with self._transaction() as connection:
            ids = connection.execute(
                "SELECT id FROM review_runs ORDER BY created_at DESC, id"
            ).fetchall()
            for row in ids:
                try:
                    run = self._read(connection, row["id"])
                except IntegrityError as exc:
                    # One damaged record must not hide the whole workspace.
                    # Nothing it records can be trusted, its author included,
                    # so it is listed to every reader with only its own id and
                    # the failure. Opening it still raises that failure.
                    summaries.append({"id": row["id"], "status": UNVERIFIABLE, "error": str(exc)})
                    continue
                if self._visible(actor, run):
                    summaries.append(
                        {
                            **{
                                key: run[key]
                                for key in (
                                    "id",
                                    "parent_id",
                                    "superseded_by",
                                    "created_by",
                                    "created_at",
                                    "status",
                                    "revision",
                                    "model_reserve",
                                    "booked_reserve",
                                    "snapshot_hash",
                                )
                            },
                            **{
                                key: run["snapshot"].get(key) for key in ("title", "as_of", "units")
                            },
                        }
                    )
        return summaries

    def get_run(self, actor: Actor, run_id: str) -> dict:
        self._reader(actor)
        with self._transaction() as connection:
            run = self._read(connection, run_id)
            if not self._visible(actor, run):
                raise Forbidden("This review is not visible to this actor")
            return run

    def set_override(self, actor, run_id, origin_period, reserve, reason, expected_revision):
        self._reader(actor)
        amount, reason = _amount(reserve), _reason(reason)
        with self._transaction(write=True) as connection:
            run = self._read(connection, run_id)
            self._owner(actor, run)
            self._revision(run, expected_revision)
            if run["status"] != "DRAFT":
                raise Conflict("Only a draft review can be edited")
            if origin_period not in {row["origin_period"] for row in run["snapshot"]["origins"]}:
                raise ReviewError("An override must refer to a known origin period")
            _totals(run["snapshot"], {**run["overrides"], origin_period: {"reserve": amount}})
            self._append(
                connection,
                run,
                actor,
                "override",
                reason,
                {
                    "origin_period": origin_period,
                    "reserve": amount,
                },
            )
            return self._read(connection, run_id)

    def submit(self, actor, run_id, reason, expected_revision):
        self._reader(actor)
        reason = _reason(reason)
        with self._transaction(write=True) as connection:
            run = self._read(connection, run_id)
            self._owner(actor, run)
            self._revision(run, expected_revision)
            if run["status"] != "DRAFT":
                raise Conflict("Only a draft review can be submitted")
            self._append(connection, run, actor, "submit", reason, {})
            return self._read(connection, run_id)

    def decide(self, actor, run_id, decision, reason, expected_revision):
        self._reader(actor)
        if "reviewer" not in actor.roles:
            raise Forbidden("The reviewer role is required to decide a review")
        if decision not in {"approve", "reject"}:
            raise ReviewError("A decision must be 'approve' or 'reject'")
        reason = _reason(reason)
        with self._transaction(write=True) as connection:
            run = self._read(connection, run_id)
            if actor.name == run["created_by"]:
                raise Forbidden("The reviewer must be a different person from the author")
            self._revision(run, expected_revision)
            if run["status"] != "SUBMITTED":
                raise Conflict("Only a submitted review can be decided")
            self._append(connection, run, actor, decision, reason, {})
            return self._read(connection, run_id)

    def revise(self, actor, run_id, expected_revision):
        """Create the one linked draft a decided run is allowed.

        A decided run appends no further events, so its revision never moves
        and a repeated request would otherwise pass the revision check again
        and again, leaving several independently approvable positions for one
        cutoff. The existing child is named instead.
        """
        self._reader(actor)
        with self._transaction(write=True) as connection:
            run = self._read(connection, run_id)
            self._owner(actor, run)
            self._revision(run, expected_revision)
            if run["status"] not in {"APPROVED", "REJECTED"}:
                raise Conflict("Only an approved or rejected review can be revised")
            if run["superseded_by"]:
                raise Conflict(
                    f"This review was already revised as {run['superseded_by']}; "
                    "open that revision instead of creating a second one"
                )
            return self._new(connection, actor, _canonical(run["snapshot"]), parent=run)

    def export_approved(self, actor, run_id):
        run = self.get_run(actor, run_id)
        if run["status"] != "APPROVED":
            raise Conflict("Only an approved review can be exported")
        return {**run, "exported_at": _now(), "exported_by": actor.name}
