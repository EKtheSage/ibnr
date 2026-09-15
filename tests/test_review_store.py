"""Real SQLite governance, durability, concurrency and integrity behavior."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from apps.reserving_review.store import (  # noqa: E402
    Actor,
    Conflict,
    Forbidden,
    IntegrityError,
    ReviewError,
    ReviewStore,
)

AUTHOR = Actor("Alex", frozenset({"analyst"}))
OTHER = Actor("Blair", frozenset({"analyst"}))
REVIEWER = Actor("Casey", frozenset({"reviewer"}))
DUAL_AUTHOR = Actor("Alex", frozenset({"analyst", "reviewer"}))
UNAUTHORIZED = Actor("Guest", frozenset())


@pytest.fixture
def snapshot():
    return {
        "title": "Liability at December 2020",
        "as_of": "2020-12-31",
        "history_start": "2011-12-31",
        "metric": "ave",
        "loss_field": "paid_loss",
        "units": "USD",
        "horizon": 96,
        "grain": "Y",
        "source_hash": "fixture provenance",
        "source_csv": "origin_period,dev_lag,value\n2020-01-01,12,100\n",
        "engine": {"name": "conventional", "version": "fixture"},
        "selected": {"name": "cl", "settings": {"method": "cl"}, "mean_rmse": 8.0},
        "ranking": [{"candidate": "cl", "eligible": True, "mean_rmse": 8.0}],
        "origins": [
            {"origin_period": "2019-01-01", "latest": 200.0, "ultimate": 230.0, "reserve": 30.0},
            {"origin_period": "2020-01-01", "latest": 100.0, "ultimate": 150.0, "reserve": 50.0},
        ],
        "factor_summary": [{"factor": 1.5, "from_dev_lag": 12}],
        "factor_selection": [],
        "history_scores": [{"as_of": "2019-12-31", "eval_date": "2020-12-31", "rmse": 8.0}],
        "warnings": ["Reference application"],
    }


@pytest.fixture
def store(tmp_path):
    return ReviewStore(tmp_path / "review.sqlite3")


def submitted(store, snapshot):
    run = store.create_run(AUTHOR, snapshot)
    return store.submit(AUTHOR, run["id"], "Ready for independent review", run["revision"])


def decided(store, snapshot, decision="approve"):
    run = submitted(store, snapshot)
    return store.decide(REVIEWER, run["id"], decision, "Evidence assessed", run["revision"])


def verify_export_hashes(run):
    def canonical(value):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )

    previous = hashlib.sha256(canonical(run["snapshot"]).encode()).hexdigest()
    assert previous == run["snapshot_hash"]
    for revision, event in enumerate(run["events"], start=1):
        assert event["revision"] == revision
        assert event["previous_hash"] == previous
        encoded = canonical({key: value for key, value in event.items() if key != "hash"})
        previous = hashlib.sha256(encoded.encode()).hexdigest()
        assert event["hash"] == previous


def test_snapshot_is_detached_canonical_durable_and_has_complete_evidence(store, snapshot):
    original = copy.deepcopy(snapshot)
    run = store.create_run(AUTHOR, snapshot)
    assert run["status"] == "DRAFT"
    assert run["revision"] == 1
    assert run["model_reserve"] == run["booked_reserve"] == 80
    snapshot["origins"][0]["reserve"] = 999
    run["snapshot"]["warnings"].append("changed only in caller memory")
    reopened = ReviewStore(store.path).get_run(AUTHOR, run["id"])
    assert reopened["snapshot"] == original
    verify_export_hashes(reopened)
    event = reopened["events"][0]
    assert event["at"].endswith("+00:00")
    assert event["actor"] == AUTHOR.name
    assert event["reason"]


def test_visibility_and_creation_require_roles_and_ownership(store, snapshot):
    for actor in (REVIEWER, UNAUTHORIZED):
        with pytest.raises(Forbidden):
            store.create_run(actor, snapshot)
    run = store.create_run(AUTHOR, snapshot)
    assert [row["id"] for row in store.list_runs(AUTHOR)] == [run["id"]]
    assert store.list_runs(OTHER) == store.list_runs(REVIEWER) == []
    for actor in (OTHER, REVIEWER, UNAUTHORIZED):
        with pytest.raises(Forbidden):
            store.get_run(actor, run["id"])
    with pytest.raises(Forbidden):
        store.list_runs(UNAUTHORIZED)
    run = store.submit(AUTHOR, run["id"], "Ready", run["revision"])
    assert store.get_run(REVIEWER, run["id"])["status"] == "SUBMITTED"
    assert store.list_runs(REVIEWER)[0]["title"] == snapshot["title"]
    assert "snapshot" not in store.list_runs(REVIEWER)[0]
    with pytest.raises(Forbidden):
        store.get_run(OTHER, run["id"])


def test_override_zero_negative_and_repeated_changes_preserve_model_and_event_hashes(
    store, snapshot
):
    run = store.create_run(AUTHOR, snapshot)
    snapshot_hash = run["snapshot_hash"]
    for amount, total in ((0, 30), (-10.5, 19.5), (12.5, 42.5)):
        run = store.set_override(
            AUTHOR, run["id"], "2020-01-01", amount, "Case reserve review", run["revision"]
        )
        assert run["booked_reserve"] == total
        assert run["model_reserve"] == 80
        assert run["snapshot_hash"] == snapshot_hash
        assert run["overrides"]["2020-01-01"] == {
            "reserve": amount,
            "reason": "Case reserve review",
            "actor": AUTHOR.name,
        }
        assert run["events"][0]["payload"]["overrides"] == {}
        verify_export_hashes(run)
    assert [e["payload"]["reserve"] for e in run["events"][1:]] == [0, -10.5, 12.5]


@pytest.mark.parametrize("amount", [float("inf"), float("-inf"), float("nan"), True, "100", None])
def test_invalid_override_values_leave_no_event(store, snapshot, amount):
    run = store.create_run(AUTHOR, snapshot)
    with pytest.raises(ReviewError, match="finite numbers"):
        store.set_override(AUTHOR, run["id"], "2020-01-01", amount, "Reason", 1)
    assert store.get_run(AUTHOR, run["id"])["revision"] == 1


def test_overflowing_booked_total_is_rejected_before_writing_an_event(store, snapshot):
    run = store.create_run(AUTHOR, snapshot)
    run = store.set_override(AUTHOR, run["id"], "2019-01-01", 1e308, "Large exposure", 1)
    with pytest.raises(ReviewError, match="Total reserves") as caught:
        store.set_override(AUTHOR, run["id"], "2020-01-01", 1e308, "Large exposure", 2)
    assert caught.value.status_code == 400
    assert store.get_run(AUTHOR, run["id"]) == run


def test_origin_and_reason_validation_and_only_owner_analyst_mutates(store, snapshot):
    run = store.create_run(AUTHOR, snapshot)
    for actor in (OTHER, REVIEWER, Actor(AUTHOR.name, frozenset({"reviewer"}))):
        with pytest.raises(Forbidden):
            store.set_override(actor, run["id"], "2020-01-01", 0, "Reason", 1)
        with pytest.raises(Forbidden):
            store.submit(actor, run["id"], "Reason", 1)
    with pytest.raises(ReviewError, match="known origin"):
        store.set_override(AUTHOR, run["id"], "2030-01-01", 0, "Reason", 1)
    for reason in ("", "  ", None):
        with pytest.raises(ReviewError, match="reason"):
            store.set_override(AUTHOR, run["id"], "2020-01-01", 0, reason, 1)
        with pytest.raises(ReviewError, match="reason"):
            store.submit(AUTHOR, run["id"], reason, 1)
    assert store.get_run(AUTHOR, run["id"])["revision"] == 1


def test_self_approval_is_forbidden_even_with_both_roles(store, snapshot):
    run = submitted(store, snapshot)
    for actor in (AUTHOR, OTHER, DUAL_AUTHOR):
        with pytest.raises(Forbidden):
            store.decide(actor, run["id"], "approve", "I checked", run["revision"])
    with pytest.raises(ReviewError, match="reason"):
        store.decide(REVIEWER, run["id"], "approve", " ", run["revision"])
    with pytest.raises(ReviewError, match="decision"):
        store.decide(REVIEWER, run["id"], "maybe", "Reason", run["revision"])
    assert store.get_run(REVIEWER, run["id"])["status"] == "SUBMITTED"


@pytest.mark.parametrize("decision,status", [("approve", "APPROVED"), ("reject", "REJECTED")])
def test_decided_run_is_immutable_and_revise_creates_linked_owned_draft(
    store, snapshot, decision, status
):
    run = store.create_run(AUTHOR, snapshot)
    run = store.set_override(AUTHOR, run["id"], "2020-01-01", 0, "No remaining payments", 1)
    run = store.submit(AUTHOR, run["id"], "Ready", run["revision"])
    run = store.decide(REVIEWER, run["id"], decision, "Assessed", run["revision"])
    assert run["status"] == status
    with pytest.raises(Conflict):
        store.set_override(AUTHOR, run["id"], "2020-01-01", 100, "Changed mind", run["revision"])
    with pytest.raises(Conflict):
        store.submit(AUTHOR, run["id"], "Again", run["revision"])
    with pytest.raises(Conflict):
        store.decide(REVIEWER, run["id"], "reject", "Again", run["revision"])
    for actor in (OTHER, REVIEWER):
        with pytest.raises(Forbidden):
            store.revise(actor, run["id"], run["revision"])
    child = store.revise(AUTHOR, run["id"], run["revision"])
    assert child["id"] != run["id"]
    assert child["parent_id"] == run["id"]
    assert child["status"] == "DRAFT"
    assert child["revision"] == 1
    assert child["created_by"] == AUTHOR.name
    assert child["snapshot"] == run["snapshot"]
    assert child["snapshot_hash"] == run["snapshot_hash"]
    assert child["overrides"] == run["overrides"]
    assert child["booked_reserve"] == 30
    assert child["events"][0]["payload"]["parent_event_hash"] == run["events"][-1]["hash"]
    assert child["events"][0]["payload"]["parent_revision"] == run["revision"]
    child = store.set_override(AUTHOR, child["id"], "2020-01-01", 40, "New assessment", 1)
    verify_export_hashes(child)
    assert store.get_run(AUTHOR, run["id"]) == run
    with pytest.raises(Forbidden):
        store.get_run(REVIEWER, child["id"])


def test_state_transitions_and_export_require_correct_state(store, snapshot):
    run = store.create_run(AUTHOR, snapshot)
    with pytest.raises(Conflict):
        store.decide(REVIEWER, run["id"], "approve", "Too early", 1)
    with pytest.raises(Conflict):
        store.revise(AUTHOR, run["id"], 1)
    with pytest.raises(Conflict):
        store.export_approved(AUTHOR, run["id"])
    run = store.submit(AUTHOR, run["id"], "Ready", 1)
    with pytest.raises(Conflict):
        store.set_override(AUTHOR, run["id"], "2020-01-01", 40, "Too late", run["revision"])
    with pytest.raises(Conflict):
        store.export_approved(REVIEWER, run["id"])
    run = store.decide(REVIEWER, run["id"], "approve", "Approved evidence", run["revision"])
    exported = ReviewStore(store.path).export_approved(REVIEWER, run["id"])
    assert exported["snapshot"] == snapshot
    assert exported["exported_by"] == REVIEWER.name
    assert exported["exported_at"].endswith("+00:00")
    verify_export_hashes(exported)
    assert store.get_run(AUTHOR, run["id"]) == run
    with pytest.raises(Forbidden):
        store.export_approved(OTHER, run["id"])
    rejected = decided(store, snapshot, "reject")
    with pytest.raises(Conflict):
        store.export_approved(REVIEWER, rejected["id"])


def test_stale_versions_on_edits_decisions_and_revision_creation_are_conflicts(store, snapshot):
    run = submitted(store, snapshot)
    for operation in (
        lambda: store.set_override(AUTHOR, run["id"], "2020-01-01", 0, "Reason", 1),
        lambda: store.submit(AUTHOR, run["id"], "Reason", 1),
        lambda: store.decide(REVIEWER, run["id"], "approve", "Reason", 1),
    ):
        with pytest.raises(Conflict, match="reload"):
            operation()
    run = store.decide(REVIEWER, run["id"], "approve", "Reason", run["revision"])
    with pytest.raises(Conflict, match="reload"):
        store.revise(AUTHOR, run["id"], run["revision"] - 1)
    with pytest.raises(ReviewError, match="positive integer"):
        store.revise(AUTHOR, run["id"], True)


def test_two_connections_racing_the_same_revision_commit_only_one_override(store, snapshot):
    run = store.create_run(AUTHOR, snapshot)
    barrier = Barrier(2)

    def write(amount):
        connection_store = ReviewStore(store.path)
        barrier.wait(timeout=10)
        try:
            return connection_store.set_override(
                AUTHOR, run["id"], "2020-01-01", amount, "Concurrent edit", 1
            )
        except Conflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [0.0, 15.0]))
    assert sum(isinstance(result, Conflict) for result in results) == 1
    successful = next(result for result in results if isinstance(result, dict))
    reopened = ReviewStore(store.path).get_run(AUTHOR, run["id"])
    assert reopened == successful
    assert reopened["revision"] == 2
    verify_export_hashes(reopened)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE review_runs SET snapshot_json = '{}'",
        "UPDATE review_runs SET snapshot_hash = 'changed'",
        "UPDATE review_runs SET created_by = 'Someone else'",
        "DELETE FROM review_runs",
        "INSERT OR REPLACE INTO review_runs SELECT * FROM review_runs",
        "UPDATE review_events SET event_json = '{}'",
        "DELETE FROM review_events",
    ],
)
def test_sql_triggers_refuse_mutating_or_deleting_snapshots_and_events(store, snapshot, sql):
    run = decided(store, snapshot)
    with (
        sqlite3.connect(store.path) as connection,
        pytest.raises(sqlite3.IntegrityError, match="immutable|append-only"),
    ):
        connection.execute(sql)
    assert store.get_run(AUTHOR, run["id"]) == run


@pytest.mark.parametrize("target", ["snapshot", "event", "metadata", "chain", "truncation"])
def test_tampering_after_bypassing_triggers_is_detected_on_read_and_export(store, snapshot, target):
    run = decided(store, snapshot)
    with sqlite3.connect(store.path) as connection:
        # An administrator can bypass the ordinary SQL protection. These edits
        # retain the previous hashes so that read-time corruption checks apply.
        connection.execute("DROP TRIGGER review_runs_no_update")
        connection.execute("DROP TRIGGER review_events_no_update")
        connection.execute("DROP TRIGGER review_events_no_delete")
        if target == "snapshot":
            altered = copy.deepcopy(snapshot)
            altered["origins"][0]["reserve"] = 999
            connection.execute(
                "UPDATE review_runs SET snapshot_json = ?",
                (json.dumps(altered, sort_keys=True, separators=(",", ":")),),
            )
        elif target == "metadata":
            connection.execute("UPDATE review_runs SET created_by = 'Intruder'")
        elif target == "event":
            event = dict(run["events"][-1])
            event.pop("hash")
            event["reason"] = "Rewritten decision"
            connection.execute(
                "UPDATE review_events SET event_json = ? WHERE revision = 3",
                (json.dumps(event, sort_keys=True, separators=(",", ":")),),
            )
        elif target == "chain":
            connection.execute("UPDATE review_events SET event_hash = 'wrong' WHERE revision = 1")
        else:
            connection.execute("DELETE FROM review_events WHERE revision = 2")
    for operation in (
        lambda: store.get_run(AUTHOR, run["id"]),
        lambda: store.list_runs(REVIEWER),
        lambda: store.export_approved(REVIEWER, run["id"]),
    ):
        with pytest.raises(IntegrityError, match="integrity verification"):
            operation()


@pytest.mark.parametrize(
    "mutation", ["nonfinite", "duplicate", "no_origins", "bad_origin", "overflow"]
)
def test_invalid_evidence_never_creates_a_run(store, snapshot, mutation):
    if mutation == "nonfinite":
        snapshot["selected"]["mean_rmse"] = float("nan")
    elif mutation == "duplicate":
        snapshot["origins"].append(snapshot["origins"][0].copy())
    elif mutation == "no_origins":
        snapshot["origins"] = []
    elif mutation == "bad_origin":
        snapshot["origins"][0]["origin_period"] = "yesterday"
    else:
        for row in snapshot["origins"]:
            row["reserve"] = 1e308
    with pytest.raises(ReviewError):
        store.create_run(AUTHOR, snapshot)
    assert store.list_runs(AUTHOR) == []
