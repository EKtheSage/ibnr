"""Real HTTP requests exercise authentication and the persistent review ledger."""

from __future__ import annotations

import base64
import copy
import json
import threading
from http.client import HTTPConnection

import pytest
from apps.reserving_review import server as review_server
from apps.reserving_review.store import ReviewStore

USERS = {
    "analyst": {"password": "analysis-secret", "roles": ["analyst"]},
    "reviewer": {"password": "review-secret", "roles": ["reviewer"]},
    "other": {"password": "other-secret", "roles": ["analyst"]},
    "dual": {"password": "dual-secret", "roles": ["analyst", "reviewer"]},
}
SNAPSHOT = {
    "title": "HTTP example",
    "as_of": "2023-12-31",
    "units": "USD",
    "origins": [
        {"origin_period": "2020-01-01", "latest": 100.0, "ultimate": 150.0, "reserve": 50.0},
        {"origin_period": "2021-01-01", "latest": 50.0, "ultimate": 75.0, "reserve": 25.0},
    ],
    "ranking": [{"candidate": "basic_cl", "mean_rmse": 9.0}],
}
_UNSET = object()


def _auth(name, password):
    token = base64.b64encode(f"{name}:{password}".encode()).decode()
    return f"Basic {token}"


@pytest.fixture
def http_app(tmp_path, monkeypatch):
    store = ReviewStore(tmp_path / "review.sqlite3")
    monkeypatch.setattr(
        review_server.analysis, "analyze_request", lambda _: copy.deepcopy(SNAPSHOT)
    )
    monkeypatch.setattr(
        review_server.analysis, "demo_request", lambda: {"title": "Synthetic demo", "csv": "header"}
    )
    assets = tmp_path / "static"
    assets.mkdir()
    for name, body in (
        ("index.html", "<!doctype html><title>Review</title>"),
        ("app.js", "document.title = 'Review';"),
        ("styles.css", "body { color: black; }"),
    ):
        (assets / name).write_text(body, encoding="utf-8")
    monkeypatch.setattr(review_server, "STATIC_DIR", assets)
    server = review_server.create_server(store=store, users=USERS)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()

    def request(method, path, *, user="analyst", payload=_UNSET, body=None, headers=None):
        request_headers = dict(headers or {})
        if user is not None:
            name, password = (user, USERS[user]["password"]) if isinstance(user, str) else user
            request_headers.setdefault("Authorization", _auth(name, password))
        if payload is not _UNSET:
            body = json.dumps(payload).encode()
            request_headers.setdefault("Content-Type", "application/json")
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            response_body = response.read()
            response_headers = dict(response.getheaders())
            if response_headers.get("Content-Type", "").startswith("application/json"):
                response_body = json.loads(response_body)
            return response.status, response_headers, response_body
        finally:
            connection.close()

    yield server, request, store
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _create(request, user="analyst"):
    status, _, run = request(
        "POST",
        "/api/analyze",
        user=user,
        payload={"title": "HTTP example", "csv": "example", "actor": "reviewer"},
    )
    assert status == 201, run
    return run


def test_config_and_static_assets_are_public_and_strictly_allowlisted(http_app):
    _, request, _ = http_app
    status, headers, config = request("GET", "/api/config", user=None)
    assert status == 200
    assert config == {"demo": False}
    assert headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in headers
    for route, content_type in (
        ("/", "text/html"),
        ("/app.js", "text/javascript"),
        ("/styles.css", "text/css"),
    ):
        status, headers, content = request("GET", route, user=None)
        assert status == 200
        assert headers["Content-Type"].startswith(content_type)
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert "'unsafe-inline'" not in headers["Content-Security-Policy"]
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert content
    for route in ("/server.py", "/../store.py", "/%2e%2e/store.py", "/static/app.js"):
        status, _, error = request("GET", route)
        assert status == 404
        assert set(error) == {"error"}


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "Bearer secret",
        "Basic !!!",
        "Basic /w==",
        _auth("analyst", "wrong"),
        _auth("none", "x"),
    ],
)
def test_api_authentication_rejects_invalid_credentials(http_app, authorization, capsys):
    _, request, _ = http_app
    headers = {"Authorization": authorization} if authorization is not None else {}
    status, response_headers, error = request("GET", "/api/me", user=None, headers=headers)
    assert status == 401
    assert "WWW-Authenticate" not in response_headers
    assert error == {"error": "Valid credentials are required."}
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


def test_authentication_is_checked_on_every_request(http_app):
    _, request, _ = http_app
    assert request("GET", "/api/me")[2] == {"name": "analyst", "roles": ["analyst"]}
    assert request("GET", "/api/me", user=None)[0] == 401
    assert request("GET", "/api/me", user="reviewer")[2] == {
        "name": "reviewer",
        "roles": ["reviewer"],
    }


def test_role_checks_happen_before_analysis_or_demo_generation(http_app, monkeypatch):
    _, request, _ = http_app

    def forbidden_call(*args):
        pytest.fail("An unauthorized request reached expensive analysis.")

    monkeypatch.setattr(review_server.analysis, "analyze_request", forbidden_call)
    monkeypatch.setattr(review_server.analysis, "demo_request", forbidden_call)
    assert request("POST", "/api/analyze", user="reviewer", payload={})[0] == 403
    assert request("GET", "/api/demo", user="reviewer")[0] == 403


def test_demo_payload_uses_the_analysis_contract(http_app):
    _, request, _ = http_app
    assert request("GET", "/api/demo")[2] == {"title": "Synthetic demo", "csv": "header"}


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://foreign.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:1"},
        {"Host": "foreign.example"},
        {"Host": "127.0.0.1:1"},
    ],
)
def test_foreign_host_and_origin_cannot_mutate(http_app, headers):
    _, request, _ = http_app
    status, response_headers, _ = request("POST", "/api/analyze", payload={}, headers=headers)
    assert status == 403
    assert "Access-Control-Allow-Origin" not in response_headers
    assert request("GET", "/api/runs")[2] == []


def test_same_origin_mutation_and_localhost_host_are_accepted(http_app):
    server, request, _ = http_app
    host = f"localhost:{server.server_port}"
    status, _, run = request(
        "POST", "/api/analyze", payload={}, headers={"Host": host, "Origin": f"http://{host}"}
    )
    assert status == 201
    assert run["created_by"] == "analyst"
    assert request("OPTIONS", "/api/analyze", headers={"Origin": f"http://{host}"})[0] == 405


@pytest.mark.parametrize(
    "body,headers,status",
    [
        (b"{", {"Content-Type": "application/json"}, 400),
        (b"[]", {"Content-Type": "application/json"}, 400),
        (b'{"reserve":NaN}', {"Content-Type": "application/json"}, 400),
        (b"\xff", {"Content-Type": "application/json"}, 400),
        (b"{}", {"Content-Type": "text/plain"}, 415),
        (b"{}", {"Content-Type": "application/json", "Content-Length": "-1"}, 400),
        (b"", {"Content-Type": "application/json", "Content-Length": "9" * 5000}, 400),
        (
            b"",
            {"Content-Type": "application/json", "Content-Length": str(2 * 1024 * 1024 + 1)},
            413,
        ),
        (b"", {"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, 400),
    ],
)
def test_json_and_body_boundaries_do_not_create_runs(http_app, body, headers, status):
    _, request, _ = http_app
    actual_status, _, error = request("POST", "/api/analyze", body=body, headers=headers)
    assert actual_status == status
    assert set(error) == {"error"}
    assert request("GET", "/api/runs")[2] == []


def test_content_length_is_required(http_app):
    server, _, _ = http_app
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.putrequest("POST", "/api/analyze")
        connection.putheader("Authorization", _auth("analyst", USERS["analyst"]["password"]))
        connection.putheader("Content-Type", "application/json")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 411
        assert json.loads(response.read())["error"] == "Content-Length is required."
    finally:
        connection.close()


def test_http_workflow_persists_separates_duties_and_exports_only_approval(http_app):
    _, request, store = http_app
    run = _create(request)
    route = f"/api/runs/{run['id']}"
    assert run["created_by"] == "analyst"
    assert run["status"] == "DRAFT" and run["revision"] == 1
    assert run["model_reserve"] == run["booked_reserve"] == 75
    assert request("GET", "/api/runs", user="reviewer")[2] == []
    assert request("GET", route, user="reviewer")[0] == 403
    assert request("GET", route, user="other")[0] == 403
    assert request("GET", route + "/export")[0] == 409
    override = {
        "origin_period": "2020-01-01",
        "reserve": 80,
        "reason": "Additional case information",
        "expected_revision": 1,
    }
    assert request("POST", route + "/override", user="reviewer", payload=override)[0] == 403
    assert request("POST", route + "/override", user="other", payload=override)[0] == 403
    status, _, edited = request("POST", route + "/override", payload=override)
    assert status == 200
    assert edited["booked_reserve"] == 105 and edited["model_reserve"] == 75
    assert edited["snapshot"] == SNAPSHOT
    assert edited["revision"] == 2
    assert request("POST", route + "/override", payload=override)[0] == 409
    assert (
        request("POST", route + "/submit", payload={"reason": "", "expected_revision": 2})[0] == 400
    )
    status, _, submitted = request(
        "POST", route + "/submit", payload={"reason": "Ready for review", "expected_revision": 2}
    )
    assert status == 200 and submitted["status"] == "SUBMITTED"
    assert request("GET", route, user="reviewer")[0] == 200
    decision = {"decision": "approve", "reason": "Evidence reviewed", "expected_revision": 3}
    assert request("POST", route + "/decision", payload=decision)[0] == 403
    status, _, approved = request("POST", route + "/decision", user="reviewer", payload=decision)
    assert status == 200 and approved["status"] == "APPROVED"
    assert approved["revision"] == 4
    status, headers, exported = request("GET", route + "/export", user="reviewer")
    assert status == 200
    assert headers["Content-Disposition"].startswith("attachment;")
    assert exported["exported_by"] == "reviewer"
    assert exported["snapshot"] == SNAPSHOT
    assert exported["booked_reserve"] == 105
    assert [event["action"] for event in exported["events"]] == [
        "create",
        "override",
        "submit",
        "approve",
    ]
    override["expected_revision"] = 4
    assert request("POST", route + "/override", payload=override)[0] == 409
    status, _, revised = request("POST", route + "/revise", payload={"expected_revision": 4})
    assert status == 200
    assert revised["parent_id"] == run["id"]
    assert revised["id"] != run["id"]
    assert revised["status"] == "DRAFT" and revised["revision"] == 1
    assert revised["overrides"] == approved["overrides"]
    assert revised["snapshot"] == SNAPSHOT
    assert request("GET", route)[2]["status"] == "APPROVED"
    # Read through another store instance to prove HTTP actions reached durable state.
    reopened = ReviewStore(store.path)
    actor = review_server.Actor("analyst", frozenset({"analyst"}))
    assert reopened.get_run(actor, run["id"])["events"] == approved["events"]


def test_dual_role_author_cannot_approve_their_own_run_and_rejection_can_be_revised(http_app):
    _, request, _ = http_app
    run = _create(request, "dual")
    route = f"/api/runs/{run['id']}"
    assert (
        request(
            "POST",
            route + "/submit",
            user="dual",
            payload={"reason": "Ready", "expected_revision": 1},
        )[0]
        == 200
    )
    payload = {"decision": "approve", "reason": "Reviewed", "expected_revision": 2}
    assert request("POST", route + "/decision", user="dual", payload=payload)[0] == 403
    payload["decision"] = "reject"
    status, _, rejected = request("POST", route + "/decision", user="reviewer", payload=payload)
    assert status == 200 and rejected["status"] == "REJECTED"
    assert request("GET", route + "/export", user="reviewer")[0] == 409
    assert (
        request("POST", route + "/revise", user="dual", payload={"expected_revision": 3})[2][
            "status"
        ]
        == "DRAFT"
    )


def test_unknown_routes_missing_fields_and_unexpected_errors_are_json(http_app, monkeypatch):
    _, request, _ = http_app
    assert request("GET", "/api/runs/missing")[0] == 404
    assert request("POST", "/api/unknown", payload={})[0] == 404
    assert request("DELETE", "/api/runs/missing")[0] == 501
    run = _create(request)
    status, _, error = request("POST", f"/api/runs/{run['id']}/submit", payload={})
    assert status == 400 and "expected_revision" in error["error"]

    def broken_analysis(payload):
        raise RuntimeError("secret internal details")

    monkeypatch.setattr(review_server.analysis, "analyze_request", broken_analysis)
    status, _, error = request("POST", "/api/analyze", payload={})
    assert status == 500
    assert error == {"error": "The request could not be completed."}


def test_normal_startup_requires_configuration_and_rejects_remote_binding(tmp_path, monkeypatch):
    monkeypatch.delenv("IBNR_REVIEW_USERS", raising=False)
    store = ReviewStore(tmp_path / "review.sqlite3")
    with pytest.raises(ValueError, match="IBNR_REVIEW_USERS"):
        review_server.create_server(store=store)
    with pytest.raises(ValueError, match="127.0.0.1"):
        review_server.create_server(host="0.0.0.0", store=store, users=USERS)
    with pytest.raises(ValueError, match="port"):
        review_server.create_server(port=-1, store=store, users=USERS)


@pytest.mark.parametrize(
    "config",
    [
        {},
        [],
        {"analyst": {"password": "", "roles": ["analyst"]}},
        {"analyst": {"password": "x", "roles": ["admin"]}},
        {"analyst": {"password": "x", "roles": "analyst"}},
        {" analyst ": {"password": "x", "roles": ["analyst"]}},
    ],
)
def test_invalid_user_configuration_is_refused(tmp_path, config):
    with pytest.raises(ValueError):
        review_server.create_server(store=ReviewStore(tmp_path / "review.sqlite3"), users=config)


def test_environment_and_explicit_demo_configuration(tmp_path, monkeypatch):
    store = ReviewStore(tmp_path / "review.sqlite3")
    monkeypatch.setenv("IBNR_REVIEW_USERS", json.dumps(USERS))
    server = review_server.create_server(store=store)
    try:
        assert not server.demo
        assert set(server.users) == set(USERS)
    finally:
        server.server_close()
    server = review_server.create_server(store=store, demo=True)
    try:
        assert server.demo
        assert set(server.users) == {"analyst", "reviewer"}
    finally:
        server.server_close()
    monkeypatch.setenv("IBNR_REVIEW_USERS", "not-json")
    with pytest.raises(ValueError, match="JSON"):
        review_server.create_server(store=store)
