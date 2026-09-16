"""Loopback-only HTTP adapter for the reserving review reference application.

This deliberately uses only the Python standard library. Authentication is
configured at startup, and the persistent store makes every workflow decision.
Run ``python -m apps.reserving_review --demo`` for explicitly labelled demo users.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import analysis
from .store import Actor, ReviewError, ReviewStore

MAX_BODY_BYTES = 2 * 1024 * 1024
# A library error can name one clause per candidate per date, which has reached
# 45 KB on one small CSV. The browser writes an error straight into the page.
MAX_ERROR_CHARS = 2000
STATIC_DIR = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
DEMO_USERS = {
    "analyst": {"password": "analyst", "roles": ["analyst"]},
    "reviewer": {"password": "reviewer", "roles": ["reviewer"]},
}
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)
_UNKNOWN_PASSWORD = hashlib.sha256(b"unknown-review-user").digest()


@dataclass(frozen=True)
class _User:
    password_digest: bytes
    roles: frozenset[str]


class _HTTPError(Exception):
    def __init__(self, status: int, message: str):
        self.status_code = status
        super().__init__(message)


def _short_error(text: str) -> str:
    """Bound one error message so a long library error stays readable."""
    if len(text) <= MAX_ERROR_CHARS:
        return text
    omitted = len(text) - MAX_ERROR_CHARS
    return f"{text[:MAX_ERROR_CHARS]}... ({omitted} more characters of this message)"


def _users(config: dict | None, *, demo: bool) -> dict[str, _User]:
    if config is None:
        raw = os.environ.get("IBNR_REVIEW_USERS")
        if demo:
            if raw:
                raise ValueError(
                    "IBNR_REVIEW_USERS is set and --demo would replace those accounts "
                    "with analyst/analyst and reviewer/reviewer. Unset IBNR_REVIEW_USERS "
                    "or start without --demo."
                )
            config = DEMO_USERS
        else:
            if not raw:
                raise ValueError("Set IBNR_REVIEW_USERS or explicitly use --demo.")
            try:
                config = json.loads(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError("IBNR_REVIEW_USERS must be a JSON object.") from exc
    if not isinstance(config, dict) or not config:
        raise ValueError("User configuration must be a nonempty JSON object.")
    result = {}
    for name, account in config.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or ":" in name
            or any(ord(char) < 32 for char in name)
            or not isinstance(account, dict)
        ):
            raise ValueError("Each user needs a nonempty username and account object.")
        password, roles = account.get("password"), account.get("roles")
        if not isinstance(password, str) or not password:
            raise ValueError(f"User {name!r} needs a nonempty password.")
        if (
            not isinstance(roles, list)
            or not roles
            or any(role not in ("analyst", "reviewer") for role in roles)
        ):
            raise ValueError(f"User {name!r} roles must contain analyst and/or reviewer.")
        result[name] = _User(hashlib.sha256(password.encode()).digest(), frozenset(roles))
    return result


class ReviewHTTPServer(ThreadingHTTPServer):
    """A local server whose store and credentials are fixed at construction."""

    daemon_threads = True

    def __init__(self, address, *, store, users, demo):
        self.store = store
        self.users = users
        self.demo = demo
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer
    server_version = "IBNRReview/1"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format, *args):
        # Request paths and headers can contain user data. Do not log them.
        pass

    def send_error(self, code, message=None, explain=None):
        self._json(code, {"error": self.responses.get(code, ("Request failed",))[0]})

    def _send(self, status, content, content_type, *, attachment=False):
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="approved-review.json"')
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status, value, *, attachment=False, canonical=False):
        # The export is written in the same canonical form the store hashes, so
        # a reader can recompute snapshot_hash from the bytes they received.
        separators = (",", ":") if canonical else None
        content = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=canonical,
            separators=separators,
        ).encode("utf-8")
        self._send(status, content, "application/json; charset=utf-8", attachment=attachment)

    def _discard_body(self):
        # Closing a Windows socket with unread POST bytes can discard the error
        # response. Read and discard only bounded bodies, and never wait on a
        # slow sender.
        if self.command != "POST" or getattr(self, "_body_consumed", False):
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
            return
        try:
            size = int(lengths[0])
        except ValueError:
            return
        if size > MAX_BODY_BYTES:
            return
        self.connection.settimeout(0.1)
        try:
            self.rfile.read(size)
        except (OSError, TimeoutError):
            pass
        finally:
            self._body_consumed = True
            self.connection.settimeout(30)

    def _local_request(self):
        allowed = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0].lower() not in allowed:
            raise _HTTPError(403, "Host must identify this local review server.")
        origins = self.headers.get_all("Origin", [])
        if origins and (
            len(origins) != 1 or origins[0].lower() not in {f"http://{host}" for host in allowed}
        ):
            raise _HTTPError(403, "Origin must identify this local review server.")

    def _actor(self):
        headers = self.headers.get_all("Authorization", [])
        try:
            if len(headers) != 1:
                raise ValueError
            scheme, token = headers[0].split(" ", 1)
            if scheme.lower() != "basic":
                raise ValueError
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
            name, password = decoded.split(":", 1)
        except (ValueError, UnicodeError, binascii.Error) as exc:
            raise _HTTPError(401, "Valid credentials are required.") from exc
        account = self.server.users.get(name)
        expected = account.password_digest if account else _UNKNOWN_PASSWORD
        valid = hmac.compare_digest(hashlib.sha256(password.encode()).digest(), expected)
        if not valid or account is None:
            raise _HTTPError(401, "Valid credentials are required.")
        return Actor(name, account.roles)

    @staticmethod
    def _require(actor, role):
        if role not in actor.roles:
            raise _HTTPError(403, f"The {role} role is required.")

    def _body(self):
        if self.headers.get_all("Transfer-Encoding", []):
            raise _HTTPError(400, "Transfer-Encoding is not supported.")
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise _HTTPError(411, "Content-Length is required.")
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
            raise _HTTPError(400, "Content-Length must be a nonnegative integer.")
        try:
            size = int(lengths[0])
        except ValueError as exc:
            raise _HTTPError(400, "Content-Length must be a nonnegative integer.") from exc
        if size > MAX_BODY_BYTES:
            raise _HTTPError(413, "Request body exceeds the 2 MB limit.")
        content_types = self.headers.get_all("Content-Type", [])
        if (
            len(content_types) != 1
            or content_types[0].split(";", 1)[0].lower() != "application/json"
        ):
            raise _HTTPError(415, "Content-Type must be application/json.")
        data = self.rfile.read(size)
        self._body_consumed = True
        if len(data) != size:
            raise _HTTPError(400, "Request body is incomplete.")

        def reject_constant(value):
            raise ValueError(f"Invalid JSON constant: {value}")

        try:
            payload = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
        except (ValueError, UnicodeError) as exc:
            raise _HTTPError(400, "Request body must contain valid JSON.") from exc
        if not isinstance(payload, dict):
            raise _HTTPError(400, "Request body must be a JSON object.")
        return payload

    def _dispatch(self):
        self._local_request()
        path = urlsplit(self.path).path
        if self.command == "GET" and path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            try:
                content = (STATIC_DIR / filename).read_bytes()
            except FileNotFoundError as exc:
                raise _HTTPError(404, "Static asset not found.") from exc
            self._send(200, content, content_type)
            return
        if self.command == "GET" and path == "/api/config":
            self._json(200, {"demo": self.server.demo})
            return
        actor = self._actor()
        if self.command == "GET":
            self._get(path, actor)
        elif self.command == "POST":
            self._post(path, actor)
        else:
            raise _HTTPError(405, "Method not allowed.")

    def _get(self, path, actor):
        if path == "/api/me":
            self._json(200, {"name": actor.name, "roles": sorted(actor.roles)})
        elif path == "/api/runs":
            self._json(200, self.server.store.list_runs(actor))
        elif path == "/api/demo":
            self._require(actor, "analyst")
            self._json(200, analysis.demo_request())
        else:
            match = re.fullmatch(r"/api/runs/([^/]+)(/export)?", path)
            if match is None:
                raise _HTTPError(404, "Route not found.")
            run_id, export = match.groups()
            if export:
                result = self.server.store.export_approved(actor, run_id)
                self._json(200, result, attachment=True, canonical=True)
            else:
                self._json(200, self.server.store.get_run(actor, run_id))

    def _post(self, path, actor):
        if path == "/api/analyze":
            self._require(actor, "analyst")
            payload = self._body()
            snapshot = analysis.analyze_request(payload)
            result = self.server.store.create_run(actor, snapshot)
            self._json(201, result)
            return
        match = re.fullmatch(r"/api/runs/([^/]+)/(override|submit|decision|revise)", path)
        if match is None:
            raise _HTTPError(404, "Route not found.")
        run_id, operation = match.groups()
        self._require(actor, "reviewer" if operation == "decision" else "analyst")
        payload = self._body()
        revision = payload["expected_revision"]
        store = self.server.store
        if operation == "override":
            result = store.set_override(
                actor,
                run_id,
                origin_period=payload["origin_period"],
                reserve=payload["reserve"],
                reason=payload["reason"],
                expected_revision=revision,
            )
        elif operation == "submit":
            result = store.submit(
                actor, run_id, reason=payload["reason"], expected_revision=revision
            )
        elif operation == "decision":
            result = store.decide(
                actor,
                run_id,
                decision=payload["decision"],
                reason=payload["reason"],
                expected_revision=revision,
            )
        else:
            result = store.revise(actor, run_id, expected_revision=revision)
        self._json(200, result)

    def _handle(self):
        try:
            self._dispatch()
        except (ReviewError, _HTTPError) as exc:
            self._discard_body()
            self._json(exc.status_code, {"error": _short_error(str(exc))})
        except KeyError as exc:
            self._discard_body()
            self._json(400, {"error": _short_error(f"Missing required field: {exc.args[0]}")})
        except ValueError as exc:
            self._discard_body()
            self._json(400, {"error": _short_error(str(exc))})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True
        except Exception:
            # The response stays generic, but an unexpected failure must leave a
            # trace somewhere. Request paths and headers are still not logged.
            print("Unhandled error in the reserving review server:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._discard_body()
            self._json(500, {"error": "The request could not be completed."})

    do_GET = _handle
    do_POST = _handle
    do_OPTIONS = _handle


def create_server(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    store: ReviewStore | None = None,
    users: dict[str, Any] | None = None,
    demo: bool = False,
) -> ReviewHTTPServer:
    """Create an unstarted loopback server; call ``serve_forever`` to run it.

    ``users`` accepts the same mapping as IBNR_REVIEW_USERS. Omit it to load the
    environment configuration, or pass ``demo=True`` for the two demo accounts.
    Tests can provide an isolated ``ReviewStore`` and request an ephemeral port.
    """
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("The review reference app must bind to 127.0.0.1.")
    if not isinstance(demo, bool):
        raise ValueError("demo must be a boolean.")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535.")
    accounts = _users(users, demo=demo)
    return ReviewHTTPServer(
        (host, port),
        store=store if store is not None else ReviewStore(Path(".ibnr-review/review.sqlite3")),
        users=accounts,
        demo=demo,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local reserving review reference app.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path(".ibnr-review"))
    parser.add_argument(
        "--demo", action="store_true", help="Use explicitly labelled demo accounts."
    )
    args = parser.parse_args(argv)
    try:
        # Validate credentials before creating persistent state.
        accounts = _users(None, demo=args.demo)
        if not 0 <= args.port <= 65535:
            raise ValueError("port must be between 0 and 65535.")
        server = ReviewHTTPServer(
            ("127.0.0.1", args.port),
            store=ReviewStore(args.data_dir / "review.sqlite3"),
            users=accounts,
            demo=args.demo,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Reserving review: http://127.0.0.1:{server.server_port}")
    if args.demo:
        print("DEMO MODE: analyst / analyst and reviewer / reviewer. Use synthetic data only.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
