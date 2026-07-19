#!/usr/bin/env python3
"""KernelWiki MCP Tool Server — Streamable HTTP transport.

This is a small, dependency-free HTTP adapter around ``scripts/mcp_server.py``.
It exposes the same JSON-RPC 2.0 MCP methods as the stdio server:

  - initialize
  - ping
  - tools/list
  - tools/call

The implementation intentionally stays stateless: every HTTP POST contains one
JSON-RPC request (or a JSON-RPC batch) and receives an ``application/json``
response.  Server-to-client streaming is not required by KernelWiki's current
tools, so GET-based SSE streams are advertised as unsupported with HTTP 405.

Examples:

    python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765

    curl -s http://127.0.0.1:8765/healthz

    curl -s http://127.0.0.1:8765/mcp \\
      -H 'Content-Type: application/json' \\
      -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

Authentication can be enabled with either:

  - ``--auth-token`` / ``MCP_AUTH_TOKEN`` for one static token, or
  - ``--token-db`` / ``MCP_TOKEN_DB`` for SQLite-backed dynamic tokens.

When a token DB is configured, token CRUD can be done while the server is
running via ``scripts/mcp_token_admin.py`` or the optional admin HTTP API
(``--admin-token`` / ``MCP_ADMIN_TOKEN``).
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


# Ensure the stdio MCP implementation can import its sibling service modules
# when this script is executed from outside the repository root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mcp_server import (  # noqa: E402
    PROTOCOL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    WIKI_ROOT,
    _jsonrpc_error,
    handle_request,
)
from mcp_token_store import TokenStore  # noqa: E402


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PATH = "/mcp"
MAX_REQUEST_BYTES = 2_000_000


def _write_log(msg: str) -> None:
    """Write diagnostics to the original stderr, not MCP stdout."""
    print(f"[mcp_http_server] {msg}", file=sys.__stderr__, flush=True)


def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _normalize_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    return origin.rstrip("/")


class KernelWikiMCPHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying runtime configuration for request handlers."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        *,
        mcp_path: str,
        auth_token: str | None,
        admin_token: str | None,
        token_store: TokenStore | None,
        allow_origins: set[str],
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.mcp_path = mcp_path
        self.auth_token = auth_token
        self.admin_token = admin_token
        self.token_store = token_store
        self.allow_origins = allow_origins


class MCPHTTPRequestHandler(BaseHTTPRequestHandler):
    """Minimal Streamable HTTP request handler for MCP JSON-RPC messages."""

    server: KernelWikiMCPHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        _write_log("%s - %s" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------------
    # Header helpers
    # ------------------------------------------------------------------

    def _origin_allowed(self) -> bool:
        origin = _normalize_origin(self.headers.get("Origin"))
        if not origin:
            return True
        return "*" in self.server.allow_origins or origin in self.server.allow_origins

    def _send_common_headers(self) -> None:
        self.send_header("Server", f"{SERVER_NAME}/{SERVER_VERSION}")
        self.send_header("MCP-Protocol-Version", PROTOCOL_VERSION)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")

        origin = _normalize_origin(self.headers.get("Origin"))
        if origin and self._origin_allowed():
            if "*" in self.server.allow_origins:
                self.send_header("Access-Control-Allow-Origin", "*")
            else:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, MCP-Protocol-Version")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str = "application/json; charset=utf-8",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: Any, *, extra_headers: dict[str, str] | None = None) -> None:
        self._send_bytes(status, _json_dumps(payload), extra_headers=extra_headers)

    def _send_plain(self, status: HTTPStatus, message: str, *, extra_headers: dict[str, str] | None = None) -> None:
        self._send_bytes(status, message.encode("utf-8"), content_type="text/plain; charset=utf-8", extra_headers=extra_headers)

    # ------------------------------------------------------------------
    # Request validation
    # ------------------------------------------------------------------

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _check_path(self) -> bool:
        path = self._path()
        if path in {self.server.mcp_path, "/healthz", "/readyz"} or path.startswith("/admin/tokens"):
            return True
        self._send_plain(HTTPStatus.NOT_FOUND, "not found\n")
        return False

    def _bearer_token(self) -> str | None:
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return None
        token = header[len(prefix):].strip()
        return token or None

    def _check_mcp_auth(self) -> bool:
        bearer = self._bearer_token()
        if self.server.token_store:
            if bearer and self.server.token_store.verify_bearer(bearer):
                return True
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "unauthorized", "message": "missing, invalid, or disabled bearer token"},
                extra_headers={"WWW-Authenticate": 'Bearer realm="kernelwiki-mcp"'},
            )
            return False

        token = self.server.auth_token
        if not token:
            return True
        if bearer and hmac.compare_digest(bearer, token):
            return True
        self._send_json(
            HTTPStatus.UNAUTHORIZED,
            {"error": "unauthorized", "message": "missing or invalid bearer token"},
            extra_headers={"WWW-Authenticate": 'Bearer realm="kernelwiki-mcp"'},
        )
        return False

    def _check_admin_auth(self) -> bool:
        token = self.server.admin_token
        if not token:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "message": "admin API is disabled; set MCP_ADMIN_TOKEN to enable it"},
            )
            return False
        bearer = self._bearer_token()
        if bearer and hmac.compare_digest(bearer, token):
            return True
        self._send_json(
            HTTPStatus.UNAUTHORIZED,
            {"error": "unauthorized", "message": "missing or invalid admin bearer token"},
            extra_headers={"WWW-Authenticate": 'Bearer realm="kernelwiki-mcp-admin"'},
        )
        return False

    def _check_origin_or_forbid(self) -> bool:
        if self._origin_allowed():
            return True
        self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden", "message": "origin not allowed"})
        return False

    # ------------------------------------------------------------------
    # HTTP methods
    # ------------------------------------------------------------------

    def do_OPTIONS(self) -> None:
        if not self._check_path():
            return
        if not self._check_origin_or_forbid():
            return
        self._send_bytes(HTTPStatus.NO_CONTENT, b"")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/healthz", "/readyz"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "server": SERVER_NAME,
                    "version": SERVER_VERSION,
                    "protocolVersion": PROTOCOL_VERSION,
                    "wikiRoot": str(WIKI_ROOT),
                    "auth": {
                        "mode": "token-db" if self.server.token_store else ("static" if self.server.auth_token else "off"),
                        "adminApi": bool(self.server.admin_token),
                    },
                },
            )
            return
        if path.startswith("/admin/tokens"):
            if not self._check_origin_or_forbid() or not self._check_admin_auth():
                return
            self._handle_admin_get(path)
            return
        if path != self.server.mcp_path:
            self._send_plain(HTTPStatus.NOT_FOUND, "not found\n")
            return
        if not self._check_origin_or_forbid() or not self._check_mcp_auth():
            return
        self._send_json(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {
                "error": "method_not_allowed",
                "message": "This stateless KernelWiki MCP server accepts JSON-RPC over HTTP POST. GET/SSE streams are not implemented.",
            },
            extra_headers={"Allow": "POST, OPTIONS"},
        )

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.rstrip("/").startswith("/admin/tokens"):
            if not self._check_origin_or_forbid() or not self._check_admin_auth():
                return
            payload = self._read_json_body()
            if isinstance(payload, tuple):
                status, body = payload
                self._send_json(status, body)
                return
            self._handle_admin_write("POST", self._path(), payload)
            return
        if path != self.server.mcp_path:
            self._send_plain(HTTPStatus.NOT_FOUND, "not found\n")
            return
        if not self._check_origin_or_forbid() or not self._check_mcp_auth():
            return

        payload = self._read_json_body()
        if isinstance(payload, tuple):
            status, body = payload
            self._send_json(status, body)
            return

        response = self._dispatch_payload(payload)
        if response is None:
            # JSON-RPC notification only: no JSON-RPC response. 204 is the
            # clearest HTTP-level representation for stateless transport.
            self._send_bytes(HTTPStatus.NO_CONTENT, b"")
            return
        self._send_json(HTTPStatus.OK, response)

    def do_PATCH(self) -> None:
        path = self._path()
        if not path.startswith("/admin/tokens"):
            self._send_plain(HTTPStatus.NOT_FOUND, "not found\n")
            return
        if not self._check_origin_or_forbid() or not self._check_admin_auth():
            return
        payload = self._read_json_body()
        if isinstance(payload, tuple):
            status, body = payload
            self._send_json(status, body)
            return
        self._handle_admin_write("PATCH", path, payload)

    def do_DELETE(self) -> None:
        path = self._path()
        if not path.startswith("/admin/tokens"):
            self._send_plain(HTTPStatus.NOT_FOUND, "not found\n")
            return
        if not self._check_origin_or_forbid() or not self._check_admin_auth():
            return
        self._handle_admin_write("DELETE", path, {})

    def _read_json_body(self) -> Any | tuple[HTTPStatus, Any]:
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            return HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "unsupported_media_type", "message": "Content-Type must be application/json"}

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            return HTTPStatus.BAD_REQUEST, _jsonrpc_error(None, -32600, "Invalid Request: empty body")
        if content_length > MAX_REQUEST_BYTES:
            return HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large", "message": f"request body exceeds {MAX_REQUEST_BYTES} bytes"}

        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            return HTTPStatus.OK, _jsonrpc_error(None, -32700, f"Parse error: {exc}")

    # ------------------------------------------------------------------
    # Admin token CRUD API
    # ------------------------------------------------------------------

    def _require_token_store(self) -> TokenStore | None:
        if self.server.token_store:
            return self.server.token_store
        self._send_json(
            HTTPStatus.CONFLICT,
            {"ok": False, "error": "token_db_disabled", "message": "set MCP_TOKEN_DB or --token-db to enable token CRUD"},
        )
        return None

    def _token_id_from_path(self, path: str) -> int | None:
        parts = path.strip("/").split("/")
        if len(parts) < 3 or parts[:2] != ["admin", "tokens"]:
            return None
        try:
            return int(parts[2])
        except ValueError:
            return None

    def _handle_admin_get(self, path: str) -> None:
        store = self._require_token_store()
        if not store:
            return
        token_id = self._token_id_from_path(path)
        if token_id is None:
            self._send_json(HTTPStatus.OK, {"ok": True, "tokens": store.list_tokens()})
            return
        row = store.get_token(token_id)
        if not row:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found", "message": f"token id {token_id} not found"})
            return
        self._send_json(HTTPStatus.OK, {"ok": True, "token": row})

    def _handle_admin_write(self, method: str, path: str, payload: Any) -> None:
        store = self._require_token_store()
        if not store:
            return
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid_request", "message": "JSON body must be an object"})
            return
        token_id = self._token_id_from_path(path)
        try:
            if method == "POST" and path == "/admin/tokens":
                if not payload.get("name"):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid_request", "message": "name is required"})
                    return
                row = store.create_token(
                    payload["name"],
                    token=payload.get("token"),
                    enabled=bool(payload.get("enabled", True)),
                    note=payload.get("note"),
                )
                self._send_json(HTTPStatus.CREATED, {"ok": True, "token": row})
                return
            if method == "POST" and path.endswith("/rotate") and token_id is not None:
                row = store.rotate_token(token_id)
                self._send_json(HTTPStatus.OK, {"ok": True, "token": row})
                return
            if method == "PATCH" and token_id is not None:
                row = store.update_token(
                    token_id,
                    name=payload.get("name"),
                    enabled=payload.get("enabled") if "enabled" in payload else None,
                    note=payload.get("note") if "note" in payload else None,
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "token": row})
                return
            if method == "DELETE" and token_id is not None:
                deleted = store.delete_token(token_id)
                status = HTTPStatus.OK if deleted else HTTPStatus.NOT_FOUND
                self._send_json(status, {"ok": deleted, "deleted": deleted, "id": token_id})
                return
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found", "message": str(exc)})
            return
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": type(exc).__name__, "message": str(exc)})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found", "message": "unknown admin token endpoint"})

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------

    def _dispatch_payload(self, payload: Any) -> Any | None:
        if isinstance(payload, list):
            if not payload:
                return _jsonrpc_error(None, -32600, "Invalid Request: empty batch")
            responses = []
            for item in payload:
                responses.append(self._dispatch_one(item))
            responses = [r for r in responses if r is not None]
            return responses or None
        return self._dispatch_one(payload)

    def _dispatch_one(self, msg: Any) -> Any | None:
        if not isinstance(msg, dict):
            return _jsonrpc_error(None, -32600, "Invalid Request: not an object")
        return handle_request(msg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve KernelWiki MCP over Streamable HTTP")
    parser.add_argument("--host", default=os.environ.get("MCP_HTTP_HOST", DEFAULT_HOST), help=f"bind host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_HTTP_PORT", DEFAULT_PORT)), help=f"bind port (default: {DEFAULT_PORT})")
    parser.add_argument("--path", default=os.environ.get("MCP_HTTP_PATH", DEFAULT_PATH), help=f"MCP endpoint path (default: {DEFAULT_PATH})")
    parser.add_argument("--auth-token", default=os.environ.get("MCP_AUTH_TOKEN"), help="optional bearer token; also read from MCP_AUTH_TOKEN")
    parser.add_argument("--token-db", default=os.environ.get("MCP_TOKEN_DB"), help="optional SQLite DB path for dynamic bearer tokens")
    parser.add_argument("--admin-token", default=os.environ.get("MCP_ADMIN_TOKEN"), help="optional admin bearer token for HTTP token CRUD API")
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="allowed browser Origin; repeatable. Use '*' to allow all. Non-browser clients usually send no Origin.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mcp_path = args.path if args.path.startswith("/") else "/" + args.path
    allow_origins = {_normalize_origin(o) or "" for o in args.allow_origin}
    allow_origins.discard("")
    token_store = TokenStore(args.token_db) if args.token_db else None
    if token_store and args.auth_token:
        inserted = token_store.ensure_bootstrap_token(
            "env-bootstrap",
            args.auth_token,
            note="Bootstrap token imported from MCP_AUTH_TOKEN/--auth-token",
        )
        if inserted:
            _write_log("inserted bootstrap token from MCP_AUTH_TOKEN into token DB as 'env-bootstrap'")

    httpd = KernelWikiMCPHTTPServer(
        (args.host, args.port),
        MCPHTTPRequestHandler,
        mcp_path=mcp_path,
        auth_token=args.auth_token,
        admin_token=args.admin_token,
        token_store=token_store,
        allow_origins=allow_origins,
    )
    host, port = httpd.server_address[:2]
    _write_log(
        f"KernelWiki MCP HTTP server listening on http://{host}:{port}{mcp_path} "
        f"(WIKI_ROOT={WIKI_ROOT}, auth={'token-db' if token_store else ('static' if args.auth_token else 'off')}, "
        f"admin={'on' if args.admin_token else 'off'})"
    )
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        _write_log("shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
