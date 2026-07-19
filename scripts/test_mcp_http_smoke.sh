#!/usr/bin/env bash
# Smoke test for KernelWiki Streamable HTTP MCP server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOST="${MCP_HTTP_TEST_HOST:-127.0.0.1}"
PORT="${MCP_HTTP_TEST_PORT:-18765}"
BASE_URL="http://$HOST:$PORT"
MCP_URL="$BASE_URL/mcp"
LOG_FILE="${MCP_HTTP_TEST_LOG:-/tmp/kernelwiki-mcp-http-test.log}"

cleanup() {
  for pid in "${SERVER_PID:-}" "${AUTH_SERVER_PID:-}" "${DB_SERVER_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

cd "$REPO_ROOT"
BLACKWELL_WIKI_ROOT="$REPO_ROOT" MCP_LOG_FILE=/dev/null \
  python3 "$SCRIPT_DIR/mcp_http_server.py" --host "$HOST" --port "$PORT" >"$LOG_FILE.out" 2>"$LOG_FILE.err" &
SERVER_PID=$!

for _ in $(seq 1 50); do
  if curl -fsS "$BASE_URL/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
curl -fsS "$BASE_URL/healthz" >/dev/null

python3 - "$MCP_URL" <<'PYEOF'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1]

def post(payload, status=200):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            actual_status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        actual_status = e.code
    assert actual_status == status, (actual_status, body)
    return None if actual_status == 204 or not body else json.loads(body)

init = post({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"http-smoke","version":"1"}}})
assert init["result"]["serverInfo"]["name"] == "kernel-wiki", init

tools = post({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
names = [t["name"] for t in tools["result"]["tools"]]
assert names == ["wiki_query", "wiki_get_page", "wiki_grep"], names

query = post({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"wiki_query","arguments":{"query":["tcgen05"],"limit":3,"compact":True}}})
qenv = json.loads(query["result"]["content"][0]["text"])
assert qenv["ok"] is True and qenv["returned"] > 0, qenv

page = post({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"wiki_get_page","arguments":{"lookup":"hw-tcgen05-mma","body_only":True}}})
penv = json.loads(page["result"]["content"][0]["text"])
assert penv["ok"] is True and "body" in penv["data"], penv

grep = post({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"wiki_grep","arguments":{"patterns":["tcgen05"],"scope":"wiki","limit":2}}})
genv = json.loads(grep["result"]["content"][0]["text"])
assert genv["ok"] is True and genv["returned"] > 0, genv

batch = post([
    {"jsonrpc":"2.0","id":6,"method":"ping","params":{}},
    {"jsonrpc":"2.0","method":"notifications/initialized","params":{}},
])
assert batch == [{"jsonrpc":"2.0","id":6,"result":{}}], batch

bad = post({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"wiki_grep","arguments":{"patterns":["["]}}})
berr = json.loads(bad["result"]["content"][0]["text"])
assert berr["ok"] is False and berr["error_code"] == "REGEX_ERROR", berr

print("HTTP MCP basic smoke tests passed")
PYEOF

cleanup
SERVER_PID=""

AUTH_PORT=$((PORT + 1))
AUTH_URL="http://$HOST:$AUTH_PORT/mcp"
AUTH_TOKEN="kernelwiki-http-smoke-token"
BLACKWELL_WIKI_ROOT="$REPO_ROOT" MCP_LOG_FILE=/dev/null MCP_AUTH_TOKEN="$AUTH_TOKEN" \
  python3 "$SCRIPT_DIR/mcp_http_server.py" --host "$HOST" --port "$AUTH_PORT" >"$LOG_FILE.auth.out" 2>"$LOG_FILE.auth.err" &
AUTH_SERVER_PID=$!

for _ in $(seq 1 50); do
  if curl -fsS "http://$HOST:$AUTH_PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
curl -fsS "http://$HOST:$AUTH_PORT/healthz" >/dev/null

python3 - "$AUTH_URL" "$AUTH_TOKEN" <<'PYEOF'
import json
import sys
import urllib.error
import urllib.request

url, token = sys.argv[1], sys.argv[2]
payload = {"jsonrpc":"2.0","id":1,"method":"ping","params":{}}
data = json.dumps(payload).encode("utf-8")

def request(headers):
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")

status, body = request({})
assert status == 401, (status, body)
status, body = request({"Authorization": f"Bearer {token}"})
assert status == 200, (status, body)
assert json.loads(body) == {"jsonrpc":"2.0","id":1,"result":{}}, body
print("HTTP MCP auth smoke tests passed")
PYEOF


TOKEN_DB=$(mktemp /tmp/kernelwiki-mcp-token-db.XXXXXX.sqlite)
python3 "$SCRIPT_DIR/mcp_token_admin.py" --db "$TOKEN_DB" --json add db-client >"$LOG_FILE.db-token.json"
DB_TOKEN=$(python3 - "$LOG_FILE.db-token.json" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1]))["token"]["token"])
PYEOF
)
DB_PORT=$((PORT + 2))
DB_URL="http://$HOST:$DB_PORT/mcp"
ADMIN_URL="http://$HOST:$DB_PORT/admin/tokens"
ADMIN_TOKEN="kernelwiki-http-admin-smoke-token"
BLACKWELL_WIKI_ROOT="$REPO_ROOT" MCP_LOG_FILE=/dev/null MCP_TOKEN_DB="$TOKEN_DB" MCP_ADMIN_TOKEN="$ADMIN_TOKEN" \
  python3 "$SCRIPT_DIR/mcp_http_server.py" --host "$HOST" --port "$DB_PORT" >"$LOG_FILE.db.out" 2>"$LOG_FILE.db.err" &
DB_SERVER_PID=$!

for _ in $(seq 1 50); do
  if curl -fsS "http://$HOST:$DB_PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
curl -fsS "http://$HOST:$DB_PORT/healthz" >/dev/null

python3 - "$DB_URL" "$ADMIN_URL" "$DB_TOKEN" "$ADMIN_TOKEN" "$SCRIPT_DIR/mcp_token_admin.py" "$TOKEN_DB" <<'PYEOF'
import json
import subprocess
import sys
import urllib.error
import urllib.request

mcp_url, admin_url, db_token, admin_token, admin_cli, db_path = sys.argv[1:]

def request(url, payload=None, token=None, method=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return e.code, json.loads(body) if body else None

def ping(token):
    return request(mcp_url, {"jsonrpc":"2.0","id":1,"method":"ping","params":{}}, token=token)

status, body = ping(None)
assert status == 401, (status, body)
status, body = ping(db_token)
assert status == 200 and body["result"] == {}, (status, body)

# CLI disable affects the already-running server without restart.
subprocess.check_call([sys.executable, admin_cli, "--db", db_path, "--json", "disable", "1"], stdout=subprocess.DEVNULL)
status, body = ping(db_token)
assert status == 401, (status, body)
subprocess.check_call([sys.executable, admin_cli, "--db", db_path, "--json", "enable", "1"], stdout=subprocess.DEVNULL)
status, body = ping(db_token)
assert status == 200, (status, body)

# HTTP admin CRUD can add, list, update/disable, rotate, and delete tokens.
status, body = request(admin_url, {"name":"admin-created","note":"smoke"}, token=admin_token)
assert status == 201, (status, body)
created = body["token"]
new_token = created["token"]
new_id = created["id"]
status, body = ping(new_token)
assert status == 200, (status, body)

status, body = request(admin_url, token=admin_token)
assert status == 200 and any(t["id"] == new_id for t in body["tokens"]), body

status, body = request(f"{admin_url}/{new_id}", {"enabled": False, "note": "disabled"}, token=admin_token, method="PATCH")
assert status == 200 and body["token"]["enabled"] is False, (status, body)
status, body = ping(new_token)
assert status == 401, (status, body)

status, body = request(f"{admin_url}/{new_id}/rotate", {}, token=admin_token)
assert status == 200 and body["token"]["token"] != new_token, (status, body)
rotated = body["token"]["token"]
request(f"{admin_url}/{new_id}", {"enabled": True}, token=admin_token, method="PATCH")
status, body = ping(rotated)
assert status == 200, (status, body)

status, body = request(f"{admin_url}/{new_id}", token=admin_token, method="DELETE")
assert status == 200 and body["deleted"] is True, (status, body)
status, body = ping(rotated)
assert status == 401, (status, body)

print("HTTP MCP token DB CRUD smoke tests passed")
PYEOF

echo "HTTP MCP all smoke tests passed"
