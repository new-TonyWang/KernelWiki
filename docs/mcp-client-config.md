# KernelWiki MCP Server — Client Configuration

## Overview

The KernelWiki MCP server exposes 3 tools (`wiki_query`, `wiki_get_page`, `wiki_grep`) through two transports:

- **stdio JSON-RPC 2.0** via `scripts/mcp_server.py` for local agent integrations.
- **Streamable HTTP** via `scripts/mcp_http_server.py` for remote deployment.

No external dependencies beyond Python 3.9+ and PyYAML.

## Claude Code

Add to your project's `.mcp.json` (or `~/.claude/mcp.json` for global):

```json
{
  "mcpServers": {
    "kernelwiki": {
      "command": "python3",
      "args": ["scripts/mcp_server.py"],
      "cwd": "/path/to/KernelWiki"
    }
  }
}
```

Or use the CLI:

```bash
claude mcp add kernelwiki -- python3 scripts/mcp_server.py
```

For VS Code integration, add to `.vscode/settings.json`:

```json
{
  "claude.mcpServers": {
    "kernelwiki": {
      "command": "python3",
      "args": ["scripts/mcp_server.py"],
      "cwd": "/path/to/KernelWiki"
    }
  }
}
```

## Codex CLI

```bash
codex mcp add kernelwiki -- python3 scripts/mcp_server.py
```

Or add to `~/.codex/config.toml`:

```toml
[mcp_servers.kernelwiki]
command = "python3"
args = ["scripts/mcp_server.py"]
cwd = "/path/to/KernelWiki"
```

### Codex CLI remote HTTP

On the server machine:

```bash
cd /path/to/KernelWiki
python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
```

On the client machine:

```bash
codex mcp add kernelwiki-remote --url http://SERVER_HOST:8765/mcp
```

If bearer-token authentication is enabled on the server:

```bash
# Server
MCP_AUTH_TOKEN='replace-with-a-long-random-token' \
  python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765

# Client
export KERNELWIKI_MCP_TOKEN='replace-with-a-long-random-token'
codex mcp add kernelwiki-remote \
  --url http://SERVER_HOST:8765/mcp \
  --bearer-token-env-var KERNELWIKI_MCP_TOKEN
```

For untrusted networks, put the HTTP server behind HTTPS (for example nginx or
Caddy) and use an `https://.../mcp` URL.

For tokens that can be added, disabled, rotated, or deleted without restarting
the server, use the SQLite token DB mode documented below.

## Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kernelwiki": {
      "command": "python3",
      "args": ["/path/to/KernelWiki/scripts/mcp_server.py"],
      "env": {
        "BLACKWELL_WIKI_ROOT": "/path/to/KernelWiki"
      }
    }
  }
}
```

## Remote HTTP Server

Start a stateless Streamable HTTP MCP endpoint:

```bash
cd /path/to/KernelWiki
BLACKWELL_WIKI_ROOT=/path/to/KernelWiki \
MCP_LOG_FILE=/tmp/kernelwiki-mcp.log \
python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
```

Useful endpoints:

- `GET /healthz` — health/readiness probe.
- `POST /mcp` — JSON-RPC MCP endpoint.

Manual probe:

```bash
curl -sS http://SERVER_HOST:8765/healthz

curl -sS http://SERVER_HOST:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

### systemd example

Create `/etc/systemd/system/kernelwiki-mcp.service`:

```ini
[Unit]
Description=KernelWiki MCP HTTP Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/KernelWiki
Environment=BLACKWELL_WIKI_ROOT=/path/to/KernelWiki
Environment=MCP_LOG_FILE=/var/log/kernelwiki-mcp.log
# Optional authentication:
# Environment=MCP_AUTH_TOKEN=replace-with-a-long-random-token
# Dynamic token DB:
# Environment=MCP_TOKEN_DB=/var/lib/kernelwiki/mcp_tokens.sqlite3
# Environment=MCP_ADMIN_TOKEN=replace-with-a-long-random-admin-token
ExecStart=/usr/bin/python3 /path/to/KernelWiki/scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

## Dynamic Token Management

For production remote access, prefer SQLite-backed tokens over one static
`MCP_AUTH_TOKEN`.  The MCP server checks the database on each authenticated
request, so changes take effect while the service is running.

### CLI CRUD

Initialize and create the first token:

```bash
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 init
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 add laptop
```

The `add` and `rotate` commands print the token secret exactly once. Store it
in your client environment or secret manager.

Start the MCP HTTP server with the token DB:

```bash
MCP_TOKEN_DB=data/mcp_tokens.sqlite3 \
python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
```

Manage tokens without restarting the MCP server:

```bash
# 查
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 list
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 get 1

# 增
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 add ci-runner --note "CI access"

# 改
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 update 1 --name laptop-new --note "renamed"
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 disable 1
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 enable 1
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 rotate 1

# 删
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 delete 1 -y
```

Client registration with a DB token is the same bearer-token flow:

```bash
export KERNELWIKI_MCP_TOKEN='<token printed by add/rotate>'
codex mcp add kernelwiki-remote \
  --url http://SERVER_HOST:8765/mcp \
  --bearer-token-env-var KERNELWIKI_MCP_TOKEN
```

If you pass both `MCP_TOKEN_DB` and `MCP_AUTH_TOKEN`, the server inserts
`MCP_AUTH_TOKEN` into the DB once as `env-bootstrap` if that name does not
already exist. This helps migrate from static-token mode to DB-token mode.

### HTTP Admin CRUD API

Enable the admin API by setting `MCP_ADMIN_TOKEN` in addition to `MCP_TOKEN_DB`:

```bash
MCP_TOKEN_DB=data/mcp_tokens.sqlite3 \
MCP_ADMIN_TOKEN='replace-with-a-long-random-admin-token' \
python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
```

Then manage tokens over HTTP:

```bash
ADMIN='replace-with-a-long-random-admin-token'

# 查
curl -sS -H "Authorization: Bearer $ADMIN" \
  http://SERVER_HOST:8765/admin/tokens

# 增
curl -sS -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"ci-runner","note":"CI access"}' \
  http://SERVER_HOST:8765/admin/tokens

# 改
curl -sS -X PATCH -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false,"note":"temporarily disabled"}' \
  http://SERVER_HOST:8765/admin/tokens/1

# 轮换 token secret
curl -sS -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{}' \
  http://SERVER_HOST:8765/admin/tokens/1/rotate

# 删
curl -sS -X DELETE -H "Authorization: Bearer $ADMIN" \
  http://SERVER_HOST:8765/admin/tokens/1
```

Notes:

- Token values are stored as salted PBKDF2 hashes; plaintext tokens are only
  shown on `add`/`rotate`.
- `list`/`GET /admin/tokens` never reveals token secrets.
- Keep `MCP_ADMIN_TOKEN` separate from client MCP tokens and expose it only on
  trusted networks or behind HTTPS.

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now kernelwiki-mcp
sudo systemctl status kernelwiki-mcp
```

### nginx reverse proxy example

```nginx
server {
    listen 443 ssl http2;
    server_name kernelwiki.example.com;

    ssl_certificate /etc/letsencrypt/live/kernelwiki.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/kernelwiki.example.com/privkey.pem;

    location /mcp {
        proxy_pass http://127.0.0.1:8765/mcp;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /healthz {
        proxy_pass http://127.0.0.1:8765/healthz;
    }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BLACKWELL_WIKI_ROOT` | Auto-detected from script location | Override wiki root path |
| `MCP_LOG_FILE` | `/dev/null` | Path to write server logs (stderr is redirected) |
| `MCP_HTTP_HOST` | `127.0.0.1` | HTTP bind host for `scripts/mcp_http_server.py` |
| `MCP_HTTP_PORT` | `8765` | HTTP bind port |
| `MCP_HTTP_PATH` | `/mcp` | HTTP MCP endpoint path |
| `MCP_AUTH_TOKEN` | unset | Optional bearer token required by HTTP clients |
| `MCP_TOKEN_DB` | unset | Optional SQLite DB path for dynamic bearer tokens |
| `MCP_ADMIN_TOKEN` | unset | Optional admin bearer token for HTTP token CRUD API |

## Available Tools

### wiki_query

Search the knowledge base by keywords and filters.

**Parameters:**
- `query` (array of strings): Free-text keywords
- `type` (string): Filter by page type (kernel, technique, hardware, pattern, etc.)
- `tag` (string): Filter by tag
- `vendor` (string): Filter by vendor (nvidia, ascend, biren, all)
- `repo` (string): Filter by source repo
- `language` (string): Filter by language/DSL
- `architecture` (string): Filter by architecture
- `symptom` (string): Filter by pattern symptom
- `confidence` (string): Filter by confidence level (verified, source-reported, inferred, experimental)
- `has_code` (boolean): Only pages with source code artifacts
- `limit` (integer, 1-200, default 10): Max results
- `compact` (boolean): Compact output format

### wiki_get_page

Retrieve a page by id, alias, or path.

**Parameters:**
- `lookup` (string, required): Page id, alias, or relative path
- `body_only` (boolean): Return only body text
- `frontmatter_only` (boolean): Return only YAML frontmatter
- `include_code` (boolean): Include artifact bundle files
- `follow_sources` (boolean): Include cited source excerpts

### wiki_grep

Regex text search across wiki files.

**Parameters:**
- `patterns` (array of strings, required): Regex patterns
- `scope` (string): wiki, sources, all, or artifacts (default: all)
- `context` (integer, 0-10, default 1): Context lines
- `any_match` (boolean): Match if ANY pattern matches (default: all must)
- `limit` (integer, 1-100, default 20): Max files reported
- `ext` (string): Comma-separated extra extensions

## Response Format

All tool responses are JSON with this envelope:

```json
{
  "ok": true,
  "data": { "..." : "..." },
  "total_hits": 42,
  "returned": 10,
  "truncated": true
}
```

Error responses use uppercase domain error codes:

```json
{
  "ok": false,
  "error_code": "PAGE_NOT_FOUND",
  "message": "No page found for 'xyz'"
}
```

**Domain error codes:**

| Code | Meaning |
|------|---------|
| `PATH_OUTSIDE_ROOT` | Lookup path escapes the wiki root |
| `REGEX_ERROR` | Invalid regex pattern |
| `PAGE_NOT_FOUND` | No page matches the lookup |
| `INVALID_PARAMS` | Missing/invalid parameter or unknown tool |
| `INTERNAL_ERROR` | Unexpected server error |

## Output Budgets

| Parameter | Value |
|-----------|-------|
| Max response size | 500 KiB |
| Max results (query) | 200 |
| Max files (grep) | 100 |
| Max artifact files | 100 |
| Max file size (artifacts) | 512 KiB |

## Troubleshooting

1. **Server doesn't start**: Ensure `BLACKWELL_WIKI_ROOT` points to a valid wiki root (must contain `data/tags.yaml` and `wiki/`).
2. **No output**: The stdio server uses newline-delimited JSON over stdio. Stderr is redirected; set `MCP_LOG_FILE` to see logs.
3. **Path traversal errors**: The server blocks any `lookup` that would resolve outside `WIKI_ROOT`. Error code: `PATH_OUTSIDE_ROOT`.
4. **Invalid regex**: Malformed regex patterns return `REGEX_ERROR` with a description of the problem.
5. **Test the stdio server**: Run `bash scripts/test_mcp_smoke.sh` to verify the server works.
6. **Test the HTTP server**: Run `bash scripts/test_mcp_http_smoke.sh`.
7. **Manual stdio probe**: Send a single JSON-RPC message to verify:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 scripts/mcp_server.py 2>/dev/null
   ```
8. **Manual HTTP probe**:
   ```bash
   curl -sS http://127.0.0.1:8765/mcp \
     -H 'Content-Type: application/json' \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
   ```
