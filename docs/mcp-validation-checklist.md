# KernelWiki MCP Server — Agent Validation Checklist

This checklist is a handoff artifact for FUT-1 (live end-to-end agent validation). It documents what to verify when testing the MCP server with real agents.

## Pre-requisites

- [ ] Python 3.9+ installed
- [ ] PyYAML installed (`pip install pyyaml`)
- [ ] Wiki root accessible (contains `data/tags.yaml` and `wiki/`)
- [ ] stdio smoke tests pass: `bash scripts/test_mcp_smoke.sh`
- [ ] HTTP smoke tests pass: `bash scripts/test_mcp_http_smoke.sh`

## Registration Commands

### Claude Code

```bash
claude mcp add kernelwiki -- python3 scripts/mcp_server.py
```

### Codex CLI

```bash
codex mcp add kernelwiki -- python3 scripts/mcp_server.py
```

### Codex CLI remote HTTP

Start the remote server:

```bash
BLACKWELL_WIKI_ROOT="$PWD" MCP_LOG_FILE=/tmp/kernelwiki-mcp.log \
  python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
```

Register from another machine:

```bash
codex mcp add kernelwiki-remote --url http://SERVER_HOST:8765/mcp
```

With bearer-token auth:

```bash
export KERNELWIKI_MCP_TOKEN='replace-with-a-long-random-token'
codex mcp add kernelwiki-remote \
  --url http://SERVER_HOST:8765/mcp \
  --bearer-token-env-var KERNELWIKI_MCP_TOKEN
```

With dynamic SQLite token CRUD:

```bash
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 add laptop
MCP_TOKEN_DB=data/mcp_tokens.sqlite3 MCP_ADMIN_TOKEN='admin-secret' \
  python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
```

### Claude Desktop

Add to `claude_desktop_config.json` (see `docs/mcp-client-config.md` for full snippet).

## Expected tools/list Output

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 scripts/mcp_server.py 2>/dev/null
```

HTTP equivalent:

```bash
curl -sS http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Expected response (abbreviated):

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {"name": "wiki_query", "description": "Search the KernelWiki knowledge base by keywords and filters. Returns ranked pages with titles, types, and key metadata.", "inputSchema": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string"}, "description": "Free-text keyword list"}, "..."  : "..."}, "required": []}},
      {"name": "wiki_get_page", "description": "Retrieve a wiki page by its id or relative path. Returns full content, frontmatter, and optionally artifact code files.", "inputSchema": {"type": "object", "properties": {"lookup": {"type": "string", "description": "Page id (e.g. kernel-flash-attention-4) or relative path"}, "..." : "..."}, "required": ["lookup"]}},
      {"name": "wiki_grep", "description": "Regex text search across wiki markdown files and optionally source code artifacts. Returns matching lines with context.", "inputSchema": {"type": "object", "properties": {"patterns": {"type": "array", "items": {"type": "string"}, "description": "Regex pattern(s) — all must match unless any_match is true"}, "..." : "..."}, "required": ["patterns"]}}
    ]
  }
}
```

## Sample tools/call Invocation and Response

### Query

```bash
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"wiki_query","arguments":{"query":["tcgen05"],"limit":3,"compact":true}}}' | python3 scripts/mcp_server.py 2>/dev/null
```

Expected response shape:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [{"type": "text", "text": "{\"ok\":true,\"total_hits\":N,\"returned\":3,\"truncated\":true,\"data\":[...]}"}],
    "isError": false
  }
}
```

### Page retrieval

```bash
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"wiki_get_page","arguments":{"lookup":"hw-tcgen05-mma","body_only":true}}}' | python3 scripts/mcp_server.py 2>/dev/null
```

### Not-found error

```bash
echo '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"wiki_get_page","arguments":{"lookup":"nonexistent"}}}' | python3 scripts/mcp_server.py 2>/dev/null
```

Expected: `{"ok":false,"error_code":"PAGE_NOT_FOUND","message":"No page found for 'nonexistent'"}`

## Claude Code Validation

1. **Setup**: `claude mcp add kernelwiki -- python3 scripts/mcp_server.py`
2. **Tool discovery**: Run `claude mcp list` — confirm `kernelwiki` appears with 3 tools
3. **Query**: Ask Claude to "search KernelWiki for tcgen05 techniques" — verify it calls `wiki_query`
4. **Page retrieval**: Ask "show me the page for hw-tcgen05-mma" — verify `wiki_get_page` call
5. **Grep**: Ask "find all mentions of tcgen05.fence in the wiki" — verify `wiki_grep` call
6. **Error handling**: Ask for a nonexistent page — verify `PAGE_NOT_FOUND` error response
7. **Artifact loading**: Ask to "show code for [page-with-artifacts]" with `include_code: true`

## Codex CLI Validation

1. **Setup**: `codex mcp add kernelwiki -- python3 scripts/mcp_server.py`
2. **Tool listing**: Verify tools are discovered
3. **Query**: Request a keyword search — verify structured response
4. **Page retrieval**: Request a specific page — verify content returned
5. **Grep**: Request a regex search — verify matches returned

## Remote HTTP Validation

1. **Start**: `python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765`
2. **Health**: `curl -fsS http://SERVER_HOST:8765/healthz`
3. **Tool discovery**: POST `tools/list` to `http://SERVER_HOST:8765/mcp`
4. **Client registration**: `codex mcp add kernelwiki-remote --url http://SERVER_HOST:8765/mcp`
5. **Auth path**: If `MCP_AUTH_TOKEN` is set, verify missing `Authorization` returns HTTP 401 and `Authorization: Bearer <token>` succeeds
6. **Network path**: From a second machine, repeat health and `tools/list`

## Dynamic Token CRUD Validation

1. **Create DB token**: `python3 scripts/mcp_token_admin.py --db /tmp/kw-tokens.sqlite add laptop`
2. **Start with DB**: `MCP_TOKEN_DB=/tmp/kw-tokens.sqlite MCP_ADMIN_TOKEN=admin python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765`
3. **Read/list**: `python3 scripts/mcp_token_admin.py --db /tmp/kw-tokens.sqlite list` and `GET /admin/tokens` with admin bearer
4. **Auth works**: Call `POST /mcp` with the created token and verify success
5. **Update/disable**: Disable the token by CLI or `PATCH /admin/tokens/{id}` and verify the same token returns HTTP 401 without restarting the server
6. **Enable**: Re-enable and verify the same token succeeds
7. **Rotate**: Rotate the token, verify old token fails and new token succeeds
8. **Create new live token**: Add another token while the server is running and verify it succeeds immediately
9. **Delete**: Delete a token and verify it fails immediately
10. **No secret leakage**: Confirm list/get responses do not include plaintext token values; only add/rotate responses do

## Claude Desktop Validation

1. **Setup**: Add config to `claude_desktop_config.json` per docs
2. **Tool visibility**: Confirm tools appear in Claude's tool list
3. **Basic query**: Run a search query through the chat interface

## Protocol Compliance

- [ ] Initialize handshake completes (protocolVersion: 2024-11-05)
- [ ] notifications/initialized accepted without response
- [ ] tools/list returns 3 tool definitions with valid inputSchema
- [ ] tools/call returns `{content: [{type: "text", text: "<JSON>"}], isError: bool}`
- [ ] Unknown methods return JSON-RPC error -32601
- [ ] Malformed JSON returns parse error -32700
- [ ] Non-object request returns invalid request -32600
- [ ] ping returns empty result `{}`
- [ ] HTTP `POST /mcp` accepts single JSON-RPC requests and batches
- [ ] HTTP `GET /healthz` returns server and wiki root metadata
- [ ] HTTP bearer-token mode rejects unauthenticated requests with 401
- [ ] SQLite token DB mode validates tokens dynamically without restart
- [ ] Token CLI supports create/list/get/update/enable/disable/rotate/delete
- [ ] HTTP admin API supports create/list/get/update/rotate/delete when `MCP_ADMIN_TOKEN` is set
- [ ] Plaintext token values are stored hashed and only printed on create/rotate

## Security

- [ ] Path traversal (`../../etc/passwd`) returns `PATH_OUTSIDE_ROOT` error
- [ ] Null bytes in lookup are rejected with `INVALID_PARAMS`
- [ ] Stderr is never written to stdout (protocol integrity)
- [ ] Large responses are truncated to 500 KiB budget
- [ ] Artifact file symlinks that escape WIKI_ROOT are silently skipped

## Domain Error Codes

- [ ] `PATH_OUTSIDE_ROOT`: lookup path escapes wiki root
- [ ] `REGEX_ERROR`: invalid regex pattern in grep
- [ ] `PAGE_NOT_FOUND`: no page matches lookup
- [ ] `INVALID_PARAMS`: missing/invalid parameter, unknown tool, bad enum value
- [ ] `INTERNAL_ERROR`: unexpected server exception

## Edge Cases

- [ ] Empty query returns all pages (sorted by path)
- [ ] Invalid regex in grep returns `REGEX_ERROR` with description
- [ ] Unknown tool name returns `INVALID_PARAMS`
- [ ] Parameters exceeding limits are clamped (e.g., limit > 200 → 200)
- [ ] Missing required parameters return `INVALID_PARAMS`
- [ ] Boolean/string values for integer params return `INVALID_PARAMS`
- [ ] `hw-tcgen05` resolves to `hw-tcgen05-mma` via prefix matching
- [ ] Server stays alive after errors (verified by multi-request session in smoke tests)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No response from server | Stderr corruption, bad import | Set `MCP_LOG_FILE=/tmp/mcp.log` and check logs |
| `BLACKWELL_WIKI_ROOT` error | Wiki root not found | Set env var or run from repo root |
| Tools not listed in agent | Server name mismatch | Use `kernelwiki` as the server name |
| `PATH_OUTSIDE_ROOT` | Lookup contains `..` or resolves outside root | Use page IDs instead of paths |
| Truncated response | Output exceeds 500 KiB | Reduce `limit` or use `compact` mode |
| `REGEX_ERROR` | Invalid regex syntax | Escape special chars: `[](){}.*+?^$|\` |
