# KernelWiki MCP Server — Agent Validation Checklist

This checklist is a handoff artifact for FUT-1 (live end-to-end agent validation). It documents what to verify when testing the MCP server with real agents.

## Pre-requisites

- [ ] Python 3.9+ installed
- [ ] PyYAML installed (`pip install pyyaml`)
- [ ] Wiki root accessible (contains `data/tags.yaml` and `wiki/`)
- [ ] Smoke tests pass: `bash scripts/test_mcp_smoke.sh`

## Claude Code Validation

1. **Setup**: Add MCP config per `docs/mcp-client-config.md`
2. **Tool discovery**: Run `claude mcp list` — confirm `kernel-wiki` appears with 3 tools
3. **Query**: Ask Claude to "search KernelWiki for tcgen05 techniques" — verify it calls `wiki_query`
4. **Page retrieval**: Ask "show me the page for hw-tcgen05-mma" — verify `wiki_get_page` call
5. **Grep**: Ask "find all mentions of tcgen05.fence in the wiki" — verify `wiki_grep` call
6. **Error handling**: Ask for a nonexistent page — verify graceful error response
7. **Artifact loading**: Ask to "show code for [page-with-artifacts]" with `include_code: true`

## Codex CLI Validation

1. **Setup**: `codex mcp add kernel-wiki -- python3 scripts/mcp_server.py`
2. **Tool listing**: Verify tools are discovered
3. **Query**: Request a keyword search — verify structured response
4. **Page retrieval**: Request a specific page — verify content returned
5. **Grep**: Request a regex search — verify matches returned

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
- [ ] ping returns empty result

## Security

- [ ] Path traversal (`../../etc/passwd`) returns `invalid_params` error
- [ ] Null bytes in lookup are rejected
- [ ] Stderr is never written to stdout (protocol integrity)
- [ ] Large responses are truncated to 500 KiB budget

## Edge Cases

- [ ] Empty query returns all pages (sorted by path)
- [ ] Invalid regex in grep returns descriptive error
- [ ] Unknown tool name returns `unknown_tool` error
- [ ] Parameters exceeding limits are clamped (e.g., limit > 200 → 200)
- [ ] Missing required parameters return clear validation errors
