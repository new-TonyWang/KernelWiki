# KernelWiki MCP Server — Client Configuration

## Overview

The KernelWiki MCP server exposes 3 tools (`wiki_query`, `wiki_get_page`, `wiki_grep`) over stdio JSON-RPC 2.0. No external dependencies beyond Python 3.9+ and PyYAML.

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BLACKWELL_WIKI_ROOT` | Auto-detected from script location | Override wiki root path |
| `MCP_LOG_FILE` | `/dev/null` | Path to write server logs (stderr is redirected) |

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
2. **No output**: The server uses newline-delimited JSON over stdio. Stderr is redirected; set `MCP_LOG_FILE` to see logs.
3. **Path traversal errors**: The server blocks any `lookup` that would resolve outside `WIKI_ROOT`. Error code: `PATH_OUTSIDE_ROOT`.
4. **Invalid regex**: Malformed regex patterns return `REGEX_ERROR` with a description of the problem.
5. **Test the server**: Run `bash scripts/test_mcp_smoke.sh` to verify the server works (25 fixture-based tests).
6. **Manual probe**: Send a single JSON-RPC message to verify:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 scripts/mcp_server.py 2>/dev/null
   ```
