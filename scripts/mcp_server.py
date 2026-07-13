#!/usr/bin/env python3
"""KernelWiki MCP Tool Server — stdio JSON-RPC 2.0.

Exposes three tools:
  - wiki_query:    keyword search with filters
  - wiki_get_page: retrieve a page by id or path
  - wiki_grep:     regex text search across wiki files

Protocol: newline-delimited JSON-RPC 2.0 over stdin/stdout.
No external MCP SDK dependency.
"""

import json
import re
import sys
import os
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Domain error types
# ---------------------------------------------------------------------------

class DomainError(Exception):
    """Typed domain error with an uppercase error code."""

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)


class PathOutsideRoot(DomainError):
    def __init__(self, message="path traversal blocked"):
        super().__init__("PATH_OUTSIDE_ROOT", message)


class RegexError(DomainError):
    def __init__(self, message):
        super().__init__("REGEX_ERROR", message)


class PageNotFound(DomainError):
    def __init__(self, message):
        super().__init__("PAGE_NOT_FOUND", message)


class InvalidParams(DomainError):
    def __init__(self, message):
        super().__init__("INVALID_PARAMS", message)

# Redirect stderr early so service module imports that might print
# (e.g. _wiki_root.py on error) don't corrupt the JSON-RPC channel.
_original_stderr = sys.stderr
_log_fd = os.environ.get("MCP_LOG_FILE")
if _log_fd:
    try:
        _log_file = open(_log_fd, "a", encoding="utf-8")
    except Exception:
        _log_file = open(os.devnull, "w")
else:
    _log_file = open(os.devnull, "w")
sys.stderr = _log_file

# Now import service modules (after stderr redirect)
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from _wiki_root import WIKI_ROOT
    from wiki_query_service import (
        load_all_pages, filter_pages, score_keyword_match, format_result,
        infer_vendor,
    )
    from wiki_page_service import (
        find_page, split_frontmatter, resolve_artifact_dir,
        load_artifact_files, ARTIFACT_EXTS,
    )
    from wiki_grep_service import search_wiki
except SystemExit:
    # _wiki_root.py calls sys.exit(2) on failure — catch it
    sys.stderr = _original_stderr
    print("FATAL: Could not resolve wiki root. Set BLACKWELL_WIKI_ROOT.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Output budget constants
# ---------------------------------------------------------------------------

MAX_RESPONSE_CHARS = 500_000
MAX_RESULTS = 200
MAX_GREP_HITS = 100
MAX_GREP_PER_FILE = 10
MAX_ARTIFACT_FILES = 100
MAX_FILE_SIZE = 512_000

# ---------------------------------------------------------------------------
# Server metadata
# ---------------------------------------------------------------------------

SERVER_NAME = "kernel-wiki"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "wiki_query",
        "description": "Search the KernelWiki knowledge base by keywords and filters. Returns ranked pages with titles, types, and key metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Free-text keyword list",
                },
                "type": {"type": "string", "description": "Filter by page type (kernel, technique, hardware, pattern, language, migration, pr, blog, doc, contest)"},
                "tag": {"type": "string", "description": "Filter by tag"},
                "vendor": {"type": "string", "description": "Filter by vendor (nvidia, ascend, biren, all). Auto-inferred when omitted."},
                "repo": {"type": "string", "description": "Filter by source repo (partial match)"},
                "language": {"type": "string", "description": "Filter by language/DSL"},
                "architecture": {"type": "string", "description": "Filter by architecture (sm100, sm90, ascend910b)"},
                "symptom": {"type": "string", "description": "Filter by pattern symptom"},
                "confidence": {"type": "string", "description": "Filter by confidence level"},
                "has_code": {"type": "boolean", "description": "Only return pages with source code artifacts", "default": False},
                "limit": {"type": "integer", "description": "Max results (1-200, default 10)", "default": 10},
                "compact": {"type": "boolean", "description": "Compact one-line output", "default": False},
            },
            "required": [],
        },
    },
    {
        "name": "wiki_get_page",
        "description": "Retrieve a wiki page by its id or relative path. Returns full content, frontmatter, and optionally artifact code files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lookup": {"type": "string", "description": "Page id (e.g. kernel-flash-attention-4) or relative path"},
                "body_only": {"type": "boolean", "description": "Return only the body text", "default": False},
                "frontmatter_only": {"type": "boolean", "description": "Return only the YAML frontmatter", "default": False},
                "include_code": {"type": "boolean", "description": "Include artifact bundle files", "default": False},
                "follow_sources": {"type": "boolean", "description": "Include excerpts from cited sources", "default": False},
            },
            "required": ["lookup"],
        },
    },
    {
        "name": "wiki_grep",
        "description": "Regex text search across wiki markdown files and optionally source code artifacts. Returns matching lines with context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Regex pattern(s) — all must match unless any_match is true",
                },
                "scope": {
                    "type": "string",
                    "enum": ["wiki", "sources", "all", "artifacts"],
                    "description": "Search scope (default: all)",
                    "default": "all",
                },
                "context": {"type": "integer", "description": "Context lines (0-10, default 1)", "default": 1},
                "any_match": {"type": "boolean", "description": "Match if ANY pattern matches (default: all must)", "default": False},
                "limit": {"type": "integer", "description": "Max files reported (1-100, default 20)", "default": 20},
                "ext": {"type": "string", "description": "Comma-separated extra extensions (without dots)"},
            },
            "required": ["patterns"],
        },
    },
]


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------

def _safe_lookup(lookup_str):
    """Validate that a page lookup doesn't escape WIKI_ROOT.

    Returns the sanitized lookup string, or raises PathOutsideRoot/InvalidParams.
    """
    if not isinstance(lookup_str, str) or not lookup_str.strip():
        raise InvalidParams("lookup must be a non-empty string")
    lookup_str = lookup_str.strip()
    # Block obvious traversal attempts
    if "\0" in lookup_str:
        raise InvalidParams("null bytes not allowed in lookup")
    # For path-style lookups, validate containment
    if "/" in lookup_str or lookup_str.endswith(".md"):
        candidate = (WIKI_ROOT / lookup_str).resolve()
        if not candidate.is_relative_to(WIKI_ROOT.resolve()):
            raise PathOutsideRoot()
    return lookup_str


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _clamp_int(val, lo, hi, default, name="parameter"):
    """Clamp an integer parameter to [lo, hi], using default if None.

    Raises InvalidParams for clearly wrong types (strings that aren't numeric).
    """
    if val is None:
        return default
    if isinstance(val, bool):
        raise InvalidParams(f"{name} must be an integer, got boolean")
    if isinstance(val, str):
        raise InvalidParams(f"{name} must be an integer, got string")
    try:
        val = int(val)
    except (TypeError, ValueError):
        raise InvalidParams(f"{name} must be an integer")
    return max(lo, min(hi, val))


def _validate_bool(val, name, default=False):
    """Validate optional boolean parameter.

    Accepts only JSON booleans (True/False) and None. Rejects strings,
    integers, and other types to prevent 'false' being truthy.
    """
    if val is None:
        return default
    if not isinstance(val, bool):
        raise InvalidParams(f"{name} must be a boolean, got {type(val).__name__}")
    return val


def _validate_str(val, name, allowed=None, max_len=200):
    """Validate optional string parameter."""
    if val is None:
        return None
    if not isinstance(val, str):
        raise InvalidParams(f"{name} must be a string")
    val = val.strip()
    if len(val) > max_len:
        raise InvalidParams(f"{name} too long (max {max_len} chars)")
    if allowed and val not in allowed:
        raise InvalidParams(f"{name} must be one of: {', '.join(allowed)}")
    return val or None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_wiki_query(params):
    """Handle wiki_query tool call."""
    query_list = params.get("query") or []
    if isinstance(query_list, str):
        query_list = [query_list]
    if not isinstance(query_list, list):
        raise InvalidParams("query must be a list of strings")
    for q in query_list:
        if not isinstance(q, str):
            raise InvalidParams("each query item must be a string")

    limit = _clamp_int(params.get("limit"), 1, MAX_RESULTS, 10, "limit")
    compact = _validate_bool(params.get("compact"), "compact", False)
    has_code = _validate_bool(params.get("has_code"), "has_code", False)

    filter_params = {
        "type": _validate_str(params.get("type"), "type"),
        "tag": _validate_str(params.get("tag"), "tag"),
        "vendor": _validate_str(params.get("vendor"), "vendor"),
        "repo": _validate_str(params.get("repo"), "repo"),
        "language": _validate_str(params.get("language"), "language"),
        "architecture": _validate_str(params.get("architecture"), "architecture"),
        "symptom": _validate_str(params.get("symptom"), "symptom"),
        "confidence": _validate_str(params.get("confidence"), "confidence",
                                     allowed={"verified", "source-reported", "inferred", "experimental"}),
        "has_code": has_code,
        "query": query_list,
    }

    # Auto-infer vendor
    if not filter_params["vendor"]:
        inferred = infer_vendor(filter_params)
        if inferred:
            filter_params["vendor"] = inferred

    pages = load_all_pages()
    pages = filter_pages(pages, filter_params)

    # Score by keywords
    keywords = []
    for q in query_list:
        for tok in re.split(r"\s+", str(q).strip()):
            if tok:
                keywords.append(tok)
    if keywords:
        for p in pages:
            p["_score"] = score_keyword_match(p["fm"], p["body"], keywords)
        pages = [p for p in pages if p["_score"] > 0]
        pages.sort(key=lambda x: (-x["_score"], x["path"]))
    else:
        pages.sort(key=lambda x: x["path"])

    total = len(pages)
    pages = pages[:limit]

    results = []
    for p in pages:
        results.append({
            "formatted": format_result(p, compact=compact),
            "path": p["path"],
            "id": p["fm"].get("id", ""),
            "title": p["fm"].get("title", ""),
            "type": p.get("_ptype", "unknown"),
            "score": p.get("_score", 0),
        })

    envelope = {
        "ok": True,
        "total_hits": total,
        "returned": len(results),
        "truncated": total > limit,
        "data": results,
    }
    return _make_text_response(envelope)


def handle_wiki_get_page(params):
    """Handle wiki_get_page tool call."""
    lookup = params.get("lookup")
    if not lookup:
        raise InvalidParams("lookup is required")
    lookup = _safe_lookup(lookup)

    body_only = _validate_bool(params.get("body_only"), "body_only", False)
    frontmatter_only = _validate_bool(params.get("frontmatter_only"), "frontmatter_only", False)
    include_code = _validate_bool(params.get("include_code"), "include_code", False)
    follow_sources = _validate_bool(params.get("follow_sources"), "follow_sources", False)

    page_path = find_page(lookup)
    if not page_path:
        raise PageNotFound(f"No page found for '{lookup}'")
    # Belt-and-suspenders: verify returned path is within WIKI_ROOT
    if not page_path.resolve().is_relative_to(WIKI_ROOT.resolve()):
        raise PathOutsideRoot()

    content = page_path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(content)

    result = {
        "path": str(page_path.relative_to(WIKI_ROOT)),
        "id": fm.get("id", "") if fm else "",
        "title": fm.get("title", "") if fm else "",
    }

    if frontmatter_only:
        result["frontmatter"] = fm
    elif body_only:
        result["body"] = body
    else:
        result["content"] = content
        result["frontmatter"] = fm

    if follow_sources and fm:
        source_excerpts = _collect_source_excerpts(fm)
        result["source_excerpts"] = source_excerpts

    if include_code and fm:
        ad, ad_path, is_fallback = resolve_artifact_dir(page_path, fm)
        if ad_path and ad_path.resolve().is_relative_to(WIKI_ROOT.resolve()) and ad_path.is_dir():
            files = load_artifact_files(ad_path,
                                         max_files=MAX_ARTIFACT_FILES,
                                         max_file_size=MAX_FILE_SIZE,
                                         containment_root=WIKI_ROOT)
            result["artifact_dir"] = ad
            result["artifact_dir_fallback"] = is_fallback
            result["artifact_files"] = [
                {"path": f["rel_path"], "size": f["size"],
                 "truncated": f["truncated"],
                 "content": f["content"]}
                for f in files
            ]

    envelope = {
        "ok": True,
        "total_hits": 1,
        "returned": 1,
        "truncated": False,
        "data": result,
    }
    return _make_text_response(envelope)


def _collect_source_excerpts(fm):
    """Collect source excerpts for --follow-sources equivalent.

    Handles all three source reference types matching get_page.py:
      - sources: list of page IDs
      - source: list of path strings or dicts with path/anchor
      - source_refs: list of dicts with source_id/path/anchor
    """
    excerpts = []
    seen = set()

    def _resolve_and_append(lookup_str, detail=None):
        src_page = None
        if "/" in lookup_str or lookup_str.endswith(".md"):
            p = (WIKI_ROOT / lookup_str).resolve()
            if p.is_relative_to(WIKI_ROOT.resolve()) and p.is_file():
                src_page = p
        if src_page is None:
            src_page = find_page(lookup_str)
        if src_page:
            src_content = src_page.read_text(encoding="utf-8")
            _, src_body = split_frontmatter(src_content)
            entry = {
                "id": lookup_str,
                "path": str(src_page.relative_to(WIKI_ROOT)),
                "excerpt": ((src_body or "")[:500]).strip(),
            }
        else:
            entry = {"id": lookup_str, "path": None, "excerpt": None}
        if detail:
            entry["detail"] = detail
        excerpts.append(entry)

    for src_id in fm.get("sources", []) or []:
        key = ("source-id", str(src_id))
        if key in seen:
            continue
        seen.add(key)
        _resolve_and_append(str(src_id))

    for src in fm.get("source", []) or []:
        if isinstance(src, dict) and src.get("path"):
            lookup = src["path"]
            anchor = src.get("anchor", "")
        elif isinstance(src, str):
            lookup = src
            anchor = ""
        else:
            continue
        key = ("source-path", lookup, anchor)
        if key in seen:
            continue
        seen.add(key)
        _resolve_and_append(str(lookup), detail=anchor or None)

    for src in fm.get("source_refs", []) or []:
        if not isinstance(src, dict) or not src.get("source_id"):
            continue
        label = src["source_id"]
        detail = " / ".join(str(x) for x in (src.get("path"), src.get("anchor")) if x)
        key = ("source-ref", label, detail)
        if key in seen:
            continue
        seen.add(key)
        _resolve_and_append(str(label), detail=detail or None)

    return excerpts


def handle_wiki_grep(params):
    """Handle wiki_grep tool call."""
    patterns = params.get("patterns")
    if not patterns or not isinstance(patterns, list):
        raise InvalidParams("patterns must be a non-empty list of regex strings")
    if len(patterns) > 20:
        raise InvalidParams("too many patterns (max 20)")

    # Validate regex patterns
    for p in patterns:
        if not isinstance(p, str):
            raise InvalidParams("each pattern must be a string")
        try:
            re.compile(p)
        except re.error as e:
            raise RegexError(f"invalid regex {p!r}: {e}")

    scope = _validate_str(params.get("scope"), "scope",
                           allowed={"wiki", "sources", "all", "artifacts"}) or "all"
    context = _clamp_int(params.get("context"), 0, 10, 1, "context")
    any_match = _validate_bool(params.get("any_match"), "any_match", False)
    limit = _clamp_int(params.get("limit"), 1, MAX_GREP_HITS, 20, "limit")

    ext_set = None
    ext_raw = params.get("ext")
    if ext_raw is not None:
        if not isinstance(ext_raw, str):
            raise InvalidParams("ext must be a string")
        ext_set = {"." + e.strip().lstrip(".").lower()
                    for e in ext_raw.split(",") if e.strip()} or None

    results, total_matching = search_wiki(
        patterns, scope=scope, context=context, any_match=any_match,
        exts=ext_set, limit=limit, per_file_limit=MAX_GREP_PER_FILE,
    )

    envelope = {
        "ok": True,
        "total_hits": total_matching,
        "returned": len(results),
        "truncated": total_matching > len(results),
        "data": results,
    }
    return _make_text_response(envelope)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _make_text_response(envelope):
    """Build MCP tools/call response content from an envelope dict."""
    text = json.dumps(envelope, ensure_ascii=False)
    if len(text) > MAX_RESPONSE_CHARS:
        envelope_trunc = {
            "ok": True,
            "truncated": True,
            "total_hits": envelope.get("total_hits", 0),
            "returned": 0,
            "message": "Response truncated to budget limit",
        }
        text = json.dumps(envelope_trunc, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _make_error_response(code, message):
    """Build an MCP tool-domain error response."""
    envelope = {"ok": False, "error_code": code, "message": message}
    text = json.dumps(envelope, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "isError": True}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 transport
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "wiki_query": handle_wiki_query,
    "wiki_get_page": handle_wiki_get_page,
    "wiki_grep": handle_wiki_grep,
}


def _jsonrpc_error(req_id, code, message):
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _jsonrpc_result(req_id, result):
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


def handle_request(msg):
    """Route a parsed JSON-RPC message and return a response dict (or None for notifications)."""
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    # Notifications (no "id" key) get no response per JSON-RPC 2.0
    is_notification = "id" not in msg

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
        return _jsonrpc_result(req_id, result)

    if method.startswith("notifications/"):
        return None  # notification, no response

    if is_notification:
        return None  # any notification without "id" gets no response

    if method == "tools/list":
        return _jsonrpc_result(req_id, {"tools": TOOLS})

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return _jsonrpc_result(req_id, _make_error_response(
                "INVALID_PARAMS", f"Unknown tool: {tool_name}"))
        try:
            result = handler(tool_args)
        except DomainError as e:
            result = _make_error_response(e.code, e.message)
        except Exception as e:
            _log(f"Tool error in {tool_name}: {traceback.format_exc()}")
            result = _make_error_response("INTERNAL_ERROR",
                                           "Internal server error")
        return _jsonrpc_result(req_id, result)

    if method == "ping":
        return _jsonrpc_result(req_id, {})

    # Unknown method
    return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


def _log(msg):
    """Write a log message to the log file (never stdout)."""
    try:
        _log_file.write(f"[mcp_server] {msg}\n")
        _log_file.flush()
    except Exception:
        pass


def main():
    """Main stdio loop: read newline-delimited JSON-RPC from stdin, write to stdout."""
    _log(f"KernelWiki MCP server starting (WIKI_ROOT={WIKI_ROOT})")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            resp = _jsonrpc_error(None, -32700, f"Parse error: {e}")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        if not isinstance(msg, dict):
            resp = _jsonrpc_error(None, -32600, "Invalid Request: not an object")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_request(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

    _log("KernelWiki MCP server shutting down")


if __name__ == "__main__":
    main()
