"""Service functions for KernelWiki text search (grep).

Extracted from grep_wiki.py for programmatic reuse by MCP server and other tools.
No argparse, no stdout/stderr writes, no sys.exit().
"""

import re
from pathlib import Path

from _wiki_root import WIKI_ROOT

_WIKI_ROOT_RESOLVED = WIKI_ROOT.resolve()


def _is_within_root(path):
    """Return True if resolved path is inside WIKI_ROOT."""
    return path.resolve().is_relative_to(_WIKI_ROOT_RESOLVED)


# ---------------------------------------------------------------------------
# Default artifact-scope extensions (R32/R33 contract)
# ---------------------------------------------------------------------------

ARTIFACT_DEFAULT_EXTS = {
    ".md",
    ".cu", ".cuh", ".ptx",
    ".cpp", ".cxx", ".cc", ".c",
    ".h", ".hpp", ".hxx", ".inl",
    ".py", ".pyx",
    ".patch",
    ".txt",
    ".sh",
    ".yaml", ".yml", ".json",
}


# ---------------------------------------------------------------------------
# File iteration
# ---------------------------------------------------------------------------

def iter_files(scope, exts=None):
    """Yield file paths under the given scope.

    scope: 'wiki', 'sources', 'all', or 'artifacts'
    exts:  optional set of lowercase extensions with dots, e.g. {'.cu', '.cuh'}
    """
    dirs = {
        "wiki": ["wiki"],
        "sources": ["sources"],
        "all": ["wiki", "sources"],
        "artifacts": ["artifacts"],
    }
    sub_list = dirs.get(scope, ["wiki", "sources"])
    if exts and "artifacts" not in sub_list and scope != "wiki" and scope != "sources":
        sub_list = sub_list + ["artifacts"]

    if scope == "artifacts" and not exts:
        search_exts = ARTIFACT_DEFAULT_EXTS
    else:
        search_exts = {".md"} | (exts or set())

    for sub in sub_list:
        base = WIKI_ROOT / sub
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            if not _is_within_root(f):
                continue
            if f.suffix.lower() in search_exts:
                yield f


# ---------------------------------------------------------------------------
# Single-file grep
# ---------------------------------------------------------------------------

def grep_file(path, compiled_patterns, context, any_match):
    """Search a single file for pattern(s).

    Returns list of (line_no_1based, context_snippet_str) tuples.
    """
    if not _is_within_root(path):
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    results = []
    for i, line in enumerate(lines):
        if any_match:
            matched = any(p.search(line) for p in compiled_patterns)
        else:
            matched = all(p.search(line) for p in compiled_patterns)
        if matched:
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            snippet = "\n".join(
                f"{j+1}{'→' if j == i else ':'} {lines[j]}"
                for j in range(start, end)
            )
            results.append((i + 1, snippet))
    return results


# ---------------------------------------------------------------------------
# High-level search
# ---------------------------------------------------------------------------

def search_wiki(patterns, scope="all", context=1, any_match=False,
                exts=None, limit=20, per_file_limit=5):
    """Run grep across wiki files.

    patterns:  list of regex pattern strings
    scope:     'wiki', 'sources', 'all', 'artifacts'
    context:   context lines around each match
    any_match: True = ANY pattern matches; False = ALL must match
    exts:      optional set of extra extensions
    limit:     max files reported
    per_file_limit: max hits per file

    Returns (matched_files, total_matching_files) where matched_files is
    a list of dicts (up to *limit*) and total_matching_files is the count
    of ALL files that matched (before truncation).
    Raises re.error if patterns are invalid.
    """
    compiled = []
    for p in patterns:
        compiled.append(re.compile(p, re.IGNORECASE))

    matched_files = []
    total_matching_files = 0
    for path in iter_files(scope, exts=exts):
        hits = grep_file(path, compiled, context, any_match)
        if hits:
            total_matching_files += 1
            if len(matched_files) < limit:
                hit_dicts = [
                    {"line_no": ln, "snippet": sn}
                    for ln, sn in hits[:per_file_limit]
                ]
                matched_files.append({
                    "path_rel": str(path.relative_to(WIKI_ROOT)),
                    "hits": hit_dicts,
                    "total_hits": len(hits),
                })

    return matched_files, total_matching_files
