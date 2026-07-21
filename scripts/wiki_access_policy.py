"""Shared read-access policy for KernelWiki file lookups.

Two independent code paths turn a caller-supplied string into a filesystem
path: ``wiki_page_service.find_page`` (page id / relative path) and
``wiki_grep_service.iter_files`` (scope walk).  Containment inside WIKI_ROOT is
not sufficient on its own, because the repo root also holds ``scripts/`` (the
MCP server's own auth code), ``corpus/`` (large third-party mirrors, partly
gitignored), ``data/`` and ``.git/`` (remote URLs of private repositories).

This module is the single place that decides which files a lookup may reach.
Both callers apply it by default, so the CLI scripts and the MCP server share
one policy rather than each enforcing its own.
"""

from __future__ import annotations

from pathlib import Path

from _wiki_root import WIKI_ROOT

WIKI_ROOT_RESOLVED = WIKI_ROOT.resolve()

# Top-level directories a caller-supplied lookup may resolve into.
PAGE_ROOTS = ("wiki", "sources", "artifacts")

# Additionally reachable when following `sources:` / `source_refs:` declared in
# a page's own frontmatter (repo-controlled, not caller-controlled).
EXCERPT_ROOTS = PAGE_ROOTS + ("corpus",)

# Text-ish extensions the knowledge base is made of. Anything else (.sqlite3,
# .pyc, .so, images, archives) is never served.
READABLE_EXTS = frozenset({
    ".md",
    ".cu", ".cuh", ".ptx",
    ".cpp", ".cxx", ".cc", ".c",
    ".h", ".hpp", ".hxx", ".inl",
    ".py", ".pyx",
    ".patch", ".diff",
    ".sh",
    ".txt", ".rst",
    ".yaml", ".yml", ".json", ".toml", ".csv", ".tsv",
})


def is_within_root(path: Path) -> bool:
    """True if *path* resolves inside WIKI_ROOT (defends against symlinks)."""
    try:
        return path.resolve().is_relative_to(WIKI_ROOT_RESOLVED)
    except (OSError, ValueError):
        return False


def _top_level(path: Path) -> str | None:
    """Return the first path component of *path* relative to WIKI_ROOT."""
    try:
        rel = path.resolve().relative_to(WIKI_ROOT_RESOLVED)
    except (OSError, ValueError):
        return None
    return rel.parts[0] if rel.parts else None


def is_readable(path: Path, *, roots: tuple[str, ...] = PAGE_ROOTS,
                require_ext: bool = True) -> bool:
    """True if *path* may be served to a caller.

    Enforces, in order: symlink-safe containment in WIKI_ROOT, membership in an
    allowed top-level directory, and an allowlisted extension.  Directories are
    never readable.
    """
    if not is_within_root(path):
        return False
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    if _top_level(path) not in roots:
        return False
    if require_ext and path.suffix.lower() not in READABLE_EXTS:
        return False
    return True


def is_allowed_dir(path: Path, *, roots: tuple[str, ...] = PAGE_ROOTS) -> bool:
    """True if *path* is a directory whose contents may be served.

    Artifact bundle locations come from a page's `artifact_dir` frontmatter
    field, which is a repo-controlled string but still resolves anywhere under
    the root — `artifact_dir: scripts` would otherwise hand out the server's
    own source.
    """
    if not is_within_root(path):
        return False
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    return _top_level(path) in roots


def resolve_lookup(rel_lookup: str, *, roots: tuple[str, ...] = PAGE_ROOTS,
                   require_ext: bool = True) -> Path | None:
    """Resolve a relative lookup string under WIKI_ROOT, or None if disallowed.

    Note ``WIKI_ROOT / "/etc/passwd"`` yields ``/etc/passwd`` — pathlib drops
    the left operand for an absolute right operand — so absolute lookups are
    rejected up front rather than relying on the containment check alone.
    """
    if not rel_lookup or Path(rel_lookup).is_absolute():
        return None
    candidate = (WIKI_ROOT / rel_lookup).resolve()
    if not is_readable(candidate, roots=roots, require_ext=require_ext):
        return None
    return candidate


def filter_exts(exts) -> set[str] | None:
    """Intersect caller-supplied extensions with the allowlist."""
    if not exts:
        return None
    allowed = {e for e in exts if e.lower() in READABLE_EXTS}
    return allowed or None
