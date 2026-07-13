"""Service functions for KernelWiki page retrieval.

Extracted from get_page.py for programmatic reuse by MCP server and other tools.
No argparse, no stdout/stderr writes, no sys.exit().
"""

import re
import yaml
from pathlib import Path

from _wiki_root import WIKI_ROOT


# ---------------------------------------------------------------------------
# Page lookup
# ---------------------------------------------------------------------------

def find_page(lookup):
    """Find a page by id, alias, or relative path. Returns Path or None.

    Matching order:
      1. Path-style lookup (contains '/' or ends with '.md')
      2. Exact id match
      3. Alias match (frontmatter 'aliases' list, case-insensitive)
    """
    if "/" in lookup or lookup.endswith(".md"):
        p = (WIKI_ROOT / lookup).resolve()
        if p.is_relative_to(WIKI_ROOT.resolve()) and p.exists():
            return p

    lookup_lower = lookup.lower()
    alias_match = None
    prefix_match = None

    for subdir in ["wiki", "sources"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            try:
                content = md.read_text(encoding="utf-8")
            except Exception:
                continue
            m = re.match(r'^---\s*\r?\n(.*?)\r?\n---', content, re.DOTALL)
            if not m:
                continue
            try:
                fm = yaml.safe_load(m.group(1))
            except yaml.YAMLError:
                continue
            if not isinstance(fm, dict):
                continue
            page_id = fm.get("id", "")
            # Exact id match — return immediately
            if page_id == lookup:
                return md
            # Alias match (case-insensitive) — first match wins
            if alias_match is None:
                for a in (fm.get("aliases") or []):
                    if str(a).lower() == lookup_lower:
                        alias_match = md
                        break
            # ID prefix match — lookup is a prefix of the page id
            if prefix_match is None and page_id.startswith(lookup + "-"):
                prefix_match = md

    return alias_match or prefix_match


# ---------------------------------------------------------------------------
# Frontmatter splitting
# ---------------------------------------------------------------------------

def split_frontmatter(content):
    """Split markdown content into (frontmatter_dict, body_str).
    Returns (None, content) if no frontmatter found."""
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', content, re.DOTALL)
    if not m:
        return None, content
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        fm = None
    return fm, m.group(2)


# ---------------------------------------------------------------------------
# Artifact directory resolution
# ---------------------------------------------------------------------------

# File extensions for artifact bundles (shared contract with validate.py and query.py).
ARTIFACT_EXTS = {
    ".cu", ".cuh", ".ptx",
    ".cpp", ".cxx", ".cc", ".c",
    ".h", ".hpp", ".hxx", ".inl",
    ".py", ".pyx",
    ".patch",
    ".sh",
    ".md", ".yaml", ".yml", ".txt", ".json",
}


def resolve_artifact_dir(page_path, fm):
    """Return (artifact_dir_rel_str, artifact_dir_abs_path, is_fallback)
    or (None, None, False) if no bundle location can be determined."""
    fm = fm or {}
    if "artifact_dir" in fm and fm.get("artifact_dir"):
        rel = str(fm["artifact_dir"])
        return rel, WIKI_ROOT / rel, False

    try:
        rel_page = page_path.relative_to(WIKI_ROOT)
    except ValueError:
        return None, None, False
    parts = rel_page.parts
    if len(parts) < 2 or parts[0] != "sources":
        return None, None, False

    candidate = None
    if parts[1] == "blogs" and len(parts) == 3:
        candidate = Path("artifacts") / "blogs" / page_path.stem
    elif parts[1] == "contests" and len(parts) == 4:
        candidate = Path("artifacts") / "contests" / parts[2] / page_path.stem
    elif parts[1] == "prs" and len(parts) == 4:
        candidate = Path("artifacts") / "prs" / parts[2] / page_path.stem

    if candidate is None:
        return None, None, False
    abs_path = WIKI_ROOT / candidate
    if not abs_path.is_dir():
        return None, None, False

    if parts[1] == "blogs":
        manifest_path = abs_path / "MANIFEST.yaml"
        if manifest_path.is_file():
            try:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
                if manifest.get("code_present") is False:
                    return None, None, False
            except Exception:
                pass
    return str(candidate), abs_path, True


# ---------------------------------------------------------------------------
# High-level page loading
# ---------------------------------------------------------------------------

def load_page(lookup):
    """Load a page by id or path.

    Returns dict with keys: path, content, fm, body, page_path
    or None if not found.
    """
    page_path = find_page(lookup)
    if not page_path:
        return None

    content = page_path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(content)

    return {
        "page_path": page_path,
        "path": str(page_path.relative_to(WIKI_ROOT)),
        "content": content,
        "fm": fm,
        "body": body,
    }


def load_artifact_files(ad_path, max_files=100, max_file_size=512000,
                        containment_root=None):
    """Load artifact files from a bundle directory.

    containment_root: if set, each file's resolved path must be within this root
                      (defends against symlink escapes).

    Returns list of dicts: {rel_path, content, size, truncated}
    """
    if not ad_path or not ad_path.is_dir():
        return []

    resolved_root = containment_root.resolve() if containment_root else None

    files = []
    for f in sorted(ad_path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in ARTIFACT_EXTS:
            continue
        # Per-file symlink containment check
        if resolved_root and not f.resolve().is_relative_to(resolved_root):
            continue
        if len(files) >= max_files:
            break
        rel = str(f.relative_to(ad_path))
        size = f.stat().st_size
        truncated = size > max_file_size
        try:
            raw = f.read_bytes()
            if truncated:
                raw = raw[:max_file_size]
            content = raw.decode("utf-8", errors="replace")
        except Exception:
            content = None
        files.append({
            "rel_path": rel,
            "content": content,
            "size": size,
            "truncated": truncated,
        })
    return files
