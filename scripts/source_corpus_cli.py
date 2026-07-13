#!/usr/bin/env python3
"""Source corpus CLI — search and read from the two-tier corpus.

Usage:
    python3 scripts/source_corpus_cli.py search "keyword" [--scope SCOPE]
    python3 scripts/source_corpus_cli.py list [--scope SCOPE]
    python3 scripts/source_corpus_cli.py read SOURCE_ID PATH

Scopes: cuda-official, blogs/colfax, whitepapers/gpu, source-code/cutlass, etc.
"""

import argparse
import re
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
MANIFEST_PATH = CORPUS_DIR / "MANIFEST.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parent))

PLACEHOLDER_RE = re.compile(r'\{\{(\w+)\}\}')


def load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def load_localize_config():
    """Load localize.yaml for resolving tier-2 paths."""
    config_path = CORPUS_DIR / "localize.yaml"
    if not config_path.exists():
        return {}
    try:
        from localize import load_config
        return load_config() or {}
    except ImportError:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        variables = dict(config.get("vars", {}) or {})
        derived = dict(config.get("derived", {}) or {})
        resolved = dict(variables)
        for key, val in derived.items():
            if isinstance(val, str):
                for m in PLACEHOLDER_RE.finditer(val):
                    ref = m.group(1)
                    if ref in resolved:
                        val = val.replace(f"{{{{{ref}}}}}", str(resolved[ref]))
            resolved[key] = val
        return resolved


def resolve_corpus_path(local_path, variables):
    """Resolve a corpus path from MANIFEST entry."""
    if PLACEHOLDER_RE.search(local_path):
        result = local_path
        for m in PLACEHOLDER_RE.finditer(local_path):
            var = m.group(1)
            if var in variables:
                result = result.replace(f"{{{{{var}}}}}", str(variables[var]))
        if PLACEHOLDER_RE.search(result):
            return None  # Unresolved
        return Path(result)
    return CORPUS_DIR / local_path


def cmd_search(args):
    """Search corpus files for a keyword."""
    manifest = load_manifest()
    variables = load_localize_config()
    keyword = args.keyword.lower()
    results = []

    for entry in manifest:
        sid = entry.get("source_id", "")
        if args.scope and not sid.startswith(args.scope):
            continue

        local_path = entry.get("local_path", "")
        resolved = resolve_corpus_path(local_path, variables)
        if resolved is None:
            if not args.scope:
                continue
            print(f"WARN: {sid} has unresolved placeholders (configure corpus/localize.yaml)", file=sys.stderr)
            continue

        if not Path(resolved).exists():
            continue

        # Search through files
        root = Path(resolved)
        if root.is_file():
            files = [root]
        else:
            files = sorted(root.rglob("*"))

        for f in files:
            if not f.is_file():
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if keyword in content.lower():
                rel = f.relative_to(resolved) if f != resolved else f.name
                results.append((sid, str(rel), f))

    if not results:
        if args.scope:
            print(f"No results for '{args.keyword}' in scope '{args.scope}'")
            # Check if scope needs localize.yaml
            for entry in manifest:
                if entry.get("source_id", "").startswith(args.scope):
                    lp = entry.get("local_path", "")
                    if PLACEHOLDER_RE.search(lp):
                        print(f"Note: scope '{args.scope}' requires corpus/localize.yaml to be configured")
        else:
            print(f"No results for '{args.keyword}'")
        return 1

    print(f"# {len(results)} result(s) for '{args.keyword}'")
    for sid, rel, fpath in results[:args.limit]:
        print(f"  {sid}: {rel}")
    return 0


def cmd_list(args):
    """List corpus entries."""
    manifest = load_manifest()
    variables = load_localize_config()

    for entry in manifest:
        sid = entry.get("source_id", "")
        if args.scope and not sid.startswith(args.scope):
            continue
        local_path = entry.get("local_path", "")
        resolved = resolve_corpus_path(local_path, variables)
        status = "OK" if resolved and Path(resolved).exists() else "MISSING"
        print(f"  [{status}] {sid}: {local_path}")
    return 0


def cmd_read(args):
    """Read a specific file from the corpus."""
    manifest = load_manifest()
    variables = load_localize_config()

    for entry in manifest:
        sid = entry.get("source_id", "")
        if sid != args.source_id:
            continue
        local_path = entry.get("local_path", "")
        resolved = resolve_corpus_path(local_path, variables)
        if resolved is None or not Path(resolved).exists():
            print(f"Error: corpus path not available for {sid}")
            return 1
        root = Path(resolved).resolve()
        target = (root / args.path).resolve()
        if not target.is_relative_to(root):
            print(f"Error: path rejected — outside corpus root: {args.path}")
            return 1
        if not target.exists():
            print(f"Error: {args.path} not found in {sid}")
            return 1
        print(target.read_text(encoding="utf-8"))
        return 0

    print(f"Error: source_id '{args.source_id}' not in MANIFEST.yaml")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Source corpus CLI")
    sub = parser.add_subparsers(dest="command")

    p_search = sub.add_parser("search", help="Search corpus for keyword")
    p_search.add_argument("keyword", help="Search keyword")
    p_search.add_argument("--scope", help="Limit to source_id prefix")
    p_search.add_argument("--limit", type=int, default=20, help="Max results")

    p_list = sub.add_parser("list", help="List corpus entries")
    p_list.add_argument("--scope", help="Filter by source_id prefix")

    p_read = sub.add_parser("read", help="Read a file from corpus")
    p_read.add_argument("source_id", help="Source ID from MANIFEST")
    p_read.add_argument("path", help="Path within source")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return {"search": cmd_search, "list": cmd_list, "read": cmd_read}[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
