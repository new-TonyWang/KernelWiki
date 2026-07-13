#!/usr/bin/env python3
"""Text search across wiki bodies and source PR descriptions.

Usage:
    grep_wiki.py "tcgen05.fence"
    grep_wiki.py "2-CTA backward" --only wiki
    grep_wiki.py "ping-pong" --context 3
    grep_wiki.py "nvfp4 block_scale" --any     # match if ANY word appears

Returns matching lines with file path, line number, and N context lines.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402
from wiki_grep_service import search_wiki  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Text search across Blackwell kernel wiki")
    parser.add_argument("patterns", nargs="+", help="Search pattern(s) — all must match a line unless --any is used")
    parser.add_argument("--only", choices=["wiki", "sources", "all", "artifacts"], default="all",
                        help="Restrict search scope (default: all)")
    parser.add_argument("--context", type=int, default=1, help="Context lines around each match (default 1)")
    parser.add_argument("--any", action="store_true", help="Match if ANY pattern matches a line (default: all must match)")
    parser.add_argument("--limit", type=int, default=20, help="Max files reported (default 20)")
    parser.add_argument("--files-only", action="store_true", help="Print only matching file paths")
    parser.add_argument("--ext", default=None, help="Comma-separated extra extensions to search (without dots), e.g. 'cu,cuh,ptx,py'; auto-expands scope to include artifacts/")
    args = parser.parse_args()

    # Validate regex patterns early (search_wiki raises re.error)
    for p in args.patterns:
        try:
            re.compile(p)
        except re.error as e:
            print(f"ERROR: invalid regex {p!r}: {e}", file=sys.stderr)
            print("       Hint: escape special chars ([](){}.*+?^$|\\) or quote the pattern.", file=sys.stderr)
            sys.exit(2)

    ext_set = None
    if args.ext:
        ext_set = {"." + e.strip().lstrip(".").lower() for e in args.ext.split(",") if e.strip()}

    results, _total = search_wiki(
        args.patterns, scope=args.only, context=args.context,
        any_match=args.any, exts=ext_set, limit=args.limit,
        per_file_limit=5,
    )

    if args.files_only:
        for entry in results:
            print(entry["path_rel"])
        return

    if not results:
        print("No matches.")
        return

    print(f"# {len(results)} file(s) match")
    for entry in results:
        total = entry["total_hits"]
        print()
        print(f"## {entry['path_rel']}  ({total} match{'es' if total != 1 else ''})")
        for hit in entry["hits"]:
            print(f"```")
            print(hit["snippet"])
            print(f"```")


if __name__ == "__main__":
    main()
