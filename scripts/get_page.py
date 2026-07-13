#!/usr/bin/env python3
"""Retrieve a single wiki page by its id or path.

Usage:
    get_page.py kernel-flash-attention-4         # by id
    get_page.py pr-cutlass-2472                   # by id
    get_page.py wiki/nvidia/kernels/flash-attention-4.md # by path
    get_page.py kernel-flash-attention-4 --body-only
    get_page.py kernel-flash-attention-4 --frontmatter-only
    get_page.py kernel-flash-attention-4 --follow-sources  # also print first 500 chars of each source
"""

import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402
from wiki_page_service import (  # noqa: E402
    find_page, split_frontmatter, resolve_artifact_dir, ARTIFACT_EXTS,
)
try:
    from source_corpus.registry import resolve_corpus_path  # noqa: E402
except ImportError:
    resolve_corpus_path = None


def main():
    parser = argparse.ArgumentParser(description="Get a wiki page by id or path")
    parser.add_argument("lookup", help="Page id (e.g. kernel-flash-attention-4) or relative path")
    parser.add_argument("--body-only", action="store_true", help="Print only the body (skip frontmatter)")
    parser.add_argument("--frontmatter-only", action="store_true", help="Print only the frontmatter as YAML")
    parser.add_argument("--follow-sources", action="store_true", help="Also print 500-char excerpt from each cited source")
    parser.add_argument("--include-code", action="store_true", help="After the page body, print all files under the page's artifact_dir (Phase 3)")
    args = parser.parse_args()

    page_path = find_page(args.lookup)
    if not page_path:
        print(f"ERROR: No page found for '{args.lookup}'", file=sys.stderr)
        sys.exit(1)

    content = page_path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(content)

    if args.frontmatter_only:
        if fm:
            print(yaml.dump(fm, allow_unicode=True, sort_keys=False))
        return

    if args.body_only:
        print(body)
        return

    # Default: full page, prefixed with a path header for context
    print(f"# {page_path.relative_to(WIKI_ROOT)}")
    print()
    print(content)

    if args.follow_sources and fm:
        print()
        print("---")
        print("## Cited Sources (excerpts)")
        print()

        source_entries = []
        for src_id in fm.get("sources", []) or []:
            source_entries.append(("source-id", src_id, src_id))
        for src in fm.get("source", []) or []:
            if isinstance(src, dict) and src.get("path"):
                source_entries.append(("source-path", src.get("path"), src.get("anchor", "")))
            elif isinstance(src, str):
                source_entries.append(("source-path", src, ""))
        for src in fm.get("source_refs", []) or []:
            if isinstance(src, dict) and src.get("source_id"):
                label = src.get("source_id")
                corpus_path = label
                if src.get("path"):
                    corpus_path = label + "/" + str(src["path"])
                detail = " / ".join(str(x) for x in (src.get("path"), src.get("anchor")) if x)
                source_entries.append(("source-ref", corpus_path, detail))

        seen = set()
        for kind, lookup, detail in source_entries:
            key = (kind, lookup, detail)
            if key in seen:
                continue
            seen.add(key)
            src_page = None
            if kind == "source-path":
                p = WIKI_ROOT / str(lookup)
                if p.is_file():
                    src_page = p
            if src_page is None:
                src_page = find_page(str(lookup))
            if src_page is None and kind == "source-ref" and resolve_corpus_path:
                try:
                    cp = resolve_corpus_path(str(lookup))
                    if cp.is_file():
                        src_page = cp
                except (ValueError, Exception):
                    pass
            if src_page:
                src_content = src_page.read_text(encoding="utf-8")
                _, src_body = split_frontmatter(src_content)
                excerpt = (src_body or "")[:500].strip()
                print(f"### {lookup}")
                try:
                    print(f"`{src_page.relative_to(WIKI_ROOT)}`")
                except ValueError:
                    print(f"`{src_page}`")
                if detail and detail != lookup:
                    print(f"anchor/detail: {detail}")
                print()
                print(excerpt)
                print()
            else:
                print(f"### {lookup}")
                if detail and detail != lookup:
                    print(f"anchor/detail: {detail}")
                print("_No local markdown page resolved; use the path/source_ref metadata above._")
                print()

    if args.include_code:
        ad, ad_path, is_fallback = resolve_artifact_dir(page_path, fm)
        if ad_path and ad_path.is_dir():
            print()
            print("---")
            suffix = " (conventional path — artifact_dir not backfilled)" if is_fallback else ""
            print(f"## Artifact Bundle: `{ad}`{suffix}")
            print()
            for f in sorted(ad_path.rglob("*")):
                if not f.is_file() or f.suffix.lower() not in ARTIFACT_EXTS:
                    continue
                rel_to_bundle = f.relative_to(ad_path)
                print(f"### `{rel_to_bundle}`")
                print()
                try:
                    body_bytes = f.read_bytes()
                    if f.stat().st_size > 200 * 1024:
                        print(f"*(file is {f.stat().st_size} bytes; showing first 200 KiB)*")
                        body_bytes = body_bytes[:200 * 1024]
                    print("```" + (f.suffix.lstrip(".") or ""))
                    print(body_bytes.decode("utf-8", errors="replace"))
                    print("```")
                    print()
                except Exception as e:
                    print(f"*(could not read: {e})*")
                    print()
        elif ad is None:
            # No explicit artifact_dir and no conventional location
            # resolves to an existing directory — the page simply has no
            # bundle. Stay silent rather than printing a misleading
            # "not found on disk" message.
            pass
        else:
            print()
            print(f"*(artifact_dir '{ad}' not found on disk)*")


if __name__ == "__main__":
    main()
