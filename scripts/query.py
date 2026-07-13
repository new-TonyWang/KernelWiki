#!/usr/bin/env python3
"""Unified query tool for the Blackwell kernel wiki.

Supports natural-language keyword queries, tag filters, repo filters, and type filters.

Usage:
    query.py "how to fuse gate-up GEMM"
    query.py --tag nvfp4 --type kernel
    query.py --repo cutlass --limit 20
    query.py --language cute-dsl
    query.py --symptom memory-bound

Returns a ranked list of matching pages with titles, paths, and key frontmatter fields.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiki_query_service import (  # noqa: E402
    load_all_pages, filter_pages, score_keyword_match, format_result,
    infer_vendor,
)


def main():
    parser = argparse.ArgumentParser(description="Query the GPU/NPU kernel wiki")
    parser.add_argument("query", nargs="*", help="Free-text keywords")
    parser.add_argument("--type", help="Filter by page type (kernel, technique, hardware, pattern, language, migration, pr, blog, doc, contest)")
    parser.add_argument("--tag", help="Filter by tag (must appear in tags/techniques/hardware_features/kernel_types/languages)")
    parser.add_argument("--repo", help="Filter by source repo (partial match, e.g. 'cutlass')")
    parser.add_argument("--language", help="Filter by language/DSL (cute-dsl, cuda-cpp, ptx, triton, ascendc, triton-ascend, etc.)")
    parser.add_argument("--architecture", help="Filter by architecture (sm100, sm90, ascend910b, etc.)")
    parser.add_argument("--symptom", help="Filter by pattern symptom (memory-bound, register-pressure, etc.)")
    parser.add_argument("--confidence", help="Filter by confidence (verified, source-reported, inferred, experimental)")
    parser.add_argument("--vendor", help="Filter by vendor (nvidia, ascend, biren, all). Auto-inferred from --architecture/--language/keywords when omitted.")
    parser.add_argument("--has-code", action="store_true", help="Only return pages whose artifact_dir contains at least one source file")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--compact", action="store_true", help="Compact one-line-per-result output")
    parser.add_argument("--paths-only", action="store_true", help="Output only file paths, one per line")
    args = parser.parse_args()

    # Build a plain dict for service functions (they accept dicts, not Namespace)
    params = {
        "type": args.type,
        "tag": args.tag,
        "vendor": args.vendor,
        "repo": args.repo,
        "language": args.language,
        "architecture": args.architecture,
        "symptom": args.symptom,
        "confidence": args.confidence,
        "has_code": args.has_code,
        "query": args.query,
    }

    # Auto-infer vendor when not explicitly specified
    if not args.vendor:
        inferred = infer_vendor(params)
        if inferred:
            params["vendor"] = inferred
            if not args.paths_only:
                print(f"# Auto-detected vendor: {inferred}")
                print()

    pages = load_all_pages()
    pages = filter_pages(pages, params)

    # Score by keywords if any. Flatten multi-word quoted args into tokens so
    # `query.py "how to fuse dual GEMM"` behaves like `query.py how to fuse dual GEMM`.
    keywords = []
    for q in args.query:
        for tok in re.split(r"\s+", q.strip()):
            if tok:
                keywords.append(tok)
    if keywords:
        for p in pages:
            p["_score"] = score_keyword_match(p["fm"], p["body"], keywords)
        pages = [p for p in pages if p["_score"] > 0]
        pages.sort(key=lambda x: (-x["_score"], x["path"]))
    else:
        pages.sort(key=lambda x: x["path"])

    pages = pages[:args.limit]

    if args.paths_only:
        for p in pages:
            print(p["path"])
        return

    if not pages:
        print("No matching pages.")
        return

    print(f"# {len(pages)} result(s)")
    print()
    for p in pages:
        print(format_result(p, compact=args.compact))
        if not args.compact:
            print()


if __name__ == "__main__":
    main()
