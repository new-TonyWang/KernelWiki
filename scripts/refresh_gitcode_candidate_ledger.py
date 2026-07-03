#!/usr/bin/env python3
"""Refresh a GitCode candidate ledger.

This is a small provider-specific companion to refresh_candidate_ledger.py for
repositories that live on gitcode.com instead of GitHub. It reads one ledger
(e.g. candidates/triton-ascend.yaml), scans closed pull requests through the
GitCode v5 API, filters by keywords and optional path_prefixes, and appends new
PR rows as `decision: defer` for later human triage.

Usage:
  python3 scripts/refresh_gitcode_candidate_ledger.py candidates/triton-ascend.yaml --cutoff 2026-06-29 --dry-run
  python3 scripts/refresh_gitcode_candidate_ledger.py candidates/triton-ascend.yaml --cutoff 2026-06-29 --max-pages 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent


def api_get_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def gitcode_api_url(repo: str, suffix: str, **query):
    owner, name = repo.split("/", 1)
    base = f"https://api.gitcode.com/api/v5/repos/{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}/{suffix.lstrip('/')}"
    if query:
        return base + "?" + urllib.parse.urlencode(query)
    return base


def normalize_date(s: str) -> str:
    return (s or "")[:10]


def fetch_pull_files(repo: str, number: int):
    url = gitcode_api_url(repo, f"pulls/{number}/files")
    try:
        rows = api_get_json(url)
    except Exception as e:  # network / API shape errors are advisory during refresh
        print(f"WARN: failed to fetch GitCode files for {repo}#{number}: {e}", file=sys.stderr)
        return []
    if not isinstance(rows, list):
        return []
    return [r.get("filename", "") for r in rows if isinstance(r, dict) and r.get("filename")]


def row_matches(row, keywords, path_prefixes, files):
    text = " ".join(str(row.get(k, "")) for k in ["title", "body"]).lower()
    file_text = " ".join(files).lower()
    kw_hit = any(str(k).lower() in text or str(k).lower() in file_text for k in keywords)
    path_hit = any(any(f.startswith(prefix) for prefix in path_prefixes) for f in files)
    # If the ledger was created from a GitCode tree URL, path_prefixes is the
    # user's intended scope. Treat it as a hard filter; keyword hits outside the
    # subtree are usually repo-wide compiler/docs noise.
    if path_prefixes:
        return path_hit
    return kw_hit


def load_existing_numbers(data):
    return {r.get("number") for r in data.get("prs", []) or [] if isinstance(r, dict)}


def recompute_counts(data):
    prs = data.get("prs") or []
    data["total_candidates"] = len(prs)
    data["included"] = sum(1 for r in prs if str(r.get("decision", "")).lower() == "include")
    data["excluded"] = sum(1 for r in prs if str(r.get("decision", "")).lower() == "exclude")
    data["deferred"] = sum(1 for r in prs if str(r.get("decision", "")).lower() == "defer")


def search_gitcode_pulls(repo, keywords, path_prefixes, cutoff_date, max_pages=5, per_page=50, sleep=0.5, workers=2, start_page=1):
    cutoff = cutoff_date.isoformat()
    hits = []
    seen = set()
    for page in range(start_page, start_page + max_pages):
        url = gitcode_api_url(repo, "pulls", state="closed", per_page=per_page, page=page)
        try:
            rows = api_get_json(url)
        except Exception as e:
            print(f"WARN: GitCode pull list failed for {repo} page={page}: {e}", file=sys.stderr)
            break
        if not isinstance(rows, list) or not rows:
            break
        candidates = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            num = row.get("number")
            if not isinstance(num, int) or num in seen:
                continue
            closed = normalize_date(row.get("closed_at") or row.get("updated_at") or row.get("created_at"))
            if closed and closed > cutoff:
                continue
            candidates.append((row, num, closed))

        file_map = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(fetch_pull_files, repo, num): num for _row, num, _closed in candidates}
            for fut in as_completed(futs):
                num = futs[fut]
                try:
                    file_map[num] = fut.result()
                except Exception as e:
                    print(f"WARN: failed to fetch GitCode files for {repo}#{num}: {e}", file=sys.stderr)
                    file_map[num] = []

        for row, num, closed in candidates:
            files = file_map.get(num, [])
            if not row_matches(row, keywords, path_prefixes, files):
                continue
            seen.add(num)
            hits.append({
                "number": num,
                "title": row.get("title", ""),
                "closedAt": closed,
                "url": row.get("html_url", ""),
                "files": files,
            })
        time.sleep(sleep)
    hits.sort(key=lambda r: r["number"])
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("ledger", help="Path to candidates/<repo>.yaml with provider: gitcode")
    parser.add_argument("--cutoff", required=True, help="Refresh cutoff date (YYYY-MM-DD)")
    parser.add_argument("--start-page", type=int, default=1, help="First GitCode API page to scan (default: 1)")
    parser.add_argument("--max-pages", type=int, default=5, help="Number of GitCode API pages to scan (default: 5)")
    parser.add_argument("--per-page", type=int, default=50, help="GitCode API page size (default: 50)")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent GitCode file-list requests per page (default: 2; keep low to avoid 429)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between GitCode list pages (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing the ledger")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    if not ledger_path.is_absolute():
        ledger_path = REPO_ROOT / ledger_path
    data = yaml.safe_load(ledger_path.read_text(encoding="utf-8")) or {}
    if data.get("provider") != "gitcode":
        raise SystemExit(f"{ledger_path}: expected provider: gitcode")
    repo = data.get("repo")
    if not repo or "/" not in repo:
        raise SystemExit(f"{ledger_path}: missing repo: owner/name")
    cutoff_date = date.fromisoformat(args.cutoff)
    keywords = data.get("keywords_used") or []
    path_prefixes = data.get("path_prefixes") or []

    print(f"  {ledger_path.name}: searching GitCode {repo} with {len(keywords)} keywords, {len(path_prefixes)} path prefixes...")
    hits = search_gitcode_pulls(
        repo, keywords, path_prefixes, cutoff_date,
        max_pages=args.max_pages,
        per_page=args.per_page,
        sleep=args.sleep,
        workers=args.workers,
        start_page=args.start_page,
    )
    print(f"    -> {len(hits)} closed PRs found within cutoff window")
    existing = load_existing_numbers(data)
    new_rows = []
    for h in hits:
        marker = "existing" if h["number"] in existing else "new"
        print(f"    [{marker}] #{h['number']} {h['closedAt']} {h['title']}")
        if h["number"] in existing:
            continue
        new_rows.append({
            "number": h["number"],
            "title": h.get("title", ""),
            "date": h.get("closedAt", ""),
            "decision": "defer",
            "reason": "from gitcode refresh; needs-triage",
            "files_changed": h.get("files", [])[:50],
        })

    if args.dry_run:
        print("\nDry-run mode; no files written.")
        return

    data["searched_at"] = cutoff_date.isoformat()
    if new_rows:
        data["prs"] = (data.get("prs") or []) + new_rows
    recompute_counts(data)
    ledger_path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=200, allow_unicode=True), encoding="utf-8")
    print(f"    -> +{len(new_rows)} new defer-rows merged into {ledger_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
