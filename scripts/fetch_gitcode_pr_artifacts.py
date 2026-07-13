#!/usr/bin/env python3
"""Fetch GitCode PR patch/key-file artifact bundles for generated source PR pages.

This is the GitCode companion for scripts/fetch_pr_diff.py. It reads
sources/prs/<repo_slug>/PR-<N>.md pages whose frontmatter repo is a GitCode
repo such as Ascend/triton-ascend, fetches /pulls/<N>/files from the GitCode v5
API, writes artifacts/prs/<repo_slug>/PR-<N>/{diff.patch,key-files/...}, emits
PROVENANCE.yaml, and backfills artifact_dir in the source page frontmatter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = REPO_ROOT / "sources" / "prs"
ARTIFACTS = REPO_ROOT / "artifacts" / "prs"
FILE_SIZE_CAP = 1 * 1024 * 1024
BUNDLE_SIZE_CAP = 5 * 1024 * 1024
RAW_SLEEP = 0.0
RAW_RETRIES = 1
RAW_BACKOFF = 2.0

SKIP_PREFIXES = (
    ".github/", ".gitcode/", "docs/", "docker/", "build/docker/",
    "test/", "tests/", "ascend/examples/generalization_cases/",
    "ascend/examples/pytest_ut/",
)
SKIP_NAMES = {"README.md", "README_zh.md", "SECURITYNOTE.md", "INSTALL.md", "INSTALL_EN.md", "OWNERS", "Makefile", "CMakeLists.txt"}
SOURCE_EXTS = (".py", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".td", ".mlir", ".ll", ".bc")
SOURCE_PREFIXES = (
    "ascend/backend/", "ascend/language/", "ascend/triton-adapter/",
    "triton_patch/python/", "triton_extension/", "bishengir/",
    "python/triton_kernels/", "include/", "lib/",
    "ascend/examples/tutorials/",
)


def api_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "KernelWiki-fetch-gitcode"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_bytes(url: str) -> bytes:
    """Fetch raw bytes with conservative retry/backoff for GitCode raw 418/429."""
    last_exc = None
    for attempt in range(max(1, RAW_RETRIES)):
        if RAW_SLEEP > 0:
            time.sleep(RAW_SLEEP)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 KernelWiki-fetch-gitcode/slow",
                "Accept": "text/plain,*/*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code not in (418, 429, 500, 502, 503, 504) or attempt == RAW_RETRIES - 1:
                raise
            sleep_s = max(RAW_SLEEP, 1.0) * (RAW_BACKOFF ** attempt)
            print(f"    WARN: raw fetch HTTP {e.code}; retry {attempt + 1}/{RAW_RETRIES - 1} after {sleep_s:.1f}s", file=sys.stderr)
            time.sleep(sleep_s)
        except Exception as e:
            last_exc = e
            if attempt == RAW_RETRIES - 1:
                raise
            sleep_s = max(RAW_SLEEP, 1.0) * (RAW_BACKOFF ** attempt)
            print(f"    WARN: raw fetch failed; retry {attempt + 1}/{RAW_RETRIES - 1} after {sleep_s:.1f}s: {e}", file=sys.stderr)
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def gitcode_url(repo: str, suffix: str) -> str:
    owner, name = repo.split("/", 1)
    return f"https://api.gitcode.com/api/v5/repos/{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}/{suffix.lstrip('/')}"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def split_frontmatter(text: str):
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    return yaml.safe_load(text[4:end]) or {}, text[end + 5:]


def write_frontmatter(path: Path, fm: dict, body: str):
    path.write_text("---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False, width=200) + "---\n" + body, encoding="utf-8")


def page_paths(repo_slug: str):
    return sorted((SOURCES / repo_slug).glob("PR-*.md"), key=lambda p: int(p.stem.split("-")[1]))


def bundle_has_key_files(ad: str) -> bool:
    if not ad:
        return False
    root = REPO_ROOT / ad
    return (root / "key-files").is_dir() and any((root / "key-files").rglob("*"))


def should_capture(filename: str) -> bool:
    if filename in SKIP_NAMES or filename.endswith((".md", ".rst", ".txt")):
        return False
    if filename.startswith(SKIP_PREFIXES):
        return False
    if not filename.endswith(SOURCE_EXTS):
        return False
    return filename.startswith(SOURCE_PREFIXES)


def patch_to_bytes(file_rows) -> bytes:
    chunks = []
    for f in file_rows:
        filename = f.get("filename") or ""
        patch = f.get("patch")
        diff = None
        if isinstance(patch, dict):
            diff = patch.get("diff")
        elif isinstance(patch, str):
            diff = patch
        if not diff:
            continue
        chunks.append(f"diff --git a/{filename} b/{filename}\n" + diff.rstrip() + "\n")
    return ("\n".join(chunks)).encode("utf-8")


def fetch_file_bytes(row, repo: str, upstream_sha: str):
    """Fetch file content through GitCode API raw endpoint.

The raw_url returned by /pulls/<N>/files often points at raw.gitcode.com,
which may return HTTP 418 for automated/bulk access. The API raw endpoint is
more stable and accepts the PR head sha as ref.
    """
    filename = row.get("filename", "")
    if not filename:
        return None
    ref = row.get("sha") or upstream_sha
    owner, name = repo.split("/", 1)
    url = (
        "https://api.gitcode.com/api/v5/repos/"
        f"{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}/raw/"
        f"{urllib.parse.quote(filename, safe='/')}?ref={urllib.parse.quote(ref)}"
    )
    return api_bytes(url)


def emit_bundle(repo: str, repo_slug: str, pr_num: int, pr_id: str, source_url: str, upstream_sha: str, file_rows, dry_run=False):
    bundle_final = ARTIFACTS / repo_slug / f"PR-{pr_num}"
    bundle_rel = bundle_final.relative_to(REPO_ROOT)
    captured = [r for r in file_rows if should_capture(r.get("filename", ""))][:30]
    diff = patch_to_bytes(file_rows)
    if dry_run:
        print(f"  DRY {pr_id}: diff={len(diff)} bytes, key_files={len(captured)}")
        return bundle_rel, 0, len(captured), False

    work = bundle_final.parent / f".{bundle_final.name}.new"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    entries = []
    total = 0
    truncated = False
    try:
        if diff:
            (work / "diff.patch").write_bytes(diff)
            total += len(diff)
            diff_entry = {"local_path": "diff.patch", "role": "pr-diff", "mode": "upstream-patch", "sha256": sha256(diff)}
            if len(diff) > FILE_SIZE_CAP:
                diff_entry["size_cap_truncated"] = True
                truncated = True
            entries.append(diff_entry)
        for row in captured:
            filename = row.get("filename", "")
            try:
                content = fetch_file_bytes(row, repo, upstream_sha)
            except Exception as e:
                print(f"    WARN: raw fetch failed for {filename}: {e}", file=sys.stderr)
                continue
            if content is None:
                continue
            out_rel = "key-files/" + filename
            out = work / out_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            file_trunc = False
            if len(content) > FILE_SIZE_CAP:
                content = (f"/* size_cap_truncated: upstream file is larger than {FILE_SIZE_CAP} bytes; see {row.get('raw_url','')} */\n").encode("utf-8")
                file_trunc = True
                truncated = True
            out.write_bytes(content)
            total += len(content)
            ent = {"local_path": out_rel, "role": "upstream-file", "mode": "verbatim", "upstream_path": filename, "sha256": sha256(content)}
            if file_trunc:
                ent["size_cap_truncated"] = True
            entries.append(ent)
        if not entries:
            note = b"No patch or key files were available from the GitCode API for this PR.\n"
            (work / "capture-note.txt").write_bytes(note)
            entries.append({"local_path": "capture-note.txt", "role": "approach-notes", "mode": "derived", "sha256": sha256(note)})
        if total > BUNDLE_SIZE_CAP and (work / "diff.patch").exists():
            total -= (work / "diff.patch").stat().st_size
            (work / "diff.patch").unlink()
            entries = [e for e in entries if e.get("local_path") != "diff.patch"]
            truncated = True
        prov = {
            "origin_url": source_url or f"https://gitcode.com/{repo}/merge_requests/{pr_num}",
            "upstream_repo": repo,
            "upstream_sha": upstream_sha or "unknown",
            "license": "inherits-from-upstream",
            "retrieved_at": date.today().isoformat(),
            "asset_mode": "verbatim",
            "size_cap_truncated": truncated,
            "generated_by": "scripts/fetch_gitcode_pr_artifacts.py",
            "source_pr_id": pr_id,
            "files": entries,
        }
        (work / "PROVENANCE.yaml").write_text(yaml.safe_dump(prov, sort_keys=False, allow_unicode=True, default_flow_style=False, width=200), encoding="utf-8")
        prev = None
        if bundle_final.exists():
            prev = bundle_final.parent / f".{bundle_final.name}.prev"
            if prev.exists():
                shutil.rmtree(prev)
            os.rename(bundle_final, prev)
        os.rename(work, bundle_final)
        if prev and prev.exists():
            shutil.rmtree(prev)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return bundle_rel, total, len(captured), truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-slug", default="triton-ascend")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.05, help="Sleep between PRs")
    ap.add_argument("--raw-sleep", type=float, default=0.0, help="Sleep before each raw file fetch")
    ap.add_argument("--raw-retries", type=int, default=1, help="Raw file fetch attempts")
    ap.add_argument("--raw-backoff", type=float, default=2.0, help="Exponential backoff multiplier for raw retries")
    ap.add_argument("--force", action="store_true", help="Refetch even when artifact_dir already exists")
    ap.add_argument("--only-missing-key-files", action="store_true", help="Only refetch pages whose bundle has no key-files")
    ap.add_argument("--ids", nargs="*", default=None, help="Specific source PR ids or PR numbers to fetch")
    args = ap.parse_args()
    global RAW_SLEEP, RAW_RETRIES, RAW_BACKOFF
    RAW_SLEEP = args.raw_sleep
    RAW_RETRIES = args.raw_retries
    RAW_BACKOFF = args.raw_backoff
    paths = page_paths(args.repo_slug)
    if args.ids:
        wanted = set(args.ids)
        wanted |= {f"pr-{args.repo_slug}-{x}" for x in args.ids if str(x).isdigit()}
        paths = [p for p in paths if (f"pr-{args.repo_slug}-{p.stem.split('-')[1]}" in wanted or p.stem.split('-')[1] in wanted)]
    if args.max:
        paths = paths[:args.max]
    ok = 0
    for i, md in enumerate(paths, 1):
        fm, body = split_frontmatter(md.read_text(encoding="utf-8"))
        if not fm:
            continue
        repo = fm.get("repo")
        pr_num = fm.get("pr")
        pr_id = fm.get("id")
        if not (repo and pr_num and pr_id):
            continue
        existing_ad = fm.get("artifact_dir")
        if existing_ad and (REPO_ROOT / existing_ad).is_dir() and not args.dry_run:
            if args.only_missing_key_files:
                if bundle_has_key_files(existing_ad):
                    print(f"[{i}/{len(paths)}] {pr_id} SKIP existing key-files")
                    ok += 1
                    continue
            elif not args.force:
                print(f"[{i}/{len(paths)}] {pr_id} SKIP existing artifact_dir")
                ok += 1
                continue
        print(f"[{i}/{len(paths)}] {pr_id}")
        try:
            pr = api_json(gitcode_url(repo, f"pulls/{pr_num}"))
            rows = api_json(gitcode_url(repo, f"pulls/{pr_num}/files"))
        except Exception as e:
            print(f"    ERROR: GitCode fetch failed: {e}", file=sys.stderr)
            continue
        if not isinstance(rows, list):
            rows = []
        head = pr.get("head") if isinstance(pr, dict) else {}
        base = pr.get("base") if isinstance(pr, dict) else {}
        upstream_sha = (head or {}).get("sha") or (base or {}).get("sha") or fm.get("merge_sha") or "unknown"
        try:
            bundle_rel, total, nfiles, truncated = emit_bundle(repo, args.repo_slug, int(pr_num), pr_id, fm.get("url", ""), upstream_sha, rows, args.dry_run)
        except Exception as e:
            print(f"    ERROR: bundle failed: {e}", file=sys.stderr)
            continue
        if not args.dry_run:
            fm["artifact_dir"] = str(bundle_rel)
            write_frontmatter(md, fm, body)
        print(f"    -> {bundle_rel} ({nfiles} key files, {total/1024:.1f} KiB, truncated={truncated})")
        ok += 1
        time.sleep(args.sleep)
    print(f"Fetched {ok}/{len(paths)} PR artifact bundles")

if __name__ == "__main__":
    main()
