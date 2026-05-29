#!/usr/bin/env python3
"""Deterministic AC-6 smoke test — exercises the full agent+corpus+validation
pipeline without requiring external API access.

Usage:
    python3 scripts/smoke_test.py

Exits 0 if all checks pass, 1 otherwise.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHECKS = []


def check(name):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("Task YAML loads")
def test_task_load():
    from agent.shared.task_schema import load_task
    task = load_task(str(REPO_ROOT / "tasks" / "build-warp-primitives.yaml"))
    assert isinstance(task, dict), f"Expected dict, got {type(task)}"
    return f"name={task.get('name', 'OK')}"


@check("Claude prompt generates")
def test_prompt():
    from agent.claude_code_path.subagent_prompt import build_prompt
    prompt = build_prompt(str(REPO_ROOT / "tasks" / "build-warp-primitives.yaml"))
    assert len(prompt) > 100, f"Prompt too short: {len(prompt)}"
    return f"{len(prompt)} chars"


@check("TOOL_REGISTRY imports")
def test_tools():
    from agent.shared.tools import TOOL_REGISTRY
    assert len(TOOL_REGISTRY) > 0
    return f"{len(TOOL_REGISTRY)} tools"


@check("Source search (tier-1)")
def test_search():
    from scripts.source_corpus.service import source_search
    result = source_search("__shfl_sync", scope="cuda-official", top_k=3)
    assert result["ok"], f"Search failed: {result.get('message')}"
    assert result["data"]["total_hits"] > 0
    return f"{result['data']['total_hits']} hits"


@check("Source read (roundtrip from search)")
def test_read():
    from scripts.source_corpus.service import source_search, source_read
    sr = source_search("__shfl_sync", scope="cuda-official", top_k=1)
    hit_path = sr["data"]["hits"][0]["path"]
    result = source_read(hit_path)
    assert result["ok"], f"Read failed: {result.get('message')}"
    return f"title={result['data']['title']}"


@check("Source list")
def test_list():
    from scripts.source_corpus.service import list_sources
    result = list_sources(scope="cuda-official")
    assert result["ok"]
    return f"count={result['data']['count']}"


@check("Source resolve")
def test_resolve():
    from scripts.source_corpus.service import resolve_source
    result = resolve_source("cuda-official")
    assert result["ok"]
    return "resolved"


@check("Provenance walk")
def test_provenance():
    from scripts.source_corpus.provenance import provenance_walk
    result = provenance_walk("wiki/nvidia/foundations/compute/gemm.md")
    assert result["ok"], f"Provenance failed: {result.get('message')}"
    return f"refs={result['data']['count']}"


@check("Validation passes")
def test_validate():
    result = subprocess.run(
        ["python3", "scripts/validate.py"],
        capture_output=True, text=True, timeout=300,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"Validation failed:\n{result.stdout[-500:]}"
    return "0 errors"


def main():
    print("=== AC-6 Deterministic Smoke Test ===\n")
    passed = 0
    failed = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print(f"  PASS  {name}: {detail}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
