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


@check("Source search via TOOL_REGISTRY (tier-1)")
def test_search():
    import json
    from agent.shared.tools import TOOL_REGISTRY
    search_fn = TOOL_REGISTRY["source_search"]
    result_str = search_fn("__shfl_sync", scope="cuda-official", top_k=3)
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    assert result["ok"], f"Search failed: {result.get('message')}"
    assert result["data"]["total_hits"] > 0
    return f"{result['data']['total_hits']} hits"


@check("Source read via TOOL_REGISTRY (roundtrip)")
def test_read():
    import json
    from agent.shared.tools import TOOL_REGISTRY
    search_fn = TOOL_REGISTRY["source_search"]
    read_fn = TOOL_REGISTRY["source_read"]
    sr_str = search_fn("__shfl_sync", scope="cuda-official", top_k=1)
    sr = json.loads(sr_str) if isinstance(sr_str, str) else sr_str
    hit_path = sr["data"]["hits"][0]["path"]
    result_str = read_fn(hit_path)
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    assert result["ok"], f"Read failed for '{hit_path}': {result.get('message')}"
    return f"title={result['data']['title']}"


@check("Source list via TOOL_REGISTRY")
def test_list():
    import json
    from agent.shared.tools import TOOL_REGISTRY
    list_fn = TOOL_REGISTRY["source_list"]
    result_str = list_fn(scope="cuda-official")
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    assert result["ok"]
    return f"count={result['data']['count']}"


@check("Source resolve via TOOL_REGISTRY")
def test_resolve():
    import json
    from agent.shared.tools import TOOL_REGISTRY
    resolve_fn = TOOL_REGISTRY["source_resolve"]
    result_str = resolve_fn("cuda-official")
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    assert result["ok"]
    return "resolved"


@check("Provenance walk via TOOL_REGISTRY")
def test_provenance():
    import json
    from agent.shared.tools import TOOL_REGISTRY
    prov_fn = TOOL_REGISTRY["provenance_walk"]
    result_str = prov_fn("wiki/nvidia/foundations/compute/gemm.md")
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
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
