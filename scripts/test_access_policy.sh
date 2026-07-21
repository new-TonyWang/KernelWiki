#!/usr/bin/env bash
# Regression test: only wiki/, sources/ and artifacts/ are retrievable.
#
# Containment inside WIKI_ROOT is not the boundary — the repo root also holds
# scripts/ (the MCP server's own auth code), data/, corpus/, .git/ and the
# working directories agent/, candidates/, reasoning/, tasks/, tests/,
# templates/. This probes every route that turns caller input into a file read:
#
#   1. wiki_get_page path lookup into each forbidden directory
#   2. wiki_grep across all four scopes for a string that only exists in scripts/
#   3. wiki_get_page include_code with artifact_dir aimed at a forbidden dir
#
# Usage: bash scripts/test_access_policy.sh
# Exit code 0 = all probes pass, non-zero = a forbidden directory is reachable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WIKI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROBE_PAGE="$WIKI_ROOT/sources/docs/__access_policy_probe_$$.md"

cleanup() { rm -f "$PROBE_PAGE"; }
trap cleanup EXIT

cd "$WIKI_ROOT"
WIKI_ROOT="$WIKI_ROOT" PROBE_PAGE="$PROBE_PAGE" python3 - <<'PYEOF'
import json, os, subprocess, sys

ROOT = os.environ["WIKI_ROOT"]
PROBE_PAGE = os.environ["PROBE_PAGE"]
PROBE_ID = "doc-" + os.path.basename(PROBE_PAGE)[:-3]
SERVER = os.path.join(ROOT, "scripts", "mcp_server.py")

FORBIDDEN = ["agent", "candidates", "corpus", "data", "reasoning",
             "scripts", "tasks", "tests", "templates"]

passed = failed = 0


def report(ok, label, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS: {label}")
    else:
        failed += 1
        print(f"  FAIL: {label}  {detail}")


def call(requests):
    payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run([sys.executable, SERVER], input=payload,
                          capture_output=True, text=True, cwd=ROOT, timeout=600)
    out = []
    for line in proc.stdout.splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def envelope(resp):
    return json.loads(resp["result"]["content"][0]["text"])


init = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}


def sample_file(d):
    for dirpath, _, names in os.walk(os.path.join(ROOT, d)):
        for n in sorted(names):
            if n.endswith((".md", ".py", ".yaml", ".yml", ".json", ".sh", ".tsv")):
                return os.path.relpath(os.path.join(dirpath, n), ROOT)
    return None


# --- 1. direct path lookup into each forbidden directory --------------------
targets = [(d, sample_file(d)) for d in FORBIDDEN]
targets = [(d, f) for d, f in targets if f]
reqs = [init] + [
    {"jsonrpc": "2.0", "id": i + 1, "method": "tools/call",
     "params": {"name": "wiki_get_page",
                "arguments": {"lookup": f, "body_only": True}}}
    for i, (_, f) in enumerate(targets)
]
for (d, f), resp in zip(targets, call(reqs)[1:]):
    env = envelope(resp)
    report(not env.get("ok"), f"wiki_get_page cannot read {d}/", f"served {f}")

# --- 2. grep must not reach scripts/ from any scope -------------------------
NEEDLE = "def load_all_pages"
reqs = [init] + [
    {"jsonrpc": "2.0", "id": i + 1, "method": "tools/call",
     "params": {"name": "wiki_grep",
                "arguments": {"patterns": [NEEDLE], "scope": s, "limit": 20}}}
    for i, s in enumerate(["all", "wiki", "sources", "artifacts"])
]
for scope, resp in zip(["all", "wiki", "sources", "artifacts"], call(reqs)[1:]):
    env = envelope(resp)
    hits = env.get("returned", 0) if env.get("ok") else 0
    report(hits == 0, f"wiki_grep scope={scope} cannot reach scripts/",
           f"{hits} hits for {NEEDLE!r}")

# --- 3. artifact_dir must not escape into a forbidden directory -------------
for target in ["scripts", "data", "tasks", "templates"]:
    with open(PROBE_PAGE, "w", encoding="utf-8") as fh:
        fh.write("---\n"
                 f"id: {PROBE_ID}\n"
                 "title: access policy probe\n"
                 "type: doc\n"
                 f"artifact_dir: {target}\n"
                 "---\n\nprobe\n")
    resp = call([init, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "wiki_get_page",
                                   "arguments": {"lookup": PROBE_ID,
                                                 "include_code": True,
                                                 "frontmatter_only": True}}}])[-1]
    env = envelope(resp)
    data = env.get("data") or {}
    n = len(data.get("artifact_files") or [])
    # A truncated envelope would hide a leak behind the output budget, so treat
    # a missing data section as a failure rather than a pass.
    ok = bool(data) and n == 0
    report(ok, f"artifact_dir={target} serves no files",
           f"{n} files, truncated={env.get('truncated')}")

print()
print(f"=== Results: {passed} passed, {failed} failed ===")
if failed:
    print("ACCESS POLICY BROKEN — a forbidden directory is retrievable!")
    sys.exit(1)
print("All access policy probes passed.")
PYEOF
