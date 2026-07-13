#!/usr/bin/env bash
# Smoke test for KernelWiki MCP server using JSON fixture transcripts.
#
# Each fixture file in test_fixtures/*.json contains an array of test cases
# with request/check pairs. All requests are sent in a single server session
# to verify server-stays-alive behavior across errors.
#
# Usage: bash scripts/test_mcp_smoke.sh
# Exit code 0 = all tests pass, non-zero = failures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURE_DIR="$SCRIPT_DIR/test_fixtures"

echo "=== KernelWiki MCP Server Smoke Tests ==="
echo

exec python3 - "$FIXTURE_DIR" "$SCRIPT_DIR/mcp_server.py" <<'PYEOF'
import json, sys, subprocess, os, glob


def navigate(obj, path):
    """Navigate a jq-like path through a JSON object.

    Supports:
      .key.subkey          - dict access
      .key[0]              - array index
      .key | length        - array/dict length
      .key | fromjson .sub - parse JSON string then navigate
    """
    if " | " in path:
        parts = path.split(" | ", 1)
        obj = navigate(obj, parts[0])
        rest = parts[1]
        if rest == "length":
            return len(obj)
        if rest.startswith("fromjson"):
            obj = json.loads(obj)
            remainder = rest[len("fromjson"):].strip()
            if remainder:
                return navigate(obj, remainder)
            return obj
        raise ValueError(f"unknown pipe op: {rest}")

    if path.startswith("."):
        path = path[1:]
    if not path:
        return obj

    tokens = []
    current = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if current:
                tokens.append(current)
                current = ""
        elif c == "[":
            if current:
                tokens.append(current)
                current = ""
            end = path.index("]", i)
            tokens.append(("index", int(path[i+1:end])))
            i = end
        else:
            current += c
        i += 1
    if current:
        tokens.append(current)

    for tok in tokens:
        if isinstance(tok, tuple) and tok[0] == "index":
            obj = obj[tok[1]]
        else:
            obj = obj[tok]
    return obj


def main():
    fixture_dir = sys.argv[1]
    server_script = sys.argv[2]

    # Collect all test cases from fixture files
    test_cases = []
    for fpath in sorted(glob.glob(os.path.join(fixture_dir, "*.json"))):
        with open(fpath) as f:
            cases = json.load(f)
        for case in cases:
            case["_fixture"] = os.path.basename(fpath)
            test_cases.append(case)

    # Build the input to send to the server (one line per request)
    input_lines = []
    for case in test_cases:
        if "request_raw" in case:
            input_lines.append(case["request_raw"])
        else:
            input_lines.append(json.dumps(case["request"]))

    stdin_data = "\n".join(input_lines) + "\n"

    # Run the server
    proc = subprocess.run(
        [sys.executable, server_script],
        input=stdin_data, capture_output=True, text=True,
        timeout=60,
        env={**os.environ, "MCP_LOG_FILE": "/dev/null"},
    )

    # Parse response lines
    resp_lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
    responses = []
    for line in resp_lines:
        try:
            responses.append(json.loads(line))
        except json.JSONDecodeError:
            responses.append({"_parse_error": line})

    # Match responses to test cases
    resp_idx = 0
    results = []

    for case in test_cases:
        name = case.get("name", "unnamed")
        fixture = case.get("_fixture", "?")
        expect_no_response = case.get("expect_no_response", False)

        if expect_no_response:
            results.append((f"[{fixture}] {name}", True, "no response expected"))
            continue

        if resp_idx >= len(responses):
            results.append((f"[{fixture}] {name}", False, "no response received (server died?)"))
            continue

        resp = responses[resp_idx]
        resp_idx += 1

        if "_parse_error" in resp:
            results.append((f"[{fixture}] {name}", False,
                            f"unparseable response: {resp['_parse_error'][:100]}"))
            continue

        checks = case.get("checks", [])
        all_ok = True
        fail_detail = None
        for chk in checks:
            path = chk["path"]
            op = chk["op"]
            expected = chk.get("expected")

            try:
                val = navigate(resp, path)
            except Exception as e:
                all_ok = False
                fail_detail = f"path {path}: {e}"
                break

            if op == "eq":
                if val != expected:
                    all_ok = False
                    fail_detail = f"path {path}: expected {expected!r}, got {val!r}"
                    break
            elif op == "neq":
                if val == expected:
                    all_ok = False
                    fail_detail = f"path {path}: expected != {expected!r}, got {val!r}"
                    break
            elif op == "gte":
                if not (isinstance(val, (int, float)) and val >= expected):
                    all_ok = False
                    fail_detail = f"path {path}: expected >= {expected}, got {val!r}"
                    break
            elif op == "lte":
                if not (isinstance(val, (int, float)) and val <= expected):
                    all_ok = False
                    fail_detail = f"path {path}: expected <= {expected}, got {val!r}"
                    break
            elif op == "contains":
                if expected not in str(val):
                    all_ok = False
                    fail_detail = f"path {path}: expected to contain {expected!r}"
                    break
            elif op == "exists":
                pass
            else:
                all_ok = False
                fail_detail = f"unknown op: {op}"
                break

        results.append((f"[{fixture}] {name}", all_ok, fail_detail or "ok"))

    # Verify server-stays-alive: we should have gotten responses for all
    # non-notification cases
    expected_resp_count = sum(1 for c in test_cases if not c.get("expect_no_response", False))
    if resp_idx < expected_resp_count:
        results.append(("server-stays-alive", False,
                        f"expected {expected_resp_count} responses, got {resp_idx}"))
    else:
        results.append(("server-stays-alive", True, "ok"))

    # Print results
    pass_count = 0
    fail_count = 0
    for name, passed, detail in results:
        if passed:
            print(f"  PASS: {name}")
            pass_count += 1
        else:
            print(f"  FAIL: {name} ({detail})")
            fail_count += 1

    print()
    print(f"=== Results: {pass_count} passed, {fail_count} failed, {pass_count + fail_count} total ===")
    if fail_count > 0:
        sys.exit(1)
    print("All smoke tests passed.")


main()
PYEOF
