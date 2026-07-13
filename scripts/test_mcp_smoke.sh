#!/usr/bin/env bash
# Smoke test for KernelWiki MCP server.
# Sends JSON-RPC 2.0 messages to the server and validates responses.
#
# Usage: bash scripts/test_mcp_smoke.sh
# Exit code 0 = all tests pass, non-zero = failures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
TOTAL=0

# Helper: send a single JSON-RPC message and capture the response
send_msg() {
    local msg="$1"
    printf '%s\n' "$msg" | python3 "$SCRIPT_DIR/mcp_server.py" 2>/dev/null
}

# Helper: check that a response line matches expectations via jq
check() {
    local test_name="$1"
    local response="$2"
    local jq_filter="$3"
    local expected="$4"
    TOTAL=$((TOTAL + 1))

    local actual
    actual=$(echo "$response" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    msg = json.loads(line)
    # Navigate using the jq-like path
    result = msg
    for key in '''$jq_filter'''.strip('.').split('.'):
        if key.startswith('['):
            result = result[int(key.strip('[]'))]
        else:
            result = result[key]
    print(result)
    break
" 2>/dev/null || echo "__ERROR__")

    if [ "$actual" = "$expected" ]; then
        echo "  PASS: $test_name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $test_name (expected '$expected', got '$actual')"
        FAIL=$((FAIL + 1))
    fi
}

# Helper: check a field inside the MCP tool result's text content
check_tool_result() {
    local test_name="$1"
    local response="$2"
    local field="$3"
    local expected="$4"
    TOTAL=$((TOTAL + 1))

    local actual
    actual=$(echo "$response" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    msg = json.loads(line)
    content_text = msg['result']['content'][0]['text']
    envelope = json.loads(content_text)
    # Navigate the field path
    result = envelope
    for key in '''$field'''.strip('.').split('.'):
        if key.startswith('['):
            result = result[int(key.strip('[]'))]
        elif key.isdigit():
            result = result[int(key)]
        else:
            result = result[key]
    print(result)
    break
" 2>/dev/null || echo "__ERROR__")

    if [ "$actual" = "$expected" ]; then
        echo "  PASS: $test_name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $test_name (expected '$expected', got '$actual')"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== KernelWiki MCP Server Smoke Tests ==="
echo

# --- Test 1: initialize ---
echo "Test: initialize"
RESP=$(send_msg '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke-test","version":"0.1"}}}')
check "protocol version" "$RESP" ".result.protocolVersion" "2024-11-05"
check "server name" "$RESP" ".result.serverInfo.name" "kernel-wiki"

# --- Test 2: tools/list ---
echo "Test: tools/list"
RESP=$(send_msg '{"jsonrpc":"2.0","id":2,"method":"tools/list"}')
TOTAL=$((TOTAL + 1))
TOOL_COUNT=$(echo "$RESP" | python3 -c "
import sys, json
for line in sys.stdin:
    msg = json.loads(line.strip())
    print(len(msg['result']['tools']))
    break
" 2>/dev/null || echo "0")
if [ "$TOOL_COUNT" = "3" ]; then
    echo "  PASS: 3 tools registered"
    PASS=$((PASS + 1))
else
    echo "  FAIL: expected 3 tools, got $TOOL_COUNT"
    FAIL=$((FAIL + 1))
fi

# --- Test 3: wiki_query ---
echo "Test: wiki_query"
RESP=$(send_msg '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"wiki_query","arguments":{"query":["tcgen05"],"limit":5,"compact":true}}}')
check_tool_result "query ok" "$RESP" ".ok" "True"
check_tool_result "query has results" "$RESP" ".truncated" "True"

# --- Test 4: wiki_get_page ---
echo "Test: wiki_get_page"
RESP=$(send_msg '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"wiki_get_page","arguments":{"lookup":"hw-tcgen05-mma","body_only":true}}}')
check_tool_result "get_page ok" "$RESP" ".ok" "True"
check_tool_result "get_page title" "$RESP" ".data.title" "tcgen05.mma — Blackwell MMA Instruction"

# --- Test 5: wiki_get_page not found ---
echo "Test: wiki_get_page (not found)"
RESP=$(send_msg '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"wiki_get_page","arguments":{"lookup":"nonexistent-page-xyz"}}}')
check_tool_result "not_found error" "$RESP" ".ok" "False"
check_tool_result "not_found code" "$RESP" ".error_code" "not_found"

# --- Test 6: wiki_grep ---
echo "Test: wiki_grep"
RESP=$(send_msg '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"wiki_grep","arguments":{"patterns":["tcgen05"],"scope":"wiki","limit":3}}}')
check_tool_result "grep ok" "$RESP" ".ok" "True"

# --- Test 7: path traversal ---
echo "Test: path traversal protection"
RESP=$(send_msg '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"wiki_get_page","arguments":{"lookup":"../../etc/passwd"}}}')
check_tool_result "traversal blocked" "$RESP" ".error_code" "invalid_params"

# --- Test 8: invalid regex ---
echo "Test: invalid regex"
RESP=$(send_msg '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"wiki_grep","arguments":{"patterns":["[invalid"]}}}')
check_tool_result "invalid regex error" "$RESP" ".error_code" "invalid_params"

# --- Test 9: unknown tool ---
echo "Test: unknown tool"
RESP=$(send_msg '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"nonexistent_tool","arguments":{}}}')
check_tool_result "unknown tool error" "$RESP" ".error_code" "unknown_tool"

# --- Test 10: unknown method ---
echo "Test: unknown method"
RESP=$(send_msg '{"jsonrpc":"2.0","id":10,"method":"nonexistent/method"}')
TOTAL=$((TOTAL + 1))
ERR_CODE=$(echo "$RESP" | python3 -c "
import sys, json
for line in sys.stdin:
    msg = json.loads(line.strip())
    print(msg.get('error', {}).get('code', 'none'))
    break
" 2>/dev/null || echo "none")
if [ "$ERR_CODE" = "-32601" ]; then
    echo "  PASS: unknown method returns -32601"
    PASS=$((PASS + 1))
else
    echo "  FAIL: expected error code -32601, got $ERR_CODE"
    FAIL=$((FAIL + 1))
fi

# --- Test 11: ping ---
echo "Test: ping"
RESP=$(send_msg '{"jsonrpc":"2.0","id":11,"method":"ping"}')
check "ping response" "$RESP" ".result" "{}"

# --- Summary ---
echo
echo "=== Results: $PASS passed, $FAIL failed, $TOTAL total ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "All smoke tests passed."
