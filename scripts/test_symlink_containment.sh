#!/usr/bin/env bash
# Regression test: symlink containment across all 3 services.
#
# Creates a temporary .md symlink under wiki/ pointing outside WIKI_ROOT,
# then probes wiki_query, wiki_grep, and wiki_get_page via the MCP server
# to verify the escaped content is never returned.
#
# Usage: bash scripts/test_symlink_containment.sh
# Exit code 0 = all probes pass, non-zero = containment leak found.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WIKI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTSIDE_FILE="/tmp/kernelwiki-symlink-escape-probe-$$.md"
SYMLINK_PATH="$WIKI_ROOT/wiki/__symlink_escape_probe_$$.md"

cleanup() {
    rm -f "$OUTSIDE_FILE" "$SYMLINK_PATH"
    rm -f "/tmp/kernelwiki-hascode-escape-$$.cu" \
          "$WIKI_ROOT/artifacts/__hascode_escape_$$.cu" \
          "$WIKI_ROOT/wiki/__hascode_escape_page_$$.md"
}
trap cleanup EXIT

# Create an outside-root file with a unique tag in frontmatter
UNIQUE_TAG="SYMLINK_ESCAPE_CANARY_$$"
cat > "$OUTSIDE_FILE" <<EOF
---
id: symlink-escape-canary-$$
title: Symlink Escape Canary
type: technique
vendor: nvidia
tags: [$UNIQUE_TAG]
confidence: experimental
---

This file lives outside WIKI_ROOT. If you can see $UNIQUE_TAG, containment is broken.
EOF

# Create symlink inside wiki/ pointing to the outside file
ln -sf "$OUTSIDE_FILE" "$SYMLINK_PATH"

echo "=== Symlink Containment Regression Test ==="
echo "Outside file: $OUTSIDE_FILE"
echo "Symlink:      $SYMLINK_PATH"
echo

# Build MCP requests: query for the canary tag, grep for the canary string, get_page by id
REQUESTS=$(cat <<JSONEOF
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"wiki_query","arguments":{"query":["$UNIQUE_TAG"],"limit":10}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"wiki_grep","arguments":{"patterns":["$UNIQUE_TAG"],"scope":"wiki","limit":10}}}
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"wiki_get_page","arguments":{"lookup":"symlink-escape-canary-$$"}}}
JSONEOF
)

# Run MCP server
RESPONSES=$(echo "$REQUESTS" | python3 "$SCRIPT_DIR/mcp_server.py" 2>/dev/null)

PASS=0
FAIL=0

check() {
    local name="$1"
    local condition="$2"
    if eval "$condition"; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
}

# Extract responses by id
resp_query=$(echo "$RESPONSES" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    obj = json.loads(line)
    if obj.get('id') == 2:
        print(line)
        break
")

resp_grep=$(echo "$RESPONSES" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    obj = json.loads(line)
    if obj.get('id') == 3:
        print(line)
        break
")

resp_page=$(echo "$RESPONSES" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    obj = json.loads(line)
    if obj.get('id') == 4:
        print(line)
        break
")

# wiki_query: canary tag must not appear in results
check "wiki_query does not return escaped symlink content" \
    '! echo "$resp_query" | grep -q "'"$UNIQUE_TAG"'"'

# wiki_grep: canary string must not appear in results
check "wiki_grep does not return escaped symlink content" \
    '! echo "$resp_grep" | grep -q "'"$UNIQUE_TAG"'"'

# wiki_get_page: must return PAGE_NOT_FOUND
check "wiki_get_page returns PAGE_NOT_FOUND for escaped symlink" \
    'echo "$resp_page" | grep -q "PAGE_NOT_FOUND"'

# --- Direct service-level regression tests ---
echo
echo "--- Direct service reader tests ---"

# Test load_frontmatter() directly with the escaped symlink
DIRECT_FM_RESULT=$(PYTHONPATH="$SCRIPT_DIR" python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from pathlib import Path
from wiki_query_service import load_frontmatter
fm, body = load_frontmatter(Path('$SYMLINK_PATH'))
print('BLOCKED' if fm is None else fm.get('id', 'UNKNOWN'))
")

check "load_frontmatter() blocks escaped symlink" \
    '[ "$DIRECT_FM_RESULT" = "BLOCKED" ]'

# Test grep_file() directly with the escaped symlink
DIRECT_GREP_RESULT=$(PYTHONPATH="$SCRIPT_DIR" python3 -c "
import re, sys
sys.path.insert(0, '$SCRIPT_DIR')
from pathlib import Path
from wiki_grep_service import grep_file
hits = grep_file(Path('$SYMLINK_PATH'), [re.compile('$UNIQUE_TAG')], 0, True)
print(len(hits))
")

check "grep_file() blocks escaped symlink" \
    '[ "$DIRECT_GREP_RESULT" = "0" ]'

# --- has_code artifact escape test ---
echo
echo "--- has_code artifact escape test ---"

# Create a real wiki page that references an escaped artifact symlink
HASCODE_OUTSIDE="/tmp/kernelwiki-hascode-escape-$$.cu"
HASCODE_SYMLINK="$WIKI_ROOT/artifacts/__hascode_escape_$$.cu"
HASCODE_PAGE="$WIKI_ROOT/wiki/__hascode_escape_page_$$.md"
HASCODE_TAG="HASCODE_ESCAPE_CANARY_$$"

cat > "$HASCODE_OUTSIDE" <<CUEOF
__global__ void canary_kernel() { /* $HASCODE_TAG */ }
CUEOF

ln -sf "$HASCODE_OUTSIDE" "$HASCODE_SYMLINK"

cat > "$HASCODE_PAGE" <<MDEOF
---
id: hascode-escape-page-$$
title: Has-Code Escape Test Page
type: technique
vendor: nvidia
tags: [$HASCODE_TAG]
confidence: experimental
artifacts:
  - artifacts/__hascode_escape_$$.cu
---

Test page for has_code artifact escape.
MDEOF

# Test that filter_pages with has_code=True does NOT return this page
HASCODE_RESULT=$(PYTHONPATH="$SCRIPT_DIR" python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from wiki_query_service import load_all_pages, filter_pages
pages = load_all_pages()
filtered = filter_pages(pages, {'tag': '$HASCODE_TAG', 'has_code': True})
print(len(filtered))
")

check "has_code filter ignores escaped artifact symlink" \
    '[ "$HASCODE_RESULT" = "0" ]'

# Cleanup extra files
rm -f "$HASCODE_OUTSIDE" "$HASCODE_SYMLINK" "$HASCODE_PAGE"

echo
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "SYMLINK CONTAINMENT BROKEN — leaked outside-root content!"
    exit 1
fi
echo "All symlink containment probes passed."
