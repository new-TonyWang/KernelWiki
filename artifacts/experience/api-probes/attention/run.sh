#!/usr/bin/env bash
# Flash-attention probe driver on H200 (sm_90a).
# Captures all evidence; fails if any kernel returns non-zero.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p profiles

RESULT=0

echo "=== build ==="
bash build.sh || { echo "BUILD FAILED"; exit 1; }

echo ""
echo "=== descriptor validation (NN canonical SW128, all-ones + random) ==="
./wgmma_desc_test 2>&1 | tee profiles/wgmma-desc-test.log
DESC_RC=${PIPESTATUS[0]}
echo "descriptor test exit code: $DESC_RC"
if [ "$DESC_RC" -ne 0 ]; then RESULT=1; fi
echo ""

echo "=== TMA+wgmma correctness gate (B=1 H=2 S=128 D=64) ==="
./flash_attn_tma_wgmma 2>&1 | tee profiles/tma-wgmma-correctness.log
TMA_RC=${PIPESTATUS[0]}
echo "TMA+wgmma exit code: $TMA_RC"
if [ "$TMA_RC" -ne 0 ]; then RESULT=1; fi
echo ""

echo "=== thread-level reference correctness (B=1 H=2 S=128 D=64) ==="
./flash_attn_minimal 2>&1 | tee profiles/thread-level-correctness.log
REF_RC=${PIPESTATUS[0]}
echo "thread-level exit code: $REF_RC"
if [ "$REF_RC" -ne 0 ]; then RESULT=1; fi
echo ""

echo "=== thread-level tuning sweep (BLOCK_M x BLOCK_N tile-shape axis) ==="
./flash_attn_minimal --tune | tee profiles/attention-sweep.csv
SWEEP_RC=${PIPESTATUS[0]}
if [ "$SWEEP_RC" -ne 0 ]; then RESULT=1; fi
echo ""

echo "=== ncu profiler TMA+wgmma (key metrics) ==="
if command -v ncu >/dev/null 2>&1; then
    ncu --target-processes all --csv \
        --metrics sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_tensor_op_hmma.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
        ./flash_attn_tma_wgmma \
        2>&1 | tee profiles/tma-wgmma-ncu-raw.csv | tail -10
else
    echo "  ncu not available; skipping profiler step"
fi
echo ""

echo "=== compute-sanitizer TMA+wgmma ==="
compute-sanitizer --tool memcheck ./flash_attn_tma_wgmma 2>&1 | tee profiles/tma-wgmma-sanitizer.log || true
echo ""

echo "=== done (overall result: $RESULT) ==="
exit $RESULT
