#!/usr/bin/env bash
# wgmma-ptx probe driver sequence:
#   1. build (with cutlass-free verification gates)
#   2. exec the kernel and capture run-log
#   3. ncu CSV at the wgmma kernel's key counters
#   4. compute-sanitizer memcheck
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

mkdir -p profiles

# Build + cutlass-free verification (in build.sh).
echo "=== build (with cutlass-free verification) ==="
bash build.sh

# Execute the kernel and capture run-log.
echo "=== kernel execution ==="
./wgmma_hello | tee profiles/2026-04-28-wgmma-ptx-hello.run.log

# ncu CSV at the wgmma key counters.
echo "=== ncu counter capture ==="
if command -v ncu >/dev/null 2>&1; then
    ncu --target-processes all --csv \
        --metrics smsp__inst_executed_pipe_tensor_op_hmma_cycles_active.sum,\
sm__cycles_active.avg,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum \
        ./wgmma_hello \
        2>&1 | tee profiles/2026-04-28-wgmma-ptx-ncu.csv | tail -10
else
    echo "  ncu not available; skipping ncu step"
fi

# compute-sanitizer memcheck.
echo "=== compute-sanitizer memcheck ==="
if command -v compute-sanitizer >/dev/null 2>&1; then
    compute-sanitizer --tool memcheck \
        ./wgmma_hello \
        2>&1 | tee profiles/2026-04-28-wgmma-ptx-sanitizer.log | tail -10
else
    echo "  compute-sanitizer not available; skipping sanitizer step"
fi

echo ""
echo "=== probe complete. Artifacts ==="
echo "  build verification : (in build.sh tee output)"
echo "  run log            : profiles/2026-04-28-wgmma-ptx-hello.run.log"
echo "  ncu CSV            : profiles/2026-04-28-wgmma-ptx-ncu.csv"
echo "  sanitizer log      : profiles/2026-04-28-wgmma-ptx-sanitizer.log"
