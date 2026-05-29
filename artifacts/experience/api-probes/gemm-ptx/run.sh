#!/usr/bin/env bash
# AC-11 ptx-gemm canonical Stage-3.5 sequence.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

mkdir -p profiles

echo "=== Step 1: build + cutlass-free verification ==="
bash build.sh

echo "=== Step 2: numeric correctness vs cuBLAS ==="
./gemm_ptx | tee profiles/2026-04-28-gemm-ptx-hello.run.log

echo "=== Step 3: ncu CSV at the wgmma + TMA key counters ==="
if command -v ncu >/dev/null 2>&1; then
    ncu --target-processes all --csv \
        --metrics smsp__inst_executed_pipe_tensor_op_hmma_cycles_active.sum,\
sm__cycles_active.avg,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum \
        ./gemm_ptx \
        2>&1 | tee profiles/2026-04-28-gemm-ptx-ncu.csv | tail -10
else
    echo "  ncu not available; skipping ncu step"
fi

echo "=== Step 4: compute-sanitizer memcheck ==="
if command -v compute-sanitizer >/dev/null 2>&1; then
    compute-sanitizer --tool memcheck \
        ./gemm_ptx \
        2>&1 | tee profiles/2026-04-28-gemm-ptx-sanitizer.log | tail -10
else
    echo "  compute-sanitizer not available; skipping sanitizer step"
fi

echo ""
echo "=== Stage-3.5 complete (numeric correctness gate is partial pending fragment-store fix) ==="
echo "  run log     : profiles/2026-04-28-gemm-ptx-hello.run.log"
echo "  ncu CSV     : profiles/2026-04-28-gemm-ptx-ncu.csv"
echo "  sanitizer   : profiles/2026-04-28-gemm-ptx-sanitizer.log"
