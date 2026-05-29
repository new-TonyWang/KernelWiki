#!/usr/bin/env bash
# AC-7 fused-GEMM canonical Stage-3.5 runner.
#
# Current scope: epilogue fusion only (gemm_compare_relu vs gemm_compare + relu_kernel).
# Prologue fusion (cutlass example 55 mixed-dtype int4 × bf16) is NOT yet integrated;
# when it lands as gemm_compare_mixed_dtype + standalone_dequant_kernel, this runner
# will be extended to time both A/Bs.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

mkdir -p profiles

echo "=== Step 1: build (gemm_compare + gemm_compare_relu + relu_kernel) ==="
bash build_fused.sh

echo "=== Step 2: epilogue A/B at 2048^3 ==="
echo "--- non-fused: gemm_compare (cooperative GEMM only) ---" | tee profiles/2026-04-28-gemm-fused.run.log
./gemm_compare      --m=2048 --n=2048 --k=2048 --iterations=20 2>&1 | tee -a profiles/2026-04-28-gemm-fused.run.log
echo "--- non-fused: relu_kernel (standalone ReLU) ---" | tee -a profiles/2026-04-28-gemm-fused.run.log
./relu_kernel       --m=2048 --n=2048             --iterations=50 2>&1 | tee -a profiles/2026-04-28-gemm-fused.run.log
echo "--- fused: gemm_compare_relu (cooperative GEMM + LinCombEltAct<ReLu>) ---" | tee -a profiles/2026-04-28-gemm-fused.run.log
./gemm_compare_relu --m=2048 --n=2048 --k=2048 --iterations=20 2>&1 | tee -a profiles/2026-04-28-gemm-fused.run.log

echo "=== Step 3: ncu CSV at 2048^3 (compare DRAM bytes between fused and non-fused) ==="
if command -v ncu >/dev/null 2>&1; then
    echo "--- ncu: gemm_compare (non-fused GEMM) ---" | tee profiles/2026-04-28-gemm-fused-ncu.csv
    ncu --target-processes all --csv \
        --metrics dram__bytes.sum,sm__cycles_active.avg,sm__warps_active.avg.pct_of_peak_sustained_active \
        ./gemm_compare      --m=2048 --n=2048 --k=2048 --iterations=2 \
        2>&1 | tee -a profiles/2026-04-28-gemm-fused-ncu.csv
    echo "--- ncu: relu_kernel (non-fused ReLU) ---" | tee -a profiles/2026-04-28-gemm-fused-ncu.csv
    ncu --target-processes all --csv \
        --metrics dram__bytes.sum,sm__cycles_active.avg \
        ./relu_kernel       --m=2048 --n=2048             --iterations=2 \
        2>&1 | tee -a profiles/2026-04-28-gemm-fused-ncu.csv
    echo "--- ncu: gemm_compare_relu (fused) ---" | tee -a profiles/2026-04-28-gemm-fused-ncu.csv
    ncu --target-processes all --csv \
        --metrics dram__bytes.sum,sm__cycles_active.avg,sm__warps_active.avg.pct_of_peak_sustained_active \
        ./gemm_compare_relu --m=2048 --n=2048 --k=2048 --iterations=2 \
        2>&1 | tee -a profiles/2026-04-28-gemm-fused-ncu.csv
else
    echo "  ncu not available; skipping ncu step"
fi

echo "=== Step 4: compute-sanitizer memcheck (fused kernel) ==="
if command -v compute-sanitizer >/dev/null 2>&1; then
    compute-sanitizer --tool memcheck \
        ./gemm_compare_relu --m=2048 --n=2048 --k=2048 --iterations=2 \
        2>&1 | tee profiles/2026-04-28-gemm-fused-sanitizer.log | tail -10
fi

echo ""
echo "=== AC-7 Stage-3.5 (epilogue half) complete. Prologue half remains queued. ==="
echo "  run log     : profiles/2026-04-28-gemm-fused.run.log"
echo "  ncu CSV     : profiles/2026-04-28-gemm-fused-ncu.csv"
echo "  sanitizer   : profiles/2026-04-28-gemm-fused-sanitizer.log"
