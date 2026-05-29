#!/usr/bin/env bash
CUDA_REPO_ROOT=${CUDA_REPO_ROOT:-/path/to/your/cuda_repo}
CUTLASS_DIR=${CUTLASS_DIR:-${CUDA_REPO_ROOT}/cutlass}
# AC-5 (non-aligned GEMM tail handling) probe driver.
#
# Two kernel templates are exercised because the cooperative kernel's M-floor
# (effectively 128 in the M dimension when ClusterShape × CtaTile.M is taken
# into account) cannot host the AC-5 "far smaller than tile" category. We use
# a smaller-tile non-cooperative variant (TileShape <64,64,32>, Cluster <1,1,1>)
# for the small shape, and the AC-4 cooperative kernel for the two larger
# shapes:
#
#   80^3   far smaller than tile     -> gemm_tail_small / gemm_compare_small (small-tile;
#                                       genuinely non-aligned: 80 = 1.25 * CtaTile.M=64,
#                                       80 = 2.5 * CtaTile.K=32, exercises the predicate-tail path)
#   200^3  tile-misaligned            -> gemm_tail / gemm_compare (cooperative)
#   1440^3 far larger w/ tail         -> gemm_tail / gemm_compare (cooperative)
#
# Sequence: build all five binaries -> run the three shapes -> direct cuBLAS
# diff at all three -> cuBLAS-only sweep -> ncu CSV at the 1440^3 anchor ->
# compute-sanitizer.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

CUTLASS_ROOT="${CUTLASS_ROOT:-${CUTLASS_DIR}}"
DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cutlass-cute-gemm-tail/${DATE_TAG}"
mkdir -p "${HOST_DIR}" profiles

NVCC_FLAGS=(
    -std=c++17 -O3 -arch=sm_90a -lineinfo
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
    --expt-relaxed-constexpr --expt-extended-lambda
    -I"${CUTLASS_ROOT}/include"
    -I"${CUTLASS_ROOT}/tools/util/include"
    -I"$(pwd)"
)

echo "=== build cooperative-tile binaries (gemm_tail, gemm_compare) ==="
nvcc "${NVCC_FLAGS[@]}" gemm_tail.cu    -o gemm_tail
nvcc "${NVCC_FLAGS[@]}" gemm_compare.cu -o gemm_compare -lcublas

echo "=== build small-tile binaries (gemm_tail_small, gemm_compare_small) ==="
nvcc "${NVCC_FLAGS[@]}" gemm_tail_small.cu    -o gemm_tail_small
nvcc "${NVCC_FLAGS[@]}" gemm_compare_small.cu -o gemm_compare_small -lcublas

echo "=== build cuBLAS reference harness ==="
nvcc -std=c++17 -O3 -arch=sm_90a cublas_reference.cu -o cublas_reference -lcublas

echo "=== AC-5 direct cuBLAS diffs at all three shapes ==="
echo "--- 80x80x80 (small-tile fallback; genuinely non-aligned) ---" | tee -a "${HOST_DIR}/compare.log"
./gemm_compare_small --m=80 --n=80 --k=80 --iterations=20 | tee -a "${HOST_DIR}/compare.log"
echo "--- 200x200x200 (cooperative tile) ---"  | tee -a "${HOST_DIR}/compare.log"
./gemm_compare       --m=200 --n=200 --k=200 --iterations=20 | tee -a "${HOST_DIR}/compare.log"
echo "--- 1440x1440x1440 (cooperative tile) ---" | tee -a "${HOST_DIR}/compare.log"
./gemm_compare       --m=1440 --n=1440 --k=1440 --iterations=20 | tee -a "${HOST_DIR}/compare.log"

echo "=== cuBLAS-only sweep at AC-5 shapes ==="
./cublas_reference --ac5 --iters 20 | tee -a "${HOST_DIR}/cublas.log"

echo "=== ncu counter capture at 1440^3 (boundary-tile predicate path) ==="
if command -v ncu >/dev/null 2>&1; then
    ncu --target-processes all --csv \
        --metrics sm__cycles_active.avg,sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,\
smsp__inst_executed_pipe_tensor_op_hmma_cycles_active.sum \
        ./gemm_tail --m=1440 --n=1440 --k=1440 --iterations=2 \
        2>&1 | tee "${HOST_DIR}/ncu-tail-1440.csv" | tail -20
fi

echo "=== compute-sanitizer memcheck (gemm_compare_small at 80^3) ==="
if command -v compute-sanitizer >/dev/null 2>&1; then
    compute-sanitizer --tool memcheck \
        ./gemm_compare_small --m=80 --n=80 --k=80 --iterations=2 \
        2>&1 | tee "${HOST_DIR}/compute-sanitizer-tail.log" | tail -10
fi

echo ""
echo "Repo-committed artifacts : gemm_tail.cu, gemm_tail_small.cu, gemm_compare.cu, gemm_compare_small.cu, build*.sh, run_tail.sh, profiles/*.csv"
echo "Host-retained artifacts  : ${HOST_DIR}/  (compare.log, cublas.log, compute-sanitizer-tail.log)"
