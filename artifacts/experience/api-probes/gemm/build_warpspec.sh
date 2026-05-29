#!/usr/bin/env bash
CUDA_REPO_ROOT=${CUDA_REPO_ROOT:-/path/to/your/cuda_repo}
CUTLASS_DIR=${CUTLASS_DIR:-${CUDA_REPO_ROOT}/cutlass}
# AC-6 (warp-specialization ablation) build. Compiles all four schedule
# variants used by the AC-6 ablation:
#   gemm_compare_tma.cu      -> KernelTma                       (warp-spec OFF baseline)
#   gemm_compare_ws.cu       -> KernelTmaWarpSpecialized        (warp-spec ON, persistent OFF)
#   gemm_compare_pingpong.cu -> KernelTmaWarpSpecializedPingpong (warp-spec ON, persistent ON, pingpong)
#   gemm_compare.cu          -> KernelTmaWarpSpecializedCooperative (warp-spec ON, persistent ON, cooperative)
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

CUTLASS_ROOT="${CUTLASS_ROOT:-${CUTLASS_DIR}}"

NVCC_FLAGS=(
    -std=c++17 -O3 -arch=sm_90a -lineinfo
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
    --expt-relaxed-constexpr --expt-extended-lambda
    -I"${CUTLASS_ROOT}/include"
    -I"${CUTLASS_ROOT}/tools/util/include"
    -I"$(pwd)"
)

nvcc "${NVCC_FLAGS[@]}" gemm_compare_tma.cu      -o gemm_compare_tma      -lcublas
nvcc "${NVCC_FLAGS[@]}" gemm_compare_ws.cu       -o gemm_compare_ws       -lcublas
nvcc "${NVCC_FLAGS[@]}" gemm_compare_pingpong.cu -o gemm_compare_pingpong -lcublas
nvcc "${NVCC_FLAGS[@]}" gemm_compare.cu          -o gemm_compare          -lcublas
