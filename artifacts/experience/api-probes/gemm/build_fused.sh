#!/usr/bin/env bash
CUDA_REPO_ROOT=${CUDA_REPO_ROOT:-/path/to/your/cuda_repo}
CUTLASS_DIR=${CUTLASS_DIR:-${CUDA_REPO_ROOT}/cutlass}
# AC-7 (fused-GEMM epilogue) build. Compiles three binaries:
#   gemm_compare       (existing) -> non-fused GEMM, cooperative auto-selected.
#   gemm_compare_relu  (this round) -> fused GEMM + ReLU via LinCombEltAct<ReLu>.
#   relu_kernel        (this round) -> standalone non-fused ReLU baseline.
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

nvcc "${NVCC_FLAGS[@]}" gemm_compare.cu      -o gemm_compare      -lcublas
nvcc "${NVCC_FLAGS[@]}" gemm_compare_relu.cu -o gemm_compare_relu -lcublas
nvcc -std=c++17 -O3 -arch=sm_90a -lineinfo relu_kernel.cu -o relu_kernel
