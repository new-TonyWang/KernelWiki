#!/usr/bin/env bash
CUDA_REPO_ROOT=${CUDA_REPO_ROOT:-/path/to/your/cuda_repo}
CUTLASS_DIR=${CUTLASS_DIR:-${CUDA_REPO_ROOT}/cutlass}
# AC-5 (non-aligned GEMM tail handling) build. Compiles gemm_tail.cu — the same
# Hopper warp-specialized cooperative GEMM as the AC-4 baseline but driven at
# non-aligned shapes so the kernel's tail-handling code path is exercised.
#
# Bootstrap from a fresh sandbox:
#   rsync -az --exclude='.git' --exclude='build' \
#         ${CUTLASS_DIR}/ \
#         h200_ncu:${CUTLASS_DIR}/
# Run on H200; assumes /usr/local/cuda-12.9/bin/nvcc.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

CUTLASS_ROOT="${CUTLASS_ROOT:-${CUTLASS_DIR}}"

nvcc -std=c++17 -O3 -arch=sm_90a -lineinfo \
     -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
     --expt-relaxed-constexpr --expt-extended-lambda \
     -I"${CUTLASS_ROOT}/include" \
     -I"${CUTLASS_ROOT}/tools/util/include" \
     -I"$(pwd)" \
     gemm_tail.cu \
     -o gemm_tail
